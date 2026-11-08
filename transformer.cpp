#include "transformer.h"
#include "attention_kernels.h"
#include <stdexcept>
#include "io_manager.h"
#include "kernel_dispatcher.h"
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>

// --- CUDA Kernel Declarations ---
__global__ void rmsNormKernel(half* d_input, const float* d_weight, half* d_output, int size, float epsilon);
__global__ void ropeKernel(half* d_query, half* d_key, int head_dim, int seq_len, int pos_offset, int num_q_heads, int num_kv_heads);
__global__ void geluKernel(half* d_activations, int size);
__global__ void elementWiseMulKernel(half* d_a, const half* d_b, int size);
__global__ void addResidualKernel(half* d_target, const half* d_addend, int size);
__global__ void addBiasF32Kernel(half* d_target, const float* d_bias, int size);

// Diagnostic helper
[[maybe_unused]] static void checkBufferNaNs(const half* d_buf, int size, const char* name, cudaStream_t stream) {
    std::vector<half> h_buf(size);
    cudaMemcpyAsync(h_buf.data(), d_buf, size * sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int nan_cnt = 0, inf_cnt = 0;
    float min_val = 1e30f, max_val = -1e30f;
    for (int i = 0; i < size; ++i) {
        float v = __half2float(h_buf[i]);
        if (std::isnan(v)) nan_cnt++;
        else if (std::isinf(v)) inf_cnt++;
        else {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
    }
    if (nan_cnt > 0 || inf_cnt > 0) {
        printf("  [ALERT] %s has %d NaNs, %d Infs! (valid min=%.4f, max=%.4f)\n", name, nan_cnt, inf_cnt, min_val, max_val);
    }
}

__global__ void dequantize_q4_0_row_kernel(const uint8_t* q4_data, half* fp16_out, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;
    int block_idx = idx / 32;
    int in_block_idx = idx % 32;
    const uint8_t* block_ptr = q4_data + block_idx * 18;
    uint16_t raw_d = (uint16_t)block_ptr[0] | ((uint16_t)block_ptr[1] << 8);
    half d = *reinterpret_cast<half*>(&raw_d);
    const uint8_t* qs = block_ptr + 2;
    int8_t v;
    if (in_block_idx < 16) {
        v = (qs[in_block_idx] & 0x0F) - 8;
    } else {
        v = (qs[in_block_idx - 16] >> 4) - 8;
    }
    float scale = __half2float(d);
    float val = scale * (float)v;
    if (isnan(val)) val = 0.0f;
    else if (val > 65504.0f) val = 65504.0f;
    else if (val < -65504.0f) val = -65504.0f;
    fp16_out[idx] = __float2half(val);
}

// GGML Q2_K row dequant for token embeddings: one thread per element,
// 84-byte blocks of 256 values (16 scale bytes, 64 quant bytes, fp16 d, fp16 dmin).
__global__ void dequantize_q2_K_row_kernel(const uint8_t* row, half* out, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    const uint8_t* b = row + (idx / 256) * 84;
    const uint8_t* scales = b;
    const uint8_t* qs = b + 16;
    const half d = *reinterpret_cast<const half*>(b + 80);
    const half dmin = *reinterpret_cast<const half*>(b + 82);
    int r = idx % 256, h = r / 128, j = (r % 128) / 32, l = r % 32;
    uint8_t sc = scales[h * 8 + j * 2 + (l >= 16 ? 1 : 0)];
    int q = (qs[h * 32 + l] >> (2 * j)) & 3;
    float v = __half2float(d) * (float)(sc & 0x0F) * (float)q
            - __half2float(dmin) * (float)(sc >> 4);
    out[idx] = __float2half(v);
}

QuantType Transformer::getQuantTypeFromDtype(const std::string& dtype_str) {
    if (dtype_str == "F16" || dtype_str == "F32") return FP16;
    if (dtype_str == "Q4_0") return INT4;
    if (dtype_str == "Q4_1") return Q4_1;
    if (dtype_str == "Q8_0" || dtype_str == "Q8_1") return INT8;
    if (dtype_str == "Q6_K") return Q6_K;
    if (dtype_str == "TERNARY") return TERNARY;
    if (dtype_str == "Q2_K") return Q2_K;
    if (dtype_str == "Q3_K") return Q3_K;
    if (dtype_str == "Q4_K") return Q4_K;
    return UNKNOWN;
}

// Constructor: V32 Esnek Hibrit Bellek (Elastic Tiered Memory)
Transformer::Transformer(
    const std::string& model_file_path,
    const std::string& tensor_map_path,
    int model_hidden_dim,
    int model_num_heads,
    int model_num_kv_heads,
    int model_num_layers,
    float model_rms_norm_epsilon,
    QuantType model_quant_type,
    AsyncIOManager* async_io_manager_ptr,
    double vram_limit_gb,
    int max_context
)
    : model_file_path_(model_file_path),
      hidden_dim_(model_hidden_dim),
      num_heads_(model_num_heads),
      num_kv_heads_(model_num_kv_heads),
      head_dim_(model_hidden_dim / model_num_heads),
      num_layers_(model_num_layers),
      rms_norm_epsilon_(model_rms_norm_epsilon),
      model_quant_type_(model_quant_type),
      async_io_manager_ptr_(async_io_manager_ptr),
      vram_limit_gb_(vram_limit_gb),
      max_ctx_(max_context)
{
    if (max_ctx_ <= 0) throw std::runtime_error("max_context must be positive");
    if (hidden_dim_ % num_heads_ != 0) {
        fprintf(stderr, "Error: hidden_dim (%d) must be divisible by num_heads (%d).\n", hidden_dim_, num_heads_);
        exit(EXIT_FAILURE);
    }
    printf("Transformer initialized with GGUF parameters: hidden_dim=%d, num_heads=%d, num_kv_heads=%d, num_layers=%d, epsilon=%f, QuantType=%d\n",
           hidden_dim_, num_heads_, num_kv_heads_, num_layers_, rms_norm_epsilon_, model_quant_type_);

    loadTensorMap(tensor_map_path);

    // Determine max intermediate size for MLP
    for (const auto& item : full_tensor_map_) {
        if (item.first.find("ffn_gate.weight") != std::string::npos) {
            if (item.second.shape.size() > 1 && item.second.shape[1] > max_intermediate_size_) {
                max_intermediate_size_ = item.second.shape[1];
            }
        }
    }
    if (max_intermediate_size_ == 0) max_intermediate_size_ = hidden_dim_ * 4;

    // Allocate and zero temporary activation buffers
    CUDA_CHECK(cudaMalloc((void**)&d_query_out_, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_key_out_, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_value_out_, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_attention_output_, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_mlp_output_, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_norm_output_, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_gate_out_, max_intermediate_size_ * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_up_out_, max_intermediate_size_ * sizeof(half)));

    CUDA_CHECK(cudaMemset(d_query_out_, 0, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_key_out_, 0, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_value_out_, 0, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_attention_output_, 0, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_mlp_output_, 0, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_norm_output_, 0, hidden_dim_ * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_gate_out_, 0, max_intermediate_size_ * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_up_out_, 0, max_intermediate_size_ * sizeof(half)));

    // KV cache on the GPU. Allocated before the residency planner measures free
    // VRAM, so it is part of the deterministic budget.
    {
        const size_t kv_elems = (size_t)num_layers_ * max_ctx_ * num_kv_heads_ * head_dim_;
        CUDA_CHECK(cudaMalloc((void**)&d_k_cache_, kv_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_v_cache_, kv_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_attn_ctx_, hidden_dim_ * sizeof(half)));
        printf("[Transformer] GPU KV cache: max context %d, %.0f MiB (K + V)\n", max_ctx_,
               (double)(2 * kv_elems * sizeof(half)) / (1024.0 * 1024.0));
    }

    // =========================================================================
    // Katman Disi Statik Tensorleri Yukle (Token Embedding, Output Norm, LM Head)
    // =========================================================================
    std::string emb_name = "token_embd.weight";
    if (!full_tensor_map_.count(emb_name)) emb_name = "model.embed_tokens.weight";
    if (full_tensor_map_.count(emb_name)) {
        const auto& info = full_tensor_map_.at(emb_name);
        CUDA_CHECK(cudaMalloc((void**)&d_token_embd_, info.size_bytes));
        std::vector<unsigned char> h_emb(info.size_bytes);
        if (async_io_manager_ptr_->readLayerToHost(info.absolute_offset, info.size_bytes, h_emb.data())) {
            CUDA_CHECK(cudaMemcpy(d_token_embd_, h_emb.data(), info.size_bytes, cudaMemcpyHostToDevice));
            printf("[Transformer] Statik tensor '%s' GPU'ya yuklendi (%.2f MB)\n",
                   emb_name.c_str(), (float)info.size_bytes / (1024.0f * 1024.0f));
        } else {
            fprintf(stderr, "Hata: '%s' tensörü okunamadi!\n", emb_name.c_str());
            exit(EXIT_FAILURE);
        }
    }

    std::string norm_name = "output_norm.weight";
    if (!full_tensor_map_.count(norm_name)) norm_name = "model.norm.weight";
    if (full_tensor_map_.count(norm_name)) {
        const auto& info = full_tensor_map_.at(norm_name);
        CUDA_CHECK(cudaMalloc((void**)&d_output_norm_, info.size_bytes));
        std::vector<unsigned char> h_norm(info.size_bytes);
        if (async_io_manager_ptr_->readLayerToHost(info.absolute_offset, info.size_bytes, h_norm.data())) {
            CUDA_CHECK(cudaMemcpy(d_output_norm_, h_norm.data(), info.size_bytes, cudaMemcpyHostToDevice));
            printf("[Transformer] Statik tensor '%s' GPU'ya yuklendi (%.2f KB)\n",
                   norm_name.c_str(), (float)info.size_bytes / 1024.0f);
        } else {
            fprintf(stderr, "Hata: '%s' tensörü okunamadi!\n", norm_name.c_str());
            exit(EXIT_FAILURE);
        }
    }

    std::string head_name = "output.weight";
    if (!full_tensor_map_.count(head_name)) head_name = "lm_head.weight";
    if (full_tensor_map_.count(head_name)) {
        const auto& info = full_tensor_map_.at(head_name);
        CUDA_CHECK(cudaMalloc((void**)&d_lm_head_, info.size_bytes));
        std::vector<unsigned char> h_head(info.size_bytes);
        if (async_io_manager_ptr_->readLayerToHost(info.absolute_offset, info.size_bytes, h_head.data())) {
            CUDA_CHECK(cudaMemcpy(d_lm_head_, h_head.data(), info.size_bytes, cudaMemcpyHostToDevice));
            printf("[Transformer] Statik tensor '%s' GPU'ya yuklendi (%.2f MB)\n",
                   head_name.c_str(), (float)info.size_bytes / (1024.0f * 1024.0f));
        } else {
            fprintf(stderr, "Hata: '%s' tensörü okunamadi!\n", head_name.c_str());
            exit(EXIT_FAILURE);
        }
    }

    // =========================================================================
    // Dynamic VRAM residency planner
    // Budget = free VRAM after static allocations (optionally capped by --vram),
    // minus a fixed safety reserve and the DMA slot reservation. Layers are
    // placed on the GPU in order until the budget is exhausted; the count is
    // fixed after construction (no cudaMalloc/cudaFree during decoding).
    // =========================================================================
    size_t free_bytes_before_residency = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes_before_residency, &total_bytes));

    const size_t kMinGpuReserveBytes = 384ULL * 1024ULL * 1024ULL;
    const size_t dma_reservation = (size_t)kDmaSlotCount * layer_buffer_size_;
    const size_t required_overhead = kMinGpuReserveBytes + dma_reservation;

    size_t budget = free_bytes_before_residency;
    if (vram_limit_gb_ > 0.1) {
        const size_t requested_total = (size_t)(vram_limit_gb_ * 1024.0 * 1024.0 * 1024.0);
        const size_t already_used = total_bytes - free_bytes_before_residency;
        const size_t capped = (requested_total > already_used) ? (requested_total - already_used) : 0;
        if (capped < budget) budget = capped;
    }

    if (budget < required_overhead) {
        throw std::runtime_error(
            "VRAM budget too small: " + std::to_string(budget / (1024 * 1024)) +
            " MiB available, " + std::to_string(required_overhead / (1024 * 1024)) +
            " MiB required for DMA slots and safety reserve");
    }

    size_t available_for_resident = budget - required_overhead;
    num_gpu_resident_layers_ = 0;
    size_t resident_weight_bytes = 0;
    for (int i = 0; i < num_layers_; ++i) {
        const size_t l_size = layers_info_[i].size_bytes;
        if (available_for_resident < l_size) break;
        available_for_resident -= l_size;
        resident_weight_bytes += l_size;
        num_gpu_resident_layers_++;
    }
    const int num_streamed = num_layers_ - num_gpu_resident_layers_;

    printf("\n=======================================================\n");
    printf("  VRAM total: %.2f GB | free before layers: %.2f GB",
           (double)total_bytes / (1u << 30), (double)free_bytes_before_residency / (1u << 30));
    if (vram_limit_gb_ > 0.1) printf(" | --vram cap: %.2f GB", vram_limit_gb_);
    printf("\n  Reserved: %zu MiB safety + %zu MiB DMA (%d slots)\n",
           kMinGpuReserveBytes >> 20, dma_reservation >> 20, kDmaSlotCount);
    printf("  GPU-resident layers: %d / %d (%.2f GB) | host-streamed: %d\n",
           num_gpu_resident_layers_, num_layers_, (double)resident_weight_bytes / (1u << 30), num_streamed);
    printf("=======================================================\n\n");

    // Streams are created before any event is recorded on them.
    CUDA_CHECK(cudaStreamCreate(&stream_compute_));
    CUDA_CHECK(cudaStreamCreate(&stream_io_));

    // 1. GPU'da kalici tutulacak katmanlari yukle
    d_gpu_resident_layers_.resize(num_gpu_resident_layers_, nullptr);
    std::vector<unsigned char> temp_load_buf(layer_buffer_size_);
    for (int l = 0; l < num_gpu_resident_layers_; ++l) {
        const auto& l_info = layers_info_[l];
        CUDA_CHECK(cudaMalloc((void**)&d_gpu_resident_layers_[l], l_info.size_bytes));
        async_io_manager_ptr_->readLayerToHost(l_info.start_offset, l_info.size_bytes, temp_load_buf.data());
        CUDA_CHECK(cudaMemcpy(d_gpu_resident_layers_[l], temp_load_buf.data(), l_info.size_bytes, cudaMemcpyHostToDevice));
    }
    if (num_gpu_resident_layers_ > 0) {
        printf("[Transformer] %d adet katman GPU VRAM'e kalici yuklendi (Sifir transfer gecikmesi).\n", num_gpu_resident_layers_);
    }

    // 2. Stream edilecek katmanlar varsa, DMA slotlarini ve Host Onbellegi hazirla
    if (num_streamed > 0) {
        for (int i = 0; i < kDmaSlotCount; ++i) {
            CUDA_CHECK(cudaMalloc((void**)&d_layer_buffer_[i], layer_buffer_size_));
            CUDA_CHECK(cudaMemset(d_layer_buffer_[i], 0, layer_buffer_size_));
            CUDA_CHECK(cudaMallocHost((void**)&h_pinned_buffer_[i], layer_buffer_size_));
            // ev_slot_free_[i]: compute finished reading slot i (safe to overwrite).
            // ev_slot_ready_[i]: copy into slot i finished (safe to compute on).
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_slot_free_[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_slot_ready_[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(ev_slot_free_[i], stream_compute_));
            CUDA_CHECK(cudaEventRecord(ev_slot_ready_[i], stream_io_));
        }
        slot_events_created_ = true;

        // Offloaded katmanlari sistem RAM'inde (Pinned Host) onbellekle
        h_offload_pinned_cache_.resize(num_streamed, nullptr);
        bool all_pinned_ok = true;
        for (int i = 0; i < num_streamed; ++i) {
            int l_idx = num_gpu_resident_layers_ + i;
            const auto& l_info = layers_info_[l_idx];
            cudaError_t err = cudaMallocHost((void**)&h_offload_pinned_cache_[i], l_info.size_bytes);
            if (err == cudaSuccess) {
                async_io_manager_ptr_->readLayerToHost(l_info.start_offset, l_info.size_bytes, h_offload_pinned_cache_[i]);
            } else {
                all_pinned_ok = false;
                for (int j = 0; j < i; ++j) {
                    if (h_offload_pinned_cache_[j]) cudaFreeHost(h_offload_pinned_cache_[j]);
                    h_offload_pinned_cache_[j] = nullptr;
                }
                h_offload_pinned_cache_.clear();
                break;
            }
        }
        if (all_pinned_ok && !h_offload_pinned_cache_.empty()) {
            printf("[Transformer] %d adet dis katman Sistem RAM'ine (Pinned Memory) onbelleklendi (Cikarim sirasinda 0 Disk I/O!).\n", num_streamed);
        } else {
            printf("[Transformer] Uyari: Host RAM siniri nedeniyle %d dis katman kalici dosya handle'i ile stream edilecek.\n", num_streamed);
        }
    }
}

// Destructor: GPU ve Host belleklerini temizler
Transformer::~Transformer() {
    printf("[Transformer] Destructor cagrildi: GPU ve Host bellekleri serbest birakiliyor...\n");
    for (auto* ptr : d_gpu_resident_layers_) {
        if (ptr) CUDA_CHECK(cudaFree(ptr));
    }
    d_gpu_resident_layers_.clear();

    for (auto* ptr : h_offload_pinned_cache_) {
        if (ptr) CUDA_CHECK(cudaFreeHost(ptr));
    }
    h_offload_pinned_cache_.clear();

    for (int i = 0; i < kDmaSlotCount; ++i) {
        if (slot_events_created_) {
            CUDA_CHECK(cudaEventDestroy(ev_slot_free_[i]));
            CUDA_CHECK(cudaEventDestroy(ev_slot_ready_[i]));
        }
        if (d_layer_buffer_[i]) CUDA_CHECK(cudaFree(d_layer_buffer_[i]));
        if (h_pinned_buffer_[i]) CUDA_CHECK(cudaFreeHost(h_pinned_buffer_[i]));
    }

    if (d_token_embd_) CUDA_CHECK(cudaFree(d_token_embd_));
    if (d_output_norm_) CUDA_CHECK(cudaFree(d_output_norm_));
    if (d_lm_head_) CUDA_CHECK(cudaFree(d_lm_head_));
    if (d_k_cache_) CUDA_CHECK(cudaFree(d_k_cache_));
    if (d_v_cache_) CUDA_CHECK(cudaFree(d_v_cache_));
    if (d_attn_ctx_) CUDA_CHECK(cudaFree(d_attn_ctx_));
    
    if (stream_compute_) CUDA_CHECK(cudaStreamDestroy(stream_compute_));
    if (stream_io_) CUDA_CHECK(cudaStreamDestroy(stream_io_));

    if (d_query_out_) CUDA_CHECK(cudaFree(d_query_out_));
    if (d_key_out_) CUDA_CHECK(cudaFree(d_key_out_));
    if (d_value_out_) CUDA_CHECK(cudaFree(d_value_out_));
    if (d_attention_output_) CUDA_CHECK(cudaFree(d_attention_output_));
    if (d_mlp_output_) CUDA_CHECK(cudaFree(d_mlp_output_));
    if (d_norm_output_) CUDA_CHECK(cudaFree(d_norm_output_));
    if (d_gate_out_) CUDA_CHECK(cudaFree(d_gate_out_));
    if (d_up_out_) CUDA_CHECK(cudaFree(d_up_out_));
}

void Transformer::loadTensorMap(const std::string& map_path) {
    printf("[Transformer] '%s' dosyasindan tensor haritasi yukleniyor...\n", map_path.c_str());
    std::ifstream f(map_path);
    if (!f.is_open()) {
        fprintf(stderr, "Hata: Tensor haritasi dosyasi acilamadi: %s\n", map_path.c_str());
        exit(EXIT_FAILURE);
    }

    try {
        nlohmann::json j = nlohmann::json::parse(f);
        for (const auto& item : j) {
            TensorInfo info;
            info.name = item.at("name").get<std::string>();
            info.absolute_offset = item.at("absolute_offset").get<size_t>();
            info.size_bytes = item.at("size_bytes").get<size_t>();
            info.shape = item.at("shape").get<std::vector<long long>>();
            info.dtype = item.at("dtype").get<std::string>();
            full_tensor_map_[info.name] = info;
        }
        printf("[Transformer] %zu adet tensor bilgisi yuklendi.\n", full_tensor_map_.size());
    } catch (const nlohmann::json::exception& e) {
        fprintf(stderr, "Hata: Tensor haritasi JSON ayristirma hatasi: %s\n", e.what());
        exit(EXIT_FAILURE);
    }
    f.close();

    layers_info_.resize(num_layers_);
    std::vector<std::vector<TensorInfo>> layer_tensors(num_layers_);

    for (const auto& [name, info] : full_tensor_map_) {
        if (name.rfind("blk.", 0) == 0) {
            size_t dot1 = name.find('.');
            size_t dot2 = name.find('.', dot1 + 1);
            if (dot1 != std::string::npos && dot2 != std::string::npos) {
                int l_idx = std::stoi(name.substr(dot1 + 1, dot2 - dot1 - 1));
                if (l_idx >= 0 && l_idx < num_layers_) {
                    layer_tensors[l_idx].push_back(info);
                }
            }
        }
    }

    size_t max_layer_size = 0;
    for (int l = 0; l < num_layers_; ++l) {
        if (layer_tensors[l].empty()) {
            fprintf(stderr, "Hata: Katman %d icin tensor bulunamadi!\n", l);
            exit(EXIT_FAILURE);
        }
        size_t min_offset = std::numeric_limits<size_t>::max();
        size_t max_end = 0;
        for (const auto& t : layer_tensors[l]) {
            if (t.absolute_offset < min_offset) min_offset = t.absolute_offset;
            size_t end_off = t.absolute_offset + t.size_bytes;
            if (end_off > max_end) max_end = end_off;
        }
        layers_info_[l].start_offset = min_offset;
        layers_info_[l].size_bytes = max_end - min_offset;
        for (const auto& t : layer_tensors[l]) {
            layers_info_[l].tensor_rel_offsets[t.name] = t.absolute_offset - min_offset;
        }
        if (layers_info_[l].size_bytes > max_layer_size) {
            max_layer_size = layers_info_[l].size_bytes;
        }
    }

    printf("[Transformer] Katman haritasi olusturuldu: %d katman, Max katman boyutu: %zu bayt (%.2f MB)\n",
           num_layers_, max_layer_size, (float)max_layer_size / (1024.0f * 1024.0f));

    size_t aligned_max = ((max_layer_size + 1048575) / 1048576) * 1048576;
    layer_buffer_size_ = std::max((size_t)(32 * 1024 * 1024), aligned_max);
}

void Transformer::applyRMSNorm(
    half* d_input,
    const float* d_weight,
    half* d_output,
    int size,
    cudaStream_t stream)
{
    int threads = 256;
    int shared_mem_size = threads * sizeof(float);

    rmsNormKernel<<<1, threads, shared_mem_size, stream>>>(
        d_input,
        d_weight,
        d_output,
        size,
        rms_norm_epsilon_
    );
    CUDA_CHECK(cudaGetLastError());
}

void Transformer::applyRoPE(
    half* d_query,
    half* d_key,
    int head_dim,
    int seq_len,
    int pos_offset,
    cudaStream_t stream)
{
    dim3 blockDim(256);
    dim3 gridDim(((head_dim / 2) + blockDim.x - 1) / blockDim.x);

    ropeKernel<<<gridDim, blockDim, 0, stream>>>(
        d_query,
        d_key,
        head_dim,
        seq_len,
        pos_offset,
        num_heads_,
        num_kv_heads_
    );
    CUDA_CHECK(cudaGetLastError());
}

void Transformer::applySelfAttention(
    int layer_idx,
    unsigned char* d_layer_buf,
    half* d_input_activations,
    half* d_query_out,
    half* d_key_out,
    half* d_value_out,
    half* d_attention_output,
    int seq_len,
    int pos_offset,
    cudaStream_t stream)
{
    const auto& l_info = layers_info_[layer_idx];
    std::string q_weight_name = "blk." + std::to_string(layer_idx) + ".attn_q.weight";
    std::string k_weight_name = "blk." + std::to_string(layer_idx) + ".attn_k.weight";
    std::string v_weight_name = "blk." + std::to_string(layer_idx) + ".attn_v.weight";
    std::string o_weight_name = "blk." + std::to_string(layer_idx) + ".attn_output.weight";

    unsigned char* d_Wq_ptr = d_layer_buf + l_info.tensor_rel_offsets.at(q_weight_name);
    unsigned char* d_Wk_ptr = d_layer_buf + l_info.tensor_rel_offsets.at(k_weight_name);
    unsigned char* d_Wv_ptr = d_layer_buf + l_info.tensor_rel_offsets.at(v_weight_name);
    unsigned char* d_Wo_ptr = d_layer_buf + l_info.tensor_rel_offsets.at(o_weight_name);

    const TensorInfo& q_info = full_tensor_map_.at(q_weight_name);
    const TensorInfo& k_info = full_tensor_map_.at(k_weight_name);
    const TensorInfo& v_info = full_tensor_map_.at(v_weight_name);
    const TensorInfo& o_info = full_tensor_map_.at(o_weight_name);

    // 1. Q Projeksiyonu + Bias
    FridayKernelDispatcher::dispatchGemvKernel(
        getQuantTypeFromDtype(q_info.dtype),
        d_input_activations,
        d_Wq_ptr,
        nullptr,
        nullptr,
        d_query_out,
        (int)q_info.shape[0],
        (int)q_info.shape[1],
        stream
    );
    std::string q_bias_name = "blk." + std::to_string(layer_idx) + ".attn_q.bias";
    if (l_info.tensor_rel_offsets.count(q_bias_name)) {
        const float* d_q_bias = (const float*)(d_layer_buf + l_info.tensor_rel_offsets.at(q_bias_name));
        dim3 block(256);
        dim3 grid(((int)q_info.shape[1] + 255) / 256);
        addBiasF32Kernel<<<grid, block, 0, stream>>>(d_query_out, d_q_bias, (int)q_info.shape[1]);
    }
    CUDA_CHECK(cudaGetLastError());

    // 2. K Projeksiyonu + Bias
    FridayKernelDispatcher::dispatchGemvKernel(
        getQuantTypeFromDtype(k_info.dtype),
        d_input_activations,
        d_Wk_ptr,
        nullptr,
        nullptr,
        d_key_out,
        (int)k_info.shape[0],
        (int)k_info.shape[1],
        stream
    );
    std::string k_bias_name = "blk." + std::to_string(layer_idx) + ".attn_k.bias";
    if (l_info.tensor_rel_offsets.count(k_bias_name)) {
        const float* d_k_bias = (const float*)(d_layer_buf + l_info.tensor_rel_offsets.at(k_bias_name));
        dim3 block(256);
        dim3 grid(((int)k_info.shape[1] + 255) / 256);
        addBiasF32Kernel<<<grid, block, 0, stream>>>(d_key_out, d_k_bias, (int)k_info.shape[1]);
    }
    CUDA_CHECK(cudaGetLastError());

    // 3. V Projeksiyonu + Bias
    FridayKernelDispatcher::dispatchGemvKernel(
        getQuantTypeFromDtype(v_info.dtype),
        d_input_activations,
        d_Wv_ptr,
        nullptr,
        nullptr,
        d_value_out,
        (int)v_info.shape[0],
        (int)v_info.shape[1],
        stream
    );
    std::string v_bias_name = "blk." + std::to_string(layer_idx) + ".attn_v.bias";
    if (l_info.tensor_rel_offsets.count(v_bias_name)) {
        const float* d_v_bias = (const float*)(d_layer_buf + l_info.tensor_rel_offsets.at(v_bias_name));
        dim3 block(256);
        dim3 grid(((int)v_info.shape[1] + 255) / 256);
        addBiasF32Kernel<<<grid, block, 0, stream>>>(d_value_out, d_v_bias, (int)v_info.shape[1]);
    }
    CUDA_CHECK(cudaGetLastError());

    // 4. RoPE
    applyRoPE(d_query_out, d_key_out, head_dim_, seq_len, pos_offset, stream);

    // Attention on the GPU: append K/V for this position to the layer's KV cache,
    // then single-query GQA attention over positions 0..pos_offset. No host copies
    // or stream synchronization.
    if (seq_len != 1) {
        throw std::runtime_error("applySelfAttention: only single-token decode is supported");
    }
    if (pos_offset < 0 || pos_offset >= max_ctx_) {
        throw std::runtime_error("context length exceeded: position " + std::to_string(pos_offset) +
                                 " >= max context " + std::to_string(max_ctx_));
    }
    const int kv_dim = num_kv_heads_ * head_dim_;
    half* k_layer = d_k_cache_ + (size_t)layer_idx * max_ctx_ * kv_dim;
    half* v_layer = d_v_cache_ + (size_t)layer_idx * max_ctx_ * kv_dim;
    CUDA_CHECK(cudaMemcpyAsync(k_layer + (size_t)pos_offset * kv_dim, d_key_out, kv_dim * sizeof(half),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_layer + (size_t)pos_offset * kv_dim, d_value_out, kv_dim * sizeof(half),
                               cudaMemcpyDeviceToDevice, stream));
    friday_attn::launchDecodeAttention(d_query_out, k_layer, v_layer, d_attn_ctx_, pos_offset + 1,
                                       num_heads_, num_kv_heads_, head_dim_, stream);
    CUDA_CHECK(cudaGetLastError());

    // Output Projeksiyonu
    FridayKernelDispatcher::dispatchGemvKernel(
        getQuantTypeFromDtype(o_info.dtype),
        d_attn_ctx_,
        d_Wo_ptr,
        nullptr,
        nullptr,
        d_attention_output,
        (int)o_info.shape[0],
        (int)o_info.shape[1],
        stream
    );
    CUDA_CHECK(cudaGetLastError());
}

void Transformer::forwardMLPLayer(
    int layer_idx,
    unsigned char* d_layer_buf,
    half* d_input_activations,
    half* d_output_activations,
    cudaStream_t stream
) {
    const auto& l_info = layers_info_[layer_idx];
    std::string gate_weight_name = "blk." + std::to_string(layer_idx) + ".ffn_gate.weight";
    std::string up_weight_name = "blk." + std::to_string(layer_idx) + ".ffn_up.weight";
    std::string down_weight_name = "blk." + std::to_string(layer_idx) + ".ffn_down.weight";

    unsigned char* d_Wgate_ptr = d_layer_buf + l_info.tensor_rel_offsets.at(gate_weight_name);
    unsigned char* d_Wup_ptr = d_layer_buf + l_info.tensor_rel_offsets.at(up_weight_name);
    unsigned char* d_Wdown_ptr = d_layer_buf + l_info.tensor_rel_offsets.at(down_weight_name);

    const TensorInfo& gate_info = full_tensor_map_.at(gate_weight_name);
    const TensorInfo& up_info = full_tensor_map_.at(up_weight_name);
    const TensorInfo& down_info = full_tensor_map_.at(down_weight_name);

    // 1. Gate Projeksiyonu
    FridayKernelDispatcher::dispatchGemvKernel(
        getQuantTypeFromDtype(gate_info.dtype),
        d_input_activations,
        d_Wgate_ptr,
        nullptr,
        nullptr,
        d_gate_out_,
        (int)gate_info.shape[0],
        (int)gate_info.shape[1],
        stream
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. Up Projeksiyonu
    FridayKernelDispatcher::dispatchGemvKernel(
        getQuantTypeFromDtype(up_info.dtype),
        d_input_activations,
        d_Wup_ptr,
        nullptr,
        nullptr,
        d_up_out_,
        (int)up_info.shape[0],
        (int)up_info.shape[1],
        stream
    );
    CUDA_CHECK(cudaGetLastError());

    // SwiGLU activation: SiLU(gate) * up
    applyGeLU(d_gate_out_, (int)gate_info.shape[1], stream);

    {
        int act_size = (int)gate_info.shape[1];
        dim3 block(256);
        dim3 grid((act_size + 255) / 256);
        elementWiseMulKernel<<<grid, block, 0, stream>>>(d_gate_out_, d_up_out_, act_size);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. Down Projeksiyonu
    FridayKernelDispatcher::dispatchGemvKernel(
        getQuantTypeFromDtype(down_info.dtype),
        d_gate_out_,
        d_Wdown_ptr,
        nullptr,
        nullptr,
        d_output_activations,
        (int)down_info.shape[0],
        (int)down_info.shape[1],
        stream
    );
    CUDA_CHECK(cudaGetLastError());
}

void Transformer::applyGeLU(half* d_activations, int size, cudaStream_t stream) {
    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    geluKernel<<<gridDim, blockDim, 0, stream>>>(d_activations, size);
    CUDA_CHECK(cudaGetLastError());
}

bool Transformer::getEmbedding(int token_id, half* d_output, int vocab_size, cudaStream_t stream) {
    if (!d_token_embd_) return false;
    std::string embed_name = "token_embd.weight";
    if (!full_tensor_map_.count(embed_name)) {
        embed_name = "model.embed_tokens.weight";
        if (!full_tensor_map_.count(embed_name)) return false;
    }
    const auto& t_info = full_tensor_map_.at(embed_name);

    if (token_id < 0 || token_id >= vocab_size) token_id = 151643;

    if (t_info.dtype == "Q4_0") {
        int row_size_bytes = (hidden_dim_ / 32) * 18;
        size_t offset = (size_t)token_id * row_size_bytes;
        const uint8_t* d_row_ptr = (const uint8_t*)d_token_embd_ + offset;
        int threads = 256;
        int blocks = (hidden_dim_ + threads - 1) / threads;
        dequantize_q4_0_row_kernel<<<blocks, threads, 0, stream>>>(d_row_ptr, d_output, hidden_dim_);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (t_info.dtype == "Q2_K") {
        const size_t row_bytes = (size_t)(hidden_dim_ / 256) * 84;
        const uint8_t* d_row = (const uint8_t*)d_token_embd_ + (size_t)token_id * row_bytes;
        dequantize_q2_K_row_kernel<<<(hidden_dim_ + 255) / 256, 256, 0, stream>>>(d_row, d_output, hidden_dim_);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (t_info.dtype != "F16") {
        // Previously every other dtype was copied as raw FP16, which read out of bounds.
        fprintf(stderr, "getEmbedding: unsupported embedding dtype '%s'\n", t_info.dtype.c_str());
        return false;
    }

    size_t offset = (size_t)token_id * hidden_dim_ * sizeof(half);
    CUDA_CHECK(cudaMemcpyAsync(d_output, (const char*)d_token_embd_ + offset, hidden_dim_ * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    return true;
}

// Ana Forward Pass: V27 Ping-Pong Double Buffering mimarisi
void Transformer::forward(
    half* d_input_activations,
    half* d_output_logits,
    int seq_len,
    int layer_offset,
    int pos_offset,
    cudaStream_t stream)
{
    cudaStream_t compute_stream = (stream != nullptr) ? stream : stream_compute_;

    // Gecici aktivasyon tamponlarini sifirla
    CUDA_CHECK(cudaMemsetAsync(d_query_out_, 0, hidden_dim_ * sizeof(half), compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_key_out_, 0, hidden_dim_ * sizeof(half), compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_value_out_, 0, hidden_dim_ * sizeof(half), compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_attention_output_, 0, hidden_dim_ * sizeof(half), compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_mlp_output_, 0, hidden_dim_ * sizeof(half), compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_norm_output_, 0, hidden_dim_ * sizeof(half), compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_gate_out_, 0, max_intermediate_size_ * sizeof(half), compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_up_out_, 0, max_intermediate_size_ * sizeof(half), compute_stream));

    if (pos_offset == 0) {
#ifdef FRIDAY_DEBUG_NANS  // diagnostic: syncs the stream and copies the buffer to host
        checkBufferNaNs(d_input_activations, hidden_dim_, "L0 initial input_activations", compute_stream);
#endif
    }

    // =========================================================================
    // V32: Esnek Hibrit Bellek (Elastic Tiered Memory) Forward Pass
    // =========================================================================

    // 1. GPU VRAM'de yerlesik (resident) katmanlar (0 .. num_gpu_resident_layers_ - 1)
    for (int layer_idx = 0; layer_idx < num_gpu_resident_layers_; ++layer_idx) {
        unsigned char* d_curr_buf = d_gpu_resident_layers_[layer_idx];
        const auto& curr_l_info = layers_info_[layer_idx];

        // 1. Pre-attn RMSNorm
        std::string attn_norm_name = "blk." + std::to_string(layer_idx + layer_offset) + ".attn_norm.weight";
        const float* d_attn_norm_ptr = (const float*)(d_curr_buf + curr_l_info.tensor_rel_offsets.at(attn_norm_name));
        applyRMSNorm(d_input_activations, d_attn_norm_ptr, d_norm_output_, hidden_dim_, compute_stream);

        // 2. Self-Attention
        applySelfAttention(
            layer_idx + layer_offset,
            d_curr_buf,
            d_norm_output_,
            d_query_out_, d_key_out_, d_value_out_,
            d_attention_output_,
            seq_len,
            pos_offset,
            compute_stream
        );

        // Residual: x = x + Attention(Norm(x))
        {
            dim3 block(256);
            dim3 grid((hidden_dim_ + 255) / 256);
            addResidualKernel<<<grid, block, 0, compute_stream>>>(d_input_activations, d_attention_output_, hidden_dim_);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3. Pre-MLP RMSNorm
        std::string ffn_norm_name = "blk." + std::to_string(layer_idx + layer_offset) + ".ffn_norm.weight";
        const float* d_ffn_norm_ptr = (const float*)(d_curr_buf + curr_l_info.tensor_rel_offsets.at(ffn_norm_name));
        applyRMSNorm(d_input_activations, d_ffn_norm_ptr, d_norm_output_, hidden_dim_, compute_stream);

        // 4. MLP Layer
        forwardMLPLayer(
            layer_idx + layer_offset,
            d_curr_buf,
            d_norm_output_,
            d_mlp_output_,
            compute_stream
        );

        // Residual: x = x + MLP(Norm(x))
        {
            dim3 block(256);
            dim3 grid((hidden_dim_ + 255) / 256);
            addResidualKernel<<<grid, block, 0, compute_stream>>>(d_input_activations, d_mlp_output_, hidden_dim_);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // 2. Kalan stream edilecek katmanlar (num_gpu_resident_layers_ .. num_layers_ - 1)
    if (num_gpu_resident_layers_ < num_layers_) {
        const int first_stream_layer = num_gpu_resident_layers_;

        // Issue the host->device copy of streamed layer `offload_idx` into DMA slot `slot`
        // on stream_io_. Ordering is enforced with per-slot events only:
        //   copy waits for ev_slot_free_[slot]  (compute is done reading the slot)
        //   compute waits for ev_slot_ready_[slot] (copy into the slot is done)
        auto enqueue_layer_copy = [&](int offload_idx, int slot) {
            const auto& info = layers_info_[first_stream_layer + offload_idx];
            CUDA_CHECK(cudaStreamWaitEvent(stream_io_, ev_slot_free_[slot], 0));
            const unsigned char* src = nullptr;
            if (!h_offload_pinned_cache_.empty() && h_offload_pinned_cache_[offload_idx]) {
                src = h_offload_pinned_cache_[offload_idx];
            } else {
                // Fallback: read from disk into this slot's pinned staging buffer.
                // Wait until the previous copy out of that staging buffer has finished.
                CUDA_CHECK(cudaEventSynchronize(ev_slot_ready_[slot]));
                if (!async_io_manager_ptr_->readLayerToHost(info.start_offset, info.size_bytes, h_pinned_buffer_[slot])) {
                    throw std::runtime_error("Failed to read streamed layer " +
                                             std::to_string(first_stream_layer + offload_idx) + " from disk");
                }
                src = h_pinned_buffer_[slot];
            }
            CUDA_CHECK(cudaMemcpyAsync(d_layer_buffer_[slot], src, info.size_bytes,
                                       cudaMemcpyHostToDevice, stream_io_));
            CUDA_CHECK(cudaEventRecord(ev_slot_ready_[slot], stream_io_));
        };

        enqueue_layer_copy(0, 0);

        for (int layer_idx = first_stream_layer; layer_idx < num_layers_; ++layer_idx) {
            const int offload_idx = layer_idx - first_stream_layer;
            const int curr_slot = offload_idx % kDmaSlotCount;
            const int next_slot = (offload_idx + 1) % kDmaSlotCount;

            // Prefetch the next streamed layer while this one is computed.
            if (layer_idx + 1 < num_layers_) {
                enqueue_layer_copy(offload_idx + 1, next_slot);
            }

            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, ev_slot_ready_[curr_slot], 0));
            unsigned char* d_curr_buf = d_layer_buffer_[curr_slot];
            const auto& curr_l_info = layers_info_[layer_idx];

            // 1. RMSNorm
            std::string attn_norm_name = "blk." + std::to_string(layer_idx + layer_offset) + ".attn_norm.weight";
            const float* d_attn_norm_ptr = (const float*)(d_curr_buf + curr_l_info.tensor_rel_offsets.at(attn_norm_name));
            applyRMSNorm(d_input_activations, d_attn_norm_ptr, d_norm_output_, hidden_dim_, compute_stream);

            // 2. Self-Attention
            applySelfAttention(
                layer_idx + layer_offset,
                d_curr_buf,
                d_norm_output_,
                d_query_out_, d_key_out_, d_value_out_,
                d_attention_output_,
                seq_len,
                pos_offset,
                compute_stream
            );

            // Residual: x = x + Attention(Norm(x))
            {
                dim3 block(256);
                dim3 grid((hidden_dim_ + 255) / 256);
                addResidualKernel<<<grid, block, 0, compute_stream>>>(d_input_activations, d_attention_output_, hidden_dim_);
                CUDA_CHECK(cudaGetLastError());
            }

            // 3. RMSNorm
            std::string ffn_norm_name = "blk." + std::to_string(layer_idx + layer_offset) + ".ffn_norm.weight";
            const float* d_ffn_norm_ptr = (const float*)(d_curr_buf + curr_l_info.tensor_rel_offsets.at(ffn_norm_name));
            applyRMSNorm(d_input_activations, d_ffn_norm_ptr, d_norm_output_, hidden_dim_, compute_stream);

            // 4. MLP Layer
            forwardMLPLayer(
                layer_idx + layer_offset,
                d_curr_buf,
                d_norm_output_,
                d_mlp_output_,
                compute_stream
            );

            // Residual: x = x + MLP(Norm(x))
            {
                dim3 block(256);
                dim3 grid((hidden_dim_ + 255) / 256);
                addResidualKernel<<<grid, block, 0, compute_stream>>>(d_input_activations, d_mlp_output_, hidden_dim_);
                CUDA_CHECK(cudaGetLastError());
            }

            CUDA_CHECK(cudaEventRecord(ev_slot_free_[curr_slot], compute_stream));
        }
    }
    // Final RMSNorm (Post-transformer norm, weights are F32)
    if (d_output_norm_) {
        applyRMSNorm(
            d_input_activations,
            (const float*)d_output_norm_,
            d_input_activations,
            hidden_dim_,
            compute_stream
        );
#ifdef FRIDAY_DEBUG_NANS  // diagnostic: syncs the stream and copies the buffer to host
        checkBufferNaNs(d_input_activations, hidden_dim_, "post-final-norm activations", compute_stream);
#endif
    }

    // LM Head (Output Projeksiyonu)
    if (d_lm_head_) {
        std::string lm_head_weight_name = "output.weight";
        if (!full_tensor_map_.count(lm_head_weight_name)) lm_head_weight_name = "lm_head.weight";
        const TensorInfo& lm_head_info = full_tensor_map_.at(lm_head_weight_name);

        FridayKernelDispatcher::dispatchGemvKernel(
            getQuantTypeFromDtype(lm_head_info.dtype),
            d_input_activations,
            d_lm_head_,
            nullptr,
            nullptr,
            d_output_logits,
            (int)lm_head_info.shape[0],
            (int)lm_head_info.shape[1],
            compute_stream
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef FRIDAY_DEBUG_NANS  // diagnostic: syncs the stream and copies the buffer to host
        checkBufferNaNs(d_output_logits, (int)lm_head_info.shape[1], "LM Head output logits", compute_stream);
#endif
    }

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
}
