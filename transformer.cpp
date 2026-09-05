#include "transformer.h"
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
static void checkBufferNaNs(const half* d_buf, int size, const char* name, cudaStream_t stream) {
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

QuantType Transformer::getQuantTypeFromDtype(const std::string& dtype_str) {
    if (dtype_str == "F16" || dtype_str == "F32") return FP16;
    if (dtype_str == "Q4_0" || dtype_str == "Q4_K") return INT4;
    if (dtype_str == "Q4_1") return Q4_1;
    if (dtype_str == "Q8_0" || dtype_str == "Q8_1") return INT8;
    if (dtype_str == "Q6_K") return Q6_K;
    if (dtype_str == "TERNARY") return TERNARY;
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
    double vram_limit_gb
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
      vram_limit_gb_(vram_limit_gb)
{
    if (hidden_dim_ % num_heads_ != 0) {
        fprintf(stderr, "Error: hidden_dim (%d) must be divisible by num_heads (%d).\n", hidden_dim_, num_heads_);
        exit(EXIT_FAILURE);
    }
    printf("Transformer initialized with GGUF parameters: hidden_dim=%d, num_heads=%d, num_kv_heads=%d, num_layers=%d, epsilon=%f, QuantType=%d\n",
           hidden_dim_, num_heads_, num_kv_heads_, num_layers_, rms_norm_epsilon_, model_quant_type_);

    loadTensorMap(tensor_map_path);

    // Initialize KV Cache sizes per layer
    k_cache_.resize(num_layers_);
    v_cache_.resize(num_layers_);

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
    // V32: Esnek Hibrit Bellek (Elastic Tiered Memory) Tahsis Stratejisi
    // =========================================================================
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    size_t usable_for_layers = 0;
    if (vram_limit_gb_ > 0.1) {
        size_t requested_total = (size_t)(vram_limit_gb_ * 1024ULL * 1024ULL * 1024ULL);
        size_t already_used = total_bytes - free_bytes;
        if (requested_total > already_used) {
            usable_for_layers = requested_total - already_used;
            if (usable_for_layers > free_bytes) usable_for_layers = free_bytes;
        } else {
            usable_for_layers = 0;
        }
    } else {
        // Otomatik mod: 768 MB guvenlik marji birak
        size_t safety_margin = 768ULL * 1024ULL * 1024ULL;
        usable_for_layers = (free_bytes > safety_margin) ? (free_bytes - safety_margin) : free_bytes;
    }

    size_t total_layers_needed = (size_t)num_layers_ * layer_buffer_size_;
    if (usable_for_layers >= total_layers_needed) {
        num_gpu_resident_layers_ = num_layers_;
    } else {
        size_t ring_buffer_pair = 2 * layer_buffer_size_;
        if (usable_for_layers > ring_buffer_pair) {
            num_gpu_resident_layers_ = (int)((usable_for_layers - ring_buffer_pair) / layer_buffer_size_);
        } else {
            num_gpu_resident_layers_ = 0;
        }
    }
    if (num_gpu_resident_layers_ > num_layers_) num_gpu_resident_layers_ = num_layers_;
    if (num_gpu_resident_layers_ < 0) num_gpu_resident_layers_ = 0;

    int num_offload = num_layers_ - num_gpu_resident_layers_;

    printf("\n=======================================================\n");
    printf("  [V32] Esnek Hibrit Bellek (Elastic Tiered Memory)\n");
    printf("  Toplam VRAM: %.2f GB | Serbest VRAM: %.2f GB\n",
           (float)total_bytes / (1024.0f * 1024.0f * 1024.0f),
           (float)free_bytes / (1024.0f * 1024.0f * 1024.0f));
    if (vram_limit_gb_ > 0.1) {
        printf("  Kullanici VRAM Limiti: %.2f GB\n", vram_limit_gb_);
    } else {
        printf("  VRAM Modu: Otomatik Algilama (Maksimum GPU Gucu)\n");
    }
    printf("  Katman Boyutu: %.2f MB | Toplam Katman: %d\n",
           (float)layer_buffer_size_ / (1024.0f * 1024.0f), num_layers_);
    printf("  -> GPU VRAM'de Sabit Yerlesik Katmanlar: %d / %d (%.2f GB)\n",
           num_gpu_resident_layers_, num_layers_,
           (float)(num_gpu_resident_layers_ * layer_buffer_size_) / (1024.0f * 1024.0f * 1024.0f));
    printf("  -> Asenkron Stream Edilecek Katmanlar : %d / %d (%.2f GB)\n",
           num_offload, num_layers_,
           (float)(num_offload * layer_buffer_size_) / (1024.0f * 1024.0f * 1024.0f));
    printf("=======================================================\n\n");

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

    // 2. Stream edilecek katmanlar varsa, Ping-Pong tamponlari ve Host Onbellegi hazirla
    if (num_offload > 0) {
        CUDA_CHECK(cudaMalloc((void**)&d_layer_buffer_A_, layer_buffer_size_));
        CUDA_CHECK(cudaMalloc((void**)&d_layer_buffer_B_, layer_buffer_size_));
        CUDA_CHECK(cudaMemset(d_layer_buffer_A_, 0, layer_buffer_size_));
        CUDA_CHECK(cudaMemset(d_layer_buffer_B_, 0, layer_buffer_size_));

        CUDA_CHECK(cudaMallocHost((void**)&h_pinned_buffer_A_, layer_buffer_size_));
        CUDA_CHECK(cudaMallocHost((void**)&h_pinned_buffer_B_, layer_buffer_size_));

        // Offloaded katmanlari sistem RAM'inde (Pinned Host) onbellekle
        h_offload_pinned_cache_.resize(num_offload, nullptr);
        bool all_pinned_ok = true;
        for (int i = 0; i < num_offload; ++i) {
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
            printf("[Transformer] %d adet dis katman Sistem RAM'ine (Pinned Memory) onbelleklendi (Cikarim sirasinda 0 Disk I/O!).\n", num_offload);
        } else {
            printf("[Transformer] Uyari: Host RAM siniri nedeniyle %d dis katman kalici dosya handle'i ile stream edilecek.\n", num_offload);
        }
    }

    CUDA_CHECK(cudaStreamCreate(&stream_compute_));
    CUDA_CHECK(cudaStreamCreate(&stream_io_));
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

    if (d_layer_buffer_A_) CUDA_CHECK(cudaFree(d_layer_buffer_A_));
    if (d_layer_buffer_B_) CUDA_CHECK(cudaFree(d_layer_buffer_B_));
    if (h_pinned_buffer_A_) CUDA_CHECK(cudaFreeHost(h_pinned_buffer_A_));
    if (h_pinned_buffer_B_) CUDA_CHECK(cudaFreeHost(h_pinned_buffer_B_));

    if (d_token_embd_) CUDA_CHECK(cudaFree(d_token_embd_));
    if (d_output_norm_) CUDA_CHECK(cudaFree(d_output_norm_));
    if (d_lm_head_) CUDA_CHECK(cudaFree(d_lm_head_));
    
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

    // 5. KV-Cache ve Attention Hesaplamasi
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int kv_dim = num_kv_heads_ * head_dim_;

    std::vector<half> h_query(hidden_dim_);
    std::vector<half> h_key(kv_dim);
    std::vector<half> h_value(kv_dim);
    std::vector<half> h_attention_output(hidden_dim_, __float2half(0.0f));

    CUDA_CHECK(cudaMemcpy(h_query.data(), d_query_out, hidden_dim_ * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_key.data(), d_key_out, kv_dim * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_value.data(), d_value_out, kv_dim * sizeof(half), cudaMemcpyDeviceToHost));

    if (pos_offset == 0) {
        k_cache_[layer_idx].clear();
        v_cache_[layer_idx].clear();
    }

    k_cache_[layer_idx].push_back(h_key);
    v_cache_[layer_idx].push_back(h_value);

    int current_seq_len = (int)k_cache_[layer_idx].size();
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim_));
    int heads_per_kv = num_heads_ / num_kv_heads_;

    for (int h_q = 0; h_q < num_heads_; ++h_q) {
        int h_kv = h_q / heads_per_kv;
        int q_offset = h_q * head_dim_;
        int kv_offset = h_kv * head_dim_;

        std::vector<float> scores(current_seq_len);
        float max_score = -1e30f;

        // Step 1: FP32 Dot-product Q * K^T with scale
        for (int p = 0; p < current_seq_len; ++p) {
            float score = 0.0f;
            const auto& past_k = k_cache_[layer_idx][p];
            for (int d = 0; d < head_dim_; ++d) {
                float q_val = __half2float(h_query[q_offset + d]);
                float k_val = __half2float(past_k[kv_offset + d]);
                score += q_val * k_val;
            }
            score *= scale;
            if (std::isnan(score)) score = -1e30f;
            scores[p] = score;
            if (score > max_score) max_score = score;
        }

        // Step 2: SAFE SOFTMAX TRICK (FP32)
        float sum_exp = 0.0f;
        for (int p = 0; p < current_seq_len; ++p) {
            float diff = scores[p] - max_score;
            float exp_val = (diff < -50.0f) ? 0.0f : expf(diff);
            scores[p] = exp_val;
            sum_exp += exp_val;
        }

        float inv_sum_exp = (sum_exp > 1e-12f) ? (1.0f / sum_exp) : 0.0f;

        // Step 3: FP32 Score * V accumulation with FP16 range clamping
        for (int d = 0; d < head_dim_; ++d) {
            float out_val = 0.0f;
            for (int p = 0; p < current_seq_len; ++p) {
                float weight = scores[p] * inv_sum_exp;
                const auto& past_v = v_cache_[layer_idx][p];
                float v_val = __half2float(past_v[kv_offset + d]);
                out_val += weight * v_val;
            }
            if (std::isnan(out_val)) out_val = 0.0f;
            else if (out_val > 65504.0f) out_val = 65504.0f;
            else if (out_val < -65504.0f) out_val = -65504.0f;
            h_attention_output[q_offset + d] = __float2half(out_val);
        }
    }

    CUDA_CHECK(cudaMemcpy(d_query_out, h_attention_output.data(), hidden_dim_ * sizeof(half), cudaMemcpyHostToDevice));

    // Output Projeksiyonu
    FridayKernelDispatcher::dispatchGemvKernel(
        getQuantTypeFromDtype(o_info.dtype),
        d_query_out,
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
        checkBufferNaNs(d_input_activations, hidden_dim_, "L0 initial input_activations", compute_stream);
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
        int first_stream_layer = num_gpu_resident_layers_;
        const auto& l_first_info = layers_info_[first_stream_layer];

        // Ilk stream katmanini Buffer A'ya yukle
        if (!h_offload_pinned_cache_.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_layer_buffer_A_, h_offload_pinned_cache_[0], l_first_info.size_bytes, cudaMemcpyHostToDevice, compute_stream));
        } else {
            async_io_manager_ptr_->readLayerToHost(l_first_info.start_offset, l_first_info.size_bytes, h_pinned_buffer_A_);
            CUDA_CHECK(cudaMemcpyAsync(d_layer_buffer_A_, h_pinned_buffer_A_, l_first_info.size_bytes, cudaMemcpyHostToDevice, compute_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        unsigned char* d_curr_buf = d_layer_buffer_A_;
        unsigned char* d_next_buf = d_layer_buffer_B_;
        unsigned char* h_curr_pinned = h_pinned_buffer_A_;
        unsigned char* h_next_pinned = h_pinned_buffer_B_;

        for (int layer_idx = first_stream_layer; layer_idx < num_layers_; ++layer_idx) {
            int offload_idx = layer_idx - first_stream_layer;

            // Bir sonraki katmani stream_io_ ile arka planda kopyala
            if (layer_idx + 1 < num_layers_) {
                int next_offload_idx = offload_idx + 1;
                const auto& next_l_info = layers_info_[layer_idx + 1];
                if (!h_offload_pinned_cache_.empty()) {
                    CUDA_CHECK(cudaMemcpyAsync(d_next_buf, h_offload_pinned_cache_[next_offload_idx], next_l_info.size_bytes, cudaMemcpyHostToDevice, stream_io_));
                } else {
                    async_io_manager_ptr_->readLayerToHost(next_l_info.start_offset, next_l_info.size_bytes, h_next_pinned);
                    CUDA_CHECK(cudaMemcpyAsync(d_next_buf, h_next_pinned, next_l_info.size_bytes, cudaMemcpyHostToDevice, stream_io_));
                }
            }

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

            // stream_compute_ senkronize et (katman hesaplamalari tamamlansin)
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));

            // stream_io_ senkronize et ve Buffer A ile Buffer B isaretcilerini takas et (swap)
            if (layer_idx + 1 < num_layers_) {
                CUDA_CHECK(cudaStreamSynchronize(stream_io_));
                std::swap(d_curr_buf, d_next_buf);
                std::swap(h_curr_pinned, h_next_pinned);
            }
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
        checkBufferNaNs(d_input_activations, hidden_dim_, "post-final-norm activations", compute_stream);
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
        checkBufferNaNs(d_output_logits, (int)lm_head_info.shape[1], "LM Head output logits", compute_stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
}
