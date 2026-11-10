#include "transformer.h"
#include "attention_kernels.h"
#include "hpc_k_quants.h"
#include "prefill_kernels.h"
#include "cpu_kquants.h"
#include "gguf_file.h"
#include "model_config.h"
#include <algorithm>
#include "cpu_ops.h"
#include "cpu_threadpool.h"
#include <cstdlib>
#include <cstring>
#include <chrono>
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
__global__ void headRmsNormKernel(half* d_x, const float* d_weight, int head_dim, float epsilon);
__global__ void ropeStoreKvKernel(half* d_query, half* d_key, const half* d_value, half* k_layer, half* v_layer,
                                  const int* d_pos, int head_dim, int num_q_heads, int num_kv_heads, float freq_base,
                                  int neox, const float* freq_factors);
__global__ void storeKvQ8Kernel(const half* k, const half* v, int8_t* k_layer, half* ks_layer, int8_t* v_layer,
                                half* vs_layer, const int* d_pos, int kv_dim);
__global__ void ropeKernel(half* d_query, half* d_key, int head_dim, int seq_len, int pos_offset, int num_q_heads,
                           int num_kv_heads, float freq_base, int neox, const float* freq_factors);
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

// GGML Q3_K / Q4_K / Q6_K row dequant (one thread per element), same formulas
// as the GEMV kernels in hpc_k_quants.cu and gguf-py's reference dequantizer.
__global__ void dequantize_q3_K_row_kernel(const uint8_t* row, half* out, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    const uint8_t* b = row + (idx / 256) * 110;   // hmask[32] qs[64] scales[12] d
    const int r = idx % 256, h = r / 128, s = (r % 128) / 32, l = r % 32;
    const int j = h * 8 + 2 * s + (l >= 16 ? 1 : 0);
    const uint8_t* sc = b + 96;
    const int low = j < 8 ? (sc[j] & 0xF) : (sc[j - 8] >> 4);
    const int high = (sc[8 + (j % 4)] >> (2 * (j / 4))) & 3;
    const int scale = (low | (high << 4)) - 32;
    const int q = ((b[32 + h * 32 + l] >> (2 * s)) & 3) - ((b[l] & (1 << (h * 4 + s))) ? 0 : 4);
    const half d = *reinterpret_cast<const half*>(b + 108);
    out[idx] = __float2half(__half2float(d) * (float)scale * (float)q);
}

__global__ void dequantize_q4_K_row_kernel(const uint8_t* row, half* out, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    const uint8_t* b = row + (idx / 256) * 144;   // d dmin scales[12] qs[128]
    const int r = idx % 256, j = r / 64, hi = (r % 64) / 32, l = r % 32;
    const int is = 2 * j + hi;
    const uint8_t* q = b + 4;
    int sc, m;
    if (is < 4) { sc = q[is] & 63; m = q[is + 4] & 63; }
    else { sc = (q[is + 4] & 0xF) | ((q[is - 4] >> 6) << 4); m = (q[is + 4] >> 4) | ((q[is] >> 6) << 4); }
    const uint8_t qb = b[16 + 32 * j + l];
    const int v = hi ? (qb >> 4) : (qb & 0xF);
    const float d = __half2float(*reinterpret_cast<const half*>(b));
    const float dmin = __half2float(*reinterpret_cast<const half*>(b + 2));
    out[idx] = __float2half(d * (float)sc * (float)v - dmin * (float)m);
}

__global__ void dequantize_q6_K_row_kernel(const uint8_t* row, half* out, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    const uint8_t* b = row + (idx / 256) * 210;   // ql[128] qh[64] scales[16] d
    const int r = idx % 256, h = r / 128, part = (r % 128) / 32, l = r % 32;
    const uint8_t ql = b[h * 64 + l + (part & 1) * 32];
    const uint8_t qh = b[128 + h * 32 + l];
    const int lo = (part < 2) ? (ql & 0xF) : (ql >> 4);
    const int q = (lo | (((qh >> (2 * part)) & 3) << 4)) - 32;
    const int8_t sc = reinterpret_cast<const int8_t*>(b + 192)[h * 8 + (l / 16) + 2 * part];
    const float d = __half2float(*reinterpret_cast<const half*>(b + 208));
    out[idx] = __float2half(d * (float)sc * (float)q);
}

__global__ void dequantize_q5_K_row_kernel(const uint8_t* row, half* out, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    const uint8_t* b = row + (idx / 256) * 176;   // d dmin scales[12] qh[32] qs[128]
    const int r = idx % 256, j = r / 64, hi = (r % 64) / 32, l = r % 32;
    const int is = 2 * j + hi;
    const uint8_t* q = b + 4;
    int sc, m;
    if (is < 4) { sc = q[is] & 63; m = q[is + 4] & 63; }
    else { sc = (q[is + 4] & 0xF) | ((q[is - 4] >> 6) << 4); m = (q[is + 4] >> 4) | ((q[is] >> 6) << 4); }
    const uint8_t qb = b[48 + 32 * j + l];
    const int v = (hi ? (qb >> 4) : (qb & 0xF)) + (((b[16 + l] >> is) & 1) << 4);
    const float d = __half2float(*reinterpret_cast<const half*>(b));
    const float dmin = __half2float(*reinterpret_cast<const half*>(b + 2));
    out[idx] = __float2half(d * (float)sc * (float)v - dmin * (float)m);
}

__global__ void dequantize_q8_0_row_kernel(const uint8_t* row, half* out, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    const uint8_t* b = row + (idx / 32) * 34;    // fp16 d, int8 qs[32]
    const float d = __half2float(*reinterpret_cast<const half*>(b));
    out[idx] = __float2half(d * (float)reinterpret_cast<const int8_t*>(b + 2)[idx % 32]);
}

bool launchDequantizeRow(const std::string& dtype, const uint8_t* d_row, half* d_out, int dim, cudaStream_t s) {
    const int threads = 256, blocks = (dim + threads - 1) / threads;
    if (dtype == "Q4_0") dequantize_q4_0_row_kernel<<<blocks, threads, 0, s>>>(d_row, d_out, dim);
    else if (dtype == "Q2_K") dequantize_q2_K_row_kernel<<<blocks, threads, 0, s>>>(d_row, d_out, dim);
    else if (dtype == "Q3_K") dequantize_q3_K_row_kernel<<<blocks, threads, 0, s>>>(d_row, d_out, dim);
    else if (dtype == "Q4_K") dequantize_q4_K_row_kernel<<<blocks, threads, 0, s>>>(d_row, d_out, dim);
    else if (dtype == "Q6_K") dequantize_q6_K_row_kernel<<<blocks, threads, 0, s>>>(d_row, d_out, dim);
    else if (dtype == "Q5_K") dequantize_q5_K_row_kernel<<<blocks, threads, 0, s>>>(d_row, d_out, dim);
    else if (dtype == "Q8_0") dequantize_q8_0_row_kernel<<<blocks, threads, 0, s>>>(d_row, d_out, dim);
    else if (dtype == "F16") return cudaMemcpyAsync(d_out, d_row, (size_t)dim * sizeof(half), cudaMemcpyDeviceToDevice, s) == cudaSuccess;
    else return false;
    return cudaGetLastError() == cudaSuccess;
}

size_t embeddingRowBytes(const std::string& dtype, int dim) {
    if (dtype == "Q4_0") return (size_t)(dim / 32) * 18;
    if (dtype == "Q2_K") return (size_t)(dim / 256) * 84;
    if (dtype == "Q3_K") return (size_t)(dim / 256) * 110;
    if (dtype == "Q4_K") return (size_t)(dim / 256) * 144;
    if (dtype == "Q6_K") return (size_t)(dim / 256) * 210;
    if (dtype == "Q5_K") return (size_t)(dim / 256) * 176;
    if (dtype == "Q8_0") return (size_t)(dim / 32) * 34;
    if (dtype == "F16") return (size_t)dim * sizeof(half);
    return 0;
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
    if (dtype_str == "Q5_K") return Q5_K;
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
    int max_context,
    int cpu_threads,
    KvCacheMode kv_mode
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
      max_ctx_(max_context),
      kv_mode_(kv_mode)
{
    if (max_ctx_ <= 0) throw std::runtime_error("max_context must be positive");
    if (hidden_dim_ % num_heads_ != 0) {
        fprintf(stderr, "Error: hidden_dim (%d) must be divisible by num_heads (%d).\n", hidden_dim_, num_heads_);
        exit(EXIT_FAILURE);
    }
    printf("Transformer initialized with GGUF parameters: hidden_dim=%d, num_heads=%d, num_kv_heads=%d, num_layers=%d, epsilon=%f, QuantType=%d\n",
           hidden_dim_, num_heads_, num_kv_heads_, num_layers_, rms_norm_epsilon_, model_quant_type_);

    if (model_file_path.size() > 5 && model_file_path.compare(model_file_path.size() - 5, 5, ".gguf") == 0) {
        loadTensorsFromGguf(model_file_path);
    } else {
        fprintf(stderr, "[Transformer] Deprecated: loading a converted .bin + tensor map. Pass the .gguf file "
                        "directly; converter.py support will be removed in a later release.\n");
        loadTensorMap(tensor_map_path);
    }
    buildLayerMap();

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

    CUDA_CHECK(cudaMalloc((void**)&d_attn_ctx_, hidden_dim_ * sizeof(half)));
    {
        const char* e = std::getenv("FRIDAY_OFFLOAD");
        cpu_offload_ = !(e && std::strcmp(e, "stream") == 0);
        const char* pr = std::getenv("FRIDAY_PROFILE");
        profile_ = pr && pr[0] == '1';
    }

    // =========================================================================
    // Katman Disi Statik Tensorleri Yukle (Token Embedding, Output Norm, LM Head)
    // =========================================================================
    std::string emb_name = "token_embd.weight";
    if (!full_tensor_map_.count(emb_name)) emb_name = "model.embed_tokens.weight";
    if (full_tensor_map_.count(emb_name)) {
        const auto& info = full_tensor_map_.at(emb_name);
        // The embedding table stays in host RAM (one row is needed per token), which
        // frees its VRAM for transformer layers. getEmbedding() stages one row.
        h_token_embd_.reset(new unsigned char[info.size_bytes]);
        if (async_io_manager_ptr_->readLayerToHost(info.absolute_offset, info.size_bytes, h_token_embd_.get())) {
            const size_t max_row = (size_t)hidden_dim_ * sizeof(half);  // largest supported row (F16)
            CUDA_CHECK(cudaMalloc((void**)&d_token_embd_, max_row));
            CUDA_CHECK(cudaMallocHost((void**)&h_emb_row_pinned_, max_row));
            printf("[Transformer] Statik tensor '%s' host RAM'de (%.2f MB)\n",
                   emb_name.c_str(), (float)info.size_bytes / (1024.0f * 1024.0f));
        } else {
            fprintf(stderr, "Hata: '%s' tensörü okunamadi!\n", emb_name.c_str());
            exit(EXIT_FAILURE);
        }
    }

    // Llama 3 RoPE frequency factors (F32 [head_dim / 2]), if the model has them.
    if (full_tensor_map_.count("rope_freqs.weight")) {
        const auto& info = full_tensor_map_.at("rope_freqs.weight");
        if (info.dtype != "F32" || info.size_bytes != (size_t)(head_dim_ / 2) * sizeof(float))
            throw std::runtime_error("rope_freqs.weight: expected F32 [head_dim / 2]");
        h_rope_freqs_.resize(head_dim_ / 2);
        if (!async_io_manager_ptr_->readLayerToHost(info.absolute_offset, info.size_bytes,
                                                    reinterpret_cast<unsigned char*>(h_rope_freqs_.data())))
            throw std::runtime_error("cannot read rope_freqs.weight");
        CUDA_CHECK(cudaMalloc((void**)&d_rope_freqs_, info.size_bytes));
        CUDA_CHECK(cudaMemcpy(d_rope_freqs_, h_rope_freqs_.data(), info.size_bytes, cudaMemcpyHostToDevice));
        printf("[Transformer] RoPE frequency factors loaded (%d)\n", head_dim_ / 2);
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

    // Fused path: every GEMV weight of every layer needs an int8-activation kernel.
    // Decided before planning because the int8 KV cache depends on it.
    {
        const char* f = std::getenv("FRIDAY_FUSED");
        const char* fp32 = std::getenv("FRIDAY_GEMV_FP32");
        fused_ok_ = !(f && f[0] == '0') && !(fp32 && fp32[0] == '1');
        for (const auto& [name, t] : full_tensor_map_) {
            if (name.rfind("blk.", 0) != 0 || name.find(".weight") == std::string::npos || t.shape.size() < 2) continue;
            if (friday_hpc::q8KindFromDtype(t.dtype) == friday_hpc::Q8Kind::None) { fused_ok_ = false; break; }
        }
    }

    size_t budget = free_bytes_before_residency;
    if (vram_limit_gb_ > 0.1) {
        const size_t requested_total = (size_t)(vram_limit_gb_ * 1024.0 * 1024.0 * 1024.0);
        const size_t already_used = total_bytes - free_bytes_before_residency;
        const size_t capped = (requested_total > already_used) ? (requested_total - already_used) : 0;
        if (capped < budget) budget = capped;
    }

    // GPU KV cache per layer (K + V). fp16, or int8 + one fp16 scale per 32 values
    // (fused path only). In CPU-offload mode only GPU layers need it, so it is
    // charged per resident layer; in stream mode every layer is computed on the
    // GPU and the whole cache is reserved up front.
    const size_t kv_elem = (size_t)max_ctx_ * num_kv_heads_ * head_dim_;
    const size_t kv_bytes_f16 = 2ULL * kv_elem * sizeof(half);
    const size_t kv_bytes_q8 = 2ULL * (kv_elem + kv_elem / 32 * sizeof(half));
    const size_t dma_reservation = cpu_offload_ ? 0 : (size_t)kDmaSlotCount * layer_buffer_size_;
    // Prefill staging buffers are charged only when the model does not fit (stage = true).
    const size_t stage_bytes = (size_t)kPrefillStageCount * layer_buffer_size_;
    bool stage_enabled = cpu_offload_;
    if (const char* ps = std::getenv("FRIDAY_PREFILL_STAGE")) stage_enabled = stage_enabled && ps[0] != '0';
    auto overheadFor = [&](size_t kv_layer, bool stage) {
        return kMinGpuReserveBytes + dma_reservation + (cpu_offload_ ? 0 : (size_t)num_layers_ * kv_layer) +
               (stage ? stage_bytes : 0);
    };
    auto residentForStage = [&](size_t kv_layer, bool stage) {
        const size_t ov = overheadFor(kv_layer, stage);
        if (budget < ov) return -1;
        size_t avail = budget - ov;
        int n = 0;
        for (int i = 0; i < num_layers_; ++i) {
            const size_t l_size = layers_info_[i].size_bytes + (cpu_offload_ ? kv_layer : 0);
            if (avail < l_size) break;
            avail -= l_size;
            ++n;
        }
        return n;
    };
    // Layers on the GPU: without staging if everything fits, otherwise with it.
    auto residentFor = [&](size_t kv_layer) {
        const int all = residentForStage(kv_layer, false);
        if (all >= num_layers_ || !stage_enabled) return all;
        return residentForStage(kv_layer, true);
    };

    // KV format: f16 / q8 as requested; auto = f16 if every layer fits on the GPU
    // with it, otherwise q8 (smaller cache, more GPU layers).
    const char* kv_reason = "";
    if (kv_mode_ == KvCacheMode::Q8 && !fused_ok_) {
        kv_q8_ = false;
        kv_reason = "int8 requested, but the fused GPU path is not available";
    } else if (kv_mode_ == KvCacheMode::Q8) {
        kv_q8_ = true;
        kv_reason = "requested";
    } else if (kv_mode_ == KvCacheMode::F16) {
        kv_q8_ = false;
        kv_reason = "requested";
    } else if (!fused_ok_ || !cpu_offload_) {
        kv_q8_ = false;
        kv_reason = "auto: int8 needs the fused path and CPU offload";
    } else if (residentFor(kv_bytes_f16) >= num_layers_) {
        kv_q8_ = false;
        kv_reason = "auto: the whole model fits on the GPU with fp16";
    } else {
        kv_q8_ = residentFor(kv_bytes_q8) > residentFor(kv_bytes_f16);
        kv_reason = kv_q8_ ? "auto: int8 fits more layers on the GPU" : "auto: int8 would not fit more layers";
    }
    const size_t kv_layer_bytes = kv_q8_ ? kv_bytes_q8 : kv_bytes_f16;
    const bool use_stage = stage_enabled && residentForStage(kv_layer_bytes, false) < num_layers_;
    const size_t required_overhead = overheadFor(kv_layer_bytes, use_stage);
    printf("[Transformer] KV cache format: %s (%s)\n", kv_q8_ ? "int8" : "fp16", kv_reason);

    if (budget < required_overhead) {
        throw std::runtime_error(
            "VRAM budget too small: " + std::to_string(budget / (1024 * 1024)) +
            " MiB available, " + std::to_string(required_overhead / (1024 * 1024)) +
            " MiB required for DMA slots and safety reserve");
    }

    num_gpu_resident_layers_ = residentFor(kv_layer_bytes);
    size_t resident_weight_bytes = 0;
    for (int i = 0; i < num_gpu_resident_layers_; ++i) resident_weight_bytes += layers_info_[i].size_bytes;
    const int num_streamed = num_layers_ - num_gpu_resident_layers_;

    printf("\n=======================================================\n");
    printf("  VRAM total: %.2f GB | free before layers: %.2f GB",
           (double)total_bytes / (1u << 30), (double)free_bytes_before_residency / (1u << 30));
    if (vram_limit_gb_ > 0.1) printf(" | --vram cap: %.2f GB", vram_limit_gb_);
    printf("\n  Reserved: %zu MiB safety + %zu MiB DMA (%d slots) + %zu MiB prefill staging\n",
           kMinGpuReserveBytes >> 20, dma_reservation >> 20, cpu_offload_ ? 0 : kDmaSlotCount,
           use_stage ? stage_bytes >> 20 : (size_t)0);
    printf("  GPU-resident layers: %d / %d (%.2f GB) | %s: %d\n",
           num_gpu_resident_layers_, num_layers_, (double)resident_weight_bytes / (1u << 30),
           cpu_offload_ ? "CPU-computed" : "host-streamed", num_streamed);
    printf("=======================================================\n\n");

    {
        const int kv_layers = cpu_offload_ ? num_gpu_resident_layers_ : num_layers_;
        const size_t kv_elems = (size_t)(kv_layers > 0 ? kv_layers : 1) * max_ctx_ * num_kv_heads_ * head_dim_;
        if (kv_q8_) {
            CUDA_CHECK(cudaMalloc((void**)&d_k_cache_, kv_elems));
            CUDA_CHECK(cudaMalloc((void**)&d_v_cache_, kv_elems));
            CUDA_CHECK(cudaMalloc((void**)&d_k_scale_, kv_elems / 32 * sizeof(half)));
            CUDA_CHECK(cudaMalloc((void**)&d_v_scale_, kv_elems / 32 * sizeof(half)));
        } else {
            CUDA_CHECK(cudaMalloc((void**)&d_k_cache_, kv_elems * sizeof(half)));
            CUDA_CHECK(cudaMalloc((void**)&d_v_cache_, kv_elems * sizeof(half)));
        }
        printf("[Transformer] GPU KV cache: %d layers, max context %d, %s, %.0f MiB (K + V)\n", kv_layers, max_ctx_,
               kv_q8_ ? "int8" : "fp16", (double)(kv_layers > 0 ? kv_layers : 1) * kv_layer_bytes / (1024.0 * 1024.0));
    }

    // Fused-path resources (fused_ok_ was decided before planning).
    {
        if (fused_ok_) friday_hpc::ensureQ8Scratch(std::max(hidden_dim_, max_intermediate_size_), nullptr);
        CUDA_CHECK(cudaMalloc((void**)&d_pos_, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_attn_ws_, friday_attn::splitAttentionWorkspaceBytes(num_heads_, max_ctx_)));
        CUDA_CHECK(cudaMallocHost((void**)&h_pos_, sizeof(int)));
        // CUDA Graph of the GPU-resident layers: fused path, layers not streamed over PCIe,
        // attention shared memory sized for max_ctx. FRIDAY_GRAPH=0 disables it.
        const char* g = std::getenv("FRIDAY_GRAPH");
        // Split-K attention (default); FRIDAY_ATTN_SPLIT=0 selects the one-block-per-head
        // kernel, which needs the context to fit in shared memory.
        const char* sp = std::getenv("FRIDAY_ATTN_SPLIT");
        attn_split_ = !(sp && sp[0] == '0') || !friday_attn::decodeAttentionFitsSharedMemory(max_ctx_);
        graph_ok_ = fused_ok_ && !(g && g[0] == '0');
        printf("[Transformer] GPU layer path: %s%s\n", fused_ok_ ? "fused (int8 activations)" : "unfused",
               graph_ok_ ? " + CUDA Graph" : "");
    }

    // Streams are created before any event is recorded on them.
    CUDA_CHECK(cudaStreamCreate(&stream_compute_));
    CUDA_CHECK(cudaStreamCreate(&stream_io_));

    // 1. GPU'da kalici tutulacak katmanlari yukle
    d_gpu_resident_layers_.resize(num_gpu_resident_layers_, nullptr);
    std::vector<unsigned char> temp_load_buf(layer_buffer_size_);
    for (int l = 0; l < num_gpu_resident_layers_; ++l) {
        const auto& l_info = layers_info_[l];
        CUDA_CHECK(cudaMalloc((void**)&d_gpu_resident_layers_[l], l_info.size_bytes));
        if (!readLayer(l, temp_load_buf.data())) throw std::runtime_error("Failed to read layer " + std::to_string(l));
        CUDA_CHECK(cudaMemcpy(d_gpu_resident_layers_[l], temp_load_buf.data(), l_info.size_bytes, cudaMemcpyHostToDevice));
    }
    if (num_gpu_resident_layers_ > 0) {
        printf("[Transformer] %d adet katman GPU VRAM'e kalici yuklendi (Sifir transfer gecikmesi).\n", num_gpu_resident_layers_);
    }
    // The staging buffers are only reserved here; batched prefill allocates them for
    // its duration and frees them afterwards. Decode with that VRAM allocated (even
    // untouched) was unstable on the dev box: 9 of 17 runs slow vs 0 of 11 with it free.
    prefill_stage_reserved_ = use_stage;

    // 2a. CPU-offload mode: keep the remaining layers in host RAM and compute them on the CPU.
    if (cpu_offload_ && num_streamed > 0) {
        friday::CpuModelParams cp;
        cp.hidden_dim = hidden_dim_;
        cp.num_heads = num_heads_;
        cp.num_kv_heads = num_kv_heads_;
        cp.head_dim = head_dim_;
        cp.max_ffn_dim = max_intermediate_size_;
        cp.rms_norm_epsilon = rms_norm_epsilon_;
        cp.rope_freq_base = rope_freq_base_;
        cp.rope_neox = rope_neox_;
        cp.rope_freqs = h_rope_freqs_;
        cp.max_ctx = max_ctx_;
        cpu_runner_.reset(new friday::CpuLayerRunner(layout_, *async_io_manager_ptr_, cp, num_gpu_resident_layers_,
                                                     num_layers_, cpu_threads));
        CUDA_CHECK(cudaMallocHost((void**)&h_act_pinned_, hidden_dim_ * sizeof(half)));
        printf("[Transformer] %d layers in host RAM (%.2f GB), computed on the CPU with %d threads (%s)\n",
               num_streamed, (double)cpu_runner_->weightBytes() / (1u << 30), cpu_runner_->threads(),
               friday_cpu::hasAvx2() ? "AVX2" : "scalar");
    }

    // 2b. Stream mode: DMA slots and pinned host cache for the streamed layers.
    if (!cpu_offload_ && num_streamed > 0) {
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
                if (!readLayer(l_idx, h_offload_pinned_cache_[i])) throw std::runtime_error("Failed to read layer " + std::to_string(l_idx));
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
    if (profile_ && !prof_gpu_ms_.empty()) {
        auto stats = [](std::vector<float> v, const char* name) {
            std::sort(v.begin(), v.end());
            auto at = [&](double p) { return v[(size_t)(p * (v.size() - 1))]; };
            printf("[profile] %-5s per token: median %7.2f ms  p95 %7.2f ms  max %7.2f ms  (n=%zu)\n", name, at(0.5),
                   at(0.95), v.back(), v.size());
        };
        stats(prof_gpu_ms_, "gpu");
        stats(prof_cpu_ms_, "cpu");
        stats(prof_tail_ms_, "tail");
    }
    printf("[Transformer] Destructor cagrildi: GPU ve Host bellekleri serbest birakiliyor...\n");
    for (auto* ptr : d_gpu_resident_layers_) {
        if (ptr) CUDA_CHECK(cudaFree(ptr));
    }
    d_gpu_resident_layers_.clear();
    for (auto* p : d_prefill_stage_) {
        if (p) CUDA_CHECK(cudaFree(p));
    }

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
    if (h_emb_row_pinned_) CUDA_CHECK(cudaFreeHost(h_emb_row_pinned_));
    if (d_output_norm_) CUDA_CHECK(cudaFree(d_output_norm_));
    if (d_lm_head_) CUDA_CHECK(cudaFree(d_lm_head_));
    if (d_k_cache_) CUDA_CHECK(cudaFree(d_k_cache_));
    if (d_v_cache_) CUDA_CHECK(cudaFree(d_v_cache_));
    if (d_attn_ctx_) CUDA_CHECK(cudaFree(d_attn_ctx_));
    if (d_rope_freqs_) CUDA_CHECK(cudaFree(d_rope_freqs_));
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (d_pos_) CUDA_CHECK(cudaFree(d_pos_));
    if (d_attn_ws_) CUDA_CHECK(cudaFree(d_attn_ws_));
    if (d_k_scale_) CUDA_CHECK(cudaFree(d_k_scale_));
    if (d_v_scale_) CUDA_CHECK(cudaFree(d_v_scale_));
    if (h_pos_) CUDA_CHECK(cudaFreeHost(h_pos_));
    if (h_act_pinned_) CUDA_CHECK(cudaFreeHost(h_act_pinned_));
    
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
    layout_.loadTensorMap(map_path);
}

void Transformer::loadTensorsFromGguf(const std::string& gguf_path) {
    const friday::GgufFile g(gguf_path);
    const friday::ModelConfig cfg = friday::ModelConfig::fromGguf(g);
    rope_freq_base_ = cfg.rope_freq_base;
    rope_neox_ = cfg.ropeNeox();
    layout_.loadGguf(g);
}

void Transformer::buildLayerMap() {
    layout_.buildLayers(num_layers_);
    const size_t aligned_max = ((layout_.max_layer_bytes + 1048575) / 1048576) * 1048576;
    layer_buffer_size_ = std::max((size_t)(32 * 1024 * 1024), aligned_max);
}

bool Transformer::readLayer(int layer_idx, unsigned char* dst) {
    return layout_.readLayer(*async_io_manager_ptr_, layer_idx, dst);
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
        num_kv_heads_,
        rope_freq_base_,
        rope_neox_ ? 1 : 0,
        d_rope_freqs_
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

    // Qwen3: per-head RMSNorm of Q and K before RoPE (attn_q_norm / attn_k_norm).
    {
        const std::string qn = "blk." + std::to_string(layer_idx) + ".attn_q_norm.weight";
        const std::string kn = "blk." + std::to_string(layer_idx) + ".attn_k_norm.weight";
        if (l_info.tensor_rel_offsets.count(qn) && l_info.tensor_rel_offsets.count(kn)) {
            const float* wq = (const float*)(d_layer_buf + l_info.tensor_rel_offsets.at(qn));
            const float* wk = (const float*)(d_layer_buf + l_info.tensor_rel_offsets.at(kn));
            headRmsNormKernel<<<num_heads_, head_dim_, head_dim_ * sizeof(float), stream>>>(d_query_out, wq, head_dim_, rms_norm_epsilon_);
            headRmsNormKernel<<<num_kv_heads_, head_dim_, head_dim_ * sizeof(float), stream>>>(d_key_out, wk, head_dim_, rms_norm_epsilon_);
            CUDA_CHECK(cudaGetLastError());
        }
    }

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
    if (!h_token_embd_) return false;
    std::string embed_name = "token_embd.weight";
    if (!full_tensor_map_.count(embed_name)) {
        embed_name = "model.embed_tokens.weight";
        if (!full_tensor_map_.count(embed_name)) return false;
    }
    const auto& t_info = full_tensor_map_.at(embed_name);

    if (token_id < 0 || token_id >= vocab_size) token_id = 151643;

    const size_t row_bytes = embeddingRowBytes(t_info.dtype, hidden_dim_);
    if (row_bytes == 0) {
        // Previously every other dtype was copied as raw FP16, which read out of bounds.
        fprintf(stderr, "getEmbedding: unsupported embedding dtype '%s'\n", t_info.dtype.c_str());
        return false;
    }
    if ((size_t)(token_id + 1) * row_bytes > t_info.size_bytes) {
        fprintf(stderr, "getEmbedding: token %d is outside the embedding table\n", token_id);
        return false;
    }

    // Stage this token's quantized row: host table -> pinned buffer -> device row
    // buffer, then dequantize on the GPU exactly as before. The sync makes sure the
    // previous copy out of the pinned buffer has finished.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::memcpy(h_emb_row_pinned_, h_token_embd_.get() + (size_t)token_id * row_bytes, row_bytes);
    CUDA_CHECK(cudaMemcpyAsync(d_token_embd_, h_emb_row_pinned_, row_bytes, cudaMemcpyHostToDevice, stream));
    if (!launchDequantizeRow(t_info.dtype, (const uint8_t*)d_token_embd_, d_output, hidden_dim_, stream)) {
        fprintf(stderr, "getEmbedding: dequantization launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    return true;
}

// One transformer layer on the GPU with the int8-activation pipeline. Same math
// as the unfused sequence (RMSNorm, Q/K/V + bias, [q/k norm], RoPE, KV write,
// attention, O + residual, RMSNorm, gate/up, SwiGLU, down + residual); only the
// kernel boundaries differ.
void Transformer::gpuLayerFused(int layer_idx, unsigned char* buf, half* d_x, int pos, cudaStream_t s) {
    using friday_hpc::Q8Kind;
    const auto& li = layers_info_[layer_idx];
    const std::string pre = "blk." + std::to_string(layer_idx) + ".";
    auto ptr = [&](const char* n) -> unsigned char* {
        auto it = li.tensor_rel_offsets.find(pre + n);
        return it == li.tensor_rel_offsets.end() ? nullptr : buf + it->second;
    };
    auto info = [&](const char* n) -> const TensorInfo& { return full_tensor_map_.at(pre + n); };
    auto kind = [&](const char* n) { return friday_hpc::q8KindFromDtype(info(n).dtype); };
    if (pos < 0 || pos >= max_ctx_)
        throw std::runtime_error("context length exceeded: position " + std::to_string(pos) + " >= max context " +
                                 std::to_string(max_ctx_));
    const int H = hidden_dim_, kv_dim = num_kv_heads_ * head_dim_;

    // Attention block
    friday_hpc::rmsNormQuantQ8(d_x, (const float*)ptr("attn_norm.weight"), H, rms_norm_epsilon_, s);
    friday_hpc::gemvQ8(kind("attn_q.weight"), ptr("attn_q.weight"), d_query_out_, H, (int)info("attn_q.weight").shape[1],
                       s, nullptr, nullptr, (const float*)ptr("attn_q.bias"));
    friday_hpc::gemvQ8(kind("attn_k.weight"), ptr("attn_k.weight"), d_key_out_, H, (int)info("attn_k.weight").shape[1],
                       s, nullptr, nullptr, (const float*)ptr("attn_k.bias"));
    friday_hpc::gemvQ8(kind("attn_v.weight"), ptr("attn_v.weight"), d_value_out_, H, (int)info("attn_v.weight").shape[1],
                       s, nullptr, nullptr, (const float*)ptr("attn_v.bias"));
    if (ptr("attn_q_norm.weight") && ptr("attn_k_norm.weight")) {
        headRmsNormKernel<<<num_heads_, head_dim_, head_dim_ * sizeof(float), s>>>(
            d_query_out_, (const float*)ptr("attn_q_norm.weight"), head_dim_, rms_norm_epsilon_);
        headRmsNormKernel<<<num_kv_heads_, head_dim_, head_dim_ * sizeof(float), s>>>(
            d_key_out_, (const float*)ptr("attn_k_norm.weight"), head_dim_, rms_norm_epsilon_);
    }
    // Position-dependent kernels read the position from d_pos_ (set by forward()),
    // so this sequence can be captured once in a CUDA Graph and replayed.
    (void)pos;
    const size_t layer_elems = (size_t)layer_idx * max_ctx_ * kv_dim;
    if (kv_q8_) {
        int8_t* kq = reinterpret_cast<int8_t*>(d_k_cache_) + layer_elems;
        int8_t* vq = reinterpret_cast<int8_t*>(d_v_cache_) + layer_elems;
        half* ks = d_k_scale_ + layer_elems / 32;
        half* vs = d_v_scale_ + layer_elems / 32;
        ropeStoreKvKernel<<<(head_dim_ / 2 + 255) / 256, 256, 0, s>>>(
            d_query_out_, d_key_out_, d_value_out_, nullptr, nullptr, d_pos_, head_dim_, num_heads_, num_kv_heads_,
            rope_freq_base_, rope_neox_ ? 1 : 0, d_rope_freqs_);
        const int warps = 8, nblk = 2 * kv_dim / 32;
        storeKvQ8Kernel<<<(nblk + warps - 1) / warps, warps * 32, 0, s>>>(d_key_out_, d_value_out_, kq, ks, vq, vs,
                                                                          d_pos_, kv_dim);
        friday_attn::launchDecodeAttentionSplit(d_query_out_, kq, vq, d_attn_ctx_, 0, d_pos_, max_ctx_, num_heads_,
                                                num_kv_heads_, head_dim_, d_attn_ws_, s, ks, vs);
    } else {
    half* k_layer = d_k_cache_ + layer_elems;
    half* v_layer = d_v_cache_ + layer_elems;
    ropeStoreKvKernel<<<(head_dim_ / 2 + 255) / 256, 256, 0, s>>>(
        d_query_out_, d_key_out_, d_value_out_, k_layer, v_layer, d_pos_, head_dim_, num_heads_, num_kv_heads_,
        rope_freq_base_, rope_neox_ ? 1 : 0, d_rope_freqs_);
    if (attn_split_) {
        friday_attn::launchDecodeAttentionSplit(d_query_out_, k_layer, v_layer, d_attn_ctx_, 0, d_pos_, max_ctx_,
                                                num_heads_, num_kv_heads_, head_dim_, d_attn_ws_, s);
    } else {
        friday_attn::launchDecodeAttentionDevPos(d_query_out_, k_layer, v_layer, d_attn_ctx_, d_pos_, max_ctx_, num_heads_,
                                                 num_kv_heads_, head_dim_, s);
    }
    }
    friday_hpc::quantizeQ8(d_attn_ctx_, num_heads_ * head_dim_, s);
    friday_hpc::gemvQ8(kind("attn_output.weight"), ptr("attn_output.weight"), d_x, num_heads_ * head_dim_, H, s,
                       nullptr, nullptr, nullptr, d_x);

    // MLP block (SwiGLU)
    friday_hpc::rmsNormQuantQ8(d_x, (const float*)ptr("ffn_norm.weight"), H, rms_norm_epsilon_, s);
    const int I = (int)info("ffn_gate.weight").shape[1];
    if (kind("ffn_gate.weight") == kind("ffn_up.weight") && info("ffn_up.weight").shape[1] == I) {
        friday_hpc::gemvQ8(kind("ffn_gate.weight"), ptr("ffn_gate.weight"), d_gate_out_, H, I, s, ptr("ffn_up.weight"),
                           d_up_out_);
    } else {
        friday_hpc::gemvQ8(kind("ffn_gate.weight"), ptr("ffn_gate.weight"), d_gate_out_, H, I, s);
        friday_hpc::gemvQ8(kind("ffn_up.weight"), ptr("ffn_up.weight"), d_up_out_, H, I, s);
    }
    friday_hpc::siluMulQuantQ8(d_gate_out_, d_up_out_, I, s);
    friday_hpc::gemvQ8(kind("ffn_down.weight"), ptr("ffn_down.weight"), d_x, I, H, s, nullptr, nullptr, nullptr, d_x);
    CUDA_CHECK(cudaGetLastError());
}

// Final norm + LM head of one hidden state (end of forward() and prefill()).
void Transformer::finalHead(half* d_x, half* d_output_logits, cudaStream_t s) {
    // Final RMSNorm (Post-transformer norm, weights are F32)
    if (d_output_norm_) {
        applyRMSNorm(
            d_x,
            (const float*)d_output_norm_,
            d_x,
            hidden_dim_,
            s
        );
#ifdef FRIDAY_DEBUG_NANS  // diagnostic: syncs the stream and copies the buffer to host
        checkBufferNaNs(d_x, hidden_dim_, "post-final-norm activations", s);
#endif
    }

    // LM Head (Output Projeksiyonu)
    if (d_lm_head_) {
        std::string lm_head_weight_name = "output.weight";
        if (!full_tensor_map_.count(lm_head_weight_name)) lm_head_weight_name = "lm_head.weight";
        const TensorInfo& lm_head_info = full_tensor_map_.at(lm_head_weight_name);

        FridayKernelDispatcher::dispatchGemvKernel(
            getQuantTypeFromDtype(lm_head_info.dtype),
            d_x,
            d_lm_head_,
            nullptr,
            nullptr,
            d_output_logits,
            (int)lm_head_info.shape[0],
            (int)lm_head_info.shape[1],
            s
        );
        CUDA_CHECK(cudaGetLastError());
#ifdef FRIDAY_DEBUG_NANS  // diagnostic: syncs the stream and copies the buffer to host
        checkBufferNaNs(d_output_logits, (int)lm_head_info.shape[1], "LM Head output logits", s);
#endif
    }
}

// ---------------------------------------------------------------------------
// Batched prefill
// ---------------------------------------------------------------------------
bool Transformer::prefillSupported() const {
    if (!fused_ok_) return false;
    if (!cpu_offload_ && num_gpu_resident_layers_ < num_layers_) return false;  // PCIe stream mode
    for (const auto& [name, t] : full_tensor_map_) {
        if (name.rfind("blk.", 0) != 0 || name.find(".weight") == std::string::npos || t.shape.size() < 2) continue;
        if (!friday_prefill::gemmQ8Supported(friday_hpc::q8KindFromDtype(t.dtype), (int)t.shape[0])) return false;
    }
    return true;
}

bool Transformer::allocPrefillScratch(PrefillScratch& ps, int cap) {
    const size_t H = hidden_dim_, QD = (size_t)num_heads_ * head_dim_, KV = (size_t)num_kv_heads_ * head_dim_;
    const size_t I = max_intermediate_size_, XQ = std::max(std::max(H, QD), I);
    ps.cap = cap;
    bool ok = cudaMalloc((void**)&ps.x, cap * H * sizeof(half)) == cudaSuccess &&
              cudaMalloc((void**)&ps.q, cap * QD * sizeof(half)) == cudaSuccess &&
              cudaMalloc((void**)&ps.k, cap * KV * sizeof(half)) == cudaSuccess &&
              cudaMalloc((void**)&ps.v, cap * KV * sizeof(half)) == cudaSuccess &&
              cudaMalloc((void**)&ps.ctx, cap * QD * sizeof(half)) == cudaSuccess &&
              cudaMalloc((void**)&ps.gate, cap * I * sizeof(half)) == cudaSuccess &&
              cudaMalloc((void**)&ps.up, cap * I * sizeof(half)) == cudaSuccess &&
              cudaMalloc((void**)&ps.xq, cap * XQ) == cudaSuccess &&
              cudaMalloc((void**)&ps.xd, cap * (XQ / 32) * sizeof(float)) == cudaSuccess &&
              cudaMalloc((void**)&ps.positions, cap * sizeof(int)) == cudaSuccess &&
              cudaMallocHost((void**)&ps.h_positions, cap * sizeof(int)) == cudaSuccess;
    if (!ok) {
        cudaGetLastError();  // clear the allocation error
        freePrefillScratch(ps);
    }
    return ok;
}

void Transformer::freePrefillScratch(PrefillScratch& ps) {
    for (void* p : {(void*)ps.x, (void*)ps.q, (void*)ps.k, (void*)ps.v, (void*)ps.ctx, (void*)ps.gate, (void*)ps.up,
                    (void*)ps.xq, (void*)ps.xd, (void*)ps.positions})
        if (p) cudaFree(p);
    if (ps.h_positions) cudaFreeHost(ps.h_positions);
    ps = PrefillScratch();
}

// gpuLayerFused for T tokens at positions *d_pos_ .. *d_pos_ + T - 1 (rows of ps).
// Attention runs once per token with the decode kernel, so each query sees
// exactly the keys it would see in decode.
void Transformer::prefillLayerFused(int layer_idx, unsigned char* buf, PrefillScratch& ps, int T, cudaStream_t s) {
    const auto& li = layers_info_[layer_idx];
    const std::string pre = "blk." + std::to_string(layer_idx) + ".";
    auto ptr = [&](const char* n) -> unsigned char* {
        auto it = li.tensor_rel_offsets.find(pre + n);
        return it == li.tensor_rel_offsets.end() ? nullptr : buf + it->second;
    };
    auto info = [&](const char* n) -> const TensorInfo& { return full_tensor_map_.at(pre + n); };
    auto gemm = [&](const char* n, int K, half* y, int ldy, const float* bias, const half* res) {
        const TensorInfo& t = info(n);
        friday_prefill::gemmQ8(friday_hpc::q8KindFromDtype(t.dtype), ptr(n), ps.xq, ps.xd, T, K, (int)t.shape[1], y,
                               ldy, s, bias, res);
    };
    const int H = hidden_dim_, kv_dim = num_kv_heads_ * head_dim_, QD = num_heads_ * head_dim_;

    friday_hpc::rmsNormQuantQ8Rows(ps.x, (const float*)ptr("attn_norm.weight"), T, H, rms_norm_epsilon_, ps.xq, ps.xd,
                                   s);
    gemm("attn_q.weight", H, ps.q, QD, (const float*)ptr("attn_q.bias"), nullptr);
    gemm("attn_k.weight", H, ps.k, kv_dim, (const float*)ptr("attn_k.bias"), nullptr);
    gemm("attn_v.weight", H, ps.v, kv_dim, (const float*)ptr("attn_v.bias"), nullptr);
    if (ptr("attn_q_norm.weight") && ptr("attn_k_norm.weight")) {
        headRmsNormKernel<<<num_heads_ * T, head_dim_, head_dim_ * sizeof(float), s>>>(
            ps.q, (const float*)ptr("attn_q_norm.weight"), head_dim_, rms_norm_epsilon_);
        headRmsNormKernel<<<num_kv_heads_ * T, head_dim_, head_dim_ * sizeof(float), s>>>(
            ps.k, (const float*)ptr("attn_k_norm.weight"), head_dim_, rms_norm_epsilon_);
    }
    const size_t layer_elems = (size_t)layer_idx * max_ctx_ * kv_dim;
    const dim3 rope_grid((head_dim_ / 2 + 255) / 256, T);
    if (kv_q8_) {
        int8_t* kq = reinterpret_cast<int8_t*>(d_k_cache_) + layer_elems;
        int8_t* vq = reinterpret_cast<int8_t*>(d_v_cache_) + layer_elems;
        half* ks = d_k_scale_ + layer_elems / 32;
        half* vs = d_v_scale_ + layer_elems / 32;
        ropeStoreKvKernel<<<rope_grid, 256, 0, s>>>(ps.q, ps.k, ps.v, nullptr, nullptr, d_pos_, head_dim_, num_heads_,
                                                    num_kv_heads_, rope_freq_base_, rope_neox_ ? 1 : 0, d_rope_freqs_);
        const int warps = 8, nblk = 2 * kv_dim / 32;
        storeKvQ8Kernel<<<dim3((nblk + warps - 1) / warps, T), warps * 32, 0, s>>>(ps.k, ps.v, kq, ks, vq, vs, d_pos_,
                                                                                  kv_dim);
        for (int t = 0; t < T; ++t)
            friday_attn::launchDecodeAttentionSplit(ps.q + (size_t)t * QD, kq, vq, ps.ctx + (size_t)t * QD, 0,
                                                    ps.positions + t, max_ctx_, num_heads_, num_kv_heads_, head_dim_,
                                                    d_attn_ws_, s, ks, vs);
    } else {
        half* k_layer = d_k_cache_ + layer_elems;
        half* v_layer = d_v_cache_ + layer_elems;
        ropeStoreKvKernel<<<rope_grid, 256, 0, s>>>(ps.q, ps.k, ps.v, k_layer, v_layer, d_pos_, head_dim_, num_heads_,
                                                    num_kv_heads_, rope_freq_base_, rope_neox_ ? 1 : 0, d_rope_freqs_);
        for (int t = 0; t < T; ++t) {
            if (attn_split_)
                friday_attn::launchDecodeAttentionSplit(ps.q + (size_t)t * QD, k_layer, v_layer, ps.ctx + (size_t)t * QD,
                                                        0, ps.positions + t, max_ctx_, num_heads_, num_kv_heads_,
                                                        head_dim_, d_attn_ws_, s);
            else
                friday_attn::launchDecodeAttentionDevPos(ps.q + (size_t)t * QD, k_layer, v_layer,
                                                         ps.ctx + (size_t)t * QD, ps.positions + t, max_ctx_,
                                                         num_heads_, num_kv_heads_, head_dim_, s);
        }
    }
    friday_hpc::quantizeQ8Rows(ps.ctx, T, QD, ps.xq, ps.xd, s);
    gemm("attn_output.weight", QD, ps.x, H, nullptr, ps.x);

    friday_hpc::rmsNormQuantQ8Rows(ps.x, (const float*)ptr("ffn_norm.weight"), T, H, rms_norm_epsilon_, ps.xq, ps.xd,
                                   s);
    const int I = (int)info("ffn_gate.weight").shape[1];
    gemm("ffn_gate.weight", H, ps.gate, I, nullptr, nullptr);
    gemm("ffn_up.weight", H, ps.up, I, nullptr, nullptr);
    friday_hpc::siluMulQuantQ8Rows(ps.gate, ps.up, T, I, ps.xq, ps.xd, s);
    gemm("ffn_down.weight", I, ps.x, H, nullptr, ps.x);
    CUDA_CHECK(cudaGetLastError());
}

void Transformer::prefill(const int* tokens, int n, int pos0, half* d_x_last, half* d_output_logits, int vocab_size,
                          cudaStream_t s) {
    if (n <= 0) return;
    if (pos0 < 0 || pos0 + n > max_ctx_)
        throw std::runtime_error("context length exceeded: prefill of " + std::to_string(n) + " tokens at position " +
                                 std::to_string(pos0) + ", max context " + std::to_string(max_ctx_));
    const int H = hidden_dim_;
    PrefillScratch ps;
    int cap = std::min(n, 512);
    while (!allocPrefillScratch(ps, cap)) {
        if (cap <= 16) throw std::runtime_error("prefill: not enough GPU memory for the batch buffers");
        cap /= 2;
    }
    for (int c0 = 0; c0 < n; c0 += cap) {
        const int T = std::min(cap, n - c0), p0 = pos0 + c0;
        CUDA_CHECK(cudaStreamSynchronize(s));  // pinned h_pos_ / h_positions are free again
        for (int t = 0; t < T; ++t) {
            if (!getEmbedding(tokens[c0 + t], ps.x + (size_t)t * H, vocab_size, s))
                throw std::runtime_error("embedding failed for token " + std::to_string(tokens[c0 + t]));
            ps.h_positions[t] = p0 + t;
        }
        *h_pos_ = p0;
        CUDA_CHECK(cudaMemcpyAsync(d_pos_, h_pos_, sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(ps.positions, ps.h_positions, T * sizeof(int), cudaMemcpyHostToDevice, s));
        for (int l = 0; l < num_gpu_resident_layers_; ++l) prefillLayerFused(l, d_gpu_resident_layers_[l], ps, T, s);
        // CPU layers: token by token, as in decode.
        if (cpu_offload_ && num_gpu_resident_layers_ < num_layers_) {
            for (int t = 0; t < T; ++t) {
                half* row = ps.x + (size_t)t * H;
                CUDA_CHECK(cudaMemcpyAsync(h_act_pinned_, row, H * sizeof(half), cudaMemcpyDeviceToHost, s));
                CUDA_CHECK(cudaStreamSynchronize(s));
                float* x = cpu_runner_->hidden();
                const uint16_t* hb = reinterpret_cast<const uint16_t*>(h_act_pinned_);
                for (int i = 0; i < H; ++i) x[i] = friday_cpu::halfToFloat(hb[i]);
                cpu_runner_->run(p0 + t);
                uint16_t* ho = reinterpret_cast<uint16_t*>(h_act_pinned_);
                for (int i = 0; i < H; ++i) ho[i] = friday_cpu::floatToHalf(x[i]);
                CUDA_CHECK(cudaMemcpyAsync(row, h_act_pinned_, H * sizeof(half), cudaMemcpyHostToDevice, s));
                CUDA_CHECK(cudaStreamSynchronize(s));
            }
        }
        if (c0 + T == n)
            CUDA_CHECK(cudaMemcpyAsync(d_x_last, ps.x + (size_t)(T - 1) * H, H * sizeof(half),
                                       cudaMemcpyDeviceToDevice, s));
    }
    finalHead(d_x_last, d_output_logits, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    freePrefillScratch(ps);
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
    using PClock = std::chrono::steady_clock;
    const PClock::time_point prof_t0 = profile_ ? PClock::now() : PClock::time_point();
    PClock::time_point prof_t1 = prof_t0, prof_t2 = prof_t0;

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

    // Position for the device-side kernels (fused path), then the resident layers:
    // replayed from a CUDA Graph when possible (captured on the first call).
    if (fused_ok_) {
        if (pos_offset < 0 || pos_offset >= max_ctx_)
            throw std::runtime_error("context length exceeded: position " + std::to_string(pos_offset) +
                                     " >= max context " + std::to_string(max_ctx_));
        *h_pos_ = pos_offset;
        CUDA_CHECK(cudaMemcpyAsync(d_pos_, h_pos_, sizeof(int), cudaMemcpyHostToDevice, compute_stream));
    }
    bool resident_done = false;
    if (graph_ok_ && seq_len == 1 && layer_offset == 0 && num_gpu_resident_layers_ > 0) {
        if (graph_exec_ && (graph_stream_ != compute_stream || graph_input_ != d_input_activations)) {
            cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        if (!graph_exec_) {
            cudaGraph_t graph = nullptr;
            CUDA_CHECK(cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeThreadLocal));
            for (int l = 0; l < num_gpu_resident_layers_; ++l)
                gpuLayerFused(l, d_gpu_resident_layers_[l], d_input_activations, pos_offset, compute_stream);
            CUDA_CHECK(cudaStreamEndCapture(compute_stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            graph_stream_ = compute_stream;
            graph_input_ = d_input_activations;
        }
        CUDA_CHECK(cudaGraphLaunch(graph_exec_, compute_stream));
        resident_done = true;
    }

    // 1. GPU VRAM'de yerlesik (resident) katmanlar (0 .. num_gpu_resident_layers_ - 1)
    for (int layer_idx = 0; !resident_done && layer_idx < num_gpu_resident_layers_; ++layer_idx) {
        unsigned char* d_curr_buf = d_gpu_resident_layers_[layer_idx];
        const auto& curr_l_info = layers_info_[layer_idx];
        if (fused_ok_) {
            gpuLayerFused(layer_idx + layer_offset, d_curr_buf, d_input_activations, pos_offset, compute_stream);
            continue;
        }

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

    // 2a. CPU-offload mode: hand the hidden state to the CPU, run the remaining
    // layers there, and copy it back for the final norm and LM head.
    if (cpu_offload_ && num_gpu_resident_layers_ < num_layers_) {
        if (seq_len != 1) throw std::runtime_error("CPU layers: only single-token decode is supported");
        if (pos_offset < 0 || pos_offset >= max_ctx_) {
            throw std::runtime_error("context length exceeded: position " + std::to_string(pos_offset) +
                                     " >= max context " + std::to_string(max_ctx_));
        }
        CUDA_CHECK(cudaMemcpyAsync(h_act_pinned_, d_input_activations, hidden_dim_ * sizeof(half),
                                   cudaMemcpyDeviceToHost, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        if (profile_) prof_t1 = PClock::now();
        float* x = cpu_runner_->hidden();
        const uint16_t* hb = reinterpret_cast<const uint16_t*>(h_act_pinned_);
        for (int i = 0; i < hidden_dim_; ++i) x[i] = friday_cpu::halfToFloat(hb[i]);
        cpu_runner_->run(pos_offset);
        uint16_t* ho = reinterpret_cast<uint16_t*>(h_act_pinned_);
        for (int i = 0; i < hidden_dim_; ++i) ho[i] = friday_cpu::floatToHalf(x[i]);
        if (profile_) prof_t2 = PClock::now();
        CUDA_CHECK(cudaMemcpyAsync(d_input_activations, h_act_pinned_, hidden_dim_ * sizeof(half),
                                   cudaMemcpyHostToDevice, compute_stream));
    }

    // 2b. Stream mode: remaining layers streamed over PCIe (num_gpu_resident_layers_ .. num_layers_ - 1)
    if (!cpu_offload_ && num_gpu_resident_layers_ < num_layers_) {
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
                if (!readLayer(first_stream_layer + offload_idx, h_pinned_buffer_[slot])) {
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
            if (fused_ok_) {
                gpuLayerFused(layer_idx + layer_offset, d_curr_buf, d_input_activations, pos_offset, compute_stream);
                CUDA_CHECK(cudaEventRecord(ev_slot_free_[curr_slot], compute_stream));
                continue;
            }

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
    finalHead(d_input_activations, d_output_logits, compute_stream);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    if (profile_) {
        const PClock::time_point t3 = PClock::now();
        if (prof_t1 == prof_t0) prof_t1 = prof_t2 = t3;  // no CPU layers: all GPU
        auto ms = [](PClock::time_point a, PClock::time_point b) {
            return std::chrono::duration<float, std::milli>(b - a).count();
        };
        prof_gpu_ms_.push_back(ms(prof_t0, prof_t1));
        prof_cpu_ms_.push_back(ms(prof_t1, prof_t2));
        prof_tail_ms_.push_back(ms(prof_t2, t3));
    }
}
