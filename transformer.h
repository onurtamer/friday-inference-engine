#ifndef FRIDAY_TRANSFORMER_H
#define FRIDAY_TRANSFORMER_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <vector>
#include <string> // std::string için
#include <nlohmann/json.hpp> // JSON ayrıştırma için

// CUDA_CHECK makrosu
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                \
    do {                                                \
        cudaError_t err = call;                         \
        if (err != cudaSuccess) {                       \
            fprintf(stderr, "CUDA error at %s:%d: %s: %s\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)
#endif

// Tensor bilgilerini tutacak yapı
struct TensorInfo {
    std::string name;
    size_t absolute_offset;
    size_t size_bytes;
    std::vector<long long> shape;
    std::string dtype; // GGUF Dtype, örn: Q4_0, F16

    TensorInfo() : name(""), absolute_offset(0), size_bytes(0), shape({}), dtype("") {}
};

// AsyncIOManager sınıfının forward deklarasyonu
class AsyncIOManager; 
namespace friday_cpu { class ThreadPool; }
enum QuantType; // kernel_dispatcher.h'den

// Katman yapısını ve tensör bağıl ofsetlerini tutan yapı (Zero cudaMalloc mimarisi)
struct LayerInfo {
    size_t start_offset = 0;
    size_t size_bytes = 0;
    std::map<std::string, size_t> tensor_rel_offsets;
};

// Dequantize one embedding row (Q4_0, Q2_K, Q3_K, Q4_K, Q6_K or F16) on the GPU.
// Returns false for unsupported types or launch errors.
bool launchDequantizeRow(const std::string& dtype, const uint8_t* d_row, half* d_out, int dim, cudaStream_t s);
size_t embeddingRowBytes(const std::string& dtype, int dim);  // 0 if unsupported

class Transformer {
public:
    Transformer(
        const std::string& model_file_path,
        const std::string& tensor_map_path,
        int model_hidden_dim,
        int model_num_heads,
        int model_num_kv_heads,
        int model_num_layers,
        float model_rms_norm_epsilon,
        QuantType model_quant_type,
        AsyncIOManager* async_io_manager_ptr,
        double vram_limit_gb = 0.0,
        int max_context = 2048,
        int cpu_threads = 0
    );

    ~Transformer();

    int getMaxContext() const { return max_ctx_; }
    int getNumGpuLayers() const { return num_gpu_resident_layers_; }
    int getNumCpuLayers() const { return cpu_offload_ ? num_layers_ - num_gpu_resident_layers_ : 0; }

    // RMSNorm (weights are float32 in GGUF)
    void applyRMSNorm(half* d_input, const float* d_weight, half* d_output, int size, cudaStream_t stream);

    // RoPE
    void applyRoPE(half* d_query, half* d_key, int head_dim, int seq_len, int pos_offset, cudaStream_t stream);

    // Self-Attention (Double Buffering d_layer_buf üzerinden)
    void applySelfAttention(
        int layer_idx,
        unsigned char* d_layer_buf,
        half* d_input_activations,
        half* d_query_out,
        half* d_key_out,
        half* d_value_out,
        half* d_attention_output,
        int seq_len,
        int pos_offset,
        cudaStream_t stream);

    // MoE Router veya Dense MLP katmanı (Double Buffering d_layer_buf üzerinden)
    void forwardMLPLayer(
        int layer_idx,
        unsigned char* d_layer_buf,
        half* d_input_activations,
        half* d_output_activations,
        cudaStream_t stream
    );

    // GeLU Aktivasyonu
    void applyGeLU(half* d_activations, int size, cudaStream_t stream);

    // Ana Forward Pass: Ping-Pong Double Buffering mimarisi ile
    void forward(
        half* d_input_activations,
        half* d_output_logits,
        int seq_len,
        int layer_offset = 0,
        int pos_offset = 0,
        cudaStream_t stream = nullptr
    );

    // Token ID'den embedding vektörünü alır
    bool getEmbedding(int token_id, half* d_embedding, int vocab_size, cudaStream_t stream);

    int getHiddenDim() const { return hidden_dim_; }
    int getVocabSize() const {
        auto it = full_tensor_map_.find("output.weight");
        if (it != full_tensor_map_.end() && it->second.shape.size() == 2) {
            return (int)it->second.shape[1];
        }
        return 0;
    }

    std::map<std::string, TensorInfo> full_tensor_map_;

private:
    std::string model_file_path_;
    int hidden_dim_;
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int num_layers_;
    float rms_norm_epsilon_;
    QuantType model_quant_type_;

    AsyncIOManager* async_io_manager_ptr_;

    // KV cache on the GPU, fp16, layout [layer][max_ctx][num_kv_heads * head_dim]
    // for K and V. Allocated once in the constructor (counted in the VRAM budget).
    int max_ctx_ = 2048;
    half* d_k_cache_ = nullptr;
    half* d_v_cache_ = nullptr;
    half* d_attn_ctx_ = nullptr;  // attention output, input of the O projection

    // Layers that do not fit in VRAM. Default: computed on the CPU from host RAM
    // (host memory bandwidth is well above PCIe on typical desktops).
    // FRIDAY_OFFLOAD=stream selects the previous mode: stream the layer weights
    // over PCIe into DMA slots every token and compute on the GPU.
    bool cpu_offload_ = true;
    std::vector<std::unique_ptr<unsigned char[]>> h_cpu_layers_;  // [layer - num_gpu_resident_layers_]
    std::unique_ptr<friday_cpu::ThreadPool> cpu_pool_;
    std::vector<float> cpu_k_cache_, cpu_v_cache_;  // [cpu layer][max_ctx][kv_dim], fp16-rounded values
    half* h_act_pinned_ = nullptr;                   // hidden state staging for GPU <-> CPU hand-off
    std::vector<float> cpu_x_, cpu_norm_, cpu_q_, cpu_k_, cpu_v_, cpu_attn_, cpu_o_, cpu_gate_, cpu_up_, cpu_down_;
    void forwardCpuLayers(int pos);
    void cpuLayer(int layer_idx, int pos);

    // ============================================================
    // V32: Esnek Hibrit Bellek (Elastic Tiered Memory)
    // ============================================================
    double vram_limit_gb_ = 0.0;
    int num_gpu_resident_layers_ = 0;
    std::vector<unsigned char*> d_gpu_resident_layers_;
    std::vector<unsigned char*> h_offload_pinned_cache_;

    // DMA slots for host-streamed layers (triple buffering).
    static constexpr int kDmaSlotCount = 3;
    unsigned char* d_layer_buffer_[kDmaSlotCount] = {nullptr, nullptr, nullptr};
    // Pinned staging buffers, used only when a layer is not in the pinned host cache.
    unsigned char* h_pinned_buffer_[kDmaSlotCount] = {nullptr, nullptr, nullptr};
    // ev_slot_free_[i]: compute has finished reading slot i.
    // ev_slot_ready_[i]: the copy into slot i has finished.
    cudaEvent_t ev_slot_free_[kDmaSlotCount] = {};
    cudaEvent_t ev_slot_ready_[kDmaSlotCount] = {};
    bool slot_events_created_ = false;

    size_t layer_buffer_size_ = 0;
    std::vector<LayerInfo> layers_info_;

    // Ping-Pong Asenkron Akışları
    cudaStream_t stream_compute_ = nullptr; // Stream 1: GPU Hesaplama
    cudaStream_t stream_io_ = nullptr;      // Stream 2: NVMe/PCIe Prefetch

    // Katman dışı statik tensörler (Emb & LM Head)
    unsigned char* d_token_embd_ = nullptr;          // device buffer for one staged embedding row
    std::unique_ptr<unsigned char[]> h_token_embd_;  // full embedding table in host RAM
    unsigned char* h_emb_row_pinned_ = nullptr;     // pinned staging for one row
    unsigned char* d_output_norm_ = nullptr;
    unsigned char* d_lm_head_ = nullptr;

    // Temporary buffers for forward pass to avoid cudaMalloc/cudaFree per token
    half *d_query_out_ = nullptr, *d_key_out_ = nullptr, *d_value_out_ = nullptr;
    half *d_attention_output_ = nullptr, *d_mlp_output_ = nullptr, *d_norm_output_ = nullptr;
    half *d_gate_out_ = nullptr, *d_up_out_ = nullptr;
    int max_intermediate_size_ = 0;

    void loadTensorMap(const std::string& map_path);
    QuantType getQuantTypeFromDtype(const std::string& dtype_str);
};

#endif // FRIDAY_TRANSFORMER_H


