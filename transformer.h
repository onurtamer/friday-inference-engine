#ifndef FRIDAY_TRANSFORMER_H
#define FRIDAY_TRANSFORMER_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <map>
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
enum QuantType; // kernel_dispatcher.h'den

// Katman yapısını ve tensör bağıl ofsetlerini tutan yapı (Zero cudaMalloc mimarisi)
struct LayerInfo {
    size_t start_offset = 0;
    size_t size_bytes = 0;
    std::map<std::string, size_t> tensor_rel_offsets;
};

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
        double vram_limit_gb = 0.0
    );

    ~Transformer();

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

    // KV-Cache storage (Host tarafında)
    std::vector<std::vector<std::vector<half>>> k_cache_;
    std::vector<std::vector<std::vector<half>>> v_cache_;

    // ============================================================
    // V32: Esnek Hibrit Bellek (Elastic Tiered Memory)
    // ============================================================
    double vram_limit_gb_ = 0.0;
    int num_gpu_resident_layers_ = 0;
    std::vector<unsigned char*> d_gpu_resident_layers_;
    std::vector<unsigned char*> h_offload_pinned_cache_;

    // Sadece akitilan (stream edilen) katmanlar icin Ping-Pong tamponlari
    unsigned char* d_layer_buffer_A_ = nullptr;
    unsigned char* d_layer_buffer_B_ = nullptr;

    // Gerçek asenkron DMA transferleri için Pinned Host Buffer'lar
    unsigned char* h_pinned_buffer_A_ = nullptr;
    unsigned char* h_pinned_buffer_B_ = nullptr;

    size_t layer_buffer_size_ = 0;
    std::vector<LayerInfo> layers_info_;

    // Ping-Pong Asenkron Akışları
    cudaStream_t stream_compute_ = nullptr; // Stream 1: GPU Hesaplama
    cudaStream_t stream_io_ = nullptr;      // Stream 2: NVMe/PCIe Prefetch

    // KV Cache VRAM Rezerv Havuzu (4-5 GB VRAM Kilidi)
    void* d_kv_cache_pool_ = nullptr;
    size_t kv_pool_size_bytes_ = 0;

    // Katman dışı statik tensörler (Emb & LM Head)
    unsigned char* d_token_embd_ = nullptr;
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


