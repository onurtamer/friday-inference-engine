#ifndef FRIDAY_TRANSFORMER_H
#define FRIDAY_TRANSFORMER_H

#include <cuda_fp16.h> // For half type
#include <cuda_runtime.h> // For cudaStream_t, device pointers
#include <map>
#include <tuple>
#include <vector> // For std::vector in expert_file_map, though map is specified.

// Define CUDA_CHECK macro for error checking
#define CUDA_CHECK(call)                                \
    do {                                                \
        cudaError_t err = call;                         \
        if (err != cudaSuccess) {                       \
            fprintf(stderr, "CUDA error at %s:%d: %s: %s\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)

// --- Asynchronous I/O Manager Stub ---
// In a real system, this would be a full-fledged class managing device memory,
// host staging buffers, and asynchronous transfers.
class AsyncIOManager {
public:
    AsyncIOManager() = default;
    ~AsyncIOManager() = default;

    // A placeholder to simulate an async read request for expert weights from SSD.
    // In a real scenario, this would likely take a destination device pointer,
    // and return a future/event to track completion.
    // For now, it just prints a message.
    void enqueueReadExpertWeights(
        int expert_id,
        size_t file_offset,
        size_t size_bytes,
        unsigned char* d_destination_ptr, // Device memory where packed weights will be read into
        cudaStream_t stream) {
        // Simulate enqueueing an I/O request
        printf("AsyncIOManager: Enqueueing read for expert %d. Offset: %zu, Size: %zu bytes. Destination address: %p\n",
               expert_id, file_offset, size_bytes, d_destination_ptr);
        // In a real implementation, this would involve:
        // 1. Allocating host staging buffer (if not pre-allocated)
        // 2. Issuing async read from SSD to host buffer
        // 3. Issuing async cudaMemcpyAsync from host buffer to d_destination_ptr
        // 4. Recording an event to track completion.
    }
};

class Transformer {
public:
    // Constructor takes device pointers to model weights and dimensions.
    // Ownership of pointers is assumed to be external for simplicity,
    // or handled by an `init` method.
    Transformer(
        int hidden_dim,
        int num_heads,
        int num_experts,
        int top_k_experts,
        float rms_norm_epsilon,
        // QKV projection weights (FP16)
        const half* d_Wq, const half* d_Wk, const half* d_Wv,
        // MoE Router weights (FP16)
        const half* d_router_weights,
        // Pointer to AsyncIOManager
        AsyncIOManager* async_io_manager_ptr
    );

    // RMSNorm: input -> output
    void applyRMSNorm(
        half* d_input,
        const half* d_weight, // LayerNorm weight (gamma)
        half* d_output,
        int size,
        cudaStream_t stream);

    // RoPE: Applies rotary positional embeddings to Q and K vectors
    void applyGeLU(half* d_activations, int size, cudaStream_t stream);

    void applyRoPE(
        half* d_query,
        half* d_key,
        int head_dim,
        int seq_len,
        int pos_offset,
        cudaStream_t stream);

    // Self-Attention: Performs QKV projections and subsequent attention logic
    void applySelfAttention(
        half* d_input_activations, // input to QKV projections (1 x hidden_dim_)
        half* d_query_out,         // device buffer for Q (1 x hidden_dim_)
        half* d_key_out,           // device buffer for K (1 x hidden_dim_)
        half* d_value_out,         // device buffer for V (1 x hidden_dim_)
        // Output attention context vector, typically after softmax(QK^T)V
        half* d_attention_output,  // (1 x hidden_dim_)
        int seq_len,
        int pos_offset,
        cudaStream_t stream);

    // MoE Router: Computes router logits, selects top-K experts, and triggers async I/O.
    void routeMixtureOfExperts(
        half* d_input_activations, // Input to the MoE router (FP16, 1 x hidden_dim_)
        // Buffers for storing selected expert data (e.g., intermediate activations)
        half* d_expert_input_buffer,   // (1 x expert_hidden_dim_)
        half* d_expert_output_buffer,  // (1 x expert_hidden_dim_)
        // For the purpose of this stub, we'll assume the loaded expert weights
        // will be put into d_expert_weights_buffer_ptr. This needs to be unsigned char* for packed weights.
        unsigned char* d_expert_weights_buffer_ptr, // Placeholder for where async-loaded *packed* weights go
        // Expert map (host-side) needed to get file offsets
        const std::map<int, std::tuple<size_t, size_t, std::vector<int>>>& expert_file_map, // expert_id -> (offset, size, original_shape)
        cudaStream_t stream);


private:
    int hidden_dim_;
    int num_heads_;
    int head_dim_; // hidden_dim / num_heads
    int num_experts_;
    int top_k_experts_;
    float rms_norm_epsilon_;

    // Device pointers to model weights (owned externally or managed via init/destroy)
    const half* d_Wq_;
    const half* d_Wk_;
    const half* d_Wv_;
    const half* d_router_weights_; // FP16 weights for router

    AsyncIOManager* async_io_manager_ptr_; // Pointer to the async I/O manager
};

#endif // FRIDAY_TRANSFORMER_H
