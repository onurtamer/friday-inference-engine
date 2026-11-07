#include "friday/runtime/paged_attention.h"
#include <cmath>
#include <cstdint>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace friday {
namespace runtime {

// Update Paged KV Cache with new key/value at current position
__global__ void updatePagedKvCacheKernel(
    const void* const* __restrict__ d_k_block_ptrs,
    const void* const* __restrict__ d_v_block_ptrs,
    const int* __restrict__ d_block_table,
    const half* __restrict__ d_key_in,
    const half* __restrict__ d_value_in,
    int pos,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    int kv_head_idx = blockIdx.x;
    int d = threadIdx.x;
    if (kv_head_idx >= num_kv_heads || d >= head_dim) return;

    int logical_block_idx = pos / block_size;
    int in_block_offset = pos % block_size;
    int phys_block_id = d_block_table[logical_block_idx];

    half* k_block = (half*)d_k_block_ptrs[phys_block_id];
    half* v_block = (half*)d_v_block_ptrs[phys_block_id];

    int offset = in_block_offset * (num_kv_heads * head_dim) + kv_head_idx * head_dim + d;
    int in_offset = kv_head_idx * head_dim + d;

    k_block[offset] = d_key_in[in_offset];
    v_block[offset] = d_value_in[in_offset];
}

// Paged Attention Kernel: Fused Softmax + Value Reduction reading across block table
__global__ void pagedFusedAttentionKernel(
    const half* __restrict__ d_query,
    const void* const* __restrict__ d_k_block_ptrs,
    const void* const* __restrict__ d_v_block_ptrs,
    const int* __restrict__ d_block_table,
    half* __restrict__ d_attention_output,
    int current_pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    int head_idx = blockIdx.x;
    if (head_idx >= num_heads) return;

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int thread_id = threadIdx.x;

    extern __shared__ float s_scores[];
    float scale = 1.0f / sqrtf((float)head_dim);

    float score = -1e30f;
    if (thread_id <= current_pos) {
        int log_b = thread_id / block_size;
        int in_b = thread_id % block_size;
        int phys_b = d_block_table[log_b];

        const half* k_block = (const half*)d_k_block_ptrs[phys_b];
        int k_offset = in_b * (num_kv_heads * head_dim) + kv_head_idx * head_dim;

        float sum = 0.0f;
        int q_offset = head_idx * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            float q = __half2float(d_query[q_offset + d]);
            float k = __half2float(k_block[k_offset + d]);
            sum += q * k;
        }
        score = sum * scale;
        if (isnan(score)) score = -1e30f;
    }
    s_scores[thread_id] = score;
    __syncthreads();

    // Reduction for Softmax on thread 0
    if (thread_id == 0) {
        float max_s = -1e30f;
        for (int i = 0; i <= current_pos; ++i) {
            if (s_scores[i] > max_s) max_s = s_scores[i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i <= current_pos; ++i) {
            float diff = s_scores[i] - max_s;
            float e = (diff < -50.0f) ? 0.0f : expf(diff);
            s_scores[i] = e;
            sum_exp += e;
        }
        for (int i = 0; i <= current_pos; ++i) {
            s_scores[i] /= (sum_exp + 1e-12f);
        }
    }
    __syncthreads();

    // Value weighted sum across historical tokens
    if (thread_id < head_dim) {
        float out_val = 0.0f;
        for (int p = 0; p <= current_pos; ++p) {
            float s = s_scores[p];
            int log_b = p / block_size;
            int in_b = p % block_size;
            int phys_b = d_block_table[log_b];

            const half* v_block = (const half*)d_v_block_ptrs[phys_b];
            int v_offset = in_b * (num_kv_heads * head_dim) + kv_head_idx * head_dim;
            float v = __half2float(v_block[v_offset + thread_id]);
            out_val += s * v;
        }
        if (isnan(out_val)) out_val = 0.0f;
        if (out_val > 65504.0f) out_val = 65504.0f;
        if (out_val < -65504.0f) out_val = -65504.0f;
        d_attention_output[head_idx * head_dim + thread_id] = __float2half(out_val);
    }
}

void launchPagedFusedAttention(
    const half* d_query,
    const void* const* d_k_block_ptrs,
    const void* const* d_v_block_ptrs,
    const int* d_block_table,
    const half* d_key_in,
    const half* d_value_in,
    half* d_attention_output,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    cudaStream_t stream
) {
    // 1. Update KV Cache at current position
    updatePagedKvCacheKernel<<<num_kv_heads, head_dim, 0, stream>>>(
        d_k_block_ptrs,
        d_v_block_ptrs,
        d_block_table,
        d_key_in,
        d_value_in,
        pos,
        num_kv_heads,
        head_dim,
        block_size
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. Fused Paged Attention
    int block_threads = (pos + 1 < 1024) ? ((pos + 1 > head_dim) ? pos + 1 : head_dim) : 1024;
    dim3 grid_attn(num_heads);
    size_t shared_mem = block_threads * sizeof(float);

    pagedFusedAttentionKernel<<<grid_attn, block_threads, shared_mem, stream>>>(
        d_query,
        d_k_block_ptrs,
        d_v_block_ptrs,
        d_block_table,
        d_attention_output,
        pos,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace runtime
} // namespace friday
