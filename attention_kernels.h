#ifndef FRIDAY_ATTENTION_KERNELS_H
#define FRIDAY_ATTENTION_KERNELS_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace friday_attn {

// Single-query (decode) grouped-query attention over a contiguous KV cache.
//   q        [num_heads * head_dim]                 (RoPE already applied)
//   k_cache  [max_ctx][num_kv_heads * head_dim]      (one layer)
//   v_cache  [max_ctx][num_kv_heads * head_dim]      (one layer)
//   out      [num_heads * head_dim]
// Attends over positions 0..seq_len-1. head_dim must be 128.
// Math matches the previous CPU implementation: fp32 dot products scaled by
// 1/sqrt(head_dim), NaN scores -> -1e30, exp underflow cut at -50, fp32
// weighted sum, output clamped to the fp16 range with NaN -> 0.
void launchDecodeAttention(const half* q, const half* k_cache, const half* v_cache, half* out,
                           int seq_len, int num_heads, int num_kv_heads, int head_dim,
                           cudaStream_t stream);

}  // namespace friday_attn

#endif  // FRIDAY_ATTENTION_KERNELS_H
