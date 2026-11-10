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

// Same kernel; the sequence length is *d_pos + 1, read on the device (so the
// launch can be captured in a CUDA Graph). Shared memory is sized for max_seq_len.
void launchDecodeAttentionDevPos(const half* q, const half* k_cache, const half* v_cache, half* out, const int* d_pos,
                                 int max_seq_len, int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream);
bool decodeAttentionFitsSharedMemory(int max_seq_len);

// Split-K (flash-decoding) variant: positions in chunks of 64, one block per
// (head, chunk), then a combine kernel. Sequence length = d_pos ? *d_pos + 1 :
// seq_len. `workspace` must hold splitAttentionWorkspaceBytes(num_heads,
// max_seq_len) bytes. No shared-memory limit on the context length.
size_t splitAttentionWorkspaceBytes(int num_heads, int max_seq_len);
// k_scale/v_scale non-null: k_cache/v_cache are int8 with one fp16 scale per 32
// values (layout [pos][kv_dim] and [pos][kv_dim / 32]); otherwise fp16.
void launchDecodeAttentionSplit(const half* q, const void* k_cache, const void* v_cache, half* out, int seq_len,
                                const int* d_pos, int max_seq_len, int num_heads, int num_kv_heads, int head_dim,
                                void* workspace, cudaStream_t stream, const half* k_scale = nullptr,
                                const half* v_scale = nullptr);

// Causal attention for T consecutive queries (prefill): query t is at position
// *d_pos0 + t and attends to positions 0 .. *d_pos0 + t of the cache (which
// already holds the T new keys/values). q and out are [T][num_heads * head_dim].
// Same per-score rules as the decode kernels (NaN score -> -1e30, exp cut off
// below -50 relative to the running max, output clamped, NaN -> 0); only the
// order of the fp32 additions differs. One block per (kv head, 8 queries).
// Needs head_dim == 128 and num_heads / num_kv_heads <= 8.
bool prefillAttentionSupported(int num_heads, int num_kv_heads, int head_dim);
void launchPrefillAttention(const half* q, const void* k_cache, const void* v_cache, half* out, int T,
                            const int* d_pos0, int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream,
                            const half* k_scale = nullptr, const half* v_scale = nullptr);

}  // namespace friday_attn

#endif  // FRIDAY_ATTENTION_KERNELS_H
