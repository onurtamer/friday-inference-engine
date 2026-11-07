#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace friday {
namespace runtime {

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
    cudaStream_t stream = 0
);

} // namespace runtime
} // namespace friday
