#include "attention_kernels.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace friday_attn {
namespace {

constexpr int kHeadDim = 128;
constexpr int kThreads = 128;          // one thread per output dimension
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float warpSum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warpMax(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

// One block per query head. Shared memory: q (128 floats), reduction scratch,
// then seq_len scores.
__global__ __launch_bounds__(kThreads)
void decodeAttentionKernel(const half* __restrict__ q, const half* __restrict__ k_cache,
                           const half* __restrict__ v_cache, half* __restrict__ out,
                           int seq_len, int num_heads, int num_kv_heads) {
    extern __shared__ float smem[];
    float* s_q = smem;                    // [128]
    float* s_red = smem + kHeadDim;       // [kWarps]
    float* s_score = s_red + kWarps;      // [seq_len]

    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int kv_dim = num_kv_heads * kHeadDim;
    const float scale = rsqrtf((float)kHeadDim);

    s_q[tid] = __half2float(q[head * kHeadDim + tid]);
    __syncthreads();

    // Scores: each warp takes positions p = warp, warp + kWarps, ...;
    // each lane covers 4 of the 128 dimensions.
    for (int p = warp; p < seq_len; p += kWarps) {
        const half* k = k_cache + (size_t)p * kv_dim + kv_head * kHeadDim;
        float part = 0.0f;
        #pragma unroll
        for (int i = 0; i < kHeadDim / 32; ++i) {
            const int d = lane + 32 * i;
            part += s_q[d] * __half2float(k[d]);
        }
        part = warpSum(part);
        if (lane == 0) {
            float score = part * scale;
            if (isnan(score)) score = -1e30f;
            s_score[p] = score;
        }
    }
    __syncthreads();

    // Max over positions.
    float m = -1e30f;
    for (int p = tid; p < seq_len; p += kThreads) m = fmaxf(m, s_score[p]);
    m = warpMax(m);
    if (lane == 0) s_red[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = s_red[0];
        for (int w = 1; w < kWarps; ++w) mm = fmaxf(mm, s_red[w]);
        s_red[0] = mm;
    }
    __syncthreads();
    const float max_score = s_red[0];
    __syncthreads();

    // Exponentials and their sum.
    float sum = 0.0f;
    for (int p = tid; p < seq_len; p += kThreads) {
        const float diff = s_score[p] - max_score;
        const float e = (diff < -50.0f) ? 0.0f : expf(diff);
        s_score[p] = e;
        sum += e;
    }
    sum = warpSum(sum);
    if (lane == 0) s_red[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (int w = 0; w < kWarps; ++w) ss += s_red[w];
        s_red[0] = ss;
    }
    __syncthreads();
    const float total = s_red[0];
    const float inv_sum = (total > 1e-12f) ? (1.0f / total) : 0.0f;

    // Weighted sum of V: thread tid owns output dimension tid (coalesced V reads).
    float acc = 0.0f;
    const half* v = v_cache + kv_head * kHeadDim + tid;
    for (int p = 0; p < seq_len; ++p) acc += (s_score[p] * inv_sum) * __half2float(v[(size_t)p * kv_dim]);
    if (isnan(acc)) acc = 0.0f;
    acc = fminf(fmaxf(acc, -65504.0f), 65504.0f);
    out[head * kHeadDim + tid] = __float2half(acc);
}

}  // namespace

void launchDecodeAttention(const half* q, const half* k_cache, const half* v_cache, half* out,
                           int seq_len, int num_heads, int num_kv_heads, int head_dim,
                           cudaStream_t stream) {
    if (head_dim != kHeadDim) throw std::runtime_error("launchDecodeAttention: head_dim must be 128");
    if (seq_len <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0)
        throw std::runtime_error("launchDecodeAttention: invalid shape");
    const size_t smem = (size_t)(kHeadDim + kWarps + seq_len) * sizeof(float);
    if (smem > 48 * 1024) throw std::runtime_error("launchDecodeAttention: seq_len exceeds shared-memory limit");
    decodeAttentionKernel<<<num_heads, kThreads, smem, stream>>>(q, k_cache, v_cache, out, seq_len, num_heads,
                                                                 num_kv_heads);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launchDecodeAttention: launch failed: ") + cudaGetErrorString(err));
}

}  // namespace friday_attn
