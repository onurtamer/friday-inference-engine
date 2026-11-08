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
                           int seq_len, int num_heads, int num_kv_heads, const int* __restrict__ d_pos) {
    if (d_pos) seq_len = *d_pos + 1;  // graph-replayable: position read on the device
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


// ---------------------------------------------------------------------------
// Split-K decode attention (flash-decoding): grid (head, split), each block
// handles kSplitChunk positions with a local max and an unnormalized V sum;
// a second kernel rescales the partials to the global max and normalizes.
// Same per-score rules as decodeAttentionKernel (NaN score -> -1e30, exp
// cut off below -50 relative to the block max, NaN output -> 0, fp16 clamp);
// the summation order differs.
// ---------------------------------------------------------------------------
constexpr int kSplitChunk = 256;

// KV element load: fp16, or int8 with one fp16 scale per 32 values (Q8 = true).
template <bool Q8>
__device__ __forceinline__ float kvLoad(const void* cache, const half* scale, size_t idx) {
    if (Q8) return (float)static_cast<const int8_t*>(cache)[idx] * __half2float(scale[idx / 32]);
    return __half2float(static_cast<const half*>(cache)[idx]);
}

template <bool Q8>
__global__ __launch_bounds__(kThreads)
void decodeAttentionSplitKernel(const half* __restrict__ q, const void* __restrict__ k_cache,
                                const void* __restrict__ v_cache, const half* __restrict__ k_scale,
                                const half* __restrict__ v_scale, float* __restrict__ part_o,
                                float* __restrict__ part_ml, int seq_len, int num_heads, int num_kv_heads,
                                int max_splits, const int* __restrict__ d_pos) {
    __shared__ float s_q[kHeadDim];
    __shared__ float s_red[kWarps];
    __shared__ float s_score[kSplitChunk];
    if (d_pos) seq_len = *d_pos + 1;
    const int head = blockIdx.x, split = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int start = split * kSplitChunk;
    if (start >= seq_len) return;  // inactive split (the combine kernel ignores it)
    const int n = min(kSplitChunk, seq_len - start);
    const int kv_head = head / (num_heads / num_kv_heads);
    const int kv_dim = num_kv_heads * kHeadDim;
    const float scale = rsqrtf((float)kHeadDim);

    s_q[tid] = __half2float(q[head * kHeadDim + tid]);
    __syncthreads();
    for (int i = warp; i < n; i += kWarps) {
        const size_t kbase = (size_t)(start + i) * kv_dim + kv_head * kHeadDim;
        float part = 0.0f;
        #pragma unroll
        for (int j = 0; j < kHeadDim / 32; ++j) {
            const int d = lane + 32 * j;
            part += s_q[d] * kvLoad<Q8>(k_cache, k_scale, kbase + d);
        }
        part = warpSum(part);
        if (lane == 0) {
            float score = part * scale;
            if (isnan(score)) score = -1e30f;
            s_score[i] = score;
        }
    }
    __syncthreads();
    float m = -1e30f;
    for (int i = tid; i < n; i += kThreads) m = fmaxf(m, s_score[i]);
    m = warpMax(m);
    if (lane == 0) s_red[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = s_red[0];
        for (int w = 1; w < kWarps; ++w) mm = fmaxf(mm, s_red[w]);
        s_red[0] = mm;
    }
    __syncthreads();
    const float block_max = s_red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = tid; i < n; i += kThreads) {
        const float diff = s_score[i] - block_max;
        const float e = (diff < -50.0f) ? 0.0f : expf(diff);
        s_score[i] = e;
        sum += e;
    }
    sum = warpSum(sum);
    if (lane == 0) s_red[warp] = sum;
    __syncthreads();
    float acc = 0.0f;
    const size_t vbase = (size_t)start * kv_dim + kv_head * kHeadDim + tid;
    for (int i = 0; i < n; ++i) acc += s_score[i] * kvLoad<Q8>(v_cache, v_scale, vbase + (size_t)i * kv_dim);
    const size_t slot = (size_t)head * max_splits + split;
    part_o[slot * kHeadDim + tid] = acc;
    if (tid == 0) {
        float ss = 0.0f;
        for (int w = 0; w < kWarps; ++w) ss += s_red[w];
        part_ml[slot * 2] = block_max;
        part_ml[slot * 2 + 1] = ss;
    }
}

__global__ __launch_bounds__(kThreads)
void decodeAttentionCombineKernel(const float* __restrict__ part_o, const float* __restrict__ part_ml,
                                  half* __restrict__ out, int seq_len, int max_splits, const int* __restrict__ d_pos) {
    if (d_pos) seq_len = *d_pos + 1;
    const int head = blockIdx.x, tid = threadIdx.x;
    const int nsp = (seq_len + kSplitChunk - 1) / kSplitChunk;
    const size_t base = (size_t)head * max_splits;
    float M = -1e30f;
    for (int s = 0; s < nsp; ++s) M = fmaxf(M, part_ml[(base + s) * 2]);
    float L = 0.0f, acc = 0.0f;
    for (int s = 0; s < nsp; ++s) {
        const float diff = part_ml[(base + s) * 2] - M;
        const float w = (diff < -50.0f) ? 0.0f : expf(diff);
        L += w * part_ml[(base + s) * 2 + 1];
        acc += w * part_o[(base + s) * kHeadDim + tid];
    }
    float r = (L > 1e-12f) ? acc / L : 0.0f;
    if (isnan(r)) r = 0.0f;
    r = fminf(fmaxf(r, -65504.0f), 65504.0f);
    out[head * kHeadDim + tid] = __float2half(r);
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
                                                                 num_kv_heads, nullptr);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launchDecodeAttention: launch failed: ") + cudaGetErrorString(err));
}

void launchDecodeAttentionDevPos(const half* q, const half* k_cache, const half* v_cache, half* out, const int* d_pos,
                                 int max_seq_len, int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream) {
    if (head_dim != kHeadDim) throw std::runtime_error("launchDecodeAttention: head_dim must be 128");
    if (max_seq_len <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0)
        throw std::runtime_error("launchDecodeAttention: invalid shape");
    const size_t smem = (size_t)(kHeadDim + kWarps + max_seq_len) * sizeof(float);
    if (smem > 48 * 1024) throw std::runtime_error("launchDecodeAttention: max context exceeds shared-memory limit");
    decodeAttentionKernel<<<num_heads, kThreads, smem, stream>>>(q, k_cache, v_cache, out, max_seq_len, num_heads,
                                                                 num_kv_heads, d_pos);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launchDecodeAttention: launch failed: ") + cudaGetErrorString(err));
}

size_t splitAttentionWorkspaceBytes(int num_heads, int max_seq_len) {
    const size_t splits = (size_t)(max_seq_len + kSplitChunk - 1) / kSplitChunk;
    return (size_t)num_heads * splits * (kHeadDim + 2) * sizeof(float);
}

void launchDecodeAttentionSplit(const half* q, const void* k_cache, const void* v_cache, half* out, int seq_len,
                                const int* d_pos, int max_seq_len, int num_heads, int num_kv_heads, int head_dim,
                                void* workspace, cudaStream_t stream, const half* k_scale, const half* v_scale) {
    if (head_dim != kHeadDim) throw std::runtime_error("launchDecodeAttentionSplit: head_dim must be 128");
    if (max_seq_len <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0)
        throw std::runtime_error("launchDecodeAttentionSplit: invalid shape");
    const int max_splits = (max_seq_len + kSplitChunk - 1) / kSplitChunk;
    float* part_o = static_cast<float*>(workspace);
    float* part_ml = part_o + (size_t)num_heads * max_splits * kHeadDim;
    // With d_pos the grid covers max_seq_len (graph-replayable); otherwise only seq_len.
    const int grid_splits = d_pos ? max_splits : (seq_len + kSplitChunk - 1) / kSplitChunk;
    if (k_scale && v_scale)
        decodeAttentionSplitKernel<true><<<dim3(num_heads, grid_splits), kThreads, 0, stream>>>(
            q, k_cache, v_cache, k_scale, v_scale, part_o, part_ml, seq_len, num_heads, num_kv_heads, max_splits, d_pos);
    else
        decodeAttentionSplitKernel<false><<<dim3(num_heads, grid_splits), kThreads, 0, stream>>>(
            q, k_cache, v_cache, nullptr, nullptr, part_o, part_ml, seq_len, num_heads, num_kv_heads, max_splits, d_pos);
    decodeAttentionCombineKernel<<<num_heads, kThreads, 0, stream>>>(part_o, part_ml, out, seq_len, max_splits, d_pos);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launchDecodeAttentionSplit: launch failed: ") + cudaGetErrorString(err));
}

bool decodeAttentionFitsSharedMemory(int max_seq_len) {
    return (size_t)(kHeadDim + kWarps + max_seq_len) * sizeof(float) <= 48 * 1024;
}

}  // namespace friday_attn
