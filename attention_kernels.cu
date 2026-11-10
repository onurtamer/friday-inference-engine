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
constexpr int kSplitChunk = 64;

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

// fixed_splits > 0: partials from decodeAttentionGqaKernel with that many splits
// (chunk = ceil(seq / fixed_splits)); 0: fixed chunks of kSplitChunk.
__global__ __launch_bounds__(kThreads)
void decodeAttentionCombineKernel(const float* __restrict__ part_o, const float* __restrict__ part_ml,
                                  half* __restrict__ out, int seq_len, int max_splits, const int* __restrict__ d_pos,
                                  int fixed_splits) {
    if (d_pos) seq_len = *d_pos + 1;
    const int head = blockIdx.x, tid = threadIdx.x;
    int nsp;
    if (fixed_splits > 0) {
        const int chunk = (seq_len + fixed_splits - 1) / fixed_splits;
        nsp = (seq_len + chunk - 1) / chunk;
        max_splits = fixed_splits;
    } else {
        nsp = (seq_len + kSplitChunk - 1) / kSplitChunk;
    }
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

// ---------------------------------------------------------------------------
// GQA split-K decode attention: one block per (kv head, chunk of kSplitChunk
// positions). K and V of the chunk are read once and used for all G query
// heads that share the kv head; each lane reads 4 consecutive dims of K with
// one 8-byte (fp16) / 4-byte (int8) load. Per-score rules as before (NaN score
// -> -1e30, exp cut off below -50 relative to the block max). Partials go to
// the same layout as before: slot = query_head * max_splits + split.
// ---------------------------------------------------------------------------
constexpr int kMaxGroup = 8;
constexpr int kGqaSplits = 32;
// Partial slots per query head: enough for both kernels' split counts.
inline int splitSlots(int max_seq_len) {
    const int a = (max_seq_len + kSplitChunk - 1) / kSplitChunk;
    return a > kGqaSplits ? a : kGqaSplits;
}

template <bool Q8>
__device__ __forceinline__ void kvLoad4(const void* cache, const half* scale, size_t idx, float out[4]) {
    if (Q8) {
        const char4 c = *reinterpret_cast<const char4*>(static_cast<const int8_t*>(cache) + idx);
        const float s = __half2float(scale[idx / 32]);
        out[0] = c.x * s; out[1] = c.y * s; out[2] = c.z * s; out[3] = c.w * s;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(cache) + idx);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
    }
}

template <bool Q8>
__global__ __launch_bounds__(kThreads)
void decodeAttentionGqaKernel(const half* __restrict__ q, const void* __restrict__ k_cache,
                              const void* __restrict__ v_cache, const half* __restrict__ k_scale,
                              const half* __restrict__ v_scale, float* __restrict__ part_o,
                              float* __restrict__ part_ml, int seq_len, int num_heads, int num_kv_heads,
                              int max_splits, const int* __restrict__ d_pos) {
    __shared__ float s_q[kMaxGroup][kHeadDim];
    __shared__ float s_m[kMaxGroup], s_l[kMaxGroup];
    extern __shared__ float s_dyn[];   // scores [G][max_chunk]
    if (d_pos) seq_len = *d_pos + 1;
    const int kvh = blockIdx.x, split = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int G = num_heads / num_kv_heads;
    // gridDim.y splits cover the sequence evenly (chunk from the length, so the
    // grid stays fixed for CUDA Graph replay).
    const int chunk = (seq_len + gridDim.y - 1) / gridDim.y;
    const int max_chunk = max_splits;  // reused parameter: score row stride
    const int start = split * chunk;
    if (start >= seq_len) return;  // inactive split (the combine kernel ignores it)
    const int n = min(chunk, seq_len - start);
    const int kv_dim = num_kv_heads * kHeadDim;
    const float scale = rsqrtf((float)kHeadDim);

    for (int i = tid; i < G * kHeadDim; i += kThreads)
        s_q[i / kHeadDim][i % kHeadDim] = __half2float(q[(kvh * G + i / kHeadDim) * kHeadDim + i % kHeadDim]);
    __syncthreads();

    // Scores: warp per position, lane covers dims 4*lane .. 4*lane+3.
    // Scores: 8 lanes per position (each lane 16 dims, one 32-byte fp16 / 16-byte
    // int8 load), so a warp scores 4 positions at once; 3-step reduction.
    {
        const int sub = lane >> 3, sl = lane & 7;
        for (int i0 = warp * 4; i0 < n; i0 += kWarps * 4) {
            const int i = i0 + sub;
            float kf[16];
            if (i < n) {
                const size_t idx = (size_t)(start + i) * kv_dim + kvh * kHeadDim + 16 * sl;
                if (Q8) {
                    const int4 raw = *reinterpret_cast<const int4*>(static_cast<const int8_t*>(k_cache) + idx);
                    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
                    const float sc = __half2float(k_scale[idx / 32]);
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) kf[j] = c[j] * sc;
                } else {
                    const uint4* src = reinterpret_cast<const uint4*>(static_cast<const half*>(k_cache) + idx);
                    const uint4 r0 = src[0], r1 = src[1];
                    const half2* h0 = reinterpret_cast<const half2*>(&r0);
                    const half2* h1 = reinterpret_cast<const half2*>(&r1);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f0 = __half22float2(h0[j]), f1 = __half22float2(h1[j]);
                        kf[2 * j] = f0.x; kf[2 * j + 1] = f0.y; kf[8 + 2 * j] = f1.x; kf[8 + 2 * j + 1] = f1.y;
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 16; ++j) kf[j] = 0.0f;
            }
            #pragma unroll
            for (int g = 0; g < kMaxGroup; ++g) {
                if (g >= G) break;
                const float* qg = &s_q[g][16 * sl];
                float part = 0.0f;
                #pragma unroll
                for (int j = 0; j < 16; ++j) part += qg[j] * kf[j];
                part += __shfl_xor_sync(0xffffffffu, part, 4);
                part += __shfl_xor_sync(0xffffffffu, part, 2);
                part += __shfl_xor_sync(0xffffffffu, part, 1);
                if (sl == 0 && i < n) {
                    float score = part * scale;
                    if (isnan(score)) score = -1e30f;
                    s_dyn[g * max_chunk + i] = score;
                }
            }
        }
    }
    __syncthreads();
    // Per query head: max, exp, sum (warp g handles heads g, g + kWarps, ...).
    for (int g = warp; g < G; g += kWarps) {
        float m = -1e30f;
        float* sg = s_dyn + g * max_chunk;
        for (int i = lane; i < n; i += 32) m = fmaxf(m, sg[i]);
        m = warpMax(m);
        m = __shfl_sync(0xffffffffu, m, 0);
        float sum = 0.0f;
        for (int i = lane; i < n; i += 32) {
            const float diff = sg[i] - m;
            const float e = (diff < -50.0f) ? 0.0f : expf(diff);
            sg[i] = e;
            sum += e;
        }
        sum = warpSum(sum);
        if (lane == 0) { s_m[g] = m; s_l[g] = sum; }
    }
    __syncthreads();
    // V: thread owns 2 dims (2*dp, 2*dp+1) for all G heads; the two halves of
    // the block take alternate positions and are summed through shared memory.
    {
        const int dp = tid & 63, ph = tid >> 6;
        float acc0[kMaxGroup], acc1[kMaxGroup];
        #pragma unroll
        for (int g = 0; g < kMaxGroup; ++g) { acc0[g] = 0.0f; acc1[g] = 0.0f; }
        const size_t vbase = (size_t)start * kv_dim + kvh * kHeadDim + 2 * dp;
        for (int i = ph; i < n; i += 2) {
            float v0, v1;
            const size_t idx = vbase + (size_t)i * kv_dim;
            if (Q8) {
                const char2 c = *reinterpret_cast<const char2*>(static_cast<const int8_t*>(v_cache) + idx);
                const float sc = __half2float(v_scale[idx / 32]);
                v0 = c.x * sc; v1 = c.y * sc;
            } else {
                const float2 f = __half22float2(*reinterpret_cast<const half2*>(static_cast<const half*>(v_cache) + idx));
                v0 = f.x; v1 = f.y;
            }
            #pragma unroll
            for (int g = 0; g < kMaxGroup; ++g) {
                if (g >= G) break;
                const float w = s_dyn[g * max_chunk + i];
                acc0[g] += w * v0;
                acc1[g] += w * v1;
            }
        }
        __syncthreads();  // s_q is free now: reuse it to combine the two halves
        if (ph == 1) {
            #pragma unroll
            for (int g = 0; g < kMaxGroup; ++g) {
                if (g >= G) break;
                s_q[g][2 * dp] = acc0[g];
                s_q[g][2 * dp + 1] = acc1[g];
            }
        }
        __syncthreads();
        if (ph == 0) {
            #pragma unroll
            for (int g = 0; g < kMaxGroup; ++g) {
                if (g >= G) break;
                const size_t slot = (size_t)(kvh * G + g) * gridDim.y + split;
                part_o[slot * kHeadDim + 2 * dp] = acc0[g] + s_q[g][2 * dp];
                part_o[slot * kHeadDim + 2 * dp + 1] = acc1[g] + s_q[g][2 * dp + 1];
                if (dp == 0) { part_ml[slot * 2] = s_m[g]; part_ml[slot * 2 + 1] = s_l[g]; }
            }
        }
    }
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
    const size_t splits = (size_t)splitSlots(max_seq_len);
    return (size_t)num_heads * splits * (kHeadDim + 2) * sizeof(float);
}

void launchDecodeAttentionSplit(const half* q, const void* k_cache, const void* v_cache, half* out, int seq_len,
                                const int* d_pos, int max_seq_len, int num_heads, int num_kv_heads, int head_dim,
                                void* workspace, cudaStream_t stream, const half* k_scale, const half* v_scale) {
    if (head_dim != kHeadDim) throw std::runtime_error("launchDecodeAttentionSplit: head_dim must be 128");
    if (max_seq_len <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0)
        throw std::runtime_error("launchDecodeAttentionSplit: invalid shape");
    const int max_splits = splitSlots(max_seq_len);
    float* part_o = static_cast<float*>(workspace);
    float* part_ml = part_o + (size_t)num_heads * max_splits * kHeadDim;
    // With d_pos the grid covers max_seq_len (graph-replayable); otherwise only seq_len.
    const int grid_splits = d_pos ? max_splits : (seq_len + kSplitChunk - 1) / kSplitChunk;
    const int G = num_heads / num_kv_heads;
    if (G <= kMaxGroup) {
        // GQA kernel: one block per (kv head, split); kGqaSplits splits share the
        // sequence evenly, so every length uses num_kv_heads * kGqaSplits blocks.
        const int max_chunk = (max_seq_len + kGqaSplits - 1) / kGqaSplits;
        const size_t smem = (size_t)G * max_chunk * sizeof(float);
        if (k_scale && v_scale)
            decodeAttentionGqaKernel<true><<<dim3(num_kv_heads, kGqaSplits), kThreads, smem, stream>>>(
                q, k_cache, v_cache, k_scale, v_scale, part_o, part_ml, seq_len, num_heads, num_kv_heads, max_chunk, d_pos);
        else
            decodeAttentionGqaKernel<false><<<dim3(num_kv_heads, kGqaSplits), kThreads, smem, stream>>>(
                q, k_cache, v_cache, nullptr, nullptr, part_o, part_ml, seq_len, num_heads, num_kv_heads, max_chunk, d_pos);
        decodeAttentionCombineKernel<<<num_heads, kThreads, 0, stream>>>(part_o, part_ml, out, seq_len, max_splits, d_pos,
                                                                         kGqaSplits);
    } else {
    if (k_scale && v_scale)
        decodeAttentionSplitKernel<true><<<dim3(num_heads, grid_splits), kThreads, 0, stream>>>(
            q, k_cache, v_cache, k_scale, v_scale, part_o, part_ml, seq_len, num_heads, num_kv_heads, max_splits, d_pos);
    else
        decodeAttentionSplitKernel<false><<<dim3(num_heads, grid_splits), kThreads, 0, stream>>>(
            q, k_cache, v_cache, nullptr, nullptr, part_o, part_ml, seq_len, num_heads, num_kv_heads, max_splits, d_pos);
    decodeAttentionCombineKernel<<<num_heads, kThreads, 0, stream>>>(part_o, part_ml, out, seq_len, max_splits, d_pos, 0);
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launchDecodeAttentionSplit: launch failed: ") + cudaGetErrorString(err));
}

bool decodeAttentionFitsSharedMemory(int max_seq_len) {
    return (size_t)(kHeadDim + kWarps + max_seq_len) * sizeof(float) <= 48 * 1024;
}

}  // namespace friday_attn
