// Batched (prefill) kernels; see prefill_kernels.h.

#include "prefill_kernels.h"

#include <cstdio>
#include <cstdlib>

namespace {

using friday_hpc::Q8Kind;

// Block tile: 64 weight rows x 64 tokens, one 256-value superblock of K per step.
// 4 warps in a 2 x 2 grid, each 32 rows x 32 tokens = 2 x 4 mma tiles (16 x 8).
constexpr int BM = 64, BT = 64, BK = 256, LDS = BK + 16, THREADS = 128;

// 48 KiB exactly. The 16-byte row padding puts the 8 rows read by one fragment
// load in different banks.
struct Smem {
    int8_t w[BM][LDS];   // dequantized weight values (integers, see dequant16)
    int8_t x[BT][LDS];   // int8 activations
    float ws[BM][16];    // per 16 values: value = ws * q - wm
    float wm[BM][16];
    float xd[BT][8];     // activation scale per 32 values
    int xs[BT][16];      // activation sum per 16 values (for the wm term)
};
static_assert(sizeof(Smem) <= 48 * 1024, "prefill GEMM tile exceeds static shared memory");

__host__ __device__ constexpr int sbBytes(Q8Kind k) {
    return k == Q8Kind::Q2_K ? 84 : k == Q8Kind::Q3_K ? 110 : k == Q8Kind::Q4_K ? 144
         : k == Q8Kind::Q5_K ? 176 : k == Q8Kind::Q6_K ? 210 : 272;   // Q8_0: 8 blocks of 34
}

__device__ __forceinline__ float ldh(const uint8_t* p) { return __half2float(*reinterpret_cast<const half*>(p)); }

// get_scale_min_k4 (Q4_K / Q5_K)
__device__ __forceinline__ void scaleMinK4(int is, const uint8_t* sc, int& s, int& m) {
    if (is < 4) { s = sc[is] & 63; m = sc[is + 4] & 63; }
    else { s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
}

// Values 16s .. 16s + 15 of superblock b as integers q with value = ws * q - wm
// (GGML layouts; the same integers the dp4a GEMV kernels multiply).
template <Q8Kind KIND>
__device__ __forceinline__ void dequant16(const uint8_t* b, int s, int8_t* q, float& ws, float& wm) {
    if constexpr (KIND == Q8Kind::Q2_K) {      // scales[16] qs[64] d dmin
        const int n = s >> 3, j = (s & 7) >> 1, h = s & 1;
        const uint8_t* qs = b + 16 + 32 * n + 16 * h;
        #pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (int8_t)((qs[i] >> (2 * j)) & 3);
        const int sc = b[s];
        ws = ldh(b + 80) * (float)(sc & 0xF);
        wm = ldh(b + 82) * (float)(sc >> 4);
    } else if constexpr (KIND == Q8Kind::Q3_K) {   // hmask[32] qs[64] scales[12] d
        const int n = s >> 3, j = (s & 7) >> 1, h = s & 1;
        const uint8_t* qs = b + 32 + 32 * n + 16 * h;
        const uint8_t* hm = b + 16 * h;
        const int bit = 4 * n + j;
        #pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (int8_t)(((qs[i] >> (2 * j)) & 3) - (((hm[i] >> bit) & 1) ? 0 : 4));
        const uint8_t* sc = b + 96;
        const int lo = s < 8 ? (sc[s] & 0xF) : (sc[s - 8] >> 4);
        const int hi = (sc[8 + (s & 3)] >> (2 * (s >> 2))) & 3;
        ws = ldh(b + 108) * (float)((lo | (hi << 4)) - 32);
        wm = 0.0f;
    } else if constexpr (KIND == Q8Kind::Q4_K) {   // d dmin scales[12] qs[128]
        const int j = s >> 2, hi = (s >> 1) & 1, l0 = 16 * (s & 1);
        const uint8_t* qs = b + 16 + 32 * j + l0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (int8_t)(hi ? (qs[i] >> 4) : (qs[i] & 0xF));
        int sc, m;
        scaleMinK4(2 * j + hi, b + 4, sc, m);
        ws = ldh(b) * (float)sc;
        wm = ldh(b + 2) * (float)m;
    } else if constexpr (KIND == Q8Kind::Q5_K) {   // d dmin scales[12] qh[32] qs[128]
        const int j = s >> 2, hi = (s >> 1) & 1, l0 = 16 * (s & 1);
        const uint8_t* qs = b + 48 + 32 * j + l0;
        const uint8_t* qh = b + 16 + l0;
        const int bit = 2 * j + hi;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            q[i] = (int8_t)((hi ? (qs[i] >> 4) : (qs[i] & 0xF)) | (((qh[i] >> bit) & 1) << 4));
        int sc, m;
        scaleMinK4(2 * j + hi, b + 4, sc, m);
        ws = ldh(b) * (float)sc;
        wm = ldh(b + 2) * (float)m;
    } else if constexpr (KIND == Q8Kind::Q6_K) {   // ql[128] qh[64] scales[16] d
        const int n = s >> 3, p = (s & 7) >> 1, l0 = 16 * (s & 1);
        const uint8_t* ql = b + 64 * n + 32 * (p & 1) + l0;
        const uint8_t* qh = b + 128 + 32 * n + l0;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            q[i] = (int8_t)(((p < 2 ? (ql[i] & 0xF) : (ql[i] >> 4)) | (((qh[i] >> (2 * p)) & 3) << 4)) - 32);
        ws = ldh(b + 208) * (float)(int8_t)b[192 + s];
        wm = 0.0f;
    } else {                                     // Q8_0: 8 x (d qs[32])
        const uint8_t* blk = b + 34 * (s >> 1);
        #pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (int8_t)blk[2 + 16 * (s & 1) + i];
        ws = ldh(blk);
        wm = 0.0f;
    }
}

__device__ __forceinline__ uint32_t pack4(const int8_t* q) {
    return (uint32_t)(uint8_t)q[0] | ((uint32_t)(uint8_t)q[1] << 8) | ((uint32_t)(uint8_t)q[2] << 16) |
           ((uint32_t)(uint8_t)q[3] << 24);
}

__device__ __forceinline__ void mmaS8(int (&c)[4], int a0, int a1, int b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
#else
    __trap();
#endif
}

__device__ __forceinline__ float clampHalfRange(float r) {
    if (isnan(r)) return 0.0f;
    return fminf(fmaxf(r, -65504.0f), 65504.0f);
}

// Same epilogue as the GEMV kernels (storeQ8 in hpc_k_quants.cu).
__device__ __forceinline__ void storeY(half* y, const half* res, const float* bias, int n, float acc) {
    half h = __float2half(acc);
    if (bias) {
        float t = __half2float(h), b = bias[n];
        if (isnan(t)) t = 0.0f;
        if (isnan(b)) b = 0.0f;
        h = __float2half(clampHalfRange(t + b));
    }
    if (res) {
        float t = __half2float(res[n]), a = __half2float(h);
        if (isnan(t)) t = 0.0f;
        if (isnan(a)) a = 0.0f;
        h = __float2half(clampHalfRange(t + a));
    }
    y[n] = h;
}

template <Q8Kind KIND>
__global__ __launch_bounds__(THREADS) void gemm_q8_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                                                          const float* __restrict__ xd, half* Y, int T, int K, int N,
                                                          int ldy, const float* __restrict__ bias, const half* res) {
    __shared__ __align__(16) Smem sm;
    constexpr int SB = sbBytes(KIND);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wr = (warp >> 1) * 32, wt = (warp & 1) * 32;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BT;
    const int nsb = K / BK;
    const size_t row_bytes = (size_t)nsb * SB;

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[mi][ni][j] = 0.0f;

    for (int sb = 0; sb < nsb; ++sb) {
        for (int it = tid; it < BM * 16; it += THREADS) {
            const int r = it >> 4, s = it & 15;
            int8_t q[16];
            float a = 0.0f, m = 0.0f;
            if (row0 + r < N) {
                dequant16<KIND>(W + (size_t)(row0 + r) * row_bytes + (size_t)sb * SB, s, q, a, m);
            } else {
                #pragma unroll
                for (int i = 0; i < 16; ++i) q[i] = 0;
            }
            *reinterpret_cast<uint4*>(&sm.w[r][16 * s]) = make_uint4(pack4(q), pack4(q + 4), pack4(q + 8), pack4(q + 12));
            sm.ws[r][s] = a;
            sm.wm[r][s] = m;
        }
        for (int it = tid; it < BT * 16; it += THREADS) {
            const int t = it >> 4, s = it & 15;
            int4 v = make_int4(0, 0, 0, 0);
            if (tok0 + t < T) v = *reinterpret_cast<const int4*>(xq + (size_t)(tok0 + t) * K + (size_t)sb * BK + 16 * s);
            *reinterpret_cast<int4*>(&sm.x[t][16 * s]) = v;
            sm.xs[t][s] = __dp4a(v.w, 0x01010101, __dp4a(v.z, 0x01010101, __dp4a(v.y, 0x01010101, __dp4a(v.x, 0x01010101, 0))));
        }
        for (int it = tid; it < BT * 8; it += THREADS) {
            const int t = it >> 3, j = it & 7;
            sm.xd[t][j] = (tok0 + t < T) ? xd[(size_t)(tok0 + t) * (K / 32) + (size_t)sb * 8 + j] : 0.0f;
        }
        __syncthreads();

        #pragma unroll 2
        for (int s = 0; s < 16; ++s) {
            int a[2][2], bq[4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wr + 16 * mi + g;
                a[mi][0] = *reinterpret_cast<const int*>(&sm.w[r][16 * s + 4 * tig]);
                a[mi][1] = *reinterpret_cast<const int*>(&sm.w[r + 8][16 * s + 4 * tig]);
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni)
                bq[ni] = *reinterpret_cast<const int*>(&sm.x[wt + 8 * ni + g][16 * s + 4 * tig]);
            float wsA[2], wsB[2], wmA[2], wmB[2];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wr + 16 * mi + g;
                wsA[mi] = sm.ws[r][s]; wmA[mi] = sm.wm[r][s];
                wsB[mi] = sm.ws[r + 8][s]; wmB[mi] = sm.wm[r + 8][s];
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int t = wt + 8 * ni + 2 * tig;
                const float d0 = sm.xd[t][s >> 1], d1 = sm.xd[t + 1][s >> 1];
                const float x0 = (float)sm.xs[t][s], x1 = (float)sm.xs[t + 1][s];
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    int c[4] = {0, 0, 0, 0};
                    mmaS8(c, a[mi][0], a[mi][1], bq[ni]);
                    acc[mi][ni][0] += d0 * (wsA[mi] * (float)c[0] - wmA[mi] * x0);
                    acc[mi][ni][1] += d1 * (wsA[mi] * (float)c[1] - wmA[mi] * x1);
                    acc[mi][ni][2] += d0 * (wsB[mi] * (float)c[2] - wmB[mi] * x0);
                    acc[mi][ni][3] += d1 * (wsB[mi] * (float)c[3] - wmB[mi] * x1);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = row0 + wr + 16 * mi + g + ((j & 2) ? 8 : 0);
                const int t = tok0 + wt + 8 * ni + 2 * tig + (j & 1);
                if (r < N && t < T)
                    storeY(Y + (size_t)t * ldy, res ? res + (size_t)t * ldy : nullptr, bias, r, acc[mi][ni][j]);
            }
}

int deviceCC() {
    static const int cc = [] {
        int dev = 0, major = 0, minor = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        return major * 10 + minor;
    }();
    return cc;
}

}  // namespace

namespace friday_prefill {

bool gemmQ8Supported(Q8Kind kind, int K) { return kind != Q8Kind::None && K % BK == 0 && deviceCC() >= 80; }

void gemmQ8(Q8Kind kind, const void* W, const int8_t* xq, const float* xd, int T, int K, int N, half* Y, int ldy,
            cudaStream_t s, const float* bias, const half* res) {
    if (!gemmQ8Supported(kind, K)) {
        std::fprintf(stderr, "gemmQ8: unsupported type, K=%d or device\n", K);
        std::abort();
    }
    const dim3 grid((N + BM - 1) / BM, (T + BT - 1) / BT);
    const uint8_t* w = static_cast<const uint8_t*>(W);
    switch (kind) {
        case Q8Kind::Q2_K: gemm_q8_kernel<Q8Kind::Q2_K><<<grid, THREADS, 0, s>>>(w, xq, xd, Y, T, K, N, ldy, bias, res); break;
        case Q8Kind::Q3_K: gemm_q8_kernel<Q8Kind::Q3_K><<<grid, THREADS, 0, s>>>(w, xq, xd, Y, T, K, N, ldy, bias, res); break;
        case Q8Kind::Q4_K: gemm_q8_kernel<Q8Kind::Q4_K><<<grid, THREADS, 0, s>>>(w, xq, xd, Y, T, K, N, ldy, bias, res); break;
        case Q8Kind::Q5_K: gemm_q8_kernel<Q8Kind::Q5_K><<<grid, THREADS, 0, s>>>(w, xq, xd, Y, T, K, N, ldy, bias, res); break;
        case Q8Kind::Q6_K: gemm_q8_kernel<Q8Kind::Q6_K><<<grid, THREADS, 0, s>>>(w, xq, xd, Y, T, K, N, ldy, bias, res); break;
        default:           gemm_q8_kernel<Q8Kind::Q8_0><<<grid, THREADS, 0, s>>>(w, xq, xd, Y, T, K, N, ldy, bias, res); break;
    }
}

}  // namespace friday_prefill
