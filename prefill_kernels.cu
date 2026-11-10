// Batched (prefill) kernels; see prefill_kernels.h.

#include "prefill_kernels.h"

#include <cstdio>
#include <cstdlib>

namespace {

using friday_hpc::Q8Kind;

// Block tile: 64 weight rows x 128 tokens, 128 values of K per step (4 blocks of
// 32, the activation scale granularity). 8 warps in a 2 x 4 grid, each 32 rows x
// 32 tokens = 2 x 4 mma tiles (16 x 8). ~39 KB of shared memory: 2 blocks per SM.
constexpr int BM = 64, BT = 128, BK = 128, LDS = BK + 16, THREADS = 256;

// Per 32-value block b of a weight row the tile holds integers q and
//   value = ws * (q, or sc0 * q | sc1 * q per 16 for the "split" types) - min term
// The 16-byte row padding puts the 8 rows read by one fragment load in different banks.
struct Smem {
    int8_t w[BM][LDS];
    int8_t x[BT][LDS];   // int8 activations
    float ws[BM][4];     // main scale per 32 values
    float wm[BM][8];     // min per 16 (Q2_K) or per 32 in [0..3] (Q4_K, Q5_K)
    int wsc[BM][8];      // integer sub-scale per 16 (Q3_K, Q6_K)
    float xd[BT][4];     // activation scale per 32 values
    float xm[BT][8];     // activation scale * sum per 16 values (min term)
};
static_assert(sizeof(Smem) <= 48 * 1024, "prefill GEMM tile exceeds static shared memory");

__host__ __device__ constexpr int sbBytes(Q8Kind k) {
    return k == Q8Kind::Q2_K ? 84 : k == Q8Kind::Q3_K ? 110 : k == Q8Kind::Q4_K ? 144
         : k == Q8Kind::Q5_K ? 176 : k == Q8Kind::Q6_K ? 210 : 272;   // Q8_0: 8 blocks of 34
}
// Sub-scales per 16 values that do not fold into int8 (q * sc can exceed 127).
__host__ __device__ constexpr bool splitType(Q8Kind k) { return k == Q8Kind::Q3_K || k == Q8Kind::Q6_K; }

__device__ __forceinline__ float ldh(const uint8_t* p) { return __half2float(*reinterpret_cast<const half*>(p)); }
__device__ __forceinline__ uint32_t ld4(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint32_t ld2x2(const uint8_t* p) {   // 4 bytes from a 2-byte aligned address
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t)q[0] | ((uint32_t)q[1] << 16);
}

// get_scale_min_k4 (Q4_K / Q5_K)
__device__ __forceinline__ void scaleMinK4(int is, const uint8_t* sc, int& s, int& m) {
    if (is < 4) { s = sc[is] & 63; m = sc[is + 4] & 63; }
    else { s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
}

struct Deq {
    uint32_t q[8];     // 32 int8 values
    float ws = 0.0f, wm0 = 0.0f, wm1 = 0.0f;
    int sc0 = 0, sc1 = 0;
};

// Values 32 * blk .. 32 * blk + 31 of superblock b (GGML layouts).
template <Q8Kind KIND>
__device__ __forceinline__ void dequant32(const uint8_t* b, int blk, Deq& o) {
    if constexpr (KIND == Q8Kind::Q2_K) {      // scales[16] qs[64] d dmin; 4-byte aligned
        const int n = blk >> 2, j = blk & 3;
        const uint8_t* qs = b + 16 + 32 * n;
        const int s0 = b[2 * blk], s1 = b[2 * blk + 1];
        #pragma unroll
        for (int w = 0; w < 8; ++w)   // q * sc <= 3 * 15: no carry between bytes
            o.q[w] = ((ld4(qs + 4 * w) >> (2 * j)) & 0x03030303u) * (uint32_t)((w < 4 ? s0 : s1) & 0xF);
        o.ws = ldh(b + 80);
        const float dmin = ldh(b + 82);
        o.wm0 = dmin * (float)(s0 >> 4);
        o.wm1 = dmin * (float)(s1 >> 4);
    } else if constexpr (KIND == Q8Kind::Q3_K) {   // hmask[32] qs[64] scales[12] d; 2-byte aligned
        const int n = blk >> 2, j = blk & 3, bit = 4 * n + j;
        const uint8_t* qs = b + 32 + 32 * n;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t lo = (ld2x2(qs + 4 * w) >> (2 * j)) & 0x03030303u;
            const uint32_t hi = (ld2x2(b + 4 * w) >> bit) & 0x01010101u;
            o.q[w] = __vsub4(lo | (hi << 2), 0x04040404u);   // -4 .. 3
        }
        const uint8_t* sc = b + 96;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int s = 2 * blk + h;
            const int lo = s < 8 ? (sc[s] & 0xF) : (sc[s - 8] >> 4);
            const int hi = (sc[8 + (s & 3)] >> (2 * (s >> 2))) & 3;
            (h ? o.sc1 : o.sc0) = (lo | (hi << 4)) - 32;
        }
        o.ws = ldh(b + 108);
    } else if constexpr (KIND == Q8Kind::Q4_K) {   // d dmin scales[12] qs[128]; 16-byte aligned
        const int j = blk >> 1, hi = blk & 1;
        const uint8_t* qs = b + 16 + 32 * j;
        #pragma unroll
        for (int w = 0; w < 8; ++w) o.q[w] = (ld4(qs + 4 * w) >> (4 * hi)) & 0x0F0F0F0Fu;
        int sc, m;
        scaleMinK4(blk, b + 4, sc, m);
        o.ws = ldh(b) * (float)sc;
        o.wm0 = ldh(b + 2) * (float)m;
    } else if constexpr (KIND == Q8Kind::Q5_K) {   // d dmin scales[12] qh[32] qs[128]; 16-byte aligned
        const int j = blk >> 1, hi = blk & 1;
        const uint8_t* qs = b + 48 + 32 * j;
        #pragma unroll
        for (int w = 0; w < 8; ++w)
            o.q[w] = ((ld4(qs + 4 * w) >> (4 * hi)) & 0x0F0F0F0Fu) | (((ld4(b + 16 + 4 * w) >> blk) & 0x01010101u) << 4);
        int sc, m;
        scaleMinK4(blk, b + 4, sc, m);
        o.ws = ldh(b) * (float)sc;
        o.wm0 = ldh(b + 2) * (float)m;
    } else if constexpr (KIND == Q8Kind::Q6_K) {   // ql[128] qh[64] scales[16] d; 2-byte aligned
        const int n = blk >> 2, p = blk & 3;
        const uint8_t* ql = b + 64 * n + 32 * (p & 1);
        const uint8_t* qh = b + 128 + 32 * n;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t lo = (ld2x2(ql + 4 * w) >> (p < 2 ? 0 : 4)) & 0x0F0F0F0Fu;
            const uint32_t hi = (ld2x2(qh + 4 * w) >> (2 * p)) & 0x03030303u;
            o.q[w] = __vsub4(lo | (hi << 4), 0x20202020u);   // -32 .. 31
        }
        o.sc0 = (int8_t)b[192 + 2 * blk];
        o.sc1 = (int8_t)b[193 + 2 * blk];
        o.ws = ldh(b + 208);
    } else {                                     // Q8_0: 8 x (d qs[32]); 2-byte aligned
        const uint8_t* bl = b + 34 * blk;
        #pragma unroll
        for (int w = 0; w < 8; ++w) o.q[w] = ld2x2(bl + 2 + 4 * w);
        o.ws = ldh(bl);
    }
}

__device__ __forceinline__ void mmaK16(int (&c)[4], int a0, int a1, int b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
#else
    __trap();
#endif
}

__device__ __forceinline__ void mmaK32(int (&c)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
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
__global__ __launch_bounds__(THREADS, 2) void gemm_q8_kernel(const uint8_t* __restrict__ W,
                                                             const int8_t* __restrict__ xq,
                                                             const float* __restrict__ xd, half* Y, int T, int K,
                                                             int N, int ldy, const float* __restrict__ bias,
                                                             const half* res) {
    __shared__ __align__(16) Smem sm;
    constexpr int SB = sbBytes(KIND);
    constexpr bool kSplit = splitType(KIND);
    constexpr bool kMin16 = KIND == Q8Kind::Q2_K;
    constexpr bool kMin32 = KIND == Q8Kind::Q4_K || KIND == Q8Kind::Q5_K;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wr = (warp >> 2) * 32, wt = (warp & 3) * 32;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BT;
    const size_t row_bytes = (size_t)(K / 256) * SB;
    const int K32 = K / 32;

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[mi][ni][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Weights: one 32-value block per thread.
        {
            const int r = tid >> 2, bb = tid & 3;
            Deq d;
            if (row0 + r < N) {
                const uint8_t* sbp = W + (size_t)(row0 + r) * row_bytes + (size_t)(k0 >> 8) * SB;
                dequant32<KIND>(sbp, ((k0 >> 7) & 1) * 4 + bb, d);
            } else {
                #pragma unroll
                for (int w = 0; w < 8; ++w) d.q[w] = 0;
            }
            uint4* dst = reinterpret_cast<uint4*>(&sm.w[r][32 * bb]);
            dst[0] = make_uint4(d.q[0], d.q[1], d.q[2], d.q[3]);
            dst[1] = make_uint4(d.q[4], d.q[5], d.q[6], d.q[7]);
            sm.ws[r][bb] = d.ws;
            if constexpr (kMin16) { sm.wm[r][2 * bb] = d.wm0; sm.wm[r][2 * bb + 1] = d.wm1; }
            if constexpr (kMin32) sm.wm[r][bb] = d.wm0;
            if constexpr (kSplit) { sm.wsc[r][2 * bb] = d.sc0; sm.wsc[r][2 * bb + 1] = d.sc1; }
        }
        // Activations: 16 bytes (one 16-value group) per item, 4 items per thread.
        #pragma unroll
        for (int i = 0; i < (BT * BK / 16) / THREADS; ++i) {
            const int it = tid + i * THREADS, t = it >> 3, c = it & 7;
            int4 v = make_int4(0, 0, 0, 0);
            float dx = 0.0f;
            if (tok0 + t < T) {
                v = *reinterpret_cast<const int4*>(xq + (size_t)(tok0 + t) * K + k0 + 16 * c);
                dx = xd[(size_t)(tok0 + t) * K32 + (k0 >> 5) + (c >> 1)];
            }
            *reinterpret_cast<int4*>(&sm.x[t][16 * c]) = v;
            if ((c & 1) == 0) sm.xd[t][c >> 1] = dx;
            if constexpr (kMin16 || kMin32) {
                const int xs = __dp4a(v.w, 0x01010101, __dp4a(v.z, 0x01010101,
                               __dp4a(v.y, 0x01010101, __dp4a(v.x, 0x01010101, 0))));
                sm.xm[t][c] = dx * (float)xs;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int bb = 0; bb < 4; ++bb) {
            const int kk = 32 * bb + 4 * tig;
            int a[2][4], bq[4][2];
            float wsA[2], wsB[2];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wr + 16 * mi + g;
                a[mi][0] = *reinterpret_cast<const int*>(&sm.w[r][kk]);
                a[mi][1] = *reinterpret_cast<const int*>(&sm.w[r + 8][kk]);
                a[mi][2] = *reinterpret_cast<const int*>(&sm.w[r][kk + 16]);
                a[mi][3] = *reinterpret_cast<const int*>(&sm.w[r + 8][kk + 16]);
                wsA[mi] = sm.ws[r][bb];
                wsB[mi] = sm.ws[r + 8][bb];
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int t = wt + 8 * ni + g;
                bq[ni][0] = *reinterpret_cast<const int*>(&sm.x[t][kk]);
                bq[ni][1] = *reinterpret_cast<const int*>(&sm.x[t][kk + 16]);
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int t = wt + 8 * ni + 2 * tig;
                const float d0 = sm.xd[t][bb], d1 = sm.xd[t + 1][bb];
                float m0a = 0.0f, m1a = 0.0f, m0b = 0.0f, m1b = 0.0f;
                if constexpr (kMin16) {
                    m0a = sm.xm[t][2 * bb]; m0b = sm.xm[t][2 * bb + 1];
                    m1a = sm.xm[t + 1][2 * bb]; m1b = sm.xm[t + 1][2 * bb + 1];
                }
                if constexpr (kMin32) {
                    m0a = sm.xm[t][2 * bb] + sm.xm[t][2 * bb + 1];
                    m1a = sm.xm[t + 1][2 * bb] + sm.xm[t + 1][2 * bb + 1];
                }
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    int c[4] = {0, 0, 0, 0};
                    if constexpr (kSplit) {
                        int lo[4] = {0, 0, 0, 0}, hi[4] = {0, 0, 0, 0};
                        mmaK16(lo, a[mi][0], a[mi][1], bq[ni][0]);
                        mmaK16(hi, a[mi][2], a[mi][3], bq[ni][1]);
                        const int r = wr + 16 * mi + g;
                        const int sA0 = sm.wsc[r][2 * bb], sA1 = sm.wsc[r][2 * bb + 1];
                        const int sB0 = sm.wsc[r + 8][2 * bb], sB1 = sm.wsc[r + 8][2 * bb + 1];
                        c[0] = sA0 * lo[0] + sA1 * hi[0];
                        c[1] = sA0 * lo[1] + sA1 * hi[1];
                        c[2] = sB0 * lo[2] + sB1 * hi[2];
                        c[3] = sB0 * lo[3] + sB1 * hi[3];
                    } else {
                        mmaK32(c, a[mi][0], a[mi][1], a[mi][2], a[mi][3], bq[ni][0], bq[ni][1]);
                    }
                    float* ac = acc[mi][ni];
                    ac[0] += (d0 * wsA[mi]) * (float)c[0];
                    ac[1] += (d1 * wsA[mi]) * (float)c[1];
                    ac[2] += (d0 * wsB[mi]) * (float)c[2];
                    ac[3] += (d1 * wsB[mi]) * (float)c[3];
                    if constexpr (kMin16 || kMin32) {
                        const int r = wr + 16 * mi + g;
                        const float wa0 = kMin16 ? sm.wm[r][2 * bb] : sm.wm[r][bb];
                        const float wb0 = kMin16 ? sm.wm[r + 8][2 * bb] : sm.wm[r + 8][bb];
                        ac[0] -= wa0 * m0a;
                        ac[1] -= wa0 * m1a;
                        ac[2] -= wb0 * m0a;
                        ac[3] -= wb0 * m1a;
                        if constexpr (kMin16) {
                            const float wa1 = sm.wm[r][2 * bb + 1], wb1 = sm.wm[r + 8][2 * bb + 1];
                            ac[0] -= wa1 * m0b;
                            ac[1] -= wa1 * m1b;
                            ac[2] -= wb1 * m0b;
                            ac[3] -= wb1 * m1b;
                        }
                    }
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

bool gemmQ8Supported(Q8Kind kind, int K) { return kind != Q8Kind::None && K % 256 == 0 && deviceCC() >= 80; }

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
