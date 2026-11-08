// ============================================================================
// B KOLU / FAZ 2 - HPC K-QUANT ABI KERNEL MİMARİSİ
// Optimize edilmiş Q2_K, Q3_K, Q4_K Warp-Coalesced GEMV Çekirdekleri
// Doğrulama Kapısı (Validation Gate) Entegrasyonu ile.
// ============================================================================

#include <cuda_fp16.h>
#include "hpc_k_quants.h"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define QK_K 256
#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFFu

// ----------------------------------------------------------------------------
// 1. K-QUANT ABI INVARIANTS & STRUCTS
// GGML Birebir Referans Layout'u
// ----------------------------------------------------------------------------
// block_q2_K / block_q3_K / block_q4_K: see hpc_k_quants.h

static_assert(sizeof(block_q2_K) == 84, "K-Quant ABI Violation: Q2_K size mismatch!");
static_assert(sizeof(block_q3_K) == 110, "K-Quant ABI Violation: Q3_K size mismatch!");
static_assert(sizeof(block_q4_K) == 144, "K-Quant ABI Violation: Q4_K size mismatch!");

// Reduction variants
enum {
    VAR_CURRENT = 0,
    VAR_FMA = 1,
    VAR_DET = 2,
    VAR_FP64 = 3,
    VAR_KAHAN = 4
};

// Global config set by dispatcher
extern int g_reduction_variant;
int g_reduction_variant = 0;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ double warp_reduce_sum_fp64(double v) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ----------------------------------------------------------------------------
// 2. Q2_K HPC KERNEL
// ----------------------------------------------------------------------------
template <int WARPS_PER_BLOCK, int VAR>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
void hpc_q2_K_gemv_kernel(const block_q2_K* __restrict__ W,
                          const half* __restrict__ x,
                          half* __restrict__ y,
                          int N, int K)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row  = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= N) return;

    const int nb = K / QK_K;
    const block_q2_K* __restrict__ w = W + row * nb;
    
    float acc = 0.0f;
    float c = 0.0f; // For Kahan
    
    // For deterministic, we need shared memory to collect partials
    __shared__ float smem_partials[WARPS_PER_BLOCK * WARP_SIZE];
    
    for (int sb = 0; sb < nb; ++sb) {
        const float d = __half2float(w[sb].d);
        const float dmin = __half2float(w[sb].dmin);
        const uint8_t* qs = w[sb].qs;
        const uint8_t* scales = w[sb].scales;
        
        int k_block_base = sb * 256;
        
        #pragma unroll
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            uint8_t q_byte = qs[half_idx * 32 + lane];
            
            #pragma unroll
            for (int shift_idx = 0; shift_idx < 4; ++shift_idx) {
                int sc_idx = half_idx * 8 + shift_idx * 2 + (lane >= 16 ? 1 : 0);
                uint8_t scale_byte = scales[sc_idx];
                float dl = d * (float)(scale_byte & 0x0F);
                float ml = dmin * (float)(scale_byte >> 4);
                
                int q = (q_byte >> (shift_idx * 2)) & 3;
                float w_val = dl * (float)q - ml;
                int base_out = k_block_base + (half_idx * 4 + shift_idx) * 32;
                float x_val = __half2float(x[base_out + lane]);
                
                if (VAR == VAR_FMA) {
                    acc = __fmaf_rn(w_val, x_val, acc);
                } else if (VAR == VAR_KAHAN) {
                    float y_val = (w_val * x_val) - c;
                    float t = acc + y_val;
                    c = (t - acc) - y_val;
                    acc = t;
                } else {
                    acc += w_val * x_val; // For DET and FP64 we also do simple local acc
                }
            }
        }
    }
    
    if (VAR == VAR_DET) {
        smem_partials[warp * WARP_SIZE + lane] = acc;
        __syncwarp();
        if (lane == 0) {
            float sum = 0.0f;
            for(int i=0; i<32; i++) sum += smem_partials[warp * WARP_SIZE + i];
            y[row] = __float2half(sum);
        }
    } else if (VAR == VAR_FP64) {
        double acc_d = warp_reduce_sum_fp64((double)acc);
        if (lane == 0) y[row] = __float2half((float)acc_d);
    } else {
        acc = warp_reduce_sum(acc);
        if (lane == 0) y[row] = __float2half(acc);
    }
}

// ----------------------------------------------------------------------------
// 3. Q3_K HPC KERNEL
// ----------------------------------------------------------------------------
template <int WARPS_PER_BLOCK, int VAR>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
void hpc_q3_K_gemv_kernel(const block_q3_K* __restrict__ W,
                          const half* __restrict__ x,
                          half* __restrict__ y,
                          int N, int K)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row  = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= N) return;
    
    const int nb = K / QK_K;
    const block_q3_K* __restrict__ w = W + row * nb;
    float acc = 0.0f;
    float c = 0.0f;
    __shared__ float smem_partials[WARPS_PER_BLOCK * WARP_SIZE];
    
    for (int sb = 0; sb < nb; ++sb) {
        const float d = __half2float(w[sb].d);
        const uint8_t* qs = w[sb].qs;
        const uint8_t* hmask = w[sb].hmask;
        const uint8_t* scales = w[sb].scales;
        
        float sc_reg[16];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int l0 = (int)scales[i] & 0x0F;
            int h0 = ((int)scales[8 + (i % 4)] >> ((i / 4) * 2)) & 3;
            sc_reg[i] = d * (float)((l0 | (h0 << 4)) - 32);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int l1 = (int)scales[i] >> 4;
            int h1 = ((int)scales[8 + (i % 4)] >> (((i / 4) + 2) * 2)) & 3;
            sc_reg[i + 8] = d * (float)((l1 | (h1 << 4)) - 32);
        }
        
        int k_block_base = sb * 256;
        
        #pragma unroll
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            uint8_t q_byte = qs[half_idx * 32 + lane];
            uint8_t h_byte = hmask[lane];
            
            #pragma unroll
            for (int shift_idx = 0; shift_idx < 4; ++shift_idx) {
                int sc_idx = half_idx * 8 + shift_idx * 2 + (lane >= 16 ? 1 : 0);
                
                int q = (q_byte >> (shift_idx * 2)) & 3;
                int qh = (h_byte >> (half_idx * 4 + shift_idx)) & 1;
                float w_val = sc_reg[sc_idx] * (float)((q | (qh << 2)) - 4);
                
                int base_out = k_block_base + (half_idx * 4 + shift_idx) * 32;
                float x_val = __half2float(x[base_out + lane]);
                
                if (VAR == VAR_FMA) {
                    acc = __fmaf_rn(w_val, x_val, acc);
                } else if (VAR == VAR_KAHAN) {
                    float y_val = (w_val * x_val) - c;
                    float t = acc + y_val;
                    c = (t - acc) - y_val;
                    acc = t;
                } else {
                    acc += w_val * x_val;
                }
            }
        }
    }
    
    if (VAR == VAR_DET) {
        smem_partials[warp * WARP_SIZE + lane] = acc;
        __syncwarp();
        if (lane == 0) {
            float sum = 0.0f;
            for(int i=0; i<32; i++) sum += smem_partials[warp * WARP_SIZE + i];
            y[row] = __float2half(sum);
        }
    } else if (VAR == VAR_FP64) {
        double acc_d = warp_reduce_sum_fp64((double)acc);
        if (lane == 0) y[row] = __float2half((float)acc_d);
    } else {
        acc = warp_reduce_sum(acc);
        if (lane == 0) y[row] = __float2half(acc);
    }
}

// ----------------------------------------------------------------------------
// 4. Q4_K HPC KERNEL
// ----------------------------------------------------------------------------
template <int WARPS_PER_BLOCK, int VAR>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
void hpc_q4_K_gemv_kernel(const block_q4_K* __restrict__ W,
                          const half* __restrict__ x,
                          half* __restrict__ y,
                          int N, int K)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row  = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= N) return;
    
    const int nb = K / QK_K;
    const block_q4_K* __restrict__ w = W + row * nb;
    float acc = 0.0f;
    float c = 0.0f;
    __shared__ float smem_partials[WARPS_PER_BLOCK * WARP_SIZE];
    
    for (int sb = 0; sb < nb; ++sb) {
        const float d = __half2float(w[sb].d);
        const float dmin = __half2float(w[sb].dmin);
        const uint8_t* sc = w[sb].scales;
        const uint8_t* qs = w[sb].qs;
        
        float scales_reg[8], mins_reg[8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t s_j = sc[j];
            uint8_t m_j = sc[j + 4];
            uint8_t md_j = sc[j + 8];
            scales_reg[j] = d * (float)(s_j & 0x3F);
            mins_reg[j]   = dmin * (float)(m_j & 0x3F);
            scales_reg[j + 4] = d * (float)((md_j & 0x0F) | ((s_j >> 2) & 0x30));
            mins_reg[j + 4]   = dmin * (float)((md_j >> 4) | ((m_j >> 2) & 0x30));
        }
        
        int k_block_base = sb * 256;
        
        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            uint8_t q_byte = qs[h * 32 + lane];
            
            float w_even = scales_reg[2*h] * (float)(q_byte & 0x0F) - mins_reg[2*h];
            float w_odd  = scales_reg[2*h+1] * (float)(q_byte >> 4) - mins_reg[2*h+1];
            
            int base_k_even = k_block_base + h * 64 + lane;
            int base_k_odd  = k_block_base + h * 64 + 32 + lane;
            
            float x_even = __half2float(x[base_k_even]);
            float x_odd = __half2float(x[base_k_odd]);
            
            if (VAR == VAR_FMA) {
                acc = __fmaf_rn(w_even, x_even, acc);
                acc = __fmaf_rn(w_odd, x_odd, acc);
            } else if (VAR == VAR_KAHAN) {
                float y1 = (w_even * x_even) - c;
                float t1 = acc + y1;
                c = (t1 - acc) - y1;
                acc = t1;
                
                float y2 = (w_odd * x_odd) - c;
                float t2 = acc + y2;
                c = (t2 - acc) - y2;
                acc = t2;
            } else {
                acc += w_even * x_even;
                acc += w_odd * x_odd;
            }
        }
    }
    
    if (VAR == VAR_DET) {
        smem_partials[warp * WARP_SIZE + lane] = acc;
        __syncwarp();
        if (lane == 0) {
            float sum = 0.0f;
            for(int i=0; i<32; i++) sum += smem_partials[warp * WARP_SIZE + i];
            y[row] = __float2half(sum);
        }
    } else if (VAR == VAR_FP64) {
        double acc_d = warp_reduce_sum_fp64((double)acc);
        if (lane == 0) y[row] = __float2half((float)acc_d);
    } else {
        acc = warp_reduce_sum(acc);
        if (lane == 0) y[row] = __float2half(acc);
    }
}


// ----------------------------------------------------------------------------
// 4b. Q6_K HPC KERNEL (warp per row, coalesced byte loads)
// GGML block_q6_K: ql[128] (low 4 bits), qh[64] (high 2 bits), int8 scales[16], fp16 d.
// Per 128-value half h: lane l produces values at offsets l, l+32, l+64, l+96.
// ----------------------------------------------------------------------------
#pragma pack(push, 1)
struct hpc_block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    half    d;
};
#pragma pack(pop)
static_assert(sizeof(hpc_block_q6_K) == 210, "K-Quant ABI Violation: Q6_K size mismatch!");

template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
void hpc_q6_K_gemv_kernel(const hpc_block_q6_K* __restrict__ W,
                          const half* __restrict__ x,
                          half* __restrict__ y,
                          int N, int K)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row  = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= N) return;

    const int nb = K / QK_K;
    const hpc_block_q6_K* __restrict__ w = W + (size_t)row * nb;
    float acc = 0.0f;

    for (int sb = 0; sb < nb; ++sb) {
        const float d = __half2float(w[sb].d);
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint8_t ql0 = w[sb].ql[h * 64 + lane];
            const uint8_t ql1 = w[sb].ql[h * 64 + lane + 32];
            const uint8_t qh  = w[sb].qh[h * 32 + lane];
            const int8_t* sc  = w[sb].scales + h * 8 + (lane >> 4);
            const int q1 = (int)((ql0 & 0x0F) | (((qh >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql1 & 0x0F) | (((qh >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql0 >> 4)   | (((qh >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql1 >> 4)   | (((qh >> 6) & 3) << 4)) - 32;
            const int base = sb * QK_K + h * 128 + lane;
            acc += d * (float)sc[0] * (float)q1 * __half2float(x[base]);
            acc += d * (float)sc[2] * (float)q2 * __half2float(x[base + 32]);
            acc += d * (float)sc[4] * (float)q3 * __half2float(x[base + 64]);
            acc += d * (float)sc[6] * (float)q4 * __half2float(x[base + 96]);
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[row] = __float2half(acc);
}

// ----------------------------------------------------------------------------
// 4c. INT8-ACTIVATION (Q8) GEMV KERNELS: dp4a integer dot products
// x is quantized once per call to int8 in blocks of 32 (xq) with one fp32
// scale per block (xd = amax / 127), the same activation format llama.cpp
// uses for its mmvq path. Weights are unpacked 4 at a time into int8 lanes
// and multiplied with __dp4a; per-group scales/mins are applied in fp32.
// Mapping: 16 lanes per superblock, so a warp works on 2 superblocks of one
// row at a time; each lane handles 16 weights per superblock.
// ----------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

// One warp per 32-element block.
__global__ void quantize_x_q8_kernel(const half* __restrict__ x, int8_t* __restrict__ xq,
                                     float* __restrict__ xd, int K) {
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk * 32 >= K) return;
    const float v = __half2float(x[blk * 32 + lane]);
    const float amax = warp_reduce_max(fabsf(v));
    const float d = amax / 127.0f;
    const float inv = (d > 0.0f) ? 1.0f / d : 0.0f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    xq[blk * 32 + lane] = (int8_t)q;
    if (lane == 0) xd[blk] = d;
}

// Epilogue / second-matrix options for the int8-activation GEMV kernels.
//   W2/y2/split: rows >= split come from W2 and go to y2 (row - split), so two
//                matrices with the same input and type run in one launch.
//   bias:  y += bias (fp32), with addBiasF32Kernel's NaN/clamp rules.
//   res:   y = res + y  (fp16), with addResidualKernel's rules; y may alias res.
struct Q8Epi {
    const uint8_t* W2 = nullptr;
    half* y2 = nullptr;
    int split = 0;
    const float* bias = nullptr;
    const half* res = nullptr;
};

__device__ __forceinline__ float clampHalfRange(float r) {
    if (isnan(r)) return 0.0f;
    return fminf(fmaxf(r, -65504.0f), 65504.0f);
}

__device__ __forceinline__ void storeQ8(half* y, int row, float acc, const Q8Epi& e) {
    half h = __float2half(acc);
    if (e.bias) {
        float t = __half2float(h), b = e.bias[row];
        if (isnan(t)) t = 0.0f;
        if (isnan(b)) b = 0.0f;
        h = __float2half(clampHalfRange(t + b));
    }
    if (e.res) {
        float t = __half2float(e.res[row]), a = __half2float(h);
        if (isnan(t)) t = 0.0f;
        if (isnan(a)) a = 0.0f;
        h = __float2half(clampHalfRange(t + a));
    }
    y[row] = h;
}

// 4-byte load from a 2-byte aligned address (Q3_K blocks are 110 bytes).
__device__ __forceinline__ uint32_t load_u32_a2(const uint8_t* p) {
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t)q[0] | ((uint32_t)q[1] << 16);
}

template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
void q2_K_q8_gemv_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                         const float* __restrict__ xd, half* __restrict__ y, int N, int K, Q8Epi e) {
    const int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    if (e.W2 && row >= e.split) { W = e.W2; y = e.y2; row -= e.split; }
    const int nb = K / QK_K;
    const uint8_t* __restrict__ wr = W + (size_t)row * nb * 84;
    const int hl = lane & 15, h = hl >> 3, m = hl & 7, c = m >> 2;
    float acc = 0.0f;
    for (int sb = lane >> 4; sb < nb; sb += 2) {
        const uint8_t* b = wr + sb * 84;
        const uint32_t q = *reinterpret_cast<const uint32_t*>(b + 16 + h * 32 + 4 * m);
        const uint32_t sc0 = *reinterpret_cast<const uint32_t*>(b + h * 8);
        const uint32_t sc1 = *reinterpret_cast<const uint32_t*>(b + h * 8 + 4);
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b + 80));
        const int8_t* xb = xq + sb * QK_K + h * 128 + 4 * m;
        const float4 d8 = *reinterpret_cast<const float4*>(xd + sb * 8 + h * 4);
        const float d8s[4] = {d8.x, d8.y, d8.z, d8.w};
        float sd = 0.0f, sm = 0.0f;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const uint32_t scb = ((s < 2 ? sc0 : sc1) >> (8 * ((s & 1) * 2 + c))) & 0xFF;
            const int v = (int)((q >> (2 * s)) & 0x03030303u);
            const int xv = *reinterpret_cast<const int*>(xb + s * 32);
            const int dot = __dp4a(v, xv, 0);
            const int xs = __dp4a(0x01010101, xv, 0);
            sd += d8s[s] * (float)(dot * (int)(scb & 0xF));
            sm += d8s[s] * (float)(xs * (int)(scb >> 4));
        }
        acc += dm.x * sd - dm.y * sm;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) storeQ8(y, row, acc, e);
}

template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
void q3_K_q8_gemv_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                         const float* __restrict__ xd, half* __restrict__ y, int N, int K, Q8Epi e) {
    const int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    if (e.W2 && row >= e.split) { W = e.W2; y = e.y2; row -= e.split; }
    const int nb = K / QK_K;
    const uint8_t* __restrict__ wr = W + (size_t)row * nb * 110;
    const int hl = lane & 15, h = hl >> 3, m = hl & 7, c = m >> 2;
    float acc = 0.0f;
    for (int sb = lane >> 4; sb < nb; sb += 2) {
        const uint8_t* b = wr + sb * 110;
        const uint32_t hm = load_u32_a2(b + 4 * m);
        const uint32_t q = load_u32_a2(b + 32 + h * 32 + 4 * m);
        const uint32_t a0 = load_u32_a2(b + 96), a1 = load_u32_a2(b + 100), a2 = load_u32_a2(b + 104);
        const float d = __half2float(*reinterpret_cast<const half*>(b + 108));
        const int8_t* xb = xq + sb * QK_K + h * 128 + 4 * m;
        const float* d8 = xd + sb * 8 + h * 4;
        float sd = 0.0f;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int li = 2 * s + c;   // scale j = 8h + li; low nibble in byte li (shift 4h), high bits in byte 8 + (li & 3)
            const uint32_t lb = ((li < 4 ? a0 : a1) >> (8 * (li & 3))) & 0xFF;
            const uint32_t hb = (a2 >> (8 * (li & 3))) & 0xFF;
            const int sc = (int)(((lb >> (4 * h)) & 0xF) | (((hb >> (2 * (2 * h + (s >> 1)))) & 3) << 4)) - 32;
            const uint32_t lo = (q >> (2 * s)) & 0x03030303u;
            const uint32_t hi = (hm >> (h * 4 + s)) & 0x01010101u;
            const int v = (int)(lo | (hi << 2));   // 0..7, weight value = v - 4
            const int xv = *reinterpret_cast<const int*>(xb + s * 32);
            const int dot = __dp4a(v, xv, 0) - 4 * __dp4a(0x01010101, xv, 0);
            sd += d8[s] * (float)(dot * sc);
        }
        acc += d * sd;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) storeQ8(y, row, acc, e);
}

template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
void q4_K_q8_gemv_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                         const float* __restrict__ xd, half* __restrict__ y, int N, int K, Q8Epi e) {
    const int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    if (e.W2 && row >= e.split) { W = e.W2; y = e.y2; row -= e.split; }
    const int nb = K / QK_K;
    const uint8_t* __restrict__ wr = W + (size_t)row * nb * 144;
    const int hl = lane & 15, j = hl >> 2, m = hl & 3;
    float acc = 0.0f;
    for (int sb = lane >> 4; sb < nb; sb += 2) {
        const uint8_t* b = wr + sb * 144;
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b));
        const uint8_t* sc = b + 4;
        const uint2 qq = *reinterpret_cast<const uint2*>(b + 16 + 32 * j + 8 * m);
        // get_scale_min_k4 for is = 2j (low nibbles) and 2j + 1 (high nibbles)
        int s_lo, m_lo, s_hi, m_hi;
        {
            const int is = 2 * j;
            if (is < 4) { s_lo = sc[is] & 63; m_lo = sc[is + 4] & 63; }
            else { s_lo = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m_lo = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
            const int is1 = is + 1;
            if (is1 < 4) { s_hi = sc[is1] & 63; m_hi = sc[is1 + 4] & 63; }
            else { s_hi = (sc[is1 + 4] & 0xF) | ((sc[is1 - 4] >> 6) << 4); m_hi = (sc[is1 + 4] >> 4) | ((sc[is1] >> 6) << 4); }
        }
        const int8_t* xb = xq + sb * QK_K + 64 * j + 8 * m;
        const int2 xl = *reinterpret_cast<const int2*>(xb);
        const int2 xh = *reinterpret_cast<const int2*>(xb + 32);
        const float2 d8 = *reinterpret_cast<const float2*>(xd + sb * 8 + 2 * j);
        const int l0 = (int)(qq.x & 0x0F0F0F0Fu), l1 = (int)(qq.y & 0x0F0F0F0Fu);
        const int h0 = (int)((qq.x >> 4) & 0x0F0F0F0Fu), h1 = (int)((qq.y >> 4) & 0x0F0F0F0Fu);
        const int dot_lo = __dp4a(l1, xl.y, __dp4a(l0, xl.x, 0));
        const int dot_hi = __dp4a(h1, xh.y, __dp4a(h0, xh.x, 0));
        const int xs_lo = __dp4a(0x01010101, xl.y, __dp4a(0x01010101, xl.x, 0));
        const int xs_hi = __dp4a(0x01010101, xh.y, __dp4a(0x01010101, xh.x, 0));
        const float sd = d8.x * (float)(dot_lo * s_lo) + d8.y * (float)(dot_hi * s_hi);
        const float sm = d8.x * (float)(xs_lo * m_lo) + d8.y * (float)(xs_hi * m_hi);
        acc += dm.x * sd - dm.y * sm;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) storeQ8(y, row, acc, e);
}

// Q6_K x int8: ql[128] qh[64] int8 scales[16] fp16 d (210 B, 2-byte aligned).
// Lane mapping as Q2_K/Q3_K: h = half, m = 4-byte column; per p = 0..3 the
// lane's 4 values sit at h*128 + 32p + 4m, scale h*8 + 2p + c.
template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
void q6_K_q8_gemv_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                         const float* __restrict__ xd, half* __restrict__ y, int N, int K, Q8Epi e) {
    const int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    if (e.W2 && row >= e.split) { W = e.W2; y = e.y2; row -= e.split; }
    const int nb = K / QK_K;
    const uint8_t* __restrict__ wr = W + (size_t)row * nb * 210;
    const int hl = lane & 15, h = hl >> 3, m = hl & 7, c = m >> 2;
    float acc = 0.0f;
    for (int sb = lane >> 4; sb < nb; sb += 2) {
        const uint8_t* b = wr + sb * 210;
        const uint32_t ql0 = load_u32_a2(b + h * 64 + 4 * m);
        const uint32_t ql1 = load_u32_a2(b + h * 64 + 32 + 4 * m);
        const uint32_t qh = load_u32_a2(b + 128 + h * 32 + 4 * m);
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + h * 8 + c;
        const float d = __half2float(*reinterpret_cast<const half*>(b + 208));
        const int8_t* xb = xq + sb * QK_K + h * 128 + 4 * m;
        const float* d8 = xd + sb * 8 + h * 4;
        float sd = 0.0f;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t qlw = (p & 1) ? ql1 : ql0;
            const uint32_t lo = (p < 2 ? qlw : (qlw >> 4)) & 0x0F0F0F0Fu;
            const int v = (int)(lo | (((qh >> (2 * p)) & 0x03030303u) << 4));  // q + 32, 0..63
            const int xv = *reinterpret_cast<const int*>(xb + p * 32);
            const int dot = __dp4a(v, xv, 0) - 32 * __dp4a(0x01010101, xv, 0);
            sd += d8[p] * (float)(dot * (int)sc[2 * p]);
        }
        acc += d * sd;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) storeQ8(y, row, acc, e);
}

// Q5_K x int8: d dmin scales[12] qh[32] qs[128] (176 B, 16-byte aligned).
// Chunk j (64 values): low nibbles + bit 2j of qh -> values 64j + l, high
// nibbles + bit 2j+1 -> 64j + 32 + l. Lane mapping as Q4_K.
template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
void q5_K_q8_gemv_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                         const float* __restrict__ xd, half* __restrict__ y, int N, int K, Q8Epi e) {
    const int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    if (e.W2 && row >= e.split) { W = e.W2; y = e.y2; row -= e.split; }
    const int nb = K / QK_K;
    const uint8_t* __restrict__ wr = W + (size_t)row * nb * 176;
    const int hl = lane & 15, j = hl >> 2, m = hl & 3;
    float acc = 0.0f;
    for (int sb = lane >> 4; sb < nb; sb += 2) {
        const uint8_t* b = wr + sb * 176;
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b));
        const uint8_t* sc = b + 4;
        const uint2 qh = *reinterpret_cast<const uint2*>(b + 16 + 8 * m);
        const uint2 qq = *reinterpret_cast<const uint2*>(b + 48 + 32 * j + 8 * m);
        int s_lo, m_lo, s_hi, m_hi;
        {
            const int is = 2 * j;
            if (is < 4) { s_lo = sc[is] & 63; m_lo = sc[is + 4] & 63; }
            else { s_lo = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m_lo = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
            const int is1 = is + 1;
            if (is1 < 4) { s_hi = sc[is1] & 63; m_hi = sc[is1 + 4] & 63; }
            else { s_hi = (sc[is1 + 4] & 0xF) | ((sc[is1 - 4] >> 6) << 4); m_hi = (sc[is1 + 4] >> 4) | ((sc[is1] >> 6) << 4); }
        }
        const int8_t* xb = xq + sb * QK_K + 64 * j + 8 * m;
        const int2 xl = *reinterpret_cast<const int2*>(xb);
        const int2 xh = *reinterpret_cast<const int2*>(xb + 32);
        const float2 d8 = *reinterpret_cast<const float2*>(xd + sb * 8 + 2 * j);
        const int l0 = (int)((qq.x & 0x0F0F0F0Fu) | (((qh.x >> (2 * j)) & 0x01010101u) << 4));
        const int l1 = (int)((qq.y & 0x0F0F0F0Fu) | (((qh.y >> (2 * j)) & 0x01010101u) << 4));
        const int h0 = (int)(((qq.x >> 4) & 0x0F0F0F0Fu) | (((qh.x >> (2 * j + 1)) & 0x01010101u) << 4));
        const int h1 = (int)(((qq.y >> 4) & 0x0F0F0F0Fu) | (((qh.y >> (2 * j + 1)) & 0x01010101u) << 4));
        const int dot_lo = __dp4a(l1, xl.y, __dp4a(l0, xl.x, 0));
        const int dot_hi = __dp4a(h1, xh.y, __dp4a(h0, xh.x, 0));
        const int xs_lo = __dp4a(0x01010101, xl.y, __dp4a(0x01010101, xl.x, 0));
        const int xs_hi = __dp4a(0x01010101, xh.y, __dp4a(0x01010101, xh.x, 0));
        const float sd = d8.x * (float)(dot_lo * s_lo) + d8.y * (float)(dot_hi * s_hi);
        const float sm = d8.x * (float)(xs_lo * m_lo) + d8.y * (float)(xs_hi * m_hi);
        acc += dm.x * sd - dm.y * sm;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) storeQ8(y, row, acc, e);
}

// Q8_0 x int8: fp16 d + int8 qs[32] (34 B, 2-byte aligned). 4 lanes per block,
// 8 values each; a warp covers 8 blocks per iteration.
template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
void q8_0_q8_gemv_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                         const float* __restrict__ xd, half* __restrict__ y, int N, int K, Q8Epi e) {
    const int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    if (e.W2 && row >= e.split) { W = e.W2; y = e.y2; row -= e.split; }
    const int nblk = K / 32;
    const uint8_t* __restrict__ wr = W + (size_t)row * nblk * 34;
    const int part = lane & 3;
    float acc = 0.0f;
    for (int bk = lane >> 2; bk < nblk; bk += 8) {
        const uint8_t* b = wr + bk * 34;
        const float d = __half2float(*reinterpret_cast<const half*>(b));
        const int w0 = (int)load_u32_a2(b + 2 + 8 * part), w1 = (int)load_u32_a2(b + 6 + 8 * part);
        const int2 xv = *reinterpret_cast<const int2*>(xq + bk * 32 + 8 * part);
        acc += d * xd[bk] * (float)__dp4a(w1, xv.y, __dp4a(w0, xv.x, 0));
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) storeQ8(y, row, acc, e);
}

// ----------------------------------------------------------------------------
// 4d. FUSED PRODUCERS OF THE INT8 ACTIVATION
// Same operations, in the same order, as the separate kernels they replace
// (rmsNormKernel / geluKernel / elementWiseMulKernel + quantize_x_q8_kernel),
// so results are bit-identical; they only save launches and a round trip.
// ----------------------------------------------------------------------------

// One block of 256 threads (as rmsNormKernel). Shared: 256 floats + K halves.
__global__ void rmsnorm_quant_q8_kernel(const half* __restrict__ x, const float* __restrict__ w,
                                        int8_t* __restrict__ xq, float* __restrict__ xd, int K, float eps) {
    extern __shared__ float sm[];
    float* red = sm;
    half* xn = reinterpret_cast<half*>(sm + blockDim.x);
    const int tid = threadIdx.x, bs = blockDim.x;
    float sum_sq = 0.0f;
    for (int i = tid; i < K; i += bs) {
        const float val = __half2float(x[i]);
        if (!isnan(val) && !isinf(val)) sum_sq += val * val;
    }
    red[tid] = sum_sq;
    __syncthreads();
    for (int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    const float mean_sq = red[0] / (float)K;
    const float rms = sqrtf(mean_sq + eps);
    const float rms_inv = (rms > 1e-12f) ? (1.0f / rms) : 1.0f;
    for (int i = tid; i < K; i += bs) {
        float val = __half2float(x[i]);
        if (isnan(val) || isinf(val)) val = 0.0f;
        float res = val * rms_inv * w[i];
        if (isnan(res)) res = 0.0f;
        else if (res > 65504.0f) res = 65504.0f;
        else if (res < -65504.0f) res = -65504.0f;
        xn[i] = __float2half(res);
    }
    __syncthreads();
    const int lane = tid & 31, warp = tid >> 5, nwarps = bs >> 5;
    for (int blk = warp; blk < K / 32; blk += nwarps) {
        const float v = __half2float(xn[blk * 32 + lane]);
        const float amax = warp_reduce_max(fabsf(v));
        const float d = amax / 127.0f;
        const float inv = (d > 0.0f) ? 1.0f / d : 0.0f;
        int q = __float2int_rn(v * inv);
        q = max(-127, min(127, q));
        xq[blk * 32 + lane] = (int8_t)q;
        if (lane == 0) xd[blk] = d;
    }
}

// One warp per 32-value block: SiLU(gate) * up, both stored as fp16 in between, then quantized.
__global__ void silu_mul_quant_q8_kernel(const half* __restrict__ gate, const half* __restrict__ up,
                                         int8_t* __restrict__ xq, float* __restrict__ xd, int K) {
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk * 32 >= K) return;
    const int i = blk * 32 + lane;
    // geluKernel (SiLU)
    const half g = gate[i];
    const float x = __half2float(g);
    half a;
    if (isnan(x)) a = __float2half(0.0f);
    else if (x > 20.0f) a = g;
    else if (x < -20.0f) a = __float2half(0.0f);
    else {
        const float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float res = x * sigmoid_x;
        if (isnan(res)) res = 0.0f;
        else if (res > 65504.0f) res = 65504.0f;
        else if (res < -65504.0f) res = -65504.0f;
        a = __float2half(res);
    }
    // elementWiseMulKernel
    float af = __half2float(a), bf = __half2float(up[i]);
    if (isnan(af)) af = 0.0f;
    if (isnan(bf)) bf = 0.0f;
    float m = af * bf;
    if (isnan(m)) m = 0.0f;
    else if (m > 65504.0f) m = 65504.0f;
    else if (m < -65504.0f) m = -65504.0f;
    // quantize_x_q8_kernel
    const float v = __half2float(__float2half(m));
    const float amax = warp_reduce_max(fabsf(v));
    const float d = amax / 127.0f;
    const float inv = (d > 0.0f) ? 1.0f / d : 0.0f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    xq[i] = (int8_t)q;
    if (lane == 0) xd[blk] = d;
}

// ----------------------------------------------------------------------------
// 5. HOST LAUNCHERS & TELEMETRY INFRASTRUCTURE
// ----------------------------------------------------------------------------
namespace friday_hpc {

constexpr int kWarpsPerBlock = 8;
constexpr int kQ8WarpsPerBlock = 4;

// FRIDAY_GEMV_FP32=1 selects the previous fp32-activation kernels (A/B testing).
static bool useFp32Gemv() {
    static const bool v = [] { const char* e = std::getenv("FRIDAY_GEMV_FP32"); return e && e[0] == '1'; }();
    return v;
}

// Scratch for the int8 activation. Grows on demand (first calls only); all GEMVs
// run on the same compute stream, so reuse between consecutive calls is ordered.
static int8_t* g_xq = nullptr;
static float* g_xd = nullptr;
static int g_xq_cap = 0;

static void quantizeX(const half* x, int K, cudaStream_t s) {
    if (K % 32 != 0) { std::fprintf(stderr, "GEMV: K=%d is not a multiple of 32\n", K); std::abort(); }
    if (K > g_xq_cap) {
        cudaStreamSynchronize(s);
        if (g_xq) { cudaFree(g_xq); cudaFree(g_xd); }
        if (cudaMalloc(&g_xq, K) != cudaSuccess || cudaMalloc(&g_xd, (K / 32) * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "GEMV: cannot allocate activation scratch\n"); std::abort();
        }
        g_xq_cap = K;
    }
    const int blocks32 = K / 32, warps = 8;
    quantize_x_q8_kernel<<<(blocks32 + warps - 1) / warps, warps * WARP_SIZE, 0, s>>>(x, g_xq, g_xd, K);
}

template <void (*KERNEL)(const uint8_t*, const int8_t*, const float*, half*, int, int, Q8Epi)>
static void launchQ8(const half* x, const void* W, half* y, int K, int N, cudaStream_t s) {
    quantizeX(x, K, s);
    const dim3 grid((N + kQ8WarpsPerBlock - 1) / kQ8WarpsPerBlock);
    KERNEL<<<grid, kQ8WarpsPerBlock * WARP_SIZE, 0, s>>>(static_cast<const uint8_t*>(W), g_xq, g_xd, y, N, K, Q8Epi());
}

void launch_q2_K_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s) {
    if (!useFp32Gemv()) { launchQ8<q2_K_q8_gemv_kernel<kQ8WarpsPerBlock>>(x, W, y, K, N, s); return; }
    const dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const dim3 block(kWarpsPerBlock * WARP_SIZE);
    switch (g_reduction_variant) {
        case VAR_FMA:   hpc_q2_K_gemv_kernel<kWarpsPerBlock, VAR_FMA><<<grid, block, 0, s>>>(static_cast<const block_q2_K*>(W), x, y, N, K); break;
        case VAR_DET:   hpc_q2_K_gemv_kernel<kWarpsPerBlock, VAR_DET><<<grid, block, 0, s>>>(static_cast<const block_q2_K*>(W), x, y, N, K); break;
        case VAR_FP64:  hpc_q2_K_gemv_kernel<kWarpsPerBlock, VAR_FP64><<<grid, block, 0, s>>>(static_cast<const block_q2_K*>(W), x, y, N, K); break;
        case VAR_KAHAN: hpc_q2_K_gemv_kernel<kWarpsPerBlock, VAR_KAHAN><<<grid, block, 0, s>>>(static_cast<const block_q2_K*>(W), x, y, N, K); break;
        default:        hpc_q2_K_gemv_kernel<kWarpsPerBlock, VAR_CURRENT><<<grid, block, 0, s>>>(static_cast<const block_q2_K*>(W), x, y, N, K); break;
    }
}

void launch_q3_K_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s) {
    if (!useFp32Gemv()) { launchQ8<q3_K_q8_gemv_kernel<kQ8WarpsPerBlock>>(x, W, y, K, N, s); return; }
    const dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const dim3 block(kWarpsPerBlock * WARP_SIZE);
    switch (g_reduction_variant) {
        case VAR_FMA:   hpc_q3_K_gemv_kernel<kWarpsPerBlock, VAR_FMA><<<grid, block, 0, s>>>(static_cast<const block_q3_K*>(W), x, y, N, K); break;
        case VAR_DET:   hpc_q3_K_gemv_kernel<kWarpsPerBlock, VAR_DET><<<grid, block, 0, s>>>(static_cast<const block_q3_K*>(W), x, y, N, K); break;
        case VAR_FP64:  hpc_q3_K_gemv_kernel<kWarpsPerBlock, VAR_FP64><<<grid, block, 0, s>>>(static_cast<const block_q3_K*>(W), x, y, N, K); break;
        case VAR_KAHAN: hpc_q3_K_gemv_kernel<kWarpsPerBlock, VAR_KAHAN><<<grid, block, 0, s>>>(static_cast<const block_q3_K*>(W), x, y, N, K); break;
        default:        hpc_q3_K_gemv_kernel<kWarpsPerBlock, VAR_CURRENT><<<grid, block, 0, s>>>(static_cast<const block_q3_K*>(W), x, y, N, K); break;
    }
}

void launch_q4_K_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s) {
    if (!useFp32Gemv()) { launchQ8<q4_K_q8_gemv_kernel<kQ8WarpsPerBlock>>(x, W, y, K, N, s); return; }
    const dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const dim3 block(kWarpsPerBlock * WARP_SIZE);
    switch (g_reduction_variant) {
        case VAR_FMA:   hpc_q4_K_gemv_kernel<kWarpsPerBlock, VAR_FMA><<<grid, block, 0, s>>>(static_cast<const block_q4_K*>(W), x, y, N, K); break;
        case VAR_DET:   hpc_q4_K_gemv_kernel<kWarpsPerBlock, VAR_DET><<<grid, block, 0, s>>>(static_cast<const block_q4_K*>(W), x, y, N, K); break;
        case VAR_FP64:  hpc_q4_K_gemv_kernel<kWarpsPerBlock, VAR_FP64><<<grid, block, 0, s>>>(static_cast<const block_q4_K*>(W), x, y, N, K); break;
        case VAR_KAHAN: hpc_q4_K_gemv_kernel<kWarpsPerBlock, VAR_KAHAN><<<grid, block, 0, s>>>(static_cast<const block_q4_K*>(W), x, y, N, K); break;
        default:        hpc_q4_K_gemv_kernel<kWarpsPerBlock, VAR_CURRENT><<<grid, block, 0, s>>>(static_cast<const block_q4_K*>(W), x, y, N, K); break;
    }
}

void launch_q6_K_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s) {
    if (!useFp32Gemv()) { launchQ8<q6_K_q8_gemv_kernel<kQ8WarpsPerBlock>>(x, W, y, K, N, s); return; }
    const dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const dim3 block(kWarpsPerBlock * WARP_SIZE);
    hpc_q6_K_gemv_kernel<kWarpsPerBlock><<<grid, block, 0, s>>>(static_cast<const hpc_block_q6_K*>(W), x, y, N, K);
}


// ---------------- int8-activation pipeline API (see hpc_k_quants.h) ----------------
void ensureQ8Scratch(int K, cudaStream_t s) {
    if (K % 32 != 0) { std::fprintf(stderr, "GEMV: K=%d is not a multiple of 32\n", K); std::abort(); }
    if (K <= g_xq_cap) return;
    cudaStreamSynchronize(s);
    if (g_xq) { cudaFree(g_xq); cudaFree(g_xd); }
    if (cudaMalloc(&g_xq, K) != cudaSuccess || cudaMalloc(&g_xd, (K / 32) * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "GEMV: cannot allocate activation scratch\n"); std::abort();
    }
    g_xq_cap = K;
}

void quantizeQ8(const half* x, int K, cudaStream_t s) { quantizeX(x, K, s); }

void rmsNormQuantQ8(const half* x, const float* w, int K, float eps, cudaStream_t s) {
    ensureQ8Scratch(K, s);
    const int threads = 256;
    const size_t smem = threads * sizeof(float) + (size_t)K * sizeof(half);
    rmsnorm_quant_q8_kernel<<<1, threads, smem, s>>>(x, w, g_xq, g_xd, K, eps);
}

void siluMulQuantQ8(const half* gate, const half* up, int K, cudaStream_t s) {
    ensureQ8Scratch(K, s);
    const int warps = 8;
    silu_mul_quant_q8_kernel<<<(K / 32 + warps - 1) / warps, warps * WARP_SIZE, 0, s>>>(gate, up, g_xq, g_xd, K);
}

bool gemvQ8Supported(Q8Kind k) { return k != Q8Kind::None; }

void gemvQ8(Q8Kind kind, const void* W, half* y, int K, int N, cudaStream_t s, const void* W2, half* y2,
            const float* bias, const half* res) {
    Q8Epi e;
    int rows = N;
    if (W2) { e.W2 = static_cast<const uint8_t*>(W2); e.y2 = y2; e.split = N; rows = 2 * N; }
    e.bias = bias;
    e.res = res;
    const dim3 grid((rows + kQ8WarpsPerBlock - 1) / kQ8WarpsPerBlock), block(kQ8WarpsPerBlock * WARP_SIZE);
    const uint8_t* w = static_cast<const uint8_t*>(W);
    switch (kind) {
        case Q8Kind::Q2_K: q2_K_q8_gemv_kernel<kQ8WarpsPerBlock><<<grid, block, 0, s>>>(w, g_xq, g_xd, y, rows, K, e); break;
        case Q8Kind::Q3_K: q3_K_q8_gemv_kernel<kQ8WarpsPerBlock><<<grid, block, 0, s>>>(w, g_xq, g_xd, y, rows, K, e); break;
        case Q8Kind::Q4_K: q4_K_q8_gemv_kernel<kQ8WarpsPerBlock><<<grid, block, 0, s>>>(w, g_xq, g_xd, y, rows, K, e); break;
        case Q8Kind::Q5_K: q5_K_q8_gemv_kernel<kQ8WarpsPerBlock><<<grid, block, 0, s>>>(w, g_xq, g_xd, y, rows, K, e); break;
        case Q8Kind::Q6_K: q6_K_q8_gemv_kernel<kQ8WarpsPerBlock><<<grid, block, 0, s>>>(w, g_xq, g_xd, y, rows, K, e); break;
        case Q8Kind::Q8_0: q8_0_q8_gemv_kernel<kQ8WarpsPerBlock><<<grid, block, 0, s>>>(w, g_xq, g_xd, y, rows, K, e); break;
        default: std::fprintf(stderr, "gemvQ8: unsupported type\n"); std::abort();
    }
}

void launch_q5_K_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s) {
    launchQ8<q5_K_q8_gemv_kernel<kQ8WarpsPerBlock>>(x, W, y, K, N, s);
}

void launch_q8_0_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s) {
    launchQ8<q8_0_q8_gemv_kernel<kQ8WarpsPerBlock>>(x, W, y, K, N, s);
}

} // namespace friday_hpc
