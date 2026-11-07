// ============================================================================
// B KOLU / FAZ 2 - HPC K-QUANT ABI KERNEL MİMARİSİ
// Optimize edilmiş Q2_K, Q3_K, Q4_K Warp-Coalesced GEMV Çekirdekleri
// Doğrulama Kapısı (Validation Gate) Entegrasyonu ile.
// ============================================================================

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#define QK_K 256
#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFFu

// ----------------------------------------------------------------------------
// 1. K-QUANT ABI INVARIANTS & STRUCTS
// GGML Birebir Referans Layout'u
// ----------------------------------------------------------------------------
#pragma pack(push, 1)
struct block_q2_K {
    uint8_t scales[16];
    uint8_t qs[64];
    half d;
    half dmin;
};

struct block_q3_K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
#pragma pack(pop)

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
// 5. HOST LAUNCHERS & TELEMETRY INFRASTRUCTURE
// ----------------------------------------------------------------------------
namespace friday_hpc {

constexpr int kWarpsPerBlock = 8;

void launch_q2_K_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s) {
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

} // namespace friday_hpc
