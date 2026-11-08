#ifndef FRIDAY_CPU_OPS_H
#define FRIDAY_CPU_OPS_H

// Host implementations of the per-layer elementwise ops, attention, and the
// fp16 rounding points used by the GPU kernels (cuda_math_kernels.cu,
// attention_kernels.cu), so CPU-computed layers follow the same math.

#include "cpu_threadpool.h"

namespace friday_cpu {

// out = round_half(x * rsqrt(mean(x^2) + eps) * w)
void rmsNorm(const float* x, const float* w, float* out, int n, float eps);

// RoPE on q (nq heads) and k (nk heads), same math as ropeKernel:
// neox: pairs (i, i + d/2), else (2i, 2i + 1); theta_i = pos / base^(2i/d) / freq_factors[i].
void rope(float* q, float* k, int head_dim, int nq, int nk, int pos, float freq_base, bool neox,
          const float* freq_factors);

// Single-query GQA attention over positions [0, seq). k/v: [seq][nkv * hd].
void decodeAttention(const float* q, const float* k_cache, const float* v_cache, float* out, int seq,
                     int num_heads, int num_kv_heads, int head_dim, ThreadPool& pool);

// gate = round_half(round_half(silu(gate)) * up)
void siluMul(float* gate, const float* up, int n);

// x = round_half(x + a)
void addResidual(float* x, const float* a, int n);

}  // namespace friday_cpu

#endif  // FRIDAY_CPU_OPS_H
