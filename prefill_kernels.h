#ifndef FRIDAY_PREFILL_KERNELS_H
#define FRIDAY_PREFILL_KERNELS_H

// Batched (prefill) kernels: several tokens per weight read.
//
// gemmQ8: Y[t][n] = sum_k W[n][k] * x_t[k] for T tokens, with W in a GGUF
// K-quant / Q8_0 layout and x in the engine's int8 activation format (blocks of
// 32, fp32 scale; see friday_hpc::quantizeQ8Rows). Int8 tensor cores
// (mma m16n8k16) compute exact integer dot products per 16 values, the same
// integers the dp4a GEMV forms, so a column of Y matches gemvQ8 on that token up
// to the order of the fp32 additions. Needs compute capability 8.0+.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "hpc_k_quants.h"

namespace friday_prefill {

// True if the current device and the weight type/shape are supported.
bool gemmQ8Supported(friday_hpc::Q8Kind kind, int K);

// xq [T][K] int8, xd [T][K / 32]; Y row t at Y + t * ldy (N values).
// Epilogue per element as gemvQ8: fp16 rounding, then + bias[n] (fp32), then
// res[t * ldy + n] + y (fp16); Y may alias res.
void gemmQ8(friday_hpc::Q8Kind kind, const void* W, const int8_t* xq, const float* xd, int T, int K, int N, half* Y,
            int ldy, cudaStream_t s, const float* bias = nullptr, const half* res = nullptr);

}  // namespace friday_prefill

#endif  // FRIDAY_PREFILL_KERNELS_H
