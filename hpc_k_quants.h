#ifndef HPC_K_QUANTS_H
#define HPC_K_QUANTS_H

#include <cuda_fp16.h>
#include <cstdint>

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

#include <cuda_runtime.h>
#include <string>

// Int8-activation GEMV pipeline. One activation scratch (int8 + per-32 scales)
// is shared by consecutive calls on the same stream:
//   rmsNormQuantQ8 / siluMulQuantQ8 / quantizeQ8  -> fill the scratch
//   gemvQ8 (any number of times)                  -> read it
namespace friday_hpc {

enum class Q8Kind { None, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 };

inline Q8Kind q8KindFromDtype(const std::string& d) {
    if (d == "Q2_K") return Q8Kind::Q2_K;
    if (d == "Q3_K") return Q8Kind::Q3_K;
    if (d == "Q4_K") return Q8Kind::Q4_K;
    if (d == "Q5_K") return Q8Kind::Q5_K;
    if (d == "Q6_K") return Q8Kind::Q6_K;
    if (d == "Q8_0") return Q8Kind::Q8_0;
    return Q8Kind::None;
}

// Allocate the scratch for activations of up to K values (call before CUDA Graph capture).
void ensureQ8Scratch(int K, cudaStream_t s);
void quantizeQ8(const half* x, int K, cudaStream_t s);
// RMSNorm(x) * w, rounded to fp16, quantized (same math as rmsNormKernel + quantize).
void rmsNormQuantQ8(const half* x, const float* w, int K, float eps, cudaStream_t s);
// SiLU(gate) * up with fp16 rounding at the same points as geluKernel + elementWiseMulKernel, quantized.
void siluMulQuantQ8(const half* gate, const half* up, int K, cudaStream_t s);
bool gemvQ8Supported(Q8Kind k);
// y = W x (x from the scratch). Optional: W2 -> y2 (same shape and type, one launch),
// bias (y += bias, fp32), res (y = res + y, fp16; y may alias res).
void gemvQ8(Q8Kind kind, const void* W, half* y, int K, int N, cudaStream_t s, const void* W2 = nullptr,
            half* y2 = nullptr, const float* bias = nullptr, const half* res = nullptr);

}  // namespace friday_hpc

#endif // HPC_K_QUANTS_H
