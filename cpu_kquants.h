#ifndef FRIDAY_CPU_KQUANTS_H
#define FRIDAY_CPU_KQUANTS_H

// CPU K-quant GEMV for decode (one activation vector). Activations are
// quantized exactly like the GPU dp4a path: int8 in blocks of 32 with one fp32
// scale per block (d = amax / 127, round-half-even), so CPU and GPU layers
// compute the same math up to fp32 summation order.
//
// Weights use the GGML block layouts (Q2_K 84 B, Q3_K 110 B, Q4_K 144 B per 256
// values) and are read in place (e.g. from the pinned host layer cache).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace friday_cpu {

enum class KQuant { Q2_K, Q3_K, Q4_K };

struct Q8Activation {
    int K = 0;
    std::vector<int8_t> q;        // [K]
    std::vector<float> d;         // [K / 32]
    std::vector<int16_t> bsum16;  // [K / 16] sum of q over each 16 values
};

float halfToFloat(uint16_t h);
uint16_t floatToHalf(float f);

void quantizeActivation(const float* x, int K, Q8Activation& out);

// Round to fp16 precision in place; NaN -> 0 and clamp to +-65504 like the GPU kernels.
void roundToHalf(float* x, int n);

size_t rowBytes(KQuant t, int K);

// y[r] = dot(W row r, x) for r in [r0, r1). K must be a multiple of 256.
void gemvRows(KQuant t, const uint8_t* W, const Q8Activation& x, float* y, int r0, int r1);

// Portable scalar reference (used by tests and on CPUs without AVX2).
void gemvRowsScalar(KQuant t, const uint8_t* W, const Q8Activation& x, float* y, int r0, int r1);

bool hasAvx2();

}  // namespace friday_cpu

#endif  // FRIDAY_CPU_KQUANTS_H
