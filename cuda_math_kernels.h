#ifndef CUDA_MATH_KERNELS_H
#define CUDA_MATH_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Macro to check for CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define QK_K 256

#pragma pack(push, 1)
// GGUF Q4_0 block structure definition
struct block_q4_0 {
    half d;         // quantization scale (FP16)
    uint8_t qs[16]; // 16 bytes, each storing 2 4-bit quantized values
};

// GGUF Q4_1 block structure definition (32 weights, 20 bytes)
struct block_q4_1 {
    half d;         // quantization scale (FP16)
    half m;         // quantization min (FP16)
    uint8_t qs[16]; // 16 bytes, each storing 2 4-bit quantized values
};

struct block_q8_0 {
    half d;        // quantization scale (FP16)
    int8_t qs[32]; // 32 bytes, each storing 1 8-bit quantized value
};

// GGUF Q6_K block structure definition (256 weights, 210 bytes)
struct block_q6_K {
    uint8_t ql[QK_K / 2];      // 128 bytes (quants, lower 4 bits)
    uint8_t qh[QK_K / 4];      // 64 bytes (quants, upper 2 bits)
    int8_t  scales[QK_K / 16]; // 16 bytes (scales)
    half    d;                 // 2 bytes (super-block scale FP16)
};
#pragma pack(pop)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Host-side wrapper function to launch the GGUF Q4_0 GEMV kernel.
 */
void launchQ4_0GemvKernel(
    const half* d_activations,
    const block_q4_0* d_quantized_weights,
    half* d_output,
    int K,
    int N,
    cudaStream_t stream);

/**
 * @brief Host-side wrapper function to launch the GGUF Q4_1 GEMV kernel.
 */
void launchQ4_1GemvKernel(
    const half* d_activations,
    const block_q4_1* d_quantized_weights,
    half* d_output,
    int K,
    int N,
    cudaStream_t stream);

/**
 * @brief Host-side wrapper function to launch the GGUF Q8_0 GEMV kernel.
 */
void launchQ8_0GemvKernel(
    const half* d_activations,
    const block_q8_0* d_quantized_weights,
    half* d_output,
    int K,
    int N,
    cudaStream_t stream);

/**
 * @brief Host-side wrapper function to launch the GGUF Q6_K GEMV kernel.
 */
void launchQ6_KGemvKernel(
    const half* d_activations,
    const block_q6_K* d_quantized_weights,
    half* d_output,
    int K,
    int N,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // CUDA_MATH_KERNELS_H
