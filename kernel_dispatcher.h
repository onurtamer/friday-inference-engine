#ifndef FRIDAY_KERNEL_DISPATCHER_H
#define FRIDAY_KERNEL_DISPATCHER_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                \
    do {                                                \
        cudaError_t err = call;                         \
        if (err != cudaSuccess) {                       \
            fprintf(stderr, "CUDA error at %s:%d: %s: %s\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)
#endif

enum QuantType {
    FP16,
    INT8,
    INT4,
    TERNARY,
    Q4_1,
    Q6_K,
    UNKNOWN
};

class FridayKernelDispatcher {
public:
    static void dispatchGemvKernel(
        QuantType quant_type,
        const half* d_activations,
        const void* d_weights,
        const half* d_scales,
        const unsigned char* d_zero_points,
        half* d_output,
        int K,
        int N,
        cudaStream_t stream = 0
    );

private:
    FridayKernelDispatcher() = delete;
    FridayKernelDispatcher(const FridayKernelDispatcher&) = delete;
    FridayKernelDispatcher& operator=(const FridayKernelDispatcher&) = delete;
};

// Declarations for kernels
void launchInt4GemvKernel(const half* d_activations, const unsigned char* d_packed_int4_weights, const half* d_scales, const unsigned char* d_zero_points, half* d_output, int K, int N, cudaStream_t stream);

#endif // FRIDAY_KERNEL_DISPATCHER_H
