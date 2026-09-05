#include "kernel_dispatcher.h"
#include "cuda_math_kernels.h"
// We avoid directly including .cu files to prevent multiple definition errors during linking.
// The int4_kernel.cu and cuda_math_kernels.cu will be compiled separately.

// Declare the external functions
// void launchInt4GemvKernel(const half*, const unsigned char*, const half*, const unsigned char*, half*, int, int, cudaStream_t);
// Assume we have some wrappers in cuda_math_kernels.cu if needed, but for now we'll stub the FP16 launch or invoke a known kernel wrapper
extern void launchTernaryGemvKernel(const half* d_act, const unsigned char* d_weights, half* d_out, int K, int N, cudaStream_t stream);
extern void launchGemvFP16Wrapper(const half* d_weights, const half* d_act, half* d_out, int K, int N, cudaStream_t stream);


void FridayKernelDispatcher::dispatchGemvKernel(
    QuantType quant_type,
    const half* d_activations,
    const void* d_weights,
    const half* d_scales,
    const unsigned char* d_zero_points,
    half* d_output,
    int K,
    int N,
    cudaStream_t stream)
{
    switch (quant_type) {
        case FP16: {
            // launchGemvFP16Wrapper(static_cast<const half*>(d_weights), d_activations, d_output, K, N, stream);
            break;
        }
        case INT8: {
            launchQ8_0GemvKernel(
                d_activations,
                static_cast<const block_q8_0*>(d_weights),
                d_output,
                K,
                N,
                stream
            );
            break;
        }
        case INT4: {
            launchQ4_0GemvKernel(
                d_activations,
                static_cast<const block_q4_0*>(d_weights),
                d_output,
                K,
                N,
                stream
            );
            break;
        }
        case TERNARY: {
            // launchTernaryGemvKernel(d_activations, static_cast<const unsigned char*>(d_weights), d_output, K, N, stream);
            break;
        }
        case Q4_1: {
            launchQ4_1GemvKernel(
                d_activations,
                static_cast<const block_q4_1*>(d_weights),
                d_output,
                K,
                N,
                stream
            );
            break;
        }
        case Q6_K: {
            launchQ6_KGemvKernel(
                d_activations,
                static_cast<const block_q6_K*>(d_weights),
                d_output,
                K,
                N,
                stream
            );
            break;
        }
        case UNKNOWN:
        default: {
            fprintf(stderr, "Error: Bilinmeyen veya desteklenmeyen nicelendirme tipi: %d\n", quant_type);
            exit(EXIT_FAILURE);
        }
    }
}
