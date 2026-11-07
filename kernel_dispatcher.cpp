#include "kernel_dispatcher.h"
#include "cuda_math_kernels.h"
// We avoid directly including .cu files to prevent multiple definition errors during linking.
// The int4_kernel.cu and cuda_math_kernels.cu will be compiled separately.

// Declare the external functions
// void launchInt4GemvKernel(const half*, const unsigned char*, const half*, const unsigned char*, half*, int, int, cudaStream_t);
// Assume we have some wrappers in cuda_math_kernels.cu if needed, but for now we'll stub the FP16 launch or invoke a known kernel wrapper
extern void launchTernaryGemvKernel(const half* d_act, const unsigned char* d_weights, half* d_out, int K, int N, cudaStream_t stream);
extern void launchGemvFP16Wrapper(const half* d_weights, const half* d_act, half* d_out, int K, int N, cudaStream_t stream);



namespace friday_hpc {
    extern void launch_q2_K_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s);
    extern void launch_q3_K_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s);
    extern void launch_q4_K_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s);
    extern void launch_q6_K_hpc(const half* x, const void* W, half* y, int K, int N, cudaStream_t s);
}

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
            // Warp-per-row coalesced kernel (hpc_k_quants.cu). The previous
            // thread-per-row launchQ6_KGemvKernel read memory uncoalesced.
            friday_hpc::launch_q6_K_hpc(d_activations, d_weights, d_output, K, N, stream);
            break;
        }
        
        case Q2_K: {
            friday_hpc::launch_q2_K_hpc(d_activations, d_weights, d_output, K, N, stream);
            break;
        }
        case Q3_K: {
            friday_hpc::launch_q3_K_hpc(d_activations, d_weights, d_output, K, N, stream);
            break;
        }
        case Q4_K: {
            friday_hpc::launch_q4_K_hpc(d_activations, d_weights, d_output, K, N, stream);
            break;
        }

        case UNKNOWN:
        default: {
            fprintf(stderr, "Error: Bilinmeyen veya desteklenmeyen nicelendirme tipi: %d\n", quant_type);
            exit(EXIT_FAILURE);
        }
    }
}
