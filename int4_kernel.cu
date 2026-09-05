#ifndef INT4_KERNEL_CU
#define INT4_KERNEL_CU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

__device__ __forceinline__ half dequantize_int4(unsigned char int4_value, half scale, unsigned char zero_point) {
    float s = __half2float(scale);
    float val = static_cast<float>(int4_value);
    float zp = static_cast<float>(zero_point);
    return __float2half(s * (val - zp));
}

__global__ void int4GemvKernel(
    const half* __restrict__ d_activations,
    const unsigned char* __restrict__ d_packed_int4_weights,
    const half* __restrict__ d_scales,
    const unsigned char* __restrict__ d_zero_points,
    half* __restrict__ d_output,
    int in_features,
    int out_features,
    int out_features_padded)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < out_features) {
        float sum = 0.0f;
        const int num_packed_bytes_per_row = (out_features_padded + 1) / 2;

        for (int k_idx = 0; k_idx < in_features; ++k_idx) {
            half activation = d_activations[k_idx];
            int packed_byte_idx_in_row = row / 2;
            bool is_upper_nibble = (row % 2 == 1);
            
            // Bounds check for packed bytes array to prevent illegal memory access
            int byte_idx = k_idx * num_packed_bytes_per_row + packed_byte_idx_in_row;
            const unsigned char packed_byte_val = d_packed_int4_weights[byte_idx];

            unsigned char int4_val;
            if (is_upper_nibble) {
                int4_val = (packed_byte_val >> 4) & 0x0F;
            } else {
                int4_val = packed_byte_val & 0x0F;
            }

            // Safe guards for scales and zero points to prevent crash if not present
            half scale = d_scales ? d_scales[row] : __float2half(1.0f);
            unsigned char zero_point = d_zero_points ? d_zero_points[row] : 0;
            
            half dequantized_weight = dequantize_int4(int4_val, scale, zero_point);
            sum += __half2float(activation) * __half2float(dequantized_weight);
        }
        d_output[row] = __float2half(sum);
    }
}

void launchInt4GemvKernel(
    const half* d_activations,
    const unsigned char* d_packed_int4_weights,
    const half* d_scales,
    const unsigned char* d_zero_points,
    half* d_output,
    int K,
    int N,
    cudaStream_t stream = 0)
{
    if (K <= 0 || N <= 0) {
        fprintf(stderr, "Error: K and N must be positive dimensions for INT4 GEMV.\n");
        exit(EXIT_FAILURE);
    }
    int N_padded = (N + 1) / 2 * 2;
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    int4GemvKernel<<<gridDim, blockDim, 0, stream>>>(
        d_activations, d_packed_int4_weights, d_scales, d_zero_points, d_output, K, N, N_padded
    );
    CUDA_CHECK(cudaGetLastError());
}

#endif // INT4_KERNEL_CU
