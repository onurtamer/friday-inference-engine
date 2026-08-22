#include "cuda_math_kernels.h"
#include <cuda_fp16.h>
#include <limits> // For numeric_limits

// Kernel Configuration - These values should be tuned for optimal performance
// Number of threads per block along X and Y dimensions
#define BLOCK_THREADS_X 32
#define BLOCK_THREADS_Y 8
// N_PER_THREAD defines how many output elements each thread computes
#define N_PER_THREAD 4
// TILE_K defines the reduction dimension processed by each block
#define TILE_K 128
// TILE_N_PER_BLOCK is the total number of N elements processed by a block
#define TILE_N_PER_BLOCK (BLOCK_THREADS_Y * N_PER_THREAD)

// Ternary value mapping from 2-bit representation:
// 00 -> -1
// 01 ->  0
// 10 ->  1
// 11 (unused)
__device__ __forceinline__ int get_ternary_value(unsigned char packed_byte, int bit_pos) {
    // bit_pos: 0 for MSB (first ternary), 1 for second, 2 for third, 3 for LSB (fourth ternary)
    unsigned char two_bits = (packed_byte >> ((3 - bit_pos) * 2)) & 0b11;
    if (two_bits == 0b00) return -1;
    if (two_bits == 0b01) return 0;
    if (two_bits == 0b10) return 1;
    return 0; // Should not happen with valid input
}

/**
 * @brief CUDA kernel for FP16 Activation Vector x 1.58-bit Ternary Weight Matrix multiplication.
 *
 * This kernel processes an activation vector (1xK) against a ternary weight matrix (KxN)
 * using only additions and subtractions, leveraging the ternary nature of the weights.
 * The weight matrix is packed with 4 ternary values per byte.
 *
 * Each thread block is responsible for computing a tile of the output vector.
 * Threads within a block cooperate to load and process data from global memory
 * into shared memory, maximizing VRAM bandwidth utilization.
 *
 * @param d_activations Device pointer to FP16 activation vector (1xK).
 * @param d_packed_weights Device pointer to packed ternary weight matrix (KxN, 2-bit per ternary value).
 * @param d_output Device pointer to FP16 output vector (1xN).
 * @param K Input dimension (length of activation vector, rows of weight matrix).
 * @param N Output dimension (columns of weight matrix, length of output vector).
 */
__global__ void ternaryGemvKernel(
    const half* __restrict__ d_activations,
    const unsigned char* __restrict__ d_packed_weights,
    half* __restrict__ d_output,
    int K,
    int N)
{
    // Shared memory for activations to be reused by threads in a block
    // Using __half2 to load 2 FP16 values at once for better memory coalescing
    extern __shared__ __half2 sh_activations[];

    // Calculate global N index for this thread block
    const int N_global_start = blockIdx.x * TILE_N_PER_BLOCK;

    // Each thread computes N_PER_THREAD output elements
    // N_idx for the current thread's first output element
    const int N_thread_start = N_global_start + threadIdx.y * N_PER_THREAD;

    // Accumulators for the output elements this thread is responsible for
    half accumulators[N_PER_THREAD];
    for (int i = 0; i < N_PER_THREAD; ++i) {
        accumulators[i] = __float2half(0.0f);
    }

    // Loop over K dimension in tiles
    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += TILE_K) {
        // Load activations into shared memory
        // Each thread loads two FP16 activations (one __half2)
        // Ensure that we don't read past K
        for (int k_idx_in_tile = threadIdx.x; k_idx_in_tile < TILE_K; k_idx_in_tile += blockDim.x) {
            const int global_k = k_tile_start + k_idx_in_tile;
            if (global_k < K) {
                // Determine the shared memory index for this thread.
                // Divide by 2 because sh_activations stores __half2 (two halfs per element).
                sh_activations[k_idx_in_tile / 2].x = (global_k < K) ? d_activations[global_k] : __float2half(0.0f);
                sh_activations[k_idx_in_tile / 2].y = (global_k + 1 < K) ? d_activations[global_k + 1] : __float2half(0.0f);
            }
        }
        __syncthreads(); // Wait for all activations to be loaded into shared memory

        // Process the tile
        for (int k_local = 0; k_local < TILE_K; ++k_local) {
            const int global_k = k_tile_start + k_local;
            if (global_k >= K) break; // Ensure we don't go out of bounds for K

            // Get activation from shared memory
            half activation = sh_activations[k_local / 2].x; // For even k_local
            if (k_local % 2 != 0) { // For odd k_local
                activation = sh_activations[k_local / 2].y;
            }

            // If activation is zero, skip multiplication for this K dimension
            if (__half2float(activation) == 0.0f) {
                continue;
            }

            // Calculate the base index for the packed weights in global memory
            // K rows, N columns, 4 ternary values per byte means N/4 bytes per row
            const int packed_row_byte_size = (N + 3) / 4; // ceil(N/4) bytes per row
            const int base_weight_byte_idx = global_k * packed_row_byte_size;

            // Iterate over N_PER_THREAD output elements this thread computes
            for (int i = 0; i < N_PER_THREAD; ++i) {
                const int current_N_idx = N_thread_start + i;
                if (current_N_idx >= N) continue; // Ensure we don't go out of bounds for N

                // Calculate which byte and which 2-bit position within that byte
                // the current ternary weight for (global_k, current_N_idx) resides.
                const int byte_offset_in_row = current_N_idx / 4;
                const int bit_pos_in_byte = current_N_idx % 4; // 0 for MSB, 3 for LSB

                const int packed_weight_idx = base_weight_byte_idx + byte_offset_in_row;

                // Load the packed byte from global memory
                unsigned char packed_byte = d_packed_weights[packed_weight_idx];

                // Extract the ternary value
                int ternary_weight = get_ternary_value(packed_byte, bit_pos_in_byte);

                // Perform the addition/subtraction based on ternary weight
                if (ternary_weight == 1) {
                    accumulators[i] = __hadd(accumulators[i], activation);
                } else if (ternary_weight == -1) {
                    accumulators[i] = __hsub(accumulators[i], activation);
                }
                // If ternary_weight is 0, nothing is added to the accumulator
            }
        }
        __syncthreads(); // Wait for all threads to finish processing their part of the tile
    }

    // Write results to global memory
    for (int i = 0; i < N_PER_THREAD; ++i) {
        const int current_N_idx = N_thread_start + i;
        if (current_N_idx < N) {
            d_output[current_N_idx] = accumulators[i];
        }
    }
}

// Host-side wrapper function to launch the kernel
void launchTernaryGemvKernel(
    const half* d_activations,
    const unsigned char* d_packed_weights,
    half* d_output,
    int K,
    int N,
    cudaStream_t stream)
{
    // Ensure dimensions are positive
    if (K <= 0 || N <= 0) {
        fprintf(stderr, "Error: K and N must be positive dimensions.\n");
        exit(EXIT_FAILURE);
    }

    // Calculate grid and block dimensions
    // Grid dimension X: Number of blocks needed to cover N output elements
    // Each block processes TILE_N_PER_BLOCK elements
    int gridDimX = (N + TILE_N_PER_BLOCK - 1) / TILE_N_PER_BLOCK;

    // Block dimensions
    dim3 blockDim(BLOCK_THREADS_X, BLOCK_THREADS_Y);
    dim3 gridDim(gridDimX, 1, 1); // Only need 1D grid for this GEMV

    // Calculate shared memory size
    // We need to store TILE_K activations. Each activation is 'half'.
    // We use __half2 to store them, so TILE_K / 2 __half2 elements.
    // Plus a bit of buffer just in case.
    size_t shared_mem_size = (sizeof(__half2) * (TILE_K / 2 + 1)); // +1 for safety with odd K

    // Launch the kernel
    // The shared memory size here is for the sh_activations array.
    ternaryGemvKernel<<<gridDim, blockDim, shared_mem_size, stream>>>(
        d_activations,
        d_packed_weights,
        d_output,
        K,
        N
    );
    CUDA_CHECK(cudaGetLastError()); // Check for errors during kernel launch
}
