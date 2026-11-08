#include "cuda_math_kernels.h"
#include <cuda_fp16.h>
#include <limits> // For numeric_limits



// Kernel Configuration - These values should be tuned for optimal performance
#define BLOCK_SIZE 128 // Number of threads per block
#define K_TILE_SIZE 32 // Number of K elements processed per iteration by each block
#define N_PER_THREAD 1 // Each thread computes one output element (simplified for now)

/**
 * @brief CUDA kernel for FP16 Activation Vector x GGUF Q4_0 Weight Matrix multiplication.
 *
 * This kernel processes an activation vector (1xK) against a GGUF Q4_0 quantized weight matrix (KxN).
 * It reads block_q4_0 structures, dequantizes the 4-bit weights on-the-fly using the block's scale,
 * and performs FP16 GEMV.
 *
 * Each thread block is responsible for computing a tile of the output vector.
 * Threads within a block cooperate to load and process data.
 *
 * @param d_activations Device pointer to FP16 activation vector (1xK).
 * @param d_quantized_weights Device pointer to GGUF Q4_0 quantized weight matrix (KxN).
 * @param d_output Device pointer to FP16 output vector (1xN).
 * @param K Input dimension (length of activation vector, rows of weight matrix).
 * @param N Output dimension (columns of weight matrix, length of output vector).
 */
__global__ void q4_0GemvKernel(
    const half* __restrict__ d_activations,
    const block_q4_0* __restrict__ d_quantized_weights,
    half* __restrict__ d_output,
    int K,
    int N)
{
    // N is out_features (number of output neurons).
    // K is in_features (number of input features).
    // Weight matrix in GGUF is logically [N, K] and contiguous along K.
    // So there are N rows, each containing K elements (or K/32 blocks).
    
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_idx >= N) return;

    float sum = 0.0f;
    int num_blocks_per_row = K / 32;
    const block_q4_0* row_weights = &d_quantized_weights[n_idx * num_blocks_per_row];

    for (int b = 0; b < num_blocks_per_row; ++b) {
        const block_q4_0* current_block = &row_weights[b];
        float scale = __half2float(current_block->d);
        
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = current_block->qs[i];
            
            // Lower nibble
            int8_t v0 = (packed & 0x0F) - 8;
            // Upper nibble
            int8_t v1 = (packed >> 4) - 8;
            
              // GGML Q4_0 layout: Low nibble -> elements 0..15, High nibble -> elements 16..31
              int k0 = b * 32 + i;
              int k1 = b * 32 + i + 16;
            
            float act0 = __half2float(d_activations[k0]);
            float act1 = __half2float(d_activations[k1]);
            
            sum += (scale * (float)v0) * act0;
            sum += (scale * (float)v1) * act1;
        }
    }
    
    if (isnan(sum)) sum = 0.0f;
    else if (sum > 65504.0f) sum = 65504.0f;
    else if (sum < -65504.0f) sum = -65504.0f;

    d_output[n_idx] = __float2half(sum);
}

// Host-side wrapper function to launch the kernel
void launchQ4_0GemvKernel(
    const half* d_activations,
    const block_q4_0* d_quantized_weights,
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
    // Each block processes (BLOCK_SIZE * N_PER_THREAD) N elements
    int gridDimX = (N + (BLOCK_SIZE * N_PER_THREAD) - 1) / (BLOCK_SIZE * N_PER_THREAD);

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(gridDimX);

    // Launch the kernel
    q4_0GemvKernel<<<gridDim, blockDim, 0, stream>>>(
        d_activations,
        d_quantized_weights,
        d_output,
        K,
        N
    );
    CUDA_CHECK(cudaGetLastError()); // Check for errors during kernel launch
}

__global__ void q8_0GemvKernel(
    const half* __restrict__ d_activations,
    const block_q8_0* __restrict__ d_quantized_weights,
    half* __restrict__ d_output,
    int K,
    int N)
{
    // N is out_features, K is in_features.
    // Matrix is [N, K], contiguous along K.
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_idx >= N) return;

    float sum = 0.0f;
    int num_blocks_per_row = K / 32;
    const block_q8_0* row_weights = &d_quantized_weights[n_idx * num_blocks_per_row];

    for (int b = 0; b < num_blocks_per_row; ++b) {
        const block_q8_0* current_block = &row_weights[b];
        float scale = __half2float(current_block->d);
        
        for (int i = 0; i < 32; ++i) {
            int8_t v = current_block->qs[i];
            int k_idx = b * 32 + i;
            float act = __half2float(d_activations[k_idx]);
            sum += (scale * (float)v) * act;
        }
    }
    
    if (isnan(sum)) sum = 0.0f;
    else if (sum > 65504.0f) sum = 65504.0f;
    else if (sum < -65504.0f) sum = -65504.0f;

    d_output[n_idx] = __float2half(sum);
}

// Host-side wrapper function to launch the kernel
void launchQ8_0GemvKernel(
    const half* d_activations,
    const block_q8_0* d_quantized_weights,
    half* d_output,
    int K,
    int N,
    cudaStream_t stream)
{
    if (K <= 0 || N <= 0) {
        fprintf(stderr, "Error: K and N must be positive dimensions.\n");
        exit(EXIT_FAILURE);
    }
    int gridDimX = (N + (BLOCK_SIZE * N_PER_THREAD) - 1) / (BLOCK_SIZE * N_PER_THREAD);
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(gridDimX);

    q8_0GemvKernel<<<gridDim, blockDim, 0, stream>>>(
        d_activations,
        d_quantized_weights,
        d_output,
        K,
        N
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Q4_1 GEMV Kernel
// ============================================================
__global__ void q4_1GemvKernel(
    const half* __restrict__ d_activations,
    const block_q4_1* __restrict__ d_quantized_weights,
    half* __restrict__ d_output,
    int K,
    int N)
{
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_idx >= N) return;

    float sum = 0.0f;
    int num_blocks_per_row = K / 32;
    const block_q4_1* row_weights = &d_quantized_weights[n_idx * num_blocks_per_row];

    for (int b = 0; b < num_blocks_per_row; ++b) {
        const block_q4_1* current_block = &row_weights[b];
        float scale = __half2float(current_block->d);
        float min_val = __half2float(current_block->m);

        for (int i = 0; i < 16; ++i) {
            uint8_t packed = current_block->qs[i];
            uint8_t v0 = (packed & 0x0F);
            uint8_t v1 = (packed >> 4);

            int k0 = b * 32 + i;
            int k1 = b * 32 + i + 16;

            float act0 = __half2float(d_activations[k0]);
            float act1 = __half2float(d_activations[k1]);

            float w0 = scale * (float)v0 + min_val;
            float w1 = scale * (float)v1 + min_val;

            sum += w0 * act0;
            sum += w1 * act1;
        }
    }

    if (isnan(sum)) sum = 0.0f;
    else if (sum > 65504.0f) sum = 65504.0f;
    else if (sum < -65504.0f) sum = -65504.0f;

    d_output[n_idx] = __float2half(sum);
}

void launchQ4_1GemvKernel(
    const half* d_activations,
    const block_q4_1* d_quantized_weights,
    half* d_output,
    int K,
    int N,
    cudaStream_t stream)
{
    if (K <= 0 || N <= 0) {
        fprintf(stderr, "Error: K and N must be positive dimensions.\n");
        exit(EXIT_FAILURE);
    }

    int gridDimX = (N + (BLOCK_SIZE * N_PER_THREAD) - 1) / (BLOCK_SIZE * N_PER_THREAD);
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(gridDimX);

    q4_1GemvKernel<<<gridDim, blockDim, 0, stream>>>(
        d_activations,
        d_quantized_weights,
        d_output,
        K,
        N
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// RMSNorm Kernel
// Uses a single block with shared memory reduction.
// Each thread processes multiple elements (size / blockDim.x).
// Weight in GGUF is 32-bit float (F32).
// Formula: output[i] = (input[i] / sqrt(mean(input^2) + epsilon)) * weight[i]
// ============================================================
// Per-head RMSNorm in place (Qwen3 attn_q_norm / attn_k_norm): one block per
// head, blockDim.x == head_dim threads; weight [head_dim] shared by all heads.
// Same rules as rmsNormKernel: fp32 sum of squares (NaN/Inf skipped), output
// NaN -> 0 and clamped to the fp16 range.
__global__ void headRmsNormKernel(half* d_x, const float* d_weight, int head_dim, float epsilon) {
    extern __shared__ float red[];
    half* v = d_x + (size_t)blockIdx.x * head_dim;
    const int tid = threadIdx.x;
    float val = __half2float(v[tid]);
    if (isnan(val) || isinf(val)) val = 0.0f;
    red[tid] = val * val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    const float rms = sqrtf(red[0] / (float)head_dim + epsilon);
    const float inv = (rms > 1e-12f) ? (1.0f / rms) : 1.0f;
    float res = val * inv * d_weight[tid];
    if (isnan(res)) res = 0.0f;
    else if (res > 65504.0f) res = 65504.0f;
    else if (res < -65504.0f) res = -65504.0f;
    v[tid] = __float2half(res);
}

__global__ void rmsNormKernel(half* d_input, const float* d_weight, half* d_output, int size, float epsilon) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Step 1: Compute sum of squares (accumulate in FP32)
    float sum_sq = 0.0f;
    for (int i = tid; i < size; i += block_size) {
        float val = __half2float(d_input[i]);
        if (!isnan(val) && !isinf(val)) {
            sum_sq += val * val;
        }
    }
    shared_data[tid] = sum_sq;
    __syncthreads();

    // Step 2: Parallel reduction to compute total sum of squares
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Step 3: Compute RMS normalization factor
    float mean_sq = shared_data[0] / (float)size;
    float rms = sqrtf(mean_sq + epsilon);
    float rms_inv = (rms > 1e-12f) ? (1.0f / rms) : 1.0f;

    // Step 4: Normalize and apply weight
    for (int i = tid; i < size; i += block_size) {
        float val = __half2float(d_input[i]);
        if (isnan(val) || isinf(val)) val = 0.0f;
        float w   = d_weight[i];
        float res = val * rms_inv * w;
        if (isnan(res)) res = 0.0f;
        else if (res > 65504.0f) res = 65504.0f;
        else if (res < -65504.0f) res = -65504.0f;
        d_output[i] = __float2half(res);
    }
}

// ============================================================
// Rotary Position Embedding (RoPE) Kernel
// Rotates Q and K for all heads at position pos_offset.
//   neox = 1: pairs (i, i + head_dim/2)  (rotate_half; Qwen2)
//   neox = 0: pairs (2i, 2i + 1)          (GGML "normal" mode; Llama)
// theta_i = pos / base^(2i / head_dim), divided by freq_factors[i] when given
// (Llama 3 rope_freqs.weight).
// ============================================================
__global__ void ropeKernel(half* d_query, half* d_key, int head_dim, int seq_len, int pos_offset, int num_q_heads,
                           int num_kv_heads, float freq_base, int neox, const float* freq_factors) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_head_dim = head_dim / 2;

    if (pair_idx >= half_head_dim) return;

    float position = (float)pos_offset;
    float theta = position / powf(freq_base, (2.0f * (float)pair_idx) / (float)head_dim);
    if (freq_factors) theta /= freq_factors[pair_idx];
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    const int off0 = neox ? pair_idx : 2 * pair_idx;
    const int off1 = neox ? pair_idx + half_head_dim : 2 * pair_idx + 1;

    for (int h = 0; h < num_q_heads + num_kv_heads; ++h) {
        half* v = h < num_q_heads ? d_query + h * head_dim : d_key + (h - num_q_heads) * head_dim;
        float x0 = __half2float(v[off0]);
        float x1 = __half2float(v[off1]);

        float res0 = x0 * cos_theta - x1 * sin_theta;
        float res1 = x0 * sin_theta + x1 * cos_theta;
        if (isnan(res0)) res0 = 0.0f; else if (res0 > 65504.0f) res0 = 65504.0f; else if (res0 < -65504.0f) res0 = -65504.0f;
        if (isnan(res1)) res1 = 0.0f; else if (res1 > 65504.0f) res1 = 65504.0f; else if (res1 < -65504.0f) res1 = -65504.0f;

        v[off0] = __float2half(res0);
        v[off1] = __float2half(res1);
    }
}

// ============================================================
// SiLU (Swish) Kernel - Named geluKernel for API compatibility
// Qwen2.5 uses SwiGLU: SiLU(gate) * up
// Clamped to avoid floating-point overflow and NaN
// ============================================================
__global__ void geluKernel(half* d_activations, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = __half2float(d_activations[idx]);
    if (isnan(x)) {
        d_activations[idx] = __float2half(0.0f);
        return;
    }
    if (x > 20.0f) {
        return;
    }
    if (x < -20.0f) {
        d_activations[idx] = __float2half(0.0f);
        return;
    }
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    float res = x * sigmoid_x;
    if (isnan(res)) res = 0.0f;
    else if (res > 65504.0f) res = 65504.0f;
    else if (res < -65504.0f) res = -65504.0f;
    d_activations[idx] = __float2half(res);
}

// ============================================================
// Element-wise Multiplication Kernel for SwiGLU
// Computes: d_a[i] = a[i] * b[i] in FP32, then clamps to FP16
// ============================================================
__global__ void elementWiseMulKernel(half* d_a, const half* d_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float a = __half2float(d_a[idx]);
    float b = __half2float(d_b[idx]);
    if (isnan(a)) a = 0.0f;
    if (isnan(b)) b = 0.0f;
    float res = a * b;
    if (isnan(res)) res = 0.0f;
    else if (res > 65504.0f) res = 65504.0f;
    else if (res < -65504.0f) res = -65504.0f;
    d_a[idx] = __float2half(res);
}

// ============================================================
// Add Residual Kernel
// Computes: target[i] += addend[i] in FP32, clamped to FP16
// ============================================================
__global__ void addResidualKernel(half* d_target, const half* d_addend, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float t = __half2float(d_target[idx]);
    float a = __half2float(d_addend[idx]);
    if (isnan(t)) t = 0.0f;
    if (isnan(a)) a = 0.0f;
    float res = t + a;
    if (isnan(res)) res = 0.0f;
    else if (res > 65504.0f) res = 65504.0f;
    else if (res < -65504.0f) res = -65504.0f;
    d_target[idx] = __float2half(res);
}

// ============================================================
// Add Bias Kernel (F32 bias)
// Computes: target[i] += bias[i] in FP32, clamped to FP16
// ============================================================
__global__ void addBiasF32Kernel(half* d_target, const float* d_bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float t = __half2float(d_target[idx]);
    float b = d_bias[idx];
    if (isnan(t)) t = 0.0f;
    if (isnan(b)) b = 0.0f;
    float res = t + b;
    if (isnan(res)) res = 0.0f;
    else if (res > 65504.0f) res = 65504.0f;
    else if (res < -65504.0f) res = -65504.0f;
    d_target[idx] = __float2half(res);
}

