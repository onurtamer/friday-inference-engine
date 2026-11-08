// GPU decode attention vs. the previous CPU implementation (same math), for
// Qwen2.5-32B shapes: 40 query heads, 8 KV heads, head_dim 128.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "attention_kernels.h"
#include "check.h"

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 2; } } while (0)

// Reference: the CPU attention formerly in Transformer::applySelfAttention.
static void cpuAttention(const std::vector<half>& q, const std::vector<half>& k, const std::vector<half>& v,
                         std::vector<float>& out, int seq, int nh, int nkv, int hd) {
    const int kv_dim = nkv * hd, per_kv = nh / nkv;
    const float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < nh; ++h) {
        const int kvh = h / per_kv;
        std::vector<float> s(seq);
        float mx = -1e30f;
        for (int p = 0; p < seq; ++p) {
            float acc = 0;
            for (int d = 0; d < hd; ++d) acc += __half2float(q[h * hd + d]) * __half2float(k[(size_t)p * kv_dim + kvh * hd + d]);
            acc *= scale;
            if (std::isnan(acc)) acc = -1e30f;
            s[p] = acc; mx = std::max(mx, acc);
        }
        float sum = 0;
        for (int p = 0; p < seq; ++p) { float df = s[p] - mx; s[p] = df < -50.0f ? 0.0f : expf(df); sum += s[p]; }
        const float inv = sum > 1e-12f ? 1.0f / sum : 0.0f;
        for (int d = 0; d < hd; ++d) {
            float o = 0;
            for (int p = 0; p < seq; ++p) o += (s[p] * inv) * __half2float(v[(size_t)p * kv_dim + kvh * hd + d]);
            if (std::isnan(o)) o = 0; o = std::min(std::max(o, -65504.0f), 65504.0f);
            out[h * hd + d] = o;
        }
    }
}

int main() {
    const int nh = 40, nkv = 8, hd = 128, max_ctx = 2048, kv_dim = nkv * hd;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<half> k((size_t)max_ctx * kv_dim), v((size_t)max_ctx * kv_dim), q(nh * hd);
    for (auto& x : k) x = __float2half(nd(rng));
    for (auto& x : v) x = __float2half(nd(rng));
    for (auto& x : q) x = __float2half(2.0f * nd(rng));   // sharper softmax

    half *dq, *dk, *dv, *dout;
    CUDA_OK(cudaMalloc(&dq, q.size() * 2)); CUDA_OK(cudaMalloc(&dk, k.size() * 2));
    CUDA_OK(cudaMalloc(&dv, v.size() * 2)); CUDA_OK(cudaMalloc(&dout, nh * hd * 2));
    CUDA_OK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice));

    void* ws = nullptr;
    int* dpos = nullptr;
    CUDA_OK(cudaMalloc(&ws, friday_attn::splitAttentionWorkspaceBytes(nh, max_ctx)));
    CUDA_OK(cudaMalloc(&dpos, sizeof(int)));
    for (int seq : {1, 2, 17, 255, 256, 257, 300, 1000, max_ctx}) {
        std::vector<float> ref(nh * hd);
        cpuAttention(q, k, v, ref, seq, nh, nkv, hd);
        friday_attn::launchDecodeAttention(dq, dk, dv, dout, seq, nh, nkv, hd, 0);
        CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
        std::vector<half> got(nh * hd);
        CUDA_OK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
        double num = 0, den = 0, mx = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double g = __half2float(got[i]), r = ref[i];
            num += (g - r) * (g - r); den += r * r; mx = std::max(mx, std::fabs(g - r));
        }
        const double rel = std::sqrt(num / (den > 0 ? den : 1e-300));
        std::printf("seq=%-5d rel_l2=%.3e max_abs=%.3e %s\n", seq, rel, mx, rel < 1e-3 ? "PASS" : "FAIL");
        CHECK(rel < 1e-3);
        // split-K, sequence length from the host and from device memory
        for (int mode = 0; mode < 2; ++mode) {
            const int pos = seq - 1;
            CUDA_OK(cudaMemcpy(dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));
            friday_attn::launchDecodeAttentionSplit(dq, dk, dv, dout, seq, mode ? dpos : nullptr, max_ctx, nh, nkv, hd, ws, 0);
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
            double n2 = 0, d2 = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                const double g = __half2float(got[i]), r = ref[i];
                n2 += (g - r) * (g - r); d2 += r * r;
            }
            const double r2 = std::sqrt(n2 / (d2 > 0 ? d2 : 1e-300));
            std::printf("  split-K (%s) rel_l2=%.3e %s\n", mode ? "device pos" : "host len", r2, r2 < 1e-3 ? "PASS" : "FAIL");
            CHECK(r2 < 1e-3);
        }
    }
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(ws); cudaFree(dpos);
    return CHECK_RESULT();
}
