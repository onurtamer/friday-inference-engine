// Causal prefill attention (launchPrefillAttention) vs. the decode split-K kernel
// run once per query at its position, on the same KV cache (fp16 and int8).
// Usage: test_prefill_attention [--bench]
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "attention_kernels.h"
#include "check.h"

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 2; } } while (0)

struct Case { int nh, nkv, T, p0; bool q8; };

static int runCase(const Case& c, std::mt19937& rng, bool bench) {
    const int hd = 128, max_ctx = 1024, qd = c.nh * hd, kv_dim = c.nkv * hd;
    const int seq = c.p0 + c.T;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> i8(-127, 127);
    std::uniform_real_distribution<float> us(0.005f, 0.03f);
    std::vector<half> q((size_t)c.T * qd), kh((size_t)max_ctx * kv_dim), vh((size_t)max_ctx * kv_dim);
    std::vector<int8_t> k8((size_t)max_ctx * kv_dim), v8((size_t)max_ctx * kv_dim);
    std::vector<half> ks((size_t)max_ctx * kv_dim / 32), vs((size_t)max_ctx * kv_dim / 32);
    for (auto& x : q) x = __float2half(nd(rng));
    for (size_t i = 0; i < kh.size(); ++i) { kh[i] = __float2half(nd(rng)); vh[i] = __float2half(nd(rng)); }
    for (size_t i = 0; i < k8.size(); ++i) { k8[i] = (int8_t)i8(rng); v8[i] = (int8_t)i8(rng); }
    for (size_t i = 0; i < ks.size(); ++i) { ks[i] = __float2half(us(rng)); vs[i] = __float2half(us(rng)); }

    half *d_q, *d_ref, *d_out, *d_ks = nullptr, *d_vs = nullptr;
    void *d_k, *d_v, *d_ws;
    int* d_pos;
    const size_t kv_bytes = c.q8 ? k8.size() : kh.size() * 2;
    CUDA_OK(cudaMalloc(&d_q, q.size() * 2));
    CUDA_OK(cudaMalloc(&d_ref, q.size() * 2));
    CUDA_OK(cudaMalloc(&d_out, q.size() * 2));
    CUDA_OK(cudaMalloc(&d_k, kv_bytes));
    CUDA_OK(cudaMalloc(&d_v, kv_bytes));
    CUDA_OK(cudaMalloc(&d_ws, friday_attn::splitAttentionWorkspaceBytes(c.nh, max_ctx)));
    CUDA_OK(cudaMalloc(&d_pos, (c.T + 1) * sizeof(int)));
    CUDA_OK(cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    if (c.q8) {
        CUDA_OK(cudaMalloc(&d_ks, ks.size() * 2));
        CUDA_OK(cudaMalloc(&d_vs, vs.size() * 2));
        CUDA_OK(cudaMemcpy(d_k, k8.data(), kv_bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_v, v8.data(), kv_bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_ks, ks.data(), ks.size() * 2, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_vs, vs.data(), vs.size() * 2, cudaMemcpyHostToDevice));
    } else {
        CUDA_OK(cudaMemcpy(d_k, kh.data(), kv_bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_v, vh.data(), kv_bytes, cudaMemcpyHostToDevice));
    }
    std::vector<int> pos(c.T + 1);
    pos[0] = c.p0;                                       // d_pos[0]: first position (prefill)
    for (int t = 0; t < c.T; ++t) pos[t + 1] = c.p0 + t; // d_pos[1 + t]: decode position of query t
    CUDA_OK(cudaMemcpy(d_pos, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice));

    for (int t = 0; t < c.T; ++t)
        friday_attn::launchDecodeAttentionSplit(d_q + (size_t)t * qd, d_k, d_v, d_ref + (size_t)t * qd, 0,
                                                d_pos + 1 + t, max_ctx, c.nh, c.nkv, hd, d_ws, nullptr, d_ks, d_vs);
    friday_attn::launchPrefillAttention(d_q, d_k, d_v, d_out, c.T, d_pos, c.nh, c.nkv, hd, nullptr, d_ks, d_vs);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<half> ref(q.size()), got(q.size());
    CUDA_OK(cudaMemcpy(ref.data(), d_ref, ref.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost));
    double num = 0, den = 0, max_abs = 0;
    int nonfinite = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = __half2float(ref[i]), g = __half2float(got[i]);
        if (!std::isfinite(g)) { ++nonfinite; continue; }
        num += (g - r) * (g - r);
        den += r * r;
        max_abs = std::fmax(max_abs, std::fabs(g - r));
    }
    const double rel = std::sqrt(num / (den > 0 ? den : 1e-300));
    const double max_rel = max_abs / std::sqrt(den / ref.size() + 1e-30);
    const bool ok = nonfinite == 0 && rel < 1e-3 && max_rel < 1e-2;
    std::printf("nh=%-3d nkv=%-2d T=%-4d p0=%-4d %s rel_l2=%.2e max_rel=%.2e %s\n", c.nh, c.nkv, c.T, c.p0,
                c.q8 ? "int8" : "fp16", rel, max_rel, ok ? "PASS" : "FAIL");
    CHECK(ok);

    if (bench) {
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int t = 0; t < c.T; ++t)
            friday_attn::launchDecodeAttentionSplit(d_q + (size_t)t * qd, d_k, d_v, d_ref + (size_t)t * qd, 0,
                                                    d_pos + 1 + t, max_ctx, c.nh, c.nkv, hd, d_ws, nullptr, d_ks, d_vs);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms_dec = 0;
        cudaEventElapsedTime(&ms_dec, a, b);
        cudaEventRecord(a);
        friday_attn::launchPrefillAttention(d_q, d_k, d_v, d_out, c.T, d_pos, c.nh, c.nkv, hd, nullptr, d_ks, d_vs);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms_pf = 0;
        cudaEventElapsedTime(&ms_pf, a, b);
        std::printf("    per-token decode kernel %.2f ms | prefill kernel %.3f ms\n", ms_dec, ms_pf);
    }
    cudaFree(d_q); cudaFree(d_ref); cudaFree(d_out); cudaFree(d_k); cudaFree(d_v); cudaFree(d_ws); cudaFree(d_pos);
    if (d_ks) { cudaFree(d_ks); cudaFree(d_vs); }
    return 0;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    std::mt19937 rng(20261110);
    const std::vector<Case> cases = bench ? std::vector<Case>{{32, 8, 512, 0, false}, {32, 8, 512, 0, true},
                                                              {40, 8, 512, 0, true}}
                                          : std::vector<Case>{{32, 8, 1, 0, false},   {32, 8, 13, 0, false},
                                                              {32, 8, 70, 100, false}, {40, 8, 37, 5, true},
                                                              {32, 8, 70, 300, true},  {16, 16, 9, 31, false},
                                                              {64, 8, 20, 64, true}};
    for (const auto& c : cases)
        if (runCase(c, rng, bench) != 0) return 2;
    return CHECK_RESULT();
}
