// Batched prefill GEMM (friday_prefill::gemmQ8) vs. the decode GEMV (gemvQ8) run
// token by token on the same int8 activations, for every supported weight type.
// Both form the same integer dot products; only the order of the fp32 additions
// differs, so the outputs should agree to about one fp16 rounding step.
// Usage: test_prefill_gemm
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "check.h"
#include "hpc_k_quants.h"
#include "prefill_kernels.h"

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 2; } } while (0)

using friday_hpc::Q8Kind;

struct Layout { Q8Kind kind; const char* name; int values, bytes; std::vector<int> scale_offsets; };

static std::vector<uint8_t> randomWeights(const Layout& L, int N, int K, std::mt19937& rng) {
    const int blocks = K / L.values;
    std::vector<uint8_t> w((size_t)N * blocks * L.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(1e-3f, 1e-2f);
    for (auto& b : w) b = (uint8_t)byte(rng);
    for (size_t blk = 0; blk < (size_t)N * blocks; ++blk)
        for (int off : L.scale_offsets) {
            const half h = __float2half(sc(rng));
            std::memcpy(&w[blk * L.bytes + off], &h, 2);
        }
    return w;
}

// --bench: time model-sized GEMMs (T=512) and report the effective rate.
struct BenchCase { int li, K, N; const char* what; };
static int bench(const std::vector<Layout>& layouts) {
    const std::vector<BenchCase> cases = {{2, 4096, 14336, "8B ffn_gate"}, {4, 14336, 4096, "8B ffn_down"},
                                          {0, 5120, 27648, "32B ffn_gate"}, {1, 27648, 5120, "32B ffn_down"}};
    const int T = 512;
    std::mt19937 rng(1);
    for (const auto& c : cases) {
        const Layout& L = layouts[c.li];
        auto w = randomWeights(L, c.N, c.K, rng);
        uint8_t* d_w; half* d_y; int8_t* d_xq; float* d_xd;
        CUDA_OK(cudaMalloc(&d_w, w.size()));
        CUDA_OK(cudaMalloc(&d_y, (size_t)T * c.N * 2));
        CUDA_OK(cudaMalloc(&d_xq, (size_t)T * c.K));
        CUDA_OK(cudaMalloc(&d_xd, (size_t)T * c.K / 32 * 4));
        CUDA_OK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(d_xq, 1, (size_t)T * c.K));
        CUDA_OK(cudaMemset(d_xd, 0, (size_t)T * c.K / 32 * 4));
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        friday_prefill::gemmQ8(L.kind, d_w, d_xq, d_xd, T, c.K, c.N, d_y, c.N, nullptr);
        const int iters = 10;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) friday_prefill::gemmQ8(L.kind, d_w, d_xq, d_xd, T, c.K, c.N, d_y, c.N, nullptr);
        cudaEventRecord(b);
        CUDA_OK(cudaEventSynchronize(b));
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        ms /= iters;
        std::printf("%-14s %s T=%d K=%d N=%d: %.2f ms  %.1f TOPS\n", c.what, L.name, T, c.K, c.N, ms,
                    2.0 * T * c.K * c.N / (ms * 1e-3) / 1e12);
        cudaFree(d_w); cudaFree(d_y); cudaFree(d_xq); cudaFree(d_xd);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (!friday_prefill::gemmQ8Supported(Q8Kind::Q4_K, 256)) {
        std::printf("SKIP: needs compute capability 8.0+\n");
        return 0;
    }
    const std::vector<Layout> layouts = {
        {Q8Kind::Q2_K, "q2_k", 256, 84, {80, 82}},  {Q8Kind::Q3_K, "q3_k", 256, 110, {108}},
        {Q8Kind::Q4_K, "q4_k", 256, 144, {0, 2}},   {Q8Kind::Q5_K, "q5_k", 256, 176, {0, 2}},
        {Q8Kind::Q6_K, "q6_k", 256, 210, {208}},    {Q8Kind::Q8_0, "q8_0", 32, 34, {0}},
    };
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench(layouts);
    struct Shape { int T, K, N; bool epi; };
    const std::vector<Shape> shapes = {{1, 512, 64, false}, {37, 1024, 200, false}, {130, 768, 96, true}};
    std::mt19937 rng(20261110);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (const auto& L : layouts) {
        for (const auto& S : shapes) {
            const int T = S.T, K = S.K, N = S.N;
            auto w = randomWeights(L, N, K, rng);
            std::vector<half> x((size_t)T * K), res((size_t)T * N);
            std::vector<float> bias(N);
            for (auto& v : x) v = __float2half(nd(rng));
            for (auto& v : res) v = __float2half(nd(rng));
            for (auto& v : bias) v = 0.1f * nd(rng);

            uint8_t* d_w; half *d_x, *d_ref, *d_y, *d_res; float *d_bias, *d_xd; int8_t* d_xq;
            CUDA_OK(cudaMalloc(&d_w, w.size()));
            CUDA_OK(cudaMalloc(&d_x, x.size() * 2));
            CUDA_OK(cudaMalloc(&d_ref, (size_t)T * N * 2));
            CUDA_OK(cudaMalloc(&d_y, (size_t)T * N * 2));
            CUDA_OK(cudaMalloc(&d_res, (size_t)T * N * 2));
            CUDA_OK(cudaMalloc(&d_bias, N * 4));
            CUDA_OK(cudaMalloc(&d_xq, (size_t)T * K));
            CUDA_OK(cudaMalloc(&d_xd, (size_t)T * K / 32 * 4));
            CUDA_OK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_res, res.data(), res.size() * 2, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_ref, res.data(), res.size() * 2, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_y, res.data(), res.size() * 2, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_bias, bias.data(), N * 4, cudaMemcpyHostToDevice));
            const float* b = S.epi ? d_bias : nullptr;

            // Reference: decode GEMV per token (residual in place, as the engine uses it).
            friday_hpc::ensureQ8Scratch(K, nullptr);
            for (int t = 0; t < T; ++t) {
                friday_hpc::quantizeQ8(d_x + (size_t)t * K, K, nullptr);
                half* yt = d_ref + (size_t)t * N;
                friday_hpc::gemvQ8(L.kind, d_w, yt, K, N, nullptr, nullptr, nullptr, b, S.epi ? yt : nullptr);
            }
            // Batched.
            friday_hpc::quantizeQ8Rows(d_x, T, K, d_xq, d_xd, nullptr);
            friday_prefill::gemmQ8(L.kind, d_w, d_xq, d_xd, T, K, N, d_y, N, nullptr, b, S.epi ? d_y : nullptr);
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaDeviceSynchronize());

            std::vector<half> ref((size_t)T * N), got((size_t)T * N);
            CUDA_OK(cudaMemcpy(ref.data(), d_ref, ref.size() * 2, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(got.data(), d_y, got.size() * 2, cudaMemcpyDeviceToHost));
            // max_rel: largest |difference| relative to the RMS of the reference. A
            // different summation order can move one fp16 rounding step of an
            // intermediate (before bias/residual), which a per-element relative
            // error would flag when the residual cancels it; a layout bug gives
            // differences of the order of the RMS.
            double num = 0, den = 0, max_abs = 0;
            int nonfinite = 0, exact = 0;
            double worst_r = 0, worst_g = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double r = __half2float(ref[i]), g = __half2float(got[i]);
                if (!std::isfinite(g)) { ++nonfinite; continue; }
                num += (g - r) * (g - r);
                den += r * r;
                if (g == r) ++exact;
                if (std::fabs(g - r) > max_abs) { max_abs = std::fabs(g - r); worst_r = r; worst_g = g; }
            }
            const double rel = std::sqrt(num / (den > 0 ? den : 1e-300));
            const double max_rel = max_abs / std::sqrt(den / ref.size() + 1e-30);
            const bool ok = nonfinite == 0 && rel < 1e-3 && max_rel < 1e-2;
            std::printf("%-5s T=%-4d K=%-5d N=%-4d %s rel_l2=%.2e max_rel=%.2e exact=%.1f%% %s\n", L.name, T, K, N,
                        S.epi ? "bias+res" : "plain   ", rel, max_rel, 100.0 * exact / ref.size(),
                        ok ? "PASS" : "FAIL");
            if (!ok) std::printf("      worst element: ref %.6g got %.6g\n", worst_r, worst_g);
            CHECK(ok);
            cudaFree(d_w); cudaFree(d_x); cudaFree(d_ref); cudaFree(d_y); cudaFree(d_res);
            cudaFree(d_bias); cudaFree(d_xq); cudaFree(d_xd);
        }
    }
    return CHECK_RESULT();
}
