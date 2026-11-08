// GPU kernel correctness: engine GEMV kernels (Q2_K/Q3_K/Q4_K/Q6_K) and the Q2_K
// embedding-row dequant kernel vs. gguf reference outputs from make_fixtures.py.
// Usage: test_kquant_kernels <fixture_dir>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include "check.h"
#include "kernel_dispatcher.h"

#include "transformer.h"   // launchDequantizeRow

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 2; } } while (0)

template <typename T>
static std::vector<T> readFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    std::vector<T> v((size_t)f.tellg() / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
    return v;
}

static double relL2(const std::vector<double>& ref, const std::vector<double>& got, int* nonfinite) {
    double num = 0, den = 0; *nonfinite = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) { ++*nonfinite; continue; }
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1e-300));
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_kquant_kernels <fixture_dir>\n"); return 2; }
    const std::string dir = argv[1];
    std::ifstream mf(dir + "/meta.json");
    if (!mf) { std::fprintf(stderr, "missing %s/meta.json\n", dir.c_str()); return 2; }
    const auto meta = nlohmann::json::parse(mf);

    for (const auto& m : meta) {
        const std::string c = m["case"];
        const int K = m["K"], N = m["N"];
        std::vector<double> ref, got;
        double tol;
        if (c.rfind("emb_", 0) == 0) {
            const std::string dtype = m["dtype"];
            auto table = readFile<uint8_t>(dir + "/" + c + "_table.bin");
            auto r = readFile<float>(dir + "/" + c + "_ref.bin");
            ref.assign(r.begin(), r.end());
            const int row_bytes = m["row_bytes"];
            uint8_t* d_t; half* d_o;
            CUDA_OK(cudaMalloc(&d_t, table.size())); CUDA_OK(cudaMalloc(&d_o, K * sizeof(half)));
            CUDA_OK(cudaMemcpy(d_t, table.data(), table.size(), cudaMemcpyHostToDevice));
            for (int id : m["ids"].get<std::vector<int>>()) {
                CHECK(launchDequantizeRow(dtype, d_t + (size_t)id * row_bytes, d_o, K, 0));
                CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
                std::vector<half> h(K);
                CUDA_OK(cudaMemcpy(h.data(), d_o, K * sizeof(half), cudaMemcpyDeviceToHost));
                for (half v : h) got.push_back(__half2float(v));
            }
            cudaFree(d_t); cudaFree(d_o);
            tol = 2e-3;   // fp16 rounding of each dequantized value
        } else {
            const QuantType qt = c == "q2_k" ? Q2_K : c == "q3_k" ? Q3_K : c == "q4_k" ? Q4_K : Q6_K;
            auto w = readFile<uint8_t>(dir + "/" + c + "_w.bin");
            auto x = readFile<uint16_t>(dir + "/" + c + "_x.bin");
            ref = readFile<double>(dir + "/" + c + "_yref.bin");
            void* d_w; half *d_x, *d_y;
            CUDA_OK(cudaMalloc(&d_w, w.size())); CUDA_OK(cudaMalloc(&d_x, x.size() * 2));
            CUDA_OK(cudaMalloc(&d_y, N * sizeof(half)));
            CUDA_OK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemset(d_y, 0, N * sizeof(half)));
            FridayKernelDispatcher::dispatchGemvKernel(qt, d_x, d_w, nullptr, nullptr, d_y, K, N, 0);
            CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
            std::vector<half> h(N);
            CUDA_OK(cudaMemcpy(h.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost));
            for (half v : h) got.push_back(__half2float(v));
            cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);
            tol = 5e-3;   // fp32 accumulation, fp16 output
            // Q2_K/Q3_K/Q4_K use int8 activations (dp4a) unless FRIDAY_GEMV_FP32=1.
            const char* fp32_env = std::getenv("FRIDAY_GEMV_FP32");
            const bool int8_act = qt != Q6_K && !(fp32_env && fp32_env[0] == '1');
            if (int8_act) {
                // Exact semantics: reference that quantizes x the same way.
                auto ref_q8 = readFile<double>(dir + "/" + c + "_yref_q8.bin");
                int nf = 0;
                const double e8 = relL2(ref_q8, got, &nf);
                std::printf("%-9s rel_l2_vs_q8_ref=%.3e %s\n", c.c_str(), e8, e8 < 2e-3 ? "PASS" : "FAIL");
                CHECK(e8 < 2e-3);
                tol = 1.5e-2;   // vs. fp reference: adds int8 activation quantization error (~0.5%)
            }
        }
        int nonfinite = 0;
        const double err = (got.size() == ref.size()) ? relL2(ref, got, &nonfinite) : 1e9;
        std::printf("%-9s rel_l2=%.3e nonfinite=%d %s\n", c.c_str(), err, nonfinite,
                    (err < tol && nonfinite == 0) ? "PASS" : "FAIL");
        CHECK(got.size() == ref.size());
        CHECK(nonfinite == 0);
        CHECK(err < tol);
    }
    return CHECK_RESULT();
}
