// CPU K-quant GEMV (scalar reference and AVX2) vs. gguf reference outputs from
// make_fixtures.py, plus a multithreaded run through the thread pool.
// Usage: test_cpu_kquants <fixture_dir>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "check.h"
#include "cpu_kquants.h"
#include "cpu_threadpool.h"

using namespace friday_cpu;

template <typename T>
static std::vector<T> readFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    std::vector<T> v((size_t)f.tellg() / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
    return v;
}

static double relL2(const std::vector<double>& ref, const std::vector<float>& got) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1e-300));
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_cpu_kquants <fixture_dir>\n"); return 2; }
    const std::string dir = argv[1];
    std::ifstream mf(dir + "/meta.json");
    if (!mf) { std::fprintf(stderr, "missing %s/meta.json\n", dir.c_str()); return 2; }
    const auto meta = nlohmann::json::parse(mf);
    std::printf("AVX2 path: %s\n", hasAvx2() ? "yes" : "no (scalar only)");

    // fp16 conversion round trip on all finite halves
    int bad = 0;
    for (uint32_t h = 0; h < 0x10000; ++h) {
        if (((h >> 10) & 0x1F) == 0x1F) continue;
        if (floatToHalf(halfToFloat((uint16_t)h)) != (uint16_t)h && (h & 0x7FFF) != 0) ++bad;
    }
    std::printf("fp16 round trip mismatches: %d\n", bad);
    CHECK(bad == 0);

    ThreadPool pool(4);
    for (const auto& m : meta) {
        const std::string c = m["case"];
        if (c.rfind("emb_", 0) == 0) continue;
        const KQuant t = c == "q2_k" ? KQuant::Q2_K : c == "q3_k" ? KQuant::Q3_K : c == "q4_k" ? KQuant::Q4_K
                       : c == "q5_k" ? KQuant::Q5_K : c == "q8_0" ? KQuant::Q8_0 : KQuant::Q6_K;
        const int K = m["K"], N = m["N"];
        auto w = readFile<uint8_t>(dir + "/" + c + "_w.bin");
        auto xh = readFile<uint16_t>(dir + "/" + c + "_x.bin");
        auto ref = readFile<double>(dir + "/" + c + "_yref.bin");
        auto ref_q8 = readFile<double>(dir + "/" + c + "_yref_q8.bin");
        std::vector<float> x(K);
        for (int i = 0; i < K; ++i) x[i] = halfToFloat(xh[i]);
        Q8Activation xa;
        quantizeActivation(x.data(), K, xa);

        std::vector<float> ys(N), yv(N), yp(N);
        gemvRowsScalar(t, w.data(), xa, ys.data(), 0, N);
        gemvRows(t, w.data(), xa, yv.data(), 0, N);
        pool.parallelFor(N, 8, [&](int b, int e) { gemvRows(t, w.data(), xa, yp.data(), b, e); });

        const double es = relL2(ref_q8, ys), ev = relL2(ref_q8, yv), ep = relL2(ref_q8, yp), ef = relL2(ref, yv);
        std::printf("%-5s vs q8 ref: scalar %.2e  simd %.2e  pool %.2e | vs fp ref %.2e\n", c.c_str(), es, ev, ep, ef);
        CHECK(es < 1e-5);   // same math as the reference; only fp32 summation order differs
        CHECK(ev < 1e-5);
        CHECK(ep < 1e-5);
        CHECK(ef < 1.5e-2); // int8 activation quantization error
    }
    return CHECK_RESULT();
}
