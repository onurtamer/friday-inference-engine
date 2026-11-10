// Engine::prefill vs. Engine::step on the same tokens (one engine, the KV cache is
// overwritten by each pass). Covers what the golden prompts (51 tokens) do not:
// prompts longer than one prefill chunk (512 tokens) and a prefill that starts at
// pos > 0 (for a split model this uploads and writes back the CPU layers' KV).
//   A: step() x n                       -> logits after token n - 1
//   B: prefill(n tokens, pos 0)
//   C: prefill(first m, pos 0) + prefill(rest, pos m)
// Then 8 greedy decode steps after each pass. Checks: the log-prob of A's top-1
// token agrees within `tol` nats, and the greedy tokens agree except where A's
// top-2 gap is below 0.1 nats (a near-tie).
// Usage: test_prefill_engine <model.gguf> <vram_gb> [n] [tol]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "check.h"
#include "engine.h"

static std::vector<double> logSoftmax(const std::vector<float>& l) {
    double m = -1e300;
    for (float v : l) if (std::isfinite(v)) m = std::max(m, (double)v);
    double s = 0;
    for (float v : l) if (std::isfinite(v)) s += std::exp(v - m);
    std::vector<double> out(l.size());
    for (size_t i = 0; i < l.size(); ++i) out[i] = std::isfinite(l[i]) ? l[i] - m - std::log(s) : -1e300;
    return out;
}

static int argmax(const std::vector<double>& v) { return (int)(std::max_element(v.begin(), v.end()) - v.begin()); }

static double gap12(const std::vector<double>& v) {
    double a = -1e300, b = -1e300;
    for (double x : v) { if (x > a) { b = a; a = x; } else if (x > b) b = x; }
    return a - b;
}

struct Pass { std::vector<double> lp; std::vector<int> gen; std::vector<double> gen_gap; };

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_prefill_engine <model.gguf> <vram_gb> [n] [tol]\n"); return 2; }
    friday::EngineOptions eo;
    eo.model_path = argv[1];
    eo.vram_gb = std::atof(argv[2]);
    const int n = argc > 3 ? std::atoi(argv[3]) : 600;
    const double tol = argc > 4 ? std::atof(argv[4]) : 0.1;
    eo.max_ctx = 2048;
    friday::Engine engine(eo);
    std::printf("[PREFILL] %s | batched prefill: %s\n", engine.summary().c_str(), engine.batchedPrefill() ? "yes" : "no");
    const int vocab = engine.vocabSize();
    // Deterministic pseudo-random tokens in a common id range.
    std::vector<int> tokens(n);
    uint32_t x = 12345;
    for (int i = 0; i < n; ++i) { x = x * 1664525u + 1013904223u; tokens[i] = 1000 + (int)(x % 20000u); }
    const int m = n / 2 + 7;
    std::vector<float> logits(vocab);

    auto finish = [&](Pass& p) {
        p.lp = logSoftmax(logits);
        int next = argmax(p.lp);
        for (int k = 0; k < 8; ++k) {
            p.gen.push_back(next);
            engine.step(next, n + k, logits.data());
            const auto lp = logSoftmax(logits);
            p.gen_gap.push_back(gap12(lp));
            next = argmax(lp);
        }
    };
    Pass A, B, C;
    for (int i = 0; i < n; ++i) engine.step(tokens[i], i, logits.data());
    finish(A);
    engine.prefill(tokens.data(), n, 0, logits.data());
    finish(B);
    engine.prefill(tokens.data(), m, 0, logits.data());
    engine.prefill(tokens.data() + m, n - m, m, logits.data());
    finish(C);

    const int top = argmax(A.lp);
    const double g = gap12(A.lp);
    for (const auto* p : {&B, &C}) {
        const char* name = p == &B ? "B prefill(n)" : "C prefill(m) + prefill(n - m, pos m)";
        const double d = std::fabs(p->lp[top] - A.lp[top]);
        int mism = 0;
        bool tied_before = g < 0.1;   // a near-tie may change every later token
        for (int k = 0; k < 8 && !tied_before; ++k) {
            if (p->gen[k] != A.gen[k]) { ++mism; break; }
            if (A.gen_gap[k] < 0.1) tied_before = true;
        }
        std::printf("[PREFILL] %-38s |lp(top1) diff| %.4f (tol %.2f) | greedy 8 tokens: %s\n", name, d, tol,
                    mism ? "DIFFER" : (tied_before ? "same up to a near-tie" : "same"));
        CHECK(d < tol);
        CHECK(mism == 0);
    }
    std::printf("[PREFILL] n=%d (chunk split at 512), m=%d | RESULT: %s\n", n, m, CHECK_RESULT() ? "FAIL" : "PASS");
    return CHECK_RESULT();
}
