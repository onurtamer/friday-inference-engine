// Golden equivalence check for FRIDAY Engine.
// Replays the golden prompt with greedy decoding through the same code path as
// main.cpp (token-by-token prefill, then decode) and compares the generated
// token ids with golden/output_tokens.json. No asserts: every check is explicit
// and the exit code is 0 only if all tokens match.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#define INT8 WIN_INT8
#include <windows.h>
#undef INT8
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "transformer.h"
#include "io_manager.h"
#include "tokenizer.h"
#include "kernel_dispatcher.h"
#include <nlohmann/json.hpp>

static QuantType quantFromString(const std::string& s) {
    if (s == "F16" || s == "F32") return FP16;
    if (s == "Q4_0") return INT4;
    if (s == "Q4_1") return Q4_1;
    if (s == "Q8_0" || s == "Q8_1") return INT8;
    if (s == "Q6_K") return Q6_K;
    if (s == "Q2_K") return Q2_K;
    if (s == "Q3_K") return Q3_K;
    if (s == "Q4_K") return Q4_K;
    return UNKNOWN;
}

// Same selection rule as main.cpp: skip NaN/Inf, first maximum wins.
static int argmaxLogits(const half* d_logits, int vocab, cudaStream_t s, int* nan_count) {
    std::vector<half> h(vocab);
    CUDA_CHECK(cudaMemcpyAsync(h.data(), d_logits, vocab * sizeof(half), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    int best = -1; float best_v = -1e30f; int nans = 0;
    for (int i = 0; i < vocab; ++i) {
        float v = __half2float(h[i]);
        if (std::isnan(v) || std::isinf(v)) { ++nans; continue; }
        if (v > best_v) { best_v = v; best = i; }
    }
    *nan_count = nans;
    return best;
}

static nlohmann::json loadJson(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot open " + path);
    return nlohmann::json::parse(f);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: golden_check <model.bin> <tensor_map.json> <golden_dir> <tokenizer_config.json> <vram_gb>\n";
        return 2;
    }
    const std::string model_path = argv[1], tensor_map = argv[2], golden_dir = argv[3], tok_cfg = argv[4];
    const double vram_gb = std::stod(argv[5]);
    try {
        const auto golden = loadJson(golden_dir + "/output_tokens.json");
        const auto cfg = loadJson(golden_dir + "/model_config.json");
        const auto gen = loadJson(golden_dir + "/generation_config.json");
        const auto prm = loadJson(golden_dir + "/prompt.json");
        const std::vector<int> golden_tokens = golden.at("tokens").get<std::vector<int>>();
        const int max_new = gen.at("max_new_tokens").get<int>();
        const std::string prompt = prm.at("prompt").get<std::string>();

        Tokenizer tokenizer(tok_cfg);
        AsyncIOManager io(model_path);
        Transformer transformer(model_path, tensor_map,
                                cfg.at("hidden_dim").get<int>(), cfg.at("num_heads").get<int>(),
                                cfg.at("num_kv_heads").get<int>(), cfg.at("num_layers").get<int>(),
                                cfg.at("rms_norm_epsilon").get<float>(),
                                quantFromString(cfg.at("quant_type").get<std::string>()), &io, vram_gb);

        int vocab = transformer.getVocabSize();
        if (vocab == 0) vocab = tokenizer.getVocabSize();
        const int hidden = cfg.at("hidden_dim").get<int>();
        half* d_logits = nullptr; half* d_act = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_logits, vocab * sizeof(half)));
        CUDA_CHECK(cudaMalloc((void**)&d_act, hidden * sizeof(half)));
        cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));

        std::vector<int> tokens;
        if (argc >= 7) {  // optional: prompt token ids from file ("n=..." line, then ids)
            std::ifstream tf(argv[6]); std::string first; std::getline(tf, first);
            int id; while (tf >> id) tokens.push_back(id);
            printf("[GOLDEN] prompt tokens from file: %s\n", argv[6]);
        } else {
            tokens = tokenizer.encode(prm.value("chatml", true) ? tokenizer.applyChatTemplate(prompt) : prompt);
        }
        const size_t prompt_len = tokens.size();
        int nans = 0, total_nans = 0;

        auto step = [&](int tok, int pos) {
            if (!transformer.getEmbedding(tok, d_act, vocab, s))
                throw std::runtime_error("embedding failed for token " + std::to_string(tok));
            CUDA_CHECK(cudaStreamSynchronize(s));
            CUDA_CHECK(cudaMemsetAsync(d_logits, 0, vocab * sizeof(half), s));
            transformer.forward(d_act, d_logits, 1, 0, pos, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
        };

        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < prompt_len; ++i) step(tokens[i], (int)i);
        auto t1 = std::chrono::steady_clock::now();

        std::vector<int> out;
        int next = argmaxLogits(d_logits, vocab, s, &nans); total_nans += nans;
        for (int k = 0; k < max_new; ++k) {
            if (next < 0) throw std::runtime_error("all logits NaN/Inf at step " + std::to_string(k));
            out.push_back(next);
            if (k + 1 == max_new) break;
            tokens.push_back(next);
            step(next, (int)tokens.size() - 1);
            next = argmaxLogits(d_logits, vocab, s, &nans); total_nans += nans;
        }
        auto t2 = std::chrono::steady_clock::now();

        int mismatches = 0, first_div = -1;
        const size_t n = std::max(out.size(), golden_tokens.size());
        for (size_t i = 0; i < n; ++i) {
            const bool same = i < out.size() && i < golden_tokens.size() && out[i] == golden_tokens[i];
            if (!same) { ++mismatches; if (first_div < 0) first_div = (int)i; }
        }
        const double pre_s = std::chrono::duration<double>(t1 - t0).count();
        const double dec_s = std::chrono::duration<double>(t2 - t1).count();

        printf("\n[GOLDEN] prompt tokens: %zu | generated: %zu | golden: %zu\n", prompt_len, out.size(), golden_tokens.size());
        printf("[GOLDEN] mismatches: %d | first divergence: %d | NaN/Inf logits seen: %d\n", mismatches, first_div, total_nans);
        if (first_div >= 0) {
            printf("[GOLDEN] at %d: expected %d, got %d\n", first_div,
                   first_div < (int)golden_tokens.size() ? golden_tokens[first_div] : -1,
                   first_div < (int)out.size() ? out[first_div] : -1);
        }
        printf("[GOLDEN] measured: prefill %.2f s (%.2f tok/s) | decode %.2f s (%.2f tok/s)\n",
               pre_s, prompt_len / pre_s, dec_s, (out.size() > 1 ? (out.size() - 1) / dec_s : 0.0));
        printf("[GOLDEN] generated ids:");
        for (int t : out) printf(" %d", t);
        printf("\n[GOLDEN] RESULT: %s\n", mismatches == 0 ? "PASS" : "FAIL");

        cudaFree(d_logits); cudaFree(d_act); cudaStreamDestroy(s);
        return mismatches == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "[GOLDEN] ERROR: " << e.what() << "\n";
        return 3;
    }
}
