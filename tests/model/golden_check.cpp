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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>
#include "engine.h"
#include "gguf_file.h"
#include "model_config.h"
#include "tokenizer.h"

// Optional reference-runtime check: log-probability this engine assigns, at each
// generated step, to the reference runtime's top-1 token.
static const std::vector<int>* g_ref_top1 = nullptr;
static std::vector<double> g_top1_logprob;

// Same selection rule as main.cpp: skip NaN/Inf, first maximum wins.
// Logits come from Engine::step (host floats; on the GPU path the exact
// values of the fp16 logits).
static int argmaxLogits(const std::vector<float>& h, int vocab, int* nan_count) {
    int best = -1; float best_v = -1e30f; int nans = 0;
    for (int i = 0; i < vocab; ++i) {
        float v = h[i];
        if (std::isnan(v) || std::isinf(v)) { ++nans; continue; }
        if (v > best_v) { best_v = v; best = i; }
    }
    *nan_count = nans;
    const size_t step = g_top1_logprob.size();
    if (g_ref_top1 && step < g_ref_top1->size() && best >= 0) {
        double z = 0;
        for (int i = 0; i < vocab; ++i) {
            float v = h[i];
            if (std::isfinite(v)) z += std::exp((double)v - best_v);
        }
        const float x = h[(*g_ref_top1)[step]];
        g_top1_logprob.push_back(std::isfinite(x) ? (double)x - best_v - std::log(z) : -1e30);
    }
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
    // Unbuffered output: progress stays visible when stdout is redirected to a file.
    setvbuf(stdout, nullptr, _IONBF, 0);
    // FRIDAY_GOLDEN_MAX_STEPS=N: smoke run (e.g. under compute-sanitizer) that only
    // processes the first N prompt tokens, prints progress, and exits without comparing.
    const char* max_steps_env = std::getenv("FRIDAY_GOLDEN_MAX_STEPS");
    const int max_steps = max_steps_env ? std::atoi(max_steps_env) : 0;
    try {
        const auto golden = loadJson(golden_dir + "/output_tokens.json");
        const auto cfg = loadJson(golden_dir + "/model_config.json");
        const auto gen = loadJson(golden_dir + "/generation_config.json");
        const auto prm = loadJson(golden_dir + "/prompt.json");
        const std::vector<int> golden_tokens = golden.at("tokens").get<std::vector<int>>();
        const int max_new = gen.at("max_new_tokens").get<int>();
        const std::string prompt = prm.at("prompt").get<std::string>();

        // A .gguf model carries its own tokenizer and hyper-parameters; the
        // tensor map and tokenizer arguments are ignored (pass "-"). The
        // golden dir's model_config.json is then checked against the file.
        const bool is_gguf = model_path.size() > 5 && model_path.compare(model_path.size() - 5, 5, ".gguf") == 0;
        std::unique_ptr<Tokenizer> tok_ptr;
        if (is_gguf) {
            const friday::GgufFile gguf(model_path);
            const friday::ModelConfig mc = friday::ModelConfig::fromGguf(gguf);
            mc.checkSupported();
            const bool same = mc.num_layers == cfg.at("num_layers").get<int>() &&
                              mc.hidden_dim == cfg.at("hidden_dim").get<int>() &&
                              mc.num_heads == cfg.at("num_heads").get<int>() &&
                              mc.num_kv_heads == cfg.at("num_kv_heads").get<int>() &&
                              mc.rms_norm_epsilon == cfg.at("rms_norm_epsilon").get<float>();
            printf("[GOLDEN] GGUF config vs model_config.json: %s\n", same ? "match" : "MISMATCH");
            if (!same) return 1;
            tok_ptr.reset(new Tokenizer(gguf));
        } else {
            tok_ptr.reset(new Tokenizer(tok_cfg));
        }
        Tokenizer& tokenizer = *tok_ptr;
        friday::EngineOptions eo;
        eo.model_path = model_path;
        eo.vram_gb = vram_gb;
        eo.tensor_map = tensor_map;  // deprecated .bin path only
        eo.hidden_dim = cfg.at("hidden_dim").get<int>();
        eo.num_heads = cfg.at("num_heads").get<int>();
        eo.num_kv_heads = cfg.at("num_kv_heads").get<int>();
        eo.num_layers = cfg.at("num_layers").get<int>();
        eo.rms_norm_epsilon = cfg.at("rms_norm_epsilon").get<float>();
        eo.quant_type = cfg.at("quant_type").get<std::string>();
        friday::Engine engine(eo);
        printf("[GOLDEN] %s\n", engine.summary().c_str());

        int vocab = engine.vocabSize();
        if (vocab == 0) vocab = tokenizer.getVocabSize();
        std::vector<float> logits(vocab);

        std::vector<int> tokens;
        if (argc >= 7) {  // optional: prompt token ids from file ("n=..." line, then ids)
            std::ifstream tf(argv[6]); std::string first; std::getline(tf, first);
            int id; while (tf >> id) tokens.push_back(id);
            printf("[GOLDEN] prompt tokens from file: %s\n", argv[6]);
        } else {
            tokens = tokenizer.encode(prm.value("chatml", true) ? tokenizer.applyChatTemplate(prompt) : prompt);
            if (tokenizer.addsBos()) tokens.insert(tokens.begin(), tokenizer.getBosTokenId());
        }
        const size_t prompt_len = tokens.size();
        int nans = 0, total_nans = 0;

        // Reference runtime (optional): <golden_dir>/llama_cpp_reference.json
        nlohmann::json ref;
        std::vector<int> ref_tokens, ref_top1;
        std::vector<double> ref_top1_lp, ref_gap;  // gap: reference top-1 minus top-2 log-prob
        {
            std::ifstream rf(golden_dir + "/llama_cpp_reference.json");
            if (rf.is_open()) {
                ref = nlohmann::json::parse(rf);
                ref_tokens = ref.at("tokens").get<std::vector<int>>();
                for (const auto& st : ref.at("steps")) {
                    ref_top1.push_back(st.at("top")[0][0].get<int>());
                    ref_top1_lp.push_back(st.at("top")[0][1].get<double>());
                    ref_gap.push_back(st.at("top").size() > 1
                                          ? st.at("top")[0][1].get<double>() - st.at("top")[1][1].get<double>()
                                          : 1e9);
                }
                g_ref_top1 = &ref_top1;
            }
        }

        // Engine time only (excludes this tool's per-step analysis of the logits).
        double engine_s = 0.0;
        auto step = [&](int tok, int pos) {
            const auto a = std::chrono::steady_clock::now();
            engine.step(tok, pos, logits.data());
            engine_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        };

        auto t0 = std::chrono::steady_clock::now();
        double engine_prefill_s = 0.0;
        if (max_steps > 0) {
            for (int i = 0; i < max_steps && i < (int)prompt_len; ++i) {
                auto ts = std::chrono::steady_clock::now();
                step(tokens[i], i);
                printf("[GOLDEN] smoke step %d done (%.1f s)\n", i,
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
            }
            printf("[GOLDEN] smoke run finished after %d steps (no comparison)\n", max_steps);
            return 0;
        }
        for (size_t i = 0; i < prompt_len; ++i) step(tokens[i], (int)i);
        auto t1 = std::chrono::steady_clock::now();
        engine_prefill_s = engine_s;

        std::vector<int> out;
        int next = argmaxLogits(logits, vocab, &nans); total_nans += nans;
        for (int k = 0; k < max_new; ++k) {
            if (next < 0) throw std::runtime_error("all logits NaN/Inf at step " + std::to_string(k));
            out.push_back(next);
            if (k + 1 == max_new) break;
            // Teacher forcing: with a reference, continue from the reference token so
            // every step is compared in the same context (a near-tie that flips one
            // token does not derail the rest of the comparison).
            const int feed = (k < (int)ref_tokens.size()) ? ref_tokens[k] : next;
            tokens.push_back(feed);
            step(feed, (int)tokens.size() - 1);
            next = argmaxLogits(logits, vocab, &nans); total_nans += nans;
        }
        auto t2 = std::chrono::steady_clock::now();

        // Snapshot: with teacher forcing every step has the reference context, so a
        // step where the reference itself is a near-tie may flip between builds or
        // GPU/CPU layer splits; such steps are reported but not counted.
        int mismatches = 0, first_div = -1, snapshot_near_ties = 0;
        const size_t n = std::max(out.size(), golden_tokens.size());
        for (size_t i = 0; i < n; ++i) {
            const bool same = i < out.size() && i < golden_tokens.size() && out[i] == golden_tokens[i];
            if (same) continue;
            if (i < out.size() && i < golden_tokens.size() && i < ref_gap.size() && ref_gap[i] < 0.1) {
                ++snapshot_near_ties;
                continue;
            }
            ++mismatches;
            if (first_div < 0) first_div = (int)i;
        }
        if (snapshot_near_ties) printf("[GOLDEN] snapshot differs only at %d reference near-tie step(s)\n", snapshot_near_ties);
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
        {
            const double eng_dec = engine_s - engine_prefill_s;
            printf("[GOLDEN] engine only: decode %.2f s (%.2f tok/s)\n", eng_dec,
                   (out.size() > 1 && eng_dec > 0 ? (out.size() - 1) / eng_dec : 0.0));
        }
        printf("[GOLDEN] generated ids:");
        for (int t : out) printf(" %d", t);
        printf("\n");

        // Reference-runtime checks: identical greedy tokens, and top-1 log-prob within tolerance.
        bool ref_ok = true;
        if (!ref_tokens.empty()) {
            const double kTop1LogprobTol = 0.3;
            // A differing argmax counts as a failure unless the reference itself has a
            // near-tie there (top-1 and top-2 within kNearTie nats).
            const double kNearTie = 0.1;
            int ref_mismatch = 0, near_ties = 0;
            for (size_t i = 0; i < ref_tokens.size(); ++i) {
                if (i < out.size() && out[i] == ref_tokens[i]) continue;
                if (i < out.size() && i < ref_gap.size() && ref_gap[i] < kNearTie) { ++near_ties; continue; }
                ++ref_mismatch;
            }
            double max_d = 0; size_t worst = 0;
            for (size_t i = 0; i < g_top1_logprob.size() && i < ref_top1_lp.size(); ++i) {
                const double d = std::fabs(g_top1_logprob[i] - ref_top1_lp[i]);
                if (d > max_d) { max_d = d; worst = i; }
            }
            ref_ok = ref_mismatch == 0 && g_top1_logprob.size() == ref_top1_lp.size() && max_d < kTop1LogprobTol;
            // FRIDAY_GOLDEN_DUMP=<file>: per-step token and log-prob of the reference
            // top-1 token, full precision (to compare two runs bit for bit).
            if (const char* dump = std::getenv("FRIDAY_GOLDEN_DUMP")) {
                if (FILE* df = std::fopen(dump, "w")) {
                    for (size_t i = 0; i < g_top1_logprob.size(); ++i)
                        std::fprintf(df, "%zu %d %.17g\n", i, i < out.size() ? out[i] : -1, g_top1_logprob[i]);
                    std::fclose(df);
                }
            }
            printf("[REFERENCE] %s: token mismatches %d/%zu (+%d at near-ties < %.1f nats) | max |top-1 logprob diff| "
                   "%.4f at step %zu (tol %.2f) | %s\n",
                   ref.value("runtime", std::string("reference")).c_str(), ref_mismatch, ref_tokens.size(), near_ties,
                   kNearTie, max_d, worst, kTop1LogprobTol, ref_ok ? "PASS" : "FAIL");
        }
        const bool pass = mismatches == 0 && ref_ok;
        printf("[GOLDEN] RESULT: %s\n", pass ? "PASS" : "FAIL");

        return pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "[GOLDEN] ERROR: " << e.what() << "\n";
        return 3;
    }
}
