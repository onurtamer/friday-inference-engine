// friday-bench: controlled decode/prefill benchmark through the public Engine API.
//
// Each run starts a new sequence: prefill P prompt tokens (Engine::prefill:
// batched on the GPU when supported, otherwise one step per token),
// then decode G tokens greedily. Recorded per run: prefill and decode tok/s,
// per-token decode latency, the system's CPU load in a 1 s window before the run
// (background load) and during it, free RAM, and GPU clocks / temperature /
// power / utilization sampled every 200 ms (NVML, if present). Results: a table
// on stdout and every raw value in a JSON file.
//
// usage: friday-bench <model.gguf | registry id> [--runs N] [--warmup N] [--prompt P] [--gen G]
//                     [--ctx N] [--vram GB] [--kv auto|f16|q8] [--backend auto|cuda|cpu]
//                     [--threads N] [--json out.json] [--label text]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>

#include "engine.h"
#include "gguf_file.h"
#include "gpu_monitor.h"
#include "model_config.h"
#include "model_registry.h"
#include "sha256.h"
#include "system_info.h"
#include "tokenizer.h"

#ifndef FRIDAY_GIT_REV
#define FRIDAY_GIT_REV "unknown"
#endif

using Clock = std::chrono::steady_clock;
using nlohmann::json;

namespace {

// Fixed prompt text (repeated to the requested length): mixed prose, code and numbers.
const char* kPromptText =
    "The quick brown fox jumps over the lazy dog. In 1969, Apollo 11 landed on the Moon; the mission "
    "lasted 8 days, 3 hours and 18 minutes. Bir sistemi hizlandirmanin ilk adimi onu olcmektir. "
    "int main() { std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6}; std::sort(v.begin(), v.end()); return v[0]; } "
    "Memory bandwidth, not arithmetic, bounds single-stream decoding on most hardware. ";

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p / 100.0 * (v.size() - 1);
    const size_t lo = (size_t)std::floor(idx), hi = (size_t)std::ceil(idx);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

int argmax(const std::vector<float>& l) {
    int best = -1;
    float bv = -1e30f;
    for (int i = 0; i < (int)l.size(); ++i)
        if (std::isfinite(l[i]) && l[i] > bv) { bv = l[i]; best = i; }
    return best;
}

std::string utcNow() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char b[32];
    std::strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return b;
}

// Background sampler: GPU telemetry and free RAM every 200 ms while running.
class Sampler {
public:
    explicit Sampler(const friday::GpuMonitor& g) : gpu_(g) {}
    void start() {
        stop_ = false;
        samples_.clear();
        th_ = std::thread([this] {
            while (!stop_) {
                json s;
                friday::GpuSample g;
                if (gpu_.available() && gpu_.sample(0, &g)) {
                    s["sm_mhz"] = g.sm_clock_mhz; s["mem_mhz"] = g.mem_clock_mhz; s["temp_c"] = g.temperature_c;
                    s["power_w"] = g.power_w; s["util_gpu"] = g.util_gpu_pct; s["pcie_gen"] = g.pcie_gen_cur;
                }
                s["free_ram_mb"] = (double)friday::memoryInfo().available_bytes / (1024.0 * 1024.0);
                { std::lock_guard<std::mutex> lk(mu_); samples_.push_back(s); }
                for (int i = 0; i < 20 && !stop_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    json stop() {
        stop_ = true;
        th_.join();
        std::lock_guard<std::mutex> lk(mu_);
        return summarize();
    }
private:
    json summarize() const {
        json out;
        out["count"] = samples_.size();
        for (const char* k : {"sm_mhz", "mem_mhz", "temp_c", "power_w", "util_gpu", "free_ram_mb"}) {
            std::vector<double> v;
            for (const auto& s : samples_) if (s.contains(k)) v.push_back(s[k].get<double>());
            if (v.empty()) continue;
            out[k] = {{"min", *std::min_element(v.begin(), v.end())}, {"median", percentile(v, 50)},
                      {"max", *std::max_element(v.begin(), v.end())}};
        }
        return out;
    }
    const friday::GpuMonitor& gpu_;
    std::thread th_;
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::vector<json> samples_;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: friday-bench <model.gguf | id> [--runs N] [--warmup N] [--prompt P] [--gen G] [--ctx N]\n"
                     "                    [--vram GB] [--kv auto|f16|q8] [--backend auto|cuda|cpu] [--threads N]\n"
                     "                    [--json out.json] [--label text]\n";
        return 2;
    }
    std::string model = argv[1], json_path, label, kv = "auto", backend = "auto";
    int runs = 5, warmup = 1, prompt_len = 512, gen = 128, ctx = 0, threads = 0;
    double vram = 0.0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--runs") runs = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--prompt") prompt_len = std::stoi(next());
        else if (a == "--gen") gen = std::stoi(next());
        else if (a == "--ctx") ctx = std::stoi(next());
        else if (a == "--vram") vram = std::stod(next());
        else if (a == "--kv") kv = next();
        else if (a == "--backend") backend = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--json") json_path = next();
        else if (a == "--label") label = next();
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    if (runs < 1 || prompt_len < 1 || gen < 2) { std::cerr << "need --runs >= 1, --prompt >= 1, --gen >= 2\n"; return 2; }

    try {
        if (!std::ifstream(model, std::ios::binary).is_open()) {
            friday::ModelRegistry reg;
            if (const friday::ModelEntry* e = reg.find(model)) model = e->path;
        }
        const friday::GgufFile gguf(model);
        const friday::ModelConfig cfg = friday::ModelConfig::fromGguf(gguf);
        Tokenizer tok(gguf);

        // Prompt: fixed text tokenized and repeated to exactly prompt_len ids.
        std::vector<int> unit = tok.encode(kPromptText);
        std::vector<int> prompt;
        if (tok.addsBos()) prompt.push_back(tok.getBosTokenId());
        while ((int)prompt.size() < prompt_len) prompt.insert(prompt.end(), unit.begin(), unit.end());
        prompt.resize(prompt_len);
        if (ctx <= 0) ctx = std::max(2048, prompt_len + gen + 8);

        friday::EngineOptions eo;
        eo.model_path = model;
        eo.vram_gb = vram;
        eo.max_ctx = ctx;
        eo.cpu_threads = threads;
        eo.kv_cache = kv;
        eo.backend = backend == "cpu" ? friday::BackendKind::Cpu
                   : backend == "cuda" ? friday::BackendKind::Cuda : friday::BackendKind::Auto;
        const auto t_load0 = Clock::now();
        friday::Engine engine(eo);
        const double load_s = std::chrono::duration<double>(Clock::now() - t_load0).count();
        std::vector<float> logits(engine.vocabSize());

        const friday::GpuMonitor gpu;
        Sampler sampler(gpu);
        json jruns = json::array();
        std::vector<double> dec_tps, pre_tps, all_lat_ms;
        std::vector<int> first_tokens;
        bool deterministic = true;

        for (int r = -warmup; r < runs; ++r) {
            // Background load: whole-system CPU busy % over 1 s before the run.
            const friday::CpuTimes b0 = friday::cpuTimes();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const friday::CpuTimes b1 = friday::cpuTimes();
            const double bg_cpu = friday::cpuBusyPercent(b0, b1);
            const double free_before = (double)friday::memoryInfo().available_bytes / (1024.0 * 1024.0);

            sampler.start();
            const friday::CpuTimes c0 = friday::cpuTimes();
            const auto t0 = Clock::now();
            engine.prefill(prompt.data(), prompt_len, 0, logits.data());
            const auto t1 = Clock::now();
            std::vector<int> out;
            std::vector<double> lat;
            int next = argmax(logits);
            for (int k = 0; k < gen; ++k) {
                out.push_back(next);
                if (k + 1 == gen) break;
                const auto a = Clock::now();
                engine.step(next, prompt_len + k, logits.data());
                lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - a).count());
                next = argmax(logits);
            }
            const auto t2 = Clock::now();
            const friday::CpuTimes c1 = friday::cpuTimes();
            json tele = sampler.stop();

            const double pre_s = std::chrono::duration<double>(t1 - t0).count();
            const double dec_s = std::chrono::duration<double>(t2 - t1).count();
            const double ptps = prompt_len / pre_s, dtps = (gen - 1) / dec_s;
            if (r < 0) {
                std::printf("warmup: prefill %.2f tok/s, decode %.2f tok/s\n", ptps, dtps);
                continue;
            }
            if (first_tokens.empty()) first_tokens = out;
            else if (out != first_tokens) deterministic = false;
            pre_tps.push_back(ptps);
            dec_tps.push_back(dtps);
            all_lat_ms.insert(all_lat_ms.end(), lat.begin(), lat.end());
            json jr;
            jr["run"] = r;
            jr["prefill_tok_s"] = ptps;
            jr["decode_tok_s"] = dtps;
            jr["decode_ms_p50"] = percentile(lat, 50);
            jr["decode_ms_p95"] = percentile(lat, 95);
            jr["decode_ms_max"] = lat.empty() ? 0.0 : *std::max_element(lat.begin(), lat.end());
            jr["cpu_busy_before_pct"] = bg_cpu;
            jr["cpu_busy_during_pct"] = friday::cpuBusyPercent(c0, c1);
            jr["free_ram_mb_before"] = free_before;
            jr["telemetry"] = tele;
            jruns.push_back(jr);
            std::printf("run %d: prefill %7.2f tok/s | decode %6.2f tok/s (p50 %.2f ms, p95 %.2f ms) | "
                        "bg CPU %4.1f%% | ", r + 1, ptps, dtps, percentile(lat, 50), percentile(lat, 95), bg_cpu);
            if (tele.contains("sm_mhz"))
                std::printf("SM %d-%d MHz, %d C", (int)tele["sm_mhz"]["min"].get<double>(),
                            (int)tele["sm_mhz"]["max"].get<double>(), (int)tele["temp_c"]["max"].get<double>());
            std::printf("\n");
        }

        // Summary
        const double dec_med = percentile(dec_tps, 50), pre_med = percentile(pre_tps, 50);
        std::printf("\nmodel   : %s (%s, %s)\n", cfg.name.c_str(), cfg.architecture.c_str(), cfg.file_type.c_str());
        std::printf("engine  : %s\n", engine.summary().c_str());
        std::printf("decode  : median %.2f tok/s (min %.2f, max %.2f) over %d runs | token latency p50 %.2f ms, p95 %.2f ms\n",
                    dec_med, *std::min_element(dec_tps.begin(), dec_tps.end()),
                    *std::max_element(dec_tps.begin(), dec_tps.end()), runs, percentile(all_lat_ms, 50),
                    percentile(all_lat_ms, 95));
        std::printf("prefill : median %.2f tok/s (%d tokens, %s)\n", pre_med, prompt_len,
                    engine.batchedPrefill() ? "batched" : "one step per token");
        std::printf("output  : %s across runs\n", deterministic ? "identical" : "NOT identical");

        if (!json_path.empty()) {
            const friday::CpuInfo ci = friday::cpuInfo();
            const friday::MemoryInfo mi = friday::memoryInfo();
            json j;
            j["tool"] = "friday-bench";
            j["git_rev"] = FRIDAY_GIT_REV;
            j["date_utc"] = utcNow();
            j["label"] = label;
            j["model"] = {{"path", model}, {"name", cfg.name}, {"architecture", cfg.architecture},
                          {"file_type", cfg.file_type}, {"layers", cfg.num_layers}, {"file_bytes", gguf.fileSize()}};
            friday::ModelRegistry reg;
            if (const friday::ModelEntry* e = reg.find(model)) j["model"]["sha256"] = e->sha256;
            j["options"] = {{"runs", runs}, {"warmup", warmup}, {"prompt_tokens", prompt_len}, {"gen_tokens", gen},
                            {"ctx", ctx}, {"vram_gb", vram}, {"kv", kv}, {"backend", backend}, {"threads", threads}};
            j["engine"] = engine.summary();
            j["prompt_ids"] = prompt;  // lets other runtimes be measured on the same tokens
            j["load_s"] = load_s;
            j["system"] = {{"cpu", ci.brand}, {"arch", ci.arch}, {"logical_cpus", ci.logical_cpus},
                           {"cpu_features", ci.features}, {"ram_total_bytes", mi.total_bytes}};
            friday::GpuStatic gs;
            if (gpu.available() && gpu.staticInfo(0, &gs))
                j["system"]["gpu"] = {{"name", gs.name}, {"driver", gs.driver}, {"sm_clock_max_mhz", gs.sm_clock_max_mhz},
                                      {"mem_clock_max_mhz", gs.mem_clock_max_mhz}, {"pcie_gen_max", gs.pcie_gen_max},
                                      {"pcie_width_max", gs.pcie_width_max}, {"memory_bytes", gs.memory_total}};
            j["runs"] = jruns;
            j["summary"] = {{"decode_tok_s_median", dec_med},
                            {"decode_tok_s_min", *std::min_element(dec_tps.begin(), dec_tps.end())},
                            {"decode_tok_s_max", *std::max_element(dec_tps.begin(), dec_tps.end())},
                            {"decode_ms_p50", percentile(all_lat_ms, 50)},
                            {"decode_ms_p95", percentile(all_lat_ms, 95)},
                            {"prefill_tok_s_median", pre_med},
                            {"prefill_mode", engine.batchedPrefill() ? "batched" : "step"},
                            {"output_identical_across_runs", deterministic}};
            std::ofstream f(json_path);
            f << j.dump(2) << "\n";
            std::printf("json    : %s\n", json_path.c_str());
        }
        return deterministic ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "friday-bench: " << e.what() << "\n";
        return 3;
    }
}
