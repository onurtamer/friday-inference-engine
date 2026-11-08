#ifndef FRIDAY_ENGINE_H
#define FRIDAY_ENGINE_H

// Public inference API, independent of the backend. With CUDA available the
// engine keeps as many layers as fit in VRAM on the GPU and runs the rest on
// the CPU; otherwise (or with backend = Cpu) the whole model runs on the CPU.
// Callers never see device memory: step() takes a token and returns logits
// in host memory.

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

namespace friday {

enum class BackendKind { Auto, Cuda, Cpu };

struct EngineOptions {
    std::string model_path;      // .gguf (or a deprecated converted .bin)
    double vram_gb = 0.0;        // 0: use the free VRAM
    int max_ctx = 2048;          // KV cache capacity (tokens)
    int cpu_threads = 0;         // 0: default (2/3 of the logical CPUs)
    BackendKind backend = BackendKind::Auto;  // FRIDAY_BACKEND=cpu|cuda overrides Auto
    bool kv_q8 = false;          // int8 GPU KV cache (--kv q8, or FRIDAY_KV=q8)

    // Deprecated .bin path only: hyper-parameters and tensor map from the converter.
    std::string tensor_map;
    int hidden_dim = 0, num_heads = 0, num_kv_heads = 0, num_layers = 0;
    float rms_norm_epsilon = 0.0f;
    std::string quant_type;
};

class Engine {
public:
    explicit Engine(const EngineOptions& opt);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    int vocabSize() const;
    int maxContext() const;
    bool usesCuda() const;
    std::string summary() const;   // one line: backend(s), layer split, threads

    // Runs one token at position pos (positions must be fed in order, starting
    // at 0 for a new sequence) and writes vocabSize() logits to `logits`.
    void step(int token, int pos, float* logits);

    // True if this build includes the CUDA backend.
    static bool cudaCompiledIn();

    struct GpuInfo {
        std::string name;
        int compute_major = 0, compute_minor = 0, multiprocessors = 0;
        uint64_t total_bytes = 0, free_bytes = 0;
        int pcie_gen_max = 0, pcie_width_max = 0;   // from NVML when available, else 0
    };
    // CUDA devices visible to this process (empty without CUDA).
    static std::vector<GpuInfo> gpuInfo();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace friday

#endif  // FRIDAY_ENGINE_H
