#include "engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cpu_backend.h"
#include "cpu_kquants.h"
#include "gpu_monitor.h"
#include "gguf_file.h"
#include "io_manager.h"
#include "model_config.h"
#include "model_layout.h"

#if FRIDAY_HAVE_CUDA
#include <cuda_runtime.h>
#include "kernel_dispatcher.h"
#include "transformer.h"
#endif

namespace friday {

namespace {
bool endsWith(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

BackendKind resolveBackend(BackendKind requested) {
    if (requested == BackendKind::Auto) {
        if (const char* e = std::getenv("FRIDAY_BACKEND")) {
            if (std::strcmp(e, "cpu") == 0) return BackendKind::Cpu;
            if (std::strcmp(e, "cuda") == 0) return BackendKind::Cuda;
        }
    }
#if FRIDAY_HAVE_CUDA
    if (requested == BackendKind::Auto) {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
            cudaGetLastError();  // clear the error state
            std::fprintf(stderr, "[Engine] No CUDA device available; using the CPU backend.\n");
            return BackendKind::Cpu;
        }
        return BackendKind::Cuda;
    }
    return requested;
#else
    if (requested == BackendKind::Cuda) throw std::runtime_error("this build has no CUDA backend");
    return BackendKind::Cpu;
#endif
}

#if FRIDAY_HAVE_CUDA
QuantType quantTypeFromString(const std::string& s) {
    if (s == "F16" || s == "F32") return FP16;
    if (s == "Q4_0") return INT4;
    if (s == "Q4_1") return Q4_1;
    if (s == "Q8_0" || s == "Q8_1") return INT8;
    if (s == "Q6_K") return Q6_K;
    if (s == "Q2_K") return Q2_K;
    if (s == "Q3_K") return Q3_K;
    if (s == "Q4_K") return Q4_K;
    if (s == "Q5_K") return Q5_K;
    return UNKNOWN;
}
#endif
}  // namespace

struct Engine::Impl {
    EngineOptions opt;
    BackendKind kind = BackendKind::Cpu;
    std::unique_ptr<AsyncIOManager> io;
    int vocab = 0;
    std::string summary;

    // CPU-only
    ModelLayout layout;
    std::unique_ptr<CpuLayerRunner> cpu;

#if FRIDAY_HAVE_CUDA
    std::unique_ptr<Transformer> gpu;
    void* d_act = nullptr;      // half[hidden]
    void* d_logits = nullptr;   // half[vocab]
    cudaStream_t stream = nullptr;
    std::vector<uint16_t> h_logits;
#endif

    void initCpu() {
        if (!endsWith(opt.model_path, ".gguf"))
            throw std::runtime_error("the CPU backend needs a .gguf model (not the deprecated converted .bin)");
        const GgufFile g(opt.model_path);
        const ModelConfig cfg = ModelConfig::fromGguf(g);
        cfg.checkSupported();
        layout.loadGguf(g);
        layout.buildLayers(cfg.num_layers);
        CpuModelParams p;
        p.hidden_dim = cfg.hidden_dim;
        p.num_heads = cfg.num_heads;
        p.num_kv_heads = cfg.num_kv_heads;
        p.head_dim = cfg.head_dim;
        for (const auto& [name, t] : layout.tensors)
            if (name.find("ffn_gate.weight") != std::string::npos && t.shape.size() > 1)
                p.max_ffn_dim = std::max(p.max_ffn_dim, (int)t.shape[1]);
        p.rms_norm_epsilon = cfg.rms_norm_epsilon;
        p.rope_freq_base = cfg.rope_freq_base;
        p.rope_neox = cfg.ropeNeox();
        if (const TensorInfo* rf = layout.find("rope_freqs.weight")) {
            p.rope_freqs.resize(rf->size_bytes / sizeof(float));
            if (!layout.readTensor(*io, *rf, reinterpret_cast<unsigned char*>(p.rope_freqs.data())))
                throw std::runtime_error("cannot read rope_freqs.weight");
        }
        p.max_ctx = opt.max_ctx;
        cpu.reset(new CpuLayerRunner(layout, *io, p, 0, cfg.num_layers, opt.cpu_threads));
        cpu->loadHead();
        vocab = cpu->vocabSize();
        char buf[256];
        std::snprintf(buf, sizeof(buf), "CPU backend: %d/%d layers on the CPU (%.2f GB), %d threads, %s",
                      cfg.num_layers, cfg.num_layers, (double)cpu->weightBytes() / (1u << 30), cpu->threads(),
                      friday_cpu::hasAvx2() ? "AVX2" : "scalar");
        summary = buf;
    }

#if FRIDAY_HAVE_CUDA
    void initCuda() {
        int hidden = opt.hidden_dim, heads = opt.num_heads, kv = opt.num_kv_heads, layers = opt.num_layers;
        float eps = opt.rms_norm_epsilon;
        QuantType qt = quantTypeFromString(opt.quant_type);
        if (endsWith(opt.model_path, ".gguf")) {
            const GgufFile g(opt.model_path);
            const ModelConfig cfg = ModelConfig::fromGguf(g);
            cfg.checkSupported();
            hidden = cfg.hidden_dim; heads = cfg.num_heads; kv = cfg.num_kv_heads; layers = cfg.num_layers;
            eps = cfg.rms_norm_epsilon;
            if (qt == UNKNOWN) qt = Q4_K;  // informational only in the GGUF path
        }
        std::string kvs = opt.kv_cache;
        if (const char* e = std::getenv("FRIDAY_KV")) kvs = e;
        KvCacheMode kvm = KvCacheMode::Auto;
        if (kvs == "f16") kvm = KvCacheMode::F16;
        else if (kvs == "q8") kvm = KvCacheMode::Q8;
        else if (kvs != "auto") throw std::runtime_error("unknown KV cache format '" + kvs + "' (auto, f16, q8)");
        gpu.reset(new Transformer(opt.model_path, opt.tensor_map, hidden, heads, kv, layers, eps, qt, io.get(),
                                  opt.vram_gb, opt.max_ctx, opt.cpu_threads, kvm));
        vocab = gpu->getVocabSize();
        if (vocab <= 0) throw std::runtime_error("cannot determine the vocabulary size (no LM head tensor)");
        if (cudaMalloc(&d_act, (size_t)hidden * 2) != cudaSuccess || cudaMalloc(&d_logits, (size_t)vocab * 2) != cudaSuccess ||
            cudaStreamCreate(&stream) != cudaSuccess)
            throw std::runtime_error("CUDA allocation failed for activation buffers");
        h_logits.resize(vocab);
        char buf[256];
        std::snprintf(buf, sizeof(buf), "CUDA backend: %d/%d layers on the GPU, %d on the CPU, %s KV cache",
                      gpu->getNumGpuLayers(), layers, gpu->getNumCpuLayers(), gpu->kvCacheInt8() ? "int8" : "fp16");
        summary = buf;
    }
#endif
};

Engine::Engine(const EngineOptions& opt) : impl_(new Impl) {
    impl_->opt = opt;
    impl_->kind = resolveBackend(opt.backend);
    impl_->io.reset(new AsyncIOManager(opt.model_path));
#if FRIDAY_HAVE_CUDA
    if (impl_->kind == BackendKind::Cuda) impl_->initCuda();
    else impl_->initCpu();
#else
    impl_->initCpu();
#endif
    std::printf("[Engine] %s\n", impl_->summary.c_str());
}

Engine::~Engine() {
#if FRIDAY_HAVE_CUDA
    if (impl_->d_act) cudaFree(impl_->d_act);
    if (impl_->d_logits) cudaFree(impl_->d_logits);
    if (impl_->stream) cudaStreamDestroy(impl_->stream);
#endif
}

int Engine::vocabSize() const { return impl_->vocab; }
int Engine::maxContext() const { return impl_->opt.max_ctx; }
bool Engine::usesCuda() const { return impl_->kind == BackendKind::Cuda; }
std::string Engine::summary() const { return impl_->summary; }


std::vector<Engine::GpuInfo> Engine::gpuInfo() {
    std::vector<GpuInfo> out;
#if FRIDAY_HAVE_CUDA
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { cudaGetLastError(); return out; }
    const GpuMonitor nvml;  // PCIe link via NVML (loaded at run time)
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
        GpuInfo g;
        g.name = p.name;
        g.compute_major = p.major;
        g.compute_minor = p.minor;
        g.multiprocessors = p.multiProcessorCount;
        size_t fr = 0, tot = 0;
        if (cudaSetDevice(i) == cudaSuccess && cudaMemGetInfo(&fr, &tot) == cudaSuccess) { g.free_bytes = fr; g.total_bytes = tot; }
        char bus[32];
        GpuStatic st;
        if (cudaDeviceGetPCIBusId(bus, sizeof(bus), i) == cudaSuccess && nvml.staticInfoByBusId(bus, &st)) {
            g.pcie_gen_max = st.pcie_gen_max;
            g.pcie_width_max = st.pcie_width_max;
        }
        out.push_back(g);
    }
#endif
    return out;
}

bool Engine::cudaCompiledIn() {
#if FRIDAY_HAVE_CUDA
    return true;
#else
    return false;
#endif
}

void Engine::step(int token, int pos, float* logits) {
    Impl& m = *impl_;
#if FRIDAY_HAVE_CUDA
    if (m.kind == BackendKind::Cuda) {
        half* act = static_cast<half*>(m.d_act);
        half* lg = static_cast<half*>(m.d_logits);
        if (!m.gpu->getEmbedding(token, act, m.vocab, m.stream))
            throw std::runtime_error("embedding failed for token " + std::to_string(token));
        if (cudaStreamSynchronize(m.stream) != cudaSuccess) throw std::runtime_error("CUDA error after embedding");
        cudaMemsetAsync(lg, 0, (size_t)m.vocab * 2, m.stream);
        m.gpu->forward(act, lg, 1, 0, pos, m.stream);
        if (cudaMemcpyAsync(m.h_logits.data(), lg, (size_t)m.vocab * 2, cudaMemcpyDeviceToHost, m.stream) != cudaSuccess ||
            cudaStreamSynchronize(m.stream) != cudaSuccess)
            throw std::runtime_error("CUDA error copying logits");
        for (int i = 0; i < m.vocab; ++i) logits[i] = friday_cpu::halfToFloat(m.h_logits[i]);
        return;
    }
#endif
    m.cpu->embed(token, m.cpu->hidden());
    m.cpu->run(pos);
    m.cpu->logits(m.cpu->hidden(), logits);
}

bool Engine::batchedPrefill() const {
#if FRIDAY_HAVE_CUDA
    const char* e = std::getenv("FRIDAY_PREFILL");
    if (e && e[0] == '0') return false;
    return impl_->kind == BackendKind::Cuda && impl_->gpu->prefillSupported();
#else
    return false;
#endif
}

void Engine::prefill(const int* tokens, int n, int pos, float* logits) {
    if (n <= 0) return;
#if FRIDAY_HAVE_CUDA
    Impl& m = *impl_;
    if (batchedPrefill()) {
        half* act = static_cast<half*>(m.d_act);
        half* lg = static_cast<half*>(m.d_logits);
        cudaMemsetAsync(lg, 0, (size_t)m.vocab * 2, m.stream);
        m.gpu->prefill(tokens, n, pos, act, lg, m.vocab, m.stream);
        if (cudaMemcpyAsync(m.h_logits.data(), lg, (size_t)m.vocab * 2, cudaMemcpyDeviceToHost, m.stream) != cudaSuccess ||
            cudaStreamSynchronize(m.stream) != cudaSuccess)
            throw std::runtime_error("CUDA error copying logits");
        for (int i = 0; i < m.vocab; ++i) logits[i] = friday_cpu::halfToFloat(m.h_logits[i]);
        return;
    }
#endif
    for (int i = 0; i < n; ++i) step(tokens[i], pos + i, logits);
}

}  // namespace friday
