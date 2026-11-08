#ifndef FRIDAY_CPU_BACKEND_H
#define FRIDAY_CPU_BACKEND_H

// CPU backend: runs a range of transformer layers on the CPU from host RAM
// (and, for a CPU-only engine, the embedding lookup and the LM head). Same
// math as the CUDA path: int8 activations for the K-quant GEMVs and fp16
// rounding at the same points. Pure C++ (no CUDA dependency).

#include <memory>
#include <string>
#include <vector>

#include "model_layout.h"

class AsyncIOManager;

namespace friday_cpu { class ThreadPool; }

namespace friday {

struct CpuModelParams {
    int hidden_dim = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int max_ffn_dim = 0;            // largest ffn_gate rows
    float rms_norm_epsilon = 0.0f;
    float rope_freq_base = 1000000.0f;
    bool rope_neox = true;
    std::vector<float> rope_freqs;  // optional per-pair factors (Llama 3)
    int max_ctx = 2048;
};

class CpuLayerRunner {
public:
    // Loads layers [first, last) into host RAM. threads <= 0: 2/3 of the logical CPUs.
    CpuLayerRunner(const ModelLayout& layout, AsyncIOManager& io, const CpuModelParams& p, int first, int last,
                   int threads);
    ~CpuLayerRunner();

    int firstLayer() const { return first_; }
    int lastLayer() const { return last_; }
    size_t weightBytes() const { return weight_bytes_; }
    int threads() const;

    // Hidden state (hidden_dim floats, fp16-rounded values) read and written by run().
    float* hidden() { return x_.data(); }
    void run(int pos);  // layers [first, last) at position pos

    // CPU-only engine: embedding table, output norm and LM head in host RAM.
    void loadHead();
    bool hasHead() const { return !head_.empty(); }
    void embed(int token, float* out) const;     // dequantized row, fp16-rounded
    void logits(const float* x, float* out);     // final norm + LM head, fp16-rounded
    int vocabSize() const { return vocab_; }

private:
    void layer(int l, int pos);

    const ModelLayout& layout_;
    AsyncIOManager& io_;
    CpuModelParams p_;
    int first_, last_;
    size_t weight_bytes_ = 0;
    std::vector<std::unique_ptr<unsigned char[]>> weights_;   // [l - first]
    std::vector<float> k_cache_, v_cache_;                    // [l - first][max_ctx][kv_dim]
    std::vector<float> x_, norm_, q_, k_, v_, attn_, o_, gate_, up_, down_;
    std::unique_ptr<friday_cpu::ThreadPool> pool_;

    // head
    std::vector<unsigned char> embd_, out_norm_raw_, head_;
    std::string embd_dtype_, head_dtype_;
    size_t embd_row_bytes_ = 0;
    int vocab_ = 0;
};

}  // namespace friday

#endif  // FRIDAY_CPU_BACKEND_H
