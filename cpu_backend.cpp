#include "cpu_backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "cpu_kquants.h"
#include "cpu_ops.h"
#include "cpu_threadpool.h"
#include "io_manager.h"

namespace friday {

using namespace friday_cpu;

namespace {
KQuant kquantOf(const std::string& dtype, const std::string& what) {
    KQuant t;
    if (!kquantFromDtype(dtype, &t))
        throw std::runtime_error("CPU backend: unsupported weight type " + dtype + " for " + what);
    return t;
}
}  // namespace

CpuLayerRunner::CpuLayerRunner(const ModelLayout& layout, AsyncIOManager& io, const CpuModelParams& p, int first,
                               int last, int threads)
    : layout_(layout), io_(io), p_(p), first_(first), last_(last) {
    const int n = last - first;
    weights_.resize(n);
    for (int i = 0; i < n; ++i) {
        const auto& li = layout_.layers[first + i];
        weights_[i].reset(new unsigned char[li.size_bytes]);
        if (!layout_.readLayer(io_, first + i, weights_[i].get()))
            throw std::runtime_error("Failed to read layer " + std::to_string(first + i));
        weight_bytes_ += li.size_bytes;
    }
    const int kv_dim = p_.num_kv_heads * p_.head_dim;
    k_cache_.assign((size_t)n * p_.max_ctx * kv_dim, 0.0f);
    v_cache_.assign((size_t)n * p_.max_ctx * kv_dim, 0.0f);
    x_.resize(p_.hidden_dim); norm_.resize(p_.hidden_dim); q_.resize(p_.num_heads * p_.head_dim);
    k_.resize(kv_dim); v_.resize(kv_dim); attn_.resize(p_.num_heads * p_.head_dim); o_.resize(p_.hidden_dim);
    gate_.resize(p_.max_ffn_dim); up_.resize(p_.max_ffn_dim); down_.resize(p_.hidden_dim);
    if (threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        // Measured on a 6-core/12-thread Ryzen 5600X: 8 threads read ~34 GB/s
        // (6: ~29, 12: ~33). Default to 2/3 of the logical CPUs.
        threads = hw >= 3 ? (int)(hw * 2 / 3) : 1;
    }
    pool_.reset(new ThreadPool(threads));
}

CpuLayerRunner::~CpuLayerRunner() = default;

int CpuLayerRunner::threads() const { return pool_->size(); }

void CpuLayerRunner::run(int pos) {
    if (pos < 0 || pos >= p_.max_ctx)
        throw std::runtime_error("context length exceeded: position " + std::to_string(pos) + " >= max context " +
                                 std::to_string(p_.max_ctx));
    for (int l = first_; l < last_; ++l) layer(l, pos);
}

void CpuLayerRunner::layer(int layer_idx, int pos) {
    const int ci = layer_idx - first_;
    const unsigned char* base = weights_[ci].get();
    const auto& li = layout_.layers[layer_idx];
    const std::string pre = "blk." + std::to_string(layer_idx) + ".";
    const int H = p_.hidden_dim, kv_dim = p_.num_kv_heads * p_.head_dim, hd = p_.head_dim;

    auto ptr = [&](const char* n) { return base + li.tensor_rel_offsets.at(pre + n); };
    auto gemv = [&](const char* n, const Q8Activation& xa, float* y) {
        const TensorInfo& ti = layout_.tensors.at(pre + n);
        const int K = (int)ti.shape[0], N = (int)ti.shape[1];
        const KQuant t = kquantOf(ti.dtype, pre + n);
        if (xa.K != K) throw std::runtime_error("CPU layers: shape mismatch for " + pre + n);
        const uint8_t* W = ptr(n);
        pool_->parallelFor(N, 32, [&](int b, int e) { gemvRows(t, W, xa, y, b, e); });
        roundToHalf(y, N);
        return N;
    };
    auto addBias = [&](const char* n, float* y, int N) {
        if (!li.tensor_rel_offsets.count(pre + n)) return;
        const float* b = reinterpret_cast<const float*>(ptr(n));
        for (int i = 0; i < N; ++i) y[i] += std::isnan(b[i]) ? 0.0f : b[i];
        roundToHalf(y, N);
    };

    Q8Activation xa;
    // Attention block
    rmsNorm(x_.data(), reinterpret_cast<const float*>(ptr("attn_norm.weight")), norm_.data(), H, p_.rms_norm_epsilon);
    quantizeActivation(norm_.data(), H, xa);
    addBias("attn_q.bias", q_.data(), gemv("attn_q.weight", xa, q_.data()));
    addBias("attn_k.bias", k_.data(), gemv("attn_k.weight", xa, k_.data()));
    addBias("attn_v.bias", v_.data(), gemv("attn_v.weight", xa, v_.data()));
    if (li.tensor_rel_offsets.count(pre + "attn_q_norm.weight") && li.tensor_rel_offsets.count(pre + "attn_k_norm.weight")) {
        // Qwen3: per-head RMSNorm of Q and K (same math as headRmsNormKernel).
        const float* wq = reinterpret_cast<const float*>(ptr("attn_q_norm.weight"));
        const float* wk = reinterpret_cast<const float*>(ptr("attn_k_norm.weight"));
        std::vector<float> tmp(hd);
        for (int h = 0; h < p_.num_heads; ++h) {
            rmsNorm(q_.data() + h * hd, wq, tmp.data(), hd, p_.rms_norm_epsilon);
            std::memcpy(q_.data() + h * hd, tmp.data(), hd * sizeof(float));
        }
        for (int h = 0; h < p_.num_kv_heads; ++h) {
            rmsNorm(k_.data() + h * hd, wk, tmp.data(), hd, p_.rms_norm_epsilon);
            std::memcpy(k_.data() + h * hd, tmp.data(), hd * sizeof(float));
        }
    }
    rope(q_.data(), k_.data(), hd, p_.num_heads, p_.num_kv_heads, pos, p_.rope_freq_base, p_.rope_neox,
         p_.rope_freqs.empty() ? nullptr : p_.rope_freqs.data());
    float* k_layer = k_cache_.data() + (size_t)ci * p_.max_ctx * kv_dim;
    float* v_layer = v_cache_.data() + (size_t)ci * p_.max_ctx * kv_dim;
    std::memcpy(k_layer + (size_t)pos * kv_dim, k_.data(), kv_dim * sizeof(float));
    std::memcpy(v_layer + (size_t)pos * kv_dim, v_.data(), kv_dim * sizeof(float));
    decodeAttention(q_.data(), k_layer, v_layer, attn_.data(), pos + 1, p_.num_heads, p_.num_kv_heads, hd, *pool_);
    quantizeActivation(attn_.data(), p_.num_heads * hd, xa);
    gemv("attn_output.weight", xa, o_.data());
    addResidual(x_.data(), o_.data(), H);

    // MLP block (SwiGLU)
    rmsNorm(x_.data(), reinterpret_cast<const float*>(ptr("ffn_norm.weight")), norm_.data(), H, p_.rms_norm_epsilon);
    quantizeActivation(norm_.data(), H, xa);
    const int I = gemv("ffn_gate.weight", xa, gate_.data());
    gemv("ffn_up.weight", xa, up_.data());
    siluMul(gate_.data(), up_.data(), I);
    quantizeActivation(gate_.data(), I, xa);
    gemv("ffn_down.weight", xa, down_.data());
    addResidual(x_.data(), down_.data(), H);
}

void CpuLayerRunner::loadHead() {
    auto load = [&](const TensorInfo& t, std::vector<unsigned char>& dst) {
        dst.resize(t.size_bytes);
        if (!layout_.readTensor(io_, t, dst.data())) throw std::runtime_error("cannot read " + t.name);
    };
    const TensorInfo* emb = layout_.find("token_embd.weight");
    const TensorInfo* norm = layout_.find("output_norm.weight");
    if (!emb || !norm) throw std::runtime_error("CPU backend: model has no token_embd / output_norm");
    // Tied embeddings: models without output.weight use the embedding table as LM head.
    const TensorInfo* head = layout_.find("output.weight");
    load(*emb, embd_);
    embd_dtype_ = emb->dtype;
    embd_row_bytes_ = rowBytesOf(emb->dtype, p_.hidden_dim);
    if (embd_row_bytes_ == 0) throw std::runtime_error("CPU backend: unsupported embedding type " + emb->dtype);
    vocab_ = (int)emb->shape[1];
    load(*norm, out_norm_raw_);
    if (head) {
        load(*head, head_);
        head_dtype_ = head->dtype;
        vocab_ = (int)head->shape[1];
    } else {
        head_ = embd_;
        head_dtype_ = embd_dtype_;
    }
    kquantOf(head_dtype_, "LM head");
}

void CpuLayerRunner::embed(int token, float* out) const {
    if (token < 0 || (size_t)(token + 1) * embd_row_bytes_ > embd_.size())
        throw std::runtime_error("token " + std::to_string(token) + " is outside the embedding table");
    if (!dequantizeRow(embd_dtype_, embd_.data() + (size_t)token * embd_row_bytes_, out, p_.hidden_dim))
        throw std::runtime_error("CPU backend: cannot dequantize embedding type " + embd_dtype_);
    roundToHalf(out, p_.hidden_dim);  // the GPU path stores the embedding as fp16
}

void CpuLayerRunner::logits(const float* x, float* out) {
    rmsNorm(x, reinterpret_cast<const float*>(out_norm_raw_.data()), norm_.data(), p_.hidden_dim, p_.rms_norm_epsilon);
    Q8Activation xa;
    quantizeActivation(norm_.data(), p_.hidden_dim, xa);
    const KQuant t = kquantOf(head_dtype_, "LM head");
    const uint8_t* W = head_.data();
    pool_->parallelFor(vocab_, 64, [&](int b, int e) { gemvRows(t, W, xa, out, b, e); });
    roundToHalf(out, vocab_);  // the GPU path writes fp16 logits
}

}  // namespace friday
