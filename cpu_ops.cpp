#include "cpu_ops.h"

#include <cmath>
#include <vector>

#include "cpu_kquants.h"

namespace friday_cpu {

void rmsNorm(const float* x, const float* w, float* out, int n, float eps) {
    double ss = 0.0;  // GPU reduces in fp32 over 256 threads; double here is closer to exact
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        if (!std::isnan(v) && !std::isinf(v)) ss += (double)v * v;
    }
    const float rms = std::sqrt((float)(ss / n) + eps);
    const float inv = rms > 1e-12f ? 1.0f / rms : 1.0f;
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        if (std::isnan(v) || std::isinf(v)) v = 0.0f;
        out[i] = v * inv * w[i];
    }
    roundToHalf(out, n);
}

void ropeNeox(float* q, float* k, int head_dim, int nq, int nk, int pos) {
    const int half_dim = head_dim / 2;
    for (int i = 0; i < half_dim; ++i) {
        const float theta = (float)pos / std::pow(1000000.0f, (2.0f * (float)i) / (float)head_dim);
        const float c = std::cos(theta), s = std::sin(theta);
        for (int h = 0; h < nq; ++h) {
            float* p = q + h * head_dim;
            const float a = p[i], b = p[i + half_dim];
            p[i] = a * c - b * s;
            p[i + half_dim] = a * s + b * c;
        }
        for (int h = 0; h < nk; ++h) {
            float* p = k + h * head_dim;
            const float a = p[i], b = p[i + half_dim];
            p[i] = a * c - b * s;
            p[i + half_dim] = a * s + b * c;
        }
    }
    roundToHalf(q, nq * head_dim);
    roundToHalf(k, nk * head_dim);
}

void decodeAttention(const float* q, const float* k_cache, const float* v_cache, float* out, int seq,
                     int num_heads, int num_kv_heads, int head_dim, ThreadPool& pool) {
    const int kv_dim = num_kv_heads * head_dim;
    const int per_kv = num_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    pool.parallelFor(num_heads, 1, [&](int h0, int h1) {
        std::vector<float> s(seq);
        std::vector<float> acc(head_dim);
        for (int h = h0; h < h1; ++h) {
            const int kvh = h / per_kv;
            const float* qh = q + h * head_dim;
            float mx = -1e30f;
            for (int p = 0; p < seq; ++p) {
                const float* kp = k_cache + (size_t)p * kv_dim + kvh * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += qh[d] * kp[d];
                dot *= scale;
                if (std::isnan(dot)) dot = -1e30f;
                s[p] = dot;
                if (dot > mx) mx = dot;
            }
            float sum = 0.0f;
            for (int p = 0; p < seq; ++p) {
                const float df = s[p] - mx;
                s[p] = df < -50.0f ? 0.0f : std::exp(df);
                sum += s[p];
            }
            const float inv = sum > 1e-12f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < head_dim; ++d) acc[d] = 0.0f;
            for (int p = 0; p < seq; ++p) {
                const float w = s[p] * inv;
                const float* vp = v_cache + (size_t)p * kv_dim + kvh * head_dim;
                for (int d = 0; d < head_dim; ++d) acc[d] += w * vp[d];
            }
            for (int d = 0; d < head_dim; ++d) out[h * head_dim + d] = acc[d];
        }
    });
    roundToHalf(out, num_heads * head_dim);
}

void siluMul(float* gate, const float* up, int n) {
    for (int i = 0; i < n; ++i) {
        const float x = gate[i];
        float a;
        if (std::isnan(x)) a = 0.0f;
        else if (x > 20.0f) a = x;
        else if (x < -20.0f) a = 0.0f;
        else a = x / (1.0f + std::exp(-x));
        gate[i] = a;
    }
    roundToHalf(gate, n);  // GPU: SiLU result stored as fp16
    for (int i = 0; i < n; ++i) {
        const float b = std::isnan(up[i]) ? 0.0f : up[i];
        gate[i] *= b;
    }
    roundToHalf(gate, n);  // GPU: product stored as fp16
}

void addResidual(float* x, const float* a, int n) {
    for (int i = 0; i < n; ++i) {
        float t = std::isnan(x[i]) ? 0.0f : x[i];
        float b = std::isnan(a[i]) ? 0.0f : a[i];
        x[i] = t + b;
    }
    roundToHalf(x, n);
}

}  // namespace friday_cpu
