#include "cpu_kquants.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace friday_cpu {

float halfToFloat(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t absx = x & 0x7FFFFFFFu;
    if (absx >= 0x7F800000u) return (uint16_t)(sign | 0x7C00u | (absx > 0x7F800000u ? 0x200u : 0));
    if (absx >= 0x477FF000u) return (uint16_t)(sign | 0x7C00u);  // overflow -> inf
    if (absx < 0x38800000u) {  // half subnormal or zero: value = n * 2^-24
        // Exact in fp32 (n < 1024); nearbyint rounds half-even. n == 1024 encodes
        // the smallest normal, which is also correct.
        const float a = std::fabs(f) * 16777216.0f;
        return (uint16_t)(sign | (uint32_t)std::nearbyint(a));
    }
    uint32_t h = ((absx >> 13) - (112u << 10));
    const uint32_t rem = absx & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t)(sign | h);
}

bool hasAvx2() {
#if defined(__AVX2__) && defined(_MSC_VER)
    int r[4];
    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] & (1 << 5)) != 0;
    __cpuid(r, 1);
    const bool fma = (r[2] & (1 << 12)) != 0;
    return avx2 && fma;
#elif defined(__AVX2__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

void quantizeActivation(const float* x, int K, Q8Activation& out) {
    if (K % 256 != 0) throw std::runtime_error("quantizeActivation: K must be a multiple of 256");
    out.K = K;
    out.q.resize(K);
    out.d.resize(K / 32);
    out.bsum16.resize(K / 16);
#if defined(__AVX2__)
    static const bool avx2 = hasAvx2();
    if (avx2) {
        const __m256 sign = _mm256_set1_ps(-0.0f);
        for (int b = 0; b < K / 32; ++b) {
            const float* p = x + b * 32;
            __m256 v[4];
            __m256 mx = _mm256_setzero_ps();
            for (int i = 0; i < 4; ++i) {
                v[i] = _mm256_loadu_ps(p + 8 * i);
                mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, v[i]));
            }
            __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
            m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
            const float amax = _mm_cvtss_f32(m4);
            const float d = amax / 127.0f;
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            out.d[b] = d;
            const __m256 iv = _mm256_set1_ps(inv);
            __m256i q[4];
            for (int i = 0; i < 4; ++i)
                q[i] = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v[i], iv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            // 16-value sums
            for (int g = 0; g < 2; ++g) {
                const __m256i s8 = _mm256_add_epi32(q[2 * g], q[2 * g + 1]);
                __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s8), _mm256_extracti128_si256(s8, 1));
                s4 = _mm_hadd_epi32(s4, s4);
                s4 = _mm_hadd_epi32(s4, s4);
                out.bsum16[b * 2 + g] = (int16_t)_mm_cvtsi128_si32(s4);
            }
            // pack int32 -> int8 (saturating; |q| <= 127 by construction)
            const __m256i p01 = _mm256_packs_epi32(q[0], q[1]);
            const __m256i p23 = _mm256_packs_epi32(q[2], q[3]);
            __m256i pk = _mm256_packs_epi16(p01, p23);
            pk = _mm256_permutevar8x32_epi32(pk, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out.q.data() + b * 32), pk);
        }
        return;
    }
#endif
    for (int b = 0; b < K / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.0f;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        out.d[b] = d;
        for (int i = 0; i < 32; ++i) {
            int q = (int)std::nearbyint(x[b * 32 + i] * inv);  // default rounding: half-even
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            out.q[b * 32 + i] = (int8_t)q;
        }
    }
    for (int g = 0; g < K / 16; ++g) {
        int s = 0;
        for (int i = 0; i < 16; ++i) s += out.q[g * 16 + i];
        out.bsum16[g] = (int16_t)s;
    }
}

void roundToHalf(float* x, int n) {
    int i = 0;
#if defined(__AVX2__)
    static const bool avx2 = hasAvx2();
    if (avx2) {
        const __m256 hi = _mm256_set1_ps(65504.0f), lo = _mm256_set1_ps(-65504.0f), zero = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            v = _mm256_blendv_ps(v, zero, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));  // NaN -> 0
            v = _mm256_min_ps(_mm256_max_ps(v, lo), hi);
            v = _mm256_cvtph_ps(_mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
            _mm256_storeu_ps(x + i, v);
        }
    }
#endif
    for (; i < n; ++i) {
        float v = x[i];
        if (std::isnan(v)) v = 0.0f;
        else if (v > 65504.0f) v = 65504.0f;
        else if (v < -65504.0f) v = -65504.0f;
        x[i] = halfToFloat(floatToHalf(v));
    }
}

size_t rowBytes(KQuant t, int K) {
    const size_t nb = (size_t)K / 256;
    switch (t) {
        case KQuant::Q2_K: return nb * 84;
        case KQuant::Q3_K: return nb * 110;
        case KQuant::Q4_K: return nb * 144;
        case KQuant::Q6_K: return nb * 210;
        case KQuant::Q5_K: return nb * 176;
        case KQuant::Q8_0: return (size_t)K / 32 * 34;
    }
    return 0;
}

static inline float f16(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return halfToFloat(h);
}

static inline void q4kScaleMin(int is, const uint8_t* q, int& sc, int& m) {
    if (is < 4) {
        sc = q[is] & 63;
        m = q[is + 4] & 63;
    } else {
        sc = (q[is + 4] & 0xF) | ((q[is - 4] >> 6) << 4);
        m = (q[is + 4] >> 4) | ((q[is] >> 6) << 4);
    }
}

static inline void q3kScales(const uint8_t* s, int* out) {
    for (int j = 0; j < 16; ++j) {
        const int low = j < 8 ? (s[j] & 0xF) : (s[j - 8] >> 4);
        const int high = (s[8 + (j % 4)] >> (2 * (j / 4))) & 3;
        out[j] = (low | (high << 4)) - 32;
    }
}

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------
static float dotQ2KScalar(const uint8_t* row, const Q8Activation& x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t* b = row + sb * 84;
        const float d = f16(b + 80), dmin = f16(b + 82);
        float sd = 0.0f, sm = 0.0f;
        for (int h = 0; h < 2; ++h)
            for (int s = 0; s < 4; ++s)
                for (int c = 0; c < 2; ++c) {
                    const int g = h * 8 + 2 * s + c;
                    const int pos = sb * 256 + h * 128 + s * 32 + c * 16;
                    int dot = 0;
                    for (int i = 0; i < 16; ++i)
                        dot += ((b[16 + h * 32 + c * 16 + i] >> (2 * s)) & 3) * x.q[pos + i];
                    const float d8 = x.d[pos / 32];
                    sd += d8 * (float)(dot * (b[g] & 0xF));
                    sm += d8 * (float)(x.bsum16[pos / 16] * (b[g] >> 4));
                }
        acc += d * sd - dmin * sm;
    }
    return acc;
}

static float dotQ3KScalar(const uint8_t* row, const Q8Activation& x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t* b = row + sb * 110;
        int sc[16];
        q3kScales(b + 96, sc);
        const float d = f16(b + 108);
        float sd = 0.0f;
        for (int h = 0; h < 2; ++h)
            for (int s = 0; s < 4; ++s)
                for (int c = 0; c < 2; ++c) {
                    const int pos = sb * 256 + h * 128 + s * 32 + c * 16;
                    int dot = 0;
                    for (int i = 0; i < 16; ++i) {
                        const int l = c * 16 + i;
                        const int q = (b[32 + h * 32 + l] >> (2 * s)) & 3;
                        const int hb = (b[l] >> (h * 4 + s)) & 1;
                        dot += (q | (hb << 2)) * x.q[pos + i];
                    }
                    dot -= 4 * x.bsum16[pos / 16];
                    sd += x.d[pos / 32] * (float)(dot * sc[h * 8 + 2 * s + c]);
                }
        acc += d * sd;
    }
    return acc;
}

static float dotQ4KScalar(const uint8_t* row, const Q8Activation& x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t* b = row + sb * 144;
        const float d = f16(b), dmin = f16(b + 2);
        const uint8_t* scl = b + 4;
        const uint8_t* qs = b + 16;
        float sd = 0.0f, sm = 0.0f;
        for (int j = 0; j < 4; ++j) {
            int s_lo, m_lo, s_hi, m_hi;
            q4kScaleMin(2 * j, scl, s_lo, m_lo);
            q4kScaleMin(2 * j + 1, scl, s_hi, m_hi);
            const int pos = sb * 256 + 64 * j;
            int dlo = 0, dhi = 0;
            for (int l = 0; l < 32; ++l) {
                dlo += (qs[32 * j + l] & 0xF) * x.q[pos + l];
                dhi += (qs[32 * j + l] >> 4) * x.q[pos + 32 + l];
            }
            const int bs_lo = x.bsum16[pos / 16] + x.bsum16[pos / 16 + 1];
            const int bs_hi = x.bsum16[pos / 16 + 2] + x.bsum16[pos / 16 + 3];
            const float d8lo = x.d[pos / 32], d8hi = x.d[pos / 32 + 1];
            sd += d8lo * (float)(dlo * s_lo) + d8hi * (float)(dhi * s_hi);
            sm += d8lo * (float)(bs_lo * m_lo) + d8hi * (float)(bs_hi * m_hi);
        }
        acc += d * sd - dmin * sm;
    }
    return acc;
}

// Q6_K: ql[128] qh[64] int8 scales[16] fp16 d. For half h and l in 0..31 the
// four values at h*128 + 32p + l (p = 0..3) use ql[h*64 + l + 32*(p&1)]
// (low nibble for p < 2, high for p >= 2), 2 bits of qh[h*32 + l] at 2p, and
// scale h*8 + 2p + l/16; value = q - 32.
static float dotQ6KScalar(const uint8_t* row, const Q8Activation& x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t* b = row + sb * 210;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
        const float d = f16(b + 208);
        float sd = 0.0f;
        for (int h = 0; h < 2; ++h)
            for (int p = 0; p < 4; ++p)
                for (int c = 0; c < 2; ++c) {
                    const int pos = sb * 256 + h * 128 + p * 32 + c * 16;
                    int dot = 0;
                    for (int i = 0; i < 16; ++i) {
                        const int l = c * 16 + i;
                        const uint8_t qlb = b[h * 64 + l + 32 * (p & 1)];
                        const int lo = p < 2 ? (qlb & 0xF) : (qlb >> 4);
                        const int q = (lo | (((b[128 + h * 32 + l] >> (2 * p)) & 3) << 4)) - 32;
                        dot += q * x.q[pos + i];
                    }
                    sd += x.d[pos / 32] * (float)(dot * sc[h * 8 + 2 * p + c]);
                }
        acc += d * sd;
    }
    return acc;
}

static float dotQ5KScalar(const uint8_t* row, const Q8Activation& x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t* b = row + sb * 176;
        const float d = f16(b), dmin = f16(b + 2);
        const uint8_t* scl = b + 4;
        const uint8_t* qh = b + 16;
        const uint8_t* qs = b + 48;
        float sd = 0.0f, sm = 0.0f;
        for (int j = 0; j < 4; ++j) {
            int s_lo, m_lo, s_hi, m_hi;
            q4kScaleMin(2 * j, scl, s_lo, m_lo);
            q4kScaleMin(2 * j + 1, scl, s_hi, m_hi);
            const int pos = sb * 256 + 64 * j;
            int dlo = 0, dhi = 0;
            for (int l = 0; l < 32; ++l) {
                dlo += ((qs[32 * j + l] & 0xF) | (((qh[l] >> (2 * j)) & 1) << 4)) * x.q[pos + l];
                dhi += ((qs[32 * j + l] >> 4) | (((qh[l] >> (2 * j + 1)) & 1) << 4)) * x.q[pos + 32 + l];
            }
            const int bs_lo = x.bsum16[pos / 16] + x.bsum16[pos / 16 + 1];
            const int bs_hi = x.bsum16[pos / 16 + 2] + x.bsum16[pos / 16 + 3];
            const float d8lo = x.d[pos / 32], d8hi = x.d[pos / 32 + 1];
            sd += d8lo * (float)(dlo * s_lo) + d8hi * (float)(dhi * s_hi);
            sm += d8lo * (float)(bs_lo * m_lo) + d8hi * (float)(bs_hi * m_hi);
        }
        acc += d * sd - dmin * sm;
    }
    return acc;
}

static float dotQ80Scalar(const uint8_t* row, const Q8Activation& x, int nblk) {
    float acc = 0.0f;
    for (int bk = 0; bk < nblk; ++bk) {
        const uint8_t* b = row + bk * 34;
        const int8_t* q = reinterpret_cast<const int8_t*>(b + 2);
        int dot = 0;
        for (int i = 0; i < 32; ++i) dot += q[i] * x.q[bk * 32 + i];
        acc += f16(b) * x.d[bk] * (float)dot;
    }
    return acc;
}

void gemvRowsScalar(KQuant t, const uint8_t* W, const Q8Activation& x, float* y, int r0, int r1) {
    const int nb = x.K / 256;
    const size_t rb = rowBytes(t, x.K);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = W + (size_t)r * rb;
        switch (t) {
            case KQuant::Q2_K: y[r] = dotQ2KScalar(row, x, nb); break;
            case KQuant::Q3_K: y[r] = dotQ3KScalar(row, x, nb); break;
            case KQuant::Q4_K: y[r] = dotQ4KScalar(row, x, nb); break;
            case KQuant::Q6_K: y[r] = dotQ6KScalar(row, x, nb); break;
            case KQuant::Q5_K: y[r] = dotQ5KScalar(row, x, nb); break;
            case KQuant::Q8_0: y[r] = dotQ80Scalar(row, x, x.K / 32); break;
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2
// ---------------------------------------------------------------------------
#if defined(__AVX2__)
static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline float f16c(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}

// 8 vectors of 8 int32 -> one vector whose lane k is the sum of all lanes of v[k].
static inline __m256i reduce8(const __m256i* v) {
    const __m256i a01 = _mm256_hadd_epi32(v[0], v[1]);
    const __m256i a23 = _mm256_hadd_epi32(v[2], v[3]);
    const __m256i a45 = _mm256_hadd_epi32(v[4], v[5]);
    const __m256i a67 = _mm256_hadd_epi32(v[6], v[7]);
    const __m256i b03 = _mm256_hadd_epi32(a01, a23);  // lane: [v0..v3 low halves | v0..v3 high halves]
    const __m256i b47 = _mm256_hadd_epi32(a45, a67);
    const __m128i s03 = _mm_add_epi32(_mm256_castsi256_si128(b03), _mm256_extracti128_si256(b03, 1));
    const __m128i s47 = _mm_add_epi32(_mm256_castsi256_si128(b47), _mm256_extracti128_si256(b47, 1));
    return _mm256_set_m128i(s47, s03);
}

// pshufb controls: from 8 int16 scales replicated in both 128-bit lanes, pick
// scale 2s (low lane, 8 copies) and 2s + 1 (high lane, 8 copies).
static inline __m256i pairControl(int s) {
    const char lo0 = (char)(4 * s), lo1 = (char)(4 * s + 1), hi0 = (char)(4 * s + 2), hi1 = (char)(4 * s + 3);
    return _mm256_setr_epi8(lo0, lo1, lo0, lo1, lo0, lo1, lo0, lo1, lo0, lo1, lo0, lo1, lo0, lo1, lo0, lo1,
                            hi0, hi1, hi0, hi1, hi0, hi1, hi0, hi1, hi0, hi1, hi0, hi1, hi0, hi1, hi0, hi1);
}

static float dotQ2KAvx2(const uint8_t* row, const Q8Activation& x, int nb) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i ctl[4] = {pairControl(0), pairControl(1), pairControl(2), pairControl(3)};
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t* b = row + sb * 84;
        const float d = f16c(b + 80), dmin = f16c(b + 82);
        const int8_t* xq = x.q.data() + sb * 256;
        const __m256 xd = _mm256_loadu_ps(x.d.data() + sb * 8);
        const __m128i sraw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
        // mins: lane k = m[2k] * bsum[2k] + m[2k+1] * bsum[2k+1]  (32-value block k)
        const __m256i mins16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sraw, 4), m4));
        const __m256i bs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.bsum16.data() + sb * 16));
        const __m256i msum = _mm256_madd_epi16(mins16, bs);
        const __m128i sc8 = _mm_and_si128(sraw, m4);
        __m256i blk[8];
        for (int h = 0; h < 2; ++h) {
            const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 16 + h * 32));
            const __m256i sch = _mm256_broadcastsi128_si256(_mm_cvtepu8_epi16((h == 0 ? sc8 : _mm_srli_si128(sc8, 8))));
            for (int s = 0; s < 4; ++s) {
                const __m256i v = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * s), m3);
                const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + h * 128 + s * 32));
                blk[h * 4 + s] = _mm256_madd_epi16(_mm256_maddubs_epi16(v, xv), _mm256_shuffle_epi8(sch, ctl[s]));
            }
        }
        const __m256i main = reduce8(blk);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(main), _mm256_mul_ps(xd, _mm256_set1_ps(d)), acc);
        acc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(msum), _mm256_mul_ps(xd, _mm256_set1_ps(dmin)), acc);
    }
    return hsum256(acc);
}

static float dotQ3KAvx2(const uint8_t* row, const Q8Activation& x, int nb) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m1 = _mm256_set1_epi8(1);
    const __m256i ctl[4] = {pairControl(0), pairControl(1), pairControl(2), pairControl(3)};
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t* b = row + sb * 110;
        const float d = f16c(b + 108);
        const int8_t* xq = x.q.data() + sb * 256;
        const __m256 xd = _mm256_loadu_ps(x.d.data() + sb * 8);
        // 16 6-bit scales (GGML layout), then -32.
        uint32_t aux[4];
        std::memcpy(aux, b + 96, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const __m128i sc8 = _mm_sub_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(aux)), _mm_set1_epi8(32));
        // -4 offset of every weight: lane k = sum over the two groups of block k of sc_g * bsum_g
        const __m256i bs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.bsum16.data() + sb * 16));
        const __m256i osum = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc8), bs);
        const __m256i hm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
        __m256i blk[8];
        for (int h = 0; h < 2; ++h) {
            const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 32 + h * 32));
            const __m256i sch = _mm256_broadcastsi128_si256(_mm_cvtepi8_epi16((h == 0 ? sc8 : _mm_srli_si128(sc8, 8))));
            for (int s = 0; s < 4; ++s) {
                const __m256i lo = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * s), m3);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(hm, h * 4 + s), m1);
                const __m256i v = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 2));
                const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + h * 128 + s * 32));
                blk[h * 4 + s] = _mm256_madd_epi16(_mm256_maddubs_epi16(v, xv), _mm256_shuffle_epi8(sch, ctl[s]));
            }
        }
        const __m256i tot = _mm256_sub_epi32(reduce8(blk), _mm256_slli_epi32(osum, 2));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot), _mm256_mul_ps(xd, _mm256_set1_ps(d)), acc);
    }
    return hsum256(acc);
}

static float dotQ4KAvx2(const uint8_t* row, const Q8Activation& x, int nb) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t* b = row + sb * 144;
        const float d = f16c(b), dmin = f16c(b + 2);
        const uint8_t* scl = b + 4;
        const int8_t* xq = x.q.data() + sb * 256;
        const __m256 xd = _mm256_loadu_ps(x.d.data() + sb * 8);
        const int16_t* bsp = x.bsum16.data() + sb * 16;
        alignas(32) int32_t mterm[8];
        __m256i blk[8];
        for (int j = 0; j < 4; ++j) {
            int s_lo, m_lo, s_hi, m_hi;
            q4kScaleMin(2 * j, scl, s_lo, m_lo);
            q4kScaleMin(2 * j + 1, scl, s_hi, m_hi);
            mterm[2 * j] = m_lo * (bsp[4 * j] + bsp[4 * j + 1]);
            mterm[2 * j + 1] = m_hi * (bsp[4 * j + 2] + bsp[4 * j + 3]);
            const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 16 + 32 * j));
            const __m256i lo = _mm256_and_si256(qv, m4);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), m4);
            const __m256i xl = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + 64 * j));
            const __m256i xh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + 64 * j + 32));
            blk[2 * j] = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xl), _mm256_set1_epi16((short)s_lo));
            blk[2 * j + 1] = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xh), _mm256_set1_epi16((short)s_hi));
        }
        const __m256i main = reduce8(blk);
        const __m256i mins = _mm256_load_si256(reinterpret_cast<const __m256i*>(mterm));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(main), _mm256_mul_ps(xd, _mm256_set1_ps(d)), acc);
        acc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(mins), _mm256_mul_ps(xd, _mm256_set1_ps(dmin)), acc);
    }
    return hsum256(acc);
}
static float dotQ6KAvx2(const uint8_t* row, const Q8Activation& x, int nb) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ctl[4] = {pairControl(0), pairControl(1), pairControl(2), pairControl(3)};
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t* b = row + sb * 210;
        const float d = f16c(b + 208);
        const int8_t* xq = x.q.data() + sb * 256;
        const __m256 xd = _mm256_loadu_ps(x.d.data() + sb * 8);
        const __m128i sc8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 192));
        // -32 offset of every weight: lane k = sum over the two groups of block k of sc_g * bsum_g
        const __m256i bs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.bsum16.data() + sb * 16));
        const __m256i osum = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc8), bs);
        __m256i blk[8];
        for (int h = 0; h < 2; ++h) {
            const __m256i ql0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + h * 64));
            const __m256i ql1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + h * 64 + 32));
            const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 128 + h * 32));
            const __m256i sch = _mm256_broadcastsi128_si256(_mm_cvtepi8_epi16(h == 0 ? sc8 : _mm_srli_si128(sc8, 8)));
            const __m256i lows[4] = {_mm256_and_si256(ql0, m4), _mm256_and_si256(ql1, m4),
                                     _mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4),
                                     _mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4)};
            for (int p = 0; p < 4; ++p) {
                const __m256i hi2 = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * p), m3);
                const __m256i v = _mm256_or_si256(lows[p], _mm256_slli_epi16(hi2, 4));  // q + 32, 0..63
                const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + h * 128 + p * 32));
                blk[h * 4 + p] = _mm256_madd_epi16(_mm256_maddubs_epi16(v, xv), _mm256_shuffle_epi8(sch, ctl[p]));
            }
        }
        const __m256i tot = _mm256_sub_epi32(reduce8(blk), _mm256_slli_epi32(osum, 5));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot), _mm256_mul_ps(xd, _mm256_set1_ps(d)), acc);
    }
    return hsum256(acc);
}
static float dotQ5KAvx2(const uint8_t* row, const Q8Activation& x, int nb) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m1 = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t* b = row + sb * 176;
        const float d = f16c(b), dmin = f16c(b + 2);
        const uint8_t* scl = b + 4;
        const int8_t* xq = x.q.data() + sb * 256;
        const __m256 xd = _mm256_loadu_ps(x.d.data() + sb * 8);
        const int16_t* bsp = x.bsum16.data() + sb * 16;
        const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 16));
        alignas(32) int32_t mterm[8];
        __m256i blk[8];
        for (int j = 0; j < 4; ++j) {
            int s_lo, m_lo, s_hi, m_hi;
            q4kScaleMin(2 * j, scl, s_lo, m_lo);
            q4kScaleMin(2 * j + 1, scl, s_hi, m_hi);
            mterm[2 * j] = m_lo * (bsp[4 * j] + bsp[4 * j + 1]);
            mterm[2 * j + 1] = m_hi * (bsp[4 * j + 2] + bsp[4 * j + 3]);
            const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 48 + 32 * j));
            const __m256i hlo = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * j), m1), 4);
            const __m256i hhi = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * j + 1), m1), 4);
            const __m256i lo = _mm256_or_si256(_mm256_and_si256(qv, m4), hlo);
            const __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qv, 4), m4), hhi);
            const __m256i xl = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + 64 * j));
            const __m256i xh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + 64 * j + 32));
            blk[2 * j] = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xl), _mm256_set1_epi16((short)s_lo));
            blk[2 * j + 1] = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xh), _mm256_set1_epi16((short)s_hi));
        }
        const __m256i main = reduce8(blk);
        const __m256i mins = _mm256_load_si256(reinterpret_cast<const __m256i*>(mterm));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(main), _mm256_mul_ps(xd, _mm256_set1_ps(d)), acc);
        acc = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(mins), _mm256_mul_ps(xd, _mm256_set1_ps(dmin)), acc);
    }
    return hsum256(acc);
}

static float dotQ80Avx2(const uint8_t* row, const Q8Activation& x, int nblk) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int bk = 0; bk < nblk; ++bk) {
        const uint8_t* b = row + bk * 34;
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 2));
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.q.data() + bk * 32));
        // signed x signed via |w| * sign(x, w)
        const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(xv, w));
        const __m256i i32 = _mm256_madd_epi16(p, ones);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32), _mm256_set1_ps(f16c(b) * x.d[bk]), acc);
    }
    return hsum256(acc);
}
#endif  // __AVX2__

void gemvRows(KQuant t, const uint8_t* W, const Q8Activation& x, float* y, int r0, int r1) {
#if defined(__AVX2__)
    static const bool avx2 = hasAvx2();
    if (avx2) {
        const int nb = x.K / 256;
        const size_t rb = rowBytes(t, x.K);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* row = W + (size_t)r * rb;
            switch (t) {
                case KQuant::Q2_K: y[r] = dotQ2KAvx2(row, x, nb); break;
                case KQuant::Q3_K: y[r] = dotQ3KAvx2(row, x, nb); break;
                case KQuant::Q4_K: y[r] = dotQ4KAvx2(row, x, nb); break;
                case KQuant::Q6_K: y[r] = dotQ6KAvx2(row, x, nb); break;
                case KQuant::Q5_K: y[r] = dotQ5KAvx2(row, x, nb); break;
                case KQuant::Q8_0: y[r] = dotQ80Avx2(row, x, x.K / 32); break;
            }
        }
        return;
    }
#endif
    gemvRowsScalar(t, W, x, y, r0, r1);
}

}  // namespace friday_cpu
