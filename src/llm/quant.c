/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * quant.c - quantized block formats and dot-product kernels (see quant.h).
 */
#include "quant.h"
#include "vmath.h"
#include "iq3s_grid.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

float janas_fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else { /* subnormal: normalise */
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            bits = sign | (exp << 23) | ((mant & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t janas_fp32_to_fp16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;

    if (((x >> 23) & 0xff) == 0xff) /* inf or nan */
        return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 0x1f)
        return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10)
            return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1)))
            half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1)))
        half++; /* may carry into the exponent, which is correct rounding */
    return (uint16_t)half;
}

static int use_avx2_fma; /* set by float_helpers_init, further down */

static void q8k_quantize_ref(const float *x, struct janas_block_q8k *y,
                             size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++, x += JANAS_QK) {
        float amax = 0.0f;
        for (int i = 0; i < JANAS_QK; i++)
            if (fabsf(x[i]) > amax)
                amax = fabsf(x[i]);
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        y[b].d = d;
        for (int i = 0; i < JANAS_QK; i++) {
            int q = (int)lrintf(x[i] * id);
            y[b].qs[i] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
        }
        for (int g = 0; g < JANAS_QK / 16; g++) {
            int s = 0;
            for (int i = 0; i < 16; i++)
                s += y[b].qs[g * 16 + i];
            y[b].bsums[g] = (int16_t)s;
        }
    }
}

static void q8k_quantize_det_ref(const float *x, struct janas_block_q8k *y,
                                 size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++, x += JANAS_QK) {
        float amax = 0.0f;
        for (int i = 0; i < JANAS_QK; i++)
            amax = fmaxf(amax, fabsf(x[i]));
        /* amax / 127 and its reciprocal without a division */
        float d = amax * 0.007874015718698502f;
        float id = d > 0.0f ? janas_recip_det(d) : 0.0f;
        y[b].d = d;
        for (int i = 0; i < JANAS_QK; i++) {
            float q = nearbyintf(x[i] * id);
            y[b].qs[i] = (int8_t)(q > 127.0f ? 127 : (q < -127.0f ? -127 : q));
        }
        for (int g = 0; g < JANAS_QK / 16; g++) {
            int s = 0;
            for (int i = 0; i < 16; i++)
                s += y[b].qs[g * 16 + i];
            y[b].bsums[g] = (int16_t)s;
        }
    }
}

#if defined(__x86_64__)
/*
 * The same, eight lanes at a time, and bit for bit the same answer. The
 * largest absolute value is a maximum, which is exact and does not care in
 * what order it is taken; d and its reciprocal stay the two scalar
 * operations they were; and _mm256_cvtps_epi32 rounds to nearest with ties
 * to even, which is what lrintf and nearbyintf do in the default rounding
 * mode. Worth doing because this runs on one thread while the eleven others
 * wait for it: on an activation of 4096 values it measured 6.83 us, one
 * third of a millisecond a token.
 */
__attribute__((target("avx2,fma"))) static float q8k_amax(const float *x)
{
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 a = _mm256_setzero_ps();
    for (int i = 0; i < JANAS_QK; i += 8)
        a = _mm256_max_ps(a, _mm256_and_ps(_mm256_loadu_ps(x + i), mask));
    __m128 m =
        _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

/* One block, given the scale and its reciprocal already computed. */
__attribute__((target("avx2,fma"))) static void
q8k_block_avx2(const float *x, struct janas_block_q8k *y, float id)
{
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    const __m256 vid = _mm256_set1_ps(id);
    for (int g = 0; g < JANAS_QK / 16; g++) {
        __m256i i0 =
            _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + g * 16), vid));
        __m256i i1 = _mm256_cvtps_epi32(
            _mm256_mul_ps(_mm256_loadu_ps(x + g * 16 + 8), vid));
        i0 = _mm256_min_epi32(_mm256_max_epi32(i0, lo), hi);
        i1 = _mm256_min_epi32(_mm256_max_epi32(i1, lo), hi);
        __m256i s = _mm256_add_epi32(i0, i1);
        __m128i h = _mm_add_epi32(_mm256_castsi256_si128(s),
                                  _mm256_extracti128_si256(s, 1));
        h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0x4e));
        h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0xb1));
        y->bsums[g] = (int16_t)_mm_cvtsi128_si32(h);
        /* packs leaves the two halves lane by lane: 0,2,1,3 puts them back */
        __m256i p = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xd8);
        _mm_storeu_si128((__m128i *)(y->qs + g * 16),
                         _mm_packs_epi16(_mm256_castsi256_si128(p),
                                         _mm256_extracti128_si256(p, 1)));
    }
}

__attribute__((target("avx2,fma"))) static void
q8k_quantize_avx2(const float *x, struct janas_block_q8k *y, size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++, x += JANAS_QK) {
        float amax = q8k_amax(x);
        float d = amax / 127.0f;
        y[b].d = d;
        q8k_block_avx2(x, &y[b], d > 0.0f ? 1.0f / d : 0.0f);
    }
}

__attribute__((target("avx2,fma"))) static void
q8k_quantize_det_avx2(const float *x, struct janas_block_q8k *y, size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++, x += JANAS_QK) {
        float d = q8k_amax(x) * 0.007874015718698502f;
        y[b].d = d;
        q8k_block_avx2(x, &y[b], d > 0.0f ? janas_recip_det(d) : 0.0f);
    }
}
#endif

void janas_q8k_quantize(const float *x, struct janas_block_q8k *y, size_t n)
{
#if defined(__x86_64__)
    if (use_avx2_fma) {
        q8k_quantize_avx2(x, y, n);
        return;
    }
#endif
    q8k_quantize_ref(x, y, n);
}

void janas_q8k_quantize_det(const float *x, struct janas_block_q8k *y, size_t n)
{
#if defined(__x86_64__)
    if (use_avx2_fma) {
        q8k_quantize_det_avx2(x, y, n);
        return;
    }
#endif
    q8k_quantize_det_ref(x, y, n);
}

/* Unpacks the eight 6-bit scales and eight 6-bit mins of a Q4_K block. */
static inline void unpack_scales(const uint8_t *q, uint8_t *sc, uint8_t *m)
{
    for (int j = 0; j < 4; j++) {
        sc[j] = q[j] & 63;
        m[j] = q[j + 4] & 63;
    }
    for (int j = 4; j < 8; j++) {
        sc[j] = (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4);
        m[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void janas_q4k_dequantize(const struct janas_block_q4k *x, float *y, size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++) {
        uint8_t sc[8], m[8];
        unpack_scales(x[b].scales, sc, m);
        float d = janas_fp16_to_fp32(x[b].d);
        float dmin = janas_fp16_to_fp32(x[b].dmin);
        for (int s = 0; s < 8; s++) {
            const uint8_t *q = x[b].qs + (s / 2) * 32;
            int shift = (s & 1) * 4;
            for (int l = 0; l < 32; l++)
                *y++ = d * sc[s] * ((q[l] >> shift) & 0xf) - dmin * m[s];
        }
    }
}

/*
 * Affine fit of one sub-block: x ~ a * q - b with q in 0..15, a >= 0 and
 * b >= 0 (Q4_K stores the offset as a subtracted min). Candidate ranges
 * shrink the data range from either end; for each, q is rounded and (a, b)
 * refit by least squares. Returns the squared error of the best candidate.
 */
static double q4k_fit_sub(const float *x, int n, float *a_out, float *b_out)
{
    float lo = 0.0f, hi = x[0];
    for (int i = 0; i < n; i++) {
        if (x[i] < lo)
            lo = x[i];
        if (x[i] > hi)
            hi = x[i];
    }
    if (hi < 0.0f)
        hi = 0.0f;
    *a_out = 0.0f;
    *b_out = -lo;
    if (hi - lo <= 0.0f)
        return 0.0;
    double best = INFINITY;
    float step = (hi - lo) / 64.0f;
    for (int i = 0; i <= 6; i++)
        for (int j = 0; j <= 6; j++) {
            float l = lo + i * step, h = hi - j * step;
            if (h <= l)
                continue;
            float inv = 15.0f / (h - l);
            /* least squares of x ~ a q + c over the rounded q, c <= 0 */
            double sq = 0, sqq = 0, sx = 0, sqx = 0;
            int qs[64];
            for (int k = 0; k < n; k++) {
                int q = (int)lrintf((x[k] - l) * inv);
                q = q < 0 ? 0 : (q > 15 ? 15 : q);
                qs[k] = q;
                sq += q;
                sqq += (double)q * q;
                sx += x[k];
                sqx += q * (double)x[k];
            }
            double det = n * sqq - sq * sq, a, c;
            if (det <= 0)
                continue;
            a = (n * sqx - sq * sx) / det;
            c = (sx - a * sq) / n;
            if (c > 0) { /* the offset must not be positive: fit a alone */
                c = 0;
                a = sqq > 0 ? sqx / sqq : 0;
            }
            if (a < 0)
                continue;
            double e = 0;
            for (int k = 0; k < n; k++) {
                double d = a * qs[k] + c - x[k];
                e += d * d;
            }
            if (e < best) {
                best = e;
                *a_out = (float)a;
                *b_out = (float)-c;
            }
        }
    return best;
}

void janas_q4k_quantize(const float *x, struct janas_block_q4k *y, size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++, x += JANAS_QK) {
        float a[8], mn[8], amax = 0.0f, bmax = 0.0f;
        for (int s = 0; s < 8; s++) {
            q4k_fit_sub(x + 32 * s, 32, &a[s], &mn[s]);
            if (a[s] > amax)
                amax = a[s];
            if (mn[s] > bmax)
                bmax = mn[s];
        }
        /* 6-bit scales and mins under the two block factors */
        float d = janas_fp16_to_fp32(janas_fp32_to_fp16(amax / 63.0f));
        float dmin = janas_fp16_to_fp32(janas_fp32_to_fp16(bmax / 63.0f));
        uint8_t sc[8], m[8], q[JANAS_QK];
        for (int s = 0; s < 8; s++) {
            int v = d > 0 ? (int)lrintf(a[s] / d) : 0;
            sc[s] = (uint8_t)(v < 0 ? 0 : (v > 63 ? 63 : v));
            v = dmin > 0 ? (int)lrintf(mn[s] / dmin) : 0;
            m[s] = (uint8_t)(v < 0 ? 0 : (v > 63 ? 63 : v));
        }
        /* alternately: the values under the factors, then the two factors
           by least squares over the whole block, x ~ d (sc q) - dmin m */
        for (int it = 0; it < 3; it++) {
            for (int s = 0; s < 8; s++) {
                float st = d * sc[s], off = dmin * m[s];
                for (int k = 0; k < 32; k++) {
                    int v =
                        st > 0 ? (int)lrintf((x[32 * s + k] + off) / st) : 0;
                    q[32 * s + k] = (uint8_t)(v < 0 ? 0 : (v > 15 ? 15 : v));
                }
            }
            if (it == 2)
                break;
            double suu = 0, suv = 0, svv = 0, sux = 0, svx = 0;
            for (int s = 0; s < 8; s++)
                for (int k = 0; k < 32; k++) {
                    double u = (double)sc[s] * q[32 * s + k], v = -(double)m[s];
                    double xv = x[32 * s + k];
                    suu += u * u;
                    suv += u * v;
                    svv += v * v;
                    sux += u * xv;
                    svx += v * xv;
                }
            double det = suu * svv - suv * suv;
            if (det > 0) {
                double nd = (sux * svv - svx * suv) / det;
                double nm = (svx * suu - sux * suv) / det;
                if (nd > 0 && nm >= 0) {
                    d = janas_fp16_to_fp32(janas_fp32_to_fp16((float)nd));
                    dmin = janas_fp16_to_fp32(janas_fp32_to_fp16((float)nm));
                }
            } else if (suu > 0 && svv == 0) {
                d = janas_fp16_to_fp32(janas_fp32_to_fp16((float)(sux / suu)));
            }
        }
        y[b].d = janas_fp32_to_fp16(d);
        y[b].dmin = janas_fp32_to_fp16(dmin);
        for (int i = 0; i < 4; i++) {
            y[b].scales[i] = (uint8_t)((sc[i] & 63) | ((sc[i + 4] >> 4) << 6));
            y[b].scales[4 + i] =
                (uint8_t)((m[i] & 63) | ((m[i + 4] >> 4) << 6));
            y[b].scales[8 + i] =
                (uint8_t)((sc[i + 4] & 0xf) | ((m[i + 4] & 0xf) << 4));
        }
        for (int s = 0; s < 8; s++)
            for (int k = 0; k < 32; k++) {
                uint8_t *qb = y[b].qs + (s / 2) * 32 + k;
                if (s & 1)
                    *qb = (uint8_t)((*qb & 0x0f) | (q[32 * s + k] << 4));
                else
                    *qb = q[32 * s + k];
            }
    }
}

float janas_q4k_dot_ref(const struct janas_block_q4k *w,
                        const struct janas_block_q8k *x, size_t nb)
{
    float acc = 0.0f, accm = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        uint8_t sc[8], m[8];
        unpack_scales(w[b].scales, sc, m);
        int32_t sumi = 0, summ = 0;
        for (int s = 0; s < 8; s++) {
            const uint8_t *q = w[b].qs + (s / 2) * 32;
            const int8_t *a = x[b].qs + s * 32;
            int shift = (s & 1) * 4;
            int32_t dot = 0;
            for (int l = 0; l < 32; l++)
                dot += ((q[l] >> shift) & 0xf) * a[l];
            sumi += sc[s] * dot;
            summ += m[s] * (x[b].bsums[2 * s] + x[b].bsums[2 * s + 1]);
        }
        /* the per-block integer-sum scheme (see quant_dotn.h): every
           implementation performs these float operations exactly */
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
        accm = fmaf(janas_fp16_to_fp32(w[b].dmin) * x[b].d, (float)summ, accm);
    }
    return acc - accm;
}

/* Value v (0..255) of a Q6_K block, unsigned 0..63. */
static inline int q6k_value(const struct janas_block_q6k *b, int v)
{
    int n = v / 128, part = (v % 128) / 32, l = v % 32;
    const uint8_t *ql = b->ql + 64 * n + (part & 1) * 32;
    const uint8_t *qh = b->qh + 32 * n;
    int low = part < 2 ? (ql[l] & 0xf) : (ql[l] >> 4);
    return low | (((qh[l] >> (2 * part)) & 3) << 4);
}

void janas_q6k_dequantize(const struct janas_block_q6k *x, float *y, size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++) {
        float d = janas_fp16_to_fp32(x[b].d);
        for (int v = 0; v < JANAS_QK; v++)
            *y++ = d * x[b].scales[v / 16] * (q6k_value(&x[b], v) - 32);
    }
}

float janas_q6k_dot_ref(const struct janas_block_q6k *w,
                        const struct janas_block_q8k *x, size_t nb)
{
    float acc = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        int32_t sumi = 0;
        for (int g = 0; g < JANAS_QK / 16; g++) {
            int32_t dot = 0;
            for (int v = 16 * g; v < 16 * g + 16; v++)
                dot += (q6k_value(&w[b], v) - 32) * x[b].qs[v];
            sumi += w[b].scales[g] * dot;
        }
        /* the per-block integer-sum scheme, as for Q4_K */
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
    }
    return acc;
}

/* Byte of weight v in a two-bit plane, and the shift of its bits. */
static inline int q6kp_index(int v, int *shift)
{
    int n = v / 128, part = (v % 128) / 32, l = v % 32;
    *shift = 2 * part;
    return 32 * n + l;
}

void janas_q6k_to_planes(const struct janas_block_q6k *src, size_t nb,
                         struct janas_block_q6kp *base, uint8_t *p1,
                         uint8_t *p0)
{
    for (size_t b = 0; b < nb; b++) {
        uint8_t *a1 = p1 + b * JANAS_Q6KP_PLANE;
        uint8_t *a0 = p0 + b * JANAS_Q6KP_PLANE;
        memset(base[b].q, 0, sizeof base[b].q);
        memset(a1, 0, JANAS_Q6KP_PLANE);
        memset(a0, 0, JANAS_Q6KP_PLANE);
        memcpy(base[b].scales, src[b].scales, sizeof src[b].scales);
        base[b].d = src[b].d;
        for (int v = 0; v < JANAS_QK; v++) {
            int sh, i = q6kp_index(v, &sh);
            int q = q6k_value(&src[b], v);
            base[b].q[i] |= (uint8_t)((q >> 4) << sh);
            a1[i] |= (uint8_t)(((q >> 2) & 3) << sh);
            a0[i] |= (uint8_t)((q & 3) << sh);
        }
    }
}

void janas_q6kp_to_q6k(const struct janas_block_q6kp *base, const uint8_t *p1,
                       const uint8_t *p0, size_t nb,
                       struct janas_block_q6k *dst)
{
    for (size_t b = 0; b < nb; b++) {
        const uint8_t *a1 = p1 + b * JANAS_Q6KP_PLANE;
        const uint8_t *a0 = p0 + b * JANAS_Q6KP_PLANE;
        memset(dst[b].ql, 0, sizeof dst[b].ql);
        memset(dst[b].qh, 0, sizeof dst[b].qh);
        memcpy(dst[b].scales, base[b].scales, sizeof base[b].scales);
        dst[b].d = base[b].d;
        for (int v = 0; v < JANAS_QK; v++) {
            int sh, i = q6kp_index(v, &sh);
            int q = (((base[b].q[i] >> sh) & 3) << 4) |
                    (((a1[i] >> sh) & 3) << 2) | ((a0[i] >> sh) & 3);
            int n = v / 128, part = (v % 128) / 32, l = v % 32;
            dst[b].qh[32 * n + l] |= (uint8_t)((q >> 4) << (2 * part));
            uint8_t *ql = dst[b].ql + 64 * n + (part & 1) * 32;
            ql[l] |= (uint8_t)(part < 2 ? (q & 0xf) : ((q & 0xf) << 4));
        }
    }
}

/* Twice the value weight v stands for, at the level the planes give. */
static inline int q6kp_value2(const struct janas_block_q6kp *b,
                              const uint8_t *a1, const uint8_t *a0, int v,
                              int level)
{
    int sh, i = q6kp_index(v, &sh);
    int q = (b->q[i] >> sh) & 3;
    if (level >= 2)
        q = (q << 2) | ((a1[i] >> sh) & 3);
    if (level >= 3)
        q = (q << 2) | ((a0[i] >> sh) & 3);
    /* the centre of the range the missing bits cover: half a step */
    return level == 3 ? 2 * q : level == 2 ? 8 * q + 3 : 32 * q + 15;
}

void janas_q6kp_dequantize(const struct janas_block_q6kp *base,
                           const uint8_t *p1, const uint8_t *p0, float *y,
                           size_t n)
{
    int level = p0 ? 3 : p1 ? 2 : 1;
    for (size_t b = 0; b < n / JANAS_QK; b++) {
        float d = janas_fp16_to_fp32(base[b].d) * 0.5f;
        const uint8_t *a1 = p1 ? p1 + b * JANAS_Q6KP_PLANE : NULL;
        const uint8_t *a0 = p0 ? p0 + b * JANAS_Q6KP_PLANE : NULL;
        for (int v = 0; v < JANAS_QK; v++)
            *y++ = d * base[b].scales[v / 16] *
                   (float)(q6kp_value2(&base[b], a1, a0, v, level) - 64);
    }
}

float janas_q6kp_dot_ref(const struct janas_block_q6kp *w, const uint8_t *p1,
                         const uint8_t *p0, const struct janas_block_q8k *x,
                         size_t nb)
{
    int level = p0 ? 3 : p1 ? 2 : 1;
    int shk = level == 3 ? 1 : level == 2 ? 3 : 5;
    int c = level == 3 ? 64 : level == 2 ? 61 : 49;
    float acc = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        const uint8_t *a1 = p1 ? p1 + b * JANAS_Q6KP_PLANE : NULL;
        const uint8_t *a0 = p0 ? p0 + b * JANAS_Q6KP_PLANE : NULL;
        int32_t sumi = 0, off = 0;
        for (int g = 0; g < JANAS_QK / 16; g++) {
            int32_t dot = 0;
            for (int v = 16 * g; v < 16 * g + 16; v++) {
                int sh, i = q6kp_index(v, &sh);
                int q = (w[b].q[i] >> sh) & 3;
                if (level >= 2)
                    q = (q << 2) | ((a1[i] >> sh) & 3);
                if (level >= 3)
                    q = (q << 2) | ((a0[i] >> sh) & 3);
                dot += q * x[b].qs[v];
            }
            sumi += w[b].scales[g] * dot;
            off += w[b].scales[g] * x[b].bsums[g];
        }
        /* the same scheme doubled: at level 3 the Q6_K value, bit for bit.
           A multiply, not a shift: sumi is signed and often negative, and
           shifting that left is undefined. It cannot overflow - at level 3,
           the widest case, |sumi| <= 16 x 128 x (16 x 63 x 127) and twice
           that is a quarter of what an int32 holds. */
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * 0.5f * x[b].d,
                   (float)(sumi * (1 << shk) - c * off), acc);
    }
    return acc;
}

/* Weight v (0..255) of a Q5_K block, 0..31. */
static inline int q5k_value(const struct janas_block_q5k *b, int v)
{
    int s = v / 32, l = v % 32;
    int low = (b->qs[(s / 2) * 32 + l] >> ((s & 1) * 4)) & 0xf;
    return low | (((b->qh[l] >> s) & 1) << 4);
}

void janas_q5k_dequantize(const struct janas_block_q5k *x, float *y, size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++) {
        uint8_t sc[8], m[8];
        unpack_scales(x[b].scales, sc, m);
        float d = janas_fp16_to_fp32(x[b].d),
              dmin = janas_fp16_to_fp32(x[b].dmin);
        for (int v = 0; v < JANAS_QK; v++)
            *y++ = d * sc[v / 32] * q5k_value(&x[b], v) - dmin * m[v / 32];
    }
}

float janas_q5k_dot_ref(const struct janas_block_q5k *w,
                        const struct janas_block_q8k *x, size_t nb)
{
    float acc = 0.0f, accm = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        uint8_t sc[8], m[8];
        unpack_scales(w[b].scales, sc, m);
        int32_t sumi = 0, summ = 0;
        for (int s = 0; s < 8; s++) {
            int32_t dot = 0;
            for (int l = 0; l < 32; l++)
                dot += q5k_value(&w[b], 32 * s + l) * x[b].qs[32 * s + l];
            sumi += sc[s] * dot;
            summ += m[s] * (x[b].bsums[2 * s] + x[b].bsums[2 * s + 1]);
        }
        /* the per-block integer-sum scheme, as for Q4_K */
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
        accm = fmaf(janas_fp16_to_fp32(w[b].dmin) * x[b].d, (float)summ, accm);
    }
    return acc - accm;
}

void janas_q8_0_dequantize(const struct janas_block_q8_0 *x, float *y, size_t n)
{
    for (size_t b = 0; b < n / 32; b++) {
        float d = janas_fp16_to_fp32(x[b].d);
        for (int l = 0; l < 32; l++)
            *y++ = d * x[b].qs[l];
    }
}

float janas_q8_0_dot_ref(const struct janas_block_q8_0 *w,
                         const struct janas_block_q8k *x, size_t nb)
{
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++)
        for (int k = 0; k < 8; k++) {
            const struct janas_block_q8_0 *wk = w + 8 * b + k;
            int32_t dot = 0;
            for (int l = 0; l < 32; l++)
                dot += wk->qs[l] * x[b].qs[32 * k + l];
            acc[k] =
                fmaf(janas_fp16_to_fp32(wk->d) * x[b].d, (float)dot, acc[k]);
        }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) +
           ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

const int8_t janas_iq4_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                     1,    13,   25,  38,  53,  69,  89,  113};

void janas_iq4nl_dequantize(const struct janas_block_iq4nl *x, float *y,
                            size_t n)
{
    for (size_t b = 0; b < n / 32; b++, y += 32) {
        float d = janas_fp16_to_fp32(x[b].d);
        for (int l = 0; l < 16; l++) {
            y[l] = d * janas_iq4_values[x[b].qs[l] & 15];
            y[l + 16] = d * janas_iq4_values[x[b].qs[l] >> 4];
        }
    }
}

const int8_t janas_q4_0_values[16] = {-8, -7, -6, -5, -4, -3, -2, -1,
                                      0,  1,  2,  3,  4,  5,  6,  7};

/* The exact sum of 32 4-bit weights (16 bytes of indices into levels)
   times a. */
static int32_t nib_sum32(const uint8_t *q, const int8_t *a,
                         const int8_t *levels)
{
    int32_t s = 0;
    for (int l = 0; l < 16; l++)
        s += levels[q[l] & 15] * a[l] + levels[q[l] >> 4] * a[l + 16];
    return s;
}

static int32_t iq4_sum32(const uint8_t *q, const int8_t *a)
{
    return nib_sum32(q, a, janas_iq4_values);
}

/* IQ4_NL and Q4_0: one lane per block position, as Q8_0. */
static float nib_dot_ref(const struct janas_block_iq4nl *w,
                         const struct janas_block_q8k *x, size_t nb,
                         const int8_t *levels)
{
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++)
        for (int k = 0; k < 8; k++) {
            const struct janas_block_iq4nl *wk = w + 8 * b + k;
            int32_t dot = nib_sum32(wk->qs, x[b].qs + 32 * k, levels);
            acc[k] =
                fmaf(janas_fp16_to_fp32(wk->d) * x[b].d, (float)dot, acc[k]);
        }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) +
           ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

float janas_iq4nl_dot_ref(const struct janas_block_iq4nl *w,
                          const struct janas_block_q8k *x, size_t nb)
{
    return nib_dot_ref(w, x, nb, janas_iq4_values);
}

_Static_assert(sizeof(struct janas_block_q4_0) ==
                       sizeof(struct janas_block_iq4nl) &&
                   offsetof(struct janas_block_q4_0, qs) ==
                       offsetof(struct janas_block_iq4nl, qs),
               "Q4_0 is laid out as IQ4_NL");

void janas_q4_0_dequantize(const struct janas_block_q4_0 *x, float *y, size_t n)
{
    for (size_t b = 0; b < n / 32; b++, y += 32) {
        float d = janas_fp16_to_fp32(x[b].d);
        for (int l = 0; l < 16; l++) {
            y[l] = d * janas_q4_0_values[x[b].qs[l] & 15];
            y[l + 16] = d * janas_q4_0_values[x[b].qs[l] >> 4];
        }
    }
}

float janas_q4_0_dot_ref(const struct janas_block_q4_0 *w,
                         const struct janas_block_q8k *x, size_t nb)
{
    return nib_dot_ref((const struct janas_block_iq4nl *)w, x, nb,
                       janas_q4_0_values);
}

/*
 * Q4_1 and Q5_1, weights d * q + m: a block is d, m (fp16), for Q5_1 the
 * fifth bits (qh, 4 bytes), then the 16 bytes of low nibbles. The code
 * reads both through the byte layout, with the block's size and whether
 * it has qh.
 */
static inline uint16_t dm_u16(const uint8_t *blk, int at)
{
    uint16_t v;
    memcpy(&v, blk + at, 2);
    return v;
}

/* The value of weight l of a Q4_1 (five = 0) or Q5_1 block. */
static inline int dm_value(const uint8_t *blk, int five, int l)
{
    const uint8_t *qs = blk + (five ? 8 : 4);
    int lo = l < 16 ? qs[l] & 15 : qs[l - 16] >> 4;
    if (!five)
        return lo;
    uint32_t qh;
    memcpy(&qh, blk + 4, 4);
    return lo | (int)((qh >> l) & 1) << 4;
}

static void dm_dequantize(const uint8_t *w, size_t size, int five, float *y,
                          size_t n)
{
    for (size_t b = 0; b < n / 32; b++, y += 32, w += size) {
        float d = janas_fp16_to_fp32(dm_u16(w, 0)),
              m = janas_fp16_to_fp32(dm_u16(w, 2));
        for (int l = 0; l < 32; l++)
            y[l] = d * (float)dm_value(w, five, l) + m;
    }
}

/* Per block position a lane for d x the exact weight sum and one for m x
   the activations' sum (two Q8_K block sums); each set of eight added as
   janas_hsum8 does, then the two. */
static float dm_dot_ref(const uint8_t *w, size_t size, int five,
                        const struct janas_block_q8k *x, size_t nb)
{
    float acc[8] = {0}, accm[8] = {0};
    for (size_t b = 0; b < nb; b++)
        for (int k = 0; k < 8; k++) {
            const uint8_t *wk = w + (8 * b + k) * size;
            int32_t dot = 0;
            for (int l = 0; l < 32; l++)
                dot += dm_value(wk, five, l) * x[b].qs[32 * k + l];
            int32_t asum = x[b].bsums[2 * k] + x[b].bsums[2 * k + 1];
            acc[k] = fmaf(janas_fp16_to_fp32(dm_u16(wk, 0)) * x[b].d,
                          (float)dot, acc[k]);
            accm[k] = fmaf(janas_fp16_to_fp32(dm_u16(wk, 2)) * x[b].d,
                           (float)asum, accm[k]);
        }
    return (((acc[0] + acc[4]) + (acc[2] + acc[6])) +
            ((acc[1] + acc[5]) + (acc[3] + acc[7]))) +
           (((accm[0] + accm[4]) + (accm[2] + accm[6])) +
            ((accm[1] + accm[5]) + (accm[3] + accm[7])));
}

void janas_q5_1_dequantize(const struct janas_block_q5_1 *x, float *y, size_t n)
{
    dm_dequantize((const uint8_t *)x, sizeof(*x), 1, y, n);
}

float janas_q5_1_dot_ref(const struct janas_block_q5_1 *w,
                         const struct janas_block_q8k *x, size_t nb)
{
    return dm_dot_ref((const uint8_t *)w, sizeof(*w), 1, x, nb);
}

void janas_q4_1_dequantize(const struct janas_block_q4_1 *x, float *y, size_t n)
{
    dm_dequantize((const uint8_t *)x, sizeof(*x), 0, y, n);
}

float janas_q4_1_dot_ref(const struct janas_block_q4_1 *w,
                         const struct janas_block_q8k *x, size_t nb)
{
    return dm_dot_ref((const uint8_t *)w, sizeof(*w), 0, x, nb);
}

/* The sixteen 6-bit scales of a Q3_K block, less their offset of 32. */
static inline void q3k_scales(const uint8_t *q, int8_t sc[16])
{
    const uint32_t low4 = 0x0f0f0f0f, top2 = 0x03030303;
    uint32_t a[3], o[4];
    memcpy(a, q, 12);
    o[0] = (a[0] & low4) | ((a[2] & top2) << 4);
    o[1] = (a[1] & low4) | (((a[2] >> 2) & top2) << 4);
    o[2] = ((a[0] >> 4) & low4) | (((a[2] >> 4) & top2) << 4);
    o[3] = ((a[1] >> 4) & low4) | (((a[2] >> 6) & top2) << 4);
    memcpy(sc, o, 16);
    for (int i = 0; i < 16; i++)
        sc[i] = (int8_t)(sc[i] - 32);
}

/* The three-bit value of weight e of a Q3_K block, -4 to 3. */
static inline int q3k_value(const struct janas_block_q3k *w, int e)
{
    int n = e / 128, j = e % 128 / 32, h = e % 32 / 16, l = e % 16;
    int q = w->qs[32 * n + 16 * h + l] >> 2 * j & 3;
    int hi = w->hmask[16 * h + l] >> (j + 4 * n) & 1;
    return q - (hi ? 0 : 4);
}

void janas_q3k_dequantize(const struct janas_block_q3k *x, float *y, size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++, y += JANAS_QK) {
        int8_t sc[16];
        q3k_scales(x[b].scales, sc);
        float d = janas_fp16_to_fp32(x[b].d);
        for (int e = 0; e < JANAS_QK; e++)
            y[e] = d * (float)sc[e / 16] * (float)q3k_value(&x[b], e);
    }
}

float janas_q3k_dot_ref(const struct janas_block_q3k *w,
                        const struct janas_block_q8k *x, size_t nb)
{
    float acc = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        int8_t sc[16];
        q3k_scales(w[b].scales, sc);
        int32_t sumi = 0;
        for (int g = 0; g < 16; g++) {
            int32_t s = 0;
            for (int e = 16 * g; e < 16 * g + 16; e++)
                s += q3k_value(&w[b], e) * x[b].qs[e];
            sumi += sc[g] * s;
        }
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
    }
    return acc;
}

/* The level of weight e (0-31) of group g of an IQ3_S block, signed. */
static inline int iq3s_value(const struct janas_block_iq3s *w, int g, int e)
{
    int k = e / 4;
    int idx = w->qs[8 * g + k] | ((w->qh[g] >> k) & 1) << 8;
    int v = (int)(iq3s_grid[idx] >> 8 * (e % 4) & 0xff);
    return w->signs[4 * g + e / 8] >> (e % 8) & 1 ? -v : v;
}

static inline int iq3s_scale(const struct janas_block_iq3s *w, int g)
{
    return 1 + 2 * (w->scales[g / 2] >> 4 * (g % 2) & 15);
}

void janas_iq3s_dequantize(const struct janas_block_iq3s *x, float *y, size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++) {
        float d = janas_fp16_to_fp32(x[b].d);
        for (int g = 0; g < 8; g++, y += 32) {
            float db = d * (float)iq3s_scale(&x[b], g);
            for (int e = 0; e < 32; e++)
                y[e] = db * (float)iq3s_value(&x[b], g, e);
        }
    }
}

float janas_iq3s_dot_ref(const struct janas_block_iq3s *w,
                         const struct janas_block_q8k *x, size_t nb)
{
    float acc = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        int32_t sumi = 0;
        for (int g = 0; g < 8; g++) {
            int32_t s = 0;
            for (int e = 0; e < 32; e++)
                s += iq3s_value(&w[b], g, e) * x[b].qs[32 * g + e];
            sumi += iq3s_scale(&w[b], g) * s;
        }
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
    }
    return acc;
}

/* The 6-bit scale of group g of an IQ4_XS block, less its offset of 32. */
static inline int iq4xs_scale(const struct janas_block_iq4xs *w, int g)
{
    return (((w->scales_l[g / 2] >> 4 * (g % 2)) & 15) |
            (((w->scales_h >> 2 * g) & 3) << 4)) -
           32;
}

void janas_iq4xs_dequantize(const struct janas_block_iq4xs *x, float *y,
                            size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++) {
        float d = janas_fp16_to_fp32(x[b].d);
        for (int g = 0; g < 8; g++, y += 32) {
            float dl = d * (float)iq4xs_scale(&x[b], g);
            const uint8_t *q = x[b].qs + 16 * g;
            for (int l = 0; l < 16; l++) {
                y[l] = dl * janas_iq4_values[q[l] & 15];
                y[l + 16] = dl * janas_iq4_values[q[l] >> 4];
            }
        }
    }
}

float janas_iq4xs_dot_ref(const struct janas_block_iq4xs *w,
                          const struct janas_block_q8k *x, size_t nb)
{
    float acc = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        int32_t sumi = 0; /* at most 8 x 31 x 32 x 127 x 127: fits */
        for (int g = 0; g < 8; g++)
            sumi += iq4xs_scale(&w[b], g) *
                    iq4_sum32(w[b].qs + 16 * g, x[b].qs + 32 * g);
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
    }
    return acc;
}

#if defined(__x86_64__)
/*
 * Unpacks the 12 scale bytes into 8 scales (bytes 0-7) and 8 mins (bytes
 * 8-15) with four 32-bit operations instead of sixteen byte extractions.
 */
static inline void unpack_scales_u32(const uint8_t *q, uint32_t out[4])
{
    const uint32_t low6 = 0x3f3f3f3f, low4 = 0x0f0f0f0f, top2 = 0x03030303;
    uint32_t a, b, c;
    memcpy(&a, q, 4);
    memcpy(&b, q + 4, 4);
    memcpy(&c, q + 8, 4);
    out[0] = a & low6;                                     /* scales 0-3 */
    out[1] = (c & low4) | (((a >> 6) & top2) << 4);        /* scales 4-7 */
    out[2] = b & low6;                                     /* mins 0-3 */
    out[3] = ((c >> 4) & low4) | (((b >> 6) & top2) << 4); /* mins 4-7 */
}

/* The kernels proper: quant_dotn.h, once for AVX2 and once for AVX-VNNI. */
#define DOTN(name) name##_avx2
#define DOTN_TARGET __attribute__((target("avx2,fma,f16c")))
#define DOTN_ACC(acc, p, s) _mm256_add_epi32(acc, _mm256_madd_epi16(p, s))
#include "quant_dotn.h"
#undef DOTN
#undef DOTN_TARGET
#undef DOTN_ACC

#define DOTN(name) name##_vnni
#define DOTN_TARGET __attribute__((target("avx2,fma,f16c,avxvnni")))
#define DOTN_ACC(acc, p, s) _mm256_dpwssd_avx_epi32(acc, p, s)
#include "quant_dotn.h"
#undef DOTN
#undef DOTN_TARGET
#undef DOTN_ACC
#endif

typedef float (*dot_fn)(const struct janas_block_q4k *,
                        const struct janas_block_q8k *, size_t);
typedef float (*dot6_fn)(const struct janas_block_q6k *,
                         const struct janas_block_q8k *, size_t);

static dot_fn dot_impl = janas_q4k_dot_ref;
static dot6_fn dot6_impl = janas_q6k_dot_ref;
typedef float (*dot5_fn)(const struct janas_block_q5k *,
                         const struct janas_block_q8k *, size_t);
typedef float (*dot80_fn)(const struct janas_block_q8_0 *,
                          const struct janas_block_q8k *, size_t);
static dot5_fn dot5_impl = janas_q5k_dot_ref;
static dot80_fn dot80_impl = janas_q8_0_dot_ref;
typedef float (*dotnl_fn)(const struct janas_block_iq4nl *,
                          const struct janas_block_q8k *, size_t);
typedef float (*dotxs_fn)(const struct janas_block_iq4xs *,
                          const struct janas_block_q8k *, size_t);
static dotnl_fn dotnl_impl = janas_iq4nl_dot_ref;
static dotxs_fn dotxs_impl = janas_iq4xs_dot_ref;
typedef float (*dot40_fn)(const struct janas_block_q4_0 *,
                          const struct janas_block_q8k *, size_t);
typedef float (*dot51_fn)(const struct janas_block_q5_1 *,
                          const struct janas_block_q8k *, size_t);
static dot40_fn dot40_impl = janas_q4_0_dot_ref;
static dot51_fn dot51_impl = janas_q5_1_dot_ref;
typedef float (*dot41_fn)(const struct janas_block_q4_1 *,
                          const struct janas_block_q8k *, size_t);
static dot41_fn dot41_impl = janas_q4_1_dot_ref;
typedef float (*dot3_fn)(const struct janas_block_q3k *,
                         const struct janas_block_q8k *, size_t);
static dot3_fn dot3_impl = janas_q3k_dot_ref;
typedef float (*dotis_fn)(const struct janas_block_iq3s *,
                          const struct janas_block_q8k *, size_t);
static dotis_fn dotis_impl = janas_iq3s_dot_ref;
typedef void (*dotn6p_fn)(const struct janas_block_q6kp *, const uint8_t *,
                          const uint8_t *, const struct janas_block_q8k *,
                          size_t, int, size_t, float *);
static dotn6p_fn dot6p_impl; /* NULL: the scalar reference */
static const char *dot_name = "ref";
static int use_vnni;

/* Resolved before main: the kernels are first called from pool threads. */
__attribute__((constructor)) static void dot_kernels_init(void)
{
#if defined(__x86_64__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        __builtin_cpu_supports("f16c")) {
        dot_impl = q4k_dot_avx2;
        dot6_impl = q6k_dot_avx2;
        dot5_impl = q5k_dot_avx2;
        dot80_impl = q8_0_dot_avx2;
        dotnl_impl = iq4nl_dot_avx2;
        dotxs_impl = iq4xs_dot_avx2;
        dot40_impl = q4_0_dot_avx2;
        dot51_impl = q5_1_dot_avx2;
        dot41_impl = q4_1_dot_avx2;
        dot3_impl = q3k_dot_avx2;
        dotis_impl = iq3s_dot_avx2;
        dot6p_impl = q6kp_dotn_avx2;
        dot_name = "avx2";
        const char *k = getenv("JANAS_KERNELS"); /* avx2: no VNNI */
        if (__builtin_cpu_supports("avxvnni") && !(k && !strcmp(k, "avx2"))) {
            use_vnni = 1;
            dot_impl = q4k_dot_vnni;
            dot6_impl = q6k_dot_vnni;
            dot5_impl = q5k_dot_vnni;
            dot80_impl = q8_0_dot_vnni;
            dotnl_impl = iq4nl_dot_vnni;
            dotxs_impl = iq4xs_dot_vnni;
            dot40_impl = q4_0_dot_vnni;
            dot51_impl = q5_1_dot_vnni;
            dot41_impl = q4_1_dot_vnni;
            dot3_impl = q3k_dot_vnni;
            dotis_impl = iq3s_dot_vnni;
            dot6p_impl = q6kp_dotn_vnni;
            dot_name = "avx-vnni";
        }
    }
#endif
}

int janas_use_vnni(void)
{
    return use_vnni;
}

float janas_q4k_dot(const struct janas_block_q4k *w,
                    const struct janas_block_q8k *x, size_t nb)
{
    return dot_impl(w, x, nb);
}

float janas_q5k_dot(const struct janas_block_q5k *w,
                    const struct janas_block_q8k *x, size_t nb)
{
    return dot5_impl(w, x, nb);
}

float janas_q8_0_dot(const struct janas_block_q8_0 *w,
                     const struct janas_block_q8k *x, size_t nb)
{
    return dot80_impl(w, x, nb);
}

float janas_iq4nl_dot(const struct janas_block_iq4nl *w,
                      const struct janas_block_q8k *x, size_t nb)
{
    return dotnl_impl(w, x, nb);
}

float janas_iq4xs_dot(const struct janas_block_iq4xs *w,
                      const struct janas_block_q8k *x, size_t nb)
{
    return dotxs_impl(w, x, nb);
}

float janas_q4_0_dot(const struct janas_block_q4_0 *w,
                     const struct janas_block_q8k *x, size_t nb)
{
    return dot40_impl(w, x, nb);
}

float janas_q5_1_dot(const struct janas_block_q5_1 *w,
                     const struct janas_block_q8k *x, size_t nb)
{
    return dot51_impl(w, x, nb);
}

float janas_q4_1_dot(const struct janas_block_q4_1 *w,
                     const struct janas_block_q8k *x, size_t nb)
{
    return dot41_impl(w, x, nb);
}

float janas_q3k_dot(const struct janas_block_q3k *w,
                    const struct janas_block_q8k *x, size_t nb)
{
    return dot3_impl(w, x, nb);
}

float janas_iq3s_dot(const struct janas_block_iq3s *w,
                     const struct janas_block_q8k *x, size_t nb)
{
    return dotis_impl(w, x, nb);
}

const char *janas_q4k_kernel_name(void)
{
    return dot_name;
}

float janas_q6k_dot(const struct janas_block_q6k *w,
                    const struct janas_block_q8k *x, size_t nb)
{
    return dot6_impl(w, x, nb);
}

float janas_q6kp_dot(const struct janas_block_q6kp *w, const uint8_t *p1,
                     const uint8_t *p0, const struct janas_block_q8k *x,
                     size_t nb)
{
    if (dot6p_impl) {
        float r;
        dot6p_impl(w, p1, p0, x, 0, 1, nb, &r);
        return r;
    }
    return janas_q6kp_dot_ref(w, p1, p0, x, nb);
}

void janas_q6kp_dot_multi(const struct janas_block_q6kp *w, const uint8_t *p1,
                          const uint8_t *p0, const struct janas_block_q8k *x,
                          size_t x_stride, size_t nv, size_t nb, float *out,
                          size_t out_stride)
{
    size_t v = 0;
    if (dot6p_impl) {
        float r[8];
        int g = 1;
        for (; v < nv; v += (size_t)g) {
            g = nv - v >= 8 ? 8 : (int)(nv - v);
            dot6p_impl(w, p1, p0, x + v * x_stride, x_stride, g, nb, r);
            for (int i = 0; i < g; i++)
                out[(v + (size_t)i) * out_stride] = r[i];
        }
        return;
    }
    for (; v < nv; v++)
        out[v * out_stride] =
            janas_q6kp_dot_ref(w, p1, p0, x + v * x_stride, nb);
}

size_t janas_qtype_block_size(int type)
{
    switch (type) {
    case JANAS_Q6_K_P:
        return sizeof(struct janas_block_q6kp);
    case JANAS_Q4_K:
        return sizeof(struct janas_block_q4k);
    case JANAS_Q6_K:
        return sizeof(struct janas_block_q6k);
    case JANAS_Q5_K:
        return sizeof(struct janas_block_q5k);
    case JANAS_Q8_0:
        return 8 * sizeof(struct janas_block_q8_0);
    case JANAS_IQ4_NL:
        return 8 * sizeof(struct janas_block_iq4nl);
    case JANAS_Q4_0:
        return 8 * sizeof(struct janas_block_q4_0);
    case JANAS_Q5_1:
        return 8 * sizeof(struct janas_block_q5_1);
    case JANAS_Q4_1:
        return 8 * sizeof(struct janas_block_q4_1);
    case JANAS_Q3_K:
        return sizeof(struct janas_block_q3k);
    case JANAS_IQ3_S:
        return sizeof(struct janas_block_iq3s);
    case JANAS_IQ4_XS:
        return sizeof(struct janas_block_iq4xs);
    default:
        return 0;
    }
}

void janas_dequantize(int type, const void *w, float *y, size_t n)
{
    switch (type) {
    case JANAS_Q4_K:
        janas_q4k_dequantize(w, y, n);
        return;
    case JANAS_Q6_K:
        janas_q6k_dequantize(w, y, n);
        return;
    case JANAS_Q5_K:
        janas_q5k_dequantize(w, y, n);
        return;
    case JANAS_Q8_0:
        janas_q8_0_dequantize(w, y, n);
        return;
    case JANAS_IQ4_NL:
        janas_iq4nl_dequantize(w, y, n);
        return;
    case JANAS_IQ4_XS:
        janas_iq4xs_dequantize(w, y, n);
        return;
    case JANAS_Q4_0:
        janas_q4_0_dequantize(w, y, n);
        return;
    case JANAS_Q5_1:
        janas_q5_1_dequantize(w, y, n);
        return;
    case JANAS_Q4_1:
        janas_q4_1_dequantize(w, y, n);
        return;
    case JANAS_Q3_K:
        janas_q3k_dequantize(w, y, n);
        return;
    case JANAS_IQ3_S:
        janas_iq3s_dequantize(w, y, n);
        return;
    default:
        /* callers check janas_qtype_block_size first: this is a bug */
        abort();
    }
}

float janas_dot(int type, const void *w, const struct janas_block_q8k *x,
                size_t nb)
{
    switch (type) {
    case JANAS_Q4_K:
        return janas_q4k_dot(w, x, nb);
    case JANAS_Q6_K:
        return janas_q6k_dot(w, x, nb);
    case JANAS_Q5_K:
        return janas_q5k_dot(w, x, nb);
    case JANAS_Q8_0:
        return janas_q8_0_dot(w, x, nb);
    case JANAS_IQ4_NL:
        return janas_iq4nl_dot(w, x, nb);
    case JANAS_IQ4_XS:
        return janas_iq4xs_dot(w, x, nb);
    case JANAS_Q4_0:
        return janas_q4_0_dot(w, x, nb);
    case JANAS_Q5_1:
        return janas_q5_1_dot(w, x, nb);
    case JANAS_Q4_1:
        return janas_q4_1_dot(w, x, nb);
    case JANAS_Q3_K:
        return janas_q3k_dot(w, x, nb);
    case JANAS_IQ3_S:
        return janas_iq3s_dot(w, x, nb);
    case JANAS_Q6_K_P: /* the base plane alone: two bits per weight */
        return janas_q6kp_dot(w, NULL, NULL, x, nb);
    default:
        /* callers check janas_qtype_block_size first: this is a bug */
        abort();
    }
}

static float dot_f32_ref(const float *a, const float *b, size_t n)
{
    float s = 0.0f;
    for (size_t i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

static void axpy_f32_ref(float *y, float a, const float *x, size_t n)
{
    for (size_t i = 0; i < n; i++)
        y[i] += a * x[i];
}

#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) static float
dot_f32_avx2(const float *a, const float *b, size_t n)
{
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 =
            _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                             _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8)
        s0 =
            _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s0 = _mm256_add_ps(s0, s1);
    __m128 s =
        _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    float r = _mm_cvtss_f32(s);
    for (; i < n; i++)
        r += a[i] * b[i];
    return r;
}

__attribute__((target("avx2,fma"))) static void
axpy_f32_avx2(float *y, float a, const float *x, size_t n)
{
    __m256 va = _mm256_set1_ps(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i),
                                                _mm256_loadu_ps(y + i)));
    for (; i < n; i++)
        y[i] += a * x[i];
}
#endif

/* Resolved before main, so the float helpers never race on it. */
__attribute__((constructor)) static void float_helpers_init(void)
{
#if defined(__x86_64__)
    __builtin_cpu_init();
    use_avx2_fma =
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}

float janas_dot_f32(const float *a, const float *b, size_t n)
{
#if defined(__x86_64__)
    if (use_avx2_fma)
        return dot_f32_avx2(a, b, n);
#endif
    return dot_f32_ref(a, b, n);
}

void janas_axpy_f32(float *y, float a, const float *x, size_t n)
{
#if defined(__x86_64__)
    if (use_avx2_fma) {
        axpy_f32_avx2(y, a, x, n);
        return;
    }
#endif
    axpy_f32_ref(y, a, x, n);
}

void janas_dot_multi(int type, const void *w, const struct janas_block_q8k *x,
                     size_t x_stride, size_t nv, size_t nb, float *out,
                     size_t out_stride)
{
    size_t v = 0;
#if defined(__x86_64__)
    if (use_avx2_fma && __builtin_cpu_supports("f16c") &&
        (type == JANAS_Q4_K || type == JANAS_Q6_K || type == JANAS_Q5_K ||
         type == JANAS_Q8_0 || type == JANAS_IQ4_NL || type == JANAS_IQ4_XS ||
         type == JANAS_Q4_0 || type == JANAS_Q5_1 || type == JANAS_Q4_1 ||
         type == JANAS_Q3_K || type == JANAS_IQ3_S)) {
        float r[8];
        int g = 1;
        for (; v < nv; v += (size_t)g) {
            /* groups of 8, then the rest in one group */
            g = nv - v >= 8 ? 8 : (int)(nv - v);
            const struct janas_block_q8k *xv = x + v * x_stride;
            switch (type) {
            case JANAS_Q4_K:
                (use_vnni ? q4k_dotn_vnni : q4k_dotn_avx2)(w, xv, x_stride, g,
                                                           nb, r);
                break;
            case JANAS_Q6_K:
                (use_vnni ? q6k_dotn_vnni : q6k_dotn_avx2)(w, xv, x_stride, g,
                                                           nb, r);
                break;
            case JANAS_Q5_K:
                (use_vnni ? q5k_dotn_vnni : q5k_dotn_avx2)(w, xv, x_stride, g,
                                                           nb, r);
                break;
            case JANAS_IQ4_NL:
                (use_vnni ? iq4nl_dotn_vnni : iq4nl_dotn_avx2)(w, xv, x_stride,
                                                               g, nb, r);
                break;
            case JANAS_IQ4_XS:
                (use_vnni ? iq4xs_dotn_vnni : iq4xs_dotn_avx2)(w, xv, x_stride,
                                                               g, nb, r);
                break;
            case JANAS_Q4_0:
                (use_vnni ? q4_0_dotn_vnni : q4_0_dotn_avx2)(w, xv, x_stride, g,
                                                             nb, r);
                break;
            case JANAS_Q5_1:
                (use_vnni ? q5_1_dotn_vnni : q5_1_dotn_avx2)(w, xv, x_stride, g,
                                                             nb, r);
                break;
            case JANAS_Q4_1:
                (use_vnni ? q4_1_dotn_vnni : q4_1_dotn_avx2)(w, xv, x_stride, g,
                                                             nb, r);
                break;
            case JANAS_Q3_K:
                (use_vnni ? q3k_dotn_vnni : q3k_dotn_avx2)(w, xv, x_stride, g,
                                                           nb, r);
                break;
            case JANAS_IQ3_S:
                (use_vnni ? iq3s_dotn_vnni : iq3s_dotn_avx2)(w, xv, x_stride, g,
                                                             nb, r);
                break;
            default:
                (use_vnni ? q8_0_dotn_vnni : q8_0_dotn_avx2)(w, xv, x_stride, g,
                                                             nb, r);
            }
            for (int i = 0; i < g; i++)
                out[(v + (size_t)i) * out_stride] = r[i];
        }
        return;
    }
#endif
    for (; v < nv; v++)
        out[v * out_stride] = janas_dot(type, w, x + v * x_stride, nb);
}
