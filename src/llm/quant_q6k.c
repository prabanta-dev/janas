/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * quant_q6k.c - Q6_K made from floats, and a row of Q6_K in bit planes added
 * to a vector at the depth asked: what Janas's down matrix by neuron
 * (JANAS_Q6_K_PN, tools/jns_neurons.c) is made of and read with.
 */
#include "quant.h"
#include "quant_q6kp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

/* The integer of x at scale a, within Q6_K's -32 .. 31. */
static int q6(float x, float a)
{
    if (a == 0.0f)
        return 0;
    int q = (int)lrintf(x / a);
    return q < -32 ? -32 : q > 31 ? 31 : q;
}

/* The scale of a group of 16, among a few near its largest magnitude, of
   least squared error. */
static float group_scale(const float *x)
{
    float amax = 0.0f, ext = 0.0f;
    for (int i = 0; i < 16; i++)
        if (fabsf(x[i]) > amax) {
            amax = fabsf(x[i]);
            ext = x[i];
        }
    if (amax == 0.0f)
        return 0.0f;
    /* the largest one at the end of the range on its side: 31 up, 32 down */
    float a0 = ext > 0 ? amax / 31.0f : amax / 32.0f, best = a0;
    double best_err = -1;
    for (int k = -6; k <= 6; k++) {
        float a = a0 * (1.0f + 0.02f * (float)k);
        double err = 0;
        for (int i = 0; i < 16; i++) {
            double e = x[i] - a * (float)q6(x[i], a);
            err += e * e;
        }
        if (best_err < 0 || err < best_err) {
            best_err = err;
            best = a;
        }
    }
    return best;
}

void janas_q6k_quantize(const float *x, struct janas_block_q6k *y, size_t n)
{
    for (size_t b = 0; b < n / JANAS_QK; b++, x += JANAS_QK) {
        float a[16], amax = 0.0f;
        for (int s = 0; s < 16; s++) {
            a[s] = group_scale(x + 16 * s);
            if (a[s] > amax)
                amax = a[s];
        }
        uint16_t d16 = janas_fp32_to_fp16(amax / 127.0f);
        float d = janas_fp16_to_fp32(d16);
        int8_t sc[16];
        int q[JANAS_QK];
        for (int s = 0; s < 16; s++) {
            int v = d > 0 ? (int)lrintf(a[s] / d) : 0;
            sc[s] = (int8_t)(v > 127 ? 127 : v < 0 ? 0 : v);
        }
        /* the values under the factors, then the block's factor by least
           squares, x ~ d (sc q), twice */
        for (int it = 0; it < 3; it++) {
            for (int v = 0; v < JANAS_QK; v++)
                q[v] = q6(x[v], d * sc[v / 16]);
            if (it == 2)
                break;
            double sxu = 0, suu = 0;
            for (int v = 0; v < JANAS_QK; v++) {
                double u = (double)sc[v / 16] * q[v];
                sxu += x[v] * u;
                suu += u * u;
            }
            if (suu > 0 && sxu > 0) {
                d16 = janas_fp32_to_fp16((float)(sxu / suu));
                d = janas_fp16_to_fp32(d16);
            }
        }
        memset(y[b].ql, 0, sizeof(y[b].ql));
        memset(y[b].qh, 0, sizeof(y[b].qh));
        memcpy(y[b].scales, sc, sizeof(sc));
        y[b].d = d16;
        /* the bits where q6k_value (quant.c) reads them */
        for (int v = 0; v < JANAS_QK; v++) {
            int u = q[v] + 32, half = v / 128, part = (v % 128) / 32,
                l = v % 32;
            uint8_t *ql = y[b].ql + 64 * half + (part & 1) * 32 + l;
            *ql |= (uint8_t)((u & 0xf) << (part < 2 ? 0 : 4));
            y[b].qh[32 * half + l] |= (uint8_t)((u >> 4) << (2 * part));
        }
    }
}

/* One block of the row: y += hd times each weight's value, on the scale
   whose 64 is zero, times its group's factor - fused, as the vector one. */
static void axpy_block_ref(const struct janas_block_q6kp *b, const uint8_t *a1,
                           const uint8_t *a0, int level, float hd, float *y)
{
    for (int v = 0; v < JANAS_QK; v++)
        y[v] = fmaf(hd * (float)b->scales[v / 16],
                    (float)(q6kp_value2(b, a1, a0, v, level) - 64), y[v]);
}

#if defined(__x86_64__)
/* use_pn: AVX2 and F16C; use_vnni: 1 AVX-VNNI, 2 AVX-512 VNNI (as quant.c) */
static int use_avx2, use_pn, use_vnni;

/* Resolved before main, as quant.c's kernels; JANAS_KERNELS=avx2 leaves
   AVX-VNNI out here too. */
__attribute__((constructor)) static void q6k_init(void)
{
    __builtin_cpu_init();
    use_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    const char *k = getenv("JANAS_KERNELS");
    use_pn = use_avx2 && __builtin_cpu_supports("f16c");
    int plain = k && !strcmp(k, "avx2");
    if (use_pn && !plain && __builtin_cpu_supports("avxvnni"))
        use_vnni = 1;
    else if (use_pn && !plain && __builtin_cpu_supports("avx512vnni") &&
             __builtin_cpu_supports("avx512vl"))
        use_vnni = 2;
}

/* The same, 32 weights at a time: a byte of a plane holds the two bits of
   four weights 32 apart (q6kp_index), so a part's shift and mask of the
   planes' 32 bytes give 32 consecutive weights; their values, less 64, in
   bytes that never carry; then floats, 8 at a time, times the factor of
   their group of 16. Bit for bit axpy_block_ref. */
__attribute__((target("avx2,fma"))) static void
axpy_block_avx2(const struct janas_block_q6kp *b, const uint8_t *a1,
                const uint8_t *a0, int level, float hd, float *y)
{
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i off = _mm256_set1_epi8(level == 3   ? 64
                                         : level == 2 ? 61
                                                      : 49);
    const __m128i up = _mm_cvtsi32_si128(level == 3 ? 1 : level == 2 ? 3 : 5);
    for (int n = 0; n < 2; n++) {
        __m256i q = _mm256_loadu_si256((const __m256i *)(b->q + 32 * n));
        __m256i p1 = level >= 2
                         ? _mm256_loadu_si256((const __m256i *)(a1 + 32 * n))
                         : _mm256_setzero_si256();
        __m256i p0 = level >= 3
                         ? _mm256_loadu_si256((const __m256i *)(a0 + 32 * n))
                         : _mm256_setzero_si256();
        for (int part = 0; part < 4; part++) {
            __m128i sh = _mm_cvtsi32_si128(2 * part);
            __m256i v = _mm256_and_si256(_mm256_srl_epi16(q, sh), m3);
            if (level >= 2)
                v = _mm256_or_si256(
                    _mm256_slli_epi16(v, 2),
                    _mm256_and_si256(_mm256_srl_epi16(p1, sh), m3));
            if (level >= 3)
                v = _mm256_or_si256(
                    _mm256_slli_epi16(v, 2),
                    _mm256_and_si256(_mm256_srl_epi16(p0, sh), m3));
            /* 2v - 64, 8v + 3 - 64, 32v + 15 - 64 (q6kp_value2) */
            v = _mm256_sub_epi8(_mm256_sll_epi16(v, up), off);
            int v0 = 128 * n + 32 * part;
            __m256 c0 = _mm256_set1_ps(hd * (float)b->scales[v0 / 16]);
            __m256 c1 = _mm256_set1_ps(hd * (float)b->scales[v0 / 16 + 1]);
            __m128i lo = _mm256_castsi256_si128(v),
                    hi = _mm256_extracti128_si256(v, 1);
            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
            __m256 w1 =
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
            __m256 w3 =
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
            float *o = y + v0;
            _mm256_storeu_ps(o, _mm256_fmadd_ps(c0, w0, _mm256_loadu_ps(o)));
            _mm256_storeu_ps(o + 8,
                             _mm256_fmadd_ps(c0, w1, _mm256_loadu_ps(o + 8)));
            _mm256_storeu_ps(o + 16,
                             _mm256_fmadd_ps(c1, w2, _mm256_loadu_ps(o + 16)));
            _mm256_storeu_ps(o + 24,
                             _mm256_fmadd_ps(c1, w3, _mm256_loadu_ps(o + 24)));
        }
    }
}
#endif

void janas_q6kpn_axpy(const struct janas_block_q6kp *base, const uint8_t *p1,
                      const uint8_t *p0, int level, float h, float *y, size_t n)
{
    if (level < 1)
        return;
    for (size_t b = 0; b < n / JANAS_QK; b++, y += JANAS_QK) {
        const uint8_t *a1 = level >= 2 ? p1 + b * JANAS_Q6KP_PLANE : NULL;
        const uint8_t *a0 = level >= 3 ? p0 + b * JANAS_Q6KP_PLANE : NULL;
        float hd = h * (janas_fp16_to_fp32(base[b].d) * 0.5f);
#if defined(__x86_64__)
        if (use_avx2) {
            axpy_block_avx2(&base[b], a1, a0, level, hd, y);
            continue;
        }
#endif
        axpy_block_ref(&base[b], a1, a0, level, hd, y);
    }
}

void janas_q6kpn_axpy_ref(const struct janas_block_q6kp *base,
                          const uint8_t *p1, const uint8_t *p0, int level,
                          float h, float *y, size_t n)
{
    if (level < 1)
        return;
    for (size_t b = 0; b < n / JANAS_QK; b++, y += JANAS_QK)
        axpy_block_ref(&base[b], level >= 2 ? p1 + b * JANAS_Q6KP_PLANE : NULL,
                       level >= 3 ? p0 + b * JANAS_Q6KP_PLANE : NULL, level,
                       h * (janas_fp16_to_fp32(base[b].d) * 0.5f), y);
}

void janas_q6kpn_chunk_ref(const struct janas_block_q6kp *base,
                           const uint8_t *p1, const uint8_t *p0, size_t rows,
                           const uint32_t *js, const uint8_t *lv,
                           const float *h, uint32_t n, size_t b, float *y)
{
    float u[JANAS_PN_CHUNK], t = 0.0f;
    if (n > JANAS_PN_CHUNK)
        n = JANAS_PN_CHUNK;
    /* the factors' common scale: the largest |h d / 2| times 128, the
       largest magnitude of an eight-bit scale */
    for (uint32_t i = 0; i < n; i++) {
        const struct janas_block_q6kp *w = base + b * rows + js[i];
        u[i] = h[i] * (janas_fp16_to_fp32(w->d) * 0.5f);
        if (fabsf(u[i]) > t)
            t = fabsf(u[i]);
    }
    if (!(t > 0.0f))
        return;
    float inv = 127.0f / (t * 128.0f), f = t * 128.0f / 127.0f;
    int8_t a[JANAS_PN_CHUNK][JANAS_QK / 16];
    int32_t sum[JANAS_QK / 16] = {0};
    for (uint32_t i = 0; i < n; i++) {
        const struct janas_block_q6kp *w = base + b * rows + js[i];
        float k = u[i] * inv;
        for (int g = 0; g < JANAS_QK / 16; g++) { /* within -127 .. 127 */
            long q = lrintf(k * (float)w->scales[g]);
            a[i][g] = (int8_t)q;
            sum[g] += (int32_t)q;
        }
    }
    /* each output: the integer sum over the neurons, less 64 times the
       factors' (the values' zero), times the scale */
    for (int v = 0; v < JANAS_QK; v++) {
        int32_t p = 0;
        for (uint32_t i = 0; i < n; i++) {
            size_t at = b * rows + js[i];
            const uint8_t *x1 = lv[i] >= 2 ? p1 + at * JANAS_Q6KP_PLANE : NULL;
            const uint8_t *x0 = lv[i] >= 3 ? p0 + at * JANAS_Q6KP_PLANE : NULL;
            p += a[i][v / 16] * q6kp_value2(base + at, x1, x0, v, lv[i]);
        }
        y[v] = fmaf(f, (float)(p - 64 * sum[v / 16]), y[v]);
    }
}

#if defined(__x86_64__)
/* The kernel proper: quant_pn.h, once for AVX2, once for AVX-VNNI and once
   for AVX-512 VNNI on 256 bits. */
#define PN(name) name##_avx2
#define PN_TARGET __attribute__((target("avx2,fma,f16c")))
#define PN_DOT(acc, u, s)                                                      \
    _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s),        \
                                            _mm256_set1_epi16(1)))
#include "quant_pn.h"
#undef PN
#undef PN_TARGET
#undef PN_DOT

#define PN(name) name##_vnni
#define PN_TARGET __attribute__((target("avx2,fma,f16c,avxvnni")))
#define PN_DOT(acc, u, s) _mm256_dpbusd_avx_epi32(acc, u, s)
#include "quant_pn.h"
#undef PN
#undef PN_TARGET
#undef PN_DOT

#define PN(name) name##_evnni
#define PN_TARGET __attribute__((target("avx2,fma,f16c,avx512vnni,avx512vl")))
#define PN_DOT(acc, u, s) _mm256_dpbusd_epi32(acc, u, s)
#include "quant_pn.h"
#undef PN
#undef PN_TARGET
#undef PN_DOT
#endif

void janas_q6kpn_chunk(const struct janas_block_q6kp *base, const uint8_t *p1,
                       const uint8_t *p0, size_t rows, const uint32_t *js,
                       const uint8_t *lv, const float *h, uint32_t n, size_t b,
                       float *y)
{
#if defined(__x86_64__)
    if (use_vnni == 2) {
        q6kpn_chunk_evnni(base, p1, p0, rows, js, lv, h, n, b, y);
        return;
    }
    if (use_vnni) {
        q6kpn_chunk_vnni(base, p1, p0, rows, js, lv, h, n, b, y);
        return;
    }
    if (use_pn) {
        q6kpn_chunk_avx2(base, p1, p0, rows, js, lv, h, n, b, y);
        return;
    }
#endif
    janas_q6kpn_chunk_ref(base, p1, p0, rows, js, lv, h, n, b, y);
}
