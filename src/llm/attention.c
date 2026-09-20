/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * attention.c - causal grouped-query attention (see attention.h).
 */
#include "attention.h"

#include <math.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "quant.h"
#include "vmath.h"

#define ATT_CHUNK 256 /* positions per work item */

/*
 * Causal attention of a block by GQA group and chunk of positions: item =
 * (KV head, chunk of ATT_CHUNK positions, subgroup of the group's query
 * heads). Every K and V row is read once for all the query heads and tokens
 * of the item, and a long context is spread over all the threads even with
 * few KV heads. An item leaves, per (token, query head), a partial softmax:
 * its largest score, the sum of exp(score - largest) and the V rows weighted
 * by them; the partials are merged in chunk order. Chunks start at absolute
 * positions, so a token's result does not depend on the block it runs in.
 */
struct janas_attn {
    uint32_t n_head, n_kv, hd, n_ctx, chunks;
    float *part; /* partials: n_kv x chunks x MAX_BLOCK x group x (hd + 2) */
    float *scr;  /* per thread: the scores of the item's slots */
    int16_t *qi; /* q to 16 bits: MAX_BLOCK x n_head x hd */
    float *qs;   /* its scale, one per token and head */
    int n_threads;
    int scores;      /* JANAS_ATTN_FLOAT or JANAS_ATTN_INT16, resolved */
    int scores_auto; /* 1: the engine chose it */
    float scale;     /* of the scores: 1 / sqrt(head_dim) unless set */
    uint32_t window; /* a token sees the last window positions; 0: all */
};

struct attn_job {
    struct janas_attn *A;
    const float *q;
    const int16_t *qi;
    const float *qs;
    const int8_t *k, *v;
    const float *ks, *vs; /* one scale per position and KV head */
    float *out;
    uint32_t n, pos0, n_chunks, split;
    atomic_uint next; /* next item to claim */
};

/* The partial of chunk c for token j and query head g of KV head kvh:
   largest score, sum of weights, then head_dim weighted values. */
static float *att_partial(const struct janas_attn *A, uint32_t kvh, uint32_t c,
                          uint32_t j, uint32_t g)
{
    uint32_t group = A->n_head / A->n_kv;
    return A->part +
           ((((size_t)kvh * A->chunks + c) * JANAS_ATTN_MAX_BLOCK + j) * group +
            g) *
               (A->hd + 2);
}

/*
 * Kernels on ns query vectors (slots) at once: the heads of a GQA group, or
 * tokens times heads of a block. Scores: every key row is converted once for
 * all of them. Values: for each 8-wide slice of head_dim the ns accumulators
 * stay in registers over all the positions of the chunk. ns is 8, 4, 2 or 1,
 * a constant in each specialisation, so the slot loops unroll into
 * registers (GCC keeps an array of accumulators in memory otherwise). Each
 * slot's arithmetic is independent of the others: grouping never changes a
 * result.
 */
/* The query as it is: the sum is a floating-point one. */
static void att_scores_f32_ref(const float *const *q, const int8_t *K,
                               const float *ks, uint32_t nt, uint32_t hd,
                               float scale, float *const *sc, int ns)
{
    for (uint32_t t = 0; t < nt; t++)
        for (int s = 0; s < ns; s++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < hd; i++)
                sum += q[s][i] * (float)K[(size_t)t * hd + i];
            sc[s][t] = sum * scale * ks[t];
        }
}

/* The query to sixteen bits: the sum is an exact integer one. */
static void att_scores_i16_ref(const int16_t *const *q, const int8_t *K,
                               const float *ks, uint32_t nt, uint32_t hd,
                               const float *qsc, float scale, float *const *sc,
                               int ns)
{
    for (uint32_t t = 0; t < nt; t++)
        for (int s = 0; s < ns; s++) {
            int32_t sum = 0;
            for (uint32_t i = 0; i < hd; i++)
                sum += (int32_t)q[s][i] * K[(size_t)t * hd + i];
            sc[s][t] = (float)sum * qsc[s] * scale * ks[t];
        }
}

static void att_values_ref(const float *const *w, const int8_t *V, uint32_t nt,
                           uint32_t hd, float *const *out, int ns)
{
    for (int s = 0; s < ns; s++)
        for (uint32_t i = 0; i < hd; i++) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < nt; t++)
                acc += w[s][t] * (float)V[(size_t)t * hd + i];
            out[s][i] = acc;
        }
}

#if defined(__x86_64__)
#define AVX2 __attribute__((target("avx2,fma,f16c")))

/* The scores over the query as it is: one key row converted for all the
   slots, one FMA per slot and eight weights. */
static inline AVX2 __attribute__((always_inline)) void
f32_body(const float *const *q, const int8_t *K, const float *ks, uint32_t nt,
         uint32_t hd, float scale, float *const *sc, const int8_t *pf,
         const int ns)
{
    for (uint32_t t = 0; t < nt; t++) {
        const int8_t *k = K + (size_t)t * hd;
        /* the values pass reads V by columns, which the hardware prefetcher
           does not follow: bring its rows into L2 in order, now */
        if (pf)
            for (uint32_t i = 0; i < hd; i += 32)
                _mm_prefetch((const char *)(pf + (size_t)t * hd + i),
                             _MM_HINT_T1);
        __m256 acc[8];
        _Pragma("GCC unroll 8") for (int s = 0; s < ns; s++) acc[s] =
            _mm256_setzero_ps();
        for (uint32_t i = 0; i < hd; i += 8) {
            __m256 kv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)(k + i))));
            _Pragma("GCC unroll 8") for (int s = 0; s < ns; s++) acc[s] =
                _mm256_fmadd_ps(_mm256_loadu_ps(q[s] + i), kv, acc[s]);
        }
        _Pragma("GCC unroll 8") for (int s = 0; s < ns; s++)
        {
            __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc[s]),
                                  _mm256_extractf128_ps(acc[s], 1));
            r = _mm_add_ps(r, _mm_movehl_ps(r, r));
            r = _mm_add_ss(r, _mm_movehdup_ps(r));
            sc[s][t] = _mm_cvtss_f32(r) * scale * ks[t];
        }
    }
}

static AVX2 void att_scores_f32_avx2(const float *const *q, const int8_t *K,
                                     const float *ks, uint32_t nt, uint32_t hd,
                                     float scale, float *const *sc,
                                     const int8_t *pf, int ns)
{
    switch (ns) {
    case 8:
        f32_body(q, K, ks, nt, hd, scale, sc, pf, 8);
        break;
    case 4:
        f32_body(q, K, ks, nt, hd, scale, sc, pf, 4);
        break;
    case 2:
        f32_body(q, K, ks, nt, hd, scale, sc, pf, 2);
        break;
    default:
        f32_body(q, K, ks, nt, hd, scale, sc, pf, 1);
    }
}

/* The scores over the sixteen-bit query, once for AVX2 and once for AVX-VNNI:
   see attention_scores.h. Both sum the same integers, so both give the same
   scores, and the same ones as att_scores_i16_ref. */
#define SCORES(name) name##_i16_avx2
#define SCORES_TARGET AVX2
#define SCORES_ACC(acc, a, b) _mm256_add_epi32(acc, _mm256_madd_epi16(a, b))
#include "attention_scores.h"
#undef SCORES
#undef SCORES_TARGET
#undef SCORES_ACC

#define SCORES(name) name##_i16_vnni
#define SCORES_TARGET __attribute__((target("avx2,fma,f16c,avxvnni")))
#define SCORES_ACC(acc, a, b) _mm256_dpwssd_avx_epi32(acc, a, b)
#include "attention_scores.h"
#undef SCORES
#undef SCORES_TARGET
#undef SCORES_ACC

static inline AVX2 __attribute__((always_inline)) void
values_body(const float *const *w, const int8_t *V, uint32_t nt, uint32_t hd,
            float *const *out, const int ns)
{
    for (uint32_t i = 0; i < hd; i += 8) {
        __m256 acc[8];
        _Pragma("GCC unroll 8") for (int s = 0; s < ns; s++) acc[s] =
            _mm256_setzero_ps();
        for (uint32_t t = 0; t < nt; t++) {
            __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)(V + (size_t)t * hd + i))));
            _Pragma("GCC unroll 8") for (int s = 0; s < ns; s++) acc[s] =
                _mm256_fmadd_ps(_mm256_broadcast_ss(w[s] + t), v, acc[s]);
        }
        _Pragma("GCC unroll 8") for (int s = 0; s < ns; s++)
            _mm256_storeu_ps(out[s] + i, acc[s]);
    }
}

static AVX2 void att_values_avx2(const float *const *w, const int8_t *V,
                                 uint32_t nt, uint32_t hd, float *const *out,
                                 int ns)
{
    switch (ns) {
    case 8:
        values_body(w, V, nt, hd, out, 8);
        break;
    case 4:
        values_body(w, V, nt, hd, out, 4);
        break;
    case 2:
        values_body(w, V, nt, hd, out, 2);
        break;
    default:
        values_body(w, V, nt, hd, out, 1);
    }
}
#endif

/*
 * Softmax numerators of one slot's row of nt scores, in place: positions at
 * and after vis (not yet visible to the token) weigh 0, and so does anything
 * below e^-80 of the largest score (skipping exp avoids denormals). Stores
 * the largest visible score in *mx, returns the sum of the weights. A row of
 * the same token in a larger block only has more zeros, which change neither
 * the largest score nor, lane by lane, the sum.
 */
static float att_weights_ref(float *row, uint32_t nt, uint32_t vis, float *mx)
{
    float m = -INFINITY, sum[8] = {0};
    for (uint32_t t = 0; t < vis; t++)
        if (row[t] > m)
            m = row[t];
    for (uint32_t t = 0; t < nt; t++) {
        float d = row[t] - m;
        row[t] = t >= vis || d < -80.0f ? 0.0f : expf(d);
        sum[t % 8] += row[t];
    }
    *mx = m;
    return ((sum[0] + sum[4]) + (sum[2] + sum[6])) +
           ((sum[1] + sum[5]) + (sum[3] + sum[7]));
}

#if defined(__x86_64__)
/* The same, eight positions at a time; row has room for a multiple of 8. */
__attribute__((target("avx2,fma"))) static float
att_weights_avx2(float *row, uint32_t nt, uint32_t vis, float *mx)
{
    __m256 vm = _mm256_set1_ps(-INFINITY);
    uint32_t t = 0;
    for (; t + 8 <= vis; t += 8)
        vm = _mm256_max_ps(vm, _mm256_loadu_ps(row + t));
    __m128 h =
        _mm_max_ps(_mm256_castps256_ps128(vm), _mm256_extractf128_ps(vm, 1));
    h = _mm_max_ps(h, _mm_movehl_ps(h, h));
    h = _mm_max_ss(h, _mm_movehdup_ps(h));
    float m = _mm_cvtss_f32(h);
    for (; t < vis; t++)
        if (row[t] > m)
            m = row[t];
    __m256 vmx = _mm256_set1_ps(m), lim = _mm256_set1_ps(-80.0f);
    __m256 acc = _mm256_setzero_ps();
    __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i vvis = _mm256_set1_epi32((int)vis);
    for (t = 0; t < nt; t += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + t), vmx);
        __m256i ti = _mm256_add_epi32(idx, _mm256_set1_epi32((int)t));
        __m256 keep =
            _mm256_and_ps(_mm256_cmp_ps(d, lim, _CMP_GE_OQ),
                          _mm256_castsi256_ps(_mm256_cmpgt_epi32(vvis, ti)));
        __m256 w = _mm256_and_ps(janas_exp8(d), keep);
        _mm256_storeu_ps(row + t, w);
        acc = _mm256_add_ps(acc, w);
    }
    *mx = m;
    /* the same pairing as att_weights_ref */
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi); /* lanes 0+4, 1+5, 2+6, 3+7 */
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4)); /* (0+4)+(2+6), ... */
    return _mm_cvtss_f32(_mm_add_ss(s2, _mm_movehdup_ps(s2)));
}
#endif

static void attn_worker(void *arg, int tid, int n_threads)
{
    (void)n_threads;
    struct attn_job *a = arg;
    struct janas_attn *A = a->A;
    uint32_t hd = A->hd;
    uint32_t group = A->n_head / A->n_kv, gs = group / a->split;
    uint32_t items = A->n_kv * a->n_chunks * a->split;
    float scale = A->scale;
    float *sc = A->scr + (size_t)tid * JANAS_ATTN_MAX_BLOCK * group * ATT_CHUNK;
#if defined(__x86_64__)
    int avx = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
              __builtin_cpu_supports("f16c");
    int vnni = avx && janas_use_vnni();
#endif
    for (;;) {
        /* items claimed in order: a thread that finishes early takes more */
        uint32_t it =
            atomic_fetch_add_explicit(&a->next, 1, memory_order_relaxed);
        if (it >= items)
            break;
        uint32_t sub = it % a->split, c = it / a->split % a->n_chunks;
        uint32_t kvh = it / a->split / a->n_chunks, g0 = sub * gs;
        uint32_t t0 = c * ATT_CHUNK, t1 = t0 + ATT_CHUNK;
        if (t1 > a->pos0 + a->n)
            t1 = a->pos0 + a->n;
        uint32_t nt = t1 - t0;
        const int8_t *K = a->k + ((size_t)kvh * A->n_ctx + t0) * hd;
        const int8_t *V = a->v + ((size_t)kvh * A->n_ctx + t0) * hd;
        const float *ks = a->ks + (size_t)kvh * A->n_ctx + t0;
        const float *vs = a->vs + (size_t)kvh * A->n_ctx + t0;
        /* token j sees positions up to pos0 + j: tokens before jlo see
           nothing of this chunk */
        uint32_t jlo = t0 > a->pos0 ? t0 - a->pos0 : 0;
        for (uint32_t j = 0; j < jlo && j < a->n; j++)
            for (uint32_t g = g0; g < g0 + gs; g++) {
                float *pm = att_partial(A, kvh, c, j, g);
                pm[0] = -INFINITY;
                pm[1] = 0.0f;
                memset(pm + 2, 0, hd * sizeof(float));
            }
        if (jlo >= a->n)
            continue;
        /* slot = (j - jlo) * gs + g, taken 8, 4, 2 or 1 at a time */
        uint32_t ns = (a->n - jlo) * gs, step;
        for (uint32_t s0 = 0; s0 < ns; s0 += step) {
            step = ns - s0 >= 8 ? 8 : ns - s0 >= 4 ? 4 : ns - s0 >= 2 ? 2 : 1;
            const int16_t *qi[8];
            const float *qf[8];
            float qsc[8], *srow[8];
            int fast = A->scores == JANAS_ATTN_INT16;
            for (uint32_t s = 0; s < step; s++) {
                uint32_t sl = s0 + s, j = jlo + sl / gs, g = g0 + sl % gs;
                size_t h = (size_t)j * A->n_head + kvh * group + g;
                if (fast) {
                    qi[s] = a->qi + h * hd;
                    qsc[s] = a->qs[h];
                } else {
                    qf[s] = a->q + h * hd;
                }
                srow[s] = sc + (size_t)sl * ATT_CHUNK;
            }
            const int8_t *pf = s0 ? NULL : V;
            if (fast) {
#if defined(__x86_64__)
                if (vnni)
                    att_scores_i16_vnni(qi, K, ks, nt, hd, qsc, scale, srow, pf,
                                        (int)step);
                else if (avx)
                    att_scores_i16_avx2(qi, K, ks, nt, hd, qsc, scale, srow, pf,
                                        (int)step);
                else
#endif
                    att_scores_i16_ref(qi, K, ks, nt, hd, qsc, scale, srow,
                                       (int)step);
            } else {
#if defined(__x86_64__)
                if (avx)
                    att_scores_f32_avx2(qf, K, ks, nt, hd, scale, srow, pf,
                                        (int)step);
                else
#endif
                    att_scores_f32_ref(qf, K, ks, nt, hd, scale, srow,
                                       (int)step);
            }
        }
        /* per slot: largest visible score, weights and their sum */
        for (uint32_t sl = 0; sl < ns; sl++) {
            uint32_t j = jlo + sl / gs;
            uint32_t vis = a->pos0 + j + 1 - t0; /* visible positions */
            float *pm = att_partial(A, kvh, c, j, g0 + sl % gs);
            if (A->window && a->pos0 + j + 1 > A->window) {
                /* a sliding window: the positions before its start weigh
                   nothing, and a chunk wholly before it nothing at all */
                uint32_t lo = a->pos0 + j + 1 - A->window;
                float *row = sc + (size_t)sl * ATT_CHUNK;
                if (lo >= t1) {
                    pm[0] = -INFINITY;
                    pm[1] = 0.0f;
                    memset(row, 0, nt * sizeof(float));
                    continue;
                }
                for (uint32_t t = t0; t < lo; t++)
                    row[t - t0] = -INFINITY;
            }
#if defined(__x86_64__)
            if (avx)
                pm[1] = att_weights_avx2(sc + (size_t)sl * ATT_CHUNK, nt,
                                         vis < nt ? vis : nt, pm);
            else
#endif
                pm[1] = att_weights_ref(sc + (size_t)sl * ATT_CHUNK, nt,
                                        vis < nt ? vis : nt, pm);
            /* the scale of a position belongs to its value: folded into the
               weight, it costs one multiply per position instead of one per
               number read. The sum in pm[1] stays what it was. */
            float *row = sc + (size_t)sl * ATT_CHUNK;
            for (uint32_t t = 0; t < nt; t++)
                row[t] *= vs[t];
        }
        for (uint32_t s0 = 0; s0 < ns; s0 += step) {
            step = ns - s0 >= 8 ? 8 : ns - s0 >= 4 ? 4 : ns - s0 >= 2 ? 2 : 1;
            const float *w[8];
            float *out[8];
            for (uint32_t s = 0; s < step; s++) {
                uint32_t sl = s0 + s;
                w[s] = sc + (size_t)sl * ATT_CHUNK;
                out[s] =
                    att_partial(A, kvh, c, jlo + sl / gs, g0 + sl % gs) + 2;
            }
#if defined(__x86_64__)
            if (avx)
                att_values_avx2(w, V, nt, hd, out, (int)step);
            else
#endif
                att_values_ref(w, V, nt, hd, out, (int)step);
        }
    }
}

/*
 * The query to sixteen bits, one scale per token and head, before the chunks
 * are touched: the scores then rest on an exact integer sum instead of a
 * float one, and the only error left is this quantization - a relative 3e-5,
 * against the 4e-3 the eight-bit keys already carry. A vector's scale comes
 * from that vector alone, so a token gives the same numbers whatever block it
 * runs in, and the integers are the same on every instruction set.
 */
static void quant_worker(void *arg, int tid, int n_threads)
{
    struct attn_job *a = arg;
    struct janas_attn *A = a->A;
    uint32_t hd = A->hd, items = a->n * A->n_head;
    for (uint32_t it = (uint32_t)tid; it < items; it += (uint32_t)n_threads) {
        const float *src = a->q + (size_t)it * hd;
        int16_t *dst = A->qi + (size_t)it * hd;
        float mx = 0.0f;
        for (uint32_t i = 0; i < hd; i++) {
            float v = src[i] < 0.0f ? -src[i] : src[i];
            if (v > mx)
                mx = v;
        }
        A->qs[it] = mx / 32767.0f;
        float inv = mx > 0.0f ? 32767.0f / mx : 0.0f;
        for (uint32_t i = 0; i < hd; i++)
            dst[i] = (int16_t)lrintf(src[i] * inv);
    }
}

/*
 * JANAS_ATTN_SKIPSTAT=1: how many chunks a row could have skipped, had it
 * known in advance which ones weigh nothing - their largest score more
 * than 8, 16 or 32 below the row's (e^-16 is below what a float sum
 * keeps). The ceiling of any way of skipping chunks by a bound on their
 * scores; printed when the program ends, over all rows and over those
 * past 16384 positions. Off, it costs one test per merge item.
 */
static int skip_stat = -1;
static _Atomic uint64_t skip_n[2], skip_k[2][3];

static void skip_report(void)
{
    static const char *what[2] = {"all rows", "rows past 16384"};
    for (int w = 0; w < 2; w++) {
        uint64_t n = atomic_load(&skip_n[w]);
        if (!n)
            continue;
        fprintf(stderr,
                "attention skip ceiling, %s: %llu chunks; below max-8 "
                "%.1f%%, max-16 %.1f%%, max-32 %.1f%%\n",
                what[w], (unsigned long long)n,
                100.0 * atomic_load(&skip_k[w][0]) / n,
                100.0 * atomic_load(&skip_k[w][1]) / n,
                100.0 * atomic_load(&skip_k[w][2]) / n);
    }
}

/* Merge of the partials, item = (token j, query head h). */
static void merge_worker(void *arg, int tid, int n_threads)
{
    struct attn_job *a = arg;
    struct janas_attn *A = a->A;
    uint32_t hd = A->hd, qd = A->n_head * hd;
    uint32_t group = A->n_head / A->n_kv, items = a->n * A->n_head;
    for (uint32_t it = (uint32_t)tid; it < items; it += (uint32_t)n_threads) {
        uint32_t j = it / A->n_head, h = it % A->n_head;
        uint32_t kvh = h / group, g = h % group;
        uint32_t nc = (a->pos0 + j + ATT_CHUNK) / ATT_CHUNK;
        float mx = -INFINITY, sum = 0.0f;
        for (uint32_t c = 0; c < nc; c++) {
            float v = att_partial(A, kvh, c, j, g)[0];
            if (v > mx)
                mx = v;
        }
        if (skip_stat > 0) {
            uint64_t k[3] = {0};
            for (uint32_t c = 0; c < nc; c++) {
                float v = att_partial(A, kvh, c, j, g)[0];
                k[0] += v < mx - 8.0f;
                k[1] += v < mx - 16.0f;
                k[2] += v < mx - 32.0f;
            }
            for (int w = 0; w < 2; w++) {
                if (w == 1 && a->pos0 + j < 16384)
                    break;
                atomic_fetch_add(&skip_n[w], nc);
                for (int t = 0; t < 3; t++)
                    atomic_fetch_add(&skip_k[w][t], k[t]);
            }
        }
        float *out = a->out + (size_t)j * qd + (size_t)h * hd;
        memset(out, 0, hd * sizeof(float));
        for (uint32_t c = 0; c < nc; c++) {
            const float *pm = att_partial(A, kvh, c, j, g);
            float w = expf(pm[0] - mx);
            sum += w * pm[1];
            janas_axpy_f32(out, w, pm + 2, hd);
        }
        float inv = 1.0f / sum;
        for (uint32_t i = 0; i < hd; i++)
            out[i] *= inv;
    }
}

/*
 * Which way the scores go, when the caller leaves it open: the sixteen-bit
 * query where the arithmetic for it is there and the caller asked for more
 * context than the usual sixteen thousand tokens. A context is a ceiling,
 * not a promise: a conversation that stays short pays the query's
 * quantization for a gain it never sees, so the engine waits to be asked for
 * a long one - there attention is the greater part of a token, and the
 * sixteen-bit query takes a fifth to a quarter off it.
 */
static int scores_choice(int asked, uint32_t head_dim, uint32_t n_ctx,
                         int *chosen_here)
{
    const char *e = getenv("JANAS_ATTN");
    if (e && !strcmp(e, "float"))
        asked = JANAS_ATTN_FLOAT;
    else if (e && !strcmp(e, "int16"))
        asked = JANAS_ATTN_INT16;
    else if (e && !strcmp(e, "auto"))
        asked = JANAS_ATTN_AUTO;
    *chosen_here = asked == JANAS_ATTN_AUTO;
    /* sixteen keys at a time, so head_dim has to be a multiple of sixteen */
    if (head_dim % 16)
        return JANAS_ATTN_FLOAT;
    if (asked == JANAS_ATTN_FLOAT || asked == JANAS_ATTN_INT16)
        return asked;
    return n_ctx > 16384 ? JANAS_ATTN_INT16 : JANAS_ATTN_FLOAT;
}

size_t janas_attn_part_bytes(uint32_t n_head, uint32_t n_head_kv,
                             uint32_t head_dim, uint32_t n_ctx)
{
    if (n_head_kv == 0)
        return 0;
    size_t chunks = (n_ctx + ATT_CHUNK - 1) / ATT_CHUNK;
    return (size_t)n_head_kv * chunks * JANAS_ATTN_MAX_BLOCK *
           (n_head / n_head_kv) * (head_dim + 2) * sizeof(float);
}

struct janas_attn *janas_attn_create(uint32_t n_head, uint32_t n_head_kv,
                                     uint32_t head_dim, uint32_t n_ctx,
                                     int n_threads, int scores)
{
    if (n_head_kv == 0 || n_head % n_head_kv || head_dim % 8 ||
        head_dim > 512 || n_threads < 1)
        return NULL;
    struct janas_attn *A = calloc(1, sizeof(*A));
    if (!A)
        return NULL;
    uint32_t group = n_head / n_head_kv;
    A->n_head = n_head;
    A->n_kv = n_head_kv;
    A->hd = head_dim;
    A->n_ctx = n_ctx;
    A->chunks = (n_ctx + ATT_CHUNK - 1) / ATT_CHUNK;
    A->n_threads = n_threads;
    A->scores = scores_choice(scores, head_dim, n_ctx, &A->scores_auto);
    A->scale = 1.0f / sqrtf((float)head_dim);
    A->part = malloc(janas_attn_part_bytes(n_head, n_head_kv, head_dim, n_ctx));
    A->scr = malloc((size_t)n_threads * (JANAS_ATTN_MAX_BLOCK * group + 2) *
                    ATT_CHUNK * sizeof(float));
    if (A->scores == JANAS_ATTN_INT16) {
        A->qi = malloc((size_t)JANAS_ATTN_MAX_BLOCK * n_head * head_dim *
                       sizeof(int16_t));
        A->qs = malloc((size_t)JANAS_ATTN_MAX_BLOCK * n_head * sizeof(float));
    }
    if (!A->part || !A->scr ||
        (A->scores == JANAS_ATTN_INT16 && (!A->qi || !A->qs))) {
        janas_attn_destroy(A);
        return NULL;
    }
    return A;
}

void janas_attn_set_scale(struct janas_attn *a, float scale)
{
    a->scale = scale;
}

void janas_attn_set_window(struct janas_attn *a, uint32_t window)
{
    a->window = window;
}

int janas_attn_scores_mode(const struct janas_attn *a)
{
    return a ? a->scores : JANAS_ATTN_FLOAT;
}

int janas_attn_scores_auto(const struct janas_attn *a)
{
    return a ? a->scores_auto : 0;
}

void janas_attn_destroy(struct janas_attn *A)
{
    if (!A)
        return;
    free(A->part);
    free(A->scr);
    free(A->qi);
    free(A->qs);
    free(A);
}

void janas_attn_run(struct janas_attn *A, struct janas_pool *pool,
                    const float *q, const int8_t *k, const float *ks,
                    const int8_t *v, const float *vs, uint32_t n, uint32_t pos0,
                    float *out)
{
    /* enough items for every thread: split the query heads of a group */
    uint32_t n_chunks = (pos0 + n + ATT_CHUNK - 1) / ATT_CHUNK, split = 1;
    uint32_t group = A->n_head / A->n_kv;
    while (A->n_kv * n_chunks * split < 2 * (uint32_t)A->n_threads &&
           group % (split * 2) == 0)
        split *= 2;
    struct attn_job job = {.A = A,
                           .q = q,
                           .k = k,
                           .v = v,
                           .ks = ks,
                           .vs = vs,
                           .out = out,
                           .n = n,
                           .pos0 = pos0,
                           .n_chunks = n_chunks,
                           .split = split};
    job.qi = A->qi;
    job.qs = A->qs;
    atomic_init(&job.next, 0);
    if (skip_stat < 0) {
        const char *e = getenv("JANAS_ATTN_SKIPSTAT");
        skip_stat = e && atoi(e) > 0;
        if (skip_stat)
            atexit(skip_report);
    }
    if (A->scores == JANAS_ATTN_INT16)
        janas_pool_run(pool, quant_worker, &job);
    janas_pool_run(pool, attn_worker, &job);
    janas_pool_run(pool, merge_worker, &job);
}
