/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_ffn.c - the feed-forward of the forward pass: router, routed and
 * shared experts, a dense feed-forward as the one expert of its layer.
 */
#define _GNU_SOURCE
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "attention.h"
#include "common/pool.h"
#include "common/sysinfo.h"
#include "deltanet.h"
#include "tuner.h"
#include "gpu.h"
#include "matvec.h"
#include "quant.h"
#include "vmath.h"

#include "model_impl.h"

/*
 * JANAS_EXPTRACE=1: la fase esperti scomposta nei suoi pezzi - i due prodotti
 * raggruppati (gate+up, down), la SiLU con la quantizzazione, l'accumulo
 * finale, la preparazione e la cache. Serve a separare i kernel dalla colla:
 * senza, la fase e' un numero solo e non si sa quale dei due guardare.
 *
 * Stampa e azzera ogni volta che qualcuno legge le fasi con
 * janas_llm_model_phases(), perche' chi le legge sta separando due tratti
 * (prefill e decodifica) e mescolarli falsa tutto: in decodifica il matvec
 * e' il 95% della fase, in prefill il 67%, e la cache passa dal 2,7% al 29,5%.
 *
 * Spenta costa una lettura di variabile per invio: zero misurabile.
 */
static double ex_setup, ex_t1, ex_act, ex_t2, ex_axpy, ex_cache;
static int ex_trace = -1;

int janas_m_exp_tracing(void)
{
    if (ex_trace < 0) {
        const char *e = getenv("JANAS_EXPTRACE");
        ex_trace = e && atoi(e) > 0;
    }
    return ex_trace;
}

void janas_m_exp_report(void)
{
    double tot = ex_setup + ex_t1 + ex_act + ex_t2 + ex_axpy + ex_cache;
    if (tot <= 0)
        return;
    fprintf(stderr,
            "expert phase: setup %.1f%%  gate+up %.1f%%  act %.1f%%  "
            "down %.1f%%  axpy %.1f%%  cache %.1f%%  (%.0f ms, matvec "
            "%.1f%%)\n",
            100 * ex_setup / tot, 100 * ex_t1 / tot, 100 * ex_act / tot,
            100 * ex_t2 / tot, 100 * ex_axpy / tot, 100 * ex_cache / tot,
            tot * 1e3, 100 * (ex_t1 + ex_t2) / tot);
    ex_setup = ex_t1 = ex_act = ex_t2 = ex_axpy = ex_cache = 0;
}

/* SiLU(gate) * up and its quantization, for the pair rows listed in glist. */
struct act_job {
    struct janas_llm_model *m;
    uint32_t n;
};

#if defined(__x86_64__)
/* h = SiLU(g) * u on 8 lanes at a time; returns how many were done. */
__attribute__((target("avx2,fma"))) static uint32_t
silu_mul_avx2(const float *g, const float *u, float *h, uint32_t n)
{
    uint32_t r = 0;
    for (; r + 8 <= n; r += 8)
        _mm256_storeu_ps(h + r, janas_silu_mul_det8(_mm256_loadu_ps(g + r),
                                                    _mm256_loadu_ps(u + r)));
    return r;
}

__attribute__((target("avx2,fma"))) static uint32_t
gelu_mul_avx2(const float *g, const float *u, float *h, uint32_t n)
{
    uint32_t r = 0;
    for (; r + 8 <= n; r += 8)
        _mm256_storeu_ps(h + r, janas_gelu_mul_det8(_mm256_loadu_ps(g + r),
                                                    _mm256_loadu_ps(u + r)));
    return r;
}
#endif

static void act_worker(void *arg, int tid, int n_threads)
{
    struct act_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t ff = m->d_ff;
    for (uint32_t i = (uint32_t)tid; i < a->n; i += (uint32_t)n_threads) {
        size_t g = m->glist[i];
        const float *gt = m->gate + g * ff, *u = m->up + g * ff;
        float *h = m->h + g * ff;
        /* deterministic operations only, so that a GPU computing an
           expert gives the same bits */
        uint32_t r = 0;
        if (m->a->gelu) {
#if defined(__x86_64__)
            if (m->avx2)
                r = gelu_mul_avx2(gt, u, h, ff);
#endif
            for (; r < ff; r++)
                h[r] = janas_gelu_mul_det(gt[r], u[r]);
        } else {
#if defined(__x86_64__)
            if (m->avx2)
                r = silu_mul_avx2(gt, u, h, ff);
#endif
            for (; r < ff; r++)
                h[r] = janas_silu_mul_det(gt[r], u[r]);
        }
        janas_q8k_quantize_det(h, m->hq + g * ff / JANAS_QK, ff);
    }
}

/* Router logits of a block: item = expert e, for every token of the block,
   so the expert's weight row is read from memory once. */
struct router_job {
    struct janas_llm_model *m;
    const uint16_t *w;
    uint32_t n;
};

static float dot_f16_ref(const uint16_t *w, const float *x, uint32_t n)
{
    float s = 0.0f;
    for (uint32_t i = 0; i < n; i++)
        s += janas_fp16_to_fp32(w[i]) * x[i];
    return s;
}

#if defined(__x86_64__)
__attribute__((target("avx2,fma,f16c"))) static float
dot_f16_avx2(const uint16_t *w, const float *x, uint32_t n)
{
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))),
            _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 8))),
            _mm256_loadu_ps(x + i + 8), a1);
    }
    float s = janas_hsum8(_mm256_add_ps(a0, a1));
    for (; i < n; i++)
        s += janas_fp16_to_fp32(w[i]) * x[i];
    return s;
}
#endif

static void router_worker(void *arg, int tid, int n_threads)
{
    struct router_job *r = arg;
    struct janas_llm_model *m = r->m;
    uint32_t ne = m->n_expert, dm = m->d_model;
    for (uint32_t e = (uint32_t)tid; e < ne; e += (uint32_t)n_threads)
        for (uint32_t j = 0; j < r->n; j++) {
            const uint16_t *w = r->w + (size_t)e * dm;
            const float *x = m->xn + (size_t)j * dm;
#if defined(__x86_64__)
            if (m->avx2 && m->f16c) {
                m->router[(size_t)j * ne + e] = dot_f16_avx2(w, x, dm);
                continue;
            }
#endif
            m->router[(size_t)j * ne + e] = dot_f16_ref(w, x, dm);
        }
}

/*
 * One expert (routed or shared) applied to the pairs order[a .. b); row g of
 * the pair buffers is position g in that order. Matrix p is at base[p] +
 * mx[p].offset.
 */
struct expert_run {
    const uint8_t *base[3];
    const uint8_t *plane[2]; /* the other two planes of down, or NULL */
    const struct janas_jns_matrix *mx;
    uint32_t a, b;
};

#define MAX_RUNS 65 /* a chunk of 64 routed experts and the shared one */

#define EX_T(v, code)                                                          \
    do {                                                                       \
        if (janas_m_exp_tracing()) {                                           \
            double _a = now();                                                 \
            code;                                                              \
            (v) += now() - _a;                                                 \
        } else {                                                               \
            code;                                                              \
        }                                                                      \
    } while (0)

static void run_experts(struct janas_llm_model *m,
                        const struct expert_run *runs, uint32_t n_runs)
{
    double t_enter = janas_m_exp_tracing() ? now() : 0;
    uint32_t dm = m->d_model, ff = m->d_ff;
    size_t nbd = dm / JANAS_QK, nbf = ff / JANAS_QK;
    struct janas_matvec_task t1[2 * MAX_RUNS], t2[MAX_RUNS];
    uint32_t ng = 0;
    for (uint32_t r = 0; r < n_runs; r++) {
        uint32_t a = runs[r].a, cnt = runs[r].b - runs[r].a;
        for (uint32_t g = a; g < runs[r].b; g++) {
            memcpy(m->xg + (size_t)g * nbd,
                   m->xq + (size_t)m->pair_tok[m->order[g]] * nbd,
                   nbd * sizeof(*m->xg));
            m->glist[ng++] = g;
        }
        for (int p = 0; p < 2; p++) {
            const struct janas_jns_matrix *mx = &runs[r].mx[p];
            t1[2 * r + p] = (struct janas_matvec_task){
                .type = (int)mx->type,
                .w = runs[r].base[p] + mx->offset,
                .x = m->xg + (size_t)a * nbd,
                .y = (p ? m->up : m->gate) + (size_t)a * ff,
                .rows = mx->rows,
                .cols = mx->cols,
                .n_vec = cnt,
                .x_stride = nbd,
                .y_stride = ff};
        }
        const struct janas_jns_matrix *d = &runs[r].mx[JANAS_JNS_DOWN];
        t2[r] = (struct janas_matvec_task){
            .type = (int)d->type,
            .w = runs[r].base[JANAS_JNS_DOWN] + d->offset,
            .plane = {runs[r].plane[0], runs[r].plane[1]},
            .x = m->hq + (size_t)a * nbf,
            .y = m->dout + (size_t)a * dm,
            .rows = d->rows,
            .cols = d->cols,
            .n_vec = cnt,
            .x_stride = nbf,
            .y_stride = dm};
    }
    if (janas_m_exp_tracing())
        ex_setup += now() - t_enter;
    EX_T(ex_t1,
         janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                                m->compute, t1, 2 * (size_t)n_runs));
    struct act_job aj = {m, ng};
    EX_T(ex_act, janas_pool_run(m->compute, act_worker, &aj));
    EX_T(ex_t2,
         janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                                m->compute, t2, n_runs));
}

/* The router's choice for token j: softmax, top-k, normalized weights. */
static void route_tok(struct janas_llm_model *m, uint32_t j)
{
    uint32_t ne = m->n_expert, k = m->k_use;
    float *r = m->router + (size_t)j * ne, mx = -INFINITY, sum = 0.0f;
    for (uint32_t e = 0; e < ne; e++)
        if (r[e] > mx)
            mx = r[e];
    for (uint32_t e = 0; e < ne; e++) {
        r[e] = expf(r[e] - mx);
        sum += r[e];
    }
    float wsum = 0.0f;
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best = 0;
        float bv = -1.0f;
        for (uint32_t e = 0; e < ne; e++)
            if (r[e] > bv) {
                bv = r[e];
                best = e;
            }
        r[best] = -2.0f; /* taken */
        uint32_t p = j * k + i;
        m->pair_tok[p] = j;
        m->pair_exp[p] = best;
        m->pair_w[p] = bv / sum;
        wsum += bv / sum;
    }
    for (uint32_t i = 0; i < k; i++)
        m->pair_w[j * k + i] /= wsum;
}

static void route_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads)
        route_tok(t->m, j);
}

/* The experts' outputs added to the tokens: each token takes its own pairs
   in the order of the pair buffers, as the caller alone would. */
struct axpy_job {
    struct janas_llm_model *m;
    const uint32_t *first; /* token j's pairs: rows list[first[j]..[j+1]) */
    const uint32_t *list;
    uint32_t n;
};

static void axpy_worker(void *arg, int tid, int n_threads)
{
    struct axpy_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t dm = m->d_model;
    for (uint32_t j = (uint32_t)tid; j < a->n; j += (uint32_t)n_threads)
        for (uint32_t c = a->first[j]; c < a->first[j + 1]; c++) {
            uint32_t g = a->list[c];
            janas_axpy_f32(m->x + (size_t)j * dm, m->pair_w[m->order[g]],
                           m->dout + (size_t)g * dm, dm);
        }
}

/* Routed (and shared) experts of a block, added to x. The routed experts
   are layer l of the file behind the cache (the main model or the MTP). */
int janas_m_moe_block(struct janas_llm_model *m, const struct layer *ly,
                      struct janas_expert_cache *cache,
                      const struct janas_jns_layer *jl, uint32_t l, uint32_t n)
{
    double t0 = now();
    uint32_t dm = m->d_model, ne = m->n_expert, k = m->k_use;
    struct router_job rj = {m, ly->router16, n};
    janas_pool_run(m->compute, router_worker, &rj);

    /* per token: softmax, top-k, normalized weights */
    struct tok_job tj = {.m = m, .n = n};
    if (n < 2)
        route_tok(m, 0);
    else
        janas_pool_run(m->compute, route_worker, &tj);
    /* counting sort of the pairs by expert */
    uint32_t np = n * k;
    memset(m->count, 0, (ne + 1) * sizeof(uint32_t));
    for (uint32_t p = 0; p < np; p++)
        m->count[m->pair_exp[p] + 1]++;
    for (uint32_t e = 0; e < ne; e++)
        m->count[e + 1] += m->count[e];
    uint32_t distinct[4096], nd = 0; /* n_expert <= 4096, checked at load */
    for (uint32_t e = 0; e < ne; e++)
        if (m->count[e + 1] > m->count[e])
            distinct[nd++] = e;
    {
        uint32_t fill[4096];
        for (uint32_t e = 0; e < ne; e++)
            fill[e] = m->count[e];
        for (uint32_t p = 0; p < np; p++)
            m->order[fill[m->pair_exp[p]]++] = p;
    }
    /* the shared expert: one more pair per token, after the routed ones */
    uint32_t nt = np;
    if (m->a->shared_expert) {
        for (uint32_t j = 0; j < n; j++, nt++) {
            m->pair_tok[nt] = j;
            m->pair_w[nt] =
                sigmoid(janas_dot_f32(ly->sh_gate, m->xn + (size_t)j * dm, dm));
            m->order[nt] = nt;
        }
    }
    double t1 = now();
    m->phase[JANAS_PH_ROUTER] += t1 - t0;

    /* experts in chunks: compute those in RAM while the others load; the
       shared expert, resident, goes with the first of them */
    for (uint32_t c0 = 0; c0 < nd || (c0 == 0 && nt > np); c0 += 64) {
        uint32_t cn = nd - c0 < 64 ? nd - c0 : 64;
        const uint8_t *slots[64];
        uint8_t ready[64];
        if (cn) {
            int rc_b;
            EX_T(ex_cache, rc_b = janas_expert_cache_begin(
                               cache, l, distinct + c0, cn, slots, ready));
            if (rc_b < 0)
                return -1;
        }
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 1 && cn) {
                int rc_f;
                EX_T(ex_cache, rc_f = janas_expert_cache_finish(cache));
                if (rc_f != 0)
                    return -1;
            }
            struct expert_run runs[MAX_RUNS];
            uint32_t nr = 0;
            if (pass == 0 && c0 == 0 && nt > np)
                runs[nr++] =
                    (struct expert_run){{ly->sh_w[0], ly->sh_w[1], ly->sh_w[2]},
                                        {NULL, NULL},
                                        ly->sh_mx,
                                        np,
                                        nt};
            for (uint32_t i = 0; i < cn; i++)
                if ((pass == 0) == (ready[i] != 0)) {
                    uint32_t e = distinct[c0 + i];
                    if (m->use_count && cache == m->cache)
                        m->use_count[(size_t)l * ne + e]++; /* the profile */
                    /* the planes of the down matrix the cache holds */
                    const uint8_t *pl[2] = {NULL, NULL};
                    for (int q = 0; q < m->exp_level - 1; q++)
                        if (jl->plane[q].type)
                            pl[q] = slots[i] + jl->plane[q].offset;
                    runs[nr++] =
                        (struct expert_run){{slots[i], slots[i], slots[i]},
                                            {pl[0], pl[1]},
                                            jl->m,
                                            m->count[e],
                                            m->count[e + 1]};
                }
            if (nr)
                run_experts(m, runs, nr);
        }
        if (nd <= c0 + 64)
            break;
    }
    if (n < 2) {
        EX_T(
            ex_axpy, for (uint32_t g = 0; g < nt; g++) {
                uint32_t p = m->order[g];
                janas_axpy_f32(m->x + (size_t)m->pair_tok[p] * dm, m->pair_w[p],
                               m->dout + (size_t)g * dm, dm);
            });
    } else {
        /* the rows of each token, in increasing order: a counting sort */
        uint32_t *first = m->tok_first, *list = m->tok_list;
        uint32_t *fill = m->tok_fill;
        memset(first, 0, (n + 1) * sizeof(uint32_t));
        for (uint32_t g = 0; g < nt; g++)
            first[m->pair_tok[m->order[g]] + 1]++;
        for (uint32_t j = 0; j < n; j++)
            first[j + 1] += first[j];
        memcpy(fill, first, n * sizeof(uint32_t));
        for (uint32_t g = 0; g < nt; g++)
            list[fill[m->pair_tok[m->order[g]]]++] = g;
        struct axpy_job aj = {m, first, list, n};
        EX_T(ex_axpy, janas_pool_run(m->compute, axpy_worker, &aj));
    }
    m->phase[JANAS_PH_EXPERTS] += now() - t1;
    return 0;
}
