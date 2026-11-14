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
 *
 * In piu' (21 ott 2026) il tratto fra il router e i primi esperti, dove il
 * thread principale lavora da solo: scelta degli esperti e ordinamento,
 * apertura della cache (begin), preparazione, in microsecondi a strato; la
 * chiusura (finish, con l'attesa del disco) a parte.
 */
static double ex_setup, ex_t1, ex_act, ex_t2, ex_axpy, ex_begin, ex_finish,
    ex_route;
static unsigned long long ex_layers;
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
    double tot =
        ex_setup + ex_t1 + ex_act + ex_t2 + ex_axpy + ex_begin + ex_finish;
    if (tot <= 0)
        return;
    fprintf(stderr,
            "expert phase: setup %.1f%%  gate+up %.1f%%  act %.1f%%  "
            "down %.1f%%  axpy %.1f%%  cache %.1f%%  (%.0f ms, matvec "
            "%.1f%%)\n",
            100 * ex_setup / tot, 100 * ex_t1 / tot, 100 * ex_act / tot,
            100 * ex_t2 / tot, 100 * ex_axpy / tot,
            100 * (ex_begin + ex_finish) / tot, tot * 1e3,
            100 * (ex_t1 + ex_t2) / tot);
    if (ex_layers)
        fprintf(stderr,
                "router to experts, a layer: choice and order %.1f us, cache "
                "begin %.1f us, setup %.1f us; cache finish %.1f us (%llu "
                "layers)\n",
                ex_route / ex_layers * 1e6, ex_begin / ex_layers * 1e6,
                ex_setup / ex_layers * 1e6, ex_finish / ex_layers * 1e6,
                ex_layers);
    ex_setup = ex_t1 = ex_act = ex_t2 = ex_axpy = ex_begin = ex_finish = 0;
    ex_route = 0;
    ex_layers = 0;
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
    uint32_t stride = m->ff_buf;
    for (uint32_t i = (uint32_t)tid; i < a->n; i += (uint32_t)n_threads) {
        size_t g = m->glist[i];
        uint32_t ff = m->hlen[g];
        const float *gt = m->gate + g * stride, *u = m->up + g * stride;
        float *h = m->h + g * stride;
        /* deterministic operations only, so that a GPU computing an
           expert gives the same bits */
        uint32_t r = 0;
        if (m->a->nh) { /* relu^2 on up, no gate (Nemotron-H) */
            for (; r < ff; r++)
                h[r] = u[r] > 0.0f ? u[r] * u[r] : 0.0f;
        } else if (m->a->gelu) {
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
        if (m->n_obs) /* the observers, before any change */
            janas_m_obs_ffn_pair(m, g, m->a->nh ? NULL : gt, u, h, ff);
        if (m->ana_tok) /* the Ana-Net probe: by token, after every pair */
            continue;
        if (m->ana_mode || m->ana_frame) /* the Ana-Net probe */
            janas_m_ana_mask(m, g, h, ff);
        if (m->nskipq && m->nskipq[g]) /* down by neuron reads h itself */
            continue;
        /* a width no multiple of 256 (qwen4exp's 640) quantized with zeros
           after it: the down matrix's rows end inside the last group */
        uint32_t fw = (ff + JANAS_QK - 1) / JANAS_QK * JANAS_QK;
        if (fw > ff)
            memset(h + ff, 0, (fw - ff) * sizeof(float));
        janas_q8k_quantize_det(h, m->hq + g * stride / JANAS_QK, fw);
    }
}

/* Router logits of a block: item = expert e, for every token of the block,
   so the expert's weight row is read from memory once. */
struct router_job {
    struct janas_llm_model *m;
    const uint16_t *w;
    uint32_t n;
    uint32_t ne; /* the layer's experts: the model's, or fewer (Nemotron-H) */
    const float *x; /* the input (NULL: m->xn) and the scores (NULL: */
    float *out;     /* m->router), n rows of each */
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
    const float *in = r->x ? r->x : m->xn;
    float *out = r->out ? r->out : m->router;
    for (uint32_t e = (uint32_t)tid; e < r->ne; e += (uint32_t)n_threads)
        for (uint32_t j = 0; j < r->n; j++) {
            const float *x = in + (size_t)j * dm;
            const uint16_t *w = r->w + (size_t)e * dm;
#if defined(__x86_64__)
            if (m->avx2 && m->f16c) {
                out[(size_t)j * ne + e] = dot_f16_avx2(w, x, dm);
                continue;
            }
#endif
            out[(size_t)j * ne + e] = dot_f16_ref(w, x, dm);
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
    int expert; /* the routed expert's number, -1 the shared one (the Ana-Net
                   probe) */
};

/* Experts are asked of the cache and computed a chunk at a time: 128, a
   layer of Qwen3-30B-A3B at once - in chunks of 64 the GPU got half a
   layer's experts per launch, and ran them at 340 GMAC/s instead of 420
   (9 Oct 2026). The cache gives an MTP block's slots of their own room for
   one chunk. */
#define EXPERT_CHUNK JANAS_EXPERT_CHUNK
#define MAX_RUNS (EXPERT_CHUNK + 1) /* a chunk and the shared expert */

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
    /* dm: the experts' input and output width (the latent one of
       Nemotron-H's mixtures), xs their input */
    uint32_t dm = m->moe_w, ff = m->ff_buf;
    size_t nbd = dm / JANAS_QK, nbf = ff / JANAS_QK;
    const struct janas_block_q8k *xs = m->moe_xq;
    struct janas_matvec_task t1[2 * MAX_RUNS], t2[MAX_RUNS];
    struct janas_neuron_run nr[MAX_RUNS]; /* down by neuron: no product */
    uint32_t ng = 0, n1 = 0, n2 = 0, nn = 0;
    /* one run - a dense model's feed-forward, a shared expert - may go to
       the NPU whole (model_npu.c), the CPU computing the others meanwhile */
    struct janas_m_npu_ffn nf = {0};
    int nf_run = -1;
    for (uint32_t r = 0; r < n_runs; r++) {
        uint32_t a = runs[r].a, cnt = runs[r].b - runs[r].a;
        for (uint32_t g = a; g < runs[r].b; g++)
            memcpy(m->xg + (size_t)g * nbd,
                   xs + (size_t)m->pair_tok[m->order[g]] * nbd,
                   nbd * sizeof(*m->xg));
        if (nf_run < 0 && m->npu &&
            janas_m_npu_ffn_begin(m, runs[r].base, runs[r].plane, runs[r].mx,
                                  m->xg + (size_t)a * nbd, cnt,
                                  m->dout + (size_t)a * dm, &nf) == 0) {
            nf_run = (int)r;
            continue;
        }
        for (uint32_t g = a; g < runs[r].b; g++) {
            m->glist[ng++] = g;
            m->hlen[g] = runs[r].mx[JANAS_JNS_UP].rows;
            if (m->nskipq)
                m->nskipq[g] = runs[r].mx[JANAS_JNS_DOWN].type == JANAS_Q6_K_PN;
            if (m->ana_exp) /* the Ana-Net probe: the pair's expert */
                m->ana_exp[g] =
                    runs[r].expert < 0 ? m->n_expert : (uint32_t)runs[r].expert;
        }
        for (int p = 0; p < 2; p++) {
            const struct janas_jns_matrix *mx = &runs[r].mx[p];
            if (mx->type == 0)
                continue; /* no gate */
            t1[n1++] = (struct janas_matvec_task){
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
        if (d->type == JANAS_Q6_K_PN) {
            const uint8_t *w = runs[r].base[2] + d->offset;
            nr[nn++] = (struct janas_neuron_run){
                .scale = (const float *)w,
                .base =
                    (const struct janas_block_q6kp *)(w +
                                                      d->cols * sizeof(float)),
                .p1 = runs[r].plane[0],
                .p0 = runs[r].plane[1],
                .rows = d->rows,
                .cols = d->cols,
                .a = a,
                .b = runs[r].b};
            continue;
        }
        t2[n2++] = (struct janas_matvec_task){
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
    EX_T(ex_t1, janas_m_products(m, janas_m_gpu(m, 1), t1, n1));
    /* the Ana-Net probe: each neuron's agreement, from gate's and up's rows,
       and the norms of down's columns, once an expert */
    if (m->ana_low)
        janas_m_ana_lowmark(m, ng);
    for (uint32_t r = 0;
         (m->ana_agree || m->ana_coarse || m->ana_low) && r < n_runs; r++)
        janas_m_ana_agree(m, runs[r].base[0] + runs[r].mx[0].offset,
                          runs[r].base[1] + runs[r].mx[1].offset, runs[r].mx,
                          runs[r].a, runs[r].b);
    for (uint32_t r = 0; m->ana_coln && r < n_runs; r++)
        janas_m_ana_coln(
            m, runs[r].expert < 0 ? m->n_expert : (uint32_t)runs[r].expert,
            runs[r].base[2] + runs[r].mx[2].offset, runs[r].plane[0],
            runs[r].plane[1], &runs[r].mx[2]);
    struct act_job aj = {m, ng};
    EX_T(ex_act, janas_pool_run(m->compute, act_worker, &aj));
    if (m->n_obs) /* the observers: the pass's pairs done */
        janas_m_obs_ffn_done(m, ng);
    /* the NPU's calibration (JANAS_NPU_A8=1): a dense model's intermediate */
    for (uint32_t i = 0; m->npu && m->n_expert <= 1 && i < ng; i++)
        janas_m_npu_calib(m, 2, m->ana_layer, m->h + (size_t)m->glist[i] * ff,
                          1, m->hlen[m->glist[i]], ff);
    if (m->ana_tok) /* the Ana-Net probe: a token's neurons ranked together */
        janas_m_ana_tok(m, ng);
    if (nn && m->neuron_rank)
        janas_m_neuron_levels(m, ng);
    EX_T(ex_t2, janas_m_neuron_down(m, nr, nn));
    if (n2)
        EX_T(ex_t2, janas_m_products(m, janas_m_gpu(m, 1), t2, n2));
    /* the Ana-Net probe: neurons kept at fewer bits, their down corrected */
    for (uint32_t r = 0; m->ana_level && r < n_runs; r++)
        janas_m_ana_grade(m, runs[r].base[2] + runs[r].mx[2].offset,
                          runs[r].plane[0], runs[r].plane[1], &runs[r].mx[2],
                          runs[r].a, runs[r].b);
    if (nf_run >= 0 && janas_m_npu_ffn_finish(m, &nf) != 0) {
        /* the NPU failed: that run on the CPU after all */
        int use = m->npu_use;
        m->npu_use = 0;
        run_experts(m, &runs[nf_run], 1);
        m->npu_use = use;
    }
}

int janas_m_expert_rows(struct janas_llm_model *m, uint32_t l, uint32_t n,
                        const float *xn, uint32_t nd, const uint32_t *ids,
                        const uint8_t *const *slots, const uint32_t *ra,
                        const uint32_t *rb, const uint32_t *tok, float *out)
{
    uint32_t dm = m->d_model, nbd = dm / JANAS_QK;
    if (n > m->blk)
        return -1;
    /* as the GPU quantizes the same values (norm_quant.comp, as the CPU's
       own path): then an expert computed here gives the GPU's bits */
    for (uint32_t j = 0; j < n; j++)
        janas_q8k_quantize(xn + (size_t)j * dm, m->xq + j * nbd, dm);
    const struct janas_jns_layer *jl = &m->j.layers[l];
    /* the CPU's products only: the GPU waits for these between two of its
       submissions */
    int off = m->gpu_off;
    m->gpu_off = 1;
    for (uint32_t c0 = 0; c0 < nd; c0 += EXPERT_CHUNK) {
        struct expert_run runs[EXPERT_CHUNK];
        uint32_t cn = nd - c0 < EXPERT_CHUNK ? nd - c0 : EXPERT_CHUNK;
        for (uint32_t c = 0; c < cn; c++) {
            uint32_t i = c0 + c;
            if (rb[i] > m->blk * (m->k + 1)) {
                m->gpu_off = off;
                return -1;
            }
            for (uint32_t r = ra[i]; r < rb[i]; r++)
                m->pair_tok[r] = tok[r], m->order[r] = r;
            const uint8_t *pl[2] = {NULL, NULL};
            for (int q = 0; q < m->exp_level - 1; q++)
                if (jl->plane[q].type)
                    pl[q] = slots[i] + jl->plane[q].offset;
            runs[c] = (struct expert_run){{slots[i], slots[i], slots[i]},
                                          {pl[0], pl[1]},
                                          jl->m,
                                          ra[i],
                                          rb[i],
                                          (int)ids[i]};
        }
        run_experts(m, runs, cn);
    }
    m->gpu_off = off;
    for (uint32_t i = 0; i < nd; i++)
        memcpy(out + (size_t)ra[i] * dm, m->dout + (size_t)ra[i] * dm,
               (size_t)(rb[i] - ra[i]) * dm * sizeof(float));
    return 0;
}

int janas_m_route_rank(struct janas_llm_model *m, uint32_t l, uint32_t n,
                       uint32_t k, uint32_t *ids)
{
    const struct layer *ly = &m->layers[l];
    uint32_t ne = m->n_expert;
    if (!ly->router16 || !ly->ffn_norm || k == 0 || k > 32 || k > ne)
        return -1;
    /* xn and router are free between layers: l + 1 norms into xn first */
    janas_m_norm_quant(m, f32(m, ly->ffn_norm), 0, n);
    struct router_job rj = {.m = m, .w = ly->router16, .n = n, .ne = ne};
    janas_pool_run(m->compute, router_worker, &rj); /* 512 x 2048 a layer */
    for (uint32_t j = 0; j < n; j++) {
        float *r = m->router + (size_t)j * ne;
        for (uint32_t i = 0; i < k; i++) {
            uint32_t best = 0;
            for (uint32_t e = 1; e < ne; e++)
                if (r[e] > r[best])
                    best = e;
            ids[(size_t)j * k + i] = best;
            r[best] = -INFINITY;
        }
    }
    return 0;
}

int janas_m_rank_ahead(struct janas_llm_model *m, uint32_t l, const float *xn,
                       uint32_t k, uint32_t *ids)
{
    uint32_t ne = m->n_expert, dm = m->d_model;
    if (l + 1 >= m->n_layer || k == 0 || k > 32 || k > ne)
        return -1;
    const struct layer *a = &m->layers[l], *b = &m->layers[l + 1];
    if (!a->ffn_norm || !b->ffn_norm || !b->router16)
        return -1;
    if (!m->ahead_x) { /* its own: the GPU's pass may hold xn and router */
        m->ahead_x = malloc(((size_t)dm + ne) * sizeof(float));
        if (!m->ahead_x)
            return -1;
    }
    /* layer l's norm weights undone, l + 1's applied: the same state */
    const float *wa = f32(m, a->ffn_norm), *wb = f32(m, b->ffn_norm);
    float *x = m->ahead_x, *r = m->ahead_x + dm;
    for (uint32_t i = 0; i < dm; i++)
        x[i] = fabsf(wa[i]) > 1e-12f ? xn[i] / wa[i] * wb[i] : 0.0f;
    struct router_job rj = {
        .m = m, .w = b->router16, .n = 1, .ne = ne, .x = x, .out = r};
    janas_pool_run(m->compute, router_worker, &rj);
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best = 0;
        for (uint32_t e = 1; e < ne; e++)
            if (r[e] > r[best])
                best = e;
        ids[i] = best;
        r[best] = -INFINITY;
    }
    return 0;
}

int janas_m_route_predict(struct janas_llm_model *m, uint32_t l, uint32_t n)
{
    uint32_t k = m->pred_k;
    uint32_t *ids = malloc((size_t)n * (k ? k : 1) * sizeof(*ids));
    if (!ids)
        return -1;
    if (janas_m_route_rank(m, l, n, k, ids) == 0)
        m->pred_fn(m->pred_ctx, l, n, k, ids);
    free(ids);
    return 0;
}

/*
 * Reads ahead the experts layer l + 1 is likely to want, of a single token:
 * its router applied to the state before layer l's experts - the time of
 * those experts and of layer l + 1's attention to read them in, some 0.8 ms
 * or more on Qwen3-Next-80B, against 0.6 for a read. On that model its
 * first 12 held 91% of the experts chosen and 85-88% of those the cache
 * would miss (30 Sep 2026). Only the likeliest, and at most pf_max: past
 * them the reads are mostly wasted and the disk fills up.
 */
void janas_m_prefetch(struct janas_llm_model *m, uint32_t l)
{
    double t0 = now();
    uint32_t ids[32];
    if (janas_m_route_rank(m, l + 1, 1, m->pf_rank, ids) == 0)
        janas_expert_cache_prefetch(m->cache, l + 1, ids, m->pf_rank,
                                    m->pf_max);
    m->phase[JANAS_PH_ROUTER] += now() - t0;
}

static int by_votes(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x < y) - (x > y); /* descending */
}

/*
 * Reads ahead for layer l + 1 of a prefill block, as Strata does for its
 * graphics card (github.com/Niko1221/Strata): layer l + 1's router on the
 * whole block, before layer l's experts, and the experts most of its tokens
 * would choose read first, while layer l's experts are computed. Without
 * it a layer whose experts are not in the cache waits for the disk chunk by
 * chunk, with only the experts already in to compute meanwhile.
 */
void janas_m_prefetch_block(struct janas_llm_model *m, uint32_t l, uint32_t n)
{
    double t0 = now();
    uint32_t k = m->k_use, ne = m->n_expert;
    uint32_t *ids = malloc((size_t)n * k * sizeof(*ids));
    if (ids && janas_m_route_rank(m, l + 1, n, k, ids) == 0) {
        uint32_t votes[4096] = {0}; /* n_expert <= 4096, checked at load */
        for (size_t p = 0; p < (size_t)n * k; p++)
            votes[ids[p]]++;
        uint64_t key[4096];
        uint32_t nk = 0;
        for (uint32_t e = 0; e < ne; e++)
            if (votes[e])
                key[nk++] = (uint64_t)votes[e] << 32 | e;
        qsort(key, nk, sizeof(*key), by_votes);
        for (uint32_t i = 0; i < nk; i++)
            ids[i] = (uint32_t)key[i];
        janas_expert_cache_prefetch(m->cache, l + 1, ids, nk, nk);
    }
    free(ids);
    m->phase[JANAS_PH_ROUTER] += now() - t0;
}

/* Nemotron-H's choice for token j: sigmoid scores, the k best by score
   plus the layer's bias, their scores as weights, normalized to one (at
   least 2^-14 for the sum, as llama.cpp), times the model's scale. */
static void route_sigmoid(struct janas_llm_model *m, const struct layer *ly,
                          uint32_t j)
{
    uint32_t ne = ly->n_exp, k = m->k_use;
    float *r = m->router + (size_t)j * m->n_expert, sel[4096];
    for (uint32_t e = 0; e < ne; e++) {
        r[e] = sigmoid(r[e]);
        sel[e] = r[e] + ly->exp_bias[e];
    }
    float wsum = 0.0f;
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best = 0;
        float bv = -INFINITY;
        for (uint32_t e = 0; e < ne; e++)
            if (sel[e] > bv) {
                bv = sel[e];
                best = e;
            }
        sel[best] = -INFINITY; /* taken */
        uint32_t p = j * k + i;
        m->pair_tok[p] = j;
        m->pair_exp[p] = best;
        m->pair_w[p] = r[best];
        wsum += r[best];
    }
    if (wsum < 6.103515625e-5f)
        wsum = 6.103515625e-5f;
    for (uint32_t i = 0; i < k; i++)
        m->pair_w[j * k + i] = m->pair_w[j * k + i] / wsum * m->route_scale;
}

/* The router's choice for token j: softmax, top-k, normalized weights. */
static void route_tok(struct janas_llm_model *m, const struct layer *ly,
                      uint32_t j)
{
    if (ly && ly->exp_bias) {
        route_sigmoid(m, ly, j);
        return;
    }
    uint32_t ne = m->n_expert, k = m->k_use;
    float *r = m->router + (size_t)j * ne, mx = -INFINITY, sum = 0.0f;
    for (uint32_t e = 0; e < ne; e++)
        if (r[e] > mx)
            mx = r[e];
    for (uint32_t e = 0; e < ne; e++) {
        r[e] = janas_expf(r[e] - mx);
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
        route_tok(t->m, t->ly, j);
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
    uint32_t w = m->moe_w;
    for (uint32_t j = (uint32_t)tid; j < a->n; j += (uint32_t)n_threads)
        for (uint32_t c = a->first[j]; c < a->first[j + 1]; c++) {
            uint32_t g = a->list[c];
            janas_axpy_f32(m->moe_out + (size_t)j * w, m->pair_w[m->order[g]],
                           m->dout + (size_t)g * w, w);
        }
}

/*
 * Cache-conditional routing (Skliar et al., "Mixture of Cache-Conditional
 * Experts", TMLR 2025): an expert that is not in RAM and that every token
 * of the block weighs below m->miss_skip is left out instead of read, and
 * each token's other weights are renormalized, as with fewer experts per
 * token. Unlike fewer experts per token, which drops the lightest whether
 * they are in RAM or not, only what would cost a read goes. Simulated on
 * Qwen3-Next-80B's routing over seven kinds of text (8 Oct 2026): below the
 * lightest quarter of the weights (0.06 of ten), a third fewer experts read
 * with a cache of 10-30% of them, for 1-4% of what the experts add.
 * Returns the pairs left, compacted in place, token by token.
 */
static uint32_t skip_misses(struct janas_llm_model *m,
                            struct janas_expert_cache *cache, uint32_t l,
                            uint32_t n, uint32_t k)
{
    uint32_t np = n * k;
    float th = m->miss_skip, wmax[4096];
    uint8_t mark[4096]; /* 1: light, asked of the cache; 2: light and out */
    for (uint32_t p = 0; p < np; p++) {
        wmax[m->pair_exp[p]] = 0.0f;
        mark[m->pair_exp[p]] = 0;
    }
    /* a weight against its token's sum: the router's may sum to a scale
       (Nemotron-H's 5.0), not to one */
    float tot = 0.0f;
    for (uint32_t p = 0; p < np; p++) {
        if (p % k == 0) {
            tot = 0.0f;
            for (uint32_t i = 0; i < k; i++)
                tot += m->pair_w[p + i];
        }
        float w = tot > 0.0f ? m->pair_w[p] / tot : 0.0f;
        if (w > wmax[m->pair_exp[p]])
            wmax[m->pair_exp[p]] = w;
    }
    uint32_t light[256];
    uint8_t in[256];
    size_t nl = 0;
    for (uint32_t p = 0; p <= np; p++) {
        if (p < np) {
            uint32_t e = m->pair_exp[p];
            if (wmax[e] >= th || mark[e])
                continue;
            mark[e] = 1;
            light[nl++] = e;
        }
        if (nl && (nl == 256 || p == np)) {
            janas_expert_cache_present(cache, l, light, nl, in);
            for (size_t i = 0; i < nl; i++)
                if (!in[i])
                    mark[light[i]] = 2;
            nl = 0;
        }
    }
    uint32_t q = 0;
    for (uint32_t j = 0; j < n; j++) {
        float kept = 0.0f, all = 0.0f;
        for (uint32_t i = 0; i < k; i++) {
            all += m->pair_w[j * k + i];
            if (mark[m->pair_exp[j * k + i]] != 2)
                kept += m->pair_w[j * k + i];
        }
        /* the weights left brought back to the sum the router gave */
        for (uint32_t i = 0; i < k; i++) {
            uint32_t p = j * k + i;
            if (mark[m->pair_exp[p]] == 2)
                continue;
            m->pair_tok[q] = j;
            m->pair_exp[q] = m->pair_exp[p];
            m->pair_w[q] = m->pair_w[p] / kept * all;
            q++;
        }
    }
    m->skip_asked += np;
    m->skip_left += q;
    return q;
}

/* Routed (and shared) experts of a block, added to x. The routed experts
   are layer l of the file behind the cache (the main model or the MTP). */
int janas_m_moe_block(struct janas_llm_model *m, const struct layer *ly,
                      struct janas_expert_cache *cache,
                      const struct janas_jns_layer *jl, uint32_t l, uint32_t n,
                      int part)
{
    double t0 = now();
    uint32_t dm = m->d_model, ne = m->n_expert;
    uint32_t k = part == MOE_SHARED ? 0 : m->k_use;
    /* the experts' input and where their weighted outputs go: the block
       and x, or (Nemotron-H) the block brought down to the latent width
       and a sum there, brought up and added to x at the end */
    m->moe_w = dm;
    m->moe_xq = m->xq;
    m->moe_out = m->x;
    if (ly->lat_down) {
        uint32_t L = m->moe_lat;
        struct janas_matvec_task t = {.type = (int)ly->lat_down->type,
                                      .w = data(m, ly->lat_down),
                                      .x = m->xq,
                                      .y = m->lat,
                                      .rows = L,
                                      .cols = dm,
                                      .n_vec = n};
        janas_m_products(m, janas_m_gpu(m, 1), &t, 1);
        for (uint32_t j = 0; j < n; j++)
            janas_q8k_quantize(m->lat + (size_t)j * L,
                               m->latq + (size_t)j * (L / JANAS_QK), L);
        memset(m->lat_acc, 0, (size_t)n * L * sizeof(float));
        m->moe_w = L;
        m->moe_xq = m->latq;
        m->moe_out = m->lat_acc;
    }
    uint32_t np = n * k;
    double t_route = 0;
    if (k) {
        struct router_job rj = {.m = m,
                                .w = ly->router16,
                                .n = n,
                                .ne = ly->n_exp ? ly->n_exp : m->n_expert};
        janas_pool_run(m->compute, router_worker, &rj);
        if (janas_m_exp_tracing())
            t_route = now();

        /* per token: softmax, top-k, normalized weights */
        struct tok_job tj = {.m = m, .ly = ly, .n = n};
        if (n < 2)
            route_tok(m, ly, 0);
        else
            janas_pool_run(m->compute, route_worker, &tj);
        /* an expert's own output scale, folded into its weight (Gemma 4) */
        if (ly->exp_scale)
            for (uint32_t p = 0; p < n * k; p++)
                m->pair_w[p] *= ly->exp_scale[m->pair_exp[p]];
        if (m->route_fn && cache == m->cache)
            m->route_fn(m->route_ctx, l, n, k, m->pair_exp);
        /* the weights must be the router's own, the same sum each token */
        if (m->miss_skip > 0 && cache && !ly->exp_scale)
            np = skip_misses(m, cache, l, n, k);
    }
    /* counting sort of the pairs by expert */
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
    if (ly->sh_w[0] && part != MOE_ROUTED) {
        for (uint32_t j = 0; j < n; j++, nt++) {
            m->pair_tok[nt] = j;
            /* its gate, or none: a branch of its own (Gemma 4) */
            m->pair_w[nt] =
                ly->sh_gate ? sigmoid(janas_dot_f32(ly->sh_gate,
                                                    m->xn + (size_t)j * dm, dm))
                            : 1.0f;
            m->order[nt] = nt;
        }
    }
    double t1 = now();
    m->phase[JANAS_PH_ROUTER] += t1 - t0;
    if (t_route > 0) {
        ex_route += t1 - t_route;
        ex_layers++;
    }

    /* experts in chunks: compute those in RAM while the others load; the
       shared expert, resident, goes with the first of them */
    /* a request is at most as many experts as the cache has slots (a
       small one: fewer than EXPERT_CHUNK) */
    uint32_t chunk = EXPERT_CHUNK;
    if (cache && janas_expert_cache_slots(cache) < chunk)
        chunk = (uint32_t)janas_expert_cache_slots(cache);
    if (!chunk)
        return -1;
    for (uint32_t c0 = 0; c0 < nd || (c0 == 0 && nt > np); c0 += chunk) {
        uint32_t cn = nd - c0 < chunk ? nd - c0 : chunk;
        const uint8_t *slots[EXPERT_CHUNK];
        uint8_t ready[EXPERT_CHUNK];
        if (cn) {
            int rc_b;
            EX_T(ex_begin, rc_b = janas_expert_cache_begin(
                               cache, l, distinct + c0, cn, slots, ready));
            if (rc_b < 0)
                return -1;
        }
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 1 && cn) {
                int rc_f;
                EX_T(ex_finish, rc_f = janas_expert_cache_finish(cache));
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
                                        nt,
                                        -1};
            for (uint32_t i = 0; i < cn; i++)
                if ((pass == 0) == (ready[i] != 0)) {
                    uint32_t e = distinct[c0 + i];
                    if (m->n_obs) /* where it came from, for a monitor */
                        janas_m_obs_route(m, l, e, ready[i] != 0);
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
                                            m->count[e + 1],
                                            (int)e};
                }
            if (nr)
                run_experts(m, runs, nr);
        }
        if (nd <= c0 + chunk)
            break;
    }
    if (n < 2) {
        EX_T(
            ex_axpy, for (uint32_t g = 0; g < nt; g++) {
                uint32_t p = m->order[g];
                janas_axpy_f32(m->moe_out + (size_t)m->pair_tok[p] * m->moe_w,
                               m->pair_w[p], m->dout + (size_t)g * m->moe_w,
                               m->moe_w);
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
    if (ly->lat_down) {
        uint32_t L = m->moe_lat;
        for (uint32_t j = 0; j < n; j++)
            janas_q8k_quantize(m->lat_acc + (size_t)j * L,
                               m->latq + (size_t)j * (L / JANAS_QK), L);
        struct janas_matvec_task t = {.type = (int)ly->lat_up->type,
                                      .w = data(m, ly->lat_up),
                                      .x = m->latq,
                                      .y = m->xn,
                                      .rows = dm,
                                      .cols = L,
                                      .n_vec = n};
        janas_m_products(m, janas_m_gpu(m, 1), &t, 1);
        for (size_t i = 0; i < (size_t)n * dm; i++)
            m->x[i] += m->xn[i];
        m->moe_w = dm;
        m->moe_xq = m->xq;
        m->moe_out = m->x;
    }
    m->phase[JANAS_PH_EXPERTS] += now() - t1;
    return 0;
}
