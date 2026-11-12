/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_monitor.c - the monitor's look inside the layers (level 2 of
 * janas-server --monitor and janas-chat's /monitor): an observer of the
 * forward pass (observe.h), attached only while somebody watches, that
 * gathers for each layer
 *   - the mean |activation| of the feed-forward's neurons, in bins (the
 *     dense feed-forward, or a mixture's shared expert),
 *   - the mean |value| of the residual after the layer, in bins, at the
 *     last token of each pass,
 *   - for a mixture, each routed expert: how many times chosen, how many
 *     times read from the file rather than found in memory, and the mean
 *     |activation| of its neurons when chosen.
 * Taken by the reader as sums since the last take, then started anew.
 *
 * The pair rows run on the compute pool's threads at once, so each writes
 * its own bins (one row of rowbins); the pass's rows are summed once they
 * are done, on the forward's thread, under the lock the reader takes too.
 */
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "model_impl.h"

#define WATCH_BINS 32

struct janas_mwatch {
    pthread_mutex_t mu;
    uint32_t n_layer, n_exp;
    size_t rows_cap;
    float *rowbins; /* rows_cap * WATCH_BINS: a pair row's bins */
    float *rowmean; /* rows_cap: its mean */
    uint32_t *rowexp;
    /* the sums since the last take */
    double *ffn;     /* n_layer * WATCH_BINS */
    uint64_t *ffn_n; /* n_layer: rows summed */
    double *res;     /* n_layer * WATCH_BINS */
    uint64_t *res_n;
    uint32_t *chosen, *read; /* n_layer * n_exp */
    double *strength;
    uint64_t passes;
    /* one forward call in every `every` is looked at (JANAS_MONITOR_EVERY,
       default 1: all) */
    uint32_t every, count;
    int sample;
};

/* bins of |v| over n values: bin b the mean of its share of them */
static void binned(const float *v, uint32_t n, float out[WATCH_BINS],
                   float *mean)
{
    double all = 0;
    for (uint32_t b = 0; b < WATCH_BINS; b++) {
        uint32_t i0 = (uint32_t)((uint64_t)n * b / WATCH_BINS);
        uint32_t i1 = (uint32_t)((uint64_t)n * (b + 1) / WATCH_BINS);
        double s = 0;
        for (uint32_t i = i0; i < i1; i++)
            s += v[i] < 0 ? -v[i] : v[i];
        all += s;
        out[b] = i1 > i0 ? (float)(s / (i1 - i0)) : 0.f;
    }
    if (mean)
        *mean = n ? (float)(all / n) : 0.f;
}

static void w_ffn_input(void *ctx, struct janas_llm_model *m, uint32_t l,
                        const float *xn, uint32_t n, uint32_t d)
{
    (void)m, (void)xn, (void)n, (void)d;
    struct janas_mwatch *w = ctx;
    if (l == 0) /* a forward call starts: looked at or not */
        w->sample = w->count++ % w->every == 0;
}

static void w_ffn_pair(void *ctx, struct janas_llm_model *m, uint32_t l,
                       uint32_t row, uint32_t tok, uint32_t expert, float wt,
                       const float *gate, const float *up, const float *h,
                       uint32_t ff)
{
    (void)m, (void)l, (void)tok, (void)wt, (void)gate, (void)up;
    struct janas_mwatch *w = ctx;
    if (!w->sample || row >= w->rows_cap)
        return;
    binned(h, ff, w->rowbins + (size_t)row * WATCH_BINS, &w->rowmean[row]);
    w->rowexp[row] = expert;
}

static void w_ffn_done(void *ctx, struct janas_llm_model *m, uint32_t l,
                       const uint32_t *rows, uint32_t n_rows)
{
    (void)m;
    struct janas_mwatch *w = ctx;
    if (!w->sample || l >= w->n_layer)
        return;
    pthread_mutex_lock(&w->mu);
    for (uint32_t i = 0; i < n_rows; i++) {
        uint32_t r = rows[i];
        if (r >= w->rows_cap)
            continue;
        uint32_t e = w->rowexp[r];
        if (e >= w->n_exp) { /* the dense feed-forward, or the shared one */
            const float *b = w->rowbins + (size_t)r * WATCH_BINS;
            double *acc = w->ffn + (size_t)l * WATCH_BINS;
            for (uint32_t k = 0; k < WATCH_BINS; k++)
                acc[k] += b[k];
            w->ffn_n[l]++;
        } else {
            w->chosen[(size_t)l * w->n_exp + e]++;
            w->strength[(size_t)l * w->n_exp + e] += w->rowmean[r];
        }
    }
    pthread_mutex_unlock(&w->mu);
}

static int w_layer_out(void *ctx, struct janas_llm_model *m, uint32_t l,
                       const float *x, uint32_t n, uint32_t d)
{
    (void)m;
    struct janas_mwatch *w = ctx;
    if (!w->sample || l >= w->n_layer || n == 0)
        return 0;
    float b[WATCH_BINS];
    binned(x + (size_t)(n - 1) * d, d, b, NULL);
    pthread_mutex_lock(&w->mu);
    double *acc = w->res + (size_t)l * WATCH_BINS;
    for (uint32_t k = 0; k < WATCH_BINS; k++)
        acc[k] += b[k];
    w->res_n[l]++;
    if (l == w->n_layer - 1)
        w->passes++;
    pthread_mutex_unlock(&w->mu);
    return 0;
}

static void w_route(void *ctx, struct janas_llm_model *m, uint32_t l,
                    uint32_t e, int in_memory)
{
    (void)m;
    struct janas_mwatch *w = ctx;
    if (!w->sample || in_memory || l >= w->n_layer || e >= w->n_exp)
        return;
    pthread_mutex_lock(&w->mu);
    w->read[(size_t)l * w->n_exp + e]++;
    pthread_mutex_unlock(&w->mu);
}

static void w_free(struct janas_mwatch *w)
{
    if (!w)
        return;
    pthread_mutex_destroy(&w->mu);
    free(w->rowbins);
    free(w->rowmean);
    free(w->rowexp);
    free(w->ffn);
    free(w->ffn_n);
    free(w->res);
    free(w->res_n);
    free(w->chosen);
    free(w->read);
    free(w->strength);
    free(w);
}

int janas_llm_model_watch(struct janas_llm_model *m, int on)
{
    if (!on) {
        if (m->watch) {
            janas_llm_model_unobserve(m, m->watch);
            w_free(m->watch);
            m->watch = NULL;
        }
        return 0;
    }
    if (m->watch)
        return 0;
    struct janas_mwatch *w = calloc(1, sizeof *w);
    if (!w || pthread_mutex_init(&w->mu, NULL) != 0) {
        free(w);
        return -1;
    }
    w->n_layer = m->n_layer;
    /* a dense model's feed-forward is one expert a layer to the cache: dense
       here */
    w->n_exp = m->n_expert > 1 ? m->n_expert : 0;
    w->rows_cap = m->pairs_cap;
    const char *ev = getenv("JANAS_MONITOR_EVERY");
    w->every = ev && atoi(ev) > 0 ? (uint32_t)atoi(ev) : 1;
    size_t L = w->n_layer, E = w->n_exp ? w->n_exp : 1;
    w->rowbins =
        calloc(w->rows_cap ? w->rows_cap * WATCH_BINS : 1, sizeof(float));
    w->rowmean = calloc(w->rows_cap ? w->rows_cap : 1, sizeof(float));
    w->rowexp = calloc(w->rows_cap ? w->rows_cap : 1, sizeof(uint32_t));
    w->ffn = calloc(L * WATCH_BINS, sizeof(double));
    w->ffn_n = calloc(L, sizeof(uint64_t));
    w->res = calloc(L * WATCH_BINS, sizeof(double));
    w->res_n = calloc(L, sizeof(uint64_t));
    w->chosen = calloc(L * E, sizeof(uint32_t));
    w->read = calloc(L * E, sizeof(uint32_t));
    w->strength = calloc(L * E, sizeof(double));
    struct janas_observer o = {.ctx = w,
                               .ffn_input = w_ffn_input,
                               .ffn_pair = w_ffn_pair,
                               .ffn_done = w_ffn_done,
                               .layer_out = w_layer_out,
                               .route = w_route};
    if (!w->rowbins || !w->rowmean || !w->rowexp || !w->ffn || !w->ffn_n ||
        !w->res || !w->res_n || !w->chosen || !w->read || !w->strength ||
        janas_llm_model_observe(m, &o) != 0) {
        w_free(w);
        return -1;
    }
    m->watch = w;
    return 0;
}

void janas_llm_model_watch_shape(const struct janas_llm_model *m,
                                 uint32_t shape[3])
{
    shape[0] = m->n_layer;
    shape[1] = WATCH_BINS;
    shape[2] = m->n_expert > 1 ? m->n_expert : 0;
}

int janas_llm_model_watch_take(struct janas_llm_model *m, float *ffn,
                               float *res, uint32_t *chosen, uint32_t *read,
                               float *strength, uint64_t *passes)
{
    struct janas_mwatch *w = m->watch;
    if (!w)
        return -1;
    size_t L = w->n_layer, E = w->n_exp;
    pthread_mutex_lock(&w->mu);
    for (size_t l = 0; l < L; l++)
        for (size_t k = 0; k < WATCH_BINS; k++) {
            size_t i = l * WATCH_BINS + k;
            if (ffn)
                ffn[i] = w->ffn_n[l] ? (float)(w->ffn[i] / w->ffn_n[l]) : 0.f;
            if (res)
                res[i] = w->res_n[l] ? (float)(w->res[i] / w->res_n[l]) : 0.f;
        }
    for (size_t i = 0; i < L * E; i++) {
        if (chosen)
            chosen[i] = w->chosen[i];
        if (read)
            read[i] = w->read[i];
        if (strength)
            strength[i] =
                w->chosen[i] ? (float)(w->strength[i] / w->chosen[i]) : 0.f;
    }
    if (passes)
        *passes = w->passes;
    memset(w->ffn, 0, L * WATCH_BINS * sizeof(double));
    memset(w->ffn_n, 0, L * sizeof(uint64_t));
    memset(w->res, 0, L * WATCH_BINS * sizeof(double));
    memset(w->res_n, 0, L * sizeof(uint64_t));
    if (E) {
        memset(w->chosen, 0, L * E * sizeof(uint32_t));
        memset(w->read, 0, L * E * sizeof(uint32_t));
        memset(w->strength, 0, L * E * sizeof(double));
    }
    w->passes = 0;
    pthread_mutex_unlock(&w->mu);
    return 0;
}
