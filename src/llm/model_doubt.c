/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_doubt.c - the doubt probe (job/docs/progetto_nucleus_cancello.md):
 * what a model's feed-forward neurons say, read as states (open, not sure,
 * closed; yes or no), without changing
 * anything the model computes. A probe: only the CPU's path has it
 * (JANAS_GPU=0), and it records only while the caller sets m->dbt_on.
 *
 * JANAS_DOUBT_STATS=file: for every pair (a token through one expert, or
 * through the dense feed-forward) of every layer a record of five 32-bit
 * words - 1, its layer, its expert (n_expert: the shared one or the dense
 * feed-forward), its token in the pass, its weight w as a float - then
 * DBT_NF floats from its gate g, up u and h = act(g) u:
 *   0-2 the share of neurons open (g > 1), not sure (-1 < g <= 1), closed;
 *   3   among the open, the share saying yes (h > 0);
 *   4   |h|, 5 its participation ratio (sum |h|)^2 / (ff sum h^2);
 *   6   the share of sum |h| from the not sure, 7 from the open saying no;
 *   8-9 the mean and the deviation of g.
 * JANAS_DOUBT_X=l1,l2,...: for those layers each token's feed-forward input
 * (the normed state) too, a record 2, its layer, d_model, its token, 0, then
 * d_model floats.
 * JANAS_DOUBT_RES=1: after every layer the residual of the pass's last
 * token, a record 6, its layer, d_model, its token, 0, then d_model floats;
 * kept too in m->dbt_rbuf, for a caller that reads them (logit lens).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_impl.h"

static void dbt_pair(void *ctx, struct janas_llm_model *m, uint32_t l,
                     uint32_t g, uint32_t tok, uint32_t expert, float w,
                     const float *gt, const float *u, const float *h,
                     uint32_t ff);
static void dbt_write(void *ctx, struct janas_llm_model *m, uint32_t l,
                      const uint32_t *rows, uint32_t ng);
static void dbt_x(void *ctx, struct janas_llm_model *m, uint32_t l,
                  const float *xn, uint32_t n, uint32_t d);
static int dbt_res(void *ctx, struct janas_llm_model *m, uint32_t l,
                   const float *x, uint32_t n, uint32_t d);

int janas_m_dbt_init(struct janas_llm_model *m, size_t pairs)
{
    const char *e = getenv("JANAS_DOUBT_STATS");
    if (!e || !*e)
        return 0;
    if (!(m->dbt_file = fopen(e, "wb")))
        return -1;
    m->dbt_feat = calloc(pairs * DBT_NF, sizeof(float));
    if (!m->dbt_feat)
        return -1;
    e = getenv("JANAS_DOUBT_RES");
    m->dbt_res = e && atoi(e);
    if (m->dbt_res &&
        !(m->dbt_rbuf = calloc((size_t)m->n_layer * m->d_model, sizeof(float))))
        return -1;
    e = getenv("JANAS_DOUBT_X");
    if (e && *e) {
        if (!(m->dbt_x = calloc(m->n_layer, 1)))
            return -1;
        char buf[512];
        snprintf(buf, sizeof(buf), "%s", e);
        for (char *t = strtok(buf, ","); t; t = strtok(NULL, ","))
            if (atoi(t) >= 0 && (uint32_t)atoi(t) < m->n_layer)
                m->dbt_x[atoi(t)] = 1;
    }
    struct janas_observer o = {.ctx = &m->dbt_file,
                               .ffn_input = dbt_x,
                               .ffn_pair = dbt_pair,
                               .ffn_done = dbt_write,
                               .layer_out = dbt_res};
    if (janas_llm_model_observe(m, &o) != 0)
        return -1;
    fprintf(stderr,
            "doubt: the neurons' states to %s (a probe: the CPU's "
            "path only)\n",
            getenv("JANAS_DOUBT_STATS"));
    return 0;
}

void janas_m_dbt_free(struct janas_llm_model *m)
{
    if (m->dbt_file)
        fclose(m->dbt_file);
    m->dbt_file = NULL;
    free(m->dbt_feat);
    free(m->dbt_x);
    free(m->dbt_rbuf);
    m->dbt_rbuf = NULL;
    m->dbt_feat = NULL;
    m->dbt_x = NULL;
}

/* Pair row g's features from its gate, up and h (act_worker's thread). */
static void dbt_pair(void *ctx, struct janas_llm_model *m, uint32_t l,
                     uint32_t g, uint32_t tok, uint32_t expert, float w,
                     const float *gt, const float *u, const float *h,
                     uint32_t ff)
{
    (void)ctx, (void)l, (void)tok, (void)expert, (void)w, (void)u;
    if (!m->dbt_on)
        return;
    float *f = m->dbt_feat + g * DBT_NF;
    double open = 0, unsure = 0, yes = 0, s1 = 0, s2 = 0, mu = 0, mno = 0;
    double sg = 0, sg2 = 0;
    for (uint32_t j = 0; j < ff; j++) {
        double a = fabs((double)h[j]), x = gt ? gt[j] : 2.0;
        s1 += a, s2 += a * a;
        sg += x, sg2 += x * x;
        if (x > 1) {
            open++;
            if (h[j] > 0)
                yes++;
            else
                mno += a;
        } else if (x > -1) {
            unsure++;
            mu += a;
        }
    }
    double n = ff ? ff : 1, mg = sg / n;
    f[0] = (float)(open / n);
    f[1] = (float)(unsure / n);
    f[2] = (float)((n - open - unsure) / n);
    f[3] = (float)(open > 0 ? yes / open : 0.5);
    f[4] = (float)sqrt(s2);
    f[5] = (float)(s2 > 0 ? s1 * s1 / (n * s2) : 0);
    f[6] = (float)(s1 > 0 ? mu / s1 : 0);
    f[7] = (float)(s1 > 0 ? mno / s1 : 0);
    f[8] = (float)mg;
    f[9] = (float)sqrt(fmax(sg2 / n - mg * mg, 0));
}

/* The records of the pair rows of this pass. */
static void dbt_write(void *ctx, struct janas_llm_model *m, uint32_t l,
                      const uint32_t *rows, uint32_t ng)
{
    (void)ctx, (void)l;
    if (!m->dbt_on)
        return;
    for (uint32_t i = 0; i < ng; i++) {
        size_t g = rows[i];
        float w = m->pair_w[m->order[g]];
        uint32_t head[5] = {1, m->ana_layer, m->ana_exp[g],
                            m->pair_tok[m->order[g]], 0};
        memcpy(&head[4], &w, sizeof(w));
        fwrite(head, sizeof(head), 1, m->dbt_file);
        fwrite(m->dbt_feat + g * DBT_NF, sizeof(float), DBT_NF, m->dbt_file);
    }
}

/* Layer l's feed-forward inputs of the pass's n tokens, if listed. */
static void dbt_x(void *ctx, struct janas_llm_model *m, uint32_t l,
                  const float *xn, uint32_t n, uint32_t d)
{
    (void)ctx;
    if (!m->dbt_on || !m->dbt_x || !m->dbt_x[l])
        return;
    for (uint32_t t = 0; t < n; t++) {
        uint32_t head[5] = {2, l, d, t, 0};
        fwrite(head, sizeof(head), 1, m->dbt_file);
        fwrite(xn + (size_t)t * d, sizeof(float), d, m->dbt_file);
    }
}

/* The residual after layer l of the pass's last token, if asked; the
   caller's stop (m->dbt_cut) after it. */
static int dbt_res(void *ctx, struct janas_llm_model *m, uint32_t l,
                   const float *x, uint32_t n, uint32_t d)
{
    (void)ctx;
    if (m->dbt_on && m->dbt_res && n) {
        const float *r = x + (size_t)(n - 1) * d;
        memcpy(m->dbt_rbuf + (size_t)l * d, r, d * sizeof(float));
        uint32_t head[5] = {6, l, d, n - 1, 0};
        fwrite(head, sizeof(head), 1, m->dbt_file);
        fwrite(r, sizeof(float), d, m->dbt_file);
    }
    if (m->dbt_cut && l + 1 == (uint32_t)m->dbt_cut) {
        m->dbt_cut_hit = 1;
        return 1;
    }
    return 0;
}
