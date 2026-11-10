/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_observe.c - the observers of the forward pass (observe.h):
 * attaching and detaching them, and calling them at the hook points of the
 * CPU's layer loop (model.c) and feed-forward (model_ffn.c).
 */
#include <stdlib.h>
#include <string.h>

#include "model_impl.h"

int janas_llm_model_observe(struct janas_llm_model *m,
                            const struct janas_observer *o)
{
    if (m->n_obs >= JANAS_MAX_OBSERVERS)
        return -1;
    /* the pairs' experts, kept by the feed-forward's setup when allocated */
    if (!m->ana_exp && m->pairs_cap &&
        !(m->ana_exp = calloc(m->pairs_cap, sizeof(uint32_t))))
        return -1;
    m->obs[m->n_obs++] = *o;
    return 0;
}

void janas_llm_model_unobserve(struct janas_llm_model *m, void *ctx)
{
    for (uint32_t i = 0; i < m->n_obs; i++)
        if (m->obs[i].ctx == ctx) {
            memmove(&m->obs[i], &m->obs[i + 1],
                    (m->n_obs - i - 1) * sizeof(m->obs[0]));
            m->n_obs--;
            return;
        }
}

void janas_m_obs_ffn_input(struct janas_llm_model *m, uint32_t l,
                           const float *xn, uint32_t n, uint32_t d)
{
    for (uint32_t i = 0; i < m->n_obs; i++)
        if (m->obs[i].ffn_input)
            m->obs[i].ffn_input(m->obs[i].ctx, m, l, xn, n, d);
}

void janas_m_obs_ffn_pair(struct janas_llm_model *m, size_t g,
                          const float *gate, const float *up, const float *h,
                          uint32_t ff)
{
    uint32_t p = m->order[g];
    uint32_t e = m->ana_exp ? m->ana_exp[g] : m->n_expert;
    for (uint32_t i = 0; i < m->n_obs; i++)
        if (m->obs[i].ffn_pair)
            m->obs[i].ffn_pair(m->obs[i].ctx, m, m->ana_layer, (uint32_t)g,
                               m->pair_tok[p], e, m->pair_w[p], gate, up, h,
                               ff);
}

void janas_m_obs_ffn_done(struct janas_llm_model *m, uint32_t n_rows)
{
    for (uint32_t i = 0; i < m->n_obs; i++)
        if (m->obs[i].ffn_done)
            m->obs[i].ffn_done(m->obs[i].ctx, m, m->ana_layer, m->glist,
                               n_rows);
}

int janas_m_obs_layer_out(struct janas_llm_model *m, uint32_t l, const float *x,
                          uint32_t n, uint32_t d)
{
    int stop = 0;
    for (uint32_t i = 0; i < m->n_obs; i++)
        if (m->obs[i].layer_out)
            stop |= m->obs[i].layer_out(m->obs[i].ctx, m, l, x, n, d) != 0;
    return stop;
}
