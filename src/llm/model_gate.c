/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_gate.c - the entrance gate on a model (model.h): its readers
 * opened against the model, armed for the next forward calls, and its
 * decision on the residual the layer loop kept (model.c).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gate.h"
#include "model_impl.h"

/* The observer: the last token's residual at the gate's layer, when
   armed. */
static int gate_layer_out(void *ctx, struct janas_llm_model *m, uint32_t l,
                          const float *x, uint32_t n, uint32_t d)
{
    (void)ctx;
    if (!m->gate_armed || l != m->gate_layer || !n)
        return 0;
    uint32_t row = n - 1;
    if (m->gate_pos != UINT32_MAX) {
        if (m->gate_pos < m->fwd_pos0 || m->gate_pos - m->fwd_pos0 >= n)
            return 0;
        row = m->gate_pos - m->fwd_pos0;
    }
    memcpy(m->gate_x, x + (size_t)row * d, d * sizeof(float));
    m->gate_got = 1;
    return 0;
}

int janas_llm_model_gate_open(struct janas_llm_model *m, const char *path,
                              char *err, size_t err_len)
{
    struct janas_gate *g =
        janas_gate_load(path, m->d_model, m->n_layer, err, err_len);
    if (!g)
        return -1;
    float *x = calloc(m->d_model, sizeof(float));
    if (!x) {
        janas_gate_free(g);
        snprintf(err, err_len, "out of memory for the gate");
        return -1;
    }
    janas_m_gate_free(m);
    struct janas_observer o = {.ctx = &m->egate, .layer_out = gate_layer_out};
    if (janas_llm_model_observe(m, &o) != 0) {
        janas_gate_free(g);
        free(x);
        snprintf(err, err_len, "too many observers of the model");
        return -1;
    }
    m->egate = g;
    m->gate_x = x;
    m->gate_layer = janas_gate_layer(g);
    return 0;
}

void janas_llm_model_gate_arm(struct janas_llm_model *m, int on)
{
    m->gate_armed = on && m->egate;
    m->gate_got = 0;
    m->gate_pos = UINT32_MAX;
}

void janas_llm_model_gate_at(struct janas_llm_model *m, uint32_t pos)
{
    m->gate_pos = pos;
}

int janas_llm_model_gate_result(const struct janas_llm_model *m,
                                uint32_t *domain, double *scores)
{
    if (!m->egate || !m->gate_got)
        return -1;
    return janas_gate_decide(m->egate, m->gate_x, domain, scores);
}

const char *janas_llm_model_gate_domain(const struct janas_llm_model *m,
                                        uint32_t domain)
{
    return m->egate ? janas_gate_domain_name(m->egate, domain) : "";
}

void janas_m_gate_free(struct janas_llm_model *m)
{
    if (m->egate)
        janas_llm_model_unobserve(m, &m->egate);
    janas_gate_free(m->egate);
    free(m->gate_x);
    m->egate = NULL;
    m->gate_x = NULL;
    m->gate_armed = m->gate_got = 0;
}
