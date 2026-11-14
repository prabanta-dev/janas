/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_seq.c - the sequences beside the model's own (A1, several
 * conversations in a pass): their keys and values, laid out as the model's
 * (kv_off, ks_off, its context), and which models may run several of them
 * at once (janas_llm_model_forward_parts, model.c).
 */
#include <stdlib.h>
#include <string.h>

#include "common/aligned.h"
#include "model_impl.h"

int janas_llm_model_parts_ok(const struct janas_llm_model *m)
{
    /* attention alone, on the positions of each piece: not yet the
       recurrent layers' states, Gemma 4's rings and per-layer inputs,
       Nemotron-H's layers, qwen4exp's streams and sparse attention */
    return !m->a->rec && !m->a->swa && !m->ple_dim && !m->a->nh && !m->hc_n &&
           !m->ix_heads;
}

struct janas_seq *janas_llm_model_seq_new(struct janas_llm_model *m)
{
    struct janas_seq *s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    size_t kvb = m->kv_bytes ? m->kv_bytes : 4096,
           ksb = m->ks_bytes ? m->ks_bytes : 4096;
    s->kcache = janas_aligned_alloc(4096, kvb);
    s->vcache = janas_aligned_alloc(4096, kvb);
    s->kscale = janas_aligned_alloc(4096, ksb);
    s->vscale = janas_aligned_alloc(4096, ksb);
    if (!s->kcache || !s->vcache || !s->kscale || !s->vscale) {
        janas_llm_model_seq_free(m, s);
        return NULL;
    }
    s->base_pos = UINT32_MAX;
    return s;
}

void janas_llm_model_seq_free(struct janas_llm_model *m, struct janas_seq *s)
{
    (void)m;
    if (!s)
        return;
    janas_aligned_free(s->kcache);
    janas_aligned_free(s->vcache);
    janas_aligned_free(s->kscale);
    janas_aligned_free(s->vscale);
    free(s);
}
