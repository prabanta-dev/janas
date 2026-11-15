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
    /* attention - Gemma 4's sliding windows and rings too, and its
       per-layer inputs, which are the tokens' - and the recurrent layers,
       Gated DeltaNet (Qwen3-Next, Qwen3.5 and 3.6) and Mamba-2
       (Nemotron-H), each piece on its own sequence: not yet qwen4exp's
       streams and sparse attention */
    return !m->hc_n && !m->ix_heads;
}

void janas_llm_model_set_seq(struct janas_llm_model *m, struct janas_seq *s)
{
    m->cur = s ? s : &m->seq0;
}

size_t janas_llm_model_seq_bytes(const struct janas_llm_model *m)
{
    size_t b = 2 * (m->kv_bytes ? m->kv_bytes : 4096) +
               2 * (m->ks_bytes ? m->ks_bytes : 4096);
    if (m->a->rec)
        b += 2 * (size_t)m->n_rec * m->ssm_per * sizeof(uint16_t) +
             2 * (size_t)m->n_rec * m->conv_ch * (m->d_conv - 1) *
                 sizeof(float) +
             (size_t)m->n_rec * JANAS_LLM_MAX_BLOCK *
                 (m->conv_ch + 2 * (size_t)m->n_vh) * sizeof(float);
    return b;
}

struct janas_seq *janas_llm_model_seq_own(struct janas_llm_model *m)
{
    return &m->seq0;
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
    if (m->a->rec) {
        /* the recurrent state in two buffers and the log of the last call,
           as the model's own (model_load.c) */
        size_t sb = 2 * (size_t)m->n_rec * m->ssm_per * sizeof(uint16_t),
               cb = 2 * (size_t)m->n_rec * m->conv_ch * (m->d_conv - 1) *
                    sizeof(float),
               L = JANAS_LLM_MAX_BLOCK;
        s->ssm_buf = janas_aligned_alloc(4096, sb ? sb : 4096);
        s->conv_buf = janas_aligned_alloc(4096, cb ? cb : 4096);
        s->log_mixed = janas_aligned_alloc(
            4096, (size_t)m->n_rec * L * m->conv_ch * sizeof(float) + 4096);
        s->log_beta = janas_aligned_alloc(
            4096, (size_t)m->n_rec * L * m->n_vh * sizeof(float) + 4096);
        s->log_decay = janas_aligned_alloc(
            4096, (size_t)m->n_rec * L * m->n_vh * sizeof(float) + 4096);
        if (!s->ssm_buf || !s->conv_buf || !s->log_mixed || !s->log_beta ||
            !s->log_decay) {
            janas_llm_model_seq_free(m, s);
            return NULL;
        }
        memset(s->ssm_buf, 0, sb);
        memset(s->conv_buf, 0, cb);
    }
    s->base_pos = UINT32_MAX;
    return s;
}

void janas_llm_model_seq_free(struct janas_llm_model *m, struct janas_seq *s)
{
    if (!s)
        return;
    if (m->cur == s) /* the single calls go back to the model's own */
        m->cur = &m->seq0;
    for (uint32_t i = 0; i < s->n_media; i++)
        free(s->media[i].embd);
    free(s->media);
    janas_aligned_free(s->kcache);
    janas_aligned_free(s->vcache);
    janas_aligned_free(s->kscale);
    janas_aligned_free(s->vscale);
    janas_aligned_free(s->ssm_buf);
    janas_aligned_free(s->conv_buf);
    janas_aligned_free(s->log_mixed);
    janas_aligned_free(s->log_beta);
    janas_aligned_free(s->log_decay);
    free(s);
}
