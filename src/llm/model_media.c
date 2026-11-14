/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_media.c - pictures in a model's sequence (model.h): their
 * embeddings in place of their tokens', and the positions their RoPE
 * angles come from. A picture of nx x ny tokens starting at sequence
 * position `at` has, for its token k, the time position s, the row s +
 * k / nx and the column s + k % nx, s being the position its first token
 * would have as text; the text after it goes on from s + max(nx, ny),
 * so every later position is shifted by max(nx, ny) - nx * ny. The
 * interleaved multimodal RoPE of Qwen3-VL and Qwen3.5 takes, for frequency
 * i of n_rot / 2, the row when i % 3 == 1 and i < 3 * sections[1], the
 * column when i % 3 == 2 and i < 3 * sections[2], the time otherwise:
 * for text the three are equal and nothing changes. A model without the
 * multimodal sections (Gemma 4) numbers a picture's tokens as text.
 */
#include "model_impl.h"

#include <stdlib.h>
#include <string.h>

int janas_llm_model_media_add(struct janas_llm_model *m, uint32_t at,
                              const float *embd, uint32_t nx, uint32_t ny,
                              int both_ways)
{
    uint64_t n = (uint64_t)nx * ny;
    if (!n || at + n > m->n_ctx || (both_ways && n > m->blk))
        return -1;
    if (both_ways && !m->att_last &&
        !(m->att_last = malloc(sizeof(uint32_t) * m->blk)))
        return -1;
    for (uint32_t i = 0; i < m->cur->n_media; i++) {
        const struct janas_media *p = &m->cur->media[i];
        if (at < p->at + p->nx * p->ny && p->at < at + n)
            return -1; /* overlapping */
    }
    float *e = malloc(sizeof(float) * (size_t)n * m->d_model);
    struct janas_media *t =
        e ? realloc(m->cur->media, sizeof(*t) * (m->cur->n_media + 1)) : NULL;
    if (!t) {
        free(e);
        return -1;
    }
    memcpy(e, embd, sizeof(float) * (size_t)n * m->d_model);
    m->cur->media = t;
    /* in order of their place in the sequence */
    uint32_t k = m->cur->n_media;
    while (k > 0 && m->cur->media[k - 1].at > at) {
        m->cur->media[k] = m->cur->media[k - 1];
        k--;
    }
    m->cur->media[k] = (struct janas_media){
        .at = at, .nx = nx, .ny = ny, .both = both_ways != 0, .embd = e};
    m->cur->n_media++;
    return 0;
}

void janas_llm_model_media_drop(struct janas_llm_model *m, uint32_t from)
{
    uint32_t keep = 0;
    for (uint32_t i = 0; i < m->cur->n_media; i++) {
        struct janas_media *p = &m->cur->media[i];
        if (p->at + p->nx * p->ny <= from)
            m->cur->media[keep++] = *p;
        else
            free(p->embd);
    }
    m->cur->n_media = keep;
    if (!keep) {
        free(m->cur->media);
        m->cur->media = NULL;
    }
}

uint32_t janas_llm_model_media_cut(const struct janas_llm_model *m,
                                   uint32_t pos0, uint32_t n)
{
    for (uint32_t i = 0; i < m->cur->n_media; i++) {
        const struct janas_media *p = &m->cur->media[i];
        uint32_t end = p->at + p->nx * p->ny;
        if (!p->both || p->at >= pos0 + n || end <= pos0 + n || end <= pos0)
            continue;
        /* cut by the pass: end before it, or (beginning with it) take it
           whole, which media_add made sure fits a pass */
        if (p->at > pos0)
            return p->at - pos0;
        if (end - pos0 <= m->blk)
            return end - pos0;
    }
    return n;
}

uint32_t janas_llm_model_media_count(const struct janas_llm_model *m)
{
    return m->cur->n_media;
}

struct janas_pool *janas_llm_model_pool(const struct janas_llm_model *m)
{
    return m->compute;
}

const float *janas_m_media_row(const struct janas_llm_model *m, uint32_t pos)
{
    for (uint32_t i = 0; i < m->cur->n_media; i++) {
        const struct janas_media *p = &m->cur->media[i];
        if (pos >= p->at && pos < p->at + p->nx * p->ny)
            return p->embd + (size_t)(pos - p->at) * m->d_model;
    }
    return NULL;
}

void janas_m_media_pos(const struct janas_llm_model *m, uint32_t pos,
                       float p3[3])
{
    if (!m->mrope[0] && !m->mrope[1] && !m->mrope[2]) {
        p3[0] = p3[1] = p3[2] = (float)pos;
        return;
    }
    int64_t shift = 0;
    for (uint32_t i = 0; i < m->cur->n_media; i++) {
        const struct janas_media *p = &m->cur->media[i];
        uint32_t n = p->nx * p->ny;
        if (pos >= p->at && pos < p->at + n) {
            uint32_t k = pos - p->at;
            float s = (float)((int64_t)p->at + shift);
            p3[0] = s;
            p3[1] = s + (float)(k / p->nx);
            p3[2] = s + (float)(k % p->nx);
            return;
        }
        if (pos >= p->at + n)
            shift += (int64_t)(p->nx > p->ny ? p->nx : p->ny) - (int64_t)n;
    }
    p3[0] = p3[1] = p3[2] = (float)((int64_t)pos + shift);
}

int janas_m_media_both(const struct janas_llm_model *m, uint32_t pos0,
                       uint32_t n)
{
    for (uint32_t i = 0; i < m->cur->n_media; i++) {
        const struct janas_media *p = &m->cur->media[i];
        if (p->both && p->at < pos0 + n && pos0 < p->at + p->nx * p->ny)
            return 1;
    }
    return 0;
}

void janas_m_media_last(const struct janas_llm_model *m, uint32_t pos0,
                        uint32_t n, uint32_t *last)
{
    for (uint32_t j = 0; j < n; j++)
        last[j] = pos0 + j;
    for (uint32_t i = 0; i < m->cur->n_media; i++) {
        const struct janas_media *p = &m->cur->media[i];
        uint32_t end = p->at + p->nx * p->ny; /* one past its last token */
        if (!p->both || p->at >= pos0 + n || end <= pos0)
            continue;
        /* a picture cut by the pass's end sees as far as the pass goes */
        uint32_t hi = (end < pos0 + n ? end : pos0 + n) - 1;
        for (uint32_t q = p->at > pos0 ? p->at : pos0; q <= hi; q++)
            last[q - pos0] = hi;
    }
}

void janas_m_media_free(struct janas_llm_model *m)
{
    for (uint32_t i = 0; i < m->cur->n_media; i++)
        free(m->cur->media[i].embd);
    free(m->cur->media);
    m->cur->media = NULL;
    m->cur->n_media = 0;
    free(m->att_last);
    m->att_last = NULL;
}
