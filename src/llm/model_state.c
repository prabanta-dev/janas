/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_state.c - what a model has computed of a sequence, copied out and
 * put back (see janas_llm_model_state_save in model.h).
 *
 * An attention layer's part is its keys and values for positions 0 .. n - 1:
 * per KV head a run of n rows, since a head's positions are contiguous in
 * the cache. The recurrent part (Gated DeltaNet) is kept in its short form,
 * the one buffer that holds the state after position n - 1, when the model
 * stands exactly there; otherwise - a draft was partly rejected, and the
 * model reaches position n by replaying the accepted tokens of its log -
 * both buffers and the log are kept as they are.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_impl.h"

struct janas_llm_state {
    uint32_t n;       /* positions 0 .. n - 1 */
    uint32_t n_slots; /* KV slots (attention layers, the MTP block's) */
    /* what else decided the numbers: a state computed with fewer experts,
       or fewer bits of them, is not the one this model would compute */
    uint32_t k_use;
    int exp_level;
    int8_t *k, *v;  /* per slot: its KV heads x n x its head_dim */
    float *ks, *vs; /* per slot: its KV heads x n */
    /* recurrent state: nbuf buffers (1: the state after n - 1, 2: base and
       tip as the model had them, with the log) */
    int nbuf;
    uint16_t *ssm;
    float *conv;
    float *log_mixed, *log_beta, *log_decay;
    int base;
    uint32_t base_pos, log_n;
    size_t bytes;
};

static size_t ssm_len(const struct janas_llm_model *m)
{
    return (size_t)m->n_rec * m->n_vh * m->ds * m->ds;
}

static size_t conv_len(const struct janas_llm_model *m)
{
    return (size_t)m->n_rec * m->conv_ch * (m->d_conv - 1);
}

void janas_llm_state_free(struct janas_llm_state *st)
{
    if (!st)
        return;
    free(st->k);
    free(st->v);
    free(st->ks);
    free(st->vs);
    free(st->ssm);
    free(st->conv);
    free(st->log_mixed);
    free(st->log_beta);
    free(st->log_decay);
    free(st);
}

uint32_t janas_llm_state_length(const struct janas_llm_state *st)
{
    return st->n;
}

size_t janas_llm_state_bytes(const struct janas_llm_state *st)
{
    return st->bytes;
}

/* The bytes of keys (or values) and the number of scales of n positions
   of the first n_slots slots. */
static void kv_lengths(const struct janas_llm_model *m, uint32_t n_slots,
                       size_t n, size_t *kv, size_t *rows)
{
    *kv = *rows = 0;
    for (uint32_t sl = 0; sl < n_slots; sl++) {
        *rows += (size_t)m->kv_heads[sl] * n;
        *kv += (size_t)m->kv_heads[sl] * n * m->kv_hd[sl];
    }
}

/* the KV rows of positions 0 .. n - 1, out of the cache (out) or into it */
static void copy_kv(struct janas_llm_model *m, struct janas_llm_state *st,
                    int out)
{
    size_t n = st->n, sb = 0, ss = 0;
    for (uint32_t sl = 0; sl < st->n_slots; sl++)
        for (uint32_t h = 0; h < m->kv_heads[sl];
             h++, sb += n * m->kv_hd[sl], ss += n) {
            size_t hd = m->kv_hd[sl];
            size_t cb = m->kv_off[sl] + h * m->n_ctx * hd;
            size_t cs = m->ks_off[sl] + h * m->n_ctx;
            if (out) {
                memcpy(st->k + sb, m->kcache + cb, n * hd);
                memcpy(st->v + sb, m->vcache + cb, n * hd);
                memcpy(st->ks + ss, m->kscale + cs, n * sizeof(float));
                memcpy(st->vs + ss, m->vscale + cs, n * sizeof(float));
            } else {
                memcpy(m->kcache + cb, st->k + sb, n * hd);
                memcpy(m->vcache + cb, st->v + sb, n * hd);
                memcpy(m->kscale + cs, st->ks + ss, n * sizeof(float));
                memcpy(m->vscale + cs, st->vs + ss, n * sizeof(float));
            }
        }
}

static void *dup_mem(const void *p, size_t len)
{
    void *d = malloc(len ? len : 1);
    if (d && len)
        memcpy(d, p, len);
    return d;
}

struct janas_llm_state *janas_llm_model_state_save(struct janas_llm_model *m,
                                                   uint32_t n)
{
    if (n > m->n_ctx)
        return NULL;
    int nbuf = 0;
    if (m->a->rec) {
        /* the state must stand at n, or be able to get there by replay */
        if (n == 0 || m->base_pos == UINT32_MAX || n < m->base_pos ||
            n > m->base_pos + m->log_n)
            return NULL;
        nbuf = n == m->base_pos + m->log_n ? 1 : 2;
    }
    struct janas_llm_state *st = calloc(1, sizeof(*st));
    if (!st)
        return NULL;
    st->n = n;
    st->k_use = m->k_use;
    st->exp_level = m->exp_level;
    st->n_slots = m->n_attn + (m->mtp ? 1u : 0u);
    st->nbuf = nbuf;
    size_t rows, kv;
    kv_lengths(m, st->n_slots, n, &kv, &rows);
    st->k = malloc(kv ? kv : 1);
    st->v = malloc(kv ? kv : 1);
    st->ks = malloc((rows ? rows : 1) * sizeof(float));
    st->vs = malloc((rows ? rows : 1) * sizeof(float));
    int bad = !st->k || !st->v || !st->ks || !st->vs;
    st->bytes = 2 * kv + 2 * rows * sizeof(float);
    if (!bad)
        copy_kv(m, st, 1);
    if (nbuf == 1) {
        /* the buffer holding the state after n - 1: the tip, or the base
           when the last call has been taken back to its start */
        int b = m->log_n > 0 ? 1 - m->base : m->base;
        st->ssm =
            dup_mem(m->ssm_buf + b * ssm_len(m), ssm_len(m) * sizeof(uint16_t));
        st->conv =
            dup_mem(m->conv_buf + b * conv_len(m), conv_len(m) * sizeof(float));
        bad |= !st->ssm || !st->conv;
        st->bytes +=
            ssm_len(m) * sizeof(uint16_t) + conv_len(m) * sizeof(float);
    } else if (nbuf == 2) {
        size_t B = JANAS_LLM_MAX_BLOCK;
        size_t lm = (size_t)m->n_rec * B * m->conv_ch * sizeof(float);
        size_t lb = (size_t)m->n_rec * B * m->n_vh * sizeof(float);
        st->ssm = dup_mem(m->ssm_buf, 2 * ssm_len(m) * sizeof(uint16_t));
        st->conv = dup_mem(m->conv_buf, 2 * conv_len(m) * sizeof(float));
        st->log_mixed = dup_mem(m->log_mixed, lm);
        st->log_beta = dup_mem(m->log_beta, lb);
        st->log_decay = dup_mem(m->log_decay, lb);
        bad |= !st->ssm || !st->conv || !st->log_mixed || !st->log_beta ||
               !st->log_decay;
        st->base = m->base;
        st->base_pos = m->base_pos;
        st->log_n = m->log_n;
        st->bytes += 2 * ssm_len(m) * sizeof(uint16_t) +
                     2 * conv_len(m) * sizeof(float) + lm + 2 * lb;
    }
    if (bad) {
        janas_llm_state_free(st);
        return NULL;
    }
    return st;
}

int janas_llm_model_state_load(struct janas_llm_model *m,
                               const struct janas_llm_state *st)
{
    /* the rows copied do not depend on the context opened, only on its
       being long enough */
    if (st->n > m->n_ctx || st->n_slots != m->n_attn + (m->mtp ? 1u : 0u) ||
        st->k_use != m->k_use || st->exp_level != m->exp_level ||
        (m->a->rec != 0) != (st->nbuf != 0))
        return -1;
    copy_kv(m, (struct janas_llm_state *)st, 0);
    if (st->nbuf == 1) {
        /* into the base, standing at n with nothing logged: the next call
           starts from it */
        m->base = 0;
        memcpy(m->ssm_buf, st->ssm, ssm_len(m) * sizeof(uint16_t));
        memcpy(m->conv_buf, st->conv, conv_len(m) * sizeof(float));
        m->base_pos = st->n;
        m->log_n = 0;
    } else if (st->nbuf == 2) {
        size_t B = JANAS_LLM_MAX_BLOCK;
        memcpy(m->ssm_buf, st->ssm, 2 * ssm_len(m) * sizeof(uint16_t));
        memcpy(m->conv_buf, st->conv, 2 * conv_len(m) * sizeof(float));
        memcpy(m->log_mixed, st->log_mixed,
               (size_t)m->n_rec * B * m->conv_ch * sizeof(float));
        memcpy(m->log_beta, st->log_beta,
               (size_t)m->n_rec * B * m->n_vh * sizeof(float));
        memcpy(m->log_decay, st->log_decay,
               (size_t)m->n_rec * B * m->n_vh * sizeof(float));
        m->base = st->base;
        m->base_pos = st->base_pos;
        m->log_n = st->log_n;
    }
    return 0;
}

/*
 * On disk: a header of 32-bit numbers - a mark, the version, and the shape
 * of the model the state came from - then the arrays as they are in
 * memory. Read back only into a model of the same shape: the file's name
 * already says which model (the caller's business), and this says the
 * numbers fit.
 */
#define STATE_MARK 0x54534b4au /* "JKST" */
#define STATE_VERSION 1u
enum {
    H_MARK,
    H_VERSION,
    H_N,
    H_SLOTS,
    H_HEADS,
    H_HD,
    H_NBUF,
    H_REC,
    H_VH,
    H_DS,
    H_CONV_CH,
    H_D_CONV,
    H_BASE,
    H_BASE_POS,
    H_LOG_N,
    H_K_USE,
    H_LEVEL,
    H_COUNT
};

static void shape(const struct janas_llm_model *m, uint32_t *h)
{
    h[H_MARK] = STATE_MARK;
    h[H_VERSION] = STATE_VERSION;
    h[H_SLOTS] = m->n_attn + (m->mtp ? 1u : 0u);
    h[H_HEADS] = m->n_head_kv;
    h[H_HD] = m->head_dim;
    h[H_REC] = m->n_rec;
    h[H_VH] = m->n_vh;
    h[H_DS] = m->ds;
    h[H_CONV_CH] = m->conv_ch;
    h[H_D_CONV] = m->d_conv;
}

static int put(FILE *f, const void *p, size_t len)
{
    return len == 0 || fwrite(p, 1, len, f) == len ? 0 : -1;
}

static int get(FILE *f, void *p, size_t len)
{
    return len == 0 || fread(p, 1, len, f) == len ? 0 : -1;
}

/* the lengths of the arrays, from the model's shape */
static void lengths(const struct janas_llm_model *m, const uint32_t *h,
                    size_t *kv, size_t *rows, size_t *ssm, size_t *conv,
                    size_t *lm, size_t *lb)
{
    size_t B = JANAS_LLM_MAX_BLOCK, nb = h[H_NBUF];
    kv_lengths(m, h[H_SLOTS], h[H_N], kv, rows);
    *ssm = nb ? nb * ssm_len(m) * sizeof(uint16_t) : 0;
    *conv = nb ? nb * conv_len(m) * sizeof(float) : 0;
    *lm = nb == 2 ? (size_t)m->n_rec * B * m->conv_ch * sizeof(float) : 0;
    *lb = nb == 2 ? (size_t)m->n_rec * B * m->n_vh * sizeof(float) : 0;
}

int janas_llm_state_write(const struct janas_llm_model *m,
                          const struct janas_llm_state *st, FILE *f)
{
    uint32_t h[H_COUNT] = {0};
    shape(m, h);
    if (h[H_SLOTS] != st->n_slots)
        return -1;
    h[H_N] = st->n;
    h[H_NBUF] = (uint32_t)st->nbuf;
    h[H_BASE] = (uint32_t)st->base;
    h[H_BASE_POS] = st->base_pos;
    h[H_LOG_N] = st->log_n;
    h[H_K_USE] = st->k_use;
    h[H_LEVEL] = (uint32_t)st->exp_level;
    size_t kv, rows, ssm, conv, lm, lb;
    lengths(m, h, &kv, &rows, &ssm, &conv, &lm, &lb);
    return put(f, h, sizeof(h)) || put(f, st->k, kv) || put(f, st->v, kv) ||
                   put(f, st->ks, rows * sizeof(float)) ||
                   put(f, st->vs, rows * sizeof(float)) ||
                   put(f, st->ssm, ssm) || put(f, st->conv, conv) ||
                   put(f, st->log_mixed, lm) || put(f, st->log_beta, lb) ||
                   put(f, st->log_decay, lb)
               ? -1
               : 0;
}

struct janas_llm_state *janas_llm_state_read(const struct janas_llm_model *m,
                                             FILE *f)
{
    uint32_t h[H_COUNT], want[H_COUNT] = {0};
    shape(m, want);
    if (get(f, h, sizeof(h)) != 0)
        return NULL;
    static const int same[] = {H_MARK, H_VERSION, H_SLOTS, H_HEADS,   H_HD,
                               H_REC,  H_VH,      H_DS,    H_CONV_CH, H_D_CONV};
    for (size_t i = 0; i < sizeof(same) / sizeof(same[0]); i++)
        if (h[same[i]] != want[same[i]])
            return NULL;
    if (h[H_N] > m->n_ctx || h[H_NBUF] > 2 ||
        (h[H_NBUF] != 0) != (m->a->rec != 0))
        return NULL;
    struct janas_llm_state *st = calloc(1, sizeof(*st));
    if (!st)
        return NULL;
    st->n = h[H_N];
    st->n_slots = h[H_SLOTS];
    st->nbuf = (int)h[H_NBUF];
    st->base = (int)h[H_BASE];
    st->base_pos = h[H_BASE_POS];
    st->log_n = h[H_LOG_N];
    st->k_use = h[H_K_USE];
    st->exp_level = (int)h[H_LEVEL];
    size_t kv, rows, ssm, conv, lm, lb;
    lengths(m, h, &kv, &rows, &ssm, &conv, &lm, &lb);
    st->k = malloc(kv ? kv : 1);
    st->v = malloc(kv ? kv : 1);
    st->ks = malloc((rows ? rows : 1) * sizeof(float));
    st->vs = malloc((rows ? rows : 1) * sizeof(float));
    if (ssm) {
        st->ssm = malloc(ssm);
        st->conv = malloc(conv);
    }
    if (lm) {
        st->log_mixed = malloc(lm);
        st->log_beta = malloc(lb);
        st->log_decay = malloc(lb);
    }
    st->bytes = 2 * kv + 2 * rows * sizeof(float) + ssm + conv + lm + 2 * lb;
    int bad = !st->k || !st->v || !st->ks || !st->vs ||
              (ssm && (!st->ssm || !st->conv)) ||
              (lm && (!st->log_mixed || !st->log_beta || !st->log_decay));
    bad = bad || get(f, st->k, kv) || get(f, st->v, kv) ||
          get(f, st->ks, rows * sizeof(float)) ||
          get(f, st->vs, rows * sizeof(float)) || get(f, st->ssm, ssm) ||
          get(f, st->conv, conv) || get(f, st->log_mixed, lm) ||
          get(f, st->log_beta, lb) || get(f, st->log_decay, lb);
    if (bad || (st->nbuf == 2 && (st->base < 0 || st->base > 1))) {
        janas_llm_state_free(st);
        return NULL;
    }
    return st;
}
