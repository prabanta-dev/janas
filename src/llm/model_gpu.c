/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_gpu.c - the model's side of the whole token on the GPU
 * (gpu_token.h): which models it takes, their layers described for it, the
 * KV cache shared with it, and the call from the forward pass.
 *
 * For now opt-in (JANAS_GPU_TOKEN=1) and for the dense attention models of
 * qwen3's shape on a GPU that reads the host's memory in place, with the
 * feed-forward weights (the experts' arena) on it already.
 */
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "expert_cache.h"
#include "gpu.h"
#include "gpu_token.h"
#include "quant.h"

#include "model_impl.h"

/* Why a model is not for the token on the GPU, NULL if it is. */
static const char *not_for_gpu(const struct janas_llm_model *m)
{
    if (!m->gpu || m->gpu_off || !janas_gpu_shares_host(m->gpu))
        return "no GPU reading the host's memory";
    if (m->a->rec || m->a->swa || m->a->attn_gate || m->a->sandwich ||
        m->a->gelu || m->kv16 || m->softcap > 0.0f || m->ple_dim || m->mtp ||
        m->asst)
        return "an architecture the token's shaders do not take yet";
    if (m->n_expert != 1 || !m->cache)
        return "not a dense model";
    if (m->arena_gpu != 1 || m->arena_stale)
        return "the feed-forward weights are not on the GPU";
    for (uint32_t l = 0; l < m->n_layer; l++) {
        const struct layer *ly = &m->layers[l];
        if (ly->rec || ly->swa || !ly->wk || !ly->wv || !ly->q_norm ||
            !ly->k_norm || ly->post_attn || ly->post_ffn || ly->pre_norm2 ||
            m->kv_ring[ly->slot] || ly->hd != m->head_dim ||
            ly->n_kv != m->n_head_kv || ly->n_rot != m->n_rot)
            return "a layer the token's shaders do not take yet";
    }
    return NULL;
}

static struct janas_matvec_task task_of(const struct janas_llm_model *m,
                                        const struct janas_jns_tensor *t,
                                        uint32_t rows, uint32_t cols)
{
    return (struct janas_matvec_task){
        .type = (int)t->type, .w = data(m, t), .rows = rows, .cols = cols};
}

/* Whether a product goes to the aligned copy: Q6_K, or Q6_K in planes with
   all three (the Q6_K block it came from). */
static int repackable(const struct janas_matvec_task *t, int tiles)
{
    /* Q4_K: rows of 32 blocks and more (the tiles slow the short ones
       down), or all of them with tiles 2 */
    return t->type == JANAS_Q6_K ||
           (t->type == JANAS_Q6_K_P && t->plane[0] && t->plane[1]) ||
           (tiles >= 1 && t->type == JANAS_Q4_K &&
            (tiles >= 2 || t->cols / JANAS_QK >= 32));
}

/* Block b of row r of a copy in tiles (gpu.h JANAS_TILE) or not, its
   size, the rows it takes. */
static size_t tile_at(size_t r, size_t b, size_t nb, int tiles)
{
    return tiles ? ((r / JANAS_TILE) * nb + b) * JANAS_TILE + r % JANAS_TILE
                 : r * nb + b;
}

static size_t copy_bs(const struct janas_matvec_task *t)
{
    return t->type == JANAS_Q4_K ? janas_qtype_block_size(JANAS_Q4_K)
                                 : JANAS_Q6KA_BLOCK;
}

static size_t copy_rows(const struct janas_matvec_task *t, int tiles)
{
    return tiles ? (t->rows + JANAS_TILE - 1) / JANAS_TILE * JANAS_TILE
                 : t->rows;
}

/* Row r of a Q6_K or three-plane Q6_K_P task into dst, 224 bytes a block:
   ql, qh, scales, d at byte 208, the rest zero. */
static void repack_row(const struct janas_matvec_task *t, size_t r,
                       uint8_t *dst)
{
    size_t nb = t->cols / JANAS_QK;
    for (size_t b = 0; b < nb; b++, dst += JANAS_Q6KA_BLOCK) {
        struct janas_block_q6k q;
        size_t k = r * nb + b;
        if (t->type == JANAS_Q6_K)
            q = ((const struct janas_block_q6k *)t->w)[k];
        else
            janas_q6kp_to_q6k(
                (const struct janas_block_q6kp *)t->w + k,
                (const uint8_t *)t->plane[0] + k * JANAS_Q6KP_PLANE,
                (const uint8_t *)t->plane[1] + k * JANAS_Q6KP_PLANE, 1, &q);
        memset(dst, 0, JANAS_Q6KA_BLOCK);
        memcpy(dst, q.ql, sizeof(q.ql));
        memcpy(dst + 128, q.qh, sizeof(q.qh));
        memcpy(dst + 192, q.scales, sizeof(q.scales));
        memcpy(dst + 208, &q.d, sizeof(q.d));
    }
}

/*
 * JANAS_GPU_REPACK=1: the Q6_K products (and the three-plane Q6_K_P ones)
 * copied for the GPU into an aligned form it reads with whole 16-byte
 * loads, in one region of its driver's memory; the tasks pointed at it.
 * tiles (JANAS_GPU_REPACK - 1): 1, those and the Q4_K matrices of long
 * rows in tiles; 2, every Q4_K too (gpu.h JANAS_TILE). The CPU keeps its
 * own; the copy costs that much memory more. 0, or -1 with the tasks
 * unchanged.
 */
static int repack(struct janas_llm_model *m, int tiles)
{
    struct janas_matvec_task *t[8 * 1024];
    size_t n = 0, bytes = 0;
    for (uint32_t l = 0; l < m->n_layer; l++) {
        struct janas_gpu_tok_layer *g = &m->gtok_layers[l];
        struct janas_matvec_task *all[7] = {&g->wq,   &g->wk, &g->wv,  &g->wo,
                                            &g->gate, &g->up, &g->down};
        for (int i = 0; i < 7; i++)
            if (repackable(all[i], tiles) && n < sizeof(t) / sizeof(*t))
                t[n++] = all[i];
    }
    if (repackable(&m->gtok_md.output, tiles) && n < sizeof(t) / sizeof(*t))
        t[n++] = &m->gtok_md.output;
    for (size_t i = 0; i < n; i++)
        bytes +=
            copy_rows(t[i], tiles) * (t[i]->cols / JANAS_QK) * copy_bs(t[i]);
    if (n == 0)
        return 0;
    if (!m->gtok_repack || m->gtok_repack_bytes < bytes) {
        m->gtok_repack = janas_gpu_alloc(m->gpu, bytes);
        m->gtok_repack_bytes = m->gtok_repack ? bytes : 0;
    }
    if (!m->gtok_repack)
        return -1;
    uint8_t *at = m->gtok_repack;
    memset(at, 0, bytes); /* the tiles' padding rows */
    for (size_t i = 0; i < n; i++) {
        size_t nb = t[i]->cols / JANAS_QK, bs = copy_bs(t[i]);
        uint8_t blk[JANAS_Q6KA_BLOCK];
        for (size_t r = 0; r < t[i]->rows; r++)
            for (size_t b = 0; b < nb; b++) {
                uint8_t *dst = at + tile_at(r, b, nb, tiles) * bs;
                if (t[i]->type == JANAS_Q4_K) {
                    memcpy(dst, (const uint8_t *)t[i]->w + (r * nb + b) * bs,
                           bs);
                    continue;
                }
                /* repack_row takes a whole row: one block through blk */
                struct janas_matvec_task one = *t[i];
                one.cols = JANAS_QK;
                one.w =
                    t[i]->type == JANAS_Q6_K
                        ? (const void *)((const struct janas_block_q6k *)t[i]
                                             ->w +
                                         r * nb + b)
                        : (const void *)((const struct janas_block_q6kp *)t[i]
                                             ->w +
                                         r * nb + b);
                if (t[i]->type == JANAS_Q6_K_P) {
                    one.plane[0] = (const uint8_t *)t[i]->plane[0] +
                                   (r * nb + b) * JANAS_Q6KP_PLANE;
                    one.plane[1] = (const uint8_t *)t[i]->plane[1] +
                                   (r * nb + b) * JANAS_Q6KP_PLANE;
                }
                repack_row(&one, 0, blk);
                memcpy(dst, blk, bs);
            }
        size_t used = copy_rows(t[i], tiles) * nb * bs;
        t[i]->type = t[i]->type == JANAS_Q4_K ? JANAS_Q4_K_T
                     : tiles                  ? JANAS_Q6_K_AT
                                              : JANAS_Q6_K_A;
        t[i]->w = at;
        t[i]->plane[0] = t[i]->plane[1] = NULL;
        at += used;
    }
    janas_gpu_flush(m->gpu, m->gtok_repack, bytes);
    return 0;
}

/* The layers described for the GPU, the KV cache imported: 0, or -1. */
static int build(struct janas_llm_model *m, char *err, size_t len)
{
    uint32_t dm = m->d_model, hd = m->head_dim, qd = m->n_head * hd,
             kvd = m->n_head_kv * hd;
    if (!m->gtok_kv) {
        if (janas_gpu_import(m->gpu, m->kcache, m->kv_bytes) != 0 ||
            janas_gpu_import(m->gpu, m->vcache, m->kv_bytes) != 0 ||
            janas_gpu_import(m->gpu, m->kscale, m->ks_bytes) != 0 ||
            janas_gpu_import(m->gpu, m->vscale, m->ks_bytes) != 0) {
            snprintf(err, len, "the KV cache cannot be imported");
            return -1;
        }
        m->gtok_kv = 1;
    }
    free(m->gtok_layers);
    m->gtok_layers = calloc(m->n_layer, sizeof(*m->gtok_layers));
    if (!m->gtok_layers) {
        snprintf(err, len, "out of memory");
        return -1;
    }
    uint32_t ff = 0;
    for (uint32_t l = 0; l < m->n_layer; l++) {
        const struct layer *ly = &m->layers[l];
        const struct janas_jns_layer *jl = &m->j.layers[l];
        struct janas_gpu_tok_layer *g = &m->gtok_layers[l];
        uint32_t e0 = 0;
        const uint8_t *slot;
        if (janas_expert_cache_fetch(m->cache, l, &e0, 1, &slot) != 0) {
            snprintf(err, len, "the feed-forward of layer %u unreadable", l);
            return -1;
        }
        g->attn_norm = f32(m, ly->attn_norm);
        g->q_norm = f32(m, ly->q_norm);
        g->k_norm = f32(m, ly->k_norm);
        g->ffn_norm = f32(m, ly->ffn_norm);
        g->wq = task_of(m, ly->wq, qd, dm);
        g->wk = task_of(m, ly->wk, kvd, dm);
        g->wv = task_of(m, ly->wv, kvd, dm);
        g->wo = task_of(m, ly->wo, dm, qd);
        for (int p = 0; p < 3; p++) {
            const struct janas_jns_matrix *mx = &jl->m[p];
            struct janas_matvec_task t = {.type = (int)mx->type,
                                          .w = slot + mx->offset,
                                          .rows = mx->rows,
                                          .cols = mx->cols};
            if (p == 2) /* the down matrix's planes the cache holds */
                for (int q = 0; q < m->exp_level - 1; q++)
                    if (jl->plane[q].type)
                        t.plane[q] = slot + jl->plane[q].offset;
            *(p == 0 ? &g->gate : p == 1 ? &g->up : &g->down) = t;
        }
        if (l == 0)
            ff = g->gate.rows;
        if (g->gate.rows != ff || g->up.rows != ff || g->down.cols != ff) {
            snprintf(err, len, "feed-forwards of different widths");
            return -1;
        }
        g->kc = m->kcache + m->kv_off[ly->slot];
        g->vc = m->vcache + m->kv_off[ly->slot];
        g->ks = m->kscale + m->ks_off[ly->slot];
        g->vs = m->vscale + m->ks_off[ly->slot];
    }
    m->gtok_md = (struct janas_gpu_tok_model){
        .n_layer = m->n_layer,
        .dm = dm,
        .n_head = m->n_head,
        .n_kv = m->n_head_kv,
        .hd = hd,
        .n_rot = m->n_rot,
        .ff = ff,
        .n_vocab = m->n_vocab,
        .cap = m->kv_cap[m->layers[0].slot],
        .eps = m->eps,
        .attn_scale = 1.0f / sqrtf((float)hd),
        .layers = m->gtok_layers,
        .out_norm = f32(m, m->out_norm),
        .output = task_of(m, m->output, m->n_vocab, dm)};
    free(m->gblk_layers);
    m->gblk_layers = NULL;
    if (getenv("JANAS_GPU_REPACK") && atoi(getenv("JANAS_GPU_REPACK")) > 0) {
        /* the blocks keep the weights as they are: the copy's shaders take
           a single vector */
        m->gblk_layers = malloc(m->n_layer * sizeof(*m->gblk_layers));
        if (!m->gblk_layers) {
            snprintf(err, len, "out of memory");
            return -1;
        }
        memcpy(m->gblk_layers, m->gtok_layers,
               m->n_layer * sizeof(*m->gblk_layers));
        m->gtok_md.blayers = m->gblk_layers;
        m->gtok_md.boutput = m->gtok_md.output;
        if (repack(m, atoi(getenv("JANAS_GPU_REPACK")) - 1) != 0)
            fprintf(stderr, "janas: no aligned copy for the GPU: out of its "
                            "memory\n");
    }
    janas_gpu_tok_destroy(m->gpu, m->gtok);
    m->gtok = janas_gpu_tok_create(m->gpu, &m->gtok_md, err, len);
    m->gtok_loads = janas_expert_cache_loads(m->cache);
    return m->gtok ? 0 : -1;
}

int janas_m_gpu_token(struct janas_llm_model *m, uint32_t pos, uint32_t n,
                      int all, float *logits)
{
    if (m->gtok_state < 0)
        return -1;
    /* blocks of more than one token: opt-in for now (JANAS_GPU_BLOCK=1) */
    static int blocks = -1;
    if (blocks < 0)
        blocks =
            getenv("JANAS_GPU_BLOCK") && atoi(getenv("JANAS_GPU_BLOCK")) > 0;
    if (n > 1 && !blocks)
        return -1;
    if (m->gtok_state == 0) {
        const char *e = getenv("JANAS_GPU_TOKEN");
        if (!e || atoi(e) <= 0) {
            m->gtok_state = -1;
            return -1;
        }
    }
    const char *why = not_for_gpu(m);
    if (why) {
        /* the arena may reach the GPU later: wait for it, say nothing */
        if (m->gtok_state == 0 && m->arena_gpu == 0 && m->gpu && !m->gpu_off)
            return -1;
        if (m->gtok_state == 0)
            fprintf(stderr, "janas: no whole token on the GPU: %s\n", why);
        m->gtok_state = -1;
        return -1;
    }
    if (m->gtok_state == 0 ||
        janas_expert_cache_loads(m->cache) != m->gtok_loads) {
        char err[160];
        if (build(m, err, sizeof(err)) != 0) {
            fprintf(stderr, "janas: no whole token on the GPU: %s\n", err);
            m->gtok_state = -1;
            return -1;
        }
        m->gtok_state = 1;
    }
    /* a longer block (a prompt, only its last logits wanted) in pieces:
       each piece's keys and values in the cache before the next */
    uint32_t nb = janas_gpu_tok_max_block(m->gtok), half = m->n_rot / 2;
    if (n > nb && all)
        return -1;
    for (uint32_t j = 0; j < n; j += nb) {
        uint32_t k = n - j < nb ? n - j : nb;
        if (janas_gpu_tok_run(m->gpu, m->gtok, &m->gtok_md,
                              m->x + (size_t)j * m->d_model,
                              m->rope_cos + (size_t)j * half,
                              m->rope_sin + (size_t)j * half, pos + j, k,
                              j + k < n ? 2
                              : all     ? 1
                                        : 0,
                              logits) != 0)
            return -1;
    }
    return 0;
}
