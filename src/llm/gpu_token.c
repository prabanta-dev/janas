/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token.c - a whole token on the GPU with one submission (see
 * gpu_token.h).
 */
#include "gpu_token.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gpu_impl.h"
#include "quant.h"

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)

#include "norm_quant.h"  /* janas_spv_norm_quant */
#include "qk_rope.h"     /* janas_spv_qk_rope */
#include "attn_decode.h" /* janas_spv_attn_decode */
#include "attn_merge.h"  /* janas_spv_attn_merge */
#include "gdn_conv.h"    /* janas_spv_gdn_conv */
#include "gdn_cstate.h"  /* janas_spv_gdn_cstate */
#include "gdn_l2.h"      /* janas_spv_gdn_l2 */
#include "gdn_rec.h"     /* janas_spv_gdn_rec */
#include "gdn_gate.h"    /* janas_spv_gdn_gate */

/* The push constants of each shader, laid out as its GLSL block. */
struct push_nq {
    uint32_t x[2], add[2], w[2], xo[2], q256[2], q32[2], xn[2];
    uint32_t d;
    float eps;
};
struct push_qk {
    uint32_t qkv[2], qn[2], kn[2], rope[2], kc[2], vc[2], ks[2], vs[2];
    uint32_t n_head, n_kv, hd, n_rot, cap, qs;
    float eps;
    uint32_t pad, pos[2]; /* the position's buffer, 8-byte aligned */
};
struct push_at {
    uint32_t q[2], kc[2], vc[2], ks[2], vs[2], part[2];
    uint32_t n_head, n_kv, hd, cap, nch;
    float scale;
    uint32_t qs, ps;
    uint32_t pos[2]; /* the position's buffer */
};
struct push_am {
    uint32_t part[2], q256[2], q32[2];
    uint32_t hd, nch, ps, gs;
    uint32_t pos[2]; /* 8-byte aligned, as the GLSL block puts it */
    uint32_t gate[2];
};
/* the Gated DeltaNet shaders (gdn_*.comp) */
struct push_gconv {
    uint32_t mixed[2], out[2], w[2], state[2], pos[2];
    uint32_t cc, d, hs, so;
};
struct push_gcst {
    uint32_t mixed[2], state[2], pos[2];
    uint32_t cc, d, hs, so, n;
};
struct push_gl2 {
    uint32_t x[2];
    uint32_t cc;
    float eps;
};
struct push_grec {
    uint32_t co[2], ba[2], att[2], state[2], pos[2], dt[2], a[2];
    uint32_t cc, nv, nkh, n, hs, so, vmap_mod;
    float scale;
};
struct push_ggate {
    uint32_t att[2], z[2], w[2], q256[2], q32[2];
    uint32_t d;
    float eps;
};
struct push_dm { /* gpu.c's dense_mm.comp */
    uint32_t w[2], x[2], y[2];
    uint32_t rows, cols, nv, ys;
    float scale;
};
/* The attention's positions a run (attn_decode.comp C), and the shapes its
   shared memory takes: the query heads of a KV head times hd, and times a
   run. */
#define ATT_RUN 128
#define ATT_MAXQ 2048
#define ATT_MAXS 1024

/* The attention's first step in subgroups of 16 where the size may be
   chosen (JANAS_GPU_ATT_SG=n another): twice the lanes a hardware thread
   of the integrated GPU, its latencies better hidden - 11.7 -> 9.9 ms a
   token at 4000 positions - where the products lose with them. */
static uint32_t att_sg(void)
{
    const char *e = getenv("JANAS_GPU_ATT_SG");
    return e ? (uint32_t)atoi(e) : 16;
}
struct push_ac {
    uint32_t g[2], u[2], x[2];
    uint32_t blocks;
};
_Static_assert(sizeof(struct push_qk) <= JANAS_G_PUSH_BYTES, "push size");
_Static_assert(sizeof(struct push_at) <= JANAS_G_PUSH_BYTES, "push size");
/* the uvec2 members where GLSL puts them: on 8 bytes */
_Static_assert(offsetof(struct push_qk, pos) % 8 == 0, "push layout");
_Static_assert(offsetof(struct push_at, pos) % 8 == 0, "push layout");
_Static_assert(offsetof(struct push_am, pos) % 8 == 0, "push layout");
_Static_assert(sizeof(struct push_grec) <= JANAS_G_PUSH_BYTES, "push size");
/* The most tokens a block takes (the model's own blocks: model.h
   JANAS_LLM_MAX_BLOCK), and the memory the attention's runs may take for
   them. */
#define TOK_MAX_BLOCK 256
#define TOK_PART_BUDGET ((size_t)64 << 20)

/* JANAS_GPU_TOKEN_PROFILE=1: the GPU's own timestamps after each phase of
   the token, summed per phase and printed when the model is closed */
enum {
    PH_NORM,
    PH_QKV,
    PH_ROPE,
    PH_ATTN,
    PH_MERGE,
    PH_WO,
    PH_GATEUP,
    PH_ACT,
    PH_DOWN,
    PH_OUT,
    PH_N
};
static const char *const ph_name[PH_N] = {
    "norm+quant", "q k v",   "q k norm+rope", "attention", "attn merge",
    "wo",         "gate+up", "act",           "down",      "output"};
#define MAX_TS 1024

struct janas_gpu_tok {
    /* its own command buffers, one per number of tokens n (1 .. nb) and
       logits of the last token, of all or of none ([n][all]: 0, 1, 2),
       each recorded once:
       the position is read from the buffer at o_pos, so every block of n
       tokens submits the same commands */
    uint32_t nb;
    /* [n][all][hidden wanted] */
    VkCommandBuffer cb[TOK_MAX_BLOCK + 1][3][2];
    uint32_t cb_nch[TOK_MAX_BLOCK + 1][3]
                   [2]; /* the runs each was recorded for */
    int prof;           /* timestamps in the commands being recorded (n = 1) */
    VkQueryPool qp;     /* timestamps, if profiling */
    uint32_t n_ts;      /* written this token */
    uint8_t ph_of[MAX_TS]; /* the phase each timestamp closes */
    double ph_ns[PH_N];
    double wall_ns; /* submission to the fence, as the CPU sees it */
    double copy_ns; /* the logits copied out */
    uint64_t tokens;
    VkShaderModule sm_nq, sm_qk, sm_at, sm_am;
    VkPipeline p_nq, p_qk, p_at, p_am;
    /* qwen3.5's recurrent layers */
    VkShaderModule sm_gc, sm_gs, sm_gl, sm_gr, sm_gg;
    VkPipeline p_gc, p_gs, p_gl, p_gr, p_gg;
    size_t *par;      /* per layer: its parameters' offsets in buf (PAR_*) */
    struct stage buf; /* every intermediate, one allocation */
    /* offsets into buf, each nb tokens' worth */
    uint32_t nch;       /* the attention's runs of positions, for the cache */
    size_t part_floats; /* room for the runs of a block, floats */
    uint32_t rec_nch;   /* the runs a token of the block being recorded has */
    size_t o_part;
    size_t o_x, o_x1, o_q256, o_q32, o_qkv, o_ywo, o_gu, o_h, o_yd, o_rope,
        o_logits, o_pos;
    /* qwen3.5: the normalized input (f32), q|k|v and z of the recurrent
       layers, their convolution's output, beta and alpha, their raw
       output, the query's gates, the final hidden states */
    size_t o_xnf, o_mix, o_z, o_conv, o_ba, o_att, o_gate, o_hid;
};

/* A recurrent layer's parameters copied into buf: beta then alpha (2 nv x
   dm), the convolution (d_conv x conv_ch), dt, A (nv), the norm (ds) */
enum { PAR_BA, PAR_CONV, PAR_DT, PAR_A, PAR_NORM, PAR_N };

static void sync_range(struct janas_gpu *g, struct janas_gpu_tok *t, size_t off,
                       size_t bytes, int flush);

int janas_gpu_shares_host(const struct janas_gpu *g)
{
    return g && !g->broken && !g->discrete && g->host_import &&
           !getenv("JANAS_GPU_COPY");
}

/* The GPU's address of host memory in one of its regions, 0 if none. */
static VkDeviceAddress addr_of(const struct janas_gpu *g, const void *p,
                               size_t bytes)
{
    const struct region *rg = janas_g_find_region(g, p, bytes);
    return rg ? rg->addr + (VkDeviceAddress)((const uint8_t *)p - rg->host) : 0;
}

static size_t take(size_t *at, size_t bytes)
{
    size_t o = *at;
    *at += (bytes + 255) / 256 * 256;
    return o;
}

static int fail(char *err, size_t len, const char *why)
{
    snprintf(err, len, "%s", why);
    return -1;
}

/* Whether the model's pieces are all where the GPU can read them. */
static int check_model(const struct janas_gpu *g,
                       const struct janas_gpu_tok_model *md, char *err,
                       size_t len)
{
    uint32_t qd = md->n_head * md->hd, kvd = md->n_kv * md->hd;
    if (md->dm % JANAS_QK || qd % JANAS_QK || md->ff % JANAS_QK ||
        md->dm > 8192 || qd > 8192 || md->hd > 512 || md->hd % 16 ||
        md->n_rot > md->hd || md->n_kv == 0 || md->n_head % md->n_kv ||
        md->n_head / md->n_kv * md->hd > ATT_MAXQ ||
        md->n_head / md->n_kv * ATT_RUN > ATT_MAXS)
        return fail(err, len, "a shape the token's shaders do not take");
    if (md->n_rec &&
        (md->ds != 128 || md->conv_ch % 256 || md->d_inner % 256 ||
         md->d_inner != md->nv * md->ds || md->nv > 64 || !md->nkh ||
         md->nv % md->nkh || md->d_conv < 2 || md->d_conv > 8 ||
         md->d_inner > md->dm * 4 ||
         md->conv_ch != 2 * md->nkh * md->ds + md->d_inner ||
         !addr_of(g, md->ssm_state,
                  2 * (size_t)md->n_rec * md->nv * md->ds * md->ds * 2) ||
         !addr_of(g, md->conv_state,
                  2 * (size_t)md->n_rec * (md->d_conv - 1) * md->conv_ch * 4)))
        return fail(err, len,
                    "recurrent layers the token's shaders do not take");
    for (uint32_t l = 0; l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &md->layers[l];
        if (ly->rec) {
            const struct janas_matvec_task *r[6] = {&ly->ssm_qkv, &ly->ssm_z,
                                                    &ly->ssm_out, &ly->gate,
                                                    &ly->up,      &ly->down};
            for (int i = 0; i < 6; i++)
                if (!janas_gpu_can(g, r[i])) {
                    snprintf(err, len,
                             "layer %u: a weight the GPU cannot compute "
                             "(product %d, type %d)",
                             l, i, r[i]->type);
                    return -1;
                }
            if (janas_g_x_in_32(ly->down.type))
                return fail(err, len, "a down matrix in blocks of 32");
            if (!addr_of(g, ly->attn_norm, md->dm * 4) ||
                !addr_of(g, ly->ffn_norm, md->dm * 4))
                return fail(err, len, "a norm out of the GPU's reach");
            continue;
        }
        const struct janas_matvec_task *t[7] = {
            &ly->wq, &ly->wk, &ly->wv, &ly->wo, &ly->gate, &ly->up, &ly->down};
        for (int i = 0; i < 7; i++)
            if (!janas_gpu_can(g, t[i])) {
                snprintf(err, len,
                         "layer %u: a weight the GPU cannot compute (product "
                         "%d, type %d)",
                         l, i, t[i]->type);
                return -1;
            }
        if (md->gated_q && (ly->wq.type == JANAS_Q6_K_P || ly->wq.plane[0]))
            return fail(err, len, "a gated query in planes");
        if (janas_g_x_in_32(ly->down.type))
            return fail(err, len, "a down matrix in blocks of 32");
        size_t cache = (size_t)md->n_kv * md->cap * md->hd;
        if (!addr_of(g, ly->attn_norm, md->dm * 4) ||
            !addr_of(g, ly->ffn_norm, md->dm * 4) ||
            !addr_of(g, ly->q_norm, md->hd * 4) ||
            !addr_of(g, ly->k_norm, md->hd * 4) || !addr_of(g, ly->kc, cache) ||
            !addr_of(g, ly->vc, cache) ||
            !addr_of(g, ly->ks, (size_t)md->n_kv * md->cap * 4) ||
            !addr_of(g, ly->vs, (size_t)md->n_kv * md->cap * 4))
            return fail(err, len,
                        "a norm or the KV cache out of the GPU's "
                        "reach");
    }
    if (!janas_gpu_can(g, &md->output) || !addr_of(g, md->out_norm, md->dm * 4))
        return fail(err, len, "the output head out of the GPU's reach");
    (void)kvd;
    return 0;
}

struct janas_gpu_tok *janas_gpu_tok_create(struct janas_gpu *g,
                                           const struct janas_gpu_tok_model *md,
                                           char *err, size_t err_len)
{
    if (!janas_gpu_shares_host(g)) {
        fail(err, err_len, "the GPU does not read the host's memory in place");
        return NULL;
    }
    if (check_model(g, md, err, err_len) != 0)
        return NULL;
    struct janas_gpu_tok *t = calloc(1, sizeof(*t));
    if (!t) {
        fail(err, err_len, "out of memory");
        return NULL;
    }
    uint32_t qd = md->n_head * md->hd, kvd = md->n_kv * md->hd;
    size_t big = md->dm > qd ? md->dm : qd;
    size_t at = 0;
    /* the attention's runs: per token, query head and run, hd sums, m and
       l; as many tokens a block as they leave room for */
    /* laid out for the runs the positions reached so far have, not for
       the whole cache's (janas_gpu_tok_block_at): a prompt's blocks stay
       long at any context - sized for the cache, 16384 positions left
       Qwen3-4B 30 tokens a block, too few for the fast prompt's tiles */
    t->nch = (md->cap + ATT_RUN - 1) / ATT_RUN;
    size_t full = (size_t)md->n_head * t->nch * (md->hd + 2);
    t->part_floats = TOK_PART_BUDGET / 4 > full ? TOK_PART_BUDGET / 4 : full;
    t->nb = TOK_MAX_BLOCK;
    size_t nb = t->nb;
    t->o_x = take(&at, nb * md->dm * 4);
    t->o_x1 = take(&at, nb * md->dm * 4); /* the residual's other buffer */
    t->o_q256 = take(&at, nb * big / JANAS_QK * 320);
    t->o_q32 = take(&at, nb * big / JANAS_QK * 320);
    t->o_qkv = take(&at, nb * (qd + 2 * kvd) * 4);
    t->o_ywo = take(&at, nb * md->dm * 4);
    t->o_gu = take(&at, nb * md->ff * 2 * 4);
    t->o_h = take(&at, nb * md->ff / JANAS_QK * 320);
    t->o_yd = take(&at, nb * md->dm * 4);
    t->o_rope = take(&at, nb * md->n_rot * 4);
    t->o_logits = take(&at, nb * md->n_vocab * 4);
    t->o_pos = take(&at, 16);
    t->o_part = take(&at, t->part_floats * 4);
    t->o_xnf = take(&at, nb * md->dm * 4);
    t->o_hid = take(&at, nb * md->dm * 4);
    if (md->gated_q)
        t->o_gate = take(&at, nb * qd * 4);
    if (md->n_rec) {
        t->o_mix = take(&at, nb * md->conv_ch * 4);
        t->o_z = take(&at, nb * md->d_inner * 4);
        t->o_conv = take(&at, nb * md->conv_ch * 4);
        t->o_ba = take(&at, nb * 2 * md->nv * 4);
        t->o_att = take(&at, nb * md->d_inner * 4);
        t->par = calloc((size_t)md->n_layer * PAR_N, sizeof(size_t));
        if (!t->par) {
            fail(err, err_len, "out of memory");
            janas_gpu_tok_destroy(g, t);
            return NULL;
        }
        for (uint32_t l = 0; l < md->n_layer; l++) {
            if (!md->layers[l].rec)
                continue;
            size_t *pr = t->par + (size_t)l * PAR_N;
            pr[PAR_BA] = take(&at, 2 * (size_t)md->nv * md->dm * 4);
            pr[PAR_CONV] = take(&at, (size_t)md->d_conv * md->conv_ch * 4);
            pr[PAR_DT] = take(&at, md->nv * 4);
            pr[PAR_A] = take(&at, md->nv * 4);
            pr[PAR_NORM] = take(&at, md->ds * 4);
        }
    }
    /* cached for the host (the logits are read on the CPU: 2.5 ms a token
       from uncached memory on the development laptop), flushed and
       invalidated by hand where it is not coherent */
    if (janas_g_make_stage(g, &t->buf, at, 1) != 0 ||
        janas_g_make_pipeline(g, janas_spv_norm_quant,
                              sizeof(janas_spv_norm_quant), &t->sm_nq, &t->p_nq,
                              0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_qk_rope, sizeof(janas_spv_qk_rope),
                              &t->sm_qk, &t->p_qk, 0) != VK_SUCCESS ||
        janas_g_make_pipeline_sg(
            g, janas_spv_attn_decode, sizeof(janas_spv_attn_decode), &t->sm_at,
            &t->p_at, md->n_head / md->n_kv, att_sg()) == 0 ||
        janas_g_make_pipeline(g, janas_spv_attn_merge,
                              sizeof(janas_spv_attn_merge), &t->sm_am, &t->p_am,
                              0) != VK_SUCCESS ||
        (md->n_rec &&
         (janas_g_make_pipeline(g, janas_spv_gdn_conv,
                                sizeof(janas_spv_gdn_conv), &t->sm_gc, &t->p_gc,
                                0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_cstate,
                                sizeof(janas_spv_gdn_cstate), &t->sm_gs,
                                &t->p_gs, 0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_l2, sizeof(janas_spv_gdn_l2),
                                &t->sm_gl, &t->p_gl, 0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_rec, sizeof(janas_spv_gdn_rec),
                                &t->sm_gr, &t->p_gr, 0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_gate,
                                sizeof(janas_spv_gdn_gate), &t->sm_gg, &t->p_gg,
                                0) != VK_SUCCESS ||
          !g->pipe_dm[0]))) {
        fail(err, err_len, "cannot create the token's buffers or shaders");
        janas_gpu_tok_destroy(g, t);
        return NULL;
    }
    /* the recurrent layers' parameters, copied once */
    for (uint32_t l = 0; md->n_rec && l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &md->layers[l];
        if (!ly->rec)
            continue;
        const size_t *pr = t->par + (size_t)l * PAR_N;
        size_t nvd = (size_t)md->nv * md->dm * 4;
        memcpy(t->buf.map + pr[PAR_BA], ly->ssm_beta, nvd);
        memcpy(t->buf.map + pr[PAR_BA] + nvd, ly->ssm_alpha, nvd);
        memcpy(t->buf.map + pr[PAR_CONV], ly->conv_w,
               (size_t)md->d_conv * md->conv_ch * 4);
        memcpy(t->buf.map + pr[PAR_DT], ly->dt, md->nv * 4);
        memcpy(t->buf.map + pr[PAR_A], ly->ssm_a, md->nv * 4);
        memcpy(t->buf.map + pr[PAR_NORM], ly->ssm_norm, md->ds * 4);
        sync_range(g, t, pr[PAR_BA], pr[PAR_NORM] + md->ds * 4 - pr[PAR_BA], 1);
    }
    const char *pe = getenv("JANAS_GPU_TOKEN_PROFILE");
    if (pe && atoi(pe) > 0 && g->ts_period > 0) {
        VkQueryPoolCreateInfo qi = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = MAX_TS};
        if (g->vkCreateQueryPool(g->dev, &qi, NULL, &t->qp) != VK_SUCCESS)
            t->qp = VK_NULL_HANDLE;
    }
    return t;
}

void janas_gpu_tok_destroy(struct janas_gpu *g, struct janas_gpu_tok *t)
{
    if (!t)
        return;
    if (t->qp) {
        double all = 0;
        for (int i = 0; i < PH_N; i++)
            all += t->ph_ns[i];
        if (t->tokens) {
            fprintf(stderr,
                    "gpu token profile, %llu tokens, %.2f ms a token "
                    "(%.2f ms from submission to fence, %.2f ms copying the "
                    "logits):\n",
                    (unsigned long long)t->tokens, all / t->tokens * 1e-6,
                    t->wall_ns / t->tokens * 1e-6,
                    t->copy_ns / t->tokens * 1e-6);
            for (int i = 0; i < PH_N; i++)
                fprintf(stderr, "  %-14s %7.3f ms %5.1f%%\n", ph_name[i],
                        t->ph_ns[i] / t->tokens * 1e-6,
                        all > 0 ? 100.0 * t->ph_ns[i] / all : 0.0);
        }
        g->vkDestroyQueryPool(g->dev, t->qp, NULL);
    }
    for (uint32_t n = 0; n <= TOK_MAX_BLOCK; n++)
        for (int a = 0; a < 3; a++)
            for (int h = 0; h < 2; h++)
                if (t->cb[n][a][h])
                    g->vkFreeCommandBuffers(g->dev, g->cp, 1, &t->cb[n][a][h]);
    VkPipeline p[9] = {t->p_nq, t->p_qk, t->p_at, t->p_am, t->p_gc,
                       t->p_gs, t->p_gl, t->p_gr, t->p_gg};
    VkShaderModule s[9] = {t->sm_nq, t->sm_qk, t->sm_at, t->sm_am, t->sm_gc,
                           t->sm_gs, t->sm_gl, t->sm_gr, t->sm_gg};
    free(t->par);
    for (int i = 0; i < 9; i++) {
        if (p[i])
            g->vkDestroyPipeline(g->dev, p[i], NULL);
        if (s[i])
            g->vkDestroyShaderModule(g->dev, s[i], NULL);
    }
    if (t->buf.buf)
        g->vkDestroyBuffer(g->dev, t->buf.buf, NULL);
    if (t->buf.mem)
        g->vkFreeMemory(g->dev, t->buf.mem, NULL);
    free(t);
}

/* Bytes [off, off + bytes) of the token's buffer out of the CPU's caches
   (flush) or reloaded from memory (!flush), where it is not coherent. */
static void sync_range(struct janas_gpu *g, struct janas_gpu_tok *t, size_t off,
                       size_t bytes, int flush)
{
    if (t->buf.coherent)
        return;
    VkDeviceSize a = off / g->atom * g->atom;
    VkDeviceSize e = (off + bytes + g->atom - 1) / g->atom * g->atom;
    VkMappedMemoryRange mr = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                              .memory = t->buf.mem,
                              .offset = a,
                              .size =
                                  e >= t->buf.bytes ? VK_WHOLE_SIZE : e - a};
    if (flush)
        g->vkFlushMappedMemoryRanges(g->dev, 1, &mr);
    else
        g->vkInvalidateMappedMemoryRanges(g->dev, 1, &mr);
}

/* A timestamp closing phase ph (with profiling on). */
static void stamp(struct janas_gpu *g, struct janas_gpu_tok *t, int ph)
{
    if (!t->qp || !t->prof || t->n_ts >= MAX_TS)
        return;
    g->vkCmdWriteTimestamp(g->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, t->qp,
                           t->n_ts);
    t->ph_of[t->n_ts++] = (uint8_t)ph;
}

static void bind(struct janas_gpu *g, VkPipeline *bound, VkPipeline p)
{
    if (*bound != p) {
        g->vkCmdBindPipeline(g->cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        *bound = p;
    }
}

static void push(struct janas_gpu *g, const void *pc, size_t bytes)
{
    g->vkCmdPushConstants(g->cb, g->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          (uint32_t)bytes, pc);
}

/* x (+ add, then stored at xo) normalized by w (0: not), quantized into
   q256 and q32: a workgroup per block of 256 values. */
static void rec_norm_quant(struct janas_gpu *g, struct janas_gpu_tok *t,
                           VkPipeline *bound, VkDeviceAddress x,
                           VkDeviceAddress add, VkDeviceAddress w,
                           VkDeviceAddress xo, VkDeviceAddress xn, uint32_t d,
                           float eps, uint32_t n)
{
    struct push_nq pc = {.d = d, .eps = eps};
    janas_g_push_addr(pc.xn, xn);
    janas_g_push_addr(pc.x, x);
    janas_g_push_addr(pc.add, add);
    janas_g_push_addr(pc.w, w);
    janas_g_push_addr(pc.xo, xo);
    janas_g_push_addr(pc.q256, t->buf.addr + t->o_q256);
    janas_g_push_addr(pc.q32, t->buf.addr + t->o_q32);
    bind(g, bound, t->p_nq);
    push(g, &pc, sizeof(pc));
    g->vkCmdDispatch(g->cb, d / JANAS_QK, n, 1); /* y: the vector */
    janas_g_compute_barrier(g);
}

/* A product of the task's weights with n quantized vectors (from byte xoff
   of the q256 or q32 layout, cols / 256 blocks apart), into y (ys floats
   apart; 0: rows). */
static int rec_product(struct janas_gpu *g, struct janas_gpu_tok *t,
                       VkPipeline *bound, const struct janas_matvec_task *w,
                       size_t xoff, VkDeviceAddress y, uint32_t n, size_t ys)
{
    struct janas_matvec_task k = *w;
    k.n_vec = n;
    k.x_stride = 0;
    k.y_stride = 0;
    VkDeviceAddress xa =
        t->buf.addr + xoff + (janas_g_x_in_32(k.type) ? t->o_q32 : t->o_q256);
    return janas_g_record_task(g, bound, &k, k.rows, xa, y, ys ? ys : k.rows);
}

/* The commands of a block of n tokens into g->cb (the caller points it at
   the block's buffer): every layer, then the output head for the tokens
   from first on (none if first is n); the position of the first token
   comes from the buffer.
   A single token takes the layers' GPU copies (JANAS_GPU_REPACK), a block
   the weights as they are (the copies' shaders take one vector). */
/* The runs of ATT_RUN positions npos positions take, rounded up to a power
   of two (so that a prompt's blocks record their commands again only a few
   times), at most the cache's. */
static uint32_t runs_for(const struct janas_gpu_tok *t, uint32_t npos)
{
    uint32_t c = (npos + ATT_RUN - 1) / ATT_RUN, r = 1;
    while (r < c)
        r *= 2;
    return r < t->nch ? r : t->nch;
}

/* A token's runs, floats, at nch runs a head. */
static size_t part_tok(const struct janas_gpu_tok *t,
                       const struct janas_gpu_tok_model *md, uint32_t nch)
{
    (void)t;
    return (size_t)md->n_head * nch * (md->hd + 2);
}

/* A Gated DeltaNet layer (qwen3.5) on the block normalized in q256 / q32
   and XNF: its output into YWO, as model_rec.c's janas_m_rec_layer. */
static int rec_gdn(struct janas_gpu *g, struct janas_gpu_tok *t,
                   const struct janas_gpu_tok_model *md,
                   const struct janas_gpu_tok_layer *ly, uint32_t l,
                   VkPipeline *bound, VkDeviceAddress YWO, uint32_t n)
{
    VkDeviceAddress B = t->buf.addr, MIX = B + t->o_mix, Z = B + t->o_z,
                    CONV = B + t->o_conv, BA = B + t->o_ba, ATT = B + t->o_att,
                    POS = B + t->o_pos;
    const size_t *pr = t->par + (size_t)l * PAR_N;
    uint32_t cc = md->conv_ch;
    /* q|k|v (the convolution's channels) and z */
    if (rec_product(g, t, bound, &ly->ssm_qkv, 0, MIX, n, cc) != 0 ||
        rec_product(g, t, bound, &ly->ssm_z, 0, Z, n, md->d_inner) != 0)
        return -1;
    /* beta then alpha, per token, from the normalized input at f32 */
    struct push_dm pd = {.rows = 2 * md->nv,
                         .cols = md->dm,
                         .nv = n,
                         .ys = 2 * md->nv,
                         .scale = 1.0f};
    janas_g_push_addr(pd.w, B + pr[PAR_BA]);
    janas_g_push_addr(pd.x, B + t->o_xnf);
    janas_g_push_addr(pd.y, BA);
    bind(g, bound, g->pipe_dm[0]);
    push(g, &pd, sizeof(pd));
    g->vkCmdDispatch(g->cb, (2 * md->nv + 63) / 64, (n + 63) / 64, 1);
    janas_g_compute_barrier(g);
    /* the convolution, then the state it leaves, the q and k heads' norms */
    uint32_t hs_c = md->n_rec * (md->d_conv - 1) * cc,
             so_c = ly->slot * (md->d_conv - 1) * cc;
    VkDeviceAddress CS = addr_of(g, md->conv_state, (size_t)hs_c * 8);
    struct push_gconv pc = {.cc = cc, .d = md->d_conv, .hs = hs_c, .so = so_c};
    janas_g_push_addr(pc.mixed, MIX);
    janas_g_push_addr(pc.out, CONV);
    janas_g_push_addr(pc.w, B + pr[PAR_CONV]);
    janas_g_push_addr(pc.state, CS);
    janas_g_push_addr(pc.pos, POS);
    bind(g, bound, t->p_gc);
    push(g, &pc, sizeof(pc));
    g->vkCmdDispatch(g->cb, cc / 256, n, 1);
    janas_g_compute_barrier(g);
    struct push_gcst ps = {
        .cc = cc, .d = md->d_conv, .hs = hs_c, .so = so_c, .n = n};
    janas_g_push_addr(ps.mixed, MIX);
    janas_g_push_addr(ps.state, CS);
    janas_g_push_addr(ps.pos, POS);
    bind(g, bound, t->p_gs);
    push(g, &ps, sizeof(ps));
    g->vkCmdDispatch(g->cb, cc / 256, 1, 1);
    struct push_gl2 pl = {.cc = cc, .eps = md->eps};
    janas_g_push_addr(pl.x, CONV);
    bind(g, bound, t->p_gl);
    push(g, &pl, sizeof(pl));
    g->vkCmdDispatch(g->cb, 2 * md->nkh, n, 1);
    janas_g_compute_barrier(g);
    /* the recurrence, then each head's norm, gate and quantization */
    uint32_t hs_s = md->n_rec * md->nv * md->ds * md->ds,
             so_s = ly->slot * md->nv * md->ds * md->ds;
    struct push_grec pg = {.cc = cc,
                           .nv = md->nv,
                           .nkh = md->nkh,
                           .n = n,
                           .hs = hs_s,
                           .so = so_s,
                           .vmap_mod = (uint32_t)md->vmap_mod,
                           .scale = 1.0f / sqrtf((float)md->ds)};
    janas_g_push_addr(pg.co, CONV);
    janas_g_push_addr(pg.ba, BA);
    janas_g_push_addr(pg.att, ATT);
    janas_g_push_addr(pg.state, addr_of(g, md->ssm_state, (size_t)hs_s * 4));
    janas_g_push_addr(pg.pos, POS);
    janas_g_push_addr(pg.dt, B + pr[PAR_DT]);
    janas_g_push_addr(pg.a, B + pr[PAR_A]);
    bind(g, bound, t->p_gr);
    push(g, &pg, sizeof(pg));
    g->vkCmdDispatch(g->cb, md->nv * (md->ds / 16), 1, 1);
    janas_g_compute_barrier(g);
    struct push_ggate pq = {.d = md->d_inner, .eps = md->eps};
    janas_g_push_addr(pq.att, ATT);
    janas_g_push_addr(pq.z, Z);
    janas_g_push_addr(pq.w, B + pr[PAR_NORM]);
    janas_g_push_addr(pq.q256, B + t->o_q256);
    janas_g_push_addr(pq.q32, B + t->o_q32);
    bind(g, bound, t->p_gg);
    push(g, &pq, sizeof(pq));
    g->vkCmdDispatch(g->cb, md->d_inner / JANAS_QK, n, 1);
    janas_g_compute_barrier(g);
    if (rec_product(g, t, bound, &ly->ssm_out, 0, YWO, n, 0) != 0)
        return -1;
    janas_g_compute_barrier(g);
    return 0;
}

static int record(struct janas_gpu *g, struct janas_gpu_tok *t,
                  const struct janas_gpu_tok_model *md, uint32_t n,
                  uint32_t first, int hid)
{
    uint32_t qd = md->n_head * md->hd, kvd = md->n_kv * md->hd;
    size_t qs = qd + 2 * kvd;
    const struct janas_gpu_tok_layer *layers =
        n > 1 && md->blayers ? md->blayers : md->layers;
    const struct janas_matvec_task *output =
        n > 1 && md->blayers ? &md->boutput : &md->output;
    /* the residual: read from X, the sum with a layer's output written to
       XN, the two swapped (every block starts from o_x, the same commands) */
    VkDeviceAddress B = t->buf.addr, X = B + t->o_x, XN = B + t->o_x1, SW,
                    QKV = B + t->o_qkv, YWO = B + t->o_ywo, GU = B + t->o_gu,
                    H = B + t->o_h, YD = B + t->o_yd, ROPE = B + t->o_rope,
                    POS = B + t->o_pos;
    if (g->vkResetCommandBuffer(g->cb, 0) != VK_SUCCESS)
        return -1;
    VkCommandBufferBeginInfo bb = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = 0};
    if (g->vkBeginCommandBuffer(g->cb, &bb) != VK_SUCCESS)
        return -1;
    VkPipeline bound = VK_NULL_HANDLE;
    size_t cache = (size_t)md->n_kv * md->cap * md->hd;
    t->prof = n == 1;
    if (t->qp && t->prof) {
        t->n_ts = 0;
        g->vkCmdResetQueryPool(g->cb, t->qp, 0, MAX_TS);
        stamp(g, t, PH_N); /* the start */
    }
    for (uint32_t l = 0; l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &layers[l];
        /* the previous layer's feed-forward added, then this one's norm */
        rec_norm_quant(g, t, &bound, X, l ? YD : 0,
                       addr_of(g, ly->attn_norm, md->dm * 4), l ? XN : 0,
                       ly->rec ? B + t->o_xnf : 0, md->dm, md->eps, n);
        if (l)
            SW = X, X = XN, XN = SW;
        stamp(g, t, PH_NORM);
        if (ly->rec) {
            if (rec_gdn(g, t, md, ly, l, &bound, YWO, n) != 0)
                return -1;
            goto ffn;
        }
        /* q, k and v of each token side by side, qs floats a token; a
           gated query (qwen3.5) head by head, its gates apart */
        if (md->gated_q) {
            size_t rb = (size_t)(md->dm / JANAS_QK) *
                        janas_qtype_block_size(ly->wq.type);
            for (uint32_t h = 0; h < md->n_head; h++) {
                struct janas_matvec_task q = ly->wq, gt = ly->wq;
                q.rows = gt.rows = md->hd;
                q.w = (const uint8_t *)ly->wq.w + (size_t)2 * h * md->hd * rb;
                gt.w = (const uint8_t *)q.w + (size_t)md->hd * rb;
                if (rec_product(g, t, &bound, &q, 0,
                                QKV + (size_t)h * md->hd * 4, n, qs) != 0 ||
                    rec_product(g, t, &bound, &gt, 0,
                                B + t->o_gate + (size_t)h * md->hd * 4, n,
                                qd) != 0)
                    return -1;
            }
        } else if (rec_product(g, t, &bound, &ly->wq, 0, QKV, n, qs) != 0)
            return -1;
        if (rec_product(g, t, &bound, &ly->wk, 0, QKV + (size_t)qd * 4, n,
                        qs) != 0 ||
            rec_product(g, t, &bound, &ly->wv, 0, QKV + (size_t)(qd + kvd) * 4,
                        n, qs) != 0)
            return -1;
        janas_g_compute_barrier(g);
        stamp(g, t, PH_QKV);
        struct push_qk pq = {.n_head = md->n_head,
                             .n_kv = md->n_kv,
                             .hd = md->hd,
                             .n_rot = md->n_rot,
                             .cap = md->cap,
                             .qs = (uint32_t)qs,
                             .eps = md->eps};
        janas_g_push_addr(pq.pos, POS);
        janas_g_push_addr(pq.qkv, QKV);
        janas_g_push_addr(pq.qn, addr_of(g, ly->q_norm, md->hd * 4));
        janas_g_push_addr(pq.kn, addr_of(g, ly->k_norm, md->hd * 4));
        janas_g_push_addr(pq.rope, ROPE);
        janas_g_push_addr(pq.kc, addr_of(g, ly->kc, cache));
        janas_g_push_addr(pq.vc, addr_of(g, ly->vc, cache));
        janas_g_push_addr(pq.ks, addr_of(g, ly->ks, cache / md->hd * 4));
        janas_g_push_addr(pq.vs, addr_of(g, ly->vs, cache / md->hd * 4));
        bind(g, &bound, t->p_qk);
        push(g, &pq, sizeof(pq));
        g->vkCmdDispatch(g->cb, md->n_head + md->n_kv, n, 1);
        janas_g_compute_barrier(g);
        stamp(g, t, PH_ROPE);
        /* the attention: runs of positions per KV head, then joined; token
           j of the block sees positions 0 .. pos + j, the block's own keys
           and values written above */
        struct push_at pa = {.n_head = md->n_head,
                             .n_kv = md->n_kv,
                             .hd = md->hd,
                             .cap = md->cap,
                             .nch = t->rec_nch,
                             .scale = md->attn_scale,
                             .qs = (uint32_t)qs,
                             .ps = (uint32_t)part_tok(t, md, t->rec_nch)};
        janas_g_push_addr(pa.pos, POS);
        janas_g_push_addr(pa.q, QKV);
        janas_g_push_addr(pa.kc, addr_of(g, ly->kc, cache));
        janas_g_push_addr(pa.vc, addr_of(g, ly->vc, cache));
        janas_g_push_addr(pa.ks, addr_of(g, ly->ks, cache / md->hd * 4));
        janas_g_push_addr(pa.vs, addr_of(g, ly->vs, cache / md->hd * 4));
        janas_g_push_addr(pa.part, B + t->o_part);
        bind(g, &bound, t->p_at);
        push(g, &pa, sizeof(pa));
        g->vkCmdDispatch(g->cb, md->n_kv * t->rec_nch, n, 1);
        janas_g_compute_barrier(g);
        stamp(g, t, PH_ATTN);
        /* the runs joined and the output quantized for wo, a workgroup a
           block of 256 */
        struct push_am pm = {.hd = md->hd,
                             .nch = t->rec_nch,
                             .ps = (uint32_t)part_tok(t, md, t->rec_nch),
                             .gs = qd};
        janas_g_push_addr(pm.gate, md->gated_q ? B + t->o_gate : 0);
        janas_g_push_addr(pm.part, B + t->o_part);
        janas_g_push_addr(pm.q256, B + t->o_q256);
        janas_g_push_addr(pm.q32, B + t->o_q32);
        janas_g_push_addr(pm.pos, POS);
        bind(g, &bound, t->p_am);
        push(g, &pm, sizeof(pm));
        g->vkCmdDispatch(g->cb, qd / JANAS_QK, n, 1);
        janas_g_compute_barrier(g);
        stamp(g, t, PH_MERGE);
        /* the output projection */
        if (rec_product(g, t, &bound, &ly->wo, 0, YWO, n, 0) != 0)
            return -1;
        janas_g_compute_barrier(g);
        stamp(g, t, PH_WO);
    ffn:
        rec_norm_quant(g, t, &bound, X, YWO,
                       addr_of(g, ly->ffn_norm, md->dm * 4), XN, 0, md->dm,
                       md->eps, n);
        SW = X, X = XN, XN = SW;
        stamp(g, t, PH_NORM);
        /* the feed-forward: gate, up (the n tokens' gates, then their ups),
           SiLU(gate) up quantized, down */
        VkDeviceAddress UP = GU + (size_t)n * md->ff * 4;
        if (rec_product(g, t, &bound, &ly->gate, 0, GU, n, 0) != 0 ||
            rec_product(g, t, &bound, &ly->up, 0, UP, n, 0) != 0)
            return -1;
        janas_g_compute_barrier(g);
        stamp(g, t, PH_GATEUP);
        struct push_ac ac = {.blocks = n * md->ff / JANAS_QK};
        janas_g_push_addr(ac.g, GU);
        janas_g_push_addr(ac.u, UP);
        janas_g_push_addr(ac.x, H);
        bind(g, &bound, g->pipea);
        push(g, &ac, sizeof(ac));
        g->vkCmdDispatch(g->cb, n * md->ff / JANAS_QK, 1, 1);
        janas_g_compute_barrier(g);
        stamp(g, t, PH_ACT);
        struct janas_matvec_task dn = ly->down;
        if (janas_g_record_task(
                g, &bound,
                &(struct janas_matvec_task){.type = dn.type,
                                            .w = dn.w,
                                            .plane = {dn.plane[0], dn.plane[1]},
                                            .rows = dn.rows,
                                            .cols = dn.cols,
                                            .n_vec = n},
                dn.rows, H, YD, dn.rows) != 0)
            return -1;
        janas_g_compute_barrier(g);
        stamp(g, t, PH_DOWN);
    }
    /* the last feed-forward added, the final norm, the output head for
       tokens first .. n - 1 (none: the residual left in X, the next block's
       keys and values already in the cache) */
    if (first < n || hid) {
        rec_norm_quant(g, t, &bound, X, YD,
                       addr_of(g, md->out_norm, md->dm * 4), 0,
                       hid ? B + t->o_hid : 0, md->dm, md->eps, n);
    }
    if (first < n) {
        if (rec_product(g, t, &bound, output,
                        (size_t)first * (md->dm / JANAS_QK) * 320,
                        B + t->o_logits, n - first, 0) != 0)
            return -1;
    }
    stamp(g, t, PH_OUT);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    g->vkCmdPipelineBarrier(g->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0,
                            NULL);
    if (g->vkEndCommandBuffer(g->cb) != VK_SUCCESS)
        return -1;
    return 0;
}

uint32_t janas_gpu_tok_max_block(const struct janas_gpu_tok *t)
{
    return t ? t->nb : 0;
}

uint32_t janas_gpu_tok_block_at(const struct janas_gpu_tok *t,
                                const struct janas_gpu_tok_model *md,
                                uint32_t pos)
{
    if (!t)
        return 0;
    size_t n = t->part_floats / part_tok(t, md, runs_for(t, pos + t->nb));
    return n < 1 ? 1 : n > t->nb ? t->nb : (uint32_t)n;
}

int janas_gpu_tok_run(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md, const float *x,
                      const float *cs, const float *sn, uint32_t pos,
                      uint32_t n, int all, float *logits, float *hidden,
                      uint32_t src, uint32_t dst)
{
    if (!t || g->busy || g->broken || n < 1 || n > t->nb || pos + n > md->cap ||
        (md->n_rec && n + 1 < md->d_conv))
        return -1;
    uint32_t nch = runs_for(t, pos + n);
    if ((size_t)n * part_tok(t, md, nch) > t->part_floats)
        return -1;
    uint32_t half = md->n_rot / 2, first = all == 1 ? 0 : all == 2 ? n : n - 1;
    memcpy(t->buf.map + t->o_x, x, (size_t)n * md->dm * sizeof(float));
    for (uint32_t j = 0; j < n; j++) {
        uint8_t *r = t->buf.map + t->o_rope + (size_t)j * md->n_rot * 4;
        memcpy(r, cs + (size_t)j * half, half * sizeof(float));
        memcpy(r + half * sizeof(float), sn + (size_t)j * half,
               half * sizeof(float));
    }
    /* the position, then the halves of the recurrent state to read and to
       write, read by the shaders at run time */
    uint32_t pw[3] = {pos, src, dst};
    memcpy(t->buf.map + t->o_pos, pw, sizeof(pw));
    sync_range(g, t, t->o_x, (size_t)n * md->dm * sizeof(float), 1);
    sync_range(g, t, t->o_rope, (size_t)n * md->n_rot * sizeof(float), 1);
    sync_range(g, t, t->o_pos, sizeof(pw), 1);
    int ai = all == 1 && n == 1 ? 0 : all, hi = hidden != NULL;
    VkCommandBuffer *cb = &t->cb[n][ai][hi];
    if (*cb && t->cb_nch[n][ai][hi] != nch) {
        /* recorded for other runs: again (a few times in a context) */
        g->vkFreeCommandBuffers(g->dev, g->cp, 1, cb);
        *cb = VK_NULL_HANDLE;
    }
    if (!*cb) {
        VkCommandBufferAllocateInfo ca = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = g->cp,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};
        if (g->vkAllocateCommandBuffers(g->dev, &ca, cb) != VK_SUCCESS) {
            *cb = VK_NULL_HANDLE;
            return -1;
        }
        VkCommandBuffer keep = g->cb;
        g->cb = *cb; /* the helpers record into g->cb */
        t->rec_nch = nch;
        t->cb_nch[n][ai][hi] = nch;
        int rc = record(g, t, md, n, first, hi);
        g->cb = keep;
        if (rc != 0) {
            g->vkFreeCommandBuffers(g->dev, g->cp, 1, cb);
            *cb = VK_NULL_HANDLE;
            return -1;
        }
    }
    VkSubmitInfo sub = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1,
                        .pCommandBuffers = cb};
    struct timespec w0, w1;
    clock_gettime(CLOCK_MONOTONIC, &w0);
    pthread_mutex_lock(&g->qlock);
    VkResult r = g->vkQueueSubmit(g->queue, 1, &sub, g->fence);
    pthread_mutex_unlock(&g->qlock);
    if (r == VK_SUCCESS)
        r = g->vkWaitForFences(g->dev, 1, &g->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        g->broken = 1;
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &w1);
    g->vkResetFences(g->dev, 1, &g->fence);
    size_t lb = (size_t)(n - first) * md->n_vocab * 4;
    if (lb) {
        sync_range(g, t, t->o_logits, lb, 0);
        memcpy(logits, t->buf.map + t->o_logits, lb);
    }
    if (hidden) {
        size_t hb = (size_t)n * md->dm * 4;
        sync_range(g, t, t->o_hid, hb, 0);
        memcpy(hidden, t->buf.map + t->o_hid, hb);
    }
    struct timespec w2;
    clock_gettime(CLOCK_MONOTONIC, &w2);
    if (t->qp && n == 1 && t->n_ts > 1) {
        uint64_t ts[MAX_TS];
        if (g->vkGetQueryPoolResults(g->dev, t->qp, 0, t->n_ts, sizeof(ts), ts,
                                     sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            for (uint32_t i = 1; i < t->n_ts; i++)
                t->ph_ns[t->ph_of[i]] +=
                    (double)(ts[i] - ts[i - 1]) * g->ts_period;
            t->tokens++;
            t->wall_ns += (double)(w1.tv_sec - w0.tv_sec) * 1e9 +
                          (double)(w1.tv_nsec - w0.tv_nsec);
            t->copy_ns += (double)(w2.tv_sec - w1.tv_sec) * 1e9 +
                          (double)(w2.tv_nsec - w1.tv_nsec);
        }
    }
    return 0;
}

#else /* no GPU in this build */

struct janas_gpu_tok *janas_gpu_tok_create(struct janas_gpu *g,
                                           const struct janas_gpu_tok_model *md,
                                           char *err, size_t err_len)
{
    (void)g;
    (void)md;
    snprintf(err, err_len, "no GPU in this build");
    return NULL;
}

void janas_gpu_tok_destroy(struct janas_gpu *g, struct janas_gpu_tok *t)
{
    (void)g;
    (void)t;
}

uint32_t janas_gpu_tok_max_block(const struct janas_gpu_tok *t)
{
    (void)t;
    return 0;
}

uint32_t janas_gpu_tok_block_at(const struct janas_gpu_tok *t,
                                const struct janas_gpu_tok_model *md,
                                uint32_t pos)
{
    (void)t;
    (void)md;
    (void)pos;
    return 0;
}

int janas_gpu_tok_run(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md, const float *x,
                      const float *cs, const float *sn, uint32_t pos,
                      uint32_t n, int all, float *logits, float *hidden,
                      uint32_t src, uint32_t dst)
{
    (void)n;
    (void)all;
    (void)g;
    (void)t;
    (void)md;
    (void)x;
    (void)cs;
    (void)sn;
    (void)pos;
    (void)logits;
    (void)hidden;
    (void)src;
    (void)dst;
    return -1;
}

int janas_gpu_shares_host(const struct janas_gpu *g)
{
    (void)g;
    return 0;
}

#endif
