/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token.c - a whole token on the GPU with one submission (see
 * gpu_token.h).
 */
#include "gpu_token.h"

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

/* The push constants of each shader, laid out as its GLSL block. */
struct push_nq {
    uint32_t x[2], add[2], w[2], xo[2], q256[2], q32[2];
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
    uint32_t hd, nch, ps, pad;
    uint32_t pos[2]; /* 8-byte aligned, as the GLSL block puts it */
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
/* The most tokens a block takes (the model's own blocks: model.h
   JANAS_LLM_MAX_BLOCK), and the memory the attention's runs may take for
   them. */
#define TOK_MAX_BLOCK 64
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
    VkCommandBuffer cb[TOK_MAX_BLOCK + 1][3];
    int prof;       /* timestamps in the commands being recorded (n = 1) */
    VkQueryPool qp; /* timestamps, if profiling */
    uint32_t n_ts;  /* written this token */
    uint8_t ph_of[MAX_TS]; /* the phase each timestamp closes */
    double ph_ns[PH_N];
    double wall_ns; /* submission to the fence, as the CPU sees it */
    double copy_ns; /* the logits copied out */
    uint64_t tokens;
    VkShaderModule sm_nq, sm_qk, sm_at, sm_am;
    VkPipeline p_nq, p_qk, p_at, p_am;
    struct stage buf; /* every intermediate, one allocation */
    /* offsets into buf, each nb tokens' worth */
    uint32_t nch;    /* the attention's runs of positions, for the cache */
    size_t part_tok; /* a token's runs, floats */
    size_t o_part;
    size_t o_x, o_x1, o_q256, o_q32, o_qkv, o_ywo, o_gu, o_h, o_yd, o_rope,
        o_logits, o_pos;
};

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
    for (uint32_t l = 0; l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &md->layers[l];
        const struct janas_matvec_task *t[7] = {
            &ly->wq, &ly->wk, &ly->wv, &ly->wo, &ly->gate, &ly->up, &ly->down};
        for (int i = 0; i < 7; i++)
            if (!janas_gpu_can(g, t[i]))
                return fail(err, len, "a weight the GPU cannot compute");
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
    t->nch = (md->cap + ATT_RUN - 1) / ATT_RUN;
    t->part_tok = (size_t)md->n_head * t->nch * (md->hd + 2);
    size_t nb = TOK_PART_BUDGET / (t->part_tok * 4);
    t->nb = nb < 1 ? 1 : nb > TOK_MAX_BLOCK ? TOK_MAX_BLOCK : (uint32_t)nb;
    nb = t->nb;
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
    t->o_part = take(&at, nb * t->part_tok * 4);
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
                              0) != VK_SUCCESS) {
        fail(err, err_len, "cannot create the token's buffers or shaders");
        janas_gpu_tok_destroy(g, t);
        return NULL;
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
            if (t->cb[n][a])
                g->vkFreeCommandBuffers(g->dev, g->cp, 1, &t->cb[n][a]);
    VkPipeline p[4] = {t->p_nq, t->p_qk, t->p_at, t->p_am};
    VkShaderModule s[4] = {t->sm_nq, t->sm_qk, t->sm_at, t->sm_am};
    for (int i = 0; i < 4; i++) {
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
                           VkDeviceAddress xo, uint32_t d, float eps,
                           uint32_t n)
{
    struct push_nq pc = {.d = d, .eps = eps};
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
static int record(struct janas_gpu *g, struct janas_gpu_tok *t,
                  const struct janas_gpu_tok_model *md, uint32_t n,
                  uint32_t first)
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
                       md->dm, md->eps, n);
        if (l)
            SW = X, X = XN, XN = SW;
        stamp(g, t, PH_NORM);
        /* q, k and v of each token side by side, qs floats a token */
        if (rec_product(g, t, &bound, &ly->wq, 0, QKV, n, qs) != 0 ||
            rec_product(g, t, &bound, &ly->wk, 0, QKV + (size_t)qd * 4, n,
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
                             .nch = t->nch,
                             .scale = md->attn_scale,
                             .qs = (uint32_t)qs,
                             .ps = (uint32_t)t->part_tok};
        janas_g_push_addr(pa.pos, POS);
        janas_g_push_addr(pa.q, QKV);
        janas_g_push_addr(pa.kc, addr_of(g, ly->kc, cache));
        janas_g_push_addr(pa.vc, addr_of(g, ly->vc, cache));
        janas_g_push_addr(pa.ks, addr_of(g, ly->ks, cache / md->hd * 4));
        janas_g_push_addr(pa.vs, addr_of(g, ly->vs, cache / md->hd * 4));
        janas_g_push_addr(pa.part, B + t->o_part);
        bind(g, &bound, t->p_at);
        push(g, &pa, sizeof(pa));
        g->vkCmdDispatch(g->cb, md->n_kv * t->nch, n, 1);
        janas_g_compute_barrier(g);
        stamp(g, t, PH_ATTN);
        /* the runs joined and the output quantized for wo, a workgroup a
           block of 256 */
        struct push_am pm = {
            .hd = md->hd, .nch = t->nch, .ps = (uint32_t)t->part_tok};
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
        rec_norm_quant(g, t, &bound, X, YWO,
                       addr_of(g, ly->ffn_norm, md->dm * 4), XN, md->dm,
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
    if (first < n) {
        rec_norm_quant(g, t, &bound, X, YD,
                       addr_of(g, md->out_norm, md->dm * 4), 0, md->dm, md->eps,
                       n);
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

int janas_gpu_tok_run(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md, const float *x,
                      const float *cs, const float *sn, uint32_t pos,
                      uint32_t n, int all, float *logits)
{
    if (!t || g->busy || g->broken || n < 1 || n > t->nb || pos + n > md->cap)
        return -1;
    uint32_t half = md->n_rot / 2, first = all == 1 ? 0 : all == 2 ? n : n - 1;
    memcpy(t->buf.map + t->o_x, x, (size_t)n * md->dm * sizeof(float));
    for (uint32_t j = 0; j < n; j++) {
        uint8_t *r = t->buf.map + t->o_rope + (size_t)j * md->n_rot * 4;
        memcpy(r, cs + (size_t)j * half, half * sizeof(float));
        memcpy(r + half * sizeof(float), sn + (size_t)j * half,
               half * sizeof(float));
    }
    memcpy(t->buf.map + t->o_pos, &pos, sizeof(pos));
    sync_range(g, t, t->o_x, (size_t)n * md->dm * sizeof(float), 1);
    sync_range(g, t, t->o_rope, (size_t)n * md->n_rot * sizeof(float), 1);
    sync_range(g, t, t->o_pos, sizeof(pos), 1);
    VkCommandBuffer *cb = &t->cb[n][all == 1 && n == 1 ? 0 : all];
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
        int rc = record(g, t, md, n, first);
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

int janas_gpu_tok_run(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md, const float *x,
                      const float *cs, const float *sn, uint32_t pos,
                      uint32_t n, int all, float *logits)
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
    return -1;
}

int janas_gpu_shares_host(const struct janas_gpu *g)
{
    (void)g;
    return 0;
}

#endif
