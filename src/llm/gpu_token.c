/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token.c - a whole token on the GPU with one submission (see
 * gpu_token.h): its buffers and shaders, the checks, the commands of a
 * block and their submission. The commands' pieces are in gpu_token_rec.c,
 * a mixture's in gpu_token_moe.c (gpu_token_impl.h).
 */
#include "gpu_token.h"

#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gpu_token_impl.h"
#include "quant.h"

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)

#include "norm_quant.h"  /* janas_spv_norm_quant */
#include "qk_rope.h"     /* janas_spv_qk_rope */
#include "attn_decode.h" /* janas_spv_attn_decode */
#include "attn_merge.h"  /* janas_spv_attn_merge */
#include "ple_act.h"     /* janas_spv_ple_act */
#include "gdn_conv.h"    /* janas_spv_gdn_conv */
#include "gdn_cstate.h"  /* janas_spv_gdn_cstate */
#include "gdn_l2.h"      /* janas_spv_gdn_l2 */
#include "gdn_rec.h"     /* janas_spv_gdn_rec */
#include "gdn_gate.h"    /* janas_spv_gdn_gate */
#include "gdn_ba.h"      /* janas_spv_gdn_ba */
#include "gdn_act.h"     /* janas_spv_gdn_act */
#include "moe_topk.h"    /* janas_spv_moe_topk */
#include "moe_gather.h"  /* janas_spv_moe_gather */
#include "moe_combine.h" /* janas_spv_moe_combine */
#include "moe_plan.h"    /* janas_spv_moe_plan */
#include "silu_mul.h"    /* janas_spv_silu_mul */

/* The attention's first step in subgroups of 16 where the size may be
   chosen (JANAS_GPU_ATT_SG=n another): twice the lanes a hardware thread
   of the integrated GPU, its latencies better hidden - 11.7 -> 9.9 ms a
   token at 4000 positions - where the products lose with them. */
static uint32_t att_sg(void)
{
    const char *e = getenv("JANAS_GPU_ATT_SG");
    return e ? (uint32_t)atoi(e) : 16;
}
static const char *const ph_name[PH_N] = {"norm+quant",    "q k v",
                                          "q k norm+rope", "attention",
                                          "attn merge",    "wo",
                                          "gate+up",       "act",
                                          "down",          "output",
                                          "ssm q k v z",   "ssm beta+alpha",
                                          "ssm conv",      "ssm recurrence",
                                          "ssm gate",      "router",
                                          "experts' sum",  "between segs"};
int janas_gpu_shares_host(const struct janas_gpu *g)
{
    return g && !g->broken && !g->discrete && g->host_import &&
           !janas_g_copy_asked();
}

size_t janas_gpu_local_room(const struct janas_gpu *g)
{
    if (!g || g->broken || !g->discrete)
        return 0;
    /* the largest heap of the card's own memory, 85% of it: the desktop
       and other programs keep the rest */
    VkDeviceSize heap = 0;
    for (uint32_t i = 0; i < g->mp.memoryHeapCount; i++)
        if ((g->mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            g->mp.memoryHeaps[i].size > heap)
            heap = g->mp.memoryHeaps[i].size;
    VkDeviceSize keep = heap / 100 * 85;
    return keep > g->local_bytes ? (size_t)(keep - g->local_bytes) : 0;
}

/* The GPU's address of host memory in one of its regions, 0 if none. */
VkDeviceAddress janas_gt_addr_of(const struct janas_gpu *g, const void *p,
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

/* Whether a layer's feed-forward is one the GPU computes: a dense one's
   weights where it reads them, a mixture's experts of the types its
   *_mmid shaders take and its shared expert's weights. NULL, or why not. */
static const char *check_ffn(const struct janas_gpu *g,
                             const struct janas_gpu_tok_model *md,
                             const struct janas_gpu_tok_layer *ly)
{
    if (!md->n_expert) {
        const struct janas_matvec_task *f[3] = {&ly->gate, &ly->up, &ly->down};
        for (int i = 0; i < 3; i++)
            if (!janas_gpu_can(g, f[i]))
                return "a feed-forward weight the GPU cannot compute";
        return NULL;
    }
    int gt = ly->gate.type, dt = ly->down.type;
    /* Q2_0 (qwen4exp's): b32_mmid.comp, rows of 640 too */
    int q20 = g->pipe_idb[2] != VK_NULL_HANDLE;
    if (!(gt == JANAS_Q4_K || (gt == JANAS_MXFP4 && g->pipe_idmx) ||
          (gt == JANAS_Q3_K && g->pipe_id3) || (gt == JANAS_Q2_0 && q20)) ||
        ly->up.type != gt ||
        !(dt == JANAS_Q4_K || dt == JANAS_Q6_K || dt == JANAS_Q6_K_P ||
          (dt == JANAS_MXFP4 && g->pipe_idmx) ||
          (dt == JANAS_Q3_K && g->pipe_id3) || (dt == JANAS_Q2_0 && q20) ||
          (dt == JANAS_Q5_1 && g->pipe_mvid[6]) ||
          (dt == JANAS_Q8_0 && g->pipe_mvid[7])) ||
        (md->eff % JANAS_QK && dt != JANAS_Q2_0 && dt != JANAS_Q5_1 &&
         dt != JANAS_Q8_0) ||
        ly->gate.rows != md->eff || ly->up.rows != md->eff ||
        ly->gate.cols != md->dm || ly->up.cols != md->dm ||
        ly->down.rows != md->dm || ly->down.cols != md->eff)
        return "experts of a type or shape the GPU's shaders do not take";
    if (!ly->router16)
        return "a mixture's layer without its router";
    if (ly->pre_norm2) { /* Gemma 4's: its dense branch, norms, scales */
        const struct janas_matvec_task *d[3] = {&ly->dn_gate, &ly->dn_up,
                                                &ly->dn_down};
        for (int i = 0; i < 3; i++)
            if (!janas_gpu_can(g, d[i]))
                return "a dense branch's weight the GPU cannot compute";
        if (ly->dn_gate.rows > md->ff ||
            !janas_gt_addr_of(g, ly->pre_norm2, md->dm * 4) ||
            !janas_gt_addr_of(g, ly->post_norm1, md->dm * 4) ||
            !janas_gt_addr_of(g, ly->post_norm2, md->dm * 4) ||
            !janas_gt_addr_of(g, ly->post_ffn, md->dm * 4) ||
            !janas_gt_addr_of(g, ly->rscale, md->dm * 4) ||
            (ly->exp_scale &&
             !janas_gt_addr_of(g, ly->exp_scale, md->n_expert * 4)))
            return "a mixture's norms out of the GPU's reach";
    }
    if (ly->sh_gate_w.type) {
        const struct janas_matvec_task *s[3] = {&ly->sh_gate_w, &ly->sh_up,
                                                &ly->sh_down};
        for (int i = 0; i < 3; i++)
            if (!janas_gpu_can(g, s[i]))
                return "a shared expert's weight the GPU cannot compute";
        if (ly->sh_gate_w.rows != md->ff || ly->sh_up.rows != md->ff ||
            ly->sh_down.cols != md->ff || ly->sh_down.rows != md->dm)
            return "a shared expert of another shape";
    }
    return NULL;
}

/* qwen4exp's streams: every mix's matrices where the GPU reads them, bf16
   (down and up Q8_0 with JANAS_HC_MIX=q8), the n-gram layer's pieces, the ring
   of the conv's inputs and the attention's inputs kept for the indexer. */
static int check_hc(const struct janas_gpu *g,
                    const struct janas_gpu_tok_model *md, char *err, size_t len)
{
    size_t hd = (size_t)md->hc * md->dm, r = md->hc_rank;
    if (md->hc < 2 || md->hc > 16 || r % 32 || hd % 256 || !g->pipe_dm[1] ||
        (md->hc_q8 && !g->pipe_dm[2]))
        return fail(err, len, "streams of a shape the shaders do not take");
    /* down and up: bf16, or Q8_0 */
    size_t mb = md->hc_q8 ? r * hd / 32 * 34 : r * hd * 2;
    for (uint32_t l = 0; l <= md->n_layer; l++)
        for (int k = 0; k < (l < md->n_layer ? 2 : 1); k++) {
            const struct janas_gpu_tok_layer *ly =
                l < md->n_layer ? &md->layers[l] : NULL;
            const float *nw = ly ? ly->hc_norm[k] : md->hco_norm;
            const uint16_t *dw = ly ? ly->hc_down[k] : md->hco_down,
                           *uw = ly ? ly->hc_up[k] : md->hco_up,
                           *iw = ly ? ly->hc_inject[k] : NULL;
            if (!janas_gt_addr_of(g, nw, hd * 4) ||
                !janas_gt_addr_of(g, dw, mb) || !janas_gt_addr_of(g, uw, mb) ||
                (ly && !janas_gt_addr_of(g, iw, md->hc * hd * 2)))
                return fail(err, len,
                            "a mix of the streams not in bf16 where the GPU "
                            "reads it");
        }
    if (md->ng_layer < md->n_layer) {
        const struct janas_gpu_tok_layer *ly = &md->layers[md->ng_layer];
        if (!md->ng_emb || md->ng_kernel == 0 || md->ng_kernel > 8 ||
            !janas_gpu_can(g, &ly->ng_key) ||
            !janas_gt_addr_of(g, ly->ng_value, (size_t)md->dm * md->dm * 2) ||
            !janas_gt_addr_of(g, ly->ng_nk, hd * 4) ||
            !janas_gt_addr_of(g, ly->ng_nq, hd * 4) ||
            !janas_gt_addr_of(g, ly->ng_nc, hd * 4) || !ly->ng_conv ||
            !janas_gt_addr_of(g, md->ng_hist, md->ng_cap * hd * 4))
            return fail(err, len, "n-gram embeddings out of the GPU's reach");
    }
    /* the sparse attention's indexer: its queries in the score shader's
       shared memory, a head's width a workgroup's, the positions of a
       token within the cache's runs */
    if (md->ix_heads) {
        uint32_t d = md->ix_dim, r = md->ix_ratio;
        if (!md->att_in || !md->ix_rope || d > 256 || d % 8 ||
            md->ix_heads * d > 2048 || md->n_rot > d || md->n_rot % 2 ||
            r == 0 || r > 256 || md->ix_top % r || md->ix_top == 0)
            return fail(err, len,
                        "a sparse attention of a shape the shaders do not "
                        "take");
        for (uint32_t l = 0; l < md->n_layer; l++) {
            const struct janas_gpu_tok_layer *ly = &md->layers[l];
            if (ly->rec)
                continue;
            if (!ly->ix_q || !ly->ix_k || !ly->ix_qn || !ly->ix_kn ||
                !janas_gt_addr_of(g, ly->ix_q,
                                  (size_t)md->ix_heads * d * md->dm * 2) ||
                !janas_gt_addr_of(g, ly->ix_k, (size_t)d * md->dm * 2) ||
                !janas_gt_addr_of(g, ly->ix_qn, (size_t)d * 4) ||
                !janas_gt_addr_of(g, ly->ix_kn, (size_t)d * 4) ||
                !janas_gt_addr_of(g, ly->ix_raw, (size_t)md->cap * d * 2) ||
                !janas_gt_addr_of(g, ly->ix_pool,
                                  (size_t)((md->cap + r - 1) / r) * d * 4))
                return fail(err, len,
                            "the sparse attention's indexer out of the GPU's "
                            "reach");
        }
    }
    return 0;
}

/* Whether the model's pieces are all where the GPU can read them. */
static int check_model(const struct janas_gpu *g,
                       const struct janas_gpu_tok_model *md, char *err,
                       size_t len)
{
    uint32_t qd = md->n_head * md->hd, kvd = md->n_kv * md->hd;
    /* feed-forwards may end inside a group of 256 (qwen4exp's 640): their
       down matrices in blocks of 32, checked by janas_gpu_can and
       check_ffn */
    /* its shaders add in clusters of 8 lanes and use the row shaders */
    if (g->sg < 8)
        return fail(err, len, "subgroups of fewer than 8 lanes");
    if (md->dm % JANAS_QK || qd % JANAS_QK || md->ff % 32 || md->dm > 8192 ||
        qd > 8192 || md->hd > 512 || md->hd % 16 || md->n_rot > md->hd ||
        md->n_kv == 0 || md->n_head % md->n_kv)
        return fail(err, len, "a shape the token's shaders do not take");
    /* each attention layer's own: at most two kinds of workgroup (Gemma
       4's), the query heads of a KV head split among workgroups where they
       do not fit one */
    uint32_t kinds[2] = {0, 0};
    for (uint32_t l = 0; l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &md->layers[l];
        if (ly->rec)
            continue;
        uint32_t ga = ly->n_kv && md->n_head % ly->n_kv == 0
                          ? janas_gt_att_g(md->n_head / ly->n_kv, ly->hd)
                          : 0;
        if (!ga || ly->hd > md->hd || ly->hd % 16 || ly->n_rot > ly->hd ||
            ly->n_rot % 2 || ly->n_kv > md->n_kv ||
            ga * ATT_RUN > ATT_MAXS_G(ga) ||
            /* sixteen-bit values: a thread 8 numbers of the G heads, at
               most 256 threads (attn_decode.comp's SH would be 0) */
            (md->kv16 && ga * ly->hd > 2048) || ly->cap == 0 ||
            ly->cap > md->cap ||
            (ly->swa ? ly->n_rot > md->n_rot_swa : ly->n_rot > md->n_rot) ||
            (md->kv16 && (md->ix_heads || md->n_rec)))
            return fail(err, len, "a shape the token's shaders do not take");
        if (!kinds[0] || kinds[0] == ga)
            kinds[0] = ga;
        else if (!kinds[1] || kinds[1] == ga)
            kinds[1] = ga;
        else
            return fail(err, len, "more than two kinds of attention layer");
    }
    if (md->n_expert &&
        (md->n_expert > 512 || md->k == 0 || md->k > 16 ||
         md->k > md->n_expert || md->eff % 32 || !md->fetch || !g->pipe_id4 ||
         !g->pipe_id6 || !g->pipe_id6s || !g->pipea || !g->pipe_dm[0]))
        return fail(err, len, "a mixture the token's shaders do not take");
    if (md->n_rec &&
        (md->ds != 128 || md->conv_ch % 256 || md->d_inner % 256 ||
         md->d_inner != md->nv * md->ds || md->nv > 64 || !md->nkh ||
         md->nv % md->nkh || md->d_conv < 2 || md->d_conv > 8 ||
         md->d_inner > md->dm * 4 ||
         md->conv_ch != 2 * md->nkh * md->ds + md->d_inner ||
         !janas_gt_addr_of(g, md->ssm_state,
                           2 * (size_t)md->n_rec * md->nv * md->ds * md->ds *
                               2) ||
         !janas_gt_addr_of(g, md->conv_state,
                           2 * (size_t)md->n_rec * (md->d_conv - 1) *
                               md->conv_ch * 4)))
        return fail(err, len,
                    "recurrent layers the token's shaders do not take");
    for (uint32_t l = 0; l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &md->layers[l];
        const char *why = check_ffn(g, md, ly);
        if (why) {
            snprintf(err, len, "layer %u: %s", l, why);
            return -1;
        }
        if (ly->rec) {
            const struct janas_matvec_task *r[3] = {&ly->ssm_qkv, &ly->ssm_z,
                                                    &ly->ssm_out};
            for (int i = 0; i < 3; i++)
                if (!janas_gpu_can(g, r[i])) {
                    snprintf(err, len,
                             "layer %u: a weight the GPU cannot compute "
                             "(product %d, type %d)",
                             l, i, r[i]->type);
                    return -1;
                }
            if (!md->hc && (!janas_gt_addr_of(g, ly->attn_norm, md->dm * 4) ||
                            !janas_gt_addr_of(g, ly->ffn_norm, md->dm * 4)))
                return fail(err, len, "a norm out of the GPU's reach");
            continue;
        }
        const struct janas_matvec_task *t[4] = {&ly->wq, &ly->wk, &ly->wv,
                                                &ly->wo};
        for (int i = 0; i < 4; i++)
            /* none: V is K, or the keys and values another layer's */
            if ((i == 0 || i == 3 || t[i]->type) && !janas_gpu_can(g, t[i])) {
                snprintf(err, len,
                         "layer %u: a weight the GPU cannot compute (product "
                         "%d, type %d)",
                         l, i, t[i]->type);
                return -1;
            }
        if (md->gated_q && (ly->wq.type == JANAS_Q6_K_P || ly->wq.plane[0]))
            return fail(err, len, "a gated query in planes");
        size_t cache = (size_t)ly->n_kv * ly->cap * ly->hd << md->kv16;
        if ((!md->hc && (!janas_gt_addr_of(g, ly->attn_norm, md->dm * 4) ||
                         !janas_gt_addr_of(g, ly->ffn_norm, md->dm * 4))) ||
            (ly->q_norm && !janas_gt_addr_of(g, ly->q_norm, ly->hd * 4)) ||
            (ly->k_norm && !janas_gt_addr_of(g, ly->k_norm, ly->hd * 4)) ||
            (ly->post_attn &&
             !janas_gt_addr_of(g, ly->post_attn, md->dm * 4)) ||
            (ly->post_ffn && !janas_gt_addr_of(g, ly->post_ffn, md->dm * 4)) ||
            (ly->ple_gate &&
             (!md->ple_dim || md->n_expert || !g->pipe_dm[0] ||
              !janas_gt_addr_of(g, ly->ple_gate,
                                (size_t)md->ple_dim * md->dm * 4) ||
              !janas_gt_addr_of(g, ly->ple_proj,
                                (size_t)md->ple_dim * md->dm * 4) ||
              !janas_gt_addr_of(g, ly->ple_post, md->dm * 4))) ||
            !janas_gt_addr_of(g, ly->kc, cache) ||
            !janas_gt_addr_of(g, ly->vc, cache) ||
            !janas_gt_addr_of(g, ly->ks, (size_t)ly->n_kv * ly->cap * 4) ||
            !janas_gt_addr_of(g, ly->vs, (size_t)ly->n_kv * ly->cap * 4))
            return fail(err, len,
                        "a norm or the KV cache out of the GPU's "
                        "reach");
    }
    if (!janas_gpu_can(g, &md->output) ||
        (!md->hc && !janas_gt_addr_of(g, md->out_norm, md->dm * 4)))
        return fail(err, len, "the output head out of the GPU's reach");
    if (md->hc && check_hc(g, md, err, len) != 0)
        return -1;
    (void)kvd;
    return 0;
}

/* The token's buffer, bytes long: host-visible where the GPU shares the
   host's memory; on a card with memory of its own in that memory, where
   the shaders read and write it at its speed, the host's few writes (the
   block's inputs, the parameters) and reads (logits, hidden states, a
   mixture's routing) through a mirror (JANAS_GPU_TOK_HOSTBUF=1: as on an
   integrated GPU, across the bus; JANAS_GPU_TOK_MIRROR=1: the mirror on an
   integrated one too, to test it with weights read in place). */
static int make_buf(struct janas_gpu *g, struct janas_gpu_tok *t, size_t bytes)
{
    if ((janas_gpu_shares_host(g) && !getenv("JANAS_GPU_TOK_MIRROR")) ||
        getenv("JANAS_GPU_TOK_HOSTBUF"))
        return janas_g_make_stage(g, &t->buf, bytes, 1);
    struct stage d;
    if (janas_g_make_local(g, &d, bytes) != 0)
        return janas_g_make_stage(g, &t->buf, bytes, 1);
    if (janas_g_make_stage(g, &t->mirror, bytes, 1) != 0) {
        janas_g_free_stage(g, &d);
        return -1;
    }
    t->buf = d;
    t->buf.map = t->mirror.map;
    t->buf.coherent = 0; /* every range through janas_gt_sync_range */
    t->local = 1;
    if (getenv("JANAS_GPU_VERBOSE"))
        fprintf(stderr,
                "gpu token buffer: %zu MiB in the GPU's memory, through a "
                "mirror\n",
                bytes >> 20);
    return 0;
}

struct janas_gpu_tok *janas_gpu_tok_create(struct janas_gpu *g,
                                           const struct janas_gpu_tok_model *md,
                                           char *err, size_t err_len)
{
    /* on a card with memory of its own the caller keeps the state the
       same in both places (model_gpu.c sync_state): every buffer named in
       md needs a region all the same, which check_model sees */
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
    /* a sparse attention: a token's positions are at most its budget and
       its own block's, the runs that take them (ix_select.comp) */
    if (md->ix_heads) {
        uint32_t r = md->ix_ratio;
        t->ix_runs = (md->ix_top + r - 1 + ATT_RUN - 1) / ATT_RUN;
        if (t->ix_runs < t->nch)
            t->nch = t->ix_runs;
        t->ix_nbs = (md->cap + r - 1) / r;
        t->ix_ls = md->ix_top + r;
    }
    size_t full = (size_t)md->n_head * t->nch * (md->hd + 2);
    t->part_floats = TOK_PART_BUDGET / 4 > full ? TOK_PART_BUDGET / 4 : full;
    t->nb = TOK_MAX_BLOCK;
    t->n_seg = md->n_layer + 1;
    size_t nb = t->nb;
    /* feed-forward widths ending inside a group of 256 (qwen4exp's 640):
       their buffers in whole groups, the rest zero */
    t->ffp = (md->ff + JANAS_QK - 1) / JANAS_QK * JANAS_QK;
    t->effp = (md->eff + JANAS_QK - 1) / JANAS_QK * JANAS_QK;
    t->o_x = take(&at, nb * md->dm * 4);
    t->o_x1 = take(&at, nb * md->dm * 4); /* the residual's other buffer */
    t->o_q256 = take(&at, nb * big / JANAS_QK * 320);
    t->o_q32 = take(&at, nb * big / JANAS_QK * 320);
    t->o_qkv = take(&at, nb * (qd + 2 * kvd) * 4);
    t->o_ywo = take(&at, nb * md->dm * 4);
    t->o_gu = take(&at, nb * t->ffp * 2 * 4);
    t->o_h = take(&at, nb * t->ffp / JANAS_QK * 320);
    t->o_yd = take(&at, nb * md->dm * 4);
    t->o_rope = take(&at, nb * md->n_rot * 4);
    t->o_rope2 = md->n_rot_swa ? take(&at, nb * md->n_rot_swa * 4) : 0;
    t->o_post = take(&at, nb * md->dm * 4);
    for (uint32_t l = 0; l < md->n_layer; l++)
        if (md->layers[l].pre_norm2) { /* Gemma 4's mixture */
            t->o_ybr = take(&at, nb * md->dm * 4);
            t->o_xbr = take(&at, nb * md->dm * 4);
            t->o_rin = take(&at, nb * md->dm * 4);
            t->o_post2 = take(&at, nb * md->dm * 4);
            break;
        }
    if (md->ple_dim) {
        t->o_ple = take(&at, nb * md->n_layer * md->ple_dim * 4);
        t->o_pg = take(&at, nb * md->ple_dim * 4);
        t->o_ph = take(&at, nb * md->ple_dim * 4);
    }
    t->gelu = md->gelu;
    /* the attention's workgroups: the query heads each takes, one or two
       kinds (check_model) */
    for (uint32_t l = 0; l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &md->layers[l];
        if (ly->rec)
            continue;
        uint32_t ga = janas_gt_att_g(md->n_head / ly->n_kv, ly->hd);
        if (!t->at_g[0] || t->at_g[0] == ga)
            t->at_g[0] = ga;
        else
            t->at_g[1] = ga;
    }
    if (!t->at_g[0]) /* qwen4exp's streams: the model's own */
        t->at_g[0] = md->n_head / md->n_kv;
    t->o_logits = take(&at, nb * md->n_vocab * 4);
    t->o_pos = take(&at, 32); /* janas_gpu_tok_run's five words */
    t->o_part = take(&at, t->part_floats * 4);
    t->o_xnf = take(&at, nb * md->dm * 4);
    t->o_hid = take(&at, nb * md->dm * 4);
    if (md->hc) {
        size_t hd = (size_t)md->hc * md->dm, r = md->hc_rank;
        t->o_res = take(&at, nb * hd * 4);
        t->o_hxn = take(&at, nb * hd * 4);
        t->o_hxb = take(&at, nb * hd * 4);
        t->o_lo = take(&at, nb * r * 4);
        t->o_lob = take(&at, nb * r * 4);
        t->o_hg = take(&at, nb * hd * 4);
        t->o_inj = take(&at, nb * md->hc * 4);
        t->o_hmean = take(&at, nb * md->dm * 4);
        if (md->ix_heads) {
            uint32_t d = md->ix_dim, r = md->ix_ratio;
            t->o_ixq = take(&at, nb * md->ix_heads * d * 4);
            t->o_ixk = take(&at, nb * d * 4);
            t->o_ixrope = take(&at, (nb + nb / r + 2) * md->n_rot * 4);
            t->o_ixsc = take(&at, nb * t->ix_nbs * 4);
            t->o_ixlist = take(&at, nb * t->ix_ls * 4);
        }
        if (md->ng_layer < md->n_layer) {
            t->o_ngk = take(&at, nb * hd * 4);
            t->o_ngv = take(&at, nb * md->dm * 4);
            t->o_ngeb = take(&at, nb * md->dm * 4);
            t->o_nggv = take(&at, nb * hd * 4);
            t->o_ngemb = take(&at, nb * md->dm * 4);
            t->o_ngw = take(&at, hd * md->ng_kernel * 4);
        }
    }
    if (md->gated_q)
        t->o_gate = take(&at, nb * qd * 4);
    if (md->n_rec) {
        t->o_mix = take(&at, nb * md->conv_ch * 4);
        t->o_z = take(&at, nb * md->d_inner * 4);
        t->o_conv =
            take(&at, (nb + JANAS_GPU_TOK_MAX_REPLAY) * md->conv_ch * 4);
        t->o_ba = take(&at, nb * 2 * md->nv * 4);
        t->o_att = take(&at, nb * md->d_inner * 4);
        t->o_bd = take(&at, nb * 2 * md->nv * 4);
        /* the log where the GPU reads and writes it: then the blocks short
           enough to be logged (a chain of drafts) may come here too */
        size_t lr = (size_t)md->n_rec * md->log_rows;
        t->log_ok = md->log_mixed && md->log_beta && md->log_decay &&
                    md->log_rows >= JANAS_GPU_TOK_MAX_REPLAY &&
                    janas_gt_addr_of(g, md->log_mixed, lr * md->conv_ch * 4) &&
                    janas_gt_addr_of(g, md->log_beta, lr * md->nv * 4) &&
                    janas_gt_addr_of(g, md->log_decay, lr * md->nv * 4);
    }
    uint32_t ne = md->n_expert;
    if (ne) {
        /* the pairs of a block of nb tokens, k each; at most a tile of 16
           of them per expert more than if they were all whole tiles */
        size_t np = nb * md->k, nbd = md->dm / JANAS_QK,
               nbf = t->effp / JANAS_QK;
        t->o_r = take(&at, nb * (ne + 1) * 4);
        t->o_pe = take(&at, np * 4);
        t->o_pw = take(&at, np * 4);
        t->o_shw = take(&at, nb * 4);
        t->o_sh = take(&at, nb * md->dm * 4);
        t->o_xg = take(&at, np * nbd * 320);
        t->o_egu = take(&at, np * t->effp * 2 * 4);
        t->o_eh = take(&at, np * nbf * 320);
        t->o_dout = take(&at, np * md->dm * 4);
        t->o_gt = take(&at, np * 4);
        t->o_inv = take(&at, np * 4);
        t->tiles1 = 3 * 2 * ne;
        t->tiles2 = 3 * ne;
        /* blocks of mvid_n tokens or fewer in tiles of MVID_NV vectors: by
           default all of them, so that an expert's bits are the CPU's in
           every block - a prompt with a tier of 300 MB went through *_mmid
           (not the CPU's bits) on the GPU's experts and through the CPU on
           the others, the split set by what the tier held, and with drafts
           and without the tokens parted at the 11th; on the laptop's Arc
           as a card of its own the prompt was 7% faster with every expert
           in the tier, 3% slower with 300 MB (24 Oct 2026) */
        const char *mv = getenv("JANAS_GPU_MVID");
        t->mvid_n = mv ? (uint32_t)atoi(mv) : (uint32_t)nb;
        if (t->mvid_n > nb)
            t->mvid_n = (uint32_t)nb;
        /* the tables for a whole block by rows, whatever mvid_n: Q5_1 and
           Q8_0, which have no tiles, go by rows at any size */
        size_t nt = np / 16, nm = (size_t)nb * md->k / MVID_NV;
        if (nm > nt)
            nt = nm;
        t->o_tab1 = take(&at, (t->tiles1 + 2 * (nt + ne)) * 16);
        t->o_tab2 = take(&at, (t->tiles2 + nt + ne) * 16);
        t->o_ind = take(&at, 32);
        t->cnt = calloc(ne + 1, sizeof(uint32_t));
        t->fill = calloc(ne, sizeof(uint32_t));
        t->ids = calloc(ne, sizeof(uint32_t));
        t->slots = calloc(ne, sizeof(*t->slots));
    }
    /* a down matrix in blocks of 32 - a shared expert's Q8_0, a dense
       layer's IQ4_NL (unsloth's UD files mix it in) - takes the activation
       as floats, quantized then into both layouts (janas_gt_rec_ffn) */
    int d32 = 0;
    for (uint32_t l = 0; l < md->n_layer; l++)
        d32 |= janas_g_x_in_32(ne ? md->layers[l].sh_down.type
                                  : md->layers[l].down.type) ||
               janas_g_x_in_32(md->layers[l].dn_down.type); /* Gemma 4's */
    if (d32) {
        t->o_hf = take(&at, nb * t->ffp * 4);
        t->o_h32 = take(&at, nb * t->ffp / JANAS_QK * 320);
    }
    if (md->n_rec || ne) {
        t->par = calloc((size_t)md->n_layer * PAR_N, sizeof(size_t));
        if (!t->par || (ne && (!t->cnt || !t->fill || !t->ids || !t->slots))) {
            fail(err, err_len, "out of memory");
            janas_gpu_tok_destroy(g, t);
            return NULL;
        }
        for (uint32_t l = 0; l < md->n_layer; l++) {
            size_t *pr = t->par + (size_t)l * PAR_N;
            if (ne)
                pr[PAR_R] = take(&at, (size_t)(ne + 1) * md->dm * 4);
            if (!md->layers[l].rec)
                continue;
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
    if (make_buf(g, t, at) != 0 ||
        janas_g_make_pipeline(g, janas_spv_norm_quant,
                              sizeof(janas_spv_norm_quant), &t->sm_nq, &t->p_nq,
                              0) != VK_SUCCESS ||
        janas_g_make_pipeline(g, janas_spv_qk_rope, sizeof(janas_spv_qk_rope),
                              &t->sm_qk, &t->p_qk, 0) != VK_SUCCESS ||
        janas_g_make_pipeline_sg(g, janas_spv_attn_decode,
                                 sizeof(janas_spv_attn_decode), &t->sm_at,
                                 &t->p_at, t->at_g[0], att_sg()) == 0 ||
        (t->at_g[1] &&
         janas_g_make_pipeline_sg(g, janas_spv_attn_decode,
                                  sizeof(janas_spv_attn_decode), &t->sm_at2,
                                  &t->p_at2, t->at_g[1], att_sg()) == 0) ||
        (md->ple_dim &&
         janas_g_make_pipeline(g, janas_spv_ple_act, sizeof(janas_spv_ple_act),
                               &t->sm_pa, &t->p_pa, 0) != VK_SUCCESS) ||
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
          janas_g_make_pipeline(g, janas_spv_gdn_ba, sizeof(janas_spv_gdn_ba),
                                &t->sm_gb, &t->p_gb, 0) != VK_SUCCESS ||
          janas_g_make_pipeline(g, janas_spv_gdn_act, sizeof(janas_spv_gdn_act),
                                &t->sm_gx, &t->p_gx, 0) != VK_SUCCESS ||
          !g->pipe_dm[0])) ||
        (ne && (janas_g_make_pipeline(g, janas_spv_moe_topk,
                                      sizeof(janas_spv_moe_topk), &t->sm_tk,
                                      &t->p_tk, 0) != VK_SUCCESS ||
                janas_g_make_pipeline(g, janas_spv_moe_gather,
                                      sizeof(janas_spv_moe_gather), &t->sm_ga,
                                      &t->p_ga, 0) != VK_SUCCESS ||
                janas_g_make_pipeline(g, janas_spv_moe_combine,
                                      sizeof(janas_spv_moe_combine), &t->sm_cb,
                                      &t->p_cb, 0) != VK_SUCCESS)) ||
        (d32 && janas_g_make_pipeline(g, janas_spv_silu_mul,
                                      sizeof(janas_spv_silu_mul), &t->sm_sm,
                                      &t->p_sm, 0) != VK_SUCCESS)) {
        fail(err, err_len, "cannot create the token's buffers or shaders");
        janas_gpu_tok_destroy(g, t);
        return NULL;
    }
    /* the plan on the GPU when the tier holds every expert (optional) */
    if (ne &&
        janas_g_make_pipeline(g, janas_spv_moe_plan, sizeof(janas_spv_moe_plan),
                              &t->sm_pl, &t->p_pl, 0) != VK_SUCCESS)
        t->p_pl = VK_NULL_HANDLE;
    /* zero: the padding of the feed-forwards' widths is never written */
    memset(t->buf.map, 0, at);
    janas_gt_sync_range(g, t, 0, at, 1);
    if (md->hc && janas_gt_hc_create(g, t, md) != 0) {
        fail(err, err_len, "cannot create the streams' shaders");
        janas_gpu_tok_destroy(g, t);
        return NULL;
    }
    if (md->ix_heads && janas_gt_qsa_create(g, t) != 0) {
        fail(err, err_len, "cannot create the sparse attention's shaders");
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
        janas_gt_sync_range(g, t, pr[PAR_BA],
                            pr[PAR_NORM] + md->ds * 4 - pr[PAR_BA], 1);
    }
    /* a mixture's routers, as f32 (the CPU's f16 weights exactly), the
       shared expert's gate after them */
    for (uint32_t l = 0; ne && l < md->n_layer; l++) {
        const struct janas_gpu_tok_layer *ly = &md->layers[l];
        size_t o = t->par[(size_t)l * PAR_N + PAR_R], n = (size_t)ne * md->dm;
        float *r = (float *)(t->buf.map + o);
        for (size_t i = 0; i < n; i++)
            r[i] = janas_fp16_to_fp32(ly->router16[i]);
        if (ly->sh_gate)
            memcpy(r + n, ly->sh_gate, md->dm * 4);
        else
            memset(r + n, 0, md->dm * 4);
        janas_gt_sync_range(g, t, o, (n + md->dm) * 4, 1);
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
    /* a mixture larger than the card: its tier of experts */
    if (ne && md->tier_bytes) {
        if (!md->fetch || !md->cpu_rows || !md->slot_bytes ||
            janas_gt_tier_create(g, t, md) != 0) {
            fail(err, err_len, "no room for a tier of experts on the card");
            janas_gpu_tok_destroy(g, t);
            return NULL;
        }
        if (getenv("JANAS_GPU_VERBOSE"))
            fprintf(stderr,
                    "gpu experts: a tier of %u slots (%zu MiB) in the "
                    "card's memory\n",
                    t->n_slot, (t->n_slot * t->slot_bytes) >> 20);
    }
    return t;
}

/* snprintf at the end of buf */
static void add(char *buf, size_t len, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));
static void add(char *buf, size_t len, const char *fmt, ...)
{
    size_t n = strlen(buf);
    if (n + 1 >= len)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf + n, len - n, fmt, ap);
    va_end(ap);
}

void janas_gpu_tok_counts(const struct janas_gpu_tok *t, uint64_t out[5])
{
    memset(out, 0, 5 * sizeof *out);
    if (!t)
        return;
    out[0] = t->tier_hits;
    out[1] = t->tier_loads;
    out[2] = t->cpu_experts;
    out[3] = t->n_slot;
    out[4] = t->slot_bytes;
}

void janas_gpu_tok_report(const struct janas_gpu_tok *t, char *buf, size_t len)
{
    if (!len)
        return;
    buf[0] = 0;
    if (!t)
        return;
    uint64_t asked = t->tier_hits + t->tier_loads + t->cpu_experts;
    if (t->n_slot)
        add(buf, len,
            "experts' tier: %u slots, %llu MiB of the card's memory; of %llu "
            "experts asked for, %.1f%% found there, %.1f%% read into it, "
            "%.1f%% computed by the CPU\n",
            t->n_slot, (unsigned long long)((t->n_slot * t->slot_bytes) >> 20),
            (unsigned long long)asked,
            asked ? 100.0 * (double)t->tier_hits / (double)asked : 0.0,
            asked ? 100.0 * (double)t->tier_loads / (double)asked : 0.0,
            asked ? 100.0 * (double)t->cpu_experts / (double)asked : 0.0);
    if (!t->qp || !t->tokens)
        return;
    double all = 0;
    for (int i = 0; i < PH_N; i++)
        all += t->ph_ns[i];
    add(buf, len,
        "gpu token profile, %llu passes of %.1f tokens, %.2f ms a pass "
        "(%.2f ms from submission to fence, %.2f ms copying the logits):\n",
        (unsigned long long)t->tokens, (double)t->pass_tokens / t->tokens,
        all / t->tokens * 1e-6, t->wall_ns / t->tokens * 1e-6,
        t->copy_ns / t->tokens * 1e-6);
    if (t->plan_ns > 0)
        add(buf, len, "  (the CPU's plans %.3f ms a pass)\n",
            t->plan_ns / t->tokens * 1e-6);
    if (t->n_slot)
        add(buf, len,
            "  (a pass's experts: %.1f in the tier, %.1f read into it, %.1f "
            "by the CPU in %.3f ms; the cache %.3f ms)\n",
            (double)t->tier_hits / t->tokens, (double)t->tier_loads / t->tokens,
            (double)t->cpu_experts / t->tokens, t->cpu_ns / t->tokens * 1e-6,
            t->fetch_ns / t->tokens * 1e-6);
    for (int i = 0; i < PH_N; i++)
        if (t->ph_ns[i] > 0)
            add(buf, len, "  %-14s %7.3f ms %5.1f%%\n", ph_name[i],
                t->ph_ns[i] / t->tokens * 1e-6,
                all > 0 ? 100.0 * t->ph_ns[i] / all : 0.0);
}

void janas_gpu_tok_destroy(struct janas_gpu *g, struct janas_gpu_tok *t)
{
    if (!t)
        return;
    if (t->qp) {
        char rep[4096];
        janas_gpu_tok_report(t, rep, sizeof(rep));
        if (t->tokens)
            fputs(rep, stderr);
        g->vkDestroyQueryPool(g->dev, t->qp, NULL);
    }
    for (uint32_t n = 0; n <= TOK_MAX_BLOCK; n++)
        for (int a = 0; a < 3; a++)
            for (int h = 0; h < 2; h++) {
                if (t->cb[n][a][h])
                    g->vkFreeCommandBuffers(g->dev, g->cp, 1, &t->cb[n][a][h]);
                janas_gt_free_segments(g, t, &t->seg_cb[n][a][h]);
            }
    if (t->cb_rows)
        g->vkFreeCommandBuffers(g->dev, g->cp, 1, &t->cb_rows);
    VkPipeline p[18] = {t->p_nq, t->p_qk, t->p_at, t->p_am, t->p_gc,  t->p_gs,
                        t->p_gl, t->p_gr, t->p_gg, t->p_tk, t->p_ga,  t->p_cb,
                        t->p_sm, t->p_gb, t->p_gx, t->p_pl, t->p_at2, t->p_pa};
    VkShaderModule s[18] = {t->sm_nq, t->sm_qk,  t->sm_at, t->sm_am, t->sm_gc,
                            t->sm_gs, t->sm_gl,  t->sm_gr, t->sm_gg, t->sm_tk,
                            t->sm_ga, t->sm_cb,  t->sm_sm, t->sm_gb, t->sm_gx,
                            t->sm_pl, t->sm_at2, t->sm_pa};
    janas_gt_hc_destroy(g, t);
    janas_gt_qsa_destroy(g, t);
    janas_gt_tier_destroy(g, t);
    free(t->par);
    free(t->cnt);
    free(t->fill);
    free(t->ids);
    free(t->slots);
    for (int i = 0; i < 18; i++) {
        if (p[i])
            g->vkDestroyPipeline(g->dev, p[i], NULL);
        if (s[i])
            g->vkDestroyShaderModule(g->dev, s[i], NULL);
    }
    janas_g_free_stage(g, &t->buf);
    janas_g_free_stage(g, &t->mirror);
    free(t);
}

/* Bytes [off, off + bytes) of the mirror out of the CPU's caches (flush)
   or reloaded from memory (!flush), where it is not coherent. */
void janas_gt_mirror_cache(struct janas_gpu *g, struct janas_gpu_tok *t,
                           size_t off, size_t bytes, int flush)
{
    if (t->mirror.coherent)
        return;
    VkDeviceSize a = off / g->atom * g->atom,
                 e = (off + bytes + g->atom - 1) / g->atom * g->atom;
    VkMappedMemoryRange mr = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                              .memory = t->mirror.mem,
                              .offset = a,
                              .size =
                                  e >= t->mirror.bytes ? VK_WHOLE_SIZE : e - a};
    if (flush)
        g->vkFlushMappedMemoryRanges(g->dev, 1, &mr);
    else
        g->vkInvalidateMappedMemoryRanges(g->dev, 1, &mr);
}

/* Bytes [off, off + bytes) of the token's buffer out of the CPU's caches
   (flush) or reloaded from memory (!flush), where it is not coherent; on a
   card with memory of its own copied too, a submission each. */
void janas_gt_sync_range(struct janas_gpu *g, struct janas_gpu_tok *t,
                         size_t off, size_t bytes, int flush)
{
    if (t->local) {
        /* the mirror's own cache lines, then the copy; or the other way */
        if (flush) {
            janas_gt_mirror_cache(g, t, off, bytes, 1);
            if (janas_g_copy(g, t->mirror.buf, off, t->buf.buf, off, bytes))
                g->broken = 1;
        } else {
            if (janas_g_copy(g, t->buf.buf, off, t->mirror.buf, off, bytes))
                g->broken = 1;
            janas_gt_mirror_cache(g, t, off, bytes, 0);
        }
        return;
    }
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

/* qwen4exp's layers (record's, for a model with hc streams), as
   model_hc.c's janas_m_hc_layers: the streams set to the embeddings; in
   each layer the n-gram embeddings where they are, the mix into the token
   mixer's input, the DeltaNet or the attention (the attention's input at
   bf16 kept for the CPU's indexer, att_in), its output combined into the
   streams, the mix into the feed-forward's, the router - the experts and
   their combine in the next submission - then the output's mix and head.
   seg as record's. */
static int record_hc(struct janas_gpu *g, struct janas_gpu_tok *t,
                     const struct janas_gpu_tok_model *md,
                     const struct janas_gpu_tok_layer *layers,
                     const struct janas_matvec_task *output, uint32_t n,
                     uint32_t first, int hid, int seg, VkPipeline *bound)
{
    VkDeviceAddress B = t->buf.addr, XNF = B + t->o_xnf, YWO = B + t->o_ywo,
                    YD = B + t->o_yd;
    uint32_t nl = md->n_layer;
    for (uint32_t l = 0; l < nl; l++) {
        const struct janas_gpu_tok_layer *ly = &layers[l];
        if (seg >= 0 && seg != (int)l)
            continue;
        if (l == 0) {
            janas_gt_rec_hc_combine(g, t, md, bound, B + t->o_x, 1, n);
        } else {
            janas_gt_rec_experts(g, t, md, &layers[l - 1], bound, YD, n, 0);
            janas_gt_rec_hc_combine(g, t, md, bound, YD, 0, n);
        }
        if (l == md->ng_layer && janas_gt_rec_ng(g, t, md, ly, bound, n) != 0)
            return -1;
        janas_gt_rec_hc_mix(g, t, md, ly, 0, bound, XNF, n);
        janas_gt_stamp(g, t, PH_NORM);
        if (ly->rec) {
            if (janas_gt_rec_gdn(g, t, md, ly, l, bound, YWO, n) != 0)
                return -1;
        } else {
            if (md->att_in) {
                VkDeviceAddress A = janas_gt_addr_of(
                    g, md->att_in, (size_t)(ly->slot + 1) * t->nb * md->dm * 4);
                if (!A)
                    return -1;
                struct push_ha pa = {.n = n * md->dm, .silu = 0};
                janas_g_push_addr(pa.x, XNF);
                janas_g_push_addr(pa.xb, A + (VkDeviceAddress)ly->slot * t->nb *
                                                 md->dm * 4);
                janas_gt_bind(g, bound, t->p_ha);
                janas_gt_push(g, &pa, sizeof(pa));
                g->vkCmdDispatch(g->cb, (pa.n + 255) / 256, 1, 1);
                janas_g_compute_barrier(g);
                /* the sparse attention's positions, from that input */
                if (md->ix_heads &&
                    janas_gt_rec_qsa(g, t, md, ly, bound,
                                     A + (VkDeviceAddress)ly->slot * t->nb *
                                             md->dm * 4,
                                     n) != 0)
                    return -1;
            }
            if (janas_gt_rec_attn(g, t, md, ly, bound, n) != 0)
                return -1;
        }
        janas_gt_rec_hc_combine(g, t, md, bound, YWO, 0, n);
        janas_gt_rec_hc_mix(g, t, md, ly, 1, bound, XNF, n);
        janas_gt_stamp(g, t, PH_NORM);
        if (janas_gt_rec_router(g, t, md, ly, l, bound, n) != 0)
            return -1;
    }
    if (seg < 0 || seg == (int)nl) {
        janas_gt_rec_experts(g, t, md, &layers[nl - 1], bound, YD, n, 0);
        janas_gt_rec_hc_combine(g, t, md, bound, YD, 0, n);
        /* the streams themselves are the hidden states (an MTP block's
           input): run copies them from o_res */
        (void)hid;
        if (first < n) {
            janas_gt_rec_hc_mix(g, t, md, NULL, 2, bound, XNF, n);
            if (janas_gt_rec_product(g, t, bound, output,
                                     (size_t)first * (md->dm / JANAS_QK) * 320,
                                     B + t->o_logits, n - first, 0) != 0)
                return -1;
        }
        janas_gt_stamp(g, t, PH_OUT);
    }
    return 0;
}

/* The commands of a block of n tokens into g->cb (the caller points it at
   the block's buffer): every layer, then the output head for the tokens
   from first on (none if first is n); the position of the first token
   comes from the buffer.
   A single token takes the layers' GPU copies (JANAS_GPU_REPACK), a block
   the weights as they are (the copies' shaders take one vector).
   A mixture's block goes in n_layer + 1 submissions, seg the one recorded
   (-1: a dense model's, all of it): submission s takes layer s - 1's
   experts, then layer s up to its router's choice, the last one the final
   norm and the output head. part 1 records only segment seg's products of
   the experts, part 2 the rest from their sum on (0 all of it): the CPU
   computes its experts between the two. */
static int record(struct janas_gpu *g, struct janas_gpu_tok *t,
                  const struct janas_gpu_tok_model *md, uint32_t n,
                  uint32_t first, int hid, int seg, int part)
{
    const struct janas_gpu_tok_layer *layers =
        n > 1 && md->blayers ? md->blayers : md->layers;
    const struct janas_matvec_task *output =
        n > 1 && md->blayers ? &md->boutput : &md->output;
    /* the residual: read from X, the sum with a layer's output written to
       XN, the two swapped (every block starts from o_x, the same commands) */
    VkDeviceAddress B = t->buf.addr, X = B + t->o_x, XN = B + t->o_x1, SW,
                    YWO = B + t->o_ywo, YD = B + t->o_yd, XNF = B + t->o_xnf,
                    POST = B + t->o_post;
    if (g->vkResetCommandBuffer(g->cb, 0) != VK_SUCCESS)
        return -1;
    VkCommandBufferBeginInfo bb = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = 0};
    if (g->vkBeginCommandBuffer(g->cb, &bb) != VK_SUCCESS)
        return -1;
    VkPipeline bound = VK_NULL_HANDLE;
    /* a pass in one submission, of any length; or a mixture's segments,
       recorded one after the other the first time a block runs, their
       timestamps one series */
    t->prof = 1;
    if (t->qp && seg <= 0) {
        t->n_ts = 0;
        g->vkCmdResetQueryPool(g->cb, t->qp, 0, MAX_TS);
        janas_gt_stamp(g, t, PH_N); /* the start */
    } else if (t->qp && part != 2) {
        janas_gt_stamp(g, t, PH_BETWEEN);
    }
    /* on a card with memory of its own: the pass's inputs from the mirror,
       in its own commands (the sizes fixed by n, the values read at run
       time) - a mixture's in its first submission, its outputs in its last,
       and between the two what the CPU's plan writes and reads */
    int last = seg < 0 || seg == (int)md->n_layer;
    if (t->local && seg <= 0) {
        VkBufferCopy in[5] = {
            {.srcOffset = t->o_x,
             .dstOffset = t->o_x,
             .size = (VkDeviceSize)n * md->dm * sizeof(float)},
            {.srcOffset = t->o_rope,
             .dstOffset = t->o_rope,
             .size = (VkDeviceSize)n * md->n_rot * sizeof(float)},
            {.srcOffset = t->o_pos,
             .dstOffset = t->o_pos,
             .size = JANAS_GT_POS_BYTES},
            /* Gemma 4's second RoPE tables, its per-layer inputs */
            {.srcOffset = t->o_rope2,
             .dstOffset = t->o_rope2,
             .size = (VkDeviceSize)n * md->n_rot_swa * sizeof(float)},
            {.srcOffset = t->o_ple,
             .dstOffset = t->o_ple,
             .size =
                 (VkDeviceSize)n * md->n_layer * md->ple_dim * sizeof(float)}};
        uint32_t nin = 3;
        if (md->n_rot_swa)
            in[nin++] = in[3];
        if (md->ple_dim)
            in[nin++] = in[4];
        g->vkCmdCopyBuffer(g->cb, t->mirror.buf, t->buf.buf, nin, in);
        VkMemoryBarrier ib = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                              .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                               VK_ACCESS_SHADER_WRITE_BIT};
        g->vkCmdPipelineBarrier(g->cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &ib,
                                0, NULL, 0, NULL);
    }
    if (seg > 0 && md->n_expert && part != 2) {
        /* the previous layer's plan (gpu_token_moe.c): its rows' tokens, its
           pairs' rows, its tables and dispatches' sizes, laid out one after
           the other; and the experts read into the tier, or the CPU's rows,
           copied in a submission of their own before this one */
        if (t->local) {
            VkBufferCopy tb = {.srcOffset = t->o_gt,
                               .dstOffset = t->o_gt,
                               .size = t->o_ind + 32 - t->o_gt};
            g->vkCmdCopyBuffer(g->cb, t->mirror.buf, t->buf.buf, 1, &tb);
        }
        VkMemoryBarrier pb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                              .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                              .dstAccessMask =
                                  VK_ACCESS_SHADER_READ_BIT |
                                  VK_ACCESS_SHADER_WRITE_BIT |
                                  VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
        g->vkCmdPipelineBarrier(g->cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                0, 1, &pb, 0, NULL, 0, NULL);
    }
    uint32_t nl = md->n_layer;
    if (md->hc) {
        if (record_hc(g, t, md, layers, output, n, first, hid, seg, &bound) !=
            0)
            return -1;
        nl = 0; /* nothing more of the loops below */
    }
    for (uint32_t l = 0; l < nl; l++) {
        const struct janas_gpu_tok_layer *ly = &layers[l];
        int on = seg < 0 || seg == (int)l;
        /* a mixture: the previous layer's experts first, into YD */
        if (on && l && md->n_expert) {
            janas_gt_rec_experts(g, t, md, &layers[l - 1], &bound, YD, n, part);
            if (part == 1)
                goto end;
            if (layers[l - 1].pre_norm2) /* Gemma 4: its branches joined */
                janas_gt_rec_gemma_moe_tail(g, t, md, &layers[l - 1], &bound,
                                            YD, n);
        }
        /* the previous layer's feed-forward added (Gemma 4: normalized,
           the sum times the layer's output scale), then this one's norm */
        if (on) {
            const struct janas_gpu_tok_layer *pl = l ? &layers[l - 1] : NULL;
            janas_gt_rec_norm_quant_s(
                g, t, &bound, X,
                !pl            ? 0
                : pl->post_ffn ? POST
                               : YD,
                janas_gt_addr_of(g, ly->attn_norm, md->dm * 4), l ? XN : 0,
                ly->rec ? XNF : 0, md->dm, md->eps, n,
                pl && pl->post_ffn ? pl->out_scale : 0.0f);
            janas_gt_stamp(g, t, PH_NORM);
        }
        if (l)
            SW = X, X = XN, XN = SW;
        if (on) {
            if ((ly->rec ? janas_gt_rec_gdn(g, t, md, ly, l, &bound, YWO, n)
                         : janas_gt_rec_attn(g, t, md, ly, &bound, n)) != 0)
                return -1;
            janas_gt_rec_norm_quant(
                g, t, &bound, X, ly->post_attn ? POST : YWO,
                janas_gt_addr_of(g, ly->ffn_norm, md->dm * 4), XN,
                md->n_expert ? XNF : 0, md->dm, md->eps, n);
            janas_gt_stamp(g, t, PH_NORM);
        }
        SW = X, X = XN, XN = SW;
        /* Gemma 4's mixture: its dense branch, the router's and the
           experts' inputs first */
        if (on && ly->pre_norm2 &&
            janas_gt_rec_gemma_moe(g, t, md, ly, &bound, X, n) != 0)
            return -1;
        if (on &&
            (md->n_expert ? janas_gt_rec_router(g, t, md, ly, l, &bound, n)
                          : janas_gt_rec_ffn(g, t, &bound, &ly->gate, &ly->up,
                                             &ly->down, n, YD)) != 0)
            return -1;
        /* Gemma 4: the feed-forward's output normalized, into o_post */
        if (on && ly->post_ffn && !md->n_expert)
            janas_gt_rec_post_norm(g, t, &bound, YD, ly->post_ffn, md, n);
        /* ... and with per-layer embeddings their branch, on the residual
           with the feed-forward added (its output scale after the branch) */
        if (on && ly->ple_gate) {
            janas_gt_rec_norm_quant(g, t, &bound, X, POST, 0, XN, 0, md->dm,
                                    md->eps, n);
            SW = X, X = XN, XN = SW;
            janas_gt_rec_ple(g, t, md, ly, l, &bound, X, YD, n);
            janas_gt_rec_post_norm(g, t, &bound, YD, ly->ple_post, md, n);
        }
        /* a mixture's pass in one submission: the plan on the GPU */
        if (seg < 0 && md->n_expert)
            janas_gt_rec_plan(g, t, md, l, &bound, n);
    }
    /* the last feed-forward added, the final norm, the output head for
       tokens first .. n - 1 (none: the residual left in X, the next block's
       keys and values already in the cache) */
    if (!md->hc && (seg < 0 || seg == (int)nl)) {
        if (md->n_expert) {
            janas_gt_rec_experts(g, t, md, &layers[nl - 1], &bound, YD, n,
                                 part);
            if (part == 1)
                goto end;
            if (layers[nl - 1].pre_norm2)
                janas_gt_rec_gemma_moe_tail(g, t, md, &layers[nl - 1], &bound,
                                            YD, n);
        }
        if (first < n || hid) {
            const struct janas_gpu_tok_layer *pl = &layers[nl - 1];
            janas_gt_rec_norm_quant_s(
                g, t, &bound, X, pl->post_ffn ? POST : YD,
                janas_gt_addr_of(g, md->out_norm, md->dm * 4), 0,
                hid ? B + t->o_hid : 0, md->dm, md->eps, n,
                pl->post_ffn ? pl->out_scale : 0.0f);
        }
        if (first < n) {
            if (janas_gt_rec_product(g, t, &bound, output,
                                     (size_t)first * (md->dm / JANAS_QK) * 320,
                                     B + t->o_logits, n - first, 0) != 0)
                return -1;
        }
        janas_gt_stamp(g, t, PH_OUT);
    }
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    VkPipelineStageFlags from = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (t->local) {
        /* the logits and the hidden states back to the mirror (qwen4exp's:
           the streams after the last layer); before, layer seg's choice of
           experts for the plan, and with a tier the experts' input for the
           CPU's */
        VkBufferCopy out[2];
        uint32_t no = 0;
        size_t ho = md->hc ? t->o_res : t->o_hid,
               hb = (size_t)n * (md->hc ? md->hc : 1) * md->dm * 4;
        if (last && first < n)
            out[no++] = (VkBufferCopy){.srcOffset = t->o_logits,
                                       .dstOffset = t->o_logits,
                                       .size = (VkDeviceSize)(n - first) *
                                               md->n_vocab * 4};
        if (last && hid)
            out[no++] =
                (VkBufferCopy){.srcOffset = ho, .dstOffset = ho, .size = hb};
        if (!last) {
            out[no++] = (VkBufferCopy){.srcOffset = t->o_pe,
                                       .dstOffset = t->o_pe,
                                       .size = (VkDeviceSize)n * md->k * 4};
            if (t->n_slot)
                out[no++] =
                    (VkBufferCopy){.srcOffset = t->o_xnf,
                                   .dstOffset = t->o_xnf,
                                   .size = (VkDeviceSize)n * md->dm * 4};
        }
        if (no) {
            VkMemoryBarrier ob = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                  .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
            g->vkCmdPipelineBarrier(g->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &ob,
                                    0, NULL, 0, NULL);
            g->vkCmdCopyBuffer(g->cb, t->buf.buf, t->mirror.buf, no, out);
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            from = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
    }
    g->vkCmdPipelineBarrier(g->cb, from, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb,
                            0, NULL, 0, NULL);
end:
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
    size_t n =
        t->part_floats / janas_gt_part_tok(t, md, runs_for(t, pos + t->nb));
    /* the rings hold a window and a block: no more of the block */
    if (md->ring_room && n > md->ring_room)
        n = md->ring_room;
    return n < 1 ? 1 : n > t->nb ? t->nb : (uint32_t)n;
}

/* A command buffer for cb, allocated, and the commands of segment seg of a
   block of n tokens recorded into it: 0, or -1 with *cb NULL. */
int janas_gt_make_cb(struct janas_gpu *g, struct janas_gpu_tok *t,
                     const struct janas_gpu_tok_model *md, VkCommandBuffer *cb,
                     uint32_t n, uint32_t first, int hi, uint32_t nch, int seg,
                     int part)
{
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
    int rc = record(g, t, md, n, first, hi, seg, part);
    g->cb = keep;
    if (rc != 0) {
        g->vkFreeCommandBuffers(g->dev, g->cp, 1, cb);
        *cb = VK_NULL_HANDLE;
        return -1;
    }
    return 0;
}

int janas_gt_submit_wait(struct janas_gpu *g, const VkCommandBuffer *cb)
{
    return janas_gt_submit_wait2(g, VK_NULL_HANDLE, cb);
}

int janas_gt_submit_list(struct janas_gpu *g, const VkCommandBuffer *cbs,
                         uint32_t n, int wait)
{
    VkCommandBuffer l[4];
    uint32_t c = 0;
    for (uint32_t i = 0; i < n && c < 4; i++)
        if (cbs[i])
            l[c++] = cbs[i];
    VkSubmitInfo sub = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = c,
                        .pCommandBuffers = l};
    pthread_mutex_lock(&g->qlock);
    VkResult r =
        g->vkQueueSubmit(g->queue, 1, &sub, wait ? g->fence : VK_NULL_HANDLE);
    pthread_mutex_unlock(&g->qlock);
    if (r == VK_SUCCESS && wait)
        r = g->vkWaitForFences(g->dev, 1, &g->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        g->broken = 1;
        return -1;
    }
    if (wait)
        g->vkResetFences(g->dev, 1, &g->fence);
    return 0;
}

int janas_gt_submit_wait2(struct janas_gpu *g, VkCommandBuffer pre,
                          const VkCommandBuffer *cb)
{
    VkCommandBuffer two[2] = {pre, *cb};
    VkSubmitInfo sub = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = pre ? 2 : 1,
                        .pCommandBuffers = pre ? two : cb};
    pthread_mutex_lock(&g->qlock);
    VkResult r = g->vkQueueSubmit(g->queue, 1, &sub, g->fence);
    pthread_mutex_unlock(&g->qlock);
    if (r == VK_SUCCESS)
        r = g->vkWaitForFences(g->dev, 1, &g->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        g->broken = 1;
        return -1;
    }
    g->vkResetFences(g->dev, 1, &g->fence);
    return 0;
}

int janas_gpu_tok_run(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md, const float *x,
                      const float *cs, const float *sn, uint32_t pos,
                      uint32_t n, int all, float *logits, float *hidden,
                      uint32_t src, uint32_t dst, uint32_t replay,
                      uint32_t log_row)
{
    if (!t || g->busy || g->broken || n < 1 || n > t->nb || pos + n > md->cap ||
        (md->ring_room && n > md->ring_room))
        return -1;
    /* a replay, or a block to log, needs the log on the GPU */
    int logs = log_row != UINT32_MAX;
    if (md->n_rec &&
        (replay > JANAS_GPU_TOK_MAX_REPLAY ||
         ((replay || logs) && !t->log_ok) ||
         (logs && log_row + n > md->log_rows) || (replay && src == dst)))
        return -1;
    if (!md->n_rec && (replay || logs))
        return -1;
    uint32_t nch = runs_for(t, pos + n);
    if ((size_t)n * janas_gt_part_tok(t, md, nch) > t->part_floats)
        return -1;
    uint32_t half = md->n_rot / 2, first = all == 1 ? 0 : all == 2 ? n : n - 1;
    memcpy(t->buf.map + t->o_x, x, (size_t)n * md->dm * sizeof(float));
    if (md->hc && md->ng_layer < md->n_layer) {
        memcpy(t->buf.map + t->o_ngemb, md->ng_emb,
               (size_t)n * md->dm * sizeof(float));
        janas_gt_sync_range(g, t, t->o_ngemb,
                            (size_t)n * md->dm * sizeof(float), 1);
    }
    if (md->ix_heads) {
        size_t rb = ((size_t)n + n / md->ix_ratio + 2) * md->n_rot * 4;
        memcpy(t->buf.map + t->o_ixrope, md->ix_rope, rb);
        janas_gt_sync_range(g, t, t->o_ixrope, rb, 1);
    }
    for (uint32_t j = 0; j < n; j++) {
        uint8_t *r = t->buf.map + t->o_rope + (size_t)j * md->n_rot * 4;
        memcpy(r, cs + (size_t)j * half, half * sizeof(float));
        memcpy(r + half * sizeof(float), sn + (size_t)j * half,
               half * sizeof(float));
    }
    /* Gemma 4's per-layer inputs */
    size_t pb = (size_t)n * md->n_layer * md->ple_dim * sizeof(float);
    if (pb)
        memcpy(t->buf.map + t->o_ple, md->ple_in, pb);
    /* Gemma 4's sliding-window layers: their own tables */
    uint32_t h2 = md->n_rot_swa / 2;
    for (uint32_t j = 0; h2 && j < n; j++) {
        uint8_t *r = t->buf.map + t->o_rope2 + (size_t)j * md->n_rot_swa * 4;
        memcpy(r, md->cs_swa + (size_t)j * h2, h2 * sizeof(float));
        memcpy(r + h2 * sizeof(float), md->sn_swa + (size_t)j * h2,
               h2 * sizeof(float));
    }
    /* the position, the halves of the recurrent state to read and to
       write, the tokens to replay, the log's row: read by the shaders at
       run time */
    uint32_t pw[5] = {pos, src, dst, replay, log_row};
    _Static_assert(sizeof(pw) == JANAS_GT_POS_BYTES, "the pass's words");
    memcpy(t->buf.map + t->o_pos, pw, sizeof(pw));
    /* the pass copies them itself (record: a mixture's first and last
       submissions) */
    void (*in)(struct janas_gpu *, struct janas_gpu_tok *, size_t, size_t,
               int) = t->local ? janas_gt_mirror_cache : janas_gt_sync_range;
    in(g, t, t->o_x, (size_t)n * md->dm * sizeof(float), 1);
    in(g, t, t->o_rope, (size_t)n * md->n_rot * sizeof(float), 1);
    if (h2)
        in(g, t, t->o_rope2, (size_t)n * md->n_rot_swa * sizeof(float), 1);
    if (pb)
        in(g, t, t->o_ple, pb, 1);
    in(g, t, t->o_pos, sizeof(pw), 1);
    int ai = all == 1 && n == 1 ? 0 : all, hi = hidden != NULL;
    struct timespec w0, w1;
    clock_gettime(CLOCK_MONOTONIC, &w0);
    /* a failure once the GPU has started: with a replay the base it
       updates in place is no longer the state the CPU would replay on */
    int failed = replay ? -2 : -1;
    /* a mixture whose experts the tier holds all: read in at its first
       pass, then its passes whole, unless the router's choices are
       watched */
    if (md->n_expert && t->n_slot && !t->gplan_tried &&
        janas_gt_tier_fill(g, t, md) != 0 && g->broken)
        return -1;
    if (md->n_expert && (!t->gplan || (md->watched && md->watched(md->ctx)))) {
        if (janas_gt_run_segments(g, t, md, n, first, ai, hi, nch) != 0)
            return failed;
    } else {
        VkCommandBuffer *cb = &t->cb[n][ai][hi];
        if (*cb && t->cb_nch[n][ai][hi] != nch) {
            /* recorded for other runs: again (a few times in a context) */
            g->vkFreeCommandBuffers(g->dev, g->cp, 1, cb);
            *cb = VK_NULL_HANDLE;
        }
        if (!*cb) {
            if (janas_gt_make_cb(g, t, md, cb, n, first, hi, nch, -1, 0) != 0)
                return -1;
            t->cb_nch[n][ai][hi] = nch;
        }
        if (janas_gt_submit_wait(g, cb) != 0)
            return failed;
    }
    clock_gettime(CLOCK_MONOTONIC, &w1);
    size_t lb = (size_t)(n - first) * md->n_vocab * 4;
    if (lb) {
        in(g, t, t->o_logits, lb, 0);
        memcpy(logits, t->buf.map + t->o_logits, lb);
    }
    if (hidden) {
        /* qwen4exp's: the streams after the last layer, hc dm a token */
        size_t hb = (size_t)n * (md->hc ? md->hc : 1) * md->dm * 4,
               ho = md->hc ? t->o_res : t->o_hid;
        in(g, t, ho, hb, 0);
        memcpy(hidden, t->buf.map + ho, hb);
    }
    struct timespec w2;
    clock_gettime(CLOCK_MONOTONIC, &w2);
    if (t->qp && t->prof && t->n_ts > 1) {
        uint64_t ts[MAX_TS];
        if (g->vkGetQueryPoolResults(g->dev, t->qp, 0, t->n_ts, sizeof(ts), ts,
                                     sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            for (uint32_t i = 1; i < t->n_ts; i++)
                t->ph_ns[t->ph_of[i]] +=
                    (double)(ts[i] - ts[i - 1]) * g->ts_period;
            t->tokens++;
            t->pass_tokens += n;
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

void janas_gpu_tok_report(const struct janas_gpu_tok *t, char *buf, size_t len)
{
    (void)t;
    if (len)
        buf[0] = 0;
}

void janas_gpu_tok_counts(const struct janas_gpu_tok *t, uint64_t out[5])
{
    (void)t;
    memset(out, 0, 5 * sizeof *out);
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
                      uint32_t src, uint32_t dst, uint32_t replay,
                      uint32_t log_row)
{
    (void)replay;
    (void)log_row;
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

size_t janas_gpu_local_room(const struct janas_gpu *g)
{
    (void)g;
    return 0;
}

#endif
