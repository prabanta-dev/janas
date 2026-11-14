/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token_moe.c - a mixture's layers in the whole token on the GPU
 * (gpu_token_impl.h): its router's choice made on the GPU, sorted on the
 * CPU between two submissions, its experts' products and their sum.
 */
#include "gpu_token_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "quant.h"

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)

/* A mixture's submissions for one block shape, freed (two for each of the
   n_layer + 1 segments, its products and the rest, the array too). */
void janas_gt_free_segments(struct janas_gpu *g, struct janas_gpu_tok *t,
                            VkCommandBuffer **seg)
{
    if (!*seg)
        return;
    for (uint32_t s = 0; s < 2 * t->n_seg; s++)
        if ((*seg)[s])
            g->vkFreeCommandBuffers(g->dev, g->cp, 1, &(*seg)[s]);
    free(*seg);
    *seg = NULL;
}

/* A mixture's layer l up to its router's choice, on the block normalized
   in q256 and XNF: the shared expert into SH, the router's logits, then
   each token's experts and weights (PE, PW, SHW), for the CPU to read. */
int janas_gt_rec_router(struct janas_gpu *g, struct janas_gpu_tok *t,
                        const struct janas_gpu_tok_model *md,
                        const struct janas_gpu_tok_layer *ly, uint32_t l,
                        VkPipeline *bound, uint32_t n)
{
    VkDeviceAddress B = t->buf.addr;
    uint32_t ne = md->n_expert;
    if (ly->sh_gate_w.type &&
        janas_gt_rec_ffn(g, t, bound, &ly->sh_gate_w, &ly->sh_up, &ly->sh_down,
                         n, B + t->o_sh) != 0)
        return -1;
    /* the router's input: the feed-forward's normalized input, or Gemma
       4's own (o_rin) */
    janas_gt_rec_dense(g, bound, 0, B + t->par[(size_t)l * PAR_N + PAR_R],
                       B + (ly->pre_norm2 ? t->o_rin : t->o_xnf), B + t->o_r,
                       ne + (ly->sh_gate ? 1 : 0), md->dm, ne + 1, n, 1.0f);
    janas_g_compute_barrier(g);
    struct push_tk pk = {
        .ne = ne, .rs = ne + 1, .k = md->k, .n = n, .sh = ly->sh_gate != NULL};
    /* Gemma 4: each expert's own output scale, on its weight */
    janas_g_push_addr(
        pk.es, ly->exp_scale ? janas_gt_addr_of(g, ly->exp_scale, ne * 4) : 0);
    janas_g_push_addr(pk.r, B + t->o_r);
    janas_g_push_addr(pk.pe, B + t->o_pe);
    janas_g_push_addr(pk.pw, B + t->o_pw);
    janas_g_push_addr(pk.shw, B + t->o_shw);
    janas_gt_bind(g, bound, t->p_tk);
    janas_gt_push(g, &pk, sizeof(pk));
    g->vkCmdDispatch(g->cb, n, 1, 1); /* a workgroup a token */
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_ROUTER);
    return 0;
}

/* The matvec with ID for an expert's matrix of type ty on a block of n
   tokens (gpu.c pipe_mvid: a few vectors an expert, read at the weights'
   pace and bit for bit with the CPU), or NULL: *_mmid's tiles, 64 rows by
   16 vectors, most of them empty then. */
static VkPipeline by_rows(const struct janas_gpu *g,
                          const struct janas_gpu_tok *t, int ty, uint32_t n)
{
    int i = ty == JANAS_Q3_K     ? 0
            : ty == JANAS_Q4_K   ? 1
            : ty == JANAS_Q6_K   ? 2
            : ty == JANAS_Q6_K_P ? 3
            : ty == JANAS_MXFP4  ? 4
            : ty == JANAS_Q2_0   ? 5
            : ty == JANAS_Q5_1   ? 6
            : ty == JANAS_Q8_0   ? 7
                                 : -1;
    /* Q5_1 and Q8_0 have no tiles (*_mmid): by rows on any block */
    if (i >= 6)
        return g->pipe_mvid[i];
    return n <= t->mvid_n && i >= 0 ? g->pipe_mvid[i] : VK_NULL_HANDLE;
}

/* An expert product's dispatch, indirect: a matvec's (rows in the
   workgroups' y) or a *_mmid's (in x), the table at tab, its tiles from
   uvec4 tiles on; cols the rows' weights, nb their groups of 256 with the
   one a row may end inside */
static void experts_product(struct janas_gpu *g, struct janas_gpu_tok *t,
                            VkPipeline *bound, VkPipeline mv, VkPipeline id,
                            VkDeviceAddress tab, uint32_t rows, uint32_t cols,
                            uint32_t nb, uint32_t tiles, uint32_t ngroups,
                            size_t ind)
{
    if (mv) {
        /* the whole groups and the tail's blocks of 32 (b32_matvec's) */
        struct push pm = {.rows = rows,
                          .nb = cols / JANAS_QK,
                          .nv = tiles,
                          .tail = cols % JANAS_QK / 32};
        janas_g_push_addr(pm.w, tab);
        janas_gt_bind(g, bound, mv);
        janas_gt_push(g, &pm, sizeof(pm));
    } else {
        struct push_id pi = {
            .rows = rows, .nb = nb, .tiles = tiles, .ngroups = ngroups};
        janas_g_push_addr(pi.t, tab);
        janas_gt_bind(g, bound, id);
        janas_gt_push(g, &pi, sizeof(pi));
    }
    g->vkCmdDispatchIndirect(g->cb, t->buf.buf, ind);
}

/* A mixture's experts on the block whose router chose them (plan wrote
   their tables and sizes): the activations gathered in the experts'
   order, gate and up, the activation, down, then each token's weighted
   sum (and the shared expert) into YD. Experts in blocks of 32 (MXFP4)
   read the activations in that layout: the block's q32, and the
   activation quantized into it (expert_act.comp's KIND 1). */
void janas_gt_rec_experts(struct janas_gpu *g, struct janas_gpu_tok *t,
                          const struct janas_gpu_tok_model *md,
                          const struct janas_gpu_tok_layer *ly,
                          VkPipeline *bound, VkDeviceAddress YD, uint32_t n,
                          int part)
{
    VkDeviceAddress B = t->buf.addr;
    uint32_t np = n * md->k, nbd = md->dm / JANAS_QK, nbf = t->effp / JANAS_QK;
    int dt = ly->down.type, gtp = ly->gate.type;
    if (part == 2)
        goto sum;
    struct push_ga pg = {.nb = nbd};
    janas_g_push_addr(
        pg.src, B + (janas_g_x_in_32(ly->gate.type) ? t->o_q32 : t->o_q256));
    janas_g_push_addr(pg.dst, B + t->o_xg);
    janas_g_push_addr(pg.tok, B + t->o_gt);
    janas_gt_bind(g, bound, t->p_ga);
    janas_gt_push(g, &pg, sizeof(pg));
    g->vkCmdDispatch(g->cb, np, 1, 1);
    janas_g_compute_barrier(g);
    experts_product(g, t, bound, by_rows(g, t, gtp, n),
                    gtp == JANAS_MXFP4  ? g->pipe_idmx
                    : gtp == JANAS_Q3_K ? g->pipe_id3
                    : gtp == JANAS_Q2_0 ? g->pipe_idb[2]
                                        : g->pipe_id4,
                    B + t->o_tab1, md->eff, md->dm, nbd, t->tiles1, md->dm / 32,
                    t->o_ind);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_GATEUP);
    struct push_ac ac = {.blocks = np * nbf, .gelu = (uint32_t)t->gelu};
    janas_g_push_addr(ac.g, B + t->o_egu);
    janas_g_push_addr(ac.u, B + t->o_egu + (size_t)np * t->effp * 4);
    janas_g_push_addr(ac.x, B + t->o_eh);
    janas_gt_bind(g, bound, janas_g_x_in_32(dt) ? g->pipea32 : g->pipea);
    janas_gt_push(g, &ac, sizeof(ac));
    g->vkCmdDispatch(g->cb, np * nbf, 1, 1);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_ACT);
    experts_product(g, t, bound, by_rows(g, t, dt, n),
                    dt == JANAS_Q4_K    ? g->pipe_id4
                    : dt == JANAS_Q6_K  ? g->pipe_id6s
                    : dt == JANAS_MXFP4 ? g->pipe_idmx
                    : dt == JANAS_Q3_K  ? g->pipe_id3
                    : dt == JANAS_Q2_0  ? g->pipe_idb[2]
                                        : g->pipe_id6,
                    B + t->o_tab2, md->dm, md->eff, nbf, t->tiles2,
                    md->eff / 32, t->o_ind + 16);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_DOWN);
    if (part == 1)
        return;
sum:;
    struct push_cb pc = {
        .dm = md->dm, .k = md->k, .has_sh = ly->sh_gate_w.type != 0};
    janas_g_push_addr(pc.dout, B + t->o_dout);
    janas_g_push_addr(pc.pw, B + t->o_pw);
    janas_g_push_addr(pc.inv, B + t->o_inv);
    janas_g_push_addr(pc.sh, B + t->o_sh);
    janas_g_push_addr(pc.shw, B + t->o_shw);
    janas_g_push_addr(pc.y, YD);
    janas_gt_bind(g, bound, t->p_cb);
    janas_gt_push(g, &pc, sizeof(pc));
    g->vkCmdDispatch(g->cb, md->dm / JANAS_QK, n, 1);
    janas_g_compute_barrier(g);
    janas_gt_stamp(g, t, PH_COMBINE);
}

/* An entry of a *_mmid table: weights, activations, outputs (ys floats
   apart), vectors, and the down matrix's bare planes (or 0). */
static void table_entry(uint32_t *e, VkDeviceAddress w, VkDeviceAddress x,
                        VkDeviceAddress y, uint32_t nv, uint32_t ys,
                        VkDeviceAddress p0, VkDeviceAddress p1)
{
    janas_g_push_addr(e, w);
    janas_g_push_addr(e + 2, x);
    janas_g_push_addr(e + 4, y);
    e[6] = nv;
    e[7] = ys;
    janas_g_push_addr(e + 8, p0);
    janas_g_push_addr(e + 10, p1);
}

/* The experts read into the tier a stage at a time */
#define UP_SLOTS 8
/* The tier's slots looked at for the one to give up, from where the last
   search stopped: the one of the expert used least among them */
#define TIER_SCAN 16
/* The tier in allocations of 1 GiB at most: some drivers take no larger */
#define TIER_PIECE ((size_t)1 << 30)

static VkDeviceAddress tier_addr(const struct janas_gpu_tok *t, uint32_t s)
{
    return t->tier[s / t->per].addr +
           (VkDeviceAddress)(s % t->per) * t->slot_bytes;
}

int janas_gt_tier_create(struct janas_gpu *g, struct janas_gpu_tok *t,
                         const struct janas_gpu_tok_model *md)
{
    size_t sb = (md->slot_bytes + 255) / 256 * 256,
           all = (size_t)md->n_layer * md->n_expert;
    /* what the caller asks, within the room the card has left */
    size_t want = md->tier_bytes, room = janas_gpu_local_room(g);
    if (room && want > room)
        want = room;
    size_t per = TIER_PIECE / sb, n = want / sb;
    if (n > all)
        n = all;
    if (per == 0 || n > per * TIER_PIECES)
        n = per * TIER_PIECES;
    if (n < md->n_expert) /* not even a layer's: no tier, nor this path */
        return -1;
    for (size_t at = 0, i = 0; at < n; at += per, i++)
        if (janas_g_make_local(g, &t->tier[i],
                               (n - at < per ? n - at : per) * sb) != 0)
            return -1;
    t->per = (uint32_t)per;
    if (janas_g_make_stage(g, &t->up, UP_SLOTS * sb, 1) != 0)
        return -1;
    t->slot_of = malloc(all * sizeof(*t->slot_of));
    t->uses = calloc(all, sizeof(*t->uses));
    t->owner = malloc(n * sizeof(*t->owner));
    t->stamp = calloc(n, sizeof(*t->stamp));
    t->ra = malloc(md->n_expert * sizeof(*t->ra));
    t->rb = malloc(md->n_expert * sizeof(*t->rb));
    t->on_gpu = malloc(md->n_expert * sizeof(*t->on_gpu));
    t->gpu_slot = malloc(md->n_expert * sizeof(*t->gpu_slot));
    t->on_cpu = malloc(md->n_expert * sizeof(*t->on_cpu));
    t->miss = malloc(md->n_expert * sizeof(*t->miss));
    t->cpu_slots = malloc(md->n_expert * sizeof(*t->cpu_slots));
    t->cps = malloc((UP_SLOTS + md->n_expert) * sizeof(*t->cps));
    if (!t->slot_of || !t->uses || !t->owner || !t->stamp || !t->ra || !t->rb ||
        !t->on_gpu || !t->gpu_slot || !t->on_cpu || !t->miss || !t->cpu_slots ||
        !t->cps)
        return -1;
    for (size_t i = 0; i < all; i++)
        t->slot_of[i] = -1;
    for (size_t i = 0; i < n; i++)
        t->owner[i] = UINT32_MAX;
    t->n_slot = t->n_free = (uint32_t)n;
    t->slot_bytes = sb;
    return 0;
}

void janas_gt_tier_destroy(struct janas_gpu *g, struct janas_gpu_tok *t)
{
    for (int i = 0; i <= TIER_PIECES + 1; i++) {
        struct stage *st = i < TIER_PIECES    ? &t->tier[i]
                           : i == TIER_PIECES ? &t->up
                                              : &t->at;
        janas_g_free_stage(g, st);
    }
    free(t->slot_of);
    free(t->uses);
    free(t->owner);
    free(t->stamp);
    free(t->ra);
    free(t->rb);
    free(t->on_gpu);
    free(t->gpu_slot);
    free(t->on_cpu);
    free(t->miss);
    free(t->cpu_slots);
    free(t->cps);
}

/* A slot of the tier for expert key (layer * n_expert + e): a free one,
   or among TIER_SCAN the one of the expert used least, if less than key
   and no expert of this plan's (stamp); -1 if none. */
static int32_t tier_take(struct janas_gpu_tok *t, uint32_t key)
{
    int32_t s = -1;
    if (t->n_free) {
        s = (int32_t)(t->n_slot - t->n_free--);
    } else {
        uint32_t best = t->uses[key];
        for (uint32_t i = 0; i < TIER_SCAN && i < t->n_slot; i++) {
            uint32_t c = (t->scan + i) % t->n_slot, o = t->owner[c];
            if (t->stamp[c] != t->plan_no && t->uses[o] < best) {
                best = t->uses[o];
                s = (int32_t)c;
            }
        }
        t->scan = (t->scan + TIER_SCAN) % t->n_slot;
        if (s < 0)
            return -1;
        t->slot_of[t->owner[s]] = -1;
    }
    t->owner[s] = key;
    t->slot_of[key] = s;
    t->stamp[s] = t->plan_no;
    return s;
}

/* Experts slots[0..n) read into the tier's slots dst[0..n), through the
   stage, a submission for each UP_SLOTS of them but the last, whose copies
   are left in t->cps (their number): or -1. */
static int tier_load(struct janas_gpu *g, struct janas_gpu_tok *t,
                     const uint8_t *const *slots, const int32_t *dst,
                     uint32_t n, size_t bytes)
{
    uint32_t c = 0;
    for (uint32_t i0 = 0; i0 < n; i0 += UP_SLOTS) {
        c = n - i0 < UP_SLOTS ? n - i0 : UP_SLOTS;
        for (uint32_t i = 0; i < c; i++)
            memcpy(t->up.map + (size_t)i * t->slot_bytes, slots[i0 + i], bytes);
        if (!t->up.coherent) {
            VkMappedMemoryRange mr = {.sType =
                                          VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                      .memory = t->up.mem,
                                      .offset = 0,
                                      .size = VK_WHOLE_SIZE};
            g->vkFlushMappedMemoryRanges(g->dev, 1, &mr);
        }
        for (uint32_t i = 0; i < c; i++)
            t->cps[i] = (struct janas_g_cp){
                .src = t->up.buf,
                .dst = t->tier[dst[i0 + i] / t->per].buf,
                .so = (VkDeviceSize)i * t->slot_bytes,
                .dof = (VkDeviceSize)(dst[i0 + i] % t->per) * t->slot_bytes,
                .bytes = bytes};
        if (i0 + c < n && janas_g_copies(g, t->cps, c) != 0)
            return -1;
    }
    t->tier_loads += n;
    return (int)c;
}

/* The experts of a layer moe_plan.comp takes at most (its MAXE) */
#define GPLAN_MAX_EXPERTS 512

int janas_gt_tier_fill(struct janas_gpu *g, struct janas_gpu_tok *t,
                       const struct janas_gpu_tok_model *md)
{
    uint32_t ne = md->n_expert, nl = md->n_layer;
    if (t->gplan_tried)
        return t->gplan ? 0 : -1;
    t->gplan_tried = 1;
    const char *ev = getenv("JANAS_GPU_GPLAN");
    if ((ev && atoi(ev) == 0) || !t->p_pl || ne > GPLAN_MAX_EXPERTS ||
        t->n_slot < (size_t)nl * ne)
        return -1;
    size_t ab = (size_t)nl * ne * 48;
    if (janas_g_make_local(g, &t->at, ab) != 0)
        return -1;
    /* layer by layer, the next one read ahead, UP_SLOTS experts a stage */
    uint32_t ids[UP_SLOTS];
    const uint8_t *src[UP_SLOTS], *keep[UP_SLOTS];
    int32_t dst[UP_SLOTS];
    if (md->prefetch)
        md->prefetch(md->ctx, 0);
    for (uint32_t l = 0; l < nl; l++) {
        if (md->prefetch && l + 1 < nl)
            md->prefetch(md->ctx, l + 1);
        for (uint32_t e0 = 0; e0 < ne; e0 += UP_SLOTS) {
            uint32_t c = ne - e0 < UP_SLOTS ? ne - e0 : UP_SLOTS, m = 0;
            for (uint32_t i = 0; i < c; i++)
                ids[i] = e0 + i;
            if (md->fetch(md->ctx, l, ids, c, src) != 0)
                return -1;
            for (uint32_t i = 0; i < c; i++) {
                uint32_t key = l * ne + ids[i];
                if (t->slot_of[key] >= 0)
                    continue; /* there from an earlier plan */
                if ((dst[m] = tier_take(t, key)) < 0)
                    return -1;
                keep[m++] = src[i];
            }
            int ncp = m ? tier_load(g, t, keep, dst, m, md->slot_bytes) : 0;
            if (ncp < 0 || (ncp && janas_g_copies(g, t->cps, (uint32_t)ncp)))
                return -1;
        }
    }
    /* the addresses, through the stage */
    uint32_t *w = (uint32_t *)t->up.map;
    size_t per = t->up.bytes / 48; /* experts a copy */
    for (size_t k0 = 0; k0 < (size_t)nl * ne; k0 += per) {
        size_t c = (size_t)nl * ne - k0 < per ? (size_t)nl * ne - k0 : per;
        for (size_t i = 0; i < c; i++) {
            uint32_t key = (uint32_t)(k0 + i), l = key / ne;
            const struct janas_gpu_tok_layer *ly = &md->layers[l];
            VkDeviceAddress s = tier_addr(t, (uint32_t)t->slot_of[key]), p0 = 0,
                            p1 = 0;
            if (ly->down.type == JANAS_Q6_K_P) {
                p0 = ly->poff[0] ? s + ly->poff[0] : 0;
                p1 = ly->poff[1] ? s + ly->poff[1] : 0;
            }
            uint32_t *e = w + 12 * i;
            janas_g_push_addr(e, s + ly->off[0]);
            janas_g_push_addr(e + 2, s + ly->off[1]);
            janas_g_push_addr(e + 4, s + ly->off[2]);
            janas_g_push_addr(e + 6, p0);
            janas_g_push_addr(e + 8, p1);
            e[10] = e[11] = 0;
        }
        if (!t->up.coherent) {
            VkMappedMemoryRange mr = {.sType =
                                          VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                      .memory = t->up.mem,
                                      .offset = 0,
                                      .size = VK_WHOLE_SIZE};
            g->vkFlushMappedMemoryRanges(g->dev, 1, &mr);
        }
        if (janas_g_copy(g, t->up.buf, 0, t->at.buf, k0 * 48, c * 48) != 0)
            return -1;
    }
    t->gplan = 1;
    if (getenv("JANAS_GPU_VERBOSE"))
        fprintf(stderr,
                "gpu experts: every one in the tier, the plan on the GPU, "
                "a pass in one submission\n");
    return 0;
}

void janas_gt_rec_plan(struct janas_gpu *g, struct janas_gpu_tok *t,
                       const struct janas_gpu_tok_model *md, uint32_t l,
                       VkPipeline *bound, uint32_t n)
{
    const struct janas_gpu_tok_layer *ly = &md->layers[l];
    int r1 = by_rows(g, t, ly->gate.type, n) != VK_NULL_HANDLE,
        r2 = by_rows(g, t, ly->down.type, n) != VK_NULL_HANDLE;
    struct push_pl pp = {.pe = (uint32_t)t->o_pe,
                         .gt = (uint32_t)t->o_gt,
                         .inv = (uint32_t)t->o_inv,
                         .t1 = (uint32_t)t->o_tab1,
                         .t2 = (uint32_t)t->o_tab2,
                         .ind = (uint32_t)t->o_ind,
                         .xg = (uint32_t)t->o_xg,
                         .egu = (uint32_t)t->o_egu,
                         .eh = (uint32_t)t->o_eh,
                         .dout = (uint32_t)t->o_dout,
                         .n = n,
                         .k = md->k,
                         .ne = md->n_expert,
                         .nbd = md->dm / JANAS_QK,
                         .nbf = t->effp / JANAS_QK,
                         .ep = t->effp,
                         .dm = md->dm,
                         .eff = md->eff,
                         .s1 = r1 ? MVID_NV : 16,
                         .s2 = r2 ? MVID_NV : 16,
                         .tiles1 = t->tiles1,
                         .tiles2 = t->tiles2};
    janas_g_push_addr(pp.at,
                      t->at.addr + (VkDeviceAddress)l * md->n_expert * 48);
    janas_g_push_addr(pp.b, t->buf.addr);
    janas_gt_bind(g, bound, t->p_pl);
    janas_gt_push(g, &pp, sizeof(pp));
    g->vkCmdDispatch(g->cb, 1, 1, 1);
    /* the tables read by the products, their sizes by the dispatches */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                           VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
    g->vkCmdPipelineBarrier(g->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                            0, 1, &mb, 0, NULL, 0, NULL);
    janas_gt_stamp(g, t, PH_BETWEEN);
}

/* The host's side of bytes [off, off + bytes) of buf: on a card with
   memory of its own the mirror's cache lines (the pass's commands copy
   them), else buf's own. */
static void host_side(struct janas_gpu *g, struct janas_gpu_tok *t, size_t off,
                      size_t bytes, int flush)
{
    if (t->local)
        janas_gt_mirror_cache(g, t, off, bytes, flush);
    else
        janas_gt_sync_range(g, t, off, bytes, flush);
}

/*
 * Between two of a mixture's submissions, on the CPU: layer l's choice (n
 * tokens, k experts each) read back and sorted by expert, as model_ffn.c
 * does - each row's token, each pair's row - the experts' slots asked of
 * the model, the tables of the products and the sizes of their dispatches
 * written. 0, or -1 (an expert unreadable, or out of the GPU's reach).
 */
static int plan(struct janas_gpu *g, struct janas_gpu_tok *t,
                const struct janas_gpu_tok_model *md, uint32_t l, uint32_t n)
{
    const struct janas_gpu_tok_layer *ly = &md->layers[l];
    uint32_t k = md->k, ne = md->n_expert, np = n * k, nd = 0;
    uint32_t nbd = md->dm / JANAS_QK, nbf = t->effp / JANAS_QK, ep = t->effp;
    host_side(g, t, t->o_pe, (size_t)np * 4, 0);
    const uint32_t *pe = (const uint32_t *)(t->buf.map + t->o_pe);
    uint32_t *cnt = t->cnt, *gt = (uint32_t *)(t->buf.map + t->o_gt),
             *inv = (uint32_t *)(t->buf.map + t->o_inv);
    memset(cnt, 0, (ne + 1) * sizeof(*cnt));
    for (uint32_t p = 0; p < np; p++) {
        if (pe[p] >= ne)
            return -1;
        cnt[pe[p] + 1]++;
    }
    if (md->routed)
        md->routed(md->ctx, l, n, k, pe);
    for (uint32_t e = 0; e < ne; e++) {
        if (cnt[e + 1])
            t->ids[nd++] = e;
        cnt[e + 1] += cnt[e];
        t->fill[e] = cnt[e];
    }
    for (uint32_t p = 0; p < np; p++) {
        uint32_t r = t->fill[pe[p]]++;
        gt[r] = p / k;
        inv[p] = r;
    }
    /* the experts the GPU computes, and where it reads them: all of them,
       as the cache holds them; or, with a tier, those in it or read into
       it now, the others computed by the CPU into their rows */
    uint32_t ng = 0, nc = 0, nm = 0;
    int ncp = 0; /* copies left for one submission */
    if (!t->n_slot) {
        if (md->fetch(md->ctx, l, t->ids, nd, t->slots) != 0)
            return -1;
        for (uint32_t i = 0; i < nd; i++)
            t->ids[ng++] = t->ids[i];
    } else {
        uint32_t base = l * ne;
        t->plan_no++;
        for (uint32_t i = 0; i < nd; i++) {
            uint32_t e = t->ids[i];
            t->uses[base + e]++;
            if (t->slot_of[base + e] >= 0) {
                t->stamp[t->slot_of[base + e]] = t->plan_no;
                t->on_gpu[ng] = e;
                t->gpu_slot[ng++] = (uint32_t)t->slot_of[base + e];
                t->tier_hits++;
            } else {
                t->miss[nm++] = e;
            }
        }
        struct timespec f0, f1;
        if (t->qp)
            clock_gettime(CLOCK_MONOTONIC, &f0);
        if (nm && md->fetch(md->ctx, l, t->miss, nm, t->cpu_slots) != 0)
            return -1;
        if (t->qp) {
            clock_gettime(CLOCK_MONOTONIC, &f1);
            t->fetch_ns += (double)(f1.tv_sec - f0.tv_sec) * 1e9 +
                           (double)(f1.tv_nsec - f0.tv_nsec);
        }
        /* misses into the tier where it gives up an expert used less;
           the rest to the CPU */
        int32_t *dst = (int32_t *)t->fill; /* free until the sort below */
        const uint8_t **src = t->slots;
        uint32_t nl = 0;
        for (uint32_t i = 0; i < nm; i++) {
            uint32_t e = t->miss[i];
            int32_t s2 = tier_take(t, base + e);
            if (s2 >= 0) {
                dst[nl] = s2;
                t->miss[nl] = e; /* nl <= i: read already */
                src[nl++] = t->cpu_slots[i];
                t->on_gpu[ng] = e;
                t->gpu_slot[ng++] = (uint32_t)s2;
            } else {
                t->on_cpu[nc] = e;
                t->cpu_slots[nc++] = t->cpu_slots[i];
            }
        }
        if (nl && (ncp = tier_load(g, t, src, dst, nl, md->slot_bytes)) < 0)
            return -1;
        /* once on the card the cache's copies go first: JANAS_GPU_DEMOTE=0
           keeps them where they were */
        if (nl && md->demote)
            md->demote(md->ctx, l, t->miss, nl);
        for (uint32_t i = 0; i < ng; i++)
            t->ids[i] = t->on_gpu[i];
        t->cpu_experts += nc;
    }
    /* the fill of each expert again (t->fill held the slots above) */
    for (uint32_t e = 0; e < ne; e++)
        t->fill[e] = cnt[e];
    VkDeviceAddress B = t->buf.addr;
    uint32_t *t1 = (uint32_t *)(t->buf.map + t->o_tab1),
             *t2 = (uint32_t *)(t->buf.map + t->o_tab2);
    uint32_t n1 = 0, n2 = 0; /* tiles */
    /* vectors a tile: a matvec's workgroup's, or a *_mmid's */
    int r1 = by_rows(g, t, ly->gate.type, n) != VK_NULL_HANDLE,
        r2 = by_rows(g, t, ly->down.type, n) != VK_NULL_HANDLE;
    uint32_t s1 = r1 ? MVID_NV : 16, s2 = r2 ? MVID_NV : 16;
    /* the matrices' bytes; a down matrix's rows may end inside a group */
    size_t gb = (size_t)md->eff * janas_qtype_row_bytes(ly->gate.type, md->dm),
           db = (size_t)md->dm * janas_qtype_row_bytes(ly->down.type, md->eff) +
                4,
           pb = (size_t)md->dm * nbf * JANAS_Q6KP_PLANE;
    for (uint32_t i = 0; i < ng; i++) {
        uint32_t e = t->ids[i], a = cnt[e], nv = cnt[e + 1] - a;
        VkDeviceAddress wg, wu, wd, p0 = 0, p1 = 0;
        if (t->n_slot) {
            VkDeviceAddress s = tier_addr(t, t->gpu_slot[i]);
            wg = s + ly->off[0], wu = s + ly->off[1], wd = s + ly->off[2];
            if (ly->down.type == JANAS_Q6_K_P) {
                p0 = ly->poff[0] ? s + ly->poff[0] : 0;
                p1 = ly->poff[1] ? s + ly->poff[1] : 0;
            }
        } else {
            const uint8_t *s = t->slots[i];
            wg = janas_gt_addr_of(g, s + ly->off[0], gb);
            wu = janas_gt_addr_of(g, s + ly->off[1], gb);
            wd = janas_gt_addr_of(g, s + ly->off[2], db);
            /* the planes the cache holds (none, one or both: level 1-3) */
            if (ly->down.type == JANAS_Q6_K_P &&
                ((ly->poff[0] &&
                  !(p0 = janas_gt_addr_of(g, s + ly->poff[0], pb))) ||
                 (ly->poff[1] &&
                  !(p1 = janas_gt_addr_of(g, s + ly->poff[1], pb)))))
                return -1;
        }
        if (!wg || !wu || !wd)
            return -1;
        VkDeviceAddress x = B + t->o_xg + (VkDeviceAddress)a * nbd * 320;
        /* gate and up at ep floats a pair: the rows past eff stay zero */
        table_entry(t1 + 12 * (2 * i), wg, x,
                    B + t->o_egu + (VkDeviceAddress)a * ep * 4, nv, ep, 0, 0);
        table_entry(t1 + 12 * (2 * i + 1), wu, x,
                    B + t->o_egu + ((VkDeviceAddress)np + a) * ep * 4, nv, ep,
                    0, 0);
        table_entry(t2 + 12 * i, wd,
                    B + t->o_eh + (VkDeviceAddress)a * nbf * 320,
                    B + t->o_dout + (VkDeviceAddress)a * md->dm * 4, nv, md->dm,
                    p0, p1);
        for (uint32_t v = 0; v < nv; v += s1) {
            uint32_t *u = t1 + 4 * (t->tiles1 + n1);
            u[0] = 2 * i, u[1] = v, u[2] = u[3] = 0;
            u[4] = 2 * i + 1, u[5] = v, u[6] = u[7] = 0;
            n1 += 2;
        }
        for (uint32_t v = 0; v < nv; v += s2) {
            uint32_t *u = t2 + 4 * (t->tiles2 + n2++);
            u[0] = i, u[1] = v, u[2] = u[3] = 0;
        }
    }
    /* a matvec's rows in its workgroups' y, a *_mmid's in x */
    uint32_t *ind = (uint32_t *)(t->buf.map + t->o_ind);
    ind[0] = r1 ? 1 : (md->eff + 63) / 64, ind[2] = n1;
    ind[1] = r1 ? (md->eff + 63) / 64 : 1;
    ind[4] = r2 ? 1 : (md->dm + 63) / 64, ind[6] = n2;
    ind[5] = r2 ? (md->dm + 63) / 64 : 1;
    /* o_gt to o_ind are one range (gpu_token.c's layout), copied by the
       next submission */
    host_side(g, t, t->o_gt, t->o_ind + 32 - t->o_gt, 1);
    /* the last experts read into the tier, in the next segment's
       submission: a submission of their own, waited for, cost a round trip
       a layer; the CPU's experts left for cpu_part */
    if (ncp && janas_g_record_copies(g, t->cps, (uint32_t)ncp, 1) != 0)
        return -1;
    t->cp_pending = ncp != 0;
    t->pend_nc = nc;
    return 0;
}

/*
 * The experts plan left to the CPU (t->pend_nc), from layer l's normalized
 * input (copied to the mirror by the last submission) into their rows,
 * while the GPU computes its own; their copies to the card's memory
 * recorded in t->cb_rows, after a barrier on all that went before, for the
 * submission of the rest of the segment. 0, or -1.
 */
static int cpu_part(struct janas_gpu *g, struct janas_gpu_tok *t,
                    const struct janas_gpu_tok_model *md, uint32_t l,
                    uint32_t n)
{
    uint32_t nc = t->pend_nc, *cnt = t->cnt,
             *gt = (uint32_t *)(t->buf.map + t->o_gt), ncp = 0;
    for (uint32_t i = 0; i < nc; i++)
        t->ra[i] = cnt[t->on_cpu[i]], t->rb[i] = cnt[t->on_cpu[i] + 1];
    host_side(g, t, t->o_xnf, (size_t)n * md->dm * 4, 0);
    /* a single token: layer l + 1's likeliest experts not on the card read
       ahead, while the CPU computes its own and the GPU the rest of the
       segment, as the CPU's path reads ahead. Without it every miss was
       read when its layer asked, the GPU waiting: Qwen3-Next-80B with a
       4 GiB cache and a card of 2.5 GiB (issue #63), 39 ms of a pass's 190
       in the cache's reads, 0 read ahead (14 Nov 2026) */
    if (n == 1 && md->rank_ahead && md->read_ahead && t->n_slot &&
        l + 1 < md->n_layer && md->ahead_k <= 32) {
        uint32_t ids[32], keep[32], nk = 0, ne = md->n_expert;
        if (md->rank_ahead(md->ctx, l, (const float *)(t->buf.map + t->o_xnf),
                           md->ahead_k, ids) == 0) {
            for (uint32_t i = 0; i < md->ahead_k; i++)
                if (t->slot_of[(size_t)(l + 1) * ne + ids[i]] < 0)
                    keep[nk++] = ids[i];
            if (nk)
                md->read_ahead(md->ctx, l + 1, keep, nk);
        }
    }
    struct timespec c0, c1;
    if (t->qp)
        clock_gettime(CLOCK_MONOTONIC, &c0);
    if (md->cpu_rows(md->ctx, l, n, (const float *)(t->buf.map + t->o_xnf), nc,
                     t->on_cpu, t->cpu_slots, t->ra, t->rb, gt,
                     (float *)(t->buf.map + t->o_dout)) != 0)
        return -1;
    if (t->qp) {
        clock_gettime(CLOCK_MONOTONIC, &c1);
        t->cpu_ns += (double)(c1.tv_sec - c0.tv_sec) * 1e9 +
                     (double)(c1.tv_nsec - c0.tv_nsec);
    }
    for (uint32_t i = 0; i < nc; i++) {
        size_t o = t->o_dout + (size_t)t->ra[i] * md->dm * 4,
               b = (size_t)(t->rb[i] - t->ra[i]) * md->dm * 4;
        host_side(g, t, o, b, 1);
        if (t->local)
            t->cps[ncp++] = (struct janas_g_cp){.src = t->mirror.buf,
                                                .dst = t->buf.buf,
                                                .so = o,
                                                .dof = o,
                                                .bytes = b};
    }
    if (!t->cb_rows) {
        VkCommandBufferAllocateInfo ca = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = g->cp,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};
        if (g->vkAllocateCommandBuffers(g->dev, &ca, &t->cb_rows) !=
            VK_SUCCESS) {
            t->cb_rows = VK_NULL_HANDLE;
            return -1;
        }
    }
    VkCommandBufferBeginInfo bb = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (g->vkResetCommandBuffer(t->cb_rows, 0) != VK_SUCCESS ||
        g->vkBeginCommandBuffer(t->cb_rows, &bb) != VK_SUCCESS)
        return -1;
    /* after the products submitted before, without a wait between */
    VkMemoryBarrier all = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                                            VK_ACCESS_MEMORY_WRITE_BIT};
    g->vkCmdPipelineBarrier(t->cb_rows, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &all, 0,
                            NULL, 0, NULL);
    for (uint32_t i = 0; i < ncp; i++) {
        VkBufferCopy cp = {.srcOffset = t->cps[i].so,
                           .dstOffset = t->cps[i].dof,
                           .size = t->cps[i].bytes};
        g->vkCmdCopyBuffer(t->cb_rows, t->cps[i].src, t->cps[i].dst, 1, &cp);
    }
    if (ncp)
        g->vkCmdPipelineBarrier(t->cb_rows, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &all,
                                0, NULL, 0, NULL);
    return g->vkEndCommandBuffer(t->cb_rows) == VK_SUCCESS ? 0 : -1;
}

/* A mixture's block: its n_layer + 1 submissions, each recorded once for
   its shape and runs, the CPU's plan between two of them. 0, or -1. */
int janas_gt_run_segments(struct janas_gpu *g, struct janas_gpu_tok *t,
                          const struct janas_gpu_tok_model *md, uint32_t n,
                          uint32_t first, int ai, int hi, uint32_t nch)
{
    VkCommandBuffer **seg = &t->seg_cb[n][ai][hi];
    if (*seg && t->seg_nch[n][ai][hi] != nch)
        janas_gt_free_segments(g, t, seg); /* recorded for other runs: again */
    if (!*seg) {
        *seg = calloc(2 * t->n_seg, sizeof(**seg));
        if (!*seg)
            return -1;
        t->seg_nch[n][ai][hi] = nch;
    }
    /* a block whose tokens choose some 4 times each expert or more: those
       few that no token takes are under 2% (e^-4) */
    int ahead = md->prefetch && (size_t)n * md->k >= 4 * (size_t)md->n_expert;
    t->cp_pending = 0, t->pend_nc = 0;
    for (uint32_t s = 0; s < t->n_seg; s++) {
        struct timespec p0, p1;
        if (t->qp)
            clock_gettime(CLOCK_MONOTONIC, &p0);
        if (s && plan(g, t, md, s - 1, n) != 0)
            return -1;
        if (t->qp) {
            clock_gettime(CLOCK_MONOTONIC, &p1);
            t->plan_ns += (double)(p1.tv_sec - p0.tv_sec) * 1e9 +
                          (double)(p1.tv_nsec - p0.tv_nsec);
        }
        /* layer s's experts read while the GPU computes up to its router's
           choice, as the CPU's path reads the next layer's (model_ffn.c
           janas_m_prefetch_block): without, the GPU waited for each
           layer's reads, and on Qwen3.6-35B, whose experts do not all fit,
           the cache found 79% of them instead of 87% and the prompt was
           no faster (10 Oct 2026) */
        if (ahead && s + 1 < t->n_seg)
            md->prefetch(md->ctx, s);
        /* past the first: the segment's products of the experts, then the
           rest - the CPU computing its experts between the two
           submissions, the GPU its own (qwen4exp's streams whole) */
        int split = s && !md->hc;
        VkCommandBuffer *a = &(*seg)[2 * s], *b = &(*seg)[2 * s + 1];
        if ((!*a && janas_gt_make_cb(g, t, md, a, n, first, hi, nch, (int)s,
                                     split) != 0) ||
            (split && !*b &&
             janas_gt_make_cb(g, t, md, b, n, first, hi, nch, (int)s, 2) != 0))
            return -1;
        VkCommandBuffer pre = t->cp_pending ? g->cb : VK_NULL_HANDLE;
        t->cp_pending = 0;
        if (!split) {
            if (janas_gt_submit_wait2(g, pre, a) != 0)
                return -1;
        } else if (!t->pend_nc) {
            VkCommandBuffer l3[3] = {pre, *a, *b};
            if (janas_gt_submit_list(g, l3, 3, 1) != 0)
                return -1;
        } else {
            VkCommandBuffer l2[2] = {pre, *a};
            if (janas_gt_submit_list(g, l2, 2, 0) != 0 ||
                cpu_part(g, t, md, s - 1, n) != 0)
                return -1;
            VkCommandBuffer r2[2] = {t->cb_rows, *b};
            if (janas_gt_submit_list(g, r2, 2, 1) != 0)
                return -1;
        }
        t->pend_nc = 0;
    }
    return 0;
}

#endif
