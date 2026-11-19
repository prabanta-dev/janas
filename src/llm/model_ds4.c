/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_ds4.c - DeepSeek-V4 (deepseek4): its hyperparameters, its layers
 * and their loop, its state per sequence. The specification it follows is
 * DeepSeek's own reference code (inference/model.py), as llama.cpp's
 * deepseek4 does.
 *
 * The residual is four streams of d_model (manifold-constrained
 * hyper-connections). Before each sublayer, attention then the mixture, a
 * mix of 24 numbers from the four streams flattened (16384 numbers, their
 * RMS taken without a weight): four weights of the streams for the
 * sublayer's input (sigmoid + eps), four for its output into each stream
 * (2 sigmoid), and a 4 x 4 matrix that mixes the old streams, made nearly
 * doubly stochastic by Sinkhorn's iterations. The final mix takes four
 * weights only. The first stream state is the token's embedding, four
 * times.
 *
 * The attention: 64 heads of 512 over one latent per position that is
 * both key and value (no per-head key or value): q from a low-rank pair
 * (wq_a, its norm, wq_b), each head normalized without a weight; the
 * latent from wkv and its norm; RoPE on the last 64 numbers of each, pairs
 * of adjacent numbers. A query sees the latents of the last 128 positions
 * (itself included) and, in the layers that have them, compressed latents:
 * one per 128 positions (all of them), or one per 4 (overlapping windows of
 * 8), of which an indexer chooses the 512 most relevant when there are
 * more. The scores scaled by 512^-0.5, a sink per head in the softmax's
 * sum; the output turned back by its position's RoPE, then a low-rank
 * projection by groups of 8 heads (wo_a, block-diagonal) and wo_b.
 *
 * The mixture (model_ffn.c, with this file's route): scores sqrt(softplus)
 * of the router's logits; the first three layers take their six experts
 * from a table by token id, the others the six best by score plus a bias;
 * weights the scores normalized, times 1.5; SwiGLU clamped at 10; a shared
 * expert, weight 1.
 *
 * What the reference rounds to FP8 or FP4 between operations (the inputs
 * of its quantized products, the latents) is kept in float here, as
 * llama.cpp does: the differences are of the reference's own noise.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "bf16.h"
#include "common/aligned.h"
#include "matvec.h"
#include "model_impl.h"

#define HC 4                /* streams */
#define MIX (HC * (2 + HC)) /* the mix's numbers: pre, post, the matrix */
#define LAT 512             /* the latent, a head */
#define ROT 64              /* its last numbers turned */
#define WIN 128             /* the window of raw latents */
#define IXD 128             /* the indexer's head and key */
#define MAX_TOPK 4096

struct ds4_layer {
    uint32_t ratio; /* 0, 4 or 128 */
    const struct janas_jns_tensor *hc_fn[2], *hc_base[2], *hc_scale[2];
    const struct janas_jns_tensor *wq_a, *q_norm, *wq_b, *wkv, *kv_norm, *sink,
        *wo_a, *wo_b;
    /* the compressor of latents (ratio > 0), the indexer's (ratio 4) */
    const struct janas_jns_tensor *c_kv, *c_gate, *c_ape, *c_norm;
    const struct janas_jns_tensor *ix_qb, *ix_proj, *ix_kv, *ix_gate, *ix_ape,
        *ix_norm;
    const float *router; /* n_expert x d_model */
    const float *bias;   /* the choice's, NULL in the hash layers */
    const int32_t *tid2eid;
};

/* What a sequence leaves behind, per layer: the raw latents of the last
   positions (a ring of cap), the compressors' products of those positions
   (a, then s, each cw wide), the compressed latents and the indexer's
   keys (from the start, one per ratio positions). */
struct janas_ds4_seq {
    uint32_t cap, n_ent;
    float **raw, **ca, **ent, **ixa, **ixk;
};

struct janas_ds4 {
    uint32_t n_hash, sk_iter, q_rank, o_groups, o_rank, ix_heads, ix_topk;
    float hc_eps, limit;
    float f_plain[ROT / 2], f_yarn[ROT / 2];
    const struct janas_jns_tensor *head_fn, *head_base, *head_scale;
    struct ds4_layer *L;
    /* a block's scratch */
    float *S, *mix, *qa, *q, *kv, *o, *r, *ca, *ixa, *qi, *ixw;
    struct janas_block_q8k *qaq, *oq, *rq;
    float *work; /* per thread: scores and the indexer's */
    size_t work_per;
    uint32_t n_threads;
    /* JANAS_DS4_LAYERS=0,2,3: only those layers, in that order (a model
       cut down to compare with llama.cpp on the same layers in a GGUF of
       their own, job/tests/c/jns2gguf_ds4.c); n_only 0: all */
    uint32_t only[64], n_only;
    /* the multi-token prediction block (its own file, load_mtp): its
       layer, and its final mix of the streams */
    struct ds4_layer mtpL;
    const struct janas_jns_tensor *mtp_head_fn, *mtp_head_base, *mtp_head_scale;
};

static int meta_u32(struct janas_llm_model *m, const char *suffix, uint32_t *v)
{
    char key[128];
    snprintf(key, sizeof(key), "deepseek4.%s", suffix);
    return janas_jns_meta_u32(&m->j, key, v);
}

static int meta_f32(struct janas_llm_model *m, const char *suffix, float *v)
{
    char key[128];
    snprintf(key, sizeof(key), "deepseek4.%s", suffix);
    return janas_jns_meta_f32(&m->j, key, v);
}

/* The rotation's frequencies: theta^(-2i / 64); with YaRN (the layers with
   compressed latents) those past the ramp divided by the factor, without
   YaRN's scale of the magnitude. */
static void freqs(float *f, float theta, float factor, float orig,
                  float beta_fast, float beta_slow)
{
    for (int i = 0; i < ROT / 2; i++)
        f[i] = powf(theta, -2.0f * (float)i / (float)ROT);
    if (factor <= 1.0f)
        return;
    double lo =
        floor(ROT * log(orig / (beta_fast * 2.0 * M_PI)) / (2.0 * log(theta)));
    double hi =
        ceil(ROT * log(orig / (beta_slow * 2.0 * M_PI)) / (2.0 * log(theta)));
    if (lo < 0)
        lo = 0;
    if (hi > ROT - 1)
        hi = ROT - 1;
    for (int i = 0; i < ROT / 2; i++) {
        double ramp = hi > lo ? (i - lo) / (hi - lo) : 1.0;
        ramp = ramp < 0 ? 0 : ramp > 1 ? 1 : ramp;
        f[i] = (float)(f[i] / factor * ramp + f[i] * (1.0 - ramp));
    }
}

int janas_m_ds4_params(struct janas_llm_model *m)
{
    struct janas_ds4 *d = calloc(1, sizeof(*d));
    if (!d)
        return -1;
    m->ds4 = d;
    float theta = 10000, ctheta = 160000, factor = 1, fast = 32, slow = 1,
          orig = 65536;
    uint32_t hc = 0;
    if (meta_u32(m, "hash_layer_count", &d->n_hash) ||
        meta_u32(m, "hyper_connection.count", &hc) ||
        meta_u32(m, "hyper_connection.sinkhorn_iterations", &d->sk_iter) ||
        meta_f32(m, "hyper_connection.epsilon", &d->hc_eps) ||
        meta_u32(m, "attention.q_lora_rank", &d->q_rank) ||
        meta_u32(m, "attention.output_group_count", &d->o_groups) ||
        meta_u32(m, "attention.output_lora_rank", &d->o_rank) ||
        meta_u32(m, "attention.indexer.head_count", &d->ix_heads) ||
        meta_u32(m, "attention.indexer.top_k", &d->ix_topk) ||
        meta_f32(m, "expert_weights_scale", &m->route_scale) ||
        meta_f32(m, "rope.freq_base", &theta) ||
        meta_f32(m, "attention.compress_rope_freq_base", &ctheta))
        return -1;
    meta_f32(m, "rope.scaling.factor", &factor);
    meta_f32(m, "rope.scaling.yarn_beta_fast", &fast);
    meta_f32(m, "rope.scaling.yarn_beta_slow", &slow);
    {
        uint32_t o = 0;
        if (meta_u32(m, "rope.scaling.original_context_length", &o) == 0)
            orig = (float)o;
    }
    /* the clamp of the experts' SwiGLU: the same for every layer */
    uint64_t v = 0, cnt = 0;
    d->limit = 10.0f;
    if (janas_jns_meta_uint_at(&m->j, "deepseek4.swiglu_clamp_exp", 0, &v,
                               &cnt) != 0) {
        float f;
        if (meta_f32(m, "swiglu_limit", &f) == 0)
            d->limit = f;
    }
    m->swiglu_limit = d->limit;
    if (hc != HC || d->sk_iter == 0 || m->head_dim != LAT || m->n_rot != ROT ||
        m->n_head_kv != 1 || d->q_rank % JANAS_QK || d->o_groups == 0 ||
        m->n_head % d->o_groups || (m->n_head / d->o_groups * LAT) % JANAS_QK ||
        d->o_rank % JANAS_QK || d->ix_heads == 0 || d->ix_topk == 0 ||
        d->ix_topk > MAX_TOPK || m->d_model % JANAS_QK)
        return -1;
    const char *ol = getenv("JANAS_DS4_LAYERS");
    for (const char *p = ol; p && *p && d->n_only < 64;) {
        char *e;
        unsigned long v = strtoul(p, &e, 10);
        if (e == p || v >= m->n_layer)
            return -1;
        d->only[d->n_only++] = (uint32_t)v;
        p = *e == ',' ? e + 1 : e;
    }
    freqs(d->f_plain, theta, 1.0f, orig, fast, slow);
    freqs(d->f_yarn, ctheta, factor, orig, fast, slow);
    return 0;
}

/* A tensor: name blk.l.suffix (l < 0: suffix alone), type (0 f32, 30 bf16,
   26 i32, -1 any type the products read), dims d0 x d1 (d1 0: one). */
static const struct janas_jns_tensor *need(const struct janas_jns *j, int l,
                                           const char *suffix, int type,
                                           uint64_t d0, uint64_t d1, char *err,
                                           size_t err_len)
{
    char name[96];
    if (l < 0)
        snprintf(name, sizeof(name), "%s", suffix);
    else
        snprintf(name, sizeof(name), "blk.%d.%s", l, suffix);
    const struct janas_jns_tensor *t = janas_jns_tensor(j, name);
    int ok = t && t->dims[0] == d0 && (d1 ? t->dims[1] == d1 : 1);
    if (ok)
        ok = type >= 0 ? (int)t->type == type
                       : janas_qtype_block_size((int)t->type) != 0;
    if (!ok)
        snprintf(err, err_len, "tensor %s missing or unexpected", name);
    return ok ? t : NULL;
}

static const struct janas_jns_tensor *maybe(const struct janas_jns *j, int l,
                                            const char *suffix)
{
    char name[96];
    snprintf(name, sizeof(name), "blk.%d.%s", l, suffix);
    return janas_jns_tensor(j, name);
}

int janas_m_ds4_layer_load(struct janas_llm_model *m, const struct janas_jns *j,
                           uint32_t l, struct layer *ly, char *err,
                           size_t err_len)
{
    struct janas_ds4 *d = m->ds4;
    int main = j == &m->j; /* else the MTP block's file, its one layer */
    if (main && !d->L && !(d->L = calloc(m->n_layer, sizeof(*d->L))))
        return -1;
    struct ds4_layer *D = main ? &d->L[l] : &d->mtpL;
    uint32_t dm = m->d_model, nh = m->n_head, ne = m->n_expert;
    uint64_t v = 0, cnt = 0;
    /* the MTP block attends its window alone */
    if (main &&
        (janas_jns_meta_uint_at(&m->j, "deepseek4.attention.compress_ratios", l,
                                &v, &cnt) != 0 ||
         (v != 0 && v != 4 && v != 128))) {
        snprintf(err, err_len, "layer %u: no compression ratio", l);
        return -1;
    }
    D->ratio = main ? (uint32_t)v : 0;
    ly->ds4l = D;
    int L = (int)l;
#define NEED(var, suffix, type, d0, d1)                                        \
    if (!(var = need(j, L, suffix, type, d0, d1, err, err_len)))               \
        return -1;
    static const char *const part[2] = {"attn", "ffn"};
    for (int s = 0; s < 2; s++) {
        char nm[48];
        snprintf(nm, sizeof nm, "hc_%s_fn.weight", part[s]);
        NEED(D->hc_fn[s], nm, 0, (uint64_t)HC * dm, MIX)
        snprintf(nm, sizeof nm, "hc_%s_base.weight", part[s]);
        NEED(D->hc_base[s], nm, 0, MIX, 0)
        snprintf(nm, sizeof nm, "hc_%s_scale.weight", part[s]);
        NEED(D->hc_scale[s], nm, 0, 3, 0)
    }
    NEED(ly->attn_norm, "attn_norm.weight", 0, dm, 0)
    NEED(ly->ffn_norm, "ffn_norm.weight", 0, dm, 0)
    NEED(D->wq_a, "attn_q_a.weight", -1, dm, d->q_rank)
    NEED(D->q_norm, "attn_q_a_norm.weight", 0, d->q_rank, 0)
    NEED(D->wq_b, "attn_q_b.weight", -1, d->q_rank, (uint64_t)nh * LAT)
    NEED(D->wkv, "attn_kv.weight", -1, dm, LAT)
    NEED(D->kv_norm, "attn_kv_a_norm.weight", 0, LAT, 0)
    NEED(D->sink, "attn_sinks.weight", 0, nh, 0)
    NEED(D->wo_a, "attn_output_a.weight", -1, (uint64_t)nh / d->o_groups * LAT,
         (uint64_t)d->o_groups * d->o_rank)
    NEED(D->wo_b, "attn_output_b.weight", -1, (uint64_t)d->o_groups * d->o_rank,
         dm)
    if (D->ratio) {
        uint32_t cw = LAT * (D->ratio == 4 ? 2 : 1);
        NEED(D->c_kv, "attn_compressor_kv.weight", 30, dm, cw)
        NEED(D->c_gate, "attn_compressor_gate.weight", 30, dm, cw)
        NEED(D->c_ape, "attn_compressor_ape.weight", 0, cw, D->ratio)
        NEED(D->c_norm, "attn_compressor_norm.weight", 0, LAT, 0)
    }
    if (D->ratio == 4) {
        NEED(D->ix_qb, "indexer.attn_q_b.weight", -1, d->q_rank,
             (uint64_t)d->ix_heads * IXD)
        NEED(D->ix_proj, "indexer.proj.weight", 30, dm, d->ix_heads)
        NEED(D->ix_kv, "indexer_compressor_kv.weight", 30, dm, 2 * IXD)
        NEED(D->ix_gate, "indexer_compressor_gate.weight", 30, dm, 2 * IXD)
        NEED(D->ix_ape, "indexer_compressor_ape.weight", 0, 2 * IXD, 4)
        NEED(D->ix_norm, "indexer_compressor_norm.weight", 0, IXD, 0)
    }
    const struct janas_jns_tensor *t;
    NEED(t, "ffn_gate_inp.weight", 0, dm, ne)
    D->router = f32(m, t);
    /* the first layers route by a table of token ids, the others by score
       and bias: as the file has them */
    if (maybe(j, L, "ffn_gate_tid2eid.weight")) {
        NEED(t, "ffn_gate_tid2eid.weight", 26, m->k, 0)
        D->tid2eid = (const int32_t *)data(m, t);
        if (t->dims[1] < m->n_vocab) {
            snprintf(err, err_len, "layer %u: the hash table is short", l);
            return -1;
        }
        for (uint64_t i = 0; i < t->dims[0] * t->dims[1]; i++)
            if (D->tid2eid[i] < 0 || (uint32_t)D->tid2eid[i] >= ne) {
                snprintf(err, err_len, "layer %u: the hash table's experts", l);
                return -1;
            }
    } else {
        NEED(t, "exp_probs_b.bias", 0, ne, 0)
        D->bias = f32(m, t);
    }
    /* the shared expert: three matrices of the resident region, as a slot
       at offset 0 each */
    static const char *const sh[3] = {"ffn_gate_shexp.weight",
                                      "ffn_up_shexp.weight",
                                      "ffn_down_shexp.weight"};
    for (int p = 0; p < 3; p++) {
        t = maybe(j, L, sh[p]);
        uint64_t c = p < 2 ? dm : m->d_ff, r = p < 2 ? m->d_ff : dm;
        if (!t || t->dims[0] != c || t->dims[1] != r ||
            janas_qtype_block_size((int)t->type) == 0) {
            snprintf(err, err_len, "layer %u: shared expert unexpected", l);
            return -1;
        }
        ly->sh_w[p] = data(m, t);
        ly->sh_mx[p] = (struct janas_jns_matrix){.type = t->type,
                                                 .rows = (uint32_t)r,
                                                 .cols = (uint32_t)c,
                                                 .offset = 0,
                                                 .bytes = t->bytes};
    }
#undef NEED
    /* not a rec layer and no slot of the KV cache: the state is this
       file's */
    ly->slot = UINT32_MAX;
    ly->hd = LAT;
    ly->n_kv = 1;
    ly->n_rot = ROT;
    if (main && l == m->n_layer - 1) {
        char e2[160];
        if (!(d->head_fn = need(j, -1, "output_hc_fn.weight", 0,
                                (uint64_t)HC * dm, HC, e2, sizeof e2)) ||
            !(d->head_base = need(j, -1, "output_hc_base.weight", 0, HC, 0, e2,
                                  sizeof e2)) ||
            !(d->head_scale = need(j, -1, "output_hc_scale.weight", 0, 1, 0, e2,
                                   sizeof e2))) {
            snprintf(err, err_len, "%s", e2);
            return -1;
        }
    }
    return 0;
}

int janas_m_ds4_alloc(struct janas_llm_model *m)
{
    struct janas_ds4 *d = m->ds4;
    size_t B = m->blk, dm = m->d_model, nh = m->n_head;
    size_t qd = nh * LAT, od = (size_t)d->o_groups * d->o_rank;
    d->S = malloc(B * HC * dm * sizeof(float));
    d->mix = malloc(B * MIX * sizeof(float));
    d->qa = malloc(B * d->q_rank * sizeof(float));
    d->qaq = malloc(B * d->q_rank / JANAS_QK * sizeof(*d->qaq));
    d->q = malloc(B * qd * sizeof(float));
    d->kv = malloc(B * LAT * sizeof(float));
    d->o = malloc(B * qd * sizeof(float));
    d->oq = malloc(B * qd / JANAS_QK * sizeof(*d->oq));
    d->r = malloc(B * od * sizeof(float));
    d->rq = malloc(B * od / JANAS_QK * sizeof(*d->rq));
    d->ca = malloc(B * 2 * 2 * LAT * sizeof(float));
    d->ixa = malloc(B * 2 * 2 * IXD * sizeof(float));
    d->qi = malloc(B * d->ix_heads * IXD * sizeof(float));
    d->ixw = malloc(B * d->ix_heads * sizeof(float));
    if (!d->S || !d->mix || !d->qa || !d->qaq || !d->q || !d->kv || !d->o ||
        !d->oq || !d->r || !d->rq || !d->ca || !d->ixa || !d->qi || !d->ixw)
        return -1;
    return 0;
}

static void seq_free(struct janas_llm_model *m, struct janas_ds4_seq *s)
{
    if (!s)
        return;
    if (s->raw)
        free(s->raw[m->n_layer]);
    for (uint32_t l = 0; l < m->n_layer; l++) {
        if (s->raw)
            free(s->raw[l]);
        if (s->ca)
            free(s->ca[l]);
        if (s->ent)
            free(s->ent[l]);
        if (s->ixa)
            free(s->ixa[l]);
        if (s->ixk)
            free(s->ixk[l]);
    }
    free(s->raw);
    free(s->ca);
    free(s->ent);
    free(s->ixa);
    free(s->ixk);
    free(s);
}

void janas_m_ds4_seq_free(struct janas_llm_model *m, struct janas_seq *sq)
{
    if (m->ds4 && sq) {
        seq_free(m, sq->ds4);
        sq->ds4 = NULL;
    }
}

void janas_m_ds4_free(struct janas_llm_model *m)
{
    struct janas_ds4 *d = m->ds4;
    if (!d)
        return;
    janas_m_ds4_seq_free(m, &m->seq0);
    free(d->L);
    free(d->S);
    free(d->mix);
    free(d->qa);
    free(d->qaq);
    free(d->q);
    free(d->kv);
    free(d->o);
    free(d->oq);
    free(d->r);
    free(d->rq);
    free(d->ca);
    free(d->ixa);
    free(d->qi);
    free(d->ixw);
    free(d->work);
    free(d);
    m->ds4 = NULL;
}

/* The state of the current sequence, made the first time it is used. */
static struct janas_ds4_seq *seq_of(struct janas_llm_model *m)
{
    struct janas_seq *sq = m->cur;
    if (sq->ds4)
        return sq->ds4;
    struct janas_ds4 *d = m->ds4;
    struct janas_ds4_seq *s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    /* the window behind a block's first query, the block, and room for the
       drafts a speculative pass takes back */
    s->cap = WIN + m->blk + 64;
    s->n_ent = m->n_ctx / 4 + 2;
    uint32_t L = m->n_layer;
    /* and the MTP block's window, at index L */
    s->raw = calloc(L + 1, sizeof(float *));
    s->ca = calloc(L, sizeof(float *));
    s->ent = calloc(L, sizeof(float *));
    s->ixa = calloc(L, sizeof(float *));
    s->ixk = calloc(L, sizeof(float *));
    if (!s->raw || !s->ca || !s->ent || !s->ixa || !s->ixk)
        goto fail;
    if (m->mtp && !(s->raw[L] = malloc((size_t)s->cap * LAT * sizeof(float))))
        goto fail;
    for (uint32_t l = 0; l < L; l++) {
        uint32_t R = d->L[l].ratio;
        if (!(s->raw[l] = malloc((size_t)s->cap * LAT * sizeof(float))))
            goto fail;
        if (!R)
            continue;
        size_t cw = LAT * (R == 4 ? 2 : 1), ne = m->n_ctx / R + 2;
        if (!(s->ca[l] = malloc((size_t)s->cap * 2 * cw * sizeof(float))) ||
            !(s->ent[l] = malloc(ne * LAT * sizeof(float))))
            goto fail;
        if (R == 4 &&
            (!(s->ixa[l] = malloc((size_t)s->cap * 4 * IXD * sizeof(float))) ||
             !(s->ixk[l] = malloc(ne * IXD * sizeof(float)))))
            goto fail;
    }
    sq->ds4 = s;
    return s;
fail:
    seq_free(m, s);
    return NULL;
}

int janas_m_ds4_rewind_ok(const struct janas_llm_model *m, uint32_t n)
{
    const struct janas_ds4_seq *s = m->cur->ds4;
    return !s || m->cur->ring_hi <= n + s->cap - WIN;
}

/* ---- the hyper-connections ---- */

/* The mix of token j's streams for sublayer s (0 attention, 1 mixture):
   pre[4], post[4], the matrix C[4][4] (row: from stream, column: to) into
   mix; the sublayer's input, sum of pre_i X_i, into out. */
static void hc_pre(struct janas_llm_model *m, const struct ds4_layer *D, int s,
                   uint32_t j, float *out)
{
    struct janas_ds4 *d = m->ds4;
    uint32_t dm = m->d_model, n = HC * dm;
    const float *X = d->S + (size_t)j * n;
    double ss = 0;
    for (uint32_t i = 0; i < n; i++)
        ss += (double)X[i] * X[i];
    float r = 1.0f / sqrtf((float)(ss / n) + m->eps);
    const float *fn = f32(m, D->hc_fn[s]), *base = f32(m, D->hc_base[s]),
                *scale = f32(m, D->hc_scale[s]);
    float mx[MIX], *mo = d->mix + (size_t)j * MIX;
    for (int k = 0; k < MIX; k++)
        mx[k] = janas_dot_f32(fn + (size_t)k * n, X, n) * r;
    float eps = d->hc_eps;
    for (int i = 0; i < HC; i++) {
        mo[i] = sigmoid(mx[i] * scale[0] + base[i]) + eps;
        mo[HC + i] = 2.0f * sigmoid(mx[HC + i] * scale[1] + base[HC + i]);
    }
    float *C = mo + 2 * HC;
    for (int i = 0; i < HC * HC; i++)
        C[i] = mx[2 * HC + i] * scale[2] + base[2 * HC + i];
    /* Sinkhorn: a softmax by row (+ eps), then by column; then rows and
       columns normalized in turn */
    for (int a = 0; a < HC; a++) {
        float mxr = -INFINITY, sum = 0;
        for (int b = 0; b < HC; b++)
            mxr = fmaxf(mxr, C[a * HC + b]);
        for (int b = 0; b < HC; b++)
            sum += C[a * HC + b] = expf(C[a * HC + b] - mxr);
        for (int b = 0; b < HC; b++)
            C[a * HC + b] = C[a * HC + b] / sum + eps;
    }
    for (uint32_t it = 0; it < d->sk_iter; it++) {
        if (it) {
            for (int a = 0; a < HC; a++) {
                float sum = 0;
                for (int b = 0; b < HC; b++)
                    sum += C[a * HC + b];
                for (int b = 0; b < HC; b++)
                    C[a * HC + b] /= sum + eps;
            }
        }
        for (int b = 0; b < HC; b++) {
            float sum = 0;
            for (int a = 0; a < HC; a++)
                sum += C[a * HC + b];
            for (int a = 0; a < HC; a++)
                C[a * HC + b] /= sum + eps;
        }
    }
    for (uint32_t c = 0; c < dm; c++) {
        float u = 0;
        for (int i = 0; i < HC; i++)
            u += mo[i] * X[(size_t)i * dm + c];
        out[c] = u;
    }
}

/* The streams after the sublayer: X'_k = post_k f + sum_i C[i][k] X_i. */
static void hc_post(struct janas_llm_model *m, uint32_t j, const float *f)
{
    struct janas_ds4 *d = m->ds4;
    uint32_t dm = m->d_model;
    float *X = d->S + (size_t)j * HC * dm;
    const float *mo = d->mix + (size_t)j * MIX, *C = mo + 2 * HC;
    for (uint32_t c = 0; c < dm; c++) {
        float x[HC];
        for (int i = 0; i < HC; i++)
            x[i] = X[(size_t)i * dm + c];
        for (int k = 0; k < HC; k++) {
            float v = mo[HC + k] * f[c];
            for (int i = 0; i < HC; i++)
                v += C[i * HC + k] * x[i];
            X[(size_t)k * dm + c] = v;
        }
    }
}

struct hc_job {
    struct janas_llm_model *m;
    const struct ds4_layer *D;
    int s, post;
    uint32_t n;
    const float *f;
};

static void hc_worker(void *arg, int tid, int n_threads)
{
    struct hc_job *h = arg;
    uint32_t dm = h->m->d_model;
    for (uint32_t j = (uint32_t)tid; j < h->n; j += (uint32_t)n_threads) {
        if (h->post)
            hc_post(h->m, j, h->f + (size_t)j * dm);
        else
            hc_pre(h->m, h->D, h->s, j, h->m->x + (size_t)j * dm);
    }
}

static void hc_run(struct janas_llm_model *m, const struct ds4_layer *D, int s,
                   int post, uint32_t n, const float *f)
{
    struct hc_job h = {.m = m, .D = D, .s = s, .post = post, .n = n, .f = f};
    if (n < 2)
        hc_worker(&h, 0, 1);
    else
        janas_pool_run(m->compute, hc_worker, &h);
}

/* The final mix of token j's streams into out: sum of pre_i X_i. */
static void hc_head(struct janas_llm_model *m, uint32_t j, float *out, int mtp)
{
    struct janas_ds4 *d = m->ds4;
    uint32_t dm = m->d_model, n = HC * dm;
    const float *X = d->S + (size_t)j * n;
    double ss = 0;
    for (uint32_t i = 0; i < n; i++)
        ss += (double)X[i] * X[i];
    float r = 1.0f / sqrtf((float)(ss / n) + m->eps);
    const float *fn = f32(m, mtp ? d->mtp_head_fn : d->head_fn),
                *base = f32(m, mtp ? d->mtp_head_base : d->head_base),
                *scale = f32(m, mtp ? d->mtp_head_scale : d->head_scale);
    float pre[HC];
    for (int k = 0; k < HC; k++)
        pre[k] =
            sigmoid(janas_dot_f32(fn + (size_t)k * n, X, n) * r * scale[0] +
                    base[k]) +
            d->hc_eps;
    for (uint32_t c = 0; c < dm; c++) {
        float u = 0;
        for (int i = 0; i < HC; i++)
            u += pre[i] * X[(size_t)i * dm + c];
        out[c] = u;
    }
}

/* ---- the attention ---- */

/* The last 64 numbers of v turned by position p (sign -1: back), pairs of
   adjacent numbers. */
static void rope(float *v, const float *f, uint32_t p, float sign)
{
    float *r = v;
    for (int i = 0; i < ROT / 2; i++) {
        float a = (float)p * f[i], c = cosf(a), s = sign * sinf(a);
        float x0 = r[2 * i], x1 = r[2 * i + 1];
        r[2 * i] = x0 * c - x1 * s;
        r[2 * i + 1] = x0 * s + x1 * c;
    }
}

/* Normalized Walsh-Hadamard of 128 numbers (Sylvester's order). */
static void hadamard128(float *v)
{
    for (int h = 1; h < IXD; h *= 2)
        for (int i = 0; i < IXD; i += 2 * h)
            for (int k = i; k < i + h; k++) {
                float a = v[k], b = v[k + h];
                v[k] = a + b;
                v[k + h] = a - b;
            }
    float s = 1.0f / sqrtf((float)IXD);
    for (int i = 0; i < IXD; i++)
        v[i] *= s;
}

/* FP4 (E2M1) rounding by blocks of 32 with a power-of-two scale, as the
   reference simulates it on the indexer's queries and keys. */
static void fp4_sim(float *v, uint32_t n)
{
    static const float q[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    for (uint32_t b = 0; b < n; b += 32) {
        float a = 6.0f * ldexpf(1.0f, -126);
        for (int i = 0; i < 32; i++)
            a = fmaxf(a, fabsf(v[b + i]));
        float s = ldexpf(1.0f, (int)ceilf(log2f(a / 6.0f)));
        for (int i = 0; i < 32; i++) {
            float x = fminf(fmaxf(v[b + i] / s, -6.0f), 6.0f), ax = fabsf(x);
            int best = 0;
            for (int k = 1; k < 8; k++) /* nearest, ties to the even code */
                if (fabsf(q[k] - ax) < fabsf(q[best] - ax) ||
                    (fabsf(q[k] - ax) == fabsf(q[best] - ax) && !(k & 1)))
                    best = k;
            v[b + i] = copysignf(q[best], x) * s;
        }
    }
}

/* Entry i of a compressor whose products of positions are in ring rows of
   2 cw (a, then s with the slot's ape added), into out (width w = cw for
   ratio 128, cw / 2 for 4, overlapping). */
static void compress(const float *ring, uint32_t cap, uint32_t cw, uint32_t R,
                     uint32_t i, float *out)
{
    uint32_t w = R == 4 ? cw / 2 : cw;
    for (uint32_t c = 0; c < w; c++) {
        float mx = -INFINITY;
        /* the candidates: the group's R positions (the second half of the
           channels where ratio 4), and with ratio 4 the previous group's
           (the first half) */
        for (int half = R == 4 ? 0 : 1; half < 2; half++) {
            if (half == 0 && i == 0)
                continue;
            uint32_t g = half == 0 ? i - 1 : i, off = R == 4 && half ? w : 0;
            for (uint32_t t = g * R; t < g * R + R; t++) {
                const float *row = ring + (size_t)(t % cap) * 2 * cw;
                mx = fmaxf(mx, row[cw + off + c]);
            }
        }
        float sum = 0, acc = 0;
        for (int half = R == 4 ? 0 : 1; half < 2; half++) {
            if (half == 0 && i == 0)
                continue;
            uint32_t g = half == 0 ? i - 1 : i, off = R == 4 && half ? w : 0;
            for (uint32_t t = g * R; t < g * R + R; t++) {
                const float *row = ring + (size_t)(t % cap) * 2 * cw;
                float e = expf(row[cw + off + c] - mx);
                sum += e;
                acc += e * row[off + c];
            }
        }
        out[c] = acc / sum;
    }
}

/* The products of the compressors of n positions from pos0 (their input
   xn) into the rings, and the entries that become complete. */
static void compressors(struct janas_llm_model *m, const struct ds4_layer *D,
                        struct janas_ds4_seq *s, uint32_t l, uint32_t n,
                        uint32_t pos0)
{
    struct janas_ds4 *d = m->ds4;
    uint32_t R = D->ratio, dm = m->d_model;
    for (int ix = 0; ix < (R == 4 ? 2 : 1); ix++) {
        uint32_t cw = ix ? 2 * IXD : LAT * (R == 4 ? 2 : 1);
        uint32_t w = R == 4 ? cw / 2 : cw;
        float *buf = ix ? d->ixa : d->ca, *ring = ix ? s->ixa[l] : s->ca[l];
        const struct janas_jns_tensor *kv = ix ? D->ix_kv : D->c_kv,
                                      *gt = ix ? D->ix_gate : D->c_gate,
                                      *ape = ix ? D->ix_ape : D->c_ape,
                                      *nw = ix ? D->ix_norm : D->c_norm;
        janas_m_bf16_product(m, data(m, kv), m->xn, buf, cw, dm, n, 1.0f);
        janas_m_bf16_product(m, data(m, gt), m->xn, buf + (size_t)n * cw, cw,
                             dm, n, 1.0f);
        for (uint32_t j = 0; j < n; j++) {
            uint32_t p = pos0 + j;
            float *row = ring + (size_t)(p % s->cap) * 2 * cw;
            memcpy(row, buf + (size_t)j * cw, cw * sizeof(float));
            const float *a = f32(m, ape) + (size_t)(p % R) * cw;
            for (uint32_t c = 0; c < cw; c++)
                row[cw + c] = buf[(size_t)(n + j) * cw + c] + a[c];
            if ((p + 1) % R)
                continue;
            uint32_t i = (p + 1) / R - 1;
            float *out =
                ix ? s->ixk[l] + (size_t)i * IXD : s->ent[l] + (size_t)i * LAT;
            compress(ring, s->cap, cw, R, i, out);
            rms_norm(out, out, f32(m, nw), w, m->eps);
            rope(out + w - ROT, d->f_yarn, i * R, 1.0f);
            if (ix) {
                hadamard128(out);
                fp4_sim(out, IXD);
            }
        }
    }
}

struct att_job {
    struct janas_llm_model *m;
    const struct ds4_layer *D;
    struct janas_ds4_seq *s;
    uint32_t l, n, pos0;
    atomic_uint next;
};

/* The compressed latents a query at p sees in a ratio-4 layer: all when
   no more than the indexer keeps, else its choice; written to sel, their
   count returned. */
static uint32_t choose(struct janas_llm_model *m, const struct att_job *a,
                       uint32_t j, uint32_t ne, uint32_t *sel, float *score)
{
    struct janas_ds4 *d = m->ds4;
    uint32_t k = d->ix_topk;
    if (ne <= k) {
        for (uint32_t i = 0; i < ne; i++)
            sel[i] = i;
        return ne;
    }
    const float *qi = d->qi + (size_t)j * d->ix_heads * IXD,
                *w = d->ixw + (size_t)j * d->ix_heads;
    const float *K = a->s->ixk[a->l];
    for (uint32_t t = 0; t < ne; t++) {
        float sc = 0;
        for (uint32_t h = 0; h < d->ix_heads; h++) {
            float v =
                janas_dot_f32(qi + (size_t)h * IXD, K + (size_t)t * IXD, IXD);
            sc += w[h] * (v > 0 ? v : 0);
        }
        score[t] = sc;
    }
    /* the k largest: a threshold by selection, then ties in order */
    for (uint32_t i = 0; i < ne; i++)
        sel[i] = i;
    uint32_t lo = 0, hi = ne;
    while (hi - lo > 1) { /* quickselect of the k-th largest */
        uint32_t pv = sel[lo + (hi - lo) / 2], st = lo;
        float pvs = score[pv];
        uint32_t tmp = sel[lo + (hi - lo) / 2];
        sel[lo + (hi - lo) / 2] = sel[hi - 1];
        sel[hi - 1] = tmp;
        for (uint32_t i = lo; i < hi - 1; i++)
            if (score[sel[i]] > pvs) {
                tmp = sel[i];
                sel[i] = sel[st];
                sel[st] = tmp;
                st++;
            }
        tmp = sel[st];
        sel[st] = sel[hi - 1];
        sel[hi - 1] = tmp;
        if (st == k - 1 || st == k)
            break;
        if (st < k)
            lo = st + 1;
        else
            hi = st;
    }
    return k;
}

/* One (token, head): the window and the compressed latents it sees. */
static void att_item(struct att_job *a, uint32_t j, uint32_t h, float *work,
                     uint32_t *sel)
{
    struct janas_llm_model *m = a->m;
    struct janas_ds4 *d = m->ds4;
    const struct ds4_layer *D = a->D;
    struct janas_ds4_seq *s = a->s;
    uint32_t p = a->pos0 + j, nh = m->n_head;
    const float *q = d->q + ((size_t)j * nh + h) * LAT;
    float *o = d->o + ((size_t)j * nh + h) * LAT;
    const float scale = 1.0f / sqrtf((float)LAT);
    uint32_t w0 = p >= WIN - 1 ? p - (WIN - 1) : 0, nw = p - w0 + 1;
    uint32_t ne = D->ratio ? (p + 1) / D->ratio : 0, nc = ne;
    const float *ent = D->ratio ? s->ent[a->l] : NULL;
    if (D->ratio == 4 && ne > d->ix_topk) {
        /* the indexer's choice is the token's, the same for every head:
           made by head 0's item into sel, read by the others */
        nc = d->ix_topk;
    } else if (D->ratio == 4) {
        for (uint32_t i = 0; i < ne; i++)
            sel[i] = i;
    }
    float *sc = work, mx = f32(m, D->sink)[h];
    for (uint32_t t = 0; t < nw; t++) {
        const float *c = s->raw[a->l] + (size_t)((w0 + t) % s->cap) * LAT;
        sc[t] = janas_dot_f32(q, c, LAT) * scale;
        mx = fmaxf(mx, sc[t]);
    }
    for (uint32_t t = 0; t < nc; t++) {
        uint32_t i = D->ratio == 4 ? sel[t] : t;
        sc[nw + t] = janas_dot_f32(q, ent + (size_t)i * LAT, LAT) * scale;
        mx = fmaxf(mx, sc[nw + t]);
    }
    float sum = expf(f32(m, D->sink)[h] - mx);
    for (uint32_t t = 0; t < nw + nc; t++)
        sum += sc[t] = expf(sc[t] - mx);
    memset(o, 0, LAT * sizeof(float));
    for (uint32_t t = 0; t < nw; t++)
        janas_axpy_f32(o, sc[t] / sum,
                       s->raw[a->l] + (size_t)((w0 + t) % s->cap) * LAT, LAT);
    for (uint32_t t = 0; t < nc; t++) {
        uint32_t i = D->ratio == 4 ? sel[t] : t;
        janas_axpy_f32(o, sc[nw + t] / sum, ent + (size_t)i * LAT, LAT);
    }
    rope(o + LAT - ROT, D->ratio ? d->f_yarn : d->f_plain, p, -1.0f);
}

static void att_worker(void *arg, int tid, int n_threads)
{
    struct att_job *a = arg;
    struct janas_ds4 *d = a->m->ds4;
    (void)n_threads;
    float *work = d->work + (size_t)tid * d->work_per * 2;
    uint32_t *sel = (uint32_t *)(work + d->work_per);
    uint32_t nh = a->m->n_head;
    /* by token: the indexer's choice once, then its heads */
    for (;;) {
        uint32_t j = atomic_fetch_add(&a->next, 1);
        if (j >= a->n)
            break;
        uint32_t p = a->pos0 + j;
        uint32_t ne = a->D->ratio ? (p + 1) / a->D->ratio : 0;
        if (a->D->ratio == 4 && ne > d->ix_topk)
            choose(a->m, a, j, ne, sel, work);
        for (uint32_t h = 0; h < nh; h++)
            att_item(a, j, h, work, sel);
    }
}

/* The attention of a block, its input normalized in xn and xq; its output
   (d_model a token) into out. */
static int attention(struct janas_llm_model *m, const struct ds4_layer *D,
                     struct janas_ds4_seq *s, uint32_t l, uint32_t n,
                     uint32_t pos0, float *out)
{
    struct janas_ds4 *d = m->ds4;
    uint32_t dm = m->d_model, nh = m->n_head, qr = d->q_rank;
    uint32_t qd = nh * LAT, og = d->o_groups, orank = d->o_rank;
    uint32_t gw = nh / og * LAT; /* a group's heads */
    const float *fr = D->ratio ? d->f_yarn : d->f_plain;
    struct janas_matvec_task t1[2] = {{.type = (int)D->wq_a->type,
                                       .w = data(m, D->wq_a),
                                       .x = m->xq,
                                       .y = d->qa,
                                       .rows = qr,
                                       .cols = dm,
                                       .n_vec = n},
                                      {.type = (int)D->wkv->type,
                                       .w = data(m, D->wkv),
                                       .x = m->xq,
                                       .y = d->kv,
                                       .rows = LAT,
                                       .cols = dm,
                                       .n_vec = n}};
    janas_m_products(m, janas_m_gpu(m, 0), t1, 2);
    for (uint32_t j = 0; j < n; j++) {
        float *qa = d->qa + (size_t)j * qr;
        rms_norm(qa, qa, f32(m, D->q_norm), qr, m->eps);
        janas_q8k_quantize(qa, d->qaq + (size_t)j * (qr / JANAS_QK), qr);
        float *kv = d->kv + (size_t)j * LAT;
        rms_norm(kv, kv, f32(m, D->kv_norm), LAT, m->eps);
        rope(kv + LAT - ROT, fr, pos0 + j, 1.0f);
        memcpy(s->raw[l] + (size_t)((pos0 + j) % s->cap) * LAT, kv,
               LAT * sizeof(float));
    }
    struct janas_matvec_task t2[2] = {{.type = (int)D->wq_b->type,
                                       .w = data(m, D->wq_b),
                                       .x = d->qaq,
                                       .y = d->q,
                                       .rows = qd,
                                       .cols = qr,
                                       .n_vec = n}};
    size_t nt2 = 1;
    if (D->ratio == 4)
        t2[nt2++] = (struct janas_matvec_task){.type = (int)D->ix_qb->type,
                                               .w = data(m, D->ix_qb),
                                               .x = d->qaq,
                                               .y = d->qi,
                                               .rows = d->ix_heads * IXD,
                                               .cols = qr,
                                               .n_vec = n};
    janas_m_products(m, janas_m_gpu(m, 0), t2, nt2);
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t h = 0; h < nh; h++) {
            float *q = d->q + ((size_t)j * nh + h) * LAT;
            double ss = 0;
            for (int i = 0; i < LAT; i++)
                ss += (double)q[i] * q[i];
            float r = 1.0f / sqrtf((float)(ss / LAT) + m->eps);
            for (int i = 0; i < LAT; i++)
                q[i] *= r;
            rope(q + LAT - ROT, fr, pos0 + j, 1.0f);
        }
    if (D->ratio)
        compressors(m, D, s, l, n, pos0);
    if (D->ratio == 4) {
        /* the indexer's queries and its heads' weights */
        for (uint32_t j = 0; j < n; j++)
            for (uint32_t h = 0; h < d->ix_heads; h++) {
                float *q = d->qi + ((size_t)j * d->ix_heads + h) * IXD;
                rope(q + IXD - ROT, d->f_yarn, pos0 + j, 1.0f);
                hadamard128(q);
                fp4_sim(q, IXD);
            }
        janas_m_bf16_product(m, data(m, D->ix_proj), m->xn, d->ixw, d->ix_heads,
                             dm, n,
                             1.0f / sqrtf((float)IXD * (float)d->ix_heads));
    }
    struct att_job a = {.m = m, .D = D, .s = s, .l = l, .n = n, .pos0 = pos0};
    atomic_init(&a.next, 0);
    if (n < 2)
        att_worker(&a, 0, 1);
    else
        janas_pool_run(m->compute, att_worker, &a);
    /* the output: by groups of heads (wo_a, block-diagonal), then wo_b */
    for (uint32_t j = 0; j < n; j++)
        janas_q8k_quantize(d->o + (size_t)j * qd,
                           d->oq + (size_t)j * (qd / JANAS_QK), qd);
    struct janas_matvec_task tg[64];
    size_t row_bytes = janas_qtype_row_bytes((int)D->wo_a->type, gw);
    for (uint32_t g = 0; g < og && g < 64; g++)
        tg[g] =
            (struct janas_matvec_task){.type = (int)D->wo_a->type,
                                       .w = (const uint8_t *)data(m, D->wo_a) +
                                            (size_t)g * orank * row_bytes,
                                       .x = d->oq + (size_t)g * (gw / JANAS_QK),
                                       .y = d->r + (size_t)g * orank,
                                       .rows = orank,
                                       .cols = gw,
                                       .n_vec = n,
                                       .x_stride = qd / JANAS_QK,
                                       .y_stride = (size_t)og * orank};
    janas_m_products(m, janas_m_gpu(m, 0), tg, og);
    uint32_t od = og * orank;
    for (uint32_t j = 0; j < n; j++)
        janas_q8k_quantize(d->r + (size_t)j * od,
                           d->rq + (size_t)j * (od / JANAS_QK), od);
    struct janas_matvec_task tb = {.type = (int)D->wo_b->type,
                                   .w = data(m, D->wo_b),
                                   .x = d->rq,
                                   .y = out,
                                   .rows = dm,
                                   .cols = od,
                                   .n_vec = n};
    janas_m_products(m, janas_m_gpu(m, 0), &tb, 1);
    return 0;
}

/* ---- the mixture's choice (model_ffn.c) ---- */

static float softplus(float x)
{
    return x > 20.0f ? x : log1pf(expf(x));
}

void janas_m_ds4_route(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t l, uint32_t n)
{
    (void)l;
    const struct ds4_layer *D = ly->ds4l;
    uint32_t ne = m->n_expert, k = m->k_use, dm = m->d_model;
    for (uint32_t j = 0; j < n; j++) {
        const float *x = m->xn + (size_t)j * dm;
        float sc[4096], sel[4096];
        for (uint32_t e = 0; e < ne; e++) {
            sc[e] = sqrtf(
                softplus(janas_dot_f32(D->router + (size_t)e * dm, x, dm)));
            sel[e] = D->bias ? sc[e] + D->bias[e] : sc[e];
        }
        float wsum = 0;
        for (uint32_t i = 0; i < k; i++) {
            uint32_t best = 0;
            if (D->tid2eid) {
                int32_t tok = m->gtok_tokens ? m->gtok_tokens[j] : 0;
                best = (uint32_t)D->tid2eid[(size_t)tok * m->k + i];
            } else {
                float bv = -INFINITY;
                for (uint32_t e = 0; e < ne; e++)
                    if (sel[e] > bv) {
                        bv = sel[e];
                        best = e;
                    }
                sel[best] = -INFINITY; /* taken */
            }
            uint32_t p = j * k + i;
            m->pair_tok[p] = j;
            m->pair_exp[p] = best;
            m->pair_w[p] = sc[best];
            wsum += sc[best];
        }
        for (uint32_t i = 0; i < k; i++)
            m->pair_w[j * k + i] = m->pair_w[j * k + i] / wsum * m->route_scale;
    }
}

/* ---- reading ahead ---- */

static int by_votes(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x < y) - (x > y); /* most votes first */
}

/*
 * The experts of layer l read ahead, while the work before them goes on.
 * A layer that routes by a table of token ids (the first three) knows its
 * experts from the tokens alone: all of them, exactly, at once. The others
 * may be guessed from the input of the layer before, normalized in xn,
 * with layer l's own router and bias, where JANAS_PREFETCH asks for it
 * (below: they cost more than they gave) - for a reply's single token
 * where the engine found it pays (pf_on), for a block's tokens where it
 * missed many (pfb_on); the experts most of the block's tokens would
 * choose first.
 */
static void read_ahead(struct janas_llm_model *m, uint32_t l, uint32_t n)
{
    struct janas_ds4 *d = m->ds4;
    const struct ds4_layer *D = &d->L[l];
    if (!m->cache || !janas_expert_cache_can_prefetch(m->cache) ||
        m->pf_mode == 0)
        return;
    int table = D->tid2eid && m->gtok_tokens;
    /* the guesses only where asked for (JANAS_PREFETCH=<rank>,<max>): on
       the whole model they read experts that were not then asked for, and
       a token took 572 and 559 ms against 476 and 462 without them, the
       experts found in memory 64.9% against 54.0%; the table's alone, 459
       and 461 ms, 59.7% (64 tokens, a 9000 MiB cache, 19 Nov 2026) */
    if (!table && m->pf_mode != 2)
        return;
    if (!table && !(n == 1 ? m->pf_on : m->pfb_on))
        return;
    uint32_t ne = m->n_expert, dm = m->d_model, k = m->k_use;
    uint32_t want = table ? k : n == 1 ? m->pf_rank : k;
    uint32_t votes[4096] = {0};
    for (uint32_t j = 0; j < n; j++) {
        if (table) {
            for (uint32_t i = 0; i < k; i++)
                votes[D->tid2eid[(size_t)m->gtok_tokens[j] * m->k + i]]++;
            continue;
        }
        const float *x = m->xn + (size_t)j * dm;
        float sel[4096];
        for (uint32_t e = 0; e < ne; e++)
            sel[e] = sqrtf(softplus(
                         janas_dot_f32(D->router + (size_t)e * dm, x, dm))) +
                     (D->bias ? D->bias[e] : 0.0f);
        for (uint32_t i = 0; i < want && i < ne; i++) {
            uint32_t best = 0;
            for (uint32_t e = 1; e < ne; e++)
                if (sel[e] > sel[best])
                    best = e;
            votes[best]++;
            sel[best] = -INFINITY;
        }
    }
    uint64_t key[4096];
    uint32_t ids[4096], nk = 0;
    for (uint32_t e = 0; e < ne; e++)
        if (votes[e])
            key[nk++] = (uint64_t)votes[e] << 32 | e;
    qsort(key, nk, sizeof(*key), by_votes);
    for (uint32_t i = 0; i < nk; i++)
        ids[i] = (uint32_t)key[i];
    /* a table's experts are certain: all; a guess: as many as the engine
       allows a token, or every one a block's tokens chose */
    size_t max = table || n > 1 ? nk : m->pf_max;
    janas_expert_cache_prefetch(m->cache, l, ids, nk, max);
}

/* ---- the layers ---- */

/* One layer on the streams in S: its attention (its latents in the
   sequence's rings at index ring), its mixture (experts from cache, the
   file's layer jl, the cache's layer cl); next: the main model's next layer
   to read ahead for, or UINT32_MAX. */
static int run_layer(struct janas_llm_model *m, const struct ds4_layer *D,
                     const struct layer *ly, struct janas_ds4_seq *s,
                     uint32_t ring, struct janas_expert_cache *cache,
                     const struct janas_jns_layer *jl, uint32_t cl, uint32_t n,
                     uint32_t pos0, uint32_t next)
{
    uint32_t dm = m->d_model;
    hc_run(m, D, 0, 0, n, NULL);
    janas_m_norm_quant(m, f32(m, ly->attn_norm), 0, n);
    trace_row(m, "attn_norm", m->xn + (size_t)(n - 1) * dm, dm);
    if (attention(m, D, s, ring, n, pos0, m->x) != 0)
        return -1;
    trace_row(m, "attn_out", m->x + (size_t)(n - 1) * dm, dm);
    hc_run(m, D, 0, 1, n, m->x);
    hc_run(m, D, 1, 0, n, NULL);
    janas_m_norm_quant(m, f32(m, ly->ffn_norm), 0, n);
    memset(m->x, 0, (size_t)n * dm * sizeof(float));
    /* the next layer's experts, read while this one's are computed, as the
       other models do (the cache reads ahead one layer at a time; this
       layer's own, asked ahead before, are settled by it) */
    if (next != UINT32_MAX)
        read_ahead(m, next, n);
    if (janas_m_moe_block(m, ly, cache, jl, cl, n, MOE_ALL) != 0)
        return -1;
    trace_row(m, "ffn_out", m->x + (size_t)(n - 1) * dm, dm);
    hc_run(m, D, 1, 1, n, m->x);
    return 0;
}

int janas_m_ds4_layers(struct janas_llm_model *m, uint32_t n, uint32_t pos0)
{
    struct janas_ds4 *d = m->ds4;
    struct janas_ds4_seq *s = seq_of(m);
    uint32_t dm = m->d_model;
    if (!s || pos0 + n > m->n_ctx)
        return -1;
    if (!d->work) {
        /* a thread's scores: the window and every compressed latent the
           context can hold (one per 4), or the indexer's; the pool is made
           after the buffers */
        d->n_threads = (uint32_t)janas_pool_size(m->compute);
        d->work_per = WIN + m->n_ctx / 4 + 16;
        d->work = malloc(d->n_threads * d->work_per * 2 * sizeof(float));
        if (!d->work)
            return -1;
    }
    /* the four streams: the embedding, four times */
    for (uint32_t j = 0; j < n; j++)
        for (int i = 0; i < HC; i++)
            memcpy(d->S + ((size_t)j * HC + i) * dm, m->x + (size_t)j * dm,
                   dm * sizeof(float));
    uint32_t nl = d->n_only ? d->n_only : m->n_layer;
    /* the first layer's experts, known from the tokens, read meanwhile */
    read_ahead(m, d->n_only ? d->only[0] : 0, n);
    for (uint32_t li = 0; li < nl; li++) {
        uint32_t l = d->n_only ? d->only[li] : li;
        uint32_t next =
            li + 1 < nl ? (d->n_only ? d->only[li + 1] : li + 1) : UINT32_MAX;
        if (run_layer(m, &d->L[l], &m->layers[l], s, l, m->cache,
                      &m->j.layers[l], l, n, pos0, next) != 0)
            return -1;
        if (m->trace) {
            const float *v = d->S + (size_t)(n - 1) * HC * dm;
            double nn = 0;
            for (uint32_t i = 0; i < HC * dm; i++)
                nn += (double)v[i] * v[i];
            fprintf(stderr, "l_out-%-3u norm %10.4f  %9.4f %9.4f %9.4f %9.4f\n",
                    l, sqrt(nn), v[0], v[1], v[2], v[3]);
        }
    }
    /* the streams after the last layer, which the MTP block reads */
    if (m->mtp && m->hid)
        memcpy(m->hid, d->S, (size_t)n * HC * dm * sizeof(float));
    /* the streams mixed into x, for the output norm and the head */
    for (uint32_t j = 0; j < n; j++)
        hc_head(m, j, m->x + (size_t)j * dm, 0);
    return 0;
}

int janas_m_ds4_mtp_load(struct janas_llm_model *m, char *err, size_t err_len)
{
    struct janas_ds4 *d = m->ds4;
    uint32_t dm = m->d_model;
    if (!(d->mtp_head_fn = need(&m->mtp_j, -1, "mtp.hc_head_fn.weight", 0,
                                (uint64_t)HC * dm, HC, err, err_len)) ||
        !(d->mtp_head_base = need(&m->mtp_j, -1, "mtp.hc_head_base.weight", 0,
                                  HC, 0, err, err_len)) ||
        !(d->mtp_head_scale = need(&m->mtp_j, -1, "mtp.hc_head_scale.weight", 0,
                                   1, 0, err, err_len)))
        return -1;
    return 0;
}

/*
 * The MTP block (DeepSeek's MTPBlock): the next tokens' embeddings, in x,
 * normalized (enorm) and projected (e_proj) into each of the main model's
 * streams after its last layer (hidden, HC x d_model a row), themselves
 * normalized (hnorm) and projected (h_proj) - one product, [e; h] by
 * eh_proj -; then a layer as the model's, its attention over its own
 * window; its streams kept for the next draft (mtp_hid); its own final mix
 * and norm into xn and xq, rows first .. n - 1, for the shared head.
 */
int janas_m_ds4_mtp(struct janas_llm_model *m, const float *hidden, uint32_t n,
                    uint32_t pos0, uint32_t first)
{
    struct janas_ds4 *d = m->ds4;
    struct janas_ds4_seq *s = seq_of(m);
    size_t dm = m->d_model, nbc = 2 * dm / JANAS_QK;
    if (!s || !s->raw[m->n_layer] || pos0 + n > m->n_ctx)
        return -1;
    const float *en = f32(m, m->enorm), *hn = f32(m, m->hnorm);
    float *c = m->cat; /* a row of [embedding; stream] */
    for (uint32_t j = 0; j < n; j++) {
        rms_norm(c, m->x + (size_t)j * dm, en, (uint32_t)dm, m->eps);
        for (uint32_t k = 0; k < HC; k++) {
            rms_norm(c + dm, hidden + ((size_t)j * HC + k) * dm, hn,
                     (uint32_t)dm, m->eps);
            janas_q8k_quantize(c, m->catq + ((size_t)j * HC + k) * nbc,
                               (uint32_t)(2 * dm));
        }
    }
    struct janas_matvec_task tp = {.type = (int)m->eh_proj->type,
                                   .w = data(m, m->eh_proj),
                                   .x = m->catq,
                                   .y = d->S,
                                   .rows = dm,
                                   .cols = 2 * dm,
                                   .n_vec = (size_t)n * HC};
    janas_m_products(m, janas_m_gpu(m, 0), &tp, 1);
    if (run_layer(m, &d->mtpL, &m->mtp_layer, s, m->n_layer, m->mtp_cache,
                  m->mtp_jl, m->mtp_cl, n, pos0, UINT32_MAX) != 0)
        return -1;
    memcpy(m->mtp_hid, d->S, (size_t)n * HC * dm * sizeof(float));
    for (uint32_t j = first; j < n; j++)
        hc_head(m, j, m->x + (size_t)j * dm, 1);
    janas_m_norm_quant(m, f32(m, m->mtp_norm), first, n);
    return 0;
}
