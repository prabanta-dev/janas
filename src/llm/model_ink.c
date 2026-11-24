/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_ink.c - Thinking Machines' Inkling-2-Mini (inkling2_loop): its
 * layers, looped, and their state.
 *
 * 24 layers, run 48 times: 0-5, then 6-17 three times (the passes share
 * the weights, each has its own keys, values and conv inputs), then 18-23.
 * A layer:
 *  - its input is a depth mix: a softmax, by a learned query, over the
 *    block states kept so far and the partial stream, each normalized
 *    (RMS, no weights) - its keys and its values; a new block starts at
 *    layers 0, 6, 12 and 18, where the stream becomes a block state and
 *    the partial stream starts again from the attention's output;
 *  - attention with no RoPE: q scaled by a learned [head][dim] and clipped
 *    to a norm of 16 sqrt(2), k clipped, v normalized by head; a position
 *    bias of the query's own (16 values a head, scaled, clipped to 8, onto
 *    a table of 1024 distances); a sink logit a head; the output gated by a
 *    sigmoid, projected, through a short conv; the layers 5, 11, 17, 23
 *    attend the whole context with q and the bias grown by
 *    1 + 0.1 ln(max((pos + 1) / 8192, 1)), the others a window of 1024;
 *  - short causal convs (4 taps, by channel) on k, v, the attention's and
 *    the experts' outputs: (taps + identity) / sqrt(5); the global layers'
 *    k and v add 4 taps a KV head made from the token: / sqrt(9);
 *  - a mixture: sigmoid router, the 8 best of 128 by score plus a bias,
 *    weighted by the scores normalized, times 8; each expert
 *    4 tanh(g/4) sigmoid(g) 25 tanh(u/25) (model_ffn.c), down.
 * Passes 2 and 3 take four messages a layer from the pass before (a mix
 * of its layer-12 block state and its output), each through a gate of
 * rank 64: (1 - tanh(forget)) x + tanh(gain) m, on the q, k, v and
 * mixture inputs. The output: the final depth mix, then (model.c) the
 * norm, the head of 200058 rows and 30 tanh(x / 30).
 *
 * What is computed is the transformers reference (modeling_inkling2_loop)
 * as llama.cpp computes it, the logits compared against llama.cpp's.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matvec.h"
#include "vmath.h"

#include "model_impl.h"

enum {
    H = 24,         /* query heads */
    HKV = 8,        /* key and value heads */
    HD = 128,       /* head size */
    QD = H * HD,    /* 3072 */
    KVD = HKV * HD, /* 1024 */
    DR = 16,        /* position states a head */
    XR = 1024,      /* distances the bias table holds */
    WIN = 1024,     /* the local layers' window: the token and 1023 before */
    KC = 4,         /* conv taps */
    LOOP_S = 6,     /* the core's first layer */
    LOOP_E = 18,    /* past its last */
    PASSES = 3,
    RANK = 64,                                     /* the messages' gates */
    SLOTS = 24 + (PASSES - 1) * (LOOP_E - LOOP_S), /* 48 */
    MARGIN = 64,                  /* positions a sequence may be taken back */
    CC = MARGIN + KC,             /* conv inputs kept a slot */
    CONV_CH = 2 * KVD + 2 * 2304, /* k, v, attention out, experts out */
};

struct ink_layer {
    int global;
    const struct janas_jns_tensor *wq, *wk, *wv, *wgate, *wo, *wrel, *wsink,
        *kcp, *vcp;
    const float *q_clip;        /* [H][HD] */
    const float *rel_clip;      /* [H][DR] */
    const float *rel_tab;       /* [XR][DR] */
    const float *kconv, *vconv; /* [KVD][KC], tap KC-1 the token */
    const float *pconv, *fconv; /* [d_model][KC] */
    const float *attn_q;        /* [1 or 3][d_model], NULL at 0 */
    const float *ffn_q;         /* [d_model] */
    const float *attn_norm, *qn, *kn, *vn;
    const float *msg_q; /* [4][d_model], the core only */
    const float *c_down[4], *c_gain[4], *c_forget[4];
    const float *gu_scale; /* each expert's of gate and up, or NULL */
    /* the small projections a file keeps in f16, bf16 or f32 (unsloth's
       sinks and dynamic conv taps), as f32 rows: computed on the input as
       it is, not through the quantized products; NULL where the products
       read them */
    float *sink32, *kcp32, *vcp32;
};

struct janas_ink {
    struct ink_layer L[24];
    const float *embd_norm, *out_q;
    float tau_alpha, tau_floor;
    /* the forward's buffers, for a block of up to m->blk tokens */
    float *h, *h6, *Bs, *Bpre, *r, *bank, *msg, *A, *xin;
    float *q, *kl, *vl, *k, *v, *rel, *sink, *gate, *o, *aout, *dk, *dv;
    struct janas_block_q8k *oq;
    float *work; /* attention: a thread's scores and position profile */
    uint32_t n_threads, work_per;
};

struct janas_ink_seq {
    /* slot s: keys and values after their conv, clip and norm; the global
       slots whole (n_ctx positions), the local ones a ring of lcap */
    float *K[SLOTS], *V[SLOTS];
    float *conv[SLOTS]; /* a ring of CC positions x CONV_CH raw inputs */
    uint32_t lcap;
};

static int slot_global(uint32_t s)
{
    /* the global layers' slots: 5, 11, 17, 23, and the core's 11 and 17 in
       passes 2 and 3 */
    return s < 24 ? s % 6 == 5 : (s - 24) % 12 % 6 == 5;
}

static int meta_u32(struct janas_llm_model *m, const char *suffix, uint32_t *v)
{
    char key[128];
    snprintf(key, sizeof(key), "%.32s.%s", m->j.h.arch, suffix);
    return janas_jns_meta_u32(&m->j, key, v);
}

static int meta_f32(struct janas_llm_model *m, const char *suffix, float *v)
{
    char key[128];
    snprintf(key, sizeof(key), "%.32s.%s", m->j.h.arch, suffix);
    return janas_jns_meta_f32(&m->j, key, v);
}

/* A number of the architecture's metadata, or the first of its array
   (one a layer, as heads are written), into v. */
static int meta_first(struct janas_llm_model *m, const char *suffix,
                      uint32_t *v)
{
    char key[128];
    uint64_t x = 0, count = 0;
    if (meta_u32(m, suffix, v) == 0)
        return 0;
    snprintf(key, sizeof(key), "%.32s.%s", m->j.h.arch, suffix);
    if (janas_jns_meta_uint_at(&m->j, key, 0, &x, &count) != 0 || !count)
        return -1;
    *v = (uint32_t)x;
    return 0;
}

int janas_m_ink_params(struct janas_llm_model *m)
{
    uint32_t ls = 0, le = 0, lc = 0, rank = 0, win = 0, dr = 0, xr = 0;
    if (meta_first(m, "attention.head_count", &m->n_head) ||
        meta_first(m, "attention.head_count_kv", &m->n_head_kv) ||
        meta_u32(m, "attention.key_length", &m->head_dim) ||
        meta_f32(m, "attention.layer_norm_rms_epsilon", &m->eps) ||
        meta_f32(m, "expert_weights_scale", &m->route_scale) ||
        meta_u32(m, "loop.start", &ls) || meta_u32(m, "loop.end", &le) ||
        meta_u32(m, "loop.count", &lc) ||
        meta_u32(m, "loop.gate_rank", &rank) ||
        meta_u32(m, "attention.sliding_window", &win) ||
        meta_u32(m, "attention.relative_length", &dr) ||
        meta_u32(m, "attention.relative_extent", &xr))
        return -1;
    /* this file's shapes, which the code below is written for */
    if (m->n_head != H || m->n_head_kv != HKV || m->head_dim != HD ||
        m->n_layer != 24 || ls != LOOP_S || le != LOOP_E || lc != PASSES ||
        rank != RANK || win != WIN || dr != DR || xr != XR ||
        m->d_model != 2304 || m->k > MAX_K)
        return -1;
    /* 30 tanh(x / 30) on the logits (model.c) */
    m->softcap = 0.0f;
    meta_f32(m, "final_logit_softcapping", &m->softcap);
    m->n_rot = HD; /* no RoPE: the tables made, never used */
    m->rope_base = 10000.0f;
    /* the widest expert, as the experts' buffers take it */
    m->d_ff = m->j.layers[0].m[JANAS_JNS_GATE].rows;
    return m->d_ff && m->d_ff % JANAS_QK == 0 ? 0 : -1;
}

/* A tensor of layer l (l < 0: the model's), of type type (0 f32, -1 any
   the products read) and shape d0 x d1 (d1 0: one dimension). */
static const struct janas_jns_tensor *need(struct janas_llm_model *m, int l,
                                           const char *suffix, int type,
                                           uint64_t d0, uint64_t d1, char *err,
                                           size_t err_len)
{
    char name[96];
    if (l < 0)
        snprintf(name, sizeof(name), "%s", suffix);
    else
        snprintf(name, sizeof(name), "blk.%d.%s", l, suffix);
    const struct janas_jns_tensor *t = janas_jns_tensor(&m->j, name);
    int ok = t && t->dims[0] == d0 && (d1 ? t->dims[1] == d1 : 1);
    if (ok)
        ok = type == 0 ? t->type == 0
                       : janas_qtype_block_size((int)t->type) != 0;
    if (!ok)
        snprintf(err, err_len, "tensor %s missing or unexpected", name);
    return ok ? t : NULL;
}

/* A small matrix (rows x d0) of layer l: one the products read, or one in
   f32, f16 or bf16 made into f32 rows in *own. */
static const struct janas_jns_tensor *
small(struct janas_llm_model *m, uint32_t l, const char *suffix, uint64_t d0,
      uint64_t rows, float **own, char *err, size_t err_len)
{
    char name[96];
    snprintf(name, sizeof(name), "blk.%u.%s", l, suffix);
    const struct janas_jns_tensor *t = janas_jns_tensor(&m->j, name);
    *own = NULL;
    if (!t || t->dims[0] != d0 || t->dims[1] != rows) {
        snprintf(err, err_len, "tensor %s missing or unexpected", name);
        return NULL;
    }
    if (t->type != 0 && t->type != 1 && t->type != 30)
        return janas_qtype_block_size((int)t->type) ? t : NULL;
    size_t n = (size_t)d0 * rows;
    if (!(*own = malloc(n * sizeof(float))))
        return NULL;
    if (t->type == 0)
        memcpy(*own, data(m, t), n * sizeof(float));
    else {
        const uint16_t *h = data(m, t);
        for (size_t i = 0; i < n; i++)
            (*own)[i] = t->type == 1 ? janas_fp16_to_fp32(h[i])
                                     : janas_u2f((uint32_t)h[i] << 16);
    }
    return t;
}

int janas_m_ink_layer_load(struct janas_llm_model *m, uint32_t l,
                           struct layer *ly, char *err, size_t err_len)
{
    uint32_t dm = m->d_model;
    if (!m->ink) {
        if (!(m->ink = calloc(1, sizeof(*m->ink))))
            return -1;
        const struct janas_jns_tensor *en, *oq;
        if (!(en = need(m, -1, "token_embd_norm.weight", 0, dm, 0, err,
                        err_len)) ||
            !(oq = need(m, -1, "output_res_query.weight", 0, dm, 0, err,
                        err_len)))
            return -1;
        m->ink->embd_norm = f32(m, en);
        m->ink->out_q = f32(m, oq);
        m->ink->tau_alpha = 0.1f;
        m->ink->tau_floor = 8192.0f;
        meta_f32(m, "attention.log_scale_alpha", &m->ink->tau_alpha);
        meta_f32(m, "attention.log_scale_floor", &m->ink->tau_floor);
    }
    struct ink_layer *L = &m->ink->L[l];
    const struct janas_jns_tensor *t;
    L->global = l % 6 == 5;
    ly->inkl = L;
#define NEED(var, suffix, type, d0, d1)                                        \
    if (!(var = need(m, (int)l, suffix, type, d0, d1, err, err_len)))          \
        return -1;
#define NEEDF(var, suffix, d0, d1)                                             \
    NEED(t, suffix, 0, d0, d1)                                                 \
    var = f32(m, t);
    NEED(L->wq, "attn_q.weight", -1, dm, QD)
    NEED(L->wk, "attn_k.weight", -1, dm, KVD)
    NEED(L->wv, "attn_v.weight", -1, dm, KVD)
    NEED(L->wgate, "attn_gate.weight", -1, dm, QD)
    NEED(L->wo, "attn_output.weight", -1, QD, dm)
    NEED(L->wrel, "attn_rel.weight", -1, dm, H * DR)
    if (!(L->wsink = small(m, l, "attn_sink_proj.weight", dm, H, &L->sink32,
                           err, err_len)))
        return -1;
    NEEDF(L->q_clip, "attn_q_clip.weight", HD, H)
    NEEDF(L->rel_clip, "attn_rel_clip.weight", DR, H)
    NEEDF(L->rel_tab, "attn_rel_bias.weight", DR, XR)
    NEEDF(L->kconv, "attn_k_conv.weight", KC, KVD)
    NEEDF(L->vconv, "attn_v_conv.weight", KC, KVD)
    NEEDF(L->pconv, "attn_post_conv.weight", KC, dm)
    NEEDF(L->fconv, "ffn_post_conv.weight", KC, dm)
    NEEDF(L->ffn_q, "ffn_res_query.weight", dm, 0)
    NEED(ly->ffn_norm, "ffn_norm.weight", 0, dm, 0)
    if (L->global) {
        NEEDF(L->attn_q, "attn_res_query.weight", dm, 3)
        NEEDF(L->qn, "attn_q_inp_norm.weight", dm, 0)
        NEEDF(L->kn, "attn_k_inp_norm.weight", dm, 0)
        NEEDF(L->vn, "attn_v_inp_norm.weight", dm, 0)
        if (!(L->kcp = small(m, l, "attn_k_conv_proj.weight", dm, HKV * KC,
                             &L->kcp32, err, err_len)) ||
            !(L->vcp = small(m, l, "attn_v_conv_proj.weight", dm, HKV * KC,
                             &L->vcp32, err, err_len)))
            return -1;
    } else {
        NEEDF(L->attn_norm, "attn_norm.weight", dm, 0)
        if (l > 0) {
            NEEDF(L->attn_q, "attn_res_query.weight", dm, 0)
        }
    }
    if (l >= LOOP_S && l < LOOP_E) {
        static const char *const nm[4] = {"q", "k", "v", "ffn"};
        char s[64];
        NEEDF(L->msg_q, "loop_msg_query.weight", dm, 4)
        for (int i = 0; i < 4; i++) {
            snprintf(s, sizeof s, "loop_%s_ctrl_down.weight", nm[i]);
            NEEDF(L->c_down[i], s, 2 * dm, RANK)
            snprintf(s, sizeof s, "loop_%s_ctrl_gain.weight", nm[i]);
            NEEDF(L->c_gain[i], s, RANK, dm)
            snprintf(s, sizeof s, "loop_%s_ctrl_forget.weight", nm[i]);
            NEEDF(L->c_forget[i], s, RANK, dm)
        }
    }
    /* the mixture: the router's experts all in the file, its choice's bias;
       the router at 32 bits, as the reference and llama.cpp have it (on
       1300 tokens against llama.cpp: the most likely token the same at 1279
       positions against 1275 with the router in half precision, the mean
       difference 0.490 against 0.511) */
    const struct janas_jns_layer *jl = &m->j.layers[l];
    NEED(t, "ffn_gate_inp.weight", 0, dm, 0)
    ly->n_exp = (uint32_t)t->dims[1];
    int absent = 0;
    for (uint32_t e = 0; e < ly->n_exp && e < m->n_expert; e++)
        absent |= janas_jns_expert_offset(&m->j, l, e) == 0;
    if (!jl->slot_bytes || ly->n_exp != m->n_expert || ly->n_exp < m->k ||
        absent) {
        snprintf(err, err_len, "layer %u: unexpected experts", l);
        return -1;
    }
    ly->router32 = f32(m, t);
    NEED(t, "exp_probs_b.bias", 0, ly->n_exp, 0)
    ly->exp_bias = f32(m, t);
    /* NVFP4 experts (unsloth's files) carry a scale an expert: of gate and
       up, applied before the activation; of down, folded into the expert's
       weight (model_ffn.c) */
    char sn[64];
    snprintf(sn, sizeof sn, "blk.%u.ffn_gate_up_exps.scale", l);
    if ((t = janas_jns_tensor(&m->j, sn)) && t->type == 0 &&
        t->dims[0] == ly->n_exp)
        L->gu_scale = f32(m, t);
    snprintf(sn, sizeof sn, "blk.%u.ffn_down_exps.scale", l);
    if ((t = janas_jns_tensor(&m->j, sn)) && t->type == 0 &&
        t->dims[0] == ly->n_exp)
        ly->exp_scale = f32(m, t);
#undef NEEDF
#undef NEED
    return 0;
}

int janas_m_ink_alloc(struct janas_llm_model *m)
{
    struct janas_ink *d = m->ink;
    size_t B = m->blk, dm = m->d_model;
#define A(p, count) (d->p = malloc((count) * sizeof(*d->p)))
    if (!A(h, B * dm) || !A(h6, B * dm) || !A(Bs, B * 4 * dm) ||
        !A(Bpre, B * 4 * dm) || !A(r, B * dm) || !A(bank, B * 2 * dm) ||
        !A(msg, B * 4 * dm) || !A(A, B * 3 * dm) || !A(xin, B * 3 * dm) ||
        !A(q, B * QD) || !A(kl, B * KVD) || !A(vl, B * KVD) || !A(k, B * KVD) ||
        !A(v, B * KVD) || !A(rel, B * H * DR) || !A(sink, B * H) ||
        !A(gate, B * QD) || !A(o, B * QD) || !A(aout, B * dm) ||
        !A(dk, B * HKV * KC) || !A(dv, B * HKV * KC) ||
        !A(oq, B * QD / JANAS_QK))
        return -1;
#undef A
    return 0;
}

static void seq_free(struct janas_ink_seq *s)
{
    if (!s)
        return;
    for (int i = 0; i < SLOTS; i++) {
        free(s->K[i]);
        free(s->V[i]);
        free(s->conv[i]);
    }
    free(s);
}

void janas_m_ink_seq_free(struct janas_llm_model *m, struct janas_seq *sq)
{
    if (m->ink && sq) {
        seq_free(sq->ink);
        sq->ink = NULL;
    }
}

void janas_m_ink_free(struct janas_llm_model *m)
{
    struct janas_ink *d = m->ink;
    if (!d)
        return;
    janas_m_ink_seq_free(m, &m->seq0);
    float *bufs[] = {d->h,    d->h6, d->Bs,  d->Bpre, d->r,    d->bank,
                     d->msg,  d->A,  d->xin, d->q,    d->kl,   d->vl,
                     d->k,    d->v,  d->rel, d->sink, d->gate, d->o,
                     d->aout, d->dk, d->dv,  d->work};
    for (size_t i = 0; i < sizeof bufs / sizeof *bufs; i++)
        free(bufs[i]);
    for (int l = 0; l < 24; l++) {
        free(d->L[l].sink32);
        free(d->L[l].kcp32);
        free(d->L[l].vcp32);
    }
    free(d->oq);
    free(d);
    m->ink = NULL;
}

/* The current sequence's state, made the first time it is used. */
static struct janas_ink_seq *seq_of(struct janas_llm_model *m)
{
    struct janas_seq *sq = m->cur;
    if (sq->ink)
        return sq->ink;
    struct janas_ink_seq *s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    /* a local ring: the window behind a block's first query, the block and
       the positions a speculative pass takes back */
    s->lcap = WIN + m->blk + MARGIN;
    for (uint32_t i = 0; i < SLOTS; i++) {
        size_t cap = slot_global(i) ? m->n_ctx : s->lcap;
        if (!(s->K[i] = malloc(cap * KVD * sizeof(float))) ||
            !(s->V[i] = malloc(cap * KVD * sizeof(float))) ||
            !(s->conv[i] = calloc((size_t)CC * CONV_CH, sizeof(float)))) {
            seq_free(s);
            return NULL;
        }
    }
    sq->ink = s;
    return s;
}

int janas_m_ink_rewind_ok(const struct janas_llm_model *m, uint32_t n)
{
    /* the conv inputs and the local rings keep MARGIN positions back */
    return !m->cur->ink || m->cur->ring_hi <= n + MARGIN;
}

/* ---- small pieces ---- */

static void rms_u(const float *x, float *y, uint32_t n, float eps)
{
    double s = 0;
    for (uint32_t i = 0; i < n; i++)
        s += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(s / n) + eps);
    for (uint32_t i = 0; i < n; i++)
        y[i] = x[i] * r;
}

static void rms_w(const float *x, const float *w, float *y, uint32_t n,
                  float eps)
{
    rms_u(x, y, n, eps);
    for (uint32_t i = 0; i < n; i++)
        y[i] *= w[i];
}

/* The depth mix of ns states S[i] (each dm wide) by nq queries Q[j]:
   out[j] = sum_i softmax_i(Q[j] . t_i / sqrt(dm)) t_i, t_i = rms_u(S[i]). */
static void mix(const float *const *S, uint32_t ns, const float *Q, uint32_t nq,
                float *out, uint32_t dm, float eps, float *t)
{
    for (uint32_t i = 0; i < ns; i++)
        rms_u(S[i], t + (size_t)i * dm, dm, eps);
    float sc = 1.0f / sqrtf((float)dm);
    for (uint32_t j = 0; j < nq; j++) {
        float a[8], mx = -INFINITY, sum = 0;
        for (uint32_t i = 0; i < ns; i++) {
            a[i] =
                janas_dot_f32(Q + (size_t)j * dm, t + (size_t)i * dm, dm) * sc;
            if (a[i] > mx)
                mx = a[i];
        }
        for (uint32_t i = 0; i < ns; i++)
            sum += a[i] = expf(a[i] - mx);
        float *y = out + (size_t)j * dm;
        memset(y, 0, dm * sizeof(float));
        for (uint32_t i = 0; i < ns; i++) {
            float w = a[i] / sum;
            const float *ti = t + (size_t)i * dm;
            for (uint32_t c = 0; c < dm; c++)
                y[c] += w * ti[c];
        }
    }
}

/* A message's gate on x, in place: (1 - tanh(forget u)) x + tanh(gain u) m,
   u = down [rms_u(x), rms_u(m)]. */
static void msg_gate(const struct ink_layer *L, int which, float *x,
                     const float *mm, uint32_t dm, float eps, float *c)
{
    rms_u(x, c, dm, eps);
    rms_u(mm, c + dm, dm, eps);
    float u[RANK];
    for (uint32_t r = 0; r < RANK; r++)
        u[r] = janas_dot_f32(L->c_down[which] + (size_t)r * 2 * dm, c, 2 * dm);
    for (uint32_t i = 0; i < dm; i++) {
        float f = tanhf(janas_dot_f32(L->c_forget[which] + (size_t)i * RANK, u,
                                      RANK)),
              g = tanhf(
                  janas_dot_f32(L->c_gain[which] + (size_t)i * RANK, u, RANK));
        x[i] = (1.0f - f) * x[i] + g * mm[i];
    }
}

static float tau_of(const struct janas_ink *d, uint32_t pos)
{
    float r = (float)(pos + 1) / d->tau_floor;
    return r > 1.0f ? 1.0f + d->tau_alpha * logf(r) : 1.0f;
}

/* v scaled to at most norm c (L2): v min(1, c / (|v| + 1e-6)) */
static void clip(float *v, uint32_t n, float c)
{
    double s = 0;
    for (uint32_t i = 0; i < n; i++)
        s += (double)v[i] * v[i];
    float f = c / ((float)sqrt(s) + 1e-6f);
    if (f < 1.0f)
        for (uint32_t i = 0; i < n; i++)
            v[i] *= f;
}

/* The products of nt matrices on the quantized rows of x (n of them). */
static void products(struct janas_llm_model *m, const float *x, uint32_t n,
                     const struct janas_jns_tensor *const *w, float *const *y,
                     int nt)
{
    uint32_t dm = m->d_model;
    for (uint32_t j = 0; j < n; j++)
        janas_q8k_quantize(x + (size_t)j * dm,
                           m->xq + (size_t)j * (dm / JANAS_QK), dm);
    struct janas_matvec_task t[8];
    for (int i = 0; i < nt; i++)
        t[i] = (struct janas_matvec_task){.type = (int)w[i]->type,
                                          .w = data(m, w[i]),
                                          .x = m->xq,
                                          .y = y[i],
                                          .rows = w[i]->dims[1],
                                          .cols = dm,
                                          .n_vec = n};
    janas_m_products(m, janas_m_gpu(m, 0), t, (size_t)nt);
}

/* y (n rows of r) = W (r rows of dm, f32) x (n rows of dm) */
static void rows32(const float *W, uint32_t r, const float *x, uint32_t n,
                   uint32_t dm, float *y)
{
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t i = 0; i < r; i++)
            y[(size_t)j * r + i] =
                janas_dot_f32(W + (size_t)i * dm, x + (size_t)j * dm, dm);
}

/* A causal conv of ch channels on the block's raw inputs x (n rows of ch,
   at positions pos0..), the slot's ring of past inputs at offset off of
   each row; static taps w[c][KC] (KC-1 the token), dynamic ones dyn (n
   rows of HKV x KC, by lag; NULL: none). The block's inputs are kept. */
static void conv(float *ring, uint32_t off, float *x, uint32_t ch, uint32_t n,
                 uint32_t pos0, const float *w, const float *dyn, float *y)
{
    float sc = 1.0f / sqrtf(dyn ? 2.0f * KC + 1.0f : KC + 1.0f);
    for (uint32_t j = 0; j < n; j++) {
        uint32_t p = pos0 + j;
        const float *in[KC]; /* in[lag]: the input lag positions back */
        for (uint32_t lag = 0; lag < KC; lag++) {
            if (lag > p)
                in[lag] = NULL;
            else if (lag <= j)
                in[lag] = x + (size_t)(j - lag) * ch;
            else
                in[lag] = ring + (size_t)((p - lag) % CC) * CONV_CH + off;
        }
        float *o = y + (size_t)j * ch;
        for (uint32_t c = 0; c < ch; c++) {
            float s = in[0][c];
            for (uint32_t lag = 0; lag < KC; lag++) {
                if (!in[lag])
                    continue;
                float wt = w[(size_t)c * KC + (KC - 1 - lag)];
                if (dyn)
                    wt += dyn[(size_t)j * HKV * KC + (c / HD) * KC + lag];
                s += wt * in[lag][c];
            }
            o[c] = s * sc;
        }
    }
    /* the block's inputs into the ring, the last CC of them */
    uint32_t from = n > CC ? n - CC : 0;
    for (uint32_t j = from; j < n; j++)
        memcpy(ring + (size_t)((pos0 + j) % CC) * CONV_CH + off,
               x + (size_t)j * ch, ch * sizeof(float));
}

/* ---- attention ---- */

struct att_job {
    struct janas_llm_model *m;
    const struct ink_layer *L;
    struct janas_ink_seq *s;
    uint32_t slot, n, pos0, global;
};

static void att_worker(void *arg, int tid, int n_threads)
{
    struct att_job *a = arg;
    struct janas_llm_model *m = a->m;
    struct janas_ink *d = m->ink;
    const struct ink_layer *L = a->L;
    float *sc = d->work + (size_t)tid * d->work_per,
          *prof = sc + d->work_per - XR;
    const float *K = a->s->K[a->slot], *V = a->s->V[a->slot];
    uint32_t cap = a->global ? m->n_ctx : a->s->lcap;
    const float qs = 1.0f / sqrtf((float)HD);
    for (uint32_t it = (uint32_t)tid; it < a->n * H;
         it += (uint32_t)n_threads) {
        uint32_t j = it / H, h = it % H, g = h / (H / HKV);
        uint32_t pos = a->pos0 + j;
        const float *q = d->q + (size_t)j * QD + (size_t)h * HD;
        const float *r = d->rel + (size_t)j * H * DR + (size_t)h * DR;
        for (uint32_t x = 0; x < XR; x++)
            prof[x] = janas_dot_f32(r, L->rel_tab + (size_t)x * DR, DR);
        uint32_t lo = a->global ? 0 : (pos + 1 > WIN ? pos + 1 - WIN : 0);
        float mx = d->sink[(size_t)j * H + h];
        for (uint32_t p = lo; p <= pos; p++) {
            const float *k = K + (size_t)(p % cap) * KVD + (size_t)g * HD;
            uint32_t dist = pos - p;
            float s =
                janas_dot_f32(q, k, HD) * qs + (dist < XR ? prof[dist] : 0.0f);
            sc[p - lo] = s;
            if (s > mx)
                mx = s;
        }
        float sum = expf(d->sink[(size_t)j * H + h] - mx);
        for (uint32_t p = lo; p <= pos; p++)
            sum += sc[p - lo] = expf(sc[p - lo] - mx);
        float *o = d->o + (size_t)j * QD + (size_t)h * HD;
        memset(o, 0, HD * sizeof(float));
        for (uint32_t p = lo; p <= pos; p++) {
            float w = sc[p - lo] / sum;
            const float *v = V + (size_t)(p % cap) * KVD + (size_t)g * HD;
            for (uint32_t c = 0; c < HD; c++)
                o[c] += w * v[c];
        }
    }
}

/* JANAS_TRACE: the sum of a stage's last row, as llama.cpp's
   eval-callback prints its tensors' */
static void tsum(const struct janas_llm_model *m, uint32_t l, const char *what,
                 const float *v, uint32_t n_rows, uint32_t w)
{
    if (!m->trace || l > 1)
        return;
    double s = 0;
    const float *r = v + (size_t)(n_rows - 1) * w;
    for (uint32_t i = 0; i < w; i++)
        s += r[i];
    fprintf(stderr, "ink %u %-10s sum %.6f\n", l, what, s);
}

/* ---- a layer ---- */

/* Layer l (physical) in slot s on the block: h the partial stream (n rows),
   Bs the block states (nb of them a token, 4 a row), M the messages (4 a
   token) or NULL. h and nb are updated. */
static int layer(struct janas_llm_model *m, uint32_t l, uint32_t slot,
                 struct janas_ink_seq *s, uint32_t n, uint32_t pos0,
                 uint32_t *nb, const float *M)
{
    struct janas_ink *d = m->ink;
    const struct ink_layer *L = &d->L[l];
    const struct layer *ly = &m->layers[l];
    uint32_t dm = m->d_model;
    float eps = m->eps;
    float *t = malloc((size_t)8 * dm * sizeof(float)); /* mix and gate room */
    if (!t)
        return -1;
    /* the attention's inputs: A (1 or 3 a token), gated, normed into xin */
    uint32_t na = L->global ? 3 : 1;
    for (uint32_t j = 0; j < n; j++) {
        float *hj = d->h + (size_t)j * dm, *Aj = d->A + (size_t)j * 3 * dm;
        const float *S[6];
        uint32_t ns = 0;
        for (uint32_t i = 0; i < *nb; i++)
            S[ns++] = d->Bs + ((size_t)j * 4 + i) * dm;
        S[ns++] = hj;
        if (l == 0) {
            memcpy(Aj, hj, dm * sizeof(float));
        } else
            mix(S, ns, L->attn_q, na, Aj, dm, eps, t);
        float *xj = d->xin + (size_t)j * 3 * dm;
        for (uint32_t i = 0; i < 3; i++) {
            /* the input of q (0), k (1), v (2) */
            float *src = Aj + (size_t)(L->global ? i : 0) * dm;
            float *dst = xj + (size_t)i * dm;
            memcpy(dst, src, dm * sizeof(float));
            if (M)
                msg_gate(L, (int)i, dst, M + ((size_t)j * 4 + i) * dm, dm, eps,
                         t);
            const float *nw = L->global ? (i == 0   ? L->qn
                                           : i == 1 ? L->kn
                                                    : L->vn)
                                        : L->attn_norm;
            rms_w(dst, nw, dst, dm, eps);
        }
    }
    if (m->trace && l <= 1) {
        tsum(m, l, "h", d->h, n, dm);
        tsum(m, l, "xq", d->xin, n, 3 * dm);
    }
    /* the projections: of xq q, the position states, the sinks and the
       gate; of xk k (and the dynamic k taps); of xv v (and its taps) */
    float *tmp = malloc((size_t)n * dm * sizeof(float));
    if (!tmp) {
        free(t);
        return -1;
    }
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < n; j++)
            memcpy(tmp + (size_t)j * dm, d->xin + ((size_t)j * 3 + i) * dm,
                   dm * sizeof(float));
        if (i == 0) {
            const struct janas_jns_tensor *w[] = {L->wq, L->wrel, L->wgate,
                                                  L->wsink};
            float *y[] = {d->q, d->rel, d->gate, d->sink};
            products(m, tmp, n, w, y, L->sink32 ? 3 : 4);
            if (L->sink32)
                rows32(L->sink32, H, tmp, n, dm, d->sink);
        } else if (i == 1) {
            const struct janas_jns_tensor *w[] = {L->wk, L->kcp};
            float *y[] = {d->kl, d->dk};
            products(m, tmp, n, w, y, L->global && !L->kcp32 ? 2 : 1);
            if (L->global && L->kcp32)
                rows32(L->kcp32, HKV * KC, tmp, n, dm, d->dk);
        } else {
            const struct janas_jns_tensor *w[] = {L->wv, L->vcp};
            float *y[] = {d->vl, d->dv};
            products(m, tmp, n, w, y, L->global && !L->vcp32 ? 2 : 1);
            if (L->global && L->vcp32)
                rows32(L->vcp32, HKV * KC, tmp, n, dm, d->dv);
        }
    }
    /* q and the position states: scaled, clipped, grown by tau (global) */
    for (uint32_t j = 0; j < n; j++) {
        float tau = L->global ? tau_of(d, pos0 + j) : 1.0f;
        for (uint32_t h = 0; h < H; h++) {
            float *q = d->q + (size_t)j * QD + (size_t)h * HD;
            for (uint32_t c = 0; c < HD; c++)
                q[c] *= L->q_clip[(size_t)h * HD + c];
            clip(q, HD, 22.627416997969522f);
            float *r = d->rel + (size_t)j * H * DR + (size_t)h * DR;
            for (uint32_t c = 0; c < DR; c++)
                r[c] *= L->rel_clip[(size_t)h * DR + c];
            clip(r, DR, 8.0f);
            if (tau != 1.0f) {
                for (uint32_t c = 0; c < HD; c++)
                    q[c] *= tau;
                for (uint32_t c = 0; c < DR; c++)
                    r[c] *= tau;
            }
        }
    }
    /* k and v through their convs, k clipped, v normalized, into the slot */
    float *ring = s->conv[slot];
    conv(ring, 0, d->kl, KVD, n, pos0, L->kconv, L->global ? d->dk : NULL,
         d->k);
    conv(ring, KVD, d->vl, KVD, n, pos0, L->vconv, L->global ? d->dv : NULL,
         d->v);
    uint32_t cap = slot_global(slot) ? m->n_ctx : s->lcap;
    for (uint32_t j = 0; j < n; j++) {
        uint32_t p = pos0 + j;
        for (uint32_t g = 0; g < HKV; g++) {
            float *k = d->k + (size_t)j * KVD + (size_t)g * HD;
            float *v = d->v + (size_t)j * KVD + (size_t)g * HD;
            clip(k, HD, 22.627416997969522f);
            rms_u(v, v, HD, eps);
        }
        memcpy(s->K[slot] + (size_t)(p % cap) * KVD, d->k + (size_t)j * KVD,
               KVD * sizeof(float));
        memcpy(s->V[slot] + (size_t)(p % cap) * KVD, d->v + (size_t)j * KVD,
               KVD * sizeof(float));
    }
    tsum(m, l, "Qcur", d->q, n, QD);
    tsum(m, l, "Kcur", d->k, n, KVD);
    tsum(m, l, "Vcur", d->v, n, KVD);
    tsum(m, l, "Rcur", d->rel, n, H * DR);
    tsum(m, l, "sink", d->sink, n, H);
    /* attention */
    struct att_job aj = {.m = m,
                         .L = L,
                         .s = s,
                         .slot = slot,
                         .n = n,
                         .pos0 = pos0,
                         .global = (uint32_t)slot_global(slot)};
    janas_pool_run(m->compute, att_worker, &aj);
    tsum(m, l, "kqv_out", d->o, n, QD);
    /* gated, projected, its conv */
    for (size_t i = 0; i < (size_t)n * QD; i++)
        d->o[i] *= sigmoid(d->gate[i]);
    for (uint32_t j = 0; j < n; j++)
        janas_q8k_quantize(d->o + (size_t)j * QD,
                           d->oq + (size_t)j * (QD / JANAS_QK), QD);
    struct janas_matvec_task wo = {.type = (int)L->wo->type,
                                   .w = data(m, L->wo),
                                   .x = d->oq,
                                   .y = tmp,
                                   .rows = dm,
                                   .cols = QD,
                                   .n_vec = n};
    janas_m_products(m, janas_m_gpu(m, 0), &wo, 1);
    tsum(m, l, "wo", tmp, n, dm);
    conv(ring, 2 * KVD, tmp, dm, n, pos0, L->pconv, NULL, d->aout);
    tsum(m, l, "aout", d->aout, n, dm);
    /* the residual: a block's first layer starts the partial stream again,
       its input becoming a block state */
    int boundary = l % 6 == 0;
    for (uint32_t j = 0; j < n; j++) {
        float *hj = d->h + (size_t)j * dm, *rj = d->r + (size_t)j * dm;
        const float *aj2 = d->aout + (size_t)j * dm;
        if (boundary) {
            memcpy(d->Bs + ((size_t)j * 4 + *nb) * dm, hj, dm * sizeof(float));
            memcpy(rj, aj2, dm * sizeof(float));
        } else
            for (uint32_t c = 0; c < dm; c++)
                rj[c] = hj[c] + aj2[c];
    }
    if (boundary)
        (*nb)++;
    /* the mixture's input: the mix of the blocks and r, gated, into x */
    for (uint32_t j = 0; j < n; j++) {
        const float *S[6];
        uint32_t ns = 0;
        for (uint32_t i = 0; i < *nb; i++)
            S[ns++] = d->Bs + ((size_t)j * 4 + i) * dm;
        S[ns++] = d->r + (size_t)j * dm;
        float *xj = m->x + (size_t)j * dm;
        mix(S, ns, L->ffn_q, 1, xj, dm, eps, t);
        if (M)
            msg_gate(L, 3, xj, M + ((size_t)j * 4 + 3) * dm, dm, eps, t);
    }
    tsum(m, l, "ffn_inp", d->r, n, dm);
    tsum(m, l, "ffn_mix", m->x, n, dm);
    janas_m_norm_quant(m, f32(m, ly->ffn_norm), 0, n);
    tsum(m, l, "ffn_norm", m->xn, n, dm);
    memset(m->x, 0, (size_t)n * dm * sizeof(float));
    m->moe_gu_scale = L->gu_scale;
    int moe =
        janas_m_moe_block(m, ly, m->cache, &m->j.layers[l], l, n, MOE_ALL);
    m->moe_gu_scale = NULL;
    if (moe != 0) {
        free(tmp);
        free(t);
        return -1;
    }
    /* its conv, then the partial stream */
    tsum(m, l, "moe_out", m->x, n, dm);
    memcpy(tmp, m->x, (size_t)n * dm * sizeof(float));
    conv(ring, 2 * KVD + dm, tmp, dm, n, pos0, L->fconv, NULL, m->x);
    tsum(m, l, "ffn_out", m->x, n, dm);
    for (size_t i = 0; i < (size_t)n * dm; i++)
        d->h[i] = d->r[i] + m->x[i];
    tsum(m, l, "l_out", d->h, n, dm);
    free(tmp);
    free(t);
    return 0;
}

static void trace_h(struct janas_llm_model *m, uint32_t eff, uint32_t n)
{
    if (!m->trace)
        return;
    uint32_t dm = m->d_model;
    const float *v = m->ink->h + (size_t)(n - 1) * dm;
    double nn = 0, sum = 0;
    for (uint32_t i = 0; i < dm; i++) {
        nn += (double)v[i] * v[i];
        sum += v[i];
    }
    fprintf(stderr, "l_out-%-3u norm %10.4f sum %12.6f\n", eff, sqrt(nn), sum);
}

int janas_m_ink_layers(struct janas_llm_model *m, uint32_t n, uint32_t pos0)
{
    struct janas_ink *d = m->ink;
    struct janas_ink_seq *s = seq_of(m);
    uint32_t dm = m->d_model;
    if (!s || pos0 + n > m->n_ctx || n > m->blk)
        return -1;
    if (!d->work) {
        /* a thread's scores (the whole context) and position profile */
        d->n_threads = (uint32_t)janas_pool_size(m->compute);
        d->work_per = m->n_ctx + XR;
        if (!(d->work =
                  malloc((size_t)d->n_threads * d->work_per * sizeof(float))))
            return -1;
    }
    /* the embedding, normalized: the first stream */
    for (uint32_t j = 0; j < n; j++)
        rms_w(m->x + (size_t)j * dm, d->embd_norm, d->h + (size_t)j * dm, dm,
              m->eps);
    uint32_t nb = 0, eff = 0;
    for (uint32_t l = 0; l < LOOP_S; l++, eff++) {
        if (layer(m, l, l, s, n, pos0, &nb, NULL) != 0)
            return -1;
        trace_h(m, eff, n);
    }
    /* the core: three passes from the prelude's output and blocks, each
       after the first taking messages from the one before */
    memcpy(d->h6, d->h, (size_t)n * dm * sizeof(float));
    memcpy(d->Bpre, d->Bs, (size_t)n * 4 * dm * sizeof(float));
    uint32_t nb_pre = nb;
    for (uint32_t p = 1; p <= PASSES; p++) {
        if (p > 1) {
            /* the bank: the pass before's layer-12 block and its output */
            for (uint32_t j = 0; j < n; j++) {
                memcpy(d->bank + (size_t)j * 2 * dm,
                       d->Bs + ((size_t)j * 4 + nb_pre + 1) * dm,
                       dm * sizeof(float));
                memcpy(d->bank + ((size_t)j * 2 + 1) * dm,
                       d->h + (size_t)j * dm, dm * sizeof(float));
            }
            memcpy(d->h, d->h6, (size_t)n * dm * sizeof(float));
            memcpy(d->Bs, d->Bpre, (size_t)n * 4 * dm * sizeof(float));
            nb = nb_pre;
        }
        for (uint32_t l = LOOP_S; l < LOOP_E; l++, eff++) {
            uint32_t slot = p == 1 ? l : 24 + (p - 2) * 12 + (l - LOOP_S);
            const float *M = NULL;
            if (p > 1) {
                float *t = malloc((size_t)2 * dm * sizeof(float));
                if (!t)
                    return -1;
                for (uint32_t j = 0; j < n; j++) {
                    const float *S[2] = {d->bank + (size_t)j * 2 * dm,
                                         d->bank + ((size_t)j * 2 + 1) * dm};
                    mix(S, 2, d->L[l].msg_q, 4, d->msg + (size_t)j * 4 * dm, dm,
                        m->eps, t);
                }
                free(t);
                M = d->msg;
            }
            if (layer(m, l, slot, s, n, pos0, &nb, M) != 0)
                return -1;
            trace_h(m, eff, n);
        }
    }
    for (uint32_t l = LOOP_E; l < 24; l++, eff++) {
        if (layer(m, l, l, s, n, pos0, &nb, NULL) != 0)
            return -1;
        trace_h(m, eff, n);
    }
    /* the final mix of the blocks and the output, for model.c's norm and
       head */
    float *t = malloc((size_t)6 * dm * sizeof(float));
    if (!t)
        return -1;
    for (uint32_t j = 0; j < n; j++) {
        const float *S[6];
        uint32_t ns = 0;
        for (uint32_t i = 0; i < nb; i++)
            S[ns++] = d->Bs + ((size_t)j * 4 + i) * dm;
        S[ns++] = d->h + (size_t)j * dm;
        mix(S, ns, d->out_q, 1, m->x + (size_t)j * dm, dm, m->eps, t);
    }
    free(t);
    return 0;
}
