/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model.c - Janas-LLM forward pass (see model.h).
 *
 * What is computed follows the model definitions (llama.cpp's graphs are the
 * reference the logits are compared against); how it is computed is Janas's
 * own: fused single-pass kernels, a KV cache only for the attention layers,
 * the shared expert run as one more expert of the MoE dispatch.
 *
 * Architectures:
 *   qwen3moe   every layer: attention (q/k norms, NeoX RoPE) + routed experts
 *   qwen3next  every full_attention_interval-th layer: gated attention with
 *              partial RoPE; the others: Gated DeltaNet (see deltanet.h)
 *              with a causal convolution; every layer: routed experts plus a
 *              shared expert scaled by a sigmoid gate
 *   qwen35moe  (Qwen3.5, Qwen3.6) the same layers; the DeltaNet projections
 *              come separate (q|k|v, z, alpha and beta in f32) instead of
 *              grouped per key head; the last block of the file is the
 *              multi-token prediction block, not a layer of the model
 *   gemma4     sliding-window layers (a 1024-position window) and every
 *              sixth a full one, each kind with its head size, KV heads
 *              and RoPE; norms after attention and after the FFN too, GELU,
 *              a scale per layer, soft-capped logits
 * What sets them apart is data, in arch.h; a dense model is a mixture with
 * one expert per layer.
 *
 * This file runs the passes (forward, MTP); the layers are in model_attn.c,
 * model_rec.c and model_ffn.c, opening and closing in model_load.c, the
 * threads and the experts' profile in model_tune.c, saved states in
 * model_state.c.
 */
#define _GNU_SOURCE
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "attention.h"
#include "common/pool.h"
#include "common/sysinfo.h"
#include "deltanet.h"
#include "tuner.h"
#include "gpu.h"
#include "matvec.h"
#include "quant.h"
#include "vmath.h"

#include "model_impl.h"

struct janas_pool *janas_llm_model_compute(struct janas_llm_model *m)
{
    return m->compute;
}

void janas_llm_model_phases(const struct janas_llm_model *m,
                            double out[JANAS_PH_COUNT])
{
    memcpy(out, m->phase, sizeof(m->phase));
    /* la sonda stampa e azzera qui: chi legge le fasi sta separando due
       tratti (prefill e decodifica), e la scomposizione deve separarli
       insieme a lui, altrimenti il prefill sporca la decodifica */
    if (janas_m_exp_tracing())
        janas_m_exp_report();
}

static void embed(struct janas_llm_model *m, int32_t token, float *x)
{
    uint32_t dm = m->d_model;
    size_t row_bytes =
        dm / JANAS_QK * janas_qtype_block_size((int)m->tok_embd->type);
    const uint8_t *row =
        (const uint8_t *)data(m, m->tok_embd) + (size_t)token * row_bytes;
    janas_dequantize((int)m->tok_embd->type, (const void *)row, x, dm);
    if (m->a->sandwich) { /* Gemma: the embedding scaled by sqrt(d_model) */
        float s = sqrtf((float)dm);
        for (uint32_t i = 0; i < dm; i++)
            x[i] *= s;
    }
}

void janas_m_embed(struct janas_llm_model *m, int32_t token, float *x)
{
    embed(m, token, x);
}

/* RoPE angles of positions pos0 .. pos0 + n - 1 for n_rot dimensions, the
   same for every head; ff: a divisor of each pair's angle, or NULL. */
static void rope_table(float *cs, float *sn, uint32_t n, uint32_t pos0,
                       uint32_t n_rot, float base, const float *ff)
{
    uint32_t half = n_rot / 2;
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t i = 0; i < half; i++) {
            float theta =
                (float)(pos0 + j) * powf(base, -2.0f * (float)i / (float)n_rot);
            if (ff)
                theta /= ff[i];
            cs[(size_t)j * half + i] = cosf(theta);
            sn[(size_t)j * half + i] = sinf(theta);
        }
}

static void set_rope(struct janas_llm_model *m, uint32_t n, uint32_t pos0)
{
    rope_table(m->rope_cos, m->rope_sin, n, pos0, m->n_rot, m->rope_base,
               m->rope_freqs);
    if (m->a->swa)
        rope_table(m->rope_cos_swa, m->rope_sin_swa, n, pos0, m->n_rot_swa,
                   m->rope_base_swa, NULL);
}

/* Gemma 4, after the FFN: x = (xres + rms_norm(x) * post_ffn) * out_scale,
   the FFN's output having been summed into a zeroed x */
static void sandwich_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    struct janas_llm_model *m = t->m;
    uint32_t dm = m->d_model;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads) {
        float *x = m->x + (size_t)j * dm;
        const float *r = m->xres + (size_t)j * dm;
        rms_norm(x, x, t->w, dm, m->eps);
        /* with per-layer embeddings the scale comes after their branch */
        float s = m->ple_dim ? 1.0f : t->ly->out_scale;
        for (uint32_t i = 0; i < dm; i++)
            x[i] = (r[i] + x[i]) * s;
    }
}

/* Gemma 4's mixture, token by token: the dense branch's output normalized
   aside (xbr), then the experts' input and the router's from the residual,
   then the experts' output normalized and the two summed */
static void branch1_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    struct janas_llm_model *m = t->m;
    uint32_t dm = m->d_model;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads)
        rms_norm(m->xbr + (size_t)j * dm, m->x + (size_t)j * dm,
                 f32(m, t->ly->post_norm1), dm, m->eps);
}

static void prep2_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    struct janas_llm_model *m = t->m;
    uint32_t dm = m->d_model;
    const float *rs = t->ly->router_scale;
    float sd = 1.0f / sqrtf((float)dm);
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads) {
        const float *r = m->xres + (size_t)j * dm;
        float *xn = m->xn + (size_t)j * dm;
        /* the experts' input, to Q8_K through xn */
        rms_norm(xn, r, f32(m, t->ly->pre_norm2), dm, m->eps);
        janas_q8k_quantize(xn, m->xq + j * (dm / JANAS_QK), dm);
        /* the router's: the residual normalized without weights, scaled
           by 1 / sqrt(d_model) and per dimension */
        double ss = 0.0;
        for (uint32_t i = 0; i < dm; i++)
            ss += (double)r[i] * r[i];
        float sc = 1.0f / sqrtf((float)(ss / dm) + m->eps);
        for (uint32_t i = 0; i < dm; i++)
            xn[i] = r[i] * sc * sd * rs[i];
    }
}

static void branch2_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    struct janas_llm_model *m = t->m;
    uint32_t dm = m->d_model;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads) {
        float *x = m->x + (size_t)j * dm;
        const float *b = m->xbr + (size_t)j * dm;
        rms_norm(x, x, f32(m, t->ly->post_norm2), dm, m->eps);
        for (uint32_t i = 0; i < dm; i++)
            x[i] += b[i];
    }
}

static void run_tok(struct janas_llm_model *m, const struct layer *ly,
                    uint32_t n, void (*fn)(void *, int, int))
{
    struct tok_job t = {.m = m, .ly = ly, .n = n};
    if (n < 2)
        fn(&t, 0, 1);
    else
        janas_pool_run(m->compute, fn, &t);
}

/* logits soft-capped: c * tanh(l / c) */
static void softcap_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    float c = t->m->softcap, *l = t->out;
    size_t len = (size_t)t->n * t->m->n_vocab;
    size_t per = (len + (size_t)n_threads - 1) / (size_t)n_threads;
    size_t a = (size_t)tid * per, b = a + per < len ? a + per : len;
    for (size_t i = a; i < b; i++)
        l[i] = c * tanhf(l[i] / c);
}

/* A key row turned back by the angles in cs, sn (half pairs, NeoX) and put
   in its bits again: a rotation can take a pair past the largest number of
   the row, so the scale is found anew. row: int8 or int16 (wide). */
static void unrotate_row(void *row, float *scale, uint32_t hd, uint32_t half,
                         const float *cs, const float *sn, int wide)
{
    float tmp[512], mx = 0.0f, z = *scale;
    int8_t *r8 = row;
    int16_t *r16 = row;
    for (uint32_t i = 0; i < hd; i++)
        tmp[i] = (wide ? (float)r16[i] : (float)r8[i]) * z;
    for (uint32_t i = 0; i < half; i++) {
        float a = tmp[i], b = tmp[i + half];
        tmp[i] = a * cs[i] + b * sn[i];
        tmp[i + half] = b * cs[i] - a * sn[i];
    }
    for (uint32_t i = 0; i < hd; i++)
        if (fabsf(tmp[i]) > mx)
            mx = fabsf(tmp[i]);
    float top = wide ? 32767.0f : 127.0f, inv = mx > 0 ? top / mx : 0.0f;
    for (uint32_t i = 0; i < hd; i++) {
        if (wide)
            r16[i] = (int16_t)lrintf(tmp[i] * inv);
        else
            r8[i] = (int8_t)lrintf(tmp[i] * inv);
    }
    *scale = mx / top;
}

/*
 * Forgets the positions keep .. keep + drop of the attention layers: what
 * follows them moves down and its keys are turned back by drop positions,
 * since the rotation a key carries depends on where it was. The recurrent
 * state is left alone: it is a summary of everything seen, and in a hybrid
 * model that is what carries the older text once the window has passed it.
 * The first `keep` positions stay where they are, which keeps the system
 * message - and the anchor it gives attention - in place. Each slot with its
 * own geometry and RoPE (Gemma 4: sliding-window layers and full ones); a
 * ring (a sliding-window layer's) keeps only its last positions, and those
 * are the ones that move.
 */
int janas_llm_model_shift(struct janas_llm_model *m, uint32_t keep,
                          uint32_t drop, uint32_t n_kv)
{
    if (!drop || n_kv > m->n_ctx || keep + drop > n_kv || m->hd_max > 512)
        return -1;
    uint32_t move = n_kv - keep - drop, slots = m->n_attn + (m->mtp ? 1u : 0u);
    size_t elt = (size_t)1 << m->kv16;
    float cs[256], sn[256];
    uint8_t *ring_tmp = NULL;
    for (uint32_t sl = 0; sl < slots; sl++) {
        uint32_t hd = m->kv_hd[sl], cap = m->kv_cap[sl];
        int swa = m->kv_swa[sl];
        uint32_t n_rot = swa ? m->n_rot_swa : m->n_rot, half = n_rot / 2;
        float base = swa ? m->rope_base_swa : m->rope_base;
        const float *ff = swa ? NULL : m->rope_freqs;
        for (uint32_t i = 0; i < half; i++) {
            float theta =
                (float)drop * powf(base, -2.0f * (float)i / (float)n_rot);
            if (ff)
                theta /= ff[i];
            cs[i] = cosf(theta);
            sn[i] = sinf(theta);
        }
        size_t row = hd * elt;
        for (uint32_t h = 0; h < m->kv_heads[sl]; h++) {
            uint8_t *k = (uint8_t *)m->kcache + m->kv_off[sl] + h * cap * row;
            uint8_t *v = (uint8_t *)m->vcache + m->kv_off[sl] + h * cap * row;
            float *kz = m->kscale + m->ks_off[sl] + (size_t)h * cap;
            float *vz = m->vscale + m->ks_off[sl] + (size_t)h * cap;
            if (!m->kv_ring[sl]) {
                if (move) {
                    memmove(k + keep * row, k + (keep + drop) * row,
                            move * row);
                    memmove(v + keep * row, v + (keep + drop) * row,
                            move * row);
                    memmove(kz + keep, kz + keep + drop, move * sizeof(float));
                    memmove(vz + keep, vz + keep + drop, move * sizeof(float));
                }
                for (uint32_t p = keep; p < keep + move; p++)
                    unrotate_row(k + p * row, kz + p, hd, half, cs, sn,
                                 m->kv16);
                continue;
            }
            /* a ring: the positions it still has, from a copy of it, each
               to its new row (they cannot land on a kept one: see the
               bound on lo) */
            if (!ring_tmp &&
                !(ring_tmp = malloc(2 * (size_t)cap * (row + sizeof(float)))))
                return -1;
            uint8_t *tk = ring_tmp, *tv = tk + cap * row;
            float *tkz = (float *)(void *)(tv + cap * row), *tvz = tkz + cap;
            memcpy(tk, k, cap * row);
            memcpy(tv, v, cap * row);
            memcpy(tkz, kz, cap * sizeof(float));
            memcpy(tvz, vz, cap * sizeof(float));
            uint32_t lo = keep + drop;
            if (n_kv > cap && n_kv - cap > lo)
                lo = n_kv - cap;
            for (uint32_t p = lo; p < n_kv; p++) {
                uint32_t src = p % cap, dst = (p - drop) % cap;
                memcpy(k + dst * row, tk + src * row, row);
                memcpy(v + dst * row, tv + src * row, row);
                kz[dst] = tkz[src];
                vz[dst] = tvz[src];
                unrotate_row(k + dst * row, kz + dst, hd, half, cs, sn,
                             m->kv16);
            }
        }
    }
    free(ring_tmp);
    if (m->ring_hi > keep + drop)
        m->ring_hi -= drop;
    /* the recurrent state holds what it has seen, dropped tokens included -
       that is what carries the older text - but the position it stands at
       moves with everything else */
    if (m->base_pos != UINT32_MAX)
        m->base_pos = m->base_pos >= keep + drop ? m->base_pos - drop
                      : m->base_pos > keep       ? keep
                                                 : m->base_pos;
    return 0;
}

static int forward(struct janas_llm_model *m, const int32_t *tokens, uint32_t n,
                   uint32_t pos0, float *logits, int all_logits);

int janas_llm_model_forward(struct janas_llm_model *m, const int32_t *tokens,
                            uint32_t n, uint32_t pos0, float *logits,
                            int all_logits)
{
    /* each pass measured for the tuner: the configuration never changes
       the results */
    int cls = janas_tune_class(n), cand = janas_tuner_pick(&m->tuner, cls);
    janas_m_apply_candidate(m, cand);
    /* measured without the time spent waiting for experts read from disk:
       it depends on the cache, not on the configuration */
    struct janas_expert_cache_stats c0, c1;
    janas_expert_cache_stats(m->cache, &c0);
    double t0 = now();
    int r = forward(m, tokens, n, pos0, logits, all_logits);
    double dt = now() - t0;
    janas_expert_cache_stats(m->cache, &c1);
    dt -= c1.wait_seconds - c0.wait_seconds;
    /* while the cache is still filling in the background, passes are not
       representative: neither the tuner nor the draft planner may learn
       from them, or a slow start decides the whole session */
    if (r == 0 && dt > 0 && !janas_llm_model_settling(m))
        janas_tuner_record(&m->tuner, cls, cand, dt / n);
    janas_m_arena_to_gpu(m);
    return r;
}

static int forward(struct janas_llm_model *m, const int32_t *tokens, uint32_t n,
                   uint32_t pos0, float *logits, int all_logits)
{
    if (n < 1 || n > m->blk || pos0 + n > m->n_ctx)
        return -1;
    for (uint32_t j = 0; j < n; j++)
        if (tokens[j] < 0 || (uint32_t)tokens[j] >= m->n_vocab)
            return -1;
    if (m->a->rec) {
        if (pos0 == 0) {
            m->src = 0;
            m->replay = 0;
            memset(m->ssm_buf, 0,
                   (size_t)m->n_rec * m->n_vh * m->ds * m->ds *
                       sizeof(uint16_t));
            memset(m->conv_buf, 0,
                   (size_t)m->n_rec * m->conv_ch * (m->d_conv - 1) *
                       sizeof(float));
        } else if (m->base_pos == UINT32_MAX || pos0 < m->base_pos ||
                   pos0 > m->base_pos + m->log_n) {
            return -1; /* before the last call: that state is gone */
        } else if (pos0 == m->base_pos + m->log_n && m->log_n > 0) {
            m->src = 1 - m->base; /* from the tip */
            m->replay = 0;
        } else {
            m->src = m->base; /* replay the accepted tokens on the base */
            m->replay = pos0 - m->base_pos;
        }
        m->base_pos = UINT32_MAX; /* invalid until this pass completes */
    }
    uint32_t dm = m->d_model;
    size_t nbd = dm / JANAS_QK;

    for (uint32_t j = 0; j < n; j++)
        embed(m, tokens[j], m->x + (size_t)j * dm);
    if (m->trace >= 2) {
        fprintf(stderr, "tokens:");
        for (uint32_t j = 0; j < n; j++)
            fprintf(stderr, " %d", tokens[j]);
        fprintf(stderr, "\n");
        trace_row(m, "embed", m->x + (size_t)(n - 1) * dm, dm);
    }
    if (m->ple_dim && janas_m_ple_inputs(m, tokens, n) != 0)
        return -1;
    set_rope(m, n, pos0);
    for (uint32_t l = 0; l < m->n_layer; l++) {
        const struct layer *ly = &m->layers[l];
        janas_m_norm_quant(m, f32(m, ly->attn_norm), 0, n);
        trace_row(m, "attn_norm", m->xn + (size_t)(n - 1) * dm, dm);
        if (ly->rec)
            janas_m_rec_layer(m, ly, n);
        else
            janas_m_attn_layer(m, ly, n, pos0);
        janas_m_norm_quant(m, f32(m, ly->ffn_norm), 0, n);
        trace_row(m, "post_norm", m->xn + (size_t)(n - 1) * dm, dm);
        if (ly->post_ffn) {
            /* the FFN's output normalized before the residual: summed
               into a zeroed x, the residual aside */
            memcpy(m->xres, m->x, (size_t)n * dm * sizeof(float));
            memset(m->x, 0, (size_t)n * dm * sizeof(float));
        }
        if (ly->pre_norm2) {
            /* Gemma 4's mixture: the dense branch on xn / xq as normed
               above, then the experts' branch */
            size_t bytes = (size_t)n * dm * sizeof(float);
            if (janas_m_moe_block(m, ly, m->cache, &m->j.layers[l], l, n,
                                  MOE_SHARED) != 0)
                return -1;
            run_tok(m, ly, n, branch1_worker);
            trace_row(m, "ffn_mlp", m->xbr + (size_t)(n - 1) * dm, dm);
            run_tok(m, ly, n, prep2_worker);
            memset(m->x, 0, bytes);
            if (janas_m_moe_block(m, ly, m->cache, &m->j.layers[l], l, n,
                                  MOE_ROUTED) != 0)
                return -1;
            run_tok(m, ly, n, branch2_worker);
            trace_row(m, "ffn_moe_comb", m->x + (size_t)(n - 1) * dm, dm);
        } else if (janas_m_moe_block(m, ly, m->cache, &m->j.layers[l], l, n,
                                     MOE_ALL) != 0) {
            return -1;
        }
        if (ly->post_ffn) {
            struct tok_job st = {
                .m = m, .ly = ly, .w = f32(m, ly->post_ffn), .n = n};
            if (n < 2)
                sandwich_worker(&st, 0, 1);
            else
                janas_pool_run(m->compute, sandwich_worker, &st);
        }
        if (m->ple_dim)
            janas_m_ple_layer(m, ly, l, n);
        if (m->trace) {
            /* JANAS_TRACE=1: the state after each layer, last token of the
               block, to compare with another implementation */
            const float *v = m->x + (size_t)(n - 1) * dm;
            double nn = 0;
            for (uint32_t i = 0; i < dm; i++)
                nn += (double)v[i] * v[i];
            fprintf(stderr, "l_out-%-3u norm %10.4f  %9.4f %9.4f %9.4f %9.4f\n",
                    l, sqrt(nn), v[0], v[1], v[2], v[3]);
        }
    }
    m->ring_hi = pos0 + n; /* the rings hold up to here */
    double td = now();
    uint32_t first = all_logits ? 0 : n - 1;
    /* with an MTP block, every token's final hidden state is kept for it */
    int keep = m->mtp || m->asst;
    janas_m_norm_quant(m, f32(m, m->out_norm), keep ? 0 : first, n);
    if (keep)
        memcpy(m->hid, m->xn, (size_t)n * dm * sizeof(float));
    struct janas_matvec_task to = {.type = (int)m->output->type,
                                   .w = data(m, m->output),
                                   .x = m->xq + first * nbd,
                                   .y = logits,
                                   .rows = m->n_vocab,
                                   .cols = dm,
                                   .n_vec = n - first};
    if (logits) /* without: the hidden states alone, an embedding */
        janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                               m->compute, &to, 1);
    if (logits && m->softcap > 0.0f) {
        struct tok_job sc = {.m = m, .n = n - first, .out = logits};
        janas_pool_run(m->compute, softcap_worker, &sc);
    }
    m->phase[JANAS_PH_OUTPUT] += now() - td;
    if (m->a->rec && n <= JANAS_LLM_MAX_BLOCK) {
        m->base = m->src;
        m->base_pos = pos0;
        m->log_n = n;
    } else if (m->a->rec) {
        /* not logged: the state stands at the end of the call, in the
           buffer the call wrote, and nothing before it can be replayed */
        m->base = 1 - m->src;
        m->base_pos = pos0 + n;
        m->log_n = 0;
    }
    return 0;
}

uint32_t janas_llm_model_max_block(const struct janas_llm_model *m)
{
    return m->blk;
}

int janas_llm_model_has_mtp(const struct janas_llm_model *m)
{
    return m->mtp || m->asst;
}

int janas_llm_mtp_chain_fixed(const struct janas_llm_model *m)
{
    return m->asst != NULL;
}

int janas_llm_model_recurrent(const struct janas_llm_model *m)
{
    return m->a->rec != JANAS_REC_NONE;
}

const float *janas_llm_model_hidden(const struct janas_llm_model *m)
{
    return m->hid;
}

const float *janas_llm_model_normed(const struct janas_llm_model *m)
{
    return m->xn;
}

const float *janas_llm_mtp_hidden(const struct janas_llm_model *m)
{
    return m->mtp_hid;
}

#if defined(__x86_64__)
/* Sum of exp(l[i] - mx) over whole groups of 8; *done: how many summed. */
__attribute__((target("avx2,fma"))) static double
exp_sum_avx2(const float *l, uint32_t n, float mx, uint32_t *done)
{
    __m256 acc = _mm256_setzero_ps(), vm = _mm256_set1_ps(mx);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_add_ps(
            acc, janas_exp8(_mm256_sub_ps(_mm256_loadu_ps(l + i), vm)));
    *done = i;
    return janas_hsum8(acc);
}
#endif

double janas_m_exp_sum(const float *l, uint32_t n, float mx)
{
    double sum = 0;
    uint32_t i = 0;
#if defined(__x86_64__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        sum = exp_sum_avx2(l, n, mx, &i);
#endif
    for (; i < n; i++)
        sum += expf(l[i] - mx);
    return sum;
}

void janas_llm_mtp_note_tokens(struct janas_llm_model *m, const int32_t *tokens,
                               uint32_t n)
{
    if (!m->dh_w)
        return;
    for (uint32_t j = 0; j < n; j++) {
        int32_t t = tokens[j];
        if (t < 0 || (uint32_t)t >= m->n_vocab || m->dh_in[t] ||
            m->dh_n == m->dh_cap)
            continue;
        memcpy(m->dh_w + (size_t)m->dh_n * m->dh_row,
               m->dh_src + (size_t)t * m->dh_row, m->dh_row);
        m->dh_ids[m->dh_n++] = t;
        m->dh_in[t] = 1;
    }
}

int janas_llm_mtp_draft(struct janas_llm_model *m, const int32_t *tokens,
                        const float *hidden, uint32_t n, uint32_t pos0,
                        int32_t *token, float *conf)
{
    if (m->asst) { /* Gemma 4's assistant: the last row alone */
        uint32_t dm = m->d_model;
        if (n < 1 || n > m->blk)
            return -1;
        return janas_m_assist_draft(
            m, tokens[n - 1], hidden + (size_t)(n - 1) * dm, pos0 + n - 1,
            m->mtp_hid + (size_t)(n - 1) * dm, token, conf);
    }
    if (janas_llm_mtp_forward(m, tokens, hidden, n, pos0, NULL, 0) != 0)
        return -1;
    janas_m_apply_candidate(m,
                            janas_tuner_best(&m->tuner, 0)); /* one-row head */
    double td = now();
    size_t nbd = m->d_model / JANAS_QK;
    struct janas_matvec_task t = {.type = (int)m->output->type,
                                  .w = m->dh_w,
                                  .x = m->xq + (n - 1) * nbd,
                                  .y = m->dh_logits,
                                  .rows = m->dh_n,
                                  .cols = m->d_model};
    janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                           m->compute, &t, 1);
    /* the best row and its softmax probability within the head */
    float *l = m->dh_logits, mx = l[0];
    uint32_t best = 0;
    for (uint32_t i = 1; i < m->dh_n; i++)
        if (l[i] > mx) {
            mx = l[i];
            best = i;
        }
    double sum = 0;
    uint32_t i = 0;
#if defined(__x86_64__)
    if (m->avx2)
        sum = exp_sum_avx2(l, m->dh_n, mx, &i);
#endif
    for (; i < m->dh_n; i++)
        sum += expf(l[i] - mx);
    *token = m->dh_ids[best];
    *conf = (float)(1.0 / sum);
    m->phase[JANAS_PH_OUTPUT] += now() - td;
    return 0;
}

int janas_llm_mtp_forward(struct janas_llm_model *m, const int32_t *tokens,
                          const float *hidden, uint32_t n, uint32_t pos0,
                          float *logits, int all_logits)
{
    if (m->asst) /* no keys and values of its own: nothing to fill */
        return logits || n < 1 || n > m->blk ? -1 : 0;
    if (!m->mtp || n < 1 || n > m->blk || pos0 + n > m->n_ctx)
        return -1;
    for (uint32_t j = 0; j < n; j++)
        if (tokens[j] < 0 || (uint32_t)tokens[j] >= m->n_vocab)
            return -1;
    janas_m_apply_candidate(m,
                            janas_tuner_best(&m->tuner, janas_tune_class(n)));
    uint32_t dm = m->d_model;
    size_t nbd = dm / JANAS_QK, nbc = 2 * dm / JANAS_QK;
    const struct layer *ly = &m->mtp_layer;
    double ta = now();
    /*
     * [norm(embedding); norm(hidden)], projected back to d_model. Settled on
     * data (tests/mtp_eval.c, 398 positions of C source): this order, with
     * the hidden state after the main model's final norm, matches the main
     * model's next choice 88.9% of the times; the state before the norm
     * 86.4%; the other order 0%.
     */
    for (uint32_t j = 0; j < n; j++) {
        float *c = m->cat + (size_t)j * 2 * dm;
        embed(m, tokens[j], m->x + (size_t)j * dm);
        rms_norm(c, m->x + (size_t)j * dm, f32(m, m->enorm), dm, m->eps);
        rms_norm(c + dm, hidden + (size_t)j * dm, f32(m, m->hnorm), dm, m->eps);
        janas_q8k_quantize(c, m->catq + j * nbc, 2 * dm);
    }
    struct janas_matvec_task tp = {.type = (int)m->eh_proj->type,
                                   .w = data(m, m->eh_proj),
                                   .x = m->catq,
                                   .y = m->x,
                                   .rows = dm,
                                   .cols = 2 * dm,
                                   .n_vec = n};
    janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                           m->compute, &tp, 1);
    m->phase[JANAS_PH_QKV] += now() - ta;
    set_rope(m, n, pos0);
    for (uint32_t j = 0; j < n; j++) {
        rms_norm(m->xn + (size_t)j * dm, m->x + (size_t)j * dm,
                 f32(m, ly->attn_norm), dm, m->eps);
        janas_q8k_quantize(m->xn + (size_t)j * dm, m->xq + j * nbd, dm);
    }
    janas_m_attn_layer(m, ly, n, pos0);
    for (uint32_t j = 0; j < n; j++) {
        rms_norm(m->xn + (size_t)j * dm, m->x + (size_t)j * dm,
                 f32(m, ly->ffn_norm), dm, m->eps);
        janas_q8k_quantize(m->xn + (size_t)j * dm, m->xq + j * nbd, dm);
    }
    if (janas_m_moe_block(m, ly, m->mtp_cache, m->mtp_jl, m->mtp_cl, n,
                          MOE_ALL) != 0)
        return -1;
    double td = now();
    uint32_t first = all_logits ? 0 : n - 1;
    for (uint32_t j = 0; j < n; j++) {
        rms_norm(m->xn + (size_t)j * dm, m->x + (size_t)j * dm,
                 f32(m, m->mtp_norm), dm, m->eps);
        janas_q8k_quantize(m->xn + (size_t)j * dm, m->xq + j * nbd, dm);
    }
    memcpy(m->mtp_hid, m->xn, (size_t)n * dm * sizeof(float));
    if (!logits) { /* only the KV cache and the hidden states wanted */
        m->phase[JANAS_PH_OUTPUT] += now() - td;
        return 0;
    }
    struct janas_matvec_task to = {.type = (int)m->output->type,
                                   .w = data(m, m->output),
                                   .x = m->xq + first * nbd,
                                   .y = logits,
                                   .rows = m->n_vocab,
                                   .cols = dm,
                                   .n_vec = n - first};
    janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                           m->compute, &to, 1);
    m->phase[JANAS_PH_OUTPUT] += now() - td;
    return 0;
}

int janas_llm_model_decode(struct janas_llm_model *m, int32_t token,
                           uint32_t pos, float *logits)
{
    return janas_llm_model_forward(m, &token, 1, pos, logits, 0);
}
