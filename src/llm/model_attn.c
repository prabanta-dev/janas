/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_attn.c - the attention layers of the forward pass: norms, q/k/v,
 * RoPE, the KV cache, the output projection (see model.h).
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

/* NeoX rotation of the first n dimensions of a head, pairs (i, i + n/2),
   with per-token tables; the other dimensions are left as they are. */
static void rope_neox(float *v, uint32_t n, const float *cs, const float *sn)
{
    uint32_t half = n / 2;
    for (uint32_t i = 0; i < half; i++) {
        float a = v[i], b = v[i + half];
        v[i] = a * cs[i] - b * sn[i];
        v[i + half] = a * sn[i] + b * cs[i];
    }
}

/* rms norm of x into xn, then xn to Q8_K in xq, tokens first .. n - 1 */
static void norm_quant_tok(struct janas_llm_model *m, const float *w,
                           uint32_t j)
{
    uint32_t dm = m->d_model;
    rms_norm(m->xn + (size_t)j * dm, m->x + (size_t)j * dm, w, dm, m->eps);
    janas_q8k_quantize(m->xn + (size_t)j * dm, m->xq + j * (dm / JANAS_QK), dm);
}

static void norm_quant_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    for (uint32_t j = t->first + (uint32_t)tid; j < t->n;
         j += (uint32_t)n_threads)
        norm_quant_tok(t->m, t->w, j);
}

void janas_m_norm_quant(struct janas_llm_model *m, const float *w,
                        uint32_t first, uint32_t n)
{
    if (n - first < 2) {
        for (uint32_t j = first; j < n; j++)
            norm_quant_tok(m, w, j);
        return;
    }
    struct tok_job t = {.m = m, .w = w, .first = first, .n = n};
    janas_pool_run(m->compute, norm_quant_worker, &t);
}

/*
 * After the q, k, v products, item (token j, head h): the heads of q are
 * normalized and turned (qwen3next: split from their output gate first),
 * those of k normalized, turned and put in the cache at eight bits with v.
 */
static void qkv_item(const struct tok_job *t, uint32_t j, uint32_t h)
{
    struct janas_llm_model *m = t->m;
    const struct layer *ly = t->ly;
    uint32_t hd = ly->hd, half = ly->n_rot / 2;
    uint32_t qd = m->n_head * hd, kvd = ly->n_kv * hd;
    const float *cs =
        (ly->swa ? m->rope_cos_swa : m->rope_cos) + (size_t)j * half;
    const float *sn =
        (ly->swa ? m->rope_sin_swa : m->rope_sin) + (size_t)j * half;
    if (h < m->n_head) {
        float *v = m->q + (size_t)j * qd + (size_t)h * hd;
        if (m->a->attn_gate) {
            const float *src = m->big + (size_t)j * 2 * qd + 2 * h * hd;
            memcpy(v, src, hd * sizeof(float));
            memcpy(m->qgate + (size_t)j * qd + (size_t)h * hd, src + hd,
                   hd * sizeof(float));
        }
        rms_norm(v, v, f32(m, ly->q_norm), hd, m->eps);
        rope_neox(v, ly->n_rot, cs, sn);
        return;
    }
    h -= m->n_head;
    float *kv = m->kk + (size_t)j * kvd + (size_t)h * hd;
    rms_norm(kv, kv, f32(m, ly->k_norm), hd, m->eps);
    rope_neox(kv, ly->n_rot, cs, sn);
    size_t at = ((size_t)h * m->n_ctx + t->pos0 + j) * hd;
    size_t as = (size_t)h * m->n_ctx + t->pos0 + j;
    float *vr = m->vv + (size_t)j * kvd + (size_t)h * hd;
    if (m->a->swa) { /* Gemma 4: V normalized, without weights */
        double ss = 0.0;
        for (uint32_t i = 0; i < hd; i++)
            ss += (double)vr[i] * vr[i];
        float sc = 1.0f / sqrtf((float)(ss / hd) + m->eps);
        for (uint32_t i = 0; i < hd; i++)
            vr[i] *= sc;
    }
    /* eight bits with a scale of their own: the largest of the head's
       numbers becomes 127, so the error is under half a step of a range
       the numbers themselves set */
    float mk = 0.0f, mv = 0.0f;
    for (uint32_t i = 0; i < hd; i++) {
        float a = fabsf(kv[i]), b = fabsf(vr[i]);
        if (a > mk)
            mk = a;
        if (b > mv)
            mv = b;
    }
    t->kc_scale[as] = mk / 127.0f;
    t->vc_scale[as] = mv / 127.0f;
    float ik = mk > 0 ? 127.0f / mk : 0.0f;
    float iv = mv > 0 ? 127.0f / mv : 0.0f;
    for (uint32_t i = 0; i < hd; i++) {
        t->kc[at + i] = (int8_t)lrintf(kv[i] * ik);
        t->vc[at + i] = (int8_t)lrintf(vr[i] * iv);
    }
}

static void qkv_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    uint32_t nh = t->m->n_head + t->ly->n_kv;
    for (uint32_t i = (uint32_t)tid; i < t->n * nh; i += (uint32_t)n_threads)
        qkv_item(t, i / nh, i % nh);
}

/* the attention output of token j gated (qwen3next) and put in Q8_K */
static void att_out_tok(struct janas_llm_model *m, const struct layer *ly,
                        uint32_t j)
{
    uint32_t qd = m->n_head * ly->hd;
    float *a = m->att + (size_t)j * qd;
    if (m->a->attn_gate) {
        const float *g = m->qgate + (size_t)j * qd;
        for (uint32_t i = 0; i < qd; i++)
            a[i] *= sigmoid(g[i]);
    }
    janas_q8k_quantize(a, m->attq + j * (qd / JANAS_QK), qd);
}

static void att_out_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads)
        att_out_tok(t->m, t->ly, j);
}

/* xn = rms_norm(xn) * w in place, token by token (sandwich norms) */
static void post_norm_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    uint32_t dm = t->m->d_model;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads) {
        float *v = t->m->xn + (size_t)j * dm;
        rms_norm(v, v, t->w, dm, t->m->eps);
    }
}

void janas_m_post_norm(struct janas_llm_model *m, const float *w, uint32_t n)
{
    struct tok_job t = {.m = m, .w = w, .n = n};
    if (n < 2)
        post_norm_worker(&t, 0, 1);
    else
        janas_pool_run(m->compute, post_norm_worker, &t);
}

/* x += xn, token by token */
static void residual_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    uint32_t dm = t->m->d_model;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads) {
        float *x = t->m->x + (size_t)j * dm;
        const float *y = t->m->xn + (size_t)j * dm;
        for (uint32_t i = 0; i < dm; i++)
            x[i] += y[i];
    }
}

void janas_m_residual(struct janas_llm_model *m, uint32_t n)
{
    if (n < 2) {
        for (uint32_t i = 0; i < n * m->d_model; i++)
            m->x[i] += m->xn[i];
        return;
    }
    struct tok_job t = {.m = m, .n = n};
    janas_pool_run(m->compute, residual_worker, &t);
}

/* Attention layer on the normalized block in xn / xq; output added to x. */
void janas_m_attn_layer(struct janas_llm_model *m, const struct layer *ly,
                        uint32_t n, uint32_t pos0)
{
    uint32_t dm = m->d_model, hd = ly->hd;
    uint32_t qd = m->n_head * hd, kvd = ly->n_kv * hd;
    double ta = now();
    /* qwen3next: each query head is followed by its output gate */
    float *qout = m->a->attn_gate ? m->big : m->q;
    struct janas_matvec_task t[3] = {{.type = (int)ly->wq->type,
                                      .w = data(m, ly->wq),
                                      .x = m->xq,
                                      .y = qout,
                                      .rows = m->a->attn_gate ? 2 * qd : qd,
                                      .cols = dm,
                                      .n_vec = n},
                                     {.type = (int)ly->wk->type,
                                      .w = data(m, ly->wk),
                                      .x = m->xq,
                                      .y = m->kk,
                                      .rows = kvd,
                                      .cols = dm,
                                      .n_vec = n},
                                     {.type = ly->wv ? (int)ly->wv->type : 0,
                                      .w = ly->wv ? data(m, ly->wv) : NULL,
                                      .x = m->xq,
                                      .y = m->vv,
                                      .rows = kvd,
                                      .cols = dm,
                                      .n_vec = n}};
    janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                           m->compute, t, ly->wv ? 3 : 2);
    if (!ly->wv) /* Gemma 4's full layers: V is K, before K's norm */
        memcpy(m->vv, m->kk, (size_t)n * kvd * sizeof(float));
    int8_t *kc = m->kcache + m->kv_off[ly->slot];
    int8_t *vc = m->vcache + m->kv_off[ly->slot];
    float *kcs = m->kscale + m->ks_off[ly->slot];
    float *vcs = m->vscale + m->ks_off[ly->slot];
    struct tok_job tj = {.m = m,
                         .ly = ly,
                         .n = n,
                         .pos0 = pos0,
                         .kc_scale = kcs,
                         .vc_scale = vcs,
                         .kc = kc,
                         .vc = vc};
    janas_pool_run(m->compute, qkv_worker, &tj);
    /* the last token's last head of each, as llama.cpp's graph names them */
    trace_row(m, "Qcur_pos", m->q + (size_t)(n - 1) * qd + (m->n_head - 1) * hd,
              hd);
    trace_row(m, "Kcur_pos",
              m->kk + (size_t)(n - 1) * kvd + (ly->n_kv - 1) * hd, hd);
    trace_row(m, "Vcur", m->vv + (size_t)(n - 1) * kvd + (ly->n_kv - 1) * hd,
              hd);
    double tb = now();
    m->phase[JANAS_PH_QKV] += tb - ta;
    /* attention takes JANAS_LLM_MAX_BLOCK tokens a call; the keys and values
       of the whole block are in the cache already, and each piece sees the
       positions up to its own end, as the block would */
    for (uint32_t s = 0; s < n; s += JANAS_LLM_MAX_BLOCK) {
        uint32_t len =
            n - s < JANAS_LLM_MAX_BLOCK ? n - s : JANAS_LLM_MAX_BLOCK;
        janas_attn_run(ly->swa ? m->attn_swa : m->attn, m->compute,
                       m->q + (size_t)s * qd, kc, kcs, vc, vcs, len, pos0 + s,
                       m->att + (size_t)s * qd);
    }
    double tc = now();
    m->phase[JANAS_PH_ATTN] += tc - tb;
    trace_row(m, "kqv_out", m->att + (size_t)(n - 1) * qd, qd);
    /* the output gate (qwen3next) goes with the quantization now */
    if (n < 2)
        for (uint32_t j = 0; j < n; j++)
            att_out_tok(m, ly, j);
    else
        janas_pool_run(m->compute, att_out_worker, &tj);
    struct janas_matvec_task to = {.type = (int)ly->wo->type,
                                   .w = data(m, ly->wo),
                                   .x = m->attq,
                                   .y = m->xn,
                                   .rows = dm,
                                   .cols = qd,
                                   .n_vec = n};
    janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                           m->compute, &to, 1);
    if (ly->post_attn)
        janas_m_post_norm(m, f32(m, ly->post_attn), n);
    janas_m_residual(m, n);
    m->phase[JANAS_PH_WO] += now() - tc;
}
