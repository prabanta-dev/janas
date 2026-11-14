/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_mamba.c - the Mamba-2 layers of Nemotron-H (model_nh.c), as
 * llama.cpp's build_mamba2_layer computes them.
 *
 * ssm_in brings the normalized input to z (d_inner), xBC (the conv's
 * channels: x, d_inner, then B and C, n_group x d_state each) and dt (a
 * number a head), each times the matrix's own scale. A causal conv of
 * d_conv taps with a bias, then SiLU, runs over xBC. Head h (of n_vh, each
 * ssm_hd numbers of x, in group g = h / (n_vh / n_group)) keeps a state of
 * ssm_hd rows of d_state:
 *
 *   dt' = softplus(dt + dt_bias),  dA = exp(dt' A)
 *   s[r][i] = s[r][i] dA + B[i] x[r] dt',  y[r] = sum_i s[r][i] C[i]
 *
 * then y + D x, times SiLU(z), an RMS norm (with weights) over each group
 * of d_inner / n_group numbers, and ssm_out, times its scale.
 *
 * The state is in float, in ssm_buf (two uint16 units a number), with the
 * two buffers, the replay of logged tokens and the log of DeltaNet's
 * layers (model_rec.c): log_mixed holds xBC before the conv, log_beta dt'
 * and log_decay dA.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matvec.h"
#include "vmath.h"

#include "model_impl.h"

struct mb_job {
    struct janas_llm_model *m;
    const struct layer *ly;
    uint32_t n;
};

/* ggml's softplus: x past 20 as it is */
static inline float softplus(float x)
{
    return x > 20.0f ? x : log1pf(expf(x));
}

/* 1. The conv over xBC, item = 8 channels across all the rows (the
      replayed ones, then the call's), as DeltaNet's: the base buffer moves
      past the replayed rows, the call's state goes to the other. */
static void conv_worker(void *arg, int tid, int n_threads)
{
    struct mb_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t cc = m->conv_ch, dc = m->d_conv, rp = m->cur->replay;
    size_t sz = (size_t)m->n_rec * (dc - 1) * cc;
    float *src = m->cur->conv_buf +
                 (size_t)(m->rec_cont ? 1 - m->cur->src : m->cur->src) * sz +
                 (size_t)a->ly->slot * (dc - 1) * cc;
    float *dst = m->cur->conv_buf + (size_t)(1 - m->cur->src) * sz +
                 (size_t)a->ly->slot * (dc - 1) * cc;
    const float *wt = a->ly->conv_t, *bias = a->ly->conv_b;
    for (uint32_t c0 = (uint32_t)tid * 8; c0 < cc;
         c0 += (uint32_t)n_threads * 8)
        for (uint32_t c = c0; c < c0 + 8 && c < cc; c++) {
            float win[MAX_CONV];
            for (uint32_t i = 0; i + 1 < dc; i++)
                win[i] = src[(size_t)i * cc + c];
            for (uint32_t t = 0; t < rp + a->n; t++) {
                if (t == rp && rp > 0)
                    for (uint32_t i = 0; i + 1 < dc; i++)
                        src[(size_t)i * cc + c] = win[i];
                win[dc - 1] = m->mixed[(size_t)t * cc + c];
                float s = 0.0f;
                for (uint32_t i = 0; i < dc; i++)
                    s += win[i] * wt[(size_t)i * cc + c];
                s += bias[c];
                m->conv_out[(size_t)t * cc + c] = s / (1.0f + expf(-s));
                for (uint32_t i = 0; i + 1 < dc; i++)
                    win[i] = win[i + 1];
            }
            for (uint32_t i = 0; i + 1 < dc; i++)
                dst[(size_t)i * cc + c] = win[i];
        }
}

/* 2. The scan, item = a head: its state rows stay in one core's cache for
      all the rows of the call. y goes to att (with D x added). */
static void scan_worker(void *arg, int tid, int n_threads)
{
    struct mb_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t nh = m->n_vh, hd = m->ssm_hd, ds = m->ds, cc = m->conv_ch;
    uint32_t hpg = nh / m->n_kh, di = m->d_inner;
    size_t per = (size_t)nh * hd * ds; /* floats a layer */
    size_t sz = (size_t)m->n_rec * per;
    float *base = (float *)(void *)m->cur->ssm_buf;
    const float *dd = a->ly->ssm_dd;
    for (uint32_t h = (uint32_t)tid; h < nh; h += (uint32_t)n_threads) {
        uint32_t g = h / hpg;
        size_t off = (size_t)a->ly->slot * per + (size_t)h * hd * ds;
        float *M = base +
                   (size_t)(m->rec_cont ? 1 - m->cur->src : m->cur->src) * sz +
                   off;
        float *D = base + (size_t)(1 - m->cur->src) * sz + off;
        for (uint32_t t = 0; t < m->cur->replay + a->n; t++) {
            if (t == m->cur->replay) {
                if (M != D)
                    memcpy(D, M, (size_t)hd * ds * sizeof(float));
                M = D;
            }
            const float *co = m->conv_out + (size_t)t * cc;
            const float *x = co + (size_t)h * hd;
            const float *B = co + di + (size_t)g * ds;
            const float *C = co + di + (size_t)m->n_kh * ds + (size_t)g * ds;
            float dt = m->beta[(size_t)t * nh + h];
            float dA = m->decay[(size_t)t * nh + h];
            float *y = t < m->cur->replay
                           ? NULL
                           : m->att + (size_t)(t - m->cur->replay) * di +
                                 (size_t)h * hd;
            for (uint32_t r = 0; r < hd; r++) {
                float *s = M + (size_t)r * ds;
                float xdt = x[r] * dt, sum = 0.0f;
                for (uint32_t i = 0; i < ds; i++) {
                    float st = s[i] * dA + B[i] * xdt;
                    sum += st * C[i];
                    s[i] = st;
                }
                if (y)
                    y[r] = sum + x[r] * dd[h];
            }
        }
    }
}

/* 3. y times SiLU(z), then the RMS norm of each group, item = (token,
      group). */
static void gate_worker(void *arg, int tid, int n_threads)
{
    struct mb_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t ng = m->n_kh, gw = m->d_inner / ng;
    const float *w = f32(m, a->ly->ssm_norm);
    for (uint32_t it = (uint32_t)tid; it < a->n * ng;
         it += (uint32_t)n_threads) {
        uint32_t j = it / ng, g = it % ng;
        float *o = m->att + (size_t)j * m->d_inner + (size_t)g * gw;
        const float *z = m->big + (size_t)j * m->qkvz_dim + (size_t)g * gw;
        for (uint32_t i = 0; i < gw; i++)
            o[i] *= z[i] / (1.0f + expf(-z[i]));
        rms_norm(o, o, w + (size_t)g * gw, gw, m->eps);
    }
}

/* The middle of a layer, on the block's rows 0 .. n - 1 of big (its
   ssm_in) and att (its y) and on m->cur's state: the replayed rows and the
   call's, the conv, the scan, the gate, the log. */
static void mamba_core(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t n, double ta)
{
    uint32_t nh = m->n_vh, cc = m->conv_ch;
    uint32_t di = m->d_inner, qz = m->qkvz_dim;
    uint32_t rp = m->cur->replay;
    size_t L = JANAS_LLM_MAX_BLOCK;
    float *lm = m->cur->log_mixed + (size_t)ly->slot * L * cc;
    float *lb = m->cur->log_beta + (size_t)ly->slot * L * nh;
    float *ld = m->cur->log_decay + (size_t)ly->slot * L * nh;
    /* rows 0 .. rp - 1: the logged tokens to replay, then this call's */
    memcpy(m->mixed, lm, (size_t)rp * cc * sizeof(float));
    memcpy(m->beta, lb, (size_t)rp * nh * sizeof(float));
    memcpy(m->decay, ld, (size_t)rp * nh * sizeof(float));
    const float *dtb = f32(m, ly->dt_bias), *A = f32(m, ly->ssm_a);
    for (uint32_t j = 0; j < n; j++) {
        const float *src = m->big + (size_t)j * qz;
        memcpy(m->mixed + (size_t)(rp + j) * cc, src + di, cc * sizeof(float));
        for (uint32_t h = 0; h < nh; h++) {
            float dt = softplus(src[di + cc + h] + dtb[h]);
            m->beta[(size_t)(rp + j) * nh + h] = dt;
            m->decay[(size_t)(rp + j) * nh + h] = expf(dt * A[h]);
        }
    }
    double tb = now();
    m->phase[JANAS_PH_QKV] += tb - ta;
    trace_row(m, "xBC", m->mixed + (size_t)(rp + n - 1) * cc, cc);
    struct mb_job mj = {m, ly, n};
    janas_pool_run(m->compute, conv_worker, &mj);
    trace_row(m, "conv_out", m->conv_out + (size_t)(rp + n - 1) * cc, cc);
    janas_pool_run(m->compute, scan_worker, &mj);
    trace_row(m, "ssm_y", m->att + (size_t)(n - 1) * di, di);
    janas_pool_run(m->compute, gate_worker, &mj);
    trace_row(m, "ssm_gated", m->att + (size_t)(n - 1) * di, di);
    /* logged for a call that may be taken back; a longer one is not */
    if (n <= JANAS_LLM_MAX_BLOCK) {
        memcpy(lm, m->mixed + (size_t)rp * cc, (size_t)n * cc * sizeof(float));
        memcpy(lb, m->beta + (size_t)rp * nh, (size_t)n * nh * sizeof(float));
        memcpy(ld, m->decay + (size_t)rp * nh, (size_t)n * nh * sizeof(float));
    }
    m->phase[JANAS_PH_SSM] += now() - tb;
}

void janas_m_mamba_layer(struct janas_llm_model *m, const struct layer *ly,
                         uint32_t n)
{
    uint32_t dm = m->d_model;
    uint32_t di = m->d_inner, qz = m->qkvz_dim;
    size_t nbi = di / JANAS_QK;
    double ta = now();
    struct janas_matvec_task t = {.type = (int)ly->ssm_in->type,
                                  .w = data(m, ly->ssm_in),
                                  .x = m->xq,
                                  .y = m->big,
                                  .rows = qz,
                                  .cols = dm,
                                  .n_vec = n};
    janas_m_products(m, janas_m_gpu(m, 0), &t, 1);
    if (ly->in_scale != 1.0f)
        for (size_t i = 0; i < (size_t)n * qz; i++)
            m->big[i] *= ly->in_scale;
    if (janas_m_on_cur(m)) {
        mamba_core(m, ly, n, ta);
    } else {
        /* several pieces (janas_llm_model_forward_parts): each on its own
           sequence's state, the core shown its rows alone */
        float *big = m->big, *att = m->att;
        struct janas_seq *cur = m->cur;
        for (uint32_t i = 0; i < m->n_parts; i++) {
            uint32_t r0 = m->part_row0[i];
            m->big = big + (size_t)r0 * qz;
            m->att = att + (size_t)r0 * di;
            m->cur = m->parts[i].seq;
            mamba_core(m, ly, m->parts[i].n, ta);
        }
        m->big = big;
        m->att = att;
        m->cur = cur;
    }
    double tc = now();
    for (uint32_t j = 0; j < n; j++)
        janas_q8k_quantize(m->att + (size_t)j * di, m->attq + j * nbi, di);
    struct janas_matvec_task to = {.type = (int)ly->ssm_out->type,
                                   .w = data(m, ly->ssm_out),
                                   .x = m->attq,
                                   .y = m->xn,
                                   .rows = dm,
                                   .cols = di,
                                   .n_vec = n};
    janas_m_products(m, janas_m_gpu(m, 0), &to, 1);
    float os = ly->out_scale2;
    for (size_t i = 0; i < (size_t)n * dm; i++)
        m->x[i] += m->xn[i] * os;
    trace_row(m, "residual", m->x + (size_t)(n - 1) * dm, dm);
    m->phase[JANAS_PH_WO] += now() - tc;
}
