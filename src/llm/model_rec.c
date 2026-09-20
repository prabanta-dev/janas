/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_rec.c - the Gated DeltaNet layers of the forward pass: causal
 * convolution, the recurrence, the gated norm (see deltanet.h).
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

/*
 * Gated DeltaNet in four steps per layer, over the rows of a call: rows
 * 0 .. replay - 1 are logged tokens of the previous call replayed on the base
 * (see the recurrent state in struct janas_llm_model), the n rows after them
 * are the tokens of the call.
 *
 * 1. Causal convolution of the q|k|v channels, then SiLU; the q and k heads
 *    are then L2-normalized. Item = one head's worth of channels (ds), done
 *    8 channels at a time across all tokens.
 */
struct ssm_job {
    struct janas_llm_model *m;
    const struct layer *ly;
    uint32_t n;
};

static void conv_head_ref(struct janas_llm_model *m, const float *wt,
                          float *src, float *dst, uint32_t c0, uint32_t n)
{
    uint32_t cc = m->conv_ch, dc = m->d_conv, a = m->replay;
    for (uint32_t c = c0; c < c0 + m->ds; c++) {
        float win[MAX_CONV];
        for (uint32_t i = 0; i + 1 < dc; i++)
            win[i] = src[(size_t)i * cc + c];
        for (uint32_t t = 0; t < a + n; t++) {
            if (t == a && a > 0) /* the base moves past the replayed tokens */
                for (uint32_t i = 0; i + 1 < dc; i++)
                    src[(size_t)i * cc + c] = win[i];
            win[dc - 1] = m->mixed[(size_t)t * cc + c];
            float s = 0.0f;
            for (uint32_t i = 0; i < dc; i++)
                s += win[i] * wt[(size_t)i * cc + c];
            m->conv_out[(size_t)t * cc + c] = s / (1.0f + expf(-s));
            for (uint32_t i = 0; i + 1 < dc; i++)
                win[i] = win[i + 1];
        }
        for (uint32_t i = 0; i + 1 < dc; i++)
            dst[(size_t)i * cc + c] = win[i];
    }
}

#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) static void
conv_head_avx2(struct janas_llm_model *m, const float *wt, float *src,
               float *dst, uint32_t c0, uint32_t n)
{
    uint32_t cc = m->conv_ch, dc = m->d_conv, a = m->replay;
    for (uint32_t c = c0; c < c0 + m->ds; c += 8) {
        __m256 win[MAX_CONV], w[MAX_CONV];
        for (uint32_t i = 0; i < dc; i++)
            w[i] = _mm256_loadu_ps(wt + (size_t)i * cc + c);
        for (uint32_t i = 0; i + 1 < dc; i++)
            win[i] = _mm256_loadu_ps(src + (size_t)i * cc + c);
        for (uint32_t t = 0; t < a + n; t++) {
            if (t == a && a > 0)
                for (uint32_t i = 0; i + 1 < dc; i++)
                    _mm256_storeu_ps(src + (size_t)i * cc + c, win[i]);
            win[dc - 1] = _mm256_loadu_ps(m->mixed + (size_t)t * cc + c);
            __m256 s = _mm256_mul_ps(win[0], w[0]);
            for (uint32_t i = 1; i < dc; i++)
                s = _mm256_fmadd_ps(win[i], w[i], s);
            _mm256_storeu_ps(m->conv_out + (size_t)t * cc + c, janas_silu8(s));
            for (uint32_t i = 0; i + 1 < dc; i++)
                win[i] = win[i + 1];
        }
        for (uint32_t i = 0; i + 1 < dc; i++)
            _mm256_storeu_ps(dst + (size_t)i * cc + c, win[i]);
    }
}

/* x / sqrt(sum x^2 + eps), in place, n a multiple of 8. */
__attribute__((target("avx2,fma"))) static void
l2_norm_avx2(float *x, uint32_t n, float eps)
{
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m256 inv = _mm256_set1_ps(1.0f / sqrtf(janas_hsum8(acc) + eps));
    for (uint32_t i = 0; i < n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), inv));
}
#endif

static void l2_norm_ref(float *x, uint32_t n, float eps)
{
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++)
        ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss + eps);
    for (uint32_t i = 0; i < n; i++)
        x[i] *= inv;
}

static void conv_worker(void *arg, int tid, int n_threads)
{
    struct ssm_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t cc = m->conv_ch, ds = m->ds, kd = m->n_kh * ds;
    size_t sz = (size_t)m->n_rec * (m->d_conv - 1) * cc;
    float *src = m->conv_buf + (size_t)m->src * sz +
                 (size_t)a->ly->slot * (m->d_conv - 1) * cc;
    float *dst = m->conv_buf + (size_t)(1 - m->src) * sz +
                 (size_t)a->ly->slot * (m->d_conv - 1) * cc;
    for (uint32_t c0 = (uint32_t)tid * ds; c0 < cc;
         c0 += (uint32_t)n_threads * ds) {
#if defined(__x86_64__)
        if (m->avx2)
            conv_head_avx2(m, a->ly->conv_t, src, dst, c0, a->n);
        else
#endif
            conv_head_ref(m, a->ly->conv_t, src, dst, c0, a->n);
        if (c0 >= 2 * kd) /* a value head: not normalized */
            continue;
        for (uint32_t t = 0; t < m->replay + a->n; t++) {
            float *x = m->conv_out + (size_t)t * cc + c0;
#if defined(__x86_64__)
            if (m->avx2)
                l2_norm_avx2(x, ds, m->eps);
            else
#endif
                l2_norm_ref(x, ds, m->eps);
        }
    }
}

/*
 * 2. The recurrence, item = (value head, block of SSM_ROWS rows of its
 *    state): rows are independent, so a head is split among threads and each
 *    slice stays in one core's cache for all the tokens. Replayed tokens
 *    update the base in place; the call's tokens start from the base (or the
 *    tip) and leave their state in the other buffer. Raw outputs go to att.
 */
static void rec_worker(void *arg, int tid, int n_threads)
{
    struct ssm_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t ds = m->ds, nv = m->n_vh, rf = nv / m->n_kh, kd = m->n_kh * ds;
    uint32_t cc = m->conv_ch, blocks = ds / SSM_ROWS;
    size_t sz = (size_t)m->n_rec * nv * ds * ds;
    float scale = 1.0f / sqrtf((float)ds), junk[SSM_ROWS];
    for (uint32_t it = (uint32_t)tid; it < nv * blocks;
         it += (uint32_t)n_threads) {
        uint32_t h = it / blocks, r0 = it % blocks * SSM_ROWS;
        uint32_t g = m->vmap_mod ? h % m->n_kh : h / rf;
        size_t off = (((size_t)a->ly->slot * nv + h) * ds + r0) * ds;
        uint16_t *M = m->ssm_buf + (size_t)m->src * sz + off;
        uint16_t *D = m->ssm_buf + (size_t)(1 - m->src) * sz + off;
        for (uint32_t t = 0; t < m->replay + a->n; t++) {
            if (t == m->replay) {
                memcpy(D, M, (size_t)SSM_ROWS * ds * sizeof(uint16_t));
                M = D;
            }
            const float *co = m->conv_out + (size_t)t * cc;
            float *o = t < m->replay
                           ? junk
                           : m->att + (size_t)(t - m->replay) * m->d_inner +
                                 (size_t)h * ds + r0;
            janas_gdn_step_h(
                M, co + (size_t)g * ds, co + kd + (size_t)g * ds,
                co + 2 * kd + (size_t)h * ds + r0, m->decay[(size_t)t * nv + h],
                m->beta[(size_t)t * nv + h], scale, o, ds, SSM_ROWS);
        }
    }
}

#if defined(__x86_64__)
/* rms_norm(o) * w * SiLU(z), in place, n a multiple of 8. */
__attribute__((target("avx2,fma"))) static void
gate_avx2(float *o, const float *z, const float *w, uint32_t n, float eps)
{
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(o + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m256 sc = _mm256_set1_ps(1.0f / sqrtf(janas_hsum8(acc) / n + eps));
    for (uint32_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(o + i), sc),
                                 _mm256_loadu_ps(w + i));
        _mm256_storeu_ps(o + i,
                         _mm256_mul_ps(v, janas_silu8(_mm256_loadu_ps(z + i))));
    }
}
#endif

/* 3. RMS norm (ssm_norm) of each head's output, gated by SiLU(z): item =
      (token, value head). */
static void gate_worker(void *arg, int tid, int n_threads)
{
    struct ssm_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t ds = m->ds, nv = m->n_vh, rf = nv / m->n_kh;
    uint32_t gsz = 2 * ds + 2 * ds * rf;
    const float *w = f32(m, a->ly->ssm_norm);
    for (uint32_t it = (uint32_t)tid; it < a->n * nv;
         it += (uint32_t)n_threads) {
        uint32_t j = it / nv, h = it % nv, g = h / rf, r = h % rf;
        float *o = m->att + (size_t)j * m->d_inner + (size_t)h * ds;
        const float *z =
            m->a->rec == JANAS_REC_Q35
                ? m->big + (size_t)j * m->qkvz_dim + m->conv_ch + (size_t)h * ds
                : m->big + (size_t)j * m->qkvz_dim + (size_t)g * gsz + 2 * ds +
                      ds * rf + r * ds;
#if defined(__x86_64__)
        if (m->avx2) {
            gate_avx2(o, z, w, ds, m->eps);
            continue;
        }
#endif
        rms_norm(o, o, w, ds, m->eps);
        for (uint32_t i = 0; i < ds; i++)
            o[i] *= z[i] / (1.0f + expf(-z[i]));
    }
}

/* Gated DeltaNet layer on the normalized block in xn / xq; added to x. */
void janas_m_rec_layer(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t n)
{
    uint32_t dm = m->d_model, ds = m->ds, nv = m->n_vh, rf = nv / m->n_kh;
    uint32_t kd = m->n_kh * ds, cc = m->conv_ch, gsz = 2 * ds + 2 * ds * rf;
    size_t nbi = m->d_inner / JANAS_QK;
    double ta = now();
    if (m->a->rec == JANAS_REC_Q35) {
        /* q|k|v and z side by side in big (row stride qkvz_dim) */
        struct janas_matvec_task t[2] = {{.type = (int)ly->ssm_qkv->type,
                                          .w = data(m, ly->ssm_qkv),
                                          .x = m->xq,
                                          .y = m->big,
                                          .rows = cc,
                                          .cols = dm,
                                          .n_vec = n,
                                          .y_stride = m->qkvz_dim},
                                         {.type = (int)ly->ssm_z->type,
                                          .w = data(m, ly->ssm_z),
                                          .x = m->xq,
                                          .y = m->big + cc,
                                          .rows = m->d_inner,
                                          .cols = dm,
                                          .n_vec = n,
                                          .y_stride = m->qkvz_dim}};
        janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                               m->compute, t, 2);
        /* beta then alpha per head, as in the grouped layout's order */
        const float *wb = ly->ssm_beta, *wa = ly->ssm_alpha;
        for (uint32_t j = 0; j < n; j++)
            for (uint32_t h = 0; h < nv; h++) {
                const float *xj = m->xn + (size_t)j * dm;
                m->ba[(size_t)j * 2 * nv + h] =
                    janas_dot_f32(wb + (size_t)h * dm, xj, dm);
                m->ba[(size_t)j * 2 * nv + nv + h] =
                    janas_dot_f32(wa + (size_t)h * dm, xj, dm);
            }
    } else {
        struct janas_matvec_task t[2] = {{.type = (int)ly->ssm_in->type,
                                          .w = data(m, ly->ssm_in),
                                          .x = m->xq,
                                          .y = m->big,
                                          .rows = m->qkvz_dim,
                                          .cols = dm,
                                          .n_vec = n},
                                         {.type = (int)ly->ssm_ba->type,
                                          .w = data(m, ly->ssm_ba),
                                          .x = m->xq,
                                          .y = m->ba,
                                          .rows = 2 * nv,
                                          .cols = dm,
                                          .n_vec = n}};
        janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                               m->compute, t, 2);
    }
    /* per key head g the projection holds q, k, the values and the z gates
       of its rf value heads, then per g: rf betas and rf alphas */
    const float *dt = f32(m, ly->dt_bias), *A = f32(m, ly->ssm_a);
    uint32_t a = m->replay;
    size_t B = JANAS_LLM_MAX_BLOCK;
    float *lm = m->log_mixed + (size_t)ly->slot * B * cc;
    float *lb = m->log_beta + (size_t)ly->slot * B * nv;
    float *ld = m->log_decay + (size_t)ly->slot * B * nv;
    /* rows 0 .. a - 1: the logged tokens to replay, then this call's */
    memcpy(m->mixed, lm, (size_t)a * cc * sizeof(float));
    memcpy(m->beta, lb, (size_t)a * nv * sizeof(float));
    memcpy(m->decay, ld, (size_t)a * nv * sizeof(float));
    for (uint32_t j = 0; j < n; j++) {
        const float *src = m->big + (size_t)j * m->qkvz_dim;
        float *mx = m->mixed + (size_t)(a + j) * cc;
        const float *b = m->ba + (size_t)j * 2 * nv;
        if (m->a->rec == JANAS_REC_Q35) {
            /* already in the convolution's channel order */
            memcpy(mx, src, (size_t)cc * sizeof(float));
            for (uint32_t h = 0; h < nv; h++) {
                float al = b[nv + h] + dt[h];
                float sp = al > 20.0f ? al : log1pf(expf(al));
                m->beta[(size_t)(a + j) * nv + h] = sigmoid(b[h]);
                m->decay[(size_t)(a + j) * nv + h] = expf(sp * A[h]);
            }
            continue;
        }
        for (uint32_t g = 0; g < m->n_kh; g++) {
            const float *s = src + (size_t)g * gsz;
            memcpy(mx + (size_t)g * ds, s, ds * sizeof(float));
            memcpy(mx + kd + (size_t)g * ds, s + ds, ds * sizeof(float));
            memcpy(mx + 2 * kd + (size_t)g * ds * rf, s + 2 * ds,
                   ds * rf * sizeof(float));
            for (uint32_t r = 0; r < rf; r++) {
                uint32_t h = g * rf + r;
                float al = b[2 * g * rf + rf + r] + dt[h];
                float sp = al > 20.0f ? al : log1pf(expf(al));
                m->beta[(size_t)(a + j) * nv + h] = sigmoid(b[2 * g * rf + r]);
                m->decay[(size_t)(a + j) * nv + h] = expf(sp * A[h]);
            }
        }
    }
    double tb = now();
    m->phase[JANAS_PH_QKV] += tb - ta;
    trace_row(m, "qkv_mixed", m->mixed + (size_t)(a + n - 1) * cc, cc);
    trace_row(m, "z",
              m->big + (size_t)(n - 1) * m->qkvz_dim +
                  (m->a->rec == JANAS_REC_Q35 ? m->conv_ch : 0),
              m->d_inner);
    struct ssm_job sj = {m, ly, n};
    janas_pool_run(m->compute, conv_worker, &sj);
    trace_row(m, "conv_out", m->conv_out + (size_t)(m->replay + n - 1) * cc,
              cc);
    if (m->trace >= 3) {
        const float *co = m->conv_out + (size_t)(m->replay + n - 1) * cc;
        trace_row(m, "conv_q_h15", co + (size_t)15 * ds, ds);
        trace_row(m, "conv_k_h15", co + kd + (size_t)15 * ds, ds);
        trace_row(m, "conv_v_h31", co + 2 * kd + (size_t)31 * ds, ds);
        trace_row(m, "conv_v_h0", co + 2 * kd, ds);
    }
    janas_pool_run(m->compute, rec_worker, &sj);
    trace_row(m, "ssm_raw", m->att + (size_t)(n - 1) * m->d_inner, m->d_inner);
    if (m->trace >= 3) {
        const float *o = m->att + (size_t)(n - 1) * m->d_inner;
        trace_row(m, "ssm_raw_h31", o + (size_t)31 * ds, ds);
        trace_row(m, "ssm_raw_h0", o, ds);
        fprintf(stderr, "  beta h0 %.6f h31 %.6f | decay h0 %.6g h31 %.6g\n",
                m->beta[(size_t)(a + n - 1) * nv],
                m->beta[(size_t)(a + n - 1) * nv + 31],
                m->decay[(size_t)(a + n - 1) * nv],
                m->decay[(size_t)(a + n - 1) * nv + 31]);
    }
    janas_pool_run(m->compute, gate_worker, &sj);
    trace_row(m, "ssm_gated", m->att + (size_t)(n - 1) * m->d_inner,
              m->d_inner);
    if (m->trace >= 3) {
        trace_row(m, "gated_h31",
                  m->att + (size_t)(n - 1) * m->d_inner + (size_t)31 * ds, ds);
        trace_row(m, "z_h31",
                  m->big + (size_t)(n - 1) * m->qkvz_dim +
                      (m->a->rec == JANAS_REC_Q35 ? m->conv_ch : 0) +
                      (size_t)31 * ds,
                  ds);
    }
    /* logged for a call that may be taken back; a longer one (a prefill
       block) is not, and forward() leaves the state at its end */
    if (n <= JANAS_LLM_MAX_BLOCK) {
        memcpy(lm, m->mixed + (size_t)a * cc, (size_t)n * cc * sizeof(float));
        memcpy(lb, m->beta + (size_t)a * nv, (size_t)n * nv * sizeof(float));
        memcpy(ld, m->decay + (size_t)a * nv, (size_t)n * nv * sizeof(float));
    }
    double tc = now();
    m->phase[JANAS_PH_SSM] += tc - tb;
    for (uint32_t j = 0; j < n; j++)
        janas_q8k_quantize(m->att + (size_t)j * m->d_inner, m->attq + j * nbi,
                           m->d_inner);
    struct janas_matvec_task to = {.type = (int)ly->ssm_out->type,
                                   .w = data(m, ly->ssm_out),
                                   .x = m->attq,
                                   .y = m->xn,
                                   .rows = dm,
                                   .cols = m->d_inner,
                                   .n_vec = n};
    janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                           m->compute, &to, 1);
    trace_row(m, "ssm_out", m->xn + (size_t)(n - 1) * dm, dm);
    for (uint32_t i = 0; i < n * dm; i++)
        m->x[i] += m->xn[i];
    trace_row(m, "residual", m->x + (size_t)(n - 1) * dm, dm);
    m->phase[JANAS_PH_WO] += now() - tc;
}
