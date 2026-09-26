/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_ple.c - Gemma 4's per-layer embeddings (E2B, E4B).
 *
 * Every token brings, besides its embedding, a few numbers for each layer
 * (ple_dim, 256): a row of a table of their own (n_layer x ple_dim numbers
 * per token, as large as half of E2B's file) plus a projection of the
 * embedding, normalized layer by layer. After its feed-forward, each layer
 * opens a branch of the residual: gelu(gate x) times its own numbers,
 * projected back to the model's width, normalized, added.
 *
 * The table is read a row at a time from the file, when a token needs it:
 * a few kilobytes a token, never gigabytes in memory.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "matvec.h"
#include "vmath.h"

#include "model_impl.h"

/* f32 rounded to bf16 as ggml does it (to nearest, ties to even), kept as
   f32: the embedding meets the bf16 projection at bf16, as in llama.cpp */
static float round_bf16(float f)
{
    uint32_t u = janas_f2u(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) /* NaN */
        return f;
    u = (u + (0x7fffu + ((u >> 16) & 1u))) & 0xffff0000u;
    return janas_u2f(u);
}

struct bf16_job {
    const uint16_t *w;
    const float *x; /* n rows of cols, rounded to bf16 */
    float *y;       /* n rows of rows */
    uint32_t rows, cols, n;
    float scale;
};

#if defined(__x86_64__)
#include <immintrin.h>

__attribute__((target("avx2,fma"))) static float
dot_bf16_avx2(const uint16_t *w, const float *x, uint32_t n)
{
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(const void *)(w + i));
        __m256 wf = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
        acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + i), acc);
    }
    __m128 s =
        _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    float r = _mm_cvtss_f32(s);
    for (; i < n; i++)
        r += janas_u2f((uint32_t)w[i] << 16) * x[i];
    return r;
}
#endif

static float dot_bf16(const uint16_t *w, const float *x, uint32_t n)
{
#if defined(__x86_64__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return dot_bf16_avx2(w, x, n);
#endif
    float r = 0.0f;
    for (uint32_t i = 0; i < n; i++)
        r += janas_u2f((uint32_t)w[i] << 16) * x[i];
    return r;
}

/* rows split in equal runs, each row read once for all the tokens */
static void bf16_worker(void *arg, int tid, int n_threads)
{
    const struct bf16_job *b = arg;
    uint32_t per = (b->rows + (uint32_t)n_threads - 1) / (uint32_t)n_threads;
    uint32_t r0 = (uint32_t)tid * per;
    uint32_t r1 = r0 + per < b->rows ? r0 + per : b->rows;
    for (uint32_t r = r0; r < r1; r++) {
        const uint16_t *w = b->w + (size_t)r * b->cols;
        for (uint32_t j = 0; j < b->n; j++)
            b->y[(size_t)j * b->rows + r] =
                dot_bf16(w, b->x + (size_t)j * b->cols, b->cols) * b->scale;
    }
}

struct f32_job {
    const float *w, *x;
    float *y;
    uint32_t rows, cols, n;
};

static void f32_worker(void *arg, int tid, int n_threads)
{
    const struct f32_job *b = arg;
    size_t total = (size_t)b->rows * b->n;
    for (size_t i = (size_t)tid; i < total; i += (size_t)n_threads) {
        uint32_t j = (uint32_t)(i / b->rows), r = (uint32_t)(i % b->rows);
        const float *w = b->w + (size_t)r * b->cols;
        const float *x = b->x + (size_t)j * b->cols;
        float s = 0.0f;
        for (uint32_t c = 0; c < b->cols; c++)
            s += w[c] * x[c];
        b->y[(size_t)j * b->rows + r] = s;
    }
}

/* y (n rows of t's rows) = t x (n rows of cols): f32 weights as they are,
   quantized ones on x put in Q8_K (q, n * cols / 256 blocks) */
static void product(struct janas_llm_model *m, const struct janas_jns_tensor *t,
                    const float *x, struct janas_block_q8k *q, float *y,
                    uint32_t n)
{
    uint32_t cols = (uint32_t)t->dims[0], rows = (uint32_t)t->dims[1];
    if (t->type == 0) {
        struct f32_job b = {
            .w = f32(m, t), .x = x, .y = y, .rows = rows, .cols = cols, .n = n};
        janas_pool_run(m->compute, f32_worker, &b);
        return;
    }
    size_t nb = cols / JANAS_QK;
    for (uint32_t j = 0; j < n; j++)
        janas_q8k_quantize(x + (size_t)j * cols, q + j * nb, cols);
    struct janas_matvec_task tk = {.type = (int)t->type,
                                   .w = data(m, t),
                                   .x = q,
                                   .y = y,
                                   .rows = rows,
                                   .cols = cols,
                                   .n_vec = n};
    janas_gpu_matvec_group(m->gpu_off || !m->gpu_use ? NULL : m->gpu,
                           m->compute, &tk, 1);
}

/*
 * The block's per-layer inputs, from the embeddings in x (already scaled by
 * sqrt(d_model)): the projection over sqrt(d_model), each layer's part
 * normalized, plus the table's row times sqrt(ple_dim), all over sqrt(2).
 */
int janas_m_ple_inputs(struct janas_llm_model *m, const int32_t *tokens,
                       uint32_t n)
{
    uint32_t dm = m->d_model, P = m->ple_dim, W = m->n_layer * P;
    float *xb = m->xn; /* the embeddings at bf16 */
    for (size_t i = 0; i < (size_t)n * dm; i++)
        xb[i] = round_bf16(m->x[i]);
    struct bf16_job b = {.w = data(m, m->ple_model),
                         .x = xb,
                         .y = m->ple,
                         .rows = W,
                         .cols = dm,
                         .n = n,
                         .scale = 1.0f / sqrtf((float)dm)};
    if (n < 2)
        bf16_worker(&b, 0, 1);
    else
        janas_pool_run(m->compute, bf16_worker, &b);
    const float *nw = f32(m, m->ple_norm);
    size_t row_bytes =
        W / JANAS_QK * janas_qtype_block_size((int)m->ple_tok->type);
    uint8_t *raw = (uint8_t *)(m->ple_row + W); /* the row as read */
    float es = sqrtf((float)P), is2 = 1.0f / sqrtf(2.0f);
    for (uint32_t j = 0; j < n; j++) {
        off_t at =
            (off_t)(m->ple_tok->offset + (uint64_t)tokens[j] * row_bytes);
        for (size_t done = 0; done < row_bytes;) {
            ssize_t r =
                pread(m->j.fd, raw + done, row_bytes - done, at + (off_t)done);
            if (r <= 0)
                return -1;
            done += (size_t)r;
        }
        janas_dequantize((int)m->ple_tok->type, raw, m->ple_row, W);
        float *p = m->ple + (size_t)j * W;
        for (uint32_t l = 0; l < m->n_layer; l++) {
            float *pl = p + (size_t)l * P;
            const float *el = m->ple_row + (size_t)l * P;
            rms_norm(pl, pl, nw, P, m->eps);
            for (uint32_t i = 0; i < P; i++)
                pl[i] = (pl[i] + el[i] * es) * is2;
        }
    }
    trace_row(m, "inp_per_layer", m->ple + (size_t)(n - 1) * W, P);
    return 0;
}

/*
 * Layer l's branch, on the residual in x after the feed-forward, then the
 * layer's output scale: x = (x + norm(proj(gelu(gate x) * input))) * s.
 */
void janas_m_ple_layer(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t l, uint32_t n)
{
    uint32_t dm = m->d_model, P = m->ple_dim, W = m->n_layer * P;
    product(m, ly->ple_gate, m->x, m->xq, m->ple_g, n);
    for (uint32_t j = 0; j < n; j++) {
        float *g = m->ple_g + (size_t)j * P;
        const float *in = m->ple + (size_t)j * W + (size_t)l * P;
        for (uint32_t i = 0; i < P; i++)
            g[i] = janas_gelu_mul_det(g[i], in[i]);
    }
    product(m, ly->ple_proj, m->ple_g, m->ple_q, m->xn, n);
    const float *pw = f32(m, ly->ple_post);
    for (uint32_t j = 0; j < n; j++) {
        float *x = m->x + (size_t)j * dm, *y = m->xn + (size_t)j * dm;
        rms_norm(y, y, pw, dm, m->eps);
        for (uint32_t i = 0; i < dm; i++)
            x[i] = (x[i] + y[i]) * ly->out_scale;
    }
    trace_row(m, "per_layer_out", m->x + (size_t)(n - 1) * dm, dm);
}
