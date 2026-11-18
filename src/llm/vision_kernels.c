/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vision_kernels.c - what the vision encoders share (vision_kernels.h):
 * the products, the attention over every patch, the norms and activations;
 * Whisper's encoder (whisper.c) too.
 */
#include "vision_kernels.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "common/pool.h"
#include "quant.h"
#include "vmath.h"

/* ---- the products: C[j][i] = sum_k B[j][k] A[k][i] (+ bias[i]) ----

   B's rows are tokens or queries (float, k contiguous); A's rows run over
   the outputs (i contiguous): the weights transposed once at open
   (bfloat16), or keys transposed and values as they are (float). A tile of
   four rows of B and twenty-four outputs keeps twelve sums in registers
   over k, a weight read once for four tokens and a token's number once for
   twenty-four outputs; the threads take blocks of outputs, whose part of A
   stays in the cache while every token goes through it. Float throughout:
   the activations of a ViT have outliers that eight bits lose. The weights
   are bfloat16, or half precision where the model was trained in it
   (Whisper's are its originals). */

struct gemm {
    int kind;      /* A: 0 float, 1 bfloat16, 2 half */
    const void *a; /* rows of A (k), lda apart */
    size_t lda;
    const float *b; /* rows of B (j), ldb apart */
    size_t ldb;
    float *c; /* C[j * ldc + i] */
    size_t ldc;
    const float *bias;
    uint32_t ni, nj, k;
};

static float a_at(const struct gemm *g, uint32_t k, uint32_t i)
{
    if (!g->kind)
        return ((const float *)g->a)[(size_t)k * g->lda + i];
    if (g->kind == 2)
        return janas_fp16_to_fp32(
            ((const uint16_t *)g->a)[(size_t)k * g->lda + i]);
    uint32_t u = (uint32_t)((const uint16_t *)g->a)[(size_t)k * g->lda + i]
                 << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void gemm_plain(const struct gemm *g, uint32_t i0, uint32_t i1)
{
    for (uint32_t j = 0; j < g->nj; j++) {
        const float *bj = g->b + (size_t)j * g->ldb;
        float *cj = g->c + (size_t)j * g->ldc;
        for (uint32_t i = i0; i < i1; i++)
            cj[i] = g->bias ? g->bias[i] : 0;
        for (uint32_t k = 0; k < g->k; k++)
            for (uint32_t i = i0; i < i1; i++)
                cj[i] += bj[k] * a_at(g, k, i);
    }
}

#if defined(__x86_64__)
#define AVX2 __attribute__((target("avx2,fma,f16c")))

AVX2 static inline __m256 load_a(const struct gemm *g, uint32_t k, uint32_t i)
{
    if (!g->kind)
        return _mm256_loadu_ps((const float *)g->a + (size_t)k * g->lda + i);
    __m128i h = _mm_loadu_si128(
        (const __m128i *)((const uint16_t *)g->a + (size_t)k * g->lda + i));
    if (g->kind == 2)
        return _mm256_cvtph_ps(h);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

AVX2 static inline float hsum(__m256 v)
{
    __m128 s =
        _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* outputs i .. i + 8 * nv of rows j .. j + nb of B */
AVX2 static inline void tile(const struct gemm *g, uint32_t j, uint32_t nb,
                             uint32_t i, uint32_t nv)
{
    __m256 acc[4][3];
    for (uint32_t r = 0; r < 4; r++)
        for (uint32_t v = 0; v < 3; v++)
            acc[r][v] = g->bias && v < nv ? _mm256_loadu_ps(g->bias + i + 8 * v)
                                          : _mm256_setzero_ps();
    const float *b0 = g->b + (size_t)j * g->ldb;
    for (uint32_t k = 0; k < g->k; k++) {
        __m256 a[3];
        for (uint32_t v = 0; v < nv; v++)
            a[v] = load_a(g, k, i + 8 * v);
        for (uint32_t r = 0; r < nb; r++) {
            __m256 x = _mm256_set1_ps(b0[(size_t)r * g->ldb + k]);
            for (uint32_t v = 0; v < nv; v++)
                acc[r][v] = _mm256_fmadd_ps(x, a[v], acc[r][v]);
        }
    }
    for (uint32_t r = 0; r < nb; r++)
        for (uint32_t v = 0; v < nv; v++)
            _mm256_storeu_ps(g->c + (size_t)(j + r) * g->ldc + i + 8 * v,
                             acc[r][v]);
}

/* the whole tile, four rows of B and twenty-four outputs, its twelve sums
   named so they stay in registers */
AVX2 static inline void tile_full(const struct gemm *g, uint32_t j, uint32_t i)
{
    __m256 c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32;
    if (g->bias) {
        c00 = c10 = c20 = c30 = _mm256_loadu_ps(g->bias + i);
        c01 = c11 = c21 = c31 = _mm256_loadu_ps(g->bias + i + 8);
        c02 = c12 = c22 = c32 = _mm256_loadu_ps(g->bias + i + 16);
    } else {
        c00 = c01 = c02 = c10 = c11 = c12 = _mm256_setzero_ps();
        c20 = c21 = c22 = c30 = c31 = c32 = _mm256_setzero_ps();
    }
    const float *b0 = g->b + (size_t)j * g->ldb, *b1 = b0 + g->ldb;
    const float *b2 = b1 + g->ldb, *b3 = b2 + g->ldb;
    for (uint32_t k = 0; k < g->k; k++) {
        __m256 a0 = load_a(g, k, i), a1 = load_a(g, k, i + 8);
        __m256 a2 = load_a(g, k, i + 16);
        __m256 x = _mm256_broadcast_ss(b0 + k);
        c00 = _mm256_fmadd_ps(x, a0, c00);
        c01 = _mm256_fmadd_ps(x, a1, c01);
        c02 = _mm256_fmadd_ps(x, a2, c02);
        x = _mm256_broadcast_ss(b1 + k);
        c10 = _mm256_fmadd_ps(x, a0, c10);
        c11 = _mm256_fmadd_ps(x, a1, c11);
        c12 = _mm256_fmadd_ps(x, a2, c12);
        x = _mm256_broadcast_ss(b2 + k);
        c20 = _mm256_fmadd_ps(x, a0, c20);
        c21 = _mm256_fmadd_ps(x, a1, c21);
        c22 = _mm256_fmadd_ps(x, a2, c22);
        x = _mm256_broadcast_ss(b3 + k);
        c30 = _mm256_fmadd_ps(x, a0, c30);
        c31 = _mm256_fmadd_ps(x, a1, c31);
        c32 = _mm256_fmadd_ps(x, a2, c32);
    }
    float *o = g->c + (size_t)j * g->ldc + i;
    _mm256_storeu_ps(o, c00);
    _mm256_storeu_ps(o + 8, c01);
    _mm256_storeu_ps(o + 16, c02);
    o += g->ldc;
    _mm256_storeu_ps(o, c10);
    _mm256_storeu_ps(o + 8, c11);
    _mm256_storeu_ps(o + 16, c12);
    o += g->ldc;
    _mm256_storeu_ps(o, c20);
    _mm256_storeu_ps(o + 8, c21);
    _mm256_storeu_ps(o + 16, c22);
    o += g->ldc;
    _mm256_storeu_ps(o, c30);
    _mm256_storeu_ps(o + 8, c31);
    _mm256_storeu_ps(o + 16, c32);
}

/* outputs [i0, i1): i0 and i1 multiples of 8, or i1 the end */
AVX2 static void gemm_avx2(const struct gemm *g, uint32_t i0, uint32_t i1)
{
    uint32_t i8 = i0 + (i1 - i0) / 8 * 8;
    /* a block of outputs at a time, every token through it */
    for (uint32_t ib = i0; ib < i8; ib += 192) {
        uint32_t ie = ib + 192 < i8 ? ib + 192 : i8;
        for (uint32_t j = 0; j < g->nj; j += 4) {
            uint32_t nb = g->nj - j < 4 ? g->nj - j : 4;
            for (uint32_t i = ib; i < ie; i += 24) {
                uint32_t nv = (ie - i) / 8 < 3 ? (ie - i) / 8 : 3;
                if (nb == 4 && nv == 3)
                    tile_full(g, j, i);
                else
                    tile(g, j, nb, i, nv);
            }
        }
    }
    if (i8 < i1) { /* the last outputs, fewer than eight */
        struct gemm t = *g;
        gemm_plain(&t, i8, i1);
    }
}

/* x[k] = e^(x[k] - mx) for the first n / 8 * 8, their sum added to *sum;
   returns how many it did */
AVX2 static uint32_t janas_exp_sub_avx2(float *x, uint32_t n, float mx,
                                        double *sum)
{
    uint32_t n8 = n / 8 * 8;
    __m256 vm = _mm256_set1_ps(mx), acc = _mm256_setzero_ps();
    for (uint32_t k = 0; k < n8; k += 8) {
        __m256 e = janas_exp8(_mm256_sub_ps(_mm256_loadu_ps(x + k), vm));
        _mm256_storeu_ps(x + k, e);
        acc = _mm256_add_ps(acc, e);
    }
    *sum += hsum(acc);
    return n8;
}
#endif

__attribute__((unused)) static int fast_cpu(void) /* x86's choice */
{
#if defined(__x86_64__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c");
#else
    return 0;
#endif
}

static void gemm_cols(const struct gemm *g, uint32_t i0, uint32_t i1)
{
#if defined(__x86_64__)
    if (fast_cpu()) {
        gemm_avx2(g, i0, i1);
        return;
    }
#endif
    gemm_plain(g, i0, i1);
}

static void gemm_worker(void *arg, int tid, int nt)
{
    const struct gemm *g = arg;
    /* outputs in multiples of 24, the tiles whole */
    uint32_t tiles = (g->ni + 23) / 24;
    uint32_t per = (tiles + (uint32_t)nt - 1) / (uint32_t)nt;
    uint32_t i0 = (uint32_t)tid * per * 24, i1 = i0 + per * 24;
    if (i1 > g->ni)
        i1 = g->ni;
    if (i0 < i1)
        gemm_cols(g, i0, i1);
}

static void matmul(struct janas_pool *pool, int kind, const uint16_t *wt,
                   const float *b, const float *x, float *y, uint32_t rows,
                   uint32_t cols, uint32_t n, size_t ldy)
{
    struct gemm g = {.kind = kind,
                     .a = wt,
                     .lda = rows,
                     .b = x,
                     .ldb = cols,
                     .c = y,
                     .ldc = ldy ? ldy : rows,
                     .bias = b,
                     .ni = rows,
                     .nj = n,
                     .k = cols};
    janas_pool_run(pool, gemm_worker, &g);
}

/* y[n][rows] = w x[n][cols] + b, wt the weights transposed (cols x rows,
   bfloat16) */
void vk_matmul(struct janas_pool *pool, const uint16_t *wt, const float *b,
               const float *x, float *y, uint32_t rows, uint32_t cols,
               uint32_t n, size_t ldy)
{
    matmul(pool, 1, wt, b, x, y, rows, cols, n, ldy);
}

/* the same with the weights in half precision */
void vk_matmul_f16(struct janas_pool *pool, const uint16_t *wt, const float *b,
                   const float *x, float *y, uint32_t rows, uint32_t cols,
                   uint32_t n, size_t ldy)
{
    matmul(pool, 2, wt, b, x, y, rows, cols, n, ldy);
}

/* ---- attention over every patch, a block of queries at a time ---- */

#define QB 128 /* queries a block: keys and values read once for them */

struct att {
    const float *qkv; /* n x 3 dim: q, k, v of every patch */
    float *kt;        /* heads x hd x n: the keys transposed */
    float *vh;        /* heads x n x hd: the values, a head together */
    float *out;       /* n x dim */
    uint32_t n, dim, heads, hd;
    float scale;
};

static void kt_worker(void *arg, int tid, int nt)
{
    struct att *a = arg;
    for (uint32_t h = (uint32_t)tid; h < a->heads; h += (uint32_t)nt)
        for (uint32_t k = 0; k < a->n; k++) {
            const float *key =
                a->qkv + (size_t)k * 3 * a->dim + a->dim + (size_t)h * a->hd;
            float *col = a->kt + (size_t)h * a->hd * a->n + k;
            for (uint32_t d = 0; d < a->hd; d++)
                col[(size_t)d * a->n] = key[d];
            memcpy(a->vh + ((size_t)h * a->n + k) * a->hd, key + a->dim,
                   sizeof(float) * a->hd);
        }
}

/* each row of s (len n) scaled, exponentiated and normalized in place */
static void softmax_rows(float *s, uint32_t rows, uint32_t n, float scale)
{
    for (uint32_t r = 0; r < rows; r++) {
        float *x = s + (size_t)r * n, mx = -INFINITY;
        for (uint32_t k = 0; k < n; k++) {
            x[k] *= scale;
            if (x[k] > mx)
                mx = x[k];
        }
        double sum = 0;
        uint32_t k = 0;
#if defined(__x86_64__)
        if (fast_cpu())
            k = janas_exp_sub_avx2(x, n, mx, &sum);
#endif
        for (; k < n; k++) {
            x[k] = janas_exp_det(x[k] - mx);
            sum += x[k];
        }
        float inv = (float)(1.0 / sum);
        for (k = 0; k < n; k++)
            x[k] *= inv;
    }
}

static void att_worker(void *arg, int tid, int nt)
{
    struct att *a = arg;
    float *s = malloc(sizeof(float) * QB * a->n);
    if (!s)
        return;
    float scale = a->scale;
    size_t stride = 3 * (size_t)a->dim;
    uint32_t blocks = (a->n + QB - 1) / QB;
    size_t total = (size_t)blocks * a->heads;
    for (size_t t = (size_t)tid; t < total; t += (size_t)nt) {
        uint32_t h = (uint32_t)(t / blocks), q0 = (uint32_t)(t % blocks) * QB;
        uint32_t nq = a->n - q0 < QB ? a->n - q0 : QB;
        /* scores: the block's queries as B, the keys transposed as A */
        struct gemm sc = {.kind = 0,
                          .a = a->kt + (size_t)h * a->hd * a->n,
                          .lda = a->n,
                          .b = a->qkv + (size_t)q0 * stride + (size_t)h * a->hd,
                          .ldb = stride,
                          .c = s,
                          .ldc = a->n,
                          .ni = a->n,
                          .nj = nq,
                          .k = a->hd};
        gemm_cols(&sc, 0, a->n);
        softmax_rows(s, nq, a->n, scale);
        /* the output: the weights as B, the values as A */
        struct gemm ov = {.kind = 0,
                          .a = a->vh + (size_t)h * a->n * a->hd,
                          .lda = a->hd,
                          .b = s,
                          .ldb = a->n,
                          .c = a->out + (size_t)q0 * a->dim + (size_t)h * a->hd,
                          .ldc = a->dim,
                          .ni = a->hd,
                          .nj = nq,
                          .k = a->n};
        gemm_cols(&ov, 0, a->hd);
    }
    free(s);
}

void vk_layer_norm(float *y, const float *x, const float *w, const float *b,
                   uint32_t n, float eps)
{
    double mean = 0, var = 0;
    for (uint32_t i = 0; i < n; i++)
        mean += x[i];
    mean /= n;
    for (uint32_t i = 0; i < n; i++)
        var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float inv = (float)(1.0 / sqrt(var + eps));
    for (uint32_t i = 0; i < n; i++)
        y[i] = ((float)(x[i] - mean)) * inv * w[i] + b[i];
}

float vk_gelu_tanh(float x)
{
    return 0.5f * x *
           (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

float vk_gelu_erf(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

float vk_bf(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void vk_attention(struct janas_pool *pool, const float *qkv, float *buf,
                  float *out, uint32_t n, uint32_t dim, uint32_t heads,
                  float scale)
{
    struct att a = {.qkv = qkv,
                    .kt = buf,
                    .vh = buf + (size_t)n * dim,
                    .out = out,
                    .n = n,
                    .dim = dim,
                    .heads = heads,
                    .hd = dim / heads,
                    .scale = scale};
    janas_pool_run(pool, kt_worker, &a);
    janas_pool_run(pool, att_worker, &a);
}

uint16_t *vk_transposed(const uint16_t *w, uint64_t rows, uint64_t cols)
{
    uint16_t *t = malloc(2 * (size_t)(rows * cols));
    for (uint64_t r = 0; t && r < rows; r++)
        for (uint64_t c = 0; c < cols; c++)
            t[c * rows + r] = w[r * cols + c];
    return t;
}

void vk_rms_norm(float *y, const float *x, const float *w, uint32_t n,
                 float eps)
{
    double ss = 0;
    for (uint32_t i = 0; i < n; i++)
        ss += (double)x[i] * x[i];
    float inv = (float)pow(ss / n + eps, -0.5);
    for (uint32_t i = 0; i < n; i++)
        y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}
