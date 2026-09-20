/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_quant.c - correctness of the quantized formats and kernels.
 *
 * Every optimised kernel must agree with the portable scalar reference; the
 * reference itself is checked against a plain float dot product over the
 * dequantized weights.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/pool.h"
#include "llm/matvec.h"
#include "llm/quant.h"
#include "llm/vmath.h"

static int failures;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            failures++;                                                        \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fputc('\n', stderr);                                               \
        }                                                                      \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint32_t rnd(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static float rndf(float lo, float hi)
{
    return lo + (hi - lo) * (float)(rnd() & 0xffffff) / (float)0x1000000;
}

static void random_q4k(struct janas_block_q4k *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        b[i].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
        b[i].dmin = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
        for (int j = 0; j < 12; j++)
            b[i].scales[j] = (uint8_t)rnd();
        for (int j = 0; j < JANAS_QK / 2; j++)
            b[i].qs[j] = (uint8_t)rnd();
    }
}

static void test_fp16(void)
{
    const float values[] = {0.0f,     1.0f,           -2.5f,          0.1f,
                            65504.0f, 6.1035156e-05f, 5.9604645e-08f, 1234.5f};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
        float v = values[i];
        float r = janas_fp16_to_fp32(janas_fp32_to_fp16(v));
        CHECK(fabsf(r - v) <= fabsf(v) * 1e-3f, "fp16 round trip %g -> %g", v,
              r);
    }
    /* every fp16 value survives fp16 -> fp32 -> fp16 exactly (NaNs aside) */
    for (uint32_t h = 0; h < 0x10000; h++) {
        if ((h & 0x7c00) == 0x7c00 && (h & 0x3ff))
            continue;
        uint16_t back = janas_fp32_to_fp16(janas_fp16_to_fp32((uint16_t)h));
        CHECK(back == h, "fp16 0x%04x -> 0x%04x", h, back);
        if (back != h)
            break;
    }
}

static void test_dot(void)
{
    enum { NB = 32, N = NB * JANAS_QK };
    struct janas_block_q4k *w = malloc(NB * sizeof(*w));
    struct janas_block_q8k *x = malloc(NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float));
    float *wf = malloc(N * sizeof(float));

    for (int trial = 0; trial < 200; trial++) {
        random_q4k(w, NB);
        for (int i = 0; i < N; i++)
            xf[i] = rndf(-1.0f, 1.0f);
        janas_q8k_quantize(xf, x, N);

        /* reference vs float product over the dequantized weights */
        janas_q4k_dequantize(w, wf, N);
        double expect = 0.0, mag = 0.0;
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < JANAS_QK; i++) {
                double t = (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                expect += t;
                mag += fabs(t);
            }
        float ref = janas_q4k_dot_ref(w, x, NB);
        CHECK(fabs(ref - expect) <= 1e-5 * mag, "ref %g vs float %g", ref,
              expect);

        /* the same float operations: bit-identical */
        float fast = janas_q4k_dot(w, x, NB);
        CHECK(fast == ref, "%s %g vs ref %g", janas_q4k_kernel_name(), fast,
              ref);
    }
    free(w);
    free(x);
    free(xf);
    free(wf);
}

static void test_dot_q6k(void)
{
    enum { NB = 16, N = NB * JANAS_QK };
    struct janas_block_q6k *w = malloc(NB * sizeof(*w));
    struct janas_block_q8k *x = malloc(NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float));
    float *wf = malloc(N * sizeof(float));

    for (int trial = 0; trial < 200; trial++) {
        unsigned char *p = (unsigned char *)w;
        for (size_t i = 0; i < NB * sizeof(*w); i++)
            p[i] = (unsigned char)rnd();
        for (int b = 0; b < NB; b++)
            w[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
        for (int i = 0; i < N; i++)
            xf[i] = rndf(-1.0f, 1.0f);
        janas_q8k_quantize(xf, x, N);

        janas_q6k_dequantize(w, wf, N);
        double expect = 0.0, mag = 0.0;
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < JANAS_QK; i++) {
                double t = (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                expect += t;
                mag += fabs(t);
            }
        float ref = janas_q6k_dot_ref(w, x, NB);
        CHECK(fabs(ref - expect) <= 1e-5 * mag, "q6k ref %g vs float %g", ref,
              expect);
        float fast = janas_q6k_dot(w, x, NB);
        CHECK(fast == ref, "q6k fast %g vs ref %g", fast, ref);
    }
    free(w);
    free(x);
    free(xf);
    free(wf);
}

/*
 * Q6_K cut into three two-bit planes: the split and its inverse lose nothing,
 * the fast kernels agree with the scalar reference at every level, with all
 * three planes the result is the Q6_K one bit for bit, and with fewer planes
 * every weight sits within half a step of the centre it stands for.
 */
static void test_q6k_planes(void)
{
    enum { NB = 16, NV = 9, N = NB * JANAS_QK };
    struct janas_block_q6k *w = malloc(NB * sizeof(*w));
    struct janas_block_q6k *back = malloc(NB * sizeof(*back));
    struct janas_block_q6kp *base = malloc(NB * sizeof(*base));
    uint8_t *p1 = malloc(NB * JANAS_Q6KP_PLANE);
    uint8_t *p0 = malloc(NB * JANAS_Q6KP_PLANE);
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float));
    float *wf = malloc(N * sizeof(float));
    float *pf = malloc(N * sizeof(float));

    for (int trial = 0; trial < 50; trial++) {
        unsigned char *p = (unsigned char *)w;
        for (size_t i = 0; i < NB * sizeof(*w); i++)
            p[i] = (unsigned char)rnd();
        for (int b = 0; b < NB; b++)
            w[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
        for (int v = 0; v < NV; v++) {
            for (int i = 0; i < N; i++)
                xf[i] = rndf(-1.0f, 1.0f);
            janas_q8k_quantize(xf, x + v * NB, N);
        }

        janas_q6k_to_planes(w, NB, base, p1, p0);
        janas_q6kp_to_q6k(base, p1, p0, NB, back);
        CHECK(!memcmp(w, back, NB * sizeof(*w)), "planes: round trip differs");

        float r6 = janas_q6k_dot_ref(w, x, NB);
        CHECK(janas_q6kp_dot_ref(base, p1, p0, x, NB) == r6,
              "planes ref level 3: %g vs q6k %g",
              janas_q6kp_dot_ref(base, p1, p0, x, NB), r6);
        CHECK(janas_q6kp_dot(base, p1, p0, x, NB) == janas_q6k_dot(w, x, NB),
              "planes fast level 3 differs from q6k");
        for (int level = 1; level <= 3; level++) {
            const uint8_t *a1 = level >= 2 ? p1 : NULL;
            const uint8_t *a0 = level >= 3 ? p0 : NULL;
            float ref = janas_q6kp_dot_ref(base, a1, a0, x, NB);
            CHECK(janas_q6kp_dot(base, a1, a0, x, NB) == ref,
                  "planes level %d: fast %g vs ref %g", level,
                  janas_q6kp_dot(base, a1, a0, x, NB), ref);
            float out[NV];
            for (int nv = 1; nv <= NV; nv++) {
                janas_q6kp_dot_multi(base, a1, a0, x, NB, (size_t)nv, NB, out,
                                     1);
                for (int v = 0; v < nv; v++) {
                    float r = janas_q6kp_dot_ref(base, a1, a0, x + v * NB, NB);
                    CHECK(out[v] == r, "planes level %d multi nv %d v %d",
                          level, nv, v);
                }
            }
        }

        janas_q6k_dequantize(w, wf, N);
        for (int level = 1; level <= 2; level++) {
            janas_q6kp_dequantize(base, level >= 2 ? p1 : NULL, NULL, pf, N);
            float half = level == 2 ? 1.5f : 7.5f;
            for (int b = 0; b < NB; b++) {
                float d = janas_fp16_to_fp32(w[b].d);
                for (int i = 0; i < JANAS_QK; i++) {
                    int k = b * JANAS_QK + i;
                    float step = fabsf(d * w[b].scales[i / 16]);
                    CHECK(fabsf(pf[k] - wf[k]) <= half * step * 1.001f,
                          "planes level %d weight %d: %g vs %g (step %g)",
                          level, k, pf[k], wf[k], step);
                }
            }
        }
    }
    free(w);
    free(back);
    free(base);
    free(p1);
    free(p0);
    free(x);
    free(xf);
    free(wf);
    free(pf);
}

static void test_matvec(void)
{
    enum { ROWS = 1000, COLS = 2048, NB = COLS / JANAS_QK };
    struct janas_block_q4k *w = malloc((size_t)ROWS * NB * sizeof(*w));
    struct janas_block_q8k *x = malloc(NB * sizeof(*x));
    float xf[COLS], y[ROWS];

    random_q4k(w, (size_t)ROWS * NB);
    for (int i = 0; i < COLS; i++)
        xf[i] = rndf(-1.0f, 1.0f);
    janas_q8k_quantize(xf, x, COLS);

    for (int threads = 1; threads <= 4; threads *= 2) {
        struct janas_pool *pool = janas_pool_create(threads, NULL);
        CHECK(pool != NULL, "pool of %d threads", threads);
        if (!pool)
            continue;
        for (int rep = 0; rep < 20; rep++) {
            memset(y, 0, sizeof(y));
            janas_matvec_q4k(pool, w, ROWS, COLS, x, y);
            for (int r = 0; r < ROWS; r++) {
                float ref = janas_q4k_dot(w + (size_t)r * NB, x, NB);
                CHECK(y[r] == ref, "matvec threads %d row %d: %g vs %g",
                      threads, r, y[r], ref);
                if (y[r] != ref)
                    break;
            }
        }
        janas_pool_destroy(pool);
    }
    free(w);
    free(x);
}

static void test_matvec_group(void)
{
    /* expert-like shapes, one empty task, different inputs per task */
    const size_t rows[] = {512, 0, 512, 2048, 37, 256};
    const size_t cols[] = {2048, 2048, 2048, 512, 768, 256};
    enum { N = 6 };
    struct janas_matvec_task tasks[N];
    for (int i = 0; i < N; i++) {
        size_t nb = cols[i] / JANAS_QK;
        struct janas_block_q4k *w = malloc((rows[i] * nb + 1) * sizeof(*w));
        struct janas_block_q8k *x = malloc(nb * sizeof(*x));
        float *xf = malloc(cols[i] * sizeof(float));
        random_q4k(w, rows[i] * nb);
        for (size_t j = 0; j < cols[i]; j++)
            xf[j] = rndf(-1.0f, 1.0f);
        janas_q8k_quantize(xf, x, cols[i]);
        free(xf);
        tasks[i] = (struct janas_matvec_task){
            .type = JANAS_Q4_K,
            .w = w,
            .x = x,
            .y = malloc((rows[i] + 1) * sizeof(float)),
            .rows = rows[i],
            .cols = cols[i]};
    }
    struct janas_pool *pool = janas_pool_create(3, NULL);
    CHECK(pool != NULL, "pool");
    for (int rep = 0; pool && rep < 10; rep++) {
        janas_matvec_q4k_group(pool, tasks, N);
        for (int i = 0; i < N; i++) {
            size_t nb = cols[i] / JANAS_QK;
            for (size_t r = 0; r < rows[i]; r++) {
                const struct janas_block_q4k *tw = tasks[i].w;
                float ref = janas_q4k_dot(tw + r * nb, tasks[i].x, nb);
                CHECK(tasks[i].y[r] == ref, "group task %d row %zu: %g vs %g",
                      i, r, tasks[i].y[r], ref);
                if (tasks[i].y[r] != ref)
                    break;
            }
        }
    }
    janas_pool_destroy(pool);
    for (int i = 0; i < N; i++) {
        free((void *)tasks[i].w);
        free((void *)tasks[i].x);
        free(tasks[i].y);
    }
}

static void test_matvec_multi(void)
{
    enum { ROWS = 300, COLS = 512, NB = COLS / JANAS_QK, NV = 5 };
    struct janas_block_q4k *w = malloc((size_t)ROWS * NB * sizeof(*w));
    struct janas_block_q8k *x = malloc((size_t)NV * NB * sizeof(*x));
    float xf[COLS], y[NV * ROWS];
    random_q4k(w, (size_t)ROWS * NB);
    for (int v = 0; v < NV; v++) {
        for (int i = 0; i < COLS; i++)
            xf[i] = rndf(-1.0f, 1.0f);
        janas_q8k_quantize(xf, x + (size_t)v * NB, COLS);
    }
    struct janas_pool *pool = janas_pool_create(3, NULL);
    struct janas_matvec_task t = {.type = JANAS_Q4_K,
                                  .w = w,
                                  .x = x,
                                  .y = y,
                                  .rows = ROWS,
                                  .cols = COLS,
                                  .n_vec = NV};
    janas_matvec_q4k_group(pool, &t, 1);
    for (int v = 0; v < NV; v++)
        for (int r = 0; r < ROWS; r++) {
            float ref =
                janas_q4k_dot(w + (size_t)r * NB, x + (size_t)v * NB, NB);
            CHECK(y[v * ROWS + r] == ref, "multi v %d row %d: %g vs %g", v, r,
                  y[v * ROWS + r], ref);
        }
    janas_pool_destroy(pool);
    free(w);
    free(x);
}

/*
 * Q4_K quantization: round trip on Gaussian-like and heavy-tailed weights.
 * The error relative to the weights' spread must stay near what 4 bits with
 * 6-bit sub-block scales allow, and below a plain min/max rounding.
 */
static void test_quantize_q4k(void)
{
    enum { N = 256 * 64 };
    float *x = malloc(N * sizeof(float)), *y = malloc(N * sizeof(float));
    struct janas_block_q4k *q = malloc(N / JANAS_QK * sizeof(*q));
    for (int kind = 0; kind < 2; kind++) {
        double var = 0, err = 0, naive = 0;
        for (int i = 0; i < N; i++) {
            /* sum of uniforms ~ Gaussian; its cube is heavy-tailed */
            float g = rndf(-1, 1) + rndf(-1, 1) + rndf(-1, 1) + rndf(-1, 1);
            x[i] = 0.02f * (kind ? g * g * g : g);
            var += (double)x[i] * x[i];
        }
        janas_q4k_quantize(x, q, N);
        janas_q4k_dequantize(q, y, N);
        for (int i = 0; i < N; i++)
            err += ((double)y[i] - x[i]) * (y[i] - x[i]);
        for (int s = 0; s < N / 32; s++) { /* min/max rounding, float scales */
            float lo = x[32 * s], hi = lo;
            for (int k = 0; k < 32; k++) {
                lo = fminf(lo, x[32 * s + k]);
                hi = fmaxf(hi, x[32 * s + k]);
            }
            float st = (hi - lo) / 15;
            for (int k = 0; k < 32; k++) {
                float v =
                    st > 0 ? lo + st * lrintf((x[32 * s + k] - lo) / st) : lo;
                naive += ((double)v - x[32 * s + k]) * (v - x[32 * s + k]);
            }
        }
        double rel = sqrt(err / var), rel_naive = sqrt(naive / var);
        printf("q4k quantize, %s weights: relative rms error %.4f (min/max "
               "rounding %.4f)\n",
               kind ? "heavy-tailed" : "Gaussian", rel, rel_naive);
        CHECK(rel < (kind ? 0.20 : 0.11) && rel < rel_naive,
              "q4k quantize %d: relative error %.4f, naive %.4f", kind, rel,
              rel_naive);
    }
    free(x);
    free(y);
    free(q);
}

/* janas_dot_multi on 1 to 17 vectors, both types: equal to the scalar
   reference for every vector (the kernels group vectors by eight). */
static void test_dot_multi(void)
{
    enum { NB = 8, NV = 17, N = NB * JANAS_QK };
    struct janas_block_q4k *w4 = malloc(NB * sizeof(*w4));
    struct janas_block_q6k *w6 = malloc(NB * sizeof(*w6));
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float));
    random_q4k(w4, NB);
    unsigned char *p = (unsigned char *)w6;
    for (size_t i = 0; i < NB * sizeof(*w6); i++)
        p[i] = (unsigned char)rnd();
    for (int b = 0; b < NB; b++)
        w6[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
    for (int v = 0; v < NV; v++) {
        for (int i = 0; i < N; i++)
            xf[i] = rndf(-1.0f, 1.0f);
        janas_q8k_quantize(xf, x + v * NB, N);
    }
    for (int nv = 1; nv <= NV; nv++) {
        float out[NV];
        janas_dot_multi(JANAS_Q4_K, w4, x, NB, (size_t)nv, NB, out, 1);
        for (int v = 0; v < nv; v++) {
            float ref = janas_q4k_dot_ref(w4, x + v * NB, NB);
            CHECK(out[v] == ref, "q4k multi nv %d v %d: %g vs %g", nv, v,
                  out[v], ref);
        }
        janas_dot_multi(JANAS_Q6_K, w6, x, NB, (size_t)nv, NB, out, 1);
        for (int v = 0; v < nv; v++) {
            float ref = janas_q6k_dot_ref(w6, x + v * NB, NB);
            CHECK(out[v] == ref, "q6k multi nv %d v %d: %g vs %g", nv, v,
                  out[v], ref);
        }
    }
    free(w4);
    free(w6);
    free(x);
    free(xf);
}

#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) static void
det8(const float *g, const float *u, float *h, int n)
{
    for (int i = 0; i < n; i += 8)
        _mm256_storeu_ps(h + i, janas_silu_mul_det8(_mm256_loadu_ps(g + i),
                                                    _mm256_loadu_ps(u + i)));
}
#endif

/* The deterministic functions: scalar and AVX2 give the same bits; the
   reciprocal is within an ulp; the deterministic quantization is within one
   step of the plain one. */
static void test_det(void)
{
    float g[1024], u[1024], a[1024], b[1024];
    int bad = 0;
    for (int round = 0; round < 1000; round++) {
        for (int i = 0; i < 1024; i++) {
            /* wide range, and the clamp edges */
            g[i] = rndf(-1.0f, 1.0f) * (i % 7 == 0 ? 120.0f : 12.0f);
            u[i] = rndf(-4.0f, 4.0f);
        }
        g[0] = -30.0f, g[1] = -30.5f, g[2] = 88.0f, g[3] = -88.0f;
        g[4] = 0.0f, g[5] = -0.0f, g[6] = 1e-30f, g[7] = 100.0f;
        for (int i = 0; i < 1024; i++)
            a[i] = janas_silu_mul_det(g[i], u[i]);
#if defined(__x86_64__)
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            det8(g, u, b, 1024);
            for (int i = 0; i < 1024; i++)
                bad += memcmp(&a[i], &b[i], 4) != 0;
        }
#endif
    }
    CHECK(bad == 0, "silu_mul_det: %d scalar/AVX2 differences", bad);
    double worst = 0;
    for (int i = 0; i < 100000; i++) {
        float d = ldexpf(rndf(1.0f, 2.0f), (int)(rnd() % 200) - 100);
        double r = janas_recip_det(d), e = fabs(r * d - 1.0);
        if (e > worst)
            worst = e;
    }
    CHECK(worst < 1.2e-7, "recip_det: worst relative error %g", worst);
    enum { N = 1024 };
    float x[N];
    struct janas_block_q8k qa[N / JANAS_QK], qb[N / JANAS_QK];
    for (int i = 0; i < N; i++)
        x[i] = rndf(-3.0f, 3.0f);
    janas_q8k_quantize(x, qa, N);
    janas_q8k_quantize_det(x, qb, N);
    int far = 0;
    for (int b = 0; b < N / JANAS_QK; b++)
        for (int i = 0; i < JANAS_QK; i++)
            far += abs(qa[b].qs[i] - qb[b].qs[i]) > 1;
    CHECK(far == 0, "quantize_det: %d values off by more than one", far);
}

/* Q5_K and Q8_0: the reference against a float product over the
   dequantized weights, the kernels (single and 1-17 vectors) equal to it. */
static void test_dot_q5k_q8_0(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    struct janas_block_q5k *w5 = malloc(NB * sizeof(*w5));
    struct janas_block_q8_0 *w8 = malloc(NB * 8 * sizeof(*w8));
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float)), *wf = malloc(N * sizeof(float));
    for (int trial = 0; trial < 100; trial++) {
        unsigned char *p5 = (unsigned char *)w5, *p8 = (unsigned char *)w8;
        for (size_t i = 0; i < NB * sizeof(*w5); i++)
            p5[i] = (unsigned char)rnd();
        for (size_t i = 0; i < NB * 8 * sizeof(*w8); i++)
            p8[i] = (unsigned char)rnd();
        for (int b = 0; b < NB; b++) {
            w5[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
            w5[b].dmin = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
        }
        for (int b = 0; b < NB * 8; b++)
            w8[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
        for (int v = 0; v < NV; v++) {
            for (int i = 0; i < N; i++)
                xf[i] = rndf(-1.0f, 1.0f);
            janas_q8k_quantize(xf, x + v * NB, N);
        }
        for (int type = 0; type < 2; type++) {
            const void *w = type ? (const void *)w8 : (const void *)w5;
            int t = type ? JANAS_Q8_0 : JANAS_Q5_K;
            if (type)
                janas_q8_0_dequantize(w8, wf, N);
            else
                janas_q5k_dequantize(w5, wf, N);
            double expect = 0.0, mag = 0.0;
            for (int b = 0; b < NB; b++)
                for (int i = 0; i < JANAS_QK; i++) {
                    double e =
                        (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                    expect += e;
                    mag += fabs(e);
                }
            float ref = type ? janas_q8_0_dot_ref(w8, x, NB)
                             : janas_q5k_dot_ref(w5, x, NB);
            CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "%s ref %g vs %g",
                  type ? "q8_0" : "q5k", ref, expect);
            float fast = janas_dot(t, w, x, NB);
            CHECK(fast == ref, "%s fast %g vs ref %g", type ? "q8_0" : "q5k",
                  fast, ref);
            for (int nv = 1; nv <= NV; nv++) {
                float out[NV];
                janas_dot_multi(t, w, x, NB, (size_t)nv, NB, out, 1);
                for (int v = 0; v < nv; v++) {
                    float r = type ? janas_q8_0_dot_ref(w8, x + v * NB, NB)
                                   : janas_q5k_dot_ref(w5, x + v * NB, NB);
                    CHECK(out[v] == r, "%s multi nv %d v %d: %g vs %g",
                          type ? "q8_0" : "q5k", nv, v, out[v], r);
                }
            }
        }
    }
    free(w5);
    free(w8);
    free(x);
    free(xf);
    free(wf);
}

/* IQ4_NL and IQ4_XS: as above - the reference against a float product
   over the dequantized weights, the kernels (single and 1-17 vectors) equal
   to it to the bit. */
static void test_dot_iq4(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    struct janas_block_iq4nl *wn = malloc(NB * 8 * sizeof(*wn));
    struct janas_block_iq4xs *wx = malloc(NB * sizeof(*wx));
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float)), *wf = malloc(N * sizeof(float));
    for (int trial = 0; trial < 100; trial++) {
        unsigned char *pn = (unsigned char *)wn, *px = (unsigned char *)wx;
        for (size_t i = 0; i < NB * 8 * sizeof(*wn); i++)
            pn[i] = (unsigned char)rnd();
        for (size_t i = 0; i < NB * sizeof(*wx); i++)
            px[i] = (unsigned char)rnd();
        for (int b = 0; b < NB * 8; b++)
            wn[b].d = janas_fp32_to_fp16(rndf(-0.01f, 0.01f));
        for (int b = 0; b < NB; b++)
            wx[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
        for (int v = 0; v < NV; v++) {
            for (int i = 0; i < N; i++)
                xf[i] = rndf(-1.0f, 1.0f);
            janas_q8k_quantize(xf, x + v * NB, N);
        }
        for (int xs4 = 0; xs4 < 2; xs4++) {
            const void *w = xs4 ? (const void *)wx : (const void *)wn;
            int t = xs4 ? JANAS_IQ4_XS : JANAS_IQ4_NL;
            const char *name = xs4 ? "iq4_xs" : "iq4_nl";
            if (xs4)
                janas_iq4xs_dequantize(wx, wf, N);
            else
                janas_iq4nl_dequantize(wn, wf, N);
            double expect = 0.0, mag = 0.0;
            for (int b = 0; b < NB; b++)
                for (int i = 0; i < JANAS_QK; i++) {
                    double e =
                        (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                    expect += e;
                    mag += fabs(e);
                }
            float ref = xs4 ? janas_iq4xs_dot_ref(wx, x, NB)
                            : janas_iq4nl_dot_ref(wn, x, NB);
            CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "%s ref %g vs %g",
                  name, ref, expect);
            float fast = janas_dot(t, w, x, NB);
            CHECK(fast == ref, "%s fast %g vs ref %g", name, fast, ref);
            for (int nv = 1; nv <= NV; nv++) {
                float out[NV];
                janas_dot_multi(t, w, x, NB, (size_t)nv, NB, out, 1);
                for (int v = 0; v < nv; v++) {
                    float r = xs4 ? janas_iq4xs_dot_ref(wx, x + v * NB, NB)
                                  : janas_iq4nl_dot_ref(wn, x + v * NB, NB);
                    CHECK(out[v] == r, "%s multi nv %d v %d: %g vs %g", name,
                          nv, v, out[v], r);
                }
            }
        }
    }
    /* the levels and a scale by hand: an index of 0 is -127, 15 is 113,
       and group 1's scale takes its high bits from bits 2-3 of scales_h */
    struct janas_block_iq4xs h = {0};
    h.d = janas_fp32_to_fp16(1.0f);
    h.scales_l[0] = 0x50;  /* group 1: low bits 5 */
    h.scales_h = 0x3 << 2; /* group 1: high bits 3 -> ls 53, scale 21 */
    h.qs[16] = 0xf0;       /* group 1, weights 0 and 16: levels 0 and 15 */
    float y[JANAS_QK];
    janas_iq4xs_dequantize(&h, y, JANAS_QK);
    CHECK(y[32] == 21.0f * -127 && y[48] == 21.0f * 113,
          "iq4_xs by hand: %g %g", y[32], y[48]);
    free(wn);
    free(wx);
    free(x);
    free(xf);
    free(wf);
}

/* Q4_0 and Q5_1: the reference against a float product over the
   dequantized weights (Q5_1's minimum times the activations' sums
   included), the kernels on 1-17 vectors equal to it to the bit. */
static void test_dot_q4_0_q5_1(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    struct janas_block_q4_0 *w4 = malloc(NB * 8 * sizeof(*w4));
    struct janas_block_q5_1 *w5 = malloc(NB * 8 * sizeof(*w5));
    struct janas_block_q4_1 *w1 = malloc(NB * 8 * sizeof(*w1));
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float)), *wf = malloc(N * sizeof(float));
    for (int trial = 0; trial < 100; trial++) {
        unsigned char *p4 = (unsigned char *)w4, *p5 = (unsigned char *)w5;
        for (size_t i = 0; i < NB * 8 * sizeof(*w4); i++)
            p4[i] = (unsigned char)rnd();
        for (size_t i = 0; i < NB * 8 * sizeof(*w5); i++)
            p5[i] = (unsigned char)rnd();
        unsigned char *p1 = (unsigned char *)w1;
        for (size_t i = 0; i < NB * 8 * sizeof(*w1); i++)
            p1[i] = (unsigned char)rnd();
        for (int b = 0; b < NB * 8; b++) {
            w4[b].d = janas_fp32_to_fp16(rndf(-0.01f, 0.01f));
            w5[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
            w5[b].m = janas_fp32_to_fp16(rndf(-0.1f, 0.0f));
            w1[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
            w1[b].m = janas_fp32_to_fp16(rndf(-0.1f, 0.0f));
        }
        for (int v = 0; v < NV; v++) {
            for (int i = 0; i < N; i++)
                xf[i] = rndf(-1.0f, 1.0f);
            janas_q8k_quantize(xf, x + v * NB, N);
        }
        for (int five = 0; five < 3; five++) {
            const void *w = five == 2 ? (const void *)w1
                            : five    ? (const void *)w5
                                      : (const void *)w4;
            int t = five == 2 ? JANAS_Q4_1 : five ? JANAS_Q5_1 : JANAS_Q4_0;
            const char *name = five == 2 ? "q4_1" : five ? "q5_1" : "q4_0";
            if (five == 2)
                janas_q4_1_dequantize(w1, wf, N);
            else if (five)
                janas_q5_1_dequantize(w5, wf, N);
            else
                janas_q4_0_dequantize(w4, wf, N);
            double expect = 0.0, mag = 0.0;
            for (int b = 0; b < NB; b++)
                for (int i = 0; i < JANAS_QK; i++) {
                    double e =
                        (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                    expect += e;
                    mag += fabs(e);
                }
            float ref = five == 2 ? janas_q4_1_dot_ref(w1, x, NB)
                        : five    ? janas_q5_1_dot_ref(w5, x, NB)
                                  : janas_q4_0_dot_ref(w4, x, NB);
            CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "%s ref %g vs %g",
                  name, ref, expect);
            float fast = janas_dot(t, w, x, NB);
            CHECK(fast == ref, "%s fast %g vs ref %g", name, fast, ref);
            for (int nv = 1; nv <= NV; nv++) {
                float out[NV];
                janas_dot_multi(t, w, x, NB, (size_t)nv, NB, out, 1);
                for (int v = 0; v < nv; v++) {
                    float r = five == 2 ? janas_q4_1_dot_ref(w1, x + v * NB, NB)
                              : five    ? janas_q5_1_dot_ref(w5, x + v * NB, NB)
                                     : janas_q4_0_dot_ref(w4, x + v * NB, NB);
                    CHECK(out[v] == r, "%s multi nv %d v %d: %g vs %g", name,
                          nv, v, out[v], r);
                }
            }
        }
    }
    /* by hand: Q5_1 weight 17 takes bit 17 of qh and the high nibble of
       byte 1; weight 1 is only the minimum */
    struct janas_block_q5_1 h = {0};
    h.d = janas_fp32_to_fp16(1.0f);
    h.m = janas_fp32_to_fp16(-2.0f);
    h.qh[2] = 0x02; /* bit 17 */
    h.qs[1] = 0x30; /* weight 17: low bits 3 */
    float y[32];
    janas_q5_1_dequantize(&h, y, 32);
    CHECK(y[17] == 19.0f - 2.0f && y[1] == -2.0f, "q5_1 by hand: %g %g", y[17],
          y[1]);
    free(w4);
    free(w5);
    free(w1);
    free(x);
    free(xf);
    free(wf);
}

/*
 * Unpacking a whole row: janas_dequantize has to reach the right unpacker for
 * every type the engine accepts in a model file, not just the two it used to
 * know. A token embedding is read this way, and reading a Q8_0 one as Q4_K
 * gave numbers so large that the first layer produced NaN - the model loaded,
 * passed its checksums and answered nonsense (reported from an i7-1365U, on a
 * GGUF whose token_embd.weight is Q8_0).
 */
static void test_dequantize(void)
{
    enum { NB = 8, N = NB * JANAS_QK };
    static const int types[] = {JANAS_Q4_K, JANAS_Q5_K,   JANAS_Q6_K,
                                JANAS_Q8_0, JANAS_IQ4_NL, JANAS_IQ4_XS,
                                JANAS_Q4_0, JANAS_Q5_1,   JANAS_Q4_1};
    float *a = malloc(N * sizeof(float)), *b = malloc(N * sizeof(float));
    for (unsigned k = 0; k < sizeof(types) / sizeof(*types); k++) {
        int type = types[k];
        size_t bytes = NB * janas_qtype_block_size(type);
        CHECK(bytes != 0, "type %d: no block size", type);
        unsigned char *w = malloc(bytes);
        for (size_t i = 0; i < bytes; i++)
            w[i] = (unsigned char)rnd();
        /* finite scales, so the two ways can be compared bit for bit */
        for (int i = 0; i < NB; i++)
            switch (type) {
            case JANAS_Q4_K: {
                struct janas_block_q4k *q = (struct janas_block_q4k *)w + i;
                q->d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                q->dmin = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                break;
            }
            case JANAS_Q5_K: {
                struct janas_block_q5k *q = (struct janas_block_q5k *)w + i;
                q->d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                q->dmin = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                break;
            }
            case JANAS_Q6_K: {
                struct janas_block_q6k *q = (struct janas_block_q6k *)w + i;
                q->d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                break;
            }
            case JANAS_IQ4_NL: { /* eight blocks cover one super-block */
                struct janas_block_iq4nl *q =
                    (struct janas_block_iq4nl *)w + i * 8;
                for (int j = 0; j < 8; j++)
                    q[j].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                break;
            }
            case JANAS_IQ4_XS: {
                struct janas_block_iq4xs *q = (struct janas_block_iq4xs *)w + i;
                q->d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                break;
            }
            case JANAS_Q4_0: {
                struct janas_block_q4_0 *q =
                    (struct janas_block_q4_0 *)w + i * 8;
                for (int j = 0; j < 8; j++)
                    q[j].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                break;
            }
            case JANAS_Q4_1: {
                struct janas_block_q4_1 *q =
                    (struct janas_block_q4_1 *)w + i * 8;
                for (int j = 0; j < 8; j++) {
                    q[j].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                    q[j].m = janas_fp32_to_fp16(rndf(-0.1f, 0.0f));
                }
                break;
            }
            case JANAS_Q5_1: {
                struct janas_block_q5_1 *q =
                    (struct janas_block_q5_1 *)w + i * 8;
                for (int j = 0; j < 8; j++) {
                    q[j].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                    q[j].m = janas_fp32_to_fp16(rndf(-0.1f, 0.0f));
                }
                break;
            }
            default: { /* eight Q8_0 blocks cover one super-block */
                struct janas_block_q8_0 *q =
                    (struct janas_block_q8_0 *)w + i * 8;
                for (int j = 0; j < 8; j++)
                    q[j].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
            }
            }
        switch (type) {
        case JANAS_Q4_K:
            janas_q4k_dequantize((const void *)w, a, N);
            break;
        case JANAS_Q5_K:
            janas_q5k_dequantize((const void *)w, a, N);
            break;
        case JANAS_Q6_K:
            janas_q6k_dequantize((const void *)w, a, N);
            break;
        case JANAS_IQ4_NL:
            janas_iq4nl_dequantize((const void *)w, a, N);
            break;
        case JANAS_IQ4_XS:
            janas_iq4xs_dequantize((const void *)w, a, N);
            break;
        case JANAS_Q4_0:
            janas_q4_0_dequantize((const void *)w, a, N);
            break;
        case JANAS_Q5_1:
            janas_q5_1_dequantize((const void *)w, a, N);
            break;
        case JANAS_Q4_1:
            janas_q4_1_dequantize((const void *)w, a, N);
            break;
        default:
            janas_q8_0_dequantize((const void *)w, a, N);
        }
        janas_dequantize(type, (const void *)w, b, N);
        CHECK(memcmp(a, b, N * sizeof(float)) == 0,
              "type %d: janas_dequantize differs from its own unpacker", type);
        /* the stride a row of a table is read with, as embed() computes it,
           has to be the one the unpacker walks: the second half of this
           buffer read as a row of its own must give the same numbers */
        size_t half = N / 2,
               row_bytes = half / JANAS_QK * janas_qtype_block_size(type);
        janas_dequantize(type, (const void *)(w + row_bytes), b, half);
        CHECK(memcmp(a + half, b, half * sizeof(float)) == 0,
              "type %d: a row read at its own offset differs", type);
        free(w);
    }
    free(a);
    free(b);
}

/*
 * The vector quantization of an activation must be what the scalar loop
 * gives, to the bit: the engine's numbers stay reproducible whatever the
 * machine, and a difference here would move every logit that follows.
 */
static void q8k_ref(const float *x, struct janas_block_q8k *y, size_t n,
                    int det)
{
    for (size_t b = 0; b < n / JANAS_QK; b++, x += JANAS_QK) {
        float amax = 0.0f;
        for (int i = 0; i < JANAS_QK; i++)
            amax = fmaxf(amax, fabsf(x[i]));
        float d = det ? amax * 0.007874015718698502f : amax / 127.0f;
        float id = d > 0.0f ? (det ? janas_recip_det(d) : 1.0f / d) : 0.0f;
        y[b].d = d;
        for (int i = 0; i < JANAS_QK; i++) {
            float q = nearbyintf(x[i] * id);
            y[b].qs[i] = (int8_t)(q > 127.0f ? 127 : (q < -127.0f ? -127 : q));
        }
        for (int g = 0; g < JANAS_QK / 16; g++) {
            int s = 0;
            for (int i = 0; i < 16; i++)
                s += y[b].qs[g * 16 + i];
            y[b].bsums[g] = (int16_t)s;
        }
    }
}

static void test_quantize_q8k(void)
{
    enum { NB = 6, N = NB * JANAS_QK };
    static float x[N];
    static struct janas_block_q8k got[NB], want[NB];
    for (int round = 0; round < 8; round++) {
        for (int i = 0; i < N; i++)
            switch (round) {
            case 0:
                x[i] = 0.0f;
                break; /* every weight zero */
            case 1:
                x[i] = rndf(-1.0f, 1.0f);
                break;
            case 2:
                x[i] = rndf(-1e-6f, 1e-6f);
                break; /* tiny */
            case 3:
                x[i] = rndf(-1e6f, 1e6f);
                break; /* large */
            case 4:
                x[i] = (i % 2) ? -0.0f : 0.0f;
                break;
            /* values that land exactly halfway, where rounding must tie to
               even the same way in both */
            case 5:
                x[i] = (float)((i % 255) - 127) + 0.5f;
                break;
            case 6:
                x[i] = (float)((i % 255) - 127);
                break;
            default:
                x[i] = rndf(-3.0f, 3.0f) * (i % 7 ? 1.0f : 1e4f);
                break;
            }
        for (int det = 0; det < 2; det++) {
            if (det)
                janas_q8k_quantize_det(x, got, N);
            else
                janas_q8k_quantize(x, got, N);
            q8k_ref(x, want, N, det);
            CHECK(memcmp(got, want, sizeof(want)) == 0,
                  "q8k quantize%s round %d differs from the reference",
                  det ? " (det)" : "", round);
        }
    }
}

/* Q3_K: the reference against a float product over the dequantized
   weights, the kernels on 1-17 vectors equal to it to the bit, and
   janas_dequantize reaching its unpacker. */
static void test_dot_q3k(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    struct janas_block_q3k *w = malloc(NB * sizeof(*w));
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float)), *wf = malloc(N * sizeof(float)),
          *w2 = malloc(N * sizeof(float));
    for (int trial = 0; trial < 100; trial++) {
        unsigned char *p = (unsigned char *)w;
        for (size_t i = 0; i < NB * sizeof(*w); i++)
            p[i] = (unsigned char)rnd();
        for (int b = 0; b < NB; b++)
            w[b].d = janas_fp32_to_fp16(rndf(-0.01f, 0.01f));
        for (int v = 0; v < NV; v++) {
            for (int i = 0; i < N; i++)
                xf[i] = rndf(-1.0f, 1.0f);
            janas_q8k_quantize(xf, x + v * NB, N);
        }
        janas_q3k_dequantize(w, wf, N);
        janas_dequantize(JANAS_Q3_K, w, w2, N);
        CHECK(memcmp(wf, w2, N * sizeof(float)) == 0,
              "q3k: janas_dequantize differs from its unpacker");
        double expect = 0.0, mag = 0.0;
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < JANAS_QK; i++) {
                double e = (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                expect += e;
                mag += fabs(e);
            }
        float ref = janas_q3k_dot_ref(w, x, NB);
        CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "q3k ref %g vs %g", ref,
              expect);
        CHECK(janas_dot(JANAS_Q3_K, w, x, NB) == ref, "q3k fast vs ref");
        for (int nv = 1; nv <= NV; nv++) {
            float out[NV];
            janas_dot_multi(JANAS_Q3_K, w, x, NB, (size_t)nv, NB, out, 1);
            for (int v = 0; v < nv; v++)
                CHECK(out[v] == janas_q3k_dot_ref(w, x + v * NB, NB),
                      "q3k multi nv %d v %d", nv, v);
        }
    }
    free(w);
    free(x);
    free(xf);
    free(wf);
    free(w2);
}

/* IQ3_S: the reference against a float product over the dequantized
   weights, the kernels on 1-17 vectors equal to it to the bit, and
   janas_dequantize reaching its unpacker. */
static void test_dot_iq3s(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    struct janas_block_iq3s *w = malloc(NB * sizeof(*w));
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float)), *wf = malloc(N * sizeof(float)),
          *w2 = malloc(N * sizeof(float));
    for (int trial = 0; trial < 100; trial++) {
        unsigned char *p = (unsigned char *)w;
        for (size_t i = 0; i < NB * sizeof(*w); i++)
            p[i] = (unsigned char)rnd();
        for (int b = 0; b < NB; b++)
            w[b].d = janas_fp32_to_fp16(rndf(-0.01f, 0.01f));
        for (int v = 0; v < NV; v++) {
            for (int i = 0; i < N; i++)
                xf[i] = rndf(-1.0f, 1.0f);
            janas_q8k_quantize(xf, x + v * NB, N);
        }
        janas_iq3s_dequantize(w, wf, N);
        janas_dequantize(JANAS_IQ3_S, w, w2, N);
        CHECK(memcmp(wf, w2, N * sizeof(float)) == 0,
              "iq3s: janas_dequantize differs from its unpacker");
        double expect = 0.0, mag = 0.0;
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < JANAS_QK; i++) {
                double e = (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                expect += e;
                mag += fabs(e);
            }
        float ref = janas_iq3s_dot_ref(w, x, NB);
        CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "iq3s ref %g vs %g", ref,
              expect);
        CHECK(janas_dot(JANAS_IQ3_S, w, x, NB) == ref, "iq3s fast vs ref");
        for (int nv = 1; nv <= NV; nv++) {
            float out[NV];
            janas_dot_multi(JANAS_IQ3_S, w, x, NB, (size_t)nv, NB, out, 1);
            for (int v = 0; v < nv; v++)
                CHECK(out[v] == janas_iq3s_dot_ref(w, x + v * NB, NB),
                      "iq3s multi nv %d v %d", nv, v);
        }
    }
    free(w);
    free(x);
    free(xf);
    free(wf);
    free(w2);
}

int main(void)
{
    /* a test that deadlocks must fail, not hold the suite for ever */
    alarm(600);
    test_fp16();
    test_dot();
    test_dot_q6k();
    test_q6k_planes();
    test_matvec();
    test_matvec_group();
    test_matvec_multi();
    test_dot_multi();
    test_det();
    test_dot_q5k_q8_0();
    test_dot_iq4();
    test_dot_q4_0_q5_1();
    test_dot_q3k();
    test_dot_iq3s();
    test_quantize_q4k();
    test_quantize_q8k();
    test_dequantize();
    printf("test_quant: kernel %s, %s\n", janas_q4k_kernel_name(),
           failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
