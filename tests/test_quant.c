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
#include "llm/bf16.h"
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

/* An activation as the product of weight type t sees it: weights in
   blocks of 32 take the eight blocks of 32 (d32, qs32), the others the
   256 with one scale. */
static double act(const struct janas_block_q8k *x, int b, int i, int t)
{
    int b32 = t == JANAS_Q8_0 || t == JANAS_Q4_0 || t == JANAS_Q4_1 ||
              t == JANAS_Q5_1 || t == JANAS_IQ4_NL || t == JANAS_MXFP4 ||
              t == JANAS_Q5_0 || t == JANAS_Q2_0;
    return b32 ? (double)x[b].d32[i / 32] * x[b].qs32[i]
               : (double)x[b].d * x[b].qs[i];
}

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

/*
 * Q6_K and Q6_K_P as tiles of eight rows (janas_q6k_tile_rows), and of
 * sixteen on 512 bits where AVX-512 VNNI is: 27 rows, 9 blocks (past the eight
 * transposed at a time), 1 to 65 vectors (past the 64 tokens at a time), Q6_K_P
 * at every level; each result janas_q6k_dot's or janas_q6kp_dot's to the bit,
 * and the rows done a multiple of eight.
 */
static void test_q6k_tiles(void)
{
    enum { ROWS = 27, NB = 9, NV = 65, N = NB * JANAS_QK };
    size_t pb = NB * JANAS_Q6KP_PLANE;
    struct janas_block_q6k *w = malloc(ROWS * NB * sizeof(*w));
    struct janas_block_q6kp *base = malloc(ROWS * NB * sizeof(*base));
    uint8_t *p1 = malloc(ROWS * pb), *p0 = malloc(ROWS * pb);
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float));
    float *out = malloc(ROWS * NV * sizeof(float));
    unsigned char *p = (unsigned char *)w;
    for (size_t i = 0; i < ROWS * NB * sizeof(*w); i++)
        p[i] = (unsigned char)rnd();
    for (int b = 0; b < ROWS * NB; b++)
        w[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
    for (int r = 0; r < ROWS; r++)
        janas_q6k_to_planes(w + r * NB, NB, base + r * NB, p1 + r * pb,
                            p0 + r * pb);
    for (int v = 0; v < NV; v++) {
        for (int i = 0; i < N; i++)
            xf[i] = rndf(-1.0f, 1.0f);
        janas_q8k_quantize(xf, x + v * NB, N);
    }
    for (int level = 0; level <= 3; level++) {
        const uint8_t *a1 = level >= 2 ? p1 : NULL;
        const uint8_t *a0 = level >= 3 ? p0 : NULL;
        int t = level ? JANAS_Q6_K_P : JANAS_Q6_K;
        const void *wr = level ? (const void *)base : (const void *)w;
        size_t rb = level ? NB * sizeof(*base) : NB * sizeof(*w);
        for (int nv = 1; nv <= NV; nv += nv < 9 ? 1 : 8) {
            for (int i = 0; i < ROWS * NV; i++)
                out[i] = -12345.0f;
            size_t done = janas_q6k_tile_rows(t, wr, rb, a1, a0, pb, x, NB,
                                              (size_t)nv, NB, ROWS, out, ROWS);
            CHECK(done == (nv >= 12 ? 24u : 0u), "q6k tiles: %zu rows", done);
            for (size_t r = 0; r < done; r++)
                for (int v = 0; v < nv; v++) {
                    float one = level
                                    ? janas_q6kp_dot(base + r * NB,
                                                     a1 ? a1 + r * pb : NULL,
                                                     a0 ? a0 + r * pb : NULL,
                                                     x + v * NB, NB)
                                    : janas_q6k_dot(w + r * NB, x + v * NB, NB);
                    CHECK(out[r + ROWS * v] == one,
                          "q6k tiles level %d nv %d r %zu v %d: %g vs %g",
                          level, nv, r, v, out[r + ROWS * v], one);
                }
        }
    }
    free(w);
    free(base);
    free(p1);
    free(p0);
    free(x);
    free(xf);
    free(out);
}

/* janas_q6kpn_axpy at every level: with h = 1 on zeros the very floats
   janas_q6kp_dequantize gives, whatever planes lie below the level; with
   any h on any y, y plus h times them. */
static void test_q6kpn_axpy(void)
{
    enum { NB = 16, N = NB * JANAS_QK };
    struct janas_block_q6k *w = malloc(NB * sizeof(*w));
    struct janas_block_q6kp *base = malloc(NB * sizeof(*base));
    uint8_t *p1 = malloc(NB * JANAS_Q6KP_PLANE);
    uint8_t *p0 = malloc(NB * JANAS_Q6KP_PLANE);
    float *ref = malloc(N * sizeof(float)), *y = malloc(N * sizeof(float));
    float *y0 = malloc(N * sizeof(float));

    for (int trial = 0; trial < 20; trial++) {
        unsigned char *p = (unsigned char *)w;
        for (size_t i = 0; i < NB * sizeof(*w); i++)
            p[i] = (unsigned char)rnd();
        for (int b = 0; b < NB; b++)
            w[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
        janas_q6k_to_planes(w, NB, base, p1, p0);
        for (int level = 0; level <= 3; level++) {
            const uint8_t *a1 = level >= 2 ? p1 : NULL;
            const uint8_t *a0 = level >= 3 ? p0 : NULL;
            if (level)
                janas_q6kp_dequantize(base, a1, a0, ref, N);
            for (int i = 0; i < N; i++) {
                if (!level)
                    ref[i] = 0.0f;
                y[i] = 0.0f;
            }
            janas_q6kpn_axpy(base, p1, p0, level, 1.0f, y, N);
            int bad = 0;
            for (int i = 0; i < N; i++)
                bad += y[i] != ref[i];
            CHECK(!bad, "q6kpn axpy level %d, h = 1: %d of %d floats differ",
                  level, bad, N);
            float h = rndf(-2.0f, 2.0f);
            for (int i = 0; i < N; i++)
                y0[i] = y[i] = rndf(-1.0f, 1.0f);
            janas_q6kpn_axpy(base, a1, a0, level, h, y, N);
            bad = 0;
            for (int i = 0; i < N; i++) {
                float want = y0[i] + h * ref[i];
                bad += fabsf(y[i] - want) > 1e-6f * (1.0f + fabsf(want));
            }
            CHECK(!bad, "q6kpn axpy level %d, h = %g: %d of %d floats off",
                  level, h, bad, N);
            /* the kernel in use against the reference, bit for bit */
            float *yr = ref;
            for (int i = 0; i < N; i++)
                yr[i] = y0[i] = y[i] = rndf(-1.0f, 1.0f);
            janas_q6kpn_axpy(base, a1, a0, level, h, y, N);
            janas_q6kpn_axpy_ref(base, a1, a0, level, h, yr, N);
            CHECK(!memcmp(y, yr, N * sizeof(float)),
                  "q6kpn axpy level %d: kernel and reference differ", level);
        }
    }
    free(w);
    free(base);
    free(p1);
    free(p0);
    free(ref);
    free(y);
    free(y0);
}

/* janas_q6kpn_chunk: the kernel in use against the reference, bit for bit,
   on chunks of 1 to 128 neurons at random levels (JANAS_KERNELS=avx2 tries
   the AVX2 one on a machine with AVX-VNNI); and the reference against the
   float axpy, the same products with factors of eight bits. */
static void test_q6kpn_chunk(void)
{
    enum { NN = 160, NB = 4 };
    struct janas_block_q6k *w = malloc(NN * NB * sizeof(*w));
    struct janas_block_q6kp *base = malloc(NN * NB * sizeof(*base));
    uint8_t *p1 = malloc(NN * NB * JANAS_Q6KP_PLANE);
    uint8_t *p0 = malloc(NN * NB * JANAS_Q6KP_PLANE);
    unsigned char *p = (unsigned char *)w;
    for (size_t i = 0; i < NN * NB * sizeof(*w); i++)
        p[i] = (unsigned char)rnd();
    for (int b = 0; b < NN * NB; b++)
        w[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
    janas_q6k_to_planes(w, NN * NB, base, p1, p0);
    double worst = 0;
    for (int trial = 0; trial < 60; trial++) {
        uint32_t js[JANAS_PN_CHUNK], perm[NN];
        uint8_t lv[JANAS_PN_CHUNK];
        float h[JANAS_PN_CHUNK], y0[JANAS_QK], y1[JANAS_QK], y2[JANAS_QK],
            ya[JANAS_QK];
        uint32_t n = trial < 3   ? (uint32_t)trial + 1
                     : trial < 6 ? JANAS_PN_CHUNK - (uint32_t)trial + 3
                                 : 1 + rnd() % JANAS_PN_CHUNK;
        for (uint32_t i = 0; i < NN; i++)
            perm[i] = i;
        for (uint32_t i = 0; i < n; i++) { /* distinct neurons */
            uint32_t k = i + rnd() % (NN - i), t = perm[i];
            perm[i] = perm[k];
            perm[k] = t;
            js[i] = perm[i];
            lv[i] = (uint8_t)(1 + rnd() % 3);
            float g = rndf(-1, 1);
            h[i] = trial % 2 ? g * g * g * 4.0f : rndf(-2.0f, 2.0f);
        }
        size_t b = rnd() % NB;
        for (int v = 0; v < JANAS_QK; v++)
            ya[v] = y0[v] = y1[v] = y2[v] = rndf(-1.0f, 1.0f);
        janas_q6kpn_chunk(base, p1, p0, NN, js, lv, h, n, b, y1);
        janas_q6kpn_chunk_ref(base, p1, p0, NN, js, lv, h, n, b, y2);
        CHECK(!memcmp(y1, y2, sizeof(y1)),
              "q6kpn chunk of %u: kernel and reference differ", n);
        for (uint32_t i = 0; i < n; i++) {
            size_t at = b * NN + js[i]; /* by block of outputs */
            janas_q6kpn_axpy_ref(base + at, p1 + at * JANAS_Q6KP_PLANE,
                                 p0 + at * JANAS_Q6KP_PLANE, lv[i], h[i], ya,
                                 JANAS_QK);
        }
        double e = 0, sg = 0;
        for (int v = 0; v < JANAS_QK; v++) {
            e += ((double)y2[v] - ya[v]) * (y2[v] - ya[v]);
            sg += ((double)ya[v] - y0[v]) * (ya[v] - y0[v]);
        }
        double rel = sg > 0 ? sqrt(e / sg) : 0;
        if (rel > worst)
            worst = rel;
        CHECK(rel < 0.02, "q6kpn chunk of %u: %.4f off the float products", n,
              rel);
    }
    printf("q6kpn chunk: factors of eight bits, at most %.4f off the float "
           "products\n",
           worst);
    free(w);
    free(base);
    free(p1);
    free(p0);
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
 * 6-bit sub-block scales allow, and below a plain min/max rounding - also
 * for the fast way, which takes each sub-block's range as it is.
 */
static void test_quantize_q4k(void)
{
    enum { N = 256 * 64 };
    float *x = malloc(N * sizeof(float)), *y = malloc(N * sizeof(float));
    struct janas_block_q4k *q = malloc(N / JANAS_QK * sizeof(*q));
    for (int k2 = 0; k2 < 4; k2++) {
        int kind = k2 & 1, fast = k2 >> 1;
        double var = 0, err = 0, naive = 0;
        for (int i = 0; i < N; i++) {
            /* sum of uniforms ~ Gaussian; its cube is heavy-tailed */
            float g = rndf(-1, 1) + rndf(-1, 1) + rndf(-1, 1) + rndf(-1, 1);
            x[i] = 0.02f * (kind ? g * g * g : g);
            var += (double)x[i] * x[i];
        }
        if (fast)
            janas_q4k_quantize_fast(x, q, N);
        else
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
        printf("q4k quantize%s, %s weights: relative rms error %.4f (min/max "
               "rounding %.4f)\n",
               fast ? " fast" : "", kind ? "heavy-tailed" : "Gaussian", rel,
               rel_naive);
        CHECK(rel < (kind ? 0.20 : 0.11) && rel < rel_naive,
              "q4k quantize %d%s: relative error %.4f, naive %.4f", kind,
              fast ? " fast" : "", rel, rel_naive);
    }
    free(x);
    free(y);
    free(q);
}

/* janas_q6k_quantize: every weight read back as the integer nearest to it
   under the scales the block keeps (so its bits sit where Q6_K reads them),
   and less error than the plain way: each group of 16 at its largest
   magnitude over 31, those scales then to the block's eight bits. */
static void test_quantize_q6k(void)
{
    enum { N = 256 * 64 };
    float *x = malloc(N * sizeof(float)), *y = malloc(N * sizeof(float));
    struct janas_block_q6k *q = malloc(N / JANAS_QK * sizeof(*q));
    for (int kind = 0; kind < 2; kind++) {
        double var = 0, err = 0, plain = 0, plain_f = 0;
        for (int i = 0; i < N; i++) {
            /* sum of uniforms ~ Gaussian; its cube is heavy-tailed */
            float g = rndf(-1, 1) + rndf(-1, 1) + rndf(-1, 1) + rndf(-1, 1);
            x[i] = 0.02f * (kind ? g * g * g : g);
            var += (double)x[i] * x[i];
        }
        janas_q6k_quantize(x, q, N);
        janas_q6k_dequantize(q, y, N);
        int far = 0;
        for (int i = 0; i < N; i++) {
            const struct janas_block_q6k *b = &q[i / JANAS_QK];
            float a = janas_fp16_to_fp32(b->d) * b->scales[i % JANAS_QK / 16];
            int k = a != 0.0f ? (int)lrintf(x[i] / a) : 0;
            k = k < -32 ? -32 : k > 31 ? 31 : k;
            far += y[i] != a * k;
            err += ((double)y[i] - x[i]) * (y[i] - x[i]);
        }
        CHECK(!far, "q6k quantize %d: %d weights not the nearest", kind, far);
        for (int b = 0; b < N / JANAS_QK; b++) {
            const float *xb = x + b * JANAS_QK;
            float a[16], amax = 0;
            for (int s = 0; s < 16; s++) {
                a[s] = 0;
                for (int k = 0; k < 16; k++)
                    a[s] = fmaxf(a[s], fabsf(xb[16 * s + k]) / 31);
                amax = fmaxf(amax, a[s]);
            }
            float d = janas_fp16_to_fp32(janas_fp32_to_fp16(amax / 127));
            for (int v = 0; v < JANAS_QK; v++) {
                float af = a[v / 16], a8 = d * lrintf(af / d);
                int kf = af > 0 ? (int)lrintf(xb[v] / af) : 0;
                int k8 = a8 > 0 ? (int)lrintf(xb[v] / a8) : 0;
                kf = kf < -32 ? -32 : kf > 31 ? 31 : kf;
                k8 = k8 < -32 ? -32 : k8 > 31 ? 31 : k8;
                plain_f += ((double)af * kf - xb[v]) * (af * kf - xb[v]);
                plain += ((double)a8 * k8 - xb[v]) * (a8 * k8 - xb[v]);
            }
        }
        double rel = sqrt(err / var), rel_plain = sqrt(plain / var);
        printf("q6k quantize, %s weights: relative rms error %.5f (plain "
               "%.5f, with float scales %.5f)\n",
               kind ? "heavy-tailed" : "Gaussian", rel, rel_plain,
               sqrt(plain_f / var));
        CHECK(rel < rel_plain,
              "q6k quantize %d: relative error %.5f, plain "
              "%.5f",
              kind, rel, rel_plain);
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
    int bad = 0, compared = 0;
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
            compared += 1024;
        }
#endif
    }
    /* without AVX2 and FMA there is nothing to compare it with: say so,
       rather than pass as if 1000 rounds had agreed (issue #26) */
    if (compared)
        printf("silu_mul_det: scalar against AVX2 on %d values\n", compared);
    else
        printf("silu_mul_det: no AVX2 with FMA here, scalar not compared\n");
    CHECK(bad == 0, "silu_mul_det: %d scalar/AVX2 differences of %d", bad,
          compared);
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
                    double e = (double)wf[b * JANAS_QK + i] * act(x, b, i, t);
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
                    double e = (double)wf[b * JANAS_QK + i] * act(x, b, i, t);
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
/*
 * MXFP4: the reference against a float product over the dequantized
 * weights, the kernels (single, 1-17 vectors, rows) equal to it to the bit,
 * the scale exactly 2^(e - 128) for every byte, and the same bits as its
 * exact rewrite into Q8_0 (janas_mxfp4_to_q8_0, gguf2jns's for resident
 * tensors), whose products take the activations the same way.
 */
static void test_dot_mxfp4(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    for (int e = 0; e < 256; e++)
        CHECK(janas_e8m0_half((uint8_t)e) == ldexpf(1.0f, e - 128),
              "e8m0 %d: %g", e, janas_e8m0_half((uint8_t)e));
    struct janas_block_mxfp4 *w = malloc(NB * 8 * sizeof(*w));
    struct janas_block_q8_0 *w8 = malloc(NB * 8 * sizeof(*w8));
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float)), *wf = malloc(N * sizeof(float));
    float rows[4];
    for (int trial = 0; trial < 100; trial++) {
        unsigned char *pw = (unsigned char *)w;
        for (size_t i = 0; i < NB * 8 * sizeof(*w); i++)
            pw[i] = (unsigned char)rnd();
        /* the powers of two of real weights (2^-10 to 2^3), and in some
           trials the subnormal and the largest ones as well */
        for (int b = 0; b < NB * 8; b++)
            w[b].e = (uint8_t)(118 + rnd() % 14);
        if (trial % 4 == 3) {
            w[0].e = 0;
            w[1].e = 1;
            w[2].e = 2;
            w[3].e = 254;
        }
        for (int v = 0; v < NV; v++) {
            for (int i = 0; i < N; i++)
                xf[i] = rndf(-1.0f, 1.0f);
            janas_q8k_quantize(xf, x + v * NB, N);
        }
        janas_mxfp4_dequantize(w, wf, N);
        double expect = 0.0, mag = 0.0;
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < JANAS_QK; i++) {
                double p =
                    (double)wf[b * JANAS_QK + i] * act(x, b, i, JANAS_MXFP4);
                expect += p;
                mag += fabs(p);
            }
        float ref = janas_mxfp4_dot_ref(w, x, NB);
        if (trial % 4 != 3) /* 2^126 times the rest: no float to compare */
            CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "mxfp4 ref %g vs %g",
                  ref, expect);
        float fast = janas_dot(JANAS_MXFP4, w, x, NB);
        CHECK(fast == ref || (isnan(fast) && isnan(ref)),
              "mxfp4 fast %g vs ref %g", fast, ref);
        for (int nv = 1; nv <= NV; nv++) {
            float out[NV];
            janas_dot_multi(JANAS_MXFP4, w, x, NB, (size_t)nv, NB, out, 1);
            for (int v = 0; v < nv; v++) {
                float r = janas_mxfp4_dot_ref(w, x + v * NB, NB);
                CHECK(out[v] == r || (isnan(out[v]) && isnan(r)),
                      "mxfp4 multi nv %d v %d: %g vs %g", nv, v, out[v], r);
            }
        }
        /* four rows of two super-blocks each, the matrix read row by row */
        janas_dot_rows(JANAS_MXFP4, w, 2 * 8 * sizeof(*w), x, 2, 4, rows);
        for (int r = 0; r < 4; r++) {
            float one = janas_mxfp4_dot_ref(w + 16 * r, x, 2);
            CHECK(rows[r] == one || (isnan(rows[r]) && isnan(one)),
                  "mxfp4 rows %d: %g vs %g", r, rows[r], one);
        }
        if (trial % 4 == 3)
            continue;
        /* the exact rewrite into Q8_0: the same products, bit for bit */
        size_t tiny = 0;
        CHECK(janas_mxfp4_to_q8_0((const uint8_t *)w, w8, NB * 8, &tiny) == 0 &&
                  tiny == 0,
              "mxfp4 to q8_0 (%zu tiny)", tiny);
        float q8 = janas_dot(JANAS_Q8_0, w8, x, NB);
        CHECK(q8 == fast, "mxfp4 %g vs its q8_0 rewrite %g", fast, q8);
    }
    free(w);
    free(w8);
    free(x);
    free(xf);
    free(wf);
}

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
                    double e = (double)wf[b * JANAS_QK + i] * act(x, b, i, t);
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

/* Random bytes for a row of n weights of a type in blocks of 32 or 64, each
   block's scale a finite fp16 (sizes from janas_qtype_row_bytes). */
static void random_blocks(int type, unsigned char *w, size_t n)
{
    size_t bytes = janas_qtype_row_bytes(type, n);
    for (size_t i = 0; i < bytes; i++)
        w[i] = (unsigned char)rnd();
    size_t size = type == JANAS_Q2_0   ? sizeof(struct janas_block_q2_0)
                  : type == JANAS_Q5_0 ? sizeof(struct janas_block_q5_0)
                  : type == JANAS_Q8_0 ? sizeof(struct janas_block_q8_0)
                                       : sizeof(struct janas_block_iq4nl);
    for (size_t b = 0; b < bytes / size; b++) {
        uint16_t d = janas_fp32_to_fp16(rndf(-0.01f, 0.01f));
        memcpy(w + b * size, &d, 2);
    }
}

/*
 * janas_dot_multi_rows on Q2_0 and the K types, on long rows: the vectors
 * reordered (Q2_0) or their scales and sums made ready (Q4_K, Q5_K, Q6_K) a few
 * at a time, or not at all where the rows are too long (130 groups), 1-65
 * vectors; each result equal to janas_dot's to the bit, and Q2_0's to the
 * reference.
 */
static void test_multi_rows_long(void)
{
    /* 27 rows: Q4_K's tiles take 24 (sixteen on 512 bits where AVX-512
       VNNI is, q4k_gemm_z, then eight, q4k_gemm), the products the rest;
       up to 65 vectors, then 300: past the tiles' 256 tokens at a time */
    enum { ROWS = 27, NV = 300 };
    static const size_t lens[] = {2, 48, 130};
    static const int types[] = {
        JANAS_Q2_0,    JANAS_Q4_K,    JANAS_Q5_K,   JANAS_Q6_K,  JANAS_Q8_0,
        JANAS_Q4_0,    JANAS_Q5_0,    JANAS_IQ4_NL, JANAS_MXFP4, JANAS_Q4_1,
        JANAS_Q5_1,    JANAS_Q3_K,    JANAS_IQ4_XS, JANAS_Q2_K,  JANAS_IQ3_S,
        JANAS_IQ3_XXS, JANAS_IQ2_XXS, JANAS_IQ2_XS, JANAS_IQ2_S, JANAS_IQ1_S,
        JANAS_IQ1_M};
    for (unsigned ti = 0; ti < sizeof(types) / sizeof(*types); ti++)
        for (unsigned li = 0; li < sizeof(lens) / sizeof(*lens); li++) {
            int t = types[ti];
            size_t nb = lens[li], n = nb * JANAS_QK;
            size_t rb = janas_qtype_row_bytes(t, n);
            unsigned char *w = malloc(ROWS * rb);
            struct janas_block_q8k *x = malloc(NV * nb * sizeof(*x));
            float *xf = malloc(n * sizeof(float)), out[ROWS * NV];
            for (int r = 0; r < ROWS; r++) {
                if (t == JANAS_Q2_0 || t == JANAS_Q8_0 || t == JANAS_Q4_0 ||
                    t == JANAS_Q5_0 || t == JANAS_IQ4_NL)
                    random_blocks(t, w + r * rb, n);
                else if (t == JANAS_MXFP4 || t == JANAS_Q4_1 ||
                         t == JANAS_Q5_1) {
                    /* blocks of 32: MXFP4's power of two, Q4_1's and
                       Q5_1's d and m, finite */
                    size_t bs = janas_qtype_row_bytes(t, 32);
                    unsigned char *q = w + r * rb;
                    for (size_t i = 0; i < rb; i++)
                        q[i] = (unsigned char)rnd();
                    for (size_t i = 0; i < rb; i += bs) {
                        if (t == JANAS_MXFP4) {
                            q[i] = (unsigned char)(118 + rnd() % 12);
                            continue;
                        }
                        uint16_t d[2] = {
                            janas_fp32_to_fp16(rndf(-0.01f, 0.01f)),
                            janas_fp32_to_fp16(rndf(-0.01f, 0.01f))};
                        memcpy(q + i, d, 4);
                    }
                } else
                    for (size_t b = 0; b < nb; b++) {
                        /* Q4_K and Q5_K: d and dmin first, the IQ types d;
                           Q6_K and Q3_K: d last, Q2_K d and dmin; IQ1_M: d
                           in the top nibbles of its four scale words */
                        size_t bs = janas_qtype_block_size(t);
                        unsigned char *q = w + r * rb + b * bs;
                        for (size_t i = 0; i < bs; i++)
                            q[i] = (unsigned char)rnd();
                        uint16_t d[2] = {
                            janas_fp32_to_fp16(rndf(-0.01f, 0.01f)),
                            janas_fp32_to_fp16(rndf(-0.01f, 0.01f))};
                        if (t == JANAS_Q6_K || t == JANAS_Q3_K)
                            memcpy(q + bs - 2, d, 2);
                        else if (t == JANAS_Q2_K)
                            memcpy(q + bs - 4, d, 4);
                        else if (t == JANAS_IQ1_M) {
                            uint16_t sc[4];
                            memcpy(sc, q + bs - 8, 8);
                            for (int k = 0; k < 4; k++)
                                sc[k] = (uint16_t)((sc[k] & 0x0fff) |
                                                   (d[0] >> 4 * k & 15) << 12);
                            memcpy(q + bs - 8, sc, 8);
                        } else
                            memcpy(q, d, 4);
                    }
            }
            for (int v = 0; v < NV; v++) {
                for (size_t i = 0; i < n; i++)
                    xf[i] = rndf(-1.0f, 1.0f);
                janas_q8k_quantize(xf, x + v * nb, n);
            }
            for (int nv = 1; nv <= NV; nv = nv < 9     ? nv + 1
                                            : nv < 65  ? nv + 8
                                            : nv == 65 ? NV
                                                       : NV + 1) {
                janas_dot_multi_rows(t, w, rb, x, nb, (size_t)nv, nb, ROWS, out,
                                     ROWS);
                for (int v = 0; v < nv; v++)
                    for (int r = 0; r < ROWS; r++) {
                        float one = janas_dot(t, w + r * rb, x + v * nb, nb);
                        CHECK(out[r + ROWS * v] == one,
                              "type %d %zu groups nv %d r %d v %d: %g vs %g", t,
                              nb, nv, r, v, out[r + ROWS * v], one);
                        if (t == JANAS_Q2_0) {
                            float ref = janas_q2_0_dot_ref(
                                (const void *)(w + r * rb), x + v * nb, nb);
                            CHECK(one == ref, "q2_0 %zu groups: %g vs %g", nb,
                                  one, ref);
                        }
                    }
            }
            float one[ROWS];
            janas_dot_rows(t, w, rb, x, nb, ROWS, one);
            for (int r = 0; r < ROWS; r++)
                CHECK(one[r] == janas_dot(t, w + r * rb, x, nb),
                      "type %d %zu groups rows %d", t, nb, r);
            free(w);
            free(x);
            free(xf);
        }
}

/*
 * Q5_0 and Q2_0: the reference against a float product over the
 * dequantized weights, the kernels (single, 1-17 vectors, rows) equal to it
 * to the bit; the values by hand as ggml's dequantize_row_q5_0 and
 * dequantize_row_q2_0 give them.
 */
static void test_dot_q5_0_q2_0(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    static const int types[] = {JANAS_Q5_0, JANAS_Q2_0};
    unsigned char *w = malloc(NB * janas_qtype_block_size(JANAS_Q5_0));
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float)), *wf = malloc(N * sizeof(float));
    float rows[4];
    for (int trial = 0; trial < 100; trial++)
        for (unsigned k = 0; k < sizeof(types) / sizeof(*types); k++) {
            int t = types[k];
            const char *name = t == JANAS_Q5_0 ? "q5_0" : "q2_0";
            random_blocks(t, w, N);
            for (int v = 0; v < NV; v++) {
                for (int i = 0; i < N; i++)
                    xf[i] = rndf(-1.0f, 1.0f);
                janas_q8k_quantize(xf, x + v * NB, N);
            }
            janas_dequantize(t, w, wf, N);
            double expect = 0.0, mag = 0.0;
            for (int b = 0; b < NB; b++)
                for (int i = 0; i < JANAS_QK; i++) {
                    double e = (double)wf[b * JANAS_QK + i] * act(x, b, i, t);
                    expect += e;
                    mag += fabs(e);
                }
            float ref = t == JANAS_Q5_0
                            ? janas_q5_0_dot_ref((const void *)w, x, NB)
                            : janas_q2_0_dot_ref((const void *)w, x, NB);
            CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "%s ref %g vs %g",
                  name, ref, expect);
            float fast = janas_dot(t, w, x, NB);
            CHECK(fast == ref, "%s fast %g vs ref %g", name, fast, ref);
            for (int nv = 1; nv <= NV; nv++) {
                float out[NV];
                janas_dot_multi(t, w, x, NB, (size_t)nv, NB, out, 1);
                for (int v = 0; v < nv; v++) {
                    float r = t == JANAS_Q5_0
                                  ? janas_q5_0_dot_ref((const void *)w,
                                                       x + v * NB, NB)
                                  : janas_q2_0_dot_ref((const void *)w,
                                                       x + v * NB, NB);
                    CHECK(out[v] == r, "%s multi nv %d v %d: %g vs %g", name,
                          nv, v, out[v], r);
                }
            }
            /* four rows of two groups each, the matrix read row by row */
            size_t gb = janas_qtype_block_size(t);
            janas_dot_rows(t, w, 2 * gb, x, 2, 4, rows);
            for (int r = 0; r < 4; r++) {
                float one = janas_dot(t, w + 2 * gb * (size_t)r, x, 2);
                CHECK(rows[r] == one, "%s rows %d: %g vs %g", name, r, rows[r],
                      one);
            }
            /* the same four rows on 1-17 vectors at once */
            for (int nv = 1; nv <= NV; nv++) {
                float out[4 * NV];
                janas_dot_multi_rows(t, w, 2 * gb, x, NB, (size_t)nv, 2, 4, out,
                                     4);
                for (int r = 0; r < 4; r++)
                    for (int v = 0; v < nv; v++) {
                        float one =
                            janas_dot(t, w + 2 * gb * (size_t)r, x + v * NB, 2);
                        CHECK(out[r + 4 * v] == one,
                              "%s multi rows nv %d r %d v %d: %g vs %g", name,
                              nv, r, v, out[r + 4 * v], one);
                    }
            }
        }
    /* by hand. Q5_0: weight 17 takes bit 17 of qh and the high nibble of
       byte 1, weight 1 the low nibble alone, less 16 */
    struct janas_block_q5_0 h5 = {0};
    h5.d = janas_fp32_to_fp16(1.0f);
    h5.qh[2] = 0x02; /* bit 17 */
    h5.qs[1] = 0x37; /* weight 1: 7; weight 17: 3 */
    float y[64];
    janas_q5_0_dequantize(&h5, y, 32);
    CHECK(y[17] == 19.0f - 16.0f && y[1] == 7.0f - 16.0f && y[0] == -16.0f,
          "q5_0 by hand: %g %g %g", y[17], y[1], y[0]);
    /* Q2_0: weight i in bits 2 (i % 4) of byte i / 4, 00 = -1, 11 = +2 */
    struct janas_block_q2_0 h2 = {0};
    h2.d = janas_fp32_to_fp16(0.5f);
    h2.qs[0] = 0xe4;  /* weights 0-3: 0, 1, 2, 3 */
    h2.qs[15] = 0x1b; /* weights 60-63: 3, 2, 1, 0 */
    janas_q2_0_dequantize(&h2, y, 64);
    CHECK(y[0] == -0.5f && y[1] == 0.0f && y[2] == 0.5f && y[3] == 1.0f &&
              y[60] == 1.0f && y[61] == 0.5f && y[62] == 0.0f &&
              y[63] == -0.5f && y[4] == -0.5f,
          "q2_0 by hand: %g %g %g %g / %g %g %g %g", y[0], y[1], y[2], y[3],
          y[60], y[61], y[62], y[63]);
    free(w);
    free(x);
    free(xf);
    free(wf);
}

/*
 * Rows that end inside a group of 256 (Qwen3.8-Flash-Next's experts, 640
 * wide), in the types stored in blocks of 32 or 64: the product is the whole
 * groups' plus the tail padded with zero blocks, the same bits whichever way
 * it is asked for (one row, rows, 1-17 vectors, the grouped products on a
 * pool); and it is the float product over the n weights.
 */
static void test_row_tails(void)
{
    /* 33 vectors and 37 rows: the tiles take 32 of them (from 24 vectors),
       the tails 16 rows at a time */
    enum { NMAX = 768, NBX = NMAX / JANAS_QK, NV = 33, ROWS = 37 };
    static const int types[] = {JANAS_Q2_0, JANAS_Q5_0, JANAS_IQ4_NL,
                                JANAS_Q4_0, JANAS_Q8_0};
    static const size_t ns[] = {640, 64, 320, 576, 704};
    size_t big = ROWS * janas_qtype_row_bytes(JANAS_Q8_0, NMAX);
    unsigned char *w = malloc(big);
    struct janas_block_q8k *x = malloc(NV * NBX * sizeof(*x));
    float xf[NMAX], *wf = malloc(NMAX * sizeof(float));
    float y[NV * ROWS];
    struct janas_pool *pool = janas_pool_create(3, NULL);
    CHECK(pool != NULL, "pool");
    for (unsigned k = 0; k < sizeof(types) / sizeof(*types); k++)
        for (unsigned j = 0; j < sizeof(ns) / sizeof(*ns); j++) {
            int t = types[k];
            size_t n = ns[j], nb = n / JANAS_QK, nbx = (n + 255) / 256;
            size_t rb = janas_qtype_row_bytes(t, n);
            if (t == JANAS_Q2_0 && n % 64) {
                CHECK(rb == 0, "q2_0 row of %zu: %zu bytes", n, rb);
                continue;
            }
            CHECK(rb != 0 && rb < (nb + 1) * janas_qtype_block_size(t),
                  "type %d row of %zu: %zu bytes", t, n, rb);
            for (int r = 0; r < ROWS; r++)
                random_blocks(t, w + (size_t)r * rb, n);
            /* activations past n zero, as the engine pads them */
            for (int v = 0; v < NV; v++) {
                for (size_t i = 0; i < NMAX; i++)
                    xf[i] = i < n ? rndf(-1.0f, 1.0f) : 0.0f;
                janas_q8k_quantize(xf, x + v * nbx, nbx * JANAS_QK);
            }
            /* the float product of row 0 */
            janas_dequantize(t, w, wf, n);
            double expect = 0.0, mag = 0.0;
            for (size_t i = 0; i < n; i++) {
                double e =
                    (double)wf[i] * act(x, (int)(i / 256), (int)(i % 256), t);
                expect += e;
                mag += fabs(e);
            }
            float one = janas_dot_n(t, w, x, n);
            CHECK(fabs(one - expect) <= 1e-5 * mag + 1e-6,
                  "type %d n %zu: %g vs %g", t, n, one, expect);
            /* its definition: whole groups, then the padded tail */
            _Alignas(64) unsigned char pad[8 * sizeof(struct janas_block_q8_0)];
            size_t full = nb * janas_qtype_block_size(t);
            memset(pad, 0, sizeof(pad));
            memcpy(pad, w + full, rb - full);
            float def = (nb ? janas_dot(t, w, x, nb) : 0.0f) +
                        janas_dot(t, pad, x + nb, 1);
            CHECK(one == def, "type %d n %zu: %g vs its definition %g", t, n,
                  one, def);
            janas_dot_rows_n(t, w, rb, x, n, ROWS, y);
            for (int r = 0; r < ROWS; r++) {
                float e = janas_dot_n(t, w + (size_t)r * rb, x, n);
                CHECK(y[r] == e, "type %d n %zu rows %d: %g vs %g", t, n, r,
                      y[r], e);
            }
            for (int nv = 1; nv <= NV; nv += 4) {
                float out[NV];
                janas_dot_multi_n(t, w, x, nbx, (size_t)nv, n, out, 1);
                for (int v = 0; v < nv; v++) {
                    float e = janas_dot_n(t, w, x + v * nbx, n);
                    CHECK(out[v] == e, "type %d n %zu multi %d/%d: %g vs %g", t,
                          n, v, nv, out[v], e);
                }
            }
            for (int nv = 1; nv <= NV; nv += 8) {
                janas_dot_multi_rows_n(t, w, rb, x, nbx, (size_t)nv, n, ROWS, y,
                                       ROWS);
                for (int v = 0; v < nv; v++)
                    for (int r = 0; r < ROWS; r++) {
                        float e =
                            janas_dot_n(t, w + (size_t)r * rb, x + v * nbx, n);
                        CHECK(y[v * ROWS + r] == e,
                              "type %d n %zu multi rows nv %d v %d row %d: %g "
                              "vs %g",
                              t, n, nv, v, r, y[v * ROWS + r], e);
                    }
            }
            /* the grouped products: one vector, five and NV */
            for (int nv = 1; pool && nv <= NV; nv = nv == 1   ? 5
                                                    : nv == 5 ? NV
                                                              : NV + 1) {
                struct janas_matvec_task task = {.type = t,
                                                 .w = w,
                                                 .x = x,
                                                 .y = y,
                                                 .rows = ROWS,
                                                 .cols = n,
                                                 .n_vec = (size_t)nv,
                                                 .x_stride = nbx,
                                                 .y_stride = ROWS};
                janas_matvec_q4k_group(pool, &task, 1);
                for (int v = 0; v < nv; v++)
                    for (int r = 0; r < ROWS; r++) {
                        float e =
                            janas_dot_n(t, w + (size_t)r * rb, x + v * nbx, n);
                        CHECK(y[v * ROWS + r] == e,
                              "type %d n %zu group nv %d v %d row %d: %g vs "
                              "%g",
                              t, n, nv, v, r, y[v * ROWS + r], e);
                    }
            }
        }
    janas_pool_destroy(pool);
    free(w);
    free(x);
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
/* Q8_0 rows on float vectors (bf16.h): one row, a row on four vectors and
   four rows on one vector, each the bits of the order bf16.h states, scales
   subnormal and zero among them */
static float q8w_ref(const unsigned char *w, const float *x, uint32_t n)
{
    float acc[8] = {0};
    for (uint32_t b = 0; b < n / 32; b++, w += 34, x += 32) {
        const int8_t *q = (const int8_t *)(w + 2);
        uint16_t h;
        memcpy(&h, w, 2);
        float d = janas_fp16_to_fp32(h), bs[8] = {0};
        for (int g = 0; g < 4; g++)
            for (int k = 0; k < 8; k++)
                bs[k] = fmaf((float)q[8 * g + k], x[8 * g + k], bs[k]);
        for (int k = 0; k < 8; k++)
            acc[k] = fmaf(bs[k], d, acc[k]);
    }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) +
           ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

static void test_q8w(void)
{
    enum { N = 320, ROW = N / 32 * 34 };
    unsigned char *w = malloc(4 * ROW);
    float *x = malloc(4 * N * sizeof(float));
    for (int rep = 0; rep < 20; rep++) {
        for (int i = 0; i < 4 * ROW; i++)
            w[i] = (unsigned char)rnd();
        for (int b = 0; b < 4 * N / 32; b++) {
            float d = rndf(0.0001f, 0.05f);
            uint16_t h = janas_fp32_to_fp16(d);
            if (b % 7 == 3)
                h = (uint16_t)(rnd() & 0x83ff); /* subnormal or zero */
            memcpy(w + b * 34, &h, 2);
        }
        for (int i = 0; i < 4 * N; i++)
            x[i] = rndf(-2.0f, 2.0f);
        float o[4], e;
        const float *xs[4] = {x, x + N, x + 2 * N, x + 3 * N};
        janas_q8w_dot4(w, xs, N, o);
        for (int t = 0; t < 4; t++) {
            e = q8w_ref(w, xs[t], N);
            CHECK(!memcmp(&o[t], &e, 4), "q8w dot4 vector %d: %g != %g", t,
                  o[t], e);
        }
        janas_q8w_dot4r(w, ROW, x, N, o);
        for (int r = 0; r < 4; r++) {
            e = q8w_ref(w + r * ROW, x, N);
            CHECK(!memcmp(&o[r], &e, 4), "q8w dot4r row %d: %g != %g", r, o[r],
                  e);
            float one = janas_q8w_dot(w + r * ROW, x, N);
            CHECK(!memcmp(&one, &e, 4), "q8w dot row %d: %g != %g", r, one, e);
        }
    }
    free(w);
    free(x);
}

/* bf16 rows on one vector four at a time (bf16.h) against one at a time,
   lengths with and without a tail; rounding to bf16 against the scalar
   rule, ties, infinities and NaNs among the values */
static void test_bf16_rows(void)
{
    enum { N = 333, ROW = 340 };
    uint16_t *w = malloc(4 * ROW * sizeof(uint16_t));
    float *x = malloc(ROW * sizeof(float));
    for (int rep = 0; rep < 20; rep++) {
        for (int i = 0; i < 4 * ROW; i++)
            w[i] = (uint16_t)(janas_f2u(rndf(-0.1f, 0.1f)) >> 16);
        for (int i = 0; i < ROW; i++)
            x[i] = rndf(-2.0f, 2.0f);
        uint32_t n = rep % 2 ? N : 320;
        float o[4];
        janas_bf16_dot4r(w, ROW, x, n, o);
        for (int r = 0; r < 4; r++) {
            float e = janas_bf16_dot(w + r * ROW, x, n);
            CHECK(!memcmp(&o[r], &e, 4), "bf16 dot4r row %d n %u: %g != %g", r,
                  n, o[r], e);
        }
    }
    enum { M = 4099 };
    float *a = malloc(M * sizeof(float)), *b = malloc(M * sizeof(float));
    for (int i = 0; i < M; i++) {
        uint32_t u = rnd();
        if (i % 5 == 1)
            u = (u & 0xffff0000u) | 0x8000u; /* a tie */
        if (i % 97 == 3)
            u = 0x7f800000u | (u & 0x80000000u); /* infinity */
        if (i % 89 == 5)
            u = 0x7f800001u | (u & 0x807fffffu); /* NaN */
        a[i] = janas_u2f(u);
    }
    janas_round_bf16_n(a, b, M);
    for (int i = 0; i < M; i++) {
        uint32_t u = janas_f2u(a[i]), e = u;
        if ((u & 0x7fffffffu) <= 0x7f800000u)
            e = (u + (0x7fffu + ((u >> 16) & 1u))) & 0xffff0000u;
        CHECK(janas_f2u(b[i]) == e, "round bf16 %08x: %08x != %08x", u,
              janas_f2u(b[i]), e);
    }
    free(w);
    free(x);
    free(a);
    free(b);
}

static void test_dequantize(void)
{
    enum { NB = 8, N = NB * JANAS_QK };
    static const int types[] = {JANAS_Q4_K,  JANAS_Q5_K,   JANAS_Q6_K,
                                JANAS_Q8_0,  JANAS_IQ4_NL, JANAS_IQ4_XS,
                                JANAS_Q4_0,  JANAS_Q5_1,   JANAS_Q4_1,
                                JANAS_MXFP4, JANAS_Q5_0,   JANAS_Q2_0};
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
            case JANAS_MXFP4: { /* eight blocks cover one super-block */
                struct janas_block_mxfp4 *q =
                    (struct janas_block_mxfp4 *)w + i * 8;
                for (int j = 0; j < 8; j++)
                    q[j].e = (uint8_t)(118 + rnd() % 14);
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
            case JANAS_Q5_0: {
                struct janas_block_q5_0 *q =
                    (struct janas_block_q5_0 *)w + i * 8;
                for (int j = 0; j < 8; j++)
                    q[j].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
                break;
            }
            case JANAS_Q2_0: { /* four blocks cover one super-block */
                struct janas_block_q2_0 *q =
                    (struct janas_block_q2_0 *)w + i * 4;
                for (int j = 0; j < 4; j++)
                    q[j].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
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
        case JANAS_MXFP4:
            janas_mxfp4_dequantize((const void *)w, a, N);
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
        case JANAS_Q5_0:
            janas_q5_0_dequantize((const void *)w, a, N);
            break;
        case JANAS_Q2_0:
            janas_q2_0_dequantize((const void *)w, a, N);
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
        for (int k = 0; k < JANAS_QK / 32; k++)
            y[b].bsums2[k] =
                (int16_t)(y[b].bsums[2 * k] + y[b].bsums[2 * k + 1]);
        for (int k = 0; k < 8; k++) {
            float m = 0.0f;
            for (int i = 0; i < 32; i++)
                m = fmaxf(m, fabsf(x[32 * k + i]));
            float dk = det ? m * 0.007874015718698502f : m / 127.0f;
            float ik =
                dk > 0.0f ? (det ? janas_recip_det(dk) : 1.0f / dk) : 0.0f;
            y[b].d32[k] = dk;
            for (int i = 0; i < 32; i++) {
                float q = nearbyintf(x[32 * k + i] * ik);
                y[b].qs32[32 * k + i] =
                    (int8_t)(q > 127.0f ? 127 : (q < -127.0f ? -127 : q));
            }
        }
        for (int g = 0; g < JANAS_QK / 16; g++) {
            int s = 0;
            for (int i = 0; i < 16; i++)
                s += y[b].qs32[g * 16 + i];
            y[b].bsums32[g] = (int16_t)s;
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

static void test_dot_iq3xxs(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    struct janas_block_iq3xxs *w = malloc(NB * sizeof(*w));
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
        janas_iq3xxs_dequantize(w, wf, N);
        janas_dequantize(JANAS_IQ3_XXS, w, w2, N);
        CHECK(memcmp(wf, w2, N * sizeof(float)) == 0,
              "iq3xxs: janas_dequantize differs from its unpacker");
        double expect = 0.0, mag = 0.0;
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < JANAS_QK; i++) {
                double e = (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                expect += e;
                mag += fabs(e);
            }
        float ref = janas_iq3xxs_dot_ref(w, x, NB);
        CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "iq3xxs ref %g vs %g",
              ref, expect);
        CHECK(janas_dot(JANAS_IQ3_XXS, w, x, NB) == ref, "iq3xxs fast vs ref");
        for (int nv = 1; nv <= NV; nv++) {
            float out[NV];
            janas_dot_multi(JANAS_IQ3_XXS, w, x, NB, (size_t)nv, NB, out, 1);
            for (int v = 0; v < nv; v++)
                CHECK(out[v] == janas_iq3xxs_dot_ref(w, x + v * NB, NB),
                      "iq3xxs multi nv %d v %d", nv, v);
        }
    }
    free(w);
    free(x);
    free(xf);
    free(wf);
    free(w2);
}

/* IQ2_XXS, IQ2_XS, IQ2_S: the reference against a float product over the
   dequantized weights, the kernels (single, 1-17 vectors, rows) equal to it
   to the bit */
static void test_dot_iq2(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    static const int types[] = {JANAS_IQ2_XXS, JANAS_IQ2_XS, JANAS_IQ2_S};
    static const char *const names[] = {"iq2_xxs", "iq2_xs", "iq2_s"};
    size_t big = NB * janas_qtype_block_size(JANAS_IQ2_S);
    unsigned char *w = malloc(big);
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float)), *wf = malloc(N * sizeof(float));
    float rows[4];
    for (int trial = 0; trial < 100; trial++)
        for (int k = 0; k < 3; k++) {
            int t = types[k];
            size_t bs = janas_qtype_block_size(t);
            for (size_t i = 0; i < NB * bs; i++)
                w[i] = (unsigned char)rnd();
            for (int b = 0; b < NB; b++) {
                uint16_t d = janas_fp32_to_fp16(rndf(-0.01f, 0.01f));
                memcpy(w + b * bs, &d, 2);
            }
            for (int v = 0; v < NV; v++) {
                for (int i = 0; i < N; i++)
                    xf[i] = rndf(-1.0f, 1.0f);
                janas_q8k_quantize(xf, x + v * NB, N);
            }
            janas_dequantize(t, w, wf, N);
            double expect = 0.0, mag = 0.0;
            for (int b = 0; b < NB; b++)
                for (int i = 0; i < JANAS_QK; i++) {
                    double e =
                        (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                    expect += e;
                    mag += fabs(e);
                }
            float ref = janas_iq2_dot_ref(t, w, x, NB);
            CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "%s ref %g vs %g",
                  names[k], ref, expect);
            CHECK(janas_dot(t, w, x, NB) == ref, "%s fast vs ref", names[k]);
            for (int nv = 1; nv <= NV; nv++) {
                float out[NV];
                janas_dot_multi(t, w, x, NB, (size_t)nv, NB, out, 1);
                for (int v = 0; v < nv; v++)
                    CHECK(out[v] == janas_iq2_dot_ref(t, w, x + v * NB, NB),
                          "%s multi nv %d v %d", names[k], nv, v);
            }
            janas_dot_rows(t, w, 2 * bs, x, 2, 4, rows);
            for (int r = 0; r < 4; r++)
                CHECK(rows[r] == janas_iq2_dot_ref(t, w + 2 * bs * r, x, 2),
                      "%s rows %d", names[k], r);
        }
    free(w);
    free(x);
    free(xf);
    free(wf);
}

static void test_dot_iq1m(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    struct janas_block_iq1m *w = malloc(NB * sizeof(*w));
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float)), *wf = malloc(N * sizeof(float)),
          *w2 = malloc(N * sizeof(float));
    for (int trial = 0; trial < 100; trial++) {
        unsigned char *p = (unsigned char *)w;
        for (size_t i = 0; i < NB * sizeof(*w); i++)
            p[i] = (unsigned char)rnd();
        for (int b = 0; b < NB; b++) {
            /* a finite d in the scale words' top nibbles */
            uint16_t d = janas_fp32_to_fp16(rndf(-0.01f, 0.01f)), sc[4];
            memcpy(sc, w[b].scales, 8);
            for (int k = 0; k < 4; k++)
                sc[k] =
                    (uint16_t)((sc[k] & 0x0fff) | ((d >> 4 * k & 15) << 12));
            memcpy(w[b].scales, sc, 8);
        }
        for (int v = 0; v < NV; v++) {
            for (int i = 0; i < N; i++)
                xf[i] = rndf(-1.0f, 1.0f);
            janas_q8k_quantize(xf, x + v * NB, N);
        }
        janas_iq1m_dequantize(w, wf, N);
        janas_dequantize(JANAS_IQ1_M, w, w2, N);
        CHECK(memcmp(wf, w2, N * sizeof(float)) == 0,
              "iq1m: janas_dequantize differs from its unpacker");
        double expect = 0.0, mag = 0.0;
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < JANAS_QK; i++) {
                double e = (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                expect += e;
                mag += fabs(e);
            }
        float ref = janas_iq1m_dot_ref(w, x, NB);
        CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "iq1m ref %g vs %g", ref,
              expect);
        CHECK(janas_dot(JANAS_IQ1_M, w, x, NB) == ref, "iq1m fast vs ref");
        for (int nv = 1; nv <= NV; nv++) {
            float out[NV];
            janas_dot_multi(JANAS_IQ1_M, w, x, NB, (size_t)nv, NB, out, 1);
            for (int v = 0; v < nv; v++)
                CHECK(out[v] == janas_iq1m_dot_ref(w, x + v * NB, NB),
                      "iq1m multi nv %d v %d", nv, v);
        }
    }
    free(w);
    free(x);
    free(xf);
    free(wf);
    free(w2);
}

static void test_dot_iq1s(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    struct janas_block_iq1s *w = malloc(NB * sizeof(*w));
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
        janas_iq1s_dequantize(w, wf, N);
        janas_dequantize(JANAS_IQ1_S, w, w2, N);
        CHECK(memcmp(wf, w2, N * sizeof(float)) == 0,
              "iq1s: janas_dequantize differs from its unpacker");
        double expect = 0.0, mag = 0.0;
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < JANAS_QK; i++) {
                double e = (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                expect += e;
                mag += fabs(e);
            }
        float ref = janas_iq1s_dot_ref(w, x, NB);
        CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "iq1s ref %g vs %g", ref,
              expect);
        CHECK(janas_dot(JANAS_IQ1_S, w, x, NB) == ref, "iq1s fast vs ref");
        for (int nv = 1; nv <= NV; nv++) {
            float out[NV];
            janas_dot_multi(JANAS_IQ1_S, w, x, NB, (size_t)nv, NB, out, 1);
            for (int v = 0; v < nv; v++)
                CHECK(out[v] == janas_iq1s_dot_ref(w, x + v * NB, NB),
                      "iq1s multi nv %d v %d", nv, v);
        }
    }
    free(w);
    free(x);
    free(xf);
    free(wf);
    free(w2);
}

static void test_dot_q2k(void)
{
    enum { NB = 8, N = NB * JANAS_QK, NV = 17 };
    struct janas_block_q2k *w = malloc(NB * sizeof(*w));
    struct janas_block_q8k *x = malloc(NV * NB * sizeof(*x));
    float *xf = malloc(N * sizeof(float)), *wf = malloc(N * sizeof(float)),
          *w2 = malloc(N * sizeof(float));
    for (int trial = 0; trial < 100; trial++) {
        unsigned char *p = (unsigned char *)w;
        for (size_t i = 0; i < NB * sizeof(*w); i++)
            p[i] = (unsigned char)rnd();
        for (int b = 0; b < NB; b++) {
            w[b].d = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
            w[b].dmin = janas_fp32_to_fp16(rndf(0.0001f, 0.01f));
        }
        for (int v = 0; v < NV; v++) {
            for (int i = 0; i < N; i++)
                xf[i] = rndf(-1.0f, 1.0f);
            janas_q8k_quantize(xf, x + v * NB, N);
        }
        janas_q2k_dequantize(w, wf, N);
        janas_dequantize(JANAS_Q2_K, w, w2, N);
        CHECK(memcmp(wf, w2, N * sizeof(float)) == 0,
              "q2k: janas_dequantize differs from its unpacker");
        double expect = 0.0, mag = 0.0;
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < JANAS_QK; i++) {
                double e = (double)wf[b * JANAS_QK + i] * x[b].d * x[b].qs[i];
                expect += e;
                mag += fabs(e);
            }
        float ref = janas_q2k_dot_ref(w, x, NB);
        CHECK(fabs(ref - expect) <= 1e-5 * mag + 1e-6, "q2k ref %g vs %g", ref,
              expect);
        CHECK(janas_dot(JANAS_Q2_K, w, x, NB) == ref, "q2k fast vs ref");
        for (int nv = 1; nv <= NV; nv++) {
            float out[NV];
            janas_dot_multi(JANAS_Q2_K, w, x, NB, (size_t)nv, NB, out, 1);
            for (int v = 0; v < nv; v++)
                CHECK(out[v] == janas_q2k_dot_ref(w, x + v * NB, NB),
                      "q2k multi nv %d v %d", nv, v);
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
    test_q6kpn_axpy();
    test_q6kpn_chunk();
    test_matvec();
    test_matvec_group();
    test_matvec_multi();
    test_dot_multi();
    test_det();
    test_dot_q5k_q8_0();
    test_dot_iq4();
    test_dot_mxfp4();
    test_dot_q4_0_q5_1();
    test_dot_q5_0_q2_0();
    test_multi_rows_long();
    test_q6k_tiles();
    test_row_tails();
    test_dot_q3k();
    test_dot_iq3s();
    test_dot_iq3xxs();
    test_dot_iq2();
    test_dot_iq1m();
    test_dot_iq1s();
    test_dot_q2k();
    test_quantize_q4k();
    test_quantize_q6k();
    test_quantize_q8k();
    test_dequantize();
    test_q8w();
    test_bf16_rows();
    printf("test_quant: kernel %s, %s\n", janas_q4k_kernel_name(),
           failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
