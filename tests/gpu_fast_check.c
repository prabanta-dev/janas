/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_fast_check.c - the fast prompt's products of the types in blocks of
 * 32 or 64 (iq4nl_mmq.comp: IQ4_NL, MXFP4, Q5_0, Q2_0) and a mixture's
 * experts in them (b32_mmid.comp: IQ4_NL, Q5_0, Q2_0), rows ending inside
 * a group of 256 among them (qwen4exp's 640), and experts in the 256-block
 * types of iqk_mmid.comp (Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S,
 * IQ1_M, IQ3_S, IQ4_XS), and the Q4_K and Q6_K tiles (q4k_mmq.comp,
 * q6k_mmq.comp; with JANAS_GPU_TILES_EXACT=1 their _exact variants, which
 * must give the CPU's bits), against the CPU. They round
 * group by group, so not bit for bit: each output within 1e-4 of the
 * largest of its product, and the GPU seen at work (outputs that differ in
 * their last bits). Skips (exit 0) when no GPU is usable.
 *
 * Usage: gpu_fast_check
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/aligned.h"
#include "llm/gpu.h"
#include "llm/matvec.h"
#include "llm/quant.h"

static uint64_t rng = 0x2545F4914F6CDD1Dull;

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

/* bytes of a block, and the weights it holds */
static size_t block_bytes(int type, size_t *weights)
{
    /* the 256-block types: their row bytes of 256 */
    *weights = type == JANAS_Q2_0                ? 64
               : janas_qtype_row_bytes(type, 32) ? 32
                                                 : 256;
    return janas_qtype_row_bytes(type, *weights);
}

/* rows x cols weights of a type, random bytes with small scales */
static void fill(uint8_t *w, int type, size_t rows, size_t cols)
{
    size_t per, bs = block_bytes(type, &per);
    size_t n = rows * (cols / per);
    for (size_t i = 0; i < n * bs; i++)
        w[i] = (uint8_t)rnd();
    for (size_t b = 0; b < n; b++) {
        uint8_t *blk = w + b * bs;
        if (type == JANAS_MXFP4) {
            blk[0] = (uint8_t)(118 + rnd() % 8);
            continue;
        }
        uint16_t d = janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-6f);
        if (type == JANAS_Q4_K) { /* d and dmin at 0 and 2 */
            uint16_t dm = janas_fp32_to_fp16(0.0002f + (rnd() % 1000) * 1e-6f);
            memcpy(blk, &d, 2);
            memcpy(blk + 2, &dm, 2);
        } else if (type == JANAS_Q6_K) { /* d at its end */
            memcpy(blk + 208, &d, 2);
        } else if (type == JANAS_Q2_K) { /* d and dmin at 80 and 82 */
            uint16_t dm = janas_fp32_to_fp16(0.0002f + (rnd() % 1000) * 1e-6f);
            memcpy(blk + 80, &d, 2);
            memcpy(blk + 82, &dm, 2);
        } else if (type == JANAS_IQ1_M) { /* d in the scales' top nibbles */
            for (int s = 0; s < 4; s++) {
                uint16_t sc;
                memcpy(&sc, blk + 48 + 2 * s, 2);
                sc = (uint16_t)((sc & 0x0fff) | ((d >> 4 * s & 15) << 12));
                memcpy(blk + 48 + 2 * s, &sc, 2);
            }
        } else {
            memcpy(blk, &d, 2);
        }
    }
}

/* nv activation vectors of cols values, xs blocks apart, zero past cols
   in their last group (as the engine's) */
static void acts(struct janas_block_q8k *x, size_t nv, size_t cols, size_t xs)
{
    size_t full = (cols + JANAS_QK - 1) / JANAS_QK * JANAS_QK;
    float *f = calloc(full, sizeof(float));
    for (size_t v = 0; v < nv; v++) {
        memset(f, 0, full * sizeof(float));
        for (size_t i = 0; i < cols; i++)
            f[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;
        janas_q8k_quantize(f, x + v * xs, (uint32_t)full);
    }
    free(f);
}

/* the outputs of a against those of b: the largest difference over the
   largest magnitude, and how many differ at all */
static double compare(const float *a, const float *b, size_t n, size_t *diff)
{
    double big = 1e-30, worst = 0;
    for (size_t i = 0; i < n; i++)
        if (fabs(a[i]) > big)
            big = fabs(a[i]);
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - b[i]);
        if (d > worst)
            worst = d;
        *diff += memcmp(&a[i], &b[i], 4) != 0;
    }
    return worst / big;
}

static void *page_alloc(size_t bytes)
{
    bytes = (bytes + 4095) / 4096 * 4096;
    void *p = janas_aligned_alloc(4096, bytes);
    if (p)
        memset(p, 0, bytes);
    return p;
}

int main(void)
{
    char err[256];
    struct janas_gpu *g = janas_gpu_create(err, sizeof(err));
    if (!g) {
        printf("gpu_fast_check: no GPU (%s), skipped\n", err);
        return 0;
    }
    if (!janas_gpu_fast_prompt()) {
        printf("gpu_fast_check: the fast prompt is off, skipped\n");
        janas_gpu_destroy(g);
        return 0;
    }
    struct janas_pool *pool = janas_pool_create(4, NULL);
    /* and the 256-block types of iqk_mmq.comp (only the shapes of whole
       groups of 256) */
    static const int types[] = {
        JANAS_Q5_0,    JANAS_Q2_0,   JANAS_IQ4_NL, JANAS_MXFP4,   JANAS_Q2_K,
        JANAS_IQ2_XXS, JANAS_IQ2_XS, JANAS_IQ2_S,  JANAS_IQ3_XXS, JANAS_IQ1_S,
        JANAS_IQ1_M,   JANAS_Q4_K,   JANAS_Q6_K};
    static const char *names[] = {
        "Q5_0",  "Q2_0",    "IQ4_NL", "MXFP4", "Q2_K", "IQ2_XXS", "IQ2_XS",
        "IQ2_S", "IQ3_XXS", "IQ1_S",  "IQ1_M", "Q4_K", "Q6_K"};
    /* rows x cols; the last three end inside a group of 256 */
    static const size_t shapes[][2] = {
        {512, 2560}, {640, 2560}, {2560, 640}, {300, 896}, {130, 128}};
    static const size_t nvs[] = {64, 100, 256};
    enum { MAXW = 32 << 20, MAXX = 1024 * 10, MAXY = 2560 * 1024 };
    uint8_t *w = page_alloc(MAXW);
    struct janas_block_q8k *x = page_alloc(sizeof(*x) * MAXX);
    float *ya = page_alloc(sizeof(float) * MAXY),
          *yb = page_alloc(sizeof(float) * MAXY);
    if (!w || !x || !ya || !yb || janas_gpu_import(g, w, MAXW) != 0 ||
        janas_gpu_import(g, ya, (sizeof(float) * MAXY + 4095) / 4096 * 4096) !=
            0 ||
        janas_gpu_import(g, yb, (sizeof(float) * MAXY + 4095) / 4096 * 4096) !=
            0) {
        printf("gpu_fast_check: memory or import failed\n");
        return 1;
    }
    int bad = 0, runs = 0;
    const char *ex = getenv("JANAS_GPU_TILES_EXACT");
    int exact = ex && atoi(ex) > 0;
    double worst_all = 0;
    for (size_t ti = 0; ti < sizeof(types) / sizeof(*types); ti++)
        for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++)
            for (size_t k = 0; k < sizeof(nvs) / sizeof(*nvs); k++) {
                int type = types[ti];
                size_t rows = shapes[s][0], cols = shapes[s][1], nv = nvs[k];
                size_t xs = (cols + JANAS_QK - 1) / JANAS_QK;
                if (janas_qtype_row_bytes(type, cols) == 0 ||
                    janas_qtype_row_bytes(type, cols) * rows > MAXW ||
                    nv * xs > MAXX || rows * nv > MAXY)
                    continue;
                fill(w, type, rows, cols);
                acts(x, nv, cols, xs);
                struct janas_matvec_task t = {.type = type,
                                              .w = w,
                                              .x = x,
                                              .rows = rows,
                                              .cols = cols,
                                              .n_vec = nv};
                t.y = ya;
                janas_matvec_q4k_group(pool, &t, 1);
                t.y = yb;
                janas_gpu_matvec_group(g, pool, &t, 1);
                size_t diff = 0;
                double e = compare(ya, yb, rows * nv, &diff);
                runs++;
                if (e > worst_all)
                    worst_all = e;
                /* JANAS_GPU_TILES_EXACT=1: Q4_K's and Q6_K's tiles the
                   CPU's bits (their _exact shaders) */
                if (exact && (type == JANAS_Q4_K || type == JANAS_Q6_K)) {
                    if (diff != 0) {
                        printf("%s %zux%zu nv %zu: exact, yet %zu of %zu "
                               "outputs differ (error %.2e)\n",
                               names[ti], rows, cols, nv, diff, rows * nv, e);
                        bad++;
                    }
                } else if (e > 1e-4 || diff == 0) {
                    printf("%s %zux%zu nv %zu: error %.2e, %zu of %zu "
                           "outputs differ%s\n",
                           names[ti], rows, cols, nv, e, diff, rows * nv,
                           diff == 0 ? " (the GPU did not compute them?)" : "");
                    bad++;
                }
            }
    /* a mixture's experts: 24 of one shape, 1 to 40 tokens each (more
       than 128 in all, as experts_id asks), each its own activations; the
       GPU takes those with the most tokens, the CPU the others */
    static const int etypes[] = {JANAS_Q2_0, JANAS_Q5_0, JANAS_IQ4_NL};
    static const char *enames[] = {"Q2_0", "Q5_0", "IQ4_NL"};
    static const size_t eshapes[][2] = {{640, 2560}, {2560, 640}};
    enum { NE = 24 };
    for (size_t ti = 0; ti < sizeof(etypes) / sizeof(*etypes); ti++)
        for (size_t s = 0; s < 2; s++) {
            int type = etypes[ti];
            size_t rows = eshapes[s][0], cols = eshapes[s][1];
            size_t xs = (cols + JANAS_QK - 1) / JANAS_QK;
            size_t rb = janas_qtype_row_bytes(type, cols);
            size_t per = rows * rb; /* an expert's matrix */
            if (per * NE > MAXW)
                continue;
            struct janas_matvec_task t[NE];
            size_t at = 0, tokens = 0;
            for (int e = 0; e < NE; e++) {
                size_t nv = 1 + rnd() % 40;
                if (at + nv > MAXX / xs || (at + nv) * rows > MAXY)
                    nv = 1;
                fill(w + e * per, type, rows, cols);
                t[e] = (struct janas_matvec_task){.type = type,
                                                  .w = w + e * per,
                                                  .x = x + at * xs,
                                                  .rows = rows,
                                                  .cols = cols,
                                                  .n_vec = nv,
                                                  .x_stride = xs,
                                                  .y_stride = rows};
                at += nv;
            }
            tokens = at;
            acts(x, tokens, cols, xs);
            for (int e = 0, o = 0; e < NE; o += (int)t[e].n_vec, e++)
                t[e].y = ya + (size_t)o * rows;
            janas_matvec_q4k_group(pool, t, NE);
            for (int e = 0, o = 0; e < NE; o += (int)t[e].n_vec, e++)
                t[e].y = yb + (size_t)o * rows;
            janas_gpu_matvec_group(g, pool, t, NE);
            size_t diff = 0;
            double err_e = compare(ya, yb, tokens * rows, &diff);
            runs++;
            if (err_e > worst_all)
                worst_all = err_e;
            if (err_e > 1e-4 || diff == 0) {
                printf("experts %s %zux%zu, %zu tokens: error %.2e, %zu of "
                       "%zu outputs differ%s\n",
                       enames[ti], rows, cols, tokens, err_e, diff,
                       tokens * rows,
                       diff == 0 ? " (the GPU did not compute them?)" : "");
                bad++;
            }
        }
    /* the same for the 256-block types of iqk_mmid.comp */
    static const int ktypes[] = {JANAS_Q2_K,  JANAS_IQ2_XXS, JANAS_IQ2_XS,
                                 JANAS_IQ2_S, JANAS_IQ3_XXS, JANAS_IQ1_S,
                                 JANAS_IQ1_M, JANAS_IQ3_S,   JANAS_IQ4_XS};
    static const char *knames[] = {"Q2_K",  "IQ2_XXS", "IQ2_XS",
                                   "IQ2_S", "IQ3_XXS", "IQ1_S",
                                   "IQ1_M", "IQ3_S",   "IQ4_XS"};
    static const size_t kshapes[][2] = {{512, 2048}, {2048, 512}};
    for (size_t ti = 0; ti < sizeof(ktypes) / sizeof(*ktypes); ti++)
        for (size_t s = 0; s < 2; s++) {
            int type = ktypes[ti];
            size_t rows = kshapes[s][0], cols = kshapes[s][1];
            size_t xs = cols / JANAS_QK;
            size_t rb = janas_qtype_row_bytes(type, cols), per = rows * rb;
            if (per * NE > MAXW)
                continue;
            struct janas_matvec_task t[NE];
            size_t at = 0;
            for (int e = 0; e < NE; e++) {
                size_t nv = 1 + rnd() % 40;
                if (at + nv > MAXX / xs || (at + nv) * rows > MAXY)
                    nv = 1;
                fill(w + e * per, type, rows, cols);
                t[e] = (struct janas_matvec_task){.type = type,
                                                  .w = w + e * per,
                                                  .x = x + at * xs,
                                                  .rows = rows,
                                                  .cols = cols,
                                                  .n_vec = nv,
                                                  .x_stride = xs,
                                                  .y_stride = rows};
                at += nv;
            }
            acts(x, at, cols, xs);
            for (int e = 0, o = 0; e < NE; o += (int)t[e].n_vec, e++)
                t[e].y = ya + (size_t)o * rows;
            janas_matvec_q4k_group(pool, t, NE);
            for (int e = 0, o = 0; e < NE; o += (int)t[e].n_vec, e++)
                t[e].y = yb + (size_t)o * rows;
            janas_gpu_matvec_group(g, pool, t, NE);
            size_t diff = 0;
            double err_e = compare(ya, yb, at * rows, &diff);
            runs++;
            if (err_e > worst_all)
                worst_all = err_e;
            if (err_e > 1e-4 || diff == 0) {
                printf("experts %s %zux%zu, %zu tokens: error %.2e, %zu of "
                       "%zu outputs differ%s\n",
                       knames[ti], rows, cols, at, err_e, diff, at * rows,
                       diff == 0 ? " (the GPU did not compute them?)" : "");
                bad++;
            }
        }
    printf("gpu_fast_check (%s): %d products, largest error %.2e, %s\n",
           janas_gpu_name(g), runs, worst_all, bad ? "FAILED" : "ok");
    janas_gpu_destroy(g);
    janas_pool_destroy(pool);
    janas_aligned_free(w);
    janas_aligned_free(x);
    janas_aligned_free(ya);
    janas_aligned_free(yb);
    return bad ? 1 : 0;
}
