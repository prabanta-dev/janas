/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_check.c - products split between the GPU and the CPU must equal the
 * CPU's bit for bit: random Q4_K matrices in shared memory, several shapes,
 * vector counts and strides. Skips (exit 0) when no GPU is usable.
 *
 * Usage: gpu_check
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llm/gpu.h"
#include "llm/matvec.h"
#include "llm/quant.h"

static uint64_t rng = 88172645463325252ull;

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

int main(void)
{
    /* GPU_CHECK_BIG_GB=n: n GiB of touched memory, like the engine's
       expert cache */
    const char *bg = getenv("GPU_CHECK_BIG_GB");
    if (bg) {
        size_t big = (size_t)atoi(bg) << 30;
        uint8_t *b = aligned_alloc(4096, big);
        if (b)
            memset(b, 1, big);
    }
    char err[256];
    struct janas_gpu *g = janas_gpu_create(err, sizeof(err));
    if (!g) {
        printf("gpu_check: no GPU (%s), skipped\n", err);
        return 0;
    }
    enum { MAXR = 4096, MAXC = 4096, MAXV = 17 };
    size_t wbytes = (size_t)MAXR * (MAXC / JANAS_QK) * 144;
    wbytes = (wbytes + 4095) / 4096 * 4096;
    uint8_t *w = aligned_alloc(4096, wbytes);
    for (size_t i = 0; i < wbytes; i++)
        w[i] = (uint8_t)rnd();
    struct janas_block_q4k *wb = (struct janas_block_q4k *)w;
    for (size_t b = 0; b < wbytes / 144; b++) {
        wb[b].d = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
        wb[b].dmin = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
    }
    /* Q6_K weights too, with small scales */
    size_t w6bytes = (size_t)MAXR * (MAXC / JANAS_QK) * 210 + 4096;
    w6bytes = (w6bytes + 4095) / 4096 * 4096;
    uint8_t *w6 = aligned_alloc(4096, w6bytes);
    for (size_t i = 0; i < w6bytes; i++)
        w6[i] = (uint8_t)rnd();
    struct janas_block_q6k *w6b = (struct janas_block_q6k *)w6;
    for (size_t b = 0; b < (w6bytes - 4096) / 210; b++)
        w6b[b].d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
    if (janas_gpu_import(g, w, wbytes) != 0 ||
        janas_gpu_import(g, w6, w6bytes) != 0) {
        printf("gpu_check: import failed\n");
        return 1;
    }
    size_t xblocks = (size_t)MAXV * 2 * (MAXC / JANAS_QK);
    struct janas_block_q8k *x = malloc(xblocks * sizeof(*x));
    float *xf = malloc(MAXC * sizeof(float));
    for (size_t v = 0; v < (size_t)MAXV * 2; v++) {
        for (int i = 0; i < MAXC; i++)
            xf[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;
        janas_q8k_quantize(xf, x + v * (MAXC / JANAS_QK), MAXC);
    }
    /* GPU_CHECK_ENGINE=1: the engine's pool (12 threads on CPUs 0-11,
       the caller pinned to CPU 0), else 4 unpinned threads */
    int cpus[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int engine = getenv("GPU_CHECK_ENGINE") != NULL;
    struct janas_pool *pool =
        engine ? janas_pool_create(12, cpus) : janas_pool_create(4, NULL);
    if (engine && getenv("GPU_CHECK_PIN"))
        janas_pin_current_thread(0);
    size_t ysz = (size_t)MAXR * MAXV * 2;
    float *ya = malloc(ysz * sizeof(float)), *yb = malloc(ysz * sizeof(float));
    const size_t shapes[][2] = {{4096, 2048}, {1000, 256},  {128, 4096},
                                {2048, 512},  {4096, 4096}, {777, 1024},
                                {1500, 768}}; /* odd blocks per row */
    const size_t nvs[] = {2, 3, 5, 8, 9, 16, 17};
    int bad = 0, runs = 0;
    for (int q6 = 0; q6 < 2; q6++)
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
            for (size_t k = 0; k < sizeof(nvs) / sizeof(nvs[0]); k++)
                for (int strided = 0; strided < 2; strided++) {
                    size_t rows = shapes[s][0], cols = shapes[s][1];
                    size_t nb = cols / JANAS_QK, nv = nvs[k];
                    struct janas_matvec_task t = {
                        .type = q6 ? JANAS_Q6_K : JANAS_Q4_K,
                        /* a row further in, every other shape */
                        .w = q6 ? w6 + (s % 2) * 210 * nb
                                : w + (s % 2) * 144 * nb,
                        .x = x,
                        .rows = rows,
                        .cols = cols,
                        .n_vec = nv,
                        .x_stride = strided ? 2 * nb : 0,
                        .y_stride = strided ? rows + 5 : 0};
                    size_t ys = strided ? rows + 5 : rows;
                    memset(ya, 0, ysz * sizeof(float));
                    memset(yb, 0, ysz * sizeof(float));
                    t.y = ya;
                    janas_matvec_q4k_group(pool, &t, 1);
                    t.y = yb;
                    if (getenv("GPU_ONLY")) {
                        /* every row on the GPU, nothing on the CPU */
                        int waited;
                        if (janas_gpu_begin(g, &t, &rows, 1) != 0 ||
                            janas_gpu_finish(g, &waited) != 0)
                            printf("gpu call failed\n");
                    } else {
                        /* several calls: the share adapts, results must not */
                        for (int c = 0; c < 4; c++) {
                            janas_gpu_matvec_group(g, pool, &t, 1);
                            /* GPU_CHECK_GAP_US: CPU-only work between calls,
                               as in the engine */
                            const char *gap = getenv("GPU_CHECK_GAP_US");
                            if (gap) {
                                struct timespec a, b;
                                clock_gettime(CLOCK_MONOTONIC, &a);
                                do
                                    clock_gettime(CLOCK_MONOTONIC, &b);
                                while ((b.tv_sec - a.tv_sec) * 1000000 +
                                           (b.tv_nsec - a.tv_nsec) / 1000 <
                                       atoi(gap));
                            }
                        }
                    }
                    runs++;
                    for (size_t v = 0; v < nv && bad < 10; v++)
                        for (size_t r = 0; r < rows; r++)
                            if (memcmp(&ya[v * ys + r], &yb[v * ys + r], 4)) {
                                size_t nbad = 0;
                                for (size_t q = 0; q < rows; q++)
                                    nbad += memcmp(&ya[v * ys + q],
                                                   &yb[v * ys + q], 4) != 0;
                                printf("[%zu wrong rows] ", nbad);
                                printf(
                                    "%s %zux%zu nv %zu strided %d: v %zu row %zu: "
                                    "%g vs %g\n",
                                    q6 ? "Q6_K" : "Q4_K", rows, cols, nv,
                                    strided, v, r, ya[v * ys + r],
                                    yb[v * ys + r]);
                                for (size_t u = 0;
                                     getenv("GPU_CHECK_DEBUG") && u < 2 * MAXV;
                                     u++)
                                    if (janas_q4k_dot(
                                            (const struct janas_block_q4k *)
                                                    t.w +
                                                r * nb,
                                            x + u * nb, nb) == yb[v * ys + r])
                                        printf("  = row %zu times vector %zu\n",
                                               r, u);
                                bad++;
                                break;
                            }
                }
    printf("gpu_check (%s, subgroups of %u): %d shapes, %s\n",
           janas_gpu_name(g), janas_gpu_subgroup(g), runs,
           bad ? "FAILED" : "ok");
    janas_gpu_destroy(g);
    janas_pool_destroy(pool);
    free(w);
    free(x);
    free(xf);
    free(ya);
    free(yb);
    return bad ? 1 : 0;
}
