/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_check.c - products split between the GPU and the CPU must equal the
 * CPU's bit for bit: random Q4_K, Q5_K, Q6_K, Q6_K_P (bit planes, at its
 * three levels), Q8_0 and the blocks of 32 (Q4_0, IQ4_NL, Q4_1, Q5_1)
 * matrices in shared memory, and the GPU's aligned copy of Q6_K (on the GPU
 * alone, against the CPU's Q6_K), several shapes, vector
 * counts and strides. Skips (exit 0) when no GPU is usable.
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
    /* the same Q6_K in the GPU's aligned copy (gpu.h JANAS_Q6KA_BLOCK):
       ql, qh, scales, d at byte 208 */
    size_t nb6 = (w6bytes - 4096) / 210;
    size_t wabytes = (nb6 * JANAS_Q6KA_BLOCK + 4095) / 4096 * 4096;
    uint8_t *wa = aligned_alloc(4096, wabytes);
    memset(wa, 0, wabytes);
    for (size_t b = 0; b < nb6; b++) {
        uint8_t *d = wa + b * JANAS_Q6KA_BLOCK;
        memcpy(d, w6b[b].ql, sizeof(w6b[b].ql));
        memcpy(d + 128, w6b[b].qh, sizeof(w6b[b].qh));
        memcpy(d + 192, w6b[b].scales, sizeof(w6b[b].scales));
        memcpy(d + 208, &w6b[b].d, sizeof(w6b[b].d));
    }
    /* Q6_K_P: the high planes (82-byte blocks), then the two bare planes
       (64 bytes a block), row after row as in an expert slot */
    size_t pbase = (size_t)MAXR * (MAXC / JANAS_QK) * 82 + 4096;
    pbase = (pbase + 4095) / 4096 * 4096;
    size_t pplane = (size_t)MAXR * (MAXC / JANAS_QK) * JANAS_Q6KP_PLANE;
    size_t wpbytes = pbase + 2 * pplane;
    uint8_t *wp = aligned_alloc(4096, wpbytes);
    for (size_t i = 0; i < wpbytes; i++)
        wp[i] = (uint8_t)rnd();
    struct janas_block_q6kp *wpb = (struct janas_block_q6kp *)wp;
    for (size_t b = 0; b < (pbase - 4096) / 82; b++)
        wpb[b].d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
    const uint8_t *pl1 = wp + pbase, *pl0 = wp + pbase + pplane;
    /* Q5_K */
    size_t w5bytes = (size_t)MAXR * (MAXC / JANAS_QK) * 176;
    w5bytes = (w5bytes + 4095) / 4096 * 4096;
    uint8_t *w5 = aligned_alloc(4096, w5bytes);
    for (size_t i = 0; i < w5bytes; i++)
        w5[i] = (uint8_t)rnd();
    struct janas_block_q5k *w5b = (struct janas_block_q5k *)w5;
    for (size_t b = 0; b < w5bytes / 176; b++) {
        w5b[b].d = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
        w5b[b].dmin = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
    }
    /* Q8_0: blocks of 32, eight to 256 weights */
    size_t w8bytes = (size_t)MAXR * (MAXC / JANAS_QK) * 272;
    w8bytes = (w8bytes + 4095) / 4096 * 4096;
    uint8_t *w8 = aligned_alloc(4096, w8bytes);
    for (size_t i = 0; i < w8bytes; i++)
        w8[i] = (uint8_t)rnd();
    struct janas_block_q8_0 *w8b = (struct janas_block_q8_0 *)w8;
    for (size_t b = 0; b < w8bytes / 34; b++)
        w8b[b].d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
    /* blocks of 32 with four or five bits: Q4_0 and IQ4_NL (18 bytes: d,
       16 of nibbles), Q4_1 (20: d, m, ...), Q5_1 (24: d, m, qh, ...);
       random bytes with d (and m) set to small scales */
    static const int t32[4] = {JANAS_Q4_0, JANAS_IQ4_NL, JANAS_Q4_1,
                               JANAS_Q5_1};
    /* Q3_K (d at byte 108) and IQ3_S (d at 0): 110-byte blocks */
    uint8_t *w3[2];
    size_t w3bytes = (size_t)MAXR * (MAXC / JANAS_QK) * 110 + 4096;
    w3bytes = (w3bytes + 4095) / 4096 * 4096;
    for (int k = 0; k < 2; k++) {
        w3[k] = aligned_alloc(4096, w3bytes);
        for (size_t i = 0; i < w3bytes; i++)
            w3[k][i] = (uint8_t)rnd();
        for (size_t b = 0; b + 110 <= w3bytes; b += 110) {
            uint16_t d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
            memcpy(w3[k] + b + (k == 0 ? 108 : 0), &d, 2);
        }
    }
    /* IQ4_XS: 136-byte blocks, d, scales_h, scales_l, nibbles */
    size_t wxbytes = (size_t)MAXR * (MAXC / JANAS_QK) * 136;
    wxbytes = (wxbytes + 4095) / 4096 * 4096;
    uint8_t *wx = aligned_alloc(4096, wxbytes);
    for (size_t i = 0; i < wxbytes; i++)
        wx[i] = (uint8_t)rnd();
    for (size_t b = 0; b + 136 <= wxbytes; b += 136) {
        uint16_t d = janas_fp32_to_fp16(0.00005f + (rnd() % 1000) * 1e-7f);
        memcpy(wx + b, &d, 2);
    }
    static const size_t s32[4] = {18, 18, 20, 24};
    uint8_t *w32[4];
    size_t w32bytes[4];
    for (int k = 0; k < 4; k++) {
        w32bytes[k] = (size_t)MAXR * (MAXC / JANAS_QK) * 8 * s32[k];
        w32bytes[k] = (w32bytes[k] + 4095) / 4096 * 4096;
        w32[k] = aligned_alloc(4096, w32bytes[k]);
        for (size_t i = 0; i < w32bytes[k]; i++)
            w32[k][i] = (uint8_t)rnd();
        for (size_t b = 0; b + s32[k] <= w32bytes[k]; b += s32[k]) {
            uint16_t d = janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-6f);
            memcpy(w32[k] + b, &d, 2);
            if (s32[k] > 18) {
                uint16_t m =
                    janas_fp32_to_fp16(-0.01f + (rnd() % 1000) * 2e-5f);
                memcpy(w32[k] + b + 2, &m, 2);
            }
        }
    }
    if (janas_gpu_import(g, w, wbytes) != 0 ||
        janas_gpu_import(g, w6, w6bytes) != 0 ||
        janas_gpu_import(g, wa, wabytes) != 0 ||
        janas_gpu_import(g, wp, wpbytes) != 0 ||
        janas_gpu_import(g, w5, w5bytes) != 0 ||
        janas_gpu_import(g, w8, w8bytes) != 0 ||
        janas_gpu_import(g, w32[0], w32bytes[0]) != 0 ||
        janas_gpu_import(g, w32[1], w32bytes[1]) != 0 ||
        janas_gpu_import(g, w32[2], w32bytes[2]) != 0 ||
        janas_gpu_import(g, w32[3], w32bytes[3]) != 0 ||
        janas_gpu_import(g, wx, wxbytes) != 0 ||
        janas_gpu_import(g, w3[0], w3bytes) != 0 ||
        janas_gpu_import(g, w3[1], w3bytes) != 0) {
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
    const size_t nvs[] = {1, 2, 3, 5, 8, 9, 16, 17};
    int bad = 0, runs = 0;
    /* 0 Q4_K, 1 Q6_K, 2-4 Q6_K_P with three, two and one planes, 5 Q5_K,
       6 Q8_0, 7-10 Q4_0, IQ4_NL, Q4_1, Q5_1, 11 IQ4_XS,
       12 Q3_K, 13 IQ3_S, 14 the aligned Q6_K */
    static const char *names[] = {"Q4_K",     "Q6_K",  "Q6_K_P/3", "Q6_K_P/2",
                                  "Q6_K_P/1", "Q5_K",  "Q8_0",     "Q4_0",
                                  "IQ4_NL",   "Q4_1",  "Q5_1",     "IQ4_XS",
                                  "Q3_K",     "IQ3_S", "Q6_K_A"};
    for (int q6 = 0; q6 < 15; q6++)
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
            for (size_t k = 0; k < sizeof(nvs) / sizeof(nvs[0]); k++)
                for (int strided = 0; strided < 2; strided++) {
                    size_t rows = shapes[s][0], cols = shapes[s][1];
                    size_t nb = cols / JANAS_QK, nv = nvs[k];
                    size_t skip = s % 2; /* a row further in, every other
                                            shape */
                    struct janas_matvec_task t = {
                        .type = q6 == 14   ? JANAS_Q6_K
                                : q6 == 13 ? JANAS_IQ3_S
                                : q6 == 12 ? JANAS_Q3_K
                                : q6 == 11 ? JANAS_IQ4_XS
                                : q6 >= 7  ? t32[q6 - 7]
                                : q6 == 6  ? JANAS_Q8_0
                                : q6 == 5  ? JANAS_Q5_K
                                : q6 >= 2  ? JANAS_Q6_K_P
                                : q6 == 1  ? JANAS_Q6_K
                                           : JANAS_Q4_K,
                        .w = q6 == 14   ? w6 + skip * 210 * nb
                             : q6 >= 12 ? w3[q6 - 12] + skip * 110 * nb
                             : q6 == 11 ? wx + skip * 136 * nb
                             : q6 >= 7
                                 ? w32[q6 - 7] + skip * 8 * s32[q6 - 7] * nb
                             : q6 == 6 ? w8 + skip * 272 * nb
                             : q6 == 5 ? w5 + skip * 176 * nb
                             : q6 >= 2 ? wp + skip * 82 * nb
                             : q6 == 1 ? w6 + skip * 210 * nb
                                       : w + skip * 144 * nb,
                        .plane = {q6 == 2 || q6 == 3
                                      ? pl1 + skip * JANAS_Q6KP_PLANE * nb
                                      : NULL,
                                  q6 == 2 ? pl0 + skip * JANAS_Q6KP_PLANE * nb
                                          : NULL},
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
                    if (q6 == 14) {
                        /* the CPU has no aligned copy: all on the GPU */
                        t.type = JANAS_Q6_K_A;
                        t.w = wa + skip * JANAS_Q6KA_BLOCK * nb;
                    }
                    if (getenv("GPU_ONLY") || q6 == 14) {
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
                                    names[q6], rows, cols, nv, strided, v, r,
                                    ya[v * ys + r], yb[v * ys + r]);
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
    free(w6);
    free(wa);
    free(wp);
    free(w5);
    free(w8);
    for (int k = 0; k < 4; k++)
        free(w32[k]);
    free(wx);
    free(w3[0]);
    free(w3[1]);
    free(x);
    free(xf);
    free(ya);
    free(yb);
    return bad ? 1 : 0;
}
