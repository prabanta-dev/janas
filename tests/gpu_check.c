/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_check.c - products on the GPU must equal the CPU's bit for bit, and
 * where they do not, the report says which shader is at fault: random Q4_K,
 * Q5_K, Q6_K, Q6_K_P (bit planes, at its three levels), Q8_0 and the blocks
 * of 32 or 64 (Q4_0, IQ4_NL, Q4_1, Q5_1, MXFP4, Q5_0, Q2_0) matrices, the
 * GPU's aligned copy of Q6_K (against the CPU's Q6_K), Q2_K, IQ2_XXS,
 * IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S and IQ1_M (Qwen3.8-Flash-Next's other
 * files, unsloth's 1- and 2-bit ones), several shapes, vector counts and
 * strides; for the blocks of 32 or 64 and Q8_0 also rows that end inside a
 * group of 256 (qwen4exp's 640, its shared experts' down matrices Q8_0).
 * Then bf16 weights on float vectors (janas_gpu_bf16_mv) against the CPU's
 * eight lanes (bf16.h), with x and y where the GPU shares them and through
 * its stages.
 *
 * Every product runs twice: all its rows on the GPU, with the name of the
 * shader that computed them, and shared with the CPU as the engine does
 * (the GPU's share adapting over four calls). Nothing stops at the first
 * error: every wrong product is counted under its shader, with the worst
 * difference in units in the last place (a few: the driver rounds a float
 * operation differently, fused or reordered; many: wrong values). Then the
 * tool isolates the culprits by itself: each wrong product again with the
 * other shaders the engine could take for it, and with every other
 * subgroup size the device allows. The report at the end is meant to be
 * pasted whole into an issue. Skips (exit 0) when no GPU is usable.
 *
 * Usage: gpu_check
 * GPU_CHECK_TYPES=Q6_K,Q8_0 checks those types only; GPU_CHECK_ENGINE=1
 * the engine's pool (12 threads on CPUs 0-11, GPU_CHECK_PIN=1 the caller
 * on CPU 0); GPU_CHECK_BIG_GB=n touches n GiB first, like the engine's
 * expert cache; GPU_CHECK_GAP_US=n CPU-only work between the shared calls.
 * JANAS_GPU_COPY=1 makes an integrated GPU work as a card with memory of
 * its own.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/aligned.h"
#include "common/sysinfo.h"
#include "llm/bf16.h"
#include "llm/gpu.h"
#include "llm/matvec.h"
#include "llm/quant.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "dev"
#endif

static uint64_t rng = 88172645463325252ull;

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

enum { MAXR = 4096, MAXC = 4096, MAXV = 64 };

/* the weights, made once and imported into every GPU tried */
static uint8_t *w, *w6, *wa, *wp, *w5, *w8, *w32[7], *wq[7], *wx, *w3[2];
static size_t wbytes, w6bytes, wabytes, wpbytes, w5bytes, w8bytes, w32bytes[7],
    wqbytes[7], wxbytes, w3bytes;
static const uint8_t *pl1, *pl0;
static struct janas_block_q8k *x;
static float *ya, *yb;
static size_t ysz;
static struct janas_pool *pool;

/* blocks of 32 with four or five bits: Q4_0 and IQ4_NL (18 bytes: d, 16 of
   nibbles), Q4_1 (20: d, m, ...), Q5_1 (24: d, m, qh, ...), MXFP4 (17: the
   E8M0 byte, 16 of nibbles), Q5_0 (22: d, qh, ...), and Q2_0's blocks of 64
   (18: d, two bits a weight; 9 bytes for 32 below) */
static const int t32[7] = {JANAS_Q4_0,  JANAS_IQ4_NL, JANAS_Q4_1, JANAS_Q5_1,
                           JANAS_MXFP4, JANAS_Q5_0,   JANAS_Q2_0};
static const size_t s32[7] = {18, 18, 20, 24, 17, 22, 9};
/* the 256-block types of iq2, iq3xxs, iq1 and q2k_matvec.comp */
static const int tq[7] = {JANAS_Q2_K,  JANAS_IQ2_XXS, JANAS_IQ2_XS,
                          JANAS_IQ2_S, JANAS_IQ3_XXS, JANAS_IQ1_S,
                          JANAS_IQ1_M};

/* the last three: rows ending inside a group of 256, for the blocks of 32
   or 64 and Q8_0 only (640: qwen4exp's experts' down matrices) */
static const size_t shapes[][2] = {{4096, 2048}, {1000, 256},  {128, 4096},
                                   {2048, 512},  {4096, 4096}, {777, 1024},
                                   {1500, 768}, /* odd blocks per row */
                                   {2560, 640},  {300, 896},   {77, 128}};
static const size_t nvs[] = {1, 2, 3, 5, 8, 9, 16, 17, 33, 64};
enum {
    N_SHAPES = sizeof(shapes) / sizeof(shapes[0]),
    N_NVS = sizeof(nvs) / sizeof(nvs[0]),
    N_KINDS = 25
};
/* 0 Q4_K, 1 Q6_K, 2-4 Q6_K_P with three, two and one planes, 5 Q5_K, 6
   Q8_0, 7-10 Q4_0, IQ4_NL, Q4_1, Q5_1, 11 IQ4_XS, 12 Q3_K, 13 IQ3_S, 14 the
   aligned Q6_K, 15 MXFP4, 16 Q5_0, 17 Q2_0, 18-24 Q2_K, IQ2_XXS, IQ2_XS,
   IQ2_S, IQ3_XXS, IQ1_S, IQ1_M */
static const char *names[N_KINDS] = {
    "Q4_K",   "Q6_K",   "Q6_K_P/3", "Q6_K_P/2", "Q6_K_P/1",
    "Q5_K",   "Q8_0",   "Q4_0",     "IQ4_NL",   "Q4_1",
    "Q5_1",   "IQ4_XS", "Q3_K",     "IQ3_S",    "Q6_K_A",
    "MXFP4",  "Q5_0",   "Q2_0",     "Q2_K",     "IQ2_XXS",
    "IQ2_XS", "IQ2_S",  "IQ3_XXS",  "IQ1_S",    "IQ1_M"};

static void fill(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)rnd();
}

static size_t page(size_t n)
{
    return (n + 4095) / 4096 * 4096;
}

static void make_weights(void)
{
    wbytes = page((size_t)MAXR * (MAXC / JANAS_QK) * 144);
    w = janas_aligned_alloc(4096, wbytes);
    fill(w, wbytes);
    struct janas_block_q4k *wb = (struct janas_block_q4k *)w;
    for (size_t b = 0; b < wbytes / 144; b++) {
        wb[b].d = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
        wb[b].dmin = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
    }
    /* Q6_K weights too, with small scales */
    w6bytes = page((size_t)MAXR * (MAXC / JANAS_QK) * 210 + 4096);
    w6 = janas_aligned_alloc(4096, w6bytes);
    fill(w6, w6bytes);
    struct janas_block_q6k *w6b = (struct janas_block_q6k *)w6;
    for (size_t b = 0; b < (w6bytes - 4096) / 210; b++)
        w6b[b].d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
    /* the same Q6_K in the GPU's aligned copy (gpu.h JANAS_Q6KA_BLOCK):
       ql, qh, scales, d at byte 208 */
    size_t nb6 = (w6bytes - 4096) / 210;
    wabytes = page(nb6 * JANAS_Q6KA_BLOCK);
    wa = janas_aligned_alloc(4096, wabytes);
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
    size_t pbase = page((size_t)MAXR * (MAXC / JANAS_QK) * 82 + 4096);
    size_t pplane = (size_t)MAXR * (MAXC / JANAS_QK) * JANAS_Q6KP_PLANE;
    wpbytes = pbase + 2 * pplane;
    wp = janas_aligned_alloc(4096, wpbytes);
    fill(wp, wpbytes);
    struct janas_block_q6kp *wpb = (struct janas_block_q6kp *)wp;
    for (size_t b = 0; b < (pbase - 4096) / 82; b++)
        wpb[b].d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
    pl1 = wp + pbase;
    pl0 = wp + pbase + pplane;
    /* Q5_K */
    w5bytes = page((size_t)MAXR * (MAXC / JANAS_QK) * 176);
    w5 = janas_aligned_alloc(4096, w5bytes);
    fill(w5, w5bytes);
    struct janas_block_q5k *w5b = (struct janas_block_q5k *)w5;
    for (size_t b = 0; b < w5bytes / 176; b++) {
        w5b[b].d = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
        w5b[b].dmin = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
    }
    /* Q8_0: blocks of 32, eight to 256 weights */
    w8bytes = page((size_t)MAXR * (MAXC / JANAS_QK) * 272);
    w8 = janas_aligned_alloc(4096, w8bytes);
    fill(w8, w8bytes);
    struct janas_block_q8_0 *w8b = (struct janas_block_q8_0 *)w8;
    for (size_t b = 0; b < w8bytes / 34; b++)
        w8b[b].d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
    /* Q3_K (d at byte 108) and IQ3_S (d at 0): 110-byte blocks */
    w3bytes = page((size_t)MAXR * (MAXC / JANAS_QK) * 110 + 4096);
    for (int k = 0; k < 2; k++) {
        w3[k] = janas_aligned_alloc(4096, w3bytes);
        fill(w3[k], w3bytes);
        for (size_t b = 0; b + 110 <= w3bytes; b += 110) {
            uint16_t d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
            memcpy(w3[k] + b + (k == 0 ? 108 : 0), &d, 2);
        }
    }
    /* IQ4_XS: 136-byte blocks, d, scales_h, scales_l, nibbles */
    wxbytes = page((size_t)MAXR * (MAXC / JANAS_QK) * 136);
    wx = janas_aligned_alloc(4096, wxbytes);
    fill(wx, wxbytes);
    for (size_t b = 0; b + 136 <= wxbytes; b += 136) {
        uint16_t d = janas_fp32_to_fp16(0.00005f + (rnd() % 1000) * 1e-7f);
        memcpy(wx + b, &d, 2);
    }
    /* the blocks of 32 or 64: random bytes with d (and m) set to small
       scales, MXFP4's byte to powers of two from 2^-10 to 2^-3 (a subnormal
       one, under 2^-125, a GPU may flush) */
    for (int k = 0; k < 7; k++) {
        w32bytes[k] = page((size_t)MAXR * (MAXC / JANAS_QK) * 8 * s32[k]);
        w32[k] = janas_aligned_alloc(4096, w32bytes[k]);
        fill(w32[k], w32bytes[k]);
        size_t step = t32[k] == JANAS_Q2_0 ? 18 : s32[k];
        for (size_t b = 0; b + step <= w32bytes[k]; b += step) {
            if (t32[k] == JANAS_MXFP4) {
                w32[k][b] = (uint8_t)(118 + rnd() % 8);
                continue;
            }
            uint16_t d = janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-6f);
            memcpy(w32[k] + b, &d, 2);
            if (t32[k] == JANAS_Q4_1 || t32[k] == JANAS_Q5_1) {
                uint16_t m =
                    janas_fp32_to_fp16(-0.01f + (rnd() % 1000) * 2e-5f);
                memcpy(w32[k] + b + 2, &m, 2);
            }
        }
    }
    /* the 256-block types: random bytes, the scales small - d at byte 0,
       Q2_K's d and dmin at 80 and 82, IQ1_M's d in the top nibbles of its
       four scale words (48-55) */
    for (int k = 0; k < 7; k++) {
        size_t bs = janas_qtype_block_size(tq[k]);
        wqbytes[k] = page((size_t)MAXR * (MAXC / JANAS_QK) * bs + 4096);
        wq[k] = janas_aligned_alloc(4096, wqbytes[k]);
        fill(wq[k], wqbytes[k]);
        for (size_t b = 0; b + bs <= wqbytes[k] - 4096; b += bs) {
            uint16_t d = janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-6f);
            if (tq[k] == JANAS_Q2_K) {
                uint16_t dm =
                    janas_fp32_to_fp16(0.0002f + (rnd() % 1000) * 1e-6f);
                memcpy(wq[k] + b + 80, &d, 2);
                memcpy(wq[k] + b + 82, &dm, 2);
            } else if (tq[k] == JANAS_IQ1_M) {
                for (int s = 0; s < 4; s++) {
                    uint16_t sc;
                    memcpy(&sc, wq[k] + b + 48 + 2 * s, 2);
                    sc = (uint16_t)((sc & 0x0fff) | ((d >> 4 * s & 15) << 12));
                    memcpy(wq[k] + b + 48 + 2 * s, &sc, 2);
                }
            } else {
                memcpy(wq[k] + b, &d, 2);
            }
        }
    }
    size_t xblocks = (size_t)MAXV * 2 * (MAXC / JANAS_QK);
    x = malloc(xblocks * sizeof(*x));
    float *xf = malloc(MAXC * sizeof(float));
    for (size_t v = 0; v < (size_t)MAXV * 2; v++) {
        for (int i = 0; i < MAXC; i++)
            xf[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;
        janas_q8k_quantize(xf, x + v * (MAXC / JANAS_QK), MAXC);
    }
    free(xf);
    ysz = (size_t)MAXR * MAXV * 2;
    ya = malloc(ysz * sizeof(float));
    yb = malloc(ysz * sizeof(float));
}

static int import_all(struct janas_gpu *g)
{
    for (int k = 0; k < 7; k++)
        if (janas_gpu_import(g, wq[k], wqbytes[k]) != 0 ||
            janas_gpu_import(g, w32[k], w32bytes[k]) != 0)
            return -1;
    return janas_gpu_import(g, w, wbytes) != 0 ||
                   janas_gpu_import(g, w6, w6bytes) != 0 ||
                   janas_gpu_import(g, wa, wabytes) != 0 ||
                   janas_gpu_import(g, wp, wpbytes) != 0 ||
                   janas_gpu_import(g, w5, w5bytes) != 0 ||
                   janas_gpu_import(g, w8, w8bytes) != 0 ||
                   janas_gpu_import(g, wx, wxbytes) != 0 ||
                   janas_gpu_import(g, w3[0], w3bytes) != 0 ||
                   janas_gpu_import(g, w3[1], w3bytes) != 0
               ? -1
               : 0;
}

/* a product: weight kind, shape, vector count, strided or not */
struct kase {
    int q6, s, k, strided;
};

/* The task of a case, for the CPU (gpu 0) or the GPU alone (1: the
   aligned copy where the kind is Q6_K_A), and the stride of its results;
   0 for a shape the kind does not take (a tail: blocks of 32 or 64, Q8_0
   only). */
static int make_task(const struct kase *c, int gpu, struct janas_matvec_task *t,
                     size_t *ys)
{
    int q6 = c->q6;
    /* the blocks of 32 or 64: their table's index */
    int b32 = q6 >= 7 && q6 <= 10    ? q6 - 7
              : q6 == 15             ? 4
              : q6 == 16 || q6 == 17 ? q6 - 11
                                     : -1;
    int iq = q6 >= 18 ? q6 - 18 : -1; /* the 256-block kinds */
    size_t rows = shapes[c->s][0], cols = shapes[c->s][1];
    if (cols % JANAS_QK && b32 < 0 && q6 != 6)
        return 0;
    /* the activations of a tail row: its group whole */
    size_t nb = cols / JANAS_QK, nv = nvs[c->k],
           nbx = (cols + JANAS_QK - 1) / JANAS_QK;
    size_t skip = (size_t)c->s % 2; /* a row further in, every other shape */
    size_t rb32 = b32 >= 0 ? janas_qtype_row_bytes(t32[b32], cols) : 0;
    *t = (struct janas_matvec_task){
        .type = iq >= 0    ? tq[iq]
                : b32 >= 0 ? t32[b32]
                : q6 == 14 ? (gpu ? JANAS_Q6_K_A : JANAS_Q6_K)
                : q6 == 13 ? JANAS_IQ3_S
                : q6 == 12 ? JANAS_Q3_K
                : q6 == 11 ? JANAS_IQ4_XS
                : q6 == 6  ? JANAS_Q8_0
                : q6 == 5  ? JANAS_Q5_K
                : q6 >= 2  ? JANAS_Q6_K_P
                : q6 == 1  ? JANAS_Q6_K
                           : JANAS_Q4_K,
        .w = iq >= 0    ? wq[iq] + skip * janas_qtype_block_size(tq[iq]) * nb
             : b32 >= 0 ? w32[b32] + skip * rb32
             : q6 == 14 ? (gpu ? wa + skip * JANAS_Q6KA_BLOCK * nb
                               : w6 + skip * 210 * nb)
             : q6 >= 12 ? w3[q6 - 12] + skip * 110 * nb
             : q6 == 11 ? wx + skip * 136 * nb
             : q6 == 6  ? w8 + skip * janas_qtype_row_bytes(JANAS_Q8_0, cols)
             : q6 == 5  ? w5 + skip * 176 * nb
             : q6 >= 2  ? wp + skip * 82 * nb
             : q6 == 1  ? w6 + skip * 210 * nb
                        : w + skip * 144 * nb,
        .plane = {q6 == 2 || q6 == 3 ? pl1 + skip * JANAS_Q6KP_PLANE * nb
                                     : NULL,
                  q6 == 2 ? pl0 + skip * JANAS_Q6KP_PLANE * nb : NULL},
        .x = x,
        .rows = rows,
        .cols = cols,
        .n_vec = nv,
        .x_stride = c->strided ? 2 * nbx : 0,
        .y_stride = c->strided ? rows + 5 : 0};
    *ys = c->strided ? rows + 5 : rows;
    return 1;
}

/* Units in the last place between two floats (NaN: the most). */
static uint64_t ulps(float a, float b)
{
    if (isnan(a) || isnan(b))
        return UINT64_MAX;
    int32_t i, j;
    memcpy(&i, &a, 4);
    memcpy(&j, &b, 4);
    int64_t oi = i >= 0 ? i : (int64_t)INT32_MIN - i,
            oj = j >= 0 ? j : (int64_t)INT32_MIN - j;
    return (uint64_t)(oi > oj ? oi - oj : oj - oi);
}

/* rows that differ from the CPU's, of all; the worst; the first that
   differs (vector, row, the two values) */
struct diff {
    size_t bad, all;
    uint64_t ulp;
    size_t v, r;
    float a, b;
};

static void compare(size_t nv, size_t rows, size_t ys, struct diff *d)
{
    memset(d, 0, sizeof(*d));
    d->all = nv * rows;
    for (size_t v = 0; v < nv; v++)
        for (size_t r = 0; r < rows; r++) {
            float a = ya[v * ys + r], b = yb[v * ys + r];
            if (memcmp(&a, &b, 4) == 0)
                continue;
            if (!d->bad++) {
                d->v = v;
                d->r = r;
                d->a = a;
                d->b = b;
            }
            uint64_t u = ulps(a, b);
            if (u > d->ulp)
                d->ulp = u;
        }
}

static void run_cpu(const struct kase *c)
{
    struct janas_matvec_task t;
    size_t ys;
    make_task(c, 0, &t, &ys);
    memset(ya, 0, ysz * sizeof(float));
    t.y = ya;
    janas_matvec_q4k_group(pool, &t, 1);
}

/* Every row on the GPU: 0 and the shader's name, -1 where the GPU does not
   take the product (the CPU does it in the engine), -2 where the call
   failed. */
static int run_gpu(struct janas_gpu *g, const struct kase *c, struct diff *d,
                   char *shader, size_t len)
{
    struct janas_matvec_task t;
    size_t ys;
    make_task(c, 1, &t, &ys);
    memset(yb, 0, ysz * sizeof(float));
    t.y = yb;
    size_t rows = t.rows;
    int waited;
    shader[0] = 0;
    /* what the engine would not give the GPU (a misaligned block: only
       some drivers read words off a 4-byte boundary) the CPU computes */
    if (!janas_gpu_can(g, &t) || janas_gpu_begin(g, &t, &rows, 1) != 0)
        return -1;
    if (janas_gpu_finish(g, &waited) != 0)
        return -2;
    janas_gpu_last_shader(g, shader, len);
    compare(t.n_vec, t.rows, ys, d);
    return 0;
}

/* As the engine does: the rows shared between GPU and CPU, the share
   adapting over several calls; results must not change. */
static void run_shared(struct janas_gpu *g, const struct kase *c,
                       struct diff *d)
{
    struct janas_matvec_task t;
    size_t ys;
    make_task(c, 0, &t, &ys);
    memset(yb, 0, ysz * sizeof(float));
    t.y = yb;
    const char *gap = getenv("GPU_CHECK_GAP_US");
    for (int n = 0; n < 4; n++) {
        janas_gpu_matvec_group(g, pool, &t, 1);
        /* CPU-only work between calls, as in the engine */
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
    compare(t.n_vec, t.rows, ys, d);
}

/* the products of a kind and a shader (or "shared with the CPU": right on
   the GPU alone, wrong when shared), with up to MAX_EX wrong ones kept */
enum { MAX_GROUPS = 512, MAX_EX = 3 };
struct group {
    int q6;
    char shader[96];
    int runs, wrong;
    size_t rows_bad, rows_all;
    uint64_t ulp;
    int nex;
    struct kase ex[MAX_EX];
    struct diff dex[MAX_EX];
};
static struct group groups[MAX_GROUPS];
static int n_groups;

static struct group *group_of(int q6, const char *shader)
{
    for (int i = 0; i < n_groups; i++)
        if (groups[i].q6 == q6 && strcmp(groups[i].shader, shader) == 0)
            return &groups[i];
    if (n_groups == MAX_GROUPS)
        return &groups[MAX_GROUPS - 1];
    struct group *gr = &groups[n_groups++];
    gr->q6 = q6;
    snprintf(gr->shader, sizeof(gr->shader), "%s", shader);
    return gr;
}

static void count(struct group *gr, const struct kase *c, const struct diff *d)
{
    gr->runs++;
    gr->rows_all += d->all;
    if (!d->bad)
        return;
    gr->wrong++;
    gr->rows_bad += d->bad;
    if (d->ulp > gr->ulp)
        gr->ulp = d->ulp;
    if (gr->nex < MAX_EX) {
        gr->ex[gr->nex] = *c;
        gr->dex[gr->nex++] = *d;
    }
}

static void print_case(const struct kase *c)
{
    printf("%zux%zu nv %zu%s", shapes[c->s][0], shapes[c->s][1], nvs[c->k],
           c->strided ? " strided" : "");
}

static void print_ulp(uint64_t u)
{
    if (u == UINT64_MAX)
        printf("NaN");
    else
        printf("%llu ulp", (unsigned long long)u);
}

/* a few units in the last place: rounding, not a wrong computation */
#define ROUNDING_ULP 64

static void print_diff(const struct diff *d)
{
    uint32_t ua, ub;
    memcpy(&ua, &d->a, 4);
    memcpy(&ub, &d->b, 4);
    printf("vector %zu row %zu: CPU %.9g (%08x) GPU %.9g (%08x), ", d->v, d->r,
           d->a, ua, d->b, ub);
    print_ulp(ulps(d->a, d->b));
}

static const char *verdict(uint64_t u)
{
    return u <= ROUNDING_ULP
               ? "the last bits only: the driver rounds a float operation "
                 "differently (fused, reordered or flushed)"
               : "wrong values: a shader or a driver bug";
}

/* Each wrong product again under one setting of the paths: what ran and
   whether it was right. */
static void try_paths(struct janas_gpu *g, const struct group *gr,
                      const char *what, int row, int mvd)
{
    int was_row = janas_gpu_set_path(g, JANAS_GPU_PATH_ROW, row),
        was_mvd = janas_gpu_set_path(g, JANAS_GPU_PATH_MVD, mvd);
    for (int e = 0; e < gr->nex; e++) {
        struct diff d;
        char sh[96];
        run_cpu(&gr->ex[e]);
        int rc = run_gpu(g, &gr->ex[e], &d, sh, sizeof(sh));
        if (rc == 0 && strcmp(sh, gr->shader) == 0)
            continue; /* the setting changes nothing here */
        printf("    %s, ", what);
        print_case(&gr->ex[e]);
        if (rc != 0)
            printf(": %s\n",
                   rc == -1 ? "the GPU does not take it" : "the call failed");
        else if (!d.bad)
            printf(": %s right\n", sh);
        else {
            printf(": %s wrong, %zu of %zu rows, worst ", sh, d.bad, d.all);
            print_ulp(d.ulp);
            printf("\n");
        }
    }
    janas_gpu_set_path(g, JANAS_GPU_PATH_ROW, was_row);
    janas_gpu_set_path(g, JANAS_GPU_PATH_MVD, was_mvd);
}

static int wanted(int q6)
{
    const char *f = getenv("GPU_CHECK_TYPES");
    if (!f || !*f)
        return 1;
    size_t n = strlen(names[q6]);
    for (const char *p = f; (p = strstr(p, names[q6])) != NULL; p += n)
        if ((p == f || p[-1] == ',') && (p[n] == 0 || p[n] == ','))
            return 1;
    return 0;
}

/* bf16 products: Qwen3.8-Flash-Next's mixes (10240 wide, 320 wide), a tail
   of three past the last eight, a row shorter than eight; single vectors
   too (as on a card with memory of its own). The count of wrong shapes;
   *runs and *refused set. */
static int check_bf16(struct janas_gpu *g, int *runs, int *refused,
                      uint16_t **w16p, float **bxp, float **byp)
{
    *runs = *refused = 0;
    if (!janas_bf16_lanes())
        return 0;
    static const size_t bsh[][2] = {{320, 10240}, {10240, 320}, {2560, 2560},
                                    {512, 2560},  {100, 1003},  {37, 13}};
    static const size_t bnv[] = {1, 2, 3, 4, 5, 8, 17, 63};
    enum { BW = 2560 * 2560, BX = 63 * 10240 };
    size_t xbytes = page((size_t)BX * sizeof(float));
    uint16_t *w16 = janas_aligned_alloc(4096, (size_t)BW * 2);
    float *bx = janas_aligned_alloc(4096, xbytes),
          *by = janas_aligned_alloc(4096, xbytes);
    *w16p = w16;
    *bxp = bx;
    *byp = by;
    float *bxs = malloc(xbytes), *bys = malloc(xbytes), *bref = malloc(xbytes);
    for (size_t i = 0; i < BW; i++) {
        float f = (float)((int)(rnd() % 20001) - 10000) * 1e-5f;
        uint32_t u;
        memcpy(&u, &f, 4);
        w16[i] = (uint16_t)(u >> 16);
    }
    for (size_t i = 0; i < BX; i++)
        bx[i] = bxs[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;
    janas_gpu_set_min_n(g, 1); /* single vectors too */
    int bad = 0;
    if (janas_gpu_import(g, w16, (size_t)BW * 2) != 0 ||
        janas_gpu_import(g, bx, xbytes) != 0 ||
        janas_gpu_import(g, by, xbytes) != 0) {
        printf("  bf16: the import failed\n");
        bad++;
    }
    for (size_t s = 0; !bad && s < sizeof(bsh) / sizeof(*bsh); s++)
        for (size_t k = 0; k < sizeof(bnv) / sizeof(*bnv); k++)
            for (int staged = 0; staged < 2; staged++) {
                size_t rows = bsh[s][0], cols = bsh[s][1], nv = bnv[k];
                float scale = staged ? 0.25f : 1.0f;
                const float *xin = staged ? bxs : bx;
                float *yout = staged ? bys : by;
                for (size_t v = 0; v < nv; v++)
                    for (size_t r = 0; r < rows; r++)
                        bref[v * rows + r] =
                            janas_bf16_dot(w16 + r * cols, xin + v * cols,
                                           (uint32_t)cols) *
                            scale;
                memset(yout, 0, nv * rows * sizeof(float));
                if (janas_gpu_bf16_mv(g, w16, xin, yout, rows, cols, nv,
                                      scale) != 0) {
                    (*refused)++;
                    continue;
                }
                (*runs)++;
                size_t nbad = 0, first = 0;
                uint64_t worst = 0;
                for (size_t i = 0; i < nv * rows; i++)
                    if (memcmp(&bref[i], &yout[i], 4) != 0) {
                        if (!nbad++)
                            first = i;
                        uint64_t u = ulps(bref[i], yout[i]);
                        worst = u > worst ? u : worst;
                    }
                if (nbad) {
                    printf("  bf16 %zux%zu nv %zu %s: %zu of %zu wrong, "
                           "worst ",
                           rows, cols, nv,
                           staged ? "through the stages" : "in place", nbad,
                           nv * rows);
                    print_ulp(worst);
                    printf(", e.g. %zu: CPU %.9g GPU %.9g\n", first,
                           bref[first], yout[first]);
                    bad++;
                }
            }
    free(bxs);
    free(bys);
    free(bref);
    return bad;
}

int main(void)
{
    /* the exact path: the fast prompt (on by default) rounds otherwise */
    setenv("JANAS_GPU_FAST_PROMPT", "0", 1);
    /* every shader as it is: the engine's check on opening would leave out
       the ones this driver gets wrong */
    setenv("JANAS_GPU_SELFCHECK", "0", 1);
    const char *bg = getenv("GPU_CHECK_BIG_GB");
    if (bg) {
        size_t big = (size_t)atoi(bg) << 30;
        uint8_t *b = janas_aligned_alloc(4096, big);
        if (b)
            memset(b, 1, big);
    }
    printf("gpu_check %s\n", JANAS_VERSION);
    {
        char cpu[128], feat[512];
        janas_cpu_name(cpu, sizeof cpu);
        janas_cpu_features(feat, sizeof feat);
        printf("CPU: %s\nCPU has: %s\nkernels: %s\n", cpu[0] ? cpu : "unknown",
               feat[0] ? feat : "(not known)", janas_q4k_kernel_name());
    }
    const char *off = getenv("JANAS_GPU");
    if (off && strcmp(off, "0") == 0)
        printf("(JANAS_GPU=0 turns the GPU off for the engine; gpu_check "
               "opens it anyway, to check it)\n");
    char err[256];
    struct janas_gpu *g = janas_gpu_create(err, sizeof(err));
    if (!g) {
        printf("gpu_check: no GPU (%s), skipped\n", err);
        return 0;
    }
    printf("device: %s\n", janas_gpu_about(g));
    printf("weights in the %s\n", janas_gpu_shares_host(g)
                                      ? "host's memory, read in place"
                                      : "card's own memory (a copy)");
    make_weights();
    if (import_all(g) != 0) {
        printf("gpu_check: import failed\n");
        return 1;
    }
    int cpus[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int engine = getenv("GPU_CHECK_ENGINE") != NULL;
    pool = engine ? janas_pool_create(12, cpus) : janas_pool_create(4, NULL);
    if (engine && getenv("GPU_CHECK_PIN"))
        janas_pin_current_thread(0);

    /* every product, all its rows on the GPU, then shared with the CPU */
    printf("\nproducts (each all on the GPU, then shared with the CPU):\n");
    int runs = 0, wrong = 0, refused = 0, failed = 0;
    for (int q6 = 0; q6 < N_KINDS; q6++) {
        if (!wanted(q6))
            continue;
        int k_runs = 0, k_wrong = 0, k_refused = 0, k_shared = 0;
        char used[512] = "";
        for (int s = 0; s < N_SHAPES; s++)
            for (int k = 0; k < N_NVS; k++)
                for (int strided = 0; strided < 2; strided++) {
                    struct kase c = {q6, s, k, strided};
                    struct janas_matvec_task t;
                    size_t ys;
                    if (!make_task(&c, 0, &t, &ys))
                        continue;
                    run_cpu(&c);
                    struct diff d;
                    char sh[96];
                    int rc = run_gpu(g, &c, &d, sh, sizeof(sh));
                    k_runs++;
                    if (rc == -1) {
                        k_refused++;
                    } else if (rc == -2) {
                        failed++;
                        printf("  %s ", names[q6]);
                        print_case(&c);
                        printf(": the GPU's call failed\n");
                    } else {
                        struct group *gr = group_of(q6, sh);
                        count(gr, &c, &d);
                        k_wrong += d.bad != 0;
                        if (!strstr(used, sh) &&
                            strlen(used) + strlen(sh) + 3 < sizeof(used)) {
                            if (used[0])
                                strcat(used, ", ");
                            strcat(used, sh);
                        }
                    }
                    /* the aligned copy has no CPU counterpart to share
                       with; a product the GPU got wrong alone is not
                       counted twice */
                    if (q6 == 14 || (rc == 0 && d.bad))
                        continue;
                    run_shared(g, &c, &d);
                    if (d.bad) {
                        count(group_of(q6, "shared with the CPU"), &c, &d);
                        k_shared++;
                    }
                }
        runs += k_runs;
        wrong += k_wrong + k_shared;
        refused += k_refused;
        printf("  %-9s %-6s %3d products", names[q6],
               k_wrong || k_shared ? "WRONG" : "ok", k_runs);
        if (k_wrong)
            printf(", %d wrong on the GPU", k_wrong);
        if (k_shared)
            printf(", %d wrong shared with the CPU", k_shared);
        if (k_refused)
            printf(", %d left to the CPU", k_refused);
        printf(" [%s]\n", used[0] ? used : "no GPU shader");
        fflush(stdout);
    }

    /* the bf16 products */
    int b16_runs, b16_refused;
    uint16_t *w16 = NULL;
    float *bx = NULL, *by = NULL;
    int b16_bad = check_bf16(g, &b16_runs, &b16_refused, &w16, &bx, &by);
    printf("  %-9s %-6s %3d products", "bf16",
           b16_bad || b16_refused ? "WRONG" : "ok", b16_runs);
    if (b16_bad)
        printf(", %d wrong", b16_bad);
    if (b16_refused)
        printf(", %d refused", b16_refused);
    printf("\n");

    /* the wrong ones, by shader */
    int culprits = 0;
    for (int i = 0; i < n_groups; i++) {
        struct group *gr = &groups[i];
        if (!gr->wrong)
            continue;
        if (!culprits++)
            printf("\nwrong products, by shader:\n");
        printf("  %s on %s: %d of %d products wrong, %zu of %zu rows, "
               "worst ",
               gr->shader, names[gr->q6], gr->wrong, gr->runs, gr->rows_bad,
               gr->rows_all);
        print_ulp(gr->ulp);
        printf(" - %s\n", verdict(gr->ulp));
        for (int e = 0; e < gr->nex; e++) {
            printf("    e.g. ");
            print_case(&gr->ex[e]);
            printf(": %zu of %zu rows, ", gr->dex[e].bad, gr->dex[e].all);
            print_diff(&gr->dex[e]);
            printf("\n");
        }
    }

    /* isolating them: the other paths the engine could take */
    if (culprits) {
        printf("\nthe same products by other ways:\n");
        for (int i = 0; i < n_groups; i++) {
            struct group *gr = &groups[i];
            if (!gr->wrong || strcmp(gr->shader, "shared with the CPU") == 0)
                continue;
            printf("  %s on %s:\n", gr->shader, names[gr->q6]);
            try_paths(g, gr, "without the subgroup-per-row shaders", 0, 1);
            try_paths(g, gr, "without the lanes-on-vectors shaders", 1, 0);
            try_paths(g, gr, "without both", 0, 0);
        }
    }
    unsigned sg = janas_gpu_subgroup(g), sg_min, sg_max;
    janas_gpu_subgroup_range(g, &sg_min, &sg_max);
    char name[256];
    snprintf(name, sizeof(name), "%s", janas_gpu_name(g));
    janas_gpu_destroy(g);
    janas_aligned_free(w16);
    janas_aligned_free(bx);
    janas_aligned_free(by);

    /* ... and with every other subgroup size the device allows */
    if (culprits && sg_min < sg_max) {
        printf("\nthe same products with other subgroup sizes:\n");
        for (unsigned z = sg_min; z <= sg_max; z *= 2) {
            if (z == sg)
                continue;
            char zs[16];
            snprintf(zs, sizeof(zs), "%u", z);
            setenv("JANAS_GPU_SG", zs, 1);
            struct janas_gpu *g2 = janas_gpu_create(err, sizeof(err));
            if (!g2 || import_all(g2) != 0) {
                printf("  subgroups of %u: the GPU did not open (%s)\n", z,
                       g2 ? "import failed" : err);
                if (g2)
                    janas_gpu_destroy(g2);
                continue;
            }
            for (int i = 0; i < n_groups; i++) {
                struct group *gr = &groups[i];
                if (!gr->wrong ||
                    strcmp(gr->shader, "shared with the CPU") == 0)
                    continue;
                for (int e = 0; e < gr->nex; e++) {
                    struct diff d;
                    char sh[96];
                    run_cpu(&gr->ex[e]);
                    int rc = run_gpu(g2, &gr->ex[e], &d, sh, sizeof(sh));
                    printf("  subgroups of %u, %s ", z, names[gr->q6]);
                    print_case(&gr->ex[e]);
                    if (rc != 0)
                        printf(": %s\n", rc == -1 ? "the GPU does not take it"
                                                  : "the call failed");
                    else if (!d.bad)
                        printf(": %s right\n", sh);
                    else {
                        printf(": %s wrong, %zu of %zu rows, worst ", sh, d.bad,
                               d.all);
                        print_ulp(d.ulp);
                        printf("\n");
                    }
                }
            }
            janas_gpu_destroy(g2);
        }
    }

    int bad = wrong || failed || b16_bad || b16_refused;
    printf("\nsummary: %d products, %d wrong, %d left to the CPU, %d calls "
           "failed; bf16 %d, %d wrong, %d refused; %d shader%s at fault\n",
           runs, wrong, refused, failed, b16_runs, b16_bad, b16_refused,
           culprits, culprits == 1 ? "" : "s");
    if (bad)
        printf("Please paste everything above into an issue at "
               "https://github.com/prabanta-dev/janas/issues\n");
    printf("gpu_check (%s, subgroups of %u): %d shapes, bf16 %d (%d "
           "refused), %s\n",
           name, sg, runs, b16_runs, b16_refused, bad ? "FAILED" : "ok");
    janas_pool_destroy(pool);
    janas_aligned_free(w);
    janas_aligned_free(w6);
    janas_aligned_free(wa);
    janas_aligned_free(wp);
    janas_aligned_free(w5);
    janas_aligned_free(w8);
    for (int k = 0; k < 7; k++) {
        janas_aligned_free(w32[k]);
        janas_aligned_free(wq[k]);
    }
    janas_aligned_free(wx);
    janas_aligned_free(w3[0]);
    janas_aligned_free(w3[1]);
    free(x);
    free(ya);
    free(yb);
    return bad ? 1 : 0;
}
