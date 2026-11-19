/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_cuda_kernels.c - the CUDA kernels' arithmetic (src/llm/cuda/), run
 * as C on the CPU one row at a time, against the CPU's own kernels: the
 * same bits for every row of every vector, every type they compute, several
 * shapes.
 * It needs no card: what it checks is that the kernels compute what the
 * CPU computes; that a card runs them as written is gpu_check's to see
 * (JANAS_GPU_API=cuda gpu_check).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/aligned.h"
#include "common/pool.h"
#include "llm/matvec.h"
#include "llm/quant.h"

#include "../src/llm/cuda/matvec.cu"

static uint32_t seed = 12345;
static uint32_t rnd(void)
{
    seed = seed * 1664525u + 1013904223u;
    return seed >> 8;
}

/* Q8_K vectors as the kernels read them (gpu_cuda.c convert_x) */
static void convert_x(uint8_t *dst, const struct janas_block_q8k *x, size_t nv,
                      size_t nb, int k32)
{
    for (size_t v = 0; v < nv; v++)
        for (size_t b = 0; b < nb; b++, dst += 320) {
            const struct janas_block_q8k *s = x + v * nb + b;
            memset(dst, 0, 320);
            if (k32) {
                memcpy(dst, s->qs32, JANAS_QK);
                memcpy(dst + JANAS_QK, s->bsums32, sizeof(s->bsums32));
                memcpy(dst + JANAS_QK + sizeof(s->bsums32), s->d32,
                       sizeof(s->d32));
            } else {
                memcpy(dst, s->qs, JANAS_QK);
                memcpy(dst + JANAS_QK, s->bsums, sizeof(s->bsums));
                memcpy(dst + JANAS_QK + sizeof(s->bsums), &s->d, sizeof(float));
            }
        }
}

/* b32_matvec's kind of a type, -1 for the others */
static int b32_kind(int type)
{
    static const int k[] = {JANAS_Q4_0,  JANAS_IQ4_NL, JANAS_Q4_1, JANAS_Q5_1,
                            JANAS_MXFP4, JANAS_Q5_0,   JANAS_Q2_0, JANAS_NVFP4};
    for (int i = 0; i < 8; i++)
        if (k[i] == type)
            return i;
    return -1;
}

/* sensible scales in a group of 256 weights of the 32-block kinds */
static void b32_scales(uint8_t *g, int kind)
{
    static const unsigned S[8] = {18, 18, 20, 24, 17, 22, 18, 36};
    unsigned n = kind == 6 || kind == 7 ? 4 : 8; /* blocks in the group */
    for (unsigned i = 0, at = 0; i < n; i++, at += S[kind]) {
        uint16_t d = janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-5f);
        if (kind == 4)
            g[at] = (uint8_t)(118 + rnd() % 12); /* E8M0: 2^-10 to 2^1 */
        else if (kind == 7)
            for (int j = 0; j < 4; j++) /* UE4M3: exponents 1 to 13 */
                g[at + j] = (uint8_t)(((1 + rnd() % 13) << 3) | (rnd() & 7));
        else {
            memcpy(g + at, &d, 2);
            if (kind == 2 || kind == 3) {
                uint16_t m =
                    janas_fp32_to_fp16(-0.01f + (rnd() % 1000) * 2e-5f);
                memcpy(g + at + 2, &m, 2);
            }
        }
    }
}

static int check(struct janas_pool *pool, int type, size_t rows, size_t cols,
                 size_t nv)
{
    size_t nb = cols / JANAS_QK, bb = janas_qtype_row_bytes(type, JANAS_QK);
    size_t wbytes = rows * nb * bb;
    uint8_t *w = janas_aligned_alloc(64, wbytes + 64);
    for (size_t i = 0; i < wbytes; i++)
        w[i] = (uint8_t)rnd();
    /* sensible scales: the halves of each block */
    for (size_t b = 0; b < rows * nb; b++) {
        uint8_t *blk = w + b * bb;
        uint16_t d = janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-5f);
        if (b32_kind(type) >= 0) {
            b32_scales(blk, b32_kind(type));
        } else if (type == JANAS_Q8_0) {
            for (size_t k = 0; k < 8; k++) { /* each block of 32 its own */
                uint16_t e =
                    janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-5f);
                memcpy(blk + 34 * k, &e, 2);
            }
        } else if (type == JANAS_IQ4_XS) {
            memcpy(blk, &d, 2); /* d first, then the scales */
        } else if (type == JANAS_Q4_K || type == JANAS_Q5_K) {
            uint16_t m = janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-5f);
            memcpy(blk, &d, 2);
            memcpy(blk + 2, &m, 2);
        } else
            memcpy(blk + bb - 2, &d, 2); /* Q6_K, Q3_K: d last */
    }
    float *xf = malloc(nv * cols * sizeof(float));
    for (size_t i = 0; i < nv * cols; i++)
        xf[i] = ((float)(rnd() % 20001) - 10000.0f) / 3000.0f;
    struct janas_block_q8k *x = janas_aligned_alloc(64, nv * nb * sizeof(*x));
    for (size_t v = 0; v < nv; v++)
        janas_q8k_quantize(xf + v * cols, x + v * nb, cols);
    uint8_t *xg = janas_aligned_alloc(64, nv * nb * 320);
    convert_x(xg, x, nv, nb, type == JANAS_Q8_0 || b32_kind(type) >= 0);
    float *ycpu = calloc(nv * rows, sizeof(float));
    float *ygpu = calloc(nv * rows, sizeof(float));
    struct janas_matvec_task t = {.type = type,
                                  .w = w,
                                  .x = x,
                                  .y = ycpu,
                                  .rows = rows,
                                  .cols = cols,
                                  .n_vec = nv,
                                  .x_stride = nb,
                                  .y_stride = rows};
    janas_matvec_q4k_group(pool, &t, 1);
    for (cuda_host_row = 0; cuda_host_row < rows; cuda_host_row++) {
        u32 r = (u32)rows, b = (u32)nb, n = (u32)nv;
        const u32 *xw = (const u32 *)xg;
        if (type == JANAS_Q4_K)
            q4k_matvec((const u32 *)w, xw, ygpu, r, b, n, b, r);
        else if (type == JANAS_Q5_K)
            q5k_matvec((const u32 *)w, xw, ygpu, r, b, n, b, r);
        else if (type == JANAS_Q6_K)
            q6k_matvec((const u16 *)w, xw, ygpu, r, b, n, b, r);
        else if (type == JANAS_Q3_K)
            q3k_matvec((const u16 *)w, xw, ygpu, r, b, n, b, r);
        else if (type == JANAS_Q8_0)
            q80_matvec((const u16 *)w, xw, ygpu, r, b, n, b, r);
        else if (type == JANAS_IQ4_XS)
            iq4xs_matvec((const u32 *)w, xw, ygpu, r, b, n, b, r);
        else
            b32_matvec(w, xw, ygpu, r, b, n, b, r, (u32)b32_kind(type));
    }
    size_t bad = 0;
    for (size_t i = 0; i < nv * rows; i++)
        if (!isfinite(ycpu[i])) { /* a check of nothing */
            fprintf(stderr, "  type %d: the CPU gave %g\n", type, ycpu[i]);
            bad++;
            break;
        }
    for (size_t i = 0; i < nv * rows; i++)
        if (memcmp(&ycpu[i], &ygpu[i], sizeof(float))) {
            if (!bad)
                fprintf(stderr,
                        "  type %d %zux%zu, %zu vectors: [%zu] %.9g, "
                        "the CPU %.9g\n",
                        type, rows, cols, nv, i, ygpu[i], ycpu[i]);
            bad++;
        }
    janas_aligned_free(w);
    janas_aligned_free(x);
    janas_aligned_free(xg);
    free(xf);
    free(ycpu);
    free(ygpu);
    return bad != 0;
}

int main(void)
{
    struct janas_pool *pool = janas_pool_create(2, NULL);
    static const size_t shapes[][2] = {
        {37, 256}, {64, 512}, {129, 1024}, {16, 2816}, {300, 2048}};
    int fails = 0, runs = 0;
    static const int types[] = {
        JANAS_Q4_K, JANAS_Q5_K,   JANAS_Q6_K,  JANAS_Q3_K,  JANAS_Q8_0,
        JANAS_Q4_0, JANAS_IQ4_NL, JANAS_Q4_1,  JANAS_Q5_1,  JANAS_MXFP4,
        JANAS_Q5_0, JANAS_Q2_0,   JANAS_NVFP4, JANAS_IQ4_XS};
    for (size_t ty = 0; ty < sizeof types / sizeof *types; ty++)
        for (size_t s = 0; s < sizeof shapes / sizeof *shapes; s++)
            for (size_t nv = 1; nv <= 5; nv += 2) {
                fails += check(pool, types[ty], shapes[s][0], shapes[s][1], nv);
                runs++;
            }
    janas_pool_destroy(pool);
    printf("test_cuda_kernels: %d of %d products the same as the CPU's, bit "
           "for bit\n",
           runs - fails, runs);
    return fails != 0;
}
