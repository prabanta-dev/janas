/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_ffn_check.c - experts computed on the GPU (janas_gpu_ffn_begin) must
 * equal the CPU's bit for bit: gate and up products, the deterministic
 * SiLU and Q8_K quantization, the down product (Q4_K or Q6_K). Random
 * experts, two geometries. Skips (exit 0) when no GPU is usable.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/gpu.h"
#include "llm/quant.h"
#include "llm/vmath.h"

static uint64_t rng = 0x9E3779B97F4A7C15ull;

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

/* rows x cols weights of a type, random with small scales */
static void fill(uint8_t *w, int type, size_t rows, size_t cols)
{
    size_t nblk = rows * cols / JANAS_QK;
    size_t bs = janas_qtype_block_size(type);
    for (size_t i = 0; i < nblk * bs; i++)
        w[i] = (uint8_t)rnd();
    for (size_t b = 0; b < nblk; b++) {
        if (type == JANAS_Q4_K) {
            struct janas_block_q4k *q = (struct janas_block_q4k *)w + b;
            q->d = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
            q->dmin = janas_fp32_to_fp16(0.001f + (rnd() % 1000) * 1e-5f);
        } else {
            struct janas_block_q6k *q = (struct janas_block_q6k *)w + b;
            q->d = janas_fp32_to_fp16(0.0001f + (rnd() % 1000) * 1e-6f);
        }
    }
}

static void ref_product(int type, const uint8_t *w, size_t rows, size_t cols,
                        const struct janas_block_q8k *x, float *y)
{
    size_t nb = cols / JANAS_QK, rb = nb * janas_qtype_block_size(type);
    for (size_t r = 0; r < rows; r++)
        y[r] = janas_dot(type, w + r * rb, x, nb);
}

int main(void)
{
    char err[256];
    struct janas_gpu *g = janas_gpu_create(err, sizeof(err));
    if (!g) {
        printf("gpu_ffn_check: no GPU (%s), skipped\n", err);
        return 0;
    }
    const size_t geo[][3] = {{2048, 512, 10}, {3072, 1024, 8}}; /* d, ff, n */
    int bad = 0;
    for (size_t k = 0; k < 2; k++) {
        size_t d = geo[k][0], ff = geo[k][1], n = geo[k][2];
        size_t gub = ff * d / JANAS_QK * 144, dq4 = d * ff / JANAS_QK * 144,
               dq6 = d * ff / JANAS_QK * 210 + 4;
        size_t per = gub * 2 + (dq4 > dq6 ? dq4 : dq6);
        size_t bytes = (n * ((per + 255) / 256 * 256) + 4095) / 4096 * 4096;
        uint8_t *w = aligned_alloc(4096, bytes);
        memset(w, 0, bytes);
        struct janas_gpu_expert e[16];
        for (size_t i = 0; i < n; i++) {
            uint8_t *base = w + i * ((per + 255) / 256 * 256);
            int dt = i % 2 ? JANAS_Q6_K : JANAS_Q4_K;
            fill(base, JANAS_Q4_K, ff, d);
            fill(base + gub, JANAS_Q4_K, ff, d);
            fill(base + 2 * gub, dt, d, ff);
            e[i] = (struct janas_gpu_expert){.gate = base,
                                             .up = base + gub,
                                             .down = base + 2 * gub,
                                             .gate_type = JANAS_Q4_K,
                                             .up_type = JANAS_Q4_K,
                                             .down_type = dt};
        }
        if (janas_gpu_import(g, w, bytes) != 0) {
            printf("gpu_ffn_check: import failed\n");
            return 1;
        }
        float *xf = malloc(d * sizeof(float));
        struct janas_block_q8k *x = malloc(d / JANAS_QK * sizeof(*x));
        for (size_t i = 0; i < d; i++)
            xf[i] = (float)((int)(rnd() % 2001) - 1000) / 400.0f;
        janas_q8k_quantize(xf, x, d);
        float *gpu = malloc(n * d * sizeof(float));
        int waited;
        if (janas_gpu_ffn_begin(g, e, n, x, d, ff, gpu) != 0 ||
            janas_gpu_finish(g, &waited) != 0) {
            printf("gpu_ffn_check: GPU call failed\n");
            return 1;
        }
        float *gt = malloc(ff * sizeof(float)),
              *up = malloc(ff * sizeof(float));
        float *h = malloc(ff * sizeof(float)), *ref = malloc(d * sizeof(float));
        struct janas_block_q8k *hq = malloc(ff / JANAS_QK * sizeof(*hq));
        for (size_t i = 0; i < n; i++) {
            ref_product(JANAS_Q4_K, e[i].gate, ff, d, x, gt);
            ref_product(JANAS_Q4_K, e[i].up, ff, d, x, up);
            for (size_t r = 0; r < ff; r++)
                h[r] = janas_silu_mul_det(gt[r], up[r]);
            janas_q8k_quantize_det(h, hq, ff);
            ref_product(e[i].down_type, e[i].down, d, ff, hq, ref);
            size_t diff = 0;
            for (size_t r = 0; r < d; r++)
                diff += memcmp(&ref[r], &gpu[i * d + r], 4) != 0;
            if (diff) {
                printf("d %zu ff %zu expert %zu (down %s): %zu of %zu differ, "
                       "e.g. %g vs %g\n",
                       d, ff, i, e[i].down_type == JANAS_Q6_K ? "Q6_K" : "Q4_K",
                       diff, d, ref[0], gpu[i * d]);
                bad++;
            }
        }
        free(gt);
        free(up);
        free(h);
        free(ref);
        free(hq);
        free(xf);
        free(x);
        free(gpu);
    }
    printf("gpu_ffn_check (%s, subgroups of %u): %s\n", janas_gpu_name(g),
           janas_gpu_subgroup(g), bad ? "FAILED" : "ok");
    janas_gpu_destroy(g);
    return bad ? 1 : 0;
}
