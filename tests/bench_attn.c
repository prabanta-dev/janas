/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bench_attn.c - time of the attention of one layer at a given context, on
 * random data: one token (decode) and a block of JANAS_ATTN_MAX_BLOCK.
 *
 * Usage: bench_attn [n_head n_head_kv head_dim context threads
 *                    [decode] [exact|fast]]
 * With "decode" only the one-token case runs, 2000 times (for profiling);
 * "exact" and "fast" ask for one way of computing the scores instead of
 * letting the context decide.
 * Defaults: Qwen3-Next (16 2 256), 8000 positions, 12 threads on CPUs 0-11.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/pool.h"
#include "llm/attention.h"
#include "llm/quant.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    uint32_t n_head = argc > 1 ? (uint32_t)atoi(argv[1]) : 16;
    uint32_t n_kv = argc > 2 ? (uint32_t)atoi(argv[2]) : 2;
    uint32_t hd = argc > 3 ? (uint32_t)atoi(argv[3]) : 256;
    uint32_t ctx = argc > 4 ? (uint32_t)atoi(argv[4]) : 8000;
    int threads = argc > 5 ? atoi(argv[5]) : 12;
    int decode_only = 0, scores = JANAS_ATTN_AUTO;
    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "decode") == 0)
            decode_only = 1;
        else if (strcmp(argv[i], "exact") == 0)
            scores = JANAS_ATTN_FLOAT;
        else if (strcmp(argv[i], "fast") == 0)
            scores = JANAS_ATTN_INT16;
    }
    int cpus[256];
    for (int i = 0; i < threads && i < 256; i++)
        cpus[i] = i;
    struct janas_pool *pool = janas_pool_create(threads, cpus);
    janas_pin_current_thread(0);
    uint32_t n_ctx = ctx + JANAS_ATTN_MAX_BLOCK;
    struct janas_attn *a =
        janas_attn_create(n_head, n_kv, hd, n_ctx, threads, scores);
    size_t qd = (size_t)n_head * hd, kv = (size_t)n_kv * n_ctx * hd;
    int8_t *K = malloc(kv), *V = malloc(kv);
    size_t nsc = (size_t)n_kv * n_ctx;
    float *ks = malloc(nsc * sizeof(float)), *vs = malloc(nsc * sizeof(float));
    float *q = malloc(JANAS_ATTN_MAX_BLOCK * qd * sizeof(float));
    float *out = malloc(JANAS_ATTN_MAX_BLOCK * qd * sizeof(float));
    if (!pool || !a || !K || !V || !ks || !vs || !q || !out)
        return 1;
    srand(1);
    for (size_t i = 0; i < kv; i++) {
        K[i] = (int8_t)(rand() % 255 - 127);
        V[i] = (int8_t)(rand() % 255 - 127);
    }
    for (size_t i = 0; i < nsc; i++)
        ks[i] = vs[i] = 0.01f;
    for (size_t i = 0; i < JANAS_ATTN_MAX_BLOCK * qd; i++)
        q[i] = (float)((double)rand() / RAND_MAX) - 0.5f;
    double kv_mb = 2.0 * n_kv * ctx * hd / 1e6;
    for (int rep = 0; rep < 2; rep++) {
        int iters = decode_only ? 2000 : 200;
        double t0 = now();
        for (int i = 0; i < iters; i++)
            janas_attn_run(a, pool, q, K, ks, V, vs, 1, ctx, out);
        double dt = (now() - t0) / iters;
        printf("decode at %u: %.3f ms per layer (%.1f GB/s of KV)\n", ctx,
               dt * 1e3, kv_mb / 1e3 / dt);
        if (decode_only)
            continue;
        iters = 10;
        t0 = now();
        for (int i = 0; i < iters; i++)
            janas_attn_run(a, pool, q, K, ks, V, vs, JANAS_ATTN_MAX_BLOCK,
                           ctx - JANAS_ATTN_MAX_BLOCK, out);
        dt = (now() - t0) / iters;
        printf("block of %d at %u: %.3f ms per layer (%.3f ms per token)\n",
               JANAS_ATTN_MAX_BLOCK, ctx, dt * 1e3,
               dt * 1e3 / JANAS_ATTN_MAX_BLOCK);
    }
    janas_attn_destroy(a);
    janas_pool_destroy(pool);
    free(K);
    free(V);
    free(q);
    free(out);
    return 0;
}
