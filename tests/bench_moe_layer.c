/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bench_moe_layer.c - routed-expert part of MoE decode layers, one token.
 *
 * Experts are stored contiguously (gate, up, down one after the other, the
 * on-disk layout planned for Janas-LLM) in a pool larger than the caches. For
 * every layer k experts are picked at random, gate and up run as one grouped
 * product, SiLU(gate) * up is quantized per expert, and down runs as a second
 * grouped product. Reports expert bytes consumed per second and the time per
 * layer, i.e. what the routed experts cost per token for a given model shape.
 *
 * Usage: bench_moe_layer <pool_MiB> <d_model> <d_ff> <k> <layers> <reps>
 *                        [cpu,cpu,...]
 * Example (Qwen3-Next): bench_moe_layer 4096 2048 512 10 48 5 0,1,2,...,11
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/pool.h"
#include "llm/matvec.h"
#include "llm/quant.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr, "usage: bench_moe_layer <pool_MiB> <d_model> <d_ff> "
                        "<k> <layers> <reps> [cpu,...]\n");
        return 2;
    }
    size_t pool_bytes = (size_t)atol(argv[1]) << 20;
    size_t dm = (size_t)atol(argv[2]), dff = (size_t)atol(argv[3]);
    int k = atoi(argv[4]), layers = atoi(argv[5]), reps = atoi(argv[6]);
    if (dm % JANAS_QK || dff % JANAS_QK || k < 1 || k > 64 || layers < 1 ||
        reps < 1 || reps > 1000) {
        fprintf(stderr, "d_model and d_ff multiples of %d, k in 1..64\n",
                JANAS_QK);
        return 2;
    }
    int cpus[256], n = 0;
    if (argc > 7)
        for (char *t = strtok(argv[7], ","); t && n < 256;
             t = strtok(NULL, ","))
            cpus[n++] = atoi(t);
    struct janas_pool *pool = janas_pool_create(n ? n : 1, n ? cpus : NULL);
    if (!pool)
        return 1;
    if (n)
        janas_pin_current_thread(cpus[0]);

    /* one expert: gate (dff x dm), up (dff x dm), down (dm x dff) */
    size_t mat_blocks = dff * dm / JANAS_QK;
    size_t expert_blocks = 3 * mat_blocks;
    size_t expert_bytes = expert_blocks * sizeof(struct janas_block_q4k);
    size_t n_experts = pool_bytes / expert_bytes;
    if (n_experts < (size_t)k) {
        fprintf(stderr, "pool too small for %d experts\n", k);
        return 2;
    }
    struct janas_block_q4k *experts = malloc(n_experts * expert_bytes);
    if (!experts)
        return 1;
    unsigned char *p = (unsigned char *)experts;
    for (size_t i = 0; i < n_experts * expert_bytes; i++)
        p[i] = (unsigned char)(i * 2654435761u >> 13);
    for (size_t b = 0; b < n_experts * expert_blocks; b++) {
        experts[b].d = janas_fp32_to_fp16(0.001f);
        experts[b].dmin = janas_fp32_to_fp16(0.0005f);
    }

    float *x = malloc(dm * sizeof(float));
    struct janas_block_q8k *xq = malloc(dm / JANAS_QK * sizeof(*xq));
    float *gate = malloc((size_t)k * dff * sizeof(float));
    float *up = malloc((size_t)k * dff * sizeof(float));
    float *h = malloc(dff * sizeof(float));
    struct janas_block_q8k *hq =
        malloc((size_t)k * dff / JANAS_QK * sizeof(*hq));
    float *out = malloc((size_t)k * dm * sizeof(float));
    struct janas_matvec_task t1[128], t2[64];
    for (size_t i = 0; i < dm; i++)
        x[i] = sinf((float)i * 0.1f);
    janas_q8k_quantize(x, xq, dm);

    unsigned rng = 12345;
    double *per_layer_us = malloc((size_t)reps * sizeof(double));
    double *gbs = malloc((size_t)reps * sizeof(double));
    for (int rep = -1; rep < reps; rep++) { /* rep -1 = warm-up */
        double t0 = now();
        for (int l = 0; l < layers; l++) {
            for (int e = 0; e < k; e++) {
                rng = rng * 1103515245u + 12345u;
                size_t id = (rng >> 8) % n_experts;
                const struct janas_block_q4k *ex = experts + id * expert_blocks;
                t1[2 * e] =
                    (struct janas_matvec_task){.type = JANAS_Q4_K,
                                               .w = ex,
                                               .x = xq,
                                               .y = gate + (size_t)e * dff,
                                               .rows = dff,
                                               .cols = dm};
                t1[2 * e + 1] =
                    (struct janas_matvec_task){.type = JANAS_Q4_K,
                                               .w = ex + mat_blocks,
                                               .x = xq,
                                               .y = up + (size_t)e * dff,
                                               .rows = dff,
                                               .cols = dm};
                t2[e] = (struct janas_matvec_task){.type = JANAS_Q4_K,
                                                   .w = ex + 2 * mat_blocks,
                                                   .x = hq + (size_t)e * dff /
                                                                 JANAS_QK,
                                                   .y = out + (size_t)e * dm,
                                                   .rows = dm,
                                                   .cols = dff};
            }
            janas_matvec_q4k_group(pool, t1, 2 * (size_t)k);
            for (int e = 0; e < k; e++) {
                const float *g = gate + (size_t)e * dff;
                const float *u = up + (size_t)e * dff;
                for (size_t i = 0; i < dff; i++)
                    h[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
                janas_q8k_quantize(h, hq + (size_t)e * dff / JANAS_QK, dff);
            }
            janas_matvec_q4k_group(pool, t2, (size_t)k);
        }
        double dt = now() - t0;
        if (rep >= 0) {
            per_layer_us[rep] = dt / layers * 1e6;
            gbs[rep] = (double)layers * k * expert_bytes / dt / 1e9;
        }
    }
    qsort(per_layer_us, (size_t)reps, sizeof(double), cmp_double);
    qsort(gbs, (size_t)reps, sizeof(double), cmp_double);
    double us = per_layer_us[reps / 2];
    printf("kernel %s  threads %d  expert %.2f MB  pool %zu experts  k %d  "
           "layers %d: %.1f us/layer, %.2f GB/s of expert weights, "
           "routed part %.1f ms/token (%.1f token/s)  (out0 %g)\n",
           janas_q4k_kernel_name(), janas_pool_size(pool), expert_bytes / 1e6,
           n_experts, k, layers, us, gbs[reps / 2], us * layers / 1000.0,
           1e6 / (us * layers), out[0]);

    janas_pool_destroy(pool);
    free(experts);
    free(x);
    free(xq);
    free(gate);
    free(up);
    free(h);
    free(hq);
    free(out);
    free(per_layer_us);
    free(gbs);
    return 0;
}
