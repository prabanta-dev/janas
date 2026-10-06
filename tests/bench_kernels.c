/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bench_kernels.c - every weight type's product, on one core and on many,
 * over 1 to 8 vectors, in GB/s of weights read: to set beside the memory's
 * bandwidth (about 81 GB/s on the development machine). A product of one
 * vector reads each weight once and does little with it, so a kernel well
 * below the others on one core is spending its time computing, not waiting
 * for memory - which is how the Q3_K, IQ4_XS and IQ3_S kernels were found
 * to convert their scale through a function call (6 Oct 2026).
 *
 * Usage: bench_kernels [MiB [threads [first_cpu]]]
 * Defaults: 256 MiB of weights per type, 12 threads on CPUs 0-11; the
 * single core is first_cpu (default 2).
 * The weights are random bytes below 0x7c, so that every fp16 scale is a
 * finite number; the speed of a kernel does not depend on the values.
 */
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

static const struct {
    const char *name;
    int type;
} TYPES[] = {
    {"Q4_K", JANAS_Q4_K},     {"Q5_K", JANAS_Q5_K},  {"Q6_K", JANAS_Q6_K},
    {"Q3_K", JANAS_Q3_K},     {"Q8_0", JANAS_Q8_0},  {"Q4_0", JANAS_Q4_0},
    {"Q4_1", JANAS_Q4_1},     {"Q5_1", JANAS_Q5_1},  {"IQ4_NL", JANAS_IQ4_NL},
    {"IQ4_XS", JANAS_IQ4_XS}, {"IQ3_S", JANAS_IQ3_S}};

/* Seconds per product, the best of passes run for at least 0.3 s. */
static double timed(struct janas_pool *pool, struct janas_matvec_task *t)
{
    double best = 1e9, start = now();
    int runs = 0;
    while (runs < 3 || now() - start < 0.3) {
        double t0 = now();
        janas_matvec_q4k_group(pool, t, 1);
        double dt = now() - t0;
        if (dt < best)
            best = dt;
        runs++;
    }
    return best;
}

int main(int argc, char **argv)
{
    size_t mib = argc > 1 ? (size_t)atol(argv[1]) : 256;
    int nt = argc > 2 ? atoi(argv[2]) : 12;
    int one = argc > 3 ? atoi(argv[3]) : 2;
    if (mib < 1 || nt < 1 || nt > 256 || one < 0) {
        fprintf(stderr, "usage: bench_kernels [MiB [threads [first_cpu]]]\n");
        return 2;
    }
    int cpus[256];
    for (int i = 0; i < nt; i++)
        cpus[i] = i;
    struct janas_pool *many = janas_pool_create(nt, cpus);
    struct janas_pool *single = janas_pool_create(1, &one);
    if (!many || !single) {
        fprintf(stderr, "bench_kernels: no thread pool\n");
        return 1;
    }
    const size_t cols = 2048, nb = cols / JANAS_QK, nvs[] = {1, 2, 4, 8};
    struct janas_block_q8k *x = calloc(8 * nb, sizeof(*x));
    float *v = malloc(8 * cols * sizeof(float));
    srand(1);
    for (size_t i = 0; i < 8 * cols; i++)
        v[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int k = 0; k < 8; k++)
        janas_q8k_quantize(v + k * cols, x + k * nb, cols);
    printf("GB/s of weights, %zu MiB a type, 2048 columns; 1 core (CPU %d) "
           "and %d threads\n",
           mib, one, nt);
    printf("%-7s %8s %8s %8s %8s | %8s %8s %8s %8s\n", "type", "1x1", "1x2",
           "1x4", "1x8", "Nx1", "Nx2", "Nx4", "Nx8");
    for (size_t k = 0; k < sizeof TYPES / sizeof *TYPES; k++) {
        size_t bs = janas_qtype_block_size(TYPES[k].type);
        size_t rows = (mib << 20) / (nb * bs), bytes = rows * nb * bs;
        unsigned char *w = malloc(bytes);
        float *y = malloc(8 * rows * sizeof(float));
        if (!w || !y) {
            fprintf(stderr, "bench_kernels: out of memory\n");
            return 1;
        }
        for (size_t i = 0; i < bytes; i++)
            w[i] = (unsigned char)(rand() % 0x7c);
        printf("%-7s", TYPES[k].name);
        for (int p = 0; p < 2; p++) {
            for (size_t i = 0; i < 4; i++) {
                struct janas_matvec_task t = {.type = TYPES[k].type,
                                              .w = w,
                                              .x = x,
                                              .y = y,
                                              .rows = rows,
                                              .cols = cols,
                                              .n_vec = nvs[i],
                                              .x_stride = nb,
                                              .y_stride = rows};
                double s = timed(p ? many : single, &t);
                printf(" %8.1f", bytes / s / 1e9);
            }
            printf(p ? "\n" : " |");
        }
        fflush(stdout);
        free(w);
        free(y);
    }
    free(x);
    free(v);
    janas_pool_destroy(single);
    janas_pool_destroy(many);
    return 0;
}
