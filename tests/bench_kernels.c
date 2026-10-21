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
 * BENCH_MEM=driver puts the weights in memory allocated by the GPU's driver
 * (the memory a GPU sharing the host's reads fastest), BENCH_MEM=huge2m in
 * ordinary memory aligned to 2 MB and collapsed into huge pages: the CPU's
 * speed on each, against malloc's (13 Oct 2026).
 * Defaults: 256 MiB of weights per type, 12 threads on CPUs 0-11; the
 * single core is first_cpu (default 2).
 * The weights are random bytes below 0x7c, so that every fp16 scale is a
 * finite number; the speed of a kernel does not depend on the values.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <time.h>

#include "common/aligned.h"
#include "common/pool.h"
#include "llm/matvec.h"
#include "llm/gpu.h"
#include "llm/quant.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

/* where the weights go (BENCH_MEM): malloc, the GPU driver, 2 MB pages */
static struct janas_gpu *gpu;
static int mem_kind; /* 0 malloc, 1 driver, 2 huge2m */

static unsigned char *weights(size_t bytes)
{
    if (mem_kind == 1)
        return janas_gpu_alloc(gpu, bytes);
    if (mem_kind == 2) {
        size_t al = (size_t)2 << 20;
        return janas_aligned_alloc(al, (bytes + al - 1) / al * al);
    }
    return malloc(bytes);
}

static const struct {
    const char *name;
    int type;
} TYPES[] = {
    {"Q4_K", JANAS_Q4_K},     {"Q5_K", JANAS_Q5_K},   {"Q6_K", JANAS_Q6_K},
    {"Q3_K", JANAS_Q3_K},     {"Q8_0", JANAS_Q8_0},   {"Q4_0", JANAS_Q4_0},
    {"Q4_1", JANAS_Q4_1},     {"Q5_1", JANAS_Q5_1},   {"IQ4_NL", JANAS_IQ4_NL},
    {"IQ4_XS", JANAS_IQ4_XS}, {"IQ3_S", JANAS_IQ3_S}, {"MXFP4", JANAS_MXFP4}};

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
    const char *bm = getenv("BENCH_MEM");
    mem_kind = !bm                     ? 0
               : !strcmp(bm, "driver") ? 1
               : !strcmp(bm, "huge2m") ? 2
                                       : 0;
    if (mem_kind == 1) {
        char err[200];
        gpu = janas_gpu_create(err, sizeof err);
        if (!gpu) {
            fprintf(stderr, "bench_kernels: no GPU: %s\n", err);
            return 1;
        }
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
        v[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    for (int k = 0; k < 8; k++)
        janas_q8k_quantize(v + k * cols, x + k * nb, cols);
    printf("GB/s of weights, %zu MiB a type, 2048 columns; 1 core (CPU %d) "
           "and %d threads; weights in %s\n",
           mib, one, nt,
           mem_kind == 1   ? "the GPU driver's memory"
           : mem_kind == 2 ? "2 MB pages"
                           : "malloc's memory");
    printf("%-7s %8s %8s %8s %8s | %8s %8s %8s %8s\n", "type", "1x1", "1x2",
           "1x4", "1x8", "Nx1", "Nx2", "Nx4", "Nx8");
    for (size_t k = 0; k < sizeof TYPES / sizeof *TYPES; k++) {
        size_t bs = janas_qtype_block_size(TYPES[k].type);
        size_t rows = (mib << 20) / (nb * bs), bytes = rows * nb * bs;
        unsigned char *w = weights(bytes);
        float *y = malloc(8 * rows * sizeof(float));
        if (!w || !y) {
            fprintf(stderr, "bench_kernels: out of memory\n");
            return 1;
        }
        for (size_t i = 0; i < bytes; i++)
            w[i] = (unsigned char)(rand() % 0x7c);
        /* MXFP4's scale byte is a power of two: the ones of real weights,
           not subnormal floats, whose products take the slow path */
        if (TYPES[k].type == JANAS_MXFP4)
            for (size_t i = 0; i < bytes; i += sizeof(struct janas_block_mxfp4))
                w[i] = (unsigned char)(118 + rand() % 12);
#if defined(__linux__)
        if (mem_kind == 2)
            madvise(w, bytes, MADV_COLLAPSE);
#endif
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
        if (mem_kind == 2) /* the driver's goes with the GPU */
            janas_aligned_free(w);
        else if (mem_kind == 0)
            free(w);
        free(y);
    }
    free(x);
    free(v);
    janas_pool_destroy(single);
    janas_pool_destroy(many);
    janas_gpu_destroy(gpu);
    return 0;
}
