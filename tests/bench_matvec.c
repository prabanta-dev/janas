/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bench_matvec.c - bandwidth of the Q4_K matrix-vector product.
 *
 * Decode is memory-bound: the figure of merit is how many bytes of weights per
 * second the product consumes, compared with the machine's read bandwidth.
 *
 * Usage: bench_matvec <weights_MiB> <cols> <reps> [cpu,cpu,...] [q4k|q6k]
 * Without a CPU list one thread is used. Prints the median over reps.
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

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
                "usage: bench_matvec <weights_MiB> <cols> <reps> [cpu,...]\n");
        return 2;
    }
    size_t bytes = (size_t)atol(argv[1]) << 20;
    size_t cols = (size_t)atol(argv[2]);
    int reps = atoi(argv[3]);
    if (cols % JANAS_QK || reps < 1 || reps > 1000) {
        fprintf(stderr, "cols must be a multiple of %d, reps in 1..1000\n",
                JANAS_QK);
        return 2;
    }

    int cpus[256], n = 0;
    if (argc > 4)
        for (char *t = strtok(argv[4], ","); t && n < 256;
             t = strtok(NULL, ","))
            cpus[n++] = atoi(t);
    struct janas_pool *pool = janas_pool_create(n ? n : 1, n ? cpus : NULL);
    if (!pool)
        return 1;
    if (n)
        janas_pin_current_thread(cpus[0]);

    int type =
        argc > 5 && strcmp(argv[5], "q6k") == 0 ? JANAS_Q6_K : JANAS_Q4_K;
    size_t bsz = janas_qtype_block_size(type);
    size_t nb = cols / JANAS_QK;
    size_t rows = bytes / (nb * bsz);
    size_t wbytes = rows * nb * bsz;
    void *w = malloc(wbytes);
    struct janas_block_q8k *x = malloc(nb * sizeof(*x));
    float *xf = malloc(cols * sizeof(float));
    float *y = malloc(rows * sizeof(float));
    if (!w || !x || !xf || !y)
        return 1;

    unsigned char *p = (unsigned char *)w;
    for (size_t i = 0; i < wbytes; i++)
        p[i] = (unsigned char)(i * 2654435761u >> 13);
    for (size_t b = 0; b < rows * nb; b++) { /* sane fp16 scales */
        if (type == JANAS_Q6_K) {
            ((struct janas_block_q6k *)w)[b].d = janas_fp32_to_fp16(0.001f);
        } else {
            ((struct janas_block_q4k *)w)[b].d = janas_fp32_to_fp16(0.001f);
            ((struct janas_block_q4k *)w)[b].dmin = janas_fp32_to_fp16(0.0005f);
        }
    }
    for (size_t i = 0; i < cols; i++)
        xf[i] = (float)((int)(i % 17) - 8) * 0.1f;
    janas_q8k_quantize(xf, x, cols);

    struct janas_matvec_task task = {
        .type = type, .w = w, .x = x, .y = y, .rows = rows, .cols = cols};
    janas_matvec_q4k_group(pool, &task, 1); /* warm-up */
    double gbs[1000];
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        janas_matvec_q4k_group(pool, &task, 1);
        gbs[r] = (double)wbytes / (now() - t0) / 1e9;
    }
    qsort(gbs, (size_t)reps, sizeof(double), cmp_double);
    printf("%s kernel %s  threads %d  weights %.1f MB  rows %zu  cols %zu  "
           "median %.2f GB/s  min %.2f  max %.2f  (y0 %g)\n",
           type == JANAS_Q6_K ? "Q6_K" : "Q4_K", janas_q4k_kernel_name(),
           janas_pool_size(pool), wbytes / 1e6, rows, cols, gbs[reps / 2],
           gbs[0], gbs[reps - 1], y[0]);

    janas_pool_destroy(pool);
    free(w);
    free(x);
    free(xf);
    free(y);
    return 0;
}
