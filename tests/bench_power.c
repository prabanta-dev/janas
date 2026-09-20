/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bench_power.c - time and package energy of Q4_K products on the CPU
 * (all compute threads, or one per physical core) and on the GPU, to see
 * which one does more work per joule under the shared power limit.
 *
 * Eight 12288 x 2048 matrices (113 MB, beyond the caches) are used in turn,
 * as the weights of a model would be. Energy from the RAPL package counter
 * (/sys/class/powercap/intel-rapl:0/energy_uj, readable by root only unless
 * made readable). Usage: bench_power [seconds per case]
 * bench_power <seconds> gpu: only GPU products over 64 vectors, for that
 * long, throughput and clock printed every second (a steady GPU load to run
 * next to something else).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <unistd.h>

#include "llm/gpu.h"
#include "llm/matvec.h"
#include "llm/quant.h"

#define ROWS 12288
#define COLS 2048
#define NMAT 8
#define MAXV 64

static const char *RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj";
static double wrap_uj;

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double energy_j(void)
{
    FILE *f = fopen(RAPL, "r");
    unsigned long long uj = 0;
    if (!f || fscanf(f, "%llu", &uj) != 1) {
        fprintf(stderr, "cannot read %s\n", RAPL);
        exit(1);
    }
    fclose(f);
    return (double)uj * 1e-6;
}

static double energy_since(double e0)
{
    double e = energy_j();
    return e >= e0 ? e - e0 : e + wrap_uj * 1e-6 - e0;
}

/* the GPU's actual frequency, sampled every millisecond */
static atomic_int sampling;
static double freq_sum;
static long freq_n;

static void *sample_freq(void *arg)
{
    (void)arg;
    while (atomic_load(&sampling)) {
        FILE *f = fopen("/sys/class/drm/card0/gt_act_freq_mhz", "r");
        int mhz;
        if (f && fscanf(f, "%d", &mhz) == 1) {
            freq_sum += mhz;
            freq_n++;
        }
        if (f)
            fclose(f);
        usleep(1000);
    }
    return NULL;
}

struct ctx {
    uint8_t *w;
    struct janas_block_q8k *x;
    float *y;
    struct janas_pool *pool;
    struct janas_gpu *gpu;
};

/* One product of matrix i with nv vectors: on_gpu 0 the pool, 1 the GPU,
   2 split between them. */
static void product(struct ctx *c, int i, size_t nv, int on_gpu)
{
    size_t nb = COLS / JANAS_QK;
    struct janas_matvec_task t = {.type = JANAS_Q4_K,
                                  .w = c->w + (size_t)i * ROWS * nb * 144,
                                  .x = c->x,
                                  .y = c->y,
                                  .rows = ROWS,
                                  .cols = COLS,
                                  .n_vec = nv};
    if (on_gpu == 2) {
        janas_gpu_matvec_group(c->gpu, c->pool, &t, 1);
    } else if (on_gpu) {
        size_t rows = ROWS;
        int waited;
        if (janas_gpu_begin(c->gpu, &t, &rows, 1) != 0 ||
            janas_gpu_finish(c->gpu, &waited) != 0) {
            fprintf(stderr, "GPU product failed\n");
            exit(1);
        }
    } else {
        janas_matvec_q4k_group(c->pool, &t, 1);
    }
}

static void run(struct ctx *c, const char *what, size_t nv, int on_gpu,
                double secs, double idle_w)
{
    for (int i = 0; i < NMAT; i++) /* warm-up */
        product(c, i, nv, on_gpu);
    pthread_t th;
    freq_sum = 0;
    freq_n = 0;
    atomic_store(&sampling, 1);
    pthread_create(&th, NULL, sample_freq, NULL);
    double e0 = energy_j(), t0 = now(), t;
    long n = 0;
    do
        product(c, (int)(n++ % NMAT), nv, on_gpu);
    while ((t = now() - t0) < secs);
    double e = energy_since(e0);
    atomic_store(&sampling, 0);
    pthread_join(th, NULL);
    double w = e / t, gmac = (double)ROWS * COLS * nv * n / 1e9;
    printf("%-14s n=%2zu: %7.3f ms, %5.1f W (%5.1f above idle), %6.1f GMAC/s, "
           "%5.2f GMAC/J (%5.2f above idle), GPU %4.0f MHz\n",
           what, nv, 1e3 * t / n, w, w - idle_w, gmac / t, gmac / e,
           gmac / (e - idle_w * t), freq_n ? freq_sum / freq_n : 0.0);
}

int main(int argc, char **argv)
{
    double secs = argc > 1 ? atof(argv[1]) : 2.0;
    FILE *f =
        fopen("/sys/class/powercap/intel-rapl:0/max_energy_range_uj", "r");
    unsigned long long mx = 0;
    if (f && fscanf(f, "%llu", &mx) == 1)
        wrap_uj = (double)mx;
    if (f)
        fclose(f);

    struct ctx c;
    size_t nb = COLS / JANAS_QK, wbytes = (size_t)NMAT * ROWS * nb * 144;
    wbytes = (wbytes + 4095) / 4096 * 4096;
    c.w = aligned_alloc(4096, wbytes);
    srand(1);
    for (size_t i = 0; i < wbytes; i++)
        c.w[i] = (uint8_t)rand();
    struct janas_block_q4k *wb = (struct janas_block_q4k *)c.w;
    for (size_t b = 0; b < wbytes / 144; b++) {
        wb[b].d = janas_fp32_to_fp16(0.001f);
        wb[b].dmin = janas_fp32_to_fp16(0.001f);
    }
    c.x = malloc((size_t)MAXV * nb * sizeof(*c.x));
    float xf[COLS];
    for (int v = 0; v < MAXV; v++) {
        for (int i = 0; i < COLS; i++)
            xf[i] = (float)(rand() % 2001 - 1000) / 500.0f;
        janas_q8k_quantize(xf, c.x + (size_t)v * nb, COLS);
    }
    c.y = malloc((size_t)MAXV * ROWS * sizeof(float));

    char gerr[256];
    int gpu_only = argc > 2 && strcmp(argv[2], "gpu") == 0;
    if (gpu_only) {
        c.gpu = janas_gpu_create(gerr, sizeof(gerr));
        if (!c.gpu || janas_gpu_import(c.gpu, c.w, wbytes) != 0)
            return 1;
        double start = now(), next = start + 1;
        long n = 0;
        pthread_t th;
        freq_sum = 0;
        freq_n = 0;
        atomic_store(&sampling, 1);
        pthread_create(&th, NULL, sample_freq, NULL);
        while (now() - start < secs) {
            product(&c, (int)(n++ % NMAT), MAXV, 1);
            double t = now();
            if (t >= next) {
                printf("t %5.1f s: %6.1f GMAC/s, GPU %4.0f MHz\n", t - start,
                       (double)ROWS * COLS * MAXV * n / 1e9 / (t - next + 1),
                       freq_n ? freq_sum / freq_n : 0.0);
                fflush(stdout);
                n = 0;
                freq_sum = 0;
                freq_n = 0;
                next = t + 1;
            }
        }
        atomic_store(&sampling, 0);
        pthread_join(th, NULL);
        janas_gpu_destroy(c.gpu);
        return 0;
    }
    /* idle package power */
    double e0 = energy_j(), t0 = now();
    usleep(1500000);
    double idle_w = energy_since(e0) / (now() - t0);
    printf("idle: %.1f W\n", idle_w);

    const size_t nvs[] = {1, 8, 64};
    int all[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int cores[6] = {0, 1, 3, 6, 8, 10}; /* one thread per P-core */
    c.gpu = NULL;
    c.pool = janas_pool_create(12, all);
    janas_pin_current_thread(0);
    for (size_t k = 0; k < 3; k++)
        run(&c, "CPU 12 thr", nvs[k], 0, secs, idle_w);
    janas_pool_destroy(c.pool);
    c.pool = janas_pool_create(6, cores);
    for (size_t k = 0; k < 3; k++)
        run(&c, "CPU 6 cores", nvs[k], 0, secs, idle_w);
    char err[256];
    c.gpu = janas_gpu_create(err, sizeof(err));
    if (!c.gpu || janas_gpu_import(c.gpu, c.w, wbytes) != 0) {
        printf("no GPU: %s\n", c.gpu ? "import failed" : err);
        return 0;
    }
    for (size_t k = 0; k < 3; k++)
        run(&c, "GPU", nvs[k], 1, secs, idle_w);
    for (size_t k = 1; k < 3; k++)
        run(&c, "6 cores + GPU", nvs[k], 2, secs, idle_w);
    janas_pool_destroy(c.pool);
    c.pool = janas_pool_create(12, all);
    for (size_t k = 1; k < 3; k++)
        run(&c, "12 thr + GPU", nvs[k], 2, secs, idle_w);
    janas_gpu_destroy(c.gpu);
    janas_pool_destroy(c.pool);
    return 0;
}
