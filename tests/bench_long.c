/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bench_long.c - generation after a long prompt, within a memory budget.
 *
 * Runs n_prompt tokens as the prefill, in blocks, then generates n_new tokens
 * greedily one at a time, and reports for both the speed, the expert cache
 * (hits, bytes read, time waited for reads) and the time per phase, then the
 * peak resident memory of the whole process.
 *
 * Usage: bench_long <model.jns> <prompt.int32> <n_prompt> <n_new> <cache_MiB>
 *                   [compute_threads]
 * Without the last one, twelve threads on CPUs 0-11: what every measure of
 * this file has used so far. With it, the first threads CPUs of the layout
 * the engine itself picks (P cores first, then E), which is what janas-chat
 * runs on.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "common/pool.h"
#include "common/sysinfo.h"
#include "llm/model.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int32_t argmax(const float *v, uint32_t n)
{
    uint32_t b = 0;
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > v[b])
            b = i;
    return (int32_t)b;
}

static void report_phases(const char *what, const double *a, const double *b,
                          uint32_t n)
{
    static const char *names[JANAS_PH_COUNT] = {
        "qkv", "attention", "wo", "router", "experts", "output", "ssm"};
    double total = 0;
    printf("%s, ms/token:", what);
    for (int i = 0; i < JANAS_PH_COUNT; i++) {
        printf(" %s %.2f", names[i], (b[i] - a[i]) / n * 1e3);
        total += b[i] - a[i];
    }
    printf(" (sum %.2f)\n", total / n * 1e3);
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: bench_long <model.jns> <prompt.int32> "
                        "<n_prompt> <n_new> <cache_MiB>\n");
        return 2;
    }
    uint32_t np = (uint32_t)atoi(argv[3]), nn = (uint32_t)atoi(argv[4]);
    int32_t *prompt = malloc((size_t)np * sizeof(int32_t));
    int32_t *out = malloc((size_t)nn * sizeof(int32_t));
    FILE *f = fopen(argv[2], "rb");
    if (!prompt || !out || !f || np == 0 || nn == 0 ||
        fread(prompt, sizeof(int32_t), np, f) != np) {
        fprintf(stderr, "cannot read %u prompt tokens\n", np);
        return 1;
    }
    fclose(f);
    int ccpus[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int icpus[8] = {12, 13, 14, 15, 16, 17, 18, 19};
    const int *compute = ccpus;
    int n_compute = 12;
    struct janas_cpu_layout lay;
    if (argc > 6 && janas_cpu_layout(NULL, 0, &lay) == 0) {
        n_compute = atoi(argv[6]);
        if (n_compute < 1 || n_compute > lay.n)
            n_compute = lay.n;
        compute = lay.cpus;
    }
    printf("compute threads: %d\n", n_compute);
    struct janas_llm_options mo = {.cache_bytes = (uint64_t)atol(argv[5]) << 20,
                                   .n_ctx = np + nn + 16,
                                   .compute_cpus = compute,
                                   .n_compute = n_compute,
                                   .io_cpus = icpus,
                                   .n_io = 8};
    char err[256];
    struct janas_llm_model *m =
        janas_llm_model_load(argv[1], &mo, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "%s: %s\n", argv[1], err);
        return 1;
    }
    uint32_t nv = janas_llm_model_n_vocab(m);
    float *logits = malloc((size_t)nv * sizeof(float));
    struct janas_expert_cache *c = janas_llm_model_cache(m);
    struct janas_expert_cache_stats cs;
    double ph[3][JANAS_PH_COUNT], t0 = now();
    if (!logits)
        return 1;
    janas_llm_model_phases(m, ph[0]);
    uint32_t most = janas_llm_model_max_block(m);
    for (uint32_t p = 0; p < np; p += most) {
        uint32_t b = np - p < most ? np - p : most;
        if (janas_llm_model_forward(m, prompt + p, b, p, logits, 0) != 0) {
            fprintf(stderr, "prefill failed at %u\n", p);
            return 1;
        }
    }
    double t1 = now();
    janas_llm_model_phases(m, ph[1]);
    janas_expert_cache_stats(c, &cs);
    printf("prefill %u tokens: %.1f s (%.1f token/s); experts %.1f%% from "
           "RAM, %.2f GB read, %.1f s waited\n",
           np, t1 - t0, np / (t1 - t0), 100.0 * cs.hits / cs.requests,
           cs.bytes_read / 1e9, cs.wait_seconds);
    report_phases("prefill", ph[0], ph[1], np);
    janas_pool_trace_report("prefill", t1 - t0);
    janas_pool_trace_reset();
    janas_expert_cache_reset_stats(c);
    for (uint32_t i = 0; i < nn; i++) {
        out[i] = argmax(logits, nv);
        if (janas_llm_model_decode(m, out[i], np + i, logits) != 0) {
            fprintf(stderr, "decode failed at %u\n", i);
            return 1;
        }
    }
    double t2 = now();
    janas_llm_model_phases(m, ph[2]);
    janas_expert_cache_stats(c, &cs);
    printf("decode %u tokens after %u: %.2f s (%.2f token/s); experts "
           "%.1f%% from RAM, %.2f GB read (%.1f MB/token), %.2f s waited\n",
           nn, np, t2 - t1, nn / (t2 - t1), 100.0 * cs.hits / cs.requests,
           cs.bytes_read / 1e9, cs.bytes_read / 1e6 / nn, cs.wait_seconds);
    report_phases("decode", ph[1], ph[2], nn);
    janas_pool_trace_report("decode", t2 - t1);
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("peak resident memory: %.2f GiB\n", ru.ru_maxrss / 1048576.0);
    free(prompt);
    free(out);
    free(logits);
    janas_llm_model_free(m);
    return 0;
}
