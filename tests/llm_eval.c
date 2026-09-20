/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * llm_eval.c - run token ids through the Janas-LLM forward pass and compare
 * the logits with a reference (llama.cpp dump: float32 x n_tokens x n_vocab).
 *
 * Usage: llm_eval <model.jns> <tokens.int32> [reference.logits]
 *                 [cache_MiB] [compute_cpus] [io_cpus] [block=B] [dump=F]
 * With block=B the tokens are also run in blocks of B (the prefill path) and
 * the logits compared with the token-by-token ones: they must be identical.
 * With dump=F the logits are written to F in the reference format.
 * Reports, per position, whether the most likely token agrees, the largest and
 * mean absolute logit difference, and the time per token.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llm/model.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int parse_cpus(const char *s, int *cpus)
{
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", s);
    int n = 0;
    for (char *t = strtok(buf, ","); t && n < 256; t = strtok(NULL, ","))
        cpus[n++] = atoi(t);
    return n;
}

static void print_phases(const char *what, const double *a, const double *b,
                         size_t n)
{
    static const char *names[JANAS_PH_COUNT] = {
        "qkv", "attention", "wo", "router", "experts", "output", "ssm"};
    double sum = 0;
    printf("%s, ms/token:", what);
    for (int i = 0; i < JANAS_PH_COUNT; i++) {
        printf(" %s %.2f", names[i], (b[i] - a[i]) / n * 1e3);
        sum += b[i] - a[i];
    }
    printf(" (sum %.2f)\n", sum / n * 1e3);
}

static uint32_t argmax(const float *v, uint32_t n)
{
    uint32_t b = 0;
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > v[b])
            b = i;
    return b;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: llm_eval <model.jns> <tokens> [reference] "
                        "[cache_MiB] [compute_cpus] [io_cpus]\n");
        return 2;
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f)
        return 1;
    int32_t tokens[4096];
    size_t n = fread(tokens, sizeof(int32_t), 4096, f);
    fclose(f);
    FILE *ref = argc > 3 && strcmp(argv[3], "-") ? fopen(argv[3], "rb") : NULL;
    FILE *dump = NULL; /* dump=<file>: the logits, as a reference file */
    for (int i = 7; i < argc; i++)
        if (strncmp(argv[i], "dump=", 5) == 0)
            dump = fopen(argv[i] + 5, "wb");

    int ccpus[256], icpus[256];
    int nc =
        parse_cpus(argc > 5 ? argv[5] : "0,1,2,3,4,5,6,7,8,9,10,11", ccpus);
    int ni = parse_cpus(argc > 6 ? argv[6] : "12,13,14,15,16,17,18,19", icpus);
    struct janas_llm_options o = {
        .cache_bytes = (uint64_t)(argc > 4 ? atol(argv[4]) : 20480) << 20,
        .n_ctx = 4096,
        .compute_cpus = ccpus,
        .n_compute = nc,
        .io_cpus = icpus,
        .n_io = ni};
    char err[256];
    struct janas_llm_model *m =
        janas_llm_model_load(argv[1], &o, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "%s: %s\n", argv[1], err);
        return 1;
    }
    uint32_t nv = janas_llm_model_n_vocab(m);
    float *logits = malloc(nv * sizeof(float));
    float *want = malloc(nv * sizeof(float));
    size_t agree = 0, compared = 0;
    double worst = 0, mean_sum = 0, t0 = now(), t_half = t0;
    double ph0[JANAS_PH_COUNT] = {0};
    for (size_t p = 0; p < n; p++) {
        if (p == n / 2) {
            t_half = now();
            janas_llm_model_phases(m, ph0);
        }
        if (janas_llm_model_decode(m, tokens[p], (uint32_t)p, logits) != 0) {
            fprintf(stderr, "decode failed at %zu\n", p);
            return 1;
        }
        if (dump)
            fwrite(logits, sizeof(float), nv, dump);
        if (ref && fread(want, sizeof(float), nv, ref) == nv) {
            double mx = 0, sum = 0;
            for (uint32_t i = 0; i < nv; i++) {
                double d = fabs((double)logits[i] - want[i]);
                sum += d;
                if (d > mx)
                    mx = d;
            }
            uint32_t a = argmax(logits, nv), b = argmax(want, nv);
            agree += a == b;
            compared++;
            mean_sum += sum / nv;
            if (mx > worst)
                worst = mx;
            if (p < 8 || a != b)
                printf("pos %3zu: top-1 %6u vs %6u %s, max |diff| %.3f, mean "
                       "%.4f\n",
                       p, a, b, a == b ? "ok  " : "DIFF", mx, sum / nv);
        }
    }
    double dt = now() - t0;
    struct janas_expert_cache_stats st;
    janas_expert_cache_stats(janas_llm_model_cache(m), &st);
    double t2 = now() - t_half, ph_end[JANAS_PH_COUNT];
    janas_llm_model_phases(m, ph_end);
    printf("%zu tokens in %.1f s (%.1f ms/token; second half %.1f ms/token = "
           "%.1f token/s), cold cache: %.1f%% of experts from RAM\n",
           n, dt, dt / n * 1e3, t2 / (n - n / 2) * 1e3, (n - n / 2) / t2,
           100.0 * st.hits / (st.requests ? st.requests : 1));
    int block = 0;
    for (int i = 7; i < argc; i++)
        if (strncmp(argv[i], "block=", 6) == 0)
            block = atoi(argv[i] + 6);
    if (block > 0) {
        /* token by token, keeping every logit row, then the same in blocks */
        size_t nn = n < 256 ? n : 256;
        float *seq = malloc(nn * nv * sizeof(float));
        float *blk = malloc((size_t)block * nv * sizeof(float));
        for (size_t p = 0; p < nn; p++)
            janas_llm_model_decode(m, tokens[p], (uint32_t)p, seq + p * nv);
        double pb0[JANAS_PH_COUNT], pb1[JANAS_PH_COUNT];
        janas_llm_model_phases(m, pb0);
        double tb = now(), maxd = 0;
        size_t same = 0;
        for (size_t p0 = 0; p0 < nn; p0 += (size_t)block) {
            uint32_t b =
                (uint32_t)(nn - p0 < (size_t)block ? nn - p0 : (size_t)block);
            if (janas_llm_model_forward(m, tokens + p0, b, (uint32_t)p0, blk,
                                        1) != 0) {
                fprintf(stderr, "block forward failed at %zu\n", p0);
                return 1;
            }
            for (uint32_t j = 0; j < b; j++) {
                const float *a = seq + (p0 + j) * nv, *c = blk + (size_t)j * nv;
                int eq = 1;
                for (uint32_t i = 0; i < nv; i++) {
                    double d = fabs((double)a[i] - c[i]);
                    if (d > maxd)
                        maxd = d;
                    eq &= a[i] == c[i];
                }
                same += eq;
            }
        }
        double tbs = now() - tb;
        janas_llm_model_phases(m, pb1);
        print_phases("blocks", pb0, pb1, nn);
        printf(
            "blocks of %d: %zu/%zu positions bit-identical to token by token, "
            "max |diff| %g; %.1f ms/token in blocks (%.1f token/s)\n",
            block, same, nn, maxd, tbs / nn * 1e3, nn / tbs);
        free(seq);
        free(blk);
    }
    print_phases("second half", ph0, ph_end, n - n / 2);
    if (compared)
        printf("top-1 agreement %zu/%zu, max |diff| %.3f, mean |diff| %.4f\n",
               agree, compared, worst, mean_sum / compared);
    free(logits);
    free(want);
    if (ref)
        fclose(ref);
    if (dump)
        fclose(dump);
    janas_llm_model_free(m);
    return 0;
}
