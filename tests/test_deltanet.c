/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_deltanet.c - the Gated DeltaNet step against the recurrence written
 * out in double precision on the untransposed state S[i][j].
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "llm/deltanet.h"

static int failures;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            failures++;                                                        \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fputc('\n', stderr);                                               \
        }                                                                      \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static float rndf(float lo, float hi)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return lo + (hi - lo) * (float)(rng_state >> 40) / (float)(1u << 24);
}

static void unit(float *x, size_t n)
{
    double ss = 0;
    for (size_t i = 0; i < n; i++)
        ss += (double)x[i] * x[i];
    for (size_t i = 0; i < n; i++)
        x[i] = (float)(x[i] / sqrt(ss));
}

typedef void (*step_fn)(float *, const float *, const float *, const float *,
                        float, float, float, float *, size_t, size_t);

/* Runs `steps` tokens through fn and through the double reference. */
static void run(const char *name, step_fn fn, size_t dim, int steps)
{
    float *M = calloc(dim * dim, sizeof(float));
    double *S = calloc(dim * dim, sizeof(double));
    float *q = malloc(dim * sizeof(float)), *k = malloc(dim * sizeof(float));
    float *v = malloc(dim * sizeof(float)), *o = malloc(dim * sizeof(float));
    double worst = 0, scale = 1.0 / sqrt((double)dim);
    for (int t = 0; t < steps; t++) {
        for (size_t i = 0; i < dim; i++) {
            q[i] = rndf(-1, 1);
            k[i] = rndf(-1, 1);
            v[i] = rndf(-2, 2);
        }
        unit(q, dim);
        unit(k, dim);
        float decay = expf(rndf(-0.5f, 0.0f)), beta = rndf(0.05f, 0.95f);
        /* the rows in two ranges, as threads would split a head */
        size_t half = dim / 2;
        fn(M, q, k, v, decay, beta, (float)scale, o, dim, half);
        fn(M + half * dim, q, k, v + half, decay, beta, (float)scale, o + half,
           dim, dim - half);
        for (size_t i = 0; i < dim * dim; i++)
            S[i] *= decay;
        for (size_t j = 0; j < dim; j++) {
            double sk = 0;
            for (size_t i = 0; i < dim; i++)
                sk += S[i * dim + j] * k[i];
            double d = (v[j] - sk) * beta;
            for (size_t i = 0; i < dim; i++)
                S[i * dim + j] += k[i] * d;
        }
        for (size_t j = 0; j < dim; j++) {
            double oj = 0;
            for (size_t i = 0; i < dim; i++)
                oj += S[i * dim + j] * q[i];
            oj *= scale;
            double e = fabs(oj - o[j]) / (1.0 + fabs(oj));
            if (e > worst)
                worst = e;
        }
    }
    double sworst = 0;
    for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++) {
            double e = fabs(S[i * dim + j] - M[j * dim + i]);
            if (e > sworst)
                sworst = e;
        }
    CHECK(worst < 1e-4 && sworst < 1e-4,
          "%s dim %zu: output error %g, state error %g", name, dim, worst,
          sworst);
    printf("%s dim %3zu, %d steps: output error %.2e, state error %.2e\n", name,
           dim, steps, worst, sworst);
    free(M);
    free(S);
    free(q);
    free(k);
    free(v);
    free(o);
}

/* The half-precision state: the same recurrence, the state rounded to f16
   after every step, against the double reference. */
static void run_h(size_t dim, int steps)
{
    uint16_t *M = calloc(dim * dim, sizeof(uint16_t));
    double *S = calloc(dim * dim, sizeof(double));
    float *q = malloc(dim * sizeof(float)), *k = malloc(dim * sizeof(float));
    float *v = malloc(dim * sizeof(float)), *o = malloc(dim * sizeof(float));
    double worst = 0, scale = 1.0 / sqrt((double)dim), omax = 0;
    for (int t = 0; t < steps; t++) {
        for (size_t i = 0; i < dim; i++) {
            q[i] = rndf(-1, 1);
            k[i] = rndf(-1, 1);
            v[i] = rndf(-2, 2);
        }
        unit(q, dim);
        unit(k, dim);
        float decay = expf(rndf(-0.5f, 0.0f)), beta = rndf(0.05f, 0.95f);
        janas_gdn_step_h(M, q, k, v, decay, beta, (float)scale, o, dim, dim);
        for (size_t i = 0; i < dim * dim; i++)
            S[i] *= decay;
        for (size_t j = 0; j < dim; j++) {
            double sk = 0;
            for (size_t i = 0; i < dim; i++)
                sk += S[i * dim + j] * k[i];
            double d = (v[j] - sk) * beta;
            for (size_t i = 0; i < dim; i++)
                S[i * dim + j] += k[i] * d;
        }
        for (size_t j = 0; j < dim; j++) {
            double oj = 0;
            for (size_t i = 0; i < dim; i++)
                oj += S[i * dim + j] * q[i];
            oj *= scale;
            if (fabs(oj) > omax)
                omax = fabs(oj);
            double e = fabs(oj - o[j]);
            if (e > worst)
                worst = e;
        }
    }
    /* relative to the output scale: f16 keeps ~3 decimal digits */
    CHECK(worst / omax < 5e-3, "half state dim %zu: error %g of %g", dim, worst,
          omax);
    printf("half state dim %3zu, %d steps: output error %.2e (of %.2f)\n", dim,
           steps, worst, omax);
    free(M);
    free(S);
    free(q);
    free(k);
    free(v);
    free(o);
}

int main(void)
{
    /* a test that deadlocks must fail, not hold the suite for ever */
    alarm(600);
    size_t dims[] = {8, 24, 128};
    for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++) {
        run("ref ", janas_gdn_step_ref, dims[i], 64);
        run("fast", janas_gdn_step, dims[i], 64);
    }
    run_h(128, 256);
    printf("kernel: %s\n", janas_gdn_kernel_name());
    if (failures) {
        fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    printf("test_deltanet: ok\n");
    return 0;
}
