/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_attention.c - grouped-query attention against a double-precision
 * softmax over the same eight-bit keys and values, and blocks against
 * token-by-token calls (bit for bit).
 *
 * The engine reads the query as sixteen-bit integers, so the scores rest on
 * an exact integer sum and the only error left is that quantization: a
 * relative 3e-5, which on these outputs shows as about 7e-5 absolute. The
 * eight-bit keys and values the reference shares already carry a relative
 * 4e-3, sixty times more, so the threshold below is set at 1e-4.
 *
 * The checksum printed at the end covers every output of the run: the AVX2
 * and the AVX-VNNI paths must print the same one (JANAS_KERNELS=avx2 asks
 * for plain AVX2), since both sum the same integers.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/pool.h"
#include "llm/attention.h"
#include "llm/quant.h"

static int failures;
static uint64_t checksum = 0xCBF29CE484222325ull;

static void feed(const void *p, size_t n)
{
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        checksum ^= b[i];
        checksum *= 0x100000001B3ull;
    }
}

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

/* Attention of query q (one head) over positions 0 .. n_pos - 1. */
static void reference(const float *q, const int8_t *K, const float *ks,
                      const int8_t *V, const float *vs, uint32_t n_pos,
                      uint32_t hd, double *out)
{
    double *s = malloc(n_pos * sizeof(double)), mx = -INFINITY, sum = 0;
    for (uint32_t t = 0; t < n_pos; t++) {
        double d = 0;
        for (uint32_t i = 0; i < hd; i++)
            d += (double)q[i] * (double)K[(size_t)t * hd + i] * ks[t];
        s[t] = d / sqrt((double)hd);
        if (s[t] > mx)
            mx = s[t];
    }
    for (uint32_t i = 0; i < hd; i++)
        out[i] = 0;
    for (uint32_t t = 0; t < n_pos; t++) {
        double w = exp(s[t] - mx);
        sum += w;
        for (uint32_t i = 0; i < hd; i++)
            out[i] += w * (double)V[(size_t)t * hd + i] * vs[t];
    }
    for (uint32_t i = 0; i < hd; i++)
        out[i] /= sum;
    free(s);
}

static void run(struct janas_pool *pool, uint32_t n_head, uint32_t n_kv,
                uint32_t hd, uint32_t n_ctx, uint32_t pos0, uint32_t n,
                int scores)
{
    struct janas_attn *a = janas_attn_create(n_head, n_kv, hd, n_ctx,
                                             janas_pool_size(pool), scores);
    size_t qd = (size_t)n_head * hd, kv = (size_t)n_kv * n_ctx * hd;
    int8_t *K = malloc(kv), *V = malloc(kv);
    size_t nsc = (size_t)n_kv * n_ctx;
    float *ks = malloc(nsc * sizeof(float)), *vs = malloc(nsc * sizeof(float));
    float *q = malloc(n * qd * sizeof(float));
    float *blk = malloc(n * qd * sizeof(float));
    float *one = malloc(qd * sizeof(float));
    double *ref = malloc(hd * sizeof(double));
    for (size_t i = 0; i < kv; i++) {
        K[i] = (int8_t)(int)(rndf(-127, 127));
        V[i] = (int8_t)(int)(rndf(-127, 127));
    }
    for (size_t i = 0; i < nsc; i++) {
        ks[i] = rndf(0.001f, 0.02f);
        vs[i] = rndf(0.001f, 0.02f);
    }
    for (size_t i = 0; i < n * qd; i++)
        q[i] = rndf(-3, 3);
    janas_attn_run(a, pool, q, K, ks, V, vs, n, pos0, blk);
    feed(blk, n * qd * sizeof(float));
    double worst = 0;
    int same = 1;
    uint32_t group = n_head / n_kv;
    for (uint32_t j = 0; j < n; j++) {
        janas_attn_run(a, pool, q + j * qd, K, ks, V, vs, 1, pos0 + j, one);
        if (memcmp(one, blk + j * qd, qd * sizeof(float)) != 0)
            same = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            size_t base = (size_t)(h / group) * n_ctx * hd;
            size_t sb = (size_t)(h / group) * n_ctx;
            reference(q + j * qd + (size_t)h * hd, K + base, ks + sb, V + base,
                      vs + sb, pos0 + j + 1, hd, ref);
            for (uint32_t i = 0; i < hd; i++) {
                double e = fabs(ref[i] - blk[j * qd + (size_t)h * hd + i]);
                if (e > worst)
                    worst = e;
            }
        }
    }
    CHECK(same,
          "heads %u/%u, pos0 %u, n %u: block differs from token by "
          "token",
          n_head, n_kv, pos0, n);
    /* the exact query keeps the float sum's own error; the sixteen-bit one
       adds its quantization, a relative 3e-5 on the scores */
    int fast = janas_attn_scores_mode(a) == JANAS_ATTN_INT16;
    double limit = fast ? 1e-4 : 1e-5;
    CHECK(worst < limit, "heads %u/%u, pos0 %u, n %u, %s: error %g", n_head,
          n_kv, pos0, n, fast ? "fast" : "exact", worst);
    printf("heads %2u/%u dim %3u, positions %4u..%4u, %-5s: error %.2e, "
           "blocks %s\n",
           n_head, n_kv, hd, pos0, pos0 + n - 1, fast ? "fast" : "exact", worst,
           same ? "bit-identical" : "DIFFER");
    free(K);
    free(V);
    free(ks);
    free(vs);
    free(q);
    free(blk);
    free(one);
    free(ref);
    janas_attn_destroy(a);
}

int main(void)
{
    /* a test that deadlocks must fail, not hold the suite for ever */
    alarm(600);
    struct janas_pool *pool = janas_pool_create(4, NULL);
    for (int i = 0; i < 2; i++) {
        int sc = i ? JANAS_ATTN_INT16 : JANAS_ATTN_FLOAT;
        run(pool, 16, 2, 256, 1024, 0, 1, sc);
        run(pool, 16, 2, 256, 1024, 0, 64, sc);
        run(pool, 16, 2, 256, 1024, 250, 17, sc); /* across a chunk boundary */
        run(pool, 16, 2, 256, 1024, 700, 5, sc);
        run(pool, 32, 4, 128, 1024, 300, 64, sc); /* qwen3moe geometry */
        run(pool, 12, 4, 64, 600, 511, 3, sc);    /* 3 heads per group */
    }
    janas_pool_destroy(pool);
    if (failures) {
        fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    printf("test_attention: ok (%s, checksum %016llx)\n",
           janas_use_vnni() ? "avx-vnni" : "avx2",
           (unsigned long long)checksum);
    return 0;
}
