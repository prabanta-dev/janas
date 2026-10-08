/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_attention.c - grouped-query attention against a double-precision
 * softmax over the same eight-bit keys and values, and blocks against
 * token-by-token calls (bit for bit).
 *
 * The exact path sums in floating point: its threshold is 1e-5. The fast
 * one reads the query as sixteen-bit integers, so the scores rest on an
 * exact integer sum and the error there is that quantization, a relative
 * 3e-5; and it weighs the values with sixteen-bit integers too (7 Oct 2026),
 * whose rounding, half a step of the largest weight of a chunk, adds up over
 * its positions: with the random scores here, whose weights crowd on a few
 * positions, the worst output is 2-6e-4 off. The eight-bit keys and values
 * the reference shares already carry a relative 4e-3, so the threshold of
 * the fast path is 1e-3; on Qwen3-30B-A3B against llama.cpp (400 tokens of
 * C) the fast path gave a mean logit difference of 0.2865 against 0.2893
 * with float weights and 0.2871 exact.
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
    /* the exact path keeps the float sum's own error; the fast one adds the
       sixteen-bit query's and weights' quantization (see the top) */
    int fast = janas_attn_scores_mode(a) == JANAS_ATTN_INT16;
    double limit = fast ? 1e-3 : 1e-5;
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

/*
 * Gemma 4's settings: keys and values of sixteen bits, a sliding window and
 * a score scale of 1, against a double-precision reference that sees the
 * same positions. Not fed to the checksum, which stays that of the
 * eight-bit runs.
 */
static void run16(struct janas_pool *pool, uint32_t n_head, uint32_t n_kv,
                  uint32_t hd, uint32_t n_ctx, uint32_t pos0, uint32_t n,
                  uint32_t window)
{
    struct janas_attn *a =
        janas_attn_create(n_head, n_kv, hd, n_ctx, janas_pool_size(pool),
                          JANAS_ATTN_INT16); /* the cache forces FLOAT */
    janas_attn_set_kv16(a, 1);
    janas_attn_set_window(a, window);
    janas_attn_set_scale(a, 0.02f);
    size_t qd = (size_t)n_head * hd, kv = (size_t)n_kv * n_ctx * hd;
    int16_t *K = malloc(kv * 2), *V = malloc(kv * 2);
    size_t nsc = (size_t)n_kv * n_ctx;
    float *ks = malloc(nsc * sizeof(float)), *vs = malloc(nsc * sizeof(float));
    float *q = malloc(n * qd * sizeof(float));
    float *blk = malloc(n * qd * sizeof(float));
    float *one = malloc(qd * sizeof(float));
    for (size_t i = 0; i < kv; i++) {
        K[i] = (int16_t)(int)(rndf(-32767, 32767));
        V[i] = (int16_t)(int)(rndf(-32767, 32767));
    }
    for (size_t i = 0; i < nsc; i++) {
        ks[i] = rndf(0.001f, 0.02f) / 258.0f;
        vs[i] = rndf(0.001f, 0.02f) / 258.0f;
    }
    for (size_t i = 0; i < n * qd; i++)
        q[i] = rndf(-3, 3);
    janas_attn_run(a, pool, q, (const int8_t *)K, ks, (const int8_t *)V, vs, n,
                   pos0, blk);
    double worst = 0;
    int same = 1;
    uint32_t group = n_head / n_kv;
    for (uint32_t j = 0; j < n; j++) {
        janas_attn_run(a, pool, q + j * qd, (const int8_t *)K, ks,
                       (const int8_t *)V, vs, 1, pos0 + j, one);
        if (memcmp(one, blk + j * qd, qd * sizeof(float)) != 0)
            same = 0;
        uint32_t p = pos0 + j,
                 lo = window && p + 1 > window ? p + 1 - window : 0;
        for (uint32_t h = 0; h < n_head; h++) {
            size_t base = (size_t)(h / group) * n_ctx * hd;
            size_t sb = (size_t)(h / group) * n_ctx;
            const float *qh = q + j * qd + (size_t)h * hd;
            double mx = -INFINITY, sum = 0, *sc = malloc((p + 1) * 8);
            double out[512] = {0};
            for (uint32_t t = lo; t <= p; t++) {
                double d = 0;
                for (uint32_t i = 0; i < hd; i++)
                    d += (double)qh[i] * K[base + (size_t)t * hd + i] *
                         ks[sb + t];
                sc[t] = d * 0.02;
                mx = sc[t] > mx ? sc[t] : mx;
            }
            for (uint32_t t = lo; t <= p; t++) {
                double w = exp(sc[t] - mx);
                sum += w;
                for (uint32_t i = 0; i < hd; i++)
                    out[i] += w * V[base + (size_t)t * hd + i] * vs[sb + t];
            }
            for (uint32_t i = 0; i < hd; i++) {
                double e =
                    fabs(out[i] / sum - blk[j * qd + (size_t)h * hd + i]);
                worst = e > worst ? e : worst;
            }
            free(sc);
        }
    }
    CHECK(same, "16-bit, window %u, pos0 %u, n %u: block differs", window, pos0,
          n);
    CHECK(janas_attn_scores_mode(a) == JANAS_ATTN_FLOAT,
          "16-bit keys with integer scores");
    CHECK(worst < 1e-5, "16-bit, window %u, pos0 %u, n %u: error %g", window,
          pos0, n, worst);
    printf("heads %2u/%u dim %3u, positions %4u..%4u, 16-bit, window %4u: "
           "error %.2e, blocks %s\n",
           n_head, n_kv, hd, pos0, pos0 + n - 1, window, worst,
           same ? "bit-identical" : "DIFFER");
    free(K);
    free(V);
    free(ks);
    free(vs);
    free(q);
    free(blk);
    free(one);
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
    run16(pool, 16, 8, 256, 1024, 0, 64, 0);
    run16(pool, 16, 8, 256, 2048, 700, 64, 256); /* a window of 256 */
    run16(pool, 16, 1, 512, 1024, 250, 17, 0);   /* Gemma 4's full layers */
    run16(pool, 16, 8, 256, 2048, 1030, 5, 1024);
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
