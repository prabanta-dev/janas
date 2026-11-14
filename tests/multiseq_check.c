/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * multiseq_check.c - several conversations in one pass
 * (janas_llm_model_forward_parts) against each alone: S streams of N tokens
 * taken from a token file, first each one alone on the model's own sequence,
 * a token a pass, their logits kept; then together, each on a sequence of
 * its own, and every logit must be the same to the bit:
 *
 *   A  all of them writing, a token each a pass;
 *   B  the first reading its tokens in blocks of 16 while the others write,
 *      each starting three passes after the one before (the first passes
 *      have one piece, on a sequence not the model's);
 *   C  as A, the second given 4 wrong tokens at position N/2 and then its
 *      own again from there, as a chain of drafts rejected;
 *   D  the model's own sequence stopped halfway before A-C and taken up
 *      after them: untouched by them;
 *   E  one piece of one token on a sequence not the model's.
 *
 * Usage: multiseq_check <model.jns> <tokens.int32> [S=4] [N=48]
 *                       [cache_MiB] [compute_cpus] [io_cpus]
 * Exit 0 when every logit agrees.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/model.h"

static int parse_cpus(const char *s, int *cpus)
{
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", s);
    int n = 0;
    for (char *t = strtok(buf, ","); t && n < 256; t = strtok(NULL, ","))
        cpus[n++] = atoi(t);
    return n;
}

static uint32_t S, N, nv;
static int32_t *tok; /* [S][N] */
static float *ref;   /* [S][N][nv] */
static float *out;   /* a pass's rows */
static size_t bad, checked;

static const float *want(uint32_t s, uint32_t t)
{
    return ref + ((size_t)s * N + t) * nv;
}

static void check(const char *what, uint32_t s, uint32_t t, const float *got)
{
    checked++;
    if (memcmp(got, want(s, t), nv * sizeof(float))) {
        if (bad < 10)
            printf("%s: stream %u, position %u differs\n", what, s, t);
        bad++;
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: multiseq_check <model.jns> <tokens.int32> "
                        "[S] [N] [cache_MiB] [compute_cpus] [io_cpus]\n");
        return 2;
    }
    S = argc > 3 ? (uint32_t)atoi(argv[3]) : 4;
    N = argc > 4 ? (uint32_t)atoi(argv[4]) : 48;
    if (S < 2 || S > 64 || N < 8 || N > 1024) {
        fprintf(stderr, "multiseq_check: 2 <= S <= 64, 8 <= N <= 1024\n");
        return 2;
    }
    tok = malloc((size_t)S * N * sizeof(int32_t));
    FILE *f = fopen(argv[2], "rb");
    if (!f || !tok ||
        fread(tok, sizeof(int32_t), (size_t)S * N, f) != (size_t)S * N) {
        fprintf(stderr, "%s: fewer than %u tokens\n", argv[2], S * N);
        return 1;
    }
    fclose(f);
    int ccpus[256], icpus[256];
    int nc =
        parse_cpus(argc > 6 ? argv[6] : "0,1,2,3,4,5,6,7,8,9,10,11", ccpus);
    int ni = parse_cpus(argc > 7 ? argv[7] : "12,13,14,15,16,17,18,19", icpus);
    struct janas_llm_options o = {
        .cache_bytes = (uint64_t)(argc > 5 ? atol(argv[5]) : 20480) << 20,
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
    if (!janas_llm_model_parts_ok(m)) {
        printf("multiseq_check: %s does not take several sequences yet\n",
               argv[1]);
        return 1;
    }
    nv = janas_llm_model_n_vocab(m);
    uint32_t blk = janas_llm_model_max_block(m);
    ref = malloc((size_t)S * N * nv * sizeof(float));
    out = malloc((size_t)blk * nv * sizeof(float));
    struct janas_seq **sq = calloc(S, sizeof(*sq));
    struct janas_llm_part *p = calloc(S, sizeof(*p));
    if (!ref || !out || !sq || !p)
        return 1;

    /* each stream alone on the model's own sequence, the first last: its
       second half is kept for D, which takes it up from there */
    for (uint32_t s = S; s-- > 0;)
        for (uint32_t t = 0; t < N && (s || t < N / 2); t++)
            if (janas_llm_model_forward(m, &tok[s * N + t], 1, t,
                                        ref + ((size_t)s * N + t) * nv,
                                        0) != 0) {
                fprintf(stderr, "forward failed\n");
                return 1;
            }
    /* stream 0's second half on a sequence of its own, for its reference
       (A-C need all of it), from its first half again */
    struct janas_seq *s0 = janas_llm_model_seq_new(m);
    for (uint32_t s = 0; s < S; s++)
        sq[s] = janas_llm_model_seq_new(m);
    for (uint32_t s = 0; s < S; s++)
        if (!sq[s] || !s0)
            return 1;
    for (uint32_t t = 0; t < N; t++) {
        struct janas_llm_part one = {
            .seq = s0, .tokens = &tok[t], .n = 1, .pos0 = t};
        if (janas_llm_model_forward_parts(m, &one, 1, out, 0) != 0)
            return 1;
        if (t < N / 2)
            check("E", 0, t, out); /* one piece, a sequence of its own */
        else
            memcpy(ref + (size_t)t * nv, out, nv * sizeof(float));
    }

    /* A: all writing together */
    for (uint32_t t = 0; t < N; t++) {
        for (uint32_t s = 0; s < S; s++)
            p[s] = (struct janas_llm_part){
                .seq = sq[s], .tokens = &tok[s * N + t], .n = 1, .pos0 = t};
        if (janas_llm_model_forward_parts(m, p, S, out, 0) != 0) {
            printf("A: pass %u failed\n", t);
            return 1;
        }
        for (uint32_t s = 0; s < S; s++)
            check("A", s, t, out + (size_t)s * nv);
    }

    /* B: the first in blocks of 16, the others a token a pass, starting
       three passes apart */
    uint32_t *at = calloc(S, sizeof(uint32_t));
    for (uint32_t step = 0;; step++) {
        uint32_t np = 0, rows = 0;
        for (uint32_t s = 0; s < S; s++) {
            if (step < 3 * s || at[s] >= N)
                continue;
            uint32_t k = s == 0 ? (N - at[s] < 16 ? N - at[s] : 16) : 1;
            p[np++] = (struct janas_llm_part){.seq = sq[s],
                                              .tokens = &tok[s * N + at[s]],
                                              .n = k,
                                              .pos0 = at[s]};
            rows += k;
        }
        if (!np)
            break;
        if (janas_llm_model_forward_parts(m, p, np, out, 1) != 0) {
            printf("B: pass %u failed\n", step);
            return 1;
        }
        uint32_t r = 0;
        for (uint32_t i = 0; i < np; i++) {
            uint32_t s = 0;
            while (sq[s] != p[i].seq)
                s++;
            for (uint32_t k = 0; k < p[i].n; k++, r++)
                check("B", s, at[s] + k, out + (size_t)r * nv);
            at[s] += p[i].n;
        }
        (void)rows;
    }

    /* C: as A, the second given wrong tokens at N/2, then its own again */
    int32_t wrong[4] = {tok[0], tok[1], tok[2], tok[3]};
    memset(at, 0, S * sizeof(uint32_t));
    int rejected = 0;
    for (uint32_t step = 0;; step++) {
        uint32_t np = 0;
        for (uint32_t s = 0; s < S; s++) {
            if (at[s] >= N)
                continue;
            int w = s == 1 && at[s] == N / 2 && !rejected;
            p[np++] = (struct janas_llm_part){.seq = sq[s],
                                              .tokens = w ? wrong
                                                          : &tok[s * N + at[s]],
                                              .n = w ? 4 : 1,
                                              .pos0 = at[s]};
        }
        if (!np)
            break;
        if (janas_llm_model_forward_parts(m, p, np, out, 1) != 0) {
            printf("C: pass %u failed\n", step);
            return 1;
        }
        uint32_t r = 0;
        for (uint32_t i = 0; i < np; i++) {
            uint32_t s = 0;
            while (sq[s] != p[i].seq)
                s++;
            if (p[i].tokens == wrong) {
                rejected = 1; /* its rows thrown away; at[1] stays */
                r += 4;
                continue;
            }
            check("C", s, at[s], out + (size_t)r * nv);
            r++;
            at[s]++;
        }
    }

    /* D: the model's own sequence taken up where it stopped */
    for (uint32_t t = N / 2; t < N; t++) {
        if (janas_llm_model_forward(m, &tok[t], 1, t, out, 0) != 0)
            return 1;
        check("D", 0, t, out);
    }

    printf("multiseq_check: %u streams of %u tokens, %zu logit rows compared, "
           "%zu different: %s\n",
           S, N, checked, bad, bad ? "FAILED" : "ok");
    for (uint32_t s = 0; s < S; s++)
        janas_llm_model_seq_free(m, sq[s]);
    janas_llm_model_seq_free(m, s0);
    janas_llm_model_free(m);
    free(at);
    free(sq);
    free(p);
    free(ref);
    free(out);
    free(tok);
    return bad != 0;
}
