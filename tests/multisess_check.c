/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * multisess_check.c - several generating sessions moved on together
 * (janas_llm_sessions_step) against each alone: S sessions, each a prompt
 * of its own length taken from a token file, each writing N tokens - first
 * one after the other on the model's own sequence (janas_llm_session_next),
 * then together, each on a sequence of its own, starting a few steps apart
 * so that prompts are read while others write. The tokens must be the
 * same: greedy, then drawn at temperature 0.8 with a seed of each session's
 * own; the sessions alone without drafts, and once more with the drafts
 * copied from the context (lookup), which may not change a greedy text.
 *
 * Usage: multisess_check <model.jns> <tokens.int32> [S=4] [N=32]
 *                        [cache_MiB] [compute_cpus] [io_cpus]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/generate.h"
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

static uint32_t S, N;
static int32_t *tok;
static size_t n_tok;

/* session i's prompt: its own stretch of the file, 17 + 13 i tokens */
static const int32_t *prompt(uint32_t i, uint32_t *len)
{
    *len = 17 + 13 * i;
    return tok + (size_t)(i * 61) % (n_tok - *len);
}

static void opts(struct janas_gen_options *o, int sampled, int lookup,
                 uint32_t i)
{
    memset(o, 0, sizeof(*o));
    o->eos = -1;
    o->spec_mode = lookup ? JANAS_SPEC_LOOKUP : JANAS_SPEC_OFF;
    o->spec_k = 4;
    if (sampled) {
        o->temperature = 0.8f;
        o->top_p = 0.95f;
        o->seed = 1000 + i;
    }
}

/* each alone, one after the other, on the model's own sequence */
static int alone(struct janas_llm_model *m, int sampled, int lookup,
                 int32_t *out)
{
    for (uint32_t i = 0; i < S; i++) {
        struct janas_gen_options o;
        opts(&o, sampled, lookup, i);
        struct janas_llm_session *s = janas_llm_session_create(m, &o);
        uint32_t len;
        const int32_t *pr = prompt(i, &len);
        if (!s || janas_llm_session_append(s, pr, len) != 0)
            return -1;
        for (uint32_t t = 0; t < N; t++)
            if (janas_llm_session_next(s, &out[i * N + t]) != 0)
                return -1;
        janas_llm_session_destroy(s);
    }
    return 0;
}

/* together, each on a sequence of its own, session i joining at step 2 i */
static int together(struct janas_llm_model *m, int sampled, int32_t *out)
{
    struct janas_llm_session **ss = calloc(S, sizeof(*ss));
    struct janas_seq **sq = calloc(S, sizeof(*sq));
    uint32_t *got = calloc(S, sizeof(uint32_t));
    struct janas_llm_session **act = calloc(S, sizeof(*act));
    int32_t *tk = calloc(S, sizeof(int32_t));
    int *st = calloc(S, sizeof(int)), *idx = calloc(S, sizeof(int));
    if (!ss || !sq || !got || !act || !tk || !st || !idx)
        return -1;
    for (uint32_t i = 0; i < S; i++) {
        struct janas_gen_options o;
        opts(&o, sampled, 0, i);
        ss[i] = janas_llm_session_create(m, &o);
        sq[i] = janas_llm_model_seq_new(m);
        uint32_t len;
        const int32_t *pr = prompt(i, &len);
        if (!ss[i] || !sq[i])
            return -1;
        janas_llm_session_set_seq(ss[i], sq[i]);
        if (janas_llm_session_append(ss[i], pr, len) != 0)
            return -1;
    }
    int rc = 0;
    for (uint32_t step = 0; rc == 0; step++) {
        int na = 0;
        for (uint32_t i = 0; i < S; i++)
            if (step >= 2 * i && got[i] < N) {
                act[na] = ss[i];
                idx[na++] = (int)i;
            }
        if (!na)
            break;
        if (janas_llm_sessions_step(act, na, tk, st) != 0) {
            printf("together: step %u failed\n", step);
            rc = -1;
            break;
        }
        for (int k = 0; k < na; k++) {
            uint32_t i = (uint32_t)idx[k];
            if (st[k] == 0)
                out[i * N + got[i]++] = tk[k];
            else if (st[k] != 2) {
                printf("together: session %u status %d\n", i, st[k]);
                rc = -1;
            }
        }
    }
    for (uint32_t i = 0; i < S; i++) {
        janas_llm_session_destroy(ss[i]);
        janas_llm_model_seq_free(m, sq[i]);
    }
    free(ss);
    free(sq);
    free(got);
    free(act);
    free(tk);
    free(st);
    free(idx);
    return rc;
}

static size_t compare(const char *what, const int32_t *a, const int32_t *b)
{
    size_t bad = 0;
    for (uint32_t i = 0; i < S; i++)
        for (uint32_t t = 0; t < N; t++)
            if (a[i * N + t] != b[i * N + t]) {
                printf("%s: session %u, token %u: %d alone, %d together\n",
                       what, i, t, a[i * N + t], b[i * N + t]);
                bad++;
                break;
            }
    printf("%s: %s\n", what, bad ? "DIFFERENT" : "the same tokens");
    return bad;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: multisess_check <model.jns> <tokens.int32> "
                        "[S] [N] [cache_MiB] [compute_cpus] [io_cpus]\n");
        return 2;
    }
    S = argc > 3 ? (uint32_t)atoi(argv[3]) : 4;
    N = argc > 4 ? (uint32_t)atoi(argv[4]) : 32;
    if (S < 2 || S > 32 || N < 1 || N > 512)
        return 2;
    FILE *f = fopen(argv[2], "rb");
    tok = malloc(65536 * sizeof(int32_t));
    n_tok = f && tok ? fread(tok, sizeof(int32_t), 65536, f) : 0;
    if (f)
        fclose(f);
    if (n_tok < 17 + 13 * S + 64) {
        fprintf(stderr, "%s: too few tokens\n", argv[2]);
        return 1;
    }
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
    int32_t *a = malloc((size_t)S * N * sizeof(int32_t)),
            *b = malloc((size_t)S * N * sizeof(int32_t)),
            *c = malloc((size_t)S * N * sizeof(int32_t));
    size_t bad = 0;
    for (int sampled = 0; sampled < 2; sampled++) {
        const char *what = sampled ? "sampled" : "greedy";
        if (alone(m, sampled, 0, a) != 0 || together(m, sampled, b) != 0) {
            printf("%s: failed\n", what);
            return 1;
        }
        bad += compare(what, a, b);
    }
    /* drafts copied from the context may not change a greedy text */
    if (alone(m, 0, 1, c) != 0 || together(m, 0, b) != 0)
        return 1;
    bad += compare("greedy, alone with lookup drafts", c, b);
    printf("multisess_check: %u sessions of %u tokens: %s\n", S, N,
           bad ? "FAILED" : "ok");
    janas_llm_model_free(m);
    free(a);
    free(b);
    free(c);
    free(tok);
    return bad != 0;
}
