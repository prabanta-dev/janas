/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * llm_session.c - checks sessions on a real model:
 *   1. turns: A, 12 generated, B, 24 generated, token by token, gives the
 *      same 24 as a fresh session given A + the 12 + B at once;
 *   2. the same with speculation on, which leaves generated tokens unread
 *      when B is appended (they must be dropped);
 *   3. sampling (temperature, top-p, seed): the same tokens with and
 *      without speculation.
 *
 * Usage: llm_session <model.jns> <tokens.int32> [mtp=<mtp.jns>]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/llm/generate.h"
#include "../src/llm/model.h"

#define NA 120
#define NB 40
#define G1 12
#define G2 24

static int run_turns(struct janas_llm_model *m,
                     const struct janas_gen_options *o, const int32_t *text,
                     int32_t *g1, int32_t *g2)
{
    struct janas_llm_session *s = janas_llm_session_create(m, o);
    if (!s || janas_llm_session_append(s, text, NA) != 0)
        return -1;
    for (int i = 0; i < G1; i++)
        if (janas_llm_session_next(s, &g1[i]) != 0)
            return -1;
    if (janas_llm_session_append(s, text + NA, NB) != 0)
        return -1;
    for (int i = 0; i < G2; i++)
        if (janas_llm_session_next(s, &g2[i]) != 0)
            return -1;
    int ok = janas_llm_session_length(s) == NA + G1 + NB + G2;
    janas_llm_session_destroy(s);
    return ok ? 0 : -1;
}

static int run_once(struct janas_llm_model *m,
                    const struct janas_gen_options *o, const int32_t *seq,
                    uint32_t n, int32_t *g, int count)
{
    struct janas_llm_session *s = janas_llm_session_create(m, o);
    if (!s || janas_llm_session_append(s, seq, n) != 0)
        return -1;
    for (int i = 0; i < count; i++)
        if (janas_llm_session_next(s, &g[i]) != 0)
            return -1;
    janas_llm_session_destroy(s);
    return 0;
}

static int same(const char *what, const int32_t *a, const int32_t *b, int n)
{
    int ok = memcmp(a, b, (size_t)n * sizeof(int32_t)) == 0;
    printf("%-44s %s\n", what, ok ? "identical" : "DIFFERENT");
    return ok;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: llm_session <model.jns> <tokens.int32> "
                        "[mtp=<mtp.jns>]\n");
        return 2;
    }
    int32_t text[NA + NB];
    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(text, sizeof(int32_t), NA + NB, f) != NA + NB)
        return 2;
    fclose(f);
    const char *mtp_path = NULL;
    if (argc > 3 && strncmp(argv[3], "mtp=", 4) == 0)
        mtp_path = argv[3] + 4;
    int ccpus[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int icpus[8] = {12, 13, 14, 15, 16, 17, 18, 19};
    struct janas_llm_options mo = {.cache_bytes = (uint64_t)20480 << 20,
                                   .n_ctx = 1024,
                                   .compute_cpus = ccpus,
                                   .n_compute = 12,
                                   .io_cpus = icpus,
                                   .n_io = 8,
                                   .mtp_path = mtp_path};
    char err[256];
    struct janas_llm_model *m =
        janas_llm_model_load(argv[1], &mo, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "%s: %s\n", argv[1], err);
        return 1;
    }
    struct janas_gen_options greedy = {.spec_k = 1, .eos = -1};
    struct janas_gen_options spec = {.spec_mode = mtp_path ? JANAS_SPEC_MTP
                                                           : JANAS_SPEC_LOOKUP,
                                     .spec_k = 4,
                                     .eos = -1};
    int32_t g1[G1], g2[G2], h1[G1], h2[G2], once[G2];
    int32_t seq[NA + G1 + NB];
    int ok = 1;

    if (run_turns(m, &greedy, text, g1, g2) != 0)
        return 1;
    memcpy(seq, text, NA * sizeof(int32_t));
    memcpy(seq + NA, g1, G1 * sizeof(int32_t));
    memcpy(seq + NA + G1, text + NA, NB * sizeof(int32_t));
    if (run_once(m, &greedy, seq, NA + G1 + NB, once, G2) != 0)
        return 1;
    ok &= same("greedy: turns vs all at once", g2, once, G2);

    if (run_turns(m, &spec, text, h1, h2) != 0)
        return 1;
    ok &= same("greedy + speculation: turn 1", g1, h1, G1);
    ok &= same("greedy + speculation: turn 2", g2, h2, G2);

    struct janas_gen_options sa = greedy, sb = spec;
    sa.temperature = sb.temperature = 1.0f;
    sa.top_p = sb.top_p = 0.95f;
    sa.seed = sb.seed = 42;
    if (run_turns(m, &sa, text, g1, g2) != 0 ||
        run_turns(m, &sb, text, h1, h2) != 0)
        return 1;
    ok &= same("sampling, speculation off vs on: turn 1", g1, h1, G1);
    ok &= same("sampling, speculation off vs on: turn 2", g2, h2, G2);
    int differs = memcmp(g2, once, sizeof(once)) != 0;
    printf("%-44s %s\n", "sampling differs from greedy",
           differs ? "yes" : "no (suspicious)");

    janas_llm_model_free(m);
    printf("llm_session: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
