/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * llm_slide.c - is the context the same after it has slid?
 *
 * Runs a token sequence twice: once with its first `drop` tokens simply left
 * out, and once whole and then slid by `drop` (janas_llm_model_shift). The
 * two runs then compute the logits of the same last token, at the same
 * position, and the two must agree: what is kept has to behave as if it had
 * always been there. Half precision and the turning back of the keys make
 * small differences; a model with a recurrent state (qwen3next) also keeps
 * what it saw of the dropped part in that state, which is what carries the
 * older text, so there the two runs are meant to differ a little.
 *
 * Usage: llm_slide <model.jns> <tokens.int32> [n] [drop] [keep] [cache_MiB]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "llm/model.h"

/* Feeds tokens in blocks of the largest size a pass takes. */
static int feed(struct janas_llm_model *m, const int32_t *tok, uint32_t n,
                uint32_t pos0, float *logits)
{
    for (uint32_t i = 0; i < n;) {
        uint32_t most = janas_llm_model_max_block(m);
        uint32_t b = n - i < most ? n - i : most;
        if (janas_llm_model_forward(m, tok + i, b, pos0 + i, logits, 0) != 0)
            return -1;
        i += b;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "usage: llm_slide <model.jns> <tokens.int32> [n] [drop] "
                "[keep] [cache_MiB]\n");
        return 2;
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    static int32_t tok[1 << 16];
    size_t have = fread(tok, sizeof(int32_t), sizeof(tok) / sizeof(int32_t), f);
    fclose(f);
    uint32_t n = argc > 3 ? (uint32_t)atoi(argv[3]) : 256;
    uint32_t drop = argc > 4 ? (uint32_t)atoi(argv[4]) : 64;
    uint32_t keep = argc > 5 ? (uint32_t)atoi(argv[5]) : 0;
    if (n > have)
        n = (uint32_t)have;
    if (keep + drop >= n) {
        fprintf(stderr, "keep + drop must leave tokens behind\n");
        return 2;
    }
    int cpus[64];
    for (int i = 0; i < 12; i++)
        cpus[i] = i;
    struct janas_llm_options o = {
        .cache_bytes = (uint64_t)(argc > 6 ? atol(argv[6]) : 20480) << 20,
        .n_ctx = n + 8,
        .compute_cpus = cpus,
        .n_compute = 12,
        .n_io = 4};
    char err[256];
    struct janas_llm_model *m =
        janas_llm_model_load(argv[1], &o, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "%s: %s\n", argv[1], err);
        return 1;
    }
    uint32_t nv = janas_llm_model_n_vocab(m);
    float *a = malloc((size_t)nv * sizeof(float));
    float *b = malloc((size_t)nv * sizeof(float));
    if (!a || !b)
        return 1;

    /* short run: the dropped tokens were never there */
    uint32_t last = n - 1;
    if ((keep && feed(m, tok, keep, 0, a) != 0) ||
        feed(m, tok + keep + drop, last - keep - drop, keep, a) != 0 ||
        janas_llm_model_decode(m, tok[last], last - drop, a) != 0) {
        fprintf(stderr, "the short run failed\n");
        return 1;
    }

    /* long run, then slid */
    if (feed(m, tok, last, 0, b) != 0) {
        fprintf(stderr, "the long run failed\n");
        return 1;
    }
    if (janas_llm_model_shift(m, keep, drop, last) != 0) {
        fprintf(stderr, "the shift was refused\n");
        return 1;
    }
    if (janas_llm_model_decode(m, tok[last], last - drop, b) != 0) {
        fprintf(stderr, "the run after the shift failed\n");
        return 1;
    }

    uint32_t ta = 0, tb = 0;
    double sum = 0, mx = 0;
    for (uint32_t i = 0; i < nv; i++) {
        if (a[i] > a[ta])
            ta = i;
        if (b[i] > b[tb])
            tb = i;
        double d = fabs((double)a[i] - b[i]);
        sum += d;
        if (d > mx)
            mx = d;
    }
    /* and then the text: the same greedy continuation, token for token */
    uint32_t gen = 24, same = 0;
    int32_t *ga = malloc(gen * sizeof(int32_t));
    int32_t *gb = malloc(gen * sizeof(int32_t));
    if (ga && gb) {
        for (int pass = 0; pass < 2; pass++) {
            float *lg = pass ? b : a;
            int32_t *out = pass ? gb : ga;
            /* both runs start from the state each left behind */
            if (pass) {
                if (feed(m, tok, last, 0, b) != 0 ||
                    janas_llm_model_shift(m, keep, drop, last) != 0 ||
                    janas_llm_model_decode(m, tok[last], last - drop, b) != 0)
                    break;
            } else if ((keep && feed(m, tok, keep, 0, a) != 0) ||
                       feed(m, tok + keep + drop, last - keep - drop, keep,
                            a) != 0 ||
                       janas_llm_model_decode(m, tok[last], last - drop, a) !=
                           0)
                break;
            for (uint32_t g = 0; g < gen; g++) {
                uint32_t best = 0;
                for (uint32_t i = 1; i < nv; i++)
                    if (lg[i] > lg[best])
                        best = i;
                out[g] = (int32_t)best;
                if (janas_llm_model_decode(m, out[g], last - drop + 1 + g,
                                           lg) != 0)
                    break;
            }
        }
        while (same < gen && ga[same] == gb[same])
            same++;
    }
    free(ga);
    free(gb);

    printf("%u tokens, %u kept in place, %u forgotten: top-1 %u vs %u %s, "
           "max |diff| %.3f, mean %.4f; the next %u tokens agree for %u\n",
           n, keep, drop, ta, tb, ta == tb ? "ok" : "DIFF", mx, sum / nv, gen,
           same);
    free(a);
    free(b);
    janas_llm_model_free(m);
    return ta == tb ? 0 : 1;
}
