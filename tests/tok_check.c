/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tok_check.c - checks the tokenizer: encodes a text (special tokens
 * parsed), compares the ids with a reference file of int32 ids when given,
 * and decodes them back, which must give the text again.
 *
 * Usage: tok_check <model.jns> <text> [reference.tokens]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/llm/jns.h"
#include "../src/llm/tokenizer.h"

static char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        b = NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return b;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: tok_check <model.jns> <text> [reference]\n");
        return 2;
    }
    char err[256] = "";
    struct janas_jns j;
    if (janas_jns_open(argv[1], &j, err, sizeof(err)) != 0) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    double t0 = now();
    struct janas_tokenizer *t = janas_tokenizer_create(&j, err, sizeof(err));
    if (!t) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    double t1 = now();
    size_t len;
    char *text = read_file(argv[2], &len);
    if (!text)
        return 1;
    int32_t *ids = malloc((len + 16) * sizeof(int32_t));
    long n = janas_tokenizer_encode(t, text, len, 1, ids, len + 16);
    double t2 = now();
    printf("vocabulary %u tokens, loaded in %.1f ms\n",
           janas_tokenizer_n_vocab(t), 1e3 * (t1 - t0));
    printf("%zu bytes -> %ld tokens in %.2f ms (%.1f MB/s)\n", len, n,
           1e3 * (t2 - t1), (double)len / (t2 - t1) / 1e6);
    int bad = 0;
    if (argc > 3) {
        size_t rl;
        int32_t *ref = (int32_t *)read_file(argv[3], &rl);
        long rn = ref ? (long)(rl / 4) : 0;
        long m = rn < n ? rn : n, i = 0;
        while (i < m && ids[i] == ref[i])
            i++;
        if (i == m && rn == n) {
            printf("reference: %ld tokens, identical\n", rn);
        } else {
            bad = 1;
            printf("reference: %ld tokens, first difference at %ld\n", rn, i);
            for (long k = i > 3 ? i - 3 : 0; k < i + 6; k++) {
                char a[64] = "", b[64] = "";
                size_t la =
                    k < n ? janas_tokenizer_decode(t, ids[k], a, 63) : 0;
                size_t lb =
                    k < rn ? janas_tokenizer_decode(t, ref[k], b, 63) : 0;
                a[la < 63 ? la : 63] = 0;
                b[lb < 63 ? lb : 63] = 0;
                printf("  %6ld  own %6d [%s]  ref %6d [%s]\n", k,
                       k < n ? ids[k] : -1, a, k < rn ? ref[k] : -1, b);
            }
        }
        free(ref);
    }
    size_t w = 0;
    for (long i = 0; i < n; i++) {
        char buf[256];
        size_t l = janas_tokenizer_decode(t, ids[i], buf, sizeof(buf));
        if (w + l > len || memcmp(text + w, buf, l) != 0) {
            printf("round trip: differs at token %ld (byte %zu)\n", i, w);
            bad = 1;
            break;
        }
        w += l;
    }
    if (!bad || w == len)
        printf("round trip: %s\n", w == len ? "identical" : "short");
    free(ids);
    free(text);
    janas_tokenizer_destroy(t);
    janas_jns_close(&j);
    return bad;
}
