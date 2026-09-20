/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_gguf.c - the reader of GGUF heads (src/llm/gguf.c).
 *
 * A GGUF file comes from anywhere, so the reader is fuzzed: a valid head
 * built here, then cut short and mutated byte by byte. Nothing may crash
 * or leak (run it under asan and ubsan), a head cut short asks for more
 * rather than failing, and whatever is accepted keeps its invariants: every
 * tensor inside the file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/gguf.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static uint64_t rng = 0x2545F4914F6CDD1Dull;
static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

struct w {
    uint8_t p[4096];
    size_t n;
};

static void put(struct w *w, const void *p, size_t n)
{
    memcpy(w->p + w->n, p, n);
    w->n += n;
}
static void put32(struct w *w, uint32_t v)
{
    put(w, &v, 4);
}
static void put64(struct w *w, uint64_t v)
{
    put(w, &v, 8);
}
static void puts_(struct w *w, const char *s)
{
    put64(w, strlen(s));
    put(w, s, strlen(s));
}

/* A head: six pairs (a string, u32 alignment, u32, i32, an array of
   strings, an array of f32) and two tensors, F32 [4] and Q4_K [256 x 2];
   its data after it, aligned to 64. Returns the file size. */
static uint64_t build(struct w *w)
{
    w->n = 0;
    put(w, "GGUF", 4);
    put32(w, 3);
    put64(w, 2); /* tensors */
    put64(w, 6); /* pairs */
    puts_(w, "general.architecture");
    put32(w, 8);
    puts_(w, "test");
    puts_(w, "general.alignment");
    put32(w, 4);
    put32(w, 64);
    puts_(w, "test.block_count");
    put32(w, 4);
    put32(w, 3);
    puts_(w, "test.signed");
    put32(w, 5);
    put32(w, (uint32_t)-5);
    puts_(w, "tokenizer.tokens");
    put32(w, 9);
    put32(w, 8);
    put64(w, 3);
    puts_(w, "a");
    puts_(w, "bb");
    puts_(w, "ccc");
    puts_(w, "test.floats");
    put32(w, 9);
    put32(w, 6);
    put64(w, 2);
    put32(w, 0);
    put32(w, 0);
    puts_(w, "output_norm.weight");
    put32(w, 1);
    put64(w, 4);
    put32(w, 0); /* F32 */
    put64(w, 0);
    puts_(w, "blk.0.ffn_down.weight");
    put32(w, 2);
    put64(w, 256);
    put64(w, 2);
    put32(w, 12); /* Q4_K: 144 bytes a row */
    put64(w, 64);
    uint64_t data = (w->n + 63) / 64 * 64;
    return data + 64 + 288;
}

static void check_accepted(const struct janas_gguf *g)
{
    for (uint64_t i = 0; i < g->n_tensors; i++)
        CHECK(g->data_start + g->t[i].offset + g->t[i].bytes <= g->file_bytes,
              "a tensor accepted past the end of the file");
    uint64_t v;
    struct janas_gguf_str s;
    (void)janas_gguf_uint(g, "test.block_count", &v);
    (void)janas_gguf_string(g, "general.architecture", &s);
    (void)janas_gguf_tensor(g, "output_norm.weight");
}

int main(void)
{
    static struct w base;
    uint64_t file = build(&base);
    char err[160];
    struct janas_gguf g;

    uint8_t *h = malloc(base.n);
    memcpy(h, base.p, base.n);
    int rc = janas_gguf_parse(h, base.n, file, &g, err, sizeof(err));
    CHECK(rc == 0, "valid head refused: %s", err);
    if (rc == 0) {
        uint64_t v = 0;
        struct janas_gguf_str s = {0};
        const struct janas_gguf_tensor *t =
            janas_gguf_tensor(&g, "blk.0.ffn_down.weight");
        CHECK(janas_gguf_uint(&g, "test.block_count", &v) == 0 && v == 3,
              "u32 value");
        CHECK(janas_gguf_uint(&g, "test.signed", &v) != 0,
              "a negative value read as a count");
        CHECK(janas_gguf_string(&g, "general.architecture", &s) == 0 &&
                  s.n == 4 && memcmp(s.p, "test", 4) == 0,
              "string value");
        CHECK(t && t->bytes == 288 && t->dims[0] == 256 && t->dims[1] == 2 &&
                  t->dims[2] == 1,
              "tensor geometry");
        CHECK(g.data_start % 64 == 0 && g.data_start >= base.n,
              "data start not aligned");
        CHECK(g.kv_end > g.kv_start && g.n_kv == 6, "pairs");
        janas_gguf_free(&g);
    }

    /* cut short: more is asked for while the file has more, never a crash */
    for (size_t n = 0; n < base.n; n++) {
        h = malloc(base.n);
        memcpy(h, base.p, n);
        rc = janas_gguf_parse(h, n, file, &g, err, sizeof(err));
        if (rc == 1)
            free(h); /* still ours */
        else if (rc == 0)
            janas_gguf_free(&g);
        CHECK(rc != 0 || n == base.n, "a head cut at %zu accepted", n);
    }
    /* the tensor's data past the end of the file */
    h = malloc(base.n);
    memcpy(h, base.p, base.n);
    CHECK(janas_gguf_parse(h, base.n, file - 1, &g, err, sizeof(err)) == -1,
          "data past the end accepted");

    /* mutations: bytes changed, and the counts and lengths made extreme */
    int accepted = 0;
    for (int it = 0; it < 20000; it++) {
        h = malloc(base.n);
        memcpy(h, base.p, base.n);
        for (int k = 1 + (int)(rnd() % 4); k > 0; k--) {
            size_t at = rnd() % base.n;
            if (rnd() % 4 == 0 && at + 8 <= base.n) {
                uint64_t big = rnd() % 2 ? ~0ull : (uint64_t)rnd() << 20;
                memcpy(h + at, &big, rnd() % 2 ? 8 : 4);
            } else {
                h[at] ^= (uint8_t)(1u << (rnd() % 8));
            }
        }
        rc = janas_gguf_parse(h, base.n, file, &g, err, sizeof(err));
        if (rc == 1)
            free(h);
        else if (rc == 0) {
            accepted++;
            check_accepted(&g);
            janas_gguf_free(&g);
        }
    }
    printf("test_gguf: %d of 20000 mutated heads accepted, %s\n", accepted,
           failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
