/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_jns.c - the JNS parser: a valid image, and mutations of it.
 *
 * The parser reads untrusted files, so it is fuzzed here: random byte flips
 * (mostly stopped by the checksums) and field mutations with the checksums
 * recomputed (which reach the validation of every offset and size). The test
 * passes when no mutation crashes the parser and every accepted image keeps
 * its invariants; run it under the asan and ubsan variants.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "llm/jns.h"

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

enum { N_LAYER = 2, N_EXPERT = 3, SLOT = 8192, META_BYTES = 64 };

/* Metadata region: {"test.block_count": u32 N_LAYER, "test.eps": f32 1e-6}. */
static size_t build_metadata(uint8_t *m)
{
    size_t p = 0;
    uint64_t n = 2, klen;
    uint32_t t, v = N_LAYER;
    float f = 1e-6f;
    memcpy(m + p, &n, 8), p += 8;
    klen = 16, memcpy(m + p, &klen, 8), p += 8;
    memcpy(m + p, "test.block_count", 16), p += 16;
    t = 4, memcpy(m + p, &t, 4), p += 4;
    memcpy(m + p, &v, 4), p += 4;
    klen = 8, memcpy(m + p, &klen, 8), p += 8;
    memcpy(m + p, "test.eps", 8), p += 8;
    t = 6, memcpy(m + p, &t, 4), p += 4;
    memcpy(m + p, &f, 4), p += 4;
    return p;
}

static uint64_t rng = 0x243F6A8885A308D3ull;

static uint64_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}

static void seal(uint8_t *img)
{
    struct janas_jns_header h;
    memcpy(&h, img, sizeof(h));
    uint64_t layer_len = N_LAYER * sizeof(struct janas_jns_layer);
    uint64_t slot_len = N_LAYER * N_EXPERT * sizeof(uint64_t);
    uint64_t dir_len = sizeof(struct janas_jns_tensor);
    /* recompute over the canonical places, even if a mutation moved them */
    uint64_t s = JANAS_FNV_INIT;
    s = janas_fnv1a(img + 4096, layer_len, s);
    s = janas_fnv1a(img + 4096 + layer_len, slot_len, s);
    s = janas_fnv1a(img + 4096 + layer_len + slot_len, slot_len, s);
    s = janas_fnv1a(img + 4096 + layer_len + 2 * slot_len, dir_len, s);
    h.tables_checksum = s;
    /* over the canonical metadata place, like the tables above */
    h.metadata_checksum = janas_crc32(0, img + 4864, META_BYTES);
    h.header_checksum = 0;
    h.header_checksum = janas_fnv1a(&h, sizeof(h), JANAS_FNV_INIT);
    memcpy(img, &h, sizeof(h));
}

/* A valid image of the header and tables; returns the file size. */
static uint64_t build(uint8_t *img)
{
    memset(img, 0, 8192);
    uint64_t layer_len = N_LAYER * sizeof(struct janas_jns_layer);
    uint64_t slot_len = N_LAYER * N_EXPERT * sizeof(uint64_t);
    struct janas_jns_header h = {0};
    memcpy(h.magic, JANAS_JNS_MAGIC, 8);
    h.version = JANAS_JNS_VERSION;
    h.header_bytes = JANAS_JNS_ALIGN;
    strcpy(h.arch, "test");
    h.n_layer = N_LAYER;
    h.n_expert = N_EXPERT;
    h.n_expert_used = 2;
    h.d_model = 256;
    h.layer_table_offset = 4096;
    h.expert_table_offset = 4096 + layer_len;
    h.expert_sums_offset = 4096 + layer_len + slot_len;
    h.tensor_dir_offset = 4096 + layer_len + 2 * slot_len;
    h.n_tensor = 1;
    h.metadata_offset = 4864;
    h.metadata_bytes = build_metadata(img + 4864);
    h.resident_offset = 8192;
    h.resident_bytes = 4096;
    h.experts_offset = 12288;
    h.experts_bytes = (uint64_t)N_LAYER * N_EXPERT * SLOT;
    h.file_bytes = h.experts_offset + h.experts_bytes;

    struct janas_jns_layer ly = {.slot_bytes = SLOT,
                                 .level_bytes = {SLOT, SLOT, SLOT}};
    for (int m = 0; m < 3; m++)
        ly.m[m] = (struct janas_jns_matrix){.type = 12,
                                            .rows = 16,
                                            .cols = 256,
                                            .offset = (uint64_t)m * 2304,
                                            .bytes = 2304};
    for (int l = 0; l < N_LAYER; l++)
        memcpy(img + h.layer_table_offset + l * sizeof(ly), &ly, sizeof(ly));
    for (int i = 0; i < N_LAYER * N_EXPERT; i++) {
        uint64_t off = h.experts_offset + (uint64_t)i * SLOT;
        memcpy(img + h.expert_table_offset + 8 * i, &off, 8);
    }
    struct janas_jns_tensor t = {.type = 0,
                                 .n_dims = 1,
                                 .dims = {4, 1, 1, 1},
                                 .offset = 8192,
                                 .bytes = 16};
    strcpy(t.name, "output_norm.weight");
    memcpy(img + h.tensor_dir_offset, &t, sizeof(t));
    memcpy(img, &h, sizeof(h));
    seal(img);
    return h.file_bytes;
}

static void check_accepted(const struct janas_jns *j)
{
    uint32_t v;
    float f;
    char str[32];
    (void)janas_jns_meta_u32(j, "test.block_count", &v);
    (void)janas_jns_meta_f32(j, "test.eps", &f);
    (void)janas_jns_meta_str(j, "missing", str, sizeof(str));
    for (uint32_t l = 0; l < j->h.n_layer; l++)
        for (uint32_t e = 0; e < j->h.n_expert; e++) {
            uint64_t off = janas_jns_expert_offset(j, l, e);
            CHECK(off >= j->h.experts_offset &&
                      off + j->layers[l].slot_bytes <=
                          j->h.experts_offset + j->h.experts_bytes,
                  "accepted an expert outside the expert region");
        }
}

int main(void)
{
    /* a test that deadlocks must fail, not hold the suite for ever */
    alarm(600);
    static uint8_t base[8192], img[8192];
    uint64_t file_bytes = build(base);
    struct janas_jns j;
    char err[256];

    int rc =
        janas_jns_parse(base, sizeof(base), file_bytes, &j, err, sizeof(err));
    CHECK(rc == 0, "valid image rejected: %s", err);
    if (rc != 0)
        return 1;
    check_accepted(&j);
    uint32_t v = 0;
    float f = 0;
    char str[8];
    CHECK(janas_jns_meta_u32(&j, "test.block_count", &v) == 0 && v == N_LAYER,
          "metadata u32");
    CHECK(janas_jns_meta_f32(&j, "test.eps", &f) == 0 && f == 1e-6f,
          "metadata f32");
    CHECK(janas_jns_meta_u32(&j, "test.eps", &v) != 0 &&
              janas_jns_meta_str(&j, "test.block_count", str, sizeof(str)) !=
                  0 &&
              janas_jns_meta_u32(&j, "missing", &v) != 0,
          "metadata type checks");
    janas_jns_close(&j);

    /* the levels left at zero, as hf2jns_mtp wrote them before issue #6: a
       slot with no planes, its levels all of it */
    {
        static uint8_t z[8192];
        memcpy(z, base, sizeof(z));
        struct janas_jns_layer zl;
        for (int l = 0; l < N_LAYER; l++) {
            memcpy(&zl, z + 4096 + l * sizeof(zl), sizeof(zl));
            memset(zl.level_bytes, 0, sizeof(zl.level_bytes));
            memcpy(z + 4096 + l * sizeof(zl), &zl, sizeof(zl));
        }
        seal(z);
        rc = janas_jns_parse(z, sizeof(z), file_bytes, &j, err, sizeof(err));
        CHECK(rc == 0, "levels at zero rejected: %s", err);
        if (rc == 0) {
            CHECK(j.layers[0].level_bytes[0] == SLOT &&
                      j.layers[0].level_bytes[2] == SLOT &&
                      j.zero_levels == N_LAYER,
                  "levels at zero not read as the whole slot, or not "
                  "counted");
            janas_jns_close(&j);
        }
    }

    /* truncated images and a wrong file size */
    for (size_t n = 0; n < sizeof(base); n += 97)
        if (janas_jns_parse(base, n, file_bytes, &j, err, sizeof(err)) == 0)
            janas_jns_close(&j);
    CHECK(janas_jns_parse(base, sizeof(base), file_bytes - 1, &j, err,
                          sizeof(err)) != 0,
          "wrong file size accepted");

    int accepted_flips = 0, accepted_fields = 0;
    for (int iter = 0; iter < 20000; iter++) {
        memcpy(img, base, sizeof(img));
        int n = 1 + (int)(rnd() % 4);
        for (int k = 0; k < n; k++) {
            /* header (192 bytes), tables (448) or metadata (64): the
               padding between them is ignored by design */
            uint32_t r = rnd() % 3;
            size_t at = r == 0   ? rnd() % 192
                        : r == 1 ? 4096 + rnd() % 448
                                 : 4864 + rnd() % META_BYTES;
            img[at] ^= (uint8_t)(1u << (rnd() % 8));
        }
        if (janas_jns_parse(img, sizeof(img), file_bytes, &j, err,
                            sizeof(err)) == 0) {
            accepted_flips++;
            check_accepted(&j);
            janas_jns_close(&j);
        }
    }
    static const uint64_t special[] = {0,
                                       1,
                                       255,
                                       4095,
                                       4096,
                                       8192,
                                       0x7fffffff,
                                       0xffffffff,
                                       0x100000000ull,
                                       0x7fffffffffffffffull,
                                       0xffffffffffffffffull,
                                       0xfffffffffffff000ull};
    for (int iter = 0; iter < 20000; iter++) {
        memcpy(img, base, sizeof(img));
        int n = 1 + (int)(rnd() % 3);
        for (int k = 0; k < n; k++) {
            /* a 4- or 8-byte field of the header, tables or metadata */
            uint32_t r = rnd() % 3;
            size_t area = r == 0   ? rnd() % 184
                          : r == 1 ? 4096 + rnd() % 504
                                   : 4864 + rnd() % (META_BYTES - 4);
            uint64_t v = rnd() % 3 ? special[rnd() % 12] : rnd();
            memcpy(img + (area & ~(size_t)3), &v, rnd() % 2 ? 4 : 8);
        }
        seal(img);
        if (janas_jns_parse(img, sizeof(img), file_bytes, &j, err,
                            sizeof(err)) == 0) {
            accepted_fields++;
            check_accepted(&j);
            janas_jns_close(&j);
        }
    }
    /* metadata lookups on accepted mutations must stay in bounds (asan) */
    printf("test_jns: %d/20000 byte flips and %d/20000 field mutations "
           "accepted, %s\n",
           accepted_flips, accepted_fields, failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
