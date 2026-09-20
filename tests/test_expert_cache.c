/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_expert_cache.c - the expert cache over a real JNS file.
 *
 * Writes a small JNS file (in $TMPDIR, else /var/tmp: O_DIRECT needs a real
 * file system, not tmpfs) whose every expert slot is filled with its own
 * (layer, expert) identity, then checks on random request sequences that
 * every returned slot holds the expert asked for, and that hits and misses
 * match a reference model of the replacement policy.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/pool.h"
#include "llm/expert_cache.h"
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

enum { N_LAYER = 3, N_EXPERT = 8, K = 3, SLOT = 8192, SLOTS = 5 };

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

static uint64_t rng = 0x9E3779B97F4A7C15ull;

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}

static void fill_slot(uint8_t *p, uint32_t l, uint32_t e)
{
    for (int i = 0; i < SLOT; i++)
        p[i] = (uint8_t)(l * 31 + e * 7 + i);
}

static int write_file(const char *path)
{
    static uint8_t img[12288];
    memset(img, 0, sizeof(img));
    uint64_t layer_len = N_LAYER * sizeof(struct janas_jns_layer);
    uint64_t slot_len = N_LAYER * N_EXPERT * sizeof(uint64_t);
    struct janas_jns_header h = {0};
    memcpy(h.magic, JANAS_JNS_MAGIC, 8);
    h.version = JANAS_JNS_VERSION;
    h.header_bytes = JANAS_JNS_ALIGN;
    strcpy(h.arch, "test");
    h.n_layer = N_LAYER;
    h.n_expert = N_EXPERT;
    h.n_expert_used = K;
    h.d_model = 256;
    h.layer_table_offset = 4096;
    h.expert_table_offset = 4096 + layer_len;
    h.expert_sums_offset = h.expert_table_offset + slot_len;
    h.tensor_dir_offset = h.expert_sums_offset + slot_len;
    h.n_tensor = 0;
    h.metadata_offset = 5120;
    h.metadata_bytes = build_metadata(img + 5120);
    h.metadata_checksum = janas_crc32(0, img + 5120, h.metadata_bytes);
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
    static uint8_t slot[SLOT];
    for (int l = 0; l < N_LAYER; l++) {
        memcpy(img + h.layer_table_offset + l * sizeof(ly), &ly, sizeof(ly));
        for (int e = 0; e < N_EXPERT; e++) {
            /* reversed order on disk: the offset table must be honoured */
            int i = l * N_EXPERT + e;
            uint64_t off = h.experts_offset +
                           (uint64_t)(N_LAYER * N_EXPERT - 1 - i) * SLOT;
            memcpy(img + h.expert_table_offset + 8 * i, &off, 8);
            fill_slot(slot, (uint32_t)l, (uint32_t)e);
            uint64_t crc = janas_crc32(0, slot, SLOT);
            memcpy(img + h.expert_sums_offset + 8 * i, &crc, 8);
        }
    }
    uint64_t s = JANAS_FNV_INIT;
    s = janas_fnv1a(img + h.layer_table_offset, layer_len, s);
    s = janas_fnv1a(img + h.expert_table_offset, slot_len, s);
    s = janas_fnv1a(img + h.expert_sums_offset, slot_len, s);
    h.tables_checksum = s;
    h.header_checksum = janas_fnv1a(&h, sizeof(h), JANAS_FNV_INIT);
    memcpy(img, &h, sizeof(h));

    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    fwrite(img, 1, sizeof(img), f);
    for (int i = N_LAYER * N_EXPERT - 1; i >= 0; i--) {
        fill_slot(slot, (uint32_t)(i / N_EXPERT), (uint32_t)(i % N_EXPERT));
        fwrite(slot, 1, SLOT, f);
    }
    return fclose(f);
}

/* Reference model: LRU list of keys, the current request never evicted. */
struct model {
    int key[SLOTS]; /* index 0 = most recent, -1 = free */
    /* the pass (layers asked in increasing order): the layer asked last,
       the experts asked in this pass and in the one before */
    int cur_layer, pass, last_pass;
};

/* The slot to reuse, never a key of the request: the least recent, or
   with every slot taken and the last pass larger than the cache, the least
   recent of the layer asked latest, going back layer by layer. */
static int model_victim(const struct model *m, const int *keys, int n)
{
    int full = 1;
    for (int s = 0; s < SLOTS; s++)
        full &= m->key[s] != -1;
    for (int d = 0; full && m->last_pass > SLOTS && d < N_LAYER; d++) {
        int l = (m->cur_layer + N_LAYER - d) % N_LAYER;
        for (int v = SLOTS - 1; v >= 0; v--) {
            int in_req = 0;
            for (int q = 0; q < n; q++)
                in_req |= m->key[v] == keys[q];
            if (!in_req && m->key[v] / N_EXPERT == l)
                return v;
        }
    }
    int v = SLOTS - 1;
    for (; v >= 0; v--) {
        int in_req = 0;
        for (int q = 0; q < n; q++)
            in_req |= m->key[v] == keys[q];
        if (!in_req)
            break;
    }
    return v;
}

static int model_fetch(struct model *m, int layer, const uint32_t *ids, int n)
{
    int hits = 0;
    int keys[K];
    if (layer < m->cur_layer) {
        m->last_pass = m->pass;
        m->pass = 0;
    }
    m->cur_layer = layer;
    m->pass += n;
    for (int i = 0; i < n; i++)
        keys[i] = layer * N_EXPERT + (int)ids[i];
    /* hits move to the front first, in request order */
    for (int i = 0; i < n; i++)
        for (int s = 0; s < SLOTS; s++)
            if (m->key[s] == keys[i]) {
                hits++;
                memmove(m->key + 1, m->key, (size_t)s * sizeof(int));
                m->key[0] = keys[i];
                break;
            }
    for (int i = 0; i < n; i++) {
        int present = 0;
        for (int s = 0; s < SLOTS; s++)
            present |= m->key[s] == keys[i];
        if (present)
            continue;
        int v = model_victim(m, keys, n);
        memmove(m->key + 1, m->key, (size_t)v * sizeof(int));
        m->key[0] = keys[i];
    }
    return hits;
}

/* The whole check, once with the reads on the ring and once without it. */
static void run(const char *path, const char *how)
{
    rng = 0x9E3779B97F4A7C15ull;
    struct janas_jns j;
    char err[256];
    CHECK(janas_jns_open(path, &j, err, sizeof(err)) == 0, "open: %s", err);
    struct janas_pool *io = janas_pool_create(3, NULL);
    struct janas_expert_cache *c = janas_expert_cache_create(
        &j, (uint64_t)SLOTS * SLOT, 3, io, err, sizeof(err));
    CHECK(c != NULL, "create: %s", err);
    if (!c) {
        janas_pool_destroy(io);
        janas_jns_close(&j);
        return;
    }
    CHECK(janas_expert_cache_slots(c) == SLOTS, "slots %zu",
          janas_expert_cache_slots(c));

    struct model m = {.cur_layer = 0};
    for (int s = 0; s < SLOTS; s++)
        m.key[s] = -1;
    uint64_t model_hits = 0, requests = 0;
    static uint8_t expect[SLOT];
    for (int iter = 0; iter < 3000; iter++) {
        int layer = (int)(rnd() % N_LAYER);
        uint32_t ids[K];
        for (int i = 0; i < K; i++) { /* distinct, skewed towards low ids */
            uint32_t e;
            int dup;
            do {
                e = rnd() % 2 ? rnd() % 3 : rnd() % N_EXPERT;
                dup = 0;
                for (int q = 0; q < i; q++)
                    dup |= ids[q] == e;
            } while (dup);
            ids[i] = e;
        }
        const uint8_t *slots[K];
        if (iter % 2) {
            CHECK(janas_expert_cache_fetch(c, (uint32_t)layer, ids, K, slots) ==
                      0,
                  "fetch failed");
        } else {
            /* ready slots must be valid before finish, while misses load */
            uint8_t ready[K];
            int q = janas_expert_cache_begin(c, (uint32_t)layer, ids, K, slots,
                                             ready);
            CHECK(q >= 0, "begin failed");
            int n_ready = 0;
            for (int i = 0; i < K; i++)
                if (ready[i]) {
                    n_ready++;
                    fill_slot(expect, (uint32_t)layer, ids[i]);
                    CHECK(memcmp(slots[i], expect, SLOT) == 0,
                          "ready slot with the wrong bytes before finish");
                }
            CHECK(q == K - n_ready, "begin queued %d with %d ready", q,
                  n_ready);
            CHECK(janas_expert_cache_finish(c) == 0, "finish failed");
        }
        for (int i = 0; i < K; i++) {
            fill_slot(expect, (uint32_t)layer, ids[i]);
            CHECK(memcmp(slots[i], expect, SLOT) == 0,
                  "iteration %d: layer %d expert %u has the wrong bytes", iter,
                  layer, ids[i]);
        }
        model_hits += (uint64_t)model_fetch(&m, layer, ids, K);
        requests += K;
    }
    struct janas_expert_cache_stats st;
    janas_expert_cache_stats(c, &st);
    CHECK(st.requests == requests && st.hits == model_hits &&
              st.misses == requests - model_hits,
          "stats: %llu hits of %llu, model %llu", (unsigned long long)st.hits,
          (unsigned long long)st.requests, (unsigned long long)model_hits);
    CHECK(st.bytes_read == st.misses * SLOT, "bytes read %llu",
          (unsigned long long)st.bytes_read);

    /* the cache is filled in the background while requests are served: the
       bytes must still be right, and stopping it must lose nothing */
    {
        struct janas_expert_cache *w = janas_expert_cache_create(
            &j, (uint64_t)SLOTS * SLOT, 3, io, err, sizeof(err));
        CHECK(w != NULL, "create (background filling): %s", err);
        uint32_t *counts = calloc(N_LAYER * N_EXPERT, sizeof(uint32_t));
        if (w && counts) {
            for (size_t k = 0; k < (size_t)N_LAYER * N_EXPERT; k++)
                counts[k] = (uint32_t)(k % 7) + 1;
            CHECK(janas_expert_cache_warm_background(w, counts) == 0,
                  "background filling did not start");
            for (int iter = 0; iter < 400; iter++) {
                int layer = (int)(rnd() % N_LAYER);
                uint32_t ids[K];
                for (int i = 0; i < K; i++) {
                    uint32_t e;
                    int dup;
                    do {
                        e = rnd() % N_EXPERT;
                        dup = 0;
                        for (int q = 0; q < i; q++)
                            dup |= ids[q] == e;
                    } while (dup);
                    ids[i] = e;
                }
                const uint8_t *slots[K];
                CHECK(janas_expert_cache_fetch(w, (uint32_t)layer, ids, K,
                                               slots) == 0,
                      "fetch while filling");
                for (int i = 0; i < K; i++) {
                    fill_slot(expect, (uint32_t)layer, ids[i]);
                    CHECK(memcmp(slots[i], expect, SLOT) == 0,
                          "wrong bytes while the cache fills");
                }
            }
            /* done never passes total; a filling overtaken by the requests
               (they filled the cache first) ends with total = done, which
               may be 0: a front end then stops showing it */
            uint64_t done = 0, total = 0;
            janas_expert_cache_warm_progress(w, &done, &total);
            CHECK(done <= total, "filling progress %llu of %llu",
                  (unsigned long long)done, (unsigned long long)total);
        }
        janas_expert_cache_destroy(w);
        free(counts);
    }

    janas_expert_cache_destroy(c);
    janas_pool_destroy(io);
    janas_jns_close(&j);
    printf("test_expert_cache (%s): %llu requests, %llu hits (model %llu)\n",
           how, (unsigned long long)requests, (unsigned long long)st.hits,
           (unsigned long long)model_hits);
}

int main(void)
{
    /* a test that deadlocks must fail, not hold the suite for ever */
    alarm(600);
    char path[512];
    const char *dir = getenv("TMPDIR");
    snprintf(path, sizeof(path), "%s/janas_test_cache_%d.jns",
             dir && *dir ? dir : "/var/tmp", (int)getpid());
    if (write_file(path) != 0) {
        fprintf(stderr,
                "cannot write %s: set TMPDIR to a writable directory on a "
                "disk (O_DIRECT does not work on tmpfs)\n",
                path);
        return 1;
    }
    /* both ways of reading the misses: io_uring where the kernel has it,
       and the pool of threads, which is what runs where it has not */
    unsetenv("JANAS_URING");
    unsetenv("JANAS_URING_BATCH");
    run(path, "io_uring if available");
    setenv("JANAS_URING_BATCH", "1", 1); /* some batches each way */
    run(path, "both ways");
    unsetenv("JANAS_URING_BATCH");
    setenv("JANAS_URING", "0", 1);
    run(path, "thread pool");
    unlink(path);
    printf("test_expert_cache: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
