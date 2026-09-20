/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * expert_cache.h - a RAM cache of routed experts over a JNS file.
 *
 * A fixed number of slots, each large enough for the largest expert of the
 * model, lives in one 4 KiB-aligned arena. Replacement is LRU; the cache can
 * be warmed with the experts most used by a prompt (the prefill predicts the
 * decode working set: bozza Fase 0). The misses of a request are read in
 * parallel with O_DIRECT by a pool of I/O threads, and the experts of the
 * request itself are never evicted to make room for each other.
 */
#ifndef JANAS_LLM_EXPERT_CACHE_H
#define JANAS_LLM_EXPERT_CACHE_H

#include <stddef.h>
#include <stdint.h>

#include "common/pool.h"
#include "jns.h"

struct janas_expert_cache_stats {
    uint64_t requests; /* experts asked for */
    uint64_t hits;
    uint64_t misses;
    uint64_t bytes_read; /* from the file, O_DIRECT, for the requests */
    uint64_t warm_bytes; /* and for the filling in the background */
    double io_seconds;   /* wall time of the reads (in the background) */
    double wait_seconds; /* time the caller actually waited for them */
};

struct janas_expert_cache;

/*
 * Creates a cache of budget_bytes (rounded down to whole slots) over an open
 * JNS file. io is the pool that performs the reads (its threads should not be
 * the compute threads). level, 1 to 3, is how much of each expert slot is
 * read: with a file whose down matrix is in bit planes, level 1 reads the
 * base plane alone (two bits per weight), 2 adds the second plane (four) and
 * 3 the whole slot (six, the Q6_K weights themselves). A smaller level means
 * smaller slots, so the same budget holds more experts. Returns NULL on
 * failure, with a message in err.
 */
struct janas_expert_cache *janas_expert_cache_create(const struct janas_jns *j,
                                                     uint64_t budget_bytes,
                                                     int level,
                                                     struct janas_pool *io,
                                                     char *err, size_t err_len);

/* The bytes of an expert slot this cache holds, at the level it was made. */
uint64_t janas_expert_cache_slot_bytes(const struct janas_expert_cache *c);
void janas_expert_cache_destroy(struct janas_expert_cache *c);

size_t janas_expert_cache_slots(const struct janas_expert_cache *c);

/*
 * Loads up to the cache capacity the experts with the highest counts
 * (counts has n_layer * n_expert entries, layer-major; zero means never).
 * The most frequent end up most recently used. Returns 0 or -1 on I/O error.
 */
int janas_expert_cache_warm(struct janas_expert_cache *c,
                            const uint32_t *counts);

/*
 * The same, on a thread of its own, so that the first replies do not wait
 * for it: the most used experts are read first, in batches, and a request
 * from the caller is served between two of them. It stops on its own when
 * the cache has no empty slot left, so it never evicts anything. counts is
 * copied. Returns 0, or -1 if it cannot start.
 */
int janas_expert_cache_warm_background(struct janas_expert_cache *c,
                                       const uint32_t *counts);

/*
 * Whether the background filling is taking the disk and the memory right
 * now. It is not while it waits for the caller to fall idle - which is most
 * of the time, since it only fills when nobody is asking - and a pass
 * measured then is as good as any other. Do not read it from "done < total":
 * the filling stops as soon as the cache is full, and the experts the caller
 * asked for on its own fill slots the filling then never counts, so done
 * ends below total and stays there.
 */
int janas_expert_cache_warming(const struct janas_expert_cache *c);

/* Experts loaded by the background filling so far, and how many it means
   to load (0 and 0 when none was started). */
void janas_expert_cache_warm_progress(const struct janas_expert_cache *c,
                                      uint64_t *done, uint64_t *total);

/*
 * Returns in slots[i] the bytes of expert ids[i] of the layer, reading the
 * missing ones. The pointers stay valid until the next call. n must not
 * exceed the number of slots. Returns 0 or -1 on I/O error.
 */
int janas_expert_cache_fetch(struct janas_expert_cache *c, uint32_t layer,
                             const uint32_t *ids, size_t n,
                             const uint8_t **slots);

/*
 * The same request in two steps, so that computing on the experts already in
 * RAM overlaps the reads of the missing ones. begin fills slots[] for every
 * expert and sets ready[i] = 1 for those already cached; the reads of the
 * others run in the background until janas_expert_cache_finish(), after which
 * every slot is valid. Only one request may be outstanding.
 * begin returns the number of experts being read, or -1.
 */
int janas_expert_cache_begin(struct janas_expert_cache *c, uint32_t layer,
                             const uint32_t *ids, size_t n,
                             const uint8_t **slots, uint8_t *ready);
int janas_expert_cache_finish(struct janas_expert_cache *c);

void janas_expert_cache_stats(const struct janas_expert_cache *c,
                              struct janas_expert_cache_stats *s);
void janas_expert_cache_reset_stats(struct janas_expert_cache *c);

#endif
