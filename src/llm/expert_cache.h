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
    /* reads ahead (janas_expert_cache_prefetch): experts read, of those
       the ones their layer then asked for, and the bytes (not in
       bytes_read) */
    uint64_t prefetch_reads, prefetch_used, prefetch_bytes;
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
 * The arena, when it holds every expert of every layer and nothing is
 * being read into it: from then on no slot changes (there is nothing left
 * to load), so a copy of it stays true - for a GPU. NULL otherwise, and
 * between begin and finish. loads: the experts ever put in a slot, to tell
 * later whether one changed.
 */
const uint8_t *janas_expert_cache_resident(struct janas_expert_cache *c,
                                           uint64_t *bytes, uint64_t *loads);
uint64_t janas_expert_cache_loads(const struct janas_expert_cache *c);
/*
 * The arena moved to arena (n_slots x slot bytes, owned by the caller, not
 * freed here), its contents copied, the file read without O_DIRECT from
 * then on: for memory a GPU's driver allocated, which it reads faster
 * than imported pages. Only when nothing is being read (as _resident); 0,
 * or -1 with nothing changed.
 */
int janas_expert_cache_rebase(struct janas_expert_cache *c, uint8_t *arena);

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

/*
 * Reads ahead up to max of the experts ids of the layer that are not in the
 * cache (256 at most, and an eighth of the cache's slots), in the order
 * given (the most likely first), while the caller goes
 * on: for a layer that will ask soon. They take the least recently used
 * slots and stay last in line, and until that layer's next request they are
 * never taken back (a read in flight must not land in a slot given to
 * another); that request waits for them, counts those it asks for as hits,
 * and leaves the others first in line to go. Called between requests, not
 * between begin and finish; one layer ahead at a time (a new call for
 * another layer settles the one before). Only where the kernel has io_uring,
 * on a ring of its own, so that a request never waits for reads ahead of
 * another layer. Returns the number of reads started (0: none, or no ring).
 */
int janas_expert_cache_prefetch(struct janas_expert_cache *c, uint32_t layer,
                                const uint32_t *ids, size_t n, size_t max);
/* Whether this cache can read ahead at all (the kernel has io_uring). */
int janas_expert_cache_can_prefetch(const struct janas_expert_cache *c);

void janas_expert_cache_stats(const struct janas_expert_cache *c,
                              struct janas_expert_cache_stats *s);
void janas_expert_cache_reset_stats(struct janas_expert_cache *c);

#endif
