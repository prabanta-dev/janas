/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * expert_cache.c - RAM cache of routed experts over a JNS file (see
 * expert_cache.h).
 */
#define _GNU_SOURCE
#include "expert_cache.h"

#include "common/aligned.h"
#include "common/reopen.h"
#include "common/uring.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define NONE (-1)
/* reads in flight at most: a request asks for 256 experts at most */
#define URING_QUEUE 256
/* and at most this many in one batch for the ring to be the better way */
#define URING_BATCH 4
/* and again from this many times the threads of the I/O pool on */
#define URING_DEEP_TIMES 4
/* reads ahead at most, for one layer (a prefill block's), and never more
   than an eighth of the cache: kept until their layer asks, they must not
   leave the layer before too few slots to work with */
#define PF_MAX 256

struct read_job {
    uint64_t offset;
    uint64_t bytes;
    uint8_t *dst;
};

struct janas_expert_cache {
    const struct janas_jns *j;
    int level;         /* 1 to 3: how much of a slot is read (see the header) */
    int fd;            /* O_DIRECT, independent of the jns descriptor */
    int arena_foreign; /* the arena belongs to another (a GPU): not freed */
    struct janas_pool *io;

    uint64_t slot_bytes;
    size_t n_slots;
    uint8_t *arena;
    /* the layers it serves: keys of layers below n_layer, experts of those
       from first on (an MTP block's slots apart: the block's first layer) */
    uint32_t first, n_layer;
    struct janas_expert_cache *wide; /* the MTP block's slots, or NULL */
    int asked_wide; /* the outstanding request is the block's (caller only) */

    int32_t *where; /* (layer, expert) -> slot, n_layer * n_expert */
    int64_t *key;   /* slot -> (layer, expert) or NONE */
    int32_t *prev, *next;
    int32_t head, tail; /* most and least recently used */
    uint64_t *stamp;    /* request that last used the slot */
    uint64_t request;
    /*
     * The slots of each layer in a list of their own, most recently used
     * first, and the pass the requests make: layer after layer, 0 to the
     * last. When the last pass asked for more experts than there are slots
     * (a prefill block large for the cache), plain LRU is the worst policy
     * there is - the expert thrown out is always the one wanted again
     * soonest, and every request misses (issue #9: 11% of hits in blocks of
     * 256 against 83% in blocks of 64). Then the victim is the one wanted
     * again latest: the current layer's other experts (wanted at the next
     * pass), then the layer before, and back from there.
     */
    int32_t *lprev, *lnext, *lhead, *ltail;
    uint32_t cur_layer;
    uint64_t pass_asked, last_pass_asked;

    struct read_job *jobs;
    size_t n_jobs;
    atomic_size_t next_job;
    atomic_int io_error;

    struct janas_expert_cache_stats st;

    /* one request or one batch of the background filling at a time */
    pthread_mutex_t api_lock;
    size_t filled;          /* slots that hold an expert */
    _Atomic uint64_t loads; /* experts ever put in a slot */
    double last_request;    /* when the caller last asked for experts */
    /* the background filling of the cache, if it was started */
    pthread_t warmer;
    int has_warmer, warm_stop;
    uint32_t *warm_counts;
    uint32_t *warm_order; /* keys, most used first */
    size_t warm_n;
    _Atomic uint64_t warm_done, warm_total;
    _Atomic uint64_t warm_last_ns; /* when it last read, 0 never */
    _Atomic int warm_over;         /* the filling has ended, done or not */

    /* background reads: two ways, chosen per batch by by_ring() - the
       kernel through the ring, or the coordinator driving the I/O pool */
    struct janas_uring *ring;
    size_t ring_batch; /* reads at most, for a batch to go on the ring */
    size_t ring_deep;  /* or this many at least (0: no such batch) */
    int via_ring;      /* how the outstanding batch was started */
    double io_t0;      /* when the reads in flight were handed to the kernel */
    pthread_t coordinator;
    int has_coordinator;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int pending;     /* a batch is queued or being read (under lock) */
    int outstanding; /* begin called, finish not yet (caller thread only) */
    int stop;
    int batch_rc;
    int lost; /* a request found no slot for an expert: finish() fails it */

    /*
     * Reads ahead (janas_expert_cache_prefetch), two sets: the engine reads
     * ahead for layer l + 1 before layer l asks, so the set for l + 1 is
     * made while the one for l still waits for its request. Each set has a
     * ring of its own (by the layer's parity), so a request waits only for
     * the reads ahead of its own layer, never for another layer's. The
     * slots they fill are flagged (never a victim) until their layer's
     * request settles them.
     */
    struct janas_uring *pf_ring[2];
    uint8_t *pf_flag; /* per slot */
    int32_t pf_slot[2][PF_MAX];
    size_t pf_n[2];
    uint32_t pf_layer[2];
    int pf_inflight[2];
};

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void unlink_slot(struct janas_expert_cache *c, int32_t s)
{
    if (c->prev[s] != NONE)
        c->next[c->prev[s]] = c->next[s];
    else
        c->head = c->next[s];
    if (c->next[s] != NONE)
        c->prev[c->next[s]] = c->prev[s];
    else
        c->tail = c->prev[s];
}

/* The slot's place in its layer's list (a slot that holds an expert). */
static uint32_t slot_layer(const struct janas_expert_cache *c, int32_t s)
{
    return (uint32_t)(c->key[s] / c->j->h.n_expert);
}

static void layer_unlink(struct janas_expert_cache *c, int32_t s)
{
    uint32_t l = slot_layer(c, s);
    if (c->lprev[s] != NONE)
        c->lnext[c->lprev[s]] = c->lnext[s];
    else
        c->lhead[l] = c->lnext[s];
    if (c->lnext[s] != NONE)
        c->lprev[c->lnext[s]] = c->lprev[s];
    else
        c->ltail[l] = c->lprev[s];
}

static void layer_push_front(struct janas_expert_cache *c, int32_t s)
{
    uint32_t l = slot_layer(c, s);
    c->lprev[s] = NONE;
    c->lnext[s] = c->lhead[l];
    if (c->lhead[l] != NONE)
        c->lprev[c->lhead[l]] = s;
    c->lhead[l] = s;
    if (c->ltail[l] == NONE)
        c->ltail[l] = s;
}

static void push_front(struct janas_expert_cache *c, int32_t s)
{
    c->prev[s] = NONE;
    c->next[s] = c->head;
    if (c->head != NONE)
        c->prev[c->head] = s;
    c->head = s;
    if (c->tail == NONE)
        c->tail = s;
}

/* Last in line: the next to go (a read ahead not yet asked for). */
static void push_back(struct janas_expert_cache *c, int32_t s)
{
    c->next[s] = NONE;
    c->prev[s] = c->tail;
    if (c->tail != NONE)
        c->next[c->tail] = s;
    c->tail = s;
    if (c->head == NONE)
        c->head = s;
}

static void layer_push_back(struct janas_expert_cache *c, int32_t s)
{
    uint32_t l = slot_layer(c, s);
    c->lnext[s] = NONE;
    c->lprev[s] = c->ltail[l];
    if (c->ltail[l] != NONE)
        c->lnext[c->ltail[l]] = s;
    c->ltail[l] = s;
    if (c->lhead[l] == NONE)
        c->lhead[l] = s;
}

static void touch(struct janas_expert_cache *c, int32_t s)
{
    if (c->head != s) {
        unlink_slot(c, s);
        push_front(c, s);
    }
    if (c->key[s] != NONE && c->lhead[slot_layer(c, s)] != s) {
        layer_unlink(c, s);
        layer_push_front(c, s);
    }
    c->stamp[s] = c->request;
}

/*
 * The slot to reuse, never one of the current request: the least recently
 * used, or - the cache full and the last pass larger than it - the one
 * wanted again latest (see lhead).
 */
static int32_t victim(struct janas_expert_cache *c)
{
    if (c->filled == c->n_slots && c->last_pass_asked > c->n_slots) {
        uint32_t nl = c->n_layer;
        for (uint32_t d = 0; d < nl; d++) {
            uint32_t l = (c->cur_layer + nl - d) % nl;
            for (int32_t s = c->ltail[l]; s != NONE; s = c->lprev[s])
                if (c->stamp[s] != c->request && !c->pf_flag[s])
                    return s;
        }
    }
    for (int32_t s = c->tail; s != NONE; s = c->prev[s])
        if (c->stamp[s] != c->request && !c->pf_flag[s])
            return s;
    return NONE;
}

static void io_worker(void *arg, int tid, int n_threads)
{
    (void)tid;
    (void)n_threads;
    struct janas_expert_cache *c = arg;
    for (;;) {
        size_t i = atomic_fetch_add(&c->next_job, 1);
        if (i >= c->n_jobs)
            return;
        const struct read_job *r = &c->jobs[i];
        uint64_t done = 0;
        while (done < r->bytes) {
            ssize_t n = pread(c->fd, r->dst + done, r->bytes - done,
                              (off_t)(r->offset + done));
            if (n < 0 && errno == EINTR)
                continue;
            if (n <= 0) {
                atomic_store(&c->io_error, 1);
                return;
            }
            done += (uint64_t)n;
        }
    }
}

/*
 * Hands the queued reads to the kernel and returns at once: they run while
 * the caller computes on the experts it already has. The ring belongs to
 * whoever holds api_lock, so a request and the background filling never
 * touch it together.
 */
static int submit_jobs(struct janas_expert_cache *c)
{
    if (c->n_jobs == 0)
        return 0;
    c->io_t0 = now();
    int rc = 0;
    for (size_t i = 0; i < c->n_jobs && rc == 0; i++) {
        if (janas_uring_read(c->ring, c->fd, c->jobs[i].dst, c->jobs[i].bytes,
                             c->jobs[i].offset) != 0) {
            /* more reads at once than the ring holds: take these first */
            if (janas_uring_submit(c->ring) != 0 ||
                janas_uring_wait(c->ring) != 0 ||
                janas_uring_read(c->ring, c->fd, c->jobs[i].dst,
                                 c->jobs[i].bytes, c->jobs[i].offset) != 0) {
                rc = -1;
                break;
            }
        }
        c->st.bytes_read += c->jobs[i].bytes;
    }
    c->n_jobs = 0;
    return rc == 0 ? janas_uring_submit(c->ring) : rc;
}

/* Collects the reads in flight. */
static int collect_jobs(struct janas_expert_cache *c)
{
    if (janas_uring_inflight(c->ring) == 0)
        return 0;
    int rc = janas_uring_wait(c->ring);
    c->st.io_seconds += now() - c->io_t0;
    return rc;
}

/* Reads the queued jobs on the I/O pool; the calling thread takes part. */
static int pool_read_jobs(struct janas_expert_cache *c)
{
    if (c->n_jobs == 0)
        return 0;
    double t0 = now();
    atomic_store(&c->next_job, 0);
    atomic_store(&c->io_error, 0);
    janas_pool_run(c->io, io_worker, c);
    c->st.io_seconds += now() - t0;
    for (size_t i = 0; i < c->n_jobs; i++)
        c->st.bytes_read += c->jobs[i].bytes;
    c->n_jobs = 0;
    return atomic_load(&c->io_error) ? -1 : 0;
}

/*
 * Which of the two ways reads this batch. The ring costs nothing to start -
 * the caller queues the reads itself - but the kernel makes its workers one
 * at a time, as each of them blocks, so a short queue is read almost one
 * read at a time. The pool is the other way round: its threads are already
 * there, and reaching them is a wake-up, which measured 271 us. So the ring
 * for a short batch, which is a reply written one token at a time and where
 * that wake-up is the whole cost; the pool for what is in between; and past
 * a queue deep enough that the kernel has made its workers, the ring again,
 * because by then they outnumber the pool's and are free to run anywhere.
 */
static int by_ring(const struct janas_expert_cache *c)
{
    if (!c->ring)
        return 0;
    return c->n_jobs <= c->ring_batch ||
           (c->ring_deep && c->n_jobs >= c->ring_deep);
}

/* Starts the reads without waiting for them (see janas_expert_cache_begin). */
static int start_jobs(struct janas_expert_cache *c)
{
    c->via_ring = 0;
    if (c->n_jobs == 0)
        return 0;
    if (by_ring(c)) {
        c->via_ring = 1;
        return submit_jobs(c);
    }
    pthread_mutex_lock(&c->lock);
    c->pending = 1;
    pthread_cond_broadcast(&c->cond);
    pthread_mutex_unlock(&c->lock);
    return 0;
}

/* Waits for what start_jobs began. */
static int wait_jobs(struct janas_expert_cache *c)
{
    if (c->via_ring)
        return collect_jobs(c);
    pthread_mutex_lock(&c->lock);
    while (c->pending)
        pthread_cond_wait(&c->cond, &c->lock);
    int rc = c->batch_rc;
    c->batch_rc = 0;
    pthread_mutex_unlock(&c->lock);
    return rc;
}

/* Reads the queued jobs to the end, on the calling thread. */
static int read_jobs(struct janas_expert_cache *c)
{
    if (c->n_jobs == 0)
        return 0;
    if (by_ring(c)) {
        int rc = submit_jobs(c);
        return rc == 0 ? collect_jobs(c) : rc;
    }
    return pool_read_jobs(c);
}

static void *coordinator_main(void *arg)
{
    struct janas_expert_cache *c = arg;
    pthread_mutex_lock(&c->lock);
    for (;;) {
        while (!c->pending && !c->stop)
            pthread_cond_wait(&c->cond, &c->lock);
        if (c->stop)
            break;
        pthread_mutex_unlock(&c->lock);
        int rc = pool_read_jobs(c);
        pthread_mutex_lock(&c->lock);
        c->batch_rc = rc;
        c->pending = 0;
        pthread_cond_broadcast(&c->cond);
    }
    pthread_mutex_unlock(&c->lock);
    return NULL;
}

/* Synchronous reads, from the caller: warm-up and janas_expert_cache_fetch. */
static int run_jobs(struct janas_expert_cache *c)
{
    double t0 = now();
    int rc = read_jobs(c);
    c->st.wait_seconds += now() - t0;
    return rc;
}

/* Assigns a slot to (layer, expert) and queues its read. */
static int32_t load(struct janas_expert_cache *c, uint32_t layer, uint32_t e)
{
    int32_t s = victim(c);
    if (s == NONE)
        return NONE;
    if (c->key[s] != NONE) {
        c->where[c->key[s]] = NONE;
        layer_unlink(c, s);
    } else {
        c->filled++;
    }
    atomic_fetch_add(&c->loads, 1);
    int64_t k = (int64_t)layer * c->j->h.n_expert + e;
    c->key[s] = k;
    c->where[k] = s;
    layer_push_front(c, s);
    touch(c, s);
    c->jobs[c->n_jobs++] = (struct read_job){
        .offset = janas_jns_expert_offset(c->j, layer, e),
        .bytes = c->j->layers[layer].level_bytes[c->level - 1],
        .dst = c->arena + (size_t)s * c->slot_bytes};
    return s;
}

/* How a budget is cut in slots: n_slots of slot bytes, and n_wide of
   wide_slot for an MTP block whose experts are larger (0: none apart). */
struct layout {
    uint64_t slot, wide_slot;
    size_t n_slots, n_wide;
};

static void clamp_layers(const struct janas_jns *j, uint32_t *n_layer,
                         uint32_t *n_main)
{
    if (*n_layer == 0 || *n_layer > j->h.n_layer)
        *n_layer = j->h.n_layer;
    if (*n_main == 0 || *n_main > *n_layer)
        *n_main = *n_layer;
}

static void layout(const struct janas_jns *j, uint32_t n_layer, uint32_t n_main,
                   int level, uint64_t budget, struct layout *o)
{
    size_t ne = j->h.n_expert, keys = (size_t)n_layer * ne,
           mkeys = (size_t)n_main * ne, wkeys = keys - mkeys;
    uint64_t main = 0, wide = 0;
    for (uint32_t l = 0; l < n_layer; l++) {
        uint64_t b = j->layers[l].level_bytes[level - 1], *m = &main;
        if (l >= n_main)
            m = &wide;
        if (b > *m)
            *m = b;
    }
    /* one size for all, the largest (a file whose every slot is empty:
       none to hold) */
    uint64_t one = main > wide ? main : wide;
    size_t n = one ? (size_t)(budget / one) : 0;
    if (!one)
        one = 1;
    *o = (struct layout){.slot = one, .n_slots = n < keys ? n : keys};
    if (wide <= main || main == 0)
        return;
    /* or the block apart: room for one request, or for all its experts
       where the budget holds every expert of both */
    uint64_t ws = (wide + main - 1) / main * main;
    size_t wn = wkeys < JANAS_EXPERT_CHUNK ? wkeys : JANAS_EXPERT_CHUNK;
    if (budget >= (uint64_t)mkeys * main + (uint64_t)wkeys * ws)
        wn = wkeys;
    if (budget < (uint64_t)wn * ws)
        return;
    size_t mn = (size_t)((budget - (uint64_t)wn * ws) / main);
    if (mn > mkeys)
        mn = mkeys;
    /* only where it holds more experts (all of them: in less memory) */
    if (mn < j->h.n_expert_used || mn + wn < o->n_slots)
        return;
    *o = (struct layout){
        .slot = main, .wide_slot = ws, .n_slots = mn, .n_wide = wn};
}

uint64_t janas_expert_cache_room(const struct janas_jns *j, uint32_t n_layer,
                                 uint32_t n_main, int level)
{
    struct layout o;
    clamp_layers(j, &n_layer, &n_main);
    layout(j, n_layer, n_main, level, UINT64_MAX, &o);
    return (uint64_t)o.n_slots * o.slot + (uint64_t)o.n_wide * o.wide_slot;
}

/* The bytes of every slot, the MTP block's after the model's. */
static uint64_t arena_used(const struct janas_expert_cache *c)
{
    uint64_t b = (uint64_t)c->n_slots * c->slot_bytes;
    if (c->wide)
        b += (uint64_t)c->wide->n_slots * c->wide->slot_bytes;
    return b;
}

/*
 * A cache of n_slots slots for the experts of layers first to n_layer - 1,
 * in arena (someone else's, not freed here), or in an arena of its own of
 * used bytes and the guard.
 */
static struct janas_expert_cache *
part_create(const struct janas_jns *j, uint32_t first, uint32_t n_layer,
            uint64_t slot_bytes, size_t n_slots, int level,
            struct janas_pool *io, uint8_t *arena, uint64_t used, char *err,
            size_t err_len)
{
    size_t n_keys = (size_t)n_layer * j->h.n_expert;
    struct janas_expert_cache *c = calloc(1, sizeof(*c));
    if (!c)
        return NULL;
    c->j = j;
    c->level = level;
    c->first = first;
    c->n_layer = n_layer;
    c->io = io;
    c->slot_bytes = slot_bytes;
    c->n_slots = n_slots;
    c->head = c->tail = NONE;

    c->fd = janas_reopen_read(j->fd, 1);
    /* on 2 MB boundaries, for a GPU that imports it (model_load.c
       janas_m_huge_pages) */
    size_t arena_bytes = (used + JANAS_ARENA_GUARD + JANAS_HUGE_PAGE - 1) /
                         JANAS_HUGE_PAGE * JANAS_HUGE_PAGE;
    c->arena_foreign = arena != NULL;
    c->arena =
        arena ? arena : janas_aligned_alloc(JANAS_HUGE_PAGE, arena_bytes);
    c->where = malloc(n_keys * sizeof(int32_t));
    c->key = malloc(n_slots * sizeof(int64_t));
    c->prev = malloc(n_slots * sizeof(int32_t));
    c->next = malloc(n_slots * sizeof(int32_t));
    c->stamp = calloc(n_slots, sizeof(uint64_t));
    c->jobs = malloc(n_slots * sizeof(struct read_job));
    /* the MTP file's block may come as a layer after the model's: one list
       more than there are layers costs nothing */
    c->lprev = malloc(n_slots * sizeof(int32_t));
    c->lnext = malloc(n_slots * sizeof(int32_t));
    c->lhead = malloc(((size_t)n_layer + 1) * sizeof(int32_t));
    c->ltail = malloc(((size_t)n_layer + 1) * sizeof(int32_t));
    c->pf_flag = calloc(n_slots, 1);
    /* before any way out: janas_expert_cache_destroy destroys them */
    pthread_mutex_init(&c->api_lock, NULL);
    pthread_mutex_init(&c->lock, NULL);
    pthread_cond_init(&c->cond, NULL);
    if (c->fd < 0 || !c->arena || !c->where || !c->key || !c->prev ||
        !c->next || !c->stamp || !c->jobs || !c->lprev || !c->lnext ||
        !c->lhead || !c->ltail || !c->pf_flag) {
        snprintf(err, err_len, "cannot allocate %zu slots of %llu bytes",
                 n_slots, (unsigned long long)slot_bytes);
        janas_expert_cache_destroy(c);
        return NULL;
    }
    /*
     * No MADV_HUGEPAGE: with the kernel's transparent huge pages on
     * "always" and their defrag on "madvise" (as on the development
     * laptop), a region so marked makes every first write to a page try a
     * synchronous compaction of memory, which on a machine with fragmented
     * memory fails and only costs. Measured on 30 Sep 2026 (Qwen3-Next-80B, a
     * fresh 20 GiB cache): ~1,800-8,800 failed compactions a reply, decode
     * 17.5 -> 19.7 tokens/s without it, prefill 51 -> 55, and no loss once
     * the cache is warm (23.0-23.6 against 23.1-23.3) - the kernel still
     * gives huge pages where they are free. JANAS_ARENA_HUGE=1 asks for
     * them again.
     */
    if (!arena && getenv("JANAS_ARENA_HUGE") &&
        atoi(getenv("JANAS_ARENA_HUGE")) > 0)
        madvise(c->arena, arena_bytes, MADV_HUGEPAGE);
    /*
     * Asynchronous reads, where the kernel has io_uring: for the batches it
     * is good at (see by_ring), begin() queues them itself, a system call,
     * instead of waking the coordinator - which on 20 Sep 2026 measured 271
     * us a time, 2.43 ms a token, because a thread woken by a busy one lands
     * on its CPU and takes its turn. JANAS_URING=0 keeps the old way whole,
     * and the two thresholds move with JANAS_URING_BATCH and _DEEP.
     */
    const char *ur = getenv("JANAS_URING");
    if (!ur || atoi(ur) != 0)
        c->ring = janas_uring_create(URING_QUEUE);
    if (c->ring) {
        c->pf_ring[0] = janas_uring_create(PF_MAX);
        c->pf_ring[1] = c->pf_ring[0] ? janas_uring_create(PF_MAX) : NULL;
        if (!c->pf_ring[1]) {
            janas_uring_destroy(c->pf_ring[0]);
            c->pf_ring[0] = NULL;
        }
    }
    c->ring_batch = URING_BATCH;
    /* deep enough that the kernel has made more readers than the pool has
       threads: measured on 21 Sep 2026 at 32 reads, with eight of them */
    c->ring_deep = URING_DEEP_TIMES * (size_t)janas_pool_size(io);
    const char *ub = getenv("JANAS_URING_BATCH");
    if (ub)
        c->ring_batch = (size_t)atoi(ub);
    const char *ud = getenv("JANAS_URING_DEEP");
    if (ud)
        c->ring_deep = (size_t)atoi(ud);
    /* the coordinator stays: it reads the batches the ring is bad at, and it
       is the whole of the reading where there is no io_uring at all */
    if (pthread_create(&c->coordinator, NULL, coordinator_main, c) != 0) {
        snprintf(err, err_len, "cannot start the I/O coordinator");
        janas_expert_cache_destroy(c);
        return NULL;
    }
    c->has_coordinator = 1;
    for (size_t k = 0; k < n_keys; k++)
        c->where[k] = NONE;
    for (uint32_t l = 0; l <= n_layer; l++)
        c->lhead[l] = c->ltail[l] = NONE;
    for (size_t s = 0; s < n_slots; s++) {
        c->key[s] = NONE;
        push_front(c, (int32_t)s);
    }
    c->request = 1; /* stamps start at 0: no slot is in use */
    return c;
}

struct janas_expert_cache *
janas_expert_cache_create_split(const struct janas_jns *j, uint32_t n_layer,
                                uint32_t n_main, uint64_t budget_bytes,
                                int level, struct janas_pool *io, char *err,
                                size_t err_len)
{
    if (level < 1 || level > 3) {
        snprintf(err, err_len, "expert level %d out of range", level);
        return NULL;
    }
    struct layout o;
    clamp_layers(j, &n_layer, &n_main);
    layout(j, n_layer, n_main, level, budget_bytes, &o);
    if (o.n_slots < j->h.n_expert_used) {
        snprintf(err, err_len, "budget below one layer of experts");
        return NULL;
    }
    uint64_t at = (uint64_t)o.n_slots * o.slot;
    if (!o.n_wide)
        return part_create(j, 0, n_layer, o.slot, o.n_slots, level, io, NULL,
                           at, err, err_len);
    /* the block's slots right after the model's, in the same arena: each a
       whole number of the model's, they start where one of those would */
    struct janas_expert_cache *c =
        part_create(j, 0, n_main, o.slot, o.n_slots, level, io, NULL,
                    at + (uint64_t)o.n_wide * o.wide_slot, err, err_len);
    if (!c)
        return NULL;
    c->wide = part_create(j, n_main, n_layer, o.wide_slot, o.n_wide, level, io,
                          c->arena + at, 0, err, err_len);
    if (!c->wide) {
        janas_expert_cache_destroy(c);
        return NULL;
    }
    return c;
}

struct janas_expert_cache *janas_expert_cache_create(const struct janas_jns *j,
                                                     uint64_t budget_bytes,
                                                     int level,
                                                     struct janas_pool *io,
                                                     char *err, size_t err_len)
{
    return janas_expert_cache_create_split(j, 0, 0, budget_bytes, level, io,
                                           err, err_len);
}

void janas_expert_cache_destroy(struct janas_expert_cache *c)
{
    if (!c)
        return;
    janas_expert_cache_destroy(c->wide); /* in this arena: before it goes */
    if (c->has_warmer) { /* it holds the lock only for one batch */
        pthread_mutex_lock(&c->api_lock);
        c->warm_stop = 1;
        pthread_mutex_unlock(&c->api_lock);
        pthread_join(c->warmer, NULL);
        c->has_warmer = 0;
    }
    free(c->warm_counts);
    free(c->warm_order);
    for (int p = 0; p < 2; p++) {
        if (c->pf_ring[p] && c->pf_inflight[p]) /* no read may land after */
            janas_uring_wait(c->pf_ring[p]);
        janas_uring_destroy(c->pf_ring[p]);
    }
    free(c->pf_flag);
    if (c->has_coordinator) {
        pthread_mutex_lock(&c->lock);
        while (c->pending)
            pthread_cond_wait(&c->cond, &c->lock);
        c->stop = 1;
        pthread_cond_broadcast(&c->cond);
        pthread_mutex_unlock(&c->lock);
        pthread_join(c->coordinator, NULL);
    }
    pthread_mutex_destroy(&c->lock);
    pthread_cond_destroy(&c->cond);
    janas_uring_destroy(c->ring);
    pthread_mutex_destroy(&c->api_lock);
    if (c->fd >= 0)
        close(c->fd);
    if (!c->arena_foreign)
        janas_aligned_free(c->arena);
    free(c->where);
    free(c->key);
    free(c->prev);
    free(c->lprev);
    free(c->lnext);
    free(c->lhead);
    free(c->ltail);
    free(c->next);
    free(c->stamp);
    free(c->jobs);
    free(c);
}

uint64_t janas_expert_cache_slot_bytes(const struct janas_expert_cache *c)
{
    return c->slot_bytes;
}

size_t janas_expert_cache_slots(const struct janas_expert_cache *c)
{
    return c->n_slots;
}

static int by_key(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* order's n keys by ascending counts[key] (the most frequent loaded last),
   ties by key; with nothing shared between callers, as two caches may sort
   at once. Left as it is without memory for the sort. */
static void sort_by_count(uint32_t *order, size_t n, const uint32_t *counts)
{
    uint64_t *k = n ? malloc(n * sizeof(*k)) : NULL;
    if (!k)
        return;
    for (size_t i = 0; i < n; i++)
        k[i] = (uint64_t)counts[order[i]] << 32 | order[i];
    qsort(k, n, sizeof(*k), by_key);
    for (size_t i = 0; i < n; i++)
        order[i] = (uint32_t)k[i];
    free(k);
}

/*
 * Fills the cache in batches while the caller works: it takes the lock for
 * one batch at a time, so a request waits at most the time of one batch (a
 * few tens of milliseconds), and stops as soon as the cache is full, so it
 * never takes a slot away from anything.
 */
/*
 * One expert at a time, so a request waits a millisecond at most, and only
 * while the caller is idle: filling the cache during a reply takes the disk
 * and the memory bandwidth away from it, which costs more than it gives
 * (measured on 20 Sep 2026: 18.4 token/s against 24.5 with the cache filled
 * before the first reply). A chat is idle most of the time - while its user
 * reads and types - and that is when the rest of the cache is filled.
 */
#define WARM_BATCH 1
#define WARM_IDLE 0.05   /* seconds since the last request */
#define WARM_RECENT 0.25 /* it counts as filling for this long after a read */

static void *warmer_main(void *arg)
{
    struct janas_expert_cache *c = arg;
    for (size_t i = 0; i < c->warm_n; i += WARM_BATCH) {
        pthread_mutex_lock(&c->api_lock);
        if (c->warm_stop || c->filled >= c->n_slots) {
            pthread_mutex_unlock(&c->api_lock);
            break;
        }
        double quiet = now() - c->last_request;
        if (quiet < WARM_IDLE) { /* the caller is working: wait */
            pthread_mutex_unlock(&c->api_lock);
            struct timespec ts = {0, (long)((WARM_IDLE - quiet) * 1e9)};
            nanosleep(&ts, NULL);
            i -= WARM_BATCH; /* this one is still to do */
            continue;
        }
        size_t end = i + WARM_BATCH < c->warm_n ? i + WARM_BATCH : c->warm_n;
        uint64_t loaded = 0;
        for (size_t k = i; k < end && c->filled < c->n_slots; k++) {
            uint32_t key = c->warm_order[k];
            if (c->where[key] != NONE)
                continue;
            c->request++; /* every insertion is its own request */
            load(c, key / c->j->h.n_expert, key % c->j->h.n_expert);
            loaded++;
        }
        uint64_t before = c->st.bytes_read;
        int rc = read_jobs(c);
        /* the bytes of the filling are not the bytes a reply had to wait
           for: they are counted apart, so the report stays honest */
        c->st.warm_bytes += c->st.bytes_read - before;
        c->st.bytes_read = before;
        atomic_fetch_add(&c->warm_done, loaded);
        if (loaded)
            atomic_store(&c->warm_last_ns, (uint64_t)(now() * 1e9));
        pthread_mutex_unlock(&c->api_lock);
        if (rc != 0)
            break;
        sched_yield();
    }
    atomic_store(&c->warm_over, 1);
    return NULL;
}

int janas_expert_cache_warm_background(struct janas_expert_cache *c,
                                       const uint32_t *counts)
{
    size_t n_keys = (size_t)c->n_layer * c->j->h.n_expert;
    if (c->has_warmer)
        return -1;
    c->warm_counts = malloc(n_keys * sizeof(uint32_t));
    c->warm_order = malloc(n_keys * sizeof(uint32_t));
    if (!c->warm_counts || !c->warm_order) {
        free(c->warm_counts);
        free(c->warm_order);
        c->warm_counts = c->warm_order = NULL;
        return -1;
    }
    memcpy(c->warm_counts, counts, n_keys * sizeof(uint32_t));
    size_t n = 0;
    for (size_t k = 0; k < n_keys; k++)
        if (c->warm_counts[k])
            c->warm_order[n++] = (uint32_t)k;
    sort_by_count(c->warm_order, n, c->warm_counts);
    /* the least used come first: read the most used first here */
    for (size_t i = 0; i < n / 2; i++) {
        uint32_t t = c->warm_order[i];
        c->warm_order[i] = c->warm_order[n - 1 - i];
        c->warm_order[n - 1 - i] = t;
    }
    c->warm_n = n < c->n_slots ? n : c->n_slots;
    atomic_store(&c->warm_done, 0);
    atomic_store(&c->warm_total, c->warm_n);
    atomic_store(&c->warm_last_ns, (uint64_t)(now() * 1e9));
    if (pthread_create(&c->warmer, NULL, warmer_main, c) != 0) {
        atomic_store(&c->warm_total, 0);
        return -1;
    } /* warm_total stays what it meant to load: done may end below it when
         the cache fills up with what the caller asked for instead */
    c->has_warmer = 1;
    return 0;
}

/* Something is being read into the part, or about to be (under
   api_lock). */
static int part_busy(struct janas_expert_cache *c)
{
    int busy = c->outstanding || c->pf_n[0] || c->pf_n[1] ||
               (c->has_warmer && !atomic_load(&c->warm_over));
    pthread_mutex_lock(&c->lock);
    busy |= c->pending;
    pthread_mutex_unlock(&c->lock);
    return busy;
}

/* The part holds every expert of its layers, and nothing is being read. */
static int part_complete(struct janas_expert_cache *c, uint64_t *loads)
{
    size_t total = (size_t)(c->n_layer - c->first) * c->j->h.n_expert;
    /* between begin and finish the caller holds api_lock: asked there, the
       answer is no rather than waiting on itself (outstanding is the
       caller's own) */
    if (c->n_slots < total || c->outstanding)
        return 0;
    pthread_mutex_lock(&c->api_lock);
    int busy = part_busy(c);
    int full = c->filled >= total;
    *loads = atomic_load(&c->loads);
    pthread_mutex_unlock(&c->api_lock);
    return !busy && full;
}

const uint8_t *janas_expert_cache_resident(struct janas_expert_cache *c,
                                           uint64_t *bytes, uint64_t *loads)
{
    uint64_t wide_loads = 0;
    if (c->asked_wide || !part_complete(c, loads) ||
        (c->wide && !part_complete(c->wide, &wide_loads)))
        return NULL;
    *loads += wide_loads;
    *bytes = arena_used(c) + JANAS_ARENA_GUARD;
    return c->arena;
}

const uint8_t *janas_expert_cache_arena(const struct janas_expert_cache *c,
                                        uint64_t *bytes)
{
    *bytes = arena_used(c) + JANAS_ARENA_GUARD;
    return c->arena;
}

int janas_expert_cache_rebase(struct janas_expert_cache *c, uint8_t *arena)
{
    struct janas_expert_cache *w = c->wide;
    pthread_mutex_lock(&c->api_lock);
    if (w)
        pthread_mutex_lock(&w->api_lock);
    int busy = c->asked_wide || part_busy(c) || (w && part_busy(w));
    /* the new memory takes no O_DIRECT reads (a GPU driver's mapping):
       the file opened again without it, for the loads that may still come
       (the MTP block's slots read through a descriptor of their own) */
    int fd = busy ? -1 : janas_reopen_read(c->j->fd, 0);
    int wfd = fd >= 0 && w ? janas_reopen_read(c->j->fd, 0) : -1;
    if (fd < 0 || (w && wfd < 0)) {
        if (fd >= 0)
            close(fd);
        if (w)
            pthread_mutex_unlock(&w->api_lock);
        pthread_mutex_unlock(&c->api_lock);
        return -1;
    }
    memcpy(arena, c->arena, arena_used(c));
    close(c->fd);
    c->fd = fd;
    if (!c->arena_foreign)
        janas_aligned_free(c->arena);
    c->arena = arena;
    c->arena_foreign = 1;
    if (w) {
        close(w->fd);
        w->fd = wfd;
        w->arena = arena + (size_t)c->n_slots * c->slot_bytes;
        pthread_mutex_unlock(&w->api_lock);
    }
    pthread_mutex_unlock(&c->api_lock);
    return 0;
}

uint64_t janas_expert_cache_loads(const struct janas_expert_cache *c)
{
    uint64_t n = atomic_load(&((struct janas_expert_cache *)c)->loads);
    return c->wide ? n + janas_expert_cache_loads(c->wide) : n;
}

int janas_expert_cache_warming(const struct janas_expert_cache *c)
{
    if (!c->has_warmer)
        return 0;
    uint64_t t = atomic_load(&((struct janas_expert_cache *)c)->warm_last_ns);
    return t != 0 && now() - (double)t * 1e-9 < WARM_RECENT;
}

void janas_expert_cache_warm_progress(const struct janas_expert_cache *c,
                                      uint64_t *done, uint64_t *total)
{
    /* over, the total is what it did: it stops short when the caller's
       own requests fill the cache first, since the experts they brought in
       are not its to count, and a front end showing done of total would
       show a filling that never ends */
    uint64_t d = atomic_load(&c->warm_done);
    if (done)
        *done = d;
    if (total)
        *total = atomic_load(&c->warm_over) ? d : atomic_load(&c->warm_total);
}

int janas_expert_cache_warm(struct janas_expert_cache *c,
                            const uint32_t *counts)
{
    size_t n_keys = (size_t)c->n_layer * c->j->h.n_expert;
    uint32_t *order = malloc(n_keys * sizeof(uint32_t));
    if (!order)
        return -1;
    size_t n = 0;
    for (size_t k = 0; k < n_keys; k++)
        if (counts[k])
            order[n++] = (uint32_t)k;
    sort_by_count(order, n, counts);
    size_t first = n > c->n_slots ? n - c->n_slots : 0;
    int rc = 0;
    for (size_t i = first; i < n && rc == 0; i++) {
        uint32_t layer = order[i] / c->j->h.n_expert;
        uint32_t e = order[i] % c->j->h.n_expert;
        c->request++; /* every insertion is its own request */
        if (c->where[order[i]] == NONE)
            load(c, layer, e);
        if (c->n_jobs == c->n_slots || c->n_jobs >= 64)
            rc = run_jobs(c);
    }
    if (rc == 0)
        rc = run_jobs(c);
    free(order);
    janas_expert_cache_reset_stats(c);
    return rc;
}

/* A set of reads ahead complete (under api_lock); on an I/O error their
   slots are emptied, so that nothing takes them for what they were to
   hold. */
static void pf_collect(struct janas_expert_cache *c, int p)
{
    if (!c->pf_inflight[p])
        return;
    double t0 = now();
    int rc = janas_uring_wait(c->pf_ring[p]);
    c->st.wait_seconds += now() - t0;
    c->pf_inflight[p] = 0;
    if (rc == 0)
        return;
    for (size_t i = 0; i < c->pf_n[p]; i++) {
        int32_t s = c->pf_slot[p][i];
        c->where[c->key[s]] = NONE;
        layer_unlink(c, s);
        c->key[s] = NONE;
        c->filled--;
    }
}

/* A set settled: complete, and no longer kept (under api_lock). */
static void pf_settle(struct janas_expert_cache *c, int p)
{
    pf_collect(c, p);
    for (size_t i = 0; i < c->pf_n[p]; i++)
        c->pf_flag[c->pf_slot[p][i]] = 0;
    c->pf_n[p] = 0;
}

int janas_expert_cache_prefetch(struct janas_expert_cache *c, uint32_t layer,
                                const uint32_t *ids, size_t n, size_t max)
{
    if (c->wide && layer >= c->n_layer)
        return janas_expert_cache_prefetch(c->wide, layer, ids, n, max);
    int p = (int)(layer & 1);
    if (!c->pf_ring[p] || layer >= c->n_layer || c->outstanding ||
        c->asked_wide)
        return 0;
    pthread_mutex_lock(&c->api_lock);
    pf_settle(c, p); /* a set left over (its layer never asked) */
    size_t base = (size_t)layer * c->j->h.n_expert, count = 0;
    uint64_t bytes = c->j->layers[layer].level_bytes[c->level - 1];
    if (max > c->n_slots / 8)
        max = c->n_slots / 8;
    for (size_t i = 0; i < n && count < max && count < PF_MAX; i++) {
        if (ids[i] >= c->j->h.n_expert || c->where[base + ids[i]] != NONE)
            continue;
        int32_t s = victim(c);
        if (s == NONE)
            break;
        if (c->key[s] != NONE) {
            c->where[c->key[s]] = NONE;
            layer_unlink(c, s);
        } else {
            c->filled++;
        }
        atomic_fetch_add(&c->loads, 1);
        int64_t k = (int64_t)base + ids[i];
        c->key[s] = k;
        c->where[k] = s;
        layer_push_back(c, s);
        unlink_slot(c, s);
        push_back(c, s);
        if (janas_uring_read(c->pf_ring[p], c->fd,
                             c->arena + (size_t)s * c->slot_bytes, bytes,
                             janas_jns_expert_offset(c->j, layer, ids[i])) !=
            0) { /* the ring is full: this one stays unread */
            c->where[k] = NONE;
            layer_unlink(c, s);
            c->key[s] = NONE;
            c->filled--;
            break;
        }
        c->pf_flag[s] = 1;
        c->pf_slot[p][c->pf_n[p]++] = s;
        count++;
    }
    if (count) {
        c->pf_layer[p] = layer;
        c->pf_inflight[p] = 1;
        c->st.prefetch_reads += count;
        c->st.prefetch_bytes += count * bytes;
        if (janas_uring_submit(c->pf_ring[p]) != 0)
            pf_settle(c, p); /* waits and empties what did not start */
    }
    pthread_mutex_unlock(&c->api_lock);
    return (int)count;
}

int janas_expert_cache_can_prefetch(const struct janas_expert_cache *c)
{
    return c->pf_ring[0] != NULL;
}

/* begin, and with same set the request before's still: its slots are not
   taken for this one's (victim) */
static int begin_req(struct janas_expert_cache *c, uint32_t layer,
                     const uint32_t *ids, size_t n, const uint8_t **slots,
                     uint8_t *ready, int same)
{
    if (layer >= c->n_layer || n > c->n_slots || n > 256 || c->outstanding ||
        c->asked_wide)
        return -1;
    /* held until finish: the background filling waits for its turn */
    pthread_mutex_lock(&c->api_lock);
    c->last_request = now();
    if (!same)
        c->request++;
    /* a layer before the last one asked: a new pass */
    if (layer < c->cur_layer) {
        c->last_pass_asked = c->pass_asked;
        c->pass_asked = 0;
    }
    c->cur_layer = layer;
    c->pass_asked += n;
    /* reads ahead for this layer (or for one it went past): wait for them;
       counted as used where a request asks for them. This layer's set is
       kept until the layer is done: a prefill block asks for its experts
       64 at a time, and settling the set at the first request left the
       slots read for the later chunks free to go before those asked (6 Oct
       2026: 42% of the reads ahead used, the cache poorer for the decoding
       after). The set of a later layer stays as it is, still in flight. */
    int pf_here[2];
    for (int p = 0; p < 2; p++) {
        pf_here[p] = c->pf_n[p] && layer >= c->pf_layer[p];
        if (pf_here[p])
            pf_collect(c, p);
    }
    int32_t s[256];
    size_t base = (size_t)layer * c->j->h.n_expert;
    /* hits first, so that they are stamped before any victim is chosen */
    for (size_t i = 0; i < n; i++) {
        if (ids[i] >= c->j->h.n_expert) {
            pthread_mutex_unlock(&c->api_lock);
            return -1;
        }
        s[i] = c->where[base + ids[i]];
        ready[i] = s[i] != NONE;
        if (s[i] != NONE) {
            /* a read ahead for this very layer: counted once */
            if (c->pf_flag[s[i]] == 1 && pf_here[layer & 1] &&
                c->pf_layer[layer & 1] == layer) {
                c->st.prefetch_used++;
                c->pf_flag[s[i]] = 2;
            }
            touch(c, s[i]);
            c->st.hits++;
        }
    }
    /* settled too: a set more than one layer ahead, which is of the pass
       before (the last layer's, when this pass reads nothing ahead) */
    for (int p = 0; p < 2; p++)
        if (c->pf_n[p] &&
            (c->pf_layer[p] < layer || c->pf_layer[p] > layer + 1))
            pf_settle(c, p); /* the unused ones: last in line, free to go */
    int queued = 0;
    for (size_t i = 0; i < n; i++)
        if (s[i] == NONE) {
            int32_t w = c->where[base + ids[i]];
            if (w != NONE) { /* the same id twice in one request */
                s[i] = w;
                continue;
            }
            s[i] = load(c, layer, ids[i]);
            if (s[i] == NONE) {
                /* every free slot held by reads ahead: settled, and
                   free to go */
                for (int p = 0; p < 2; p++)
                    pf_settle(c, p);
                s[i] = load(c, layer, ids[i]);
            }
            c->st.misses++;
            queued++;
        }
    c->st.requests += n;
    int lost = 0;
    for (size_t i = 0; i < n; i++)
        lost |= s[i] == NONE;
    for (size_t i = 0; i < n; i++)
        slots[i] =
            s[i] != NONE ? c->arena + (size_t)s[i] * c->slot_bytes : NULL;
    c->outstanding = 1;
    c->lost = lost; /* a slot not found is NULL, never to be read */
    if (start_jobs(c) != 0)
        c->batch_rc = -1; /* finish() reports it, and waits all the same */
    return queued;
}

int janas_expert_cache_begin(struct janas_expert_cache *c, uint32_t layer,
                             const uint32_t *ids, size_t n,
                             const uint8_t **slots, uint8_t *ready)
{
    if (c->wide && layer >= c->n_layer) {
        if (c->outstanding || c->asked_wide)
            return -1;
        int rc = janas_expert_cache_begin(c->wide, layer, ids, n, slots, ready);
        c->asked_wide = rc >= 0;
        return rc;
    }
    return begin_req(c, layer, ids, n, slots, ready, 0);
}

void janas_expert_cache_demote(struct janas_expert_cache *c, uint32_t layer,
                               const uint32_t *ids, size_t n)
{
    if (c->wide && layer >= c->n_layer) {
        janas_expert_cache_demote(c->wide, layer, ids, n);
        return;
    }
    uint32_t ne = c->j->h.n_expert;
    if (layer >= c->n_layer)
        return;
    size_t base = (size_t)layer * ne;
    pthread_mutex_lock(&c->api_lock);
    for (size_t i = 0; i < n; i++) {
        if (ids[i] >= ne)
            continue;
        int32_t s = c->where[base + ids[i]];
        if (s == NONE || c->pf_flag[s])
            continue;
        if (c->tail != s) {
            unlink_slot(c, s);
            push_back(c, s);
        }
        if (c->ltail[layer] != s) {
            layer_unlink(c, s);
            layer_push_back(c, s);
        }
    }
    pthread_mutex_unlock(&c->api_lock);
}

void janas_expert_cache_present(struct janas_expert_cache *c, uint32_t layer,
                                const uint32_t *ids, size_t n, uint8_t *in)
{
    if (c->wide && layer >= c->n_layer) {
        janas_expert_cache_present(c->wide, layer, ids, n, in);
        return;
    }
    uint32_t ne = c->j->h.n_expert;
    size_t base = (size_t)layer * ne;
    /* the background filling changes the map under this lock */
    pthread_mutex_lock(&c->api_lock);
    for (size_t i = 0; i < n; i++)
        in[i] = layer < c->n_layer && ids[i] < ne &&
                c->where[base + ids[i]] != NONE;
    pthread_mutex_unlock(&c->api_lock);
}

int janas_expert_cache_finish(struct janas_expert_cache *c)
{
    if (c->asked_wide) {
        c->asked_wide = 0;
        return janas_expert_cache_finish(c->wide);
    }
    double t0 = now();
    int rc = wait_jobs(c);
    if ((c->via_ring && c->batch_rc) || c->lost) /* the submission itself
                                                     failed, or a slot */
        rc = -1;
    c->batch_rc = 0;
    c->lost = 0;
    c->outstanding = 0;
    c->st.wait_seconds += now() - t0;
    pthread_mutex_unlock(&c->api_lock);
    return rc;
}

int janas_expert_cache_fetch(struct janas_expert_cache *c, uint32_t layer,
                             const uint32_t *ids, size_t n,
                             const uint8_t **slots)
{
    uint8_t ready[256];
    if (janas_expert_cache_begin(c, layer, ids, n, slots, ready) < 0)
        return -1;
    return janas_expert_cache_finish(c);
}

int janas_expert_cache_fetch_many(struct janas_expert_cache *c, uint32_t layer,
                                  const uint32_t *ids, size_t n,
                                  const uint8_t **slots)
{
    if (c->wide && layer >= c->n_layer)
        return janas_expert_cache_fetch_many(c->wide, layer, ids, n, slots);
    /* room besides the slots read ahead, which a request never takes */
    if (2 * n > c->n_slots)
        return -1;
    uint8_t ready[256];
    for (size_t i = 0; i < n; i += 256) {
        size_t k = n - i < 256 ? n - i : 256;
        if (begin_req(c, layer, ids + i, k, slots + i, ready, i > 0) < 0)
            return -1;
        if (janas_expert_cache_finish(c) != 0)
            return -1;
    }
    for (size_t i = 0; i < n; i++)
        if (!slots[i])
            return -1;
    return 0;
}

void janas_expert_cache_stats(const struct janas_expert_cache *c,
                              struct janas_expert_cache_stats *s)
{
    struct janas_expert_cache *m = (struct janas_expert_cache *)c;
    pthread_mutex_lock(&m->api_lock);
    *s = c->st;
    pthread_mutex_unlock(&m->api_lock);
    if (!c->wide)
        return;
    /* the MTP block's lookups with the model's, as when they shared the
       slots: the hit rate stays the one of every request */
    struct janas_expert_cache_stats w;
    janas_expert_cache_stats(c->wide, &w);
    s->requests += w.requests;
    s->hits += w.hits;
    s->misses += w.misses;
    s->bytes_read += w.bytes_read;
    s->warm_bytes += w.warm_bytes;
    s->io_seconds += w.io_seconds;
    s->wait_seconds += w.wait_seconds;
    s->prefetch_reads += w.prefetch_reads;
    s->prefetch_used += w.prefetch_used;
    s->prefetch_bytes += w.prefetch_bytes;
}

void janas_expert_cache_reset_stats(struct janas_expert_cache *c)
{
    memset(&c->st, 0, sizeof(c->st));
    if (c->wide)
        janas_expert_cache_reset_stats(c->wide);
}
