/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * expert_cache.c - RAM cache of routed experts over a JNS file (see
 * expert_cache.h).
 */
#define _GNU_SOURCE
#include "expert_cache.h"

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

struct read_job {
    uint64_t offset;
    uint64_t bytes;
    uint8_t *dst;
};

struct janas_expert_cache {
    const struct janas_jns *j;
    int level; /* 1 to 3: how much of a slot is read (see the header) */
    int fd;    /* O_DIRECT, independent of the jns descriptor */
    struct janas_pool *io;

    uint64_t slot_bytes;
    size_t n_slots;
    uint8_t *arena;

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
    size_t filled;       /* slots that hold an expert */
    double last_request; /* when the caller last asked for experts */
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
        uint32_t nl = c->j->h.n_layer;
        for (uint32_t d = 0; d < nl; d++) {
            uint32_t l = (c->cur_layer + nl - d) % nl;
            for (int32_t s = c->ltail[l]; s != NONE; s = c->lprev[s])
                if (c->stamp[s] != c->request)
                    return s;
        }
    }
    for (int32_t s = c->tail; s != NONE; s = c->prev[s])
        if (c->stamp[s] != c->request)
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

struct janas_expert_cache *janas_expert_cache_create(const struct janas_jns *j,
                                                     uint64_t budget_bytes,
                                                     int level,
                                                     struct janas_pool *io,
                                                     char *err, size_t err_len)
{
    if (level < 1 || level > 3) {
        snprintf(err, err_len, "expert level %d out of range", level);
        return NULL;
    }
    uint64_t slot_bytes = 0;
    for (uint32_t l = 0; l < j->h.n_layer; l++)
        if (j->layers[l].level_bytes[level - 1] > slot_bytes)
            slot_bytes = j->layers[l].level_bytes[level - 1];
    size_t n_slots = (size_t)(budget_bytes / slot_bytes);
    size_t n_keys = (size_t)j->h.n_layer * j->h.n_expert;
    if (n_slots > n_keys)
        n_slots = n_keys;
    if (n_slots < j->h.n_expert_used) {
        snprintf(err, err_len, "budget below one layer of experts");
        return NULL;
    }

    struct janas_expert_cache *c = calloc(1, sizeof(*c));
    if (!c)
        return NULL;
    c->j = j;
    c->level = level;
    c->io = io;
    c->slot_bytes = slot_bytes;
    c->n_slots = n_slots;
    c->head = c->tail = NONE;

    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", j->fd);
    c->fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    size_t arena_bytes = n_slots * slot_bytes;
    c->arena = aligned_alloc(JANAS_JNS_ALIGN, arena_bytes);
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
    c->lhead = malloc(((size_t)j->h.n_layer + 1) * sizeof(int32_t));
    c->ltail = malloc(((size_t)j->h.n_layer + 1) * sizeof(int32_t));
    if (c->fd < 0 || !c->arena || !c->where || !c->key || !c->prev ||
        !c->next || !c->stamp || !c->jobs || !c->lprev || !c->lnext ||
        !c->lhead || !c->ltail) {
        snprintf(err, err_len, "cannot allocate %zu slots of %llu bytes",
                 n_slots, (unsigned long long)slot_bytes);
        janas_expert_cache_destroy(c);
        return NULL;
    }
    madvise(c->arena, arena_bytes, MADV_HUGEPAGE);
    pthread_mutex_init(&c->api_lock, NULL);
    pthread_mutex_init(&c->lock, NULL);
    pthread_cond_init(&c->cond, NULL);
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
    for (uint32_t l = 0; l <= j->h.n_layer; l++)
        c->lhead[l] = c->ltail[l] = NONE;
    for (size_t s = 0; s < n_slots; s++) {
        c->key[s] = NONE;
        push_front(c, (int32_t)s);
    }
    c->request = 1; /* stamps start at 0: no slot is in use */
    return c;
}

void janas_expert_cache_destroy(struct janas_expert_cache *c)
{
    if (!c)
        return;
    if (c->has_warmer) { /* it holds the lock only for one batch */
        pthread_mutex_lock(&c->api_lock);
        c->warm_stop = 1;
        pthread_mutex_unlock(&c->api_lock);
        pthread_join(c->warmer, NULL);
        c->has_warmer = 0;
    }
    free(c->warm_counts);
    free(c->warm_order);
    if (c->has_coordinator) {
        pthread_mutex_lock(&c->lock);
        while (c->pending)
            pthread_cond_wait(&c->cond, &c->lock);
        c->stop = 1;
        pthread_cond_broadcast(&c->cond);
        pthread_mutex_unlock(&c->lock);
        pthread_join(c->coordinator, NULL);
        pthread_mutex_destroy(&c->lock);
        pthread_cond_destroy(&c->cond);
    }
    janas_uring_destroy(c->ring);
    pthread_mutex_destroy(&c->api_lock);
    if (c->fd >= 0)
        close(c->fd);
    free(c->arena);
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

static const uint32_t *sort_counts;

static int by_count(const void *a, const void *b)
{
    uint32_t x = sort_counts[*(const uint32_t *)a];
    uint32_t y = sort_counts[*(const uint32_t *)b];
    return (x > y) - (x < y); /* ascending: the most frequent loaded last */
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
    size_t n_keys = (size_t)c->j->h.n_layer * c->j->h.n_expert;
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
    sort_counts = c->warm_counts;
    qsort(c->warm_order, n, sizeof(uint32_t), by_count);
    /* by_count puts the least used first: read the most used first here */
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
    size_t n_keys = (size_t)c->j->h.n_layer * c->j->h.n_expert;
    uint32_t *order = malloc(n_keys * sizeof(uint32_t));
    if (!order)
        return -1;
    size_t n = 0;
    for (size_t k = 0; k < n_keys; k++)
        if (counts[k])
            order[n++] = (uint32_t)k;
    sort_counts = counts;
    qsort(order, n, sizeof(uint32_t), by_count);
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

int janas_expert_cache_begin(struct janas_expert_cache *c, uint32_t layer,
                             const uint32_t *ids, size_t n,
                             const uint8_t **slots, uint8_t *ready)
{
    if (layer >= c->j->h.n_layer || n > c->n_slots || n > 256 || c->outstanding)
        return -1;
    /* held until finish: the background filling waits for its turn */
    pthread_mutex_lock(&c->api_lock);
    c->last_request = now();
    c->request++;
    /* a layer before the last one asked: a new pass */
    if (layer < c->cur_layer) {
        c->last_pass_asked = c->pass_asked;
        c->pass_asked = 0;
    }
    c->cur_layer = layer;
    c->pass_asked += n;
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
            touch(c, s[i]);
            c->st.hits++;
        }
    }
    int queued = 0;
    for (size_t i = 0; i < n; i++)
        if (s[i] == NONE) {
            int32_t w = c->where[base + ids[i]];
            if (w != NONE) { /* the same id twice in one request */
                s[i] = w;
                continue;
            }
            s[i] = load(c, layer, ids[i]);
            c->st.misses++;
            queued++;
        }
    c->st.requests += n;
    for (size_t i = 0; i < n; i++)
        slots[i] = c->arena + (size_t)s[i] * c->slot_bytes;
    c->outstanding = 1;
    if (start_jobs(c) != 0)
        c->batch_rc = -1; /* finish() reports it, and waits all the same */
    return queued;
}

int janas_expert_cache_finish(struct janas_expert_cache *c)
{
    double t0 = now();
    int rc = wait_jobs(c);
    if (c->via_ring && c->batch_rc) /* the submission itself failed */
        rc = -1;
    c->batch_rc = 0;
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

void janas_expert_cache_stats(const struct janas_expert_cache *c,
                              struct janas_expert_cache_stats *s)
{
    struct janas_expert_cache *m = (struct janas_expert_cache *)c;
    pthread_mutex_lock(&m->api_lock);
    *s = c->st;
    pthread_mutex_unlock(&m->api_lock);
}

void janas_expert_cache_reset_stats(struct janas_expert_cache *c)
{
    memset(&c->st, 0, sizeof(c->st));
}
