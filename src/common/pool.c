/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * pool.c - persistent thread pool (see pool.h).
 */
#define _GNU_SOURCE
#include "pool.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__)
#include <immintrin.h>
#define cpu_relax() _mm_pause()
#else
#define cpu_relax() ((void)0)
#endif

#define SPIN_ITERATIONS 200000

/*
 * The end of a run is where twenty threads meet, and a single counter that
 * every one of them decrements has to travel from core to core once per
 * thread, in series. So each thread gets a cache line of its own and writes
 * in it the number of the run it has just finished; only the caller reads
 * them, and the writes never meet. An empty run at twenty threads went from
 * 2.75 us to 0.53 us (22 Sep 2026, job/tests/c/pool_cost.c).
 *
 * On the engine this is worth almost nothing - the transfers were hiding
 * under the caller's own share of the work - but it is what makes it certain
 * that dispatching is not where the time goes.
 */
#define LINE 64

struct done_flag {
    _Alignas(LINE) atomic_ullong run;
    double work; /* seconds inside fn, this thread; the trace only */
    char pad[LINE - sizeof(atomic_ullong) - sizeof(double)];
};

/*
 * JANAS_POOLTRACE=1: for every call site, how long its runs last, how many
 * threads they use and how much of that time is spent waiting at the end of
 * the run for the slowest thread. Reading it: a high idle share means the
 * work does not divide by the threads; the time between runs, which is the
 * caller working alone, is not here - it is the difference between the wall
 * of the whole pass and the sum of this column. Off it costs one test of a
 * static variable per run.
 */
struct site {
    janas_pool_fn fn;
    const char *name;
    double wall, work;
    unsigned long long runs, threads;
};

static struct site sites[32];
static int n_sites;
/* read by every worker, so atomic: they all decide the same thing, but a
   plain int written by twenty threads is still a race */
static atomic_int trace = -1;

static int tracing(void)
{
    int t = atomic_load_explicit(&trace, memory_order_relaxed);
    if (t < 0) {
        const char *e = getenv("JANAS_POOLTRACE");
        t = e && atoi(e) > 0;
        atomic_store_explicit(&trace, t, memory_order_relaxed);
    }
    return t;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

struct janas_pool {
    int n_threads;
    pthread_t *threads;
    int *cpus;

    janas_pool_fn fn;
    void *arg;

    /* every janas_pool_run publishes a new ticket: its number times 256
       plus its count of active threads, in one atomic value so that a
       worker never pairs one run with another run's count */
    atomic_ullong ticket;
    unsigned long long runs; /* written by the caller only */
    struct done_flag *done;  /* done[i].run: the last run thread i finished */
    atomic_int sleepers;     /* threads waiting on a condition variable */
    double work_prev;        /* the trace: sum of done[].work, last run */
    atomic_bool stop;
    atomic_int spin; /* pause iterations before a worker sleeps */
    int active;      /* threads that take part in the next runs */

    pthread_mutex_t lock;
    /*
     * Two of them, because a run must not wake a thread it has no work for.
     * Threads left out by janas_pool_set_active sleep on wake_idle, which is
     * broadcast only when the count grows again; the others wait on wake,
     * where a broadcast costs nothing while they are spinning. With one
     * condition variable a pool of twenty running twelve woke the other
     * eight at every call, and they queued on this mutex to find out they
     * were not wanted: 6.72 us a run instead of 1.82, measured 21 Sep 2026.
     */
    pthread_cond_t wake, wake_idle;
    int last_active; /* to notice that the count has grown */
};

struct worker_arg {
    struct janas_pool *p;
    int tid;
};

int janas_pin_current_thread(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) ? -1 : 0;
}

static void *worker(void *v)
{
    struct worker_arg wa = *(struct worker_arg *)v;
    struct janas_pool *p = wa.p;
    free(v);
    if (p->cpus)
        janas_pin_current_thread(p->cpus[wa.tid]);

    unsigned long long seen = 0;
    for (;;) {
        unsigned long long t;
        int spins = 0;
        /* a thread left out of the last run sleeps at once, and apart, on
           wake_idle: spinning would take resources from the busy thread
           sharing its core. Before the first run there is no such thing as
           left out - seen is zero - and a thread that took it for one would
           sleep where nothing wakes it. */
        int idle = seen != 0 && wa.tid >= (int)(seen & 255);
        while ((t = atomic_load_explicit(&p->ticket, memory_order_acquire)) ==
                   seen &&
               !atomic_load(&p->stop)) {
            if (!idle && ++spins < atomic_load_explicit(&p->spin,
                                                        memory_order_relaxed)) {
                cpu_relax();
                continue;
            }
            pthread_mutex_lock(&p->lock);
            /* published before reading the ticket again: a run that saw no
               sleeper had already stored its own */
            atomic_fetch_add(&p->sleepers, 1);
            while (atomic_load(&p->ticket) == seen && !atomic_load(&p->stop))
                pthread_cond_wait(idle ? &p->wake_idle : &p->wake, &p->lock);
            atomic_fetch_sub(&p->sleepers, 1);
            pthread_mutex_unlock(&p->lock);
        }
        if (atomic_load(&p->stop))
            return NULL;
        seen = t;
        int active = (int)(t & 255);
        if (wa.tid < active) {
            double w0 = tracing() ? now() : 0;
            p->fn(p->arg, wa.tid, active);
            if (tracing())
                p->done[wa.tid].work += now() - w0;
            atomic_store_explicit(&p->done[wa.tid].run, t >> 8,
                                  memory_order_release);
        }
    }
}

struct janas_pool *janas_pool_create(int n_threads, const int *cpus)
{
    if (n_threads < 1 || n_threads > 255)
        return NULL;
    struct janas_pool *p = calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->n_threads = n_threads;
    p->active = n_threads;
    atomic_init(&p->ticket, 0);
    atomic_init(&p->spin, SPIN_ITERATIONS);
    p->threads = calloc((size_t)n_threads, sizeof(pthread_t));
    if (posix_memalign((void **)&p->done, LINE,
                       (size_t)n_threads * sizeof(*p->done)) != 0)
        p->done = NULL;
    else
        memset(p->done, 0, (size_t)n_threads * sizeof(*p->done));
    if (cpus) {
        p->cpus = malloc((size_t)n_threads * sizeof(int));
        if (p->cpus)
            for (int i = 0; i < n_threads; i++)
                p->cpus[i] = cpus[i];
    }
    if (!p->threads || !p->done || (cpus && !p->cpus)) {
        free(p->threads);
        free(p->done);
        free(p->cpus);
        free(p);
        return NULL;
    }
    p->last_active = 0;
    pthread_mutex_init(&p->lock, NULL);
    pthread_cond_init(&p->wake, NULL);
    pthread_cond_init(&p->wake_idle, NULL);
    for (int i = 1; i < n_threads; i++) {
        struct worker_arg *wa = malloc(sizeof(*wa));
        if (!wa)
            abort();
        wa->p = p;
        wa->tid = i;
        if (pthread_create(&p->threads[i], NULL, worker, wa) != 0)
            abort();
    }
    return p;
}

static void pool_trace_run(struct janas_pool *p, janas_pool_fn fn,
                           const char *name, double wall, int active);

void janas_pool_run_named(struct janas_pool *p, janas_pool_fn fn, void *arg,
                          const char *name)
{
    p->fn = fn;
    p->arg = arg;
    int active = p->active;
    double t0 = tracing() ? now() : 0;
    /*
     * The lock and the broadcast are for the threads that are asleep, and
     * between two runs a few microseconds apart there are none: they are all
     * spinning on the ticket. Publishing the ticket first and then reading
     * sleepers, both sequentially consistent so that the two cannot be seen
     * out of order, makes the usual run a single store. A thread that decides
     * to sleep after that read reads the ticket again under the lock, and
     * finds it already changed.
     */
    int grew = active > p->last_active; /* threads wanted back: they sleep
                                           apart, on wake_idle */
    p->last_active = active;
    atomic_store_explicit(&p->ticket, ++p->runs << 8 | (unsigned)active,
                          memory_order_seq_cst);
    if (grew || atomic_load_explicit(&p->sleepers, memory_order_seq_cst)) {
        pthread_mutex_lock(&p->lock);
        pthread_cond_broadcast(&p->wake);
        if (grew)
            pthread_cond_broadcast(&p->wake_idle);
        pthread_mutex_unlock(&p->lock);
    }

    double w0 = tracing() ? now() : 0;
    fn(arg, 0, active);
    if (tracing())
        p->done[0].work += now() - w0;
    for (int i = 1; i < active; i++)
        while (atomic_load_explicit(&p->done[i].run, memory_order_acquire) !=
               p->runs)
            cpu_relax();
    if (tracing())
        pool_trace_run(p, fn, name, now() - t0, active);
}

/* The trace of one run, from the caller's thread: nothing here is shared. */
static void pool_trace_run(struct janas_pool *p, janas_pool_fn fn,
                           const char *name, double wall, int active)
{
    double w = 0;
    for (int i = 0; i < active; i++)
        w += p->done[i].work;
    int i = 0;
    while (i < n_sites && sites[i].fn != fn)
        i++;
    if (i == n_sites) {
        if (n_sites == 32)
            return;
        sites[n_sites].fn = fn;
        sites[n_sites].name = name;
        n_sites++;
    }
    sites[i].wall += wall;
    sites[i].work += w - p->work_prev;
    sites[i].threads += (unsigned long long)active;
    sites[i].runs++;
    p->work_prev = w;
}

void janas_pool_trace_reset(void)
{
    for (int i = 0; i < n_sites; i++) {
        sites[i].wall = sites[i].work = 0;
        sites[i].runs = sites[i].threads = 0;
    }
}

void janas_pool_trace_report(const char *what, double wall)
{
    if (!tracing())
        return;
    double tw = 0, tk = 0;
    for (int i = 0; i < n_sites; i++) {
        tw += sites[i].wall;
        tk += sites[i].work;
    }
    double off = 0; /* thread time offered: wall times the threads used */
    for (int i = 0; i < n_sites; i++)
        if (sites[i].runs)
            off += sites[i].wall * (double)sites[i].threads /
                   (double)sites[i].runs;
    fprintf(stderr, "pool, %s: %.0f ms in runs of %.0f ms", what, tw * 1e3,
            wall * 1e3);
    if (wall > 0)
        fprintf(stderr, " (the caller alone %.1f%%)", 100 * (1 - tw / wall));
    fprintf(stderr, "\n%-18s %9s %6s %9s %8s\n", "  site", "runs", "threads",
            "wall ms", "idle");
    for (int i = 0; i < n_sites; i++) {
        double o = sites[i].runs ? sites[i].wall * (double)sites[i].threads /
                                       (double)sites[i].runs
                                 : 0.0;
        fprintf(stderr, "  %-16s %9llu %6.1f %9.1f %7.1f%%\n", sites[i].name,
                sites[i].runs,
                sites[i].runs ? (double)sites[i].threads / (double)sites[i].runs
                              : 0,
                sites[i].wall * 1e3, o > 0 ? 100 * (o - sites[i].work) / o : 0);
    }
    fprintf(stderr, "  %-16s %9s %6s %9.1f %7.1f%%\n", "all", "", "", tw * 1e3,
            off > 0 ? 100 * (off - tk) / off : 0);
}

int janas_pool_get_spin(const struct janas_pool *p)
{
    return atomic_load_explicit(&((struct janas_pool *)p)->spin,
                                memory_order_relaxed);
}

void janas_pool_set_spin(struct janas_pool *p, int iterations)
{
    atomic_store_explicit(&p->spin, iterations, memory_order_relaxed);
}

int janas_pool_size(const struct janas_pool *p)
{
    return p->n_threads;
}

void janas_pool_set_active(struct janas_pool *p, int n)
{
    p->active = n < 1 ? 1 : (n > p->n_threads ? p->n_threads : n);
}

int janas_pool_active(const struct janas_pool *p)
{
    return p->active;
}

void janas_pool_destroy(struct janas_pool *p)
{
    if (!p)
        return;
    pthread_mutex_lock(&p->lock);
    atomic_store(&p->stop, true);
    pthread_cond_broadcast(&p->wake);
    pthread_cond_broadcast(&p->wake_idle);
    pthread_mutex_unlock(&p->lock);
    for (int i = 1; i < p->n_threads; i++)
        pthread_join(p->threads[i], NULL);
    pthread_mutex_destroy(&p->lock);
    pthread_cond_destroy(&p->wake);
    pthread_cond_destroy(&p->wake_idle);
    free(p->threads);
    free(p->done);
    free(p->cpus);
    free(p);
}
