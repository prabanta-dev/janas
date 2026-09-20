/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * pool.h - a persistent thread pool that runs one function on every thread.
 *
 * janas_pool_run() executes fn(arg, tid, n_threads) on all n_threads threads,
 * the caller included as tid 0, and returns when every thread has finished.
 * Idle workers spin briefly before sleeping, so back-to-back calls (one per
 * layer) do not pay a wake-up each time.
 */
#ifndef JANAS_COMMON_POOL_H
#define JANAS_COMMON_POOL_H

struct janas_pool;

typedef void (*janas_pool_fn)(void *arg, int tid, int n_threads);

/*
 * Creates a pool of n_threads threads (caller included). If cpus is not NULL
 * it lists n_threads CPU numbers: worker i (1 <= i < n_threads) is pinned to
 * cpus[i]. cpus[0] is the caller's slot, and the pool never pins the caller:
 * a thread driving two pools would otherwise end up on the CPU of the last one
 * created. Use janas_pin_current_thread() to pin the caller explicitly.
 * Returns NULL on failure.
 */
struct janas_pool *janas_pool_create(int n_threads, const int *cpus);
void janas_pool_run_named(struct janas_pool *p, janas_pool_fn fn, void *arg,
                          const char *name);
/* the name of the function is the name of the call site, for the trace */
#define janas_pool_run(p, fn, arg) janas_pool_run_named((p), (fn), (arg), #fn)
int janas_pool_size(const struct janas_pool *p);

/*
 * JANAS_POOLTRACE=1: one line per call site with its runs, the threads they
 * used and how much of that thread time was spent at the end of a run
 * waiting for the slowest. The time between runs - the caller working alone -
 * is not in the table: it is what the wall of the pass has more than the sum
 * of the wall column. Off, it costs one test of a static variable per run.
 */
void janas_pool_trace_report(const char *what, double wall);
void janas_pool_trace_reset(void);

/*
 * The next runs use only the first n threads (the caller included): fn sees
 * n_threads = n, the others sleep. Between runs only. For the threads to
 * be one per physical core, list those CPUs first at creation.
 */
void janas_pool_set_active(struct janas_pool *p, int n);
int janas_pool_active(const struct janas_pool *p);

/*
 * How long an idle worker spins (pause iterations) before sleeping on a
 * condition variable. Spinning answers a new job within a microsecond but
 * keeps the core busy: fine for compute threads fed every few microseconds,
 * a waste of package power for threads that wait on the disk.
 */
void janas_pool_set_spin(struct janas_pool *p, int iterations);
int janas_pool_get_spin(const struct janas_pool *p);
void janas_pool_destroy(struct janas_pool *p);

/* Pins the calling thread to one CPU. Returns 0 or -1. */
int janas_pin_current_thread(int cpu);

#endif
