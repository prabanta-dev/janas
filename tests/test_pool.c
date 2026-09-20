/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_pool.c - the thread pool: every run reaches exactly the active
 * threads, each once, with n_threads equal to their count, through changes
 * of the active count in both directions.
 */
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "common/pool.h"

#define N 8

static atomic_int calls[N];
static atomic_int bad_n;

struct job {
    int expect_n;
};

static void fn(void *arg, int tid, int n_threads)
{
    const struct job *j = arg;
    if (n_threads != j->expect_n || tid < 0 || tid >= n_threads)
        atomic_fetch_add(&bad_n, 1);
    else
        atomic_fetch_add(&calls[tid], 1);
}

int main(void)
{
    /* a test that deadlocks must fail, not hold the suite for ever */
    alarm(600);
    struct janas_pool *p = janas_pool_create(N, NULL);
    int failures = 0;
    const int seq[] = {8, 3, 1, 5, 8, 2, 8, 1, 7};
    for (size_t s = 0; s < sizeof(seq) / sizeof(seq[0]); s++) {
        int a = seq[s];
        janas_pool_set_active(p, a);
        for (int t = 0; t < N; t++)
            atomic_store(&calls[t], 0);
        struct job j = {.expect_n = a};
        for (int r = 0; r < 1000; r++)
            janas_pool_run(p, fn, &j);
        for (int t = 0; t < N; t++) {
            int want = t < a ? 1000 : 0;
            if (atomic_load(&calls[t]) != want) {
                printf("active %d: thread %d ran %d times, not %d\n", a, t,
                       atomic_load(&calls[t]), want);
                failures++;
            }
        }
    }
    if (atomic_load(&bad_n)) {
        printf("%d calls with a wrong thread count\n", atomic_load(&bad_n));
        failures++;
    }
    janas_pool_destroy(p);
    printf("test_pool: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
