/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * matvec.c - quantized matrix-vector product on a thread pool (see matvec.h).
 */
#include "matvec.h"

#include <stdatomic.h>
#include <stdlib.h>

/*
 * Rows are claimed in chunks: large enough that each thread streams a long
 * contiguous run (the hardware prefetcher restarts at every jump: 16-row
 * chunks reached 44 GB/s where 256-row chunks reach 52, on 6 P-cores), small
 * enough that every thread gets about four chunks on small matrices.
 *
 * The chunk is not fixed: it is a fraction of what is left. On cores of three
 * different speeds a fixed chunk leaves everyone waiting for the last one
 * claimed - with four chunks a thread that is half the tail of the run. Taking
 * 1/(2*threads) of the remaining rows keeps the first chunks long, where the
 * prefetcher earns its keep, and ends with chunks of CHUNK_MIN, where nobody
 * waits long. Chunks of different sizes still partition the rows exactly: a
 * fetch_add publishes its own size, so [old, old+chunk) is nobody else's.
 */
#define CHUNK_MIN 16
#define CHUNK_MAX 256

/* Claims the next chunk of [0, total): returns its first row, or total when
   there is nothing left, and writes the end in *end. */
static inline size_t claim(atomic_size_t *next, size_t total, size_t div,
                           size_t *end)
{
    *end = total;
    size_t seen = atomic_load_explicit(next, memory_order_relaxed);
    if (seen >= total)
        return total;
    size_t rem = total - seen, c = rem / div;
    if (c < CHUNK_MIN)
        c = CHUNK_MIN;
    else if (c > CHUNK_MAX)
        c = CHUNK_MAX;
    size_t g0 = atomic_fetch_add_explicit(next, c, memory_order_relaxed);
    if (g0 >= total)
        return total;
    *end = g0 + c < total ? g0 + c : total;
    return g0;
}

struct matvec_job {
    const struct janas_block_q4k *w;
    const struct janas_block_q8k *x;
    float *y;
    size_t rows;
    size_t nb;  /* blocks per row */
    size_t div; /* the chunk is 1/div of the rows still to claim */
    atomic_size_t next;
};

static void matvec_worker(void *arg, int tid, int n_threads)
{
    (void)tid;
    (void)n_threads;
    struct matvec_job *job = arg;
    for (;;) {
        size_t r1;
        size_t r0 = claim(&job->next, job->rows, job->div, &r1);
        if (r0 >= job->rows)
            break;
        for (size_t r = r0; r < r1; r++)
            job->y[r] = janas_q4k_dot(job->w + r * job->nb, job->x, job->nb);
    }
}

void janas_matvec_q4k(struct janas_pool *pool, const struct janas_block_q4k *w,
                      size_t rows, size_t cols, const struct janas_block_q8k *x,
                      float *y)
{
    struct matvec_job job = {
        .w = w, .x = x, .y = y, .rows = rows, .nb = cols / JANAS_QK};
    job.div = (size_t)janas_pool_active(pool) * 2;
    atomic_init(&job.next, 0);
    janas_pool_run(pool, matvec_worker, &job);
}

struct group_job {
    const struct janas_matvec_task *tasks;
    size_t n;
    size_t *first_row; /* first global row of each task, n + 1 entries */
    size_t div;        /* the chunk is 1/div of the rows still to claim */
    atomic_size_t next;
};

static void group_worker(void *arg, int tid, int n_threads)
{
    (void)tid;
    (void)n_threads;
    struct group_job *job = arg;
    size_t total = job->first_row[job->n];
    size_t t = 0;
    for (;;) {
        size_t g1;
        size_t g0 = claim(&job->next, total, job->div, &g1);
        if (g0 >= total)
            break;
        /* a thread's claims grow, whatever the size: the task index only
           goes forward */
        for (size_t g = g0; g < g1; g++) {
            while (job->first_row[t + 1] <= g)
                t++;
            const struct janas_matvec_task *k = &job->tasks[t];
            size_t r = g - job->first_row[t];
            size_t nb = k->cols / JANAS_QK;
            size_t row_bytes = nb * janas_qtype_block_size(k->type);
            const void *row = (const char *)k->w + r * row_bytes;
            size_t nv = k->n_vec ? k->n_vec : 1;
            size_t xs = k->x_stride ? k->x_stride : nb;
            size_t ys = k->y_stride ? k->y_stride : k->rows;
            if (k->type == JANAS_Q6_K_P) {
                size_t pb = nb * JANAS_Q6KP_PLANE;
                const uint8_t *p1 =
                    k->plane[0] ? (const uint8_t *)k->plane[0] + r * pb : NULL;
                const uint8_t *p0 =
                    k->plane[1] ? (const uint8_t *)k->plane[1] + r * pb : NULL;
                janas_q6kp_dot_multi(row, p1, p0, k->x, xs, nv, nb, k->y + r,
                                     ys);
            } else if (nv == 1)
                k->y[r] = janas_dot(k->type, row, k->x, nb);
            else
                janas_dot_multi(k->type, row, k->x, xs, nv, nb, k->y + r, ys);
        }
    }
}

void janas_matvec_q4k_group(struct janas_pool *pool,
                            const struct janas_matvec_task *tasks, size_t n)
{
    size_t first_row_small[65];
    size_t *first_row = first_row_small;
    if (n + 1 > sizeof(first_row_small) / sizeof(first_row_small[0])) {
        first_row = malloc((n + 1) * sizeof(size_t));
        if (!first_row)
            abort();
    }
    first_row[0] = 0;
    for (size_t i = 0; i < n; i++)
        first_row[i + 1] = first_row[i] + tasks[i].rows;

    struct group_job job = {.tasks = tasks, .n = n, .first_row = first_row};
    job.div = (size_t)janas_pool_active(pool) * 2;
    atomic_init(&job.next, 0);
    janas_pool_run(pool, group_worker, &job);
    if (first_row != first_row_small)
        free(first_row);
}
