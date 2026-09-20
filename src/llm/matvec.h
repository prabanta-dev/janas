/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * matvec.h - quantized matrix-vector product on a thread pool.
 */
#ifndef JANAS_LLM_MATVEC_H
#define JANAS_LLM_MATVEC_H

#include <stddef.h>

#include "common/pool.h"
#include "quant.h"

/*
 * y[r] = dot(row r of w, x) for r in [0, rows). w holds rows * cols / JANAS_QK
 * Q4_K blocks, row-major; x holds cols / JANAS_QK Q8_K blocks. cols must be a
 * multiple of JANAS_QK. Rows are handed out in chunks through a shared
 * counter, so slower cores (E-cores) take fewer of them instead of holding the
 * whole product back.
 */
void janas_matvec_q4k(struct janas_pool *pool, const struct janas_block_q4k *w,
                      size_t rows, size_t cols, const struct janas_block_q8k *x,
                      float *y);

/*
 * One product of a group: y = w * x, w has rows x cols weights of the given
 * type (enum janas_qtype). With n_vec > 1 the same weights multiply n_vec
 * vectors - the tokens of a block - and every weight row is read from memory
 * once for all of them: vector v starts at x + v * x_stride (in blocks) and
 * its result at y + v * y_stride. Zero means one vector, x_stride = cols /
 * JANAS_QK and y_stride = rows, so a four-field initializer still works.
 */
struct janas_matvec_task {
    int type;
    const void *w;
    /* JANAS_Q6_K_P only: the bare planes 1 and 0 of the same rows, NULL when
       that level is not in memory (see quant.h). */
    const void *plane[2];
    const struct janas_block_q8k *x;
    float *y;
    size_t rows;
    size_t cols;
    size_t n_vec;
    size_t x_stride;
    size_t y_stride;
};

/*
 * Runs n independent products in a single pool dispatch, handing out rows of
 * all of them through one counter. For the experts of a MoE layer: ten
 * experts times gate and up are twenty small products, and dispatching each
 * one separately would pay the pool synchronisation twenty times.
 */
void janas_matvec_q4k_group(struct janas_pool *pool,
                            const struct janas_matvec_task *tasks, size_t n);

#endif
