/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu.h - an optional integrated GPU for quantized products (Vulkan).
 *
 * Vulkan is loaded at run time: without a driver or a suitable GPU,
 * janas_gpu_create returns NULL and the engine runs on the CPU alone. The
 * GPU reads the model's weights where they are (host memory imported, no
 * copy) and computes with the same arithmetic as the CPU kernels, so a
 * product split between the two gives the same bits as either alone.
 *
 * A product group is split by rows: janas_gpu_begin submits the first rows
 * of each task to the GPU (one submission for the group), the caller
 * computes the other rows on the CPU meanwhile, janas_gpu_finish waits and
 * writes the GPU's rows into the tasks' outputs. One group at a time.
 */
#ifndef JANAS_LLM_GPU_H
#define JANAS_LLM_GPU_H

#include <stddef.h>

#include "matvec.h"

struct janas_gpu;

struct janas_gpu *janas_gpu_create(char *err, size_t err_len);
void janas_gpu_destroy(struct janas_gpu *g);

/* Makes host memory readable by the GPU: p and bytes multiples of 4096.
   Returns 0 or -1. */
int janas_gpu_import(struct janas_gpu *g, const void *p, size_t bytes);

/*
 * Memory allocated by the GPU driver, mapped and cached for the host, that
 * the GPU reads with no import: bytes rounded up to 4096, NULL on failure,
 * released by janas_gpu_destroy. After writing it on the CPU, call
 * janas_gpu_flush before the GPU reads it.
 */
void *janas_gpu_alloc(struct janas_gpu *g, size_t bytes);
void janas_gpu_flush(struct janas_gpu *g, const void *p, size_t bytes);

/* Whether the GPU can compute this task (type supported, weights imported,
   sizes within its buffers). */
int janas_gpu_can(const struct janas_gpu *g, const struct janas_matvec_task *t);

/*
 * Submits rows [0, gpu_rows[i]) of each task (tasks the GPU cannot compute
 * must have gpu_rows[i] = 0). Returns 0, or -1 with nothing submitted.
 */
int janas_gpu_begin(struct janas_gpu *g, const struct janas_matvec_task *tasks,
                    const size_t *gpu_rows, size_t n);

/*
 * Experts on the GPU, bit for bit as the CPU computes them: for each expert
 * i, gate_i x and up_i x (ff rows), h = SiLU(gate) * up quantized to Q8_K
 * (the deterministic functions of vmath.h and quant.c), then down_i h (d
 * rows), stored at out + i * d by janas_gpu_finish. x is one Q8_K vector of
 * d values. One submission; the weights must be in shared or copied
 * regions. Returns 0, or -1 with nothing submitted.
 */
struct janas_gpu_expert {
    const void *gate, *up, *down;
    int gate_type, up_type, down_type; /* JANAS_Q4_K or JANAS_Q6_K */
};
int janas_gpu_ffn_begin(struct janas_gpu *g, const struct janas_gpu_expert *e,
                        size_t n, const struct janas_block_q8k *x, size_t d,
                        size_t ff, float *out);

/* Waits for the submitted rows and stores them. Returns 0 or -1. Sets
 *waited to whether the GPU was still running when called. */
int janas_gpu_finish(struct janas_gpu *g, int *waited);

/*
 * y = w x for a product group, the rows of each task split between the GPU
 * and the pool: the GPU takes the first rows, a share that adapts per
 * product shape so that both sides finish together. With g NULL, blocks of
 * fewer vectors than JANAS_GPU_MIN_N (default 2), or after a GPU failure,
 * the pool computes everything. The results are the same in every case.
 */
void janas_gpu_matvec_group(struct janas_gpu *g, struct janas_pool *pool,
                            const struct janas_matvec_task *tasks, size_t n);

/* Keeps the GPU out of its sleep states (an empty dispatch every 0.5 ms,
   JANAS_GPU_KEEPALIVE_US) while on: for the passes that use it. */
void janas_gpu_keep_awake(struct janas_gpu *g, int on);

/* The device's name, and the subgroup size the shaders run with. */
const char *janas_gpu_name(const struct janas_gpu *g);
unsigned janas_gpu_subgroup(const struct janas_gpu *g);

#endif
