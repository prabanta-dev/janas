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

/*
 * Q6_K in the GPU's aligned copy (JANAS_GPU_REPACK=1 with the whole token,
 * model_gpu.c): each 210-byte block padded to 224 - ql 128, qh 64, scales
 * 16, d at byte 208 - so that every part starts on 16 bytes. Only the GPU
 * reads it, only for single vectors; a Q6_K_P block with its three planes
 * becomes the Q6_K block it came from.
 */
#define JANAS_Q6_K_A 216
#define JANAS_Q6KA_BLOCK 224

/*
 * The single token's copies in tiles of JANAS_TILE rows (JANAS_GPU_REPACK=2:
 * the Q6_K and the Q4_K matrices with rows of 32 blocks and more; 3: every
 * Q4_K), block b of a tile's rows side by side, so that the subgroups (a
 * row each) read neighbouring memory at the same time: faster for long
 * rows (Qwen3-4B's down matrices), slower for short ones on the
 * development laptop's integrated GPU. Block b of row r is block ((r / T)
 * nb + b) T + r % T; rows padded to a whole tile.
 */
#define JANAS_TILE 64
#define JANAS_Q4_K_T 217
#define JANAS_Q6_K_AT 218

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

/*
 * The fast prompt's product of weights left as they are, f32 (bf16 0) or
 * bf16 (1), rows x cols, with n vectors of cols floats in x: y[v * rows + r]
 * = scale x row r . vector v, on the GPU alone, waited for. Only for n of
 * JANAS_GPU_FAST_N and more, cols a multiple of 16, and w, x and y in memory
 * the GPU holds; 0, or -1 with nothing written (the caller computes it).
 */
int janas_gpu_dense_mm(struct janas_gpu *g, int bf16, const void *w,
                       const float *x, float *y, size_t rows, size_t cols,
                       size_t n, float scale);

/*
 * Whether the fast prompt is on (unless JANAS_GPU_FAST_PROMPT=0): blocks of
 * JANAS_GPU_FAST_N vectors and more of the types it has shaders for go to
 * the GPU whole, rounded group by group - close to the CPU's results, not
 * bit for bit - and, so that a prompt always gives the same results, every
 * such block goes to the GPU, whatever the tuner would choose.
 */
/* 64: a mixture's experts get some 16 tokens each of a block of 256, too
   few to fill a tile of 64 - forced whole onto the GPU, Qwen3-30B-A3B read
   a prompt at 84 tokens/s instead of 97 (9 Oct 2026) */
#define JANAS_GPU_FAST_N 64
int janas_gpu_fast_prompt(void);
/* For the passes that follow: only the fast prompt's products on the GPU,
   the rest on the CPU (a pass the tuner gave the CPU alone). */
void janas_gpu_set_only_fast(struct janas_gpu *g, int on);

/* Keeps the GPU out of its sleep states (an empty dispatch every 0.5 ms,
   JANAS_GPU_KEEPALIVE_US) while on: for the passes that use it. */
void janas_gpu_keep_awake(struct janas_gpu *g, int on);

/* The device's name, and the subgroup size the shaders run with. */
const char *janas_gpu_name(const struct janas_gpu *g);
unsigned janas_gpu_subgroup(const struct janas_gpu *g);
/* The fewest vectors a product needs to be given to the GPU: 1 on a
   discrete GPU (single tokens too), 2 on an integrated one;
   JANAS_GPU_MIN_N sets it. */
size_t janas_gpu_min_n(const struct janas_gpu *g);

#endif
