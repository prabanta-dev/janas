/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * npu.h - an optional NPU (Intel's, through Level Zero) for the products of
 * blocks of tokens: y = x w^T with x m x k, w n x k, y m x n, all floats on
 * the caller's side.
 *
 * Level Zero is loaded at run time: without its loader, the NPU's driver or
 * an NPU, janas_npu_create returns NULL and the caller computes on the CPU.
 * Nothing here knows a model: Janas-LLM hands it matrices.
 *
 * The NPU runs graphs its driver compiles, of fixed shapes. A product is
 * compiled once with its weights inside (as a constant: integer weights
 * passed as an input are dequantized apart and run slower than the CPU),
 * as INT8 with a scale a row or as FP16; the driver keeps its own copy of
 * them, so the caller may free its own. x and y live in memory both sides
 * see (janas_npu_alloc), bound when the product is compiled; x is read as
 * floats and rounded to FP16 by the NPU, y written as floats. The NPU's
 * results are close to the CPU's, not the same bits (FP16 arithmetic).
 *
 * Measured on Meteor Lake (20 Oct 2026, job/tests/c/npu_probe.c): INT8
 * 2.7-3.6 TFLOP/s at 256 x 2048-6144, some 3.4 times the CPU's Q4_K blocks;
 * about 0.45 ms of fixed cost a run, so small products do not pay.
 *
 * One thread at a time per janas_npu.
 */
#ifndef JANAS_COMMON_NPU_H
#define JANAS_COMMON_NPU_H

#include <stddef.h>
#include <stdint.h>

struct janas_npu;
struct janas_npu_mm;

/* Opens the first NPU. NULL with a reason in err when there is none. */
struct janas_npu *janas_npu_create(char *err, size_t err_len);
/* Whether an NPU can be opened here (once a process, the answer kept): for
   a caller that has to plan memory before it opens one. */
int janas_npu_available(void);
/* Releases its products and memory too. */
void janas_npu_destroy(struct janas_npu *u);
const char *janas_npu_name(const struct janas_npu *u);

/* Memory the NPU reads and writes (host memory, 4096-aligned), released by
   janas_npu_free or janas_npu_destroy. NULL on failure. */
void *janas_npu_alloc(struct janas_npu *u, size_t bytes);
void janas_npu_free(struct janas_npu *u, void *p);

enum { JANAS_NPU_F16 = 0, JANAS_NPU_I8 = 1 };

/*
 * Quantizes n rows of k floats to INT8 with a scale a row (symmetric, -127
 * to 127; the scale in FP16, rounded up, so that q s is what the NPU
 * computes and no weight is clipped): q n x k, scale n values. For
 * JANAS_NPU_I8.
 */
void janas_npu_quant_rows(const float *w, size_t n, size_t k, int8_t *q,
                          uint16_t *scale);

/*
 * Compiles y = x w^T for x m x k and n rows of weights: JANAS_NPU_I8 with w
 * n x k int8 and scale n FP16 values, or JANAS_NPU_F16 with w n x k FP16
 * (scale NULL). x (m x k floats) and y (m x n floats) from janas_npu_alloc.
 * NULL with a reason in err on failure. It runs once before returning (the
 * first run after a compilation is slow), on whatever x holds.
 */
struct janas_npu_mm *janas_npu_mm_create(struct janas_npu *u, int m, int n,
                                         int k, int wtype, const void *w,
                                         const uint16_t *scale, float *x,
                                         float *y, char *err, size_t err_len);
void janas_npu_mm_destroy(struct janas_npu_mm *p);

/*
 * Any graph written with common/npu_ir.h (taken: freed here), compiled -
 * or loaded from the cache - with its arguments bound: args[i] is the
 * memory (janas_npu_alloc) of its i-th argument, inputs in the order of
 * their parameters then outputs in the order of their results, each
 * rows[i] x cols[i] floats. Runs as a product.
 */
struct npu_ir;
struct janas_npu_mm *janas_npu_graph_create(struct janas_npu *u,
                                            struct npu_ir *ir, int nargs,
                                            float *const *args, const int *rows,
                                            const int *cols, char *err,
                                            size_t err_len);

enum { JANAS_NPU_SILU = 0, JANAS_NPU_GELU_TANH = 1 };

/*
 * A feed-forward in one run of the NPU: y = down (act(gate x) * up x), x
 * m x d, gate and up ff x d, down d x ff, each INT8 with a scale a row
 * (janas_npu_quant_rows); act SiLU or the tanh form of GELU. One trip
 * instead of a product's for gate and up, the activation on the CPU and
 * another for down - and the intermediate m x ff never leaves the NPU.
 * Runs, waits and frees as a product (janas_npu_mm_begin, _finish,
 * _destroy); x and y as for a product.
 */
struct janas_npu_mm *janas_npu_ffn_create(struct janas_npu *u, int m, int d,
                                          int ff, int act, const int8_t *gq,
                                          const uint16_t *gs, const int8_t *uq,
                                          const uint16_t *us, const int8_t *dq,
                                          const uint16_t *ds, float *x,
                                          float *y, char *err, size_t err_len);

/* Its arguments bound to other memory (as at creation: inputs, then
   outputs; janas_npu_alloc), for the runs that follow; not while in
   flight. 0 or -1. */
int janas_npu_mm_rebind(struct janas_npu_mm *p, int nargs, float *const *args);

/*
 * A second binding of the same graph (no second copy of its weights): the
 * arguments are taken when a run is recorded, so a second list recorded
 * with others runs the graph on them - for two halves of a block, one in
 * flight while the CPU prepares the other. Then janas_npu_mm_begin_v(p,
 * 1) and _finish_v(p, 1); variant 0 is the first binding (begin, finish).
 */
int janas_npu_mm_bind_alt(struct janas_npu_mm *p, int nargs,
                          float *const *args);
int janas_npu_mm_begin_v(struct janas_npu_mm *p, int v);
int janas_npu_mm_finish_v(struct janas_npu_mm *p, int v);

/* Submits the product (x as it is now); the caller works meanwhile and
   janas_npu_mm_finish waits for y. Several may be in flight; they run in
   the order submitted. 0, or -1 with nothing submitted. */
int janas_npu_mm_begin(struct janas_npu_mm *p);
int janas_npu_mm_finish(struct janas_npu_mm *p);
/* begin and finish */
int janas_npu_mm_run(struct janas_npu_mm *p);

/*
 * Where compiled products are kept between runs: dir (created if missing),
 * a file a product named by the SHA-256 of its graph and weights and of the
 * driver's and compiler's versions, holding the weights too; NULL (the
 * default) keeps nothing. The driver's own cache ($HOME/.cache/
 * ze_intel_npu_cache, up to 4 GB) is never used. 0, or -1 if dir cannot
 * be made.
 */
int janas_npu_set_cache(struct janas_npu *u, const char *dir);

/*
 * While on, the NPU is not let sleep: a thread of the NPU's runs a tiny
 * product whenever it has been idle for 1 ms (the kernel suspends it after
 * 10, and waking it costs a product some 5 ms; its clock also falls after
 * longer pauses). For the passes that use it
 * with work of the CPU's between its products. 0, or -1 if it cannot be
 * kept awake. Products and janas_npu_keep_awake from one thread, as
 * everything here.
 */
int janas_npu_keep_awake(struct janas_npu *u, int on);

/* Products made so far, how many came from the cache, and the seconds spent
   compiling (or loading) them. */
void janas_npu_stats(const struct janas_npu *u, int *products, int *cached,
                     double *compile_seconds);

#endif
