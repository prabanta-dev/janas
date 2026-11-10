/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * observe.h - observers of a model's forward pass, layer by layer: the
 * entrance gate (model_gate.c), the doubt probe (model_doubt.c), and
 * whatever else wants to see the states the CPU's layer loop computes
 * (a monitor of the engine at work). None is attached by default, and then
 * each hook point costs one test. A forward done whole on a GPU or an NPU
 * does not pass through these points (yet).
 *
 * Every callback may be NULL. They run on the forward's thread, except
 * ffn_pair, which runs on the compute pool's threads, one pair row each,
 * concurrently: it must only write what belongs to its pair row.
 */
#ifndef JANAS_LLM_OBSERVE_H
#define JANAS_LLM_OBSERVE_H

#include <stdint.h>

struct janas_llm_model;

struct janas_observer {
    void *ctx;
    /* layer l's feed-forward input, after its norm: n rows of d floats */
    void (*ffn_input)(void *ctx, struct janas_llm_model *m, uint32_t l,
                      const float *xn, uint32_t n, uint32_t d);
    /* a pair (a token through one expert, or through the dense
       feed-forward; expert n_expert for that or the shared one) after the
       activation, before anything changes it: its pair row, token in the
       pass, weight w, and ff floats of gate (NULL without one), up and
       h = act(gate) up */
    void (*ffn_pair)(void *ctx, struct janas_llm_model *m, uint32_t l,
                     uint32_t row, uint32_t tok, uint32_t expert, float w,
                     const float *gate, const float *up, const float *h,
                     uint32_t ff);
    /* the pairs of a pass done: their rows */
    void (*ffn_done)(void *ctx, struct janas_llm_model *m, uint32_t l,
                     const uint32_t *rows, uint32_t n_rows);
    /* the residual after layer l: n rows of d floats. Nonzero: the forward
       stops after this layer (no logits then) */
    int (*layer_out)(void *ctx, struct janas_llm_model *m, uint32_t l,
                     const float *x, uint32_t n, uint32_t d);
};

enum { JANAS_MAX_OBSERVERS = 4 };

/* Attach o (copied) to m; -1 when JANAS_MAX_OBSERVERS are attached. */
int janas_llm_model_observe(struct janas_llm_model *m,
                            const struct janas_observer *o);
/* Detach the observer whose ctx is ctx. */
void janas_llm_model_unobserve(struct janas_llm_model *m, void *ctx);

#endif
