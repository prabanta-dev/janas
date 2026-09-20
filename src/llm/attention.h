/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * attention.h - causal grouped-query attention over an eight-bit KV cache.
 *
 * The keys and values of one layer are laid out n_head_kv x n_ctx x
 * head_dim as signed bytes, with one float scale per position and KV head
 * (n_head_kv x n_ctx): the value of a weight is its byte times the scale of
 * its position. Half the bytes of half precision, which at a long context is
 * what attention costs, and the error is well under what the model's own
 * quantization already carries. The positions of a KV head are contiguous. A
 * call attends a block of n tokens at positions pos0 .. pos0 + n - 1, whose
 * keys and values must already be in the cache. The result of a token does not
 * depend on the block it runs in: blocks and token-by-token calls agree bit for
 * bit.
 */
#ifndef JANAS_LLM_ATTENTION_H
#define JANAS_LLM_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

#include "common/pool.h"

#define JANAS_ATTN_MAX_BLOCK 64

/*
 * How the scores are computed. FLOAT reads the query as it is and sums in
 * floating point. INT16 reads it as sixteen-bit integers with a scale of its
 * own per token and head: the dot product is then an exact integer sum, which
 * the machine does sixteen products at a time instead of eight - about twice
 * as fast - at the cost of the query's quantization, a relative 3e-5. AUTO
 * takes INT16 when the instructions for it are there and the caller asked
 * for a context longer than the usual one (more than 16384 positions), where
 * attention is the greater part of a token; FLOAT otherwise, since below that
 * the exact query costs little and a conversation may never reach the far
 * end of the context anyway. The environment variable JANAS_ATTN (float,
 * int16, auto) overrides the lot.
 */
enum janas_attn_scores {
    JANAS_ATTN_AUTO = 0,
    JANAS_ATTN_FLOAT = 1,
    JANAS_ATTN_INT16 = 2
};

struct janas_attn;

/*
 * The bytes of the partial results an instance keeps for a context of n_ctx
 * positions: a softmax partial per chunk of positions, token of a block and
 * query head. They grow with the context like the KV cache itself - 1.08 GB
 * for Qwen3-Next-80B at 262,144 positions - so whoever sizes the memory has
 * to count them (janas_llm_open does).
 */
size_t janas_attn_part_bytes(uint32_t n_head, uint32_t n_head_kv,
                             uint32_t head_dim, uint32_t n_ctx);

/* Scratch for a pool of n_threads threads; NULL when out of memory. */
struct janas_attn *janas_attn_create(uint32_t n_head, uint32_t n_head_kv,
                                     uint32_t head_dim, uint32_t n_ctx,
                                     int n_threads, int scores);
void janas_attn_destroy(struct janas_attn *a);

/* The scale of the scores, 1 / sqrt(head_dim) until set (Gemma 4: 1, its
   q and k norms do the scaling). */
void janas_attn_set_scale(struct janas_attn *a, float scale);
/* A sliding window: token p sees positions p - window + 1 .. p only; 0 (the
   default) for all. The cache still holds every position. */
void janas_attn_set_window(struct janas_attn *a, uint32_t window);

/* What the scores are actually computed with: FLOAT or INT16. */
int janas_attn_scores_mode(const struct janas_attn *a);
/* 1 when the engine chose it rather than the caller. */
int janas_attn_scores_auto(const struct janas_attn *a);

/*
 * q: n x n_head x head_dim, token after token; k and v: the layer's cache,
 * ks and vs their scales; out: n x n_head x head_dim. Scores are scaled by
 * 1 / sqrt(head_dim) unless set otherwise.
 */
void janas_attn_run(struct janas_attn *a, struct janas_pool *pool,
                    const float *q, const int8_t *k, const float *ks,
                    const int8_t *v, const float *vs, uint32_t n, uint32_t pos0,
                    float *out);

#endif
