/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * deltanet.h - the Gated DeltaNet recurrence of Qwen3-Next's linear
 * attention layers, one head and one token at a time.
 *
 * The state of a head is a dim x dim matrix S (key index i, value index j),
 * stored transposed as M[j][i] = S[i][j] so that every row of M is a
 * contiguous vector over the key index. One step, as in llama.cpp's fused
 * gated_delta_net operator:
 *
 *   S     *= decay                       (decay = exp(g))
 *   d[j]   = (v[j] - sum_i S[i][j] k[i]) * beta
 *   S[i][j] += k[i] d[j]
 *   o[j]   = scale * sum_i S[i][j] q[i]
 *
 * Row j of M depends only on d[j], so a step is a single pass over the state,
 * and any range of rows can be stepped on its own (v and o then start at the
 * range's first row): a head can be split among threads.
 */
#ifndef JANAS_LLM_DELTANET_H
#define JANAS_LLM_DELTANET_H

#include <stddef.h>
#include <stdint.h>

/* One step on rows rows of the state M (dim wide, dim a multiple of 8). */
void janas_gdn_step(float *M, const float *q, const float *k, const float *v,
                    float decay, float beta, float scale, float *o, size_t dim,
                    size_t rows);

/* Portable version, for tests. Not bit-identical to the AVX2 one. */
void janas_gdn_step_ref(float *M, const float *q, const float *k,
                        const float *v, float decay, float beta, float scale,
                        float *o, size_t dim, size_t rows);

/*
 * The same on a state stored in IEEE half precision (uint16_t), for half the
 * memory traffic: each row is computed in float and rounded once, when it is
 * stored back. dim at most 256.
 */
void janas_gdn_step_h(uint16_t *M, const float *q, const float *k,
                      const float *v, float decay, float beta, float scale,
                      float *o, size_t dim, size_t rows);

/* "avx2" or "ref": the kernel janas_gdn_step uses on this machine. */
const char *janas_gdn_kernel_name(void);

#endif
