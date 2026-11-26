/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_drv.h - what the GPUs reached through their makers' own APIs share,
 * behind gpu_cuda.h's interface: NVIDIA's CUDA and AMD's HIP (gpu_cuda.c),
 * Apple's Metal (gpu_metal.m). The kernels are one file for all of them
 * (cuda/matvec.cu); so is the way their inputs are laid out: the
 * activations as the kernels read them (a Q8_K block in 320 bytes), the
 * weights' regions the GPU was given, the kernel of a type.
 */
#ifndef JANAS_LLM_GPU_DRV_H
#define JANAS_LLM_GPU_DRV_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "quant.h"

#define GPU_BLOCK 320       /* a Q8_K block as the kernels read it */
#define X_STAGE (64u << 20) /* the activations of a group */
#define Y_STAGE (32u << 20) /* the rows a group's products give */
#define MAX_PENDING 64
#define MAX_XCACHE 16

/* host memory the GPU was given: where it is on the host, and the GPU's
   handle of it (a device address, or a buffer) with the offset of the
   host's first byte in it */
struct drv_region {
    const uint8_t *host;
    size_t bytes;
    uint64_t dev;
    size_t dev_off;
    int mapped; /* the GPU reads the host's memory in place */
};

/* a product's rows waiting to be copied where the caller wants them */
struct drv_pending {
    float *y;
    size_t rows, nv, y_stride, off;
};

struct drv_regions {
    struct drv_region *rg;
    size_t n, cap;
};

static inline struct drv_region *drv_add_region(struct drv_regions *r)
{
    if (r->n == r->cap) {
        size_t cap = r->cap ? 2 * r->cap : 64;
        struct drv_region *n = realloc(r->rg, cap * sizeof(*n));
        if (!n)
            return NULL;
        r->rg = n;
        r->cap = cap;
    }
    return &r->rg[r->n++];
}

/* the region holding [p, p + bytes), or NULL */
static inline const struct drv_region *drv_find(const struct drv_regions *r,
                                                const void *p, size_t bytes)
{
    const uint8_t *q = p;
    for (size_t i = 0; i < r->n; i++)
        if (q >= r->rg[i].host && q + bytes <= r->rg[i].host + r->rg[i].bytes)
            return &r->rg[i];
    return NULL;
}

/* b32_matvec's kind of a type, -1 if it is not one of its (as gpu.c's) */
static inline int drv_b32_kind(int type)
{
    return type == JANAS_Q4_0     ? 0
           : type == JANAS_IQ4_NL ? 1
           : type == JANAS_Q4_1   ? 2
           : type == JANAS_Q5_1   ? 3
           : type == JANAS_MXFP4  ? 4
           : type == JANAS_Q5_0   ? 5
           : type == JANAS_Q2_0   ? 6
           : type == JANAS_NVFP4  ? 7
                                  : -1;
}

/* the types that read the activations in blocks of 32 */
static inline int drv_in_32(int type)
{
    return type == JANAS_Q8_0 || drv_b32_kind(type) >= 0;
}

/* the kernel of a type, for the reports and the lookups */
static inline const char *drv_kernel_name(int type)
{
    return type == JANAS_Q4_K        ? "q4k_matvec"
           : type == JANAS_Q6_K      ? "q6k_matvec"
           : type == JANAS_Q5_K      ? "q5k_matvec"
           : type == JANAS_Q3_K      ? "q3k_matvec"
           : type == JANAS_Q8_0      ? "q80_matvec"
           : type == JANAS_IQ4_XS    ? "iq4xs_matvec"
           : drv_b32_kind(type) >= 0 ? "b32_matvec"
                                     : NULL;
}

/* Q8_K vectors as the kernels read them, as gpu.c's convert_x: the quants,
   the sums, the scale; for the weights in blocks of 32 (k32) the same 256
   in blocks of 32 with a scale each */
static inline void drv_convert_x(uint8_t *dst, const struct janas_block_q8k *x,
                                 size_t xs, size_t nv, size_t nb, int k32)
{
    for (size_t v = 0; v < nv; v++)
        for (size_t b = 0; b < nb; b++, dst += GPU_BLOCK) {
            const struct janas_block_q8k *s = x + v * xs + b;
            memset(dst, 0, GPU_BLOCK);
            if (k32) {
                memcpy(dst, s->qs32, JANAS_QK);
                memcpy(dst + JANAS_QK, s->bsums32, sizeof(s->bsums32));
                memcpy(dst + JANAS_QK + sizeof(s->bsums32), s->d32,
                       sizeof(s->d32));
            } else {
                memcpy(dst, s->qs, JANAS_QK);
                memcpy(dst + JANAS_QK, s->bsums, sizeof(s->bsums));
                memcpy(dst + JANAS_QK + sizeof(s->bsums), &s->d, sizeof(float));
            }
        }
}

#endif
