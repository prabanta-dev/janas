/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_cuda.h - an NVIDIA card through CUDA's driver API, loaded at run time
 * (libcuda.so.1, nvcuda.dll: only NVIDIA's driver is needed), behind the
 * same struct janas_gpu as Vulkan's (gpu.c passes its calls here when the
 * GPU it opened is this one: JANAS_GPU_API=cuda).
 *
 * The products split by rows as Vulkan's are (janas_gpu_matvec_group), on
 * the kernels of src/llm/cuda/ (PTX, the same arithmetic as the CPU's, bit
 * for bit). The weights go into the card's own memory when imported; what
 * a kernel has not been written for yet stays on the CPU (janas_cuda_can
 * says no), as on Vulkan for a type with no shader.
 */
#ifndef JANAS_LLM_GPU_CUDA_H
#define JANAS_LLM_GPU_CUDA_H

#include <stddef.h>

#include "matvec.h"

struct janas_cuda;

/* The first NVIDIA card of compute capability 6.1 and later (dp4a), or NULL
   with the reason in err (no driver, no card, the kernels not loaded). */
struct janas_cuda *janas_cuda_create(char *err, size_t err_len);
/* The first AMD card through HIP's runtime (libamdhip64, amdhip64_6.dll:
   ROCm, or AMD's HIP SDK on Windows), the kernels from the bundle of AMD's
   code objects; NULL with the reason in err. */
struct janas_cuda *janas_hip_create(char *err, size_t err_len);
void janas_cuda_destroy(struct janas_cuda *c);

const char *janas_cuda_name(const struct janas_cuda *c);
/* One line for a report: the card, its compute capability, the driver. */
const char *janas_cuda_about(const struct janas_cuda *c);
/* The card's memory, free and in all. */
void janas_cuda_card(const struct janas_cuda *c, char *buf, size_t len);
/* Bytes of the card's memory still free for Janas (85% of what was free at
   the start, less what it took). */
size_t janas_cuda_local_room(const struct janas_cuda *c);

/* Host memory copied into the card's: 0 or -1 (no room). */
int janas_cuda_import(struct janas_cuda *c, const void *p, size_t bytes);
/* Host memory the card reads in place (pinned, mapped), NULL on failure. */
void *janas_cuda_alloc(struct janas_cuda *c, size_t bytes);
/* Imported memory made the same in both places: the host's copied to the
   card (to_device 1) or the card's back to the host (0): 0, or -1. */
int janas_cuda_sync(struct janas_cuda *c, void *p, size_t bytes, int to_device);

int janas_cuda_can(const struct janas_cuda *c,
                   const struct janas_matvec_task *t);
/* As janas_gpu_begin and janas_gpu_finish (gpu.h). */
int janas_cuda_begin(struct janas_cuda *c,
                     const struct janas_matvec_task *tasks,
                     const size_t *gpu_rows, size_t n);
int janas_cuda_finish(struct janas_cuda *c, int *waited);
/* The kernel of the last product, as "q4k_matvec (CUDA)", "" if none. */
const char *janas_cuda_last(const struct janas_cuda *c);

#endif
