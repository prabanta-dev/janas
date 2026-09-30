/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_impl.h - the insides of the GPU (private to gpu.c and gpu_token.c):
 * the Vulkan functions it loads, its regions and stages, and the helpers
 * that record a product for a command buffer. Only with the GPU in the
 * build.
 */
#ifndef JANAS_LLM_GPU_IMPL_H
#define JANAS_LLM_GPU_IMPL_H

#include "gpu.h"

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)

/* The push constants a shader may take: every Vulkan device gives 128. */
#define JANAS_G_PUSH_BYTES 128
/* the kinds of product shader (gpu.c's prod_index) */
#define N_PROD 12

#define VK_NO_PROTOTYPES
#include <dlfcn.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define MAX_REGIONS 32
#define MAX_PENDING 64
#define MAX_XCACHE 8
#define X_STAGE_BYTES ((size_t)8 << 20)
#define Y_STAGE_BYTES ((size_t)32 << 20)
#define ROWS_PER_GROUP 64 /* local_size_x of the shader */
#define NV_PER_GROUP 8    /* its NV */
#define GPU_BLOCK 320     /* a Q8_K block in the shader's layout */
#define SHARE_SLOTS 64
#define MAX_GROUP 64

/* instance-level functions, then device-level ones */
#define VK_INSTANCE_FUNCS(X)                                                   \
    X(vkDestroyInstance)                                                       \
    X(vkEnumeratePhysicalDevices)                                              \
    X(vkGetPhysicalDeviceProperties)                                           \
    X(vkGetPhysicalDeviceProperties2)                                          \
    X(vkGetPhysicalDeviceFeatures2)                                            \
    X(vkGetPhysicalDeviceMemoryProperties)                                     \
    X(vkGetPhysicalDeviceQueueFamilyProperties)                                \
    X(vkEnumerateDeviceExtensionProperties)                                    \
    X(vkCreateDevice)                                                          \
    X(vkGetDeviceProcAddr)
#define VK_DEVICE_FUNCS(X)                                                     \
    X(vkDestroyDevice)                                                         \
    X(vkGetDeviceQueue)                                                        \
    X(vkCreateBuffer)                                                          \
    X(vkDestroyBuffer)                                                         \
    X(vkGetBufferMemoryRequirements)                                           \
    X(vkAllocateMemory)                                                        \
    X(vkFreeMemory)                                                            \
    X(vkBindBufferMemory)                                                      \
    X(vkMapMemory)                                                             \
    X(vkInvalidateMappedMemoryRanges)                                          \
    X(vkFlushMappedMemoryRanges)                                               \
    X(vkGetBufferDeviceAddress)                                                \
    X(vkGetMemoryHostPointerPropertiesEXT)                                     \
    X(vkCreateShaderModule)                                                    \
    X(vkDestroyShaderModule)                                                   \
    X(vkCreatePipelineLayout)                                                  \
    X(vkDestroyPipelineLayout)                                                 \
    X(vkCreateComputePipelines)                                                \
    X(vkDestroyPipeline)                                                       \
    X(vkCreateCommandPool)                                                     \
    X(vkDestroyCommandPool)                                                    \
    X(vkAllocateCommandBuffers)                                                \
    X(vkFreeCommandBuffers)                                                    \
    X(vkResetCommandBuffer)                                                    \
    X(vkBeginCommandBuffer)                                                    \
    X(vkEndCommandBuffer)                                                      \
    X(vkCmdBindPipeline)                                                       \
    X(vkCmdPushConstants)                                                      \
    X(vkCmdDispatch)                                                           \
    X(vkCmdPipelineBarrier)                                                    \
    X(vkCmdCopyBuffer)                                                         \
    X(vkWaitForFences)                                                         \
    X(vkCreateFence)                                                           \
    X(vkDestroyFence)                                                          \
    X(vkResetFences)                                                           \
    X(vkGetFenceStatus)                                                        \
    X(vkQueueSubmit)                                                           \
    X(vkCreateQueryPool)                                                       \
    X(vkDestroyQueryPool)                                                      \
    X(vkCmdResetQueryPool)                                                     \
    X(vkCmdWriteTimestamp)                                                     \
    X(vkGetQueryPoolResults)

struct region {
    const uint8_t *host;
    size_t bytes;
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceAddress addr;
    int coherent; /* allocated regions: else flushed after host writes */
};

struct stage {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *map;
    VkDeviceAddress addr;
    size_t bytes;
    int coherent; /* else invalidated before the host reads it */
};

struct pending {
    float *y;        /* the task's output */
    size_t rows, nv; /* rows computed by the GPU, vectors */
    size_t y_stride; /* the task's */
    size_t off;      /* in the Y stage */
};

/* The push constants of the shaders (level and the bare planes: Q6_K_P
   only). */
struct push {
    uint32_t w[2], x[2], y[2];
    uint32_t rows, nb, nv, xs, ys, level;
    uint32_t p1[2], p0[2];
};

struct janas_gpu {
    void *lib;
    PFN_vkGetInstanceProcAddr gipa;
#define DECL(f) PFN_##f f;
    VK_INSTANCE_FUNCS(DECL)
    VK_DEVICE_FUNCS(DECL)
#undef DECL
    VkInstance inst;
    VkPhysicalDevice pd;
    VkDevice dev;
    VkQueue queue;
    VkShaderModule sm, sm6, sm6p, sm5, sm80, sm32[4], smx4, sm3, smi3, sma;
    VkPipelineLayout pl;
    VkPipeline pipe, pipe6, pipe6p, pipe5, pipe80, pipea; /* Q4_K, Q6_K,
                                                             Q6_K_P, Q5_K,
                                                             Q8_0, experts'
                                                             activation */
    /* blocks of 32: Q4_0, IQ4_NL, Q4_1, Q5_1 (b32_matvec's KIND) */
    VkPipeline pipe32[4];
    VkPipeline pipex4, pipe3, pipei3; /* IQ4_XS, Q3_K, IQ3_S */
    /* the product shaders made for a single vector (prod_index) */
    VkShaderModule pm1[N_PROD];
    VkPipeline pp1[N_PROD];
    /* ... and a subgroup per row for a single vector, where written (NULL:
       pp1): neighbouring lanes read neighbouring memory */
    VkShaderModule rm[N_PROD];
    VkPipeline rp[N_PROD];
    VkShaderModule rm_q6ka; /* ... and the aligned copy of Q6_K */
    VkPipeline rp_q6ka;
    VkShaderModule rm_tile[2]; /* ... and the tiled copies: Q4_K_T, Q6_K_AT */
    VkPipeline rp_tile[2];
    /* the same for several vectors at once (a speculative pass's drafts):
       [k] up to 2 << k of them; [][3] the aligned Q6_K */
    VkShaderModule rmv[3][4];
    VkPipeline rpv[3][4];
    VkCommandPool cp;
    VkCommandBuffer cb;
    VkFence fence;
    VkPhysicalDeviceMemoryProperties mp;
    struct region reg[MAX_REGIONS];
    int n_reg;
    struct stage xs, ys;
    struct pending pend[MAX_PENDING];
    size_t n_pend;
    int busy, broken;
    char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint32_t sg;     /* subgroup size the shader runs with */
    int discrete;    /* weights copied to its memory, not shared */
    int host_import; /* VK_EXT_external_memory_host available */
    int sg_control;  /* the size is required (else the device's fixed one) */
    uint32_t sg_min, sg_max; /* the sizes it may be given (sg_control) */
    /* the GPU's share of the rows, per product shape */
    struct share {
        uint64_t key;
        float f;
        uint32_t calls, waits;
        double t_begin, t_cpu, t_wait, t_copy; /* seconds, verbose only */
        uint64_t rows_gpu, rows_all;
        uint32_t off;     /* calls left on the CPU alone */
        uint32_t seen[4]; /* the type sets printed (JANAS_GPU_VERBOSE=2) */
    } share[SHARE_SLOTS];
    size_t min_n;
    size_t full_n; /* from this many vectors the GPU takes every row */
    int verbose, nospin, block_wait;
    double t_waited; /* last janas_gpu_finish */
    size_t y_used;   /* bytes of the Y stage written by the last group */
    double t_conv, t_rec, t_sub; /* verbose: time in janas_gpu_begin */
    VkDeviceSize atom;           /* nonCoherentAtomSize */
    float ts_period;             /* nanoseconds a timestamp tick */
    /* keep-alive: an empty dispatch every keep_us while running, so that
       the GPU does not fall asleep between products (waking it costs up to
       ~0.5 ms per submission) */
    pthread_mutex_t qlock; /* the queue is shared with the keep-alive */
    pthread_t ka;
    int ka_running;
    atomic_int ka_stop;
    atomic_int ka_wanted; /* only while passes use the GPU */
    unsigned keep_us;
    VkCommandBuffer ka_cb;
    VkFence ka_fence;
};

/* Helpers shared by gpu.c and gpu_token.c (defined in gpu.c). */
int janas_g_mem_type(const struct janas_gpu *g, uint32_t bits,
                     VkMemoryPropertyFlags want);
VkDeviceAddress janas_g_buffer_address(struct janas_gpu *g, VkBuffer b);
int janas_g_make_stage(struct janas_gpu *g, struct stage *s, size_t bytes,
                       int cached);
VkResult janas_g_make_pipeline(struct janas_gpu *g, const uint32_t *code,
                               size_t bytes, VkShaderModule *sm,
                               VkPipeline *pipe, uint32_t kind);
/* ... with nv vectors per workgroup instead of eight */
VkResult janas_g_make_pipeline_nv(struct janas_gpu *g, const uint32_t *code,
                                  size_t bytes, VkShaderModule *sm,
                                  VkPipeline *pipe, uint32_t kind, uint32_t nv);
/* ... with subgroups of sg lanes where the device lets the size be chosen
   and allows it (else the chosen size): the size the pipeline got, 0 on
   failure. */
uint32_t janas_g_make_pipeline_sg(struct janas_gpu *g, const uint32_t *code,
                                  size_t bytes, VkShaderModule *sm,
                                  VkPipeline *pipe, uint32_t kind, uint32_t sg);
const struct region *janas_g_find_region(const struct janas_gpu *g,
                                         const void *p, size_t bytes);
void janas_g_push_addr(uint32_t out[2], VkDeviceAddress a);
void janas_g_compute_barrier(struct janas_gpu *g);
int janas_g_x_in_32(int type);
/*
 * Records the product of rows of a task (its weights in a region the GPU
 * holds) with nv = t->n_vec vectors at xa (the shader's layout, blocks of
 * GPU_BLOCK bytes: the 32-block one where janas_g_x_in_32), into ya with
 * row stride ys floats: binds its pipeline if bound is another. 0 or -1.
 */
int janas_g_record_task(struct janas_gpu *g, VkPipeline *bound,
                        const struct janas_matvec_task *t, size_t rows,
                        VkDeviceAddress xa, VkDeviceAddress ya, size_t ys);

#endif
#endif
