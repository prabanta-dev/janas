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
#define N_PROD 15
/* the kinds of b32_matvec.comp (gpu.c's b32_kind) */
#define N_B32 8
/* the 256-block types of iq2_matvec, iq3xxs_matvec, iq1_matvec and
   q2k_matvec.comp (gpu.c's iqk_index) */
#define N_IQK 7

#define VK_NO_PROTOTYPES
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

/* the spin-wait hint while a submission runs, as pool.c's */
#if defined(__x86_64__)
#include <immintrin.h>
#define gpu_relax() _mm_pause()
#elif defined(__aarch64__)
#define gpu_relax() __asm__ __volatile__("yield" ::: "memory")
#else
#define gpu_relax() ((void)0)
#endif

/* the resident region, the experts' arena (4 GiB a region: 15 on a
   machine whose cache holds 60 GB), the engine's buffers (some 30), the
   KV cache */
#define MAX_REGIONS 80
#define MAX_PENDING 64
#define MAX_XCACHE 8
#define X_STAGE_BYTES                                                          \
    ((size_t)16 << 20) /* a 512-token block's experts: 10.5 MB */
#define Y_STAGE_BYTES ((size_t)32 << 20)
#define ROWS_PER_GROUP 64 /* local_size_x of the shader */
#define NV_PER_GROUP 8    /* its NV */
#define MV_NV 8           /* q4k_mv's */
#define B16_NV 4          /* bf16_matvec's vectors a workgroup */
#define MVD_ROWS 32       /* q4k_mvd's rows a workgroup */
#define GPU_BLOCK 320     /* a Q8_K block in the shader's layout */
#define SHARE_SLOTS 64
#define MAX_GROUP 64
#define MAX_MADE 384 /* pipelines a device gets (some 200) */

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
    X(vkCmdDispatchIndirect)                                                   \
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
    /* the host's own memory (imported, or the driver's mapped): what the
       GPU writes there the CPU reads. 0: a copy in a card's own memory
       (copy_to_device), which the GPU's writes never leave */
    int host_shared;
    VkDeviceSize local; /* of the card's own memory, given back with it */
};

struct stage {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *map;
    VkDeviceAddress addr;
    size_t bytes;
    int coherent;       /* else invalidated before the host reads it */
    VkDeviceSize local; /* of the card's own memory (janas_g_make_local) */
};

struct pending {
    float *y;        /* the task's output */
    size_t rows, nv; /* rows computed by the GPU, vectors */
    size_t y_stride; /* the task's */
    size_t off;      /* in the Y stage */
};

/* The vectors a workgroup of pipe_mvid takes */
#define MVID_NV 4

/* The push constants of the shaders (level and the bare planes: Q6_K_P
   only). */
struct push {
    uint32_t w[2], x[2], y[2];
    uint32_t rows, nb, nv, xs, ys, level;
    uint32_t p1[2], p0[2];
    /* b32_matvec.comp, q8_0_matvec.comp: blocks of 32 past the row's
       whole groups of 256 */
    uint32_t tail;
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
    VkShaderModule sm, sm6, sm6p, sm5, sm80, sm32[N_B32], smx4, sm3, smi3, sma,
        sma32;
    VkPipelineLayout pl;
    VkPipeline pipe, pipe6, pipe6p, pipe5, pipe80, pipea; /* Q4_K, Q6_K,
                                                             Q6_K_P, Q5_K,
                                                             Q8_0, experts'
                                                             activation */
    /* the experts' activation in blocks of 32 (expert_act.comp's KIND 1),
       for a down matrix in blocks of 32 (MXFP4) */
    VkPipeline pipea32;
    /* blocks of 32: Q4_0, IQ4_NL, Q4_1, Q5_1, MXFP4 (b32_matvec's KIND) */
    VkPipeline pipe32[N_B32];
    VkPipeline pipex4, pipe3, pipei3; /* IQ4_XS, Q3_K, IQ3_S */
    /* Q4_K on many vectors in tiles (q4k_mm.comp), MM_ROWS x MM_VECS */
    VkShaderModule sm_mm, sm_mm6, sm_mm6s, sm_mm80, sm_mm5, sm_mm3, sm_mmx4,
        sm_mmi3, sm_mmn4, sm_mmmx, sm_mm50, sm_mm20, sm_dm[3],
        sm_dv[3]; /* sm_mmx4 IQ4_XS, mmi3 IQ3_S, mmn4 IQ4_NL,
                     mmmx MXFP4, mm50 Q5_0, mm20 Q2_0; dm f32,
                     bf16 and Q8_0 weights on float vectors, dv
                     the same on a few (optional) */
    VkPipeline pipe_mm, pipe_mm6, pipe_mm6s, pipe_mm80, pipe_mm5, pipe_mm3,
        pipe_mmx4, pipe_mmi3, pipe_mmn4, pipe_mmmx, pipe_mm50, pipe_mm20,
        pipe_dm[3], pipe_dv[3];
    /* q3k, q4k, q6k, q6kp_matvec and b32_matvec's MXFP4 and Q2_0 with ID:
       a mixture's experts on a few vectors, MVID_NV a workgroup, from
       *_mmid.comp's table (optional) */
    VkShaderModule sm_mvid[9];
    VkPipeline pipe_mvid[9];
    int only_fast;   /* janas_gpu_set_only_fast */
    uint32_t pass_n; /* janas_gpu_set_pass; UINT32_MAX until said */
    uint32_t nvg;    /* vectors a product's workgroup takes (NV_PER_GROUP) */
    /* a mixture's experts in one dispatch (experts_id): Q4_K, Q6_K_P,
       Q6_K, MXFP4 */
    VkShaderModule sm_id4, sm_id6, sm_id6s, sm_idmx, sm_id3, sm_id5;
    VkPipeline pipe_id4, pipe_id6, pipe_id6s, pipe_idmx, pipe_id3, pipe_id5;
    /* ... and in blocks of 32 or 64 (b32_mmid.comp): IQ4_NL, Q5_0, Q2_0 */
    VkShaderModule sm_idb[3];
    VkPipeline pipe_idb[3];
    /* bf16 weights on float vectors, bit for bit (bf16_matvec.comp) */
    VkShaderModule sm_b16;
    VkPipeline pipe_b16;
    /* Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S, IQ1_M, a row an
       invocation, for groups of vectors and for one (NULL: the CPU's) */
    VkShaderModule sm_iqk[N_IQK], sm1_iqk[N_IQK];
    VkPipeline pipe_iqk[N_IQK], pp1_iqk[N_IQK];
    /* ... and their experts in one dispatch, IQ3_S's and IQ4_XS's too
       (iqk_mmid.comp, gpu.c's iqm_kind) */
    VkShaderModule sm_iqm[N_IQK + 2];
    VkPipeline pipe_iqm[N_IQK + 2];
    /* ... and their tiles for the fast prompt (iqk_mmq.comp) */
    VkShaderModule sm_iqq[N_IQK];
    VkPipeline pipe_iqq[N_IQK];
    struct stage ts; /* their table, 3 uvec4 an expert */
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
    VkShaderModule sm_mv; /* q4k_mv, MV_NV vectors (JANAS_GPU_MV=1) */
    VkPipeline pipe_mv;
    /* q4k_mvd, q6k_mvd, q5k_mvd, iq4xs_mvd: 2-8 vectors, subgroups of 8 */
    VkShaderModule sm_mvd, sm_mvd6, sm_mvd5, sm_mvdx4;
    VkPipeline pipe_mvd, pipe_mvd6, pipe_mvd5, pipe_mvdx4;
    VkCommandPool cp;
    VkCommandBuffer cb;
    VkFence fence;
    VkPhysicalDeviceMemoryProperties mp;
    struct region reg[MAX_REGIONS];
    int n_reg;
    struct stage xs, ys;
    struct stage kas; /* the keep-alive's float: its own, never a stage in
                         use (another thread writes it, unordered) */
    struct pending pend[MAX_PENDING];
    size_t n_pend;
    int busy, broken;
    char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint32_t sg;              /* subgroup size the shader runs with */
    int discrete;             /* weights copied to its memory, not shared */
    VkDeviceSize local_bytes; /* taken of its own memory, less what was
                                 given back */
    int host_import;          /* VK_EXT_external_memory_host available */
    int sg_control; /* the size is required (else the device's fixed one) */
    uint32_t sg_min, sg_max; /* the sizes it may be given (sg_control) */
    /* the GPU's share of the rows, per product shape */
    struct share {
        uint64_t key;
        float f;
        /* alone: calls with every row on the GPU, where the CPU waits
           having nothing to do; waits: the others the GPU was late in */
        uint32_t calls, waits, alone;
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
    double t_conv, t_rec, t_sub;    /* verbose: time in janas_gpu_begin */
    uint64_t b16_calls, b16_staged; /* janas_gpu_bf16_mv's, staged among them */
    VkDeviceSize atom;              /* nonCoherentAtomSize */
    float ts_period;                /* nanoseconds a timestamp tick */
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
    /* every pipeline made, for janas_gpu_last_shader: its SPIR-V, vectors
       a workgroup, subgroup size and kind */
    struct made {
        VkPipeline p;
        const uint32_t *code;
        uint32_t nv, sg, kind;
    } made[MAX_MADE];
    int n_made;
    VkPipeline last_pipe; /* the last product janas_g_record_task recorded */
    /* the paths a product may take (janas_gpu_set_path; at creation from
       JANAS_GPU_NO_ROW, JANAS_GPU_ROW_NV, JANAS_GPU_MVD, JANAS_GPU_MV) */
    int row_on, row_nv, mvd_on, mv_on;
    char about[768]; /* janas_gpu_about */
    uint32_t vendor_id, device_id, driver_version;
    /* what the check on opening left out (gpu_selfcheck.c): types the GPU
       got wrong whatever the shader, and a line for the user */
    int off_types[16], n_off;
    char check_note[256];
    /* gpu_fma.c: shaders with float64 (shaderFloat64), and the _xfma
       variants taken because this driver splits fma() */
    int has_f64, xfma;
};

/* The products checked against the CPU's when the GPU opens, and the
   shaders or types this driver gets wrong left out (gpu_selfcheck.c). */
void janas_g_selfcheck(struct janas_gpu *g);
/* Whether the check left the type out. */
int janas_g_type_off(const struct janas_gpu *g, int type);
/* Whether this driver rounds fma() once, as the CPU does: if not, the
   shaders' _xfma variants from then on (g->xfma), when the device has
   float64 and they pass the same probe (gpu_fma.c). Needs g->cb. */
void janas_g_fma_probe(struct janas_gpu *g);
/* The SPIR-V to build for code: its _xfma variant where g->xfma. */
const uint32_t *janas_g_spv(const struct janas_gpu *g, const uint32_t *code,
                            size_t *bytes);
/* A buffer in the device's own memory (no host mapping), for shaders and
   copies; 0, or -1. */
int janas_g_make_local(struct janas_gpu *g, struct stage *s, size_t bytes);
/* Copies bytes between two buffers and waits: 0, or -1. */
int janas_g_copy(struct janas_gpu *g, VkBuffer src, VkDeviceSize so,
                 VkBuffer dst, VkDeviceSize dof, VkDeviceSize bytes);
/* A copy of janas_g_copies'. */
struct janas_g_cp {
    VkBuffer src, dst;
    VkDeviceSize so, dof, bytes;
};
/* Copies c[0..n) in one submission and waits: 0, or -1. */
int janas_g_copies(struct janas_gpu *g, const struct janas_g_cp *c, uint32_t n);
/* The same recorded in g->cb, not submitted: with barrier, followed by a
   barrier before what reads them in the same submission. 0, or -1. */
int janas_g_record_copies(struct janas_gpu *g, const struct janas_g_cp *c,
                          uint32_t n, int barrier);
/* Forgets the last region imported, at host (the check's own weights). */
void janas_g_drop_region(struct janas_gpu *g, const void *host);

/* Helpers shared by gpu.c and gpu_token.c (defined in gpu.c). */
int janas_g_mem_type(const struct janas_gpu *g, uint32_t bits,
                     VkMemoryPropertyFlags want);
VkDeviceAddress janas_g_buffer_address(struct janas_gpu *g, VkBuffer b);
/* A stage's buffer and memory freed, what it took of the card's own memory
   given back; the stage zeroed. */
void janas_g_free_stage(struct janas_gpu *g, struct stage *s);
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
int janas_g_copy_asked(void); /* JANAS_GPU_COPY=1 */
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
