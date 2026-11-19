/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_cuda.c - an NVIDIA card through CUDA's driver API, or an AMD card
 * through HIP's module API, which follows CUDA's function for function
 * (gpu_cuda.h): one code, each function found by its name in the one or
 * the other library.
 *
 * Nothing of NVIDIA's or AMD's is linked: the library is opened at run time,
 * and the kernels (src/llm/cuda/) are what build.sh made of them with
 * clang - PTX, which NVIDIA's driver compiles for the card, and AMD's code
 * objects for each of its GPUs, in one bundle the runtime chooses from.
 * Without them (no clang with nvptx, no lld for AMD's when building) there
 * is no such card to open, and the GPU is Vulkan's alone.
 */
#include "gpu_cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quant.h"

#if defined(JANAS_GPU_CUDA) || defined(JANAS_GPU_HIP)

#include <dlfcn.h>

#ifdef JANAS_GPU_CUDA
#include "matvec_ptx.h" /* build.sh: src/llm/cuda/matvec.cu as PTX */
#endif
#ifdef JANAS_GPU_HIP
#include "matvec_hsaco.h" /* and as AMD's code objects, a bundle */
#endif

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;

#define CU_ATTR_CC_MAJOR 75
#define CU_ATTR_CC_MINOR 76
#define CU_NOT_READY 600
#define CU_HOSTALLOC_PORTABLE 0x01
#define CU_HOSTALLOC_DEVICEMAP 0x02

/* each function: its field, as CUDA names it; HIP's name; its arguments
   (HIP's take the same ones, the same values for their flags and states) */
#define CU_FUNCS(X)                                                            \
    X(cuInit, hipInit, (unsigned))                                             \
    X(cuDriverGetVersion, hipDriverGetVersion, (int *))                        \
    X(cuDeviceGetCount, hipGetDeviceCount, (int *))                            \
    X(cuDeviceGet, hipDeviceGet, (CUdevice *, int))                            \
    X(cuDeviceGetName, hipDeviceGetName, (char *, int, CUdevice))              \
    X(cuDeviceGetAttribute, hipDeviceGetAttribute, (int *, int, CUdevice))     \
    X(cuDeviceTotalMem_v2, hipDeviceTotalMem, (size_t *, CUdevice))            \
    X(cuCtxCreate_v2, hipCtxCreate, (CUcontext *, unsigned, CUdevice))         \
    X(cuCtxDestroy_v2, hipCtxDestroy, (CUcontext))                             \
    X(cuCtxSetCurrent, hipCtxSetCurrent, (CUcontext))                          \
    X(cuModuleLoadData, hipModuleLoadData, (CUmodule *, const void *))         \
    X(cuModuleUnload, hipModuleUnload, (CUmodule))                             \
    X(cuModuleGetFunction, hipModuleGetFunction,                               \
      (CUfunction *, CUmodule, const char *))                                  \
    X(cuMemGetInfo_v2, hipMemGetInfo, (size_t *, size_t *))                    \
    X(cuMemAlloc_v2, hipMalloc, (CUdeviceptr *, size_t))                       \
    X(cuMemFree_v2, hipFree, (CUdeviceptr))                                    \
    X(cuMemHostAlloc, hipHostMalloc, (void **, size_t, unsigned))              \
    X(cuMemFreeHost, hipHostFree, (void *))                                    \
    X(cuMemHostGetDevicePointer_v2, hipHostGetDevicePointer,                   \
      (CUdeviceptr *, void *, unsigned))                                       \
    X(cuMemcpyHtoD_v2, hipMemcpyHtoD, (CUdeviceptr, const void *, size_t))     \
    X(cuMemcpyDtoH_v2, hipMemcpyDtoH, (void *, CUdeviceptr, size_t))           \
    X(cuMemcpyHtoDAsync_v2, hipMemcpyHtoDAsync,                                \
      (CUdeviceptr, const void *, size_t, CUstream))                           \
    X(cuMemcpyDtoHAsync_v2, hipMemcpyDtoHAsync,                                \
      (void *, CUdeviceptr, size_t, CUstream))                                 \
    X(cuStreamCreate, hipStreamCreateWithFlags, (CUstream *, unsigned))        \
    X(cuStreamDestroy_v2, hipStreamDestroy, (CUstream))                        \
    X(cuStreamQuery, hipStreamQuery, (CUstream))                               \
    X(cuStreamSynchronize, hipStreamSynchronize, (CUstream))                   \
    X(cuLaunchKernel, hipModuleLaunchKernel,                                   \
      (CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, \
       unsigned, CUstream, void **, void **))                                  \
    X(cuGetErrorString, hipDrvGetErrorString, (CUresult, const char **))

#define GPU_BLOCK 320       /* a Q8_K block as the kernels read it */
#define X_STAGE (64u << 20) /* the activations of a group */
#define Y_STAGE (32u << 20) /* the rows a group's products give */
#define MAX_PENDING 64
#define MAX_XCACHE 16
#define ROWS_A_BLOCK 128 /* threads of a kernel's block */

struct region {
    const uint8_t *host;
    size_t bytes;
    CUdeviceptr dev;
    int mapped; /* janas_cuda_alloc's: the card reads the host's in place */
};

struct pending {
    float *y;
    size_t rows, nv, y_stride, off;
};

struct janas_cuda {
    void *lib;
#define DECL(f, hipname, args) CUresult(*f) args;
    CU_FUNCS(DECL)
#undef DECL
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction q3k, q4k, q5k, q6k, q80, b32, iq4xs;
    /* the same with 32 threads a row (WARP_NB blocks at most) */
    CUfunction q3k_w, q4k_w, q5k_w, q6k_w, iq4xs_w;
    CUstream st;
    char name[256], about[512];
    int cc_major, cc_minor, driver;
    size_t mem_total, room, used;
    struct region *rg;
    size_t n_rg, cap_rg;
    uint8_t *xh; /* the stages, pinned on the host, and their copies */
    float *yh;
    CUdeviceptr xd, yd;
    struct pending pend[MAX_PENDING];
    size_t n_pend;
    int busy, broken;
    int hip;       /* AMD's HIP, not NVIDIA's CUDA */
    char last[48]; /* the kernel of the last product */
};

static const char *why(const struct janas_cuda *c, CUresult r)
{
    const char *s = NULL;
    if (c->cuGetErrorString)
        c->cuGetErrorString(r, &s);
    return s ? s : "error";
}

/* CUDA's driver (hip 0) or HIP's runtime (1), its first card that can run
   the kernels; NULL with the reason in err */
static struct janas_cuda *open_api(int hip, char *err, size_t err_len)
{
    const char *api = hip ? "HIP" : "CUDA";
    const void *code = NULL;
#ifdef JANAS_GPU_CUDA
    if (!hip)
        code = janas_ptx_matvec;
#endif
#ifdef JANAS_GPU_HIP
    if (hip)
        code = janas_hsaco_matvec;
#endif
    if (!code) {
        snprintf(err, err_len, "built without %s's kernels", api);
        return NULL;
    }
    struct janas_cuda *c = calloc(1, sizeof(*c));
    if (!c) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    c->hip = hip;
#ifdef _WIN32
    static const char *const LIBS[2][3] = {{"nvcuda.dll"},
                                           {"amdhip64_6.dll", "amdhip64.dll"}};
#else
    static const char *const LIBS[2][3] = {
        {"libcuda.so.1"}, {"libamdhip64.so.6", "libamdhip64.so"}};
#endif
    for (int i = 0; i < 3 && LIBS[hip][i] && !c->lib; i++)
        c->lib = dlopen(LIBS[hip][i], RTLD_NOW | RTLD_LOCAL);
    if (!c->lib) {
        snprintf(err, err_len, "no %s (%s)",
                 hip ? "HIP runtime: ROCm, or AMD's HIP SDK" : "CUDA driver",
                 LIBS[hip][0]);
        free(c);
        return NULL;
    }
    /* every function but the error's words, which older HIPs lack */
#define LOAD(f, hipname, args)                                                 \
    *(void **)&c->f = dlsym(c->lib, hip ? #hipname : #f);                      \
    if (!c->f && strcmp(#f, "cuGetErrorString")) {                             \
        snprintf(err, err_len, "the %s library has no %s", api,                \
                 hip ? #hipname : #f);                                         \
        goto fail;                                                             \
    }
    CU_FUNCS(LOAD)
#undef LOAD
    CUresult r;
    int n = 0;
    if ((r = c->cuInit(0)) || (r = c->cuDeviceGetCount(&n)) || n < 1) {
        snprintf(err, err_len, "no %s card (%s)", hip ? "AMD" : "NVIDIA",
                 n < 1 && !r ? "none found" : why(c, r));
        goto fail;
    }
    /* NVIDIA: the first card that has dp4a (6.1: Pascal and later); AMD:
       the first, the bundle holding code for each of its GPUs */
    int found = 0;
    for (int i = 0; i < n && !found; i++) {
        int ma = 0, mi = 0;
        if (c->cuDeviceGet(&c->dev, i))
            continue;
        if (hip) {
            found = 1;
            break;
        }
        if (c->cuDeviceGetAttribute(&ma, CU_ATTR_CC_MAJOR, c->dev) ||
            c->cuDeviceGetAttribute(&mi, CU_ATTR_CC_MINOR, c->dev))
            continue;
        if (ma > 6 || (ma == 6 && mi >= 1)) {
            found = 1;
            c->cc_major = ma;
            c->cc_minor = mi;
        }
    }
    if (!found) {
        snprintf(err, err_len,
                 hip ? "no AMD card"
                     : "no NVIDIA card of compute capability "
                       "6.1 or later");
        goto fail;
    }
    c->cuDeviceGetName(c->name, sizeof c->name - 1, c->dev);
    c->cuDriverGetVersion(&c->driver);
    c->cuDeviceTotalMem_v2(&c->mem_total, c->dev);
    if ((r = c->cuCtxCreate_v2(&c->ctx, 0, c->dev))) {
        snprintf(err, err_len, "no %s context (%s)", api, why(c, r));
        goto fail;
    }
    if ((r = c->cuModuleLoadData(&c->mod, code)) ||
        (r = c->cuModuleGetFunction(&c->q4k, c->mod, "q4k_matvec")) ||
        (r = c->cuModuleGetFunction(&c->q6k, c->mod, "q6k_matvec")) ||
        (r = c->cuModuleGetFunction(&c->q5k, c->mod, "q5k_matvec")) ||
        (r = c->cuModuleGetFunction(&c->q3k, c->mod, "q3k_matvec")) ||
        (r = c->cuModuleGetFunction(&c->q80, c->mod, "q80_matvec")) ||
        (r = c->cuModuleGetFunction(&c->b32, c->mod, "b32_matvec")) ||
        (r = c->cuModuleGetFunction(&c->iq4xs, c->mod, "iq4xs_matvec")) ||
        (r = c->cuModuleGetFunction(&c->q4k_w, c->mod, "q4k_matvec_w")) ||
        (r = c->cuModuleGetFunction(&c->q6k_w, c->mod, "q6k_matvec_w")) ||
        (r = c->cuModuleGetFunction(&c->q5k_w, c->mod, "q5k_matvec_w")) ||
        (r = c->cuModuleGetFunction(&c->q3k_w, c->mod, "q3k_matvec_w")) ||
        (r = c->cuModuleGetFunction(&c->iq4xs_w, c->mod, "iq4xs_matvec_w"))) {
        snprintf(err, err_len, "the %s kernels did not load (%s)", api,
                 why(c, r));
        goto fail;
    }
    size_t free_b = 0, total_b = 0;
    c->cuMemGetInfo_v2(&free_b, &total_b);
    c->room = free_b / 100 * 85;
    if ((r = c->cuStreamCreate(&c->st, 0)) ||
        (r = c->cuMemHostAlloc((void **)&c->xh, X_STAGE,
                               CU_HOSTALLOC_PORTABLE)) ||
        (r = c->cuMemHostAlloc((void **)&c->yh, Y_STAGE,
                               CU_HOSTALLOC_PORTABLE)) ||
        (r = c->cuMemAlloc_v2(&c->xd, X_STAGE)) ||
        (r = c->cuMemAlloc_v2(&c->yd, Y_STAGE))) {
        snprintf(err, err_len, "the %s buffers were not made (%s)", api,
                 why(c, r));
        goto fail;
    }
    c->used = X_STAGE + Y_STAGE;
    if (hip)
        snprintf(c->about, sizeof c->about, "%s, HIP runtime %d, %.1f GiB",
                 c->name, c->driver, (double)c->mem_total / (1u << 30));
    else
        snprintf(c->about, sizeof c->about,
                 "%s, compute capability %d.%d, CUDA driver %d.%d, %.1f GiB",
                 c->name, c->cc_major, c->cc_minor, c->driver / 1000,
                 c->driver % 1000 / 10, (double)c->mem_total / (1u << 30));
    return c;
fail:
    janas_cuda_destroy(c);
    return NULL;
}

struct janas_cuda *janas_cuda_create(char *err, size_t err_len)
{
    return open_api(0, err, err_len);
}

struct janas_cuda *janas_hip_create(char *err, size_t err_len)
{
    return open_api(1, err, err_len);
}

void janas_cuda_destroy(struct janas_cuda *c)
{
    if (!c)
        return;
    if (c->ctx) {
        c->cuCtxSetCurrent(c->ctx);
        for (size_t i = 0; i < c->n_rg; i++)
            if (c->rg[i].mapped)
                c->cuMemFreeHost((void *)c->rg[i].host);
            else
                c->cuMemFree_v2(c->rg[i].dev);
        if (c->xd)
            c->cuMemFree_v2(c->xd);
        if (c->yd)
            c->cuMemFree_v2(c->yd);
        if (c->xh)
            c->cuMemFreeHost(c->xh);
        if (c->yh)
            c->cuMemFreeHost(c->yh);
        if (c->st)
            c->cuStreamDestroy_v2(c->st);
        if (c->mod)
            c->cuModuleUnload(c->mod);
        c->cuCtxDestroy_v2(c->ctx);
    }
    free(c->rg);
    if (c->lib)
        dlclose(c->lib);
    free(c);
}

const char *janas_cuda_name(const struct janas_cuda *c)
{
    return c->name;
}
const char *janas_cuda_last(const struct janas_cuda *c)
{
    return c->last;
}
const char *janas_cuda_about(const struct janas_cuda *c)
{
    return c->about;
}

void janas_cuda_card(const struct janas_cuda *c, char *buf, size_t len)
{
    size_t free_b = 0, total_b = 0;
    c->cuCtxSetCurrent(c->ctx);
    c->cuMemGetInfo_v2(&free_b, &total_b);
    snprintf(buf, len, "%s: %.1f GiB of memory, %.1f GiB free", c->name,
             (double)total_b / (1u << 30), (double)free_b / (1u << 30));
}

size_t janas_cuda_local_room(const struct janas_cuda *c)
{
    return c->room > c->used ? c->room - c->used : 0;
}

static struct region *add_region(struct janas_cuda *c)
{
    if (c->n_rg == c->cap_rg) {
        size_t cap = c->cap_rg ? 2 * c->cap_rg : 64;
        struct region *r = realloc(c->rg, cap * sizeof(*r));
        if (!r)
            return NULL;
        c->rg = r;
        c->cap_rg = cap;
    }
    return &c->rg[c->n_rg++];
}

/* the region holding [p, p + bytes), or NULL */
static const struct region *find(const struct janas_cuda *c, const void *p,
                                 size_t bytes)
{
    const uint8_t *q = p;
    for (size_t i = 0; i < c->n_rg; i++)
        if (q >= c->rg[i].host && q + bytes <= c->rg[i].host + c->rg[i].bytes)
            return &c->rg[i];
    return NULL;
}

int janas_cuda_import(struct janas_cuda *c, const void *p, size_t bytes)
{
    if (!p || !bytes || c->busy || find(c, p, bytes))
        return find(c, p, bytes) ? 0 : -1;
    if (bytes > janas_cuda_local_room(c))
        return -1; /* the card is full: these rows stay on the CPU */
    c->cuCtxSetCurrent(c->ctx);
    CUdeviceptr d;
    if (c->cuMemAlloc_v2(&d, bytes))
        return -1;
    if (c->cuMemcpyHtoD_v2(d, p, bytes)) {
        c->cuMemFree_v2(d);
        return -1;
    }
    struct region *r = add_region(c);
    if (!r) {
        c->cuMemFree_v2(d);
        return -1;
    }
    *r = (struct region){.host = p, .bytes = bytes, .dev = d};
    c->used += bytes;
    return 0;
}

void *janas_cuda_alloc(struct janas_cuda *c, size_t bytes)
{
    bytes = (bytes + 4095) / 4096 * 4096;
    c->cuCtxSetCurrent(c->ctx);
    void *p = NULL;
    CUdeviceptr d;
    if (c->cuMemHostAlloc(&p, bytes,
                          CU_HOSTALLOC_PORTABLE | CU_HOSTALLOC_DEVICEMAP))
        return NULL;
    struct region *r = NULL;
    if (c->cuMemHostGetDevicePointer_v2(&d, p, 0) || !(r = add_region(c))) {
        c->cuMemFreeHost(p);
        return NULL;
    }
    *r = (struct region){.host = p, .bytes = bytes, .dev = d, .mapped = 1};
    return p;
}

int janas_cuda_sync(struct janas_cuda *c, void *p, size_t bytes, int to_device)
{
    const struct region *r = find(c, p, bytes);
    if (!r || c->busy)
        return -1;
    if (r->mapped)
        return 0; /* the same memory */
    c->cuCtxSetCurrent(c->ctx);
    CUdeviceptr d = r->dev + (CUdeviceptr)((const uint8_t *)p - r->host);
    return (to_device ? c->cuMemcpyHtoD_v2(d, p, bytes)
                      : c->cuMemcpyDtoH_v2(p, d, bytes))
               ? -1
               : 0;
}

/* b32_matvec's kind of a type, -1 if it is not one of its (as gpu.c's) */
static int b32_kind(int type)
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

/* the types the kernels compute */
static CUfunction kernel_of(const struct janas_cuda *c, int type)
{
    return type == JANAS_Q4_K     ? c->q4k
           : type == JANAS_Q6_K   ? c->q6k
           : type == JANAS_Q5_K   ? c->q5k
           : type == JANAS_Q3_K   ? c->q3k
           : type == JANAS_Q8_0   ? c->q80
           : b32_kind(type) >= 0  ? c->b32
           : type == JANAS_IQ4_XS ? c->iq4xs
                                  : NULL;
}

/* the kernel with 32 threads a row, for rows of 256 blocks at most (the
   shared memory it keeps a row's terms in): NULL for the others */
#define WARP_NB 256
static CUfunction warp_of(const struct janas_cuda *c, int type, size_t nb)
{
    if (nb > WARP_NB)
        return NULL;
    return type == JANAS_Q4_K     ? c->q4k_w
           : type == JANAS_Q6_K   ? c->q6k_w
           : type == JANAS_Q5_K   ? c->q5k_w
           : type == JANAS_Q3_K   ? c->q3k_w
           : type == JANAS_IQ4_XS ? c->iq4xs_w
                                  : NULL;
}

/* its name, for the reports */
static const char *kernel_name(int type)
{
    return type == JANAS_Q4_K     ? "q4k_matvec"
           : type == JANAS_Q6_K   ? "q6k_matvec"
           : type == JANAS_Q5_K   ? "q5k_matvec"
           : type == JANAS_Q3_K   ? "q3k_matvec"
           : type == JANAS_Q8_0   ? "q80_matvec"
           : type == JANAS_IQ4_XS ? "iq4xs_matvec"
                                  : "b32_matvec";
}

int janas_cuda_can(const struct janas_cuda *c,
                   const struct janas_matvec_task *t)
{
    if (c->broken || !kernel_of(c, t->type) || t->cols % JANAS_QK)
        return 0;
    size_t nb = t->cols / JANAS_QK, nv = t->n_vec ? t->n_vec : 1;
    if (nv * nb * GPU_BLOCK > X_STAGE || nv * t->rows * 4 > Y_STAGE)
        return 0;
    return find(c, t->w, t->rows * janas_qtype_row_bytes(t->type, t->cols)) !=
           NULL;
}

/* Q8_K vectors as the kernels read them, as gpu.c's convert_x: the quants,
   the sums, the scale; for the weights in blocks of 32 (k32) the same 256
   in blocks of 32 with a scale each */
static void convert_x(uint8_t *dst, const struct janas_block_q8k *x, size_t xs,
                      size_t nv, size_t nb, int k32)
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

/* the types that read the activations in blocks of 32 */
static int in_32(int type)
{
    return type == JANAS_Q8_0 || b32_kind(type) >= 0;
}

int janas_cuda_begin(struct janas_cuda *c,
                     const struct janas_matvec_task *tasks,
                     const size_t *gpu_rows, size_t n)
{
    if (c->busy || c->broken)
        return -1;
    c->cuCtxSetCurrent(c->ctx);
    struct {
        const void *x;
        size_t nv, xs, nb, off;
        int k32;
    } xc[MAX_XCACHE];
    size_t n_xc = 0, x_used = 0, y_used = 0, launched = 0;
    size_t xk[MAX_PENDING];
    c->n_pend = 0;
    /* the activations, each once, then their copy */
    for (size_t i = 0; i < n; i++) {
        if (!gpu_rows[i])
            continue;
        const struct janas_matvec_task *t = &tasks[i];
        if (i >= MAX_PENDING || !kernel_of(c, t->type))
            return -1;
        size_t nb = t->cols / JANAS_QK, nv = t->n_vec ? t->n_vec : 1;
        size_t xs = t->x_stride ? t->x_stride : nb, k = 0;
        int k32 = in_32(t->type);
        while (k < n_xc &&
               !(xc[k].x == t->x && xc[k].nv == nv && xc[k].xs == xs &&
                 xc[k].nb == nb && xc[k].k32 == k32))
            k++;
        if (k == n_xc) {
            size_t bytes = nv * nb * GPU_BLOCK;
            if (n_xc == MAX_XCACHE || x_used + bytes > X_STAGE)
                return -1;
            convert_x(c->xh + x_used, t->x, xs, nv, nb, k32);
            xc[k].k32 = k32;
            xc[k].x = t->x;
            xc[k].nv = nv;
            xc[k].xs = xs;
            xc[k].nb = nb;
            xc[k].off = x_used;
            x_used += bytes;
            n_xc++;
        }
        xk[i] = xc[k].off;
    }
    if (x_used && c->cuMemcpyHtoDAsync_v2(c->xd, c->xh, x_used, c->st))
        goto broke;
    for (size_t i = 0; i < n; i++) {
        if (!gpu_rows[i])
            continue;
        const struct janas_matvec_task *t = &tasks[i];
        size_t nb = t->cols / JANAS_QK, nv = t->n_vec ? t->n_vec : 1;
        size_t rows = gpu_rows[i];
        size_t ybytes = rows * nv * sizeof(float);
        const struct region *r =
            find(c, t->w, rows * janas_qtype_row_bytes(t->type, t->cols));
        if (!r || c->n_pend == MAX_PENDING || y_used + ybytes > Y_STAGE) {
            c->cuStreamSynchronize(c->st); /* nothing left running */
            return -1;
        }
        CUdeviceptr w = r->dev + (CUdeviceptr)((const uint8_t *)t->w - r->host);
        CUdeviceptr x = c->xd + xk[i], y = c->yd + y_used;
        uint32_t a_rows = (uint32_t)rows, a_nb = (uint32_t)nb,
                 a_nv = (uint32_t)nv, a_xs = (uint32_t)nb, a_ys = a_rows;
        uint32_t a_kind = (uint32_t)b32_kind(t->type); /* b32_matvec's */
        void *args[] = {&w,    &x,    &y,    &a_rows, &a_nb,
                        &a_nv, &a_xs, &a_ys, &a_kind};
        /* 32 threads a row where the type has such a kernel: 4 rows a block */
        CUfunction wf = warp_of(c, t->type, nb);
        unsigned blocks =
            wf ? (unsigned)((rows + 3) / 4)
               : (unsigned)((rows + ROWS_A_BLOCK - 1) / ROWS_A_BLOCK);
        if (c->cuLaunchKernel(wf ? wf : kernel_of(c, t->type), blocks, 1, 1,
                              ROWS_A_BLOCK, 1, 1, 0, c->st, args, NULL))
            goto broke;
        launched++;
        snprintf(c->last, sizeof c->last, "%s%s (%s)", kernel_name(t->type),
                 wf ? "_w" : "", c->hip ? "HIP" : "CUDA");
        c->pend[c->n_pend++] =
            (struct pending){.y = t->y,
                             .rows = rows,
                             .nv = nv,
                             .y_stride = t->y_stride ? t->y_stride : t->rows,
                             .off = y_used};
        y_used += ybytes;
    }
    if (y_used && c->cuMemcpyDtoHAsync_v2(c->yh, c->yd, y_used, c->st))
        goto broke;
    c->busy = 1;
    return 0;
broke:
    c->broken = 1; /* the card failed: the CPU from now on */
    c->cuStreamSynchronize(c->st);
    return -1;
}

int janas_cuda_finish(struct janas_cuda *c, int *waited)
{
    if (waited)
        *waited = 0;
    if (!c->busy)
        return 0;
    c->busy = 0;
    c->cuCtxSetCurrent(c->ctx);
    if (waited)
        *waited = c->cuStreamQuery(c->st) == CU_NOT_READY;
    if (c->cuStreamSynchronize(c->st)) {
        c->broken = 1;
        return -1;
    }
    for (size_t i = 0; i < c->n_pend; i++) {
        const struct pending *p = &c->pend[i];
        const float *src = (const float *)((const uint8_t *)c->yh + p->off);
        for (size_t v = 0; v < p->nv; v++)
            memcpy(p->y + v * p->y_stride, src + v * p->rows,
                   p->rows * sizeof(float));
    }
    c->n_pend = 0;
    return 0;
}

#else /* built without CUDA's or HIP's kernels */

struct janas_cuda *janas_cuda_create(char *err, size_t err_len)
{
    snprintf(err, err_len, "built without CUDA (no clang with nvptx)");
    return NULL;
}
struct janas_cuda *janas_hip_create(char *err, size_t err_len)
{
    snprintf(err, err_len, "built without HIP (no clang with lld for AMD)");
    return NULL;
}
void janas_cuda_destroy(struct janas_cuda *c)
{
    (void)c;
}
const char *janas_cuda_name(const struct janas_cuda *c)
{
    (void)c;
    return "";
}
const char *janas_cuda_about(const struct janas_cuda *c)
{
    (void)c;
    return "";
}
void janas_cuda_card(const struct janas_cuda *c, char *buf, size_t len)
{
    (void)c;
    if (len)
        buf[0] = 0;
}
size_t janas_cuda_local_room(const struct janas_cuda *c)
{
    (void)c;
    return 0;
}
int janas_cuda_import(struct janas_cuda *c, const void *p, size_t bytes)
{
    (void)c, (void)p, (void)bytes;
    return -1;
}
void *janas_cuda_alloc(struct janas_cuda *c, size_t bytes)
{
    (void)c, (void)bytes;
    return NULL;
}
int janas_cuda_sync(struct janas_cuda *c, void *p, size_t bytes, int to_device)
{
    (void)c, (void)p, (void)bytes, (void)to_device;
    return -1;
}
int janas_cuda_can(const struct janas_cuda *c,
                   const struct janas_matvec_task *t)
{
    (void)c, (void)t;
    return 0;
}
int janas_cuda_begin(struct janas_cuda *c,
                     const struct janas_matvec_task *tasks,
                     const size_t *gpu_rows, size_t n)
{
    (void)c, (void)tasks, (void)gpu_rows, (void)n;
    return -1;
}
int janas_cuda_finish(struct janas_cuda *c, int *waited)
{
    (void)c;
    if (waited)
        *waited = 0;
    return -1;
}
const char *janas_cuda_last(const struct janas_cuda *c)
{
    (void)c;
    return "";
}

#endif
