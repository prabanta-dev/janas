/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_metal.m - Apple's GPU through Metal, behind gpu_cuda.h's interface:
 * on macOS there is no CUDA nor HIP, and gpu.c asks this file what it asks
 * of them (the products split by rows, what has no kernel on the CPU).
 *
 * The kernels are cuda/matvec.cu's, compiled when the GPU is opened as
 * Metal's shading language (build.sh puts its text in matvec_metal.h),
 * with Metal's exact arithmetic: no fast math, its fma rounded once - the
 * same bits as the CPU's. The GPU shares the memory with the CPU: the
 * weights are handed to it where they are (a buffer made on the host's
 * pages, nothing copied), the activations and the products' rows go
 * through two buffers both see.
 */
#if defined(__APPLE__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu_cuda.h"
#include "gpu_drv.h"
#include "quant.h"

#include <mach/mach.h>
#include <stdio.h>

#include "matvec_metal.h" /* build.sh: src/llm/cuda/matvec.cu's text */

#define THREADS 128 /* threads of a group */

struct janas_cuda {
    void *dev;   /* id<MTLDevice>, kept as a retained pointer */
    void *queue; /* id<MTLCommandQueue> */
    void *k_q3k, *k_q4k, *k_q5k, *k_q6k, *k_q80, *k_b32, *k_iq4xs;
    void *xbuf, *ybuf; /* the stages, shared */
    uint8_t *xh;
    float *yh;
    void *cmd; /* the command buffer of the products running */
    struct drv_regions regs;
    struct drv_pending pend[MAX_PENDING];
    size_t n_pend;
    size_t room, used;
    int busy, broken;
    char name[128], about[256], last[48];
};

#define OBJ(p) ((__bridge id)(p))
static void *keep(id o) { return o ? (__bridge_retained void *)o : NULL; }
static void drop(void *p)
{
    if (p)
        CFRelease(p);
}

static void *pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, const char *name,
                      char *err, size_t err_len)
{
    id<MTLFunction> f =
        [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    NSError *e = nil;
    id<MTLComputePipelineState> ps =
        f ? [dev newComputePipelineStateWithFunction:f error:&e] : nil;
    if (!ps)
        snprintf(err, err_len, "Metal: kernel %s: %s", name,
                 e ? e.localizedDescription.UTF8String : "not in the library");
    return keep(ps);
}

struct janas_cuda *janas_cuda_create(char *err, size_t err_len)
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            snprintf(err, err_len, "Metal: no GPU");
            return NULL;
        }
        MTLCompileOptions *o = [MTLCompileOptions new];
        /* the CPU's bits: no fast math, no fused operations of its own */
        if (@available(macOS 15.0, *))
            o.mathMode = MTLMathModeSafe;
        else
            o.fastMathEnabled = NO;
        NSString *src = [[NSString alloc]
            initWithBytes:janas_matvec_metal
                   length:sizeof(janas_matvec_metal) - 1
                 encoding:NSUTF8StringEncoding];
        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&e];
        if (!lib) {
            snprintf(err, err_len, "Metal: the kernels do not compile: %.200s",
                     e ? e.localizedDescription.UTF8String : "");
            return NULL;
        }
        struct janas_cuda *c = calloc(1, sizeof(*c));
        if (!c) {
            snprintf(err, err_len, "out of memory");
            return NULL;
        }
        c->dev = keep(dev);
        c->queue = keep([dev newCommandQueue]);
        if (!(c->k_q3k = pipeline(dev, lib, "q3k_matvec", err, err_len)) ||
            !(c->k_q4k = pipeline(dev, lib, "q4k_matvec", err, err_len)) ||
            !(c->k_q5k = pipeline(dev, lib, "q5k_matvec", err, err_len)) ||
            !(c->k_q6k = pipeline(dev, lib, "q6k_matvec", err, err_len)) ||
            !(c->k_q80 = pipeline(dev, lib, "q80_matvec", err, err_len)) ||
            !(c->k_b32 = pipeline(dev, lib, "b32_matvec", err, err_len)) ||
            !(c->k_iq4xs = pipeline(dev, lib, "iq4xs_matvec", err, err_len))) {
            janas_cuda_destroy(c);
            return NULL;
        }
        id<MTLBuffer> xb = [dev newBufferWithLength:X_STAGE
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> yb = [dev newBufferWithLength:Y_STAGE
                                            options:MTLResourceStorageModeShared];
        if (!xb || !yb) {
            snprintf(err, err_len, "Metal: no memory for the stages");
            janas_cuda_destroy(c);
            return NULL;
        }
        c->xbuf = keep(xb);
        c->ybuf = keep(yb);
        c->xh = xb.contents;
        c->yh = yb.contents;
        /* what Metal says it can keep resident, less the stages */
        size_t ws = (size_t)dev.recommendedMaxWorkingSetSize;
        c->room = ws > X_STAGE + Y_STAGE ? ws - X_STAGE - Y_STAGE : 0;
        snprintf(c->name, sizeof c->name, "%s", dev.name.UTF8String);
        snprintf(c->about, sizeof c->about,
                 "%s through Metal, unified memory %s, working set %.1f GB",
                 c->name, dev.hasUnifiedMemory ? "yes" : "no", ws / 1e9);
        return c;
    }
}

struct janas_cuda *janas_hip_create(char *err, size_t err_len)
{
    snprintf(err, err_len, "no HIP on macOS: Apple's GPU is Metal's");
    return NULL;
}

void janas_cuda_destroy(struct janas_cuda *c)
{
    if (!c)
        return;
    janas_cuda_finish(c, NULL);
    for (size_t i = 0; i < c->regs.n; i++) {
        drop((void *)(uintptr_t)c->regs.rg[i].dev);
        if (c->regs.rg[i].mapped)
            free((void *)c->regs.rg[i].host);
    }
    free(c->regs.rg);
    drop(c->xbuf);
    drop(c->ybuf);
    drop(c->k_q3k), drop(c->k_q4k), drop(c->k_q5k), drop(c->k_q6k);
    drop(c->k_q80), drop(c->k_b32), drop(c->k_iq4xs);
    drop(c->queue);
    drop(c->dev);
    free(c);
}

const char *janas_cuda_name(const struct janas_cuda *c) { return c->name; }
const char *janas_cuda_about(const struct janas_cuda *c) { return c->about; }
const char *janas_cuda_last(const struct janas_cuda *c) { return c->last; }

void janas_cuda_card(const struct janas_cuda *c, char *buf, size_t len)
{
    snprintf(buf, len, "%.1f GB the GPU may keep resident (shared with the CPU)",
             (double)c->room / 1e9);
}

size_t janas_cuda_local_room(const struct janas_cuda *c)
{
    return c->room > c->used ? c->room - c->used : 0;
}

/* a buffer on the host's own pages, [p, p + bytes) rounded out to them:
   Metal takes only whole pages (16 KB on Apple's processors) */
static int give(struct janas_cuda *c, const void *p, size_t bytes, int mapped)
{
    size_t pg = vm_page_size;
    uintptr_t a = (uintptr_t)p / pg * pg;
    uintptr_t e = ((uintptr_t)p + bytes + pg - 1) / pg * pg;
    id<MTLBuffer> b =
        [OBJ(c->dev) newBufferWithBytesNoCopy:(void *)a
                                       length:e - a
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
    if (!b)
        return -1;
    struct drv_region *r = drv_add_region(&c->regs);
    if (!r)
        return -1;
    *r = (struct drv_region){.host = p,
                             .bytes = bytes,
                             .dev = (uint64_t)(uintptr_t)keep(b),
                             .dev_off = (uintptr_t)p - a,
                             .mapped = mapped};
    return 0;
}

int janas_cuda_import(struct janas_cuda *c, const void *p, size_t bytes)
{
    if (!p || !bytes || c->busy)
        return -1;
    if (drv_find(&c->regs, p, bytes))
        return 0;
    if (bytes > janas_cuda_local_room(c) || give(c, p, bytes, 0) != 0)
        return -1; /* these rows stay on the CPU */
    c->used += bytes;
    return 0;
}

void *janas_cuda_alloc(struct janas_cuda *c, size_t bytes)
{
    size_t pg = vm_page_size;
    bytes = (bytes + pg - 1) / pg * pg;
    void *p = NULL;
    if (posix_memalign(&p, pg, bytes) != 0)
        return NULL;
    if (give(c, p, bytes, 1) != 0) {
        free(p);
        return NULL;
    }
    return p;
}

int janas_cuda_sync(struct janas_cuda *c, void *p, size_t bytes, int to_device)
{
    (void)to_device; /* one memory: nothing to copy */
    return drv_find(&c->regs, p, bytes) && !c->busy ? 0 : -1;
}

static void *kernel_of(const struct janas_cuda *c, int type)
{
    return type == JANAS_Q4_K         ? c->k_q4k
           : type == JANAS_Q6_K       ? c->k_q6k
           : type == JANAS_Q5_K       ? c->k_q5k
           : type == JANAS_Q3_K       ? c->k_q3k
           : type == JANAS_Q8_0       ? c->k_q80
           : drv_b32_kind(type) >= 0  ? c->k_b32
           : type == JANAS_IQ4_XS     ? c->k_iq4xs
                                      : NULL;
}

int janas_cuda_can(const struct janas_cuda *c,
                   const struct janas_matvec_task *t)
{
    if (c->broken || !kernel_of(c, t->type) || t->cols % JANAS_QK)
        return 0;
    size_t nb = t->cols / JANAS_QK, nv = t->n_vec ? t->n_vec : 1;
    if (nv * nb * GPU_BLOCK > X_STAGE || nv * t->rows * 4 > Y_STAGE)
        return 0;
    return drv_find(&c->regs, t->w,
                    t->rows * janas_qtype_row_bytes(t->type, t->cols)) != NULL;
}

int janas_cuda_begin(struct janas_cuda *c,
                     const struct janas_matvec_task *tasks,
                     const size_t *gpu_rows, size_t n)
{
    if (c->busy || c->broken)
        return -1;
    @autoreleasepool {
        struct {
            const void *x;
            size_t nv, xs, nb, off;
            int k32;
        } xc[MAX_XCACHE];
        size_t n_xc = 0, x_used = 0, y_used = 0, xk[MAX_PENDING];
        c->n_pend = 0;
        /* the activations, each set once */
        for (size_t i = 0; i < n; i++) {
            if (!gpu_rows[i])
                continue;
            const struct janas_matvec_task *t = &tasks[i];
            if (i >= MAX_PENDING || !kernel_of(c, t->type))
                return -1;
            size_t nb = t->cols / JANAS_QK, nv = t->n_vec ? t->n_vec : 1;
            size_t xs = t->x_stride ? t->x_stride : nb, k = 0;
            int k32 = drv_in_32(t->type);
            while (k < n_xc &&
                   !(xc[k].x == t->x && xc[k].nv == nv && xc[k].xs == xs &&
                     xc[k].nb == nb && xc[k].k32 == k32))
                k++;
            if (k == n_xc) {
                size_t bytes = nv * nb * GPU_BLOCK;
                if (n_xc == MAX_XCACHE || x_used + bytes > X_STAGE)
                    return -1;
                drv_convert_x(c->xh + x_used, t->x, xs, nv, nb, k32);
                xc[k].x = t->x;
                xc[k].nv = nv;
                xc[k].xs = xs;
                xc[k].nb = nb;
                xc[k].k32 = k32;
                xc[k].off = x_used;
                x_used += bytes;
                n_xc++;
            }
            xk[i] = xc[k].off;
        }
        id<MTLCommandBuffer> cb = [OBJ(c->queue) commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (size_t i = 0; i < n; i++) {
            if (!gpu_rows[i])
                continue;
            const struct janas_matvec_task *t = &tasks[i];
            size_t nb = t->cols / JANAS_QK, nv = t->n_vec ? t->n_vec : 1;
            size_t rows = gpu_rows[i], ybytes = rows * nv * sizeof(float);
            const struct drv_region *r = drv_find(
                &c->regs, t->w, rows * janas_qtype_row_bytes(t->type, t->cols));
            if (!r || c->n_pend == MAX_PENDING || y_used + ybytes > Y_STAGE) {
                [e endEncoding];
                return -1;
            }
            id<MTLComputePipelineState> ps = OBJ(kernel_of(c, t->type));
            uint32_t a[6] = {(uint32_t)rows, (uint32_t)nb, (uint32_t)nv,
                             (uint32_t)nb, (uint32_t)rows,
                             (uint32_t)drv_b32_kind(t->type)};
            [e setComputePipelineState:ps];
            [e setBuffer:OBJ((void *)(uintptr_t)r->dev)
                  offset:r->dev_off + (size_t)((const uint8_t *)t->w - r->host)
                 atIndex:0];
            [e setBuffer:OBJ(c->xbuf) offset:xk[i] atIndex:1];
            [e setBuffer:OBJ(c->ybuf) offset:y_used atIndex:2];
            for (int k = 0; k < 6; k++)
                [e setBytes:&a[k] length:sizeof a[k] atIndex:3 + k];
            NSUInteger tg = ps.maxTotalThreadsPerThreadgroup < THREADS
                                ? ps.maxTotalThreadsPerThreadgroup
                                : THREADS;
            [e dispatchThreads:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            snprintf(c->last, sizeof c->last, "%s (Metal)",
                     drv_kernel_name(t->type));
            c->pend[c->n_pend++] =
                (struct drv_pending){.y = t->y,
                                     .rows = rows,
                                     .nv = nv,
                                     .y_stride = t->y_stride ? t->y_stride
                                                             : t->rows,
                                     .off = y_used};
            y_used += ybytes;
        }
        [e endEncoding];
        [cb commit];
        c->cmd = keep(cb);
        c->busy = 1;
        return 0;
    }
}

int janas_cuda_finish(struct janas_cuda *c, int *waited)
{
    if (waited)
        *waited = 0;
    if (!c->busy)
        return 0;
    c->busy = 0;
    id<MTLCommandBuffer> cb = (__bridge_transfer id)c->cmd;
    c->cmd = NULL;
    if (waited)
        *waited = cb.status < MTLCommandBufferStatusCompleted;
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        c->broken = 1; /* the GPU failed: the CPU from now on */
        return -1;
    }
    for (size_t i = 0; i < c->n_pend; i++) {
        const struct drv_pending *p = &c->pend[i];
        const float *src = (const float *)((const uint8_t *)c->yh + p->off);
        for (size_t v = 0; v < p->nv; v++)
            memcpy(p->y + v * p->y_stride, src + v * p->rows,
                   p->rows * sizeof(float));
    }
    c->n_pend = 0;
    return 0;
}

#endif /* __APPLE__ */
