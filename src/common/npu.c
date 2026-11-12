/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * npu.c - the NPU of npu.h, through Level Zero loaded at run time.
 *
 * A product is the graph x (f32) -> FP16 -> x w^T -> f32 -> y, with w a
 * constant: INT8 converted to FP16 and multiplied by its row's scale (the
 * compiler folds the dequantization into the product) or FP16. Each product
 * has its graph, a command list recorded once and a fence; begin executes
 * the list on the NPU's one queue, finish waits on the fence.
 *
 * JANAS_NPU_LOADER names the Level Zero loader to open (default
 * libze_loader.so.1): the NPU's driver may need a newer loader than the
 * system's. ZE_INTEL_NPU_CACHE_SIZE, the driver's, is set to 0 unless set.
 */
#include "npu.h"
#include "npu_amd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* No Intel NPU to use: where the machine has one of AMD's, the reason says
   so (npu_amd.c), rather than that there is none. */
static void amd_reason(char *err, size_t err_len)
{
    char d[320];
    if (janas_npu_amd_find(d, sizeof d))
        snprintf(err, err_len, "%s: Janas does not use AMD's NPUs yet", d);
}

#if defined(__linux__)

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sha256.h"

#include "npu_ir.h"
#include "npu_ze.h"

#define NPU_MAX_ALLOCS 1024

#define ZE_FUNCS(X)                                                            \
    X(zeInit, (uint32_t))                                                      \
    X(zeDriverGet, (uint32_t *, ze_driver_handle_t *))                         \
    X(zeDriverGetProperties, (ze_driver_handle_t, ze_driver_properties_t *))   \
    X(zeDeviceGet, (ze_driver_handle_t, uint32_t *, ze_device_handle_t *))     \
    X(zeDeviceGetProperties, (ze_device_handle_t, ze_device_properties_t *))   \
    X(zeContextCreate,                                                         \
      (ze_driver_handle_t, const ze_simple_desc_t *, ze_context_handle_t *))   \
    X(zeContextDestroy, (ze_context_handle_t))                                 \
    X(zeDriverGetExtensionFunctionAddress,                                     \
      (ze_driver_handle_t, const char *, void **))                             \
    X(zeMemAllocHost, (ze_context_handle_t, const ze_simple_desc_t *, size_t,  \
                       size_t, void **))                                       \
    X(zeMemFree, (ze_context_handle_t, void *))                                \
    X(zeCommandQueueCreate,                                                    \
      (ze_context_handle_t, ze_device_handle_t,                                \
       const ze_command_queue_desc_t *, ze_command_queue_handle_t *))          \
    X(zeCommandQueueDestroy, (ze_command_queue_handle_t))                      \
    X(zeCommandQueueSynchronize, (ze_command_queue_handle_t, uint64_t))        \
    X(zeCommandQueueExecuteCommandLists,                                       \
      (ze_command_queue_handle_t, uint32_t, ze_command_list_handle_t *,        \
       ze_fence_handle_t))                                                     \
    X(zeCommandListCreate,                                                     \
      (ze_context_handle_t, ze_device_handle_t,                                \
       const ze_command_list_desc_t *, ze_command_list_handle_t *))            \
    X(zeCommandListDestroy, (ze_command_list_handle_t))                        \
    X(zeCommandListClose, (ze_command_list_handle_t))                          \
    X(zeCommandListReset, (ze_command_list_handle_t))                          \
    X(zeFenceCreate, (ze_command_queue_handle_t, const ze_simple_desc_t *,     \
                      ze_fence_handle_t *))                                    \
    X(zeFenceDestroy, (ze_fence_handle_t))                                     \
    X(zeFenceHostSynchronize, (ze_fence_handle_t, uint64_t))                   \
    X(zeFenceReset, (ze_fence_handle_t))

struct janas_npu {
    void *lib;
#define DECL(f, args) ze_result_t(*f) args;
    ZE_FUNCS(DECL)
#undef DECL
    ze_driver_handle_t drv;
    ze_device_handle_t dev;
    ze_context_handle_t ctx;
    ze_command_queue_handle_t q;
    ze_graph_dditable_ext_t *g;
    uint16_t cmajor, cminor;
    uint32_t driver_version, vendor_id, device_id;
    int driver_known; /* its version read: else no cache, which keys on it */
    char name[256], *cache;
    void *allocs[NPU_MAX_ALLOCS];
    int nallocs;
    struct janas_npu_mm *mms; /* every product, for destroy */
    int products, cached;
    double compile_s;
    /* keeping the NPU awake (janas_npu_keep_awake): a thread runs a tiny
       product on a queue of its own when the NPU has been idle for
       NPU_AWAKE_NS - the kernel suspends it after 10 ms
       (power/autosuspend_delay_ms), and each wake-up cost a product of 256
       x 12288 x 2048 some 5 ms (10.4-11.6 ms after pauses of 20-40 ms,
       5.4-6.6 without; 20 Oct 2026) */
    ze_command_queue_handle_t kq;
    struct janas_npu_mm *ka;
    pthread_t ka_thread;
    int ka_started;
    int ka_failed; /* once: what it made is freed with the NPU, not again */
    atomic_int ka_on, ka_stop;
    /* off, the thread waits here rather than wake every half millisecond:
       2000 times a second beside the cores writing a reply */
    pthread_mutex_t ka_mx;
    pthread_cond_t ka_cv;
    atomic_llong last_ns; /* the last submission or wait */
};

/* 1 ms: with 4 the NPU still lost speed after pauses of 40 ms (9.1 ms a
   product against 7.0 with 1 and 11.6 without; 20 Oct 2026);
   JANAS_NPU_AWAKE_US to try others */
#define NPU_AWAKE_NS 1000000LL

/* the keep-awake threads (below), stopped by janas_npu_destroy too */
#define NPU_AWAKE_MAX 8
static struct janas_npu *awake_npus[NPU_AWAKE_MAX];
static pthread_mutex_t awake_lock;
static void stop_awake(struct janas_npu *u);

struct janas_npu_mm {
    struct janas_npu *u;
    struct janas_npu_mm *next;
    ze_graph_handle_t graph;
    ze_command_list_handle_t list;
    ze_fence_handle_t fence;
    ze_command_queue_handle_t q; /* where it runs */
    /* a second binding (janas_npu_mm_bind_alt): the same graph, another
       list recorded with other arguments, its own fence */
    ze_command_list_handle_t alt_list;
    ze_fence_handle_t alt_fence;
    int alt_pending;
    uint8_t *blob; /* a graph from the cache: the driver reads it later */
    int pending;
};

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

struct janas_npu *janas_npu_create(char *err, size_t err_len)
{
    struct janas_npu *u = calloc(1, sizeof(*u));
    if (!u) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    /* the driver's own cache off, its directory not even made (unless the
       caller set its size): Janas keeps its own (janas_npu_set_cache) */
    setenv("ZE_INTEL_NPU_CACHE_SIZE", "0", 0);
    const char *lib = getenv("JANAS_NPU_LOADER");
    if (!lib || !*lib)
        lib = "libze_loader.so.1";
    u->lib = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!u->lib) {
        snprintf(err, err_len, "no Level Zero loader (%s)", lib);
        amd_reason(err, err_len);
        free(u);
        return NULL;
    }
#define LOAD(f, args)                                                          \
    if (!(*(void **)&u->f = dlsym(u->lib, #f))) {                              \
        snprintf(err, err_len, "Level Zero loader without " #f);               \
        goto fail;                                                             \
    }
    ZE_FUNCS(LOAD)
#undef LOAD
    uint32_t n = 1;
    ze_device_properties_t dp = {.stype = ZE_STYPE_DEVICE_PROPERTIES};
    if (u->zeInit(ZE_INIT_FLAG_VPU_ONLY) || u->zeDriverGet(&n, &u->drv) || !n ||
        (n = 1, u->zeDeviceGet(u->drv, &n, &u->dev)) || !n ||
        u->zeDeviceGetProperties(u->dev, &dp) ||
        dp.type != ZE_DEVICE_TYPE_VPU) {
        snprintf(err, err_len,
                 "no NPU (driver missing, or no access to "
                 "/dev/accel: group render)");
        amd_reason(err, err_len);
        goto fail;
    }
    memcpy(u->name, dp.name, sizeof u->name - 1);
    u->vendor_id = dp.vendorId;
    u->device_id = dp.deviceId;
    ze_driver_properties_t drp = {.stype = ZE_STYPE_DRIVER_PROPERTIES};
    if (!u->zeDriverGetProperties(u->drv, &drp)) {
        u->driver_version = drp.driverVersion;
        u->driver_known = 1;
    }
    ze_simple_desc_t cd = {.stype = ZE_STYPE_CONTEXT_DESC};
    if (u->zeContextCreate(u->drv, &cd, &u->ctx)) {
        snprintf(err, err_len, "NPU context refused");
        goto fail;
    }
    ze_device_graph_properties_t gp = {.stype =
                                           ZE_GSTYPE_DEVICE_GRAPH_PROPERTIES};
    if (u->zeDriverGetExtensionFunctionAddress(u->drv, ZE_GRAPH_EXT_NAME,
                                               (void **)&u->g) ||
        !u->g || u->g->pfnDeviceGetGraphProperties(u->dev, &gp) ||
        gp.graphExtensionVersion < ((1u << 16) | 5u) ||
        !(gp.graphFormatsSupported & ZE_GRAPH_FORMAT_NGRAPH_LITE)) {
        snprintf(err, err_len,
                 "NPU driver without its graph compiler "
                 "(extension 1.5 and IR input needed)");
        goto fail;
    }
    u->cmajor = gp.compilerMajor, u->cminor = gp.compilerMinor;
    /* turbo (unless JANAS_NPU_TURBO=0): the queue's work at the NPU's
       higher clocks */
    const char *tb = getenv("JANAS_NPU_TURBO");
    ze_command_queue_desc_npu_ext_t qx = {
        .stype = ZE_NSTYPE_COMMAND_QUEUE_DESC_NPU_EXT,
        .turbo = !tb || atoi(tb) > 0};
    ze_command_queue_desc_t qd = {.stype = ZE_STYPE_COMMAND_QUEUE_DESC,
                                  .pNext = &qx};
    if (u->zeCommandQueueCreate(u->ctx, u->dev, &qd, &u->q) &&
        (qd.pNext = NULL,
         u->zeCommandQueueCreate(u->ctx, u->dev, &qd, &u->q))) {
        snprintf(err, err_len, "NPU queue refused");
        goto fail;
    }
    return u;
fail:
    if (u->ctx)
        u->zeContextDestroy(u->ctx);
    dlclose(u->lib);
    free(u);
    return NULL;
}

void janas_npu_mm_destroy(struct janas_npu_mm *p)
{
    if (!p)
        return;
    struct janas_npu *u = p->u;
    if (p->pending)
        janas_npu_mm_finish(p);
    if (p->alt_pending)
        janas_npu_mm_finish_v(p, 1);
    if (p->alt_fence)
        u->zeFenceDestroy(p->alt_fence);
    if (p->alt_list)
        u->zeCommandListDestroy(p->alt_list);
    for (struct janas_npu_mm **a = &u->mms; *a; a = &(*a)->next)
        if (*a == p) {
            *a = p->next;
            break;
        }
    if (p->fence)
        u->zeFenceDestroy(p->fence);
    if (p->list)
        u->zeCommandListDestroy(p->list);
    if (p->graph)
        u->g->pfnDestroy(p->graph);
    free(p->blob);
    free(p);
}

void janas_npu_destroy(struct janas_npu *u)
{
    if (!u)
        return;
    pthread_mutex_lock(&awake_lock);
    stop_awake(u);
    for (int i = 0; i < NPU_AWAKE_MAX; i++)
        if (awake_npus[i] == u)
            awake_npus[i] = NULL;
    pthread_mutex_unlock(&awake_lock);
    while (u->mms)
        janas_npu_mm_destroy(u->mms);
    for (int i = 0; i < u->nallocs; i++)
        u->zeMemFree(u->ctx, u->allocs[i]);
    u->zeCommandQueueDestroy(u->q);
    if (u->kq)
        u->zeCommandQueueDestroy(u->kq);
    u->zeContextDestroy(u->ctx);
    dlclose(u->lib);
    free(u->cache);
    free(u);
}

int janas_npu_set_cache(struct janas_npu *u, const char *dir)
{
    free(u->cache);
    u->cache = NULL;
    if (!dir || !*dir)
        return 0;
    if (mkdir(dir, 0755) && errno != EEXIST)
        return -1;
    u->cache = strdup(dir);
    return u->cache ? 0 : -1;
}

/* the cache's file of a model: its SHA-256 with the device's ids and the
   driver's and the compiler's versions (a graph compiled for one NPU is
   not for another) */
static void cache_path(const struct janas_npu *u, const uint8_t *model,
                       size_t bytes, char *path, size_t len)
{
    struct janas_sha256 h;
    char hex[65];
    janas_sha256_init(&h);
    janas_sha256_update(&h, model, bytes);
    janas_sha256_update(&h, &u->vendor_id, sizeof u->vendor_id);
    janas_sha256_update(&h, &u->device_id, sizeof u->device_id);
    janas_sha256_update(&h, &u->driver_version, sizeof u->driver_version);
    janas_sha256_update(&h, &u->cmajor, sizeof u->cmajor);
    janas_sha256_update(&h, &u->cminor, sizeof u->cminor);
    janas_sha256_hex(&h, hex);
    snprintf(path, len, "%s/%s.npu", u->cache, hex);
}

static uint8_t *read_file(const char *path, size_t *bytes)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    uint8_t *b = NULL;
    long n = fseek(f, 0, SEEK_END) ? -1 : ftell(f);
    if (n > 0 && !fseek(f, 0, SEEK_SET) && (b = malloc((size_t)n)) &&
        fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        b = NULL;
    }
    fclose(f);
    *bytes = n > 0 ? (size_t)n : 0;
    return b;
}

/* the compiled graph into the cache: written aside, then renamed */
static void cache_store(struct janas_npu *u, ze_graph_handle_t graph,
                        const char *path)
{
    size_t n = 0;
    if (u->g->pfnGetNativeBinary(graph, &n, NULL) || !n)
        return;
    uint8_t *b = malloc(n);
    /* its own name: two NPUs of a process (a model and its drafts) may
       store the same graph at once */
    static _Atomic unsigned seq;
    char tmp[4200];
    snprintf(tmp, sizeof tmp, "%s.%d.%u.tmp", path, (int)getpid(), ++seq);
    FILE *f =
        b && !u->g->pfnGetNativeBinary(graph, &n, b) ? fopen(tmp, "wb") : NULL;
    if (f) {
        int ok = fwrite(b, 1, n, f) == n;
        ok = !fclose(f) && ok;
        if (!ok || rename(tmp, path))
            unlink(tmp);
    }
    free(b);
}

const char *janas_npu_name(const struct janas_npu *u)
{
    return u->name;
}

void *janas_npu_alloc(struct janas_npu *u, size_t bytes)
{
    void *p = NULL;
    ze_simple_desc_t hd = {.stype = ZE_STYPE_HOST_MEM_ALLOC_DESC};
    if (u->nallocs >= NPU_MAX_ALLOCS ||
        u->zeMemAllocHost(u->ctx, &hd, (bytes + 4095) & ~(size_t)4095, 4096,
                          &p))
        return NULL;
    u->allocs[u->nallocs++] = p;
    return p;
}

void janas_npu_free(struct janas_npu *u, void *p)
{
    for (int i = 0; i < u->nallocs; i++)
        if (u->allocs[i] == p) {
            u->zeMemFree(u->ctx, p);
            u->allocs[i] = u->allocs[--u->nallocs];
            return;
        }
}

/* the graph's arguments, in its order (inputs, then outputs), each rows
   x cols as expected, bound to args */
static int bind(struct janas_npu_mm *p, int nargs, float *const *args,
                const int *rows, const int *cols)
{
    struct janas_npu *u = p->u;
    ze_graph_properties_t pr = {.stype = ZE_GSTYPE_GRAPH_PROPERTIES};
    int say = getenv("JANAS_NPU_STATS") != NULL;
    if (u->g->pfnGetProperties(p->graph, &pr) ||
        pr.numGraphArgs != (uint32_t)nargs) {
        if (say)
            fprintf(stderr, "janas: NPU graph with %u arguments, %d expected\n",
                    pr.numGraphArgs, nargs);
        return -1;
    }
    for (uint32_t i = 0; i < (uint32_t)nargs; i++) {
        ze_graph_argument_properties_t ap = {
            .stype = ZE_GSTYPE_GRAPH_ARGUMENT_PROPERTIES};
        if (u->g->pfnGetArgumentProperties(p->graph, i, &ap) ||
            ap.dims[0] != (uint32_t)rows[i] ||
            ap.dims[1] != (uint32_t)cols[i] ||
            u->g->pfnSetArgumentValue(p->graph, i, args[i])) {
            if (say) /* which one, as the driver sees it */
                fprintf(stderr,
                        "janas: NPU argument %u of %d (%s, %s): dims %u %u "
                        "%u %u, expected %d x %d\n",
                        i, nargs, ap.name,
                        ap.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT ? "in" : "out",
                        ap.dims[0], ap.dims[1], ap.dims[2], ap.dims[3], rows[i],
                        cols[i]);
            return -1;
        }
    }
    return 0;
}

static struct janas_npu_mm *from_ir(struct janas_npu *u, struct npu_ir *irp,
                                    int nargs, float *const *args,
                                    const int *rows, const int *cols, char *err,
                                    size_t err_len);

struct janas_npu_mm *janas_npu_graph_create(struct janas_npu *u,
                                            struct npu_ir *ir, int nargs,
                                            float *const *args, const int *rows,
                                            const int *cols, char *err,
                                            size_t err_len)
{
    return from_ir(u, ir, nargs, args, rows, cols, err, err_len);
}

struct janas_npu_mm *janas_npu_ffn_create(struct janas_npu *u, int m, int d,
                                          int ff, int act, const int8_t *gq,
                                          const uint16_t *gs, const int8_t *uq,
                                          const uint16_t *us, const int8_t *dq,
                                          const uint16_t *ds, float *x,
                                          float *y, char *err, size_t err_len)
{
    if (m <= 0 || d <= 0 || ff <= 0 || !gq || !gs || !uq || !us || !dq || !ds ||
        !x || !y || (act != JANAS_NPU_SILU && act != JANAS_NPU_GELU_TANH)) {
        snprintf(err, err_len, "bad feed-forward");
        return NULL;
    }
    struct npu_ir ir;
    npu_ir_begin(&ir, "janas_ffn");
    int xd[2] = {m, d};
    int lx =
        npu_ir_convert(&ir, "x16", npu_ir_param(&ir, "x", "f32", 2, xd), "f16");
    int lf =
        npu_ir_ffn(&ir, "ffn", lx, npu_ir_weights_i8(&ir, "wg", gq, gs, ff, d),
                   npu_ir_weights_i8(&ir, "wu", uq, us, ff, d),
                   npu_ir_weights_i8(&ir, "wd", dq, ds, d, ff),
                   act == JANAS_NPU_GELU_TANH);
    int ly = npu_ir_convert(&ir, "y", lf, "f32");
    npu_ir_result(&ir, ly);
    float *args[2] = {x, y};
    int rows[2] = {m, m}, cols[2] = {d, d};
    return from_ir(u, &ir, 2, args, rows, cols, err, err_len);
}

struct janas_npu_mm *janas_npu_mm_create(struct janas_npu *u, int m, int n,
                                         int k, int wtype, const void *w,
                                         const uint16_t *scale, float *x,
                                         float *y, char *err, size_t err_len)
{
    if (m <= 0 || n <= 0 || k <= 0 || !w || !x || !y ||
        (wtype == JANAS_NPU_I8 && !scale) ||
        (wtype != JANAS_NPU_I8 && wtype != JANAS_NPU_F16)) {
        snprintf(err, err_len, "bad product");
        return NULL;
    }
    struct npu_ir ir;
    npu_ir_begin(&ir, "janas_mm");
    int xd[2] = {m, k}, wd[2] = {n, k}, sd[2] = {n, 1};
    int lx =
        npu_ir_convert(&ir, "x16", npu_ir_param(&ir, "x", "f32", 2, xd), "f16");
    int lw;
    if (wtype == JANAS_NPU_I8) {
        int lq = npu_ir_const(&ir, "wq", "i8", 2, wd, w, (size_t)n * k);
        int ls = npu_ir_const(&ir, "ws", "f16", 2, sd, scale, (size_t)n * 2);
        lw =
            npu_ir_multiply(&ir, "w", npu_ir_convert(&ir, "wc", lq, "f16"), ls);
    } else
        lw = npu_ir_const(&ir, "w", "f16", 2, wd, w, (size_t)n * k * 2);
    int ly = npu_ir_convert(&ir, "y", npu_ir_matmul_tb(&ir, "mm", lx, lw, NULL),
                            "f32");
    npu_ir_result(&ir, ly);
    float *args[2] = {x, y};
    int rows[2] = {m, m}, cols[2] = {k, n};
    return from_ir(u, &ir, 2, args, rows, cols, err, err_len);
}

/* a graph written in ir (freed here) compiled - or loaded from the cache -
   with x (m x k floats) and y (m x n floats) bound, its command list
   recorded and a first run made */
static struct janas_npu_mm *from_ir(struct janas_npu *u, struct npu_ir *irp,
                                    int nargs, float *const *args,
                                    const int *rows, const int *cols, char *err,
                                    size_t err_len)
{
    struct npu_ir ir = *irp;
    size_t bytes = 0;
    uint8_t *model = npu_ir_end(&ir)
                         ? NULL
                         : npu_ir_model(&ir, u->cmajor, u->cminor, &bytes);
    npu_ir_free(&ir);
    struct janas_npu_mm *p = model ? calloc(1, sizeof(*p)) : NULL;
    if (!p) {
        free(model);
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    p->u = u;
    p->q = u->q;
    ze_graph_desc_2_t gd = {.stype = ZE_GSTYPE_GRAPH_DESC_2,
                            .format = ZE_GRAPH_FORMAT_NGRAPH_LITE,
                            .inputSize = bytes,
                            .pInput = model,
                            .pBuildFlags = "",
                            .flags = ZE_GRAPH_FLAG_DISABLE_CACHING};
    double t0 = now();
    ze_result_t r = 1;
    char path[4096];
    if (u->cache && u->driver_known) {
        cache_path(u, model, bytes, path, sizeof path);
        size_t nb = 0;
        uint8_t *blob = read_file(path, &nb);
        if (blob) {
            ze_graph_desc_2_t nd = gd;
            nd.format = ZE_GRAPH_FORMAT_NATIVE, nd.inputSize = nb;
            nd.pInput = blob, nd.pBuildFlags = NULL;
            r = u->g->pfnCreate2(u->ctx, u->dev, &nd, &p->graph);
            if (r) {
                free(blob);
                unlink(path); /* stale: another driver's */
            } else
                p->blob = blob, u->cached++;
        }
    }
    if (r) {
        r = u->g->pfnCreate2(u->ctx, u->dev, &gd, &p->graph);
        if (!r && u->cache && u->driver_known)
            cache_store(u, p->graph, path);
    }
    free(model);
    if (r) {
        snprintf(err, err_len, "NPU compiler refused the product (0x%x)",
                 (unsigned)r);
        free(p);
        return NULL;
    }
    u->compile_s += now() - t0;
    u->products++;
    p->next = u->mms, u->mms = p;
    ze_command_list_desc_t ld = {.stype = ZE_STYPE_COMMAND_LIST_DESC};
    ze_simple_desc_t fd = {.stype = ZE_STYPE_FENCE_DESC};
    /* step by step, so that a failure says where */
    int step = 0;
    ze_result_t sr = 0;
    if (bind(p, nargs, args, rows, cols))
        step = 1;
    else if ((sr = u->zeCommandListCreate(u->ctx, u->dev, &ld, &p->list)))
        step = 2;
    else if ((sr = u->g->pfnAppendGraphInitialize(p->list, p->graph, NULL, 0,
                                                  NULL)))
        step = 3;
    else if ((sr = u->zeCommandListClose(p->list)) ||
             (sr = u->zeCommandQueueExecuteCommandLists(u->q, 1, &p->list,
                                                        NULL)) ||
             (sr = u->zeCommandQueueSynchronize(u->q, UINT64_MAX)))
        step = 4;
    else if ((sr = u->zeCommandListReset(p->list)) ||
             (sr = u->g->pfnAppendGraphExecute(p->list, p->graph, NULL, NULL, 0,
                                               NULL)) ||
             (sr = u->zeCommandListClose(p->list)))
        step = 5;
    else if ((sr = u->zeFenceCreate(u->q, &fd, &p->fence)))
        step = 6;
    if (step) {
        static const char *what[7] = {"",
                                      "arguments not as expected",
                                      "no command list",
                                      "initialization not recorded",
                                      "initialization failed",
                                      "run not recorded",
                                      "no fence"};
        snprintf(err, err_len, "NPU product not set up: %s (0x%x)", what[step],
                 (unsigned)sr);
        janas_npu_mm_destroy(p);
        return NULL;
    }
    /* a first run: the NPU's first one after a compilation costs several
       times the next (512 x 1024 x 1024 INT8: 4.9 ms, 2.2, then 0.8-1.3;
       20 Oct 2026), counted with the compilation */
    t0 = now();
    if (janas_npu_mm_run(p)) {
        snprintf(err, err_len, "NPU product does not run");
        janas_npu_mm_destroy(p);
        return NULL;
    }
    u->compile_s += now() - t0;
    return p;
}

int janas_npu_mm_rebind(struct janas_npu_mm *p, int nargs, float *const *args)
{
    struct janas_npu *u = p->u;
    if (p->pending)
        return -1;
    for (int i = 0; i < nargs; i++)
        if (u->g->pfnSetArgumentValue(p->graph, (uint32_t)i, args[i]))
            return -1;
    /* the arguments are read when the list is recorded (20 Oct 2026: a
       new x set after it ran on the old one): recorded again, ~90 us */
    if (u->zeCommandListReset(p->list) ||
        u->g->pfnAppendGraphExecute(p->list, p->graph, NULL, NULL, 0, NULL) ||
        u->zeCommandListClose(p->list))
        return -1;
    return 0;
}

int janas_npu_mm_bind_alt(struct janas_npu_mm *p, int nargs, float *const *args)
{
    struct janas_npu *u = p->u;
    ze_command_list_desc_t ld = {.stype = ZE_STYPE_COMMAND_LIST_DESC};
    ze_simple_desc_t fd = {.stype = ZE_STYPE_FENCE_DESC};
    if (p->alt_pending)
        return -1;
    for (int i = 0; i < nargs; i++)
        if (u->g->pfnSetArgumentValue(p->graph, (uint32_t)i, args[i]))
            return -1;
    if ((!p->alt_list &&
         u->zeCommandListCreate(u->ctx, u->dev, &ld, &p->alt_list)) ||
        u->zeCommandListReset(p->alt_list) ||
        u->g->pfnAppendGraphExecute(p->alt_list, p->graph, NULL, NULL, 0,
                                    NULL) ||
        u->zeCommandListClose(p->alt_list) ||
        (!p->alt_fence && u->zeFenceCreate(p->q, &fd, &p->alt_fence)))
        return -1;
    return 0;
}

int janas_npu_mm_begin_v(struct janas_npu_mm *p, int v)
{
    struct janas_npu *u = p->u;
    int *pending = v ? &p->alt_pending : &p->pending;
    ze_command_list_handle_t *l = v ? &p->alt_list : &p->list;
    if (*pending || !*l ||
        u->zeCommandQueueExecuteCommandLists(p->q, 1, l,
                                             v ? p->alt_fence : p->fence))
        return -1;
    *pending = 1;
    atomic_store(&u->last_ns, (long long)(now() * 1e9));
    return 0;
}

int janas_npu_mm_finish_v(struct janas_npu_mm *p, int v)
{
    struct janas_npu *u = p->u;
    int *pending = v ? &p->alt_pending : &p->pending;
    ze_fence_handle_t f = v ? p->alt_fence : p->fence;
    if (!*pending)
        return -1;
    *pending = 0;
    int r =
        u->zeFenceHostSynchronize(f, UINT64_MAX) || u->zeFenceReset(f) ? -1 : 0;
    atomic_store(&u->last_ns, (long long)(now() * 1e9));
    return r;
}

int janas_npu_mm_begin(struct janas_npu_mm *p)
{
    return janas_npu_mm_begin_v(p, 0);
}

int janas_npu_mm_finish(struct janas_npu_mm *p)
{
    return janas_npu_mm_finish_v(p, 0);
}

/* the threads keeping NPUs awake, stopped at exit: a program that ends
   without janas_npu_destroy left one running while the driver was torn
   down, and crashed on its way out */
static pthread_mutex_t awake_lock = PTHREAD_MUTEX_INITIALIZER;

static void stop_awake(struct janas_npu *u)
{
    if (!u->ka_started)
        return;
    atomic_store(&u->ka_stop, 1);
    pthread_mutex_lock(&u->ka_mx);
    pthread_cond_broadcast(&u->ka_cv);
    pthread_mutex_unlock(&u->ka_mx);
    pthread_join(u->ka_thread, NULL);
    pthread_cond_destroy(&u->ka_cv);
    pthread_mutex_destroy(&u->ka_mx);
    u->ka_started = 0;
}

static void stop_all_awake(void)
{
    pthread_mutex_lock(&awake_lock);
    for (int i = 0; i < NPU_AWAKE_MAX; i++)
        if (awake_npus[i]) {
            stop_awake(awake_npus[i]);
            awake_npus[i] = NULL;
        }
    pthread_mutex_unlock(&awake_lock);
}

static void *keep_awake(void *arg)
{
    struct janas_npu *u = arg;
    const char *e = getenv("JANAS_NPU_AWAKE_US");
    long long idle = e && atol(e) > 0 ? atoll(e) * 1000 : NPU_AWAKE_NS;
    struct timespec nap = {.tv_nsec = idle < 2000000 ? idle / 2 : 1000000};
    while (!atomic_load(&u->ka_stop)) {
        if (!atomic_load(&u->ka_on)) { /* asleep until a pass needs it */
            pthread_mutex_lock(&u->ka_mx);
            while (!atomic_load(&u->ka_on) && !atomic_load(&u->ka_stop))
                pthread_cond_wait(&u->ka_cv, &u->ka_mx);
            pthread_mutex_unlock(&u->ka_mx);
            continue;
        }
        nanosleep(&nap, NULL);
        long long t = (long long)(now() * 1e9);
        if (atomic_load(&u->ka_on) && t - atomic_load(&u->last_ns) > idle &&
            janas_npu_mm_begin(u->ka) == 0)
            janas_npu_mm_finish(u->ka);
    }
    return NULL;
}

int janas_npu_keep_awake(struct janas_npu *u, int on)
{
    int was = atomic_exchange(&u->ka_on, on);
    if (u->ka_started) {
        if (on && !was) { /* the thread asleep: woken */
            pthread_mutex_lock(&u->ka_mx);
            pthread_cond_signal(&u->ka_cv);
            pthread_mutex_unlock(&u->ka_mx);
        }
        return 0;
    }
    if (!on)
        return 0;
    if (u->ka_failed) {
        atomic_store(&u->ka_on, 0);
        return -1;
    }
    u->ka_failed = 1; /* until it is running */
    /* the tiny product, its fence moved to a queue of its own */
    char err[64];
    ze_command_queue_desc_t qd = {.stype = ZE_STYPE_COMMAND_QUEUE_DESC};
    ze_simple_desc_t fd = {.stype = ZE_STYPE_FENCE_DESC};
    static const int8_t w[64 * 64];
    static const uint16_t s[64];
    float *x = janas_npu_alloc(u, sizeof(float) * 8 * 64),
          *y = janas_npu_alloc(u, sizeof(float) * 8 * 64);
    int compiled = u->products;
    double cs = u->compile_s;
    if (!x || !y || u->zeCommandQueueCreate(u->ctx, u->dev, &qd, &u->kq) ||
        !(u->ka = janas_npu_mm_create(u, 8, 64, 64, JANAS_NPU_I8, w, s, x, y,
                                      err, sizeof err))) {
        atomic_store(&u->ka_on, 0);
        return -1;
    }
    u->products = compiled, u->compile_s = cs; /* not the caller's */
    u->zeFenceDestroy(u->ka->fence);
    u->ka->fence = NULL;
    if (u->zeFenceCreate(u->kq, &fd, &u->ka->fence)) {
        atomic_store(&u->ka_on, 0);
        return -1;
    }
    u->ka->q = u->kq;
    pthread_mutex_init(&u->ka_mx, NULL);
    pthread_cond_init(&u->ka_cv, NULL);
    pthread_mutex_lock(&awake_lock);
    static int registered;
    int slot = -1;
    for (int i = 0; i < NPU_AWAKE_MAX && slot < 0; i++)
        if (!awake_npus[i])
            slot = i;
    if (slot < 0 || (!registered && atexit(stop_all_awake)) ||
        pthread_create(&u->ka_thread, NULL, keep_awake, u)) {
        pthread_mutex_unlock(&awake_lock);
        pthread_cond_destroy(&u->ka_cv);
        pthread_mutex_destroy(&u->ka_mx);
        atomic_store(&u->ka_on, 0);
        return -1;
    }
    registered = 1;
    awake_npus[slot] = u;
    u->ka_started = 1;
    u->ka_failed = 0;
    pthread_mutex_unlock(&awake_lock);
    return 0;
}

void janas_npu_stats(const struct janas_npu *u, int *products, int *cached,
                     double *compile_seconds)
{
    *products = u->products;
    *cached = u->cached;
    *compile_seconds = u->compile_s;
}

#else /* no Level Zero here */

struct janas_npu *janas_npu_create(char *err, size_t err_len)
{
    snprintf(err, err_len, "no NPU support on this system");
    amd_reason(err, err_len);
    return NULL;
}
void janas_npu_destroy(struct janas_npu *u)
{
    (void)u;
}
const char *janas_npu_name(const struct janas_npu *u)
{
    (void)u;
    return "";
}
void *janas_npu_alloc(struct janas_npu *u, size_t bytes)
{
    (void)u, (void)bytes;
    return NULL;
}
void janas_npu_free(struct janas_npu *u, void *p)
{
    (void)u, (void)p;
}
struct janas_npu_mm *janas_npu_mm_create(struct janas_npu *u, int m, int n,
                                         int k, int wtype, const void *w,
                                         const uint16_t *scale, float *x,
                                         float *y, char *err, size_t err_len)
{
    (void)u, (void)m, (void)n, (void)k, (void)wtype, (void)w, (void)scale;
    (void)x, (void)y;
    snprintf(err, err_len, "no NPU support on this system");
    return NULL;
}
void janas_npu_mm_destroy(struct janas_npu_mm *p)
{
    (void)p;
}
struct janas_npu_mm *janas_npu_graph_create(struct janas_npu *u,
                                            struct npu_ir *ir, int nargs,
                                            float *const *args, const int *rows,
                                            const int *cols, char *err,
                                            size_t err_len)
{
    (void)u, (void)ir, (void)nargs, (void)args, (void)rows, (void)cols;
    snprintf(err, err_len, "no NPU support on this system");
    return NULL;
}
struct janas_npu_mm *janas_npu_ffn_create(struct janas_npu *u, int m, int d,
                                          int ff, int act, const int8_t *gq,
                                          const uint16_t *gs, const int8_t *uq,
                                          const uint16_t *us, const int8_t *dq,
                                          const uint16_t *ds, float *x,
                                          float *y, char *err, size_t err_len)
{
    (void)u, (void)m, (void)d, (void)ff, (void)act, (void)gq, (void)gs;
    (void)uq, (void)us, (void)dq, (void)ds, (void)x, (void)y;
    snprintf(err, err_len, "no NPU support on this system");
    return NULL;
}
int janas_npu_mm_rebind(struct janas_npu_mm *p, int nargs, float *const *args)
{
    (void)p, (void)nargs, (void)args;
    return -1;
}
int janas_npu_mm_bind_alt(struct janas_npu_mm *p, int nargs, float *const *args)
{
    (void)p, (void)nargs, (void)args;
    return -1;
}
int janas_npu_mm_begin_v(struct janas_npu_mm *p, int v)
{
    (void)p, (void)v;
    return -1;
}
int janas_npu_mm_finish_v(struct janas_npu_mm *p, int v)
{
    (void)p, (void)v;
    return -1;
}
int janas_npu_mm_begin(struct janas_npu_mm *p)
{
    (void)p;
    return -1;
}
int janas_npu_mm_finish(struct janas_npu_mm *p)
{
    (void)p;
    return -1;
}
int janas_npu_set_cache(struct janas_npu *u, const char *dir)
{
    (void)u, (void)dir;
    return -1;
}
int janas_npu_keep_awake(struct janas_npu *u, int on)
{
    (void)u, (void)on;
    return -1;
}
void janas_npu_stats(const struct janas_npu *u, int *products, int *cached,
                     double *compile_seconds)
{
    (void)u;
    *products = *cached = 0;
    *compile_seconds = 0;
}

#endif

int janas_npu_available(void)
{
    static int known = -1;
    if (known < 0) {
        char err[256];
        struct janas_npu *u = janas_npu_create(err, sizeof err);
        known = u != NULL;
        janas_npu_destroy(u);
    }
    return known;
}

int janas_npu_mm_run(struct janas_npu_mm *p)
{
    return janas_npu_mm_begin(p) ? -1 : janas_npu_mm_finish(p);
}

/* FP16 by hand (round to nearest even), so that this file needs no F16C */
static uint16_t to_f16(float f)
{
    uint32_t b;
    memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000u, e = (b >> 23) & 0xff,
             mant = b & 0x7fffffu;
    if (e == 0xff)
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    int ee = (int)e - 127 + 15;
    if (ee >= 31)
        return (uint16_t)(sign | 0x7c00u);
    if (ee <= 0) {
        if (ee < -10)
            return (uint16_t)sign;
        mant |= 0x800000u;
        int sh = 14 - ee;
        uint32_t v = mant >> sh, rest = mant & ((1u << sh) - 1),
                 half = 1u << (sh - 1);
        if (rest > half || (rest == half && (v & 1)))
            v++;
        return (uint16_t)(sign | v);
    }
    uint32_t v = ((uint32_t)ee << 10) | (mant >> 13), rest = mant & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (v & 1)))
        v++; /* may carry into the exponent: still right */
    return (uint16_t)(sign | v);
}

static float from_f16(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31,
             mant = h & 0x3ffu, b;
    if (e == 0) {
        if (!mant)
            b = sign;
        else {
            e = 127 - 15 + 1;
            while (!(mant & 0x400u))
                mant <<= 1, e--;
            b = sign | (e << 23) | ((mant & 0x3ffu) << 13);
        }
    } else if (e == 31)
        b = sign | 0x7f800000u | (mant << 13);
    else
        b = sign | ((e - 15 + 127) << 23) | (mant << 13);
    float f;
    memcpy(&f, &b, 4);
    return f;
}

void janas_npu_quant_rows(const float *w, size_t n, size_t k, int8_t *q,
                          uint16_t *scale)
{
    for (size_t r = 0; r < n; r++) {
        const float *p = w + r * k;
        float a = 0;
        for (size_t t = 0; t < k; t++)
            a = fmaxf(a, fabsf(p[t]));
        /* the scale rounded up, so that no weight is clipped (rows of tiny
           weights get a subnormal FP16 scale) */
        uint16_t sh = to_f16(a / 127.0f);
        if (from_f16(sh) * 127.0f < a)
            sh++;
        float s = from_f16(sh);
        scale[r] = sh;
        for (size_t t = 0; t < k; t++) {
            long v = s > 0 ? lrintf(p[t] / s) : 0;
            q[r * k + t] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
        }
    }
}
