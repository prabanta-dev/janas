/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_npu.c - the products of a block on the NPU (common/npu.h): looked
 * for when a model opens, JANAS_NPU=0 to leave it alone; the tuner gives it
 * the blocks where it measures faster.
 *
 * Which, for any model: a product over JANAS_NPU_MIN_N vectors and more
 * (default 64) whose weights stay where they are - in the resident region
 * (attention, recurrent layers, shared experts, the output head), or in the
 * experts' arena once it holds every expert and nothing changes there (a
 * dense model's feed-forwards) - in a type the engine dequantizes, while
 * the copies for the NPU fit JANAS_NPU_MAX_MB (default 4096). The routed
 * experts of a mixture whose experts do not all fit, a few tokens each,
 * stay on the CPU, which computes them while the NPU works. A matrix is
 * compiled the first time a block uses it (INT8, a scale a row, from its
 * rows dequantized), for blocks of the model's largest, and kept in
 * ~/.cache/janas/npu (JANAS_NPU_CACHE=<dir> elsewhere, empty for none) for
 * the next load: prepared once. Only in the passes whose tuner candidate has the NPU (npu_use):
 * the tuner and janas-bench weigh it against the CPU and the GPU.
 *
 * x reaches the NPU as the CPU's products see it (its Q8_K blocks expanded
 * to floats); the NPU computes in FP16, so the results are close to the
 * CPU's, not the same bits.
 */
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/npu.h"
#include "common/npu_ir.h"
#include "expert_cache.h"
#include "gpu.h"
#include "model_impl.h"
#include "quant.h"
#include "vmath.h"

/* the inputs of a layer's products, for INT8 activations: the attention's
   output (wo), the FFN's norm (gate and up), the FFN's intermediate
   (down), a layer's norm before the attention (q, k, v) */
enum { SITE_ATT, SITE_FFN, SITE_MID, SITE_QKV, N_SITES };

#define NPU_SLOTS 4096 /* matrices: an open-addressed table on w */
#define NPU_XBUFS 16   /* inputs, one per distinct width */

struct npu_entry {
    const void *w;        /* NULL: free slot */
    const void *plane[2]; /* Q6_K_P: the planes it was compiled with */
    struct janas_npu_mm *mm;
    float *y;
    size_t elems; /* rows x cols, against the budget */
    int bad;      /* could not be compiled: CPU */
    int in_arena; /* from the experts' arena: stale when it changes */
};

struct janas_m_npu {
    struct janas_npu *u;
    int block, min_n;
    size_t budget, used; /* bytes of INT8 copies allowed, made */
    /* the experts' arena as seen when this pass began, the cache idle
       (janas_expert_cache_resident says no between its begin and finish,
       where the feed-forwards' products are): NULL if it was not whole or
       not idle; its loads then, and those the arena's matrices were
       compiled at */
    const uint8_t *arena;
    uint64_t arena_bytes, arena_loads, compiled_loads;
    struct npu_entry e[NPU_SLOTS];
    struct {
        size_t cols;
        float *x;
        const struct janas_block_q8k *src; /* what x holds now */
        size_t n_vec;
    } xb[NPU_XBUFS];
    int nx;
    double compile_s;
    uint64_t runs, cpu_fallbacks;
    /* products left to the CPU, by reason: too few vectors, type or
       layout, weights that may change */
    uint64_t no_n, no_type, no_stable;
    /* seconds: inputs expanded, the CPU's share of the groups, waiting for
       the NPU after it, outputs copied (JANAS_NPU_STATS) */
    double t_expand, t_cpu, t_wait, t_copy;
    /* the matrices looked up (and the first time loaded or compiled), the
       runs begun, and the whole of the products' calls that use the NPU */
    double t_entry, t_begin, t_calls;
    /* by shape (JANAS_NPU_STATS=2): runs and their wait, and the products
       refused with that shape, by reason (0 vectors, 1 type or layout,
       2 weights that may change) */
    struct {
        size_t rows, cols;
        int type;
        uint64_t runs, refused[3];
        double wait;
    } sh[32];
    int nsh;
    /* the products' outputs, shared by the matrices of as many rows that
       are never in flight together: by rows and the order among those of
       the same rows in a group (q's, k's and v's, k and v alike); a
       buffer a matrix took 2.3 GB on Qwen3-4B */
    struct {
        size_t rows;
        int ord;
        float *y;
    } yb[64];
    int ny;
    /* the feed-forwards in one run (janas_m_npu_ffn_begin): their input
       and output by width, one in flight */
    struct {
        size_t d;
        float *x, *y;
    } fb[4];
    int nf;
    uint64_t ffn_runs, ffn_refused, y_clash;
    double t_ffn_wait;
    /* seconds spent making the NPU's graphs - the weights dequantized and
       requantized, the graphs written, compiled and run once: kept out of
       the tuner's costs and janas-bench's rates (the first row with the
       NPU measured Qwen3-4B's prompt at 54 tokens/s for 330 with only the
       compilation taken out, 20 Oct 2026) */
    double setup_s;
    /* the preparation as another thread watches it (janas_llm_npu_preparing):
       on from the first graph a pass makes to the pass's end, or through a
       making of the whole layers' graphs; the graphs (or matrices) made of
       those to make (0: not known ahead, the products made as the block
       meets them); when it began; and the microseconds of the session's
       preparations before it */
    atomic_int prep_on, prep_done, prep_total;
    atomic_llong prep_t0_us, prep_us;
    /* whole layers (janas_m_npu_layers): a graph for the first layer's
       norm and input products, then one a layer - its attention's (or
       recurrence's) output in, wo (ssm_out), the residual, the FFN's norm
       and the FFN, the residual, and the next layer's norm and input
       products - the residual between xs[0] and xs[1] by turns; the first
       nl layers, as many as the NPU holds; -1 not possible here, 0 not yet
       made, 1 made */
    int lmode;
    uint32_t nl;
    /* the first layers alone when fewer than nine in ten fit
       (JANAS_NPU_LAYERS=2): slower than the products on all of them where
       measured - Qwen3.5-9B, 17 layers of 32, 55.4 tokens/s against 58.9;
       Gemma 4 E4B holds 41 of 42 (21 Oct 2026) */
    int partial;
    struct janas_npu_mm *lpre, **lg;
    /* the block in two halves of hb rows, each its buffers (the graphs'
       two bindings): while the NPU runs a layer on one, the CPU does the
       attention or the recurrence of the other. An attention layer's
       inputs in q (with the output gates where there are, as m->big holds
       them), k, v; a recurrent one's in big (q|k|v and z, as m->big) and
       hn (the norm, for the betas and alphas the CPU makes) */
    int hb;
    float *xs[2][2], *att[2], *q[2], *k[2], *v[2], *big[2], *hn[2];
    float *ple[2]; /* Gemma 4: a layer's per-layer inputs, ple_dim a row */
    /* INT8 activations (JANAS_NPU_A8=1): a8 0 off, 1 calibrating - the
       blocks the NPU would take run on the CPU's path, which gives each
       product input's channel maxima into amax (janas_m_npu_calib; graphs
       to give them did not fit the NPU's memory beside the others) - 2
       the INT8 graphs; a8_left the blocks still to calibrate on, in_calib
       this pass is one */
    int a8, a8_left, in_calib;
    int a8_sites; /* the sites in INT8 (JANAS_NPU_A8_SITES, bit = site) */
    float **amax;
    uint64_t lmode_loads; /* the arena's loads the graphs were made at */
    uint64_t layer_passes;
    double t_layer_wait, t_layer_attn, layers_compile_s;
};

static void see_arena(struct janas_llm_model *m, struct janas_m_npu *n);
static void layers_free(struct janas_m_npu *n, uint32_t n_layer);

static long long now_us(void)
{
    return (long long)(now() * 1e6);
}

/* a preparation begins, of total graphs (0: not known ahead), or goes on */
static void prep_begin(struct janas_m_npu *n, int total)
{
    if (atomic_load(&n->prep_on))
        return;
    atomic_store(&n->prep_done, 0);
    atomic_store(&n->prep_total, total);
    atomic_store(&n->prep_t0_us, now_us());
    atomic_store(&n->prep_on, 1);
}

static void prep_end(struct janas_m_npu *n)
{
    if (atomic_exchange(&n->prep_on, 0))
        atomic_fetch_add(&n->prep_us, now_us() - atomic_load(&n->prep_t0_us));
}

void janas_m_npu_pass_end(struct janas_llm_model *m)
{
    if (m->npu)
        prep_end(m->npu);
}

int janas_llm_model_npu_preparing(const struct janas_llm_model *m,
                                  int32_t *done, int32_t *total,
                                  double *seconds)
{
    struct janas_m_npu *n = m->npu;
    int on = n ? atomic_load(&n->prep_on) : 0;
    if (done)
        *done = n ? atomic_load(&n->prep_done) : 0;
    if (total)
        *total = n ? atomic_load(&n->prep_total) : 0;
    if (seconds)
        *seconds = !n   ? 0
                   : on ? (double)(now_us() - atomic_load(&n->prep_t0_us)) / 1e6
                        : (double)atomic_load(&n->prep_us) / 1e6;
    return on;
}

void janas_m_npu_open(struct janas_llm_model *m)
{
    const char *e = getenv("JANAS_NPU");
    if (e && atoi(e) <= 0) {
        snprintf(m->npu_why, sizeof m->npu_why, "turned off by JANAS_NPU=0");
        return;
    }
    char err[256] = "no memory";
    struct janas_m_npu *n = calloc(1, sizeof(*n));
    if (!n || !(n->u = janas_npu_create(err, sizeof err))) {
        snprintf(m->npu_why, sizeof m->npu_why, "%.150s", err);
        if (e) /* asked for: say why not */
            fprintf(stderr, "janas: NPU not used: %s\n", err);
        free(n);
        return;
    }
    /* the compiled matrices kept across sessions: their preparation is
       paid once, until the cache is deleted - in JANAS_NPU_CACHE, or
       ~/.cache/janas/npu (XDG_CACHE_HOME honoured); JANAS_NPU_CACHE= (empty)
       keeps none */
    const char *c = getenv("JANAS_NPU_CACHE");
    char dir[1024];
    if (!c && janas_cache_dir(dir, sizeof dir, 1) == 0) {
        strcat(dir, "/npu");
        c = dir;
    }
    if (c && *c && janas_npu_set_cache(n->u, c))
        fprintf(stderr, "janas: NPU cache %s not usable\n", c);
    const char *mn = getenv("JANAS_NPU_MIN_N");
    n->min_n = mn && atoi(mn) > 0 ? atoi(mn) : 64;
    n->block = (int)janas_llm_model_max_block(m);
    const char *mb = getenv("JANAS_NPU_MAX_MB");
    n->budget = (size_t)(mb && atol(mb) > 0 ? atol(mb) : 4096) << 20;
    m->npu = n;
    m->npu_why[0] = 0;
    if (getenv("JANAS_NPU_STATS"))
        fprintf(stderr,
                "janas: NPU %s, blocks of %d, products of %d tokens and "
                "more\n",
                janas_npu_name(n->u), n->block, n->min_n);
}

int janas_m_npu_takes(const struct janas_llm_model *m, uint32_t n)
{
    return m->npu && m->npu_use && (int)n >= m->npu->min_n;
}

void janas_m_npu_awake(struct janas_llm_model *m, uint32_t n)
{
    if (!m->npu)
        return;
    int takes = janas_m_npu_takes(m, n);
    janas_npu_keep_awake(m->npu->u, takes);
    if (takes)
        see_arena(m, m->npu);
}

double janas_llm_model_npu_compile_seconds(const struct janas_llm_model *m)
{
    return m->npu ? m->npu->setup_s : 0;
}

const char *janas_llm_model_npu_why(const struct janas_llm_model *m)
{
    return m->npu_why;
}

void janas_m_npu_close(struct janas_llm_model *m)
{
    struct janas_m_npu *n = m->npu;
    if (!n)
        return;
    int np, nc;
    double cs;
    janas_npu_stats(n->u, &np, &nc, &cs);
    if (getenv("JANAS_NPU_STATS"))
        fprintf(stderr,
                "janas: NPU %d matrices (%d from the cache, %.1f s), %llu "
                "products, %llu to the CPU; left to the CPU: %llu too few "
                "vectors, %llu type or layout, %llu weights that may "
                "change\n",
                np, nc, cs, (unsigned long long)n->runs,
                (unsigned long long)n->cpu_fallbacks,
                (unsigned long long)n->no_n, (unsigned long long)n->no_type,
                (unsigned long long)n->no_stable);
    if (getenv("JANAS_NPU_STATS"))
        fprintf(stderr,
                "janas: NPU time: inputs expanded %.2f s, the CPU's share of "
                "the groups %.2f s, waiting for the NPU %.2f s, outputs "
                "copied %.2f s; matrices looked up %.2f s, runs begun %.2f "
                "s, the products' calls %.2f s\n",
                n->t_expand, n->t_cpu, n->t_wait, n->t_copy, n->t_entry,
                n->t_begin, n->t_calls);
    if (getenv("JANAS_NPU_STATS"))
        fprintf(stderr,
                "janas: NPU feed-forwards in one run: %llu (%.2f s waited), "
                "%llu refused; outputs in use twice: %llu\n",
                (unsigned long long)n->ffn_runs, n->t_ffn_wait,
                (unsigned long long)n->ffn_refused,
                (unsigned long long)n->y_clash);
    const char *st = getenv("JANAS_NPU_STATS");
    for (int i = 0; st && atoi(st) >= 2 && i < n->nsh; i++)
        fprintf(stderr,
                "janas: NPU %5zu x %5zu type %3d: %6llu runs, %.2f ms each "
                "waited; refused %llu (vectors) %llu (type, layout) %llu "
                "(weights)\n",
                n->sh[i].rows, n->sh[i].cols, n->sh[i].type,
                (unsigned long long)n->sh[i].runs,
                n->sh[i].runs ? n->sh[i].wait / n->sh[i].runs * 1e3 : 0,
                (unsigned long long)n->sh[i].refused[0],
                (unsigned long long)n->sh[i].refused[1],
                (unsigned long long)n->sh[i].refused[2]);
    if (getenv("JANAS_NPU_STATS") && n->layer_passes)
        fprintf(stderr,
                "janas: NPU whole layers: %u of %u, %llu blocks, %.2f s "
                "waited for the graphs, %.2f s of attention or recurrence on "
                "the CPU, graphs made in %.1f s\n",
                n->nl, m->n_layer, (unsigned long long)n->layer_passes,
                n->t_layer_wait, n->t_layer_attn, n->layers_compile_s);
    layers_free(n, m->n_layer);
    for (size_t i = 0; n->amax && i < (size_t)m->n_layer * N_SITES; i++)
        free(n->amax[i]);
    free(n->amax);
    janas_npu_destroy(n->u);
    free(n);
    m->npu = NULL;
}

/* the types janas_dequantize unpacks (it aborts on any other), and the
   down matrix in planes: every quantization of the catalog's files - Q3_K
   of the 30B in three bits, Q5_1 and Q8_0 of Gemma 4 26B-A4B among them -
   and Qwen3.8-Flash-Next's (Q2_0, Q5_0, the IQ1, IQ2 and IQ3 kinds, Q2_K) */
static int dequant_type(int type)
{
    switch (type) {
    case JANAS_Q4_K:
    case JANAS_Q5_K:
    case JANAS_Q6_K:
    case JANAS_Q8_0:
    case JANAS_Q3_K:
    case JANAS_Q4_0:
    case JANAS_Q4_1:
    case JANAS_Q5_1:
    case JANAS_IQ4_NL:
    case JANAS_IQ4_XS:
    case JANAS_IQ3_S:
    case JANAS_MXFP4:
    case JANAS_Q5_0:
    case JANAS_Q2_0:
    case JANAS_Q2_K:
    case JANAS_IQ3_XXS:
    case JANAS_IQ2_XXS:
    case JANAS_IQ2_XS:
    case JANAS_IQ2_S:
    case JANAS_IQ1_M:
    case JANAS_IQ1_S:
        return 1;
    default:
        return 0;
    }
}

int janas_llm_npu_type(int type)
{
    return dequant_type(type);
}

static int npu_type(int type)
{
    return dequant_type(type) || type == JANAS_Q6_K_P;
}

static float *xbuf(struct janas_m_npu *n, size_t cols)
{
    for (int i = 0; i < n->nx; i++)
        if (n->xb[i].cols == cols)
            return n->xb[i].x;
    if (n->nx == NPU_XBUFS)
        return NULL;
    float *x = janas_npu_alloc(n->u, sizeof(float) * (size_t)n->block * cols);
    if (!x)
        return NULL;
    memset(x, 0, sizeof(float) * (size_t)n->block * cols);
    n->xb[n->nx].cols = cols, n->xb[n->nx].x = x, n->xb[n->nx].src = NULL;
    n->nx++;
    return x;
}

/* the matrix's entry, compiled on first use; NULL if it stays on the CPU */
/* a matrix's NPU copy given up (the arena changed, or its planes did): the
   slot stays its own, compiled again at its next use */
static void drop(struct janas_m_npu *n, struct npu_entry *e)
{
    janas_npu_mm_destroy(e->mm);
    if (e->mm)
        n->used -= e->elems;
    e->mm = NULL, e->y = NULL, e->bad = 0;
}

/* JANAS_NPU_CHECK=1: each result of the NPU scanned - not finite, and the
   largest magnitude - the first bad one said, with its inputs' largest */
static int checking(void)
{
    static int c = -1;
    if (c < 0) {
        const char *e = getenv("JANAS_NPU_CHECK");
        c = e && atoi(e) > 0;
    }
    return c;
}

static void check(const char *what, size_t rows, size_t cols, const float *x,
                  size_t nx, const float *y, size_t ny)
{
    static int said;
    float xm = 0, ym = 0;
    size_t bad = 0;
    for (size_t i = 0; i < nx; i++)
        xm = fabsf(x[i]) > xm ? fabsf(x[i]) : xm;
    for (size_t i = 0; i < ny; i++) {
        if (!isfinite(y[i]))
            bad++;
        else if (fabsf(y[i]) > ym)
            ym = fabsf(y[i]);
    }
    if (bad && said++ < 8)
        fprintf(stderr,
                "janas: NPU %s %zu x %zu: %zu of %zu outputs not finite; "
                "largest input %g, largest finite output %g\n",
                what, rows, cols, bad, ny, xm, ym);
}

static float *ybuf(struct janas_m_npu *n, size_t rows, int ord)
{
    for (int i = 0; i < n->ny; i++)
        if (n->yb[i].rows == rows && n->yb[i].ord == ord)
            return n->yb[i].y;
    if (n->ny == 64)
        return NULL;
    float *y = janas_npu_alloc(n->u, sizeof(float) * (size_t)n->block * rows);
    if (y)
        n->yb[n->ny].rows = rows, n->yb[n->ny].ord = ord, n->yb[n->ny++].y = y;
    return y;
}

/* a matrix made ready for the NPU on the pool's threads, a few rows each:
   rows of a quantized type dequantized and quantized again to INT8 a row
   at a time, or bf16 rows turned to FP16 - each value as one thread made
   it, so the graph and its fingerprint in the cache are the same. On one
   thread the 422 matrices of Qwen3.8-Flash-Next took 12 s of the first
   prompt, the graphs already compiled (24 Oct 2026). */
struct prep_job {
    int type;          /* the weights' type, or -1: bf16 to FP16 */
    const uint8_t *w;  /* quantized rows of row_bytes */
    const uint16_t *b; /* or bf16 rows */
    size_t row_bytes, rows, cols;
    float *wf;
    int8_t *q;
    uint16_t *s, *f16;
    atomic_size_t next;
};

#define PREP_ROWS 32

static void prep_worker(void *arg, int tid, int n_threads)
{
    (void)tid;
    (void)n_threads;
    struct prep_job *p = arg;
    for (;;) {
        size_t r0 = atomic_fetch_add(&p->next, PREP_ROWS);
        if (r0 >= p->rows)
            break;
        size_t nr = r0 + PREP_ROWS < p->rows ? PREP_ROWS : p->rows - r0;
        size_t a = r0 * p->cols, c = nr * p->cols;
        if (p->type < 0) {
            for (size_t k = a; k < a + c; k++)
                p->f16[k] =
                    janas_fp32_to_fp16(janas_u2f((uint32_t)p->b[k] << 16));
            continue;
        }
        janas_dequantize(p->type, p->w + r0 * p->row_bytes, p->wf + a, c);
        janas_npu_quant_rows(p->wf + a, nr, p->cols, p->q + a, p->s + r0);
    }
}

static void prep_rows(struct janas_llm_model *m, struct prep_job *p)
{
    atomic_init(&p->next, 0);
    janas_pool_run(m->compute, prep_worker, p);
}

static struct npu_entry *entry(struct janas_llm_model *m, struct janas_m_npu *n,
                               const struct janas_matvec_task *t, int arena,
                               int ord)
{
    size_t h = ((uintptr_t)t->w >> 6) % NPU_SLOTS;
    for (size_t i = 0; i < NPU_SLOTS; i++, h = (h + 1) % NPU_SLOTS) {
        struct npu_entry *e = &n->e[h];
        if (e->w == t->w && e->mm &&
            (t->type != JANAS_Q6_K_P ||
             (e->plane[0] == t->plane[0] && e->plane[1] == t->plane[1])))
            return e;
        if (e->w == t->w && e->bad)
            return NULL;
        if (e->w && e->w != t->w)
            continue;
        if (e->w)
            drop(n, e); /* other planes, or given up: again */
        double t_setup = now();
        e->w = t->w, e->bad = 1, e->in_arena = arena;
        e->plane[0] = t->plane[0], e->plane[1] = t->plane[1];
        e->elems = t->rows * t->cols;
        if (n->used + t->rows * t->cols > n->budget)
            return NULL; /* stays on the CPU for good */
        prep_begin(n, 0);
        float *x = xbuf(n, t->cols);
        size_t rc = t->rows * t->cols;
        float *wf = x ? malloc(sizeof(float) * rc) : NULL;
        int8_t *q = wf ? malloc(rc) : NULL;
        uint16_t *s = q ? malloc(2 * t->rows) : NULL;
        e->y = s ? ybuf(n, t->rows, ord) : NULL;
        if (e->y) {
            /* rows apart from a multiple of 256 columns, each its own
               bytes: on the pool */
            size_t rb = janas_qtype_row_bytes(t->type, t->cols);
            if (t->type != JANAS_Q6_K_P && t->cols % JANAS_QK == 0 && rb) {
                struct prep_job p = {.type = t->type,
                                     .w = t->w,
                                     .row_bytes = rb,
                                     .rows = t->rows,
                                     .cols = t->cols,
                                     .wf = wf,
                                     .q = q,
                                     .s = s};
                prep_rows(m, &p);
            } else {
                if (t->type == JANAS_Q6_K_P) /* at the level in memory */
                    janas_q6kp_dequantize(t->w, t->plane[0], t->plane[1], wf,
                                          rc);
                else
                    janas_dequantize(t->type, t->w, wf, rc);
                janas_npu_quant_rows(wf, t->rows, t->cols, q, s);
            }
            char err[256];
            e->mm = janas_npu_mm_create(n->u, n->block, (int)t->rows,
                                        (int)t->cols, JANAS_NPU_I8, q, s, x,
                                        e->y, err, sizeof err);
            if (!e->mm)
                fprintf(stderr, "janas: NPU: %zu x %zu on the CPU: %s\n",
                        t->rows, t->cols, err);
        }
        free(wf), free(q), free(s);
        atomic_fetch_add(&n->prep_done, 1);
        e->bad = !e->mm;
        if (e->mm)
            n->used += t->rows * t->cols;
        if (e->bad)
            e->y = NULL;
        n->setup_s += now() - t_setup;
        return e->bad ? NULL : e;
    }
    return NULL;
}

/* weights that stay as they are: the resident region, or the experts'
   arena while it holds every expert (nothing more is loaded into it) */
/* 1: in the resident region, 2: in the arena as this pass found it whole,
   0: weights that may change */
static int stable(const struct janas_llm_model *m, const struct janas_m_npu *n,
                  const void *p)
{
    const uint8_t *w = p, *r = m->resident;
    if (!w)
        return 1; /* a plane not in memory: nothing to read */
    if (w >= r && w < r + m->j.h.resident_bytes)
        return 1;
    return n->arena && w >= n->arena && w < n->arena + n->arena_bytes ? 2 : 0;
}

/* at a pass's start, the cache idle: the arena whole or not, and the
   matrices compiled from it given up if it changed since */
static void see_arena(struct janas_llm_model *m, struct janas_m_npu *n)
{
    n->arena = m->cache ? janas_expert_cache_resident(m->cache, &n->arena_bytes,
                                                      &n->arena_loads)
                        : NULL;
    if (!n->arena || n->arena_loads == n->compiled_loads)
        return;
    for (size_t i = 0; i < NPU_SLOTS; i++)
        if (n->e[i].w && n->e[i].in_arena && n->e[i].mm)
            drop(n, &n->e[i]);
    n->compiled_loads = n->arena_loads;
}

static int shape(struct janas_m_npu *n, const struct janas_matvec_task *t)
{
    for (int i = 0; i < n->nsh; i++)
        if (n->sh[i].rows == t->rows && n->sh[i].cols == t->cols &&
            n->sh[i].type == t->type)
            return i;
    if (n->nsh == 32)
        return -1;
    n->sh[n->nsh].rows = t->rows, n->sh[n->nsh].cols = t->cols;
    n->sh[n->nsh].type = t->type;
    return n->nsh++;
}

static void refused(struct janas_m_npu *n, const struct janas_matvec_task *t,
                    int why)
{
    int i = t->n_vec > 1 ? shape(n, t) : -1;
    if (i >= 0)
        n->sh[i].refused[why]++;
}

static int eligible(const struct janas_llm_model *m, struct janas_m_npu *n,
                    const struct janas_matvec_task *t)
{
    /* a row may end inside a group of 256 (qwen4exp's 640): its
       activations' last group whole. Outputs further apart than their rows
       (a recurrent layer's q|k|v and z side by side in one buffer) are
       copied there: until 22 Oct 2026 they stayed on the CPU */
    size_t nb = (t->cols + JANAS_QK - 1) / JANAS_QK;
    /* the output head stays on the CPU: a chat's prompt asks it for one
       token, and over many (an evaluation's every logit) its INT8 copy
       would add an error no chat has */
    if (m->output && t->w == data(m, m->output))
        return n->no_type++, 0;
    if (!t->w || (int)t->n_vec < n->min_n || (int)t->n_vec > n->block) {
        return refused(n, t, 0), n->no_n++, 0;
    }
    if (!npu_type(t->type) ||
        (t->cols % JANAS_QK &&
         (t->type == JANAS_Q6_K_P ||
          janas_qtype_row_bytes(t->type, t->cols) == 0)) ||
        (t->x_stride && t->x_stride != nb) ||
        (t->y_stride && t->y_stride < t->rows))
        return refused(n, t, 1), n->no_type++, 0;
    int st = stable(m, n, t->w);
    if (t->type == JANAS_Q6_K_P && st &&
        (stable(m, n, t->plane[0]) == 0 || stable(m, n, t->plane[1]) == 0))
        st = 0;
    if (!st)
        return refused(n, t, 2), n->no_stable++, 0;
    return st;
}

const char *janas_llm_model_npu_name(const struct janas_llm_model *m)
{
    return m->npu ? janas_npu_name(m->npu->u) : NULL;
}

/* the input as floats, once for the tasks that share it; -1 when the
   buffer of that width holds another input a product in flight reads */
/* ---- a run's inputs and outputs moved on the pool's threads ---- */

/* n_vec vectors of len floats from src (every sstride) to dst (every
   dstride), times scale where scaled; or, with q, Q8_K inputs of len
   values expanded to floats (a row ending inside a group: that group's
   first values only). A few vectors a thread at a time, each value as one
   thread would make it: 10 MB of a mix's input and as much of its output
   went one thread's way, the NPU's runs waiting on them */
struct mover {
    const float *src;
    const struct janas_block_q8k *q;
    float *dst;
    size_t sstride, dstride, len, n_vec;
    float scale;
    int scaled;
    atomic_size_t next;
};

#define MOVE_VECS 8

static void move_worker(void *arg, int tid, int n_threads)
{
    (void)tid;
    (void)n_threads;
    struct mover *mv = arg;
    size_t nb = (mv->len + JANAS_QK - 1) / JANAS_QK;
    for (;;) {
        size_t v0 = atomic_fetch_add(&mv->next, MOVE_VECS);
        if (v0 >= mv->n_vec)
            break;
        size_t v1 = v0 + MOVE_VECS < mv->n_vec ? v0 + MOVE_VECS : mv->n_vec;
        for (size_t v = v0; v < v1; v++) {
            float *o = mv->dst + v * mv->dstride;
            if (mv->q) {
                for (size_t b = 0; b < nb; b++) {
                    const struct janas_block_q8k *q = mv->q + v * nb + b;
                    size_t e = mv->len - b * JANAS_QK < JANAS_QK
                                   ? mv->len - b * JANAS_QK
                                   : JANAS_QK;
                    for (size_t k = 0; k < e; k++)
                        o[b * JANAS_QK + k] = q->d * (float)q->qs[k];
                }
            } else if (mv->scaled) {
                const float *i = mv->src + v * mv->sstride;
                for (size_t k = 0; k < mv->len; k++)
                    o[k] = i[k] * mv->scale;
            } else
                memcpy(o, mv->src + v * mv->sstride, sizeof(float) * mv->len);
        }
    }
}

/* on the pool from 256 KB on; below, the caller alone */
static void move(struct janas_llm_model *m, struct mover *mv)
{
    atomic_init(&mv->next, 0);
    if (mv->n_vec * mv->len * sizeof(float) < (256u << 10))
        move_worker(mv, 0, 1);
    else
        janas_pool_run(m->compute, move_worker, mv);
}

static int expand(struct janas_llm_model *m, struct janas_m_npu *n,
                  const struct janas_matvec_task *t)
{
    int i = 0;
    while (n->xb[i].cols != t->cols)
        i++;
    if (n->xb[i].src == t->x && n->xb[i].n_vec == t->n_vec)
        return 0;
    if (n->xb[i].src)
        return -1;
    struct mover mv = {.q = t->x,
                       .dst = n->xb[i].x,
                       .dstride = t->cols,
                       .len = t->cols,
                       .n_vec = t->n_vec};
    move(m, &mv);
    n->xb[i].src = t->x, n->xb[i].n_vec = t->n_vec;
    return 0;
}

/* the pool's idle workers asleep while the CPU waits for the NPU (unless
   JANAS_NPU_SPIN=1): spinning at their highest clock they slowed it - a
   Qwen3-4B prompt 261 tokens/s spinning, 320 asleep (20 Oct 2026). The
   spin as it was is returned, for npu_loud. */
static int npu_quiet(struct janas_llm_model *m)
{
    static int keep = -1;
    if (keep < 0) {
        const char *e = getenv("JANAS_NPU_SPIN");
        keep = e && atoi(e) > 0;
    }
    int spin = janas_pool_get_spin(m->compute);
    if (!keep)
        janas_pool_set_spin(m->compute, 0);
    return spin;
}

static void npu_loud(struct janas_llm_model *m, int spin)
{
    janas_pool_set_spin(m->compute, spin);
}

void janas_m_products(struct janas_llm_model *m, struct janas_gpu *g,
                      const struct janas_matvec_task *t, size_t nt)
{
    /* not beside the whole layers: their graphs hold the NPU's memory
       (the layers after those it holds stay here) */
    struct janas_m_npu *n =
        m->npu_use && m->npu && m->npu->lmode != 1 ? m->npu : NULL;
    if (!n) {
        janas_gpu_matvec_group(g, m->compute, t, nt);
        return;
    }
    enum { MAXT = 64 };
    struct janas_matvec_task cpu[MAXT];
    struct npu_entry *on[MAXT];
    const struct janas_matvec_task *ont[MAXT];
    size_t nc = 0, nn = 0;
    if (nt > MAXT) {
        janas_gpu_matvec_group(g, m->compute, t, nt);
        return;
    }
    double t_call = now();
    for (int i = 0; i < n->nx; i++)
        n->xb[i].src = NULL; /* the engine rewrites its inputs between calls */
    for (size_t i = 0; i < nt; i++) {
        double tl = now();
        int st = eligible(m, n, &t[i]), ord = 0;
        for (size_t j = 0; st && j < nn; j++)
            ord += ont[j]->rows == t[i].rows;
        struct npu_entry *e = st ? entry(m, n, &t[i], st == 2, ord) : NULL;
        n->t_entry += now() - tl;
        for (size_t j = 0; e && j < nn; j++)
            if (on[j]->y == e->y) { /* another order than when compiled */
                n->y_clash++;
                e = NULL;
            }
        double te = now();
        int ok = e && expand(m, n, &t[i]) == 0;
        n->t_expand += now() - te;
        if (ok) {
            double tb = now();
            int begun = janas_npu_mm_begin(e->mm) == 0;
            n->t_begin += now() - tb;
            if (begun) {
                on[nn] = e, ont[nn++] = &t[i];
                continue;
            }
            n->cpu_fallbacks++;
        }
        cpu[nc++] = t[i];
    }
    double t0 = now();
    if (nc)
        janas_gpu_matvec_group(g, m->compute, cpu, nc);
    double t1 = now();
    n->t_cpu += t1 - t0;
    int spin = nn ? npu_quiet(m) : 0; /* from here the CPU only waits */
    for (size_t i = 0; i < nn; i++) {
        const struct janas_matvec_task *k = ont[i];
        double tw = now();
        int failed = janas_npu_mm_finish(on[i]->mm) != 0;
        double tc = now();
        n->t_wait += tc - tw;
        int si = shape(n, k);
        if (si >= 0)
            n->sh[si].runs++, n->sh[si].wait += tc - tw;
        if (failed) {
            /* the NPU failed: this product on the CPU after all */
            n->cpu_fallbacks++;
            janas_gpu_matvec_group(g, m->compute, k, 1);
            continue;
        }
        if (checking()) {
            int xi = 0;
            while (n->xb[xi].cols != k->cols)
                xi++;
            check("product", k->rows, k->cols, n->xb[xi].x, k->n_vec * k->cols,
                  on[i]->y, k->n_vec * k->rows);
        }
        struct mover mv = {.src = on[i]->y,
                           .dst = k->y,
                           .sstride = k->rows,
                           .dstride = k->y_stride ? k->y_stride : k->rows,
                           .len = k->rows,
                           .n_vec = k->n_vec};
        move(m, &mv);
        n->t_copy += now() - tc;
        n->runs++;
    }
    if (nn)
        npu_loud(m, spin);
    n->t_calls += now() - t_call;
}

/* ---- bf16 weights on float inputs (qwen4exp's mixes, model_hc.c) ---- */

/* the matrix's entry, compiled on first use with its weights in FP16 -
   bf16's values exactly, save those below FP16's normal range - keyed by
   w as the products'; NULL if it stays on the CPU */
static struct npu_entry *entry16(struct janas_llm_model *m,
                                 struct janas_m_npu *n, const uint16_t *w,
                                 size_t rows, size_t cols)
{
    size_t h = ((uintptr_t)w >> 6) % NPU_SLOTS;
    for (size_t i = 0; i < NPU_SLOTS; i++, h = (h + 1) % NPU_SLOTS) {
        struct npu_entry *e = &n->e[h];
        if (e->w == w && e->mm)
            return e;
        if (e->w == w && e->bad)
            return NULL;
        if (e->w && e->w != w)
            continue;
        double t_setup = now();
        e->w = w, e->bad = 1, e->in_arena = 0;
        e->plane[0] = e->plane[1] = NULL;
        e->elems = 2 * rows * cols; /* two bytes a weight, against the
                                       budget of INT8 copies */
        if (n->used + e->elems > n->budget)
            return NULL; /* stays on the CPU for good */
        prep_begin(n, 0);
        float *x = xbuf(n, cols);
        uint16_t *f16 = x ? malloc(2 * rows * cols) : NULL;
        e->y = f16 ? ybuf(n, rows, 0) : NULL;
        if (e->y) {
            struct prep_job p = {
                .type = -1, .b = w, .rows = rows, .cols = cols, .f16 = f16};
            prep_rows(m, &p);
            char err[256];
            e->mm = janas_npu_mm_create(n->u, n->block, (int)rows, (int)cols,
                                        JANAS_NPU_F16, f16, NULL, x, e->y, err,
                                        sizeof err);
            if (!e->mm)
                fprintf(stderr, "janas: NPU: %zu x %zu (bf16) on the CPU: %s\n",
                        rows, cols, err);
        }
        free(f16);
        atomic_fetch_add(&n->prep_done, 1);
        e->bad = !e->mm;
        if (e->mm)
            n->used += e->elems;
        if (e->bad)
            e->y = NULL;
        n->setup_s += now() - t_setup;
        return e->bad ? NULL : e;
    }
    return NULL;
}

int janas_m_npu_dense(struct janas_llm_model *m, const uint16_t *w,
                      const float *x, float *y, uint32_t rows, uint32_t cols,
                      uint32_t nv, float scale)
{
    struct janas_m_npu *n =
        m->npu_use && m->npu && m->npu->lmode != 1 ? m->npu : NULL;
    /* not the narrow ones (a mix's inject, 4 rows): a run's fixed cost,
       some 0.45 ms, is more than the CPU takes */
    if (!n || (int)nv < n->min_n || (int)nv > n->block || rows < 64 ||
        stable(m, n, w) != 1)
        return -1;
    struct npu_entry *e = entry16(m, n, w, rows, cols);
    if (!e)
        return -1;
    int i = 0;
    while (n->xb[i].cols != cols)
        i++;
    double te = now();
    struct mover in = {.src = x,
                       .dst = n->xb[i].x,
                       .sstride = cols,
                       .dstride = cols,
                       .len = cols,
                       .n_vec = nv};
    move(m, &in);
    n->xb[i].src = NULL; /* floats, not a Q8_K input expanded */
    n->t_expand += now() - te;
    int spin = npu_quiet(m);
    double tw = now();
    int r = janas_npu_mm_run(e->mm);
    double tc = now();
    n->t_wait += tc - tw;
    npu_loud(m, spin);
    if (r != 0) {
        n->cpu_fallbacks++;
        return -1;
    }
    struct mover out = {.src = e->y,
                        .dst = y,
                        .sstride = rows,
                        .dstride = rows,
                        .len = rows,
                        .n_vec = nv,
                        .scale = scale,
                        .scaled = 1};
    move(m, &out);
    n->t_copy += now() - tc;
    n->runs++;
    return 0;
}

/* ---- a feed-forward in one run ---- */

static int ffn_type(int type, int down)
{
    return dequant_type(type) || (down && type == JANAS_Q6_K_P);
}

static float *fbuf(struct janas_m_npu *n, size_t d, int out)
{
    for (int i = 0; i < n->nf; i++)
        if (n->fb[i].d == d)
            return out ? n->fb[i].y : n->fb[i].x;
    if (n->nf == 4)
        return NULL;
    float *x = janas_npu_alloc(n->u, sizeof(float) * (size_t)n->block * d),
          *y = janas_npu_alloc(n->u, sizeof(float) * (size_t)n->block * d);
    if (!x || !y)
        return NULL;
    memset(x, 0, sizeof(float) * (size_t)n->block * d);
    n->fb[n->nf].d = d, n->fb[n->nf].x = x, n->fb[n->nf++].y = y;
    return out ? y : x;
}

/* a matrix of the run as floats: Q6_K_P at the level of its planes */
static void deq(const struct janas_jns_matrix *mx, const uint8_t *w,
                const uint8_t *const plane[2], float *f)
{
    size_t n = (size_t)mx->rows * mx->cols;
    if (mx->type == JANAS_Q6_K_P)
        janas_q6kp_dequantize((const struct janas_block_q6kp *)w, plane[0],
                              plane[1], f, n);
    else
        janas_dequantize((int)mx->type, w, f, n);
}

/* the feed-forward's entry, keyed by its gate's address plus one (no
   matrix of the products starts at an odd address), compiled on first
   use; NULL if it stays on the CPU */
static struct npu_entry *ffn_entry(struct janas_llm_model *m,
                                   struct janas_m_npu *n,
                                   const uint8_t *const w[3],
                                   const uint8_t *const plane[2],
                                   const struct janas_jns_matrix *mx, int arena)
{
    const void *key = w[0] + 1;
    size_t h = ((uintptr_t)w[0] >> 6) % NPU_SLOTS;
    for (size_t i = 0; i < NPU_SLOTS; i++, h = (h + 1) % NPU_SLOTS) {
        struct npu_entry *e = &n->e[h];
        if (e->w == key && e->mm && e->plane[0] == plane[0] &&
            e->plane[1] == plane[1])
            return e;
        if (e->w == key && e->bad)
            return NULL;
        if (e->w && e->w != key)
            continue;
        if (e->w)
            drop(n, e);
        double t_setup = now();
        size_t d = mx[0].cols, ff = mx[0].rows, nw = d * ff;
        e->w = key, e->bad = 1, e->in_arena = arena;
        e->plane[0] = plane[0], e->plane[1] = plane[1];
        e->elems = 3 * nw;
        if (n->used + e->elems > n->budget)
            return NULL;
        prep_begin(n, 0);
        float *x = fbuf(n, d, 0), *y = fbuf(n, d, 1);
        float *f = x && y ? malloc(sizeof(float) * nw) : NULL;
        int8_t *q[3] = {0};
        uint16_t *s[3] = {0};
        int ok = f != NULL;
        for (int k = 0; ok && k < 3; k++) {
            size_t rows = mx[k].rows, cols = mx[k].cols;
            q[k] = malloc(rows * cols), s[k] = malloc(2 * rows);
            ok = q[k] && s[k];
            if (ok) {
                deq(&mx[k], w[k], plane, f);
                if (checking()) { /* the weights as floats, finite? */
                    size_t bad = 0;
                    float mxw = 0;
                    for (size_t i = 0; i < rows * cols; i++) {
                        if (!isfinite(f[i]))
                            bad++;
                        else if (fabsf(f[i]) > mxw)
                            mxw = fabsf(f[i]);
                    }
                    if (bad)
                        fprintf(stderr,
                                "janas: NPU feed-forward matrix %d (type %u, "
                                "%zu x %zu): %zu weights not finite, largest "
                                "%g\n",
                                k, mx[k].type, rows, cols, bad, mxw);
                }
                janas_npu_quant_rows(f, rows, cols, q[k], s[k]);
            }
        }
        if (ok) {
            char err[256];
            e->mm = janas_npu_ffn_create(
                n->u, n->block, (int)d, (int)ff,
                m->a->gelu ? JANAS_NPU_GELU_TANH : JANAS_NPU_SILU, q[0], s[0],
                q[1], s[1], q[2], s[2], x, y, err, sizeof err);
            if (!e->mm)
                fprintf(stderr,
                        "janas: NPU: feed-forward %zu x %zu on the "
                        "CPU: %s\n",
                        ff, d, err);
        }
        free(f);
        for (int k = 0; k < 3; k++)
            free(q[k]), free(s[k]);
        atomic_fetch_add(&n->prep_done, 1);
        e->bad = !e->mm;
        if (e->mm)
            n->used += e->elems;
        n->setup_s += now() - t_setup;
        return e->bad ? NULL : e;
    }
    return NULL;
}

int janas_m_npu_ffn_begin(struct janas_llm_model *m, const uint8_t *const w[3],
                          const uint8_t *const plane[2],
                          const struct janas_jns_matrix *mx,
                          const struct janas_block_q8k *x, uint32_t n_vec,
                          float *dout, struct janas_m_npu_ffn *job)
{
    struct janas_m_npu *n =
        m->npu_use && m->npu && m->npu->lmode != 1 ? m->npu : NULL;
    job->e = NULL;
    if (!n || (int)n_vec < n->min_n || (int)n_vec > n->block)
        return -1;
    /* the Ana-Net probes and the down by neuron read the intermediate */
    if (m->ana_mode || m->ana_tok || m->ana_agree || m->ana_coarse ||
        m->ana_coln || m->ana_level || m->ana_low || m->ana_frame ||
        m->ana_exp || m->nskipq || m->neuron_down)
        return -1;
    const struct janas_jns_matrix *g = &mx[0], *u = &mx[1], *d = &mx[2];
    if (!ffn_type((int)g->type, 0) || !ffn_type((int)u->type, 0) ||
        !ffn_type((int)d->type, 1) || g->rows != u->rows ||
        g->cols != u->cols || d->cols != g->rows || d->rows != g->cols ||
        g->cols % JANAS_QK ||
        (g->rows % JANAS_QK &&
         (d->type == JANAS_Q6_K_P ||
          janas_qtype_row_bytes((int)d->type, d->cols) == 0))) {
        n->ffn_refused++;
        return -1;
    }
    /* the matrices where they are: each at its offset in its slot */
    const uint8_t *wm[3] = {w[0] + g->offset, w[1] + u->offset,
                            w[2] + d->offset};
    int st = stable(m, n, wm[0]);
    if (st && (stable(m, n, wm[1]) != st || stable(m, n, wm[2]) != st ||
               (d->type == JANAS_Q6_K_P &&
                (!stable(m, n, plane[0]) || !stable(m, n, plane[1])))))
        st = 0;
    struct npu_entry *e = st ? ffn_entry(m, n, wm, plane, mx, st == 2) : NULL;
    if (!e) {
        n->ffn_refused++;
        return -1;
    }
    double te = now();
    size_t dm = g->cols, nb = dm / JANAS_QK;
    float *xf = fbuf(n, dm, 0);
    for (uint32_t v = 0; v < n_vec; v++)
        for (size_t b = 0; b < nb; b++) {
            const struct janas_block_q8k *q = x + v * nb + b;
            float *o = xf + v * dm + b * JANAS_QK;
            for (int k = 0; k < JANAS_QK; k++)
                o[k] = q->d * (float)q->qs[k];
        }
    n->t_expand += now() - te;
    if (janas_npu_mm_begin(e->mm))
        return -1;
    job->e = e, job->dout = dout, job->n_vec = n_vec, job->d = dm;
    return 0;
}

int janas_m_npu_ffn_finish(struct janas_llm_model *m,
                           struct janas_m_npu_ffn *job)
{
    struct janas_m_npu *n = m->npu;
    if (!job->e)
        return -1;
    double t0 = now();
    int spin = npu_quiet(m);
    int r = janas_npu_mm_finish(job->e->mm);
    npu_loud(m, spin);
    double t1 = now();
    n->t_ffn_wait += t1 - t0;
    if (r == 0) {
        if (checking())
            check("feed-forward", job->d, job->d, fbuf(n, job->d, 0),
                  job->n_vec * job->d, fbuf(n, job->d, 1), job->n_vec * job->d);
        memcpy(job->dout, fbuf(n, job->d, 1),
               sizeof(float) * job->n_vec * job->d);
        n->ffn_runs++;
    }
    n->t_copy += now() - t1;
    job->e = NULL;
    return r;
}

/* ---- whole layers ---- */

/* a matrix of the model as floats (malloc'd), Q6_K_P at the level of its
   planes */
static float *deq_matrix(int type, const void *w, const uint8_t *p1,
                         const uint8_t *p0, size_t rows, size_t cols)
{
    float *f = malloc(sizeof(float) * rows * cols);
    if (f && type == JANAS_Q6_K_P)
        janas_q6kp_dequantize(w, p1, p0, f, rows * cols);
    else if (f)
        janas_dequantize(type, w, f, rows * cols);
    return f;
}

static float *deq_tensor(const struct janas_llm_model *m,
                         const struct janas_jns_tensor *t, size_t *rows,
                         size_t *cols)
{
    *cols = t->dims[0], *rows = t->dims[1];
    if (t->type == 0) { /* F32 (Gemma 4's per-layer branch): as it is */
        float *f = malloc(sizeof(float) * *rows * *cols);
        if (f)
            memcpy(f, f32(m, t), sizeof(float) * *rows * *cols);
        return f;
    }
    if (!npu_type((int)t->type) || t->type == JANAS_Q6_K_P)
        return NULL;
    return deq_matrix((int)t->type, data(m, t), NULL, NULL, *rows, *cols);
}

/* dense layer l's FFN from the arena (its expert 0): gate, up, down */
static int deq_ffn(struct janas_llm_model *m, uint32_t l, float *f[3],
                   size_t rows[3], size_t cols[3])
{
    uint32_t e0 = 0;
    const uint8_t *slot = NULL;
    const struct janas_jns_layer *jl = &m->j.layers[l];
    f[0] = f[1] = f[2] = NULL;
    if (!m->cache || janas_expert_cache_fetch(m->cache, l, &e0, 1, &slot))
        return -1;
    const uint8_t *pl[2] = {NULL, NULL};
    for (int q = 0; q < m->exp_level - 1; q++)
        if (jl->plane[q].type)
            pl[q] = slot + jl->plane[q].offset;
    for (int k = 0; k < 3; k++) {
        const struct janas_jns_matrix *mx = &jl->m[k];
        rows[k] = mx->rows, cols[k] = mx->cols;
        f[k] = ffn_type((int)mx->type, k == 2)
                   ? deq_matrix((int)mx->type, slot + mx->offset, pl[0], pl[1],
                                mx->rows, mx->cols)
                   : NULL;
        if (!f[k])
            return -1;
    }
    return 0;
}

/* floats into the graph (freed): columns times cs, rows times rs (either
   NULL), then INT8 with a scale a row */
static int weights_f(struct npu_ir *ir, const char *name, float *f, size_t rows,
                     size_t cols, const float *cs, const float *rs, int *ok)
{
    if (!f) {
        *ok = 0;
        return -1;
    }
    for (size_t r = 0; (cs || rs) && r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            f[r * cols + c] *= (cs ? cs[c] : 1.0f) * (rs ? rs[r] : 1.0f);
    int8_t *q = malloc(rows * cols);
    uint16_t *s = q ? malloc(2 * rows) : NULL;
    int l = -1;
    if (s) {
        janas_npu_quant_rows(f, rows, cols, q, s);
        l = npu_ir_weights_i8(ir, name, q, s, (int)rows, (int)cols);
    } else
        *ok = 0;
    free(q), free(s), free(f);
    return l;
}

static void col_max(const float *f, size_t rows, size_t cols, float *w)
{
    for (size_t r = 0; f && r < rows; r++)
        for (size_t c = 0; c < cols; c++) {
            float a = fabsf(f[r * cols + c]);
            w[c] = a > w[c] ? a : w[c];
        }
}

/* SmoothQuant (alpha 1/2): s = sqrt(max |x| / max |w|) a channel, x / s
   and w s; the range of x / s returned */
static float smooth(const float *amax, const float *wmax, size_t k, float *s)
{
    float range = 0;
    for (size_t c = 0; c < k; c++) {
        float a = amax[c] > 1e-6f ? amax[c] : 1e-6f,
              w = wmax[c] > 1e-6f ? wmax[c] : 1e-6f, v = sqrtf(a / w);
        v = v < 1e-3f ? 1e-3f : v > 1e3f ? 1e3f : v;
        s[c] = v;
        range = a / v > range ? a / v : range;
    }
    return range;
}

/* x (FP16) on the 256 steps of an INT8 over [-range 128/127, range] */
static int fq(struct npu_ir *ir, const char *name, int x, float range)
{
    return npu_ir_fake_quantize(ir, name, x, -range * 128.0f / 127.0f, range,
                                256);
}

static float *amax(struct janas_m_npu *n, uint32_t l, int site)
{
    return n->amax ? n->amax[(size_t)l * N_SITES + site] : NULL;
}

/* norm then q, k, v of layer lq from x (f32), as results; mode 2 INT8 on
   its calibration */
static void qkv_part(struct npu_ir *ir, struct janas_llm_model *m,
                     struct janas_m_npu *n, uint32_t lq, int x, int mode,
                     int *ok)
{
    const struct layer *ly = &m->layers[lq];
    size_t d = m->d_model, rows[3] = {0}, cols[3] = {0};
    /* Gemma 4: V is K in its full layers, and its last layers have no K,
       the keys and values of an earlier layer's in the cache */
    const struct janas_jns_tensor *t[3] = {ly->wq, ly->wk, ly->wv};
    float *f[3] = {NULL, NULL, NULL}, *wn = malloc(sizeof(float) * d),
          *s = NULL;
    int bad = !wn;
    for (int i = 0; i < 3; i++)
        if (t[i] && !(f[i] = deq_tensor(m, t[i], &rows[i], &cols[i])))
            bad = 1;
    if (bad || !f[0]) {
        *ok = 0;
        free(wn);
        for (int i = 0; i < 3; i++)
            free(f[i]);
        return;
    }
    memcpy(wn, f32(m, ly->attn_norm), sizeof(float) * d);
    float range = 0;
    if (mode == 2 && !(n->a8_sites & 1 << SITE_QKV))
        mode = 0;
    if (mode == 2) {
        float *wm = calloc(d, sizeof(float));
        s = malloc(sizeof(float) * d);
        for (int i = 0; wm && i < 3; i++)
            col_max(f[i], rows[i], cols[i], wm);
        range = wm && s ? smooth(amax(n, lq, SITE_QKV), wm, d, s) : 0;
        for (size_t c = 0; s && c < d; c++)
            wn[c] /= s[c];
        free(wm);
    }
    int hn = npu_ir_rms_norm(ir, "hn", x, wn, (int)d, m->eps);
    free(wn);
    int h = npu_ir_convert(ir, "hn16", hn, "f16");
    if (mode == 2)
        h = fq(ir, "hnq", h, range);
    static const char *nm[3][3] = {
        {"wq", "q", "q32"}, {"wk", "k", "k32"}, {"wv", "v", "v32"}};
    for (int i = 0; i < 3; i++)
        if (f[i])
            npu_ir_result(
                ir, npu_ir_convert(
                        ir, nm[i][2],
                        npu_ir_matmul_tb(ir, nm[i][1], h,
                                         weights_f(ir, nm[i][0], f[i], rows[i],
                                                   cols[i], s, NULL, ok),
                                         NULL),
                        "f32"));
    free(s);
}

/* layer lq's norm and input products from x (f32), as results: an
   attention layer's q (with its output gates where it has them), k, v; a
   recurrent layer's norm in f32 (the CPU's betas and alphas read it) and
   q|k|v and z in one product, as m->big holds them (Qwen3.5) */
static void in_part(struct npu_ir *ir, struct janas_llm_model *m,
                    struct janas_m_npu *n, uint32_t lq, int x, int mode,
                    int *ok)
{
    const struct layer *ly = &m->layers[lq];
    if (!ly->rec) {
        qkv_part(ir, m, n, lq, x, mode, ok);
        return;
    }
    size_t d = m->d_model, r1, c1, r2, c2;
    float *f1 = deq_tensor(m, ly->ssm_qkv, &r1, &c1),
          *f2 = deq_tensor(m, ly->ssm_z, &r2, &c2);
    float *f = f1 && f2 && c1 == d && c2 == d && r1 + r2 == (size_t)m->qkvz_dim
                   ? realloc(f1, sizeof(float) * (r1 + r2) * d)
                   : NULL;
    if (!f) {
        free(f1), free(f2);
        *ok = 0;
        return;
    }
    memcpy(f + r1 * d, f2, sizeof(float) * r2 * d);
    free(f2);
    int hn =
        npu_ir_rms_norm(ir, "hn", x, f32(m, ly->attn_norm), (int)d, m->eps);
    npu_ir_result(ir, hn);
    int h = npu_ir_convert(ir, "hn16", hn, "f16");
    npu_ir_result(
        ir, npu_ir_convert(ir, "qkvz32",
                           npu_ir_matmul_tb(ir, "qkvz", h,
                                            weights_f(ir, "wqkvz", f, r1 + r2,
                                                      d, NULL, NULL, ok),
                                            NULL),
                           "f32"));
}

/* the arguments in_part's results go to, half h's, from na on: the next
   na */
static int in_args(const struct janas_llm_model *m, struct janas_m_npu *n,
                   uint32_t lq, int h, float **args, int *rows, int *cols,
                   int na)
{
    const struct layer *ly = &m->layers[lq];
    int B = n->hb;
    if (ly->rec) {
        args[na] = n->hn[h], rows[na] = B, cols[na++] = (int)m->d_model;
        args[na] = n->big[h], rows[na] = B, cols[na++] = (int)m->qkvz_dim;
        return na;
    }
    int qd = (int)(m->n_head * ly->hd), kvn = (int)(ly->n_kv * ly->hd);
    args[na] = n->q[h], rows[na] = B;
    cols[na++] = m->a->attn_gate ? 2 * qd : qd;
    if (ly->wk)
        args[na] = n->k[h], rows[na] = B, cols[na++] = kvn;
    if (ly->wv)
        args[na] = n->v[h], rows[na] = B, cols[na++] = kvn;
    return na;
}

/* layer l's graph in mode 0 (FP16) or 2 (INT8 activations, SmoothQuant
   from the calibration); arguments: att, x in, x' out, then (unless last:
   the last layer the NPU runs) the next layer's input part out */
static int layer_graph(struct janas_llm_model *m, struct janas_m_npu *n,
                       uint32_t l, int mode, int last, char *err,
                       size_t err_len)
{
    const struct layer *ly = &m->layers[l];
    int B = n->hb, d = (int)m->d_model,
        qd = ly->rec ? (int)m->d_inner : (int)(m->n_head * ly->hd);
    int ok = 1;
    struct npu_ir ir;
    npu_ir_begin(&ir, "janas_layer");
    int ad[2] = {B, qd}, xd[2] = {B, d}, P = (int)m->ple_dim, pd[2] = {B, P};
    int la = npu_ir_param(&ir, "att", "f32", 2, ad);
    int lx = npu_ir_param(&ir, "x", "f32", 2, xd);
    int lp = P ? npu_ir_param(&ir, "ple", "f32", 2, pd) : -1;
    /* wo */
    size_t orow, ocol;
    float *fo = deq_tensor(m, ly->rec ? ly->ssm_out : ly->wo, &orow, &ocol),
          *so = NULL, ro = 0;
    int q_att = mode == 2 && (n->a8_sites & 1 << SITE_ATT),
        q_ffn = mode == 2 && (n->a8_sites & 1 << SITE_FFN),
        q_mid = mode == 2 && (n->a8_sites & 1 << SITE_MID);
    if (q_att && fo) {
        float *wm = calloc(ocol, sizeof(float)),
              *inv = malloc(sizeof(float) * ocol);
        so = malloc(sizeof(float) * ocol);
        col_max(fo, orow, ocol, wm);
        ro = wm && so && inv ? smooth(amax(n, l, SITE_ATT), wm, ocol, so) : 0;
        for (size_t c = 0; inv && so && c < ocol; c++)
            inv[c] = 1.0f / so[c];
        int one = (int)ocol;
        if (inv)
            la = npu_ir_binary(&ir, "att_s", "Multiply", la,
                               npu_ir_const(&ir, "att_inv", "f32", 1, &one, inv,
                                            sizeof(float) * ocol));
        free(wm), free(inv);
    }
    int a16 = npu_ir_convert(&ir, "a16", la, "f16");
    if (q_att)
        a16 = fq(&ir, "aq", a16, ro);
    int o = npu_ir_matmul_tb(
        &ir, "o", a16, weights_f(&ir, "wo", fo, orow, ocol, so, NULL, &ok),
        NULL);
    free(so);
    int o32 = npu_ir_convert(&ir, "o32", o, "f32");
    if (ly->post_attn) /* Gemma 4: the attention's output normalized */
        o32 = npu_ir_rms_norm(&ir, "on", o32, f32(m, ly->post_attn), d, m->eps);
    int x1 = npu_ir_add(&ir, "x1", lx, o32);
    /* the FFN */
    float *ff[3];
    size_t fr[3], fc[3];
    if (deq_ffn(m, l, ff, fr, fc))
        ok = 0;
    size_t nf = fr[0];
    float *wn = malloc(sizeof(float) * d), *sf = NULL, *sm = NULL, *rinv = NULL,
          rf = 0, rm = 0;
    if (wn)
        memcpy(wn, f32(m, ly->ffn_norm), sizeof(float) * d);
    if (q_ffn && ok && wn) { /* gate's and up's input: the norm's */
        float *wm = calloc((size_t)d, sizeof(float));
        sf = malloc(sizeof(float) * d);
        if (wm && sf) {
            col_max(ff[0], fr[0], fc[0], wm);
            col_max(ff[1], fr[1], fc[1], wm);
            rf = smooth(amax(n, l, SITE_FFN), wm, (size_t)d, sf);
            for (int c = 0; c < d; c++)
                wn[c] /= sf[c];
        } else
            ok = 0;
        free(wm);
    }
    if (q_mid && ok) { /* down's input: the intermediate, through up */
        float *wd = calloc(nf, sizeof(float));
        sm = malloc(sizeof(float) * nf);
        rinv = malloc(sizeof(float) * nf);
        if (wd && sm && rinv) {
            col_max(ff[2], fr[2], fc[2], wd);
            rm = smooth(amax(n, l, SITE_MID), wd, nf, sm);
            for (size_t c = 0; c < nf; c++)
                rinv[c] = 1.0f / sm[c]; /* into up's rows: h / s */
        } else
            ok = 0;
        free(wd);
    }
    int hf = wn ? npu_ir_rms_norm(&ir, "hf", x1, wn, d, m->eps) : -1;
    free(wn);
    int h16 = npu_ir_convert(&ir, "h16", hf, "f16");
    if (q_ffn)
        h16 = fq(&ir, "hq", h16, rf);
    int wg = weights_f(&ir, "fg", ff[0], fr[0], fc[0], sf, NULL, &ok);
    int wu = weights_f(&ir, "fu", ff[1], fr[1], fc[1], sf, rinv, &ok);
    int wdn = weights_f(&ir, "fd", ff[2], fr[2], fc[2], sm, NULL, &ok);
    free(sf), free(sm), free(rinv);
    int lg = npu_ir_matmul_tb(&ir, "g", h16, wg, NULL);
    int lu = npu_ir_matmul_tb(&ir, "u", h16, wu, NULL);
    int act;
    if (!m->a->gelu)
        act = npu_ir_multiply(&ir, "silu", lg,
                              npu_ir_unary(&ir, "sig", "Sigmoid", lg));
    else {
        int g3 =
            npu_ir_multiply(&ir, "g3", npu_ir_multiply(&ir, "g2", lg, lg), lg);
        int in =
            npu_ir_add(&ir, "gin", lg,
                       npu_ir_multiply(&ir, "gc3", g3,
                                       npu_ir_scalar(&ir, "c3", 0.044715f)));
        int t = npu_ir_unary(
            &ir, "gt", "Tanh",
            npu_ir_multiply(&ir, "gk", in,
                            npu_ir_scalar(&ir, "k", 0.7978845608f)));
        act = npu_ir_multiply(
            &ir, "gelu",
            npu_ir_multiply(&ir, "gh", lg, npu_ir_scalar(&ir, "half", 0.5f)),
            npu_ir_add(&ir, "g1t", t, npu_ir_scalar(&ir, "one", 1.0f)));
    }
    int hm = npu_ir_multiply(&ir, "hm", act, lu);
    if (q_mid)
        hm = fq(&ir, "mq", hm, rm);
    int f = npu_ir_matmul_tb(&ir, "dn", hm, wdn, NULL);
    int f32o = npu_ir_convert(&ir, "f32", f, "f32");
    if (ly->post_ffn) /* Gemma 4: the FFN's output normalized */
        f32o =
            npu_ir_rms_norm(&ir, "fn", f32o, f32(m, ly->post_ffn), d, m->eps);
    int x2 = npu_ir_add(&ir, "x2", x1, f32o);
    if (P && ok) {
        /* the per-layer embeddings' branch (janas_m_ple_layer):
           gelu(gate x) times the layer's inputs, projected back,
           normalized, added */
        size_t gr, gc, pr, pc;
        float *fg = deq_tensor(m, ly->ple_gate, &gr, &gc),
              *fp = deq_tensor(m, ly->ple_proj, &pr, &pc);
        int x16 = npu_ir_convert(&ir, "x2h", x2, "f16");
        int g = npu_ir_matmul_tb(
            &ir, "pg", x16, weights_f(&ir, "wpg", fg, gr, gc, NULL, NULL, &ok),
            NULL);
        /* g sigmoid(1.5958 (g + 0.044715 g^3)): the tanh form, as
           janas_gelu_mul_det */
        int g3 =
            npu_ir_multiply(&ir, "pg3", npu_ir_multiply(&ir, "pg2", g, g), g);
        int z = npu_ir_multiply(
            &ir, "pz",
            npu_ir_add(&ir, "pin", g,
                       npu_ir_multiply(&ir, "pgc", g3,
                                       npu_ir_scalar(&ir, "pc3", 0.044715f))),
            npu_ir_scalar(&ir, "pk", 1.5957691216057308f));
        int gl = npu_ir_multiply(&ir, "pgl", g,
                                 npu_ir_unary(&ir, "psg", "Sigmoid", z));
        int gi = npu_ir_multiply(&ir, "pgi", gl,
                                 npu_ir_convert(&ir, "ple16", lp, "f16"));
        int p = npu_ir_matmul_tb(
            &ir, "pp", gi, weights_f(&ir, "wpp", fp, pr, pc, NULL, NULL, &ok),
            NULL);
        int pn =
            npu_ir_rms_norm(&ir, "pn", npu_ir_convert(&ir, "pp32", p, "f32"),
                            f32(m, ly->ple_post), d, m->eps);
        x2 = npu_ir_add(&ir, "x3", x2, pn);
    }
    if (ly->out_scale != 1.0f) { /* in f32: it scales the residual itself */
        int one = 1;
        x2 = npu_ir_multiply(&ir, "xs", x2,
                             npu_ir_const(&ir, "os", "f32", 1, &one,
                                          &ly->out_scale, sizeof(float)));
    }
    npu_ir_result(&ir, x2);
    if (!last)
        in_part(&ir, m, n, l + 1, x2, mode, &ok);
    if (!ok) {
        npu_ir_end(&ir);
        npu_ir_free(&ir);
        snprintf(err, err_len, "layer %u: a type the NPU does not take", l);
        return -1;
    }
    float *args[2][10];
    int rows[10], cols[10], na = 0;
    for (int h = 0; h < 2; h++) {
        na = 0;
        args[h][na] = n->att[h], rows[na] = B, cols[na++] = qd;
        args[h][na] = n->xs[h][l % 2], rows[na] = B, cols[na++] = d;
        if (P)
            args[h][na] = n->ple[h], rows[na] = B, cols[na++] = P;
        args[h][na] = n->xs[h][(l + 1) % 2], rows[na] = B, cols[na++] = d;
        if (!last)
            na = in_args(m, n, l + 1, h, args[h], rows, cols, na);
    }
    n->lg[l] = janas_npu_graph_create(n->u, &ir, na, args[0], rows, cols, err,
                                      err_len);
    if (n->lg[l] && janas_npu_mm_bind_alt(n->lg[l], na, args[1])) {
        snprintf(err, err_len, "layer %u: no second binding", l);
        janas_npu_mm_destroy(n->lg[l]);
        n->lg[l] = NULL;
        return -1;
    }
    return n->lg[l] ? 0 : -1;
}

/* the first layer's norm and input products, in the same modes */
static int first_graph(struct janas_llm_model *m, struct janas_m_npu *n,
                       int mode, char *err, size_t err_len)
{
    int B = n->hb, d = (int)m->d_model, ok = 1;
    struct npu_ir ir;
    npu_ir_begin(&ir, "janas_first");
    int xd[2] = {B, d};
    int lx = npu_ir_param(&ir, "x", "f32", 2, xd);
    in_part(&ir, m, n, 0, lx, mode, &ok);
    if (!ok) {
        npu_ir_end(&ir);
        npu_ir_free(&ir);
        snprintf(err, err_len, "the first layer: a type the NPU does not take");
        return -1;
    }
    float *args[2][4];
    int rows[4], cols[4], na = 0;
    for (int h = 0; h < 2; h++) {
        args[h][0] = n->xs[h][0], rows[0] = B, cols[0] = d;
        na = in_args(m, n, 0, h, args[h], rows, cols, 1);
    }
    n->lpre = janas_npu_graph_create(n->u, &ir, na, args[0], rows, cols, err,
                                     err_len);
    if (n->lpre && janas_npu_mm_bind_alt(n->lpre, na, args[1])) {
        snprintf(err, err_len, "the first layer: no second binding");
        return -1;
    }
    return n->lpre ? 0 : -1;
}

/* the calibration's store: every site's maxima, zero */
static int calib_alloc(struct janas_llm_model *m, struct janas_m_npu *n)
{
    n->amax = calloc((size_t)m->n_layer * N_SITES, sizeof(float *));
    if (!n->amax)
        return -1;
    for (uint32_t l = 0; l < m->n_layer; l++)
        for (int s = 0; s < N_SITES; s++) {
            size_t w = s == SITE_ATT   ? m->n_head * m->layers[l].hd
                       : s == SITE_MID ? m->j.layers[l].m[0].rows
                                       : m->d_model;
            if (!(n->amax[(size_t)l * N_SITES + s] = calloc(w, sizeof(float))))
                return -1;
        }
    return 0;
}

/* whether this model's layers can go to the NPU whole: dense, a norm
   before the attention and before the FFN and nothing else, attention with
   its own keys and values (Qwen3's layout), or Qwen3.5's: attention with an
   output gate, and Gated DeltaNet layers whose q|k|v and z are two
   products */
static int layers_fit(const struct janas_llm_model *m)
{
    const struct janas_arch *a = m->a;
    /* Gemma 4's dense models too (sandwich norms, windows, per-layer
       embeddings): its mixture keeps the experts on the CPU */
    int gemma = a->sandwich && a->swa && a->rec == JANAS_REC_NONE;
    if ((a->rec != JANAS_REC_NONE && a->rec != JANAS_REC_Q35) ||
        (a->attn_gate && a->rec != JANAS_REC_Q35) || a->shared_expert ||
        a->hc || ((a->sandwich || a->swa || m->ple_dim) && !gemma) ||
        m->n_expert > 1 || m->pred_fn || !m->cache)
        return 0;
    uint32_t qd = 0;
    for (uint32_t l = 0; l < m->n_layer; l++) {
        const struct layer *ly = &m->layers[l];
        if (!ly->attn_norm || !ly->ffn_norm || ly->pre_norm2 || ly->sh_w[0] ||
            (!gemma && (ly->post_attn || ly->post_ffn || ly->swa)) ||
            (gemma && (!ly->post_attn || !ly->post_ffn)) ||
            (m->ple_dim && (!ly->ple_gate || !ly->ple_proj || !ly->ple_post)))
            return 0;
        if (ly->rec) {
            if (!ly->ssm_qkv || !ly->ssm_z || !ly->ssm_beta || !ly->ssm_alpha ||
                !ly->ssm_out)
                return 0;
            continue;
        }
        /* every attention layer its keys and values, but Gemma's: V is K
           in its full layers, and its last layers share an earlier one's;
           its window layers' heads are narrower than its full layers' */
        if (!ly->wq || !ly->wo || (!gemma && (!ly->wk || !ly->wv)) ||
            (!gemma && qd && m->n_head * ly->hd != qd))
            return 0;
        qd = m->n_head * ly->hd;
    }
    return 1;
}

/* the weights of layer l's input part, and of its graph without it */
static size_t in_elems(const struct janas_llm_model *m, uint32_t l)
{
    const struct layer *ly = &m->layers[l];
    if (ly->rec)
        return (size_t)m->qkvz_dim * m->d_model;
    size_t qd = (size_t)m->n_head * ly->hd, kv = (size_t)ly->n_kv * ly->hd;
    return ((m->a->attn_gate ? 2 : 1) * qd + (ly->wk ? kv : 0) +
            (ly->wv ? kv : 0)) *
           m->d_model;
}

static size_t body_elems(const struct janas_llm_model *m, uint32_t l)
{
    const struct layer *ly = &m->layers[l];
    const struct janas_jns_layer *jl = &m->j.layers[l];
    size_t e = (ly->rec ? (size_t)m->d_inner : (size_t)m->n_head * ly->hd) *
               m->d_model;
    for (int k = 0; k < 3; k++)
        e += (size_t)jl->m[k].rows * jl->m[k].cols;
    if (m->ple_dim) /* the per-layer branch's gate and projection */
        e += 2 * (size_t)m->ple_dim * m->d_model;
    return e;
}

/* how many of the first layers fit the NPU: their INT8 weights within
   JANAS_NPU_LAYERS_MB (default 3700: Qwen3-4B's 36 layers are 3.63 GB, and
   held; a graph beyond what the NPU holds fails, and the layers stop at the
   one before) */
static uint32_t layers_budget(const struct janas_llm_model *m)
{
    const char *e = getenv("JANAS_NPU_LAYERS_MB");
    size_t budget = (size_t)(e && atol(e) > 0 ? atol(e) : 3700) << 20,
           used = in_elems(m, 0);
    uint32_t l = 0;
    while (l < m->n_layer) {
        size_t add = body_elems(m, l) + (l ? in_elems(m, l) : 0);
        if (used + add > budget)
            break;
        used += add, l++;
    }
    return l;
}

static void layers_free(struct janas_m_npu *n, uint32_t n_layer)
{
    janas_npu_mm_destroy(n->lpre);
    n->lpre = NULL;
    for (uint32_t l = 0; n->lg && l < n_layer; l++)
        janas_npu_mm_destroy(n->lg[l]);
    free(n->lg);
    n->lg = NULL;
}

/* the graphs, the first time (or after the arena changed) */
static int layers_make_(struct janas_llm_model *m, struct janas_m_npu *n);

static int layers_make(struct janas_llm_model *m, struct janas_m_npu *n)
{
    double t0 = now();
    int r = layers_make_(m, n);
    n->setup_s += now() - t0;
    prep_end(n);
    return r;
}

static int layers_make_(struct janas_llm_model *m, struct janas_m_npu *n)
{
    char err[256] = "";
    int B = n->hb, d = (int)m->d_model, qd = 0, aw = 0, kvd = 0, rec = 0;
    for (uint32_t l = 0; l < m->n_layer; l++) {
        const struct layer *ly = &m->layers[l];
        int w = ly->rec ? (int)m->d_inner : (int)(m->n_head * ly->hd);
        aw = w > aw ? w : aw;
        if (ly->rec) {
            rec = 1;
            continue;
        }
        int q = (m->a->attn_gate ? 2 : 1) * (int)(m->n_head * ly->hd);
        qd = q > qd ? q : qd;
        if (ly->n_kv * ly->hd > (uint32_t)kvd)
            kvd = (int)(ly->n_kv * ly->hd);
    }
    for (int h = 0; h < 2 && !n->att[1]; h++) {
        n->xs[h][0] = janas_npu_alloc(n->u, sizeof(float) * B * d);
        n->xs[h][1] = janas_npu_alloc(n->u, sizeof(float) * B * d);
        n->att[h] = janas_npu_alloc(n->u, sizeof(float) * B * aw);
        if (qd) {
            n->q[h] = janas_npu_alloc(n->u, sizeof(float) * B * qd);
            n->k[h] = janas_npu_alloc(n->u, sizeof(float) * B * kvd);
            n->v[h] = janas_npu_alloc(n->u, sizeof(float) * B * kvd);
        }
        if (rec) {
            n->big[h] = janas_npu_alloc(n->u, sizeof(float) * B * m->qkvz_dim);
            n->hn[h] = janas_npu_alloc(n->u, sizeof(float) * B * d);
        }
        if (m->ple_dim && !(n->ple[h] = janas_npu_alloc(
                                n->u, sizeof(float) * B * m->ple_dim)))
            return -1;
        if (!n->xs[h][0] || !n->xs[h][1] || !n->att[h] ||
            (qd && (!n->q[h] || !n->k[h] || !n->v[h])) ||
            (rec && (!n->big[h] || !n->hn[h])))
            return -1;
        memset(n->xs[h][0], 0, sizeof(float) * B * d);
        memset(n->xs[h][1], 0, sizeof(float) * B * d);
        memset(n->att[h], 0, sizeof(float) * B * aw);
    }
    double w0 = now();
    n->lg = calloc(m->n_layer, sizeof(*n->lg));
    int mode = n->a8 == 2 ? 2 : 0;
    uint32_t want = layers_budget(m), made = 0;
    if (10 * (size_t)want < 9 * (size_t)m->n_layer && !n->partial) {
        if (getenv("JANAS_NPU_STATS"))
            fprintf(stderr,
                    "janas: NPU: %u layers of %u would fit: the products "
                    "instead\n",
                    want, m->n_layer);
        layers_free(n, m->n_layer);
        return -1;
    }
    prep_begin(n, (int)want + 1);
    int rc = n->lg && want ? first_graph(m, n, mode, err, sizeof err) : -1;
    atomic_fetch_add(&n->prep_done, 1);
    while (rc == 0 && made < want) {
        rc = layer_graph(m, n, made, mode, made + 1 == want, err, sizeof err);
        made += rc == 0;
        atomic_fetch_add(&n->prep_done, 1);
    }
    if (rc && made > 0 &&
        (n->partial || 10 * (size_t)made >= 9 * (size_t)m->n_layer)) {
        /* the NPU held fewer than the budget: the layers made, the last
           one's graph again without the next layer's input part */
        if (getenv("JANAS_NPU_STATS"))
            fprintf(stderr, "janas: NPU: %u layers of %u (%s)\n", made, want,
                    err[0] ? err : "no memory");
        janas_npu_mm_destroy(n->lg[made - 1]);
        n->lg[made - 1] = NULL;
        rc = layer_graph(m, n, made - 1, mode, 1, err, sizeof err);
    }
    if (rc || !made) {
        fprintf(stderr, "janas: NPU: the layers stay apart: %s\n",
                err[0]  ? err
                : !want ? "none fits"
                        : "no memory");
        layers_free(n, m->n_layer);
        return -1;
    }
    n->nl = made;
    n->layers_compile_s += now() - w0;
    /* the products' and the FFNs' graphs are not used beside these */
    for (size_t i = 0; i < NPU_SLOTS; i++)
        if (n->e[i].mm)
            drop(n, &n->e[i]);
    n->lmode_loads = n->arena_loads;
    return 0;
}

/* the attention's output gated (Qwen3.5), item = token: as
   att_out_tok does it on the CPU's path, before wo */
struct gate_job {
    struct janas_llm_model *m;
    float *att;
    uint32_t qd, n;
};

static void npu_gate_worker(void *arg, int tid, int n_threads)
{
    struct gate_job *g = arg;
    for (uint32_t j = (uint32_t)tid; j < g->n; j += (uint32_t)n_threads) {
        float *a = g->att + (size_t)j * g->qd;
        const float *q = g->m->qgate + (size_t)j * g->qd;
        for (uint32_t i = 0; i < g->qd; i++)
            a[i] *= sigmoid(q[i]);
    }
}

static void npu_gate(struct janas_llm_model *m, const struct layer *ly,
                     float *att, uint32_t n)
{
    struct gate_job g = {m, att, m->n_head * ly->hd, n};
    janas_pool_run(m->compute, npu_gate_worker, &g);
}

/*
 * The NPU's answers against llama.cpp's (llm_eval, blocks of 256, two texts
 * of 256 tokens, the CPU's in brackets; 21 Oct 2026): at par in whole
 * layers - Qwen3.5-0.8B 490 (497), 2B 489 (493), Gemma 4 E2B 487 (490), E4B
 * 491 (495) of 512 - and Qwen's mixtures in products - Qwen3-Next-80B 492
 * (497), Qwen3-30B-A3B 454 (446), Qwen3.6-35B-A3B 485 (485); Gemma 4 in
 * products lost - 26B-A4B 363 (398), E2B 476 (490), the 12B 389 (383).
 * There the tuner leaves the NPU alone unless asked.
 */
int janas_m_npu_auto(const struct janas_llm_model *m)
{
    if (!m->npu)
        return 0;
    if (!m->a->sandwich)
        return 1;
    const char *e = getenv("JANAS_NPU_LAYERS");
    return !(e && atoi(e) <= 0) && layers_fit(m) &&
           10 * (size_t)layers_budget(m) >= 9 * (size_t)m->n_layer;
}

int janas_llm_model_npu_auto(const struct janas_llm_model *m)
{
    return janas_m_npu_auto(m);
}

static int wait_graph(struct janas_m_npu *n, struct janas_npu_mm *g, int h)
{
    double t0 = now();
    int r = janas_npu_mm_finish_v(g, h);
    n->t_layer_wait += now() - t0;
    return r;
}

int janas_m_npu_layers(struct janas_llm_model *m, uint32_t nt, uint32_t pos0)
{
    struct janas_m_npu *n = m->npu;
    if (!n || !janas_m_npu_takes(m, nt) || (int)nt > n->block || n->lmode < 0)
        return -1;
    /* a picture whose tokens attend both ways: the CPU's attention */
    if (m->n_media && janas_m_media_both(m, pos0, nt))
        return -1;
    if (n->lmode == 0 && !n->lg) {
        const char *e = getenv("JANAS_NPU_LAYERS");
        if ((e && atoi(e) <= 0) || !layers_fit(m)) {
            n->lmode = -1;
            return -1;
        }
        n->partial = e && atoi(e) >= 2;
        /* too few layers fit: the products from the first block on, known
           before the arena is whole (below) */
        if (10 * (size_t)layers_budget(m) < 9 * (size_t)m->n_layer &&
            !n->partial) {
            n->lmode = -1;
            return -1;
        }
        /* INT8 activations: calibrated on the first blocks
           (JANAS_NPU_A8_CALIB, default 1) in FP16, then INT8 graphs */
        const char *a8 = getenv("JANAS_NPU_A8"),
                   *ca = getenv("JANAS_NPU_A8_CALIB");
        if (!n->amax && n->a8 == 0 && a8 && atoi(a8) > 0 &&
            m->a->rec == JANAS_REC_NONE && !m->a->attn_gate &&
            !m->a->sandwich && calib_alloc(m, n) == 0) {
            n->a8 = 1;
            n->a8_left = ca && atoi(ca) > 0 ? atoi(ca) : 1;
            /* by default the inputs of q, k, v and of gate, up: on
               Qwen3-4B 299 tokens/s against 268 in FP16, the last token's
               logits 0.60 from the CPU's on average against 0.33; all
               four 307 and 4.32, the attention's output and the
               intermediate needing a scale for each token (20 Oct 2026) */
            const char *si = getenv("JANAS_NPU_A8_SITES");
            n->a8_sites = si ? atoi(si) & 15 : 1 << SITE_QKV | 1 << SITE_FFN;
        }
    }
    /* the pass before was a calibration's: counted, and after the last
       the INT8 graphs */
    if (n->in_calib) {
        n->in_calib = 0;
        if (--n->a8_left <= 0) {
            n->a8 = 2;
            if (getenv("JANAS_NPU_STATS"))
                fprintf(stderr, "janas: NPU calibrated: INT8 activations "
                                "from this block\n");
        }
    }
    if (n->a8 == 1) { /* on the CPU's path, the maxima taken on the way */
        n->in_calib = 1;
        m->npu_use = 0;
        return -1;
    }
    /* the FFNs are read in the arena: whole and unchanged since made. While
       it fills - the first blocks after the model opens - a block goes
       without the NPU rather than have it make the products' graphs that
       the whole layers drop once the arena is whole (5 s of Qwen3.5-2B's
       first block in janas-chat, 21 Oct 2026) */
    if (!n->arena) {
        m->npu_use = 0;
        return -1;
    }
    if (n->lmode == 1 && n->arena_loads != n->lmode_loads) {
        layers_free(n, m->n_layer);
        n->lmode = 0;
    }
    if (n->lmode == 0) {
        n->hb = n->block / 2;
        if (layers_make(m, n)) {
            n->lmode = -1;
            return -1;
        }
        n->lmode = 1;
    }
    uint32_t d = m->d_model, hb = (uint32_t)n->hb;
    uint32_t c[2] = {nt < hb ? nt : hb, nt > hb ? nt - hb : 0};
    int H = c[1] ? 2 : 1;
    for (int h = 0; h < H; h++)
        memcpy(n->xs[h][0], m->x + (size_t)h * hb * d,
               sizeof(float) * c[h] * d);
    int rc = 0;
    for (int h = 0; h < H && rc == 0; h++)
        rc = janas_npu_mm_begin_v(n->lpre, h);
    for (int h = 0; h < H; h++)
        if (wait_graph(n, n->lpre, h))
            rc = -1;

    if (rc)
        return -1; /* nothing written yet: the CPU's path from here */
    float *sq = m->q, *sk = m->kk, *sv = m->vv, *sa = m->att, *sb = m->big;
    /* the pool's idle workers asleep at once, not spinning: they wait
       most of the pass for the NPU, and cores spinning at their highest
       clock slowed it (unless JANAS_NPU_SPIN=1) */
    int spin = npu_quiet(m);
    /* layer by layer, half by half: the attention (or recurrence) of half
       h on the CPU while the NPU runs the layer on the other */
    for (uint32_t l = 0; l < n->nl && rc == 0; l++)
        for (int h = 0; h < H && rc == 0; h++) {
            if (l > 0 && wait_graph(n, n->lg[l - 1], h)) {
                rc = -1;
                break;
            }
            const struct layer *ly = &m->layers[l];
            m->att = n->att[h];
            double ta = now();
            if (ly->rec) {
                /* the second half goes on from the state the first left,
                   the tokens to replay (if any) already in it */
                m->big = n->big[h];
                janas_m_rec_ba(m, ly, n->hn[h], c[h]);
                uint32_t rp = m->replay;
                if (h)
                    m->replay = 0, m->rec_cont = 1;
                janas_m_rec_core(m, ly, c[h], ta);
                m->replay = rp, m->rec_cont = 0;
            } else {
                /* with output gates, q and the gates as m->big holds them:
                   the core puts q in m->q, the gates in m->qgate */
                if (m->a->attn_gate)
                    m->big = n->q[h];
                else
                    m->q = n->q[h];
                m->kk = n->k[h], m->vv = n->v[h];
                if (ly->wk && !ly->wv) /* Gemma 4's full layers: V is K, as
                                          before K's norm */
                    memcpy(n->v[h], n->k[h],
                           sizeof(float) * c[h] * ly->n_kv * ly->hd);
                /* the RoPE tables are the block's, row j its token j: the
                   second half starts hb rows in (a window layer's tables
                   apart) */
                size_t ro = (size_t)h * hb * (ly->n_rot / 2);
                float **rc_ = ly->swa ? &m->rope_cos_swa : &m->rope_cos,
                      **rs_ = ly->swa ? &m->rope_sin_swa : &m->rope_sin;
                *rc_ += ro, *rs_ += ro;
                janas_m_attn_core(m, ly, c[h], pos0 + (uint32_t)h * hb, ta);
                *rc_ -= ro, *rs_ -= ro;
                if (m->a->attn_gate)
                    npu_gate(m, ly, n->att[h], c[h]);
            }
            n->t_layer_attn += now() - ta;
            if (m->ple_dim) { /* this layer's per-layer inputs, half h's */
                uint32_t P = m->ple_dim, W = m->n_layer * P;
                for (uint32_t j = 0; j < c[h]; j++)
                    memcpy(n->ple[h] + (size_t)j * P,
                           m->ple + ((size_t)h * hb + j) * W + (size_t)l * P,
                           sizeof(float) * P);
            }
            rc = janas_npu_mm_begin_v(n->lg[l], h);
        }
    for (int h = 0; h < H; h++) /* the last layer's, and any in flight */
        for (uint32_t l = 0; l < n->nl; l++)
            janas_npu_mm_finish_v(n->lg[l], h);
    m->q = sq, m->kk = sk, m->vv = sv, m->att = sa, m->big = sb;
    npu_loud(m, spin);
    if (rc) { /* the keys and values of some layers are written: no way
                 back to the CPU's path for this block */
        fprintf(stderr, "janas: NPU failed in the middle of a block\n");
        return -2;
    }
    for (int h = 0; h < H; h++)
        memcpy(m->x + (size_t)h * hb * d, n->xs[h][n->nl % 2],
               sizeof(float) * c[h] * d);
    n->layer_passes++;
    return (int)n->nl;
}

void janas_m_npu_calib(struct janas_llm_model *m, int site, uint32_t l,
                       const float *x, uint32_t rows, uint32_t width,
                       size_t stride)
{
    struct janas_m_npu *n = m->npu;
    if (!n || !n->in_calib || l >= m->n_layer)
        return;
    float *a = amax(n, l, site);
    for (uint32_t r = 0; a && r < rows; r++)
        for (uint32_t c = 0; c < width; c++) {
            float v = fabsf(x[r * stride + c]);
            a[c] = v > a[c] ? v : a[c];
        }
}
