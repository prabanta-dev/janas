/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_selfcheck.c - when the GPU opens, a product of every type it takes,
 * on the vector counts that pick each of its shaders, against the CPU's,
 * bit for bit (a fraction of a second, small random matrices). A shader
 * this device's driver gets wrong is left out: the subgroup-per-row ones,
 * the ones with the lanes on the vectors, q4k_mv (each family has another
 * shader for the same products); a type still wrong without them goes to
 * the CPU. On an Arc B580 a Q6_K shader gave text with no sense, issue
 * #43: with the check the engine answers right from the first run. The
 * result is kept in ~/.cache/janas/gpu_selfcheck under the device, its
 * driver's version, the subgroup size and Janas's version, so that the
 * check runs once per driver. JANAS_GPU_SELFCHECK=0 leaves it out
 * (gpu_check does, to see every shader), =2 runs it even when the cache
 * has the answer; JANAS_GPU_VERBOSE prints what it did. Not the fast
 * prompt's tiles (64 vectors and more, rounded otherwise: gpu_fast_check).
 */
#include "gpu.h"

#if defined(JANAS_GPU_SPV) && __has_include(<vulkan/vulkan.h>)
#include "gpu_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/aligned.h"
#include "common/pool.h"
#include "matvec.h"
#include "quant.h"
#include "tuner.h" /* janas_cache_dir */

#ifndef JANAS_VERSION
#define JANAS_VERSION "dev"
#endif

enum { ROWS = 128, MAXC = 640, NV_MAX = 9, SLOT = 128 << 10 };

/* the types the products take, their block's bytes and where its scales
   are (offset of d, of a second scale or -1); 640 wide only for the types
   that sum a tail */
static const struct kind {
    int type;
    unsigned block, step; /* bytes a block of 256 weights, a block's */
    int d, m;
    int tail;
} kinds[] = {
    {JANAS_Q4_K, 144, 144, 0, 2, 0},    {JANAS_Q5_K, 176, 176, 0, 2, 0},
    {JANAS_Q6_K, 210, 210, 208, -1, 0}, {JANAS_Q6_K_P, 82, 82, 80, -1, 0},
    {JANAS_Q8_0, 272, 34, 0, -1, 1},    {JANAS_Q4_0, 144, 18, 0, -1, 1},
    {JANAS_IQ4_NL, 144, 18, 0, -1, 1},  {JANAS_Q4_1, 160, 20, 0, 2, 1},
    {JANAS_Q5_1, 192, 24, 0, 2, 1},     {JANAS_MXFP4, 136, 17, -1, -1, 1},
    {JANAS_Q5_0, 176, 22, 0, -1, 1},    {JANAS_Q2_0, 72, 18, 0, -1, 1},
    {JANAS_IQ4_XS, 136, 136, 0, -1, 0}, {JANAS_Q3_K, 110, 110, 108, -1, 0},
    {JANAS_IQ3_S, 110, 110, 0, -1, 0},  {JANAS_Q2_K, 84, 84, 80, 82, 0},
    {JANAS_IQ2_XXS, 66, 66, 0, -1, 0},  {JANAS_IQ2_XS, 74, 74, 0, -1, 0},
    {JANAS_IQ2_S, 82, 82, 0, -1, 0},    {JANAS_IQ3_XXS, 98, 98, 0, -1, 0},
    {JANAS_IQ1_S, 50, 50, 0, -1, 0},    {JANAS_IQ1_M, 56, 56, -2, -1, 0}};
enum { N_KINDS = sizeof(kinds) / sizeof(kinds[0]) };

/* vector counts: one (the row shaders, or the one-vector ones), 2 and 3
   (the row shaders' variants, the lanes on the vectors), 5 and 9 (the
   shaders of many vectors) */
static const size_t nvs[] = {1, 2, 3, 5, 9};
enum { N_NVS = sizeof(nvs) / sizeof(nvs[0]) };

static uint64_t rng = 0x9e3779b97f4a7c15ull;

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

/* random weights of kind k, scales small and normal */
static void fill(const struct kind *k, uint8_t *w, size_t bytes)
{
    for (size_t i = 0; i < bytes; i++)
        w[i] = (uint8_t)rnd();
    for (size_t b = 0; b + k->step <= bytes; b += k->step) {
        uint16_t d = janas_fp32_to_fp16(0.0005f + (rnd() % 1000) * 1e-6f);
        if (k->type == JANAS_MXFP4) {
            w[b] = (uint8_t)(118 + rnd() % 8);
        } else if (k->d == -2) {
            /* IQ1_M: d in the top nibbles of its four scale words */
            for (int s = 0; s < 4; s++) {
                uint16_t sc;
                memcpy(&sc, w + b + 48 + 2 * s, 2);
                sc = (uint16_t)((sc & 0x0fff) | ((d >> 4 * s & 15) << 12));
                memcpy(w + b + 48 + 2 * s, &sc, 2);
            }
        } else {
            memcpy(w + b + k->d, &d, 2);
        }
        if (k->m >= 0) {
            uint16_t m =
                k->type == JANAS_Q4_1 || k->type == JANAS_Q5_1
                    ? janas_fp32_to_fp16(-0.01f + (rnd() % 1000) * 2e-5f)
                    : janas_fp32_to_fp16(0.0002f + (rnd() % 1000) * 1e-6f);
            memcpy(w + b + k->m, &m, 2);
        }
    }
}

/* one product wrong: kind, width, vectors and the shader that ran it */
struct wrong {
    int k;
    size_t cols, nv;
    char shader[96];
};

/* Runs kind k's product of cols columns on nv vectors on the GPU, all
   its rows, and compares it with the CPU's: 1 right, 0 wrong (shader
   named), -1 not taken by the GPU. */
static int check_one(struct janas_gpu *g, struct janas_pool *pool,
                     const uint8_t *w, const struct janas_block_q8k *x,
                     float *ya, float *yb, int k, size_t cols, size_t nv,
                     char *shader, size_t len)
{
    size_t nbx = (cols + JANAS_QK - 1) / JANAS_QK;
    struct janas_matvec_task t = {.type = kinds[k].type,
                                  .w = w,
                                  .x = x,
                                  .rows = ROWS,
                                  .cols = cols,
                                  .n_vec = nv,
                                  .x_stride = nbx};
    t.y = ya;
    memset(ya, 0, ROWS * nv * sizeof(float));
    janas_matvec_q4k_group(pool, &t, 1);
    if (!janas_gpu_can(g, &t))
        return -1;
    t.y = yb;
    memset(yb, 0, ROWS * nv * sizeof(float));
    size_t rows = ROWS;
    int waited;
    if (janas_gpu_begin(g, &t, &rows, 1) != 0)
        return -1;
    if (janas_gpu_finish(g, &waited) != 0)
        return 0;
    janas_gpu_last_shader(g, shader, len);
    return memcmp(ya, yb, ROWS * nv * sizeof(float)) == 0;
}

/* The cache's key for this device and driver, and its file. */
static void cache_key(const struct janas_gpu *g, char *key, size_t len)
{
    snprintf(key, len, "%04x:%04x:%08x:sg%u:%s:%s%s", g->vendor_id,
             g->device_id, g->driver_version, g->sg,
             janas_gpu_shares_host(g) ? "shared" : "copy", JANAS_VERSION,
             g->xfma ? ":xfma" : "");
}

/* What a saved check says: 1 found (switches and types applied). */
static int load(struct janas_gpu *g, const char *key, int *flags)
{
    char path[512];
    if (janas_cache_dir(path, sizeof(path) - 32, 0) != 0)
        return 0;
    strcat(path, "/gpu_selfcheck");
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    char line[512];
    int found = 0, row = 1, mvd = 1, mv = 1, off[16], n_off = 0;
    while (fgets(line, sizeof(line), f)) {
        char k[160], types[256] = "";
        int r, d, v;
        if (sscanf(line, "%159s row=%d mvd=%d mv=%d off=%255s", k, &r, &d, &v,
                   types) < 4 ||
            strcmp(k, key) != 0)
            continue;
        /* the last line of the key counts */
        found = 1;
        row = r;
        mvd = d;
        mv = v;
        n_off = 0;
        for (char *p = types; *p && n_off < 16;) {
            off[n_off++] = atoi(p);
            p = strchr(p, ',');
            if (!p)
                break;
            p++;
        }
        if (strcmp(types, "-") == 0)
            n_off = 0;
    }
    fclose(f);
    if (!found)
        return 0;
    g->row_on &= row;
    g->mvd_on &= mvd;
    g->mv_on &= mv;
    flags[0] = row;
    flags[1] = mvd;
    flags[2] = mv;
    for (int i = 0; i < n_off; i++)
        g->off_types[g->n_off++] = off[i];
    return 1;
}

static void save(const struct janas_gpu *g, const char *key, int row, int mvd,
                 int mv)
{
    char path[512];
    if (janas_cache_dir(path, sizeof(path) - 32, 1) != 0)
        return;
    strcat(path, "/gpu_selfcheck");
    FILE *f = fopen(path, "a");
    if (!f)
        return;
    fprintf(f, "%s row=%d mvd=%d mv=%d off=", key, row, mvd, mv);
    for (int i = 0; i < g->n_off; i++)
        fprintf(f, "%s%d", i ? "," : "", g->off_types[i]);
    fprintf(f, "%s\n", g->n_off ? "" : "-");
    fclose(f);
}

int janas_g_type_off(const struct janas_gpu *g, int type)
{
    for (int i = 0; i < g->n_off; i++)
        if (g->off_types[i] == type)
            return 1;
    return 0;
}

static const char *type_name(int type)
{
    switch (type) {
    case JANAS_Q4_K:
        return "Q4_K";
    case JANAS_Q5_K:
        return "Q5_K";
    case JANAS_Q6_K:
        return "Q6_K";
    case JANAS_Q6_K_P:
        return "Q6_K_P";
    case JANAS_Q8_0:
        return "Q8_0";
    case JANAS_Q4_0:
        return "Q4_0";
    case JANAS_IQ4_NL:
        return "IQ4_NL";
    case JANAS_Q4_1:
        return "Q4_1";
    case JANAS_Q5_1:
        return "Q5_1";
    case JANAS_MXFP4:
        return "MXFP4";
    case JANAS_Q5_0:
        return "Q5_0";
    case JANAS_Q2_0:
        return "Q2_0";
    case JANAS_IQ4_XS:
        return "IQ4_XS";
    case JANAS_Q3_K:
        return "Q3_K";
    case JANAS_IQ3_S:
        return "IQ3_S";
    case JANAS_Q2_K:
        return "Q2_K";
    case JANAS_IQ2_XXS:
        return "IQ2_XXS";
    case JANAS_IQ2_XS:
        return "IQ2_XS";
    case JANAS_IQ2_S:
        return "IQ2_S";
    case JANAS_IQ3_XXS:
        return "IQ3_XXS";
    case JANAS_IQ1_S:
        return "IQ1_S";
    case JANAS_IQ1_M:
        return "IQ1_M";
    }
    return "?";
}

/* The line for the user: what the check left out (row, mvd, mv: 0 where
   it turned that family off). */
static void make_note(struct janas_gpu *g, int row, int mvd, int mv)
{
    char *p = g->check_note;
    size_t len = sizeof(g->check_note);
    int n = 0;
    p[0] = 0;
    const char *fam[3] = {"the subgroup-per-row shaders",
                          "the lanes-on-vectors shaders", "q4k_mv"};
    int off[3] = {!row, !mvd, !mv};
    for (int i = 0; i < 3; i++)
        if (off[i] && (size_t)n < len)
            n +=
                snprintf(p + n, len - (size_t)n, "%s%s", n ? ", " : "", fam[i]);
    for (int i = 0; i < g->n_off && (size_t)n < len; i++)
        n += snprintf(p + n, len - (size_t)n, "%s%s",
                      i   ? " "
                      : n ? "; on the CPU: "
                          : "on the CPU: ",
                      type_name(g->off_types[i]));
    if (p[0] && (size_t)n < len)
        snprintf(p + n, len - (size_t)n, " (wrong on this driver)");
}

/* Every product once; the wrong ones in bad (up to max), their count
   returned. Only the cases in only (n_only of them), where given. */
static int run_all(struct janas_gpu *g, struct janas_pool *pool, uint8_t *w,
                   const struct janas_block_q8k *x, float *ya, float *yb,
                   const struct wrong *only, int n_only, struct wrong *bad,
                   int max)
{
    int n = 0;
    for (int k = 0; k < N_KINDS; k++)
        for (int c = 0; c < 1 + kinds[k].tail; c++)
            for (int v = 0; v < N_NVS; v++) {
                size_t cols = c ? MAXC : 512, nv = nvs[v];
                int want = !only;
                for (int i = 0; only && i < n_only; i++)
                    want |= only[i].k == k && only[i].cols == cols &&
                            only[i].nv == nv;
                if (!want)
                    continue;
                char sh[96];
                if (check_one(g, pool, w + (size_t)k * SLOT, x, ya, yb, k, cols,
                              nv, sh, sizeof(sh)) == 0) {
                    if (n < max)
                        bad[n] = (struct wrong){.k = k, .cols = cols, .nv = nv};
                    if (n < max)
                        snprintf(bad[n].shader, sizeof(bad[n].shader), "%s",
                                 sh);
                    n++;
                }
            }
    return n < max ? n : max;
}

void janas_g_selfcheck(struct janas_gpu *g)
{
    const char *e = getenv("JANAS_GPU_SELFCHECK");
    int mode = e ? atoi(e) : 1;
    if (mode == 0 || g->broken)
        return;
    char key[160];
    cache_key(g, key, sizeof(key));
    int flags[3];
    if (mode == 1 && load(g, key, flags)) {
        make_note(g, flags[0], flags[1], flags[2]);
        return;
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    size_t wbytes = (size_t)N_KINDS * SLOT;
    uint8_t *w = janas_aligned_alloc(4096, wbytes);
    size_t nbx = (MAXC + JANAS_QK - 1) / JANAS_QK;
    struct janas_block_q8k *x = malloc(NV_MAX * nbx * sizeof(*x));
    float *ya = malloc(ROWS * NV_MAX * sizeof(float)),
          *yb = malloc(ROWS * NV_MAX * sizeof(float)), xf[3 * JANAS_QK];
    struct janas_pool *pool = janas_pool_create(2, NULL);
    if (!w || !x || !ya || !yb || !pool)
        goto done;
    for (int k = 0; k < N_KINDS; k++)
        fill(&kinds[k], w + (size_t)k * SLOT, SLOT);
    for (size_t v = 0; v < NV_MAX; v++) {
        /* the activations of a row ending inside a group: its last group
           whole, past the row's end zero (model_ffn.c) */
        for (size_t i = 0; i < nbx * JANAS_QK; i++)
            xf[i] =
                i < MAXC ? (float)((int)(rnd() % 2001) - 1000) / 500.0f : 0.0f;
        janas_q8k_quantize(xf, x + v * nbx, nbx * JANAS_QK);
    }
    if (janas_gpu_import(g, w, wbytes) != 0)
        goto done;
    int was_row = g->row_on, was_mvd = g->mvd_on, was_mv = g->mv_on;
    enum { MAXW = 64 };
    struct wrong bad[MAXW];
    int n = run_all(g, pool, w, x, ya, yb, NULL, 0, bad, MAXW);
    if (n) {
        /* the families with another shader for the same products off, as
           far as one of theirs was wrong; then again */
        for (int i = 0; i < n; i++) {
            if (strstr(bad[i].shader, "_row "))
                g->row_on = 0;
            if (strstr(bad[i].shader, "_mvd "))
                g->mvd_on = 0;
            if (strstr(bad[i].shader, "q4k_mv "))
                g->mv_on = 0;
        }
        struct wrong still[MAXW];
        int m = run_all(g, pool, w, x, ya, yb, bad, n, still, MAXW);
        /* what is still wrong: the type to the CPU */
        for (int i = 0; i < m; i++)
            if (!janas_g_type_off(g, kinds[still[i].k].type) && g->n_off < 16)
                g->off_types[g->n_off++] = kinds[still[i].k].type;
        make_note(g, g->row_on || !was_row, g->mvd_on || !was_mvd,
                  g->mv_on || !was_mv);
        fprintf(stderr,
                "gpu: %d of the GPU's products differ from the CPU's on this "
                "driver (first: %s %s, %zu wide, %zu vectors); left out: "
                "%s (JANAS_GPU_SELFCHECK=0 keeps them, gpu_check tells "
                "more)\n",
                n, bad[0].shader, type_name(kinds[bad[0].k].type), bad[0].cols,
                bad[0].nv, g->check_note);
    }
    save(g, key, g->row_on || !was_row, g->mvd_on || !was_mvd,
         g->mv_on || !was_mv);
    janas_g_drop_region(g, w);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (g->verbose || getenv("JANAS_GPU_VERBOSE"))
        fprintf(stderr, "gpu self-check: %d wrong, %.0f ms (%s)\n", n,
                (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6,
                g->check_note[0] ? g->check_note : "all right");
done:
    janas_pool_destroy(pool);
    janas_aligned_free(w);
    free(x);
    free(ya);
    free(yb);
}

const char *janas_gpu_check_note(const struct janas_gpu *g)
{
    if (g->cuda)
        return "";
    return g->check_note;
}

#else

const char *janas_gpu_check_note(const struct janas_gpu *g)
{
    (void)g;
    return "";
}

#endif
