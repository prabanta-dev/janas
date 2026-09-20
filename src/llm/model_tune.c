/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_tune.c - the threads and GPU each kind of pass runs with, and the
 * profile of the experts a machine uses (see model.h).
 */
#define _GNU_SOURCE
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "attention.h"
#include "common/pool.h"
#include "common/sysinfo.h"
#include "deltanet.h"
#include "tuner.h"
#include "gpu.h"
#include "matvec.h"
#include "quant.h"
#include "vmath.h"

#include "model_impl.h"

/*
 * The tuner's candidates: every usable core - the performance cores with
 * all their threads and the efficiency cores, not the low-power island,
 * which the layout leaves out - with and without the GPU if there is one.
 * Measured on 26 Sep 2026 on six models (Qwen3.5-2B to Qwen3-Next-80B),
 * that was the fastest or level in every kind of pass; one thread per
 * performance core, or their threads alone, lost 5-20% writing. Trying
 * them in use only made a chat run on fewer cores than janas-bench had
 * reported, whenever a stretch of long context had tipped the averages:
 * the same machine, two speeds. So the threads are fixed, and the tuner
 * only decides whether the GPU helps. JANAS_TUNE_THREADS=1 brings back
 * those candidates (one per performance core, all their threads), to
 * experiment with. JANAS_TUNE=0 turns tuning off; JANAS_BLOCK_THREADS fixes
 * the threads for blocks (and turns it off too); JANAS_GPU_ALWAYS=1 makes
 * the GPU variants the defaults for several tokens.
 */
void janas_m_init_tuner(struct janas_llm_model *m)
{
    struct janas_tune_cand c[JANAS_TUNE_MAX_CANDS];
    int n = 0, opts[3] = {m->all_threads, m->p_cores, m->p_threads};
    const char *bt = getenv("JANAS_BLOCK_THREADS"), *tn = getenv("JANAS_TUNE"),
               *tt = getenv("JANAS_TUNE_THREADS");
    int n_opts = tt && atoi(tt) > 0 ? 3 : 1;
    int block = bt && atoi(bt) > 0 ? atoi(bt) : m->all_threads;
    if (block > m->all_threads)
        block = m->all_threads;
    for (int gpu = 0; gpu <= (m->gpu && !m->gpu_off ? 1 : 0); gpu++)
        for (int i = 0; i < n_opts; i++) {
            int dup = 0;
            for (int k = 0; k < n; k++)
                dup |= c[k].threads == opts[i] && c[k].gpu == gpu;
            if (!dup && n < JANAS_TUNE_MAX_CANDS)
                c[n++] =
                    (struct janas_tune_cand){.threads = opts[i], .gpu = gpu};
        }
    int def[JANAS_TUNE_CLASSES] = {0, 0, 0};
    for (int k = 0; k < n; k++) {
        if (!c[k].gpu && c[k].threads == m->all_threads)
            def[0] = k;
        if (!c[k].gpu && c[k].threads == block)
            def[1] = def[2] = k;
    }
    /* JANAS_GPU_ALWAYS=1: the GPU variant as the default for passes over
       several tokens (tests, and with JANAS_TUNE=0 a fixed GPU setup) */
    const char *ga = getenv("JANAS_GPU_ALWAYS");
    if (ga && atoi(ga) > 0)
        for (int k = 0; k < n; k++)
            if (c[k].gpu && c[k].threads == block)
                def[1] = def[2] = k;
    if (bt && block != c[def[1]].threads && n < JANAS_TUNE_MAX_CANDS) {
        c[n] = (struct janas_tune_cand){.threads = block, .gpu = 0};
        def[1] = def[2] = n++;
    }
    janas_tuner_init(&m->tuner, c, n, def, (tn && atoi(tn) == 0) || bt);
    janas_tuner_allow_gpu(&m->tuner, 1);
    /* the saved choices of this machine and model */
    char cpu[128], name[128] = "";
    janas_cpu_name(cpu, sizeof(cpu));
    janas_jns_meta_str(&m->j, "general.name", name, sizeof(name));
    int w = snprintf(m->tune_key, sizeof(m->tune_key), "v2|%s|%d/%d/%d|%s|%s|",
                     cpu, m->p_cores, m->p_threads, m->all_threads,
                     m->gpu && !m->gpu_off ? janas_gpu_name(m->gpu) : "no GPU",
                     name);
    for (int k = 0; k < n && w > 0 && (size_t)w < sizeof(m->tune_key); k++)
        w += snprintf(m->tune_key + w, sizeof(m->tune_key) - (size_t)w, "%d%s,",
                      c[k].threads, c[k].gpu ? "g" : "");
    janas_tuner_load(&m->tuner, m->tune_key);
}

/* Threads and GPU of a candidate, for the passes that follow. */
void janas_m_apply_candidate(struct janas_llm_model *m, int cand)
{
    /* a borrowed pool belongs to the model that made it: a drafting model
       setting its own thread count would set the target's too */
    if (m->own_compute)
        janas_pool_set_active(m->compute, m->tuner.cand[cand].threads);
    m->gpu_use = m->tuner.cand[cand].gpu;
    if (m->gpu)
        janas_gpu_keep_awake(m->gpu, m->gpu_use);
}

void janas_llm_model_set_eco(struct janas_llm_model *m, int eco)
{
    janas_tuner_allow_gpu(&m->tuner, !eco);
}

void janas_llm_model_tuning(const struct janas_llm_model *m, char *buf,
                            size_t len)
{
    janas_tuner_report(&m->tuner, buf, len);
}

int janas_llm_model_threads(const struct janas_llm_model *m, int cls)
{
    if (cls < 0 || cls >= JANAS_TUNE_CLASSES || m->tuner.n_cand <= 0)
        return 0;
    return m->tuner.cand[janas_tuner_best(&m->tuner, cls)].threads;
}

int janas_llm_model_gpu_used(const struct janas_llm_model *m)
{
    if (!m->gpu || m->gpu_off)
        return 0;
    int used = 0;
    for (int k = 0; k < JANAS_TUNE_CLASSES; k++) {
        int b = janas_tuner_best(&m->tuner, k);
        if (m->tuner.cand[b].gpu)
            used |= 1 << k;
    }
    return used;
}

int janas_llm_model_candidates(const struct janas_llm_model *m, int *threads,
                               int *gpu, int max)
{
    int n = m->tuner.n_cand < max ? m->tuner.n_cand : max;
    for (int i = 0; i < n; i++) {
        threads[i] = m->tuner.cand[i].threads;
        gpu[i] = m->tuner.cand[i].gpu;
    }
    return n;
}

void janas_llm_model_force(struct janas_llm_model *m, int cand)
{
    janas_tuner_force(&m->tuner, cand);
}

/*
 * The experts this machine has used most with this model, counted across
 * sessions in ~/.cache/janas. Without them the first long reply discovers
 * its experts one token at a time: measured on 20 Sep 2026, the second reply
 * of a chat read 14 GB and waited 3.3 seconds for the disk, while every
 * reply after it read 3 GB and waited half a second. The file is small (four
 * bytes per expert of the model) and its counts are added to, so the profile
 * follows how the machine is used.
 */
struct expert_profile {
    char magic[8]; /* JNSXPRT1 */
    uint32_t n_layer, n_expert;
    uint64_t model; /* name, geometry and size of the model file */
    uint64_t tokens;
};

static uint64_t model_fingerprint(const struct janas_llm_model *m)
{
    char name[128] = "";
    janas_jns_meta_str(&m->j, "general.name", name, sizeof(name));
    uint64_t h = janas_fnv1a(name, strlen(name), JANAS_FNV_INIT);
    uint64_t v[3] = {m->j.h.n_layer, m->j.h.n_expert, m->j.h.file_bytes};
    return janas_fnv1a(v, sizeof(v), h);
}

static int profile_path(const struct janas_llm_model *m, char *path, size_t len,
                        int create)
{
    if (janas_cache_dir(path, len, create) != 0)
        return -1;
    size_t n = strlen(path);
    snprintf(path + n, len - n, "/experts-%016llx.bin",
             (unsigned long long)model_fingerprint(m));
    return 0;
}

int janas_llm_model_cache_file(const struct janas_llm_model *m,
                               const char *what, char *path, size_t len,
                               int create)
{
    if (janas_cache_dir(path, len, create) != 0)
        return -1;
    size_t n = strlen(path);
    snprintf(path + n, len - n, "/%s-%016llx.bin", what,
             (unsigned long long)model_fingerprint(m));
    return 0;
}

/* The saved counts, or NULL; *used gets how many experts have a count. */
uint32_t *janas_m_profile_read(const struct janas_llm_model *m, size_t *used)
{
    char path[1024];
    if (profile_path(m, path, sizeof(path), 0) != 0)
        return NULL;
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    struct expert_profile h;
    size_t n_keys = (size_t)m->j.h.n_layer * m->j.h.n_expert;
    uint32_t *c = NULL;
    if (fread(&h, sizeof(h), 1, f) == 1 &&
        memcmp(h.magic, "JNSXPRT1", 8) == 0 && h.n_layer == m->j.h.n_layer &&
        h.n_expert == m->j.h.n_expert && h.model == model_fingerprint(m) &&
        (c = calloc(n_keys, sizeof(*c))) != NULL &&
        fread(c, sizeof(*c), n_keys, f) != n_keys) {
        free(c);
        c = NULL;
    }
    fclose(f);
    if (c && used)
        for (size_t k = 0; k < n_keys; k++)
            *used += c[k] != 0;
    return c;
}

/* This session's counts added to the file's, halved if they grow large. */
void janas_m_profile_write(const struct janas_llm_model *m)
{
    size_t n_keys = (size_t)m->j.h.n_layer * m->j.h.n_expert;
    uint32_t *c = janas_m_profile_read(m, NULL);
    if (!c && !(c = calloc(n_keys, sizeof(*c))))
        return;
    uint32_t mx = 0;
    for (size_t k = 0; k < n_keys; k++) {
        uint64_t v = (uint64_t)c[k] + m->use_count[k];
        c[k] = v > 0xfffffffful ? 0xfffffffful : (uint32_t)v;
        if (c[k] > mx)
            mx = c[k];
    }
    /* old sessions must not freeze the profile: halve when counts grow */
    if (mx > (1u << 28))
        for (size_t k = 0; k < n_keys; k++)
            c[k] /= 2;
    char path[1024], tmp[1100];
    if (profile_path(m, path, sizeof(path), 1) == 0) {
        snprintf(tmp, sizeof(tmp), "%s.new", path);
        FILE *f = fopen(tmp, "wb");
        if (f) {
            struct expert_profile h = {
                .magic = {'J', 'N', 'S', 'X', 'P', 'R', 'T', '1'},
                .n_layer = m->j.h.n_layer,
                .n_expert = m->j.h.n_expert,
                .model = model_fingerprint(m),
                .tokens = 0};
            int ok = fwrite(&h, sizeof(h), 1, f) == 1 &&
                     fwrite(c, sizeof(*c), n_keys, f) == n_keys;
            ok = fclose(f) == 0 && ok;
            if (ok)
                rename(tmp, path);
            else
                remove(tmp);
        }
    }
    free(c);
}
