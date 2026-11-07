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
#include "common/devices.h"
#include "tuner.h"
#include "gpu.h"
#include "matvec.h"
#include "quant.h"
#include "vmath.h"

#include "model_impl.h"

/* a mask's combinations without the NPU: what they are in the passes it
   does not take */
static int without_npu(int mask)
{
    int out = 0;
    for (int c = 0; c < JANAS_DEV_COMBOS; c++)
        if (mask >> c & 1)
            out |= 1 << (c & ~JANAS_DEV_NPU);
    return out;
}

/*
 * The user's devices for each kind of pass (--prompt-on, --drafts-on,
 * --reply-on; common/devices.h), in on[] in the order of
 * janas_devices_vars, as far as this machine has them and the pass can use
 * them. The NPU takes blocks only (JANAS_TUNE_BLOCK tokens and more): for
 * the other passes a combination with it is the rest of it, said where it
 * was asked for those passes alone. Prompts kept off the GPU keep its fast
 * prompt off them too; replies asked of it give it single tokens, which an
 * integrated one does not take by default. A device that is not here: the
 * engine chooses, and says why.
 */
static void user_devices(struct janas_llm_model *m, int on[])
{
    static const char *const what[JANAS_DEVICES_OPTIONS] = {"prompts", "drafts",
                                                            "replies"};
    int gpu_here = m->gpu && !m->gpu_off,
        have = 1 | (gpu_here ? 1 << JANAS_DEV_GPU : 0) |
               (m->npu ? 1 << JANAS_DEV_NPU : 0) |
               (gpu_here && m->npu ? 1 << (JANAS_DEV_GPU | JANAS_DEV_NPU) : 0);
    for (int k = 0; k < JANAS_DEVICES_OPTIONS; k++) {
        const char *var = janas_devices_vars[k];
        char why[200], nm[64];
        int mask = janas_devices_mask(getenv(var), why, sizeof why), asked = 0;
        if (mask < 0) {
            fprintf(stderr, "janas: %s: %s; the engine chooses\n", var, why);
            mask = 0;
        }
        for (int c = 0; c < JANAS_DEV_COMBOS; c++)
            asked |= mask >> c & 1 ? c : 0;
        janas_devices_name(mask, nm, sizeof nm);
        if (mask && !(mask & have)) {
            fprintf(stderr,
                    "janas: %s=%s: not on this machine; the engine chooses\n",
                    var, nm);
            if (asked & JANAS_DEV_GPU && !gpu_here)
                fprintf(stderr, "janas:   GPU: %s\n",
                        m->gpu ? "it cannot share the model's memory"
                               : m->gpu_why);
            if (asked & JANAS_DEV_NPU && !m->npu)
                fprintf(stderr, "janas:   NPU: %s\n", m->npu_why);
            mask = 0;
        }
        mask &= mask ? have : 0;
        if (k > 0 && asked & JANAS_DEV_NPU && mask) {
            char rest[64];
            mask = without_npu(mask);
            janas_devices_name(mask, rest, sizeof rest);
            fprintf(stderr,
                    "janas: %s=%s: the NPU takes passes of %d tokens and more "
                    "only; %s on %s\n",
                    var, nm, JANAS_TUNE_BLOCK, what[k], rest);
        }
        on[k] = mask;
    }
    m->prompts_off_gpu =
        on[0] &&
        !(on[0] & (1 << JANAS_DEV_GPU | 1 << (JANAS_DEV_GPU | JANAS_DEV_NPU)));
    if (on[2] >> JANAS_DEV_GPU & 1 && janas_gpu_min_n(m->gpu) > 1)
        janas_gpu_set_min_n(m->gpu, 1);
}

/*
 * The tuner's candidates: every usable core - the performance cores with
 * all their threads and the efficiency cores, not the low-power island,
 * which the layout leaves out - one thread per performance core, and the
 * performance cores' threads alone; each with and without the GPU if there
 * is one. On the development laptop (6 performance cores, 8 efficiency)
 * every core was the fastest or level on six models; on an i9-14900HX (8
 * and 16) every thread replied at 9.9 tokens/s where 16 gave 24.7 (issue
 * #13), and on a 9950X3D one thread per core was 4-15% ahead (#15). So the
 * first passes of each kind try them all and the tuner keeps the threads
 * of the fastest (tuner.h: settled once, never tried again at a longer
 * context, which had moved chats onto fewer cores than they could use).
 * JANAS_TUNE_THREADS=0 keeps every core without trying the others;
 * JANAS_TUNE=0 turns tuning off; JANAS_BLOCK_THREADS fixes the threads for
 * blocks (and turns it off too); JANAS_GPU_ALWAYS=1 makes the GPU variants
 * the defaults for several tokens. The GPU variants are tried for one token
 * too where the GPU takes single tokens (a discrete GPU, or
 * JANAS_GPU_MIN_N=1): the default stays without it.
 */
void janas_m_init_tuner(struct janas_llm_model *m)
{
    struct janas_tune_cand c[JANAS_TUNE_MAX_CANDS];
    int n = 0, opts[3] = {m->all_threads, m->p_cores, m->p_threads};
    const char *bt = getenv("JANAS_BLOCK_THREADS"), *tn = getenv("JANAS_TUNE"),
               *tt = getenv("JANAS_TUNE_THREADS");
    int n_opts = tt && atoi(tt) == 0 ? 1 : 3;
    int block = bt && atoi(bt) > 0 ? atoi(bt) : m->all_threads;
    if (block > m->all_threads)
        block = m->all_threads;
    /* with an NPU every candidate with it too, for blocks only (tuner.c):
       the defaults stay without, unless the user asks for it */
    for (int npu = 0; npu <= (m->npu ? 1 : 0); npu++)
        for (int gpu = 0; gpu <= (m->gpu && !m->gpu_off ? 1 : 0); gpu++)
            for (int i = 0; i < n_opts; i++) {
                int dup = 0;
                for (int k = 0; k < n; k++)
                    dup |= c[k].threads == opts[i] && c[k].gpu == gpu &&
                           c[k].npu == npu;
                if (!dup && n < JANAS_TUNE_MAX_CANDS)
                    c[n++] = (struct janas_tune_cand){
                        .threads = opts[i], .gpu = gpu, .npu = npu};
            }
    int on[JANAS_DEVICES_OPTIONS];
    user_devices(m, on);
    memcpy(m->dev_on, on, sizeof m->dev_on);
    int def[JANAS_TUNE_CLASSES] = {0, 0, 0, 0};
    for (int k = 0; k < n && !c[k].npu; k++) {
        if (!c[k].gpu && c[k].threads == m->all_threads)
            def[0] = k;
        if (!c[k].gpu && c[k].threads == block)
            def[1] = def[2] = def[3] = k;
    }
    /* JANAS_GPU_ALWAYS=1: the GPU variant as the default for passes over
       several tokens (tests, and with JANAS_TUNE=0 a fixed GPU setup) */
    /* with the fast prompt the GPU computes nearly all of a block, and the
       CPU's threads mostly draw the power it would use (the laptop's PL4):
       blocks start on the performance cores' threads, not on all of them -
       of 12 models the fastest blocks were on 6 or 12 threads 11 times,
       Qwen3.5-2B 318 tokens/s on 6 and 309 on 12 against 272 on 20 (10 Oct
       2026) */
    /* the same with the NPU asked for blocks: it computes nearly all of
       one too (Qwen3-4B in whole layers, the threads spinning while they
       waited: 270.6 tokens/s on 6 threads, 262.0 on 12, 252.1 on 20; once
       they slept, 325.5 on 6 and 320.4 on 12; 20 Oct 2026) */
    int fast = m->gpu && !m->gpu_off && janas_gpu_fast_prompt() && !bt &&
               !m->prompts_off_gpu,
        npu_asked =
            on[0] & (1 << JANAS_DEV_NPU | 1 << (JANAS_DEV_GPU | JANAS_DEV_NPU));
    if (fast || (npu_asked && !bt))
        for (int k = 0; k < n; k++)
            if (!c[k].gpu && !c[k].npu && c[k].threads == m->p_threads)
                def[2] = k;
    /* every thread on a performance core with a sibling (no efficiency
       cores): one thread per core first, for replies and blocks. Of nine
       testers' processors (Zen 2 to Zen 5, an i5-1135G7) one thread per
       core wrote a reply faster in every run, 5-54% (a 9800X3D 27.6
       tokens/s against 20.1, issue #24), and read a prompt up to 35%
       faster (a 9800X3D 207.7 against 187.6; with the fast prompt on an
       RTX 5080 149.5 against 70.6, which the first runs chose), where
       both threads of a core won by 17% at most (Ryzen 5 5500, #45; 9900X
       609 against 561, #50). The tuner tries the others all the same */
    int smt = m->all_threads == m->p_threads && m->p_threads > m->p_cores;
    if (smt && !bt)
        for (int k = 0; k < n; k++)
            if (!c[k].gpu && !c[k].npu && c[k].threads == m->p_cores)
                def[0] = def[2] = k;
    const char *ga = getenv("JANAS_GPU_ALWAYS");
    if (ga && atoi(ga) > 0)
        for (int k = 0; k < n; k++)
            if (c[k].gpu && !c[k].npu && c[k].threads == block)
                def[1] = def[2] = def[3] = k;
    /* JANAS_NPU_ALWAYS=1: the blocks' default with the NPU (tests: with
       JANAS_TUNE=0 every block then runs with it) */
    const char *na = getenv("JANAS_NPU_ALWAYS");
    if (na && atoi(na) > 0 && m->npu)
        for (int k = 0; k < n; k++)
            if (c[k].npu && c[k].threads == c[def[2]].threads &&
                c[k].gpu == c[def[2]].gpu) {
                def[2] = k;
                break;
            }
    if (bt && block != c[def[1]].threads && n < JANAS_TUNE_MAX_CANDS) {
        c[n] = (struct janas_tune_cand){.threads = block, .gpu = 0};
        def[1] = def[2] = def[3] = n++;
    }
    /* with the fast prompt, which reads a prompt's blocks in every
       candidate above, the CPU alone for blocks too: a fast CPU reads them
       sooner - a Ryzen 9 9900X read Qwen3.6-35B-A3B's prompt at 413
       tokens/s on its own, with its 512-bit tiles, and at 346 with the fast
       prompt on an RTX 5060 Ti (issue #50) */
    for (int i = 0; fast && i < n_opts && n < JANAS_TUNE_MAX_CANDS; i++) {
        int dup = 0;
        for (int k = 0; k < n; k++)
            dup |= c[k].alone && c[k].threads == opts[i];
        if (!dup)
            c[n++] = (struct janas_tune_cand){.threads = opts[i], .alone = 1};
    }
    janas_tuner_init(&m->tuner, c, n, def, (tn && atoi(tn) == 0) || bt);
    janas_tuner_allow_gpu(&m->tuner, 1);
    /* a GPU that takes single tokens (a discrete one, or one the user
       asked for replies) is tried for replies too, and kept only if it
       measures faster */
    int one = m->gpu && !m->gpu_off && janas_gpu_min_n(m->gpu) <= 1;
    janas_tuner_allow_gpu_one(&m->tuner, one);
    /* the user's devices: prompts' blocks and short turns (no NPU there),
       drafts, replies */
    /* the NPU for blocks by itself only where its answers measured at par
       (janas_m_npu_auto); elsewhere when asked */
    const char *nal = getenv("JANAS_NPU_ALWAYS");
    int blocks = on[0];
    if (m->npu && !blocks && !(nal && atoi(nal) > 0) && !janas_m_npu_auto(m))
        blocks = 1 | (m->gpu && !m->gpu_off ? 1 << JANAS_DEV_GPU : 0);
    janas_tuner_allow(&m->tuner, 2, blocks);
    janas_tuner_allow(&m->tuner, 3, without_npu(on[0]));
    janas_tuner_allow(&m->tuner, 1, on[1]);
    janas_tuner_allow(&m->tuner, 0, on[2]);
    /* the saved choices of this machine and model; "1g" in the key when
       the GPU may take replies, so that choices saved without it do not
       stand in for the tries */
    char cpu[128], name[128] = "";
    janas_cpu_name(cpu, sizeof(cpu));
    janas_jns_meta_str(&m->j, "general.name", name, sizeof(name));
    int w =
        snprintf(m->tune_key, sizeof(m->tune_key), "v2|%s|%d/%d/%d|%s%s%s|%s|",
                 cpu, m->p_cores, m->p_threads, m->all_threads,
                 m->gpu && !m->gpu_off ? janas_gpu_name(m->gpu) : "no GPU",
                 one ? " 1g" : "", fast ? " f" : "", name);
    /* the user's devices, when chosen: their own saved choices */
    if ((on[0] || on[1] || on[2]) && w > 0 && (size_t)w < sizeof(m->tune_key))
        w += snprintf(m->tune_key + w, sizeof(m->tune_key) - (size_t)w,
                      "on %x%x%x|", on[0], on[1], on[2]);
    for (int k = 0; k < n && w > 0 && (size_t)w < sizeof(m->tune_key); k++)
        w += snprintf(m->tune_key + w, sizeof(m->tune_key) - (size_t)w,
                      "%d%s%s%s,", c[k].threads, c[k].gpu ? "g" : "",
                      c[k].npu ? "n" : "", c[k].alone ? "a" : "");
    janas_tuner_load(&m->tuner, m->tune_key);
}

/* Threads and GPU of a candidate, for the passes that follow, of nv
   vectors. The GPU is kept awake only where it takes such a pass: on an
   integrated GPU a single token's products stay on the CPU
   (janas_gpu_min_n 2) unless the whole token is asked of it
   (JANAS_GPU_TOKEN=1), and keeping it awake there only took the power the
   cores need - a Radeon 780M wrote at 10.5 tokens/s in janas-bench's
   "+ GPU" rows against 17.3 on the CPU alone (issue #16). */
void janas_m_apply_candidate(struct janas_llm_model *m, int cand, uint32_t nv)
{
    /* a borrowed pool belongs to the model that made it: a drafting model
       setting its own thread count would set the target's too */
    if (m->own_compute)
        janas_pool_set_active(m->compute, m->tuner.cand[cand].threads);
    m->gpu_use = m->tuner.cand[cand].gpu;
    /* a block the tuner gives the CPU alone */
    m->prompt_cpu = m->tuner.cand[cand].alone && nv >= JANAS_GPU_FAST_N;
    if (m->gpu && nv >= JANAS_GPU_FAST_N && janas_gpu_repeatable() &&
        janas_gpu_fast_prompt())
        m->gpu_use = 0; /* the fast products alone: gpu.h */
    m->npu_use = m->tuner.cand[cand].npu;
    janas_m_npu_awake(m, nv);
    if (m->gpu) {
        janas_gpu_keep_awake(m->gpu,
                             m->gpu_use && (nv >= janas_gpu_min_n(m->gpu) ||
                                            janas_m_gpu_token_asked()));
        janas_gpu_set_only_fast(m->gpu, !m->gpu_use);
        janas_gpu_set_pass(m->gpu, nv);
    }
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

int janas_llm_model_gpu_prompts(const struct janas_llm_model *m)
{
    return m->gpu && !m->gpu_off && m->tuner.allow_gpu &&
           janas_gpu_fast_prompt() && !m->prompts_off_gpu;
}

int janas_llm_model_devices(const struct janas_llm_model *m, int cls)
{
    /* the user's: the tuner's own limits (janas_m_npu_auto) apart */
    return cls == 2   ? m->dev_on[0]
           : cls == 3 ? without_npu(m->dev_on[0])
           : cls == 1 ? m->dev_on[1]
           : cls == 0 ? m->dev_on[2]
                      : 0;
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

int janas_llm_model_npu_used(const struct janas_llm_model *m)
{
    int used = 0;
    for (int k = 0; m->npu && k < JANAS_TUNE_CLASSES; k++)
        if (m->tuner.cand[janas_tuner_best(&m->tuner, k)].npu)
            used |= 1 << k;
    return used;
}

int janas_llm_model_candidates(const struct janas_llm_model *m, int *threads,
                               int *gpu, int *npu, int *alone, int max)
{
    int n = m->tuner.n_cand < max ? m->tuner.n_cand : max;
    for (int i = 0; i < n; i++) {
        threads[i] = m->tuner.cand[i].threads;
        gpu[i] = m->tuner.cand[i].gpu;
        if (npu)
            npu[i] = m->tuner.cand[i].npu;
        if (alone)
            alone[i] = m->tuner.cand[i].alone;
    }
    return n;
}

int janas_llm_model_blocks_alone(const struct janas_llm_model *m)
{
    return m->tuner.n_cand > 0 &&
           m->tuner.cand[janas_tuner_best(&m->tuner, 2)].alone;
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
