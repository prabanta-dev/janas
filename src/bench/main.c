/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas-bench - measures what this machine does with a model and reports
 * it: prefill and decoding speed for every configuration the engine can
 * use (thread counts, GPU), on the same text, and the choices the engine
 * makes from them. The measurements also fill the engine's tuning profile
 * (~/.cache/janas/), so the next start is tuned already.
 *
 * Usage: janas-bench [<model>] [--models <dir>] [--mtp <file> | --no-mtp]
 *                    [--prompt <tokens>] [--gen <tokens>] [--ctx <tokens>]
 *                    [--cache <GiB>] [--reserve <GiB>]
 *                    [--rounds <n>] (default 3, alternating the order)
 *                    [--no-npu] (without the NPU, as JANAS_NPU=0;
 *                    with one, its configurations are measured too)
 *                    [--prompt-on <devices>] [--drafts-on <devices>]
 *                    [--reply-on <devices>] (as in janas-chat: the
 *                    devices the engine may choose for each kind of pass;
 *                    the rows measured are the configurations it may
 *                    choose for some kind)
 *                    [--npu-int8] (the NPU's rows with eight-bit
 *                    activations)
 *                    [--gpu-profile] (the GPU's own timestamps of each
 *                    phase of a pass on the card, in the report: as
 *                    JANAS_GPU_TOKEN_PROFILE=1)
 *                    [--settle <seconds>] (after the warm-up, wait up to
 *                    that long for the cache's background filling to end:
 *                    the configurations measured on a warm cache, as a
 *                    chat runs after its first minutes)
 *
 * With the fast prompt the GPU reads a prompt's blocks in every
 * configuration, "+ GPU" or not, and the report says so under the table;
 * "+ GPU" is its part in the shorter passes.
 *
 * The model is taken as janas-chat takes it (common/model_pick.c): a .jns
 * file, a name of the catalog (qwen3-4b), downloaded if it is not there
 * yet, or nothing, and then the list to choose from. Its prediction file
 * comes with it as in the chat: the one the catalog names, or for a file
 * of no catalog the one found beside it (--no-mtp: without).
 *
 * The prompt is prose that does not repeat itself (only a prompt longer
 * than the text repeats it, and the report says so). The drafts of the
 * prediction block are measured as a chat uses them: a question in the
 * model's chat format and the reply it writes, greedily and sampled as the
 * chat samples, each with the share of drafts that survived. Until 26 Sep
 * 2026 they were measured continuing the prompt, one paragraph repeated,
 * and so is any raw continuation of prose: a model copies the paragraphs it
 * has seen, the drafts guess the copy, and "with MTP" came out near twice
 * the plain rate on Qwen3.5-2B (95% of the drafts kept) where its chat got
 * a tenth more (40% kept).
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>

#include "common/devices.h"
#include "common/env_names.h"
#include "common/model_pick.h"
#include "common/sysinfo.h"
#include "llm/quant.h"
#include "common/words.h"
#include "janas/llm.h"
#include "llm/api_internal.h"
#include "llm/expert_cache.h"
#include "llm/generate.h"
#include "llm/gpu.h"
#include "llm/tuner.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

/* Plain English prose on several subjects, none of it repeated: what comes
   after it has to be written, not copied. */
static const char *TEXT =
    "The history of computing is a history of trade-offs between memory and "
    "arithmetic. Early machines stored a few thousand words, and every "
    "program was written around what fitted. Caches, virtual memory and "
    "solid-state disks each moved the boundary, but never removed it: the "
    "fastest code still reads the fewest bytes.\n\n"
    "On the west coast of Sardinia the wind comes from the north-west for "
    "most of the year, and the villages turn their backs to it. Stone walls "
    "divide the fields into small, irregular plots, some older than any "
    "written record of the families who farm them. In late spring the "
    "hillsides are yellow with broom, and by August they are the colour of "
    "straw.\n\n"
    "A good recipe for bread asks for less than people expect: flour, water, "
    "salt, a little yeast, and time. The dough should rest overnight in a "
    "cool room, where slow fermentation builds flavour that no shortcut can "
    "imitate. Bakers argue about hydration and flour the way musicians "
    "argue about tuning.\n\n"
    "The committee met on a Tuesday and, after two hours of discussion, "
    "postponed the decision on the new library until the budget for the "
    "following year was known. Three members voted against the delay, "
    "noting that construction costs had risen by a fifth since the "
    "project was first proposed.\n\n"
    "Birds that migrate at night navigate by the stars, by the Earth's "
    "magnetic field and, near the end of their journey, by smell. Young "
    "birds on their first flight make more mistakes than adults, and many "
    "of them end the autumn far from where their species usually winters.\n\n"
    "When the train finally left the station, Marta opened the letter her "
    "grandmother had given her that morning. It was short, written in pencil, "
    "and it began with a question she had not expected.";

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static FILE *report;

/* Printed to the terminal and to the report file. */
static void out(const char *fmt, ...) __attribute__((format(gnu_printf, 1, 2)));
static void out(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    if (report) {
        va_start(ap, fmt);
        vfprintf(report, fmt, ap);
        va_end(ap);
    }
}

struct run {
    double prefill, decode, mtp, mtp_s; /* token/s; MTP greedy and sampled */
    double hits;                        /* experts served from the cache, % */
    uint64_t drafted, accepted, drafted_s, accepted_s;
};

/* Prefill then n_gen tokens, greedy, without drafts: one token a pass,
   whatever it says, costs the same. */
static int measure(struct janas_llm_model *m, const int32_t *prompt,
                   uint32_t n_prompt, uint32_t n_gen, double *prefill,
                   double *decode)
{
    struct janas_gen_options o = {
        .spec_mode = JANAS_SPEC_OFF, .spec_k = 1, .spec_auto = 1, .eos = -1};
    struct janas_llm_session *s = janas_llm_session_create(m, &o);
    if (!s || janas_llm_session_append(s, prompt, n_prompt) != 0)
        return -1;
    /* without the NPU's compilations (a matrix's first block, once a
       session): said apart in the report */
    double k0 = janas_llm_model_npu_compile_seconds(m), t0 = now();
    if (janas_llm_session_prefill(s) != 0)
        return -1;
    double t1 = now();
    t0 += janas_llm_model_npu_compile_seconds(m) - k0;
    for (uint32_t i = 0; i < n_gen; i++) {
        int32_t t;
        if (janas_llm_session_next(s, &t) != 0)
            return -1;
    }
    double t2 = now();
    janas_llm_session_destroy(s);
    if (prefill)
        *prefill = (n_prompt - 1) / (t1 - t0);
    *decode = n_gen / (t2 - t1);
    return 0;
}

/* An ordinary request, nothing in it to copy from. */
static const char *QUESTION =
    "Explain in plain words how bread is made at home, from the flour to "
    "the oven, for someone who has never baked.";

/* A reply of n_gen tokens to QUESTION through the chat, drafts from the
   MTP block, greedy or sampled as the chat samples (with a fixed seed):
   its rate, and the drafts made and kept. */
static int measure_chat(janas_llm *llm, uint32_t n_gen, int sampled,
                        double *rate, uint64_t *drafted, uint64_t *accepted)
{
    struct janas_llm_chat_params cp;
    janas_llm_chat_params_default(&cp);
    if (!sampled)
        cp.temperature = 0;
    cp.seed = 20260926;
    cp.speculate = 1;
    cp.thinking = 0;
    cp.max_reply = (int32_t)n_gen;
    janas_llm_chat *c;
    if (janas_llm_chat_create(llm, &cp, &c) != JANAS_LLM_OK)
        return -1;
    char buf[256];
    int32_t len, rc = janas_llm_chat_send(c, QUESTION, -1);
    while (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_next(c, buf, sizeof(buf), &len);
    struct janas_llm_chat_stats st = {.size = sizeof(st)};
    if (rc != JANAS_LLM_DONE || janas_llm_chat_stats(c, &st) != JANAS_LLM_OK ||
        st.output_seconds <= 0) {
        janas_llm_chat_destroy(c);
        return -1;
    }
    *rate = st.output_tokens / st.output_seconds;
    *drafted += st.drafted;
    *accepted += st.accepted;
    janas_llm_chat_destroy(c);
    return 0;
}

int main(int argc, char **argv)
{
    janas_env_check("janas-bench");
    if (argc > 1 &&
        (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        fprintf(stderr,
                "usage: janas-bench [<model>] [--models <dir>] "
                "[--mtp <file> | --no-mtp] [--prompt <tokens>] "
                "[--gen <tokens>] [--ctx <tokens>] [--cache <GiB>] "
                "[--reserve <GiB>] [--rounds <n>] [--settle <seconds>] "
                "[--no-npu] [--prompt-on <devices>] [--drafts-on <devices>] "
                "[--reply-on <devices>] [--npu-int8] [--gpu-profile]\n"
                "<model>: a .jns file, or a name of the models Janas knows "
                "(downloaded if it is not there yet); none: the list\n"
                "<devices>: auto (default), cpu, gpu, npu, npu+gpu (both in "
                "each pass), or a list apart by commas to choose from\n");
        return 2;
    }
    struct janas_llm_params p;
    janas_llm_params_default(&p);
    p.mode = JANAS_LLM_MODE_MAX; /* every configuration is measured */
    p.n_ctx = 4096;
    uint32_t n_prompt = 256, n_gen = 64;
    int rounds = 3;
    long long settle = 0;
    const char *model = NULL, *models = NULL;
    int no_mtp = 0;
    static const char *const options[] = {
        "--mtp",       "--no-mtp",   "--models",   "--prompt",
        "--gen",       "--ctx",      "--rounds",   "--cache",
        "--reserve",   "--settle",   "--no-npu",   "--prompt-on",
        "--drafts-on", "--reply-on", "--npu-int8", "--gpu-profile"};
    for (int i = 1; i < argc; i += 2) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        char why[200] = "";
        long long x = 0;
        double r = 0;
        if (a[0] != '-' && !model) {
            model = a;
            i--;
            continue;
        }
        if (strcmp(a, "--no-mtp") == 0) {
            no_mtp = 1;
            i--;
            continue;
        }
        if (strcmp(a, "--no-npu") == 0) {
            setenv("JANAS_NPU", "0", 1); /* read when the model opens */
            i--;
            continue;
        }
        if (strcmp(a, "--npu-int8") == 0) {
            setenv("JANAS_NPU_A8", "1", 1); /* so is this */
            i--;
            continue;
        }
        if (strcmp(a, "--gpu-profile") == 0) {
            setenv("JANAS_GPU_TOKEN_PROFILE", "1", 1); /* and this */
            i--;
            continue;
        }
        int dk = janas_devices_option(a, v, why, sizeof why);
        if (dk < 0) {
            fprintf(stderr, "janas-bench: %s\n", why);
            return 2;
        }
        if (dk > 0)
            continue;
        if (strcmp(a, "--mtp") == 0 && v)
            p.mtp_path = v;
        else if (strcmp(a, "--models") == 0 && v)
            models = v;
        else if (strcmp(a, "--prompt") == 0 || strcmp(a, "--gen") == 0 ||
                 strcmp(a, "--ctx") == 0 || strcmp(a, "--rounds") == 0) {
            if (janas_word_int(v, 1, UINT32_MAX, &x, why, sizeof why) == 0) {
                if (a[2] == 'p')
                    n_prompt = (uint32_t)x;
                else if (a[2] == 'g')
                    n_gen = (uint32_t)x;
                else if (a[2] == 'c')
                    p.n_ctx = (uint32_t)x;
                else
                    rounds = x < 1000 ? (int)x : 1000;
            }
        } else if (strcmp(a, "--settle") == 0) {
            janas_word_int(v, 0, 86400, &settle, why, sizeof why);
        } else if (strcmp(a, "--cache") == 0 || strcmp(a, "--reserve") == 0) {
            if (janas_word_real(v, 0, 1 << 20, &r, why, sizeof why) == 0)
                *(a[2] == 'c' ? &p.cache_bytes : &p.reserve_bytes) =
                    (uint64_t)(r * (1 << 30));
        } else {
            janas_unknown_word(why, sizeof why, a, "an option", options,
                               sizeof options / sizeof *options);
            fprintf(stderr, "janas-bench: %s\n", why);
            return 2;
        }
        if (why[0] || !v) {
            fprintf(stderr, "janas-bench: %s: %s\n", a,
                    why[0] ? why : "a value is missing");
            return 2;
        }
    }
    if (n_prompt < 2 || n_gen < 1) {
        fprintf(stderr, "janas-bench: --prompt 2 or more, --gen 1 or more\n");
        return 2;
    }
    /* the generated tokens twice (the drafts' rounds after the plain
       ones) and 64 for the chat's turns: issue #27 met it without
       knowing why */
    if (n_prompt + 2 * n_gen + 64 > p.n_ctx) {
        fprintf(stderr,
                "janas-bench: --ctx %u is too small for --prompt %u and "
                "--gen %u: it needs %u tokens (the prompt, the generated "
                "tokens twice, 64 more)\n",
                (unsigned)p.n_ctx, (unsigned)n_prompt, (unsigned)n_gen,
                (unsigned)(n_prompt + 2 * n_gen + 64));
        return 2;
    }
    if (no_mtp && p.mtp_path) {
        fprintf(stderr, "janas-bench: --mtp and --no-mtp together\n");
        return 2;
    }
    if (models)
        setenv("JANAS_MODELS", models, 1);
    struct janas_model_pick mp;
    if (janas_model_pick("janas-bench", model, &mp) != 0)
        return 1;
    static char mtp_found[4400];
    if (!p.mtp_path && !no_mtp &&
        janas_model_pick_mtp(&mp, mtp_found, sizeof(mtp_found))) {
        p.mtp_path = mtp_found;
        const char *b = strrchr(mtp_found, '/');
        fprintf(stderr,
                "multi-token prediction from %s, found beside the model "
                "(--no-mtp: without)\n",
                b ? b + 1 : mtp_found);
    }

    char path[512], stamp[32];
    time_t tt = time(NULL);
    strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", localtime(&tt));
    snprintf(path, sizeof(path), "janas-bench-%s.txt", stamp);
    report = fopen(path, "w");

    /* the machine */
    char cpu[128];
    struct utsname un;
    struct janas_cpu_layout lay;
    janas_cpu_name(cpu, sizeof(cpu));
    uname(&un);
    out("janas-bench %s, %s\n", JANAS_VERSION, stamp);
    out("system:  %s %s %s\n", un.sysname, un.release, un.machine);
    out("CPU:     %s", cpu);
    if (janas_cpu_layout(NULL, 0, &lay) == 0)
        out(" (%d performance cores, %d threads on them, %d usable in all)",
            lay.n_p_cores, lay.n_p_threads, lay.n);
    char feat[512];
    janas_cpu_features(feat, sizeof(feat));
    out("\nCPU has: %s", feat[0] ? feat : "(not known)");
    out("\nkernels: %s%s", janas_q4k_kernel_name(),
        getenv("JANAS_KERNELS") ? " (JANAS_KERNELS set)" : "");
    int mains = janas_on_mains();
    out("\nmemory:  %.1f GiB, %.1f GiB available; power: %s\n",
        (double)janas_mem_total() / (1 << 30),
        (double)janas_mem_available() / (1 << 30),
        mains < 0 ? "unknown" : (mains ? "mains" : "battery"));

    janas_llm *llm;
    if (janas_llm_open(mp.path, &p, &llm) != JANAS_LLM_OK) {
        out("cannot open the model: %s\n", janas_llm_last_error());
        return 1;
    }
    struct janas_llm_model *m = janas_llm_internal_model(llm);
    struct janas_gpu *gpu = janas_llm_model_gpu(m);
    char desc[1024]; /* 256 cut it short (issue #47) */
    int32_t dl;
    janas_llm_describe(llm, desc, sizeof(desc), &dl);
    if (gpu)
        out("GPU:     %s\n", janas_gpu_name(gpu));
    else
        out("GPU:     none usable: %s\n", janas_llm_model_gpu_why(m));
    const char *npu = janas_llm_model_npu_name(m);
    if (npu)
        out("NPU:     %s\n", npu);
    else
        out("NPU:     none usable: %s\n", janas_llm_model_npu_why(m));
    out("model:   %s\n", desc);
    /* the user's devices, as this machine has them (the engine said on
       stderr what it could not take) */
    int on_p = janas_llm_model_devices(m, 2),
        on_d = janas_llm_model_devices(m, 1),
        on_r = janas_llm_model_devices(m, 0);
    if (on_p || on_d || on_r) {
        char dp[64], dd[64], dr[64];
        janas_devices_name(on_p, dp, sizeof dp);
        janas_devices_name(on_d, dd, sizeof dd);
        janas_devices_name(on_r, dr, sizeof dr);
        out("devices: prompts %s, drafts %s, replies %s\n", dp, dd, dr);
    }

    /* the prompt: the text once, repeated only when more is asked for */
    struct janas_tokenizer *tok = janas_llm_internal_tokenizer(llm);
    size_t tl = strlen(TEXT), reps = n_prompt / 100 + 2;
    int32_t once[1024];
    long n_once = janas_tokenizer_encode(tok, TEXT, tl, 0, once, 1024);
    char *text = malloc(tl * reps + 1);
    for (size_t r = 0; r < reps; r++)
        memcpy(text + r * tl, TEXT, tl);
    text[tl * reps] = 0;
    int32_t *prompt = malloc((tl * reps + 1) * sizeof(int32_t));
    long nt =
        janas_tokenizer_encode(tok, text, tl * reps, 0, prompt, tl * reps + 1);
    if (nt < (long)n_prompt) {
        out("prompt too short\n");
        return 1;
    }
    int has_mtp = janas_llm_model_has_mtp(m);
    struct janas_llm_chat_params cp;
    janas_llm_chat_params_default(&cp);
    out("test:    prompt %u tokens of prose%s, %u generated tokens, greedy",
        n_prompt,
        n_once >= (long)n_prompt ? " (not repeated)"
                                 : " (repeated: the text is shorter)",
        n_gen);
    if (has_mtp)
        out(";\n         with MTP drafts: a chat reply of as many tokens, "
            "greedy and sampled\n         as the chat samples (temperature "
            "%.2g, top-k %d, top-p %.2g), no reasoning",
            cp.temperature, cp.top_k, cp.top_p);
    out("\n\n");

    /* warm-up: the expert cache filled by the same work */
    double d;
    out("warming up the expert cache...\n");
    if (measure(m, prompt, n_prompt, n_gen, NULL, &d) != 0) {
        out("generation failed\n");
        return 1;
    }
    if (settle > 0) {
        /* the filling runs while nobody asks: wait until it has stopped
           (not filling, and no expert more for 3 s: it may not have
           resumed yet when first asked) */
        struct timespec t0, t1, nap = {.tv_sec = 1};
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint64_t last = janas_llm_model_warm_done(m);
        int still = 0;
        do {
            nanosleep(&nap, NULL);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            uint64_t now_in = janas_llm_model_warm_done(m);
            still =
                janas_llm_model_settling(m) || now_in != last ? 0 : still + 1;
            last = now_in;
        } while (still < 3 && t1.tv_sec - t0.tv_sec < settle);
        out("waited %lld s for the cache's filling (%llu experts in)%s\n",
            (long long)(t1.tv_sec - t0.tv_sec), (unsigned long long)last,
            still < 3 ? ", not over" : "");
        if (measure(m, prompt, n_prompt, n_gen, NULL, &d) != 0) {
            out("generation failed\n");
            return 1;
        }
    }

    int th[JANAS_TUNE_MAX_CANDS], gp[JANAS_TUNE_MAX_CANDS],
        np[JANAS_TUNE_MAX_CANDS], al[JANAS_TUNE_MAX_CANDS];
    int nc =
        janas_llm_model_candidates(m, th, gp, np, al, JANAS_TUNE_MAX_CANDS);
    /* the rows: the configurations the engine may choose for some kind of
       pass, every one unless the devices are limited - the NPU's for
       blocks only, the GPU's for replies only where it takes single
       tokens, the CPU alone for blocks only (its replies are the row
       without "+ GPU") */
    int rows[JANAS_TUNE_MAX_CANDS], n_rows = 0, npu_row = -1;
    int gpu_one = gpu && janas_gpu_min_n(gpu) <= 1;
    for (int c = 0; c < nc; c++) {
        rows[c] = !on_p || on_p >> (gp[c] | np[c] << 1) & 1;
        if (!np[c] && !al[c])
            rows[c] |= (!on_d || on_d >> gp[c] & 1) ||
                       ((!gp[c] || gpu_one) && (!on_r || on_r >> gp[c] & 1));
        n_rows += rows[c];
        if (rows[c] && np[c] && npu_row < 0)
            npu_row = c;
    }
    /* the NPU's copies of the weights and its graphs made (and with
       --npu-int8 the scales of the activations taken) before the rows, as
       a chat has them after its first prompt */
    if (npu_row >= 0) {
        out("warming up the NPU...\n");
        janas_llm_model_force(m, npu_row);
        if (measure(m, prompt, n_prompt, n_gen, NULL, &d) != 0) {
            out("generation failed\n");
            return 1;
        }
        janas_llm_model_force(m, -1);
    }
    struct run sum[JANAS_TUNE_MAX_CANDS];
    memset(sum, 0, sizeof(sum));
    int has_cache[JANAS_TUNE_MAX_CANDS] = {0};
    uint64_t drafted = 0, accepted = 0, drafted_s = 0, accepted_s = 0;
    /* several rounds, every other one in reverse order: heat and cache
       drift affect every configuration alike */
    out("measuring %d configurations, %d rounds...\n", n_rows, rounds);
    for (int k = 0; k < rounds * nc; k++) {
        int c = (k / nc) % 2 ? nc - 1 - k % nc : k % nc;
        if (!rows[c])
            continue;
        janas_llm_model_force(m, c);
        struct janas_expert_cache_stats s0, s1;
        janas_expert_cache_stats(janas_llm_model_cache(m), &s0);
        struct run r = {0};
        if (measure(m, prompt, n_prompt, n_gen, &r.prefill, &r.decode) != 0 ||
            (has_mtp && !al[c] &&
             (measure_chat(llm, n_gen, 0, &r.mtp, &drafted, &accepted) != 0 ||
              measure_chat(llm, n_gen, 1, &r.mtp_s, &drafted_s, &accepted_s) !=
                  0))) {
            out("generation failed\n");
            return 1;
        }
        janas_expert_cache_stats(janas_llm_model_cache(m), &s1);
        uint64_t req = s1.requests - s0.requests;
        r.hits = req ? 100.0 * (double)(s1.hits - s0.hits) / (double)req : 0;
        has_cache[c] |= req > 0;
        sum[c].prefill += r.prefill / rounds;
        sum[c].decode += r.decode / rounds;
        sum[c].mtp += r.mtp / rounds;
        sum[c].mtp_s += r.mtp_s / rounds;
        sum[c].hits += r.hits / rounds;
    }
    out("\n%-22s %12s %12s %12s %12s %9s\n", "configuration", "prefill",
        "decode", has_mtp ? "MTP greedy" : "", has_mtp ? "MTP sampled" : "",
        "cache");
    for (int c = 0; c < nc; c++) {
        const struct run *r = &sum[c];
        char name[64];
        if (!rows[c])
            continue;
        snprintf(name, sizeof(name), "%d threads%s", th[c],
                 al[c]            ? ", CPU alone"
                 : gp[c] && np[c] ? " + GPU + NPU"
                 : gp[c]          ? " + GPU"
                 : np[c]          ? " + NPU"
                                  : "");
        char mt[32] = "", ms[32] = "", dec[32] = "-";
        if (has_mtp && !al[c]) {
            snprintf(mt, sizeof(mt), "%.1f tok/s", r->mtp);
            snprintf(ms, sizeof(ms), "%.1f tok/s", r->mtp_s);
        }
        /* no expert asked for (a dense model): no cache to speak of */
        char hit[16] = "-";
        if (has_cache[c])
            snprintf(hit, sizeof(hit), "%.1f%%", r->hits);
        /* the CPU alone: a block's row; its replies are the row of the
           same threads without "+ GPU" */
        if (!al[c])
            snprintf(dec, sizeof(dec), "%.1f tok/s", r->decode);
        out("%-22s %8.1f tok/s %12s %12s %12s %9s\n", name, r->prefill, dec, mt,
            ms, hit);
    }
    /* the fast prompt runs on the GPU in every row but the CPU alone's:
       the rows without "+ GPU" are not the CPU alone (issue #16 read them
       so, and found the GPU of no help where it read the prompt 5.2 times
       faster) */
    int cpu_rows = 0, alone_rows = 0;
    for (int c = 0; c < nc; c++) {
        cpu_rows += rows[c] && !gp[c] && !al[c];
        alone_rows += rows[c] && al[c];
    }
    /* a card with memory of its own takes every pass whole, in every row
       (issue #54: a "12 threads" row at 834 tokens/s where the CPU alone
       read 152) */
    if (janas_llm_model_gpu_whole(m))
        out("note:    on this card every pass of the model runs whole on "
            "the GPU, in every row%s: the threads feed it and compute a "
            "mixture's experts the card does not hold; "
            "JANAS_GPU=0 measures the CPU alone, JANAS_GPU_WHOLE=0 the "
            "products shared with the CPU\n",
            alone_rows ? " but a prompt's in \"CPU alone\"" : "");
    else if (cpu_rows && janas_llm_model_gpu_prompts(m))
        out("note:    in every row%s the GPU reads a prompt's blocks of %d "
            "tokens or more (the fast prompt); \"+ GPU\" also gives it a "
            "part in the shorter passes%s\n",
            alone_rows ? " but \"CPU alone\"" : "", JANAS_GPU_FAST_N,
            gpu && janas_gpu_min_n(gpu) > 1
                ? ", from 2 tokens on: a reply's single tokens are the "
                  "CPU's in every row"
                : "");
    if (npu_row >= 0)
        out("note:    \"+ NPU\": the NPU computes the products of blocks of "
            "%s tokens or more whose weights stay in memory (attention, "
            "recurrent layers, shared experts, a dense model's "
            "feed-forwards; a dense model's layers whole where nine in ten "
            "fit%s), the CPU%s the "
            "rest%s; its preparation (copies of the weights, graphs: %.1f s "
            "in all) is in a warm-up of its own, not in the rows\n",
            getenv("JANAS_NPU_MIN_N") ? getenv("JANAS_NPU_MIN_N") : "64",
            getenv("JANAS_NPU_A8") && atoi(getenv("JANAS_NPU_A8")) > 0
                ? ", with eight-bit activations"
                : "",
            janas_llm_model_gpu_prompts(m) ? " and the GPU's fast prompt" : "",
            janas_llm_model_gpu_prompts(m)
                ? " (--prompt-on npu for the CPU and the NPU alone)"
                : "",
            janas_llm_model_npu_compile_seconds(m));
    if (npu_row >= 0 && !janas_llm_model_npu_auto(m) && !on_p)
        out("note:    the engine does not give the NPU this model's blocks by "
            "itself: in products on Gemma 4 the NPU's answers measured below "
            "the CPU's against llama.cpp's (26B-A4B, E2B); --prompt-on npu "
            "asks for it\n");
    janas_llm_model_force(m, -1);
    char tune[512];
    int32_t tn;
    janas_llm_set_mode(llm, JANAS_LLM_MODE_MAX);
    janas_llm_tuning(llm, tune, sizeof(tune), &tn);
    /* the result is the row the engine chose for each kind of pass - what
       the chat and the server will run - not the fastest of each column,
       which could mix configurations no pass uses together */
    int row[3];
    for (int k = 0; k < 3; k++) {
        int t = janas_llm_model_threads(m, k),
            g = (janas_llm_model_gpu_used(m) >> k) & 1,
            u = (janas_llm_model_npu_used(m) >> k) & 1,
            a = k == 2 && janas_llm_model_blocks_alone(m);
        row[k] = 0;
        for (int c = 0; c < nc; c++)
            if (rows[c] && th[c] == t && gp[c] == g && np[c] == u && al[c] == a)
                row[k] = c;
    }
    out("\nresult:  prefill %.1f tok/s, decode %.1f tok/s", sum[row[2]].prefill,
        sum[row[0]].decode);
    if (has_mtp)
        out(", with MTP %.1f tok/s greedy, %.1f tok/s sampled", sum[row[1]].mtp,
            sum[row[1]].mtp_s);
    out("\n         (the configuration the engine chose for each kind of "
        "pass, as it runs in a chat)");
    if (has_mtp && drafted && drafted_s)
        out("\nMTP:     drafts kept %.0f%% greedy, %.0f%% sampled (all "
            "configurations)",
            100.0 * (double)accepted / (double)drafted,
            100.0 * (double)accepted_s / (double)drafted_s);
    out("\nchoices: %s\n", tune);
    /* what the card did: a tester's report has to tell, without the card
       here, where a discrete GPU's time went */
    char gr[8192];
    janas_llm_model_gpu_report(m, gr, sizeof(gr));
    if (gr[0])
        out("%s", gr);
    janas_llm_close(llm); /* saves the tuning profile */
    if (report) {
        fclose(report);
        printf("\nreport written to %s\n", path);
    }
    free(text);
    free(prompt);
    return 0;
}
