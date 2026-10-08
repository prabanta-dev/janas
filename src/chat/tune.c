/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tune.c - janas-chat's measure of a model on this machine (see tune.h).
 *
 * Each way is a model opened anew - the prediction file and the head's bits
 * are taken when it opens - a reply to each prompt to warm its caches, then
 * three replies of different kinds measured together: output tokens over
 * output seconds, at temperature 0, so every way writes the same text.
 * Drafting off is measured on the first opening, since it only takes a
 * parameter of the chat, one prompt at a time beside drafting on.
 */
#include "tune.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define TUNE_REPLY 128
/* a way other than the one the model starts with has to be this much
   faster: a couple of per cent is the noise of the measure
   (Qwen3-0.6B: 154.6 and 154.7 drafting, 157.2 and 156.3 without) */
#define TUNE_MARGIN 1.03
/* three kinds of reply - prose, code, a list - since how far ahead the
   drafts are right depends on the text: one reply alone swung by a fifth
   between two runs (Gemma 4 E2B with its assistant: 38.6, then 46.9) */
static const char *const PROMPTS[] = {
    "Explain in about one hundred words how a bicycle stays upright.",
    "Write a C function that reverses a singly linked list, with a short "
    "comment.",
    "List ten Italian cities with one line about each."};
#define N_PROMPTS (sizeof PROMPTS / sizeof *PROMPTS)

int chat_tune_path(char *buf, size_t len, const char *key, int fast)
{
    const char *x = getenv("XDG_CACHE_HOME"), *h = getenv("HOME");
    char dir[2048];
    if (x && *x)
        snprintf(dir, sizeof(dir), "%s/janas", x);
    else if (h && *h)
        snprintf(dir, sizeof(dir), "%s/.cache/janas", h);
    else
        return -1;
    int n = snprintf(buf, len, "%s/tune-%s-%s.conf", dir, key,
                     fast ? "fast" : "quality");
    return n > 0 && (size_t)n < len ? 0 : -1;
}

/* One reply to prompt from a fresh conversation: its output tokens and
   seconds added to *tok and *sec. 0, or -1. */
static int reply1(janas_llm_chat *c, const char *prompt, double *tok,
                  double *sec)
{
    if (janas_llm_chat_reset(c) != JANAS_LLM_OK ||
        janas_llm_chat_send(c, prompt, -1) != JANAS_LLM_OK)
        return -1;
    char buf[256];
    int32_t len, r;
    while ((r = janas_llm_chat_next(c, buf, sizeof(buf), &len)) == JANAS_LLM_OK)
        ;
    struct janas_llm_chat_stats s;
    memset(&s, 0, sizeof(s));
    s.size = sizeof(s);
    if (r != JANAS_LLM_DONE || janas_llm_chat_stats(c, &s) != JANAS_LLM_OK ||
        s.output_seconds <= 0 || s.output_tokens < 8)
        return -1;
    *tok += s.output_tokens;
    *sec += s.output_seconds;
    return 0;
}

/* The replies to every prompt: tokens a second over all of them, or -1. */
static double reply(janas_llm_chat *c)
{
    double tok = 0, sec = 0;
    for (size_t i = 0; i < N_PROMPTS; i++)
        if (reply1(c, PROMPTS[i], &tok, &sec) != 0)
            return -1;
    return tok / sec;
}

/*
 * Every prompt once, unmeasured. The first reply to a kind of text is slow
 * on a mixture of experts - its experts are touched for the first time -
 * and warming on the first prompt alone left the code and the list cold for
 * whichever way came first: Qwen3-30B-A3B wrote the code at 27.9 tokens/s
 * the first time and 34.5 the second, drafting or not, and since drafting
 * always came first the tuning chose --no-spec on a difference of 10% that
 * was all order (8 Oct 2026: 31.4 drafting and 34.4 without, 34.5 and 31.9
 * with the order swapped).
 */
static int warm(janas_llm_chat *c)
{
    double tok = 0, sec = 0;
    for (size_t i = 0; i < N_PROMPTS; i++)
        if (reply1(c, PROMPTS[i], &tok, &sec) != 0)
            return -1;
    return 0;
}

/* One reply to prompt with drafting as spec says, added to tok and sec. */
static int reply_way(janas_llm_chat *c, struct janas_llm_chat_params *q,
                     int spec, const char *prompt, double *tok, double *sec)
{
    q->speculate = spec;
    if (janas_llm_chat_set_params(c, q) != JANAS_LLM_OK)
        return -1;
    return reply1(c, prompt, tok, sec);
}

/*
 * Drafting on and off, prompt by prompt, which goes first taking turns:
 * the caches go on warming while a model is measured - most of all on a
 * mixture larger than its expert cache - and two series one after the
 * other gave the second the warmer machine.
 */
static int both(janas_llm_chat *c, struct janas_llm_chat_params *q, double *on,
                double *off)
{
    double tok[2] = {0, 0}, sec[2] = {0, 0}; /* without, with drafts */
    for (size_t i = 0; i < N_PROMPTS; i++)
        for (int j = 0; j < 2; j++) {
            int spec = (int)((i + (size_t)j) % 2 == 0);
            if (reply_way(c, q, spec, PROMPTS[i], &tok[spec], &sec[spec]))
                return -1;
        }
    *on = tok[1] / sec[1];
    *off = tok[0] / sec[0];
    return 0;
}

/* One way: opened, warmed, measured drafting (*on) and, if off is given,
   without drafting. 0, or -1. */
static int measure(const char *path, const struct janas_llm_params *mp,
                   const struct janas_llm_chat_params *cp, int gpu, double *on,
                   double *off)
{
    janas_llm *llm = NULL;
    janas_llm_chat *c = NULL;
    struct janas_llm_params p = *mp;
    p.no_preload = 1; /* the same start for every way */
    struct janas_llm_chat_params q = *cp;
    q.temperature = 0;
    q.max_reply = TUNE_REPLY;
    q.thinking = 0;
    q.speculate = 1;
    if (janas_llm_open(path, &p, &llm) != JANAS_LLM_OK) {
        fprintf(stderr, "tuning: %s\n", janas_llm_last_error());
        return -1;
    }
    if (!gpu)
        janas_llm_set_gpu(llm, 0);
    int rc = -1;
    if (janas_llm_chat_create(llm, &q, &c) == JANAS_LLM_OK && warm(c) == 0) {
        if (off)
            rc = both(c, &q, on, off);
        else
            rc = (*on = reply(c)) > 0 ? 0 : -1;
    }
    janas_llm_chat_destroy(c);
    janas_llm_close(llm);
    return rc;
}

int chat_tune_run(const char *path, const struct janas_llm_params *mp,
                  const struct janas_llm_chat_params *cp, int gpu,
                  struct chat_tune_dims dims, const char *file)
{
    const char *src = mp->draft_path ? "a second model"
                      : mp->mtp_path ? "the prediction file"
                                     : "the model or the conversation";
    double on = 0, off = 0, conv = 0;
    fprintf(stderr, "tuning: drafts from %s...\n", src);
    if (measure(path, mp, cp, gpu, &on, dims.spec ? &off : NULL) != 0)
        return -1;
    fprintf(stderr, "tuning: %.1f tokens/s drafting", on);
    if (dims.spec)
        fprintf(stderr, ", %.1f without drafts", off);
    fputc('\n', stderr);
    /* the drafts copied from the conversation, where a file or a second
       model gives them now */
    int alt = dims.spec && (mp->mtp_path || mp->draft_path);
    if (alt) {
        struct janas_llm_params p = *mp;
        p.mtp_path = NULL;
        p.draft_path = NULL;
        fprintf(stderr, "tuning: drafts from the conversation...\n");
        if (measure(path, &p, cp, gpu, &conv, NULL) != 0)
            conv = 0;
        else
            fprintf(stderr, "tuning: %.1f tokens/s\n", conv);
    }
    /* the winner: 0 as it starts, 1 without drafts, 2 from the
       conversation */
    int best = 0;
    double speed = on;
    if (dims.spec && off > on * TUNE_MARGIN) {
        best = 1;
        speed = off;
    }
    if (conv > on * TUNE_MARGIN && conv > speed) {
        best = 2;
        speed = conv;
    }
    int head6 = 0;
    double h6 = 0;
    if (dims.head) {
        struct janas_llm_params p = *mp;
        p.head_bits = 6;
        if (best == 2)
            p.mtp_path = p.draft_path = NULL;
        struct janas_llm_chat_params q = *cp;
        if (best == 1)
            q.speculate = 0;
        fprintf(stderr, "tuning: the output head as in the file...\n");
        if (measure(path, &p, &q, gpu, &h6, NULL) == 0) {
            fprintf(stderr, "tuning: %.1f tokens/s\n", h6);
            head6 = h6 > speed * TUNE_MARGIN;
        }
    }
    char dir[2048];
    snprintf(dir, sizeof(dir), "%s", file);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = 0;
        mkdir(dir, 0755);
    }
    FILE *f = fopen(file, "w");
    if (!f) {
        fprintf(stderr, "tuning: cannot write %s\n", file);
        return -1;
    }
    char when[32];
    time_t t = time(NULL);
    strftime(when, sizeof(when), "%Y-%m-%d %H:%M", localtime(&t));
    fprintf(
        f,
        "# janas-chat: measured on this machine, %s, %zu replies of up "
        "to %d tokens\n# at temperature 0 each way (tokens/s): drafting from "
        "%s %.1f",
        when, N_PROMPTS, TUNE_REPLY, src, on);
    if (dims.spec)
        fprintf(f, ", without drafts %.1f", off);
    if (alt)
        fprintf(f, ", drafts from the conversation %.1f", conv);
    if (dims.head)
        fprintf(f, ", output head as in the file %.1f", h6);
    fprintf(f, "\n# janas-chat --tune measures again; delete the file to "
               "forget it\n");
    if (best == 1)
        fprintf(f, "--no-spec\n");
    if (best == 2)
        fprintf(f, "--no-mtp\n");
    if (head6)
        fprintf(f, "--head 6\n");
    fclose(f);
    fprintf(stderr, "tuning: kept in %s\n", file);
    return 0;
}
