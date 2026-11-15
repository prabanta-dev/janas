/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * multichat_check.c - several conversations at once through the public API
 * (janas_llm_chat_create_parallel, janas_llm_chats_step): S chats, each
 * with a question of its own and a second turn after its first reply,
 * first one after the other on the model's own chat, then together - the
 * model's own and S - 1 parallel ones, the later ones sent a few steps
 * after the first, so that prompts are read while others write. Each
 * reply must be the same text, greedy and then drawn with a seed of each
 * chat's own, and the reasoning too where the model reasons.
 *
 * Usage: multichat_check <model.jns> [S=4] [max_reply=48]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "janas/llm.h"

static const char *const ASK[] = {
    "Name three rivers of Italy.",
    "Write a haiku about the sea.",
    "What is 17 times 23? Answer briefly.",
    "Spiega in due frasi cos'è un nuraghe.",
    "Give a Python one-liner that reverses a list.",
    "What colour do you get mixing blue and yellow?",
    "List four planets of the solar system.",
    "Describe a cat in one sentence.",
};
static const char *const AGAIN = "Now say it again in fewer words.";

#define MAXC 8
#define TEXT 16384

struct conv {
    janas_llm_chat *c;
    char text[2][TEXT], think[2][TEXT];
    size_t n[2], nt[2];
    int turn; /* 0, 1; 2: done */
    int sent;
};

static uint32_t S, R;

static void params(janas_llm *llm, struct janas_llm_chat_params *p, int sampled,
                   uint32_t i)
{
    (void)llm;
    janas_llm_chat_params_default(p);
    p->max_reply = (int32_t)R;
    if (sampled) {
        p->temperature = 0.8f;
        p->seed = 2000 + i;
    } else {
        p->temperature = 0;
        p->seed = 1;
    }
}

static int send(struct conv *v, uint32_t i)
{
    const char *q = v->turn == 0 ? ASK[i % 8] : AGAIN;
    v->sent = 1;
    return janas_llm_chat_send(v->c, q, (int32_t)strlen(q));
}

/* one piece of v's reply: 1 while it goes on, 0 when its turns are done */
static int take(struct conv *v, uint32_t i)
{
    char buf[256];
    int32_t len = 0;
    int32_t r = janas_llm_chat_next(v->c, buf, sizeof buf, &len);
    if (r != JANAS_LLM_OK && r != JANAS_LLM_DONE)
        return -1;
    int t = v->turn;
    int th = janas_llm_chat_thinking(v->c);
    char *dst = th ? v->think[t] : v->text[t];
    size_t *n = th ? &v->nt[t] : &v->n[t];
    if (len > 0 && *n + (size_t)len < TEXT) {
        memcpy(dst + *n, buf, (size_t)len);
        *n += (size_t)len;
        dst[*n] = 0;
    }
    if (r == JANAS_LLM_DONE) {
        v->turn++;
        v->sent = 0;
        if (v->turn < 2 && send(v, i) != JANAS_LLM_OK)
            return -1;
    }
    return v->turn < 2;
}

static int alone(janas_llm *llm, int sampled, struct conv *v)
{
    for (uint32_t i = 0; i < S; i++) {
        struct janas_llm_chat_params p;
        params(llm, &p, sampled, i);
        memset(&v[i], 0, sizeof(v[i]));
        if (janas_llm_chat_create(llm, &p, &v[i].c) != JANAS_LLM_OK ||
            send(&v[i], i) != JANAS_LLM_OK)
            return -1;
        int r;
        while ((r = take(&v[i], i)) > 0)
            ;
        janas_llm_chat_destroy(v[i].c);
        if (r < 0)
            return -1;
    }
    return 0;
}

static int together(janas_llm *llm, int sampled, struct conv *v)
{
    for (uint32_t i = 0; i < S; i++) {
        struct janas_llm_chat_params p;
        params(llm, &p, sampled, i);
        memset(&v[i], 0, sizeof(v[i]));
        int32_t rc = i == 0 ? janas_llm_chat_create(llm, &p, &v[i].c)
                            : janas_llm_chat_create_parallel(llm, &p, &v[i].c);
        if (rc != JANAS_LLM_OK) {
            printf("together: chat %u: %s\n", i, janas_llm_last_error());
            return -1;
        }
    }
    int rc = 0;
    for (uint32_t step = 0; rc == 0; step++) {
        janas_llm_chat *act[MAXC];
        uint32_t idx[MAXC], na = 0;
        for (uint32_t i = 0; i < S; i++) {
            if (v[i].turn >= 2)
                continue;
            if (!v[i].sent) {
                if (step < 3 * i)
                    continue;
                if (send(&v[i], i) != JANAS_LLM_OK) {
                    rc = -1;
                    break;
                }
            }
            act[na] = v[i].c;
            idx[na++] = i;
        }
        if (rc || !na)
            break;
        if (janas_llm_chats_step(act, (int32_t)na) != JANAS_LLM_OK) {
            printf("together: step %u: %s\n", step, janas_llm_last_error());
            rc = -1;
            break;
        }
        for (uint32_t k = 0; k < na && rc == 0; k++)
            if (take(&v[idx[k]], idx[k]) < 0)
                rc = -1;
    }
    for (uint32_t i = 0; i < S; i++)
        janas_llm_chat_destroy(v[i].c);
    return rc;
}

static int compare(const char *what, const struct conv *a, const struct conv *b)
{
    int bad = 0;
    for (uint32_t i = 0; i < S; i++)
        for (int t = 0; t < 2; t++)
            if (a[i].n[t] != b[i].n[t] || a[i].nt[t] != b[i].nt[t] ||
                memcmp(a[i].text[t], b[i].text[t], a[i].n[t]) ||
                memcmp(a[i].think[t], b[i].think[t], a[i].nt[t])) {
                printf("%s: chat %u, turn %d differs:\n  alone:    "
                       "%.200s\n  together: %.200s\n",
                       what, i, t, a[i].text[t], b[i].text[t]);
                bad++;
            }
    size_t chars = 0;
    for (uint32_t i = 0; i < S; i++)
        for (int t = 0; t < 2; t++)
            chars += a[i].n[t] + a[i].nt[t];
    printf("%s: %s, %zu bytes of replies and reasoning (chat 0: \"%.60s\" "
           "after reasoning \"%.40s\")\n",
           what, bad ? "DIFFERENT" : "the same replies", chars, a[0].text[0],
           a[0].think[0]);
    return bad;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: multichat_check <model.jns> [S] [max_reply]\n");
        return 2;
    }
    S = argc > 2 ? (uint32_t)atoi(argv[2]) : 4;
    R = argc > 3 ? (uint32_t)atoi(argv[3]) : 48;
    if (S < 2 || S > MAXC || R < 1)
        return 2;
    struct janas_llm_params lp;
    janas_llm_params_default(&lp);
    lp.n_ctx = 4096;
    janas_llm *llm;
    if (janas_llm_open(argv[1], &lp, &llm) != JANAS_LLM_OK) {
        fprintf(stderr, "%s: %s\n", argv[1], janas_llm_last_error());
        return 1;
    }
    if (!janas_llm_parallel_ok(llm)) {
        printf("multichat_check: %s takes one conversation at a time\n",
               argv[1]);
        return 1;
    }
    struct conv *a = calloc(S, sizeof(*a)), *b = calloc(S, sizeof(*b));
    int bad = 0;
    for (int sampled = 0; sampled < 2 && a && b; sampled++) {
        const char *what = sampled ? "sampled" : "greedy";
        if (alone(llm, sampled, a) != 0 || together(llm, sampled, b) != 0) {
            printf("%s: failed\n", what);
            return 1;
        }
        bad += compare(what, a, b);
    }
    printf("multichat_check: %u chats, two turns each: %s\n", S,
           bad ? "FAILED" : "ok");
    free(a);
    free(b);
    janas_llm_close(llm);
    return bad != 0;
}
