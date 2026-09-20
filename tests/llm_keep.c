/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * llm_keep.c - checks the conversations kept computed (janas_llm_chat_keep)
 * on a real model. Two conversations take turns, as two clients of a server
 * do: A, then B, then A with one more message, then B with one more. With
 * nothing kept, each return is read from the start; kept, it is copied back
 * and only the new message is read. The replies (greedy) must be the same
 * text either way, and kept must spare the tokens: the second turn of each
 * conversation finds its first turn computed. Then B's second turn is asked
 * again, as a client does for another reply: its prompt is found computed,
 * on a model with a recurrent state too (which keeps it for this). A third run
 * keeps them in too little memory to hold one, and must behave as the first.
 * Last, the disk (janas_llm_chat_keep_disk): a conversation with a long
 * system message is written when its chat ends, and a new chat finds it
 * computed, with the same reply. The disk is a directory of the test's own
 * (XDG_CACHE_HOME), removed at the end.
 *
 * Usage: llm_keep <model.jns> [mtp=<file>] [draft=<file>]
 */
#define _GNU_SOURCE
#include <ftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "janas/llm.h"

#define REPLY 96
#define CAP 8192

struct turn {
    char text[CAP];
    uint32_t prompt, cached;
};

static const char *A1 = "Tell the story of the nuraghi of Sardinia: who "
                        "built them, when, and what they were for.";
static const char *A2 = "Now say which one is the best to visit, and why.";
static const char *B1 = "Write a short poem about a lighthouse in winter.";
static const char *B2 = "Make it rhyme, and give it a title.";

static int reply(janas_llm_chat *c, int n, const int32_t *roles,
                 const char *const *texts, struct turn *t)
{
    if (janas_llm_chat_load(c, n, roles, texts, NULL) != JANAS_LLM_OK) {
        fprintf(stderr, "load: %s\n", janas_llm_last_error());
        return -1;
    }
    size_t len = 0;
    for (;;) {
        char buf[256];
        int32_t k = 0;
        int32_t rc = janas_llm_chat_next(c, buf, sizeof(buf), &k);
        if (rc == JANAS_LLM_DONE)
            break;
        if (rc != JANAS_LLM_OK) {
            fprintf(stderr, "next: %s\n", janas_llm_last_error());
            return -1;
        }
        if (len + (size_t)k < CAP) {
            memcpy(t->text + len, buf, (size_t)k);
            len += (size_t)k;
        }
    }
    t->text[len] = 0;
    struct janas_llm_chat_stats st = {.size = sizeof(st)};
    janas_llm_chat_stats(c, &st);
    t->prompt = st.prompt_tokens;
    t->cached = st.cached_tokens;
    return 0;
}

/* A, B, A + one more, B + one more, B's second turn again; t[0..4]. */
static int run(janas_llm *llm, int32_t keep, uint64_t bytes, struct turn *t)
{
    struct janas_llm_chat_params p;
    janas_llm_chat_params_default(&p);
    p.temperature = 0;
    p.max_reply = REPLY;
    p.thinking = 0;
    janas_llm_chat *c;
    if (janas_llm_chat_create(llm, &p, &c) != JANAS_LLM_OK ||
        janas_llm_chat_keep(c, keep, bytes) != JANAS_LLM_OK) {
        fprintf(stderr, "chat: %s\n", janas_llm_last_error());
        return -1;
    }
    const int32_t r1[] = {JANAS_LLM_ROLE_USER},
                  r3[] = {JANAS_LLM_ROLE_USER, JANAS_LLM_ROLE_ASSISTANT,
                          JANAS_LLM_ROLE_USER};
    int err = reply(c, 1, r1, &A1, &t[0]) || reply(c, 1, r1, &B1, &t[1]);
    if (!err) {
        const char *a[] = {A1, t[0].text, A2}, *b[] = {B1, t[1].text, B2};
        err = reply(c, 3, r3, a, &t[2]) || reply(c, 3, r3, b, &t[3]) ||
              reply(c, 3, r3, b, &t[4]);
    }
    int32_t n = 0;
    uint64_t kb = 0;
    janas_llm_chat_kept(c, &n, &kb);
    printf("keep %d (%llu bytes): prompt/cached", keep,
           (unsigned long long)bytes);
    for (int i = 0; i < 5; i++)
        printf("  %u/%u", t[i].prompt, t[i].cached);
    printf("; %d kept, %.1f MiB\n", n, (double)kb / (1 << 20));
    janas_llm_chat_destroy(c);
    return err ? -1 : 0;
}

static int rm_one(const char *path, const struct stat *st, int flag,
                  struct FTW *ftw)
{
    (void)st;
    (void)flag;
    (void)ftw;
    return remove(path);
}

/* A system message of more than 1024 tokens, then a question; in a chat
   that ends, then in a new one. 0 when the second finds it computed and
   replies the same. */
static int run_disk(janas_llm *llm)
{
    static char sys[32768];
    size_t n = 0;
    for (int i = 1; i <= 90; i++)
        n += (size_t)snprintf(sys + n, sizeof(sys) - n,
                              "Rule %d: answer briefly, in English, and "
                              "never mention the number %d.\n",
                              i, 1000 + 7 * i);
    const int32_t roles[] = {JANAS_LLM_ROLE_SYSTEM, JANAS_LLM_ROLE_USER};
    const char *texts[] = {sys, A1};
    struct janas_llm_chat_params p;
    janas_llm_chat_params_default(&p);
    p.temperature = 0;
    p.max_reply = REPLY;
    p.thinking = 0;
    static struct turn t[2];
    int32_t files[2] = {0, 0};
    uint64_t bytes = 0;
    for (int k = 0; k < 2; k++) {
        janas_llm_chat *c;
        if (janas_llm_chat_create(llm, &p, &c) != JANAS_LLM_OK ||
            janas_llm_chat_keep(c, 4, 0) != JANAS_LLM_OK ||
            janas_llm_chat_keep_disk(c, (uint64_t)1 << 30) != JANAS_LLM_OK) {
            fprintf(stderr, "chat: %s\n", janas_llm_last_error());
            return -1;
        }
        janas_llm_chat_kept_disk(c, &files[k], &bytes);
        int err = reply(c, 2, roles, texts, &t[k]);
        janas_llm_chat_destroy(c); /* and to the disk */
        if (err)
            return -1;
    }
    printf("disk: prompt/cached %u/%u, then %u/%u; %d files (%.1f MiB) "
           "found by the second chat\n",
           t[0].prompt, t[0].cached, t[1].prompt, t[1].cached, files[1],
           (double)bytes / (1 << 20));
    if (files[1] < 1 || t[1].cached + 1 < t[1].prompt ||
        strcmp(t[0].text, t[1].text)) {
        printf("FAIL: disk: %s\n--- first\n%s\n--- second\n%s\n",
               files[1] < 1 ? "nothing written"
                            : "not found computed, or another reply",
               t[0].text, t[1].text);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: llm_keep <model.jns> [mtp=<file>] "
                        "[draft=<file>]\n");
        return 2;
    }
    struct janas_llm_params lp;
    janas_llm_params_default(&lp);
    lp.n_ctx = 4096;
    for (int i = 2; i < argc; i++) {
        if (!strncmp(argv[i], "mtp=", 4))
            lp.mtp_path = argv[i] + 4;
        else if (!strncmp(argv[i], "draft=", 6))
            lp.draft_path = argv[i] + 6;
    }
    janas_llm *llm;
    if (janas_llm_open(argv[1], &lp, &llm) != JANAS_LLM_OK) {
        fprintf(stderr, "open: %s\n", janas_llm_last_error());
        return 1;
    }
    static struct turn none[5], kept[5], tight[5];
    int fail =
        run(llm, 0, 0, none) || run(llm, 4, 0, kept) || run(llm, 4, 1, tight);
    for (int i = 0; !fail && i < 5; i++) {
        if (strcmp(none[i].text, kept[i].text) ||
            strcmp(none[i].text, tight[i].text)) {
            printf("FAIL: reply %d differs\n--- none\n%s\n--- kept\n%s\n"
                   "--- tight\n%s\n",
                   i, none[i].text, kept[i].text, tight[i].text);
            fail = 1;
        }
        if (tight[i].cached != none[i].cached) {
            printf("FAIL: turn %d, too little memory still kept (%u/%u)\n", i,
                   tight[i].cached, none[i].cached);
            fail = 1;
        }
    }
    /* the second turns find their first turn computed: its prompt at least,
       less the last token, which is always read again */
    for (int i = 2; !fail && i < 4; i++)
        if (kept[i].cached < kept[i - 2].prompt - 1 ||
            kept[i].cached <= none[i].cached) {
            printf("FAIL: turn %d, kept computed %u of %u (first turn %u, "
                   "nothing kept %u)\n",
                   i, kept[i].cached, kept[i].prompt, kept[i - 2].prompt,
                   none[i].cached);
            fail = 1;
        }
    /* the same prompt again: all of it computed but its last token */
    if (!fail && (strcmp(kept[4].text, kept[3].text) ||
                  kept[4].cached + 1 < kept[4].prompt)) {
        printf("FAIL: B's second turn asked again: computed %u of %u, "
               "reply %s\n",
               kept[4].cached, kept[4].prompt,
               strcmp(kept[4].text, kept[3].text) ? "differs" : "the same");
        fail = 1;
    }
    /* the disk, in a cache directory of the test's own */
    char dir[4096];
    const char *tmp = getenv("TMPDIR");
    snprintf(dir, sizeof(dir), "%s/janas-keep-test-%d",
             tmp && *tmp ? tmp : "/tmp", (int)getpid());
    if (!fail) {
        setenv("XDG_CACHE_HOME", dir, 1);
        fail = run_disk(llm) != 0;
        nftw(dir, rm_one, 16, FTW_DEPTH | FTW_PHYS);
    }
    janas_llm_close(llm);
    printf("%s\n", fail ? "FAIL" : "OK: same replies, first turns kept");
    return fail;
}
