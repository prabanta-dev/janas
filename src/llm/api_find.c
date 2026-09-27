/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_find.c - the multi-token prediction file of a model, found beside it
 * (janas_llm_find_mtp in llm.h).
 *
 * A file fits by what its header says, not by its name: Gemma 4's
 * assistant (gemma4-assistant) whose output is as wide as the model and
 * whose vocabulary is the model's, or an MTP block (tools/hf2jns_mtp) of
 * the model's architecture, one layer, as wide, with as many experts. The
 * name only settles a tie: the file sharing the longest start with the
 * model's name wins, so that gemma-4-e4b.jns takes gemma-4-e4b-mtp.jns
 * over another E4B assistant beside it.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "llm/chat.h"
#include "jns.h"

/* no MTP block or assistant is near this size; the models beside one are
   larger, and their headers are not worth reading */
#define MTP_MAX_BYTES ((uint64_t)2 << 30)

struct model_shape {
    char arch[32];
    uint32_t d_model, n_expert;
    uint64_t n_vocab;
};

static uint64_t vocab_of(const struct janas_jns *j)
{
    const struct janas_jns_tensor *t = janas_jns_tensor(j, "token_embd.weight");
    return t ? t->dims[1] : 0;
}

static int fits(const struct janas_jns *j, const struct model_shape *m)
{
    const struct janas_jns_header *h = &j->h;
    if (strncmp(h->arch, "gemma4-assistant", sizeof(h->arch)) == 0) {
        uint32_t out = 0;
        return strcmp(m->arch, "gemma4") == 0 &&
               janas_jns_meta_u32(j, "gemma4-assistant.embedding_length_out",
                                  &out) == 0 &&
               out == m->d_model && vocab_of(j) == m->n_vocab;
    }
    return strncmp(h->arch, m->arch, sizeof(h->arch)) == 0 && h->n_layer == 1 &&
           h->d_model == m->d_model && h->n_expert == m->n_expert &&
           janas_jns_tensor(j, "mtp.eh_proj.weight") != NULL;
}

static size_t common_start(const char *a, const char *b)
{
    size_t n = 0;
    while (a[n] && a[n] == b[n])
        n++;
    return n;
}

int32_t janas_llm_find_mtp(const char *model_path, char *out, int32_t cap)
{
    if (!model_path || !out || cap < 2)
        return janas_api_fail(JANAS_LLM_EINVAL, "no path or no room");
    struct janas_jns j;
    char err[256];
    if (janas_jns_open(model_path, &j, err, sizeof(err)) != 0)
        return janas_api_fail(JANAS_LLM_EOPEN, "%s: %s", model_path, err);
    struct model_shape m = {.d_model = j.h.d_model,
                            .n_expert = j.h.n_expert,
                            .n_vocab = vocab_of(&j)};
    snprintf(m.arch, sizeof(m.arch), "%.31s", j.h.arch);
    janas_jns_close(&j);

    /* the model's directory and its own name */
    char dir[4096];
    snprintf(dir, sizeof(dir), "%s", model_path);
    char *slash = strrchr(dir, '/');
    const char *base = slash ? model_path + (slash - dir) + 1 : model_path;
    if (slash)
        *(slash == dir ? slash + 1 : slash) = 0;
    else
        snprintf(dir, sizeof(dir), ".");
    DIR *d = opendir(dir);
    if (!d)
        return janas_api_fail(JANAS_LLM_EOPEN, "cannot read %s", dir);
    char best[8192] = "";
    size_t best_score = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 5 || strcmp(e->d_name + n - 4, ".jns") != 0 ||
            strcmp(e->d_name, base) == 0)
            continue;
        char path[8192];
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode) ||
            (uint64_t)st.st_size > MTP_MAX_BYTES)
            continue;
        if (janas_jns_open(path, &j, err, sizeof(err)) != 0)
            continue;
        int ok = fits(&j, &m);
        janas_jns_close(&j);
        size_t score = common_start(e->d_name, base) + 1;
        if (ok && (score > best_score ||
                   (score == best_score && strcmp(path, best) < 0))) {
            best_score = score;
            snprintf(best, sizeof(best), "%s", path);
        }
    }
    closedir(d);
    if (!best[0])
        return janas_api_fail(JANAS_LLM_EOPEN, "no MTP file beside %s",
                              model_path);
    if (strlen(best) + 1 > (size_t)cap)
        return janas_api_fail(JANAS_LLM_ESMALL, "path longer than %d", cap);
    memcpy(out, best, strlen(best) + 1);
    return JANAS_LLM_OK;
}
