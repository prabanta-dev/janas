/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_scan.c - the models installed here (see model_scan.h), read from
 * the header and metadata of each .jns: nothing of the weights.
 */
#include "common/model_scan.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "llm/jns.h"

static int ends(const char *s, const char *tail)
{
    size_t n = strlen(s), t = strlen(tail);
    return n >= t && strcmp(s + n - t, tail) == 0;
}

/* files beside the models that are not chat models of their own */
static int not_chat(const char *file)
{
    static const char *const words[] = {"-mtp",    ".vision", "embedding",
                                        "whisper", "silero",  "wespeaker",
                                        ".flat",   "deepnsm"};
    for (size_t i = 0; i < sizeof words / sizeof *words; i++)
        if (strstr(file, words[i]))
            return 1;
    return 0;
}

static int has_token(const struct janas_jns *j, const char *tok)
{
    uint64_t n;
    const uint8_t *p;
    if (janas_jns_meta_strings(j, "tokenizer.ggml.tokens", &n, &p) != 0)
        return 0;
    size_t want = strlen(tok);
    for (uint64_t i = 0; i < n; i++) {
        uint64_t len;
        memcpy(&len, p, sizeof len);
        p += sizeof len;
        if (len == want && memcmp(p, tok, want) == 0)
            return 1;
        p += len;
    }
    return 0;
}

static int recurrent_arch(const char *arch)
{
    static const char *const names[] = {"qwen3next", "qwen35", "qwen35moe",
                                        "qwen4exp", "nemotron_h_moe"};
    for (size_t i = 0; i < sizeof names / sizeof *names; i++)
        if (strcmp(arch, names[i]) == 0)
            return 1;
    return 0;
}

static int by_name(const void *a, const void *b)
{
    return strcmp(((const struct janas_model_file *)a)->name,
                  ((const struct janas_model_file *)b)->name);
}

int janas_models_scan(const char *dir, struct janas_model_file *out, int max)
{
    DIR *d = opendir(dir);
    if (!d)
        return 0;
    int n = 0;
    struct dirent *e;
    while (n < max && (e = readdir(d))) {
        if (!ends(e->d_name, ".jns") || not_chat(e->d_name))
            continue;
        struct janas_model_file *m = &out[n];
        memset(m, 0, sizeof *m);
        snprintf(m->path, sizeof m->path, "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(m->path, &st) != 0)
            continue;
        m->bytes = (uint64_t)st.st_size;
        struct janas_jns j;
        char err[256];
        if (janas_jns_open(m->path, &j, err, sizeof err) != 0)
            continue;
        janas_jns_meta_str(&j, "general.architecture", m->arch, sizeof m->arch);
        char key[64];
        snprintf(key, sizeof key, "%s.context_length", m->arch);
        janas_jns_meta_u32(&j, key, &m->ctx);
        char head[8];
        int chat = janas_jns_meta_str(&j, "tokenizer.chat_template", head,
                                      sizeof head) == 0;
        m->fim = has_token(&j, "<|fim_prefix|>") &&
                 has_token(&j, "<|fim_suffix|>") &&
                 has_token(&j, "<|fim_middle|>");
        janas_jns_close(&j);
        if (!chat)
            continue;
        size_t len = strlen(e->d_name) - 4;
        if (len >= sizeof m->name)
            len = sizeof m->name - 1;
        memcpy(m->name, e->d_name, len);
        m->name[len] = 0;
        m->recurrent = recurrent_arch(m->arch);
        m->coder = strstr(m->name, "coder") != NULL;
        n++;
    }
    closedir(d);
    qsort(out, (size_t)n, sizeof *out, by_name);
    return n;
}

int janas_model_completes(const struct janas_model_file *m)
{
    /* every Qwen3 has the tokens in its vocabulary; the coders are the
       ones trained to use them */
    return m->fim && !m->recurrent && m->coder;
}
