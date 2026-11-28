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

/* the engine's reading (src/llm/api.c, find_format_in), in short: Gemma 4
   with <|channel>, <channel|>, <|think|> and a template that turns it on;
   DeepSeek with <think> and </think>; ChatML with them and a template that
   turns it on. The others do not reason. */
static int thinks(const struct janas_jns *j)
{
    size_t cap = (size_t)1 << 17;
    char *t = malloc(cap);
    if (!t)
        return 0;
    if (janas_jns_meta_str(j, "tokenizer.chat_template", t, cap) != 0)
        t[0] = 0;
    int on = strstr(t, "enable_thinking") != NULL, r;
    if (has_token(j, "<|turn>") && strstr(t, "<|turn>"))
        r = on && has_token(j, "<|channel>") && has_token(j, "<channel|>") &&
            has_token(j, "<|think|>");
    else if (has_token(j, "<｜User｜>"))
        r = has_token(j, "<think>") && has_token(j, "</think>");
    else
        r = on && has_token(j, "<think>") && has_token(j, "</think>");
    free(t);
    return r;
}

/* the sampling the file recommends, by the server's names */
static void recommended(const struct janas_jns *j, char *out, size_t len)
{
    static const struct {
        const char *key, *name;
        int whole;
    } K[] = {{"general.sampling.temp", "temperature", 0},
             {"general.sampling.top_k", "top_k", 1},
             {"general.sampling.top_p", "top_p", 0},
             {"general.sampling.min_p", "min_p", 0},
             {"general.sampling.penalty_repeat", "repeat_penalty", 0},
             {"general.sampling.penalty_last_n", "repeat_last_n", 1},
             {"general.sampling.mirostat", "mirostat", 1},
             {"general.sampling.mirostat_tau", "mirostat_tau", 0},
             {"general.sampling.mirostat_eta", "mirostat_eta", 0},
             {"general.sampling.xtc_probability", "xtc_probability", 0},
             {"general.sampling.xtc_threshold", "xtc_threshold", 0}};
    size_t at = 0;
    out[at++] = '{';
    out[at] = 0;
    for (size_t i = 0; i < sizeof K / sizeof *K; i++) {
        char v[32];
        uint32_t u;
        float f;
        if (K[i].whole && janas_jns_meta_u32(j, K[i].key, &u) == 0)
            snprintf(v, sizeof v, "%d", u >= 1u << 30 ? -1 : (int)u);
        else if (!K[i].whole && janas_jns_meta_f32(j, K[i].key, &f) == 0 &&
                 f >= 0 && f < 1e6f)
            snprintf(v, sizeof v, "%g", (double)f);
        else
            continue;
        int w = snprintf(out + at, len - at, "%s\"%s\": %s", at > 1 ? ", " : "",
                         K[i].name, v);
        if (w < 0 || (size_t)w >= len - at - 1)
            break;
        at += (size_t)w;
    }
    out[at++] = '}';
    out[at] = 0;
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
        /* a chat model: one with a template, or DeepSeek-V4, whose files
           (tools/hf2jns_ds4.c) have none and whose format the engine reads
           from its tokens (src/llm/api.c) */
        int chat =
            janas_jns_meta_str(&j, "tokenizer.chat_template", head,
                               sizeof head) == 0 ||
            (has_token(&j, "<｜User｜>") && has_token(&j, "<｜Assistant｜>"));
        m->fim = has_token(&j, "<|fim_prefix|>") &&
                 has_token(&j, "<|fim_suffix|>") &&
                 has_token(&j, "<|fim_middle|>");
        m->thinks = chat && thinks(&j);
        recommended(&j, m->sampling, sizeof m->sampling);
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
