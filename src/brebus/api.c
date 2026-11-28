/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api.c - what the page asks: the machine (its processor, memory and
 * power), the models installed (janas_models_scan in the models' folder)
 * and the catalog of janas-get with whether each model is meant for a
 * machine of this memory.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/model_scan.h"
#include "common/sysinfo.h"
#include "llm/jns.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

static void machine(struct janas_buf *o)
{
    char cpu[256], feat[512];
    janas_cpu_name(cpu, sizeof cpu);
    janas_cpu_features(feat, sizeof feat);
    janas_buf_puts(o, "{\"version\": ");
    str(o, JANAS_VERSION);
    janas_buf_puts(o, ", \"cpu\": ");
    str(o, cpu);
    janas_buf_puts(o, ", \"features\": ");
    str(o, feat);
    janas_buf_printf(o,
                     ", \"memory_total\": %llu, \"memory_available\": %llu, "
                     "\"mains\": %d, \"autostart\": %s}",
                     (unsigned long long)janas_mem_total(),
                     (unsigned long long)janas_mem_available(),
                     janas_on_mains(),
                     brebus_autostart_on() ? "true" : "false");
}

/* whether a model sees pictures: a .vision.jns beside it whose name is a
   prefix of its (qwen3.5-0.8b.vision.jns serves qwen3.5-0.8b-q4km), as the
   engine finds it (src/llm/api_image.c) */
int brebus_model_sees(const char *dir, const char *name)
{
    DIR *d = opendir(dir);
    struct dirent *e;
    int yes = 0;
    while (d && !yes && (e = readdir(d))) {
        size_t n = strlen(e->d_name), sfx = strlen(".vision.jns");
        if (n > sfx && !strcmp(e->d_name + n - sfx, ".vision.jns") &&
            !strncmp(name, e->d_name, n - sfx))
            yes = 1;
    }
    if (d)
        closedir(d);
    return yes;
}

/* the speech models (Whisper) in the models' folder: not chat models, so
   not in the list, but the server's --speech-model */
static void speech(const char *dir, struct janas_buf *o)
{
    DIR *d = opendir(dir);
    struct dirent *e;
    int k = 0;
    janas_buf_puts(o, "[");
    while (d && (e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n > 11 && !strncmp(e->d_name, "whisper-", 8) &&
            !strcmp(e->d_name + n - 4, ".jns")) {
            janas_buf_puts(o, k++ ? ", " : "");
            janas_json_write_str(o, e->d_name, n - 4);
        }
    }
    if (d)
        closedir(d);
    janas_buf_puts(o, "]");
}

static void models(struct brebus *b, struct janas_buf *o)
{
    static struct janas_model_file m[JANAS_MODELS_MAX];
    int n = janas_models_scan(b->models_dir, m, JANAS_MODELS_MAX);
    janas_buf_puts(o, "{\"dir\": ");
    str(o, b->models_dir);
    janas_buf_puts(o, ", \"speech\": ");
    speech(b->models_dir, o);
    janas_buf_puts(o, ", \"models\": [");
    for (int i = 0; i < n; i++) {
        janas_buf_puts(o, i ? ", {\"name\": " : "{\"name\": ");
        str(o, m[i].name);
        janas_buf_puts(o, ", \"path\": ");
        str(o, m[i].path);
        janas_buf_puts(o, ", \"arch\": ");
        str(o, m[i].arch);
        janas_buf_printf(o,
                         ", \"bytes\": %llu, \"context\": %u, \"fim\": %d, "
                         "\"recurrent\": %d, \"coder\": %d, \"thinks\": %d, "
                         "\"vision\": %d, \"recommended\": %s}",
                         (unsigned long long)m[i].bytes, m[i].ctx, m[i].fim,
                         m[i].recurrent, m[i].coder, m[i].thinks,
                         brebus_model_sees(b->models_dir, m[i].name),
                         m[i].sampling[0] ? m[i].sampling : "{}");
    }
    janas_buf_puts(o, "]}");
}

/* a companion's own suffix: an MTP head's, a vision encoder's */
static size_t companion(const char *f)
{
    static const char *const sfx[] = {"-mtp.jns", ".vision.jns"};
    size_t n = strlen(f);
    for (size_t i = 0; i < sizeof sfx / sizeof sfx[0]; i++) {
        size_t k = strlen(sfx[i]);
        if (n > k && !strcmp(f + n - k, sfx[i]))
            return k;
    }
    return 0;
}

int brebus_model_remove(struct brebus *b, const char *name, char *why,
                        size_t wn)
{
    static struct janas_model_file m[JANAS_MODELS_MAX];
    int n = janas_models_scan(b->models_dir, m, JANAS_MODELS_MAX), at = -1;
    for (int i = 0; name && i < n; i++)
        if (!strcmp(m[i].name, name))
            at = i;
    /* a speech model (Whisper): its file alone */
    if (at < 0 && name && !strncmp(name, "whisper-", 8) && !strchr(name, '/') &&
        !strchr(name, '\\') && !strstr(name, "..")) {
        char p[4400];
        snprintf(p, sizeof p, "%s/%s.jns", b->models_dir, name);
        if (remove(p) == 0)
            return 0;
    }
    if (at < 0) {
        snprintf(why, wn, "no model named so in %s", b->models_dir);
        return -1;
    }
    /* loaded but not the one used now: unloaded, then removed; the one
       the chat uses stays */
    if (brebus_server_unload_idle(b, name) != 0) {
        snprintf(why, wn,
                 "%s is the model in use: choose another one under the "
                 "chat's field first",
                 name);
        return -1;
    }
    if (remove(m[at].path) != 0) {
        snprintf(why, wn, "%s could not be removed", m[at].path);
        return -1;
    }
    /* its companions too, when no other model left needs them: an MTP
       head or a vision encoder serves every model its name begins */
    DIR *d = opendir(b->models_dir);
    struct dirent *e;
    while (d && (e = readdir(d))) {
        size_t k = companion(e->d_name), stem = strlen(e->d_name) - k;
        if (!k || strncmp(name, e->d_name, stem))
            continue;
        int needed = 0;
        for (int i = 0; i < n && !needed; i++)
            needed = i != at && !strncmp(m[i].name, e->d_name, stem);
        if (!needed) {
            char p[4400];
            snprintf(p, sizeof p, "%s/%s", b->models_dir, e->d_name);
            remove(p);
        }
    }
    if (d)
        closedir(d);
    return 0;
}

static void catalog(struct janas_buf *o)
{
    /* the catalog's memory is the machine's as sold, in GB: a 32 GB
       laptop shows 30.8 GiB to the system, 33.1 GB */
    double gb = (double)janas_mem_total() / 1e9;
    janas_buf_puts(o, "{\"models\": [");
    for (int i = 0; i < janas_catalog_n; i++) {
        const struct janas_model_entry *e = &janas_catalog[i];
        janas_buf_puts(o, i ? ", {\"name\": " : "{\"name\": ");
        str(o, e->name);
        janas_buf_puts(o, ", \"what\": ");
        str(o, e->what);
        janas_buf_puts(o, ", \"file\": ");
        str(o, e->out);
        janas_buf_puts(o, ", \"repo\": ");
        str(o, e->repo ? e->repo : "");
        janas_buf_printf(o,
                         ", \"download_gb\": %.2f, \"ram_gb\": %d, \"fits\": "
                         "%d, \"chat\": %d, \"test_only\": %d, \"speech\": %d}",
                         e->jns_repo ? e->jns_gb : e->gguf_gb, e->ram_gb,
                         gb + 1.0 >= e->ram_gb, !e->not_chat, e->test_only,
                         e->speech);
    }
    janas_buf_puts(o, "]}");
}

/*
 * A model's chat template, as its file has it: {"template": "..."}, or
 * {"error": "..."}; for My models to show and to start a template of the
 * user's own from.
 */
int brebus_model_template(struct brebus *b, const char *name,
                          struct janas_buf *out)
{
    static struct janas_model_file m[JANAS_MODELS_MAX];
    int n = janas_models_scan(b->models_dir, m, JANAS_MODELS_MAX), at = -1;
    for (int i = 0; name && i < n; i++)
        if (!strcmp(m[i].name, name))
            at = i;
    if (at < 0) {
        janas_buf_puts(out, "{\"error\": \"no model named so\"}");
        return 404;
    }
    struct janas_jns j;
    char err[256];
    size_t cap = (size_t)1 << 20;
    char *t = malloc(cap);
    int got = -1;
    if (t && janas_jns_open(m[at].path, &j, err, sizeof err) == 0) {
        got = janas_jns_meta_str(&j, "tokenizer.chat_template", t, cap);
        janas_jns_close(&j);
    }
    if (got != 0) {
        free(t);
        janas_buf_puts(out, "{\"error\": \"the model's file has no chat "
                            "template\"}");
        return 404;
    }
    janas_buf_puts(out, "{\"template\": ");
    janas_json_write_str(out, t, strlen(t));
    janas_buf_puts(out, "}");
    free(t);
    return 200;
}

/* a key of the model's file as text into the card (when it has it) */
static void card_str(struct janas_buf *o, const struct janas_jns *j,
                     const char *key, const char *as)
{
    char v[2048];
    if (janas_jns_meta_str(j, key, v, sizeof v) != 0 || !v[0])
        return;
    janas_buf_printf(o, ", \"%s\": ", as);
    janas_json_write_str(o, v, strlen(v));
}

/*
 * A model's card, from its file: what its makers and its converter wrote
 * of it (GGUF's general.*): its name, size, licence and the licence's
 * address, the model it was made from, who quantized it, its repository,
 * its tags, its description. {"name": ...}, or {"error": ...}.
 */
int brebus_model_card(struct brebus *b, const char *name, struct janas_buf *out)
{
    static struct janas_model_file m[JANAS_MODELS_MAX];
    int n = janas_models_scan(b->models_dir, m, JANAS_MODELS_MAX), at = -1;
    for (int i = 0; name && i < n; i++)
        if (!strcmp(m[i].name, name))
            at = i;
    struct janas_jns j;
    char err[256];
    if (at < 0 || janas_jns_open(m[at].path, &j, err, sizeof err) != 0) {
        janas_buf_puts(out, "{\"error\": \"no model named so\"}");
        return 404;
    }
    janas_buf_puts(out, "{\"file\": ");
    const char *f = strrchr(m[at].path, '/');
    f = f ? f + 1 : m[at].path;
    janas_json_write_str(out, f, strlen(f));
    card_str(out, &j, "general.name", "name");
    card_str(out, &j, "general.organization", "organization");
    card_str(out, &j, "general.size_label", "size");
    card_str(out, &j, "general.finetune", "finetune");
    card_str(out, &j, "general.license", "license");
    card_str(out, &j, "general.license.name", "license_name");
    card_str(out, &j, "general.license.link", "license_link");
    card_str(out, &j, "general.quantized_by", "quantized_by");
    card_str(out, &j, "general.repo_url", "repo_url");
    card_str(out, &j, "general.url", "url");
    card_str(out, &j, "general.description", "description");
    uint32_t nb = 0;
    if (janas_jns_meta_u32(&j, "general.base_model.count", &nb) == 0 && nb) {
        janas_buf_puts(out, ", \"base\": [");
        for (uint32_t i = 0; i < nb && i < 8; i++) {
            char k[96];
            janas_buf_puts(out, i ? ", {\"i\": 0" : "{\"i\": 0");
            snprintf(k, sizeof k, "general.base_model.%u.name", i);
            card_str(out, &j, k, "name");
            snprintf(k, sizeof k, "general.base_model.%u.organization", i);
            card_str(out, &j, k, "organization");
            snprintf(k, sizeof k, "general.base_model.%u.repo_url", i);
            card_str(out, &j, k, "repo_url");
            janas_buf_puts(out, "}");
        }
        janas_buf_puts(out, "]");
    }
    uint64_t nt = 0;
    const uint8_t *tp = NULL;
    if (janas_jns_meta_strings(&j, "general.tags", &nt, &tp) == 0 && nt) {
        /* each a 64-bit length, then its bytes */
        janas_buf_puts(out, ", \"tags\": [");
        for (uint64_t i = 0; i < nt && i < 32; i++) {
            uint64_t len;
            memcpy(&len, tp, sizeof len);
            tp += sizeof len;
            janas_buf_puts(out, i ? ", " : "");
            janas_json_write_str(out, (const char *)tp,
                                 len < 256 ? (size_t)len : 256);
            tp += len;
        }
        janas_buf_puts(out, "]");
    }
    janas_buf_puts(out, "}");
    janas_jns_close(&j);
    return 200;
}

int brebus_api(struct brebus *b, const char *what, struct janas_buf *out)
{
    if (!strcmp(what, "machine"))
        machine(out);
    else if (!strcmp(what, "models"))
        models(b, out);
    else if (!strcmp(what, "catalog"))
        catalog(out);
    else
        return -1;
    return 0;
}
