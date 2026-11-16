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
#include <string.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/model_scan.h"
#include "common/sysinfo.h"

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
static int sees(const char *dir, const char *name)
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

static void models(struct brebus *b, struct janas_buf *o)
{
    static struct janas_model_file m[JANAS_MODELS_MAX];
    int n = janas_models_scan(b->models_dir, m, JANAS_MODELS_MAX);
    janas_buf_puts(o, "{\"dir\": ");
    str(o, b->models_dir);
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
                         "\"recurrent\": %d, \"coder\": %d, \"vision\": %d}",
                         (unsigned long long)m[i].bytes, m[i].ctx, m[i].fim,
                         m[i].recurrent, m[i].coder,
                         sees(b->models_dir, m[i].name));
    }
    janas_buf_puts(o, "]}");
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
        janas_buf_printf(o,
                         ", \"download_gb\": %.2f, \"ram_gb\": %d, \"fits\": "
                         "%d, \"chat\": %d, \"test_only\": %d}",
                         e->jns_repo ? e->jns_gb : e->gguf_gb, e->ram_gb,
                         gb + 1.0 >= e->ram_gb, !e->not_chat, e->test_only);
    }
    janas_buf_puts(o, "]}");
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
