/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf.c - Hugging Face searched from Discover: the models whose names hold
 * the words asked, with GGUF files, the most downloaded first (the public
 * API, /api/models), and the GGUF and JNS files of one, with their sizes.
 * A file is then downloaded as any hf:<owner>/<repo>/<file> (get.c). The
 * page cannot reach another site; Brebus asks for it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brebus.h"
#include "common/hf_fetch.h"

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

static int fail(struct janas_buf *out, const char *why)
{
    out->n = 0;
    janas_buf_puts(out, "{\"error\": ");
    str(out, why);
    janas_buf_puts(out, "}");
    return -1;
}

/* the words of a query %-encoded into an address */
static void urlenc(const char *s, char *out, size_t cap)
{
    static const char hex[] = "0123456789ABCDEF";
    size_t k = 0;
    for (; *s && k + 4 < cap; s++) {
        unsigned char c = (unsigned char)*s;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.') {
            out[k++] = (char)c;
        } else {
            out[k++] = '%';
            out[k++] = hex[c >> 4];
            out[k++] = hex[c & 15];
        }
    }
    out[k] = 0;
}

static int get_json(const char *url, struct janas_buf *b, char *err, size_t en)
{
    return janas_hf_get_url(url, NULL, NULL, b, NULL, 0, err, en);
}

int brebus_hf_search(struct brebus *b, const char *q, struct janas_buf *out)
{
    (void)b;
    if (!q || !*q || strlen(q) > 200)
        return fail(out, "say a few words of the model's name");
    char enc[700], url[1000], err[300];
    urlenc(q, enc, sizeof enc);
    snprintf(url, sizeof url,
             "https://huggingface.co/api/models?search=%s&filter=gguf"
             "&sort=downloads&direction=-1&limit=30",
             enc);
    struct janas_buf r = {0};
    if (get_json(url, &r, err, sizeof err) != 0) {
        janas_buf_free(&r);
        return fail(out, err);
    }
    struct janas_json_doc *d = janas_json_parse(r.p, r.n, err, sizeof err);
    const struct janas_json *a = d ? janas_json_root(d) : NULL;
    janas_buf_puts(out, "{\"models\": [");
    int k = 0;
    for (const struct janas_json *e =
             a && a->type == JANAS_JSON_ARRAY ? a->child : NULL;
         e; e = e->next) {
        const char *id = janas_json_str(janas_json_get(e, "id"));
        if (!id)
            continue;
        janas_buf_puts(out, k++ ? ", {\"id\": " : "{\"id\": ");
        str(out, id);
        janas_buf_printf(out, ", \"downloads\": %.0f, \"likes\": %.0f}",
                         janas_json_num(janas_json_get(e, "downloads"), 0),
                         janas_json_num(janas_json_get(e, "likes"), 0));
    }
    janas_buf_puts(out, "]}");
    janas_json_free(d);
    janas_buf_free(&r);
    return 0;
}

int brebus_hf_files(struct brebus *b, const char *repo, struct janas_buf *out)
{
    (void)b;
    /* owner/name, as Hugging Face names them */
    int slash = 0, ok = repo && *repo && strlen(repo) < 200;
    for (const char *c = repo; ok && *c; c++) {
        if (*c == '/')
            slash++;
        else
            ok = (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
                 (*c >= '0' && *c <= '9') || *c == '-' || *c == '_' ||
                 *c == '.';
    }
    const char *name = ok ? strchr(repo, '/') : NULL;
    if (!ok || slash != 1 || repo[0] == '/' || repo[0] == '.' || !name ||
        name[1] == '.' || name[1] == 0)
        return fail(out, "not a model of Hugging Face: owner/name");
    char url[600], err[300];
    snprintf(url, sizeof url, "https://huggingface.co/api/models/%s?blobs=true",
             repo);
    struct janas_buf r = {0};
    if (get_json(url, &r, err, sizeof err) != 0) {
        janas_buf_free(&r);
        return fail(out, err);
    }
    struct janas_json_doc *d = janas_json_parse(r.p, r.n, err, sizeof err);
    const struct janas_json *s =
        d ? janas_json_get(janas_json_root(d), "siblings") : NULL;
    janas_buf_puts(out, "{\"repo\": ");
    str(out, repo);
    janas_buf_puts(out, ", \"files\": [");
    int k = 0;
    for (const struct janas_json *e =
             s && s->type == JANAS_JSON_ARRAY ? s->child : NULL;
         e; e = e->next) {
        const char *f = janas_json_str(janas_json_get(e, "rfilename"));
        size_t n = f ? strlen(f) : 0;
        int gguf = n > 5 && !strcmp(f + n - 5, ".gguf");
        int jns = n > 4 && !strcmp(f + n - 4, ".jns");
        if (!gguf && !jns)
            continue;
        /* a GGUF split in parts is not taken yet; a vision encoder is not
           a model to chat with */
        int split = strstr(f, "-of-0") != NULL;
        int side = strstr(f, "mmproj") != NULL;
        janas_buf_puts(out, k++ ? ", {\"file\": " : "{\"file\": ");
        str(out, f);
        janas_buf_printf(out,
                         ", \"bytes\": %.0f, \"split\": %s, \"mmproj\": %s}",
                         janas_json_num(janas_json_get(e, "size"), 0),
                         split ? "true" : "false", side ? "true" : "false");
    }
    janas_buf_puts(out, "]}");
    janas_json_free(d);
    janas_buf_free(&r);
    return 0;
}
