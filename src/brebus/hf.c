/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf.c - Hugging Face searched from Discover: the models whose names hold
 * the words asked, with GGUF files, the most downloaded first (the public
 * API, /api/models), and the GGUF and JNS files of one, with their sizes.
 * A file is then downloaded as any hf:<owner>/<repo>/<file> (get.c). The
 * page cannot reach another site; Brebus asks for it.
 *
 * Before a download, one file's first megabyte, asked as a range: its
 * GGUF header names the model's architecture, which this engine runs or
 * does not - a likely conversion, not a proof (its tensors' types and the
 * rest are known only once converted and loaded).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brebus.h"
#include "common/hf_fetch.h"
#include "llm/arch.h"

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
    /* the licence its card declares, and whether it asks to agree first
       (gated), for the page to show before a download */
    const struct janas_json *root = d ? janas_json_root(d) : NULL;
    const struct janas_json *card = janas_json_get(root, "cardData");
    static const char *const LK[][2] = {{"license", "license"},
                                        {"license_name", "license_name"},
                                        {"license_link", "license_link"}};
    for (int i = 0; i < 3; i++) {
        const char *v = janas_json_str(janas_json_get(card, LK[i][0]));
        if (v && strlen(v) < 512) {
            janas_buf_printf(out, ", \"%s\": ", LK[i][1]);
            str(out, v);
        }
    }
    const struct janas_json *gated = janas_json_get(root, "gated");
    if (gated && gated->type != JANAS_JSON_FALSE &&
        gated->type != JANAS_JSON_NULL)
        janas_buf_puts(out, ", \"gated\": true");
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

/* ---- one GGUF's header, read from its first megabyte ---- */

struct rd {
    const unsigned char *p;
    size_t n, at;
};

static int rd_u(struct rd *r, size_t k, uint64_t *v)
{
    if (r->at + k > r->n)
        return -1;
    uint64_t x = 0;
    for (size_t i = 0; i < k; i++)
        x |= (uint64_t)r->p[r->at + i] << (8 * i);
    r->at += k;
    *v = x;
    return 0;
}

/* a value of type t skipped (its string kept when s is not NULL) */
static int rd_skip(struct rd *r, uint32_t t, const char **s, size_t *sn)
{
    static const size_t w[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    uint64_t n, t2;
    if (t == 8) { /* string */
        if (rd_u(r, 8, &n) || r->at + n > r->n)
            return -1;
        if (s)
            *s = (const char *)r->p + r->at, *sn = (size_t)n;
        r->at += (size_t)n;
        return 0;
    }
    if (t == 9) { /* array: its type, its length, its values */
        if (rd_u(r, 4, &t2) || rd_u(r, 8, &n) || t2 > 12)
            return -1;
        if (t2 != 8 && t2 != 9) {
            if (r->at + n * w[t2] > r->n)
                return -1;
            r->at += (size_t)(n * w[t2]);
            return 0;
        }
        for (uint64_t i = 0; i < n; i++)
            if (rd_skip(r, (uint32_t)t2, NULL, NULL))
                return -1;
        return 0;
    }
    if (t > 12)
        return -1;
    if (r->at + w[t] > r->n)
        return -1;
    r->at += w[t];
    return 0;
}

int brebus_hf_check(struct brebus *b, const char *repo, const char *file,
                    struct janas_buf *out)
{
    (void)b;
    int ok = repo && file && strlen(repo) < 200 && strlen(file) < 300 &&
             strchr(repo, '/') && !strstr(file, "..") && file[0] != '/';
    for (const char *c = repo; ok && *c; c++)
        ok = (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
             (*c >= '0' && *c <= '9') || strchr("-_./", *c);
    size_t fn = ok ? strlen(file) : 0;
    if (!ok || fn < 6 || strcmp(file + fn - 5, ".gguf"))
        return fail(out, "not a GGUF file of Hugging Face");
    char url[1100], fe[700], err[300];
    urlenc(file, fe, sizeof fe);
    /* the path's slashes kept */
    for (char *c = fe; (c = strstr(c, "%2F"));)
        c[0] = '/', memmove(c + 1, c + 3, strlen(c + 3) + 1);
    snprintf(url, sizeof url, "https://huggingface.co/%s/resolve/main/%s", repo,
             fe);
    struct janas_buf r = {0};
    int64_t from = 0, to = (1 << 20) - 1;
    if (janas_hf_get_url(url, &from, &to, &r, NULL, 0, err, sizeof err) != 0) {
        janas_buf_free(&r);
        janas_buf_puts(out, "{\"state\": \"closed\", \"why\": ");
        str(out, err);
        janas_buf_puts(out, "}");
        return 0; /* an answer: Hugging Face would not give it */
    }
    struct rd d = {(const unsigned char *)r.p, r.n, 0};
    uint64_t magic, ver, nt, nkv;
    const char *arch = NULL;
    size_t an = 0;
    int good = !rd_u(&d, 4, &magic) && magic == 0x46554747 /* "GGUF" */ &&
               !rd_u(&d, 4, &ver) && ver >= 2 && !rd_u(&d, 8, &nt) &&
               !rd_u(&d, 8, &nkv);
    for (uint64_t i = 0; good && i < nkv && !arch; i++) {
        uint64_t kl, t;
        if (rd_u(&d, 8, &kl) || d.at + kl > d.n) {
            good = 0;
            break;
        }
        const char *k = (const char *)d.p + d.at;
        d.at += (size_t)kl;
        if (rd_u(&d, 4, &t)) {
            good = 0;
            break;
        }
        const char *s = NULL;
        size_t sn = 0;
        if (rd_skip(&d, (uint32_t)t, &s, &sn)) {
            good = 0;
            break;
        }
        if (kl == 20 && !memcmp(k, "general.architecture", 20) && t == 8)
            arch = s, an = sn;
    }
    char a[64] = "";
    if (arch)
        snprintf(a, sizeof a, "%.*s", (int)(an < 63 ? an : 63), arch);
    int runs = arch && janas_arch_find(a, (unsigned)strlen(a)) != NULL;
    janas_buf_puts(out, "{\"state\": ");
    str(out, !good || !arch ? "unknown" : runs ? "likely" : "unsupported");
    janas_buf_puts(out, ", \"arch\": ");
    str(out, a);
    janas_buf_puts(out, "}");
    janas_buf_free(&r);
    return 0;
}
