/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * hf_fetch.c - parts of a Hugging Face repository (see hf_fetch.h), with
 * the HTTP client of the MCP transport (mcp_http.c) and its TLS.
 */
#include "common/hf_fetch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/mcp_http.h"

#define TIMEOUT_MS 60000
#define MAX_REDIRECTS 8
/* neighbouring tensors are read in one request: a Qwen3-Next block is
   1,500 of them, each a TLS connection of its own otherwise */
#define JOIN_GAP (1u << 20)
#define JOIN_MAX (256u << 20)

/* The address of a file of the repository: 0, or -1 when too long. */
static int file_url(char *url, size_t len, const char *repo, const char *rev,
                    const char *path, char *err, size_t err_len)
{
    int k = snprintf(url, len, "https://huggingface.co/%s/resolve/%s/%s", repo,
                     rev ? rev : "main", path);
    if (k < 0 || (size_t)k >= len) {
        snprintf(err, err_len, "%s: a name too long", path);
        return -1;
    }
    return 0;
}

/*
 * GET of url, following redirects, with an optional range. *final gets the
 * address that answered (reused for the next ranges of the same file: the
 * redirect is asked once). A range must come back as 206 with its length.
 */
static int get_url(const char *url, const int64_t *from, const int64_t *to,
                   struct janas_buf *out, char *final, size_t final_len,
                   char *err, size_t err_len)
{
    struct janas_url u;
    if (janas_url_parse(url, &u, err, err_len) != 0)
        return -1;
    for (int hop = 0; hop <= MAX_REDIRECTS; hop++) {
        char range[80], auth[512];
        const char *hdr[2];
        size_t nh = 0;
        if (from) {
            snprintf(range, sizeof(range), "Range: bytes=%lld-%lld",
                     (long long)*from, (long long)*to - 1);
            hdr[nh++] = range;
        }
        const char *tok = getenv("HF_TOKEN");
        if (tok && *tok && strcmp(u.host, "huggingface.co") == 0 &&
            strlen(tok) < sizeof(auth) - 32) {
            snprintf(auth, sizeof(auth), "Authorization: Bearer %s", tok);
            hdr[nh++] = auth;
        }
        struct janas_http h;
        if (janas_http_request(&h, &u, "GET", hdr, nh, NULL, 0, TIMEOUT_MS,
                               NULL, err, err_len) != 0)
            return -1;
        int st = h.head.status;
        if (st == 301 || st == 302 || st == 303 || st == 307 || st == 308) {
            char loc[sizeof(h.head.location)];
            snprintf(loc, sizeof(loc), "%s", h.head.location);
            janas_http_close(&h);
            if (!*loc) {
                snprintf(err, err_len, "%s: a redirect without an address",
                         u.host);
                return -1;
            }
            if (loc[0] == '/') { /* on the same server */
                if (strlen(loc) >= sizeof(u.path)) {
                    snprintf(err, err_len, "%s: a redirect too long", u.host);
                    return -1;
                }
                snprintf(u.path, sizeof(u.path), "%s", loc);
            } else if (janas_url_parse(loc, &u, err, err_len) != 0) {
                return -1;
            }
            continue;
        }
        int64_t want = from ? *to - *from : -1;
        if (st != (from ? 206 : 200) ||
            (from && h.head.length >= 0 && h.head.length != want)) {
            snprintf(err, err_len,
                     st == 200 && from ? "%s: the range was not honoured"
                     : st == 401 || st == 403
                         ? "%s: access refused (HTTP %d): no such repository, "
                           "or a gated one that wants HF_TOKEN"
                         : "%s: HTTP %d",
                     u.host, st);
            janas_http_close(&h);
            return -1;
        }
        size_t before = out->n;
        int r;
        while ((r = janas_http_body(&h, out, TIMEOUT_MS)) == 1)
            ;
        janas_http_close(&h);
        if (r != 0 || out->oom ||
            (want >= 0 && (int64_t)(out->n - before) != want)) {
            snprintf(err, err_len, "%s: the transfer broke off", u.host);
            return -1;
        }
        if (final)
            snprintf(final, final_len, "%s://%s:%s%s", u.tls ? "https" : "http",
                     u.host, u.port, u.path);
        return 0;
    }
    snprintf(err, err_len, "%s: too many redirects", url);
    return -1;
}

int janas_hf_get(const char *repo, const char *rev, const char *path,
                 const int64_t *from, const int64_t *to, struct janas_buf *out,
                 char *err, size_t err_len)
{
    char url[1024];
    if (file_url(url, sizeof(url), repo, rev, path, err, err_len) != 0)
        return -1;
    return get_url(url, from, to, out, NULL, 0, err, err_len);
}

/* A shard of the checkpoint: where it answers, and its parsed header. */
struct shard {
    char name[256];
    char *url; /* after the redirect */
    uint64_t data_start;
    struct janas_json_doc *doc;
};

static int open_shard(const char *repo, const char *rev, struct shard *s,
                      char *err, size_t err_len)
{
    char url[1024];
    if (file_url(url, sizeof(url), repo, rev, s->name, err, err_len) != 0)
        return -1;
    s->url = malloc(8192);
    struct janas_buf b = {0};
    int64_t a = 0, z = 8;
    if (!s->url || get_url(url, &a, &z, &b, s->url, 8192, err, err_len) != 0) {
        janas_buf_free(&b);
        return -1;
    }
    uint64_t hl = 0;
    memcpy(&hl, b.p, 8);
    b.n = 0;
    a = 8;
    z = 8 + (int64_t)hl;
    if (hl == 0 || hl > (100u << 20) ||
        get_url(s->url, &a, &z, &b, NULL, 0, err, err_len) != 0) {
        if (hl == 0 || hl > (100u << 20))
            snprintf(err, err_len, "%s: not a safetensors file", s->name);
        janas_buf_free(&b);
        return -1;
    }
    s->data_start = 8 + hl;
    s->doc = janas_json_parse(b.p, b.n, err, err_len);
    janas_buf_free(&b);
    return s->doc ? 0 : -1;
}

/* One tensor to fetch: its shard, and where it lies there and here. */
struct want {
    const struct janas_json *name, *entry;
    int shard;
    uint64_t begin, end, at;
};

/* By shard, then by where the tensor lies in it. */
static int by_place(const void *a, const void *b)
{
    const struct want *x = *(const struct want *const *)a,
                      *y = *(const struct want *const *)b;
    if (x->shard != y->shard)
        return x->shard < y->shard ? -1 : 1;
    return (x->begin > y->begin) - (x->begin < y->begin);
}

static int by_name(const void *a, const void *b)
{
    const struct want *x = a, *y = b;
    size_t n =
        x->name->key_n < y->name->key_n ? x->name->key_n : y->name->key_n;
    int c = memcmp(x->name->key, y->name->key, n);
    return c ? c
             : (x->name->key_n > y->name->key_n) -
                   (x->name->key_n < y->name->key_n);
}

int janas_hf_fetch_tensors(const char *repo, const char *rev,
                           const char *prefix, const char *out_path, char *err,
                           size_t err_len)
{
    int rc = -1, n_shard = 0, n_want = 0;
    struct shard shards[64] = {0};
    struct want *wants = NULL, **order = NULL;
    struct janas_json_doc *index = NULL;
    FILE *out = NULL;
    struct janas_buf idx = {0}, hdr = {0}, data = {0};
    size_t plen = strlen(prefix);

    /* which shards hold the tensors: the index, or the one file */
    if (janas_hf_get(repo, rev, "model.safetensors.index.json", NULL, NULL,
                     &idx, err, err_len) == 0) {
        if (!(index = janas_json_parse(idx.p, idx.n, err, err_len)))
            goto done;
        const struct janas_json *map =
            janas_json_get(janas_json_root(index), "weight_map");
        for (const struct janas_json *e = map ? map->child : NULL; e;
             e = e->next) {
            const char *f = janas_json_str(e);
            if (!f || e->key_n < plen || memcmp(e->key, prefix, plen) != 0)
                continue;
            int k = 0;
            while (k < n_shard && strcmp(shards[k].name, f) != 0)
                k++;
            if (k == n_shard) {
                if (n_shard == 64 || strlen(f) >= sizeof(shards[0].name)) {
                    snprintf(err, err_len, "too many shards");
                    goto done;
                }
                snprintf(shards[n_shard++].name, sizeof(shards[0].name), "%s",
                         f);
            }
        }
    } else {
        snprintf(shards[n_shard++].name, sizeof(shards[0].name),
                 "model.safetensors");
    }
    printf("%s: %d file%s to read from\n", repo, n_shard,
           n_shard == 1 ? "" : "s");

    /* the headers, and the tensors wanted in each */
    for (int k = 0; k < n_shard; k++) {
        if (open_shard(repo, rev, &shards[k], err, err_len) != 0)
            goto done;
        const struct janas_json *root = janas_json_root(shards[k].doc);
        for (const struct janas_json *e = root->child; e; e = e->next) {
            if (e->key_n < plen || memcmp(e->key, prefix, plen) != 0)
                continue;
            const struct janas_json *off = janas_json_get(e, "data_offsets");
            if (!janas_json_is(janas_json_get(e, "dtype"), "BF16") || !off ||
                off->n != 2) {
                snprintf(err, err_len, "%.*s: not a bf16 tensor", (int)e->key_n,
                         e->key);
                goto done;
            }
            struct want *w = realloc(wants, (n_want + 1) * sizeof(*w));
            if (!w)
                goto oom;
            wants = w;
            wants[n_want++] = (struct want){
                .name = e,
                .entry = e,
                .shard = k,
                .begin = (uint64_t)janas_json_num(off->child, -1),
                .end = (uint64_t)janas_json_num(off->child->next, -1)};
        }
    }
    if (n_want == 0) {
        snprintf(err, err_len, "%s: no tensor named %s*", repo, prefix);
        goto done;
    }
    qsort(wants, (size_t)n_want, sizeof(*wants), by_name);

    /* the new header: same entries, offsets in the new file */
    uint64_t at = 0;
    janas_buf_puts(&hdr, "{");
    for (int i = 0; i < n_want; i++) {
        struct want *w = &wants[i];
        if (w->end < w->begin) {
            snprintf(err, err_len, "bad offsets");
            goto done;
        }
        w->at = at;
        janas_buf_puts(&hdr, i ? "," : "");
        janas_json_write_str(&hdr, w->name->key, w->name->key_n);
        janas_buf_puts(&hdr, ":{\"dtype\":\"BF16\",\"shape\":");
        janas_json_write(&hdr, janas_json_get(w->entry, "shape"));
        janas_buf_printf(&hdr, ",\"data_offsets\":[%llu,%llu]}",
                         (unsigned long long)at,
                         (unsigned long long)(at + w->end - w->begin));
        at += w->end - w->begin;
    }
    janas_buf_puts(&hdr, "}");
    while (hdr.n % 8)
        janas_buf_puts(&hdr, " ");
    if (hdr.oom)
        goto oom;
    uint64_t hl = hdr.n;
    if (!(out = fopen(out_path, "wb")) || fwrite(&hl, 8, 1, out) != 1 ||
        fwrite(hdr.p, 1, hdr.n, out) != hdr.n) {
        snprintf(err, err_len, "cannot write %s", out_path);
        goto done;
    }

    /* the data: runs of neighbouring tensors of a shard in one request
       each, every tensor then written where the new header puts it */
    if (!(order = malloc((size_t)n_want * sizeof(*order))))
        goto oom;
    for (int i = 0; i < n_want; i++)
        order[i] = &wants[i];
    qsort(order, (size_t)n_want, sizeof(*order), by_place);
    uint64_t base = 8 + hl, got = 0;
    for (int i = 0, j; i < n_want; i = j) {
        const struct shard *s = &shards[order[i]->shard];
        uint64_t first = order[i]->begin, last = order[i]->end;
        for (j = i + 1;
             j < n_want && order[j]->shard == order[i]->shard &&
             order[j]->begin >= last && order[j]->begin - last <= JOIN_GAP &&
             order[j]->end - first <= JOIN_MAX;
             j++)
            last = order[j]->end;
        int64_t a = (int64_t)(s->data_start + first),
                z = (int64_t)(s->data_start + last);
        data.n = 0;
        if (last > first &&
            get_url(s->url, &a, &z, &data, NULL, 0, err, err_len) != 0)
            goto done;
        for (int k = i; k < j; k++) {
            const struct want *w = order[k];
            size_t len = (size_t)(w->end - w->begin);
            if (fseeko(out, (off_t)(base + w->at), SEEK_SET) != 0 ||
                fwrite(data.p + (w->begin - first), 1, len, out) != len) {
                snprintf(err, err_len, "cannot write %s", out_path);
                goto done;
            }
        }
        got += last - first;
        printf("  %d tensor%s, %.1f MB (%.0f MB so far)\n", j - i,
               j - i == 1 ? "" : "s", (last - first) / 1e6, got / 1e6);
    }
    if (fclose(out) != 0) {
        out = NULL;
        snprintf(err, err_len, "cannot write %s", out_path);
        goto done;
    }
    out = NULL;
    rc = n_want;
    goto done;
oom:
    snprintf(err, err_len, "out of memory");
done:
    if (out)
        fclose(out);
    for (int k = 0; k < n_shard; k++) {
        free(shards[k].url);
        janas_json_free(shards[k].doc);
    }
    free(wants);
    free(order);
    janas_json_free(index);
    janas_buf_free(&idx);
    janas_buf_free(&hdr);
    janas_buf_free(&data);
    return rc;
}
