/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * embed.c - the vectors of the pieces and of the queries: from a
 * janas-server's POST /v1/embeddings (or any server with OpenAI's
 * embeddings API), or from an embedding model opened here through
 * libjanas_llm. A query is given the instruction an instructed embedding
 * model (Qwen3-Embedding) reads it by; the pieces go as they are.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/mcp_http.h"
#include "docs.h"

#define QUERY_PREFIX                                                           \
    "Instruct: Given a search query, retrieve relevant passages that "         \
    "answer the query\nQuery: "
#define TIMEOUT_MS 120000 /* a batch of pieces on a slow machine */
#define MAX_ANSWER (256u << 20)

static void unit(float *v, int dim)
{
    double s = 0;
    for (int i = 0; i < dim; i++)
        s += (double)v[i] * v[i];
    float k = s > 0 ? (float)(1 / sqrt(s)) : 0;
    for (int i = 0; i < dim; i++)
        v[i] *= k;
}

int dz_embed_parse(const char *body, size_t n, size_t want, float **out,
                   int *dim, char *model, size_t model_cap, char *why,
                   size_t wn)
{
    *out = NULL;
    char err[128];
    struct janas_json_doc *d = janas_json_parse(body, n, err, sizeof err);
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const struct janas_json *data = o ? janas_json_get(o, "data") : NULL;
    if (!data || data->type != JANAS_JSON_ARRAY) {
        const struct janas_json *e = o ? janas_json_get(o, "error") : NULL;
        const char *m = e ? janas_json_str(janas_json_get(e, "message")) : NULL;
        snprintf(why, wn, "%s", m ? m : "no vectors in the answer");
        janas_json_free(d);
        return -1;
    }
    const char *mn = janas_json_str(janas_json_get(o, "model"));
    if (model_cap)
        snprintf(model, model_cap, "%s", mn ? mn : "");
    size_t k = 0;
    int dm = 0;
    float *v = NULL;
    for (const struct janas_json *it = data->child; it; it = it->next) {
        const struct janas_json *e = janas_json_get(it, "embedding");
        double idx = janas_json_num(janas_json_get(it, "index"), (double)k);
        if (!e || e->type != JANAS_JSON_ARRAY || !e->n || e->n > 65536 ||
            (dm && e->n != (size_t)dm) || idx != (double)k || k >= want)
            break;
        if (!dm) {
            dm = (int)e->n;
            v = malloc(want * (size_t)dm * sizeof(float));
            if (!v)
                break;
        }
        size_t j = 0;
        for (const struct janas_json *x = e->child; x && j < (size_t)dm;
             x = x->next)
            v[k * (size_t)dm + j++] = (float)janas_json_num(x, 0);
        if (j != (size_t)dm)
            break;
        unit(v + k * (size_t)dm, dm);
        k++;
    }
    janas_json_free(d);
    if (!v || k != want) {
        free(v);
        snprintf(why, wn, "the answer holds not one vector for each text");
        return -1;
    }
    *out = v;
    *dim = dm;
    return 0;
}

/* The server's vectors of n texts into a malloc'd *v. 0, or -1. */
static int server_embed(struct dz_embedder *e, const char *const *texts,
                        const size_t *lens, size_t n, int query, float **v,
                        int *dim, char *model, size_t model_cap)
{
    char url[1100];
    snprintf(url, sizeof url, "%s/embeddings", e->url);
    struct janas_url u;
    if (janas_url_parse(url, &u, e->err, sizeof e->err) != 0)
        return -1;
    struct janas_buf body = {0};
    janas_buf_puts(&body, "{\"model\": \"embedding\", \"input\": [");
    for (size_t i = 0; i < n; i++) {
        struct janas_buf t = {0};
        if (query)
            janas_buf_puts(&t, QUERY_PREFIX);
        janas_buf_put(&t, texts[i], lens[i]);
        janas_buf_puts(&body, i ? ", " : "");
        janas_json_write_str(&body, t.p ? t.p : "", t.n);
        body.oom |= t.oom;
        janas_buf_free(&t);
    }
    janas_buf_puts(&body, "]}");
    char auth[sizeof e->key + 32];
    const char *hdr[2] = {"Content-Type: application/json", auth};
    snprintf(auth, sizeof auth, "Authorization: Bearer %s", e->key);
    if (body.oom) {
        janas_buf_free(&body);
        snprintf(e->err, sizeof e->err, "out of memory");
        return -1;
    }
    struct janas_http h;
    int rc =
        janas_http_request(&h, &u, "POST", hdr, e->key[0] ? 2 : 1, body.p,
                           body.n, TIMEOUT_MS, NULL, e->err, sizeof e->err);
    janas_buf_free(&body);
    if (rc != 0)
        return -1;
    struct janas_buf ans = {0};
    int r;
    while ((r = janas_http_body(&h, &ans, TIMEOUT_MS)) == 1)
        if (ans.n > MAX_ANSWER) {
            r = -1;
            break;
        }
    int status = h.head.status;
    janas_http_close(&h);
    if (r != 0 || ans.oom) {
        janas_buf_free(&ans);
        snprintf(e->err, sizeof e->err, "%s: the answer broke off", e->url);
        return -1;
    }
    rc = dz_embed_parse(ans.p ? ans.p : "", ans.n, n, v, dim, model, model_cap,
                        e->err, sizeof e->err);
    if (rc != 0 && status != 200) {
        char m[sizeof e->err];
        snprintf(m, sizeof m, "%s", e->err);
        snprintf(e->err, sizeof e->err, "%.300s answered %d: %.1000s", e->url,
                 status, m);
    }
    janas_buf_free(&ans);
    return rc;
}

int dz_embed_probe(struct dz_embedder *e)
{
    if (e->kind != DZ_EMB_SERVER)
        return dz_embed_state(e, NULL, 0, NULL) ? 0 : -1;
    const char *t = "hello";
    size_t l = 5;
    float *v = NULL;
    int dim = 0;
    char model[256] = "";
    pthread_mutex_lock(&e->mu);
    int rc = server_embed(e, &t, &l, 1, 0, &v, &dim, model, sizeof model);
    free(v);
    if (rc == 0) {
        snprintf(e->name, sizeof e->name, "server %s, model %s, %d numbers",
                 e->url, model[0] ? model : "?", dim);
        e->dim = dim;
        e->ready = 1;
        e->err[0] = 0;
    }
    pthread_mutex_unlock(&e->mu);
    return rc;
}

int dz_embedder_open(struct dz_embedder *e, const char *server, const char *key,
                     const char *model, char *why, size_t wn)
{
    memset(e, 0, sizeof *e);
    pthread_mutex_init(&e->mu, NULL);
    if (server && *server) {
        e->kind = DZ_EMB_SERVER;
        /* the base URL, with its /v1 or without: the service adds it */
        size_t n = strlen(server);
        while (n && server[n - 1] == '/')
            n--;
        int v1 = n >= 3 && !memcmp(server + n - 3, "/v1", 3);
        snprintf(e->url, sizeof e->url, "%.*s%s", (int)n, server,
                 v1 ? "" : "/v1");
        snprintf(e->key, sizeof e->key, "%s", key ? key : "");
        struct janas_url u;
        if (janas_url_parse(e->url, &u, why, wn) != 0)
            return -1;
        snprintf(e->name, sizeof e->name, "server %s", e->url);
        if (dz_embed_probe(e) != 0)
            fprintf(stderr, "janas-docs: %s, tried again later\n", e->err);
        return 0;
    }
    if (model && *model) {
        struct janas_llm_params p;
        janas_llm_params_default(&p);
        p.n_ctx = 2048; /* a piece and its instruction */
        p.no_preload = 1;
        if (janas_llm_open(model, &p, &e->llm) != JANAS_LLM_OK) {
            snprintf(why, wn, "%s: %s", model, janas_llm_last_error());
            return -1;
        }
        e->dim = janas_llm_embed_dim(e->llm);
        if (!e->dim) {
            janas_llm_close(e->llm);
            e->llm = NULL;
            snprintf(why, wn, "%s: not an embedding model", model);
            return -1;
        }
        char nm[256] = "";
        int32_t len = 0;
        janas_llm_name(e->llm, nm, sizeof nm, &len);
        const char *base = strrchr(model, '/');
        snprintf(e->name, sizeof e->name, "model %s (%s), %d numbers",
                 base ? base + 1 : model, nm[0] ? nm : "?", e->dim);
        e->kind = DZ_EMB_MODEL;
        e->ready = 1;
        return 0;
    }
    e->kind = DZ_EMB_NONE;
    snprintf(e->err, sizeof e->err,
             "no embedding model: the documents are searched by their words");
    return 0;
}

int dz_embed(struct dz_embedder *e, const char *const *texts,
             const size_t *lens, size_t n, int query, float *out)
{
    pthread_mutex_lock(&e->mu);
    if (!e->ready) {
        pthread_mutex_unlock(&e->mu);
        return -1;
    }
    int rc = 0;
    if (e->kind == DZ_EMB_SERVER) {
        float *v = NULL;
        int dim = 0;
        rc = server_embed(e, texts, lens, n, query, &v, &dim, NULL, 0);
        if (rc == 0 && dim != e->dim) {
            snprintf(e->err, sizeof e->err,
                     "the server's vectors have %d numbers, not %d: another "
                     "model",
                     dim, e->dim);
            e->ready = 0; /* probed again, then the index is made again */
            rc = -1;
        }
        if (rc == 0)
            memcpy(out, v, n * (size_t)dim * sizeof(float));
        free(v);
    } else {
        for (size_t i = 0; i < n && rc == 0; i++) {
            struct janas_buf t = {0};
            if (query)
                janas_buf_puts(&t, QUERY_PREFIX);
            janas_buf_put(&t, texts[i], lens[i]);
            int32_t got = 0;
            if (t.oom ||
                janas_llm_embed(e->llm, t.p, (int32_t)t.n, e->dim,
                                out + i * (size_t)e->dim, &got,
                                NULL) != JANAS_LLM_OK ||
                got != e->dim) {
                snprintf(e->err, sizeof e->err, "the model: %s",
                         janas_llm_last_error());
                rc = -1;
            }
            janas_buf_free(&t);
        }
    }
    pthread_mutex_unlock(&e->mu);
    return rc;
}

int dz_embed_state(struct dz_embedder *e, char *name, size_t cap, int *dim)
{
    pthread_mutex_lock(&e->mu);
    int ready = e->ready;
    if (cap)
        snprintf(name, cap, "%s", e->name);
    if (dim)
        *dim = e->dim;
    pthread_mutex_unlock(&e->mu);
    return ready;
}

void dz_embed_error(struct dz_embedder *e, char *out, size_t cap)
{
    pthread_mutex_lock(&e->mu);
    snprintf(out, cap, "%s", e->err);
    pthread_mutex_unlock(&e->mu);
}

void dz_embedder_close(struct dz_embedder *e)
{
    if (e->llm)
        janas_llm_close(e->llm);
    e->llm = NULL;
    pthread_mutex_destroy(&e->mu);
}
