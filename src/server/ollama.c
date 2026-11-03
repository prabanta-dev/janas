/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * ollama.c - Ollama's API beside OpenAI's, for the programs that know a
 * local model only as Ollama's: VS Code's chat among them, which finds
 * the models with /api/version, /api/tags and /api/show and then talks to
 * /v1/chat/completions. Here the routes that describe the models, and the
 * embeddings; the chat and the completions are in ollama_chat.c.
 *
 * The models are this server's: the chat model and the embedding model,
 * under the names their clients use (--name). Nothing is pulled,
 * pushed, copied, created or deleted: those routes answer that the models
 * are the files the server was started with.
 */
#include "ollama.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* The version of Ollama's API these routes follow. VS Code's chat asks for
   0.6.4 at least. */
#define OL_VERSION "0.13.0"

int ol_error(struct srv_req *r, unsigned status, const char *fmt, ...)
{
    char msg[600];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_strcpy(d, o, "error", msg);
    size_t n = 0;
    char *json = yyjson_mut_write(d, 0, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, status, json, n);
}

static void rfc3339(time_t t, long ns, char *buf, size_t cap)
{
    struct tm tm;
    gmtime_r(&t, &tm);
    char s[32];
    strftime(s, sizeof s, "%Y-%m-%dT%H:%M:%S", &tm);
    snprintf(buf, cap, "%s.%09ldZ", s, ns);
}

void ol_now(char *buf, size_t cap)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    rfc3339(ts.tv_sec, ts.tv_nsec, buf, cap);
}

/* name, or name followed by ":latest", is id */
static int same(const char *name, const char *id)
{
    if (!id)
        return 0;
    size_t n = strlen(id);
    return !strncmp(name, id, n) && (!name[n] || !strcmp(name + n, ":latest"));
}

int ol_model(const struct srv *s, const char *name)
{
    if (!name)
        return 0;
    if (same(name, s->cfg.model_id))
        return 1;
    if (s->cfg.embed_id && same(name, s->cfg.embed_id))
        return 2;
    return 0;
}

static int reply_doc(struct srv_req *r, yyjson_mut_doc *d)
{
    size_t n = 0;
    char *json = yyjson_mut_write(d, 0, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, n);
}

static uint64_t file_size(const char *path)
{
    struct stat st;
    return path && stat(path, &st) == 0 ? (uint64_t)st.st_size : 0;
}

/* A model's details: the file's format, the architecture as its family. */
static void details(yyjson_mut_doc *d, yyjson_mut_val *o, const char *arch)
{
    yyjson_mut_val *dt = yyjson_mut_obj_add_obj(d, o, "details");
    yyjson_mut_obj_add_str(d, dt, "parent_model", "");
    yyjson_mut_obj_add_str(d, dt, "format", "jns");
    yyjson_mut_obj_add_strcpy(d, dt, "family", arch);
    yyjson_mut_val *fs = yyjson_mut_obj_add_arr(d, dt, "families");
    yyjson_mut_arr_add_strcpy(d, fs, arch);
    yyjson_mut_obj_add_str(d, dt, "parameter_size", "");
    yyjson_mut_obj_add_str(d, dt, "quantization_level", "");
}

/* One model of /api/tags and /api/ps: kind 1 the chat model, 2 the
   embedding model. */
static yyjson_mut_val *model_entry(struct srv_req *r, yyjson_mut_doc *d,
                                   int kind, int running)
{
    const struct srv_config *c = &r->srv->cfg;
    const struct srv_model_info *mi = srv_engine_info(r->srv->engine);
    const char *id = kind == 1 ? c->model_id : c->embed_id;
    const char *path = kind == 1 ? c->model_path : c->embed_path;
    char when[48];
    rfc3339((time_t)srv_engine_created(r->srv->engine), 0, when, sizeof when);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, o, "name", id);
    yyjson_mut_obj_add_strcpy(d, o, "model", id);
    if (!running)
        yyjson_mut_obj_add_strcpy(d, o, "modified_at", when);
    yyjson_mut_obj_add_uint(d, o, "size", file_size(path));
    yyjson_mut_obj_add_str(d, o, "digest", "");
    details(d, o, kind == 1 && mi->arch[0] ? mi->arch : "");
    if (running) {
        /* loaded for as long as the server runs */
        yyjson_mut_obj_add_str(d, o, "expires_at", "2318-01-01T00:00:00Z");
        yyjson_mut_obj_add_uint(d, o, "size_vram", 0);
        if (kind == 1)
            yyjson_mut_obj_add_uint(d, o, "context_length", mi->n_ctx);
    }
    return o;
}

static int models(struct srv_req *r, int running)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, root);
    yyjson_mut_val *arr = yyjson_mut_obj_add_arr(d, root, "models");
    yyjson_mut_arr_append(arr, model_entry(r, d, 1, running));
    if (r->srv->cfg.embed_id)
        yyjson_mut_arr_append(arr, model_entry(r, d, 2, running));
    return reply_doc(r, d);
}

static int ol_tags(struct srv_req *r)
{
    return models(r, 0);
}

static int ol_ps(struct srv_req *r)
{
    return models(r, 1);
}

static int ol_version(struct srv_req *r)
{
    static const char v[] = "{\"version\":\"" OL_VERSION "\"}";
    return srv_reply_json(r, 200, strdup(v), sizeof v - 1);
}

/* The body as a JSON object, or an answer already given (*ret). */
static yyjson_doc *read_body(struct srv_req *r, int *ret)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc))) {
        yyjson_doc_free(doc);
        *ret = ol_error(r, 400, "the body must be a JSON object");
        return NULL;
    }
    return doc;
}

static int not_found(struct srv_req *r, const char *name)
{
    return ol_error(r, 404, "model '%s' not found: this server runs '%s'",
                    name ? name : "", r->srv->cfg.model_id);
}

/*
 * What a model is and can do. VS Code's chat reads the capabilities (tools
 * and vision), the architecture and the context from model_info; Ollama's
 * keys are GGUF's, which is where Janas's files come from too.
 */
static int ol_show(struct srv_req *r)
{
    int ret;
    yyjson_doc *doc = read_body(r, &ret);
    if (!doc)
        return ret;
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *name = yyjson_get_str(yyjson_obj_get(root, "model"));
    if (!name) /* the older field */
        name = yyjson_get_str(yyjson_obj_get(root, "name"));
    int kind = ol_model(r->srv, name);
    if (!kind) {
        ret = not_found(r, name);
        yyjson_doc_free(doc);
        return ret;
    }
    yyjson_doc_free(doc);
    const struct srv_model_info *mi = srv_engine_info(r->srv->engine);
    const char *arch = kind == 1 && mi->arch[0] ? mi->arch : "";
    char when[48];
    rfc3339((time_t)srv_engine_created(r->srv->engine), 0, when, sizeof when);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_str(d, o, "modelfile", "");
    yyjson_mut_obj_add_str(d, o, "parameters", "");
    yyjson_mut_obj_add_str(d, o, "template", "");
    details(d, o, arch);
    yyjson_mut_val *info = yyjson_mut_obj_add_obj(d, o, "model_info");
    yyjson_mut_obj_add_strcpy(d, info, "general.architecture", arch);
    if (kind == 1) {
        yyjson_mut_obj_add_strcpy(d, info, "general.basename",
                                  mi->name[0] ? mi->name
                                              : r->srv->cfg.model_id);
        char key[96];
        snprintf(key, sizeof key, "%s.context_length", arch);
        yyjson_mut_obj_add(info, yyjson_mut_strcpy(d, key),
                           yyjson_mut_uint(d, mi->n_ctx));
    }
    yyjson_mut_val *caps = yyjson_mut_obj_add_arr(d, o, "capabilities");
    if (kind == 2) {
        yyjson_mut_arr_add_str(d, caps, "embedding");
    } else {
        yyjson_mut_arr_add_str(d, caps, "completion");
        if (mi->tools)
            yyjson_mut_arr_add_str(d, caps, "tools");
        yyjson_mut_arr_add_str(d, caps, "insert"); /* a suffix: fill in */
        if (mi->sees)
            yyjson_mut_arr_add_str(d, caps, "vision");
        if (mi->reasons)
            yyjson_mut_arr_add_str(d, caps, "thinking");
    }
    yyjson_mut_obj_add_strcpy(d, o, "modified_at", when);
    return reply_doc(r, d);
}

/* What a model's files cannot do through an API. */
static int ol_refuse(struct srv_req *r)
{
    return ol_error(r, 403,
                    "the models of this server are the files it was started "
                    "with: they are not pulled, pushed, created, copied or "
                    "deleted through the API");
}

/*
 * Embeddings: the request made OpenAI's and answered by /embeddings,
 * whose answer is made Ollama's. /api/embeddings, the older route, takes
 * one prompt and gives one vector.
 */
static int embed(struct srv_req *r, int old)
{
    int ret;
    yyjson_doc *doc = read_body(r, &ret);
    if (!doc)
        return ret;
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *name = yyjson_get_str(yyjson_obj_get(root, "model"));
    int kind = ol_model(r->srv, name);
    if (kind != 2) {
        ret = !r->srv->cfg.embed_id
                  ? ol_error(r, 400,
                             "this server has no embedding model: start it "
                             "with --embedding-model")
              : kind == 1 ? ol_error(r, 400,
                                     "\"%s\" does not support embeddings: "
                                     "the embedding model is '%s'",
                                     name, r->srv->cfg.embed_id)
                          : not_found(r, name);
        yyjson_doc_free(doc);
        return ret;
    }
    yyjson_val *in = yyjson_obj_get(root, old ? "prompt" : "input");
    yyjson_val *dims = yyjson_obj_get(root, "dimensions");
    yyjson_mut_doc *q = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *qo = yyjson_mut_obj(q);
    yyjson_mut_doc_set_root(q, qo);
    yyjson_mut_obj_add_strcpy(q, qo, "model", r->srv->cfg.embed_id);
    if (in)
        yyjson_mut_obj_add_val(q, qo, "input", yyjson_val_mut_copy(q, in));
    if (dims && !old)
        yyjson_mut_obj_add_val(q, qo, "dimensions",
                               yyjson_val_mut_copy(q, dims));
    yyjson_doc_free(doc);
    size_t n = 0;
    char *body = yyjson_mut_write(q, 0, &n);
    yyjson_mut_doc_free(q);
    if (!body)
        return ol_error(r, 500, "out of memory");
    struct srv_capture cap = {0};
    struct srv_req sub = *r;
    sub.conn = NULL;
    sub.cap = &cap;
    sub.body = body;
    sub.n_body = n;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    srv_embeddings_create(&sub);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    free(body);
    yyjson_doc *ans = cap.body ? yyjson_read(cap.body, cap.n, 0) : NULL;
    yyjson_val *ar = yyjson_doc_get_root(ans);
    if (cap.status != 200 || !ar) {
        yyjson_val *e = yyjson_obj_get(ar, "error");
        const char *msg = yyjson_get_str(yyjson_obj_get(e, "message"));
        ret = ol_error(r, cap.status >= 400 ? cap.status : 500, "%s",
                       msg ? msg : "the embedding failed");
        yyjson_doc_free(ans);
        free(cap.body);
        return ret;
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_val *data = yyjson_obj_get(ar, "data");
    size_t idx, max;
    yyjson_val *it;
    if (old) {
        yyjson_val *first = yyjson_arr_get_first(data);
        yyjson_mut_obj_add_val(
            d, o, "embedding",
            yyjson_val_mut_copy(d, yyjson_obj_get(first, "embedding")));
    } else {
        yyjson_mut_obj_add_strcpy(d, o, "model", r->srv->cfg.embed_id);
        yyjson_mut_val *embs = yyjson_mut_obj_add_arr(d, o, "embeddings");
        yyjson_arr_foreach(data, idx, max, it)
        {
            yyjson_mut_arr_append(
                embs, yyjson_val_mut_copy(d, yyjson_obj_get(it, "embedding")));
        }
        double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                    (double)(t1.tv_nsec - t0.tv_nsec);
        yyjson_mut_obj_add_uint(d, o, "total_duration", (uint64_t)ns);
        yyjson_mut_obj_add_uint(d, o, "load_duration", 0);
        yyjson_val *u = yyjson_obj_get(ar, "usage");
        yyjson_mut_obj_add_uint(
            d, o, "prompt_eval_count",
            yyjson_get_uint(yyjson_obj_get(u, "prompt_tokens")));
    }
    yyjson_doc_free(ans);
    free(cap.body);
    return reply_doc(r, d);
}

static int ol_embed(struct srv_req *r)
{
    return embed(r, 0);
}

static int ol_embeddings(struct srv_req *r)
{
    return embed(r, 1);
}

static const struct {
    const char *method, *path;
    srv_handler h;
} ol_routes[] = {
    {"GET", "/api/version", ol_version},
    {"GET", "/api/tags", ol_tags},
    {"GET", "/api/ps", ol_ps},
    {"POST", "/api/show", ol_show},
    {"POST", "/api/chat", ol_chat},
    {"POST", "/api/generate", ol_generate},
    {"POST", "/api/embed", ol_embed},
    {"POST", "/api/embeddings", ol_embeddings},
    {"POST", "/api/pull", ol_refuse},
    {"POST", "/api/push", ol_refuse},
    {"POST", "/api/create", ol_refuse},
    {"POST", "/api/copy", ol_refuse},
    {"DELETE", "/api/delete", ol_refuse},
};

srv_handler srv_ollama_route(const char *method, const char *path)
{
    for (size_t i = 0; i < sizeof ol_routes / sizeof ol_routes[0]; i++)
        if (!strcmp(path, ol_routes[i].path) &&
            !strcmp(method, ol_routes[i].method))
            return ol_routes[i].h;
    return NULL;
}
