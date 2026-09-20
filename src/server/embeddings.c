/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * embeddings.c - POST /embeddings: the vectors of texts, from the embedding
 * model the server was started with (--embedding-model), as numbers or as
 * base64 of little-endian floats.
 */
#include "request.h"

#include <stdlib.h>
#include <string.h>

static const char B64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/* n bytes as base64 */
static char *base64(const unsigned char *p, size_t n)
{
    char *o = malloc(4 * ((n + 2) / 3) + 1), *w = o;
    if (!o)
        return NULL;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16;
        if (i + 1 < n)
            v |= (uint32_t)p[i + 1] << 8;
        if (i + 2 < n)
            v |= p[i + 2];
        *w++ = B64[v >> 18 & 63];
        *w++ = B64[v >> 12 & 63];
        *w++ = i + 1 < n ? B64[v >> 6 & 63] : '=';
        *w++ = i + 2 < n ? B64[v & 63] : '=';
    }
    *w = 0;
    return o;
}

/* One input: a string, or token ids. */
static int read_one(yyjson_val *v, struct srv_prompt *out)
{
    if (yyjson_is_str(v)) {
        out->n = yyjson_get_len(v);
        return (out->text = srv_dup_n(yyjson_get_str(v), out->n)) ? 0 : -2;
    }
    if (!yyjson_is_arr(v) || yyjson_arr_size(v) == 0)
        return -1;
    out->ids = malloc(yyjson_arr_size(v) * sizeof(int32_t));
    if (!out->ids)
        return -2;
    size_t idx, max;
    yyjson_val *t;
    yyjson_arr_foreach(v, idx, max, t)
    {
        if (!yyjson_is_int(t) || yyjson_get_sint(t) < 0 ||
            yyjson_get_sint(t) > INT32_MAX)
            return -1;
        out->ids[out->n_ids++] = (int32_t)yyjson_get_sint(t);
    }
    return 0;
}

int srv_embeddings_create(struct srv_req *r)
{
    if (!srv_engine_embed_dim(r->srv->engine))
        return srv_reply_error(r, 400, "invalid_request_error",
                               "model_not_found",
                               "This server has no embedding model: start it "
                               "with --embedding-model FILE (such as "
                               "Qwen3-Embedding-0.6B).");
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *in = yyjson_obj_get(root, "input");
    const char *fmt = yyjson_get_str(yyjson_obj_get(root, "encoding_format"));
    int b64 = fmt && !strcmp(fmt, "base64");
    double dims = 0;
    int bad = !yyjson_is_obj(root) || !in ||
              (fmt && !b64 && strcmp(fmt, "float")) ||
              srv_num(yyjson_obj_get(root, "dimensions"), 1, 1e6, &dims) < 0;
    /* many inputs: an array of strings, or of arrays of token ids */
    int many = yyjson_is_arr(in) && yyjson_arr_size(in) > 0 &&
               (yyjson_is_str(yyjson_arr_get_first(in)) ||
                yyjson_is_arr(yyjson_arr_get_first(in)));
    size_t n = many ? yyjson_arr_size(in) : 1;
    if (!bad && n > 2048)
        bad = 1;
    struct srv_job *j = bad ? NULL : srv_job_new();
    int rc = 0;
    if (j) {
        j->kind = JOB_EMBED;
        j->dims = (int32_t)dims;
        j->prompts = calloc(n, sizeof(*j->prompts));
        rc = j->prompts ? 0 : -2;
        if (!rc)
            j->n_prompts = (int)n;
        for (size_t i = 0; !rc && i < n; i++)
            rc = read_one(many ? yyjson_arr_get(in, i) : in, &j->prompts[i]);
        if (!rc && srv_job_choices(j, 1, 1) != 0)
            rc = -2;
    }
    yyjson_doc_free(doc);
    if (bad || rc == -1) {
        srv_job_release(j);
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "'input' is a string, token ids, or an array "
                               "of either (at most 2048); 'encoding_format' "
                               "float or base64; 'dimensions' 1 or more.");
    }
    if (!j || rc) {
        srv_job_release(j);
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    }
    j->n_choices = 0; /* nothing to generate */
    if (srv_engine_submit(r->srv->engine, j) != 0) {
        srv_job_release(j);
        return srv_reply_error(r, 429, "rate_limit_exceeded", NULL,
                               "The server is busy. Try again shortly.");
    }
    pthread_mutex_lock(&j->mu);
    while (!j->done)
        pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);
    if (j->err != JANAS_LLM_OK) {
        int ret =
            srv_reply_error(r, j->err == JANAS_LLM_EFULL ? 400 : 500,
                            j->err == JANAS_LLM_EFULL ? "invalid_request_error"
                                                      : "server_error",
                            NULL, "%s", j->errmsg);
        srv_job_release(j);
        return ret;
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_str(d, o, "object", "list");
    yyjson_mut_val *data = yyjson_mut_obj_add_arr(d, o, "data");
    for (int i = 0; i < j->n_prompts; i++) {
        const float *v = j->emb + (size_t)i * (size_t)j->emb_dim;
        yyjson_mut_val *e = yyjson_mut_arr_add_obj(d, data);
        yyjson_mut_obj_add_str(d, e, "object", "embedding");
        yyjson_mut_obj_add_int(d, e, "index", i);
        if (b64) {
            char *s = base64((const unsigned char *)v,
                             (size_t)j->emb_dim * sizeof(float));
            yyjson_mut_obj_add_strcpy(d, e, "embedding", s ? s : "");
            free(s);
        } else {
            yyjson_mut_val *a = yyjson_mut_obj_add_arr(d, e, "embedding");
            for (int32_t k = 0; k < j->emb_dim; k++)
                yyjson_mut_arr_add_real(d, a, v[k]);
        }
    }
    yyjson_mut_obj_add_str(d, o, "model", r->srv->cfg.embed_id);
    yyjson_mut_val *u = yyjson_mut_obj_add_obj(d, o, "usage");
    yyjson_mut_obj_add_uint(d, u, "prompt_tokens", j->prompt_tokens);
    yyjson_mut_obj_add_uint(d, u, "total_tokens", j->prompt_tokens);
    srv_job_release(j);
    size_t len;
    char *json = yyjson_mut_write(d, 0, &len);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, len);
}
