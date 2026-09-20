/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * stored.c - the chat completions a client asked the server to keep
 * (store: true): listed, read, their metadata changed, deleted, and the
 * messages of the request that made them. They are kept in the store
 * (store.c), which outlives the server unless it runs with --no-store.
 */
#include "server.h"

#include <microhttpd.h>
#include <stdlib.h>
#include <string.h>
#include <yyjson.h>

#define KIND "chat.completion"

/* A page of a list: limit (default 20, 1..100) and order. */
static int paging(struct srv_req *r, long *limit, int *desc)
{
    const char *l = srv_query(r, "limit"), *o = srv_query(r, "order");
    *limit = 20;
    if (l) {
        char *end;
        *limit = strtol(l, &end, 10);
        if (*end || *limit < 1 || *limit > 100)
            return srv_reply_error(r, 400, "invalid_request_error",
                                   "invalid_value",
                                   "'limit' is a number from 1 to 100.");
    }
    *desc = o && strcmp(o, "desc") == 0;
    if (o && !*desc && strcmp(o, "asc") != 0)
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "'order' is asc or desc.");
    return -1; /* no answer given: go on */
}

struct meta_filter {
    char key[8][80];
    const char *val[8];
    int n, bad;
};

static enum MHD_Result meta_arg(void *cls, enum MHD_ValueKind kind,
                                const char *key, const char *value)
{
    (void)kind;
    struct meta_filter *f = cls;
    size_t n = strlen(key);
    if (strncmp(key, "metadata[", 9) != 0 || n < 11 || key[n - 1] != ']')
        return MHD_YES;
    if (f->n == 8 || n - 10 >= sizeof(f->key[0])) {
        f->bad = 1;
        return MHD_YES;
    }
    memcpy(f->key[f->n], key + 9, n - 10);
    f->key[f->n][n - 10] = 0;
    f->val[f->n++] = value ? value : "";
    return MHD_YES;
}

/* Whether a stored completion passes the model and metadata filters. */
static int passes(yyjson_val *doc, const char *model,
                  const struct meta_filter *f)
{
    if (model &&
        strcmp(model, yyjson_get_str(yyjson_obj_get(doc, "model"))
                          ? yyjson_get_str(yyjson_obj_get(doc, "model"))
                          : ""))
        return 0;
    yyjson_val *md = yyjson_obj_get(doc, "metadata");
    for (int i = 0; i < f->n; i++) {
        const char *v = yyjson_get_str(yyjson_obj_get(md, f->key[i]));
        if (!v || strcmp(v, f->val[i]) != 0)
            return 0;
    }
    return 1;
}

static int reply_doc(struct srv_req *r, yyjson_mut_doc *d)
{
    size_t n = 0;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, n);
}

int srv_stored_list(struct srv_req *r)
{
    long limit;
    int desc, rc;
    if ((rc = paging(r, &limit, &desc)) != -1)
        return rc;
    struct meta_filter f = {0};
    MHD_get_connection_values(r->conn, MHD_GET_ARGUMENT_KIND, meta_arg, &f);
    if (f.bad)
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "At most 8 metadata filters, of short keys.");
    const char *model = srv_query(r, "model"), *after = srv_query(r, "after");
    char **ids;
    size_t n = srv_store_list(r->srv->store, KIND, &ids);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, root);
    yyjson_mut_obj_add_str(d, root, "object", "list");
    yyjson_mut_val *data = yyjson_mut_obj_add_arr(d, root, "data");
    int started = !after, more = 0;
    long got = 0;
    const char *first = NULL, *last = NULL;
    for (size_t q = 0; q < n; q++) {
        const char *id = ids[desc ? n - 1 - q : q];
        if (!started) {
            started = strcmp(id, after) == 0;
            continue;
        }
        char *json = NULL;
        if (srv_store_get(r->srv->store, KIND, id, &json, NULL) != 0)
            continue;
        yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
        free(json);
        yyjson_val *v = yyjson_doc_get_root(doc);
        if (v && passes(v, model, &f)) {
            if (got == limit) {
                more = 1;
            } else {
                yyjson_mut_arr_append(data, yyjson_val_mut_copy(d, v));
                first = first ? first : id;
                last = id;
                got++;
            }
        }
        yyjson_doc_free(doc);
        if (more)
            break;
    }
    if (first) {
        yyjson_mut_obj_add_strcpy(d, root, "first_id", first);
        yyjson_mut_obj_add_strcpy(d, root, "last_id", last);
    } else {
        yyjson_mut_obj_add_null(d, root, "first_id");
        yyjson_mut_obj_add_null(d, root, "last_id");
    }
    yyjson_mut_obj_add_bool(d, root, "has_more", more);
    srv_store_list_free(ids, n);
    return reply_doc(r, d);
}

static int not_found(struct srv_req *r)
{
    return srv_reply_error(r, 404, "invalid_request_error", "not_found",
                           "No stored chat completion has the id '%s'.",
                           r->param[0]);
}

int srv_stored_get(struct srv_req *r)
{
    char *json = NULL;
    int rc = srv_store_get(r->srv->store, KIND, r->param[0], &json, NULL);
    if (rc == -1)
        return not_found(r);
    if (rc != 0)
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    return srv_reply_json(r, 200, json, strlen(json));
}

int srv_stored_update(struct srv_req *r)
{
    yyjson_doc *body = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *md = yyjson_obj_get(yyjson_doc_get_root(body), "metadata");
    int ok = md && (yyjson_is_obj(md) || yyjson_is_null(md)) &&
             yyjson_obj_size(md) <= 16;
    size_t idx, max;
    yyjson_val *k, *v;
    if (ok && yyjson_is_obj(md))
        yyjson_obj_foreach(md, idx, max, k, v)
        {
            ok &= yyjson_is_str(v);
        }
    if (!ok) {
        yyjson_doc_free(body);
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "The body is {\"metadata\": {...}}, at most "
                               "16 strings.");
    }
    char *json = NULL;
    int rc = srv_store_get(r->srv->store, KIND, r->param[0], &json, NULL);
    if (rc != 0) {
        yyjson_doc_free(body);
        return rc == -1 ? not_found(r)
                        : srv_reply_error(r, 500, "server_error", NULL,
                                          "out of memory");
    }
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    free(json);
    yyjson_mut_doc *d = yyjson_doc_mut_copy(doc, NULL);
    yyjson_doc_free(doc);
    yyjson_mut_val *root = yyjson_mut_doc_get_root(d);
    yyjson_mut_obj_remove_key(root, "metadata");
    yyjson_mut_obj_add_val(d, root, "metadata",
                           yyjson_is_obj(md) ? yyjson_val_mut_copy(d, md)
                                             : yyjson_mut_obj(d));
    yyjson_doc_free(body);
    size_t n = 0;
    char *out = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    if (!out ||
        srv_store_update(r->srv->store, KIND, r->param[0], out, NULL) != 0) {
        free(out);
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    }
    return srv_reply_json(r, 200, out, n);
}

int srv_stored_delete(struct srv_req *r)
{
    if (srv_store_del(r->srv->store, KIND, r->param[0]) != 0)
        return not_found(r);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, root);
    yyjson_mut_obj_add_str(d, root, "object", "chat.completion.deleted");
    yyjson_mut_obj_add_strcpy(d, root, "id", r->param[0]);
    yyjson_mut_obj_add_bool(d, root, "deleted", 1);
    return reply_doc(r, d);
}

int srv_stored_messages(struct srv_req *r)
{
    long limit;
    int desc, rc;
    if ((rc = paging(r, &limit, &desc)) != -1)
        return rc;
    char *msgs = NULL;
    rc = srv_store_get(r->srv->store, KIND, r->param[0], NULL, &msgs);
    if (rc != 0)
        return rc == -1 ? not_found(r)
                        : srv_reply_error(r, 500, "server_error", NULL,
                                          "out of memory");
    yyjson_doc *doc = msgs ? yyjson_read(msgs, strlen(msgs), 0) : NULL;
    free(msgs);
    yyjson_val *arr = yyjson_doc_get_root(doc);
    size_t n = yyjson_is_arr(arr) ? yyjson_arr_size(arr) : 0;
    const char *after = srv_query(r, "after");
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, root);
    yyjson_mut_obj_add_str(d, root, "object", "list");
    yyjson_mut_val *data = yyjson_mut_obj_add_arr(d, root, "data");
    int started = !after, more = 0;
    long got = 0;
    char first[320] = "", last[320] = "";
    for (size_t q = 0; q < n; q++) {
        size_t i = desc ? n - 1 - q : q;
        char id[320];
        snprintf(id, sizeof(id), "%s-%zu", r->param[0], i);
        if (!started) {
            started = strcmp(id, after) == 0;
            continue;
        }
        if (got == limit) {
            more = 1;
            break;
        }
        yyjson_mut_val *m = yyjson_val_mut_copy(d, yyjson_arr_get(arr, i));
        yyjson_mut_obj_add_strcpy(d, m, "id", id);
        yyjson_mut_val *content = yyjson_mut_obj_get(m, "content");
        if (content && yyjson_mut_is_arr(content)) {
            /* parts: their text as the content, the parts beside it */
            yyjson_mut_obj_remove_key(m, "content");
            yyjson_mut_obj_add_val(d, m, "content_parts", content);
            yyjson_mut_obj_add_null(d, m, "content");
        }
        yyjson_mut_arr_append(data, m);
        if (!first[0])
            snprintf(first, sizeof(first), "%s", id);
        snprintf(last, sizeof(last), "%s", id);
        got++;
    }
    if (first[0]) {
        yyjson_mut_obj_add_strcpy(d, root, "first_id", first);
        yyjson_mut_obj_add_strcpy(d, root, "last_id", last);
    } else {
        yyjson_mut_obj_add_null(d, root, "first_id");
        yyjson_mut_obj_add_null(d, root, "last_id");
    }
    yyjson_mut_obj_add_bool(d, root, "has_more", more);
    yyjson_doc_free(doc);
    return reply_doc(r, d);
}
