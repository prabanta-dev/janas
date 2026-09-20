/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * conversations.c - conversations kept by the server (OpenAI's
 * Conversations API): made, read, their metadata changed, deleted, and
 * their items listed, added, read and deleted. A response that names a
 * conversation reads its items as the history and adds its input and its
 * output to them (responses_out.c). Kept in the store (store.c).
 */
#include "responses.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* the items of a conversation are read, changed and written back whole */
static pthread_mutex_t conv_mu = PTHREAD_MUTEX_INITIALIZER;

static int reply_doc(struct srv_req *r, yyjson_mut_doc *d)
{
    size_t n = 0;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, n);
}

static int no_conv(struct srv_req *r)
{
    return srv_reply_error(r, 404, "invalid_request_error", "not_found",
                           "No conversation has the id '%s'.", r->param[0]);
}

static int oom(struct srv_req *r)
{
    return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
}

/* metadata of a request body: at most 16 strings; NULL absent, (void*)1
   invalid */
static yyjson_val *metadata(yyjson_val *root)
{
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    if (!md || yyjson_is_null(md))
        return NULL;
    if (!yyjson_is_obj(md) || yyjson_obj_size(md) > 16)
        return (yyjson_val *)1;
    size_t idx, max;
    yyjson_val *k, *v;
    yyjson_obj_foreach(md, idx, max, k, v)
    {
        if (!yyjson_is_str(v))
            return (yyjson_val *)1;
    }
    return md;
}

/* Items given by a client, each with an id, into arr; 0, or -1 with the
   reason in err. */
static int take_items(yyjson_mut_doc *d, yyjson_mut_val *arr, yyjson_val *in,
                      char *err, size_t el)
{
    if (!in || yyjson_is_null(in))
        return 0;
    if (!yyjson_is_arr(in) || yyjson_arr_size(in) > 20) {
        snprintf(err, el, "'items' is an array of at most 20 items");
        return -1;
    }
    size_t idx, max;
    yyjson_val *it;
    yyjson_arr_foreach(in, idx, max, it)
    {
        if (!yyjson_is_obj(it)) {
            snprintf(err, el, "items[%zu] is not an item", idx);
            return -1;
        }
        yyjson_mut_val *m = yyjson_val_mut_copy(d, it);
        resp_item_id(d, m);
        yyjson_mut_arr_append(arr, m);
    }
    return 0;
}

int conv_append(struct srv *srv, const char *conv, const char *items_json)
{
    pthread_mutex_lock(&conv_mu);
    char *items = NULL;
    int rc = srv_store_get(srv->store, CONV_KIND, conv, NULL, &items);
    yyjson_doc *a =
        rc == 0 && items ? yyjson_read(items, strlen(items), 0) : NULL;
    yyjson_doc *b = yyjson_read(items_json, strlen(items_json), 0);
    free(items);
    if (rc == 0) {
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *arr = yyjson_mut_arr(d);
        size_t idx, max;
        yyjson_val *it;
        yyjson_arr_foreach(yyjson_doc_get_root(a), idx, max, it)
            yyjson_mut_arr_append(arr, yyjson_val_mut_copy(d, it));
        yyjson_arr_foreach(yyjson_doc_get_root(b), idx, max, it)
            yyjson_mut_arr_append(arr, yyjson_val_mut_copy(d, it));
        size_t n;
        char *out =
            yyjson_mut_val_write(arr, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
        rc =
            out ? srv_store_update(srv->store, CONV_KIND, conv, NULL, out) : -1;
        free(out);
        yyjson_mut_doc_free(d);
    }
    yyjson_doc_free(a);
    yyjson_doc_free(b);
    pthread_mutex_unlock(&conv_mu);
    return rc == 0 ? 0 : -1;
}

char *conv_items(struct srv *srv, const char *conv)
{
    char *items = NULL;
    pthread_mutex_lock(&conv_mu);
    int rc = srv_store_get(srv->store, CONV_KIND, conv, NULL, &items);
    pthread_mutex_unlock(&conv_mu);
    if (rc != 0)
        return NULL;
    return items ? items : strdup("[]");
}

int srv_conv_create(struct srv_req *r)
{
    yyjson_doc *body = r->n_body ? yyjson_read(r->body, r->n_body, 0) : NULL;
    yyjson_val *root = yyjson_doc_get_root(body);
    yyjson_val *md = metadata(root);
    char err[160] = "";
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *items = yyjson_mut_arr(d);
    if ((r->n_body && !yyjson_is_obj(root)) || md == (yyjson_val *)1 ||
        take_items(d, items, yyjson_obj_get(root, "items"), err, sizeof(err)) !=
            0) {
        yyjson_mut_doc_free(d);
        yyjson_doc_free(body);
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "%s",
                               err[0] ? err
                                      : "The body is {\"metadata\": {...}, "
                                        "\"items\": [...]}.");
    }
    char id[64];
    srv_new_id(id, sizeof(id), "conv_");
    int64_t now = (int64_t)time(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_strcpy(d, o, "id", id);
    yyjson_mut_obj_add_str(d, o, "object", "conversation");
    yyjson_mut_obj_add_int(d, o, "created_at", now);
    yyjson_mut_obj_add_val(d, o, "metadata",
                           md ? yyjson_val_mut_copy(d, md) : yyjson_mut_obj(d));
    size_t n, ni;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    char *its =
        yyjson_mut_val_write(items, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &ni);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(body);
    if (!json || !its ||
        srv_store_put(r->srv->store, CONV_KIND, id, now, json, its) != 0) {
        free(json);
        free(its);
        return oom(r);
    }
    free(its);
    return srv_reply_json(r, 200, json, n);
}

int srv_conv_get(struct srv_req *r)
{
    char *json = NULL;
    int rc = srv_store_get(r->srv->store, CONV_KIND, r->param[0], &json, NULL);
    if (rc == -1)
        return no_conv(r);
    if (rc != 0)
        return oom(r);
    return srv_reply_json(r, 200, json, strlen(json));
}

int srv_conv_update(struct srv_req *r)
{
    yyjson_doc *body = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *md = metadata(yyjson_doc_get_root(body));
    if (!md || md == (yyjson_val *)1) {
        yyjson_doc_free(body);
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "The body is {\"metadata\": {...}}, at most "
                               "16 strings.");
    }
    char *json = NULL;
    int rc = srv_store_get(r->srv->store, CONV_KIND, r->param[0], &json, NULL);
    if (rc != 0) {
        yyjson_doc_free(body);
        return rc == -1 ? no_conv(r) : oom(r);
    }
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    free(json);
    yyjson_mut_doc *d = yyjson_doc_mut_copy(doc, NULL);
    yyjson_doc_free(doc);
    yyjson_mut_val *o = yyjson_mut_doc_get_root(d);
    yyjson_mut_obj_remove_key(o, "metadata");
    yyjson_mut_obj_add_val(d, o, "metadata", yyjson_val_mut_copy(d, md));
    yyjson_doc_free(body);
    size_t n;
    char *out = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    if (!out ||
        srv_store_update(r->srv->store, CONV_KIND, r->param[0], out, NULL)) {
        free(out);
        return oom(r);
    }
    return srv_reply_json(r, 200, out, n);
}

int srv_conv_delete(struct srv_req *r)
{
    pthread_mutex_lock(&conv_mu);
    int rc = srv_store_del(r->srv->store, CONV_KIND, r->param[0]);
    pthread_mutex_unlock(&conv_mu);
    if (rc != 0)
        return no_conv(r);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_strcpy(d, o, "id", r->param[0]);
    yyjson_mut_obj_add_str(d, o, "object", "conversation.deleted");
    yyjson_mut_obj_add_bool(d, o, "deleted", 1);
    return reply_doc(r, d);
}

/* A page of items as a list: after an id, up to limit, in order. */
static int item_list(struct srv_req *r, yyjson_val *arr, int desc_default)
{
    const char *l = srv_query(r, "limit"), *o = srv_query(r, "order");
    const char *after = srv_query(r, "after");
    long limit = l ? strtol(l, NULL, 10) : 20;
    if (limit < 1 || limit > 100)
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "'limit' is a number from 1 to 100.");
    int desc = o ? !strcmp(o, "desc") : desc_default;
    size_t n = yyjson_is_arr(arr) ? yyjson_arr_size(arr) : 0;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, root);
    yyjson_mut_obj_add_str(d, root, "object", "list");
    yyjson_mut_val *data = yyjson_mut_obj_add_arr(d, root, "data");
    int started = !after, more = 0;
    long got = 0;
    const char *first = NULL, *last = NULL;
    for (size_t q = 0; q < n; q++) {
        yyjson_val *it = yyjson_arr_get(arr, desc ? n - 1 - q : q);
        const char *id = yyjson_get_str(yyjson_obj_get(it, "id"));
        if (!started) {
            started = id && !strcmp(id, after);
            continue;
        }
        if (got == limit) {
            more = 1;
            break;
        }
        yyjson_mut_arr_append(data, yyjson_val_mut_copy(d, it));
        first = first ? first : id;
        last = id;
        got++;
    }
    if (first) {
        yyjson_mut_obj_add_strcpy(d, root, "first_id", first);
        yyjson_mut_obj_add_strcpy(d, root, "last_id", last);
    } else {
        yyjson_mut_obj_add_null(d, root, "first_id");
        yyjson_mut_obj_add_null(d, root, "last_id");
    }
    yyjson_mut_obj_add_bool(d, root, "has_more", more);
    return reply_doc(r, d);
}

int srv_conv_items_list(struct srv_req *r)
{
    char *items = conv_items(r->srv, r->param[0]);
    if (!items)
        return no_conv(r);
    yyjson_doc *doc = yyjson_read(items, strlen(items), 0);
    free(items);
    int ret = item_list(r, yyjson_doc_get_root(doc), 1);
    yyjson_doc_free(doc);
    return ret;
}

int srv_conv_items_create(struct srv_req *r)
{
    yyjson_doc *body = yyjson_read(r->body, r->n_body, 0);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *arr = yyjson_mut_arr(d);
    char err[160] = "";
    yyjson_val *in = yyjson_obj_get(yyjson_doc_get_root(body), "items");
    if (!in || take_items(d, arr, in, err, sizeof(err)) != 0) {
        yyjson_mut_doc_free(d);
        yyjson_doc_free(body);
        return srv_reply_error(
            r, 400, "invalid_request_error", "invalid_value", "%s",
            err[0] ? err : "The body is {\"items\": [...]}.");
    }
    yyjson_doc_free(body);
    size_t n;
    char *its =
        yyjson_mut_val_write(arr, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    int rc = its ? conv_append(r->srv, r->param[0], its) : -2;
    if (rc != 0) {
        free(its);
        yyjson_mut_doc_free(d);
        return rc == -1 ? no_conv(r) : oom(r);
    }
    yyjson_doc *added = yyjson_read(its, n, 0);
    free(its);
    yyjson_mut_doc_free(d);
    int ret = item_list(r, yyjson_doc_get_root(added), 0);
    yyjson_doc_free(added);
    return ret;
}

/* Item param[1] of conversation param[0]: its index, the items in *doc. */
static long find_item(struct srv_req *r, yyjson_doc **doc)
{
    char *items = conv_items(r->srv, r->param[0]);
    *doc = items ? yyjson_read(items, strlen(items), 0) : NULL;
    free(items);
    if (!*doc)
        return -2;
    size_t idx, max;
    yyjson_val *it;
    yyjson_arr_foreach(yyjson_doc_get_root(*doc), idx, max, it)
    {
        const char *id = yyjson_get_str(yyjson_obj_get(it, "id"));
        if (id && !strcmp(id, r->param[1]))
            return (long)idx;
    }
    return -1;
}

static int no_item(struct srv_req *r)
{
    return srv_reply_error(r, 404, "invalid_request_error", "not_found",
                           "The conversation '%s' has no item '%s'.",
                           r->param[0], r->param[1]);
}

int srv_conv_item_get(struct srv_req *r)
{
    yyjson_doc *doc;
    long i = find_item(r, &doc);
    if (i < 0) {
        yyjson_doc_free(doc);
        return i == -2 ? no_conv(r) : no_item(r);
    }
    size_t n;
    char *json =
        yyjson_val_write(yyjson_arr_get(yyjson_doc_get_root(doc), (size_t)i),
                         YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_doc_free(doc);
    return srv_reply_json(r, 200, json, n);
}

int srv_conv_item_delete(struct srv_req *r)
{
    pthread_mutex_lock(&conv_mu);
    yyjson_doc *doc;
    char *items = NULL;
    int rc = srv_store_get(r->srv->store, CONV_KIND, r->param[0], NULL, &items);
    doc = rc == 0 && items ? yyjson_read(items, strlen(items), 0) : NULL;
    free(items);
    long at = -1;
    size_t idx, max;
    yyjson_val *it;
    yyjson_arr_foreach(yyjson_doc_get_root(doc), idx, max, it)
    {
        const char *id = yyjson_get_str(yyjson_obj_get(it, "id"));
        if (id && !strcmp(id, r->param[1]))
            at = (long)idx;
    }
    if (rc == 0 && at >= 0) {
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *arr = yyjson_mut_arr(d);
        yyjson_arr_foreach(yyjson_doc_get_root(doc), idx, max, it)
        {
            if ((long)idx != at)
                yyjson_mut_arr_append(arr, yyjson_val_mut_copy(d, it));
        }
        size_t n;
        char *out =
            yyjson_mut_val_write(arr, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
        rc = out ? srv_store_update(r->srv->store, CONV_KIND, r->param[0], NULL,
                                    out)
                 : -2;
        free(out);
        yyjson_mut_doc_free(d);
    }
    yyjson_doc_free(doc);
    pthread_mutex_unlock(&conv_mu);
    if (rc == -1)
        return no_conv(r);
    if (at < 0)
        return no_item(r);
    if (rc != 0)
        return oom(r);
    return srv_conv_get(r); /* the conversation, as OpenAI answers */
}
