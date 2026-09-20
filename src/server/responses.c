/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * responses.c - OpenAI's Responses API: POST /responses (whole, streamed,
 * or in the background, found later by its id), GET and DELETE of a
 * response kept, the cancelling of one in the background, its input items,
 * the tokens an input would take (POST /responses/input_tokens), and the
 * compaction of a conversation into a summary (POST /responses/compact).
 * The request is read in responses_in.c, the answer made in
 * responses_out.c.
 */
#include "responses.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void resp_free(struct resp *x)
{
    if (!x)
        return;
    if (x->j) {
        pthread_mutex_lock(&x->j->mu);
        if (!x->j->done && !x->background)
            x->j->cancel = 1; /* no one reads it any more */
        pthread_mutex_unlock(&x->j->mu);
        srv_job_release(x->j);
    }
    resp_mcp_free(x->mcp);
    free(x->done_items);
    free(x->echo);
    free(x->context);
    free(x->input);
    free(x->pend);
    free(x);
}

static struct resp *resp_new(struct srv_req *r)
{
    struct resp *x = calloc(1, sizeof(*x));
    if (!x)
        return NULL;
    x->srv = r->srv;
    x->j = srv_job_new();
    if (!x->j) {
        free(x);
        return NULL;
    }
    x->created = (int64_t)time(NULL);
    srv_new_id(x->id, sizeof(x->id), "resp_");
    return x;
}

static void wait_done(struct srv_job *j)
{
    pthread_mutex_lock(&j->mu);
    while (!j->done)
        pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);
}

/* ---- the responses in the background, and their cancelling ---- */

struct bg {
    struct resp *x;
    struct bg *next;
};
static pthread_mutex_t bg_mu = PTHREAD_MUTEX_INITIALIZER;
static struct bg *bg_list;

static void *bg_main(void *arg)
{
    struct resp *x = arg;
    /* kept as in progress once the engine takes it */
    struct srv_job *j = x->j;
    pthread_mutex_lock(&j->mu);
    while (!j->started && !j->done)
        pthread_cond_wait(&j->cv, &j->mu);
    int running = !j->done;
    pthread_mutex_unlock(&j->mu);
    if (running) {
        size_t n;
        pthread_mutex_lock(&j->mu);
        char *json = resp_object(x, "in_progress", &n);
        pthread_mutex_unlock(&j->mu);
        if (json)
            srv_store_put(x->srv->store, RESP_KIND, x->id, x->created, json,
                          x->context);
        free(json);
    }
    wait_done(x->j);
    while (resp_mcp_round(x) == 1) /* MCP calls run, and the model again */
        wait_done(x->j);
    resp_keep(x);
    pthread_mutex_lock(&bg_mu);
    for (struct bg **p = &bg_list; *p; p = &(*p)->next)
        if ((*p)->x == x) {
            struct bg *b = *p;
            *p = b->next;
            free(b);
            break;
        }
    pthread_mutex_unlock(&bg_mu);
    resp_free(x);
    return NULL;
}

int resp_adopt(struct resp *x, struct srv_job *nj)
{
    int rc = -1;
    pthread_mutex_lock(&bg_mu);
    if (!x->cancelled && srv_engine_submit(x->srv->engine, nj) == 0) {
        x->j = nj;
        rc = 0;
    }
    pthread_mutex_unlock(&bg_mu);
    return rc;
}

/* The response goes on without its client: kept now as queued, kept again
   when it is over. 0, or -1 (then x is still the caller's). */
static int background(struct resp *x)
{
    struct bg *b = malloc(sizeof(*b));
    size_t n;
    pthread_mutex_lock(&x->j->mu);
    char *json = resp_object(x, "queued", &n);
    pthread_mutex_unlock(&x->j->mu);
    if (!b || !json ||
        srv_store_put(x->srv->store, RESP_KIND, x->id, x->created, json,
                      x->context) != 0) {
        free(b);
        free(json);
        return -1;
    }
    free(json);
    pthread_mutex_lock(&bg_mu);
    b->x = x;
    b->next = bg_list;
    bg_list = b;
    pthread_t th;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    int rc = pthread_create(&th, &at, bg_main, x);
    pthread_attr_destroy(&at);
    if (rc != 0) {
        bg_list = b->next;
        free(b);
    }
    pthread_mutex_unlock(&bg_mu);
    return rc ? -1 : 0;
}

void srv_responses_shutdown(void)
{
    /* the engine has stopped, so every job is over: the threads only keep
       their responses and go. Waited for, since the store goes next. */
    for (int i = 0; i < 500; i++) {
        pthread_mutex_lock(&bg_mu);
        int left = bg_list != NULL;
        pthread_mutex_unlock(&bg_mu);
        if (!left)
            return;
        struct timespec t = {0, 10 * 1000 * 1000};
        nanosleep(&t, NULL);
    }
}

/*
 * A response kept as queued or in progress by a server that was then
 * killed will never be done: said so, as failed. A server that stops
 * normally waits for them (srv_responses_shutdown). Returns how many.
 */
int srv_responses_recover(struct srv_store *st)
{
    char **ids;
    size_t n = srv_store_list(st, RESP_KIND, &ids);
    int fixed = 0;
    for (size_t i = 0; i < n; i++) {
        char *json = NULL;
        if (srv_store_get(st, RESP_KIND, ids[i], &json, NULL) != 0)
            continue;
        yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
        free(json);
        const char *status =
            yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(doc), "status"));
        yyjson_mut_doc *d = status && (!strcmp(status, "queued") ||
                                       !strcmp(status, "in_progress"))
                                ? yyjson_doc_mut_copy(doc, NULL)
                                : NULL;
        yyjson_doc_free(doc);
        if (!d)
            continue;
        yyjson_mut_val *o = yyjson_mut_doc_get_root(d);
        yyjson_mut_set_str(yyjson_mut_obj_get(o, "status"), "failed");
        yyjson_mut_obj_remove_key(o, "completed_at");
        yyjson_mut_obj_add_int(d, o, "completed_at", (int64_t)time(NULL));
        yyjson_mut_obj_remove_key(o, "error");
        yyjson_mut_val *e = yyjson_mut_obj_add_obj(d, o, "error");
        yyjson_mut_obj_add_str(d, e, "code", "server_error");
        yyjson_mut_obj_add_str(d, e, "message",
                               "the server stopped before the response was "
                               "done");
        char *out = yyjson_mut_write(d, YYJSON_WRITE_NOFLAG, NULL);
        yyjson_mut_doc_free(d);
        if (out && srv_store_update(st, RESP_KIND, ids[i], out, NULL) == 0)
            fixed++;
        free(out);
    }
    srv_store_list_free(ids, n);
    return fixed;
}

/* ---- the operations ---- */

static int no_response(struct srv_req *r)
{
    return srv_reply_error(r, 404, "invalid_request_error", "not_found",
                           "No response has the id '%s'.", r->param[0]);
}

static int engine_error(struct srv_req *r, struct srv_job *j)
{
    if (j->err == JANAS_LLM_ELIMIT)
        return srv_reply_limit(r, j);
    if (j->err == JANAS_LLM_EFULL)
        return srv_reply_error(r, 400, "invalid_request_error",
                               "context_length_exceeded", "%s", j->errmsg);
    if (j->err == JANAS_LLM_EINVAL || j->err == JANAS_LLM_EMODEL)
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "%s", j->errmsg);
    return srv_reply_error(r, 500, "server_error", NULL, "%s",
                           j->errmsg[0] ? j->errmsg : "generation failed");
}

static int queue_full(struct srv_req *r)
{
    return srv_reply_error(r, 429, "rate_limit_exceeded", NULL,
                           "The server is busy: %u requests are waiting. Try "
                           "again shortly.",
                           r->srv->cfg.max_queue);
}

int srv_responses_create(struct srv_req *r)
{
    struct resp *x = resp_new(r);
    int ret = 0;
    if (!x)
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    if (resp_read(r, x, &ret) != 0) {
        resp_free(x);
        return ret;
    }
    if (srv_engine_submit(r->srv->engine, x->j) != 0) {
        resp_free(x);
        return queue_full(r);
    }
    if (x->background) {
        size_t n;
        pthread_mutex_lock(&x->j->mu);
        char *json = resp_object(x, "queued", &n);
        pthread_mutex_unlock(&x->j->mu);
        if (!json || background(x) != 0) {
            free(json);
            resp_free(x);
            return srv_reply_error(r, 500, "server_error", NULL,
                                   "out of memory");
        }
        return srv_reply_json(r, 200, json, n);
    }
    if (x->stream) /* the stream holds x from here on */
        return srv_reply_stream(r, resp_stream_next, resp_stream_done, x);
    wait_done(x->j);
    while (resp_mcp_round(x) == 1) /* MCP calls run, and the model again */
        wait_done(x->j);
    if (x->j->err != JANAS_LLM_OK) {
        ret = engine_error(r, x->j);
        resp_free(x);
        return ret;
    }
    size_t n;
    char *json = resp_object(x, NULL, &n);
    resp_keep(x);
    resp_free(x);
    return srv_reply_json(r, 200, json, n);
}

int srv_responses_get(struct srv_req *r)
{
    char *json = NULL;
    int rc = srv_store_get(r->srv->store, RESP_KIND, r->param[0], &json, NULL);
    if (rc == -1)
        return no_response(r);
    if (rc != 0)
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    return srv_reply_json(r, 200, json, strlen(json));
}

int srv_responses_delete(struct srv_req *r)
{
    if (srv_store_del(r->srv->store, RESP_KIND, r->param[0]) != 0)
        return no_response(r);
    char out[160];
    int n = snprintf(out, sizeof(out),
                     "{\"id\":\"%s\",\"object\":\"response.deleted\","
                     "\"deleted\":true}",
                     r->param[0]);
    return srv_reply_json(r, 200, strndup(out, (size_t)n), (size_t)n);
}

int srv_responses_cancel(struct srv_req *r)
{
    struct srv_job *j = NULL;
    pthread_mutex_lock(&bg_mu);
    for (struct bg *b = bg_list; b; b = b->next)
        if (!strcmp(b->x->id, r->param[0])) {
            /* the job running, and no round after it; a tool call running
               between two rounds is stopped too */
            b->x->cancelled = 1;
            resp_mcp_cancel(b->x->mcp);
            j = b->x->j;
            pthread_mutex_lock(&j->mu);
            j->cancel = 1;
            j->refs++; /* ours, while we wait */
            pthread_mutex_unlock(&j->mu);
            break;
        }
    pthread_mutex_unlock(&bg_mu);
    char *json = NULL;
    if (!j) { /* over already, or never in the background */
        int rc =
            srv_store_get(r->srv->store, RESP_KIND, r->param[0], &json, NULL);
        free(json);
        if (rc == -1)
            return no_response(r);
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "Only a response in the background that is "
                               "not over can be cancelled.");
    }
    wait_done(j);
    srv_job_release(j);
    /* the background thread keeps it cancelled; wait for that */
    for (int i = 0; i < 200; i++) {
        free(json);
        json = NULL;
        srv_store_get(r->srv->store, RESP_KIND, r->param[0], &json, NULL);
        if (json && !strstr(json, "\"status\":\"queued\"") &&
            !strstr(json, "\"status\":\"in_progress\""))
            break; /* kept as it ended: cancelled, or over before */
        struct timespec t = {0, 10 * 1000 * 1000};
        nanosleep(&t, NULL);
    }
    if (!json)
        return no_response(r);
    return srv_reply_json(r, 200, json, strlen(json));
}

int srv_responses_input_items(struct srv_req *r)
{
    char *items = NULL;
    int rc = srv_store_get(r->srv->store, RESP_KIND, r->param[0], NULL, &items);
    if (rc == -1)
        return no_response(r);
    if (rc != 0)
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    const char *l = srv_query(r, "limit"), *o = srv_query(r, "order");
    const char *after = srv_query(r, "after");
    long limit = l ? strtol(l, NULL, 10) : 20;
    if (limit < 1 || limit > 100)
        limit = 20;
    int desc = o && !strcmp(o, "desc");
    yyjson_doc *doc = items ? yyjson_read(items, strlen(items), 0) : NULL;
    free(items);
    yyjson_val *arr = yyjson_doc_get_root(doc);
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
    size_t len;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &len);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(doc);
    return srv_reply_json(r, 200, json, len);
}

int srv_responses_input_tokens(struct srv_req *r)
{
    struct resp *x = resp_new(r);
    int ret = 0;
    if (!x)
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    if (resp_read(r, x, &ret) != 0) {
        resp_free(x);
        return ret;
    }
    x->j->kind = JOB_COUNT;
    if (srv_engine_submit(r->srv->engine, x->j) != 0) {
        resp_free(x);
        return queue_full(r);
    }
    wait_done(x->j);
    if (x->j->err != JANAS_LLM_OK) {
        ret = engine_error(r, x->j);
        resp_free(x);
        return ret;
    }
    char out[128];
    int n = snprintf(out, sizeof(out),
                     "{\"object\":\"response.input_tokens\","
                     "\"input_tokens\":%u}",
                     x->j->prompt_tokens);
    resp_free(x);
    return srv_reply_json(r, 200, strndup(out, (size_t)n), (size_t)n);
}

static const char *const COMPACT_ASK =
    "Summarize the conversation so far so that it can go on from the "
    "summary alone: keep the facts, the names, the decisions, the open "
    "questions and anything asked to be remembered. Write only the summary.";

int srv_responses_compact(struct srv_req *r)
{
    struct resp *x = resp_new(r);
    int ret = 0;
    if (!x)
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    if (resp_read(r, x, &ret) != 0) {
        resp_free(x);
        return ret;
    }
    /* the context, then the request for a summary; no tools, free text */
    struct srv_job *j = x->j;
    struct srv_msg *t = realloc(j->msg, (size_t)(j->n_msg + 1) * sizeof(*t));
    char *ask = strdup(COMPACT_ASK);
    if (!t || !ask) {
        free(ask);
        if (t)
            j->msg = t;
        resp_free(x);
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    }
    j->msg = t;
    j->msg[j->n_msg++] = (struct srv_msg){
        .role = JANAS_LLM_ROLE_USER, .text = ask, .n = strlen(ask)};
    free(j->tools);
    j->tools = NULL;
    j->n_tools = 0;
    j->format = JANAS_LLM_FORMAT_TEXT;
    j->p.thinking = 0;
    if (srv_engine_submit(r->srv->engine, j) != 0) {
        resp_free(x);
        return queue_full(r);
    }
    wait_done(j);
    if (j->err != JANAS_LLM_OK) {
        ret = engine_error(r, j);
        resp_free(x);
        return ret;
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_str(d, o, "id", x->id);
    yyjson_mut_obj_add_str(d, o, "object", "response.compaction");
    yyjson_mut_val *out = yyjson_mut_obj_add_arr(d, o, "output");
    yyjson_mut_val *it = yyjson_mut_arr_add_obj(d, out);
    char cid[64];
    srv_new_id(cid, sizeof(cid), "cmp_");
    yyjson_mut_obj_add_str(d, it, "type", "compaction");
    yyjson_mut_obj_add_str(d, it, "id", cid);
    /* not encrypted: the summary itself, which this server reads back */
    yyjson_mut_obj_add_strncpy(d, it, "encrypted_content",
                               j->ch[0].text ? j->ch[0].text : "",
                               j->ch[0].n_text);
    yyjson_mut_obj_add_int(d, o, "created_at", x->created);
    yyjson_mut_val *u = yyjson_mut_obj_add_obj(d, o, "usage");
    yyjson_mut_obj_add_uint(d, u, "input_tokens", j->prompt_tokens);
    yyjson_mut_val *id = yyjson_mut_obj_add_obj(d, u, "input_tokens_details");
    yyjson_mut_obj_add_uint(d, id, "cached_tokens", j->st.cached_tokens);
    yyjson_mut_obj_add_uint(d, u, "output_tokens", j->completion_tokens);
    yyjson_mut_val *od = yyjson_mut_obj_add_obj(d, u, "output_tokens_details");
    yyjson_mut_obj_add_uint(d, od, "reasoning_tokens", j->reasoning_tokens);
    yyjson_mut_obj_add_uint(d, u, "total_tokens",
                            (uint64_t)j->prompt_tokens + j->completion_tokens);
    size_t n;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    resp_free(x);
    return srv_reply_json(r, 200, json, n);
}
