/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * responses_out.c - the Response object and its output items (reasoning,
 * the message, the function calls), the response kept, and the events of
 * the stream: response.created, then every item added, its text told piece
 * by piece and done, then response.completed (or incomplete, or failed).
 */
#include "chat_out.h"
#include "responses.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* s without the white space at either end, as a new yyjson string */
static yyjson_mut_val *trimmed(yyjson_mut_doc *d, const char *s, size_t n)
{
    while (n && (unsigned char)s[0] <= ' ') {
        s++;
        n--;
    }
    while (n && (unsigned char)s[n - 1] <= ' ')
        n--;
    return yyjson_mut_strncpy(d, s, n);
}

static int blank(const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if ((unsigned char)s[i] > ' ')
            return 0;
    return 1;
}

static void ids(struct resp *x)
{
    if (!x->item_think[0])
        srv_new_id(x->item_think, sizeof(x->item_think), "rs_");
    if (!x->item_text[0])
        srv_new_id(x->item_text, sizeof(x->item_text), "msg_");
}

static yyjson_mut_val *reasoning_item(yyjson_mut_doc *d, struct resp *x,
                                      const struct srv_choice *ch, int done)
{
    yyjson_mut_val *it = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, it, "id", x->item_think);
    yyjson_mut_obj_add_str(d, it, "type", "reasoning");
    yyjson_mut_obj_add_arr(d, it, "summary");
    yyjson_mut_val *c = yyjson_mut_obj_add_arr(d, it, "content");
    if (done) {
        yyjson_mut_val *p = yyjson_mut_arr_add_obj(d, c);
        yyjson_mut_obj_add_str(d, p, "type", "reasoning_text");
        yyjson_mut_obj_add_val(
            d, p, "text", trimmed(d, ch->think ? ch->think : "", ch->n_think));
    }
    yyjson_mut_obj_add_str(d, it, "status", done ? "completed" : "in_progress");
    return it;
}

static yyjson_mut_val *text_part(yyjson_mut_doc *d, const struct srv_choice *ch,
                                 int done)
{
    yyjson_mut_val *p = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, p, "type", "output_text");
    if (done)
        yyjson_mut_obj_add_val(
            d, p, "text", trimmed(d, ch->text ? ch->text : "", ch->n_text));
    else
        yyjson_mut_obj_add_str(d, p, "text", "");
    yyjson_mut_obj_add_arr(d, p, "annotations");
    if (done && ch->n_lp)
        yyjson_mut_obj_add_val(
            d, p, "logprobs",
            yyjson_mut_obj_get(out_chat_lp(d, ch, 0, ch->n_lp), "content"));
    else
        yyjson_mut_obj_add_arr(d, p, "logprobs");
    return p;
}

static yyjson_mut_val *message_item(yyjson_mut_doc *d, struct resp *x,
                                    const struct srv_choice *ch, int done)
{
    yyjson_mut_val *it = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, it, "id", x->item_text);
    yyjson_mut_obj_add_str(d, it, "type", "message");
    yyjson_mut_obj_add_str(d, it, "role", "assistant");
    yyjson_mut_obj_add_str(d, it, "status", done ? "completed" : "in_progress");
    yyjson_mut_val *c = yyjson_mut_obj_add_arr(d, it, "content");
    if (done)
        yyjson_mut_arr_append(c, text_part(d, ch, 1));
    return it;
}

static yyjson_mut_val *call_item(yyjson_mut_doc *d, const struct srv_call *k,
                                 int done)
{
    yyjson_mut_val *it = yyjson_mut_obj(d);
    char id[64];
    snprintf(id, sizeof(id), "fc_%s", k->id + 5); /* after "call_" */
    yyjson_mut_obj_add_strcpy(d, it, "id", id);
    yyjson_mut_obj_add_str(d, it, "type", "function_call");
    yyjson_mut_obj_add_str(d, it, "call_id", k->id);
    yyjson_mut_obj_add_str(d, it, "name", k->name);
    yyjson_mut_obj_add_str(d, it, "arguments", done ? k->args : "");
    yyjson_mut_obj_add_str(d, it, "status", done ? "completed" : "in_progress");
    return it;
}

void resp_job_items(yyjson_mut_doc *d, yyjson_mut_val *arr, struct resp *x)
{
    const struct srv_choice *ch = &x->j->ch[0];
    ids(x);
    if (ch->n_think && !blank(ch->think, ch->n_think))
        yyjson_mut_arr_append(arr, reasoning_item(d, x, ch, 1));
    if (ch->n_text && !blank(ch->text, ch->n_text))
        yyjson_mut_arr_append(arr, message_item(d, x, ch, 1));
    for (int i = 0; i < ch->n_calls; i++)
        if (!resp_mcp_owns(x, ch->calls[i].name))
            yyjson_mut_arr_append(arr, call_item(d, &ch->calls[i], 1));
}

void resp_output(yyjson_mut_doc *d, yyjson_mut_val *arr, struct resp *x)
{
    /* the rounds over first (MCP servers' lists and calls among them) */
    if (x->done_items) {
        yyjson_doc *dd = yyjson_read(x->done_items, strlen(x->done_items), 0);
        size_t idx, max;
        yyjson_val *it;
        yyjson_arr_foreach(yyjson_doc_get_root(dd), idx, max, it)
            yyjson_mut_arr_append(arr, yyjson_val_mut_copy(d, it));
        yyjson_doc_free(dd);
    }
    if (!x->consumed)
        resp_job_items(d, arr, x);
}

void resp_round_reset(struct resp *x)
{
    x->item_think[0] = x->item_text[0] = 0;
    x->think_open = x->text_open = 0;
    x->told_think = x->told_text = 0;
    x->think_began = x->text_began = 0;
    x->calls_told = 0;
    x->round_checked = 0;
}

/* The status the job gives the response. */
static const char *status_of(const struct srv_job *j, const char **why)
{
    *why = NULL;
    if (j->done && j->cancel)
        return "cancelled"; /* running or still waiting in the queue */
    if (!j->started)
        return "queued";
    if (!j->done)
        return "in_progress";
    if (j->err != JANAS_LLM_OK || j->ch[0].finish == JANAS_LLM_FINISH_ERROR)
        return "failed";
    if (j->ch[0].finish == JANAS_LLM_FINISH_LENGTH ||
        j->ch[0].finish == JANAS_LLM_FINISH_CONTEXT) {
        *why = "max_output_tokens";
        return "incomplete";
    }
    return "completed";
}

char *resp_object(struct resp *x, const char *status, size_t *n)
{
    struct srv_job *j = x->j;
    const char *why = NULL;
    if (!status)
        status = status_of(j, &why);
    yyjson_doc *echo = yyjson_read(x->echo, strlen(x->echo), 0);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_str(d, o, "id", x->id);
    yyjson_mut_obj_add_str(d, o, "object", "response");
    yyjson_mut_obj_add_int(d, o, "created_at", x->created);
    yyjson_mut_obj_add_str(d, o, "status", status);
    int over = !strcmp(status, "completed") || !strcmp(status, "incomplete") ||
               !strcmp(status, "failed") || !strcmp(status, "cancelled");
    if (over)
        yyjson_mut_obj_add_int(d, o, "completed_at", (int64_t)time(NULL));
    else
        yyjson_mut_obj_add_null(d, o, "completed_at");
    if (!strcmp(status, "failed")) {
        yyjson_mut_val *e = yyjson_mut_obj_add_obj(d, o, "error");
        yyjson_mut_obj_add_str(d, e, "code", "server_error");
        yyjson_mut_obj_add_strcpy(
            d, e, "message", j->errmsg[0] ? j->errmsg : "generation failed");
    } else {
        yyjson_mut_obj_add_null(d, o, "error");
    }
    if (why) {
        yyjson_mut_val *w = yyjson_mut_obj_add_obj(d, o, "incomplete_details");
        yyjson_mut_obj_add_str(d, w, "reason", why);
    } else {
        yyjson_mut_obj_add_null(d, o, "incomplete_details");
    }
    size_t idx, max;
    yyjson_val *k, *v;
    yyjson_obj_foreach(yyjson_doc_get_root(echo), idx, max, k, v)
        yyjson_mut_obj_add(o, yyjson_val_mut_copy(d, k),
                           yyjson_val_mut_copy(d, v));
    yyjson_mut_val *out = yyjson_mut_obj_add_arr(d, o, "output");
    if (over || j->done)
        resp_output(d, out, x);
    const struct srv_choice *ch = &j->ch[0];
    if (over && ch->n_text && !blank(ch->text, ch->n_text))
        yyjson_mut_obj_add_val(d, o, "output_text",
                               trimmed(d, ch->text, ch->n_text));
    if (over) {
        yyjson_mut_val *u = yyjson_mut_obj_add_obj(d, o, "usage");
        /* with the rounds before this job, when MCP calls made several */
        uint64_t in = (uint64_t)j->prompt_tokens + x->prev_in;
        uint64_t out = (uint64_t)j->completion_tokens + x->prev_out;
        yyjson_mut_obj_add_uint(d, u, "input_tokens", in);
        yyjson_mut_val *id =
            yyjson_mut_obj_add_obj(d, u, "input_tokens_details");
        yyjson_mut_obj_add_uint(d, id, "cached_tokens",
                                (uint64_t)j->st.cached_tokens + x->prev_cached);
        yyjson_mut_obj_add_uint(d, u, "output_tokens", out);
        yyjson_mut_val *od =
            yyjson_mut_obj_add_obj(d, u, "output_tokens_details");
        yyjson_mut_obj_add_uint(d, od, "reasoning_tokens",
                                (uint64_t)j->reasoning_tokens + x->prev_reason);
        yyjson_mut_obj_add_uint(d, u, "total_tokens", in + out);
    } else {
        yyjson_mut_obj_add_null(d, o, "usage");
    }
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, n);
    yyjson_mut_doc_free(d);
    yyjson_doc_free(echo);
    return json;
}

void resp_keep(struct resp *x)
{
    size_t n;
    char *json = resp_object(x, NULL, &n);
    if (!json)
        return;
    if (x->store)
        srv_store_put(x->srv->store, RESP_KIND, x->id, x->created, json,
                      x->context);
    if (x->conv[0] && x->j->err == JANAS_LLM_OK) {
        /* the conversation gets the input, then what the model wrote */
        yyjson_doc *in = yyjson_read(x->input, strlen(x->input), 0);
        yyjson_doc *ob = yyjson_read(json, n, 0);
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *arr = yyjson_mut_arr(d);
        size_t idx, max;
        yyjson_val *it;
        yyjson_arr_foreach(yyjson_doc_get_root(in), idx, max, it)
            yyjson_mut_arr_append(arr, yyjson_val_mut_copy(d, it));
        yyjson_arr_foreach(yyjson_obj_get(yyjson_doc_get_root(ob), "output"),
                           idx, max, it)
            yyjson_mut_arr_append(arr, yyjson_val_mut_copy(d, it));
        size_t m;
        char *items =
            yyjson_mut_val_write(arr, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &m);
        if (items)
            conv_append(x->srv, x->conv, items);
        free(items);
        yyjson_mut_doc_free(d);
        yyjson_doc_free(in);
        yyjson_doc_free(ob);
    }
    free(json);
}

/* ---- the stream ---- */

/* An event: "event: type" and its data, the sequence number added. */
static int event(struct resp *x, yyjson_mut_doc *d, yyjson_mut_val *e,
                 const char *type)
{
    yyjson_mut_obj_add_str(d, e, "type", type);
    yyjson_mut_obj_add_int(d, e, "sequence_number", x->seq++);
    size_t n = 0;
    char *json =
        yyjson_mut_val_write(e, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    if (!json)
        return -1;
    size_t tl = strlen(type);
    char *p = realloc(x->pend, x->n_pend + tl + n + 17);
    if (!p) {
        free(json);
        return -1;
    }
    x->pend = p;
    x->n_pend += (size_t)sprintf(p + x->n_pend, "event: %s\ndata: ", type);
    memcpy(p + x->n_pend, json, n);
    x->n_pend += n;
    memcpy(p + x->n_pend, "\n\n", 2);
    x->n_pend += 2;
    free(json);
    return 0;
}

/* An event whose body is the response as it stands. */
static int with_response(struct resp *x, const char *type, const char *status)
{
    size_t n;
    char *json = resp_object(x, status, &n);
    yyjson_doc *r = json ? yyjson_read(json, n, 0) : NULL;
    free(json);
    if (!r)
        return -1;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *e = yyjson_mut_obj(d);
    yyjson_mut_obj_add_val(d, e, "response",
                           yyjson_val_mut_copy(d, yyjson_doc_get_root(r)));
    yyjson_doc_free(r);
    return event(x, d, e, type);
}

/* An event about an item: output_index, item_id, and more fields. */
static yyjson_mut_val *item_ev(yyjson_mut_doc *d, struct resp *x,
                               const char *item_id)
{
    yyjson_mut_val *e = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, e, "item_id", item_id);
    yyjson_mut_obj_add_int(d, e, "output_index", x->out_index);
    return e;
}

static int item_added(struct resp *x, yyjson_mut_doc *d, yyjson_mut_val *it,
                      const char *type)
{
    yyjson_mut_val *e = yyjson_mut_obj(d);
    yyjson_mut_obj_add_int(d, e, "output_index", x->out_index);
    yyjson_mut_obj_add_val(d, e, "item", it);
    return event(x, d, e, type);
}

/* The reasoning item, done: its text whole. */
static int close_think(struct resp *x, const struct srv_choice *ch)
{
    if (!x->think_open)
        return 0;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *e = item_ev(d, x, x->item_think);
    yyjson_mut_obj_add_int(d, e, "content_index", 0);
    yyjson_mut_obj_add_val(d, e, "text",
                           trimmed(d, ch->think ? ch->think : "", ch->n_think));
    int bad = event(x, d, e, "response.reasoning_text.done");
    d = yyjson_mut_doc_new(NULL);
    bad |= item_added(x, d, reasoning_item(d, x, ch, 1),
                      "response.output_item.done");
    x->think_open = 0;
    x->out_index++;
    return bad;
}

/* The message item, done. */
static int close_text(struct resp *x, const struct srv_choice *ch)
{
    if (!x->text_open)
        return 0;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *e = item_ev(d, x, x->item_text);
    yyjson_mut_obj_add_int(d, e, "content_index", 0);
    yyjson_mut_obj_add_val(d, e, "text",
                           trimmed(d, ch->text ? ch->text : "", ch->n_text));
    yyjson_mut_obj_add_arr(d, e, "logprobs");
    int bad = event(x, d, e, "response.output_text.done");
    d = yyjson_mut_doc_new(NULL);
    e = item_ev(d, x, x->item_text);
    yyjson_mut_obj_add_int(d, e, "content_index", 0);
    yyjson_mut_obj_add_val(d, e, "part", text_part(d, ch, 1));
    bad |= event(x, d, e, "response.content_part.done");
    d = yyjson_mut_doc_new(NULL);
    bad |= item_added(x, d, message_item(d, x, ch, 1),
                      "response.output_item.done");
    x->text_open = 0;
    x->out_index++;
    return bad;
}

/* An event of an item: its id and output_index, and one more field. */
static int item_event(struct resp *x, const char *id, const char *type,
                      const char *key, const char *value)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *e = item_ev(d, x, id);
    if (key)
        yyjson_mut_obj_add_strcpy(d, e, key, value ? value : "");
    return event(x, d, e, type);
}

/* The items of the rounds over that the stream has not told: the lists of
   the MCP servers' tools, the calls run, the requests for approval. */
static int tell_items(struct resp *x)
{
    if (!x->done_items)
        return 0;
    yyjson_doc *dd = yyjson_read(x->done_items, strlen(x->done_items), 0);
    yyjson_val *all = yyjson_doc_get_root(dd);
    int bad = 0;
    for (size_t i = (size_t)x->items_told; i < yyjson_arr_size(all); i++) {
        yyjson_val *it = yyjson_arr_get(all, i);
        const char *type = yyjson_get_str(yyjson_obj_get(it, "type"));
        const char *id = yyjson_get_str(yyjson_obj_get(it, "id"));
        if (!type || !id)
            continue;
        int call = !strcmp(type, "mcp_call"),
            list = !strcmp(type, "mcp_list_tools");
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *first = yyjson_val_mut_copy(d, it);
        if (call) { /* as it was when it began */
            yyjson_mut_obj_remove_key(first, "status");
            yyjson_mut_obj_add_str(d, first, "status", "in_progress");
            yyjson_mut_obj_remove_key(first, "output");
            yyjson_mut_obj_add_null(d, first, "output");
            yyjson_mut_obj_remove_key(first, "error");
            yyjson_mut_obj_add_null(d, first, "error");
        }
        bad |= item_added(x, d, first, "response.output_item.added");
        if (list) {
            bad |= item_event(x, id, "response.mcp_list_tools.in_progress",
                              NULL, NULL);
            bad |= item_event(x, id, "response.mcp_list_tools.completed", NULL,
                              NULL);
        } else if (call) {
            const char *a = yyjson_get_str(yyjson_obj_get(it, "arguments"));
            const char *st = yyjson_get_str(yyjson_obj_get(it, "status"));
            bad |=
                item_event(x, id, "response.mcp_call.in_progress", NULL, NULL);
            bad |= item_event(x, id, "response.mcp_call_arguments.delta",
                              "delta", a);
            bad |= item_event(x, id, "response.mcp_call_arguments.done",
                              "arguments", a);
            bad |= item_event(x, id,
                              st && !strcmp(st, "failed")
                                  ? "response.mcp_call.failed"
                                  : "response.mcp_call.completed",
                              NULL, NULL);
        }
        d = yyjson_mut_doc_new(NULL);
        bad |= item_added(x, d, yyjson_val_mut_copy(d, it),
                          "response.output_item.done");
        x->out_index++;
    }
    x->items_told = (int)yyjson_arr_size(all);
    yyjson_doc_free(dd);
    return bad;
}

/* The events the job has made possible; 1 once the stream is complete, 2
   when the job is over and its MCP calls are to be run first. Under
   j->mu. */
static int produce(struct resp *x)
{
    struct srv_job *j = x->j;
    const struct srv_choice *ch = &j->ch[0];
    int bad = 0;
    ids(x);
    if (!x->started) {
        bad |= with_response(x, "response.created", NULL);
        x->started = 1;
    }
    if (!j->started)
        return bad ? -1 : 0;
    if (x->started == 1) {
        bad |= with_response(x, "response.in_progress", "in_progress");
        x->started = 2;
    }
    bad |= tell_items(x);
    if (ch->n_think > x->told_think) {
        size_t n = out_skip_ws(ch->think, ch->n_think, &x->told_think,
                               &x->think_began);
        if (n && !x->think_open) {
            yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
            bad |= item_added(x, d, reasoning_item(d, x, ch, 0),
                              "response.output_item.added");
            x->think_open = 1;
        }
        if (n) {
            yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
            yyjson_mut_val *e = item_ev(d, x, x->item_think);
            yyjson_mut_obj_add_int(d, e, "content_index", 0);
            yyjson_mut_obj_add_strncpy(d, e, "delta", ch->think + x->told_think,
                                       n);
            bad |= event(x, d, e, "response.reasoning_text.delta");
        }
        x->told_think = ch->n_think;
    }
    size_t vis = srv_job_visible(j, 0);
    if (vis > x->told_text) {
        size_t n = out_skip_ws(ch->text, vis, &x->told_text, &x->text_began);
        if (n && !blank(ch->text + x->told_text, n)) {
            bad |= close_think(x, ch);
            if (!x->text_open) {
                yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
                bad |= item_added(x, d, message_item(d, x, ch, 0),
                                  "response.output_item.added");
                d = yyjson_mut_doc_new(NULL);
                yyjson_mut_val *e = item_ev(d, x, x->item_text);
                yyjson_mut_obj_add_int(d, e, "content_index", 0);
                yyjson_mut_obj_add_val(d, e, "part", text_part(d, ch, 0));
                bad |= event(x, d, e, "response.content_part.added");
                x->text_open = 1;
            }
            yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
            yyjson_mut_val *e = item_ev(d, x, x->item_text);
            yyjson_mut_obj_add_int(d, e, "content_index", 0);
            yyjson_mut_obj_add_strncpy(d, e, "delta", ch->text + x->told_text,
                                       n);
            yyjson_mut_obj_add_arr(d, e, "logprobs");
            bad |= event(x, d, e, "response.output_text.delta");
            x->told_text = vis;
        }
    }
    while (x->calls_told < ch->n_calls) {
        const struct srv_call *k = &ch->calls[x->calls_told++];
        if (resp_mcp_owns(x, k->name))
            continue; /* an mcp_call, or a request for approval, later */
        bad |= close_think(x, ch) | close_text(x, ch);
        char fid[64];
        snprintf(fid, sizeof(fid), "fc_%s", k->id + 5);
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        bad |=
            item_added(x, d, call_item(d, k, 0), "response.output_item.added");
        d = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *e = item_ev(d, x, fid);
        yyjson_mut_obj_add_str(d, e, "delta", k->args);
        bad |= event(x, d, e, "response.function_call_arguments.delta");
        d = yyjson_mut_doc_new(NULL);
        e = item_ev(d, x, fid);
        yyjson_mut_obj_add_str(d, e, "arguments", k->args);
        bad |= event(x, d, e, "response.function_call_arguments.done");
        d = yyjson_mut_doc_new(NULL);
        bad |=
            item_added(x, d, call_item(d, k, 1), "response.output_item.done");
        x->out_index++;
    }
    if (!j->done)
        return bad ? -1 : 0;
    bad |= close_think(x, ch) | close_text(x, ch);
    if (x->mcp && !x->round_checked)
        return bad ? -1 : 2;
    bad |= tell_items(x);
    const char *why;
    const char *st = status_of(j, &why);
    bad |= with_response(x,
                         !strcmp(st, "completed")    ? "response.completed"
                         : !strcmp(st, "incomplete") ? "response.incomplete"
                                                     : "response.failed",
                         NULL);
    x->done_sent = 1;
    return bad ? -1 : 1;
}

long resp_stream_next(void *ctx, char *buf, size_t cap)
{
    struct resp *x = ctx;
    if (x->off == x->n_pend) {
        x->off = x->n_pend = 0;
        if (x->done_sent)
            return -1;
        struct srv_job *j = x->j;
        pthread_mutex_lock(&j->mu);
        int end = produce(x);
        if (end == 2) {
            /* the round is over: its MCP calls run outside the job's lock,
               and a next round, if any, is the job the stream follows */
            pthread_mutex_unlock(&j->mu);
            x->round_checked = 1;
            resp_mcp_round(x);
            j = x->j;
            pthread_mutex_lock(&j->mu);
            end = produce(x);
        }
        if (end == 0 && x->n_pend == 0) {
            struct timespec t;
            clock_gettime(CLOCK_REALTIME, &t);
            t.tv_sec += 5;
            int rc = pthread_cond_timedwait(&j->cv, &j->mu, &t);
            end = produce(x);
            if (x->n_pend == 0 && rc == ETIMEDOUT) {
                char *p = realloc(x->pend, 16);
                if (p) {
                    x->pend = p;
                    memcpy(p, ": keep-alive\n\n", 14);
                    x->n_pend = 14;
                }
            }
        }
        if (end < 0)
            x->done_sent = 1;
        pthread_mutex_unlock(&j->mu);
        if (end == 1)
            resp_keep(x);
        if (x->n_pend == 0)
            return x->done_sent ? -1 : 0;
    }
    size_t n = x->n_pend - x->off < cap ? x->n_pend - x->off : cap;
    memcpy(buf, x->pend + x->off, n);
    x->off += n;
    return (long)n;
}

void resp_stream_done(void *ctx)
{
    resp_free(ctx);
}
