/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * anthropic.c - POST /v1/messages and /v1/messages/count_tokens, Anthropic's
 * Messages API, for Claude Code and the other clients made for it: the
 * request made OpenAI's (anthropic_req.c) and read by request.c into a job,
 * the reply written as Anthropic writes it - whole, its content blocks
 * (thinking, text, tool_use with its input as an object) and stop_reason,
 * or as server-sent events: message_start, each block's start, deltas and
 * stop, message_delta with the reason and the usage, message_stop; a ping
 * while the request waits or the model reads.
 *
 * Any model name is taken - a client asks for Anthropic's names - and the
 * answer says the one asked for. The reasoning has no signature of
 * Anthropic's: it is given an empty one, and is not read back.
 */
#include "anthropic.h"
#include "chat_out.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* An error as Anthropic gives it. */
static int an_error(struct srv_req *r, unsigned status, const char *type,
                    const char *fmt, ...)
    __attribute__((format(gnu_printf, 4, 5)));
static int an_error(struct srv_req *r, unsigned status, const char *type,
                    const char *fmt, ...)
{
    char msg[600];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_str(d, o, "type", "error");
    yyjson_mut_val *e = yyjson_mut_obj_add_obj(d, o, "error");
    yyjson_mut_obj_add_str(d, e, "type", type);
    yyjson_mut_obj_add_strcpy(d, e, "message", msg);
    size_t n = 0;
    char *json = yyjson_mut_write(d, 0, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, status, json, n);
}

/* A reply being made for Anthropic's client. */
struct an_out {
    struct srv_job *j;
    char id[48], model[256];
    size_t r_think, r_text;
    int r_calls, think_began, text_began;
    int started, block, open; /* blocks so far; the open one's kind */
    int sent_end;
    int hide_think; /* thinking's display omitted: not given back */
    char *pend;
    size_t n_pend, off;
};

enum { NONE, THINKING, TEXT };

static const char *stop_reason(const struct srv_job *j)
{
    const struct srv_choice *c = &j->ch[0];
    if (c->n_calls)
        return "tool_use";
    if (c->finish == JANAS_LLM_FINISH_LENGTH)
        return "max_tokens";
    if (c->by_stop)
        return "stop_sequence";
    return "end_turn";
}

static void usage(yyjson_mut_doc *d, yyjson_mut_val *o, const struct srv_job *j,
                  int start)
{
    const struct janas_llm_chat_stats *st = &j->st;
    yyjson_mut_val *u = yyjson_mut_obj_add_obj(d, o, "usage");
    uint32_t cached = st->cached_tokens <= st->prompt_tokens
                          ? st->cached_tokens
                          : st->prompt_tokens;
    yyjson_mut_obj_add_uint(d, u, "input_tokens", st->prompt_tokens - cached);
    yyjson_mut_obj_add_uint(d, u, "cache_creation_input_tokens", 0);
    yyjson_mut_obj_add_uint(d, u, "cache_read_input_tokens", cached);
    yyjson_mut_obj_add_uint(d, u, "output_tokens",
                            start ? 1 : j->completion_tokens);
}

/* The stop sequence the reply ended on, when it is known: the only one. */
static void add_stop_sequence(yyjson_mut_doc *d, yyjson_mut_val *o,
                              const struct srv_job *j)
{
    if (j->ch[0].by_stop && j->n_stop == 1)
        yyjson_mut_obj_add_strcpy(d, o, "stop_sequence", j->stop[0]);
    else
        yyjson_mut_obj_add_null(d, o, "stop_sequence");
}

static yyjson_mut_val *tool_use(yyjson_mut_doc *d, const struct srv_call *c,
                                int with_input)
{
    yyjson_mut_val *b = yyjson_mut_obj(d);
    yyjson_mut_obj_add_str(d, b, "type", "tool_use");
    yyjson_mut_obj_add_strcpy(d, b, "id", c->id);
    yyjson_mut_obj_add_strcpy(d, b, "name", c->name);
    yyjson_doc *ad = NULL;
    yyjson_val *av = NULL;
    if (with_input) {
        const char *a = c->args ? c->args : "{}";
        ad = yyjson_read(a, strlen(a), 0);
        av = yyjson_doc_get_root(ad);
    }
    yyjson_mut_obj_add_val(d, b, "input",
                           yyjson_is_obj(av) ? yyjson_val_mut_copy(d, av)
                                             : yyjson_mut_obj(d));
    yyjson_doc_free(ad);
    return b;
}

/* The message's head: id, type, role, model. */
static yyjson_mut_val *head(yyjson_mut_doc *d, const struct an_out *o)
{
    yyjson_mut_val *m = yyjson_mut_obj(d);
    yyjson_mut_obj_add_strcpy(d, m, "id", o->id);
    yyjson_mut_obj_add_str(d, m, "type", "message");
    yyjson_mut_obj_add_str(d, m, "role", "assistant");
    yyjson_mut_obj_add_strcpy(d, m, "model", o->model);
    return m;
}

static int failed(const struct srv_job *j)
{
    return j->err != JANAS_LLM_OK || j->ch[0].finish == JANAS_LLM_FINISH_ERROR;
}

static unsigned err_status(const struct srv_job *j)
{
    return j->err == JANAS_LLM_EFULL || j->err == JANAS_LLM_ELIMIT ||
                   j->err == JANAS_LLM_EINVAL || j->err == JANAS_LLM_EMODEL
               ? 400
               : 500;
}

static int reply_whole(struct srv_req *r, struct an_out *o)
{
    struct srv_job *j = o->j;
    pthread_mutex_lock(&j->mu);
    while (!j->done)
        pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);
    if (failed(j))
        return an_error(r, err_status(j),
                        err_status(j) == 400 ? "invalid_request_error"
                                             : "api_error",
                        "%s", j->errmsg[0] ? j->errmsg : "generation failed");
    const struct srv_choice *ch = &j->ch[0];
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *m = head(d, o);
    yyjson_mut_doc_set_root(d, m);
    yyjson_mut_val *content = yyjson_mut_obj_add_arr(d, m, "content");
    size_t ts = 0, xs = 0;
    int b1 = 0, b2 = 0;
    size_t thn = o->hide_think ? 0
                               : out_skip_ws(ch->think ? ch->think : "",
                                             ch->n_think, &ts, &b1);
    while (thn && (unsigned char)ch->think[ts + thn - 1] <= ' ')
        thn--;
    if (thn) {
        yyjson_mut_val *b = yyjson_mut_arr_add_obj(d, content);
        yyjson_mut_obj_add_str(d, b, "type", "thinking");
        yyjson_mut_obj_add_strncpy(d, b, "thinking", ch->think + ts, thn);
        yyjson_mut_obj_add_str(d, b, "signature", "");
    }
    const char *text = ch->text ? ch->text : "";
    size_t tn = out_skip_ws(text, ch->n_text, &xs, &b2);
    while (tn && ch->n_calls && (unsigned char)text[xs + tn - 1] <= ' ')
        tn--; /* the line breaks around the calls are not content */
    if (tn || !ch->n_calls) {
        yyjson_mut_val *b = yyjson_mut_arr_add_obj(d, content);
        yyjson_mut_obj_add_str(d, b, "type", "text");
        yyjson_mut_obj_add_strncpy(d, b, "text", text + xs, tn);
    }
    for (int i = 0; i < ch->n_calls; i++)
        yyjson_mut_arr_append(content, tool_use(d, &ch->calls[i], 1));
    yyjson_mut_obj_add_str(d, m, "stop_reason", stop_reason(j));
    add_stop_sequence(d, m, j);
    usage(d, m, j, 0);
    size_t n = 0;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, n);
}

/* One event, appended to o->pend. */
static int event(struct an_out *o, const char *name, yyjson_mut_doc *d)
{
    size_t n = 0;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    if (!json)
        return -1;
    size_t k = strlen(name);
    char *p = realloc(o->pend, o->n_pend + k + n + 16);
    if (!p) {
        free(json);
        return -1;
    }
    o->pend = p;
    int w = sprintf(p + o->n_pend, "event: %s\ndata: ", name);
    memcpy(p + o->n_pend + w, json, n);
    memcpy(p + o->n_pend + w + n, "\n\n", 2);
    o->n_pend += (size_t)w + n + 2;
    free(json);
    return 0;
}

/* A new document whose root is {"type": type}. */
static yyjson_mut_doc *typed(const char *type, yyjson_mut_val **root)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    *root = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, *root);
    yyjson_mut_obj_add_str(d, *root, "type", type);
    return d;
}

static int block_stop(struct an_out *o)
{
    int bad = 0;
    yyjson_mut_val *root;
    yyjson_mut_doc *d;
    if (o->open == THINKING) { /* its signature, which is empty here */
        d = typed("content_block_delta", &root);
        yyjson_mut_obj_add_int(d, root, "index", o->block);
        yyjson_mut_val *dl = yyjson_mut_obj_add_obj(d, root, "delta");
        yyjson_mut_obj_add_str(d, dl, "type", "signature_delta");
        yyjson_mut_obj_add_str(d, dl, "signature", "");
        bad |= event(o, "content_block_delta", d);
    }
    d = typed("content_block_stop", &root);
    yyjson_mut_obj_add_int(d, root, "index", o->block);
    bad |= event(o, "content_block_stop", d);
    o->open = NONE;
    o->block++;
    return bad;
}

/* text or thinking, s (n bytes), in a block of that kind */
static int delta(struct an_out *o, int kind, const char *s, size_t n)
{
    int bad = 0;
    yyjson_mut_val *root;
    yyjson_mut_doc *d;
    if (o->open != kind) {
        if (o->open != NONE)
            bad |= block_stop(o);
        d = typed("content_block_start", &root);
        yyjson_mut_obj_add_int(d, root, "index", o->block);
        yyjson_mut_val *b = yyjson_mut_obj_add_obj(d, root, "content_block");
        if (kind == THINKING) {
            yyjson_mut_obj_add_str(d, b, "type", "thinking");
            yyjson_mut_obj_add_str(d, b, "thinking", "");
            yyjson_mut_obj_add_str(d, b, "signature", "");
        } else {
            yyjson_mut_obj_add_str(d, b, "type", "text");
            yyjson_mut_obj_add_str(d, b, "text", "");
        }
        bad |= event(o, "content_block_start", d);
        o->open = kind;
    }
    d = typed("content_block_delta", &root);
    yyjson_mut_obj_add_int(d, root, "index", o->block);
    yyjson_mut_val *dl = yyjson_mut_obj_add_obj(d, root, "delta");
    yyjson_mut_obj_add_str(d, dl, "type",
                           kind == THINKING ? "thinking_delta" : "text_delta");
    yyjson_mut_obj_add_strncpy(d, dl, kind == THINKING ? "thinking" : "text", s,
                               n);
    return bad | event(o, "content_block_delta", d);
}

/* a tool call closed: a block of its own, its input in one delta */
static int call_block(struct an_out *o, const struct srv_call *c)
{
    int bad = o->open != NONE ? block_stop(o) : 0;
    yyjson_mut_val *root;
    yyjson_mut_doc *d = typed("content_block_start", &root);
    yyjson_mut_obj_add_int(d, root, "index", o->block);
    yyjson_mut_obj_add_val(d, root, "content_block", tool_use(d, c, 0));
    bad |= event(o, "content_block_start", d);
    d = typed("content_block_delta", &root);
    yyjson_mut_obj_add_int(d, root, "index", o->block);
    yyjson_mut_val *dl = yyjson_mut_obj_add_obj(d, root, "delta");
    yyjson_mut_obj_add_str(d, dl, "type", "input_json_delta");
    yyjson_mut_obj_add_strcpy(d, dl, "partial_json", c->args ? c->args : "{}");
    bad |= event(o, "content_block_delta", d);
    d = typed("content_block_stop", &root);
    yyjson_mut_obj_add_int(d, root, "index", o->block);
    bad |= event(o, "content_block_stop", d);
    o->block++;
    return bad;
}

/* The events the job has made possible since the last call; 1 when the
   stream is complete. Under j->mu. */
static int produce(struct an_out *o)
{
    struct srv_job *j = o->j;
    struct srv_choice *ch = &j->ch[0];
    int bad = 0;
    yyjson_mut_val *root;
    yyjson_mut_doc *d;
    if (!o->started) {
        d = typed("message_start", &root);
        yyjson_mut_val *m = head(d, o);
        yyjson_mut_obj_add_arr(d, m, "content");
        yyjson_mut_obj_add_null(d, m, "stop_reason");
        yyjson_mut_obj_add_null(d, m, "stop_sequence");
        usage(d, m, j, 1);
        yyjson_mut_obj_add_val(d, root, "message", m);
        bad |= event(o, "message_start", d);
        o->started = 1;
    }
    if (ch->n_think > o->r_think && o->hide_think)
        o->r_think = ch->n_think;
    if (ch->n_think > o->r_think) {
        size_t n =
            out_skip_ws(ch->think, ch->n_think, &o->r_think, &o->think_began);
        if (n)
            bad |= delta(o, THINKING, ch->think + o->r_think, n);
        o->r_think = ch->n_think;
    }
    size_t vis = srv_job_visible(j, 0);
    if (vis > o->r_text) {
        size_t n = out_skip_ws(ch->text, vis, &o->r_text, &o->text_began);
        const char *s = ch->text + o->r_text;
        int blank = ch->n_calls > 0;
        for (size_t i = 0; blank && i < n; i++)
            blank = (unsigned char)s[i] <= ' ';
        if (n && !blank)
            bad |= delta(o, TEXT, s, n);
        o->r_text = vis;
    }
    while (o->r_calls < ch->n_calls)
        bad |= call_block(o, &ch->calls[o->r_calls++]);
    if (!j->done)
        return bad ? -1 : 0;
    if (failed(j)) {
        d = typed("error", &root);
        yyjson_mut_val *e = yyjson_mut_obj_add_obj(d, root, "error");
        yyjson_mut_obj_add_str(d, e, "type", "api_error");
        yyjson_mut_obj_add_strcpy(
            d, e, "message", j->errmsg[0] ? j->errmsg : "generation failed");
        bad |= event(o, "error", d);
    } else {
        if (o->open != NONE)
            bad |= block_stop(o);
        d = typed("message_delta", &root);
        yyjson_mut_val *dl = yyjson_mut_obj_add_obj(d, root, "delta");
        yyjson_mut_obj_add_str(d, dl, "stop_reason", stop_reason(j));
        add_stop_sequence(d, dl, j);
        usage(d, root, j, 0);
        bad |= event(o, "message_delta", d);
        d = typed("message_stop", &root);
        bad |= event(o, "message_stop", d);
    }
    o->sent_end = 1;
    return bad ? -1 : 1;
}

static long stream_next(void *ctx, char *buf, size_t cap)
{
    struct an_out *o = ctx;
    if (o->off == o->n_pend) {
        o->off = o->n_pend = 0;
        if (o->sent_end)
            return -1;
        struct srv_job *j = o->j;
        pthread_mutex_lock(&j->mu);
        int end = produce(o);
        if (end == 0 && o->n_pend == 0) {
            /* nothing new: wait for the engine, then a ping, which is how
               a client gone away is noticed */
            struct timespec t;
            clock_gettime(CLOCK_REALTIME, &t);
            t.tv_sec += 5;
            int rc = pthread_cond_timedwait(&j->cv, &j->mu, &t);
            end = produce(o);
            if (o->n_pend == 0 && rc == ETIMEDOUT) {
                yyjson_mut_val *root;
                event(o, "ping", typed("ping", &root));
            }
        }
        if (end < 0)
            o->sent_end = 1;
        pthread_mutex_unlock(&j->mu);
        if (o->n_pend == 0)
            return o->sent_end ? -1 : 0;
    }
    size_t n = o->n_pend - o->off < cap ? o->n_pend - o->off : cap;
    memcpy(buf, o->pend + o->off, n);
    o->off += n;
    return (long)n;
}

static void out_done(void *ctx)
{
    struct an_out *o = ctx;
    if (!o)
        return;
    if (o->j) {
        pthread_mutex_lock(&o->j->mu);
        if (!o->j->done)
            o->j->cancel = 1; /* no one reads it any more */
        pthread_mutex_unlock(&o->j->mu);
        srv_job_release(o->j);
    }
    free(o->pend);
    free(o);
}

/* The request read into a job (count: only how many tokens); 0, or an
   answer already given in *ret. */
static struct srv_job *read_job(struct srv_req *r, int count, int *stream,
                                int *hide, char *model, size_t model_cap,
                                int *ret)
{
    yyjson_read_err re;
    yyjson_doc *doc =
        yyjson_read_opts((char *)r->body, r->n_body, 0, NULL, &re);
    yyjson_val *root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        *ret = an_error(r, 400, "invalid_request_error",
                        "The body must be a JSON object.");
        return NULL;
    }
    if (!count && !yyjson_is_int(yyjson_obj_get(root, "max_tokens"))) {
        yyjson_doc_free(doc);
        *ret = an_error(r, 400, "invalid_request_error",
                        "max_tokens: Field required");
        return NULL;
    }
    const char *m = yyjson_get_str(yyjson_obj_get(root, "model"));
    snprintf(model, model_cap, "%s", m ? m : r->srv->cfg.model_id);
    yyjson_val *sv = yyjson_obj_get(root, "stream");
    *stream = yyjson_is_bool(sv) && yyjson_get_bool(sv);
    const char *disp = yyjson_get_str(
        yyjson_obj_get(yyjson_obj_get(root, "thinking"), "display"));
    *hide = disp && !strcmp(disp, "omitted");
    char why[300];
    size_t n = 0;
    char *body = an_request_body(root, &n, why, sizeof why);
    yyjson_doc_free(doc);
    if (!body) {
        *ret = an_error(r, 400, "invalid_request_error", "%s", why);
        return NULL;
    }
    struct srv_job *j = srv_job_new();
    if (!j) {
        free(body);
        *ret = an_error(r, 500, "api_error", "out of memory");
        return NULL;
    }
    srv_reasoning_default(&r->srv->cfg, &j->p);
    struct srv_parsed pr;
    struct srv_perr e;
    int bad = srv_parse_request(body, n, 1, j, &pr, &e);
    srv_parsed_free(&pr);
    free(body);
    if (bad) {
        srv_job_release(j);
        *ret = an_error(r, e.oom ? 500 : 400,
                        e.oom ? "api_error" : "invalid_request_error", "%s",
                        e.msg);
        return NULL;
    }
    char ierr[300];
    if (srv_images_fetch(r->srv, j, ierr, sizeof ierr) != 0) {
        srv_job_release(j);
        *ret = an_error(r, 400, "invalid_request_error", "%s", ierr);
        return NULL;
    }
    if (count)
        j->kind = JOB_COUNT;
    return j;
}

int srv_anthropic_messages(struct srv_req *r)
{
    int ret = 0, stream = 0, hide = 0;
    char model[256];
    struct srv_job *j =
        read_job(r, 0, &stream, &hide, model, sizeof model, &ret);
    if (!j)
        return ret;
    struct an_out *o = calloc(1, sizeof(*o));
    if (!o) {
        srv_job_release(j);
        return an_error(r, 500, "api_error", "out of memory");
    }
    o->j = j;
    o->hide_think = hide;
    snprintf(o->model, sizeof o->model, "%s", model);
    srv_new_id(o->id, sizeof o->id, "msg_");
    if (srv_engine_submit(r->srv->engine, j) != 0) {
        out_done(o);
        return an_error(r, 529, "overloaded_error",
                        "The server is busy: %u requests are waiting.",
                        r->srv->cfg.max_queue);
    }
    if (stream) /* the stream holds the job through o from here on */
        return srv_reply_stream(r, stream_next, out_done, o);
    ret = reply_whole(r, o);
    out_done(o);
    return ret;
}

int srv_anthropic_count(struct srv_req *r)
{
    int ret = 0, stream = 0, hide = 0;
    char model[256];
    struct srv_job *j =
        read_job(r, 1, &stream, &hide, model, sizeof model, &ret);
    if (!j)
        return ret;
    if (srv_engine_submit(r->srv->engine, j) != 0) {
        srv_job_release(j);
        return an_error(r, 529, "overloaded_error", "The server is busy.");
    }
    pthread_mutex_lock(&j->mu);
    while (!j->done)
        pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);
    if (j->err != JANAS_LLM_OK) {
        ret = an_error(r, err_status(j), "invalid_request_error", "%s",
                       j->errmsg[0] ? j->errmsg : "the count failed");
        srv_job_release(j);
        return ret;
    }
    char out[64];
    int n =
        snprintf(out, sizeof out, "{\"input_tokens\":%u}", j->prompt_tokens);
    srv_job_release(j);
    return srv_reply_json(r, 200, strndup(out, (size_t)n), (size_t)n);
}
