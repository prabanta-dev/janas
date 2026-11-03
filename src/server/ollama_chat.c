/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * ollama_chat.c - POST /api/chat and POST /api/generate, Ollama's: the
 * request made the body of an OpenAI one and read by request.c into a job,
 * so that both APIs take the same path to the engine; the reply written
 * the way Ollama writes it, whole or, by default, as lines of JSON while it
 * is written: the reasoning as "thinking", the answer as "content" (or
 * "response"), each tool call once it is closed, its arguments an object,
 * then a last line with done, the reason and the counts.
 *
 * Read from Ollama's request: the messages with their pictures (base64)
 * and tool calls, tools, think (true, false or a level, which is on),
 * format ("json" or a JSON Schema), and of the options temperature, top_k,
 * top_p, min_p, seed, stop, presence_penalty, frequency_penalty and
 * num_predict. The others (num_ctx, repeat_penalty, keep_alive...) are
 * left: the context is the server's, set at its start. /api/generate with
 * raw or a suffix continues the prompt as it is, as /completions does;
 * without a prompt it only says the model is loaded, as Ollama does.
 */
#include "chat_out.h"
#include "ollama.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* A reply being made for Ollama's client. */
struct ol_out {
    struct srv_job *j;
    int chat; /* /api/chat: a message; /api/generate: a response */
    char model[256];
    struct timespec t0;
    size_t r_think, r_text;
    int r_calls, think_began, text_began, sent_end;
    char *pend; /* the lines ready, not yet sent */
    size_t n_pend, off;
};

/* "load" or "unload": what a request without a prompt asks for. */
static const char *loading(yyjson_val *root)
{
    yyjson_val *ka = yyjson_obj_get(root, "keep_alive");
    int zero = (yyjson_is_num(ka) && yyjson_get_num(ka) == 0) ||
               (yyjson_is_str(ka) && !strcmp(yyjson_get_str(ka), "0"));
    return zero ? "unload" : "load";
}

static yyjson_mut_val *head(yyjson_mut_doc *d, const char *model)
{
    char now[48];
    ol_now(now, sizeof now);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_strcpy(d, o, "model", model);
    yyjson_mut_obj_add_strcpy(d, o, "created_at", now);
    return o;
}

/* The answer to a request that only loads or unloads: the model is always
   loaded, for as long as the server runs. */
static int reply_load(struct srv_req *r, const char *model, int chat,
                      const char *why)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = head(d, model);
    if (chat) {
        yyjson_mut_val *m = yyjson_mut_obj_add_obj(d, o, "message");
        yyjson_mut_obj_add_str(d, m, "role", "assistant");
        yyjson_mut_obj_add_str(d, m, "content", "");
    } else {
        yyjson_mut_obj_add_str(d, o, "response", "");
    }
    yyjson_mut_obj_add_str(d, o, "done_reason", why);
    yyjson_mut_obj_add_bool(d, o, "done", true);
    size_t n = 0;
    char *json = yyjson_mut_write(d, 0, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, n);
}

/* Tool calls from..to of the reply, their arguments as an object. */
static yyjson_mut_val *calls(yyjson_mut_doc *d, const struct srv_choice *c,
                             int from, int to)
{
    yyjson_mut_val *arr = yyjson_mut_arr(d);
    for (int i = from; i < to; i++) {
        yyjson_mut_val *t = yyjson_mut_arr_add_obj(d, arr);
        yyjson_mut_obj_add_strcpy(d, t, "id", c->calls[i].id);
        yyjson_mut_val *f = yyjson_mut_obj_add_obj(d, t, "function");
        yyjson_mut_obj_add_int(d, f, "index", i);
        yyjson_mut_obj_add_strcpy(d, f, "name", c->calls[i].name);
        const char *a = c->calls[i].args ? c->calls[i].args : "{}";
        yyjson_doc *ad = yyjson_read(a, strlen(a), 0);
        yyjson_val *av = yyjson_doc_get_root(ad);
        yyjson_mut_obj_add_val(d, f, "arguments",
                               yyjson_is_obj(av) ? yyjson_val_mut_copy(d, av)
                                                 : yyjson_mut_strcpy(d, a));
        yyjson_doc_free(ad);
    }
    return arr;
}

/* A line, or the whole answer: the answer's text s (n bytes), the
   reasoning th (tn bytes), the tool calls from..to. */
static yyjson_mut_val *piece(yyjson_mut_doc *d, const struct ol_out *o,
                             const char *s, size_t n, const char *th, size_t tn,
                             int from, int to)
{
    yyjson_mut_val *root = head(d, o->model);
    yyjson_mut_val *m = root;
    if (o->chat) {
        m = yyjson_mut_obj_add_obj(d, root, "message");
        yyjson_mut_obj_add_str(d, m, "role", "assistant");
    }
    yyjson_mut_obj_add_strncpy(d, m, o->chat ? "content" : "response",
                               s ? s : "", s ? n : 0);
    if (tn)
        yyjson_mut_obj_add_strncpy(d, m, "thinking", th, tn);
    if (o->chat && to > from)
        yyjson_mut_obj_add_val(d, m, "tool_calls",
                               calls(d, &o->j->ch[0], from, to));
    return root;
}

/* The last fields: the reason, done, and the counts and times (in
   nanoseconds) of the reply. */
static void finish(yyjson_mut_doc *d, yyjson_mut_val *root,
                   const struct ol_out *o)
{
    const struct srv_job *j = o->j;
    const struct janas_llm_chat_stats *st = &j->st;
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (double)(t1.tv_sec - o->t0.tv_sec) * 1e9 +
                   (double)(t1.tv_nsec - o->t0.tv_nsec);
    yyjson_mut_obj_add_str(d, root, "done_reason",
                           j->ch[0].finish == JANAS_LLM_FINISH_LENGTH ? "length"
                                                                      : "stop");
    yyjson_mut_obj_add_bool(d, root, "done", true);
    yyjson_mut_obj_add_uint(d, root, "total_duration", (uint64_t)total);
    yyjson_mut_obj_add_uint(d, root, "load_duration", 0);
    yyjson_mut_obj_add_uint(d, root, "prompt_eval_count", st->prompt_tokens);
    yyjson_mut_obj_add_uint(d, root, "prompt_eval_cached_count",
                            st->cached_tokens);
    yyjson_mut_obj_add_uint(d, root, "prompt_eval_duration",
                            (uint64_t)(st->input_seconds * 1e9));
    yyjson_mut_obj_add_uint(d, root, "eval_count", st->output_tokens);
    yyjson_mut_obj_add_uint(d, root, "eval_duration",
                            (uint64_t)(st->output_seconds * 1e9));
}

static unsigned err_status(const struct srv_job *j)
{
    return j->err == JANAS_LLM_EFULL || j->err == JANAS_LLM_ELIMIT ||
                   j->err == JANAS_LLM_EINVAL || j->err == JANAS_LLM_EMODEL
               ? 400
               : 500;
}

static int failed(const struct srv_job *j)
{
    return j->err != JANAS_LLM_OK || j->ch[0].finish == JANAS_LLM_FINISH_ERROR;
}

/* The whole reply, once the engine is done with it. */
static int reply_whole(struct srv_req *r, struct ol_out *o)
{
    struct srv_job *j = o->j;
    pthread_mutex_lock(&j->mu);
    while (!j->done)
        pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);
    if (failed(j))
        return ol_error(r, err_status(j), "%s",
                        j->errmsg[0] ? j->errmsg : "generation failed");
    const struct srv_choice *ch = &j->ch[0];
    const char *text = ch->text ? ch->text : "";
    size_t xs = 0, ts = 0, tn = ch->n_text, thn = 0;
    int b1 = 0, b2 = 0;
    if (j->kind == JOB_CHAT) {
        tn = out_skip_ws(text, ch->n_text, &xs, &b1);
        while (tn && ch->n_calls && (unsigned char)text[xs + tn - 1] <= ' ')
            tn--; /* the line breaks around the calls are not content */
        thn = out_skip_ws(ch->think ? ch->think : "", ch->n_think, &ts, &b2);
        while (thn && (unsigned char)ch->think[ts + thn - 1] <= ' ')
            thn--;
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root =
        piece(d, o, text + xs, tn, ch->think ? ch->think + ts : NULL, thn, 0,
              ch->n_calls);
    finish(d, root, o);
    size_t n = 0;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, n);
}

/* One line of the stream, appended to o->pend: a piece of the reply says
   it is not the last (Ollama's clients read done on every line). */
static int line(struct ol_out *o, yyjson_mut_doc *d)
{
    yyjson_mut_val *root = yyjson_mut_doc_get_root(d);
    if (root && !yyjson_mut_obj_get(root, "done") &&
        !yyjson_mut_obj_get(root, "error"))
        yyjson_mut_obj_add_bool(d, root, "done", false);
    size_t n = 0;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    if (!json)
        return -1;
    char *p = realloc(o->pend, o->n_pend + n + 1);
    if (!p) {
        free(json);
        return -1;
    }
    o->pend = p;
    memcpy(p + o->n_pend, json, n);
    p[o->n_pend + n] = '\n';
    o->n_pend += n + 1;
    free(json);
    return 0;
}

/* The lines the job has made possible since the last call; 1 when the
   stream is complete. Under j->mu. */
static int produce(struct ol_out *o)
{
    struct srv_job *j = o->j;
    struct srv_choice *ch = &j->ch[0];
    int bad = 0;
    if (ch->n_think > o->r_think) {
        size_t n =
            out_skip_ws(ch->think, ch->n_think, &o->r_think, &o->think_began);
        if (n) {
            yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
            piece(d, o, NULL, 0, ch->think + o->r_think, n, 0, 0);
            bad |= line(o, d);
        }
        o->r_think = ch->n_think;
    }
    size_t vis = srv_job_visible(j, 0);
    if (vis > o->r_text) {
        size_t n = j->kind == JOB_CHAT
                       ? out_skip_ws(ch->text, vis, &o->r_text, &o->text_began)
                       : vis - o->r_text;
        const char *s = ch->text + o->r_text;
        int blank = ch->n_calls > 0;
        for (size_t i = 0; blank && i < n; i++)
            blank = (unsigned char)s[i] <= ' ';
        if (n && !blank) {
            yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
            piece(d, o, s, n, NULL, 0, 0, 0);
            bad |= line(o, d);
        }
        o->r_text = vis;
    }
    if (o->chat && ch->n_calls > o->r_calls) {
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        piece(d, o, NULL, 0, NULL, 0, o->r_calls, ch->n_calls);
        bad |= line(o, d);
        o->r_calls = ch->n_calls;
    }
    if (!j->done)
        return bad ? -1 : 0;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    if (failed(j)) {
        yyjson_mut_val *root = yyjson_mut_obj(d);
        yyjson_mut_doc_set_root(d, root);
        yyjson_mut_obj_add_strcpy(
            d, root, "error", j->errmsg[0] ? j->errmsg : "generation failed");
    } else {
        finish(d, piece(d, o, NULL, 0, NULL, 0, 0, 0), o);
    }
    bad |= line(o, d);
    o->sent_end = 1;
    return bad ? -1 : 1;
}

static long stream_next(void *ctx, char *buf, size_t cap)
{
    struct ol_out *o = ctx;
    if (o->off == o->n_pend) {
        o->off = o->n_pend = 0;
        if (o->sent_end)
            return -1;
        struct srv_job *j = o->j;
        pthread_mutex_lock(&j->mu);
        int end = produce(o);
        if (end == 0 && o->n_pend == 0) {
            /* nothing new: wait for the engine a while */
            struct timespec t;
            clock_gettime(CLOCK_REALTIME, &t);
            t.tv_sec += 5;
            pthread_cond_timedwait(&j->cv, &j->mu, &t);
            end = produce(o);
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
    struct ol_out *o = ctx;
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

/*
 * Both routes: the request read (root), its OpenAI body made by build
 * (chat: messages, else a prompt), the job run, the reply given.
 */
static int run(struct srv_req *r, yyjson_doc *doc, int chat_route, int kind)
{
    yyjson_val *root = yyjson_doc_get_root(doc);
    const char *model = yyjson_get_str(yyjson_obj_get(root, "model"));
    if (model && ol_model(r->srv, model) != 1) {
        int ret = ol_model(r->srv, model) == 2
                      ? ol_error(r, 400, "\"%s\" does not support %s", model,
                                 chat_route ? "chat" : "generate")
                      : ol_error(r, 404,
                                 "model '%s' not found: this server runs "
                                 "'%s'",
                                 model, r->srv->cfg.model_id);
        yyjson_doc_free(doc);
        return ret;
    }
    char name[256]; /* the name the answer gives, kept past the request */
    snprintf(name, sizeof name, "%s", model ? model : r->srv->cfg.model_id);
    yyjson_val *opts = yyjson_obj_get(root, "options");
    size_t n = 0;
    char *body = ol_request_body(root, chat_route, kind == JOB_CHAT, &n);
    yyjson_val *sv = yyjson_obj_get(root, "stream");
    int stream = !yyjson_is_bool(sv) || yyjson_get_bool(sv); /* Ollama's */
    double np = yyjson_get_num(yyjson_obj_get(opts, "num_predict"));
    yyjson_doc_free(doc);

    struct srv_job *j = srv_job_new();
    struct ol_out *o = calloc(1, sizeof(*o));
    if (!body || !j || !o) {
        free(body);
        free(o);
        srv_job_release(j);
        return ol_error(r, 500, "out of memory");
    }
    clock_gettime(CLOCK_MONOTONIC, &o->t0);
    srv_reasoning_default(&r->srv->cfg, &j->p); /* unless the request says */
    struct srv_parsed pr;
    struct srv_perr e;
    int bad = srv_parse_request(body, n, kind == JOB_CHAT, j, &pr, &e);
    srv_parsed_free(&pr);
    free(body);
    if (bad) {
        free(o);
        srv_job_release(j);
        return ol_error(r, e.oom ? 500 : 400, "%s", e.msg);
    }
    /* num_predict counts every token written, the reasoning too; -1 and
       -2 (to the end of the context) are no limit, as is its absence */
    j->p.max_answer = 0;
    j->p.max_reply = np >= 1 ? (int32_t)(np < 1e9 ? np : 1e9) : 0;
    char ierr[300];
    if (srv_images_fetch(r->srv, j, ierr, sizeof ierr) != 0) {
        free(o);
        srv_job_release(j);
        return ol_error(r, 400, "%s", ierr);
    }
    o->j = j;
    o->chat = chat_route;
    snprintf(o->model, sizeof o->model, "%s", name);
    if (srv_engine_submit(r->srv->engine, j) != 0) {
        out_done(o);
        return ol_error(r, 503,
                        "the server is busy: %u requests are waiting, try "
                        "again shortly",
                        r->srv->cfg.max_queue);
    }
    if (stream) /* the stream holds the job through o from here on */
        return srv_reply_stream_as(r, stream_next, out_done, o,
                                   "application/x-ndjson");
    int ret = reply_whole(r, o);
    out_done(o);
    return ret;
}

static yyjson_doc *read_body(struct srv_req *r, int *ret)
{
    yyjson_read_err re;
    yyjson_doc *doc =
        yyjson_read_opts((char *)r->body, r->n_body, 0, NULL, &re);
    if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc))) {
        yyjson_doc_free(doc);
        *ret = ol_error(r, 400, "the body must be a JSON object");
        return NULL;
    }
    return doc;
}

int ol_chat(struct srv_req *r)
{
    int ret;
    yyjson_doc *doc = read_body(r, &ret);
    if (!doc)
        return ret;
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *msgs = yyjson_obj_get(root, "messages");
    if (!msgs || (yyjson_is_arr(msgs) && !yyjson_arr_size(msgs))) {
        const char *model = yyjson_get_str(yyjson_obj_get(root, "model"));
        if (model && ol_model(r->srv, model) != 1)
            return run(r, doc, 1, JOB_CHAT); /* which says why */
        ret = reply_load(r, model ? model : r->srv->cfg.model_id, 1,
                         loading(root));
        yyjson_doc_free(doc);
        return ret;
    }
    if (!yyjson_is_arr(msgs)) {
        yyjson_doc_free(doc);
        return ol_error(r, 400, "'messages' must be an array");
    }
    return run(r, doc, 1, JOB_CHAT);
}

int ol_generate(struct srv_req *r)
{
    int ret;
    yyjson_doc *doc = read_body(r, &ret);
    if (!doc)
        return ret;
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *p = yyjson_obj_get(root, "prompt");
    yyjson_val *sx = yyjson_obj_get(root, "suffix");
    yyjson_val *raw = yyjson_obj_get(root, "raw");
    yyjson_val *im = yyjson_obj_get(root, "images");
    int text = (yyjson_is_bool(raw) && yyjson_get_bool(raw)) ||
               (yyjson_is_str(sx) && yyjson_get_len(sx));
    int pics = yyjson_is_arr(im) && yyjson_arr_size(im);
    if (p && !yyjson_is_str(p)) {
        yyjson_doc_free(doc);
        return ol_error(r, 400, "'prompt' must be a string");
    }
    if (!yyjson_get_len(p) && !pics && !text) {
        const char *model = yyjson_get_str(yyjson_obj_get(root, "model"));
        if (model && ol_model(r->srv, model) != 1)
            return run(r, doc, 0, JOB_CHAT); /* which says why */
        ret = reply_load(r, model ? model : r->srv->cfg.model_id, 0,
                         loading(root));
        yyjson_doc_free(doc);
        return ret;
    }
    if (text && pics) {
        yyjson_doc_free(doc);
        return ol_error(r, 400,
                        "pictures go with a prompt in the chat format: not "
                        "with raw or a suffix");
    }
    return run(r, doc, 0, text ? JOB_TEXT : JOB_CHAT);
}
