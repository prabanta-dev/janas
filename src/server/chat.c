/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * chat.c - POST /chat/completions and POST /completions: the request's JSON
 * into a job for the engine, and the reply back, whole or as server-sent
 * events while it is written (chat_stream.c).
 *
 * The request itself is read in request.c and request_opts.c; the JSON of
 * the answer is made in chat_out.c.
 */
#include "chat_out.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

long srv_chat_stream_next(void *ctx, char *buf, size_t cap);
void srv_chat_stream_done(void *ctx);

/* The whole reply, once the engine is done with it. */
static int reply_whole(struct srv_req *r, struct out *o)
{
    struct srv_job *j = o->j;
    pthread_mutex_lock(&j->mu);
    while (!j->done)
        pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);
    if (j->err == JANAS_LLM_ELIMIT)
        return srv_reply_limit(r, j);
    if (j->err == JANAS_LLM_EFULL)
        return srv_reply_error(r, 400, "invalid_request_error",
                               "context_length_exceeded", "%s", j->errmsg);
    if (j->err == JANAS_LLM_EINVAL || j->err == JANAS_LLM_EMODEL)
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "%s", j->errmsg);
    if (j->err != JANAS_LLM_OK || j->ch[0].finish == JANAS_LLM_FINISH_ERROR)
        return srv_reply_error(r, 500, "server_error", NULL, "%s",
                               j->errmsg[0] ? j->errmsg : "generation failed");
    size_t n = 0;
    char *json = out_whole(o, &n);
    out_store(o);
    return srv_reply_json(r, 200, json, n);
}

/* Shared by both endpoints: parse, queue, answer. */
static int create(struct srv_req *r, int chat)
{
    struct srv_parsed pr;
    struct srv_perr e;
    struct srv_job *j = srv_job_new();
    struct out *o = calloc(1, sizeof(*o));
    int ret;
    if (!j || !o) {
        ret = srv_reply_error(r, 500, "server_error", NULL, "out of memory");
        free(o);
        srv_job_release(j);
        return ret;
    }
    j->p.thinking = r->srv->cfg.thinking; /* unless the request says */
    if (srv_parse_request(r->body, r->n_body, chat, j, &pr, &e) != 0) {
        ret = e.oom ? srv_reply_error(r, 500, "server_error", NULL, "%s", e.msg)
                    : srv_reply_error(r, 400, "invalid_request_error", e.code,
                                      "%s", e.msg);
        srv_parsed_free(&pr);
        free(o);
        srv_job_release(j);
        return ret;
    }
    o->j = j;
    o->srv = r->srv;
    o->chat = chat;
    o->include_usage = pr.include_usage;
    o->echo = pr.echo;
    o->logprobs = pr.logprobs;
    o->store = pr.store;
    o->metadata = pr.metadata; /* the answer's from here */
    o->messages = pr.messages;
    o->created = (int64_t)time(NULL);
    srv_new_id(o->id, sizeof(o->id), chat ? "chatcmpl-" : "cmpl-");
    snprintf(o->model, sizeof(o->model), "%s", r->srv->cfg.model_id);
    if (srv_engine_submit(r->srv->engine, j) != 0) {
        ret = srv_reply_error(r, 429, "rate_limit_exceeded", NULL,
                              "The server is busy: %u requests are waiting. "
                              "Try again shortly.",
                              r->srv->cfg.max_queue);
        out_free(o);
        return ret;
    }
    if (pr.stream) /* the stream holds the job through o from here on */
        return srv_reply_stream(r, srv_chat_stream_next, srv_chat_stream_done,
                                o);
    ret = reply_whole(r, o);
    out_free(o);
    return ret;
}

int srv_chat_create(struct srv_req *r)
{
    return create(r, 1);
}

int srv_completion_create(struct srv_req *r)
{
    return create(r, 0);
}
