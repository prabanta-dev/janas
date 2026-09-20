/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * chat_stream.c - the replies of /chat/completions and /completions as
 * server-sent events while the engine writes them: the reasoning, the
 * answer with its log-probabilities, each tool call once it is closed (whole,
 * in one delta), then the reason it ended; the replies of n one after the
 * other; the usage when asked for, and [DONE]. While the request waits for
 * the ones before it, a comment says where it stands in the queue.
 */
#include "chat_out.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* One event of the stream, appended to o->pend. */
static int event(struct out *o, yyjson_mut_doc *d)
{
    size_t n = 0;
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    if (!json)
        return -1;
    char *p = realloc(o->pend, o->n_pend + n + 9);
    if (!p) {
        free(json);
        return -1;
    }
    o->pend = p;
    memcpy(p + o->n_pend, "data: ", 6);
    memcpy(p + o->n_pend + 6, json, n);
    memcpy(p + o->n_pend + 6 + n, "\n\n", 2);
    o->n_pend += n + 8;
    free(json);
    return 0;
}

static int raw_event(struct out *o, const char *s)
{
    size_t n = strlen(s);
    char *p = realloc(o->pend, o->n_pend + n);
    if (!p)
        return -1;
    o->pend = p;
    memcpy(p + o->n_pend, s, n);
    o->n_pend += n;
    return 0;
}

/* A chunk for choice o->k: a field of the delta (chat) or text, the tool
   calls closed since the last, the log-probabilities of the tokens written
   since the last, and the reason it ended. */
static int chunk(struct out *o, const char *field, const char *s, size_t n,
                 const char *finish)
{
    struct srv_choice *ch = &o->j->ch[o->k];
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root =
        out_head(d, o, o->chat ? "chat.completion.chunk" : "text_completion");
    yyjson_mut_val *choices = yyjson_mut_obj_add_arr(d, root, "choices");
    yyjson_mut_val *c = yyjson_mut_arr_add_obj(d, choices);
    yyjson_mut_obj_add_int(d, c, "index", o->k);
    if (o->chat) {
        yyjson_mut_val *dl = yyjson_mut_obj_add_obj(d, c, "delta");
        if (field && strcmp(field, "role") == 0) {
            yyjson_mut_obj_add_str(d, dl, "role", "assistant");
            yyjson_mut_obj_add_str(d, dl, "content", "");
        } else if (field && strcmp(field, "tool_calls") == 0) {
            yyjson_mut_obj_add_val(
                d, dl, "tool_calls",
                out_calls(d, ch, o->r_calls, ch->n_calls, 1));
            o->r_calls = ch->n_calls;
        } else if (field) {
            yyjson_mut_obj_add_strncpy(d, dl, field, s, n);
        }
    } else {
        yyjson_mut_obj_add_strncpy(d, c, "text", s ? s : "", s ? n : 0);
    }
    int lp =
        o->logprobs &&
        ((field && !strcmp(field, "content")) || (!o->chat && s) || finish) &&
        o->r_lp < ch->n_lp;
    if (lp && o->chat)
        yyjson_mut_obj_add_val(d, c, "logprobs",
                               out_chat_lp(d, ch, o->r_lp, ch->n_lp));
    else if (lp)
        yyjson_mut_obj_add_val(
            d, c, "logprobs",
            out_text_lp(d, ch, o->r_lp, ch->n_lp, 0, o->logprobs - 1));
    else
        yyjson_mut_obj_add_null(d, c, "logprobs");
    if (lp)
        o->r_lp = ch->n_lp;
    if (finish)
        yyjson_mut_obj_add_str(d, c, "finish_reason", finish);
    else
        yyjson_mut_obj_add_null(d, c, "finish_reason");
    return event(o, d);
}

/* The events choice o->k has made possible; 1 when it is over. Under
   j->mu. */
static int produce_choice(struct out *o, int *bad)
{
    struct srv_job *j = o->j;
    struct srv_choice *ch = &j->ch[o->k];
    if (!o->sent_role) {
        if (o->chat)
            *bad |= chunk(o, "role", NULL, 0, NULL);
        else if (o->echo) {
            const struct srv_prompt *pp =
                &j->prompts[o->k / (j->n_choices / j->n_prompts)];
            if (!pp->text && !ch->done && !j->done)
                return 0; /* token ids: their text comes with the reply */
            *bad |= chunk(o, NULL, pp->text ? pp->text : "", pp->n, NULL);
        }
        o->sent_role = 1;
    }
    if (o->chat && ch->n_think > o->r_think) {
        size_t n =
            out_skip_ws(ch->think, ch->n_think, &o->r_think, &o->think_began);
        if (n)
            *bad |=
                chunk(o, "reasoning_content", ch->think + o->r_think, n, NULL);
        o->r_think = ch->n_think;
    }
    size_t vis = srv_job_visible(j, o->k);
    if (vis > o->r_text) {
        size_t n = o->chat
                       ? out_skip_ws(ch->text, vis, &o->r_text, &o->text_began)
                       : vis - o->r_text;
        /* the line breaks around tool calls are not content */
        const char *s = ch->text + o->r_text;
        int blank = ch->n_calls > 0;
        for (size_t i = 0; blank && i < n; i++)
            blank = (unsigned char)s[i] <= ' ';
        if (n && !blank)
            *bad |= chunk(o, o->chat ? "content" : NULL, s, n, NULL);
        o->r_text = vis;
    }
    if (o->chat && ch->n_calls > o->r_calls)
        *bad |= chunk(o, "tool_calls", NULL, 0, NULL);
    if (!ch->done || o->r_text < ch->n_text)
        return 0;
    if (j->err == JANAS_LLM_OK && ch->finish != JANAS_LLM_FINISH_ERROR)
        *bad |= chunk(o, NULL, NULL, 0, out_finish(ch));
    return 1;
}

/* The events the job has made possible since the last call; 1 when the
   stream is complete. Under j->mu. */
static int produce(struct out *o)
{
    struct srv_job *j = o->j;
    int bad = 0;
    if (!j->started && j->queue_pos != o->queue_seen) {
        char c[64];
        snprintf(c, sizeof(c), ": queue %u\n\n", j->queue_pos);
        bad |= raw_event(o, c);
        o->queue_seen = j->queue_pos;
    }
    while (o->k < j->n_choices && produce_choice(o, &bad) == 1) {
        o->k++; /* the next reply, from its start */
        o->r_think = o->r_text = o->r_lp = 0;
        o->r_calls = o->think_began = o->text_began = o->sent_role = 0;
    }
    if (o->k < j->n_choices && !j->done)
        return bad ? -1 : 0;
    if (!j->done)
        return bad ? -1 : 0;
    if (j->err != JANAS_LLM_OK || j->ch[0].finish == JANAS_LLM_FINISH_ERROR) {
        char msg[400];
        snprintf(msg, sizeof(msg),
                 "data: {\"error\":{\"message\":\"generation failed\","
                 "\"type\":\"server_error\",\"param\":null,\"code\":null}}"
                 "\n\n");
        bad |= raw_event(o, msg);
    } else if (o->include_usage) {
        yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *root = out_head(
            d, o, o->chat ? "chat.completion.chunk" : "text_completion");
        yyjson_mut_obj_add_arr(d, root, "choices");
        out_usage(d, root, j);
        bad |= event(o, d);
    }
    bad |= raw_event(o, "data: [DONE]\n\n");
    o->sent_end = 1;
    return bad ? -1 : 1;
}

long srv_chat_stream_next(void *ctx, char *buf, size_t cap)
{
    struct out *o = ctx;
    if (o->off == o->n_pend) {
        o->off = o->n_pend = 0;
        if (o->sent_end)
            return -1;
        struct srv_job *j = o->j;
        pthread_mutex_lock(&j->mu);
        int end = produce(o);
        if (end == 0 && o->n_pend == 0) {
            /* nothing new: wait for the engine, and after a while send a
               comment, which is how a client gone away is noticed */
            struct timespec t;
            clock_gettime(CLOCK_REALTIME, &t);
            t.tv_sec += 5;
            int rc = pthread_cond_timedwait(&j->cv, &j->mu, &t);
            end = produce(o);
            if (o->n_pend == 0 && rc == ETIMEDOUT)
                raw_event(o, ": keep-alive\n\n");
        }
        if (end < 0)
            o->sent_end = 1;
        int finished = end == 1;
        pthread_mutex_unlock(&j->mu);
        if (finished)
            out_store(o);
        if (o->n_pend == 0)
            return o->sent_end ? -1 : 0;
    }
    size_t n = o->n_pend - o->off < cap ? o->n_pend - o->off : cap;
    memcpy(buf, o->pend + o->off, n);
    o->off += n;
    return (long)n;
}

void srv_chat_stream_done(void *ctx)
{
    out_free(ctx);
}
