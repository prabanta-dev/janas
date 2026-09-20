/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * chat_out.c - the JSON of the answers of /chat/completions and
 * /completions (see chat_out.h).
 */
#include "chat_out.h"

#include <microhttpd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

yyjson_mut_val *out_head(yyjson_mut_doc *d, const struct out *o,
                         const char *object)
{
    yyjson_mut_val *root = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, root);
    yyjson_mut_obj_add_str(d, root, "id", o->id);
    yyjson_mut_obj_add_str(d, root, "object", object);
    yyjson_mut_obj_add_int(d, root, "created", o->created);
    yyjson_mut_obj_add_str(d, root, "model", o->model);
    yyjson_mut_obj_add_str(d, root, "system_fingerprint",
                           "janas-" JANAS_VERSION);
    return root;
}

void out_usage(yyjson_mut_doc *d, yyjson_mut_val *obj, const struct srv_job *j)
{
    yyjson_mut_val *u = yyjson_mut_obj_add_obj(d, obj, "usage");
    yyjson_mut_obj_add_uint(d, u, "prompt_tokens", j->prompt_tokens);
    yyjson_mut_obj_add_uint(d, u, "completion_tokens", j->completion_tokens);
    yyjson_mut_obj_add_uint(d, u, "total_tokens",
                            (uint64_t)j->prompt_tokens + j->completion_tokens);
    yyjson_mut_val *pd = yyjson_mut_obj_add_obj(d, u, "prompt_tokens_details");
    yyjson_mut_obj_add_uint(d, pd, "cached_tokens", j->st.cached_tokens);
    yyjson_mut_val *cd =
        yyjson_mut_obj_add_obj(d, u, "completion_tokens_details");
    yyjson_mut_obj_add_uint(d, cd, "reasoning_tokens", j->reasoning_tokens);
    /* not in OpenAI's schema, which allows more: the context and the time
       the engine spent reading and writing */
    yyjson_mut_val *jn = yyjson_mut_obj_add_obj(d, u, "janas");
    yyjson_mut_obj_add_uint(d, jn, "context_used", j->st.context_used);
    yyjson_mut_obj_add_uint(d, jn, "context_size", j->st.context_size);
    yyjson_mut_obj_add_real(d, jn, "input_seconds", j->st.input_seconds);
    yyjson_mut_obj_add_real(d, jn, "output_seconds", j->st.output_seconds);
}

const char *out_finish(const struct srv_choice *c)
{
    if (c->by_stop)
        return "stop";
    switch (c->finish) {
    case JANAS_LLM_FINISH_LENGTH:
    case JANAS_LLM_FINISH_CONTEXT:
        return "length";
    case JANAS_LLM_FINISH_TOOLS:
        return "tool_calls";
    default:
        return "stop"; /* stopped, or cancelled: nobody reads it */
    }
}

size_t out_skip_ws(const char *s, size_t n, size_t *from, int *began)
{
    if (!*began) {
        while (*from < n && (unsigned char)s[*from] <= ' ')
            (*from)++;
        if (*from < n)
            *began = 1;
    }
    return n > *from ? n - *from : 0;
}

/* A token's bytes as a string that is valid UTF-8: a token can end inside a
   character, and what it holds of it becomes U+FFFD (its bytes are given
   exactly beside it). */
static yyjson_mut_val *tok_str(yyjson_mut_doc *d, const char *s, size_t n)
{
    char buf[1024];
    size_t w = 0;
    for (size_t i = 0; i < n && w + 4 < sizeof(buf);) {
        unsigned char b = (unsigned char)s[i];
        size_t k = b < 0x80         ? 1
                   : (b >> 5) == 6  ? 2
                   : (b >> 4) == 14 ? 3
                   : (b >> 3) == 30 ? 4
                                    : 0;
        int ok = k > 0 && i + k <= n;
        for (size_t q = 1; ok && q < k; q++)
            ok = ((unsigned char)s[i + q] >> 6) == 2;
        if (ok) {
            memcpy(buf + w, s + i, k);
            w += k;
            i += k;
        } else {
            memcpy(buf + w, "\xef\xbf\xbd", 3);
            w += 3;
            i++;
        }
    }
    return yyjson_mut_strncpy(d, buf, w);
}

static yyjson_mut_val *bytes_arr(yyjson_mut_doc *d, const char *s, size_t n)
{
    yyjson_mut_val *a = yyjson_mut_arr(d);
    for (size_t i = 0; i < n; i++)
        yyjson_mut_arr_add_uint(d, a, (unsigned char)s[i]);
    return a;
}

/* {token, logprob, bytes} */
static yyjson_mut_val *lp_entry(yyjson_mut_doc *d, const struct srv_alt *a)
{
    yyjson_mut_val *e = yyjson_mut_obj(d);
    yyjson_mut_obj_add_val(d, e, "token", tok_str(d, a->tok, a->n));
    yyjson_mut_obj_add_real(d, e, "logprob", a->lp);
    yyjson_mut_obj_add_val(d, e, "bytes", bytes_arr(d, a->tok, a->n));
    return e;
}

yyjson_mut_val *out_chat_lp(yyjson_mut_doc *d, const struct srv_choice *c,
                            size_t from, size_t to)
{
    yyjson_mut_val *lp = yyjson_mut_obj(d);
    yyjson_mut_val *content = yyjson_mut_obj_add_arr(d, lp, "content");
    for (size_t i = from; i < to && i < c->n_lp; i++) {
        yyjson_mut_val *e = lp_entry(d, &c->lp[i].t);
        yyjson_mut_val *top = yyjson_mut_obj_add_arr(d, e, "top_logprobs");
        for (int k = 0; k < c->lp[i].n_top; k++)
            yyjson_mut_arr_append(top, lp_entry(d, &c->lp[i].top[k]));
        yyjson_mut_arr_append(content, e);
    }
    yyjson_mut_obj_add_null(d, lp, "refusal");
    return lp;
}

yyjson_mut_val *out_text_lp(yyjson_mut_doc *d, const struct srv_choice *c,
                            size_t from, size_t to, size_t offset, int top)
{
    yyjson_mut_val *lp = yyjson_mut_obj(d);
    yyjson_mut_val *toks = yyjson_mut_obj_add_arr(d, lp, "tokens");
    yyjson_mut_val *lps = yyjson_mut_obj_add_arr(d, lp, "token_logprobs");
    yyjson_mut_val *tops = yyjson_mut_obj_add_arr(d, lp, "top_logprobs");
    yyjson_mut_val *offs = yyjson_mut_obj_add_arr(d, lp, "text_offset");
    for (size_t i = from; i < to && i < c->n_lp; i++) {
        const struct srv_lp *t = &c->lp[i];
        yyjson_mut_arr_append(toks, tok_str(d, t->t.tok, t->t.n));
        yyjson_mut_arr_add_real(d, lps, t->t.lp);
        yyjson_mut_val *m = yyjson_mut_obj(d);
        for (int k = 0; k < t->n_top && k < top; k++)
            yyjson_mut_obj_add(m, tok_str(d, t->top[k].tok, t->top[k].n),
                               yyjson_mut_real(d, t->top[k].lp));
        yyjson_mut_arr_append(tops, m);
        yyjson_mut_arr_add_uint(d, offs, offset);
        offset += t->t.n;
    }
    return lp;
}

yyjson_mut_val *out_calls(yyjson_mut_doc *d, const struct srv_choice *c,
                          int from, int to, int with_index)
{
    yyjson_mut_val *arr = yyjson_mut_arr(d);
    for (int i = from; i < to && i < c->n_calls; i++) {
        yyjson_mut_val *t = yyjson_mut_arr_add_obj(d, arr);
        if (with_index)
            yyjson_mut_obj_add_int(d, t, "index", i);
        yyjson_mut_obj_add_str(d, t, "id", c->calls[i].id);
        yyjson_mut_obj_add_str(d, t, "type", "function");
        yyjson_mut_val *f = yyjson_mut_obj_add_obj(d, t, "function");
        yyjson_mut_obj_add_str(d, f, "name", c->calls[i].name);
        yyjson_mut_obj_add_str(d, f, "arguments", c->calls[i].args);
    }
    return arr;
}

/* The choices to return: all of them, or the best n_keep of each prompt's
   best_of by their mean log-probability. Into order[]; how many. */
static int pick(const struct srv_job *j, int *order)
{
    int groups = j->kind == JOB_TEXT && j->n_prompts > 0 ? j->n_prompts : 1;
    int per = j->n_choices / groups, keep = j->n_keep / groups, w = 0;
    if (keep >= per) { /* no best_of: all of them, in their order */
        for (int i = 0; i < j->n_choices; i++)
            order[i] = i;
        return j->n_choices;
    }
    for (int g = 0; g < groups; g++) {
        int *mine = order + w;
        for (int i = 0; i < per; i++) {
            int x = g * per + i, at = i < keep ? i : keep;
            /* insertion into the best `keep` so far, best first */
            while (at > 0 && j->ch[mine[at - 1]].score < j->ch[x].score) {
                if (at < keep)
                    mine[at] = mine[at - 1];
                at--;
            }
            if (at < keep)
                mine[at] = x;
        }
        w += keep;
    }
    return w;
}

static void chat_choice(yyjson_mut_doc *d, yyjson_mut_val *c,
                        const struct out *o, const struct srv_choice *ch)
{
    size_t ts = 0, xs = 0;
    int b1 = 0, b2 = 0;
    size_t tn = out_skip_ws(ch->text ? ch->text : "", ch->n_text, &xs, &b1);
    const char *text = ch->text ? ch->text + xs : "";
    while (tn && ch->n_calls && (unsigned char)text[tn - 1] <= ' ')
        tn--; /* the line breaks around the calls are not content */
    yyjson_mut_val *m = yyjson_mut_obj_add_obj(d, c, "message");
    yyjson_mut_obj_add_str(d, m, "role", "assistant");
    if (tn || !ch->n_calls)
        yyjson_mut_obj_add_strncpy(d, m, "content", text, tn);
    else
        yyjson_mut_obj_add_null(d, m, "content");
    size_t thn = out_skip_ws(ch->think ? ch->think : "", ch->n_think, &ts, &b2);
    while (thn && (unsigned char)ch->think[ts + thn - 1] <= ' ')
        thn--;
    if (thn)
        yyjson_mut_obj_add_strncpy(d, m, "reasoning_content", ch->think + ts,
                                   thn);
    if (ch->n_calls)
        yyjson_mut_obj_add_val(d, m, "tool_calls",
                               out_calls(d, ch, 0, ch->n_calls, 0));
    yyjson_mut_obj_add_null(d, m, "refusal");
    if (o->logprobs)
        yyjson_mut_obj_add_val(d, c, "logprobs",
                               out_chat_lp(d, ch, 0, ch->n_lp));
    else
        yyjson_mut_obj_add_null(d, c, "logprobs");
}

static void text_choice(yyjson_mut_doc *d, yyjson_mut_val *c,
                        const struct out *o, const struct srv_choice *ch,
                        const struct srv_prompt *pp)
{
    size_t pn = o->echo && pp ? pp->n : 0;
    char *all = malloc(pn + ch->n_text + 1);
    if (all) {
        if (pn)
            memcpy(all, pp->text, pn);
        memcpy(all + pn, ch->text ? ch->text : "", ch->n_text);
        yyjson_mut_obj_add_strncpy(d, c, "text", all, pn + ch->n_text);
        free(all);
    }
    if (o->logprobs)
        yyjson_mut_obj_add_val(
            d, c, "logprobs",
            out_text_lp(d, ch, 0, ch->n_lp, pn, o->logprobs - 1));
    else
        yyjson_mut_obj_add_null(d, c, "logprobs");
}

char *out_whole(struct out *o, size_t *n)
{
    struct srv_job *j = o->j;
    int *order = malloc((size_t)j->n_choices * sizeof(int));
    if (!order)
        return NULL;
    int count = pick(j, order);
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root =
        out_head(d, o, o->chat ? "chat.completion" : "text_completion");
    yyjson_mut_val *choices = yyjson_mut_obj_add_arr(d, root, "choices");
    int per = j->kind == JOB_TEXT && j->n_prompts ? j->n_choices / j->n_prompts
                                                  : j->n_choices;
    for (int i = 0; i < count; i++) {
        const struct srv_choice *ch = &j->ch[order[i]];
        yyjson_mut_val *c = yyjson_mut_arr_add_obj(d, choices);
        yyjson_mut_obj_add_int(d, c, "index", i);
        if (o->chat)
            chat_choice(d, c, o, ch);
        else
            text_choice(d, c, o, ch, &j->prompts[order[i] / per]);
        yyjson_mut_obj_add_str(d, c, "finish_reason", out_finish(ch));
    }
    out_usage(d, root, j);
    if (o->store && o->metadata) {
        yyjson_doc *md = yyjson_read(o->metadata, strlen(o->metadata), 0);
        if (md)
            yyjson_mut_obj_add_val(
                d, root, "metadata",
                yyjson_val_mut_copy(d, yyjson_doc_get_root(md)));
        yyjson_doc_free(md);
    } else if (o->store) {
        yyjson_mut_obj_add_obj(d, root, "metadata");
    }
    char *json = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, n);
    yyjson_mut_doc_free(d);
    free(order);
    return json;
}

void out_store(struct out *o)
{
    if (!o->store || !o->chat || o->j->err != JANAS_LLM_OK)
        return;
    size_t n;
    char *json = out_whole(o, &n);
    if (json)
        srv_store_put(o->srv->store, "chat.completion", o->id, o->created, json,
                      o->messages);
    free(json);
}

void out_free(struct out *o)
{
    if (!o)
        return;
    if (o->j) {
        pthread_mutex_lock(&o->j->mu);
        if (!o->j->done)
            o->j->cancel = 1; /* no one reads it any more */
        pthread_mutex_unlock(&o->j->mu);
        srv_job_release(o->j);
    }
    free(o->metadata);
    free(o->messages);
    free(o->pend);
    free(o);
}
