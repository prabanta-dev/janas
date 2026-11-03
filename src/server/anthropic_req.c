/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * anthropic_req.c - a request of Anthropic's /v1/messages as the body of
 * OpenAI's /chat/completions, which request.c then reads into a job: the
 * system prompt (a string or text blocks), the messages and their blocks -
 * text, pictures (base64 or a URL), the assistant's tool_use, the user's
 * tool_result (each a message of the tool's, before the rest of the user's
 * message) -, the tools (their input_schema the parameters), tool_choice,
 * thinking (off unless asked, as Anthropic has it), max_tokens (the
 * reasoning counted, as Anthropic counts it), stop_sequences and the
 * sampling; a message of the system's within them (Claude Code's) joins
 * the system prompt. The assistant's thinking blocks are not read back: the
 * chat format drops the reasoning of past turns. Kept apart from the HTTP side
 * so that tests/test_server_anthropic.c can try it on its own.
 */
#include "anthropic.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The text of a string, or of the text blocks of an array, joined. */
static int text_of(yyjson_mut_doc *q, yyjson_val *v, yyjson_mut_val **out)
{
    if (!v || yyjson_is_null(v)) {
        *out = yyjson_mut_str(q, "");
        return 0;
    }
    if (yyjson_is_str(v)) {
        *out = yyjson_val_mut_copy(q, v);
        return 0;
    }
    if (!yyjson_is_arr(v))
        return -1;
    size_t total = 0, idx, max;
    yyjson_val *b;
    yyjson_arr_foreach(v, idx, max, b)
    {
        yyjson_val *t = yyjson_obj_get(b, "text");
        if (yyjson_is_str(t))
            total += yyjson_get_len(t) + 2;
    }
    char *s = malloc(total + 1), *w = s;
    if (!s)
        return -1;
    yyjson_arr_foreach(v, idx, max, b)
    {
        yyjson_val *t = yyjson_obj_get(b, "text");
        if (!yyjson_is_str(t))
            continue;
        if (w > s)
            memcpy(w, "\n\n", 2), w += 2;
        memcpy(w, yyjson_get_str(t), yyjson_get_len(t));
        w += yyjson_get_len(t);
    }
    *out = yyjson_mut_strncpy(q, s, (size_t)(w - s));
    free(s);
    return 0;
}

/* An image block as OpenAI's image_url part. */
static yyjson_mut_val *image_part(yyjson_mut_doc *q, yyjson_val *b)
{
    yyjson_val *src = yyjson_obj_get(b, "source");
    const char *type = yyjson_get_str(yyjson_obj_get(src, "type"));
    yyjson_mut_val *p = yyjson_mut_obj(q);
    yyjson_mut_obj_add_str(q, p, "type", "image_url");
    yyjson_mut_val *iu = yyjson_mut_obj_add_obj(q, p, "image_url");
    if (type && !strcmp(type, "url")) {
        yyjson_val *u = yyjson_obj_get(src, "url");
        if (yyjson_is_str(u))
            yyjson_mut_obj_add_val(q, iu, "url", yyjson_val_mut_copy(q, u));
        return p;
    }
    const char *mt = yyjson_get_str(yyjson_obj_get(src, "media_type"));
    yyjson_val *data = yyjson_obj_get(src, "data");
    if (!yyjson_is_str(data))
        return p; /* request.c says what is missing */
    mt = mt ? mt : "image/png";
    size_t n = strlen(mt) + yyjson_get_len(data) + 16;
    char *url = malloc(n);
    if (url) {
        int k = snprintf(url, n, "data:%s;base64,", mt);
        memcpy(url + k, yyjson_get_str(data), yyjson_get_len(data));
        yyjson_mut_obj_add_strncpy(q, iu, "url", url,
                                   (size_t)k + yyjson_get_len(data));
        free(url);
    }
    return p;
}

/* A user's message: its tool_result blocks as messages of the tool's
   first, then the rest (text and pictures) as the user's. */
static int user_message(yyjson_mut_doc *q, yyjson_mut_val *out, yyjson_val *c,
                        char *why, size_t why_len)
{
    if (!yyjson_is_arr(c)) {
        yyjson_mut_val *m = yyjson_mut_arr_add_obj(q, out);
        yyjson_mut_obj_add_str(q, m, "role", "user");
        yyjson_mut_val *t;
        if (text_of(q, c, &t) != 0) {
            snprintf(why, why_len,
                     "a message's content is a string or "
                     "an array of blocks");
            return -1;
        }
        yyjson_mut_obj_add_val(q, m, "content", t);
        return 0;
    }
    size_t idx, max;
    yyjson_val *b;
    yyjson_arr_foreach(c, idx, max, b)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(b, "type"));
        if (!type || strcmp(type, "tool_result"))
            continue;
        yyjson_mut_val *m = yyjson_mut_arr_add_obj(q, out);
        yyjson_mut_obj_add_str(q, m, "role", "tool");
        yyjson_val *id = yyjson_obj_get(b, "tool_use_id");
        if (yyjson_is_str(id))
            yyjson_mut_obj_add_val(q, m, "tool_call_id",
                                   yyjson_val_mut_copy(q, id));
        yyjson_mut_val *t;
        if (text_of(q, yyjson_obj_get(b, "content"), &t) != 0) {
            snprintf(why, why_len,
                     "a tool_result's content is a string or "
                     "text blocks");
            return -1;
        }
        yyjson_val *err = yyjson_obj_get(b, "is_error");
        if (yyjson_is_bool(err) && yyjson_get_bool(err)) {
            /* the model is told: the result is an error */
            const char *s = yyjson_mut_get_str(t);
            size_t n = yyjson_mut_get_len(t);
            char *e = malloc(n + 8);
            if (e) {
                memcpy(e, "Error: ", 7);
                memcpy(e + 7, s, n);
                t = yyjson_mut_strncpy(q, e, n + 7);
                free(e);
            }
        }
        yyjson_mut_obj_add_val(q, m, "content", t);
    }
    yyjson_mut_val *parts = yyjson_mut_arr(q);
    yyjson_arr_foreach(c, idx, max, b)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(b, "type"));
        if (!type) {
            snprintf(why, why_len, "a content block has no type");
            return -1;
        }
        if (!strcmp(type, "text")) {
            yyjson_mut_val *p = yyjson_mut_arr_add_obj(q, parts);
            yyjson_mut_obj_add_str(q, p, "type", "text");
            yyjson_val *t = yyjson_obj_get(b, "text");
            yyjson_mut_obj_add_val(q, p, "text",
                                   t ? yyjson_val_mut_copy(q, t)
                                     : yyjson_mut_str(q, ""));
        } else if (!strcmp(type, "image")) {
            yyjson_mut_arr_append(parts, image_part(q, b));
        } else if (strcmp(type, "tool_result")) {
            snprintf(why, why_len,
                     "content blocks of type '%s' are not read here: "
                     "text, image, tool_use and tool_result are",
                     type);
            return -1;
        }
    }
    if (yyjson_mut_arr_size(parts)) {
        yyjson_mut_val *m = yyjson_mut_arr_add_obj(q, out);
        yyjson_mut_obj_add_str(q, m, "role", "user");
        yyjson_mut_obj_add_val(q, m, "content", parts);
    }
    return 0;
}

/* An assistant's message: its text, and its tool_use blocks as calls. */
static void assistant_message(yyjson_mut_doc *q, yyjson_mut_val *out,
                              yyjson_val *c)
{
    yyjson_mut_val *m = yyjson_mut_arr_add_obj(q, out);
    yyjson_mut_obj_add_str(q, m, "role", "assistant");
    if (!yyjson_is_arr(c)) {
        yyjson_mut_val *t;
        if (text_of(q, c, &t) == 0)
            yyjson_mut_obj_add_val(q, m, "content", t);
        return;
    }
    yyjson_mut_val *calls = yyjson_mut_arr(q);
    size_t idx, max, total = 0;
    yyjson_val *b;
    yyjson_arr_foreach(c, idx, max, b)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(b, "type"));
        yyjson_val *t = yyjson_obj_get(b, "text");
        if (type && !strcmp(type, "text") && yyjson_is_str(t))
            total += yyjson_get_len(t);
    }
    char *s = malloc(total + 1), *w = s;
    yyjson_arr_foreach(c, idx, max, b)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(b, "type"));
        if (!type)
            continue;
        yyjson_val *t = yyjson_obj_get(b, "text");
        if (!strcmp(type, "text") && yyjson_is_str(t) && s) {
            memcpy(w, yyjson_get_str(t), yyjson_get_len(t));
            w += yyjson_get_len(t);
        } else if (!strcmp(type, "tool_use")) {
            yyjson_mut_val *x = yyjson_mut_arr_add_obj(q, calls);
            yyjson_val *id = yyjson_obj_get(b, "id");
            if (yyjson_is_str(id))
                yyjson_mut_obj_add_val(q, x, "id", yyjson_val_mut_copy(q, id));
            yyjson_mut_obj_add_str(q, x, "type", "function");
            yyjson_mut_val *f = yyjson_mut_obj_add_obj(q, x, "function");
            yyjson_val *nm = yyjson_obj_get(b, "name");
            if (nm)
                yyjson_mut_obj_add_val(q, f, "name",
                                       yyjson_val_mut_copy(q, nm));
            yyjson_val *in = yyjson_obj_get(b, "input");
            yyjson_mut_obj_add_val(q, f, "arguments",
                                   in ? yyjson_val_mut_copy(q, in)
                                      : yyjson_mut_obj(q));
        }
    }
    yyjson_mut_obj_add_strncpy(q, m, "content", s ? s : "",
                               s ? (size_t)(w - s) : 0);
    free(s);
    if (yyjson_mut_arr_size(calls))
        yyjson_mut_obj_add_val(q, m, "tool_calls", calls);
}

/* The tools a client gives, as OpenAI's functions; Anthropic's own tools
   (a type other than custom: its web search, its computer use) are run by
   Anthropic and cannot be here. */
static int add_tools(yyjson_mut_doc *q, yyjson_mut_val *qo, yyjson_val *tools,
                     char *why, size_t why_len)
{
    if (!yyjson_is_arr(tools) || !yyjson_arr_size(tools))
        return 0;
    yyjson_mut_val *out = yyjson_mut_obj_add_arr(q, qo, "tools");
    size_t idx, max;
    yyjson_val *t;
    yyjson_arr_foreach(tools, idx, max, t)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
        if (type && strcmp(type, "custom")) {
            snprintf(why, why_len,
                     "tools[%zu] is of type '%s', one Anthropic runs itself: "
                     "only the client's own tools are called here",
                     idx, type);
            return -1;
        }
        yyjson_mut_val *x = yyjson_mut_arr_add_obj(q, out);
        yyjson_mut_obj_add_str(q, x, "type", "function");
        yyjson_mut_val *f = yyjson_mut_obj_add_obj(q, x, "function");
        static const char *const keep[] = {"name", "description"};
        for (int i = 0; i < 2; i++) {
            yyjson_val *v = yyjson_obj_get(t, keep[i]);
            if (v)
                yyjson_mut_obj_add_val(q, f, keep[i],
                                       yyjson_val_mut_copy(q, v));
        }
        yyjson_val *s = yyjson_obj_get(t, "input_schema");
        if (s)
            yyjson_mut_obj_add_val(q, f, "parameters",
                                   yyjson_val_mut_copy(q, s));
    }
    return 0;
}

static void add_choice(yyjson_mut_doc *q, yyjson_mut_val *qo, yyjson_val *tc)
{
    const char *type = yyjson_get_str(yyjson_obj_get(tc, "type"));
    if (!type)
        return;
    if (!strcmp(type, "auto") || !strcmp(type, "none"))
        yyjson_mut_obj_add_str(q, qo, "tool_choice", type);
    else if (!strcmp(type, "any"))
        yyjson_mut_obj_add_str(q, qo, "tool_choice", "required");
    else if (!strcmp(type, "tool")) {
        yyjson_mut_val *c = yyjson_mut_obj_add_obj(q, qo, "tool_choice");
        yyjson_mut_obj_add_str(q, c, "type", "function");
        yyjson_mut_val *f = yyjson_mut_obj_add_obj(q, c, "function");
        yyjson_val *nm = yyjson_obj_get(tc, "name");
        if (nm)
            yyjson_mut_obj_add_val(q, f, "name", yyjson_val_mut_copy(q, nm));
    }
    yyjson_val *dp = yyjson_obj_get(tc, "disable_parallel_tool_use");
    if (yyjson_is_bool(dp) && yyjson_get_bool(dp))
        yyjson_mut_obj_add_bool(q, qo, "parallel_tool_calls", false);
}

/* thinking: off unless asked; enabled with its budget, or adaptive */
static void add_thinking(yyjson_mut_doc *q, yyjson_mut_val *qo, yyjson_val *th)
{
    const char *type = yyjson_get_str(yyjson_obj_get(th, "type"));
    int on = type && (!strcmp(type, "enabled") || !strcmp(type, "adaptive"));
    yyjson_mut_val *kw = yyjson_mut_obj_add_obj(q, qo, "chat_template_kwargs");
    yyjson_mut_obj_add_bool(q, kw, "enable_thinking", on);
    yyjson_val *bt = yyjson_obj_get(th, "budget_tokens");
    if (on && yyjson_is_int(bt) && yyjson_get_sint(bt) > 0)
        yyjson_mut_obj_add_int(q, qo, "reasoning_budget_tokens",
                               yyjson_get_sint(bt));
}

char *an_request_body(yyjson_val *root, size_t *n, char *why, size_t why_len)
{
    yyjson_mut_doc *q = yyjson_mut_doc_new(NULL);
    if (!q) {
        snprintf(why, why_len, "out of memory");
        return NULL;
    }
    yyjson_mut_val *qo = yyjson_mut_obj(q);
    yyjson_mut_doc_set_root(q, qo);
    yyjson_val *mt = yyjson_obj_get(root, "max_tokens");
    if (mt) /* the reasoning counted, as Anthropic counts it */
        yyjson_mut_obj_add_val(q, qo, "max_completion_tokens",
                               yyjson_val_mut_copy(q, mt));
    static const char *const same[] = {"temperature", "top_p", "top_k"};
    for (int i = 0; i < 3; i++) {
        yyjson_val *v = yyjson_obj_get(root, same[i]);
        if (v)
            yyjson_mut_obj_add_val(q, qo, same[i], yyjson_val_mut_copy(q, v));
    }
    yyjson_val *ss = yyjson_obj_get(root, "stop_sequences");
    if (ss)
        yyjson_mut_obj_add_val(q, qo, "stop", yyjson_val_mut_copy(q, ss));
    add_thinking(q, qo, yyjson_obj_get(root, "thinking"));
    add_choice(q, qo, yyjson_obj_get(root, "tool_choice"));
    int bad = add_tools(q, qo, yyjson_obj_get(root, "tools"), why, why_len);

    yyjson_mut_val *out = yyjson_mut_obj_add_arr(q, qo, "messages");
    yyjson_val *sys = yyjson_obj_get(root, "system");
    if (!bad && sys) {
        yyjson_mut_val *t;
        if (text_of(q, sys, &t) != 0) {
            snprintf(why, why_len, "'system' is a string or text blocks");
            bad = -1;
        } else {
            yyjson_mut_val *m = yyjson_mut_arr_add_obj(q, out);
            yyjson_mut_obj_add_str(q, m, "role", "system");
            yyjson_mut_obj_add_val(q, m, "content", t);
        }
    }
    yyjson_val *msgs = yyjson_obj_get(root, "messages");
    if (!bad && !yyjson_is_arr(msgs)) {
        snprintf(why, why_len, "'messages' must be an array");
        bad = -1;
    }
    size_t idx, max;
    yyjson_val *m;
    yyjson_arr_foreach(msgs, idx, max, m)
    {
        if (bad)
            break;
        const char *role = yyjson_get_str(yyjson_obj_get(m, "role"));
        yyjson_val *c = yyjson_obj_get(m, "content");
        if (role && !strcmp(role, "user"))
            bad = user_message(q, out, c, why, why_len);
        else if (role && !strcmp(role, "assistant"))
            assistant_message(q, out, c);
        else if (role && !strcmp(role, "system")) {
            /* Claude Code's: request.c joins it to the system message */
            yyjson_mut_val *t;
            if (text_of(q, c, &t) != 0) {
                snprintf(why, why_len,
                         "messages[%zu]: a system message's "
                         "content is a string or text blocks",
                         idx);
                bad = -1;
            } else {
                yyjson_mut_val *x = yyjson_mut_arr_add_obj(q, out);
                yyjson_mut_obj_add_str(q, x, "role", "system");
                yyjson_mut_obj_add_val(q, x, "content", t);
            }
        } else {
            snprintf(why, why_len,
                     "messages[%zu]: the role is user, "
                     "assistant or system",
                     idx);
            bad = -1;
        }
    }
    char *body = bad ? NULL : yyjson_mut_write(q, 0, n);
    if (!bad && !body)
        snprintf(why, why_len, "out of memory");
    yyjson_mut_doc_free(q);
    return body;
}
