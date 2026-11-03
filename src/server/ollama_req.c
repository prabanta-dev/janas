/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * ollama_req.c - a request of Ollama's /api/chat or /api/generate as the
 * body of OpenAI's /chat/completions or /completions, which request.c then
 * reads into a job: the messages with their pictures (base64, made data:
 * URLs) and tool calls (their arguments an object, which request.c takes),
 * the tools, think, format and the options OpenAI has under its own names.
 * Kept apart from the HTTP side so that tests/test_server_ollama.c can try
 * it on its own.
 */
#include "ollama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The options OpenAI's request has under the same names. */
static void add_options(yyjson_mut_doc *q, yyjson_mut_val *qo, yyjson_val *opts)
{
    static const char *const same[] = {"top_p",
                                       "top_k",
                                       "min_p",
                                       "seed",
                                       "stop",
                                       "presence_penalty",
                                       "frequency_penalty"};
    if (!yyjson_is_obj(opts))
        return;
    for (size_t i = 0; i < sizeof same / sizeof same[0]; i++) {
        yyjson_val *v = yyjson_obj_get(opts, same[i]);
        if (v)
            yyjson_mut_obj_add_val(q, qo, same[i], yyjson_val_mut_copy(q, v));
    }
    yyjson_val *t = yyjson_obj_get(opts, "temperature");
    if (yyjson_is_num(t)) { /* Ollama takes more than OpenAI's 2 */
        double x = yyjson_get_num(t);
        yyjson_mut_obj_add_real(q, qo, "temperature", x > 2 ? 2 : x);
    }
}

/* think and format into OpenAI's fields. */
static void add_think_format(yyjson_mut_doc *q, yyjson_mut_val *qo,
                             yyjson_val *root)
{
    yyjson_val *th = yyjson_obj_get(root, "think");
    if (yyjson_is_bool(th) || yyjson_is_str(th)) {
        yyjson_mut_val *kw =
            yyjson_mut_obj_add_obj(q, qo, "chat_template_kwargs");
        yyjson_mut_obj_add_bool(q, kw, "enable_thinking",
                                yyjson_is_str(th) || yyjson_get_bool(th));
    }
    yyjson_val *f = yyjson_obj_get(root, "format");
    if (yyjson_is_str(f) && !strcmp(yyjson_get_str(f), "json")) {
        yyjson_mut_val *rf = yyjson_mut_obj_add_obj(q, qo, "response_format");
        yyjson_mut_obj_add_str(q, rf, "type", "json_object");
    } else if (yyjson_is_obj(f)) {
        yyjson_mut_val *rf = yyjson_mut_obj_add_obj(q, qo, "response_format");
        yyjson_mut_obj_add_str(q, rf, "type", "json_schema");
        yyjson_mut_val *js = yyjson_mut_obj_add_obj(q, rf, "json_schema");
        yyjson_mut_obj_add_str(q, js, "name", "answer");
        yyjson_mut_obj_add_val(q, js, "schema", yyjson_val_mut_copy(q, f));
    }
}

/* A message's content: its text, after its pictures (base64, as Ollama
   sends them) when it has any. */
static yyjson_mut_val *content(yyjson_mut_doc *q, yyjson_val *text,
                               yyjson_val *images)
{
    yyjson_mut_val *t = yyjson_is_str(text) ? yyjson_val_mut_copy(q, text)
                                            : yyjson_mut_str(q, "");
    if (!yyjson_is_arr(images) || !yyjson_arr_size(images))
        return t;
    yyjson_mut_val *parts = yyjson_mut_arr(q);
    size_t idx, max;
    yyjson_val *im;
    yyjson_arr_foreach(images, idx, max, im)
    {
        const char *b64 = yyjson_get_str(im);
        size_t n = yyjson_get_len(im);
        char *url = b64 ? malloc(n + 32) : NULL;
        if (!url)
            continue; /* not a string: the reader refuses the part */
        int k = snprintf(url, n + 32, "data:image/png;base64,");
        memcpy(url + k, b64, n);
        yyjson_mut_val *p = yyjson_mut_arr_add_obj(q, parts);
        yyjson_mut_obj_add_str(q, p, "type", "image_url");
        yyjson_mut_val *iu = yyjson_mut_obj_add_obj(q, p, "image_url");
        yyjson_mut_obj_add_strncpy(q, iu, "url", url, (size_t)k + n);
        free(url);
    }
    yyjson_mut_val *p = yyjson_mut_arr_add_obj(q, parts);
    yyjson_mut_obj_add_str(q, p, "type", "text");
    yyjson_mut_obj_add_val(q, p, "text", t);
    return parts;
}

/* Ollama's messages into OpenAI's. */
static void add_messages(yyjson_mut_doc *q, yyjson_mut_val *qo,
                         yyjson_val *msgs)
{
    yyjson_mut_val *out = yyjson_mut_obj_add_arr(q, qo, "messages");
    size_t idx, max;
    yyjson_val *m;
    yyjson_arr_foreach(msgs, idx, max, m)
    {
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(q, out);
        yyjson_val *role = yyjson_obj_get(m, "role");
        if (role)
            yyjson_mut_obj_add_val(q, o, "role", yyjson_val_mut_copy(q, role));
        yyjson_mut_obj_add_val(q, o, "content",
                               content(q, yyjson_obj_get(m, "content"),
                                       yyjson_obj_get(m, "images")));
        yyjson_val *tc = yyjson_obj_get(m, "tool_calls");
        if (!yyjson_is_arr(tc))
            continue;
        yyjson_mut_val *calls = yyjson_mut_obj_add_arr(q, o, "tool_calls");
        size_t ci, cm;
        yyjson_val *c;
        yyjson_arr_foreach(tc, ci, cm, c)
        {
            yyjson_mut_val *x = yyjson_mut_arr_add_obj(q, calls);
            yyjson_mut_obj_add_str(q, x, "type", "function");
            yyjson_val *fn = yyjson_obj_get(c, "function");
            if (fn)
                yyjson_mut_obj_add_val(q, x, "function",
                                       yyjson_val_mut_copy(q, fn));
        }
    }
}

/* /api/generate's prompt as a conversation: its system message, then the
   prompt as the user's, with its pictures. */
static void add_prompt_messages(yyjson_mut_doc *q, yyjson_mut_val *qo,
                                yyjson_val *root)
{
    yyjson_mut_val *out = yyjson_mut_obj_add_arr(q, qo, "messages");
    yyjson_val *sys = yyjson_obj_get(root, "system");
    if (yyjson_is_str(sys) && yyjson_get_len(sys)) {
        yyjson_mut_val *o = yyjson_mut_arr_add_obj(q, out);
        yyjson_mut_obj_add_str(q, o, "role", "system");
        yyjson_mut_obj_add_val(q, o, "content", yyjson_val_mut_copy(q, sys));
    }
    yyjson_mut_val *o = yyjson_mut_arr_add_obj(q, out);
    yyjson_mut_obj_add_str(q, o, "role", "user");
    yyjson_mut_obj_add_val(q, o, "content",
                           content(q, yyjson_obj_get(root, "prompt"),
                                   yyjson_obj_get(root, "images")));
}

char *ol_request_body(yyjson_val *root, int chat_route, int chat, size_t *n)
{
    yyjson_mut_doc *q = yyjson_mut_doc_new(NULL);
    if (!q)
        return NULL;
    yyjson_mut_val *qo = yyjson_mut_obj(q);
    yyjson_mut_doc_set_root(q, qo);
    add_options(q, qo, yyjson_obj_get(root, "options"));
    if (chat) {
        add_think_format(q, qo, root);
        yyjson_val *tools = yyjson_obj_get(root, "tools");
        if (yyjson_is_arr(tools))
            yyjson_mut_obj_add_val(q, qo, "tools",
                                   yyjson_val_mut_copy(q, tools));
        if (chat_route)
            add_messages(q, qo, yyjson_obj_get(root, "messages"));
        else
            add_prompt_messages(q, qo, root);
    } else {
        yyjson_val *p = yyjson_obj_get(root, "prompt");
        yyjson_val *sx = yyjson_obj_get(root, "suffix");
        yyjson_mut_obj_add_val(q, qo, "prompt",
                               p ? yyjson_val_mut_copy(q, p)
                                 : yyjson_mut_str(q, ""));
        if (sx)
            yyjson_mut_obj_add_val(q, qo, "suffix", yyjson_val_mut_copy(q, sx));
    }
    char *body = yyjson_mut_write(q, 0, n);
    yyjson_mut_doc_free(q);
    return body;
}
