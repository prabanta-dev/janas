/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * responses_in.c - a request to /responses into a job for the engine: its
 * input items after the history it names (previous_response_id or
 * conversation), instructions, tools, the form of the text, the reasoning.
 */
#include "request.h"
#include "responses.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void resp_item_id(yyjson_mut_doc *d, yyjson_mut_val *item)
{
    const char *type = yyjson_mut_get_str(yyjson_mut_obj_get(item, "type"));
    if (!type) {
        type = "message";
        yyjson_mut_obj_add_str(d, item, "type", "message");
    }
    if (!yyjson_mut_obj_get(item, "id") ||
        yyjson_mut_is_null(yyjson_mut_obj_get(item, "id"))) {
        const char *p = !strcmp(type, "message")                ? "msg_"
                        : !strcmp(type, "function_call")        ? "fc_"
                        : !strcmp(type, "function_call_output") ? "fco_"
                        : !strcmp(type, "reasoning")            ? "rs_"
                        : !strcmp(type, "compaction")           ? "cmp_"
                                                                : "item_";
        char id[64];
        srv_new_id(id, sizeof(id), p);
        yyjson_mut_obj_remove_key(item, "id");
        yyjson_mut_obj_add_strcpy(d, item, "id", id);
    }
    if (!strcmp(type, "function_call") &&
        !yyjson_mut_obj_get(item, "call_id")) {
        char id[64];
        srv_new_id(id, sizeof(id), "call_");
        yyjson_mut_obj_add_strcpy(d, item, "call_id", id);
    }
}

/* The text of a content: a string, or its text parts joined. NULL with err
   set when it holds something else (an image, a file). */
static char *content_text(yyjson_val *c, size_t *n, char *err, size_t el)
{
    if (!c || yyjson_is_null(c)) {
        *n = 0;
        return srv_dup_n("", 0);
    }
    if (yyjson_is_str(c)) {
        *n = yyjson_get_len(c);
        return srv_dup_n(yyjson_get_str(c), *n);
    }
    if (!yyjson_is_arr(c)) {
        snprintf(err, el, "a content is a string or an array of parts");
        return NULL;
    }
    size_t total = 0, idx, max;
    yyjson_val *p;
    yyjson_arr_foreach(c, idx, max, p)
    {
        const char *t = yyjson_get_str(yyjson_obj_get(p, "type"));
        if (!t || (strcmp(t, "input_text") && strcmp(t, "output_text") &&
                   strcmp(t, "text") && strcmp(t, "refusal"))) {
            snprintf(err, el, "only text parts are supported (got '%s')",
                     t ? t : "?");
            return NULL;
        }
        total += yyjson_get_len(yyjson_obj_get(p, "text"));
    }
    char *s = malloc(total + 1);
    if (!s) {
        snprintf(err, el, "out of memory");
        return NULL;
    }
    size_t w = 0;
    yyjson_arr_foreach(c, idx, max, p)
    {
        yyjson_val *tx = yyjson_obj_get(p, "text");
        memcpy(s + w, yyjson_get_str(tx) ? yyjson_get_str(tx) : "",
               yyjson_get_len(tx));
        w += yyjson_get_len(tx);
    }
    s[w] = 0;
    *n = w;
    return s;
}

/* A message appended to the job; NULL on memory. */
struct srv_msg *resp_push(struct srv_job *j, int32_t role, char *text, size_t n)
{
    struct srv_msg *t = realloc(j->msg, (size_t)(j->n_msg + 1) * sizeof(*t));
    if (!t) {
        free(text);
        return NULL;
    }
    j->msg = t;
    struct srv_msg *m = &j->msg[j->n_msg++];
    *m = (struct srv_msg){.role = role, .text = text, .n = n};
    return m;
}

/* Text added to the system message, which is the job's first. */
static int to_system(struct srv_job *j, const char *s, size_t n)
{
    if (j->n_msg == 0 || j->msg[0].role != JANAS_LLM_ROLE_SYSTEM) {
        struct srv_msg *m = resp_push(j, JANAS_LLM_ROLE_SYSTEM, NULL, 0);
        if (!m)
            return -1;
        memmove(j->msg + 1, j->msg, (size_t)(j->n_msg - 1) * sizeof(*m));
        j->msg[0] = (struct srv_msg){.role = JANAS_LLM_ROLE_SYSTEM};
    }
    struct srv_msg *sys = &j->msg[0];
    char *t = realloc(sys->text, sys->n + n + 3);
    if (!t)
        return -1;
    sys->text = t;
    if (sys->n) {
        memcpy(t + sys->n, "\n\n", 2);
        sys->n += 2;
    }
    memcpy(t + sys->n, s, n);
    sys->n += n;
    t[sys->n] = 0;
    return 0;
}

int resp_add_call(struct srv_msg *m, const char *name, char *args)
{
    char **nn = realloc(m->call_name, (size_t)(m->n_calls + 1) * sizeof(*nn));
    if (nn)
        m->call_name = nn;
    char **na = realloc(m->call_args, (size_t)(m->n_calls + 1) * sizeof(*na));
    if (na)
        m->call_args = na;
    char *nd = srv_dup_n(name, strlen(name));
    if (!nn || !na || !nd || !args) {
        free(nd);
        free(args);
        return -1;
    }
    m->call_name[m->n_calls] = nd;
    m->call_args[m->n_calls++] = args;
    return 0;
}

int resp_items_to_msgs(struct srv_job *j, yyjson_val *items, char *err,
                       size_t el)
{
    size_t idx, max;
    yyjson_val *it;
    yyjson_arr_foreach(items, idx, max, it)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(it, "type"));
        if (!yyjson_is_obj(it)) {
            snprintf(err, el, "input[%zu] is not an item", idx);
            return -1;
        }
        if (!type || !strcmp(type, "message")) {
            const char *role = yyjson_get_str(yyjson_obj_get(it, "role"));
            size_t n;
            char *t = content_text(yyjson_obj_get(it, "content"), &n, err, el);
            if (!t)
                return -1;
            int32_t rid =
                !role                        ? -1
                : !strcmp(role, "user")      ? JANAS_LLM_ROLE_USER
                : !strcmp(role, "assistant") ? JANAS_LLM_ROLE_ASSISTANT
                : !strcmp(role, "system") || !strcmp(role, "developer")
                    ? JANAS_LLM_ROLE_SYSTEM
                    : -1;
            if (rid < 0) {
                free(t);
                snprintf(err, el, "input[%zu]: unknown role '%s'", idx,
                         role ? role : "");
                return -1;
            }
            if (rid == JANAS_LLM_ROLE_SYSTEM) {
                int bad = to_system(j, t, n);
                free(t);
                if (bad)
                    goto oom;
                continue;
            }
            if (!resp_push(j, rid, t, n))
                goto oom;
        } else if (!strcmp(type, "function_call")) {
            const char *name = yyjson_get_str(yyjson_obj_get(it, "name"));
            yyjson_val *a = yyjson_obj_get(it, "arguments");
            if (!name) {
                snprintf(err, el, "input[%zu]: a function call needs a name",
                         idx);
                return -1;
            }
            /* the calls go with the assistant's message before them */
            struct srv_msg *m = j->n_msg ? &j->msg[j->n_msg - 1] : NULL;
            if (!m || m->role != JANAS_LLM_ROLE_ASSISTANT)
                m = resp_push(j, JANAS_LLM_ROLE_ASSISTANT, srv_dup_n("", 0), 0);
            char *args = yyjson_is_str(a)
                             ? srv_dup_n(yyjson_get_str(a), yyjson_get_len(a))
                             : srv_dup_n("{}", 2);
            if (!m || resp_add_call(m, name, args) != 0)
                goto oom;
        } else if (!strcmp(type, "function_call_output")) {
            size_t n;
            char *t = content_text(yyjson_obj_get(it, "output"), &n, err, el);
            if (!t)
                return -1;
            if (!resp_push(j, JANAS_LLM_ROLE_TOOL, t, n))
                goto oom;
        } else if (!strcmp(type, "compaction")) {
            const char *s =
                yyjson_get_str(yyjson_obj_get(it, "encrypted_content"));
            char buf[64];
            snprintf(buf, sizeof(buf), "Summary of the conversation so far:\n");
            size_t a = strlen(buf), b = s ? strlen(s) : 0;
            char *t = malloc(a + b + 1);
            if (!t)
                goto oom;
            memcpy(t, buf, a);
            memcpy(t + a, s ? s : "", b + 1);
            if (!resp_push(j, JANAS_LLM_ROLE_USER, t, a + b))
                goto oom;
        } else if (!strcmp(type, "mcp_approval_request") ||
                   !strcmp(type, "mcp_call")) {
            /* a call to an MCP server's tool, as the model named it; a
               call run after its approval has its request before it */
            const char *label =
                yyjson_get_str(yyjson_obj_get(it, "server_label"));
            const char *tool = yyjson_get_str(yyjson_obj_get(it, "name"));
            const char *a = yyjson_get_str(yyjson_obj_get(it, "arguments"));
            if (!label || !tool) {
                snprintf(err, el,
                         "input[%zu]: an MCP item needs a "
                         "server_label and a name",
                         idx);
                return -1;
            }
            int approved =
                yyjson_is_str(yyjson_obj_get(it, "approval_request_id"));
            if (!strcmp(type, "mcp_approval_request") || !approved) {
                char name[256];
                snprintf(name, sizeof(name), "%s__%s", label, tool);
                struct srv_msg *m = j->n_msg ? &j->msg[j->n_msg - 1] : NULL;
                if (!m || m->role != JANAS_LLM_ROLE_ASSISTANT)
                    m = resp_push(j, JANAS_LLM_ROLE_ASSISTANT, srv_dup_n("", 0),
                                  0);
                if (!m || resp_add_call(m, name,
                                        srv_dup_n(a ? a : "{}",
                                                  strlen(a ? a : "{}"))) != 0)
                    goto oom;
            }
            if (!strcmp(type, "mcp_call")) {
                /* its output, or what went wrong */
                const char *out = yyjson_get_str(yyjson_obj_get(it, "output"));
                yyjson_val *e = yyjson_obj_get(it, "error");
                const char *why = yyjson_get_str(yyjson_obj_get(e, "message"));
                if (!why)
                    why = yyjson_get_str(yyjson_obj_get(
                        yyjson_arr_get_first(yyjson_obj_get(e, "content")),
                        "text"));
                if (!why)
                    why = yyjson_get_str(e);
                char *t;
                if (out) {
                    t = srv_dup_n(out, strlen(out));
                } else {
                    size_t n = strlen(why ? why : "failed") + 8;
                    t = malloc(n);
                    if (t)
                        snprintf(t, n, "Error: %s", why ? why : "failed");
                }
                if (!t || !resp_push(j, JANAS_LLM_ROLE_TOOL, t, strlen(t)))
                    goto oom;
            }
        } else if (!strcmp(type, "mcp_approval_response")) {
            /* approved: the call ran in this request (its output is on the
               job), or in the one that answered it (an mcp_call follows);
               refused: the model is told so */
            const char *rid =
                yyjson_get_str(yyjson_obj_get(it, "approval_request_id"));
            const char *t = NULL;
            if (!yyjson_get_bool(yyjson_obj_get(it, "approve")))
                t = "The user did not allow this call: it was not run.";
            for (int k = 0; !t && rid && k < j->n_mcp; k++)
                if (!strcmp(j->mcp_id[k], rid))
                    t = j->mcp_out[k];
            if (t && !resp_push(j, JANAS_LLM_ROLE_TOOL, srv_dup_n(t, strlen(t)),
                                strlen(t)))
                goto oom;
        } else if (!strcmp(type, "mcp_list_tools")) {
            continue; /* the tools are the model's through the request */
        } else if (strcmp(type, "reasoning") != 0) {
            snprintf(err, el,
                     "input[%zu]: items of type '%s' are not supported here",
                     idx, type);
            return -1;
        }
    }
    return 0;
oom:
    snprintf(err, el, "out of memory");
    return -1;
}

/* The history a request names: a stored response's context and output, or
   a conversation's items. Into arr; 0, or an answer given. */
static int history(struct srv_req *r, struct resp *x, yyjson_val *root,
                   yyjson_mut_doc *d, yyjson_mut_val *arr, int *ret)
{
    const char *prev =
        yyjson_get_str(yyjson_obj_get(root, "previous_response_id"));
    yyjson_val *cv = yyjson_obj_get(root, "conversation");
    const char *conv = yyjson_is_str(cv)
                           ? yyjson_get_str(cv)
                           : yyjson_get_str(yyjson_obj_get(cv, "id"));
    if (prev && conv) {
        *ret = srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "'previous_response_id' and 'conversation' "
                               "cannot go together.");
        return -1;
    }
    char *items = NULL, *resp = NULL;
    if (prev) {
        int rc = srv_store_get(r->srv->store, RESP_KIND, prev, &resp, &items);
        if (rc != 0) {
            *ret = srv_reply_error(r, 404, "invalid_request_error",
                                   "previous_response_not_found",
                                   "No response has the id '%s'.", prev);
            return -1;
        }
    } else if (conv) {
        if (strlen(conv) >= sizeof(x->conv) ||
            !(items = conv_items(r->srv, conv))) {
            *ret = srv_reply_error(r, 404, "invalid_request_error",
                                   "conversation_not_found",
                                   "No conversation has the id '%s'.", conv);
            return -1;
        }
        snprintf(x->conv, sizeof(x->conv), "%s", conv);
    }
    yyjson_doc *di = items ? yyjson_read(items, strlen(items), 0) : NULL;
    yyjson_doc *dr = resp ? yyjson_read(resp, strlen(resp), 0) : NULL;
    free(items);
    free(resp);
    size_t idx, max;
    yyjson_val *it;
    yyjson_arr_foreach(yyjson_doc_get_root(di), idx, max, it)
        yyjson_mut_arr_append(arr, yyjson_val_mut_copy(d, it));
    yyjson_arr_foreach(yyjson_obj_get(yyjson_doc_get_root(dr), "output"), idx,
                       max, it)
        yyjson_mut_arr_append(arr, yyjson_val_mut_copy(d, it));
    yyjson_doc_free(di);
    yyjson_doc_free(dr);
    return 0;
}

/* What the Response object repeats of the request. */
static int make_echo(struct srv_req *r, struct resp *x, yyjson_val *root)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    static const char *const keys[] = {"instructions",
                                       "max_output_tokens",
                                       "previous_response_id",
                                       "temperature",
                                       "top_p",
                                       "tools",
                                       "tool_choice",
                                       "top_logprobs",
                                       "parallel_tool_calls"};
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        yyjson_val *v = yyjson_obj_get(root, keys[i]);
        yyjson_mut_obj_add_val(
            d, o, keys[i], v ? yyjson_val_mut_copy(d, v) : yyjson_mut_null(d));
    }
    struct srv_job *j = x->j;
    yyjson_mut_obj_remove_key(o, "temperature");
    yyjson_mut_obj_add_real(d, o, "temperature", j->p.temperature);
    yyjson_mut_obj_remove_key(o, "top_p");
    yyjson_mut_obj_add_real(d, o, "top_p", j->p.top_p);
    if (!yyjson_obj_get(root, "tools")) {
        yyjson_mut_obj_remove_key(o, "tools");
        yyjson_mut_obj_add_arr(d, o, "tools");
    }
    if (!yyjson_obj_get(root, "tool_choice")) {
        yyjson_mut_obj_remove_key(o, "tool_choice");
        yyjson_mut_obj_add_str(d, o, "tool_choice", "auto");
    }
    if (!yyjson_obj_get(root, "parallel_tool_calls")) {
        yyjson_mut_obj_remove_key(o, "parallel_tool_calls");
        yyjson_mut_obj_add_bool(d, o, "parallel_tool_calls", 1);
    }
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    yyjson_mut_obj_add_val(d, o, "metadata",
                           yyjson_is_obj(md) ? yyjson_val_mut_copy(d, md)
                                             : yyjson_mut_obj(d));
    yyjson_mut_obj_add_str(d, o, "model", r->srv->cfg.model_id);
    yyjson_mut_val *rs = yyjson_mut_obj_add_obj(d, o, "reasoning");
    const char *eff = yyjson_get_str(
        yyjson_obj_get(yyjson_obj_get(root, "reasoning"), "effort"));
    if (eff)
        yyjson_mut_obj_add_strcpy(d, rs, "effort", eff);
    else
        yyjson_mut_obj_add_null(d, rs, "effort");
    yyjson_mut_obj_add_null(d, rs, "summary");
    yyjson_val *tx = yyjson_obj_get(root, "text");
    if (yyjson_is_obj(tx)) {
        yyjson_mut_obj_add_val(d, o, "text", yyjson_val_mut_copy(d, tx));
    } else {
        yyjson_mut_val *t = yyjson_mut_obj_add_obj(d, o, "text");
        yyjson_mut_val *f = yyjson_mut_obj_add_obj(d, t, "format");
        yyjson_mut_obj_add_str(d, f, "type", "text");
    }
    yyjson_mut_obj_add_bool(d, o, "store", x->store);
    yyjson_mut_obj_add_bool(d, o, "background", x->background);
    yyjson_mut_obj_add_str(d, o, "truncation", "disabled");
    yyjson_mut_obj_add_null(d, o, "user");
    if (x->conv[0]) {
        yyjson_mut_val *c = yyjson_mut_obj_add_obj(d, o, "conversation");
        yyjson_mut_obj_add_strcpy(d, c, "id", x->conv);
    } else {
        yyjson_mut_obj_add_null(d, o, "conversation");
    }
    size_t n;
    x->echo = yyjson_mut_write(d, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    yyjson_mut_doc_free(d);
    return x->echo ? 0 : -1;
}

/* tools of the Responses API: functions, flat ({type, name, ...}), for the
   model; those of type mcp are reached in responses_mcp.c */
static int read_tools(struct srv_req *r, struct srv_job *j, yyjson_val *root,
                      int *ret)
{
    yyjson_val *tools = yyjson_obj_get(root, "tools");
    if (yyjson_is_arr(tools) && yyjson_arr_size(tools) > 0) {
        size_t idx, max, fns = 0;
        yyjson_val *t;
        yyjson_arr_foreach(tools, idx, max, t)
        {
            const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
            if (!type ||
                (strcmp(type, "function") != 0 && strcmp(type, "mcp") != 0)) {
                *ret = srv_reply_error(r, 400, "invalid_request_error",
                                       "unsupported_value",
                                       "tools[%zu]: only functions and MCP "
                                       "servers can be called here (got '%s').",
                                       idx, type ? type : "?");
                return -1;
            }
            fns += !strcmp(type, "function");
        }
        if (fns == yyjson_arr_size(tools)) {
            if (!(j->tools = srv_json_text(tools, &j->n_tools)))
                return -1;
        } else if (fns) {
            yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
            yyjson_mut_val *a = yyjson_mut_arr(d);
            yyjson_arr_foreach(tools, idx, max, t)
            {
                if (!strcmp(yyjson_get_str(yyjson_obj_get(t, "type")),
                            "function"))
                    yyjson_mut_arr_append(a, yyjson_val_mut_copy(d, t));
            }
            j->tools = yyjson_mut_val_write(a, 0, &j->n_tools);
            yyjson_mut_doc_free(d);
            if (!j->tools)
                return -1;
        }
    }
    yyjson_val *pc = yyjson_obj_get(root, "parallel_tool_calls");
    if (pc && yyjson_is_bool(pc))
        j->parallel = yyjson_get_bool(pc);
    yyjson_val *tc = yyjson_obj_get(root, "tool_choice");
    const char *s = yyjson_get_str(tc);
    const char *name = yyjson_get_str(yyjson_obj_get(tc, "name"));
    if (s)
        j->tool_choice = !strcmp(s, "none")       ? JANAS_LLM_TOOLS_NONE
                         : !strcmp(s, "required") ? JANAS_LLM_TOOLS_REQUIRED
                                                  : JANAS_LLM_TOOLS_AUTO;
    else if (name && strlen(name) < sizeof(j->tool_name)) {
        j->tool_choice = JANAS_LLM_TOOLS_FUNCTION;
        memcpy(j->tool_name, name, strlen(name) + 1);
    }
    return 0;
}

/* text.format: text, json_object, json_schema {name, schema, strict} */
static int read_format(struct srv_req *r, struct srv_job *j, yyjson_val *root,
                       int *ret)
{
    yyjson_val *f = yyjson_obj_get(yyjson_obj_get(root, "text"), "format");
    const char *type = yyjson_get_str(yyjson_obj_get(f, "type"));
    if (!type || !strcmp(type, "text"))
        return 0;
    if (!strcmp(type, "json_object")) {
        j->format = JANAS_LLM_FORMAT_JSON;
        return 0;
    }
    yyjson_val *schema = yyjson_obj_get(f, "schema");
    if (strcmp(type, "json_schema") != 0 ||
        !(yyjson_is_obj(schema) || yyjson_is_bool(schema))) {
        *ret = srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "'text.format' is text, json_object, or "
                               "json_schema with a schema.");
        return -1;
    }
    j->format = JANAS_LLM_FORMAT_SCHEMA;
    return (j->schema = srv_json_text(schema, &j->n_schema)) ? 0 : -1;
}

static int bad(struct srv_req *r, int *ret, const char *msg)
{
    *ret = srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                           "%s", msg);
    return -1;
}

int resp_read(struct srv_req *r, struct resp *x, int *ret)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        *ret = srv_reply_error(r, 400, "invalid_request_error", "invalid_json",
                               "The body must be a JSON object.");
        return -1;
    }
    int rc = -1;
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    struct srv_job *j = x->j;
    j->kind = JOB_CHAT;
    j->p.thinking = r->srv->cfg.thinking;
    if (srv_job_choices(j, 1, 1) != 0)
        goto oom;
    double v;
    int got;
    if ((got = srv_num(yyjson_obj_get(root, "temperature"), 0, 2, &v)) < 0) {
        bad(r, ret, "'temperature' must be between 0 and 2.");
        goto out;
    }
    if (got)
        j->p.temperature = (float)v;
    if ((got = srv_num(yyjson_obj_get(root, "top_p"), 0, 1, &v)) < 0) {
        bad(r, ret, "'top_p' must be between 0 and 1.");
        goto out;
    }
    if (got)
        j->p.top_p = (float)v;
    if ((got = srv_num(yyjson_obj_get(root, "max_output_tokens"), 1, 1e9, &v)) <
        0) {
        bad(r, ret, "'max_output_tokens' must be 1 or more.");
        goto out;
    }
    if (got)
        j->p.max_reply = (int32_t)v;
    if ((got = srv_num(yyjson_obj_get(root, "top_logprobs"), 0, 20, &v)) < 0) {
        bad(r, ret, "'top_logprobs' is from 0 to 20.");
        goto out;
    }
    if (got) {
        j->p.logprobs = 1;
        j->p.top_logprobs = (int32_t)v;
    }
    const char *eff = yyjson_get_str(
        yyjson_obj_get(yyjson_obj_get(root, "reasoning"), "effort"));
    if (eff) {
        j->p.thinking = !strcmp(eff, "none") || !strcmp(eff, "minimal") ? 0 : 1;
        x->reasoning_asked = 1;
    }
    yyjson_val *st = yyjson_obj_get(root, "store");
    x->store = !st || !yyjson_is_bool(st) || yyjson_get_bool(st); /* default
                                                                     true */
    yyjson_val *bg = yyjson_obj_get(root, "background");
    x->background = bg && yyjson_is_bool(bg) && yyjson_get_bool(bg);
    yyjson_val *sm = yyjson_obj_get(root, "stream");
    x->stream = sm && yyjson_is_bool(sm) && yyjson_get_bool(sm);
    if (x->background && !x->store) {
        bad(r, ret,
            "'background' needs 'store': the response is found "
            "later by its id.");
        goto out;
    }
    if (read_tools(r, j, root, ret) != 0 || read_format(r, j, root, ret) != 0)
        goto out;
    /* the context: the history, then the input, every item with an id */
    yyjson_mut_val *ctx = yyjson_mut_arr(d), *in = yyjson_mut_arr(d);
    if (history(r, x, root, d, ctx, ret) != 0)
        goto out;
    yyjson_val *input = yyjson_obj_get(root, "input");
    if (yyjson_is_str(input)) {
        yyjson_mut_val *m = yyjson_mut_arr_add_obj(d, in);
        yyjson_mut_obj_add_str(d, m, "type", "message");
        yyjson_mut_obj_add_str(d, m, "role", "user");
        yyjson_mut_obj_add_val(d, m, "content", yyjson_val_mut_copy(d, input));
    } else if (yyjson_is_arr(input)) {
        size_t idx, max;
        yyjson_val *it;
        yyjson_arr_foreach(input, idx, max, it)
        {
            if (!yyjson_is_obj(it)) {
                bad(r, ret, "'input' holds items (objects).");
                goto out;
            }
            yyjson_mut_arr_append(in, yyjson_val_mut_copy(d, it));
        }
    } else if (input && !yyjson_is_null(input)) {
        bad(r, ret, "'input' is a string or an array of items.");
        goto out;
    }
    size_t idx, max;
    yyjson_mut_val *it;
    yyjson_mut_arr_foreach(in, idx, max, it)
    {
        resp_item_id(d, it);
        yyjson_mut_arr_append(ctx, yyjson_mut_val_mut_copy(d, it));
    }
    size_t n;
    x->context =
        yyjson_mut_val_write(ctx, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    x->input = yyjson_mut_val_write(in, YYJSON_WRITE_ALLOW_INVALID_UNICODE, &n);
    if (!x->context || !x->input)
        goto oom;
    /* the MCP servers, reached before the messages are made: an approved
       call runs now, and its output is one of them */
    if (resp_mcp_read(r, x, root, ret) != 0)
        goto out;
    /* the messages: the instructions first, then the context */
    const char *ins = yyjson_get_str(yyjson_obj_get(root, "instructions"));
    if (ins && to_system(j, ins, strlen(ins)) != 0)
        goto oom;
    yyjson_doc *cd = yyjson_read(x->context, strlen(x->context), 0);
    char err[256];
    int e = resp_items_to_msgs(j, yyjson_doc_get_root(cd), err, sizeof(err));
    yyjson_doc_free(cd);
    if (e != 0) {
        bad(r, ret, err);
        goto out;
    }
    if (make_echo(r, x, root) != 0)
        goto oom;
    rc = 0;
    goto out;
oom:
    *ret = srv_reply_error(r, 500, "server_error", NULL, "out of memory");
out:
    yyjson_mut_doc_free(d);
    yyjson_doc_free(doc);
    return rc;
}
