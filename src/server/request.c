/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * request.c - the JSON of a request to /chat/completions or /completions
 * into a job for the engine: the messages, or the prompts; the parameters
 * are read in request_opts.c. Kept apart from the HTTP side so that it can
 * be tried on its own: it reads what anybody on the network sends, and
 * tests/test_server_request.c feeds it valid requests and mangled ones.
 */
#include "request.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int srv_perr(struct srv_perr *e, const char *code, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->msg, sizeof(e->msg), fmt, ap);
    va_end(ap);
    e->code = code;
    return 0;
}

char *srv_dup_n(const char *s, size_t n)
{
    char *d = malloc(n + 1);
    if (d) {
        memcpy(d, s, n);
        d[n] = 0;
    }
    return d;
}

int srv_pfail(struct srv_parsed *pr, int ret)
{
    pr->failed = 1;
    pr->ret = ret;
    return -1;
}

int srv_num(yyjson_val *v, double lo, double hi, double *out)
{
    if (!v || yyjson_is_null(v))
        return 0;
    if (!yyjson_is_num(v))
        return -1;
    double x = yyjson_get_num(v);
    if (x < lo || x > hi)
        return -1;
    *out = x;
    return 1;
}

char *srv_json_text(yyjson_val *v, size_t *n)
{
    size_t len = 0;
    char *t = yyjson_val_write(v, YYJSON_WRITE_NOFLAG, &len);
    if (n)
        *n = len;
    return t;
}

void srv_parsed_free(struct srv_parsed *pr)
{
    free(pr->metadata);
    free(pr->messages);
    pr->metadata = pr->messages = NULL;
}

/* The text of a message's content: a string, or the text parts of an array
   joined; NULL content is empty. Anything else (images, audio) refused. */
static int content_text(struct srv_perr *e, struct srv_parsed *pr,
                        yyjson_val *c, int i, char **out, size_t *n)
{
    if (!c || yyjson_is_null(c)) {
        *out = srv_dup_n("", 0);
        *n = 0;
        return *out ? 0 : -1;
    }
    if (yyjson_is_str(c)) {
        *n = yyjson_get_len(c);
        *out = srv_dup_n(yyjson_get_str(c), *n);
        return *out ? 0 : -1;
    }
    if (!yyjson_is_arr(c))
        return srv_pfail(pr, BAD("invalid_value",
                                 "messages[%d].content must be a string or "
                                 "an array of parts.",
                                 i));
    size_t total = 0, idx, max;
    yyjson_val *part;
    yyjson_arr_foreach(c, idx, max, part)
    {
        const char *type = yyjson_get_str(yyjson_obj_get(part, "type"));
        if (!type || (strcmp(type, "text") != 0 && strcmp(type, "refusal")))
            return srv_pfail(pr, BAD("unsupported_content",
                                     "messages[%d]: only text parts are "
                                     "supported (got '%s').",
                                     i, type ? type : "?"));
        total += yyjson_get_len(yyjson_obj_get(part, "text"));
    }
    char *t = malloc(total + 1);
    if (!t)
        return -1;
    size_t w = 0;
    yyjson_arr_foreach(c, idx, max, part)
    {
        yyjson_val *tx = yyjson_obj_get(part, "text");
        memcpy(t + w, yyjson_get_str(tx) ? yyjson_get_str(tx) : "",
               yyjson_get_len(tx));
        w += yyjson_get_len(tx);
    }
    t[w] = 0;
    *out = t;
    *n = w;
    return 0;
}

/* One call into message m: its name and its arguments (JSON text). */
static int add_call(struct srv_perr *e, struct srv_parsed *pr,
                    struct srv_msg *m, yyjson_val *fn, int i)
{
    const char *name = yyjson_get_str(yyjson_obj_get(fn, "name"));
    yyjson_val *args = yyjson_obj_get(fn, "arguments");
    if (!name || !*name)
        return srv_pfail(pr, BAD("invalid_value",
                                 "messages[%d]: a tool call has no function "
                                 "name.",
                                 i));
    char **nn = realloc(m->call_name, (size_t)(m->n_calls + 1) * sizeof(*nn));
    if (!nn)
        return -1;
    m->call_name = nn;
    char **na = realloc(m->call_args, (size_t)(m->n_calls + 1) * sizeof(*na));
    if (!na)
        return -1;
    m->call_args = na;
    char *a = NULL;
    if (!args || yyjson_is_null(args))
        a = srv_dup_n("{}", 2);
    else if (yyjson_is_str(args)) /* what OpenAI sends: JSON as a string */
        a = srv_dup_n(yyjson_get_str(args), yyjson_get_len(args));
    else
        a = srv_json_text(args, NULL);
    char *n = srv_dup_n(name, strlen(name));
    if (!a || !n) {
        free(a);
        free(n);
        return -1;
    }
    m->call_name[m->n_calls] = n;
    m->call_args[m->n_calls] = a;
    m->n_calls++;
    return 0;
}

/* An assistant's calls: tool_calls, or the older function_call. */
static int read_calls(struct srv_perr *e, struct srv_parsed *pr,
                      yyjson_val *msg, struct srv_msg *m, int i)
{
    yyjson_val *tc = yyjson_obj_get(msg, "tool_calls");
    if (tc && yyjson_is_arr(tc)) {
        size_t idx, max;
        yyjson_val *c;
        yyjson_arr_foreach(tc, idx, max, c)
        {
            yyjson_val *fn = yyjson_obj_get(c, "function");
            if (!yyjson_is_obj(fn))
                return srv_pfail(pr, BAD("invalid_value",
                                         "messages[%d].tool_calls[%zu] is "
                                         "not a function call.",
                                         i, idx));
            if (add_call(e, pr, m, fn, i) != 0)
                return -1;
        }
    } else if (tc && !yyjson_is_null(tc)) {
        return srv_pfail(pr,
                         BAD("invalid_value",
                             "messages[%d].tool_calls must be an array.", i));
    }
    yyjson_val *fc = yyjson_obj_get(msg, "function_call");
    if (fc && yyjson_is_obj(fc))
        return add_call(e, pr, m, fc, i);
    return 0;
}

/* messages[] into the job: every system (or developer) message joined into
   one at the start, which is where the chat format keeps it. */
static int read_messages(struct srv_perr *e, struct srv_parsed *pr,
                         yyjson_val *msgs, struct srv_job *j)
{
    if (!yyjson_is_arr(msgs) || yyjson_arr_size(msgs) == 0)
        return srv_pfail(pr, BAD("missing_messages",
                                 "'messages' must be a non-empty array."));
    size_t n = yyjson_arr_size(msgs);
    j->msg = calloc(n + 1, sizeof(*j->msg));
    if (!j->msg)
        return -1;
    /* slot 0: the system message, if any; freed with the job from here */
    j->n_msg = 1;
    struct srv_msg *sys = &j->msg[0];
    sys->role = JANAS_LLM_ROLE_SYSTEM;
    size_t idx, max;
    yyjson_val *m;
    yyjson_arr_foreach(msgs, idx, max, m)
    {
        const char *role = yyjson_get_str(yyjson_obj_get(m, "role"));
        int i = (int)idx;
        if (!role)
            return srv_pfail(
                pr, BAD("invalid_value", "messages[%d] has no role.", i));
        char *t;
        size_t tn;
        if (content_text(e, pr, yyjson_obj_get(m, "content"), i, &t, &tn))
            return -1;
        if (strcmp(role, "system") == 0 || strcmp(role, "developer") == 0) {
            char *s = realloc(sys->text, sys->n + tn + 3);
            if (!s) {
                free(t);
                return -1;
            }
            sys->text = s;
            if (sys->n) {
                memcpy(s + sys->n, "\n\n", 2);
                sys->n += 2;
            }
            memcpy(s + sys->n, t, tn);
            sys->n += tn;
            s[sys->n] = 0;
            free(t);
            continue;
        }
        int32_t role_id =
            strcmp(role, "user") == 0        ? JANAS_LLM_ROLE_USER
            : strcmp(role, "assistant") == 0 ? JANAS_LLM_ROLE_ASSISTANT
            : strcmp(role, "tool") == 0 || strcmp(role, "function") == 0
                ? JANAS_LLM_ROLE_TOOL
                : -1;
        if (role_id < 0) {
            free(t);
            return srv_pfail(pr,
                             BAD("invalid_value",
                                 "messages[%d]: unknown role '%s'.", i, role));
        }
        struct srv_msg *x = &j->msg[j->n_msg++];
        x->role = role_id;
        x->text = t;
        x->n = tn;
        if (role_id == JANAS_LLM_ROLE_ASSISTANT &&
            read_calls(e, pr, m, x, i) != 0)
            return -1;
    }
    if (j->n_msg == 1)
        return srv_pfail(pr, BAD("missing_messages",
                                 "'messages' has only system messages."));
    if (!sys->text) { /* no system message: shift down over its slot */
        memmove(j->msg, j->msg + 1, (size_t)(j->n_msg - 1) * sizeof(*j->msg));
        j->n_msg--;
        memset(&j->msg[j->n_msg], 0, sizeof(j->msg[0]));
    }
    int32_t last = j->msg[j->n_msg - 1].role;
    if (last != JANAS_LLM_ROLE_USER && last != JANAS_LLM_ROLE_TOOL)
        return srv_pfail(pr, BAD("invalid_value",
                                 "The last message must be the user's or a "
                                 "tool's."));
    return 0;
}

/* One prompt: a string, or an array of token ids. */
static int read_prompt(struct srv_perr *e, struct srv_parsed *pr, yyjson_val *p,
                       struct srv_prompt *out)
{
    if (yyjson_is_str(p)) {
        out->n = yyjson_get_len(p);
        out->text = srv_dup_n(yyjson_get_str(p), out->n);
        return out->text ? 0 : -1;
    }
    if (!yyjson_is_arr(p) || yyjson_arr_size(p) == 0)
        return srv_pfail(pr, BAD("invalid_value",
                                 "'prompt' must be a string, an array of "
                                 "strings, or of token ids."));
    size_t n = yyjson_arr_size(p), idx, max;
    out->ids = malloc(n * sizeof(int32_t));
    if (!out->ids)
        return -1;
    yyjson_val *v;
    yyjson_arr_foreach(p, idx, max, v)
    {
        if (!yyjson_is_int(v) || yyjson_get_sint(v) < 0 ||
            yyjson_get_sint(v) > INT32_MAX)
            return srv_pfail(pr, BAD("invalid_value",
                                     "'prompt': a token id is a number of "
                                     "0 or more."));
        out->ids[out->n_ids++] = (int32_t)yyjson_get_sint(v);
    }
    return 0;
}

/* 'prompt': a string, token ids, or a list of either (one reply each). */
static int read_prompts(struct srv_perr *e, struct srv_parsed *pr,
                        yyjson_val *p, struct srv_job *j)
{
    int many = yyjson_is_arr(p) && yyjson_arr_size(p) > 0 &&
               (yyjson_is_str(yyjson_arr_get_first(p)) ||
                yyjson_is_arr(yyjson_arr_get_first(p)));
    size_t n = many ? yyjson_arr_size(p) : 1;
    if (n > 64)
        return srv_pfail(
            pr, BAD("invalid_value", "'prompt' holds at most 64 prompts."));
    if (!p)
        return srv_pfail(pr, BAD("invalid_value", "'prompt' is missing."));
    j->prompts = calloc(n, sizeof(*j->prompts));
    if (!j->prompts)
        return -1;
    j->n_prompts = (int)n;
    if (!many)
        return read_prompt(e, pr, p, &j->prompts[0]);
    size_t idx, max;
    yyjson_val *v;
    yyjson_arr_foreach(p, idx, max, v)
    {
        if (read_prompt(e, pr, v, &j->prompts[idx]) != 0)
            return -1;
    }
    return 0;
}

int srv_parse_request(const char *body, size_t n, int chat, struct srv_job *j,
                      struct srv_parsed *pr, struct srv_perr *e)
{
    memset(pr, 0, sizeof(*pr));
    e->code = NULL;
    e->msg[0] = 0;
    e->oom = 0;
    yyjson_read_err re;
    yyjson_doc *doc =
        yyjson_read_opts((char *)body, n, YYJSON_READ_NOFLAG, NULL, &re);
    if (!doc) {
        srv_perr(e, "invalid_json",
                 "The body is not valid JSON: %s at byte %zu.", re.msg, re.pos);
        return -1;
    }
    yyjson_val *root = yyjson_doc_get_root(doc);
    int bad = 0;
    if (!yyjson_is_obj(root)) {
        srv_perr(e, "invalid_json", "The body must be a JSON object.");
        bad = -1;
    }
    j->kind = chat ? JOB_CHAT : JOB_TEXT;
    if (!bad)
        bad = srv_read_params(e, pr, root, j, chat);
    if (!bad && chat) {
        yyjson_val *msgs = yyjson_obj_get(root, "messages");
        bad =
            srv_read_chat_opts(e, pr, root, j) || read_messages(e, pr, msgs, j);
        if (!bad && pr->store && !(pr->messages = srv_json_text(msgs, NULL)))
            bad = -1;
    }
    if (!bad && !chat)
        bad = read_prompts(e, pr, yyjson_obj_get(root, "prompt"), j);
    if (!bad && j->n_prompts * j->n_choices > 128)
        bad = srv_pfail(pr, BAD("invalid_value",
                                "At most 128 replies per request (prompts "
                                "times n or best_of)."));
    if (!bad && !chat && j->n_prompts > 1) { /* choices for every prompt */
        int per = j->n_choices, keep = j->n_keep;
        struct srv_choice *old = j->ch;
        j->ch = NULL;
        j->n_choices = 0;
        free(old);
        bad = srv_job_choices(j, per * j->n_prompts, keep * j->n_prompts);
    }
    yyjson_doc_free(doc);
    if (bad && !e->code) {
        e->oom = 1;
        snprintf(e->msg, sizeof(e->msg), "out of memory");
    }
    return bad ? -1 : 0;
}
