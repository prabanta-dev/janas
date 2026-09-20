/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_tools.c - the public API: tools, the form of the answer, the logit
 * bias, and what a reply leaves besides its text - its tool calls and its
 * tokens with their log-probabilities (see include/janas/llm.h).
 */
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"
#include "llm/schema.h"

int32_t janas_llm_chat_tools(janas_llm_chat *c, const char *json, int32_t len)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    size_t n = json ? janas_api_text_len(json, len) : 0;
    if (n == 0) {
        janas_toolset_free(c->tools);
        c->tools = NULL;
        return JANAS_LLM_OK;
    }
    if (c->llm->tools == JANAS_TOOLS_NONE)
        return janas_api_fail(JANAS_LLM_EMODEL,
                              "this model's chat template has no tools");
    char err[200];
    struct janas_toolset *t = janas_toolset_parse(json, n, err, sizeof(err));
    if (!t)
        return janas_api_fail(JANAS_LLM_EINVAL, "tools: %s", err);
    if (janas_toolset_count(t) == 0) {
        janas_toolset_free(t);
        t = NULL;
    }
    janas_toolset_free(c->tools);
    c->tools = t;
    if (c->tool_choice == JANAS_LLM_TOOLS_FUNCTION)
        c->tool_choice = JANAS_LLM_TOOLS_AUTO; /* the function may be gone */
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_tool_choice(janas_llm_chat *c, int32_t choice,
                                   const char *name, int32_t len,
                                   int32_t parallel)
{
    if (!c || choice < JANAS_LLM_TOOLS_AUTO ||
        choice > JANAS_LLM_TOOLS_FUNCTION)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid tool choice");
    int only = -1;
    if (choice == JANAS_LLM_TOOLS_FUNCTION) {
        if (!name)
            return janas_api_fail(JANAS_LLM_EINVAL, "no function named");
        only =
            janas_toolset_find(c->tools, name, janas_api_text_len(name, len));
        if (only < 0)
            return janas_api_fail(JANAS_LLM_EINVAL, "no tool is called '%.*s'",
                                  (int)janas_api_text_len(name, len), name);
    }
    c->tool_choice = choice;
    c->tool_only = only;
    c->parallel = parallel != 0;
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_format(janas_llm_chat *c, int32_t kind,
                              const char *schema, int32_t len)
{
    if (!c || kind < JANAS_LLM_FORMAT_TEXT || kind > JANAS_LLM_FORMAT_SCHEMA)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid format");
    struct janas_json_doc *d = NULL;
    if (kind == JANAS_LLM_FORMAT_SCHEMA) {
        if (!schema)
            return janas_api_fail(JANAS_LLM_EINVAL, "no schema");
        char err[200];
        d = janas_json_parse(schema, janas_api_text_len(schema, len), err,
                             sizeof(err));
        if (!d)
            return janas_api_fail(JANAS_LLM_EINVAL, "schema: %s", err);
        enum janas_json_type ty = janas_json_root(d)->type;
        if (ty != JANAS_JSON_OBJECT && ty != JANAS_JSON_TRUE &&
            ty != JANAS_JSON_FALSE) {
            janas_json_free(d);
            return janas_api_fail(JANAS_LLM_EINVAL,
                                  "a schema is an object or a boolean");
        }
    }
    janas_json_free(c->schema);
    c->schema = d;
    c->format = kind;
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_logit_bias(janas_llm_chat *c, int32_t n,
                                  const int32_t *ids, const float *bias)
{
    if (!c || n < 0 || (n > 0 && (!ids || !bias)))
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    for (int32_t i = 0; i < n; i++)
        if (!(bias[i] >= -100 && bias[i] <= 100))
            return janas_api_fail(JANAS_LLM_EINVAL,
                                  "a bias is between -100 and 100");
    if (janas_llm_session_bias(c->s, ids, bias, (uint32_t)n) != 0)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "a token of the bias is not in the vocabulary");
    return JANAS_LLM_OK;
}

/* ---- what holds a reply ---- */

/*
 * <tool_call> body, then more of them (parallel) or nothing. Between and
 * around the calls, a line break or two and nothing else: with the white
 * space of JSON allowed there, Qwen3-0.6B filled it with tabs after the
 * call instead of ending the reply.
 */
static uint32_t calls_rule(struct janas_gbuild *b, const janas_llm *llm,
                           uint32_t call, int parallel)
{
    uint32_t nl = janas_gb_rule(b);
    janas_gb_alt(b, nl);
    janas_gb_lit(b, janas_gb_alt(b, nl), "\n", 1);
    janas_gb_lit(b, janas_gb_alt(b, nl), "\n\n", 2);
    uint32_t one = janas_gb_rule(b);
    uint32_t a = janas_gb_alt(b, one);
    janas_gb_token(b, a, llm->call_open);
    janas_gb_ref(b, a, call);
    uint32_t more = janas_gb_rule(b); /* nothing, or one more */
    janas_gb_alt(b, more);
    if (parallel) {
        a = janas_gb_alt(b, more);
        janas_gb_ref(b, a, nl);
        janas_gb_ref(b, a, one);
        janas_gb_ref(b, a, more);
    }
    uint32_t r = janas_gb_rule(b);
    a = janas_gb_alt(b, r);
    janas_gb_ref(b, a, nl);
    janas_gb_ref(b, a, one);
    janas_gb_ref(b, a, more);
    janas_gb_ref(b, a, nl);
    return r;
}

void janas_api_reply_clear(janas_llm_chat *c)
{
    if (c->s)
        janas_llm_session_filter(c->s, NULL);
    janas_constraint_free(c->cons);
    c->cons = NULL;
    janas_grammar_free(c->g);
    c->g = NULL;
    for (size_t i = 0; i < c->n_calls; i++) {
        free(c->calls[i].name);
        free(c->calls[i].args);
    }
    c->n_calls = 0;
    c->n_toks = 0;
    c->tbytes.n = 0;
    c->in_call = 0;
    c->body.n = 0;
}

int janas_api_reply_setup(janas_llm_chat *c)
{
    janas_api_reply_clear(c);
    const janas_llm *llm = c->llm;
    int tools =
        c->tools && llm->tools != JANAS_TOOLS_NONE && c->reply_at != UINT32_MAX;
    int choice = tools ? c->tool_choice : JANAS_LLM_TOOLS_NONE;
    int must = choice == JANAS_LLM_TOOLS_REQUIRED ||
               choice == JANAS_LLM_TOOLS_FUNCTION;
    if (!tools && c->format == JANAS_LLM_FORMAT_TEXT)
        return 0; /* nothing to hold: the sampler asks nobody */
    struct janas_gbuild *b = janas_gbuild_new();
    struct janas_jsg *j = b ? janas_jsg_new(b) : NULL;
    if (!j) {
        janas_gbuild_free(b);
        return -1;
    }
    uint32_t call = JANAS_NO_RULE, answer = JANAS_NO_RULE;
    if (tools && choice != JANAS_LLM_TOOLS_NONE)
        call = janas_tools_rule(
            b, j, llm->tools, c->tools,
            choice == JANAS_LLM_TOOLS_FUNCTION ? c->tool_only : -1,
            llm->call_close);
    if (must) {
        answer = calls_rule(b, llm, call, c->parallel);
    } else if (c->format != JANAS_LLM_FORMAT_TEXT) {
        const struct janas_json *s = janas_json_root(c->schema);
        uint32_t val = c->format == JANAS_LLM_FORMAT_JSON
                           ? janas_jsg_object(j)
                           : janas_jsg_schema(j, s, s);
        uint32_t ws = janas_jsg_ws(j);
        answer = janas_gb_rule(b);
        uint32_t a = janas_gb_alt(b, answer);
        janas_gb_ref(b, a, ws);
        janas_gb_ref(b, a, val);
        janas_gb_ref(b, a, ws);
        if (call != JANAS_NO_RULE) /* or calls instead, when allowed */
            janas_gb_ref(b, janas_gb_alt(b, answer),
                         calls_rule(b, llm, call, 1));
    }
    c->g = janas_grammar_make(b);
    janas_jsg_free(j);
    janas_gbuild_free(b);
    if (!c->g)
        return -1;
    struct janas_constraint_spec spec = {
        .g = c->g,
        .answer = answer,
        .call = answer == JANAS_NO_RULE ? call : JANAS_NO_RULE,
        .call_open = llm->call_open,
        .ban_calls = tools && choice == JANAS_LLM_TOOLS_NONE,
        .thinking = c->in_think,
        .think_close = llm->think_close,
        .stop = llm->stop,
        .n_stop = llm->n_stop,
    };
    c->cons = janas_constraint_new(&spec, llm->tok);
    if (!c->cons)
        return -1;
    struct janas_sample_filter f;
    janas_constraint_filter(c->cons, &f);
    janas_llm_session_filter(c->s, &f);
    return 0;
}

void janas_api_tools_free(janas_llm_chat *c)
{
    janas_api_reply_clear(c);
    janas_toolset_free(c->tools);
    c->tools = NULL;
    janas_json_free(c->schema);
    c->schema = NULL;
    free(c->calls);
    c->calls = NULL;
    free(c->toks);
    c->toks = NULL;
    janas_buf_free(&c->tbytes);
    janas_buf_free(&c->body);
    free(c->top_id);
    free(c->top_lp);
    c->top_id = NULL;
    c->top_lp = NULL;
    c->top_cap = 0;
}

/* ---- the reply's calls ---- */

void janas_api_close_call(janas_llm_chat *c, struct janas_buf *text)
{
    struct janas_buf name = {0}, args = {0};
    int ok =
        c->tools &&
        janas_tools_parse(c->llm->tools, c->tools, c->body.p ? c->body.p : "",
                          c->body.n, &name, &args) == 0 &&
        !name.oom && !args.oom;
    if (ok && c->n_calls == c->cap_calls) {
        size_t cap = c->cap_calls ? 2 * c->cap_calls : 4;
        struct rcall *t = realloc(c->calls, cap * sizeof(*t));
        if (t) {
            c->calls = t;
            c->cap_calls = cap;
        } else {
            ok = 0;
        }
    }
    if (ok) {
        janas_buf_put(&name, "", 0);
        janas_buf_put(&args, "", 0);
        c->calls[c->n_calls++] = (struct rcall){.name = name.p, .args = args.p};
    } else {
        /* not a call after all: it stays in the text as it was written */
        janas_buf_puts(text, "<tool_call>");
        janas_buf_put(text, c->body.p ? c->body.p : "", c->body.n);
        janas_buf_puts(text, "</tool_call>");
        janas_buf_free(&name);
        janas_buf_free(&args);
    }
    c->body.n = 0;
    c->in_call = 0;
}

int32_t janas_llm_chat_calls(const janas_llm_chat *c)
{
    return c ? (int32_t)c->n_calls : 0;
}

int32_t janas_llm_chat_call(const janas_llm_chat *c, int32_t i, char *name,
                            int32_t name_cap, int32_t *name_len, char *args,
                            int32_t args_cap, int32_t *args_len)
{
    if (!c || i < 0 || (size_t)i >= c->n_calls)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such call");
    const struct rcall *r = &c->calls[i];
    int32_t a =
        janas_api_copy_out(r->name, strlen(r->name), name, name_cap, name_len);
    int32_t b =
        janas_api_copy_out(r->args, strlen(r->args), args, args_cap, args_len);
    return a != JANAS_LLM_OK ? a : b;
}

/* ---- the reply's tokens ---- */

int janas_api_record(janas_llm_chat *c, int32_t t, int32_t part,
                     const char *bytes, size_t n)
{
    int keep_top = c->p.logprobs && c->p.top_logprobs > 0;
    if (c->n_toks == c->cap_toks) {
        size_t cap = c->cap_toks ? 2 * c->cap_toks : 256;
        struct rtok *r = realloc(c->toks, cap * sizeof(*r));
        if (!r)
            return -1;
        c->toks = r;
        c->cap_toks = cap;
    }
    /* room for the alternatives, as much as for the tokens: made the first
       time a reply keeps them, grown with the tokens after that */
    if ((keep_top || c->top_id) && c->top_cap < c->cap_toks) {
        size_t cap = c->cap_toks;
        int32_t *ti =
            realloc(c->top_id, cap * JANAS_SAMPLE_MAX_TOP * sizeof(int32_t));
        if (ti)
            c->top_id = ti;
        float *tl =
            realloc(c->top_lp, cap * JANAS_SAMPLE_MAX_TOP * sizeof(float));
        if (tl)
            c->top_lp = tl;
        if (!ti || !tl)
            return -1;
        c->top_cap = cap;
    }
    struct rtok *r = &c->toks[c->n_toks];
    *r = (struct rtok){
        .id = t, .part = part, .off = (uint32_t)c->tbytes.n, .n = (uint32_t)n};
    janas_buf_put(&c->tbytes, bytes, n);
    struct janas_token_lp lp;
    uint32_t pos = janas_llm_session_length(c->s) - 1;
    if (c->p.logprobs && janas_llm_session_logprob(c->s, pos, &lp) == 0) {
        r->lp = lp.logprob;
        if (keep_top && c->top_id && c->top_lp) {
            r->n_top = lp.n_top;
            memcpy(c->top_id + c->n_toks * JANAS_SAMPLE_MAX_TOP, lp.top_id,
                   (size_t)lp.n_top * sizeof(int32_t));
            memcpy(c->top_lp + c->n_toks * JANAS_SAMPLE_MAX_TOP, lp.top_lp,
                   (size_t)lp.n_top * sizeof(float));
        }
    }
    c->n_toks++;
    return c->tbytes.oom ? -1 : 0;
}

int32_t janas_llm_chat_tokens(const janas_llm_chat *c)
{
    return c ? (int32_t)c->n_toks : 0;
}

int32_t janas_llm_chat_token(const janas_llm_chat *c, int32_t i, int32_t *id,
                             int32_t *part, float *logprob, char *buf,
                             int32_t cap, int32_t *len)
{
    if (!c || i < 0 || (size_t)i >= c->n_toks)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such token");
    const struct rtok *r = &c->toks[i];
    if (id)
        *id = r->id;
    if (part)
        *part = r->part;
    if (logprob)
        *logprob = r->lp;
    if (!buf && !len)
        return JANAS_LLM_OK;
    return janas_api_copy_out(c->tbytes.p + r->off, r->n, buf, cap, len);
}

int32_t janas_llm_chat_token_top(const janas_llm_chat *c, int32_t i,
                                 int32_t max, int32_t *ids, float *logprobs,
                                 int32_t *n)
{
    if (!c || i < 0 || (size_t)i >= c->n_toks || max < 0 || !n)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such token");
    const struct rtok *r = &c->toks[i];
    int32_t k = r->n_top < max ? r->n_top : max;
    for (int32_t q = 0; q < k; q++) {
        if (ids)
            ids[q] = c->top_id[(size_t)i * JANAS_SAMPLE_MAX_TOP + (size_t)q];
        if (logprobs)
            logprobs[q] =
                c->top_lp[(size_t)i * JANAS_SAMPLE_MAX_TOP + (size_t)q];
    }
    *n = k;
    return JANAS_LLM_OK;
}
