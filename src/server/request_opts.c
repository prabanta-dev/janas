/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * request_opts.c - the parameters of a request to /chat/completions or
 * /completions: sampling, stop strings, the number of replies, and what
 * only chat has - tools, the form of the answer, log-probabilities.
 */
#include "request.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* logit_bias: token ids (as the object's keys) to a bias of -100..100 */
static int read_bias(struct srv_perr *e, struct srv_parsed *pr, yyjson_val *lb,
                     struct srv_job *j)
{
    if (!lb || yyjson_is_null(lb))
        return 0;
    if (!yyjson_is_obj(lb))
        return srv_pfail(
            pr, BAD("invalid_value", "'logit_bias' must be an object."));
    size_t n = yyjson_obj_size(lb);
    if (n > 1024)
        return srv_pfail(pr, BAD("invalid_value",
                                 "'logit_bias' holds at most 1024 tokens."));
    j->bias_id = malloc((n + 1) * sizeof(int32_t));
    j->bias = malloc((n + 1) * sizeof(float));
    if (!j->bias_id || !j->bias)
        return -1;
    size_t idx, max;
    yyjson_val *k, *v;
    yyjson_obj_foreach(lb, idx, max, k, v)
    {
        char *end;
        long id = strtol(yyjson_get_str(k), &end, 10);
        double x;
        if (*end || id < 0 || id > INT32_MAX || srv_num(v, -100, 100, &x) != 1)
            return srv_pfail(pr, BAD("invalid_value",
                                     "'logit_bias' maps token ids to numbers "
                                     "from -100 to 100."));
        j->bias_id[j->n_bias] = (int32_t)id;
        j->bias[j->n_bias++] = (float)x;
    }
    return 0;
}

static int read_stop(struct srv_perr *e, struct srv_parsed *pr, yyjson_val *st,
                     struct srv_job *j)
{
    if (st && yyjson_is_str(st) && yyjson_get_len(st) > 0) {
        j->stop[j->n_stop++] =
            srv_dup_n(yyjson_get_str(st), yyjson_get_len(st));
    } else if (st && yyjson_is_arr(st)) {
        size_t idx, max;
        yyjson_val *s;
        if (yyjson_arr_size(st) > SRV_MAX_STOP)
            return srv_pfail(pr, BAD("invalid_value",
                                     "'stop' holds at most sixteen strings."));
        yyjson_arr_foreach(st, idx, max, s)
        {
            if (!yyjson_is_str(s))
                return srv_pfail(
                    pr, BAD("invalid_value", "'stop' must hold strings."));
            if (yyjson_get_len(s) > 0)
                j->stop[j->n_stop++] =
                    srv_dup_n(yyjson_get_str(s), yyjson_get_len(s));
        }
    } else if (st && !yyjson_is_null(st) && !yyjson_is_str(st)) {
        return srv_pfail(
            pr, BAD("invalid_value", "'stop' must be a string or an array."));
    }
    for (int i = 0; i < j->n_stop; i++)
        if (!j->stop[i])
            return -1;
    return 0;
}

/* store and metadata: the completion kept for GET /chat/completions */
static int read_store(struct srv_perr *e, struct srv_parsed *pr,
                      yyjson_val *root)
{
    yyjson_val *st = yyjson_obj_get(root, "store");
    pr->store = st && yyjson_is_bool(st) && yyjson_get_bool(st);
    yyjson_val *md = yyjson_obj_get(root, "metadata");
    if (!md || yyjson_is_null(md))
        return 0;
    if (!yyjson_is_obj(md) || yyjson_obj_size(md) > 16)
        return srv_pfail(pr, BAD("invalid_value",
                                 "'metadata' is an object of at most 16 "
                                 "strings."));
    size_t idx, max;
    yyjson_val *k, *v;
    yyjson_obj_foreach(md, idx, max, k, v)
    {
        if (!yyjson_is_str(v) || yyjson_get_len(k) > 64 ||
            yyjson_get_len(v) > 512)
            return srv_pfail(pr, BAD("invalid_value",
                                     "'metadata': keys of at most 64 "
                                     "characters, string values of at most "
                                     "512."));
    }
    return (pr->metadata = srv_json_text(md, NULL)) ? 0 : -1;
}

/*
 * llama.cpp's sampling, by its server's names: the numbers into the chat's
 * parameters (the library checks them with the rest when the reply
 * starts), the order of its stages (an array of names, or one string),
 * DRY's breakers (an array of strings) as JSON.
 */
static int read_more(struct srv_perr *e, struct srv_parsed *pr,
                     yyjson_val *root, struct srv_job *j)
{
    struct janas_llm_chat_params *p = &j->p;
    static const struct {
        const char *name;
        double lo, hi;
        size_t at;
        int whole;
    } N[] = {
        {"repeat_penalty", 0, 100,
         offsetof(struct janas_llm_chat_params, repeat_penalty), 0},
        {"repeat_last_n", -1, 1e9,
         offsetof(struct janas_llm_chat_params, repeat_last_n), 1},
        {"typical_p", 0, 1, offsetof(struct janas_llm_chat_params, typical_p),
         0},
        {"top_n_sigma", -1, 100,
         offsetof(struct janas_llm_chat_params, top_n_sigma), 0},
        {"xtc_probability", 0, 1,
         offsetof(struct janas_llm_chat_params, xtc_probability), 0},
        {"xtc_threshold", 0, 1,
         offsetof(struct janas_llm_chat_params, xtc_threshold), 0},
        {"dry_multiplier", 0, 100,
         offsetof(struct janas_llm_chat_params, dry_multiplier), 0},
        {"dry_base", 1, 100, offsetof(struct janas_llm_chat_params, dry_base),
         0},
        {"dry_allowed_length", 0, 1e6,
         offsetof(struct janas_llm_chat_params, dry_allowed_length), 1},
        {"dry_penalty_last_n", -1, 1e9,
         offsetof(struct janas_llm_chat_params, dry_last_n), 1},
        {"mirostat", 0, 2, offsetof(struct janas_llm_chat_params, mirostat), 1},
        {"mirostat_tau", 0, 100,
         offsetof(struct janas_llm_chat_params, mirostat_tau), 0},
        {"mirostat_eta", 0, 1,
         offsetof(struct janas_llm_chat_params, mirostat_eta), 0},
        {"dynatemp_range", 0, 100,
         offsetof(struct janas_llm_chat_params, dynatemp_range), 0},
        {"dynatemp_exponent", 0, 100,
         offsetof(struct janas_llm_chat_params, dynatemp_exponent), 0},
    };
    double x;
    for (size_t i = 0; i < sizeof N / sizeof *N; i++) {
        int got =
            srv_num(yyjson_obj_get(root, N[i].name), N[i].lo, N[i].hi, &x);
        if (got < 0 || (got && N[i].whole && x != (double)(int64_t)x))
            return srv_pfail(
                pr, BAD("invalid_value", "'%s' must be %s from %g to %g.",
                        N[i].name, N[i].whole ? "a whole number" : "a number",
                        N[i].lo, N[i].hi));
        if (!got)
            continue;
        /* llama.cpp's -1, the whole context: here the whole reply */
        if (N[i].whole) {
            int32_t v = x < 0 ? 0 : (int32_t)x;
            memcpy((char *)p + N[i].at, &v, sizeof v);
        } else {
            float v = x < 0 ? 0 : (float)x;
            memcpy((char *)p + N[i].at, &v, sizeof v);
        }
    }
    /* the order: ["top_k", "temperature"] or "top_k;temperature" */
    yyjson_val *sm = yyjson_obj_get(root, "samplers");
    if (sm) {
        char o[sizeof p->samplers];
        size_t at = 0;
        int ok = yyjson_is_str(sm) || yyjson_is_arr(sm);
        if (yyjson_is_str(sm)) {
            ok = yyjson_get_len(sm) < sizeof o;
            if (ok)
                memcpy(o, yyjson_get_str(sm), (at = yyjson_get_len(sm)));
        } else if (ok) {
            size_t idx, max;
            yyjson_val *v;
            yyjson_arr_foreach(sm, idx, max, v)
            {
                size_t l = yyjson_get_len(v);
                if (!yyjson_is_str(v) || !l || at + l + 2 > sizeof o ||
                    memchr(yyjson_get_str(v), ';', l)) {
                    ok = 0;
                    break;
                }
                if (at)
                    o[at++] = ';';
                memcpy(o + at, yyjson_get_str(v), l);
                at += l;
            }
        }
        if (!ok)
            return srv_pfail(pr, BAD("invalid_value",
                                     "'samplers' is an array of llama.cpp's "
                                     "stage names, or one string of them."));
        o[at] = 0;
        memcpy(p->samplers, o, at + 1);
    }
    /* DRY's breakers, as JSON for the library */
    yyjson_val *br = yyjson_obj_get(root, "dry_sequence_breakers");
    if (br) {
        size_t idx, max;
        yyjson_val *v;
        int ok = yyjson_is_arr(br);
        yyjson_arr_foreach(br, idx, max, v) ok &= yyjson_is_str(v);
        size_t n = 0;
        char *t = ok ? yyjson_val_write(br, 0, &n) : NULL;
        if (!t || n >= sizeof p->dry_breakers) {
            free(t);
            return srv_pfail(pr, BAD("invalid_value",
                                     "'dry_sequence_breakers' is an array of "
                                     "strings, %zu bytes at most as JSON.",
                                     sizeof p->dry_breakers - 1));
        }
        memcpy(p->dry_breakers, t, n + 1);
        free(t);
    }
    return 0;
}

/* the sampling of a request, or of the defaults: the fields it has */
static int read_sampling(struct srv_perr *e, struct srv_parsed *pr,
                         yyjson_val *root, struct srv_job *j)
{
    double x;
    int got;
    if ((got = srv_num(yyjson_obj_get(root, "temperature"), 0, 2, &x)) < 0)
        return srv_pfail(
            pr, BAD("invalid_value", "'temperature' must be between 0 and 2."));
    if (got)
        j->p.temperature = (float)x;
    if ((got = srv_num(yyjson_obj_get(root, "top_p"), 0, 1, &x)) < 0)
        return srv_pfail(
            pr, BAD("invalid_value", "'top_p' must be between 0 and 1."));
    if (got)
        j->p.top_p = (float)x;
    if ((got = srv_num(yyjson_obj_get(root, "top_k"), 0, 1e6, &x)) < 0)
        return srv_pfail(pr,
                         BAD("invalid_value", "'top_k' must be 0 or more."));
    if (got)
        j->p.top_k = (int32_t)x;
    if ((got = srv_num(yyjson_obj_get(root, "min_p"), 0, 1, &x)) < 0)
        return srv_pfail(
            pr, BAD("invalid_value", "'min_p' must be between 0 and 1."));
    if (got)
        j->p.min_p = (float)x;
    if (read_more(e, pr, root, j) != 0)
        return -1;
    static const char *const pen[] = {"presence_penalty", "frequency_penalty"};
    for (int i = 0; i < 2; i++) {
        if ((got = srv_num(yyjson_obj_get(root, pen[i]), -2, 2, &x)) < 0)
            return srv_pfail(pr, BAD("invalid_value",
                                     "'%s' must be between -2 and 2.", pen[i]));
        if (got && i == 0)
            j->p.presence_penalty = (float)x;
        if (got && i == 1)
            j->p.frequency_penalty = (float)x;
    }
    return 0;
}

/* --chat-template-kwargs: the template's variables of every request, each
   request's own over them */
static yyjson_doc *kw_defaults;

int srv_kwargs_defaults(const char *json, char *why, size_t why_len)
{
    yyjson_doc *d = json ? yyjson_read(json, strlen(json), 0) : NULL;
    if (!d || !yyjson_is_obj(yyjson_doc_get_root(d))) {
        yyjson_doc_free(d);
        snprintf(why, why_len,
                 "--chat-template-kwargs: a JSON object of the chat "
                 "template's variables");
        return -1;
    }
    yyjson_doc_free(kw_defaults);
    kw_defaults = d;
    return 0;
}

/*
 * The chat template's keywords of a request: the server's, the request's
 * chat_template_kwargs over them (every member, as Hugging Face hands them
 * to the template), and OpenAI's reasoning_effort for a template that
 * reads it when they do not say; the request's own template, if it gives
 * one (chat_template, as vLLM takes it).
 */
int srv_template_opts(yyjson_val *root, yyjson_val *kw, const char *effort,
                      struct srv_job *j)
{
    yyjson_val *t = yyjson_obj_get(root, "chat_template");
    if (t && !yyjson_is_null(t)) {
        if (!yyjson_is_str(t) || !yyjson_get_len(t))
            return -1;
        free(j->tmpl);
        if (!(j->tmpl = strdup(yyjson_get_str(t))))
            return -1;
    }
    yyjson_mut_doc *m = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = m ? yyjson_mut_obj(m) : NULL;
    if (!o) {
        yyjson_mut_doc_free(m);
        return -1;
    }
    yyjson_mut_doc_set_root(m, o);
    yyjson_val *src[2] = {kw_defaults ? yyjson_doc_get_root(kw_defaults) : NULL,
                          kw};
    for (int s = 0; s < 2; s++) {
        size_t i, n;
        yyjson_val *k, *v;
        if (!src[s])
            continue;
        yyjson_obj_foreach(src[s], i, n, k, v)
        {
            yyjson_mut_obj_remove_str(o, yyjson_get_str(k));
            yyjson_mut_obj_add(o, yyjson_mut_strcpy(m, yyjson_get_str(k)),
                               yyjson_val_mut_copy(m, v));
        }
    }
    if (effort && !yyjson_mut_obj_get(o, "reasoning_effort"))
        yyjson_mut_obj_add_strcpy(m, o, "reasoning_effort", effort);
    free(j->kwargs);
    j->kwargs = yyjson_mut_obj_size(o) ? yyjson_mut_write(m, 0, NULL) : NULL;
    yyjson_mut_doc_free(m);
    return 0;
}

/* --sampling: the sampling of the requests that do not say, read once at
   the start; each request's own fields over it */
static yyjson_doc *defaults;

int srv_sampling_defaults(const char *json, char *why, size_t why_len)
{
    yyjson_doc *d = json ? yyjson_read(json, strlen(json), 0) : NULL;
    if (!d || !yyjson_is_obj(yyjson_doc_get_root(d))) {
        yyjson_doc_free(d);
        snprintf(why, why_len,
                 "--sampling: a JSON object of the requests' "
                 "sampling fields");
        return -1;
    }
    /* tried on a request of its own: what it would refuse, refused now */
    struct srv_job *j = srv_job_new();
    struct srv_perr pe = {0};
    struct srv_parsed pr = {0};
    int bad = !j || read_sampling(&pe, &pr, yyjson_doc_get_root(d), j) != 0;
    if (bad)
        snprintf(why, why_len, "--sampling: %s", pe.msg);
    if (j)
        srv_job_release(j);
    srv_parsed_free(&pr);
    if (bad) {
        yyjson_doc_free(d);
        return -1;
    }
    yyjson_doc_free(defaults);
    defaults = d;
    return 0;
}

void srv_sampling_apply(struct srv_job *j)
{
    struct srv_perr pe = {0};
    struct srv_parsed pr = {0};
    if (defaults)
        read_sampling(&pe, &pr, yyjson_doc_get_root(defaults), j);
    srv_parsed_free(&pr);
}

/* The sampling as the request (or else the defaults) says, then the rest. */
int srv_read_params(struct srv_perr *e, struct srv_parsed *pr, yyjson_val *root,
                    struct srv_job *j, int chat)
{
    double x;
    int got;
    if (defaults && read_sampling(e, pr, yyjson_doc_get_root(defaults), j) != 0)
        return -1;
    if (read_sampling(e, pr, root, j) != 0)
        return -1;
    /*
     * max_completion_tokens counts the reasoning, as OpenAI defines it. The
     * older max_tokens of a chat is taken for the answer alone: OpenAI does
     * not take it for the models that reason, and a client that sends it
     * (Open WebUI asks for a title with 1000) means the answer - counted
     * with the reasoning, a model that reasons at length spent it all
     * before writing a word. In completions there is no reasoning.
     */
    int total = !chat || yyjson_obj_get(root, "max_completion_tokens");
    const char *mt = yyjson_obj_get(root, "max_completion_tokens")
                         ? "max_completion_tokens"
                         : "max_tokens";
    if ((got = srv_num(yyjson_obj_get(root, mt), 1, 1e9, &x)) < 0)
        return srv_pfail(pr,
                         BAD("invalid_value", "'%s' must be 1 or more.", mt));
    if (got && !total)
        j->p.max_answer = (int32_t)x;
    else
        j->p.max_reply = got ? (int32_t)x : chat ? 0 : 16; /* OpenAI's */
    yyjson_val *seed = yyjson_obj_get(root, "seed");
    if (seed && yyjson_is_int(seed)) {
        j->p.seed = (uint64_t)yyjson_get_sint(seed)
                        ? (uint64_t)yyjson_get_sint(seed)
                        : 1;
        j->seeded = 1;
    }
    /* n replies; completions: best_of candidates, the best n returned */
    if ((got = srv_num(yyjson_obj_get(root, "n"), 1, 128, &x)) < 0 ||
        (got && x != (int)x))
        return srv_pfail(pr, BAD("invalid_value",
                                 "'n' must be a whole number from 1 to 128."));
    int n = got ? (int)x : 1, gen = n;
    if (!chat) {
        if ((got = srv_num(yyjson_obj_get(root, "best_of"), 1, 128, &x)) < 0 ||
            (got && (x != (int)x || x < n)))
            return srv_pfail(pr, BAD("invalid_value",
                                     "'best_of' must be a whole number from "
                                     "n to 128."));
        if (got)
            gen = (int)x;
    }
    if (read_bias(e, pr, yyjson_obj_get(root, "logit_bias"), j) != 0 ||
        read_stop(e, pr, yyjson_obj_get(root, "stop"), j) != 0 ||
        read_store(e, pr, root) != 0)
        return -1;
    yyjson_val *sv = yyjson_obj_get(root, "stream");
    pr->stream = sv && yyjson_is_bool(sv) && yyjson_get_bool(sv);
    yyjson_val *so = yyjson_obj_get(root, "stream_options");
    yyjson_val *iu = so ? yyjson_obj_get(so, "include_usage") : NULL;
    pr->include_usage = iu && yyjson_is_bool(iu) && yyjson_get_bool(iu);
    if (!chat) {
        yyjson_val *lp = yyjson_obj_get(root, "logprobs");
        if ((got = srv_num(lp, 0, 20, &x)) < 0)
            return srv_pfail(
                pr, BAD("invalid_value", "'logprobs' must be from 0 to 20."));
        if (got) { /* the token's own, and x alternatives */
            pr->logprobs = (int)x + 1;
            j->p.logprobs = 1;
            j->p.top_logprobs = (int32_t)x;
        }
        yyjson_val *ec = yyjson_obj_get(root, "echo");
        pr->echo = ec && yyjson_is_bool(ec) && yyjson_get_bool(ec);
        yyjson_val *sx = yyjson_obj_get(root, "suffix");
        if (sx && yyjson_is_str(sx)) {
            j->n_suffix = yyjson_get_len(sx);
            if (!(j->suffix = srv_dup_n(yyjson_get_str(sx), j->n_suffix)))
                return -1;
        } else if (sx && !yyjson_is_null(sx)) {
            return srv_pfail(
                pr, BAD("invalid_value", "'suffix' must be a string."));
        }
        if (gen > n) /* the best are chosen by their log-probability */
            j->p.logprobs = 1;
    }
    if (pr->stream && gen > n)
        return srv_pfail(pr, BAD("invalid_value",
                                 "'best_of' cannot be streamed: the best is "
                                 "known only at the end."));
    return srv_job_choices(j, gen, n);
}

/* tool_choice, or the older function_call */
static int read_choice(struct srv_perr *e, struct srv_parsed *pr,
                       yyjson_val *tc, struct srv_job *j)
{
    if (!tc || yyjson_is_null(tc))
        return 0;
    const char *s = yyjson_get_str(tc);
    if (s) {
        j->tool_choice = strcmp(s, "none") == 0       ? JANAS_LLM_TOOLS_NONE
                         : strcmp(s, "required") == 0 ? JANAS_LLM_TOOLS_REQUIRED
                         : strcmp(s, "auto") == 0     ? JANAS_LLM_TOOLS_AUTO
                                                      : -1;
        if (j->tool_choice < 0)
            return srv_pfail(pr, BAD("invalid_value",
                                     "'tool_choice' is none, auto, required "
                                     "or a function."));
        return 0;
    }
    yyjson_val *fn = yyjson_obj_get(tc, "function");
    const char *name = yyjson_get_str(fn ? yyjson_obj_get(fn, "name")
                                         : yyjson_obj_get(tc, "name"));
    if (!name || strlen(name) >= sizeof(j->tool_name))
        return srv_pfail(
            pr, BAD("invalid_value", "'tool_choice' names no function."));
    j->tool_choice = JANAS_LLM_TOOLS_FUNCTION;
    memcpy(j->tool_name, name, strlen(name) + 1);
    return 0;
}

/* tools, or the older functions (the functions alone) */
static int read_tools(struct srv_perr *e, struct srv_parsed *pr,
                      yyjson_val *root, struct srv_job *j)
{
    yyjson_val *tools = yyjson_obj_get(root, "tools");
    if (!tools || yyjson_is_null(tools))
        tools = yyjson_obj_get(root, "functions");
    if (!tools || yyjson_is_null(tools) ||
        (yyjson_is_arr(tools) && yyjson_arr_size(tools) == 0))
        return 0;
    if (!yyjson_is_arr(tools))
        return srv_pfail(pr, BAD("invalid_value", "'tools' must be an array."));
    size_t idx, max;
    yyjson_val *t;
    yyjson_arr_foreach(tools, idx, max, t)
    {
        yyjson_val *fn = yyjson_obj_get(t, "function");
        const char *type = yyjson_get_str(yyjson_obj_get(t, "type"));
        if (type && strcmp(type, "function") != 0)
            return srv_pfail(pr, BAD("unsupported_value",
                                     "tools[%zu]: only functions can be "
                                     "called here (got '%s').",
                                     idx, type));
        if (!yyjson_get_str(yyjson_obj_get(fn ? fn : t, "name")))
            return srv_pfail(pr, BAD("invalid_value",
                                     "tools[%zu] has no function name.", idx));
    }
    if (!(j->tools = srv_json_text(tools, &j->n_tools)))
        return -1;
    yyjson_val *pc = yyjson_obj_get(root, "parallel_tool_calls");
    if (pc && yyjson_is_bool(pc))
        j->parallel = yyjson_get_bool(pc);
    yyjson_val *tc = yyjson_obj_get(root, "tool_choice");
    if (!tc || yyjson_is_null(tc))
        tc = yyjson_obj_get(root, "function_call");
    return read_choice(e, pr, tc, j);
}

/* response_format: text, json_object, or json_schema */
static int read_format(struct srv_perr *e, struct srv_parsed *pr,
                       yyjson_val *rf, struct srv_job *j)
{
    if (!rf || yyjson_is_null(rf))
        return 0;
    const char *type = yyjson_get_str(yyjson_obj_get(rf, "type"));
    if (!type)
        return srv_pfail(
            pr, BAD("invalid_value", "'response_format' needs a type."));
    if (strcmp(type, "text") == 0)
        return 0;
    if (strcmp(type, "json_object") == 0) {
        j->format = JANAS_LLM_FORMAT_JSON;
        return 0;
    }
    if (strcmp(type, "json_schema") != 0)
        return srv_pfail(pr, BAD("invalid_value",
                                 "'response_format' is text, json_object or "
                                 "json_schema (got '%s').",
                                 type));
    yyjson_val *js = yyjson_obj_get(rf, "json_schema");
    yyjson_val *schema = js ? yyjson_obj_get(js, "schema") : NULL;
    if (!schema || !(yyjson_is_obj(schema) || yyjson_is_bool(schema)))
        return srv_pfail(pr, BAD("invalid_value",
                                 "'response_format.json_schema.schema' must "
                                 "be a JSON Schema."));
    j->format = JANAS_LLM_FORMAT_SCHEMA;
    return (j->schema = srv_json_text(schema, &j->n_schema)) ? 0 : -1;
}

int srv_read_chat_opts(struct srv_perr *e, struct srv_parsed *pr,
                       yyjson_val *root, struct srv_job *j)
{
    if (read_tools(e, pr, root, j) != 0 ||
        read_format(e, pr, yyjson_obj_get(root, "response_format"), j) != 0)
        return -1;
    yyjson_val *lp = yyjson_obj_get(root, "logprobs");
    if (lp && yyjson_is_bool(lp) && yyjson_get_bool(lp)) {
        pr->logprobs = 1;
        j->p.logprobs = 1;
    }
    double x;
    int got = srv_num(yyjson_obj_get(root, "top_logprobs"), 0, 20, &x);
    if (got < 0 || (got && !pr->logprobs && x > 0))
        return srv_pfail(pr, BAD("invalid_value",
                                 "'top_logprobs' is from 0 to 20, and asks "
                                 "for 'logprobs': true."));
    if (got)
        j->p.top_logprobs = (int32_t)x;
    yyjson_val *mod = yyjson_obj_get(root, "modalities");
    if (mod && yyjson_is_arr(mod)) {
        size_t idx, max;
        yyjson_val *v;
        yyjson_arr_foreach(mod, idx, max, v)
        {
            const char *s = yyjson_get_str(v);
            if (!s || strcmp(s, "text") != 0)
                return srv_pfail(pr,
                                 BAD("unsupported_value",
                                     "Only the text modality is supported."));
        }
    }
    /* the reasoning: OpenAI's reasoning_effort, or the chat template's own
       switch as other servers take it */
    const char *eff = yyjson_get_str(yyjson_obj_get(root, "reasoning_effort"));
    if (eff)
        j->p.thinking =
            strcmp(eff, "none") == 0 || strcmp(eff, "minimal") == 0 ? 0 : 1;
    yyjson_val *kw = yyjson_obj_get(root, "chat_template_kwargs");
    yyjson_val *et = kw ? yyjson_obj_get(kw, "enable_thinking") : NULL;
    if (et && yyjson_is_bool(et))
        j->p.thinking = yyjson_get_bool(et) ? 1 : 0;
    if (kw && !yyjson_is_obj(kw))
        return srv_pfail(pr, BAD("invalid_value",
                                 "'chat_template_kwargs' is an object of the "
                                 "chat template's variables."));
    if (srv_template_opts(root, kw, eff, j) != 0)
        return srv_pfail(pr, BAD("invalid_value",
                                 "'chat_template' is the text of a chat "
                                 "template."));
    if (srv_reasoning_budget(root, &j->p) != 0)
        return srv_pfail(
            pr, BAD("invalid_value",
                    "'reasoning_budget_tokens' is -1 (no limit), 0 (no "
                    "reasoning) or a number of tokens."));
    return 0;
}

void srv_reasoning_default(const struct srv_config *cfg,
                           struct janas_llm_chat_params *p)
{
    p->thinking = cfg->thinking;
    if (cfg->reasoning_budget == 0)
        p->thinking = 0;
    else if (cfg->reasoning_budget > 0)
        p->max_reasoning = cfg->reasoning_budget;
}

int srv_reasoning_budget(yyjson_val *root, struct janas_llm_chat_params *p)
{
    yyjson_val *b = yyjson_obj_get(root, "reasoning_budget_tokens");
    if (!b)
        b = yyjson_obj_get(root, "thinking_budget_tokens");
    double x;
    int got = srv_num(b, -1, 2147483647.0, &x);
    if (got < 0 || (got && x != (int32_t)x))
        return -1;
    if (got && x == 0) {
        p->thinking = 0;
        p->max_reasoning = 0;
    } else if (got)
        p->max_reasoning = x < 0 ? 0 : (int32_t)x;
    return 0;
}
