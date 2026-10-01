/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * infill.c - POST /infill, llama.cpp's route for filling in code, which
 * the editors' extensions made for it ask (llama.vscode, llama.vim): the
 * lines before the cursor (input_prefix), the line up to it (prompt), and
 * what follows (input_suffix), into the same window as a completion with
 * a suffix (fim.c), the answer as llama.cpp gives it (content, timings).
 *
 * Read as llama.cpp reads them: n_predict (0: the prompt only, read so
 * that the next request finds it computed), n_indent (a new line indented
 * less ends the text), t_max_predict_ms (after a new line, past that time
 * the text ends). The extra context of other files (input_extra) is not
 * used: it changes often, comes first, and every change would have the
 * whole prompt read again. The samplers are not either: the most likely
 * token is taken, so that the same place gives the same proposal.
 */
#include "request.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *str_of(yyjson_val *o, const char *k, size_t *n, int *bad)
{
    yyjson_val *v = yyjson_obj_get(o, k);
    if (!v || yyjson_is_null(v)) {
        *n = 0;
        return NULL;
    }
    if (!yyjson_is_str(v)) {
        *bad = 1;
        return NULL;
    }
    *n = yyjson_get_len(v);
    return yyjson_get_str(v);
}

static long num_of(yyjson_val *o, const char *k, long dflt, int *bad)
{
    yyjson_val *v = yyjson_obj_get(o, k);
    if (!v || yyjson_is_null(v))
        return dflt;
    if (!yyjson_is_num(v)) {
        *bad = 1;
        return dflt;
    }
    return (long)yyjson_get_num(v);
}

/* The answer, as llama.cpp's: the fields the extensions read. */
static char *answer(const struct srv_req *r, const struct srv_job *j, size_t *n)
{
    const struct srv_choice *ch = &j->ch[0];
    const struct janas_llm_chat_stats *st = &j->st;
    const char *text = j->prefill_only || !ch->text ? "" : ch->text;
    double in_ms = st->input_seconds * 1000.0,
           out_ms = st->output_seconds * 1000.0;
    uint32_t out_n = j->prefill_only ? 0 : st->output_tokens;
    const char *stop_type =
        ch->finish == JANAS_LLM_FINISH_LENGTH ? "limit" : "eos";
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    if (!d)
        return NULL;
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    yyjson_mut_obj_add_strncpy(d, o, "content", text, strlen(text));
    yyjson_mut_obj_add_strcpy(d, o, "model", r->srv->cfg.model_id);
    yyjson_mut_obj_add_bool(d, o, "stop", true);
    yyjson_mut_obj_add_strcpy(d, o, "stop_type", stop_type);
    yyjson_mut_obj_add_uint(d, o, "tokens_predicted", out_n);
    yyjson_mut_obj_add_uint(d, o, "tokens_evaluated", st->prompt_tokens);
    yyjson_mut_obj_add_uint(d, o, "tokens_cached", st->context_used);
    yyjson_mut_obj_add_bool(d, o, "truncated", false);
    yyjson_mut_val *gs = yyjson_mut_obj_add_obj(d, o, "generation_settings");
    yyjson_mut_obj_add_uint(d, gs, "n_ctx", st->context_size);
    yyjson_mut_obj_add_strcpy(d, gs, "model", r->srv->cfg.model_id);
    yyjson_mut_val *t = yyjson_mut_obj_add_obj(d, o, "timings");
    yyjson_mut_obj_add_uint(d, t, "cache_n", st->cached_tokens);
    yyjson_mut_obj_add_uint(d, t, "prompt_n", st->input_tokens);
    yyjson_mut_obj_add_real(d, t, "prompt_ms", in_ms);
    yyjson_mut_obj_add_real(d, t, "prompt_per_second",
                            in_ms > 0 ? st->input_tokens * 1000.0 / in_ms : 0);
    yyjson_mut_obj_add_uint(d, t, "predicted_n", out_n);
    yyjson_mut_obj_add_real(d, t, "predicted_ms", out_ms);
    yyjson_mut_obj_add_real(d, t, "predicted_per_second",
                            out_ms > 0 ? out_n * 1000.0 / out_ms : 0);
    char *json = yyjson_mut_write(d, 0, n);
    yyjson_mut_doc_free(d);
    return json;
}

int srv_infill(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *root = doc ? yyjson_doc_get_root(doc) : NULL;
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return srv_reply_error(r, 400, "invalid_request_error", NULL,
                               "The body must be a JSON object.");
    }
    int bad = 0;
    size_t n_pre = 0, n_line = 0, n_suf = 0;
    const char *pre = str_of(root, "input_prefix", &n_pre, &bad);
    const char *line = str_of(root, "prompt", &n_line, &bad);
    const char *suf = str_of(root, "input_suffix", &n_suf, &bad);
    long n_predict = num_of(root, "n_predict", -1, &bad);
    long n_indent = num_of(root, "n_indent", 0, &bad);
    long t_max = num_of(root, "t_max_predict_ms", 0, &bad);
    yyjson_val *ex = yyjson_obj_get(root, "input_extra");
    yyjson_val *sm = yyjson_obj_get(root, "stream");
    const char *why = NULL;
    if (bad)
        why = "'input_prefix', 'input_suffix' and 'prompt' must be strings, "
              "'n_predict', 'n_indent' and 't_max_predict_ms' numbers.";
    else if (!pre || !suf)
        why = "'input_prefix' and 'input_suffix' are required.";
    else if (ex && !yyjson_is_null(ex) && !yyjson_is_arr(ex))
        why = "'input_extra' must be an array of {\"filename\": string, "
              "\"text\": string}.";
    else if (sm && yyjson_is_bool(sm) && yyjson_get_bool(sm))
        why = "'stream' is not supported on /infill: the answer comes whole.";
    if (why) {
        yyjson_doc_free(doc);
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "%s", why);
    }

    struct srv_job *j = srv_job_new();
    if (!j || srv_job_choices(j, 1, 1) != 0 ||
        !(j->prompts = calloc(1, sizeof(*j->prompts)))) {
        yyjson_doc_free(doc);
        srv_job_release(j);
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    }
    j->kind = JOB_TEXT;
    j->n_prompts = 1;
    struct srv_prompt *pp = &j->prompts[0];
    pp->n = n_pre + n_line;
    pp->text = malloc(pp->n + 1);
    j->suffix = srv_dup_n(suf, n_suf);
    j->n_suffix = n_suf;
    if (!pp->text || !j->suffix) {
        yyjson_doc_free(doc);
        srv_job_release(j);
        return srv_reply_error(r, 500, "server_error", NULL, "out of memory");
    }
    memcpy(pp->text, pre, n_pre);
    if (n_line)
        memcpy(pp->text + n_pre, line, n_line);
    pp->text[pp->n] = 0;
    if (r->srv->cfg.verbose && ex && yyjson_arr_size(ex))
        fprintf(stderr,
                "janas-server: /infill: %zu chunks of other files "
                "left out\n",
                yyjson_arr_size(ex));
    yyjson_doc_free(doc);

    j->p.temperature = 0;
    j->p.thinking = 0;
    /* n_predict 0: the prompt only, which costs one token written */
    j->prefill_only = n_predict == 0;
    j->p.max_reply = n_predict == 0  ? 1
                     : n_predict > 0 ? (int32_t)n_predict
                                     : 0;
    j->n_indent = n_indent > 0 ? (int)n_indent : 0;
    j->t_max_predict_ms = t_max > 0 ? (int32_t)t_max : 0;

    if (srv_engine_submit(r->srv->engine, j) != 0) {
        srv_job_release(j);
        return srv_reply_error(r, 429, "rate_limit_exceeded", NULL,
                               "The server is busy: %u requests are waiting. "
                               "Try again shortly.",
                               r->srv->cfg.max_queue);
    }
    pthread_mutex_lock(&j->mu);
    while (!j->done)
        pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);
    int ret;
    if (j->err != JANAS_LLM_OK || j->ch[0].finish == JANAS_LLM_FINISH_ERROR) {
        unsigned code =
            j->err == JANAS_LLM_EFULL || j->err == JANAS_LLM_ELIMIT ||
                    j->err == JANAS_LLM_EINVAL || j->err == JANAS_LLM_EMODEL
                ? 400
                : 500;
        ret = srv_reply_error(
            r, code, code == 400 ? "invalid_request_error" : "server_error",
            NULL, "%s", j->errmsg[0] ? j->errmsg : "generation failed");
    } else {
        size_t n = 0;
        char *json = answer(r, j, &n);
        ret = srv_reply_json(r, 200, json, n);
    }
    srv_job_release(j);
    return ret;
}
