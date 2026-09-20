/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * moderations.c - POST /moderations, answered by the chat model: asked, at
 * temperature 0, to classify the text into OpenAI's thirteen categories,
 * its answer held to a schema of thirteen booleans. A category's score is
 * the chance the model gave "true" against "false" where it wrote it, from
 * the log-probabilities of that token. This is a general model reading the
 * text with instructions, not a classifier trained for it: the scores say
 * how sure the model was, not how often it is right.
 */
#include "request.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const CATS[] = {
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
};
#define N_CATS (sizeof(CATS) / sizeof(CATS[0]))

static const char SYSTEM[] =
    "You are a content moderation classifier. For the text the user gives, "
    "decide for each category whether the text belongs to it, and answer "
    "with the JSON object only. Categories: harassment (harassing language "
    "towards anyone); harassment/threatening (harassment that also threatens "
    "violence or serious harm); hate (hatred based on race, gender, "
    "ethnicity, religion, nationality, sexual orientation, disability or "
    "caste); hate/threatening (hateful content that also threatens violence "
    "or serious harm); illicit (advice or instructions for wrongdoing); "
    "illicit/violent (the same, involving violence or weapons); self-harm "
    "(promoting or depicting self-harm, suicide, eating disorders); "
    "self-harm/intent (the speaker intends to harm themselves); "
    "self-harm/instructions (instructions or advice for self-harm); sexual "
    "(content meant to arouse, or sexual services); sexual/minors (sexual "
    "content involving anyone under 18); violence (depicting death, violence "
    "or physical injury); violence/graphic (the same, in graphic detail). "
    "Ordinary text belongs to none.";

/* The schema: every category a required boolean, in order. */
static char *schema(void)
{
    char *s = malloc(2048);
    if (!s)
        return NULL;
    int w = snprintf(s, 2048, "{\"type\":\"object\",\"properties\":{");
    for (size_t i = 0; i < N_CATS; i++)
        w += snprintf(s + w, 2048 - (size_t)w,
                      "%s\"%s\":{\"type\":\"boolean\"}", i ? "," : "", CATS[i]);
    w += snprintf(s + w, 2048 - (size_t)w, "},\"required\":[");
    for (size_t i = 0; i < N_CATS; i++)
        w += snprintf(s + w, 2048 - (size_t)w, "%s\"%s\"", i ? "," : "",
                      CATS[i]);
    snprintf(s + w, 2048 - (size_t)w, "],\"additionalProperties\":false}");
    return s;
}

/* 1 "true", 0 "false", -1 neither: a token's text, spaces and punctuation
   around it aside. */
static int truth(const char *t, size_t n)
{
    while (n && (*t == ' ' || *t == ':' || *t == '\n')) {
        t++;
        n--;
    }
    if (n >= 4 && !memcmp(t, "true", 4))
        return 1;
    if (n >= 5 && !memcmp(t, "false", 5))
        return 0;
    return -1;
}

/* The categories and their scores from the answer's tokens: the k-th
   boolean written is category k. 0, or -1 when the answer is not whole. */
static int scores(const struct srv_choice *ch, int *flag, double *score)
{
    size_t k = 0;
    for (size_t i = 0; i < ch->n_lp && k < N_CATS; i++) {
        const struct srv_lp *t = &ch->lp[i];
        int v = truth(t->t.tok, t->t.n);
        if (v < 0)
            continue;
        double pt = 0, pf = 0;
        for (int q = 0; q < t->n_top; q++) {
            int a = truth(t->top[q].tok, t->top[q].n);
            if (a == 1)
                pt += exp(t->top[q].lp);
            else if (a == 0)
                pf += exp(t->top[q].lp);
        }
        flag[k] = v;
        score[k] = pt + pf > 0 ? pt / (pt + pf) : v;
        k++;
    }
    return k == N_CATS ? 0 : -1;
}

/* One text through the model. */
static int classify(struct srv_req *r, const char *text, size_t n, int *flag,
                    double *score, char *err, size_t el)
{
    struct srv_job *j = srv_job_new();
    char *sch = schema();
    char *sys = srv_dup_n(SYSTEM, sizeof(SYSTEM) - 1);
    char *user = malloc(n + 64);
    int rc = -1;
    if (!j || !sch || !sys || !user || srv_job_choices(j, 1, 1) != 0 ||
        !(j->msg = calloc(2, sizeof(*j->msg)))) {
        snprintf(err, el, "out of memory");
        goto out;
    }
    int un = snprintf(user, n + 64, "Text to classify:\n\n%.*s", (int)n, text);
    j->msg[0] = (struct srv_msg){
        .role = JANAS_LLM_ROLE_SYSTEM, .text = sys, .n = sizeof(SYSTEM) - 1};
    j->msg[1] = (struct srv_msg){
        .role = JANAS_LLM_ROLE_USER, .text = user, .n = (size_t)un};
    sys = user = NULL;
    j->n_msg = 2;
    j->kind = JOB_CHAT;
    j->format = JANAS_LLM_FORMAT_SCHEMA;
    j->schema = sch;
    j->n_schema = strlen(sch);
    sch = NULL;
    j->p.temperature = 0;
    j->p.thinking = 0;
    j->p.max_reply = 400;
    j->p.logprobs = 1;
    j->p.top_logprobs = 5;
    if (srv_engine_submit(r->srv->engine, j) != 0) {
        snprintf(err, el, "the server is busy");
        goto out;
    }
    pthread_mutex_lock(&j->mu);
    while (!j->done)
        pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);
    if (j->err != JANAS_LLM_OK)
        snprintf(err, el, "%s", j->errmsg);
    else if (scores(&j->ch[0], flag, score) != 0)
        snprintf(err, el, "the model's answer was cut short");
    else
        rc = 0;
out:
    free(sch);
    free(sys);
    free(user);
    srv_job_release(j);
    return rc;
}

/* The text of an input: a string, or its text parts. */
static char *input_text(yyjson_val *v, size_t *n)
{
    if (yyjson_is_str(v)) {
        *n = yyjson_get_len(v);
        return srv_dup_n(yyjson_get_str(v), *n);
    }
    const char *t = yyjson_get_str(yyjson_obj_get(v, "type"));
    const char *s = yyjson_get_str(yyjson_obj_get(v, "text"));
    if (!t || strcmp(t, "text") || !s)
        return NULL;
    *n = strlen(s);
    return srv_dup_n(s, *n);
}

int srv_moderations_create(struct srv_req *r)
{
    yyjson_doc *doc = yyjson_read(r->body, r->n_body, 0);
    yyjson_val *in = yyjson_obj_get(yyjson_doc_get_root(doc), "input");
    size_t n = yyjson_is_arr(in) ? yyjson_arr_size(in) : 1;
    if (!in || n == 0 || n > 32) {
        yyjson_doc_free(doc);
        return srv_reply_error(r, 400, "invalid_request_error", "invalid_value",
                               "'input' is a string, or an array of at most "
                               "32 strings or text parts.");
    }
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *o = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, o);
    char id[64];
    srv_new_id(id, sizeof(id), "modr-");
    yyjson_mut_obj_add_strcpy(d, o, "id", id);
    yyjson_mut_obj_add_str(d, o, "model", r->srv->cfg.model_id);
    yyjson_mut_val *results = yyjson_mut_obj_add_arr(d, o, "results");
    for (size_t i = 0; i < n; i++) {
        size_t tn;
        char *text =
            input_text(yyjson_is_arr(in) ? yyjson_arr_get(in, i) : in, &tn);
        if (!text) {
            yyjson_mut_doc_free(d);
            yyjson_doc_free(doc);
            return srv_reply_error(r, 400, "invalid_request_error",
                                   "unsupported_value",
                                   "input %zu: only text can be moderated "
                                   "here.",
                                   i);
        }
        int flag[N_CATS];
        double score[N_CATS];
        char err[256];
        int rc = classify(r, text, tn, flag, score, err, sizeof(err));
        free(text);
        if (rc != 0) {
            yyjson_mut_doc_free(d);
            yyjson_doc_free(doc);
            return srv_reply_error(r, 500, "server_error", NULL, "%s", err);
        }
        yyjson_mut_val *res = yyjson_mut_arr_add_obj(d, results);
        int any = 0;
        yyjson_mut_val *cats = yyjson_mut_obj(d), *sc = yyjson_mut_obj(d),
                       *ty = yyjson_mut_obj(d);
        for (size_t k = 0; k < N_CATS; k++) {
            any |= flag[k];
            yyjson_mut_obj_add_bool(d, cats, CATS[k], flag[k]);
            yyjson_mut_obj_add_real(d, sc, CATS[k], score[k]);
            yyjson_mut_val *a = yyjson_mut_obj_add_arr(d, ty, CATS[k]);
            yyjson_mut_arr_add_str(d, a, "text");
        }
        yyjson_mut_obj_add_bool(d, res, "flagged", any);
        yyjson_mut_obj_add_val(d, res, "categories", cats);
        yyjson_mut_obj_add_val(d, res, "category_scores", sc);
        yyjson_mut_obj_add_val(d, res, "category_applied_input_types", ty);
    }
    yyjson_doc_free(doc);
    size_t len;
    char *json = yyjson_mut_write(d, 0, &len);
    yyjson_mut_doc_free(d);
    return srv_reply_json(r, 200, json, len);
}
