/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_nucleus.c - the public API's Nucleus mode (include/janas/llm.h,
 * janas_llm_nucleus; job/docs/progetto_nucleus_cancello.md): the entrance
 * gate opened on the model, the conversation kept for the Specialist, the
 * gate's decision at the reply's first token, and the Specialist's reply
 * passed through (api_reply.c calls these).
 */
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"
#include "llm/gate.h"
#include "llm/model.h"

int32_t janas_llm_nucleus(janas_llm *llm, const char *gate_path,
                          janas_llm *specialist)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    if (!gate_path) {
        janas_llm_model_gate_arm(llm->m, 0);
        llm->nuc_on = 0;
        llm->nuc_spec = NULL;
        return JANAS_LLM_OK;
    }
    if (specialist == llm)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "the specialist is the model itself");
    char err[256];
    if (janas_llm_model_gate_open(llm->m, gate_path, err, sizeof(err)) != 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "%s", err);
    llm->nuc_on = 1;
    llm->nuc_spec = specialist;
    return JANAS_LLM_OK;
}

void janas_api_nuc_forget(janas_llm_chat *c)
{
    for (size_t i = 0; i < c->n_nuc; i++)
        free(c->nuc_texts[i]);
    free(c->nuc_texts);
    free(c->nuc_roles);
    c->nuc_texts = NULL;
    c->nuc_roles = NULL;
    c->n_nuc = 0;
}

void janas_api_nuc_free(janas_llm_chat *c)
{
    janas_api_nuc_forget(c);
    if (c->nuc_chat)
        janas_llm_chat_destroy(c->nuc_chat);
    c->nuc_chat = NULL;
}

/* The conversation about to run (janas_llm_chat_run's messages), kept
   for the Specialist. */
int janas_api_nuc_keep(janas_llm_chat *c)
{
    janas_api_nuc_forget(c);
    if (!c->llm->nuc_on || c->n_msgs == 0)
        return 0;
    size_t n = c->n_msgs;
    c->nuc_roles = malloc(n * sizeof(int32_t));
    c->nuc_texts = calloc(n, sizeof(char *));
    if (!c->nuc_roles || !c->nuc_texts)
        return -1;
    for (size_t i = 0; i < n; i++) {
        const struct cmsg *m = &c->msgs[i];
        if (!(c->nuc_texts[i] = malloc(m->n + 1)))
            return -1;
        memcpy(c->nuc_texts[i], m->text ? m->text : "", m->n);
        c->nuc_texts[i][m->n] = 0;
        c->nuc_roles[i] = m->role;
        c->n_nuc = i + 1;
    }
    return 0;
}

/* Before the reply's first token: the gate keeps the state of the
   prompt's last token, which that first pass computes. */
void janas_api_nuc_arm(janas_llm_chat *c)
{
    c->nuc_route = JANAS_LLM_ROUTE_SELF;
    c->stats.route = JANAS_LLM_ROUTE_SELF;
    c->stats.route_domain[0] = 0;
    c->nuc_pending = 0;
    /* tools and grammars stay with the model: a reply bound to them is not
       the gate's to divert */
    if (!c->llm->nuc_on || c->tools || c->g)
        return;
    janas_llm_model_gate_arm(c->llm->m, 1);
    janas_llm_model_gate_at(c->llm->m, janas_llm_session_computed(c->s));
    c->nuc_pending = 1;
}

/* Whether the last user message reads as Italian: its common words
   against English ones (the sentence of "I don't know" in its language). */
static int italian(const janas_llm_chat *c)
{
    static const char *const IT[] = {
        "il",  "la",   "di",     "che", "è",     "un", "una", "cos",
        "chi", "come", "perché", "del", "della", "mi", "per", "non"};
    static const char *const EN[] = {"the", "is",  "of",  "what", "who",  "a",
                                     "an",  "how", "why", "did",  "does", "me",
                                     "for", "and", "was", "tell"};
    const char *t = NULL;
    for (size_t i = c->n_nuc; i-- > 0;)
        if (c->nuc_roles[i] == JANAS_LLM_ROLE_USER) {
            t = c->nuc_texts[i];
            break;
        }
    if (!t)
        return 0;
    int it = 0, en = 0;
    char w[32];
    for (const char *p = t; *p;) {
        size_t n = 0;
        while (*p && (isalpha((unsigned char)*p) || (*p & 0x80))) {
            if (n + 1 < sizeof(w))
                w[n++] = (char)tolower((unsigned char)*p);
            p++;
        }
        w[n] = 0;
        for (size_t k = 0; n && k < sizeof(IT) / sizeof(IT[0]); k++)
            it += !strcmp(w, IT[k]);
        for (size_t k = 0; n && k < sizeof(EN) / sizeof(EN[0]); k++)
            en += !strcmp(w, EN[k]);
        if (*p)
            p++;
    }
    return it > en;
}

/* After the reply's first pass: the gate's decision. Returns the sentence
   to answer with ("I don't know"), or NULL; the route set. */
const char *janas_api_nuc_decide(janas_llm_chat *c)
{
    if (!c->nuc_pending)
        return NULL;
    c->nuc_pending = 0;
    struct janas_llm_model *m = c->llm->m;
    uint32_t dom = 0;
    int o = janas_llm_model_gate_result(m, &dom, NULL);
    janas_llm_model_gate_arm(m, 0);
    if (o < 0)
        return NULL;
    snprintf(c->stats.route_domain, sizeof(c->stats.route_domain), "%s",
             janas_llm_model_gate_domain(m, dom));
    /* "I don't know" is the last word of the model that knows the most: the
       Nucleus's goes to the Specialist, which reads it with a gate of its
       own (janas_llm_nucleus on it, without a Specialist) */
    if (o == JANAS_GATE_UNKNOWN && !c->llm->nuc_spec) {
        c->nuc_route = c->stats.route = JANAS_LLM_ROUTE_UNKNOWN;
        return italian(c) ? "Non ne ho notizia: non conosco ciò di cui mi "
                            "parli. Forse è un errore, o qualcosa che non so. "
                            "Puoi dirmi dove l'hai trovato?"
                          : "I have no record of this: I don't know what you "
                            "are referring to. It may be a mistake, or "
                            "something I don't know. Where did you find it?";
    }
    if ((o == JANAS_GATE_SPECIALIST || o == JANAS_GATE_UNKNOWN) &&
        c->llm->nuc_spec && c->n_nuc) {
        if (!c->nuc_chat && janas_llm_chat_create(c->llm->nuc_spec, &c->p,
                                                  &c->nuc_chat) != JANAS_LLM_OK)
            return NULL; /* the Specialist busy elsewhere: answered here */
        /* the request's parameters (reasoning, lengths, sampling) */
        if (janas_llm_chat_set_params(c->nuc_chat, &c->p) != JANAS_LLM_OK)
            return NULL;
        if (janas_llm_chat_load(c->nuc_chat, (int32_t)c->n_nuc, c->nuc_roles,
                                (const char *const *)c->nuc_texts,
                                NULL) != JANAS_LLM_OK ||
            janas_api_nuc_keep(c->nuc_chat) != 0) /* its gate's language */
            return NULL;
        c->nuc_route = c->stats.route = JANAS_LLM_ROUTE_SPECIALIST;
        c->nuc_handoff = janas_api_now() - c->t_send;
    }
    return NULL;
}
