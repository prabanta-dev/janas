/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_aside.c - a request on the side (janas_llm_chat_aside in
 * include/janas/llm.h): the conversation saved, a system message and a
 * user message answered greedily, the conversation put back as it was.
 * janas-chat asks the model this way to translate a service's layout once,
 * in the middle of a conversation it must not disturb.
 */
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"

/* A token the reply is not made of: the markers of the format. */
static int marker(const janas_llm *llm, int32_t t)
{
    return t == llm->im_start || t == llm->im_end || t == llm->think_open ||
           t == llm->think_close || t == llm->bos;
}

static int is_stop(const janas_llm *llm, int32_t t)
{
    if (t == llm->im_end || t == llm->eos_id)
        return 1;
    for (int i = 0; i < llm->n_stop; i++)
        if (t == llm->stop[i])
            return 1;
    return 0;
}

int32_t janas_llm_chat_aside(janas_llm_chat *c, const char *system,
                             int32_t system_len, const char *text,
                             int32_t text_len, int32_t max_tokens, char *buf,
                             int32_t cap, int32_t *len)
{
    if (len)
        *len = 0;
    if (!c || !text || max_tokens <= 0 || (cap > 0 && !buf))
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    if (c->replying && !c->ended)
        return janas_api_fail(JANAS_LLM_EBUSY,
                              "a reply is being read: ask aside after it");
    const janas_llm *llm = c->llm;
    struct janas_llm_saved *sv = janas_llm_session_save(c->s);
    if (!sv)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    /* what building a message changes, kept to be put back */
    int thinking = c->p.thinking, in_think = c->in_think, in_turn = c->in_turn,
        chan = c->chan_label;
    c->p.thinking = 0; /* no reasoning: an answer, at once */
    c->in_turn = 0;
    c->n_ids = c->n_text = 0;
    int err = janas_api_begin_sequence(c);
    if (system)
        err |= janas_api_add_message(c, "system", system,
                                     janas_api_text_len(system, system_len));
    err |= janas_api_add_message(c, "user", text,
                                 janas_api_text_len(text, text_len));
    err |= janas_api_start_reply(c);
    struct janas_gen_options o;
    int32_t rc = JANAS_LLM_OK;
    struct janas_buf out = {0};
    if (err) {
        rc = janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    } else if (janas_api_options(c, &o) != JANAS_LLM_OK) {
        rc = JANAS_LLM_EINVAL;
    } else {
        o.temperature = 0; /* greedy: the same answer every time */
        o.presence_penalty = o.frequency_penalty = 0;
        o.logprobs = o.top_logprobs = 0;
        janas_llm_session_filter(c->s, NULL);
        janas_llm_session_reset(c->s);
        if (janas_llm_session_set_options(c->s, &o) != 0 ||
            janas_llm_session_append(c->s, c->ids, (uint32_t)c->n_ids) != 0)
            rc = janas_api_fail(JANAS_LLM_EFULL,
                                "the request does not fit in the context");
        for (int32_t k = 0; rc == JANAS_LLM_OK && k < max_tokens; k++) {
            int32_t t;
            int r = janas_llm_session_next(c->s, &t);
            if (r != 0) {
                if (r < 0)
                    rc = janas_api_fail(JANAS_LLM_EFAIL, "the pass failed");
                break;
            }
            if (is_stop(llm, t))
                break;
            if (marker(llm, t))
                continue;
            char piece[256];
            size_t n = janas_tokenizer_decode(llm->tok, t, piece, sizeof piece);
            janas_buf_put(&out, piece, n < sizeof piece ? n : sizeof piece);
        }
    }
    /* the conversation back, as it was */
    if (janas_llm_session_restore(c->s, sv) != 0 && rc == JANAS_LLM_OK)
        rc = janas_api_fail(JANAS_LLM_ENOMEM,
                            "the conversation could not be put back");
    janas_llm_saved_free(sv);
    c->p.thinking = thinking;
    c->in_think = in_think;
    c->in_turn = in_turn;
    c->chan_label = chan;
    c->n_ids = c->n_text = 0;
    if (janas_api_options(c, &o) == JANAS_LLM_OK)
        janas_llm_session_set_options(c->s, &o);
    if (rc == JANAS_LLM_OK && out.oom)
        rc = janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    if (rc == JANAS_LLM_OK)
        rc = janas_api_copy_out(out.p ? out.p : "", out.n, buf, cap, len);
    janas_buf_free(&out);
    return rc;
}
