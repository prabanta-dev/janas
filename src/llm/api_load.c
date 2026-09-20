/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_load.c - the public API: a conversation handed over whole, by a
 * caller that keeps it (an HTTP client does), or raw text to continue (see
 * include/janas/llm.h). What it shares with the sequence already computed
 * is not read again.
 */
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"

static void free_msgs(janas_llm_chat *c)
{
    for (size_t i = 0; i < c->n_msgs; i++) {
        struct cmsg *m = &c->msgs[i];
        free(m->text);
        for (size_t k = 0; k < m->n_calls; k++) {
            free((char *)m->calls[k].name);
            free((char *)m->calls[k].args);
        }
        free(m->calls);
    }
    c->n_msgs = 0;
}

void janas_api_msgs_free(janas_llm_chat *c)
{
    free_msgs(c);
    free(c->msgs);
    c->msgs = NULL;
    c->cap_msgs = 0;
}

static char *dup_n(const char *s, size_t n)
{
    char *d = malloc(n + 1);
    if (d) {
        memcpy(d, s, n);
        d[n] = 0;
    }
    return d;
}

int32_t janas_llm_chat_begin(janas_llm_chat *c)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    free_msgs(c);
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_add(janas_llm_chat *c, int32_t role, const char *text,
                           int32_t len)
{
    if (!c || !text || role < JANAS_LLM_ROLE_SYSTEM ||
        role > JANAS_LLM_ROLE_TOOL)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    if (role == JANAS_LLM_ROLE_SYSTEM && c->n_msgs > 0)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "a system message comes first or not at all");
    if (c->n_msgs == c->cap_msgs) {
        size_t cap = c->cap_msgs ? 2 * c->cap_msgs : 16;
        struct cmsg *t = realloc(c->msgs, cap * sizeof(*t));
        if (!t)
            return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
        c->msgs = t;
        c->cap_msgs = cap;
    }
    size_t n = janas_api_text_len(text, len);
    char *t = dup_n(text, n);
    if (!t)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    c->msgs[c->n_msgs++] = (struct cmsg){.role = role, .text = t, .n = n};
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_add_call(janas_llm_chat *c, const char *name,
                                int32_t name_len, const char *args,
                                int32_t args_len)
{
    if (!c || !name || !args)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    if (c->n_msgs == 0 ||
        c->msgs[c->n_msgs - 1].role != JANAS_LLM_ROLE_ASSISTANT)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "a tool call belongs to an assistant's message");
    struct cmsg *m = &c->msgs[c->n_msgs - 1];
    struct janas_tool_call *t =
        realloc(m->calls, (m->n_calls + 1) * sizeof(*t));
    if (!t)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    m->calls = t;
    size_t nn = janas_api_text_len(name, name_len);
    size_t an = janas_api_text_len(args, args_len);
    char *nd = dup_n(name, nn), *ad = dup_n(args, an);
    if (!nd || !ad) {
        free(nd);
        free(ad);
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    }
    m->calls[m->n_calls++] = (struct janas_tool_call){
        .name = nd, .name_n = nn, .args = ad, .args_n = an};
    return JANAS_LLM_OK;
}

static const struct memo *memo_find(const janas_llm_chat *c,
                                    const struct cmsg *msg)
{
    struct janas_buf key = {0};
    janas_api_memo_key(&key, msg->text, msg->n, msg->calls, msg->n_calls);
    const struct memo *found = NULL;
    for (size_t i = 0; !key.oom && i < sizeof(c->memo) / sizeof(c->memo[0]);
         i++) {
        const struct memo *m = &c->memo[i];
        if (m->ids && m->n_text == key.n &&
            (key.n == 0 || memcmp(m->text, key.p, key.n) == 0)) {
            found = m;
            break;
        }
    }
    janas_buf_free(&key);
    return found;
}

static int add_ids(janas_llm_chat *c, const int32_t *ids, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        if (janas_api_add_special(c, ids[i]) != 0)
            return -1;
    return 0;
}

/*
 * An assistant's message the model did not write here (or not lately): as
 * the template writes it, with its calls. A Qwen3.5 or 3.6 template gives
 * the turns after the user's last question their reasoning back, empty
 * when there is none to give.
 */
static int add_assistant(janas_llm_chat *c, const struct cmsg *m, int after_q)
{
    const janas_llm *llm = c->llm;
    enum janas_tool_dialect d = llm->tools;
    int err = janas_api_add_special(c, llm->im_start);
    /* Qwen3-Coder writes the calls right after the role */
    int bare = d == JANAS_TOOLS_XML_CODER && m->n_calls;
    err |= janas_api_add_text(c, bare ? "assistant" : "assistant\n",
                              bare ? 9 : 10);
    if (d == JANAS_TOOLS_XML && llm->thinker && after_q)
        err |= janas_api_add_markup(c, "<think>\n\n</think>\n\n", 19);
    if (m->n_calls && d != JANAS_TOOLS_NONE) {
        struct janas_buf b = {0};
        janas_tools_assistant(&b, d, m->text, m->n, m->calls, m->n_calls);
        err |= b.oom || janas_api_add_markup(c, b.p, b.n);
        janas_buf_free(&b);
    } else {
        err |= janas_api_add_text(c, m->text, m->n);
    }
    err |= janas_api_add_special(c, llm->im_end);
    err |= janas_api_add_text(c, "\n", 1);
    return err ? -1 : 0;
}

/* The messages built, in the model's format, into c->ids; the reply's turn
   started. *reply_off gets where it starts. */
static int render(janas_llm_chat *c, uint32_t *reply_off)
{
    const janas_llm *llm = c->llm;
    size_t last_q = 0; /* the user's last question */
    for (size_t i = 0; i < c->n_msgs; i++)
        if (c->msgs[i].role == JANAS_LLM_ROLE_USER)
            last_q = i;
    int err = 0;
    if (c->tools && c->msgs[0].role != JANAS_LLM_ROLE_SYSTEM) {
        /* with tools there is a system message whatever the caller says */
        err |= janas_api_add_system(c, NULL, 0);
        err |= janas_api_flush_text(c);
        c->sys_tokens = (uint32_t)c->n_ids;
    }
    for (size_t i = 0; i < c->n_msgs && !err; i++) {
        const struct cmsg *m = &c->msgs[i];
        switch (m->role) {
        case JANAS_LLM_ROLE_SYSTEM:
            err |= janas_api_add_system(c, m->text, m->n);
            err |= janas_api_flush_text(c);
            c->sys_tokens = (uint32_t)c->n_ids;
            break;
        case JANAS_LLM_ROLE_USER:
            err |= janas_api_flush_text(c);
            c->last_at = (uint32_t)c->n_ids; /* the last one's, at the end */
            err |= janas_api_note_turn(c, (uint32_t)c->n_ids);
            err |= janas_api_add_message(c, "user", m->text, m->n);
            break;
        case JANAS_LLM_ROLE_ASSISTANT: {
            const struct memo *mm = memo_find(c, m);
            if (mm) { /* one of the last replies: as the model wrote it */
                err |= add_ids(c, mm->ids, mm->n);
                if (!mm->closed)
                    err |= janas_api_add_special(c, llm->im_end);
                err |= janas_api_add_text(c, "\n", 1);
            } else {
                err |= add_assistant(c, m, i > last_q);
            }
            break;
        }
        case JANAS_LLM_ROLE_TOOL: {
            /* the answers of the tools, one user turn for those in a row */
            int first = i == 0 || c->msgs[i - 1].role != JANAS_LLM_ROLE_TOOL;
            int last = i + 1 == c->n_msgs ||
                       c->msgs[i + 1].role != JANAS_LLM_ROLE_TOOL;
            if (first) {
                err |= janas_api_flush_text(c);
                c->last_at = (uint32_t)c->n_ids;
                err |= janas_api_add_special(c, llm->im_start) ||
                       janas_api_add_text(c, "user", 4);
            }
            err |= janas_api_add_markup(c, "\n<tool_response>\n", 17);
            err |= janas_api_add_text(c, m->text, m->n);
            err |= janas_api_add_markup(c, "\n</tool_response>", 17);
            if (last)
                err |= janas_api_add_special(c, llm->im_end) ||
                       janas_api_add_text(c, "\n", 1);
            break;
        }
        }
    }
    err |= janas_api_flush_text(c);
    *reply_off = (uint32_t)c->n_ids;
    err |= janas_api_start_reply(c);
    return err ? -1 : 0;
}

/*
 * The sequence built in c->ids replaces the session's, reusing the longest
 * part they share: when it continues the conversation already computed
 * (the usual case), only the new tokens are read. reply_off is where the
 * reply's turn starts in c->ids, or UINT32_MAX for raw text.
 */
static int32_t load_ids(janas_llm_chat *c, uint32_t reply_off)
{
    if (janas_api_flush_text(c) != 0)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    uint32_t reserve = c->p.max_reply > 0 && c->p.max_reply < 1024
                           ? (uint32_t)c->p.max_reply
                           : 1024;
    int32_t lim = janas_api_check_input(c, (uint32_t)c->n_ids);
    if (lim != JANAS_LLM_OK)
        return lim;
    if (c->n_ids + reserve > c->llm->n_ctx)
        return janas_api_fail(
            JANAS_LLM_EFULL,
            "the conversation (%zu tokens) and a reply do not fit in "
            "the context (%u tokens)",
            c->n_ids, c->llm->n_ctx);
    if (c->llm->mode == JANAS_LLM_MODE_AUTO)
        janas_api_apply_mode(c->llm);
    janas_api_keep_pick(c);
    const int32_t *h;
    uint32_t hn = janas_llm_session_tokens(c->s, &h), same = 0;
    while (same < hn && same < c->n_ids && h[same] == c->ids[same])
        same++;
    /* the last token must be read for its logits: when it has been, keep
       one less (a prompt kept before its reply has not: it is kept whole,
       which on a recurrent model is the only way it can be) */
    if (same == c->n_ids && same > 0 &&
        janas_llm_session_computed(c->s) >= same)
        same--;
    if (janas_llm_session_truncate(c->s, same) != 0) {
        janas_llm_session_reset(c->s); /* a recurrent state: from the start */
        same = 0;
    }
    janas_llm_session_stats(c->s, &c->st0);
    janas_expert_cache_stats(janas_llm_model_cache(c->llm->m), &c->cs0);
    c->t_send = janas_api_now();
    uint32_t from = janas_api_keep_system(c, same);
    if (janas_llm_session_append(c->s, c->ids + from,
                                 (uint32_t)c->n_ids - from) != 0) {
        c->replying = 0;
        return janas_api_fail(JANAS_LLM_EFULL,
                              "the context is full (%u tokens)", c->llm->n_ctx);
    }
    c->dropped = 0;
    c->loaded = (uint32_t)c->n_ids;
    if (janas_api_begin_reply(c, reply_off, (uint32_t)c->n_ids, same) != 0)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    /* the prompt is kept once read, which janas_llm_chat_next does a block
       at a time: reading it here would leave the caller without word for
       as long as it takes */
    c->keep_prompt = 1;
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_run(janas_llm_chat *c)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    if (c->n_msgs == 0 || (c->msgs[c->n_msgs - 1].role != JANAS_LLM_ROLE_USER &&
                           c->msgs[c->n_msgs - 1].role != JANAS_LLM_ROLE_TOOL))
        return janas_api_fail(
            JANAS_LLM_EINVAL,
            "the last message must be the user's or a tool's");
    janas_api_prep_start(c);
    c->tools_tokens = 0; /* until a system message with tools says */
    janas_api_reply_clear(c);
    c->n_ids = c->n_text = 0;
    c->n_turn = 0;
    c->sys_tokens = 0;
    free(c->system);
    c->system = NULL;
    uint32_t reply_off;
    if (render(c, &reply_off) != 0)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    return load_ids(c, reply_off);
}

int32_t janas_llm_chat_count(janas_llm_chat *c, int32_t *n)
{
    if (!c || !n)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    if (c->n_msgs == 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "no messages");
    if (c->replying)
        return janas_api_fail(JANAS_LLM_EBUSY, "a reply is being read");
    /* built as a run would build it, then forgotten: the turns and the
       system message's length belong to the conversation computed */
    uint32_t sys = c->sys_tokens;
    size_t turns = c->n_turn;
    uint32_t *saved = turns ? malloc(turns * sizeof(*saved)) : NULL;
    if (turns && !saved)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    if (turns)
        memcpy(saved, c->turn, turns * sizeof(*saved));
    c->n_ids = c->n_text = 0;
    c->n_turn = 0;
    uint32_t reply_off;
    int err = render(c, &reply_off) || janas_api_flush_text(c);
    *n = (int32_t)c->n_ids;
    c->n_ids = c->n_text = 0;
    c->sys_tokens = sys;
    c->n_turn = 0;
    for (size_t i = 0; i < turns && !err; i++)
        err = janas_api_note_turn(c, saved[i]);
    free(saved);
    if (err)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_load(janas_llm_chat *c, int32_t n, const int32_t *roles,
                            const char *const *texts, const int32_t *lens)
{
    if (!c || n < 1 || !roles || !texts)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    free_msgs(c);
    for (int32_t i = 0; i < n; i++) {
        if (!texts[i] || roles[i] < JANAS_LLM_ROLE_SYSTEM ||
            roles[i] > JANAS_LLM_ROLE_TOOL ||
            (roles[i] == JANAS_LLM_ROLE_SYSTEM && i > 0))
            return janas_api_fail(
                JANAS_LLM_EINVAL,
                "message %d: invalid role, or a system message after the "
                "first",
                (int)i);
        int32_t rc =
            janas_llm_chat_add(c, roles[i], texts[i], lens ? lens[i] : -1);
        if (rc != JANAS_LLM_OK)
            return rc;
    }
    return janas_llm_chat_run(c);
}

int32_t janas_llm_chat_prompt(janas_llm_chat *c, const char *text, int32_t len)
{
    if (!c || !text)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    janas_api_prep_start(c);
    c->tools_tokens = 0;
    janas_api_reply_clear(c);
    c->n_ids = c->n_text = 0;
    c->n_turn = 0;
    c->sys_tokens = 0;
    if (janas_api_add_text(c, text, janas_api_text_len(text, len)) != 0)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    c->in_think = 0;
    return load_ids(c, UINT32_MAX);
}

int32_t janas_llm_chat_infill(janas_llm_chat *c, const char *prefix,
                              int32_t len, const char *suffix,
                              int32_t suffix_len)
{
    if (!c || !prefix || !suffix)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    const struct janas_tokenizer *tok = c->llm->tok;
    int32_t pre = janas_tokenizer_find(tok, "<|fim_prefix|>");
    int32_t suf = janas_tokenizer_find(tok, "<|fim_suffix|>");
    int32_t mid = janas_tokenizer_find(tok, "<|fim_middle|>");
    if (pre < 0 || suf < 0 || mid < 0)
        return janas_api_fail(JANAS_LLM_EMODEL,
                              "this model cannot fill in text: its "
                              "vocabulary has no <|fim_...|> tokens");
    janas_api_prep_start(c);
    c->tools_tokens = 0;
    janas_api_reply_clear(c);
    c->n_ids = c->n_text = 0;
    c->n_turn = 0;
    c->sys_tokens = 0;
    int err =
        janas_api_add_special(c, pre) ||
        janas_api_add_text(c, prefix, janas_api_text_len(prefix, len)) ||
        janas_api_add_special(c, suf) ||
        janas_api_add_text(c, suffix, janas_api_text_len(suffix, suffix_len)) ||
        janas_api_add_special(c, mid);
    if (err)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    c->in_think = 0;
    return load_ids(c, UINT32_MAX);
}
