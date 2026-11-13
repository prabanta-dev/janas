/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_transcript.c - the conversation as it was said (see include/janas/llm.h,
 * janas_llm_chat_messages): every message of a chat, the replies with their
 * reasoning and their calls, written down as they come, so that a program
 * can show it again, save it, export it, or take it up later. A
 * conversation taken up (api_chatfile.c) is loaded whole at its next
 * message, the oldest exchanges left out when it does not fit.
 */
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llm/chat.h"

static char *dup_n(const char *s, size_t n)
{
    char *d = malloc(n + 1);
    if (d) {
        if (n)
            memcpy(d, s, n);
        d[n] = 0;
    }
    return d;
}

static void msg_free(struct tmsg *m)
{
    free(m->text);
    free(m->reasoning);
    for (size_t k = 0; k < m->n_calls; k++) {
        free(m->calls[k].name);
        free(m->calls[k].args);
    }
    free(m->calls);
    *m = (struct tmsg){0};
}

void janas_api_tr_clear(janas_llm_chat *c)
{
    for (size_t i = 0; i < c->n_tr; i++)
        msg_free(&c->tr[i]);
    c->n_tr = 0;
    c->tr_reply = c->tr_pending = 0;
    free(c->tr_title);
    c->tr_title = NULL;
    c->tr_titled = 0;
    c->tr_created = 0;
    free(c->tr_path); /* a new conversation, a new file */
    c->tr_path = NULL;
}

void janas_api_tr_free(janas_llm_chat *c)
{
    janas_api_tr_clear(c);
    free(c->tr);
    c->tr = NULL;
    c->cap_tr = 0;
    free(c->tr_dir);
    c->tr_dir = NULL;
}

/* The title from the first question: its first line, at most 60 bytes cut
   at a word, whole characters. */
void janas_api_tr_autotitle(janas_llm_chat *c)
{
    if (c->tr_titled)
        return;
    for (size_t i = 0; i < c->n_tr; i++) {
        if (c->tr[i].role != JANAS_LLM_ROLE_USER)
            continue;
        const char *s = c->tr[i].text;
        while (*s && (unsigned char)*s <= ' ')
            s++;
        size_t n = strcspn(s, "\r\n");
        if (n > 60) {
            size_t k = 60;
            while (k > 0 && ((unsigned char)s[k] & 0xC0) == 0x80)
                k--;
            size_t w = k;
            while (w > 30 && s[w] != ' ')
                w--;
            n = s[w] == ' ' ? w : k;
        }
        while (n && (unsigned char)s[n - 1] <= ' ')
            n--;
        if (n == 0)
            continue;
        char *t = dup_n(s, n);
        if (t) {
            free(c->tr_title);
            c->tr_title = t;
        }
        return;
    }
}

struct tmsg *janas_api_tr_push(janas_llm_chat *c, int32_t role,
                               const char *text, size_t n)
{
    if (c->n_tr == c->cap_tr) {
        size_t cap = c->cap_tr ? 2 * c->cap_tr : 16;
        struct tmsg *t = realloc(c->tr, cap * sizeof(*t));
        if (!t)
            return NULL;
        c->tr = t;
        c->cap_tr = cap;
    }
    char *d = dup_n(text ? text : "", text ? n : 0);
    if (!d)
        return NULL;
    if (c->tr_created == 0)
        c->tr_created = (int64_t)time(NULL);
    struct tmsg *m = &c->tr[c->n_tr++];
    *m = (struct tmsg){.role = role, .text = d, .at = (int64_t)time(NULL)};
    if (role == JANAS_LLM_ROLE_USER && !c->tr_title)
        janas_api_tr_autotitle(c);
    return m;
}

int janas_api_tr_call(struct tmsg *m, const char *name, size_t nn,
                      const char *args, size_t an)
{
    struct tcall *t = realloc(m->calls, (m->n_calls + 1) * sizeof(*t));
    if (!t)
        return -1;
    m->calls = t;
    char *nd = dup_n(name, nn), *ad = dup_n(args, an);
    if (!nd || !ad) {
        free(nd);
        free(ad);
        return -1;
    }
    m->calls[m->n_calls++] = (struct tcall){.name = nd, .args = ad};
    return 0;
}

int janas_api_tr_system(janas_llm_chat *c, const char *s)
{
    int has = c->n_tr && c->tr[0].role == JANAS_LLM_ROLE_SYSTEM;
    if (!s || !*s) {
        if (has) {
            msg_free(&c->tr[0]);
            memmove(c->tr, c->tr + 1, (c->n_tr - 1) * sizeof(*c->tr));
            c->n_tr--;
        }
        return 0;
    }
    if (has) {
        char *d = dup_n(s, strlen(s));
        if (!d)
            return -1;
        free(c->tr[0].text);
        c->tr[0].text = d;
        return 0;
    }
    if (!janas_api_tr_push(c, JANAS_LLM_ROLE_SYSTEM, s, strlen(s)))
        return -1;
    /* pushed last: it goes first */
    struct tmsg m = c->tr[c->n_tr - 1];
    memmove(c->tr + 1, c->tr, (c->n_tr - 1) * sizeof(*c->tr));
    c->tr[0] = m;
    return 0;
}

void janas_api_tr_turn(janas_llm_chat *c, const char *text, size_t n,
                       int32_t n_res, const char *const *texts,
                       const int32_t *lens)
{
    if (!texts) {
        janas_api_tr_push(c, JANAS_LLM_ROLE_USER, text, n);
    } else {
        for (int32_t i = 0; i < n_res; i++)
            janas_api_tr_push(
                c, JANAS_LLM_ROLE_TOOL, texts[i],
                janas_api_text_len(texts[i], lens ? lens[i] : -1));
    }
    janas_api_tr_saved(c);
}

void janas_api_tr_finish(janas_llm_chat *c)
{
    if (!c->tr_reply)
        return;
    c->tr_reply = 0;
    /* the Specialist's reply is the one said, in Nucleus mode */
    const janas_llm_chat *r =
        c->nuc_route == JANAS_LLM_ROUTE_SPECIALIST && c->nuc_chat ? c->nuc_chat
                                                                  : c;
    struct janas_buf why = {0};
    for (size_t i = 0; i < r->n_toks; i++)
        if (r->toks[i].part == JANAS_LLM_PART_REASONING)
            janas_buf_put(&why, r->tbytes.p + r->toks[i].off, r->toks[i].n);
    if (r->n_rtext == 0 && why.n == 0 && r->n_calls == 0) {
        janas_buf_free(&why);
        return; /* nothing said: not a message */
    }
    /* the blank lines around them, which the model writes after its
       reasoning and before its end, are not what it said */
    const char *t = r->rtext ? r->rtext : "";
    size_t tn = r->n_rtext, wa = 0, wn = why.n;
    while (tn && (unsigned char)*t <= ' ')
        t++, tn--;
    while (tn && (unsigned char)t[tn - 1] <= ' ')
        tn--;
    while (wa < wn && (unsigned char)why.p[wa] <= ' ')
        wa++;
    while (wn > wa && (unsigned char)why.p[wn - 1] <= ' ')
        wn--;
    struct tmsg *m = janas_api_tr_push(c, JANAS_LLM_ROLE_ASSISTANT, t, tn);
    if (m && wn > wa && !why.oom)
        m->reasoning = dup_n(why.p + wa, wn - wa);
    janas_buf_free(&why);
    for (size_t i = 0; m && i < r->n_calls; i++)
        janas_api_tr_call(m, r->calls[i].name, strlen(r->calls[i].name),
                          r->calls[i].args, strlen(r->calls[i].args));
    janas_api_tr_saved(c);
}

void janas_api_tr_settle(janas_llm_chat *c)
{
    if (c->tr_reply && !c->replying)
        janas_api_tr_finish(c);
}

void janas_api_tr_from_msgs(janas_llm_chat *c)
{
    /* the caller's conversation in place of the one written down: the
       title stays if the first question does, the file too */
    char *title = c->tr_title, *path = c->tr_path;
    int titled = c->tr_titled;
    int64_t created = c->tr_created;
    c->tr_title = c->tr_path = NULL;
    janas_api_tr_clear(c);
    c->tr_path = path;
    c->tr_created = created;
    for (size_t i = 0; i < c->n_msgs; i++) {
        const struct cmsg *s = &c->msgs[i];
        struct tmsg *m = janas_api_tr_push(c, s->role, s->text, s->n);
        for (size_t k = 0; m && k < s->n_calls; k++)
            janas_api_tr_call(m, s->calls[k].name, s->calls[k].name_n,
                              s->calls[k].args, s->calls[k].args_n);
    }
    if (titled) {
        free(c->tr_title);
        c->tr_title = title;
        c->tr_titled = 1;
    } else {
        free(title);
    }
}

/*
 * A conversation taken up, loaded whole with the message that follows it:
 * the system message as the chat has it now, then the messages written
 * down, then the new ones (a question, or the tools' answers). When they do
 * not fit, the oldest exchanges are left out of what is loaded, a question
 * with what followed it at a time; the record keeps them.
 */
int32_t janas_api_tr_resume(janas_llm_chat *c, const char *text, size_t n,
                            int32_t n_res, const char *const *texts,
                            const int32_t *lens)
{
    janas_api_tr_finish(c);
    size_t from = c->n_tr && c->tr[0].role == JANAS_LLM_ROLE_SYSTEM ? 1 : 0;
    char *sys = c->system ? strdup(c->system) : NULL;
    if (c->system && !sys)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    int32_t rc;
    for (;;) {
        janas_llm_chat_begin(c);
        rc = sys ? janas_llm_chat_add(c, JANAS_LLM_ROLE_SYSTEM, sys, -1)
                 : JANAS_LLM_OK;
        for (size_t i = from; rc == JANAS_LLM_OK && i < c->n_tr; i++) {
            const struct tmsg *m = &c->tr[i];
            if (m->role == JANAS_LLM_ROLE_SYSTEM)
                continue;
            rc = janas_llm_chat_add(c, m->role, m->text, -1);
            for (size_t k = 0; rc == JANAS_LLM_OK && k < m->n_calls; k++)
                rc = janas_llm_chat_add_call(c, m->calls[k].name, -1,
                                             m->calls[k].args, -1);
        }
        if (rc == JANAS_LLM_OK && !texts)
            rc = janas_llm_chat_add(c, JANAS_LLM_ROLE_USER, text, (int32_t)n);
        for (int32_t i = 0; rc == JANAS_LLM_OK && texts && i < n_res; i++)
            rc = janas_llm_chat_add(c, JANAS_LLM_ROLE_TOOL, texts[i],
                                    lens ? lens[i] : -1);
        if (rc == JANAS_LLM_OK)
            rc = janas_api_run(c);
        if (rc != JANAS_LLM_EFULL && rc != JANAS_LLM_ELIMIT)
            break;
        /* the oldest exchange out: up to the next question */
        size_t k = from + 1;
        while (k < c->n_tr && c->tr[k].role != JANAS_LLM_ROLE_USER)
            k++;
        if (k >= c->n_tr)
            break; /* the last exchange alone does not fit */
        from = k;
    }
    free(sys);
    if (rc != JANAS_LLM_OK)
        return rc;
    c->tr_pending = 0;
    janas_api_tr_turn(c, text, n, n_res, texts, lens); /* the reply: begun */
    return JANAS_LLM_OK;
}

/* The public accessors. */

int32_t janas_llm_chat_messages(janas_llm_chat *c)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    janas_api_tr_settle(c);
    return (int32_t)c->n_tr;
}

int32_t janas_llm_chat_message(janas_llm_chat *c, int32_t i, int32_t what,
                               int32_t *role, char *buf, int32_t cap,
                               int32_t *len)
{
    if (!c || i < 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    janas_api_tr_settle(c);
    if ((size_t)i >= c->n_tr)
        return janas_api_fail(JANAS_LLM_EINVAL, "no message %d", (int)i);
    const struct tmsg *m = &c->tr[i];
    if (role)
        *role = m->role;
    switch (what) {
    case JANAS_LLM_MESSAGE_TEXT:
        return janas_api_copy_out(m->text, strlen(m->text), buf, cap, len);
    case JANAS_LLM_MESSAGE_REASONING: {
        const char *s = m->reasoning ? m->reasoning : "";
        return janas_api_copy_out(s, strlen(s), buf, cap, len);
    }
    case JANAS_LLM_MESSAGE_CALLS: {
        struct janas_buf b = {0};
        janas_buf_puts(&b, "[");
        for (size_t k = 0; k < m->n_calls; k++) {
            janas_buf_puts(&b, k ? ",{\"name\":" : "{\"name\":");
            janas_json_write_str(&b, m->calls[k].name,
                                 strlen(m->calls[k].name));
            janas_buf_puts(&b, ",\"arguments\":");
            janas_json_write_str(&b, m->calls[k].args,
                                 strlen(m->calls[k].args));
            janas_buf_puts(&b, "}");
        }
        janas_buf_puts(&b, "]");
        int32_t rc = b.oom ? janas_api_fail(JANAS_LLM_ENOMEM, "out of memory")
                           : janas_api_copy_out(b.p, b.n, buf, cap, len);
        janas_buf_free(&b);
        return rc;
    }
    }
    return janas_api_fail(JANAS_LLM_EINVAL, "no such part of a message");
}

int32_t janas_llm_chat_title(janas_llm_chat *c, char *buf, int32_t cap,
                             int32_t *len)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    const char *t = c->tr_title ? c->tr_title : "";
    return janas_api_copy_out(t, strlen(t), buf, cap, len);
}

int32_t janas_llm_chat_set_title(janas_llm_chat *c, const char *text,
                                 int32_t len)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    size_t n = text ? janas_api_text_len(text, len) : 0;
    free(c->tr_title);
    c->tr_title = NULL;
    c->tr_titled = n > 0; /* empty: from the first question again */
    if (n && !(c->tr_title = dup_n(text, n)))
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    if (!n)
        janas_api_tr_autotitle(c);
    janas_api_tr_saved(c);
    return JANAS_LLM_OK;
}
