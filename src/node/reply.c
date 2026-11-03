/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * reply.c - the Reply class: a reply the model writes, read piece by piece
 * as an async iterator (for await), or whole (text()), stopped (stop(), an
 * AbortSignal, or leaving the loop), and counted (stats()).
 *
 * The engine's thread sends the message (janas_llm_chat_send) or loads the
 * conversation (janas_llm_chat_load), then reads the pieces with
 * janas_llm_chat_next and gives each back as it comes; JavaScript's thread
 * hands them to the next() waiting, or keeps them for the next() to come.
 * The job comes back after its last piece, with the reply's counters.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "node/node_impl.h"

struct reply_job {
    struct janas_node_reply *r;
    int stopped;
    int have_stats;
    struct janas_llm_chat_stats st;
};

/* --- JavaScript's thread: the reply's state ---------------------------- */

static int push(napi_deferred **a, int *n, int *cap, napi_deferred d)
{
    if (*n == *cap) {
        int c = *cap ? *cap * 2 : 4;
        napi_deferred *b = realloc(*a, (size_t)c * sizeof(*b));
        if (!b)
            return -1;
        *a = b;
        *cap = c;
    }
    (*a)[(*n)++] = d;
    return 0;
}

static napi_deferred shift(napi_deferred *a, int *n)
{
    napi_deferred d = a[0];
    memmove(a, a + 1, (size_t)(--*n) * sizeof(*a));
    return d;
}

static napi_value iter_result(napi_env env, napi_value value, int done)
{
    napi_value o;
    if (napi_create_object(env, &o) != napi_ok)
        return NULL;
    janas_node_put(env, o, "value", value ? value : janas_node_undefined(env));
    janas_node_put(env, o, "done", janas_node_bool(env, done));
    return o;
}

static napi_value piece_value(napi_env env, struct janas_node_msg *msg)
{
    napi_value o;
    if (napi_create_object(env, &o) != napi_ok)
        return NULL;
    if (msg->back.kind == JANAS_NODE_BACK_TEXT) {
        janas_node_put(env, o, "type",
                       janas_node_string(env, "text", JANAS_NAPI_AUTO_LENGTH));
        janas_node_put(env, o, "text",
                       janas_node_string(env, msg->text, msg->len));
        janas_node_put(env, o, "reasoning",
                       janas_node_bool(env, msg->reasoning));
    } else {
        janas_node_put(
            env, o, "type",
            janas_node_string(env, "progress", JANAS_NAPI_AUTO_LENGTH));
        janas_node_put(env, o, "read", janas_node_number(env, msg->read));
        janas_node_put(env, o, "total", janas_node_number(env, msg->total));
    }
    return o;
}

static const char *finish_name(const struct janas_node_reply *r)
{
    if (r->stopped)
        return "stopped";
    switch (r->st.finish) {
    case JANAS_LLM_FINISH_STOP:
        return "stop";
    case JANAS_LLM_FINISH_LENGTH:
        return "length";
    case JANAS_LLM_FINISH_CONTEXT:
        return "context";
    case JANAS_LLM_FINISH_TOOLS:
        return "tools";
    case JANAS_LLM_FINISH_ERROR:
        return "error";
    default:
        return r->rc < 0 ? "error" : NULL;
    }
}

static napi_value stats_value(napi_env env, const struct janas_node_reply *r)
{
    const struct janas_llm_chat_stats *s = &r->st;
    napi_value o;
    if (napi_create_object(env, &o) != napi_ok)
        return NULL;
    const char *f = finish_name(r);
    napi_value fin = NULL;
    if (f)
        fin = janas_node_string(env, f, JANAS_NAPI_AUTO_LENGTH);
    else
        napi_get_null(env, &fin);
    janas_node_put(env, o, "finish", fin);
    struct {
        const char *name;
        double v;
    } n[] = {
        {"promptTokens", s->prompt_tokens},
        {"cachedTokens", s->cached_tokens},
        {"inputTokens", s->input_tokens},
        {"outputTokens", s->output_tokens},
        {"reasoningTokens", s->reasoning_tokens},
        {"contextUsed", s->context_used},
        {"contextSize", s->context_size},
        {"drafted", s->drafted},
        {"accepted", s->accepted},
        {"inputSeconds", s->input_seconds},
        {"outputSeconds", s->output_seconds},
        {"totalSeconds", s->total_seconds},
        {"firstTokenSeconds", s->first_token_seconds},
        {"inputThreads", s->input_threads},
        {"outputThreads", s->output_threads},
        {"expertsUsed", (double)s->experts_used},
        {"expertsRead", (double)s->experts_read},
        {"bytesRead", (double)s->bytes_read},
        {"ioWaitSeconds", s->io_wait_seconds},
        {"peakRssBytes", (double)s->peak_rss_bytes},
    };
    for (size_t i = 0; i < sizeof(n) / sizeof(n[0]); i++)
        janas_node_put(env, o, n[i].name, janas_node_number(env, n[i].v));
    janas_node_put(env, o, "reasoningCut",
                   janas_node_bool(env, s->reasoning_cut));
    return o;
}

static napi_value reply_error(napi_env env, const struct janas_node_reply *r)
{
    return janas_node_error(env, r->rc, r->error);
}

static void free_queue(struct janas_node_reply *r)
{
    while (r->q) {
        struct janas_node_msg *msg = r->q;
        r->q = msg->next;
        free(msg);
    }
    r->q_tail = NULL;
}

void janas_node_unwatch(napi_env env, napi_ref *sig, napi_ref *listener)
{
    if (!*sig)
        return;
    napi_value signal, fn, rm, argv[2];
    if (napi_get_reference_value(env, *sig, &signal) == napi_ok &&
        napi_get_reference_value(env, *listener, &fn) == napi_ok &&
        napi_get_named_property(env, signal, "removeEventListener", &rm) ==
            napi_ok) {
        argv[0] = janas_node_string(env, "abort", JANAS_NAPI_AUTO_LENGTH);
        argv[1] = fn;
        if (napi_call_function(env, signal, rm, 2, argv, NULL) != napi_ok) {
            napi_value ignored;
            napi_get_and_clear_last_exception(env, &ignored);
        }
    }
    napi_delete_reference(env, *sig);
    napi_delete_reference(env, *listener);
    *sig = *listener = NULL;
}

static void unlink_reply(struct janas_node_reply *r)
{
    for (struct janas_node_reply **p = &r->m->replies; *p; p = &(*p)->next)
        if (*p == r) {
            *p = r->next;
            break;
        }
}

static void reply_unref(napi_env env, struct janas_node_reply *r)
{
    if (--r->refs > 0)
        return;
    struct janas_node_model *m = r->m;
    int js = env && !m->st->teardown;
    free_queue(r);
    if (js) {
        janas_node_unwatch(env, &r->signal, &r->listener);
        if (r->chat)
            napi_delete_reference(env, r->chat);
    }
    free(r->wait);
    free(r->stats_wait);
    free(r->text_wait);
    free(r->answer);
    free(r->text);
    if (r->texts)
        for (int32_t i = 0; i < r->n; i++)
            free(r->texts[i]);
    free(r->texts);
    free(r->roles);
    free(r->lens);
    free(r);
    janas_node_unref(env, m);
}

static int append_answer(struct janas_node_reply *r, const char *s, size_t len)
{
    if (r->answer_len + len + 1 > r->answer_cap) {
        size_t c = r->answer_cap ? r->answer_cap : 256;
        while (c < r->answer_len + len + 1)
            c *= 2;
        char *b = realloc(r->answer, c);
        if (!b)
            return -1;
        r->answer = b;
        r->answer_cap = c;
    }
    memcpy(r->answer + r->answer_len, s, len);
    r->answer_len += len;
    r->answer[r->answer_len] = 0;
    return 0;
}

/* a piece for text(): the answer's text, not the reasoning */
static void collect(struct janas_node_reply *r, struct janas_node_msg *msg)
{
    if (msg->back.kind == JANAS_NODE_BACK_TEXT && !msg->reasoning &&
        append_answer(r, msg->text, msg->len) < 0 && r->rc >= 0) {
        r->rc = JANAS_LLM_ENOMEM;
        snprintf(r->error, sizeof(r->error), "janas: out of memory");
    }
}

void janas_node_reply_back(napi_env env, struct janas_node_msg *msg)
{
    struct janas_node_reply *r = msg->reply;
    if (r->gone || r->returned) {
        free(msg);
        return;
    }
    if (r->collect) {
        collect(r, msg);
        free(msg);
        return;
    }
    if (r->n_wait) {
        napi_deferred d = shift(r->wait, &r->n_wait);
        napi_value v = iter_result(env, piece_value(env, msg), 0);
        free(msg);
        if (v)
            napi_resolve_deferred(env, d, v);
        else
            napi_reject_deferred(env, d,
                                 janas_node_error(env, JANAS_LLM_ENOMEM,
                                                  "janas: out of memory"));
        return;
    }
    msg->next = NULL;
    if (r->q_tail)
        r->q_tail->next = msg;
    else
        r->q = msg;
    r->q_tail = msg;
}

/* the waiting promises, once the reply is over (or left) */
static void settle(napi_env env, struct janas_node_reply *r)
{
    while (r->n_wait) {
        napi_deferred d = shift(r->wait, &r->n_wait);
        if (r->rc < 0 && !r->error_given && !r->returned) {
            r->error_given = 1;
            napi_reject_deferred(env, d, reply_error(env, r));
        } else
            napi_resolve_deferred(env, d, iter_result(env, NULL, 1));
    }
    if (!r->ended)
        return;
    while (r->n_text_wait) {
        napi_deferred d = shift(r->text_wait, &r->n_text_wait);
        if (r->rc < 0)
            napi_reject_deferred(env, d, reply_error(env, r));
        else
            napi_resolve_deferred(
                env, d, janas_node_string(env, r->answer, r->answer_len));
    }
    while (r->n_stats_wait)
        napi_resolve_deferred(env, shift(r->stats_wait, &r->n_stats_wait),
                              stats_value(env, r));
}

static void reply_after(napi_env env, struct janas_node_job *j)
{
    struct reply_job *rj = j->data;
    struct janas_node_reply *r = rj->r;
    r->ended = 1;
    if (r->rc >= 0) /* text() may have run out of memory meanwhile */
        r->rc = j->rc;
    if (j->rc < 0)
        memcpy(r->error, j->error, sizeof(r->error));
    r->stopped = rj->stopped;
    if (rj->have_stats)
        r->st = rj->st;
    unlink_reply(r);
    janas_node_unwatch(env, &r->signal, &r->listener);
    if (r->chat) {
        napi_delete_reference(env, r->chat);
        r->chat = NULL;
    }
    settle(env, r);
    reply_unref(env, r); /* the job's */
}

void janas_node_stop_replies(struct janas_node_model *m)
{
    for (struct janas_node_reply *r = m->replies; r; r = r->next)
        atomic_store(&r->stop, 1);
}

/* --- the engine's thread ------------------------------------------------ */

static int give_piece(struct janas_node_model *m, struct janas_node_reply *r,
                      int kind, const char *text, size_t len, int reasoning,
                      uint32_t read, uint32_t total)
{
    struct janas_node_msg *msg = malloc(sizeof(*msg) + len);
    if (!msg)
        return -1;
    memset(msg, 0, sizeof(*msg));
    msg->back.kind = kind;
    msg->reply = r;
    msg->reasoning = reasoning;
    msg->read = read;
    msg->total = total;
    msg->len = len;
    if (len)
        memcpy(msg->text, text, len);
    janas_node_give(m, &msg->back);
    return 0;
}

static void reply_run(struct janas_node_job *j)
{
    struct reply_job *rj = j->data;
    struct janas_node_reply *r = rj->r;
    struct janas_node_model *m = j->m;
    janas_llm_chat *c = m->chat;
    if (atomic_load(&r->stop)) { /* stopped before it began */
        rj->stopped = 1;
        return;
    }
    if (!c) {
        j->rc = JANAS_NODE_ECLOSED;
        snprintf(j->error, sizeof(j->error), "janas: the chat is closed");
        return;
    }
    int32_t rc =
        r->load ? janas_llm_chat_load(c, r->n, r->roles,
                                      (const char *const *)r->texts, r->lens)
                : janas_llm_chat_send(c, r->text, (int32_t)r->len);
    char buf[4096];
    while (rc == JANAS_LLM_OK) {
        if (atomic_load_explicit(&r->stop, memory_order_relaxed)) {
            rj->stopped = 1;
            break;
        }
        int32_t len = 0;
        rc = janas_llm_chat_next(c, buf, sizeof(buf), &len);
        if (rc != JANAS_LLM_OK)
            break;
        int lost;
        if (len > 0) {
            lost = give_piece(m, r, JANAS_NODE_BACK_TEXT, buf, (size_t)len,
                              janas_llm_chat_thinking(c), 0, 0);
        } else { /* a block of the prompt read, or nothing to say yet */
            struct janas_llm_chat_stats s = {.size = sizeof(s)};
            lost = janas_llm_chat_stats(c, &s) == JANAS_LLM_OK &&
                           s.stage == JANAS_LLM_STAGE_INPUT
                       ? give_piece(m, r, JANAS_NODE_BACK_PROGRESS, NULL, 0, 0,
                                    s.input_done, s.input_tokens)
                       : 0;
        }
        if (lost) { /* a piece missing would be a wrong text */
            rc = JANAS_LLM_ENOMEM;
            snprintf(j->error, sizeof(j->error),
                     "janas: out of memory, the reply cut short");
            break;
        }
    }
    if (rc == JANAS_LLM_DONE)
        rc = JANAS_LLM_OK;
    if (rc < 0 && !j->error[0])
        snprintf(j->error, sizeof(j->error), "%s", janas_llm_last_error());
    j->rc = rc;
    rj->st.size = sizeof(rj->st);
    rj->have_stats = janas_llm_chat_stats(c, &rj->st) == JANAS_LLM_OK;
}

/* --- the class ---------------------------------------------------------- */

static struct janas_node_reply *self(napi_env env, napi_callback_info info,
                                     size_t *argc, napi_value *argv,
                                     const char *what)
{
    return janas_node_self(env, info, &janas_node_reply_tag, argc, argv, what);
}

static napi_value reply_next(napi_env env, napi_callback_info info)
{
    struct janas_node_reply *r = self(env, info, NULL, NULL, "Reply.next");
    if (!r)
        return NULL;
    if (r->q) {
        struct janas_node_msg *msg = r->q;
        r->q = msg->next;
        if (!r->q)
            r->q_tail = NULL;
        napi_value v = iter_result(env, piece_value(env, msg), 0);
        free(msg);
        return janas_node_resolved(env, v);
    }
    if (r->ended || r->returned || r->collect) {
        if (r->rc < 0 && !r->error_given && !r->returned && !r->collect) {
            r->error_given = 1;
            return janas_node_rejected(env, reply_error(env, r));
        }
        return janas_node_resolved(env, iter_result(env, NULL, 1));
    }
    napi_deferred d;
    napi_value p = janas_node_promise(env, &d);
    if (p && push(&r->wait, &r->n_wait, &r->cap_wait, d) < 0) {
        napi_reject_deferred(
            env, d,
            janas_node_error(env, JANAS_LLM_ENOMEM, "janas: out of memory"));
    }
    return p;
}

/* leaving a for await loop early (break, return, throw) */
static napi_value reply_return(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value argv[1] = {NULL};
    struct janas_node_reply *r = self(env, info, &argc, argv, "Reply.return");
    if (!r)
        return NULL;
    atomic_store(&r->stop, 1);
    r->returned = 1;
    free_queue(r);
    settle(env, r);
    return janas_node_resolved(env, iter_result(env, argc ? argv[0] : NULL, 1));
}

static napi_value reply_iterator(napi_env env, napi_callback_info info)
{
    napi_value self_obj = NULL;
    napi_get_cb_info(env, info, NULL, NULL, &self_obj, NULL);
    return self_obj;
}

static napi_value reply_stop(napi_env env, napi_callback_info info)
{
    struct janas_node_reply *r = self(env, info, NULL, NULL, "Reply.stop");
    if (r)
        atomic_store(&r->stop, 1);
    return NULL;
}

static napi_value reply_text(napi_env env, napi_callback_info info)
{
    struct janas_node_reply *r = self(env, info, NULL, NULL, "Reply.text");
    if (!r)
        return NULL;
    if (!r->collect) { /* the pieces from now on, and those kept */
        r->collect = 1;
        while (r->q) {
            struct janas_node_msg *msg = r->q;
            r->q = msg->next;
            collect(r, msg);
            free(msg);
        }
        r->q_tail = NULL;
        settle(env, r); /* next() calls waiting: done, the text took them */
    }
    if (r->ended) {
        if (r->rc < 0)
            return janas_node_rejected(env, reply_error(env, r));
        return janas_node_resolved(
            env, janas_node_string(env, r->answer, r->answer_len));
    }
    napi_deferred d;
    napi_value p = janas_node_promise(env, &d);
    if (p && push(&r->text_wait, &r->n_text_wait, &r->cap_text_wait, d) < 0)
        napi_reject_deferred(
            env, d,
            janas_node_error(env, JANAS_LLM_ENOMEM, "janas: out of memory"));
    return p;
}

static napi_value reply_stats(napi_env env, napi_callback_info info)
{
    struct janas_node_reply *r = self(env, info, NULL, NULL, "Reply.stats");
    if (!r)
        return NULL;
    if (r->ended)
        return janas_node_resolved(env, stats_value(env, r));
    napi_deferred d;
    napi_value p = janas_node_promise(env, &d);
    if (p && push(&r->stats_wait, &r->n_stats_wait, &r->cap_stats_wait, d) < 0)
        napi_reject_deferred(
            env, d,
            janas_node_error(env, JANAS_LLM_ENOMEM, "janas: out of memory"));
    return p;
}

static void reply_finalize(napi_env env, void *data, void *hint)
{
    (void)hint;
    struct janas_node_reply *r = data;
    struct janas_node_model *m = r->m;
    atomic_store(&r->stop, 1);
    r->gone = 1; /* no promise of its can be waited for: none was pending */
    free_queue(r);
    janas_node_ref(m); /* m outlives r's last reference */
    reply_unref(env, r);
    janas_node_release(env, m); /* the object's hold */
    janas_node_unref(env, m);
}

static napi_value on_abort(napi_env env, napi_callback_info info)
{
    void *data = NULL;
    napi_get_cb_info(env, info, NULL, NULL, NULL, &data);
    if (data)
        atomic_store((atomic_int *)data, 1);
    return NULL;
}

int janas_node_watch(napi_env env, napi_value signal, atomic_int *stop,
                     napi_ref *sig, napi_ref *listener)
{
    napi_value aborted, add, fn, argv[2];
    napi_valuetype t;
    bool b = false;
    if (napi_typeof(env, signal, &t) != napi_ok || t != napi_object ||
        napi_get_named_property(env, signal, "aborted", &aborted) != napi_ok ||
        napi_get_named_property(env, signal, "addEventListener", &add) !=
            napi_ok ||
        napi_typeof(env, add, &t) != napi_ok || t != napi_function) {
        janas_node_type_error(env, "janas: the signal must be an AbortSignal");
        return -1;
    }
    napi_get_value_bool(env, aborted, &b);
    if (b) {
        atomic_store(stop, 1);
        return 0;
    }
    if (napi_create_function(env, "abort", JANAS_NAPI_AUTO_LENGTH, on_abort,
                             stop, &fn) != napi_ok)
        return -1;
    argv[0] = janas_node_string(env, "abort", JANAS_NAPI_AUTO_LENGTH);
    argv[1] = fn;
    if (napi_call_function(env, signal, add, 2, argv, NULL) != napi_ok)
        return -1;
    if (napi_create_reference(env, signal, 1, sig) != napi_ok ||
        napi_create_reference(env, fn, 1, listener) != napi_ok) {
        janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
        return -1;
    }
    return 0;
}

static int message_role(const char *s)
{
    static const char *const roles[] = {"system", "user", "assistant", "tool"};
    static const int32_t ids[] = {JANAS_LLM_ROLE_SYSTEM, JANAS_LLM_ROLE_USER,
                                  JANAS_LLM_ROLE_ASSISTANT,
                                  JANAS_LLM_ROLE_TOOL};
    for (int i = 0; i < 4; i++)
        if (!strcmp(s, roles[i]))
            return ids[i];
    return -1;
}

static int messages(napi_env env, napi_value v, struct janas_node_reply *r)
{
    bool array = false;
    uint32_t n = 0;
    if (!v || napi_is_array(env, v, &array) != napi_ok || !array ||
        napi_get_array_length(env, v, &n) != napi_ok || n == 0 ||
        n > INT32_MAX) {
        janas_node_type_error(env, "janas: chat.load(messages): an array of "
                                   "{role, content}, one at least");
        return -1;
    }
    r->roles = calloc(n, sizeof(*r->roles));
    r->texts = calloc(n, sizeof(*r->texts));
    r->lens = calloc(n, sizeof(*r->lens));
    if (!r->roles || !r->texts || !r->lens) {
        janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
        return -1;
    }
    r->n = (int32_t)n;
    for (uint32_t i = 0; i < n; i++) {
        napi_value el, role, content;
        napi_valuetype t;
        size_t len = 0;
        char *s = NULL;
        if (napi_get_element(env, v, i, &el) != napi_ok ||
            napi_typeof(env, el, &t) != napi_ok || t != napi_object ||
            napi_get_named_property(env, el, "role", &role) != napi_ok ||
            napi_get_named_property(env, el, "content", &content) != napi_ok ||
            !(s = janas_node_text(env, role, &len)) ||
            (r->roles[i] = message_role(s)) < 0) {
            free(s);
            janas_node_type_error(env,
                                  "janas: chat.load: message %u: a role "
                                  "(system, user, assistant, tool) and a "
                                  "content",
                                  i);
            return -1;
        }
        free(s);
        r->texts[i] = janas_node_text(env, content, &len);
        if (!r->texts[i] || len > INT32_MAX) {
            janas_node_type_error(env,
                                  "janas: chat.load: message %u: the content "
                                  "must be a string",
                                  i);
            return -1;
        }
        r->lens[i] = (int32_t)len;
    }
    return 0;
}

static void reply_job_drop(void *data)
{
    free(data);
}

napi_value janas_node_reply(napi_env env, struct janas_node_chat *c,
                            napi_value chat_obj, int load, napi_value what,
                            napi_value options)
{
    static const char *const known[] = {"signal", NULL};
    struct janas_node_model *m = c->m;
    const char *name = load ? "chat.load" : "chat.send";
    struct janas_node_reply *r = calloc(1, sizeof(*r));
    struct reply_job *rj = calloc(1, sizeof(*rj));
    struct janas_node_job *j = janas_node_job(m, reply_run);
    if (!r || !rj || !j) {
        free(r);
        free(rj);
        free(j);
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    }
    r->m = m;
    janas_node_ref(m); /* the reply's, released with it */
    r->refs = 1;       /* until the object is made: then the object's */
    r->load = load;
    int bad = 0;
    if (load)
        bad = messages(env, what, r) < 0;
    else if (!(r->text = janas_node_text(env, what, &r->len)) ||
             r->len > INT32_MAX) {
        janas_node_type_error(env, "janas: chat.send(text): the text must be "
                                   "a string");
        bad = 1;
    }
    napi_value signal = NULL;
    int has_signal = 0;
    if (!bad && janas_node_options(env, options, name, known) < 0)
        bad = 1;
    if (!bad && (has_signal = janas_node_opt(env, options, name, "signal",
                                             napi_object, &signal)) < 0)
        bad = 1;
    napi_value obj = NULL;
    if (!bad) {
        obj = janas_node_new(env, m->st->reply_class, &janas_node_reply_tag, r,
                             reply_finalize);
        if (!obj) {
            janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
            bad = 1;
        }
    }
    if (bad) {
        free(rj);
        free(j);
        reply_unref(env, r);
        return NULL;
    }
    janas_node_hold(m); /* the object's, released when it is collected */
    if (has_signal > 0 &&
        janas_node_watch(env, signal, &r->stop, &r->signal, &r->listener) < 0) {
        /* the object exists: it ends at once, with nothing written */
        atomic_store(&r->stop, 1);
        r->ended = 1;
        free(rj);
        free(j);
        return NULL;
    }
    if (chat_obj)
        napi_create_reference(env, chat_obj, 1, &r->chat);
    rj->r = r;
    j->data = rj;
    j->drop = reply_job_drop;
    j->after = reply_after;
    r->refs++; /* the job's, until it is back */
    r->next = m->replies;
    m->replies = r;
    janas_node_post(env, j);
    return obj;
}

napi_value janas_node_reply_class(napi_env env)
{
    napi_value global, symbol, iterator;
    if (napi_get_global(env, &global) != napi_ok ||
        napi_get_named_property(env, global, "Symbol", &symbol) != napi_ok ||
        napi_get_named_property(env, symbol, "asyncIterator", &iterator) !=
            napi_ok)
        return NULL;
    const napi_property_attributes method = napi_writable | napi_configurable;
    napi_property_descriptor d[] = {
        {.utf8name = "next", .method = reply_next, .attributes = method},
        {.utf8name = "return", .method = reply_return, .attributes = method},
        {.name = iterator, .method = reply_iterator, .attributes = method},
        {.utf8name = "stop", .method = reply_stop, .attributes = method},
        {.utf8name = "text", .method = reply_text, .attributes = method},
        {.utf8name = "stats", .method = reply_stats, .attributes = method},
    };
    napi_value cls = NULL;
    if (napi_define_class(env, "Reply", JANAS_NAPI_AUTO_LENGTH,
                          janas_node_constructor,
                          (void *)"a Reply comes from chat.send() or "
                                  "chat.load()",
                          sizeof(d) / sizeof(d[0]), d, &cls) != napi_ok)
        return NULL;
    return cls;
}
