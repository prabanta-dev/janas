/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * engine.c - a model's engine thread: the queue of the jobs JavaScript's
 * thread posts, what the engine's gives back, and the model's life - open,
 * held by JavaScript's objects, closed when asked or when none of them is
 * left, and with the environment when Node leaves it.
 *
 * The engine's thread runs the jobs in the order they were posted, one at a
 * time, so a reply asked for while another is written waits for it: the
 * model holds one conversation. Every job comes back to JavaScript's thread
 * through the thread-safe function, after the pieces of the reply it wrote;
 * the last one (a close, or an open that failed) ends the thread, and
 * JavaScript's thread joins it there.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "node/node_impl.h"

/* the engine's thread's stack: as large as a program's main thread, where
   the library otherwise runs */
#define ENGINE_STACK (8u << 20)

static void back_free(struct janas_node_back *b)
{
    if (b->kind != JANAS_NODE_BACK_JOB) {
        free(b);
        return;
    }
    struct janas_node_job *j = (struct janas_node_job *)b;
    if (j->drop)
        j->drop(j->data);
    else
        free(j->data);
    free(j);
}

struct janas_node_job *janas_node_job(struct janas_node_model *m,
                                      void (*run)(struct janas_node_job *j))
{
    struct janas_node_job *j = calloc(1, sizeof(*j));
    if (!j)
        return NULL;
    j->back.kind = JANAS_NODE_BACK_JOB;
    j->m = m;
    j->run = run;
    return j;
}

static void *engine_main(void *arg)
{
    struct janas_node_model *m = arg;
    for (;;) {
        pthread_mutex_lock(&m->mu);
        while (!m->head)
            pthread_cond_wait(&m->cv, &m->mu);
        struct janas_node_job *j = m->head;
        m->head = j->next;
        if (!m->head)
            m->tail = NULL;
        pthread_mutex_unlock(&m->mu);
        j->run(j);
        int last = j->last; /* j is JavaScript's once given back */
        janas_node_give(m, &j->back);
        if (last)
            return NULL;
    }
}

void janas_node_give(struct janas_node_model *m, struct janas_node_back *b)
{
    if (napi_call_threadsafe_function(m->tsfn, b, napi_tsfn_nonblocking) ==
        napi_ok)
        return;
    /* the environment is going away: freed after the thread is joined */
    pthread_mutex_lock(&m->mu);
    b->next = m->undelivered;
    m->undelivered = b;
    pthread_mutex_unlock(&m->mu);
}

void janas_node_post(napi_env env, struct janas_node_job *j)
{
    struct janas_node_model *m = j->m;
    if (m->pending++ == 0 && !m->st->teardown)
        napi_ref_threadsafe_function(env, m->tsfn);
    pthread_mutex_lock(&m->mu);
    j->next = NULL;
    if (m->tail)
        m->tail->next = j;
    else
        m->head = j;
    m->tail = j;
    pthread_cond_signal(&m->cv);
    pthread_mutex_unlock(&m->mu);
}

void janas_node_ref(struct janas_node_model *m)
{
    m->refs++;
}

void janas_node_unref(napi_env env, struct janas_node_model *m)
{
    (void)env;
    if (--m->refs > 0)
        return;
    struct janas_node_env *st = m->st;
    free(m->close_job);
    free(m->close_wait);
    pthread_cond_destroy(&m->cv);
    pthread_mutex_destroy(&m->mu);
    free(m);
    janas_node_env_unref(st);
}

void janas_node_hold(struct janas_node_model *m)
{
    m->holders++;
    m->refs++;
}

void janas_node_release(napi_env env, struct janas_node_model *m)
{
    if (--m->holders == 0 && !m->closing && m->engine && !m->st->teardown)
        janas_node_close(env, m, NULL);
    janas_node_unref(env, m);
}

static void unlink_model(struct janas_node_model *m)
{
    for (struct janas_node_model **p = &m->st->models; *p; p = &(*p)->next)
        if (*p == m) {
            *p = m->next;
            break;
        }
}

/* After the thread is joined: whatever it left behind, that never came
   back or never ran (the open failed with jobs queued after it, or the
   environment went first). */
static void leftovers(struct janas_node_model *m)
{
    while (m->undelivered) {
        struct janas_node_back *b = m->undelivered;
        m->undelivered = b->next;
        back_free(b);
    }
    while (m->head) {
        struct janas_node_job *j = m->head;
        m->head = j->next;
        back_free(&j->back);
    }
    m->tail = NULL;
}

/* JavaScript's thread, the last job back: the thread is ending */
static void engine_ended(napi_env env, struct janas_node_model *m)
{
    pthread_join(m->thread, NULL);
    m->engine = 0;
    napi_release_threadsafe_function(m->tsfn, napi_tsfn_release);
    leftovers(m);
    unlink_model(m);
    for (int i = 0; i < m->n_close_wait; i++)
        napi_resolve_deferred(env, m->close_wait[i], janas_node_undefined(env));
    m->n_close_wait = 0;
    janas_node_unref(env, m); /* the engine's */
}

static void job_back(napi_env env, struct janas_node_model *m,
                     struct janas_node_job *j)
{
    if (j->deferred) {
        napi_value v = NULL;
        if (j->rc >= 0) {
            v = j->value ? j->value(env, j) : janas_node_undefined(env);
            if (!v) {
                j->rc = JANAS_LLM_ENOMEM;
                snprintf(j->error, sizeof(j->error),
                         "janas: the result could not be made");
            }
        }
        if (v)
            napi_resolve_deferred(env, j->deferred, v);
        else
            napi_reject_deferred(env, j->deferred,
                                 janas_node_error(env, j->rc, j->error));
    }
    if (j->after)
        j->after(env, j);
    if (--m->pending == 0 && !j->last)
        napi_unref_threadsafe_function(env, m->tsfn);
    if (j->last)
        engine_ended(env, m);
    back_free(&j->back);
}

static void call_js(napi_env env, napi_value cb, void *context, void *data)
{
    (void)cb;
    struct janas_node_back *b = data;
    if (!env) { /* the environment went: the memory, and nothing it points to */
        back_free(b);
        return;
    }
    if (b->kind == JANAS_NODE_BACK_JOB)
        job_back(env, context, (struct janas_node_job *)b);
    else
        janas_node_reply_back(env, (struct janas_node_msg *)b);
}

static void close_run(struct janas_node_job *j)
{
    struct janas_node_model *m = j->m;
    if (m->chat) {
        janas_llm_chat_destroy(m->chat);
        m->chat = NULL;
    }
    if (m->llm) {
        janas_llm_close(m->llm);
        m->llm = NULL;
    }
    if (m->speech) {
        janas_llm_speech_close(m->speech);
        m->speech = NULL;
    }
    j->last = 1;
}

struct janas_node_model *janas_node_engine(napi_env env,
                                           struct janas_node_env *st)
{
    struct janas_node_model *m = calloc(1, sizeof(*m));
    if (!m)
        return NULL;
    m->st = st;
    m->refs = 1; /* the engine's */
    m->close_job = janas_node_job(m, close_run);
    if (!m->close_job) {
        free(m);
        return NULL;
    }
    pthread_mutex_init(&m->mu, NULL);
    pthread_cond_init(&m->cv, NULL);
    pthread_attr_t attr;
    int ok = pthread_attr_init(&attr) == 0;
    if (ok)
        pthread_attr_setstacksize(&attr, ENGINE_STACK);
    if (!ok || napi_create_threadsafe_function(
                   env, NULL, NULL,
                   janas_node_string(env, "janas", JANAS_NAPI_AUTO_LENGTH), 0,
                   1, NULL, NULL, m, call_js, &m->tsfn) != napi_ok) {
        if (ok)
            pthread_attr_destroy(&attr);
        goto fail;
    }
    if (pthread_create(&m->thread, &attr, engine_main, m) != 0) {
        pthread_attr_destroy(&attr);
        napi_release_threadsafe_function(m->tsfn, napi_tsfn_abort);
        goto fail;
    }
    pthread_attr_destroy(&attr);
    m->engine = 1;
    st->refs++;
    m->next = st->models;
    st->models = m;
    /* nothing posted yet: an idle model does not keep Node running */
    napi_unref_threadsafe_function(env, m->tsfn);
    return m;
fail:
    pthread_cond_destroy(&m->cv);
    pthread_mutex_destroy(&m->mu);
    free(m->close_job);
    free(m);
    return NULL;
}

void janas_node_close(napi_env env, struct janas_node_model *m, napi_deferred d)
{
    if (d && !m->engine) {
        napi_resolve_deferred(env, d, janas_node_undefined(env));
        return;
    }
    if (d) {
        napi_deferred *w =
            realloc(m->close_wait, (size_t)(m->n_close_wait + 1) * sizeof(*w));
        if (w) {
            m->close_wait = w;
            w[m->n_close_wait++] = d;
        } else
            napi_reject_deferred(
                env, d,
                janas_node_error(env, JANAS_LLM_ENOMEM,
                                 "janas: out of memory; the model "
                                 "closes all the same"));
    }
    if (m->closing)
        return;
    m->closing = 1;
    m->has_chat = 0;
    janas_node_stop_replies(m);
    struct janas_node_job *j = m->close_job;
    m->close_job = NULL;
    janas_node_post(env, j);
}

void janas_node_teardown(struct janas_node_env *st)
{
    st->teardown = 1;
    while (st->models) {
        struct janas_node_model *m = st->models;
        st->models = m->next;
        janas_node_stop_replies(m);
        if (!m->closing) {
            m->closing = 1;
            struct janas_node_job *j = m->close_job;
            m->close_job = NULL;
            janas_node_post(NULL, j);
        }
        /* what the thread gives back from now on is not delivered: it is
           left for leftovers() */
        napi_release_threadsafe_function(m->tsfn, napi_tsfn_abort);
        pthread_join(m->thread, NULL);
        m->engine = 0;
        leftovers(m);
        janas_node_unref(NULL, m); /* the engine's */
    }
}
