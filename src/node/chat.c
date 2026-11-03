/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * chat.c - the Chat class: a model's conversation (one at a time per model,
 * as the library has it), its parameters, its system message, a reset, the
 * pictures that go with a message, and the replies it asks for (reply.c).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "node/node_impl.h"

struct chat_job {
    struct janas_node_chat *c; /* the chat's object owns it once made */
    struct janas_llm_chat_params p;
    char *system;
    size_t system_len;
    int32_t keep;
    int has_keep;
    int made; /* the engine made the chat (set on its thread) */
};

static void chat_job_drop(void *data)
{
    struct chat_job *cj = data;
    if (!cj)
        return;
    free(cj->c);
    free(cj->system);
    free(cj);
}

static const char *const chat_known[] = {"system",
                                         "temperature",
                                         "topK",
                                         "topP",
                                         "minP",
                                         "seed",
                                         "speculate",
                                         "maxReply",
                                         "maxAnswer",
                                         "maxReasoning",
                                         "maxInput",
                                         "thinking",
                                         "recap",
                                         "presencePenalty",
                                         "frequencyPenalty",
                                         "keep",
                                         NULL};

/* the options of model.chat() and chat.set() over cj->p */
static int chat_options(napi_env env, napi_value o, const char *w,
                        struct chat_job *cj)
{
    struct janas_llm_chat_params *p = &cj->p;
    double d;
    int64_t i;
    int b, r;
    if (janas_node_options(env, o, w, chat_known) < 0)
        return -1;
#define NUMBER(name, field, lo, hi)                                            \
    if ((r = janas_node_opt_number(env, o, w, name, lo, hi, &d)) < 0)          \
        return -1;                                                             \
    if (r)                                                                     \
        p->field = (float)d;
#define WHOLE(name, field, type, lo, hi)                                       \
    if ((r = janas_node_opt_int(env, o, w, name, lo, hi, &i)) < 0)             \
        return -1;                                                             \
    if (r)                                                                     \
        p->field = (type)i;
    NUMBER("temperature", temperature, 0, 100)
    NUMBER("topP", top_p, 0, 1)
    NUMBER("minP", min_p, 0, 1)
    NUMBER("presencePenalty", presence_penalty, -2, 2)
    NUMBER("frequencyPenalty", frequency_penalty, -2, 2)
    WHOLE("topK", top_k, int32_t, 0, INT32_MAX)
    WHOLE("seed", seed, uint64_t, 0, 9007199254740991LL)
    WHOLE("maxReply", max_reply, int32_t, 0, INT32_MAX)
    WHOLE("maxAnswer", max_answer, int32_t, 0, INT32_MAX)
    WHOLE("maxReasoning", max_reasoning, int32_t, 0, INT32_MAX)
    WHOLE("maxInput", max_input, uint32_t, 0, UINT32_MAX)
#undef NUMBER
#undef WHOLE
    if ((r = janas_node_opt_bool(env, o, w, "speculate", &b)) < 0)
        return -1;
    if (r)
        p->speculate = b;
    if ((r = janas_node_opt_bool(env, o, w, "recap", &b)) < 0)
        return -1;
    if (r)
        p->recap = b;
    /* thinking: true, false, or null for the model's own way */
    napi_value v;
    napi_valuetype t;
    if (o && napi_typeof(env, o, &t) == napi_ok && t == napi_object &&
        napi_get_named_property(env, o, "thinking", &v) == napi_ok &&
        napi_typeof(env, v, &t) == napi_ok && t == napi_null)
        p->thinking = -1;
    else if ((r = janas_node_opt_bool(env, o, w, "thinking", &b)) < 0)
        return -1;
    else if (r)
        p->thinking = b;
    if ((r = janas_node_opt(env, o, w, "system", napi_string, &v)) < 0)
        return -1;
    if (r) {
        cj->system = janas_node_text(env, v, &cj->system_len);
        if (!cj->system) {
            janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
            return -1;
        }
    }
    if ((r = janas_node_opt_int(env, o, w, "keep", 0, 64, &i)) < 0)
        return -1;
    if (r) {
        cj->keep = (int32_t)i;
        cj->has_keep = 1;
    }
    return 0;
}

static void failed(struct janas_node_job *j)
{
    snprintf(j->error, sizeof(j->error), "%s", janas_llm_last_error());
}

/* the system message and the conversations kept, after the parameters */
static void chat_extras(struct janas_node_job *j, struct chat_job *cj)
{
    janas_llm_chat *c = j->m->chat;
    if (j->rc == JANAS_LLM_OK && cj->system)
        j->rc = janas_llm_chat_system(c, cj->system, (int32_t)cj->system_len);
    if (j->rc == JANAS_LLM_OK && cj->has_keep)
        j->rc = janas_llm_chat_keep(c, cj->keep, 0);
    if (j->rc != JANAS_LLM_OK)
        failed(j);
}

static void chat_run(struct janas_node_job *j)
{
    struct janas_node_model *m = j->m;
    struct chat_job *cj = j->data;
    j->rc = janas_llm_chat_create(m->llm, &cj->p, &m->chat);
    if (j->rc != JANAS_LLM_OK) {
        failed(j);
        m->chat = NULL;
        return;
    }
    chat_extras(j, cj);
    if (j->rc != JANAS_LLM_OK) {
        janas_llm_chat_destroy(m->chat);
        m->chat = NULL;
    } else
        cj->made = 1;
}

static void destroy_run(struct janas_node_job *j)
{
    if (j->m->chat) {
        janas_llm_chat_destroy(j->m->chat);
        j->m->chat = NULL;
    }
}

/* the chat ends: its replies stop, and the engine destroys it after them */
static void chat_end(napi_env env, struct janas_node_chat *c, napi_deferred d)
{
    struct janas_node_model *m = c->m;
    c->closed = 1;
    if (m->closing || !m->engine || m->st->teardown) {
        if (d)
            napi_resolve_deferred(env, d, janas_node_undefined(env));
        return;
    }
    m->has_chat = 0;
    janas_node_stop_replies(m);
    struct janas_node_job *j = janas_node_job(m, destroy_run);
    if (!j) { /* the model's close destroys it then */
        if (d)
            napi_reject_deferred(env, d,
                                 janas_node_error(env, JANAS_LLM_ENOMEM,
                                                  "janas: out of memory"));
        return;
    }
    j->deferred = d;
    janas_node_post(env, j);
}

static void chat_finalize(napi_env env, void *data, void *hint)
{
    (void)hint;
    struct janas_node_chat *c = data;
    struct janas_node_model *m = c->m;
    if (!c->closed)
        chat_end(env, c, NULL);
    free(c);
    janas_node_release(env, m);
}

static napi_value chat_value(napi_env env, struct janas_node_job *j)
{
    struct chat_job *cj = j->data;
    napi_value obj = janas_node_new(env, j->m->st->chat_class,
                                    &janas_node_chat_tag, cj->c, chat_finalize);
    if (!obj)
        return NULL;
    cj->c = NULL; /* the object's now */
    janas_node_hold(j->m);
    return obj;
}

static void chat_after(napi_env env, struct janas_node_job *j)
{
    struct chat_job *cj = j->data;
    struct janas_node_model *m = j->m;
    if (!cj->c)
        return; /* made */
    /* not made, here or in the engine: the model is free for another, and
       a chat the engine made for nobody is destroyed */
    if (m->has_chat && !m->closing)
        m->has_chat = 0;
    if (cj->made && !m->closing) {
        struct janas_node_job *d = janas_node_job(m, destroy_run);
        if (d)
            janas_node_post(env, d);
    }
}

static napi_value closed(napi_env env, int promise, const char *what)
{
    char msg[160];
    snprintf(msg, sizeof(msg), "janas: %s: the chat is closed", what);
    if (!promise)
        return janas_node_throw(env, JANAS_NODE_ECLOSED, msg);
    return janas_node_rejected(env,
                               janas_node_error(env, JANAS_NODE_ECLOSED, msg));
}

napi_value janas_node_chat(napi_env env, struct janas_node_model *m,
                           napi_value options)
{
    if (m->closing)
        return janas_node_rejected(
            env, janas_node_error(env, JANAS_NODE_ECLOSED,
                                  "janas: model.chat: the model is closed"));
    if (m->has_chat)
        return janas_node_rejected(
            env, janas_node_error(env, JANAS_LLM_EBUSY,
                                  "janas: model.chat: the model has a chat "
                                  "already, and holds one at a time: close "
                                  "it first"));
    struct chat_job *cj = calloc(1, sizeof(*cj));
    if (!cj)
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    janas_llm_chat_params_default(&cj->p);
    if (chat_options(env, options, "model.chat", cj) < 0) {
        chat_job_drop(cj);
        return NULL;
    }
    cj->c = calloc(1, sizeof(*cj->c));
    struct janas_node_job *j = cj->c ? janas_node_job(m, chat_run) : NULL;
    napi_value p = j ? janas_node_promise(env, &j->deferred) : NULL;
    if (!p) {
        chat_job_drop(cj);
        free(j);
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    }
    cj->c->m = m;
    cj->c->p = cj->p;
    j->data = cj;
    j->drop = chat_job_drop;
    j->value = chat_value;
    j->after = chat_after;
    m->has_chat = 1;
    janas_node_post(env, j);
    return p;
}

static void set_run(struct janas_node_job *j)
{
    struct chat_job *cj = j->data;
    janas_llm_chat *c = j->m->chat;
    if (!c) {
        j->rc = JANAS_NODE_ECLOSED;
        snprintf(j->error, sizeof(j->error), "janas: the chat is closed");
        return;
    }
    j->rc = janas_llm_chat_set_params(c, &cj->p);
    if (j->rc != JANAS_LLM_OK)
        failed(j);
    chat_extras(j, cj);
}

static napi_value chat_set(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value argv[1] = {NULL};
    struct janas_node_chat *c = janas_node_self(env, info, &janas_node_chat_tag,
                                                &argc, argv, "Chat.set");
    if (!c)
        return NULL;
    if (c->closed || c->m->closing)
        return closed(env, 1, "chat.set");
    struct chat_job *cj = calloc(1, sizeof(*cj));
    if (!cj)
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    cj->p = c->p;
    if (chat_options(env, argc ? argv[0] : NULL, "chat.set", cj) < 0) {
        chat_job_drop(cj);
        return NULL;
    }
    struct janas_node_job *j = janas_node_job(c->m, set_run);
    napi_value p = j ? janas_node_promise(env, &j->deferred) : NULL;
    if (!p) {
        chat_job_drop(cj);
        free(j);
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    }
    /* the next set() starts from these, whatever the engine makes of them:
       the whole parameters go every time */
    c->p = cj->p;
    j->data = cj;
    j->drop = chat_job_drop;
    janas_node_post(env, j);
    return p;
}

static void reset_run(struct janas_node_job *j)
{
    janas_llm_chat *c = j->m->chat;
    if (!c) {
        j->rc = JANAS_NODE_ECLOSED;
        snprintf(j->error, sizeof(j->error), "janas: the chat is closed");
        return;
    }
    j->rc = janas_llm_chat_reset(c);
    if (j->rc != JANAS_LLM_OK)
        failed(j);
}

static napi_value chat_reset(napi_env env, napi_callback_info info)
{
    struct janas_node_chat *c = janas_node_self(env, info, &janas_node_chat_tag,
                                                NULL, NULL, "Chat.reset");
    if (!c)
        return NULL;
    if (c->closed || c->m->closing)
        return closed(env, 1, "chat.reset");
    struct janas_node_job *j = janas_node_job(c->m, reset_run);
    napi_value p = j ? janas_node_promise(env, &j->deferred) : NULL;
    if (!p) {
        free(j);
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    }
    janas_node_post(env, j);
    return p;
}

struct image_job {
    void *data;
    size_t n;
};

static void image_drop(void *data)
{
    struct image_job *ij = data;
    if (ij)
        free(ij->data);
    free(ij);
}

static void image_run(struct janas_node_job *j)
{
    struct image_job *ij = j->data;
    janas_llm_chat *c = j->m->chat;
    if (!c) {
        j->rc = JANAS_NODE_ECLOSED;
        snprintf(j->error, sizeof(j->error), "janas: the chat is closed");
        return;
    }
    j->rc = janas_llm_chat_image(c, ij->data, (int64_t)ij->n);
    if (j->rc != JANAS_LLM_OK)
        failed(j);
}

/* a picture for the next message: the bytes copied now, read by the
   engine in order with the messages */
static napi_value chat_image(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value argv[1] = {NULL};
    struct janas_node_chat *c = janas_node_self(env, info, &janas_node_chat_tag,
                                                &argc, argv, "Chat.image");
    if (!c)
        return NULL;
    if (c->closed || c->m->closing)
        return closed(env, 1, "chat.image");
    struct image_job *ij = calloc(1, sizeof(*ij));
    if (!ij)
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    ij->data = argc ? janas_node_bytes(env, argv[0], &ij->n) : NULL;
    if (!ij->data) {
        int oom = ij->n == SIZE_MAX;
        image_drop(ij);
        return oom ? janas_node_throw(env, JANAS_LLM_ENOMEM,
                                      "janas: out of memory")
                   : janas_node_type_error(
                         env, "janas: chat.image(data): the bytes of a PNG or "
                              "a JPEG, as a Buffer, a Uint8Array or an "
                              "ArrayBuffer");
    }
    struct janas_node_job *j = janas_node_job(c->m, image_run);
    napi_value p = j ? janas_node_promise(env, &j->deferred) : NULL;
    if (!p) {
        image_drop(ij);
        free(j);
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    }
    j->data = ij;
    j->drop = image_drop;
    janas_node_post(env, j);
    return p;
}

static napi_value chat_close(napi_env env, napi_callback_info info)
{
    struct janas_node_chat *c = janas_node_self(env, info, &janas_node_chat_tag,
                                                NULL, NULL, "Chat.close");
    if (!c)
        return NULL;
    if (c->closed)
        return janas_node_resolved(env, NULL);
    napi_deferred d;
    napi_value p = janas_node_promise(env, &d);
    if (p)
        chat_end(env, c, d);
    return p;
}

static napi_value chat_reply(napi_env env, napi_callback_info info, int load)
{
    size_t argc = 2;
    napi_value argv[2] = {NULL, NULL}, self = NULL;
    const char *what = load ? "Chat.load" : "Chat.send";
    struct janas_node_chat *c =
        janas_node_self(env, info, &janas_node_chat_tag, &argc, argv, what);
    if (!c)
        return NULL;
    if (c->closed || c->m->closing)
        return closed(env, 0, load ? "chat.load" : "chat.send");
    napi_get_cb_info(env, info, NULL, NULL, &self, NULL);
    return janas_node_reply(env, c, self, load, argc ? argv[0] : NULL,
                            argc > 1 ? argv[1] : NULL);
}

static napi_value chat_send(napi_env env, napi_callback_info info)
{
    return chat_reply(env, info, 0);
}

static napi_value chat_load(napi_env env, napi_callback_info info)
{
    return chat_reply(env, info, 1);
}

napi_value janas_node_chat_class(napi_env env)
{
    const napi_property_attributes method = napi_writable | napi_configurable;
    napi_property_descriptor d[] = {
        {.utf8name = "send", .method = chat_send, .attributes = method},
        {.utf8name = "load", .method = chat_load, .attributes = method},
        {.utf8name = "image", .method = chat_image, .attributes = method},
        {.utf8name = "set", .method = chat_set, .attributes = method},
        {.utf8name = "reset", .method = chat_reset, .attributes = method},
        {.utf8name = "close", .method = chat_close, .attributes = method},
    };
    napi_value cls = NULL;
    if (napi_define_class(env, "Chat", JANAS_NAPI_AUTO_LENGTH,
                          janas_node_constructor,
                          (void *)"a Chat comes from model.chat()",
                          sizeof(d) / sizeof(d[0]), d, &cls) != napi_ok)
        return NULL;
    return cls;
}
