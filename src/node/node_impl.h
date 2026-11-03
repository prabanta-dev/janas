/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * node_impl.h - what the files of the module for Node.js share: janas.node,
 * the library's API for JavaScript (janas.d.ts says it in TypeScript).
 *
 * Every open model, a speech model too, has a thread of its own, the
 * engine's: the library wants a model and its chat used from one thread at
 * a time, and JavaScript's
 * thread must never wait for a model. JavaScript's thread posts jobs to it
 * (open, a chat, a reply, close), and the engine's gives back each job when
 * done, and the pieces of a reply as they come, through a thread-safe
 * function: every JavaScript value is made on JavaScript's thread. The
 * fields belong to JavaScript's thread where a comment does not say
 * otherwise.
 */
#ifndef JANAS_NODE_IMPL_H
#define JANAS_NODE_IMPL_H

#include <pthread.h>
#include <stdatomic.h>

#include "janas/llm.h"
#include "node/napi.h"

/* the module's own failure, beside the library's JANAS_LLM_E... */
#define JANAS_NODE_ECLOSED (-100) /* the model or the chat is closed */

struct janas_node_model;
struct janas_node_reply;

/* The classes and the engines of one Node environment (the main thread's,
   or a worker's): every environment loads the module anew. */
struct janas_node_env {
    int refs;         /* the environment, and each model opened in it */
    int teardown;     /* it is going away: no JavaScript any more */
    int constructing; /* the module is making an object: without it the
                         classes refuse new */
    napi_ref model_class, chat_class, reply_class, speech_class;
    struct janas_node_model *models; /* those whose engine runs */
};

/* What the engine's thread gives back: a job, or a message of a reply.
   The first member of both. */
#define JANAS_NODE_BACK_JOB 0      /* a job is done */
#define JANAS_NODE_BACK_TEXT 1     /* a piece of a reply */
#define JANAS_NODE_BACK_PROGRESS 2 /* a block of a reply's prompt read */
struct janas_node_back {
    int kind;
    struct janas_node_back *next; /* undelivered, when the environment went */
};

/* A job for the engine's thread. */
struct janas_node_job {
    struct janas_node_back back;
    struct janas_node_job *next; /* in the queue */
    struct janas_node_model *m;
    /* on the engine's thread: the work, which sets rc (and error) */
    void (*run)(struct janas_node_job *j);
    /* on JavaScript's: the value its promise resolves with (NULL:
       undefined); called only when rc is not negative */
    napi_value (*value)(napi_env env, struct janas_node_job *j);
    /* on JavaScript's, after its promise is settled (may be NULL) */
    void (*after)(napi_env env, struct janas_node_job *j);
    /* frees data, and nothing else (NULL: free) */
    void (*drop)(void *data);
    napi_deferred deferred; /* the promise it settles, or NULL */
    int last;               /* the engine's thread ends after it */
    int32_t rc;
    char error[512];
    void *data;
};

/* A piece of a reply, or how much of its prompt is read. */
struct janas_node_msg {
    struct janas_node_back back;
    struct janas_node_msg *next; /* in the reply's queue */
    struct janas_node_reply *reply;
    int reasoning;        /* the piece is the model's reasoning */
    uint32_t read, total; /* prompt tokens read, of total */
    size_t len;
    char text[];
};

struct janas_node_model {
    struct janas_node_model *next; /* in the environment's list */
    struct janas_node_env *st;
    int refs;    /* its object, its chat's and its replies', the engine */
    int holders; /* of them, JavaScript's objects: the last one gone closes
                    the model */
    int pending; /* jobs posted and not back: while there are some, the
                    thread-safe function keeps the event loop alive */
    int closing; /* close asked for, or the model did not open */
    int engine;  /* the engine's thread runs */
    int has_chat;
    napi_deferred *close_wait; /* close() calls waiting */
    int n_close_wait;
    struct janas_node_job *close_job; /* made with the engine, so that a
                                         model can always be closed */
    struct janas_node_reply *replies; /* those not over */
    napi_threadsafe_function tsfn;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    /* under mu */
    struct janas_node_job *head, *tail;  /* the queue */
    struct janas_node_back *undelivered; /* the environment went first */
    /* the engine's thread's; JavaScript's reads llm (status()) only after
       the open is back and before close is asked for */
    janas_llm *llm;
    janas_llm_chat *chat;
    janas_llm_speech *speech; /* a speech model's engine has this instead */
};

struct janas_node_reply {
    struct janas_node_reply *next; /* in the model's list, until it is over */
    struct janas_node_model *m;
    int refs;        /* its object, and its job until the job is back */
    int gone;        /* its object was collected */
    atomic_int stop; /* asked to stop: read by the engine's thread */
    int ended, error_given, returned;
    int32_t rc;
    char error[512];
    int stopped; /* it ended because it was asked to */
    struct janas_llm_chat_stats st;
    struct janas_node_msg *q, *q_tail; /* pieces not taken yet */
    napi_deferred *wait;               /* next() calls waiting */
    int n_wait, cap_wait;
    napi_deferred *stats_wait; /* stats() calls waiting */
    int n_stats_wait, cap_stats_wait;
    napi_ref signal, listener; /* an AbortSignal and the listener on it */
    napi_ref chat;             /* the chat's object, until the reply is over */
    int collect;               /* text() took the pieces: the answer's */
    char *answer;              /* text so far */
    size_t answer_len, answer_cap;
    napi_deferred *text_wait; /* text() calls waiting */
    int n_text_wait, cap_text_wait;
    /* what is sent, read by the engine's thread */
    int load; /* janas_llm_chat_load, else janas_llm_chat_send */
    char *text;
    size_t len;
    int32_t n;
    int32_t *roles;
    char **texts;
    int32_t *lens;
};

struct janas_node_chat {
    struct janas_node_model *m;
    int closed;
    struct janas_llm_chat_params p; /* as last set, for set() to change */
};

/* addon.c: the environment, values, errors, options */
struct janas_node_env *janas_node_env(napi_env env);
void janas_node_env_unref(struct janas_node_env *st);
napi_value janas_node_undefined(napi_env env);
napi_value janas_node_string(napi_env env, const char *s, size_t len);
napi_value janas_node_number(napi_env env, double v);
napi_value janas_node_bool(napi_env env, int v);
void janas_node_put(napi_env env, napi_value obj, const char *name,
                    napi_value v);
const char *janas_node_code(int32_t rc);
napi_value janas_node_error(napi_env env, int32_t rc, const char *msg);
/* throw, and return NULL for the callback to return */
napi_value janas_node_throw(napi_env env, int32_t rc, const char *msg);
napi_value janas_node_type_error(napi_env env, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
/* a promise and its deferred; one already settled */
napi_value janas_node_promise(napi_env env, napi_deferred *d);
napi_value janas_node_resolved(napi_env env, napi_value v);
napi_value janas_node_rejected(napi_env env, napi_value err);
/* the classes' tags: an object of one is never taken for another's */
extern const napi_type_tag janas_node_model_tag, janas_node_chat_tag,
    janas_node_reply_tag, janas_node_speech_tag;
/* an object of a class, tagged and wrapping native, fin called when it is
   collected; NULL when it could not be made (fin not called then) */
napi_value janas_node_new(napi_env env, napi_ref cls, const napi_type_tag *tag,
                          void *native, napi_finalize fin);
/* the classes' constructor: refuses new from JavaScript (its data says
   where an object of the class comes from) */
napi_value janas_node_constructor(napi_env env, napi_callback_info info);
/* the native object of this, and the arguments (argc in-out, may be NULL);
   NULL after throwing when this is not an object of the class */
void *janas_node_self(napi_env env, napi_callback_info info,
                      const napi_type_tag *tag, size_t *argc, napi_value *argv,
                      const char *what);
/* a string's UTF-8 bytes, NUL-terminated, malloc'd; NULL when v is not a
   string (nothing thrown) or memory ran out (*len SIZE_MAX) */
char *janas_node_text(napi_env env, napi_value v, size_t *len);
/* a copy of the bytes of a Buffer, a typed array or an ArrayBuffer,
   malloc'd; NULL when v is none of them or is empty (nothing thrown), or
   memory ran out (*len SIZE_MAX) */
void *janas_node_bytes(napi_env env, napi_value v, size_t *len);
/*
 * Options, held to the names known: janas_node_options throws a TypeError
 * for anything but an object, undefined or null, and for a name it does not
 * know, and returns -1 then (0 otherwise). The getters return 1 with the
 * value, 0 when the option is absent (undefined), -1 after throwing a
 * TypeError when it has another type.
 */
int janas_node_options(napi_env env, napi_value obj, const char *what,
                       const char *const *known);
int janas_node_opt(napi_env env, napi_value obj, const char *what,
                   const char *name, napi_valuetype type, napi_value *v);
int janas_node_opt_number(napi_env env, napi_value obj, const char *what,
                          const char *name, double lo, double hi, double *out);
int janas_node_opt_int(napi_env env, napi_value obj, const char *what,
                       const char *name, int64_t lo, int64_t hi, int64_t *out);
int janas_node_opt_bool(napi_env env, napi_value obj, const char *what,
                        const char *name, int *out);
/* *out malloc'd */
int janas_node_opt_string(napi_env env, napi_value obj, const char *what,
                          const char *name, char **out);

/* engine.c: the engine's thread, its jobs, a model's life */
struct janas_node_model *janas_node_engine(napi_env env,
                                           struct janas_node_env *st);
struct janas_node_job *janas_node_job(struct janas_node_model *m,
                                      void (*run)(struct janas_node_job *j));
void janas_node_post(napi_env env, struct janas_node_job *j);
/* on the engine's thread */
void janas_node_give(struct janas_node_model *m, struct janas_node_back *b);
void janas_node_ref(struct janas_node_model *m);
void janas_node_unref(napi_env env, struct janas_node_model *m);
void janas_node_hold(struct janas_node_model *m);
void janas_node_release(napi_env env, struct janas_node_model *m);
/* asks the engine to close the model; d (may be NULL) settles when it has */
void janas_node_close(napi_env env, struct janas_node_model *m,
                      napi_deferred d);
void janas_node_teardown(struct janas_node_env *st);

/* model.c */
napi_value janas_node_model_class(napi_env env);
napi_value janas_node_open(napi_env env, napi_callback_info info);

/* chat.c */
napi_value janas_node_chat_class(napi_env env);
napi_value janas_node_chat(napi_env env, struct janas_node_model *m,
                           napi_value options);

/* speech.c: openSpeech() and the Speech class, a Whisper model on an
   engine of its own */
napi_value janas_node_speech_class(napi_env env);
napi_value janas_node_open_speech(napi_env env, napi_callback_info info);

/* reply.c */
/* an AbortSignal watched: *stop set to 1 when it aborts (at once if it
   has); its references in *sig and *listener, for janas_node_unwatch,
   which must come before stop's memory goes. 0, or -1 after throwing */
int janas_node_watch(napi_env env, napi_value signal, atomic_int *stop,
                     napi_ref *sig, napi_ref *listener);
void janas_node_unwatch(napi_env env, napi_ref *sig, napi_ref *listener);
napi_value janas_node_reply_class(napi_env env);
/* a reply of c (its object chat_obj, kept alive while the reply runs) */
napi_value janas_node_reply(napi_env env, struct janas_node_chat *c,
                            napi_value chat_obj, int load, napi_value what,
                            napi_value options);
void janas_node_reply_back(napi_env env, struct janas_node_msg *msg);
/* asks every reply of the model not over to stop */
void janas_node_stop_replies(struct janas_node_model *m);

#endif
