/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * server.h - janas-server: the model behind an HTTP API that follows
 * OpenAI's (https://github.com/openai/openai-openapi).
 *
 * Three parts. http.c takes the requests (libmicrohttpd, a thread per
 * connection), finds the route in the table generated from the
 * specification (routes.inc) and answers. engine.c owns the model: one
 * thread runs the requests one at a time, in the order they came, and hands
 * the text over as it is written. The handlers (models.c, chat.c) read the
 * JSON of a request into a job and write the JSON of the answer.
 */
#ifndef JANAS_SERVER_H
#define JANAS_SERVER_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "janas/llm.h"
#include "store.h"

struct MHD_Connection;

struct srv_config {
    const char *model_path;
    struct janas_llm_params llm; /* how the model is opened */
    int no_gpu;
    const char *model_id; /* the name clients use; default: the file name */
    const char *host;     /* address to listen on (default 127.0.0.1) */
    uint16_t port;        /* default 8080 */
    const char *api_key;  /* NULL: none asked */
    uint32_t max_queue;   /* requests waiting at most (default 16) */
    size_t max_body;      /* bytes of a request body at most */
    int max_connections;
    int verbose;
    int test;     /* serve the test chat page at /test */
    int thinking; /* reasoning when a request does not say: 1, 0, or -1
                     for the model's own default */
    const char *embed_path; /* the embedding model, or NULL */
    const char *embed_id;   /* its name for clients */
    const char *store_dir;  /* where stored documents outlive the server;
                               NULL: memory only (--no-store) */
    int keep;               /* conversations kept computed besides the
                               current one (default 8) */
    uint64_t keep_bytes;    /* memory for them; 0: automatic */
    uint64_t keep_disk;     /* disk for them, below the memory; 0: none
                               (default 8 GiB) */
    int no_mcp;             /* refuse the tools of type mcp (Responses):
                               no connection goes out on a request's word */
    uint32_t max_input;     /* longest prompt taken, tokens; 0: the context */
    uint32_t warn_input;    /* prompts longer are taken, and said on stderr */
};

/* The engine: the model, its conversations, and the queue. */
struct srv_engine;

/* JOB_COUNT: only how many tokens the messages are (j->prompt_tokens) */
enum job_kind { JOB_CHAT, JOB_TEXT, JOB_COUNT, JOB_EMBED };

/* A message of a chat request, with the tool calls of an assistant's. */
struct srv_msg {
    int32_t role; /* JANAS_LLM_ROLE_... */
    char *text;
    size_t n;
    int n_calls;
    char **call_name, **call_args;
};

/* A prompt of a completion: text, or token ids (the engine turns them
   into text, which is where an echo finds it). */
struct srv_prompt {
    char *text;
    size_t n;
    int32_t *ids;
    size_t n_ids;
};

/* A token of the answer with its log-probability, and the alternatives. */
struct srv_alt {
    char *tok; /* its bytes */
    size_t n;
    float lp;
};
struct srv_lp {
    struct srv_alt t;
    int n_top;
    struct srv_alt *top;
};

/* A tool call of the reply. */
struct srv_call {
    char id[40];
    char *name, *args;
};

/* One of the replies a request asks for (n, best_of). */
struct srv_choice {
    char *think, *text; /* the reasoning and the answer as they grow */
    size_t n_think, cap_think, n_text, cap_text;
    struct srv_call *calls;
    int n_calls, cap_calls;
    struct srv_lp *lp; /* the answer's tokens, when logprobs are asked */
    size_t n_lp, cap_lp;
    int by_stop;    /* ended on one of the job's stop strings */
    int32_t finish; /* JANAS_LLM_FINISH_... */
    int done;
    double score; /* the mean log-probability, for best_of */
    uint32_t tokens, reasoning_tokens;
};

/* A request for the model, filled by a handler, run by the engine. */
struct srv_job {
    enum job_kind kind;
    /* JOB_CHAT: the messages; JOB_TEXT: the texts to continue (or to fill
       in before suffix), n_choices / n_prompts replies each */
    struct srv_msg *msg;
    int32_t n_msg;
    struct srv_prompt *prompts;
    int n_prompts;
    char *suffix;
    size_t n_suffix;
    struct janas_llm_chat_params p;
    int seeded;    /* the request gave a seed: choice k takes seed + k */
    char *stop[4]; /* strings that end the reply where they appear */
    int n_stop;
    /* tools (a JSON array, as the request has it), tool_choice, and the
       form of the answer */
    char *tools;
    size_t n_tools;
    int32_t tool_choice;
    char tool_name[128];
    int parallel;
    int32_t format;
    char *schema;
    size_t n_schema;
    int32_t *bias_id;
    float *bias;
    int32_t n_bias;
    int32_t dims; /* JOB_EMBED: the vectors' length (0: the model's) */
    float *emb;   /* JOB_EMBED: n_prompts vectors of emb_dim, the engine's */
    int32_t emb_dim;
    /* Responses, tools of type mcp: the outputs of the calls approved in
       this request, by the id of the approval request */
    char **mcp_id, **mcp_out;
    int n_mcp;
    int n_choices; /* replies to generate */
    int n_keep;    /* of which to return (best_of: the best n_keep) */

    /* what the engine writes, under mu */
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct srv_choice *ch; /* n_choices of them */
    int cur;               /* the one being written */
    int started;           /* the engine took it */
    int done;              /* it is over */
    int cancel;            /* the client went away: stop */
    int32_t err;           /* JANAS_LLM_OK or the library's code */
    char errmsg[256];
    struct janas_llm_chat_stats st;               /* of the first reply */
    uint32_t prompt_tokens;                       /* of every prompt, once */
    uint32_t completion_tokens, reasoning_tokens; /* of all of them */
    uint32_t queue_pos; /* requests ahead of it, while it waits */
    int refs;           /* the handler and the engine: freed by the last one */
    struct srv_job *next;
};

struct srv_engine *srv_engine_start(const struct srv_config *cfg, char *err,
                                    size_t err_len);
void srv_engine_stop(struct srv_engine *e);
/* Whether an embedding model is open, and the length of its vectors. */
int32_t srv_engine_embed_dim(const struct srv_engine *e);
/* The model's description (janas_llm_describe), for /v1/models. */
const char *srv_engine_describe(const struct srv_engine *e);
int64_t srv_engine_created(const struct srv_engine *e);
/* The system message janas-chat gives the model (janas_llm_default_system),
   offered to the test page. */
const char *srv_engine_system(const struct srv_engine *e);

struct srv_job *srv_job_new(void);
void srv_job_release(struct srv_job *j);
/* The choices the job's replies go into: 0, or -1 on memory. */
int srv_job_choices(struct srv_job *j, int n, int keep);
/* Queues the job; -1 when the queue is full. */
int srv_engine_submit(struct srv_engine *e, struct srv_job *j);
/* How much of choice k's answer may be handed out now: while the reply
   runs, the end that could still turn out to be the start of a stop string
   is held back. Call with j->mu held. */
size_t srv_job_visible(const struct srv_job *j, int k);

/* The server: configuration, engine, and the HTTP side. */
struct srv {
    struct srv_config cfg;
    struct srv_engine *engine;
    struct srv_store *store; /* completions, responses, conversations */
    struct MHD_Daemon *mhd;
};

struct sockaddr;
/* Listening on addr (the port inside it), and no longer. */
int srv_http_start(struct srv *s, const struct sockaddr *addr, char *err,
                   size_t err_len);
void srv_http_stop(struct srv *s);

/* One request, as the handlers see it. */
struct srv_req {
    struct srv *srv;
    struct MHD_Connection *conn;
    const char *method, *path; /* path without /v1 */
    char param[2][256];        /* the {…} segments of the route, in order */
    int n_param;
    const char *body;
    size_t n_body;
};

/* A handler answers through the functions below and returns their result
   (an MHD_Result). */
typedef int (*srv_handler)(struct srv_req *r);

/* Answers: JSON text (taken over, freed here), an OpenAI error, or a
   stream of server-sent events. */
int srv_reply_json(struct srv_req *r, unsigned status, char *json, size_t n);
int srv_reply_error(struct srv_req *r, unsigned status, const char *type,
                    const char *code, const char *fmt, ...)
    __attribute__((format(printf, 5, 6)));
/* A job refused for an input above --max-input: a 400 with the numbers
   (input_tokens, max_input_tokens, excess_tokens and the parts) as fields
   of the error, for a client to shorten the input by. */
struct srv_job;
int srv_reply_limit(struct srv_req *r, const struct srv_job *j);
/* The stream: next(ctx, buf, cap) writes up to cap bytes and returns how
   many, 0 when it has nothing yet (it may wait a little first), -1 at the
   end; done(ctx) is called once the connection is over. */
typedef long (*srv_stream_next)(void *ctx, char *buf, size_t cap);
typedef void (*srv_stream_done)(void *ctx);
int srv_reply_stream(struct srv_req *r, srv_stream_next next,
                     srv_stream_done done, void *ctx);

/* The handlers */
int srv_models_list(struct srv_req *r);
int srv_models_get(struct srv_req *r);
int srv_models_delete(struct srv_req *r);
int srv_embeddings_create(struct srv_req *r);
int srv_moderations_create(struct srv_req *r);
int srv_chat_create(struct srv_req *r);
int srv_completion_create(struct srv_req *r);
/* stored.c: the completions kept with store: true */
int srv_stored_list(struct srv_req *r);
int srv_stored_get(struct srv_req *r);
int srv_stored_update(struct srv_req *r);
int srv_stored_delete(struct srv_req *r);
int srv_stored_messages(struct srv_req *r);
/* responses.c, conversations.c: OpenAI's Responses API */
int srv_responses_create(struct srv_req *r);
int srv_responses_get(struct srv_req *r);
int srv_responses_delete(struct srv_req *r);
int srv_responses_cancel(struct srv_req *r);
int srv_responses_input_items(struct srv_req *r);
int srv_responses_input_tokens(struct srv_req *r);
int srv_responses_compact(struct srv_req *r);
/* Waits for the responses in the background to be kept, at the end. */
void srv_responses_shutdown(void);
/* At start, the responses a killed server left unfinished are marked as
   failed: how many. */
int srv_responses_recover(struct srv_store *st);
int srv_conv_create(struct srv_req *r);
int srv_conv_get(struct srv_req *r);
int srv_conv_update(struct srv_req *r);
int srv_conv_delete(struct srv_req *r);
int srv_conv_items_list(struct srv_req *r);
int srv_conv_items_create(struct srv_req *r);
int srv_conv_item_get(struct srv_req *r);
int srv_conv_item_delete(struct srv_req *r);
/* The value of a query argument of the request, or NULL. */
const char *srv_query(struct srv_req *r, const char *key);

/* A request read (request.c): what it asks of the answer, or why it
   cannot be served. */
struct srv_parsed {
    int stream, include_usage, echo;
    int logprobs;   /* chat: true or false; completions: how many */
    int store;      /* keep the completion (GET /chat/completions) */
    char *metadata; /* its metadata, as JSON (NULL: none) */
    char *messages; /* the request's messages, as JSON, when stored */
    int failed;     /* read_...: the error is in the srv_perr */
    int ret;
};
void srv_parsed_free(struct srv_parsed *pr);
struct srv_perr {
    const char *code; /* OpenAI's error code, for a 400 */
    char msg[512];
    int oom; /* not the request's fault: out of memory */
};
/* The body of a request to /chat/completions (chat) or /completions into
   the job; 0, or -1 with e filled. */
int srv_parse_request(const char *body, size_t n, int chat, struct srv_job *j,
                      struct srv_parsed *pr, struct srv_perr *e);

/* A new id with a prefix ("chatcmpl-", "cmpl-"), into buf (>= 48 bytes). */
void srv_new_id(char *buf, size_t cap, const char *prefix);

#endif
