/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * responses.h - OpenAI's Responses API and its conversations: what
 * responses.c (the operations), responses_in.c (a request into a job),
 * responses_out.c (the Response object and its events) and conversations.c
 * share. Private to them.
 *
 * Items are the unit of the Responses API: messages, function calls, their
 * outputs, reasoning. They are kept as JSON, every one with an id, in the
 * store: a response keeps the items it was given (its whole context) beside
 * it, a conversation keeps its items beside it, and a new request finds
 * its history there (previous_response_id, conversation).
 */
#ifndef JANAS_SERVER_RESPONSES_H
#define JANAS_SERVER_RESPONSES_H

#include <yyjson.h>

#include "server.h"

#define RESP_KIND "response"
#define CONV_KIND "conversation"

/* A request to /responses, from its reading to its answer. */
struct resp {
    struct srv *srv;
    struct srv_job *j;
    char id[64];
    int64_t created;
    char conv[80]; /* the conversation it belongs to, or "" */
    int store, stream, background;
    int reasoning_asked; /* reasoning.effort said something */
    /* the request's settings the Response object repeats (a JSON object),
       every item of its context (the history and its own input, with ids),
       and its own input alone */
    char *echo, *context, *input;
    /* the stream: the events ready, and how far the reply has been told */
    int seq, started, out_index, think_open, text_open, calls_told;
    size_t told_think, told_text;
    int text_began, think_began, done_sent;
    char item_think[40], item_text[40];
    char *pend;
    size_t n_pend, off;
    /* tools of type mcp (responses_mcp.c): the servers; the output items of
       the rounds over (a JSON array, NULL: none), the job's own among them
       once consumed; how many of them the stream has told; the counts of
       the rounds before the job running */
    struct resp_mcp *mcp;
    char *done_items;
    int consumed, items_told, round_checked;
    int cancelled; /* /responses/{id}/cancel came: no next round (bg_mu) */
    uint32_t prev_in, prev_cached, prev_out, prev_reason;
};

void resp_free(struct resp *x);

/* responses_in.c: the body of POST /responses (or of /responses/input_tokens
   and /responses/compact) into x and its job. 0, or an answer given (the
   handler returns it through *ret). */
int resp_read(struct srv_req *r, struct resp *x, int *ret);
/* Items (a JSON array) as the job's messages, after the history. */
int resp_items_to_msgs(struct srv_job *j, yyjson_val *items, char *err,
                       size_t err_len);
/* A message added to the job (text taken over), a call added to one. */
struct srv_msg *resp_push(struct srv_job *j, int32_t role, char *text,
                          size_t n);
int resp_add_call(struct srv_msg *m, const char *name, char *args);
/* An id for an item that has none ("msg_", "fc_", "rs_"...). */
void resp_item_id(yyjson_mut_doc *d, yyjson_mut_val *item);

/* responses_out.c: the Response object as it stands (status: "completed",
   "in_progress", "queued", "cancelled", "failed", "incomplete" from the
   job when NULL), as JSON text. */
char *resp_object(struct resp *x, const char *status, size_t *n);
/* The output items of the reply, into arr. */
void resp_output(yyjson_mut_doc *d, yyjson_mut_val *arr, struct resp *x);
/* The items of the job's reply alone (the calls to MCP servers left out). */
void resp_job_items(yyjson_mut_doc *d, yyjson_mut_val *arr, struct resp *x);
/* The stream's state of a reply, for the next round's. */
void resp_round_reset(struct resp *x);
/* Once the reply is over: the response kept, and its conversation told. */
void resp_keep(struct resp *x);
/* The events of the stream. */
long resp_stream_next(void *ctx, char *buf, size_t cap);
void resp_stream_done(void *ctx);

/* responses_mcp.c: the tools of type mcp of the request, reached and
   given to the model (after the context is read, before its messages): 0,
   or an answer given. */
int resp_mcp_read(struct srv_req *r, struct resp *x, yyjson_val *root,
                  int *ret);
/* Whether the model's tool name is one of an MCP server's. */
int resp_mcp_owns(const struct resp *x, const char *name);
/* The job is over: its MCP calls run or held for approval; 1 when a next
   round was submitted (x->j is the new job), 0 when the response is over. */
int resp_mcp_round(struct resp *x);
void resp_mcp_free(struct resp_mcp *mc);
/* Stops the MCP calls running for x, from another thread. */
void resp_mcp_cancel(struct resp_mcp *mc);
/* responses.c: the next round's job submitted and made x's, under the lock
   the cancelling takes, so that a cancel never lands on a job already left
   behind; -1 (nothing changed) when x was cancelled or the queue is full. */
int resp_adopt(struct resp *x, struct srv_job *nj);

/* conversations.c: items added to a conversation (a JSON array); 0, or -1
   when there is no such conversation (or memory). */
int conv_append(struct srv *srv, const char *conv, const char *items_json);
/* Its items, as a JSON array text (NULL: no such conversation). */
char *conv_items(struct srv *srv, const char *conv);

#endif
