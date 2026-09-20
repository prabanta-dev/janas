/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * chat_out.h - the answers of /chat/completions and /completions, shared by
 * chat.c (the whole answer), chat_stream.c (the events) and chat_out.c (the
 * JSON they are made of). Private to them.
 */
#ifndef JANAS_SERVER_CHAT_OUT_H
#define JANAS_SERVER_CHAT_OUT_H

#include <yyjson.h>

#include "server.h"

/* An answer being made for a job. */
struct out {
    struct srv_job *j;
    struct srv *srv;
    int chat, include_usage, echo;
    int logprobs; /* chat: 1; completions: the alternatives + 1 */
    int store;
    char *metadata, *messages; /* taken from the srv_parsed */
    char id[64], model[256];
    int64_t created;
    /* the stream: the choice being sent, and how far */
    int k;
    size_t r_think, r_text, r_lp;
    int r_calls, think_began, text_began, sent_role, sent_end;
    uint32_t queue_seen;
    char *pend; /* the events ready, not yet sent */
    size_t n_pend, off;
};

/* The object's start: id, object, created, model, fingerprint. */
yyjson_mut_val *out_head(yyjson_mut_doc *d, const struct out *o,
                         const char *object);
void out_usage(yyjson_mut_doc *d, yyjson_mut_val *obj, const struct srv_job *j);
const char *out_finish(const struct srv_choice *c);
/* s[*from .. n) without the white space at its start, the first time */
size_t out_skip_ws(const char *s, size_t n, size_t *from, int *began);

/* chat: tool calls from..to of choice c, as a message's (with_index 0) or a
   delta's; the log-probabilities of tokens from..to */
yyjson_mut_val *out_calls(yyjson_mut_doc *d, const struct srv_choice *c,
                          int from, int to, int with_index);
yyjson_mut_val *out_chat_lp(yyjson_mut_doc *d, const struct srv_choice *c,
                            size_t from, size_t to);
/* completions: the log-probabilities of tokens from..to, their text
   starting at offset */
yyjson_mut_val *out_text_lp(yyjson_mut_doc *d, const struct srv_choice *c,
                            size_t from, size_t to, size_t offset, int top);

/* The whole answer, once the job is done, as JSON text (NULL on memory):
   what a client that does not stream gets, and what a store keeps. */
char *out_whole(struct out *o, size_t *n);
/* Keeps the answer when the request asked for it (store: true). */
void out_store(struct out *o);
void out_free(struct out *o);

#endif
