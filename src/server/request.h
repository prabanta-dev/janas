/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * request.h - what request.c and request_opts.c share: the reading of a
 * request's JSON into a job. Private to them.
 */
#ifndef JANAS_SERVER_REQUEST_H
#define JANAS_SERVER_REQUEST_H

#include <yyjson.h>

#include "common/printf_like.h"
#include "server.h"

/* The error of a request that cannot be served, for the 400 answer: always
   returns 0, so that BAD() can sit inside an expression. */
int srv_perr(struct srv_perr *e, const char *code, const char *fmt, ...)
    JANAS_PRINTF_LIKE(3, 4);
#define BAD(...) srv_perr(e, __VA_ARGS__)

/* Marks the request failed; returns -1. */
int srv_pfail(struct srv_parsed *pr, int ret);
char *srv_dup_n(const char *s, size_t n);
/* A number between lo and hi: 1 with *out set, 0 absent (or null), -1 not a
   number or out of range. */
int srv_num(yyjson_val *v, double lo, double hi, double *out);
/* A ceiling on the reasoning by llama.cpp's names, reasoning_budget_tokens
   or thinking_budget_tokens: -1 none, 0 no reasoning, N at most N tokens of
   it, then the answer. 0, or -1 when the value is not one of those. */
int srv_reasoning_budget(yyjson_val *root, struct janas_llm_chat_params *p);
/* A value as JSON text (NULL on memory). */
char *srv_json_text(yyjson_val *v, size_t *n);

/* request_opts.c: the parameters of both endpoints, and those only chat
   has (tools, response_format, logprobs...). */
int srv_read_params(struct srv_perr *e, struct srv_parsed *pr, yyjson_val *root,
                    struct srv_job *j, int chat);
int srv_read_chat_opts(struct srv_perr *e, struct srv_parsed *pr,
                       yyjson_val *root, struct srv_job *j);

#endif
