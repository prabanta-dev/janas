/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * constrain.h - what a reply may say, token by token: a filter for the
 * sampler (sample.h) that follows the reply through its parts.
 *
 * - The reasoning, when the reply starts with it, is free.
 * - The answer is free, or held to a grammar (a JSON Schema, the tool calls
 *   that tool_choice requires); held, it may end - a stop token - only
 *   where the grammar is complete.
 * - A tool call the model opens of its own accord (tool_choice auto) is
 *   held to the grammar of the calls from its opening token to its closing
 *   one, so its arguments are always valid; then the answer is free again.
 * - With calls forbidden (tool_choice none) the opening token never comes.
 */
#ifndef JANAS_LLM_CONSTRAIN_H
#define JANAS_LLM_CONSTRAIN_H

#include <stdint.h>

#include "grammar.h"
#include "sample.h"
#include "tokenizer.h"

#define JANAS_NO_RULE UINT32_MAX

struct janas_constraint_spec {
    const struct janas_grammar *g; /* NULL when nothing is held to one */
    uint32_t answer;     /* the answer's rule, or JANAS_NO_RULE: free */
    uint32_t call;       /* a call's body up to its closing token, or none */
    int32_t call_open;   /* the token that opens a call (-1: none) */
    int ban_calls;       /* the answer may not open a call */
    int thinking;        /* the reply starts in its reasoning... */
    int32_t think_close; /* ...which this token ends */
    const int32_t *stop; /* the tokens that end the reply */
    int n_stop;
};

struct janas_constraint;

/* NULL on memory. The grammar and the tokenizer must outlive it. */
struct janas_constraint *
janas_constraint_new(const struct janas_constraint_spec *spec,
                     const struct janas_tokenizer *tok);
void janas_constraint_free(struct janas_constraint *c);

/* The filter for the sampler, bound to c. */
void janas_constraint_filter(struct janas_constraint *c,
                             struct janas_sample_filter *f);

/* 1 while a grammar holds the reply (an answer or a call not complete). */
int janas_constraint_held(const struct janas_constraint *c);

#endif
