/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.h - the tools a model may call, in the words of its chat template.
 *
 * The Qwen models speak one of three dialects, told apart by their
 * templates:
 * - JSON (Qwen3, Qwen3-Next): the tools listed as JSON in the system
 *   message, a call written as <tool_call>{"name": ..., "arguments": ...}
 *   </tool_call>;
 * - XML (Qwen3.5, Qwen3.6): the tools as JSON, a call as
 *   <tool_call><function=name><parameter=key>value</parameter>...;
 * - XML_CODER (Qwen3-Coder): calls as XML, and the tools described in XML
 *   too, under a system message of its own when there is none.
 * <tool_call> and </tool_call> are special tokens of the vocabulary; the
 * rest is text. Texts made here carry the markers as text, and whoever turns
 * them into tokens turns the markers into their ids.
 */
#ifndef JANAS_LLM_TOOLS_H
#define JANAS_LLM_TOOLS_H

#include <stddef.h>
#include <stdint.h>

#include "grammar.h"
#include "json.h"
#include "schema.h"

enum janas_tool_dialect {
    JANAS_TOOLS_NONE,
    JANAS_TOOLS_JSON,
    JANAS_TOOLS_XML,
    JANAS_TOOLS_XML_CODER,
    /* Gemma 4: <|tool>declaration:f{...}<tool|> in the system turn, calls
       <|tool_call>call:f{key:value,...}<tool_call|>, strings between <|"|>
       tokens, answers <|tool_response>response:f{value:...}<tool_response|>
       inside the model's own turn */
    JANAS_TOOLS_GEMMA,
};

/* The dialect of a chat template (NONE when it has no tools). */
enum janas_tool_dialect janas_tools_dialect(const char *tmpl);

/* Tool definitions, as OpenAI's API gives them: an array of
   {"type": "function", "function": {...}}, or of the functions alone. */
struct janas_toolset;

struct janas_toolset *janas_toolset_parse(const char *json, size_t n, char *err,
                                          size_t err_len);
void janas_toolset_free(struct janas_toolset *t);
uint32_t janas_toolset_count(const struct janas_toolset *t);
/* The index of the function called name, or -1. */
int janas_toolset_find(const struct janas_toolset *t, const char *name,
                       size_t n);

/* The system message's content when there are tools: sys (NULL for none)
   with the tools, as the dialect's template writes them. */
void janas_tools_system(struct janas_buf *out, enum janas_tool_dialect d,
                        const struct janas_toolset *t, const char *sys,
                        size_t sys_n);

/* A call made in an earlier turn: its name and its arguments (JSON text). */
struct janas_tool_call {
    const char *name, *args;
    size_t name_n, args_n;
};

/* An assistant message with calls, after the role: its content, then each
   call as the dialect writes it. */
void janas_tools_assistant(struct janas_buf *out, enum janas_tool_dialect d,
                           const char *content, size_t n,
                           const struct janas_tool_call *calls, size_t n_calls);

/*
 * A call the model wrote, from what came between <tool_call> and
 * </tool_call>, into its name and its arguments as a JSON object. Returns 0,
 * or -1 when it is no call.
 */
int janas_tools_parse(enum janas_tool_dialect d, const struct janas_toolset *t,
                      const char *body, size_t n, struct janas_buf *name,
                      struct janas_buf *args);

/*
 * A rule for the body of a call and its closing token, `close`: any of the
 * functions (only < 0) or function `only`, with arguments valid against its
 * parameters.
 */
/* Gemma 4's call rule: the quote token <|"|> too. */
uint32_t janas_tools_rule_gemma(struct janas_gbuild *b,
                                const struct janas_toolset *t, int only,
                                int32_t close, int32_t quote);

/* Gemma 4: a tool's answer as its template writes it, without the opening
   <|tool_response> (the model's stop token, already in the sequence, or
   written by the caller): response:name{value:<|"|>text<|"|>}<tool_response|>
   - the text as it is, JSON or not, as the template quotes a message's
   content. */
void janas_tools_gemma_response(struct janas_buf *out, const char *name,
                                size_t name_n, const char *text, size_t n);

uint32_t janas_tools_rule(struct janas_gbuild *b, struct janas_jsg *j,
                          enum janas_tool_dialect d,
                          const struct janas_toolset *t, int only,
                          int32_t close);

#endif
