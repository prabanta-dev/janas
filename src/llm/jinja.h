/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jinja.h - the chat templates of the models, run: Jinja as Hugging Face's
 * transformers renders them (trim_blocks, lstrip_blocks, the loop
 * controls, tojson, raise_exception, strftime_now), Janas's own
 * interpreter (jinja_parse.c, jinja_eval.c, jinja_lib.c).
 *
 * The text that comes from the conversation is remembered byte by byte as
 * it goes through the template - sliced, trimmed, joined, written as JSON
 * - so that the prompt's caller can read the model's special tokens only
 * in what the template itself wrote: a user who types "<|im_start|>" in a
 * message writes text, never the token.
 */
#ifndef JANAS_LLM_JINJA_H
#define JANAS_LLM_JINJA_H

#include <stddef.h>
#include <stdint.h>

#include "llm/json.h"

struct janas_jinja;

/* A template read; NULL with the reason and where in err. */
struct janas_jinja *janas_jinja_parse(const char *src, size_t n, char *err,
                                      size_t err_len);
void janas_jinja_free(struct janas_jinja *t);

/*
 * Renders t with the variables of two JSON objects (either may be NULL):
 * vars, Janas's own (add_generation_prompt, bos_token, the kwargs...), and
 * data, what the conversation brings (messages, tools, documents), whose
 * strings are marked. out gets the text; marks, as many bytes, 1 where
 * the byte came from data. Returns 0, or -1 with the reason in err (a
 * raise_exception's message, an error of the template, a limit).
 */
int janas_jinja_render(const struct janas_jinja *t,
                       const struct janas_json *vars,
                       const struct janas_json *data, struct janas_buf *out,
                       struct janas_buf *marks, char *err, size_t err_len);

#endif
