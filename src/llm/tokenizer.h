/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tokenizer.h - the BPE tokenizers: byte-level (Qwen) and SentencePiece
 * style (Gemma 4).
 *
 * Built from the model file's metadata (tokenizer.ggml.tokens, .merges,
 * .token_type). Byte-level: encoding splits the text as the model's
 * pre-tokenizer does (its regular expression, rendered as a hand-written
 * matcher), maps bytes to the byte-level alphabet and applies the merges by
 * rank. SentencePiece style: spaces become U+2581, the merges run on UTF-8
 * characters over whole lines, and a character with no token is written as
 * its bytes (<0xXX>). Text is taken
 * as given: the model's NFC normalization is not applied (typed text is
 * normally already NFC).
 */
#ifndef JANAS_LLM_TOKENIZER_H
#define JANAS_LLM_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#include "jns.h"

struct janas_tokenizer;

struct janas_tokenizer *janas_tokenizer_create(const struct janas_jns *j,
                                               char *err, size_t err_len);
void janas_tokenizer_destroy(struct janas_tokenizer *t);

uint32_t janas_tokenizer_n_vocab(const struct janas_tokenizer *t);

/*
 * Encodes len bytes of UTF-8 text into at most max token ids. With special,
 * the model's special tokens written in the text ("<|im_start|>" ...) become
 * their ids; without, they are ordinary text (for text typed by a user).
 * Returns the number of tokens of the whole text, which can exceed max (call
 * again with room for them), or -1 on error.
 */
long janas_tokenizer_encode(const struct janas_tokenizer *t, const char *text,
                            size_t len, int special, int32_t *out, size_t max);

/*
 * The bytes a token stands for (a special token: its text). Writes at most
 * max bytes and returns the full length. A token can end inside a UTF-8
 * sequence: concatenate, do not decode tokens one by one as text.
 */
size_t janas_tokenizer_decode(const struct janas_tokenizer *t, int32_t id,
                              char *out, size_t max);

/* The special tokens (control and user-defined: "<|im_end|>",
   "<tool_call>"...), longest text first; returns how many. */
uint32_t janas_tokenizer_specials(const struct janas_tokenizer *t,
                                  const int32_t **ids);

/* The id of a token given by its text (e.g. "<|im_end|>"), or -1. */
int32_t janas_tokenizer_find(const struct janas_tokenizer *t, const char *s);

#endif
