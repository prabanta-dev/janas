/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * fim.h - the window of a fill-in-the-middle request (fim.c): the part of
 * the text before the cursor and after it that the model reads, chosen so
 * that the next keystroke finds most of it already computed.
 */
#ifndef JANAS_SERVER_FIM_H
#define JANAS_SERVER_FIM_H

#include <stddef.h>
#include <stdint.h>

struct srv_fim;

/* The tokens of n bytes of s, or -1. */
typedef int32_t (*srv_fim_count)(void *ctx, const char *s, size_t n);

/* prefix_tokens: the text before the cursor read at most (0: all of it);
   suffix_lines: the lines after it (0: all of them). */
struct srv_fim *srv_fim_new(uint32_t prefix_tokens, uint32_t suffix_lines);
void srv_fim_free(struct srv_fim *f);

/* The window: the prefix from *pre_off on, the first *suf_len bytes of the
   suffix. *how says why the prefix was cut ("" when it was not). */
void srv_fim_window(struct srv_fim *f, srv_fim_count count, void *ctx,
                    const char *pre, size_t n_pre, const char *suf,
                    size_t n_suf, size_t *pre_off, size_t *suf_len,
                    const char **how);

#endif
