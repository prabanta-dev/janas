/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * template.h - a layout filled with data (template.c), for the services'
 * answers: the service gives the data and an English layout; a client
 * shows the user the layout filled, in the user's language, and spares
 * the model reading and writing it.
 *
 * A layout is text with fields, and after a line "---" its words, one a
 * line, "key = text":
 *
 *   {{name}}          the value of name in the data ("a.b": b in a)
 *   {{name|w}}        the word "w.<value>" when the words have it, else
 *                     the value (a compass point, a weather code...)
 *   {{#name}}...{{/name}}  the part for each item of a list, or once when
 *                     the value is there (not null, false, 0 or empty)
 *   {{^name}}...{{/name}}  the part when it is not
 *   {{@n}}            the item's number in its list, from 1
 *   {{.}}             the item itself (a list of texts)
 *
 * A line holding only a part's tag is left out whole. Numbers are written
 * as the data has them, with the word "decimal" as their point when the
 * words give one ("decimal = ,"). A translation changes the text and the
 * words, never the fields: janas_tpl_same_shape checks it (a "decimal" may
 * be added).
 */
#ifndef JANAS_COMMON_TEMPLATE_H
#define JANAS_COMMON_TEMPLATE_H

#include <stddef.h>

#include "llm/json.h"

/* The layout tpl (n bytes) filled with data (an object) into out. 0, or -1
   with the reason in err: a part not closed, a tag not ended. */
int janas_tpl_render(const char *tpl, size_t n, const struct janas_json *data,
                     struct janas_buf *out, char *err, size_t err_len);

/* 1 when b has the fields, parts and words of a (a translation of it), 0
   with the first difference in why. */
int janas_tpl_same_shape(const char *a, size_t an, const char *b, size_t bn,
                         char *why, size_t why_len);

/*
 * A tools/call result made of data and a layout, the members into b: the
 * layout filled in English as its text, for the clients that read text;
 * the data as structuredContent; and in _meta, under
 * "dev.prabanta.janas/layout", its name, the layout and the brief - a
 * layout for the model, filled by a client that shows the user the layout
 * in the user's language (janas-chat) and gives the model only the brief.
 * 0, or -1 with the reason in err when the layout could not be filled.
 */
int janas_tpl_result(struct janas_buf *b, const char *name, const char *data,
                     size_t data_n, const char *layout, const char *brief,
                     char *err, size_t err_len);

#endif
