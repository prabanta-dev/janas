/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jsonc.h - a member of a JSON file changed in place (jsonc.c): the file
 * as VS Code writes and reads its settings, JSON with comments (// and
 * slash-star) and trailing commas. Only the bytes of the member changed
 * are touched; every other byte - comments, order, spacing - stays as it
 * was. A file that is not such JSON is left alone: an error, not a guess.
 */
#ifndef JANAS_COMMON_JSONC_H
#define JANAS_COMMON_JSONC_H

#include <stddef.h>

/*
 * The text of the value at path (n keys, each into an object) in text:
 * *at and *len, 1; 0 when a key is not there; -1 when text is not JSON
 * with comments or a step is not an object.
 */
int janas_jsonc_find(const char *text, size_t n, const char *const *path,
                     int n_path, size_t *at, size_t *len);

/*
 * text with the value at path set to value (JSON, written as it is): the
 * objects on the way made when they are not there, a new member added
 * last with the indentation of the members beside it. A new text
 * (malloc'd, NUL-ended) in *out, its length in *out_n: 0, or -1 when text
 * is not JSON with comments or a step is not an object. An empty text is
 * an empty object.
 */
int janas_jsonc_set(const char *text, size_t n, const char *const *path,
                    int n_path, const char *value, char **out, size_t *out_n);

/* text without the member at path: 1 removed (*out the new text), 0 it was
   not there (*out NULL), -1 as janas_jsonc_set. */
int janas_jsonc_remove(const char *text, size_t n, const char *const *path,
                       int n_path, char **out, size_t *out_n);

/*
 * In the array at the top of text (an empty text is an empty one): the
 * object whose member key is the string match replaced by value, or value
 * added last when there is none. *out as janas_jsonc_set; 0 or -1.
 */
int janas_jsonc_array_put(const char *text, size_t n, const char *key,
                          const char *match, const char *value, char **out,
                          size_t *out_n);

/* The same object taken out of the array: 1, 0 none, -1 as above. */
int janas_jsonc_array_remove(const char *text, size_t n, const char *key,
                             const char *match, char **out, size_t *out_n);

#endif
