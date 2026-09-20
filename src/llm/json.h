/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * json.h - a small JSON reader and writer for the library: tool definitions,
 * JSON schemas and the arguments of tool calls come in as text.
 *
 * The reader builds a tree in an arena of its own, freed at once. Strings
 * are decoded (escapes resolved, NUL-terminated, with their byte length,
 * which counts embedded NULs); numbers keep their text as written, since a
 * tool definition is written back to the model as it came. The writer
 * follows the tojson filter of the models' chat templates: ", " and ": "
 * between items, non-ASCII characters as they are.
 */
#ifndef JANAS_LLM_JSON_H
#define JANAS_LLM_JSON_H

#include <stddef.h>

enum janas_json_type {
    JANAS_JSON_NULL,
    JANAS_JSON_FALSE,
    JANAS_JSON_TRUE,
    JANAS_JSON_NUMBER,
    JANAS_JSON_STRING,
    JANAS_JSON_ARRAY,
    JANAS_JSON_OBJECT,
};

struct janas_json {
    enum janas_json_type type;
    const char *s;   /* string: decoded; number: its text */
    size_t n;        /* string, number: bytes; array, object: items */
    const char *key; /* inside an object: the member's name */
    size_t key_n;
    struct janas_json *child, *next; /* first item; next sibling */
};

struct janas_json_doc;

/* Reads n bytes of JSON; NULL on error, with the reason (and its byte
   offset) in err. Nesting deeper than 128 is an error. */
struct janas_json_doc *janas_json_parse(const char *text, size_t n, char *err,
                                        size_t err_len);
const struct janas_json *janas_json_root(const struct janas_json_doc *d);
void janas_json_free(struct janas_json_doc *d);

/* The member named key of an object (NULL: none, or not an object). */
const struct janas_json *janas_json_get(const struct janas_json *obj,
                                        const char *key);
/* The string of v, or NULL when v is not a string. */
const char *janas_json_str(const struct janas_json *v);
/* 1 when v is a string equal to s. */
int janas_json_is(const struct janas_json *v, const char *s);
/* The value of a number; def when v is not one. */
double janas_json_num(const struct janas_json *v, double def);

/* A growing byte buffer; on a failed allocation it stops growing and
   remembers it (oom), so a chain of writes needs one check at the end. */
struct janas_buf {
    char *p;
    size_t n, cap;
    int oom;
};

void janas_buf_put(struct janas_buf *b, const char *s, size_t n);
void janas_buf_puts(struct janas_buf *b, const char *s);
void janas_buf_printf(struct janas_buf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
void janas_buf_free(struct janas_buf *b);

/* v as JSON text, the way the chat templates' tojson writes it. */
void janas_json_write(struct janas_buf *b, const struct janas_json *v);
/* n bytes of text as a JSON string, quotes included. */
void janas_json_write_str(struct janas_buf *b, const char *s, size_t n);

#endif
