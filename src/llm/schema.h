/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * schema.h - JSON and JSON Schema as grammar rules (grammar.h).
 *
 * What a schema says is enforced where a grammar can say it: types, the
 * properties of an object (in the order the schema lists them, the required
 * ones always there, no others), array items and small bounds on their
 * count, string lengths up to 64, enum and const, anyOf and oneOf, $ref to
 * the same document (recursion included), nullable. What it cannot say
 * (pattern, format, minimum and maximum, allOf of more than one schema) is
 * left out: the value is still valid JSON of the right type, and
 * janas_jsg_relaxed counts the keywords so treated.
 */
#ifndef JANAS_LLM_SCHEMA_H
#define JANAS_LLM_SCHEMA_H

#include <stdint.h>

#include "grammar.h"
#include "json.h"

struct janas_jsg;

/* Rules for JSON, made in b (which must outlive the result's use). */
struct janas_jsg *janas_jsg_new(struct janas_gbuild *b);
void janas_jsg_free(struct janas_jsg *j);

/* White space between tokens of JSON: at most 20 bytes in a row. */
uint32_t janas_jsg_ws(struct janas_jsg *j);
/* Any JSON value; any object. */
uint32_t janas_jsg_any(struct janas_jsg *j);
uint32_t janas_jsg_object(struct janas_jsg *j);
/* A value valid against schema; root is the document $ref points into. */
uint32_t janas_jsg_schema(struct janas_jsg *j, const struct janas_json *schema,
                          const struct janas_json *root);
/* Keywords seen and not enforced. */
int janas_jsg_relaxed(const struct janas_jsg *j);

#endif
