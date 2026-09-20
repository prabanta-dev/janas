/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * grammar.h - grammars over bytes, to say which tokens may come next.
 *
 * A grammar is a set of rules; a rule is a choice of alternatives; an
 * alternative is a sequence of elements: a set of bytes, a special token of
 * the model (<tool_call> is one symbol, not eleven bytes), or another rule.
 * Recursion is allowed on the right (a list goes on in its last element),
 * never on the left.
 *
 * Matching keeps every way the input read so far can go on: a set of
 * stacks, each the positions still to finish, innermost first. The stacks
 * share their tails and every node is made once (same position, same
 * parent: same node), so two ways that meet become one and a set compares
 * by pointers. A token is tried by feeding its bytes to a copy of the set
 * and throwing the copy away.
 */
#ifndef JANAS_LLM_GRAMMAR_H
#define JANAS_LLM_GRAMMAR_H

#include <stddef.h>
#include <stdint.h>

/* ---- building ---- */

struct janas_gbuild;

struct janas_gbuild *janas_gbuild_new(void);
void janas_gbuild_free(struct janas_gbuild *b);

/* A new rule with no alternatives yet: its id. */
uint32_t janas_gb_rule(struct janas_gbuild *b);
/* A new alternative of rule r, empty: its handle, to append elements to.
   Alternatives of different rules may be built at the same time. */
uint32_t janas_gb_alt(struct janas_gbuild *b, uint32_t r);

/* Elements, appended to alternative a. */
void janas_gb_set(struct janas_gbuild *b, uint32_t a, const uint8_t set[32]);
void janas_gb_byte(struct janas_gbuild *b, uint32_t a, uint8_t c);
void janas_gb_lit(struct janas_gbuild *b, uint32_t a, const char *s, size_t n);
void janas_gb_token(struct janas_gbuild *b, uint32_t a, int32_t token);
void janas_gb_ref(struct janas_gbuild *b, uint32_t a, uint32_t r);

/* Byte sets: set[c >> 3] bit (c & 7). */
void janas_set_clear(uint8_t set[32]);
void janas_set_add(uint8_t set[32], uint8_t lo, uint8_t hi);
void janas_set_invert(uint8_t set[32]);

/*
 * A rule for any text that does not contain the n bytes of end, followed
 * by end itself: an XML parameter's value up to its closing tag.
 */
uint32_t janas_gb_until(struct janas_gbuild *b, const char *end, size_t n);

/* A rule of the alternatives: a rule of one empty alternative (nothing). */
uint32_t janas_gb_empty(struct janas_gbuild *b);

/* Whether a building step ran out of memory (the grammar would be wrong). */
int janas_gbuild_failed(const struct janas_gbuild *b);

/* ---- the grammar ---- */

struct janas_grammar;

/* The grammar built so far (the builder stays usable); NULL on memory. */
struct janas_grammar *janas_grammar_make(const struct janas_gbuild *b);
void janas_grammar_free(struct janas_grammar *g);

/* ---- matching ---- */

struct janas_gmatch;

/* A match of rule `root` from its start; NULL on memory. The grammar must
   outlive it. */
struct janas_gmatch *janas_gmatch_new(const struct janas_grammar *g,
                                      uint32_t root);
void janas_gmatch_free(struct janas_gmatch *m);

/* Back to the start of the root rule. */
void janas_gmatch_reset(struct janas_gmatch *m);

/* 1 when the input so far is a whole match of the root rule (it may still
   go on, a number may take more digits). */
int janas_gmatch_accepting(const struct janas_gmatch *m);

/*
 * Whether the input could go on with these n bytes (token < 0) or with the
 * special token `token` (bytes ignored): 1 yes, 0 no. The match does not
 * change.
 */
int janas_gmatch_try(struct janas_gmatch *m, const char *bytes, size_t n,
                     int32_t token);

/* The same, and the match goes on with them: 0, or -1 when they do not fit
   (the match does not change) or memory ran out. */
int janas_gmatch_feed(struct janas_gmatch *m, const char *bytes, size_t n,
                      int32_t token);

#endif
