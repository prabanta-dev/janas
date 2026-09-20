/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * grammar_impl.h - the layout of a grammar, shared by the builder
 * (grammar.c) and the matcher (gmatch.c). Private.
 */
#ifndef JANAS_LLM_GRAMMAR_IMPL_H
#define JANAS_LLM_GRAMMAR_IMPL_H

#include <stdint.h>

enum gel_kind {
    GEL_END,   /* the end of an alternative */
    GEL_SET,   /* arg: a byte set */
    GEL_TOKEN, /* arg: a special token */
    GEL_RULE,  /* arg: a rule */
};

struct gel {
    uint32_t kind;
    uint32_t arg;
};

/*
 * el: every alternative's elements, each followed by GEL_END. Rule r has
 * rule_n[r] alternatives, starting at el[alt_pos[rule_first[r] + i]].
 */
struct janas_grammar {
    struct gel *el;
    uint32_t n_el;
    uint32_t *alt_pos;
    uint32_t *rule_first, *rule_n;
    uint32_t n_rules;
    uint8_t (*sets)[32];
    uint32_t n_sets;
};

static inline int gset_has(const uint8_t *set, uint8_t c)
{
    return set[c >> 3] >> (c & 7) & 1;
}

#endif
