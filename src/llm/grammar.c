/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * grammar.c - building grammars (see grammar.h); matching is in gmatch.c.
 */
#include "grammar.h"

#include <stdlib.h>
#include <string.h>

#include "grammar_impl.h"

struct balt {
    uint32_t rule;
    struct gel *el;
    uint32_t n, cap;
};

struct janas_gbuild {
    struct balt *alts;
    uint32_t n_alts, cap_alts;
    uint32_t n_rules;
    uint8_t (*sets)[32];
    uint32_t n_sets, cap_sets;
    uint32_t byte_set[256]; /* the set of one byte, + 1; 0: not made yet */
    int failed;
};

struct janas_gbuild *janas_gbuild_new(void)
{
    return calloc(1, sizeof(struct janas_gbuild));
}

void janas_gbuild_free(struct janas_gbuild *b)
{
    if (!b)
        return;
    for (uint32_t i = 0; i < b->n_alts; i++)
        free(b->alts[i].el);
    free(b->alts);
    free(b->sets);
    free(b);
}

int janas_gbuild_failed(const struct janas_gbuild *b)
{
    return b->failed;
}

uint32_t janas_gb_rule(struct janas_gbuild *b)
{
    return b->n_rules++;
}

uint32_t janas_gb_alt(struct janas_gbuild *b, uint32_t r)
{
    if (b->n_alts == b->cap_alts) {
        uint32_t cap = b->cap_alts ? 2 * b->cap_alts : 64;
        struct balt *t = realloc(b->alts, cap * sizeof(*t));
        if (!t) {
            b->failed = 1;
            return UINT32_MAX;
        }
        b->alts = t;
        b->cap_alts = cap;
    }
    b->alts[b->n_alts] = (struct balt){.rule = r};
    return b->n_alts++;
}

static void push(struct janas_gbuild *b, uint32_t a, uint32_t kind,
                 uint32_t arg)
{
    if (a >= b->n_alts) {
        b->failed = 1;
        return;
    }
    struct balt *x = &b->alts[a];
    if (x->n == x->cap) {
        uint32_t cap = x->cap ? 2 * x->cap : 8;
        struct gel *t = realloc(x->el, cap * sizeof(*t));
        if (!t) {
            b->failed = 1;
            return;
        }
        x->el = t;
        x->cap = cap;
    }
    x->el[x->n++] = (struct gel){.kind = kind, .arg = arg};
}

static uint32_t add_set(struct janas_gbuild *b, const uint8_t set[32])
{
    if (b->n_sets == b->cap_sets) {
        uint32_t cap = b->cap_sets ? 2 * b->cap_sets : 64;
        uint8_t(*t)[32] = realloc(b->sets, cap * sizeof(*t));
        if (!t) {
            b->failed = 1;
            return 0;
        }
        b->sets = t;
        b->cap_sets = cap;
    }
    memcpy(b->sets[b->n_sets], set, 32);
    return b->n_sets++;
}

void janas_gb_set(struct janas_gbuild *b, uint32_t a, const uint8_t set[32])
{
    uint32_t s = add_set(b, set);
    if (!b->failed)
        push(b, a, GEL_SET, s);
}

void janas_gb_byte(struct janas_gbuild *b, uint32_t a, uint8_t c)
{
    if (!b->byte_set[c]) {
        uint8_t set[32];
        janas_set_clear(set);
        janas_set_add(set, c, c);
        uint32_t s = add_set(b, set);
        if (b->failed)
            return;
        b->byte_set[c] = s + 1;
    }
    push(b, a, GEL_SET, b->byte_set[c] - 1);
}

void janas_gb_lit(struct janas_gbuild *b, uint32_t a, const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++)
        janas_gb_byte(b, a, (uint8_t)s[i]);
}

void janas_gb_token(struct janas_gbuild *b, uint32_t a, int32_t token)
{
    push(b, a, GEL_TOKEN, (uint32_t)token);
}

void janas_gb_ref(struct janas_gbuild *b, uint32_t a, uint32_t r)
{
    push(b, a, GEL_RULE, r);
}

uint32_t janas_gb_empty(struct janas_gbuild *b)
{
    uint32_t r = janas_gb_rule(b);
    janas_gb_alt(b, r);
    return r;
}

void janas_set_clear(uint8_t set[32])
{
    memset(set, 0, 32);
}

void janas_set_add(uint8_t set[32], uint8_t lo, uint8_t hi)
{
    for (unsigned c = lo; c <= hi; c++)
        set[c >> 3] |= (uint8_t)(1u << (c & 7));
}

void janas_set_invert(uint8_t set[32])
{
    for (int i = 0; i < 32; i++)
        set[i] = (uint8_t)~set[i];
}

/*
 * Text without `end` in it, then `end`: the automaton that finds a word in a
 * text (Knuth, Morris and Pratt), one rule per state - how many bytes of end
 * the text read so far ends with. Every byte leads to one state; the bytes
 * that lead to the same state make one set, so a state has as many
 * alternatives as it has distinct destinations, and each alternative ends
 * in its destination, which keeps the stack from growing.
 */
uint32_t janas_gb_until(struct janas_gbuild *b, const char *end, size_t n)
{
    if (n == 0)
        return janas_gb_empty(b);
    uint32_t *fail = malloc(n * sizeof(uint32_t));
    uint32_t *rule = malloc(n * sizeof(uint32_t));
    if (!fail || !rule) {
        free(fail);
        free(rule);
        b->failed = 1;
        return 0;
    }
    fail[0] = 0;
    for (size_t i = 1, k = 0; i < n; i++) {
        while (k && end[i] != end[k])
            k = fail[k - 1];
        if (end[i] == end[k])
            k++;
        fail[i] = (uint32_t)k;
    }
    for (size_t i = 0; i < n; i++)
        rule[i] = janas_gb_rule(b);
    for (size_t i = 0; i < n; i++) {
        /* where each byte leads from state i: n means end is complete */
        uint32_t dest[256];
        for (unsigned c = 0; c < 256; c++) {
            size_t k = i;
            while (k && (uint8_t)end[k] != c)
                k = fail[k - 1];
            dest[c] = (uint8_t)end[k] == c ? (uint32_t)k + 1 : 0;
        }
        uint8_t done[256] = {0};
        for (unsigned c = 0; c < 256; c++) {
            if (done[c])
                continue;
            uint8_t set[32];
            janas_set_clear(set);
            for (unsigned d = c; d < 256; d++)
                if (!done[d] && dest[d] == dest[c]) {
                    janas_set_add(set, (uint8_t)d, (uint8_t)d);
                    done[d] = 1;
                }
            uint32_t a = janas_gb_alt(b, rule[i]);
            janas_gb_set(b, a, set);
            if (dest[c] < n)
                janas_gb_ref(b, a, rule[dest[c]]);
        }
    }
    uint32_t r = rule[0];
    free(fail);
    free(rule);
    return r;
}

struct janas_grammar *janas_grammar_make(const struct janas_gbuild *b)
{
    if (b->failed)
        return NULL;
    struct janas_grammar *g = calloc(1, sizeof(*g));
    if (!g)
        return NULL;
    size_t n_el = 0;
    for (uint32_t i = 0; i < b->n_alts; i++)
        n_el += b->alts[i].n + 1;
    uint32_t nr = b->n_rules ? b->n_rules : 1;
    g->el = malloc((n_el + 1) * sizeof(struct gel));
    g->alt_pos = malloc((b->n_alts + 1) * sizeof(uint32_t));
    g->rule_first = calloc(nr, sizeof(uint32_t));
    g->rule_n = calloc(nr, sizeof(uint32_t));
    g->sets = malloc((b->n_sets + 1) * 32);
    if (!g->el || !g->alt_pos || !g->rule_first || !g->rule_n || !g->sets) {
        janas_grammar_free(g);
        return NULL;
    }
    g->n_rules = b->n_rules;
    g->n_sets = b->n_sets;
    memcpy(g->sets, b->sets, (size_t)b->n_sets * 32);
    /* alternatives grouped by rule, in the order they were made */
    for (uint32_t i = 0; i < b->n_alts; i++)
        if (b->alts[i].rule < b->n_rules)
            g->rule_n[b->alts[i].rule]++;
    uint32_t at = 0;
    for (uint32_t r = 0; r < b->n_rules; r++) {
        g->rule_first[r] = at;
        at += g->rule_n[r];
        g->rule_n[r] = 0;
    }
    uint32_t w = 0;
    for (uint32_t i = 0; i < b->n_alts; i++) {
        const struct balt *x = &b->alts[i];
        if (x->rule >= b->n_rules)
            continue;
        g->alt_pos[g->rule_first[x->rule] + g->rule_n[x->rule]++] = w;
        if (x->n)
            memcpy(g->el + w, x->el, x->n * sizeof(struct gel));
        w += x->n;
        g->el[w++] = (struct gel){.kind = GEL_END};
    }
    g->n_el = w;
    return g;
}

void janas_grammar_free(struct janas_grammar *g)
{
    if (!g)
        return;
    free(g->el);
    free(g->alt_pos);
    free(g->rule_first);
    free(g->rule_n);
    free(g->sets);
    free(g);
}
