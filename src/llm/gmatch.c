/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gmatch.c - matching a grammar, a byte or a special token at a time (see
 * grammar.h).
 *
 * A stack is a chain of nodes: the node on top is the element to match next,
 * its parent is where to go on when the rule it is in has ended. The nodes
 * are made once each (a hash table finds an equal one), in chunks that are
 * freed in the order opposite to making them: trying a token makes nodes, and
 * forgetting the try unmakes them.
 */
#include <stdlib.h>
#include <string.h>

#include "grammar.h"
#include "grammar_impl.h"

#define MAX_SET 1024         /* ways kept at once */
#define MAX_NODES (4u << 20) /* nodes a match may make */
#define MAX_DEPTH 512        /* rules entered without reading a byte */
#define CHUNK 4096
#define BUCKETS (1u << 16)

struct gnode {
    uint32_t pos, h;
    const struct gnode *parent;
    struct gnode *hnext;
};

/* The top of a stack that has matched the root rule whole. */
static const struct gnode accept_node;
#define ACCEPT (&accept_node)

struct chunk {
    struct chunk *prev;
    uint32_t used;
    struct gnode n[CHUNK];
};

struct mark {
    struct chunk *c;
    uint32_t used;
};

struct janas_gmatch {
    const struct janas_grammar *g;
    uint32_t root;
    struct gnode **bucket;
    struct chunk *top;
    size_t n_nodes;
    /* the ways the input so far can go on, and two sets to step with */
    const struct gnode **cur, **a, **b;
    uint32_t n_cur;
    int failed; /* memory, or too many ways: the last step is not to be
                   trusted */
};

static uint32_t hash(uint32_t pos, const struct gnode *parent)
{
    uint64_t x = (uint64_t)pos * 0x9e3779b97f4a7c15ull ^
                 (uint64_t)(uintptr_t)parent * 0xbf58476d1ce4e5b9ull;
    return (uint32_t)(x >> 32);
}

static const struct gnode *mk(struct janas_gmatch *m, uint32_t pos,
                              const struct gnode *parent)
{
    uint32_t h = hash(pos, parent);
    for (struct gnode *n = m->bucket[h & (BUCKETS - 1)]; n; n = n->hnext)
        if (n->pos == pos && n->parent == parent)
            return n;
    if (m->n_nodes >= MAX_NODES) {
        m->failed = 1;
        return NULL;
    }
    if (!m->top || m->top->used == CHUNK) {
        struct chunk *c = malloc(sizeof(*c));
        if (!c) {
            m->failed = 1;
            return NULL;
        }
        c->prev = m->top;
        c->used = 0;
        m->top = c;
    }
    struct gnode *n = &m->top->n[m->top->used++];
    *n = (struct gnode){.pos = pos, .h = h, .parent = parent};
    n->hnext = m->bucket[h & (BUCKETS - 1)];
    m->bucket[h & (BUCKETS - 1)] = n;
    m->n_nodes++;
    return n;
}

static struct mark mark(const struct janas_gmatch *m)
{
    return (struct mark){m->top, m->top ? m->top->used : 0};
}

/* Unmakes every node made since k, newest first: each is still the head of
   its bucket's chain when its turn comes. */
static void undo(struct janas_gmatch *m, struct mark k)
{
    while (m->top && (m->top != k.c || m->top->used > k.used)) {
        if (m->top->used == 0) {
            struct chunk *prev = m->top->prev;
            free(m->top);
            m->top = prev;
            continue;
        }
        struct gnode *n = &m->top->n[--m->top->used];
        m->bucket[n->h & (BUCKETS - 1)] = n->hnext;
        m->n_nodes--;
    }
}

static void add(struct janas_gmatch *m, const struct gnode **set, uint32_t *n,
                const struct gnode *x)
{
    for (uint32_t i = 0; i < *n; i++)
        if (set[i] == x)
            return;
    if (*n == MAX_SET) {
        m->failed = 1;
        return;
    }
    set[(*n)++] = x;
}

/* Every way node x can reach an element that reads input (or the end of the
   root rule), added to set. */
static void closure(struct janas_gmatch *m, const struct gnode *x,
                    const struct gnode **set, uint32_t *n, int depth)
{
    const struct janas_grammar *g = m->g;
    while (x) {
        if (depth > MAX_DEPTH) {
            m->failed = 1;
            return;
        }
        const struct gel *e = &g->el[x->pos];
        if (e->kind == GEL_END) {
            if (!x->parent) {
                add(m, set, n, ACCEPT);
                return;
            }
            x = x->parent; /* its position is the element after the rule */
            continue;
        }
        if (e->kind != GEL_RULE) {
            add(m, set, n, x);
            return;
        }
        /* where to go on after the rule; nowhere new when the rule is the
           last thing of its alternative (so right recursion stays flat) */
        const struct gnode *cont = x->parent;
        if (g->el[x->pos + 1].kind != GEL_END &&
            !(cont = mk(m, x->pos + 1, x->parent)))
            return;
        uint32_t r = e->arg;
        if (r >= g->n_rules)
            return;
        for (uint32_t i = 0; i < g->rule_n[r]; i++) {
            const struct gnode *y =
                mk(m, g->alt_pos[g->rule_first[r] + i], cont);
            if (!y)
                return;
            closure(m, y, set, n, depth + 1);
        }
        return;
    }
}

/* One step: the ways of `in` that read this byte (or token), into out. */
static uint32_t step(struct janas_gmatch *m, const struct gnode **in,
                     uint32_t n_in, uint8_t c, int32_t token,
                     const struct gnode **out)
{
    const struct janas_grammar *g = m->g;
    uint32_t n = 0;
    for (uint32_t i = 0; i < n_in; i++) {
        const struct gnode *x = in[i];
        if (x == ACCEPT)
            continue;
        const struct gel *e = &g->el[x->pos];
        int ok = token >= 0
                     ? e->kind == GEL_TOKEN && (int32_t)e->arg == token
                     : e->kind == GEL_SET && gset_has(g->sets[e->arg], c);
        if (!ok)
            continue;
        const struct gnode *y = mk(m, x->pos + 1, x->parent);
        if (y)
            closure(m, y, out, &n, 0);
    }
    return n;
}

/* Runs the input over the current set; the result in *res (one of a, b, or
   cur itself for no input). Returns how many ways are left. */
static uint32_t run(struct janas_gmatch *m, const char *bytes, size_t len,
                    int32_t token, const struct gnode ***res)
{
    const struct gnode **in = m->cur;
    uint32_t n = m->n_cur;
    if (token >= 0) {
        n = step(m, in, n, 0, token, m->a);
        in = m->a;
    } else {
        for (size_t i = 0; i < len && n; i++) {
            const struct gnode **out = in == m->a ? m->b : m->a;
            n = step(m, in, n, (uint8_t)bytes[i], -1, out);
            in = out;
        }
    }
    *res = in;
    return n;
}

void janas_gmatch_reset(struct janas_gmatch *m)
{
    undo(m, (struct mark){NULL, 0});
    m->failed = 0;
    m->n_cur = 0;
    const struct janas_grammar *g = m->g;
    if (m->root >= g->n_rules)
        return;
    for (uint32_t i = 0; i < g->rule_n[m->root]; i++) {
        const struct gnode *y =
            mk(m, g->alt_pos[g->rule_first[m->root] + i], NULL);
        if (y)
            closure(m, y, m->cur, &m->n_cur, 0);
    }
}

struct janas_gmatch *janas_gmatch_new(const struct janas_grammar *g,
                                      uint32_t root)
{
    struct janas_gmatch *m = calloc(1, sizeof(*m));
    if (!m)
        return NULL;
    m->g = g;
    m->root = root;
    m->bucket = calloc(BUCKETS, sizeof(*m->bucket));
    m->cur = malloc(MAX_SET * sizeof(*m->cur));
    m->a = malloc(MAX_SET * sizeof(*m->a));
    m->b = malloc(MAX_SET * sizeof(*m->b));
    if (!m->bucket || !m->cur || !m->a || !m->b) {
        janas_gmatch_free(m);
        return NULL;
    }
    janas_gmatch_reset(m);
    return m;
}

void janas_gmatch_free(struct janas_gmatch *m)
{
    if (!m)
        return;
    if (m->bucket)
        undo(m, (struct mark){NULL, 0});
    free(m->bucket);
    free(m->cur);
    free(m->a);
    free(m->b);
    free(m);
}

int janas_gmatch_accepting(const struct janas_gmatch *m)
{
    for (uint32_t i = 0; i < m->n_cur; i++)
        if (m->cur[i] == ACCEPT)
            return 1;
    return 0;
}

int janas_gmatch_try(struct janas_gmatch *m, const char *bytes, size_t n,
                     int32_t token)
{
    if (token < 0 && n == 0)
        return 0; /* a token of no bytes would leave the grammar stuck */
    struct mark k = mark(m);
    int failed = m->failed;
    const struct gnode **res;
    uint32_t left = run(m, bytes, n, token, &res);
    undo(m, k);
    m->failed = failed;
    return left > 0;
}

int janas_gmatch_feed(struct janas_gmatch *m, const char *bytes, size_t n,
                      int32_t token)
{
    if (token < 0 && n == 0)
        return -1;
    struct mark k = mark(m);
    const struct gnode **res;
    uint32_t left = run(m, bytes, n, token, &res);
    if (left == 0) {
        undo(m, k);
        return -1;
    }
    memcpy(m->cur, res, left * sizeof(*res));
    m->n_cur = left;
    return 0;
}
