/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * constrain.c - what a reply may say, token by token (see constrain.h).
 */
#include "constrain.h"

#include <stdlib.h>
#include <string.h>

enum part {
    PART_THINK,  /* the reasoning: free */
    PART_ANSWER, /* the answer: free, or held to spec.answer */
    PART_CALL,   /* a call opened in a free answer: held to spec.call */
    PART_DONE,   /* the held answer is complete and a stop token came */
};

struct janas_constraint {
    struct janas_constraint_spec spec;
    const struct janas_tokenizer *tok;
    uint8_t *special; /* a bit per token */
    int32_t *stop;
    enum part part;
    struct janas_gmatch *answer, *call;
    int broken; /* a grammar gave out (memory): let everything through */
};

struct janas_constraint *
janas_constraint_new(const struct janas_constraint_spec *spec,
                     const struct janas_tokenizer *tok)
{
    struct janas_constraint *c = calloc(1, sizeof(*c));
    if (!c)
        return NULL;
    c->spec = *spec;
    c->tok = tok;
    uint32_t nv = janas_tokenizer_n_vocab(tok);
    c->special = calloc((nv + 7) / 8, 1);
    c->stop = malloc(((size_t)spec->n_stop + 1) * sizeof(int32_t));
    if (!c->special || !c->stop) {
        janas_constraint_free(c);
        return NULL;
    }
    if (spec->n_stop)
        memcpy(c->stop, spec->stop, (size_t)spec->n_stop * sizeof(int32_t));
    c->spec.stop = c->stop;
    const int32_t *ids;
    uint32_t ns = janas_tokenizer_specials(tok, &ids);
    for (uint32_t i = 0; i < ns; i++)
        c->special[ids[i] >> 3] |= (uint8_t)(1u << (ids[i] & 7));
    if (spec->g && spec->answer != JANAS_NO_RULE &&
        !(c->answer = janas_gmatch_new(spec->g, spec->answer))) {
        janas_constraint_free(c);
        return NULL;
    }
    if (spec->g && spec->call != JANAS_NO_RULE &&
        !(c->call = janas_gmatch_new(spec->g, spec->call))) {
        janas_constraint_free(c);
        return NULL;
    }
    c->part =
        spec->thinking && spec->think_close >= 0 ? PART_THINK : PART_ANSWER;
    return c;
}

void janas_constraint_free(struct janas_constraint *c)
{
    if (!c)
        return;
    janas_gmatch_free(c->answer);
    janas_gmatch_free(c->call);
    free(c->special);
    free(c->stop);
    free(c);
}

static int is_stop(const struct janas_constraint *c, int32_t t)
{
    for (int i = 0; i < c->spec.n_stop; i++)
        if (c->stop[i] == t)
            return 1;
    return 0;
}

static int is_special(const struct janas_constraint *c, int32_t t)
{
    return c->special[t >> 3] >> (t & 7) & 1;
}

/* Whether token t may come next in match m. */
static int fits(struct janas_constraint *c, struct janas_gmatch *m, int32_t t)
{
    if (is_stop(c, t))
        return janas_gmatch_accepting(m);
    if (is_special(c, t))
        return janas_gmatch_try(m, NULL, 0, t);
    char buf[256];
    size_t n = janas_tokenizer_decode(c->tok, t, buf, sizeof(buf));
    return n <= sizeof(buf) && janas_gmatch_try(m, buf, n, -1);
}

/* Match m goes on with token t (a stop token only ends it). */
static void feed(struct janas_constraint *c, struct janas_gmatch *m, int32_t t)
{
    if (is_stop(c, t))
        return;
    int rc;
    if (is_special(c, t)) {
        rc = janas_gmatch_feed(m, NULL, 0, t);
    } else {
        char buf[256];
        size_t n = janas_tokenizer_decode(c->tok, t, buf, sizeof(buf));
        rc = n <= sizeof(buf) ? janas_gmatch_feed(m, buf, n, -1) : -1;
    }
    if (rc != 0)
        c->broken = 1; /* a token the filter did not allow: stop holding */
}

int janas_constraint_held(const struct janas_constraint *c)
{
    if (c->broken)
        return 0;
    return (c->part == PART_ANSWER && c->answer) || c->part == PART_CALL;
}

static int active(void *ctx)
{
    struct janas_constraint *c = ctx;
    return janas_constraint_held(c) ||
           (c->part == PART_ANSWER && c->spec.ban_calls);
}

static int allows(void *ctx, int32_t t)
{
    struct janas_constraint *c = ctx;
    if (c->broken)
        return 1;
    switch (c->part) {
    case PART_ANSWER:
        if (c->answer)
            return fits(c, c->answer, t);
        return !(c->spec.ban_calls && t == c->spec.call_open);
    case PART_CALL:
        return fits(c, c->call, t);
    default:
        return 1;
    }
}

static void accept(void *ctx, int32_t t)
{
    struct janas_constraint *c = ctx;
    if (c->broken)
        return;
    switch (c->part) {
    case PART_THINK:
        if (t == c->spec.think_close)
            c->part = PART_ANSWER;
        break;
    case PART_ANSWER:
        if (c->answer) {
            if (is_stop(c, t))
                c->part = PART_DONE;
            else
                feed(c, c->answer, t);
        } else if (c->call && t == c->spec.call_open) {
            janas_gmatch_reset(c->call);
            c->part = PART_CALL;
        }
        break;
    case PART_CALL:
        feed(c, c->call, t);
        if (!c->broken && janas_gmatch_accepting(c->call))
            c->part = PART_ANSWER; /* the call is closed */
        break;
    case PART_DONE:
        break;
    }
}

void janas_constraint_filter(struct janas_constraint *c,
                             struct janas_sample_filter *f)
{
    *f = (struct janas_sample_filter){
        .ctx = c, .active = active, .allows = allows, .accept = accept};
}
