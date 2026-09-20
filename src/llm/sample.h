/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sample.h - choosing a token from a row of logits.
 *
 * The plain draw: temperature, then top-k, top-p and min-p on the sorted
 * probabilities, then one uniform number from the sampler's own sequence.
 * Around it, all off unless asked for: a bias per token (logit_bias), the
 * presence and frequency penalties over the tokens picked since the last
 * mark, a filter that says which tokens may come next (a grammar), and the
 * log-probability of the token picked with the most likely alternatives.
 *
 * The random sequence advances by one number per draw whatever is on, so a
 * reply with the same seed and the same settings is the same text.
 */
#ifndef JANAS_LLM_SAMPLE_H
#define JANAS_LLM_SAMPLE_H

#include <stdint.h>

#define JANAS_SAMPLE_MAX_TOP 20 /* alternatives reported at most */

struct janas_sample_params {
    float temperature; /* 0: the most likely token */
    int32_t top_k;     /* 0: no limit */
    float top_p;       /* 0 or 1: no limit */
    float min_p;
    float presence, frequency; /* penalties, OpenAI's meaning */
    int32_t logprobs; /* alternatives to report, 0..20; -1: none at all */
};

/*
 * Which tokens may come next. allows(ctx, t) is asked only while
 * active(ctx) says so; accept(ctx, t) is told every token picked, whether
 * or not the filter was active, since the tokens themselves may switch it
 * on (a tool call begins) or off (the reasoning ends).
 */
struct janas_sample_filter {
    void *ctx;
    int (*active)(void *ctx);
    int (*allows)(void *ctx, int32_t token);
    void (*accept)(void *ctx, int32_t token);
};

/* What was learned of one picked token. */
struct janas_token_lp {
    float logprob; /* of the token picked, in the model's own distribution */
    int32_t n_top;
    int32_t top_id[JANAS_SAMPLE_MAX_TOP];
    float top_lp[JANAS_SAMPLE_MAX_TOP];
};

struct janas_sampler;

struct janas_sampler *janas_sampler_create(uint32_t n_vocab);
void janas_sampler_destroy(struct janas_sampler *s);

/* New settings; the random sequence restarts from seed. */
void janas_sampler_set(struct janas_sampler *s,
                       const struct janas_sample_params *p, uint64_t seed);
/* A bias added to these tokens' logits (replaces the previous ones; n = 0
   clears them). Returns 0, or -1 (an id out of the vocabulary, memory). */
int janas_sampler_bias(struct janas_sampler *s, const int32_t *ids,
                       const float *bias, uint32_t n);
/* The filter (NULL: none); the sampler keeps the pointer's contents. */
void janas_sampler_filter(struct janas_sampler *s,
                          const struct janas_sample_filter *f);
/* The penalties count the tokens picked from here on. */
void janas_sampler_mark(struct janas_sampler *s);

/*
 * The token after this row of logits (not modified). Returns -1 when the
 * filter allows no token at all. lp, when not NULL and logprobs are on,
 * gets what was learned of it.
 */
int32_t janas_sampler_pick(struct janas_sampler *s, const float *logits,
                           struct janas_token_lp *lp);

/* Draws that had one candidate left, since the sampler was created. */
uint32_t janas_sampler_forced(const struct janas_sampler *s);

#endif
