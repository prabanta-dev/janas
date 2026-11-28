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
 * And llama.cpp's chain (struct janas_sample_more): the repetition
 * penalty, DRY, top-n-sigma, typical, XTC, mirostat, the dynamic
 * temperature, in the order asked; with none of it the plain draw is the
 * one it always was, to the bit.
 *
 * The random sequence advances by one number per draw whatever is on, so a
 * reply with the same seed and the same settings is the same text.
 */
#ifndef JANAS_LLM_SAMPLE_H
#define JANAS_LLM_SAMPLE_H

#include <stdint.h>

#define JANAS_SAMPLE_MAX_TOP 20 /* alternatives reported at most */

/* The stages of the chain, as llama.cpp names them (--samplers):
   penalties and dry change the whole row first, wherever they stand. */
enum janas_sample_stage {
    JANAS_STAGE_PENALTIES = 1,
    JANAS_STAGE_DRY,
    JANAS_STAGE_TOP_N_SIGMA,
    JANAS_STAGE_TOP_K,
    JANAS_STAGE_TYPICAL,
    JANAS_STAGE_TOP_P,
    JANAS_STAGE_MIN_P,
    JANAS_STAGE_XTC,
    JANAS_STAGE_TEMPERATURE,
};
#define JANAS_SAMPLE_MAX_STAGES 16

/*
 * The rest of the sampling, all off when zero: what llama.cpp offers
 * beyond the plain draw, with its meanings.
 */
struct janas_sample_more {
    /* a token among the last last_n picked (0: since the mark): its logit
       divided by repeat when positive, multiplied when not (1 or 0: off);
       the window of the presence and frequency penalties too */
    float repeat;
    int32_t last_n;
    float typical_p;   /* locally typical sampling (0 or 1: off) */
    float top_n_sigma; /* logits within n standard deviations of the top */
    float xtc_probability, xtc_threshold; /* exclude the top choices */
    /* DRY: a token that would go on repeating a run of the last dry_last_n
       tokens (0: since the mark) longer than allowed loses multiplier x
       base^(length - allowed); runs do not cross the breakers' tokens */
    float dry_multiplier, dry_base;
    int32_t dry_allowed, dry_last_n;
    int32_t mirostat; /* 1 or 2, its tau and eta; 0: off */
    float mirostat_tau, mirostat_eta;
    /* dynamic temperature: temperature +- range, by the entropy^exponent */
    float dynatemp_range, dynatemp_exponent;
    /* the order (0 stages: Janas's own, top_k, temperature, top_p,
       min_p, with the others where llama.cpp puts them) */
    uint8_t stage[JANAS_SAMPLE_MAX_STAGES];
    int32_t n_stage;
};

struct janas_sample_params {
    float temperature; /* 0: the most likely token */
    int32_t top_k;     /* 0: no limit */
    float top_p;       /* 0 or 1: no limit */
    float min_p;
    float presence, frequency; /* penalties, OpenAI's meaning */
    int32_t logprobs; /* alternatives to report, 0..20; -1: none at all */
    struct janas_sample_more more;
};

/* The stages named as llama.cpp names them ("penalties;dry;top_k;..." or
   its letters "edkypmxt"), separated by ';', ',' or spaces: their count,
   or -1 for a name it does not know. */
int janas_sample_stages(const char *names, uint8_t *stage, int max);

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
/* DRY's breakers: for each token of the vocabulary, 1 when its text holds
   one of them (NULL: none); the sampler keeps the pointer. */
void janas_sampler_breakers(struct janas_sampler *s, const uint8_t *is);
/* The filter (NULL: none); the sampler keeps the pointer's contents. */
void janas_sampler_filter(struct janas_sampler *s,
                          const struct janas_sample_filter *f);
/* The random sequence n draws back, as if the last n picks had not been
   made: a speculative pass draws for drafts that a reply cut short never
   returns. The tokens the penalties and DRY count, and mirostat's state,
   go back with it. */
void janas_sampler_rewind(struct janas_sampler *s, uint32_t n);
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
