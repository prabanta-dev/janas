/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * generate.h - token generation with optional speculative decoding.
 *
 * Speculation proposes k tokens cheaply and verifies them in one forward
 * pass over the block; the output is exactly the same as without it (block
 * logits are bit-identical to token-by-token ones, and sampling spends one
 * random number per token), so it is purely a speed setting.
 */
#ifndef JANAS_LLM_GENERATE_H
#define JANAS_LLM_GENERATE_H

#include <stdint.h>
#include <stdio.h>

#include "model.h"
#include "sample.h"

enum janas_spec_mode {
    JANAS_SPEC_OFF = 0,
    JANAS_SPEC_LOOKUP = 1, /* drafts copied from the context (n-gram match) */
    JANAS_SPEC_MTP = 2,    /* drafts from the model's MTP block, chained */
    JANAS_SPEC_DRAFT = 3,  /* drafts from a second, smaller model */
};

#define JANAS_GEN_MAX_STOP 4

struct janas_gen_options {
    int spec_mode; /* enum janas_spec_mode */
    /*
     * The drafting model for JANAS_SPEC_DRAFT: a small model of the same
     * family, sharing the target's vocabulary and its compute pool (see
     * shared_compute in janas_llm_options - two pools take the cores from
     * each other). The caller loads and frees it.
     */
    struct janas_llm_model *draft;
    int spec_k;    /* tokens proposed per verification, 1..16 */
    int spec_auto; /* lookup: stop speculating while it does not pay, retry
                      later; MTP: plan the drafts of each pass from measured
                      costs and calibrated odds */
    int32_t eos;   /* stop at this token, -1 for none */
    float spec_min_conf; /* MTP: draft while the product of the drafts'
                            probabilities stays above this (0: always) */
    /* sampling; all zero: greedy */
    float temperature; /* 0: greedy (the most likely token) */
    int32_t top_k;     /* keep the k most likely tokens; 0: no limit */
    float top_p;       /* keep the most likely tokens up to this mass; 0: 1 */
    float min_p;       /* drop tokens below min_p x the top probability */
    uint64_t seed;
    /* OpenAI's penalties, over the tokens generated since the last
       janas_llm_session_mark */
    float presence_penalty, frequency_penalty;
    /* the log-probability of every generated token, and that many of the
       most likely alternatives (janas_llm_session_logprob) */
    int logprobs, top_logprobs;
    /* more stop tokens, besides eos (a speculative pass never runs past a
       stop token) */
    int32_t stop[JANAS_GEN_MAX_STOP];
    int n_stop;
};

struct janas_gen_stats {
    uint32_t prompt_tokens, new_tokens;
    uint32_t passes;        /* forward passes after the prefill */
    double draft_seconds;   /* time inside the drafting model */
    double sample_seconds;  /* time choosing the tokens */
    uint32_t forced_tokens; /* drawn where only one candidate was left */
    uint32_t spec_passes;   /* of which verifying a draft */
    uint32_t drafted, accepted;
    uint32_t auto_off_passes; /* passes run without drafting by spec_auto */
    double prefill_seconds, decode_seconds;
    /* the planner's calibration: of the drafts whose guessed chance of
       surviving fell in tenth b, how many were tried and how many survived.
       A bin the planner never finds worth trying stays empty, so this says
       what it has actually learned. */
    uint32_t cal_tried[10], cal_won[10];
};

/*
 * A session: one token sequence that grows by appending tokens (a prompt, a
 * user's turn) and by generating them, token by token on request. Every
 * token is computed once: a new turn continues from the position reached,
 * with the KV cache and the recurrent state kept. The model holds the state
 * of one sequence, so a model serves one session at a time.
 *
 * Sampling draws one random number per generated token, in order, so with
 * the same seed the text is the same with or without speculation (block
 * logits are bit-identical to token-by-token ones).
 */
struct janas_llm_session;

struct janas_llm_session *
janas_llm_session_create(struct janas_llm_model *m,
                         const struct janas_gen_options *o);

/*
 * Forgets drop tokens of the sequence, starting at keep: the context slides
 * instead of ending when it is full. What is kept behaves as if it had
 * always been there (see janas_llm_model_shift); the model's recurrent
 * state, which has seen everything, is not touched. Returns 0 or -1.
 */
int janas_llm_session_slide(struct janas_llm_session *s, uint32_t keep,
                            uint32_t drop);
void janas_llm_session_destroy(struct janas_llm_session *s);

/* New sampling and speculation settings, from the next pass on. The seed
   restarts the random sequence. Returns 0 or -1 (invalid options). */
int janas_llm_session_set_options(struct janas_llm_session *s,
                                  const struct janas_gen_options *o);

/*
 * Appends tokens to the sequence. Generated tokens not yet returned by
 * janas_llm_session_next are dropped first. Returns 0, or -1 when the
 * context cannot hold them.
 */
int janas_llm_session_append(struct janas_llm_session *s, const int32_t *tokens,
                             uint32_t n);

/* Runs the appended tokens through the model now (janas_llm_session_next
   does it anyway): all but the last, which the next pass takes. */
int janas_llm_session_prefill(struct janas_llm_session *s);
/* The same a block at a time (janas_llm_model_max_block tokens): 1 when
   more is left, 0 when it is all done, -1 on error. A caller that reads a
   long prompt this way can say how far it has gone between the blocks. */
int janas_llm_session_prefill_step(struct janas_llm_session *s);

/*
 * The next token after the sequence, appended to it. Returns 0, 1 when the
 * context is full, -1 on error (or an empty sequence).
 */
int janas_llm_session_next(struct janas_llm_session *s, int32_t *token);

/* Tokens of the sequence the model has computed. */
uint32_t janas_llm_session_computed(const struct janas_llm_session *s);

/* Tokens in the sequence (appended and returned). */
uint32_t janas_llm_session_length(const struct janas_llm_session *s);

/* The tokens of the sequence (*tokens, valid until the next call that
   changes the session); returns how many. */
uint32_t janas_llm_session_tokens(const struct janas_llm_session *s,
                                  const int32_t **tokens);

/*
 * Keeps only the first n tokens of the sequence, so that what follows can be
 * appended in place of the rest: the positions after them are computed again
 * when needed. A model with a recurrent state (qwen3next, qwen35moe) cannot
 * go back over what it has computed: there it succeeds only when nothing
 * computed is dropped. Returns 0, or -1 when it cannot.
 */
int janas_llm_session_truncate(struct janas_llm_session *s, uint32_t n);

/*
 * Sampling beyond the options: a bias per token (n = 0 clears it; -1 for an
 * id out of the vocabulary), a filter of the tokens that may come next
 * (NULL: none; see sample.h), and the mark from which the penalties count.
 */
int janas_llm_session_bias(struct janas_llm_session *s, const int32_t *ids,
                           const float *bias, uint32_t n);
void janas_llm_session_filter(struct janas_llm_session *s,
                              const struct janas_sample_filter *f);
void janas_llm_session_mark(struct janas_llm_session *s);

/*
 * What was learned of the generated token at position pos of the sequence
 * (options.logprobs): 0, or -1 when there is nothing for it - logprobs off,
 * a token that was appended rather than generated, or one returned too long
 * ago (the last few dozen are kept).
 */
int janas_llm_session_logprob(const struct janas_llm_session *s, uint32_t pos,
                              struct janas_token_lp *lp);

/* Back to an empty sequence. */
void janas_llm_session_reset(struct janas_llm_session *s);

/*
 * A copy of the sequence and of what the model has computed of it, so that
 * the model can serve another sequence and this one come back later without
 * being read again (see janas_llm_model_state_save). Saved: the tokens
 * appended or returned, all computed but the last, as after
 * janas_llm_session_truncate. NULL when there is nothing computed, when out
 * of memory, or when a model with a recurrent state cannot give its state at
 * that position. Restore puts it in place of the session's sequence (0, or
 * -1 for a copy from a model of another shape); the copy stays valid and
 * belongs to the caller. The drafting model's state goes with it; if it
 * could not be kept, that model reads the sequence again.
 */
struct janas_llm_saved;
struct janas_llm_saved *janas_llm_session_save(struct janas_llm_session *s);
int janas_llm_session_restore(struct janas_llm_session *s,
                              const struct janas_llm_saved *sv);
/* Its tokens (returns how many) and how many of them are computed. */
uint32_t janas_llm_saved_tokens(const struct janas_llm_saved *sv,
                                const int32_t **tokens, uint32_t *n_kv);
size_t janas_llm_saved_bytes(const struct janas_llm_saved *sv);
void janas_llm_saved_free(struct janas_llm_saved *sv);
/* A copy to a file and back, for the session's models (NULL when the file
   is not one, is short, or is of another model's shape). With tokens_only
   only the tokens are read: what is needed to choose among many copies;
   such a copy cannot be restored. */
int janas_llm_saved_write(const struct janas_llm_session *s,
                          const struct janas_llm_saved *sv, FILE *f);
struct janas_llm_saved *janas_llm_saved_read(const struct janas_llm_session *s,
                                             FILE *f, int tokens_only);

/* Counters since the session was created. */
void janas_llm_session_stats(const struct janas_llm_session *s,
                             struct janas_gen_stats *st);

/*
 * Runs the prompt, then appends up to max_new tokens to out (stopping after
 * a stop token). Returns the number of tokens generated, or -1.
 */
int janas_llm_generate(struct janas_llm_model *m, const int32_t *prompt,
                       uint32_t n_prompt, uint32_t max_new,
                       const struct janas_gen_options *o, int32_t *out,
                       struct janas_gen_stats *st);

#endif
