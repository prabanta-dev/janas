/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model.h - Janas-LLM forward pass over a JNS file.
 *
 * Decode, one token at a time, with the routed experts streamed through the
 * expert cache. Supported architectures: qwen3moe, qwen3next.
 */
#ifndef JANAS_LLM_MODEL_H
#define JANAS_LLM_MODEL_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "expert_cache.h"

struct janas_llm_options {
    uint64_t cache_bytes; /* RAM for routed experts */
    uint32_t n_ctx;       /* KV cache length in tokens */
    const int *compute_cpus;
    int n_compute;
    const int *io_cpus;
    int n_io;
    /*
     * A compute pool to use instead of making one. A draft model wants this:
     * two pools of twenty threads on twenty cores take the cores from each
     * other, because the pool that is idle spins before it sleeps. Measured
     * 22 Sep 2026 with a 0.6B drafting a 4B: a pass of its own cost 22.1 ms
     * beside the target's pool and 7.2 ms alone. The pool is not destroyed
     * with the model; the thread counts are still read from compute_cpus.
     */
    struct janas_pool *shared_compute;
    const char *mtp_path;     /* MTP block (tools/hf2jns_mtp), or NULL */
    uint64_t mtp_cache_bytes; /* RAM for its experts; 0: 256 MiB */
    uint32_t draft_vocab;     /* lowest token ids in the draft head; 0: 32768 */
    int use_gpu; /* integrated GPU for products over blocks, if usable
                    (environment JANAS_GPU=0/1 overrides) */
    int warm;    /* preload the experts this machine used most with this model,
                    from the profile in ~/.cache/janas (environment JANAS_WARM=0/1
                    overrides); the caller sets it, 0 means no preloading */
    int attn_scores; /* how the attention scores are computed: 0 leaves the
                        choice to the engine (see janas_attn_scores in
                        attention.h), 1 the query as it is, 2 the query to
                        sixteen bits (environment JANAS_ATTN overrides) */
    int expert_bits; /* 2, 4 or 6 bits per weight of the experts' down matrix,
                        for files that hold it in bit planes (JNS 3); 6 is the
                        whole matrix, the same weights as Q6_K. 0 leaves the
                        choice to the engine: every bit when the experts fit
                        in cache_bytes, four otherwise (environment
                        JANAS_EXPERT_BITS overrides) */
};

struct janas_llm_model;

struct janas_llm_model *janas_llm_model_load(const char *path,
                                             const struct janas_llm_options *o,
                                             char *err, size_t err_len);
void janas_llm_model_free(struct janas_llm_model *m);

uint32_t janas_llm_model_n_vocab(const struct janas_llm_model *m);
uint32_t janas_llm_model_n_embd(const struct janas_llm_model *m);
uint32_t janas_llm_model_n_ctx(const struct janas_llm_model *m);
/* The length the model was trained for (0 if its file does not say). The
   context never goes past it: without a rope scaling the file carries, text
   beyond that length falls apart. */
uint32_t janas_llm_model_n_ctx_train(const struct janas_llm_model *m);

/*
 * A place in ~/.cache/janas for something the engine learns about this
 * model on this machine ("experts", "drafts"): the name is <what> and the
 * model's fingerprint. Returns 0 or -1 if there is nowhere to put it.
 */
int janas_llm_model_cache_file(const struct janas_llm_model *m,
                               const char *what, char *path, size_t len,
                               int create);

/* Bits per weight the experts' down matrix is read with: 2, 4 or 6, and
   whether that was chosen by the engine from the memory available. */
int janas_llm_model_expert_bits(const struct janas_llm_model *m);
int janas_llm_model_expert_bits_auto(const struct janas_llm_model *m);
/* How the attention scores are computed (janas_attn_scores), and whether the
   engine chose it. */
int janas_llm_model_attn_scores(const struct janas_llm_model *m);
int janas_llm_model_attn_scores_auto(const struct janas_llm_model *m);

/* Whether the engine also chose how many experts a token uses, because the
   cache holds too small a share of the model to use them all at speed. */
int janas_llm_model_experts_auto(const struct janas_llm_model *m);

/*
 * The experts the engine means to preload when the model is opened (from the
 * profile of this machine), and how many of them are in already: the filling
 * runs on a thread of its own, so the first replies do not wait for it.
 * *seconds is left at zero, for callers that timed a synchronous filling.
 */
uint64_t janas_llm_model_warm(const struct janas_llm_model *m, double *seconds);
uint64_t janas_llm_model_warm_done(const struct janas_llm_model *m);

/* True while the cache is still being filled in the background: what a pass
   costs then says nothing about the machine, so nothing is learned from it. */
int janas_llm_model_settling(const struct janas_llm_model *m);

/* The block attention and the recurrent log work in: a pass that can be
   taken back (a draft) is at most this long. */
#define JANAS_LLM_MAX_BLOCK 64

/*
 * Largest block of tokens one forward call accepts: larger than
 * JANAS_LLM_MAX_BLOCK, so that a prefill reads each expert once for many
 * tokens (256 unless JANAS_PREFILL_BLOCK says otherwise, 64 .. 4096). A call
 * longer than JANAS_LLM_MAX_BLOCK cannot be taken back on a model with a
 * recurrent state: the next call starts where it ended.
 */
uint32_t janas_llm_model_max_block(const struct janas_llm_model *m);

/*
 * Runs n tokens (1 <= n <= janas_llm_model_max_block) at positions pos0 .. pos0
 * + n - 1 in one pass: every weight is read once for the whole block. Positions
 * at and after pos0 are overwritten in the KV cache, so running a block again
 * from an earlier pos0 discards what followed (rollback of rejected tokens).
 * Models with a recurrent state (qwen3next) can go back only as far as the
 * start of the previous call: pos0 = 0 starts a new sequence, otherwise pos0
 * must lie between the first position of the previous call and the one
 * after its last token.
 * With all_logits the n_vocab logits of every token are written, token after
 * token; otherwise only those of the last token. Returns 0 or -1.
 */
int janas_llm_model_forward(struct janas_llm_model *m, const int32_t *tokens,
                            uint32_t n, uint32_t pos0, float *logits,
                            int all_logits);

/*
 * Forgets drop positions starting at keep, of the n_kv already computed:
 * the keys and values of the attention layers move down and the keys are
 * turned back by drop positions, so what is left behaves as if it had
 * always been there. The recurrent state (qwen3next and the like) is left
 * as it is. Returns 0, or -1 if the range makes no sense.
 */
int janas_llm_model_shift(struct janas_llm_model *m, uint32_t keep,
                          uint32_t drop, uint32_t n_kv);

/*
 * What the model has computed of positions 0 .. n - 1 (keys and values of
 * every attention layer, the MTP block's included, and the recurrent state),
 * copied out so that another sequence can use the model and this one come
 * back later without being computed again. The next call after
 * janas_llm_model_state_load continues at position n. A model with a
 * recurrent state saves only a position it stands at, or can reach again by
 * replaying its last call (from the start of that call to its end): NULL
 * otherwise, as when out of memory. Load returns 0, or -1 for a state taken
 * from a model of another shape.
 */
struct janas_llm_state;
struct janas_llm_state *janas_llm_model_state_save(struct janas_llm_model *m,
                                                   uint32_t n);
int janas_llm_model_state_load(struct janas_llm_model *m,
                               const struct janas_llm_state *st);
uint32_t janas_llm_state_length(const struct janas_llm_state *st);
size_t janas_llm_state_bytes(const struct janas_llm_state *st);
void janas_llm_state_free(struct janas_llm_state *st);
/* The state to a file and back (into a model of the same shape, with the
   same experts in use: NULL otherwise, or when the file is short). */
int janas_llm_state_write(const struct janas_llm_model *m,
                          const struct janas_llm_state *st, FILE *f);
struct janas_llm_state *janas_llm_state_read(const struct janas_llm_model *m,
                                             FILE *f);

/* One token at position pos: janas_llm_model_forward with n = 1. */
int janas_llm_model_decode(struct janas_llm_model *m, int32_t token,
                           uint32_t pos, float *logits);

/*
 * Multi-token prediction, when the model was loaded with an MTP block.
 * janas_llm_model_hidden: the final hidden states (n x d_model) of the
 * tokens of the last forward call. janas_llm_mtp_forward runs the MTP block
 * on n rows at positions pos0 .. pos0 + n - 1: row j combines tokens[j]
 * with the hidden state in row j of hidden (the main model's state for the
 * token before it), and its logits guess the token after tokens[j].
 * Positions at and after pos0 are overwritten in the block's KV cache, as in
 * janas_llm_model_forward. With logits NULL the LM head is skipped (the
 * call only fills the KV cache). janas_llm_mtp_hidden: the block's own
 * hidden states of that call, which stand in for the main model's when
 * guesses are chained.
 */
int janas_llm_model_has_mtp(const struct janas_llm_model *m);
/* 1 when the model carries a recurrent state (qwen3next, qwen35moe): it can
   go back no further than the start of its last forward call. */
int janas_llm_model_recurrent(const struct janas_llm_model *m);
const float *janas_llm_model_hidden(const struct janas_llm_model *m);
/* The final hidden states after the output norm, rows of d_model floats,
   of the last forward call: its last row, or all of them when it was asked
   for all logits. With logits NULL that call computes these and no logits
   (an embedding). Valid until the next call. */
const float *janas_llm_model_normed(const struct janas_llm_model *m);
int janas_llm_mtp_forward(struct janas_llm_model *m, const int32_t *tokens,
                          const float *hidden, uint32_t n, uint32_t pos0,
                          float *logits, int all_logits);
const float *janas_llm_mtp_hidden(const struct janas_llm_model *m);

/*
 * A draft from the MTP block: janas_llm_mtp_forward on the rows, then the
 * guess after the last row over the draft vocabulary only (the lowest
 * draft_vocab token ids plus the tokens noted with
 * janas_llm_mtp_note_tokens), with its softmax probability within that
 * vocabulary in *conf. Reads a fraction of the full LM head.
 */
int janas_llm_mtp_draft(struct janas_llm_model *m, const int32_t *tokens,
                        const float *hidden, uint32_t n, uint32_t pos0,
                        int32_t *token, float *conf);
void janas_llm_mtp_note_tokens(struct janas_llm_model *m, const int32_t *tokens,
                               uint32_t n);

/* Wall time per phase since load, in seconds. */
enum {
    JANAS_PH_QKV,
    JANAS_PH_ATTN,
    JANAS_PH_WO,
    JANAS_PH_ROUTER,
    JANAS_PH_EXPERTS,
    JANAS_PH_OUTPUT,
    JANAS_PH_SSM, /* Gated DeltaNet convolution and recurrence */
    JANAS_PH_COUNT
};
/* The model's compute pool, to hand to a second model as shared_compute. */
struct janas_pool *janas_llm_model_compute(struct janas_llm_model *m);

void janas_llm_model_phases(const struct janas_llm_model *m,
                            double out[JANAS_PH_COUNT]);

/*
 * Self-tuning (see tuner.h): with eco, configurations using the GPU are not
 * chosen (less power, when the GPU would only make it faster). The report
 * names the configuration chosen for each kind of pass.
 */
void janas_llm_model_set_eco(struct janas_llm_model *m, int eco);
void janas_llm_model_tuning(const struct janas_llm_model *m, char *buf,
                            size_t len);

/* For measurements (janas-bench): the tuner's candidates, and every pass on
   one of them (-1: back to tuning). */
/*
 * Which passes the engine actually gives the GPU: bit k for class k of
 * janas_tune_class (one token, a few, a block), zero for none. It is opened
 * unless eco, and the tuner then decides pass by pass, so having one and
 * using one are two different things - and which passes it is given is the
 * part worth telling anybody.
 */
int janas_llm_model_gpu_used(const struct janas_llm_model *m);
/* The compute threads of the configuration the tuner holds best for a
   kind of pass (janas_tune_class: 0 one token, 2 a block); 0 unknown. */
int janas_llm_model_threads(const struct janas_llm_model *m, int cls);

int janas_llm_model_candidates(const struct janas_llm_model *m, int *threads,
                               int *gpu, int max);
void janas_llm_model_force(struct janas_llm_model *m, int cand);

/*
 * How many experts a token uses, of the count the model was trained with.
 * Fewer means fewer bytes read per token, so faster replies of slightly lower
 * quality: the experts left out are those the router weights least. Returns 0
 * and changes nothing if n is outside 1 .. the model's own count; n of 0 asks
 * for that count. Takes effect from the next pass.
 */
uint32_t janas_llm_model_experts(const struct janas_llm_model *m,
                                 uint32_t *most);
int janas_llm_model_set_experts(struct janas_llm_model *m, uint32_t n);

/* The GPU in use, or NULL. */
struct janas_gpu *janas_llm_model_gpu(struct janas_llm_model *m);

/* The expert cache, for statistics and warm-up. */
struct janas_expert_cache *janas_llm_model_cache(struct janas_llm_model *m);

#endif
