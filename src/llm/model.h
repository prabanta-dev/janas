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
    uint64_t mtp_cache_bytes; /* RAM for its experts; 0: all of them */
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
    int min_quality; /* JANAS_LLM_QUALITY_*: the lowest the engine's own
                        choices go (expert bits, experts left out) */
    int head_bits;   /* 4: the output head, if the file holds it at six bits
                        (Q6_K), requantized to four (Q4_K) at load - faster, a
                        little less accurate; 0 or 6: as in the file
                        (environment JANAS_HEAD_BITS overrides) */
    int no_npu;      /* 1: the NPU left off, its copies' memory being the
                        expert cache's (janas_llm_open) */
    int ple_ram;     /* 1: the per-layer table (Gemma 4's per-layer
                        embeddings, Qwen3.8-Flash-Next's n-grams) read into
                        memory at load, where the machine holds it beside the
                        experts (janas_llm_open); 0: read a row at a time
                        from the file (environment JANAS_PLE_RAM=0/1
                        overrides) */
    uint32_t min_block; /* the prefill block at least this (a picture whose
                           tokens attend both ways goes in one pass) */
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
/* The memory the model holds: its resident weights as read (a table read
   from the file a row at a time, Gemma 4's per-layer embeddings or
   Qwen3.8-Flash-Next's n-grams, is not held unless read into memory), its
   expert cache's arena, and its keys and values with their scales. */
void janas_llm_model_memory(const struct janas_llm_model *m, uint64_t *resident,
                            uint64_t *cache, uint64_t *kv);

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
/* The bytes of the output head requantized at load (head_bits 4), before
   and after; 0 and 0 if it was not. */
void janas_llm_model_head_requant(const struct janas_llm_model *m,
                                  uint64_t *before, uint64_t *after);
/* The output head's type in the file (BF16 30, F16 1) when it was rewritten
   as Q8_0 at load, else 0; and its bytes before and after. */
/* qwen4exp's mixes of the streams as JANAS_HC_MIX rewrote them at load:
   its value ("q8"), or NULL when they are as in the file; and
   their bytes before and after. */
const char *janas_llm_model_hc_quant(const struct janas_llm_model *m,
                                     uint64_t *before, uint64_t *after);
int janas_llm_model_head16(const struct janas_llm_model *m, uint64_t *before,
                           uint64_t *after);
/* How the attention scores are computed (janas_attn_scores), and whether the
   engine chose it. */
int janas_llm_model_attn_scores(const struct janas_llm_model *m);
int janas_llm_model_attn_scores_auto(const struct janas_llm_model *m);

/* Whether the engine also chose to leave out the light experts that are not
   in RAM, because the cache holds too small a share of the model to read
   them all at speed; and the weight below which it does (0: never). Setting
   the number of experts (janas_llm_model_set_experts) turns it off. */
int janas_llm_model_experts_auto(const struct janas_llm_model *m);
float janas_llm_model_miss_skip(const struct janas_llm_model *m);

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

/* The last janas_llm_model_forward's time without what it waited for
   experts read from the disk, nor the NPU's first compilations: what the
   pass costs with the experts in memory, as the tuner learns it. 0 when
   the tuner ran the pass on an alternative it was trying, not on its
   choice: what that costs says nothing of the passes to come. */
double janas_llm_model_pass_seconds(const struct janas_llm_model *m);

/* The block attention and the recurrent log work in: a pass that can be
   taken back (a draft) is at most this long. */
#define JANAS_LLM_MAX_BLOCK 64

/*
 * The prefill block of the model in j: JANAS_PREFILL_BLOCK, else 512 for a
 * model whose feed-forwards (its experts, or the one of a dense model in
 * the expert cache) the GPU's fast prompt takes whole (gate and up Q4_K,
 * down Q4_K or Q6_K, plain or in planes; janas_gpu_fast_prompt on), else
 * 256;
 * 64 .. 4096; at_least raises it (pictures in one pass).
 */
uint32_t janas_llm_prefill_block(const struct janas_jns *j, uint32_t at_least);

/* The model's own layers in j: all of them less an MTP block in the file
   (qwen35moe: nextn_predict_layers, the last ones). */
uint32_t janas_llm_main_layers(const struct janas_jns *j);

/*
 * Largest block of tokens one forward call accepts: larger than
 * JANAS_LLM_MAX_BLOCK, so that a prefill reads each expert once for many
 * tokens (janas_llm_prefill_block). A call
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
/*
 * Pictures in the sequence (model_media.c): the nx * ny embeddings
 * (d_model wide, row-major over the grid) taken for the tokens at
 * at .. at + nx * ny - 1 in place of theirs, with positions for the RoPE
 * of their row and column; the text after a picture goes on from its start
 * plus max(nx, ny), as Qwen3-VL's get_rope_index has it. both_ways: its
 * tokens see one another, the later ones too (Gemma 4 12B and 26B), so
 * a pass must hold the whole picture. _add: 0, or -1 (memory, overlap, no
 * room). _drop: those that begin at or after `from`
 * forgotten (one cut by it as well). _count: how many there are.
 */
int janas_llm_model_media_add(struct janas_llm_model *m, uint32_t at,
                              const float *embd, uint32_t nx, uint32_t ny,
                              int both_ways);
void janas_llm_model_media_drop(struct janas_llm_model *m, uint32_t from);
/* A pass of n tokens from pos0 shortened so that it does not cut a picture
   whose tokens attend both ways: ending before it, or taking it whole. */
uint32_t janas_llm_model_media_cut(const struct janas_llm_model *m,
                                   uint32_t pos0, uint32_t n);
uint32_t janas_llm_model_media_count(const struct janas_llm_model *m);
/* The model's compute threads, for work between its passes (a picture's
   encoding). */
struct janas_pool *janas_llm_model_pool(const struct janas_llm_model *m);

int janas_llm_model_decode(struct janas_llm_model *m, int32_t token,
                           uint32_t pos, float *logits);

/*
 * Multi-token prediction, when the model was loaded with an MTP block.
 * janas_llm_model_hidden: the final hidden states (n rows of
 * janas_llm_model_hidden_width floats: d_model, or for qwen4exp its
 * residual's streams before the last mix) of the tokens of the last forward
 * call. janas_llm_mtp_forward runs the MTP block
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

/*
 * The tokens the model may give from now on, n of them (n 0: all of them
 * again). Their rows of the LM head are copied into one matrix and the
 * forward pass computes those alone - a few thousand of 128,257 rows, at a
 * fraction of the head's reads - the other logits set to -inf. The MTP
 * block's and the draft head's logits are not restricted: a draft outside
 * the vocabulary is rejected by the pass that checks it. -1 when the head's
 * rows cannot be copied (its weights in bit planes) or memory runs out.
 */
int janas_llm_model_set_vocab(struct janas_llm_model *m, const int32_t *ids,
                              uint32_t n);
/* 1 when chained drafts keep the position of the first (Gemma 4's
   assistant, which reads the main model's cache and has none of its
   own); 0 when each is one position further (an MTP block) */
int janas_llm_mtp_chain_fixed(const struct janas_llm_model *m);
/* 1 when the model carries a recurrent state (qwen3next, qwen35moe): it can
   go back no further than the start of its last forward call. */
int janas_llm_model_recurrent(const struct janas_llm_model *m);
const float *janas_llm_model_hidden(const struct janas_llm_model *m);
/* The floats of a row of janas_llm_model_hidden and janas_llm_mtp_hidden:
   d_model, or qwen4exp's residual of hc_n streams. */
uint32_t janas_llm_model_hidden_width(const struct janas_llm_model *m);
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
    JANAS_PH_MIX, /* qwen4exp's mixes of the residual's streams */
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
/* The most tokens a pass of drafts is worth (0: no limit): where the
   passes of a few tokens go whole to the GPU, as many as its shaders of a
   few vectors take (janas_gpu_few_max) - one more sends every product to
   the shaders of many vectors, and Qwen3.8-27B's pass of 9 tokens took
   855 ms against 437 for one of 5 (13 Oct 2026). */
uint32_t janas_llm_model_pass_cap(const struct janas_llm_model *m);
/* Whether the GPU reads the prompts' blocks whatever the tuner chooses
   (the fast prompt, outside eco mode, unless the user keeps prompts off
   the GPU). */
int janas_llm_model_gpu_prompts(const struct janas_llm_model *m);
/* Whether every pass of the model runs whole on a card with memory of its
   own, whatever the tuner chooses (JANAS_GPU_WHOLE): so once it has, or,
   before the first pass, where nothing stands in its way. */
int janas_llm_model_gpu_whole(const struct janas_llm_model *m);
/* What the GPU did for the model, for a report, into buf (len bytes, a
   line a fact, each ending in a newline): the card, its memory and link;
   on a card with memory of its own whether every pass runs whole there,
   or why not; the experts' tier and where the experts were found; the
   phases' times with JANAS_GPU_TOKEN_PROFILE=1. Empty without a GPU. */
void janas_llm_model_gpu_report(const struct janas_llm_model *m, char *buf,
                                size_t len);
/* Whether the engine gives the NPU a prompt's blocks by itself, where it
   measures faster: on the models whose answers on the NPU measured at par
   with the CPU's against llama.cpp; on the others only when asked
   (--prompt-on npu). */
int janas_llm_model_npu_auto(const struct janas_llm_model *m);
/* The combinations of devices the user allowed a kind of pass
   (common/devices.h, janas_tune_class), as this machine has them: 0 any
   (the engine's own limits, as janas_llm_model_npu_auto, apart). */
int janas_llm_model_devices(const struct janas_llm_model *m, int cls);
/* The compute threads of the configuration the tuner holds best for a
   kind of pass (janas_tune_class: 0 one token, 1 a few, 2 a block, 3 a
   short one); 0 unknown. */
int janas_llm_model_threads(const struct janas_llm_model *m, int cls);

/* The tuner's candidates: threads, GPU, NPU and the CPU alone for a
   prompt's blocks (npu and alone may be NULL). */
int janas_llm_model_candidates(const struct janas_llm_model *m, int *threads,
                               int *gpu, int *npu, int *alone, int max);
/* Whether the tuner holds the CPU alone best for a prompt's blocks. */
int janas_llm_model_blocks_alone(const struct janas_llm_model *m);
/* As janas_llm_model_gpu_used, for the NPU (looked for at load unless
   JANAS_NPU=0). */
int janas_llm_model_npu_used(const struct janas_llm_model *m);
/* The NPU's name, or NULL when none is used. */
const char *janas_llm_model_npu_name(const struct janas_llm_model *m);
/* Whether the NPU takes products of weights of this type (model_npu.c). */
int janas_llm_npu_type(int type);
/* Seconds spent making the NPU's graphs so far (a matrix's or a layer's
   first block: its weights prepared, the graph compiled and run once):
   kept out of the tuner's costs, as the waits for the disk are, and out of
   janas-bench's rates. */
double janas_llm_model_npu_compile_seconds(const struct janas_llm_model *m);
/* The NPU's preparation, safe from another thread while the engine works
   (janas_llm_npu_preparing): 1 while it goes on, *done of *total graphs
   made (*total 0: not known ahead), *seconds since it began; 0 after,
   *seconds the session's preparations; 0 and zeros without an NPU. */
int janas_llm_model_npu_preparing(const struct janas_llm_model *m,
                                  int32_t *done, int32_t *total,
                                  double *seconds);
/* Without an NPU: why ("" with one). */
const char *janas_llm_model_npu_why(const struct janas_llm_model *m);
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

/*
 * The router's choices as they are made: after each MoE layer of the main
 * model routes a pass of n tokens, fn gets the layer and, token after
 * token, the k experts each chose (exp[j * k + i]). For measures of which
 * experts a text uses; NULL stops it. Runs on the calling thread, inside
 * the pass: it must be quick and must not call the model.
 */
typedef void (*janas_llm_route_fn)(void *ctx, uint32_t layer, uint32_t n,
                                   uint32_t k, const uint32_t *exp);
void janas_llm_model_route_observer(struct janas_llm_model *m,
                                    janas_llm_route_fn fn, void *ctx);
/*
 * For measures of prefetching: after each layer l of the main model, the
 * router of layer l + 1 applied ahead to the state after l (normed as l + 1
 * will norm it, without l + 1's attention in between), and fn gets layer
 * l + 1 and, token after token, the k experts it ranks first (k <= 32). Costs
 * a router and a norm per layer while set; NULL stops it. The pass itself
 * is unchanged.
 */
void janas_llm_model_route_predictor(struct janas_llm_model *m,
                                     janas_llm_route_fn fn, void *ctx,
                                     uint32_t k);
/* With early set, the prediction for layer l + 1 is made sooner: on the
   state after layer l's attention, before its experts (more time to read
   ahead, a state further from the one l + 1 routes). */
void janas_llm_model_route_predictor_early(struct janas_llm_model *m,
                                           int early);

/* The GPU in use, or NULL. */
struct janas_gpu *janas_llm_model_gpu(struct janas_llm_model *m);
/* Without one, why: the driver's or the build's reason, or that it was
   turned off ("" when there is a GPU). */
const char *janas_llm_model_gpu_why(const struct janas_llm_model *m);

/* The expert cache, for statistics and warm-up. */
struct janas_expert_cache *janas_llm_model_cache(struct janas_llm_model *m);

#endif
