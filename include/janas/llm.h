/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas/llm.h - the public C API of Janas-LLM.
 *
 * A model is opened once (weights stay on disk; experts are streamed from
 * it), then a chat holds a conversation with it: messages go in as UTF-8
 * text, the reply comes out piece by piece on request, with no callbacks.
 *
 * Conventions, chosen for foreign-function interfaces (BASIC MODERN, the
 * dialect Prabanta implements, among them):
 * - objects are opaque handles; no struct is passed or returned by value;
 * - fixed-width types only; strings are UTF-8, as a pointer and a byte
 *   length (a length of -1 means NUL-terminated);
 * - every call that can fail returns an int32_t code (JANAS_LLM_OK or a
 *   negative JANAS_LLM_E...); janas_llm_last_error() describes the last
 *   failure on the calling thread;
 * - parameter structs start with their own size: fill them with the
 *   _default function, change the fields needed, pass their address;
 * - memory the library returns (handles) is freed by the library; buffers
 *   are the caller's.
 * A model and its chat are used from one thread at a time.
 */
#ifndef JANAS_LLM_H
#define JANAS_LLM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define JANAS_LLM_API __declspec(dllexport)
#elif defined(__GNUC__)
#define JANAS_LLM_API __attribute__((visibility("default")))
#else
#define JANAS_LLM_API
#endif

#define JANAS_LLM_ABI_VERSION 1

#define JANAS_LLM_OK 0
#define JANAS_LLM_DONE 1      /* janas_llm_chat_next: the reply is over */
#define JANAS_LLM_EINVAL (-1) /* invalid argument */
#define JANAS_LLM_ENOMEM (-2) /* out of memory */
#define JANAS_LLM_EOPEN (-3)  /* model file missing, unreadable, invalid */
#define JANAS_LLM_EMODEL (-4) /* model not supported (architecture, chat) */
#define JANAS_LLM_EFULL (-5)  /* the context is full */
#define JANAS_LLM_EBUSY (-6)  /* the model already has a chat */
#define JANAS_LLM_ESMALL (-7) /* buffer too small */
#define JANAS_LLM_EFAIL (-8)  /* internal failure */
#define JANAS_LLM_ELIMIT (-9) /* the prompt is longer than max_input */

/* Power modes: eco never uses the GPU where it would only buy speed with
   power; max uses whatever is fastest; auto is max on mains power and eco
   on battery. */
#define JANAS_LLM_MODE_AUTO 0
#define JANAS_LLM_MODE_ECO 1
#define JANAS_LLM_MODE_MAX 2

typedef struct janas_llm janas_llm;           /* an open model */
typedef struct janas_llm_chat janas_llm_chat; /* a conversation */

/* JANAS_LLM_ABI_VERSION of the library: check it against the header's. */
JANAS_LLM_API int32_t janas_llm_abi_version(void);

/* The last failure on this thread, as text (never NULL). */
JANAS_LLM_API const char *janas_llm_last_error(void);

struct janas_llm_params {
    uint32_t size;        /* sizeof(struct janas_llm_params) */
    uint32_t n_ctx;       /* context, in tokens (default 16384) */
    uint64_t cache_bytes; /* RAM for streamed experts; 0: the free memory
                             less what the model needs, leaving a fifth of
                             the machine's memory free (default) */
    const char *mtp_path; /* the model's multi-token prediction block, for
                             faster replies; NULL: none (default) */
    /*
     * A small model of the same family, sharing this one's vocabulary, to
     * guess the next tokens so that several can be verified in one pass.
     * NULL: none (default), and then the guesses are copied from the
     * conversation instead, which is free and much less often right. Worth
     * it on a dense model, where a second token in a pass costs about six
     * per cent; on a mixture of experts a second token costs nearer thirty,
     * because it routes to other experts, and a whole model guessing is too
     * dear. Qwen3-0.6B for Qwen3-4B gives about 1.14x.
     */
    const char *draft_path;
    int32_t mode;        /* JANAS_LLM_MODE_* (default auto) */
    int32_t no_preload;  /* 1: do not fill the expert cache at open with the
                            experts this machine used most with this model
                            (a profile in ~/.cache/janas). Preloading costs
                            a few seconds once and saves the first replies
                            from finding their experts one token at a time */
    int32_t expert_bits; /* bits per weight of the experts' down matrix,
                            2, 4 or 6, for model files that hold it in bit
                            planes; 0 or 6: all of it, the weights as they
                            were quantized (default) */
    int32_t attn_scores; /* JANAS_LLM_ATTN_* (default auto) */
    /* memory left free for other programs when the expert cache takes the
       rest (cache_bytes 0); 0: a fifth of the machine's memory, at least
       2 GiB (default). More keeps a busy desktop out of swap, at the price
       of a smaller cache on a model larger than the memory */
    uint64_t reserve_bytes;
};

/*
 * How the attention scores are computed. EXACT reads the query as it is and
 * sums in floating point. FAST reads it as sixteen-bit integers with a scale
 * of its own: the sum is then an exact integer one, which the machine does
 * about twice as fast, and the only error left is the query's quantization -
 * a relative 3e-5, against the 4e-3 the eight-bit keys and values already
 * carry. AUTO takes FAST when the machine has the arithmetic for it and the
 * caller asked for more than the usual 16384 tokens of context, where
 * attention is the greater part of a token; below that it takes EXACT, since
 * a conversation that stays short would pay the quantization for a gain it
 * never sees. The environment variable JANAS_ATTN (float, int16, auto)
 * overrides whatever the program asks for.
 */
#define JANAS_LLM_ATTN_AUTO 0
#define JANAS_LLM_ATTN_EXACT 1
#define JANAS_LLM_ATTN_FAST 2

JANAS_LLM_API void janas_llm_params_default(struct janas_llm_params *p);

/* Opens a model file (.jns). p may be NULL for the defaults. */
JANAS_LLM_API int32_t janas_llm_open(const char *path,
                                     const struct janas_llm_params *p,
                                     janas_llm **out);
JANAS_LLM_API void janas_llm_close(janas_llm *llm);

/* The model's name from its metadata ("Qwen3 Next 80B A3B Instruct"; empty
   if the file has none). Writes at most cap bytes, NUL-terminated; *len gets
   the full length. */
/*
 * Whether the GPU may be given work at all. It is opened unless the mode is
 * eco, and the engine then decides pass by pass whether it helps; this says
 * no to the whole question. Fails where the machine has no usable GPU.
 */
JANAS_LLM_API int32_t janas_llm_set_gpu(janas_llm *llm, int32_t on);

JANAS_LLM_API int32_t janas_llm_name(const janas_llm *llm, char *buf,
                                     int32_t cap, int32_t *len);

/* The system message the programs of Janas give the model when none is
   asked for: its name is Janas, and it says which model it thinks with and
   which engine runs it. Text as for janas_llm_name. */
JANAS_LLM_API int32_t janas_llm_default_system(const janas_llm *llm, char *buf,
                                               int32_t cap, int32_t *len);

/*
 * The engine tunes itself while it runs: for each kind of pass (one token,
 * a few, a block) it tries the configurations the machine offers (thread
 * counts, the GPU) and keeps the fastest, remembering its choices per
 * machine and model in ~/.cache/janas/. janas_llm_set_mode changes the
 * power mode; janas_llm_tuning describes the current choices.
 */
JANAS_LLM_API int32_t janas_llm_set_mode(janas_llm *llm, int32_t mode);
JANAS_LLM_API int32_t janas_llm_tuning(const janas_llm *llm, char *buf,
                                       int32_t cap, int32_t *len);

/*
 * How many experts every token uses, of the count the model was trained with
 * (*used and *most, either may be NULL). Using fewer reads fewer bytes per
 * token, so replies come faster and lose a little quality: the experts left
 * out are those the router weights least. With Qwen3-Next-80B-A3B, eight of
 * its ten keep the most likely token in 387 positions of 400 against 394 for
 * all ten. n of 0 restores the model's own count; the change applies from the
 * next reply.
 */
JANAS_LLM_API int32_t janas_llm_experts(const janas_llm *llm, int32_t *used,
                                        int32_t *most);
JANAS_LLM_API int32_t janas_llm_set_experts(janas_llm *llm, int32_t n);

/*
 * Bits per weight the experts' down matrix is read with: 6 (all of it), 4 or
 * 2. A model file may hold that matrix as three planes of two bits, so the
 * engine can read a prefix of every expert - fewer bytes from disk and more
 * experts in the same memory - at some cost in quality. Chosen when the model
 * is opened (janas_llm_params.expert_bits); this tells what is in use, which
 * is 6 for a file without planes.
 */
JANAS_LLM_API int32_t janas_llm_expert_bits(const janas_llm *llm);

/* How the attention scores are being computed: JANAS_LLM_ATTN_EXACT or
   JANAS_LLM_ATTN_FAST, never AUTO. */
JANAS_LLM_API int32_t janas_llm_attn_scores(const janas_llm *llm);

/*
 * How the filling of the expert cache is going: *done of *total experts.
 * It runs while the machine is idle, so a front end can show it. Both are
 * zero when there was no profile of this machine to fill it from. Once it
 * is over *total is *done, also when it ended short of what it meant to
 * load: the conversation's own requests had filled the cache first.
 */
JANAS_LLM_API int32_t janas_llm_preload(const janas_llm *llm, int64_t *done,
                                        int64_t *total);

/* A short description of the model ("qwen3next, 48 layers, ..."). Writes at
   most cap bytes, NUL-terminated; *len gets the full length. */
JANAS_LLM_API int32_t janas_llm_describe(const janas_llm *llm, char *buf,
                                         int32_t cap, int32_t *len);

/*
 * Embeddings, from models trained to give them (Qwen3-Embedding): the
 * model's metadata says how the states of a text's tokens make one vector.
 * janas_llm_embed_dim: the length of the vector, 0 for a model that gives
 * none. janas_llm_embed: the vector of a text, of unit length, into out
 * (room for dim numbers): all of it when dim is 0 or larger than the
 * model's, else its first dim numbers made unit length again - what models
 * trained for it (Matryoshka) allow; *n gets its length and *tokens (may be
 * NULL) how many tokens the text took. JANAS_LLM_EBUSY while the model has
 * a chat: the text would overwrite the conversation.
 */
JANAS_LLM_API int32_t janas_llm_embed_dim(const janas_llm *llm);
JANAS_LLM_API int32_t janas_llm_embed(janas_llm *llm, const char *text,
                                      int32_t len, int32_t dim, float *out,
                                      int32_t *n, int32_t *tokens);

/*
 * Tokens. janas_llm_tokenize: the ids of a text, at most max; *n gets the
 * full count (JANAS_LLM_ESMALL when above max). With special, the model's
 * special tokens written in the text become their ids. janas_llm_token_text:
 * the bytes of one token (it can end inside a UTF-8 character).
 */
JANAS_LLM_API int32_t janas_llm_tokenize(const janas_llm *llm, const char *text,
                                         int32_t len, int32_t special,
                                         int32_t *ids, int32_t max, int32_t *n);
JANAS_LLM_API int32_t janas_llm_token_text(const janas_llm *llm, int32_t id,
                                           char *buf, int32_t cap,
                                           int32_t *len);

struct janas_llm_chat_params {
    uint32_t size;     /* sizeof(struct janas_llm_chat_params) */
    float temperature; /* 0: always the most likely token (default 0.7) */
    int32_t top_k;     /* 0: no limit (default 20) */
    float top_p;       /* default 0.8 */
    float min_p;       /* default 0 */
    uint64_t seed;     /* 0: a new one each chat (default) */
    int32_t speculate; /* 1: draft tokens ahead when it pays, with the
                          MTP block if open, else from the context
                          (default); 0: never. The text is the same. */
    int32_t max_reply; /* tokens per reply; 0: no limit (default) */
    int32_t thinking;  /* models that reason before replying (Qwen3.5 and
                          later): 1 reasoning on, 0 off, -1 the model's own
                          default (default) */
    int32_t recap;     /* when the context fills and the oldest exchanges
                          have to go, first ask the model for a short recap
                          of them, which stays in the window: 1 yes
                          (default), 0 no - the oldest turns are simply
                          forgotten */
    /* OpenAI's penalties (default 0): a token's logit loses presence once
       it has appeared in the reply, and frequency for every time it has */
    float presence_penalty, frequency_penalty;
    int32_t logprobs;     /* 1: keep the log-probability of every token of
                             the reply (janas_llm_chat_token); 0: no (default) */
    int32_t top_logprobs; /* and that many of the most likely alternatives,
                             0..20 (default 0) */
    /* the longest prompt a reply is given, in tokens, below what the
       context holds: a conversation loaded (janas_llm_chat_load, _run) or
       a message sent whose prompt would be longer fails with
       JANAS_LLM_ELIMIT and nothing is computed; janas_llm_chat_stats then
       has its length in prompt_tokens, and how it is made up. Nothing is
       ever cut to fit. 0: no limit but the context (default) */
    uint32_t max_input;
    /* the longest answer, in tokens, the reasoning not counted (the text
       and the tool calls a reader gets): what a client means by a limit
       when it does not know the model reasons first. The reply ends with
       JANAS_LLM_FINISH_LENGTH there, like at max_reply, which counts the
       reasoning too and still applies. 0: no limit (default) */
    int32_t max_answer;
};

JANAS_LLM_API void
janas_llm_chat_params_default(struct janas_llm_chat_params *p);

/* A conversation with the model (one at a time per model). p may be NULL. */
JANAS_LLM_API int32_t
janas_llm_chat_create(janas_llm *llm, const struct janas_llm_chat_params *p,
                      janas_llm_chat **out);
JANAS_LLM_API void janas_llm_chat_destroy(janas_llm_chat *c);

/* New parameters, from the next reply on. */
JANAS_LLM_API int32_t janas_llm_chat_set_params(
    janas_llm_chat *c, const struct janas_llm_chat_params *p);

/* Sets the system message. Only before the first message (see reset). */
JANAS_LLM_API int32_t janas_llm_chat_system(janas_llm_chat *c, const char *text,
                                            int32_t len);

/*
 * Sends a user message; the reply follows through janas_llm_chat_next. A
 * reply still being read is cut where it is and closed first.
 */
JANAS_LLM_API int32_t janas_llm_chat_send(janas_llm_chat *c, const char *text,
                                          int32_t len);

/*
 * What the tools answered to the calls of the last reply (see
 * janas_llm_chat_calls), in the conversation kept here: n answers, in the
 * order of the calls, texts[i] of lens[i] bytes (lens NULL, or an entry of
 * -1: NUL-terminated). The model's reply to them follows through
 * janas_llm_chat_next, as after janas_llm_chat_send; a reply still being
 * read is closed first. A caller that keeps the conversation itself gives
 * them as JANAS_LLM_ROLE_TOOL messages to janas_llm_chat_load instead.
 */
JANAS_LLM_API int32_t janas_llm_chat_send_results(janas_llm_chat *c, int32_t n,
                                                  const char *const *texts,
                                                  const int32_t *lens);

/*
 * The next piece of the reply: whole UTF-8 characters, at most cap bytes
 * (cap >= 8), not NUL-terminated, length in *len. Returns JANAS_LLM_OK with
 * a piece (possibly empty while the model works), JANAS_LLM_DONE when the
 * reply is over (*len = 0), or an error (JANAS_LLM_EFULL: the context is
 * full, the reply ends there; start over with reset).
 *
 * The prompt is read here, not when it is sent or loaded: those only build
 * it and return at once. A long prompt is read a block of tokens at a time
 * (256), and every block returns an empty piece, so the caller has control
 * between them - to say how far the reading has gone (janas_llm_chat_stats:
 * stage, input_done of input_tokens), or to stop. On a slow machine a
 * prompt of 260,000 tokens takes hours, and this is how a program shows it
 * is working and not stuck.
 */
JANAS_LLM_API int32_t janas_llm_chat_next(janas_llm_chat *c, char *buf,
                                          int32_t cap, int32_t *len);

/* Roles of the messages of janas_llm_chat_load. */
#define JANAS_LLM_ROLE_SYSTEM 0
#define JANAS_LLM_ROLE_USER 1
#define JANAS_LLM_ROLE_ASSISTANT 2
#define JANAS_LLM_ROLE_TOOL 3 /* what a tool the model called answered */

/*
 * A whole conversation in place of the current one, the reply to its last
 * message following through janas_llm_chat_next: n messages, roles[i] one
 * of JANAS_LLM_ROLE_..., texts[i] of lens[i] bytes (lens NULL, or an entry
 * of -1: NUL-terminated). Only the first may be a system message; the last
 * must be the user's or a tool's. This is how a caller that keeps the
 * conversation itself (an HTTP client) talks to the model.
 *
 * Whatever the new conversation shares with the one already computed is not
 * read again: sending the same messages plus a new one reads only the new
 * one. The last replies are remembered with their reasoning, so an
 * assistant message that is one of them, as its answer alone, counts as the
 * same turn. Models with a recurrent state (Qwen3-Next, Qwen3.5 and 3.6)
 * can only go forward: a conversation that differs from the computed one
 * before its end is read from the start. JANAS_LLM_EFULL when the messages
 * and a reply do not fit in the context: nothing slides here.
 */
JANAS_LLM_API int32_t janas_llm_chat_load(janas_llm_chat *c, int32_t n,
                                          const int32_t *roles,
                                          const char *const *texts,
                                          const int32_t *lens);

/*
 * The same, message by message, with what janas_llm_chat_load cannot say:
 * the tool calls of an assistant's message. janas_llm_chat_begin starts an
 * empty conversation; janas_llm_chat_add appends a message;
 * janas_llm_chat_add_call appends a call to the last message, which must be
 * the assistant's (name, and the arguments as a JSON object); and
 * janas_llm_chat_run loads it all as janas_llm_chat_load does, the reply
 * following through janas_llm_chat_next.
 */
JANAS_LLM_API int32_t janas_llm_chat_begin(janas_llm_chat *c);
JANAS_LLM_API int32_t janas_llm_chat_add(janas_llm_chat *c, int32_t role,
                                         const char *text, int32_t len);
JANAS_LLM_API int32_t janas_llm_chat_add_call(janas_llm_chat *c,
                                              const char *name,
                                              int32_t name_len,
                                              const char *args,
                                              int32_t args_len);
JANAS_LLM_API int32_t janas_llm_chat_run(janas_llm_chat *c);
/* How many tokens the conversation built would be, with the tools and the
   start of the reply, in *n: nothing is computed and the conversation
   already computed is kept. */
JANAS_LLM_API int32_t janas_llm_chat_count(janas_llm_chat *c, int32_t *n);

/*
 * Conversations kept computed besides the current one, for a caller that
 * switches between several: an HTTP server with many clients, or a client
 * that asks on the side for titles and suggestions. When a conversation is
 * loaded (janas_llm_chat_load, _run, _prompt, _infill) and a kept one shares
 * more of it than the current one, the two change places; a current one
 * that would lose much of what it has computed is kept before it is cut.
 * At most n are kept, in at most bytes of memory (0: half of what the
 * expert cache and the margin left to other programs leave free at the time
 * of the call, at least 256 MiB); the least recently used go first.
 * n of 0 (the default) keeps none, and frees those kept. What is kept is
 * copied, not computed: memory is its only cost. Models with a recurrent
 * state (Qwen3-Next, Qwen3.5 and 3.6), which cannot go back from a reply to
 * its prompt, also keep the prompt, computed when the conversation is
 * loaded: a second reply to it does not read it again.
 */
JANAS_LLM_API int32_t janas_llm_chat_keep(janas_llm_chat *c, int32_t n,
                                          uint64_t bytes);
/* How many conversations are kept, and the memory they take (either may be
   NULL). */
JANAS_LLM_API int32_t janas_llm_chat_kept(const janas_llm_chat *c, int32_t *n,
                                          uint64_t *bytes);

/*
 * Below the memory, the disk: the conversations kept go there when they
 * leave the memory to make room, and when the chat ends, so that they
 * outlive the program - a long system message, tools included, is then
 * read once and never again. In ~/.cache/janas, in a directory of the
 * model's own, at most bytes of it (and never more than a quarter of the
 * space left on that disk); the least recently used go first. Only copies
 * of at least 1024 tokens are written, and only then, not at every turn: a
 * disk wears with what is written to it. A file is used only by a model
 * that computes the same numbers: a file written with fewer experts, or
 * fewer of their bits, in use is deleted when found. bytes of 0 (the
 * default) turns it off; the files stay. It works with janas_llm_chat_keep,
 * not without it. janas_llm_chat_kept_disk: how many files, and their
 * bytes.
 */
JANAS_LLM_API int32_t janas_llm_chat_keep_disk(janas_llm_chat *c,
                                               uint64_t bytes);
JANAS_LLM_API int32_t janas_llm_chat_kept_disk(const janas_llm_chat *c,
                                               int32_t *n, uint64_t *bytes);

/*
 * Tools the model may call: a JSON array as OpenAI's API has it, of
 * {"type": "function", "function": {"name", "description", "parameters"}}
 * (or of the functions alone); NULL or a length of 0 takes them away. They
 * are written into the system message, the way the model's chat template
 * does, so they take effect from the next conversation loaded (or from the
 * first message after a reset). JANAS_LLM_EMODEL where the model's template
 * has no tools.
 *
 * A call the model writes is held to a grammar of the functions and their
 * parameters from the moment it opens one, so its arguments are always a
 * JSON object valid against the function's schema (see
 * janas_llm_chat_format for what a schema can hold it to). The calls of a
 * reply are read with janas_llm_chat_calls and janas_llm_chat_call; they
 * are not part of the text janas_llm_chat_next returns.
 */
JANAS_LLM_API int32_t janas_llm_chat_tools(janas_llm_chat *c, const char *json,
                                           int32_t len);

/* Whether the model calls a tool, from the next reply on: as it judges
   (AUTO, default), never (NONE), at least one (REQUIRED), or the function
   named (FUNCTION, name of len bytes). parallel 0 allows one call at most
   where calls are required. */
#define JANAS_LLM_TOOLS_AUTO 0
#define JANAS_LLM_TOOLS_NONE 1
#define JANAS_LLM_TOOLS_REQUIRED 2
#define JANAS_LLM_TOOLS_FUNCTION 3
JANAS_LLM_API int32_t janas_llm_chat_tool_choice(janas_llm_chat *c,
                                                 int32_t choice,
                                                 const char *name, int32_t len,
                                                 int32_t parallel);

/*
 * The form of the answer (the reasoning stays free), from the next reply
 * on: text (default); a JSON object; or JSON valid against a JSON Schema,
 * given as text. The answer is held to a grammar token by token, so it is
 * valid when the model ends it; a reply cut by max_reply or the context is
 * not. A grammar holds the types, the properties of an object in the
 * schema's order (the required ones always there, no others), array items
 * and bounds up to 32, string lengths up to 64, enum, const, anyOf, oneOf,
 * $ref within the schema, nullable; pattern, format and numeric bounds are
 * not enforced.
 */
#define JANAS_LLM_FORMAT_TEXT 0
#define JANAS_LLM_FORMAT_JSON 1
#define JANAS_LLM_FORMAT_SCHEMA 2
JANAS_LLM_API int32_t janas_llm_chat_format(janas_llm_chat *c, int32_t kind,
                                            const char *schema, int32_t len);

/* The tool calls of the reply so far (one is counted once it is closed),
   and call i: its name and its arguments (a JSON object), text as for
   janas_llm_name. */
JANAS_LLM_API int32_t janas_llm_chat_calls(const janas_llm_chat *c);
JANAS_LLM_API int32_t janas_llm_chat_call(const janas_llm_chat *c, int32_t i,
                                          char *name, int32_t name_cap,
                                          int32_t *name_len, char *args,
                                          int32_t args_cap, int32_t *args_len);

/* A bias added to the logits of these tokens, from the next reply on
   (OpenAI's logit_bias, -100..100); n of 0 clears it. */
JANAS_LLM_API int32_t janas_llm_chat_logit_bias(janas_llm_chat *c, int32_t n,
                                                const int32_t *ids,
                                                const float *bias);

/*
 * The tokens of the reply so far, one by one: janas_llm_chat_tokens counts
 * them; janas_llm_chat_token gives token i - its id, which part of the reply
 * it belongs to (JANAS_LLM_PART_...), its log-probability in the model's own
 * distribution (0 unless janas_llm_chat_params.logprobs) and its bytes
 * (text as for janas_llm_name; a token may end inside a UTF-8 character);
 * janas_llm_chat_token_top gives up to max of the most likely tokens in its
 * place (params.top_logprobs of them were kept).
 */
#define JANAS_LLM_PART_ANSWER 0
#define JANAS_LLM_PART_REASONING 1
#define JANAS_LLM_PART_CALL 2 /* inside a tool call */
#define JANAS_LLM_PART_MARK 3 /* a marker: reasoning or call opened, closed */
JANAS_LLM_API int32_t janas_llm_chat_tokens(const janas_llm_chat *c);
JANAS_LLM_API int32_t janas_llm_chat_token(const janas_llm_chat *c, int32_t i,
                                           int32_t *id, int32_t *part,
                                           float *logprob, char *buf,
                                           int32_t cap, int32_t *len);
JANAS_LLM_API int32_t janas_llm_chat_token_top(const janas_llm_chat *c,
                                               int32_t i, int32_t max,
                                               int32_t *ids, float *logprobs,
                                               int32_t *n);

/* Raw text to continue, with no chat format around it; the continuation
   follows through janas_llm_chat_next. Shared text is reused as above. */
JANAS_LLM_API int32_t janas_llm_chat_prompt(janas_llm_chat *c, const char *text,
                                            int32_t len);

/*
 * Text to fill in between prefix and suffix, for models trained to do it
 * (their vocabulary has <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>): the
 * middle follows through janas_llm_chat_next. JANAS_LLM_EMODEL where the
 * model has no such tokens.
 */
JANAS_LLM_API int32_t janas_llm_chat_infill(janas_llm_chat *c,
                                            const char *prefix, int32_t len,
                                            const char *suffix,
                                            int32_t suffix_len);

/* 1 while the reply's pieces are the model's reasoning (between <think>
   and </think>), 0 for the answer itself. */
JANAS_LLM_API int32_t janas_llm_chat_thinking(const janas_llm_chat *c);

/* 1 where the model reasons before replying and can be told not to: its
   chat template carries the switch and its vocabulary both markers. Where
   this is 0, janas_llm_chat_params.thinking changes nothing. */
JANAS_LLM_API int32_t janas_llm_reasons(const janas_llm *llm);

/* Forgets the conversation (the system message too). */
JANAS_LLM_API int32_t janas_llm_chat_reset(janas_llm_chat *c);

struct janas_llm_chat_stats {
    uint32_t size;          /* sizeof(struct janas_llm_chat_stats) */
    uint32_t input_tokens;  /* last message, with the chat format's tokens */
    uint32_t output_tokens; /* last reply */
    uint32_t context_used, context_size; /* tokens */
    uint32_t drafted, accepted;          /* last reply, speculation */
    double input_seconds;                /* reading the message */
    double output_seconds;               /* writing the reply */
    double total_seconds;                /* from send to the end of the reply */
    /* the routed experts of message and reply: those used, those read from
       the file (not in the cache), the bytes read, the time spent waiting */
    uint64_t experts_used, experts_read, bytes_read;
    double io_wait_seconds;
    /* passes of the reply the engine ran without drafting although drafts
       were asked for, because it judged they were not paying: when this is
       not zero the reply was slower than the drafts would have made it, and
       the engine, not the caller, decided so */
    uint32_t auto_off_passes;
    /* why the reply ended: JANAS_LLM_FINISH_... (0 while it runs) */
    int32_t finish;
    /* the whole prompt of the reply, and how much of it was already
       computed (janas_llm_chat_load, janas_llm_chat_prompt): input_tokens
       is the part read for this reply */
    uint32_t prompt_tokens, cached_tokens;
    /* --- from here on, added with ABI version 1 still (a caller that
       sets size to the older struct gets the older fields only) --- */
    /* what the reply is doing (JANAS_LLM_STAGE_...), and of input_tokens
       how many have been read: the progress while the stage is INPUT */
    int32_t stage;
    uint32_t input_done;
    /* seconds from the send (or load) to the first token of the reply, 0
       before it; seconds spent building the prompt before reading it (the
       chat format, the tokens, a kept conversation looked for) */
    double first_token_seconds, prepare_seconds;
    /* processor time the reading of the prompt took, every thread of the
       process summed: above input_seconds when several threads worked */
    double input_cpu_seconds;
    /* compute threads of the passes that read the prompt and of those that
       write the reply, as the engine chose them for this machine */
    int32_t input_threads, output_threads;
    /* memory of the conversation computed, bytes: attention's keys and
       values for context_used tokens (the recurrent state of the models
       that have one does not grow and is not counted) */
    uint64_t context_bytes;
    /* the most memory the process has held resident so far, bytes */
    uint64_t peak_rss_bytes;
    /* the prompt by where its tokens come from: the system message (the
       tools' descriptions included, which are prompt_tools of it), the
       conversation before the last message, and the last message with the
       start of the reply (the user's, or the tools' answers). The three sum
       to prompt_tokens */
    uint32_t prompt_system, prompt_tools, prompt_history, prompt_last;
    /* while the stage is INPUT: seconds left to read the prompt, an
       estimate (-1 while there is too little to go on). A token costs more
       the more context it follows, attention having more to look at, so
       the estimate follows that cost as it grows with the position rather
       than the latest speed, which would promise too little */
    double input_eta_seconds;
};

/* janas_llm_chat_stats.stage */
#define JANAS_LLM_STAGE_IDLE 0   /* no reply asked for yet */
#define JANAS_LLM_STAGE_INPUT 1  /* reading the prompt */
#define JANAS_LLM_STAGE_OUTPUT 2 /* writing the reply */
#define JANAS_LLM_STAGE_DONE 3   /* the reply is over */

/*
 * context_used counts every token the conversation holds, so after a reply
 * it is prompt_tokens + output_tokens + 1: the token that ended the reply
 * stays in the conversation (the next turn reads on from it) and is not
 * part of the reply's text or of output_tokens. A reply that ended because
 * the context was full (JANAS_LLM_FINISH_CONTEXT) leaves context_used at
 * context_size + 1: its last token was written, and there is no room left
 * to read it.
 *
 * Environment, for any program on the library, its own code unchanged:
 * JANAS_PROGRESS=s writes a line on standard error every s seconds while a
 * prompt is read (tokens read, rate, an estimate of the time left) and one
 * at the end of each reply; JANAS_METRICS=file appends the same as JSON
 * lines (events open, prompt, progress, first_token, reply), counters and
 * times only, never the text of a prompt or a reply. JANAS_METRICS_EVERY=s
 * sets how often a progress event is written (default 10).
 */

#define JANAS_LLM_FINISH_STOP 1    /* the model ended it */
#define JANAS_LLM_FINISH_LENGTH 2  /* max_reply tokens */
#define JANAS_LLM_FINISH_CONTEXT 3 /* the context is full */
#define JANAS_LLM_FINISH_ERROR 4
#define JANAS_LLM_FINISH_TOOLS 5 /* the model ended it after calling tools */

/* Counters of the last message and reply. s->size must be set. */
JANAS_LLM_API int32_t janas_llm_chat_stats(const janas_llm_chat *c,
                                           struct janas_llm_chat_stats *s);

#ifdef __cplusplus
}
#endif

#endif
