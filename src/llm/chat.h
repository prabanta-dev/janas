/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * chat.h - the objects behind the public handles of janas/llm.h and what
 * the files of the API share: api.c (the model, tokens), api_chat.c (the
 * conversation), api_reply.c (the reply). Private.
 */
#ifndef JANAS_LLM_CHAT_H
#define JANAS_LLM_CHAT_H

#include <stddef.h>
#include <stdint.h>

#include "janas/llm.h"
#include "llm/expert_cache.h"
#include "llm/constrain.h"
#include "llm/generate.h"
#include "llm/json.h"
#include "llm/model.h"
#include "llm/tokenizer.h"
#include "llm/tools.h"

enum chat_format {
    FORMAT_CHATML, /* <|im_start|>role\ntext<|im_end|>\n */
};

struct janas_llm {
    struct janas_llm_model *m;
    struct janas_tokenizer *tok;
    int format;
    int32_t im_start, im_end;
    int32_t think_open, think_close; /* reasoning markers, or -1 */
    int thinker;                     /* the template knows enable_thinking */
    enum janas_tool_dialect tools;   /* how its template writes tool calls */
    int32_t call_open, call_close;   /* <tool_call>, </tool_call>, or -1 */
    int32_t resp_open, resp_close;   /* <tool_response>, </tool_response> */
    uint32_t pooling; /* embeddings: 1 mean, 2 first, 3 last; 0 none */
    int add_eos;      /* the tokenizer ends every text with eos_id */
    int32_t eos_id;
    int32_t stop[JANAS_GEN_MAX_STOP];
    int n_stop;
    int has_mtp;
    struct janas_llm_model *draft; /* the drafting model, or NULL */
    uint32_t n_ctx;
    struct janas_llm_chat *chat;
    char desc[1024];
    /* what the description is built from, so that it can be built again:
       whether the GPU is given work is a thing that changes */
    uint64_t cache_bytes, reserve_bytes;
    int n_compute;
    char arch[32], bits[96], warm[64];
    char name[96];
    int32_t mode;
    int gpu_wanted;    /* the caller's say, on top of what the mode decides */
    uint64_t kv_token; /* bytes of attention's keys and values per token */
};

struct janas_llm_chat {
    janas_llm *llm;
    struct janas_llm_session *s;
    struct janas_llm_chat_params p;
    char *system;
    int replying; /* a reply is being read */
    int closed;   /* the last reply ended with a stop token */
    int ended;    /* the reply is over: the next call says DONE */
    int in_think; /* the reply is in its reasoning part */
    uint32_t reply_tokens;
    uint32_t answer_tokens; /* of them, outside the reasoning */
    char pend[1024];        /* reply bytes not yet returned */
    size_t npend;
    /* where each turn starts in the sequence, so that sliding the context
       forgets whole exchanges, and how long the system message is */
    uint32_t *turn;
    size_t n_turn, cap_turn;
    uint32_t sys_tokens;
    uint64_t dropped; /* tokens forgotten in this conversation */
    /* the message being built: ids, and plain text not yet tokenized */
    int32_t *ids;
    size_t n_ids, cap_ids;
    char *text;
    size_t n_text, cap_text;
    /* counters */
    struct janas_gen_stats st0;
    struct janas_expert_cache_stats cs0;
    double t_send;
    struct janas_llm_chat_stats stats;
    /* the stages of the reply and the reading of its prompt (api_reply.c):
       tokens computed when it began, when building its prompt began, the
       processor time of the reading, when its first token came; the prompt
       kept once read (a model with a recurrent state, api_keep.c); where
       the last message begins in the sequence, and how many tokens of the
       system message are the tools' */
    uint32_t kv_from;
    double t_prep0, input_cpu, t_first;
    /* the prompt's reading, block by block, for the time left: the steps
       read and the sums of a weighted least-squares line of a token's
       cost (seconds) against its position (api_reply.c) */
    uint32_t eta_steps;
    double eta_w, eta_sx, eta_sc, eta_sxx, eta_sxc, eta_last;
    int keep_prompt;
    uint32_t last_at, tools_tokens;
    /* the metrics (metrics.c): the reply's number, the last progress told */
    unsigned long m_seq;
    double m_last;
    uint32_t m_last_done;
    /*
     * The last replies, remembered with their reasoning: a client that sends
     * the whole conversation back (an HTTP client does, at every request)
     * returns only the answer, and the answer alone would tokenize into a
     * turn the model never saw. A reply found here is put back as it was
     * generated, so the sequence stays the one already computed.
     */
    struct memo {
        char *text; /* the answer, without the reasoning, and the calls
                       (janas_api_memo_key) */
        size_t n_text;
        int32_t *ids; /* the whole turn, from its first special token */
        uint32_t n;
        int closed; /* it ended with a stop token */
    } memo[8];
    int memo_next;
    uint32_t reply_at; /* where the turn of the reply begins; UINT32_MAX:
                          not a turn (raw text) */
    char *rtext;       /* the answer of the reply being read */
    size_t n_rtext, cap_rtext;

    /* the conversation of janas_llm_chat_begin and _add, not yet run */
    struct cmsg {
        int32_t role;
        char *text;
        size_t n;
        struct janas_tool_call *calls; /* name and args owned here */
        size_t n_calls;
    } *msgs;
    size_t n_msgs, cap_msgs;

    /* tools, and the form of the answer (api_tools.c) */
    struct janas_toolset *tools;
    int32_t tool_choice, tool_only, parallel;
    int32_t format;
    struct janas_json_doc *schema;
    struct janas_grammar *g;       /* of the reply being read, with ... */
    struct janas_constraint *cons; /* ... what holds it to it */

    /* the reply's tool calls: the one being written, and those closed */
    int in_call;
    struct janas_buf body;
    struct rcall {
        char *name, *args;
    } *calls;
    size_t n_calls, cap_calls;

    /* the reply's tokens, with their bytes and what was learned of them */
    struct rtok {
        int32_t id, part, n_top;
        float lp;
        uint32_t off, n; /* bytes in tbytes */
    } *toks;
    size_t n_toks, cap_toks;
    struct janas_buf tbytes;
    int32_t *top_id; /* JANAS_SAMPLE_MAX_TOP per token, when kept */

    /* conversations kept computed besides the current one (api_keep.c) */
    struct kept {
        struct janas_llm_saved *sv;
        uint64_t used; /* when it was last current, on keep_clock */
        int prefix;    /* a system message alone, on a model with a
                          recurrent state: not dropped for being held
                          whole by another (api_keep.c) */
    } *kept;
    int32_t n_kept, keep_max;
    uint64_t keep_bytes, kept_bytes, keep_clock;
    /* and on disk, below them (api_keep_disk.c): what each file holds */
    struct kept_file {
        char name[40];
        int32_t *tok;
        uint32_t n, n_kv;
        uint64_t bytes;
        int64_t used; /* the file's time, ns */
        int prefix;
    } *disk;
    int32_t n_disk, cap_disk;
    uint64_t disk_max, disk_bytes;
    char *disk_dir;
    uint32_t loaded; /* tokens the last conversation loaded had, before its
                        reply */
    float *top_lp;
    size_t top_cap; /* tokens they have room for */
};

/* The last error, on this thread; returns code. */
int32_t janas_api_fail(int32_t code, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
double janas_api_now(void);
/* The power mode in effect: auto follows the power source. */
void janas_api_apply_mode(janas_llm *llm);
/* A string into the caller's buffer, NUL-terminated; *len the full length. */
int32_t janas_api_copy_out(const char *s, size_t n, char *buf, int32_t cap,
                           int32_t *len);
size_t janas_api_text_len(const char *text, int32_t len);
int janas_api_is_stop(const janas_llm *llm, int32_t t);
/* What identifies an assistant's turn among the remembered ones: its
   answer, trimmed, and its calls. */
void janas_api_memo_key(struct janas_buf *b, const char *text, size_t n,
                        const struct janas_tool_call *calls, size_t n_calls);
/* The reply that just ended, remembered with its reasoning. */
void janas_api_memo_store(janas_llm_chat *c);
/* Bytes of the answer being read (not the reasoning). */
int janas_api_rtext_add(janas_llm_chat *c, const char *s, size_t n);
/* Building a message (api_chat.c): plain text accumulates, a special token
   flushes it into ids. The rest writes whole parts of the format. Each
   returns 0, or -1 on memory. */
int janas_api_add_text(janas_llm_chat *c, const char *s, size_t n);
int janas_api_flush_text(janas_llm_chat *c);
int janas_api_add_special(janas_llm_chat *c, int32_t id);
int janas_api_add_message(janas_llm_chat *c, const char *role, const char *text,
                          size_t n);
int janas_api_add_system(janas_llm_chat *c, const char *sys, size_t n);
int janas_api_start_reply(janas_llm_chat *c);
int janas_api_note_turn(janas_llm_chat *c, uint32_t at);
/* The reply starts: reply_at is where its turn begins in the sequence
   (UINT32_MAX for raw text), total the prompt's length, cached how much of
   it was computed already. 0, or -1 on memory. */
int janas_api_begin_reply(janas_llm_chat *c, uint32_t reply_at, uint32_t total,
                          uint32_t cached);
/* Building a prompt begins (its time is prepare_seconds). */
void janas_api_prep_start(janas_llm_chat *c);
/* A prompt of total tokens against max_input: 0, or JANAS_LLM_ELIMIT with
   its size and parts in the stats and said. last_at is set. */
int32_t janas_api_check_input(janas_llm_chat *c, uint32_t total);
/* The prompt's parts into the stats, for total tokens. */
void janas_api_parts(janas_llm_chat *c, uint32_t total);
/* The counters of the reply brought up to date (api_reply.c). */
void janas_api_update_stats(janas_llm_chat *c);
/* Text of the chat format's making, where the markers of reasoning and
   tools ("<tool_call>" ...) are their special tokens. */
int janas_api_add_markup(janas_llm_chat *c, const char *s, size_t n);

/* api_tools.c: what holds the reply that starts now (0, or -1 on memory),
   and the end of it; the reply's calls and tokens forgotten. */
int janas_api_reply_setup(janas_llm_chat *c);
void janas_api_reply_clear(janas_llm_chat *c);
/* Everything the chat holds for tools, at its end. */
void janas_api_tools_free(janas_llm_chat *c);
/* A token of the reply, recorded with what was learned of it (part:
   JANAS_LLM_PART_...); 0, or -1 on memory. */
int janas_api_record(janas_llm_chat *c, int32_t t, int32_t part,
                     const char *bytes, size_t n);
/* A call's closing token: the call written since its opening one becomes
   a call of the reply (or, not being one, text: returned in *text). */
void janas_api_close_call(janas_llm_chat *c, struct janas_buf *text);

/* api_load.c: the messages of janas_llm_chat_add, at the chat's end. */
void janas_api_msgs_free(janas_llm_chat *c);

/* api_keep.c: before a conversation (c->ids) is loaded over the current
   one, puts in its place the kept one that shares most of it, keeping the
   current one when it would lose much; and at the end frees them all,
   writing them to the disk first when there is one. */
void janas_api_keep_pick(janas_llm_chat *c);
/* api_keep.c: after it is loaded, the prompt of a model with a recurrent
   state is computed and kept, for a second reply to it. */
void janas_api_keep_prompt(janas_llm_chat *c);
/* api_keep.c: on a model with a recurrent state, the system message of a
   conversation about to be read from `from` is computed first and kept;
   returns where the rest is to be read from. */
uint32_t janas_api_keep_system(janas_llm_chat *c, uint32_t from);
/* api_keep.c: what a copy of hn tokens, n_kv computed, spares of c->ids */
uint32_t janas_api_keep_reuse(const int32_t *h, uint32_t hn, uint32_t n_kv,
                              const int32_t *ids, size_t n_ids, int recurrent);

/* api_keep_disk.c: the files. best: the one that spares most of c->ids,
   more than floor (-1: none; *r how much). take: that file read whole
   (NULL: it could not be, and it is gone). put: a copy written, if long
   enough and not there already. close: the index freed (files stay). */
int32_t janas_api_disk_best(janas_llm_chat *c, uint32_t floor, uint32_t *r);
struct janas_llm_saved *janas_api_disk_take(janas_llm_chat *c, int32_t i,
                                            int *prefix);
void janas_api_disk_put(janas_llm_chat *c, const struct janas_llm_saved *sv,
                        int prefix);
void janas_api_disk_close(janas_llm_chat *c);
void janas_api_keep_free(janas_llm_chat *c);

#endif
