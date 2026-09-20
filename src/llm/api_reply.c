/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_reply.c - the public API: the reply, piece by piece, and its counters
 * (see include/janas/llm.h).
 */
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "llm/chat.h"
#include "llm/metrics.h"
#include "llm/model.h"

int janas_api_is_stop(const janas_llm *llm, int32_t t)
{
    for (int i = 0; i < llm->n_stop; i++)
        if (t == llm->stop[i])
            return 1;
    return 0;
}

/* Bytes of pend that form whole UTF-8 characters, at most max. */
static size_t whole_chars(const char *s, size_t n, size_t max)
{
    size_t i = 0;
    while (i < n) {
        unsigned char b = (unsigned char)s[i];
        size_t k = b < 0x80         ? 1
                   : (b >> 5) == 6  ? 2
                   : (b >> 4) == 14 ? 3
                   : (b >> 3) == 30 ? 4
                                    : 1;
        if (i + k > n || i + k > max)
            break;
        i += k;
    }
    return i;
}

static double cpu_now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/*
 * The time left to read the prompt: a token's cost taken as a line in its
 * position, a + b p (attention grows with the context it looks at), fitted
 * on the blocks read so far and integrated over the positions left. On a
 * 260,954-token prompt of Qwen3-Next-80B the elapsed time followed a n +
 * b n^2 to R^2 0.99999 (issue #9), where the latest speed promised 2.5
 * times too little at 32K. Never below what the last block's speed says:
 * the cost does not fall as the context grows.
 */
static double input_eta(const janas_llm_chat *c, uint32_t left)
{
    if (left == 0)
        return 0;
    if (c->eta_last < 0)
        return -1;
    double flat = c->eta_last * left;
    double w = c->eta_w, det = w * c->eta_sxx - c->eta_sx * c->eta_sx;
    if (c->eta_steps < 4 || det <= 0)
        return flat;
    double b = (w * c->eta_sxc - c->eta_sx * c->eta_sc) / det;
    double a = (c->eta_sc - b * c->eta_sx) / w;
    double p0 = janas_llm_session_computed(c->s), p1 = p0 + left;
    double fit = a * (p1 - p0) + 0.5 * b * (p1 * p1 - p0 * p0);
    return fit > flat ? fit : flat;
}

/* One block of the prompt read: n tokens from position p0, in dt seconds.
   The first two blocks are left out: they warm the caches and the tuner. */
static void eta_note(janas_llm_chat *c, uint32_t p0, uint32_t n, double dt)
{
    if (n == 0 || dt <= 0)
        return;
    double cost = dt / n, x = p0 + 0.5 * n;
    c->eta_last = cost;
    if (++c->eta_steps <= 2)
        return;
    c->eta_w += n;
    c->eta_sx += n * x;
    c->eta_sc += n * cost;
    c->eta_sxx += n * x * x;
    c->eta_sxc += n * x * cost;
}

void janas_api_update_stats(janas_llm_chat *c)
{
    struct janas_gen_stats st;
    janas_llm_session_stats(c->s, &st);
    struct janas_llm_chat_stats *s = &c->stats;
    s->size = sizeof(*s);
    s->output_tokens = c->reply_tokens;
    s->context_used = janas_llm_session_length(c->s);
    s->context_size = c->llm->n_ctx;
    s->drafted = st.drafted - c->st0.drafted;
    s->accepted = st.accepted - c->st0.accepted;
    s->auto_off_passes = st.auto_off_passes - c->st0.auto_off_passes;
    s->input_seconds = st.prefill_seconds - c->st0.prefill_seconds;
    s->output_seconds = st.decode_seconds - c->st0.decode_seconds;
    s->total_seconds = janas_api_now() - c->t_send;
    struct janas_expert_cache_stats cs;
    janas_expert_cache_stats(janas_llm_model_cache(c->llm->m), &cs);
    s->experts_used = cs.requests - c->cs0.requests;
    s->experts_read = cs.misses - c->cs0.misses;
    s->bytes_read = cs.bytes_read - c->cs0.bytes_read;
    s->io_wait_seconds = cs.wait_seconds - c->cs0.wait_seconds;
    /* the reading of the prompt, and the rest of what a reply costs */
    uint32_t done = janas_llm_session_computed(c->s) - c->kv_from;
    if (janas_llm_session_computed(c->s) < c->kv_from)
        done = 0;
    s->input_done = s->stage == JANAS_LLM_STAGE_INPUT && done < s->input_tokens
                        ? done
                    : s->stage == JANAS_LLM_STAGE_IDLE ? 0
                                                       : s->input_tokens;
    s->first_token_seconds = c->t_first;
    s->input_cpu_seconds = c->input_cpu;
    s->input_threads = janas_llm_model_threads(c->llm->m, 2);
    s->output_threads = janas_llm_model_threads(c->llm->m, 0);
    s->context_bytes = c->llm->kv_token * s->context_used;
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        s->peak_rss_bytes = (uint64_t)ru.ru_maxrss * 1024;
    s->input_eta_seconds = s->stage == JANAS_LLM_STAGE_INPUT
                               ? input_eta(c, s->input_tokens - s->input_done)
                               : 0;
}

/*
 * A block of the prompt read: 1 when more is left, 0 when it has all been
 * read (the prompt is kept then, where it is to be), -1 on an error.
 */
static int read_input(janas_llm_chat *c)
{
    double c0 = cpu_now(), t0 = janas_api_now();
    uint32_t p0 = janas_llm_session_computed(c->s);
    int r = janas_llm_session_prefill_step(c->s);
    c->input_cpu += cpu_now() - c0;
    uint32_t p1 = janas_llm_session_computed(c->s);
    if (r >= 0 && p1 > p0)
        eta_note(c, p0, p1 - p0, janas_api_now() - t0);
    if (r == 0 && c->keep_prompt) {
        c->keep_prompt = 0;
        janas_api_keep_prompt(c); /* read already: only kept */
    }
    return r;
}

/*
 * Returns up to cap bytes of whole characters from pend. Once the reply has
 * ended, a character the last token left unfinished (max_reply can stop a
 * reply between the tokens of an emoji) will never be finished: its bytes
 * are dropped, since what comes out is promised to be whole characters, and
 * a client that decodes it would fail on them.
 */
static size_t emit(janas_llm_chat *c, char *buf, size_t cap)
{
    size_t w = whole_chars(c->pend, c->npend, cap);
    memcpy(buf, c->pend, w);
    memmove(c->pend, c->pend + w, c->npend - w);
    c->npend -= w;
    if (c->ended && w == 0 && c->npend < 4)
        c->npend = 0; /* a character left half written */
    return w;
}

/* Bytes for the caller, as many as pend has room for. */
static void give(janas_llm_chat *c, const char *s, size_t n)
{
    if (n > sizeof(c->pend) - c->npend)
        n = sizeof(c->pend) - c->npend;
    memcpy(c->pend + c->npend, s, n);
    c->npend += n;
}

/*
 * A token of the reply, where it belongs. The reasoning markers are not
 * part of the text: the opening one is written into the prompt and never
 * generated, so letting the closing one through printed half a pair; and
 * where a model writes both, the colour of the reasoning says the same
 * thing twice. Either both or neither: neither. A tool call, from its
 * opening token to its closing one, is not text either: it is read into a
 * call of the reply.
 */
static void take(janas_llm_chat *c, int32_t t)
{
    const janas_llm *llm = c->llm;
    char tmp[sizeof(c->pend)];
    if (!c->in_think && t != llm->think_open && t != llm->think_close)
        c->answer_tokens++; /* text or a call: what max_answer counts */
    if (t == llm->think_open || t == llm->think_close) {
        c->in_think = t == llm->think_open;
        janas_api_record(c, t, JANAS_LLM_PART_MARK, "", 0);
        return;
    }
    if (c->tools && t == llm->call_open && !c->in_think && !c->in_call) {
        c->in_call = 1;
        c->body.n = 0;
        janas_api_record(c, t, JANAS_LLM_PART_MARK, "", 0);
        return;
    }
    if (c->in_call && t == llm->call_close) {
        janas_api_record(c, t, JANAS_LLM_PART_MARK, "", 0);
        struct janas_buf text = {0};
        janas_api_close_call(c, &text);
        if (text.n) {
            janas_api_rtext_add(c, text.p, text.n);
            give(c, text.p, text.n);
        }
        janas_buf_free(&text);
        return;
    }
    size_t n = janas_tokenizer_decode(llm->tok, t, tmp, sizeof(tmp));
    if (n > sizeof(tmp))
        n = sizeof(tmp);
    if (c->in_call) {
        janas_buf_put(&c->body, tmp, n);
        janas_api_record(c, t, JANAS_LLM_PART_CALL, tmp, n);
        return;
    }
    janas_api_record(
        c, t, c->in_think ? JANAS_LLM_PART_REASONING : JANAS_LLM_PART_ANSWER,
        tmp, n);
    if (!c->in_think)
        janas_api_rtext_add(c, tmp, n); /* the answer, for memo_store */
    give(c, tmp, n);
}

int32_t janas_llm_chat_next(janas_llm_chat *c, char *buf, int32_t cap,
                            int32_t *len)
{
    if (!c || !buf || !len || cap < 8)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument (cap >= 8)");
    *len = 0;
    if (!c->replying)
        return JANAS_LLM_DONE;
    /* the prompt, a block at a time: the caller hears between blocks */
    if (c->stats.stage == JANAS_LLM_STAGE_INPUT) {
        int r = read_input(c);
        if (r > 0) {
            janas_api_update_stats(c);
            janas_metrics_progress(c, 0);
            return JANAS_LLM_OK;
        }
        c->stats.stage = JANAS_LLM_STAGE_OUTPUT;
        if (r < 0) {
            c->replying = 0;
            c->stats.finish = JANAS_LLM_FINISH_ERROR;
            c->stats.stage = JANAS_LLM_STAGE_DONE;
            janas_api_update_stats(c);
            janas_metrics_reply(c);
            return janas_api_fail(JANAS_LLM_EFAIL, "generation failed");
        }
        janas_api_update_stats(c);
        if (c->stats.input_tokens >= 1024)
            janas_metrics_progress(c, 1); /* the whole prompt, read */
    }
    if (!c->ended && c->npend < 4) {
        int32_t t = -1;
        int r = 0, got = 0;
        if ((c->p.max_reply > 0 &&
             c->reply_tokens >= (uint32_t)c->p.max_reply) ||
            (c->p.max_answer > 0 &&
             c->answer_tokens >= (uint32_t)c->p.max_answer)) {
            c->ended = 1;
            c->stats.finish = JANAS_LLM_FINISH_LENGTH;
        } else {
            r = janas_llm_session_next(c->s, &t);
            got = r == 0;
        }
        if (got && c->t_first == 0) { /* the first token of the reply */
            c->t_first = janas_api_now() - c->t_send;
            if (c->t_first <= 0)
                c->t_first = 1e-6;
            c->stats.first_token_seconds = c->t_first;
            janas_metrics_first_token(c);
        }
        if (r != 0) {
            c->ended = 1;
            c->stats.finish =
                r > 0 ? JANAS_LLM_FINISH_CONTEXT : JANAS_LLM_FINISH_ERROR;
        } else if (got && janas_api_is_stop(c->llm, t)) {
            c->ended = c->closed = 1;
            c->stats.finish =
                c->n_calls ? JANAS_LLM_FINISH_TOOLS : JANAS_LLM_FINISH_STOP;
        } else if (got) {
            c->reply_tokens++;
            take(c, t);
        }
        janas_api_update_stats(c);
        if (r != 0) {
            c->replying = 0;
            c->stats.stage = JANAS_LLM_STAGE_DONE;
            janas_metrics_reply(c);
            return r > 0 ? janas_api_fail(JANAS_LLM_EFULL,
                                          "the context is full (%u "
                                          "tokens)",
                                          c->llm->n_ctx)
                         : janas_api_fail(JANAS_LLM_EFAIL, "generation failed");
        }
    }
    *len = (int32_t)emit(c, buf, (size_t)cap);
    if (c->ended && c->npend == 0 && *len == 0) {
        c->replying = 0;
        c->stats.stage = JANAS_LLM_STAGE_DONE;
        janas_api_update_stats(c);
        janas_metrics_reply(c);
        janas_api_memo_store(c);
        return JANAS_LLM_DONE;
    }
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_stats(const janas_llm_chat *c,
                             struct janas_llm_chat_stats *s)
{
    if (!c || !s || s->size < sizeof(uint32_t))
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    uint32_t size = s->size < sizeof(c->stats) ? s->size : sizeof(c->stats);
    memcpy(s, &c->stats, size);
    s->size = size;
    return JANAS_LLM_OK;
}
