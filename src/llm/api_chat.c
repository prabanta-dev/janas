/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_chat.c - the public API: a conversation, built message by message in
 * the model's chat format, and the reply started (see include/janas/llm.h).
 * A message is turned into tokens the way the rendered conversation would
 * be: the format's special tokens as ids, the text between them tokenized as
 * one run (so the user's text never produces special tokens).
 */
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llm/chat.h"
#include "llm/metrics.h"

void janas_llm_chat_params_default(struct janas_llm_chat_params *p)
{
    memset(p, 0, sizeof(*p));
    p->size = sizeof(*p);
    p->temperature = 0.7f;
    p->top_k = 20;
    p->top_p = 0.8f;
    p->speculate = 1;
    p->thinking = -1;
    p->recap = 1;
}

static int32_t gen_options(const janas_llm_chat *c, struct janas_gen_options *o)
{
    const struct janas_llm_chat_params *p = &c->p;
    if (!(p->temperature >= 0) || p->top_k < 0 || !(p->top_p >= 0) ||
        !(p->min_p >= 0) || p->max_reply < 0 || p->top_logprobs < 0 ||
        p->top_logprobs > JANAS_SAMPLE_MAX_TOP ||
        !(p->presence_penalty >= -2 && p->presence_penalty <= 2) ||
        !(p->frequency_penalty >= -2 && p->frequency_penalty <= 2))
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid chat parameters");
    memset(o, 0, sizeof(*o));
    int mtp = c->llm->has_mtp;
    o->spec_mode = !p->speculate   ? JANAS_SPEC_OFF
                   : c->llm->draft ? JANAS_SPEC_DRAFT
                   : mtp           ? JANAS_SPEC_MTP
                                   : JANAS_SPEC_LOOKUP;
    o->draft = c->llm->draft;
    /*
     * The ceiling on how many tokens may be proposed, not how many will be:
     * a planner picks the number from the cost of a longer pass and the rate
     * at which the drafts in each position have been surviving. Until 22 Sep
     * 2026 the drafts copied from the context had no planner and always
     * asked for four, which on ordinary prose was slower than not drafting
     * at all - 0.89x on Qwen3-4B, 0.82x on Qwen3-30B - because acceptance
     * falls to about one in nine and a failed verification costs a pass over
     * five tokens.
     */
    o->spec_k = mtp ? 8 : 4;
    o->spec_auto = 1;
    o->eos = -1;
    o->temperature = p->temperature;
    o->top_k = p->top_k;
    o->top_p = p->top_p;
    o->min_p = p->min_p;
    o->presence_penalty = p->presence_penalty;
    o->frequency_penalty = p->frequency_penalty;
    o->logprobs = p->logprobs != 0;
    o->top_logprobs = p->top_logprobs;
    o->seed = p->seed ? p->seed
                      : (uint64_t)time(NULL) * 0x9e3779b97f4a7c15ull ^
                            (uint64_t)(uintptr_t)c;
    memcpy(o->stop, c->llm->stop, sizeof(o->stop));
    o->n_stop = c->llm->n_stop;
    return JANAS_LLM_OK;
}

static int32_t take_params(janas_llm_chat *c,
                           const struct janas_llm_chat_params *pp)
{
    struct janas_llm_chat_params p;
    janas_llm_chat_params_default(&p);
    if (pp) {
        if (pp->size < sizeof(uint32_t) || pp->size > sizeof(p))
            return janas_api_fail(JANAS_LLM_EINVAL, "parameters: unknown size");
        memcpy(&p, pp, pp->size);
        p.size = sizeof(p);
    }
    struct janas_llm_chat_params old = c->p;
    c->p = p;
    struct janas_gen_options o;
    int32_t rc = gen_options(c, &o);
    if (rc == 0 && c->s && janas_llm_session_set_options(c->s, &o) != 0)
        rc = janas_api_fail(JANAS_LLM_EINVAL, "invalid chat parameters");
    if (rc != 0)
        c->p = old;
    return rc;
}

int32_t janas_llm_chat_create(janas_llm *llm,
                              const struct janas_llm_chat_params *p,
                              janas_llm_chat **out)
{
    if (!llm || !out)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model or no output handle");
    *out = NULL;
    if (llm->chat)
        return janas_api_fail(JANAS_LLM_EBUSY, "the model already has a chat");
    janas_llm_chat *c = calloc(1, sizeof(*c));
    if (!c)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    c->llm = llm;
    int32_t rc = take_params(c, p);
    struct janas_gen_options o;
    if (rc == 0 && (rc = gen_options(c, &o)) == 0 &&
        !(c->s = janas_llm_session_create(llm->m, &o)))
        rc = janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    if (rc != 0) {
        free(c);
        return rc;
    }
    llm->chat = c;
    *out = c;
    return JANAS_LLM_OK;
}

void janas_llm_chat_destroy(janas_llm_chat *c)
{
    if (!c)
        return;
    c->llm->chat = NULL;
    /* the tools first: the session holds a filter bound to them */
    janas_api_tools_free(c);
    janas_api_msgs_free(c);
    janas_api_keep_free(c); /* before the session: it may be written */
    janas_llm_session_destroy(c->s);
    free(c->system);
    free(c->ids);
    free(c->text);
    free(c->turn);
    for (size_t i = 0; i < sizeof(c->memo) / sizeof(c->memo[0]); i++) {
        free(c->memo[i].text);
        free(c->memo[i].ids);
    }
    free(c->rtext);
    free(c);
}

int32_t janas_llm_chat_set_params(janas_llm_chat *c,
                                  const struct janas_llm_chat_params *p)
{
    if (!c || !p)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    return take_params(c, p);
}

int32_t janas_llm_chat_system(janas_llm_chat *c, const char *text, int32_t len)
{
    if (!c || !text)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    if (janas_llm_session_length(c->s) > 0)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "the conversation has started: reset it first");
    size_t n = janas_api_text_len(text, len);
    char *s = malloc(n + 1);
    if (!s)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    memcpy(s, text, n);
    s[n] = 0;
    free(c->system);
    c->system = s;
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_reset(janas_llm_chat *c)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    janas_llm_session_reset(c->s);
    free(c->system);
    c->system = NULL;
    c->n_turn = 0;
    c->sys_tokens = 0;
    c->tools_tokens = 0;
    c->dropped = 0;
    c->replying = c->closed = c->ended = 0;
    c->npend = 0;
    return JANAS_LLM_OK;
}

/* Message building: plain text accumulates, a special token flushes it. */
int janas_api_add_text(janas_llm_chat *c, const char *s, size_t n)
{
    if (c->n_text + n > c->cap_text) {
        size_t cap = 2 * (c->n_text + n) + 256;
        char *t = realloc(c->text, cap);
        if (!t)
            return -1;
        c->text = t;
        c->cap_text = cap;
    }
    memcpy(c->text + c->n_text, s, n);
    c->n_text += n;
    return 0;
}

int janas_api_flush_text(janas_llm_chat *c)
{
    if (c->n_text == 0)
        return 0;
    long need =
        janas_tokenizer_encode(c->llm->tok, c->text, c->n_text, 0, NULL, 0);
    if (need < 0)
        return -1;
    if (c->n_ids + (size_t)need + 1 > c->cap_ids) {
        size_t cap = 2 * (c->n_ids + (size_t)need) + 64;
        int32_t *t = realloc(c->ids, cap * sizeof(int32_t));
        if (!t)
            return -1;
        c->ids = t;
        c->cap_ids = cap;
    }
    janas_tokenizer_encode(c->llm->tok, c->text, c->n_text, 0,
                           c->ids + c->n_ids, (size_t)need);
    c->n_ids += (size_t)need;
    c->n_text = 0;
    return 0;
}

int janas_api_add_special(janas_llm_chat *c, int32_t id)
{
    if (janas_api_flush_text(c) != 0)
        return -1;
    if (c->n_ids + 1 > c->cap_ids) {
        size_t cap = 2 * c->n_ids + 64;
        int32_t *t = realloc(c->ids, cap * sizeof(int32_t));
        if (!t)
            return -1;
        c->ids = t;
        c->cap_ids = cap;
    }
    c->ids[c->n_ids++] = id;
    return 0;
}

/* One message of the conversation, in the model's format. */
int janas_api_add_message(janas_llm_chat *c, const char *role, const char *text,
                          size_t n)
{
    const janas_llm *llm = c->llm;
    switch (llm->format) {
    case FORMAT_CHATML:
    default:
        return janas_api_add_special(c, llm->im_start) ||
               janas_api_add_text(c, role, strlen(role)) ||
               janas_api_add_text(c, "\n", 1) ||
               janas_api_add_text(c, text, n) ||
               janas_api_add_special(c, llm->im_end) ||
               janas_api_add_text(c, "\n", 1);
    }
}

int janas_api_start_reply(janas_llm_chat *c)
{
    const janas_llm *llm = c->llm;
    switch (llm->format) {
    case FORMAT_CHATML:
    default:
        if (janas_api_add_special(c, llm->im_start) ||
            janas_api_add_text(c, "assistant\n", 10))
            return -1;
        /* a reasoning model: open the reasoning, or pass it empty */
        c->in_think = 0;
        if (llm->thinker) {
            if (c->p.thinking != 0) {
                c->in_think = 1;
                return janas_api_add_special(c, llm->think_open) ||
                       janas_api_add_text(c, "\n", 1) ||
                       janas_api_flush_text(c);
            }
            return janas_api_add_special(c, llm->think_open) ||
                   janas_api_add_text(c, "\n\n", 2) ||
                   janas_api_add_special(c, llm->think_close) ||
                   janas_api_add_text(c, "\n\n", 2) || janas_api_flush_text(c);
        }
        return janas_api_flush_text(c);
    }
}

/*
 * Text the chat format writes (never the user's): the markers of the
 * reasoning and of the tools become their tokens, as they are in the
 * conversations the model was trained on. "<tool_call>" inside the system
 * message's instructions is one token there, not eleven bytes.
 */
int janas_api_add_markup(janas_llm_chat *c, const char *s, size_t n)
{
    const janas_llm *llm = c->llm;
    const struct {
        const char *text;
        int32_t id;
    } marks[] = {{"<tool_call>", llm->call_open},
                 {"</tool_call>", llm->call_close},
                 {"<tool_response>", llm->resp_open},
                 {"</tool_response>", llm->resp_close},
                 {"<think>", llm->think_open},
                 {"</think>", llm->think_close}};
    size_t from = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] != '<')
            continue;
        for (size_t k = 0; k < sizeof(marks) / sizeof(marks[0]); k++) {
            size_t m = strlen(marks[k].text);
            if (marks[k].id < 0 || n - i < m || memcmp(s + i, marks[k].text, m))
                continue;
            if (janas_api_add_text(c, s + from, i - from) ||
                janas_api_add_special(c, marks[k].id))
                return -1;
            i += m - 1;
            from = i + 1;
            break;
        }
    }
    return janas_api_add_text(c, s + from, n - from);
}

/* The system message, with the tools when there are: the dialect's
   template writes them into it. */
int janas_api_add_system(janas_llm_chat *c, const char *sys, size_t n)
{
    const janas_llm *llm = c->llm;
    c->tools_tokens = 0;
    if (!c->tools || llm->tools == JANAS_TOOLS_NONE)
        return janas_api_add_message(c, "system", sys ? sys : "", n);
    /* how many of its tokens the tools take: the message without them is
       built and taken back, for the count alone */
    size_t ids0 = c->n_ids;
    if (janas_api_flush_text(c) != 0)
        return -1;
    ids0 = c->n_ids;
    if (janas_api_add_message(c, "system", sys ? sys : "", n) != 0 ||
        janas_api_flush_text(c) != 0)
        return -1;
    size_t plain = c->n_ids - ids0;
    c->n_ids = ids0;
    struct janas_buf b = {0};
    janas_tools_system(&b, llm->tools, c->tools, sys, n);
    int err = b.oom || janas_api_add_special(c, llm->im_start) ||
              janas_api_add_text(c, "system\n", 7) ||
              janas_api_add_markup(c, b.p, b.n) ||
              janas_api_add_special(c, llm->im_end) ||
              janas_api_add_text(c, "\n", 1) || janas_api_flush_text(c);
    janas_buf_free(&b);
    if (!err && c->n_ids - ids0 > plain)
        c->tools_tokens = (uint32_t)(c->n_ids - ids0 - plain);
    return err ? -1 : 0;
}

int32_t janas_llm_chat_thinking(const janas_llm_chat *c)
{
    return c && c->in_think;
}

int32_t janas_llm_reasons(const janas_llm *llm)
{
    return llm ? llm->thinker : 0;
}

/*
 * Before the oldest exchanges go, the model is asked for a short recap of
 * the conversation. It is one more turn, so it stays in the window when the
 * older ones are dropped, and the thread of the conversation with it. A few
 * seconds, once, when the context fills.
 */

static const char *const RECAP_ASK =
    "Before we go on: sum up our conversation so far in about 100 words, "
    "keeping the names, the facts and anything you were asked to remember. "
    "Write only the summary.";

static int recap(janas_llm_chat *c)
{
    int32_t *keep = NULL;
    size_t n_keep = c->n_ids;
    if (n_keep) { /* the user's message is already built: set it aside */
        keep = malloc(n_keep * sizeof(int32_t));
        if (!keep)
            return -1;
        memcpy(keep, c->ids, n_keep * sizeof(int32_t));
    }
    c->n_ids = c->n_text = 0;
    int err = 0;
    if (!c->closed)
        err |= janas_api_add_special(c, c->llm->im_end);
    err |= janas_api_add_text(c, "\n", 1);
    err |= janas_api_add_message(c, "user", RECAP_ASK, strlen(RECAP_ASK));
    err |= janas_api_start_reply(c);
    if (!err && janas_llm_session_append(c->s, c->ids, (uint32_t)c->n_ids) == 0)
        for (int i = 0; i < 400; i++) { /* the recap, token by token */
            int32_t t;
            if (janas_llm_session_next(c->s, &t) != 0 ||
                janas_api_is_stop(c->llm, t))
                break;
        }
    else
        err = 1;
    c->closed = 0; /* the recap was cut off or ended: the next turn closes it */
    c->n_ids = 0;
    if (keep) {
        memcpy(c->ids, keep, n_keep * sizeof(int32_t));
        c->n_ids = n_keep;
        free(keep);
    }
    return err ? -1 : 0;
}

/*
 * Makes room for `need` more tokens by forgetting the oldest exchanges: the
 * system message stays, and whole turns go, so what is left reads as a
 * conversation. Returns 0 when there is room, -1 when even that is not
 * enough (a single message longer than the context).
 */
static int make_room(janas_llm_chat *c, uint32_t need)
{
    uint32_t cap = c->llm->n_ctx;
    uint32_t have = (uint32_t)janas_llm_session_length(c->s);
    if (have + need <= cap)
        return 0;
    /* the first tokens stay where they are: attention leans on them, and
       without that anchor the text after a slide drifts much further from
       what the same text would have given (measured with llm_slide: mean
       |logit difference| 2.7 with nothing kept, 0.5 with eight tokens) */
    uint32_t must = have + need - cap;
    uint32_t keep = c->sys_tokens ? c->sys_tokens : (have > 16 ? 8u : 0u);
    size_t k = 0;
    uint32_t drop = 0;
    while (k < c->n_turn) { /* oldest turns first, always whole */
        uint32_t end = c->turn[k] > keep ? c->turn[k] - keep : 0;
        k++;
        if (end > drop) {
            drop = end;
            if (drop >= must)
                break;
        }
    }
    if (drop < must) { /* nothing left to forget but the current turn */
        drop = have > keep ? have - keep : 0;
        k = c->n_turn;
    }
    if (!drop || janas_llm_session_slide(c->s, keep, drop) != 0)
        return -1;
    memmove(c->turn, c->turn + k, (c->n_turn - k) * sizeof(*c->turn));
    c->n_turn -= k;
    for (size_t i = 0; i < c->n_turn; i++)
        c->turn[i] -= drop;
    c->dropped += drop;
    return have + need - drop <= cap ? 0 : -1;
}

/* Records where this turn begins, for a later slide. */
int janas_api_note_turn(janas_llm_chat *c, uint32_t at)
{
    if (c->n_turn == c->cap_turn) {
        size_t cap = c->cap_turn ? 2 * c->cap_turn : 32;
        uint32_t *t = realloc(c->turn, cap * sizeof(*t));
        if (!t)
            return -1;
        c->turn = t;
        c->cap_turn = cap;
    }
    c->turn[c->n_turn++] = at;
    return 0;
}

void janas_api_prep_start(janas_llm_chat *c)
{
    c->t_prep0 = janas_api_now();
    c->last_at = 0;
    c->stats.stage = JANAS_LLM_STAGE_IDLE;
}

void janas_api_parts(janas_llm_chat *c, uint32_t total)
{
    struct janas_llm_chat_stats *s = &c->stats;
    uint32_t sys = c->sys_tokens < total ? c->sys_tokens : total;
    uint32_t last = c->last_at < sys ? sys : c->last_at;
    if (last > total)
        last = total;
    s->prompt_system = sys;
    s->prompt_tools = c->tools_tokens < sys ? c->tools_tokens : sys;
    s->prompt_history = last - sys;
    s->prompt_last = total - last;
    s->prompt_tokens = total;
    s->context_size = c->llm->n_ctx;
}

int32_t janas_api_check_input(janas_llm_chat *c, uint32_t total)
{
    uint32_t max = c->p.max_input;
    if (!max || total <= max)
        return JANAS_LLM_OK;
    memset(&c->stats, 0, sizeof(c->stats));
    janas_api_parts(c, total);
    c->stats.input_tokens = total;
    c->stats.stage = JANAS_LLM_STAGE_IDLE;
    janas_metrics_refused(c, max);
    return janas_api_fail(JANAS_LLM_ELIMIT,
                          "the prompt is %u tokens, above the limit of %u "
                          "(max_input): %u too many",
                          total, max, total - max);
}

/* The reply starts: reply_at is where its turn begins in the sequence
   (UINT32_MAX for raw text), total the length of the prompt, cached how
   much of it was already computed. */
int janas_api_begin_reply(janas_llm_chat *c, uint32_t reply_at, uint32_t total,
                          uint32_t cached)
{
    c->replying = 1;
    c->closed = c->ended = 0;
    c->reply_tokens = 0;
    c->answer_tokens = 0;
    c->npend = 0;
    c->n_rtext = 0;
    c->reply_at = reply_at;
    memset(&c->stats, 0, sizeof(c->stats));
    c->stats.input_tokens = total - cached;
    c->stats.cached_tokens = cached;
    janas_api_parts(c, total);
    /* the prompt is read in janas_llm_chat_next, a block at a time */
    c->stats.stage = JANAS_LLM_STAGE_INPUT;
    c->kv_from = janas_llm_session_computed(c->s);
    c->input_cpu = c->t_first = 0;
    c->eta_steps = 0;
    c->eta_w = c->eta_sx = c->eta_sc = c->eta_sxx = c->eta_sxc = 0;
    c->eta_last = -1;
    c->stats.prepare_seconds =
        c->t_send > c->t_prep0 ? c->t_send - c->t_prep0 : 0;
    janas_llm_session_mark(c->s); /* the penalties count from here */
    int rc = janas_api_reply_setup(c);
    janas_api_update_stats(c);
    janas_metrics_prompt(c);
    return rc;
}

/* The reply that just ended, remembered with its reasoning (see memo). */
void janas_api_memo_key(struct janas_buf *b, const char *text, size_t n,
                        const struct janas_tool_call *calls, size_t n_calls)
{
    while (n && (unsigned char)text[0] <= ' ') {
        text++;
        n--;
    }
    while (n && (unsigned char)text[n - 1] <= ' ')
        n--;
    janas_buf_put(b, text, n);
    for (size_t i = 0; i < n_calls; i++) {
        janas_buf_put(b, "\x1e", 1);
        janas_buf_put(b, calls[i].name, calls[i].name_n);
        janas_buf_put(b, "\x1f", 1);
        janas_buf_put(b, calls[i].args, calls[i].args_n);
    }
}

void janas_api_memo_store(janas_llm_chat *c)
{
    if (c->reply_at == UINT32_MAX)
        return;
    const int32_t *h;
    uint32_t hn = janas_llm_session_tokens(c->s, &h);
    if (c->reply_at >= hn)
        return;
    struct memo *m = &c->memo[c->memo_next];
    uint32_t n = hn - c->reply_at;
    int32_t *ids = malloc(n * sizeof(int32_t));
    struct janas_tool_call *calls =
        malloc((c->n_calls + 1) * sizeof(struct janas_tool_call));
    struct janas_buf key = {0};
    if (calls) {
        for (size_t i = 0; i < c->n_calls; i++)
            calls[i] =
                (struct janas_tool_call){.name = c->calls[i].name,
                                         .name_n = strlen(c->calls[i].name),
                                         .args = c->calls[i].args,
                                         .args_n = strlen(c->calls[i].args)};
        janas_api_memo_key(&key, c->rtext ? c->rtext : "", c->n_rtext, calls,
                           c->n_calls);
        janas_buf_put(&key, "", 0); /* a key of nothing is still a string */
    }
    free(calls);
    if (!ids || !calls || key.oom) {
        free(ids);
        janas_buf_free(&key);
        return;
    }
    memcpy(ids, h + c->reply_at, n * sizeof(int32_t));
    free(m->ids);
    free(m->text);
    *m = (struct memo){.text = key.p,
                       .n_text = key.n,
                       .ids = ids,
                       .n = n,
                       .closed = c->closed};
    c->memo_next =
        (c->memo_next + 1) % (int)(sizeof(c->memo) / sizeof(c->memo[0]));
}

int janas_api_rtext_add(janas_llm_chat *c, const char *s, size_t n)
{
    if (c->n_rtext + n > c->cap_rtext) {
        size_t cap = 2 * (c->n_rtext + n) + 256;
        char *t = realloc(c->rtext, cap);
        if (!t)
            return -1;
        c->rtext = t;
        c->cap_rtext = cap;
    }
    memcpy(c->rtext + c->n_rtext, s, n);
    c->n_rtext += n;
    return 0;
}

/*
 * One more turn of the conversation computed so far: the user's message
 * (texts NULL), or what the tools answered to the calls of the last reply,
 * n answers as one user turn, the way the chat templates write them.
 */
static int32_t send_turn(janas_llm_chat *c, const char *text, size_t n,
                         int32_t n_res, const char *const *texts,
                         const int32_t *lens)
{
    janas_api_prep_start(c);
    janas_api_reply_clear(c);
    c->n_ids = c->n_text = 0;
    int err = 0;
    if (janas_llm_session_length(c->s) > 0) {
        /* close the previous reply: its stop token is in the sequence, or
           it was cut and gets one now; then the newline after it */
        if (!c->closed)
            err |= janas_api_add_special(c, c->llm->im_end);
        err |= janas_api_add_text(c, "\n", 1);
    } else if (c->system || c->tools) {
        err |= janas_api_add_system(c, c->system,
                                    c->system ? strlen(c->system) : 0);
        c->sys_tokens = (uint32_t)c->n_ids;
    }
    err |= janas_api_note_turn(c, (uint32_t)janas_llm_session_length(c->s));
    if (texts) {
        err |= janas_api_add_special(c, c->llm->im_start) ||
               janas_api_add_text(c, "user", 4);
        for (int32_t i = 0; i < n_res; i++) {
            err |= janas_api_add_markup(c, "\n<tool_response>\n", 17);
            err |= janas_api_add_text(
                c, texts[i], janas_api_text_len(texts[i], lens ? lens[i] : -1));
            err |= janas_api_add_markup(c, "\n</tool_response>", 17);
        }
        err |= janas_api_add_special(c, c->llm->im_end) ||
               janas_api_add_text(c, "\n", 1);
    } else {
        err |= janas_api_add_message(c, "user", text, n);
    }
    err |= janas_api_flush_text(c);
    size_t reply_off = c->n_ids;
    err |= janas_api_start_reply(c);
    if (err)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    /* the whole prompt the reply would read, against max_input: the
       conversation so far and this turn */
    c->last_at = (uint32_t)janas_llm_session_length(c->s);
    int32_t lim = janas_api_check_input(
        c, (uint32_t)janas_llm_session_length(c->s) + (uint32_t)c->n_ids);
    if (lim != JANAS_LLM_OK) {
        c->n_ids = 0;
        return lim;
    }
    if (c->llm->mode == JANAS_LLM_MODE_AUTO)
        janas_api_apply_mode(c->llm); /* the power source may have changed */
    /* the recap, if the context has to make room, runs before the clocks
       start: it is not part of this turn's speed */
    uint32_t want0 =
        (uint32_t)c->n_ids + (c->p.max_reply > 0 && c->p.max_reply < 1024
                                  ? (uint32_t)c->p.max_reply
                                  : 1024);
    if (c->p.recap && janas_llm_session_length(c->s) + want0 > c->llm->n_ctx)
        recap(c);
    janas_llm_session_stats(c->s, &c->st0);
    janas_expert_cache_stats(janas_llm_model_cache(c->llm->m), &c->cs0);
    c->t_send = janas_api_now();
    /* room for this message and for a reply of its own: rather than end the
       conversation when the context fills, the oldest exchanges are
       forgotten and the rest slides down */
    uint32_t reserve = c->p.max_reply > 0 && c->p.max_reply < 1024
                           ? (uint32_t)c->p.max_reply
                           : 1024;
    if (make_room(c, (uint32_t)c->n_ids + reserve) != 0) {
        c->replying = 0;
        return janas_api_fail(JANAS_LLM_EFULL,
                              "the context is full (%u tokens)", c->llm->n_ctx);
    }
    uint32_t at = (uint32_t)janas_llm_session_length(c->s);
    c->last_at = at; /* after the slide, if there was one */
    if (janas_llm_session_append(c->s, c->ids, (uint32_t)c->n_ids) != 0) {
        c->replying = 0;
        return janas_api_fail(JANAS_LLM_EFULL,
                              "the context is full (%u tokens)", c->llm->n_ctx);
    }
    if (janas_api_begin_reply(c, at + (uint32_t)reply_off,
                              at + (uint32_t)c->n_ids, at) != 0)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_send(janas_llm_chat *c, const char *text, int32_t len)
{
    if (!c || !text)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    return send_turn(c, text, janas_api_text_len(text, len), 0, NULL, NULL);
}

int32_t janas_llm_chat_send_results(janas_llm_chat *c, int32_t n,
                                    const char *const *texts,
                                    const int32_t *lens)
{
    if (!c || n <= 0 || !texts)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    for (int32_t i = 0; i < n; i++)
        if (!texts[i])
            return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    if (janas_llm_session_length(c->s) == 0)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "no reply called a tool: nothing to answer");
    if (c->llm->tools == JANAS_TOOLS_NONE)
        return janas_api_fail(JANAS_LLM_EMODEL,
                              "the model's chat template has no tools");
    return send_turn(c, NULL, 0, n, texts, lens);
}
