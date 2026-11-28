/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * engine.c - the model on a thread of its own. The model holds one sequence
 * at a time, so the requests run one after the other, in the order they
 * came; each one is loaded over the conversation left by the one before,
 * which is what lets a client that sends the same conversation plus a
 * message have only the message read (janas_llm_chat_load). The last few
 * conversations are kept computed besides it (janas_llm_chat_keep), so that
 * clients taking turns, or a client's requests on the side (titles, tags),
 * do not have each other's conversations read again. A request for
 * several replies (n, best_of) runs them one after the other on the same
 * prompt, which is then read once where the model can go back over it.
 */
#include "common/sysinfo.h"
#include "server.h"
#include "llm/nucleus.h"
#include "fim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SLOTS 16

/* A reply being written: its job, its choice, the chat it runs on, and
   run_one's state between its pieces */
struct slot {
    struct srv_job *j;
    int k;
    janas_llm_chat *c;
    int32_t rc;
    size_t max_stop, nl;
    int32_t seen;
    struct timespec t0;
    int begun; /* its job set up and its first reply begun */
};

struct srv_engine {
    janas_llm *llm;
    janas_llm *emb;  /* the embedding model, or NULL */
    janas_llm *spec; /* Nucleus mode: the Specialist, or NULL */
    int nucleus;
    janas_llm_chat *chat;
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct srv_job *head, *tail;
    struct srv_job *running; /* the job being run, or NULL */
    /* several replies at once (janas_llm_chats_step): n_slots places,
       place 0 on chat (with the conversations kept), the others on
       parallel chats made when first needed; the jobs they run */
    int n_slots;
    janas_llm_chat *par[MAX_SLOTS];
    struct srv_job *in_slot[MAX_SLOTS];
    uint32_t queued, max_queue;
    int stop;
    int closing; /* the server is stopping: no client's request taken */
    int verbose;
    uint32_t max_input, warn_input;
    struct srv_fim *fim; /* the windows of fill-in-the-middle requests */
    uint32_t fim_max;    /* the tokens they write at most; 0: as asked */
    char desc[1024];
    char system[1024]; /* the system message Janas's programs use */
    int64_t created;
    struct srv_model_info info;
};

struct srv_job *srv_job_new(void)
{
    struct srv_job *j = calloc(1, sizeof(*j));
    if (!j)
        return NULL;
    pthread_mutex_init(&j->mu, NULL);
    pthread_cond_init(&j->cv, NULL);
    janas_llm_chat_params_default(&j->p);
    j->parallel = 1;
    j->refs = 1;
    return j;
}

int srv_job_choices(struct srv_job *j, int n, int keep)
{
    j->ch = calloc((size_t)n, sizeof(*j->ch));
    if (!j->ch)
        return -1;
    j->n_choices = n;
    j->n_keep = keep;
    return 0;
}

static void choice_free(struct srv_choice *c)
{
    free(c->think);
    free(c->text);
    for (int i = 0; i < c->n_calls; i++) {
        free(c->calls[i].name);
        free(c->calls[i].args);
    }
    free(c->calls);
    for (size_t i = 0; i < c->n_lp; i++) {
        free(c->lp[i].t.tok);
        for (int k = 0; k < c->lp[i].n_top; k++)
            free(c->lp[i].top[k].tok);
        free(c->lp[i].top);
    }
    free(c->lp);
}

void srv_job_release(struct srv_job *j)
{
    if (!j)
        return;
    pthread_mutex_lock(&j->mu);
    int last = --j->refs == 0;
    pthread_mutex_unlock(&j->mu);
    if (!last)
        return;
    for (int32_t i = 0; i < j->n_msg; i++) {
        struct srv_msg *m = &j->msg[i];
        free(m->text);
        for (int k = 0; k < m->n_calls; k++) {
            free(m->call_name[k]);
            free(m->call_args[k]);
        }
        free(m->call_name);
        free(m->call_args);
        srv_imgs_free(m->imgs, m->n_imgs);
    }
    free(j->msg);
    for (int i = 0; i < j->n_prompts; i++) {
        free(j->prompts[i].text);
        free(j->prompts[i].ids);
    }
    free(j->prompts);
    free(j->suffix);
    for (int i = 0; i < j->n_stop; i++)
        free(j->stop[i]);
    free(j->tools);
    free(j->schema);
    free(j->kwargs);
    free(j->tmpl);
    free(j->bias_id);
    free(j->bias);
    free(j->emb);
    for (int i = 0; i < j->n_mcp; i++) {
        free(j->mcp_id[i]);
        free(j->mcp_out[i]);
    }
    free(j->mcp_id);
    free(j->mcp_out);
    for (int i = 0; i < j->n_choices; i++)
        choice_free(&j->ch[i]);
    free(j->ch);
    pthread_mutex_destroy(&j->mu);
    pthread_cond_destroy(&j->cv);
    free(j);
}

size_t srv_job_visible(const struct srv_job *j, int k)
{
    const struct srv_choice *c = &j->ch[k];
    if (c->done)
        return c->n_text;
    size_t hold = 0;
    for (int i = 0; i < j->n_stop; i++) {
        size_t n = strlen(j->stop[i]);
        if (n > hold + 1)
            hold = n - 1;
    }
    return c->n_text > hold ? c->n_text - hold : 0;
}

static int grow(char **buf, size_t *n, size_t *cap, const char *s, size_t k)
{
    if (*n + k + 1 > *cap) {
        size_t c = 2 * (*n + k) + 256;
        char *t = realloc(*buf, c);
        if (!t)
            return -1;
        *buf = t;
        *cap = c;
    }
    memcpy(*buf + *n, s, k);
    *n += k;
    (*buf)[*n] = 0;
    return 0;
}

/* The client went away, or the submitter's flag says stop (j->mu held, or
   the flag alone). */
static int stopped(const struct srv_job *j)
{
    return j->cancel || (j->watch && atomic_load(j->watch));
}

/* The first of the stop strings in the answer at or after `from`, or -1. */
static long find_stop(const struct srv_job *j, const struct srv_choice *c,
                      size_t from)
{
    long best = -1;
    for (int i = 0; i < j->n_stop; i++) {
        const char *hit = strstr(c->text + from, j->stop[i]);
        if (hit && (best < 0 || hit - c->text < best))
            best = hit - c->text;
    }
    return best;
}

static char *dup_n(const char *s, size_t n)
{
    char *d = malloc(n + 1);
    if (d) {
        memcpy(d, s, n);
        d[n] = 0;
    }
    return d;
}

/* The tool calls the reply has closed since the last look. Under j->mu. */
static int take_calls(janas_llm_chat *c, struct srv_choice *ch)
{
    int32_t n = janas_llm_chat_calls(c);
    while (ch->n_calls < n) {
        if (ch->n_calls == ch->cap_calls) {
            int cap = ch->cap_calls ? 2 * ch->cap_calls : 4;
            struct srv_call *t = realloc(ch->calls, (size_t)cap * sizeof(*t));
            if (!t)
                return -1;
            ch->calls = t;
            ch->cap_calls = cap;
        }
        int32_t nl = 0, al = 0;
        janas_llm_chat_call(c, ch->n_calls, NULL, 0, &nl, NULL, 0, &al);
        char *name = malloc((size_t)nl + 1), *args = malloc((size_t)al + 1);
        if (!name || !args) {
            free(name);
            free(args);
            return -1;
        }
        janas_llm_chat_call(c, ch->n_calls, name, nl + 1, &nl, args, al + 1,
                            &al);
        struct srv_call *k = &ch->calls[ch->n_calls];
        srv_new_id(k->id, sizeof(k->id), "call_");
        k->name = name;
        k->args = args;
        ch->n_calls++;
    }
    return 0;
}

/* A token's bytes and log-probability. */
static int alt(janas_llm *llm, int32_t id, float lp, struct srv_alt *a)
{
    char buf[256];
    int32_t n = 0;
    janas_llm_token_text(llm, id, buf, sizeof(buf), &n);
    if (n >= (int32_t)sizeof(buf))
        n = (int32_t)sizeof(buf) - 1;
    a->tok = dup_n(buf, (size_t)n);
    a->n = (size_t)n;
    a->lp = lp;
    return a->tok ? 0 : -1;
}

/*
 * The answer's tokens since token *seen, with their log-probabilities: what
 * logprobs asks for. The reasoning, the calls and the markers are not the
 * answer. Under j->mu.
 */
static int take_tokens(janas_llm *llm, janas_llm_chat *c, struct srv_job *j,
                       struct srv_choice *ch, int32_t *seen)
{
    int32_t n = janas_llm_chat_tokens(c);
    for (; *seen < n; (*seen)++) {
        int32_t id, part;
        float lp;
        janas_llm_chat_token(c, *seen, &id, &part, &lp, NULL, 0, NULL);
        if (part == JANAS_LLM_PART_REASONING)
            ch->reasoning_tokens++;
        if (part != JANAS_LLM_PART_ANSWER || !j->p.logprobs)
            continue;
        ch->score += lp;
        if (ch->n_lp == ch->cap_lp) {
            size_t cap = ch->cap_lp ? 2 * ch->cap_lp : 64;
            struct srv_lp *t = realloc(ch->lp, cap * sizeof(*t));
            if (!t)
                return -1;
            ch->lp = t;
            ch->cap_lp = cap;
        }
        struct srv_lp *r = &ch->lp[ch->n_lp];
        memset(r, 0, sizeof(*r));
        if (alt(llm, id, lp, &r->t) != 0)
            return -1;
        ch->n_lp++;
        int32_t ids[20], k = 0;
        float lps[20];
        janas_llm_chat_token_top(c, *seen, 20, ids, lps, &k);
        if (k > 0 && !(r->top = calloc((size_t)k, sizeof(*r->top))))
            return -1;
        for (int32_t q = 0; q < k; q++, r->n_top++)
            if (alt(llm, ids[q], lps[q], &r->top[q]) != 0)
                return -1;
    }
    return 0;
}

/* What the request says about tools, the answer's form and the bias. */
static int32_t setup(janas_llm_chat *c, struct srv_job *j)
{
    int32_t rc = janas_llm_chat_tools(c, j->tools, (int32_t)j->n_tools);
    /* the request's template and keywords, or none: what an earlier
       request of this chat set goes */
    if (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_template(c, j->tmpl, -1);
    if (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_template_kwargs(c, j->kwargs, -1);
    if (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_tool_choice(
            c, j->tool_choice,
            j->tool_choice == JANAS_LLM_TOOLS_FUNCTION ? j->tool_name : NULL,
            -1, j->parallel);
    if (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_format(c, j->format, j->schema,
                                   (int32_t)j->n_schema);
    if (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_logit_bias(c, j->n_bias, j->bias_id, j->bias);
    return rc;
}

/*
 * A prompt given as token ids, as text: the library reads text. The text is
 * kept in the prompt, which is where an echo finds it. Under nothing: the
 * handler reads it only once the job is done.
 */
static int32_t ids_text(struct srv_engine *e, struct srv_prompt *pp)
{
    char buf[1024]; /* as long as the library's longest token */
    size_t cap = 0;
    for (size_t i = 0; i < pp->n_ids; i++) {
        int32_t n = 0;
        if (janas_llm_token_text(e->llm, pp->ids[i], buf, sizeof(buf), &n) !=
            JANAS_LLM_OK)
            return JANAS_LLM_EINVAL; /* not a token of this model */
        if (pp->n + (size_t)n + 1 > cap) {
            cap = 2 * (pp->n + (size_t)n) + 256;
            char *t = realloc(pp->text, cap);
            if (!t)
                return JANAS_LLM_ENOMEM;
            pp->text = t;
        }
        memcpy(pp->text + pp->n, buf, (size_t)n);
        pp->n += (size_t)n;
        pp->text[pp->n] = 0;
    }
    return pp->text ? JANAS_LLM_OK : JANAS_LLM_EINVAL;
}

/* The tokens of a text, for the window of a fill-in-the-middle request. */
static int32_t count_tokens(void *ctx, const char *s, size_t n)
{
    int32_t t = 0;
    if (n > INT32_MAX)
        return -1;
    int32_t rc = janas_llm_tokenize(ctx, s, (int32_t)n, 0, NULL, 0, &t);
    return rc == JANAS_LLM_OK || rc == JANAS_LLM_ESMALL ? t : -1;
}

/* The prompt of a request with a suffix: the window around the cursor
   (fim.c), so that the next keystroke finds it computed. */
static int32_t load_infill(struct srv_engine *e, janas_llm_chat *c,
                           struct srv_job *j, const struct srv_prompt *pp)
{
    size_t off = 0, sl = j->n_suffix;
    const char *how = "";
    if (e->fim)
        srv_fim_window(e->fim, count_tokens, e->llm, pp->text, pp->n, j->suffix,
                       j->n_suffix, &off, &sl, &how);
    if (e->verbose && (off || sl < j->n_suffix))
        fprintf(stderr,
                "janas-server: fill in the middle: prefix from byte %zu of "
                "%zu%s%s, suffix %zu of %zu bytes\n",
                off, pp->n, *how ? ", " : "", how, sl, j->n_suffix);
    return janas_llm_chat_infill(c, pp->text + off, (int32_t)(pp->n - off),
                                 j->suffix, (int32_t)sl);
}

/* The request's prompt for reply k. */
static int32_t load(struct srv_engine *e, janas_llm_chat *c, struct srv_job *j,
                    int k)
{
    if (j->kind == JOB_TEXT) {
        struct srv_prompt *pp = &j->prompts[k / (j->n_choices / j->n_prompts)];
        if (pp->ids && !pp->text) {
            int32_t rc = ids_text(e, pp);
            if (rc != JANAS_LLM_OK)
                return rc;
        }
        return j->suffix ? load_infill(e, c, j, pp)
                         : janas_llm_chat_prompt(c, pp->text, (int32_t)pp->n);
    }
    int32_t rc = janas_llm_chat_begin(c);
    for (int32_t i = 0; rc == JANAS_LLM_OK && i < j->n_msg; i++) {
        const struct srv_msg *m = &j->msg[i];
        /* its pictures first: they go with the message added next */
        for (int k = 0; rc == JANAS_LLM_OK && k < m->n_imgs; k++)
            rc =
                janas_llm_chat_image(c, m->imgs[k].data, (int64_t)m->imgs[k].n);
        if (rc == JANAS_LLM_OK)
            rc = janas_llm_chat_add(c, m->role, m->text, (int32_t)m->n);
        for (int k = 0; rc == JANAS_LLM_OK && k < m->n_calls; k++)
            rc = janas_llm_chat_add_call(c, m->call_name[k], -1,
                                         m->call_args[k], -1);
    }
    return rc == JANAS_LLM_OK ? janas_llm_chat_run(c) : rc;
}

/* /infill's n_indent: a line of the text started past *nl indented less
   than n_indent (blank lines aside) ends it there, as llama.cpp does. 1
   when cut. */
static int indent_cut(struct srv_choice *ch, int n_indent, size_t *nl)
{
    for (;;) {
        if (*nl > 0) {
            size_t pos = *nl;
            int ind = 0;
            while (pos < ch->n_text &&
                   (ch->text[pos] == ' ' || ch->text[pos] == '\t')) {
                ind++;
                pos++;
            }
            if (pos < ch->n_text && ch->text[pos] != '\n' &&
                ch->text[pos] != '\r' && ind < n_indent) {
                ch->n_text = *nl;
                ch->text[*nl] = 0;
                return 1;
            }
        }
        const char *p = *nl < ch->n_text
                            ? memchr(ch->text + *nl, '\n', ch->n_text - *nl)
                            : NULL;
        if (!p)
            return 0;
        *nl = (size_t)(p - ch->text) + 1;
    }
}

/* One reply, into choice k. Returns the library's code. */
/* A reply begins on s->c: its parameters, its prompt. s->rc. */
static void one_begin(struct srv_engine *e, struct slot *s)
{
    janas_llm_chat *c = s->c;
    struct srv_job *j = s->j;
    int k = s->k;
    struct janas_llm_chat_params p = j->p;
    if (j->seeded)
        p.seed += (uint64_t)k;
    p.max_input = e->max_input; /* the server's limit, not the request's */
    if (j->suffix && e->fim_max &&
        (p.max_reply <= 0 || p.max_reply > (int32_t)e->fim_max))
        p.max_reply = (int32_t)e->fim_max;
    s->rc = janas_llm_chat_set_params(c, &p);
    if (s->rc == JANAS_LLM_OK)
        s->rc = load(e, c, j, k);
    s->max_stop = 0;
    for (int i = 0; i < j->n_stop; i++)
        if (strlen(j->stop[i]) > s->max_stop)
            s->max_stop = strlen(j->stop[i]);
    s->seen = 0;
    s->nl = 0;                    /* n_indent: where the last line started */
    s->t0 = (struct timespec){0}; /* t_max_predict_ms: the first token */
}

/* A piece of the reply: 1 while it goes on, 0 when it is over (s->rc) */
static int one_piece(struct srv_engine *e, struct slot *s)
{
    janas_llm_chat *c = s->c;
    struct srv_job *j = s->j;
    struct srv_choice *ch = &j->ch[s->k];
    if (s->rc != JANAS_LLM_OK)
        return 0;
    char buf[512];
    int32_t len = 0;
    int32_t rc = janas_llm_chat_next(c, buf, sizeof(buf), &len);
    if (rc != JANAS_LLM_OK && rc != JANAS_LLM_DONE) {
        s->rc = rc;
        return 0;
    }
    int think = janas_llm_chat_thinking(c);
    struct janas_llm_chat_stats rs = {.size = sizeof rs};
    int reading = janas_llm_chat_stats(c, &rs) == JANAS_LLM_OK &&
                  rs.stage == JANAS_LLM_STAGE_INPUT;
    pthread_mutex_lock(&j->mu);
    j->read_done = reading ? rs.input_done : 0;
    j->read_total = reading ? rs.input_tokens : 0;
    int bad =
        take_calls(c, ch) != 0 || take_tokens(e->llm, c, j, ch, &s->seen) != 0;
    if (len > 0 && think)
        bad |= grow(&ch->think, &ch->n_think, &ch->cap_think, buf, (size_t)len);
    else if (len > 0) {
        size_t from = ch->n_text > s->max_stop ? ch->n_text - s->max_stop : 0;
        bad |= grow(&ch->text, &ch->n_text, &ch->cap_text, buf, (size_t)len);
        long at = bad || !j->n_stop ? -1 : find_stop(j, ch, from);
        if (at >= 0) {
            ch->n_text = (size_t)at;
            ch->text[at] = 0;
            ch->by_stop = 1;
        }
        if (!bad && j->n_indent > 0 && indent_cut(ch, j->n_indent, &s->nl))
            ch->by_stop = 1;
        if (!bad && j->t_max_predict_ms > 0) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            if (!s->t0.tv_sec && !s->t0.tv_nsec)
                s->t0 = now;
            double ms = (double)(now.tv_sec - s->t0.tv_sec) * 1e3 +
                        (double)(now.tv_nsec - s->t0.tv_nsec) / 1e6;
            if (ms > j->t_max_predict_ms && memchr(ch->text, '\n', ch->n_text))
                ch->by_stop = 1;
        }
    }
    int quit = stopped(j) || ch->by_stop || bad || rc == JANAS_LLM_DONE;
    pthread_cond_broadcast(&j->cv);
    pthread_mutex_unlock(&j->mu);
    if (bad) {
        s->rc = JANAS_LLM_ENOMEM;
        return 0;
    }
    if (rc == JANAS_LLM_DONE)
        s->rc = JANAS_LLM_DONE;
    return !quit;
}

/* The reply's end: its statistics, its choice done. */
static int32_t one_end(struct srv_engine *e, struct slot *s)
{
    janas_llm_chat *c = s->c;
    struct srv_job *j = s->j;
    int k = s->k;
    struct srv_choice *ch = &j->ch[k];
    int32_t rc = s->rc;
    struct janas_llm_chat_stats st = {.size = sizeof(st)};
    janas_llm_chat_stats(c, &st);
    if (e->nucleus || e->verbose) /* the reply's times; in Nucleus mode who
                                     answered */
        fprintf(stderr,
                "janas-server: reply%s%s%s, prompt %u (%u computed before), "
                "output %u tokens, first token %.2f s, total %.2f s, threads "
                "%d/%d, drafts %u/%u, peak memory %.2f GiB\n",
                !e->nucleus                              ? ""
                : st.route == JANAS_LLM_ROUTE_SPECIALIST ? " route specialist"
                : st.route == JANAS_LLM_ROUTE_UNKNOWN    ? " route unknown"
                                                         : " route self",
                st.route_domain[0] ? " " : "", st.route_domain,
                st.prompt_tokens, st.cached_tokens, st.output_tokens,
                st.first_token_seconds, st.total_seconds, st.input_threads,
                st.output_threads, st.accepted, st.drafted,
                (double)st.peak_rss_bytes / (1 << 30));
    if (k == 0 && e->warn_input && st.prompt_tokens > e->warn_input &&
        rc != JANAS_LLM_ELIMIT)
        fprintf(stderr,
                "janas-server: a prompt of %u tokens, above --warn-input %u "
                "(system %u, of which tools %u; history %u; last %u): taken\n",
                st.prompt_tokens, e->warn_input, st.prompt_system,
                st.prompt_tools, st.prompt_history, st.prompt_last);
    pthread_mutex_lock(&j->mu);
    if (k == 0)
        j->st = st;
    /* the prompt is counted once, as OpenAI counts it: each prompt's */
    int per = j->kind == JOB_TEXT ? j->n_choices / j->n_prompts : j->n_choices;
    if (k % per == 0)
        j->prompt_tokens += st.prompt_tokens;
    ch->finish = ch->by_stop ? JANAS_LLM_FINISH_STOP : st.finish;
    ch->tokens = st.output_tokens;
    if (ch->n_lp)
        ch->score /= (double)ch->n_lp;
    j->completion_tokens += st.output_tokens;
    j->reasoning_tokens += ch->reasoning_tokens;
    ch->done = 1;
    pthread_cond_broadcast(&j->cv);
    pthread_mutex_unlock(&j->mu);
    return rc == JANAS_LLM_DONE ? JANAS_LLM_OK : rc;
}

static int32_t run_one(struct srv_engine *e, struct srv_job *j, int k)
{
    struct slot s = {.j = j, .k = k, .c = e->chat};
    one_begin(e, &s);
    while (one_piece(e, &s))
        ;
    return one_end(e, &s);
}

/* How many tokens the messages would be, with the tools: nothing run. */
static int32_t count(struct srv_engine *e, struct srv_job *j)
{
    janas_llm_chat *c = e->chat;
    int32_t rc = janas_llm_chat_begin(c), n = 0;
    for (int32_t i = 0; rc == JANAS_LLM_OK && i < j->n_msg; i++) {
        const struct srv_msg *m = &j->msg[i];
        /* its pictures first: they go with the message added next */
        for (int k = 0; rc == JANAS_LLM_OK && k < m->n_imgs; k++)
            rc =
                janas_llm_chat_image(c, m->imgs[k].data, (int64_t)m->imgs[k].n);
        if (rc == JANAS_LLM_OK)
            rc = janas_llm_chat_add(c, m->role, m->text, (int32_t)m->n);
        for (int k = 0; rc == JANAS_LLM_OK && k < m->n_calls; k++)
            rc = janas_llm_chat_add_call(c, m->call_name[k], -1,
                                         m->call_args[k], -1);
    }
    if (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_count(c, &n);
    pthread_mutex_lock(&j->mu);
    j->prompt_tokens = (uint32_t)n;
    pthread_mutex_unlock(&j->mu);
    return rc;
}

/* The vectors of the job's texts, from the embedding model. */
static int32_t embed(struct srv_engine *e, struct srv_job *j)
{
    if (!e->emb) {
        pthread_mutex_lock(&j->mu);
        snprintf(j->errmsg, sizeof(j->errmsg),
                 "no embedding model: start the server with "
                 "--embedding-model");
        pthread_mutex_unlock(&j->mu);
        return JANAS_LLM_EMODEL;
    }
    int32_t full = janas_llm_embed_dim(e->emb);
    int32_t dim = j->dims > 0 && j->dims < full ? j->dims : full;
    float *v = malloc((size_t)j->n_prompts * (size_t)dim * sizeof(float));
    if (!v)
        return JANAS_LLM_ENOMEM;
    uint32_t tokens = 0;
    int32_t rc = JANAS_LLM_OK;
    for (int i = 0; i < j->n_prompts && rc == JANAS_LLM_OK; i++) {
        struct srv_prompt *pp = &j->prompts[i];
        if (pp->ids && !pp->text) {
            char buf[1024]; /* as long as the library's longest token */
            for (size_t k = 0; k < pp->n_ids && rc == JANAS_LLM_OK; k++) {
                int32_t n = 0;
                if (janas_llm_token_text(e->emb, pp->ids[k], buf, sizeof(buf),
                                         &n) != JANAS_LLM_OK) {
                    rc = JANAS_LLM_EINVAL;
                    break;
                }
                char *t = realloc(pp->text, pp->n + (size_t)n + 1);
                if (!t) {
                    rc = JANAS_LLM_ENOMEM;
                    break;
                }
                pp->text = t;
                memcpy(t + pp->n, buf, (size_t)n);
                pp->n += (size_t)n;
                t[pp->n] = 0;
            }
        }
        int32_t n = 0, nt = 0;
        if (rc == JANAS_LLM_OK)
            rc = janas_llm_embed(e->emb, pp->text ? pp->text : "",
                                 (int32_t)pp->n, dim, v + (size_t)i * dim, &n,
                                 &nt);
        tokens += (uint32_t)nt;
    }
    pthread_mutex_lock(&j->mu);
    j->emb = v;
    j->emb_dim = dim;
    j->prompt_tokens = tokens;
    pthread_mutex_unlock(&j->mu);
    return rc;
}

static void run(struct srv_engine *e, struct srv_job *j)
{
    int32_t rc = setup(e->chat, j);
    int k = 0;
    if (rc == JANAS_LLM_OK && j->kind == JOB_COUNT) {
        rc = count(e, j);
        k = j->n_choices; /* nothing to generate */
    }
    if (j->kind == JOB_EMBED) {
        rc = embed(e, j);
        k = j->n_choices;
    }
    for (; rc == JANAS_LLM_OK && k < j->n_choices; k++) {
        pthread_mutex_lock(&j->mu);
        j->cur = k;
        int cancel = stopped(j);
        pthread_mutex_unlock(&j->mu);
        if (cancel)
            break;
        rc = run_one(e, j, k);
    }
    pthread_mutex_lock(&j->mu);
    if (rc < 0) {
        j->err = rc;
        if (!j->errmsg[0]) /* unless the engine said it itself */
            snprintf(j->errmsg, sizeof(j->errmsg), "%s",
                     janas_llm_last_error());
    }
    for (int i = 0; i < j->n_choices; i++)
        j->ch[i].done = 1; /* the ones not run, on error or cancel */
    int cancelled = j->cancel;
    j->done = 1;
    pthread_cond_broadcast(&j->cv);
    pthread_mutex_unlock(&j->mu);
    if (e->verbose) {
        int32_t kept = 0, files = 0;
        uint64_t kb = 0, fb = 0;
        janas_llm_chat_kept(e->chat, &kept, &kb);
        janas_llm_chat_kept_disk(e->chat, &files, &fb);
        fprintf(stderr,
                "janas-server: %u prompt tokens (%u already computed), %u "
                "generated in %d replies%s; %d conversations kept (%.0f "
                "MiB), %d on disk (%.0f MiB)\n",
                j->st.prompt_tokens, j->st.cached_tokens, j->completion_tokens,
                k, cancelled ? ", cancelled" : "", kept, (double)kb / (1 << 20),
                files, (double)fb / (1 << 20));
    }
}

/* Where the jobs still waiting stand in the queue. Under e->mu. */
static void number_queue(struct srv_engine *e)
{
    uint32_t pos = 1;
    for (struct srv_job *j = e->head; j; j = j->next, pos++) {
        pthread_mutex_lock(&j->mu);
        j->queue_pos = pos;
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
}

/* the job's end, as run() ends it: its error, its choices done, its
   statistics; then it is released */
static void job_end(struct srv_engine *e, struct srv_job *j, int32_t rc, int k)
{
    pthread_mutex_lock(&j->mu);
    if (rc < 0) {
        j->err = rc;
        if (!j->errmsg[0])
            snprintf(j->errmsg, sizeof(j->errmsg), "%s",
                     janas_llm_last_error());
    }
    for (int i = 0; i < j->n_choices; i++)
        j->ch[i].done = 1;
    int cancelled = j->cancel;
    j->done = 1;
    pthread_cond_broadcast(&j->cv);
    pthread_mutex_unlock(&j->mu);
    if (e->verbose)
        fprintf(stderr,
                "janas-server: %u prompt tokens (%u already computed), %u "
                "generated in %d replies%s, beside other requests\n",
                j->st.prompt_tokens, j->st.cached_tokens, j->completion_tokens,
                k, cancelled ? ", cancelled" : "");
    srv_job_release(j);
}

/* A job a place can take: a reply of text or of chat, without pictures
   (a sequence of several does not take them yet) */
static int slot_job(const struct srv_job *j)
{
    if (j->kind != JOB_TEXT && j->kind != JOB_CHAT)
        return 0;
    for (int32_t i = 0; i < j->n_msg; i++)
        if (j->msg[i].n_imgs)
            return 0;
    return 1;
}

/* The next choice of the job in s, or its end. 1 when s goes on. */
static int slot_next(struct srv_engine *e, struct slot *s, int slot)
{
    struct srv_job *j = s->j;
    int32_t rc = one_end(e, s);
    int k = s->k + 1;
    pthread_mutex_lock(&j->mu);
    int cancel = stopped(j);
    if (k < j->n_choices && rc == JANAS_LLM_OK && !cancel)
        j->cur = k;
    pthread_mutex_unlock(&j->mu);
    if (k < j->n_choices && rc == JANAS_LLM_OK && !cancel) {
        s->k = k;
        one_begin(e, s);
        return 1;
    }
    job_end(e, j, rc, k);
    pthread_mutex_lock(&e->mu);
    e->in_slot[slot] = NULL;
    pthread_mutex_unlock(&e->mu);
    s->j = NULL;
    return 0;
}

/* Several replies at once: the jobs waiting take the free places (place 0
   first: its chat keeps the conversations), each place writes a piece of
   its reply a round, the passes of all of them in one step. Jobs a place
   does not take run as before, when no place is busy. */
static void slots_main(struct srv_engine *e)
{
    struct slot sl[MAX_SLOTS] = {{0}};
    for (;;) {
        pthread_mutex_lock(&e->mu);
        int busy = 0;
        for (int i = 0; i < e->n_slots; i++)
            busy += sl[i].j != NULL;
        while (!busy && !e->head && !e->stop)
            pthread_cond_wait(&e->cv, &e->mu);
        if (e->stop) {
            pthread_mutex_unlock(&e->mu);
            break;
        }
        /* a job no place takes waits for all of them to be free, then runs
           alone, as before */
        struct srv_job *solo = NULL;
        while (e->head) {
            struct srv_job *j = e->head;
            int free_at = -1;
            for (int i = 0; i < e->n_slots && free_at < 0; i++)
                if (!sl[i].j)
                    free_at = i;
            if (!slot_job(j)) {
                if (!busy) {
                    solo = j;
                    e->head = j->next;
                    if (!e->head)
                        e->tail = NULL;
                    e->queued--;
                    e->running = j;
                }
                break;
            }
            if (free_at < 0)
                break;
            if (free_at > 0 && !e->par[free_at] &&
                janas_llm_chat_create_parallel(
                    e->llm, NULL, &e->par[free_at]) != JANAS_LLM_OK) {
                break; /* no memory for another: it waits */
            }
            e->head = j->next;
            if (!e->head)
                e->tail = NULL;
            e->queued--;
            e->in_slot[free_at] = j;
            sl[free_at] = (struct slot){
                .j = j, .k = 0, .c = free_at ? e->par[free_at] : e->chat};
            busy++;
        }
        number_queue(e);
        pthread_mutex_unlock(&e->mu);
        if (solo) {
            pthread_mutex_lock(&solo->mu);
            solo->started = 1;
            solo->queue_pos = 0;
            pthread_mutex_unlock(&solo->mu);
            run(e, solo);
            pthread_mutex_lock(&e->mu);
            e->running = NULL;
            pthread_mutex_unlock(&e->mu);
            srv_job_release(solo);
            continue;
        }
        /* the places just taken begin their first reply */
        for (int i = 0; i < e->n_slots; i++) {
            struct slot *s = &sl[i];
            if (!s->j || s->begun)
                continue;
            struct srv_job *j = s->j;
            s->begun = 1;
            pthread_mutex_lock(&j->mu);
            j->started = 1;
            j->queue_pos = 0;
            int skip = stopped(j);
            pthread_mutex_unlock(&j->mu);
            s->rc = skip ? JANAS_LLM_EFAIL : setup(s->c, j);
            if (s->rc == JANAS_LLM_OK)
                one_begin(e, s);
            if (s->rc != JANAS_LLM_OK) {
                job_end(e, j, skip ? JANAS_LLM_OK : s->rc, 0);
                pthread_mutex_lock(&e->mu);
                e->in_slot[i] = NULL;
                pthread_mutex_unlock(&e->mu);
                *s = (struct slot){0};
            }
        }
        /* one step for all, then a piece each */
        janas_llm_chat *act[MAX_SLOTS];
        int na = 0;
        for (int i = 0; i < e->n_slots; i++)
            if (sl[i].j && sl[i].rc == JANAS_LLM_OK)
                act[na++] = sl[i].c;
        if (na > 1)
            janas_llm_chats_step(act, na); /* failed: each runs alone */
        for (int i = 0; i < e->n_slots; i++) {
            struct slot *s = &sl[i];
            if (!s->j)
                continue;
            if (!one_piece(e, s) && !slot_next(e, s, i))
                *s = (struct slot){0};
        }
    }
    /* the server stops: the replies under way end as they are */
    for (int i = 0; i < e->n_slots; i++)
        if (sl[i].j) {
            sl[i].rc = JANAS_LLM_EFAIL;
            job_end(e, sl[i].j, JANAS_LLM_EFAIL, sl[i].k);
        }
}

static void *engine_main(void *arg)
{
    struct srv_engine *e = arg;
    if (e->n_slots > 1) {
        slots_main(e);
        return NULL;
    }
    for (;;) {
        pthread_mutex_lock(&e->mu);
        while (!e->head && !e->stop)
            pthread_cond_wait(&e->cv, &e->mu);
        if (e->stop) {
            pthread_mutex_unlock(&e->mu);
            break;
        }
        struct srv_job *j = e->head;
        e->head = j->next;
        if (!e->head)
            e->tail = NULL;
        e->queued--;
        e->running = j;
        number_queue(e);
        pthread_mutex_unlock(&e->mu);
        pthread_mutex_lock(&j->mu);
        j->started = 1;
        j->queue_pos = 0;
        int skip = stopped(j); /* gone while it waited */
        pthread_mutex_unlock(&j->mu);
        if (skip) {
            pthread_mutex_lock(&j->mu);
            j->done = 1;
            for (int i = 0; i < j->n_choices; i++)
                j->ch[i].done = 1;
            pthread_cond_broadcast(&j->cv);
            pthread_mutex_unlock(&j->mu);
        } else {
            run(e, j);
        }
        pthread_mutex_lock(&e->mu);
        e->running = NULL;
        pthread_mutex_unlock(&e->mu);
        srv_job_release(j);
    }
    return NULL;
}

_Thread_local int srv_engine_low;
_Thread_local const _Atomic int *srv_engine_watch;

int srv_engine_submit(struct srv_engine *e, struct srv_job *j)
{
    j->low = srv_engine_low;
    j->watch = srv_engine_watch;
    pthread_mutex_lock(&e->mu);
    if (e->queued >= e->max_queue || e->stop || (e->closing && !j->low)) {
        pthread_mutex_unlock(&e->mu);
        return -1;
    }
    pthread_mutex_lock(&j->mu);
    j->refs++; /* the engine's */
    pthread_mutex_unlock(&j->mu);
    j->next = NULL;
    /* a request of a client before the batches' waiting */
    struct srv_job **at = &e->head;
    while (*at && (j->low || !(*at)->low))
        at = &(*at)->next;
    j->next = *at;
    *at = j;
    if (!j->next)
        e->tail = j;
    e->queued++;
    number_queue(e);
    pthread_cond_signal(&e->cv);
    pthread_mutex_unlock(&e->mu);
    return 0;
}

/* What the model is, from the library: its name, its architecture (the
   description's "name (arch), ..."), its context and what it can do. */
static void model_info(struct srv_engine *e)
{
    struct srv_model_info *mi = &e->info;
    int32_t n;
    janas_llm_name(e->llm, mi->name, sizeof(mi->name), &n);
    const char *open = strstr(e->desc, " ("), *close = NULL;
    if (open)
        close = strchr(open + 2, ')');
    if (open && close && close - open - 2 < (long)sizeof(mi->arch))
        snprintf(mi->arch, sizeof(mi->arch), "%.*s", (int)(close - open - 2),
                 open + 2);
    const char *ctx = strstr(e->desc, ", context ");
    if (ctx)
        mi->n_ctx = (uint32_t)strtoul(ctx + 10, NULL, 10);
    mi->tools = janas_llm_uses_tools(e->llm);
    mi->sees = janas_llm_sees(e->llm);
    mi->reasons = janas_llm_reasons(e->llm);
}

struct srv_engine *srv_engine_start(const struct srv_config *cfg, char *err,
                                    size_t err_len)
{
    struct srv_engine *e = calloc(1, sizeof(*e));
    if (!e) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    e->max_queue = cfg->max_queue ? cfg->max_queue : 16;
    e->verbose = cfg->verbose;
    e->max_input = cfg->max_input;
    e->warn_input = cfg->warn_input;
    pthread_mutex_init(&e->mu, NULL);
    pthread_cond_init(&e->cv, NULL);
    /* the model's MTP block or assistant, when not named: the one beside
       it that fits */
    static char mtp_found[4096];
    struct janas_llm_params lp = cfg->llm;
    if (!lp.mtp_path && !cfg->no_mtp &&
        janas_llm_find_mtp(cfg->model_path, mtp_found, sizeof(mtp_found)) ==
            JANAS_LLM_OK) {
        lp.mtp_path = mtp_found;
        fprintf(stderr, "janas-server: multi-token prediction from %s\n",
                mtp_found);
    }
    if (janas_llm_open(cfg->model_path, &lp, &e->llm) != JANAS_LLM_OK ||
        janas_llm_chat_create(e->llm, NULL, &e->chat) != JANAS_LLM_OK) {
        snprintf(err, err_len, "%s", janas_llm_last_error());
        janas_llm_close(e->llm);
        free(e);
        return NULL;
    }
    if (janas_llm_chat_keep(e->chat, cfg->keep, cfg->keep_bytes) !=
            JANAS_LLM_OK ||
        (cfg->keep > 0 &&
         janas_llm_chat_keep_disk(e->chat, cfg->keep_disk) != JANAS_LLM_OK)) {
        snprintf(err, err_len, "%s", janas_llm_last_error());
        janas_llm_chat_destroy(e->chat);
        janas_llm_close(e->llm);
        free(e);
        return NULL;
    }
    if (cfg->chat_template) {
        /* every chat's template, in place of the model's */
        FILE *f = fopen(cfg->chat_template, "rb");
        char *t = NULL;
        long n = -1;
        if (f && fseek(f, 0, SEEK_END) == 0 && (n = ftell(f)) >= 0 &&
            fseek(f, 0, SEEK_SET) == 0 && (t = malloc((size_t)n + 1)) &&
            fread(t, 1, (size_t)n, f) == (size_t)n) {
            t[n] = 0;
            if (janas_llm_set_template(e->llm, t, (int32_t)n) != JANAS_LLM_OK)
                snprintf(err, err_len, "%s: %s", cfg->chat_template,
                         janas_llm_last_error());
            else
                n = -2; /* taken */
        } else {
            snprintf(err, err_len, "%s: cannot be read", cfg->chat_template);
        }
        if (f)
            fclose(f);
        free(t);
        if (n != -2) {
            janas_llm_chat_destroy(e->chat);
            janas_llm_close(e->llm);
            free(e);
            return NULL;
        }
    }
    if (cfg->no_gpu)
        janas_llm_set_gpu(e->llm, 0);
    if (cfg->nucleus_gate) { /* Nucleus mode: the Specialist, the gate */
        struct janas_llm_params sp;
        janas_llm_params_default(&sp);
        sp.n_ctx = cfg->llm.n_ctx;
        if ((cfg->specialist_path &&
             janas_llm_open(cfg->specialist_path, &sp, &e->spec) !=
                 JANAS_LLM_OK) ||
            janas_llm_nucleus(e->llm, cfg->nucleus_gate, e->spec) !=
                JANAS_LLM_OK ||
            /* the Specialist's own gate: its "I don't know" is the last */
            (e->spec && cfg->specialist_gate &&
             janas_llm_nucleus(e->spec, cfg->specialist_gate, NULL) !=
                 JANAS_LLM_OK)) {
            snprintf(err, err_len, "Nucleus mode: %s", janas_llm_last_error());
            janas_llm_chat_destroy(e->chat);
            janas_llm_close(e->llm);
            janas_llm_close(e->spec);
            free(e);
            return NULL;
        }
        if (cfg->no_gpu && e->spec)
            janas_llm_set_gpu(e->spec, 0);
        e->nucleus = 1;
        fprintf(stderr,
                "janas-server: Nucleus mode, gate %s, specialist %s, its gate "
                "%s\n",
                cfg->nucleus_gate,
                cfg->specialist_path ? cfg->specialist_path : "none",
                cfg->specialist_gate ? cfg->specialist_gate : "none");
    }
    if (cfg->embed_path) {
        struct janas_llm_params ep;
        janas_llm_params_default(&ep);
        ep.n_ctx = 8192;
        ep.no_preload = 1;
        if (janas_llm_open(cfg->embed_path, &ep, &e->emb) != JANAS_LLM_OK ||
            !janas_llm_embed_dim(e->emb)) {
            snprintf(err, err_len, "%s: %s", cfg->embed_path,
                     e->emb ? "not an embedding model"
                            : janas_llm_last_error());
            janas_llm_close(e->emb);
            janas_llm_chat_destroy(e->chat);
            janas_llm_close(e->llm);
            free(e);
            return NULL;
        }
    }
    int32_t n;
    janas_llm_describe(e->llm, e->desc, sizeof(e->desc), &n);
    janas_llm_default_system(e->llm, e->system, sizeof(e->system), &n);
    e->created = (int64_t)time(NULL);
    model_info(e);
    e->fim = srv_fim_new(cfg->fim_prefix, cfg->fim_suffix);
    e->fim_max = cfg->fim_max;
    /* replies at once: as asked, or as many as half the memory free now
       holds of a conversation's keys, values and state (the context as
       asked, never cut), up to 8 */
    e->n_slots = 1;
    if (janas_llm_parallel_ok(e->llm) && !e->nucleus && cfg->parallel != 1) {
        int64_t per = janas_llm_parallel_bytes(e->llm);
        uint64_t avail = janas_mem_available();
        int fit = per > 0 ? 1 + (int)(avail / 2 / (uint64_t)per) : 1;
        int want = cfg->parallel > 0 ? cfg->parallel : 8;
        e->n_slots = fit < want ? fit : want;
        if (e->n_slots > MAX_SLOTS)
            e->n_slots = MAX_SLOTS;
        if (e->n_slots < 1)
            e->n_slots = 1;
        fprintf(stderr,
                "janas-server: %d replies at once (%.2f GiB a conversation, "
                "%.1f GiB free%s)\n",
                e->n_slots, (double)per / (1 << 30), (double)avail / (1 << 30),
                cfg->parallel > 0 && e->n_slots < cfg->parallel
                    ? "; fewer than asked: the memory"
                    : "");
    }
    if (pthread_create(&e->th, NULL, engine_main, e) != 0) {
        snprintf(err, err_len, "cannot start the engine's thread");
        srv_fim_free(e->fim);
        janas_llm_chat_destroy(e->chat);
        janas_llm_close(e->llm);
        free(e);
        return NULL;
    }
    return e;
}

/* A client's job ended because the server stops, or because it was asked
   (why): an error that says so, for the client still there. */
static void cancel_job_why(struct srv_job *j, const char *why)
{
    pthread_mutex_lock(&j->mu);
    j->cancel = 1;
    if (j->err == JANAS_LLM_OK) {
        j->err = JANAS_LLM_EFAIL;
        snprintf(j->errmsg, sizeof(j->errmsg), "%s", why);
    }
    pthread_mutex_unlock(&j->mu);
}

static void cancel_job(struct srv_job *j)
{
    cancel_job_why(j, "the server is stopping");
}

int srv_engine_cancel_all(struct srv_engine *e)
{
    if (!e)
        return 0;
    static const char why[] = "cancelled: the program that runs this server "
                              "asked for its work to stop";
    int n = 0;
    pthread_mutex_lock(&e->mu);
    if (e->running && !e->running->low)
        cancel_job_why(e->running, why), n++;
    for (int i = 0; i < e->n_slots; i++)
        if (e->in_slot[i] && !e->in_slot[i]->low && e->in_slot[i] != e->running)
            cancel_job_why(e->in_slot[i], why), n++;
    for (struct srv_job *j = e->head; j; j = j->next)
        if (!j->low)
            cancel_job_why(j, why), n++;
    pthread_mutex_unlock(&e->mu);
    return n;
}

void srv_engine_cancel_clients(struct srv_engine *e)
{
    if (!e)
        return;
    pthread_mutex_lock(&e->mu);
    e->closing = 1;
    if (e->running && !e->running->low)
        cancel_job(e->running);
    for (int i = 0; i < e->n_slots; i++)
        if (e->in_slot[i] && !e->in_slot[i]->low)
            cancel_job(e->in_slot[i]);
    for (struct srv_job *j = e->head; j; j = j->next)
        if (!j->low)
            cancel_job(j);
    pthread_mutex_unlock(&e->mu);
}

void srv_engine_stop(struct srv_engine *e)
{
    if (!e)
        return;
    pthread_mutex_lock(&e->mu);
    e->stop = 1;
    pthread_cond_broadcast(&e->cv);
    pthread_mutex_unlock(&e->mu);
    pthread_join(e->th, NULL);
    for (struct srv_job *j = e->head; j;) {
        struct srv_job *nx = j->next;
        pthread_mutex_lock(&j->mu);
        j->done = 1;
        for (int i = 0; i < j->n_choices; i++)
            j->ch[i].done = 1;
        j->err = JANAS_LLM_EFAIL;
        snprintf(j->errmsg, sizeof(j->errmsg), "the server is stopping");
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->mu);
        srv_job_release(j);
        j = nx;
    }
    for (int i = 1; i < MAX_SLOTS; i++)
        janas_llm_chat_destroy(e->par[i]);
    janas_llm_chat_destroy(e->chat);
    janas_llm_close(e->llm);
    janas_llm_close(e->spec); /* after the model: its chat holds one here */
    janas_llm_close(e->emb);
    srv_fim_free(e->fim);
    pthread_mutex_destroy(&e->mu);
    pthread_cond_destroy(&e->cv);
    free(e);
}

int32_t srv_engine_embed_dim(const struct srv_engine *e)
{
    return e->emb ? janas_llm_embed_dim(e->emb) : 0;
}

const char *srv_engine_describe(const struct srv_engine *e)
{
    return e->desc;
}

const char *srv_engine_system(const struct srv_engine *e)
{
    return e->system;
}

int64_t srv_engine_created(const struct srv_engine *e)
{
    return e->created;
}

const struct srv_model_info *srv_engine_info(const struct srv_engine *e)
{
    return &e->info;
}

janas_llm *srv_engine_llm(const struct srv_engine *e)
{
    return e->llm;
}

int srv_engine_queued(struct srv_engine *e)
{
    pthread_mutex_lock(&e->mu);
    int n = (int)e->queued + (e->running ? 1 : 0);
    for (int i = 0; i < e->n_slots; i++)
        n += e->in_slot[i] != NULL;
    pthread_mutex_unlock(&e->mu);
    return n;
}
