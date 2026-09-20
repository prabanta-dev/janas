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
#include "server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct srv_engine {
    janas_llm *llm;
    janas_llm *emb; /* the embedding model, or NULL */
    janas_llm_chat *chat;
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct srv_job *head, *tail;
    uint32_t queued, max_queue;
    int stop;
    int verbose;
    uint32_t max_input, warn_input;
    char desc[1024];
    char system[1024]; /* the system message Janas's programs use */
    int64_t created;
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
static int32_t setup(struct srv_engine *e, struct srv_job *j)
{
    janas_llm_chat *c = e->chat;
    int32_t rc = janas_llm_chat_tools(c, j->tools, (int32_t)j->n_tools);
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
    char buf[256];
    size_t cap = 0;
    for (size_t i = 0; i < pp->n_ids; i++) {
        int32_t n = 0;
        if (janas_llm_token_text(e->llm, pp->ids[i], buf, sizeof(buf), &n) !=
                JANAS_LLM_OK &&
            n < (int32_t)sizeof(buf))
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

/* The request's prompt for reply k. */
static int32_t load(struct srv_engine *e, struct srv_job *j, int k)
{
    janas_llm_chat *c = e->chat;
    if (j->kind == JOB_TEXT) {
        struct srv_prompt *pp = &j->prompts[k / (j->n_choices / j->n_prompts)];
        if (pp->ids && !pp->text) {
            int32_t rc = ids_text(e, pp);
            if (rc != JANAS_LLM_OK)
                return rc;
        }
        return j->suffix
                   ? janas_llm_chat_infill(c, pp->text, (int32_t)pp->n,
                                           j->suffix, (int32_t)j->n_suffix)
                   : janas_llm_chat_prompt(c, pp->text, (int32_t)pp->n);
    }
    int32_t rc = janas_llm_chat_begin(c);
    for (int32_t i = 0; rc == JANAS_LLM_OK && i < j->n_msg; i++) {
        const struct srv_msg *m = &j->msg[i];
        rc = janas_llm_chat_add(c, m->role, m->text, (int32_t)m->n);
        for (int k = 0; rc == JANAS_LLM_OK && k < m->n_calls; k++)
            rc = janas_llm_chat_add_call(c, m->call_name[k], -1,
                                         m->call_args[k], -1);
    }
    return rc == JANAS_LLM_OK ? janas_llm_chat_run(c) : rc;
}

/* One reply, into choice k. Returns the library's code. */
static int32_t run_one(struct srv_engine *e, struct srv_job *j, int k)
{
    janas_llm_chat *c = e->chat;
    struct srv_choice *ch = &j->ch[k];
    struct janas_llm_chat_params p = j->p;
    if (j->seeded)
        p.seed += (uint64_t)k;
    p.max_input = e->max_input; /* the server's limit, not the request's */
    int32_t rc = janas_llm_chat_set_params(c, &p);
    if (rc == JANAS_LLM_OK)
        rc = load(e, j, k);
    size_t max_stop = 0;
    for (int i = 0; i < j->n_stop; i++)
        if (strlen(j->stop[i]) > max_stop)
            max_stop = strlen(j->stop[i]);
    int32_t seen = 0;
    while (rc == JANAS_LLM_OK) {
        char buf[512];
        int32_t len = 0;
        rc = janas_llm_chat_next(c, buf, sizeof(buf), &len);
        if (rc != JANAS_LLM_OK && rc != JANAS_LLM_DONE)
            break;
        int think = janas_llm_chat_thinking(c);
        pthread_mutex_lock(&j->mu);
        int bad =
            take_calls(c, ch) != 0 || take_tokens(e->llm, c, j, ch, &seen) != 0;
        if (len > 0 && think)
            bad |= grow(&ch->think, &ch->n_think, &ch->cap_think, buf,
                        (size_t)len);
        else if (len > 0) {
            size_t from = ch->n_text > max_stop ? ch->n_text - max_stop : 0;
            bad |=
                grow(&ch->text, &ch->n_text, &ch->cap_text, buf, (size_t)len);
            long at = bad || !j->n_stop ? -1 : find_stop(j, ch, from);
            if (at >= 0) {
                ch->n_text = (size_t)at;
                ch->text[at] = 0;
                ch->by_stop = 1;
            }
        }
        int quit = j->cancel || ch->by_stop || bad || rc == JANAS_LLM_DONE;
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->mu);
        if (bad) {
            rc = JANAS_LLM_ENOMEM;
            break;
        }
        if (quit)
            break;
    }
    struct janas_llm_chat_stats st = {.size = sizeof(st)};
    janas_llm_chat_stats(c, &st);
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

/* How many tokens the messages would be, with the tools: nothing run. */
static int32_t count(struct srv_engine *e, struct srv_job *j)
{
    janas_llm_chat *c = e->chat;
    int32_t rc = janas_llm_chat_begin(c), n = 0;
    for (int32_t i = 0; rc == JANAS_LLM_OK && i < j->n_msg; i++) {
        const struct srv_msg *m = &j->msg[i];
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
            char buf[256];
            for (size_t k = 0; k < pp->n_ids && rc == JANAS_LLM_OK; k++) {
                int32_t n = 0;
                janas_llm_token_text(e->emb, pp->ids[k], buf, sizeof(buf), &n);
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
    int32_t rc = setup(e, j);
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
        int cancel = j->cancel;
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

static void *engine_main(void *arg)
{
    struct srv_engine *e = arg;
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
        number_queue(e);
        pthread_mutex_unlock(&e->mu);
        pthread_mutex_lock(&j->mu);
        j->started = 1;
        j->queue_pos = 0;
        int skip = j->cancel; /* gone while it waited */
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
        srv_job_release(j);
    }
    return NULL;
}

int srv_engine_submit(struct srv_engine *e, struct srv_job *j)
{
    pthread_mutex_lock(&e->mu);
    if (e->queued >= e->max_queue || e->stop) {
        pthread_mutex_unlock(&e->mu);
        return -1;
    }
    pthread_mutex_lock(&j->mu);
    j->refs++; /* the engine's */
    pthread_mutex_unlock(&j->mu);
    j->next = NULL;
    if (e->tail)
        e->tail->next = j;
    else
        e->head = j;
    e->tail = j;
    e->queued++;
    number_queue(e);
    pthread_cond_signal(&e->cv);
    pthread_mutex_unlock(&e->mu);
    return 0;
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
    if (janas_llm_open(cfg->model_path, &cfg->llm, &e->llm) != JANAS_LLM_OK ||
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
    if (cfg->no_gpu)
        janas_llm_set_gpu(e->llm, 0);
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
    if (pthread_create(&e->th, NULL, engine_main, e) != 0) {
        snprintf(err, err_len, "cannot start the engine's thread");
        janas_llm_chat_destroy(e->chat);
        janas_llm_close(e->llm);
        free(e);
        return NULL;
    }
    return e;
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
    janas_llm_chat_destroy(e->chat);
    janas_llm_close(e->llm);
    janas_llm_close(e->emb);
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
