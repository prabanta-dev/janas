/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * metrics.c - progress lines and JSON events of a reply (see metrics.h).
 */
#include "llm/metrics.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "llm/chat.h"
#include "llm/json.h"
#include "llm/model.h"

static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int json_fd = -1;
static double json_every = 10, human_every; /* seconds; 0: off */
static unsigned long seq;

static void set_up(void)
{
    const char *f = getenv("JANAS_METRICS"), *e = getenv("JANAS_METRICS_EVERY");
    const char *h = getenv("JANAS_PROGRESS");
    if (f && *f)
        json_fd =
            strcmp(f, "-") == 0
                ? dup(STDERR_FILENO)
                : open(f, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (f && *f && json_fd < 0) /* said once: otherwise it looks on */
        fprintf(stderr, "janas: JANAS_METRICS: cannot open %s: %s\n", f,
                strerror(errno));
    if (e && atof(e) > 0)
        json_every = atof(e);
    if (h && *h)
        human_every = atof(h) > 0 ? atof(h) : 10;
}

static int on(void)
{
    pthread_once(&once, set_up);
    return json_fd >= 0 || human_every > 0;
}

/* One JSON line: the event, its time, and the members (text without the
   braces), written in one piece so lines from several threads never mix. */
static void emit(const char *event, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static void emit(const char *event, const char *fmt, ...)
{
    if (json_fd < 0)
        return;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm;
    gmtime_r(&tv.tv_sec, &tm);
    char when[40];
    strftime(when, sizeof(when), "%Y-%m-%dT%H:%M:%S", &tm);
    struct janas_buf b = {0};
    janas_buf_printf(&b, "{\"event\":\"%s\",\"time\":\"%s.%03dZ\",\"pid\":%d",
                     event, when, (int)(tv.tv_usec / 1000), (int)getpid());
    va_list ap;
    va_start(ap, fmt);
    char body[2048];
    vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);
    if (*body)
        janas_buf_printf(&b, ",%s", body);
    janas_buf_puts(&b, "}\n");
    if (!b.oom) {
        pthread_mutex_lock(&mu);
        ssize_t w = write(json_fd, b.p, b.n);
        (void)w;
        pthread_mutex_unlock(&mu);
    }
    janas_buf_free(&b);
}

static void human(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void human(const char *fmt, ...)
{
    if (human_every <= 0)
        return;
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    fprintf(stderr, "janas: %s\n", line);
}

void janas_metrics_open(const janas_llm *llm, double seconds)
{
    if (!on())
        return;
    struct janas_buf name = {0};
    janas_json_write_str(&name, llm->name, strlen(llm->name));
    emit("open",
         "\"model\":%s,\"arch\":\"%s\",\"context_size\":%u,"
         "\"cache_bytes\":%llu,\"compute_threads\":%d,\"expert_bits\":%d,"
         "\"attention\":\"%s\",\"load_seconds\":%.3f",
         name.p ? name.p : "\"\"", llm->arch, llm->n_ctx,
         (unsigned long long)llm->cache_bytes, llm->n_compute,
         janas_llm_model_expert_bits(llm->m),
         janas_llm_model_attn_scores(llm->m) == JANAS_LLM_ATTN_FAST ? "fast"
                                                                    : "exact",
         seconds);
    janas_buf_free(&name);
    human("model opened in %.1f s: %s, context %u tokens", seconds, llm->name,
          llm->n_ctx);
}

/* The prompt's parts, as JSON members. */
static void parts(char *buf, size_t n, const struct janas_llm_chat_stats *s)
{
    snprintf(buf, n,
             "\"prompt_tokens\":%u,\"cached_tokens\":%u,\"input_tokens\":%u,"
             "\"system_tokens\":%u,\"tools_tokens\":%u,\"history_tokens\":%u,"
             "\"last_tokens\":%u,\"context_size\":%u",
             s->prompt_tokens, s->cached_tokens, s->input_tokens,
             s->prompt_system, s->prompt_tools, s->prompt_history,
             s->prompt_last, s->context_size);
}

void janas_metrics_prompt(janas_llm_chat *c)
{
    if (!on())
        return;
    const struct janas_llm_chat_stats *s = &c->stats;
    pthread_mutex_lock(&mu);
    c->m_seq = ++seq;
    pthread_mutex_unlock(&mu);
    c->m_last = janas_api_now();
    c->m_last_done = 0;
    char p[512];
    parts(p, sizeof(p), s);
    uint32_t reserve = c->p.max_reply > 0 && c->p.max_reply < 1024
                           ? (uint32_t)c->p.max_reply
                           : 1024;
    uint32_t free_tokens = s->context_size > s->prompt_tokens
                               ? s->context_size - s->prompt_tokens
                               : 0;
    emit("prompt",
         "\"reply\":%lu,%s,\"reply_reserve\":%u,\"max_input\":%u,"
         "\"context_free\":%u,\"context_utilization\":%.4f,"
         "\"prepare_seconds\":%.3f",
         c->m_seq, p, reserve, c->p.max_input, free_tokens,
         s->context_size ? (double)s->prompt_tokens / s->context_size : 0.0,
         s->prepare_seconds);
    if (s->input_tokens >= 1024)
        human("reply %lu: reading %u tokens of prompt (%u already computed; "
              "system %u, of which tools %u, history %u, last %u; context "
              "%u of %u)",
              c->m_seq, s->input_tokens, s->cached_tokens, s->prompt_system,
              s->prompt_tools, s->prompt_history, s->prompt_last,
              s->prompt_tokens, s->context_size);
}

void janas_metrics_progress(janas_llm_chat *c, int force)
{
    if (!on())
        return;
    const struct janas_llm_chat_stats *s = &c->stats;
    double now = janas_api_now(), dt = now - c->m_last;
    double every = json_fd >= 0 ? json_every : human_every;
    if (json_fd >= 0 && human_every > 0 && human_every < every)
        every = human_every;
    if (!force && dt < every)
        return;
    double elapsed = now - c->t_send;
    double rate = dt > 0 ? (s->input_done - c->m_last_done) / dt : 0;
    double avg = s->input_seconds > 0 ? s->input_done / s->input_seconds : 0;
    /* an estimate that follows the cost of a token as the context grows
       (janas_llm_chat_stats.input_eta_seconds) */
    double eta = s->input_eta_seconds;
    emit("progress",
         "\"reply\":%lu,\"stage\":\"input\",\"input_done\":%u,"
         "\"input_tokens\":%u,\"elapsed_seconds\":%.1f,"
         "\"tokens_per_second\":%.2f,\"tokens_per_second_avg\":%.2f,"
         "\"eta_seconds_estimate\":%.0f",
         c->m_seq, s->input_done, s->input_tokens, elapsed, rate, avg, eta);
    if (human_every > 0 && (force || dt >= human_every)) {
        char e[48] = "unknown";
        if (eta >= 3600)
            snprintf(e, sizeof(e), "~%dh%02dm", (int)(eta / 3600),
                     (int)(eta / 60) % 60);
        else if (eta >= 60)
            snprintf(e, sizeof(e), "~%dm%02ds", (int)(eta / 60), (int)eta % 60);
        else if (eta >= 0)
            snprintf(e, sizeof(e), "~%ds", (int)eta);
        human("reply %lu: prompt %u / %u (%.1f%%), %.1f tok/s now, %.1f "
              "average, %.0f s so far, %s left (estimate)",
              c->m_seq, s->input_done, s->input_tokens,
              s->input_tokens ? 100.0 * s->input_done / s->input_tokens : 100.0,
              rate, avg, elapsed, e);
    }
    c->m_last = now;
    c->m_last_done = s->input_done;
}

void janas_metrics_first_token(janas_llm_chat *c)
{
    if (!on())
        return;
    emit("first_token", "\"reply\":%lu,\"first_token_seconds\":%.3f", c->m_seq,
         c->stats.first_token_seconds);
}

void janas_metrics_reply(janas_llm_chat *c)
{
    if (!on())
        return;
    const struct janas_llm_chat_stats *s = &c->stats;
    static const char *const finish[] = {"",        "stop",  "length",
                                         "context", "error", "tools"};
    const char *f = s->finish >= 0 && s->finish <= 5 ? finish[s->finish] : "";
    char p[512];
    parts(p, sizeof(p), s);
    emit("reply",
         "\"reply\":%lu,\"finish\":\"%s\",%s,\"output_tokens\":%u,"
         "\"context_used\":%u,\"prepare_seconds\":%.3f,"
         "\"input_seconds\":%.3f,\"input_cpu_seconds\":%.3f,"
         "\"input_tokens_per_second\":%.3f,\"first_token_seconds\":%.3f,"
         "\"output_seconds\":%.3f,\"output_tokens_per_second\":%.3f,"
         "\"total_seconds\":%.3f,\"input_threads\":%d,\"output_threads\":%d,"
         "\"experts_used\":%llu,\"experts_read\":%llu,\"bytes_read\":%llu,"
         "\"io_wait_seconds\":%.3f,\"context_bytes\":%llu,"
         "\"peak_rss_bytes\":%llu,\"drafted\":%u,\"accepted\":%u",
         c->m_seq, f, p, s->output_tokens, s->context_used, s->prepare_seconds,
         s->input_seconds, s->input_cpu_seconds,
         s->input_seconds > 0 ? s->input_tokens / s->input_seconds : 0.0,
         s->first_token_seconds, s->output_seconds,
         s->output_seconds > 0 ? s->output_tokens / s->output_seconds : 0.0,
         s->total_seconds, s->input_threads, s->output_threads,
         (unsigned long long)s->experts_used,
         (unsigned long long)s->experts_read, (unsigned long long)s->bytes_read,
         s->io_wait_seconds, (unsigned long long)s->context_bytes,
         (unsigned long long)s->peak_rss_bytes, s->drafted, s->accepted);
    if (s->input_tokens >= 1024 || human_every > 0)
        human("reply %lu: %s; prompt %u tokens in %.1f s (%.1f tok/s, first "
              "token after %.1f s), reply %u tokens in %.1f s (%.2f tok/s), "
              "%.0f MB read, context %u of %u",
              c->m_seq, *f ? f : "over", s->input_tokens, s->input_seconds,
              s->input_seconds > 0 ? s->input_tokens / s->input_seconds : 0.0,
              s->first_token_seconds, s->output_tokens, s->output_seconds,
              s->output_seconds > 0 ? s->output_tokens / s->output_seconds
                                    : 0.0,
              (double)s->bytes_read / 1e6, s->context_used, s->context_size);
}

void janas_metrics_refused(janas_llm_chat *c, uint32_t max_input)
{
    if (!on())
        return;
    char p[512];
    parts(p, sizeof(p), &c->stats);
    emit("refused", "%s,\"max_input\":%u,\"excess_tokens\":%u", p, max_input,
         c->stats.prompt_tokens > max_input ? c->stats.prompt_tokens - max_input
                                            : 0);
    human("prompt refused: %u tokens, max_input %u", c->stats.prompt_tokens,
          max_input);
}
