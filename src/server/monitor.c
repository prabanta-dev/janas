/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * monitor.c - the engine at work, in a page (discussion 61): /monitor is
 * the page (monitor.html, carried by the program), /monitor/events its
 * live data, one event a second as Server-Sent Events: the reply being
 * written, the last replies, the totals since the start, the expert cache
 * and the memory. Counters and times only, never the text of a prompt or a
 * reply. The data is behind the server's key, where one is set; the page
 * asks for it, as /test does.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common/sysinfo.h"
#include "server.h"

static _Atomic int stopping;

void srv_monitor_stop(void)
{
    atomic_store(&stopping, 1);
}

struct mon_stream {
    struct srv *srv;
    int sent;         /* events made */
    uint64_t replies; /* the totals' count when the last list was sent */
    double t0;
    /* the event being handed out, a piece a call */
    char out[65536];
    size_t len, pos;
};

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char *stage_name(int32_t s)
{
    return s == JANAS_LLM_STAGE_INPUT    ? "input"
           : s == JANAS_LLM_STAGE_OUTPUT ? "output"
           : s == JANAS_LLM_STAGE_DONE   ? "done"
                                         : "idle";
}

/* A reply's counters as a JSON object, appended at p (n left). */
static int reply_json(char *p, size_t n, const struct janas_llm_chat_stats *s)
{
    return snprintf(
        p, n,
        "{\"stage\":\"%s\",\"finish\":%d,\"prompt_tokens\":%u,"
        "\"cached_tokens\":%u,\"input_tokens\":%u,\"input_done\":%u,"
        "\"output_tokens\":%u,\"reasoning_tokens\":%u,"
        "\"input_seconds\":%.3f,\"output_seconds\":%.3f,"
        "\"first_token_seconds\":%.3f,\"total_seconds\":%.3f,"
        "\"input_eta_seconds\":%.1f,\"drafted\":%u,\"accepted\":%u,"
        "\"experts_used\":%llu,\"experts_read\":%llu,\"bytes_read\":%llu,"
        "\"io_wait_seconds\":%.3f,\"context_used\":%u,\"context_size\":%u,"
        "\"context_bytes\":%llu,\"input_threads\":%d,\"output_threads\":%d,"
        "\"prompt_system\":%u,\"prompt_tools\":%u,\"prompt_history\":%u,"
        "\"prompt_last\":%u,\"reasoning_cut\":%d,\"auto_off_passes\":%u,"
        "\"prepare_seconds\":%.3f,\"input_cpu_seconds\":%.3f,"
        "\"peak_rss\":%llu}",
        stage_name(s->stage), s->finish, s->prompt_tokens, s->cached_tokens,
        s->input_tokens, s->input_done, s->output_tokens, s->reasoning_tokens,
        s->input_seconds, s->output_seconds, s->first_token_seconds,
        s->total_seconds, s->input_eta_seconds, s->drafted, s->accepted,
        (unsigned long long)s->experts_used,
        (unsigned long long)s->experts_read, (unsigned long long)s->bytes_read,
        s->io_wait_seconds, s->context_used, s->context_size,
        (unsigned long long)s->context_bytes, s->input_threads,
        s->output_threads, s->prompt_system, s->prompt_tools, s->prompt_history,
        s->prompt_last, s->reasoning_cut, s->auto_off_passes,
        s->prepare_seconds, s->input_cpu_seconds,
        (unsigned long long)s->peak_rss_bytes);
}

/* One event: "data: {...}\n\n" in buf, its length; 0 when it did not fit */
static long event(struct mon_stream *ms, char *buf, size_t cap)
{
    janas_llm *llm = srv_engine_llm(ms->srv->engine);
    struct janas_llm_monitor m = {.size = sizeof m};
    if (janas_llm_monitor(llm, &m) != JANAS_LLM_OK)
        return 0;
    const struct srv_model_info *mi = srv_engine_info(ms->srv->engine);
    size_t at = 0;
#define PUT(...)                                                               \
    do {                                                                       \
        int k_ = snprintf(buf + at, cap - at, __VA_ARGS__);                    \
        if (k_ < 0 || (size_t)k_ >= cap - at)                                  \
            return 0;                                                          \
        at += (size_t)k_;                                                      \
    } while (0)
    PUT("data: {\"t\":%.3f,\"model\":\"", now() - ms->t0);
    /* the model's name, as JSON wants it */
    for (const char *c = ms->srv->cfg.model_id; c && *c; c++) {
        if (*c == '"' || *c == '\\')
            PUT("\\%c", *c);
        else if ((unsigned char)*c >= 0x20)
            PUT("%c", *c);
    }
    PUT("\",\"arch\":\"%s\",\"description\":\"", mi ? mi->arch : "");
    for (const char *c = srv_engine_describe(ms->srv->engine); c && *c; c++) {
        if (*c == '"' || *c == '\\')
            PUT("\\%c", *c);
        else if ((unsigned char)*c >= 0x20)
            PUT("%c", *c);
    }
    char cpu[128];
    janas_cpu_name(cpu, sizeof cpu);
    PUT("\",\"cpu\":\"");
    for (const char *c = cpu; *c; c++)
        if (*c != '"' && *c != '\\' && (unsigned char)*c >= 0x20)
            PUT("%c", *c);
    PUT("\",\"on_mains\":%d,\"queued\":%d,\"replies\":%llu,\"input_tokens\":%llu,"
        "\"output_tokens\":%llu,\"stage\":\"%s\",\"context_size\":%u,"
        "\"cache\":{\"slots\":%llu,\"slot_bytes\":%llu,\"used\":%llu,"
        "\"hit\":%llu,\"read\":%llu,\"bytes_read\":%llu,\"warm_bytes\":%llu,"
        "\"io_wait_seconds\":%.3f,\"missed\":%llu,\"prefetch_reads\":%llu,"
        "\"prefetch_used\":%llu,\"prefetch_bytes\":%llu,"
        "\"io_seconds\":%.3f},\"tier\":{\"hits\":%llu,\"loads\":%llu,"
        "\"cpu\":%llu,\"slots\":%llu,\"slot_bytes\":%llu},"
        "\"memory\":{\"rss\":%llu,\"total\":%llu,\"available\":%llu},"
        "\"cpu_seconds\":%.3f,\"cpus\":%d,\"energy_j\":%.3f",
        janas_on_mains(), srv_engine_queued(ms->srv->engine),
        (unsigned long long)m.replies, (unsigned long long)m.input_tokens,
        (unsigned long long)m.output_tokens, stage_name(m.stage),
        m.context_size, (unsigned long long)m.cache_slots,
        (unsigned long long)m.slot_bytes, (unsigned long long)m.experts_used,
        (unsigned long long)m.experts_hit, (unsigned long long)m.experts_read,
        (unsigned long long)m.bytes_read, (unsigned long long)m.warm_bytes,
        m.io_wait_seconds, (unsigned long long)m.experts_missed,
        (unsigned long long)m.prefetch_reads,
        (unsigned long long)m.prefetch_used,
        (unsigned long long)m.prefetch_bytes, m.io_seconds,
        (unsigned long long)m.tier_hits, (unsigned long long)m.tier_loads,
        (unsigned long long)m.tier_cpu, (unsigned long long)m.tier_slots,
        (unsigned long long)m.tier_slot_bytes, (unsigned long long)m.rss_bytes,
        (unsigned long long)m.memory_bytes,
        (unsigned long long)m.memory_available, janas_cpu_seconds(),
        janas_cpu_count(), janas_package_joules());
    /* the latest reply every time; the ones before it when they changed */
    int all = ms->sent == 0 || m.replies != ms->replies;
    ms->replies = m.replies;
    struct janas_llm_chat_stats s[16];
    int ns = 0;
    for (; ns < (all ? 16 : 1); ns++) {
        s[ns] = (struct janas_llm_chat_stats){.size = sizeof s[ns]};
        if (janas_llm_monitor_reply(llm, ns, &s[ns]) != JANAS_LLM_OK)
            break;
    }
    for (int k = 0; k < ns; k++) {
        PUT(k == 0 ? ",\"latest\":" : k == 1 ? ",\"recent\":[" : ",");
        int w = reply_json(buf + at, cap - at, &s[k]);
        if (w < 0 || (size_t)w >= cap - at)
            return 0;
        at += (size_t)w;
    }
    if (ns > 1)
        PUT("]");
    PUT("}\n\n");
#undef PUT
    ms->sent++;
    return (long)at;
}

static long mon_next(void *ctx, char *buf, size_t cap)
{
    struct mon_stream *ms = ctx;
    if (atomic_load(&stopping))
        return -1;
    if (ms->sent && ms->pos == ms->len) /* one a second; the first at once */
        for (int i = 0; i < 10 && !atomic_load(&stopping); i++)
            usleep(100000);
    if (atomic_load(&stopping))
        return -1;
    if (ms->pos == ms->len) {
        long n = event(ms, ms->out, sizeof ms->out);
        if (n <= 0)
            return -1;
        ms->len = (size_t)n;
        ms->pos = 0;
    }
    size_t k = ms->len - ms->pos < cap ? ms->len - ms->pos : cap;
    memcpy(buf, ms->out + ms->pos, k);
    ms->pos += k;
    return (long)k;
}

static void mon_done(void *ctx)
{
    free(ctx);
}

int srv_monitor_events(struct srv_req *r)
{
    struct mon_stream *ms = calloc(1, sizeof *ms);
    if (!ms)
        return srv_reply_error(r, 500, "server_error", NULL, "Out of memory.");
    ms->srv = r->srv;
    ms->t0 = now();
    return srv_reply_stream(r, mon_next, mon_done, ms);
}
