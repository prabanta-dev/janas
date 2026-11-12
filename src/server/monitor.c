/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * monitor.c - the engine at work, in a page (discussion 61): /monitor is
 * the page (monitor.html, carried by the program), /monitor/events its
 * live data, one event a second as Server-Sent Events: the reply being
 * written, the last replies, the totals since the start, the expert cache
 * and the memory. Counters and times only, never the text of a prompt or a
 * reply. The data is behind the server's key, where one is set; the page
 * asks for it, as /test does. With ?layers=1 the events also carry a look
 * inside the layers (janas_llm_monitor_layers), which the engine gathers,
 * and pays for, only while a stream asks for it.
 */
#include <pthread.h>
#include <stdarg.h>
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
    char *out;
    size_t len, pos, cap;
    int bad;
    /* the look inside the layers, when the page asked for it
       (?layers=1): its shape and what each event takes */
    int layers;
    uint32_t shape[3];
    float *ffn, *res, *strength;
    uint32_t *chosen, *read;
};

/* appended to the event, the buffer grown as needed */
static void put(struct mon_stream *ms, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static void put(struct mon_stream *ms, const char *fmt, ...)
{
    while (!ms->bad) {
        va_list ap;
        va_start(ap, fmt);
        int k = vsnprintf(ms->out + ms->len, ms->cap - ms->len, fmt, ap);
        va_end(ap);
        if (k < 0) {
            ms->bad = 1;
            return;
        }
        if ((size_t)k < ms->cap - ms->len) {
            ms->len += (size_t)k;
            return;
        }
        size_t want = (ms->len + (size_t)k + 1) * 2;
        char *bigger = realloc(ms->out, want);
        if (!bigger) {
            ms->bad = 1;
            return;
        }
        ms->out = bigger;
        ms->cap = want;
    }
}

/* a string, as JSON wants it */
static void put_str(struct mon_stream *ms, const char *s)
{
    put(ms, "\"");
    for (const char *c = s; c && *c; c++) {
        if (*c == '"' || *c == '\\')
            put(ms, "\\%c", *c);
        else if ((unsigned char)*c >= 0x20)
            put(ms, "%c", *c);
    }
    put(ms, "\"");
}

static void put_floats(struct mon_stream *ms, const char *name, const float *v,
                       size_t n)
{
    put(ms, ",\"%s\":[", name);
    for (size_t i = 0; i < n; i++)
        put(ms, i ? ",%.4g" : "%.4g", (double)v[i]);
    put(ms, "]");
}

static void put_counts(struct mon_stream *ms, const char *name,
                       const uint32_t *v, size_t n)
{
    put(ms, ",\"%s\":[", name);
    for (size_t i = 0; i < n; i++)
        put(ms, i ? ",%u" : "%u", v[i]);
    put(ms, "]");
}

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

/* One event, "data: {...}\n\n", made in ms->out; 0, or -1 when it could
   not be */
static int event(struct mon_stream *ms)
{
    janas_llm *llm = srv_engine_llm(ms->srv->engine);
    struct janas_llm_monitor m = {.size = sizeof m};
    if (janas_llm_monitor(llm, &m) != JANAS_LLM_OK)
        return -1;
    const struct srv_model_info *mi = srv_engine_info(ms->srv->engine);
    ms->len = ms->pos = 0;
    ms->bad = 0;
    put(ms, "data: {\"t\":%.3f,\"model\":", now() - ms->t0);
    put_str(ms, ms->srv->cfg.model_id);
    put(ms, ",\"arch\":");
    put_str(ms, mi ? mi->arch : "");
    put(ms, ",\"description\":");
    put_str(ms, srv_engine_describe(ms->srv->engine));
    char cpu[128];
    janas_cpu_name(cpu, sizeof cpu);
    put(ms, ",\"cpu\":");
    put_str(ms, cpu);
    put(ms, ",\"choices\":");
    put_str(ms, m.choices);
    put(ms,
        ",\"on_mains\":%d,\"queued\":%d,\"replies\":%llu,\"input_tokens\":%llu,"
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
    /* level 1: the phases' seconds since the start, in the order of
       janas_llm_monitor.phase_seconds */
    put(ms, ",\"phases\":[");
    for (int i = 0; i < 8; i++)
        put(ms, i ? ",%.4f" : "%.4f", m.phase_seconds[i]);
    put(ms, "]");
    /* the latest reply every time; the ones before it when they changed */
    int all = ms->sent == 0 || m.replies != ms->replies;
    struct janas_llm_chat_stats s[16];
    int ns = 0;
    for (; ns < (all ? 16 : 1); ns++) {
        s[ns] = (struct janas_llm_chat_stats){.size = sizeof s[ns]};
        if (janas_llm_monitor_reply(llm, ns, &s[ns]) != JANAS_LLM_OK)
            break;
    }
    for (int k = 0; k < ns; k++) {
        char one[2048];
        reply_json(one, sizeof one, &s[k]);
        put(ms, "%s%s",
            k == 0   ? ",\"latest\":"
            : k == 1 ? ",\"recent\":["
                     : ",",
            one);
    }
    if (ns > 1)
        put(ms, "]");
    /* level 2: the layers, when asked and gathered */
    uint64_t passes = 0;
    if (ms->layers &&
        janas_llm_monitor_layers(llm, ms->ffn, ms->res, ms->chosen, ms->read,
                                 ms->strength, &passes) == JANAS_LLM_OK) {
        size_t L = ms->shape[0], B = ms->shape[1], E = ms->shape[2];
        put(ms,
            ",\"layers\":{\"n\":%u,\"bins\":%u,\"experts\":%u,"
            "\"passes\":%llu",
            (unsigned)L, (unsigned)B, (unsigned)E, (unsigned long long)passes);
        put_floats(ms, "ffn", ms->ffn, L * B);
        put_floats(ms, "resid", ms->res, L * B);
        if (E) {
            put_counts(ms, "chosen", ms->chosen, L * E);
            put_counts(ms, "read", ms->read, L * E);
            put_floats(ms, "strength", ms->strength, L * E);
        }
        put(ms, "}");
    } else if (ms->layers)
        put(ms, ",\"layers\":null");
    put(ms, "}\n\n");
    if (ms->bad)
        return -1;
    ms->replies = m.replies;
    ms->sent++;
    return 0;
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
    if (ms->pos == ms->len && event(ms) != 0)
        return -1;
    size_t k = ms->len - ms->pos < cap ? ms->len - ms->pos : cap;
    memcpy(buf, ms->out + ms->pos, k);
    ms->pos += k;
    return (long)k;
}

static void mon_free(struct mon_stream *ms)
{
    free(ms->out);
    free(ms->ffn);
    free(ms->res);
    free(ms->strength);
    free(ms->chosen);
    free(ms->read);
    free(ms);
}

static void mon_done(void *ctx)
{
    struct mon_stream *ms = ctx;
    if (ms->layers) /* one fewer wants it: the engine stops paying for it
                       when nobody does */
        janas_llm_monitor_layers_on(srv_engine_llm(ms->srv->engine), 0);
    mon_free(ms);
}

int srv_monitor_events(struct srv_req *r)
{
    struct mon_stream *ms = calloc(1, sizeof *ms);
    if (!ms)
        return srv_reply_error(r, 500, "server_error", NULL, "Out of memory.");
    ms->srv = r->srv;
    ms->t0 = now();
    const char *lv = srv_query(r, "layers");
    if (lv && strcmp(lv, "1") == 0) {
        janas_llm *llm = srv_engine_llm(r->srv->engine);
        janas_llm_monitor_layers_shape(llm, ms->shape);
        size_t L = ms->shape[0], B = ms->shape[1], E = ms->shape[2];
        size_t LE = L * (E ? E : 1);
        ms->ffn = malloc(L * B * sizeof(float));
        ms->res = malloc(L * B * sizeof(float));
        ms->strength = malloc(LE * sizeof(float));
        ms->chosen = malloc(LE * sizeof(uint32_t));
        ms->read = malloc(LE * sizeof(uint32_t));
        if (!ms->ffn || !ms->res || !ms->strength || !ms->chosen || !ms->read) {
            mon_free(ms);
            return srv_reply_error(r, 500, "server_error", NULL,
                                   "Out of memory.");
        }
        ms->layers = 1;
        janas_llm_monitor_layers_on(llm, 1);
    }
    return srv_reply_stream(r, mon_next, mon_done, ms);
}
