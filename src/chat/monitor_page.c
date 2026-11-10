/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * monitor_page.c - see monitor_page.h.
 *
 * Laid out as everything in the chat is: a title on its own line and the
 * facts under it, each a line that wraps, nothing in fixed columns, so that
 * a narrow window reads it as well as a wide one.
 */
#include "monitor_page.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/sysinfo.h"

struct text {
    char *s;
    size_t n, cap;
    int bad;
};

static void put(struct text *t, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static void put(struct text *t, const char *fmt, ...)
{
    if (t->bad)
        return;
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        int k = vsnprintf(t->s + t->n, t->cap - t->n, fmt, ap);
        va_end(ap);
        if (k < 0) {
            t->bad = 1;
            return;
        }
        if ((size_t)k < t->cap - t->n) {
            t->n += (size_t)k;
            return;
        }
        size_t want = (t->n + (size_t)k + 1) * 2;
        char *bigger = realloc(t->s, want);
        if (!bigger) {
            t->bad = 1;
            return;
        }
        t->s = bigger;
        t->cap = want;
    }
}

/* bytes with the unit that reads best */
static const char *size(uint64_t b, char *buf, size_t len)
{
    if (b >= 1000000000ull)
        snprintf(buf, len, "%.1f GB", (double)b / 1e9);
    else if (b >= 1000000ull)
        snprintf(buf, len, "%.0f MB", (double)b / 1e6);
    else
        snprintf(buf, len, "%.0f kB", (double)b / 1e3);
    return buf;
}

/* a share as a bar of 20 cells, then the number */
static void bar(struct text *t, double share)
{
    if (share < 0)
        share = 0;
    if (share > 1)
        share = 1;
    int full = (int)(share * 20 + 0.5);
    put(t, "  ");
    for (int i = 0; i < 20; i++)
        put(t, "%s", i < full ? "█" : "░");
    put(t, " %.1f%%\n", share * 100);
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char *finish(int32_t f)
{
    switch (f) {
    case JANAS_LLM_FINISH_STOP:
        return "the model ended it";
    case JANAS_LLM_FINISH_LENGTH:
        return "the length asked ended it";
    case JANAS_LLM_FINISH_CONTEXT:
        return "the full context ended it";
    case JANAS_LLM_FINISH_ERROR:
        return "an error ended it";
    case JANAS_LLM_FINISH_TOOLS:
        return "it called tools";
    default:
        return "";
    }
}

static double per(double n, double s)
{
    return s > 0 ? n / s : 0;
}

/* One reply in a line or two: the prompt, the reply, what it read. */
static void reply_line(struct text *t, const struct janas_llm_chat_stats *r)
{
    char b[32];
    put(t, "  prompt %u tokens (%u kept from before)", r->prompt_tokens,
        r->cached_tokens);
    if (r->input_seconds > 0)
        put(t, ", read at %.1f tok/s", per(r->input_tokens, r->input_seconds));
    if (r->first_token_seconds > 0)
        put(t, ", first token after %.2f s", r->first_token_seconds);
    put(t, "; reply %u tokens", r->output_tokens);
    if (r->output_seconds > 0)
        put(t, " at %.1f tok/s", per(r->output_tokens, r->output_seconds));
    if (r->drafted)
        put(t, ", drafts %.0f%% taken", 100.0 * r->accepted / r->drafted);
    if (r->experts_used)
        put(t, ", experts %.1f%% in memory",
            100.0 - 100.0 * (double)r->experts_read / (double)r->experts_used);
    if (r->bytes_read)
        put(t, ", %s read from the disk", size(r->bytes_read, b, sizeof b));
    if (r->stage == JANAS_LLM_STAGE_DONE && finish(r->finish)[0])
        put(t, "; %s", finish(r->finish));
    put(t, "\n");
}

char *chat_monitor_text(void *arg)
{
    struct chat_monitor *cm = arg;
    struct text t = {.s = malloc(4096), .cap = 4096};
    if (!t.s)
        return NULL;
    t.s[0] = 0;
    struct janas_llm_monitor m = {.size = sizeof m};
    if (janas_llm_monitor(cm->llm, &m) != JANAS_LLM_OK) {
        put(&t, "The engine's counters cannot be read.\n");
        return t.s;
    }
    double tn = now(), cpu = janas_cpu_seconds(),
           joules = janas_package_joules();
    double dt = cm->have ? tn - cm->t : 0;
    char b1[32], b2[32], b3[32];

    put(&t, "The engine at work, a sample a second\n\n");
    put(&t, "Since the start\n");
    put(&t, "  %llu %s, %llu tokens read, %llu written\n\n",
        (unsigned long long)m.replies, m.replies == 1 ? "reply" : "replies",
        (unsigned long long)m.input_tokens,
        (unsigned long long)m.output_tokens);

    struct janas_llm_chat_stats r = {.size = sizeof r};
    if (janas_llm_monitor_reply(cm->llm, 0, &r) == JANAS_LLM_OK) {
        put(&t, "The latest reply%s\n",
            r.stage == JANAS_LLM_STAGE_INPUT    ? ", reading its prompt"
            : r.stage == JANAS_LLM_STAGE_OUTPUT ? ", being written"
                                                : "");
        put(&t,
            "  prompt %u tokens: system message %u (tools' descriptions "
            "%u), conversation before %u, last message %u; %u kept from "
            "before, %u read",
            r.prompt_tokens, r.prompt_system, r.prompt_tools, r.prompt_history,
            r.prompt_last, r.cached_tokens, r.input_tokens);
        if (r.input_seconds > 0)
            put(&t, " at %.1f tok/s with %d threads",
                per(r.input_tokens, r.input_seconds), r.input_threads);
        if (r.input_cpu_seconds > 0 && r.input_seconds > 0)
            put(&t, " (%.1f busy)", r.input_cpu_seconds / r.input_seconds);
        if (r.prepare_seconds > 0)
            put(&t, ", built in %.2f s", r.prepare_seconds);
        put(&t, "\n");
        if (r.first_token_seconds > 0)
            put(&t, "  first token after %.2f s\n", r.first_token_seconds);
        put(&t, "  reply %u tokens", r.output_tokens);
        if (r.output_seconds > 0)
            put(&t, " at %.1f tok/s with %d threads",
                per(r.output_tokens, r.output_seconds), r.output_threads);
        if (r.reasoning_tokens)
            put(&t, ", %u of them reasoning%s", r.reasoning_tokens,
                r.reasoning_cut ? " (cut at the limit)" : "");
        if (r.drafted)
            put(&t, "; drafts %u, %.0f%% taken", r.drafted,
                100.0 * r.accepted / r.drafted);
        if (r.auto_off_passes)
            put(&t, "; %u passes without drafts, which were not paying",
                r.auto_off_passes);
        if (r.stage == JANAS_LLM_STAGE_DONE && finish(r.finish)[0])
            put(&t, "; %s", finish(r.finish));
        put(&t, "\n");
        put(&t, "  context %u of %u tokens, %s of keys and values\n\n",
            r.context_used, r.context_size,
            size(r.context_bytes, b1, sizeof b1));
    }

    put(&t, "Experts\n");
    if (m.cache_slots) {
        put(&t, "  found in memory since the start\n");
        bar(&t, m.experts_used ? (double)m.experts_hit / m.experts_used : 0);
        if (dt > 0 && m.experts_used > cm->used)
            put(&t, "  in the last second %.1f%% of %llu\n",
                100.0 * (double)(m.experts_hit - cm->hit) /
                    (double)(m.experts_used - cm->used),
                (unsigned long long)(m.experts_used - cm->used));
        put(&t,
            "  read from the disk: %s for the replies, %s filling the "
            "cache",
            size(m.bytes_read, b1, sizeof b1),
            size(m.warm_bytes, b2, sizeof b2));
        if (dt > 0)
            put(&t, "; now %.0f MB/s",
                (double)(m.bytes_read + m.warm_bytes - cm->bytes) / 1e6 / dt);
        put(&t, "\n");
        if (m.prefetch_reads)
            put(&t, "  read ahead %llu, %.1f%% of them used, %s\n",
                (unsigned long long)m.prefetch_reads,
                100.0 * (double)m.prefetch_used / (double)m.prefetch_reads,
                size(m.prefetch_bytes, b1, sizeof b1));
        put(&t,
            "  the reads took %.1f s in the background, the replies waited "
            "%.1f s\n",
            m.io_seconds, m.io_wait_seconds);
        put(&t, "  cache %llu slots, %s\n", (unsigned long long)m.cache_slots,
            size(m.cache_slots * m.slot_bytes, b1, sizeof b1));
    } else
        put(&t, "  no expert cache: a dense model\n");
    uint64_t tier = m.tier_hits + m.tier_loads + m.tier_cpu;
    if (m.tier_slots && tier)
        put(&t,
            "  on the card: %.1f%% found there, %.1f%% read into it, "
            "%.1f%% computed by the CPU; %llu slots, %s\n",
            100.0 * (double)m.tier_hits / (double)tier,
            100.0 * (double)m.tier_loads / (double)tier,
            100.0 * (double)m.tier_cpu / (double)tier,
            (unsigned long long)m.tier_slots,
            size(m.tier_slots * m.tier_slot_bytes, b1, sizeof b1));
    put(&t, "\n");

    put(&t, "Memory and processor\n");
    put(&t, "  the process holds %s of %s; %s available\n",
        size(m.rss_bytes, b1, sizeof b1), size(m.memory_bytes, b2, sizeof b2),
        size(m.memory_available, b3, sizeof b3));
    if (m.memory_bytes)
        bar(&t, (double)m.rss_bytes / (double)m.memory_bytes);
    if (dt > 0 && cpu >= 0 && cm->cpu >= 0)
        put(&t, "  threads busy %.1f of %d\n", (cpu - cm->cpu) / dt,
            janas_cpu_count());
    if (dt > 0 && joules >= 0 && cm->joules >= 0)
        put(&t, "  the whole processor %.1f W\n", (joules - cm->joules) / dt);
    put(&t, "\n");

    int shown = 0;
    for (int k = 1; k < 16; k++) {
        struct janas_llm_chat_stats o = {.size = sizeof o};
        if (janas_llm_monitor_reply(cm->llm, k, &o) != JANAS_LLM_OK)
            break;
        if (!shown++)
            put(&t, "The replies before, the latest first\n");
        reply_line(&t, &o);
    }
    if (shown)
        put(&t, "\n");
    put(&t, "Counters and times only, never the text of a prompt or a "
            "reply.\n");

    cm->have = 1;
    cm->t = tn;
    cm->cpu = cpu;
    cm->joules = joules;
    cm->bytes = m.bytes_read + m.warm_bytes;
    cm->used = m.experts_used;
    cm->hit = m.experts_hit;
    if (t.bad) {
        free(t.s);
        return NULL;
    }
    return t.s;
}
