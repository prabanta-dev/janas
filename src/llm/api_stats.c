/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_stats.c - the engine at work, for a monitor (janas_llm_monitor and
 * janas_llm_monitor_reply in janas/llm.h): the counters a reply keeps
 * (janas_llm_chat_stats) copied, at every step, where another thread can
 * read them, with the last replies and the totals since the model opened.
 *
 * A step of a reply already updates its counters (janas_api_update_stats);
 * the copy is one struct under a lock nobody else holds for long, so a
 * monitor costs the reply nothing it would notice, whether one is reading
 * or not.
 */
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"
#include "common/sysinfo.h"

#define MON_REPLIES 16

struct janas_api_monitor {
    pthread_mutex_t mu;
    /* the replies, the latest at head; a reply is told from the one before
       by its chat and the time it was sent */
    struct janas_llm_chat_stats ring[MON_REPLIES];
    uint32_t head, n;
    const janas_llm_chat *chat;
    double t_send;
    /* of the replies before the latest */
    uint64_t replies, input_tokens, output_tokens;
    /* the card's tier, copied on the replying thread, which alone
       changes it */
    uint64_t tier[5];
    /* the phases' times since the model opened, and the tuner's choices,
       copied there too (the choices once a second: they are words) */
    double phase[JANAS_PH_COUNT];
    char choices[256];
    double t_choices;
    /* the look inside the layers: how many want it, and whether the
       replying thread has attached it (it alone attaches and detaches) */
    int layers_want, layers_on;
};

static pthread_mutex_t make_mu = PTHREAD_MUTEX_INITIALIZER;

static struct janas_api_monitor *monitor_of(janas_llm *llm)
{
    pthread_mutex_lock(&make_mu);
    if (!llm->mon) {
        struct janas_api_monitor *mo = calloc(1, sizeof *mo);
        if (mo && pthread_mutex_init(&mo->mu, NULL) == 0)
            llm->mon = mo;
        else
            free(mo);
    }
    pthread_mutex_unlock(&make_mu);
    return llm->mon;
}

void janas_api_monitor_note(janas_llm_chat *c)
{
    int stage = c->stats.stage;
    if (stage == JANAS_LLM_STAGE_IDLE)
        return;
    struct janas_api_monitor *mo = monitor_of(c->llm);
    if (!mo)
        return;
    pthread_mutex_lock(&mo->mu);
    if (mo->n == 0 || mo->chat != c || mo->t_send != c->t_send) {
        if (mo->n) { /* the one before is over */
            const struct janas_llm_chat_stats *p = &mo->ring[mo->head];
            mo->replies++;
            mo->input_tokens += p->input_tokens;
            mo->output_tokens += p->output_tokens;
            mo->head = (mo->head + 1) % MON_REPLIES;
        }
        if (mo->n < MON_REPLIES)
            mo->n++;
        mo->chat = c;
        mo->t_send = c->t_send;
    }
    mo->ring[mo->head] = c->stats;
    janas_llm_model_tier_counts(c->llm->m, mo->tier);
    janas_llm_model_phase_times(c->llm->m, mo->phase);
    double t = janas_api_now();
    if (t - mo->t_choices >= 1.0) {
        int32_t len;
        if (janas_llm_tuning(c->llm, mo->choices, sizeof mo->choices, &len) !=
            JANAS_LLM_OK)
            mo->choices[0] = 0;
        mo->t_choices = t;
    }
    /* between two passes: the observer may come or go */
    int want = mo->layers_want > 0;
    if (want != mo->layers_on && janas_llm_model_watch(c->llm->m, want) == 0)
        mo->layers_on = want;
    pthread_mutex_unlock(&mo->mu);
}

void janas_api_monitor_free(janas_llm *llm)
{
    struct janas_api_monitor *mo = llm->mon;
    if (!mo)
        return;
    pthread_mutex_destroy(&mo->mu);
    free(mo);
    llm->mon = NULL;
}

int32_t janas_llm_monitor(janas_llm *llm, struct janas_llm_monitor *m)
{
    if (!llm || !m || m->size < sizeof(uint32_t))
        return janas_api_fail(JANAS_LLM_EINVAL, "no model or no size");
    struct janas_llm_monitor o;
    memset(&o, 0, sizeof o);
    o.size = sizeof o;
    struct janas_api_monitor *mo = monitor_of(llm);
    if (mo) {
        pthread_mutex_lock(&mo->mu);
        const struct janas_llm_chat_stats *s = &mo->ring[mo->head];
        o.replies = mo->replies + (mo->n ? 1 : 0);
        o.input_tokens = mo->input_tokens + (mo->n ? s->input_tokens : 0);
        o.output_tokens = mo->output_tokens + (mo->n ? s->output_tokens : 0);
        o.stage = mo->n ? s->stage : JANAS_LLM_STAGE_IDLE;
        o.tier_hits = mo->tier[0];
        o.tier_loads = mo->tier[1];
        o.tier_cpu = mo->tier[2];
        o.tier_slots = mo->tier[3];
        o.tier_slot_bytes = mo->tier[4];
        for (int i = 0; i < 8 && i < JANAS_PH_COUNT; i++)
            o.phase_seconds[i] = mo->phase[i];
        memcpy(o.choices, mo->choices, sizeof o.choices);
        o.choices[sizeof o.choices - 1] = 0;
        o.layers_on = mo->layers_on;
        pthread_mutex_unlock(&mo->mu);
    }
    struct janas_expert_cache *ec = janas_llm_model_cache(llm->m);
    if (ec) {
        struct janas_expert_cache_stats cs;
        janas_expert_cache_stats(ec, &cs);
        o.experts_used = cs.requests;
        o.experts_hit = cs.hits;
        o.experts_read = cs.misses + cs.prefetch_used;
        o.bytes_read = cs.bytes_read + cs.prefetch_bytes;
        o.warm_bytes = cs.warm_bytes;
        o.io_wait_seconds = cs.wait_seconds;
        o.experts_missed = cs.misses;
        o.prefetch_reads = cs.prefetch_reads;
        o.prefetch_used = cs.prefetch_used;
        o.prefetch_bytes = cs.prefetch_bytes;
        o.io_seconds = cs.io_seconds;
        o.cache_slots = janas_expert_cache_slots(ec);
        o.slot_bytes = janas_expert_cache_slot_bytes(ec);
    }
    o.context_size = llm->n_ctx;
    o.rss_bytes = janas_rss_bytes();
    o.memory_bytes = janas_mem_total();
    o.memory_available = janas_mem_available();
    /* as much as the caller's struct holds: an older one gets the fields
       it knows */
    size_t n = m->size < sizeof o ? m->size : sizeof o;
    uint32_t size = m->size;
    memcpy(m, &o, n);
    m->size = size;
    return JANAS_LLM_OK;
}

int32_t janas_llm_monitor_reply(janas_llm *llm, int32_t k,
                                struct janas_llm_chat_stats *s)
{
    if (!llm || !s || k < 0 || s->size < sizeof(uint32_t))
        return janas_api_fail(JANAS_LLM_EINVAL, "no model, no size or k < 0");
    struct janas_api_monitor *mo = monitor_of(llm);
    if (!mo)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    pthread_mutex_lock(&mo->mu);
    if ((uint32_t)k >= mo->n) {
        pthread_mutex_unlock(&mo->mu);
        return JANAS_LLM_DONE;
    }
    struct janas_llm_chat_stats o =
        mo->ring[(mo->head + MON_REPLIES - (uint32_t)k) % MON_REPLIES];
    pthread_mutex_unlock(&mo->mu);
    size_t n = s->size < sizeof o ? s->size : sizeof o;
    uint32_t size = s->size;
    memcpy(s, &o, n);
    s->size = size;
    return JANAS_LLM_OK;
}

int32_t janas_llm_monitor_layers_on(janas_llm *llm, int32_t on)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    struct janas_api_monitor *mo = monitor_of(llm);
    if (!mo)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    pthread_mutex_lock(&mo->mu);
    mo->layers_want += on ? 1 : -1;
    if (mo->layers_want < 0)
        mo->layers_want = 0;
    pthread_mutex_unlock(&mo->mu);
    return JANAS_LLM_OK;
}

int32_t janas_llm_monitor_layers_shape(janas_llm *llm, uint32_t shape[3])
{
    if (!llm || !shape)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model or no shape");
    janas_llm_model_watch_shape(llm->m, shape);
    return JANAS_LLM_OK;
}

int32_t janas_llm_monitor_layers(janas_llm *llm, float *ffn, float *resid,
                                 uint32_t *chosen, uint32_t *read,
                                 float *strength, uint64_t *passes)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    struct janas_api_monitor *mo = monitor_of(llm);
    if (!mo)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    /* under the monitor's lock: the replying thread detaches the observer
       under it too */
    pthread_mutex_lock(&mo->mu);
    int rc = mo->layers_on
                 ? janas_llm_model_watch_take(llm->m, ffn, resid, chosen, read,
                                              strength, passes)
                 : -1;
    pthread_mutex_unlock(&mo->mu);
    return rc == 0 ? JANAS_LLM_OK : JANAS_LLM_DONE;
}
