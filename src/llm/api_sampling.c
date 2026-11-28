/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_sampling.c - the public API: how a conversation samples. A
 * parameter left negative is the model's own, what its makers recommend
 * as its file says (general.sampling.*, which llama.cpp's converter writes
 * from the model's generation_config.json: Qwen3.8-Flash-Next temperature
 * 1, top_k 20, top_p 0.95; Gemma 4 1, 64, 0.95), and Janas's where the
 * file says nothing. Then llama.cpp's chain (sample.h) from the parameters
 * that have its names, the order of its stages, and DRY's breakers made
 * into a flag per token of the vocabulary.
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"
#include "llm/jns.h"
#include "llm/json.h"

/* the negative values: "the model's" */
static void unset(struct janas_llm_chat_params *p)
{
    p->temperature = p->top_p = p->min_p = -1;
    p->top_k = -1;
    p->repeat_penalty = -1;
    p->repeat_last_n = -1;
    p->xtc_probability = p->xtc_threshold = -1;
    p->mirostat = -1;
    p->mirostat_tau = p->mirostat_eta = -1;
    p->samplers[0] = 0;
}

void janas_api_read_rec(janas_llm *llm, const struct janas_jns *j)
{
    struct janas_llm_chat_params *r = &llm->rec;
    memset(r, 0, sizeof(*r));
    unset(r);
    r->size = sizeof(*r); /* read: a model opened otherwise has 0 */
    static const struct {
        const char *key;
        size_t at;
    } F[] = {
        {"general.sampling.temp",
         offsetof(struct janas_llm_chat_params, temperature)},
        {"general.sampling.top_p",
         offsetof(struct janas_llm_chat_params, top_p)},
        {"general.sampling.min_p",
         offsetof(struct janas_llm_chat_params, min_p)},
        {"general.sampling.penalty_repeat",
         offsetof(struct janas_llm_chat_params, repeat_penalty)},
        {"general.sampling.xtc_probability",
         offsetof(struct janas_llm_chat_params, xtc_probability)},
        {"general.sampling.xtc_threshold",
         offsetof(struct janas_llm_chat_params, xtc_threshold)},
        {"general.sampling.mirostat_tau",
         offsetof(struct janas_llm_chat_params, mirostat_tau)},
        {"general.sampling.mirostat_eta",
         offsetof(struct janas_llm_chat_params, mirostat_eta)},
    };
    for (size_t i = 0; i < sizeof F / sizeof *F; i++) {
        float v;
        if (janas_jns_meta_f32(j, F[i].key, &v) == 0 && v >= 0 && v < 1e6f)
            memcpy((char *)r + F[i].at, &v, sizeof v);
    }
    uint32_t u;
    if (janas_jns_meta_u32(j, "general.sampling.top_k", &u) == 0 &&
        u < 1u << 30)
        r->top_k = (int32_t)u;
    if (janas_jns_meta_u32(j, "general.sampling.mirostat", &u) == 0 && u <= 2)
        r->mirostat = (int32_t)u;
    /* llama.cpp's window: -1 the whole context (here: the whole reply), 0
       none (the penalties off) */
    if (janas_jns_meta_u32(j, "general.sampling.penalty_last_n", &u) == 0) {
        int32_t n = u >= 1u << 30 ? -1 : (int32_t)u; /* -1 as written */
        r->repeat_last_n = n > 0 ? n : 0;
        if (n == 0)
            r->repeat_penalty = 1;
    }
    char seq[sizeof r->samplers];
    uint8_t st[JANAS_SAMPLE_MAX_STAGES];
    if (janas_jns_meta_str(j, "general.sampling.sequence", seq, sizeof seq) ==
            0 &&
        janas_sample_stages(seq, st, JANAS_SAMPLE_MAX_STAGES) > 0)
        memcpy(r->samplers, seq, sizeof seq);
}

/* the stages' names, from their numbers */
static void stage_names(const uint8_t *st, int n, char *out, size_t len)
{
    static const char *const NAME[] = {
        "",      "penalties", "dry",   "top_n_sigma", "top_k",
        "typ_p", "top_p",     "min_p", "xtc",         "temperature"};
    size_t at = 0;
    out[0] = 0;
    for (int i = 0; i < n && st[i] < sizeof NAME / sizeof *NAME; i++) {
        int w = snprintf(out + at, len - at, "%s%s", i ? ";" : "", NAME[st[i]]);
        if (w < 0 || (size_t)w >= len - at)
            break;
        at += (size_t)w;
    }
}

static int32_t resolve(const janas_llm *llm,
                       const struct janas_llm_chat_params *p,
                       struct janas_llm_chat_params *out)
{
    struct janas_llm_chat_params none;
    memset(&none, 0, sizeof none);
    unset(&none);
    /* the model's, once read from its file */
    const struct janas_llm_chat_params *r = llm->rec.size ? &llm->rec : &none;
    *out = *p;
    /* the model's where the chat says nothing, then Janas's */
#define TAKE(f, def) out->f = out->f >= 0 ? out->f : r->f >= 0 ? r->f : (def)
    TAKE(temperature, 0.7f);
    TAKE(top_k, 20);
    TAKE(top_p, 0.8f);
    TAKE(min_p, 0);
    TAKE(repeat_penalty, 1);
    TAKE(repeat_last_n, 0);
    TAKE(xtc_probability, 0);
    TAKE(xtc_threshold, 0.1f);
    TAKE(mirostat, 0);
    TAKE(mirostat_tau, 5);
    TAKE(mirostat_eta, 0.1f);
#undef TAKE
    if (!out->samplers[0] && r->samplers[0])
        memcpy(out->samplers, r->samplers, sizeof out->samplers);
    out->samplers[sizeof out->samplers - 1] = 0;
    out->dry_breakers[sizeof out->dry_breakers - 1] = 0;
    uint8_t st[JANAS_SAMPLE_MAX_STAGES];
    int ns = janas_sample_stages(out->samplers, st, JANAS_SAMPLE_MAX_STAGES);
    if (!(out->temperature >= 0 && out->temperature <= 100) ||
        !(out->top_p >= 0 && out->top_p <= 1) ||
        !(out->min_p >= 0 && out->min_p <= 1) ||
        !(out->repeat_penalty >= 0 && out->repeat_penalty <= 100) ||
        out->repeat_last_n < 0 ||
        !(out->typical_p >= 0 && out->typical_p <= 1) ||
        !(out->top_n_sigma >= 0 && out->top_n_sigma <= 100) ||
        !(out->xtc_probability >= 0 && out->xtc_probability <= 1) ||
        !(out->xtc_threshold >= 0 && out->xtc_threshold <= 1) ||
        !(out->dry_multiplier >= 0 && out->dry_multiplier <= 100) ||
        !(out->dry_multiplier == 0 ||
          (out->dry_base >= 1 && out->dry_base <= 100)) ||
        out->dry_allowed_length < 0 || out->dry_last_n < 0 ||
        out->mirostat > 2 ||
        !(out->mirostat_tau >= 0 && out->mirostat_tau <= 100) ||
        !(out->mirostat_eta >= 0 && out->mirostat_eta <= 1) ||
        !(out->dynatemp_range >= 0 && out->dynatemp_range <= 100) ||
        !(out->dynatemp_exponent >= 0 && out->dynatemp_exponent <= 100) ||
        ns < 0)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "invalid sampling parameters (samplers: "
                              "llama.cpp's names separated by ';')");
    /* the order said in llama.cpp's names, Janas's when none was asked */
    static const uint8_t OWN[] = {
        JANAS_STAGE_TOP_N_SIGMA, JANAS_STAGE_TOP_K, JANAS_STAGE_TYPICAL,
        JANAS_STAGE_TEMPERATURE, JANAS_STAGE_TOP_P, JANAS_STAGE_MIN_P,
        JANAS_STAGE_XTC};
    if (ns > 0)
        stage_names(st, ns, out->samplers, sizeof out->samplers);
    else
        stage_names(OWN, (int)sizeof OWN, out->samplers, sizeof out->samplers);
    return JANAS_LLM_OK;
}

int32_t janas_api_sampling(const janas_llm_chat *c,
                           struct janas_llm_chat_params *out)
{
    return resolve(c->llm, &c->p, out);
}

int32_t janas_llm_chat_sampling(const janas_llm_chat *c,
                                struct janas_llm_chat_params *out)
{
    if (!c || !out)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    return janas_api_sampling(c, out);
}

int32_t janas_llm_sampling(const janas_llm *llm,
                           const struct janas_llm_chat_params *p,
                           struct janas_llm_chat_params *out)
{
    if (!llm || !out)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    struct janas_llm_chat_params d;
    janas_llm_chat_params_default(&d);
    if (p) {
        if (p->size < sizeof(uint32_t) || p->size > sizeof d)
            return janas_api_fail(JANAS_LLM_EINVAL, "parameters: unknown size");
        memcpy(&d, p, p->size);
        d.size = sizeof d;
    }
    return resolve(llm, &d, out);
}

/* For each token, whether its text holds one of the breakers (a JSON
   array of strings; "" llama.cpp's). */
static int make_breakers(janas_llm_chat *c, const char *list)
{
    static const char *const LLAMA[] = {"\n", ":", "\"", "*"};
    const char *b[64];
    size_t bn[64], n = 0;
    struct janas_json_doc *d = NULL;
    if (list[0]) {
        char err[128];
        d = janas_json_parse(list, strlen(list), err, sizeof err);
        const struct janas_json *a = d ? janas_json_root(d) : NULL;
        if (!a || a->type != JANAS_JSON_ARRAY) {
            janas_json_free(d);
            return janas_api_fail(JANAS_LLM_EINVAL,
                                  "dry_breakers: a JSON array of strings");
        }
        for (const struct janas_json *e = a->child; e; e = e->next) {
            if (e->type != JANAS_JSON_STRING || n == 64) {
                janas_json_free(d);
                return janas_api_fail(JANAS_LLM_EINVAL,
                                      "dry_breakers: up to 64 strings");
            }
            if (e->n)
                b[n] = e->s, bn[n++] = e->n;
        }
    } else
        for (; n < sizeof LLAMA / sizeof *LLAMA; n++)
            b[n] = LLAMA[n], bn[n] = strlen(LLAMA[n]);
    uint32_t nv = janas_llm_model_n_vocab(c->llm->m),
             nt = janas_tokenizer_n_vocab(c->llm->tok);
    uint8_t *f = calloc(nv, 1);
    if (!f) {
        janas_json_free(d);
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    }
    char text[256];
    for (uint32_t t = 0; t < nv && t < nt; t++) {
        size_t l = janas_tokenizer_decode(c->llm->tok, (int32_t)t, text,
                                          sizeof text - 1);
        if (l >= sizeof text)
            l = sizeof text - 1;
        for (size_t i = 0; i < n && !f[t]; i++)
            for (size_t a = 0; a + bn[i] <= l && !f[t]; a++)
                f[t] = !memcmp(text + a, b[i], bn[i]);
    }
    janas_json_free(d);
    free(c->breakers);
    c->breakers = f;
    memcpy(c->breakers_of, list, sizeof c->breakers_of);
    return JANAS_LLM_OK;
}

int32_t janas_api_sampling_options(janas_llm_chat *c,
                                   struct janas_gen_options *o)
{
    struct janas_llm_chat_params p;
    int32_t rc = janas_api_sampling(c, &p);
    if (rc != JANAS_LLM_OK)
        return rc;
    o->temperature = p.temperature;
    o->top_k = p.top_k;
    o->top_p = p.top_p;
    o->min_p = p.min_p;
    o->presence_penalty = p.presence_penalty;
    o->frequency_penalty = p.frequency_penalty;
    struct janas_sample_more *m = &o->more;
    memset(m, 0, sizeof(*m));
    m->repeat = p.repeat_penalty;
    m->last_n = p.repeat_last_n;
    m->typical_p = p.typical_p;
    m->top_n_sigma = p.top_n_sigma;
    m->xtc_probability = p.xtc_probability;
    m->xtc_threshold = p.xtc_threshold;
    m->dry_multiplier = p.dry_multiplier;
    m->dry_base = p.dry_base;
    m->dry_allowed = p.dry_allowed_length;
    m->dry_last_n = p.dry_last_n;
    m->mirostat = p.mirostat;
    m->mirostat_tau = p.mirostat_tau;
    m->mirostat_eta = p.mirostat_eta;
    m->dynatemp_range = p.dynatemp_range;
    m->dynatemp_exponent = p.dynatemp_exponent;
    /* an order asked (or the model's) as stages; Janas's own when none,
       which the plain draw is unless something else is asked */
    char asked[sizeof p.samplers];
    memcpy(asked,
           c->p.samplers[0] || !c->llm->rec.size ? c->p.samplers
                                                 : c->llm->rec.samplers,
           sizeof asked);
    asked[sizeof asked - 1] = 0;
    m->n_stage =
        asked[0] ? janas_sample_stages(asked, m->stage, JANAS_SAMPLE_MAX_STAGES)
                 : 0;
    if (m->n_stage < 0)
        m->n_stage = 0;
    o->breakers = NULL;
    if (p.dry_multiplier > 0) {
        if (!c->breakers || strcmp(c->breakers_of, p.dry_breakers)) {
            rc = make_breakers(c, p.dry_breakers);
            if (rc != JANAS_LLM_OK)
                return rc;
        }
        o->breakers = c->breakers;
    }
    return JANAS_LLM_OK;
}

void janas_api_sampling_free(janas_llm_chat *c)
{
    free(c->breakers);
    c->breakers = NULL;
}
