/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api.c - the public API (include/janas/llm.h): the model and its tokens;
 * the chat is in api_chat.c and api_reply.c.
 *
 * Chat formats are recognized from the model's own chat template (metadata
 * tokenizer.chat_template) and its special tokens; each format says how a
 * message is framed. A message is turned into tokens the way the rendered
 * conversation would be: the format's special tokens as ids, the text
 * between them tokenized as one run (so the user's text never produces
 * special tokens).
 */
#include "janas/llm.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "common/sysinfo.h"
#include "llm/arch.h"
#include "llm/attention.h"
#include "llm/expert_cache.h"
#include "llm/generate.h"
#include "llm/jns.h"
#include "llm/model.h"
#include "llm/tokenizer.h"
#include "llm/chat.h"
#include "llm/metrics.h"

static _Thread_local char last_error[256];

int32_t janas_api_fail(int32_t code, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(last_error, sizeof(last_error), fmt, ap);
    va_end(ap);
    return code;
}

double janas_api_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int32_t janas_llm_abi_version(void)
{
    return JANAS_LLM_ABI_VERSION;
}

const char *janas_llm_last_error(void)
{
    return last_error;
}

/* For tools built with the library (janas-bench): the objects behind a
   handle. Not part of the public API. */
struct janas_llm_model *janas_llm_internal_model(janas_llm *llm)
{
    return llm->m;
}

struct janas_tokenizer *janas_llm_internal_tokenizer(janas_llm *llm)
{
    return llm->tok;
}

/* The power mode in effect: auto follows the power source. */
void janas_api_apply_mode(janas_llm *llm)
{
    int eco = llm->mode == JANAS_LLM_MODE_ECO ||
              (llm->mode == JANAS_LLM_MODE_AUTO && janas_on_mains() == 0);
    janas_llm_model_set_eco(llm->m, eco || !llm->gpu_wanted);
}

void janas_llm_params_default(struct janas_llm_params *p)
{
    memset(p, 0, sizeof(*p));
    p->size = sizeof(*p);
    p->n_ctx = 16384;
}

static uint64_t file_size(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 ? (uint64_t)st.st_size : 0;
}

/* Recognizes the chat format; returns 0 or an error code. */
static int32_t find_format(janas_llm *llm, const struct janas_jns *j)
{
    char tmpl[16384];
    if (janas_jns_meta_str(j, "tokenizer.chat_template", tmpl, sizeof(tmpl)) !=
        0)
        tmpl[0] = 0;
    llm->im_start = janas_tokenizer_find(llm->tok, "<|im_start|>");
    llm->im_end = janas_tokenizer_find(llm->tok, "<|im_end|>");
    if (llm->im_start >= 0 && llm->im_end >= 0 &&
        (tmpl[0] == 0 || strstr(tmpl, "<|im_start|>"))) {
        llm->format = FORMAT_CHATML;
        llm->stop[llm->n_stop++] = llm->im_end;
        llm->think_open = janas_tokenizer_find(llm->tok, "<think>");
        llm->think_close = janas_tokenizer_find(llm->tok, "</think>");
        llm->thinker = strstr(tmpl, "enable_thinking") &&
                       llm->think_open >= 0 && llm->think_close >= 0;
        /* tools: the template says how calls are written, the vocabulary
           must have the markers as tokens of their own */
        llm->call_open = janas_tokenizer_find(llm->tok, "<tool_call>");
        llm->call_close = janas_tokenizer_find(llm->tok, "</tool_call>");
        llm->resp_open = janas_tokenizer_find(llm->tok, "<tool_response>");
        llm->resp_close = janas_tokenizer_find(llm->tok, "</tool_response>");
        llm->tools = janas_tools_dialect(tmpl);
        if (llm->call_open < 0 || llm->call_close < 0 || llm->resp_open < 0 ||
            llm->resp_close < 0)
            llm->tools = JANAS_TOOLS_NONE;
    } else {
        return janas_api_fail(JANAS_LLM_EMODEL, "chat format not supported");
    }
    /* an embedding model: how its states make one vector */
    char arch[32] = "", key[96];
    uint32_t v;
    janas_jns_meta_str(j, "general.architecture", arch, sizeof(arch));
    snprintf(key, sizeof(key), "%.32s.pooling_type", arch);
    if (janas_jns_meta_u32(j, key, &v) == 0)
        llm->pooling = v;
    if (janas_jns_meta_u32(j, "tokenizer.ggml.add_eos_token", &v) == 0)
        llm->add_eos = v != 0;
    llm->eos_id = -1;
    uint32_t eos;
    if (janas_jns_meta_u32(j, "tokenizer.ggml.eos_token_id", &eos) == 0)
        llm->eos_id = (int32_t)eos;
    if (janas_jns_meta_u32(j, "tokenizer.ggml.eos_token_id", &eos) == 0 &&
        (int32_t)eos != llm->stop[0])
        llm->stop[llm->n_stop++] = (int32_t)eos;
    int32_t eot = janas_tokenizer_find(llm->tok, "<|endoftext|>");
    if (eot >= 0 && eot != llm->stop[0] &&
        (llm->n_stop < 2 || eot != llm->stop[1]))
        llm->stop[llm->n_stop++] = eot;
    return 0;
}

int32_t janas_llm_open(const char *path, const struct janas_llm_params *pp,
                       janas_llm **out)
{
    struct janas_llm_params p;
    janas_llm_params_default(&p);
    if (!path || !out)
        return janas_api_fail(JANAS_LLM_EINVAL, "no path or no output handle");
    double t_open = janas_api_now();
    *out = NULL;
    if (pp) {
        if (pp->size < sizeof(uint32_t) || pp->size > sizeof(p))
            return janas_api_fail(JANAS_LLM_EINVAL, "parameters: unknown size");
        memcpy(&p, pp, pp->size);
        p.size = sizeof(p);
    }
    if (p.n_ctx == 0)
        p.n_ctx = 16384;
    janas_llm *llm = calloc(1, sizeof(*llm));
    if (!llm)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    llm->gpu_wanted = 1; /* wanted unless the caller says otherwise */

    /* the tokenizer and the chat format, from the file's metadata */
    struct janas_jns j;
    char err[256] = "";
    if (janas_jns_open(path, &j, err, sizeof(err)) != 0) {
        free(llm);
        return janas_api_fail(JANAS_LLM_EOPEN, "%s: %s", path, err);
    }
    char kind[64] = "", pre[64] = "", name[96] = "", arch[32] = "";
    janas_jns_meta_str(&j, "tokenizer.ggml.model", kind, sizeof(kind));
    janas_jns_meta_str(&j, "tokenizer.ggml.pre", pre, sizeof(pre));
    janas_jns_meta_str(&j, "general.name", name, sizeof(name));
    janas_jns_meta_str(&j, "general.architecture", arch, sizeof(arch));
    int32_t rc = 0;
    if (strcmp(kind, "gpt2") != 0 ||
        (strcmp(pre, "qwen2") != 0 && strcmp(pre, "qwen35") != 0))
        rc = janas_api_fail(JANAS_LLM_EMODEL, "tokenizer %s/%s not supported",
                            kind, pre);
    else if (!(llm->tok = janas_tokenizer_create(&j, err, sizeof(err))))
        rc = janas_api_fail(JANAS_LLM_EMODEL, "tokenizer: %s", err);
    else
        rc = find_format(llm, &j);
    uint64_t resident = j.h.resident_bytes;
    /* what the cache would need to hold every slot at once. Not the size of
       the expert region: the cache's slots are all as big as the largest, so
       where the layers differ (a down matrix in Q6_K in one and Q4_K in the
       next) the region is smaller than the room the same slots take in it.
       A dense model is the case that cannot survive the difference - its one
       expert a layer is wanted at every token, in order, so a cache one slot
       short of the layers is a cache that never hits. */
    uint64_t slot = 0;
    for (uint32_t l = 0; l < j.h.n_layer; l++)
        if (j.layers[l].slot_bytes > slot)
            slot = j.layers[l].slot_bytes;
    uint64_t experts = (uint64_t)j.h.n_layer * j.h.n_expert * slot;
    if (experts < j.h.experts_bytes)
        experts = j.h.experts_bytes;
    /* what a token of context costs: keys and values of the layers that
       have them, one byte a number with a scale per position and KV head */
    uint32_t n_layer = j.h.n_layer, kv_heads = 0, kv_len = 0, train = 0;
    char key[96];
    snprintf(key, sizeof(key), "%.32s.attention.head_count_kv", arch);
    janas_jns_meta_u32(&j, key, &kv_heads);
    snprintf(key, sizeof(key), "%.32s.attention.key_length", arch);
    janas_jns_meta_u32(&j, key, &kv_len);
    snprintf(key, sizeof(key), "%.32s.context_length", arch);
    janas_jns_meta_u32(&j, key, &train);
    uint64_t kv_token = (uint64_t)n_layer * kv_heads * (kv_len * 2 + 8);
    /* hybrids that give one count for all layers (Qwen3-Next, Qwen3.5 and
       3.6): only every full_attention_interval-th layer has keys and values
       (see model.c), plus the one of a prediction block. Counted on all of
       them, 262144 tokens of Qwen3-Next-80B took 13.1 GB from the expert
       cache for 3.3 GB of cache, and the engine fell back to four bits and
       eight experts to fit (23 Sep 2026) */
    const struct janas_arch *a = janas_arch_find(arch, sizeof(arch));
    if (a && a->rec != JANAS_REC_NONE) {
        uint32_t interval = 4;
        snprintf(key, sizeof(key), "%.32s.full_attention_interval", arch);
        janas_jns_meta_u32(&j, key, &interval);
        if (interval > 0)
            kv_token = (uint64_t)(n_layer / interval + 1) * kv_heads *
                       (kv_len * 2 + 8);
    }
    /* hybrids say it layer by layer, with nothing where the layer is
       recurrent: those keep a state of fixed size instead of a cache */
    uint64_t n_arr = 0, n_pat = 0;
    const int32_t *per_layer = NULL;
    /* Gemma 4: sliding-window layers with a head size of their own, and an
       attention (its partial results) for each kind of layer */
    const uint8_t *pat = NULL;
    uint32_t len_swa = 0, kv_swa = 0;
    snprintf(key, sizeof(key), "%.32s.attention.sliding_window_pattern", arch);
    if (janas_jns_meta_bytes(&j, key, &n_pat, &pat) != 0 || n_pat != n_layer)
        pat = NULL;
    snprintf(key, sizeof(key), "%.32s.attention.key_length_swa", arch);
    janas_jns_meta_u32(&j, key, &len_swa);
    snprintf(key, sizeof(key), "%.32s.attention.head_count_kv", arch);
    if (janas_jns_meta_ints(&j, key, &n_arr, &per_layer) == 0 && per_layer &&
        n_arr == n_layer) {
        kv_token = 0;
        for (uint64_t i = 0; i < n_arr; i++) {
            int32_t h;
            memcpy(&h, per_layer + i, sizeof(h));
            uint32_t len = pat && pat[i] && len_swa ? len_swa : kv_len;
            if (h > 0)
                kv_token += (uint64_t)h * (len * 2 + 8);
            if (h > 0 && pat && pat[i])
                kv_swa = (uint32_t)h;
            else if (h > 0 && pat)
                kv_heads = (uint32_t)h;
        }
    }
    if (!kv_token || kv_token > (1 << 20))
        kv_token = 64 << 10; /* unsaid or odd: the old bound */
    /* and what attention keeps beside the cache, which grows with the
       context as well: uncounted, 262,144 tokens of Qwen3-Next-80B put
       1.08 GB of partial results in the margin left to other programs */
    uint32_t n_head = 0;
    snprintf(key, sizeof(key), "%.32s.attention.head_count", arch);
    janas_jns_meta_u32(&j, key, &n_head);
    uint64_t attn_part =
        n_head && kv_heads && kv_len
            ? janas_attn_part_bytes(n_head, kv_heads, kv_len, p.n_ctx)
            : 0;
    if (pat && n_head && kv_swa && len_swa)
        attn_part += janas_attn_part_bytes(n_head, kv_swa, len_swa, p.n_ctx);
    /* and the scratch of a pass, which grows with the prefill block
       (janas_llm_model_max_block): per token, the rows of its experts (gate,
       up, their product, down) and a few rows of the model's width. An
       estimate from the shapes, on the generous side */
    uint32_t embd = 0, ff = 0, k_used = 1;
    snprintf(key, sizeof(key), "%.32s.embedding_length", arch);
    janas_jns_meta_u32(&j, key, &embd);
    snprintf(key, sizeof(key), "%.32s.expert_feed_forward_length", arch);
    if (janas_jns_meta_u32(&j, key, &ff) != 0) {
        snprintf(key, sizeof(key), "%.32s.feed_forward_length", arch);
        janas_jns_meta_u32(&j, key, &ff);
    }
    snprintf(key, sizeof(key), "%.32s.expert_used_count", arch);
    janas_jns_meta_u32(&j, key, &k_used);
    const char *pb = getenv("JANAS_PREFILL_BLOCK");
    uint64_t blk = pb && atol(pb) > 256 ? (uint64_t)atol(pb) : 256;
    uint64_t scratch =
        blk * (((uint64_t)k_used + 1) * (3ull * ff + embd) + 32ull * embd) *
        sizeof(float);
    janas_jns_close(&j);
    if (rc != 0) {
        janas_tokenizer_destroy(llm->tok);
        free(llm);
        return rc;
    }

    /* the expert cache takes the free memory, less what the rest needs and
       a fifth of the machine's memory left free: the computer stays usable
       for other programs (a tighter margin pushed the desktop into swap) */
    /* what is left to other programs: the caller's say, or a fifth of the
       memory and at least 2 GiB */
    uint64_t reserve = p.reserve_bytes;
    if (reserve == 0) {
        reserve = janas_mem_total() / 5;
        if (reserve < ((uint64_t)2 << 30))
            reserve = (uint64_t)2 << 30;
    }
    uint64_t cache = p.cache_bytes;
    if (cache == 0) {
        uint64_t avail = janas_mem_available(), need = resident;
        need += reserve;
        need += (uint64_t)p.n_ctx * kv_token + attn_part + scratch;
        if (p.mtp_path)
            need += file_size(p.mtp_path) + ((uint64_t)256 << 20);
        cache = avail > need + ((uint64_t)1 << 30) ? avail - need
                                                   : (uint64_t)1 << 30;
        if (cache > experts)
            cache = experts;
    }
    /* compute: every usable core (the tuner picks how many per pass);
       I/O: the efficiency cores, or unpinned */
    int ccpus[256], icpus[256], nc = 0, ni = 0;
    int split = janas_cpu_split(ccpus, &nc, icpus, &ni, 256);
    struct janas_cpu_layout lay;
    int have_lay = janas_cpu_layout(NULL, 0, &lay) == 0;
    if (p.mode < JANAS_LLM_MODE_AUTO || p.mode > JANAS_LLM_MODE_MAX)
        p.mode = JANAS_LLM_MODE_AUTO;
    struct janas_llm_options mo = {
        .cache_bytes = cache,
        .n_ctx = p.n_ctx,
        .compute_cpus = have_lay ? lay.cpus : (split >= 0 ? ccpus : NULL),
        .n_compute = have_lay ? lay.n : (split >= 0 ? nc : 4),
        .io_cpus = split > 0 ? icpus : NULL,
        .n_io = split >= 0 ? ni : 4,
        .mtp_path = p.mtp_path,
        /* the GPU is opened unless eco: the tuner decides whether it helps,
           and only what it decides is worth telling the user about */
        .use_gpu = p.mode != JANAS_LLM_MODE_ECO,
        .warm = !p.no_preload,
        .expert_bits = p.expert_bits,
        .attn_scores = p.attn_scores};
    llm->m = janas_llm_model_load(path, &mo, err, sizeof(err));
    if (!llm->m) {
        janas_tokenizer_destroy(llm->tok);
        free(llm);
        return janas_api_fail(JANAS_LLM_EOPEN, "%s: %s", path, err);
    }
    llm->has_mtp = janas_llm_model_has_mtp(llm->m);
    if (p.draft_path) {
        /* the drafting model borrows the target's threads: two pools of
           twenty on twenty cores take the cores from each other */
        struct janas_llm_options dmo = mo;
        dmo.mtp_path = NULL;
        dmo.cache_bytes = 2ull << 30;
        dmo.warm = 0;
        dmo.use_gpu = 0;
        dmo.shared_compute = janas_llm_model_compute(llm->m);
        llm->draft = janas_llm_model_load(p.draft_path, &dmo, err, sizeof(err));
        if (!llm->draft) {
            janas_llm_model_free(llm->m);
            janas_tokenizer_destroy(llm->tok);
            free(llm);
            return janas_api_fail(JANAS_LLM_EOPEN, "%s: %s", p.draft_path, err);
        }
        if (janas_llm_model_n_vocab(llm->draft) !=
            janas_llm_model_n_vocab(llm->m)) {
            uint32_t a = janas_llm_model_n_vocab(llm->m);
            uint32_t b = janas_llm_model_n_vocab(llm->draft);
            janas_llm_model_free(llm->draft);
            janas_llm_model_free(llm->m);
            janas_tokenizer_destroy(llm->tok);
            free(llm);
            return janas_api_fail(
                JANAS_LLM_EOPEN,
                "%s: a drafting model must share the vocabulary "
                "(%u tokens against %u)",
                p.draft_path, b, a);
        }
    }
    llm->mode = p.mode;
    janas_api_apply_mode(llm);
    snprintf(llm->name, sizeof(llm->name), "%s", name);
    llm->n_ctx = janas_llm_model_n_ctx(llm->m); /* capped at what it knows */
    p.n_ctx = llm->n_ctx;
    char warm[64] = "";
    uint64_t warm_n = janas_llm_model_warm(llm->m, NULL);
    if (warm_n)
        snprintf(warm, sizeof(warm), ", %llu experts loading in the background",
                 (unsigned long long)warm_n);
    char bits[96] = "";
    int nb = janas_llm_model_expert_bits(llm->m);
    uint32_t used = janas_llm_model_experts(llm->m, NULL);
    int w = 0;
    if (nb != 6)
        w = snprintf(bits, sizeof(bits), ", down matrix at %d bits%s", nb,
                     janas_llm_model_expert_bits_auto(llm->m) ? " (memory)"
                                                              : "");
    if (janas_llm_model_experts_auto(llm->m) && w >= 0 &&
        (size_t)w < sizeof(bits))
        snprintf(bits + w, sizeof(bits) - (size_t)w,
                 ", %u experts per token (memory)", used);
    llm->cache_bytes = cache;
    llm->reserve_bytes = reserve;
    llm->kv_token = kv_token;
    llm->n_compute = mo.n_compute;
    snprintf(llm->arch, sizeof(llm->arch), "%s", arch);
    snprintf(llm->bits, sizeof(llm->bits), "%s", bits);
    snprintf(llm->warm, sizeof(llm->warm), "%s", warm);
    *out = llm;
    janas_metrics_open(llm, janas_api_now() - t_open);
    return JANAS_LLM_OK;
}

void janas_llm_close(janas_llm *llm)
{
    if (!llm)
        return;
    if (llm->chat)
        janas_llm_chat_destroy(llm->chat);
    /* the drafting model first: it borrowed the target's threads */
    janas_llm_model_free(llm->draft);
    janas_llm_model_free(llm->m);
    janas_tokenizer_destroy(llm->tok);
    free(llm);
}

int32_t janas_api_copy_out(const char *s, size_t n, char *buf, int32_t cap,
                           int32_t *len)
{
    if (len)
        *len = (int32_t)n;
    if (!buf || cap <= 0)
        return janas_api_fail(JANAS_LLM_ESMALL, "no buffer");
    size_t w = n < (size_t)cap - 1 ? n : (size_t)cap - 1;
    memcpy(buf, s, w);
    buf[w] = 0;
    return w == n ? JANAS_LLM_OK
                  : janas_api_fail(JANAS_LLM_ESMALL, "buffer too small");
}

int32_t janas_llm_set_mode(janas_llm *llm, int32_t mode)
{
    if (!llm || mode < JANAS_LLM_MODE_AUTO || mode > JANAS_LLM_MODE_MAX)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid mode");
    llm->mode = mode;
    janas_api_apply_mode(llm);
    return JANAS_LLM_OK;
}

int32_t janas_llm_experts(const janas_llm *llm, int32_t *used, int32_t *most)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    uint32_t mx;
    uint32_t u = janas_llm_model_experts(llm->m, &mx);
    if (used)
        *used = (int32_t)u;
    if (most)
        *most = (int32_t)mx;
    return JANAS_LLM_OK;
}

int32_t janas_llm_attn_scores(const janas_llm *llm)
{
    if (!llm || !llm->m)
        return JANAS_LLM_ATTN_EXACT;
    return janas_llm_model_attn_scores(llm->m);
}

int32_t janas_llm_expert_bits(const janas_llm *llm)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    return janas_llm_model_expert_bits(llm->m);
}

int32_t janas_llm_preload(const janas_llm *llm, int64_t *done, int64_t *total)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    if (done)
        *done = (int64_t)janas_llm_model_warm_done(llm->m);
    if (total)
        *total = (int64_t)janas_llm_model_warm(llm->m, NULL);
    return JANAS_LLM_OK;
}

int32_t janas_llm_set_experts(janas_llm *llm, int32_t n)
{
    if (!llm || n < 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    if (!janas_llm_model_set_experts(llm->m, (uint32_t)n))
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "too many experts for this model");
    return JANAS_LLM_OK;
}

int32_t janas_llm_tuning(const janas_llm *llm, char *buf, int32_t cap,
                         int32_t *len)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    char tmp[512];
    static const char *names[] = {"auto", "eco", "max"};
    int n = snprintf(tmp, sizeof(tmp), "mode %s", names[llm->mode]);
    if (llm->mode == JANAS_LLM_MODE_AUTO)
        n += snprintf(tmp + n, sizeof(tmp) - (size_t)n, " (%s)",
                      janas_on_mains() == 0 ? "battery: eco" : "mains: max");
    n += snprintf(tmp + n, sizeof(tmp) - (size_t)n, "; ");
    janas_llm_model_tuning(llm->m, tmp + n, sizeof(tmp) - (size_t)n);
    return janas_api_copy_out(tmp, strlen(tmp), buf, cap, len);
}

int32_t janas_llm_name(const janas_llm *llm, char *buf, int32_t cap,
                       int32_t *len)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    return janas_api_copy_out(llm->name, strlen(llm->name), buf, cap, len);
}

int32_t janas_llm_default_system(const janas_llm *llm, char *buf, int32_t cap,
                                 int32_t *len)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    /*
     * The name is the assistant's, the model is what it thinks with: asked
     * who it is, it should answer both and not take the model's name for
     * its own. Left to itself a model says whatever its training says, and
     * they very often claim to be somebody else's assistant. But a system
     * message about nothing but who it is makes a small model introduce
     * itself at every greeting: measured on Qwen3-4B (23 Sep 2026), "Ciao!"
     * got the whole presentation, and with this sentence "Ciao! 😊".
     */
    const char *which = llm->name[0] ? llm->name : "one this machine holds";
    char text[1024];
    int n = snprintf(
        text, sizeof(text),
        "Your name is Janas. You are an assistant running on the user's own "
        "computer, with no access to the internet. The model you think with "
        "is %s, and the engine running it is Janas-LLM, written in C by "
        "Maurizio \"camauri\" Cammalleri. When you are asked who or what you "
        "are, give both: the name is Janas, the model is %s; and when you "
        "name the engine, name its author with it. Do not introduce "
        "yourself unless the user asks who you are; to a greeting, just "
        "greet back, in the user's language. Answer in the language the "
        "user writes in.",
        which, which);
    return janas_api_copy_out(text, (size_t)n, buf, cap, len);
}

int32_t janas_llm_describe(const janas_llm *llm, char *buf, int32_t cap,
                           int32_t *len)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    /*
     * Built when it is asked for, not when the model was opened: having a
     * GPU and giving it work are two different things, the tuner decides
     * the second pass by pass, and the answer changes while the engine
     * runs. The same goes for the attention, which the context decides.
     */
    janas_llm *w = (janas_llm *)llm;
    char gpubuf[96];
    const char *gpu = ", GPU not present";
    if (janas_llm_model_gpu(llm->m)) {
        /* which passes it is given, in the words /mode uses for them */
        int use = janas_llm_model_gpu_used(llm->m);
        if (!use)
            gpu = ", GPU present (not used)";
        else {
            const char *where[3] = {"single tokens", "draft checks", "blocks"};
            char list[64] = "";
            for (int k = 0; k < 3; k++)
                if (use & (1 << k))
                    snprintf(list + strlen(list), sizeof(list) - strlen(list),
                             "%s%s", list[0] ? " and " : "", where[k]);
            snprintf(gpubuf, sizeof(gpubuf), ", GPU present (active, on %s)",
                     list);
            gpu = gpubuf;
        }
    }
    char attn[64] = "";
    if (janas_llm_model_attn_scores(llm->m) == JANAS_LLM_ATTN_FAST)
        snprintf(attn, sizeof(attn), ", attention scores at 16 bits%s",
                 janas_llm_model_attn_scores_auto(llm->m) ? " (long context)"
                                                          : "");
    snprintf(w->desc, sizeof(w->desc),
             "%s (%s), context %u tokens, expert cache %.1f GiB, up to %d "
             "compute threads%s, drafts %s%s%s%s",
             llm->name[0] ? llm->name : "model", llm->arch, llm->n_ctx,
             (double)llm->cache_bytes / (1 << 30), llm->n_compute, gpu,
             llm->draft     ? "from a second model"
             : llm->has_mtp ? "from the MTP block"
                            : "from the conversation",
             llm->bits, attn, llm->warm);
    return janas_api_copy_out(llm->desc, strlen(llm->desc), buf, cap, len);
}

int32_t janas_llm_set_gpu(janas_llm *llm, int32_t on)
{
    if (!llm)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    if (on && !janas_llm_model_gpu(llm->m))
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "this machine has no usable GPU");
    llm->gpu_wanted = on ? 1 : 0;
    janas_api_apply_mode(llm);
    return JANAS_LLM_OK;
}

size_t janas_api_text_len(const char *text, int32_t len)
{
    return len < 0 ? strlen(text) : (size_t)len;
}

int32_t janas_llm_tokenize(const janas_llm *llm, const char *text, int32_t len,
                           int32_t special, int32_t *ids, int32_t max,
                           int32_t *n)
{
    if (!llm || !text || !n || max < 0 || (max > 0 && !ids))
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    long r =
        janas_tokenizer_encode(llm->tok, text, janas_api_text_len(text, len),
                               special != 0, ids, (size_t)max);
    if (r < 0)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    if (r > INT32_MAX)
        return janas_api_fail(JANAS_LLM_EINVAL, "text too long");
    *n = (int32_t)r;
    return r > max ? janas_api_fail(JANAS_LLM_ESMALL, "%ld tokens, room for %d",
                                    r, max)
                   : JANAS_LLM_OK;
}

int32_t janas_llm_token_text(const janas_llm *llm, int32_t id, char *buf,
                             int32_t cap, int32_t *len)
{
    if (!llm || id < 0 || (uint32_t)id >= janas_tokenizer_n_vocab(llm->tok))
        return janas_api_fail(JANAS_LLM_EINVAL, "no such token");
    char tmp[1024];
    size_t n = janas_tokenizer_decode(llm->tok, id, tmp, sizeof(tmp));
    return janas_api_copy_out(tmp, n < sizeof(tmp) ? n : sizeof(tmp), buf, cap,
                              len);
}
