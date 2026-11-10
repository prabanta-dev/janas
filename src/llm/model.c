/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model.c - Janas-LLM forward pass (see model.h).
 *
 * What is computed follows the model definitions (llama.cpp's graphs are the
 * reference the logits are compared against); how it is computed is Janas's
 * own: fused single-pass kernels, a KV cache only for the attention layers,
 * the shared expert run as one more expert of the MoE dispatch.
 *
 * Architectures:
 *   qwen3moe   every layer: attention (q/k norms, NeoX RoPE) + routed experts
 *   qwen3next  every full_attention_interval-th layer: gated attention with
 *              partial RoPE; the others: Gated DeltaNet (see deltanet.h)
 *              with a causal convolution; every layer: routed experts plus a
 *              shared expert scaled by a sigmoid gate
 *   qwen35moe  (Qwen3.5, Qwen3.6) the same layers; the DeltaNet projections
 *              come separate (q|k|v, z, alpha and beta in f32) instead of
 *              grouped per key head; the last block of the file is the
 *              multi-token prediction block, not a layer of the model
 *   gemma4     sliding-window layers (a 1024-position window) and every
 *              sixth a full one, each kind with its head size, KV heads
 *              and RoPE; norms after attention and after the FFN too, GELU,
 *              a scale per layer, soft-capped logits
 * What sets them apart is data, in arch.h; a dense model is a mixture with
 * one expert per layer.
 *
 * This file runs the passes (forward, MTP); the layers are in model_attn.c,
 * model_rec.c and model_ffn.c, opening and closing in model_load.c, the
 * threads and the experts' profile in model_tune.c, saved states in
 * model_state.c.
 */
#define _GNU_SOURCE
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "attention.h"
#include "common/pool.h"
#include "common/sysinfo.h"
#include "deltanet.h"
#include "tuner.h"
#include "gpu.h"
#include "jmath.h"
#include "matvec.h"
#include "quant.h"
#include "vmath.h"

#include "model_impl.h"

struct janas_pool *janas_llm_model_compute(struct janas_llm_model *m)
{
    return m->compute;
}

void janas_llm_model_phases(const struct janas_llm_model *m,
                            double out[JANAS_PH_COUNT])
{
    memcpy(out, m->phase, sizeof(m->phase));
    /* la sonda stampa e azzera qui: chi legge le fasi sta separando due
       tratti (prefill e decodifica), e la scomposizione deve separarli
       insieme a lui, altrimenti il prefill sporca la decodifica */
    if (janas_m_exp_tracing())
        janas_m_exp_report();
}

static void embed(struct janas_llm_model *m, int32_t token, float *x)
{
    uint32_t dm = m->d_model;
    size_t row_bytes =
        dm / JANAS_QK * janas_qtype_block_size((int)m->tok_embd->type);
    const uint8_t *row =
        (const uint8_t *)data(m, m->tok_embd) + (size_t)token * row_bytes;
    janas_dequantize((int)m->tok_embd->type, (const void *)row, x, dm);
    if (m->a->sandwich) { /* Gemma: the embedding scaled by sqrt(d_model) */
        float s = sqrtf((float)dm);
        for (uint32_t i = 0; i < dm; i++)
            x[i] *= s;
    }
}

void janas_m_embed(struct janas_llm_model *m, int32_t token, float *x)
{
    embed(m, token, x);
}

/* RoPE angles of positions pos0 .. pos0 + n - 1 for n_rot dimensions, the
   same for every head; ff: a divisor of each pair's angle, or NULL. */
static void rope_table(float *cs, float *sn, uint32_t n, uint32_t pos0,
                       uint32_t n_rot, float base, const float *ff)
{
    uint32_t half = n_rot / 2;
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t i = 0; i < half; i++) {
            float theta = (float)(pos0 + j) *
                          janas_powf(base, -2.0f * (float)i / (float)n_rot);
            if (ff)
                theta /= ff[i];
            janas_sincosf(theta, &sn[(size_t)j * half + i],
                          &cs[(size_t)j * half + i]);
        }
}

/* With pictures in the sequence: each position's three (time, row, column;
   model_media.c), each frequency taking the one the multimodal sections
   give it */
static void rope_media(struct janas_llm_model *m, uint32_t n, uint32_t pos0)
{
    uint32_t half = m->n_rot / 2;
    for (uint32_t j = 0; j < n; j++) {
        float p3[3];
        janas_m_media_pos(m, pos0 + j, p3);
        for (uint32_t i = 0; i < half; i++) {
            int axis = i % 3 == 1 && i < 3 * m->mrope[1]   ? 1
                       : i % 3 == 2 && i < 3 * m->mrope[2] ? 2
                                                           : 0;
            float theta =
                p3[axis] *
                janas_powf(m->rope_base, -2.0f * (float)i / (float)m->n_rot);
            if (m->rope_freqs)
                theta /= m->rope_freqs[i];
            janas_sincosf(theta, &m->rope_sin[(size_t)j * half + i],
                          &m->rope_cos[(size_t)j * half + i]);
        }
    }
}

static void set_rope(struct janas_llm_model *m, uint32_t n, uint32_t pos0)
{
    if (m->n_media && (m->mrope[0] || m->mrope[1] || m->mrope[2]))
        rope_media(m, n, pos0);
    else
        rope_table(m->rope_cos, m->rope_sin, n, pos0, m->n_rot, m->rope_base,
                   m->rope_freqs);
    if (m->a->swa)
        rope_table(m->rope_cos_swa, m->rope_sin_swa, n, pos0, m->n_rot_swa,
                   m->rope_base_swa, NULL);
}

/* Gemma 4, after the FFN: x = (xres + rms_norm(x) * post_ffn) * out_scale,
   the FFN's output having been summed into a zeroed x */
static void sandwich_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    struct janas_llm_model *m = t->m;
    uint32_t dm = m->d_model;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads) {
        float *x = m->x + (size_t)j * dm;
        const float *r = m->xres + (size_t)j * dm;
        rms_norm(x, x, t->w, dm, m->eps);
        /* with per-layer embeddings the scale comes after their branch */
        float s = m->ple_dim ? 1.0f : t->ly->out_scale;
        for (uint32_t i = 0; i < dm; i++)
            x[i] = (r[i] + x[i]) * s;
    }
}

/* Gemma 4's mixture, token by token: the dense branch's output normalized
   aside (xbr), then the experts' input and the router's from the residual,
   then the experts' output normalized and the two summed */
static void branch1_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    struct janas_llm_model *m = t->m;
    uint32_t dm = m->d_model;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads)
        rms_norm(m->xbr + (size_t)j * dm, m->x + (size_t)j * dm,
                 f32(m, t->ly->post_norm1), dm, m->eps);
}

static void prep2_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    struct janas_llm_model *m = t->m;
    uint32_t dm = m->d_model;
    const float *rs = t->ly->router_scale;
    float sd = 1.0f / sqrtf((float)dm);
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads) {
        const float *r = m->xres + (size_t)j * dm;
        float *xn = m->xn + (size_t)j * dm;
        /* the experts' input, to Q8_K through xn */
        rms_norm(xn, r, f32(m, t->ly->pre_norm2), dm, m->eps);
        janas_q8k_quantize(xn, m->xq + j * (dm / JANAS_QK), dm);
        /* the router's: the residual normalized without weights, scaled
           by 1 / sqrt(d_model) and per dimension */
        double ss = 0.0;
        for (uint32_t i = 0; i < dm; i++)
            ss += (double)r[i] * r[i];
        float sc = 1.0f / sqrtf((float)(ss / dm) + m->eps);
        for (uint32_t i = 0; i < dm; i++)
            xn[i] = r[i] * sc * sd * rs[i];
    }
}

static void branch2_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    struct janas_llm_model *m = t->m;
    uint32_t dm = m->d_model;
    for (uint32_t j = (uint32_t)tid; j < t->n; j += (uint32_t)n_threads) {
        float *x = m->x + (size_t)j * dm;
        const float *b = m->xbr + (size_t)j * dm;
        rms_norm(x, x, f32(m, t->ly->post_norm2), dm, m->eps);
        for (uint32_t i = 0; i < dm; i++)
            x[i] += b[i];
    }
}

static void run_tok(struct janas_llm_model *m, const struct layer *ly,
                    uint32_t n, void (*fn)(void *, int, int))
{
    struct tok_job t = {.m = m, .ly = ly, .n = n};
    if (n < 2)
        fn(&t, 0, 1);
    else
        janas_pool_run(m->compute, fn, &t);
}

static int head_of_vocab(struct janas_llm_model *m,
                         const struct janas_block_q8k *x, float *logits,
                         uint32_t n);
static int mask_to_vocab(struct janas_llm_model *m, float *logits, uint32_t n);

/* logits soft-capped: c * tanh(l / c) */
static void softcap_worker(void *arg, int tid, int n_threads)
{
    struct tok_job *t = arg;
    float c = t->m->softcap, *l = t->out;
    size_t len = (size_t)t->n * t->m->n_vocab;
    size_t per = (len + (size_t)n_threads - 1) / (size_t)n_threads;
    size_t a = (size_t)tid * per, b = a + per < len ? a + per : len;
    for (size_t i = a; i < b; i++)
        l[i] = c * janas_tanhf(l[i] / c);
}

/* A key row turned back by the angles in cs, sn (half pairs, NeoX) and put
   in its bits again: a rotation can take a pair past the largest number of
   the row, so the scale is found anew. row: int8 or int16 (wide). */
static void unrotate_row(void *row, float *scale, uint32_t hd, uint32_t half,
                         const float *cs, const float *sn, int wide)
{
    float tmp[512], mx = 0.0f, z = *scale;
    int8_t *r8 = row;
    int16_t *r16 = row;
    for (uint32_t i = 0; i < hd; i++)
        tmp[i] = (wide ? (float)r16[i] : (float)r8[i]) * z;
    for (uint32_t i = 0; i < half; i++) {
        float a = tmp[i], b = tmp[i + half];
        tmp[i] = a * cs[i] + b * sn[i];
        tmp[i + half] = b * cs[i] - a * sn[i];
    }
    for (uint32_t i = 0; i < hd; i++)
        if (fabsf(tmp[i]) > mx)
            mx = fabsf(tmp[i]);
    float top = wide ? 32767.0f : 127.0f, inv = mx > 0 ? top / mx : 0.0f;
    for (uint32_t i = 0; i < hd; i++) {
        if (wide)
            r16[i] = (int16_t)lrintf(tmp[i] * inv);
        else
            r8[i] = (int8_t)lrintf(tmp[i] * inv);
    }
    *scale = mx / top;
}

/*
 * Forgets the positions keep .. keep + drop of the attention layers: what
 * follows them moves down and its keys are turned back by drop positions,
 * since the rotation a key carries depends on where it was. The recurrent
 * state is left alone: it is a summary of everything seen, and in a hybrid
 * model that is what carries the older text once the window has passed it.
 * The first `keep` positions stay where they are, which keeps the system
 * message - and the anchor it gives attention - in place. Each slot with its
 * own geometry and RoPE (Gemma 4: sliding-window layers and full ones); a
 * ring (a sliding-window layer's) keeps only its last positions, and those
 * are the ones that move.
 */
int janas_llm_model_shift(struct janas_llm_model *m, uint32_t keep,
                          uint32_t drop, uint32_t n_kv)
{
    /* pictures' positions are not slid (yet): the caller reads the prompt
       again instead */
    if (!drop || n_kv > m->n_ctx || keep + drop > n_kv || m->hd_max > 512 ||
        m->n_media)
        return -1;
    /* qwen4exp: its n-gram history first (the one part that can refuse) */
    if (m->hc_n && janas_m_hc_shift(m, keep, drop, n_kv) != 0)
        return -1;
    uint32_t move = n_kv - keep - drop, slots = m->n_attn + (m->mtp ? 1u : 0u);
    size_t elt = (size_t)1 << m->kv16;
    float cs[256], sn[256];
    uint8_t *ring_tmp = NULL;
    for (uint32_t sl = 0; sl < slots; sl++) {
        uint32_t hd = m->kv_hd[sl], cap = m->kv_cap[sl];
        int swa = m->kv_swa[sl];
        /* Nemotron-H's attention has no RoPE (its n_rot is the buffers'):
           the keys move, nothing to turn back */
        uint32_t n_rot = m->a->nh ? 0 : swa ? m->n_rot_swa : m->n_rot;
        uint32_t half = n_rot / 2;
        float base = swa ? m->rope_base_swa : m->rope_base;
        const float *ff = swa ? NULL : m->rope_freqs;
        for (uint32_t i = 0; i < half; i++) {
            float theta =
                (float)drop * janas_powf(base, -2.0f * (float)i / (float)n_rot);
            if (ff)
                theta /= ff[i];
            janas_sincosf(theta, &sn[i], &cs[i]);
        }
        size_t row = hd * elt;
        for (uint32_t h = 0; h < m->kv_heads[sl]; h++) {
            uint8_t *k = (uint8_t *)m->kcache + m->kv_off[sl] + h * cap * row;
            uint8_t *v = (uint8_t *)m->vcache + m->kv_off[sl] + h * cap * row;
            float *kz = m->kscale + m->ks_off[sl] + (size_t)h * cap;
            float *vz = m->vscale + m->ks_off[sl] + (size_t)h * cap;
            if (!m->kv_ring[sl]) {
                if (move) {
                    memmove(k + keep * row, k + (keep + drop) * row,
                            move * row);
                    memmove(v + keep * row, v + (keep + drop) * row,
                            move * row);
                    memmove(kz + keep, kz + keep + drop, move * sizeof(float));
                    memmove(vz + keep, vz + keep + drop, move * sizeof(float));
                }
                for (uint32_t p = keep; p < keep + move && half; p++)
                    unrotate_row(k + p * row, kz + p, hd, half, cs, sn,
                                 m->kv16);
                continue;
            }
            /* a ring: the positions it still has, from a copy of it, each
               to its new row (they cannot land on a kept one: see the
               bound on lo) */
            if (!ring_tmp &&
                !(ring_tmp = malloc(2 * (size_t)cap * (row + sizeof(float)))))
                return -1;
            uint8_t *tk = ring_tmp, *tv = tk + cap * row;
            float *tkz = (float *)(void *)(tv + cap * row), *tvz = tkz + cap;
            memcpy(tk, k, cap * row);
            memcpy(tv, v, cap * row);
            memcpy(tkz, kz, cap * sizeof(float));
            memcpy(tvz, vz, cap * sizeof(float));
            uint32_t lo = keep + drop;
            if (n_kv > cap && n_kv - cap > lo)
                lo = n_kv - cap;
            for (uint32_t p = lo; p < n_kv; p++) {
                uint32_t src = p % cap, dst = (p - drop) % cap;
                memcpy(k + dst * row, tk + src * row, row);
                memcpy(v + dst * row, tv + src * row, row);
                kz[dst] = tkz[src];
                vz[dst] = tvz[src];
                if (half)
                    unrotate_row(k + dst * row, kz + dst, hd, half, cs, sn,
                                 m->kv16);
            }
        }
    }
    free(ring_tmp);
    if (m->hc_n) /* and the indexer's keys and blocks */
        janas_m_qsa_shift(m, keep, drop, n_kv);
    if (m->ring_hi > keep + drop)
        m->ring_hi -= drop;
    /* the recurrent state holds what it has seen, dropped tokens included -
       that is what carries the older text - but the position it stands at
       moves with everything else */
    if (m->base_pos != UINT32_MAX) {
        uint32_t b = m->base_pos, t = b + m->log_n;
        if (b >= keep + drop)
            m->base_pos = b - drop;
        else if (m->log_n && t > keep) {
            /* the log runs into what is dropped (or past it): replaying
               it would count the dropped tokens and land short of the
               tip, so the state goes on from the tip, the other buffer */
            m->base = 1 - m->base;
            m->base_pos = t >= keep + drop ? t - drop : keep;
            m->log_n = 0;
        } else if (b > keep)
            m->base_pos = keep;
    }
    return 0;
}

static int forward(struct janas_llm_model *m, const int32_t *tokens, uint32_t n,
                   uint32_t pos0, float *logits, int all_logits);

/*
 * Reading ahead where it pays: it costs a router and the reads' submission
 * on every token (~4 ms on Qwen3-Next-80B), and gains only where the cache
 * misses many experts. Measured on 30 Sep 2026 (that model, every bit): -3%
 * at ~12 misses a token, even at ~40, +4.5% at ~77, +5-7.5% at ~145. So on
 * above PF_ON experts a token the cache would have missed (its misses and
 * those the reads ahead spared it), off below PF_OFF, as it was between
 * them; off, it costs nothing.
 */
#define PF_WINDOW 32
#define PF_ON 60
#define PF_OFF 40

/*
 * Reading ahead in prefill blocks (unless JANAS_PREFETCH_PREFILL=0 or 2):
 * on after a block whose layers would have missed more than PFB_ON experts
 * each, as a rule (the misses and those the reads ahead spared); only for
 * blocks of PFB_MIN_N tokens or more, the passes that verify drafts being
 * decoding. It costs layer l + 1's router once more on every token (1.1
 * ms a token on Qwen3-Next-80B) and spares half the time waited for the
 * disk where the cache is small for the model (6 Oct 2026).
 */
#define PFB_MIN_N 32
#define PFB_ON 32

static void prefetch_block_decide(struct janas_llm_model *m,
                                  const struct janas_expert_cache_stats *a,
                                  const struct janas_expert_cache_stats *b)
{
    uint64_t would =
        (b->misses - a->misses) + (b->prefetch_used - a->prefetch_used);
    m->pfb_on = would > (uint64_t)PFB_ON * m->n_layer;
}

static void prefetch_decide(struct janas_llm_model *m,
                            const struct janas_expert_cache_stats *a,
                            const struct janas_expert_cache_stats *b)
{
    m->pf_would +=
        (b->misses - a->misses) + (b->prefetch_used - a->prefetch_used);
    if (++m->pf_tokens < PF_WINDOW)
        return;
    uint64_t per = m->pf_would / m->pf_tokens;
    if (!m->pf_on && per > PF_ON)
        m->pf_on = 1;
    else if (m->pf_on && per < PF_OFF)
        m->pf_on = 0;
    m->pf_tokens = 0;
    m->pf_would = 0;
}

/*
 * A prompt's blocks read with the GPU's whole pass or without it (the
 * CPU's attention, the products where they go), whichever this machine
 * reads faster at this position: the attention's cost grows with the
 * position, and on the GPU it may grow faster or slower than on the CPU -
 * Qwen3-4B on the development laptop's Arc read 4,096 tokens at 72.6
 * tokens/s with it and 95.9 without, 16,384 at 31.8 and 55.6 (26 Oct
 * 2026), where a card with fast memory of its own may well win. So every
 * LP_BLOCKS of the engine's blocks of positions one block each way,
 * measured in turn (the first of a band alternating: a later block costs
 * more), and the faster for the rest of the band - no threshold, only how
 * often to look again, at most one slower block in LP_BLOCKS.
 * JANAS_GPU_LONG=0: the whole pass always, as before.
 */
#define LP_BLOCKS 8

static int long_prompt_on(const struct janas_llm_model *m, uint32_t n,
                          uint32_t pos0)
{
    static int off = -1;
    if (off < 0) {
        const char *e = getenv("JANAS_GPU_LONG");
        off = e && atoi(e) <= 0;
    }
    (void)pos0;
    /* the whole pass asked for (JANAS_GPU_TOKEN=1): as asked */
    return !off && n >= JANAS_GPU_FAST_N && m->gtok_state == 1 &&
           janas_llm_model_gpu_prompts(m) && !janas_m_gpu_token_asked();
}

/* this block's way: lp_cpu set for the one without the whole pass */
static void long_prompt_begin(struct janas_llm_model *m, uint32_t pos0)
{
    uint32_t band = pos0 / (LP_BLOCKS * m->blk);
    if (band != m->lp.band || m->lp.state == 0) {
        if (band != m->lp.band)
            m->lp.first = (int)(band & 1u);
        m->lp.band = band;
        m->lp.state = 0;
    }
    if (m->lp.state < 2) /* the first way, then the other */
        m->lp_cpu = m->lp.state == 0 ? m->lp.first : !m->lp.first;
    else
        m->lp_cpu = m->lp.cpu < m->lp.gpu;
}

static void long_prompt_end(struct janas_llm_model *m, double sec_per_token,
                            int on_gpu)
{
    int cpu = m->lp_cpu;
    m->lp_cpu = 0;
    if (m->lp.state >= 2 || sec_per_token <= 0 || on_gpu == cpu)
        return; /* chosen already; or not the way meant (none measured) */
    *(cpu ? &m->lp.cpu : &m->lp.gpu) = sec_per_token;
    if (++m->lp.state == 2 && getenv("JANAS_GPU_VERBOSE"))
        fprintf(stderr,
                "janas: a prompt from position %u: %.2f ms a token with the "
                "GPU's whole pass, %.2f without; %s\n",
                m->lp.band * LP_BLOCKS * m->blk, m->lp.gpu * 1e3,
                m->lp.cpu * 1e3, m->lp.cpu < m->lp.gpu ? "without" : "with it");
}

int janas_llm_model_forward(struct janas_llm_model *m, const int32_t *tokens,
                            uint32_t n, uint32_t pos0, float *logits,
                            int all_logits)
{
    m->fwd_pos0 = pos0; /* for the observers (observe.h) */
    int lp = long_prompt_on(m, n, pos0);
    uint64_t runs0 = m->gtok_runs[2];
    if (lp)
        long_prompt_begin(m, pos0);
    /* each pass measured for the tuner: the configuration never changes
       the results */
    int cls = janas_tune_class(n), cand = janas_tuner_pick(&m->tuner, cls);
    janas_m_apply_candidate(m, cand, n);
    /* measured without the time spent waiting for experts read from disk:
       it depends on the cache, not on the configuration */
    struct janas_expert_cache_stats c0, c1;
    janas_expert_cache_stats(m->cache, &c0);
    double k0 = janas_llm_model_npu_compile_seconds(m), t0 = now();
    int r = forward(m, tokens, n, pos0, logits, all_logits);
    double dt = now() - t0;
    janas_m_npu_pass_end(m);
    janas_expert_cache_stats(m->cache, &c1);
    dt -= c1.wait_seconds - c0.wait_seconds;
    /* nor the NPU's first compilation of a matrix: once a session */
    dt -= janas_llm_model_npu_compile_seconds(m) - k0;
    m->last_pass_s = dt > 0 && !m->tuner.trying ? dt : 0;
    /* while the cache is still filling in the background, passes are not
       representative: neither the tuner nor the draft planner may learn
       from them, or a slow start decides the whole session */
    if (lp)
        long_prompt_end(m, r == 0 && dt > 0 ? dt / n : 0,
                        m->gtok_runs[2] > runs0);
    if (r == 0 && dt > 0 && !janas_llm_model_settling(m))
        janas_tuner_record(&m->tuner, cls, cand, dt / n);
    if (r == 0 && n == 1 && m->pf_mode == 1)
        prefetch_decide(m, &c0, &c1);
    if (r == 0 && n >= PFB_MIN_N && m->pfb_mode == 1)
        prefetch_block_decide(m, &c0, &c1);
    janas_m_arena_to_gpu(m);
    return r;
}

double janas_llm_model_pass_seconds(const struct janas_llm_model *m)
{
    return m->last_pass_s;
}

static int forward(struct janas_llm_model *m, const int32_t *tokens, uint32_t n,
                   uint32_t pos0, float *logits, int all_logits)
{
    if (n < 1 || n > m->blk || pos0 + n > m->n_ctx)
        return -1;
    for (uint32_t j = 0; j < n; j++)
        if (tokens[j] < 0 || (uint32_t)tokens[j] >= m->n_vocab)
            return -1;
    uint32_t dm = m->d_model;
    size_t nbd = dm / JANAS_QK;
    for (uint32_t j = 0; j < n; j++) {
        /* a picture's token: its embedding from the encoder */
        const float *pic = m->n_media ? janas_m_media_row(m, pos0 + j) : NULL;
        if (pic)
            memcpy(m->x + (size_t)j * dm, pic, sizeof(float) * dm);
        else
            embed(m, tokens[j], m->x + (size_t)j * dm);
    }
    if (m->trace >= 2) {
        fprintf(stderr, "tokens:");
        for (uint32_t j = 0; j < n; j++)
            fprintf(stderr, " %d", tokens[j]);
        fprintf(stderr, "\n");
        trace_row(m, "embed", m->x + (size_t)(n - 1) * dm, dm);
    }
    /* what can fail before the state is touched: a failure leaves it as
       it was */
    if (m->ple_dim && janas_m_ple_inputs(m, tokens, n, pos0) != 0)
        return -1;
    /* from the start: nothing of the GPU's copy of the state matters, the
       host's (the recurrent states zeroed below) is the one */
    if (m->gtok_dev && pos0 == 0)
        m->gtok_where = JANAS_GT_HOST;
    if (m->a->rec) {
        if (pos0 == 0) {
            m->src = 0;
            m->replay = 0;
            memset(m->ssm_buf, 0,
                   (size_t)m->n_rec * m->ssm_per * sizeof(uint16_t));
            memset(m->conv_buf, 0,
                   (size_t)m->n_rec * m->conv_ch * (m->d_conv - 1) *
                       sizeof(float));
        } else if (m->base_pos == UINT32_MAX || pos0 < m->base_pos ||
                   pos0 > m->base_pos + m->log_n) {
            return -1; /* before the last call: that state is gone */
        } else if (pos0 == m->base_pos + m->log_n && m->log_n > 0) {
            m->src = 1 - m->base; /* from the tip */
            m->replay = 0;
        } else {
            m->src = m->base; /* replay the accepted tokens on the base */
            m->replay = pos0 - m->base_pos;
        }
        m->base_pos = UINT32_MAX; /* invalid until this pass completes */
    }
    set_rope(m, n, pos0);
    /* a token (or, with JANAS_GPU_BLOCK=1, a block) on the GPU, whole,
       where it can (model_gpu.c) */
    /* not where the NPU takes this pass's products: then the block goes
       product by product, the NPU's to it, the others as before */
    m->gtok_tokens = tokens; /* qwen4exp's n-gram embeddings */
    int gpu = logits && !janas_m_npu_takes(m, n)
                  ? janas_m_gpu_token(m, pos0, n, all_logits, logits)
                  : -1;
    if (gpu == -2)
        return -1; /* the GPU failed after replaying on the base in place:
                      no state to go on from (base_pos stays invalid) */
    if (gpu != 0)  /* the CPU's pass: the state the GPU made newer back */
        janas_m_gtok_host(m, 1);
    if (gpu == 0) {
        /* Gemma 4's soft cap, as below */
        if (m->softcap > 0.0f) {
            struct tok_job sc = {
                .m = m, .n = all_logits ? n : 1, .out = logits};
            janas_pool_run(m->compute, softcap_worker, &sc);
        }
        /* its head the whole vocabulary's: the others -inf here */
        if (m->vs_n && mask_to_vocab(m, logits, all_logits ? n : 1) != 0)
            return -1;
        m->ring_hi = pos0 + n;
        if (m->a->rec && n <= JANAS_LLM_MAX_BLOCK) {
            /* replayed and logged by the GPU: as at the end */
            m->base = m->src;
            m->base_pos = pos0;
            m->log_n = n;
        } else if (m->a->rec) { /* a long block: as at the end */
            m->base = 1 - m->src;
            m->base_pos = pos0 + n;
            m->log_n = 0;
        }
        return 0;
    }
    /* Nemotron-H: one part a layer (model_nh.c) */
    if (m->a->nh) {
        if (janas_m_nh_layers(m, n, pos0) != 0)
            return -1;
        goto output;
    }
    /* qwen4exp: its layers on the streams of the residual (model_hc.c) */
    if (m->hc_n) {
        if (janas_m_hc_layers(m, tokens, n, pos0) != 0)
            return -1;
        goto output;
    }
    /* a dense block's first layers on the NPU, graph by graph, as many as
       it holds (model_npu.c); the others here */
    int npu_layers = janas_m_npu_layers(m, n, pos0);
    if (npu_layers == -2)
        return -1; /* the NPU failed halfway: no state to go on from */
    for (uint32_t l = npu_layers > 0 ? (uint32_t)npu_layers : 0; l < m->n_layer;
         l++) {
        m->ana_layer = l; /* the Ana-Net probe's row */
        const struct layer *ly = &m->layers[l];
        janas_m_norm_quant(m, f32(m, ly->attn_norm), 0, n);
        if (m->npu) /* the NPU's calibration (JANAS_NPU_A8=1) */
            janas_m_npu_calib(m, 3, l, m->xn, n, dm, dm);
        trace_row(m, "attn_norm", m->xn + (size_t)(n - 1) * dm, dm);
        if (ly->rec)
            janas_m_rec_layer(m, ly, n);
        else
            janas_m_attn_layer(m, ly, n, pos0);
        /* the early predictor: layer l + 1's router on the state before
           layer l's experts (its norm and router buffers are made again
           just below) */
        if (m->pred_fn && m->pred_early && l + 1 < m->n_layer &&
            janas_m_route_predict(m, l + 1, n) != 0)
            return -1;
        if (m->pf_on && n == 1 && l + 1 < m->n_layer)
            janas_m_prefetch(m, l);
        else if (m->pfb_on && n >= PFB_MIN_N && l + 1 < m->n_layer)
            janas_m_prefetch_block(m, l, n);
        janas_m_norm_quant(m, f32(m, ly->ffn_norm), 0, n);
        if (m->npu)
            janas_m_npu_calib(m, 1, l, m->xn, n, dm, dm);
        trace_row(m, "post_norm", m->xn + (size_t)(n - 1) * dm, dm);
        if (ly->post_ffn) {
            /* the FFN's output normalized before the residual: summed
               into a zeroed x, the residual aside */
            memcpy(m->xres, m->x, (size_t)n * dm * sizeof(float));
            memset(m->x, 0, (size_t)n * dm * sizeof(float));
        }
        if (m->n_obs) /* the observers (observe.h) */
            janas_m_obs_ffn_input(m, l, m->xn, n, dm);
        if (ly->pre_norm2) {
            /* Gemma 4's mixture: the dense branch on xn / xq as normed
               above, then the experts' branch */
            size_t bytes = (size_t)n * dm * sizeof(float);
            if (janas_m_moe_block(m, ly, m->cache, &m->j.layers[l], l, n,
                                  MOE_SHARED) != 0)
                return -1;
            run_tok(m, ly, n, branch1_worker);
            trace_row(m, "ffn_mlp", m->xbr + (size_t)(n - 1) * dm, dm);
            run_tok(m, ly, n, prep2_worker);
            memset(m->x, 0, bytes);
            if (janas_m_moe_block(m, ly, m->cache, &m->j.layers[l], l, n,
                                  MOE_ROUTED) != 0)
                return -1;
            run_tok(m, ly, n, branch2_worker);
            trace_row(m, "ffn_moe_comb", m->x + (size_t)(n - 1) * dm, dm);
        } else if (janas_m_moe_block(m, ly, m->cache, &m->j.layers[l], l, n,
                                     MOE_ALL) != 0) {
            return -1;
        }
        if (ly->post_ffn) {
            struct tok_job st = {
                .m = m, .ly = ly, .w = f32(m, ly->post_ffn), .n = n};
            if (n < 2)
                sandwich_worker(&st, 0, 1);
            else
                janas_pool_run(m->compute, sandwich_worker, &st);
        }
        if (m->ple_dim)
            janas_m_ple_layer(m, ly, l, n);
        if (m->pred_fn && !m->pred_early && l + 1 < m->n_layer &&
            janas_m_route_predict(m, l + 1, n) != 0)
            return -1;
        if (m->trace) {
            /* JANAS_TRACE=1: the state after each layer, last token of the
               block, to compare with another implementation */
            const float *v = m->x + (size_t)(n - 1) * dm;
            double nn = 0;
            for (uint32_t i = 0; i < dm; i++)
                nn += (double)v[i] * v[i];
            fprintf(stderr, "l_out-%-3u norm %10.4f  %9.4f %9.4f %9.4f %9.4f\n",
                    l, sqrt(nn), v[0], v[1], v[2], v[3]);
        }
        if (m->n_obs && janas_m_obs_layer_out(m, l, m->x, n, dm))
            break; /* an observer stops the forward here */
    }
    janas_m_ana_frame(m, n); /* the Ana-Net probe's picture */
output:
    m->ring_hi = pos0 + n; /* the rings hold up to here */
    double td = now();
    uint32_t first = all_logits ? 0 : n - 1;
    /* with an MTP block, every token's final hidden state is kept for it:
       qwen4exp's is its residual's streams before the last mix, which is
       the output norm (llama.cpp's h_nextn) */
    int keep = m->mtp || m->asst;
    if (m->hc_n && keep)
        memcpy(m->hid, m->hc_res, (size_t)n * m->hc_n * dm * sizeof(float));
    if (m->hc_n) {
        janas_m_hc_head(m, first, n);
        td = now(); /* the mix's time is the mixes' */
    } else
        janas_m_norm_quant(m, f32(m, m->out_norm), keep ? 0 : first, n);
    if (keep && !m->hc_n)
        memcpy(m->hid, m->xn, (size_t)n * dm * sizeof(float));
    if (logits && m->vs_n) { /* a vocabulary of its own: its rows alone */
        if (head_of_vocab(m, m->xq + first * nbd, logits, n - first) != 0)
            return -1;
    } else if (logits) { /* without: the hidden states alone, an embedding */
        struct janas_matvec_task to = {.type = (int)m->output->type,
                                       .w = data(m, m->output),
                                       .x = m->xq + first * nbd,
                                       .y = logits,
                                       .rows = m->n_vocab,
                                       .cols = dm,
                                       .n_vec = n - first};
        janas_m_products(m, janas_m_gpu(m, 0), &to, 1);
    }
    if (logits && m->softcap > 0.0f && !m->vs_n) {
        struct tok_job sc = {.m = m, .n = n - first, .out = logits};
        janas_pool_run(m->compute, softcap_worker, &sc);
    }
    m->phase[JANAS_PH_OUTPUT] += now() - td;
    if (m->a->rec && n <= JANAS_LLM_MAX_BLOCK) {
        m->base = m->src;
        m->base_pos = pos0;
        m->log_n = n;
    } else if (m->a->rec) {
        /* not logged: the state stands at the end of the call, in the
           buffer the call wrote, and nothing before it can be replayed */
        m->base = 1 - m->src;
        m->base_pos = pos0 + n;
        m->log_n = 0;
    }
    return 0;
}

uint32_t janas_llm_model_max_block(const struct janas_llm_model *m)
{
    return m->blk;
}

int janas_llm_model_has_mtp(const struct janas_llm_model *m)
{
    return m->mtp || m->asst;
}

int janas_llm_mtp_chain_fixed(const struct janas_llm_model *m)
{
    return m->asst != NULL;
}

int janas_llm_model_recurrent(const struct janas_llm_model *m)
{
    return m->a->rec != JANAS_REC_NONE;
}

const float *janas_llm_model_hidden(const struct janas_llm_model *m)
{
    return m->hid;
}

uint32_t janas_llm_model_hidden_width(const struct janas_llm_model *m)
{
    return m->hc_n ? m->hc_n * m->d_model : m->d_model;
}

const float *janas_llm_model_normed(const struct janas_llm_model *m)
{
    return m->xn;
}

const float *janas_llm_mtp_hidden(const struct janas_llm_model *m)
{
    return m->mtp_hid;
}

#if defined(__x86_64__)
/* Sum of exp(l[i] - mx) over whole groups of 8; *done: how many summed. */
__attribute__((target("avx2,fma"))) static double
exp_sum_avx2(const float *l, uint32_t n, float mx, uint32_t *done)
{
    __m256 acc = _mm256_setzero_ps(), vm = _mm256_set1_ps(mx);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_add_ps(
            acc, janas_exp8(_mm256_sub_ps(_mm256_loadu_ps(l + i), vm)));
    *done = i;
    return janas_hsum8(acc);
}
#endif

double janas_m_exp_sum(const float *l, uint32_t n, float mx)
{
    double sum = 0;
    uint32_t i = 0;
#if defined(__x86_64__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        sum = exp_sum_avx2(l, n, mx, &i);
#endif
    for (; i < n; i++)
        sum += janas_exp_det(l[i] - mx);
    return sum;
}

int janas_llm_model_set_vocab(struct janas_llm_model *m, const int32_t *ids,
                              uint32_t n)
{
    m->vs_n = m->vs_cap = 0;
    if (!n)
        return 0;
    int type = (int)m->output->type;
    if (type == JANAS_Q6_K_P || !janas_qtype_block_size(type))
        return -1;
    size_t row = m->d_model / JANAS_QK * janas_qtype_block_size(type);
    uint8_t *w = realloc(m->vs_w, (size_t)n * row);
    if (!w)
        return -1;
    m->vs_w = w;
    int32_t *v = realloc(m->vs_ids, n * sizeof(int32_t));
    if (!v)
        return -1;
    m->vs_ids = v;
    const uint8_t *src = data(m, m->output);
    uint32_t k = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= m->n_vocab)
            continue;
        memcpy(m->vs_w + (size_t)k * row, src + (size_t)ids[i] * row, row);
        m->vs_ids[k++] = ids[i];
    }
    m->vs_n = k;
    return 0;
}

/* The logits of n vectors through the vocabulary's rows alone, the others
   -inf (janas_llm_model_set_vocab). */
static int vocab_room(struct janas_llm_model *m, uint32_t n)
{
    if (n > m->vs_cap) {
        float *y = realloc(m->vs_y, (size_t)n * m->vs_n * sizeof(float));
        if (!y)
            return -1;
        m->vs_y = y;
        m->vs_cap = n;
    }
    return 0;
}

/* The vocabulary's logits of n rows into vs_y, then every row -inf but
   for them. */
static void scatter_vocab(struct janas_llm_model *m, float *logits, uint32_t n)
{
    for (uint32_t j = 0; j < n; j++) {
        float *l = logits + (size_t)j * m->n_vocab;
        for (uint32_t i = 0; i < m->n_vocab; i++)
            l[i] = -INFINITY;
        for (uint32_t i = 0; i < m->vs_n; i++)
            l[m->vs_ids[i]] = m->vs_y[(size_t)j * m->vs_n + i];
    }
}

/* Logits of the whole head (the GPU's whole pass) cut to the vocabulary. */
static int mask_to_vocab(struct janas_llm_model *m, float *logits, uint32_t n)
{
    if (vocab_room(m, n) != 0)
        return -1;
    for (uint32_t j = 0; j < n; j++)
        for (uint32_t i = 0; i < m->vs_n; i++)
            m->vs_y[(size_t)j * m->vs_n + i] =
                logits[(size_t)j * m->n_vocab + m->vs_ids[i]];
    scatter_vocab(m, logits, n);
    return 0;
}

static int head_of_vocab(struct janas_llm_model *m,
                         const struct janas_block_q8k *x, float *logits,
                         uint32_t n)
{
    if (vocab_room(m, n) != 0)
        return -1;
    struct janas_matvec_task t = {.type = (int)m->output->type,
                                  .w = m->vs_w,
                                  .x = x,
                                  .y = m->vs_y,
                                  .rows = m->vs_n,
                                  .cols = m->d_model,
                                  .n_vec = n};
    /* on the CPU: the GPU and the NPU know the model's weights, not these */
    janas_matvec_q4k_group(m->compute, &t, 1);
    if (m->softcap > 0.0f) /* as softcap_worker, on the rows computed */
        for (size_t i = 0; i < (size_t)n * m->vs_n; i++)
            m->vs_y[i] = m->softcap * janas_tanhf(m->vs_y[i] / m->softcap);
    scatter_vocab(m, logits, n);
    return 0;
}

void janas_llm_mtp_note_tokens(struct janas_llm_model *m, const int32_t *tokens,
                               uint32_t n)
{
    if (!m->dh_w)
        return;
    for (uint32_t j = 0; j < n; j++) {
        int32_t t = tokens[j];
        if (t < 0 || (uint32_t)t >= m->n_vocab || m->dh_in[t] ||
            m->dh_n == m->dh_cap)
            continue;
        memcpy(m->dh_w + (size_t)m->dh_n * m->dh_row,
               m->dh_src + (size_t)t * m->dh_row, m->dh_row);
        m->dh_ids[m->dh_n++] = t;
        m->dh_in[t] = 1;
    }
}

int janas_llm_mtp_draft(struct janas_llm_model *m, const int32_t *tokens,
                        const float *hidden, uint32_t n, uint32_t pos0,
                        int32_t *token, float *conf)
{
    if (m->asst) { /* Gemma 4's assistant: the last row alone */
        uint32_t dm = m->d_model;
        if (n < 1 || n > m->blk)
            return -1;
        return janas_m_assist_draft(
            m, tokens[n - 1], hidden + (size_t)(n - 1) * dm, pos0 + n - 1,
            m->mtp_hid + (size_t)(n - 1) * dm, token, conf);
    }
    if (janas_llm_mtp_forward(m, tokens, hidden, n, pos0, NULL, 0) != 0)
        return -1;
    janas_m_apply_candidate(m, janas_tuner_best(&m->tuner, 0),
                            1); /* one-row head */
    double td = now();
    size_t nbd = m->d_model / JANAS_QK;
    struct janas_matvec_task t = {.type = (int)m->output->type,
                                  .w = m->dh_w,
                                  .x = m->xq + (n - 1) * nbd,
                                  .y = m->dh_logits,
                                  .rows = m->dh_n,
                                  .cols = m->d_model};
    janas_m_products(m, janas_m_gpu(m, 0), &t, 1);
    /* the best row and its softmax probability within the head */
    float *l = m->dh_logits, mx = l[0];
    uint32_t best = 0;
    for (uint32_t i = 1; i < m->dh_n; i++)
        if (l[i] > mx) {
            mx = l[i];
            best = i;
        }
    double sum = 0;
    uint32_t i = 0;
#if defined(__x86_64__)
    if (m->avx2)
        sum = exp_sum_avx2(l, m->dh_n, mx, &i);
#endif
    for (; i < m->dh_n; i++)
        sum += janas_exp_det(l[i] - mx);
    *token = m->dh_ids[best];
    *conf = (float)(1.0 / sum);
    m->phase[JANAS_PH_OUTPUT] += now() - td;
    return 0;
}

int janas_llm_mtp_forward(struct janas_llm_model *m, const int32_t *tokens,
                          const float *hidden, uint32_t n, uint32_t pos0,
                          float *logits, int all_logits)
{
    if (m->asst) /* no keys and values of its own: nothing to fill */
        return logits || n < 1 || n > m->blk ? -1 : 0;
    if (!m->mtp || n < 1 || n > m->blk || pos0 + n > m->n_ctx)
        return -1;
    for (uint32_t j = 0; j < n; j++)
        if (tokens[j] < 0 || (uint32_t)tokens[j] >= m->n_vocab)
            return -1;
    janas_m_apply_candidate(m, janas_tuner_best(&m->tuner, janas_tune_class(n)),
                            n);
    uint32_t dm = m->d_model;
    size_t nbd = dm / JANAS_QK, nbc = 2 * dm / JANAS_QK;
    const struct layer *ly = &m->mtp_layer;
    double ta = now();
    if (m->hc_n) { /* qwen4exp's block, on the residual's streams */
        uint32_t first = all_logits ? 0 : n - 1;
        for (uint32_t j = 0; j < n; j++)
            embed(m, tokens[j], m->x + (size_t)j * dm);
        set_rope(m, n, pos0);
        if (janas_m_hc_mtp(m, hidden, n, pos0, first) != 0)
            return -1;
        m->phase[JANAS_PH_QKV] += now() - ta;
        if (!logits)
            return 0;
        double tl = now();
        struct janas_matvec_task to = {.type = (int)m->output->type,
                                       .w = data(m, m->output),
                                       .x = m->xq + first * nbd,
                                       .y = logits,
                                       .rows = m->n_vocab,
                                       .cols = dm,
                                       .n_vec = n - first};
        janas_m_products(m, janas_m_gpu(m, 0), &to, 1);
        m->phase[JANAS_PH_OUTPUT] += now() - tl;
        return 0;
    }
    /*
     * [norm(embedding); norm(hidden)], projected back to d_model. Settled on
     * data (tests/mtp_eval.c, 398 positions of C source): this order, with
     * the hidden state after the main model's final norm, matches the main
     * model's next choice 88.9% of the times; the state before the norm
     * 86.4%; the other order 0%.
     */
    for (uint32_t j = 0; j < n; j++) {
        float *c = m->cat + (size_t)j * 2 * dm;
        embed(m, tokens[j], m->x + (size_t)j * dm);
        rms_norm(c, m->x + (size_t)j * dm, f32(m, m->enorm), dm, m->eps);
        rms_norm(c + dm, hidden + (size_t)j * dm, f32(m, m->hnorm), dm, m->eps);
        janas_q8k_quantize(c, m->catq + j * nbc, 2 * dm);
    }
    struct janas_matvec_task tp = {.type = (int)m->eh_proj->type,
                                   .w = data(m, m->eh_proj),
                                   .x = m->catq,
                                   .y = m->x,
                                   .rows = dm,
                                   .cols = 2 * dm,
                                   .n_vec = n};
    janas_m_products(m, janas_m_gpu(m, 0), &tp, 1);
    m->phase[JANAS_PH_QKV] += now() - ta;
    set_rope(m, n, pos0);
    for (uint32_t j = 0; j < n; j++) {
        rms_norm(m->xn + (size_t)j * dm, m->x + (size_t)j * dm,
                 f32(m, ly->attn_norm), dm, m->eps);
        janas_q8k_quantize(m->xn + (size_t)j * dm, m->xq + j * nbd, dm);
    }
    janas_m_attn_layer(m, ly, n, pos0);
    for (uint32_t j = 0; j < n; j++) {
        rms_norm(m->xn + (size_t)j * dm, m->x + (size_t)j * dm,
                 f32(m, ly->ffn_norm), dm, m->eps);
        janas_q8k_quantize(m->xn + (size_t)j * dm, m->xq + j * nbd, dm);
    }
    if (janas_m_moe_block(m, ly, m->mtp_cache, m->mtp_jl, m->mtp_cl, n,
                          MOE_ALL) != 0)
        return -1;
    double td = now();
    uint32_t first = all_logits ? 0 : n - 1;
    for (uint32_t j = 0; j < n; j++) {
        rms_norm(m->xn + (size_t)j * dm, m->x + (size_t)j * dm,
                 f32(m, m->mtp_norm), dm, m->eps);
        janas_q8k_quantize(m->xn + (size_t)j * dm, m->xq + j * nbd, dm);
    }
    memcpy(m->mtp_hid, m->xn, (size_t)n * dm * sizeof(float));
    if (!logits) { /* only the KV cache and the hidden states wanted */
        m->phase[JANAS_PH_OUTPUT] += now() - td;
        return 0;
    }
    struct janas_matvec_task to = {.type = (int)m->output->type,
                                   .w = data(m, m->output),
                                   .x = m->xq + first * nbd,
                                   .y = logits,
                                   .rows = m->n_vocab,
                                   .cols = dm,
                                   .n_vec = n - first};
    janas_m_products(m, janas_m_gpu(m, 0), &to, 1);
    m->phase[JANAS_PH_OUTPUT] += now() - td;
    return 0;
}

int janas_llm_model_decode(struct janas_llm_model *m, int32_t token,
                           uint32_t pos, float *logits)
{
    return janas_llm_model_forward(m, &token, 1, pos, logits, 0);
}
