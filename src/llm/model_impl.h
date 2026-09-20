/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_impl.h - the insides of a model, for the files that make up
 * model.c's work (model.c, model_state.c). Nothing outside src/llm/model*.c
 * includes it: the rest of the engine sees the opaque struct of model.h.
 */
#ifndef JANAS_LLM_MODEL_IMPL_H
#define JANAS_LLM_MODEL_IMPL_H

#include "model.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

#include "arch.h"
#include "attention.h"
#include "common/pool.h"
#include "gpu.h"
#include "jns.h"
#include "quant.h"
#include "tuner.h"

#define MAX_K 64
#define MAX_CONV 8
#define MAX_SSM_DIM 256
#define SSM_ROWS 32 /* state rows per recurrence work item */

struct layer {
    int rec;       /* Gated DeltaNet layer (qwen3next), else attention */
    uint32_t slot; /* index among the layers of its kind: KV or state */
    /* attention geometry: the model's, or per kind of layer (Gemma 4) */
    uint32_t hd, n_kv, n_rot;
    int swa; /* sliding window, its own RoPE table and attention */
    /* wv NULL: V is K (Gemma 4's full layers) */
    const struct janas_jns_tensor *attn_norm, *wq, *wk, *wv, *wo, *q_norm,
        *k_norm, *ffn_norm, *gate_inp;
    /* sandwich norms of attention's and the FFN's outputs, and the layer's
       output scale (1 if none) */
    const struct janas_jns_tensor *post_attn, *post_ffn;
    float out_scale;
    /* Gated DeltaNet */
    const struct janas_jns_tensor *ssm_in, *ssm_ba, *conv, *dt_bias, *ssm_a,
        *ssm_norm, *ssm_out;
    /* qwen35: q|k|v (conv_ch rows), z (d_inner rows), alpha and beta
       (n_vh rows each, as f32: the file's, or made from its Q8_0) */
    const struct janas_jns_tensor *ssm_qkv, *ssm_z;
    const float *ssm_alpha, *ssm_beta;
    float *alpha_own, *beta_own;
    /* shared expert: gate, up, down as matrices of a slot at offset 0 */
    const uint8_t *sh_w[3];
    struct janas_jns_matrix sh_mx[3];
    float *sh_gate;           /* d_model weights of the sigmoid gate, as f32 */
    const uint16_t *router16; /* router weights, f16 (in place of the f32) */
    uint16_t *router_own;     /* the zero router of a dense layer, if any */
    float *conv_t;            /* conv weights transposed: d_conv x conv_ch */
};

_Static_assert(JANAS_ATTN_MAX_BLOCK == JANAS_LLM_MAX_BLOCK, "block size");

struct janas_llm_model {
    struct janas_jns j;
    uint8_t *resident; /* the resident region, read at load */
    struct janas_pool *compute, *io;
    int own_compute; /* zero when the pool came from the caller */
    /* compute threads: the pool lists one CPU per performance core, their
       other threads, then the efficiency cores; the tuner picks a prefix
       (and whether the GPU helps) for each kind of pass */
    int p_cores, p_threads, all_threads;
    struct janas_tuner tuner;
    char tune_key[768];
    int gpu_use;           /* the current pass may use the GPU */
    struct janas_gpu *gpu; /* optional, for products over blocks */
    int resident_gpu;      /* the resident region belongs to the GPU */
    int gpu_off;           /* weights not shared with it: CPU only */
    struct janas_expert_cache *cache;

    const struct janas_arch *a; /* what this architecture has (arch.h) */
    uint32_t n_layer, d_model, n_head, n_head_kv, head_dim, n_rot, n_expert, k,
        n_vocab, n_ctx, d_ff, n_attn, n_rec;
    float eps, rope_base;
    /* Gemma 4: the sliding-window layers' geometry (head size and RoPE in
       head_swa, n_rot_swa, rope_base_swa; n_head_kv, head_dim and n_rot
       are the full layers'), the window, the full layers' RoPE factors
       (theta / factor per pair, NULL: none), the logits' soft cap (0:
       none), the widest head and KV row of any layer */
    uint32_t head_swa, n_rot_swa, n_kv_swa, window, hd_max, kvd_max;
    float rope_base_swa, softcap;
    const float *rope_freqs;
    /* the KV cache of slot s (attention layer, then the MTP block's) starts
       at kv_off[s] bytes and its scales at ks_off[s] floats; kv_hd and
       kv_heads its geometry. Uniform unless the layers differ (Gemma 4) */
    size_t *kv_off, *ks_off;
    uint32_t *kv_hd, *kv_heads;
    /* Gated DeltaNet geometry: ds = head size (keys and values), n_kh key
       heads, n_vh value heads (n_vh / n_kh per key head), conv over the
       q|k|v channels */
    uint32_t d_conv, ds, n_kh, n_vh, d_inner, conv_ch, qkvz_dim;
    int avx2; /* AVX2 and FMA: the vector kernels of this file */
    int f16c;
    int trace; /* JANAS_TRACE=1: print the state after every layer */
    /* which key head a value head shares: qwen3next repeats each key head
       for its consecutive value heads (h / rf), qwen35moe cycles through
       them (h % n_kh); JANAS_VMAP=div|mod overrides */
    int vmap_mod;
    struct layer *layers;
    const struct janas_jns_tensor *tok_embd, *out_norm, *output;

    /*
     * n_attn x n_head_kv x n_ctx x head_dim: the positions of one KV head
     * are contiguous, so attention streams them. Interleaving the heads of a
     * position (a 2 KiB stride between the rows one head reads) defeated the
     * hardware prefetcher, which does not cross 4 KiB pages: every position
     * cost a DRAM access and attention took 6.5 ms per token instead of ~1.
     * Half precision: attention at long context is bound by reading it.
     */
    int8_t *kcache, *vcache;
    float *kscale, *vscale; /* one per position and KV head */
    /*
     * Recurrent state in two buffers, a base and a tip, so that rolling back
     * costs no memory traffic. Each holds, per DeltaNet layer, the state
     * (n_vh x ds x ds, deltanet.h layout) and the last d_conv - 1 inputs of
     * every channel ((d_conv - 1) x conv_ch). The base holds positions
     * 0 .. base_pos - 1; the tip the log_n positions after them, run by the
     * last call, whose per-token inputs (conv inputs, decays, betas) are in
     * the log.
     *
     * A call starting where the last one ended (plain decoding, prefill, a
     * fully accepted draft) starts from the tip, which becomes the base. A
     * call starting earlier (rejected draft tokens) replays the accepted
     * logged tokens on the base in place. Either way the call's tokens leave
     * their state in the other buffer: the state is read and written once
     * per call, as with an in-place update.
     */
    /* two buffers each, back to back. The DeltaNet state, read and written
       in full for every call, in half precision: half the traffic; its
       rounding stays within the model's own numerical noise (an int16
       format with row scales, 7x more precise, changed nothing measurable) */
    uint16_t *ssm_buf;
    float *conv_buf;
    float *log_mixed, *log_beta, *log_decay;
    int base; /* buffer holding the base */
    int src;  /* this call: buffer the tokens start from */
    uint32_t base_pos, log_n, replay;
    uint32_t n_ctx_train; /* what the model was trained for, 0 if unsaid */

    /*
     * scratch for a pass of up to blk tokens (janas_llm_model_max_block):
     * a prefill block, larger than JANAS_LLM_MAX_BLOCK so that an expert is
     * read once for many tokens; attention takes it 64 at a time
     */
    uint32_t blk;
    uint32_t *tok_first, *tok_fill, *tok_list; /* moe_block's per token rows */
    float *x, *xn, *q, *kk, *vv, *att, *router, *rope_cos, *rope_sin;
    struct janas_attn *attn;
    /* Gemma 4: the sliding-window layers' RoPE table and attention, and the
       residual kept aside while the FFN's output is normalized */
    float *rope_cos_swa, *rope_sin_swa, *xres;
    struct janas_attn *attn_swa;

    /*
     * Multi-token prediction block (optional, a second JNS file): the
     * embedding of token t_p and the final hidden state h_{p-1} of the main
     * model, each RMS-normalized, concatenated and projected, then one
     * decoder layer (attention with its own KV slot, routed and shared
     * experts through their own cache), a norm and the main LM head: a
     * guess at t_{p+1}. hid holds the main model's hidden states of the last
     * call, mtp_hid the MTP's own (to chain guesses).
     */
    struct janas_jns mtp_j;
    uint8_t *mtp_res;
    struct layer mtp_layer;
    struct janas_expert_cache *mtp_cache;
    int mtp;           /* an MTP block: its own file, or in the main one */
    int mtp_own_cache; /* mtp_cache belongs to the MTP file */
    const struct janas_jns_layer *mtp_jl; /* its experts: layer table entry */
    uint32_t mtp_cl;                      /* and layer index in mtp_cache */
    const struct janas_jns_tensor *eh_proj, *enorm, *hnorm, *mtp_norm;
    float *hid, *mtp_hid, *cat;
    struct janas_block_q8k *catq;
    int mtp_open;
    /*
     * Draft head: the rows of the LM head for a subset of the vocabulary,
     * copied into one matrix. A draft only has to be likely, not exact (the
     * main model checks it), so it is chosen among the dh_base lowest token
     * ids (byte-pair merges in order: the most frequent tokens) plus every
     * token seen in the context, at a fraction of the full head's reads.
     */
    uint8_t *dh_w, *dh_in; /* rows; per token id: 1 if in the head */
    int32_t *dh_ids;
    uint32_t dh_n, dh_cap;
    size_t dh_row;
    float *dh_logits;
    float *big, *qgate, *ba, *mixed, *conv_out, *beta, *decay;
    struct janas_block_q8k *xq, *attq;
    /* routed experts: one pair per (token, chosen expert), sorted by expert,
       then one pair per token for the shared expert */
    uint32_t *pair_tok, *pair_exp, *order, *count, *glist;
    float *pair_w, *gate, *up, *h, *dout;
    struct janas_block_q8k *xg, *hq;
    uint32_t k_use; /* experts used per token, at most k (see set_experts) */
    int exp_level;  /* planes of the experts' down matrix read: 1, 2 or 3 */
    int exp_planes; /* some layer keeps its down matrix in planes */
    int exp_auto;   /* the level was chosen from the memory available */
    int k_auto;     /* and so was the number of experts per token */
    int warm_use;   /* preload the most used experts, and count them */
    uint32_t *use_count;   /* experts used this session, n_layer * n_expert */
    uint64_t warm_experts; /* experts preloaded at open from the profile */
    double warm_seconds;

    double phase[JANAS_PH_COUNT];
};

/* Shared by the files of the forward pass (model*.c). */

static inline double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* The bytes of a resident tensor, from the main file or the MTP file. */
static inline const void *data(const struct janas_llm_model *m,
                               const struct janas_jns_tensor *t)
{
    if (m->mtp_res && t >= m->mtp_j.tensors &&
        t < m->mtp_j.tensors + m->mtp_j.h.n_tensor)
        return m->mtp_res + (t->offset - m->mtp_j.h.resident_offset);
    return m->resident + (t->offset - m->j.h.resident_offset);
}

static inline const float *f32(const struct janas_llm_model *m,
                               const struct janas_jns_tensor *t)
{
    return data(m, t);
}

/* The tensors of layer l of file j (attention or DeltaNet, then the FFN). */
/* JANAS_TRACE=2: a checkpoint of one row, to compare with another
   implementation, layer by layer. */
static inline void trace_row(const struct janas_llm_model *m, const char *label,
                             const float *v, uint32_t dim)
{
    if (m->trace < 2)
        return;
    double nn = 0;
    for (uint32_t i = 0; i < dim; i++)
        nn += (double)v[i] * v[i];
    fprintf(stderr, "  %-14s dim %5u norm %10.4f  %9.4f %9.4f %9.4f %9.4f\n",
            label, dim, sqrt(nn), v[0], v[1], v[2], v[3]);
}

static inline void rms_norm(float *out, const float *x, const float *w,
                            uint32_t n, float eps)
{
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++)
        ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (uint32_t i = 0; i < n; i++)
        out[i] = x[i] * scale * w[i];
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/*
 * The per-token work of a block, spread over the pool: every token (or every
 * head of every token) is done by one thread with the same operations in the
 * same order as on its own, so the results are the same bits. A single
 * token stays on the caller: there is nothing to spread.
 */
struct tok_job {
    struct janas_llm_model *m;
    const struct layer *ly;
    const float *w; /* the norm's weights */
    uint32_t first, n, pos0;
    float *kc_scale, *vc_scale;
    int8_t *kc, *vc;
    float *out; /* rows the job rewrites (the soft-capped logits) */
};

void janas_m_init_tuner(struct janas_llm_model *m);
void janas_m_apply_candidate(struct janas_llm_model *m, int cand);
uint32_t *janas_m_profile_read(const struct janas_llm_model *m, size_t *used);
void janas_m_profile_write(const struct janas_llm_model *m);
void janas_m_norm_quant(struct janas_llm_model *m, const float *w,
                        uint32_t first, uint32_t n);
void janas_m_residual(struct janas_llm_model *m, uint32_t n);
void janas_m_post_norm(struct janas_llm_model *m, const float *w, uint32_t n);
void janas_m_attn_layer(struct janas_llm_model *m, const struct layer *ly,
                        uint32_t n, uint32_t pos0);
void janas_m_rec_layer(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t n);
int janas_m_exp_tracing(void);
void janas_m_exp_report(void);
int janas_m_moe_block(struct janas_llm_model *m, const struct layer *ly,
                      struct janas_expert_cache *cache,
                      const struct janas_jns_layer *jl, uint32_t l, uint32_t n);

#endif
