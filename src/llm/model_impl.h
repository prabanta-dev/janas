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
#include "observe.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

#include "arch.h"
#include "attention.h"
#include "common/pool.h"
#include "gpu.h"
#include "gpu_token.h"
#include "jns.h"
#include "jmath.h"
#include "quant.h"
#include "tuner.h"

#define MAX_K 64
#define MAX_CONV 8
#define MAX_SSM_DIM 256
#define SSM_ROWS 32           /* state rows per recurrence work item */
#define JANAS_NG_MAX_LEN 4    /* qwen4exp's n-grams: the longest */
#define JANAS_NG_MAX_HEADS 32 /* and their heads, all lengths together */
#define JANAS_QSA_MAX 8192    /* qwen4exp's sparse attention: positions kept */

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
    /* Gemma 4's mixture: the dense feed-forward (the shared expert, weight
       1, no gate) and the experts are two branches, each normalized on the
       way in and out; the router reads its own input, scaled per dimension,
       and each expert's output has a scale of its own */
    const struct janas_jns_tensor *pre_norm2, *post_norm1, *post_norm2;
    const float *router_scale, *exp_scale;
    /* Gemma 4's per-layer embeddings (E2B, E4B): after the FFN, the layer's
       own input of ple_dim numbers per token gates a branch of the
       residual, gelu(gate x) * input, projected back, normalized, added */
    const struct janas_jns_tensor *ple_gate, *ple_proj, *ple_post;
    /* Gated DeltaNet */
    const struct janas_jns_tensor *ssm_in, *ssm_ba, *conv, *dt_bias, *ssm_a,
        *ssm_norm, *ssm_out;
    /* qwen35: q|k|v (conv_ch rows), z (d_inner rows), alpha and beta
       (n_vh rows each, as f32: the file's, or made from its Q8_0) */
    const struct janas_jns_tensor *ssm_qkv, *ssm_z;
    /* qwen3next's fused ssm_in as q|k|v and z apart, the same rows in
       qwen35's order, for the whole token on the GPU (in m->qkvz_split;
       NULL until it is built) */
    const uint8_t *qkv_split, *z_split;
    const float *ssm_alpha, *ssm_beta;
    float *alpha_own, *beta_own;
    /* shared expert: gate, up, down as matrices of a slot at offset 0 */
    const uint8_t *sh_w[3];
    struct janas_jns_matrix sh_mx[3];
    float *sh_gate;           /* d_model weights of the sigmoid gate, as f32 */
    const uint16_t *router16; /* router weights, f16 (in place of the f32) */
    uint16_t *router_own;     /* the zero router of a dense layer, if any */
    float *conv_t;            /* conv weights transposed: d_conv x conv_ch */
    /* qwen4exp (model_hc.c): the hyper-connections before the token mixer
       (0) and before the feed-forward (1) - the streams' norms (f32, hc_n
       x d_model), the gate's two matrices of low rank and the inject (bf16)
       - and the n-gram embeddings' at the layer that has them: the key and
       value projections, the norms of the keys, of the streams as queries
       and of the conv's inputs, the conv's weights (kernel x channels, as
       f32) */
    const struct janas_jns_tensor *hc_norm[2], *hc_down[2], *hc_up[2],
        *hc_inject[2];
    const struct janas_jns_tensor *ng_key, *ng_value, *ng_nkey, *ng_nquery,
        *ng_nconv;
    float *ng_conv;
    /* qwen4exp's sparse attention (model_qsa.c): the indexer's query and key
       projections (bf16) and their norms */
    const struct janas_jns_tensor *ix_q, *ix_k, *ix_qn, *ix_kn;
    /* Nemotron-H (model_nh.c): what the layer is (JANAS_NH_*); a Mamba-2
       mixer's conv bias, D, and its projections' scales (the file's
       per-tensor ones of NVFP4); a mixture's experts (the router's rows),
       their choice's bias, the latent projections, its shared expert
       (up, down: relu^2, no gate) */
    int nh_kind;
    const float *conv_b, *ssm_dd;
    float in_scale, out_scale2;
    uint32_t n_exp;
    const float *exp_bias;
    const struct janas_jns_tensor *lat_down, *lat_up, *sh_up, *sh_down;
};

enum { JANAS_NH_MAMBA, JANAS_NH_ATTN, JANAS_NH_MOE };

_Static_assert(JANAS_ATTN_MAX_BLOCK == JANAS_LLM_MAX_BLOCK, "block size");

struct janas_llm_model {
    struct janas_jns j;
    uint8_t *resident; /* the resident region, read at load */
    /* its bytes [res_hole[0], res_hole[1]) never read there: the
       per-layer table, read a row at a time from the file (model_ple.c,
       model_hc.c), kept out of the GPU's imports, which would pin its
       untouched pages (28.8 GB on Qwen3.8-Flash-Next) */
    uint64_t res_hole[2];
    int ple_ram; /* that table read into memory all the same (ple_ram) */
    struct janas_pool *compute, *io;
    int own_compute; /* zero when the pool came from the caller */
    /* compute threads: the pool lists one CPU per performance core, their
       other threads, then the efficiency cores; the tuner picks a prefix
       (and whether the GPU helps) for each kind of pass */
    int p_cores, p_threads, all_threads;
    struct janas_tuner tuner;
    double last_pass_s; /* janas_llm_model_pass_seconds */
    char tune_key[768];
    int gpu_use;             /* the current pass may use the GPU */
    struct janas_gpu *gpu;   /* optional, for products over blocks */
    struct janas_m_npu *npu; /* optional (JANAS_NPU=1), model_npu.c */
    int npu_use;             /* the current pass may use the NPU */
    char npu_why[160];       /* no NPU: why (janas_llm_model_npu_why) */
    int resident_gpu;        /* the resident region belongs to the GPU */
    int gpu_off;             /* weights not shared with it: CPU only */
    /* the user keeps prompts off the GPU (--prompt-on cpu, npu): its fast
       prompt, which reads them whatever the tuner chose, does not */
    int prompts_off_gpu;
    /* this pass a block the tuner gave the CPU alone: neither the fast
       prompt nor the whole pass on the GPU (janas_m_apply_candidate) */
    int prompt_cpu;
    /* a long prompt's blocks: the GPU's whole pass against the path
       without it, measured in turn every LP_BLOCKS blocks (model.c
       long_prompt_*); lp_cpu: this block without it */
    struct {
        uint32_t band;
        int state;       /* 0: none measured, 1: one, 2: both */
        int first;       /* which went first: 0 the GPU's */
        double gpu, cpu; /* seconds a token */
    } lp;
    int lp_cpu;
    /* the user's devices for prompts, drafts, replies (common/devices.h:
       janas_devices_vars' order), as this machine has them; 0 as the
       engine chooses */
    int dev_on[3];
    /* the experts' arena on the GPU (dense models, and mixtures whose
       every expert fits the cache): 0 not yet, 1 there, -1 not possible;
       arena_stale when a slot changed since - the GPU's copy no longer
       true, the experts' products stay on the CPU */
    int arena_gpu, arena_stale;
    uint64_t arena_loads;
    int arena_in_place; /* read by the GPU where it is: never stale */
    char gpu_why[160];  /* no GPU: why (janas_llm_model_gpu_why) */
    struct janas_expert_cache *cache;

    const struct janas_arch *a; /* what this architecture has (arch.h) */
    uint32_t n_layer, d_model, n_head, n_head_kv, head_dim, n_rot, n_expert, k,
        n_vocab, n_ctx, d_ff, n_attn, n_rec;
    uint32_t ff_buf; /* rows of a pair's buffers: d_ff, or a wider shared
                        feed-forward's (Gemma 4's 26B) */
    float eps, rope_base;
    /* Gemma 4: the sliding-window layers' geometry (head size and RoPE in
       head_swa, n_rot_swa, rope_base_swa; n_head_kv, head_dim and n_rot
       are the full layers'), the window, the full layers' RoPE factors
       (theta / factor per pair, NULL: none), the logits' soft cap (0:
       none), the widest head and KV row of any layer */
    uint32_t head_swa, n_rot_swa, n_kv_swa, window, hd_max, kvd_max;
    float rope_base_swa, softcap;
    const float *rope_freqs;
    /* pictures in the sequence (model_media.c): where each begins, its
       grid of tokens and their embeddings; the multimodal RoPE's sections
       (rope.dimension_sections, Qwen3.5: 11, 11, 10; 0: none), which give
       a picture's tokens their row and column as positions; both: its
       tokens attend to one another both ways; att_last, a pass's last
       positions seen (blk) when one is */
    struct janas_media {
        uint32_t at, nx, ny;
        int both;
        float *embd;
    } *media;
    uint32_t n_media, mrope[3];
    uint32_t *att_last;
    /* Gemma 4's per-layer embeddings (0: none): ple_dim numbers per layer
       and token, from a table of their own read a row at a time from the
       file (ple_tok, never in memory) and a projection of the embedding
       (ple_model, bf16), normalized by ple_norm; ple holds the block's,
       token after token, layer after layer, ple_g a branch's rows,
       ple_row a row of the table as floats, then as read (room for twice
       its numbers in floats) */
    uint32_t ple_dim;
    const struct janas_jns_tensor *ple_tok, *ple_model, *ple_norm;
    float *ple, *ple_g, *ple_row;
    struct janas_block_q8k *ple_q;
    /* layers from n_kv_layers on keep no keys and values of their own:
       they read those of the last earlier layer of their kind */
    uint32_t n_kv_layers;
    /* the KV cache of slot s (attention layer, then the MTP block's) starts
       at kv_off[s] bytes and its scales at ks_off[s] floats; kv_hd and
       kv_heads its geometry. Uniform unless the layers differ (Gemma 4) */
    size_t *kv_off, *ks_off;
    uint32_t *kv_hd, *kv_heads;
    int kv16; /* keys and values of 16 bits (arch, or JANAS_KV=8|16) */
    /* positions a slot holds per KV head: n_ctx, or for a sliding-window
       layer a ring of fewer (kv_ring), position p in row p mod kv_cap;
       ring_hi: one past the last position written */
    uint32_t *kv_cap;
    uint8_t *kv_ring, *kv_swa; /* and whether its layer is a sliding one */
    uint32_t ring_hi;
    /* Gated DeltaNet geometry: ds = head size (keys and values), n_kh key
       heads, n_vh value heads (n_vh / n_kh per key head), conv over the
       q|k|v channels */
    uint32_t d_conv, ds, n_kh, n_vh, d_inner, conv_ch, qkvz_dim;
    /* the recurrent state of a layer, in uint16 units (each buffer of
       ssm_buf holds n_rec of them): DeltaNet's n_vh heads of ds x ds at
       sixteen bits; Mamba-2's (JANAS_REC_MAMBA2) n_vh heads of ssm_hd rows
       of ds, in float, two units each */
    size_t ssm_per;
    uint32_t ssm_hd;
    /* Nemotron-H's mixtures (model_nh.c, model_ffn.c): the latent width
       (0: none), the weights' scale, and per call the experts' input and
       where their weighted outputs go - d_model, xq and x, or the latent
       ones; lat the latent input, lat_acc the sum, latq either quantized;
       sh_h and sh_hq the shared expert's rows */
    uint32_t moe_lat, moe_w;
    float route_scale;
    const struct janas_block_q8k *moe_xq;
    float *moe_out, *lat, *lat_acc, *sh_h;
    struct janas_block_q8k *latq, *sh_hq;
    uint32_t sh_ff;
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
    size_t kv_bytes, ks_bytes; /* the caches' and the scales' allocations */
    /* the whole token on the GPU (gpu_token.h, JANAS_GPU_TOKEN=1): 0 not
       tried yet, 1 ready, -1 not for this model; loads: the experts' loads
       it was built with (a change and it is built again) */
    struct janas_gpu_tok *gtok;
    uint8_t *qkvz_split; /* the layers' qkv_split and z_split (aligned) */
    struct janas_gpu_tok_model gtok_md;
    struct janas_gpu_tok_layer *gtok_layers;
    struct janas_gpu_tok_layer *gblk_layers; /* the weights as they are,
                                                 with a GPU copy */
    int gtok_state, gtok_kv;
    uint64_t gtok_loads;
    /* on a card with memory of its own (gtok_dev): the token's state - the
       KV rows of the layers it takes, the recurrent states and their log -
       is in two places, newer where gtok_where says (JANAS_GT_*), made the
       same when the passes change hands (janas_m_gtok_host) */
    int gtok_dev, gtok_where, gtok_rec_in;
    /* passes whole on the GPU by length (1, 2-16, more), the state's
       copies there and back (JANAS_GPU_VERBOSE prints them at the end) */
    uint64_t gtok_runs[3], gtok_push, gtok_pull;
    /* qwen4exp on the GPU: the pass's tokens (its n-gram embeddings), each
       attention layer's input at bf16 as the GPU made it (n_attn x blk x
       d_model, for the indexer's keys), the ring of the n-gram conv
       imported; the indexer's raw and pooled keys imported, the cos and
       sin of a piece's positions for it (janas_gpu_tok_model ix_rope) */
    const int32_t *gtok_tokens;
    float *gtok_att_in;
    int gtok_ng_in, gtok_ix_in;
    float *gtok_ix_rope;
    /* the aligned copy of its Q6_K weights for the GPU (JANAS_GPU_REPACK),
       in the GPU driver's memory, kept for a rebuild */
    uint8_t *gtok_repack;
    size_t gtok_repack_bytes;
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
    /* the call goes on from the one before it in the same pass (the NPU's
       second half of a block): the state and the convolution's inputs read
       where that one left them, in the other buffer */
    int rec_cont;
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
    float *rope_cos_swa, *rope_sin_swa, *xres, *xbr; /* xbr: a branch */
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
    /* Gemma 4's assistant, in place of an MTP block (model_assist.c) */
    struct janas_assist *asst;
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
    const uint8_t *dh_src; /* the full head the rows come from */
    int32_t *dh_ids;
    uint32_t dh_n, dh_cap;
    size_t dh_row;
    float *dh_logits;
    /* The vocabulary of the replies (janas_llm_model_set_vocab): the head's
       rows of those tokens alone, copied as the draft head's, computed in
       place of the whole head, the other logits -inf; vs_n 0: the whole */
    uint8_t *vs_w;
    int32_t *vs_ids;
    uint32_t vs_n;
    float *vs_y; /* their logits, vs_cap vectors */
    uint32_t vs_cap;
    float *big, *qgate, *ba, *mixed, *conv_out, *beta, *decay;
    struct janas_block_q8k *xq, *attq;
    /* routed experts: one pair per (token, chosen expert), sorted by expert,
       then one pair per token for the shared expert */
    uint32_t *pair_tok, *pair_exp, *order, *count, *glist;
    float *pair_w, *gate, *up, *h, *dout;
    uint32_t *hlen; /* per pair row: the rows of its expert's gate */
    /* the Ana-Net probe (model_ana.c): its mode (0 off), the fraction of a
       pair's neurons switched off, each pair row's agreements (ff_buf floats
       a row), the layer being computed, a token's picture and where it goes */
    int ana_mode, ana_tok; /* ana_tok: the fraction is of a token's neurons */
    float ana_frac;
    float *ana_agree;
    /* each pair row's expert (n_expert: the shared one), and the norms of
       the columns of each layer's experts' down matrices, with whether
       computed yet */
    uint32_t *ana_exp;
    float *ana_coln;
    uint8_t *ana_coln_ok;
    /* JANAS_ANA_STATS: every pair's record, and the tokens done so far */
    FILE *ana_stats, *ana_xstats; /* ana_xstats: JANAS_ANA_XSTATS */
    uint64_t ana_base;
    /* the doubt probe (model_doubt.c, JANAS_DOUBT_STATS): its file, each
       pair row's features, the layers whose inputs it writes; recording
       while dbt_on, which the caller sets */
    FILE *dbt_file;
    float *dbt_feat;
    uint8_t *dbt_x;
    int dbt_on, dbt_res;
    int dbt_cut, dbt_cut_hit;
    /* the entrance gate (gate.c, model_gate.c): its readers, its layer,
       the last token's residual there when armed, and whether it came */
    struct janas_gate *egate;
    /* the monitor's look inside the layers (model_monitor.c), or NULL */
    struct janas_mwatch *watch;
    uint32_t gate_layer, gate_pos; /* the position kept (UINT32_MAX: the
                                      last of each forward call) */
    uint32_t fwd_pos0; /* the current forward call's first position */
    float *gate_x;
    int gate_armed, gate_got;
    /* the observers of the forward pass (observe.h, model_observe.c) and
       the pair rows the buffers hold */
    struct janas_observer obs[JANAS_MAX_OBSERVERS];
    uint32_t n_obs;
    size_t pairs_cap; /* the caller's: stop the forward after layer
                                 dbt_cut - 1 (0: never); set when it did */
    float *dbt_rbuf;  /* JANAS_DOUBT_RES: the last token's residual after
                         each layer, n_layer rows of d_model */
    /* JANAS_ANA_COARSE: each pair row's h from gate and up at two bits, the
       choice made on it; the neurons both choices keep, and the true one's */
    float *ana_coarse;
    uint64_t ana_both_kept, ana_true_kept;
    /* JANAS_ANA_GRADE: the fractions of a token's neurons kept at 6, 4 and 2
       bits of down, and each pair row's levels (3, 2, 1; 0 off) */
    float ana_grade[3];
    uint8_t *ana_level;
    /* JANAS_ANA_GULOW: the routed experts of least weight in a token whose
       gate and up are read at two bits, and each pair row's mark */
    int ana_gulow;
    uint8_t *ana_low;
    uint32_t ana_layer, ana_width;
    uint8_t *ana_frame;
    char ana_dir[512];
    int ana_frames, ana_frames_max;
    /* the experts' down matrix by neuron (model_neurons.c): whether a layer
       has it, whether a token's neurons are ranked into levels and the
       fractions read at 6, 4 and 2 bits, each pair row's levels, a pass's
       scores and its pair rows by token */
    int neuron_down, neuron_rank;
    float neuron_frac[3];
    uint8_t *nlevel;
    float *nscore;
    /* a token's chunks of neurons, each its sum, and its length; the pair
       rows whose h needs no eight-bit copy */
    float *npart;
    size_t npart_len;
    uint8_t *nskipq;
    uint32_t *ntok_first, *ntok_fill, *ntok_list;
    struct janas_block_q8k *xg, *hq;
    uint32_t k_use; /* experts used per token, at most k (see set_experts) */
    int exp_level;  /* planes of the experts' down matrix read: 1, 2 or 3 */
    int exp_planes; /* some layer keeps its down matrix in planes */
    int exp_auto;   /* the level was chosen from the memory available */
    int k_auto;     /* and so was the number of experts per token */
    /* an expert not in RAM that every token of the block weighs below this
       is left out, not read (see skip_misses); 0: never. And how many were */
    float miss_skip;
    uint64_t skip_asked, skip_left;
    /* the products' output buffers and the recurrent state and log, whole
       pages, shared with the GPU */
    struct janas_out_buf {
        float *p;
        size_t bytes;
    } out[40];
    int n_out;
    int warm_use;        /* preload the most used experts, and count them */
    uint32_t *use_count; /* experts used this session, n_layer * n_expert */
    uint64_t head_before, head_after; /* output head requantized (head_bits) */
    /* the head's type in the file (BF16 30, F16 1) when rewritten as Q8_0 at
       load (head16_to_q8_0), else 0; and the bytes before and after */
    int head16;
    uint64_t head16_before, head16_after;
    /* qwen4exp's mixes of the streams rewritten at load (JANAS_HC_MIX:
       "q8"; empty: as in the file), their bytes before and
       after (model_hc_quant.c) */
    char hc_quant[8];
    uint64_t hc_quant_before, hc_quant_after;
    /* the caller's CPUs before the load pinned it (a cpu_set_t), given
       back when the model is freed */
    unsigned char caller_cpus[128];
    int caller_pinned;
    janas_llm_route_fn route_fn; /* janas_llm_model_route_observer */
    void *route_ctx;
    janas_llm_route_fn pred_fn; /* janas_llm_model_route_predictor */
    void *pred_ctx;
    uint32_t pred_k;
    int pred_early; /* ahead of layer l's experts, not after layer l */
    /* reads ahead (janas_m_prefetch): the next layer's router applied
       before this layer's experts, of its first r experts at most b read
       ahead. pf_mode 0 never, 1 where it pays (pf_on, decided every
       PF_WINDOW tokens from the experts the cache would have missed), 2
       always (JANAS_PREFETCH=0, unset, =r,b) */
    uint32_t pf_rank, pf_max;
    float *ahead_x; /* janas_m_rank_ahead's input and scores */
    int pf_mode, pf_on;
    uint32_t pf_tokens;
    uint64_t pf_would;
    /* reads ahead in a prefill block (janas_m_prefetch_block): pfb_mode 0
       never, 1 after a block that missed many experts (pfb_on), 2 always
       (JANAS_PREFETCH_PREFILL=0, unset, =2) */
    int pfb_mode, pfb_on;
    uint64_t warm_experts; /* experts preloaded at open from the profile */
    double warm_seconds;

    /* qwen4exp (model_hc.c): hc_n streams of the residual (0: the usual
       one) and the rank of their gates; a block's streams (blk x hc_n x
       d_model), the same normalized and rounded to bf16, the gates' inner
       activations (blk x hc_rank) and outputs (blk x hc_n x d_model), the
       inject weights (blk x hc_n); the final mix's tensors */
    uint32_t hc_n, hc_rank;
    float *hc_res, *hc_xn, *hc_lo, *hc_gate, *hc_inj;
    const struct janas_jns_tensor *hco_norm, *hco_down, *hco_up;
    /* the inputs of the mixes' quantized matrices (an MTP block's), in Q8_K
       (blk x hc_n x d_model / 256); its MTP block's final mix */
    struct janas_block_q8k *hc_q;
    const struct janas_jns_tensor *mtp_hco_norm, *mtp_hco_down, *mtp_hco_up;
    /* its n-gram embeddings: the layer (n_layer: none), the longest n-gram,
       heads per length and a head's numbers (a row of the table), the
       conv's kernel (its dilation is the longest n-gram), the hash's
       multipliers, the heads' rows in the table, the token that cuts a
       context. Tokens and the conv's inputs kept by position in rings of
       ng_cap, ng_hi one past the last written. A block's embeddings, keys
       (later the queries' norms), values, gated values and conv inputs, a
       row as read; the table read a row at a time from the file */
    uint32_t ng_layer, ng_len, ng_per, ng_dim, ng_kernel, ng_cap, ng_hi;
    uint64_t ng_mult[JANAS_NG_MAX_LEN];
    uint32_t ng_off[JANAS_NG_MAX_HEADS], ng_vocab[JANAS_NG_MAX_HEADS];
    int32_t ng_eos;
    const struct janas_jns_tensor *ng_tab;
    int32_t *ng_tok;
    float *ng_hist, *ng_emb, *ng_kq, *ng_val, *ng_gv;
    uint8_t *ng_raw;
    struct janas_block_q8k *ng_q;
    /* its sparse attention (model_qsa.c; ix_heads 0: none): the indexer's
       heads and their size, the positions a query keeps in whole blocks
       and the block; the raw keys of every position (f16, n_attn x n_ctx x
       ix_dim), the pooled keys of every block, a block's queries and keys,
       its input at bf16, a query's blocks' scores, a selection, the
       positions chosen */
    uint32_t ix_heads, ix_dim, ix_top, ix_ratio;
    uint16_t *ix_raw;
    float *ix_pool, *ix_qs, *ix_kb, *ix_xb, *ix_score;
    uint32_t *ix_sel, *ix_pos;

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
    return 1.0f / (1.0f + janas_expf(-x));
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
    uint32_t cap, ring; /* the slot's positions per head, and whether a ring */
    float *out;         /* rows the job rewrites (the soft-capped logits) */
};

void janas_m_init_tuner(struct janas_llm_model *m);
/* model_media.c: the embedding of a picture's token at sequence position
   pos (NULL: not a picture's), the three RoPE positions of any (time, row,
   column), and the pictures forgotten */
const float *janas_m_media_row(const struct janas_llm_model *m, uint32_t pos);
void janas_m_media_pos(const struct janas_llm_model *m, uint32_t pos,
                       float p3[3]);
void janas_m_media_free(struct janas_llm_model *m);
/* 1 when a picture whose tokens attend both ways is among the n tokens
   from pos0; then janas_m_media_last fills the last position each sees */
int janas_m_media_both(const struct janas_llm_model *m, uint32_t pos0,
                       uint32_t n);
void janas_m_media_last(const struct janas_llm_model *m, uint32_t pos0,
                        uint32_t n, uint32_t *last);
/* Memory the GPU will import collapsed into 2 MB pages (Linux; see
   model_load.c), before the import pins it. */
/* The Ana-Net probe (model_ana.c, JANAS_ANA_*): set up at load (0, or -1
   out of memory), each run's agreements after its gate and up products, a
   pair row's neurons switched off before its down product, a token's
   picture written once its layers are done, freed. */
int janas_m_ana_init(struct janas_llm_model *m, size_t pairs);
void janas_m_ana_agree(struct janas_llm_model *m, const uint8_t *wg,
                       const uint8_t *wu, const struct janas_jns_matrix *mx,
                       uint32_t a, uint32_t b);
void janas_m_ana_mask(struct janas_llm_model *m, size_t g, float *h,
                      uint32_t ff);
void janas_m_ana_coln(struct janas_llm_model *m, uint32_t e,
                      const uint8_t *down, const uint8_t *p1, const uint8_t *p0,
                      const struct janas_jns_matrix *mx);
void janas_m_ana_tok(struct janas_llm_model *m, uint32_t n);
/* The doubt probe (model_doubt.c): DBT_NF features a pair row. */
#define DBT_NF 10
int janas_m_dbt_init(struct janas_llm_model *m, size_t pairs);
void janas_m_dbt_free(struct janas_llm_model *m);
/* The entrance gate's model side (model_gate.c). */
void janas_m_gate_free(struct janas_llm_model *m);
/* The observers' hook points (observe.h): one test each when none. */
void janas_m_obs_ffn_input(struct janas_llm_model *m, uint32_t l,
                           const float *xn, uint32_t n, uint32_t d);
void janas_m_obs_ffn_pair(struct janas_llm_model *m, size_t g,
                          const float *gate, const float *up, const float *h,
                          uint32_t ff);
void janas_m_obs_ffn_done(struct janas_llm_model *m, uint32_t n_rows);
void janas_m_obs_route(struct janas_llm_model *m, uint32_t l, uint32_t e,
                       int in_memory);
int janas_m_obs_layer_out(struct janas_llm_model *m, uint32_t l, const float *x,
                          uint32_t n, uint32_t d);
void janas_m_ana_lowmark(struct janas_llm_model *m, uint32_t n);
void janas_m_ana_grade(struct janas_llm_model *m, const uint8_t *down,
                       const uint8_t *p1, const uint8_t *p0,
                       const struct janas_jns_matrix *mx, uint32_t a,
                       uint32_t b);
void janas_m_ana_frame(struct janas_llm_model *m, uint32_t n);
void janas_m_ana_free(struct janas_llm_model *m);
/* The experts' down matrix by neuron (model_neurons.c): set up at load (0,
   or -1 out of memory), the levels of a pass's neurons, its experts' down
   products, freed. An expert whose down matrix is by neuron, applied to the
   pair rows [a, b): */
struct janas_neuron_run {
    const float *scale; /* each output channel's, after the sum */
    const struct janas_block_q6kp *base;
    const uint8_t *p1, *p0; /* the planes the cache holds, or NULL */
    uint32_t rows, cols;    /* neurons, outputs */
    uint32_t a, b;
};
int janas_m_neurons_init(struct janas_llm_model *m, size_t pairs);
void janas_m_neuron_levels(struct janas_llm_model *m, uint32_t ng);
void janas_m_neuron_down(struct janas_llm_model *m,
                         const struct janas_neuron_run *runs, uint32_t n);
void janas_m_neurons_free(struct janas_llm_model *m);
void janas_m_huge_pages(void *p, uint64_t bytes);
/* The longest region of memory the GPU imports (gpu.c janas_gpu_import): a
   page short of 4 GiB, which Mesa's Intel driver gave the kernel as 0 bytes
   (21 Oct 2026). */
#define JANAS_M_REGION_MAX (((uint64_t)4 << 30) - 4096)
/* The next region the GPU imports of bytes, from at: its first byte into
   *from (at, or the 2 MB boundary before it where huge: memory allocated on
   one), and its end, returned: bytes, or past at a multiple of unit (unit
   0: the farthest it may reach, for the caller to cut), never more than
   JANAS_M_REGION_MAX from *from. */
uint64_t janas_m_import_next(uint64_t at, uint64_t bytes, uint64_t unit,
                             int huge, uint64_t *from);
int janas_m_gpu_import(struct janas_llm_model *m, const uint8_t *p,
                       uint64_t bytes, uint64_t unit);
/* A buffer the products read or write, in whole pages, listed for the GPU
   to be given (model_load.c); janas_aligned_free releases it. */
float *janas_m_out_alloc(struct janas_llm_model *m, size_t bytes);
void janas_m_arena_to_gpu(struct janas_llm_model *m);
/* A block on the GPU whole (model_gpu.c): 0; -1, the CPU to run it; -2,
   the GPU failed after replaying on the base in place, the state lost. */
int janas_m_gpu_token(struct janas_llm_model *m, uint32_t pos, uint32_t n,
                      int all, float *logits);
void janas_m_apply_candidate(struct janas_llm_model *m, int cand, uint32_t nv);
/* The token's state for the CPU, on a card with memory of its own: what
   the GPU made newer copied back first; write: the CPU is about to change
   it (the GPU's copy older from then on). Nothing elsewhere. */
#define JANAS_GT_SAME 0
#define JANAS_GT_DEV 1
#define JANAS_GT_HOST 2
void janas_m_gtok_host(struct janas_llm_model *m, int write);
/* Whether the whole token is asked of the GPU (JANAS_GPU_TOKEN=1). */
int janas_m_gpu_token_asked(void);
/* model_npu.c: the NPU for the products of blocks (JANAS_NPU=1), opened
   once the resident region is read; janas_m_products runs a group of
   products as janas_gpu_matvec_group does, those the NPU takes there. */
void janas_m_npu_open(struct janas_llm_model *m);
void janas_m_npu_close(struct janas_llm_model *m);
/* Whether the NPU takes products of a pass of n tokens (its candidate has
   it, and n reaches JANAS_NPU_MIN_N). */
int janas_m_npu_takes(const struct janas_llm_model *m, uint32_t n);
/* Whether the tuner may give the NPU the blocks by itself: where its
   answers measured at par with the CPU's (janas_llm_model_npu_auto). */
int janas_m_npu_auto(const struct janas_llm_model *m);
/* The NPU kept awake through a pass of n tokens it takes products of
   (between them the CPU works long enough for it to be suspended), let
   sleep otherwise. */
void janas_m_npu_awake(struct janas_llm_model *m, uint32_t n);
/* A pass is over: a preparation of the NPU's graphs it began, too
   (janas_llm_model_npu_preparing). */
void janas_m_npu_pass_end(struct janas_llm_model *m);
/*
 * A feed-forward run (an expert, a shared expert, a dense model's) on the
 * NPU in one trip: gate, up, the activation and down, its n_vec rows of x
 * (Q8_K) in, n_vec rows of d_model floats out at dout once finished. w:
 * the slots of gate, up and down (each matrix at its mx offset in it),
 * plane: down's planes (Q6_K_P), mx their matrices. 0 with the run in flight
 * (the caller computes the others, then janas_m_npu_ffn_finish), -1 if it stays
 * on the CPU. One at a time.
 */
struct janas_m_npu_ffn {
    struct npu_entry *e;
    float *dout;
    size_t n_vec, d;
};
int janas_m_npu_ffn_begin(struct janas_llm_model *m, const uint8_t *const w[3],
                          const uint8_t *const plane[2],
                          const struct janas_jns_matrix *mx,
                          const struct janas_block_q8k *x, uint32_t n_vec,
                          float *dout, struct janas_m_npu_ffn *job);
/* Waits for it and stores its rows; -1 if the NPU failed (the run is then
   the caller's to compute). */
/*
 * y (nv rows of rows) = scale w x (nv rows of cols floats) for bf16 weights
 * in the resident region, on the NPU (its candidate has it, nv reaches
 * JANAS_NPU_MIN_N): the weights in FP16, the result close to the CPU's.
 * 0, or -1 with nothing written (the caller computes it).
 */
int janas_m_npu_dense(struct janas_llm_model *m, const uint16_t *w,
                      const float *x, float *y, uint32_t rows, uint32_t cols,
                      uint32_t nv, float scale);
int janas_m_npu_ffn_finish(struct janas_llm_model *m,
                           struct janas_m_npu_ffn *job);
/*
 * The block's first layers on the NPU, a graph each (norms, projections,
 * the FFN, the residual), the attention or the recurrence between them on
 * the CPU: for dense models laid out as Qwen3, Qwen3.5 or Gemma 4, when at
 * least nine layers in ten fit the NPU (JANAS_NPU_LAYERS=2: the first ones
 * alone, however few; 0: never). The number of layers done
 * (m->n_layer: all), m->x their output, the caller's loop going on from
 * there; -1 if not done (nothing touched), -2 if the NPU failed halfway
 * (the block cannot be finished).
 */
int janas_m_npu_layers(struct janas_llm_model *m, uint32_t n, uint32_t pos0);
/* The calibration of the NPU's INT8 activations (JANAS_NPU_A8=1): rows of
   width values (stride apart) of an input of layer l's products, at site
   0 the attention's output (wo), 1 the FFN's norm (gate, up), 2 the FFN's
   intermediate (down), 3 the norm before the attention (q, k, v); their
   channel maxima kept. Nothing outside a calibration's pass. */
void janas_m_npu_calib(struct janas_llm_model *m, int site, uint32_t l,
                       const float *x, uint32_t rows, uint32_t width,
                       size_t stride);
void janas_m_products(struct janas_llm_model *m, struct janas_gpu *g,
                      const struct janas_matvec_task *t, size_t n);
uint32_t *janas_m_profile_read(const struct janas_llm_model *m, size_t *used);
void janas_m_profile_write(const struct janas_llm_model *m);
void janas_m_norm_quant(struct janas_llm_model *m, const float *w,
                        uint32_t first, uint32_t n);
void janas_m_residual(struct janas_llm_model *m, uint32_t n);
void janas_m_post_norm(struct janas_llm_model *m, const float *w, uint32_t n);
void janas_m_attn_core(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t n, uint32_t pos0, double ta);
void janas_m_attn_layer(struct janas_llm_model *m, const struct layer *ly,
                        uint32_t n, uint32_t pos0);
void janas_m_rec_layer(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t n);
/* Its recurrence on a block whose input products are in m->big and betas
   and alphas in m->ba: the convolution, the state, the gated norm, into
   m->att. ta: when the layer began (its products counted with it). */
void janas_m_rec_core(struct janas_llm_model *m, const struct layer *ly,
                      uint32_t n, double ta);
/* Qwen3.5's betas then alphas into m->ba, from n normalized rows of xn. */
void janas_m_rec_ba(struct janas_llm_model *m, const struct layer *ly,
                    const float *xn, uint32_t n);
void janas_m_embed(struct janas_llm_model *m, int32_t token, float *x);
/* sum of exp(l[i] - mx), i < n, vectorized where the CPU can */
double janas_m_exp_sum(const float *l, uint32_t n, float mx);
/* the draft head: the first base rows of the head src (rows of row_bytes) */
int janas_m_init_draft_head(struct janas_llm_model *m, const uint8_t *src,
                            size_t row_bytes, uint32_t base);
int janas_m_assist_is(const char *path);
int janas_m_assist_load(struct janas_llm_model *m, const char *path,
                        int attn_scores, char *err, size_t err_len);
void janas_m_assist_free(struct janas_llm_model *m);
int janas_m_assist_head(struct janas_llm_model *m, uint32_t base);
int janas_m_assist_draft(struct janas_llm_model *m, int32_t token,
                         const float *h, uint32_t p, float *hnext, int32_t *out,
                         float *conf);
int janas_m_ple_inputs(struct janas_llm_model *m, const int32_t *tokens,
                       uint32_t n, uint32_t pos0);
/* y (n rows of rows) = scale w x (n rows of cols): w bf16, rows of cols,
   x already at bf16 (janas_m_round_bf16), as llama.cpp meets a bf16 matrix;
   each row and token one thread's, the same bits whatever the threads
   (model_ple.c) */
void janas_m_bf16_product(struct janas_llm_model *m, const uint16_t *w,
                          const float *x, float *y, uint32_t rows,
                          uint32_t cols, uint32_t n, float scale);
/* the same for Q8_0 weights, x kept as it is (janas_q8w_dot): the CPU's,
   the same bits whatever the threads */
void janas_m_q8w_product(struct janas_llm_model *m, const void *w,
                         const float *x, float *y, uint32_t rows, uint32_t cols,
                         uint32_t n, float scale);
/* a tensor's reserved field, in memory: its products on float inputs
   (janas_m_q8w_product), a mix rewritten at load from BF16 */
#define JANAS_TENSOR_FLOAT_IN 1
float janas_m_round_bf16(float f);
void janas_m_ple_layer(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t l, uint32_t n);
int janas_m_exp_tracing(void);
void janas_m_exp_report(void);
/* qwen4exp (model_hc.c): its tensors and buffers at load (0, or -1 with
   err), freed with the model; a block's layers on the embeddings in x,
   the final mix of the streams into xn and xq (tokens first .. n - 1) */
int janas_m_hc_load(struct janas_llm_model *m, char *err, size_t err_len);
/* layer l's tensors from file j: the main model's, or its MTP block's */
int janas_m_hc_layer_load(struct janas_llm_model *m, const struct janas_jns *j,
                          uint32_t l, struct layer *ly, char *err,
                          size_t err_len);
int janas_m_hc_alloc(struct janas_llm_model *m, char *err, size_t err_len);
/* the mixes' BF16 matrices quantized in place as JANAS_HC_MIX asks, once
   the resident region is read (model_hc_quant.c): 0, or -1 with err */
int janas_m_hc_requant(struct janas_llm_model *m, char *err, size_t err_len);
void janas_m_hc_free(struct janas_llm_model *m);
int janas_m_hc_layers(struct janas_llm_model *m, const int32_t *tokens,
                      uint32_t n, uint32_t pos0);
void janas_m_hc_head(struct janas_llm_model *m, uint32_t first, uint32_t n);
/* The MTP block of qwen4exp (janas_llm_mtp_forward): the tokens'
   embeddings in x and the rope set, hidden the trunk's residual after its
   last layer (hc_n x d_model a row, janas_llm_model_hidden); the block's
   own residual into mtp_hid, its final mix of rows first .. n - 1 into xn
   and xq. 0, or -1. */
int janas_m_hc_mtp(struct janas_llm_model *m, const float *hidden, uint32_t n,
                   uint32_t pos0, uint32_t first);
/* the n-gram layer's start for a block (the GPU's, model_gpu.c): its
   history checked, its tokens into the ring, their rows of the table into
   ng_emb (n x d_model); the caller sets ng_hi to pos0 + n once the block
   is done. 0, or -1 (the history is not there). */
int janas_m_hc_ng_begin(struct janas_llm_model *m, const int32_t *tokens,
                        uint32_t n, uint32_t pos0);
/* The cos and sin (n_rot floats a row: the cos, then the sin) of
   positions pos0 .. pos0 + n - 1, then of the first position of each block
   from pos0 / ix_ratio on (n / ix_ratio + 2 rows), as the indexer turns its
   queries and its blocks' keys: janas_gpu_tok_model's ix_rope */
void janas_m_qsa_rope(const struct janas_llm_model *m, float *out,
                      uint32_t pos0, uint32_t n);
/* the indexer's keys of n positions from pos0 of attention layer ly, from
   its input at bf16 (xb, n x d_model) - the CPU's own, after a block
   on the GPU (model_gpu.c; model_qsa.c) */
void janas_m_qsa_keys(struct janas_llm_model *m, const struct layer *ly,
                      const float *xb, uint32_t n, uint32_t pos0);
/* the n-gram rings when the context moves along (janas_llm_model_shift):
   0, or -1 if the history to move is no longer in them */
int janas_m_hc_shift(struct janas_llm_model *m, uint32_t keep, uint32_t drop,
                     uint32_t n_kv);
/* qwen4exp's sparse attention (model_qsa.c): its tensors and buffers once
   the layers are loaded (0, or -1 with err), freed; after a block's dense
   attention, the rows of the queries that see more than it keeps, again
   over the positions chosen */
int janas_m_qsa_load(struct janas_llm_model *m, char *err, size_t err_len);
void janas_m_qsa_free(struct janas_llm_model *m);
void janas_m_qsa(struct janas_llm_model *m, const struct layer *ly, uint32_t n,
                 uint32_t pos0);
/* The pooled keys of the blocks before n, from the raw keys put back (a
   state loaded) */
void janas_m_qsa_restore(struct janas_llm_model *m, uint32_t n);
/* the raw keys and the blocks when the context moves along */
void janas_m_qsa_shift(struct janas_llm_model *m, uint32_t keep, uint32_t drop,
                       uint32_t n_kv);
/* which experts a moe_block call runs */
enum { MOE_ALL, MOE_SHARED, MOE_ROUTED };

/* Nemotron-H (model_nh.c, model_mamba.c) */
int janas_m_nh_kind(struct janas_llm_model *m, uint32_t l);
int janas_m_nh_params(struct janas_llm_model *m);
int janas_m_nh_layer_load(struct janas_llm_model *m, uint32_t l,
                          struct layer *ly, char *err, size_t err_len);
int janas_m_nh_alloc(struct janas_llm_model *m);
void janas_m_nh_free(struct janas_llm_model *m);
int janas_m_nh_layers(struct janas_llm_model *m, uint32_t n, uint32_t pos0);
void janas_m_mamba_layer(struct janas_llm_model *m, const struct layer *ly,
                         uint32_t n);
/* The router of layer l applied ahead to the state x (for the predictor
   probe, janas_llm_model_route_predictor): 0 or -1. */
int janas_m_route_predict(struct janas_llm_model *m, uint32_t l, uint32_t n);
/* Layer l + 1's k likeliest experts from layer l's normalized input xn of
   a single token, its norm undone and l + 1's applied (the GPU's token,
   which keeps the state on the card, reads ahead with them): 0, or -1. */
int janas_m_rank_ahead(struct janas_llm_model *m, uint32_t l, const float *xn,
                       uint32_t k, uint32_t *ids);
/* The router of layer l applied to the state x of n tokens: its first k
   experts for each token into ids (n x k), best first. 0, or -1 when the
   layer has no router to apply ahead. Uses xn, xq and router. */
int janas_m_route_rank(struct janas_llm_model *m, uint32_t l, uint32_t n,
                       uint32_t k, uint32_t *ids);
/* Reads ahead for layer l + 1 of a single token (JANAS_PREFETCH). */
void janas_m_prefetch(struct janas_llm_model *m, uint32_t l);
/* The same for a prefill block of n tokens (JANAS_PREFETCH_PREFILL). */
void janas_m_prefetch_block(struct janas_llm_model *m, uint32_t l, uint32_t n);
/* Whether a card with memory of its own runs every pass whole
   (JANAS_GPU_WHOLE, on unless 0), and whether a mixture's experts go
   through its tier there (JANAS_GPU_TIER=0: not), model_gpu.c. */
int janas_m_gpu_whole(void);
int janas_m_gpu_tier(const struct janas_llm_model *m);
/* Experts ids[0..nd) of layer l (their slots) on the pairs of rows
   [ra[i], rb[i]) of out, row r token tok[r] of xn (n tokens of d_model
   floats, normalized), on the CPU: their down products before the
   router's weights, for the GPU's tier (gpu_token.h cpu_rows). 0, or -1. */
int janas_m_expert_rows(struct janas_llm_model *m, uint32_t l, uint32_t n,
                        const float *xn, uint32_t nd, const uint32_t *ids,
                        const uint8_t *const *slots, const uint32_t *ra,
                        const uint32_t *rb, const uint32_t *tok, float *out);
int janas_m_moe_block(struct janas_llm_model *m, const struct layer *ly,
                      struct janas_expert_cache *cache,
                      const struct janas_jns_layer *jl, uint32_t l, uint32_t n,
                      int part);

/*
 * The GPU a product may use, or NULL: the tuner's choice for this pass
 * (gpu_use); with the fast prompt the GPU too where the tuner chose the
 * CPU alone, for the fast prompt's products only (janas_gpu_set_only_fast),
 * which must always run there to give the same results. arena: a product
 * of the experts' arena, kept off the GPU while its copy there is stale.
 */
static inline struct janas_gpu *janas_m_gpu(const struct janas_llm_model *m,
                                            int arena)
{
    /* JANAS_GPU_REPEATABLE=1: the experts' products never on the GPU - the
       memory it can read of the arena changes as the cache fills, and a
       product it could read went to the fast prompt's tiles, one it could
       not to the CPU (gpu.h) */
    if (!m->gpu || m->gpu_off || (arena && m->arena_stale) ||
        (arena && janas_gpu_repeatable()))
        return NULL;
    if (m->gpu_use || (m->tuner.allow_gpu && janas_gpu_fast_prompt() &&
                       !m->prompts_off_gpu && !m->prompt_cpu))
        return m->gpu;
    return NULL;
}

#endif
