/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * gpu_token.h - a whole token on the GPU, with one submission: every layer's
 * norms, products, RoPE, KV cache and attention, the feed-forward and the
 * output head recorded in one command buffer, the logits back at the end
 * (private to the library).
 *
 * For now the dense attention models (qwen3's shape: q and k norms, NeoX
 * RoPE, an eight-bit KV cache, SiLU feed-forward), and qwen3.5's (Gated
 * DeltaNet layers between them, the query with an output gate) for blocks,
 * both with a mixture of experts too (qwen3moe, qwen35moe: a submission a
 * layer, the CPU sorting the router's choice in between),
 * on a GPU that shares the host's memory: the weights, the experts' arena and
 * the KV cache are the CPU's own memory, imported, so the CPU's passes (a
 * prompt in blocks) and the GPU's tokens read and write the same cache. The
 * products are the CPU's bit for bit (their shaders); the norms and the
 * attention sum in the GPU's order.
 */
#ifndef JANAS_LLM_GPU_TOKEN_H
#define JANAS_LLM_GPU_TOKEN_H

#include <stdint.h>

#include "gpu.h"
#include "matvec.h"

/* One layer: host pointers, all in regions the GPU holds. The products'
   tasks give type, weights (and planes), rows and cols; their x, y and
   n_vec are the GPU's own. */
struct janas_gpu_tok_layer {
    const float *attn_norm, *q_norm, *k_norm, *ffn_norm;
    struct janas_matvec_task wq, wk, wv, wo, gate, up, down;
    /* the attention's geometry, the model's or the layer's kind's (Gemma
       4: sliding-window layers and full ones): head size, KV heads, RoPE
       dimensions, the positions its cache holds (a ring of them where
       ring: position p in row p mod cap), the window a token sees (0: all
       of them); wv's type 0: V is K before its norm (Gemma 4's full
       layers). Gemma 4's sandwich norms of the attention's and the
       feed-forward's outputs (NULL: none) and the layer's output scale */
    uint32_t hd, n_kv, n_rot, cap, window;
    int ring, swa;
    const float *post_attn, *post_ffn;
    float out_scale;
    /* Gemma 4's per-layer embeddings (E2B, E4B; NULL: none): the branch's
       gate (ple_dim x dm) and projection (dm x ple_dim), f32, its norm; the
       layer's keys and values another's (n_kv_layers on: wk's type 0) */
    const float *ple_gate, *ple_proj, *ple_post;
    /* Gemma 4's mixture (26B; pre_norm2 NULL: none): its dense branch (the
       shared expert, weight 1, no gate: sh_gate_w's type 0), normalized by
       post_norm1; the experts' input by pre_norm2, the router's the
       residual without weights times rscale / sqrt(dm), each expert's
       output scale, the experts' sum normalized by post_norm2 */
    struct janas_matvec_task dn_gate, dn_up, dn_down;
    const float *pre_norm2, *post_norm1, *post_norm2, *rscale, *exp_scale;
    int8_t *kc, *vc; /* the layer's KV cache: n_kv x cap x hd bytes */
    float *ks, *vs;  /* its scales: n_kv x cap */
    /* a Gated DeltaNet layer (qwen3.5) in place of the attention: q|k|v and
       z products, beta then alpha (2 nv x dm, f32), the convolution
       (d_conv x conv_ch, transposed), dt and A (nv), the heads' norm (ds),
       the output product; its state slot (an attention layer's: its KV
       slot, its row of att_in) */
    int rec;
    struct janas_matvec_task ssm_qkv, ssm_z, ssm_out;
    const float *ssm_beta, *ssm_alpha, *conv_w, *dt, *ssm_a, *ssm_norm;
    uint32_t slot;
    /* a mixture's layer (the model's n_expert > 1): the router (n_expert x
       dm, f16) and the shared expert's gate (dm floats, or NULL); gate, up
       and down the experts' types and shapes, with no weights - those come
       from the model's slots (janas_gpu_tok_model fetch): off, each
       matrix's bytes into a slot, poff the down matrix's two bare planes'
       (Q6_K_P: 0, a plane the cache does not hold); sh_gate_w, sh_up, sh_down
       the shared expert (type 0: none) */
    const uint16_t *router16;
    const float *sh_gate;
    size_t off[3], poff[2];
    struct janas_matvec_task sh_gate_w, sh_up, sh_down;
    /* qwen4exp (the model's hc): its mixes of the streams in place of the
       norms, [0] before the token mixer and [1] before the feed-forward:
       each stream's norm (hc x dm), down (rank x hc dm), up (hc dm x rank)
       and inject (hc x hc dm), bf16; on its n-gram layer the key product
       (hc dm x dm), the value (dm x dm, bf16), the key's, query's and
       conv's norms (hc dm each) and the conv's taps (hc dm x kernel,
       channel-major) */
    const float *hc_norm[2];
    const uint16_t *hc_down[2], *hc_up[2], *hc_inject[2];
    struct janas_matvec_task ng_key;
    const uint16_t *ng_value;
    const float *ng_nk, *ng_nq, *ng_nc, *ng_conv;
    /* qwen4exp's indexer on an attention layer (the model's ix_heads; or
       NULL): its query and key products (ix_heads ix_dim x dm and ix_dim x
       dm, bf16), their norms (ix_dim), the raw keys of every position
       (cap x ix_dim, half precision) and the pooled keys of every block
       (cap / ix_ratio x ix_dim), the model's: the CPU reads and writes
       them too */
    const uint16_t *ix_q, *ix_k;
    const float *ix_qn, *ix_kn;
    uint16_t *ix_raw;
    float *ix_pool;
};

struct janas_gpu_tok_model {
    /* n_kv, hd and n_rot the largest of any layer's, cap the full
       layers' */
    uint32_t n_layer, dm, n_head, n_kv, hd, n_rot, ff, n_vocab, cap;
    float eps, attn_scale;
    /* Gemma 4: GELU in place of SiLU, keys and values at sixteen bits (q
       left in float for them), V normalized without weights; the
       sliding-window layers' RoPE dimensions and their tables (n_rot_swa
       / 2 cos and sin a token, set before each run), the most tokens a
       block the rings leave room for (0: no ring) */
    int gelu, kv16, vnorm;
    uint32_t n_rot_swa, ring_room;
    const float *cs_swa, *sn_swa;
    /* Gemma 4's per-layer embeddings: ple_dim numbers a layer and token
       (0: none), the block's from the CPU (janas_m_ple_inputs, n_layer
       ple_dim a token), set before each run */
    uint32_t ple_dim;
    const float *ple_in;
    const struct janas_gpu_tok_layer *layers;
    const float *out_norm;
    struct janas_matvec_task output;
    /* for blocks of more than one token, where layers and output hold the
       GPU's single-vector copies (JANAS_GPU_REPACK): the weights as they
       are; NULL: layers and output */
    const struct janas_gpu_tok_layer *blayers;
    struct janas_matvec_task boutput;
    /* qwen3.5: wq's rows head by head, each head's q then its output gate;
       the recurrent layers' shapes and their states, two halves each (the
       model's double buffer): n_rec x nv x ds x ds sixteen-bit numbers and
       n_rec x (d_conv - 1) x conv_ch floats */
    int gated_q;
    uint32_t n_rec, ds, nv, nkh, d_conv, conv_ch, d_inner;
    int vmap_mod;
    uint16_t *ssm_state;
    float *conv_state;
    /* their log (the model's, model_impl.h): per recurrent layer log_rows
       rows of conv_ch floats (the convolution's inputs) and of nv floats
       (betas, decays), what a call replays of the last one when that one's
       last tokens were taken back (rejected drafts); NULL: blocks short
       enough to be logged stay on the CPU */
    float *log_mixed, *log_beta, *log_decay;
    uint32_t log_rows;
    /* a mixture (n_expert > 1, 0 for a dense model): k experts a token, of
       eff rows each; ff above is the shared expert's (eff without one).
       Its layers run one at a time, the CPU between two of them: the
       router's choice read, its experts' slots asked of fetch (0, or -1:
       the block goes to the CPU), the pairs given to routed (or NULL) */
    uint32_t n_expert, k, eff;
    int (*fetch)(void *ctx, uint32_t layer, const uint32_t *ids, size_t n,
                 const uint8_t **slots);
    void (*routed)(void *ctx, uint32_t layer, uint32_t n, uint32_t k,
                   const uint32_t *ids);
    /* the experts just copied into the card's tier, which the cache may
       give up first (or NULL): the card's memory added to the machine's */
    void (*demote)(void *ctx, uint32_t layer, const uint32_t *ids, size_t n);
    /* reads every expert of the layer that is not in the cache, while the
       caller goes on (or NULL): a block of many tokens chooses nearly all
       of them, so none is read for nothing */
    void (*prefetch)(void *ctx, uint32_t layer);
    /* whether routed must see each layer's choice (or NULL): then a layer
       at a time even when the card holds every expert */
    int (*watched)(void *ctx);
    /* a card whose memory does not hold all the experts (tier_bytes > 0):
       a tier of slots of slot_bytes in its memory keeps those used most,
       read into it from the slots fetch gives; the others computed by
       cpu_rows, layer's experts ids[0..nd) of slots[], their rows
       [ra[i], rb[i]) of out (dm floats each, before their weights), row r
       a pair of token tok[r], from xn (n tokens of dm floats, the
       feed-forward's normalized input): 0, or -1 (the block goes to the
       CPU) */
    size_t tier_bytes, slot_bytes;
    int (*cpu_rows)(void *ctx, uint32_t layer, uint32_t n, const float *xn,
                    uint32_t nd, const uint32_t *ids,
                    const uint8_t *const *slots, const uint32_t *ra,
                    const uint32_t *rb, const uint32_t *tok, float *out);
    void *ctx;
    /* qwen4exp: hc streams of the residual (0: one, the others), the
       mixes' rank, the output's mix (no inject); ng_layer the n-gram
       layer (n_layer: none), its conv's kernel and dilation, its ring of
       inputs (ng_cap positions of hc dm floats, the model's: the CPU's
       passes read and write it too) and the block's embeddings, which the
       caller writes before each run (n x dm); att_in each attention
       layer's input as the GPU made it (n_attn x the block's n x dm), for
       the CPU's indexer keys (model_qsa.c) after the run, or NULL. The
       feed-forwards' widths (eff, ff) may end inside a group of 256: then
       their buffers are padded to whole groups of zeros. */
    uint32_t hc, hc_rank;
    /* the mixes' down and up rewritten as Q8_0 (JANAS_HC_MIX=q8: blocks
       of 32, rows of a whole number of them), their injects still bf16 */
    int hc_q8;
    const float *hco_norm;
    const uint16_t *hco_down, *hco_up;
    uint32_t ng_layer, ng_kernel, ng_len, ng_cap;
    float *ng_hist;
    const float *ng_emb;
    float *att_in;
    /* qwen4exp's sparse attention (0 heads: none): the indexer's heads,
       their width, the positions a token keeps past the dense ones and the
       blocks' length; ix_rope the cos and sin (n_rot floats a row) of the
       block's n tokens' positions, then of the first position of each
       block from pos / ix_ratio on (n / ix_ratio + 2 rows), which the
       caller writes before each run as model_qsa.c turns them; it needs
       att_in */
    uint32_t ix_heads, ix_dim, ix_top, ix_ratio;
    const float *ix_rope;
};

struct janas_gpu_tok;

/* The most tokens a run takes (janas_gpu_tok_max_block's), and the rows a
   layer has in att_in. */
#define JANAS_GPU_TOK_BLOCK 256

/* The most tokens of the last call a block replays before its own (see
   janas_gpu_tok_run): a chain of drafts is shorter. */
#define JANAS_GPU_TOK_MAX_REPLAY 16

/* Buffers and shaders for a model of that shape; NULL (the GPU cannot, or
   out of memory) with the reason in err. */
struct janas_gpu_tok *janas_gpu_tok_create(struct janas_gpu *g,
                                           const struct janas_gpu_tok_model *md,
                                           char *err, size_t err_len);
void janas_gpu_tok_destroy(struct janas_gpu *g, struct janas_gpu_tok *t);
/* What the whole token did on the card, for a report, into buf (len
   bytes, one line a fact, each ending in a newline): the tier of experts
   and how the experts asked for were found; with JANAS_GPU_TOKEN_PROFILE
   the time a pass took and its phases. Empty when there is nothing to
   say. */
void janas_gpu_tok_report(const struct janas_gpu_tok *t, char *buf, size_t len);

/* The most tokens janas_gpu_tok_run takes at once (0 without t). */
uint32_t janas_gpu_tok_max_block(const struct janas_gpu_tok *t);
/* ... starting at position pos: fewer as the context grows (the
   attention's runs take room). */
uint32_t janas_gpu_tok_block_at(const struct janas_gpu_tok *t,
                                const struct janas_gpu_tok_model *md,
                                uint32_t pos);

/*
 * A block of n tokens (1 .. janas_gpu_tok_max_block) whose embeddings are x
 * (n x dm floats), token j at position pos + j (its keys and values go to
 * row pos + j of the cache; positions 0 .. pos + j attended), cs and sn
 * their RoPE tables (n x n_rot / 2 each): the logits (n_vocab each) of
 * the last token (all 0), of every token (1) or of none (2) into logits;
 * with hidden, the final hidden states (n x dm: the last norm's output, an
 * MTP block's input) there. The recurrent layers read their state from
 * half src and write it to half dst (the same for a block's later pieces);
 * first they replay the first replay tokens (at most
 * JANAS_GPU_TOK_MAX_REPLAY) of the log on half src in place, as the CPU
 * does when a call starts before the end of the last one (src is then the
 * base); with log_row other than UINT32_MAX they log the block's tokens
 * from that row of the log on.
 * 0; -1 with nothing changed but the cache rows pos .. pos + n - 1 (the
 * caller runs the block on the CPU then); -2 the GPU failed after it had
 * started, with the state no longer what it was.
 */
int janas_gpu_tok_run(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md, const float *x,
                      const float *cs, const float *sn, uint32_t pos,
                      uint32_t n, int all, float *logits, float *hidden,
                      uint32_t src, uint32_t dst, uint32_t replay,
                      uint32_t log_row);

#endif
