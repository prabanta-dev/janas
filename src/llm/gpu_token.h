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
    int8_t *kc, *vc; /* the layer's KV cache: n_kv x cap x hd bytes */
    float *ks, *vs;  /* its scales: n_kv x cap */
    /* a Gated DeltaNet layer (qwen3.5) in place of the attention: q|k|v and
       z products, beta then alpha (2 nv x dm, f32), the convolution
       (d_conv x conv_ch, transposed), dt and A (nv), the heads' norm (ds),
       the output product; its state slot */
    int rec;
    struct janas_matvec_task ssm_qkv, ssm_z, ssm_out;
    const float *ssm_beta, *ssm_alpha, *conv_w, *dt, *ssm_a, *ssm_norm;
    uint32_t slot;
};

struct janas_gpu_tok_model {
    uint32_t n_layer, dm, n_head, n_kv, hd, n_rot, ff, n_vocab, cap;
    float eps, attn_scale;
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
};

struct janas_gpu_tok;

/* Buffers and shaders for a model of that shape; NULL (the GPU cannot, or
   out of memory) with the reason in err. */
struct janas_gpu_tok *janas_gpu_tok_create(struct janas_gpu *g,
                                           const struct janas_gpu_tok_model *md,
                                           char *err, size_t err_len);
void janas_gpu_tok_destroy(struct janas_gpu *g, struct janas_gpu_tok *t);

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
 * half src and write it to half dst (the same for a block's later pieces).
 * 0, or -1 with nothing changed but the cache rows pos .. pos + n - 1 (the
 * caller runs the block on the CPU then).
 */
int janas_gpu_tok_run(struct janas_gpu *g, struct janas_gpu_tok *t,
                      const struct janas_gpu_tok_model *md, const float *x,
                      const float *cs, const float *sn, uint32_t pos,
                      uint32_t n, int all, float *logits, float *hidden,
                      uint32_t src, uint32_t dst);

/* Whether the GPU reads the host's memory in place (an integrated GPU with
   the host-memory import, not JANAS_GPU_COPY): what the token needs. */
int janas_gpu_shares_host(const struct janas_gpu *g);

#endif
