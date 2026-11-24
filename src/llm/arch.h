/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * arch.h - what sets one model architecture apart from another, as data.
 *
 * The forward pass is written once; each architecture says which of its
 * parts it has. A new family is a row of the table in arch.c, plus the code
 * of whatever it has that no other family had.
 */
#ifndef JANAS_LLM_ARCH_H
#define JANAS_LLM_ARCH_H

/* The recurrent layers, if any: Gated DeltaNet in two layouts. */
enum janas_rec {
    JANAS_REC_NONE,  /* attention in every layer */
    JANAS_REC_NEXT,  /* qwen3next: q|k|v|z and b|a from two fused matrices */
    JANAS_REC_Q35,   /* qwen35: q|k|v, z, alpha, beta from four */
    JANAS_REC_MAMBA2 /* nemotron_h_moe: Mamba-2 (model_mamba.c) */
};

struct janas_arch {
    const char *name;   /* general.architecture */
    enum janas_rec rec; /* DeltaNet layers, full_attention_interval apart */
    int attn_gate;      /* the query projection carries an output gate */
    int shared_expert;  /* a shared expert with a sigmoid gate, per layer */
    int post_attn_norm; /* the FFN norm is post_attention_norm, not ffn_norm */
    int vmap_mod;       /* value head h shares key head h % n_kh, else h / rf */
    int nextn_in_file;  /* the file's last layers are the MTP block */
    /* Gemma 4 */
    int gelu;     /* the feed-forward's activation is GELU (tanh), not SiLU */
    int sandwich; /* attention and FFN outputs normalized before the
                     residual (post_attention_norm, post_ffw_norm), the
                     embedding scaled by sqrt(d_model), each layer's output
                     by layer_output_scale if the file has it */
    int swa;      /* sliding-window layers beside full ones, each kind with
                     its head size, KV heads and RoPE; V normalized without
                     weights, V = K where attn_v is missing, scores not
                     scaled (the q and k norms do it) */
    int kv16;     /* keys and values kept at sixteen bits, not eight */
    /* Llama 3: no norm on the heads of q and k; the file's queries and keys
       in the checkpoint's row order (gguf2jns), rotated by halves */
    int no_qk_norm;
    /* Qwen3.8-Flash-Next (qwen4exp, model_hc.c): hyper-connections - the
       residual in several streams, mixed into each sublayer's input by a
       gate of low rank, in place of its norm, and each sublayer's output
       added to every stream with a weight of its own - n-gram embeddings
       at the layers the file names, the DeltaNet's output gated by a
       sigmoid instead of SiLU, and the attention's sparse selection */
    int hc;
    /* Nemotron-H (nemotron_h_moe, model_nh.c): each layer one part, its
       norm and residual of its own - a Mamba-2 mixer, an attention without
       RoPE or a mixture - the mixture's experts in a latent space (down
       from d_model before them, up after), relu^2 without a gate, routed
       by sigmoids with a bias for the choice */
    int nh;
    /* DeepSeek-V4 (deepseek4, model_ds4.c): four streams of the residual
       mixed by Sinkhorn's hyper-connections, attention over one latent a
       position (a window of raw ones and compressed ones, an indexer
       choosing among them), experts chosen by sqrt(softplus) scores or by a
       table of token ids, SwiGLU clamped */
    int ds4;
    /* Inkling-2-Mini (inkling2_loop, model_ink.c): 24 layers run 48 times
       (6-17 three times, messages between the passes), depth mixes of
       block states before every sublayer, attention with no RoPE (a
       position bias of the query's own, sinks, an output gate), short
       convs, experts 4 tanh(g/4) sigmoid(g) 25 tanh(u/25) */
    int ink;
};

/* The architecture called name, or NULL if Janas does not know it. */
const struct janas_arch *janas_arch_find(const char *name, unsigned len);

#endif
