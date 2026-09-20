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
    JANAS_REC_NONE, /* attention in every layer */
    JANAS_REC_NEXT, /* qwen3next: q|k|v|z and b|a from two fused matrices */
    JANAS_REC_Q35   /* qwen35: q|k|v, z, alpha, beta from four */
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
};

/* The architecture called name, or NULL if Janas does not know it. */
const struct janas_arch *janas_arch_find(const char *name, unsigned len);

#endif
