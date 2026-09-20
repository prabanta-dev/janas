/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * arch.c - the architectures Janas-LLM runs (see arch.h).
 */
#include "arch.h"

#include <string.h>

static const struct janas_arch archs[] = {
    {.name = "qwen3", .rec = JANAS_REC_NONE},
    {.name = "qwen3moe", .rec = JANAS_REC_NONE},
    {.name = "qwen3next",
     .rec = JANAS_REC_NEXT,
     .attn_gate = 1,
     .shared_expert = 1,
     .post_attn_norm = 1},
    {.name = "qwen35moe",
     .rec = JANAS_REC_Q35,
     .attn_gate = 1,
     .shared_expert = 1,
     .post_attn_norm = 1,
     .vmap_mod = 1,
     .nextn_in_file = 1},
    /* Qwen3.5, 3.6 and 3.8 dense: qwen35moe with a dense feed-forward */
    {.name = "qwen35",
     .rec = JANAS_REC_Q35,
     .attn_gate = 1,
     .post_attn_norm = 1,
     .vmap_mod = 1,
     .nextn_in_file = 1},
    {.name = "gemma4",
     .rec = JANAS_REC_NONE,
     .gelu = 1,
     .sandwich = 1,
     .swa = 1},
};

const struct janas_arch *janas_arch_find(const char *name, unsigned len)
{
    for (size_t i = 0; i < sizeof(archs) / sizeof(archs[0]); i++)
        if (strncmp(name, archs[i].name, len) == 0)
            return &archs[i];
    return NULL;
}
