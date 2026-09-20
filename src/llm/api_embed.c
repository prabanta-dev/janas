/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_embed.c - the public API: the embedding of a text (see
 * include/janas/llm.h). The text runs through the model like a prompt, with
 * no logits computed; the final states of its tokens, after the output
 * norm, are pooled as the model's metadata says (the last token's for
 * Qwen3-Embedding, the mean for others) and made unit length.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"

enum { POOL_NONE, POOL_MEAN, POOL_CLS, POOL_LAST };

int32_t janas_llm_embed_dim(const janas_llm *llm)
{
    if (!llm || llm->pooling == POOL_NONE || llm->pooling > POOL_LAST)
        return 0;
    return (int32_t)janas_llm_model_n_embd(llm->m);
}

int32_t janas_llm_embed(janas_llm *llm, const char *text, int32_t len,
                        int32_t dim, float *out, int32_t *n, int32_t *tokens)
{
    if (!llm || !text || !out || !n || dim < 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    int32_t full = janas_llm_embed_dim(llm);
    if (!full)
        return janas_api_fail(JANAS_LLM_EMODEL,
                              "this model gives no embeddings");
    if (llm->chat)
        return janas_api_fail(JANAS_LLM_EBUSY,
                              "the model has a chat: an embedding would "
                              "overwrite its conversation");
    if (dim == 0 || dim > full)
        dim = full;
    size_t tn = janas_api_text_len(text, len);
    long cnt = janas_tokenizer_encode(llm->tok, text, tn, 0, NULL, 0);
    if (cnt < 0)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    uint32_t nt = (uint32_t)cnt + (llm->add_eos && llm->eos_id >= 0);
    if (nt == 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "no text to embed");
    if (nt > llm->n_ctx)
        return janas_api_fail(JANAS_LLM_EFULL,
                              "the text is %u tokens, the context %u", nt,
                              llm->n_ctx);
    int32_t *ids = malloc((size_t)nt * sizeof(int32_t));
    double *acc = calloc((size_t)full, sizeof(double));
    if (!ids || !acc) {
        free(ids);
        free(acc);
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    }
    janas_tokenizer_encode(llm->tok, text, tn, 0, ids, (size_t)cnt);
    if (nt > (uint32_t)cnt)
        ids[cnt] = llm->eos_id;
    int mean = llm->pooling == POOL_MEAN;
    int32_t rc = JANAS_LLM_OK;
    for (uint32_t p = 0; p < nt && rc == JANAS_LLM_OK;) {
        uint32_t most = janas_llm_model_max_block(llm->m);
        uint32_t b = nt - p < most ? nt - p : most;
        if (janas_llm_model_forward(llm->m, ids + p, b, p, NULL, mean) != 0) {
            rc = janas_api_fail(JANAS_LLM_EFAIL, "the model failed");
            break;
        }
        const float *h = janas_llm_model_normed(llm->m);
        if (mean) {
            for (uint32_t r = 0; r < b; r++)
                for (int32_t i = 0; i < full; i++)
                    acc[i] += h[(size_t)r * (size_t)full + (size_t)i];
        } else if (llm->pooling == POOL_CLS ? p == 0 : p + b == nt) {
            /* the first token of the first block, or the last of the last:
               without all logits the pass keeps its last row only, so the
               first token alone makes a block of its own */
            if (llm->pooling == POOL_CLS && b > 1) {
                b = 1;
                continue;
            }
            const float *row =
                h +
                (llm->pooling == POOL_CLS ? 0 : (size_t)(b - 1) * (size_t)full);
            for (int32_t i = 0; i < full; i++)
                acc[i] = row[i];
            if (llm->pooling == POOL_CLS)
                break; /* a causal model's first state ignores the rest */
        }
        p += b;
    }
    if (rc == JANAS_LLM_OK) {
        double norm = 0;
        for (int32_t i = 0; i < dim; i++)
            norm += acc[i] * acc[i];
        norm = norm > 0 ? 1.0 / sqrt(norm) : 0;
        for (int32_t i = 0; i < dim; i++)
            out[i] = (float)(acc[i] * norm);
        *n = dim;
        if (tokens)
            *tokens = (int32_t)nt;
    }
    free(ids);
    free(acc);
    return rc;
}
