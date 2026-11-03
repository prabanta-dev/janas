/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_image.c - pictures in a conversation (janas_llm_chat_image): read by
 * the model's vision encoder (vision.c) from the file beside the model -
 * the .vision.jns whose name is the longest prefix of the model's
 * (qwen3.5-0.8b.vision.jns serves qwen3.5-0.8b-q4km.jns) - and queued for
 * the next user message, sent or added. In the prompt a picture is its
 * start token (<|vision_start|>, Gemma's <|image>), one image token
 * (<|image_pad|>, <|image|>) a token of its grid and its end token, before
 * the message's text as the chat templates put it - ChatML's turn and
 * Gemma's alike;
 * once the prompt is in the session the model takes the pictures'
 * embeddings at those places (model_media.c). Every token id of a picture
 * is the same whatever the picture, so no computed prefix is reused past
 * the first picture, and a conversation with pictures is not kept.
 */
#include "chat.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/image.h"
#include "jns.h"
#include "model.h"
#include "tokenizer.h"
#include "vision.h"

#define MAX_IMAGE_TOKENS 1024 /* a picture's tokens at most: 1 Mpixel */

/* The encoder's file beside the model at path into best (empty: none) */
static void vision_path(const char *path, char *best, size_t best_len)
{
    char dir[4096];
    best[0] = 0;
    const char *slash = strrchr(path, '/');
    const char *base = slash ? slash + 1 : path;
    if (slash)
        snprintf(dir, sizeof dir, "%.*s", (int)(slash - path), path);
    else
        snprintf(dir, sizeof dir, ".");
    size_t best_stem = 0;
    DIR *d = opendir(dir);
    struct dirent *e;
    while (d && (e = readdir(d))) {
        size_t n = strlen(e->d_name), sfx = strlen(".vision.jns");
        if (n <= sfx || strcmp(e->d_name + n - sfx, ".vision.jns"))
            continue;
        size_t stem = n - sfx;
        if (stem > best_stem && !strncmp(base, e->d_name, stem)) {
            best_stem = stem;
            snprintf(best, best_len, "%s/%s", dir, e->d_name);
        }
    }
    if (d)
        closedir(d);
}

uint32_t janas_api_picture_block(const char *path)
{
    char best[8448];
    vision_path(path, best, sizeof best);
    struct janas_jns j;
    char err[256];
    uint32_t both = 0, most = 0;
    if (!best[0] || janas_jns_open(best, &j, err, sizeof err) != 0)
        return 0;
    janas_jns_meta_u32(&j, "vision.bidirectional", &both);
    janas_jns_meta_u32(&j, "vision.max_soft_tokens", &most);
    janas_jns_close(&j);
    return both && most && most <= 4096 ? (most + 63) / 64 * 64 : 0;
}

int32_t janas_llm_sees(const janas_llm *llm)
{
    if (!llm || !llm->path ||
        (llm->format != FORMAT_CHATML && llm->format != FORMAT_GEMMA))
        return 0;
    if (llm->vis)
        return 1;
    char best[8448];
    vision_path(llm->path, best, sizeof best);
    return best[0] != 0;
}

/* The encoder beside the model: 0, or -1 with the reason in the library's
   error. */
static int vision_of(janas_llm *llm)
{
    if (llm->vis)
        return 0;
    if (!llm->path)
        return janas_api_fail(JANAS_LLM_EMODEL, "no vision encoder"), -1;
    char best[8448];
    vision_path(llm->path, best, sizeof best);
    if (!best[0])
        return janas_api_fail(JANAS_LLM_EMODEL,
                              "this model has no vision encoder beside it "
                              "(a <model>.vision.jns, made by hf2jns_vision)"),
               -1;
    char err[300];
    llm->vis = janas_vision_open(best, err, sizeof err);
    if (!llm->vis)
        return janas_api_fail(JANAS_LLM_EMODEL, "%s", err), -1;
    if (janas_vision_dim(llm->vis) != janas_llm_model_n_embd(llm->m)) {
        janas_vision_close(llm->vis);
        llm->vis = NULL;
        return janas_api_fail(JANAS_LLM_EMODEL,
                              "%s: its embeddings are not this model's width",
                              best),
               -1;
    }
    return 0;
}

int32_t janas_llm_chat_image(janas_llm_chat *c, const void *data, int64_t n)
{
    if (!c || !data || n <= 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    if (c->llm->format != FORMAT_CHATML && c->llm->format != FORMAT_GEMMA)
        return janas_api_fail(JANAS_LLM_EMODEL,
                              "pictures are read with ChatML and Gemma "
                              "models only, for now");
    if (vision_of(c->llm) != 0)
        return JANAS_LLM_EMODEL;
    struct janas_image im;
    char why[200];
    if (janas_image_decode(data, (size_t)n, &im, why, sizeof why) != 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "%s", why);
    int max = MAX_IMAGE_TOKENS;
    const char *env = getenv("JANAS_IMAGE_MAX_TOKENS");
    if (env && atoi(env) > 0)
        max = atoi(env);
    float *embd;
    int nx, ny;
    int rc = janas_vision_encode(c->llm->vis, &im, max,
                                 janas_llm_model_pool(c->llm->m), &embd, &nx,
                                 &ny, why, sizeof why);
    janas_image_free(&im);
    if (rc != 0)
        return janas_api_fail(JANAS_LLM_ENOMEM, "%s", why);
    struct cpic *t = realloc(c->pics, sizeof(*t) * (c->n_pics + 1));
    if (!t) {
        free(embd);
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    }
    c->pics = t;
    c->pics[c->n_pics++] =
        (struct cpic){.embd = embd, .nx = (uint32_t)nx, .ny = (uint32_t)ny};
    return JANAS_LLM_OK;
}

void janas_api_pics_free(struct cpic *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        free(p[i].embd);
    free(p);
}

int janas_api_add_user(janas_llm_chat *c, const char *text, size_t n,
                       const struct cpic *pics, size_t n_pics)
{
    if (!n_pics)
        return janas_api_add_message(c, "user", text, n);
    const janas_llm *llm = c->llm;
    int32_t img, vs, ve;
    janas_vision_tokens(llm->vis, &img, &vs, &ve);
    int err = janas_api_add_special(c, llm->im_start) ||
              janas_api_add_text(c, "user\n", 5);
    for (size_t i = 0; i < n_pics && !err; i++) {
        err |= janas_api_add_special(c, vs);
        struct ppic *t = realloc(c->idpics, sizeof(*t) * (c->n_idpics + 1));
        if (!t)
            return -1;
        c->idpics = t;
        c->idpics[c->n_idpics++] =
            (struct ppic){.off = (uint32_t)c->n_ids, .pic = &pics[i]};
        for (uint32_t k = 0; k < pics[i].nx * pics[i].ny && !err; k++)
            err |= janas_api_add_special(c, img);
        err |= janas_api_add_special(c, ve);
    }
    return err || janas_api_add_text(c, text, n) ||
           janas_api_add_special(c, llm->im_end) || janas_api_end_turn_nl(c);
}

uint32_t janas_api_media_first(const janas_llm_chat *c)
{
    return c->n_idpics ? c->idpics[0].off : UINT32_MAX;
}

int janas_api_media_register(janas_llm_chat *c, uint32_t base, uint32_t from)
{
    for (size_t i = 0; i < c->n_idpics; i++) {
        const struct ppic *p = &c->idpics[i];
        if (p->off < from)
            continue; /* in the session already, as it was */
        if (janas_llm_model_media_add(c->llm->m, base + p->off, p->pic->embd,
                                      p->pic->nx, p->pic->ny,
                                      janas_vision_both_ways(c->llm->vis)) != 0)
            return -1;
    }
    return 0;
}
