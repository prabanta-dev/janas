/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * ocr.c - the text of a scanned page, read by a model that sees pictures
 * (--ocr-model, or ocr_model = in docs.conf): a Gemma 4 or a Qwen3.5 with
 * its <model>.vision.jns beside it. The picture of a PDF's page without
 * text (pdf.c) is given to it with a request to write its text as it is;
 * the reply is the page's text. The model is opened when a first page
 * needs it, and closed at the end of the indexing round (dz_ocr_rest), so
 * that its memory is held only while there are pages to read.
 *
 * Pictures that are not a JPEG (a PDF's own raw samples) are given to it
 * as a PNG, written here: its pixels stored, not compressed - it lives in
 * memory for one call.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "docs.h"
#include "janas/llm.h"

static char model_path[4096];
static janas_llm *llm;
/* the index and a tool's call read pictures at once: one model, used by
   one at a time, not closed under the other */
static pthread_mutex_t llm_mu = PTHREAD_MUTEX_INITIALIZER;
static int broken; /* it would not open: not tried again in this run */

static const char PROMPT[] =
    "Transcribe all the text of this page exactly as written, line by "
    "line, in its language. Write only the text.";

void dz_ocr_set(const char *model)
{
    snprintf(model_path, sizeof model_path, "%s", model ? model : "");
}

int dz_ocr_on(void)
{
    return model_path[0] && !broken;
}

static int open_model(char *why, size_t wn)
{
    if (llm)
        return 0;
    struct janas_llm_params p;
    janas_llm_params_default(&p);
    p.n_ctx = 4096; /* a page's picture, about a thousand tokens, and text */
    if (janas_llm_open(model_path, &p, &llm) != JANAS_LLM_OK) {
        snprintf(why, wn, "%.300s: %.200s", model_path, janas_llm_last_error());
        broken = 1;
        return -1;
    }
    if (!janas_llm_sees(llm)) {
        snprintf(why, wn,
                 "%.300s sees no pictures: it needs its .vision.jns "
                 "beside it",
                 model_path);
        janas_llm_close(llm);
        llm = NULL;
        broken = 1;
        return -1;
    }
    return 0;
}

static int ocr_locked(const uint8_t *img, size_t n, struct janas_buf *out,
                      char *why, size_t wn);

int dz_ocr_image(const uint8_t *img, size_t n, struct janas_buf *out, char *why,
                 size_t wn)
{
    pthread_mutex_lock(&llm_mu);
    int rc = ocr_locked(img, n, out, why, wn);
    pthread_mutex_unlock(&llm_mu);
    return rc;
}

static int ocr_locked(const uint8_t *img, size_t n, struct janas_buf *out,
                      char *why, size_t wn)
{
    if (!dz_ocr_on() || open_model(why, wn) != 0)
        return -1;
    struct janas_llm_chat_params cp;
    janas_llm_chat_params_default(&cp);
    cp.temperature = 0;
    cp.thinking = 0;
    cp.max_reply = 3000;
    janas_llm_chat *c = NULL;
    if (janas_llm_chat_create(llm, &cp, &c) != JANAS_LLM_OK) {
        snprintf(why, wn, "%.200s", janas_llm_last_error());
        return -1;
    }
    int rc = -1;
    if (janas_llm_chat_image(c, img, (int64_t)n) != JANAS_LLM_OK)
        snprintf(why, wn, "the picture: %.200s", janas_llm_last_error());
    else if (janas_llm_chat_send(c, PROMPT, (int32_t)strlen(PROMPT)) !=
             JANAS_LLM_OK)
        snprintf(why, wn, "%.200s", janas_llm_last_error());
    else {
        char buf[4096];
        int32_t len = 0, r;
        size_t start = out->n;
        while ((r = janas_llm_chat_next(c, buf, sizeof buf, &len)) ==
               JANAS_LLM_OK)
            janas_buf_put(out, buf, (size_t)len);
        rc = r == JANAS_LLM_DONE || r == JANAS_LLM_EFULL ? 0 : -1;
        if (rc)
            snprintf(why, wn, "%.200s", janas_llm_last_error());
        /* the page's text, without the spaces around it */
        while (out->n > start &&
               (out->p[out->n - 1] == ' ' || out->p[out->n - 1] == '\n'))
            out->n--;
    }
    janas_llm_chat_destroy(c);
    return out->oom ? -1 : rc;
}

void dz_ocr_rest(void)
{
    pthread_mutex_lock(&llm_mu);
    if (llm)
        janas_llm_close(llm);
    llm = NULL;
    pthread_mutex_unlock(&llm_mu);
}

/* ---- a PNG of raw samples, its deflate blocks stored ---- */

static uint32_t crc_table[256];

static uint32_t crc(uint32_t c, const uint8_t *p, size_t n)
{
    if (!crc_table[1])
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t v = i;
            for (int k = 0; k < 8; k++)
                v = v & 1 ? 0xedb88320u ^ (v >> 1) : v >> 1;
            crc_table[i] = v;
        }
    c = ~c;
    while (n--)
        c = crc_table[(c ^ *p++) & 0xff] ^ (c >> 8);
    return ~c;
}

static void be32(struct janas_buf *o, uint32_t v)
{
    uint8_t b[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8),
                    (uint8_t)v};
    janas_buf_put(o, (const char *)b, 4);
}

static void chunk(struct janas_buf *o, const char *type, const uint8_t *p,
                  size_t n)
{
    be32(o, (uint32_t)n);
    size_t at = o->n;
    janas_buf_put(o, type, 4);
    if (n)
        janas_buf_put(o, (const char *)p, n);
    if (!o->oom)
        be32(o, crc(0, (const uint8_t *)o->p + at, n + 4));
}

int dz_png(const uint8_t *px, uint32_t w, uint32_t h, int ch,
           struct janas_buf *o)
{
    if (!w || !h || (ch != 1 && ch != 3))
        return -1;
    size_t row = (size_t)w * (size_t)ch, raw = (row + 1) * h;
    /* the rows, each after its filter byte (0: none) */
    uint8_t *r = malloc(raw);
    if (!r)
        return -1;
    for (uint32_t y = 0; y < h; y++) {
        r[y * (row + 1)] = 0;
        memcpy(r + y * (row + 1) + 1, px + y * row, row);
    }
    /* zlib: its header, stored blocks of at most 65535 bytes, Adler-32 */
    size_t blocks = raw / 65535 + 1;
    uint8_t *z = malloc(2 + raw + blocks * 5 + 4);
    if (!z) {
        free(r);
        return -1;
    }
    size_t zn = 0;
    z[zn++] = 0x78;
    z[zn++] = 0x01;
    uint32_t a = 1, b = 0;
    for (size_t at = 0; at < raw || at == 0;) {
        size_t k = raw - at > 65535 ? 65535 : raw - at;
        z[zn++] = at + k == raw ? 1 : 0;
        z[zn++] = (uint8_t)k;
        z[zn++] = (uint8_t)(k >> 8);
        z[zn++] = (uint8_t)~k;
        z[zn++] = (uint8_t)(~k >> 8);
        memcpy(z + zn, r + at, k);
        zn += k;
        for (size_t i = 0; i < k; i++) {
            a = (a + r[at + i]) % 65521;
            b = (b + a) % 65521;
        }
        at += k;
        if (!k)
            break;
    }
    uint32_t ad = (b << 16) | a;
    z[zn++] = (uint8_t)(ad >> 24);
    z[zn++] = (uint8_t)(ad >> 16);
    z[zn++] = (uint8_t)(ad >> 8);
    z[zn++] = (uint8_t)ad;
    free(r);
    janas_buf_put(o, "\x89PNG\r\n\x1a\n", 8);
    uint8_t ihdr[13] = {(uint8_t)(w >> 24),
                        (uint8_t)(w >> 16),
                        (uint8_t)(w >> 8),
                        (uint8_t)w,
                        (uint8_t)(h >> 24),
                        (uint8_t)(h >> 16),
                        (uint8_t)(h >> 8),
                        (uint8_t)h,
                        8,
                        ch == 3 ? 2 : 0,
                        0,
                        0,
                        0};
    chunk(o, "IHDR", ihdr, sizeof ihdr);
    chunk(o, "IDAT", z, zn);
    chunk(o, "IEND", NULL, 0);
    free(z);
    return o->oom ? -1 : 0;
}
