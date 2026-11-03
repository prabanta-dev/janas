/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_alac.c - the Apple Lossless Audio Codec, as Apple's open source
 * decoder defines it: the ALACSpecificConfig, frames of single, pair and
 * LFE elements, each channel's adaptive Rice-coded residual with its runs
 * of zeros and its sign-LMS predictor, the stereo decorrelation, the
 * low bits sent apart for 20, 24 and 32-bit sound, and uncompressed
 * frames. Exact: the integers are those the encoder started from.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_aac.h"
#include "audio_ogg.h"

struct alac {
    uint32_t frame_length;
    int sample_size, pb, mb, kb, channels;
    uint32_t rate;
    int32_t *err[2], *out[2], *extra[2];
};

static int log2u(unsigned v)
{
    return v ? 31 - __builtin_clz(v) : 0;
}

/* ones before a zero, at most 9 (the zero taken only below 9) */
static unsigned unary_0_9(struct abr *b)
{
    unsigned x = 0;
    while (x < 9 && abr_bit(b))
        x++;
    return x;
}

static unsigned decode_scalar(struct abr *b, int k, int bps)
{
    unsigned x = unary_0_9(b);
    if (x > 8) {
        x = abr_bits(b, bps);
    } else if (k != 1) {
        unsigned extra = abr_peek(b) >> (32 - k);
        x = (x << k) - x;
        if (extra > 1) {
            x += extra - 1;
            b->pos += (size_t)k;
        } else {
            b->pos += (size_t)(k - 1);
        }
    }
    return x;
}

static int rice_decompress(const struct alac *c, struct abr *b, int32_t *out,
                           int n, int bps, int mult)
{
    unsigned history = (unsigned)c->mb;
    int sign_modifier = 0;
    for (int i = 0; i < n; i++) {
        if (abr_left(b) == 0)
            return -1;
        int k = log2u((history >> 9) + 3);
        if (k > c->kb)
            k = c->kb;
        unsigned x = decode_scalar(b, k, bps);
        x += (unsigned)sign_modifier;
        sign_modifier = 0;
        out[i] = (int32_t)((x >> 1) ^ -(x & 1));
        if (x > 0xffff)
            history = 0xffff;
        else
            history += x * (unsigned)mult - ((history * (unsigned)mult) >> 9);
        /* runs of zeros */
        if (history < 128 && i + 1 < n) {
            k = 7 - log2u(history) + (int)((history + 16) >> 6);
            if (k > c->kb)
                k = c->kb;
            int block = (int)decode_scalar(b, k, 16);
            if (block > 0) {
                if (block >= n - i)
                    block = n - i - 1;
                memset(out + i + 1, 0, sizeof(int32_t) * (size_t)block);
                i += block;
            }
            if (block <= 0xffff)
                sign_modifier = 1;
            history = 0;
        }
    }
    return 0;
}

static int32_t sign_extend(uint32_t v, int bits)
{
    unsigned shift = 32u - (unsigned)bits;
    return (int32_t)(v << shift) >> shift;
}

static int sign_of(int v)
{
    return (v > 0) - (v < 0);
}

static void lpc_prediction(const int32_t *error, int32_t *outs, int n, int bps,
                           int16_t *coefs, int order, int quant)
{
    uint32_t *out = (uint32_t *)outs;
    uint32_t *pred = out;
    out[0] = (uint32_t)error[0];
    if (n <= 1)
        return;
    if (!order) {
        memcpy(out + 1, error + 1, sizeof(int32_t) * (size_t)(n - 1));
        return;
    }
    if (order == 31) {
        for (int i = 1; i < n; i++)
            out[i] =
                (uint32_t)sign_extend(out[i - 1] + (uint32_t)error[i], bps);
        return;
    }
    int i;
    for (i = 1; i <= order && i < n; i++)
        out[i] = (uint32_t)sign_extend(out[i - 1] + (uint32_t)error[i], bps);
    for (; i < n; i++) {
        /* in 32 bits that wrap, as the reference decoder's arithmetic */
        uint32_t acc = 0;
        unsigned error_val = (unsigned)error[i];
        int d = (int)*pred++;
        for (int j = 0; j < order; j++)
            acc += (pred[j] - (unsigned)d) * (unsigned)coefs[j];
        int val =
            (int)(((int64_t)(int32_t)acc + (1LL << (quant - 1))) >> quant);
        out[i] =
            (uint32_t)sign_extend((uint32_t)val + (unsigned)d + error_val, bps);
        int error_sign = sign_of((int)error_val);
        if (error_sign) {
            for (int j = 0;
                 j < order && (int)(error_val * (unsigned)error_sign) > 0;
                 j++) {
                val = (int)((unsigned)d - pred[j]);
                int sign = sign_of(val) * error_sign;
                coefs[j] = (int16_t)(coefs[j] - sign);
                val = (int)((unsigned)val * (unsigned)sign);
                error_val -= (unsigned)(val >> quant) * (unsigned)(j + 1);
            }
        }
    }
}

/* an element of ch channels into out[0..ch-1] at its first sample: the
   samples it holds, or -1 */
static int decode_element(struct alac *c, struct abr *b, int ch, int *nb)
{
    abr_bits(b, 4);
    abr_bits(b, 12);
    int has_size = abr_bit(b);
    int extra_bits = (int)abr_bits(b, 2) << 3;
    int bps = c->sample_size - extra_bits + ch - 1;
    if (bps > 32 || bps < 1)
        return -1;
    int compressed = !abr_bit(b);
    uint32_t n = has_size ? abr_bits(b, 32) : c->frame_length;
    if (!n || n > c->frame_length)
        return -1;
    if (*nb && (int)n != *nb)
        return -1;
    *nb = (int)n;
    int shift = 0, weight = 0;
    if (compressed) {
        int16_t coefs[2][32];
        int order[2], ptype[2], quant[2], mult[2];
        if (!c->kb)
            return -1;
        shift = (int)abr_bits(b, 8);
        weight = (int)abr_bits(b, 8);
        if (ch == 2 && weight && shift > 31)
            return -1;
        for (int k = 0; k < ch; k++) {
            ptype[k] = (int)abr_bits(b, 4);
            quant[k] = (int)abr_bits(b, 4);
            mult[k] = (int)abr_bits(b, 3);
            order[k] = (int)abr_bits(b, 5);
            if ((uint32_t)order[k] >= c->frame_length || !quant[k])
                return -1;
            for (int i = order[k] - 1; i >= 0; i--)
                coefs[k][i] = (int16_t)abr_bits(b, 16);
        }
        if (extra_bits) {
            if (abr_left(b) < (size_t)n * (size_t)ch * (size_t)extra_bits)
                return -1;
            for (uint32_t i = 0; i < n; i++)
                for (int k = 0; k < ch; k++)
                    c->extra[k][i] = (int32_t)abr_bits(b, extra_bits);
        }
        for (int k = 0; k < ch; k++) {
            if (rice_decompress(c, b, c->err[k], (int)n, bps,
                                mult[k] * c->pb / 4) != 0)
                return -1;
            if (ptype[k] == 15)
                lpc_prediction(c->err[k], c->err[k], (int)n, bps, NULL, 31, 0);
            lpc_prediction(c->err[k], c->out[k], (int)n, bps, coefs[k],
                           order[k], quant[k]);
        }
    } else {
        if (abr_left(b) < (size_t)n * (size_t)ch * (size_t)c->sample_size)
            return -1;
        for (uint32_t i = 0; i < n; i++)
            for (int k = 0; k < ch; k++)
                c->out[k][i] =
                    sign_extend(abr_bits(b, c->sample_size), c->sample_size);
        extra_bits = 0;
    }
    if (ch == 2 && weight) {
        for (uint32_t i = 0; i < n; i++) {
            int32_t a = c->out[0][i], bb = c->out[1][i];
            a = (int32_t)((uint32_t)a -
                          (uint32_t)((int)((int64_t)bb * weight) >> shift));
            bb = (int32_t)((uint32_t)bb + (uint32_t)a);
            c->out[0][i] = bb;
            c->out[1][i] = a;
        }
    }
    if (extra_bits)
        for (int k = 0; k < ch; k++)
            for (uint32_t i = 0; i < n; i++)
                c->out[k][i] = (int32_t)((uint32_t)c->out[k][i] << extra_bits |
                                         (uint32_t)c->extra[k][i]);
    return 0;
}

int janas_alac_stream(const uint8_t *cookie, size_t cn, const uint8_t *data,
                      const size_t *off, const size_t *len, size_t count,
                      const struct audio_edit *edit, struct audio_out *ao,
                      char *why, size_t wn)
{
    /* the cookie, past the atoms' heads it may carry */
    if (cn >= 12 && !memcmp(cookie + 4, "frma", 4)) {
        cookie += 12;
        cn -= 12;
    }
    if (cn >= 12 && !memcmp(cookie + 4, "alac", 4)) {
        cookie += 12;
        cn -= 12;
    } else if (cn >= 28 && !(cookie[0] | cookie[1] | cookie[2] | cookie[3])) {
        cookie += 4; /* the alac box's version and flags */
        cn -= 4;
    }
    if (cn < 24) {
        snprintf(why, wn, "an ALAC configuration cut short");
        return -1;
    }
    struct alac c;
    memset(&c, 0, sizeof c);
    c.frame_length = (uint32_t)cookie[0] << 24 | (uint32_t)cookie[1] << 16 |
                     (uint32_t)cookie[2] << 8 | cookie[3];
    c.sample_size = cookie[5];
    c.pb = cookie[6];
    c.mb = cookie[7];
    c.kb = cookie[8];
    c.channels = cookie[9];
    c.rate = (uint32_t)cookie[20] << 24 | (uint32_t)cookie[21] << 16 |
             (uint32_t)cookie[22] << 8 | cookie[23];
    if (!c.frame_length || c.frame_length > 65536 || !c.channels ||
        c.channels > 8 || !c.rate || c.rate > 384000 ||
        (c.sample_size != 16 && c.sample_size != 20 && c.sample_size != 24 &&
         c.sample_size != 32)) {
        snprintf(why, wn, "an ALAC configuration out of range");
        return -1;
    }
    float *pcm = NULL;
    size_t got = 0, cap = 0;
    int rc = -1;
    for (int k = 0; k < 2; k++) {
        c.err[k] = malloc(sizeof(int32_t) * c.frame_length);
        c.out[k] = malloc(sizeof(int32_t) * c.frame_length);
        c.extra[k] = malloc(sizeof(int32_t) * c.frame_length);
        if (!c.err[k] || !c.out[k] || !c.extra[k])
            goto oom;
    }
    double *mix = calloc(c.frame_length, sizeof(double));
    if (!mix)
        goto oom;
    double scale = 1.0 / (double)(1u << (c.sample_size - 1)) / c.channels;
    int bad = 0;
    for (size_t pk = 0; pk < count; pk++) {
        struct abr b = {.p = data + off[pk], .n = len[pk]};
        int nb = 0, ch = 0, ok = 1;
        memset(mix, 0, sizeof(double) * c.frame_length);
        while (abr_left(&b) >= 3) {
            int el = (int)abr_bits(&b, 3);
            if (el == 7)
                break;
            if (el > 1 && el != 3) {
                ok = 0;
                break;
            }
            int n = el == 1 ? 2 : 1;
            if (ch + n > c.channels || decode_element(&c, &b, n, &nb) != 0) {
                ok = 0;
                break;
            }
            for (int k = 0; k < n; k++)
                for (int i = 0; i < nb; i++)
                    mix[i] += c.out[k][i];
            ch += n;
        }
        if (!ok || !nb) {
            /* a packet that cannot be read is silence of a frame */
            if (++bad > 8 && !got) {
                snprintf(why, wn, "an ALAC stream that is none");
                goto end;
            }
            nb = (int)c.frame_length;
            memset(mix, 0, sizeof(double) * c.frame_length);
        }
        if ((size_t)nb > cap) { /* a packet's samples */
            size_t nc = (size_t)nb;
            float *t = realloc(pcm, sizeof(float) * nc);
            if (!t)
                goto oom2;
            pcm = t;
            cap = nc;
        }
        for (int i = 0; i < nb; i++)
            pcm[i] = (float)(mix[i] * scale);
        ao->rate = c.rate;
        if (audio_put(ao, pcm, (size_t)nb, why, wn) != 0)
            goto end;
        got += (size_t)nb;
    }
    size_t skip = 0, total = got;
    audio_edit_apply(edit, c.rate, &skip, &total);
    if (!total) {
        snprintf(why, wn, "an ALAC stream without sound");
        goto end;
    }
    ao->skip = skip;
    ao->total = skip + total;
    rc = 0;
    goto end;
oom2:
    snprintf(why, wn, "out of memory");
end:
    free(mix);
    free(pcm);
    for (int k = 0; k < 2; k++) {
        free(c.err[k]);
        free(c.out[k]);
        free(c.extra[k]);
    }
    return rc;
oom:
    for (int k = 0; k < 2; k++) {
        free(c.err[k]);
        free(c.out[k]);
        free(c.extra[k]);
    }
    snprintf(why, wn, "out of memory");
    return -1;
}
