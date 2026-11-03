/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_wav.c - a WAV file (audio.h): the RIFF chunks walked, "fmt " read
 * (PCM 8, 16, 24 or 32 bits, float 32 or 64; WAVE_FORMAT_EXTENSIBLE by
 * its subformat), the samples of "data" averaged over the channels. A
 * data chunk longer than the file (a stream whose length was never
 * written) is read to the end of the file.
 */
#include "audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t le16(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8;
}

static uint32_t le32(const uint8_t *p)
{
    return le16(p) | le16(p + 2) << 16;
}

static double sample(const uint8_t *p, int bits, int fl)
{
    if (fl) {
        if (bits == 32) {
            float f;
            memcpy(&f, p, 4);
            return f;
        }
        double d;
        memcpy(&d, p, 8);
        return d;
    }
    switch (bits) {
    case 8:
        return (p[0] - 128) / 128.0;
    case 16:
        return (int16_t)le16(p) / 32768.0;
    case 24: {
        int32_t v = (int32_t)(le16(p) | (uint32_t)p[2] << 16);
        if (v & 0x800000)
            v -= 0x1000000;
        return v / 8388608.0;
    }
    default:
        return (int32_t)le32(p) / 2147483648.0;
    }
}

int janas_audio_wav(const uint8_t *d, size_t n, struct audio_out *o,
                    char *why, size_t wn)
{
    if (n < 12 || memcmp(d, "RIFF", 4) || memcmp(d + 8, "WAVE", 4))
        return 1;
    uint32_t fmt = 0, ch = 0, rate = 0, bits = 0, align = 0;
    const uint8_t *data = NULL;
    size_t dn = 0;
    for (size_t at = 12; at + 8 <= n;) {
        uint32_t len = le32(d + at + 4);
        const uint8_t *body = d + at + 8;
        size_t room = n - at - 8;
        if (!memcmp(d + at, "fmt ", 4)) {
            if (len < 16 || len > room) {
                snprintf(why, wn, "a WAV file whose format is cut short");
                return -1;
            }
            fmt = le16(body);
            ch = le16(body + 2);
            rate = le32(body + 4);
            align = le16(body + 12);
            bits = le16(body + 14);
            if (fmt == 0xFFFE && len >= 40)
                fmt = le16(body + 24); /* the subformat's GUID begins so */
        } else if (!memcmp(d + at, "data", 4)) {
            data = body;
            dn = len <= room ? len : room;
            break;
        }
        if (len > room)
            break;
        at += 8 + (size_t)len + (len & 1);
    }
    int fl = fmt == 3;
    if (!fmt || !data) {
        snprintf(why, wn, "a WAV file without %s", fmt ? "samples" : "format");
        return -1;
    }
    if ((fmt != 1 && fmt != 3) || !ch || ch > 64 || !rate || rate > 768000 ||
        (fl ? bits != 32 && bits != 64
            : bits != 8 && bits != 16 && bits != 24 && bits != 32) ||
        align != ch * (bits / 8)) {
        snprintf(why, wn,
                 "a WAV file of a kind this does not read (format %u, %u "
                 "channels, %u bits)",
                 fmt, ch, bits);
        return -1;
    }
    size_t frames = dn / align;
    uint32_t bs = bits / 8;
    float buf[4096];
    o->rate = rate;
    for (size_t f0 = 0; f0 < frames; f0 += 4096) {
        size_t k = frames - f0 < 4096 ? frames - f0 : 4096;
        for (size_t f = 0; f < k; f++) {
            const uint8_t *p = data + (f0 + f) * align;
            double s = 0;
            for (uint32_t c = 0; c < ch; c++)
                s += sample(p + c * bs, (int)bits, fl);
            buf[f] = (float)(s / ch);
        }
        if (audio_put(o, buf, k, why, wn) != 0)
            return -1;
    }
    return 0;
}
