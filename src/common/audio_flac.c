/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_flac.c - a FLAC file (audio.h), as RFC 9639 lays it out: the
 * "fLaC" marker, the metadata blocks (STREAMINFO read: rate, channels,
 * bits), then frames - a header checked by its CRC-8, a subframe a channel
 * (constant, verbatim, fixed predictors of order 0 to 4, linear predictors
 * of order 1 to 32, with their Rice-coded residuals and the bits a channel
 * wastes), the stereo decorrelations (left/side, side/right, mid/side), a
 * CRC-16 over the whole frame. A frame that does not check is skipped and
 * the next sync code sought, so a damaged file gives what it holds intact.
 * The samples are kept as 64-bit integers until the channels are averaged.
 */
#include "audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- bits, most significant first ---- */

struct br {
    const uint8_t *p;
    size_t n;   /* bytes */
    size_t pos; /* bits */
    int err;    /* read past the end */
};

/* k bits (0 to 57) as an unsigned number */
static uint64_t bits(struct br *b, int k)
{
    if (!k)
        return 0;
    if (b->err || b->pos + (size_t)k > b->n * 8) {
        b->err = 1;
        b->pos = b->n * 8;
        return 0;
    }
    size_t at = b->pos >> 3;
    uint64_t w = 0;
    for (int i = 0; i < 8; i++)
        w = w << 8 | (at + (size_t)i < b->n ? b->p[at + (size_t)i] : 0);
    uint64_t v = (w << (b->pos & 7)) >> (64 - k);
    b->pos += (size_t)k;
    return v;
}

static int64_t sbits(struct br *b, int k)
{
    if (k == 0)
        return 0;
    uint64_t v = bits(b, k);
    if (k < 64 && v >> (k - 1))
        return (int64_t)(v - ((uint64_t)1 << k));
    return (int64_t)v;
}

/* zeros before a one */
static uint32_t unary(struct br *b)
{
    uint32_t q = 0;
    for (;;) {
        if (b->err || b->pos >= b->n * 8) {
            b->err = 1;
            return 0;
        }
        uint8_t byte = (uint8_t)(b->p[b->pos >> 3] << (b->pos & 7));
        if (byte) {
            int lead = __builtin_clz((unsigned)byte) - 24;
            q += (uint32_t)lead;
            b->pos += (size_t)lead + 1;
            return q;
        }
        uint32_t left = 8 - (uint32_t)(b->pos & 7);
        q += left;
        b->pos += left;
        if (q > (1u << 24)) { /* no residual is that long */
            b->err = 1;
            return 0;
        }
    }
}

/* ---- checks ---- */

static uint8_t crc8(const uint8_t *p, size_t n)
{
    uint8_t c = 0;
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++)
            c = (uint8_t)(c & 0x80 ? c << 1 ^ 0x07 : c << 1);
    }
    return c;
}

static uint16_t crc16(const uint8_t *p, size_t n)
{
    uint16_t c = 0;
    for (size_t i = 0; i < n; i++) {
        c ^= (uint16_t)(p[i] << 8);
        for (int k = 0; k < 8; k++)
            c = (uint16_t)(c & 0x8000 ? c << 1 ^ 0x8005 : c << 1);
    }
    return c;
}

/* ---- a subframe ---- */

static int residual(struct br *b, int64_t *s, uint32_t n, uint32_t order)
{
    uint32_t method = (uint32_t)bits(b, 2);
    if (method > 1)
        return -1;
    int pbits = method ? 5 : 4;
    uint32_t esc = method ? 31 : 15;
    uint32_t porder = (uint32_t)bits(b, 4), parts = 1u << porder;
    if ((n >> porder) << porder != n || (n >> porder) < order)
        return -1;
    uint32_t i = order;
    for (uint32_t p = 0; p < parts && !b->err; p++) {
        uint32_t end = (p + 1) * (n >> porder);
        uint32_t k = (uint32_t)bits(b, pbits);
        if (k == esc) {
            int raw = (int)bits(b, 5);
            for (; i < end; i++)
                s[i] = sbits(b, raw);
            continue;
        }
        for (; i < end && !b->err; i++) {
            uint64_t v = (uint64_t)unary(b) << k | bits(b, (int)k);
            s[i] = (int64_t)(v >> 1) ^ -(int64_t)(v & 1);
        }
    }
    return b->err ? -1 : 0;
}

static int subframe(struct br *b, int64_t *s, uint32_t n, int bps)
{
    if (bits(b, 1))
        return -1;
    uint32_t type = (uint32_t)bits(b, 6);
    int wasted = 0;
    if (bits(b, 1)) {
        uint32_t w = unary(b) + 1;
        if ((int)w >= bps)
            return -1;
        wasted = (int)w;
        bps -= wasted;
    }
    if (b->err)
        return -1;
    if (type == 0) {
        int64_t v = sbits(b, bps);
        for (uint32_t i = 0; i < n; i++)
            s[i] = v;
    } else if (type == 1) {
        for (uint32_t i = 0; i < n; i++)
            s[i] = sbits(b, bps);
    } else if (type >= 8 && type <= 12) {
        uint32_t order = type & 7;
        if (order > n)
            return -1;
        for (uint32_t i = 0; i < order; i++)
            s[i] = sbits(b, bps);
        if (residual(b, s, n, order))
            return -1;
        /* in 64 bits that wrap, as below */
        uint64_t *u = (uint64_t *)s;
        for (uint32_t i = order; i < n; i++)
            switch (order) {
            case 1:
                u[i] += u[i - 1];
                break;
            case 2:
                u[i] += 2 * u[i - 1] - u[i - 2];
                break;
            case 3:
                u[i] += 3 * u[i - 1] - 3 * u[i - 2] + u[i - 3];
                break;
            case 4:
                u[i] += 4 * u[i - 1] - 6 * u[i - 2] + 4 * u[i - 3] - u[i - 4];
                break;
            }
    } else if (type >= 32) {
        uint32_t order = (type & 31) + 1;
        if (order > n)
            return -1;
        for (uint32_t i = 0; i < order; i++)
            s[i] = sbits(b, bps);
        int prec = (int)bits(b, 4) + 1;
        int shift = (int)sbits(b, 5);
        if (prec == 16 || shift < 0)
            return -1;
        int64_t c[32];
        for (uint32_t i = 0; i < order; i++)
            c[i] = sbits(b, prec);
        if (residual(b, s, n, order))
            return -1;
        /* in 64 bits that wrap: a valid stream never needs more, a
           broken one must not overflow */
        for (uint32_t i = order; i < n; i++) {
            uint64_t sum = 0;
            for (uint32_t k = 0; k < order; k++)
                sum += (uint64_t)c[k] * (uint64_t)s[i - 1 - k];
            s[i] =
                (int64_t)((uint64_t)s[i] + (uint64_t)((int64_t)sum >> shift));
        }
    } else {
        return -1;
    }
    if (wasted)
        for (uint32_t i = 0; i < n; i++)
            s[i] = (int64_t)((uint64_t)s[i] << wasted);
    return b->err ? -1 : 0;
}

/* ---- frames ---- */

struct stream {
    uint32_t rate, ch, bps;
    size_t n;
    int64_t *s[8];
    float *out; /* a block's mixed samples */
    struct audio_out *o;
    char *why;
    size_t wn;
    int stopped; /* the sink stopped it */
};

/* the frame at d[at]: its length, or 0 when it is not one */
static size_t frame(struct stream *st, const uint8_t *d, size_t n, size_t at)
{
    struct br b = {.p = d + at, .n = n - at};
    if (bits(&b, 15) != 0x7FFC) /* sync and the reserved bit */
        return 0;
    bits(&b, 1); /* fixed or variable block size: the same to us */
    uint32_t bs = (uint32_t)bits(&b, 4), rc = (uint32_t)bits(&b, 4);
    uint32_t ca = (uint32_t)bits(&b, 4), sz = (uint32_t)bits(&b, 3);
    if (bits(&b, 1) || bs == 0 || rc == 15 || ca > 10 || sz == 3)
        return 0;
    /* the frame's number, coded as UTF-8 */
    uint32_t c0 = (uint32_t)bits(&b, 8), more = 0;
    while (more < 7 && (c0 << more & 0x80))
        more++;
    if (more == 1 || more > 7)
        return 0;
    for (uint32_t i = 1; i < more; i++)
        if ((bits(&b, 8) & 0xC0) != 0x80)
            return 0;
    uint32_t block = bs == 1   ? 192
                     : bs <= 5 ? 576u << (bs - 2)
                     : bs == 6 ? (uint32_t)bits(&b, 8) + 1
                     : bs == 7 ? (uint32_t)bits(&b, 16) + 1
                               : 256u << (bs - 8);
    if (rc == 12)
        bits(&b, 8);
    else if (rc == 13 || rc == 14)
        bits(&b, 16);
    size_t hdr = b.pos / 8;
    if (b.err || bits(&b, 8) != crc8(d + at, hdr))
        return 0;
    static const uint32_t SZ[8] = {0, 8, 12, 0, 16, 20, 24, 32};
    uint32_t bps = sz ? SZ[sz] : st->bps;
    uint32_t ch = ca < 8 ? ca + 1 : 2;
    if (ch != st->ch || bps != st->bps || block > 65535)
        return 0;
    for (uint32_t c = 0; c < ch; c++) {
        int side =
            (ca == 8 && c == 1) || (ca == 9 && c == 0) || (ca == 10 && c == 1);
        if (subframe(&b, st->s[c], block, (int)bps + side))
            return 0;
    }
    b.pos = (b.pos + 7) & ~(size_t)7;
    size_t len = b.pos / 8;
    if (bits(&b, 16) != crc16(d + at, len) || b.err)
        return 0;
    int64_t *L = st->s[0], *R = st->s[1];
    for (uint32_t i = 0; ch == 2 && ca >= 8 && i < block; i++) {
        if (ca == 8) {
            R[i] = L[i] - R[i];
        } else if (ca == 9) {
            L[i] += R[i];
        } else {
            int64_t mid = (int64_t)((uint64_t)L[i] << 1) | (R[i] & 1);
            L[i] = (mid + R[i]) >> 1;
            R[i] = (mid - R[i]) >> 1;
        }
    }
    double scale = 1.0 / ((double)((uint64_t)1 << (bps - 1)) * ch);
    for (uint32_t i = 0; i < block; i++) {
        double v = 0;
        for (uint32_t c = 0; c < ch; c++)
            v += (double)st->s[c][i];
        st->out[i] = (float)(v * scale);
    }
    if (audio_put(st->o, st->out, block, st->why, st->wn) != 0) {
        st->stopped = 1;
        return 0;
    }
    st->n += block;
    return len + 2;
}

int janas_audio_flac(const uint8_t *d, size_t n, struct audio_out *o,
                     char *why, size_t wn)
{
    if (n < 4 || memcmp(d, "fLaC", 4))
        return 1;
    struct stream st = {.o = o, .why = why, .wn = wn};
    size_t at = 4;
    int last = 0, info = 0;
    while (!last) {
        if (at + 4 > n) {
            snprintf(why, wn, "a FLAC file cut short in its metadata");
            return -1;
        }
        last = d[at] >> 7;
        uint32_t type = d[at] & 0x7F;
        size_t len =
            (size_t)d[at + 1] << 16 | (size_t)d[at + 2] << 8 | d[at + 3];
        at += 4;
        if (len > n - at) {
            snprintf(why, wn, "a FLAC file cut short in its metadata");
            return -1;
        }
        if (type == 0 && len >= 34) {
            struct br b = {.p = d + at, .n = len};
            b.pos += 80; /* block and frame sizes */
            st.rate = (uint32_t)bits(&b, 20);
            st.ch = (uint32_t)bits(&b, 3) + 1;
            st.bps = (uint32_t)bits(&b, 5) + 1;
            info = 1;
        }
        at += len;
    }
    if (!info || !st.rate || st.bps < 4) {
        snprintf(why, wn, "a FLAC file without its stream's description");
        return -1;
    }
    st.out = malloc(sizeof(float) * 65536);
    for (uint32_t c = 0; c < st.ch; c++)
        if (!st.out || !(st.s[c] = malloc(sizeof(int64_t) * 65536))) {
            for (uint32_t k = 0; k < c; k++)
                free(st.s[k]);
            free(st.out);
            snprintf(why, wn, "out of memory");
            return -1;
        }
    o->rate = st.rate;
    while (at + 2 <= n && !st.stopped) {
        if (d[at] != 0xFF || (d[at + 1] & 0xFE) != 0xF8) {
            at++;
            continue;
        }
        size_t len = frame(&st, d, n, at);
        at += len ? len : 1;
    }
    for (uint32_t c = 0; c < st.ch; c++)
        free(st.s[c]);
    free(st.out);
    if (st.stopped)
        return -1;
    if (!st.n) {
        snprintf(why, wn, "a FLAC file without a frame this reads");
        return -1;
    }
    return 0;
}
