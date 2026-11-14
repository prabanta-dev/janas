/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * inflate.c - DEFLATE (RFC 1951), bare or in a zlib wrapping (RFC 1950):
 * stored, fixed and dynamic Huffman blocks. Codes are decoded canonically,
 * a bit at a time against the count of codes of each length - plain, and
 * fast enough for the pictures and the documents it is for. Every read is
 * bounded: broken data ends with a reason, never past the input or the
 * output.
 */
#include "inflate.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

struct in {
    const uint8_t *p;
    size_t n, pos;
    uint32_t bits;
    int nbits;
    int bad;
    uint8_t *out;
    size_t cap, len;
};

static int bit(struct in *s)
{
    if (!s->nbits) {
        if (s->pos >= s->n) {
            s->bad = 1;
            return 0;
        }
        s->bits = s->p[s->pos++];
        s->nbits = 8;
    }
    int b = (int)(s->bits & 1);
    s->bits >>= 1;
    s->nbits--;
    return b;
}

static uint32_t bits(struct in *s, int n)
{
    uint32_t v = 0;
    for (int i = 0; i < n; i++)
        v |= (uint32_t)bit(s) << i;
    return v;
}

/* a canonical Huffman code: how many codes of each length, the symbols in
   code order */
struct huff {
    short count[16];
    short sym[320];
};

/* 0, or -1 when the lengths do not make a code */
static int build(struct huff *h, const uint8_t *len, int n)
{
    memset(h->count, 0, sizeof h->count);
    for (int i = 0; i < n; i++)
        h->count[len[i]]++;
    if (h->count[0] == n)
        return 0; /* no codes: an empty distance code is allowed */
    int left = 1;
    for (int l = 1; l < 16; l++) {
        left <<= 1;
        left -= h->count[l];
        if (left < 0)
            return -1; /* too many codes */
    }
    short offs[16];
    offs[1] = 0;
    for (int l = 1; l < 15; l++)
        offs[l + 1] = (short)(offs[l] + h->count[l]);
    for (int i = 0; i < n; i++)
        if (len[i])
            h->sym[offs[len[i]]++] = (short)i;
    return 0;
}

static int decode(struct in *s, const struct huff *h)
{
    int code = 0, first = 0, index = 0;
    for (int l = 1; l < 16; l++) {
        code |= bit(s);
        int c = h->count[l];
        if (code - c < first)
            return h->sym[index + (code - first)];
        index += c;
        first += c;
        first <<= 1;
        code <<= 1;
        if (s->bad)
            return -1;
    }
    s->bad = 1;
    return -1;
}

static const short LBASE[29] = {3,  4,  5,  6,   7,   8,   9,   10,  11, 13,
                                15, 17, 19, 23,  27,  31,  35,  43,  51, 59,
                                67, 83, 99, 115, 131, 163, 195, 227, 258};
static const short LEXT[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                               2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const short DBASE[30] = {
    1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
    33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const short DEXT[30] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                               4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                               9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

/* the symbols of one compressed block: 0, or -1 */
static int codes(struct in *s, const struct huff *lc, const struct huff *dc)
{
    for (;;) {
        int sym = decode(s, lc);
        if (s->bad || sym < 0)
            return -1;
        if (sym < 256) {
            if (s->len >= s->cap)
                return -1;
            s->out[s->len++] = (uint8_t)sym;
        } else if (sym == 256) {
            return 0;
        } else {
            sym -= 257;
            if (sym >= 29)
                return -1;
            size_t len = (size_t)LBASE[sym] + bits(s, LEXT[sym]);
            int ds = decode(s, dc);
            if (s->bad || ds < 0 || ds >= 30)
                return -1;
            size_t dist = (size_t)DBASE[ds] + bits(s, DEXT[ds]);
            if (dist > s->len || len > s->cap - s->len)
                return -1;
            for (size_t i = 0; i < len; i++, s->len++)
                s->out[s->len] = s->out[s->len - dist];
        }
    }
}

static int stored(struct in *s)
{
    s->nbits = 0; /* to the byte */
    if (s->n - s->pos < 4)
        return -1;
    unsigned len = s->p[s->pos] | (unsigned)s->p[s->pos + 1] << 8;
    unsigned nlen = s->p[s->pos + 2] | (unsigned)s->p[s->pos + 3] << 8;
    s->pos += 4;
    if ((len ^ 0xffff) != nlen || s->n - s->pos < len || s->cap - s->len < len)
        return -1;
    memcpy(s->out + s->len, s->p + s->pos, len);
    s->pos += len;
    s->len += len;
    return 0;
}

static struct huff fixed_lc, fixed_dc;

static void fixed_make(void)
{
    uint8_t l[288];
    int i = 0;
    for (; i < 144; i++)
        l[i] = 8;
    for (; i < 256; i++)
        l[i] = 9;
    for (; i < 280; i++)
        l[i] = 7;
    for (; i < 288; i++)
        l[i] = 8;
    build(&fixed_lc, l, 288);
    for (i = 0; i < 30; i++)
        l[i] = 5;
    build(&fixed_dc, l, 30);
}

static int fixed(struct in *s)
{
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, fixed_make); /* the same every time: made once */
    return codes(s, &fixed_lc, &fixed_dc);
}

static int dynamic(struct in *s)
{
    int nlen = (int)bits(s, 5) + 257, ndist = (int)bits(s, 5) + 1;
    int ncode = (int)bits(s, 4) + 4;
    if (nlen > 286 || ndist > 30 || s->bad)
        return -1;
    static const uint8_t ORDER[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                      11, 4,  12, 3, 13, 2, 14, 1, 15};
    uint8_t len[320] = {0};
    for (int i = 0; i < ncode; i++)
        len[ORDER[i]] = (uint8_t)bits(s, 3);
    struct huff h, lc, dc;
    if (build(&h, len, 19) != 0)
        return -1;
    /* the lengths of both codes, run-length coded */
    int k = 0;
    while (k < nlen + ndist) {
        int sym = decode(s, &h);
        if (s->bad || sym < 0)
            return -1;
        if (sym < 16) {
            len[k++] = (uint8_t)sym;
            continue;
        }
        uint8_t v = 0;
        int rep;
        if (sym == 16) {
            if (!k)
                return -1;
            v = len[k - 1];
            rep = 3 + (int)bits(s, 2);
        } else if (sym == 17) {
            rep = 3 + (int)bits(s, 3);
        } else {
            rep = 11 + (int)bits(s, 7);
        }
        if (k + rep > nlen + ndist)
            return -1;
        while (rep--)
            len[k++] = v;
    }
    if (!len[256]) /* no end of block */
        return -1;
    if (build(&lc, len, nlen) != 0 || build(&dc, len + nlen, ndist) != 0)
        return -1;
    return codes(s, &lc, &dc);
}

/* the blocks from s's position to the last one: 0, or -1 */
static int blocks(struct in *s)
{
    int last;
    do {
        last = bit(s);
        int type = (int)bits(s, 2);
        int rc = s->bad      ? -1
                 : type == 0 ? stored(s)
                 : type == 1 ? fixed(s)
                 : type == 2 ? dynamic(s)
                             : -1;
        if (rc != 0 || s->bad)
            return -1;
    } while (!last);
    return 0;
}

int janas_zlib_inflate(const uint8_t *p, size_t n, uint8_t *out, size_t cap,
                       char *why, size_t wn)
{
    if (n < 2 || (p[0] & 0x0f) != 8 || ((p[0] << 8) | p[1]) % 31 ||
        (p[1] & 0x20)) {
        snprintf(why, wn, "not zlib data");
        return -1;
    }
    struct in s = {.p = p, .n = n, .pos = 2, .out = out, .cap = cap};
    if (blocks(&s) != 0) {
        snprintf(why, wn, "broken compressed data");
        return -1;
    }
    if (s.len != cap) {
        snprintf(why, wn, "%zu bytes of pixels where %zu were expected", s.len,
                 cap);
        return -1;
    }
    return 0;
}

int janas_raw_inflate(const uint8_t *p, size_t n, uint8_t *out, size_t cap,
                      char *why, size_t wn)
{
    struct in s = {.p = p, .n = n, .out = out, .cap = cap};
    if (blocks(&s) != 0) {
        snprintf(why, wn, "broken compressed data");
        return -1;
    }
    if (s.len != cap) {
        snprintf(why, wn, "%zu bytes where %zu were expected", s.len, cap);
        return -1;
    }
    return 0;
}
