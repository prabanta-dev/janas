/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * image_jpeg.c - JPEG (ITU T.81) into 8-bit RGB (image.h): baseline and
 * extended sequential and progressive Huffman coding, 8-bit samples, one
 * component (gray) or three (YCbCr, or RGB when the file says so), any
 * sampling factors, restart intervals.
 *
 * Every coefficient of the picture is decoded first (progressive scans
 * refine them in place), then each block is dequantized and turned into
 * samples by an integer inverse DCT, the color planes are brought to full
 * size and converted. The arithmetic is chosen to give the pixels the
 * common decoders give (Pillow, through libjpeg): the IDCT is the
 * separable integer form of the Loeffler-Ligtenberg-Moschytz transform
 * with 13-bit constants and two extra bits between the passes; planes
 * sampled at half width, half height or both are brought up by
 * interpolating with the neighbours at 3:1 ("fancy" upsampling), other
 * ratios by repeating samples; YCbCr goes to RGB with the JFIF
 * coefficients in 16-bit fixed point.
 */
#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t ZZ[64] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

struct huff {
    int ok;
    uint8_t val[256];
    int mincode[17], maxcode[18], valptr[17];
};

struct comp {
    int id, h, v, tq;
    int td, ta; /* the scan's tables */
    int bw, bh; /* blocks across and down, padded to whole MCUs */
    int cw, ch; /* blocks the component's samples cover */
    int dc_pred;
    int16_t *coef; /* bw * bh blocks of 64, in natural order */
    uint8_t *plane;
};

struct jpg {
    const uint8_t *p;
    size_t n, pos;
    uint32_t bits;
    int nbits;
    int marker; /* a marker met inside entropy data */
    int w, h, nc, hmax, vmax, mcux, mcuy, progressive;
    struct comp c[3];
    uint16_t q[4][64];
    struct huff dc[4], ac[4];
    int restart;
    int eobrun;
    int adobe, transform;
    const char *err;
};

/* ---- bits ---- */

static int bit(struct jpg *j)
{
    if (!j->nbits) {
        int b = 0;
        if (!j->marker && j->pos < j->n) {
            b = j->p[j->pos];
            if (b == 0xff) {
                int nx = j->pos + 1 < j->n ? j->p[j->pos + 1] : 0;
                if (nx == 0) {
                    j->pos += 2; /* a stuffed 0xff */
                } else {
                    j->marker = nx; /* the data ends: zeros from here */
                    b = 0;
                }
            } else {
                j->pos++;
            }
        }
        j->bits = (uint32_t)b;
        j->nbits = 8;
    }
    j->nbits--;
    return (int)(j->bits >> j->nbits) & 1;
}

static int receive(struct jpg *j, int s)
{
    int v = 0;
    for (int i = 0; i < s; i++)
        v = v << 1 | bit(j);
    return v;
}

static int extend(int v, int s)
{
    return s && v < (1 << (s - 1)) ? v - (1 << s) + 1 : v;
}

static int decode(struct jpg *j, const struct huff *h)
{
    if (!h->ok) {
        j->err = "a scan uses a Huffman table the file did not give";
        return 0;
    }
    int code = 0;
    for (int l = 1; l <= 16; l++) {
        code = code << 1 | bit(j);
        if (code <= h->maxcode[l])
            return h->val[h->valptr[l] + code - h->mincode[l]];
    }
    j->err = "a bad Huffman code";
    return 0;
}

/* ---- markers ---- */

static int u16(const uint8_t *p)
{
    return p[0] << 8 | p[1];
}

static void dht(struct jpg *j, const uint8_t *d, size_t len)
{
    while (len > 0 && !j->err) {
        if (len < 17) {
            j->err = "a bad Huffman table";
            return;
        }
        int tc = d[0] >> 4, th = d[0] & 15;
        if (tc > 1 || th > 3) {
            j->err = "a bad Huffman table";
            return;
        }
        struct huff *h = tc ? &j->ac[th] : &j->dc[th];
        int count = 0, code = 0, k = 0;
        for (int l = 1; l <= 16; l++)
            count += d[l];
        if (count > 256 || (size_t)(17 + count) > len) {
            j->err = "a bad Huffman table";
            return;
        }
        memcpy(h->val, d + 17, (size_t)count);
        for (int l = 1; l <= 16; l++) {
            h->valptr[l] = k;
            h->mincode[l] = code;
            code += d[l];
            k += d[l];
            h->maxcode[l] = d[l] ? code - 1 : -1;
            code <<= 1;
        }
        h->maxcode[17] = 0x7fffffff;
        h->ok = 1;
        d += 17 + count;
        len -= (size_t)(17 + count);
    }
}

static void dqt(struct jpg *j, const uint8_t *d, size_t len)
{
    while (len > 0) {
        int pq = d[0] >> 4, tq = d[0] & 15;
        size_t need = 1 + (pq ? 128 : 64);
        if (pq > 1 || tq > 3 || len < need) {
            j->err = "a bad quantization table";
            return;
        }
        for (int i = 0; i < 64; i++)
            j->q[tq][ZZ[i]] = (uint16_t)(pq ? u16(d + 1 + 2 * i) : d[1 + i]);
        d += need;
        len -= need;
    }
}

static void sof(struct jpg *j, const uint8_t *d, size_t len)
{
    if (len < 6 || d[0] != 8) {
        j->err = "a JPEG of other than 8-bit samples";
        return;
    }
    j->h = u16(d + 1);
    j->w = u16(d + 3);
    j->nc = d[5];
    if (j->nc != 1 && j->nc != 3) {
        j->err = "a JPEG of other than one or three components (CMYK?)";
        return;
    }
    if (len < 6 + 3 * (size_t)j->nc || !j->w || !j->h) {
        j->err = "a bad frame header";
        return;
    }
    if (j->w > JANAS_IMAGE_MAX_SIDE || j->h > JANAS_IMAGE_MAX_SIDE ||
        (size_t)j->w * (size_t)j->h > JANAS_IMAGE_MAX_PIXELS) {
        j->err = "a picture larger than allowed (16384 a side, 64 Mpixel)";
        return;
    }
    j->hmax = j->vmax = 1;
    for (int i = 0; i < j->nc; i++) {
        struct comp *c = &j->c[i];
        c->id = d[6 + 3 * i];
        c->h = d[7 + 3 * i] >> 4;
        c->v = d[7 + 3 * i] & 15;
        c->tq = d[8 + 3 * i];
        if (c->h < 1 || c->h > 4 || c->v < 1 || c->v > 4 || c->tq > 3) {
            j->err = "a bad frame header";
            return;
        }
        if (c->h > j->hmax)
            j->hmax = c->h;
        if (c->v > j->vmax)
            j->vmax = c->v;
    }
    /* each component's sampling a divisor of the largest (1, 2, 4 with 4;
       1, 3 with 3...): the planes are scaled up by whole factors */
    for (int i = 0; i < j->nc; i++)
        if (j->hmax % j->c[i].h || j->vmax % j->c[i].v) {
            j->err = "a sampling this decoder does not take";
            return;
        }
    j->mcux = (j->w + 8 * j->hmax - 1) / (8 * j->hmax);
    j->mcuy = (j->h + 8 * j->vmax - 1) / (8 * j->vmax);
    /* every block costs at least a bit of entropy-coded data: a header
       claiming more blocks than the file can hold is refused before the
       memory is taken */
    size_t blocks = 0;
    for (int i = 0; i < j->nc; i++)
        blocks += (size_t)j->mcux * (size_t)j->c[i].h * (size_t)j->mcuy *
                  (size_t)j->c[i].v;
    if (blocks > 8 * j->n + 4096) {
        j->err = "a picture larger than its data";
        return;
    }
    for (int i = 0; i < j->nc; i++) {
        struct comp *c = &j->c[i];
        int sw = (j->w * c->h + j->hmax - 1) / j->hmax;
        int sh = (j->h * c->v + j->vmax - 1) / j->vmax;
        c->cw = (sw + 7) / 8;
        c->ch = (sh + 7) / 8;
        c->bw = j->mcux * c->h;
        c->bh = j->mcuy * c->v;
        c->coef = calloc((size_t)c->bw * (size_t)c->bh * 64, sizeof(int16_t));
        if (!c->coef) {
            j->err = "out of memory";
            return;
        }
    }
}

/* ---- entropy-coded blocks ---- */

static void block_seq(struct jpg *j, struct comp *c, int16_t *b)
{
    int t = decode(j, &j->dc[c->td]);
    if (t > 11) {
        j->err = "a bad DC difference";
        return;
    }
    c->dc_pred += extend(receive(j, t), t);
    b[0] = (int16_t)c->dc_pred;
    for (int k = 1; k < 64 && !j->err;) {
        int rs = decode(j, &j->ac[c->ta]), r = rs >> 4, s = rs & 15;
        if (s) {
            k += r;
            if (k > 63) {
                j->err = "a bad AC run";
                return;
            }
            b[ZZ[k++]] = (int16_t)extend(receive(j, s), s);
        } else if (r == 15) {
            k += 16;
        } else {
            break;
        }
    }
}

static void dc_first(struct jpg *j, struct comp *c, int16_t *b, int al)
{
    int t = decode(j, &j->dc[c->td]);
    if (t > 11) {
        j->err = "a bad DC difference";
        return;
    }
    c->dc_pred += extend(receive(j, t), t);
    b[0] = (int16_t)(c->dc_pred * (1 << al));
}

static void dc_refine(struct jpg *j, int16_t *b, int al)
{
    if (bit(j))
        b[0] = (int16_t)(b[0] | (1 << al));
}

static void ac_first(struct jpg *j, struct comp *c, int16_t *b, int ss, int se,
                     int al)
{
    if (j->eobrun) {
        j->eobrun--;
        return;
    }
    for (int k = ss; k <= se && !j->err;) {
        int rs = decode(j, &j->ac[c->ta]), r = rs >> 4, s = rs & 15;
        if (s) {
            k += r;
            if (k > se) {
                j->err = "a bad AC run";
                return;
            }
            b[ZZ[k++]] = (int16_t)(extend(receive(j, s), s) * (1 << al));
        } else if (r == 15) {
            k += 16;
        } else {
            j->eobrun = (1 << r) - 1;
            if (r)
                j->eobrun += receive(j, r);
            break;
        }
    }
}

/* the coefficients already non-zero in k.. get their correction bits;
   with zeros to skip (r >= 0) it stops at the r+1-th zero (its index) */
static int refine_nonzero(struct jpg *j, int16_t *b, int k, int se, int r,
                          int p1)
{
    for (; k <= se; k++) {
        int16_t *c = &b[ZZ[k]];
        if (*c) {
            if (bit(j) && (*c & p1) == 0)
                *c = (int16_t)(*c >= 0 ? *c + p1 : *c - p1);
        } else if (r >= 0) {
            if (r == 0)
                return k;
            r--;
        }
    }
    return k;
}

static void ac_refine(struct jpg *j, struct comp *c, int16_t *b, int ss, int se,
                      int al)
{
    int p1 = 1 << al;
    int k = ss;
    if (!j->eobrun) {
        while (k <= se && !j->err) {
            int rs = decode(j, &j->ac[c->ta]), r = rs >> 4, s = rs & 15;
            int value = 0;
            if (s) {
                if (s != 1) {
                    j->err = "a bad refinement";
                    return;
                }
                value = bit(j) ? p1 : -p1;
            } else if (r != 15) {
                j->eobrun = 1 << r;
                if (r)
                    j->eobrun += receive(j, r);
                break; /* the rest of the band below, as end of band */
            }
            /* r zeros skipped (the non-zero ones refined on the way),
               then the new coefficient (or 16 zeros) */
            k = refine_nonzero(j, b, k, se, r, p1);
            if (k > se) {
                if (s)
                    j->err = "a bad refinement";
                return;
            }
            if (s)
                b[ZZ[k]] = (int16_t)value;
            k++;
        }
    }
    if (j->eobrun) {
        refine_nonzero(j, b, k, se, -1, p1);
        j->eobrun--;
    }
}

/* ---- scans ---- */

static int expect_restart(struct jpg *j)
{
    /* to the marker: RSTn expected */
    j->nbits = 0;
    if (!j->marker) {
        while (j->pos + 1 < j->n &&
               !(j->p[j->pos] == 0xff && j->p[j->pos + 1] != 0 &&
                 j->p[j->pos + 1] != 0xff))
            j->pos++;
        if (j->pos + 1 < j->n)
            j->marker = j->p[j->pos + 1];
    }
    if (j->marker < 0xd0 || j->marker > 0xd7)
        return -1;
    j->pos += 2;
    j->marker = 0;
    j->eobrun = 0;
    for (int i = 0; i < j->nc; i++)
        j->c[i].dc_pred = 0;
    return 0;
}

static void one_block(struct jpg *j, struct comp *c, int16_t *b, int ss, int se,
                      int ah, int al)
{
    if (!j->progressive)
        block_seq(j, c, b);
    else if (ss == 0)
        ah ? dc_refine(j, b, al) : dc_first(j, c, b, al);
    else
        ah ? ac_refine(j, c, b, ss, se, al) : ac_first(j, c, b, ss, se, al);
}

static void sos(struct jpg *j, const uint8_t *d, size_t len)
{
    if (!j->nc || len < 1) {
        j->err = "a scan before the frame";
        return;
    }
    int ns = d[0];
    if (ns < 1 || ns > j->nc || len < 4 + 2 * (size_t)ns) {
        j->err = "a bad scan header";
        return;
    }
    struct comp *sc[3];
    for (int i = 0; i < ns; i++) {
        int id = d[1 + 2 * i];
        sc[i] = NULL;
        for (int k = 0; k < j->nc; k++)
            if (j->c[k].id == id)
                sc[i] = &j->c[k];
        if (!sc[i]) {
            j->err = "a scan of a component the frame does not have";
            return;
        }
        sc[i]->td = d[2 + 2 * i] >> 4;
        sc[i]->ta = d[2 + 2 * i] & 15;
        if (sc[i]->td > 3 || sc[i]->ta > 3) {
            j->err = "a bad scan header";
            return;
        }
    }
    int ss = d[1 + 2 * ns], se = d[2 + 2 * ns];
    int ah = d[3 + 2 * ns] >> 4, al = d[3 + 2 * ns] & 15;
    if (j->progressive ? (ss > se || se > 63 || (ss == 0 && se) ||
                          (ss && ns != 1) || al > 13)
                       : (ss || se != 63 || ah || al)) {
        j->err = "a bad scan header";
        return;
    }
    j->pos += len; /* the data follows */
    j->bits = 0;
    j->nbits = 0;
    j->marker = 0;
    j->eobrun = 0;
    for (int i = 0; i < j->nc; i++)
        j->c[i].dc_pred = 0;
    int todo = j->restart;
    if (ns == 1) { /* not interleaved: the component's own blocks */
        struct comp *c = sc[0];
        for (int by = 0; by < c->ch && !j->err; by++)
            for (int bx = 0; bx < c->cw && !j->err; bx++) {
                if (j->restart && !todo) {
                    if (expect_restart(j) != 0) {
                        j->err = "a restart marker missing";
                        return;
                    }
                    todo = j->restart;
                }
                one_block(j, c,
                          c->coef +
                              ((size_t)by * (size_t)c->bw + (size_t)bx) * 64,
                          ss, se, ah, al);
                todo--;
            }
    } else {
        for (int my = 0; my < j->mcuy && !j->err; my++)
            for (int mx = 0; mx < j->mcux && !j->err; mx++) {
                if (j->restart && !todo) {
                    if (expect_restart(j) != 0) {
                        j->err = "a restart marker missing";
                        return;
                    }
                    todo = j->restart;
                }
                for (int i = 0; i < ns; i++) {
                    struct comp *c = sc[i];
                    for (int v = 0; v < c->v; v++)
                        for (int h = 0; h < c->h; h++) {
                            size_t by = (size_t)(my * c->v + v);
                            size_t bx = (size_t)(mx * c->h + h);
                            one_block(j, c,
                                      c->coef + (by * (size_t)c->bw + bx) * 64,
                                      ss, se, ah, al);
                        }
                }
                todo--;
            }
    }
    /* past the data, to the next marker */
    j->nbits = 0;
    while (j->pos + 1 < j->n &&
           !(j->p[j->pos] == 0xff && j->p[j->pos + 1] != 0 &&
             !(j->p[j->pos + 1] >= 0xd0 && j->p[j->pos + 1] <= 0xd7)))
        j->pos++;
}

/* ---- samples ---- */

#define CB 13 /* bits of the constants */
#define P1 2  /* bits kept between the passes */
#define F(x) ((int64_t)((x) * (1 << CB) + 0.5))
#define F16(x) ((int32_t)((x) * 65536 + 0.5)) /* the color constants */

static uint8_t clamp(int64_t v)
{
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

/* one 8x8 block: dequantized coefficients in, samples out (stride s); in
   64 bits, as libjpeg-turbo's long: a corrupt file's coefficients cannot
   overflow it */
static void idct(const int16_t *in, const uint16_t *q, uint8_t *out, size_t s)
{
    int64_t ws[64];
    for (int x = 0; x < 8; x++) { /* columns */
        int64_t c[8];
        for (int y = 0; y < 8; y++)
            c[y] = (int64_t)in[y * 8 + x] * q[y * 8 + x];
        int64_t z1 = (c[2] + c[6]) * F(0.541196100);
        int64_t t2 = z1 + c[6] * -F(1.847759065);
        int64_t t3 = z1 + c[2] * F(0.765366865);
        int64_t t0 = (c[0] + c[4]) * (1 << CB);
        int64_t t1 = (c[0] - c[4]) * (1 << CB);
        int64_t e10 = t0 + t3, e13 = t0 - t3, e11 = t1 + t2, e12 = t1 - t2;
        int64_t o0 = c[7], o1 = c[5], o2 = c[3], o3 = c[1];
        int64_t a1 = o0 + o3, a2 = o1 + o2, a3 = o0 + o2, a4 = o1 + o3;
        int64_t z5 = (a3 + a4) * F(1.175875602);
        o0 *= F(0.298631336);
        o1 *= F(2.053119869);
        o2 *= F(3.072711026);
        o3 *= F(1.501321110);
        a1 *= -F(0.899976223);
        a2 *= -F(2.562915447);
        a3 = a3 * -F(1.961570560) + z5;
        a4 = a4 * -F(0.390180644) + z5;
        o0 += a1 + a3;
        o1 += a2 + a4;
        o2 += a2 + a3;
        o3 += a1 + a4;
        const int sh = CB - P1;
        const int64_t rd = 1 << (sh - 1);
        ws[0 * 8 + x] = (e10 + o3 + rd) >> sh;
        ws[7 * 8 + x] = (e10 - o3 + rd) >> sh;
        ws[1 * 8 + x] = (e11 + o2 + rd) >> sh;
        ws[6 * 8 + x] = (e11 - o2 + rd) >> sh;
        ws[2 * 8 + x] = (e12 + o1 + rd) >> sh;
        ws[5 * 8 + x] = (e12 - o1 + rd) >> sh;
        ws[3 * 8 + x] = (e13 + o0 + rd) >> sh;
        ws[4 * 8 + x] = (e13 - o0 + rd) >> sh;
    }
    for (int y = 0; y < 8; y++) { /* rows */
        const int64_t *c = ws + y * 8;
        int64_t z1 = (c[2] + c[6]) * F(0.541196100);
        int64_t t2 = z1 + c[6] * -F(1.847759065);
        int64_t t3 = z1 + c[2] * F(0.765366865);
        int64_t t0 = (c[0] + c[4]) * (1 << CB);
        int64_t t1 = (c[0] - c[4]) * (1 << CB);
        int64_t e10 = t0 + t3, e13 = t0 - t3, e11 = t1 + t2, e12 = t1 - t2;
        int64_t o0 = c[7], o1 = c[5], o2 = c[3], o3 = c[1];
        int64_t a1 = o0 + o3, a2 = o1 + o2, a3 = o0 + o2, a4 = o1 + o3;
        int64_t z5 = (a3 + a4) * F(1.175875602);
        o0 *= F(0.298631336);
        o1 *= F(2.053119869);
        o2 *= F(3.072711026);
        o3 *= F(1.501321110);
        a1 *= -F(0.899976223);
        a2 *= -F(2.562915447);
        a3 = a3 * -F(1.961570560) + z5;
        a4 = a4 * -F(0.390180644) + z5;
        o0 += a1 + a3;
        o1 += a2 + a4;
        o2 += a2 + a3;
        o3 += a1 + a4;
        const int sh = CB + P1 + 3;
        const int64_t rd = 1 << (sh - 1);
        uint8_t *o = out + (size_t)y * s;
        o[0] = clamp(((e10 + o3 + rd) >> sh) + 128);
        o[7] = clamp(((e10 - o3 + rd) >> sh) + 128);
        o[1] = clamp(((e11 + o2 + rd) >> sh) + 128);
        o[6] = clamp(((e11 - o2 + rd) >> sh) + 128);
        o[2] = clamp(((e12 + o1 + rd) >> sh) + 128);
        o[5] = clamp(((e12 - o1 + rd) >> sh) + 128);
        o[3] = clamp(((e13 + o0 + rd) >> sh) + 128);
        o[4] = clamp(((e13 - o0 + rd) >> sh) + 128);
    }
}

/* the component's samples at full size, row y (w samples) into out */
static void full_row(const struct jpg *j, const struct comp *c, int y,
                     uint8_t *out)
{
    int fx = j->hmax / c->h, fy = j->vmax / c->v;
    size_t pw = (size_t)c->bw * 8;
    /* the component's real size, as the decoders see it */
    int sw = (j->w * c->h + j->hmax - 1) / j->hmax;
    int sh = (j->h * c->v + j->vmax - 1) / j->vmax;
    /* interpolated where the decoders interpolate: half width only when
       the plane is wider than two samples */
    int fancy_h = fx == 2 && j->hmax % c->h == 0 && sw > 2;
    int fancy_v = fy == 2 && j->vmax % c->v == 0 && (fx == 1 || sw > 2);
    int sy = y / fy;
    const uint8_t *row = c->plane + (size_t)sy * pw;
    if (fx == 1 && fy == 1) {
        memcpy(out, row, (size_t)j->w);
        return;
    }
    if (fancy_h && fancy_v && fy == 2 && fx == 2) {
        /* 2x2: rows mixed 3:1 with the one above or below, then columns */
        int ny =
            y % 2 ? (sy + 1 < sh ? sy + 1 : sh - 1) : (sy > 0 ? sy - 1 : 0);
        const uint8_t *r1 = c->plane + (size_t)ny * pw;
        int last = 0, cur = row[0] * 3 + r1[0];
        int nx = sw > 1 ? row[1] * 3 + r1[1] : cur;
        uint8_t *o = out;
        int ow = 0;
        if (sw == 1) {
            o[0] = (uint8_t)((cur * 4 + 8) >> 4);
            if (j->w > 1)
                o[1] = (uint8_t)((cur * 4 + 7) >> 4);
            return;
        }
        o[ow++] = (uint8_t)((cur * 4 + 8) >> 4);
        o[ow++] = (uint8_t)((cur * 3 + nx + 7) >> 4);
        last = cur;
        cur = nx;
        for (int x = 1; x < sw - 1; x++) {
            nx = row[x + 1] * 3 + r1[x + 1];
            if (ow < j->w)
                o[ow++] = (uint8_t)((cur * 3 + last + 8) >> 4);
            if (ow < j->w)
                o[ow++] = (uint8_t)((cur * 3 + nx + 7) >> 4);
            last = cur;
            cur = nx;
        }
        if (ow < j->w)
            o[ow++] = (uint8_t)((cur * 3 + last + 8) >> 4);
        if (ow < j->w)
            o[ow++] = (uint8_t)((cur * 4 + 7) >> 4);
        return;
    }
    if (fancy_h && fx == 2 && fy == 1) {
        /* 2x1: columns mixed 3:1 with the left or right one */
        uint8_t *o = out;
        int ow = 0;
        if (sw == 1) {
            o[0] = row[0];
            if (j->w > 1)
                o[1] = row[0];
            return;
        }
        o[ow++] = row[0];
        o[ow++] = (uint8_t)((row[0] * 3 + row[1] + 2) >> 2);
        for (int x = 1; x < sw - 1; x++) {
            int v = row[x] * 3;
            if (ow < j->w)
                o[ow++] = (uint8_t)((v + row[x - 1] + 1) >> 2);
            if (ow < j->w)
                o[ow++] = (uint8_t)((v + row[x + 1] + 2) >> 2);
        }
        if (ow < j->w)
            o[ow++] = (uint8_t)((row[sw - 1] * 3 + row[sw - 2] + 1) >> 2);
        if (ow < j->w)
            o[ow++] = row[sw - 1];
        return;
    }
    if (fancy_v && fy == 2 && fx == 1) {
        /* 1x2: rows mixed 3:1 with the one above or below */
        int ny =
            y % 2 ? (sy + 1 < sh ? sy + 1 : sh - 1) : (sy > 0 ? sy - 1 : 0);
        const uint8_t *r1 = c->plane + (size_t)ny * pw;
        int bias = y % 2 ? 2 : 1;
        for (int x = 0; x < j->w; x++)
            out[x] = (uint8_t)((row[x] * 3 + r1[x] + bias) >> 2);
        return;
    }
    for (int x = 0; x < j->w; x++) /* other ratios: samples repeated */
        out[x] = row[x / fx];
}

static int finish(struct jpg *j, struct janas_image *im)
{
    for (int i = 0; i < j->nc; i++) {
        struct comp *c = &j->c[i];
        size_t pw = (size_t)c->bw * 8;
        c->plane = malloc(pw * (size_t)c->bh * 8);
        if (!c->plane)
            return -1;
        for (int by = 0; by < c->bh; by++)
            for (int bx = 0; bx < c->bw; bx++)
                idct(c->coef + ((size_t)by * (size_t)c->bw + (size_t)bx) * 64,
                     j->q[c->tq],
                     c->plane + (size_t)by * 8 * pw + (size_t)bx * 8, pw);
    }
    im->rgb = malloc((size_t)j->w * (size_t)j->h * 3);
    uint8_t *rows = malloc((size_t)j->w * 3);
    if (!im->rgb || !rows) {
        free(rows);
        return -1;
    }
    /* RGB when the file says so: an Adobe marker without transform, or
       components named R, G, B */
    int rgb = j->nc == 3 &&
              ((j->adobe && j->transform == 0) ||
               (j->c[0].id == 'R' && j->c[1].id == 'G' && j->c[2].id == 'B'));
    int32_t cr_r[256], cb_b[256], cr_g[256], cb_g[256];
    for (int i = 0; i < 256; i++) {
        int x = i - 128;
        const int32_t half = 1 << 15;
        cr_r[i] = (F16(1.40200) * x + half) >> 16;
        cb_b[i] = (F16(1.77200) * x + half) >> 16;
        cr_g[i] = -F16(0.71414) * x;
        cb_g[i] = -F16(0.34414) * x + half;
    }
    for (int y = 0; y < j->h; y++) {
        for (int i = 0; i < j->nc; i++)
            full_row(j, &j->c[i], y, rows + (size_t)i * (size_t)j->w);
        uint8_t *o = im->rgb + (size_t)y * (size_t)j->w * 3;
        for (int x = 0; x < j->w; x++, o += 3) {
            int Y = rows[x];
            if (j->nc == 1) {
                o[0] = o[1] = o[2] = (uint8_t)Y;
            } else if (rgb) {
                o[0] = (uint8_t)Y;
                o[1] = rows[(size_t)j->w + (size_t)x];
                o[2] = rows[2 * (size_t)j->w + (size_t)x];
            } else {
                int cb = rows[(size_t)j->w + (size_t)x];
                int cr = rows[2 * (size_t)j->w + (size_t)x];
                o[0] = clamp(Y + cr_r[cr]);
                o[1] = clamp(Y + ((cb_g[cb] + cr_g[cr]) >> 16));
                o[2] = clamp(Y + cb_b[cb]);
            }
        }
    }
    free(rows);
    im->w = j->w;
    im->h = j->h;
    return 0;
}

int janas_jpeg_decode(const uint8_t *p, size_t n, struct janas_image *im,
                      char *why, size_t wn)
{
    struct jpg *j = calloc(1, sizeof(*j));
    memset(im, 0, sizeof(*im));
    if (!j) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    j->p = p;
    j->n = n;
    j->pos = 2;
    int frame = 0, ended = 0, scans = 0;
    while (!j->err && !ended) {
        /* the next marker (fill bytes 0xff allowed) */
        while (j->pos < n && p[j->pos] != 0xff)
            j->pos++;
        while (j->pos < n && p[j->pos] == 0xff)
            j->pos++;
        if (j->pos >= n) {
            if (!scans)
                j->err = "the file ends before its picture";
            break;
        }
        int m = p[j->pos++];
        if (m == 0xd9) {
            ended = 1;
            break;
        }
        if (m == 0xd8 || (m >= 0xd0 && m <= 0xd7) || m == 0x01)
            continue;
        if (n - j->pos < 2) {
            j->err = "the file ends inside a marker";
            break;
        }
        size_t len = (size_t)u16(p + j->pos);
        if (len < 2 || len > n - j->pos) {
            j->err = "a marker runs past the end of the file";
            break;
        }
        const uint8_t *d = p + j->pos + 2;
        size_t dl = len - 2;
        switch (m) {
        case 0xc0:
        case 0xc1:
        case 0xc2:
            if (frame) {
                j->err = "two frames in one file";
                break;
            }
            frame = 1;
            j->progressive = m == 0xc2;
            sof(j, d, dl);
            break;
        case 0xc4:
            dht(j, d, dl);
            break;
        case 0xdb:
            dqt(j, d, dl);
            break;
        case 0xdd:
            j->restart = dl >= 2 ? u16(d) : 0;
            break;
        case 0xee: /* Adobe: whether YCbCr is used */
            if (dl >= 12 && !memcmp(d, "Adobe", 5)) {
                j->adobe = 1;
                j->transform = d[11];
            }
            break;
        case 0xda:
            if (!frame) {
                j->err = "a scan before the frame";
                break;
            }
            j->pos += 2;
            sos(j, d, dl);
            scans++;
            continue; /* sos moved pos past the scan's data */
        default:
            if ((m >= 0xc3 && m <= 0xcf) && m != 0xc4 && m != 0xc8 && m != 0xcc)
                j->err = "a JPEG coding this does not read (lossless or "
                         "arithmetic)";
        }
        j->pos += len;
    }
    int rc = -1;
    if (!j->err && !scans)
        j->err = "no picture in the file";
    if (!j->err) {
        rc = finish(j, im);
        if (rc != 0)
            j->err = "out of memory";
    }
    if (j->err) {
        snprintf(why, wn, "JPEG: %s", j->err);
        free(im->rgb);
        memset(im, 0, sizeof(*im));
        rc = -1;
    }
    for (int i = 0; i < 3; i++) {
        free(j->c[i].coef);
        free(j->c[i].plane);
    }
    free(j);
    return rc;
}
