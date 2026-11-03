/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_vorbis.c - Vorbis I (the Xiph.Org specification; audio_ogg.h): the
 * three headers - identification, comments (skipped), setup with its
 * codebooks (Huffman codes from their lengths, ordered or sparse, vector
 * lookups of type 1 and 2), floors (type 0, LSP, and type 1, line
 * segments), residues (types 0, 1 and 2), mappings with their channel
 * coupling and modes - then every audio packet: its mode and window, the
 * floor and residue of each channel, the inverse coupling, the floor
 * times the residue, the inverse MDCT (audio_fft.c), the window, the
 * overlap with the packet before; the start and the end trimmed by the
 * granule positions. Bits are read least significant first. The channels
 * are averaged.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_fft.h"
#include "audio_ogg.h"

/* ---- bits, least significant first ---- */

struct vbr {
    const uint8_t *p;
    size_t n, pos;
    int eop; /* read past the end of the packet */
};

static uint32_t vb(struct vbr *b, int k)
{
    if (!k)
        return 0;
    if (b->pos + (size_t)k > b->n * 8) {
        b->eop = 1;
        b->pos = b->n * 8;
        return 0;
    }
    uint64_t w = 0;
    size_t at = b->pos >> 3;
    for (int i = 7; i >= 0; i--)
        w = w << 8 | (at + (size_t)i < b->n ? b->p[at + (size_t)i] : 0);
    uint32_t v =
        (uint32_t)(w >> (b->pos & 7)) & (k == 32 ? 0xFFFFFFFFu : (1u << k) - 1);
    b->pos += (size_t)k;
    return v;
}

static int ilog(uint32_t v)
{
    int r = 0;
    while (v) {
        r++;
        v >>= 1;
    }
    return r;
}

static float f32_unpack(uint32_t x)
{
    double m = x & 0x1FFFFF;
    int e = (int)((x & 0x7FE00000u) >> 21);
    if (x & 0x80000000u)
        m = -m;
    return (float)ldexp(m, e - 788);
}

/* ---- codebooks ---- */

#define LUT 10

struct cb {
    int dim, entries, used;
    uint8_t *len;
    uint32_t *code;      /* the codewords bit-reversed, as read */
    int *longs, n_longs; /* entries with codes longer than LUT bits */
    int16_t lut[1 << LUT];
    float *vq; /* entries x dim */
};

static void cb_free(struct cb *c)
{
    free(c->len);
    free(c->code);
    free(c->longs);
    free(c->vq);
}

static uint32_t rev(uint32_t v, int n)
{
    uint32_t r = 0;
    for (int i = 0; i < n; i++)
        r = r << 1 | (v >> i & 1);
    return r;
}

/* the codewords from the lengths, as the specification assigns them */
static int cb_codes(struct cb *c)
{
    uint32_t marker[33] = {0};
    c->code = calloc((size_t)c->entries, sizeof *c->code);
    c->longs = malloc(sizeof(int) * (size_t)c->entries);
    if (!c->code || !c->longs)
        return -1;
    for (int i = 0; i < 1 << LUT; i++)
        c->lut[i] = -1;
    int used = 0, only = -1;
    for (int i = 0; i < c->entries; i++)
        if (c->len[i]) {
            used++;
            only = i;
        }
    c->used = used;
    if (used == 1) { /* one entry: a code of one bit, 0 */
        c->len[only] = 1;
        c->code[only] = 0;
        for (int i = 0; i < 1 << LUT; i += 2)
            c->lut[i] = (int16_t)only;
        return 0;
    }
    for (int i = 0; i < c->entries; i++) {
        int l = c->len[i];
        if (!l)
            continue;
        uint32_t entry = marker[l];
        if (l < 32 && (entry >> l)) /* overspecified */
            return -1;
        c->code[i] = rev(entry, l);
        for (int j = l; j > 0; j--) {
            if (marker[j] & 1) {
                marker[j] = j == 1 ? marker[1] + 1 : marker[j - 1] << 1;
                break;
            }
            marker[j]++;
        }
        for (int j = l + 1; j < 33; j++) {
            if ((marker[j] >> 1) == entry) {
                entry = marker[j];
                marker[j] = marker[j - 1] << 1;
            } else {
                break;
            }
        }
        if (l <= LUT) {
            for (uint32_t hi = 0; hi < 1u << (LUT - l); hi++)
                c->lut[c->code[i] | hi << l] = (int16_t)i;
        } else {
            c->longs[c->n_longs++] = i;
        }
    }
    return 0;
}

static int cb_read(const struct cb *c, struct vbr *b)
{
    if (!c->used) {
        b->eop = 1;
        return -1;
    }
    uint32_t peek = 0;
    size_t left = b->n * 8 - b->pos;
    int k = left < 32 ? (int)left : 32;
    struct vbr t = *b;
    peek = vb(&t, k);
    int e = c->lut[peek & ((1u << LUT) - 1)];
    if (e >= 0) {
        if ((size_t)c->len[e] > left) {
            b->eop = 1;
            return -1;
        }
        b->pos += c->len[e];
        return e;
    }
    for (int i = 0; i < c->n_longs; i++) {
        int x = c->longs[i], l = c->len[x];
        if (l <= k &&
            (peek & (l == 32 ? 0xFFFFFFFFu : (1u << l) - 1)) == c->code[x]) {
            b->pos += (size_t)l;
            return x;
        }
    }
    b->eop = 1;
    return -1;
}

static long lookup1(int entries, int dim)
{
    long r = (long)floor(exp(log((double)entries) / dim));
    while (pow((double)r + 1, dim) <= entries)
        r++;
    while (r > 0 && pow((double)r, dim) > entries)
        r--;
    return r;
}

static int cb_setup(struct cb *c, struct vbr *b)
{
    if (vb(b, 24) != 0x564342)
        return -1;
    c->dim = (int)vb(b, 16);
    c->entries = (int)vb(b, 24);
    if (!c->dim || !c->entries || c->entries > (1 << 24))
        return -1;
    c->len = calloc((size_t)c->entries, 1);
    if (!c->len)
        return -1;
    if (vb(b, 1)) { /* ordered */
        int cur = 0, l = (int)vb(b, 5) + 1;
        while (cur < c->entries) {
            int num = (int)vb(b, ilog((uint32_t)(c->entries - cur)));
            if (cur + num > c->entries || l > 32)
                return -1;
            memset(c->len + cur, l, (size_t)num);
            cur += num;
            l++;
            if (b->eop)
                return -1;
        }
    } else {
        int sparse = (int)vb(b, 1);
        for (int i = 0; i < c->entries; i++)
            c->len[i] = !sparse || vb(b, 1) ? (uint8_t)(vb(b, 5) + 1) : 0;
    }
    if (b->eop || cb_codes(c))
        return -1;
    int type = (int)vb(b, 4);
    if (type == 0)
        return b->eop ? -1 : 0;
    if (type > 2)
        return -1;
    float min = f32_unpack(vb(b, 32)), delta = f32_unpack(vb(b, 32));
    int bits = (int)vb(b, 4) + 1, seq = (int)vb(b, 1);
    long nv =
        type == 1 ? lookup1(c->entries, c->dim) : (long)c->entries * c->dim;
    if (nv <= 0 || nv > (1 << 26))
        return -1;
    uint32_t *mul = malloc(sizeof(uint32_t) * (size_t)nv);
    c->vq = malloc(sizeof(float) * (size_t)c->entries * (size_t)c->dim);
    if (!mul || !c->vq) {
        free(mul);
        return -1;
    }
    for (long i = 0; i < nv; i++)
        mul[i] = vb(b, bits);
    for (int e = 0; e < c->entries; e++) {
        float last = 0;
        long div = 1;
        for (int i = 0; i < c->dim; i++) {
            long off = type == 1 ? (e / div) % nv : (long)e * c->dim + i;
            float v = (float)mul[off] * delta + min + last;
            if (seq)
                last = v;
            c->vq[(size_t)e * c->dim + i] = v;
            div *= nv;
        }
    }
    free(mul);
    return b->eop ? -1 : 0;
}

/* ---- the setup ---- */

struct floor_t {
    int type;
    /* type 0 */
    int order, rate, bark_size, amp_bits, amp_off, nbooks, books[16];
    /* type 1 */
    int parts, pclass[32], cdim[16], csub[16], cmaster[16], sbooks[16][8];
    int mult, values, x[65], sorted[65], low[65], high[65];
};

struct residue {
    int type, begin, end, psize, classes, classbook;
    int books[64][8];
};

struct mapping {
    int submaps, steps, mag[256], ang[256], mux[256];
    int sfloor[16], sres[16];
};

struct vorbis {
    int ch;
    uint32_t rate;
    int bs[2];
    int ncb, nfloor, nres, nmap, nmode;
    struct cb *cb;
    struct floor_t *floor;
    struct residue *res;
    struct mapping *map;
    int mode_block[64], mode_map[64];
    struct aimdct *mdct[2];
    float *win[2][2][2]; /* [blockflag][prev][next] */
};

static int floor_setup(struct vorbis *v, struct floor_t *f, struct vbr *b)
{
    f->type = (int)vb(b, 16);
    if (f->type == 0) {
        f->order = (int)vb(b, 8);
        f->rate = (int)vb(b, 16);
        f->bark_size = (int)vb(b, 16);
        f->amp_bits = (int)vb(b, 6);
        f->amp_off = (int)vb(b, 8);
        f->nbooks = (int)vb(b, 4) + 1;
        for (int i = 0; i < f->nbooks; i++)
            if ((f->books[i] = (int)vb(b, 8)) >= v->ncb)
                return -1;
        return f->order < 1 || !f->bark_size || !f->rate ? -1 : 0;
    }
    if (f->type != 1)
        return -1;
    f->parts = (int)vb(b, 5);
    int maxc = -1;
    for (int i = 0; i < f->parts; i++) {
        f->pclass[i] = (int)vb(b, 4);
        if (f->pclass[i] > maxc)
            maxc = f->pclass[i];
    }
    for (int c = 0; c <= maxc; c++) {
        f->cdim[c] = (int)vb(b, 3) + 1;
        f->csub[c] = (int)vb(b, 2);
        if (f->csub[c] && (f->cmaster[c] = (int)vb(b, 8)) >= v->ncb)
            return -1;
        for (int j = 0; j < 1 << f->csub[c]; j++)
            if ((f->sbooks[c][j] = (int)vb(b, 8) - 1) >= v->ncb)
                return -1;
    }
    f->mult = (int)vb(b, 2) + 1;
    int rb = (int)vb(b, 4);
    f->x[0] = 0;
    f->x[1] = 1 << rb;
    f->values = 2;
    for (int i = 0; i < f->parts; i++)
        for (int j = 0; j < f->cdim[f->pclass[i]]; j++) {
            if (f->values >= 65)
                return -1;
            f->x[f->values++] = (int)vb(b, rb);
        }
    /* the order of the X values, and each one's neighbours before it */
    for (int i = 0; i < f->values; i++)
        f->sorted[i] = i;
    for (int i = 1; i < f->values; i++)
        for (int j = i; j > 0 && f->x[f->sorted[j - 1]] > f->x[f->sorted[j]];
             j--) {
            int t = f->sorted[j];
            f->sorted[j] = f->sorted[j - 1];
            f->sorted[j - 1] = t;
        }
    for (int i = 2; i < f->values; i++) {
        int lo = 0, hi = 1;
        for (int j = 0; j < i; j++) {
            if (f->x[j] < f->x[i] && f->x[j] > f->x[lo])
                lo = j;
            if (f->x[j] > f->x[i] && f->x[j] < f->x[hi])
                hi = j;
        }
        f->low[i] = lo;
        f->high[i] = hi;
    }
    return b->eop ? -1 : 0;
}

static int residue_setup(struct vorbis *v, struct residue *r, struct vbr *b)
{
    r->type = (int)vb(b, 16);
    if (r->type > 2)
        return -1;
    r->begin = (int)vb(b, 24);
    r->end = (int)vb(b, 24);
    r->psize = (int)vb(b, 24) + 1;
    r->classes = (int)vb(b, 6) + 1;
    r->classbook = (int)vb(b, 8);
    if (r->classbook >= v->ncb)
        return -1;
    int cascade[64];
    for (int i = 0; i < r->classes; i++) {
        int low = (int)vb(b, 3), high = vb(b, 1) ? (int)vb(b, 5) : 0;
        cascade[i] = high * 8 + low;
    }
    for (int i = 0; i < r->classes; i++)
        for (int j = 0; j < 8; j++) {
            r->books[i][j] = -1;
            if (cascade[i] & (1 << j))
                if ((r->books[i][j] = (int)vb(b, 8)) >= v->ncb)
                    return -1;
        }
    return b->eop ? -1 : 0;
}

static int mapping_setup(struct vorbis *v, struct mapping *m, struct vbr *b)
{
    if (vb(b, 16) != 0)
        return -1;
    m->submaps = vb(b, 1) ? (int)vb(b, 4) + 1 : 1;
    m->steps = 0;
    if (vb(b, 1)) {
        m->steps = (int)vb(b, 8) + 1;
        int bits = ilog((uint32_t)v->ch - 1);
        for (int i = 0; i < m->steps; i++) {
            m->mag[i] = (int)vb(b, bits);
            m->ang[i] = (int)vb(b, bits);
            if (m->mag[i] == m->ang[i] || m->mag[i] >= v->ch ||
                m->ang[i] >= v->ch)
                return -1;
        }
    }
    if (vb(b, 2))
        return -1;
    for (int c = 0; c < v->ch; c++) {
        m->mux[c] = m->submaps > 1 ? (int)vb(b, 4) : 0;
        if (m->mux[c] >= m->submaps)
            return -1;
    }
    for (int i = 0; i < m->submaps; i++) {
        vb(b, 8);
        m->sfloor[i] = (int)vb(b, 8);
        m->sres[i] = (int)vb(b, 8);
        if (m->sfloor[i] >= v->nfloor || m->sres[i] >= v->nres)
            return -1;
    }
    return b->eop ? -1 : 0;
}

static int setup(struct vorbis *v, const uint8_t *p, size_t n)
{
    struct vbr b = {.p = p, .n = n, .pos = 7 * 8};
    if (n < 7 || p[0] != 5 || memcmp(p + 1, "vorbis", 6))
        return -1;
    v->ncb = (int)vb(&b, 8) + 1;
    v->cb = calloc((size_t)v->ncb, sizeof *v->cb);
    if (!v->cb)
        return -1;
    for (int i = 0; i < v->ncb; i++)
        if (cb_setup(&v->cb[i], &b))
            return -1;
    int times = (int)vb(&b, 6) + 1;
    for (int i = 0; i < times; i++)
        if (vb(&b, 16))
            return -1;
    v->nfloor = (int)vb(&b, 6) + 1;
    v->floor = calloc((size_t)v->nfloor, sizeof *v->floor);
    if (!v->floor)
        return -1;
    for (int i = 0; i < v->nfloor; i++)
        if (floor_setup(v, &v->floor[i], &b))
            return -1;
    v->nres = (int)vb(&b, 6) + 1;
    v->res = calloc((size_t)v->nres, sizeof *v->res);
    if (!v->res)
        return -1;
    for (int i = 0; i < v->nres; i++)
        if (residue_setup(v, &v->res[i], &b))
            return -1;
    v->nmap = (int)vb(&b, 6) + 1;
    v->map = calloc((size_t)v->nmap, sizeof *v->map);
    if (!v->map)
        return -1;
    for (int i = 0; i < v->nmap; i++)
        if (mapping_setup(v, &v->map[i], &b))
            return -1;
    v->nmode = (int)vb(&b, 6) + 1;
    for (int i = 0; i < v->nmode; i++) {
        v->mode_block[i] = (int)vb(&b, 1);
        if (vb(&b, 16) || vb(&b, 16))
            return -1;
        v->mode_map[i] = (int)vb(&b, 8);
        if (v->mode_map[i] >= v->nmap)
            return -1;
    }
    return b.eop || !vb(&b, 1) ? -1 : 0;
}

/* ---- the floors ---- */

static float inv_db(int y)
{
    if (y < 0)
        y = 0;
    if (y > 255)
        y = 255;
    return (float)pow(10.0, (y - 255) * 7.0 / 256.0);
}

static int render_point(int x0, int y0, int x1, int y1, int x)
{
    int dy = y1 - y0, adx = x1 - x0, ady = dy < 0 ? -dy : dy;
    int off = ady * (x - x0) / adx;
    return dy < 0 ? y0 - off : y0 + off;
}

static void render_line(int x0, int y0, int x1, int y1, int *v, int n)
{
    int dy = y1 - y0, adx = x1 - x0, ady = dy < 0 ? -dy : dy;
    int base = dy / adx, sy = dy < 0 ? base - 1 : base + 1;
    int err = 0, y = y0;
    ady -= (base < 0 ? -base : base) * adx;
    if (x0 < n)
        v[x0] = y;
    for (int x = x0 + 1; x < x1 && x < n; x++) {
        err += ady;
        if (err >= adx) {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }
        v[x] = y;
    }
}

/* floor 1 of a channel into out (n / 2 values): 0, or 1 when the channel
   is unused in this packet */
static int floor1(struct vorbis *v, const struct floor_t *f, struct vbr *b,
                  int half, float *out, int *iv)
{
    if (!vb(b, 1))
        return 1;
    static const int RANGE[4] = {256, 128, 86, 64};
    int range = RANGE[f->mult - 1], bits = ilog((uint32_t)range - 1);
    int y[65], fy[65], step2[65];
    y[0] = (int)vb(b, bits);
    y[1] = (int)vb(b, bits);
    int off = 2;
    for (int i = 0; i < f->parts; i++) {
        int c = f->pclass[i], cdim = f->cdim[c], cbits = f->csub[c];
        int csub = (1 << cbits) - 1, cval = 0;
        if (cbits) {
            cval = cb_read(&v->cb[f->cmaster[c]], b);
            if (cval < 0)
                return 1;
        }
        for (int j = 0; j < cdim; j++) {
            int book = f->sbooks[c][cval & csub];
            cval >>= cbits;
            int val = 0;
            if (book >= 0 && (val = cb_read(&v->cb[book], b)) < 0)
                return 1;
            y[off++] = val;
        }
    }
    if (b->eop)
        return 1;
    fy[0] = y[0];
    fy[1] = y[1];
    step2[0] = step2[1] = 1;
    for (int i = 2; i < f->values; i++) {
        int lo = f->low[i], hi = f->high[i];
        int pred = render_point(f->x[lo], fy[lo], f->x[hi], fy[hi], f->x[i]);
        int val = y[i], hroom = range - pred, lroom = pred;
        int room = (hroom < lroom ? hroom : lroom) * 2;
        if (val) {
            step2[lo] = step2[hi] = step2[i] = 1;
            if (val >= room)
                fy[i] =
                    hroom > lroom ? val - lroom + pred : pred - val + hroom - 1;
            else
                fy[i] = val & 1 ? pred - (val + 1) / 2 : pred + val / 2;
        } else {
            step2[i] = 0;
            fy[i] = pred;
        }
    }
    int lx = 0, ly = fy[0] * f->mult, hx = 0, hy = 0;
    for (int k = 1; k < f->values; k++) {
        int i = f->sorted[k];
        if (!step2[i])
            continue;
        hy = fy[i] * f->mult;
        hx = f->x[i];
        render_line(lx, ly, hx, hy, iv, half);
        lx = hx;
        ly = hy;
    }
    if (hx < half)
        render_line(hx, hy, half, hy, iv, half);
    for (int i = 0; i < half; i++)
        out[i] = inv_db(iv[i]);
    return 0;
}

static double bark(double x)
{
    return 13.1 * atan(.00074 * x) + 2.24 * atan(.0000000185 * x * x) +
           .0001 * x;
}

static int floor0(struct vorbis *v, const struct floor_t *f, struct vbr *b,
                  int half, float *out)
{
    int amp = (int)vb(b, f->amp_bits);
    if (amp <= 0 || b->eop)
        return 1;
    int book = (int)vb(b, ilog((uint32_t)f->nbooks));
    if (book >= f->nbooks)
        return 1;
    const struct cb *c = &v->cb[f->books[book]];
    if (!c->vq)
        return 1;
    float coef[256 + 16];
    int n = 0;
    float last = 0;
    while (n < f->order) {
        int e = cb_read(c, b);
        if (e < 0)
            return 1;
        for (int k = 0; k < c->dim && n < f->order + 16; k++)
            coef[n++] = c->vq[(size_t)e * c->dim + k] + last;
        last = coef[n - 1];
    }
    double nyq = bark(.5 * f->rate);
    int prev = -2;
    float val = 0;
    for (int i = 0; i < half; i++) {
        int map = (int)floor(bark((double)f->rate * i / (2.0 * half)) *
                             f->bark_size / nyq);
        if (map > f->bark_size - 1)
            map = f->bark_size - 1;
        if (map != prev) {
            double w = M_PI * map / f->bark_size, cw = cos(w), p, q;
            if (f->order & 1) {
                p = 1 - cw * cw;
                q = .25;
                for (int j = 0; j < (f->order - 1) / 2; j++)
                    p *= 4 * (cos(coef[2 * j + 1]) - cw) *
                         (cos(coef[2 * j + 1]) - cw);
                for (int j = 0; j < (f->order + 1) / 2; j++)
                    q *= 4 * (cos(coef[2 * j]) - cw) * (cos(coef[2 * j]) - cw);
            } else {
                p = (1 - cw) / 2;
                q = (1 + cw) / 2;
                for (int j = 0; j < f->order / 2; j++) {
                    p *= 4 * (cos(coef[2 * j + 1]) - cw) *
                         (cos(coef[2 * j + 1]) - cw);
                    q *= 4 * (cos(coef[2 * j]) - cw) * (cos(coef[2 * j]) - cw);
                }
            }
            val = (float)exp(
                .11512925 *
                (amp * f->amp_off / (((1 << f->amp_bits) - 1) * sqrt(p + q)) -
                 f->amp_off));
            prev = map;
        }
        out[i] = val;
    }
    return 0;
}

/* ---- the residues ---- */

static void residue(struct vorbis *v, const struct residue *r, struct vbr *b,
                    float **vec, const int *dnd, int nch, int half, int *cls)
{
    int size = r->type == 2 ? half * nch : half;
    int begin = r->begin < size ? r->begin : size;
    int end = r->end < size ? r->end : size;
    int ntoread = end - begin, parts = ntoread / r->psize;
    const struct cb *ccb = &v->cb[r->classbook];
    int cpw = ccb->dim;
    if (ntoread <= 0)
        return;
    float *tmp = NULL;
    float *vv[256];
    int dd[256], nv = nch;
    if (r->type == 2) { /* the channels interleaved into one vector */
        int any = 0;
        for (int j = 0; j < nch; j++)
            any |= !dnd[j];
        if (!any)
            return;
        tmp = calloc((size_t)size, sizeof(float));
        if (!tmp)
            return;
        vv[0] = tmp;
        dd[0] = 0;
        nv = 1;
    } else {
        for (int j = 0; j < nch; j++) {
            vv[j] = vec[j];
            dd[j] = dnd[j];
        }
    }
    for (int pass = 0; pass < 8; pass++) {
        int pc = 0;
        while (pc < parts) {
            if (pass == 0)
                for (int j = 0; j < nv; j++) {
                    if (dd[j])
                        continue;
                    int t = cb_read(ccb, b);
                    if (t < 0)
                        goto done;
                    for (int i = cpw - 1; i >= 0; i--) {
                        if (i + pc < parts)
                            cls[j * parts + i + pc] = t % r->classes;
                        t /= r->classes;
                    }
                }
            for (int i = 0; i < cpw && pc < parts; i++, pc++)
                for (int j = 0; j < nv; j++) {
                    if (dd[j])
                        continue;
                    int book = r->books[cls[j * parts + pc]][pass];
                    if (book < 0)
                        continue;
                    const struct cb *c = &v->cb[book];
                    if (!c->vq)
                        goto done;
                    int off = begin + pc * r->psize, dim = c->dim;
                    if (r->type == 0) {
                        int step = r->psize / dim;
                        for (int s = 0; s < step; s++) {
                            int e = cb_read(c, b);
                            if (e < 0)
                                goto done;
                            for (int k = 0; k < dim; k++)
                                vv[j][off + s + k * step] +=
                                    c->vq[(size_t)e * dim + k];
                        }
                    } else {
                        for (int s = 0; s < r->psize;) {
                            int e = cb_read(c, b);
                            if (e < 0)
                                goto done;
                            for (int k = 0; k < dim && s < r->psize; k++, s++)
                                vv[j][off + s] += c->vq[(size_t)e * dim + k];
                        }
                    }
                }
        }
    }
done:
    if (tmp) {
        for (int i = 0; i < half; i++)
            for (int j = 0; j < nch; j++)
                vec[j][i] = tmp[i * nch + j];
        free(tmp);
    }
}

/* ---- windows ---- */

static float *make_window(int n, int bs0, int prev, int next, int longb)
{
    float *w = malloc(sizeof(float) * (size_t)n);
    if (!w)
        return NULL;
    int ls, le, ln, rs, re, rn;
    if (longb && !prev) {
        ls = n / 4 - bs0 / 4;
        le = n / 4 + bs0 / 4;
        ln = bs0 / 2;
    } else {
        ls = 0;
        le = n / 2;
        ln = n / 2;
    }
    if (longb && !next) {
        rs = n * 3 / 4 - bs0 / 4;
        re = n * 3 / 4 + bs0 / 4;
        rn = bs0 / 2;
    } else {
        rs = n / 2;
        re = n;
        rn = n / 2;
    }
    for (int i = 0; i < n; i++) {
        double x;
        if (i < ls || i >= re)
            x = 0;
        else if (i < le) {
            double s = sin((i - ls + .5) / ln * M_PI / 2);
            x = sin(M_PI / 2 * s * s);
        } else if (i < rs)
            x = 1;
        else {
            double s = sin((i - rs + .5) / rn * M_PI / 2 + M_PI / 2);
            x = sin(M_PI / 2 * s * s);
        }
        w[i] = (float)x;
    }
    return w;
}

static void vorbis_free(struct vorbis *v)
{
    for (int i = 0; v->cb && i < v->ncb; i++)
        cb_free(&v->cb[i]);
    free(v->cb);
    free(v->floor);
    free(v->res);
    free(v->map);
    for (int i = 0; i < 2; i++)
        aimdct_free(v->mdct[i]);
    for (int i = 0; i < 8; i++)
        free(v->win[i >> 2][(i >> 1) & 1][i & 1]);
}

/* ---- the stream ---- */

int janas_vorbis_stream(const struct ogg_packets *o, struct audio_out *ao,
                        char *why, size_t wn)
{
    struct vorbis v = {0};
    const uint8_t *id = o->data + o->off[0];
    int rc = -1;
    float *out = NULL, **prev = NULL, **vec = NULL, **pcm = NULL;
    float *floorv = NULL;
    int *iv = NULL, *cls = NULL, *noresid = NULL, *dnd = NULL;
    if (o->count < 3 || o->len[0] < 30 || id[0] != 1 ||
        memcmp(id + 1, "vorbis", 6) || id[7] | id[8] | id[9] | id[10]) {
        snprintf(why, wn, "a Vorbis stream without its headers");
        return -1;
    }
    v.ch = id[11];
    v.rate = (uint32_t)id[12] | (uint32_t)id[13] << 8 | (uint32_t)id[14] << 16 |
             (uint32_t)id[15] << 24;
    v.bs[0] = 1 << (id[28] & 15);
    v.bs[1] = 1 << (id[28] >> 4);
    if (!v.ch || !v.rate || v.bs[0] < 64 || v.bs[1] > 8192 ||
        v.bs[0] > v.bs[1] || !(id[29] & 1)) {
        snprintf(why, wn, "a Vorbis stream of sizes out of range");
        return -1;
    }
    if (setup(&v, o->data + o->off[2], o->len[2])) {
        snprintf(why, wn, "a Vorbis stream whose setup does not read");
        goto end;
    }
    for (int i = 0; i < 2; i++)
        if (!(v.mdct[i] = aimdct_new(v.bs[i])))
            goto oom;
    for (int bl = 0; bl < 2; bl++)
        for (int p = 0; p < 2; p++)
            for (int nx = 0; nx < 2; nx++)
                if (!(v.win[bl][p][nx] =
                          make_window(v.bs[bl], v.bs[0], p, nx, bl)))
                    goto oom;
    int maxn = v.bs[1], half = maxn / 2;
    prev = calloc((size_t)v.ch, sizeof *prev);
    vec = calloc((size_t)v.ch, sizeof *vec);
    pcm = calloc((size_t)v.ch, sizeof *pcm);
    floorv = malloc(sizeof(float) * (size_t)half);
    iv = malloc(sizeof(int) * (size_t)half);
    cls = malloc(sizeof(int) * (size_t)half * (size_t)v.ch);
    noresid = malloc(sizeof(int) * (size_t)v.ch);
    dnd = malloc(sizeof(int) * (size_t)v.ch);
    if (!prev || !vec || !pcm || !floorv || !iv || !cls || !noresid || !dnd)
        goto oom;
    for (int c = 0; c < v.ch; c++)
        if (!(prev[c] = calloc((size_t)maxn, sizeof(float))) ||
            !(vec[c] = malloc(sizeof(float) * (size_t)half)) ||
            !(pcm[c] = malloc(sizeof(float) * (size_t)maxn)))
            goto oom;
    /* a block gives at most half the larger size */
    if (!(out = malloc(sizeof(float) * (size_t)maxn)))
        goto oom;
    size_t got = 0;
    ao->rate = v.rate;
    int pn = 0; /* the previous block's size, 0 before the first */
    int64_t first_granule = -1;
    size_t first_page_end = 0; /* samples out when its page ended */
    int64_t last_granule = -1;
    int pages = 0; /* with a granule position, after the first */
    for (size_t pk = 3; pk < o->count; pk++) {
        struct vbr b = {.p = o->data + o->off[pk], .n = o->len[pk]};
        if (!o->len[pk] || vb(&b, 1))
            continue; /* not audio */
        int mode = (int)vb(&b, ilog((uint32_t)v.nmode - 1));
        if (mode >= v.nmode || b.eop)
            continue;
        int bl = v.mode_block[mode], n = v.bs[bl], pf = 0, nf = 0;
        if (bl) {
            pf = (int)vb(&b, 1);
            nf = (int)vb(&b, 1);
        }
        const struct mapping *m = &v.map[v.mode_map[mode]];
        int h2 = n / 2;
        for (int c = 0; c < v.ch; c++) {
            const struct floor_t *f = &v.floor[m->sfloor[m->mux[c]]];
            float *fl = pcm[c]; /* the floor curve, kept here a while */
            noresid[c] = f->type == 1 ? floor1(&v, f, &b, h2, fl, iv)
                                      : floor0(&v, f, &b, h2, fl);
            memset(vec[c], 0, sizeof(float) * (size_t)h2);
        }
        /* a channel whose floor is unused stays silent, though coupling
           may have its residue read */
        memcpy(dnd, noresid, sizeof(int) * (size_t)v.ch);
        for (int s = 0; s < m->steps; s++)
            if (!noresid[m->mag[s]] || !noresid[m->ang[s]])
                noresid[m->mag[s]] = noresid[m->ang[s]] = 0;
        for (int sm = 0; sm < m->submaps; sm++) {
            float *cv[256];
            int cd[256], k = 0;
            for (int c = 0; c < v.ch; c++)
                if (m->mux[c] == sm) {
                    cv[k] = vec[c];
                    cd[k++] = noresid[c];
                }
            residue(&v, &v.res[m->sres[sm]], &b, cv, cd, k, h2, cls);
        }
        for (int s = m->steps - 1; s >= 0; s--) {
            float *M = vec[m->mag[s]], *A = vec[m->ang[s]];
            for (int i = 0; i < h2; i++) {
                float mg = M[i], an = A[i], nm, na;
                if (mg > 0) {
                    if (an > 0) {
                        nm = mg;
                        na = mg - an;
                    } else {
                        na = mg;
                        nm = mg + an;
                    }
                } else {
                    if (an > 0) {
                        nm = mg;
                        na = mg + an;
                    } else {
                        na = mg;
                        nm = mg - an;
                    }
                }
                M[i] = nm;
                A[i] = na;
            }
        }
        for (int c = 0; c < v.ch; c++) {
            float *fl = pcm[c];
            if (dnd[c])
                memset(vec[c], 0, sizeof(float) * (size_t)h2);
            else
                for (int i = 0; i < h2; i++)
                    vec[c][i] *= fl[i];
            aimdct_run(v.mdct[bl], vec[c], pcm[c]);
            const float *w = v.win[bl][pf][nf];
            for (int i = 0; i < n; i++)
                pcm[c][i] *= w[i];
        }
        /* from the previous block's centre to this one's */
        if (pn) {
            int len = pn / 4 + n / 4;
            for (int k = 0; k < len; k++) {
                double s = 0;
                int pi = pn / 2 + k, ci = n / 4 - pn / 4 + k;
                for (int c = 0; c < v.ch; c++)
                    s += (pi < pn ? prev[c][pi] : 0) +
                         (ci >= 0 && ci < n ? pcm[c][ci] : 0);
                out[k] = (float)(s / v.ch);
            }
            if (audio_put(ao, out, (size_t)len, why, wn) != 0)
                goto end;
            got += (size_t)len;
        }
        for (int c = 0; c < v.ch; c++)
            memcpy(prev[c], pcm[c], sizeof(float) * (size_t)n);
        pn = n;
        if (o->granule[pk] >= 0) {
            if (first_granule < 0) {
                first_granule = o->granule[pk];
                first_page_end = got;
            } else {
                pages++;
            }
            last_granule = o->granule[pk];
        }
    }
    /* the start: a first page ending before the samples it gave began
       later than zero; the end: the last granule position. A stream of a
       single page is only trimmed at its end, as libvorbisfile does. */
    size_t skip = 0, total = got;
    if (first_granule >= 0 && pages > 0 &&
        (int64_t)first_page_end > first_granule)
        skip = first_page_end - (size_t)first_granule;
    if (pages == 0 && last_granule >= 0) {
        if ((uint64_t)last_granule < total)
            total = (size_t)last_granule;
    } else if (last_granule >= 0 && first_granule >= 0) {
        int64_t want = last_granule - (first_granule - (int64_t)first_page_end +
                                       (int64_t)skip);
        if (want >= 0 && (size_t)want + skip < total)
            total = (size_t)want + skip;
    }
    if (skip > total)
        skip = total;
    if (total - skip == 0) {
        snprintf(why, wn, "a Vorbis stream without sound");
        goto end;
    }
    ao->skip = skip;
    ao->total = total;
    rc = 0;
    goto end;
oom:
    snprintf(why, wn, "out of memory");
end:
    free(out);
    for (int c = 0; c < v.ch; c++) {
        if (prev)
            free(prev[c]);
        if (vec)
            free(vec[c]);
        if (pcm)
            free(pcm[c]);
    }
    free(prev);
    free(vec);
    free(pcm);
    free(floorv);
    free(iv);
    free(cls);
    free(noresid);
    free(dnd);
    vorbis_free(&v);
    return rc;
}
