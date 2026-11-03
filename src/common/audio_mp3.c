/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_mp3.c - MPEG audio files (audio.h): MPEG-1, 2 and 2.5, Layer III
 * here, Layers I and II in audio_mp2.c, as ISO/IEC 11172-3 and 13818-3
 * describe them, the tables of the standards in audio_mpa_tab.h.
 *
 * Layer III: the side information, the bit reservoir (a frame's granules
 * may begin in the frames before it), the scale factors (MPEG-1's, with the
 * bands they share between granules; MPEG-2's, with the intensity stereo
 * variant), the Huffman-coded big values and quadruples, requantization
 * (|x|^4/3 by the global gain, the scale factors, the pre-emphasis and the
 * subblock gains), mid/side and intensity stereo, the short blocks
 * reordered, the aliasing butterflies, the IMDCT of 36 or three of 12 with
 * their windows and the overlap, the frequency inversion, and the
 * polyphase filter bank. The file: an ID3v2 tag skipped (audio.c), frames
 * found by their sync and checked by the next one, free format, the Xing,
 * Info or VBRI frame skipped and LAME's encoder delay and padding taken off
 * as FFmpeg does (529 samples of the decoder's own delay with them). The
 * channels are averaged.
 */
#include "audio.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_mpa.h"
#include "audio_mpa_tab.h"

/* ---- the header and the synthesis ---- */

int mpa_header(const uint8_t *p, struct mpa_hdr *h)
{
    static const int FREQ[3] = {44100, 48000, 32000};
    if (p[0] != 0xFF || (p[1] & 0xE0) != 0xE0)
        return -1;
    int ver = (p[1] >> 3) & 3, layer = (p[1] >> 1) & 3;
    int br = p[2] >> 4, sr = (p[2] >> 2) & 3;
    /* the emphasis "reserved" (10) is let through: the standard's own
       test streams have it */
    if (ver == 1 || layer == 0 || br == 15 || sr == 3)
        return -1;
    int v25 = ver == 0;
    h->lsf = ver != 3;
    h->layer = 4 - layer;
    h->crc = !(p[1] & 1);
    h->rate_index = sr + 3 * (h->lsf + v25);
    h->rate = FREQ[sr] >> (h->lsf + v25);
    h->bitrate = MPA_BITRATE[h->lsf * 45 + (h->layer - 1) * 15 + br];
    h->pad = (p[2] >> 1) & 1;
    h->mode = p[3] >> 6;
    h->mode_ext = (p[3] >> 4) & 3;
    h->ch = h->mode == MPA_MODE_MONO ? 1 : 2;
    if (!h->bitrate)
        h->size = 0;
    else if (h->layer == 1)
        h->size = ((size_t)12000 * h->bitrate / h->rate + h->pad) * 4;
    else if (h->layer == 2 || !h->lsf)
        h->size = (size_t)144000 * h->bitrate / h->rate + h->pad;
    else
        h->size = (size_t)72000 * h->bitrate / h->rate + h->pad;
    return 0;
}

void mpa_synth_init(struct mpa_synth *s)
{
    memset(s->v, 0, sizeof s->v);
    s->off[0] = s->off[1] = 0;
    for (int i = 0; i < 64; i++)
        for (int k = 0; k < 32; k++)
            s->n[i][k] = (float)cos((16 + i) * (2 * k + 1) * M_PI / 64);
    for (int i = 0; i <= 256; i++)
        s->d[i] = (float)(MPA_WINDOW[i] / 65536.0);
    /* D is odd about 256 but at the multiples of 64 */
    for (int i = 1; i < 256; i++)
        s->d[512 - i] = i % 64 ? -s->d[i] : s->d[i];
}

void mpa_synth(struct mpa_synth *s, int ch, const float *sb, float *pcm)
{
    int off = s->off[ch] = (s->off[ch] - 64) & 1023;
    float *v = s->v[ch];
    for (int i = 0; i < 64; i++) {
        float x = 0;
        for (int k = 0; k < 32; k++)
            x += s->n[i][k] * sb[k];
        v[(off + i) & 1023] = x;
    }
    for (int j = 0; j < 32; j++) {
        float sum = 0;
        for (int i = 0; i < 8; i++)
            sum += s->d[64 * i + j] * v[(off + 128 * i + j) & 1023] +
                   s->d[64 * i + 32 + j] * v[(off + 128 * i + 96 + j) & 1023];
        pcm[j] = sum;
    }
}

/* ---- Layer III ---- */

struct gran {
    int part23, big, gain, sfc, btype, mixed, tsel[3], sbg[3];
    int region[3]; /* the big values' regions end at these lines */
    int preflag, sfscale, c1sel, scfsi;
    int long_end, short_start; /* the bands long and short */
    int sf[40];
};

struct l3 {
    struct mpa_synth syn;
    float pow43[8207];
    uint32_t hleft[1378];  /* the codes left-aligned in 32 bits */
    uint32_t qleft[2][16]; /* the count1 tables', in their order */
    uint8_t qlen[2][16], qval[2][16];
    float cos36[36][18], cos12[12][6];
    float win[4][36]; /* the long windows of block types 0..3 (2: short) */
    float overlap[2][32][18];
    uint8_t res[8192]; /* the main data of the frames before */
    size_t res_n;
    struct gran g[2][2];
    float xr[2][576];
    float sb[2][36][32]; /* the subband samples of a frame */
};

static void l3_init(struct l3 *d)
{
    memset(d, 0, sizeof *d);
    mpa_synth_init(&d->syn);
    for (int i = 0; i < 8207; i++)
        d->pow43[i] = (float)pow(i, 4.0 / 3.0);
    for (int i = 0; i < 1378; i++)
        d->hleft[i] = MPA_HUFF_CODE[i] << (32 - MPA_HUFF_LEN[i]);
    for (int t = 0; t < 2; t++) {
        int order[16];
        for (int v = 0; v < 16; v++)
            order[v] = v;
        /* sorted by their left-aligned codes */
        for (int i = 0; i < 16; i++)
            for (int j = i + 1; j < 16; j++) {
                uint32_t a = (uint32_t)MPA_QUAD_CODE[t * 16 + order[i]]
                             << (32 - MPA_QUAD_LEN[t * 16 + order[i]]);
                uint32_t b = (uint32_t)MPA_QUAD_CODE[t * 16 + order[j]]
                             << (32 - MPA_QUAD_LEN[t * 16 + order[j]]);
                if (b < a) {
                    int x = order[i];
                    order[i] = order[j];
                    order[j] = x;
                }
            }
        for (int i = 0; i < 16; i++) {
            int v = order[i];
            d->qlen[t][i] = MPA_QUAD_LEN[t * 16 + v];
            d->qleft[t][i] = (uint32_t)MPA_QUAD_CODE[t * 16 + v]
                             << (32 - d->qlen[t][i]);
            d->qval[t][i] = (uint8_t)v;
        }
    }
    for (int i = 0; i < 36; i++)
        for (int k = 0; k < 18; k++)
            d->cos36[i][k] =
                (float)cos(M_PI / 72 * (2 * i + 1 + 18) * (2 * k + 1));
    for (int i = 0; i < 12; i++)
        for (int k = 0; k < 6; k++)
            d->cos12[i][k] =
                (float)cos(M_PI / 24 * (2 * i + 1 + 6) * (2 * k + 1));
    for (int i = 0; i < 36; i++) {
        double l = sin(M_PI / 36 * (i + 0.5));
        d->win[0][i] = (float)l;
        d->win[1][i] = (float)(i < 18   ? l
                               : i < 24 ? 1
                               : i < 30 ? sin(M_PI / 12 * (i - 18 + 0.5))
                                        : 0);
        d->win[3][i] = (float)(i < 6    ? 0
                               : i < 12 ? sin(M_PI / 12 * (i - 6 + 0.5))
                               : i < 18 ? 1
                                        : l);
        d->win[2][i] = i < 12 ? (float)sin(M_PI / 12 * (i + 0.5)) : 0;
    }
}

static uint32_t peek32(const struct mpa_br *b)
{
    struct mpa_br c = *b;
    return mpa_bits(&c, 32);
}

/* a pair of big values of table l (1..15 of MPA_HUFF_START) */
static int huff_pair(const struct l3 *d, struct mpa_br *b, int l, int *x,
                     int *y)
{
    uint32_t v = peek32(b);
    int lo = MPA_HUFF_START[l - 1], hi = MPA_HUFF_START[l] - 1;
    while (lo < hi) { /* the last code at or below v */
        int mid = (lo + hi + 1) / 2;
        if (d->hleft[mid] <= v)
            lo = mid;
        else
            hi = mid - 1;
    }
    b->pos += MPA_HUFF_LEN[lo];
    *x = MPA_HUFF_SYM[lo] >> 4;
    *y = MPA_HUFF_SYM[lo] & 15;
    return 0;
}

static int huff_quad(const struct l3 *d, struct mpa_br *b, int t)
{
    uint32_t v = peek32(b);
    int i = 15;
    while (i > 0 && d->qleft[t][i] > v)
        i--;
    b->pos += d->qlen[t][i];
    return d->qval[t][i];
}

static const uint8_t *band_long(const struct mpa_hdr *h)
{
    return MPA_BAND_LONG + 22 * h->rate_index;
}

static const uint8_t *band_short(const struct mpa_hdr *h)
{
    return MPA_BAND_SHORT + 13 * h->rate_index;
}

/* the first line of long band i */
static int long_start(const struct mpa_hdr *h, int i)
{
    int s = 0;
    for (int k = 0; k < i && k < 22; k++)
        s += band_long(h)[k];
    return s;
}

static int side_info(struct mpa_br *b, const struct mpa_hdr *h,
                     struct gran (*g)[2], int *mdb)
{
    int ng = h->lsf ? 1 : 2;
    if (h->lsf) {
        *mdb = (int)mpa_bits(b, 8);
        mpa_bits(b, h->ch);
    } else {
        *mdb = (int)mpa_bits(b, 9);
        mpa_bits(b, h->ch == 2 ? 3 : 5);
        for (int ch = 0; ch < h->ch; ch++) {
            g[0][ch].scfsi = 0;
            g[1][ch].scfsi = (int)mpa_bits(b, 4);
        }
    }
    for (int gr = 0; gr < ng; gr++)
        for (int ch = 0; ch < h->ch; ch++) {
            struct gran *q = &g[gr][ch];
            q->part23 = (int)mpa_bits(b, 12);
            q->big = (int)mpa_bits(b, 9);
            if (q->big > 288)
                return -1;
            q->gain = (int)mpa_bits(b, 8);
            /* mid/side alone: its 1 / sqrt(2) in the gain */
            if ((h->mode == MPA_MODE_JOINT) && (h->mode_ext & 3) == 2)
                q->gain -= 2;
            q->sfc = (int)mpa_bits(b, h->lsf ? 9 : 4);
            int r0, r1;
            if (mpa_bits(b, 1)) {
                q->btype = (int)mpa_bits(b, 2);
                if (!q->btype)
                    return -1;
                q->mixed = (int)mpa_bits(b, 1);
                q->tsel[0] = (int)mpa_bits(b, 5);
                q->tsel[1] = (int)mpa_bits(b, 5);
                q->tsel[2] = 0;
                for (int i = 0; i < 3; i++)
                    q->sbg[i] = (int)mpa_bits(b, 3);
                if (q->btype == 2)
                    r0 = h->rate_index != 8 ? 36 : 72;
                else
                    r0 = h->rate_index <= 2   ? 36
                         : h->rate_index != 8 ? 54
                                              : 108;
                r1 = 576;
            } else {
                q->btype = q->mixed = 0;
                for (int i = 0; i < 3; i++)
                    q->tsel[i] = (int)mpa_bits(b, 5);
                int a1 = (int)mpa_bits(b, 4), a2 = (int)mpa_bits(b, 3);
                r0 = long_start(h, a1 + 1);
                r1 = long_start(h, a1 + a2 + 2 < 22 ? a1 + a2 + 2 : 22);
            }
            int bv = 2 * q->big;
            q->region[0] = r0 < bv ? r0 : bv;
            q->region[1] = r1 < bv ? r1 : bv;
            q->region[2] = bv;
            if (q->btype == 2) {
                q->long_end = q->mixed ? (h->rate_index <= 2 ? 8 : 6) : 0;
                q->short_start = q->mixed ? 3 : 0;
            } else {
                q->long_end = 22;
                q->short_start = 13;
            }
            q->preflag = h->lsf ? 0 : (int)mpa_bits(b, 1);
            q->sfscale = (int)mpa_bits(b, 1);
            q->c1sel = (int)mpa_bits(b, 1);
        }
    return 0;
}

/* MPEG-2's scale factor lengths: v split by the radices n1, n2, n3 (0: no
   such length) */
static void lsf_expand(int *sl, int v, int n1, int n2, int n3)
{
    if (n3) {
        sl[3] = v % n3;
        v /= n3;
    }
    if (n2) {
        sl[2] = v % n2;
        v /= n2;
    }
    sl[1] = v % n1;
    sl[0] = v / n1;
}

static void scale_factors(struct mpa_br *b, const struct mpa_hdr *h,
                          struct gran *q, const struct gran *first, int ch)
{
    int j = 0;
    memset(q->sf, 0, sizeof q->sf);
    if (!h->lsf) {
        int s1 = MPA_SLEN[q->sfc], s2 = MPA_SLEN[16 + q->sfc];
        if (q->btype == 2) {
            int n = q->mixed ? 17 : 18;
            for (int i = 0; i < n; i++)
                q->sf[j++] = (int)mpa_bits(b, s1);
            for (int i = 0; i < 18; i++)
                q->sf[j++] = (int)mpa_bits(b, s2);
        } else {
            for (int k = 0; k < 4; k++) {
                int n = k == 0 ? 6 : 5;
                if (q->scfsi & (8 >> k)) {
                    for (int i = 0; i < n; i++, j++)
                        q->sf[j] = first->sf[j];
                } else {
                    for (int i = 0; i < n; i++)
                        q->sf[j++] = (int)mpa_bits(b, k < 2 ? s1 : s2);
                }
            }
        }
        return;
    }
    int t1 = q->btype == 2 ? (q->mixed ? 2 : 1) : 0, t2, sl[4] = {0};
    int sf = q->sfc;
    if ((h->mode_ext & 1) && ch == 1) {
        sf >>= 1;
        if (sf < 180) {
            lsf_expand(sl, sf, 6, 6, 0);
            t2 = 3;
        } else if (sf < 244) {
            lsf_expand(sl, sf - 180, 4, 4, 0);
            t2 = 4;
        } else {
            lsf_expand(sl, sf - 244, 3, 0, 0);
            t2 = 5;
        }
    } else if (sf < 400) {
        lsf_expand(sl, sf, 5, 4, 4);
        t2 = 0;
    } else if (sf < 500) {
        lsf_expand(sl, sf - 400, 5, 4, 0);
        t2 = 1;
    } else {
        lsf_expand(sl, sf - 500, 3, 0, 0);
        t2 = 2;
        q->preflag = 1;
    }
    for (int k = 0; k < 4; k++) {
        int n = MPA_LSF_NSF[t2 * 12 + t1 * 4 + k];
        for (int i = 0; i < n && j < 40; i++)
            q->sf[j++] = (int)mpa_bits(b, sl[k]);
    }
}

/* the granule's lines, requantized: the scale of each line in quarters of
   an octave below the global gain */
static void exponents(const struct mpa_hdr *h, const struct gran *q, int *e)
{
    int gain = q->gain - 210, shift = q->sfscale + 1, at = 0;
    const uint8_t *bl = band_long(h), *bs = band_short(h);
    for (int i = 0; i < q->long_end; i++) {
        int v = gain -
                ((q->sf[i] + (q->preflag ? MPA_PRETAB[22 + i] : 0)) << shift);
        for (int j = 0; j < bl[i] && at < 576; j++)
            e[at++] = v;
    }
    int k = q->long_end;
    for (int i = q->short_start; i < 13; i++)
        for (int w = 0; w < 3; w++) {
            int v = gain - (q->sbg[w] << 3) - (q->sf[k++] << shift);
            for (int j = 0; j < bs[i] && at < 576; j++)
                e[at++] = v;
        }
    while (at < 576)
        e[at++] = gain;
}

static float requant(const struct l3 *d, int v, int e)
{
    float x = d->pow43[v < 0 ? -v : v] * exp2f(0.25f * (float)e);
    return v < 0 ? -x : x;
}

static void huffman(const struct l3 *d, struct mpa_br *b, const struct gran *q,
                    const int *e, size_t end, float *xr)
{
    int at = 0;
    for (int r = 0; r < 3; r++) {
        int tab = MPA_HUFF_DATA[2 * q->tsel[r]];
        int lin = MPA_HUFF_DATA[2 * q->tsel[r] + 1];
        for (; at < q->region[r]; at += 2) {
            if (b->pos >= end || !tab) {
                xr[at] = xr[at + 1] = 0;
                continue;
            }
            int x, y;
            huff_pair(d, b, tab, &x, &y);
            if (x == 15 && lin)
                x += (int)mpa_bits(b, lin);
            if (x && mpa_bits(b, 1))
                x = -x;
            if (y == 15 && lin)
                y += (int)mpa_bits(b, lin);
            if (y && mpa_bits(b, 1))
                y = -y;
            xr[at] = x ? requant(d, x, e[at]) : 0;
            xr[at + 1] = y ? requant(d, y, e[at + 1]) : 0;
        }
    }
    size_t last = 0;
    while (at <= 572) {
        if (b->pos >= end) {
            if (b->pos > end && last) { /* the last quadruple ran over */
                at -= 4;
                b->pos = last;
            }
            break;
        }
        last = b->pos;
        int v = huff_quad(d, b, q->c1sel);
        for (int k = 0; k < 4; k++) {
            int s = (v >> (3 - k)) & 1;
            if (s && mpa_bits(b, 1))
                s = -1;
            xr[at + k] = s ? requant(d, s, e[at + k]) : 0;
        }
        at += 4;
    }
    while (at < 576)
        xr[at++] = 0;
    b->pos = end;
}

/* mid/side and intensity stereo, on the lines as they were coded */
static void stereo(const struct mpa_hdr *h, const struct gran *g1, float *t0,
                   float *t1)
{
    static const float IS_SQRT2 = 0.70710678118654752f;
    if (!(h->mode_ext & 1)) {
        if (h->mode_ext & 2)
            for (int i = 0; i < 576; i++) {
                float a = t0[i], b = t1[i];
                t0[i] = a + b;
                t1[i] = a - b;
            }
        return;
    }
    /* intensity: the bands above the last line of the right channel that
       is not zero, from the top down */
    int sf_max = h->lsf ? 16 : 7;
    float is0[16], is1[16];
    for (int i = 0; i < 16; i++) {
        if (!h->lsf) {
            double f = tan(i * M_PI / 12);
            is0[i] = i < 7 ? (float)(f / (1 + f)) : 0;
            is1[i] = i < 7 ? (float)(1 / (1 + f)) : 0;
            if (i == 6) { /* tan(pi / 2) */
                is0[i] = 1;
                is1[i] = 0;
            }
        } else {
            int j = g1->sfc & 1;
            float f = (float)exp2(-(j + 1) * ((i + 1) >> 1) / 4.0);
            is0[i] = i & 1 ? f : 1;
            is1[i] = i & 1 ? 1 : f;
        }
    }
    const uint8_t *bl = band_long(h), *bs = band_short(h);
    int found_short[3] = {0, 0, 0}, at = 576;
    int k = g1->long_end + (12 - g1->short_start) * 3;
    for (int i = 12; i >= g1->short_start; i--) {
        if (i != 11)
            k -= 3;
        int len = bs[i];
        for (int l = 2; l >= 0; l--) {
            at -= len;
            int ms = 1;
            if (!found_short[l]) {
                for (int j = 0; j < len; j++)
                    if (t1[at + j] != 0) {
                        found_short[l] = 1;
                        break;
                    }
                int sf = g1->sf[k + l];
                if (!found_short[l] && sf < sf_max) {
                    for (int j = 0; j < len; j++) {
                        float x = t0[at + j];
                        t0[at + j] = x * is0[sf];
                        t1[at + j] = x * is1[sf];
                    }
                    ms = 0;
                }
            }
            if (ms && (h->mode_ext & 2))
                for (int j = 0; j < len; j++) {
                    float a = t0[at + j], b = t1[at + j];
                    t0[at + j] = (a + b) * IS_SQRT2;
                    t1[at + j] = (a - b) * IS_SQRT2;
                }
        }
    }
    int found = found_short[0] | found_short[1] | found_short[2];
    for (int i = g1->long_end - 1; i >= 0; i--) {
        int len = bl[i], ms = 1;
        at -= len;
        if (!found) {
            for (int j = 0; j < len; j++)
                if (t1[at + j] != 0) {
                    found = 1;
                    break;
                }
            int sf = g1->sf[i == 21 ? 20 : i];
            if (!found && sf < sf_max) {
                for (int j = 0; j < len; j++) {
                    float x = t0[at + j];
                    t0[at + j] = x * is0[sf];
                    t1[at + j] = x * is1[sf];
                }
                ms = 0;
            }
        }
        if (ms && (h->mode_ext & 2))
            for (int j = 0; j < len; j++) {
                float a = t0[at + j], b = t1[at + j];
                t0[at + j] = (a + b) * IS_SQRT2;
                t1[at + j] = (a - b) * IS_SQRT2;
            }
    }
}

/* short bands from coded order (window after window) to interleaved */
static void reorder(const struct mpa_hdr *h, const struct gran *q, float *xr)
{
    if (q->btype != 2)
        return;
    float tmp[576];
    int at = q->mixed ? (h->rate_index != 8 ? 36 : 72) : 0;
    const uint8_t *bs = band_short(h);
    for (int i = q->short_start; i < 13 && at < 576; i++) {
        int len = bs[i];
        if (at + 3 * len > 576)
            break;
        for (int j = 0; j < len; j++)
            for (int w = 0; w < 3; w++)
                tmp[3 * j + w] = xr[at + w * len + j];
        memcpy(xr + at, tmp, sizeof(float) * 3 * (size_t)len);
        at += 3 * len;
    }
}

static void antialias(const struct gran *q, float *xr)
{
    static const double C[8] = {-0.6,   -0.535, -0.33,   -0.185,
                                -0.095, -0.041, -0.0142, -0.0037};
    int n = q->btype == 2 ? (q->mixed ? 1 : 0) : 31;
    for (int sb = 1; sb <= n; sb++)
        for (int i = 0; i < 8; i++) {
            double cs = 1 / sqrt(1 + C[i] * C[i]), ca = C[i] * cs;
            float a = xr[18 * sb - 1 - i], b = xr[18 * sb + i];
            xr[18 * sb - 1 - i] = (float)(a * cs - b * ca);
            xr[18 * sb + i] = (float)(b * cs + a * ca);
        }
}

/* the IMDCT of each subband, its overlap, the frequency inversion */
static void imdct(struct l3 *d, const struct gran *q, int ch, const float *xr,
                  float (*sb)[32])
{
    for (int s = 0; s < 32; s++) {
        float z[36];
        const float *x = xr + 18 * s;
        int lng = q->btype != 2 || (q->mixed && s < 2);
        if (lng) {
            /* with the mixed flag the two lowest subbands take the normal
               window, whatever the block type */
            const float *w = d->win[q->mixed && s < 2 ? 0 : q->btype];
            for (int i = 0; i < 36; i++) {
                float v = 0;
                for (int k = 0; k < 18; k++)
                    v += x[k] * d->cos36[i][k];
                z[i] = v * w[i];
            }
        } else {
            memset(z, 0, sizeof z);
            for (int w = 0; w < 3; w++)
                for (int i = 0; i < 12; i++) {
                    float v = 0;
                    for (int k = 0; k < 6; k++)
                        v += x[3 * k + w] * d->cos12[i][k];
                    z[6 + 6 * w + i] += v * d->win[2][i];
                }
        }
        float *o = d->overlap[ch][s];
        for (int i = 0; i < 18; i++) {
            float v = z[i] + o[i];
            o[i] = z[18 + i];
            sb[i][s] = (s & 1) && (i & 1) ? -v : v;
        }
    }
}

/* a Layer III frame (its bytes from the header on) into pcm[ch][]: the
   samples a channel */
static int l3_frame(struct l3 *d, const uint8_t *f, size_t n,
                    const struct mpa_hdr *h, float (*pcm)[1152])
{
    size_t side = h->lsf ? (h->ch == 1 ? 9 : 17) : (h->ch == 1 ? 17 : 32);
    size_t at = 4 + (h->crc ? 2 : 0);
    int ng = h->lsf ? 1 : 2, mdb;
    if (n < at + side)
        return -1;
    struct mpa_br b = {.p = f + at, .n = side};
    struct gran g[2][2];
    memset(g, 0, sizeof g);
    if (side_info(&b, h, g, &mdb))
        return -1;
    /* the main data: the reservoir, then this frame's */
    size_t md = n - at - side, old = d->res_n, skip = 0;
    if (md > sizeof d->res) {
        /* more main data than the reservoir holds: its last part only */
        skip = md - sizeof d->res;
        md = sizeof d->res;
        old = 0;
    } else if (old + md > sizeof d->res) {
        size_t drop = old + md - sizeof d->res;
        memmove(d->res, d->res + drop, old - drop);
        old -= drop;
    }
    memcpy(d->res + old, f + at + side + skip, md);
    d->res_n = old + md;
    struct mpa_br m = {.p = d->res, .n = d->res_n};
    long pos = ((long)old - mdb) * 8;
    int e[576];
    for (int gr = 0; gr < ng; gr++) {
        for (int ch = 0; ch < h->ch; ch++) {
            struct gran *q = &g[gr][ch];
            if (pos < 0) { /* its data was in frames not here */
                memset(d->xr[ch], 0, sizeof d->xr[ch]);
                pos += q->part23;
                continue;
            }
            m.pos = (size_t)pos;
            size_t end = (size_t)pos + (size_t)q->part23;
            scale_factors(&m, h, q, &g[0][ch], ch);
            exponents(h, q, e);
            huffman(d, &m, q, e, end, d->xr[ch]);
            pos = (long)end;
        }
        if (h->mode == MPA_MODE_JOINT && h->ch == 2)
            stereo(h, &g[gr][1], d->xr[0], d->xr[1]);
        for (int ch = 0; ch < h->ch; ch++) {
            reorder(h, &g[gr][ch], d->xr[ch]);
            antialias(&g[gr][ch], d->xr[ch]);
            imdct(d, &g[gr][ch], ch, d->xr[ch], d->sb[ch] + 18 * gr);
        }
    }
    /* keep what the next frames may reach back to */
    if (d->res_n > 4096) {
        memmove(d->res, d->res + d->res_n - 4096, 4096);
        d->res_n = 4096;
    }
    for (int ch = 0; ch < h->ch; ch++)
        for (int t = 0; t < 18 * ng; t++)
            mpa_synth(&d->syn, ch, d->sb[ch][t], pcm[ch] + 32 * t);
    return 576 * ng;
}

/* ---- the file ---- */

/* the frame at d[at], its size found by the next sync when it is free
   format: 0, or -1 */
static int frame_at(const uint8_t *d, size_t n, size_t at, struct mpa_hdr *h)
{
    if (at + 4 > n || mpa_header(d + at, h))
        return -1;
    if (h->size)
        return 0;
    for (size_t k = at + 4; k + 4 <= n; k++) {
        struct mpa_hdr x;
        if (!mpa_header(d + k, &x) && x.layer == h->layer &&
            x.rate == h->rate && x.lsf == h->lsf && !x.bitrate) {
            h->size = k - at;
            return 0;
        }
    }
    h->size = n - at;
    return 0;
}

/* a frame there with another after it, or the end of the bytes */
static int frame_ok(const uint8_t *d, size_t n, size_t at, struct mpa_hdr *h)
{
    if (frame_at(d, n, at, h))
        return 0;
    struct mpa_hdr x;
    size_t next = at + h->size;
    if (next >= n || next + 4 > n)
        return h->size > 4;
    return !frame_at(d, n, next, &x) && x.layer == h->layer &&
           x.rate == h->rate;
}

/* Xing, Info or VBRI in the first frame: 1 when it is that frame, with
   LAME's encoder delay and padding when it says them */
static int vbr_frame(const uint8_t *f, const struct mpa_hdr *h, int *delay,
                     int *padding)
{
    size_t side = h->lsf ? (h->ch == 1 ? 9 : 17) : (h->ch == 1 ? 17 : 32);
    size_t x = 4 + side;
    if (h->layer != 3 || h->size < x + 8)
        return 0;
    if (h->size >= 36 + 4 && !memcmp(f + 36, "VBRI", 4))
        return 1;
    if (memcmp(f + x, "Xing", 4) && memcmp(f + x, "Info", 4))
        return 0;
    uint32_t flags = (uint32_t)f[x + 4] << 24 | (uint32_t)f[x + 5] << 16 |
                     (uint32_t)f[x + 6] << 8 | f[x + 7];
    size_t t = x + 8 + (flags & 1 ? 4 : 0) + (flags & 2 ? 4 : 0) +
               (flags & 4 ? 100 : 0) + (flags & 8 ? 4 : 0);
    if (t + 24 <= h->size &&
        (!memcmp(f + t, "LAME", 4) || !memcmp(f + t, "Lavf", 4) ||
         !memcmp(f + t, "Lavc", 4))) {
        uint32_t v =
            (uint32_t)f[t + 21] << 16 | (uint32_t)f[t + 22] << 8 | f[t + 23];
        *delay = (int)(v >> 12);
        *padding = (int)(v & 0xFFF);
    }
    return 1;
}

int janas_audio_mpeg(const uint8_t *d, size_t n, struct audio_out *o,
                     char *why, size_t wn)
{
    struct mpa_hdr h;
    size_t at = 0;
    /* the first frame, a frame after it, within the first 64 KiB */
    while (at + 4 <= n && at < 65536 && !frame_ok(d, n, at, &h))
        at++;
    if (at + 4 > n || at >= 65536)
        return 1;
    struct l3 *l3 = malloc(sizeof *l3);
    float(*pcm)[1152] = malloc(sizeof(float) * 2 * 1152);
    if (!l3 || !pcm) {
        free(l3);
        free(pcm);
        snprintf(why, wn, "out of memory");
        return -1;
    }
    l3_init(l3);
    int rate = h.rate, layer = h.layer, delay = -1, padding = 0;
    if (vbr_frame(d + at, &h, &delay, &padding))
        at += h.size;
    float out[1152];
    size_t got = 0;
    int rc = 0;
    o->rate = (uint32_t)rate;
    while (at + 4 <= n) {
        if (frame_at(d, n, at, &h) || h.rate != rate || h.layer != layer) {
            at++; /* lost: the next sync */
            continue;
        }
        /* the last frame cut short: what of it there is */
        size_t len = at + h.size <= n ? h.size : n - at;
        int k = h.layer == 3 ? l3_frame(l3, d + at, len, &h, pcm)
                             : mpa_layer12(d + at, len, &h, &l3->syn, pcm);
        at += len;
        if (k <= 0)
            continue;
        for (int i = 0; i < k; i++)
            out[i] = h.ch == 2 ? 0.5f * (pcm[0][i] + pcm[1][i]) : pcm[0][i];
        if (audio_put(o, out, (size_t)k, why, wn) != 0) {
            rc = -1;
            break;
        }
        got += (size_t)k;
    }
    free(l3);
    free(pcm);
    if (!rc && !got) {
        snprintf(why, wn, "an MPEG audio file without a frame this reads");
        rc = -1;
    }
    if (rc)
        return -1;
    /* gapless: the encoder's delay and padding, and the decoder's 529 */
    size_t skip = delay >= 0 ? (size_t)delay + 529 : 0;
    size_t trim = padding > 529 ? (size_t)padding - 529 : 0;
    if (skip + trim >= got) {
        skip = 0;
        trim = 0;
    }
    o->skip = skip;
    o->total = got - trim;
    return 0;
}
