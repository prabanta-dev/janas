/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_aac_usac.c - USAC, xHE-AAC (ISO/IEC 23003-3), in the frequency
 * domain: the UsacConfig and its elements, the frames of single channels,
 * pairs and LFEs, their scalefactors, the spectrum's context-adaptive
 * arithmetic code, noise filling, mid/side and complex stereo prediction
 * (its imaginary part estimated from the real one, MDST), TNS; then the
 * AAC core's inverse MDCT and windows (audio_aac.c), frames of 1024 and
 * 768 samples. Not read, refused with a reason: the linear-prediction
 * coding of speech (LPD), its transitions (FAC), the time-warped MDCT, SBR
 * within USAC (eSBR, plain SBR) and MPEG Surround 2-1-2, as FFmpeg reads
 * none of them but plain SBR. Extension elements are skipped, AudioPreRoll
 * among them (its frames are those before the stream's first).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_aac_core.h"
#include "audio_aac_usac_tab.h"

#define MAX_STE (8 * 16)

/* a channel's own state */
struct usac_ch {
    /* the arithmetic coder's context: the last frame's 2-tuples, the ones
       before in this frame */
    uint8_t last[513];
    int last_len;
    uint8_t cur[4];
    uint32_t pre;
    uint32_t seed; /* noise filling's signs */
    int tns_data, noise_level, noise_offset, grouping;
    int prev_seq;
    float prev[1024]; /* the spectrum of the frame before */
};

/* a pair's stereo state */
struct usac_st {
    int common, ms_mode, tns_on_lr, max_ste, prev_groups;
    int pred_dir, complex, use_prev;
    uint8_t ms[MAX_STE], pred_used[MAX_STE];
    float alpha_re[MAX_STE], alpha_im[MAX_STE];
    float prev_re[MAX_STE], prev_im[MAX_STE];
    float dmx[1024], dmx_prev[1024], dmx_im[1024];
};

/* an escaped value: nb1 bits, then nb2 more when they are all ones, then
   nb3 more */
static uint32_t escaped(struct abr *b, int nb1, int nb2, int nb3)
{
    uint32_t v = abr_bits(b, nb1);
    if (v < (1u << nb1) - 1)
        return v;
    uint32_t v2 = abr_bits(b, nb2);
    v += v2;
    if (nb3 && v2 == (1u << nb2) - 1)
        v += abr_bits(b, nb3);
    return v;
}

int usac_config(struct abr *b, struct aac_cfg *c, char *why, size_t wn)
{
    int fi = (int)abr_bits(b, 5);
    int rate = fi == 31 ? (int)abr_bits(b, 24) : USAC_RATES[fi];
    int csfl = (int)abr_bits(b, 3); /* coreSbrFrameLengthIndex */
    if (rate <= 0 || rate > 96000) {
        snprintf(why, wn, "a USAC configuration without a rate");
        return -1;
    }
    if (csfl > 1) {
        snprintf(why, wn, "a USAC stream with SBR (not read)");
        return -1;
    }
    c->aot = AOT_USAC;
    c->rate = rate;
    c->rate_index = aac_rate_index(rate);
    c->frame = csfl == 0 ? 768 : 1024;
    c->sbr = c->ps = 0;
    if (!abr_bits(b, 5)) { /* channelConfigurationIndex 0: the positions */
        uint32_t n = escaped(b, 5, 8, 16);
        if (n > 64) {
            snprintf(why, wn, "a USAC configuration of %u channels", n);
            return -1;
        }
        b->pos += 5 * (size_t)n;
    }
    uint32_t n = escaped(b, 4, 8, 16) + 1;
    if (n > 64) {
        snprintf(why, wn, "a USAC configuration of %u elements", n);
        return -1;
    }
    c->usac_n = (int)n;
    c->channels = 0;
    for (uint32_t i = 0; i < n; i++) {
        struct usac_el *e = &c->usac_el[i];
        memset(e, 0, sizeof *e);
        e->type = (uint8_t)abr_bits(b, 2);
        if (e->type <= 1) { /* UsacCoreConfig, no SBR to configure */
            if (abr_bit(b)) {
                snprintf(why, wn,
                         "a USAC stream with the time-warped MDCT (not read)");
                return -1;
            }
            e->noise_fill = (uint8_t)abr_bit(b);
            c->channels += e->type + 1;
        } else if (e->type == 2) {
            c->channels++;
        } else {
            escaped(b, 4, 8, 16); /* usacExtElementType */
            uint32_t cl = escaped(b, 4, 8, 16);
            if (abr_bit(b))
                e->default_len = escaped(b, 8, 16, 0) + 1;
            e->frag = (uint8_t)abr_bit(b);
            b->pos += 8 * (size_t)cl;
        }
        if (abr_over(b))
            break;
    }
    if (abr_bit(b)) { /* usacConfigExtension: loudness and others, skipped */
        uint32_t k = escaped(b, 2, 4, 8) + 1;
        for (uint32_t i = 0; i < k && !abr_over(b); i++) {
            escaped(b, 4, 8, 16);
            b->pos += 8 * (size_t)escaped(b, 4, 8, 16);
        }
    }
    if (abr_over(b)) {
        snprintf(why, wn, "a USAC configuration cut short");
        return -1;
    }
    return 0;
}

/* ---- the arithmetic code of the spectrum (7.4) ---- */

/* read past the end by more than the arithmetic decoder's look-ahead: a
   frame's last spectrum may end a few bits past its last byte, zeros */
static int over(const struct abr *b)
{
    return b->pos > b->n * 8 + 16;
}

struct ari {
    uint16_t low, high, val;
};

/* the symbol whose interval holds the value: the count of the
   cumulative frequencies (falling, the last 0) above it */
static int ari_decode(struct ari *a, struct abr *b, const uint16_t *cdf,
                      int len)
{
    int val = a->val, low = a->low, high = a->high;
    int rng = high - low + 1;
    int c = ((val - low + 1) << 14) - 1;
    int sym = 0;
    while (sym < len && cdf[sym] * rng > c)
        sym++;
    if (sym)
        high = low + ((rng * cdf[sym - 1]) >> 14) - 1;
    low += (rng * cdf[sym]) >> 14;
    for (;;) {
        if (high < 32768) {
        } else if (low >= 32768) {
            val -= 32768;
            low -= 32768;
            high -= 32768;
        } else if (low >= 16384 && high < 49152) {
            val -= 16384;
            low -= 16384;
            high -= 16384;
        } else {
            break;
        }
        low += low;
        high += high + 1;
        val = val << 1 | abr_bit(b);
    }
    a->low = (uint16_t)low;
    a->high = (uint16_t)high;
    a->val = (uint16_t)val;
    return sym;
}

/* the probability model of a context: its index in the hash, or the
   lookup of the interval it falls in */
static int ari_model(uint32_t c)
{
    int lo = -1, hi = (int)(sizeof USAC_LOOKUP_M / sizeof *USAC_LOOKUP_M) - 1;
    while (hi - lo > 1) {
        int i = lo + (hi - lo) / 2;
        uint32_t j = USAC_HASH_M[i];
        if (c < (j >> 8))
            hi = i;
        else if (c > (j >> 8))
            lo = i;
        else
            return (int)(j & 0xff);
    }
    return USAC_LOOKUP_M[hi];
}

/* the context at a window's start: reset, or the last one's mapped to n
   lines */
static uint32_t ari_start(struct usac_ch *u, int reset, int n)
{
    if (reset) {
        memset(u->last, 0, sizeof u->last);
        u->last_len = n;
    } else if (u->last_len != n) {
        uint8_t old[sizeof u->last];
        memcpy(old, u->last, sizeof old);
        float ratio = (float)u->last_len / (float)n;
        int i;
        for (i = 0; i < n / 2; i++)
            u->last[i] = old[(int)((float)i * ratio)];
        for (; i < (int)sizeof u->last; i++)
            u->last[i] = 0;
        u->last_len = n;
    }
    u->cur[0] = 1;
    u->cur[1] = u->cur[2] = u->cur[3] = 0;
    u->pre = (uint32_t)u->last[0] << 12;
    return u->pre;
}

static uint32_t ari_context(struct usac_ch *u, int i)
{
    uint32_t c = ((u->pre >> 8) + ((uint32_t)u->last[i + 1] << 8)) << 4;
    c += u->cur[1];
    u->pre = c;
    if (i > 3 && u->cur[3] + u->cur[2] + u->cur[1] < 5)
        return c + 0x10000;
    return c;
}

static void ari_update(struct usac_ch *u, int i, int a, int b)
{
    int q = a + b + 1;
    u->cur[0] = (uint8_t)(q > 15 ? 15 : q);
    u->cur[3] = u->cur[2];
    u->cur[2] = u->cur[1];
    u->cur[1] = u->cur[0];
    u->last[i] = u->cur[0];
}

/* past the last 2-tuple decoded, the context of zeros */
static void ari_finish(struct usac_ch *u, int from, int n)
{
    int i;
    for (i = from; i < n / 2; i++)
        u->last[i] = 1;
    for (; i < (int)sizeof u->last; i++)
        u->last[i] = 0;
}

static float pow43(int q)
{
    return (float)(q * cbrt((double)q));
}

/* a window's lg lines of n into x (its 2-tuples, then their signs): 0,
   or -1 */
static int ari_spectrum(struct abr *b, struct usac_ch *u, float *x, int reset,
                        int lg, int n)
{
    ari_start(u, reset, n);
    memset(x, 0, sizeof(float) * (size_t)n);
    if (!lg) {
        ari_finish(u, 0, n);
        return 0;
    }
    struct ari a = {.low = 0, .high = 0xffff, .val = (uint16_t)abr_bits(b, 16)};
    struct abr b2 = *b;
    int i;
    for (i = 0; i < lg / 2; i++) {
        uint32_t c = ari_context(u, i);
        int lvl = 0, esc = 0, m;
        for (;;) { /* the most significant bits, escaped */
            int pki = ari_model(c + ((uint32_t)esc << 17));
            m = ari_decode(&a, &b2, USAC_MSB_CDF[pki], 17);
            if (m < 16)
                break;
            if (++lvl > 23 || over(&b2))
                return -1;
            esc = lvl > 7 ? 7 : lvl;
        }
        int qb = m >> 2, qa = m & 3;
        if (!m) {
            if (esc)
                break; /* ARITH_STOP */
            qa = qb = 0;
        }
        for (int l = lvl; l > 0; l--) { /* the least significant bits */
            int k = !qa ? 1 : !qb ? 0 : 2;
            int r = ari_decode(&a, &b2, USAC_LSB_CDF[k], 4);
            qa = qa << 1 | (r & 1);
            qb = qb << 1 | (r >> 1 & 1);
        }
        x[2 * i] = pow43(qa);
        x[2 * i + 1] = pow43(qb);
        ari_update(u, i, qa, qb);
    }
    /* the decoder reads 14 bits ahead of what it took */
    b->pos = lg > 1 ? b2.pos - 14 : b2.pos;
    ari_finish(u, i, n);
    for (int k = 0; k < lg; k++)
        if (x[k] != 0 && !abr_bit(b))
            x[k] = -x[k];
    return over(b) ? -1 : 0;
}

/* ---- a channel's stream ---- */

static int max_sfb_bits(const struct ics *ics)
{
    return ics->seq == EIGHT_SHORT ? 4 : 6;
}

/* the window's bands and groups after ics_info: 0, or -1 */
static int setup(struct aac_dec *d, struct chan *c)
{
    struct ics *ics = &c->ics;
    ics->groups = 1;
    ics->group_len[0] = 1;
    if (ics->seq == EIGHT_SHORT) {
        for (int i = 6; i >= 0; i--) {
            if (c->u->grouping >> i & 1)
                ics->group_len[ics->groups - 1]++;
            else
                ics->group_len[ics->groups++] = 1;
        }
        ics->num_windows = 8;
        ics->swb =
            aac_swb_table(d->short_len, d->cfg.rate_index, &ics->num_swb);
    } else {
        ics->num_windows = 1;
        ics->swb = aac_swb_table(d->frame, d->cfg.rate_index, &ics->num_swb);
    }
    /* every band a spectral one, for the tools of the core */
    memset(c->cb, 11, sizeof c->cb);
    return ics->max_sfb > ics->num_swb ? -1 : 0;
}

/* ics_info of one channel: the window, its shape, its bands */
static int read_ics_info(struct aac_dec *d, struct abr *b, struct chan *c)
{
    struct ics *ics = &c->ics;
    c->u->prev_seq = ics->seq;
    ics->seq = (int)abr_bits(b, 2);
    ics->prev_shape = ics->shape;
    ics->shape = abr_bit(b);
    ics->max_sfb = (int)abr_bits(b, max_sfb_bits(ics));
    if (ics->seq == EIGHT_SHORT)
        c->u->grouping = (int)abr_bits(b, 7);
    return setup(d, c);
}

/* the scalefactors, the first the global gain, then differences */
static int read_sf(struct aac_dec *d, struct abr *b, struct chan *c, int gain)
{
    const struct ics *ics = &c->ics;
    int stride = ics->seq == EIGHT_SHORT ? 16 : 0, sf = gain;
    for (int g = 0; g < ics->groups; g++)
        for (int i = 0; i < ics->max_sfb; i++) {
            if (g || i) {
                int v = vlc_get(b, &d->sf_vlc);
                if (v < 0)
                    return -1;
                sf += v - 60;
                if (sf < 0 || sf > 255)
                    return -1;
            }
            c->sf[g * stride + i] = sf - 100;
        }
    return 0;
}

/* noise in the lines left zero past the start, a band all zero raised
   by the offset; then each band scaled by its scalefactor */
static void scale(struct aac_dec *d, struct chan *c)
{
    const struct ics *ics = &c->ics;
    struct usac_ch *u = c->u;
    int is8 = ics->seq == EIGHT_SHORT;
    int s = is8 ? d->short_len : d->frame, stride = is8 ? 16 : 0;
    int start = USAC_NOISE_START[d->frame == 768][is8];
    float noise = powf(2, ((float)u->noise_level - 14.0f) / 3.0f);
    int win = 0;
    for (int g = 0; g < ics->groups; g++) {
        int glen = ics->group_len[g];
        for (int i = 0; i < ics->max_sfb; i++) {
            int lo = ics->swb[i], hi = ics->swb[i + 1];
            int *sf = &c->sf[g * stride + i];
            if (u->noise_level && lo >= start) {
                int zero = 1;
                for (int w = 0; w < glen; w++) {
                    float *x = c->coef + (win + w) * s;
                    for (int k = lo; k < hi; k++) {
                        if (x[k] == 0) {
                            u->seed = u->seed * 69069 + 5;
                            x[k] = u->seed & 0x10000 ? -noise : noise;
                        } else {
                            zero = 0;
                        }
                    }
                }
                if (zero)
                    *sf += u->noise_offset - 16;
            }
            float f = powf(2, 0.25f * (float)*sf);
            for (int w = 0; w < glen; w++) {
                float *x = c->coef + (win + w) * s;
                for (int k = lo; k < hi; k++)
                    x[k] *= f;
            }
        }
        win += glen;
    }
}

/* ---- stereo ---- */

/* cplx_pred_data (7.7.2): which bands are predicted, the prediction's
   coefficients, coded as differences in frequency or in time */
static int read_cplx(struct aac_dec *d, struct abr *b, struct usac_st *st,
                     const struct chan *l, int indep)
{
    const struct ics *ics = &l->ics;
    int m = st->max_ste;
    if (!abr_bit(b)) { /* cplx_pred_all */
        for (int g = 0; g < ics->groups; g++)
            for (int i = 0; i < m; i += 2) {
                uint8_t v = (uint8_t)abr_bit(b);
                st->pred_used[g * m + i] = v;
                if (i + 1 < m)
                    st->pred_used[g * m + i + 1] = v;
            }
    } else {
        memset(st->pred_used, 1, sizeof st->pred_used);
    }
    st->pred_dir = abr_bit(b);
    st->complex = abr_bit(b);
    st->use_prev = st->complex && !indep ? abr_bit(b) : 0;
    int in_time = !indep ? abr_bit(b) : 0;
    for (int g = 0; g < ics->groups; g++)
        for (int i = 0; i < m; i += 2) {
            int at = g * m + i;
            float re = 0, im = 0;
            if (in_time) {
                /* from the group before; the first from the frame before,
                   its last group when both are short */
                if (g) {
                    re = st->alpha_re[at - m];
                    im = st->alpha_im[at - m];
                } else {
                    int p =
                        ics->seq == EIGHT_SHORT && l->u->prev_seq == EIGHT_SHORT
                            ? (st->prev_groups - 1) * m + i
                            : at;
                    re = st->prev_re[p];
                    im = st->prev_im[p];
                }
            } else if (i) {
                re = st->alpha_re[at - 1];
                im = st->alpha_im[at - 1];
            }
            if (st->pred_used[at]) {
                int v = vlc_get(b, &d->sf_vlc);
                if (v < 0)
                    return -1;
                re += (float)(60 - v) * 0.1f;
                if (st->complex) {
                    if ((v = vlc_get(b, &d->sf_vlc)) < 0)
                        return -1;
                    im += (float)(60 - v) * 0.1f;
                }
            } else {
                re = im = 0;
            }
            st->alpha_re[at] = re;
            st->alpha_im[at] = im;
            if (i + 1 < m) {
                st->alpha_re[at + 1] = re;
                st->alpha_im[at + 1] = im;
            }
        }
    return 0;
}

/* StereoCoreToolInfo of a pair coded in the frequency domain: 0, or -1 */
static int read_stereo(struct aac_dec *d, struct abr *b, struct usac_st *st,
                       struct chan *l, struct chan *r, int indep)
{
    memset(st->alpha_re, 0, sizeof st->alpha_re);
    memset(st->alpha_im, 0, sizeof st->alpha_im);
    int tns_active = abr_bit(b);
    st->common = abr_bit(b);
    st->ms_mode = 0;
    if (!st->common || indep) {
        memset(st->prev_re, 0, sizeof st->prev_re);
        memset(st->prev_im, 0, sizeof st->prev_im);
    }
    if (st->common) {
        l->u->prev_seq = l->ics.seq;
        r->u->prev_seq = r->ics.seq;
        l->ics.seq = r->ics.seq = (int)abr_bits(b, 2);
        l->ics.prev_shape = l->ics.shape;
        r->ics.prev_shape = r->ics.shape;
        l->ics.shape = r->ics.shape = abr_bit(b);
        /* a change to or from short windows: no prediction from the frame
           before */
        if ((l->ics.seq == EIGHT_SHORT) != (l->u->prev_seq == EIGHT_SHORT) ||
            (r->ics.seq == EIGHT_SHORT) != (r->u->prev_seq == EIGHT_SHORT)) {
            memset(st->prev_re, 0, sizeof st->prev_re);
            memset(st->prev_im, 0, sizeof st->prev_im);
        }
        l->ics.max_sfb = r->ics.max_sfb =
            (int)abr_bits(b, max_sfb_bits(&l->ics));
        if (l->ics.seq == EIGHT_SHORT)
            l->u->grouping = r->u->grouping = (int)abr_bits(b, 7);
        if (!abr_bit(b)) /* common_max_sfb */
            r->ics.max_sfb = (int)abr_bits(b, max_sfb_bits(&r->ics));
        if (setup(d, l) != 0 || setup(d, r) != 0)
            return -1;
        st->max_ste =
            l->ics.max_sfb > r->ics.max_sfb ? l->ics.max_sfb : r->ics.max_sfb;
        if (l->ics.seq == EIGHT_SHORT ? st->max_ste > 16 : st->max_ste > 64)
            return -1;
        st->ms_mode = (int)abr_bits(b, 2);
        memset(st->ms, 0, sizeof st->ms);
        memset(st->pred_used, 0, sizeof st->pred_used);
        if (st->ms_mode == 1) {
            for (int g = 0; g < l->ics.groups; g++)
                for (int i = 0; i < st->max_ste; i++)
                    st->ms[g * st->max_ste + i] = (uint8_t)abr_bit(b);
        } else if (st->ms_mode == 2) {
            memset(st->ms, 1, sizeof st->ms);
        } else if (st->ms_mode == 3 && read_cplx(d, b, st, l, indep) != 0) {
            return -1;
        }
    }
    st->tns_on_lr = 0;
    l->u->tns_data = r->u->tns_data = 0;
    if (tns_active) {
        int common_tns = st->common ? abr_bit(b) : 0;
        st->tns_on_lr = abr_bit(b);
        if (common_tns) {
            if (aac_read_tns(d, b, &l->ics, &l->tns) != 0)
                return -1;
            r->tns = l->tns;
            l->tns.present = r->tns.present = 1;
        } else if (abr_bit(b)) {
            l->u->tns_data = r->u->tns_data = 1;
        } else {
            r->u->tns_data = abr_bit(b);
            l->u->tns_data = !r->u->tns_data;
        }
    }
    return 0;
}

/* the imaginary spectrum's estimate from a real one (7.7.2.3), window by
   window: a filter of 7 taps, the edges mirrored */
static void mdst(float *im, const float *re, const float *f, int n, float even,
                 float odd)
{
    for (int i = 0; i < n; i++) {
        float s = 0;
        for (int t = -3; t <= 3; t++) {
            int j = i + t;
            if (j < 0)
                j = -1 - j;
            else if (j >= n)
                j = 2 * n - 1 - j;
            s += f[3 - t] * re[j];
        }
        im[i] += s * (i & 1 ? odd : even);
    }
}

/* the filter of the current window (its sequence, its shape and the one
   before) and of the frame before (its shape) */
static const float *mdst_cur(const struct ics *ics)
{
    int win = ics->seq == LONG_START ? 1 : ics->seq == LONG_STOP ? 2 : 0;
    int shape = ics->shape == ics->prev_shape ? ics->shape
                : !ics->shape                 ? 2
                                              : 3;
    return USAC_MDST_CUR[win][shape];
}

static void complex_stereo(struct aac_dec *d, struct usac_st *st,
                           struct chan *l, struct chan *r)
{
    const struct ics *ics = &l->ics;
    int is8 = ics->seq == EIGHT_SHORT, s = is8 ? d->short_len : d->frame;
    int n = d->frame, m = st->max_ste;
    float sign = st->pred_dir ? -1.0f : 1.0f;
    /* the downmix: of this frame the transmitted one, of the frame before
       the mean of its decoded channels */
    memcpy(st->dmx, l->coef, sizeof(float) * (size_t)n);
    for (int k = 0; k < n; k++)
        st->dmx_prev[k] = 0.5f * (l->u->prev[k] + sign * r->u->prev[k]);
    memset(st->dmx_im, 0, sizeof(float) * (size_t)n);
    if (st->complex) {
        const float *fc = mdst_cur(ics);
        const float *fp =
            USAC_MDST_PREV[ics->seq == LONG_STOP][ics->prev_shape];
        for (int w = 0; w < ics->num_windows; w++)
            mdst(st->dmx_im + w * s, st->dmx + w * s, fc, s, 1, 1);
        if (st->use_prev && !is8)
            mdst(st->dmx_im, st->dmx_prev, fp, n, -1, 1);
    }
    int win = 0;
    for (int g = 0; g < ics->groups; g++) {
        for (int i = 0; i < m; i++) {
            if (!st->pred_used[g * m + i])
                continue;
            float ar = st->alpha_re[g * m + i], ai = st->alpha_im[g * m + i];
            for (int w = 0; w < ics->group_len[g]; w++) {
                float *a = l->coef + (win + w) * s,
                      *b = r->coef + (win + w) * s;
                const float *im = st->dmx_im + (win + w) * s;
                for (int k = ics->swb[i]; k < ics->swb[i + 1]; k++) {
                    float res = b[k] - ar * a[k] - ai * im[k];
                    if (!st->pred_dir) {
                        b[k] = a[k] - res;
                        a[k] = a[k] + res;
                    } else {
                        b[k] = res - a[k];
                        a[k] = res + a[k];
                    }
                }
            }
        }
        win += ics->group_len[g];
    }
}

static void mid_side(struct aac_dec *d, struct usac_st *st, struct chan *l,
                     struct chan *r)
{
    const struct ics *ics = &l->ics;
    int s = ics->seq == EIGHT_SHORT ? d->short_len : d->frame;
    int m = st->max_ste, win = 0;
    for (int g = 0; g < ics->groups; g++) {
        for (int i = 0; i < m; i++) {
            if (!st->ms[g * m + i])
                continue;
            for (int w = 0; w < ics->group_len[g]; w++) {
                float *a = l->coef + (win + w) * s,
                      *b = r->coef + (win + w) * s;
                for (int k = ics->swb[i]; k < ics->swb[i + 1]; k++) {
                    float mm = a[k], ss = b[k];
                    a[k] = mm + ss;
                    b[k] = mm - ss;
                }
            }
        }
        win += ics->group_len[g];
    }
}

/* ---- the elements ---- */

static int need_state(struct aac_dec *d, struct element *e)
{
    for (int j = 0; j < e->ch; j++) {
        struct chan *c = d->ch[e->first + j];
        if (!c->u) {
            if (!(c->u = calloc(1, sizeof *c->u)))
                return -1;
            c->u->seed = j ? 0x10932 : 0x3039;
        }
    }
    if (e->ch == 2 && !e->us && !(e->us = calloc(1, sizeof *e->us)))
        return -1;
    return 0;
}

/* UsacCoreCoderData of an element of nch channels: 0, or -1 with why */
static int core(struct aac_dec *d, struct abr *b, struct element *e,
                const struct usac_el *ec, int indep, char *why, size_t wn)
{
    int nch = e->ch;
    struct chan *c[2] = {d->ch[e->first], nch > 1 ? d->ch[e->first + 1] : NULL};
    struct usac_st *st = e->us;
    for (int j = 0; j < nch; j++) {
        c[j]->tns.present = 0;
        c[j]->u->tns_data = 0;
        if (abr_bit(b)) { /* core_mode */
            snprintf(why, wn,
                     "a USAC stream with linear-prediction coding "
                     "(not read)");
            return -1;
        }
    }
    if (nch == 2 && read_stereo(d, b, st, c[0], c[1], indep) != 0)
        goto bad;
    int common = nch == 2 && st->common;
    for (int j = 0; j < nch; j++) {
        struct chan *ch = c[j];
        struct usac_ch *u = ch->u;
        if (nch == 1)
            u->tns_data = abr_bit(b);
        int gain = (int)abr_bits(b, 8);
        u->noise_level = 0;
        if (ec->noise_fill) {
            u->noise_level = (int)abr_bits(b, 3);
            u->noise_offset = (int)abr_bits(b, 5);
        }
        if (!common && read_ics_info(d, b, ch) != 0)
            goto bad;
        if (read_sf(d, b, ch, gain) != 0)
            goto bad;
        if (u->tns_data) {
            ch->tns.present = 1;
            if (aac_read_tns(d, b, &ch->ics, &ch->tns) != 0)
                goto bad;
        }
        int reset = indep ? 1 : abr_bit(b); /* arith_reset_flag */
        const struct ics *ics = &ch->ics;
        int s = ics->seq == EIGHT_SHORT ? d->short_len : d->frame;
        for (int w = 0; w < ics->num_windows; w++)
            if (ari_spectrum(b, u, ch->coef + w * s, reset && !w,
                             ics->swb[ics->max_sfb], s) != 0)
                goto bad;
        if (abr_bit(b)) { /* fac_data_present */
            snprintf(why, wn,
                     "a USAC stream with linear-prediction coding "
                     "(not read)");
            return -1;
        }
    }
    for (int j = 0; j < nch; j++)
        scale(d, c[j]);
    if (common) {
        for (int j = 0; j < 2; j++)
            if (c[j]->tns.present && !st->tns_on_lr)
                aac_apply_tns(d, c[j]->coef, &c[j]->tns, &c[j]->ics, 0);
        if (st->ms_mode == 3)
            complex_stereo(d, st, c[0], c[1]);
        else if (st->ms_mode)
            mid_side(d, st, c[0], c[1]);
    }
    if (nch == 2) {
        for (int j = 0; j < 2; j++)
            memcpy(c[j]->u->prev, c[j]->coef, sizeof(float) * (size_t)d->frame);
        memcpy(st->prev_re, st->alpha_re, sizeof st->prev_re);
        memcpy(st->prev_im, st->alpha_im, sizeof st->prev_im);
        st->prev_groups = c[0]->ics.groups;
    }
    for (int j = 0; j < nch; j++) {
        if (c[j]->tns.present && (!common || st->tns_on_lr))
            aac_apply_tns(d, c[j]->coef, &c[j]->tns, &c[j]->ics, 0);
        aac_synthesize(d, c[j]);
    }
    return 0;
bad:
    snprintf(why, wn, "a USAC frame that is none");
    return -1;
}

/* an extension element's payload, skipped */
static int skip_ext(struct abr *b, const struct usac_el *ec)
{
    if (!abr_bit(b)) /* usacExtElementPresent */
        return 0;
    size_t len;
    if (abr_bit(b)) {
        len = ec->default_len;
    } else {
        len = abr_bits(b, 8);
        if (len == 255)
            len += abr_bits(b, 16) - 2;
    }
    if (len && ec->frag)
        abr_bits(b, 2); /* usacExtElementStart, Stop */
    b->pos += 8 * len;
    return abr_over(b) ? -1 : 0;
}

int usac_frame(struct aac_dec *d, struct abr *b, char *why, size_t wn)
{
    int indep = abr_bit(b), tags[4] = {0}, out = 0;
    for (int i = 0; i < d->cfg.usac_n; i++) {
        const struct usac_el *ec = &d->cfg.usac_el[i];
        if (ec->type == 3) {
            if (skip_ext(b, ec) != 0) {
                snprintf(why, wn, "a USAC frame cut short");
                return -1;
            }
            continue;
        }
        int type = ec->type == 2 ? 3 : ec->type;
        struct element *e = aac_element(d, type, tags[type]++);
        if (!e || need_state(d, e) != 0) {
            snprintf(why, wn, "a USAC stream of too many channels");
            return -1;
        }
        if (core(d, b, e, ec, indep, why, wn) != 0)
            return -1;
        for (int j = 0; j < e->ch && out < MAX_CHANNELS; j++)
            memcpy(d->outp[out++], d->ch[e->first + j]->out,
                   sizeof(float) * (size_t)d->frame);
    }
    if (over(b)) {
        snprintf(why, wn, "a USAC frame cut short");
        return -1;
    }
    d->out_ch = out;
    if (d->sbr_on < 0)
        d->sbr_on = 0;
    return d->frame;
}
