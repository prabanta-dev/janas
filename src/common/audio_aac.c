/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_aac.c - AAC's core decoder (ISO/IEC 14496-3, 4.4-4.6): the
 * AudioSpecificConfig, raw data blocks of single, pair and LFE channel
 * elements, the scalefactors and the Huffman-coded spectrum, pulses,
 * perceptual noise substitution, mid/side and intensity stereo, temporal
 * noise shaping, AAC Main's backward-adaptive prediction and AAC LTP's
 * long-term prediction, the inverse MDCT with sine and Kaiser-Bessel
 * windows, long and short. Fill elements carry the SBR data
 * (audio_aac_sbr.c) that doubles the rate of HE-AAC. Frames of 1024 and
 * of 960 samples. The error resilient syntax of ER AAC LC, AAC LD (frames
 * of 512 and 480, its low-overlap window) and AAC ELD (its low-delay filter
 * bank, audio_aac_eld.c), without the resilience tools, error protection
 * and ELD's low-delay SBR, as FFmpeg. The rounding of AAC Main's predictor is
 * libfaad's and FFmpeg's, the noise's generator FFmpeg's, so the output can be
 * compared with theirs. The decoder of a stream (aac_stream) mixes the channels
 * to mono and trims the start and the end as the container says.
 */
#include "audio_aac_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_aac_ld_tab.h"
#include "audio_aac_tab.h"
#include "audio_aac_usac_tab.h"
#include "audio_fft.h"

/* ---- Huffman codes ---- */

static int vlc_fill(struct vlc *v, int bits, const uint32_t *code,
                    const uint8_t *len, const uint8_t *sym, int n)
{
    size_t size = (size_t)1 << bits;
    /* the subtables: one for each prefix of bits bits that longer codes
       share, as wide as the longest of them */
    int sub[1 << 12];
    if (bits > 12)
        return -1;
    memset(sub, 0, sizeof(int) * size);
    for (int i = 0; i < n; i++) {
        if (len[i] > bits) {
            int extra = len[i] - bits;
            if (len[i] > 32 || extra > 16)
                return -1;
            uint32_t pre = code[i] >> extra;
            if (extra > sub[pre])
                sub[pre] = extra;
        }
    }
    size_t total = size;
    for (size_t p = 0; p < size; p++)
        total += sub[p] ? (size_t)1 << sub[p] : 0;
    if (total > 32767)
        return -1;
    v->t = calloc(total, sizeof *v->t);
    if (!v->t)
        return -1;
    v->bits = bits;
    size_t at = size;
    for (size_t p = 0; p < size; p++) {
        if (sub[p]) {
            v->t[p].sym = (int16_t)at;
            v->t[p].len = (int8_t)-sub[p];
            at += (size_t)1 << sub[p];
        }
    }
    for (int i = 0; i < n; i++) {
        int s = sym ? sym[i] : i;
        if (!len[i])
            continue;
        if (len[i] <= bits) {
            uint32_t first = code[i] << (bits - len[i]);
            uint32_t cnt = 1u << (bits - len[i]);
            for (uint32_t k = 0; k < cnt; k++) {
                struct vlc_e *e = &v->t[first + k];
                if (e->len)
                    return -1;
                e->sym = (int16_t)s;
                e->len = (int8_t)len[i];
            }
        } else {
            int extra = len[i] - bits;
            uint32_t pre = code[i] >> extra;
            int w = -v->t[pre].len;
            uint32_t low = code[i] & ((1u << extra) - 1);
            uint32_t first = low << (w - extra);
            uint32_t cnt = 1u << (w - extra);
            for (uint32_t k = 0; k < cnt; k++) {
                struct vlc_e *e = &v->t[v->t[pre].sym + (int)(first + k)];
                if (e->len)
                    return -1;
                e->sym = (int16_t)s;
                e->len = (int8_t)extra;
            }
        }
    }
    return 0;
}

int vlc_build(struct vlc *v, int bits, const uint32_t *code, const uint8_t *len,
              const uint8_t *sym, int n)
{
    v->t = NULL;
    if (vlc_fill(v, bits, code, len, sym, n) == 0)
        return 0;
    vlc_free(v);
    return -1;
}

void vlc_free(struct vlc *v)
{
    free(v->t);
    v->t = NULL;
}

/* ---- the configuration ---- */

static const int RATES[13] = {96000, 88200, 64000, 48000, 44100, 32000, 24000,
                              22050, 16000, 12000, 11025, 8000,  7350};

int aac_index_rate(int index)
{
    return index >= 0 && index < 13 ? RATES[index] : 0;
}

/* the index of the nearest rate's tables (14496-3, table 4.82) */
int aac_rate_index(int rate)
{
    static const int AT[12] = {92017, 75132, 55426, 46009, 37566, 27713,
                               23004, 18783, 13856, 11502, 9391,  0};
    for (int i = 0; i < 12; i++)
        if (rate >= AT[i])
            return i;
    return 11;
}

static int read_aot(struct abr *b)
{
    int aot = (int)abr_bits(b, 5);
    return aot == 31 ? 32 + (int)abr_bits(b, 6) : aot;
}

static int read_rate(struct abr *b, int *index)
{
    *index = (int)abr_bits(b, 4);
    if (*index == 15) {
        int rate = (int)abr_bits(b, 24);
        *index = aac_rate_index(rate);
        return rate;
    }
    return aac_index_rate(*index);
}

/* a program_config_element (4.4.1.1): the channels it makes, or -1 */
static int read_pce(struct abr *b)
{
    abr_bits(b, 4); /* element_instance_tag */
    abr_bits(b, 2); /* object_type */
    abr_bits(b, 4); /* sampling_frequency_index */
    int front = (int)abr_bits(b, 4), side = (int)abr_bits(b, 4);
    int back = (int)abr_bits(b, 4), lfe = (int)abr_bits(b, 2);
    int assoc = (int)abr_bits(b, 3), cc = (int)abr_bits(b, 4);
    if (abr_bit(b))
        abr_bits(b, 4); /* mono_mixdown */
    if (abr_bit(b))
        abr_bits(b, 4); /* stereo_mixdown */
    if (abr_bit(b))
        abr_bits(b, 3); /* matrix_mixdown */
    int ch = lfe;
    for (int i = 0; i < front + side + back; i++) {
        ch += abr_bit(b) ? 2 : 1;
        abr_bits(b, 4);
    }
    abr_bits(b, 4 * lfe);
    abr_bits(b, 4 * assoc);
    for (int i = 0; i < cc; i++)
        abr_bits(b, 5);
    b->pos = (b->pos + 7) & ~(size_t)7;
    b->pos += 8 * (size_t)abr_bits(b, 8); /* the comment */
    return abr_over(b) ? -1 : ch;
}

static const int CONFIG_CHANNELS[16] = {0, 1, 2, 3, 4, 5,  6, 8,
                                        0, 0, 0, 7, 8, 24, 8, 0};

/* the elements of an error resilient frame, in order, for each channel
   configuration: a single channel, a pair, an LFE */
static const char *const ER_LAYOUT[16] = {
    [1] = "S",    [2] = "C",     [3] = "SC",     [4] = "SCS",   [5] = "SCC",
    [6] = "SCCL", [7] = "SCCCL", [11] = "SCCSL", [12] = "SCCCL"};

static int is_er(int aot)
{
    return aot == AOT_ER_LC || aot == AOT_ER_LD || aot == AOT_ER_ELD;
}

/* the scalefactor bands of a low-delay frame: their count, 0 at the rates
   it is not defined for */
static int ld_bands(int frame, int index)
{
    return frame == 512 ? AAC_NUM_SWB_512[index] : AAC_NUM_SWB_480[index];
}

/* ELDSpecificConfig (4.4.1.1) after the channel configuration */
static int eld_config(struct abr *b, struct aac_cfg *c, char *why, size_t wn)
{
    c->frame = abr_bit(b) ? 480 : 512;
    if (abr_bits(b, 3)) {
        snprintf(why, wn, "an AAC-ELD with error resilience tools (not read)");
        return -1;
    }
    if (abr_bit(b)) {
        snprintf(why, wn, "an AAC-ELD with low-delay SBR (not read)");
        return -1;
    }
    while (abr_bits(b, 4) != 0) { /* eldExtType, until ELDEXT_TERM */
        size_t len = abr_bits(b, 4);
        if (len == 15)
            len += abr_bits(b, 8);
        if (len == 15 + 255)
            len += abr_bits(b, 16);
        b->pos += 8 * len;
        if (abr_over(b))
            break;
    }
    if (abr_bits(b, 2)) {
        snprintf(why, wn, "an AAC with error protection (not read)");
        return -1;
    }
    return 0;
}

int aac_config(const uint8_t *p, size_t n, struct aac_cfg *c, char *why,
               size_t wn)
{
    struct abr b = {.p = p, .n = n};
    memset(c, 0, sizeof *c);
    c->sbr = c->ps = -1;
    int aot = read_aot(&b);
    c->rate = read_rate(&b, &c->rate_index);
    c->chan_config = (int)abr_bits(&b, 4);
    if (aot == AOT_SBR || aot == AOT_PS) {
        c->sbr = 1;
        c->ps = aot == AOT_PS;
        int idx;
        c->sbr_rate = read_rate(&b, &idx);
        aot = read_aot(&b);
        if (aot == 22)
            abr_bits(&b, 4); /* extensionChannelConfiguration */
    }
    if (aot == AOT_USAC)
        return usac_config(&b, c, why, wn);
    if (aot != AOT_MAIN && aot != AOT_LC && aot != AOT_LTP && !is_er(aot)) {
        static const char *const NAMES[] = {
            [3] = "AAC SSR",
            [6] = "AAC Scalable",
            [8] = "CELP",
            [9] = "HVXC",
            [19] = "ER AAC LTP",
            [20] = "ER AAC Scalable",
            [22] = "ER BSAC",
            [34] = "MPEG-1/2 Layer 1",
            [35] = "MPEG-1/2 Layer 2",
            [36] = "ALS",
        };
        const char *name =
            aot >= 0 && aot < (int)(sizeof NAMES / sizeof *NAMES) && NAMES[aot]
                ? NAMES[aot]
                : NULL;
        if (name)
            snprintf(why, wn, "an AAC of a kind not read: %s", name);
        else
            snprintf(why, wn, "an AAC of a kind not read: object type %d", aot);
        return -1;
    }
    c->aot = aot;
    if (!c->rate || c->rate > 96000) {
        snprintf(why, wn, "an AAC configuration without a rate");
        return -1;
    }
    if (is_er(aot)) {
        if (!ER_LAYOUT[c->chan_config]) {
            snprintf(why, wn,
                     "an error resilient AAC of channel configuration %d",
                     c->chan_config);
            return -1;
        }
        c->channels = CONFIG_CHANNELS[c->chan_config];
        c->sbr = c->ps = 0;
    }
    if (aot == AOT_ER_ELD) {
        if (eld_config(&b, c, why, wn) != 0)
            return -1;
        goto checked;
    }
    /* GASpecificConfig */
    int fl = abr_bit(&b);
    c->frame = aot == AOT_ER_LD ? (fl ? 480 : 512) : (fl ? 960 : 1024);
    if (abr_bit(&b))
        abr_bits(&b, 14); /* coreCoderDelay */
    int ext = abr_bit(&b);
    if (is_er(aot)) {
        if (ext && abr_bits(&b, 3)) {
            snprintf(why, wn, "an AAC with error resilience tools (not read)");
            return -1;
        }
        if (ext)
            abr_bit(&b); /* extensionFlag3 */
        if (abr_bits(&b, 2)) {
            snprintf(why, wn, "an AAC with error protection (not read)");
            return -1;
        }
        goto checked;
    }
    if (c->chan_config == 0) {
        int ch = read_pce(&b);
        if (ch <= 0) {
            snprintf(why, wn, "an AAC program configuration that is none");
            return -1;
        }
        c->channels = ch;
    } else {
        c->channels = CONFIG_CHANNELS[c->chan_config];
    }
    if (ext)
        abr_bit(&b); /* extensionFlag3 */
    /* the explicit, backward compatible signal of SBR and PS */
    if (c->sbr < 0 && abr_left(&b) >= 16) {
        if (abr_bits(&b, 11) == 0x2b7) {
            int e = read_aot(&b);
            if (e == AOT_SBR) {
                c->sbr = abr_bit(&b);
                if (c->sbr) {
                    int idx;
                    c->sbr_rate = read_rate(&b, &idx);
                    if (abr_left(&b) >= 12 && abr_bits(&b, 11) == 0x548)
                        c->ps = abr_bit(&b);
                }
            }
        }
    }
    if (c->sbr == 1 && !c->sbr_rate)
        c->sbr_rate = 2 * c->rate;
checked:
    if (c->frame <= 512 && !ld_bands(c->frame, c->rate_index)) {
        snprintf(why, wn,
                 "a low-delay AAC at %d Hz, a rate it has no bands for",
                 c->rate);
        return -1;
    }
    if (abr_over(&b)) {
        snprintf(why, wn, "an AAC configuration cut short");
        return -1;
    }
    return 0;
}

/* ---- the decoder ---- */

static double bessel_i0(double x)
{
    double s = 1, t = 1;
    for (int k = 1; k < 50; k++) {
        t *= (x / (2 * k)) * (x / (2 * k));
        s += t;
    }
    return s;
}

/* FFmpeg's ff_kbd_window_init: the left half, n samples */
static void kbd_window(float *w, double alpha, int n)
{
    double temp[1025], scale = 0, sum = 0;
    double a2 = 4 * (alpha * M_PI / n) * (alpha * M_PI / n);
    for (int i = 0; i <= n / 2; i++) {
        temp[i] = bessel_i0(sqrt((double)i * (n - i) * a2));
        scale += temp[i] * (1 + (i && i < n / 2));
    }
    scale = 1.0 / (scale + 1);
    int i;
    for (i = 0; i <= n / 2; i++) {
        sum += temp[i];
        w[i] = (float)sqrt(sum * scale);
    }
    for (; i < n; i++) {
        sum += temp[n - i];
        w[i] = (float)sqrt(sum * scale);
    }
}

static void sine_window(float *w, int n)
{
    for (int i = 0; i < n; i++)
        w[i] = (float)sin((i + 0.5) * (M_PI / (2.0 * n)));
}

void aac_free(struct aac_dec *d)
{
    if (!d)
        return;
    vlc_free(&d->sf_vlc);
    for (int i = 0; i < 11; i++)
        vlc_free(&d->spec_vlc[i]);
    aimdct_free(d->mdct_long);
    aimdct_free(d->mdct_short);
    for (int i = 0; i < d->n_el; i++) {
        sbr_free(d->el[i].sbr);
        free(d->el[i].cce);
        free(d->el[i].us);
    }
    for (int i = 0; i < MAX_CHANNELS; i++) {
        if (d->ch[i])
            free(d->ch[i]->u);
        free(d->ch[i]);
        free(d->outp[i]);
    }
    free(d);
}

struct aac_dec *aac_new(const struct aac_cfg *c)
{
    struct aac_dec *d = calloc(1, sizeof *d);
    if (!d)
        return NULL;
    d->cfg = *c;
    d->frame = c->frame;
    /* low delay: no short windows, the low-overlap window's slopes a quarter
       of the frame */
    d->short_len = c->frame <= 512 ? c->frame / 4 : c->frame / 8;
    d->noise_state = 0x1f2e3d4c;
    d->sbr_on = c->sbr == 0 ? 0 : -1;
    d->sbr_down = c->sbr == 1 && c->sbr_rate && c->sbr_rate < 2 * c->rate;
    static const uint16_t *const CODES[11] = {
        AAC_CODES1, AAC_CODES2, AAC_CODES3, AAC_CODES4,  AAC_CODES5, AAC_CODES6,
        AAC_CODES7, AAC_CODES8, AAC_CODES9, AAC_CODES10, AAC_CODES11};
    static const uint8_t *const BITS[11] = {
        AAC_BITS1, AAC_BITS2, AAC_BITS3, AAC_BITS4,  AAC_BITS5, AAC_BITS6,
        AAC_BITS7, AAC_BITS8, AAC_BITS9, AAC_BITS10, AAC_BITS11};
    static const int SIZES[11] = {81, 81, 81,  81,  81, 81,
                                  64, 64, 169, 169, 289};
    int bad = vlc_build(&d->sf_vlc, 9, AAC_SF_CODE, AAC_SF_BITS, NULL, 121);
    for (int i = 0; i < 11; i++) {
        uint32_t code[289];
        for (int k = 0; k < SIZES[i]; k++)
            code[k] = CODES[i][k];
        if (bad ||
            vlc_build(&d->spec_vlc[i], 9, code, BITS[i], NULL, SIZES[i]) != 0) {
            bad = 1;
            /* the sbr and vlc fields are zero: aac_free copes */
        }
    }
    d->mdct_long = aimdct_new(2 * d->frame);
    d->mdct_short = aimdct_new(2 * d->short_len);
    if (bad || !d->mdct_long || !d->mdct_short) {
        aac_free(d);
        return NULL;
    }
    sine_window(d->sine_long, d->frame);
    sine_window(d->sine_short, d->short_len);
    kbd_window(d->kbd_long, 4.0, d->frame);
    kbd_window(d->kbd_short, 6.0, d->short_len);
    for (int i = 0; i < 8192; i++)
        d->pow43[i] = (float)(cbrt((double)i) * i);
    return d;
}

int aac_out_rate(const struct aac_dec *d)
{
    return d->sbr_on == 1 && !d->sbr_down ? 2 * d->cfg.rate : d->cfg.rate;
}

int aac_out(const struct aac_dec *d, float ***pcm)
{
    *pcm = (float **)d->outp;
    return d->out_ch;
}

/* ---- the syntax ---- */

const uint16_t *aac_swb_table(int frame, int index, int *num)
{
    const uint16_t *t;
    switch (frame) {
    case 1024:
        *num = AAC_NUM_SWB_1024[index];
        t = AAC_SWB_1024 + AAC_SWB_1024_AT[index];
        break;
    case 960:
        *num = AAC_NUM_SWB_960[index];
        t = AAC_SWB_960 + AAC_SWB_960_AT[index];
        break;
    case 512:
        *num = AAC_NUM_SWB_512[index];
        t = AAC_SWB_512 + AAC_SWB_512_AT[index];
        break;
    case 480:
        *num = AAC_NUM_SWB_480[index];
        t = AAC_SWB_480 + AAC_SWB_480_AT[index];
        break;
    case 768:
        *num = AAC_NUM_SWB_768[index];
        t = AAC_SWB_768 + AAC_SWB_768_AT[index];
        break;
    case 96:
        *num = AAC_NUM_SWB_96[index];
        t = AAC_SWB_96 + AAC_SWB_96_AT[index];
        break;
    case 128:
        *num = AAC_NUM_SWB_128[index];
        t = AAC_SWB_128 + AAC_SWB_128_AT[index];
        break;
    default:
        *num = AAC_NUM_SWB_120[index];
        t = AAC_SWB_120 + AAC_SWB_120_AT[index];
        break;
    }
    return t;
}

static int read_ltp(struct abr *b, struct ics *ics)
{
    ics->ltp_lag = (int)abr_bits(b, 11);
    ics->ltp_coef = AAC_LTP_COEF[abr_bits(b, 3)];
    int n = ics->max_sfb < MAX_LTP_SFB ? ics->max_sfb : MAX_LTP_SFB;
    for (int i = 0; i < MAX_LTP_SFB; i++)
        ics->ltp_used[i] = i < n ? abr_bit(b) : 0;
    return 0;
}

/* ics_info (4.4.2.1): 0, or -1 */
static int read_ics_info(struct aac_dec *d, struct abr *b, struct ics *ics)
{
    int eld = d->cfg.aot == AOT_ER_ELD;
    ics->prev_shape = ics->shape;
    if (!eld) { /* ELD's are long, of one shape */
        if (abr_bit(b))
            return -1; /* reserved */
        ics->seq = (int)abr_bits(b, 2);
        ics->shape = abr_bit(b);
    }
    if (d->frame <= 512 && ics->seq != ONLY_LONG)
        return -1; /* low delay: long windows only */
    ics->pred_present = ics->ltp_present = 0;
    ics->pred_reset = 0;
    if (ics->seq == EIGHT_SHORT) {
        ics->max_sfb = (int)abr_bits(b, 4);
        int grouping = (int)abr_bits(b, 7);
        ics->groups = 1;
        ics->group_len[0] = 1;
        for (int i = 6; i >= 0; i--) {
            if (grouping >> i & 1)
                ics->group_len[ics->groups - 1]++;
            else
                ics->group_len[ics->groups++] = 1;
        }
        ics->num_windows = 8;
        ics->swb =
            aac_swb_table(d->short_len, d->cfg.rate_index, &ics->num_swb);
    } else {
        ics->max_sfb = (int)abr_bits(b, 6);
        ics->groups = 1;
        ics->group_len[0] = 1;
        ics->num_windows = 1;
        ics->swb = aac_swb_table(d->frame, d->cfg.rate_index, &ics->num_swb);
        if (!eld && (ics->pred_present = abr_bit(b))) {
            if (d->cfg.aot == AOT_MAIN) {
                if (abr_bit(b)) {
                    ics->pred_reset = (int)abr_bits(b, 5);
                    if (ics->pred_reset == 0 || ics->pred_reset > 30)
                        return -1;
                }
                int max = AAC_PRED_SFB_MAX[d->cfg.rate_index];
                int n = ics->max_sfb < max ? ics->max_sfb : max;
                for (int i = 0; i < 41; i++)
                    ics->pred_used[i] = i < n ? abr_bit(b) : 0;
            } else if (d->cfg.aot == AOT_LTP) {
                if ((ics->ltp_present = abr_bit(b)))
                    read_ltp(b, ics);
            } else {
                return -1; /* prediction in AAC LC, LTP in AAC LD */
            }
        }
    }
    return ics->max_sfb > ics->num_swb ? -1 : 0;
}

static int read_section_data(struct abr *b, struct ics *ics, uint8_t *cb)
{
    int bits = ics->seq == EIGHT_SHORT ? 3 : 5;
    int esc = (1 << bits) - 1;
    for (int g = 0; g < ics->groups; g++) {
        int k = 0;
        while (k < ics->max_sfb) {
            int c = (int)abr_bits(b, 4);
            if (c == 12)
                return -1;
            int len = 0, inc;
            do {
                inc = (int)abr_bits(b, bits);
                len += inc;
                if (abr_over(b))
                    return -1;
            } while (inc == esc);
            if (k + len > ics->max_sfb)
                return -1;
            for (int i = 0; i < len; i++)
                cb[g * 16 + k + i] = (uint8_t)c;
            k += len;
        }
        for (; k < (ics->seq == EIGHT_SHORT ? 16 : 64); k++)
            cb[g * (ics->seq == EIGHT_SHORT ? 16 : 0) + k] = ZERO_BT;
    }
    return 0;
}

/* the scalefactors (4.6.2.3): 0, or -1 */
static int read_scalefactors(struct aac_dec *d, struct abr *b, struct ics *ics,
                             const uint8_t *cb, int *sf, int global_gain)
{
    int offset[3] = {global_gain, 0, global_gain - 90};
    int noise_first = 1;
    int stride = ics->seq == EIGHT_SHORT ? 16 : 0;
    for (int g = 0; g < ics->groups; g++) {
        for (int i = 0; i < ics->max_sfb; i++) {
            int idx = g * stride + i, v;
            switch (cb[idx]) {
            case ZERO_BT:
                sf[idx] = 0;
                break;
            case INTENSITY_BT:
            case INTENSITY_BT2:
                if ((v = vlc_get(b, &d->sf_vlc)) < 0)
                    return -1;
                offset[1] += v - 60;
                sf[idx] = offset[1] < -155  ? -155
                          : offset[1] > 100 ? 100
                                            : offset[1];
                break;
            case NOISE_BT:
                if (noise_first) {
                    offset[2] += (int)abr_bits(b, 9) - 256;
                    noise_first = 0;
                } else {
                    if ((v = vlc_get(b, &d->sf_vlc)) < 0)
                        return -1;
                    offset[2] += v - 60;
                }
                sf[idx] = offset[2] < -100  ? -100
                          : offset[2] > 155 ? 155
                                            : offset[2];
                break;
            default:
                if ((v = vlc_get(b, &d->sf_vlc)) < 0)
                    return -1;
                offset[0] += v - 60;
                if (offset[0] < 0 || offset[0] > 255)
                    return -1;
                sf[idx] = offset[0];
                break;
            }
        }
    }
    return 0;
}

int aac_read_tns(struct aac_dec *d, struct abr *b, const struct ics *ics,
                 struct tns *t)
{
    int is8 = ics->seq == EIGHT_SHORT;
    for (int w = 0; w < ics->num_windows; w++) {
        t->n_filt[w] = (int)abr_bits(b, 2 - is8);
        if (!t->n_filt[w])
            continue;
        int res = abr_bit(b);
        for (int f = 0; f < t->n_filt[w]; f++) {
            t->length[w][f] = (int)abr_bits(b, 6 - 2 * is8);
            /* USAC's orders take a bit less and have no limit of their own */
            int usac = d->cfg.aot == AOT_USAC;
            int order = (int)abr_bits(b, usac ? 4 - is8 : 5 - 2 * is8);
            if (!usac && order > (is8 ? 7 : 20))
                return -1;
            t->order[w][f] = order;
            if (!order)
                continue;
            t->dir[w][f] = abr_bit(b);
            int compress = abr_bit(b);
            int bits = res + 3 - compress;
            const float *map = compress ? (res ? AAC_TNS_1_4 : AAC_TNS_1_3)
                                        : (res ? AAC_TNS_0_4 : AAC_TNS_0_3);
            for (int i = 0; i < order; i++)
                t->coef[w][f][i] = map[abr_bits(b, bits)];
        }
    }
    return 0;
}

static uint32_t noise_next(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

/* the spectral data, dequantized, with noise in the noise bands (4.6.3,
   4.6.13): 0, or -1 */
static int read_spectrum(struct aac_dec *d, struct abr *b, struct chan *c,
                         const int *pulse)
{
    struct ics *ics = &c->ics;
    float *coef = c->coef;
    int stride = ics->seq == EIGHT_SHORT ? 16 : 0;
    int wlen = ics->seq == EIGHT_SHORT ? d->short_len : d->frame;
    memset(coef, 0, sizeof(float) * (size_t)d->frame);
    int win = 0;
    for (int g = 0; g < ics->groups; g++) {
        int glen = ics->group_len[g];
        for (int i = 0; i < ics->max_sfb; i++) {
            int idx = g * stride + i, cbt = c->cb[idx];
            int lo = ics->swb[i], hi = ics->swb[i + 1];
            if (cbt == ZERO_BT || cbt >= INTENSITY_BT2)
                continue;
            if (cbt == NOISE_BT) {
                for (int w = 0; w < glen; w++) {
                    float *x = coef + (win + w) * wlen;
                    float energy = 0;
                    for (int k = lo; k < hi; k++) {
                        x[k] = (float)(int32_t)noise_next(&d->noise_state);
                        energy += x[k] * x[k];
                    }
                    float scale =
                        (float)pow(2.0, 0.25 * c->sf[idx]) / sqrtf(energy);
                    for (int k = lo; k < hi; k++)
                        x[k] *= scale;
                }
                continue;
            }
            const struct vlc *v = &d->spec_vlc[cbt - 1];
            int dim4 = cbt < 5, unsign = cbt == 3 || cbt == 4 || cbt >= 7;
            int mod = cbt < 5    ? 3
                      : cbt < 7  ? 9
                      : cbt < 9  ? 8
                      : cbt < 11 ? 13
                                 : 17;
            int off = unsign ? 0 : mod / 2;
            for (int w = 0; w < glen; w++) {
                float *x = coef + (win + w) * wlen;
                for (int k = lo; k < hi; k += dim4 ? 4 : 2) {
                    int s = vlc_get(b, v);
                    if (s < 0)
                        return -1;
                    int q[4], n = dim4 ? 4 : 2;
                    if (dim4) {
                        q[0] = s / 27 - off;
                        q[1] = s / 9 % 3 - off;
                        q[2] = s / 3 % 3 - off;
                        q[3] = s % 3 - off;
                    } else {
                        q[0] = s / mod - off;
                        q[1] = s % mod - off;
                    }
                    if (unsign) {
                        for (int j = 0; j < n; j++)
                            if (q[j] && abr_bit(b))
                                q[j] = -q[j];
                        if (cbt == 11) {
                            for (int j = 0; j < 2; j++) {
                                if (q[j] != 16 && q[j] != -16)
                                    continue;
                                int nb = 4;
                                while (abr_bit(b)) {
                                    if (++nb > 12)
                                        return -1;
                                }
                                int e = (1 << nb) + (int)abr_bits(b, nb);
                                q[j] = q[j] < 0 ? -e : e;
                            }
                        }
                    }
                    for (int j = 0; j < n && k + j < hi; j++)
                        x[k + j] = q[j];
                }
                if (abr_over(b))
                    return -1;
            }
        }
        win += glen;
    }
    /* pulses, on the quantized values of coded bands (4.6.4) */
    if (pulse) {
        int at = ics->swb[pulse[1]], sfb = pulse[1];
        for (int i = 0; i < pulse[0]; i++) {
            at += pulse[2 + 2 * i];
            if (at >= d->frame)
                return -1;
            while (sfb < ics->num_swb && ics->swb[sfb + 1] <= at)
                sfb++;
            if (sfb >= ics->max_sfb || c->cb[sfb] == ZERO_BT ||
                c->cb[sfb] >= NOISE_BT)
                continue;
            float q = coef[at];
            int amp = pulse[3 + 2 * i];
            coef[at] = q > 0 ? q + amp : q - amp;
        }
    }
    /* dequantized */
    win = 0;
    for (int g = 0; g < ics->groups; g++) {
        int glen = ics->group_len[g];
        for (int i = 0; i < ics->max_sfb; i++) {
            int idx = g * stride + i, cbt = c->cb[idx];
            if (cbt == ZERO_BT || cbt >= NOISE_BT)
                continue;
            float scale = (float)pow(2.0, 0.25 * (c->sf[idx] - 100));
            for (int w = 0; w < glen; w++) {
                float *x = coef + (win + w) * wlen;
                for (int k = ics->swb[i]; k < ics->swb[i + 1]; k++) {
                    float q = x[k];
                    int a = (int)fabsf(q);
                    if (a >= 8192)
                        return -1;
                    x[k] = q < 0 ? -d->pow43[a] * scale : d->pow43[a] * scale;
                }
            }
        }
        win += glen;
    }
    return 0;
}

/* individual_channel_stream (4.4.2.7): 0, or -1 */
static int read_ics(struct aac_dec *d, struct abr *b, struct chan *c,
                    int common_window)
{
    int global_gain = (int)abr_bits(b, 8);
    if (!common_window && read_ics_info(d, b, &c->ics) != 0)
        return -1;
    if (read_section_data(b, &c->ics, c->cb) != 0)
        return -1;
    if (read_scalefactors(d, b, &c->ics, c->cb, c->sf, global_gain) != 0)
        return -1;
    int er = is_er(d->cfg.aot), eld = d->cfg.aot == AOT_ER_ELD;
    int pulse[2 + 8], *pp = NULL;
    if (!eld && abr_bit(b)) {
        if (c->ics.seq == EIGHT_SHORT)
            return -1;
        pulse[0] = (int)abr_bits(b, 2) + 1;
        pulse[1] = (int)abr_bits(b, 6);
        if (pulse[1] >= c->ics.num_swb)
            return -1;
        for (int i = 0; i < pulse[0]; i++) {
            pulse[2 + 2 * i] = (int)abr_bits(b, 5);
            pulse[3 + 2 * i] = (int)abr_bits(b, 4);
        }
        pp = pulse;
    }
    /* the error resilient syntax has TNS's data after the gain control's
       flag */
    c->tns.present = abr_bit(b);
    if (c->tns.present && !er && aac_read_tns(d, b, &c->ics, &c->tns) != 0)
        return -1;
    if (!eld && abr_bit(b))
        return -1; /* gain control: AAC SSR only */
    if (c->tns.present && er && aac_read_tns(d, b, &c->ics, &c->tns) != 0)
        return -1;
    if (read_spectrum(d, b, c, pp) != 0)
        return -1;
    return abr_over(b) ? -1 : 0;
}

/* ---- the tools ---- */

static float flt16_round(float f)
{
    uint32_t i;
    memcpy(&i, &f, 4);
    i = (i + 0x00008000u) & 0xFFFF0000u;
    memcpy(&f, &i, 4);
    return f;
}

static float flt16_even(float f)
{
    uint32_t i;
    memcpy(&i, &f, 4);
    i = (i + 0x00007FFFu + (i & 0x00010000u >> 16)) & 0xFFFF0000u;
    memcpy(&f, &i, 4);
    return f;
}

static float flt16_trunc(float f)
{
    uint32_t i;
    memcpy(&i, &f, 4);
    i &= 0xFFFF0000u;
    memcpy(&f, &i, 4);
    return f;
}

static void pred_reset(struct pred *p)
{
    p->r0 = p->r1 = p->cor0 = p->cor1 = 0;
    p->var0 = p->var1 = 1;
}

static void predict(struct pred *ps, float *coef, int output)
{
    const float a = 0.953125f, alpha = 0.90625f;
    float r0 = ps->r0, r1 = ps->r1;
    float cor0 = ps->cor0, cor1 = ps->cor1;
    float var0 = ps->var0, var1 = ps->var1;
    float k1 = var0 > 1 ? cor0 * flt16_even(a / var0) : 0;
    float k2 = var1 > 1 ? cor1 * flt16_even(a / var1) : 0;
    float pv = flt16_round(k1 * r0 + k2 * r1);
    if (output)
        *coef += pv;
    float e0 = *coef, e1 = e0 - k1 * r0;
    ps->cor1 = flt16_trunc(alpha * cor1 + r1 * e1);
    ps->var1 = flt16_trunc(alpha * var1 + 0.5f * (r1 * r1 + e1 * e1));
    ps->cor0 = flt16_trunc(alpha * cor0 + r0 * e0);
    ps->var0 = flt16_trunc(alpha * var0 + 0.5f * (r0 * r0 + e0 * e0));
    ps->r1 = flt16_trunc(a * (r0 - k1 * e0));
    ps->r0 = flt16_trunc(a * e0);
}

/* AAC Main's prediction (4.6.6) */
static void apply_prediction(struct aac_dec *d, struct chan *c)
{
    struct ics *ics = &c->ics;
    if (!c->pred_init) {
        for (int i = 0; i < MAX_PRED; i++)
            pred_reset(&c->pred[i]);
        c->pred_init = 1;
    }
    if (ics->seq == EIGHT_SHORT) {
        for (int i = 0; i < MAX_PRED; i++)
            pred_reset(&c->pred[i]);
        return;
    }
    int max = AAC_PRED_SFB_MAX[d->cfg.rate_index];
    for (int sfb = 0; sfb < max; sfb++)
        for (int k = ics->swb[sfb]; k < ics->swb[sfb + 1] && k < MAX_PRED; k++)
            predict(&c->pred[k], &c->coef[k],
                    ics->pred_present && ics->pred_used[sfb]);
    if (ics->pred_reset)
        for (int i = ics->pred_reset - 1; i < MAX_PRED; i += 30)
            pred_reset(&c->pred[i]);
}

/* mid/side (4.6.8.1), on the bands that are neither noise nor intensity */
void aac_apply_ms(struct aac_dec *d, struct chan *l, struct chan *r,
                  const uint8_t *ms)
{
    const struct ics *ics = &l->ics;
    int stride = ics->seq == EIGHT_SHORT ? 16 : 0;
    int wlen = ics->seq == EIGHT_SHORT ? d->short_len : d->frame;
    int win = 0;
    for (int g = 0; g < ics->groups; g++) {
        for (int i = 0; i < ics->max_sfb; i++) {
            int idx = g * stride + i;
            if (!ms[idx] || l->cb[idx] >= NOISE_BT || r->cb[idx] >= NOISE_BT)
                continue;
            for (int w = 0; w < ics->group_len[g]; w++) {
                float *a = l->coef + (win + w) * wlen;
                float *b = r->coef + (win + w) * wlen;
                for (int k = ics->swb[i]; k < ics->swb[i + 1]; k++) {
                    float m = a[k], s = b[k];
                    a[k] = m + s;
                    b[k] = m - s;
                }
            }
        }
        win += ics->group_len[g];
    }
}

/* intensity stereo (4.6.8.2) */
static void apply_intensity(struct aac_dec *d, struct chan *l, struct chan *r,
                            const uint8_t *ms, int ms_present)
{
    const struct ics *ics = &r->ics;
    int stride = ics->seq == EIGHT_SHORT ? 16 : 0;
    int wlen = ics->seq == EIGHT_SHORT ? d->short_len : d->frame;
    int win = 0;
    for (int g = 0; g < ics->groups; g++) {
        for (int i = 0; i < ics->max_sfb; i++) {
            int idx = g * stride + i, bt = r->cb[idx];
            if (bt != INTENSITY_BT && bt != INTENSITY_BT2)
                continue;
            float c = bt == INTENSITY_BT ? 1.0f : -1.0f;
            if (ms_present)
                c *= ms[idx] ? -1.0f : 1.0f;
            float scale = c * (float)pow(2.0, -0.25 * r->sf[idx]);
            for (int w = 0; w < ics->group_len[g]; w++) {
                const float *a = l->coef + (win + w) * wlen;
                float *b = r->coef + (win + w) * wlen;
                for (int k = ics->swb[i]; k < ics->swb[i + 1]; k++)
                    b[k] = a[k] * scale;
            }
        }
        win += ics->group_len[g];
    }
}

/* temporal noise shaping (4.6.9): the all-pole filter, or with inverse
   the all-zero one LTP's prediction takes */
void aac_apply_tns(struct aac_dec *d, float *coef, const struct tns *t,
                   const struct ics *ics, int inverse)
{
    int is8 = ics->seq == EIGHT_SHORT;
    int wlen = is8 ? d->short_len : d->frame;
    int usac = d->cfg.aot == AOT_USAC;
    int mmm = usac && is8       ? USAC_TNS_MAX_BANDS_128[d->cfg.rate_index]
              : usac            ? USAC_TNS_MAX_BANDS_1024[d->cfg.rate_index]
              : is8             ? AAC_TNS_MAX_BANDS_128[d->cfg.rate_index]
              : d->frame == 512 ? AAC_TNS_MAX_BANDS_512[d->cfg.rate_index]
              : d->frame == 480 ? AAC_TNS_MAX_BANDS_480[d->cfg.rate_index]
                                : AAC_TNS_MAX_BANDS_1024[d->cfg.rate_index];
    if (mmm > ics->max_sfb)
        mmm = ics->max_sfb;
    if (!mmm)
        return;
    for (int w = 0; w < ics->num_windows; w++) {
        int bottom = ics->num_swb;
        for (int f = 0; f < t->n_filt[w]; f++) {
            int top = bottom;
            bottom = top - t->length[w][f];
            if (bottom < 0)
                bottom = 0;
            int order = t->order[w][f];
            if (!order)
                continue;
            /* the reflection coefficients to the LPC ones */
            float lpc[20];
            for (int i = 0; i < order; i++) {
                float r = -t->coef[w][f][i];
                lpc[i] = r;
                for (int j = 0; j < (i + 1) >> 1; j++) {
                    float fj = lpc[j], bj = lpc[i - 1 - j];
                    lpc[j] = fj + r * bj;
                    lpc[i - 1 - j] = bj + r * fj;
                }
            }
            int start = ics->swb[bottom < mmm ? bottom : mmm];
            int end = ics->swb[top < mmm ? top : mmm];
            int size = end - start;
            if (size <= 0)
                continue;
            int inc = 1;
            if (t->dir[w][f]) {
                inc = -1;
                start = end - 1;
            }
            float *x = coef + w * wlen;
            if (!inverse) {
                for (int m = 0; m < size; m++, start += inc)
                    for (int i = 1; i <= (m < order ? m : order); i++)
                        x[start] -= x[start - i * inc] * lpc[i - 1];
            } else {
                float tmp[1024];
                for (int m = 0; m < size; m++, start += inc) {
                    tmp[m] = x[start];
                    for (int i = 1; i <= (m < order ? m : order); i++)
                        x[start] += tmp[m - i] * lpc[i - 1];
                }
            }
        }
    }
}

/* the window of a long frame (or the left of the first short): left
   half by the previous shape, right half by this one */
static void long_window(struct aac_dec *d, const struct ics *ics, float *w)
{
    int n = d->frame, s = d->short_len;
    /* AAC LD's second shape is no Kaiser-Bessel window but the low-overlap
       one: flat, its slopes a sine of a quarter of the frame */
    int ld = d->cfg.aot == AOT_ER_LD;
    int stop = ics->seq == LONG_STOP || (ld && ics->prev_shape);
    int start = ics->seq == LONG_START || (ld && ics->shape);
    const float *lp = ics->prev_shape && !ld ? d->kbd_long : d->sine_long;
    const float *lc = ics->shape && !ld ? d->kbd_long : d->sine_long;
    const float *sp = ics->prev_shape && !ld ? d->kbd_short : d->sine_short;
    const float *sc = ics->shape && !ld ? d->kbd_short : d->sine_short;
    int flat = (n - s) / 2; /* 448; LD 192 */
    for (int i = 0; i < n; i++) {
        if (stop)
            w[i] = i < flat ? 0 : i < flat + s ? sp[i - flat] : 1;
        else
            w[i] = lp[i];
    }
    for (int i = 0; i < n; i++) {
        float v;
        if (start)
            v = i < flat ? 1 : i < flat + s ? sc[s - 1 - (i - flat)] : 0;
        else
            v = lc[n - 1 - i];
        w[n + i] = v;
    }
}

/* the forward MDCT of 2n windowed samples to n coefficients, as an
   encoder makes them (LTP's prediction): X[k] = 2 sum x[i] cos(...) */
static void mdct_forward(struct aac_dec *d, const float *x, float *X)
{
    int n = d->frame, h = n / 2;
    float *u = d->fwd;
    for (int i = 0; i < h; i++) {
        u[i] = -x[3 * h - 1 - i] - x[3 * h + i];
        u[h + i] = x[i] - x[n - 1 - i];
    }
    aimdct_dct4(d->mdct_long, u, 1, X);
}

/* AAC LTP's prediction (4.6.7), added to the spectrum's bands */

static void apply_ltp(struct aac_dec *d, struct chan *c)
{
    struct ics *ics = &c->ics;
    if (!ics->ltp_present || ics->seq == EIGHT_SHORT)
        return;
    int n = d->frame;
    float *pt = d->buf, w[2048], pf[1024];
    int num = ics->ltp_lag < n ? ics->ltp_lag + n : 2 * n;
    for (int i = 0; i < 2 * n; i++)
        pt[i] = i < num && i + 2 * n - ics->ltp_lag >= 0
                    ? c->ltp_state[i + 2 * n - ics->ltp_lag] * ics->ltp_coef
                    : 0;
    long_window(d, ics, w);
    for (int i = 0; i < 2 * n; i++)
        pt[i] *= w[i] * 32768.0f * 2.0f;
    mdct_forward(d, pt, pf);
    if (c->tns.present)
        aac_apply_tns(d, pf, &c->tns, ics, 1);
    int max = ics->max_sfb < MAX_LTP_SFB ? ics->max_sfb : MAX_LTP_SFB;
    for (int sfb = 0; sfb < max; sfb++)
        if (ics->ltp_used[sfb])
            for (int k = ics->swb[sfb]; k < ics->swb[sfb + 1]; k++)
                c->coef[k] += pf[k];
}

/* the inverse MDCT, windows and overlap (4.6.11, 4.6.18), into c->out
   (n samples) and c->overlap; z: the 2n samples windowed */
void aac_synthesize(struct aac_dec *d, struct chan *c)
{
    struct ics *ics = &c->ics;
    int n = d->frame, s = d->short_len;
    float *z = d->buf;
    float y[2048];
    if (d->cfg.aot == AOT_ER_ELD) {
        aac_eld_synth(d->mdct_long, n, c->coef, c->eld, c->out, y, z);
        return;
    }
    if (ics->seq == EIGHT_SHORT) {
        const float *sp = ics->prev_shape ? d->kbd_short : d->sine_short;
        const float *sc = ics->shape ? d->kbd_short : d->sine_short;
        int flat = (n - s) / 2;
        memset(z, 0, sizeof(float) * 2 * (size_t)n);
        for (int w = 0; w < 8; w++) {
            aimdct_run(d->mdct_short, c->coef + w * s, y);
            const float *left = w ? sc : sp;
            float *o = z + flat + w * s;
            float g = 1.0f / (float)s / 32768.0f;
            for (int i = 0; i < s; i++) {
                o[i] += y[i] * left[i] * g;
                o[s + i] += y[s + i] * sc[s - 1 - i] * g;
            }
        }
    } else {
        float w[2048];
        aimdct_run(d->mdct_long, c->coef, y);
        long_window(d, ics, w);
        float g = 1.0f / (float)n / 32768.0f;
        for (int i = 0; i < 2 * n; i++)
            z[i] = y[i] * w[i] * g;
    }
    for (int i = 0; i < n; i++) {
        c->out[i] = z[i] + c->overlap[i];
        c->overlap[i] = z[n + i];
    }
    if (d->cfg.aot == AOT_LTP) {
        memmove(c->ltp_state, c->ltp_state + n, sizeof(float) * (size_t)n);
        memcpy(c->ltp_state + n, c->out, sizeof(float) * (size_t)n);
        memcpy(c->ltp_state + 2 * n, z + n, sizeof(float) * (size_t)n);
    }
}

/* ---- the elements ---- */

/* the element of type and tag, made at its first sight */
struct element *aac_element(struct aac_dec *d, int type, int tag)
{
    for (int i = 0; i < d->n_el; i++)
        if (d->el[i].type == type && d->el[i].tag == tag)
            return &d->el[i];
    int ch = type == 1 ? 2 : 1;
    if (d->n_el == MAX_ELEMENTS || d->n_ch + ch > MAX_CHANNELS)
        return NULL;
    struct element *e = &d->el[d->n_el];
    for (int k = 0; k < ch; k++) {
        if (!d->ch[d->n_ch + k] &&
            !(d->ch[d->n_ch + k] = calloc(1, sizeof(struct chan))))
            return NULL;
        if (!d->outp[d->n_ch + k] &&
            !(d->outp[d->n_ch + k] = calloc(2 * 2048, sizeof(float))))
            return NULL;
    }
    e->type = type;
    e->tag = tag;
    e->first = d->n_ch;
    e->ch = ch;
    e->sbr = NULL;
    e->cce = NULL;
    e->us = NULL;
    d->n_ch += ch;
    d->n_el++;
    return e;
}

static int read_sce(struct aac_dec *d, struct abr *b, struct element *e)
{
    struct chan *c = d->ch[e->first];
    if (read_ics(d, b, c, 0) != 0)
        return -1;
    if (d->cfg.aot == AOT_MAIN)
        apply_prediction(d, c);
    return 0;
}

static int read_cpe(struct aac_dec *d, struct abr *b, struct element *e)
{
    struct chan *l = d->ch[e->first], *r = d->ch[e->first + 1];
    int common = d->cfg.aot == AOT_ER_ELD || abr_bit(b), ms_present = 0;
    uint8_t ms[128] = {0};
    if (common) {
        if (read_ics_info(d, b, &l->ics) != 0)
            return -1;
        int shape = r->ics.shape;
        r->ics = l->ics;
        r->ics.prev_shape = shape;
        if (r->ics.pred_present && d->cfg.aot == AOT_LTP)
            if ((r->ics.ltp_present = abr_bit(b)))
                read_ltp(b, &r->ics);
        ms_present = (int)abr_bits(b, 2);
        if (ms_present == 3)
            return -1;
        int stride = l->ics.seq == EIGHT_SHORT ? 16 : 0;
        for (int g = 0; g < l->ics.groups; g++)
            for (int i = 0; i < l->ics.max_sfb; i++)
                ms[g * stride + i] = ms_present == 1 ? (uint8_t)abr_bit(b) : 1;
        if (!ms_present)
            memset(ms, 0, sizeof ms);
    }
    if (read_ics(d, b, l, common) != 0 || read_ics(d, b, r, common) != 0)
        return -1;
    if (!common && d->cfg.aot == AOT_MAIN) {
        apply_prediction(d, l);
        apply_prediction(d, r);
    }
    if (common) {
        if (ms_present)
            aac_apply_ms(d, l, r, ms);
        if (d->cfg.aot == AOT_MAIN) {
            apply_prediction(d, l);
            apply_prediction(d, r);
        }
    }
    apply_intensity(d, l, r, ms, ms_present);
    return 0;
}

static int read_cce(struct aac_dec *d, struct abr *b, struct element *e)
{
    static const float CCE_SCALE[4] = {
        1.09050773266525765921f, 1.18920711500272106672f, (float)M_SQRT2, 2.0f};
    if (!e->cce && !(e->cce = calloc(1, sizeof *e->cce)))
        return -1;
    struct cce *cc = e->cce;
    int num_gain = 0;
    cc->point = 2 * abr_bit(b);
    cc->num_coupled = (int)abr_bits(b, 3);
    for (int c = 0; c <= cc->num_coupled; c++) {
        num_gain++;
        cc->type[c] = abr_bit(b);
        cc->id[c] = (int)abr_bits(b, 4);
        if (cc->type[c] == 1) {
            cc->ch_select[c] = (int)abr_bits(b, 2);
            if (cc->ch_select[c] == 3)
                num_gain++;
        } else {
            cc->ch_select[c] = 2;
        }
    }
    cc->point += abr_bit(b) || (cc->point >> 1);
    int sign = abr_bit(b);
    float scale = CCE_SCALE[abr_bits(b, 2)];
    if (read_sce(d, b, e) != 0)
        return -1;
    const struct chan *ch = d->ch[e->first];
    const struct ics *ics = &ch->ics;
    int stride = ics->seq == EIGHT_SHORT ? 16 : 0;
    for (int c = 0; c < num_gain; c++) {
        int cge = 1, gain = 0;
        float cache = 1.0f;
        if (c) {
            cge = cc->point == AFTER_IMDCT ? 1 : abr_bit(b);
            if (cge) {
                int v = vlc_get(b, &d->sf_vlc);
                if (v < 0)
                    return -1;
                gain = v - 60;
            }
            cache = powf(scale, (float)-gain);
        }
        if (cc->point == AFTER_IMDCT) {
            cc->gain[c][0] = cache;
            continue;
        }
        for (int g = 0; g < ics->groups; g++)
            for (int sfb = 0; sfb < ics->max_sfb; sfb++) {
                int idx = g * stride + sfb;
                if (ch->cb[idx] == ZERO_BT)
                    continue;
                if (!cge) {
                    int v = vlc_get(b, &d->sf_vlc);
                    if (v < 0)
                        return -1;
                    int t = v - 60;
                    if (t) {
                        int s1 = 1;
                        t = gain += t;
                        if (sign) {
                            s1 -= 2 * (t & 1);
                            t >>= 1;
                        }
                        cache = powf(scale, (float)-t) * (float)s1;
                    }
                }
                cc->gain[c][idx] = cache;
            }
    }
    return abr_over(b) ? -1 : 0;
}

/* the coupling channels' spectra added to a target's, before the IMDCT */
static void couple_dependent(struct aac_dec *d, struct chan *t,
                             const struct chan *src, const float *gain)
{
    const struct ics *ics = &src->ics;
    int stride = ics->seq == EIGHT_SHORT ? 16 : 0;
    int wlen = ics->seq == EIGHT_SHORT ? d->short_len : d->frame;
    int win = 0;
    for (int g = 0; g < ics->groups; g++) {
        for (int i = 0; i < ics->max_sfb; i++) {
            int idx = g * stride + i;
            if (src->cb[idx] == ZERO_BT)
                continue;
            for (int w = 0; w < ics->group_len[g]; w++) {
                float *dst = t->coef + (win + w) * wlen;
                const float *s = src->coef + (win + w) * wlen;
                for (int k = ics->swb[i]; k < ics->swb[i + 1]; k++)
                    dst[k] += gain[idx] * s[k];
            }
        }
        win += ics->group_len[g];
    }
}

/* the coupling at point of the frame's CCEs into element e: its
   spectra (before the IMDCT), or its outputs out[] of n samples */
static void couple(struct aac_dec *d, struct element *const *seen, int n_seen,
                   const struct element *e, int point, float **out, int n)
{
    for (int k = 0; k < n_seen; k++) {
        const struct element *x = seen[k];
        if (x->type != 2 || !x->cce || x->cce->point != point)
            continue;
        const struct cce *cc = x->cce;
        int index = 0;
        for (int c = 0; c <= cc->num_coupled; c++) {
            if (cc->type[c] != e->type || cc->id[c] != e->tag) {
                index += 1 + (cc->ch_select[c] == 3);
                continue;
            }
            for (int j = 0; j < 2 && j < e->ch; j++) {
                int use =
                    j == 0 ? cc->ch_select[c] != 1 : cc->ch_select[c] != 2;
                if (!use)
                    continue;
                if (point == AFTER_IMDCT) {
                    float gain = cc->gain[index][0];
                    for (int i = 0; i < n; i++)
                        out[j][i] += gain * cc->out[i];
                } else {
                    couple_dependent(d, d->ch[e->first + j], d->ch[x->first],
                                     cc->gain[index]);
                }
                if (j == 0 && cc->ch_select[c] != 0)
                    index++;
                else if (j == 1)
                    index++;
            }
        }
    }
}

/* a fill element's payloads: SBR for the element before it */
static int read_fil(struct aac_dec *d, struct abr *b, struct element *prev)
{
    int cnt = (int)abr_bits(b, 4);
    if (cnt == 15)
        cnt += (int)abr_bits(b, 8) - 1;
    size_t end = b->pos + 8 * (size_t)cnt;
    if (cnt > 0) {
        int type = (int)abr_bits(b, 4);
        if ((type == 13 || type == 14) && prev && d->sbr_on != 0) {
            if (!prev->sbr) {
                prev->sbr =
                    sbr_new(d->cfg.rate, prev->type == 1 ? 2 : 1, d->sbr_down);
                if (!prev->sbr)
                    return -1;
            }
            if (d->sbr_on < 0)
                d->sbr_on = 1;
            sbr_read(prev->sbr, b, type == 14, cnt);
        }
    }
    b->pos = end;
    return abr_over(b) ? -1 : 0;
}

static int read_dse(struct abr *b)
{
    abr_bits(b, 4);
    int align = abr_bit(b);
    int cnt = (int)abr_bits(b, 8);
    if (cnt == 255)
        cnt += (int)abr_bits(b, 8);
    if (align)
        b->pos = (b->pos + 7) & ~(size_t)7;
    b->pos += 8 * (size_t)cnt;
    return abr_over(b) ? -1 : 0;
}

int aac_frame(struct aac_dec *d, const uint8_t *p, size_t n, char *why,
              size_t wn)
{
    struct abr b = {.p = p, .n = n};
    struct element *prev = NULL;
    struct element *seen[MAX_ELEMENTS];
    int n_seen = 0;
    if (d->cfg.aot == AOT_USAC)
        return usac_frame(d, &b, why, wn);
    if (is_er(d->cfg.aot)) {
        /* the elements the configuration says, in its order, without ids
           (but for a tag before each, ELD's excepted) */
        int tags[4] = {0};
        for (const char *t = ER_LAYOUT[d->cfg.chan_config]; *t; t++) {
            int type = *t == 'S' ? 0 : *t == 'C' ? 1 : 3;
            if (d->cfg.aot != AOT_ER_ELD)
                abr_bits(&b, 4);
            struct element *e = aac_element(d, type, tags[type]++);
            if (!e) {
                snprintf(why, wn, "an AAC stream of too many channels");
                return -1;
            }
            int rc = type == 1 ? read_cpe(d, &b, e) : read_sce(d, &b, e);
            if (rc != 0 || abr_over(&b)) {
                snprintf(why, wn, "an AAC frame that is none");
                return -1;
            }
            seen[n_seen++] = e;
        }
    }
    while (!is_er(d->cfg.aot)) {
        int id = (int)abr_bits(&b, 3);
        if (abr_over(&b)) {
            snprintf(why, wn, "an AAC frame cut short");
            return -1;
        }
        if (id == 7)
            break;
        int rc = 0;
        switch (id) {
        case 0:
        case 1:
        case 2:
        case 3: {
            int tag = (int)abr_bits(&b, 4);
            struct element *e = aac_element(d, id, tag);
            if (!e) {
                snprintf(why, wn, "an AAC stream of too many channels");
                return -1;
            }
            rc = id == 1   ? read_cpe(d, &b, e)
                 : id == 2 ? read_cce(d, &b, e)
                           : read_sce(d, &b, e);
            prev = e;
            if (n_seen < MAX_ELEMENTS)
                seen[n_seen++] = e;
            break;
        }
        case 4:
            rc = read_dse(&b);
            break;
        case 5:
            rc = read_pce(&b) < 0 ? -1 : 0;
            break;
        case 6:
            rc = read_fil(d, &b, prev);
            break;
        }
        if (rc != 0) {
            snprintf(why, wn, "an AAC frame that is none");
            return -1;
        }
    }
    if (d->sbr_on < 0)
        d->sbr_on = 0;
    int nsamp = d->sbr_on == 1 && !d->sbr_down ? 2 * d->frame : d->frame;
    /* the coupling channels first: their spectra with their TNS, and
       their samples when they are added after the IMDCT */
    for (int k = 0; k < n_seen; k++) {
        struct element *e = seen[k];
        if (e->type != 2)
            continue;
        struct chan *c = d->ch[e->first];
        if (c->tns.present)
            aac_apply_tns(d, c->coef, &c->tns, &c->ics, 0);
        if (e->cce->point != AFTER_IMDCT)
            continue;
        aac_synthesize(d, c);
        float *core[1] = {c->out}, *o[2] = {e->cce->out, NULL};
        if (d->sbr_on == 1) {
            if (!e->sbr && !(e->sbr = sbr_new(d->cfg.rate, 1, d->sbr_down)))
                return -1;
            float tmp[2048];
            o[1] = tmp;
            sbr_apply(e->sbr, 1, core, o);
        } else {
            memcpy(e->cce->out, c->out, sizeof(float) * (size_t)d->frame);
        }
    }
    /* the spectra to samples, element by element */
    int out = 0;
    for (int k = 0; k < n_seen; k++) {
        struct element *e = seen[k];
        if (e->type == 2)
            continue;
        float *core[2];
        int coupled = e->type <= 1;
        if (coupled)
            couple(d, seen, n_seen, e, BEFORE_TNS, NULL, 0);
        for (int j = 0; j < e->ch; j++) {
            struct chan *c = d->ch[e->first + j];
            if (d->cfg.aot == AOT_LTP)
                apply_ltp(d, c);
            if (c->tns.present)
                aac_apply_tns(d, c->coef, &c->tns, &c->ics, 0);
        }
        if (coupled)
            couple(d, seen, n_seen, e, BETWEEN_TNS_AND_IMDCT, NULL, 0);
        for (int j = 0; j < e->ch; j++) {
            struct chan *c = d->ch[e->first + j];
            aac_synthesize(d, c);
            core[j] = c->out;
        }
        int first_out = out;
        if (d->sbr_on == 1) {
            int ps = e->ch == 1 && e->sbr && sbr_has_ps(e->sbr);
            int och = ps ? 2 : e->ch;
            if (out + och > MAX_CHANNELS)
                break;
            /* outputs, not channels: PS makes two of a mono element */
            int ok = 1;
            for (int j = 0; j < och; j++)
                if (!d->outp[out + j] &&
                    !(d->outp[out + j] = calloc(2 * 2048, sizeof(float))))
                    ok = 0;
            if (!ok)
                break;
            float *o[2] = {d->outp[out], och > 1 ? d->outp[out + 1] : NULL};
            if (e->sbr) {
                sbr_apply(e->sbr, e->ch, core, o);
            } else {
                /* no SBR data for this element yet: the core upsampled
                   by the same filter bank, as an empty SBR frame */
                if (!(e->sbr = sbr_new(d->cfg.rate, e->ch, d->sbr_down)))
                    break;
                sbr_apply(e->sbr, e->ch, core, o);
            }
            out += och;
        } else {
            for (int j = 0; j < e->ch && out < MAX_CHANNELS; j++)
                memcpy(d->outp[out++], core[j],
                       sizeof(float) * (size_t)d->frame);
        }
        if (coupled && out - first_out >= e->ch)
            couple(d, seen, n_seen, e, AFTER_IMDCT, d->outp + first_out, nsamp);
    }
    d->out_ch = out;
    return nsamp;
}

/* ---- a stream ---- */

int aac_stream(const struct aac_cfg *c, const uint8_t *data, const size_t *off,
               const size_t *len, size_t count, const struct audio_edit *edit,
               struct audio_out *ao, char *why, size_t wn)
{
    struct aac_dec *d = aac_new(c);
    if (!d) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    float *pcm = NULL;
    size_t got = 0, cap = 0;
    int rc = -1, rate = 0, bad = 0;
    for (size_t i = 0; i < count; i++) {
        char w[160];
        int ns = aac_frame(d, data + off[i], len[i], w, sizeof w);
        if (ns < 0) {
            /* a frame that cannot be read is silence, as a decoder
               conceals it; a stream of nothing else is refused */
            if (++bad > 8 && got == 0) {
                snprintf(why, wn, "%s", w);
                goto end;
            }
            if (!rate)
                continue;
            ns = rate == 2 * c->rate ? 2 * d->frame : d->frame;
            d->out_ch = 0;
        }
        if (!rate)
            rate = aac_out_rate(d);
        if ((size_t)ns > cap) { /* a frame's samples */
            size_t nc = (size_t)ns;
            float *t = realloc(pcm, sizeof(float) * nc);
            if (!t) {
                snprintf(why, wn, "out of memory");
                goto end;
            }
            pcm = t;
            cap = nc;
        }
        float **o;
        int nch = aac_out(d, &o);
        for (int k = 0; k < ns; k++) {
            float s = 0;
            for (int j = 0; j < nch; j++)
                s += o[j][k];
            pcm[k] = nch ? s / (float)nch : 0;
        }
        ao->rate = (uint32_t)rate;
        if (audio_put(ao, pcm, (size_t)ns, why, wn) != 0)
            goto end;
        got += (size_t)ns;
    }
    size_t skip = 0, total = got;
    audio_edit_apply(edit, (uint32_t)rate, &skip, &total);
    if (!total) {
        snprintf(why, wn, "an AAC stream without sound");
        goto end;
    }
    ao->skip = skip;
    ao->total = skip + total;
    rc = 0;
end:
    free(pcm);
    aac_free(d);
    return rc;
}

/* ---- ADTS ---- */

int janas_audio_adts(const uint8_t *d, size_t n, struct audio_out *ao,
                     char *why, size_t wn)
{
    /* two headers in a row, or one that fills the file */
    if (n < 7 || d[0] != 0xFF || (d[1] & 0xF6) != 0xF0)
        return 1;
    size_t flen = (size_t)(d[3] & 3) << 11 | (size_t)d[4] << 3 | d[5] >> 5;
    if (flen < 7 || (flen < n && (flen + 2 > n || d[flen] != 0xFF ||
                                  (d[flen + 1] & 0xF6) != 0xF0)))
        return 1;
    struct aac_cfg c;
    memset(&c, 0, sizeof c);
    c.aot = (d[2] >> 6) + 1;
    c.rate_index = d[2] >> 2 & 15;
    c.rate = aac_index_rate(c.rate_index);
    c.chan_config = (d[2] & 1) << 2 | d[3] >> 6;
    c.channels = CONFIG_CHANNELS[c.chan_config];
    c.frame = 1024;
    c.sbr = c.ps = -1;
    if (!c.rate || (c.aot != AOT_MAIN && c.aot != AOT_LC && c.aot != AOT_LTP)) {
        snprintf(why, wn, "an ADTS stream of a kind not read (profile %d)",
                 c.aot - 1);
        return -1;
    }
    size_t cap = 1024, count = 0;
    size_t *off = malloc(sizeof(size_t) * cap),
           *len = malloc(sizeof(size_t) * cap);
    int rc = -1;
    if (!off || !len) {
        snprintf(why, wn, "out of memory");
        goto end;
    }
    for (size_t at = 0; at + 7 <= n;) {
        const uint8_t *h = d + at;
        if (h[0] != 0xFF || (h[1] & 0xF6) != 0xF0)
            break; /* a tag, or the end of the sound */
        size_t fl = (size_t)(h[3] & 3) << 11 | (size_t)h[4] << 3 | h[5] >> 5;
        int crc = !(h[1] & 1), blocks = (h[6] & 3) + 1;
        size_t hl = 7 + (crc ? 2 * (size_t)blocks : 0);
        if (fl < hl || (h[2] >> 2 & 15) != c.rate_index)
            break;
        if (fl > n - at)
            fl = n - at; /* the last frame cut short */
        /* the blocks: after one header, each with its CRC when there are
           several and they are protected */
        size_t start = at + 7 + (crc && blocks == 1 ? 2 : 0);
        size_t posn[4] = {0};
        if (crc && blocks > 1)
            for (int i = 1; i < blocks; i++)
                posn[i] = (size_t)h[7 + 2 * (i - 1)] << 8 | h[8 + 2 * (i - 1)];
        if (crc && blocks > 1)
            start = at + 7 + 2 * (size_t)blocks;
        for (int i = 0; i < blocks; i++) {
            size_t bs = start, be = at + fl;
            if (blocks > 1) {
                if (crc) {
                    bs = i ? at + posn[i] : start;
                    be = i + 1 < blocks ? at + posn[i + 1] : at + fl;
                    if (be < bs + 2 || be > at + fl)
                        break;
                    be -= 2;
                } else {
                    /* unprotected blocks are read one after another:
                       the decoder tells where each ends */
                    be = at + fl;
                }
            }
            if (count == cap) {
                cap *= 2;
                size_t *o2 = realloc(off, sizeof(size_t) * cap);
                if (o2)
                    off = o2;
                size_t *l2 = realloc(len, sizeof(size_t) * cap);
                if (l2)
                    len = l2;
                if (!o2 || !l2) {
                    snprintf(why, wn, "out of memory");
                    goto end;
                }
            }
            off[count] = bs;
            len[count++] = be > bs ? be - bs : 0;
            if (blocks > 1 && !crc)
                break; /* not separable: the first block only */
        }
        at += fl;
    }
    rc = aac_stream(&c, d, off, len, count, NULL, ao, why, wn);
end:
    free(off);
    free(len);
    return rc;
}
