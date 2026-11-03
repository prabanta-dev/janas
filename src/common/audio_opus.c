/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_opus.c - Opus (RFC 6716; in Ogg, RFC 7845; in Matroska): the
 * identification header (channels, pre-skip, output gain, the channel
 * mapping and its streams), the packets (the table-of-contents byte, the
 * four ways frames are packed, self-delimited framing between the streams
 * of a multistream packet), and each frame as the reference decoder takes
 * it: SILK, hybrid or CELT; a CELT frame's bands from 17 up in hybrid, its
 * last band by the bandwidth; the redundant 5 ms CELT frames that smooth a
 * change of mode, the concealment that stands in for them when there are
 * none, the fades. Everything at 48 kHz; the channels averaged; the
 * pre-skip and the samples past the last granule position taken off.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_ogg.h"
#include "audio_opus.h"
#include "audio_opus_tab.h"

#define MODE_SILK 1000
#define MODE_HYBRID 1001
#define MODE_CELT 1002
#define BW_NB 1101
#define BW_MB 1102
#define BW_WB 1103
#define BW_SWB 1104
#define BW_FB 1105
#define F20 960
#define F10 480
#define F5 240
#define F2_5 120

struct odec {
    int channels; /* the output's */
    struct celt_dec *celt;
    struct silk_dec *silk;
    int mode, prev_mode, prev_redundancy, bandwidth, frame_size;
    int stream_channels;
    uint32_t range_final;
    float *pcm[2], *silk_pcm[2], *redund[2], *trans[2], *tmp[2];
};

static struct odec *odec_new(int channels)
{
    struct odec *o = calloc(1, sizeof *o);
    if (!o)
        return NULL;
    o->channels = channels;
    o->celt = jcelt_new(channels);
    o->silk = jsilk_new();
    for (int c = 0; c < 2; c++) {
        o->silk_pcm[c] = malloc(sizeof(float) * 2880);
        o->redund[c] = malloc(sizeof(float) * F5);
        o->trans[c] = malloc(sizeof(float) * F5);
        o->tmp[c] = malloc(sizeof(float) * 2880);
        if (!o->silk_pcm[c] || !o->redund[c] || !o->trans[c] || !o->tmp[c])
            o->celt = NULL;
    }
    if (!o->celt || !o->silk) {
        free(o);
        return NULL;
    }
    return o;
}

static void odec_free(struct odec *o)
{
    if (!o)
        return;
    jcelt_free(o->celt);
    jsilk_free(o->silk);
    for (int c = 0; c < 2; c++) {
        free(o->silk_pcm[c]);
        free(o->redund[c]);
        free(o->trans[c]);
        free(o->tmp[c]);
    }
    free(o);
}

static void smooth_fade(float **in1, int o1, float **in2, int o2, float **out,
                        int oo, int n, int ch)
{
    for (int c = 0; c < ch; c++)
        for (int i = 0; i < n; i++) {
            float w = CELT_WINDOW[i] * CELT_WINDOW[i];
            out[c][oo + i] = w * in2[c][o2 + i] + (1.f - w) * in1[c][o1 + i];
        }
}

/* one frame of a packet (len bytes, NULL for concealment) into pcm[c][0
   ..]: the samples a channel, or -1 */
static int decode_frame(struct odec *st, const uint8_t *data, int len,
                        float **pcm, int frame_size)
{
    int ch = st->channels, audiosize, mode, bandwidth;
    struct rc dec;
    if (len <= 1) {
        /* a lost or empty frame: no more concealed than the TOC says */
        data = NULL;
        if (frame_size > st->frame_size)
            frame_size = st->frame_size;
    }
    if (frame_size < F2_5 && !data) {
        for (int c = 0; c < ch; c++)
            memset(pcm[c], 0,
                   sizeof(float) * (size_t)(frame_size > 0 ? frame_size : 0));
        return frame_size > 0 ? frame_size : 0;
    }
    if (data) {
        audiosize = st->frame_size;
        mode = st->mode;
        bandwidth = st->bandwidth;
        rc_init(&dec, data, (uint32_t)len);
    } else {
        audiosize = frame_size;
        mode = st->prev_redundancy ? MODE_CELT : st->prev_mode;
        bandwidth = 0;
        if (mode == 0) {
            for (int c = 0; c < ch; c++)
                memset(pcm[c], 0, sizeof(float) * (size_t)audiosize);
            return audiosize;
        }
        if (audiosize > F20) {
            int done = 0;
            while (done < audiosize) {
                float *p[2] = {pcm[0] + done, ch > 1 ? pcm[1] + done : NULL};
                int k = audiosize - done < F20 ? audiosize - done : F20;
                int ret = decode_frame(st, NULL, 0, p, k);
                if (ret < 0)
                    return ret;
                done += ret;
            }
            return frame_size;
        } else if (audiosize < F20) {
            if (audiosize > F10)
                audiosize = F10;
            else if (mode != MODE_SILK && audiosize > F5 && audiosize < F10)
                audiosize = F5;
        }
    }
    int transition = 0;
    float **ptrans = NULL;
    if (data && st->prev_mode > 0 &&
        ((mode == MODE_CELT && st->prev_mode != MODE_CELT &&
          !st->prev_redundancy) ||
         (mode != MODE_CELT && st->prev_mode == MODE_CELT))) {
        transition = 1;
    }
    if (transition && mode == MODE_CELT) {
        ptrans = st->trans;
        decode_frame(st, NULL, 0, ptrans, audiosize < F5 ? audiosize : F5);
    }
    if (audiosize > frame_size)
        return -1;
    frame_size = audiosize;
    if (mode != MODE_CELT) {
        if (st->prev_mode == MODE_CELT)
            jsilk_reset(st->silk);
        int khz = mode == MODE_SILK ? (bandwidth == BW_NB   ? 8
                                       : bandwidth == BW_MB ? 12
                                                            : 16)
                                    : 16;
        int ms = 1000 * audiosize / 48000;
        if (ms < 10)
            ms = 10;
        int got = 0;
        while (got < frame_size) {
            float *p[2] = {st->silk_pcm[0] + got, st->silk_pcm[1] + got};
            int n = jsilk_frame(st->silk, data ? &dec : NULL, ms, khz,
                                st->stream_channels, ch, p, got == 0);
            if (n <= 0) {
                if (!data) {
                    for (int c = 0; c < ch; c++)
                        memset(st->silk_pcm[c] + got, 0,
                               sizeof(float) * (size_t)(frame_size - got));
                    break;
                }
                return -1;
            }
            got += n;
        }
    }
    int start_band = 0, redundancy = 0, celt_to_silk = 0, redund_bytes = 0;
    if (mode != MODE_CELT && data &&
        rc_tell(&dec) + 17 + 20 * (mode == MODE_HYBRID) <= 8 * len) {
        redundancy = mode == MODE_HYBRID ? rc_bit_logp(&dec, 12) : 1;
        if (redundancy) {
            celt_to_silk = rc_bit_logp(&dec, 1);
            redund_bytes = mode == MODE_HYBRID
                               ? (int)rc_uint(&dec, 256) + 2
                               : len - ((rc_tell(&dec) + 7) >> 3);
            len -= redund_bytes;
            if (len * 8 < rc_tell(&dec)) {
                len = 0;
                redund_bytes = 0;
                redundancy = 0;
            }
            dec.storage -= (uint32_t)redund_bytes;
        }
    }
    if (mode != MODE_CELT)
        start_band = 17;
    if (redundancy)
        transition = 0;
    if (transition && mode != MODE_CELT) {
        ptrans = st->trans;
        decode_frame(st, NULL, 0, ptrans, audiosize < F5 ? audiosize : F5);
    }
    int end_band = 21;
    if (bandwidth) {
        end_band = bandwidth == BW_NB                         ? 13
                   : bandwidth == BW_MB || bandwidth == BW_WB ? 17
                   : bandwidth == BW_SWB                      ? 19
                                                              : 21;
    }
    uint32_t redundant_rng = 0;
    struct rc rd;
    if (redundancy && celt_to_silk) {
        jcelt_bands(st->celt, 0, end_band);
        rc_init(&rd, data + len, (uint32_t)redund_bytes);
        jcelt_frame(st->celt, &rd, redund_bytes, F5, st->stream_channels,
                    st->redund);
        redundant_rng = jcelt_rng(st->celt);
    }
    jcelt_bands(st->celt, start_band, end_band);
    if (mode != MODE_SILK) {
        int n = frame_size < F20 ? frame_size : F20;
        if (mode != st->prev_mode && st->prev_mode > 0 && !st->prev_redundancy)
            jcelt_reset(st->celt);
        if (data) {
            if (jcelt_frame(st->celt, &dec, len, n, st->stream_channels, pcm))
                return -1;
        } else {
            jcelt_plc(st->celt, n, pcm);
        }
    } else {
        for (int c = 0; c < ch; c++)
            memset(pcm[c], 0, sizeof(float) * (size_t)frame_size);
        if (st->prev_mode == MODE_HYBRID &&
            !(redundancy && celt_to_silk && st->prev_redundancy)) {
            static const uint8_t SILENCE[2] = {0xFF, 0xFF};
            struct rc sr;
            jcelt_bands(st->celt, 0, end_band);
            rc_init(&sr, SILENCE, 2);
            jcelt_frame(st->celt, &sr, 2, F2_5, st->stream_channels, pcm);
        }
    }
    if (mode != MODE_CELT)
        for (int c = 0; c < ch; c++)
            for (int i = 0; i < frame_size; i++)
                pcm[c][i] += st->silk_pcm[c][i];
    if (redundancy && !celt_to_silk) {
        jcelt_reset(st->celt);
        jcelt_bands(st->celt, 0, end_band);
        rc_init(&rd, data + len, (uint32_t)redund_bytes);
        jcelt_frame(st->celt, &rd, redund_bytes, F5, st->stream_channels,
                    st->redund);
        redundant_rng = jcelt_rng(st->celt);
        smooth_fade(pcm, frame_size - F2_5, st->redund, F2_5, pcm,
                    frame_size - F2_5, F2_5, ch);
    }
    if (redundancy && celt_to_silk &&
        (st->prev_mode != MODE_SILK || st->prev_redundancy)) {
        for (int c = 0; c < ch; c++)
            memcpy(pcm[c], st->redund[c], sizeof(float) * F2_5);
        smooth_fade(st->redund, F2_5, pcm, F2_5, pcm, F2_5, F2_5, ch);
    }
    if (transition) {
        if (audiosize >= F5) {
            for (int c = 0; c < ch; c++)
                memcpy(pcm[c], ptrans[c], sizeof(float) * F2_5);
            smooth_fade(ptrans, F2_5, pcm, F2_5, pcm, F2_5, F2_5, ch);
        } else {
            smooth_fade(ptrans, 0, pcm, 0, pcm, 0, F2_5, ch);
        }
    }
    st->range_final = len <= 1 ? 0 : dec.rng ^ redundant_rng;
    st->prev_mode = mode;
    st->prev_redundancy = redundancy && !celt_to_silk;
    return audiosize;
}

/* ---- packets ---- */

static int frame_samples(int toc)
{
    int cfg = toc >> 3;
    if (cfg < 12)
        return (cfg & 3) == 3 ? 2880 : 480 << (cfg & 3);
    if (cfg < 16)
        return cfg & 1 ? 960 : 480;
    return 120 << (cfg & 3);
}

int janas_opus_packet_samples(const uint8_t *p, size_t n)
{
    if (!n)
        return 0;
    int code = p[0] & 3, count = code == 0 ? 1 : code < 3 ? 2 : -1;
    if (count < 0)
        count = n >= 2 ? p[1] & 0x3F : 0;
    return frame_samples(p[0]) * count;
}

static int toc_mode(int toc)
{
    return toc & 0x80             ? MODE_CELT
           : (toc & 0x60) == 0x60 ? MODE_HYBRID
                                  : MODE_SILK;
}

static int toc_bandwidth(int toc)
{
    if (toc & 0x80) {
        int b = (toc >> 5) & 3;
        return b == 0 ? BW_NB : BW_WB + b - 1;
    }
    if ((toc & 0x60) == 0x60)
        return toc & 0x10 ? BW_FB : BW_SWB;
    return BW_NB + ((toc >> 5) & 3);
}

/* a size field: 1 or 2 bytes; -1 when cut short */
static int size_field(const uint8_t *p, int n, int *size)
{
    if (n < 1)
        return -1;
    if (p[0] < 252) {
        *size = p[0];
        return 1;
    }
    if (n < 2)
        return -1;
    *size = 4 * p[1] + p[0];
    return 2;
}

/* the frames of a packet: their count, sizes in size[], the offset of the
   first in *off, the packet's bytes used in *used (self-delimited), or -1 */
static int parse_packet(const uint8_t *data, int len, int self_delim, int *size,
                        int *off, int *used)
{
    if (len < 1)
        return -1;
    const uint8_t *p = data + 1;
    int left = len - 1, count, fs = frame_samples(data[0]), last = 0;
    int pad = 0, cbr = 0;
    switch (data[0] & 3) {
    case 0:
        count = 1;
        last = left;
        break;
    case 1:
        count = 2;
        cbr = 1;
        if (!self_delim) {
            if (left & 1)
                return -1;
            last = size[0] = left / 2;
        }
        break;
    case 2: {
        count = 2;
        int k = size_field(p, left, &size[0]);
        if (k < 0 || size[0] > left - k)
            return -1;
        p += k;
        left -= k;
        last = left - size[0];
        break;
    }
    default: {
        if (left < 1)
            return -1;
        int ch = *p++;
        left--;
        count = ch & 0x3F;
        if (count <= 0 || fs * count > 5760)
            return -1;
        if (ch & 0x40) {
            int t;
            do {
                if (left <= 0)
                    return -1;
                t = *p++;
                left--;
                int tmp = t == 255 ? 254 : t;
                left -= tmp;
                pad += tmp;
            } while (t == 255);
        }
        if (left < 0)
            return -1;
        cbr = !(ch & 0x80);
        if (!cbr) {
            last = left;
            for (int i = 0; i < count - 1; i++) {
                int k = size_field(p, left, &size[i]);
                if (k < 0 || size[i] > left - k)
                    return -1;
                p += k;
                left -= k;
                last -= k + size[i];
            }
            if (last < 0)
                return -1;
        } else if (!self_delim) {
            last = left / count;
            if (last * count != left)
                return -1;
            for (int i = 0; i < count - 1; i++)
                size[i] = last;
        }
    }
    }
    if (self_delim) {
        int k = size_field(p, left, &size[count - 1]);
        if (k < 0 || size[count - 1] > left - k)
            return -1;
        p += k;
        left -= k;
        if (cbr) {
            if (size[count - 1] * count > left)
                return -1;
            for (int i = 0; i < count - 1; i++)
                size[i] = size[count - 1];
        } else if (k + size[count - 1] > last) {
            return -1;
        }
    } else {
        if (last > 1275)
            return -1;
        size[count - 1] = last;
    }
    *off = (int)(p - data);
    int total = *off;
    for (int i = 0; i < count; i++)
        total += size[i];
    *used = total + pad;
    return count;
}

/* the frames of one stream's packet into pcm[c] from at: samples, or -1 */
static int decode_packet(struct odec *st, const uint8_t *data, int len,
                         int self_delim, float **pcm, int *used)
{
    int size[48], off,
        count = parse_packet(data, len, self_delim, size, &off, used);
    if (count < 0 || count * frame_samples(data[0]) > 5760)
        return -1;
    st->mode = toc_mode(data[0]);
    st->bandwidth = toc_bandwidth(data[0]);
    st->frame_size = frame_samples(data[0]);
    st->stream_channels = data[0] & 4 ? 2 : 1;
    int got = 0;
    const uint8_t *p = data + off;
    for (int i = 0; i < count; i++) {
        float *o[2] = {pcm[0] + got, st->channels > 1 ? pcm[1] + got : NULL};
        int k = decode_frame(st, p, size[i], o, 5760 - got);
        if (k < 0)
            return -1;
        p += size[i];
        got += k;
    }
    return got;
}

/* ---- the stream ---- */

int janas_opus_stream(const struct ogg_packets *o, struct audio_out *ao,
                      char *why, size_t wn)
{
    const uint8_t *h = o->data + o->off[0];
    size_t hn = o->len[0];
    if (hn < 19 || memcmp(h, "OpusHead", 8) || (h[8] & 0xF0)) {
        snprintf(why, wn, "an Opus stream without its header");
        return -1;
    }
    int channels = h[9], pre_skip = h[10] | h[11] << 8;
    int16_t gain_q8 = (int16_t)(h[16] | h[17] << 8);
    int family = h[18], streams = 1, coupled = channels > 1;
    uint8_t map[255] = {0, 1};
    if (o->pre_skip >= 0)
        pre_skip = (int)o->pre_skip;
    if (!channels) {
        snprintf(why, wn, "an Opus stream of no channels");
        return -1;
    }
    if (family == 0) {
        if (channels > 2) {
            snprintf(why, wn,
                     "an Opus stream of more channels than its "
                     "mapping allows");
            return -1;
        }
    } else {
        if (hn < 21 + (size_t)channels) {
            snprintf(why, wn,
                     "an Opus stream whose channel mapping is cut "
                     "short");
            return -1;
        }
        streams = h[19];
        coupled = h[20];
        memcpy(map, h + 21, (size_t)channels);
        if (!streams || coupled > streams || streams + coupled > 255) {
            snprintf(why, wn,
                     "an Opus stream of a channel mapping out of "
                     "range");
            return -1;
        }
    }
    struct odec **dec = calloc((size_t)streams, sizeof *dec);
    float *buf = malloc(sizeof(float) * 2 * 5760 * (size_t)streams);
    float *out = malloc(sizeof(float) * 5760);
    int rc = -1;
    size_t got = 0;
    if (!dec || !buf || !out)
        goto oom;
    ao->rate = 48000;
    for (int s = 0; s < streams; s++)
        if (!(dec[s] = odec_new(s < coupled ? 2 : 1)))
            goto oom;
    /* the comment header is the second packet in Ogg; in Matroska none */
    size_t first = o->count > 1 && o->len[1] >= 8 &&
                           !memcmp(o->data + o->off[1], "OpusTags", 8)
                       ? 2
                       : 1;
    float gain = gain_q8 ? (float)pow(10.0, gain_q8 / (20.0 * 256)) : 1.f;
    int64_t last_granule = -1;
    for (size_t pk = first; pk < o->count; pk++) {
        const uint8_t *p = o->data + o->off[pk];
        int left = (int)o->len[pk], n = -1;
        if (o->granule[pk] >= 0)
            last_granule = o->granule[pk];
        if (!left)
            continue;
        for (int s = 0; s < streams; s++) {
            float *pc[2] = {buf + (size_t)s * 2 * 5760,
                            buf + (size_t)s * 2 * 5760 + 5760};
            int used = 0;
            int k = decode_packet(dec[s], p, left, s < streams - 1, pc, &used);
            if (k < 0 || (n >= 0 && k != n)) {
                n = -1;
                break;
            }
            n = k;
            p += used;
            left -= used;
        }
        if (n <= 0)
            continue; /* a packet that does not read: skipped */
        for (int i = 0; i < n; i++) {
            double sum = 0;
            int used_ch = 0;
            for (int c = 0; c < channels; c++) {
                int m = family == 0 ? c : map[c];
                if (m == 255) { /* a silent channel */
                    used_ch++;
                    continue;
                }
                int s = m < 2 * coupled ? m / 2 : m - coupled;
                int sc = m < 2 * coupled ? m % 2 : 0;
                sum += buf[(size_t)s * 2 * 5760 + (size_t)sc * 5760 + i];
                used_ch++;
            }
            out[i] = (float)(sum / used_ch) * gain;
        }
        if (audio_put(ao, out, (size_t)n, why, wn) != 0)
            goto end;
        got += (size_t)n;
    }
    size_t skip = (size_t)pre_skip, total = got;
    if (last_granule >= 0 && (size_t)last_granule < total)
        total = (size_t)last_granule;
    if (skip >= total) {
        snprintf(why, wn, "an Opus stream without sound");
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
    free(buf);
    for (int s = 0; dec && s < streams; s++)
        odec_free(dec[s]);
    free(dec);
    return rc;
}
