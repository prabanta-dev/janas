/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_mkv.c - Matroska and WebM: the EBML elements, sizes unknown too
 * (what browsers' MediaRecorder writes while it records), the first track
 * of sound with its codec and private data, the blocks of its clusters
 * (simple or in groups, without lacing or with Xiph, EBML or fixed-size
 * lacing), Opus's codec delay and the discard padding of its last block.
 * The codecs: Opus, Vorbis, FLAC, AAC, MPEG audio, ALAC and PCM.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio.h"
#include "audio_aac.h"
#include "audio_ogg.h"

#define ID_EBML 0x1A45DFA3
#define ID_SEGMENT 0x18538067
#define ID_INFO 0x1549A966
#define ID_TIMESCALE 0x2AD7B1
#define ID_TRACKS 0x1654AE6B
#define ID_TRACKENTRY 0xAE
#define ID_TRACKNUMBER 0xD7
#define ID_TRACKTYPE 0x83
#define ID_CODECID 0x86
#define ID_CODECPRIVATE 0x63A2
#define ID_CODECDELAY 0x56AA
#define ID_AUDIO 0xE1
#define ID_RATE 0xB5
#define ID_CHANNELS 0x9F
#define ID_BITDEPTH 0x6264
#define ID_CLUSTER 0x1F43B675
#define ID_SIMPLEBLOCK 0xA3
#define ID_BLOCKGROUP 0xA0
#define ID_BLOCK 0xA1
#define ID_DISCARDPADDING 0x75A2
#define ID_CONTENTENCODINGS 0x6D80

#define UNKNOWN ((uint64_t)-1)

struct ebml {
    const uint8_t *p, *end;
};

/* a variable-length integer: its value (the marker removed when strip),
   its length in *len; -1 when cut short */
static int vint(const uint8_t *p, const uint8_t *end, int strip, uint64_t *v,
                int *len)
{
    if (p >= end || !*p)
        return -1;
    int l = 1;
    while (!(*p & (0x80 >> (l - 1))))
        l++;
    if (l > 8 || end - p < l)
        return -1;
    uint64_t x = strip ? *p & (0xFF >> l) : *p;
    int all_ones = strip && x == (uint64_t)(0xFF >> l);
    for (int i = 1; i < l; i++) {
        x = x << 8 | p[i];
        if (p[i] != 0xFF)
            all_ones = 0;
    }
    *v = all_ones ? UNKNOWN : x;
    *len = l;
    return 0;
}

/* the next element's id and payload: 0, or -1 */
static int element(struct ebml *e, uint32_t *id, const uint8_t **body,
                   uint64_t *size)
{
    uint64_t v;
    int l1, l2;
    if (vint(e->p, e->end, 0, &v, &l1) != 0 || l1 > 4)
        return -1;
    *id = (uint32_t)v;
    if (vint(e->p + l1, e->end, 1, size, &l2) != 0)
        return -1;
    *body = e->p + l1 + l2;
    if (*size != UNKNOWN && *size > (uint64_t)(e->end - *body))
        *size = (uint64_t)(e->end - *body); /* a file cut short */
    return 0;
}

static uint64_t uint_of(const uint8_t *p, uint64_t n)
{
    uint64_t v = 0;
    for (uint64_t i = 0; i < n && i < 8; i++)
        v = v << 8 | p[i];
    return v;
}

static double float_of(const uint8_t *p, uint64_t n)
{
    if (n == 4) {
        uint32_t u = (uint32_t)uint_of(p, 4);
        float f;
        memcpy(&f, &u, 4);
        return f;
    }
    if (n == 8) {
        uint64_t u = uint_of(p, 8);
        double d;
        memcpy(&d, &u, 8);
        return d;
    }
    return 0;
}

struct mtrack {
    uint64_t number;
    char codec[64];
    const uint8_t *priv;
    size_t priv_len;
    uint64_t codec_delay; /* ns */
    double rate;
    int channels, bits;
    int encoded; /* content encodings: compression or encryption */
};

struct mframes {
    size_t *off, *len;
    size_t count, cap;
    int64_t discard; /* the last block's discard padding, ns */
};

static int add_frame(struct mframes *f, size_t off, size_t len)
{
    if (f->count == f->cap) {
        size_t c = f->cap ? f->cap * 2 : 1024;
        size_t *o = realloc(f->off, sizeof(size_t) * c);
        if (o)
            f->off = o;
        size_t *l = realloc(f->len, sizeof(size_t) * c);
        if (l)
            f->len = l;
        if (!o || !l)
            return -1;
        f->cap = c;
    }
    f->off[f->count] = off;
    f->len[f->count] = len;
    f->count++;
    return 0;
}

static void read_track(const uint8_t *p, const uint8_t *end, struct mtrack *t)
{
    struct ebml e = {p, end};
    memset(t, 0, sizeof *t);
    int type = 0;
    while (e.p < e.end) {
        uint32_t id;
        const uint8_t *b;
        uint64_t n;
        if (element(&e, &id, &b, &n) != 0 || n == UNKNOWN)
            break;
        switch (id) {
        case ID_TRACKNUMBER:
            t->number = uint_of(b, n);
            break;
        case ID_TRACKTYPE:
            type = (int)uint_of(b, n);
            break;
        case ID_CODECID: {
            size_t l =
                n < sizeof t->codec - 1 ? (size_t)n : sizeof t->codec - 1;
            memcpy(t->codec, b, l);
            t->codec[l] = 0;
            break;
        }
        case ID_CODECPRIVATE:
            t->priv = b;
            t->priv_len = (size_t)n;
            break;
        case ID_CODECDELAY:
            t->codec_delay = uint_of(b, n);
            break;
        case ID_CONTENTENCODINGS:
            t->encoded = 1;
            break;
        case ID_AUDIO: {
            struct ebml a = {b, b + n};
            while (a.p < a.end) {
                uint32_t aid;
                const uint8_t *ab;
                uint64_t an;
                if (element(&a, &aid, &ab, &an) != 0 || an == UNKNOWN)
                    break;
                if (aid == ID_RATE)
                    t->rate = float_of(ab, an);
                else if (aid == ID_CHANNELS)
                    t->channels = (int)uint_of(ab, an);
                else if (aid == ID_BITDEPTH)
                    t->bits = (int)uint_of(ab, an);
                a.p = ab + an;
            }
            break;
        }
        }
        e.p = b + n;
    }
    if (type != 2)
        t->number = 0;
}

/* a block's frames of the track: 0, or -1 */
static int read_block(const uint8_t *file, const uint8_t *p, uint64_t n,
                      uint64_t track, struct mframes *f, int *ours)
{
    const uint8_t *end = p + n;
    uint64_t num;
    int l;
    *ours = 0;
    if (vint(p, end, 1, &num, &l) != 0 || num != track)
        return 0;
    *ours = 1;
    p += l;
    if (end - p < 3)
        return -1;
    int flags = p[2];
    p += 3;
    int lacing = (flags >> 1) & 3;
    if (!lacing)
        return add_frame(f, (size_t)(p - file), (size_t)(end - p));
    if (p >= end)
        return -1;
    int count = *p++ + 1;
    size_t sizes[256], total = 0;
    if (lacing == 1) { /* Xiph */
        for (int i = 0; i < count - 1; i++) {
            size_t s = 0;
            int c;
            do {
                if (p >= end)
                    return -1;
                c = *p++;
                s += (size_t)c;
            } while (c == 255);
            sizes[i] = s;
            total += s;
        }
    } else if (lacing == 3) { /* EBML */
        uint64_t s;
        if (vint(p, end, 1, &s, &l) != 0)
            return -1;
        p += l;
        sizes[0] = (size_t)s;
        total = (size_t)s;
        for (int i = 1; i < count - 1; i++) {
            uint64_t raw;
            if (vint(p, end, 1, &raw, &l) != 0)
                return -1;
            p += l;
            int64_t diff = (int64_t)raw - (((int64_t)1 << (7 * l - 1)) - 1);
            int64_t next = (int64_t)sizes[i - 1] + diff;
            if (next < 0)
                return -1;
            sizes[i] = (size_t)next;
            total += (size_t)next;
        }
    } else { /* fixed */
        size_t left = (size_t)(end - p);
        if (left % (size_t)count)
            return -1;
        for (int i = 0; i < count - 1; i++) {
            sizes[i] = left / (size_t)count;
            total += sizes[i];
        }
    }
    if (total > (size_t)(end - p))
        return -1;
    sizes[count - 1] = (size_t)(end - p) - total;
    for (int i = 0; i < count; i++) {
        if (add_frame(f, (size_t)(p - file), sizes[i]) != 0)
            return -1;
        p += sizes[i];
    }
    return 0;
}

/* a cluster's blocks; an unknown size ends at the next element that is
   not a cluster's: its end */
static const uint8_t *read_cluster(const uint8_t *file, const uint8_t *p,
                                   const uint8_t *end, uint64_t track,
                                   struct mframes *f, int *err)
{
    struct ebml e = {p, end};
    while (e.p < e.end) {
        uint32_t id;
        const uint8_t *b;
        uint64_t n;
        if (element(&e, &id, &b, &n) != 0)
            return e.end;
        if (id == ID_CLUSTER || id == ID_TRACKS || id == 0x1C53BB6B ||
            id == 0x1254C367 || id == 0x114D9B74 || id == ID_INFO ||
            id == 0x1941A469 || id == 0x1043A770)
            return e.p; /* the next top-level element */
        if (n == UNKNOWN)
            return e.end;
        int ours;
        if (id == ID_SIMPLEBLOCK) {
            if (read_block(file, b, n, track, f, &ours) != 0)
                *err = 1;
        } else if (id == ID_BLOCKGROUP) {
            struct ebml g = {b, b + n};
            int64_t discard = 0;
            ours = 0;
            while (g.p < g.end) {
                uint32_t gid;
                const uint8_t *gb;
                uint64_t gn;
                if (element(&g, &gid, &gb, &gn) != 0 || gn == UNKNOWN)
                    break;
                if (gid == ID_BLOCK) {
                    if (read_block(file, gb, gn, track, f, &ours) != 0)
                        *err = 1;
                } else if (gid == ID_DISCARDPADDING && gn <= 8 && gn) {
                    uint64_t u = uint_of(gb, gn);
                    if (gn < 8 && u >> (8 * gn - 1))
                        u |= ~(uint64_t)0 << (8 * gn);
                    discard = (int64_t)u;
                }
                g.p = gb + gn;
            }
            if (ours)
                f->discard = discard;
        }
        e.p = b + n;
    }
    return e.end;
}

/* Vorbis's private data: the three headers, Xiph-laced */
static int vorbis_headers(const uint8_t *p, size_t n, struct ogg_packets *o)
{
    if (n < 1 || p[0] != 2)
        return -1;
    size_t at = 1, s[2];
    for (int i = 0; i < 2; i++) {
        size_t v = 0;
        int c;
        do {
            if (at >= n)
                return -1;
            c = p[at++];
            v += (size_t)c;
        } while (c == 255);
        s[i] = v;
    }
    if (s[0] + s[1] > n - at)
        return -1;
    if (ogg_add_packet(o, p + at, s[0], -1) != 0 ||
        ogg_add_packet(o, p + at + s[0], s[1], -1) != 0 ||
        ogg_add_packet(o, p + at + s[0] + s[1], n - at - s[0] - s[1], -1) != 0)
        return -1;
    return 0;
}

static int concat_decode(const uint8_t *head, size_t hn, const uint8_t *d,
                         const struct mframes *f,
                         int (*dec)(const uint8_t *, size_t,
                                    struct audio_out *, char *, size_t),
                         struct audio_out *ao, char *why, size_t wn)
{
    size_t total = hn;
    for (size_t i = 0; i < f->count; i++)
        total += f->len[i];
    uint8_t *buf = malloc(total ? total : 1);
    if (!buf) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    if (hn)
        memcpy(buf, head, hn);
    size_t at = hn;
    for (size_t i = 0; i < f->count; i++) {
        memcpy(buf + at, d + f->off[i], f->len[i]);
        at += f->len[i];
    }
    int rc = dec(buf, total, ao, why, wn);
    free(buf);
    if (rc > 0) {
        snprintf(why, wn, "a Matroska track that is not what it says");
        rc = -1;
    }
    return rc;
}

/* the AudioSpecificConfig of an old A_AAC/MPEGn/PROFILE[/SBR] codec id */
static size_t aac_config_of(const struct mtrack *t, uint8_t *asc)
{
    int aot = strstr(t->codec, "/MAIN")  ? 1
              : strstr(t->codec, "/LTP") ? 4
              : strstr(t->codec, "/SSR") ? 3
                                         : 2;
    int sbr = strstr(t->codec, "/SBR") != NULL;
    int rate = (int)t->rate, core = sbr ? rate / 2 : rate;
    int idx = aac_rate_index(core), ch = t->channels ? t->channels : 2;
    asc[0] = (uint8_t)(aot << 3 | idx >> 1);
    asc[1] = (uint8_t)((idx & 1) << 7 | (ch & 15) << 3);
    if (!sbr)
        return 2;
    int eidx = aac_rate_index(rate);
    asc[2] = 0x56; /* sync extension 0x2b7, SBR */
    asc[3] = 0xE5;
    asc[4] = (uint8_t)(0x80 | eidx << 3);
    return 5;
}

static int pcm_decode(const struct mtrack *t, const uint8_t *d,
                      const struct mframes *f, struct audio_out *ao, char *why,
                      size_t wn)
{
    int big = strstr(t->codec, "BIG") != NULL;
    int fl = strstr(t->codec, "FLOAT") != NULL;
    int bits = t->bits ? t->bits : 16, ch = t->channels ? t->channels : 1;
    int bps = bits / 8;
    if (bps < 1 || bps > 8 || bits % 8 || ch > 64 || !t->rate ||
        (fl && bps != 4 && bps != 8)) {
        snprintf(why, wn, "a Matroska track of PCM out of range");
        return -1;
    }
    float pcm[4096];
    size_t at = 0;
    ao->rate = (uint32_t)t->rate;
    for (size_t i = 0; i < f->count; i++) {
        const uint8_t *p = d + f->off[i];
        size_t nf = f->len[i] / ((size_t)bps * (size_t)ch);
        for (size_t k = 0; k < nf; k++) {
            double s = 0;
            for (int c = 0; c < ch; c++, p += bps) {
                uint64_t u = 0;
                for (int j = 0; j < bps; j++)
                    u |= (uint64_t)p[big ? j : bps - 1 - j]
                         << (8 * (bps - 1 - j));
                if (fl && bps == 4) {
                    uint32_t w = (uint32_t)u;
                    float x;
                    memcpy(&x, &w, 4);
                    s += x;
                } else if (fl) {
                    double x;
                    memcpy(&x, &u, 8);
                    s += x;
                } else if (bps == 1) {
                    s += ((double)u - 128) / 128;
                } else {
                    int64_t v = (int64_t)(u << (64 - bits)) >> (64 - bits);
                    s += (double)v / (double)((int64_t)1 << (bits - 1));
                }
            }
            pcm[at++] = (float)(s / ch);
            if (at == 4096) {
                if (audio_put(ao, pcm, at, why, wn) != 0)
                    return -1;
                at = 0;
            }
        }
    }
    return audio_put(ao, pcm, at, why, wn);
}

int janas_audio_mkv(const uint8_t *d, size_t n, struct audio_out *ao,
                    char *why, size_t wn)
{
    if (n < 4 || memcmp(d, "\x1a\x45\xdf\xa3", 4))
        return 1;
    struct ebml top = {d, d + n};
    const uint8_t *seg = NULL, *seg_end = NULL;
    while (top.p < top.end) {
        uint32_t id;
        const uint8_t *b;
        uint64_t s;
        if (element(&top, &id, &b, &s) != 0)
            break;
        if (id == ID_SEGMENT) {
            seg = b;
            seg_end = s == UNKNOWN ? d + n : b + s;
            break;
        }
        if (s == UNKNOWN)
            break;
        top.p = b + s;
    }
    if (!seg) {
        snprintf(why, wn, "a Matroska file without a segment");
        return -1;
    }
    struct mtrack t;
    memset(&t, 0, sizeof t);
    struct mframes f;
    memset(&f, 0, sizeof f);
    int found = 0, err = 0, rc = -1;
    struct ebml e = {seg, seg_end};
    while (e.p < e.end) {
        uint32_t id;
        const uint8_t *b;
        uint64_t s;
        if (element(&e, &id, &b, &s) != 0)
            break;
        if (id == ID_TRACKS && !found && s != UNKNOWN) {
            struct ebml tr = {b, b + s};
            while (tr.p < tr.end && !found) {
                uint32_t tid;
                const uint8_t *tb;
                uint64_t tn;
                if (element(&tr, &tid, &tb, &tn) != 0 || tn == UNKNOWN)
                    break;
                if (tid == ID_TRACKENTRY) {
                    read_track(tb, tb + tn, &t);
                    found = t.number != 0;
                }
                tr.p = tb + tn;
            }
            e.p = b + s;
        } else if (id == ID_CLUSTER) {
            if (!found) {
                if (s == UNKNOWN)
                    break;
                e.p = b + s;
                continue;
            }
            const uint8_t *cend = s == UNKNOWN ? e.end : b + s;
            const uint8_t *next = read_cluster(d, b, cend, t.number, &f, &err);
            if (f.count > JANAS_AUDIO_MAX_SAMPLES / 64) {
                snprintf(why, wn, "a Matroska file of too many blocks");
                goto end;
            }
            e.p = s == UNKNOWN ? next : b + s;
        } else {
            if (s == UNKNOWN)
                break;
            e.p = b + s;
        }
    }
    if (!found) {
        snprintf(why, wn, "a Matroska file without a track of sound");
        goto end;
    }
    if (t.encoded) {
        snprintf(why, wn,
                 "a Matroska track whose blocks are compressed or "
                 "encrypted, not read");
        goto end;
    }
    if (!f.count) {
        snprintf(why, wn, "a Matroska track of sound without blocks");
        goto end;
    }
    const char *c = t.codec;
    if (!strcmp(c, "A_OPUS")) {
        struct ogg_packets o;
        memset(&o, 0, sizeof o);
        o.codec = 2;
        o.pre_skip = t.codec_delay
                         ? (int64_t)((t.codec_delay * 48000 + 500000000ull) /
                                     1000000000ull)
                         : -1;
        if (!t.priv || ogg_add_packet(&o, t.priv, t.priv_len, -1) != 0) {
            ogg_packets_free(&o);
            snprintf(why, wn, "a Matroska track of Opus without its header");
            goto end;
        }
        int64_t total = 0;
        for (size_t i = 0; i < f.count; i++)
            total += janas_opus_packet_samples(d + f.off[i], f.len[i]);
        int64_t last = -1;
        if (f.discard > 0)
            last = total - (f.discard * 48000 + 500000000) / 1000000000;
        for (size_t i = 0; i < f.count; i++)
            if (ogg_add_packet(&o, d + f.off[i], f.len[i],
                               i + 1 == f.count ? last : -1) != 0) {
                ogg_packets_free(&o);
                snprintf(why, wn, "out of memory");
                goto end;
            }
        rc = janas_opus_stream(&o, ao, why, wn);
        ogg_packets_free(&o);
    } else if (!strcmp(c, "A_VORBIS")) {
        struct ogg_packets o;
        memset(&o, 0, sizeof o);
        o.codec = 1;
        o.pre_skip = -1;
        if (!t.priv || vorbis_headers(t.priv, t.priv_len, &o) != 0) {
            ogg_packets_free(&o);
            snprintf(why, wn, "a Matroska track of Vorbis without its headers");
            goto end;
        }
        for (size_t i = 0; i < f.count; i++)
            if (ogg_add_packet(&o, d + f.off[i], f.len[i], -1) != 0) {
                ogg_packets_free(&o);
                snprintf(why, wn, "out of memory");
                goto end;
            }
        rc = janas_vorbis_stream(&o, ao, why, wn);
        ogg_packets_free(&o);
    } else if (!strcmp(c, "A_FLAC")) {
        if (!t.priv || t.priv_len < 4 + 4 + 34 || memcmp(t.priv, "fLaC", 4)) {
            snprintf(why, wn, "a Matroska track of FLAC without its header");
            goto end;
        }
        rc = concat_decode(t.priv, t.priv_len, d, &f, janas_audio_flac, ao,
                           why, wn);
    } else if (!strncmp(c, "A_AAC", 5)) {
        struct aac_cfg cfg;
        uint8_t asc[8];
        const uint8_t *cp = t.priv;
        size_t cn = t.priv_len;
        if (!cp || !cn) {
            cn = aac_config_of(&t, asc);
            cp = asc;
        }
        if (aac_config(cp, cn, &cfg, why, wn) != 0)
            goto end;
        rc = aac_stream(&cfg, d, f.off, f.len, f.count, NULL, ao, why, wn);
    } else if (!strncmp(c, "A_MPEG/L", 8)) {
        rc = concat_decode(NULL, 0, d, &f, janas_audio_mpeg, ao, why, wn);
    } else if (!strcmp(c, "A_ALAC")) {
        if (!t.priv) {
            snprintf(why, wn,
                     "a Matroska track of ALAC without its configuration");
            goto end;
        }
        rc = janas_alac_stream(t.priv, t.priv_len, d, f.off, f.len, f.count,
                               NULL, ao, why, wn);
    } else if (!strncmp(c, "A_PCM/", 6)) {
        rc = pcm_decode(&t, d, &f, ao, why, wn);
    } else {
        snprintf(why, wn, "a Matroska track of a codec not read (%s)",
                 c[0] ? c : "none named");
    }
    /* the codec delay and the last block's discard padding, for the
       codecs that do not take them themselves (Opus does) */
    if (rc == 0 && strcmp(c, "A_OPUS") && (t.codec_delay || f.discard > 0)) {
        uint64_t skip =
            (t.codec_delay * ao->rate + 500000000ull) / 1000000000ull;
        uint64_t cut =
            f.discard > 0
                ? ((uint64_t)f.discard * ao->rate + 500000000ull) / 1000000000ull
                : 0;
        uint64_t kept = audio_out_kept(ao);
        if (skip + cut >= kept) {
            snprintf(why, wn, "a Matroska track without sound");
            rc = -1;
        } else
            audio_out_trim(ao, skip, kept - skip - cut);
    }
end:
    (void)err;
    free(f.off);
    free(f.len);
    return rc;
}
