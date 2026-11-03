/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_mp4.c - the ISO base media file format (MP4, M4A, 3GP, MOV): the
 * first track of sound, its sample description (AAC and MP3 by their
 * esds, ALAC, Opus, FLAC, PCM), its samples found through the sample
 * tables (stsz or stz2, stco or co64, stsc) or the movie fragments (moof,
 * traf, tfhd, trun), the edit list's start and length that trim the
 * encoder's delay and padding. The codecs decode the samples where they
 * lie in the file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio.h"
#include "audio_aac.h"
#include "audio_ogg.h"

static uint32_t be32(const uint8_t *p)
{
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 |
           p[3];
}

static uint64_t be64(const uint8_t *p)
{
    return (uint64_t)be32(p) << 32 | be32(p + 4);
}

static uint16_t be16(const uint8_t *p)
{
    return (uint16_t)(p[0] << 8 | p[1]);
}

/* a box at p of the n bytes left: its type, payload and length; 0, or -1 */
struct box {
    uint32_t type;
    const uint8_t *body;
    size_t size, len; /* the whole box, its payload */
};

static int box_at(const uint8_t *p, size_t n, struct box *b)
{
    if (n < 8)
        return -1;
    uint64_t size = be32(p);
    size_t head = 8;
    if (size == 1) {
        if (n < 16)
            return -1;
        size = be64(p + 8);
        head = 16;
    } else if (size == 0) {
        size = n;
    }
    if (size < head || size > n)
        return -1;
    b->type = be32(p + 4);
    b->body = p + head;
    b->size = (size_t)size;
    b->len = (size_t)size - head;
    return 0;
}

#define TAG(s)                                                                 \
    ((uint32_t)(s)[0] << 24 | (uint32_t)(s)[1] << 16 | (uint32_t)(s)[2] << 8 | \
     (uint32_t)(s)[3])

/* the first child of type t in a container's payload */
static int child(const uint8_t *p, size_t n, uint32_t t, struct box *out)
{
    while (n >= 8) {
        struct box b;
        if (box_at(p, n, &b) != 0)
            return -1;
        if (b.type == t) {
            *out = b;
            return 0;
        }
        p += b.size;
        n -= b.size;
    }
    return -1;
}

struct track {
    uint32_t codec; /* the sample entry's type */
    int channels, bits, rate;
    uint32_t timescale;
    const uint8_t *config; /* esds' DecoderSpecificInfo, ALAC's cookie,
                              dOps, dfLa */
    size_t config_len;
    int object_type;     /* esds' objectTypeIndication */
    uint32_t lpcm_flags; /* QuickTime's version 2: 1 float, 2 big-endian,
                            4 signed; little-endian when an enda says */
    int little;
    /* the samples */
    size_t *off, *len;
    size_t count, cap;
    uint64_t total_duration;
    /* the edit list */
    int64_t media_time;   /* -1: none */
    uint64_t edit_length; /* in the movie's timescale, 0: none */
    uint32_t movie_timescale;
    uint32_t track_id;
    /* fragments' defaults (trex) */
    uint32_t trex_duration, trex_size;
};

static int add_sample(struct track *t, size_t off, size_t len)
{
    if (t->count == t->cap) {
        size_t c = t->cap ? t->cap * 2 : 1024;
        size_t *o = realloc(t->off, sizeof(size_t) * c);
        if (o)
            t->off = o;
        size_t *l = realloc(t->len, sizeof(size_t) * c);
        if (l)
            t->len = l;
        if (!o || !l)
            return -1;
        t->cap = c;
    }
    t->off[t->count] = off;
    t->len[t->count] = len;
    t->count++;
    return 0;
}

/* an MPEG-4 descriptor's tag and length at p: its payload, or NULL */
static const uint8_t *descriptor(const uint8_t *p, const uint8_t *end, int *tag,
                                 size_t *len)
{
    if (p >= end)
        return NULL;
    *tag = *p++;
    size_t l = 0;
    for (int i = 0; i < 4 && p < end; i++) {
        uint8_t c = *p++;
        l = l << 7 | (c & 0x7F);
        if (!(c & 0x80))
            break;
    }
    if (l > (size_t)(end - p))
        return NULL;
    *len = l;
    return p;
}

static void read_esds(struct track *t, const uint8_t *p, size_t n)
{
    if (n < 4)
        return;
    const uint8_t *end = p + n;
    p += 4; /* version, flags */
    int tag;
    size_t len;
    const uint8_t *d = descriptor(p, end, &tag, &len);
    if (!d || tag != 3 || len < 3)
        return;
    const uint8_t *de = d + len;
    int flags = d[2];
    d += 3;
    if (flags & 0x80)
        d += 2;
    if (flags & 0x40) {
        if (d >= de)
            return;
        d += 1 + *d;
    }
    if (flags & 0x20)
        d += 2;
    if (d >= de)
        return;
    const uint8_t *dc = descriptor(d, de, &tag, &len);
    if (!dc || tag != 4 || len < 13)
        return;
    t->object_type = dc[0];
    const uint8_t *dce = dc + len;
    const uint8_t *ds = descriptor(dc + 13, dce, &tag, &len);
    if (ds && tag == 5) {
        t->config = ds;
        t->config_len = len;
    }
}

static void read_stsd(struct track *t, const uint8_t *p, size_t n)
{
    if (n < 8 || be32(p + 4) == 0)
        return;
    struct box e;
    if (box_at(p + 8, n - 8, &e) != 0 || e.len < 28)
        return;
    t->codec = e.type;
    const uint8_t *b = e.body;
    int version = be16(b + 8);
    t->channels = be16(b + 16);
    t->bits = be16(b + 18);
    t->rate = (int)(be32(b + 24) >> 16);
    size_t at = 28;
    if (version == 1)
        at += 16;
    else if (version == 2) {
        at += 36;
        if (e.len >= 64) {
            /* QuickTime's version 2: a double rate and 32-bit fields */
            union {
                uint64_t u;
                double d;
            } r = {.u = be64(b + 32)};
            t->rate = (int)r.d;
            t->channels = (int)be32(b + 40);
            t->bits = (int)be32(b + 48);
            t->lpcm_flags = be32(b + 52);
        }
    }
    if (at > e.len)
        return;
    const uint8_t *q = b + at;
    size_t left = e.len - at;
    struct box c;
    if (child(q, left, TAG("esds"), &c) == 0)
        read_esds(t, c.body, c.len);
    else if (child(q, left, TAG("wave"), &c) == 0) {
        struct box en;
        if (child(c.body, c.len, TAG("enda"), &en) == 0 && en.len >= 2)
            t->little = en.body[1] & 1;
        /* QuickTime's AAC: the esds inside a wave box */
        struct box c2;
        if (child(c.body, c.len, TAG("esds"), &c2) == 0)
            read_esds(t, c2.body, c2.len);
        else if (child(c.body, c.len, TAG("alac"), &c2) == 0) {
            t->config = c2.body;
            t->config_len = c2.len;
        }
    }
    if (e.type == TAG("alac") && child(q, left, TAG("alac"), &c) == 0) {
        t->config = c.body;
        t->config_len = c.len;
    } else if (e.type == TAG("Opus") && child(q, left, TAG("dOps"), &c) == 0) {
        t->config = c.body;
        t->config_len = c.len;
    } else if (e.type == TAG("fLaC") && child(q, left, TAG("dfLa"), &c) == 0) {
        t->config = c.body;
        t->config_len = c.len;
    }
}

/* the sample tables: 0, or -1 */
static int read_stbl(struct track *t, const uint8_t *p, size_t n,
                     const uint8_t *file, size_t fn)
{
    struct box stsz, stsc, stco, stts;
    int co64 = 0, z2 = 0;
    if (child(p, n, TAG("stsd"), &stsz) == 0)
        read_stsd(t, stsz.body, stsz.len);
    if (child(p, n, TAG("stsz"), &stsz) != 0) {
        if (child(p, n, TAG("stz2"), &stsz) != 0)
            return 0; /* fragmented: the samples come with the fragments */
        z2 = 1;
    }
    if (child(p, n, TAG("stco"), &stco) != 0) {
        if (child(p, n, TAG("co64"), &stco) != 0)
            return -1;
        co64 = 1;
    }
    if (child(p, n, TAG("stsc"), &stsc) != 0 || stsz.len < 12 || stco.len < 8 ||
        stsc.len < 8)
        return -1;
    if (child(p, n, TAG("stts"), &stts) == 0 && stts.len >= 8) {
        uint32_t e = be32(stts.body + 4);
        for (uint32_t i = 0; i < e && 8 + 8 * (size_t)i + 8 <= stts.len; i++)
            t->total_duration += (uint64_t)be32(stts.body + 8 + 8 * i) *
                                 be32(stts.body + 12 + 8 * i);
    }
    uint32_t fixed = 0, field = 0, nsamples;
    if (z2) {
        field = stsz.body[7];
        nsamples = be32(stsz.body + 8);
        if (field != 4 && field != 8 && field != 16)
            return -1;
        if ((size_t)nsamples * field / 8 > stsz.len - 12)
            return -1;
    } else {
        fixed = be32(stsz.body + 4);
        nsamples = be32(stsz.body + 8);
        if (!fixed && (size_t)nsamples * 4 > stsz.len - 12)
            return -1;
    }
    uint32_t nchunks = be32(stco.body + 4);
    if ((size_t)nchunks * (co64 ? 8 : 4) > stco.len - 8)
        return -1;
    uint32_t nstsc = be32(stsc.body + 4);
    if ((size_t)nstsc * 12 > stsc.len - 8)
        return -1;
    uint32_t s = 0;
    for (uint32_t i = 0; i < nstsc && s < nsamples; i++) {
        const uint8_t *e = stsc.body + 8 + 12 * (size_t)i;
        uint32_t first = be32(e), per = be32(e + 4);
        uint32_t last = i + 1 < nstsc ? be32(e + 12) : nchunks + 1;
        if (first == 0 || last < first)
            return -1;
        for (uint32_t ch = first; ch < last && ch <= nchunks && s < nsamples;
             ch++) {
            uint64_t off = co64 ? be64(stco.body + 8 + 8 * (size_t)(ch - 1))
                                : be32(stco.body + 8 + 4 * (size_t)(ch - 1));
            for (uint32_t k = 0; k < per && s < nsamples; k++, s++) {
                uint32_t len;
                if (z2) {
                    const uint8_t *z = stsz.body + 12;
                    len = field == 16 ? be16(z + 2 * (size_t)s)
                          : field == 8
                              ? z[s]
                              : (uint32_t)(z[s / 2] >> (s & 1 ? 0 : 4)) & 15;
                } else {
                    len = fixed ? fixed : be32(stsz.body + 12 + 4 * (size_t)s);
                }
                if (off > fn || len > fn - off)
                    return t->count ? 0 : -1; /* a file cut short */
                if (add_sample(t, (size_t)off, len) != 0)
                    return -1;
                off += len;
            }
        }
    }
    (void)file;
    return 0;
}

static int read_trak(struct track *t, const uint8_t *p, size_t n,
                     const uint8_t *file, size_t fn)
{
    struct box mdia, hdlr, minf, stbl, mdhd, tkhd, edts, elst;
    if (child(p, n, TAG("mdia"), &mdia) != 0 ||
        child(mdia.body, mdia.len, TAG("hdlr"), &hdlr) != 0 || hdlr.len < 12 ||
        be32(hdlr.body + 8) != TAG("soun"))
        return 1;
    if (child(p, n, TAG("tkhd"), &tkhd) == 0 && tkhd.len >= 16)
        t->track_id = be32(tkhd.body + (tkhd.body[0] == 1 ? 20 : 12));
    if (child(mdia.body, mdia.len, TAG("mdhd"), &mdhd) == 0 && mdhd.len >= 24)
        t->timescale = be32(mdhd.body + (mdhd.body[0] == 1 ? 20 : 12));
    t->media_time = -1;
    if (child(p, n, TAG("edts"), &edts) == 0 &&
        child(edts.body, edts.len, TAG("elst"), &elst) == 0 && elst.len >= 8) {
        int v = elst.body[0];
        uint32_t cnt = be32(elst.body + 4);
        size_t es = v == 1 ? 20 : 12;
        /* an empty edit first (a delay) is passed over: the sound's own
           edit is the one with a media time */
        for (uint32_t i = 0; i < cnt && 8 + es * (i + 1) <= elst.len; i++) {
            const uint8_t *e = elst.body + 8 + es * i;
            uint64_t dur = v == 1 ? be64(e) : be32(e);
            int64_t mt = v == 1 ? (int64_t)be64(e + 8) : (int32_t)be32(e + 4);
            if (mt < 0)
                continue;
            t->media_time = mt;
            t->edit_length = dur;
            break;
        }
    }
    if (child(mdia.body, mdia.len, TAG("minf"), &minf) != 0 ||
        child(minf.body, minf.len, TAG("stbl"), &stbl) != 0)
        return -1;
    return read_stbl(t, stbl.body, stbl.len, file, fn);
}

/* a movie fragment's runs of the track */
static int read_moof(struct track *t, const uint8_t *moof, size_t n,
                     const uint8_t *file, size_t fn)
{
    size_t moof_at = (size_t)(moof - file);
    const uint8_t *p = moof + 8;
    size_t left = n - 8;
    if (n >= 16 && be32(moof) == 1)
        p = moof + 16, left = n - 16;
    while (left >= 8) {
        struct box traf;
        if (box_at(p, left, &traf) != 0)
            return -1;
        p += traf.size;
        left -= traf.size;
        if (traf.type != TAG("traf"))
            continue;
        struct box tfhd;
        if (child(traf.body, traf.len, TAG("tfhd"), &tfhd) != 0 || tfhd.len < 8)
            continue;
        uint32_t tf = be32(tfhd.body) & 0xFFFFFF;
        if (be32(tfhd.body + 4) != t->track_id && t->track_id)
            continue;
        size_t at = 8;
        uint64_t base = moof_at;
        uint32_t def_size = t->trex_size;
        if (tf & 1) {
            if (at + 8 > tfhd.len)
                return -1;
            base = be64(tfhd.body + at);
            at += 8;
        }
        if (tf & 2)
            at += 4;
        if (tf & 8)
            at += 4;
        if (tf & 0x10) {
            if (at + 4 > tfhd.len)
                return -1;
            def_size = be32(tfhd.body + at);
            at += 4;
        }
        const uint8_t *q = traf.body;
        size_t ql = traf.len;
        uint64_t next = base;
        while (ql >= 8) {
            struct box tr;
            if (box_at(q, ql, &tr) != 0)
                return -1;
            q += tr.size;
            ql -= tr.size;
            if (tr.type != TAG("trun") || tr.len < 8)
                continue;
            uint32_t fl = be32(tr.body) & 0xFFFFFF;
            uint32_t cnt = be32(tr.body + 4);
            size_t r = 8;
            uint64_t off = next;
            if (fl & 1) {
                if (r + 4 > tr.len)
                    return -1;
                off = base + (uint64_t)(int64_t)(int32_t)be32(tr.body + r);
                r += 4;
            }
            if (fl & 4)
                r += 4;
            size_t per = (fl & 0x100 ? 4 : 0) + (fl & 0x200 ? 4 : 0) +
                         (fl & 0x400 ? 4 : 0) + (fl & 0x800 ? 4 : 0);
            if (r > tr.len || (per && cnt > (tr.len - r) / per))
                return -1;
            for (uint32_t i = 0; i < cnt; i++) {
                const uint8_t *e = tr.body + r + per * i;
                uint32_t size = def_size;
                if (fl & 0x100) {
                    t->total_duration += be32(e);
                    e += 4;
                } else {
                    t->total_duration += t->trex_duration;
                }
                if (fl & 0x200)
                    size = be32(e);
                if (off > fn || size > fn - off)
                    return 0;
                if (add_sample(t, (size_t)off, size) != 0)
                    return -1;
                off += size;
            }
            next = off;
        }
    }
    return 0;
}

/* ---- the codecs ---- */

/* MP3 in MP4: the frames one after another, as an MP3 file */
static int mp4_mpeg(const struct track *t, const uint8_t *d,
                    struct audio_out *ao, char *why, size_t wn)
{
    /* the frames carry no gapless tag here: the edit list trims */
    size_t total = 0;
    for (size_t i = 0; i < t->count; i++)
        total += t->len[i];
    uint8_t *buf = malloc(total ? total : 1);
    if (!buf) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    size_t at = 0;
    for (size_t i = 0; i < t->count; i++) {
        memcpy(buf + at, d + t->off[i], t->len[i]);
        at += t->len[i];
    }
    int rc = janas_audio_mpeg(buf, total, ao, why, wn);
    free(buf);
    if (rc > 0) {
        snprintf(why, wn, "an MP4 track of MPEG audio without frames");
        rc = -1;
    }
    if (rc == 0) {
        struct audio_edit ed = {t->media_time, t->edit_length,
                                t->timescale ? t->timescale : ao->rate,
                                t->movie_timescale};
        size_t skip = 0, keep = (size_t)audio_out_kept(ao);
        audio_edit_apply(&ed, ao->rate, &skip, &keep);
        if (!keep) {
            snprintf(why, wn, "an MP4 track of MPEG audio without sound");
            return -1;
        }
        audio_out_trim(ao, skip, keep);
    }
    return rc;
}

/* FLAC in MP4: the dfLa's metadata blocks and the frames, as a FLAC file */
static int mp4_flac(const struct track *t, const uint8_t *d,
                    struct audio_out *ao, char *why, size_t wn)
{
    if (!t->config || t->config_len < 4 + 4 + 34) {
        snprintf(why, wn, "an MP4 track of FLAC without its STREAMINFO");
        return -1;
    }
    size_t total = 4 + t->config_len - 4;
    for (size_t i = 0; i < t->count; i++)
        total += t->len[i];
    uint8_t *buf = malloc(total);
    if (!buf) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    memcpy(buf, "fLaC", 4);
    memcpy(buf + 4, t->config + 4, t->config_len - 4);
    size_t at = t->config_len;
    for (size_t i = 0; i < t->count; i++) {
        memcpy(buf + at, d + t->off[i], t->len[i]);
        at += t->len[i];
    }
    int rc = janas_audio_flac(buf, total, ao, why, wn);
    free(buf);
    if (rc > 0) {
        snprintf(why, wn, "an MP4 track of FLAC that is none");
        rc = -1;
    }
    return rc;
}

/* Opus in MP4: an OpusHead made from the dOps, the edit list's length as
   the last granule position */
static int mp4_opus(const struct track *t, const uint8_t *d,
                    struct audio_out *ao, char *why, size_t wn)
{
    const uint8_t *c = t->config;
    if (!c || t->config_len < 11 || c[0] != 0) {
        snprintf(why, wn, "an MP4 track of Opus without its dOps");
        return -1;
    }
    uint8_t head[19 + 2 + 255];
    int channels = c[1], family = c[10];
    size_t hn = 19;
    memcpy(head, "OpusHead", 8);
    head[8] = 1;
    head[9] = (uint8_t)channels;
    head[10] = c[3]; /* the pre-skip, little-endian in Ogg */
    head[11] = c[2];
    head[12] = c[7];
    head[13] = c[6];
    head[14] = c[5];
    head[15] = c[4];
    head[16] = c[9];
    head[17] = c[8];
    head[18] = (uint8_t)family;
    if (family) {
        if (t->config_len < 13 + (size_t)channels) {
            snprintf(why, wn,
                     "an MP4 track of Opus with its mapping cut short");
            return -1;
        }
        memcpy(head + 19, c + 11, 2 + (size_t)channels);
        hn += 2 + (size_t)channels;
    }
    struct ogg_packets o;
    memset(&o, 0, sizeof o);
    o.codec = 2;
    o.pre_skip = -1;
    int rc = -1;
    if (ogg_add_packet(&o, head, hn, -1) != 0)
        goto oom;
    int64_t last = -1;
    if (t->media_time >= 0 && t->edit_length && t->movie_timescale &&
        t->timescale)
        last = t->media_time * 48000 / (int64_t)t->timescale +
               (int64_t)(t->edit_length * 48000 / t->movie_timescale);
    for (size_t i = 0; i < t->count; i++)
        if (ogg_add_packet(&o, d + t->off[i], t->len[i],
                           i + 1 == t->count ? last : -1) != 0)
            goto oom;
    rc = janas_opus_stream(&o, ao, why, wn);
    ogg_packets_free(&o);
    return rc;
oom:
    ogg_packets_free(&o);
    snprintf(why, wn, "out of memory");
    return -1;
}

/* uncompressed sound in a sample entry of its own */
static int mp4_pcm(const struct track *t, const uint8_t *d,
                   struct audio_out *ao, char *why, size_t wn)
{
    int big = !t->little, fl = 0, bits = t->bits, uns = 0;
    uint32_t c = t->codec;
    if (c == TAG("lpcm")) {
        fl = t->lpcm_flags & 1;
        big = (t->lpcm_flags & 2) != 0;
        uns = !fl && !(t->lpcm_flags & 4) && bits == 8;
    } else if (c == TAG("sowt"))
        big = 0, bits = 16;
    else if (c == TAG("twos"))
        bits = t->bits == 8 ? 8 : 16;
    else if (c == TAG("in24"))
        bits = 24;
    else if (c == TAG("in32"))
        bits = 32;
    else if (c == TAG("fl32"))
        bits = 32, fl = 1;
    else if (c == TAG("fl64"))
        bits = 64, fl = 1;
    else if (c == TAG("raw "))
        bits = 8;
    int ch = t->channels, bps = bits / 8;
    if (ch <= 0 || ch > 64 || !t->rate || !bps || bps > 8 ||
        (fl && bps != 4 && bps != 8)) {
        snprintf(why, wn, "an MP4 track of uncompressed sound out of range");
        return -1;
    }
    float pcm[4096];
    size_t at = 0;
    ao->rate = (uint32_t)t->rate;
    for (size_t i = 0; i < t->count; i++) {
        const uint8_t *p = d + t->off[i];
        size_t nf = t->len[i] / ((size_t)bps * (size_t)ch);
        for (size_t f = 0; f < nf; f++) {
            double s = 0;
            for (int k = 0; k < ch; k++, p += bps) {
                uint64_t u = 0;
                for (int j = 0; j < bps; j++)
                    u |= (uint64_t)p[big ? j : bps - 1 - j]
                         << (8 * (bps - 1 - j));
                double v;
                if (fl && bits == 32) {
                    uint32_t w = (uint32_t)u;
                    float x;
                    memcpy(&x, &w, 4);
                    v = x;
                } else if (fl) {
                    double x;
                    memcpy(&x, &u, 8);
                    v = x;
                } else if (c == TAG("raw ") || uns) {
                    v = ((double)u - 128) / 128;
                } else {
                    int64_t sv = (int64_t)(u << (64 - bits)) >> (64 - bits);
                    v = (double)sv / (double)((int64_t)1 << (bits - 1));
                }
                s += v;
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

int janas_audio_mp4(const uint8_t *d, size_t n, struct audio_out *ao,
                    char *why, size_t wn)
{
    struct box ftyp;
    if (box_at(d, n, &ftyp) != 0 ||
        (ftyp.type != TAG("ftyp") && ftyp.type != TAG("moov") &&
         ftyp.type != TAG("mdat") && ftyp.type != TAG("free") &&
         ftyp.type != TAG("wide") && ftyp.type != TAG("skip")))
        return 1;
    struct track t;
    memset(&t, 0, sizeof t);
    t.media_time = -1;
    int rc = -1, found = 0;
    /* the movie, then the fragments */
    const uint8_t *p = d;
    size_t left = n;
    while (left >= 8) {
        struct box b;
        if (box_at(p, left, &b) != 0)
            break; /* a box cut short at the end */
        if (b.type == TAG("moov") && !found) {
            struct box mvhd, mvex, trex;
            if (child(b.body, b.len, TAG("mvhd"), &mvhd) == 0 && mvhd.len >= 20)
                t.movie_timescale =
                    be32(mvhd.body + (mvhd.body[0] == 1 ? 20 : 12));
            const uint8_t *q = b.body;
            size_t ql = b.len;
            while (ql >= 8 && !found) {
                struct box tk;
                if (box_at(q, ql, &tk) != 0)
                    break;
                if (tk.type == TAG("trak")) {
                    int r = read_trak(&t, tk.body, tk.len, d, n);
                    if (r < 0) {
                        snprintf(why, wn,
                                 "an MP4 file whose sample tables are broken");
                        goto end;
                    }
                    if (r == 0)
                        found = 1;
                }
                q += tk.size;
                ql -= tk.size;
            }
            if (found && child(b.body, b.len, TAG("mvex"), &mvex) == 0) {
                const uint8_t *m = mvex.body;
                size_t ml = mvex.len;
                while (ml >= 8 && child(m, ml, TAG("trex"), &trex) == 0) {
                    if (trex.len >= 24 && be32(trex.body + 4) == t.track_id) {
                        t.trex_duration = be32(trex.body + 12);
                        t.trex_size = be32(trex.body + 16);
                        break;
                    }
                    size_t skip = (size_t)(trex.body + trex.len - m);
                    m += skip;
                    ml -= skip;
                }
            }
        } else if (b.type == TAG("moof") && found) {
            if (read_moof(&t, p, b.size, d, n) != 0) {
                snprintf(why, wn, "an MP4 file whose fragments are broken");
                goto end;
            }
        }
        p += b.size;
        left -= b.size;
    }
    if (!found) {
        snprintf(why, wn, "an MP4 file without a track of sound");
        goto end;
    }
    if (!t.count) {
        snprintf(why, wn, "an MP4 track of sound without samples");
        goto end;
    }
    uint32_t c = t.codec;
    char fourcc[5] = {(char)(c >> 24), (char)(c >> 16), (char)(c >> 8), (char)c,
                      0};
    for (int i = 0; i < 4; i++)
        if (fourcc[i] < 32 || fourcc[i] > 126)
            fourcc[i] = '?';
    if (c == TAG("mp4a") || c == TAG("enca")) {
        int ot = t.object_type;
        if (ot == 0x69 || ot == 0x6B) {
            rc = mp4_mpeg(&t, d, ao, why, wn);
        } else if (ot == 0x40 || ot == 0x66 || ot == 0x67 || ot == 0x68 ||
                   (!ot && t.config)) {
            struct aac_cfg cfg;
            if (!t.config) {
                snprintf(why, wn,
                         "an MP4 track of AAC without its configuration");
                goto end;
            }
            if (aac_config(t.config, t.config_len, &cfg, why, wn) != 0)
                goto end;
            if (ot >= 0x66 && ot <= 0x68)
                cfg.aot = ot == 0x66 ? AOT_MAIN : ot == 0x67 ? AOT_LC : cfg.aot;
            struct audio_edit ed = {t.media_time, t.edit_length,
                                    t.timescale ? t.timescale
                                                : (uint32_t)cfg.rate,
                                    t.movie_timescale};
            rc = aac_stream(&cfg, d, t.off, t.len, t.count, &ed, ao, why, wn);
        } else if (ot == 0xA5 || ot == 0xA6) {
            snprintf(why, wn, "an MP4 track of AC-3 or E-AC-3, not read");
        } else {
            snprintf(why, wn,
                     "an MP4 track of a codec not read (object type 0x%02x)",
                     ot);
        }
    } else if (c == TAG(".mp3")) {
        rc = mp4_mpeg(&t, d, ao, why, wn);
    } else if (c == TAG("alac")) {
        if (!t.config) {
            snprintf(why, wn, "an MP4 track of ALAC without its configuration");
            goto end;
        }
        struct audio_edit ed = {t.media_time, t.edit_length,
                                t.timescale ? t.timescale : (uint32_t)t.rate,
                                t.movie_timescale};
        rc = janas_alac_stream(t.config, t.config_len, d, t.off, t.len, t.count,
                               &ed, ao, why, wn);
    } else if (c == TAG("Opus")) {
        rc = mp4_opus(&t, d, ao, why, wn);
    } else if (c == TAG("fLaC")) {
        rc = mp4_flac(&t, d, ao, why, wn);
    } else if (c == TAG("sowt") || c == TAG("twos") || c == TAG("in24") ||
               c == TAG("in32") || c == TAG("fl32") || c == TAG("fl64") ||
               c == TAG("raw ") || c == TAG("lpcm")) {
        rc = mp4_pcm(&t, d, ao, why, wn);
    } else {
        snprintf(why, wn, "an MP4 track of a codec not read (%s)", fourcc);
    }
end:
    free(t.off);
    free(t.len);
    return rc;
}
