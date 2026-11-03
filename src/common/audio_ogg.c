/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_ogg.c - an Ogg file (RFC 3533; audio.h): its pages, checked by
 * their CRC, joined into the packets of the first logical stream that
 * holds sound, with the granule position of the page each packet ends on;
 * the codec found by its first packet - Vorbis (audio_vorbis.c), Opus
 * (audio_opus.c), FLAC (its packets made a FLAC stream again for
 * audio_flac.c).
 */
#include "audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_ogg.h"

static uint32_t crc_table[256];

static void crc_init(void)
{
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t r = i << 24;
        for (int k = 0; k < 8; k++)
            r = r & 0x80000000u ? r << 1 ^ 0x04C11DB7u : r << 1;
        crc_table[i] = r;
    }
}

static uint32_t page_crc(const uint8_t *p, size_t n)
{
    uint32_t c = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t b = i >= 22 && i < 26 ? 0 : p[i]; /* its own field as 0 */
        c = c << 8 ^ crc_table[(c >> 24) ^ b];
    }
    return c;
}

static uint32_t le32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
           (uint32_t)p[3] << 24;
}

void ogg_packets_free(struct ogg_packets *o)
{
    free(o->data);
    free(o->off);
    free(o->len);
    free(o->granule);
    memset(o, 0, sizeof *o);
}

int ogg_add_packet(struct ogg_packets *o, const uint8_t *p, size_t n,
                   int64_t granule)
{
    if (o->count == o->cap) {
        size_t c = o->cap ? o->cap * 2 : 256;
        size_t *off = realloc(o->off, sizeof(size_t) * c);
        if (off)
            o->off = off;
        size_t *len = realloc(o->len, sizeof(size_t) * c);
        if (len)
            o->len = len;
        int64_t *g = realloc(o->granule, sizeof(int64_t) * c);
        if (g)
            o->granule = g;
        if (!off || !len || !g)
            return -1;
        o->cap = c;
    }
    if (o->used + n > o->room) {
        size_t r = o->room ? o->room * 2 : 1 << 16;
        while (r < o->used + n)
            r *= 2;
        uint8_t *d = realloc(o->data, r);
        if (!d)
            return -1;
        o->data = d;
        o->room = r;
    }
    memcpy(o->data + o->used, p, n);
    o->off[o->count] = o->used;
    o->len[o->count] = n;
    o->granule[o->count] = granule;
    o->used += n;
    o->count++;
    return 0;
}

/* the codec of a stream's first packet: 1 Vorbis, 2 Opus, 3 FLAC, 0 not
   sound this reads */
static int codec_of(const uint8_t *p, size_t n)
{
    if (n >= 7 && p[0] == 1 && !memcmp(p + 1, "vorbis", 6))
        return 1;
    if (n >= 8 && !memcmp(p, "OpusHead", 8))
        return 2;
    if (n >= 13 && p[0] == 0x7F && !memcmp(p + 1, "FLAC", 4))
        return 3;
    return 0;
}

int ogg_read(const uint8_t *d, size_t n, struct ogg_packets *o, char *why,
             size_t wn)
{
    memset(o, 0, sizeof *o);
    o->pre_skip = -1;
    if (!crc_table[1])
        crc_init();
    uint32_t serial = 0;
    int chosen = 0;
    /* a packet across pages: its parts so far */
    uint8_t *part = NULL;
    size_t part_n = 0, part_cap = 0;
    for (size_t at = 0; at + 27 <= n;) {
        const uint8_t *pg = d + at;
        if (memcmp(pg, "OggS", 4) || pg[4] != 0) {
            at++; /* lost: the next capture pattern */
            continue;
        }
        size_t segs = pg[26], body = 0;
        if (at + 27 + segs > n)
            break;
        for (size_t i = 0; i < segs; i++)
            body += pg[27 + i];
        size_t size = 27 + segs + body;
        if (at + size > n || page_crc(pg, size) != le32(pg + 22)) {
            at++;
            continue;
        }
        uint32_t ser = le32(pg + 14);
        int64_t granule =
            (int64_t)((uint64_t)le32(pg + 6) | (uint64_t)le32(pg + 10) << 32);
        const uint8_t *seg = pg + 27, *data = pg + 27 + segs;
        if (!chosen && (pg[5] & 2)) { /* a stream begins: is it sound? */
            int c = segs && seg[0] < 255 ? codec_of(data, seg[0]) : 0;
            if (c) {
                chosen = c;
                serial = ser;
            }
        }
        if (chosen && ser == serial) {
            if (!(pg[5] & 1))
                part_n = 0; /* not a continuation: nothing pending */
            /* the packets ending on this page; the last ends here only
               when its last segment is shorter than 255 */
            int last_end = -1;
            for (size_t i = 0; i < segs; i++)
                if (seg[i] < 255)
                    last_end = (int)i;
            size_t pos = 0;
            for (size_t i = 0; i < segs; i++) {
                size_t l = seg[i];
                if (part_n + l > part_cap) {
                    size_t c = part_cap ? part_cap * 2 : 4096;
                    while (c < part_n + l)
                        c *= 2;
                    uint8_t *q = realloc(part, c);
                    if (!q) {
                        free(part);
                        ogg_packets_free(o);
                        snprintf(why, wn, "out of memory");
                        return -1;
                    }
                    part = q;
                    part_cap = c;
                }
                memcpy(part + part_n, data + pos, l);
                part_n += l;
                pos += l;
                if (l < 255) {
                    if (ogg_add_packet(o, part, part_n,
                                       (int)i == last_end ? granule : -1)) {
                        free(part);
                        ogg_packets_free(o);
                        snprintf(why, wn, "out of memory");
                        return -1;
                    }
                    part_n = 0;
                }
            }
            if (pg[5] & 4) /* the stream's last page */
                break;
        }
        at += size;
    }
    free(part);
    if (!chosen || !o->count) {
        ogg_packets_free(o);
        snprintf(why, wn, "an Ogg file without Vorbis, Opus or FLAC sound");
        return -1;
    }
    o->codec = chosen;
    return 0;
}

/* Ogg FLAC: 0x7F "FLAC", the mapping's version and header count, then
   "fLaC" and STREAMINFO; the other packets metadata blocks then frames */
static int ogg_flac(const struct ogg_packets *o, struct audio_out *ao,
                    char *why, size_t wn)
{
    size_t total = 0;
    for (size_t i = 0; i < o->count; i++)
        total += o->len[i];
    uint8_t *s = malloc(total + 4);
    if (!s) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    size_t n = 0;
    const uint8_t *first = o->data + o->off[0];
    if (o->len[0] < 51) {
        free(s);
        snprintf(why, wn, "an Ogg FLAC stream without its STREAMINFO");
        return -1;
    }
    memcpy(s, first + 9, o->len[0] - 9); /* "fLaC" and STREAMINFO */
    n = o->len[0] - 9;
    for (size_t i = 1; i < o->count; i++) {
        memcpy(s + n, o->data + o->off[i], o->len[i]);
        n += o->len[i];
    }
    /* the metadata blocks keep their own last-block flags; the first is
       marked last when no other follows it in the header packets */
    int rc = janas_audio_flac(s, n, ao, why, wn);
    free(s);
    return rc;
}

int janas_audio_ogg(const uint8_t *d, size_t n, struct audio_out *ao,
                    char *why, size_t wn)
{
    if (n < 4 || memcmp(d, "OggS", 4))
        return 1;
    struct ogg_packets o;
    if (ogg_read(d, n, &o, why, wn))
        return -1;
    int rc = o.codec == 1   ? janas_vorbis_stream(&o, ao, why, wn)
             : o.codec == 2 ? janas_opus_stream(&o, ao, why, wn)
                            : ogg_flac(&o, ao, why, wn);
    ogg_packets_free(&o);
    return rc;
}
