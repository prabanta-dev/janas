/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_ogg.h - the packets of a stream, as the Ogg (audio_ogg.c), MP4
 * (audio_mp4.c) and Matroska/WebM (audio_mkv.c) containers give them to
 * the codecs that read them: Vorbis (audio_vorbis.c), Opus (audio_opus.c),
 * ALAC (audio_alac.c); and the trim a container's edit says. Inside
 * src/common.
 */
#ifndef JANAS_COMMON_AUDIO_OGG_H
#define JANAS_COMMON_AUDIO_OGG_H

#include <stddef.h>
#include <stdint.h>

#include "audio.h"

struct ogg_packets {
    int codec; /* 1 Vorbis, 2 Opus, 3 FLAC */
    uint8_t *data;
    size_t used, room;
    size_t *off, *len;
    int64_t *granule; /* of the page a packet ends on, or -1 */
    size_t count, cap;
    /* from a container that says them apart from the packets (Matroska):
       the codec's private data and the samples to drop at the start */
    int64_t pre_skip; /* -1: as the codec's header says */
};

/* a packet added, copied: 0, or -1 when out of memory */
int ogg_add_packet(struct ogg_packets *o, const uint8_t *p, size_t n,
                   int64_t granule);

/* an edit list's trim: the samples skipped at the start (media_time, in
   the track's timescale; -1: none) and the length kept (in the movie's
   timescale; 0: all) */
struct audio_edit {
    int64_t media_time;
    uint64_t length;
    uint32_t timescale, movie_timescale;
};

/* got samples at rate trimmed as e says (NULL: not at all): the first
 *skip dropped, *total kept */
void audio_edit_apply(const struct audio_edit *e, uint32_t rate, size_t *skip,
                      size_t *total);

/* the packets of the first stream holding sound: 0, or -1 with why */
int ogg_read(const uint8_t *d, size_t n, struct ogg_packets *o, char *why,
             size_t wn);
void ogg_packets_free(struct ogg_packets *o);

/* the codecs: 0, or -1 with why */
int janas_vorbis_stream(const struct ogg_packets *o, struct audio_out *ao,
                        char *why, size_t wn);
int janas_opus_stream(const struct ogg_packets *o, struct audio_out *ao,
                      char *why, size_t wn);
/* the samples at 48 kHz an Opus packet makes (its TOC and frame count) */
int janas_opus_packet_samples(const uint8_t *p, size_t n);
/* ALAC: its magic cookie (ALACSpecificConfig, with or without the frma
   and alac atoms' heads), the packets at data + off[i] */
int janas_alac_stream(const uint8_t *cookie, size_t cn, const uint8_t *data,
                      const size_t *off, const size_t *len, size_t count,
                      const struct audio_edit *edit, struct audio_out *ao,
                      char *why, size_t wn);

#endif
