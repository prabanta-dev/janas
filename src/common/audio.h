/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio.h - sound read from its files into mono float samples in [-1, 1],
 * as the speech models take it: the channels averaged, then resampled to
 * the model's rate. Janas's own decoders (audio_wav.c, audio_flac.c,
 * audio_mp3.c and audio_mp2.c, audio_ogg.c with audio_vorbis.c and
 * audio_opus*.c), with bounds on sizes so a file cannot make them take
 * more memory than its sound.
 */
#ifndef JANAS_COMMON_AUDIO_H
#define JANAS_COMMON_AUDIO_H

#include <stddef.h>
#include <stdint.h>

/* janas_audio_decode's bound, the samples it keeps in memory: four hours
   at 48 kHz, after mixing (a stream has none) */
#define JANAS_AUDIO_MAX_SAMPLES ((size_t)48000 * 3600 * 4)

struct janas_audio {
    float *pcm; /* n samples, mono */
    size_t n;
    uint32_t rate; /* samples a second */
};

/* n bytes of a sound file (WAV, FLAC, MP3 and MPEG audio, Ogg with
   Vorbis, Opus or FLAC), the kind told by its bytes: 0 with *a filled (free it
   with janas_audio_free), -1 with the reason in why. */
int janas_audio_decode(const void *data, size_t n, struct janas_audio *a,
                       char *why, size_t wn);
/* the same keeping at most most samples as decoded (before the trim): 1
   when the sound has more, nothing kept */
int janas_audio_decode_max(const void *data, size_t n, struct janas_audio *a,
                           size_t most, char *why, size_t wn);
/* the sound resampled to rate, in place (a windowed sinc, its cut-off
   below the lower of the two Nyquist frequencies): 0, or -1 when out of
   memory */
int janas_audio_resample(struct janas_audio *a, uint32_t rate);
void janas_audio_free(struct janas_audio *a);

/*
 * Decoding without keeping the sound: the samples are given to a sink as
 * they are decoded, mono, in order, the rate first. Of those given, the
 * sound is those from skip to total: a codec's delay and a container's
 * edit drop some at the start, the padding of the last frame some at the
 * end, known only once the file is read.
 */
struct janas_audio_sink {
    int (*rate)(void *user, uint32_t rate); /* nonzero: stop */
    int (*put)(void *user, const float *x, size_t n); /* nonzero: stop */
    void *user;
};
struct janas_audio_info {
    uint32_t rate;
    uint64_t got, skip, total;
};
/* 0 with *info, -1 with the reason in why ("stopped" when the sink
   stopped it) */
int janas_audio_decode_sink(const void *data, size_t n,
                            const struct janas_audio_sink *s,
                            struct janas_audio_info *info, char *why,
                            size_t wn);

/*
 * audio_stream.c: a sound of any length read a piece at a time. The scan
 * decodes the file keeping nothing (0 with *info, -1 with why); a stream
 * opened with what it found gives the sound at rate, the same samples
 * janas_audio_decode and janas_audio_resample would, its length known
 * before: janas_audio_stream_length. The file's bytes must outlive it.
 */
int janas_audio_scan(const void *data, size_t n, struct janas_audio_info *info,
                     char *why, size_t wn);
uint64_t janas_audio_stream_length(const struct janas_audio_info *info,
                                   uint32_t rate);
struct janas_audio_stream;
/* NULL when out of memory or threads */
struct janas_audio_stream *
janas_audio_stream_open(const void *data, size_t n,
                        const struct janas_audio_info *info, uint32_t rate);
/* up to max samples into out: their count, 0 at the end, -1 with why */
long janas_audio_stream_read(struct janas_audio_stream *s, float *out,
                             size_t max, char *why, size_t wn);
void janas_audio_stream_close(struct janas_audio_stream *s);

/*
 * The decoders, inside src/common: each gives its samples to o (audio_put)
 * and sets o->rate before the first; o->skip and o->total, when it trims
 * (audio_out_trim). They return 1 when the bytes are not theirs, having
 * given nothing.
 */
struct audio_out {
    const struct janas_audio_sink *sink;
    uint32_t rate;
    int told;            /* the sink has the rate */
    uint64_t got;        /* samples given */
    uint64_t skip, total; /* of them, the sound's; total UINT64_MAX: got */
};
/* n samples given: 0, or -1 with why when the sink stops */
int audio_put(struct audio_out *o, const float *x, size_t n, char *why,
              size_t wn);
/* the samples kept so far (total less skip) */
uint64_t audio_out_kept(const struct audio_out *o);
/* a container's trim over what its codec kept: the first skip of those
   dropped, keep kept (fewer if there are fewer) */
void audio_out_trim(struct audio_out *o, uint64_t skip, uint64_t keep);

/* The resampler janas_audio_resample uses, for one sample out at a time:
   a windowed sinc over the input around it. n: the input's length. */
struct audio_rs {
    uint64_t n;
    double fc, hw;
    uint32_t L, M;
    int taps; /* 0: the rates are the same */
    float *table;
};
/* the samples out of n in, from one rate to another */
uint64_t audio_resampled_length(uint64_t n, uint32_t from, uint32_t to);
int audio_rs_init(struct audio_rs *r, uint32_t from, uint32_t to, uint64_t n);
void audio_rs_free(struct audio_rs *r);
/* the first input sample out i reads (it reads taps from there) */
int64_t audio_rs_first(const struct audio_rs *r, uint64_t i);
/* out i, the input from sample x0 at x */
float audio_rs_out(const struct audio_rs *r, uint64_t i, const float *x,
                   int64_t x0);

/* audio_wav.c: RIFF WAVE, PCM of 8 to 32 bits or float of 32 or 64,
   plain or extensible; 1 when the bytes are not a WAV file */
int janas_audio_wav(const uint8_t *d, size_t n, struct audio_out *o,
                    char *why, size_t wn);
/* audio_mp3.c (and audio_mp2.c): MPEG-1, 2 and 2.5 audio, Layers I, II
   and III; 1 when no frames are found in the first 64 KiB */
int janas_audio_mpeg(const uint8_t *d, size_t n, struct audio_out *o,
                     char *why, size_t wn);
/* audio_ogg.c: Ogg with Vorbis, Opus or FLAC; 1 when not Ogg */
int janas_audio_ogg(const uint8_t *d, size_t n, struct audio_out *o,
                    char *why, size_t wn);
/* audio_flac.c: FLAC, every kind RFC 9639 describes; 1 when not FLAC */
int janas_audio_flac(const uint8_t *d, size_t n, struct audio_out *o,
                     char *why, size_t wn);
/* audio_mp4.c: MP4, M4A, MOV with AAC, ALAC, MP3, Opus, FLAC or PCM; 1
   when not MP4 */
int janas_audio_mp4(const uint8_t *d, size_t n, struct audio_out *o,
                    char *why, size_t wn);
/* audio_mkv.c: Matroska and WebM with Opus, Vorbis, FLAC, AAC, MPEG
   audio, ALAC or PCM; 1 when not EBML */
int janas_audio_mkv(const uint8_t *d, size_t n, struct audio_out *o,
                    char *why, size_t wn);
/* audio_aac.c: AAC in ADTS frames (.aac); 1 when not ADTS */
int janas_audio_adts(const uint8_t *d, size_t n, struct audio_out *o,
                     char *why, size_t wn);

#endif
