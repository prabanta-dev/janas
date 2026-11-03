/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio.c - a sound file recognized by its first bytes and decoded
 * (audio.h); resampling.
 */
#include "audio.h"
#include "audio_ogg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int audio_put(struct audio_out *o, const float *x, size_t n, char *why,
              size_t wn)
{
    if (!o->told) {
        if (o->sink->rate && o->sink->rate(o->sink->user, o->rate) != 0) {
            snprintf(why, wn, "stopped");
            return -1;
        }
        o->told = 1;
    }
    if (!n)
        return 0;
    if (o->sink->put(o->sink->user, x, n) != 0) {
        snprintf(why, wn, "stopped");
        return -1;
    }
    o->got += n;
    return 0;
}

uint64_t audio_out_kept(const struct audio_out *o)
{
    uint64_t t = o->total < o->got ? o->total : o->got;
    return t > o->skip ? t - o->skip : 0;
}

void audio_out_trim(struct audio_out *o, uint64_t skip, uint64_t keep)
{
    uint64_t kept = audio_out_kept(o);
    if (skip > kept)
        skip = kept;
    if (keep > kept - skip)
        keep = kept - skip;
    o->skip += skip;
    o->total = o->skip + keep;
}

int janas_audio_decode_sink(const void *data, size_t n,
                            const struct janas_audio_sink *sink,
                            struct janas_audio_info *info, char *why,
                            size_t wn)
{
    memset(info, 0, sizeof *info);
    const uint8_t *d = data;
    /* an ID3v2 tag before the sound (MP3's habit, FLAC's sometimes) */
    while (n >= 10 && !memcmp(d, "ID3", 3) && d[3] < 0xFF && d[4] < 0xFF &&
           !((d[6] | d[7] | d[8] | d[9]) & 0x80)) {
        size_t len = 10 +
                     ((size_t)d[6] << 21 | (size_t)d[7] << 14 |
                      (size_t)d[8] << 7 | d[9]) +
                     (d[5] & 0x10 ? 10 : 0); /* a footer */
        if (len > n)
            break;
        d += len;
        n -= len;
    }
    static int (*const decoders[])(const uint8_t *, size_t,
                                   struct audio_out *, char *, size_t) = {
        janas_audio_wav,  janas_audio_flac, janas_audio_ogg,
        janas_audio_mp4,  janas_audio_adts, janas_audio_mkv,
        janas_audio_mpeg};
    for (size_t i = 0; i < sizeof decoders / sizeof *decoders; i++) {
        struct audio_out o = {.sink = sink, .total = UINT64_MAX};
        int rc = decoders[i](d, n, &o, why, wn);
        if (rc > 0)
            continue;
        if (rc < 0)
            return -1;
        if (audio_put(&o, NULL, 0, why, wn) != 0) /* no samples: the rate */
            return -1;
        info->rate = o.rate;
        info->got = o.got;
        info->skip = o.skip < o.got ? o.skip : o.got;
        info->total = o.total < o.got ? o.total : o.got;
        if (info->total < info->skip)
            info->total = info->skip;
        return 0;
    }
    snprintf(why, wn,
             "not a sound file this reads (WAV, FLAC, MP3 and MPEG audio, "
             "Ogg, MP4/M4A, AAC, WebM/Matroska)");
    return -1;
}

/* janas_audio_decode's sink: the samples kept in memory, up to its bound */
struct gather {
    float *pcm;
    size_t n, cap, most;
    int long_;
};

static int gather_put(void *user, const float *x, size_t n)
{
    struct gather *g = user;
    if (n > g->most - g->n) {
        g->long_ = 1;
        return -1;
    }
    if (g->n + n > g->cap) {
        size_t c = g->cap ? g->cap * 2 : (size_t)1 << 18;
        while (c < g->n + n)
            c *= 2;
        float *p = realloc(g->pcm, sizeof(float) * c);
        if (!p)
            return -1;
        g->pcm = p;
        g->cap = c;
    }
    memcpy(g->pcm + g->n, x, sizeof(float) * n);
    g->n += n;
    return 0;
}

int janas_audio_decode_max(const void *data, size_t n, struct janas_audio *a,
                           size_t most, char *why, size_t wn)
{
    memset(a, 0, sizeof *a);
    struct gather g = {.most = most};
    struct janas_audio_sink s = {.put = gather_put, .user = &g};
    struct janas_audio_info info;
    if (janas_audio_decode_sink(data, n, &s, &info, why, wn) != 0) {
        free(g.pcm);
        if (g.long_)
            return 1;
        if (!strcmp(why, "stopped"))
            snprintf(why, wn, "out of memory");
        return -1;
    }
    size_t keep = (size_t)(info.total - info.skip);
    if (keep)
        memmove(g.pcm, g.pcm + info.skip, sizeof(float) * keep);
    a->pcm = g.pcm;
    a->n = keep;
    a->rate = info.rate;
    return 0;
}

int janas_audio_decode(const void *data, size_t n, struct janas_audio *a,
                       char *why, size_t wn)
{
    int rc = janas_audio_decode_max(data, n, a, JANAS_AUDIO_MAX_SAMPLES, why,
                                    wn);
    if (rc > 0)
        snprintf(why, wn, "a sound longer than this reads in memory");
    return rc ? -1 : 0;
}

void audio_edit_apply(const struct audio_edit *e, uint32_t rate, size_t *skip,
                      size_t *total)
{
    size_t got = *total;
    *skip = 0;
    if (!e || e->media_time < 0 || !e->timescale)
        return;
    uint64_t s = (uint64_t)e->media_time * rate / e->timescale;
    *skip = s < got ? (size_t)s : got;
    *total = got - *skip;
    if (e->length && e->movie_timescale) {
        uint64_t k = e->length * rate / e->movie_timescale;
        if (k < *total)
            *total = (size_t)k;
    }
}

void janas_audio_free(struct janas_audio *a)
{
    free(a->pcm);
    memset(a, 0, sizeof *a);
}

static uint32_t gcd(uint32_t a, uint32_t b)
{
    while (b) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* the zeroth-order modified Bessel function, for the Kaiser window */
static double bessel_i0(double x)
{
    double s = 1, t = 1;
    for (int k = 1; k < 50; k++) {
        t *= (x / (2 * k)) * (x / (2 * k));
        s += t;
        if (t < 1e-12 * s)
            break;
    }
    return s;
}

#define ZEROS 32     /* zero crossings of the sinc on either side */
#define BETA 8.6     /* the Kaiser window's: about 90 dB of rejection */
#define ROLLOFF 0.95 /* the cut-off, of the lower Nyquist frequency */

/* the filter at distance x (in input samples), cut-off fc (cycles a
   sample), half-width hw samples */
static double kernel(double x, double fc, double hw)
{
    if (fabs(x) >= hw)
        return 0;
    double s = x == 0 ? 1 : sin(M_PI * 2 * fc * x) / (M_PI * 2 * fc * x);
    double r = x / hw;
    return 2 * fc * s * bessel_i0(BETA * sqrt(1 - r * r)) / bessel_i0(BETA);
}

uint64_t audio_resampled_length(uint64_t n, uint32_t from, uint32_t to)
{
    if (from == to)
        return n;
    double ratio = (double)to / from;
    return (uint64_t)floor((size_t)n * ratio);
}

int audio_rs_init(struct audio_rs *r, uint32_t from, uint32_t to, uint64_t n)
{
    memset(r, 0, sizeof *r);
    r->n = n;
    if (from == to)
        return 0;
    double ratio = (double)to / from;
    r->fc = 0.5 * ROLLOFF * (ratio < 1 ? ratio : 1);
    r->hw = ZEROS / (2 * r->fc); /* in input samples */
    /* output i sits at input i * in / out: with in / g and out / g the
       rates' reduced ratio the phases repeat every L = out / g outputs */
    uint32_t g = gcd(from, to);
    r->L = to / g;
    r->M = from / g;
    r->taps = (int)ceil(2 * r->hw) + 1;
    if ((size_t)r->L * (size_t)r->taps <= ((size_t)1 << 22))
        r->table = malloc(sizeof(float) * r->L * (size_t)r->taps);
    if (r->table)
        for (uint32_t p = 0; p < r->L; p++) {
            double frac = (double)((uint64_t)p * r->M % r->L) / r->L;
            for (int t = 0; t < r->taps; t++) {
                long k = t - r->taps / 2;
                r->table[(size_t)p * r->taps + t] =
                    (float)kernel((double)k - frac, r->fc, r->hw);
            }
        }
    return 0;
}

void audio_rs_free(struct audio_rs *r)
{
    free(r->table);
    r->table = NULL;
}

int64_t audio_rs_first(const struct audio_rs *r, uint64_t i)
{
    return (int64_t)((uint64_t)i * r->M / r->L) - r->taps / 2;
}

float audio_rs_out(const struct audio_rs *r, uint64_t i, const float *x,
                   int64_t x0)
{
    uint64_t num = (uint64_t)i * r->M;
    long base = (long)(num / r->L);
    uint32_t p = (uint32_t)(i % r->L);
    double frac = (double)(num % r->L) / r->L, s = 0;
    for (int t = 0; t < r->taps; t++) {
        long k = base + t - r->taps / 2;
        if (k < 0 || (uint64_t)k >= r->n)
            continue;
        double h = r->table
                       ? r->table[(size_t)p * r->taps + t]
                       : kernel((double)(t - r->taps / 2) - frac, r->fc, r->hw);
        s += h * x[k - x0];
    }
    return (float)s;
}

int janas_audio_resample(struct janas_audio *a, uint32_t rate)
{
    if (a->rate == rate || !a->n)
        return a->rate = rate, 0;
    size_t out_n = (size_t)audio_resampled_length(a->n, a->rate, rate);
    float *out = malloc(sizeof(float) * (out_n ? out_n : 1));
    struct audio_rs r;
    if (!out || audio_rs_init(&r, a->rate, rate, a->n) != 0) {
        free(out);
        return -1;
    }
    for (size_t i = 0; i < out_n; i++)
        out[i] = audio_rs_out(&r, i, a->pcm, 0);
    audio_rs_free(&r);
    free(a->pcm);
    a->pcm = out;
    a->n = out_n;
    a->rate = rate;
    return 0;
}
