/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_stream.c - a sound read as a stream (audio.h), for recordings of
 * any length: memory for a few seconds of it, whatever it lasts.
 *
 * A scan first decodes the file and keeps nothing but the count and the
 * trim (janas_audio_scan); a stream then decodes it again on a thread of
 * its own, drops what the trim drops, resamples as janas_audio_resample
 * does - the same sums in the same order, the input's length being known -
 * and hands the samples out through a bounded buffer as they are read.
 * Read again, it is opened again: decoding costs little beside what reads
 * it (a speech model).
 */
#include "audio.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the scan: a sink that counts */
static int count_put(void *user, const float *x, size_t n)
{
    (void)user;
    (void)x;
    (void)n;
    return 0;
}

int janas_audio_scan(const void *data, size_t n, struct janas_audio_info *info,
                     char *why, size_t wn)
{
    struct janas_audio_sink s = {.put = count_put};
    return janas_audio_decode_sink(data, n, &s, info, why, wn);
}

uint64_t janas_audio_stream_length(const struct janas_audio_info *info,
                                   uint32_t rate)
{
    return audio_resampled_length(info->total - info->skip, info->rate, rate);
}

enum { RING = 1 << 16, HIST = 1 << 15 };

struct janas_audio_stream {
    const uint8_t *data;
    size_t n;
    struct janas_audio_info info;
    struct audio_rs rs;
    uint64_t out_n; /* samples out in all */
    /* the decoder's thread: the samples given, the input kept for the
       resampler (from in0), the next out */
    uint64_t raw, in0, next;
    float *in;
    size_t in_n, in_cap;
    float *rbuf; /* the samples out it makes, before the ring */
    pthread_t thread;
    int started;
    /* the ring, under mu */
    pthread_mutex_t mu;
    pthread_cond_t cv;
    float *ring;
    size_t head, count; /* read from head, count of them */
    int done, failed, cancel;
    char why[256];
};

/* the decoder's thread: samples out into the ring, waiting for room */
static int ring_push(struct janas_audio_stream *s, const float *x, size_t n)
{
    while (n) {
        pthread_mutex_lock(&s->mu);
        while (s->count == RING && !s->cancel)
            pthread_cond_wait(&s->cv, &s->mu);
        if (s->cancel) {
            pthread_mutex_unlock(&s->mu);
            return -1;
        }
        size_t room = RING - s->count, k = n < room ? n : room;
        for (size_t i = 0; i < k; i++)
            s->ring[(s->head + s->count + i) % RING] = x[i];
        s->count += k;
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mu);
        x += k;
        n -= k;
    }
    return 0;
}

/* the outputs whose input is all in (or past the input's end) */
static int resample_ready(struct janas_audio_stream *s, int all)
{
    uint64_t have = s->in0 + s->in_n, N = s->info.total - s->info.skip;
    size_t k = 0;
    while (s->next < s->out_n) {
        int64_t first = audio_rs_first(&s->rs, s->next);
        uint64_t last = (uint64_t)(first + s->rs.taps); /* one past */
        if (!all && last > have && have < N)
            break;
        s->rbuf[k++] = audio_rs_out(&s->rs, s->next, s->in, (int64_t)s->in0);
        s->next++;
        if (k == 4096) {
            if (ring_push(s, s->rbuf, k) != 0)
                return -1;
            k = 0;
        }
    }
    if (k && ring_push(s, s->rbuf, k) != 0)
        return -1;
    /* what no output to come reads */
    if (s->next < s->out_n) {
        int64_t first = audio_rs_first(&s->rs, s->next);
        if (first > (int64_t)s->in0) {
            size_t drop = (size_t)(first - (int64_t)s->in0);
            if (drop > s->in_n)
                drop = s->in_n;
            memmove(s->in, s->in + drop, sizeof(float) * (s->in_n - drop));
            s->in_n -= drop;
            s->in0 += drop;
        }
    }
    return 0;
}

static int stream_put(void *user, const float *x, size_t n)
{
    struct janas_audio_stream *s = user;
    const struct janas_audio_info *f = &s->info;
    /* the trim: [skip, total) of the samples given */
    uint64_t r0 = s->raw, r1 = s->raw + n;
    s->raw = r1;
    uint64_t a = r0 > f->skip ? r0 : f->skip, b = r1 < f->total ? r1 : f->total;
    if (a < b) {
        const float *p = x + (a - r0);
        size_t k = (size_t)(b - a);
        if (!s->rs.taps) { /* the same rate */
            if (ring_push(s, p, k) != 0)
                return -1;
        } else {
            if (s->in_n + k > s->in_cap) {
                size_t c = s->in_cap ? s->in_cap : HIST;
                while (c < s->in_n + k)
                    c *= 2;
                float *q = realloc(s->in, sizeof(float) * c);
                if (!q)
                    return -1;
                s->in = q;
                s->in_cap = c;
            }
            memcpy(s->in + s->in_n, p, sizeof(float) * k);
            s->in_n += k;
            if (resample_ready(s, 0) != 0)
                return -1;
        }
    }
    return r1 >= f->total ? 1 : 0; /* the sound is all out: stop */
}

static void *stream_main(void *arg)
{
    struct janas_audio_stream *s = arg;
    struct janas_audio_sink sink = {.put = stream_put, .user = s};
    struct janas_audio_info info;
    char why[256] = "";
    int rc = s->info.total > s->info.skip
                 ? janas_audio_decode_sink(s->data, s->n, &sink, &info, why,
                                           sizeof why)
                 : 0;
    int ok = s->raw >= s->info.total || rc == 0;
    if (ok && s->rs.taps && resample_ready(s, 1) != 0)
        ok = 0;
    pthread_mutex_lock(&s->mu);
    s->done = 1;
    if (!ok && !s->cancel) {
        s->failed = 1;
        snprintf(s->why, sizeof s->why, "%s", why[0] ? why : "out of memory");
    }
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return NULL;
}

struct janas_audio_stream *
janas_audio_stream_open(const void *data, size_t n,
                        const struct janas_audio_info *info, uint32_t rate)
{
    struct janas_audio_stream *s = calloc(1, sizeof *s);
    if (!s)
        return NULL;
    s->data = data;
    s->n = n;
    s->info = *info;
    s->out_n = janas_audio_stream_length(info, rate);
    s->ring = malloc(sizeof(float) * RING);
    s->rbuf = malloc(sizeof(float) * 4096);
    if (!s->ring || !s->rbuf ||
        audio_rs_init(&s->rs, info->rate, rate, info->total - info->skip)) {
        free(s->ring);
        free(s->rbuf);
        free(s);
        return NULL;
    }
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv, NULL);
    if (pthread_create(&s->thread, NULL, stream_main, s) != 0) {
        janas_audio_stream_close(s);
        return NULL;
    }
    s->started = 1;
    return s;
}

long janas_audio_stream_read(struct janas_audio_stream *s, float *out,
                             size_t max, char *why, size_t wn)
{
    pthread_mutex_lock(&s->mu);
    while (!s->count && !s->done)
        pthread_cond_wait(&s->cv, &s->mu);
    if (!s->count && s->failed) {
        snprintf(why, wn, "%s", s->why);
        pthread_mutex_unlock(&s->mu);
        return -1;
    }
    size_t k = s->count < max ? s->count : max;
    for (size_t i = 0; i < k; i++)
        out[i] = s->ring[(s->head + i) % RING];
    s->head = (s->head + k) % RING;
    s->count -= k;
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return (long)k;
}

void janas_audio_stream_close(struct janas_audio_stream *s)
{
    if (!s)
        return;
    if (s->started) {
        pthread_mutex_lock(&s->mu);
        s->cancel = 1;
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mu);
        pthread_join(s->thread, NULL);
    }
    pthread_cond_destroy(&s->cv);
    pthread_mutex_destroy(&s->mu);
    audio_rs_free(&s->rs);
    free(s->in);
    free(s->ring);
    free(s->rbuf);
    free(s);
}
