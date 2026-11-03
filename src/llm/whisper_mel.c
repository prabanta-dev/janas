/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * whisper_mel.c - Whisper's log-mel spectrogram (whisper.h), as OpenAI's
 * log_mel_spectrogram computes it: the sound followed by zeros, reflected
 * at both ends by half a window (torch.stft's center), a periodic Hann
 * window of n_fft samples every hop, the power of each frequency, the last
 * frame dropped; the model's mel filters over the powers, log10 (at least
 * 1e-10), nothing below the largest less 8, then (x + 4) / 4.
 *
 * A long recording's comes a window at a time (janas_whisper_mel_window),
 * from the samples about it, and the largest from a pass over all of it
 * (janas_whisper_mel_max): the same numbers as the whole one's.
 */
#include "whisper.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/pool.h"

struct mel_job {
    const struct janas_whisper *w;
    const float *pcm; /* the samples from p0 on (those the frames read) */
    size_t p0;
    size_t n, total; /* samples, and with the zeros after them */
    float *out;      /* frame f0 + j at column j, rows of stride */
    uint32_t f0, frames, stride;
    float *maxes; /* each thread's largest */
};

/* sample i of the sound padded with zeros, reflected past either end */
static float at(const struct mel_job *m, long i)
{
    long t = (long)m->total;
    if (i < 0)
        i = -i;
    if (i >= t)
        i = 2 * (t - 1) - i;
    return i >= 0 && (size_t)i < m->n ? m->pcm[(size_t)i - m->p0] : 0.0f;
}

static void mel_worker(void *arg, int tid, int nt)
{
    struct mel_job *m = arg;
    const struct janas_whisper *w = m->w;
    uint32_t N = w->n_fft, bins = N / 2 + 1, half = N / 2;
    float *frame = malloc(sizeof(float) * (N + bins));
    float mx = -INFINITY;
    if (!frame) {
        m->maxes[tid] = NAN; /* told to the caller */
        return;
    }
    float *pw = frame + N;
    uint32_t per = (m->frames + (uint32_t)nt - 1) / (uint32_t)nt;
    uint32_t f0 = (uint32_t)tid * per, f1 = f0 + per;
    if (f1 > m->frames)
        f1 = m->frames;
    for (uint32_t f = f0; f < f1; f++) {
        long s0 = (long)(m->f0 + f) * w->hop - (long)half;
        if (s0 >= 0 && (size_t)s0 + N <= m->n)
            memcpy(frame, m->pcm + ((size_t)s0 - m->p0), sizeof(float) * N);
        else
            for (uint32_t i = 0; i < N; i++)
                frame[i] = at(m, s0 + (long)i);
        for (uint32_t k = 0; k < bins; k++) {
            const float *c = w->dft_cos + (size_t)k * N;
            const float *s = w->dft_sin + (size_t)k * N;
            float re = 0, im = 0;
            for (uint32_t i = 0; i < N; i++) {
                re += frame[i] * c[i];
                im += frame[i] * s[i];
            }
            pw[k] = re * re + im * im;
        }
        for (uint32_t b = 0; b < w->n_mels; b++) {
            const float *fl = w->mel_filters + (size_t)b * bins;
            double e = 0;
            for (uint32_t k = 0; k < bins; k++)
                e += (double)fl[k] * pw[k];
            float v = (float)log10(e > 1e-10 ? e : 1e-10);
            m->out[(size_t)b * m->stride + f] = v;
            if (v > mx)
                mx = v;
        }
    }
    m->maxes[tid] = mx;
    free(frame);
}

/* frames f0 .. f0 + count of the sound of n samples followed by pad
   zeros, their log10 into out (rows of stride), from the samples from p0
   at pcm (those they read: [f0 hop - n_fft / 2, (f0 + count) hop +
   n_fft / 2) within the sound); their largest in *mx. 0, or -1 when out
   of memory */
static int mel_range(struct janas_whisper *w, const float *pcm, size_t p0,
                     size_t n, size_t pad, uint32_t f0, uint32_t count,
                     float *out, uint32_t stride, float *mx)
{
    int nt = janas_pool_size(w->pool);
    float *maxes = malloc(sizeof(float) * (size_t)nt);
    if (!maxes)
        return -1;
    struct mel_job m = {.w = w,
                        .pcm = pcm,
                        .p0 = p0,
                        .n = n,
                        .total = n + pad,
                        .out = out,
                        .f0 = f0,
                        .frames = count,
                        .stride = stride,
                        .maxes = maxes};
    janas_pool_run(w->pool, mel_worker, &m);
    *mx = -INFINITY;
    for (int t = 0; t < nt; t++) {
        if (isnan(maxes[t])) {
            free(maxes);
            return -1;
        }
        if (maxes[t] > *mx)
            *mx = maxes[t];
    }
    free(maxes);
    return 0;
}

/* the log10s made the model's input: nothing below the largest less 8 */
static void mel_norm(float *x, size_t n, float mx)
{
    for (size_t i = 0; i < n; i++) {
        float v = x[i] < mx - 8.0f ? mx - 8.0f : x[i];
        x[i] = (v + 4.0f) / 4.0f;
    }
}

float *janas_whisper_mel(struct janas_whisper *w, const float *pcm, size_t n,
                         size_t pad, uint32_t *frames)
{
    size_t total = n + pad, nf = total / w->hop;
    *frames = 0;
    if (nf > UINT32_MAX / 2)
        return NULL;
    float *out = malloc(sizeof(float) * (nf ? nf : 1) * w->n_mels);
    float mx;
    if (!out || mel_range(w, pcm, 0, n, pad, 0, (uint32_t)nf, out,
                          (uint32_t)nf, &mx) != 0) {
        free(out);
        return NULL;
    }
    mel_norm(out, nf * w->n_mels, mx);
    *frames = (uint32_t)nf;
    return out;
}

/* the samples frames f0 .. f1 read, as [*from, *from + *count) within the
   sound */
static void reads(const struct janas_whisper *w, size_t n, uint32_t f0,
                  uint32_t f1, size_t *from, size_t *count)
{
    long a = (long)f0 * w->hop - (long)(w->n_fft / 2);
    size_t b = (size_t)f1 * w->hop + w->n_fft / 2;
    *from = a > 0 ? (size_t)a : 0;
    if (b > n)
        b = n;
    *count = b > *from ? b - *from : 0;
}

int janas_whisper_mel_max(struct janas_whisper *w,
                          const struct janas_whisper_sound *snd, size_t pad,
                          float *mx, char *err, size_t err_len)
{
    const uint32_t step = 3000;
    size_t nf = (snd->n + pad) / w->hop;
    float *buf = malloc(sizeof(float) * step * w->n_mels);
    *mx = -INFINITY;
    if (!buf || nf > UINT32_MAX / 2) {
        free(buf);
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    for (size_t f = 0; f < nf; f += step) {
        uint32_t k = nf - f < step ? (uint32_t)(nf - f) : step;
        size_t from, count;
        reads(w, snd->n, (uint32_t)f, (uint32_t)f + k, &from, &count);
        const float *p = count ? snd->get(snd->user, from, count, err, err_len)
                               : NULL;
        float m;
        if ((count && !p) || mel_range(w, p, from, snd->n, pad, (uint32_t)f,
                                       k, buf, k, &m) != 0) {
            if (!count || p)
                snprintf(err, err_len, "out of memory");
            free(buf);
            return -1;
        }
        if (m > *mx)
            *mx = m;
    }
    free(buf);
    return 0;
}

int janas_whisper_mel_window(struct janas_whisper *w,
                             const struct janas_whisper_sound *snd,
                             size_t pad, float mx, uint32_t f0, uint32_t count,
                             float *out, uint32_t stride, char *err,
                             size_t err_len)
{
    size_t from, len;
    reads(w, snd->n, f0, f0 + count, &from, &len);
    const float *p =
        len ? snd->get(snd->user, from, len, err, err_len) : NULL;
    float m;
    if (len && !p)
        return -1;
    if (mel_range(w, p, from, snd->n, pad, f0, count, out, stride, &m) != 0) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    for (uint32_t b = 0; b < w->n_mels; b++)
        mel_norm(out + (size_t)b * stride, count, mx);
    return 0;
}
