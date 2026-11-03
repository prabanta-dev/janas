/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vad.c - Silero VAD v5, as its 16 kHz model computes it: each 512
 * samples with the 64 before them, reflected 64 to the right, through the
 * STFT's basis (258 filters of 256 taps every 128 samples: 129 bins'
 * magnitudes over 4 frames), four convolutions with ReLU (strides 1, 2,
 * 2, 1: one frame of 128), an LSTM cell of 128 carried from window to
 * window, ReLU and the output convolution's sigmoid: the probability of
 * speech. The stretches of speech from those probabilities as Silero's
 * get_speech_timestamps finds them (a threshold, its hysteresis 0.15
 * below, the shortest speech and silence, padding split between close
 * stretches).
 */
#include "vad.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "jns.h"

#define WIN 512
#define CTX 64
#define HID 128

struct janas_vad {
    struct janas_jns j;
    uint8_t *data;
    const float *stft, *w[4], *b[4], *lw, *lr, *lb, *ow, *ob;
};

static const int CIN[4] = {129, 128, 64, 64}, COUT[4] = {128, 64, 64, 128};
static const int STRIDE[4] = {1, 2, 2, 1};

static const float *tensor(struct janas_vad *v, const char *name,
                           uint64_t count)
{
    const struct janas_jns_tensor *t = janas_jns_tensor(&v->j, name);
    if (!t || t->type != 0 || t->bytes != 4 * count)
        return NULL;
    return (const float *)(v->data + (t->offset - v->j.h.resident_offset));
}

void janas_vad_close(struct janas_vad *v)
{
    if (!v)
        return;
    free(v->data);
    janas_jns_close(&v->j);
    free(v);
}

struct janas_vad *janas_vad_open(const char *path, char *err, size_t err_len)
{
    struct janas_vad *v = calloc(1, sizeof *v);
    if (!v) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    v->j.fd = -1;
    if (janas_jns_open(path, &v->j, err, err_len) != 0) {
        free(v);
        return NULL;
    }
    const char *why = NULL;
    v->data = malloc(v->j.h.resident_bytes ? v->j.h.resident_bytes : 1);
    size_t got = 0;
    while (v->data && got < v->j.h.resident_bytes) {
        ssize_t k = pread(v->j.fd, v->data + got, v->j.h.resident_bytes - got,
                          (off_t)(v->j.h.resident_offset + got));
        if (k <= 0)
            break;
        got += (size_t)k;
    }
    if (!v->data || got != v->j.h.resident_bytes)
        why = "cannot read the weights";
    char name[32];
    if (!why) {
        v->stft = tensor(v, "vad.stft", 258 * 256);
        for (int i = 0; i < 4 && !why; i++) {
            snprintf(name, sizeof name, "vad.enc%d.w", i);
            v->w[i] = tensor(v, name, (uint64_t)COUT[i] * CIN[i] * 3);
            snprintf(name, sizeof name, "vad.enc%d.b", i);
            v->b[i] = tensor(v, name, (uint64_t)COUT[i]);
            if (!v->w[i] || !v->b[i])
                why = "not a Silero VAD model (a convolution is missing)";
        }
        v->lw = tensor(v, "vad.lstm.w", 4 * HID * HID);
        v->lr = tensor(v, "vad.lstm.r", 4 * HID * HID);
        v->lb = tensor(v, "vad.lstm.b", 8 * HID);
        v->ow = tensor(v, "vad.out.w", HID);
        v->ob = tensor(v, "vad.out.b", 1);
        if (!why &&
            (!v->stft || !v->lw || !v->lr || !v->lb || !v->ow || !v->ob))
            why = "not a Silero VAD model (made by silero2jns)";
    }
    if (why) {
        snprintf(err, err_len, "%s: %s", path, why);
        janas_vad_close(v);
        return NULL;
    }
    return v;
}

uint32_t janas_vad_window(const struct janas_vad *v)
{
    (void)v;
    return WIN;
}

uint32_t janas_vad_rate(const struct janas_vad *v)
{
    (void)v;
    return 16000;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* one window: x the 576 samples (context and window), h and c the LSTM's
   state; the probability of speech */
static float step(const struct janas_vad *v, const float *x, float *h, float *c)
{
    float pad[CTX + WIN + 64];
    memcpy(pad, x, sizeof(float) * (CTX + WIN));
    for (int i = 0; i < 64; i++) /* reflected: the last sample not repeated */
        pad[CTX + WIN + i] = x[CTX + WIN - 2 - i];
    /* the STFT: 4 frames of 129 magnitudes */
    float a[129 * 4], b[128 * 4];
    for (int t = 0; t < 4; t++) {
        float re[258];
        for (int f = 0; f < 258; f++) {
            const float *basis = v->stft + (size_t)f * 256;
            const float *s = pad + t * 128;
            float sum = 0;
            for (int k = 0; k < 256; k++)
                sum += basis[k] * s[k];
            re[f] = sum;
        }
        for (int f = 0; f < 129; f++)
            a[f * 4 + t] = sqrtf(re[f] * re[f] + re[129 + f] * re[129 + f]);
    }
    /* the encoder: [channels][frames] */
    float *in = a, *out = b, tmp[128 * 4];
    int len = 4;
    for (int l = 0; l < 4; l++) {
        int ol = (len + 2 - 3) / STRIDE[l] + 1;
        for (int o = 0; o < COUT[l]; o++)
            for (int t = 0; t < ol; t++) {
                float sum = v->b[l][o];
                for (int i = 0; i < CIN[l]; i++) {
                    const float *w = v->w[l] + ((size_t)o * CIN[l] + i) * 3;
                    for (int k = 0; k < 3; k++) {
                        int p = t * STRIDE[l] + k - 1;
                        if (p >= 0 && p < len)
                            sum += w[k] * in[i * len + p];
                    }
                }
                out[o * ol + t] = sum > 0 ? sum : 0;
            }
        len = ol;
        in = out;
        out = in == b ? tmp : b;
    }
    /* the LSTM (ONNX's gates: input, output, forget, cell) */
    float g[4 * HID];
    for (int r = 0; r < 4 * HID; r++) {
        float sum = v->lb[r] + v->lb[4 * HID + r];
        const float *wr = v->lw + (size_t)r * HID,
                    *rr = v->lr + (size_t)r * HID;
        for (int k = 0; k < HID; k++)
            sum += wr[k] * in[k * len] + rr[k] * h[k];
        g[r] = sum;
    }
    float y = v->ob[0];
    for (int k = 0; k < HID; k++) {
        float ig = sigmoid(g[k]), og = sigmoid(g[HID + k]);
        float fg = sigmoid(g[2 * HID + k]), cg = tanhf(g[3 * HID + k]);
        c[k] = fg * c[k] + ig * cg;
        h[k] = og * tanhf(c[k]);
        y += v->ow[k] * (h[k] > 0 ? h[k] : 0);
    }
    return sigmoid(y);
}

long janas_vad_probs(struct janas_vad *v, const float *pcm, size_t n,
                     float **probs)
{
    size_t nw = (n + WIN - 1) / WIN;
    float *p = malloc(sizeof(float) * (nw ? nw : 1));
    if (!p)
        return -1;
    float x[CTX + WIN], h[HID] = {0}, c[HID] = {0};
    memset(x, 0, sizeof(float) * CTX);
    for (size_t w = 0; w < nw; w++) {
        size_t at = w * WIN, k = n - at < WIN ? n - at : WIN;
        memcpy(x + CTX, pcm + at, sizeof(float) * k);
        memset(x + CTX + k, 0, sizeof(float) * (WIN - k));
        p[w] = step(v, x, h, c);
        memmove(x, x + WIN, sizeof(float) * CTX); /* the next one's context */
    }
    *probs = p;
    return (long)nw;
}

long janas_vad_probs_read(struct janas_vad *v, size_t n, float gain,
                          const float *(*get)(void *user, size_t from,
                                              size_t count, char *err,
                                              size_t err_len),
                          void *user, float **probs, char *err,
                          size_t err_len)
{
    enum { PIECE = 256 * WIN };
    size_t nw = (n + WIN - 1) / WIN;
    float *p = malloc(sizeof(float) * (nw ? nw : 1));
    float *g = malloc(sizeof(float) * PIECE);
    if (!p || !g) {
        free(p);
        free(g);
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    float x[CTX + WIN], h[HID] = {0}, c[HID] = {0};
    memset(x, 0, sizeof(float) * CTX);
    for (size_t at0 = 0, w = 0; at0 < n; at0 += PIECE) {
        size_t len = n - at0 < PIECE ? n - at0 : PIECE;
        const float *src = get(user, at0, len, err, err_len);
        if (!src) {
            free(p);
            free(g);
            return -1;
        }
        for (size_t i = 0; i < len; i++)
            g[i] = src[i] * gain; /* as the whole sound normalized is */
        for (size_t at = 0; at < len; at += WIN, w++) {
            size_t k = len - at < WIN ? len - at : WIN;
            memcpy(x + CTX, g + at, sizeof(float) * k);
            memset(x + CTX + k, 0, sizeof(float) * (WIN - k));
            p[w] = step(v, x, h, c);
            memmove(x, x + WIN, sizeof(float) * CTX);
        }
    }
    free(g);
    *probs = p;
    return (long)nw;
}

long janas_vad_segments(const struct janas_vad *v, const float *probs,
                        long n_probs, size_t n,
                        const struct janas_vad_params *p, size_t **seg)
{
    (void)v;
    const double sr = 16000;
    double threshold = p->threshold;
    double neg = threshold - 0.15 > 0.01 ? threshold - 0.15 : 0.01;
    double min_speech = sr * p->min_speech_ms / 1000;
    double pad = sr * p->speech_pad_ms / 1000;
    double max_speech =
        p->max_speech_s > 0 ? sr * p->max_speech_s - WIN - 2 * pad : 1e300;
    double min_silence = sr * p->min_silence_ms / 1000;
    double min_silence_at_max = sr * 98 / 1000;
    size_t cap = 16, cnt = 0;
    size_t *s = malloc(sizeof(size_t) * 2 * cap);
    if (!s)
        return -1;
    int triggered = 0, has = 0;
    size_t start = 0, temp_end = 0, prev_end = 0, next_start = 0;
#define PUSH(a, b)                                                             \
    do {                                                                       \
        if (cnt == cap) {                                                      \
            size_t *t2 = realloc(s, sizeof(size_t) * 4 * cap);                 \
            if (!t2) {                                                         \
                free(s);                                                       \
                return -1;                                                     \
            }                                                                  \
            s = t2;                                                            \
            cap *= 2;                                                          \
        }                                                                      \
        s[2 * cnt] = (a);                                                      \
        s[2 * cnt + 1] = (b);                                                  \
        cnt++;                                                                 \
    } while (0)
    for (long i = 0; i < n_probs; i++) {
        double pr = probs[i];
        size_t cur = (size_t)i * WIN;
        if (pr >= threshold && temp_end) {
            temp_end = 0;
            if (next_start < prev_end)
                next_start = cur;
        }
        if (pr >= threshold && !triggered) {
            triggered = 1;
            has = 1;
            start = cur;
            continue;
        }
        if (triggered && (double)(cur - start) > max_speech) {
            if (prev_end) {
                PUSH(start, prev_end);
                has = 0;
                if (next_start < prev_end) {
                    triggered = 0;
                } else {
                    start = next_start;
                    has = 1;
                }
                prev_end = next_start = temp_end = 0;
            } else {
                PUSH(start, cur);
                has = 0;
                prev_end = next_start = temp_end = 0;
                triggered = 0;
                continue;
            }
        }
        if (pr < neg && triggered) {
            if (!temp_end)
                temp_end = cur;
            if ((double)(cur - temp_end) > min_silence_at_max)
                prev_end = temp_end;
            if ((double)(cur - temp_end) < min_silence)
                continue;
            if ((double)(temp_end - start) > min_speech)
                PUSH(start, temp_end);
            has = 0;
            prev_end = next_start = temp_end = 0;
            triggered = 0;
            continue;
        }
    }
    if (has && triggered && (double)(n - start) > min_speech)
        PUSH(start, n);
#undef PUSH
    for (size_t i = 0; i < cnt; i++) {
        size_t *a = &s[2 * i];
        size_t ipad = (size_t)pad;
        if (i == 0)
            a[0] = a[0] > ipad ? a[0] - ipad : 0;
        if (i + 1 < cnt) {
            size_t *b = &s[2 * i + 2];
            size_t silence = b[0] > a[1] ? b[0] - a[1] : 0;
            if (silence < 2 * ipad) {
                a[1] += silence / 2;
                b[0] = b[0] > silence / 2 ? b[0] - silence / 2 : 0;
            } else {
                a[1] = a[1] + ipad < n ? a[1] + ipad : n;
                b[0] = b[0] > ipad ? b[0] - ipad : 0;
            }
        } else {
            a[1] = a[1] + ipad < n ? a[1] + ipad : n;
        }
    }
    *seg = s;
    return (long)cnt;
}
