/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * speaker.c - WeSpeaker's ResNet34 speaker embedding as its ONNX model
 * computes it: Kaldi's log mel filter banks (torchaudio's kaldi.fbank: 25
 * ms frames every 10 ms, DC removed, pre-emphasis 0.97, Hamming window,
 * 512-point power spectrum, 80 triangular bands on Kaldi's mel scale from
 * 20 Hz to Nyquist, the samples scaled to 16 bits), their mean over the
 * window taken off; a 3x3 convolution to 32 channels, then 3, 4, 6 and 3
 * residual blocks of 32, 64, 128 and 256 channels (batch norms folded in,
 * strides of 2 and 1x1 shortcuts where the size halves), the frequency and
 * channel axes kept, the mean and the standard deviation over time, their
 * projection to 256 less the embeddings' mean. The convolutions as tiled
 * products of patches (im2col) on the vision encoders' kernels.
 */
#include "speaker.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/audio_fft.h"
#include "common/pool.h"
#include "common/sysinfo.h"
#include "jns.h"
#include "vision_kernels.h"

#define MELS 80
#define NCONV 36
#define DIM 256

struct janas_speaker {
    struct janas_jns j;
    uint8_t *data;
    const uint16_t *w[NCONV];
    const float *b[NCONV];
    int kernel[NCONV], stride[NCONV], cin[NCONV], cout[NCONV];
    const uint16_t *proj;
    const float *proj_b, *mean;
    struct janas_pool *pool;
};

static const void *tensor(struct janas_speaker *s, const char *name,
                          uint32_t type, uint64_t count)
{
    const struct janas_jns_tensor *t = janas_jns_tensor(&s->j, name);
    if (!t || t->type != type || t->bytes != count * (type ? 2 : 4))
        return NULL;
    return s->data + (t->offset - s->j.h.resident_offset);
}

void janas_speaker_close(struct janas_speaker *s)
{
    if (!s)
        return;
    if (s->pool)
        janas_pool_destroy(s->pool);
    free(s->data);
    janas_jns_close(&s->j);
    free(s);
}

struct janas_speaker *janas_speaker_open(const char *path, int n_threads,
                                         char *err, size_t err_len)
{
    static const int CH[NCONV] = {32,  32,  32,  32,  32,  32,  32,  64,  64,
                                  64,  64,  64,  64,  64,  64,  64,  128, 128,
                                  128, 128, 128, 128, 128, 128, 128, 128, 128,
                                  128, 128, 256, 256, 256, 256, 256, 256, 256};
    struct janas_speaker *s = calloc(1, sizeof *s);
    if (!s) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    s->j.fd = -1;
    if (janas_jns_open(path, &s->j, err, err_len) != 0) {
        free(s);
        return NULL;
    }
    const char *why = NULL;
    s->data = malloc(s->j.h.resident_bytes ? s->j.h.resident_bytes : 1);
    size_t got = 0;
    while (s->data && got < s->j.h.resident_bytes) {
        ssize_t k = pread(s->j.fd, s->data + got, s->j.h.resident_bytes - got,
                          (off_t)(s->j.h.resident_offset + got));
        if (k <= 0)
            break;
        got += (size_t)k;
    }
    if (!s->data || got != s->j.h.resident_bytes)
        why = "cannot read the weights";
    const int32_t *ks = NULL, *ss = NULL;
    uint64_t nk = 0, ns = 0;
    if (!why && (janas_jns_meta_ints(&s->j, "spk.kernels", &nk, &ks) ||
                 janas_jns_meta_ints(&s->j, "spk.strides", &ns, &ss) ||
                 nk != NCONV || ns != NCONV))
        why = "not a WeSpeaker ResNet34 model (made by wespeaker2jns)";
    for (int i = 0; i < NCONV && !why; i++) {
        s->kernel[i] = ks[i];
        s->stride[i] = ss[i];
        s->cout[i] = CH[i];
        /* the input: the block's, or the layer before's for its first */
        s->cin[i] = i == 0                 ? 1
                    : (i == 7 || i == 9)   ? 32
                    : (i == 16 || i == 18) ? 64
                    : (i == 29 || i == 31) ? 128
                                           : CH[i];
        if ((ks[i] != 1 && ks[i] != 3) || (ss[i] != 1 && ss[i] != 2)) {
            why = "a convolution of a kernel or stride not read";
            break;
        }
        char name[32];
        snprintf(name, sizeof name, "spk.conv%d.w", i);
        s->w[i] = tensor(s, name, 1,
                         (uint64_t)s->cout[i] * s->cin[i] * ks[i] * ks[i]);
        snprintf(name, sizeof name, "spk.conv%d.b", i);
        s->b[i] = tensor(s, name, 0, (uint64_t)s->cout[i]);
        if (!s->w[i] || !s->b[i])
            why = "a convolution's weights are missing or of another size";
    }
    if (!why) {
        s->proj = tensor(s, "spk.proj.w", 1, 256 * 5120);
        s->proj_b = tensor(s, "spk.proj.b", 0, 256);
        s->mean = tensor(s, "spk.mean", 0, 256);
        if (!s->proj || !s->proj_b || !s->mean)
            why = "the projection is missing";
    }
    if (!why) {
        int n = n_threads;
        struct janas_cpu_layout lay;
        if (n <= 0)
            n = janas_cpu_layout(NULL, 0, &lay) == 0 ? lay.n_p_cores : 1;
        s->pool = janas_pool_create(n > 0 ? n : 1, NULL);
        if (!s->pool)
            why = "cannot start the threads";
    }
    if (why) {
        snprintf(err, err_len, "%s: %s", path, why);
        janas_speaker_close(s);
        return NULL;
    }
    return s;
}

/* ---- the filter banks ---- */

static double mel(double f)
{
    return 1127.0 * log(1.0 + f / 700.0);
}

enum { FB_LEN = 400, FB_SHIFT = 160, FB_NFFT = 512, FB_BINS = 256 };

struct janas_fbank {
    struct afft *f;
    float *bank;
    float win[FB_LEN];
};

uint32_t janas_fbank_count(size_t n)
{
    if (n < FB_LEN)
        return 0;
    size_t nf = 1 + (n - FB_LEN) / FB_SHIFT;
    return nf > UINT32_MAX ? UINT32_MAX : (uint32_t)nf;
}

void janas_fbank_free(struct janas_fbank *b)
{
    if (!b)
        return;
    afft_free(b->f);
    free(b->bank);
    free(b);
}

struct janas_fbank *janas_fbank_new(void)
{
    const int len = FB_LEN, nfft = FB_NFFT, bins = FB_BINS;
    struct janas_fbank *fb = calloc(1, sizeof *fb);
    if (!fb)
        return NULL;
    fb->f = afft_new(nfft);
    fb->bank = calloc((size_t)MELS * bins, sizeof(float));
    if (!fb->f || !fb->bank) {
        janas_fbank_free(fb);
        return NULL;
    }
    float *win = fb->win, *bank = fb->bank;
    for (int i = 0; i < len; i++)
        win[i] = (float)(0.54 - 0.46 * cos(2 * M_PI * i / (len - 1)));
    /* the triangles, on Kaldi's mel scale from 20 Hz to 8 kHz */
    double lo = mel(20), hi = mel(8000), delta = (hi - lo) / (MELS + 1);
    for (int m = 0; m < MELS; m++) {
        double l = lo + m * delta, c = lo + (m + 1) * delta,
               r = lo + (m + 2) * delta;
        for (int k = 0; k < bins; k++) {
            double x = mel(16000.0 / nfft * k);
            double up = (x - l) / (c - l), down = (r - x) / (r - c);
            double v = up < down ? up : down;
            bank[m * bins + k] = v > 0 ? (float)v : 0;
        }
    }
    return fb;
}

void janas_fbank_frames(struct janas_fbank *fb, const float *pcm, size_t nf,
                        float *out)
{
    const int len = FB_LEN, shift = FB_SHIFT, bins = FB_BINS;
    const float *win = fb->win, *bank = fb->bank;
    struct afft *f = fb->f;
    float in[1024], spec[1024], frame[400];
    for (size_t t = 0; t < nf; t++) {
        const float *x = pcm + t * (size_t)shift;
        float mean = 0;
        for (int i = 0; i < len; i++) {
            frame[i] = x[i] * 32768.0f;
            mean += frame[i];
        }
        mean /= (float)len;
        for (int i = 0; i < len; i++)
            frame[i] -= mean;
        for (int i = len - 1; i > 0; i--)
            frame[i] -= 0.97f * frame[i - 1];
        frame[0] -= 0.97f * frame[0];
        memset(in, 0, sizeof in);
        for (int i = 0; i < len; i++)
            in[2 * i] = frame[i] * win[i]; /* (re, im) pairs */
        afft_run(f, in, spec);
        float pw[256];
        for (int k = 0; k < bins; k++)
            pw[k] =
                spec[2 * k] * spec[2 * k] + spec[2 * k + 1] * spec[2 * k + 1];
        for (int m = 0; m < MELS; m++) {
            float e = 0;
            const float *bk = bank + m * bins;
            for (int k = 0; k < bins; k++)
                e += bk[k] * pw[k];
            out[t * MELS + m] = logf(e > 1.1920929e-07f ? e : 1.1920929e-07f);
        }
    }
}

float *janas_speaker_fbank(const float *pcm, size_t n, uint32_t *frames)
{
    *frames = 0;
    if (n < FB_LEN)
        return calloc(1, sizeof(float));
    size_t nf = 1 + (n - FB_LEN) / FB_SHIFT;
    if (nf > UINT32_MAX)
        return NULL;
    float *out = malloc(sizeof(float) * nf * MELS);
    struct janas_fbank *fb = janas_fbank_new();
    if (!out || !fb) {
        free(out);
        janas_fbank_free(fb);
        return NULL;
    }
    janas_fbank_frames(fb, pcm, nf, out);
    janas_fbank_free(fb);
    *frames = (uint32_t)nf;
    return out;
}

/* ---- the network ---- */

/* a convolution of in (h x w x cin, channels last) into out (h' x w' x
   cout); col: the patches' room */
static void conv(struct janas_speaker *s, int i, const float *in, int h, int w,
                 float *out, int *oh, int *ow, float *col)
{
    int k = s->kernel[i], st = s->stride[i], pad = (k - 1) / 2, cin = s->cin[i];
    int H = (h + 2 * pad - k) / st + 1, Wd = (w + 2 * pad - k) / st + 1;
    size_t cols = (size_t)k * k * cin;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < Wd; x++) {
            float *c = col + ((size_t)y * Wd + x) * cols;
            for (int ky = 0; ky < k; ky++)
                for (int kx = 0; kx < k; kx++) {
                    int sy = y * st + ky - pad, sx = x * st + kx - pad;
                    float *d = c + ((size_t)ky * k + kx) * cin;
                    if (sy < 0 || sy >= h || sx < 0 || sx >= w)
                        memset(d, 0, sizeof(float) * (size_t)cin);
                    else
                        memcpy(d, in + ((size_t)sy * w + sx) * cin,
                               sizeof(float) * (size_t)cin);
                }
        }
    vk_matmul_f16(s->pool, s->w[i], s->b[i], col, out, (uint32_t)s->cout[i],
                  (uint32_t)cols, (uint32_t)(H * Wd), 0);
    *oh = H;
    *ow = Wd;
}

static void relu(float *x, size_t n)
{
    for (size_t i = 0; i < n; i++)
        x[i] = x[i] > 0 ? x[i] : 0;
}

int janas_speaker_embed(struct janas_speaker *s, const float *fbank,
                        uint32_t f0, uint32_t nf, float *emb)
{
    if (nf < 8)
        return -1;
    /* the input, frequency down, time across: h 80 x w nf x 1 */
    size_t big = (size_t)MELS * nf * 32;
    float *a = malloc(sizeof(float) * big), *b = malloc(sizeof(float) * big);
    float *c = malloc(sizeof(float) * big), *t = malloc(sizeof(float) * big);
    float *col = malloc(sizeof(float) * (size_t)MELS * nf * 9 * 32);
    int rc = -1;
    if (!a || !b || !c || !t || !col)
        goto end;
    for (int m = 0; m < MELS; m++) {
        double mean = 0;
        for (uint32_t i = 0; i < nf; i++)
            mean += fbank[(size_t)(f0 + i) * MELS + m];
        mean /= nf;
        for (uint32_t i = 0; i < nf; i++)
            a[(size_t)m * nf + i] =
                (float)(fbank[(size_t)(f0 + i) * MELS + m] - mean);
    }
    int h = MELS, w = (int)nf, oh, ow;
    conv(s, 0, a, h, w, b, &oh, &ow, col);
    relu(b, (size_t)oh * ow * 32);
    h = oh;
    w = ow;
    /* the residual blocks: x in b */
    static const int BLOCKS[4] = {3, 4, 6, 3};
    int ci = 1;
    float *x = b, *y = a, *z = c;
    for (int l = 0; l < 4; l++)
        for (int k = 0; k < BLOCKS[l]; k++) {
            int down = l > 0 && k == 0;
            int ia = ci, ib = ci + 1, is = down ? ci + 2 : -1;
            ci += down ? 3 : 2;
            int h1, w1, h2, w2;
            conv(s, ia, x, h, w, y, &h1, &w1, col);
            relu(y, (size_t)h1 * w1 * s->cout[ia]);
            conv(s, ib, y, h1, w1, z, &h2, &w2, col);
            const float *sc = x;
            if (is >= 0) {
                int h3, w3;
                conv(s, is, x, h, w, t, &h3, &w3, col);
                sc = t;
            }
            size_t nn = (size_t)h2 * w2 * s->cout[ib];
            for (size_t q = 0; q < nn; q++) {
                float v = z[q] + sc[q];
                z[q] = v > 0 ? v : 0;
            }
            float *tmp = x;
            x = z;
            z = tmp;
            h = h2;
            w = w2;
        }
    /* the statistics over time of each channel and frequency: c f order */
    float st[5120];
    int ch = 256;
    if (h * ch != 2560) /* 80 / 8 frequencies */
        goto end;
    for (int cc = 0; cc < ch; cc++)
        for (int f = 0; f < h; f++) {
            double sum = 0;
            for (int q = 0; q < w; q++)
                sum += x[((size_t)f * w + q) * ch + cc];
            double mean = sum / w, var = 0;
            for (int q = 0; q < w; q++) {
                double d = x[((size_t)f * w + q) * ch + cc] - mean;
                var += d * d;
            }
            var = w > 1 ? var / (w - 1) : 0;
            st[cc * h + f] = (float)mean;
            st[2560 + cc * h + f] = (float)sqrt(var + 1e-8);
        }
    vk_matmul_f16(s->pool, s->proj, s->proj_b, st, emb, DIM, 5120, 1, 0);
    for (int i = 0; i < DIM; i++)
        emb[i] -= s->mean[i];
    rc = 0;
end:
    free(a);
    free(b);
    free(c);
    free(t);
    free(col);
    return rc;
}
