/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_aac_eld.c - AAC ELD's low-delay synthesis filter bank (ISO/IEC
 * 14496-3, 4.6.20.2): the inverse transform of a frame's n coefficients
 * over four frames, x[i] = -2/N sum X[k] cos(2 pi / N (i + n0)(k + 1/2)),
 * N = 2n, n0 = (1 - n) / 2, made of the inverse MDCT of 2n samples (the
 * same cosines a frame later, their sign changing every 2n); the
 * low-delay window over the four frames; each output the sum of four
 * frames' windowed samples, the newest one's first quarter and the three
 * before. The samples are taken a quarter of a frame later than the text
 * of the standard says, as its reference decoder takes them (and FFmpeg).
 */
#include <string.h>

#include "audio_aac.h"
#include "audio_aac_ld_tab.h"
#include "audio_fft.h"

void aac_eld_synth(struct aimdct *m, int n, float *coef, float *hist,
                   float *out, float *y, float *z)
{
    const float *w = n == 480 ? AAC_ELD_WINDOW_480 : AAC_ELD_WINDOW_512;
    int wlen = 4 * n - n / 4; /* the window's last quarter frame is zero */
    aimdct_run(m, coef, y);
    float g = -1.0f / (float)n / 32768.0f;
    for (int i = 0; i < 4 * n; i++) {
        int j = i - n + n / 4; /* x[i] is the inverse MDCT's y[i - n] */
        float s = g;
        while (j < 0) {
            j += 2 * n;
            s = -s;
        }
        while (j >= 2 * n) {
            j -= 2 * n;
            s = -s;
        }
        z[i] = i < wlen ? w[i] * y[j] * s : 0;
    }
    for (int i = 0; i < n; i++)
        out[i] = z[i] + hist[i];
    for (int i = 0; i < 2 * n; i++)
        hist[i] = hist[n + i] + z[n + i];
    memcpy(hist + 2 * n, z + 3 * n, sizeof(float) * (size_t)n);
}
