/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_diarize.c - the diarization's grouping (src/llm/diarize.c) on made
 * embeddings: four speakers, each a direction in 256 dimensions, each
 * window its speaker's direction plus noise, in turns of random length.
 * The grouping must find four groups and put nearly every window in its
 * speaker's: a short recording grouped whole, a long one by chunks, and one
 * whose chunks' centroids are reduced again (a small chunk standing for
 * days of speech).
 */
#include "llm/speaker.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 256
#define SPEAKERS 4

static uint64_t state = 0x2545f4914f6cdd1dull;

static uint64_t next(void)
{
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

/* a normal sample (Box-Muller) */
static double gauss(void)
{
    double u = ((double)(next() >> 11) + 0.5) / 9007199254740992.0;
    double v = (double)(next() >> 11) / 9007199254740992.0;
    return sqrt(-2 * log(u)) * cos(6.283185307179586 * v);
}

static void unit(float *v)
{
    double s = 0;
    for (int i = 0; i < DIM; i++)
        s += (double)v[i] * v[i];
    for (int i = 0; i < DIM; i++)
        v[i] = (float)(v[i] / sqrt(s));
}

static int fails;

static void check(uint32_t n, uint32_t chunk, const char *what)
{
    float centre[SPEAKERS][DIM];
    for (int s = 0; s < SPEAKERS; s++) {
        for (int i = 0; i < DIM; i++)
            centre[s][i] = (float)gauss();
        unit(centre[s]);
    }
    float *emb = malloc(sizeof(float) * DIM * n);
    int *who = malloc(sizeof(int) * n);
    uint32_t *label = malloc(sizeof(uint32_t) * n);
    if (!emb || !who || !label) {
        printf("FAIL %s: out of memory\n", what);
        fails++;
        free(emb);
        free(who);
        free(label);
        return;
    }
    int s = 0;
    for (uint32_t i = 0; i < n;) {
        uint32_t turn = 5 + (uint32_t)(next() % 56);
        for (uint32_t k = 0; k < turn && i < n; k++, i++) {
            who[i] = s;
            for (int t = 0; t < DIM; t++)
                emb[(size_t)i * DIM + t] =
                    centre[s][t] + (float)(0.07 * gauss());
            unit(emb + (size_t)i * DIM);
        }
        s = (s + 1 + (int)(next() % (SPEAKERS - 1))) % SPEAKERS;
    }
    if (janas_diar_cluster(emb, n, 20, chunk, label) != 0) {
        printf("FAIL %s: the grouping failed\n", what);
        fails++;
    } else {
        /* each group stands for the speaker most of its windows have */
        uint32_t groups = 0, wrong = 0;
        for (uint32_t i = 0; i < n; i++)
            groups = label[i] + 1 > groups ? label[i] + 1 : groups;
        uint32_t *count = calloc((size_t)groups * SPEAKERS, sizeof *count);
        uint32_t used = 0;
        for (uint32_t i = 0; count && i < n; i++)
            count[(size_t)label[i] * SPEAKERS + who[i]]++;
        for (uint32_t g = 0; count && g < groups; g++) {
            uint32_t tot = 0, most = 0;
            for (int k = 0; k < SPEAKERS; k++) {
                tot += count[(size_t)g * SPEAKERS + k];
                if (count[(size_t)g * SPEAKERS + k] > most)
                    most = count[(size_t)g * SPEAKERS + k];
            }
            used += tot > 0;
            wrong += tot - most;
        }
        free(count);
        if (used != SPEAKERS || wrong > n / 100) {
            printf("FAIL %s: %u groups, %u of %u windows wrong\n", what, used,
                   wrong, n);
            fails++;
        }
    }
    free(emb);
    free(who);
    free(label);
}

int main(void)
{
    check(300, 0, "300 windows grouped whole");
    check(5000, 0, "5000 windows by chunks");
    check(3000, 200, "3000 windows by chunks of 200, reduced again");
    if (fails)
        return 1;
    printf("test_diarize: ok\n");
    return 0;
}
