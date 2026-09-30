/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_sample.c - the sampler's random sequence: the same seed draws the
 * same tokens, and going back n draws (janas_sampler_rewind) draws the last
 * n again, as a reply cut short after a speculative pass needs; at
 * temperature 0 the most likely token, going back or not.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "llm/sample.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

#define NV 64
#define DRAWS 40

/* logits spread enough that the draws vary */
static void fill(float *lg, int step)
{
    for (int i = 0; i < NV; i++)
        lg[i] = (float)((i * 7 + step * 13) % NV) / 16.0f;
}

static void picks(struct janas_sampler *s, int from, int n, int32_t *out)
{
    float lg[NV];
    for (int i = 0; i < n; i++) {
        fill(lg, from + i);
        out[i] = janas_sampler_pick(s, lg, NULL);
    }
}

int main(void)
{
    alarm(60);
    struct janas_sampler *a = janas_sampler_create(NV);
    struct janas_sampler *b = janas_sampler_create(NV);
    if (!a || !b) {
        printf("test_sample: out of memory\n");
        return 1;
    }
    const struct janas_sample_params warm = {.temperature = 1.0f};
    janas_sampler_set(a, &warm, 12345);
    janas_sampler_set(b, &warm, 12345);
    int32_t ta[DRAWS], tb[DRAWS], again[DRAWS];
    picks(a, 0, DRAWS, ta);
    picks(b, 0, DRAWS, tb);
    int differ = 0;
    for (int i = 0; i < DRAWS; i++) {
        CHECK(ta[i] == tb[i], "same seed, draw %d: %d against %d", i, ta[i],
              tb[i]);
        differ |= ta[i] != ta[0];
    }
    CHECK(differ, "the draws never vary: the test proves nothing");

    /* b goes back 15 draws and draws them again: the same tokens */
    janas_sampler_rewind(b, 15);
    picks(b, DRAWS - 15, 15, again);
    for (int i = 0; i < 15; i++)
        CHECK(again[i] == ta[DRAWS - 15 + i],
              "after going back 15, draw %d: %d against %d", i, again[i],
              ta[DRAWS - 15 + i]);
    /* and from there on both go on alike */
    picks(a, DRAWS, 10, ta);
    picks(b, DRAWS, 10, tb);
    for (int i = 0; i < 10; i++)
        CHECK(ta[i] == tb[i], "after the rewind, draw %d: %d against %d", i,
              ta[i], tb[i]);

    /* greedy: the most likely token every time, going back or not */
    const struct janas_sample_params cold = {.temperature = 0.0f};
    janas_sampler_set(a, &cold, 7);
    picks(a, 0, 5, ta);
    janas_sampler_rewind(a, 3);
    picks(a, 5, 5, ta + 5);
    for (int i = 0; i < 10; i++) {
        float lg[NV];
        fill(lg, i);
        int best = 0;
        for (int k = 1; k < NV; k++)
            if (lg[k] > lg[best])
                best = k;
        CHECK(ta[i] == best, "greedy, draw %d: %d against %d", i, ta[i], best);
    }

    janas_sampler_destroy(a);
    janas_sampler_destroy(b);
    if (failures) {
        printf("test_sample: %d failures\n", failures);
        return 1;
    }
    printf("test_sample: ok\n");
    return 0;
}
