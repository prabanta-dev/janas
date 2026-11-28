/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_sample.c - the sampler's random sequence: the same seed draws the
 * same tokens, and going back n draws (janas_sampler_rewind) draws the last
 * n again, as a reply cut short after a speculative pass needs; at
 * temperature 0 the most likely token, going back or not. llama.cpp's
 * chain (struct janas_sample_more): Janas's order asked as stages draws
 * what the plain draw does; DRY, the repetition penalty and top-n-sigma
 * change what they should; mirostat and XTC draw again the same after a
 * rewind; the stages' names read as llama.cpp writes them.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* the most likely of a row where t is best, to build a history */
static void force(struct janas_sampler *s, int32_t t)
{
    float lg[NV] = {0};
    lg[t] = 50;
    janas_sampler_pick(s, lg, NULL);
}

static void chain(void)
{
    /* Janas's order asked as stages: the plain draw's tokens, at a
       temperature of 1 (where 1/t and x t are the same numbers) */
    struct janas_sampler *a = janas_sampler_create(NV);
    struct janas_sampler *b = janas_sampler_create(NV);
    struct janas_sample_params pa = {
        .temperature = 1.0f, .top_k = 10, .top_p = 0.9f, .min_p = 0.05f};
    struct janas_sample_params pb = pa;
    pb.more.n_stage =
        janas_sample_stages("top_k;temperature;top_p;min_p", pb.more.stage,
                            JANAS_SAMPLE_MAX_STAGES);
    CHECK(pb.more.n_stage == 4, "stages: %d", pb.more.n_stage);
    janas_sampler_set(a, &pa, 777);
    janas_sampler_set(b, &pb, 777);
    int32_t ta[DRAWS], tb[DRAWS];
    picks(a, 0, DRAWS, ta);
    picks(b, 0, DRAWS, tb);
    int same = 1;
    for (int i = 0; i < DRAWS; i++)
        same &= ta[i] == tb[i];
    CHECK(same, "Janas's order as stages draws other tokens than the plain");

    /* mirostat 2, typical, XTC and the dynamic temperature: a rewind
       draws the same again */
    struct janas_sample_params pm = {.temperature = 0.8f, .top_k = 40};
    pm.more.mirostat = 2;
    pm.more.mirostat_tau = 3;
    pm.more.mirostat_eta = 0.2f;
    struct janas_sample_params px = {.temperature = 1.0f, .top_k = 30};
    px.more.typical_p = 0.9f;
    px.more.xtc_probability = 0.5f;
    px.more.xtc_threshold = 0.05f;
    px.more.dynatemp_range = 0.5f;
    const struct janas_sample_params *both[2] = {&pm, &px};
    for (int v = 0; v < 2; v++) {
        janas_sampler_set(a, both[v], 99);
        janas_sampler_mark(a);
        int32_t first[DRAWS], again[DRAWS];
        picks(a, 0, DRAWS, first);
        janas_sampler_rewind(a, 10);
        picks(a, DRAWS - 10, 10, again);
        int ok = 1, valid = 1;
        for (int i = 0; i < 10; i++)
            ok &= again[i] == first[DRAWS - 10 + i];
        for (int i = 0; i < DRAWS; i++)
            valid &= first[i] >= 0 && first[i] < NV;
        CHECK(ok && valid, "%s: the draws after a rewind differ",
              v ? "typical, XTC, dynamic temperature" : "mirostat 2");
    }

    /* DRY: after 1 2 3 1 2, a 3 would go on repeating a run of two */
    struct janas_sample_params pd = {0};
    pd.more.dry_multiplier = 0.8f;
    pd.more.dry_base = 1.75f;
    pd.more.dry_allowed = 2;
    janas_sampler_set(a, &pd, 1);
    janas_sampler_mark(a);
    force(a, 1), force(a, 2), force(a, 3), force(a, 1), force(a, 2);
    float lg[NV] = {0};
    lg[3] = 1.0f, lg[4] = 0.5f;
    int32_t t = janas_sampler_pick(a, lg, NULL);
    CHECK(t == 4, "DRY: picked %d, not 4", t);
    pd.more.dry_multiplier = 0;
    janas_sampler_set(a, &pd, 1);
    janas_sampler_mark(a);
    force(a, 1), force(a, 2), force(a, 3), force(a, 1), force(a, 2);
    t = janas_sampler_pick(a, lg, NULL);
    CHECK(t == 3, "without DRY: picked %d, not 3", t);

    /* the repetition penalty, over the last two tokens alone */
    struct janas_sample_params pr = {0};
    pr.more.repeat = 1.5f;
    pr.more.last_n = 2;
    janas_sampler_set(a, &pr, 1);
    janas_sampler_mark(a);
    force(a, 5);
    memset(lg, 0, sizeof lg);
    lg[5] = 2.0f, lg[6] = 1.5f;
    t = janas_sampler_pick(a, lg, NULL);
    CHECK(t == 6, "repeat penalty: picked %d, not 6", t);
    force(a, 7), force(a, 8); /* 5 out of the window */
    t = janas_sampler_pick(a, lg, NULL);
    CHECK(t == 5, "repeat penalty past its window: picked %d, not 5", t);

    /* top-n-sigma: one logit far above the rest leaves it alone */
    struct janas_sample_params ps = {.temperature = 1.5f};
    ps.more.top_n_sigma = 1;
    janas_sampler_set(a, &ps, 5);
    int alone = 1;
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < NV; j++)
            lg[j] = (float)(j % 3) * 0.1f;
        lg[17] = 9;
        alone &= janas_sampler_pick(a, lg, NULL) == 17;
    }
    CHECK(alone, "top-n-sigma kept more than the one far above");

    /* the stages' names */
    uint8_t st[JANAS_SAMPLE_MAX_STAGES];
    CHECK(janas_sample_stages("edskypmxt", st, JANAS_SAMPLE_MAX_STAGES) == 9 &&
              st[0] == JANAS_STAGE_PENALTIES &&
              st[8] == JANAS_STAGE_TEMPERATURE,
          "the letters edskypmxt");
    CHECK(janas_sample_stages("top_k, top_p;min_p temperature", st,
                              JANAS_SAMPLE_MAX_STAGES) == 4 &&
              st[1] == JANAS_STAGE_TOP_P,
          "names separated three ways");
    CHECK(janas_sample_stages("dry", st, JANAS_SAMPLE_MAX_STAGES) == 1 &&
              st[0] == JANAS_STAGE_DRY,
          "dry is a name, not d r y");
    CHECK(janas_sample_stages("top_k;bogus", st, JANAS_SAMPLE_MAX_STAGES) < 0,
          "an unknown stage accepted");
    janas_sampler_destroy(a);
    janas_sampler_destroy(b);
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
    chain();
    if (failures) {
        printf("test_sample: %d failures\n", failures);
        return 1;
    }
    printf("test_sample: ok\n");
    return 0;
}
