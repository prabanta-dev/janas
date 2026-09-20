/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_tuner.c - the self-tuner on simulated costs: it settles on the
 * fastest candidate, keeps the default on near ties, never picks a GPU
 * candidate when the GPU is not allowed, tries the others ever more rarely
 * while they lose, and its saved choices come back.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "llm/tuner.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* runs passes of class 2 with fixed costs per candidate */
static void run(struct janas_tuner *t, const double *cost, int passes)
{
    for (int i = 0; i < passes; i++) {
        int c = janas_tuner_pick(t, 2);
        janas_tuner_record(t, 2, c, cost[c]);
    }
}

int main(void)
{
    /* a test that deadlocks must fail, not hold the suite for ever */
    alarm(600);
    const struct janas_tune_cand cand[4] = {{.threads = 6},
                                            {.threads = 12},
                                            {.threads = 20},
                                            {.threads = 20, .gpu = 1}};
    const int def[JANAS_TUNE_CLASSES] = {1, 0, 0};
    struct janas_tuner t;

    /* a clearly faster candidate wins */
    janas_tuner_init(&t, cand, 4, def, 0);
    janas_tuner_allow_gpu(&t, 1);
    run(&t, (const double[]){1.0, 1.2, 0.8, 0.9}, 40);
    CHECK(janas_tuner_best(&t, 2) == 2, "best %d, want 2",
          janas_tuner_best(&t, 2));

    /* a near tie keeps the default */
    janas_tuner_init(&t, cand, 4, def, 0);
    janas_tuner_allow_gpu(&t, 1);
    run(&t, (const double[]){1.0, 1.2, 0.99, 1.1}, 40);
    CHECK(janas_tuner_best(&t, 2) == 0, "near tie: best %d, want 0",
          janas_tuner_best(&t, 2));

    /* the GPU candidate is fastest, but not allowed */
    janas_tuner_init(&t, cand, 4, def, 0);
    janas_tuner_allow_gpu(&t, 0);
    run(&t, (const double[]){1.0, 1.2, 0.9, 0.5}, 200);
    for (int i = 0; i < 300; i++)
        CHECK(!t.cand[janas_tuner_pick(&t, 2)].gpu, "GPU picked in eco");
    CHECK(janas_tuner_best(&t, 2) == 2, "eco best %d, want 2",
          janas_tuner_best(&t, 2));
    janas_tuner_allow_gpu(&t, 1);
    run(&t, (const double[]){1.0, 1.2, 0.9, 0.5}, 40);
    CHECK(janas_tuner_best(&t, 2) == 3, "max best %d, want 3",
          janas_tuner_best(&t, 2));

    /* no GPU candidate for one token */
    for (int i = 0; i < 300; i++) {
        int c = janas_tuner_pick(&t, 0);
        CHECK(!t.cand[c].gpu, "GPU picked for one token");
        janas_tuner_record(&t, 0, c, c == 3 ? 0.1 : 1.0);
    }
    CHECK(!t.cand[janas_tuner_best(&t, 0)].gpu, "GPU best for one token");

    /* saved and loaded under a key (in a scratch cache directory) */
    char dir[512];
    const char *tmp = getenv("TMPDIR");
    snprintf(dir, sizeof(dir), "%s/janas_test_tuner_XXXXXX",
             tmp && *tmp ? tmp : "/tmp");
    CHECK(mkdtemp(dir) != NULL, "cannot make a directory in %s\n",
          tmp && *tmp ? tmp : "/tmp");
    char home_dir[600]; /* one level more than exists: made by the save */
    snprintf(home_dir, sizeof(home_dir), "%s/cache", dir);
    setenv("XDG_CACHE_HOME", home_dir, 1);
    janas_tuner_save(&t, "test|key");
    struct janas_tuner u;
    janas_tuner_init(&u, cand, 4, def, 0);
    janas_tuner_allow_gpu(&u, 1);
    janas_tuner_load(&u, "test|key");
    CHECK(janas_tuner_best(&u, 2) == 3, "loaded best %d, want 3",
          janas_tuner_best(&u, 2));
    struct janas_tuner v;
    janas_tuner_init(&v, cand, 4, def, 0);
    janas_tuner_load(&v, "other|key");
    CHECK(janas_tuner_best(&v, 2) == 0, "unknown key: best %d, want 0",
          janas_tuner_best(&v, 2));
    char f[700];
    snprintf(f, sizeof(f), "%s/janas/tuning.txt", home_dir);
    unlink(f);
    snprintf(f, sizeof(f), "%s/janas", home_dir);
    rmdir(f);
    rmdir(home_dir);
    rmdir(dir);

    /* tries of the alternatives: rarer while they change nothing (at a
       fixed 128 passes, 20000 passes would make 156), and often again once
       one wins */
    janas_tuner_init(&t, cand, 4, def, 0);
    janas_tuner_allow_gpu(&t, 1);
    int tries = 0;
    for (int i = 0; i < 20000; i++) {
        int c = janas_tuner_pick(&t, 2);
        tries += i >= 100 && c != 2;
        janas_tuner_record(&t, 2, c, (const double[]){1.0, 1.2, 0.8, 0.9}[c]);
    }
    CHECK(tries >= 5 && tries <= 30, "%d tries in 20000 passes", tries);
    tries = 0;
    for (int i = 0; i < 6000; i++) { /* the machine changes: 1 is best now */
        int c = janas_tuner_pick(&t, 2);
        tries += c == 1;
        janas_tuner_record(&t, 2, c, (const double[]){1.0, 0.3, 0.8, 0.9}[c]);
    }
    CHECK(janas_tuner_best(&t, 2) == 1, "after the change best %d, want 1",
          janas_tuner_best(&t, 2));
    CHECK(tries > 2000, "the new best taken %d times in 6000", tries);

    /* forced: always that candidate */
    janas_tuner_force(&v, 1);
    for (int i = 0; i < 10; i++)
        CHECK(janas_tuner_pick(&v, 0) == 1, "forced candidate not used");

    printf("test_tuner: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
