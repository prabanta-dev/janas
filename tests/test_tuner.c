/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_tuner.c - the self-tuner on simulated costs: it settles on the
 * fastest candidate, keeps the default on near ties, never picks a GPU
 * candidate when the GPU is not allowed, tries the others ever more rarely
 * while they lose, and its saved choices come back; the devices the user
 * allows (common/devices.h) as written and as the tuner keeps to them.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/devices.h"
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

    /* ... unless the GPU takes single tokens: then it is tried, and kept
       when faster */
    {
        struct janas_tuner g;
        janas_tuner_init(&g, cand, 4, def, 0);
        janas_tuner_allow_gpu(&g, 1);
        janas_tuner_allow_gpu_one(&g, 1);
        int tried_gpu = 0;
        for (int i = 0; i < 300; i++) {
            int c = janas_tuner_pick(&g, 0);
            tried_gpu += g.cand[c].gpu;
            janas_tuner_record(&g, 0, c, c == 3 ? 0.5 : 1.0);
        }
        CHECK(tried_gpu > 0, "GPU never tried for one token");
        CHECK(g.cand[janas_tuner_best(&g, 0)].gpu,
              "faster GPU not best for one token (best %d)",
              janas_tuner_best(&g, 0));
        /* slower: tried, not kept */
        janas_tuner_init(&g, cand, 4, def, 0);
        janas_tuner_allow_gpu(&g, 1);
        janas_tuner_allow_gpu_one(&g, 1);
        for (int i = 0; i < 300; i++) {
            int c = janas_tuner_pick(&g, 0);
            janas_tuner_record(&g, 0, c, c == 3 ? 2.0 : 1.0);
        }
        CHECK(!g.cand[janas_tuner_best(&g, 0)].gpu,
              "slower GPU best for one token");
    }

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
    CHECK(u.cls[2].threads == 20 && u.cls[2].threads == t.cls[2].threads,
          "loaded threads %d, want 20", u.cls[2].threads);
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
    for (int i = 0; i < 6000; i++) { /* the machine changes: the GPU wins */
        int c = janas_tuner_pick(&t, 2);
        tries += c == 3;
        janas_tuner_record(&t, 2, c, (const double[]){1.0, 1.2, 0.8, 0.3}[c]);
    }
    CHECK(janas_tuner_best(&t, 2) == 3, "after the change best %d, want 3",
          janas_tuner_best(&t, 2));
    CHECK(tries > 2000, "the new best taken %d times in 6000", tries);

    /* the threads are settled by the first tries and kept: a pass costing
       more as the context grows must not hand the choice to a candidate
       whose cost was measured at short context (a chat drifting onto
       fewer cores). 20 threads win the first tries; then every pass, of
       any candidate, costs more: the other threads, tried now and then
       with fresh passes, never win */
    janas_tuner_init(&t, cand, 4, def, 0);
    janas_tuner_allow_gpu(&t, 0);
    run(&t, (const double[]){1.0, 1.2, 0.8, 0.9}, 40);
    CHECK(janas_tuner_best(&t, 2) == 2, "settled best %d, want 2",
          janas_tuner_best(&t, 2));
    int others = 0;
    for (int i = 0; i < 5000; i++) {
        int c = janas_tuner_pick(&t, 2);
        others += t.cand[c].threads != 20;
        janas_tuner_record(&t, 2, c, 0.8 + i * 0.001);
    }
    CHECK(janas_tuner_best(&t, 2) == 2, "drifted: best %d, want 2",
          janas_tuner_best(&t, 2));
    CHECK(others > 0 && others < 50, "%d passes on other threads in 5000",
          others);

    /* the machine changes after the threads are settled (a laptop that
       warms up): 12 threads become clearly faster than the 20 chosen, and
       tries won in a row hand the threads over */
    janas_tuner_init(&t, cand, 4, def, 0);
    janas_tuner_allow_gpu(&t, 0);
    run(&t, (const double[]){1.0, 1.2, 0.8, 0.9}, 40);
    CHECK(janas_tuner_best(&t, 2) == 2, "settled best %d, want 2",
          janas_tuner_best(&t, 2));
    run(&t, (const double[]){1.0, 0.6, 0.9, 0.9}, 5000);
    CHECK(janas_tuner_best(&t, 2) == 1, "warmed up: best %d, want 1",
          janas_tuner_best(&t, 2));

    /* another program takes the efficiency cores: 20 threads 30 times
       slower, 6 and 12 alike - the first try moves the threads (it took
       645 passes, the 6 and 12 taking the tries in turn, 10 Oct 2026) */
    janas_tuner_init(&t, cand, 4, def, 0);
    janas_tuner_allow_gpu(&t, 0);
    run(&t, (const double[]){1.0, 1.2, 0.8, 0.9}, 40);
    int slow = 0, i30;
    for (i30 = 0; i30 < 2000 && janas_tuner_best(&t, 2) == 2; i30++) {
        int c = janas_tuner_pick(&t, 2);
        slow += c == 2;
        janas_tuner_record(&t, 2, c, (const double[]){0.9, 0.9, 27, 0.9}[c]);
    }
    CHECK(janas_tuner_best(&t, 2) != 2 && slow < 5,
          "20 threads 30 times slower: best %d after %d passes, %d slow",
          janas_tuner_best(&t, 2), i30, slow);

    /* ... 6 and 12 both a third faster, not twice: their wins count
       together, the next try soon after one won */
    janas_tuner_init(&t, cand, 4, def, 0);
    janas_tuner_allow_gpu(&t, 0);
    run(&t, (const double[]){1.0, 1.2, 0.8, 0.9}, 40);
    slow = 0;
    for (i30 = 0; i30 < 2000 && janas_tuner_best(&t, 2) == 2; i30++) {
        int c = janas_tuner_pick(&t, 2);
        slow += c == 2;
        janas_tuner_record(&t, 2, c, (const double[]){0.6, 0.6, 0.8, 0.9}[c]);
    }
    CHECK(janas_tuner_best(&t, 2) != 2 && i30 < 400,
          "6 and 12 a third faster: best %d after %d passes",
          janas_tuner_best(&t, 2), i30);
    /* forced: always that candidate */
    janas_tuner_force(&v, 1);
    for (int i = 0; i < 10; i++)
        CHECK(janas_tuner_pick(&v, 0) == 1, "forced candidate not used");

    /* the user's devices: a class kept to the NPU runs it, slower or not;
       a list chooses among its own; a fixed tuner runs the allowed */
    const struct janas_tune_cand dc[4] = {{.threads = 12},
                                          {.threads = 12, .gpu = 1},
                                          {.threads = 12, .npu = 1},
                                          {.threads = 12, .gpu = 1, .npu = 1}};
    const int dd[JANAS_TUNE_CLASSES] = {0, 0, 0, 0};
    struct janas_tuner d;
    janas_tuner_init(&d, dc, 4, dd, 0);
    janas_tuner_allow_gpu(&d, 1);
    janas_tuner_allow(&d, 2, 1 << JANAS_DEV_NPU);
    int elsewhere = 0;
    for (int i = 0; i < 200; i++) {
        int c = janas_tuner_pick(&d, 2);
        elsewhere += c != 2;
        janas_tuner_record(&d, 2, c, (const double[]){0.5, 0.4, 1.0, 0.9}[c]);
    }
    CHECK(!elsewhere && janas_tuner_best(&d, 2) == 2,
          "npu: %d passes elsewhere, best %d", elsewhere,
          janas_tuner_best(&d, 2));
    janas_tuner_init(&d, dc, 4, dd, 0);
    janas_tuner_allow_gpu(&d, 1);
    janas_tuner_allow(&d, 2, 1 << JANAS_DEV_NPU | 1 << JANAS_DEV_GPU);
    elsewhere = 0;
    for (int i = 0; i < 200; i++) {
        int c = janas_tuner_pick(&d, 2);
        elsewhere += c == 0 || c == 3;
        janas_tuner_record(&d, 2, c, (const double[]){0.3, 1.0, 0.6, 0.2}[c]);
    }
    CHECK(!elsewhere && janas_tuner_best(&d, 2) == 2,
          "npu,gpu: %d passes elsewhere, best %d", elsewhere,
          janas_tuner_best(&d, 2));
    janas_tuner_init(&d, dc, 4, dd, 1);
    janas_tuner_allow(&d, 2, 1 << (JANAS_DEV_GPU | JANAS_DEV_NPU));
    CHECK(janas_tuner_pick(&d, 2) == 3, "fixed, npu+gpu: candidate %d",
          janas_tuner_pick(&d, 2));
    /* the devices as written */
    char why[200], nm[64];
    static const struct {
        const char *spec;
        int mask;
    } specs[] = {{"auto", 0},
                 {"", 0},
                 {"cpu", 1},
                 {"npu", 1 << 2},
                 {"NPU+gpu", 1 << 3},
                 {"gpu+cpu", 1 << 1},
                 {"cpu,npu", 1 | 1 << 2},
                 {"npu,gpu,npu+gpu", 1 << 1 | 1 << 2 | 1 << 3},
                 {"tpu", -1},
                 {"npu,", -1},
                 {"+npu", -1},
                 {"npu++gpu", -1},
                 {"gpu,,npu", -1}};
    for (size_t i = 0; i < sizeof specs / sizeof *specs; i++) {
        int got = janas_devices_mask(specs[i].spec, why, sizeof why);
        CHECK(got == specs[i].mask, "devices \"%s\": %d, want %d",
              specs[i].spec, got, specs[i].mask);
    }
    CHECK(janas_devices_mask(NULL, why, sizeof why) == 0, "devices NULL");
    janas_devices_name(1 << 2 | 1 << 3, nm, sizeof nm);
    CHECK(!strcmp(nm, "npu,npu+gpu"), "name \"%s\"", nm);
    janas_devices_name(0, nm, sizeof nm);
    CHECK(!strcmp(nm, "auto"), "name \"%s\"", nm);
    CHECK(janas_devices_option("--prompt-on", "tpu", why, sizeof why) < 0 &&
              janas_devices_option("--ctx", "4096", why, sizeof why) == 0 &&
              janas_devices_option("--reply-on", NULL, why, sizeof why) < 0,
          "devices options");

    /* the CPU alone for a prompt's blocks (issue #50): chosen where it is
       faster, never for the other passes, and once its side is settled
       the tries keep to it (the fast prompt's sums differ from the CPU's
       in the last bits); the side saved and loaded */
    const struct janas_tune_cand ca[4] = {{.threads = 6},
                                          {.threads = 12},
                                          {.threads = 6, .alone = 1},
                                          {.threads = 12, .alone = 1}};
    const int da[JANAS_TUNE_CLASSES] = {0, 0, 0, 0};
    janas_tuner_init(&t, ca, 4, da, 0);
    run(&t, (const double[]){1.0, 1.1, 0.7, 0.8}, 40);
    CHECK(janas_tuner_best(&t, 2) == 2 && t.cls[2].alone == 1,
          "CPU alone: best %d (side %d), want 2 (1)", janas_tuner_best(&t, 2),
          t.cls[2].alone);
    int alone_picked = 0;
    for (int i = 0; i < 400; i++)
        for (int k = 0; k < JANAS_TUNE_CLASSES; k++) {
            if (k == 2)
                continue;
            int c = janas_tuner_pick(&t, k);
            alone_picked += ca[c].alone;
            janas_tuner_record(&t, k, c, ca[c].alone ? 0.1 : 1.0);
        }
    CHECK(alone_picked == 0, "CPU alone picked %d times for other passes",
          alone_picked);
    snprintf(dir, sizeof(dir), "%s/janas_test_tuner_XXXXXX",
             tmp && *tmp ? tmp : "/tmp");
    CHECK(mkdtemp(dir) != NULL, "cannot make a directory in %s\n",
          tmp && *tmp ? tmp : "/tmp");
    snprintf(home_dir, sizeof(home_dir), "%s/cache", dir);
    setenv("XDG_CACHE_HOME", home_dir, 1);
    janas_tuner_save(&t, "alone|key");
    janas_tuner_init(&u, ca, 4, da, 0);
    janas_tuner_load(&u, "alone|key");
    CHECK(u.cls[2].alone == 1 && janas_tuner_best(&u, 2) == 2,
          "loaded side %d, best %d, want 1, 2", u.cls[2].alone,
          janas_tuner_best(&u, 2));
    janas_tuner_init(&t, ca, 4, da, 0);
    run(&t, (const double[]){1.0, 1.1, 1.5, 1.6}, 40);
    CHECK(t.cls[2].alone == 0, "fast prompt settled: side %d, want 0",
          t.cls[2].alone);
    alone_picked = 0;
    for (int i = 0; i < 4000; i++) {
        int c = janas_tuner_pick(&t, 2);
        alone_picked += ca[c].alone;
        janas_tuner_record(&t, 2, c, ca[c].alone ? 0.3 : 1.0);
    }
    CHECK(alone_picked == 0 && !ca[janas_tuner_best(&t, 2)].alone,
          "side settled, yet the CPU alone picked %d times", alone_picked);
    snprintf(f, sizeof(f), "%s/janas/tuning.txt", home_dir);
    unlink(f);
    snprintf(f, sizeof(f), "%s/janas", home_dir);
    rmdir(f);
    rmdir(home_dir);
    rmdir(dir);

    printf("test_tuner: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
