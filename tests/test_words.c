/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_words.c - the strict reading of what a user gives Janas: the
 * nearest of known names, numbers and choices read whole, a service's
 * settings file, and the table of the variables of the environment,
 * checked against every JANAS_ name the sources read.
 */
#define _GNU_SOURCE /* nftw's FTW_PHYS */
#include <ftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/env_names.h"
#include "common/words.h"
#include "services/common/conf.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char *const OPTS[] = {
    "--temp", "--top-k",  "--top-p", "--ctx",   "--cache",   "--reserve",
    "--max",  "--no-mtp", "--mtp",   "--think", "--mcp-auto"};
#define N_OPTS (sizeof OPTS / sizeof *OPTS)

static void closest(void)
{
    struct {
        const char *w, *want;
    } t[] = {
        {"--tmep", "--temp"},        /* two letters swapped */
        {"--temperature", "--temp"}, /* one the start of the other */
        {"--chache", "--cache"},     /* one letter more */
        {"--reserv", "--reserve"},   /* one less */
        {"--TEMP", "--temp"},        /* case aside */
        {"--mcp-aut", "--mcp-auto"}, /* the start, four letters or more */
        {"--frobnicate", NULL},      /* nothing near */
        {"--x", NULL},
        {"", NULL},
    };
    for (size_t i = 0; i < sizeof t / sizeof *t; i++) {
        const char *got = janas_closest(t[i].w, OPTS, N_OPTS);
        CHECK(t[i].want ? got && strcmp(got, t[i].want) == 0 : !got,
              "%s: %s, wanted %s", t[i].w, got ? got : "none",
              t[i].want ? t[i].want : "none");
    }
    char m[200];
    janas_unknown_word(m, sizeof m, "--tmep", "an option", OPTS, N_OPTS);
    CHECK(strcmp(m, "--tmep is not an option (did you mean --temp?)") == 0,
          "%s", m);
    janas_unknown_word(m, sizeof m, "--zzz", "an option", OPTS, N_OPTS);
    CHECK(strcmp(m, "--zzz is not an option") == 0, "%s", m);
    const char *env[] = {"JANAS_EXPERT_BITS", "JANAS_EXPERTS"};
    const char *got = janas_closest("JANAS_EXPERT_BIT", env, 2);
    CHECK(got && strcmp(got, "JANAS_EXPERT_BITS") == 0, "env: %s",
          got ? got : "none");
}

static void numbers(void)
{
    long long x = -1;
    double r = -1;
    char why[160];
    CHECK(janas_word_int("16384", 0, 1 << 30, &x, why, sizeof why) == 0 &&
              x == 16384,
          "16384");
    CHECK(janas_word_int("32k", 0, 1 << 30, &x, why, sizeof why) != 0 &&
              strcmp(why, "32k is not a whole number") == 0,
          "32k: %s", why);
    CHECK(janas_word_int("70000", 1, 65535, &x, why, sizeof why) != 0 &&
              strcmp(why, "70000 is above 65535") == 0,
          "70000: %s", why);
    CHECK(janas_word_int("-1", 0, 10, &x, why, sizeof why) != 0 &&
              strcmp(why, "-1 is below 0") == 0,
          "-1: %s", why);
    CHECK(janas_word_int("", 0, 10, &x, why, sizeof why) != 0, "empty");
    CHECK(janas_word_int(" 5", 0, 10, &x, why, sizeof why) != 0, "space");
    CHECK(janas_word_int(NULL, 0, 10, &x, why, sizeof why) != 0 &&
              strcmp(why, "a value is missing") == 0,
          "missing: %s", why);
    CHECK(janas_word_int("99999999999999999999", 0, 10, &x, why, sizeof why) !=
              0,
          "overflow");
    CHECK(janas_word_real("0.7", 0, 100, &r, why, sizeof why) == 0 && r == 0.7,
          "0.7");
    CHECK(janas_word_real("1e-05", 0, 1, &r, why, sizeof why) == 0, "1e-05");
    CHECK(janas_word_real("0,7", 0, 100, &r, why, sizeof why) != 0 &&
              strcmp(why, "0,7 is not a number") == 0,
          "0,7: %s", why);
    CHECK(janas_word_real("nan", 0, 100, &r, why, sizeof why) != 0, "nan");
    CHECK(janas_word_real("1.5", 0, 1, &r, why, sizeof why) != 0 &&
              strcmp(why, "1.5 is above 1") == 0,
          "1.5: %s", why);
    const char *on_off[] = {"off", "on"};
    CHECK(janas_word_choice("on", on_off, 2, why, sizeof why) == 1, "on");
    CHECK(janas_word_choice("of", on_off, 2, why, sizeof why) == -1 &&
              strcmp(why, "of is not one of off, on (did you mean off?)") == 0,
          "of: %s", why);
}

static void settings(void)
{
    char path[512];
    const char *dir = getenv("TMPDIR");
    snprintf(path, sizeof path, "%s/janas_test_words_%d.conf",
             dir && *dir ? dir : "/tmp", (int)getpid());
    FILE *f = fopen(path, "w");
    CHECK(f, "cannot write %s", path);
    if (!f)
        return;
    fputs("# a comment\n\n  aviationstak_key = WRONG\n"
          "aviationstack_key = ABCDEFGH12  \n",
          f);
    fclose(f);
    const char *names[] = {"aviationstack_key"};
    char val[64], why[300];
    int got = janas_conf_get(path, names, 1, names[0], val, sizeof val, why,
                             sizeof why);
    CHECK(got == 1 && strcmp(val, "ABCDEFGH12") == 0, "value: %s", val);
    CHECK(strstr(why, ", line 3: aviationstak_key is not a setting (did you "
                      "mean aviationstack_key?)") != NULL,
          "why: %s", why);
    f = fopen(path, "w");
    if (f) {
        fputs("aviationstack_key KEY2KEY2\n", f); /* no = : taken too */
        fclose(f);
    }
    got = janas_conf_get(path, names, 1, names[0], val, sizeof val, why,
                         sizeof why);
    CHECK(got == 1 && strcmp(val, "KEY2KEY2") == 0 && !why[0],
          "without =: %s / %s", val, why);
    unlink(path);
    CHECK(janas_conf_get(path, names, 1, names[0], val, sizeof val, why,
                         sizeof why) == 0 &&
              !val[0] && !why[0],
          "no file");
}

/* every "JANAS_..." string of the sources must be in the table, and every
   name of the table in the sources */
static char seen[256][48];
static int n_seen;

static int scan(const char *path, const struct stat *st, int kind,
                struct FTW *ftw)
{
    (void)st;
    (void)ftw;
    size_t n = strlen(path);
    if (kind != FTW_F || n < 3 || strstr(path, "third_party") ||
        (strcmp(path + n - 2, ".c") && strcmp(path + n - 2, ".h")) ||
        strstr(path, "env_names.c"))
        return 0;
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    static char text[1 << 20];
    size_t len = fread(text, 1, sizeof text - 1, f);
    fclose(f);
    text[len] = 0;
    for (char *p = text; (p = strstr(p, "\"JANAS_")); p++) {
        size_t k = 1;
        while (p[k] == '_' || (p[k] >= 'A' && p[k] <= 'Z') ||
               (p[k] >= '0' && p[k] <= '9'))
            k++;
        if (p[k] != '"' || k - 1 >= sizeof seen[0])
            continue;
        char name[48];
        memcpy(name, p + 1, k - 1);
        name[k - 1] = 0;
        int known = 0;
        for (size_t i = 0; i < JANAS_N_ENV_NAMES && !known; i++)
            known = strcmp(name, JANAS_ENV_NAMES[i]) == 0;
        CHECK(known, "%s reads %s, not in src/common/env_names.c", path, name);
        int dup = 0;
        for (int i = 0; i < n_seen && !dup; i++)
            dup = strcmp(seen[i], name) == 0;
        if (!dup && n_seen < 256)
            strcpy(seen[n_seen++], name);
    }
    return 0;
}

static void env_table(void)
{
    if (access("src/common/env_names.c", R_OK) != 0) {
        printf("test_words: the sources are not here, table not checked\n");
        return;
    }
    nftw("src", scan, 16, FTW_PHYS);
    for (size_t i = 0; i < JANAS_N_ENV_NAMES; i++) {
        int found = 0;
        for (int k = 0; k < n_seen && !found; k++)
            found = strcmp(seen[k], JANAS_ENV_NAMES[i]) == 0;
        CHECK(found, "%s is in the table and read nowhere", JANAS_ENV_NAMES[i]);
    }
    setenv("JANAS_EXPERT_BIT", "2", 1);
    setenv("JANAS_EXPERT_BITS", "2", 1);
    CHECK(janas_env_check("test_words") == 1, "one unknown variable");
    unsetenv("JANAS_EXPERT_BIT");
    unsetenv("JANAS_EXPERT_BITS");
}

int main(void)
{
    closest();
    numbers();
    settings();
    env_table();
    if (failures) {
        printf("test_words: %d failures\n", failures);
        return 1;
    }
    printf("test_words: ok\n");
    return 0;
}
