/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * bench_report.c - see bench_report.h.
 */
#include "common/bench_report.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A field's text after its label, to the end of the line, trimmed. */
static void field(const char *l, size_t n, char *out, size_t cap)
{
    while (n && (l[0] == ' ' || l[0] == '\t'))
        l++, n--;
    while (n && (unsigned char)l[n - 1] <= ' ')
        n--;
    if (n >= cap)
        n = cap - 1;
    memcpy(out, l, n);
    out[n] = 0;
    for (char *p = out; *p; p++) /* a table's cells */
        if (*p == '|')
            *p = '/';
}

/* The text up to " (" or ", ": a name without what follows it. */
static void cut_at(char *s, const char *const *stops, int n_stops)
{
    for (int i = 0; i < n_stops; i++) {
        char *p = strstr(s, stops[i]);
        if (p)
            *p = 0;
    }
}

/* s without every occurrence of w */
static void drop(char *s, const char *w)
{
    size_t n = strlen(w);
    char *p;
    while ((p = strstr(s, w)))
        memmove(p, p + n, strlen(p + n) + 1);
}

/* A processor's or a card's name as people say it: "Intel(R) Core(TM)
   i5-1135G7 @ 2.40GHz" is "Intel Core i5-1135G7", "AMD Ryzen 9 9950X3D
   16-Core Processor" "AMD Ryzen 9 9950X3D" */
static void clean_hw(char *s)
{
    static const char *const marks[] = {"(R)",       "(TM)",      "11th Gen ",
                                        "12th Gen ", "13th Gen ", "14th Gen "};
    for (size_t i = 0; i < sizeof marks / sizeof *marks; i++)
        drop(s, marks[i]);
    static const char *const s2[] = {" w/ ", " with Radeon", " @ "};
    cut_at(s, s2, 3);
    char *c = strstr(s, "-Core Processor");
    if (c) { /* back over the count and its space */
        while (c > s && c[-1] >= '0' && c[-1] <= '9')
            c--;
        if (c > s && c[-1] == ' ')
            c--;
        *c = 0;
    }
    size_t n = strlen(s);
    if (n > 4 && !strcmp(s + n - 4, " GPU"))
        s[n - 4] = 0;
}

/* A model's name as the catalog writes it: the file's own words joined by
   dashes, without what tells the variant ("Qwen3 Next 80B A3B Instruct" is
   "Qwen3-Next-80B-A3B") */
static void clean_model(char *s)
{
    for (char *p = s; *p; p++)
        if (*p == ' ')
            *p = '-';
    static const char *const tails[] = {"-Instruct", "-Awq", "-GSQ-RCO"};
    for (size_t i = 0; i < sizeof tails / sizeof *tails; i++)
        drop(s, tails[i]);
    size_t n = strlen(s);
    if (n > 3 && !strcmp(s + n - 3, "-It"))
        s[n - 2] = 'i';
}

static double number_before(const char *l, const char *word)
{
    const char *p = strstr(l, word);
    if (!p)
        return 0;
    while (p > l && (p[-1] == ' '))
        p--;
    while (p > l && ((p[-1] >= '0' && p[-1] <= '9') || p[-1] == '.'))
        p--;
    return atof(p);
}

int janas_bench_report_scan(const char *text, size_t len,
                            int (*fn)(void *ctx,
                                      const struct janas_bench_row *r),
                            void *ctx)
{
    struct janas_bench_row r;
    int open = 0, found = 0;
    for (size_t i = 0; i < len;) {
        size_t e = i;
        while (e < len && text[e] != '\n')
            e++;
        const char *l = text + i;
        size_t n = e - i;
        i = e + 1;
        while (n && (*l == ' ' || *l == '>' || *l == '\t'))
            l++, n--; /* quoted, or indented */
        char line[1024];
        if (n >= sizeof line)
            n = sizeof line - 1;
        memcpy(line, l, n);
        line[n] = 0;
        if (n > 12 && !strncmp(line, "janas-bench ", 12) && strchr(line, ',')) {
            memset(&r, 0, sizeof r);
            open = 1;
            const char *v = line + 12;
            size_t vn = strcspn(v, ", ");
            if (vn >= sizeof r.version)
                vn = sizeof r.version - 1;
            memcpy(r.version, v, vn);
            r.version[vn] = 0;
            continue;
        }
        if (!open)
            continue;
        if (!strncmp(line, "system:", 7)) {
            field(line + 7, strlen(line + 7), r.sys, sizeof r.sys);
            r.sys[strcspn(r.sys, " ")] = 0; /* Linux, Windows */
        } else if (!strncmp(line, "CPU:", 4)) {
            field(line + 4, strlen(line + 4), r.cpu, sizeof r.cpu);
            static const char *const s[] = {" (", "  "};
            cut_at(r.cpu, s, 2);
            clean_hw(r.cpu);
        } else if (!strncmp(line, "memory:", 7)) {
            field(line + 7, strlen(line + 7), r.ram, sizeof r.ram);
            static const char *const s[] = {","};
            cut_at(r.ram, s, 1);
        } else if (!strncmp(line, "GPU:", 4)) {
            field(line + 4, strlen(line + 4), r.gpu, sizeof r.gpu);
            if (!strncmp(r.gpu, "none", 4))
                snprintf(r.gpu, sizeof r.gpu, "none");
            static const char *const s[] = {" ("};
            cut_at(r.gpu, s, 1);
            clean_hw(r.gpu);
        } else if (!strncmp(line, "model:", 6)) {
            field(line + 6, strlen(line + 6), r.model, sizeof r.model);
            static const char *const s[] = {" ("};
            cut_at(r.model, s, 1);
            clean_model(r.model);
        } else if (!strncmp(line, "result:", 7)) {
            open = 0;
            r.prefill = number_before(line, " tok/s, decode");
            const char *dp = strstr(line, "decode ");
            r.decode = dp ? atof(dp + 7) : 0;
            r.mtp_greedy = number_before(line, " tok/s greedy");
            r.mtp_sampled = number_before(line, " tok/s sampled");
            if (r.decode > 0 && r.model[0] && r.cpu[0]) {
                found++;
                if (fn(ctx, &r) != 0)
                    return -1;
            }
        }
    }
    return found;
}
