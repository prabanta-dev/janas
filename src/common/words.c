/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * words.c - the words a user gives a program, read strictly (see words.h).
 */
#include "common/words.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LONGEST 64 /* longer words are compared by their start */

/* Damerau's distance (two letters swapped count one), case aside */
static int distance(const char *a, const char *b)
{
    size_t na = strlen(a), nb = strlen(b);
    if (na > LONGEST)
        na = LONGEST;
    if (nb > LONGEST)
        nb = LONGEST;
    int d[LONGEST + 1][LONGEST + 1]; /* 17 KB: on the stack, thread-safe */
    for (size_t i = 0; i <= na; i++)
        d[i][0] = (int)i;
    for (size_t j = 0; j <= nb; j++)
        d[0][j] = (int)j;
    for (size_t i = 1; i <= na; i++)
        for (size_t j = 1; j <= nb; j++) {
            int ca = tolower((unsigned char)a[i - 1]);
            int cb = tolower((unsigned char)b[j - 1]);
            int v = d[i - 1][j - 1] + (ca != cb);
            if (d[i - 1][j] + 1 < v)
                v = d[i - 1][j] + 1;
            if (d[i][j - 1] + 1 < v)
                v = d[i][j - 1] + 1;
            if (i > 1 && j > 1 && ca == tolower((unsigned char)b[j - 2]) &&
                tolower((unsigned char)a[i - 2]) == cb &&
                d[i - 2][j - 2] + 1 < v)
                v = d[i - 2][j - 2] + 1;
            d[i][j] = v;
        }
    return d[na][nb];
}

/* one the start of the other, the shorter of four letters or more */
static int starts(const char *a, const char *b)
{
    size_t na = strlen(a), nb = strlen(b), n = na < nb ? na : nb;
    if (n < 4 || na == nb)
        return 0;
    for (size_t i = 0; i < n; i++)
        if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i]))
            return 0;
    return 1;
}

const char *janas_closest(const char *w, const char *const *names, size_t n)
{
    if (!w || !*w)
        return NULL;
    size_t len = strlen(w);
    int most = len >= 12 ? 3 : len >= 6 ? 2 : 1;
    const char *best = NULL;
    int best_d = most + 1;
    for (size_t i = 0; i < n && names[i]; i++) {
        int d = distance(w, names[i]);
        if (d < best_d) {
            best_d = d;
            best = names[i];
        }
    }
    for (size_t i = 0; !best && i < n && names[i]; i++)
        if (starts(w, names[i]))
            best = names[i];
    return best;
}

void janas_unknown_word(char *buf, size_t cap, const char *w, const char *what,
                        const char *const *names, size_t n)
{
    const char *near = janas_closest(w, names, n);
    if (near)
        snprintf(buf, cap, "%s is not %s (did you mean %s?)", w, what, near);
    else
        snprintf(buf, cap, "%s is not %s", w, what);
}

int janas_word_int(const char *v, long long lo, long long hi, long long *out,
                   char *why, size_t why_len)
{
    char *end;
    errno = 0;
    long long x =
        v && *v && !isspace((unsigned char)*v) ? strtoll(v, &end, 10) : 0;
    if (!v || !*v || isspace((unsigned char)*v) || *end || errno) {
        if (v)
            snprintf(why, why_len, "%s is not a whole number", v);
        else
            snprintf(why, why_len, "a value is missing");
        return -1;
    }
    if (x < lo || x > hi) {
        snprintf(why, why_len, "%s is %s %lld", v, x < lo ? "below" : "above",
                 x < lo ? lo : hi);
        return -1;
    }
    *out = x;
    return 0;
}

int janas_word_real(const char *v, double lo, double hi, double *out, char *why,
                    size_t why_len)
{
    char *end;
    double x = v && *v && !isspace((unsigned char)*v) ? strtod(v, &end) : 0;
    if (!v || !*v || isspace((unsigned char)*v) || *end || !isfinite(x)) {
        if (v)
            snprintf(why, why_len, "%s is not a number", v);
        else
            snprintf(why, why_len, "a value is missing");
        return -1;
    }
    if (x < lo || x > hi) {
        snprintf(why, why_len, "%s is %s %g", v, x < lo ? "below" : "above",
                 x < lo ? lo : hi);
        return -1;
    }
    *out = x;
    return 0;
}

int janas_word_choice(const char *v, const char *const *names, size_t n,
                      char *why, size_t why_len)
{
    for (size_t i = 0; v && i < n && names[i]; i++)
        if (strcmp(v, names[i]) == 0)
            return (int)i;
    char list[256] = "";
    size_t k = 0;
    for (size_t i = 0; i < n && names[i] && k < sizeof list; i++) {
        int m = snprintf(list + k, sizeof list - k, "%s%s", i ? ", " : "",
                         names[i]);
        if (m < 0)
            break;
        k += (size_t)m;
    }
    /* a word mistyped has a nearest; a number (3 for 2, 4 or 6) has not */
    int letters = 0;
    for (const char *c = v; c && *c && !letters; c++)
        letters = isalpha((unsigned char)*c);
    const char *near = letters ? janas_closest(v, names, n) : NULL;
    if (near)
        snprintf(why, why_len, "%s is not one of %s (did you mean %s?)",
                 v ? v : "nothing", list, near);
    else
        snprintf(why, why_len, "%s is not one of %s", v ? v : "nothing", list);
    return -1;
}
