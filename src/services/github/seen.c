/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * seen.c - when the user last looked at a repository's issues or pull
 * requests (see github.h), so that "are there new issues?" has an answer:
 * a line a repository in ~/.config/janas/github-seen.txt, "owner/name what
 * time". Nothing but these lines is kept.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "github.h"

#define LINES_MAX 256

static int path_of(char *buf, size_t cap)
{
    const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
    int n = xdg && *xdg ? snprintf(buf, cap, "%s/janas", xdg)
            : home      ? snprintf(buf, cap, "%s/.config/janas", home)
                        : -1;
    if (n < 0 || (size_t)n + 20 >= cap)
        return -1;
    if (mkdir(buf, 0700) != 0 && errno != EEXIST)
        return -1;
    strcat(buf, "/github-seen.txt");
    return 0;
}

time_t gh_seen(const char *repo, const char *what)
{
    char path[1024], line[512], r[256], w[32];
    long long t;
    time_t at = 0;
    if (path_of(path, sizeof path) != 0)
        return 0;
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "%255s %31s %lld", r, w, &t) == 3 &&
            strcmp(r, repo) == 0 && strcmp(w, what) == 0)
            at = (time_t)t;
    fclose(f);
    return at;
}

void gh_seen_set(const char *repo, const char *what, time_t t)
{
    char path[1024], tmp[1100], line[512], r[256], w[32];
    long long old;
    if (path_of(path, sizeof path) != 0)
        return;
    snprintf(tmp, sizeof tmp, "%s.new", path);
    FILE *in = fopen(path, "r"), *out = fopen(tmp, "w");
    if (!out) {
        if (in)
            fclose(in);
        return;
    }
    /* the others' lines as they were, the newest last (LINES_MAX of
       them: the oldest go) */
    char *keep[LINES_MAX];
    int n = 0;
    while (in && fgets(line, sizeof line, in)) {
        if (sscanf(line, "%255s %31s %lld", r, w, &old) != 3 ||
            (strcmp(r, repo) == 0 && strcmp(w, what) == 0))
            continue;
        if (n == LINES_MAX - 1) {
            free(keep[0]);
            memmove(keep, keep + 1, (size_t)(n - 1) * sizeof *keep);
            n--;
        }
        if ((keep[n] = strdup(line)))
            n++;
    }
    if (in)
        fclose(in);
    for (int i = 0; i < n; i++) {
        fputs(keep[i], out);
        free(keep[i]);
    }
    fprintf(out, "%s %s %lld\n", repo, what, (long long)t);
    if (fclose(out) == 0)
        rename(tmp, path);
    else
        remove(tmp);
}
