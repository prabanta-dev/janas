/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * repo.c - the repository the user means (see git.h): a path, a name
 * listed in ~/.config/janas/git-repos.txt, or the directory janas-git runs
 * in (janas-chat's). git itself says where its top is.
 */
#define _GNU_SOURCE /* strcasecmp */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "git.h"

void gt_short_path(const char *path, char *out, size_t cap)
{
    const char *home = getenv("HOME");
    size_t h = home ? strlen(home) : 0;
    if (h > 1 && strncmp(path, home, h) == 0 && (path[h] == '/' || !path[h]))
        snprintf(out, cap, "~%s", path + h);
    else
        snprintf(out, cap, "%s", path);
}

/* "~/x" with the home's path */
static void expand(const char *q, char *out, size_t cap)
{
    const char *home = getenv("HOME");
    if (q[0] == '~' && (q[1] == '/' || !q[1]) && home)
        snprintf(out, cap, "%s%s", home, q + 1);
    else
        snprintf(out, cap, "%s", q);
}

/* A name of the list: its path into out. 1, 0 when not there. */
static int listed(const char *name, char *out, size_t cap)
{
    char file[1024], line[1200], path[1100];
    const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
    if (xdg && *xdg)
        snprintf(file, sizeof file, "%s/janas/git-repos.txt", xdg);
    else if (home)
        snprintf(file, sizeof file, "%s/.config/janas/git-repos.txt", home);
    else
        return 0;
    FILE *f = fopen(file, "r");
    if (!f)
        return 0;
    int found = 0;
    while (!found && fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0] || line[0] == '#')
            continue;
        expand(line, path, sizeof path);
        size_t n = strlen(path);
        while (n > 1 && path[n - 1] == '/')
            path[--n] = 0;
        const char *last = strrchr(path, '/');
        if (strcasecmp(last ? last + 1 : path, name) == 0) {
            snprintf(out, cap, "%s", path);
            found = 1;
        }
    }
    fclose(f);
    return found;
}

int gt_repo_find(const char *q, char *top, size_t cap, char *why,
                 size_t why_len)
{
    char dir[PATH_MAX + 64];
    top[0] = 0;
    if (!q || !*q) {
        if (!getcwd(dir, sizeof dir)) {
            snprintf(why, why_len, "no repository given");
            return -1;
        }
    } else if (q[0] == '/' || q[0] == '~' || q[0] == '.') {
        expand(q, dir, sizeof dir);
    } else if (!strchr(q, '/') && listed(q, dir, sizeof dir)) {
        ;
    } else if (strchr(q, '/') || access(q, F_OK) == 0) {
        expand(q, dir, sizeof dir); /* relative to where janas-git runs */
    } else {
        snprintf(why, why_len,
                 "no repository named %s: give its path, or list it in "
                 "~/.config/janas/git-repos.txt",
                 q);
        return -1;
    }
    char real[PATH_MAX];
    if (!realpath(dir, real)) {
        snprintf(why, why_len, "%s is not there", q && *q ? q : dir);
        return -1;
    }
    const char *argv[] = {"rev-parse", "--show-toplevel", NULL};
    struct janas_buf out = {0}, err = {0};
    int st = gt_run(real, argv, 10000, &out, &err, why, why_len);
    if (st == 0 && out.n) {
        out.p[strcspn(out.p, "\n")] = 0;
        snprintf(top, cap, "%s", out.p);
    } else if (st > 0) {
        char s[PATH_MAX];
        gt_short_path(real, s, sizeof s);
        snprintf(why, why_len, "%s is not in a git repository", s);
    }
    janas_buf_free(&out);
    janas_buf_free(&err);
    return st == 0 && top[0] ? 0 : -1;
}
