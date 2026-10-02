/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * repo.c - the repository the user means (see github.h): given whole
 * ("owner/name", a link), or by its name alone, first among the user's own
 * (with a token), then the one of that name with the most stars - the
 * answer says which it took.
 */
#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "github.h"

/* A name GitHub allows: letters, digits, "-", "_", "." */
static int plain(const char *s, size_t n)
{
    if (!n || n > 100)
        return 0;
    for (size_t i = 0; i < n; i++)
        if (!isalnum((unsigned char)s[i]) && !strchr("-_.", s[i]))
            return 0;
    return 1;
}

int gh_repo_find(const char *q, char *full, size_t cap, char *how,
                 size_t how_cap, char *err, size_t err_len)
{
    how[0] = 0;
    while (q && isspace((unsigned char)*q))
        q++;
    if (!q || !*q) {
        snprintf(err, err_len, "No repository given");
        return -1;
    }
    /* a link: github.com/owner/name, whatever follows */
    const char *g = strstr(q, "github.com/");
    if (g)
        q = g + 11;
    const char *slash = strchr(q, '/');
    if (slash) {
        size_t on = (size_t)(slash - q);
        const char *name = slash + 1;
        size_t nn = strcspn(name, "/?# \t");
        if (nn > 4 && strncmp(name + nn - 4, ".git", 4) == 0)
            nn -= 4;
        if (plain(q, on) && plain(name, nn)) {
            snprintf(full, cap, "%.*s/%.*s", (int)on, q, (int)nn, name);
            return 0;
        }
        snprintf(err, err_len, "\"%s\" is not a repository's name", q);
        return -1;
    }
    size_t n = strlen(q);
    while (n && isspace((unsigned char)q[n - 1]))
        n--;
    char name[128];
    snprintf(name, sizeof name, "%.*s", (int)(n < 127 ? n : 127), q);
    struct janas_buf b = {0};
    if (gh_token_kind() && plain(name, strlen(name)) &&
        gh_get("/user/repos?per_page=100&sort=pushed", &b, err, err_len) ==
            200 &&
        gh_read_mine(b.p, b.n, name, full, cap, err, err_len) == 1) {
        janas_buf_free(&b);
        snprintf(how, how_cap, "yours");
        return 0;
    }
    janas_buf_free(&b);
    struct janas_buf path = {0};
    janas_buf_puts(&path, "/search/repositories?q=");
    gh_url_put(&path, name);
    janas_buf_puts(&path, "+in:name&sort=stars&per_page=10");
    int st = path.oom ? -1 : gh_get(path.p, &b, err, err_len);
    janas_buf_free(&path);
    int r = st == 200
                ? gh_read_search_repo(b.p, b.n, name, full, cap, err, err_len)
                : -1;
    janas_buf_free(&b);
    if (r == 0)
        snprintf(how, how_cap, "found by its name");
    return r;
}
