/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * fetch.c - janas-github asks GitHub's REST API (see github.h). Without a
 * token GitHub answers 60 requests an hour and 10 searches a minute; with
 * one 5,000 and 30, and the user's private repositories too. The token is
 * GH_TOKEN's or GITHUB_TOKEN's, else the gh command's (gh auth token): it
 * is kept in memory and never written anywhere. Answers are kept a
 * minute, so that a question asked again costs nothing.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/https_get.h"
#include "github.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

#define API "https://api.github.com"
#define KEEP_S 60
#define N_KEPT 32

static struct {
    char *path;
    time_t at;
    struct janas_buf body;
} kept[N_KEPT];

static char token[256];
static int kind = -1; /* not looked for yet */

/* A token is letters, digits and underscores: anything else is not one
   (a message of gh's, a line end). */
static int take(const char *s, int k)
{
    size_t n = s ? strlen(s) : 0;
    while (n && isspace((unsigned char)s[n - 1]))
        n--;
    if (n < 20 || n >= sizeof token)
        return 0;
    for (size_t i = 0; i < n; i++)
        if (!isalnum((unsigned char)s[i]) && s[i] != '_')
            return 0;
    memcpy(token, s, n);
    token[n] = 0;
    kind = k;
    return 1;
}

int gh_token_kind(void)
{
    if (kind >= 0)
        return kind;
    kind = 0;
    if (take(getenv("GH_TOKEN"), 1) || take(getenv("GITHUB_TOKEN"), 1))
        return kind;
    FILE *p = popen("gh auth token 2>/dev/null", "r");
    if (p) {
        char line[512] = "";
        if (!fgets(line, sizeof line, p))
            line[0] = 0;
        pclose(p);
        take(line, 2);
        memset(line, 0, sizeof line);
    }
    return kind;
}

void gh_url_put(struct janas_buf *b, const char *s)
{
    static const char hex[] = "0123456789ABCDEF";
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
            janas_buf_put(b, (const char *)&c, 1);
        else {
            char e[3] = {'%', hex[c >> 4], hex[c & 15]};
            janas_buf_put(b, e, 3);
        }
    }
}

static int ask(const char *url, int with_token, struct janas_buf *out,
               char *err, size_t err_len)
{
    char auth[300];
    const char *h[4] = {"User-Agent: janas-github/" JANAS_VERSION
                        " (+https://github.com/prabanta-dev/janas)",
                        "Accept: application/vnd.github+json",
                        "X-GitHub-Api-Version: 2022-11-28", auth};
    snprintf(auth, sizeof auth, "Authorization: Bearer %s", token);
    int status = janas_https_get(url, h, with_token ? 4 : 3, out, err, err_len);
    memset(auth, 0, sizeof auth);
    return status;
}

int gh_get(const char *path, struct janas_buf *out, char *err, size_t err_len)
{
    time_t now = time(NULL);
    int slot = 0;
    for (int i = 0; i < N_KEPT; i++) {
        if (kept[i].path && strcmp(kept[i].path, path) == 0 &&
            now - kept[i].at < KEEP_S) {
            out->n = 0;
            janas_buf_put(out, kept[i].body.p, kept[i].body.n);
            janas_buf_put(out, "", 1);
            out->n--;
            return out->oom ? -1 : 200;
        }
        if (kept[i].at < kept[slot].at)
            slot = i;
    }
    char url[2048];
    snprintf(url, sizeof url, "%s%s", API, path);
    int tok = gh_token_kind() > 0;
    int status = ask(url, tok, out, err, err_len);
    if (status == 401 && tok) { /* a token GitHub no longer takes */
        kind = 0;
        memset(token, 0, sizeof token);
        status = ask(url, 0, out, err, err_len);
    }
    if (status == 200 && !out->oom) {
        free(kept[slot].path);
        janas_buf_free(&kept[slot].body);
        kept[slot].path = strdup(path);
        kept[slot].at = now;
        janas_buf_put(&kept[slot].body, out->p, out->n);
        return 200;
    }
    if (status == 403 || status == 429)
        snprintf(err, err_len,
                 "GitHub refused (HTTP %d): its limit of requests is "
                 "reached (%s), or this is not allowed",
                 status,
                 gh_token_kind() ? "5,000 an hour, 30 searches a minute"
                                 : "60 an hour and 10 searches a minute "
                                   "without a token: gh auth login gives one");
    else if (status == 404)
        snprintf(err, err_len,
                 "GitHub has nothing there (HTTP 404): no such repository, "
                 "or a private one%s",
                 gh_token_kind() ? " of someone else" : " (no token is used)");
    else if (status == 422)
        snprintf(err, err_len, "GitHub could not do this search (HTTP 422)");
    else if (status > 0 && status != 200)
        snprintf(err, err_len, "GitHub: HTTP %d", status);
    return status;
}

void gh_fetch_free(void)
{
    for (int i = 0; i < N_KEPT; i++) {
        free(kept[i].path);
        janas_buf_free(&kept[i].body);
    }
    memset(kept, 0, sizeof kept);
    memset(token, 0, sizeof token);
}
