/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * read.c - janas-github reads GitHub's answers (see github.h): issues and
 * pull requests as the search gives them, releases, tags, commits, a
 * repository, a search of repositories. No network here, so that the
 * tests can read answers kept as they came.
 */
#define _GNU_SOURCE /* timegm, strcasecmp */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "github.h"

static void copy(char *to, size_t cap, const char *s)
{
    snprintf(to, cap, "%s", s ? s : "");
}

static const char *str(const struct janas_json *o, const char *key)
{
    const char *s = janas_json_str(janas_json_get(o, key));
    return s && *s ? s : NULL;
}

static long num(const struct janas_json *o, const char *key)
{
    return (long)janas_json_num(janas_json_get(o, key), 0);
}

static int truth(const struct janas_json *o, const char *key)
{
    const struct janas_json *v = janas_json_get(o, key);
    return v && v->type == JANAS_JSON_TRUE;
}

/* The first line of a text, cut to cap */
static void first_line(char *to, size_t cap, const char *s)
{
    s = s ? s : "";
    size_t n = strcspn(s, "\r\n");
    snprintf(to, cap, "%.*s", (int)(n < cap ? n : cap - 1), s);
}

static struct janas_json_doc *parse(const char *text, size_t n, char *err,
                                    size_t err_len)
{
    char why[128];
    struct janas_json_doc *d = janas_json_parse(text, n, why, sizeof why);
    if (!d)
        snprintf(err, err_len, "GitHub: an answer it could not read (%s)", why);
    return d;
}

static const struct janas_json *array(struct janas_json_doc *d, char *err,
                                      size_t err_len)
{
    const struct janas_json *a = janas_json_root(d);
    if (a && a->type == JANAS_JSON_ARRAY)
        return a;
    const char *m = str(a, "message");
    snprintf(err, err_len, "GitHub: %s", m ? m : "not a list");
    return NULL;
}

time_t gh_utc(const char *s)
{
    struct tm tm = {0};
    if (!s || sscanf(s, "%d-%d-%dT%d:%d:%d", &tm.tm_year, &tm.tm_mon,
                     &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec) < 5)
        return -1;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return timegm(&tm);
}

int gh_read_issues(const char *text, size_t n, int max, struct gh_issues *v,
                   char *err, size_t err_len)
{
    memset(v, 0, sizeof *v);
    struct janas_json_doc *d = parse(text, n, err, err_len);
    if (!d)
        return -1;
    const struct janas_json *r = janas_json_root(d);
    const struct janas_json *items = janas_json_get(r, "items");
    if (!items || items->type != JANAS_JSON_ARRAY) {
        const char *m = str(r, "message");
        snprintf(err, err_len, "GitHub: %s", m ? m : "no items");
        janas_json_free(d);
        return -1;
    }
    v->total = num(r, "total_count");
    if (max > GH_ITEMS)
        max = GH_ITEMS;
    for (const struct janas_json *it = items->child; it && v->n < max;
         it = it->next) {
        struct gh_issue *x = &v->i[v->n++];
        x->number = (int)num(it, "number");
        copy(x->title, sizeof x->title, str(it, "title"));
        copy(x->user, sizeof x->user, str(janas_json_get(it, "user"), "login"));
        copy(x->url, sizeof x->url, str(it, "html_url"));
        x->created = gh_utc(str(it, "created_at"));
        x->closed = gh_utc(str(it, "closed_at"));
        x->comments = (int)num(it, "comments");
        x->open = janas_json_is(janas_json_get(it, "state"), "open");
        const struct janas_json *pr = janas_json_get(it, "pull_request");
        x->pr = pr != NULL;
        x->merged = pr && str(pr, "merged_at") != NULL;
        x->draft = truth(it, "draft");
        size_t k = 0;
        const struct janas_json *ls = janas_json_get(it, "labels");
        for (const struct janas_json *l = ls ? ls->child : NULL; l;
             l = l->next) {
            const char *name = str(l, "name");
            if (name && k < sizeof x->labels)
                k += (size_t)snprintf(x->labels + k, sizeof x->labels - k,
                                      "%s%s", k ? ", " : "", name);
        }
    }
    janas_json_free(d);
    return 0;
}

/* Release notes as the user reads them in a terminal: the lines without
   HTML's tags (llama.cpp's open with <details>) and without the empty
   ones, up to cap at a line's end. */
static void notes(char *to, size_t cap, const char *s)
{
    size_t k = 0;
    to[0] = 0;
    while (s && *s && k + 1 < cap) {
        size_t n = strcspn(s, "\r\n");
        char line[512];
        size_t m = 0;
        int in_tag = 0, in_link = 0;
        for (size_t i = 0; i < n && m + 1 < sizeof line; i++) {
            if (s[i] == '<' && i + 5 < n && strncmp(s + i + 1, "http", 4) == 0)
                in_link = 1; /* <https://...>: a link, kept */
            else if (in_link && s[i] == '>')
                in_link = 0;
            else if (s[i] == '<' && i + 1 < n &&
                     (s[i + 1] == '/' || (s[i + 1] >= 'a' && s[i + 1] <= 'z')))
                in_tag = 1;
            else if (in_tag && s[i] == '>')
                in_tag = 0;
            else if (!in_tag)
                line[m++] = s[i];
        }
        while (m && line[m - 1] == ' ')
            m--;
        line[m] = 0;
        if (m) {
            if (k + m + 2 > cap) {
                if (k + 5 < cap)
                    k += (size_t)snprintf(to + k, cap - k, "...\n");
                break;
            }
            k += (size_t)snprintf(to + k, cap - k, "%s\n", line);
        }
        s += n;
        while (*s == '\r' || *s == '\n')
            s++;
    }
    while (k && to[k - 1] == '\n')
        to[--k] = 0;
}

int gh_read_releases(const char *text, size_t n, int max, struct gh_releases *v,
                     char *err, size_t err_len)
{
    memset(v, 0, sizeof *v);
    struct janas_json_doc *d = parse(text, n, err, err_len);
    if (!d)
        return -1;
    const struct janas_json *a = array(d, err, err_len);
    if (max > GH_ITEMS)
        max = GH_ITEMS;
    for (const struct janas_json *it = a ? a->child : NULL; it && v->n < max;
         it = it->next) {
        if (truth(it, "draft"))
            continue;
        struct gh_release *x = &v->r[v->n++];
        copy(x->tag, sizeof x->tag, str(it, "tag_name"));
        const char *name = str(it, "name");
        if (name && strcmp(name, x->tag) != 0)
            copy(x->name, sizeof x->name, name);
        copy(x->author, sizeof x->author,
             str(janas_json_get(it, "author"), "login"));
        copy(x->url, sizeof x->url, str(it, "html_url"));
        notes(x->notes, sizeof x->notes, str(it, "body"));
        x->at = gh_utc(str(it, "published_at"));
        x->pre = truth(it, "prerelease");
        const struct janas_json *as = janas_json_get(it, "assets");
        x->assets = as && as->type == JANAS_JSON_ARRAY ? (int)as->n : 0;
    }
    janas_json_free(d);
    return a ? 0 : -1;
}

int gh_read_tags(const char *text, size_t n, int max, struct gh_releases *v,
                 char *err, size_t err_len)
{
    memset(v, 0, sizeof *v);
    v->tags = 1;
    struct janas_json_doc *d = parse(text, n, err, err_len);
    if (!d)
        return -1;
    const struct janas_json *a = array(d, err, err_len);
    if (max > GH_ITEMS)
        max = GH_ITEMS;
    for (const struct janas_json *it = a ? a->child : NULL; it && v->n < max;
         it = it->next) {
        struct gh_release *x = &v->r[v->n++];
        copy(x->tag, sizeof x->tag, str(it, "name"));
        x->at = -1; /* a tag's answer has no date */
    }
    janas_json_free(d);
    return a ? 0 : -1;
}

int gh_read_commits(const char *text, size_t n, int max, struct gh_commits *v,
                    char *err, size_t err_len)
{
    memset(v, 0, sizeof *v);
    struct janas_json_doc *d = parse(text, n, err, err_len);
    if (!d)
        return -1;
    const struct janas_json *a = array(d, err, err_len);
    if (max > GH_ITEMS)
        max = GH_ITEMS;
    int all = 0;
    for (const struct janas_json *it = a ? a->child : NULL; it; it = it->next) {
        all++;
        if (v->n >= max)
            continue;
        struct gh_commit *x = &v->c[v->n++];
        copy(x->sha, sizeof x->sha, str(it, "sha"));
        const struct janas_json *c = janas_json_get(it, "commit");
        first_line(x->message, sizeof x->message, str(c, "message"));
        const struct janas_json *au = janas_json_get(c, "author");
        copy(x->author, sizeof x->author, str(au, "name"));
        x->at = gh_utc(str(au, "date"));
        copy(x->url, sizeof x->url, str(it, "html_url"));
    }
    v->more = all > v->n;
    janas_json_free(d);
    return a ? 0 : -1;
}

int gh_read_repo(const char *text, size_t n, struct gh_repo *r, char *err,
                 size_t err_len)
{
    memset(r, 0, sizeof *r);
    struct janas_json_doc *d = parse(text, n, err, err_len);
    if (!d)
        return -1;
    const struct janas_json *o = janas_json_root(d);
    if (!str(o, "full_name")) {
        const char *m = str(o, "message");
        snprintf(err, err_len, "GitHub: %s", m ? m : "not a repository");
        janas_json_free(d);
        return -1;
    }
    copy(r->full, sizeof r->full, str(o, "full_name"));
    copy(r->description, sizeof r->description, str(o, "description"));
    copy(r->language, sizeof r->language, str(o, "language"));
    const struct janas_json *l = janas_json_get(o, "license");
    const char *spdx = str(l, "spdx_id");
    copy(r->license, sizeof r->license,
         spdx && strcmp(spdx, "NOASSERTION") != 0 ? spdx : str(l, "name"));
    copy(r->branch, sizeof r->branch, str(o, "default_branch"));
    copy(r->homepage, sizeof r->homepage, str(o, "homepage"));
    copy(r->url, sizeof r->url, str(o, "html_url"));
    size_t k = 0;
    const struct janas_json *t = janas_json_get(o, "topics");
    for (const struct janas_json *x = t ? t->child : NULL; x; x = x->next) {
        const char *s = janas_json_str(x);
        if (s && k < sizeof r->topics)
            k += (size_t)snprintf(r->topics + k, sizeof r->topics - k, "%s%s",
                                  k ? ", " : "", s);
    }
    r->stars = num(o, "stargazers_count");
    r->forks = num(o, "forks_count");
    r->watchers = num(o, "subscribers_count");
    r->open_issues = num(o, "open_issues_count");
    r->pushed = gh_utc(str(o, "pushed_at"));
    r->created = gh_utc(str(o, "created_at"));
    r->archived = truth(o, "archived");
    r->private_ = truth(o, "private");
    r->fork = truth(o, "fork");
    janas_json_free(d);
    return 0;
}

/* "name", or "owner/name" whose name part is name */
static int named(const char *full, const char *name)
{
    const char *slash = full ? strrchr(full, '/') : NULL;
    return slash && strcasecmp(slash + 1, name) == 0;
}

int gh_read_search_repo(const char *text, size_t n, const char *name,
                        char *full, size_t cap, char *err, size_t err_len)
{
    struct janas_json_doc *d = parse(text, n, err, err_len);
    if (!d)
        return -1;
    const struct janas_json *items =
        janas_json_get(janas_json_root(d), "items");
    const char *best = NULL;
    long stars = -1;
    for (const struct janas_json *it = items ? items->child : NULL; it;
         it = it->next) {
        const char *f = str(it, "full_name");
        if (f && named(f, name) && num(it, "stargazers_count") > stars) {
            best = f;
            stars = num(it, "stargazers_count");
        }
    }
    if (!best && items && items->child)
        best = str(items->child, "full_name");
    if (best)
        copy(full, cap, best);
    else
        snprintf(err, err_len, "GitHub has no repository named %s", name);
    janas_json_free(d);
    return best ? 0 : -1;
}

int gh_read_mine(const char *text, size_t n, const char *name, char *full,
                 size_t cap, char *err, size_t err_len)
{
    struct janas_json_doc *d = parse(text, n, err, err_len);
    if (!d)
        return -1;
    const struct janas_json *a = array(d, err, err_len);
    int found = 0;
    for (const struct janas_json *it = a ? a->child : NULL; it && !found;
         it = it->next) {
        const char *f = str(it, "full_name");
        if (f && named(f, name)) {
            copy(full, cap, f);
            found = 1;
        }
    }
    janas_json_free(d);
    return a ? found : -1;
}
