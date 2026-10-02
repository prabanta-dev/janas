/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * data.c - what janas-github read, as the data of its layouts (see
 * github.h, layouts.c and common/template.h). Every field is written, even
 * empty: a layout looks a name up in the item, then in what holds it, and
 * an issue without labels would show another's.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "github.h"

void gh_jstr(struct janas_buf *b, const char *key, const char *v)
{
    if (!v)
        v = "";
    janas_buf_printf(b, ", \"%s\": ", key);
    janas_json_write_str(b, v, strlen(v));
}

void gh_jwhen(struct janas_buf *b, const char *key, time_t t)
{
    if (t < 0) {
        janas_buf_printf(b,
                         ", \"%s\": {\"day\": 0, \"mon\": 0, \"year\": 0, "
                         "\"at\": \"\", \"ago_d\": 0}",
                         key);
        return;
    }
    struct tm lt, nt;
    time_t now = time(NULL);
    localtime_r(&t, &lt);
    localtime_r(&now, &nt);
    long ago = (long)((now - t) / 86400);
    janas_buf_printf(b,
                     ", \"%s\": {\"day\": %d, \"mon\": %d, \"year\": %d, "
                     "\"at\": \"%02d:%02d\", \"ago_d\": %ld}",
                     key, lt.tm_mday, lt.tm_mon + 1,
                     lt.tm_year == nt.tm_year ? 0 : lt.tm_year + 1900,
                     lt.tm_hour, lt.tm_min, ago < 0 ? 0 : ago);
}

static void head(struct janas_buf *b, const char *repo, const char *how)
{
    janas_buf_puts(b, "{\"repo\": ");
    janas_json_write_str(b, repo, strlen(repo));
    gh_jstr(b, "how", how);
}

void gh_issues_data(struct janas_buf *b, const char *repo, const char *how,
                    int pr, const char *state, time_t since, int first,
                    const struct gh_issues *v)
{
    head(b, repo, how);
    char kind[32]; /* "open_issues", "new_pulls": a word of the layout */
    snprintf(kind, sizeof kind, "%s_%s", since > 0 ? "new" : state,
             pr ? "pulls" : "issues");
    gh_jstr(b, "kind", kind);
    janas_buf_printf(b, ", \"total\": %ld, \"shown\": %d, \"more\": %s",
                     v->total, v->n, v->total > v->n ? "true" : "false");
    janas_buf_printf(b, ", \"new\": %s, \"first\": %s",
                     since > 0 ? "true" : "false", first ? "true" : "false");
    gh_jwhen(b, "since", since > 0 ? since : -1);
    int open_n = 0;
    for (int i = 0; i < v->n; i++)
        open_n += v->i[i].open;
    janas_buf_printf(b, ", \"open_n\": %d, \"closed_n\": %d", open_n,
                     v->n - open_n);
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < v->n; i++) {
        const struct gh_issue *x = &v->i[i];
        janas_buf_printf(b, "%s{\"n\": %d", i ? ", " : "", x->number);
        gh_jstr(b, "title", x->title);
        gh_jstr(b, "user", x->user);
        gh_jwhen(b, "created", x->created);
        janas_buf_printf(b, ", \"comments\": %d", x->comments);
        gh_jstr(b, "labels", x->labels);
        gh_jstr(b, "status",
                x->merged             ? "merged"
                : x->draft && x->open ? "draft"
                : x->open             ? "open"
                                      : "closed");
        gh_jstr(b, "url", x->url);
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]}");
}

void gh_releases_data(struct janas_buf *b, const char *repo, const char *how,
                      const struct gh_releases *v)
{
    head(b, repo, how);
    janas_buf_printf(b, ", \"n\": %d, \"one\": %s, \"tags\": %s", v->n,
                     v->n == 1 ? "true" : "false", v->tags ? "true" : "false");
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < v->n; i++) {
        const struct gh_release *x = &v->r[i];
        janas_buf_printf(b, "%s{\"tag\": ", i ? ", " : "");
        janas_json_write_str(b, x->tag, strlen(x->tag));
        gh_jstr(b, "name", x->name);
        gh_jwhen(b, "when", x->at);
        gh_jstr(b, "author", x->author);
        janas_buf_printf(b, ", \"pre\": %s, \"assets\": %d",
                         x->pre ? "true" : "false", x->assets);
        gh_jstr(b, "notes", x->notes);
        gh_jstr(b, "url", x->url);
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]}");
}

void gh_commits_data(struct janas_buf *b, const char *repo, const char *how,
                     const char *branch, int days, const struct gh_commits *v)
{
    head(b, repo, how);
    gh_jstr(b, "branch", branch);
    janas_buf_printf(b, ", \"days\": %d, \"n\": %d, \"more\": %s", days, v->n,
                     v->more ? "true" : "false");
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < v->n; i++) {
        const struct gh_commit *x = &v->c[i];
        janas_buf_printf(b, "%s{\"sha\": ", i ? ", " : "");
        janas_json_write_str(b, x->sha, strlen(x->sha));
        gh_jstr(b, "message", x->message);
        gh_jstr(b, "author", x->author);
        gh_jwhen(b, "when", x->at);
        gh_jstr(b, "url", x->url);
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]}");
}

void gh_repo_data(struct janas_buf *b, const char *how, const struct gh_repo *r)
{
    head(b, r->full, how);
    gh_jstr(b, "description", r->description);
    gh_jstr(b, "language", r->language);
    gh_jstr(b, "license", r->license);
    gh_jstr(b, "branch", r->branch);
    gh_jstr(b, "homepage", r->homepage);
    gh_jstr(b, "topics", r->topics);
    gh_jstr(b, "url", r->url);
    janas_buf_printf(b,
                     ", \"stars\": %ld, \"forks\": %ld, \"watchers\": %ld, "
                     "\"open_issues\": %ld",
                     r->stars, r->forks, r->watchers, r->open_issues);
    gh_jwhen(b, "pushed", r->pushed);
    gh_jwhen(b, "created", r->created);
    janas_buf_printf(b, ", \"archived\": %s, \"private\": %s, \"fork\": %s}",
                     r->archived ? "true" : "false",
                     r->private_ ? "true" : "false",
                     r->fork ? "true" : "false");
}
