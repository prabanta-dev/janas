/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * github.h - janas-github, GitHub as an MCP service that only reads: what
 * its parts share. fetch.c asks GitHub's REST API (with the token of the
 * gh command when there is one), repo.c finds a repository by its name,
 * read.c reads the answers, seen.c remembers when the user last looked,
 * data.c and layouts.c write the answers as data with a layout, tools.c
 * offers the tools.
 */
#ifndef JANAS_GITHUB_H
#define JANAS_GITHUB_H

#include <stddef.h>
#include <time.h>

#include "llm/json.h"

/* ---- fetch.c ---- */

/* GET of a path of the API ("/repos/o/r"), the body into out; answers
   kept a minute. The HTTP status, or -1 with the reason in err; a status
   that is not 200 with what it means in err. */
int gh_get(const char *path, struct janas_buf *out, char *err, size_t err_len);
void gh_fetch_free(void);
/* Whether a token is used: 1 from GH_TOKEN or GITHUB_TOKEN, 2 from the gh
   command, 0 none (60 requests an hour). */
int gh_token_kind(void);
/* s into b as a part of a URL's query, escaped. */
void gh_url_put(struct janas_buf *b, const char *s);

/* ---- read.c: the answers (no network: the tests read kept ones) ---- */

/* A text of the API ("2026-10-02T11:18:31Z") as a time; -1 when none. */
time_t gh_utc(const char *s);

#define GH_ITEMS 10

struct gh_issue {
    int number;
    char title[200], user[64], labels[160], url[160];
    time_t created, closed;
    int comments, pr, draft, merged, open;
};

struct gh_issues {
    long total; /* all those the search found */
    struct gh_issue i[GH_ITEMS];
    int n;
};

struct gh_release {
    char tag[64], name[128], author[64], url[200];
    char notes[640]; /* the first lines, without HTML's tags */
    time_t at;
    int pre, assets;
};

struct gh_releases {
    struct gh_release r[GH_ITEMS];
    int n;
    int tags; /* none: the repository's tags instead */
};

struct gh_commit {
    char sha[8], message[200], author[64], url[160];
    time_t at;
};

struct gh_commits {
    struct gh_commit c[GH_ITEMS];
    int n, more; /* more: the page was full */
};

struct gh_repo {
    char full[128], description[400], language[48], license[64];
    char branch[64], homepage[160], topics[200], url[160];
    long stars, forks, watchers, open_issues; /* issues: with the PRs */
    time_t pushed, created;
    int archived, private_, fork;
};

/* One read each, 0 or -1 with the reason in err. */
int gh_read_issues(const char *text, size_t n, int max, struct gh_issues *v,
                   char *err, size_t err_len);
int gh_read_releases(const char *text, size_t n, int max, struct gh_releases *v,
                     char *err, size_t err_len);
int gh_read_tags(const char *text, size_t n, int max, struct gh_releases *v,
                 char *err, size_t err_len);
int gh_read_commits(const char *text, size_t n, int max, struct gh_commits *v,
                    char *err, size_t err_len);
int gh_read_repo(const char *text, size_t n, struct gh_repo *r, char *err,
                 size_t err_len);
/* A search of repositories: the one named name (case aside) with the most
   stars, else the first; its full name into full. 0, or -1. */
int gh_read_search_repo(const char *text, size_t n, const char *name,
                        char *full, size_t cap, char *err, size_t err_len);
/* The user's own repositories (/user/repos): the full name of the one
   named name into full. 1 found, 0 not, -1 on error. */
int gh_read_mine(const char *text, size_t n, const char *name, char *full,
                 size_t cap, char *err, size_t err_len);

/* ---- repo.c ---- */

/* The repository the user means: "owner/name", a github.com link, or a
   name alone - among the user's repositories (with a token), else the
   one of that name with the most stars. full: "owner/name"; how: "" when
   given whole, else how it was chosen ("yours", "found by its name").
   0, or -1 with the reason in err. */
int gh_repo_find(const char *q, char *full, size_t cap, char *how,
                 size_t how_cap, char *err, size_t err_len);

/* ---- seen.c ---- */

/* When the user last looked at what of repo (0: never); and now. */
time_t gh_seen(const char *repo, const char *what);
void gh_seen_set(const char *repo, const char *what, time_t t);

/* ---- data.c ---- */

void gh_jstr(struct janas_buf *b, const char *key, const char *v);
/* , "key": {"day", "mon", "year" (0 this year), "at", "ago_d"} */
void gh_jwhen(struct janas_buf *b, const char *key, time_t t);
void gh_issues_data(struct janas_buf *b, const char *repo, const char *how,
                    int pr, const char *state, time_t since, int first,
                    const struct gh_issues *v);
void gh_releases_data(struct janas_buf *b, const char *repo, const char *how,
                      const struct gh_releases *v);
void gh_commits_data(struct janas_buf *b, const char *repo, const char *how,
                     const char *branch, int days, const struct gh_commits *v);
void gh_repo_data(struct janas_buf *b, const char *how,
                  const struct gh_repo *r);

/* ---- layouts.c ---- */

extern const char GH_ISSUES_LAYOUT[], GH_ISSUES_BRIEF[];
extern const char GH_RELEASES_LAYOUT[], GH_RELEASES_BRIEF[];
extern const char GH_COMMITS_LAYOUT[], GH_COMMITS_BRIEF[];
extern const char GH_REPO_LAYOUT[], GH_REPO_BRIEF[];

/* ---- tools.c ---- */

void gh_tools_list(void *ctx, struct janas_buf *b);
int gh_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);

#endif
