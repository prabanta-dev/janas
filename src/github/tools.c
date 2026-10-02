/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - janas-github's tools (see github.h): github_issues,
 * github_releases, github_commits and github_repo. They only read, and say
 * so (readOnlyHint): a client may run them without asking.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "common/mcp_server.h"
#include "common/template.h"
#include "github.h"

#define DAY 86400
#define FIRST_DAYS 7 /* "new", asked the first time: the last week */

#define REPO_ARG                                                               \
    "\"repo\": {\"type\": \"string\", \"description\": \"owner/name, a "       \
    "github.com link, or a name alone (\\\"Janas\\\", \\\"llama.cpp\\\"): "    \
    "the user's own first, else the one with the most stars\"}"

#define READ_ONLY                                                              \
    "\"annotations\": {\"readOnlyHint\": true, \"openWorldHint\": true}"

void gh_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    janas_buf_puts(
        b,
        "\"tools\": [{\"name\": \"github_issues\", \"title\": \"Issues and pull "
        "requests\", \"description\": \"A repository's issues or pull "
        "requests, the newest first, with how many there are.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": "
        "{" REPO_ARG ", \"kind\": {\"type\": \"string\", \"enum\": "
        "[\"issues\", \"pulls\"]}, \"state\": {\"type\": \"string\", "
        "\"enum\": [\"open\", \"closed\", \"all\"]}, \"new\": {\"type\": "
        "\"boolean\", \"description\": \"only those opened since the user "
        "last asked (\\\"are there new issues?\\\")\"}, \"days\": "
        "{\"type\": \"number\", \"description\": \"only those opened in "
        "the last days\"}}, \"required\": [\"repo\"], "
        "\"additionalProperties\": false}, " READ_ONLY "}, "
        "{\"name\": \"github_releases\", \"title\": \"Releases\", "
        "\"description\": \"A repository's latest releases (its tags when "
        "it has none), with their notes.\", \"inputSchema\": {\"type\": "
        "\"object\", \"properties\": {" REPO_ARG ", \"count\": {\"type\": "
        "\"number\", \"description\": \"1 for the latest; at most 10\"}}, "
        "\"required\": [\"repo\"], \"additionalProperties\": false}, "
        "" READ_ONLY "}, "
        "{\"name\": \"github_commits\", \"title\": \"Recent commits\", "
        "\"description\": \"A branch's commits of the last days, the "
        "newest first.\", \"inputSchema\": {\"type\": \"object\", "
        "\"properties\": {" REPO_ARG ", \"branch\": {\"type\": "
        "\"string\", \"description\": \"leave it out for the main "
        "one\"}, \"days\": {\"type\": \"number\", \"description\": "
        "\"default 7\"}, \"author\": {\"type\": \"string\", "
        "\"description\": \"a GitHub login or an e-mail\"}}, "
        "\"required\": [\"repo\"], \"additionalProperties\": false}, "
        "" READ_ONLY "}, "
        "{\"name\": \"github_repo\", \"title\": \"A repository\", "
        "\"description\": \"What a repository is: description, stars, "
        "language, licence, last push.\", \"inputSchema\": {\"type\": "
        "\"object\", \"properties\": {" REPO_ARG "}, \"required\": "
        "[\"repo\"], \"additionalProperties\": false}, " READ_ONLY "}]");
}

static const char *arg_str(const struct janas_json *args, const char *name)
{
    const char *s = janas_json_str(janas_json_get(args, name));
    return s && *s ? s : NULL;
}

static double arg_num(const struct janas_json *args, const char *name,
                      double def)
{
    const struct janas_json *v = janas_json_get(args, name);
    return v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, def) : def;
}

static int arg_bool(const struct janas_json *args, const char *name)
{
    const struct janas_json *v = janas_json_get(args, name);
    return v && v->type == JANAS_JSON_TRUE;
}

static int fail(struct janas_buf *b, const char *fmt, const char *a,
                const char *c)
{
    struct janas_buf out = {0};
    janas_buf_printf(&out, fmt, a, c);
    janas_mcps_text_result(b, out.p ? out.p : "", out.n, 1);
    janas_buf_free(&out);
    return 0;
}

static int answer(struct janas_buf *b, const char *name, struct janas_buf *d,
                  const char *layout, const char *brief)
{
    char err[300];
    int oom = d->oom;
    int r = oom ? -1
                : janas_tpl_result(b, name, d->p, d->n, layout, brief, err,
                                   sizeof err);
    janas_buf_free(d);
    if (r != 0)
        return fail(b, "The answer could not be written: %s%s.",
                    oom ? "out of memory" : err, "");
    return 0;
}

/* The repository of the arguments: 0, or -1 with the error written. */
static int repo_of(const struct janas_json *args, char *full, size_t cap,
                   char *how, size_t how_cap, struct janas_buf *b)
{
    char err[400];
    if (gh_repo_find(arg_str(args, "repo"), full, cap, how, how_cap, err,
                     sizeof err) == 0)
        return 0;
    fail(b, "%s. Ask the user which repository (owner/name)%s.", err, "");
    return -1;
}

/* "2026-09-25T10:00:00Z" */
static void iso(char *out, size_t cap, time_t t)
{
    struct tm g;
    gmtime_r(&t, &g);
    strftime(out, cap, "%Y-%m-%dT%H:%M:%SZ", &g);
}

static int tool_issues(const struct janas_json *args, struct janas_buf *b)
{
    char repo[160], how[48], err[400];
    if (repo_of(args, repo, sizeof repo, how, sizeof how, b) != 0)
        return 0;
    int pr = janas_json_is(janas_json_get(args, "kind"), "pulls");
    const char *state = arg_str(args, "state");
    if (!state || (strcmp(state, "closed") && strcmp(state, "all")))
        state = "open";
    int fresh = arg_bool(args, "new"), first = 0;
    double days = arg_num(args, "days", 0);
    time_t now = time(NULL), since = 0;
    const char *what = pr ? "pulls" : "issues";
    if (fresh) {
        since = gh_seen(repo, what);
        if (!since) {
            first = 1;
            since = now - FIRST_DAYS * DAY;
        }
        state = "all"; /* opened since then, closed or not */
    } else if (days > 0) {
        since = now - (time_t)(days * DAY);
    }
    struct janas_buf q = {0}, path = {0}, body = {0};
    janas_buf_printf(&q, "repo:%s is:%s", repo, pr ? "pr" : "issue");
    if (strcmp(state, "all") != 0)
        janas_buf_printf(&q, " is:%s", state);
    if (since) {
        char t[32];
        iso(t, sizeof t, since);
        janas_buf_printf(&q, " created:>=%s", t);
    }
    janas_buf_puts(&path, "/search/issues?q=");
    if (!q.oom)
        gh_url_put(&path, q.p);
    janas_buf_printf(&path, "&sort=created&order=desc&per_page=%d", GH_ITEMS);
    janas_buf_free(&q);
    int st = path.oom ? -1 : gh_get(path.p, &body, err, sizeof err);
    janas_buf_free(&path);
    struct gh_issues v;
    int ok = st == 200 &&
             gh_read_issues(body.p, body.n, GH_ITEMS, &v, err, sizeof err) == 0;
    janas_buf_free(&body);
    if (!ok)
        return fail(b, "The issues of %s could not be read: %s.", repo, err);
    if (fresh)
        gh_seen_set(repo, what, now);
    struct janas_buf d = {0};
    gh_issues_data(&d, repo, how, pr, state, fresh ? since : 0, first, &v);
    return answer(b, "github_issues", &d, GH_ISSUES_LAYOUT, GH_ISSUES_BRIEF);
}

static int tool_releases(const struct janas_json *args, struct janas_buf *b)
{
    char repo[160], how[48], err[400], path[256];
    if (repo_of(args, repo, sizeof repo, how, sizeof how, b) != 0)
        return 0;
    int count = (int)arg_num(args, "count", 3);
    count = count < 1 ? 1 : count > GH_ITEMS ? GH_ITEMS : count;
    struct janas_buf body = {0};
    struct gh_releases v;
    /* a few more than asked: drafts are left out */
    snprintf(path, sizeof path, "/repos/%s/releases?per_page=%d", repo,
             count + 2);
    int ok = gh_get(path, &body, err, sizeof err) == 200 &&
             gh_read_releases(body.p, body.n, count, &v, err, sizeof err) == 0;
    if (ok && v.n == 0) { /* no releases: the tags */
        snprintf(path, sizeof path, "/repos/%s/tags?per_page=%d", repo, count);
        ok = gh_get(path, &body, err, sizeof err) == 200 &&
             gh_read_tags(body.p, body.n, count, &v, err, sizeof err) == 0;
        if (ok && v.n == 0)
            v.tags = 0;
    }
    janas_buf_free(&body);
    if (!ok)
        return fail(b, "The releases of %s could not be read: %s.", repo, err);
    struct janas_buf d = {0};
    gh_releases_data(&d, repo, how, &v);
    return answer(b, "github_releases", &d, GH_RELEASES_LAYOUT,
                  GH_RELEASES_BRIEF);
}

static int tool_commits(const struct janas_json *args, struct janas_buf *b)
{
    char repo[160], how[48], err[400], path[256];
    if (repo_of(args, repo, sizeof repo, how, sizeof how, b) != 0)
        return 0;
    struct janas_buf body = {0}, p = {0};
    const char *branch = arg_str(args, "branch");
    struct gh_repo r;
    if (!branch) { /* the main one, by its name */
        snprintf(path, sizeof path, "/repos/%s", repo);
        if (gh_get(path, &body, err, sizeof err) != 200 ||
            gh_read_repo(body.p, body.n, &r, err, sizeof err) != 0) {
            janas_buf_free(&body);
            return fail(b, "%s could not be read: %s.", repo, err);
        }
        branch = r.branch;
    }
    int days = (int)arg_num(args, "days", 7);
    days = days < 1 ? 1 : days > 365 ? 365 : days;
    char since[32];
    iso(since, sizeof since, time(NULL) - (time_t)days * DAY);
    /* one more than shown: whether there are more */
    janas_buf_printf(&p, "/repos/%s/commits?per_page=%d&since=%s&sha=", repo,
                     GH_ITEMS + 1, since);
    gh_url_put(&p, branch);
    const char *author = arg_str(args, "author");
    if (author) {
        janas_buf_puts(&p, "&author=");
        gh_url_put(&p, author);
    }
    struct gh_commits v;
    int ok =
        !p.oom && gh_get(p.p, &body, err, sizeof err) == 200 &&
        gh_read_commits(body.p, body.n, GH_ITEMS, &v, err, sizeof err) == 0;
    janas_buf_free(&p);
    janas_buf_free(&body);
    if (!ok)
        return fail(b, "The commits of %s could not be read: %s.", repo, err);
    struct janas_buf d = {0};
    gh_commits_data(&d, repo, how, branch, days, &v);
    return answer(b, "github_commits", &d, GH_COMMITS_LAYOUT, GH_COMMITS_BRIEF);
}

static int tool_repo(const struct janas_json *args, struct janas_buf *b)
{
    char repo[160], how[48], err[400], path[256];
    if (repo_of(args, repo, sizeof repo, how, sizeof how, b) != 0)
        return 0;
    struct janas_buf body = {0};
    struct gh_repo r;
    snprintf(path, sizeof path, "/repos/%s", repo);
    int ok = gh_get(path, &body, err, sizeof err) == 200 &&
             gh_read_repo(body.p, body.n, &r, err, sizeof err) == 0;
    janas_buf_free(&body);
    if (!ok)
        return fail(b, "%s could not be read: %s.", repo, err);
    struct janas_buf d = {0};
    gh_repo_data(&d, how, &r);
    return answer(b, "github_repo", &d, GH_REPO_LAYOUT, GH_REPO_BRIEF);
}

int gh_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    (void)ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    if (janas_json_is(name, "github_issues"))
        return tool_issues(args, b);
    if (janas_json_is(name, "github_releases"))
        return tool_releases(args, b);
    if (janas_json_is(name, "github_commits"))
        return tool_commits(args, b);
    if (janas_json_is(name, "github_repo"))
        return tool_repo(args, b);
    return -1;
}
