/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - janas-git's tools (see git.h) and what they share: those that
 * read (git_status, git_log, git_diff, git_branches) say so; those that
 * change something (git_commit, git_pull, git_push, git_switch, in
 * write.c) say they do not only read, so that a client asks the user
 * every time (janas-chat does, --mcp-auto or not).
 */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "services/common/template.h"
#include "git.h"

#define LOCAL_MS 15000
#define PATCH_LINES 400  /* of a diff shown to the user */
#define PATCH_MAX 32000  /* bytes of it */
#define BRIEF_PATCH 3000 /* of it the model reads */

#define REPO_ARG                                                               \
    "\"repo\": {\"type\": \"string\", \"description\": \"a path, or a name "   \
    "the user listed; leave it out for the one janas-chat runs in\"}"

#define READS "\"annotations\": {\"readOnlyHint\": true}"
#define WRITES                                                                 \
    "\"annotations\": {\"readOnlyHint\": false, \"destructiveHint\": false}"

void gt_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    janas_buf_puts(
        b,
        "\"tools\": [{\"name\": \"git_status\", \"title\": \"What has "
        "changed\", \"description\": \"A repository's branch, what it is "
        "ahead or behind, the files staged, changed, new.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": "
        "{" REPO_ARG "}, \"additionalProperties\": false}, " READS "}, "
        "{\"name\": \"git_log\", \"title\": \"Commits\", \"description\": "
        "\"A branch's latest commits.\", \"inputSchema\": {\"type\": "
        "\"object\", \"properties\": {" REPO_ARG ", \"count\": {\"type\": "
        "\"number\", \"description\": \"default 10, at most 20\"}, "
        "\"branch\": {\"type\": \"string\"}, \"path\": {\"type\": "
        "\"string\", \"description\": \"only those changing it\"}, "
        "\"author\": {\"type\": \"string\"}, \"days\": {\"type\": "
        "\"number\"}}, \"additionalProperties\": false}, " READS "}, "
        "{\"name\": \"git_diff\", \"title\": \"The changes\", "
        "\"description\": \"The changes not staged, or those staged for "
        "the next commit: the files, and the diff shown to the user.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": "
        "{" REPO_ARG ", \"staged\": {\"type\": \"boolean\"}, \"path\": "
        "{\"type\": \"string\"}}, \"additionalProperties\": false}, " READS
        "}, "
        "{\"name\": \"git_branches\", \"title\": \"Branches\", "
        "\"description\": \"The local branches, the latest worked on "
        "first.\", \"inputSchema\": {\"type\": \"object\", \"properties\": "
        "{" REPO_ARG "}, \"additionalProperties\": false}, " READS "}, "
        "{\"name\": \"git_commit\", \"title\": \"Commit\", \"description\": "
        "\"Commits what is staged, after staging the files named, or all "
        "the tracked files changed. The user is asked first.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": "
        "{" REPO_ARG ", \"message\": {\"type\": \"string\"}, \"files\": "
        "{\"type\": \"array\", \"items\": {\"type\": \"string\"}}, \"all\": "
        "{\"type\": \"boolean\", \"description\": \"stage every tracked "
        "file changed\"}, \"signoff\": {\"type\": \"boolean\"}}, "
        "\"required\": [\"message\"], \"additionalProperties\": false}, " WRITES
        "}, "
        "{\"name\": \"git_pull\", \"title\": \"Pull\", \"description\": "
        "\"Brings the branch up to its remote one, only when that needs "
        "no merge (fast-forward). The user is asked first.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": "
        "{" REPO_ARG "}, \"additionalProperties\": false}, " WRITES "}, "
        "{\"name\": \"git_push\", \"title\": \"Push\", \"description\": "
        "\"Sends the branch's commits to its remote branch (never by "
        "force). The user is asked first.\", \"inputSchema\": {\"type\": "
        "\"object\", \"properties\": {" REPO_ARG ", \"set_upstream\": "
        "{\"type\": \"boolean\", \"description\": \"a branch not on the "
        "remote yet: push it to origin and follow it\"}}, "
        "\"additionalProperties\": false}, " WRITES "}, "
        "{\"name\": \"git_switch\", \"title\": \"Switch branch\", "
        "\"description\": \"Goes to another branch, or makes a new one and "
        "goes to it; git refuses when changes would be lost. The user is "
        "asked first.\", \"inputSchema\": {\"type\": \"object\", "
        "\"properties\": {" REPO_ARG ", \"branch\": {\"type\": \"string\"}, "
        "\"create\": {\"type\": \"boolean\"}}, \"required\": [\"branch\"], "
        "\"additionalProperties\": false}, " WRITES "}]");
}

const char *gt_arg_str(const struct janas_json *args, const char *name)
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

int gt_arg_bool(const struct janas_json *args, const char *name)
{
    const struct janas_json *v = janas_json_get(args, name);
    return v && v->type == JANAS_JSON_TRUE;
}

int gt_plain_arg(const char *s)
{
    return s && *s && *s != '-' && !strpbrk(s, "\r\n");
}

int gt_fail(struct janas_buf *b, const char *fmt, ...)
{
    struct janas_buf out = {0};
    char line[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    janas_buf_puts(&out, line);
    janas_mcps_text_result(b, out.p ? out.p : "", out.n, 1);
    janas_buf_free(&out);
    return 0;
}

int gt_answer(struct janas_buf *b, const char *name, struct janas_buf *d,
              const char *layout, const char *brief)
{
    char err[300];
    int oom = d->oom;
    int r = oom ? -1
                : janas_tpl_result(b, name, d->p, d->n, layout, brief, err,
                                   sizeof err);
    janas_buf_free(d);
    if (r != 0)
        return gt_fail(b, "The answer could not be written: %s.",
                       oom ? "out of memory" : err);
    return 0;
}

/* The repository of the arguments: 0, or -1 with the error written. */
static int repo_of(const struct janas_json *args, char *top, size_t cap,
                   struct janas_buf *b)
{
    char why[600];
    if (gt_repo_find(gt_arg_str(args, "repo"), top, cap, why, sizeof why) == 0)
        return 0;
    gt_fail(b, "%s.", why);
    return -1;
}

/* git's complaint, its first lines */
static const char *said(const struct janas_buf *err, const struct janas_buf *o)
{
    const char *s = err->n ? err->p : o->n ? o->p : "no reason given";
    return s;
}

static int tool_status(const struct janas_json *args, struct janas_buf *b)
{
    char top[4096], why[300];
    if (repo_of(args, top, sizeof top, b) != 0)
        return 0;
    const char *argv[] = {"status", "--porcelain=v2", "--branch", "-z", NULL};
    struct janas_buf out = {0}, err = {0};
    int st = gt_run(top, argv, LOCAL_MS, &out, &err, why, sizeof why);
    struct gt_status s;
    int ok = st == 0 && gt_read_status(out.p, out.n, &s) == 0;
    if (!ok)
        gt_fail(b, "git status failed: %.1500s",
                st < 0 ? why : said(&err, &out));
    janas_buf_free(&out);
    janas_buf_free(&err);
    if (!ok)
        return 0;
    struct janas_buf d = {0};
    gt_status_data(&d, top, &s);
    return gt_answer(b, "git_status", &d, GT_STATUS_LAYOUT, GT_STATUS_BRIEF);
}

/* The branch one is on ("HEAD" when none) */
static void current_branch(const char *top, char *out, size_t cap)
{
    const char *argv[] = {"rev-parse", "--abbrev-ref", "HEAD", NULL};
    struct janas_buf o = {0}, e = {0};
    char why[200];
    if (gt_run(top, argv, LOCAL_MS, &o, &e, why, sizeof why) == 0 && o.n)
        snprintf(out, cap, "%.*s", (int)strcspn(o.p, "\n"), o.p);
    else
        snprintf(out, cap, "HEAD");
    janas_buf_free(&o);
    janas_buf_free(&e);
}

static int tool_log(const struct janas_json *args, struct janas_buf *b)
{
    char top[4096], why[300], n[16], author[300], since[48], branch[256];
    if (repo_of(args, top, sizeof top, b) != 0)
        return 0;
    int count = (int)arg_num(args, "count", 10);
    count = count < 1 ? 1 : count > GT_ITEMS ? GT_ITEMS : count;
    snprintf(n, sizeof n, "-n%d", count + 1); /* one more: are there more? */
    const char *argv[16];
    int k = 0;
    argv[k++] = "log";
    argv[k++] = n;
    argv[k++] = "--format=%h%x1f%an%x1f%at%x1f%s%x1f%D%x1e";
    const char *a = gt_arg_str(args, "author");
    if (a) {
        snprintf(author, sizeof author, "--author=%s", a);
        argv[k++] = author;
    }
    double days = arg_num(args, "days", 0);
    if (days > 0) {
        snprintf(since, sizeof since, "--since=%.0f.days.ago", days);
        argv[k++] = since;
    }
    const char *br = gt_arg_str(args, "branch");
    if (br && !gt_plain_arg(br))
        return gt_fail(b, "\"%s\" is not a branch's name.", br);
    if (br) {
        argv[k++] = br;
        snprintf(branch, sizeof branch, "%s", br);
    } else {
        current_branch(top, branch, sizeof branch);
    }
    argv[k++] = "--";
    const char *path = gt_arg_str(args, "path");
    if (path)
        argv[k++] = path;
    argv[k] = NULL;
    struct janas_buf out = {0}, err = {0};
    int st = gt_run(top, argv, LOCAL_MS, &out, &err, why, sizeof why);
    struct gt_log l;
    if (st == 0)
        gt_read_log(out.p, out.n, count, &l);
    else
        gt_fail(b, "git log failed: %.1500s", st < 0 ? why : said(&err, &out));
    janas_buf_free(&out);
    janas_buf_free(&err);
    if (st != 0)
        return 0;
    struct janas_buf d = {0};
    gt_log_data(&d, top, branch, &l);
    return gt_answer(b, "git_log", &d, GT_LOG_LAYOUT, GT_LOG_BRIEF);
}

static int tool_branches(const struct janas_json *args, struct janas_buf *b)
{
    char top[4096], why[300];
    if (repo_of(args, top, sizeof top, b) != 0)
        return 0;
    const char *argv[] = {
        "for-each-ref", "refs/heads", "--sort=-committerdate",
        "--format=%(HEAD)%1f%(refname:short)%1f%(upstream:short)%1f"
        "%(upstream:track)%1f%(committerdate:unix)%1f%(subject)%1e",
        NULL};
    struct janas_buf out = {0}, err = {0};
    int st = gt_run(top, argv, LOCAL_MS, &out, &err, why, sizeof why);
    struct gt_branches v;
    if (st == 0)
        gt_read_branches(out.p, out.n, GT_ITEMS, &v);
    else
        gt_fail(b, "git for-each-ref failed: %.1500s",
                st < 0 ? why : said(&err, &out));
    janas_buf_free(&out);
    janas_buf_free(&err);
    if (st != 0)
        return 0;
    struct janas_buf d = {0};
    gt_branches_data(&d, top, &v);
    return gt_answer(b, "git_branches", &d, GT_BRANCHES_LAYOUT,
                     GT_BRANCHES_BRIEF);
}

/* The first PATCH_LINES lines, PATCH_MAX bytes at most, ending at a line */
static size_t patch_cut(const char *p, size_t n, size_t max_bytes)
{
    size_t at = 0;
    int lines = 0;
    while (at < n && lines < PATCH_LINES) {
        const char *nl = memchr(p + at, '\n', n - at);
        size_t next = nl ? (size_t)(nl - p) + 1 : n;
        if (next > max_bytes)
            break;
        at = next;
        lines++;
    }
    return at;
}

static int tool_diff(const struct janas_json *args, struct janas_buf *b)
{
    char top[4096], why[300];
    if (repo_of(args, top, sizeof top, b) != 0)
        return 0;
    int staged = gt_arg_bool(args, "staged");
    const char *path = gt_arg_str(args, "path");
    const char *num[8], *pat[8];
    int k = 0, j = 0;
    num[k++] = "diff", pat[j++] = "diff";
    if (staged)
        num[k++] = "--cached", pat[j++] = "--cached";
    num[k++] = "--numstat";
    num[k++] = "-z";
    pat[j++] = "--no-color";
    pat[j++] = "--no-ext-diff";
    num[k++] = "--", pat[j++] = "--";
    if (path)
        num[k++] = path, pat[j++] = path;
    num[k] = pat[j] = NULL;
    struct janas_buf out = {0}, err = {0}, patch = {0};
    int st = gt_run(top, num, LOCAL_MS, &out, &err, why, sizeof why);
    struct gt_diff d;
    if (st == 0) {
        gt_read_numstat(out.p, out.n, &d);
        janas_buf_free(&err);
        st = gt_run(top, pat, LOCAL_MS, &patch, &err, why, sizeof why);
    }
    if (st != 0)
        gt_fail(b, "git diff failed: %.1500s", st < 0 ? why : said(&err, &out));
    janas_buf_free(&out);
    janas_buf_free(&err);
    if (st != 0) {
        janas_buf_free(&patch);
        return 0;
    }
    size_t shown = patch_cut(patch.p, patch.n, PATCH_MAX);
    size_t start = patch_cut(patch.p, shown, BRIEF_PATCH);
    struct janas_buf data = {0};
    gt_diff_data(&data, top, staged, path, &d, patch.p, shown, start);
    janas_buf_free(&patch);
    return gt_answer(b, "git_diff", &data, GT_DIFF_LAYOUT, GT_DIFF_BRIEF);
}

int gt_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    (void)ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    if (janas_json_is(name, "git_status"))
        return tool_status(args, b);
    if (janas_json_is(name, "git_log"))
        return tool_log(args, b);
    if (janas_json_is(name, "git_diff"))
        return tool_diff(args, b);
    if (janas_json_is(name, "git_branches"))
        return tool_branches(args, b);
    if (janas_json_is(name, "git_commit"))
        return gt_tool_commit(args, b);
    if (janas_json_is(name, "git_pull"))
        return gt_tool_pull(args, b);
    if (janas_json_is(name, "git_push"))
        return gt_tool_push(args, b);
    if (janas_json_is(name, "git_switch"))
        return gt_tool_switch(args, b);
    return -1;
}
