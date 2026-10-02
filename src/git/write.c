/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * write.c - janas-git's tools that change something (see git.h): commit,
 * pull, push, switch. Each is asked of the user first by the client
 * (readOnlyHint false). What they never do: amend, skip the hooks, merge
 * or rebase on a pull (fast-forward only), push by force, throw away a
 * change (git switch refuses when it would), take an argument of the
 * model's as an option.
 */
#include <stdio.h>
#include <string.h>

#include "git.h"

#define LOCAL_MS 15000
#define HOOKS_MS 45000  /* a commit's hooks may run tests */
#define REMOTE_MS 50000 /* janas-chat waits a minute for a tool */
#define FILES_MAX 32
#define OUTPUT_MAX 1500

static int run(const char *top, const char *const *argv, int ms,
               struct janas_buf *out, struct janas_buf *err, char *why,
               size_t why_len)
{
    out->n = err->n = 0;
    return gt_run(top, argv, ms, out, err, why, why_len);
}

/* git's words, cut: what it printed, its error stream first */
static void words(char *to, size_t cap, const struct janas_buf *out,
                  const struct janas_buf *err)
{
    size_t k = 0;
    to[0] = 0;
    const struct janas_buf *v[2] = {err, out};
    for (int i = 0; i < 2; i++)
        if (v[i]->n && k + 1 < cap)
            k += (size_t)snprintf(to + k, cap - k, "%s%.*s", k ? "\n" : "",
                                  (int)v[i]->n, v[i]->p);
    while (k && (to[k - 1] == '\n' || to[k - 1] == ' '))
        to[--k] = 0;
}

static int repo_of(const struct janas_json *args, char *top, size_t cap,
                   struct janas_buf *b)
{
    char why[600];
    if (gt_repo_find(gt_arg_str(args, "repo"), top, cap, why, sizeof why) == 0)
        return 0;
    gt_fail(b, "%s.", why);
    return -1;
}

static void branch_of(const char *top, char *out, size_t cap)
{
    const char *argv[] = {"rev-parse", "--abbrev-ref", "HEAD", NULL};
    struct janas_buf o = {0}, e = {0};
    char why[200];
    out[0] = 0;
    if (gt_run(top, argv, LOCAL_MS, &o, &e, why, sizeof why) == 0 && o.n)
        snprintf(out, cap, "%.*s", (int)strcspn(o.p, "\n"), o.p);
    janas_buf_free(&o);
    janas_buf_free(&e);
}

static int done(struct janas_buf *b, const char *top, const char *what,
                const char *branch, const char *sha, const char *message,
                const char *output)
{
    struct janas_buf d = {0};
    gt_done_data(&d, top, what, branch, sha, message, output);
    char name[32];
    snprintf(name, sizeof name, "git_%s",
             strcmp(what, "create") == 0 ? "switch" : what);
    return gt_answer(b, name, &d, GT_DONE_LAYOUT, GT_DONE_BRIEF);
}

int gt_tool_commit(const struct janas_json *args, struct janas_buf *b)
{
    char top[4096], why[300], said[OUTPUT_MAX];
    if (repo_of(args, top, sizeof top, b) != 0)
        return 0;
    const char *message = gt_arg_str(args, "message");
    if (!message)
        return gt_fail(b, "No message given for the commit.");
    struct janas_buf out = {0}, err = {0};
    int st = 0;
    /* the files named, after "--": a path is never an option */
    const struct janas_json *files = janas_json_get(args, "files");
    if (files && files->type == JANAS_JSON_ARRAY && files->child) {
        const char *argv[FILES_MAX + 4];
        int k = 0;
        argv[k++] = "add";
        argv[k++] = "--";
        for (const struct janas_json *f = files->child; f; f = f->next) {
            const char *p = janas_json_str(f);
            if (!p || !*p || strpbrk(p, "\r\n") || k >= FILES_MAX + 2) {
                janas_buf_free(&out);
                janas_buf_free(&err);
                return gt_fail(b,
                               "The files to stage: at most %d paths, "
                               "each a line.",
                               FILES_MAX);
            }
            argv[k++] = p;
        }
        argv[k] = NULL;
        st = run(top, argv, LOCAL_MS, &out, &err, why, sizeof why);
    } else if (gt_arg_bool(args, "all")) {
        const char *argv[] = {"add", "-u", NULL}; /* the tracked ones */
        st = run(top, argv, LOCAL_MS, &out, &err, why, sizeof why);
    }
    if (st != 0) {
        words(said, sizeof said, &out, &err);
        janas_buf_free(&out);
        janas_buf_free(&err);
        return gt_fail(b, "git add failed: %s", st < 0 ? why : said);
    }
    const char *check[] = {"diff", "--cached", "--quiet", NULL};
    st = run(top, check, LOCAL_MS, &out, &err, why, sizeof why);
    if (st == 0) {
        janas_buf_free(&out);
        janas_buf_free(&err);
        return gt_fail(b, "Nothing is staged: name the files to commit, or "
                          "ask for all the tracked files changed.");
    }
    const char *commit[] = {"commit", "-m", message, NULL, NULL};
    if (gt_arg_bool(args, "signoff"))
        commit[3] = "--signoff";
    st = run(top, commit, HOOKS_MS, &out, &err, why, sizeof why);
    if (st != 0) {
        words(said, sizeof said, &out, &err);
        janas_buf_free(&out);
        janas_buf_free(&err);
        return gt_fail(b, "git commit failed: %s", st < 0 ? why : said);
    }
    /* the files the commit holds: those staged before count too */
    const char *show[] = {"show", "--stat=72", "--format=%h", "HEAD", NULL};
    char sha[16] = "", stat[OUTPUT_MAX] = "", branch[256];
    if (run(top, show, LOCAL_MS, &out, &err, why, sizeof why) == 0 && out.n) {
        snprintf(sha, sizeof sha, "%.*s", (int)strcspn(out.p, "\n"), out.p);
        const char *s = strchr(out.p, '\n');
        while (s && *s == '\n')
            s++;
        if (s)
            snprintf(stat, sizeof stat, "%s", s);
        size_t k = strlen(stat);
        while (k && stat[k - 1] == '\n')
            stat[--k] = 0;
    }
    janas_buf_free(&out);
    janas_buf_free(&err);
    branch_of(top, branch, sizeof branch);
    char first[200];
    snprintf(first, sizeof first, "%.*s", (int)strcspn(message, "\n"), message);
    return done(b, top, "commit", branch, sha, first, stat);
}

int gt_tool_pull(const struct janas_json *args, struct janas_buf *b)
{
    char top[4096], why[300], said[OUTPUT_MAX], branch[256];
    if (repo_of(args, top, sizeof top, b) != 0)
        return 0;
    const char *argv[] = {"pull", "--ff-only", NULL};
    struct janas_buf out = {0}, err = {0};
    int st = run(top, argv, REMOTE_MS, &out, &err, why, sizeof why);
    words(said, sizeof said, &out, &err);
    janas_buf_free(&out);
    janas_buf_free(&err);
    if (st != 0)
        return gt_fail(b, "git pull (fast-forward only) failed: %s",
                       st < 0 ? why : said);
    branch_of(top, branch, sizeof branch);
    return done(b, top, "pull", branch, "", "", said);
}

int gt_tool_push(const struct janas_json *args, struct janas_buf *b)
{
    char top[4096], why[300], said[OUTPUT_MAX], branch[256];
    if (repo_of(args, top, sizeof top, b) != 0)
        return 0;
    branch_of(top, branch, sizeof branch);
    struct janas_buf out = {0}, err = {0};
    const char *up[] = {"rev-parse", "--abbrev-ref", "--symbolic-full-name",
                        "@{upstream}", NULL};
    int has_up = run(top, up, LOCAL_MS, &out, &err, why, sizeof why) == 0;
    int set = gt_arg_bool(args, "set_upstream");
    if (!has_up && !set) {
        janas_buf_free(&out);
        janas_buf_free(&err);
        return gt_fail(b,
                       "The branch %s follows no remote branch: ask the user "
                       "whether to push it to origin and follow it "
                       "(set_upstream).",
                       branch);
    }
    const char *plain[] = {"push", NULL};
    const char *first[] = {"push", "-u", "origin", "HEAD", NULL};
    int st = run(top, has_up ? plain : first, REMOTE_MS, &out, &err, why,
                 sizeof why);
    words(said, sizeof said, &out, &err);
    janas_buf_free(&out);
    janas_buf_free(&err);
    if (st != 0)
        return gt_fail(b, "git push failed: %s", st < 0 ? why : said);
    return done(b, top, "push", branch, "", "", said);
}

int gt_tool_switch(const struct janas_json *args, struct janas_buf *b)
{
    char top[4096], why[300], said[OUTPUT_MAX];
    if (repo_of(args, top, sizeof top, b) != 0)
        return 0;
    const char *name = gt_arg_str(args, "branch");
    struct janas_buf out = {0}, err = {0};
    const char *check[] = {"check-ref-format", "--branch", name, NULL};
    if (!gt_plain_arg(name) ||
        run(top, check, LOCAL_MS, &out, &err, why, sizeof why) != 0) {
        janas_buf_free(&out);
        janas_buf_free(&err);
        return gt_fail(b, "\"%s\" is not a branch's name.", name ? name : "");
    }
    int create = gt_arg_bool(args, "create");
    const char *go[] = {"switch", name, NULL};
    const char *make[] = {"switch", "-c", name, NULL};
    int st =
        run(top, create ? make : go, LOCAL_MS, &out, &err, why, sizeof why);
    words(said, sizeof said, &out, &err);
    janas_buf_free(&out);
    janas_buf_free(&err);
    if (st != 0)
        return gt_fail(b, "git switch failed: %s", st < 0 ? why : said);
    return done(b, top, create ? "create" : "switch", name, "", "", said);
}
