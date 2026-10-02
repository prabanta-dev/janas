/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_git_parse.c - janas-git's reading of git's output, without running
 * git, and the layouts filled with what was read: a status with a file
 * changed, one renamed and staged, one new and one in conflict (that line
 * written from git's documentation), a log, the branches and a numstat
 * with a rename, as git 2.47 gave them on 2 October 2026 in a repository
 * made for the purpose.
 * The sources belong to the program, so they are compiled in here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "services/common/template.h"
#include "services/git/data.c"
#include "services/git/layouts.c"
#include "services/git/read.c"
#include "services/git/repo.c"
#include "services/git/run.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char S_STATUS[] =
    "# branch.oid dc786f796ff004c8599141031781b0e8e40d5ed2\0# branch.head m"
    "ain\0# branch.upstream origin/main\0# branch.ab +1 -0\0"
    "1 .M N... 100644 100644 100644 fb85c0a2eacf48a66ef1b6802bbaacfd179e118"
    "1 fb85c0a2eacf48a66ef1b6802bbaacfd179e1181 a.txt\0"
    "2 R. N... 100644 100644 100644 b31b0b130d568d7e685646a4c191c5b7d11d49e"
    "a b31b0b130d568d7e685646a4c191c5b7d11d49ea R100 e.txt\0"
    "c.txt\0? d.txt\0u UU N... 100644 100644 100644 100644 1111111111111111"
    "111111111111111111111111 2222222222222222222222222222222222222222 3333"
    "333333333333333333333333333333333333 x y.txt\0";
#define S_STATUS_N (sizeof S_STATUS - 1)
static const char S_LOG[] =
    "dc786f7\x1fTester\x1f"
    "1790943106\x1fterzo\x1fHEAD -> main\x1e\n33f8fc2\x1fTester\x1f"
    "1790943024\x1fsecondo: a e b\x1forigin/prova, origin/main, prova\x1e\n"
    "ed69737\x1fTester\x1f"
    "1790943024\x1fprimo\x1f\x1e\n";
#define S_LOG_N (sizeof S_LOG - 1)
static const char S_BRANCHES[] =
    "*\x1fmain\x1forigin/main\x1f[ahead 1]\x1f"
    "1790943106\x1fterzo\x1e\n \x1fprova\x1forigin/prova\x1f\x1f"
    "1790943024\x1fsecondo: a e b\x1e\n";
#define S_BRANCHES_N (sizeof S_BRANCHES - 1)
static const char S_NUMSTAT[] = "0\t0\t\0"
                                "c.txt\0"
                                "e.txt\0"
                                "1\t0\ta.txt\0";
#define S_NUMSTAT_N (sizeof S_NUMSTAT - 1)

/* The layout filled with the data: the text, or "" when it failed. */
static char text[16384];

static const char *fill(const char *layout, const struct janas_buf *d)
{
    text[0] = 0;
    struct janas_json_doc *doc =
        janas_json_parse(d->p ? d->p : "", d->n, NULL, 0);
    CHECK(doc != NULL, "the data is not JSON:\n%.*s", (int)d->n, d->p);
    if (!doc)
        return text;
    struct janas_buf out = {0};
    char err[200];
    if (janas_tpl_render(layout, strlen(layout), janas_json_root(doc), &out,
                         err, sizeof err) == 0)
        snprintf(text, sizeof text, "%.*s", (int)out.n, out.p);
    else
        CHECK(0, "a layout not filled: %s", err);
    janas_buf_free(&out);
    janas_json_free(doc);
    return text;
}

#define TOP "/home/someone/src/work"

static void status(void)
{
    struct gt_status s;
    CHECK(gt_read_status(S_STATUS, S_STATUS_N, &s) == 0, "the status");
    CHECK(strcmp(s.branch, "main") == 0 &&
              strcmp(s.upstream, "origin/main") == 0 && s.ahead == 1 &&
              s.behind == 0 && !s.detached && strcmp(s.oid, "dc786f7") == 0,
          "the branch: %s %s +%d -%d %s", s.branch, s.upstream, s.ahead,
          s.behind, s.oid);
    CHECK(s.n == 4 && s.n_staged == 1 && s.n_changed == 1 &&
              s.n_untracked == 1 && s.n_conflicts == 1,
          "the counts: %d files", s.n);
    CHECK(strcmp(s.f[0].path, "a.txt") == 0 && s.f[0].changed == 'M' &&
              !s.f[0].staged,
          "a file changed");
    CHECK(strcmp(s.f[1].path, "e.txt") == 0 && s.f[1].staged == 'R' &&
              !s.f[1].changed,
          "a file renamed: %s", s.f[1].path);
    CHECK(strcmp(s.f[2].path, "d.txt") == 0 && s.f[2].untracked &&
              strcmp(s.f[3].path, "x y.txt") == 0 && s.f[3].conflict,
          "a new file, one in conflict: %s, %s", s.f[2].path, s.f[3].path);
    struct janas_buf d = {0};
    gt_status_data(&d, TOP, &s);
    const char *t = fill(GT_STATUS_LAYOUT, &d);
    CHECK(strcmp(t, "work (~/src/work), branch main, following origin/main; "
                    "commits to push: 1.\n"
                    "staged: 1, changed: 1, new: 1, in conflict: 1\n"
                    "  a.txt: modified\n"
                    "  e.txt: renamed, staged\n"
                    "  d.txt: new, not tracked\n"
                    "  x y.txt: in conflict\n") == 0,
          "the status' layout:\n%s", t);
    t = fill(GT_STATUS_BRIEF, &d);
    CHECK(strstr(t, "a.txt M; e.txt R; d.txt ?; x y.txt U;"),
          "the status' brief:\n%s", t);
    janas_buf_free(&d);
    struct gt_status clean = {.branch = "main"};
    gt_status_data(&d, TOP, &clean);
    t = fill(GT_STATUS_LAYOUT, &d);
    CHECK(strcmp(t, "work (~/src/work), branch main, following no remote "
                    "branch.\nNothing to commit: the working tree is "
                    "clean.\n") == 0,
          "clean:\n%s", t);
    janas_buf_free(&d);
}

static void log_branches(void)
{
    struct gt_log l;
    gt_read_log(S_LOG, S_LOG_N, 2, &l);
    CHECK(l.n == 2 && l.more && strcmp(l.c[0].sha, "dc786f7") == 0 &&
              strcmp(l.c[0].message, "terzo") == 0 &&
              strcmp(l.c[0].refs, "HEAD -> main") == 0 &&
              l.c[0].at == 1790943106 &&
              strcmp(l.c[1].message, "secondo: a e b") == 0,
          "the log: %d", l.n);
    struct janas_buf d = {0};
    gt_log_data(&d, TOP, "main", &l);
    const char *t = fill(GT_LOG_LAYOUT, &d);
    CHECK(strstr(t, "work (~/src/work), main: the latest commits, 2\n"
                    "  dc786f7 terzo (HEAD -> main)\n      Tester, 2 "
                    "October") &&
              strstr(t, "  33f8fc2 secondo: a e b (origin/prova"),
          "the log's layout:\n%s", t);
    janas_buf_free(&d);
    struct gt_branches b;
    gt_read_branches(S_BRANCHES, S_BRANCHES_N, 20, &b);
    CHECK(b.n == 2 && b.b[0].current && strcmp(b.b[0].name, "main") == 0 &&
              strcmp(b.b[0].track, "[ahead 1]") == 0 && !b.b[1].current &&
              strcmp(b.b[1].upstream, "origin/prova") == 0 && !b.b[1].track[0],
          "the branches");
    gt_read_branches(S_BRANCHES, S_BRANCHES_N, 1, &b);
    CHECK(b.n == 1 && b.more == 1 && b.b[0].current, "one branch listed");
    gt_read_branches(S_BRANCHES + 0, S_BRANCHES_N, 20, &b);
    gt_branches_data(&d, TOP, &b);
    t = fill(GT_BRANCHES_LAYOUT, &d);
    CHECK(strstr(t, "* main -> origin/main [ahead 1]\n      2 October") &&
              strstr(t, "\n  prova -> origin/prova\n"),
          "the branches' layout:\n%s", t);
    janas_buf_free(&d);
}

static void diff_done(void)
{
    struct gt_diff df;
    gt_read_numstat(S_NUMSTAT, S_NUMSTAT_N, &df);
    CHECK(df.n == 2 && strcmp(df.f[0].path, "e.txt") == 0 && df.f[0].add == 0 &&
              strcmp(df.f[1].path, "a.txt") == 0 && df.f[1].add == 1 &&
              df.add == 1 && df.del == 0,
          "the numstat: %d %s", df.n, df.f[0].path);
    gt_read_numstat("-\t-\tlogo.png\0", 13, &df);
    CHECK(df.n == 1 && df.f[0].add == -1 &&
              strcmp(df.f[0].path, "logo.png") == 0,
          "a binary file");
    static const char patch[] = "diff --git a/x b/x\n+uno\n+due\n";
    struct janas_buf d = {0};
    gt_diff_data(&d, TOP, 0, NULL, &df, patch, sizeof patch - 1, 19);
    const char *t = fill(GT_DIFF_LAYOUT, &d);
    CHECK(strstr(t, "the changes not staged; files: 1, lines +0 -0\n  logo.png "
                    "(binary)\n\ndiff --git a/x b/x\n+uno\n+due\n"),
          "the diff's layout:\n%s", t);
    t = fill(GT_DIFF_BRIEF, &d);
    CHECK(strstr(t, "; it begins:\ndiff --git a/x b/x\n") && !strstr(t, "+uno"),
          "the diff's brief:\n%s", t);
    janas_buf_free(&d);
    gt_done_data(&d, TOP, "commit", "main", "abc1234", "terzo",
                 "2 files changed, 1 insertion(+)");
    t = fill(GT_DONE_LAYOUT, &d);
    CHECK(strcmp(t, "work (~/src/work): committed, branch main, abc1234: "
                    "terzo\n2 files changed, 1 insertion(+)\n") == 0,
          "done:\n%s", t);
    janas_buf_free(&d);
}

int main(void)
{
    setenv("TZ", "UTC", 1);
    setenv("HOME", "/home/someone", 1);
    status();
    log_branches();
    diff_done();
    if (failures) {
        printf("test_git_parse: %d failures\n", failures);
        return 1;
    }
    printf("test_git_parse: ok\n");
    return 0;
}
