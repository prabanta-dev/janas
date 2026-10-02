/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * git.h - janas-git, the git repositories of this computer as an MCP
 * service: what its parts share. run.c runs git (no shell, no terminal),
 * repo.c finds the repository meant, read.c reads git's output, data.c and
 * layouts.c write the answers as data with a layout, tools.c offers the
 * tools: those that read, and those that change something (commit, pull,
 * push, switch), which say so (readOnlyHint false) so that a client asks
 * the user every time. Nothing that throws work away is offered: no
 * reset, restore, clean, force push, amend.
 */
#ifndef JANAS_GIT_H
#define JANAS_GIT_H

#include <stddef.h>
#include <time.h>

#include "llm/json.h"

/* ---- run.c ---- */

/* git with the arguments argv (NULL-ended, without "git") in the directory
   dir, at most timeout_ms; its output into out, what it said on its error
   stream into err (both cut at 1 MiB). git's exit status, or -1 when it
   could not be run or took too long (the reason in why). */
int gt_run(const char *dir, const char *const *argv, int timeout_ms,
           struct janas_buf *out, struct janas_buf *err, char *why,
           size_t why_len);

/* ---- repo.c ---- */

/* The repository meant: a path ("~/src/x", absolute, or relative to where
   janas-git runs), a name listed in ~/.config/janas/git-repos.txt (a path
   a line; its last part is its name), or, with q NULL or empty, the one
   janas-git runs in. Its top directory into top. 0, or -1 with the reason
   in why. */
int gt_repo_find(const char *q, char *top, size_t cap, char *why,
                 size_t why_len);
/* "~/Progetti/x" for the home's paths */
void gt_short_path(const char *path, char *out, size_t cap);

/* ---- read.c: git's output (no git here: the tests read kept output) ---- */

#define GT_FILES 40
#define GT_ITEMS 20

struct gt_file {
    char path[512];
    char staged, changed; /* git's letter (M, A, D, R...), 0 when not */
    int untracked, conflict;
};

struct gt_status {
    char branch[128], upstream[160], oid[16];
    int ahead, behind, detached;
    int n_staged, n_changed, n_untracked, n_conflicts;
    struct gt_file f[GT_FILES];
    int n, more;
};

struct gt_commit {
    char sha[16], author[96], message[200], refs[160];
    time_t at;
};

struct gt_log {
    struct gt_commit c[GT_ITEMS];
    int n, more;
};

struct gt_branch {
    char name[128], upstream[160], track[64], message[200];
    time_t at;
    int current;
};

struct gt_branches {
    struct gt_branch b[GT_ITEMS];
    int n, more;
};

struct gt_numstat {
    char path[512];
    long add, del; /* -1: a binary file */
};

struct gt_diff {
    struct gt_numstat f[GT_FILES];
    int n, more;
    long add, del;
};

/* git status --porcelain=v2 --branch -z */
int gt_read_status(const char *text, size_t n, struct gt_status *s);
/* git log --format=%h%x1f%an%x1f%at%x1f%s%x1f%D%x1e; max shown, more when
   there were more */
int gt_read_log(const char *text, size_t n, int max, struct gt_log *l);
/* git for-each-ref refs/heads, fields as gt_branch, %x1f between, %x1e
   after each: %(HEAD) %(refname:short) %(upstream:short)
   %(upstream:track) %(committerdate:unix) %(subject) */
int gt_read_branches(const char *text, size_t n, int max,
                     struct gt_branches *b);
/* git diff --numstat -z */
int gt_read_numstat(const char *text, size_t n, struct gt_diff *d);

/* ---- data.c ---- */

void gt_jstr(struct janas_buf *b, const char *key, const char *v);
void gt_jwhen(struct janas_buf *b, const char *key, time_t t);
/* {"repo": its name, "path": where it is (no "}") */
void gt_jhead(struct janas_buf *b, const char *top);
void gt_status_data(struct janas_buf *b, const char *top,
                    const struct gt_status *s);
void gt_log_data(struct janas_buf *b, const char *top, const char *branch,
                 const struct gt_log *l);
void gt_branches_data(struct janas_buf *b, const char *top,
                      const struct gt_branches *v);
/* patch: the diff for the user (cut), brief: its start for the model */
void gt_diff_data(struct janas_buf *b, const char *top, int staged,
                  const char *path, const struct gt_diff *d, const char *patch,
                  size_t patch_n, size_t brief_n);
/* what a command that changes something did: git's own words */
void gt_done_data(struct janas_buf *b, const char *top, const char *what,
                  const char *branch, const char *sha, const char *message,
                  const char *output);

/* ---- layouts.c ---- */

extern const char GT_STATUS_LAYOUT[], GT_STATUS_BRIEF[];
extern const char GT_LOG_LAYOUT[], GT_LOG_BRIEF[];
extern const char GT_BRANCHES_LAYOUT[], GT_BRANCHES_BRIEF[];
extern const char GT_DIFF_LAYOUT[], GT_DIFF_BRIEF[];
extern const char GT_DONE_LAYOUT[], GT_DONE_BRIEF[];

/* ---- tools.c, write.c ---- */

void gt_tools_list(void *ctx, struct janas_buf *b);
int gt_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);
const char *gt_arg_str(const struct janas_json *args, const char *name);
int gt_arg_bool(const struct janas_json *args, const char *name);
/* The text as the tool's result, an error when is_error */
int gt_fail(struct janas_buf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
int gt_answer(struct janas_buf *b, const char *name, struct janas_buf *d,
              const char *layout, const char *brief);
/* A text git takes as an argument and never as an option: not empty, not
   beginning with "-", no line end. */
int gt_plain_arg(const char *s);
int gt_tool_commit(const struct janas_json *args, struct janas_buf *b);
int gt_tool_pull(const struct janas_json *args, struct janas_buf *b);
int gt_tool_push(const struct janas_json *args, struct janas_buf *b);
int gt_tool_switch(const struct janas_json *args, struct janas_buf *b);

#endif
