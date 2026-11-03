/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * files.h - janas-files, the user's files as an MCP service: what its
 * parts share. place.c says where a path leads and whether it may be
 * reached (the home and the folders the user listed, never the private
 * files: keys, passwords, the browsers' data); read.c lists, reads and
 * searches; write.c writes, edits and makes folders, a copy of what it
 * replaces kept first; tools.c offers the tools: those that write say so
 * (readOnlyHint false), so that a client asks the user every time. Nothing
 * is deleted.
 */
#ifndef JANAS_FILES_H
#define JANAS_FILES_H

#include <limits.h>
#include <stddef.h>

#include "llm/json.h"

/* ---- place.c ---- */

#define FL_ROOTS 16

/* Where the tools may go: the home, and the folders listed one a line in
   ~/.config/janas/files-dirs.txt; cwd is where relative paths start. */
struct fl_roots {
    char home[PATH_MAX], cwd[PATH_MAX];
    char root[FL_ROOTS][PATH_MAX];
    int n_root;
};

/* The path meant (~/x, absolute, or relative to cwd) as an absolute one,
   not yet followed: 0, or -1 when too long. */
int fl_expand(const struct fl_roots *r, const char *in, char *out, size_t cap);
/* The roots from the environment (HOME, the working directory) and the
   list. 0, or -1 when there is no home. */
int fl_roots_load(struct fl_roots *r);
/* The path meant (~/x, absolute, or relative to cwd) into out, with every
   link followed: must_exist 1 for a file or folder there is, 0 for one to
   make (its folder must be there). 0, or -1 with the reason in why: not
   there, outside the roots, or private. */
int fl_resolve(const struct fl_roots *r, const char *in, int must_exist,
               char *out, size_t cap, char *why, size_t why_len);
/* 1 when the path (resolved) is private: keys, passwords, tokens, the
   browsers' data, the shells' histories. */
int fl_private(const struct fl_roots *r, const char *path);
/* "~/x" for the home's paths, else the path */
void fl_short(const struct fl_roots *r, const char *path, char *out,
              size_t cap);

/* ---- read.c ---- */

int fl_tool_list(const struct fl_roots *r, const struct janas_json *args,
                 struct janas_buf *b);
int fl_tool_read(const struct fl_roots *r, const struct janas_json *args,
                 struct janas_buf *b);
int fl_tool_search(const struct fl_roots *r, const struct janas_json *args,
                   struct janas_buf *b);
/* 1 when the n bytes look like text (no NUL) */
int fl_is_text(const char *p, size_t n);

/* ---- write.c ---- */

int fl_tool_write(const struct fl_roots *r, const struct janas_json *args,
                  struct janas_buf *b);
int fl_tool_edit(const struct fl_roots *r, const struct janas_json *args,
                 struct janas_buf *b);
int fl_tool_mkdir(const struct fl_roots *r, const struct janas_json *args,
                  struct janas_buf *b);

/* ---- tools.c ---- */

void fl_tools_list(void *ctx, struct janas_buf *b);
int fl_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);
const char *fl_arg_str(const struct janas_json *args, const char *name);
int fl_arg_bool(const struct janas_json *args, const char *name);
double fl_arg_num(const struct janas_json *args, const char *name, double def);
/* The text as the tool's result; an error with fl_fail */
int fl_ok(struct janas_buf *b, struct janas_buf *text);
int fl_fail(struct janas_buf *b, const char *fmt, ...)
    __attribute__((format(gnu_printf, 2, 3)));

#endif
