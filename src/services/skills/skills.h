/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * skills.h - janas-skills, the user's skills as an MCP service: skills in
 * the format of Claude's (common/mcp_skills.h), listed, read, and their
 * scripts run in a cage (common/sandbox.h) when the user allows it, so
 * that any MCP client has them.
 *
 * What its parts share: tools.c finds the skills and offers the tools,
 * run.c runs a script in the cage.
 */
#ifndef JANAS_SKILLS_H
#define JANAS_SKILLS_H

#include <stddef.h>

#include "llm/json.h"

#define SK_MAX 64     /* skills */
#define SK_SCRIPTS 32 /* tools of their own for scripts */
#define SK_DIRS 8     /* folders of skills given with --dir */

struct sk_skill {
    char name[128], about[1024], dir[4200];
};

/* the folders searched: those given with --dir first, then every chat's */
struct sk_ctx {
    const char *dirs[SK_DIRS];
    int n_dirs;
};

void sk_tools_list(void *ctx, struct janas_buf *b);
int sk_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);

/* p, dir/a file, is inside dir, not by a link outside it: 1, or 0 */
int sk_within(const char *dir, const char *p);

/* script, a file of the skill k's folder, run in a cage with args (an
   array of strings, or NULL) and input on its standard input (or NULL):
   what it wrote and how it ended into b as a tool's result; 0 */
int sk_run(const struct sk_skill *k, const char *script,
           const struct janas_json *args, const char *input,
           struct janas_buf *b);

#endif
