/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * run.c - janas-git runs git (see git.h) as the services run a program
 * (services/common/run.h): no shell, no terminal, so that a password asked
 * (ssh, https) fails at once, and a time limit; with git's own settings
 * for that.
 */
#include <stdio.h>

#include "git.h"
#include "services/common/run.h"

int gt_run(const char *dir, const char *const *argv, int timeout_ms,
           struct janas_buf *out, struct janas_buf *err, char *why,
           size_t why_len)
{
    static const char *const env[] = {
        "GIT_TERMINAL_PROMPT=0", "GIT_EDITOR=true",
        "GIT_OPTIONAL_LOCKS=0", /* status takes no lock */
        "LC_ALL=C",             /* git's words, in English */
        NULL};
    const char *av[64];
    size_t n = 0;
    av[n++] = "git";
    for (size_t i = 0; argv[i] && n < 63; i++)
        av[n++] = argv[i];
    av[n] = NULL;
    return janas_run(dir, av, env, timeout_ms, out, err, why, why_len);
}
