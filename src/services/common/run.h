/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * run.h - a program run by a service (run.c): with its arguments as they
 * are, no shell between; in a session of its own, without a terminal and
 * with nothing on its input, so that a question it asks (a password)
 * fails at once instead of waiting for an answer nobody can give; with a
 * time limit, after which it and what it started are stopped. POSIX only.
 */
#ifndef JANAS_SERVICES_RUN_H
#define JANAS_SERVICES_RUN_H

#include <stddef.h>

#include "llm/json.h"

#define JANAS_RUN_OUT_MAX (1u << 20) /* of each stream kept */

/*
 * argv[0] (looked for in PATH) with argv (NULL-ended), in the directory
 * dir (NULL: where the service runs), with env added to the environment
 * ("NAME=value", NULL-ended; NULL: none), at most timeout_ms. Its output
 * into out, what it wrote on its error stream into err, each cut at
 * JANAS_RUN_OUT_MAX and ended by a 0 not counted. Its exit status, or -1
 * when it could not be run or took too long (the reason in why).
 */
int janas_run(const char *dir, const char *const *argv, const char *const *env,
              int timeout_ms, struct janas_buf *out, struct janas_buf *err,
              char *why, size_t why_len);

#endif
