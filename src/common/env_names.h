/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * env_names.h - the variables of the environment Janas reads (env_names.c),
 * so that one set with a name Janas does not know - JANAS_EXPERT_BIT for
 * JANAS_EXPERT_BITS - is told when a program starts, instead of doing
 * nothing in silence. A warning, not a refusal: a variable may be meant for
 * another version of Janas.
 */
#ifndef JANAS_COMMON_ENV_NAMES_H
#define JANAS_COMMON_ENV_NAMES_H

#include <stddef.h>

/* every JANAS_ variable read, by the programs, the library or the
   services; tests/test_env_names.c checks it against the sources */
extern const char *const JANAS_ENV_NAMES[];
extern const size_t JANAS_N_ENV_NAMES;

/* On stderr, prefixed by prog, a line for each JANAS_ variable of the
   environment not among them, with the name nearest to it. The number of
   them. */
int janas_env_check(const char *prog);

#endif
