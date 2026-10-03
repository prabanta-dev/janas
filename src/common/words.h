/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * words.h - the words a user gives a program (options, their values, the
 * keys of a configuration, the names of variables), read strictly: a word
 * that is not known is refused with the known one nearest to it, a number
 * is read whole and within its limits, so that "--ctx 32k" is not taken
 * for 32 nor "--tmep" left aside in silence.
 */
#ifndef JANAS_COMMON_WORDS_H
#define JANAS_COMMON_WORDS_H

#include <stddef.h>

/* Of names[0..n) (NULL ends them early), the one nearest to w: a letter or
   two away (three for long names), two letters swapped, or one the start
   of the other; NULL when none is that near. Case does not count. */
const char *janas_closest(const char *w, const char *const *names, size_t n);

/* "w is not <what>", and " (did you mean X?)" when one of names is near:
   into buf, always ended. */
void janas_unknown_word(char *buf, size_t cap, const char *w, const char *what,
                        const char *const *names, size_t n);

/* v read whole as a number within [lo, hi]: 0, or -1 with the reason in
   why ("32k is not a whole number", "70000 is above 65535"). */
int janas_word_int(const char *v, long long lo, long long hi, long long *out,
                   char *why, size_t why_len);
int janas_word_real(const char *v, double lo, double hi, double *out, char *why,
                    size_t why_len);

/* The index of v among names[0..n), or -1 with the reason in why (the
   choices, and the nearest). */
int janas_word_choice(const char *v, const char *const *names, size_t n,
                      char *why, size_t why_len);

#endif
