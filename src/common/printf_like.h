/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * printf_like.h - the format attribute, in the spelling the compiler takes
 * (JANAS_PRINTF_LIKE). gnu_printf checks the GNU extensions in a format
 * string; Apple clang does not accept the argument, so it gets the plain
 * one there, and -Wformat on the other compilers does not move.
 */
#ifndef JANAS_COMMON_PRINTF_LIKE_H
#define JANAS_COMMON_PRINTF_LIKE_H

#if defined(__APPLE__)
#define JANAS_PRINTF_LIKE(a, b) __attribute__((format(printf, a, b)))
#else
#define JANAS_PRINTF_LIKE(a, b) __attribute__((format(gnu_printf, a, b)))
#endif

#endif
