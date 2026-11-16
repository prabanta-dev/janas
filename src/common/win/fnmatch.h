/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* fnmatch.h for Windows (src/common/win/): a name against a shell pattern
   (*, ?, [set], [!set], [a-z]), as POSIX's; FNM_CASEFOLD ignores case
   (win_posix.c). */
#ifndef JANAS_WIN_FNMATCH_H
#define JANAS_WIN_FNMATCH_H
#define FNM_NOMATCH 1
#define FNM_NOESCAPE 2
#define FNM_PATHNAME 4
#define FNM_PERIOD 8
#define FNM_CASEFOLD 16
int fnmatch(const char *pattern, const char *name, int flags);
#endif
