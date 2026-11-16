/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * paths.h - paths as Janas keeps them, on every system: absolute, with '/'
 * between their parts ("C:/Users/x/work" on Windows, whose calls take '/'
 * as they take '\\'). The paths the system gives (realpath, the
 * environment's folders) have their '\\' turned to '/', so that "inside a
 * folder" is the same comparison everywhere; on Windows it ignores the
 * case of the letters, as Windows does. Brebus, janas-docs and janas-files
 * keep their paths so.
 */
#ifndef JANAS_COMMON_PATHS_H
#define JANAS_COMMON_PATHS_H

#include <stddef.h>

/* '\\' turned to '/' (Windows; nothing elsewhere) */
void janas_path_slashes(char *p);
/* absolute: "/x"; on Windows "C:/x", "C:\\x", "//server/x" */
int janas_path_abs(const char *p);
/* a root: "/"; on Windows "C:/" */
int janas_path_root(const char *p);
/* realpath of an absolute path, its '\\' turned and no '/' at its end
   (out of 4096 bytes); NULL: not there, or not absolute */
char *janas_path_real(const char *p, char *out);
/* path inside dir, not dir itself */
int janas_path_in(const char *path, const char *dir);
/* Windows: the user's Documents (or Downloads) folder into out, 0; -1
   elsewhere, where user-dirs.dirs says it */
int janas_known_dir(int downloads, char *out, size_t n);

/* between the paths of a list in the environment, as PATH has it */
#ifdef _WIN32
#define JANAS_PATH_LIST_SEP ';'
#define JANAS_PATH_LIST_SEP_S ";"
#else
#define JANAS_PATH_LIST_SEP ':'
#define JANAS_PATH_LIST_SEP_S ":"
#endif

#endif
