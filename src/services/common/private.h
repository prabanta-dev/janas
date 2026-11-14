/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * private.h - the files no service opens, wherever it is asked to look
 * (private.c): those that hold keys, passwords and tokens, a browser's or
 * a mail program's data, a shell's history. janas-files and janas-docs
 * keep away from the same ones.
 */
#ifndef JANAS_SERVICES_PRIVATE_H
#define JANAS_SERVICES_PRIVATE_H

/* 1 when path is root or below it */
int janas_path_under(const char *path, const char *root);

/* 1 when path (absolute, its links followed) is private: one of the
   folders and files of the home below, or anything below them, or a part
   of it named as a private file is (.env, *.pem, id_rsa...). */
int janas_private_path(const char *home, const char *path);

#endif
