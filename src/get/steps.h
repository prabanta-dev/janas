/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * steps.h - what janas-get does to a file: download it and check it, run a
 * converter beside janas-get, check what it wrote.
 */
#ifndef JANAS_GET_STEPS_H
#define JANAS_GET_STEPS_H

#include <stddef.h>

/* dir/the last part of path */
void get_local(char *out, size_t len, const char *dir, const char *path);

/* File path of repo downloaded into dir (resumed if part of it is there),
   its SHA-256 checked against want (NULL: the one Hugging Face lists). 0,
   or -1 having said why. */
int get_fetch(const char *repo, const char *path, const char *dir,
              const char *want);

/* 1 when file's SHA-256 is want; 0 when it is another, -1 unreadable. */
int get_same(const char *file, const char *want);

/* A converter of the same build (gguf2jns, jns_planes, hf2jns_mtp) run on
   argv (argv[0] its name, NULL-terminated). Its exit status, or -1. */
int get_run(const char *tool, char *const argv[]);

#endif
