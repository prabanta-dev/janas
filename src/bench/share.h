/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * share.h - janas-bench --share: the report just written, made into the
 * address of a new issue of the project's repository with the hardware
 * report's form filled in. Nothing is sent: the address is printed, and
 * opened in the browser only when the user says yes; the user reads the
 * form there and sends it, or does not.
 */
#ifndef JANAS_BENCH_SHARE_H
#define JANAS_BENCH_SHARE_H

/* The report in the file at path, offered as an issue. */
void bench_share(const char *path);

#endif
