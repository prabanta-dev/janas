/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sysinfo_os.h - what each system tells of its CPUs, for sysinfo.c:
 * Linux's in sysinfo_linux.c, macOS's in sysinfo_mac.c, Windows's in
 * sysinfo_win.c.
 */
#ifndef JANAS_COMMON_SYSINFO_OS_H
#define JANAS_COMMON_SYSINFO_OS_H

/* The CPUs the calling thread may use, at most max. The count, or -1. */
int janas_os_allowed_cpus(int *out, int max);

/* For each of the nc CPUs of cand: its kind (0 performance, 1 efficiency,
   2 low-power efficiency) and the lowest CPU of its physical core. Every
   CPU a performance core of its own where nothing is known. */
void janas_os_cpu_kinds(const int *cand, int nc, int *kind, int *leader);

#endif
