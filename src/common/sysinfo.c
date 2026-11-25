/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sysinfo.c - see sysinfo.h. The order of the CPUs, for every system, and
 * what POSIX tells alike on Linux and macOS. What each system tells of its
 * CPUs and memory: sysinfo_linux.c, sysinfo_mac.c, sysinfo_win.c.
 */
#define _GNU_SOURCE
#include "sysinfo.h"
#include "pool.h"
#include "sysinfo_os.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int janas_cpu_layout(const int *allowed, int n_allowed,
                     struct janas_cpu_layout *l)
{
    int cand[JANAS_MAX_CPUS], nc = 0;
    if (allowed) {
        for (int i = 0; i < n_allowed && nc < JANAS_MAX_CPUS; i++)
            cand[nc++] = allowed[i];
    } else {
        nc = janas_os_allowed_cpus(cand, JANAS_MAX_CPUS);
    }
    if (nc <= 0)
        return -1;
    /* kind 0 performance, 1 efficiency, 2 low-power efficiency */
    int kind[JANAS_MAX_CPUS], leader[JANAS_MAX_CPUS];
    janas_os_cpu_kinds(cand, nc, kind, leader);
    l->n = l->n_p_cores = l->n_p_threads = 0;
    /* P leaders, P siblings, E leaders, E siblings */
    for (int pass = 0; pass < 4; pass++)
        for (int i = 0; i < nc; i++) {
            int k = pass < 2 ? 0 : 1, want_leader = pass % 2 == 0;
            if (kind[i] != k)
                continue;
            int first = 1;
            for (int j = 0; j < i && first; j++)
                first = !(kind[j] == k && leader[j] == leader[i]);
            if (first != want_leader)
                continue;
            l->cpus[l->n++] = cand[i];
            if (pass == 0)
                l->n_p_cores++;
            if (pass <= 1)
                l->n_p_threads++;
        }
    if (l->n == 0)
        return -1;
    return 0;
}

#ifndef _WIN32
#include <sys/resource.h>
#include <unistd.h>

double janas_cpu_seconds(void)
{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0)
        return -1;
    return (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec * 1e-6 +
           (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec * 1e-6;
}

int janas_cpu_count(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

#endif /* !_WIN32 */
