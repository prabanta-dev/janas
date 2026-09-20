/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sysinfo.h - what the machine offers: CPUs for computing and for I/O
 * threads, free memory. Used for the defaults of the library.
 */
#ifndef JANAS_COMMON_SYSINFO_H
#define JANAS_COMMON_SYSINFO_H

#include <stddef.h>
#include <stdint.h>

/*
 * CPUs for the compute threads and for the I/O threads, at most max each.
 * On a hybrid processor the performance cores compute and the efficiency
 * cores serve the I/O; otherwise every CPU the process may use computes and
 * the I/O threads are left unpinned (*n_io > 0 with io unused: see the
 * return value). Returns 1 when io lists CPUs, 0 when the I/O threads
 * should not be pinned, -1 when nothing is known.
 */
int janas_cpu_split(int *compute, int *n_compute, int *io, int *n_io, int max);

/*
 * The CPUs for computing, ordered so that useful thread counts are
 * prefixes: one CPU per performance core, then the other threads of those
 * cores, then the efficiency cores (low-power efficiency cores, much slower,
 * left out). n_p_cores and n_p_threads count the first two groups; n is the
 * total. Without topology information every CPU counts as a performance
 * core of its own. allowed lists the candidates (NULL: every CPU the
 * process may use). Returns 0 or -1.
 */
#define JANAS_MAX_CPUS 256
struct janas_cpu_layout {
    int cpus[JANAS_MAX_CPUS];
    int n, n_p_cores, n_p_threads;
};
int janas_cpu_layout(const int *allowed, int n_allowed,
                     struct janas_cpu_layout *l);

/* The processor's name ("Intel(R) Core(TM) Ultra 9 185H"), or "". */
void janas_cpu_name(char *buf, size_t len);

/* Whether the machine runs on mains power: 1 yes, 0 on battery, -1 unknown
   (no power supply information: a desktop, most likely). */
int janas_on_mains(void);

/* Memory available to new allocations, in bytes (0 if unknown). */
uint64_t janas_mem_available(void);

/* Installed memory, in bytes (0 if unknown). */
uint64_t janas_mem_total(void);

#endif
