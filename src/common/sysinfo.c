/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sysinfo.c - see sysinfo.h. Linux: sysfs and procfs.
 */
#define _GNU_SOURCE
#include "sysinfo.h"

#include <dirent.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reads a CPU list ("0-11,14") from a sysfs file. Returns the count. */
static int read_cpu_list(const char *path, int *out, int max)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    char buf[1024];
    int n = 0;
    if (fgets(buf, sizeof(buf), f)) {
        for (char *p = buf; *p && *p != '\n';) {
            char *end;
            long a = strtol(p, &end, 10), b = a;
            if (end == p)
                break;
            if (*end == '-')
                b = strtol(end + 1, &end, 10);
            for (long c = a; c <= b && n < max; c++)
                out[n++] = (int)c;
            p = *end == ',' ? end + 1 : end;
        }
    }
    fclose(f);
    return n;
}

int janas_cpu_split(int *compute, int *n_compute, int *io, int *n_io, int max)
{
    *n_compute = read_cpu_list("/sys/devices/cpu_core/cpus", compute, max);
    *n_io = read_cpu_list("/sys/devices/cpu_atom/cpus", io, max < 8 ? max : 8);
    if (*n_compute > 0 && *n_io > 0)
        return 1;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0)
        return -1;
    *n_compute = 0;
    for (int c = 0; c < CPU_SETSIZE && *n_compute < max; c++)
        if (CPU_ISSET(c, &set))
            compute[(*n_compute)++] = c;
    if (*n_compute == 0)
        return -1;
    *n_io = *n_compute < 8 ? *n_compute : 8;
    return 0;
}

/* The lowest CPU sharing a physical core with cpu, or -1. */
static int core_leader(int cpu)
{
    char path[96];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
             cpu);
    int list[64];
    int n = read_cpu_list(path, list, 64);
    if (n <= 0)
        return -1;
    int lo = list[0];
    for (int i = 1; i < n; i++)
        if (list[i] < lo)
            lo = list[i];
    return lo;
}

/* A number from a sysfs file, or -1. */
static long read_long(const char *path)
{
    FILE *f = fopen(path, "r");
    long v = -1;
    if (f && fscanf(f, "%ld", &v) != 1)
        v = -1;
    if (f)
        fclose(f);
    return v;
}

static int in_list(const int *list, int n, int cpu)
{
    for (int i = 0; i < n; i++)
        if (list[i] == cpu)
            return 1;
    return 0;
}

int janas_cpu_layout(const int *allowed, int n_allowed,
                     struct janas_cpu_layout *l)
{
    int cand[JANAS_MAX_CPUS], nc = 0;
    if (allowed) {
        for (int i = 0; i < n_allowed && nc < JANAS_MAX_CPUS; i++)
            cand[nc++] = allowed[i];
    } else {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) != 0)
            return -1;
        for (int c = 0; c < CPU_SETSIZE && nc < JANAS_MAX_CPUS; c++)
            if (CPU_ISSET(c, &set))
                cand[nc++] = c;
    }
    if (nc == 0)
        return -1;
    /* hybrid processors list their core types */
    int pcores[JANAS_MAX_CPUS], ecores[JANAS_MAX_CPUS];
    int np =
        read_cpu_list("/sys/devices/cpu_core/cpus", pcores, JANAS_MAX_CPUS);
    int ne =
        read_cpu_list("/sys/devices/cpu_atom/cpus", ecores, JANAS_MAX_CPUS);
    int kind[JANAS_MAX_CPUS], leader[JANAS_MAX_CPUS]; /* kind 0 P, 1 E, 2 LP */
    long efreq_max = 0, freq[JANAS_MAX_CPUS];
    for (int i = 0; i < nc; i++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
                 cand[i]);
        freq[i] = read_long(path);
        kind[i] = np > 0 && !in_list(pcores, np, cand[i]) &&
                          in_list(ecores, ne, cand[i])
                      ? 1
                      : 0;
        if (kind[i] == 1 && freq[i] > efreq_max)
            efreq_max = freq[i];
        leader[i] = core_leader(cand[i]);
        if (leader[i] < 0)
            leader[i] = cand[i];
    }
    /* efficiency cores well below the others' top clock are the low-power
       ones (on a separate tile, without the shared L3) */
    for (int i = 0; i < nc; i++)
        if (kind[i] == 1 && freq[i] > 0 && freq[i] < efreq_max * 85 / 100)
            kind[i] = 2;
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

void janas_cpu_name(char *buf, size_t len)
{
    buf[0] = 0;
    FILE *f = fopen("/proc/cpuinfo", "r");
    char line[512];
    while (f && fgets(line, sizeof(line), f)) {
        char *c = strchr(line, ':');
        if (strncmp(line, "model name", 10) == 0 && c) {
            c++;
            while (*c == ' ')
                c++;
            c[strcspn(c, "\n")] = 0;
            snprintf(buf, len, "%s", c);
            break;
        }
    }
    if (f)
        fclose(f);
}

int janas_on_mains(void)
{
    /* every power supply of type Mains; on mains if any is online */
    DIR *d = opendir("/sys/class/power_supply");
    if (!d)
        return -1;
    int found = 0, online = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.')
            continue;
        char path[512], type[32] = "";
        snprintf(path, sizeof(path), "/sys/class/power_supply/%s/type",
                 e->d_name);
        FILE *f = fopen(path, "r");
        if (!f)
            continue;
        if (!fgets(type, sizeof(type), f))
            type[0] = 0;
        fclose(f);
        if (strncmp(type, "Mains", 5) != 0)
            continue;
        found = 1;
        snprintf(path, sizeof(path), "/sys/class/power_supply/%s/online",
                 e->d_name);
        if (read_long(path) > 0)
            online = 1;
    }
    closedir(d);
    return found ? online : -1;
}

/* A field of /proc/meminfo, in bytes (0 if missing). */
static uint64_t meminfo(const char *field)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f)
        return 0;
    char line[256];
    unsigned long long kb = 0;
    size_t n = strlen(field);
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, field, n) == 0 && line[n] == ':' &&
            sscanf(line + n + 1, "%llu", &kb) == 1)
            break;
    fclose(f);
    return (uint64_t)kb << 10;
}

uint64_t janas_mem_available(void)
{
    return meminfo("MemAvailable");
}

uint64_t janas_mem_total(void)
{
    return meminfo("MemTotal");
}
