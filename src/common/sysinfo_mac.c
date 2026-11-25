/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sysinfo_mac.c - see sysinfo.h and sysinfo_os.h: what macOS tells of the
 * machine, from sysctl and the Mach kernel.
 *
 * macOS gives no thread a CPU of its own: there is no affinity to set. The
 * CPUs are numbered as their counts: the performance cores first
 * (hw.perflevel0), then the efficiency ones (hw.perflevel1). Pinning a
 * thread to one of them asks the scheduler for that kind of core instead,
 * through the thread's quality of service: the interactive class for a
 * performance core, the utility class for an efficiency one.
 */
#ifdef __APPLE__
#include "sysinfo.h"
#include "pool.h"
#include "sysinfo_os.h"

#include <limits.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>

/* An integer from sysctl, or -1. */
static long long sysctl_ll(const char *name)
{
    long long v = 0;
    size_t n = sizeof v;
    if (sysctlbyname(name, &v, &n, NULL, 0) != 0)
        return -1;
    if (n == sizeof(int)) {
        int i;
        memcpy(&i, &v, sizeof i);
        return i;
    }
    return v;
}

/* The performance and efficiency cores' logical CPUs; on a processor of
   one kind, all of them performance. */
static void counts(int *np, int *ne)
{
    long long p = sysctl_ll("hw.perflevel0.logicalcpu");
    long long e = sysctl_ll("hw.perflevel1.logicalcpu");
    long long all = sysctl_ll("hw.logicalcpu");
    if (all <= 0)
        all = 1;
    if (p <= 0 || p > all) {
        p = all;
        e = 0;
    }
    if (e < 0 || p + e > all)
        e = all - p;
    *np = (int)p;
    *ne = (int)e;
}

int janas_os_allowed_cpus(int *out, int max)
{
    int np, ne;
    counts(&np, &ne);
    int n = 0;
    for (int c = 0; c < np + ne && n < max; c++)
        out[n++] = c;
    return n;
}

void janas_os_cpu_kinds(const int *cand, int nc, int *kind, int *leader)
{
    int np, ne;
    counts(&np, &ne);
    /* Apple's cores run one thread each */
    for (int i = 0; i < nc; i++) {
        kind[i] = cand[i] < np ? 0 : 1;
        leader[i] = cand[i];
    }
}

int janas_cpu_split(int *compute, int *n_compute, int *io, int *n_io, int max)
{
    int np, ne;
    counts(&np, &ne);
    *n_compute = *n_io = 0;
    for (int c = 0; c < np && *n_compute < max; c++)
        compute[(*n_compute)++] = c;
    for (int c = np; c < np + ne && *n_io < (max < 8 ? max : 8); c++)
        io[(*n_io)++] = c;
    if (*n_compute > 0 && *n_io > 0)
        return 1;
    if (*n_compute <= 0)
        return -1;
    *n_io = 0;
    for (int c = 0; c < *n_compute && c < 8; c++)
        io[(*n_io)++] = c;
    return 0;
}

int janas_cpus_save(unsigned char save[JANAS_CPU_SAVE_BYTES])
{
    (void)save; /* no affinity to keep */
    return 0;
}

void janas_cpus_restore(const unsigned char save[JANAS_CPU_SAVE_BYTES])
{
    (void)save;
}

int janas_pin_current_thread(int cpu)
{
    int np, ne;
    counts(&np, &ne);
    qos_class_t q = cpu < np ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_UTILITY;
    return pthread_set_qos_class_self_np(q, 0) == 0 ? 0 : -1;
}

int janas_exe_path(char *buf, size_t len)
{
    if (len == 0)
        return -1;
    char raw[PATH_MAX];
    uint32_t n = sizeof raw;
    if (_NSGetExecutablePath(raw, &n) != 0)
        return -1;
    char real[PATH_MAX];
    const char *p = realpath(raw, real) ? real : raw;
    size_t l = strlen(p);
    if (l == 0 || l >= len)
        return -1;
    memcpy(buf, p, l + 1);
    return (int)l;
}

void janas_cpu_name(char *buf, size_t len)
{
    buf[0] = 0;
    size_t n = len;
    if (len && sysctlbyname("machdep.cpu.brand_string", buf, &n, NULL, 0) != 0)
        buf[0] = 0;
    if (len)
        buf[len - 1] = 0;
}

int janas_on_mains(void)
{
    return -1; /* not known: IOKit's power sources not asked */
}

uint64_t janas_mem_total(void)
{
    long long v = sysctl_ll("hw.memsize");
    return v > 0 ? (uint64_t)v : 0;
}

/* Free, inactive, purgeable and speculative pages: what the system gives
   without writing anything out. */
uint64_t janas_mem_available(void)
{
    vm_statistics64_data_t s;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&s,
                          &n) != KERN_SUCCESS)
        return 0;
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS || !page)
        page = (vm_size_t)sysconf(_SC_PAGESIZE);
    uint64_t pages = (uint64_t)s.free_count + s.inactive_count +
                     s.purgeable_count + s.speculative_count;
    return pages * (uint64_t)page;
}

uint64_t janas_rss_bytes(void)
{
    mach_task_basic_info_data_t t;
    mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&t,
                  &n) != KERN_SUCCESS)
        return 0;
    return t.resident_size;
}

size_t janas_l2_bytes(void)
{
    long long v = sysctl_ll("hw.perflevel0.l2cachesize");
    if (v <= 0)
        v = sysctl_ll("hw.l2cachesize");
    return v > 0 ? (size_t)v : 0;
}

double janas_package_joules(void)
{
    return -1; /* no energy counter read */
}

#endif /* __APPLE__ */
