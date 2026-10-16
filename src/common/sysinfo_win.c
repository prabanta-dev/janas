/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sysinfo_win.c - see sysinfo.h and sysinfo_os.h: what Windows tells of the
 * machine. A CPU is numbered as Windows numbers its logical processors
 * within their group, the groups one after the other (64 a group).
 *
 * The kinds of core come from GetLogicalProcessorInformationEx: a hybrid
 * processor gives its cores efficiency classes, the performance cores the
 * highest. The low-power efficiency cores (Meteor Lake's, on a tile of
 * their own) are told apart by the level-3 cache they do not share with
 * the others. Not by the clock, as on Linux: on Windows 11 (build 26300)
 * the clocks Windows reports kept the two low-power cores of a Core Ultra
 * 9 185H and left out its eight ordinary efficiency cores (12 Oct 2026);
 * the clock is the fallback where Windows says nothing of the caches.
 */
#ifdef _WIN32
#include "sysinfo.h"
#include "pool.h"
#include "sysinfo_os.h"

#include <windows.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#include <powrprof.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* PROCESSOR_POWER_INFORMATION, which MinGW's headers lack */
struct power_info {
    ULONG number, max_mhz, current_mhz, mhz_limit, max_idle_state,
        current_idle_state;
};

/* The physical cores: each one's CPUs and efficiency class. */
struct core {
    int first, n_cpus, cls;
    int cpus[8];
};

static int cores_of(struct core *out, int max)
{
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0)
        return -1;
    char *buf = malloc(len);
    if (!buf || !GetLogicalProcessorInformationEx(
                    RelationProcessorCore,
                    (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len)) {
        free(buf);
        return -1;
    }
    int n = 0;
    for (DWORD off = 0; off < len && n < max;) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + off);
        if (e->Relationship == RelationProcessorCore) {
            struct core *c = &out[n++];
            c->n_cpus = 0;
            c->first = -1;
            /* the byte after Flags: EfficiencyClass in the headers of
               MinGW-w64 12 and later, Reserved[0] in the older ones */
            c->cls = ((const BYTE *)&e->Processor)[1];
            for (WORD g = 0; g < e->Processor.GroupCount; g++) {
                GROUP_AFFINITY *ga = &e->Processor.GroupMask[g];
                for (int b = 0; b < 64; b++)
                    if ((ga->Mask >> b) & 1) {
                        int cpu = ga->Group * 64 + b;
                        if (c->n_cpus < 8)
                            c->cpus[c->n_cpus++] = cpu;
                        if (c->first < 0 || cpu < c->first)
                            c->first = cpu;
                    }
            }
        }
        off += e->Size;
    }
    free(buf);
    return n;
}

int janas_os_allowed_cpus(int *out, int max)
{
    GROUP_AFFINITY ga;
    if (!GetThreadGroupAffinity(GetCurrentThread(), &ga))
        return -1;
    int n = 0;
    for (int b = 0; b < 64 && n < max; b++)
        if ((ga.Mask >> b) & 1)
            out[n++] = ga.Group * 64 + b;
    return n;
}

/* The level-3 cache that serves the most CPUs, the one the performance
   and the ordinary efficiency cores share: Meteor Lake's low-power cores,
   on a tile of their own, are outside it. 1, or 0 when Windows says
   nothing of the caches. Its first group only (64 CPUs), as the old
   CACHE_RELATIONSHIP of MinGW's headers has it, and the new one starts. */
static int main_l3(GROUP_AFFINITY *out)
{
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, NULL, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0)
        return 0;
    char *buf = malloc(len);
    if (!buf || !GetLogicalProcessorInformationEx(
                    RelationCache,
                    (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len)) {
        free(buf);
        return 0;
    }
    int best = 0;
    for (DWORD off = 0; off < len;) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + off);
        if (e->Relationship == RelationCache && e->Cache.Level == 3) {
            int cpus = __builtin_popcountll(
                (unsigned long long)e->Cache.GroupMask.Mask);
            if (cpus > best) {
                best = cpus;
                *out = e->Cache.GroupMask;
            }
        }
        off += e->Size;
    }
    free(buf);
    return best > 0;
}

/* Each CPU's top clock in MHz (0 where unknown), indexed by CPU. */
static void top_clocks(long *mhz, int max)
{
    memset(mhz, 0, sizeof(*mhz) * (size_t)max);
    int n = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (n <= 0 || n > max)
        return;
    struct power_info *pi = calloc((size_t)n, sizeof *pi);
    if (pi && CallNtPowerInformation(ProcessorInformation, NULL, 0, pi,
                                     (ULONG)(sizeof *pi * (size_t)n)) == 0)
        for (int i = 0; i < n; i++)
            if ((int)pi[i].number < max)
                mhz[pi[i].number] = (long)pi[i].max_mhz;
    free(pi);
}

void janas_os_cpu_kinds(const int *cand, int nc, int *kind, int *leader)
{
    static struct core cores[JANAS_MAX_CPUS];
    int n = cores_of(cores, JANAS_MAX_CPUS);
    int top_cls = 0, bottom_cls = 255;
    for (int i = 0; i < n; i++) {
        if (cores[i].cls > top_cls)
            top_cls = cores[i].cls;
        if (cores[i].cls < bottom_cls)
            bottom_cls = cores[i].cls;
    }
    int hybrid = n > 0 && top_cls != bottom_cls;
    GROUP_AFFINITY l3;
    int have_l3 = main_l3(&l3);
    static long mhz[JANAS_MAX_CPUS];
    if (!have_l3)
        top_clocks(mhz, JANAS_MAX_CPUS);
    long efreq_max = 0;
    for (int i = 0; i < nc; i++) {
        kind[i] = 0;
        leader[i] = cand[i];
        for (int k = 0; k < n; k++)
            for (int j = 0; j < cores[k].n_cpus; j++)
                if (cores[k].cpus[j] == cand[i]) {
                    leader[i] = cores[k].first;
                    kind[i] = hybrid && cores[k].cls != top_cls ? 1 : 0;
                }
        /* the low-power ones: outside the L3 the others share */
        if (have_l3 && kind[i] == 1 &&
            (cand[i] / 64 != l3.Group || !((l3.Mask >> (cand[i] % 64)) & 1)))
            kind[i] = 2;
        long f = !have_l3 && cand[i] < JANAS_MAX_CPUS ? mhz[cand[i]] : 0;
        if (kind[i] == 1 && f > efreq_max)
            efreq_max = f;
    }
    /* without a word on the caches: the clock, as Linux has it */
    for (int i = 0; i < nc && !have_l3; i++) {
        long f = cand[i] < JANAS_MAX_CPUS ? mhz[cand[i]] : 0;
        if (kind[i] == 1 && f > 0 && f < efreq_max * 85 / 100)
            kind[i] = 2;
    }
}

int janas_cpu_split(int *compute, int *n_compute, int *io, int *n_io, int max)
{
    int cand[JANAS_MAX_CPUS];
    int nc = janas_os_allowed_cpus(cand, JANAS_MAX_CPUS);
    if (nc <= 0)
        return -1;
    int kind[JANAS_MAX_CPUS], leader[JANAS_MAX_CPUS];
    janas_os_cpu_kinds(cand, nc, kind, leader);
    /* as Linux's cpu_core and cpu_atom: the performance cores, the others */
    *n_compute = *n_io = 0;
    int max_io = max < 8 ? max : 8;
    for (int i = 0; i < nc; i++) {
        if (kind[i] == 0 && *n_compute < max)
            compute[(*n_compute)++] = cand[i];
        else if (kind[i] != 0 && *n_io < max_io)
            io[(*n_io)++] = cand[i];
    }
    if (*n_compute > 0 && *n_io > 0)
        return 1;
    *n_compute = 0;
    for (int i = 0; i < nc && *n_compute < max; i++)
        compute[(*n_compute)++] = cand[i];
    *n_io = *n_compute < 8 ? *n_compute : 8;
    return 0;
}

int janas_cpus_save(unsigned char save[JANAS_CPU_SAVE_BYTES])
{
    _Static_assert(sizeof(GROUP_AFFINITY) <= JANAS_CPU_SAVE_BYTES, "cpu set");
    return GetThreadGroupAffinity(GetCurrentThread(), (GROUP_AFFINITY *)save)
               ? 0
               : -1;
}

void janas_cpus_restore(const unsigned char save[JANAS_CPU_SAVE_BYTES])
{
    SetThreadGroupAffinity(GetCurrentThread(), (const GROUP_AFFINITY *)save,
                           NULL);
}

int janas_pin_current_thread(int cpu)
{
    if (cpu < 0)
        return -1;
    GROUP_AFFINITY ga = {.Mask = (KAFFINITY)1 << (cpu % 64),
                         .Group = (WORD)(cpu / 64)};
    return SetThreadGroupAffinity(GetCurrentThread(), &ga, NULL) ? 0 : -1;
}

int janas_exe_path(char *buf, size_t len)
{
    if (len == 0)
        return -1;
    DWORD n = GetModuleFileNameA(NULL, buf, (DWORD)len);
    if (n == 0 || n >= len)
        return -1;
    for (char *p = buf; *p; p++)
        if (*p == '\\')
            *p = '/';
    return (int)n;
}

int janas_cpu_arm_dotprod(void)
{
#if defined(__aarch64__)
    return IsProcessorFeaturePresent(
               43 /* PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE */) != 0;
#else
    return 0;
#endif
}

int janas_cpu_arm_i8mm(void)
{
#if defined(__aarch64__)
    if (IsProcessorFeaturePresent(
            66 /* PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE */))
        return 1;
    /* Windows releases that predate the flag: ID_AA64ISAR1_EL1 as the
       registry keeps it (CP 4031), I8MM in bits 55-52 */
    uint64_t isar1 = 0;
    DWORD n = sizeof(isar1);
    if (RegGetValueA(HKEY_LOCAL_MACHINE,
                     "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     "CP 4031", RRF_RT_REG_QWORD, NULL, &isar1,
                     &n) != ERROR_SUCCESS)
        return 0;
    return ((isar1 >> 52) & 15) != 0;
#else
    return 0;
#endif
}

void janas_cpu_name(char *buf, size_t len)
{
#if !(defined(__x86_64__) || defined(__i386__))
    /* ARM64 has no brand-string instruction: Windows keeps the name in the
     * registry */
    DWORD n = (DWORD)len;
    buf[0] = 0;
    if (RegGetValueA(HKEY_LOCAL_MACHINE,
                     "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     "ProcessorNameString", RRF_RT_REG_SZ, NULL, buf,
                     &n) != ERROR_SUCCESS)
        buf[0] = 0;
#else
    /* the brand string, from the processor itself */
    unsigned r[12];
    buf[0] = 0;
    if (__get_cpuid_max(0x80000000, NULL) < 0x80000004)
        return;
    for (unsigned i = 0; i < 3; i++)
        __get_cpuid(0x80000002 + i, &r[i * 4], &r[i * 4 + 1], &r[i * 4 + 2],
                    &r[i * 4 + 3]);
    char s[49];
    memcpy(s, r, 48);
    s[48] = 0;
    char *p = s;
    while (*p == ' ')
        p++;
    snprintf(buf, len, "%s", p);
#endif
}

int janas_on_mains(void)
{
    SYSTEM_POWER_STATUS ps;
    if (!GetSystemPowerStatus(&ps) || ps.ACLineStatus == 255)
        return -1;
    /* a desktop says "online" with no battery: as Linux's unknown */
    if (ps.BatteryFlag & 128)
        return -1;
    return ps.ACLineStatus == 1;
}

uint64_t janas_mem_available(void)
{
    MEMORYSTATUSEX m = {.dwLength = sizeof m};
    return GlobalMemoryStatusEx(&m) ? m.ullAvailPhys : 0;
}

uint64_t janas_mem_total(void)
{
    MEMORYSTATUSEX m = {.dwLength = sizeof m};
    return GlobalMemoryStatusEx(&m) ? m.ullTotalPhys : 0;
}

#endif /* _WIN32 */
