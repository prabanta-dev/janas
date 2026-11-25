/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * cpu_features.c - the instructions of this processor that matter to a
 * model's products, those Janas uses and those it does not use yet, for the
 * reports of janas-bench, gpu_check, janas-chat's /about and janas-server:
 * a user's report then says what the machine could do, not only what it
 * did. On x86-64 from cpuid, each family counted only where the system has
 * enabled its registers (XCR0: AVX's, AVX-512's, AMX's); on ARM from what
 * the kernel says of the processor (Linux's auxiliary vector).
 */
#include <stdio.h>
#include <string.h>

#include "common/sysinfo.h"

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>

static unsigned long long xcr0(void)
{
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (unsigned long long)hi << 32 | lo;
}
#elif defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#elif defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__aarch64__) && defined(_WIN32)
#include <windows.h>
#endif

#if defined(__aarch64__)
/* ARM's instructions that matter to a model: each with its bit in Linux's
   HWCAP (word 1) or HWCAP2 (word 2), its name in macOS's sysctl and its
   number for Windows's IsProcessorFeaturePresent (0: Windows does not say;
   PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43, PF_ARM_SVE_INSTRUCTIONS_AVAILABLE
   46, PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE 47) */
static const struct arm_feat {
    const char *name, *mac;
    int word;
    unsigned long bit;
    unsigned win;
} arm_feats[] = {
    {"fp16", "hw.optional.arm.FEAT_FP16", 1, 1UL << 10, 0},
    {"dotprod", "hw.optional.arm.FEAT_DotProd", 1, 1UL << 20, 43},
    {"i8mm", "hw.optional.arm.FEAT_I8MM", 2, 1UL << 13, 0},
    {"bf16", "hw.optional.arm.FEAT_BF16", 2, 1UL << 14, 0},
    {"sve", NULL, 1, 1UL << 22, 46},
    {"sve2", NULL, 2, 1UL << 1, 47},
    {"sme", "hw.optional.arm.FEAT_SME", 2, 1UL << 23, 0},
};

static int arm_has(const struct arm_feat *f)
{
#if defined(__linux__)
    return (getauxval(f->word == 1 ? AT_HWCAP : AT_HWCAP2) & f->bit) != 0;
#elif defined(__APPLE__)
    int v = 0;
    size_t n = sizeof v;
    return f->mac && sysctlbyname(f->mac, &v, &n, NULL, 0) == 0 && v;
#elif defined(_WIN32)
    return f->win && IsProcessorFeaturePresent(f->win);
#else
    (void)f;
    return 0;
#endif
}
#endif

int janas_arm_feature(const char *name)
{
#if defined(__aarch64__)
    for (size_t i = 0; i < sizeof arm_feats / sizeof arm_feats[0]; i++)
        if (strcmp(arm_feats[i].name, name) == 0)
            return arm_has(&arm_feats[i]);
#else
    (void)name;
#endif
    return 0;
}

static void add(char *buf, size_t len, const char *name)
{
    size_t n = strlen(buf);
    if (n + strlen(name) + 2 < len)
        snprintf(buf + n, len - n, "%s%s", n ? " " : "", name);
}

void janas_cpu_features(char *buf, size_t len)
{
    if (!len)
        return;
    buf[0] = 0;
#if defined(__x86_64__) || defined(__i386__)
    unsigned a, b, c, d, max;
    if (!__get_cpuid(0, &max, &b, &c, &d) || max < 1)
        return;
    __get_cpuid(1, &a, &b, &c, &d);
    unsigned long long x = c >> 27 & 1 ? xcr0() : 0; /* OSXSAVE */
    int avx_os = (x & 6) == 6, avx512_os = avx_os && (x & 0xe0) == 0xe0,
        amx_os = (x & 0x60000) == 0x60000;
    int avx = avx_os && c >> 28 & 1;
    if (c >> 20 & 1)
        add(buf, len, "sse4.2");
    if (avx)
        add(buf, len, "avx");
    if (avx && c >> 12 & 1)
        add(buf, len, "fma");
    if (avx && c >> 29 & 1)
        add(buf, len, "f16c");
    if (max < 7)
        return;
    unsigned b7, c7, d7, a71 = 0, d71 = 0, t;
    __cpuid_count(7, 0, t, b7, c7, d7);
    if (t >= 1)
        __cpuid_count(7, 1, a71, b, c, d71);
    if (avx && b7 >> 5 & 1)
        add(buf, len, "avx2");
    if (avx && a71 >> 4 & 1)
        add(buf, len, "avx-vnni");
    if (avx && d71 >> 4 & 1)
        add(buf, len, "avx-vnni-int8");
    if (avx && d71 >> 10 & 1)
        add(buf, len, "avx-vnni-int16");
    if (avx && d71 >> 5 & 1)
        add(buf, len, "avx-ne-convert");
    if (avx && a71 >> 23 & 1)
        add(buf, len, "avx-ifma");
    if (avx512_os) {
        static const struct {
            int reg, bit;
            const char *name;
        } f512[] = {
            {0, 16, "avx512f"},         {0, 17, "avx512dq"},
            {0, 30, "avx512bw"},        {0, 31, "avx512vl"},
            {1, 11, "avx512vnni"},      {2, 5, "avx512-bf16"},
            {3, 23, "avx512-fp16"},     {1, 1, "avx512vbmi"},
            {1, 6, "avx512vbmi2"},      {0, 21, "avx512ifma"},
            {1, 14, "avx512vpopcntdq"},
        };
        for (size_t i = 0; i < sizeof f512 / sizeof f512[0]; i++) {
            unsigned r = f512[i].reg == 0   ? b7
                         : f512[i].reg == 1 ? c7
                         : f512[i].reg == 2 ? a71
                                            : d7;
            if (r >> f512[i].bit & 1)
                add(buf, len, f512[i].name);
        }
    }
    if (avx && d71 >> 19 & 1 && max >= 0x24) { /* AVX10, and its version */
        unsigned a24, b24, c24, d24;
        __cpuid_count(0x24, 0, a24, b24, c24, d24);
        char v[32];
        snprintf(v, sizeof v, "avx10.%u%s", b24 & 0xff,
                 b24 >> 18 & 1 ? "/512" : "/256");
        add(buf, len, v);
    }
    if (amx_os) {
        if (d7 >> 25 & 1)
            add(buf, len, "amx-int8");
        if (d7 >> 22 & 1)
            add(buf, len, "amx-bf16");
        if (a71 >> 21 & 1)
            add(buf, len, "amx-fp16");
    }
#elif defined(__aarch64__)
    for (size_t i = 0; i < sizeof arm_feats / sizeof arm_feats[0]; i++)
        if (arm_has(&arm_feats[i]))
            add(buf, len, arm_feats[i].name);
#endif
}
