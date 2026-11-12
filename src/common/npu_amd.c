/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * npu_amd.c - AMD's NPUs (XDNA), found and named (see npu_amd.h). Janas does
 * not compute on them yet; saying that one is there, which generation and
 * whether its driver is loaded, turns a report's "no NPU" into what a
 * tester can check and a reason the engine leaves it alone.
 *
 * Found by the PCI ids AMD's amdxdna driver takes: 1022:1502 for the first
 * generation (Phoenix and Hawk Point, the Ryzen 7040 and 8040), 1022:17f0
 * for the second (Strix, Strix Halo, Krackan Point, the Ryzen AI 300).
 * Linux reads them from sysfs, Windows from the PCI devices SetupAPI lists.
 */
#include "npu_amd.h"

#include <stdio.h>
#include <string.h>

static const char *generation(unsigned dev)
{
    return dev == 0x1502 ? "first generation (XDNA: Phoenix, Hawk Point)"
           : dev == 0x17f0
               ? "second generation (XDNA 2: Strix, Strix Halo, Krackan "
                 "Point)"
               : NULL;
}

#if defined(__linux__)
#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

static unsigned read_hex(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    unsigned v = 0;
    if (fscanf(f, "%x", &v) != 1)
        v = 0;
    fclose(f);
    return v;
}

int janas_npu_amd_find(char *desc, size_t len)
{
    DIR *d = opendir("/sys/bus/pci/devices");
    if (!d)
        return 0;
    int found = 0;
    struct dirent *e;
    while (!found && (e = readdir(d))) {
        if (e->d_name[0] == '.')
            continue;
        char p[512], link[PATH_MAX];
        snprintf(p, sizeof p, "/sys/bus/pci/devices/%s/vendor", e->d_name);
        if (read_hex(p) != 0x1022)
            continue;
        snprintf(p, sizeof p, "/sys/bus/pci/devices/%s/device", e->d_name);
        const char *g = generation(read_hex(p));
        if (!g)
            continue;
        /* the driver bound to it, if any: amdxdna is AMD's */
        snprintf(p, sizeof p, "/sys/bus/pci/devices/%s/driver", e->d_name);
        ssize_t n = readlink(p, link, sizeof link - 1);
        const char *drv = "no driver loaded";
        if (n > 0) {
            link[n] = 0;
            const char *b = strrchr(link, '/');
            drv = b ? b + 1 : link;
        }
        snprintf(desc, len, "AMD NPU, %s, driver %s", g, drv);
        found = 1;
    }
    closedir(d);
    return found;
}

#elif defined(_WIN32)
#include <windows.h>
#include <setupapi.h>

int janas_npu_amd_find(char *desc, size_t len)
{
    HDEVINFO set = SetupDiGetClassDevsA(NULL, "PCI", NULL,
                                        DIGCF_PRESENT | DIGCF_ALLCLASSES);
    if (set == INVALID_HANDLE_VALUE)
        return 0;
    int found = 0;
    SP_DEVINFO_DATA info = {.cbSize = sizeof info};
    for (DWORD i = 0; !found && SetupDiEnumDeviceInfo(set, i, &info); i++) {
        char ids[2048];
        if (!SetupDiGetDeviceRegistryPropertyA(set, &info, SPDRP_HARDWAREID,
                                               NULL, (BYTE *)ids,
                                               sizeof ids - 2, NULL))
            continue;
        ids[sizeof ids - 2] = ids[sizeof ids - 1] = 0;
        /* a list of strings, the first the most precise */
        unsigned dev = 0;
        if (strncmp(ids, "PCI\\VEN_1022&DEV_", 17) == 0)
            sscanf(ids + 17, "%4x", &dev);
        const char *g = generation(dev);
        if (!g)
            continue;
        char name[256] = "";
        SetupDiGetDeviceRegistryPropertyA(set, &info, SPDRP_DEVICEDESC, NULL,
                                          (BYTE *)name, sizeof name - 1, NULL);
        name[sizeof name - 1] = 0;
        snprintf(desc, len, "AMD NPU, %s%s%s", g, name[0] ? ", " : "", name);
        found = 1;
    }
    SetupDiDestroyDeviceInfoList(set);
    return found;
}

#else
int janas_npu_amd_find(char *desc, size_t len)
{
    (void)desc, (void)len;
    return 0;
}
#endif
