/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* How this Brebus was installed, and so what an update replaces
   (update.h). */
#include "update.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#define ARCH "x86_64"
#elif defined(__aarch64__) || defined(_M_ARM64)
#define ARCH "aarch64"
#else
#define ARCH ""
#endif

__attribute__((unused)) static int is_file(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

/* a folder this user may write in: where the new version goes beside */
static int writable(const char *dir)
{
    return access(dir, W_OK) == 0;
}

static void parent(const char *p, char *out, size_t len)
{
    snprintf(out, len, "%s", p);
    char *s = strrchr(out, '/');
    if (s && s != out)
        *s = 0;
}

void brebus_up_asset(enum brebus_up_kind kind, char *out, size_t len)
{
    const char *a = "";
    switch (kind) {
#if defined(_WIN32)
    case BREBUS_UP_SETUP:
        a = !strcmp(ARCH, "x86_64") ? "brebus-windows-x86_64-setup.exe"
                                    : "brebus-windows-aarch64-setup.exe";
        break;
    case BREBUS_UP_FOLDER:
        a = !strcmp(ARCH, "x86_64") ? "brebus-windows-x86_64.zip"
                                    : "brebus-windows-aarch64.zip";
        break;
#elif defined(__APPLE__)
    case BREBUS_UP_MACAPP:
        a = "brebus-macos-arm64.zip";
        break;
#else
    case BREBUS_UP_APPIMAGE:
        a = !strcmp(ARCH, "x86_64") ? "brebus-x86_64.AppImage" : "";
        break;
    case BREBUS_UP_FOLDER:
        a = !strcmp(ARCH, "x86_64")    ? "brebus-linux-x86_64.tar.gz"
            : !strcmp(ARCH, "aarch64") ? "brebus-linux-aarch64.tar.gz"
                                       : "";
        break;
#endif
    default:
        break;
    }
    snprintf(out, len, "%s", a);
}

void brebus_up_program(enum brebus_up_kind kind, const char *target, char *out,
                       size_t len)
{
    switch (kind) {
    case BREBUS_UP_APPIMAGE:
        snprintf(out, len, "%s", target);
        break;
    case BREBUS_UP_MACAPP:
        snprintf(out, len, "%s/Contents/MacOS/brebus", target);
        break;
    default:
#ifdef _WIN32
        snprintf(out, len, "%s/brebus.exe", target);
#else
        snprintf(out, len, "%s/brebus", target);
#endif
    }
}

void brebus_up_where(const char *bin_dir, const char *appimage,
                     struct brebus_up_where *w)
{
    memset(w, 0, sizeof *w);
    char up[4096];
#if defined(__linux__)
    if (appimage && appimage[0] == '/' && is_file(appimage)) {
        w->kind = BREBUS_UP_APPIMAGE;
        snprintf(w->target, sizeof w->target, "%s", appimage);
        parent(appimage, up, sizeof up);
        if (!writable(up)) {
            w->kind = BREBUS_UP_NONE;
            snprintf(w->why, sizeof w->why,
                     "the AppImage's folder is not this user's to write");
        }
    } else if (!strncmp(bin_dir, "/usr/", 5) || !strncmp(bin_dir, "/opt/", 5)) {
        /* .deb, .rpm, pacman: /opt/brebus-janas */
        w->kind = BREBUS_UP_SYSTEM;
        snprintf(w->target, sizeof w->target, "%s", bin_dir);
        snprintf(w->why, sizeof w->why,
                 "installed as a package of the system: the system's "
                 "updates install the new one");
    } else {
        w->kind = BREBUS_UP_FOLDER;
        snprintf(w->target, sizeof w->target, "%s", bin_dir);
    }
#elif defined(__APPLE__)
    const char *app = strstr(bin_dir, ".app/Contents/MacOS");
    if (app && !app[strlen(".app/Contents/MacOS")]) {
        w->kind = BREBUS_UP_MACAPP;
        snprintf(w->target, sizeof w->target, "%.*s", (int)(app - bin_dir + 4),
                 bin_dir);
    } else {
        w->kind = BREBUS_UP_NONE;
        snprintf(w->why, sizeof w->why,
                 "started from Janas's folder, not from Brebus.app: "
                 "download the new version");
    }
#elif defined(_WIN32)
    char un[4200];
    snprintf(un, sizeof un, "%s/uninstall.exe", bin_dir);
    w->kind = is_file(un) ? BREBUS_UP_SETUP : BREBUS_UP_FOLDER;
    snprintf(w->target, sizeof w->target, "%s", bin_dir);
#else
    w->kind = BREBUS_UP_NONE;
    snprintf(w->why, sizeof w->why, "no updates for this system");
#endif
    (void)appimage;
    if (w->kind == BREBUS_UP_FOLDER || w->kind == BREBUS_UP_MACAPP) {
        parent(w->target, up, sizeof up);
        if (!writable(up) || !writable(w->target)) {
            w->kind = BREBUS_UP_NONE;
            snprintf(w->why, sizeof w->why,
                     "its folder is not this user's to write");
        }
    }
    if (w->kind != BREBUS_UP_NONE && w->kind != BREBUS_UP_SYSTEM) {
        brebus_up_asset(w->kind, w->asset, sizeof w->asset);
        if (!w->asset[0]) {
            w->kind = BREBUS_UP_NONE;
            snprintf(w->why, sizeof w->why,
                     "no package of Brebus for this processor yet");
        }
    }
}
