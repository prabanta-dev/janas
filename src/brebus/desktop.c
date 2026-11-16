/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * desktop.c - Brebus in the desktop's menu: the Prabanta logo as its icon,
 * a PNG of each size in ~/.local/share/icons/hicolor/<size>/apps and the SVG
 * in hicolor/scalable/apps, and brebus.desktop in
 * ~/.local/share/applications, starting this brebus by its full path, with
 * the JANAS_MODELS given when it was written. The desktop matches the
 * window to it by the name window.c gives GTK ("brebus"): on Wayland the
 * icon in the task bar comes from there. Written by --install-desktop, and
 * each time Brebus opens its window: the icons where one is missing or
 * differs, the entry where there is none.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/sysinfo.h"

/* path written with p, unless it holds those bytes already: 0, or -1 */
static int write_file(const char *path, const void *p, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (f) {
        char buf[65536];
        size_t got = fread(buf, 1, sizeof buf, f);
        int same =
            got == n && n < sizeof buf && fgetc(f) == EOF && !memcmp(buf, p, n);
        fclose(f);
        if (same)
            return 0;
    }
    char tmp[4400];
    snprintf(tmp, sizeof tmp, "%s.new", path);
    f = fopen(tmp, "wb");
    if (!f)
        return -1;
    int bad = fwrite(p, 1, n, f) != n;
    bad |= fclose(f) != 0;
    if (bad || rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

static int icon(const char *share, const char *size, const char *name,
                const void *p, size_t n)
{
    char dir[4300], path[4400];
    snprintf(dir, sizeof dir, "%s/icons/hicolor/%s/apps", share, size);
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (janas_mkdirs(dir) != 0 || write_file(path, p, n) != 0) {
        fprintf(stderr, "brebus: cannot write %s\n", path);
        return -1;
    }
    return 0;
}

#ifdef _WIN32
/* Windows: the Start menu's entry is the installer's (or the zip's user
   makes one); the icon is in brebus.exe (brebus.rc); Brebus at login is
   the user's Run key, its server without a window */
#include <windows.h>

#define RUN_KEY "Software\\Microsoft\\Windows\\CurrentVersion\\Run"

int brebus_install_desktop(int quiet)
{
    (void)write_file, (void)icon;
    if (!quiet)
        printf("brebus: on Windows the Start menu's entry comes with the "
               "installer\n");
    return 0;
}

int brebus_autostart_on(void)
{
    HKEY k;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_READ, &k) != 0)
        return 0;
    int on = RegQueryValueExA(k, "Brebus", NULL, NULL, NULL, NULL) == 0;
    RegCloseKey(k);
    return on;
}

int brebus_autostart(int on)
{
    HKEY k;
    if (RegCreateKeyExA(HKEY_CURRENT_USER, RUN_KEY, 0, NULL, 0, KEY_SET_VALUE,
                        NULL, &k, NULL) != 0)
        return -1;
    int rc;
    if (!on) {
        LONG r = RegDeleteValueA(k, "Brebus");
        rc = r == 0 || r == ERROR_FILE_NOT_FOUND ? 0 : -1;
    } else {
        char exe[4096], cmd[4200];
        rc = -1;
        if (janas_exe_path(exe, sizeof exe) > 0) {
            for (char *c = exe; *c; c++)
                if (*c == '/')
                    *c = '\\';
            int n = snprintf(cmd, sizeof cmd, "\"%s\" --no-open", exe);
            rc = n > 0 && (size_t)n < sizeof cmd &&
                         RegSetValueExA(k, "Brebus", 0, REG_SZ,
                                        (const BYTE *)cmd, (DWORD)n + 1) == 0
                     ? 0
                     : -1;
        }
    }
    RegCloseKey(k);
    return rc;
}
#else
int brebus_install_desktop(int quiet)
{
    const char *home = getenv("HOME"), *xd = getenv("XDG_DATA_HOME");
    char share[4096], dir[4200], entry[4300], exe[4096], size[16];
    if (xd && *xd == '/')
        snprintf(share, sizeof share, "%s", xd);
    else if (home)
        snprintf(share, sizeof share, "%s/.local/share", home);
    else {
        fprintf(stderr, "brebus: no home\n");
        return 1;
    }
    if (janas_exe_path(exe, sizeof exe) <= 0) {
        fprintf(stderr, "brebus: where is brebus?\n");
        return 1;
    }
    for (const struct brebus_icon *i = brebus_icons; i->size; i++) {
        snprintf(size, sizeof size, "%dx%d", i->size, i->size);
        if (icon(share, size, "brebus.png", i->png, i->len) != 0)
            return 1;
    }
    if (icon(share, "scalable", "brebus.svg", brebus_icon_svg,
             brebus_icon_svg_len) != 0)
        return 1;
    snprintf(dir, sizeof dir, "%s/applications", share);
    snprintf(entry, sizeof entry, "%s/brebus.desktop", dir);
    /* opening the window, an entry there already is kept as it is: its
       JANAS_MODELS is the one --install-desktop was given */
    FILE *had = quiet ? fopen(entry, "r") : NULL;
    if (had) {
        fclose(had);
        return 0;
    }
    /* the models' folder given now, kept for the menu's start, which has
       no shell to set it */
    const char *jm = getenv("JANAS_MODELS");
    char env[4200] = "";
    if (jm && *jm == '/' && !strpbrk(jm, "\"\\`$ \n"))
        snprintf(env, sizeof env, "env JANAS_MODELS=%s ", jm);
    char text[8192];
    int n = snprintf(text, sizeof text,
                     "[Desktop Entry]\n"
                     "Type=Application\n"
                     "Name=Brebus\n"
                     "Comment=Janas's app for the models of this computer\n"
                     "Exec=%s\"%s\"\n"
                     "Icon=brebus\n"
                     "Terminal=false\n"
                     "Categories=Utility;\n"
                     "StartupWMClass=brebus\n",
                     env, exe);
    if (n < 0 || (size_t)n >= sizeof text || janas_mkdirs(dir) != 0 ||
        write_file(entry, text, (size_t)n) != 0) {
        fprintf(stderr, "brebus: cannot write %s\n", entry);
        return 1;
    }
    if (!quiet)
        printf("Brebus is in the desktop's menu: %s, its icons in "
               "%s/icons/hicolor\n",
               entry, share);
    return 0;
}

static void autostart_path(char *p, size_t n)
{
    const char *xc = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
    if (xc && *xc == '/')
        snprintf(p, n, "%s/autostart/brebus.desktop", xc);
    else
        snprintf(p, n, "%s/.config/autostart/brebus.desktop",
                 home ? home : ".");
}

int brebus_autostart_on(void)
{
    char p[4200];
    autostart_path(p, sizeof p);
    return access(p, F_OK) == 0;
}

int brebus_autostart(int on)
{
    char p[4200], dir[4200], exe[4096];
    autostart_path(p, sizeof p);
    if (!on)
        return remove(p) == 0 || errno == ENOENT ? 0 : -1;
    if (janas_exe_path(exe, sizeof exe) <= 0)
        return -1;
    snprintf(dir, sizeof dir, "%s", p);
    *strrchr(dir, '/') = 0;
    /* its server only: the window opens from the menu, onto it */
    const char *jm = getenv("JANAS_MODELS");
    char env[4200] = "";
    if (jm && *jm == '/' && !strpbrk(jm, "\"\\`$ \n"))
        snprintf(env, sizeof env, "env JANAS_MODELS=%s ", jm);
    char text[8800];
    int n = snprintf(text, sizeof text,
                     "[Desktop Entry]\n"
                     "Type=Application\n"
                     "Name=Brebus\n"
                     "Comment=Janas's app, its server started at login\n"
                     "Exec=%s\"%s\" --no-open\n"
                     "Icon=brebus\n"
                     "Terminal=false\n"
                     "X-GNOME-Autostart-enabled=true\n",
                     env, exe);
    return n > 0 && (size_t)n < sizeof text && janas_mkdirs(dir) == 0 &&
                   write_file(p, text, (size_t)n) == 0
               ? 0
               : -1;
}
#endif
