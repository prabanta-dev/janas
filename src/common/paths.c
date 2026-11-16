/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * paths.c - see paths.h: paths as Janas keeps them, on every system:
 * absolute, with '/' between their parts. On Windows that is "C:/Users/x/work",
 * which its calls take as they take "C:\Users\x\work": the paths the system
 * gives (realpath, the environment's folders) have their '\' turned to '/', so
 * that a check of "inside the workspace" is the same comparison
 * everywhere.
 */
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "paths.h"

#ifdef _WIN32
#include <windows.h>
#include <knownfolders.h>
#include <shlobj.h>
#endif

void janas_path_slashes(char *p)
{
#ifdef _WIN32
    for (; p && *p; p++)
        if (*p == '\\')
            *p = '/';
#else
    (void)p;
#endif
}

int janas_path_abs(const char *p)
{
    if (!p)
        return 0;
#ifdef _WIN32
    /* a drive's ("C:/", "C:\"), or a share's ("//server/x", "\\server\x") */
    if (((p[0] >= 'A' && p[0] <= 'Z') || (p[0] >= 'a' && p[0] <= 'z')) &&
        p[1] == ':' && (p[2] == '/' || p[2] == '\\'))
        return 1;
    return (p[0] == '/' || p[0] == '\\') && (p[1] == '/' || p[1] == '\\');
#else
    return p[0] == '/';
#endif
}

int janas_path_root(const char *p)
{
#ifdef _WIN32
    /* a drive's root, "C:/" */
    return p && p[0] && p[1] == ':' && p[2] == '/' && !p[3];
#else
    return p && !strcmp(p, "/");
#endif
}

char *janas_path_real(const char *p, char *out)
{
    if (!janas_path_abs(p) || !realpath(p, out))
        return NULL;
    janas_path_slashes(out);
#ifdef _WIN32
    /* "C:/x/" from a folder's path written with its last '/': as "C:/x" */
    size_t n = strlen(out);
    if (n > 3 && out[n - 1] == '/')
        out[n - 1] = 0;
#endif
    return out;
}

int janas_path_in(const char *path, const char *dir)
{
    size_t n = strlen(dir);
    if (n && dir[n - 1] == '/') /* a root: "/" or "C:/" */
        return !strncmp(path, dir, n);
#ifdef _WIN32
    /* the same folder whatever the case of its letters, as Windows has it */
    return !strncasecmp(path, dir, n) && path[n] == '/';
#else
    return !strncmp(path, dir, n) && path[n] == '/';
#endif
}

int janas_known_dir(int downloads, char *out, size_t n)
{
#ifdef _WIN32
    /* the user's Documents or Downloads, wherever they were moved */
    PWSTR w = NULL;
    int rc = -1;
    if (SHGetKnownFolderPath(downloads ? &FOLDERID_Downloads
                                       : &FOLDERID_Documents,
                             0, NULL, &w) == S_OK &&
        WideCharToMultiByte(CP_UTF8, 0, w, -1, out, (int)n, NULL, NULL) > 0) {
        janas_path_slashes(out);
        rc = 0;
    }
    CoTaskMemFree(w);
    return rc;
#else
    (void)downloads, (void)out, (void)n;
    return -1;
#endif
}
