/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * roots.c - the folders janas-docs indexes: those listed, a path a line, in
 * ~/.config/janas/docs-dirs.txt, or else the system's Documents folder
 * (XDG_DOCUMENTS_DIR in ~/.config/user-dirs.dirs, ~/Documents without
 * it). A path is followed to its end before it is judged, and the private
 * files (services/common/private.c) are never read, wherever they are.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "docs.h"
#include "services/common/private.h"

static int expand(const struct dz_roots *r, const char *in, char *out,
                  size_t cap)
{
    int n;
    if (in[0] == '~' && (in[1] == 0 || in[1] == '/'))
        n = snprintf(out, cap, "%s%s", r->home, in + 1);
    else if (!strncmp(in, "$HOME", 5) && (in[5] == 0 || in[5] == '/'))
        n = snprintf(out, cap, "%s%s", r->home, in + 5);
    else
        n = snprintf(out, cap, "%s", in);
    return n < 0 || (size_t)n >= cap || out[0] != '/' ? -1 : 0;
}

static void add_root(struct dz_roots *r, const char *dir)
{
    char real[PATH_MAX];
    struct stat st;
    if (r->n_root < DZ_ROOTS && realpath(dir, real) && !stat(real, &st) &&
        S_ISDIR(st.st_mode) && !janas_private_path(r->home, real))
        snprintf(r->root[r->n_root++], PATH_MAX, "%s", real);
}

/* XDG_DOCUMENTS_DIR="$HOME/Documenti" of user-dirs.dirs, into out */
static int xdg_documents(const struct dz_roots *r, char *out, size_t cap)
{
    char p[PATH_MAX + 64], line[PATH_MAX];
    const char *cfg = getenv("XDG_CONFIG_HOME");
    if (cfg && *cfg == '/')
        snprintf(p, sizeof p, "%s/user-dirs.dirs", cfg);
    else
        snprintf(p, sizeof p, "%s/.config/user-dirs.dirs", r->home);
    FILE *f = fopen(p, "r");
    int found = 0;
    while (f && !found && fgets(line, sizeof line, f)) {
        if (strncmp(line, "XDG_DOCUMENTS_DIR=", 18))
            continue;
        char *v = line + 18;
        if (*v == '"')
            v++;
        v[strcspn(v, "\"\r\n")] = 0;
        found = expand(r, v, out, cap) == 0;
    }
    if (f)
        fclose(f);
    return found;
}

int dz_roots_load(struct dz_roots *r)
{
    memset(r, 0, sizeof *r);
    const char *h = getenv("HOME");
    if (!h || !*h || !realpath(h, r->home))
        return -1;
    char list[PATH_MAX + 64];
    snprintf(list, sizeof list, "%s/.config/janas/docs-dirs.txt", r->home);
    FILE *f = fopen(list, "r");
    char line[PATH_MAX];
    while (f && fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        char *s = line, dir[PATH_MAX];
        while (*s == ' ' || *s == '\t')
            s++;
        size_t n = strlen(s);
        while (n && (s[n - 1] == ' ' || s[n - 1] == '\t'))
            s[--n] = 0;
        if (!*s || *s == '#')
            continue;
        r->listed = 1;
        if (expand(r, s, dir, sizeof dir) == 0)
            add_root(r, dir);
    }
    if (f)
        fclose(f);
    if (!r->listed) {
        char dir[PATH_MAX + 16];
        if (xdg_documents(r, dir, sizeof dir) && strcmp(dir, r->home))
            add_root(r, dir);
        if (!r->n_root) {
            snprintf(dir, sizeof dir, "%s/Documents", r->home);
            add_root(r, dir);
        }
    }
    return 0;
}

void dz_short(const struct dz_roots *r, const char *path, char *out, size_t cap)
{
    if (janas_path_under(path, r->home) && strcmp(r->home, "/"))
        snprintf(out, cap, "~%s", path + strlen(r->home));
    else
        snprintf(out, cap, "%s", path);
}

int dz_resolve(const struct dz_roots *r, const char *in, char *out, size_t cap,
               char *why, size_t wn)
{
    char path[PATH_MAX], real[PATH_MAX];
    if (!in || !*in) {
        snprintf(why, wn, "no path given");
        return -1;
    }
    if (expand(r, in, path, sizeof path) != 0) {
        snprintf(why, wn,
                 "%s: give the path from the search, ~/... or absolute", in);
        return -1;
    }
    if (!realpath(path, real)) {
        snprintf(why, wn, "%s: %s", in, strerror(errno));
        return -1;
    }
    int inside = 0;
    for (int i = 0; i < r->n_root && !inside; i++)
        inside = janas_path_under(real, r->root[i]);
    char s[PATH_MAX];
    dz_short(r, real, s, sizeof s);
    if (!inside) {
        snprintf(why, wn,
                 "%s is outside the folders of documents (those listed in "
                 "~/.config/janas/docs-dirs.txt)",
                 s);
        return -1;
    }
    if (janas_private_path(r->home, real)) {
        snprintf(why, wn, "%s is private: not read", s);
        return -1;
    }
    if (dz_kind(real) == DZ_NONE) {
        snprintf(
            why, wn,
            "%s is not a document read here (text, Markdown, docx, odt, PDF)",
            s);
        return -1;
    }
    snprintf(out, cap, "%s", real);
    return 0;
}
