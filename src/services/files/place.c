/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * place.c - where a path leads, and whether janas-files may go there. A
 * path is followed to its end - every link, every ".." - before it is
 * judged, so that a link in the home pointing elsewhere leads nowhere: what
 * counts is the file it reaches. Allowed: the home and the folders the
 * user listed in ~/.config/janas/files-dirs.txt; or, for an app's
 * workspace (JANAS_FILES_WORKSPACE), that folder to write in and those of
 * JANAS_FILES_READ_ONLY to read only. Never, wherever they are:
 * the private files (services/common/private.c), which hold keys,
 * passwords and tokens.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common/paths.h"
#include "files.h"
#include "services/common/private.h"

int fl_expand(const struct fl_roots *r, const char *in, char *out, size_t cap)
{
    int n;
    if (in[0] == '~' && (in[1] == 0 || in[1] == '/'))
        n = snprintf(out, cap, "%s%s", r->home, in + 1);
    else if (janas_path_abs(in))
        n = snprintf(out, cap, "%s", in);
    else
        n = snprintf(out, cap, "%s/%s", r->cwd, in);
    janas_path_slashes(out);
    return n < 0 || (size_t)n >= cap ? -1 : 0;
}

#define under janas_path_under

static void add_root(struct fl_roots *r, const char *dir)
{
    char real[PATH_MAX];
    if (r->n_root < FL_ROOTS && janas_path_real(dir, real))
        snprintf(r->root[r->n_root++], PATH_MAX, "%s", real);
}

/* a workspace and its folders read only: those alone */
static int roots_workspace(struct fl_roots *r, const char *ws)
{
    char real[PATH_MAX];
    if (!janas_path_real(ws, real))
        return -1;
    r->confined = 1;
    snprintf(r->cwd, sizeof r->cwd, "%s", real);
    add_root(r, real);
    r->n_write = r->n_root;
    const char *ro = getenv("JANAS_FILES_READ_ONLY");
    char dir[PATH_MAX];
    while (ro && *ro) {
        size_t n = strcspn(ro, JANAS_PATH_LIST_SEP_S);
        if (n && n < sizeof dir) {
            memcpy(dir, ro, n);
            dir[n] = 0;
            if (janas_path_abs(dir))
                add_root(r, dir);
        }
        ro += n + (ro[n] == JANAS_PATH_LIST_SEP);
    }
    return 0;
}

int fl_roots_load(struct fl_roots *r)
{
    memset(r, 0, sizeof *r);
    const char *h = getenv("HOME");
    if (!h || !*h || !janas_path_real(h, r->home))
        return -1;
    const char *ws = getenv("JANAS_FILES_WORKSPACE");
    if (ws && *ws)
        return roots_workspace(r, ws);
    if (!getcwd(r->cwd, sizeof r->cwd))
        snprintf(r->cwd, sizeof r->cwd, "%s", r->home);
    janas_path_slashes(r->cwd);
    add_root(r, r->home);
    char list[PATH_MAX + 64];
    snprintf(list, sizeof list, "%s/.config/janas/files-dirs.txt", r->home);
    FILE *f = fopen(list, "r");
    char line[PATH_MAX];
    while (f && fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        char *s = line;
        while (*s == ' ' || *s == '\t')
            s++;
        if (!*s || *s == '#')
            continue;
        char dir[PATH_MAX];
        if (fl_expand(r, s, dir, sizeof dir) == 0)
            add_root(r, dir);
    }
    if (f)
        fclose(f);
    r->n_write = r->n_root;
    return 0;
}

int fl_private(const struct fl_roots *r, const char *path)
{
    return janas_private_path(r->home, path);
}

void fl_short(const struct fl_roots *r, const char *path, char *out, size_t cap)
{
    if (under(path, r->home) && !janas_path_root(r->home))
        snprintf(out, cap, "~%s", path + strlen(r->home));
    else
        snprintf(out, cap, "%s", path);
}

static int allowed(const struct fl_roots *r, const char *real, char *why,
                   size_t why_len)
{
    int in = 0;
    for (int i = 0; i < r->n_root && !in; i++)
        in = under(real, r->root[i]);
    if (!in && r->confined) {
        snprintf(why, why_len,
                 "%s is outside the folders these tools reach: the "
                 "workspace and the folders added to it",
                 real);
        return -1;
    }
    if (!in) {
        snprintf(why, why_len,
                 "%s is outside the folders these tools reach: the home, "
                 "and those listed in ~/.config/janas/files-dirs.txt",
                 real);
        return -1;
    }
    if (fl_private(r, real)) {
        char s[PATH_MAX];
        fl_short(r, real, s, sizeof s);
        snprintf(why, why_len,
                 "%s is private (keys, passwords, tokens, a browser's or a "
                 "shell's data): these tools do not open it",
                 s);
        return -1;
    }
    return 0;
}

int fl_resolve(const struct fl_roots *r, const char *in, int must_exist,
               char *out, size_t cap, char *why, size_t why_len)
{
    if (!in || !*in) {
        snprintf(why, why_len, "no path given");
        return -1;
    }
    char path[PATH_MAX], real[PATH_MAX];
    if (fl_expand(r, in, path, sizeof path) != 0) {
        snprintf(why, why_len, "path too long");
        return -1;
    }
    if (janas_path_real(path, real)) {
        snprintf(out, cap, "%s", real);
        return allowed(r, real, why, why_len);
    }
    if (must_exist || errno != ENOENT) {
        snprintf(why, why_len, "%s: %s", in, strerror(errno));
        return -1;
    }
    /* a file to make: its folder must be there, its name a plain one */
    char *slash = strrchr(path, '/');
    const char *name = slash + 1;
    if (!*name || !strcmp(name, ".") || !strcmp(name, "..")) {
        snprintf(why, why_len, "%s: not a file's name", in);
        return -1;
    }
    /* its folder: "/" or "C:/" when that is a root */
    char parent[PATH_MAX];
    size_t pl = (size_t)(slash - path);
    if (pl == 0 || (pl == 2 && path[1] == ':'))
        pl++;
    snprintf(parent, sizeof parent, "%.*s", (int)pl, path);
    char dir[PATH_MAX];
    if (!janas_path_real(parent, dir)) {
        snprintf(why, why_len, "%s: its folder is not there", in);
        return -1;
    }
    /* a dangling link of that name is replaced, not followed (write.c
       renames over it) */
    if ((size_t)snprintf(out, cap, "%s%s%s", janas_path_root(dir) ? "" : dir,
                         janas_path_root(dir) ? dir : "/", name) >= cap) {
        snprintf(why, why_len, "%s: path too long", in);
        return -1;
    }
    return allowed(r, out, why, why_len);
}

int fl_resolve_write(const struct fl_roots *r, const char *in, int must_exist,
                     char *out, size_t cap, char *why, size_t why_len)
{
    if (fl_resolve(r, in, must_exist, out, cap, why, why_len) != 0)
        return -1;
    for (int i = 0; i < r->n_write; i++)
        if (under(out, r->root[i]))
            return 0;
    snprintf(why, why_len,
             "%s is in a folder added to be read only: write in the "
             "workspace (%s)",
             out, r->n_write ? r->root[0] : "none");
    return -1;
}
