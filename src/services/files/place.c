/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * place.c - where a path leads, and whether janas-files may go there. A
 * path is followed to its end - every link, every ".." - before it is
 * judged, so that a link in the home pointing elsewhere leads nowhere: what
 * counts is the file it reaches. Allowed: the home and the folders the
 * user listed in ~/.config/janas/files-dirs.txt. Never, wherever they are:
 * the private files below, which hold keys, passwords and tokens.
 */
#include <errno.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "files.h"

/* Private, under the home: these folders and files and all they hold. */
static const char *const PRIVATE[] = {
    ".ssh",
    ".gnupg",
    ".password-store",
    ".pki",
    ".local/share/keyrings",
    ".config/janas", /* the services' keys */
    ".config/gh",
    ".config/hub",
    ".config/google-chrome",
    ".config/chromium",
    ".config/BraveSoftware",
    ".config/microsoft-edge",
    ".config/vivaldi",
    ".config/opera",
    ".mozilla",
    ".thunderbird",
    ".var/app/org.mozilla.firefox",
    ".var/app/com.google.Chrome",
    ".aws",
    ".azure",
    ".config/gcloud",
    ".kube",
    ".docker",
    ".netrc",
    ".git-credentials",
    ".pgpass",
    ".my.cnf",
    ".npmrc",
    ".pypirc",
    ".cargo/credentials",
    ".cargo/credentials.toml",
    ".bash_history",
    ".zsh_history",
    ".python_history",
    ".lesshst",
    ".local/share/fish/fish_history",
    ".local/share/recently-used.xbel",
};

/* Private by their name, anywhere: environment files, private keys. */
static const char *const PRIVATE_NAMES[] = {
    ".env",      ".env.*",
    "*.pem",     "*.key",
    "*.p12",     "*.pfx",
    "id_rsa*",   "id_ed25519*",
    "id_ecdsa*", "id_dsa*",
    "*.kdbx",    "*.keystore",
    "*.jks",     "credentials.json",
    "secrets.*",
};

int fl_expand(const struct fl_roots *r, const char *in, char *out, size_t cap)
{
    int n;
    if (in[0] == '~' && (in[1] == 0 || in[1] == '/'))
        n = snprintf(out, cap, "%s%s", r->home, in + 1);
    else if (in[0] == '/')
        n = snprintf(out, cap, "%s", in);
    else
        n = snprintf(out, cap, "%s/%s", r->cwd, in);
    return n < 0 || (size_t)n >= cap ? -1 : 0;
}

/* 1 when path is root or below it */
static int under(const char *path, const char *root)
{
    size_t n = strlen(root);
    if (n > 1 && root[n - 1] == '/')
        n--;
    return strncmp(path, root, n) == 0 && (path[n] == 0 || path[n] == '/');
}

static void add_root(struct fl_roots *r, const char *dir)
{
    char real[PATH_MAX];
    if (r->n_root < FL_ROOTS && realpath(dir, real))
        snprintf(r->root[r->n_root++], PATH_MAX, "%s", real);
}

int fl_roots_load(struct fl_roots *r)
{
    memset(r, 0, sizeof *r);
    const char *h = getenv("HOME");
    if (!h || !*h || !realpath(h, r->home))
        return -1;
    if (!getcwd(r->cwd, sizeof r->cwd))
        snprintf(r->cwd, sizeof r->cwd, "%s", r->home);
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
    return 0;
}

int fl_private(const struct fl_roots *r, const char *path)
{
    if (under(path, r->home) && strlen(path) > strlen(r->home)) {
        const char *rel = path + strlen(r->home) + 1;
        for (size_t i = 0; i < sizeof PRIVATE / sizeof PRIVATE[0]; i++) {
            size_t n = strlen(PRIVATE[i]);
            if (!strncmp(rel, PRIVATE[i], n) && (!rel[n] || rel[n] == '/'))
                return 1;
        }
    }
    /* any part of the path named as a private file */
    char copy[PATH_MAX];
    snprintf(copy, sizeof copy, "%s", path);
    for (char *save = NULL, *part = strtok_r(copy, "/", &save); part;
         part = strtok_r(NULL, "/", &save))
        for (size_t i = 0; i < sizeof PRIVATE_NAMES / sizeof PRIVATE_NAMES[0];
             i++)
            if (!fnmatch(PRIVATE_NAMES[i], part, 0))
                return 1;
    return 0;
}

void fl_short(const struct fl_roots *r, const char *path, char *out, size_t cap)
{
    if (under(path, r->home))
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
    if (realpath(path, real)) {
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
    *slash = 0;
    char dir[PATH_MAX];
    if (!realpath(*path ? path : "/", dir)) {
        snprintf(why, why_len, "%s: its folder is not there", in);
        return -1;
    }
    /* a dangling link of that name is replaced, not followed (write.c
       renames over it) */
    if ((size_t)snprintf(out, cap, "%s/%s", strcmp(dir, "/") ? dir : "",
                         name) >= cap) {
        snprintf(why, why_len, "%s: path too long", in);
        return -1;
    }
    return allowed(r, out, why, why_len);
}
