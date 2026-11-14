/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * private.c - see private.h. The names are matched with '*' alone, the
 * only wildcard they use, so that the same rules hold where the system
 * has no fnmatch.
 */
#include "private.h"

#include <stdio.h>
#include <string.h>

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

int janas_path_under(const char *path, const char *root)
{
    size_t n = strlen(root);
    if (n > 1 && root[n - 1] == '/')
        n--;
    return strncmp(path, root, n) == 0 && (path[n] == 0 || path[n] == '/');
}

/* s matches pattern p, whose '*' stand for any run of characters */
static int match(const char *p, const char *s, size_t n)
{
    while (*p && *p != '*') {
        if (!n || *p != *s)
            return 0;
        p++, s++, n--;
    }
    if (!*p)
        return n == 0;
    while (*p == '*')
        p++;
    for (size_t k = 0; k <= n; k++)
        if (match(p, s + k, n - k))
            return 1;
    return 0;
}

int janas_private_path(const char *home, const char *path)
{
    size_t hn = strlen(home);
    if (janas_path_under(path, home) && strlen(path) > hn) {
        const char *rel = path + hn + (path[hn] == '/');
        for (size_t i = 0; i < sizeof PRIVATE / sizeof PRIVATE[0]; i++) {
            size_t n = strlen(PRIVATE[i]);
            if (!strncmp(rel, PRIVATE[i], n) && (!rel[n] || rel[n] == '/'))
                return 1;
        }
    }
    /* any part of the path named as a private file */
    for (const char *s = path; *s;) {
        while (*s == '/')
            s++;
        size_t n = strcspn(s, "/");
        for (size_t i = 0;
             n && i < sizeof PRIVATE_NAMES / sizeof PRIVATE_NAMES[0]; i++)
            if (match(PRIVATE_NAMES[i], s, n))
                return 1;
        s += n;
    }
    return 0;
}
