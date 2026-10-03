/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * conf.c - a service's settings file (see conf.h).
 */
#include "services/common/conf.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "common/words.h"

int janas_conf_get(const char *path, const char *const *names, size_t n,
                   const char *name, char *val, size_t cap, char *why,
                   size_t why_len)
{
    if (why_len)
        why[0] = 0;
    if (cap)
        val[0] = 0;
    FILE *f = path ? fopen(path, "r") : NULL;
    if (!f)
        return 0;
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    char line[512];
    int found = 0, at = 0;
    while (fgets(line, sizeof line, f)) {
        at++;
        char *a = line + strspn(line, " \t");
        size_t len = strlen(a);
        while (len && isspace((unsigned char)a[len - 1]))
            a[--len] = 0;
        if (!*a || *a == '#')
            continue;
        size_t k = strcspn(a, " \t=");
        char *v = a + k;
        v += strspn(v, " \t");
        if (*v == '=')
            v += 1 + strspn(v + 1, " \t");
        a[k] = 0;
        if (strcmp(a, name) == 0) {
            snprintf(val, cap, "%s", v);
            found = 1;
            continue;
        }
        int known = 0;
        for (size_t i = 0; i < n && !known; i++)
            known = strcmp(a, names[i]) == 0;
        if (!known && why_len && !why[0]) {
            char w[200];
            janas_unknown_word(w, sizeof w, a, "a setting", names, n);
            snprintf(why, why_len, "%s, line %d: %s", base, at, w);
        }
    }
    memset(line, 0, sizeof line); /* a key read is not left about */
    fclose(f);
    return found;
}
