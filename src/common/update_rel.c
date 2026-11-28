/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* What Brebus's updates read of a release (update_rel.h). */
#include "common/update_rel.h"

#include "llm/json.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

int janas_rel_version(const char *s, int part[4])
{
    if (!s)
        return -1;
    if (*s == 'v')
        s++;
    int n = 0;
    part[3] = 0;
    for (;;) {
        if (!isdigit((unsigned char)*s) || n == 4)
            return -1;
        long v = 0;
        int digits = 0;
        while (isdigit((unsigned char)*s)) {
            v = v * 10 + (*s++ - '0');
            if (++digits > 6)
                return -1;
        }
        part[n++] = (int)v;
        if (*s == 0)
            break;
        if (*s++ != '.')
            return -1; /* "-3-gabc", "-dirty", anything else */
    }
    /* a year, a month, a day, and perhaps the day's number */
    if (n < 3 || part[0] < 2000 || part[1] < 1 || part[1] > 12 || part[2] < 1 ||
        part[2] > 31)
        return -1;
    return 0;
}

int janas_rel_newer(const char *a, const char *b)
{
    int pa[4], pb[4];
    if (janas_rel_version(a, pa) != 0 || janas_rel_version(b, pb) != 0)
        return 0;
    for (int i = 0; i < 4; i++)
        if (pa[i] != pb[i])
            return pa[i] > pb[i];
    return 0;
}

int janas_rel_parse(const char *json, size_t len, char *tag, size_t tag_len,
                    struct janas_rel_asset *a, int max, int *n, char *err,
                    size_t err_len)
{
    *n = 0;
    struct janas_json_doc *d = janas_json_parse(json, len, err, err_len);
    if (!d)
        return -1;
    const struct janas_json *r = janas_json_root(d);
    const struct janas_json *t = janas_json_get(r, "tag_name"),
                            *dr = janas_json_get(r, "draft"),
                            *pre = janas_json_get(r, "prerelease"),
                            *as = janas_json_get(r, "assets");
    int rc = -1;
    if (!janas_json_str(t) || !as || as->type != JANAS_JSON_ARRAY) {
        const char *msg = janas_json_str(janas_json_get(r, "message"));
        snprintf(err, err_len, "not a release%s%.200s", msg ? ": " : "",
                 msg ? msg : "");
    } else if ((dr && dr->type == JANAS_JSON_TRUE) ||
               (pre && pre->type == JANAS_JSON_TRUE))
        snprintf(err, err_len, "%.60s: a draft or a pre-release",
                 janas_json_str(t));
    else if (strlen(janas_json_str(t)) >= tag_len)
        snprintf(err, err_len, "a tag too long");
    else {
        snprintf(tag, tag_len, "%s", janas_json_str(t));
        for (const struct janas_json *e = as->child; e && *n < max;
             e = e->next) {
            const char *nm = janas_json_str(janas_json_get(e, "name")),
                       *u = janas_json_str(
                           janas_json_get(e, "browser_download_url"));
            double sz = janas_json_num(janas_json_get(e, "size"), -1);
            /* only what fits whole, over HTTPS (or from this machine:
               a test's server; the signature is checked all the same) */
            if (!nm || !u || strlen(nm) >= sizeof a->name ||
                strlen(u) >= sizeof a->url ||
                (strncmp(u, "https://", 8) &&
                 strncmp(u, "http://127.0.0.1:", 17)) ||
                sz < 0)
                continue;
            snprintf(a[*n].name, sizeof a->name, "%s", nm);
            snprintf(a[*n].url, sizeof a->url, "%s", u);
            a[*n].size = (uint64_t)sz;
            (*n)++;
        }
        rc = 0;
    }
    janas_json_free(d);
    return rc;
}

int janas_rel_sum(const char *sums, size_t len, const char *name, char sha[65])
{
    size_t nl = strlen(name);
    int found = 0;
    for (size_t i = 0; i < len;) {
        size_t e = i;
        while (e < len && sums[e] != '\n')
            e++;
        size_t l = e - i;
        const char *s = sums + i;
        if (l && s[l - 1] == '\r')
            l--;
        /* 64 hex digits, then "  name" or " *name" */
        if (l == 66 + nl && s[64] == ' ' && (s[65] == ' ' || s[65] == '*') &&
            memcmp(s + 66, name, nl) == 0) {
            int ok = 1;
            for (int k = 0; k < 64 && ok; k++)
                ok = isxdigit((unsigned char)s[k]);
            if (ok) {
                if (found++)
                    return -1; /* twice: which one? */
                for (int k = 0; k < 64; k++)
                    sha[k] = (char)tolower((unsigned char)s[k]);
                sha[64] = 0;
            }
        }
        i = e + 1;
    }
    return found == 1 ? 0 : -1;
}
