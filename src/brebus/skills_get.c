/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * skills_get.c - skills taken from a repository on GitHub, as the user
 * names it (https://github.com/owner/repo, a folder of it, owner/repo) or
 * as Janas's list of skills names it: the repository's tree read through
 * GitHub's API (two requests: the repository, its tree), the folders that
 * hold a SKILL.md listed with their name and description, and the one
 * chosen downloaded file by file from raw.githubusercontent.com into a
 * folder of the data folder's, then installed among every chat's skills as
 * a folder of the user's is (skills.c). Bounded: 400 files, 50 MB, a file
 * 20 MB; a path that climbs out (..) or is absolute refused.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/hf_fetch.h"
#include "common/mcp_skills.h"

#define MAX_FILES 400
#define MAX_BYTES (50u << 20)
#define MAX_FILE (20u << 20)
#define MAX_SCAN 40

struct gh {
    char owner[100], repo[100], ref[200], path[1024];
};

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

/* a name GitHub allows for an owner, a repository or a branch's part */
static int gh_word(const char *s, size_t n)
{
    if (!n)
        return 0;
    for (size_t i = 0; i < n; i++)
        if (!((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z') ||
              (s[i] >= '0' && s[i] <= '9') || s[i] == '-' || s[i] == '_' ||
              s[i] == '.'))
            return 0;
    return !(n == 2 && s[0] == '.' && s[1] == '.') && !(n == 1 && *s == '.');
}

/* a path inside a repository: no "..", not absolute, no '\' or ':' */
static int inside(const char *p)
{
    if (*p == '/' || strchr(p, '\\') || strchr(p, ':'))
        return 0;
    for (const char *s = p; *s;) {
        const char *e = strchr(s, '/');
        size_t n = e ? (size_t)(e - s) : strlen(s);
        if ((n == 2 && s[0] == '.' && s[1] == '.') || n == 0)
            return 0;
        s += n + (e ? 1 : 0);
    }
    return 1;
}

/* https://github.com/o/r[/tree|blob/ref[/path]] or o/r into g: 0, or -1 */
static int gh_parse(const char *u, struct gh *g)
{
    memset(g, 0, sizeof *g);
    if (!u)
        return -1;
    while (*u == ' ')
        u++;
    if (!strncmp(u, "https://", 8))
        u += 8;
    else if (!strncmp(u, "http://", 7))
        u += 7;
    if (!strncmp(u, "www.", 4))
        u += 4;
    if (!strncmp(u, "github.com/", 11))
        u += 11;
    const char *parts[5] = {0};
    size_t lens[5] = {0};
    int k = 0;
    const char *s = u;
    while (*s && k < 4) {
        const char *e = strchr(s, '/');
        size_t n = e ? (size_t)(e - s) : strlen(s);
        parts[k] = s;
        lens[k++] = n;
        s += n + (e ? 1 : 0);
    }
    if (k < 2 || !gh_word(parts[0], lens[0]))
        return -1;
    size_t rn = lens[1];
    if (rn > 4 && !memcmp(parts[1] + rn - 4, ".git", 4))
        rn -= 4;
    if (!gh_word(parts[1], rn) || lens[0] >= sizeof g->owner ||
        rn >= sizeof g->repo)
        return -1;
    memcpy(g->owner, parts[0], lens[0]);
    memcpy(g->repo, parts[1], rn);
    if (k >= 4 && (lens[2] == 4 && (!memcmp(parts[2], "tree", 4) ||
                                    !memcmp(parts[2], "blob", 4)))) {
        if (!gh_word(parts[3], lens[3]) || lens[3] >= sizeof g->ref)
            return -1;
        memcpy(g->ref, parts[3], lens[3]);
        snprintf(g->path, sizeof g->path, "%s", s);
        size_t pn = strlen(g->path);
        while (pn && (g->path[pn - 1] == '/' || g->path[pn - 1] == ' '))
            g->path[--pn] = 0;
        /* a file named: its folder */
        if (parts[2][0] == 'b') {
            char *sl = strrchr(g->path, '/');
            if (sl)
                *sl = 0;
            else
                g->path[0] = 0;
        }
        if (*g->path && !inside(g->path))
            return -1;
    }
    return 0;
}

/* a path for a URL: each byte but the unreserved and '/' escaped */
static void encode_path(const char *p, char *out, size_t n)
{
    static const char HEX[] = "0123456789ABCDEF";
    size_t k = 0;
    for (const unsigned char *c = (const unsigned char *)p; *c && k + 4 < n;
         c++)
        if ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
            (*c >= '0' && *c <= '9') || strchr("-_.~/", *c))
            out[k++] = (char)*c;
        else {
            out[k++] = '%';
            out[k++] = HEX[*c >> 4];
            out[k++] = HEX[*c & 15];
        }
    out[k] = 0;
}

/* the repository (its branch when g has none, its licence) and its tree:
   the tree's document, or NULL with why */
static struct janas_json_doc *gh_tree(struct gh *g, char *license, size_t ln,
                                      char *why, size_t wn)
{
    char url[1400];
    struct janas_buf t = {0};
    snprintf(url, sizeof url, "https://api.github.com/repos/%s/%s", g->owner,
             g->repo);
    if (janas_hf_get_url(url, NULL, NULL, &t, NULL, 0, why, wn) != 0)
        return NULL;
    struct janas_json_doc *d = janas_json_parse(t.p, t.n, why, wn);
    janas_buf_free(&t);
    const struct janas_json *r = d ? janas_json_root(d) : NULL;
    const char *br =
        r ? janas_json_str(janas_json_get(r, "default_branch")) : NULL;
    const struct janas_json *lic = r ? janas_json_get(r, "license") : NULL;
    const char *spdx =
        lic ? janas_json_str(janas_json_get(lic, "spdx_id")) : NULL;
    snprintf(license, ln, "%s",
             spdx && strcmp(spdx, "NOASSERTION") ? spdx : "");
    if (!*g->ref && br && gh_word(br, strlen(br)))
        snprintf(g->ref, sizeof g->ref, "%s", br);
    janas_json_free(d);
    if (!*g->ref) {
        snprintf(why, wn, "%s/%s: no such repository", g->owner, g->repo);
        return NULL;
    }
    snprintf(url, sizeof url,
             "https://api.github.com/repos/%s/%s/git/trees/%s?recursive=1",
             g->owner, g->repo, g->ref);
    if (janas_hf_get_url(url, NULL, NULL, &t, NULL, 0, why, wn) != 0)
        return NULL;
    d = t.n < (32u << 20) ? janas_json_parse(t.p, t.n, why, wn) : NULL;
    janas_buf_free(&t);
    if (d && !janas_json_get(janas_json_root(d), "tree")) {
        janas_json_free(d);
        snprintf(why, wn, "GitHub gave no tree for %s", g->ref);
        return NULL;
    }
    return d;
}

/* a file of the repository into path: 0, or -1 */
static int gh_file(const struct gh *g, const char *file, const char *path,
                   char *why, size_t wn)
{
    char ep[3200], url[4000];
    encode_path(file, ep, sizeof ep);
    snprintf(url, sizeof url, "https://raw.githubusercontent.com/%s/%s/%s/%s",
             g->owner, g->repo, g->ref, ep);
    struct janas_buf t = {0};
    int rc = janas_hf_get_url(url, NULL, NULL, &t, NULL, 0, why, wn);
    if (rc == 0) {
        FILE *f = fopen(path, "wb");
        rc = f ? 0 : -1;
        if (f) {
            if (t.n && fwrite(t.p, 1, t.n, f) != t.n)
                rc = -1;
            if (fclose(f) != 0)
                rc = -1;
        }
        if (rc)
            snprintf(why, wn, "%s could not be written", file);
    }
    janas_buf_free(&t);
    return rc;
}

/* a folder of the data folder's for one download, made empty */
static int staging(struct brebus *b, char *dir, size_t n)
{
    snprintf(dir, n, "%s/skill-get/%ld-%ld", b->data_dir, (long)getpid(),
             (long)time(NULL));
    brebus_remove_tree(dir, 0);
    return janas_mkdirs(dir);
}

/* d's folder is under p (or p is the whole repository) */
static int under(const char *file, const char *p)
{
    size_t n = strlen(p);
    return !n || (!strncmp(file, p, n) && file[n] == '/');
}

int brebus_skills_scan(struct brebus *b, const char *body, size_t n,
                       struct janas_buf *out)
{
    char why[300] = "", license[64] = "", tmp[4300];
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, why, sizeof why) : NULL;
    const char *u =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "url")) : NULL;
    struct gh g;
    int ok = gh_parse(u, &g) == 0;
    janas_json_free(d);
    if (!ok) {
        janas_buf_puts(out, "{\"error\": \"not a repository of GitHub: "
                            "https://github.com/owner/repository\"}");
        return -1;
    }
    struct janas_json_doc *t =
        gh_tree(&g, license, sizeof license, why, sizeof why);
    if (!t || staging(b, tmp, sizeof tmp) != 0) {
        janas_json_free(t);
        janas_buf_puts(out, "{\"error\": ");
        str(out, why[0] ? why : "no folder to download into");
        janas_buf_puts(out, "}");
        return -1;
    }
    const struct janas_json *tree = janas_json_get(janas_json_root(t), "tree");
    janas_buf_puts(out, "{\"repo\": ");
    janas_buf_printf(out, "\"%s/%s\", \"ref\": ", g.owner, g.repo);
    str(out, g.ref);
    janas_buf_puts(out, ", \"license\": ");
    str(out, license);
    janas_buf_puts(out, ", \"skills\": [");
    int k = 0;
    for (const struct janas_json *e = tree->child; e && k < MAX_SCAN;
         e = e->next) {
        const char *p = janas_json_str(janas_json_get(e, "path"));
        size_t pn = p ? strlen(p) : 0;
        if (!p || pn < 8 || strcmp(p + pn - 8, "SKILL.md") ||
            (pn > 8 && p[pn - 9] != '/') || !inside(p))
            continue;
        char dir[1024];
        snprintf(dir, sizeof dir, "%.*s", (int)(pn > 8 ? pn - 9 : 0), p);
        if (*g.path && strcmp(dir, g.path) && !under(dir, g.path))
            continue;
        /* its files, their size, a licence of its own */
        long files = 0, bytes = 0;
        int lic = 0;
        for (const struct janas_json *f = tree->child; f; f = f->next) {
            const char *fp = janas_json_str(janas_json_get(f, "path"));
            const char *ty = janas_json_str(janas_json_get(f, "type"));
            if (!fp || !ty || strcmp(ty, "blob") || !under(fp, dir))
                continue;
            const struct janas_json *sz = janas_json_get(f, "size");
            files++;
            bytes += (long)janas_json_num(sz, 0);
            const char *base = fp + strlen(dir) + (*dir ? 1 : 0);
            lic |= !strncmp(base, "LICENSE", 7) || !strncmp(base, "LICENCE", 7);
        }
        /* its name and description, as Brebus reads them once installed */
        char sd[4400], sp[4500], name[128], about[1024];
        snprintf(sd, sizeof sd, "%s/%d", tmp, k);
        snprintf(sp, sizeof sp, "%s/SKILL.md", sd);
        const char *folder = strrchr(dir, '/');
        folder = folder ? folder + 1 : *dir ? dir : g.repo;
        if (janas_mkdirs(sd) != 0 || gh_file(&g, p, sp, why, sizeof why) ||
            janas_skill_info(sd, folder, name, sizeof name, about,
                             sizeof about) != 0)
            continue;
        janas_buf_puts(out, k++ ? ", {\"path\": " : "{\"path\": ");
        str(out, dir);
        janas_buf_puts(out, ", \"name\": ");
        str(out, name);
        janas_buf_puts(out, ", \"description\": ");
        str(out, about);
        janas_buf_printf(out,
                         ", \"files\": %ld, \"bytes\": %ld, \"license_file\": "
                         "%s}",
                         files, bytes, lic ? "true" : "false");
    }
    janas_buf_puts(out, "]}");
    janas_json_free(t);
    char top[4300];
    snprintf(top, sizeof top, "%s/skill-get", b->data_dir);
    brebus_remove_tree(top, 0);
    return 0;
}

int brebus_skills_fetch(struct brebus *b, const char *body, size_t n,
                        struct janas_buf *out)
{
    char why[300] = "", license[64], tmp[4300], top[4300], url[1400];
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, why, sizeof why) : NULL;
    const struct janas_json *r = d ? janas_json_root(d) : NULL;
    const char *repo = r ? janas_json_str(janas_json_get(r, "repo")) : NULL;
    const char *ref = r ? janas_json_str(janas_json_get(r, "ref")) : NULL;
    const char *path = r ? janas_json_str(janas_json_get(r, "path")) : NULL;
    struct gh g;
    snprintf(url, sizeof url, "%s%s%s%s%s", repo ? repo : "",
             ref && *ref ? "/tree/" : "", ref && *ref ? ref : "",
             ref && *ref && path && *path ? "/" : "",
             ref && *ref && path ? path : "");
    int ok =
        repo && gh_parse(url, &g) == 0 && (!path || !*path || inside(path));
    if (ok && path && *path && !*g.path)
        snprintf(g.path, sizeof g.path, "%s", path);
    janas_json_free(d);
    if (!ok) {
        janas_buf_puts(out, "{\"error\": \"not a skill of GitHub\"}");
        return -1;
    }
    struct janas_json_doc *t =
        gh_tree(&g, license, sizeof license, why, sizeof why);
    int rc = -1;
    char name[128] = "";
    snprintf(top, sizeof top, "%s/skill-get", b->data_dir);
    if (t && staging(b, tmp, sizeof tmp) == 0) {
        const struct janas_json *tree =
            janas_json_get(janas_json_root(t), "tree");
        const char *folder = strrchr(g.path, '/');
        folder = folder ? folder + 1 : *g.path ? g.path : g.repo;
        char dst[5400];
        snprintf(dst, sizeof dst, "%s/%s", tmp, folder);
        long files = 0, bytes = 0;
        rc = 0;
        /* counted first: too many or too large, nothing downloaded */
        for (const struct janas_json *e = tree->child; e; e = e->next) {
            const char *fp = janas_json_str(janas_json_get(e, "path"));
            const char *ty = janas_json_str(janas_json_get(e, "type"));
            const struct janas_json *sz = janas_json_get(e, "size");
            if (!fp || !ty || strcmp(ty, "blob") || !under(fp, g.path))
                continue;
            long s = (long)janas_json_num(sz, 0);
            files++;
            bytes += s;
            if (!inside(fp) || s > (long)MAX_FILE)
                rc = -1;
        }
        if (rc || files > MAX_FILES || bytes > (long)MAX_BYTES || !files) {
            snprintf(why, sizeof why,
                     !files ? "no files there"
                            : "too large for a skill: %ld files, %ld MB",
                     files, bytes >> 20);
            rc = -1;
        }
        for (const struct janas_json *e = tree->child; e && rc == 0;
             e = e->next) {
            const char *fp = janas_json_str(janas_json_get(e, "path"));
            const char *ty = janas_json_str(janas_json_get(e, "type"));
            if (!fp || !ty || strcmp(ty, "blob") || !under(fp, g.path))
                continue;
            const char *rel = fp + strlen(g.path) + (*g.path ? 1 : 0);
            char to[6000], dir[6000];
            snprintf(to, sizeof to, "%s/%s", dst, rel);
            snprintf(dir, sizeof dir, "%s", to);
            char *sl = strrchr(dir, '/');
            if (sl)
                *sl = 0;
            if (janas_mkdirs(dir) != 0 ||
                gh_file(&g, fp, to, why, sizeof why) != 0)
                rc = -1;
        }
        if (rc == 0)
            rc = brebus_skills_install(dst, folder, name, sizeof name, why,
                                       sizeof why);
    } else if (t)
        snprintf(why, sizeof why, "no folder to download into");
    janas_json_free(t);
    brebus_remove_tree(top, 0);
    janas_buf_puts(out, rc ? "{\"error\": " : "{\"name\": ");
    str(out, rc ? (why[0] ? why : "not installed") : name);
    if (rc == 0) {
        janas_buf_puts(out, ", \"license\": ");
        str(out, license);
    }
    janas_buf_puts(out, "}");
    return rc;
}
