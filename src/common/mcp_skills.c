/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_skills.c - skills as Brebus and janas-skills find them
 * (mcp_skills.h), in libjanas_mcp beside the configuration they live by.
 */
#include "common/mcp_skills.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "janas/mcp.h"
#include "llm/json.h"

int janas_skills_dir(char *out, size_t n)
{
    char cfg[4096];
    int32_t len = 0;
    if (janas_mcp_config_path(cfg, sizeof cfg, &len) != JANAS_MCP_OK)
        return -1;
    char *sl = strrchr(cfg, '/');
#ifdef _WIN32
    char *bs = strrchr(cfg, '\\');
    if (!sl || (bs && bs > sl))
        sl = bs;
#endif
    if (!sl)
        return -1;
    *sl = 0;
    return snprintf(out, n, "%s/skills", cfg) < (int)n ? 0 : -1;
}

int janas_skills_any(void)
{
    char root[4200];
    if (janas_skills_dir(root, sizeof root) != 0)
        return 0;
    DIR *d = opendir(root);
    struct dirent *e;
    int any = 0;
    while (d && !any && (e = readdir(d))) {
        if (e->d_name[0] == '.')
            continue;
        char p[4500];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s/SKILL.md", root, e->d_name);
        any = stat(p, &st) == 0 && S_ISREG(st.st_mode);
    }
    if (d)
        closedir(d);
    return any;
}

/* a value of the frontmatter, its quotes off */
static void value(const char *v, size_t vn, char *out, size_t cap)
{
    while (vn && (*v == ' ' || *v == '\t'))
        v++, vn--;
    while (vn && (v[vn - 1] == ' ' || v[vn - 1] == '\r'))
        vn--;
    if (vn >= 2 && (v[0] == '"' || v[0] == '\'') && v[vn - 1] == v[0])
        v++, vn -= 2;
    snprintf(out, cap, "%.*s", (int)(vn < cap ? vn : cap - 1), v);
}

int janas_skill_info(const char *dir, const char *folder, char *name,
                     size_t name_n, char *about, size_t about_n)
{
    char p[4400];
    snprintf(p, sizeof p, "%s/SKILL.md", dir);
    FILE *f = fopen(p, "rb");
    if (!f)
        return -1;
    char *t = malloc(65537);
    size_t r = t ? fread(t, 1, 65536, f) : 0;
    fclose(f);
    if (!t)
        return -1;
    t[r] = 0;
    snprintf(name, name_n, "%s", folder);
    about[0] = 0;
    if (!strncmp(t, "---", 3)) {
        char *line = strchr(t, '\n');
        while (line) {
            line++;
            char *nl = strchr(line, '\n');
            size_t ln = nl ? (size_t)(nl - line) : strlen(line);
            if (ln >= 3 && !strncmp(line, "---", 3))
                break;
            char *c = memchr(line, ':', ln);
            if (c) {
                size_t kn = (size_t)(c - line);
                if (kn == 4 && !strncmp(line, "name", 4))
                    value(c + 1, ln - kn - 1, name, name_n);
                else if (kn == 11 && !strncmp(line, "description", 11)) {
                    value(c + 1, ln - kn - 1, about, about_n);
                    /* YAML's block (> or |, -, +): the indented lines
                       after it, joined by a space */
                    if ((about[0] == '>' || about[0] == '|') &&
                        strspn(about + 1, "-+0123456789") ==
                            strlen(about + 1)) {
                        size_t k = 0;
                        about[0] = 0;
                        while (nl && (nl[1] == ' ' || nl[1] == '\t' ||
                                      nl[1] == '\r' || nl[1] == '\n')) {
                            char *b = nl + 1, *e = strchr(b, '\n');
                            size_t bn = e ? (size_t)(e - b) : strlen(b);
                            char part[1024];
                            value(b, bn, part, sizeof part);
                            if (*part && k + 2 < about_n)
                                k += (size_t)snprintf(about + k, about_n - k,
                                                      "%s%s", k ? " " : "",
                                                      part);
                            if (k >= about_n)
                                k = about_n - 1;
                            nl = e;
                        }
                    }
                }
            }
            line = nl;
        }
    }
    free(t);
    for (char *c = name; *c; c++)
        if (!((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
              (*c >= '0' && *c <= '9') || *c == '-' || *c == '_' || *c == '.'))
            *c = '-';
    return 0;
}

void janas_skills_about(struct janas_buf *o)
{
    char root[4200];
    if (janas_skills_dir(root, sizeof root) != 0)
        return;
    DIR *d = opendir(root);
    struct dirent *e;
    int n = 0;
    while (d && n < 64 && (e = readdir(d))) {
        if (e->d_name[0] == '.')
            continue;
        char dir[4500], name[128], about[1024];
        snprintf(dir, sizeof dir, "%s/%s", root, e->d_name);
        if (janas_skill_info(dir, e->d_name, name, sizeof name, about,
                             sizeof about) != 0)
            continue;
        if (!n++)
            janas_buf_puts(o, "the user's skills - for these tasks follow "
                              "the user's skill rather than doing them "
                              "yourself: ");
        else
            janas_buf_puts(o, "; ");
        janas_buf_printf(o, "%s (%s)", name,
                         about[0] ? about : "no description");
    }
    if (d)
        closedir(d);
}
