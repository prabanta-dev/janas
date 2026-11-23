/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * skills.c - skills for the model of Brebus's chat, in the format of
 * Claude's: a folder holding SKILL.md - a frontmatter with its name and
 * description, then the instructions - and the files the instructions
 * name (scripts, templates, references). A skill made for Claude works here
 * as it is.
 *
 * Where: the folder skills/ beside the user's mcp.json (~/.config/janas/
 * skills, %APPDATA%\janas\skills) for every chat, and a project's
 * .brebus/skills/ for that project's. The model's catalog names each skill
 * with its description; the model reads one when a task needs it, with
 * janas__skill (its SKILL.md, or a file of its folder, never outside it).
 * settings.json's _skills.off turns some off.
 */
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/mcp_skills.h"
#include "common/paths.h"
#include "janas/mcp.h"

#define MAX_SKILLS 64
#define MAX_READ (256 * 1024)

struct skill {
    char name[128], about[1024], dir[4200];
    int project; /* the project's own, not every chat's */
};

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

/* a file whole, at most max bytes and a NUL; NULL when it cannot be read */
static char *slurp(const char *path, size_t max, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    char *b = malloc(max + 1);
    size_t r = b ? fread(b, 1, max, f) : 0;
    fclose(f);
    if (!b)
        return NULL;
    b[r] = 0;
    if (n)
        *n = r;
    return b;
}

/* a skill's name and description from its SKILL.md: 0, or -1 with none */
static int read_skill(const char *dir, const char *folder, struct skill *s)
{
    snprintf(s->dir, sizeof s->dir, "%s", dir);
    return janas_skill_info(dir, folder, s->name, sizeof s->name, s->about,
                            sizeof s->about);
}

static int scan_dir(const char *root, int project, struct skill *out, int n,
                    int max)
{
    DIR *d = opendir(root);
    struct dirent *e;
    while (d && n < max && (e = readdir(d))) {
        if (e->d_name[0] == '.')
            continue;
        char dir[4200];
        struct stat st;
        snprintf(dir, sizeof dir, "%s/%s", root, e->d_name);
        if (stat(dir, &st) != 0 || !S_ISDIR(st.st_mode))
            continue;
        if (read_skill(dir, e->d_name, &out[n]) == 0)
            out[n++].project = project;
    }
    if (d)
        closedir(d);
    return n;
}

/* every skill there is: the project's first (a name it shares with one of
   every chat's is the project's) */
static int scan(struct brebus *b, struct skill *out, int max)
{
    char ws[4096], dir[4200];
    int n = 0;
    if (brebus_ws_path(b, ws, sizeof ws) == 0) {
        snprintf(dir, sizeof dir, "%s/.brebus/skills", ws);
        n = scan_dir(dir, 1, out, n, max);
    }
    int own = n;
    if (janas_skills_dir(dir, sizeof dir) == 0)
        n = scan_dir(dir, 0, out, n, max);
    /* the user's skills a project's of the same name hides */
    for (int i = own; i < n; i++)
        for (int j = 0; j < own; j++)
            if (!strcmp(out[i].name, out[j].name)) {
                out[i--] = out[--n];
                break;
            }
    return n;
}

/* the names turned off, settings.json's _skills.off */
static int is_off(struct brebus *b, const char *name)
{
    struct janas_buf f = {0};
    char err[160];
    brebus_settings_get(b, &f);
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *s =
        d ? janas_json_get(janas_json_root(d), "_skills") : NULL;
    const struct janas_json *off = s ? janas_json_get(s, "off") : NULL;
    int yes = 0;
    for (const struct janas_json *e =
             off && off->type == JANAS_JSON_ARRAY ? off->child : NULL;
         e && !yes; e = e->next)
        yes = janas_json_is(e, name);
    janas_json_free(d);
    janas_buf_free(&f);
    return yes;
}

void brebus_skills_get(struct brebus *b, struct janas_buf *out)
{
    static struct skill s[MAX_SKILLS];
    int n = scan(b, s, MAX_SKILLS);
    janas_buf_puts(out, "{\"skills\": [");
    for (int i = 0; i < n; i++) {
        janas_buf_puts(out, i ? ", {\"name\": " : "{\"name\": ");
        str(out, s[i].name);
        janas_buf_puts(out, ", \"description\": ");
        str(out, s[i].about);
        janas_buf_puts(out, ", \"path\": ");
        str(out, s[i].dir);
        janas_buf_printf(out, ", \"project\": %s, \"on\": %s}",
                         s[i].project ? "true" : "false",
                         is_off(b, s[i].name) ? "false" : "true");
    }
    char dir[4200] = "";
    janas_skills_dir(dir, sizeof dir);
    janas_buf_puts(out, "], \"dir\": ");
    str(out, dir);
    janas_buf_puts(out, "}");
}

void brebus_skills_tool(struct brebus *b, struct janas_buf *tool,
                        struct janas_buf *catalog)
{
    static struct skill s[MAX_SKILLS];
    int n = scan(b, s, MAX_SKILLS), on = 0;
    for (int i = 0; i < n; i++) {
        if (is_off(b, s[i].name))
            continue;
        if (!on++)
            janas_buf_puts(catalog,
                           "Skills: instructions for particular tasks. Before "
                           "doing one of these tasks, read its skill with "
                           "janas__skill, then follow it.");
        janas_buf_printf(catalog, "\n- %s: %s", s[i].name,
                         s[i].about[0] ? s[i].about : "(no description)");
    }
    if (!on)
        return;
    janas_buf_puts(
        tool,
        "{\"type\": \"function\", \"function\": {\"name\": \"janas__skill\", "
        "\"description\": \"Read a skill's instructions (its SKILL.md) or one "
        "of the files it names, before the task it is for.\", "
        "\"parameters\": {\"type\": \"object\", \"properties\": {\"name\": "
        "{\"type\": \"string\", \"description\": \"the skill's name\"}, "
        "\"file\": {\"type\": \"string\", \"description\": \"a file of the "
        "skill's folder, by its path in it; none: SKILL.md\"}}, "
        "\"required\": [\"name\"]}}}");
}

int brebus_skills_call(struct brebus *b, const char *args, size_t n,
                       struct janas_buf *out)
{
    char err[160];
    struct janas_json_doc *d =
        n ? janas_json_parse(args, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *name = o ? janas_json_str(janas_json_get(o, "name")) : NULL;
    const char *file = o ? janas_json_str(janas_json_get(o, "file")) : NULL;
    static struct skill s[MAX_SKILLS];
    int ns = scan(b, s, MAX_SKILLS);
    const struct skill *k = NULL;
    for (int i = 0; name && i < ns && !k; i++)
        if (!strcmp(s[i].name, name) && !is_off(b, name))
            k = &s[i];
    struct janas_buf t = {0};
    int bad = 1;
    if (!k)
        janas_buf_printf(&t, "There is no skill \"%s\".", name ? name : "");
    else if (file && *file &&
             (strstr(file, "..") || file[0] == '/' || file[0] == '\\' ||
              strchr(file, ':')))
        janas_buf_puts(&t, "A skill's file is named by its path inside the "
                           "skill's folder.");
    else {
        char p[4600];
        snprintf(p, sizeof p, "%s/%s", k->dir,
                 file && *file ? file : "SKILL.md");
        size_t len = 0;
        /* where it says, not by a link out of the skill's folder */
        char rd[4096], rp[4096];
        char *c = janas_path_real(k->dir, rd) && janas_path_real(p, rp) &&
                          janas_path_in(rp, rd)
                      ? slurp(p, MAX_READ, &len)
                      : NULL;
        if (!c)
            janas_buf_printf(&t, "The skill %s has no file %s.", k->name,
                             file && *file ? file : "SKILL.md");
        else {
            janas_buf_put(&t, c, len);
            if (len == MAX_READ)
                janas_buf_puts(&t, "\n[... the file goes on]");
            free(c);
            bad = 0;
        }
    }
    janas_buf_puts(out, "{\"result\": ");
    janas_json_write_str(out, t.p ? t.p : "", t.n);
    janas_buf_printf(out, ", \"is_error\": %s}", bad ? "true" : "false");
    janas_buf_free(&t);
    janas_json_free(d);
    return 0;
}

/* a folder copied whole into another, made: 0, or -1 */
static int copy_tree(const char *from, const char *to, int depth)
{
    if (depth > 8 || janas_mkdirs(to) != 0)
        return -1;
    DIR *d = opendir(from);
    if (!d)
        return -1;
    struct dirent *e;
    int rc = 0;
    while (rc == 0 && (e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        char a[4400], z[4400];
        struct stat st;
        snprintf(a, sizeof a, "%s/%s", from, e->d_name);
        snprintf(z, sizeof z, "%s/%s", to, e->d_name);
        /* links are not copied: a skill is its own files, not what a link
           in it points to */
        if (lstat(a, &st) != 0 || S_ISLNK(st.st_mode))
            continue;
        if (S_ISDIR(st.st_mode)) {
            rc = copy_tree(a, z, depth + 1);
            continue;
        }
        FILE *in = fopen(a, "rb"), *o = in ? fopen(z, "wb") : NULL;
        char buf[65536];
        size_t r;
        while (o && (r = fread(buf, 1, sizeof buf, in)) > 0)
            if (fwrite(buf, 1, r, o) != r)
                rc = -1;
        if (!o || fclose(o) != 0)
            rc = -1;
        if (in)
            fclose(in);
    }
    closedir(d);
    return rc;
}

/* a folder and what it holds removed: 0, or -1 */
static int remove_tree(const char *dir, int depth)
{
    if (depth > 8)
        return -1;
    DIR *d = opendir(dir);
    if (!d)
        return -1;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        char p[4400];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        /* a link is removed, never followed */
        if (lstat(p, &st) == 0 && S_ISDIR(st.st_mode))
            remove_tree(p, depth + 1);
        else
            remove(p);
    }
    closedir(d);
    return rmdir(dir) == 0 ? 0 : -1;
}

int brebus_skills_add(struct brebus *b, const char *body, size_t n,
                      struct janas_buf *out)
{
    char err[200], dst[4400], root[4200];
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const char *src =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "path")) : NULL;
    struct skill s;
    const char *folder = src ? strrchr(src, '/') : NULL;
#ifdef _WIN32
    const char *bs = src ? strrchr(src, '\\') : NULL;
    if (!folder || (bs && bs > folder))
        folder = bs;
#endif
    folder = folder ? folder + 1 : src;
    int rc = -1;
    if (src && !brebus_confined(b, src))
        snprintf(err, sizeof err, "a skill of your own folder");
    else if (!src || !*src || read_skill(src, folder, &s) != 0)
        snprintf(err, sizeof err, "no SKILL.md in that folder");
    else if (janas_skills_dir(root, sizeof root) != 0)
        snprintf(err, sizeof err, "no folder for the skills");
    else {
        snprintf(dst, sizeof dst, "%s/%s", root, s.name);
        struct stat st;
        if (stat(dst, &st) == 0)
            snprintf(err, sizeof err, "a skill named %s is there already",
                     s.name);
        else if (copy_tree(src, dst, 0) != 0) {
            remove_tree(dst, 0);
            snprintf(err, sizeof err, "the folder could not be copied");
        } else
            rc = 0;
    }
    janas_json_free(d);
    janas_buf_puts(out, rc ? "{\"error\": " : "{\"name\": ");
    str(out, rc ? err : s.name);
    janas_buf_puts(out, "}");
    return rc;
}

int brebus_skills_remove(struct brebus *b, const char *body, size_t n,
                         struct janas_buf *out)
{
    char err[200];
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const char *name =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "name")) : NULL;
    static struct skill s[MAX_SKILLS];
    int ns = scan(b, s, MAX_SKILLS), rc = -1;
    /* only every chat's skills, copied there by Brebus or by the user: a
       project's live in its folder, the user's to keep */
    for (int i = 0; name && i < ns; i++)
        if (!s[i].project && !strcmp(s[i].name, name))
            rc = remove_tree(s[i].dir, 0);
    janas_json_free(d);
    janas_buf_puts(out,
                   rc ? "{\"error\": \"no such skill of every chat\"}" : "{}");
    return rc;
}
