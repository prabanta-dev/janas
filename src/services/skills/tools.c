/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - janas-skills's tools (see skills.h): skills_list and
 * skill_read only read, and say so; skill_run runs a script of a skill in
 * a cage, without network, and says it is no mere reading, so a client
 * asks the user first.
 */
#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "common/mcp_server.h"
#include "common/mcp_skills.h"
#include "common/paths.h"
#include "skills.h"

#define MAX_READ (256 * 1024)
#define MAX_FILES 100

static void script_tools(const struct sk_ctx *c, struct janas_buf *b);

void sk_tools_list(void *ctx, struct janas_buf *b)
{
    janas_buf_puts(
        b,
        "\"tools\": [{\"name\": \"skills_list\", \"title\": \"Skills\", "
        "\"description\": \"The user's skills, each with what it is for: "
        "instructions for particular tasks.\", \"inputSchema\": {\"type\": "
        "\"object\", \"properties\": {}, \"additionalProperties\": false}, "
        "\"annotations\": {\"readOnlyHint\": true, \"openWorldHint\": "
        "false}}, "
        "{\"name\": \"skill_read\", \"title\": \"Read a skill\", "
        "\"description\": \"A skill's instructions (its SKILL.md) and the "
        "list of its files, or one of its files.\", \"inputSchema\": "
        "{\"type\": \"object\", \"properties\": {\"name\": {\"type\": "
        "\"string\", \"description\": \"the skill's name\"}, \"file\": "
        "{\"type\": \"string\", \"description\": \"a file of the skill, by "
        "its path in the skill's folder; leave it out for SKILL.md\"}}, "
        "\"required\": [\"name\"], \"additionalProperties\": false}, "
        "\"annotations\": {\"readOnlyHint\": true, \"openWorldHint\": "
        "false}}, "
        "{\"name\": \"skill_run\", \"title\": \"Run a skill's script\", "
        "\"description\": \"Runs a script of a skill (Python, shell or "
        "JavaScript, by its extension), as its instructions say, in a cage "
        "without network: what it printed, and the files it wrote.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": {\"name\": "
        "{\"type\": \"string\", \"description\": \"the skill's name\"}, "
        "\"script\": {\"type\": \"string\", \"description\": \"the script, by "
        "its path in the skill's folder\"}, \"args\": {\"type\": \"array\", "
        "\"items\": {\"type\": \"string\"}, \"description\": \"its "
        "arguments\"}, \"input\": {\"type\": \"string\", \"description\": "
        "\"text for its standard input\"}}, \"required\": [\"name\", "
        "\"script\"], \"additionalProperties\": false}, \"annotations\": "
        "{\"readOnlyHint\": false, \"destructiveHint\": false, "
        "\"openWorldHint\": false}}");
    script_tools(ctx, b);
    janas_buf_puts(b, "]");
}

static int fail(struct janas_buf *b, const char *fmt, ...)
    __attribute__((format(gnu_printf, 2, 3)));

static int fail(struct janas_buf *b, const char *fmt, ...)
{
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    janas_mcps_text_result(b, line, strlen(line), 1);
    return 0;
}

static int scan_dir(const char *root, struct sk_skill *out, int n)
{
    DIR *d = opendir(root);
    struct dirent *e;
    while (d && n < SK_MAX && (e = readdir(d))) {
        if (e->d_name[0] == '.')
            continue;
        struct sk_skill *k = &out[n];
        struct stat st;
        snprintf(k->dir, sizeof k->dir, "%s/%s", root, e->d_name);
        if (stat(k->dir, &st) != 0 || !S_ISDIR(st.st_mode) ||
            janas_skill_info(k->dir, e->d_name, k->name, sizeof k->name,
                             k->about, sizeof k->about) != 0)
            continue;
        /* a name found already, in a folder searched before, wins */
        int seen = 0;
        for (int i = 0; i < n && !seen; i++)
            seen = !strcmp(out[i].name, k->name);
        n += !seen;
    }
    if (d)
        closedir(d);
    return n;
}

/* every skill there is, those of the folders given first */
static int scan(const struct sk_ctx *c, struct sk_skill *out)
{
    int n = 0;
    for (int i = 0; i < c->n_dirs; i++)
        n = scan_dir(c->dirs[i], out, n);
    char dir[4200];
    if (janas_skills_dir(dir, sizeof dir) == 0)
        n = scan_dir(dir, out, n);
    return n;
}

static struct sk_skill S[SK_MAX];

static const struct sk_skill *find(const struct sk_ctx *c, const char *name,
                                   int *n)
{
    *n = scan(c, S);
    for (int i = 0; name && i < *n; i++)
        if (!strcmp(S[i].name, name))
            return &S[i];
    return NULL;
}

/* a path inside a skill's folder, as a model may give it */
static int inside(const char *p)
{
    return p && *p && !strstr(p, "..") && p[0] != '/' && p[0] != '\\' &&
           !strchr(p, ':');
}

/* p (dir/a file) where it says, not by a link outside dir: 1, or 0 */
int sk_within(const char *dir, const char *p)
{
    char rd[4096], rp[4096];
    return janas_path_real(dir, rd) && janas_path_real(p, rp) &&
           janas_path_in(rp, rd);
}

/* p opened for reading when it is in dir: its real path opened, no link
   followed at its end, and on Linux the file opened checked to be that
   path (a folder swapped for a link after the check is refused) */
static FILE *open_within(const char *dir, const char *p)
{
    char rd[4096], rp[4096];
    if (!janas_path_real(dir, rd) || !janas_path_real(p, rp) ||
        !janas_path_in(rp, rd))
        return NULL;
    int fd = open(rp, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0)
        return NULL;
#ifdef __linux__
    char self[64], at[4096];
    snprintf(self, sizeof self, "/proc/self/fd/%d", fd);
    ssize_t an = readlink(self, at, sizeof at - 1);
    if (an <= 0 || (at[an] = 0, strcmp(at, rp))) {
        close(fd);
        return NULL;
    }
#endif
    struct stat st;
    FILE *f =
        fstat(fd, &st) == 0 && S_ISREG(st.st_mode) ? fdopen(fd, "rb") : NULL;
    if (!f)
        close(fd);
    return f;
}

static int t_list(const struct sk_ctx *c, const struct janas_json *args,
                  struct janas_buf *b)
{
    (void)args;
    int n = scan(c, S);
    struct janas_buf t = {0};
    if (!n)
        janas_buf_puts(&t, "The user has no skills.");
    for (int i = 0; i < n; i++)
        janas_buf_printf(&t, "%s- %s: %s", i ? "\n" : "", S[i].name,
                         S[i].about[0] ? S[i].about : "(no description)");
    janas_mcps_text_result(b, t.p ? t.p : "", t.n, 0);
    janas_buf_free(&t);
    return 0;
}

/* the files of a folder, by their paths in the skill's, at most MAX_FILES
   in all */
static void files_of(const char *top, const char *dir, const char *rel,
                     struct janas_buf *t, int *count, int depth)
{
    DIR *d = depth < 4 ? opendir(dir) : NULL;
    struct dirent *e;
    while (d && *count < MAX_FILES && (e = readdir(d))) {
        if (e->d_name[0] == '.')
            continue;
        char p[4600], r[1024];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        snprintf(r, sizeof r, "%s%s%s", rel, *rel ? "/" : "", e->d_name);
        if (stat(p, &st) != 0)
            continue;
        if (S_ISDIR(st.st_mode))
            files_of(top, p, r, t, count, depth + 1);
        else if (S_ISREG(st.st_mode) && strcmp(r, "SKILL.md") &&
                 sk_within(top, p)) {
            janas_buf_printf(t, "\n- %s", r);
            ++*count;
        }
    }
    if (d)
        closedir(d);
}

static int t_read(const struct sk_ctx *c, const struct janas_json *args,
                  struct janas_buf *b)
{
    const char *name = janas_json_str(janas_json_get(args, "name"));
    const char *file = janas_json_str(janas_json_get(args, "file"));
    int n;
    const struct sk_skill *k = find(c, name, &n);
    if (!k)
        return fail(b, "There is no skill \"%s\".", name ? name : "");
    if (file && *file && !inside(file))
        return fail(b, "A skill's file is named by its path inside the "
                       "skill's folder.");
    char p[4600];
    snprintf(p, sizeof p, "%s/%s", k->dir, file && *file ? file : "SKILL.md");
    FILE *f = open_within(k->dir, p);
    char *txt = f ? malloc(MAX_READ + 1) : NULL;
    size_t len = txt ? fread(txt, 1, MAX_READ, f) : 0;
    if (f)
        fclose(f);
    if (!txt)
        return fail(b, "The skill %s has no file %s.", k->name,
                    file && *file ? file : "SKILL.md");
    struct janas_buf t = {0};
    janas_buf_put(&t, txt, len);
    free(txt);
    if (len == MAX_READ)
        janas_buf_puts(&t, "\n[... the file goes on]");
    if (!file || !*file) {
        int count = 0;
        struct janas_buf l = {0};
        files_of(k->dir, k->dir, "", &l, &count, 0);
        if (count)
            janas_buf_printf(&t, "\n\n[The files of the skill %s:%.*s]",
                             k->name, (int)l.n, l.p);
        janas_buf_free(&l);
    }
    janas_mcps_text_result(b, t.p ? t.p : "", t.n, 0);
    janas_buf_free(&t);
    return 0;
}

static int t_run(const struct sk_ctx *c, const struct janas_json *args,
                 struct janas_buf *b)
{
    const char *name = janas_json_str(janas_json_get(args, "name"));
    const char *script = janas_json_str(janas_json_get(args, "script"));
    const struct janas_json *a = janas_json_get(args, "args");
    const char *input = janas_json_str(janas_json_get(args, "input"));
    int n;
    const struct sk_skill *k = find(c, name, &n);
    if (!k)
        return fail(b, "There is no skill \"%s\".", name ? name : "");
    if (!inside(script))
        return fail(b, "A script is named by its path inside the skill's "
                       "folder.");
    if (a && a->type != JANAS_JSON_ARRAY)
        return fail(b, "The arguments are a list of strings.");
    for (const struct janas_json *e = a ? a->child : NULL; e; e = e->next)
        if (e->type != JANAS_JSON_STRING)
            return fail(b, "The arguments are a list of strings.");
    return sk_run(k, script, a, input, b);
}

/* A tool for each script of a skill: skill_<skill>_<script>, its
   description the skill's, run as skill_run runs it. */
struct script {
    char tool[65], rel[512];
    const struct sk_skill *k;
};

static struct script SC[SK_SCRIPTS];

static int is_script(const char *f)
{
    const char *dot = strrchr(f, '.');
#ifdef _WIN32
    return dot && !strcmp(dot, ".py");
#else
    return dot && (!strcmp(dot, ".py") || !strcmp(dot, ".sh") ||
                   !strcmp(dot, ".js") || !strcmp(dot, ".mjs"));
#endif
}

static void scripts_in(const struct sk_skill *k, const char *dir,
                       const char *rel, int *n, int depth)
{
    DIR *d = depth < 4 ? opendir(dir) : NULL;
    struct dirent *e;
    while (d && *n < SK_SCRIPTS && (e = readdir(d))) {
        if (e->d_name[0] == '.')
            continue;
        char p[4600], r[512];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        snprintf(r, sizeof r, "%s%s%s", rel, *rel ? "/" : "", e->d_name);
        if (stat(p, &st) != 0)
            continue;
        if (S_ISDIR(st.st_mode)) {
            scripts_in(k, p, r, n, depth + 1);
            continue;
        }
        if (!S_ISREG(st.st_mode) || !is_script(r) || !sk_within(k->dir, p))
            continue;
        struct script *s = &SC[*n];
        s->k = k;
        snprintf(s->rel, sizeof s->rel, "%s", r);
        /* its name, its extension in it (hi.py: skill_<skill>_hi_py):
           letters, digits, _ and -, 64 at most, never twice */
        char base[512];
        snprintf(base, sizeof base, "%s", e->d_name);
        snprintf(s->tool, sizeof s->tool, "skill_%.24s_%.30s", k->name, base);
        for (char *c = s->tool; *c; c++)
            if (!((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
                  (*c >= '0' && *c <= '9') || *c == '-'))
                *c = '_';
        for (int i = 0, again = 1; again && i < 100; i++) {
            again =
                !strcmp(s->tool, "skill_run") || !strcmp(s->tool, "skill_read");
            for (int j = 0; j < *n && !again; j++)
                again = !strcmp(SC[j].tool, s->tool);
            if (again) {
                size_t l = strlen(s->tool);
                snprintf(s->tool + (l > 60 ? 60 : l),
                         sizeof s->tool - (l > 60 ? 60 : l), "_%d", i + 2);
            }
        }
        ++*n;
    }
    if (d)
        closedir(d);
}

static int scripts(const struct sk_ctx *c)
{
    int ns = scan(c, S), n = 0;
    for (int i = 0; i < ns; i++)
        scripts_in(&S[i], S[i].dir, "", &n, 0);
    return n;
}

static void script_tools(const struct sk_ctx *c, struct janas_buf *b)
{
    int n = scripts(c);
    for (int i = 0; i < n; i++) {
        const struct script *s = &SC[i];
        janas_buf_puts(b, ", {\"name\": ");
        janas_json_write_str(b, s->tool, strlen(s->tool));
        struct janas_buf d = {0};
        janas_buf_printf(&d,
                         "Runs %s of the skill %s, in a cage without "
                         "network; read the skill first (skill_read), "
                         "which says how. The skill: %s",
                         s->rel, s->k->name,
                         s->k->about[0] ? s->k->about : "(no description)");
        janas_buf_puts(b, ", \"description\": ");
        janas_json_write_str(b, d.p ? d.p : "", d.n);
        janas_buf_free(&d);
        janas_buf_puts(
            b, ", \"inputSchema\": {\"type\": \"object\", \"properties\": "
               "{\"args\": {\"type\": \"array\", \"items\": {\"type\": "
               "\"string\"}, \"description\": \"its arguments\"}, "
               "\"input\": {\"type\": \"string\", \"description\": \"text "
               "for its standard input\"}}, \"additionalProperties\": false}, "
               "\"annotations\": {\"readOnlyHint\": false, "
               "\"destructiveHint\": false, \"openWorldHint\": false}}");
    }
}

int sk_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    const struct sk_ctx *c = ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    static const struct {
        const char *name;
        int (*fn)(const struct sk_ctx *, const struct janas_json *,
                  struct janas_buf *);
    } T[] = {
        {"skills_list", t_list}, {"skill_read", t_read}, {"skill_run", t_run}};
    for (size_t i = 0; i < sizeof T / sizeof *T; i++)
        if (janas_json_is(name, T[i].name))
            return T[i].fn(c, args, b);
    int n = scripts(c);
    for (int i = 0; i < n; i++)
        if (janas_json_is(name, SC[i].tool)) {
            const struct janas_json *a = janas_json_get(args, "args");
            if (a && a->type != JANAS_JSON_ARRAY)
                return fail(b, "The arguments are a list of strings.");
            for (const struct janas_json *e = a ? a->child : NULL; e;
                 e = e->next)
                if (e->type != JANAS_JSON_STRING)
                    return fail(b, "The arguments are a list of strings.");
            return sk_run(SC[i].k, SC[i].rel, a,
                          janas_json_str(janas_json_get(args, "input")), b);
        }
    return -1;
}
