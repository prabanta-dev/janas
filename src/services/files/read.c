/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * read.c - the tools that only read: what a folder holds (files_list), a
 * text file's lines (files_read), the files found by their name or by the
 * text they hold (files_search). The private files (place.c) are neither
 * listed nor read nor searched; a file that is not text is not read.
 */
#include <ctype.h>
#include <dirent.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "files.h"

#define LIST_MAX 400          /* lines of a folder's listing */
#define READ_LINES 400        /* a file's lines read, unless asked */
#define READ_LINES_MAX 2000   /* at most */
#define READ_BYTES 100000     /* of text in one answer */
#define FILE_MAX (32 << 20)   /* the largest file read */
#define SEARCH_MAX 100        /* hits */
#define SEARCH_FILE (4 << 20) /* the largest file searched for text */
#define SEARCH_VISIT 50000    /* files looked at, at most */
#define SEARCH_DEPTH 12

int fl_is_text(const char *p, size_t n)
{
    return memchr(p, 0, n < 65536 ? n : 65536) == NULL;
}

/* The whole file, NUL-ended, at most FILE_MAX bytes; NULL with why. */
static char *load(const char *path, size_t *n, char *why, size_t why_len)
{
    FILE *f = fopen(path, "rb");
    struct stat st;
    if (!f || fstat(fileno(f), &st) != 0 || !S_ISREG(st.st_mode)) {
        snprintf(why, why_len, "not a file that can be read");
        if (f)
            fclose(f);
        return NULL;
    }
    if (st.st_size > FILE_MAX) {
        snprintf(why, why_len, "%lld bytes: larger than %d MiB",
                 (long long)st.st_size, FILE_MAX >> 20);
        fclose(f);
        return NULL;
    }
    char *p = malloc((size_t)st.st_size + 1);
    size_t got = p ? fread(p, 1, (size_t)st.st_size, f) : 0;
    fclose(f);
    if (!p) {
        snprintf(why, why_len, "out of memory");
        return NULL;
    }
    p[got] = 0;
    *n = got;
    return p;
}

struct entry {
    char *name;
    int dir, link;
    long long size;
};

static int by_kind_name(const void *a, const void *b)
{
    const struct entry *x = a, *y = b;
    if (x->dir != y->dir)
        return y->dir - x->dir;
    return strcmp(x->name, y->name);
}

/* Folders that hold no one's work, only what programs keep: not searched,
   not walked into by a listing. */
static const char *const NOISE[] = {
    ".git",    "node_modules", ".cache", ".local",      ".var",        ".npm",
    ".cargo",  ".rustup",      ".venv",  "venv",        "__pycache__", ".Trash",
    ".gradle", ".m2",          ".nuget", ".vscode-oss", ".vscode",     "snap",
};

static int noise(const char *name)
{
    for (size_t i = 0; i < sizeof NOISE / sizeof NOISE[0]; i++)
        if (!strcmp(name, NOISE[i]))
            return 1;
    return 0;
}

/* The entries of dir, sorted, the private ones and (unless hidden) those
   whose name starts with a dot left out; *dots counts the second. */
static struct entry *entries(const struct fl_roots *r, const char *dir,
                             int hidden, int *n, int *dots)
{
    *n = 0;
    DIR *d = opendir(dir);
    if (!d)
        return NULL;
    struct entry *e = NULL;
    int cap = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        const char *nm = de->d_name;
        if (!strcmp(nm, ".") || !strcmp(nm, ".."))
            continue;
        char full[PATH_MAX];
        if ((size_t)snprintf(full, sizeof full, "%s/%s", dir, nm) >=
                sizeof full ||
            fl_private(r, full))
            continue;
        if (!hidden && nm[0] == '.') {
            (*dots)++;
            continue;
        }
        struct stat ls, st;
        if (lstat(full, &ls) != 0)
            continue;
        int is_link = S_ISLNK(ls.st_mode);
        char real[PATH_MAX];
        if (is_link && realpath(full, real) && fl_private(r, real))
            continue; /* a link to a private file is not shown either */
        if (is_link && stat(full, &st) != 0)
            st = ls;
        else if (!is_link)
            st = ls;
        if (*n == cap) {
            cap = cap ? 2 * cap : 64;
            struct entry *t = realloc(e, (size_t)cap * sizeof *e);
            if (!t)
                break;
            e = t;
        }
        e[*n] = (struct entry){.name = strdup(nm),
                               .dir = S_ISDIR(st.st_mode),
                               .link = is_link,
                               .size = (long long)st.st_size};
        if (e[*n].name)
            (*n)++;
    }
    closedir(d);
    if (*n)
        qsort(e, (size_t)*n, sizeof *e, by_kind_name);
    return e;
}

static void free_entries(struct entry *e, int n)
{
    for (int i = 0; i < n; i++)
        free(e[i].name);
    free(e);
}

static void size_text(long long n, char *out, size_t cap)
{
    if (n < 1024)
        snprintf(out, cap, "%lld B", n);
    else if (n < 1024 * 1024)
        snprintf(out, cap, "%.1f KiB", n / 1024.0);
    else if (n < 1024LL * 1024 * 1024)
        snprintf(out, cap, "%.1f MiB", n / 1048576.0);
    else
        snprintf(out, cap, "%.1f GiB", n / 1073741824.0);
}

/* The listing of dir into t, depth levels down; lines counts them. */
static void list_dir(const struct fl_roots *r, const char *dir, int depth,
                     int level, int hidden, struct janas_buf *t, int *lines,
                     int *more, int *dots)
{
    int n;
    struct entry *e = entries(r, dir, hidden, &n, dots);
    for (int i = 0; i < n; i++) {
        if (*lines >= LIST_MAX) {
            (*more)++;
            continue;
        }
        char sz[32] = "";
        if (!e[i].dir)
            size_text(e[i].size, sz, sizeof sz);
        janas_buf_printf(t, "%*s%s%s%s%s%s\n", 2 * level, "", e[i].name,
                         e[i].dir ? "/" : "", e[i].link ? " (link)" : "",
                         sz[0] ? "  " : "", sz);
        (*lines)++;
        /* a link to a folder is shown, not walked; nor are git's insides
           and the dependencies of a project */
        if (e[i].dir && !e[i].link && level < depth && !noise(e[i].name)) {
            char sub[PATH_MAX];
            if ((size_t)snprintf(sub, sizeof sub, "%s/%s", dir, e[i].name) <
                sizeof sub)
                list_dir(r, sub, depth, level + 1, hidden, t, lines, more,
                         dots);
        }
    }
    free_entries(e, n);
}

int fl_tool_list(const struct fl_roots *r, const struct janas_json *args,
                 struct janas_buf *b)
{
    const char *in = fl_arg_str(args, "path");
    char dir[PATH_MAX], why[PATH_MAX + 200], s[PATH_MAX];
    if (fl_resolve(r, in ? in : ".", 1, dir, sizeof dir, why, sizeof why))
        return fl_fail(b, "%s.", why);
    struct stat st;
    if (stat(dir, &st) != 0 || !S_ISDIR(st.st_mode))
        return fl_fail(b, "%s is not a folder: files_read reads a file.",
                       in ? in : ".");
    int depth = (int)fl_arg_num(args, "depth", 1);
    depth = depth < 1 ? 1 : depth > 3 ? 3 : depth;
    struct janas_buf t = {0};
    fl_short(r, dir, s, sizeof s);
    janas_buf_printf(&t, "%s/\n", s);
    int lines = 0, more = 0, dots = 0, hidden = fl_arg_bool(args, "hidden");
    size_t head = t.n;
    list_dir(r, dir, depth, 1, hidden, &t, &lines, &more, &dots);
    if (!lines && dots) {
        /* nothing but names starting with a dot, as a Gambas project's
           .project and .src: those are what the folder holds */
        t.n = head;
        hidden = 1;
        lines = more = dots = 0;
        list_dir(r, dir, depth, 1, hidden, &t, &lines, &more, &dots);
    }
    if (!lines)
        janas_buf_puts(&t, "  (empty)\n");
    if (more)
        janas_buf_printf(&t, "... and %d more\n", more);
    if (dots)
        janas_buf_printf(&t,
                         "(%d name%s starting with a dot not shown: hidden "
                         "true shows them)\n",
                         dots, dots == 1 ? "" : "s");
    return fl_ok(b, &t);
}

int fl_tool_read(const struct fl_roots *r, const struct janas_json *args,
                 struct janas_buf *b)
{
    const char *in = fl_arg_str(args, "path");
    char path[PATH_MAX], why[PATH_MAX + 200], s[PATH_MAX];
    if (fl_resolve(r, in, 1, path, sizeof path, why, sizeof why))
        return fl_fail(b, "%s.", why);
    size_t n = 0;
    char *p = load(path, &n, why, sizeof why);
    if (!p)
        return fl_fail(b, "%s: %s.", in, why);
    if (!fl_is_text(p, n)) {
        free(p);
        return fl_fail(b, "%s is not text: it is not read.", in);
    }
    long total = 0;
    for (size_t i = 0; i < n; i++)
        total += p[i] == '\n';
    if (n && p[n - 1] != '\n')
        total++;
    long from = (long)fl_arg_num(args, "offset", 1);
    long count = (long)fl_arg_num(args, "lines", READ_LINES);
    from = from < 1 ? 1 : from;
    count = count < 1 ? 1 : count > READ_LINES_MAX ? READ_LINES_MAX : count;
    /* the lines from..from+count-1, at most READ_BYTES of them */
    const char *a = p;
    for (long l = 1; l < from && a < p + n; l++) {
        const char *nl = memchr(a, '\n', (size_t)(p + n - a));
        a = nl ? nl + 1 : p + n;
    }
    const char *e = a;
    long last = from - 1;
    while (e < p + n && last < from + count - 1 && e - a < READ_BYTES) {
        const char *nl = memchr(e, '\n', (size_t)(p + n - e));
        e = nl ? nl + 1 : p + n;
        last++;
    }
    struct janas_buf t = {0};
    fl_short(r, path, s, sizeof s);
    char sz[32];
    size_text((long long)n, sz, sizeof sz);
    if (from > total)
        janas_buf_printf(&t, "%s: %ld lines (%s); nothing from line %ld.\n", s,
                         total, sz, from);
    else
        janas_buf_printf(&t, "%s: lines %ld-%ld of %ld (%s)\n", s, from, last,
                         total, sz);
    janas_buf_put(&t, a, (size_t)(e - a));
    if (e > a && e[-1] != '\n')
        janas_buf_puts(&t, "\n");
    if (last < total)
        janas_buf_printf(&t, "[%ld more lines: offset %ld reads on]\n",
                         total - last, last + 1);
    free(p);
    return fl_ok(b, &t);
}

/* 1 when s holds what (ASCII letters in either case) */
static int holds(const char *s, size_t n, const char *what, size_t w)
{
    for (size_t i = 0; i + w <= n; i++) {
        size_t k = 0;
        while (k < w && tolower((unsigned char)s[i + k]) ==
                            tolower((unsigned char)what[k]))
            k++;
        if (k == w)
            return 1;
    }
    return 0;
}

struct search {
    const struct fl_roots *r;
    const char *name, *text;
    int hidden, hits, visited, more;
    struct janas_buf *t;
};

static void search_file(struct search *q, const char *path)
{
    char s[PATH_MAX];
    fl_short(q->r, path, s, sizeof s);
    if (!q->text) {
        if (q->hits++ < SEARCH_MAX)
            janas_buf_printf(q->t, "%s\n", s);
        return;
    }
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size > SEARCH_FILE)
        return;
    char why[64];
    size_t n = 0;
    char *p = load(path, &n, why, sizeof why);
    if (!p)
        return;
    if (fl_is_text(p, n)) {
        size_t w = strlen(q->text);
        long line = 1;
        for (const char *a = p; a < p + n; line++) {
            const char *nl = memchr(a, '\n', (size_t)(p + n - a));
            size_t len = (size_t)((nl ? nl : p + n) - a);
            if (holds(a, len, q->text, w) && q->hits++ < SEARCH_MAX) {
                while (len && (a[len - 1] == '\r' || a[len - 1] == ' '))
                    len--;
                while (len && (*a == ' ' || *a == '\t'))
                    a++, len--;
                janas_buf_printf(q->t, "%s:%ld: %.*s%s\n", s, line,
                                 (int)(len > 200 ? 200 : len), a,
                                 len > 200 ? "..." : "");
            }
            a = nl ? nl + 1 : p + n;
        }
    }
    free(p);
}

static void search_dir(struct search *q, const char *dir, int level)
{
    int n, dots = 0;
    struct entry *e = entries(q->r, dir, q->hidden, &n, &dots);
    for (int i = 0; i < n; i++) {
        if (q->visited >= SEARCH_VISIT) {
            q->more = 1;
            break;
        }
        char full[PATH_MAX];
        if ((size_t)snprintf(full, sizeof full, "%s/%s", dir, e[i].name) >=
                sizeof full ||
            e[i].link)
            continue;
        if (e[i].dir) {
            if (level < SEARCH_DEPTH && !noise(e[i].name))
                search_dir(q, full, level + 1);
            continue;
        }
        q->visited++;
        if (q->name && fnmatch(q->name, e[i].name, FNM_CASEFOLD))
            continue;
        search_file(q, full);
    }
    free_entries(e, n);
}

int fl_tool_search(const struct fl_roots *r, const struct janas_json *args,
                   struct janas_buf *b)
{
    const char *in = fl_arg_str(args, "path");
    struct search q = {.r = r,
                       .name = fl_arg_str(args, "name"),
                       .text = fl_arg_str(args, "text"),
                       .hidden = 1};
    /* the names starting with a dot are searched unless asked not to: a
       project's own may be all it has (.src) */
    const struct janas_json *hv = janas_json_get(args, "hidden");
    if (hv && hv->type == JANAS_JSON_FALSE)
        q.hidden = 0;
    if (!q.name && !q.text)
        return fl_fail(b, "Give a name (a pattern such as *.class) or a "
                          "text to look for, or both.");
    char dir[PATH_MAX], why[PATH_MAX + 200], s[PATH_MAX];
    if (fl_resolve(r, in ? in : ".", 1, dir, sizeof dir, why, sizeof why))
        return fl_fail(b, "%s.", why);
    struct janas_buf t = {0};
    q.t = &t;
    search_dir(&q, dir, 0);
    struct janas_buf all = {0};
    fl_short(r, dir, s, sizeof s);
    janas_buf_printf(&all, "In %s, %s%s%s%s%s: %d found%s.\n", s,
                     q.name ? "files named " : "", q.name ? q.name : "",
                     q.name && q.text ? ", " : "",
                     q.text ? "lines holding " : "", q.text ? q.text : "",
                     q.hits,
                     q.hits > SEARCH_MAX ? " (the first 100 shown)" : "");
    janas_buf_put(&all, t.p ? t.p : "", t.n);
    if (q.more)
        janas_buf_printf(&all, "[stopped after %d files]\n", SEARCH_VISIT);
    janas_buf_free(&t);
    return fl_ok(b, &all);
}
