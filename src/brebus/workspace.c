/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * workspace.c - the folder Brebus works in, as Claude Code does: its
 * conversations (.brebus/chats), what is copied or downloaded into it
 * (sources/) and what it writes (exports/). Given with --workspace, or the
 * folder brebus was started in from a terminal; else the page asks: a
 * recent one, a path, or a new one, <Documents>/Brebus/<date> (then
 * "<date> (2)"...). Never the home itself nor /: too wide for a model's
 * files. The recent ones are listed in the data folder (workspaces.json).
 */
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"

#define MAX_RECENT 20

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

int brebus_ws_path(struct brebus *b, char *out, size_t n)
{
    pthread_mutex_lock(&b->mu);
    int on = b->ws[0] != 0;
    snprintf(out, n, "%s", b->ws);
    pthread_mutex_unlock(&b->mu);
    return on ? 0 : -1;
}

/* the documents' folder: user-dirs.dirs's XDG_DOCUMENTS_DIR, ~/Documents */
static void documents(char *dir, size_t n)
{
    const char *home = getenv("HOME"), *xc = getenv("XDG_CONFIG_HOME");
    char conf[4200], line[1024];
    if (!home)
        home = ".";
    if (xc && *xc == '/')
        snprintf(conf, sizeof conf, "%s/user-dirs.dirs", xc);
    else
        snprintf(conf, sizeof conf, "%s/.config/user-dirs.dirs", home);
    dir[0] = 0;
    FILE *f = fopen(conf, "r");
    while (f && fgets(line, sizeof line, f)) {
        static const char key[] = "XDG_DOCUMENTS_DIR=\"";
        if (strncmp(line, key, sizeof key - 1))
            continue;
        char *v = line + sizeof key - 1, *q = strchr(v, '"');
        if (q) {
            *q = 0;
            if (!strncmp(v, "$HOME/", 6))
                snprintf(dir, n, "%s/%s", home, v + 6);
            else if (*v == '/')
                snprintf(dir, n, "%s", v);
        }
        break;
    }
    if (f)
        fclose(f);
    if (!dir[0])
        snprintf(dir, n, "%s/Documents", home);
}

/* a folder too wide to be one: the home, / */
static int too_wide(const char *real)
{
    char home[PATH_MAX];
    const char *h = getenv("HOME");
    return !strcmp(real, "/") ||
           (h && realpath(h, home) && !strcmp(real, home));
}

static void recent_file(const struct brebus *b, char *p, size_t n)
{
    snprintf(p, n, "%s/workspaces.json", b->data_dir);
}

/* the recent ones, newest first, with path put first */
static void recent_add(struct brebus *b, const char *path)
{
    char rp[4200], err[160];
    recent_file(b, rp, sizeof rp);
    struct janas_buf f = {0}, o = {0};
    FILE *in = fopen(rp, "rb");
    if (in) {
        char buf[16384];
        size_t r;
        while ((r = fread(buf, 1, sizeof buf, in)) > 0 && f.n < (1u << 20))
            janas_buf_put(&f, buf, r);
        fclose(in);
    }
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *list =
        d ? janas_json_get(janas_json_root(d), "recent") : NULL;
    janas_buf_puts(&o, "{\"recent\": [");
    str(&o, path);
    int k = 1;
    for (const struct janas_json *e =
             list && list->type == JANAS_JSON_ARRAY ? list->child : NULL;
         e && k < MAX_RECENT; e = e->next) {
        const char *p = janas_json_str(e);
        if (!p || !strcmp(p, path))
            continue;
        janas_buf_puts(&o, ", ");
        str(&o, p);
        k++;
    }
    janas_buf_puts(&o, "]}\n");
    janas_json_free(d);
    char tmp[4300];
    snprintf(tmp, sizeof tmp, "%s.new", rp);
    FILE *out = o.oom ? NULL : fopen(tmp, "wb");
    if (out) {
        int bad = fwrite(o.p, 1, o.n, out) != o.n;
        bad |= fclose(out) != 0;
        if (bad || rename(tmp, rp) != 0)
            remove(tmp);
    }
    janas_buf_free(&f);
    janas_buf_free(&o);
}

/* the conversations the first Brebus kept in its data folder, moved once
   into the first workspace opened */
static void move_old_chats(struct brebus *b, const char *ws)
{
    char old[4200], to[4400];
    snprintf(old, sizeof old, "%s/chats", b->data_dir);
    DIR *d = opendir(old);
    if (!d)
        return;
    struct dirent *de;
    while ((de = readdir(d))) {
        size_t n = strlen(de->d_name);
        if (n < 6 || strcmp(de->d_name + n - 5, ".json"))
            continue;
        char from[4500], dest[4600];
        snprintf(from, sizeof from, "%s/%s", old, de->d_name);
        snprintf(to, sizeof to, "%s/.brebus/chats", ws);
        snprintf(dest, sizeof dest, "%s/%s", to, de->d_name);
        if (access(dest, F_OK) != 0)
            rename(from, dest);
    }
    closedir(d);
    rmdir(old); /* only once empty */
}

int brebus_ws_open(struct brebus *b, const char *path, int create, char *why,
                   size_t wn)
{
    char real[PATH_MAX], sub[PATH_MAX + 32];
    if (!path || *path != '/') {
        snprintf(why, wn, "a folder by its full path, as /home/...");
        return -1;
    }
    if (create && janas_mkdirs(path) != 0) {
        snprintf(why, wn, "cannot make %s", path);
        return -1;
    }
    struct stat st;
    if (!realpath(path, real) || stat(real, &st) != 0 || !S_ISDIR(st.st_mode)) {
        snprintf(why, wn, "%s is not a folder", path);
        return -1;
    }
    if (too_wide(real)) {
        snprintf(why, wn,
                 "%s is too wide to work in: choose or make a folder in it",
                 real);
        return -1;
    }
    snprintf(sub, sizeof sub, "%s/.brebus/chats", real);
    if (janas_mkdirs(sub) != 0 || access(real, W_OK) != 0) {
        snprintf(why, wn, "cannot write in %s", real);
        return -1;
    }
    pthread_mutex_lock(&b->mu);
    snprintf(b->ws, sizeof b->ws, "%s", real);
    pthread_mutex_unlock(&b->mu);
    move_old_chats(b, real);
    recent_add(b, real);
    brebus_tools_start(b); /* its documents and files for the chat */
    return 0;
}

int brebus_ws_new(struct brebus *b, char *why, size_t wn)
{
    char docs[4096], base[4200], path[4300], day[16];
    documents(docs, sizeof docs);
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(day, sizeof day, "%Y-%m-%d", &tm);
    snprintf(base, sizeof base, "%s/Brebus", docs);
    for (int i = 1; i < 1000; i++) {
        if (i == 1)
            snprintf(path, sizeof path, "%s/%s", base, day);
        else
            snprintf(path, sizeof path, "%s/%s (%d)", base, day, i);
        if (access(path, F_OK) != 0)
            return brebus_ws_open(b, path, 1, why, wn);
    }
    snprintf(why, wn, "no free name in %s", base);
    return -1;
}

void brebus_ws_status(struct brebus *b, struct janas_buf *out)
{
    char ws[4096], rp[4200], err[160], docs[4096];
    int on = brebus_ws_path(b, ws, sizeof ws) == 0;
    janas_buf_puts(out, "{\"path\": ");
    str(out, on ? ws : "");
    const char *slash = on ? strrchr(ws, '/') : NULL;
    janas_buf_puts(out, ", \"name\": ");
    str(out, slash ? slash + 1 : "");
    documents(docs, sizeof docs);
    janas_buf_puts(out, ", \"new_in\": ");
    snprintf(rp, sizeof rp, "%s/Brebus", docs);
    str(out, rp);
    janas_buf_puts(out, ", \"recent\": [");
    recent_file(b, rp, sizeof rp);
    struct janas_buf f = {0};
    FILE *in = fopen(rp, "rb");
    if (in) {
        char buf[16384];
        size_t r;
        while ((r = fread(buf, 1, sizeof buf, in)) > 0 && f.n < (1u << 20))
            janas_buf_put(&f, buf, r);
        fclose(in);
    }
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *list =
        d ? janas_json_get(janas_json_root(d), "recent") : NULL;
    int k = 0;
    for (const struct janas_json *e =
             list && list->type == JANAS_JSON_ARRAY ? list->child : NULL;
         e; e = e->next) {
        const char *p = janas_json_str(e);
        struct stat st;
        /* those still there */
        if (!p || stat(p, &st) != 0 || !S_ISDIR(st.st_mode))
            continue;
        janas_buf_puts(out, k++ ? ", " : "");
        str(out, p);
    }
    janas_buf_puts(out, "]}");
    janas_json_free(d);
    janas_buf_free(&f);
}
