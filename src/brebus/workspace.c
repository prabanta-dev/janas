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
    if (janas_known_dir(0, dir, n) == 0)
        return;
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
    char home[4096];
    const char *h = getenv("HOME");
    return janas_path_root(real) ||
           (h && janas_path_real(h, home) && !strcmp(real, home));
}

static void recent_file(const struct brebus *b, char *p, size_t n)
{
    snprintf(p, n, "%s/workspaces.json", b->data_dir);
}

void brebus_free_path(const struct brebus *b, char *out, size_t n)
{
    snprintf(out, n, "%s/free-chats", b->data_dir);
    janas_path_slashes(out);
}

/* whether path is the free chats' workspace */
static int is_free(const struct brebus *b, const char *path)
{
    char f[4200], real[4096];
    brebus_free_path(b, f, sizeof f);
    return janas_path_real(f, real) && path && !strcmp(path, real);
}

/* the recent ones, newest first, with path put first; never the free
   chats' own */
static void recent_add(struct brebus *b, const char *path)
{
    char rp[4200], err[160];
    if (is_free(b, path))
        return;
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
    char real[4096], sub[4096 + 32];
    if (!janas_path_abs(path)) {
        snprintf(why, wn, "a folder by its full path, as %s",
#ifdef _WIN32
                 "C:/Users/..."
#else
                 "/home/..."
#endif
        );
        return -1;
    }
    if (!brebus_confined(b, path)) {
        snprintf(why, wn, "a project of yours is in your own folder, %s",
                 b->root);
        return -1;
    }
    if (create && janas_mkdirs(path) != 0) {
        snprintf(why, wn, "cannot make %s", path);
        return -1;
    }
    struct stat st;
    if (!janas_path_real(path, real) || stat(real, &st) != 0 ||
        !S_ISDIR(st.st_mode)) {
        snprintf(why, wn, "%s is not a folder", path);
        return -1;
    }
    if (too_wide(real)) {
        snprintf(why, wn,
                 "%s is too wide to work in: choose or make a folder in it",
                 real);
        return -1;
    }
    /* the folder found, checked again: a link put in the path between the
       first look and this one leads nowhere else */
    if (!brebus_confined(b, real)) {
        snprintf(why, wn, "a project of yours is in your own folder, %s",
                 b->root);
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

int brebus_ws_free(struct brebus *b, char *why, size_t wn)
{
    char f[4200];
    brebus_free_path(b, f, sizeof f);
    return brebus_ws_open(b, f, 1, why, wn);
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
    if (b->root[0]) /* a user's Brebus: their own folder */
        snprintf(base, sizeof base, "%s", b->root);
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
    janas_buf_printf(out, ", \"free\": %s, \"given\": %s",
                     on && is_free(b, ws) ? "true" : "false",
                     b->ws_given ? "true" : "false");
    documents(docs, sizeof docs);
    janas_buf_puts(out, ", \"new_in\": ");
    snprintf(rp, sizeof rp, "%s/Brebus", docs);
    str(out, b->root[0] ? b->root : rp);
    janas_buf_printf(out, ", \"window\": %s, \"windows\": %s",
                     brebus_window_on() ? "true" : "false",
#ifdef _WIN32
                     "true"
#else
                     "false"
#endif
    );
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

/* the recent projects' paths, those still there, into p[] */
static int recent_paths(struct brebus *b, char (*p)[4096], int max)
{
    char rp[4200], err[160];
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
         e && k < max; e = e->next) {
        const char *s = janas_json_str(e);
        struct stat st;
        if (!s || stat(s, &st) != 0 || !S_ISDIR(st.st_mode) || is_free(b, s))
            continue;
        snprintf(p[k++], 4096, "%s", s);
    }
    janas_json_free(d);
    janas_buf_free(&f);
    return k;
}

int brebus_ws_known(struct brebus *b, const char *path, char *real, size_t n)
{
    static char p[MAX_RECENT][4096];
    char f[4200], fr[4096];
    if (!path)
        return -1;
    brebus_free_path(b, f, sizeof f);
    if (janas_path_real(f, fr) && !strcmp(path, fr))
        return snprintf(real, n, "%s", fr) > 0 ? 0 : -1;
    int k = recent_paths(b, p, MAX_RECENT);
    for (int i = 0; i < k; i++)
        if (!strcmp(p[i], path))
            return snprintf(real, n, "%s", p[i]) > 0 ? 0 : -1;
    return -1;
}

void brebus_projects(struct brebus *b, struct janas_buf *out)
{
    char f[4200], real[4096];
    static char p[MAX_RECENT][4096];
    brebus_free_path(b, f, sizeof f);
    janas_buf_puts(out, "{\"free\": ");
    int have = janas_path_real(f, real) != NULL;
    str(out, have ? real : f);
    janas_buf_puts(out, ", \"free_chats\": ");
    brebus_chats_list_in(have ? real : NULL, out);
    janas_buf_puts(out, ", \"projects\": [");
    int n = recent_paths(b, p, MAX_RECENT);
    for (int i = 0; i < n; i++) {
        const char *slash = strrchr(p[i], '/');
        janas_buf_puts(out, i ? ", {\"path\": " : "{\"path\": ");
        str(out, p[i]);
        janas_buf_puts(out, ", \"name\": ");
        str(out, slash ? slash + 1 : p[i]);
        janas_buf_puts(out, ", \"chats\": ");
        brebus_chats_list_in(p[i], out);
        janas_buf_puts(out, "}");
    }
    janas_buf_puts(out, "]}");
}

int brebus_chat_move_to(struct brebus *b, const char *body, size_t n, char *why,
                        size_t wn)
{
    char ws[4096], err[160], to[4200], real[4096];
    static char p[MAX_RECENT][4096];
    if (brebus_ws_path(b, ws, sizeof ws) != 0) {
        snprintf(why, wn, "no workspace open");
        return -1;
    }
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *id = o ? janas_json_str(janas_json_get(o, "id")) : NULL;
    const char *t = o ? janas_json_str(janas_json_get(o, "to")) : NULL;
    /* "from": a chat of another project than the one open */
    const char *from = o ? janas_json_str(janas_json_get(o, "from")) : NULL;
    char fr[4096];
    if (from && *from) {
        if (brebus_ws_known(b, from, fr, sizeof fr) != 0) {
            snprintf(why, wn, "not a project of Brebus");
            janas_json_free(d);
            return -1;
        }
        snprintf(ws, sizeof ws, "%s", fr);
    }
    /* only to the free chats or to a project Brebus knows */
    int ok = 0;
    if (t && !*t) {
        brebus_free_path(b, to, sizeof to);
        janas_mkdirs(to);
        ok = janas_path_real(to, real) != NULL;
    } else if (t) {
        int k = recent_paths(b, p, MAX_RECENT);
        for (int i = 0; i < k && !ok; i++)
            if (!strcmp(p[i], t))
                ok = snprintf(real, sizeof real, "%s", t) > 0;
    }
    int rc = -1;
    if (!ok)
        snprintf(why, wn, "not a project of Brebus");
    else if (!strcmp(real, ws))
        snprintf(why, wn, "it is there already");
    else if ((rc = brebus_chat_move(id, ws, real)) != 0)
        snprintf(why, wn, "it could not be moved there");
    janas_json_free(d);
    return rc;
}

/* ---- the conversations of every recent workspace, searched ---- */

#define MAX_HITS 50

struct hit {
    double updated;
    char *json; /* its item of the answer */
};

static int newer(const void *a, const void *b)
{
    const struct hit *x = a, *y = b;
    return x->updated < y->updated ? 1 : x->updated > y->updated ? -1 : 0;
}

/* where the text holds q (q in lower case, ASCII folded): its offset or -1 */
static long find_ci(const char *t, size_t n, const char *q)
{
    size_t ql = strlen(q);
    if (!ql)
        return 0;
    for (size_t i = 0; i + ql <= n; i++) {
        size_t k = 0;
        while (k < ql) {
            char c = t[i + k];
            if (c >= 'A' && c <= 'Z')
                c = (char)(c - 'A' + 'a');
            if (c != q[k])
                break;
            k++;
        }
        if (k == ql)
            return (long)i;
    }
    return -1;
}

static void find_in(const char *ws, const char *q, const struct brebus_find *f,
                    struct hit *hits, int *nh, int free_one)
{
    char dir[4300], err[160];
    snprintf(dir, sizeof dir, "%s/.brebus/chats", ws);
    DIR *dd = opendir(dir);
    struct dirent *de;
    while (dd && *nh < 4096 && (de = readdir(dd))) {
        size_t l = strlen(de->d_name);
        if (l < 6 || strcmp(de->d_name + l - 5, ".json") || l - 5 > 64)
            continue;
        char p[4400], id[72];
        snprintf(id, sizeof id, "%.*s", (int)(l - 5), de->d_name);
        snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
        struct janas_buf c = {0};
        FILE *cf = fopen(p, "rb");
        if (cf) {
            char buf[65536];
            size_t r;
            while ((r = fread(buf, 1, sizeof buf, cf)) > 0 && c.n < (32u << 20))
                janas_buf_put(&c, buf, r);
            fclose(cf);
        }
        struct janas_json_doc *cd =
            c.n ? janas_json_parse(c.p, c.n, err, sizeof err) : NULL;
        const struct janas_json *cr = cd ? janas_json_root(cd) : NULL;
        const char *title =
            cr ? janas_json_str(janas_json_get(cr, "title")) : NULL;
        const struct janas_json *msgs =
            cr ? janas_json_get(cr, "messages") : NULL;
        /* the filters: archived, time, model */
        const struct janas_json *ar =
            cr ? janas_json_get(cr, "archived") : NULL;
        int archived = ar && ar->type == JANAS_JSON_TRUE;
        double upd = cr ? janas_json_num(janas_json_get(cr, "updated"), 0) : 0;
        const char *model =
            cr ? janas_json_str(janas_json_get(cr, "model")) : NULL;
        int keep =
            cr && (f->archived || !archived) && upd >= f->since &&
            (!f->model || !*f->model || (model && !strcmp(model, f->model)));
        /* the title, else the first message holding the words (only
           the user's, or none, as asked) */
        const char *snip = NULL;
        size_t sn = 0;
        const char *in = f->in ? f->in : "all";
        long at = keep && title ? find_ci(title, strlen(title), q) : -1;
        if (keep && !*q)
            at = 0; /* no words: every chat the filters keep */
        for (const struct janas_json *m =
                 keep && at < 0 && strcmp(in, "titles") && msgs &&
                         msgs->type == JANAS_JSON_ARRAY
                     ? msgs->child
                     : NULL;
             m && at < 0; m = m->next) {
            const char *role = janas_json_str(janas_json_get(m, "role"));
            if (!strcmp(in, "mine") && (!role || strcmp(role, "user")))
                continue;
            const struct janas_json *t = janas_json_get(m, "content");
            if (t && t->type == JANAS_JSON_STRING &&
                (at = find_ci(t->s, t->n, q)) >= 0) {
                size_t from = at > 60 ? (size_t)at - 60 : 0;
                snip = t->s + from;
                sn = t->n - from < 160 ? t->n - from : 160;
            }
        }
        if (cd && at >= 0) {
            struct janas_buf o = {0};
            const char *slash = strrchr(ws, '/');
            janas_buf_puts(&o, "{\"workspace\": ");
            janas_json_write_str(&o, ws, strlen(ws));
            janas_buf_puts(&o, ", \"workspace_name\": ");
            janas_json_write_str(&o, slash ? slash + 1 : ws,
                                 strlen(slash ? slash + 1 : ws));
            janas_buf_puts(&o, ", \"id\": ");
            janas_json_write_str(&o, id, strlen(id));
            janas_buf_puts(&o, ", \"title\": ");
            janas_json_write_str(&o, title ? title : "",
                                 title ? strlen(title) : 0);
            double up = upd;
            janas_buf_puts(&o, ", \"model\": ");
            janas_json_write_str(&o, model ? model : "",
                                 model ? strlen(model) : 0);
            janas_buf_printf(&o,
                             ", \"archived\": %s, \"free\": %s, "
                             "\"updated\": %.0f, \"snippet\": ",
                             archived ? "true" : "false",
                             free_one ? "true" : "false", up);
            janas_json_write_str(&o, snip ? snip : "", sn);
            janas_buf_puts(&o, "}");
            janas_buf_put(&o, "", 1);
            if (!o.oom) {
                hits[*nh].updated = up;
                hits[(*nh)++].json = o.p;
            } else
                janas_buf_free(&o);
        }
        janas_json_free(cd);
        janas_buf_free(&c);
    }
    if (dd)
        closedir(dd);
}

void brebus_chats_search(struct brebus *b, const char *query,
                         struct janas_buf *out)
{
    static const struct brebus_find all = {.in = "all", .archived = 1};
    brebus_chats_find(b, query, &all, out);
}

void brebus_chats_find(struct brebus *b, const char *query,
                       const struct brebus_find *f, struct janas_buf *out)
{
    char q[256], rp[4200], err[160], fp[4200], freal[4096];
    size_t k = 0;
    for (const char *c = query ? query : ""; *c && k + 1 < sizeof q; c++)
        q[k++] = (char)(*c >= 'A' && *c <= 'Z' ? *c - 'A' + 'a' : *c);
    q[k] = 0;
    static struct hit hits[4096];
    int nh = 0;
    /* the recent workspaces, from workspaces.json */
    recent_file(b, rp, sizeof rp);
    struct janas_buf fb = {0};
    FILE *in = fopen(rp, "rb");
    if (in) {
        char buf[16384];
        size_t r;
        while ((r = fread(buf, 1, sizeof buf, in)) > 0 && fb.n < (1u << 20))
            janas_buf_put(&fb, buf, r);
        fclose(in);
    }
    struct janas_json_doc *d =
        fb.n ? janas_json_parse(fb.p, fb.n, err, sizeof err) : NULL;
    const struct janas_json *list =
        d ? janas_json_get(janas_json_root(d), "recent") : NULL;
    /* the free chats first, then the recent projects */
    brebus_free_path(b, fp, sizeof fp);
    const char *where = f && f->where ? f->where : "";
    if (janas_path_real(fp, freal) && (!*where || !strcmp(where, "free")))
        find_in(freal, q, f, hits, &nh, 1);
    for (const struct janas_json *e =
             list && list->type == JANAS_JSON_ARRAY ? list->child : NULL;
         e && nh < 4096; e = e->next) {
        const char *ws = janas_json_str(e);
        if (ws && (!*where || !strcmp(where, ws)))
            find_in(ws, q, f, hits, &nh, 0);
    }
    janas_json_free(d);
    janas_buf_free(&fb);
    qsort(hits, (size_t)nh, sizeof *hits, newer);
    janas_buf_puts(out, "{\"chats\": [");
    for (int i = 0; i < nh; i++) {
        if (i < MAX_HITS) {
            janas_buf_puts(out, i ? ", " : "");
            janas_buf_puts(out, hits[i].json);
        }
        free(hits[i].json);
    }
    janas_buf_printf(out, "], \"more\": %d}",
                     nh > MAX_HITS ? nh - MAX_HITS : 0);
}
