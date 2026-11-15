/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * store.c - what Brebus keeps on disk: the page's conversations, in the
 * workspace (workspace.c), one JSON file each in .brebus/chats/ (named by the
 * page's id of the conversation, letters, digits, - and _ only), and the
 * settings of each model in settings.json, in Brebus's data folder. Brebus
 * checks that what it is given is a JSON object, and writes it aside first,
 * then renames it: a file is never left half written. A conversation exported
 * goes to the workspace's exports/ (with none, the downloads folder:
 * XDG_DOWNLOAD_DIR of user-dirs.dirs, else ~/Downloads).
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"

#define MAX_FILE (32u << 20)
#define MAX_CHATS 4096

static int good_id(const char *id)
{
    size_t n = strlen(id);
    if (n < 1 || n > 64)
        return 0;
    for (size_t i = 0; i < n; i++) {
        char c = id[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '-' || c == '_'))
            return 0;
    }
    return 1;
}

/* the whole file into out (NUL-terminated); 0, or -1 */
static int slurp(const char *path, struct janas_buf *out)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    char buf[65536];
    size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0 && out->n < MAX_FILE)
        janas_buf_put(out, buf, r);
    int bad = ferror(f) || out->n >= MAX_FILE || out->oom;
    fclose(f);
    janas_buf_put(out, "", 1);
    out->n--;
    return bad ? -1 : 0;
}

/* a JSON object, written aside and renamed over path */
static int put(const char *path, const char *body, size_t n)
{
    char err[160];
    struct janas_json_doc *d = janas_json_parse(body, n, err, sizeof err);
    int object = d && janas_json_root(d)->type == JANAS_JSON_OBJECT;
    janas_json_free(d);
    if (!object)
        return -1;
    char tmp[4300];
    snprintf(tmp, sizeof tmp, "%s.new", path);
    FILE *f = fopen(tmp, "wb");
    if (!f)
        return -1;
    int bad = fwrite(body, 1, n, f) != n;
    bad |= fclose(f) != 0;
    if (bad || rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

/* the workspace's conversations' folder: 0, or -1 with no workspace */
static int chats_dir(struct brebus *b, char *dir, size_t n)
{
    char ws[4096];
    if (brebus_ws_path(b, ws, sizeof ws) != 0)
        return -1;
    snprintf(dir, n, "%s/.brebus/chats", ws);
    return 0;
}

static int chat_path(struct brebus *b, const char *id, char *p, size_t n)
{
    char dir[4200];
    if (chats_dir(b, dir, sizeof dir) != 0)
        return -1;
    snprintf(p, n, "%s/%s.json", dir, id);
    return 0;
}

struct entry {
    char id[72];
    double updated;
    char *line; /* its item of the list, as JSON */
};

static int newest_first(const void *x, const void *y)
{
    const struct entry *a = x, *c = y;
    return a->updated < c->updated ? 1 : a->updated > c->updated ? -1 : 0;
}

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

void brebus_chats_list(struct brebus *b, struct janas_buf *out)
{
    char dir[4200];
    struct entry *e = calloc(MAX_CHATS, sizeof *e);
    int n = 0;
    DIR *d = e && chats_dir(b, dir, sizeof dir) == 0 ? opendir(dir) : NULL;
    struct dirent *de;
    while (d && n < MAX_CHATS && (de = readdir(d))) {
        size_t len = strlen(de->d_name);
        if (len < 6 || len - 5 >= sizeof e[n].id ||
            strcmp(de->d_name + len - 5, ".json"))
            continue;
        memcpy(e[n].id, de->d_name, len - 5);
        e[n].id[len - 5] = 0;
        if (!good_id(e[n].id))
            continue;
        char p[4400], err[160];
        struct janas_buf f = {0};
        if (chat_path(b, e[n].id, p, sizeof p) != 0)
            continue;
        struct janas_json_doc *doc =
            slurp(p, &f) == 0 ? janas_json_parse(f.p, f.n, err, sizeof err)
                              : NULL;
        janas_buf_free(&f);
        if (!doc)
            continue;
        const struct janas_json *r = janas_json_root(doc);
        const char *title = janas_json_str(janas_json_get(r, "title"));
        const char *model = janas_json_str(janas_json_get(r, "model"));
        const struct janas_json *msgs = janas_json_get(r, "messages");
        e[n].updated = janas_json_num(janas_json_get(r, "updated"), 0);
        struct janas_buf o = {0};
        janas_buf_puts(&o, "{\"id\": ");
        str(&o, e[n].id);
        janas_buf_puts(&o, ", \"title\": ");
        str(&o, title ? title : "");
        janas_buf_puts(&o, ", \"model\": ");
        str(&o, model ? model : "");
        janas_buf_printf(&o, ", \"updated\": %.0f, \"messages\": %zu}",
                         e[n].updated,
                         msgs && msgs->type == JANAS_JSON_ARRAY ? msgs->n : 0);
        janas_buf_put(&o, "", 1);
        janas_json_free(doc);
        if (o.oom) {
            janas_buf_free(&o);
            continue;
        }
        e[n++].line = o.p;
    }
    if (d)
        closedir(d);
    if (n)
        qsort(e, (size_t)n, sizeof *e, newest_first);
    janas_buf_puts(out, "{\"chats\": [");
    for (int i = 0; i < n; i++) {
        janas_buf_puts(out, i ? ", " : "");
        janas_buf_puts(out, e[i].line);
        free(e[i].line);
    }
    janas_buf_puts(out, "]}");
    free(e);
}

int brebus_chat_get(struct brebus *b, const char *id, struct janas_buf *out)
{
    char p[4400];
    if (!good_id(id) || chat_path(b, id, p, sizeof p) != 0)
        return -1;
    return slurp(p, out);
}

int brebus_chat_put(struct brebus *b, const char *id, const char *body,
                    size_t n)
{
    char dir[4200], p[4400];
    if (!good_id(id) || chats_dir(b, dir, sizeof dir) != 0)
        return -1;
    janas_mkdirs(dir);
    if (chat_path(b, id, p, sizeof p) != 0)
        return -1;
    return put(p, body, n);
}

int brebus_chat_delete(struct brebus *b, const char *id)
{
    char p[4400];
    if (!good_id(id) || chat_path(b, id, p, sizeof p) != 0)
        return -1;
    return remove(p);
}

void brebus_settings_get(struct brebus *b, struct janas_buf *out)
{
    char p[4200];
    snprintf(p, sizeof p, "%s/settings.json", b->data_dir);
    struct janas_buf f = {0};
    if (slurp(p, &f) == 0 && f.n)
        janas_buf_put(out, f.p, f.n);
    else
        janas_buf_puts(out, "{}");
    janas_buf_free(&f);
}

int brebus_settings_put(struct brebus *b, const char *body, size_t n)
{
    char p[4200];
    snprintf(p, sizeof p, "%s/settings.json", b->data_dir);
    return put(p, body, n);
}

/* the downloads folder: user-dirs.dirs's, ~/Downloads, or the home */
static void downloads(char *dir, size_t n)
{
    const char *home = getenv("HOME"), *xc = getenv("XDG_CONFIG_HOME");
    char conf[4200], line[4200];
    if (!home)
        home = ".";
    if (xc && *xc == '/')
        snprintf(conf, sizeof conf, "%s/user-dirs.dirs", xc);
    else
        snprintf(conf, sizeof conf, "%s/.config/user-dirs.dirs", home);
    dir[0] = 0;
    FILE *f = fopen(conf, "r");
    while (f && fgets(line, sizeof line, f)) {
        static const char key[] = "XDG_DOWNLOAD_DIR=\"";
        if (strncmp(line, key, sizeof key - 1))
            continue;
        char *v = line + sizeof key - 1, *q = strchr(v, '"');
        if (!q)
            break;
        *q = 0;
        if (!strncmp(v, "$HOME/", 6))
            snprintf(dir, n, "%s/%s", home, v + 6);
        else if (*v == '/')
            snprintf(dir, n, "%s", v);
        break;
    }
    if (f)
        fclose(f);
    struct stat st;
    if (dir[0] && stat(dir, &st) == 0 && S_ISDIR(st.st_mode))
        return;
    snprintf(dir, n, "%s/Downloads", home);
    if (stat(dir, &st) == 0 && S_ISDIR(st.st_mode))
        return;
    snprintf(dir, n, "%s", home);
}

int brebus_export(struct brebus *b, const char *body, size_t n,
                  struct janas_buf *out)
{
    char err[160], name[96], dir[4200], path[4400];
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *nm = o ? janas_json_str(janas_json_get(o, "name")) : NULL;
    const struct janas_json *text = o ? janas_json_get(o, "text") : NULL;
    if (!text || text->type != JANAS_JSON_STRING) {
        janas_json_free(d);
        janas_buf_puts(out, "{\"error\": \"nothing to save\"}");
        return -1;
    }
    /* letters, digits, spaces, - and _ of the name (bytes past ASCII too,
       for names in other scripts), never a path */
    size_t k = 0;
    for (const unsigned char *c = (const unsigned char *)(nm ? nm : "");
         *c && k + 1 < 64; c++)
        if ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
            (*c >= '0' && *c <= '9') || *c == ' ' || *c == '-' || *c == '_' ||
            *c >= 0x80)
            name[k++] = (char)*c;
    while (k && name[k - 1] == ' ')
        k--;
    name[k] = 0;
    if (!k)
        snprintf(name, sizeof name, "chat");
    /* the workspace's exports/, or the downloads folder with none */
    char ws[4096];
    if (brebus_ws_path(b, ws, sizeof ws) == 0) {
        snprintf(dir, sizeof dir, "%s/exports", ws);
        janas_mkdirs(dir);
    } else
        downloads(dir, sizeof dir);
    int fd = -1;
    for (int i = 1; i < 100 && fd < 0; i++) {
        if (i == 1)
            snprintf(path, sizeof path, "%s/%s.md", dir, name);
        else
            snprintf(path, sizeof path, "%s/%s (%d).md", dir, name, i);
        fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (fd < 0 && errno != EEXIST)
            break;
    }
    int bad = fd < 0;
    if (!bad) {
        bad = write(fd, text->s, text->n) != (ssize_t)text->n;
        bad |= close(fd) != 0;
        if (bad)
            remove(path);
    }
    janas_json_free(d);
    if (bad) {
        janas_buf_puts(out, "{\"error\": \"cannot write in ");
        janas_buf_puts(out, "the downloads folder\"}");
        return -1;
    }
    janas_buf_puts(out, "{\"path\": ");
    str(out, path);
    janas_buf_puts(out, "}");
    return 0;
}
