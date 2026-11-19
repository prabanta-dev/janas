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
#include <strings.h>
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
    char ws[4096];
    brebus_chats_list_in(brebus_ws_path(b, ws, sizeof ws) == 0 ? ws : NULL,
                         out);
}

/* the chats of the workspace (a project, or the free chats' own) ws */
void brebus_chats_list_in(const char *ws, struct janas_buf *out)
{
    char dir[4200];
    struct entry *e = calloc(MAX_CHATS, sizeof *e);
    int n = 0;
    if (ws)
        snprintf(dir, sizeof dir, "%s/.brebus/chats", ws);
    DIR *d = e && ws ? opendir(dir) : NULL;
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
        snprintf(p, sizeof p, "%s/%s.json", dir, e[n].id);
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
        const char *folder = janas_json_str(janas_json_get(r, "folder"));
        janas_buf_puts(&o, ", \"folder\": ");
        str(&o, folder ? folder : "");
        const struct janas_json *arch = janas_json_get(r, "archived");
        janas_buf_printf(
            &o,
            ", \"updated\": %.0f, \"messages\": %zu, "
            "\"archived\": %s}",
            e[n].updated, msgs && msgs->type == JANAS_JSON_ARRAY ? msgs->n : 0,
            arch && arch->type == JANAS_JSON_TRUE ? "true" : "false");
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

int brebus_chat_move(const char *id, const char *from, const char *to)
{
    char a[4400], c[4400], dir[4300];
    if (!good_id(id))
        return -1;
    snprintf(a, sizeof a, "%s/.brebus/chats/%s.json", from, id);
    snprintf(dir, sizeof dir, "%s/.brebus/chats", to);
    snprintf(c, sizeof c, "%s/%s.json", dir, id);
    if (janas_mkdirs(dir) != 0 || access(c, F_OK) == 0)
        return -1;
    /* Windows renames no file another reader holds open - the side bar's
       list reading it, an antivirus's scan of a file just written: a few
       tries, a little apart (tried on Windows 11, 17 Nov 2026) */
    for (int i = 0; i < 20; i++) {
        if (rename(a, c) == 0)
            return 0;
        if (access(a, F_OK) != 0)
            return -1;
        usleep(50000);
    }
    return -1;
}

int brebus_chat_get_in(const char *ws, const char *id, struct janas_buf *out)
{
    char p[4400];
    if (!good_id(id))
        return -1;
    snprintf(p, sizeof p, "%s/.brebus/chats/%s.json", ws, id);
    return slurp(p, out);
}

int brebus_chat_put_in(const char *ws, const char *id, const char *body,
                       size_t n)
{
    char p[4400];
    if (!good_id(id))
        return -1;
    snprintf(p, sizeof p, "%s/.brebus/chats/%s.json", ws, id);
    return access(p, F_OK) == 0 ? put(p, body, n) : -1; /* only one there */
}

int brebus_chat_delete_in(const char *ws, const char *id)
{
    char p[4400];
    if (!good_id(id))
        return -1;
    snprintf(p, sizeof p, "%s/.brebus/chats/%s.json", ws, id);
    return remove(p);
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
    if (janas_known_dir(1, dir, n) == 0)
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
    const char *e = o ? janas_json_str(janas_json_get(o, "ext")) : NULL;
    const char *ext = e && !strcmp(e, "txt") ? "txt" : "md";
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
            snprintf(path, sizeof path, "%s/%s.%s", dir, name, ext);
        else
            snprintf(path, sizeof path, "%s/%s (%d).%s", dir, name, i, ext);
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

/* ---- artifacts: what the model makes to keep, in the workspace ---- */

/* an artifact's folder, by its title made safe */
static void art_name(const char *title, char *name, size_t cap)
{
    size_t k = 0;
    for (const unsigned char *p = (const unsigned char *)(title ? title : "");
         *p && k + 1 < cap && k < 63; p++)
        if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
            (*p >= '0' && *p <= '9') || *p == ' ' || *p == '-' || *p == '_' ||
            *p >= 0x80)
            name[k++] = (char)*p;
    while (k && name[k - 1] == ' ')
        k--;
    name[k] = 0;
    if (!k)
        snprintf(name, cap, "artifact");
}

int brebus_artifact_rename(struct brebus *b, const char *body, size_t n,
                           struct janas_buf *out)
{
    char ws[4096], err[160], from[96], to[96], a[4400], c[4400];
    if (brebus_ws_path(b, ws, sizeof ws) != 0) {
        janas_buf_puts(out, "{\"error\": \"no workspace\"}");
        return -1;
    }
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    art_name(o ? janas_json_str(janas_json_get(o, "from")) : NULL, from,
             sizeof from);
    art_name(o ? janas_json_str(janas_json_get(o, "to")) : NULL, to, sizeof to);
    janas_json_free(d);
    snprintf(a, sizeof a, "%s/artifact/%s", ws, from);
    snprintf(c, sizeof c, "%s/artifact/%s", ws, to);
    struct stat st;
    /* its versions' folder follows the name; one of that name already
       there is never written over */
    if (!strcmp(from, to) || stat(a, &st) != 0) {
        janas_buf_puts(out, "{\"ok\": true}");
        return 0;
    }
    if (stat(c, &st) == 0 || rename(a, c) != 0) {
        janas_buf_puts(out, "{\"error\": \"an artifact of that name is there "
                            "already\"}");
        return -1;
    }
    janas_buf_puts(out, "{\"ok\": true}");
    return 0;
}

int brebus_artifact_save(struct brebus *b, const char *body, size_t n,
                         struct janas_buf *out)
{
    char ws[4096], err[160], dir[4400], path[4700];
    if (brebus_ws_path(b, ws, sizeof ws) != 0) {
        janas_buf_puts(out, "{\"error\": \"no workspace\"}");
        return -1;
    }
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *title = o ? janas_json_str(janas_json_get(o, "title")) : NULL;
    const char *type = o ? janas_json_str(janas_json_get(o, "type")) : NULL;
    const char *lang = o ? janas_json_str(janas_json_get(o, "lang")) : NULL;
    const struct janas_json *c = o ? janas_json_get(o, "content") : NULL;
    if (!c || c->type != JANAS_JSON_STRING || c->n > (16u << 20)) {
        janas_json_free(d);
        janas_buf_puts(out, "{\"error\": \"no content\"}");
        return -1;
    }
    /* its folder by its title, made safe; the extension by its type */
    char name[96];
    art_name(title, name, sizeof name);
    static const struct {
        const char *what, *ext;
    } EXT[] = {{"html", "html"}, {"svg", "svg"},       {"markdown", "md"},
               {"python", "py"}, {"javascript", "js"}, {"js", "js"},
               {"c", "c"},       {"cpp", "cpp"},       {"bash", "sh"},
               {"sh", "sh"},     {"json", "json"},     {"css", "css"},
               {"sql", "sql"},   {"rust", "rs"},       {"go", "go"},
               {"java", "java"}, {"typescript", "ts"}, {"csv", "csv"},
               {"react", "jsx"}, {"mermaid", "mmd"}};
    const char *ext = "txt";
    const char *what = type && strcmp(type, "code") ? type : lang;
    for (size_t i = 0; what && i < sizeof EXT / sizeof *EXT; i++)
        if (!strcasecmp(what, EXT[i].what))
            ext = EXT[i].ext;
    snprintf(dir, sizeof dir, "%s/artifact/%s", ws, name);
    if (janas_mkdirs(dir) != 0) {
        janas_json_free(d);
        janas_buf_puts(out, "{\"error\": \"cannot make artifact/\"}");
        return -1;
    }
    /* the next version; the same content as the last one is not saved
       again */
    int v = 1;
    for (;; v++) {
        snprintf(path, sizeof path, "%s/v%d.%s", dir, v, ext);
        if (access(path, F_OK) != 0)
            break;
    }
    if (v > 1) {
        char last[4700];
        snprintf(last, sizeof last, "%s/v%d.%s", dir, v - 1, ext);
        struct janas_buf old = {0};
        if (slurp(last, &old) == 0 && old.n == c->n &&
            !memcmp(old.p, c->s, c->n)) {
            janas_buf_free(&old);
            janas_json_free(d);
            janas_buf_puts(out, "{\"path\": ");
            str(out, last);
            janas_buf_printf(out, ", \"version\": %d, \"same\": true}", v - 1);
            return 0;
        }
        janas_buf_free(&old);
    }
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    int bad = fd < 0;
    if (!bad) {
        bad = write(fd, c->s, c->n) != (ssize_t)c->n;
        bad |= close(fd) != 0;
    }
    janas_json_free(d);
    if (bad) {
        remove(path);
        janas_buf_puts(out, "{\"error\": \"cannot write the artifact\"}");
        return -1;
    }
    janas_buf_puts(out, "{\"path\": ");
    str(out, path);
    janas_buf_printf(out, ", \"version\": %d}", v);
    return 0;
}

/* window.storage of an artifact: its data, a JSON object in
   artifact/<title>/storage.json; {"title", "act": get, set, delete or
   list, "key", "value", "prefix"}, answered as Claude's storage answers */
#define STORE_MAX (20u << 20)
int brebus_artifact_store(struct brebus *b, const char *body, size_t n,
                          struct janas_buf *out)
{
    char ws[4096], err[160], name[96], dir[4400], path[4500];
    if (brebus_ws_path(b, ws, sizeof ws) != 0) {
        janas_buf_puts(out, "{\"error\": \"no workspace\"}");
        return -1;
    }
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *title = o ? janas_json_str(janas_json_get(o, "title")) : NULL;
    const char *act = o ? janas_json_str(janas_json_get(o, "act")) : NULL;
    const char *key = o ? janas_json_str(janas_json_get(o, "key")) : NULL;
    const char *prefix = o ? janas_json_str(janas_json_get(o, "prefix")) : NULL;
    const struct janas_json *value = o ? janas_json_get(o, "value") : NULL;
    int set = act && !strcmp(act, "set"), del = act && !strcmp(act, "delete");
    int get = act && !strcmp(act, "get"), list = act && !strcmp(act, "list");
    if (!title || !(set || del || get || list) ||
        (!list && (!key || !*key || strlen(key) > 200)) || (set && !value)) {
        janas_json_free(d);
        janas_buf_puts(out, "{\"error\": \"title, act and key\"}");
        return -1;
    }
    art_name(title, name, sizeof name);
    snprintf(dir, sizeof dir, "%s/artifact/%s", ws, name);
    snprintf(path, sizeof path, "%s/storage.json", dir);
    struct janas_buf f = {0}, w = {0};
    struct janas_json_doc *sd = NULL;
    if (slurp(path, &f) == 0 && f.n)
        sd = janas_json_parse(f.p, f.n, err, sizeof err);
    const struct janas_json *root = sd ? janas_json_root(sd) : NULL;
    if (root && root->type != JANAS_JSON_OBJECT)
        root = NULL;
    int rc = 0;
    if (get || list) {
        const struct janas_json *v = NULL;
        if (list)
            janas_buf_puts(out, "{\"keys\": [");
        int k = 0;
        for (const struct janas_json *e = root ? root->child : NULL; e;
             e = e->next) {
            if (get && e->key_n == strlen(key) &&
                !memcmp(e->key, key, e->key_n))
                v = e;
            if (list &&
                (!prefix || (e->key_n >= strlen(prefix) &&
                             !memcmp(e->key, prefix, strlen(prefix))))) {
                janas_buf_puts(out, k++ ? ", " : "");
                janas_json_write_str(out, e->key, e->key_n);
            }
        }
        if (list)
            janas_buf_puts(out, "]}");
        else if (!v)
            janas_buf_puts(out, "null");
        else {
            janas_buf_puts(out, "{\"key\": ");
            str(out, key);
            janas_buf_puts(out, ", \"value\": ");
            janas_json_write(out, v);
            janas_buf_puts(out, "}");
        }
    } else {
        /* the object written again, the key changed or left out */
        janas_buf_puts(&w, "{");
        int k = 0;
        for (const struct janas_json *e = root ? root->child : NULL; e;
             e = e->next) {
            if (e->key_n == strlen(key) && !memcmp(e->key, key, e->key_n))
                continue;
            janas_buf_puts(&w, k++ ? ", " : "");
            janas_json_write_str(&w, e->key, e->key_n);
            janas_buf_puts(&w, ": ");
            janas_json_write(&w, e);
        }
        if (set) {
            janas_buf_puts(&w, k++ ? ", " : "");
            str(&w, key);
            janas_buf_puts(&w, ": ");
            janas_json_write(&w, value);
        }
        janas_buf_puts(&w, "}");
        if (w.oom || w.n > STORE_MAX) {
            janas_buf_puts(out, "{\"error\": \"this artifact's storage is full "
                                "(20 MB)\"}");
            rc = -1;
        } else {
            char tmp[4600];
            snprintf(tmp, sizeof tmp, "%s.part", path);
            FILE *fp = janas_mkdirs(dir) == 0 ? fopen(tmp, "wb") : NULL;
            int bad = !fp || fwrite(w.p, 1, w.n, fp) != w.n;
            if (fp)
                bad |= fclose(fp) != 0;
            if (bad || rename(tmp, path) != 0) {
                remove(tmp);
                janas_buf_puts(out, "{\"error\": \"cannot write the "
                                    "artifact's storage\"}");
                rc = -1;
            } else if (set) {
                janas_buf_puts(out, "{\"key\": ");
                str(out, key);
                janas_buf_puts(out, ", \"value\": ");
                janas_json_write(out, value);
                janas_buf_puts(out, "}");
            } else {
                janas_buf_puts(out, "{\"key\": ");
                str(out, key);
                janas_buf_puts(out, ", \"deleted\": true}");
            }
        }
    }
    janas_json_free(sd);
    janas_json_free(d);
    janas_buf_free(&f);
    janas_buf_free(&w);
    return rc;
}
