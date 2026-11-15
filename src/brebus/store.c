/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * store.c - what Brebus keeps on disk, in its data folder: the page's
 * conversations, one JSON file each in chats/ (named by the page's id of
 * the conversation, letters, digits, - and _ only), and the settings of
 * each model in settings.json. Brebus checks that what it is given is a
 * JSON object, and writes it aside first, then renames it: a file is
 * never left half written.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void chat_path(const struct brebus *b, const char *id, char *p, size_t n)
{
    snprintf(p, n, "%s/chats/%s.json", b->data_dir, id);
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
    snprintf(dir, sizeof dir, "%s/chats", b->data_dir);
    struct entry *e = calloc(MAX_CHATS, sizeof *e);
    int n = 0;
    DIR *d = e ? opendir(dir) : NULL;
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
        chat_path(b, e[n].id, p, sizeof p);
        struct janas_buf f = {0};
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
    if (!good_id(id))
        return -1;
    chat_path(b, id, p, sizeof p);
    return slurp(p, out);
}

int brebus_chat_put(struct brebus *b, const char *id, const char *body,
                    size_t n)
{
    char dir[4200], p[4400];
    if (!good_id(id))
        return -1;
    snprintf(dir, sizeof dir, "%s/chats", b->data_dir);
    janas_mkdirs(dir);
    chat_path(b, id, p, sizeof p);
    return put(p, body, n);
}

int brebus_chat_delete(struct brebus *b, const char *id)
{
    char p[4400];
    if (!good_id(id))
        return -1;
    chat_path(b, id, p, sizeof p);
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
