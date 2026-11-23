/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * store_log.c - the history of an artifact's saved data (window.storage):
 * beside its state, storage-<chat>.json, each change written in
 * storage-<chat>.log, one JSON line each, with the point of the chat it was
 * made at (how many messages the chat had):
 *   {"at": 7, "k": "key", "v": value}   set
 *   {"at": 7, "k": "key", "d": true}    deleted
 *   {"at": 0, "base": {...}}            a state to start from
 * A branch from a point, or a rewind to it, takes the lines up to there: the
 * data as they were then. A state saved before there was a history becomes
 * the log's base. Past 20 MB the log is folded into one base line, the
 * state of now: the history before that is gone, and a point earlier than
 * it gets that state, the oldest there is.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "brebus.h"
#include "llm/json.h"

#define LOG_MAX (20u << 20)

/* the keys and values of a state, in their order */
struct kv {
    char *k, *v; /* v: the value as JSON text */
    size_t kn, vn;
};
struct state {
    struct kv *e;
    size_t n, cap;
    int oom;
};

static void st_free(struct state *s)
{
    for (size_t i = 0; i < s->n; i++)
        free(s->e[i].k), free(s->e[i].v);
    free(s->e);
    memset(s, 0, sizeof *s);
}

static void st_del(struct state *s, const char *k, size_t kn)
{
    for (size_t i = 0; i < s->n; i++)
        if (s->e[i].kn == kn && !memcmp(s->e[i].k, k, kn)) {
            free(s->e[i].k), free(s->e[i].v);
            memmove(&s->e[i], &s->e[i + 1], (s->n - i - 1) * sizeof *s->e);
            s->n--;
            return;
        }
}

static void st_set(struct state *s, const char *k, size_t kn,
                   const struct janas_json *v)
{
    struct janas_buf t = {0};
    janas_json_write(&t, v);
    char *kc = malloc(kn + 1);
    if (t.oom || !kc) {
        free(kc);
        janas_buf_free(&t);
        s->oom = 1;
        return;
    }
    memcpy(kc, k, kn);
    kc[kn] = 0;
    for (size_t i = 0; i < s->n; i++)
        if (s->e[i].kn == kn && !memcmp(s->e[i].k, k, kn)) {
            free(s->e[i].v), free(kc);
            s->e[i].v = t.p, s->e[i].vn = t.n;
            return;
        }
    if (s->n == s->cap) {
        size_t c = s->cap ? s->cap * 2 : 16;
        struct kv *e = realloc(s->e, c * sizeof *e);
        if (!e) {
            free(kc);
            janas_buf_free(&t);
            s->oom = 1;
            return;
        }
        s->e = e, s->cap = c;
    }
    s->e[s->n++] = (struct kv){.k = kc, .kn = kn, .v = t.p, .vn = t.n};
}

static void st_write(const struct state *s, struct janas_buf *o)
{
    janas_buf_puts(o, "{");
    for (size_t i = 0; i < s->n; i++) {
        janas_buf_puts(o, i ? ", " : "");
        janas_json_write_str(o, s->e[i].k, s->e[i].kn);
        janas_buf_puts(o, ": ");
        janas_buf_put(o, s->e[i].v, s->e[i].vn);
    }
    janas_buf_puts(o, "}");
}

/* a line of the log applied to s */
static void apply(struct state *s, const struct janas_json *l)
{
    const struct janas_json *base = janas_json_get(l, "base");
    if (base && base->type == JANAS_JSON_OBJECT) {
        st_free(s);
        for (const struct janas_json *e = base->child; e; e = e->next)
            st_set(s, e->key, e->key_n, e);
        return;
    }
    const struct janas_json *k = janas_json_get(l, "k");
    if (!k || k->type != JANAS_JSON_STRING)
        return;
    const struct janas_json *v = janas_json_get(l, "v");
    if (v)
        st_set(s, k->s, k->n, v);
    else
        st_del(s, k->s, k->n);
}

static int read_all(const char *path, struct janas_buf *o)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    char buf[65536];
    size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0 && o->n <= 2 * LOG_MAX)
        janas_buf_put(o, buf, r);
    int bad = ferror(f) || o->n > 2 * LOG_MAX || o->oom;
    fclose(f);
    return bad ? -1 : 0;
}

/* data written aside, then renamed over path */
static int put_file(const char *path, const char *p, size_t n)
{
    char tmp[4700];
    snprintf(tmp, sizeof tmp, "%s.part", path);
    FILE *f = fopen(tmp, "wb");
    int bad = !f || (n && fwrite(p, 1, n, f) != n);
    if (f)
        bad |= fclose(f) != 0;
    if (bad || rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

static void paths(const char *dir, const char *chat, char *state, char *log,
                  size_t cap)
{
    snprintf(state, cap, "%s/storage-%s.json", dir, chat);
    snprintf(log, cap, "%s/storage-%s.log", dir, chat);
}

int brebus_store_log(const char *dir, const char *chat, const char *before,
                     size_t before_n, long at, const char *key,
                     const struct janas_json *value)
{
    char state[4600], log[4600];
    paths(dir, chat, state, log, sizeof state);
    struct stat ls;
    int have = stat(log, &ls) == 0;
    struct janas_buf o = {0};
    if (!have && before_n > 2) /* data of before the history: its base */
        janas_buf_printf(&o, "{\"at\": 0, \"base\": %.*s}\n", (int)before_n,
                         before);
    janas_buf_printf(&o, "{\"at\": %ld, \"k\": ", at < 0 ? 0 : at);
    janas_json_write_str(&o, key, strlen(key));
    if (value) {
        janas_buf_puts(&o, ", \"v\": ");
        janas_json_write(&o, value);
        janas_buf_puts(&o, "}\n");
    } else
        janas_buf_puts(&o, ", \"d\": true}\n");
    int rc = -1;
    if (!o.oom && (!have || (size_t)ls.st_size + o.n <= LOG_MAX)) {
        FILE *f = fopen(log, "ab");
        if (f) {
            rc = fwrite(o.p, 1, o.n, f) == o.n ? 0 : -1;
            rc |= fclose(f) != 0 ? -1 : 0;
        }
    } else if (!o.oom) {
        /* full: folded into the state of now */
        struct janas_buf s = {0}, b = {0};
        if (read_all(state, &s) == 0) {
            janas_buf_printf(&b, "{\"at\": %ld, \"base\": %.*s}\n",
                             at < 0 ? 0 : at, (int)s.n, s.p ? s.p : "{}");
            rc = b.oom ? -1 : put_file(log, b.p, b.n);
        }
        janas_buf_free(&s);
        janas_buf_free(&b);
    }
    janas_buf_free(&o);
    return rc;
}

int brebus_store_upto(const char *dir, const char *from, const char *to,
                      long upto)
{
    char fstate[4600], flog[4600], tstate[4600], tlog[4600];
    paths(dir, from, fstate, flog, sizeof fstate);
    paths(dir, to, tstate, tlog, sizeof tstate);
    struct janas_buf in = {0}, keep = {0}, out = {0};
    int rc = -1;
    if (read_all(flog, &in) != 0) {
        janas_buf_free(&in);
        /* no history: a branch takes the state of now; a rewind leaves it */
        if (!strcmp(from, to) || read_all(fstate, &in) != 0 || !in.n) {
            janas_buf_free(&in);
            return strcmp(from, to) ? -1 : 0;
        }
        janas_buf_printf(&keep, "{\"at\": 0, \"base\": %.*s}\n", (int)in.n,
                         in.p);
        rc = keep.oom || put_file(tlog, keep.p, keep.n) ||
                     put_file(tstate, in.p, in.n)
                 ? -1
                 : 0;
        janas_buf_free(&in);
        janas_buf_free(&keep);
        return rc;
    }
    struct state s = {0};
    char err[120];
    for (size_t i = 0; i < in.n;) {
        const char *l = in.p + i, *nl = memchr(l, '\n', in.n - i);
        size_t ln = nl ? (size_t)(nl - l) : in.n - i;
        i += ln + 1;
        struct janas_json_doc *d =
            ln ? janas_json_parse(l, ln, err, sizeof err) : NULL;
        const struct janas_json *o = d ? janas_json_root(d) : NULL;
        /* a base is the oldest state there is: kept whatever its point */
        if (o && (janas_json_get(o, "base") ||
                  janas_json_num(janas_json_get(o, "at"), 0) <= upto)) {
            apply(&s, o);
            janas_buf_put(&keep, l, ln);
            janas_buf_puts(&keep, "\n");
        }
        janas_json_free(d);
    }
    st_write(&s, &out);
    if (!s.oom && !keep.oom && !out.oom &&
        put_file(tlog, keep.p, keep.n) == 0 &&
        put_file(tstate, out.p, out.n) == 0)
        rc = 0;
    st_free(&s);
    janas_buf_free(&in);
    janas_buf_free(&keep);
    janas_buf_free(&out);
    return rc;
}
