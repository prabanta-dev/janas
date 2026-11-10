/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jsonc.c - a member of a JSON file with comments changed in place (see
 * jsonc.h). The text is walked, not parsed into a tree: a member is found
 * by its key's bytes between the quotes (the keys asked for have no
 * escapes), and the new text is the old one with one span replaced, one
 * span cut out, or one piece put in.
 */
#include "jsonc.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cur {
    const char *s;
    size_t n, i;
};

/* spaces and comments; -1 at a comment never closed */
static int ws(struct cur *c)
{
    while (c->i < c->n) {
        char ch = c->s[c->i];
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            c->i++;
        } else if (ch == '/' && c->i + 1 < c->n && c->s[c->i + 1] == '/') {
            while (c->i < c->n && c->s[c->i] != '\n')
                c->i++;
        } else if (ch == '/' && c->i + 1 < c->n && c->s[c->i + 1] == '*') {
            const char *e = NULL;
            for (size_t k = c->i + 2; k + 1 < c->n; k++)
                if (c->s[k] == '*' && c->s[k + 1] == '/') {
                    e = c->s + k;
                    break;
                }
            if (!e)
                return -1;
            c->i = (size_t)(e - c->s) + 2;
        } else {
            break;
        }
    }
    return 0;
}

/* a string, from its opening quote; -1 when it never closes */
static int str(struct cur *c)
{
    if (c->i >= c->n || c->s[c->i] != '"')
        return -1;
    for (c->i++; c->i < c->n; c->i++) {
        if (c->s[c->i] == '\\')
            c->i++;
        else if (c->s[c->i] == '"') {
            c->i++;
            return 0;
        } else if (c->s[c->i] == '\n')
            return -1;
    }
    return -1;
}

static int value(struct cur *c, int depth);

/* the items of an object or an array, from its bracket; each one's key
   (objects) and value spans passed to fn, which stops the walk with a
   value other than 0 */
typedef int (*item_fn)(void *arg, size_t key_at, size_t key_len, size_t val_at,
                       size_t val_end);

static int items(struct cur *c, int depth, item_fn fn, void *arg)
{
    if (depth > 64)
        return -1;
    char close = c->s[c->i] == '{' ? '}' : ']';
    int obj = close == '}';
    c->i++;
    for (;;) {
        if (ws(c) != 0 || c->i >= c->n)
            return -1;
        if (c->s[c->i] == close) {
            c->i++;
            return 0;
        }
        size_t key_at = 0, key_len = 0;
        if (obj) {
            size_t k = c->i;
            if (str(c) != 0)
                return -1;
            key_at = k + 1;
            key_len = c->i - k - 2;
            if (ws(c) != 0 || c->i >= c->n || c->s[c->i] != ':')
                return -1;
            c->i++;
            if (ws(c) != 0)
                return -1;
        }
        size_t at = c->i;
        if (value(c, depth + 1) != 0)
            return -1;
        if (fn) {
            int r = fn(arg, key_at, key_len, at, c->i);
            if (r)
                return r;
        }
        if (ws(c) != 0 || c->i >= c->n)
            return -1;
        if (c->s[c->i] == ',')
            c->i++; /* a trailing one too: the bracket comes next */
        else if (c->s[c->i] != close)
            return -1;
    }
}

static int value(struct cur *c, int depth)
{
    if (c->i >= c->n)
        return -1;
    char ch = c->s[c->i];
    if (ch == '{' || ch == '[')
        return items(c, depth, NULL, NULL);
    if (ch == '"')
        return str(c);
    size_t at = c->i;
    while (c->i < c->n && !strchr(" \t\r\n,]}/", c->s[c->i]))
        c->i++;
    return c->i > at ? 0 : -1;
}

/* the members of the object at c, one by key */
struct seek {
    const char *s, *key;
    size_t key_at, val_at, val_end, last_end;
    size_t prev_end; /* the end of the item before the one found, 0: none */
    int found, n_items;
};

static int seek_fn(void *arg, size_t key_at, size_t key_len, size_t val_at,
                   size_t val_end)
{
    struct seek *k = arg;
    k->n_items++;
    k->prev_end = k->last_end;
    k->last_end = val_end;
    if (k->key && strlen(k->key) == key_len &&
        memcmp(k->s + key_at, k->key, key_len) == 0) {
        k->found = 1;
        k->key_at = key_at - 1;
        k->val_at = val_at;
        k->val_end = val_end;
        return 1;
    }
    return 0;
}

/* the object or array at c->i walked with sk; c->i left at its bracket */
static int walk(struct cur *c, struct seek *sk)
{
    size_t open = c->i;
    int r = items(c, 0, seek_fn, sk);
    c->i = open;
    return r < 0 ? -1 : 0;
}

/* the indentation of the line of position at */
static size_t indent_of(const char *s, size_t at, size_t *line)
{
    size_t b = at;
    while (b > 0 && s[b - 1] != '\n')
        b--;
    size_t e = b;
    while (e < at && (s[e] == ' ' || s[e] == '\t'))
        e++;
    *line = b;
    return e - b;
}

struct edit {
    size_t at, cut;   /* replace s[at, at + cut) */
    char *put;        /* with this (malloc'd) */
    size_t at2, cut2; /* and cut s[at2, at2 + cut2) too, after it */
};

static int apply(const char *s, size_t n, struct edit *e, char **out,
                 size_t *out_n)
{
    size_t put = strlen(e->put), cut = e->cut + e->cut2;
    char *b = malloc(n - cut + put + 1);
    if (!b) {
        free(e->put);
        return -1;
    }
    size_t o = 0, tail = e->at + e->cut;
    memcpy(b, s, e->at);
    o = e->at;
    memcpy(b + o, e->put, put);
    o += put;
    if (e->cut2) {
        memcpy(b + o, s + tail, e->at2 - tail);
        o += e->at2 - tail;
        tail = e->at2 + e->cut2;
    }
    memcpy(b + o, s + tail, n - tail);
    o += n - tail;
    b[o] = 0;
    *out_n = o;
    *out = b;
    free(e->put);
    return 0;
}

static char *fmt(const char *f, ...) __attribute__((format(printf, 1, 2)));
static char *fmt(const char *f, ...)
{
    va_list ap;
    va_start(ap, f);
    int n = vsnprintf(NULL, 0, f, ap);
    va_end(ap);
    char *b = n >= 0 ? malloc((size_t)n + 1) : NULL;
    if (b) {
        va_start(ap, f);
        vsnprintf(b, (size_t)n + 1, f, ap);
        va_end(ap);
    }
    return b;
}

/*
 * An item put last into the object or array whose opening bracket is at
 * open (sk: what the walk found in it): after the last one, with a comma,
 * on a line of its own indented as the items are (one level past the
 * bracket's line when it has none).
 */
static int put_last(const char *s, size_t n, size_t open, struct seek *sk,
                    const char *item, struct edit *e)
{
    *e = (struct edit){0};
    size_t line;
    size_t bind = indent_of(s, open, &line);
    if (sk->n_items == 0) {
        /* the bracket's own line indentation, plus four */
        struct cur c = {.s = s, .n = n, .i = open + 1};
        if (ws(&c) != 0)
            return -1;
        size_t close = c.i; /* the closing bracket */
        int comment = 0;
        for (size_t k = open + 1; k < close; k++)
            if (s[k] == '/')
                comment = 1;
        if (comment) {
            e->at = open + 1;
            e->cut = 0;
            e->put = fmt("\n%.*s    %s", (int)bind, s + line, item);
        } else {
            e->at = open + 1;
            e->cut = close - open - 1;
            e->put = fmt("\n%.*s    %s\n%.*s", (int)bind, s + line, item,
                         (int)bind, s + line);
        }
        return e->put ? 0 : -1;
    }
    size_t iline;
    /* the indentation of the first item: the line of the last one */
    size_t iind = indent_of(s, sk->last_end, &iline);
    /* the last item's line: where its value starts is enough when its
       line holds only it; otherwise one level past the bracket */
    char ind[256];
    if (iline > open && iind < sizeof ind) {
        memcpy(ind, s + iline, iind);
        ind[iind] = 0;
    } else {
        snprintf(ind, sizeof ind, "%.*s    ", (int)(bind < 200 ? bind : 200),
                 s + line);
    }
    size_t at = sk->last_end;
    int comma = 0;
    struct cur c = {.s = s, .n = n, .i = at};
    while (c.i < n && (s[c.i] == ' ' || s[c.i] == '\t'))
        c.i++;
    if (c.i < n && s[c.i] == ',') {
        comma = 1;
        at = c.i + 1;
    }
    /* a comment to the end of the line stays on its line */
    size_t k = at;
    while (k < n && (s[k] == ' ' || s[k] == '\t'))
        k++;
    if (k + 1 < n && s[k] == '/' && s[k + 1] == '/') {
        while (k < n && s[k] != '\n')
            k++;
        if (!comma) {
            /* the comma goes before the comment: two edits in one */
            e->at = sk->last_end;
            e->cut = k - sk->last_end;
            e->put = fmt(",%.*s\n%s%s", (int)(k - sk->last_end),
                         s + sk->last_end, ind, item);
            return e->put ? 0 : -1;
        }
        at = k;
    }
    /* after a trailing comma the new item gets one too: the file's way,
       and taking the item out again gives back the same bytes */
    e->at = at;
    e->cut = 0;
    e->put = comma ? fmt("\n%s%s,", ind, item) : fmt(",\n%s%s", ind, item);
    return e->put ? 0 : -1;
}

/* a key as JSON writes it */
static char *quoted(const char *key)
{
    return fmt("\"%s\"", key);
}

int janas_jsonc_find(const char *text, size_t n, const char *const *path,
                     int n_path, size_t *at, size_t *len)
{
    struct cur c = {.s = text, .n = n, .i = 0};
    if (ws(&c) != 0 || c.i >= n || text[c.i] != '{')
        return -1;
    for (int p = 0; p < n_path; p++) {
        if (text[c.i] != '{')
            return -1;
        struct seek sk = {.s = text, .key = path[p]};
        if (walk(&c, &sk) != 0)
            return -1;
        if (!sk.found)
            return 0;
        c.i = sk.val_at;
        if (p == n_path - 1) {
            *at = sk.val_at;
            *len = sk.val_end - sk.val_at;
            return 1;
        }
    }
    return -1;
}

/* the whole text checked, so that an edit never lands in a broken file */
static int whole(const char *text, size_t n, char want)
{
    struct cur c = {.s = text, .n = n, .i = 0};
    if (ws(&c) != 0)
        return -1;
    if (c.i == n)
        return 1; /* empty */
    if (text[c.i] != want || value(&c, 0) != 0 || ws(&c) != 0 || c.i != n)
        return -1;
    return 0;
}

int janas_jsonc_set(const char *text, size_t n, const char *const *path,
                    int n_path, const char *val, char **out, size_t *out_n)
{
    *out = NULL;
    int w = whole(text, n, '{');
    if (w < 0 || n_path < 1)
        return -1;
    /* the value with the keys not there yet around it */
    if (w == 1) {
        char *v = strdup(val);
        for (int p = n_path - 1; v && p >= 1; p--) {
            char *o = fmt("{\"%s\": %s}", path[p], v);
            free(v);
            v = o;
        }
        if (!v)
            return -1;
        struct edit e = {.at = 0,
                         .cut = n,
                         .put = fmt("{\n    \"%s\": %s\n}\n", path[0], v)};
        free(v);
        return e.put ? apply(text, n, &e, out, out_n) : -1;
    }
    struct cur c = {.s = text, .n = n, .i = 0};
    ws(&c);
    for (int p = 0; p < n_path; p++) {
        if (text[c.i] != '{')
            return -1;
        struct seek sk = {.s = text, .key = path[p]};
        size_t open = c.i;
        if (walk(&c, &sk) != 0)
            return -1;
        if (sk.found && p < n_path - 1) {
            c.i = sk.val_at;
            continue;
        }
        struct edit e;
        if (sk.found) {
            e = (struct edit){.at = sk.val_at,
                              .cut = sk.val_end - sk.val_at,
                              .put = strdup(val)};
            if (!e.put)
                return -1;
        } else {
            char *v = strdup(val);
            for (int q = n_path - 1; v && q > p; q--) {
                char *o = fmt("{\"%s\": %s}", path[q], v);
                free(v);
                v = o;
            }
            char *k = quoted(path[p]);
            char *item = v && k ? fmt("%s: %s", k, v) : NULL;
            free(v);
            free(k);
            if (!item)
                return -1;
            int r = put_last(text, n, open, &sk, item, &e);
            free(item);
            if (r != 0)
                return -1;
        }
        return apply(text, n, &e, out, out_n);
    }
    return -1;
}

/* An item cut out with its comma - its own after it, or for the last one
   the comma after the item before it (prev_end), a comment between them
   kept - and its line when the line holds nothing else, or the spaces
   before it when it shares its line. */
static void cut_item(const char *s, size_t n, size_t from, size_t to,
                     size_t prev_end, struct edit *e)
{
    *e = (struct edit){0};
    size_t k = to;
    while (k < n && (s[k] == ' ' || s[k] == '\t'))
        k++;
    int own = k < n && s[k] == ',';
    if (own)
        to = k + 1;
    size_t comma = 0; /* the comma before it, +1; 0: none */
    if (!own && prev_end) {
        k = prev_end;
        while (k < n && (s[k] == ' ' || s[k] == '\t'))
            k++;
        if (k < from && s[k] == ',')
            comma = k + 1;
    }
    size_t b = from;
    while (b > 0 && (s[b - 1] == ' ' || s[b - 1] == '\t'))
        b--;
    size_t a = to;
    while (a < n && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r'))
        a++;
    if ((b == 0 || s[b - 1] == '\n') && a < n && s[a] == '\n') {
        from = b;
        to = a + 1;
    } else if (comma) {
        /* on a line with other things: the spaces and breaks before it
           go, back to the comma */
        while (from > comma && strchr(" \t\r\n", s[from - 1]))
            from--;
    }
    e->put = strdup("");
    if (comma) {
        e->at = comma - 1;
        e->cut = 1;
        if (from == comma) {
            e->cut = to - e->at;
        } else {
            e->at2 = from;
            e->cut2 = to - from;
        }
    } else {
        e->at = from;
        e->cut = to - from;
    }
}

int janas_jsonc_remove(const char *text, size_t n, const char *const *path,
                       int n_path, char **out, size_t *out_n)
{
    *out = NULL;
    int w = whole(text, n, '{');
    if (w < 0 || n_path < 1)
        return -1;
    if (w == 1)
        return 0;
    struct cur c = {.s = text, .n = n, .i = 0};
    ws(&c);
    for (int p = 0; p < n_path; p++) {
        if (text[c.i] != '{')
            return p == 0 ? -1 : 0;
        struct seek sk = {.s = text, .key = path[p]};
        if (walk(&c, &sk) != 0)
            return -1;
        if (!sk.found)
            return 0;
        if (p < n_path - 1) {
            c.i = sk.val_at;
            continue;
        }
        struct edit e;
        cut_item(text, n, sk.key_at, sk.val_end, sk.prev_end, &e);
        if (!e.put)
            return -1;
        return apply(text, n, &e, out, out_n) == 0 ? 1 : -1;
    }
    return -1;
}

/* the item of the array whose member key is the string match */
struct pick {
    const char *s, *key, *match;
    size_t at, end, last_end, prev_end;
    int found, n_items;
};

static int pick_fn(void *arg, size_t key_at, size_t key_len, size_t val_at,
                   size_t val_end)
{
    (void)key_at;
    (void)key_len;
    struct pick *p = arg;
    p->n_items++;
    p->prev_end = p->last_end;
    p->last_end = val_end;
    if (p->s[val_at] != '{')
        return 0;
    struct cur c = {.s = p->s, .n = val_end, .i = val_at};
    struct seek sk = {.s = p->s, .key = p->key};
    if (walk(&c, &sk) != 0 || !sk.found)
        return 0;
    size_t m = strlen(p->match);
    if (sk.val_end - sk.val_at == m + 2 && p->s[sk.val_at] == '"' &&
        memcmp(p->s + sk.val_at + 1, p->match, m) == 0) {
        p->found = 1;
        p->at = val_at;
        p->end = val_end;
        return 1;
    }
    return 0;
}

static int array_pick(const char *text, size_t n, struct pick *p, size_t *open)
{
    struct cur c = {.s = text, .n = n, .i = 0};
    ws(&c);
    *open = c.i;
    return items(&c, 0, pick_fn, p) < 0 ? -1 : 0;
}

int janas_jsonc_array_put(const char *text, size_t n, const char *key,
                          const char *match, const char *val, char **out,
                          size_t *out_n)
{
    *out = NULL;
    int w = whole(text, n, '[');
    if (w < 0)
        return -1;
    if (w == 1) {
        struct edit e = {.at = 0, .cut = n, .put = fmt("[\n    %s\n]\n", val)};
        return e.put ? apply(text, n, &e, out, out_n) : -1;
    }
    struct pick p = {.s = text, .key = key, .match = match};
    size_t open;
    if (array_pick(text, n, &p, &open) != 0)
        return -1;
    struct edit e;
    if (p.found) {
        e = (struct edit){.at = p.at, .cut = p.end - p.at, .put = strdup(val)};
        if (!e.put)
            return -1;
    } else {
        struct seek sk = {.n_items = p.n_items, .last_end = p.last_end};
        if (put_last(text, n, open, &sk, val, &e) != 0)
            return -1;
    }
    return apply(text, n, &e, out, out_n);
}

int janas_jsonc_array_remove(const char *text, size_t n, const char *key,
                             const char *match, char **out, size_t *out_n)
{
    *out = NULL;
    int w = whole(text, n, '[');
    if (w < 0)
        return -1;
    if (w == 1)
        return 0;
    struct pick p = {.s = text, .key = key, .match = match};
    size_t open;
    if (array_pick(text, n, &p, &open) != 0)
        return -1;
    if (!p.found)
        return 0;
    struct edit e;
    cut_item(text, n, p.at, p.end, p.prev_end, &e);
    if (!e.put)
        return -1;
    return apply(text, n, &e, out, out_n) == 0 ? 1 : -1;
}
