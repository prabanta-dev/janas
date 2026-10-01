/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * template.c - a layout filled with data (see template.h).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/template.h"

#define DEPTH 32

struct frame {
    const struct janas_json *v;
    int index; /* in its list, from 1; 0: not an item */
};

struct ctx {
    const char *t; /* the layout */
    size_t n;
    const char *w; /* the words */
    size_t wn;
    struct frame st[DEPTH];
    int depth;
    struct janas_buf *out;
    char *err;
    size_t err_len;
};

struct tag {
    size_t at, end; /* "{{" and past "}}" */
    char kind;      /* 0 a value, '#', '^', '/', '@' */
    const char *name, *filt;
    size_t name_n, filt_n;
};

/* The layout and its words: what follows a line "---". */
static void split(const char *t, size_t n, size_t *layout_n, const char **w,
                  size_t *wn)
{
    *layout_n = n;
    *w = t + n;
    *wn = 0;
    for (size_t i = 0; i + 3 <= n; i++) {
        if ((i == 0 || t[i - 1] == '\n') && memcmp(t + i, "---", 3) == 0 &&
            (i + 3 == n || t[i + 3] == '\n')) {
            *layout_n = i > 0 ? i - 1 : 0; /* without its newline */
            size_t from = i + 3 < n ? i + 4 : n;
            *w = t + from;
            *wn = n - from;
            return;
        }
    }
}

static int is_space(char c)
{
    return c == ' ' || c == '\t';
}

/* The tag that begins at i ("{{"): 0, or -1 when it is not ended. */
static int read_tag(const char *t, size_t n, size_t i, struct tag *g)
{
    size_t k = i + 2;
    while (k + 1 < n && !(t[k] == '}' && t[k + 1] == '}'))
        k++;
    if (k + 1 >= n)
        return -1;
    size_t a = i + 2, b = k;
    while (a < b && is_space(t[a]))
        a++;
    while (b > a && is_space(t[b - 1]))
        b--;
    *g = (struct tag){.at = i, .end = k + 2};
    if (a < b && strchr("#^/@", t[a])) {
        g->kind = t[a++];
        while (a < b && is_space(t[a]))
            a++;
    }
    const char *bar = memchr(t + a, '|', b - a);
    g->name = t + a;
    g->name_n = bar ? (size_t)(bar - (t + a)) : b - a;
    while (g->name_n && is_space(g->name[g->name_n - 1]))
        g->name_n--;
    if (bar) {
        g->filt = bar + 1;
        g->filt_n = (size_t)(t + b - g->filt);
        while (g->filt_n && is_space(*g->filt)) {
            g->filt++;
            g->filt_n--;
        }
    }
    return 0;
}

/* The next tag in [i, to): 1, 0 when none, -1 when one is not ended. */
static int next_tag(const char *t, size_t to, size_t i, struct tag *g)
{
    for (; i + 1 < to; i++)
        if (t[i] == '{' && t[i + 1] == '{')
            return read_tag(t, to, i, g) == 0 ? 1 : -1;
    return 0;
}

/* A part's tag alone on its line: where the line begins, and past its end;
   0 when other text shares the line. */
static int alone(const struct ctx *c, const struct tag *g, size_t *line,
                 size_t *past)
{
    if (g->kind != '#' && g->kind != '^' && g->kind != '/')
        return 0;
    size_t a = g->at;
    while (a > 0 && c->t[a - 1] != '\n')
        a--;
    for (size_t k = a; k < g->at; k++)
        if (!is_space(c->t[k]))
            return 0;
    size_t e = g->end;
    while (e < c->n && is_space(c->t[e]))
        e++;
    if (e < c->n && c->t[e] != '\n')
        return 0;
    *line = a;
    *past = e < c->n ? e + 1 : e;
    return 1;
}

static int fail(struct ctx *c, const char *what, const struct tag *g)
{
    snprintf(c->err, c->err_len, "%s: {{%c%.*s}} at byte %zu", what,
             g->kind ? g->kind : ' ', (int)g->name_n, g->name, g->at);
    return -1;
}

/* The tag closing the part opened by g, from i. */
static int find_close(struct ctx *c, const struct tag *g, size_t i, size_t to,
                      struct tag *close)
{
    int depth = 0;
    struct tag h;
    int r;
    while ((r = next_tag(c->t, to, i, &h)) == 1) {
        i = h.end;
        if (h.kind == '#' || h.kind == '^')
            depth++;
        else if (h.kind == '/' && depth > 0)
            depth--;
        else if (h.kind == '/') {
            if (h.name_n != g->name_n || memcmp(h.name, g->name, h.name_n))
                return fail(c, "closed by another name", g);
            *close = h;
            return 0;
        }
    }
    return fail(c, r < 0 ? "a tag not ended" : "a part not closed", g);
}

static const struct janas_json *member(const struct janas_json *o,
                                       const char *k, size_t kn)
{
    char key[128];
    if (!o || o->type != JANAS_JSON_OBJECT || kn >= sizeof key)
        return NULL;
    memcpy(key, k, kn);
    key[kn] = 0;
    return janas_json_get(o, key);
}

/* name in the innermost frame that has its first part, then the rest. */
static const struct janas_json *lookup(const struct ctx *c, const char *name,
                                       size_t n)
{
    if (n == 1 && name[0] == '.')
        return c->depth ? c->st[c->depth - 1].v : NULL;
    const char *dot = memchr(name, '.', n);
    size_t first = dot ? (size_t)(dot - name) : n;
    const struct janas_json *v = NULL;
    for (int f = c->depth - 1; f >= 0 && !v; f--)
        v = member(c->st[f].v, name, first);
    while (v && dot) {
        const char *s = dot + 1;
        size_t left = n - (size_t)(s - name);
        dot = memchr(s, '.', left);
        v = member(v, s, dot ? (size_t)(dot - s) : left);
    }
    return v;
}

static int truthy(const struct janas_json *v)
{
    if (!v)
        return 0;
    switch (v->type) {
    case JANAS_JSON_NULL:
    case JANAS_JSON_FALSE:
        return 0;
    case JANAS_JSON_NUMBER:
        return janas_json_num(v, 0) != 0;
    case JANAS_JSON_STRING:
        return v->n > 0;
    case JANAS_JSON_ARRAY:
        return v->child != NULL;
    default:
        return 1;
    }
}

/* The word key, from the words: 1 with it in *s, *n. */
static int word(const struct ctx *c, const char *key, size_t kn, const char **s,
                size_t *n)
{
    const char *p = c->w, *end = c->w + c->wn;
    while (p < end) {
        const char *nl = memchr(p, '\n', (size_t)(end - p));
        const char *e = nl ? nl : end;
        const char *eq = memchr(p, '=', (size_t)(e - p));
        if (eq && *p != '#') {
            const char *a = p, *b = eq;
            while (a < b && is_space(*a))
                a++;
            while (b > a && is_space(b[-1]))
                b--;
            if ((size_t)(b - a) == kn && memcmp(a, key, kn) == 0) {
                const char *v = eq + 1, *ve = e;
                while (v < ve && is_space(*v))
                    v++;
                while (ve > v && (is_space(ve[-1]) || ve[-1] == '\r'))
                    ve--;
                *s = v;
                *n = (size_t)(ve - v);
                return 1;
            }
        }
        p = nl ? nl + 1 : end;
    }
    return 0;
}

/* A word as written: "\n" in it is a new line. */
static void put_word(struct janas_buf *b, const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\\' && i + 1 < n && s[i + 1] == 'n') {
            janas_buf_put(b, "\n", 1);
            i++;
        } else {
            janas_buf_put(b, s + i, 1);
        }
    }
}

static void put_value(struct ctx *c, const struct tag *g,
                      const struct janas_json *v)
{
    if (!v || (v->type != JANAS_JSON_STRING && v->type != JANAS_JSON_NUMBER))
        return;
    if (g->filt_n) {
        char key[192];
        int kn = snprintf(key, sizeof key, "%.*s.%.*s", (int)g->filt_n, g->filt,
                          (int)v->n, v->s);
        const char *s;
        size_t n;
        if (kn > 0 && (size_t)kn < sizeof key &&
            word(c, key, (size_t)kn, &s, &n)) {
            put_word(c->out, s, n);
            return;
        }
    }
    if (v->type == JANAS_JSON_NUMBER) {
        const char *d;
        size_t dn = 0;
        int point = word(c, "decimal", 7, &d, &dn) && dn > 0;
        for (size_t i = 0; i < v->n; i++)
            if (point && v->s[i] == '.')
                janas_buf_put(c->out, d, dn);
            else
                janas_buf_put(c->out, v->s + i, 1);
        return;
    }
    janas_buf_put(c->out, v->s, v->n);
}

static int render(struct ctx *c, size_t from, size_t to)
{
    size_t i = from;
    struct tag g;
    int r;
    while ((r = next_tag(c->t, to, i, &g)) == 1) {
        size_t line, past;
        int solo = alone(c, &g, &line, &past) && line >= i;
        janas_buf_put(c->out, c->t + i, (solo ? line : g.at) - i);
        i = solo ? past : g.end;
        if (g.kind == '/')
            return fail(c, "a part closed but not opened", &g);
        if (g.kind == '@') {
            for (int f = c->depth - 1; f >= 0; f--)
                if (c->st[f].index) {
                    janas_buf_printf(c->out, "%d", c->st[f].index);
                    break;
                }
            continue;
        }
        if (!g.kind) {
            put_value(c, &g, lookup(c, g.name, g.name_n));
            continue;
        }
        struct tag close;
        if (find_close(c, &g, g.end, to, &close) != 0)
            return -1;
        size_t cl, cp;
        int csolo = alone(c, &close, &cl, &cp);
        size_t body = i, body_end = csolo ? cl : close.at;
        if (body_end < body)
            body_end = body;
        i = csolo ? cp : close.end;
        const struct janas_json *v = lookup(c, g.name, g.name_n);
        if (g.kind == '^') {
            if (!truthy(v) && render(c, body, body_end) != 0)
                return -1;
            continue;
        }
        if (!truthy(v))
            continue;
        if (c->depth >= DEPTH)
            return fail(c, "parts too deep", &g);
        if (v->type == JANAS_JSON_ARRAY) {
            int k = 0;
            for (const struct janas_json *it = v->child; it; it = it->next) {
                c->st[c->depth++] = (struct frame){.v = it, .index = ++k};
                int e = render(c, body, body_end);
                c->depth--;
                if (e != 0)
                    return -1;
            }
        } else {
            c->st[c->depth++] = (struct frame){.v = v};
            int e = render(c, body, body_end);
            c->depth--;
            if (e != 0)
                return -1;
        }
    }
    if (r < 0) {
        snprintf(c->err, c->err_len, "a tag not ended at byte %zu", i);
        return -1;
    }
    janas_buf_put(c->out, c->t + i, to - i);
    return 0;
}

int janas_tpl_render(const char *tpl, size_t n, const struct janas_json *data,
                     struct janas_buf *out, char *err, size_t err_len)
{
    struct ctx c = {.t = tpl, .out = out, .err = err, .err_len = err_len};
    split(tpl, n, &c.n, &c.w, &c.wn);
    c.st[c.depth++] = (struct frame){.v = data};
    if (render(&c, 0, c.n) != 0)
        return -1;
    if (out->oom) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    return 0;
}

int janas_tpl_result(struct janas_buf *b, const char *name, const char *data,
                     size_t data_n, const char *layout, const char *brief,
                     char *err, size_t err_len)
{
    char why[160];
    struct janas_json_doc *d = janas_json_parse(data, data_n, why, sizeof why);
    if (!d) {
        snprintf(err, err_len, "the data: %s", why);
        return -1;
    }
    struct janas_buf text = {0};
    int r = janas_tpl_render(layout, strlen(layout), janas_json_root(d), &text,
                             err, err_len);
    janas_json_free(d);
    if (r != 0) {
        janas_buf_free(&text);
        return -1;
    }
    janas_buf_puts(b, "\"content\": [{\"type\": \"text\", \"text\": ");
    janas_json_write_str(b, text.p ? text.p : "", text.n);
    janas_buf_puts(b, "}], \"structuredContent\": ");
    janas_buf_put(b, data, data_n);
    janas_buf_puts(b, ", \"_meta\": {\"dev.prabanta.janas/layout\": "
                      "{\"name\": ");
    janas_json_write_str(b, name, strlen(name));
    janas_buf_puts(b, ", \"text\": ");
    janas_json_write_str(b, layout, strlen(layout));
    janas_buf_puts(b, ", \"brief\": ");
    janas_json_write_str(b, brief, strlen(brief));
    janas_buf_puts(b, "}}, \"isError\": false");
    janas_buf_free(&text);
    return 0;
}

/* ---- the shape of a layout ---- */

struct list {
    char **s;
    size_t n, cap;
};

static int list_add(struct list *l, const char *s, size_t n)
{
    if (l->n == l->cap) {
        size_t cap = l->cap ? 2 * l->cap : 32;
        char **g = realloc(l->s, cap * sizeof *g);
        if (!g)
            return -1;
        l->s = g;
        l->cap = cap;
    }
    char *d = malloc(n + 1);
    if (!d)
        return -1;
    memcpy(d, s, n);
    d[n] = 0;
    l->s[l->n++] = d;
    return 0;
}

static void list_free(struct list *l)
{
    for (size_t i = 0; i < l->n; i++)
        free(l->s[i]);
    free(l->s);
}

static int by_text(const void *a, const void *b)
{
    return strcmp(*(char *const *)a, *(char *const *)b);
}

/* Its parts in order, its fields and its words' keys sorted. */
static int shape(const char *t, size_t n, struct list *parts,
                 struct list *fields, struct list *keys)
{
    size_t ln, wn;
    const char *w;
    split(t, n, &ln, &w, &wn);
    struct tag g;
    int r;
    size_t i = 0;
    char buf[256];
    while ((r = next_tag(t, ln, i, &g)) == 1) {
        i = g.end;
        int k = snprintf(buf, sizeof buf, "%c%.*s|%.*s", g.kind ? g.kind : '=',
                         (int)g.name_n, g.name, (int)g.filt_n,
                         g.filt ? g.filt : "");
        if (k < 0 || (size_t)k >= sizeof buf)
            k = sizeof buf - 1;
        if (list_add(g.kind == '#' || g.kind == '^' || g.kind == '/' ? parts
                                                                     : fields,
                     buf, (size_t)k) != 0)
            return -1;
    }
    if (r < 0)
        return -1;
    const char *p = w, *end = w + wn;
    while (p < end) {
        const char *nl = memchr(p, '\n', (size_t)(end - p));
        const char *e = nl ? nl : end;
        const char *eq = memchr(p, '=', (size_t)(e - p));
        if (eq && *p != '#') {
            const char *a = p, *b = eq;
            while (a < b && is_space(*a))
                a++;
            while (b > a && is_space(b[-1]))
                b--;
            /* the decimal point is the language's: a translation may add
               it */
            if (!((size_t)(b - a) == 7 && memcmp(a, "decimal", 7) == 0) &&
                list_add(keys, a, (size_t)(b - a)) != 0)
                return -1;
        }
        p = nl ? nl + 1 : end;
    }
    qsort(fields->s, fields->n, sizeof *fields->s, by_text);
    qsort(keys->s, keys->n, sizeof *keys->s, by_text);
    return 0;
}

static int same(const struct list *a, const struct list *b, const char *what,
                char *why, size_t why_len)
{
    for (size_t i = 0; i < a->n || i < b->n; i++) {
        const char *x = i < a->n ? a->s[i] : "(none)";
        const char *y = i < b->n ? b->s[i] : "(none)";
        if (strcmp(x, y) != 0) {
            snprintf(why, why_len, "%s differ: %s against %s", what, x, y);
            return 0;
        }
    }
    return 1;
}

int janas_tpl_same_shape(const char *a, size_t an, const char *b, size_t bn,
                         char *why, size_t why_len)
{
    struct list pa = {0}, fa = {0}, ka = {0}, pb = {0}, fb = {0}, kb = {0};
    int ok = 0;
    if (shape(a, an, &pa, &fa, &ka) != 0 || shape(b, bn, &pb, &fb, &kb) != 0)
        snprintf(why, why_len, "a layout could not be read");
    else
        ok = same(&pa, &pb, "parts", why, why_len) &&
             same(&fa, &fb, "fields", why, why_len) &&
             same(&ka, &kb, "words", why, why_len);
    list_free(&pa);
    list_free(&fa);
    list_free(&ka);
    list_free(&pb);
    list_free(&fb);
    list_free(&kb);
    return ok;
}
