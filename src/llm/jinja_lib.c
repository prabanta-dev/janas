/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jinja_lib.c - the values of a template's run (jinja.h) and what they do:
 * written as Python writes them (str(), repr(), json.dumps as Hugging
 * Face's tojson calls it), compared, added; the members of strings, dicts
 * and lists the chat templates use; Jinja's filters and tests; range,
 * namespace, dict, raise_exception, strftime_now. A string carries the
 * marks of its bytes through all of it: a slice, a strip, a replace, a
 * join or a JSON escape keeps each byte's mark, or gives what it writes
 * of a marked byte the same mark.
 */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llm/jinja_int.h"

/* ---- constructors */
struct jval jv_undef(void)
{
    return (struct jval){.t = J_UNDEF};
}
struct jval jv_none(void)
{
    return (struct jval){.t = J_NONE};
}
struct jval jv_bool(int b)
{
    return (struct jval){.t = J_BOOL, .u.b = !!b};
}
struct jval jv_int(int64_t i)
{
    return (struct jval){.t = J_INT, .u.i = i};
}
struct jval jv_float(double f)
{
    return (struct jval){.t = J_FLOAT, .u.f = f};
}
struct jval jv_str(struct jstr s)
{
    return (struct jval){.t = J_STR, .u.s = s};
}
struct jval jv_cstr(const char *s)
{
    return jv_str((struct jstr){.p = s, .n = strlen(s)});
}

struct jstr js_make(struct jrun *r, const char *p, size_t n, int mark)
{
    char *c = ja_alloc(&r->ar, n + 1);
    uint8_t *m = mark && n ? ja_alloc(&r->ar, n) : NULL;
    if (!c || (mark && n && !m)) {
        jinja_fail(r, 0, "out of memory");
        return (struct jstr){.p = "", .n = 0};
    }
    memcpy(c, p, n);
    if (m)
        memset(m, 1, n);
    return (struct jstr){.p = c, .m = m, .n = n};
}

int js_eq(struct jstr a, const char *s)
{
    return a.n == strlen(s) && !memcmp(a.p, s, a.n);
}

/* ---- a string being built, with its marks */
struct sb {
    struct jrun *r;
    char *p;
    uint8_t *m;
    size_t n, cap;
    int any;
};

static void sb_grow(struct sb *b, size_t more)
{
    if (b->n + more <= b->cap)
        return;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < b->n + more)
        cap *= 2;
    char *p = realloc(b->p, cap);
    uint8_t *m = p ? realloc(b->m, cap) : NULL;
    if (p)
        b->p = p;
    if (m)
        b->m = m;
    if (!p || !m) {
        jinja_fail(b->r, 0, "out of memory");
        return;
    }
    b->cap = cap;
}

/* n bytes, each marked as m says (NULL: all mark) */
static void sb_put(struct sb *b, const char *p, size_t n, const uint8_t *m,
                   uint8_t mark)
{
    if (!n || b->r->failed)
        return;
    sb_grow(b, n);
    if (b->r->failed)
        return;
    memcpy(b->p + b->n, p, n);
    for (size_t i = 0; i < n; i++) {
        uint8_t x = m ? m[i] : mark;
        b->m[b->n + i] = x;
        b->any |= x;
    }
    b->n += n;
}
static void sb_str(struct sb *b, struct jstr s)
{
    sb_put(b, s.p, s.n, s.m, 0);
}
static void sb_lit(struct sb *b, const char *s)
{
    sb_put(b, s, strlen(s), NULL, 0);
}

static struct jstr sb_done(struct sb *b)
{
    struct jstr s = {.p = "", .n = 0};
    if (!b->r->failed && b->n) {
        char *p = ja_alloc(&b->r->ar, b->n + 1);
        uint8_t *m = b->any ? ja_alloc(&b->r->ar, b->n) : NULL;
        if (!p || (b->any && !m))
            jinja_fail(b->r, 0, "out of memory");
        else {
            memcpy(p, b->p, b->n);
            if (m)
                memcpy(m, b->m, b->n);
            s = (struct jstr){.p = p, .m = m, .n = b->n};
        }
    }
    free(b->p);
    free(b->m);
    b->p = NULL, b->m = NULL;
    return s;
}

static struct jstr js_sub(struct jstr s, size_t a, size_t b)
{
    if (b > s.n)
        b = s.n;
    if (a > b)
        a = b;
    return (struct jstr){.p = s.p + a, .m = s.m ? s.m + a : NULL, .n = b - a};
}

/* ---- UTF-8: Python counts characters */
static size_t u8_len(const char *p, size_t n)
{
    size_t c = 0;
    for (size_t i = 0; i < n; i++)
        c += ((unsigned char)p[i] & 0xc0) != 0x80;
    return c;
}
/* the byte where character k starts (n past the end) */
static size_t u8_at(const char *p, size_t n, size_t k)
{
    size_t i = 0;
    while (i < n && k) {
        i++;
        while (i < n && ((unsigned char)p[i] & 0xc0) == 0x80)
            i++;
        k--;
    }
    return i;
}
static uint32_t u8_get(const char *p, size_t n, size_t *i)
{
    unsigned char c = (unsigned char)p[*i];
    int len = c < 0x80 ? 1 : c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc0 ? 2 : 1;
    if (*i + (size_t)len > n)
        len = 1;
    uint32_t v = len == 1 ? c : c & (0xff >> (len + 1));
    for (int k = 1; k < len; k++)
        v = v << 6 | ((unsigned char)p[*i + (size_t)k] & 63);
    *i += (size_t)len;
    return v;
}

static int u_space(uint32_t c)
{
    return (c >= 9 && c <= 13) || c == ' ' || (c >= 0x1c && c <= 0x1f) ||
           c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 ||
           c == 0x202f || c == 0x205f || c == 0x3000;
}

/* ---- truth, text */
int jv_truthy(const struct jval *v)
{
    switch (v->t) {
    case J_BOOL:
        return v->u.b;
    case J_INT:
        return v->u.i != 0;
    case J_FLOAT:
        return v->u.f != 0;
    case J_STR:
        return v->u.s.n > 0;
    case J_LIST:
        return v->u.l->n > 0;
    case J_DICT:
        return v->u.d->n > 0;
    case J_MACRO:
    case J_FUNC:
        return 1;
    default:
        return 0;
    }
}

/* a float as Python's repr writes it: the shortest that reads back */
static void float_repr(double f, char *o, size_t len)
{
    if (isnan(f)) {
        snprintf(o, len, "nan");
        return;
    }
    if (isinf(f)) {
        snprintf(o, len, f > 0 ? "inf" : "-inf");
        return;
    }
    for (int p = 1; p <= 17; p++) {
        snprintf(o, len, "%.*g", p, f);
        if (strtod(o, NULL) == f)
            break;
    }
    /* Python: 1e+16 and up, 1e-05 and down, in exponents; else digits */
    char *e = strchr(o, 'e');
    double a = fabs(f);
    if (!e && a >= 1e16) {
        snprintf(o, len, "%.16e", f);
        for (int p = 0; p <= 16; p++) {
            char t[64];
            snprintf(t, sizeof t, "%.*e", p, f);
            if (strtod(t, NULL) == f) {
                snprintf(o, len, "%s", t);
                break;
            }
        }
        e = strchr(o, 'e');
    } else if (e && a >= 1e-4 && a < 1e16) {
        /* C chose an exponent Python would not: digits */
        for (int p = 0; p <= 20; p++) {
            char t[64];
            snprintf(t, sizeof t, "%.*f", p, f);
            if (strtod(t, NULL) == f) {
                snprintf(o, len, "%s", t);
                break;
            }
        }
        e = NULL;
    }
    if (e) {
        /* Python: e+16, e-05 */
        char mant[48], *x;
        snprintf(mant, sizeof mant, "%.*s", (int)(e - o), o);
        int ex = atoi(e + 1);
        snprintf(o, len, "%se%c%02d", mant, ex < 0 ? '-' : '+', abs(ex));
        (void)x;
        return;
    }
    if (!strchr(o, '.'))
        strncat(o, ".0", len - strlen(o) - 1);
}

static void repr_str(struct sb *b, struct jstr s)
{
    int sq = 0, dq = 0;
    for (size_t i = 0; i < s.n; i++)
        sq |= s.p[i] == '\'', dq |= s.p[i] == '"';
    char q = sq && !dq ? '"' : '\'';
    sb_put(b, &q, 1, NULL, 0);
    for (size_t i = 0; i < s.n; i++) {
        unsigned char c = (unsigned char)s.p[i];
        uint8_t mk = s.m ? s.m[i] : 0;
        char e[8];
        if (c == '\\' || c == (unsigned char)q) {
            e[0] = '\\', e[1] = (char)c;
            sb_put(b, e, 2, NULL, mk);
        } else if (c == '\n')
            sb_put(b, "\\n", 2, NULL, mk);
        else if (c == '\r')
            sb_put(b, "\\r", 2, NULL, mk);
        else if (c == '\t')
            sb_put(b, "\\t", 2, NULL, mk);
        else if (c < 0x20 || c == 0x7f) {
            snprintf(e, sizeof e, "\\x%02x", c);
            sb_put(b, e, 4, NULL, mk);
        } else
            sb_put(b, s.p + i, 1, NULL, mk);
    }
    sb_put(b, &q, 1, NULL, 0);
}

static void to_text(struct sb *b, const struct jval *v, int repr)
{
    char num[64];
    switch (v->t) {
    case J_UNDEF:
        break;
    case J_NONE:
        sb_lit(b, "None");
        break;
    case J_BOOL:
        sb_lit(b, v->u.b ? "True" : "False");
        break;
    case J_INT:
        snprintf(num, sizeof num, "%lld", (long long)v->u.i);
        sb_lit(b, num);
        break;
    case J_FLOAT:
        float_repr(v->u.f, num, sizeof num);
        sb_lit(b, num);
        break;
    case J_STR:
        if (repr)
            repr_str(b, v->u.s);
        else
            sb_str(b, v->u.s);
        break;
    case J_LIST:
        sb_lit(b, "[");
        for (size_t i = 0; i < v->u.l->n; i++) {
            if (i)
                sb_lit(b, ", ");
            to_text(b, &v->u.l->v[i], 1);
        }
        sb_lit(b, "]");
        break;
    case J_DICT:
        sb_lit(b, "{");
        for (size_t i = 0; i < v->u.d->n; i++) {
            if (i)
                sb_lit(b, ", ");
            repr_str(b, v->u.d->k[i]);
            sb_lit(b, ": ");
            to_text(b, &v->u.d->v[i], 1);
        }
        sb_lit(b, "}");
        break;
    default:
        sb_lit(b, "<function>");
    }
}

struct jstr jv_tostr(struct jrun *r, const struct jval *v)
{
    if (v->t == J_STR)
        return v->u.s;
    struct sb b = {.r = r};
    to_text(&b, v, 0);
    return sb_done(&b);
}

static int str_cmp(struct jstr a, struct jstr b);

/* ---- JSON, as Python's json.dumps(ensure_ascii=False) writes it */
static void json_str(struct sb *b, struct jstr s)
{
    sb_lit(b, "\"");
    size_t i = 0;
    while (i < s.n) {
        unsigned char c = (unsigned char)s.p[i];
        uint8_t mk = s.m ? s.m[i] : 0;
        char e[8];
        if (c == '"' || c == '\\') {
            e[0] = '\\', e[1] = (char)c;
            sb_put(b, e, 2, NULL, mk);
        } else if (c == '\n')
            sb_put(b, "\\n", 2, NULL, mk);
        else if (c == '\r')
            sb_put(b, "\\r", 2, NULL, mk);
        else if (c == '\t')
            sb_put(b, "\\t", 2, NULL, mk);
        else if (c == '\b')
            sb_put(b, "\\b", 2, NULL, mk);
        else if (c == '\f')
            sb_put(b, "\\f", 2, NULL, mk);
        else if (c < 0x20) {
            snprintf(e, sizeof e, "\\u%04x", c);
            sb_put(b, e, 6, NULL, mk);
        } else {
            sb_put(b, s.p + i, 1, NULL, mk);
        }
        i++;
    }
    sb_lit(b, "\"");
}

/* json.dumps's settings: indent (-1 none), the separators, sort_keys */
struct jopts {
    int indent;
    struct jstr item, key;
    int sort;
};

static void json_val(struct sb *b, const struct jval *v, const struct jopts *o,
                     int level)
{
    int indent = o->indent;
    char num[64];
    switch (v->t) {
    case J_NONE:
    case J_UNDEF:
        sb_lit(b, "null");
        break;
    case J_BOOL:
        sb_lit(b, v->u.b ? "true" : "false");
        break;
    case J_INT:
        snprintf(num, sizeof num, "%lld", (long long)v->u.i);
        sb_lit(b, num);
        break;
    case J_FLOAT:
        if (isnan(v->u.f))
            sb_lit(b, "NaN");
        else if (isinf(v->u.f))
            sb_lit(b, v->u.f > 0 ? "Infinity" : "-Infinity");
        else {
            float_repr(v->u.f, num, sizeof num);
            sb_lit(b, num);
        }
        break;
    case J_STR:
        json_str(b, v->u.s);
        break;
    case J_LIST:
    case J_DICT: {
        int dict = v->t == J_DICT;
        size_t n = dict ? v->u.d->n : v->u.l->n;
        sb_lit(b, dict ? "{" : "[");
        if (!n) {
            sb_lit(b, dict ? "}" : "]");
            break;
        }
        /* sort_keys: the order of the keys, by their characters */
        size_t ord[256], *order = ord;
        if (dict && o->sort && n > 256)
            order = malloc(n * sizeof *order);
        if (!order)
            order = ord, n = n > 256 ? 256 : n;
        for (size_t i = 0; i < n; i++)
            order[i] = i;
        if (dict && o->sort)
            for (size_t i = 1; i < n; i++) {
                size_t x = order[i], j = i;
                while (j > 0 &&
                       str_cmp(v->u.d->k[order[j - 1]], v->u.d->k[x]) > 0) {
                    order[j] = order[j - 1];
                    j--;
                }
                order[j] = x;
            }
        for (size_t q = 0; q < n; q++) {
            size_t i = order[q];
            if (q)
                sb_str(b, o->item);
            if (indent >= 0) {
                sb_lit(b, "\n");
                for (int k = 0; k < indent * (level + 1); k++)
                    sb_lit(b, " ");
            }
            if (dict) {
                json_str(b, v->u.d->k[i]);
                sb_str(b, o->key);
                json_val(b, &v->u.d->v[i], o, level + 1);
            } else
                json_val(b, &v->u.l->v[i], o, level + 1);
        }
        if (order != ord)
            free(order);
        if (indent >= 0) {
            sb_lit(b, "\n");
            for (int k = 0; k < indent * level; k++)
                sb_lit(b, " ");
        }
        sb_lit(b, dict ? "}" : "]");
        break;
    }
    default:
        sb_lit(b, "null");
    }
}

struct jstr jv_tojson(struct jrun *r, const struct jval *v, int indent)
{
    struct sb b = {.r = r};
    struct jopts o = {
        .indent = indent,
        .item = {.p = indent >= 0 ? "," : ", ", .n = indent >= 0 ? 1 : 2},
        .key = {.p = ": ", .n = 2}};
    json_val(&b, v, &o, 0);
    return sb_done(&b);
}

/* tojson's arguments as json.dumps takes them */
static struct jstr tojson_with(struct jrun *r, const struct jval *v,
                               const struct jval *ind, const struct jval *sep,
                               const struct jval *sort)
{
    int indent = ind && ind->t == J_INT ? (int)ind->u.i : -1;
    struct jopts o = {
        .indent = indent,
        .item = {.p = indent >= 0 ? "," : ", ", .n = indent >= 0 ? 1 : 2},
        .key = {.p = ": ", .n = 2},
        .sort = sort && jv_truthy(sort)};
    if (sep && sep->t == J_LIST && sep->u.l->n == 2 &&
        sep->u.l->v[0].t == J_STR && sep->u.l->v[1].t == J_STR)
        o.item = sep->u.l->v[0].u.s, o.key = sep->u.l->v[1].u.s;
    struct sb b = {.r = r};
    json_val(&b, v, &o, 0);
    return sb_done(&b);
}

/* ---- equality, order */
static int numeric(const struct jval *v, double *f)
{
    if (v->t == J_INT)
        return *f = (double)v->u.i, 1;
    if (v->t == J_FLOAT)
        return *f = v->u.f, 1;
    if (v->t == J_BOOL)
        return *f = v->u.b, 1;
    return 0;
}

static int str_cmp(struct jstr a, struct jstr b)
{
    size_t n = a.n < b.n ? a.n : b.n;
    int c = memcmp(a.p, b.p, n);
    return c ? c : a.n < b.n ? -1 : a.n > b.n;
}

int jv_equal(const struct jval *a, const struct jval *b)
{
    double x, y;
    if (numeric(a, &x) && numeric(b, &y)) {
        if (a->t == J_INT && b->t == J_INT)
            return a->u.i == b->u.i;
        return x == y;
    }
    if (a->t != b->t)
        return 0;
    switch (a->t) {
    case J_UNDEF:
    case J_NONE:
        return 1;
    case J_STR:
        return !str_cmp(a->u.s, b->u.s);
    case J_LIST:
        if (a->u.l->n != b->u.l->n)
            return 0;
        for (size_t i = 0; i < a->u.l->n; i++)
            if (!jv_equal(&a->u.l->v[i], &b->u.l->v[i]))
                return 0;
        return 1;
    case J_DICT:
        if (a->u.d->n != b->u.d->n)
            return 0;
        for (size_t i = 0; i < a->u.d->n; i++) {
            const struct jval *o =
                jv_dict_get(b, a->u.d->k[i].p, a->u.d->k[i].n);
            if (!o || !jv_equal(&a->u.d->v[i], o))
                return 0;
        }
        return 1;
    case J_MACRO:
        return a->u.macro == b->u.macro;
    default:
        return 0;
    }
}

int jv_compare(const struct jval *a, const struct jval *b)
{
    double x, y;
    if (numeric(a, &x) && numeric(b, &y))
        return x < y ? -1 : x > y;
    if (a->t == J_STR && b->t == J_STR) {
        int c = str_cmp(a->u.s, b->u.s);
        return c < 0 ? -1 : c > 0;
    }
    if (a->t == J_LIST && b->t == J_LIST) {
        for (size_t i = 0; i < a->u.l->n && i < b->u.l->n; i++) {
            int c = jv_compare(&a->u.l->v[i], &b->u.l->v[i]);
            if (c)
                return c;
        }
        return a->u.l->n < b->u.l->n ? -1 : a->u.l->n > b->u.l->n;
    }
    return -2;
}

/* ---- lists, dicts */
struct jval jv_list_new(struct jrun *r, size_t cap)
{
    struct jlist *l = ja_alloc(&r->ar, sizeof *l);
    if (!l) {
        jinja_fail(r, 0, "out of memory");
        static struct jlist empty;
        return (struct jval){.t = J_LIST, .u.l = &empty};
    }
    if (cap) {
        l->v = ja_alloc(&r->ar, cap * sizeof *l->v);
        l->cap = l->v ? cap : 0;
    }
    return (struct jval){.t = J_LIST, .u.l = l};
}

int jv_list_add(struct jrun *r, struct jval *list, struct jval v)
{
    struct jlist *l = list->u.l;
    if (r->failed)
        return -1;
    if (l->n == l->cap) {
        size_t cap = l->cap ? 2 * l->cap : 4;
        struct jval *n = ja_alloc(&r->ar, cap * sizeof *n);
        if (!n)
            return jinja_fail(r, 0, "out of memory");
        if (l->n)
            memcpy(n, l->v, l->n * sizeof *n);
        l->v = n, l->cap = cap;
    }
    l->v[l->n++] = v;
    return 0;
}

struct jval jv_dict_new(struct jrun *r, size_t cap)
{
    struct jdict *d = ja_alloc(&r->ar, sizeof *d);
    if (!d) {
        jinja_fail(r, 0, "out of memory");
        static struct jdict empty;
        return (struct jval){.t = J_DICT, .u.d = &empty};
    }
    if (cap) {
        d->k = ja_alloc(&r->ar, cap * sizeof *d->k);
        d->v = ja_alloc(&r->ar, cap * sizeof *d->v);
        d->cap = d->k && d->v ? cap : 0;
    }
    return (struct jval){.t = J_DICT, .u.d = d};
}

const struct jval *jv_dict_get(const struct jval *dict, const char *k, size_t n)
{
    if (dict->t != J_DICT)
        return NULL;
    const struct jdict *d = dict->u.d;
    for (size_t i = 0; i < d->n; i++)
        if (d->k[i].n == n && !memcmp(d->k[i].p, k, n))
            return &d->v[i];
    return NULL;
}

int jv_dict_set(struct jrun *r, struct jval *dict, struct jstr k, struct jval v)
{
    struct jdict *d = dict->u.d;
    if (r->failed)
        return -1;
    for (size_t i = 0; i < d->n; i++)
        if (d->k[i].n == k.n && !memcmp(d->k[i].p, k.p, k.n)) {
            d->v[i] = v;
            return 0;
        }
    if (d->n == d->cap) {
        size_t cap = d->cap ? 2 * d->cap : 4;
        struct jstr *nk = ja_alloc(&r->ar, cap * sizeof *nk);
        struct jval *nv = ja_alloc(&r->ar, cap * sizeof *nv);
        if (!nk || !nv)
            return jinja_fail(r, 0, "out of memory");
        if (d->n) {
            memcpy(nk, d->k, d->n * sizeof *nk);
            memcpy(nv, d->v, d->n * sizeof *nv);
        }
        d->k = nk, d->v = nv, d->cap = cap;
    }
    d->k[d->n] = k;
    d->v[d->n++] = v;
    return 0;
}

struct jval jv_from_json(struct jrun *r, const struct janas_json *j, int mark)
{
    switch (j->type) {
    case JANAS_JSON_NULL:
        return jv_none();
    case JANAS_JSON_FALSE:
        return jv_bool(0);
    case JANAS_JSON_TRUE:
        return jv_bool(1);
    case JANAS_JSON_NUMBER: {
        char t[64];
        size_t n = j->n < sizeof t - 1 ? j->n : sizeof t - 1;
        memcpy(t, j->s, n);
        t[n] = 0;
        if (!strpbrk(t, ".eE")) {
            char *end;
            long long v = strtoll(t, &end, 10);
            if (!*end && v != LLONG_MAX && v != LLONG_MIN)
                return jv_int(v);
        }
        return jv_float(strtod(t, NULL));
    }
    case JANAS_JSON_STRING:
        return jv_str(js_make(r, j->s, j->n, mark));
    case JANAS_JSON_ARRAY: {
        struct jval l = jv_list_new(r, j->n);
        for (const struct janas_json *e = j->child; e && !r->failed;
             e = e->next)
            jv_list_add(r, &l, jv_from_json(r, e, mark));
        return l;
    }
    case JANAS_JSON_OBJECT: {
        struct jval d = jv_dict_new(r, j->n);
        for (const struct janas_json *e = j->child; e && !r->failed;
             e = e->next)
            jv_dict_set(r, &d, js_make(r, e->key, e->key_n, mark),
                        jv_from_json(r, e, mark));
        return d;
    }
    }
    return jv_undef();
}

int jv_iter(struct jrun *r, const struct jval *v, struct jval *out)
{
    if (v->t == J_LIST) {
        *out = *v;
        return 0;
    }
    if (v->t == J_DICT) {
        *out = jv_list_new(r, v->u.d->n);
        for (size_t i = 0; i < v->u.d->n; i++)
            jv_list_add(r, out, jv_str(v->u.d->k[i]));
        return r->failed ? -1 : 0;
    }
    if (v->t == J_STR) {
        struct jstr s = v->u.s;
        *out = jv_list_new(r, s.n);
        size_t i = 0;
        while (i < s.n) {
            size_t a = i;
            u8_get(s.p, s.n, &i);
            jv_list_add(r, out, jv_str(js_sub(s, a, i)));
        }
        return r->failed ? -1 : 0;
    }
    return -1;
}

/* ---- functions and methods */
enum {
    F_RANGE = 1,
    F_NAMESPACE,
    F_DICT,
    F_RAISE,
    F_STRFTIME,
    /* str */
    M_STRIP,
    M_LSTRIP,
    M_RSTRIP,
    M_SPLIT,
    M_RSPLIT,
    M_STARTSWITH,
    M_ENDSWITH,
    M_UPPER,
    M_LOWER,
    M_TITLE,
    M_CAPITALIZE,
    M_REPLACE,
    M_FIND,
    M_COUNT,
    M_JOIN,
    M_SPLITLINES,
    M_ISDIGIT,
    M_ISSPACE,
    /* dict */
    M_GET,
    M_ITEMS,
    M_KEYS,
    M_VALUES,
    /* list */
    M_INDEX,
};

static const struct {
    const char *name;
    int id;
} GLOBALS[] = {{"range", F_RANGE},
               {"namespace", F_NAMESPACE},
               {"dict", F_DICT},
               {"raise_exception", F_RAISE},
               {"strftime_now", F_STRFTIME}};

struct jval jinja_global(struct jstr name)
{
    for (size_t i = 0; i < sizeof GLOBALS / sizeof *GLOBALS; i++)
        if (js_eq(name, GLOBALS[i].name))
            return (struct jval){.t = J_FUNC, .u.fn = {GLOBALS[i].id, NULL}};
    return jv_undef();
}

static struct jval method(struct jrun *r, const struct jval *self, int id)
{
    struct jval *s = ja_alloc(&r->ar, sizeof *s);
    if (!s) {
        jinja_fail(r, 0, "out of memory");
        return jv_undef();
    }
    *s = *self;
    return (struct jval){.t = J_FUNC, .u.fn = {id, s}};
}

struct jval jv_attr(struct jrun *r, const struct jval *obj, struct jstr name)
{
    static const struct {
        const char *name;
        int id;
    } STR[] = {{"strip", M_STRIP},
               {"lstrip", M_LSTRIP},
               {"rstrip", M_RSTRIP},
               {"split", M_SPLIT},
               {"rsplit", M_RSPLIT},
               {"startswith", M_STARTSWITH},
               {"endswith", M_ENDSWITH},
               {"upper", M_UPPER},
               {"lower", M_LOWER},
               {"title", M_TITLE},
               {"capitalize", M_CAPITALIZE},
               {"replace", M_REPLACE},
               {"find", M_FIND},
               {"count", M_COUNT},
               {"join", M_JOIN},
               {"splitlines", M_SPLITLINES},
               {"isdigit", M_ISDIGIT},
               {"isspace", M_ISSPACE}},
      DICT[] = {{"get", M_GET},
                {"items", M_ITEMS},
                {"keys", M_KEYS},
                {"values", M_VALUES}},
      LIST[] = {{"index", M_INDEX}, {"count", M_COUNT}};
    if (obj->t == J_STR) {
        for (size_t i = 0; i < sizeof STR / sizeof *STR; i++)
            if (js_eq(name, STR[i].name))
                return method(r, obj, STR[i].id);
        return jv_undef();
    }
    if (obj->t == J_DICT) {
        /* as Jinja: the attribute first, then the item - but a namespace
           and a loop are their members alone */
        if (!obj->u.d->ns)
            for (size_t i = 0; i < sizeof DICT / sizeof *DICT; i++)
                if (js_eq(name, DICT[i].name) &&
                    !jv_dict_get(obj, name.p, name.n))
                    return method(r, obj, DICT[i].id);
        const struct jval *v = jv_dict_get(obj, name.p, name.n);
        if (v)
            return *v;
        for (size_t i = 0; i < sizeof DICT / sizeof *DICT; i++)
            if (js_eq(name, DICT[i].name))
                return method(r, obj, DICT[i].id);
        return jv_undef();
    }
    if (obj->t == J_LIST) {
        for (size_t i = 0; i < sizeof LIST / sizeof *LIST; i++)
            if (js_eq(name, LIST[i].name))
                return method(r, obj, LIST[i].id);
    }
    return jv_undef();
}

static int64_t as_int(const struct jval *v)
{
    return v->t == J_INT     ? v->u.i
           : v->t == J_BOOL  ? v->u.b
           : v->t == J_FLOAT ? (int64_t)v->u.f
                             : 0;
}

struct jval jv_index(struct jrun *r, const struct jval *obj,
                     const struct jval *idx, int line)
{
    if (obj->t == J_DICT) {
        struct jstr k = jv_tostr(r, idx);
        const struct jval *v = jv_dict_get(obj, k.p, k.n);
        return v ? *v : jv_undef();
    }
    if ((obj->t == J_LIST || obj->t == J_STR) &&
        (idx->t == J_INT || idx->t == J_BOOL)) {
        int64_t i = as_int(idx);
        size_t n =
            obj->t == J_LIST ? obj->u.l->n : u8_len(obj->u.s.p, obj->u.s.n);
        if (i < 0)
            i += (int64_t)n;
        if (i < 0 || (size_t)i >= n)
            return jv_undef();
        if (obj->t == J_LIST)
            return obj->u.l->v[i];
        size_t a = u8_at(obj->u.s.p, obj->u.s.n, (size_t)i);
        size_t b = u8_at(obj->u.s.p, obj->u.s.n, (size_t)i + 1);
        return jv_str(js_sub(obj->u.s, a, b));
    }
    if (idx->t == J_STR)
        return jv_attr(r, obj, idx->u.s);
    (void)line;
    return jv_undef();
}

struct jval jv_slice(struct jrun *r, const struct jval *obj,
                     const struct jval *a, const struct jval *b,
                     const struct jval *c, int line)
{
    if (obj->t != J_LIST && obj->t != J_STR) {
        if (obj->t == J_UNDEF || obj->t == J_NONE)
            return jv_undef();
        jinja_fail(r, line, "a slice of what is not a list or a string");
        return jv_undef();
    }
    int64_t n = obj->t == J_LIST ? (int64_t)obj->u.l->n
                                 : (int64_t)u8_len(obj->u.s.p, obj->u.s.n);
    int64_t step = c->t == J_NONE ? 1 : as_int(c);
    if (!step) {
        jinja_fail(r, line, "a slice's step is 0");
        return jv_undef();
    }
    int64_t lo, hi;
    /* Python's slice.indices */
    if (step > 0) {
        lo = a->t == J_NONE ? 0 : as_int(a);
        hi = b->t == J_NONE ? n : as_int(b);
        if (lo < 0)
            lo = lo + n < 0 ? 0 : lo + n;
        if (lo > n)
            lo = n;
        if (hi < 0)
            hi = hi + n < 0 ? 0 : hi + n;
        if (hi > n)
            hi = n;
    } else {
        lo = a->t == J_NONE ? n - 1 : as_int(a);
        hi = b->t == J_NONE ? -1 : as_int(b);
        if (a->t != J_NONE) {
            if (lo < 0)
                lo = lo + n < 0 ? -1 : lo + n;
            if (lo >= n)
                lo = n - 1;
        }
        if (b->t != J_NONE) {
            if (hi < 0)
                hi = hi + n < 0 ? -1 : hi + n;
            if (hi >= n)
                hi = n - 1;
        }
    }
    if (obj->t == J_LIST) {
        struct jval l = jv_list_new(r, 0);
        for (int64_t i = lo; step > 0 ? i < hi : i > hi; i += step)
            jv_list_add(r, &l, obj->u.l->v[i]);
        return l;
    }
    struct jstr s = obj->u.s;
    if (step == 1) {
        if (hi <= lo)
            return jv_cstr("");
        return jv_str(js_sub(s, u8_at(s.p, s.n, (size_t)lo),
                             u8_at(s.p, s.n, (size_t)hi)));
    }
    struct sb out = {.r = r};
    for (int64_t i = lo; step > 0 ? i < hi : i > hi; i += step) {
        size_t x = u8_at(s.p, s.n, (size_t)i),
               y = u8_at(s.p, s.n, (size_t)i + 1);
        sb_str(&out, js_sub(s, x, y));
    }
    return jv_str(sb_done(&out));
}

/* ---- arithmetic */
struct jval jv_binary(struct jrun *r, int op, const struct jval *a,
                      const struct jval *b, int line)
{
    double x = 0, y = 0;
    int na = numeric(a, &x), nb = numeric(b, &y);
    int ints = na && nb && a->t != J_FLOAT && b->t != J_FLOAT;
    switch (op) {
    case OP_CONCAT: {
        struct sb s = {.r = r};
        sb_str(&s, jv_tostr(r, a));
        sb_str(&s, jv_tostr(r, b));
        return jv_str(sb_done(&s));
    }
    case OP_EQ:
        return jv_bool(jv_equal(a, b));
    case OP_NE:
        return jv_bool(!jv_equal(a, b));
    case OP_LT:
    case OP_LE:
    case OP_GT:
    case OP_GE: {
        int c = jv_compare(a, b);
        if (c == -2 && !(a->t == b->t && a->t == J_LIST)) {
            jinja_fail(r, line, "values that cannot be compared");
            return jv_undef();
        }
        return jv_bool(op == OP_LT   ? c < 0
                       : op == OP_LE ? c <= 0
                       : op == OP_GT ? c > 0
                                     : c >= 0);
    }
    case OP_IN:
    case OP_NOTIN: {
        int in = 0;
        if (b->t == J_STR && a->t == J_STR) {
            struct jstr h = b->u.s, n = a->u.s;
            in = n.n == 0;
            for (size_t i = 0; !in && i + n.n <= h.n; i++)
                in = !memcmp(h.p + i, n.p, n.n);
        } else if (b->t == J_LIST) {
            for (size_t i = 0; !in && i < b->u.l->n; i++)
                in = jv_equal(a, &b->u.l->v[i]);
        } else if (b->t == J_DICT) {
            struct jstr k = jv_tostr(r, a);
            in = a->t == J_STR && jv_dict_get(b, k.p, k.n) != NULL;
        } else if (b->t != J_UNDEF && b->t != J_NONE) {
            jinja_fail(r, line, "'in' what is not a string, list or dict");
            return jv_undef();
        }
        return jv_bool(op == OP_IN ? in : !in);
    }
    case OP_ADD:
        if (na && nb)
            return ints ? jv_int(as_int(a) + as_int(b)) : jv_float(x + y);
        if (a->t == J_STR && b->t == J_STR) {
            struct sb s = {.r = r};
            sb_str(&s, a->u.s);
            sb_str(&s, b->u.s);
            return jv_str(sb_done(&s));
        }
        if (a->t == J_LIST && b->t == J_LIST) {
            struct jval l = jv_list_new(r, a->u.l->n + b->u.l->n);
            for (size_t i = 0; i < a->u.l->n; i++)
                jv_list_add(r, &l, a->u.l->v[i]);
            for (size_t i = 0; i < b->u.l->n; i++)
                jv_list_add(r, &l, b->u.l->v[i]);
            return l;
        }
        break;
    case OP_SUB:
        if (na && nb)
            return ints ? jv_int(as_int(a) - as_int(b)) : jv_float(x - y);
        break;
    case OP_MUL:
        if (na && nb)
            return ints ? jv_int(as_int(a) * as_int(b)) : jv_float(x * y);
        if ((b->t == J_STR || b->t == J_LIST) && a->t == J_INT) {
            const struct jval *t = a;
            a = b, b = t;
        }
        if ((a->t == J_STR || a->t == J_LIST) && (b->t == J_INT)) {
            int64_t k = b->u.i;
            if (k > 100000) {
                jinja_fail(r, line, "a repetition too long");
                return jv_undef();
            }
            if (a->t == J_STR) {
                struct sb s = {.r = r};
                for (int64_t i = 0; i < k; i++)
                    sb_str(&s, a->u.s);
                return jv_str(sb_done(&s));
            }
            struct jval l = jv_list_new(r, 0);
            for (int64_t i = 0; i < k; i++)
                for (size_t j = 0; j < a->u.l->n; j++)
                    jv_list_add(r, &l, a->u.l->v[j]);
            return l;
        }
        break;
    case OP_DIV:
        if (na && nb) {
            if (y == 0)
                break;
            return jv_float(x / y);
        }
        break;
    case OP_FLOORDIV:
    case OP_MOD:
        if (na && nb) {
            if (y == 0)
                break;
            if (ints) {
                int64_t p = as_int(a), q = as_int(b), d = p / q, m = p % q;
                if (m && ((m < 0) != (q < 0)))
                    d--, m += q;
                return jv_int(op == OP_FLOORDIV ? d : m);
            }
            double d = floor(x / y), m = x - d * y;
            return jv_float(op == OP_FLOORDIV ? d : m);
        }
        break;
    case OP_POW:
        if (na && nb) {
            if (ints && as_int(b) >= 0) {
                int64_t v = 1, base = as_int(a);
                for (int64_t i = 0; i < as_int(b) && i < 64; i++)
                    v *= base;
                return jv_int(v);
            }
            return jv_float(pow(x, y));
        }
        break;
    }
    jinja_fail(r, line, "an operation these values do not take");
    return jv_undef();
}

/* ---- the members' bodies */
static int arg_str(const struct jval *args, size_t n, size_t i,
                   struct jstr *out)
{
    if (i >= n || args[i].t != J_STR)
        return 0;
    *out = args[i].u.s;
    return 1;
}

/* strip: of whitespace, or of the characters of chars */
static struct jstr strip(struct jstr s, const struct jstr *chars, int left,
                         int right)
{
    size_t a = 0, b = s.n;
    for (;;) { /* from the left */
        if (!left || a >= b)
            break;
        size_t i = a;
        uint32_t c = u8_get(s.p, b, &i);
        int in = 0;
        if (chars) {
            size_t k = 0;
            while (k < chars->n && !in)
                in = u8_get(chars->p, chars->n, &k) == c;
        } else
            in = u_space(c);
        if (!in)
            break;
        a = i;
    }
    while (right && b > a) {
        size_t st = b - 1;
        while (st > a && ((unsigned char)s.p[st] & 0xc0) == 0x80)
            st--;
        size_t i = st;
        uint32_t c = u8_get(s.p, b, &i);
        int in = 0;
        if (chars) {
            size_t k = 0;
            while (k < chars->n && !in)
                in = u8_get(chars->p, chars->n, &k) == c;
        } else
            in = u_space(c);
        if (!in)
            break;
        b = st;
    }
    return js_sub(s, a, b);
}

static struct jval split(struct jrun *r, struct jstr s, const struct jval *sep,
                         int64_t max, int from_right)
{
    struct jval l = jv_list_new(r, 4);
    if (sep && sep->t == J_STR) {
        struct jstr d = sep->u.s;
        if (!d.n) {
            jinja_fail(r, 0, "split: an empty separator");
            return l;
        }
        if (from_right && max >= 0) {
            /* rsplit with a limit: the cuts from the end */
            size_t cuts[64], nc = 0, end = s.n;
            for (size_t i = s.n; i >= d.n && nc < (size_t)max && nc < 64;) {
                if (!memcmp(s.p + i - d.n, d.p, d.n)) {
                    cuts[nc++] = i - d.n;
                    i -= d.n;
                } else
                    i--;
            }
            size_t at = 0;
            for (size_t k = nc; k-- > 0;) {
                jv_list_add(r, &l, jv_str(js_sub(s, at, cuts[k])));
                at = cuts[k] + d.n;
            }
            jv_list_add(r, &l, jv_str(js_sub(s, at, end)));
            return l;
        }
        size_t at = 0, i = 0;
        int64_t done = 0;
        while (i + d.n <= s.n) {
            if ((max < 0 || done < max) && !memcmp(s.p + i, d.p, d.n)) {
                jv_list_add(r, &l, jv_str(js_sub(s, at, i)));
                i += d.n;
                at = i;
                done++;
            } else
                i++;
        }
        jv_list_add(r, &l, jv_str(js_sub(s, at, s.n)));
        return l;
    }
    /* on runs of whitespace, none empty */
    size_t i = 0;
    int64_t done = 0;
    while (i < s.n) {
        size_t k = i;
        while (k < s.n) {
            size_t q = k;
            if (!u_space(u8_get(s.p, s.n, &q)))
                break;
            k = q;
        }
        i = k;
        if (i >= s.n)
            break;
        if (max >= 0 && done >= max) {
            jv_list_add(r, &l, jv_str(strip(js_sub(s, i, s.n), NULL, 0, 1)));
            break;
        }
        size_t e = i;
        while (e < s.n) {
            size_t q = e;
            if (u_space(u8_get(s.p, s.n, &q)))
                break;
            e = q;
        }
        jv_list_add(r, &l, jv_str(js_sub(s, i, e)));
        done++;
        i = e;
    }
    return l;
}

/* a letter's other case, in the alphabets the chat templates and their
   users write (Latin, Greek, Cyrillic); 0 when it has none */
static uint32_t to_upper(uint32_t c)
{
    if (c >= 'a' && c <= 'z')
        return c - 32;
    if ((c >= 0xe0 && c <= 0xfe && c != 0xf7))
        return c - 32;
    if (c == 0xff)
        return 0x178;
    if (c >= 0x100 && c <= 0x17f && c != 0x130 && c != 0x131 && c != 0x138 &&
        c != 0x149 && c != 0x17f) {
        int odd_lower = !(c >= 0x139 && c <= 0x148) && !(c >= 0x179);
        if (odd_lower ? (c & 1) : !(c & 1))
            return c - 1;
        return 0;
    }
    if (c >= 0x3b1 && c <= 0x3c9 && c != 0x3c2)
        return c - 32;
    if (c == 0x3c2)
        return 0x3a3;
    if (c >= 0x430 && c <= 0x44f)
        return c - 32;
    if (c >= 0x450 && c <= 0x45f)
        return c - 80;
    return 0;
}
static uint32_t to_lower(uint32_t c)
{
    if (c >= 'A' && c <= 'Z')
        return c + 32;
    if (c >= 0xc0 && c <= 0xde && c != 0xd7)
        return c + 32;
    if (c == 0x178)
        return 0xff;
    if (c >= 0x100 && c <= 0x17f && c != 0x130 && c != 0x131 && c != 0x138 &&
        c != 0x149 && c != 0x17f) {
        int odd_lower = !(c >= 0x139 && c <= 0x148) && !(c >= 0x179);
        if (odd_lower ? !(c & 1) : (c & 1))
            return c + 1;
        return 0;
    }
    if (c >= 0x391 && c <= 0x3a9 && c != 0x3a2)
        return c + 32;
    if (c >= 0x410 && c <= 0x42f)
        return c + 32;
    if (c >= 0x400 && c <= 0x40f)
        return c + 80;
    return 0;
}
static int is_letter(uint32_t c)
{
    return to_upper(c) || to_lower(c) || c == 0xdf;
}

static void put_cp(struct sb *b, uint32_t c, uint8_t mk)
{
    char o[4];
    int n;
    if (c < 0x80)
        o[0] = (char)c, n = 1;
    else if (c < 0x800)
        o[0] = (char)(0xc0 | c >> 6), o[1] = (char)(0x80 | (c & 63)), n = 2;
    else if (c < 0x10000)
        o[0] = (char)(0xe0 | c >> 12), o[1] = (char)(0x80 | ((c >> 6) & 63)),
        o[2] = (char)(0x80 | (c & 63)), n = 3;
    else
        o[0] = (char)(0xf0 | c >> 18), o[1] = (char)(0x80 | ((c >> 12) & 63)),
        o[2] = (char)(0x80 | ((c >> 6) & 63)), o[3] = (char)(0x80 | (c & 63)),
        n = 4;
    sb_put(b, o, (size_t)n, NULL, mk);
}

/* 0 upper, 1 lower, 2 title, 3 capitalize; each character keeps the mark
   of its first byte */
static struct jstr ascii_case(struct jrun *r, struct jstr s, int mode)
{
    struct sb b = {.r = r};
    int start = 1, first = 1;
    size_t i = 0;
    while (i < s.n) {
        size_t at = i;
        uint32_t c = u8_get(s.p, s.n, &i);
        uint8_t mk = s.m ? s.m[at] : 0;
        int up = mode == 0 || (mode == 2 && start) || (mode == 3 && first);
        uint32_t o = up ? to_upper(c) : to_lower(c);
        if (c == 0xdf && up && mode == 0) {
            put_cp(&b, 'S', mk);
            put_cp(&b, 'S', mk);
        } else if (o)
            put_cp(&b, o, mk);
        else
            sb_put(&b, s.p + at, i - at, s.m ? s.m + at : NULL, 0);
        if (mode == 2)
            start = !is_letter(c) && !(c >= '0' && c <= '9') && c != '\'';
        first = 0;
    }
    return sb_done(&b);
}

static struct jstr replace(struct jrun *r, struct jstr s, struct jstr old,
                           struct jstr new, int64_t count)
{
    struct sb b = {.r = r};
    size_t i = 0;
    int64_t done = 0;
    if (!old.n) { /* Python: between every character */
        while (i <= s.n && (count < 0 || done < count)) {
            sb_str(&b, new);
            done++;
            if (i == s.n)
                break;
            size_t k = i;
            u8_get(s.p, s.n, &k);
            sb_str(&b, js_sub(s, i, k));
            i = k;
        }
        sb_str(&b, js_sub(s, i, s.n));
        return sb_done(&b);
    }
    while (i < s.n) {
        if ((count < 0 || done < count) && i + old.n <= s.n &&
            !memcmp(s.p + i, old.p, old.n)) {
            sb_str(&b, new);
            i += old.n;
            done++;
        } else {
            sb_str(&b, js_sub(s, i, i + 1));
            i++;
        }
    }
    return sb_done(&b);
}

static struct jval join(struct jrun *r, const struct jval *items,
                        struct jstr sep, const struct jstr *attr)
{
    struct jval l;
    if (jv_iter(r, items, &l) != 0) {
        jinja_fail(r, 0, "join: not a list");
        return jv_undef();
    }
    struct sb b = {.r = r};
    for (size_t i = 0; i < l.u.l->n; i++) {
        if (i)
            sb_str(&b, sep);
        struct jval v = l.u.l->v[i];
        if (attr)
            v = jv_attr(r, &v, *attr);
        sb_str(&b, jv_tostr(r, &v));
    }
    return jv_str(sb_done(&b));
}

static const struct jval *kwarg(const struct jstr *kw, const struct jval *kwv,
                                size_t nkw, const char *name)
{
    for (size_t i = 0; i < nkw; i++)
        if (js_eq(kw[i], name))
            return &kwv[i];
    return NULL;
}

struct jval jinja_call(struct jrun *r, const struct jval *fn,
                       const struct jval *args, size_t n, const struct jstr *kw,
                       const struct jval *kwv, size_t nkw, int line)
{
    if (fn->t != J_FUNC) {
        jinja_fail(r, line, "a call of what is not a function");
        return jv_undef();
    }
    const struct jval *self = fn->u.fn.self;
    struct jstr a0, a1;
    switch (fn->u.fn.id) {
    case F_RANGE: {
        int64_t lo = 0, hi = 0, st = 1;
        if (n == 1)
            hi = as_int(&args[0]);
        else if (n >= 2)
            lo = as_int(&args[0]), hi = as_int(&args[1]);
        if (n >= 3)
            st = as_int(&args[2]);
        if (!st || (hi - lo) / st > 1000000) {
            jinja_fail(r, line, "range: too many, or a step of 0");
            return jv_undef();
        }
        struct jval l = jv_list_new(r, 0);
        for (int64_t i = lo; st > 0 ? i < hi : i > hi; i += st)
            jv_list_add(r, &l, jv_int(i));
        return l;
    }
    case F_NAMESPACE:
    case F_DICT: {
        struct jval d = jv_dict_new(r, nkw);
        d.u.d->ns = fn->u.fn.id == F_NAMESPACE;
        if (n && args[0].t == J_DICT)
            for (size_t i = 0; i < args[0].u.d->n; i++)
                jv_dict_set(r, &d, args[0].u.d->k[i], args[0].u.d->v[i]);
        for (size_t i = 0; i < nkw; i++)
            jv_dict_set(r, &d, kw[i], kwv[i]);
        return d;
    }
    case F_RAISE: {
        struct jstr m =
            n ? jv_tostr(r, &args[0]) : (struct jstr){.p = "", .n = 0};
        jinja_fail(r, 0, "%.*s", (int)m.n, m.p);
        return jv_undef();
    }
    case F_STRFTIME: {
        char out[256] = "";
        struct jstr f;
        if (arg_str(args, n, 0, &f) && f.n < 200) {
            char fmt[256];
            memcpy(fmt, f.p, f.n);
            fmt[f.n] = 0;
            time_t now = time(NULL);
            struct tm tm;
#ifdef _WIN32
            localtime_s(&tm, &now);
#else
            localtime_r(&now, &tm);
#endif
            strftime(out, sizeof out, fmt, &tm);
        }
        return jv_str(js_make(r, out, strlen(out), 0));
    }
    }
    if (!self)
        return jv_undef();
    if (self->t == J_STR) {
        struct jstr s = self->u.s;
        switch (fn->u.fn.id) {
        case M_STRIP:
        case M_LSTRIP:
        case M_RSTRIP: {
            int has = arg_str(args, n, 0, &a0);
            int id = fn->u.fn.id;
            return jv_str(
                strip(s, has ? &a0 : NULL, id != M_RSTRIP, id != M_LSTRIP));
        }
        case M_SPLIT:
        case M_RSPLIT: {
            const struct jval *sep = n ? &args[0] : kwarg(kw, kwv, nkw, "sep");
            const struct jval *mx =
                n > 1 ? &args[1] : kwarg(kw, kwv, nkw, "maxsplit");
            if (sep && sep->t == J_NONE)
                sep = NULL;
            return split(r, s, sep, mx ? as_int(mx) : -1,
                         fn->u.fn.id == M_RSPLIT);
        }
        case M_STARTSWITH:
        case M_ENDSWITH: {
            int end = fn->u.fn.id == M_ENDSWITH;
            if (!n)
                return jv_bool(0);
            const struct jval *opts = &args[0];
            size_t k = opts->t == J_LIST ? opts->u.l->n : 1;
            for (size_t i = 0; i < k; i++) {
                const struct jval *o =
                    opts->t == J_LIST ? &opts->u.l->v[i] : opts;
                if (o->t != J_STR)
                    continue;
                struct jstr x = o->u.s;
                if (x.n <= s.n &&
                    !memcmp(s.p + (end ? s.n - x.n : 0), x.p, x.n))
                    return jv_bool(1);
            }
            return jv_bool(0);
        }
        case M_UPPER:
            return jv_str(ascii_case(r, s, 0));
        case M_LOWER:
            return jv_str(ascii_case(r, s, 1));
        case M_TITLE:
            return jv_str(ascii_case(r, s, 2));
        case M_CAPITALIZE:
            return jv_str(ascii_case(r, s, 3));
        case M_REPLACE:
            if (arg_str(args, n, 0, &a0) && arg_str(args, n, 1, &a1))
                return jv_str(
                    replace(r, s, a0, a1, n > 2 ? as_int(&args[2]) : -1));
            break;
        case M_FIND:
            if (arg_str(args, n, 0, &a0)) {
                for (size_t i = 0; i + a0.n <= s.n; i++)
                    if (!memcmp(s.p + i, a0.p, a0.n))
                        return jv_int((int64_t)u8_len(s.p, i));
                return jv_int(-1);
            }
            break;
        case M_COUNT:
            if (arg_str(args, n, 0, &a0) && a0.n) {
                int64_t c = 0;
                for (size_t i = 0; i + a0.n <= s.n;)
                    if (!memcmp(s.p + i, a0.p, a0.n))
                        c++, i += a0.n;
                    else
                        i++;
                return jv_int(c);
            }
            break;
        case M_JOIN:
            if (n)
                return join(r, &args[0], s, NULL);
            break;
        case M_SPLITLINES: {
            struct jval l = jv_list_new(r, 4);
            size_t at = 0;
            for (size_t i = 0; i < s.n; i++)
                if (s.p[i] == '\n' || s.p[i] == '\r') {
                    jv_list_add(r, &l, jv_str(js_sub(s, at, i)));
                    if (s.p[i] == '\r' && i + 1 < s.n && s.p[i + 1] == '\n')
                        i++;
                    at = i + 1;
                }
            if (at < s.n)
                jv_list_add(r, &l, jv_str(js_sub(s, at, s.n)));
            return l;
        }
        case M_ISDIGIT:
        case M_ISSPACE: {
            int ok = s.n > 0;
            for (size_t i = 0; i < s.n && ok; i++)
                ok = fn->u.fn.id == M_ISDIGIT ? s.p[i] >= '0' && s.p[i] <= '9'
                                              : u_space((unsigned char)s.p[i]);
            return jv_bool(ok);
        }
        }
    } else if (self->t == J_DICT) {
        const struct jdict *d = self->u.d;
        switch (fn->u.fn.id) {
        case M_GET: {
            if (!n)
                break;
            struct jstr k = jv_tostr(r, &args[0]);
            const struct jval *v = jv_dict_get(self, k.p, k.n);
            return v ? *v : n > 1 ? args[1] : jv_none();
        }
        case M_ITEMS: {
            struct jval l = jv_list_new(r, d->n);
            for (size_t i = 0; i < d->n; i++) {
                struct jval pair = jv_list_new(r, 2);
                jv_list_add(r, &pair, jv_str(d->k[i]));
                jv_list_add(r, &pair, d->v[i]);
                jv_list_add(r, &l, pair);
            }
            return l;
        }
        case M_KEYS: {
            struct jval l;
            jv_iter(r, self, &l);
            return l;
        }
        case M_VALUES: {
            struct jval l = jv_list_new(r, d->n);
            for (size_t i = 0; i < d->n; i++)
                jv_list_add(r, &l, d->v[i]);
            return l;
        }
        }
    } else if (self->t == J_LIST) {
        const struct jlist *l = self->u.l;
        if (fn->u.fn.id == M_INDEX && n) {
            for (size_t i = 0; i < l->n; i++)
                if (jv_equal(&l->v[i], &args[0]))
                    return jv_int((int64_t)i);
            jinja_fail(r, line, "index: not in the list");
            return jv_undef();
        }
        if (fn->u.fn.id == M_COUNT && n) {
            int64_t c = 0;
            for (size_t i = 0; i < l->n; i++)
                c += jv_equal(&l->v[i], &args[0]);
            return jv_int(c);
        }
    }
    jinja_fail(r, line, "a method called with arguments it does not take");
    return jv_undef();
}

/* ---- tests */
int jinja_test(struct jrun *r, struct jstr name, const struct jval *v,
               const struct jval *args, size_t n, int line)
{
    const struct jval *a = n ? &args[0] : NULL;
    double x = 0;
    if (js_eq(name, "defined"))
        return v->t != J_UNDEF;
    if (js_eq(name, "undefined"))
        return v->t == J_UNDEF;
    if (js_eq(name, "none"))
        return v->t == J_NONE;
    if (js_eq(name, "boolean"))
        return v->t == J_BOOL;
    if (js_eq(name, "true"))
        return v->t == J_BOOL && v->u.b;
    if (js_eq(name, "false"))
        return v->t == J_BOOL && !v->u.b;
    if (js_eq(name, "integer"))
        return v->t == J_INT;
    if (js_eq(name, "float"))
        return v->t == J_FLOAT;
    if (js_eq(name, "number"))
        return v->t == J_INT || v->t == J_FLOAT || v->t == J_BOOL;
    if (js_eq(name, "string"))
        return v->t == J_STR;
    if (js_eq(name, "mapping"))
        return v->t == J_DICT;
    if (js_eq(name, "iterable") || js_eq(name, "sequence"))
        return v->t == J_LIST || v->t == J_DICT || v->t == J_STR;
    if (js_eq(name, "callable"))
        return v->t == J_FUNC || v->t == J_MACRO;
    if (js_eq(name, "even") || js_eq(name, "odd")) {
        if (v->t != J_INT)
            return jinja_fail(r, line, "even/odd of what is not an integer");
        return (v->u.i % 2 == 0) == js_eq(name, "even");
    }
    if (js_eq(name, "divisibleby")) {
        int64_t d = a ? as_int(a) : 0;
        return d && v->t == J_INT && v->u.i % d == 0;
    }
    if (js_eq(name, "lower") || js_eq(name, "upper")) {
        if (v->t != J_STR)
            return 0;
        int up = js_eq(name, "upper"), cased = 0;
        for (size_t i = 0; i < v->u.s.n; i++) {
            char c = v->u.s.p[i];
            if ((up && c >= 'a' && c <= 'z') || (!up && c >= 'A' && c <= 'Z'))
                return 0;
            cased |= (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        }
        return cased;
    }
    if (js_eq(name, "sameas"))
        return a && v->t == a->t &&
               (v->t == J_NONE || v->t == J_UNDEF ||
                (v->t == J_BOOL && v->u.b == a->u.b));
    if (a && (js_eq(name, "eq") || js_eq(name, "equalto") || js_eq(name, "==")))
        return jv_equal(v, a);
    if (a && (js_eq(name, "ne") || js_eq(name, "!=")))
        return !jv_equal(v, a);
    if (a && (js_eq(name, "lt") || js_eq(name, "le") || js_eq(name, "gt") ||
              js_eq(name, "ge") || js_eq(name, "lessthan") ||
              js_eq(name, "greaterthan"))) {
        int c = jv_compare(v, a);
        if (c == -2)
            return jinja_fail(r, line, "values that cannot be compared");
        if (js_eq(name, "lt") || js_eq(name, "lessthan"))
            return c < 0;
        if (js_eq(name, "le"))
            return c <= 0;
        if (js_eq(name, "gt") || js_eq(name, "greaterthan"))
            return c > 0;
        return c >= 0;
    }
    if (a && js_eq(name, "in")) {
        struct jval b = jv_binary(r, OP_IN, v, a, line);
        return r->failed ? -1 : b.u.b;
    }
    (void)x;
    return jinja_fail(r, line, "no test named %.*s", (int)name.n, name.p);
}

/* ---- filters */
static int cmp_items(struct jrun *r, const struct jval *a, const struct jval *b,
                     int nocase)
{
    if (nocase && a->t == J_STR && b->t == J_STR) {
        struct jstr x = a->u.s, y = b->u.s;
        for (size_t i = 0; i < x.n && i < y.n; i++) {
            int c = (unsigned char)x.p[i], d = (unsigned char)y.p[i];
            if (c >= 'A' && c <= 'Z')
                c += 32;
            if (d >= 'A' && d <= 'Z')
                d += 32;
            if (c != d)
                return c < d ? -1 : 1;
        }
        return x.n < y.n ? -1 : x.n > y.n;
    }
    int c = jv_compare(a, b);
    if (c == -2) {
        jinja_fail(r, 0, "sort: values that cannot be compared");
        return 0;
    }
    return c;
}

/* a stable sort of l by key (attr: a member of each), reversed or not */
static struct jval sorted(struct jrun *r, struct jval l,
                          const struct jstr *attr, int rev, int nocase)
{
    size_t n = l.u.l->n;
    struct jval out = jv_list_new(r, n);
    for (size_t i = 0; i < n; i++)
        jv_list_add(r, &out, l.u.l->v[i]);
    struct jval *v = out.u.l->v;
    /* insertion sort: stable, and the lists are short */
    for (size_t i = 1; i < n && !r->failed; i++) {
        struct jval x = v[i];
        struct jval kx = attr ? jv_attr(r, &x, *attr) : x;
        size_t j = i;
        while (j > 0) {
            struct jval ky = attr ? jv_attr(r, &v[j - 1], *attr) : v[j - 1];
            int c = cmp_items(r, &ky, &kx, nocase);
            if (rev ? c >= 0 : c <= 0)
                break;
            v[j] = v[j - 1];
            j--;
        }
        v[j] = x;
    }
    return out;
}

struct jval jinja_filter(struct jrun *r, struct jstr name, const struct jval *v,
                         const struct jval *args, size_t n,
                         const struct jstr *kw, const struct jval *kwv,
                         size_t nkw, int line)
{
    const struct jval *a0 = n ? &args[0] : NULL;
    struct jstr s0;
    if (js_eq(name, "safe"))
        return *v;
    if (js_eq(name, "string"))
        return jv_str(jv_tostr(r, v));
    if (js_eq(name, "trim")) {
        struct jstr s = jv_tostr(r, v);
        int has = arg_str(args, n, 0, &s0);
        const struct jval *c = kwarg(kw, kwv, nkw, "chars");
        if (c && c->t == J_STR)
            s0 = c->u.s, has = 1;
        return jv_str(strip(s, has ? &s0 : NULL, 1, 1));
    }
    if (js_eq(name, "upper") || js_eq(name, "lower") || js_eq(name, "title") ||
        js_eq(name, "capitalize")) {
        int mode = js_eq(name, "upper")   ? 0
                   : js_eq(name, "lower") ? 1
                   : js_eq(name, "title") ? 2
                                          : 3;
        return jv_str(ascii_case(r, jv_tostr(r, v), mode));
    }
    if (js_eq(name, "length") || js_eq(name, "count")) {
        if (v->t == J_STR)
            return jv_int((int64_t)u8_len(v->u.s.p, v->u.s.n));
        if (v->t == J_LIST)
            return jv_int((int64_t)v->u.l->n);
        if (v->t == J_DICT)
            return jv_int((int64_t)v->u.d->n);
        if (v->t == J_UNDEF)
            return jv_int(0);
        jinja_fail(r, line, "length of what has none");
        return jv_undef();
    }
    if (js_eq(name, "default") || js_eq(name, "d")) {
        const struct jval *b =
            n > 1 ? &args[1] : kwarg(kw, kwv, nkw, "boolean");
        int use = v->t == J_UNDEF || (b && jv_truthy(b) && !jv_truthy(v));
        return use ? (a0 ? *a0 : jv_cstr("")) : *v;
    }
    if (js_eq(name, "tojson")) {
        const struct jval *ind = a0 ? a0 : kwarg(kw, kwv, nkw, "indent");
        return jv_str(tojson_with(r, v, ind, kwarg(kw, kwv, nkw, "separators"),
                                  kwarg(kw, kwv, nkw, "sort_keys")));
    }
    if (js_eq(name, "join")) {
        struct jstr sep = {.p = "", .n = 0};
        if (a0 && a0->t == J_STR)
            sep = a0->u.s;
        const struct jval *at = kwarg(kw, kwv, nkw, "attribute");
        if (n > 1 && args[1].t == J_STR)
            at = &args[1];
        return join(r, v, sep, at && at->t == J_STR ? &at->u.s : NULL);
    }
    if (js_eq(name, "replace")) {
        struct jstr s1;
        if (!arg_str(args, n, 0, &s0) || !arg_str(args, n, 1, &s1)) {
            jinja_fail(r, line, "replace: two strings");
            return jv_undef();
        }
        return jv_str(
            replace(r, jv_tostr(r, v), s0, s1, n > 2 ? as_int(&args[2]) : -1));
    }
    if (js_eq(name, "first") || js_eq(name, "last")) {
        struct jval l;
        if (jv_iter(r, v, &l) != 0 || !l.u.l->n)
            return jv_undef();
        return l.u.l->v[js_eq(name, "first") ? 0 : l.u.l->n - 1];
    }
    if (js_eq(name, "list")) {
        struct jval l;
        if (jv_iter(r, v, &l) != 0) {
            jinja_fail(r, line, "list of what is not a sequence");
            return jv_undef();
        }
        return l;
    }
    if (js_eq(name, "items")) {
        if (v->t == J_UNDEF)
            return jv_list_new(r, 0);
        if (v->t != J_DICT) {
            jinja_fail(r, line, "items of what is not a mapping");
            return jv_undef();
        }
        struct jval m = jv_attr(r, v, (struct jstr){.p = "items", .n = 5});
        struct jval fake = {.t = J_FUNC, .u.fn = {M_ITEMS, (struct jval *)v}};
        (void)m;
        return jinja_call(r, &fake, NULL, 0, NULL, NULL, 0, line);
    }
    if (js_eq(name, "dictsort")) {
        if (v->t != J_DICT) {
            jinja_fail(r, line, "dictsort of what is not a mapping");
            return jv_undef();
        }
        struct jval fake = {.t = J_FUNC, .u.fn = {M_ITEMS, (struct jval *)v}};
        struct jval items = jinja_call(r, &fake, NULL, 0, NULL, NULL, 0, line);
        const struct jval *cs = kwarg(kw, kwv, nkw, "case_sensitive");
        const struct jval *by = kwarg(kw, kwv, nkw, "by");
        const struct jval *rv = kwarg(kw, kwv, nkw, "reverse");
        if (n > 0)
            cs = &args[0];
        if (n > 1)
            by = &args[1];
        if (n > 2)
            rv = &args[2];
        struct jstr key = {
            .p = by && by->t == J_STR && js_eq(by->u.s, "value") ? "1" : "0",
            .n = 1};
        /* sorted by the pair's first or second, as a list index */
        size_t cnt = items.u.l->n;
        struct jval keyed = jv_list_new(r, cnt);
        for (size_t i = 0; i < cnt; i++)
            jv_list_add(r, &keyed, items.u.l->v[i]);
        struct jval *e = keyed.u.l->v;
        int nocase = !(cs && jv_truthy(cs)), rev = rv && jv_truthy(rv);
        size_t which = key.p[0] == '1';
        for (size_t i = 1; i < cnt; i++) {
            struct jval x = e[i];
            size_t j = i;
            while (j > 0) {
                int c = cmp_items(r, &e[j - 1].u.l->v[which], &x.u.l->v[which],
                                  nocase);
                if (rev ? c >= 0 : c <= 0)
                    break;
                e[j] = e[j - 1];
                j--;
            }
            e[j] = x;
        }
        return keyed;
    }
    if (js_eq(name, "sort")) {
        struct jval l;
        if (jv_iter(r, v, &l) != 0) {
            jinja_fail(r, line, "sort of what is not a sequence");
            return jv_undef();
        }
        const struct jval *rv =
            n > 0 ? &args[0] : kwarg(kw, kwv, nkw, "reverse");
        const struct jval *cs =
            n > 1 ? &args[1] : kwarg(kw, kwv, nkw, "case_sensitive");
        const struct jval *at =
            n > 2 ? &args[2] : kwarg(kw, kwv, nkw, "attribute");
        return sorted(r, l, at && at->t == J_STR ? &at->u.s : NULL,
                      rv && jv_truthy(rv), !(cs && jv_truthy(cs)));
    }
    if (js_eq(name, "reverse")) {
        if (v->t == J_STR) {
            struct jval none = jv_none(), step = jv_int(-1);
            return jv_slice(r, v, &none, &none, &step, line);
        }
        struct jval l;
        if (jv_iter(r, v, &l) != 0) {
            jinja_fail(r, line, "reverse of what is not a sequence");
            return jv_undef();
        }
        struct jval out = jv_list_new(r, l.u.l->n);
        for (size_t i = l.u.l->n; i-- > 0;)
            jv_list_add(r, &out, l.u.l->v[i]);
        return out;
    }
    if (js_eq(name, "unique")) {
        struct jval l, out = jv_list_new(r, 4);
        if (jv_iter(r, v, &l) != 0)
            return out;
        for (size_t i = 0; i < l.u.l->n; i++) {
            int seen = 0;
            for (size_t k = 0; k < out.u.l->n && !seen; k++)
                seen = jv_equal(&out.u.l->v[k], &l.u.l->v[i]);
            if (!seen)
                jv_list_add(r, &out, l.u.l->v[i]);
        }
        return out;
    }
    if (js_eq(name, "map") || js_eq(name, "select") || js_eq(name, "reject") ||
        js_eq(name, "selectattr") || js_eq(name, "rejectattr")) {
        struct jval l, out = jv_list_new(r, 4);
        if (v->t == J_UNDEF)
            return out;
        if (jv_iter(r, v, &l) != 0) {
            jinja_fail(r, line, "%.*s of what is not a sequence", (int)name.n,
                       name.p);
            return jv_undef();
        }
        int is_map = js_eq(name, "map");
        int by_attr = js_eq(name, "selectattr") || js_eq(name, "rejectattr");
        int reject = js_eq(name, "reject") || js_eq(name, "rejectattr");
        const struct jval *attr = is_map ? kwarg(kw, kwv, nkw, "attribute")
                                  : by_attr && n ? &args[0]
                                                 : NULL;
        size_t at = by_attr ? 1 : 0; /* the test's or filter's name */
        for (size_t i = 0; i < l.u.l->n && !r->failed; i++) {
            struct jval x = l.u.l->v[i];
            struct jval y =
                attr && attr->t == J_STR ? jv_attr(r, &x, attr->u.s) : x;
            if (is_map) {
                if (attr) {
                    if (y.t == J_UNDEF) {
                        const struct jval *d = kwarg(kw, kwv, nkw, "default");
                        if (d)
                            y = *d;
                    }
                    jv_list_add(r, &out, y);
                } else if (n && args[0].t == J_STR)
                    jv_list_add(r, &out,
                                jinja_filter(r, args[0].u.s, &x, args + 1,
                                             n - 1, NULL, NULL, 0, line));
                continue;
            }
            int t;
            if (n > at && args[at].t == J_STR)
                t = jinja_test(r, args[at].u.s, &y, args + at + 1, n - at - 1,
                               line);
            else
                t = jv_truthy(&y);
            if (t < 0)
                return jv_undef();
            if (t != reject)
                jv_list_add(r, &out, x);
        }
        return out;
    }
    if (js_eq(name, "e") || js_eq(name, "escape")) {
        struct jstr s = jv_tostr(r, v);
        struct sb b = {.r = r};
        for (size_t i = 0; i < s.n; i++) {
            uint8_t mk = s.m ? s.m[i] : 0;
            const char *e = s.p[i] == '&'    ? "&amp;"
                            : s.p[i] == '<'  ? "&lt;"
                            : s.p[i] == '>'  ? "&gt;"
                            : s.p[i] == '"'  ? "&#34;"
                            : s.p[i] == '\'' ? "&#39;"
                                             : NULL;
            if (e)
                sb_put(&b, e, strlen(e), NULL, mk);
            else
                sb_put(&b, s.p + i, 1, NULL, mk);
        }
        return jv_str(sb_done(&b));
    }
    if (js_eq(name, "int") || js_eq(name, "float")) {
        int fl = js_eq(name, "float");
        if (v->t == J_INT || v->t == J_FLOAT || v->t == J_BOOL) {
            double x = 0; /* set by numeric(): older GCCs cannot tell */
            numeric(v, &x);
            return fl ? jv_float(x) : jv_int((int64_t)x);
        }
        if (v->t == J_STR) {
            char t[64];
            size_t k = v->u.s.n < 63 ? v->u.s.n : 63;
            memcpy(t, v->u.s.p, k);
            t[k] = 0;
            char *end;
            double x = strtod(t, &end);
            if (end != t)
                return fl ? jv_float(x) : jv_int((int64_t)x);
        }
        return a0 ? *a0 : fl ? jv_float(0) : jv_int(0);
    }
    if (js_eq(name, "abs")) {
        if (v->t == J_INT)
            return jv_int(v->u.i < 0 ? -v->u.i : v->u.i);
        if (v->t == J_FLOAT)
            return jv_float(fabs(v->u.f));
    }
    if (js_eq(name, "round")) {
        double x = 0;
        if (numeric(v, &x)) {
            int p = a0 ? (int)as_int(a0) : 0;
            double k = pow(10, p);
            return jv_float(round(x * k) / k);
        }
    }
    if (js_eq(name, "sum") || js_eq(name, "min") || js_eq(name, "max")) {
        struct jval l;
        if (jv_iter(r, v, &l) != 0) {
            jinja_fail(r, line, "%.*s of what is not a sequence", (int)name.n,
                       name.p);
            return jv_undef();
        }
        if (js_eq(name, "sum")) {
            struct jval acc = jv_int(0);
            const struct jval *at = kwarg(kw, kwv, nkw, "attribute");
            for (size_t i = 0; i < l.u.l->n && !r->failed; i++) {
                struct jval x = l.u.l->v[i];
                if (at && at->t == J_STR)
                    x = jv_attr(r, &x, at->u.s);
                acc = jv_binary(r, OP_ADD, &acc, &x, line);
            }
            return acc;
        }
        if (!l.u.l->n)
            return jv_undef();
        struct jval best = l.u.l->v[0];
        int want = js_eq(name, "min") ? -1 : 1;
        for (size_t i = 1; i < l.u.l->n; i++)
            if (cmp_items(r, &l.u.l->v[i], &best, 1) == want)
                best = l.u.l->v[i];
        return best;
    }
    if (js_eq(name, "indent")) {
        int w = a0 ? (int)as_int(a0) : 4;
        const struct jval *f = n > 1 ? &args[1] : kwarg(kw, kwv, nkw, "first");
        const struct jval *bl = n > 2 ? &args[2] : kwarg(kw, kwv, nkw, "blank");
        struct jstr s = jv_tostr(r, v);
        struct sb b = {.r = r};
        int at_start = 1, first = 1;
        for (size_t i = 0; i < s.n; i++) {
            if (at_start) {
                int blank = s.p[i] == '\n';
                if ((!first || (f && jv_truthy(f))) &&
                    (!blank || (bl && jv_truthy(bl))))
                    for (int k = 0; k < w; k++)
                        sb_lit(&b, " ");
                at_start = 0;
                first = 0;
            }
            sb_str(&b, js_sub(s, i, i + 1));
            if (s.p[i] == '\n')
                at_start = 1;
        }
        return jv_str(sb_done(&b));
    }
    if (js_eq(name, "wordcount")) {
        struct jval w = split(r, jv_tostr(r, v), NULL, -1, 0);
        return jv_int((int64_t)w.u.l->n);
    }
    if (js_eq(name, "attr") && a0 && a0->t == J_STR)
        return jv_attr(r, v, a0->u.s);
    jinja_fail(r, line, "no filter named %.*s", (int)name.n, name.p);
    return jv_undef();
}
