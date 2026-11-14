/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * pdf.c - the text of a PDF: what its pages show, in their order, as
 * lines and paragraphs. The objects are found by scanning the file for
 * "N G obj", not through its cross-reference table, which a file edited
 * many times or cut short often has wrong; the later definition of an
 * object wins, as an update appended to the file means it to. The objects
 * kept in object streams (PDF 1.5) are read from them. The pages are
 * walked from the catalog, each with the resources it inherits; their
 * content streams, and the forms they draw (Do), are read for the text
 * operators (Tj, TJ, ', ") and the moves that make a new line or a space
 * (Td, TD, Tm, T*). The bytes of a string become characters through the
 * font's ToUnicode map when it has one, through WinAnsi otherwise; a font
 * of two-byte codes without a map cannot be read, and its text is left
 * out. Filters: FlateDecode, or none. An encrypted file is not read.
 *
 * The file is input from outside: every read is bounded by its length,
 * every nesting by a depth, every count by a limit; all that is allocated
 * for a document goes into one arena, freed at once.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/inflate.h"
#include "docs.h"

#define PDF_OBJ (1u << 21) /* object numbers */
#define PDF_DEPTH 32       /* nested arrays and dictionaries */
#define PDF_PAGES 20000
#define PDF_FORMS 4 /* forms drawn by forms */
#define PDF_STREAM DZ_MAX_FILE
#define PDF_TEXT (32 << 20) /* of a document's text */

/* ---- the arena ---- */

struct block {
    struct block *next;
    size_t used, cap;
    unsigned char data[];
};

struct arena {
    struct block *b;
    int oom;
};

static void *alloc(struct arena *a, size_t n)
{
    n = (n + 15) & ~(size_t)15;
    if (!a->b || a->b->cap - a->b->used < n) {
        size_t cap = n > 65536 ? n : 65536;
        struct block *b = malloc(sizeof(*b) + cap);
        if (!b) {
            a->oom = 1;
            return NULL;
        }
        b->next = a->b;
        b->used = 0;
        b->cap = cap;
        a->b = b;
    }
    void *p = a->b->data + a->b->used;
    a->b->used += n;
    memset(p, 0, n);
    return p;
}

static void arena_free(struct arena *a)
{
    while (a->b) {
        struct block *n = a->b->next;
        free(a->b);
        a->b = n;
    }
}

/* ---- values ---- */

enum {
    PV_NULL,
    PV_BOOL,
    PV_NUM,
    PV_STR,
    PV_NAME,
    PV_ARR,
    PV_DICT,
    PV_REF,
    PV_KW,
    PV_END
};

struct pv {
    int t;
    double num;        /* PV_NUM, PV_BOOL; PV_REF: its number */
    const uint8_t *s;  /* PV_STR (decoded), PV_NAME, PV_KW */
    size_t n;          /* their bytes; PV_ARR: items; PV_DICT: pairs */
    struct pv **items; /* PV_ARR: n; PV_DICT: 2n, key and value */
};

static struct pv NUL = {.t = PV_NULL};

struct lx {
    const uint8_t *p;
    size_t n, at;
    struct arena *a;
};

static int ws(uint8_t c)
{
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' ||
           c == 0;
}

static int delim(uint8_t c)
{
    return c == '(' || c == ')' || c == '<' || c == '>' || c == '[' ||
           c == ']' || c == '{' || c == '}' || c == '/' || c == '%';
}

static void skip(struct lx *l)
{
    while (l->at < l->n) {
        if (ws(l->p[l->at])) {
            l->at++;
        } else if (l->p[l->at] == '%') {
            while (l->at < l->n && l->p[l->at] != '\n' && l->p[l->at] != '\r')
                l->at++;
        } else {
            break;
        }
    }
}

static int hexv(uint8_t c)
{
    return c >= '0' && c <= '9'   ? c - '0'
           : c >= 'a' && c <= 'f' ? c - 'a' + 10
           : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                  : -1;
}

static struct pv *mk(struct lx *l, int t)
{
    struct pv *v = alloc(l->a, sizeof(*v));
    if (v)
        v->t = t;
    return v;
}

static struct pv *literal(struct lx *l)
{
    size_t at = ++l->at, depth = 1, k = 0;
    /* the decoded string is never longer than the source */
    size_t end = at;
    for (int esc = 0; end < l->n; end++) {
        uint8_t c = l->p[end];
        if (esc)
            esc = 0;
        else if (c == '\\')
            esc = 1;
        else if (c == '(')
            depth++;
        else if (c == ')' && !--depth)
            break;
    }
    uint8_t *o = alloc(l->a, end - at + 1);
    struct pv *v = mk(l, PV_STR);
    if (!o || !v) {
        l->at = l->n;
        return NULL;
    }
    depth = 1;
    size_t i = at;
    while (i < end) {
        uint8_t c = l->p[i++];
        if (c == '\\' && i < end) {
            uint8_t e = l->p[i++];
            if (e >= '0' && e <= '7') {
                int x = e - '0';
                for (int j = 0;
                     j < 2 && i < end && l->p[i] >= '0' && l->p[i] <= '7'; j++)
                    x = x * 8 + (l->p[i++] - '0');
                o[k++] = (uint8_t)x;
            } else if (e == '\r') { /* a line continued */
                if (i < end && l->p[i] == '\n')
                    i++;
            } else if (e != '\n') {
                o[k++] = e == 'n'   ? '\n'
                         : e == 'r' ? '\r'
                         : e == 't' ? '\t'
                         : e == 'b' ? '\b'
                         : e == 'f' ? '\f'
                                    : e;
            }
        } else {
            o[k++] = c;
        }
    }
    l->at = end < l->n ? end + 1 : l->n;
    v->s = o;
    v->n = k;
    return v;
}

static struct pv *hexstr(struct lx *l)
{
    size_t at = ++l->at, end = at;
    while (end < l->n && l->p[end] != '>')
        end++;
    uint8_t *o = alloc(l->a, (end - at) / 2 + 2);
    struct pv *v = mk(l, PV_STR);
    if (!o || !v) {
        l->at = l->n;
        return NULL;
    }
    size_t k = 0;
    int hi = -1;
    for (size_t i = at; i < end; i++) {
        int h = hexv(l->p[i]);
        if (h < 0)
            continue;
        if (hi < 0) {
            hi = h;
        } else {
            o[k++] = (uint8_t)(hi << 4 | h);
            hi = -1;
        }
    }
    if (hi >= 0)
        o[k++] = (uint8_t)(hi << 4);
    l->at = end < l->n ? end + 1 : l->n;
    v->s = o;
    v->n = k;
    return v;
}

static struct pv *name(struct lx *l)
{
    size_t at = ++l->at, end = at;
    while (end < l->n && !ws(l->p[end]) && !delim(l->p[end]))
        end++;
    uint8_t *o = alloc(l->a, end - at + 1);
    struct pv *v = mk(l, PV_NAME);
    if (!o || !v) {
        l->at = l->n;
        return NULL;
    }
    size_t k = 0;
    for (size_t i = at; i < end; i++) {
        if (l->p[i] == '#' && i + 2 < end && hexv(l->p[i + 1]) >= 0 &&
            hexv(l->p[i + 2]) >= 0) {
            o[k++] = (uint8_t)(hexv(l->p[i + 1]) << 4 | hexv(l->p[i + 2]));
            i += 2;
        } else {
            o[k++] = l->p[i];
        }
    }
    v->s = o;
    v->n = k;
    l->at = end;
    return v;
}

/* a number's text at l->at: its value, l->at past it; 0 when none */
static int number(struct lx *l, double *out)
{
    size_t i = l->at;
    int neg = 0, digits = 0;
    if (i < l->n && (l->p[i] == '+' || l->p[i] == '-'))
        neg = l->p[i++] == '-';
    double v = 0, scale = 0;
    for (; i < l->n; i++) {
        uint8_t c = l->p[i];
        if (c >= '0' && c <= '9') {
            digits++;
            if (scale) {
                v += (c - '0') * scale;
                scale /= 10;
            } else {
                v = v * 10 + (c - '0');
            }
        } else if (c == '.' && !scale) {
            scale = 0.1;
        } else {
            break;
        }
    }
    if (!digits)
        return 0;
    if (v > 1e15)
        v = 1e15;
    *out = neg ? -v : v;
    l->at = i;
    return 1;
}

static int is_int_at(const struct lx *l, size_t i, size_t *end, double *v)
{
    size_t j = i;
    double x = 0;
    while (j < l->n && l->p[j] >= '0' && l->p[j] <= '9' && j - i < 10)
        x = x * 10 + (l->p[j++] - '0');
    if (j == i || (j < l->n && !ws(l->p[j]) && !delim(l->p[j])))
        return 0;
    *end = j;
    *v = x;
    return 1;
}

static struct pv *value(struct lx *l, int depth);

static struct pv *container(struct lx *l, int depth, int dict)
{
    struct pv *v = mk(l, dict ? PV_DICT : PV_ARR);
    if (!v)
        return NULL;
    size_t cap = 0;
    l->at += dict ? 2 : 1;
    for (;;) {
        skip(l);
        if (l->at >= l->n)
            break;
        if (dict && l->at + 1 < l->n && l->p[l->at] == '>' &&
            l->p[l->at + 1] == '>') {
            l->at += 2;
            break;
        }
        if (!dict && l->p[l->at] == ']') {
            l->at++;
            break;
        }
        struct pv *x = value(l, depth + 1);
        if (!x || x->t == PV_END)
            break;
        if (x->t == PV_KW && !dict)
            continue; /* a stray word inside an array */
        size_t have = dict ? 2 * v->n : v->n;
        if (have + 2 > cap) {
            size_t nc = cap ? 2 * cap : 8;
            struct pv **t = alloc(l->a, nc * sizeof(*t));
            if (!t)
                return v;
            if (have)
                memcpy(t, v->items, have * sizeof(*t));
            v->items = t;
            cap = nc;
        }
        if (dict) {
            if (x->t != PV_NAME)
                continue; /* a key must be a name */
            skip(l);
            struct pv *y = l->at < l->n ? value(l, depth + 1) : NULL;
            if (!y || y->t == PV_END)
                break;
            v->items[have] = x;
            v->items[have + 1] = y;
        } else {
            v->items[have] = x;
        }
        v->n++;
    }
    return v;
}

/* the next value; PV_END at the end, NULL out of memory */
static struct pv *value(struct lx *l, int depth)
{
    skip(l);
    if (l->at >= l->n || depth > PDF_DEPTH)
        return mk(l, PV_END);
    uint8_t c = l->p[l->at];
    if (c == '/')
        return name(l);
    if (c == '(')
        return literal(l);
    if (c == '<' && l->at + 1 < l->n && l->p[l->at + 1] == '<')
        return container(l, depth, 1);
    if (c == '<')
        return hexstr(l);
    if (c == '[')
        return container(l, depth, 0);
    if (c == ']' || c == '>' || c == ')' || c == '{' || c == '}') {
        l->at++;
        return mk(l, PV_KW); /* stray: a word of no meaning */
    }
    double x;
    size_t at = l->at;
    if (number(l, &x)) {
        /* "N G R": a reference */
        size_t e1, e2;
        double g;
        if (x >= 0 && x == floor(x) && l->at < l->n && ws(l->p[l->at])) {
            size_t j = l->at;
            while (j < l->n && ws(l->p[j]))
                j++;
            if (is_int_at(l, j, &e1, &g)) {
                e2 = e1;
                while (e2 < l->n && ws(l->p[e2]))
                    e2++;
                if (e2 < l->n && l->p[e2] == 'R' &&
                    (e2 + 1 >= l->n || ws(l->p[e2 + 1]) ||
                     delim(l->p[e2 + 1]))) {
                    struct pv *v = mk(l, PV_REF);
                    if (v)
                        v->num = x;
                    l->at = e2 + 1;
                    return v;
                }
            }
        }
        struct pv *v = mk(l, PV_NUM);
        if (v)
            v->num = x;
        return v;
    }
    l->at = at;
    size_t end = at;
    while (end < l->n && !ws(l->p[end]) && !delim(l->p[end]))
        end++;
    if (end == at)
        end++; /* a lone delimiter */
    struct pv *v;
    size_t len = end - at;
    if (len == 4 && !memcmp(l->p + at, "true", 4)) {
        v = mk(l, PV_BOOL);
        if (v)
            v->num = 1;
    } else if (len == 5 && !memcmp(l->p + at, "false", 5)) {
        v = mk(l, PV_BOOL);
    } else if (len == 4 && !memcmp(l->p + at, "null", 4)) {
        v = mk(l, PV_NULL);
    } else {
        v = mk(l, PV_KW);
        if (v) {
            v->s = l->p + at;
            v->n = len;
        }
    }
    l->at = end;
    return v;
}

static int is_name(const struct pv *v, const char *s)
{
    return v && v->t == PV_NAME && v->n == strlen(s) && !memcmp(v->s, s, v->n);
}

static int is_kw(const struct pv *v, const char *s)
{
    return v && v->t == PV_KW && v->n == strlen(s) && !memcmp(v->s, s, v->n);
}

static struct pv *get(const struct pv *d, const char *key)
{
    if (!d || d->t != PV_DICT)
        return NULL;
    for (size_t i = 0; i < d->n; i++)
        if (is_name(d->items[2 * i], key))
            return d->items[2 * i + 1];
    return NULL;
}

/* ---- the objects ---- */

struct obj {
    struct pv *v;
    const uint8_t *data; /* its stream's bytes as stored, or NULL */
    size_t dn;
    long at; /* where in the file, for the latest one: -1 in an ObjStm */
};

struct doc {
    const uint8_t *p;
    size_t n;
    struct arena a;
    struct obj *o;
    uint32_t n_obj;
    struct janas_buf *out;
    int cut;             /* the text reached PDF_TEXT */
    int unreadable;      /* strings of fonts that could not be read */
    unsigned char *seen; /* pages and forms already read */
    struct font *fonts;
};

static struct obj *slot(struct doc *d, double num)
{
    if (num < 0 || num >= PDF_OBJ)
        return NULL;
    uint32_t k = (uint32_t)num;
    if (k >= d->n_obj) {
        uint32_t nn = d->n_obj ? d->n_obj : 1024;
        while (nn <= k)
            nn *= 2;
        struct obj *t = realloc(d->o, nn * sizeof(*t));
        if (!t)
            return NULL;
        memset(t + d->n_obj, 0, (nn - d->n_obj) * sizeof(*t));
        d->o = t;
        d->n_obj = nn;
    }
    return &d->o[k];
}

static struct pv *resolve(struct doc *d, struct pv *v);

static double num_of(struct doc *d, struct pv *v, double def)
{
    v = resolve(d, v);
    return v->t == PV_NUM ? v->num : def;
}

static struct pv *resolve(struct doc *d, struct pv *v)
{
    for (int k = 0; v && v->t == PV_REF && k < 8; k++) {
        uint32_t n = (uint32_t)v->num;
        v = n < d->n_obj && d->o[n].v ? d->o[n].v : &NUL;
    }
    return v ? v : &NUL;
}

static struct obj *obj_of(struct doc *d, struct pv *ref)
{
    if (!ref || ref->t != PV_REF)
        return NULL;
    uint32_t n = (uint32_t)ref->num;
    return n < d->n_obj && d->o[n].v ? &d->o[n] : NULL;
}

/* "N G obj" at the start of each object, the later one winning */
static void scan(struct doc *d)
{
    const uint8_t *p = d->p;
    size_t n = d->n;
    for (size_t i = 0; i + 3 <= n; i++) {
        const uint8_t *f = memchr(p + i, 'o', n - i);
        if (!f)
            break;
        i = (size_t)(f - p);
        if (i + 3 > n || memcmp(p + i, "obj", 3) ||
            (i + 3 < n && !ws(p[i + 3]) && !delim(p[i + 3])) || i < 4 ||
            !ws(p[i - 1]))
            continue;
        /* back over the generation and the number */
        size_t j = i - 1;
        while (j > 0 && ws(p[j]))
            j--;
        size_t ge = j + 1;
        while (j > 0 && p[j] >= '0' && p[j] <= '9')
            j--;
        if (j + 1 == ge || !ws(p[j]))
            continue;
        while (j > 0 && ws(p[j]))
            j--;
        size_t ne = j + 1;
        while (j > 0 && p[j] >= '0' && p[j] <= '9')
            j--;
        size_t nb = p[j] >= '0' && p[j] <= '9' ? j : j + 1;
        if (nb == ne || ne - nb > 9 ||
            (nb > 0 && !ws(p[nb - 1]) && !delim(p[nb - 1])))
            continue;
        double num = 0;
        for (size_t k = nb; k < ne; k++)
            num = num * 10 + (p[k] - '0');
        struct lx l = {.p = p, .n = n, .at = i + 3, .a = &d->a};
        struct pv *v = value(&l, 0);
        struct obj *o = slot(d, num);
        if (!v || !o)
            return;
        if (v->t == PV_END || v->t == PV_KW)
            continue;
        o->v = v;
        o->at = (long)nb;
        o->data = NULL;
        o->dn = 0;
        size_t after = l.at;
        skip(&l);
        if (v->t == PV_DICT && l.at + 6 <= n &&
            !memcmp(p + l.at, "stream", 6)) {
            size_t s = l.at + 6;
            if (s < n && p[s] == '\r')
                s++;
            if (s < n && p[s] == '\n')
                s++;
            struct pv *len = get(v, "Length");
            size_t dn = 0;
            int ok = 0;
            if (len && len->t == PV_NUM && len->num >= 0 &&
                len->num <= (double)(n - s)) {
                dn = (size_t)len->num;
                /* "endstream" close after it, or the length is wrong */
                size_t e = s + dn;
                while (e < n && ws(p[e]) && e - (s + dn) < 4)
                    e++;
                ok = e + 9 <= n && !memcmp(p + e, "endstream", 9);
            }
            if (!ok) {
                const uint8_t *e = NULL;
                for (size_t k = s; k + 9 <= n; k++) {
                    const uint8_t *q = memchr(p + k, 'e', n - k);
                    if (!q || (size_t)(q - p) + 9 > n)
                        break;
                    k = (size_t)(q - p);
                    if (!memcmp(q, "endstream", 9)) {
                        e = q;
                        break;
                    }
                }
                if (!e)
                    continue;
                dn = (size_t)(e - (p + s));
                while (dn && (p[s + dn - 1] == '\n' || p[s + dn - 1] == '\r'))
                    dn--;
            }
            o->data = p + s;
            o->dn = dn;
            after = s + dn;
        }
        i = after > i ? after - 1 : i;
    }
}

/* a stream's bytes, its filters undone, into a malloc'd *out: its length,
   -1 when it cannot be (a filter not read here, broken data) */
static long stream_bytes(struct doc *d, struct obj *o, uint8_t **out)
{
    *out = NULL;
    if (!o || !o->data)
        return -1;
    struct pv *f = resolve(d, get(o->v, "Filter"));
    if (f->t == PV_ARR && f->n == 1)
        f = resolve(d, f->items[0]);
    if (f->t == PV_ARR && f->n == 0)
        f = &NUL;
    char why[64];
    if (f->t == PV_NULL) {
        uint8_t *b = malloc(o->dn + 1);
        if (!b)
            return -1;
        memcpy(b, o->data, o->dn);
        b[o->dn] = 0;
        *out = b;
        return (long)o->dn;
    }
    if (is_name(f, "FlateDecode") || is_name(f, "Fl"))
        return janas_zlib_inflate_alloc(o->data, o->dn, PDF_STREAM, out, why,
                                        sizeof why);
    return -1;
}

/* the objects of the object streams, where no object of their number is
   in the file itself */
static void objstms(struct doc *d)
{
    uint32_t n = d->n_obj;
    for (uint32_t i = 0; i < n; i++) {
        struct obj *o = &d->o[i];
        if (!o->v || !o->data || !is_name(get(o->v, "Type"), "ObjStm"))
            continue;
        struct pv *N = resolve(d, get(o->v, "N")),
                  *F = resolve(d, get(o->v, "First"));
        if (N->t != PV_NUM || F->t != PV_NUM || N->num < 0 || F->num < 0 ||
            N->num > 100000)
            continue;
        uint8_t *b;
        long bn = stream_bytes(d, o, &b);
        if (bn < 0)
            continue;
        /* kept for the objects' strings: in the arena */
        uint8_t *keep = alloc(&d->a, (size_t)bn + 1);
        if (!keep) {
            free(b);
            return;
        }
        memcpy(keep, b, (size_t)bn + 1);
        free(b);
        struct lx h = {.p = keep, .n = (size_t)bn, .a = &d->a};
        size_t first = (size_t)F->num;
        for (int k = 0; k < (int)N->num; k++) {
            struct pv *num = value(&h, 0), *off = value(&h, 0);
            if (!num || !off || num->t != PV_NUM || off->t != PV_NUM ||
                off->num < 0 || first + off->num >= (double)bn)
                break;
            struct obj *t = slot(d, num->num);
            if (!t)
                return;
            o = &d->o[i]; /* slot may have moved the table */
            if (t->v && t->at >= 0)
                continue; /* the file's own wins */
            struct lx l = {.p = keep,
                           .n = (size_t)bn,
                           .at = first + (size_t)off->num,
                           .a = &d->a};
            struct pv *v = value(&l, 0);
            if (v && v->t != PV_END) {
                t->v = v;
                t->at = -1;
            }
        }
        n = d->n_obj < n ? d->n_obj : n;
    }
}

/* ---- fonts ---- */

struct range {
    uint32_t lo, hi;
    uint32_t dst;        /* the first code point; or */
    const uint32_t *arr; /* each code point (a bfrange's array), or NULL */
    uint32_t extra[3];   /* more code points of one code (ligatures) */
    int n_extra;
};

struct wrange {
    uint32_t lo, hi;
    float w;
};

struct font {
    struct font *next;
    uint32_t num; /* its object */
    int two;      /* codes of two bytes */
    int unread;   /* two-byte codes and no map: not read */
    struct range *r;
    size_t nr;
    /* the glyphs' widths, in thousandths of the size: a simple font's by
       code (first .. first + nw), a CID font's by ranges (/W), dw the rest */
    float *w;
    uint32_t first, nw;
    struct wrange *wr;
    size_t nwr;
    float dw;
};

static float width(const struct font *f, uint32_t code)
{
    if (!f)
        return 500;
    if (f->w && code >= f->first && code - f->first < f->nw)
        return f->w[code - f->first];
    for (size_t i = 0; i < f->nwr; i++)
        if (code >= f->wr[i].lo && code <= f->wr[i].hi)
            return f->wr[i].w;
    return f->dw;
}

static double num_of(struct doc *d, struct pv *v, double def);

/* the widths of a simple font (/FirstChar, /Widths) or of a Type0 font's
   descendant (/W, /DW) */
static void widths(struct doc *d, struct font *f, struct pv *fd)
{
    f->dw = f->two ? 1000 : 500;
    struct pv *ws = resolve(d, get(fd, "Widths"));
    if (ws->t == PV_ARR && ws->n && ws->n <= 65536) {
        f->w = alloc(&d->a, ws->n * sizeof(float));
        if (!f->w)
            return;
        f->first = (uint32_t)num_of(d, get(fd, "FirstChar"), 0);
        f->nw = (uint32_t)ws->n;
        for (size_t i = 0; i < ws->n; i++)
            f->w[i] = (float)num_of(d, ws->items[i], 500);
        return;
    }
    struct pv *ds = resolve(d, get(fd, "DescendantFonts"));
    struct pv *cid = ds->t == PV_ARR && ds->n ? resolve(d, ds->items[0]) : ds;
    if (cid->t != PV_DICT)
        return;
    f->dw = (float)num_of(d, get(cid, "DW"), 1000);
    struct pv *w = resolve(d, get(cid, "W"));
    if (w->t != PV_ARR)
        return;
    size_t cap = 0;
    for (size_t i = 0; i + 1 < w->n;) {
        double c = num_of(d, w->items[i], -1);
        struct pv *nx = resolve(d, w->items[i + 1]);
        struct wrange r[64];
        size_t nr = 0;
        if (c < 0)
            break;
        if (nx->t == PV_ARR) { /* c [w1 w2 ...] */
            for (size_t k = 0; k < nx->n && nr < 64; k++)
                r[nr++] =
                    (struct wrange){.lo = (uint32_t)c + (uint32_t)k,
                                    .hi = (uint32_t)c + (uint32_t)k,
                                    .w = (float)num_of(d, nx->items[k], f->dw)};
            i += 2;
        } else if (i + 2 < w->n) { /* c1 c2 w */
            double c2 = num_of(d, w->items[i + 1], -1);
            if (c2 < c)
                break;
            r[nr++] =
                (struct wrange){.lo = (uint32_t)c,
                                .hi = (uint32_t)c2,
                                .w = (float)num_of(d, w->items[i + 2], f->dw)};
            i += 3;
        } else {
            break;
        }
        if (f->nwr + nr > cap) {
            size_t nc = cap ? 2 * cap : 256;
            while (nc < f->nwr + nr)
                nc *= 2;
            struct wrange *t = alloc(&d->a, nc * sizeof(*t));
            if (!t)
                return;
            if (f->nwr)
                memcpy(t, f->wr, f->nwr * sizeof(*t));
            f->wr = t;
            cap = nc;
        }
        memcpy(f->wr + f->nwr, r, nr * sizeof(*r));
        f->nwr += nr;
        if (f->nwr > 200000)
            break;
    }
}

static uint32_t be(const uint8_t *s, size_t n)
{
    uint32_t v = 0;
    for (size_t i = 0; i < n && i < 4; i++)
        v = v << 8 | s[i];
    return v;
}

/* UTF-16BE to code points: the first into *cp, the rest into extra */
static int utf16(const uint8_t *s, size_t n, uint32_t *cp, uint32_t *extra,
                 int max_extra)
{
    int k = 0;
    for (size_t i = 0; i + 1 < n; i += 2) {
        uint32_t u = (uint32_t)s[i] << 8 | s[i + 1];
        if (u >= 0xd800 && u < 0xdc00 && i + 3 < n) {
            uint32_t w = (uint32_t)s[i + 2] << 8 | s[i + 3];
            if (w >= 0xdc00 && w < 0xe000) {
                u = 0x10000 + ((u - 0xd800) << 10) + (w - 0xdc00);
                i += 2;
            }
        }
        if (k == 0)
            *cp = u;
        else if (k - 1 < max_extra)
            extra[k - 1] = u;
        k++;
    }
    return k;
}

static void cmap(struct doc *d, struct font *f, const uint8_t *b, size_t n)
{
    struct lx l = {.p = b, .n = n, .a = &d->a};
    size_t cap = 0;
    int mode = 0; /* 1 codespace, 2 bfchar, 3 bfrange */
    for (int guard = 0; guard < 2000000; guard++) {
        struct pv *v = value(&l, 0);
        if (!v || v->t == PV_END)
            break;
        if (v->t == PV_KW) {
            mode = is_kw(v, "begincodespacerange") ? 1
                   : is_kw(v, "beginbfchar")       ? 2
                   : is_kw(v, "beginbfrange")      ? 3
                   : (is_kw(v, "endcodespacerange") || is_kw(v, "endbfchar") ||
                      is_kw(v, "endbfrange"))
                       ? 0
                       : mode;
            continue;
        }
        if (v->t != PV_STR || !mode)
            continue;
        if (mode == 1) {
            f->two = v->n >= 2;
            value(&l, 0); /* its end */
            continue;
        }
        struct pv *hi = mode == 3 ? value(&l, 0) : v;
        struct pv *dst = value(&l, 0);
        if (!hi || !dst || hi->t != PV_STR || v->n > 4 || hi->n > 4)
            continue;
        if (f->nr == cap) {
            size_t nc = cap ? 2 * cap : 64;
            struct range *t = alloc(&d->a, nc * sizeof(*t));
            if (!t)
                return;
            if (f->nr)
                memcpy(t, f->r, f->nr * sizeof(*t));
            f->r = t;
            cap = nc;
        }
        struct range *r = &f->r[f->nr];
        r->lo = be(v->s, v->n);
        r->hi = be(hi->s, hi->n);
        if (r->hi < r->lo || r->hi - r->lo > 65535)
            continue;
        if (dst->t == PV_STR) {
            r->n_extra = utf16(dst->s, dst->n, &r->dst, r->extra, 3) - 1;
            if (r->n_extra < 0)
                continue;
            if (r->n_extra > 3)
                r->n_extra = 3;
            if (r->hi != r->lo)
                r->n_extra = 0; /* a range moves the last unit only */
        } else if (dst->t == PV_ARR && mode == 3 &&
                   dst->n >= (size_t)(r->hi - r->lo + 1)) {
            uint32_t *a = alloc(&d->a, (r->hi - r->lo + 1) * sizeof(*a));
            if (!a)
                return;
            for (uint32_t k = 0; k <= r->hi - r->lo; k++) {
                struct pv *x = dst->items[k];
                uint32_t ex[3];
                a[k] = 0xfffd;
                if (x->t == PV_STR)
                    utf16(x->s, x->n, &a[k], ex, 0);
            }
            r->arr = a;
        } else {
            continue;
        }
        f->nr++;
    }
}

static struct font *font_of(struct doc *d, struct pv *ref)
{
    struct obj *o = obj_of(d, ref);
    uint32_t num = ref && ref->t == PV_REF ? (uint32_t)ref->num : 0;
    for (struct font *f = d->fonts; f; f = f->next)
        if (num && f->num == num)
            return f;
    struct font *f = alloc(&d->a, sizeof(*f));
    if (!f)
        return NULL;
    f->num = num;
    f->next = d->fonts;
    d->fonts = f;
    struct pv *fd = o ? o->v : resolve(d, ref);
    f->two = is_name(resolve(d, get(fd, "Subtype")), "Type0");
    struct obj *tu = obj_of(d, get(fd, "ToUnicode"));
    uint8_t *b;
    long bn = tu ? stream_bytes(d, tu, &b) : -1;
    if (bn >= 0) {
        int two = f->two;
        cmap(d, f, b, (size_t)bn);
        free(b);
        if (two)
            f->two = 1; /* a Type0 font's codes are two bytes, whatever
                           its map's codespace says */
    }
    f->unread = f->two && !f->nr;
    widths(d, f, fd);
    return f;
}

/* ---- text out ---- */

static const uint16_t WINANSI[32] = {
    0x20ac, 0xfffd, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021,
    0x02c6, 0x2030, 0x0160, 0x2039, 0x0152, 0xfffd, 0x017d, 0xfffd,
    0xfffd, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014,
    0x02dc, 0x2122, 0x0161, 0x203a, 0x0153, 0xfffd, 0x017e, 0x0178};

struct tx {
    struct doc *d;
    struct font *f;
    double lx, ly, sc;    /* the text line matrix: its origin, its scale */
    double x, y, psc;     /* where the last line written started, its scale */
    double adv;           /* written since then, in text units */
    double size, cs, wsp; /* Tf's size, Tc, Tw */
    int have_y;
    int space_pending;
};

static void put_cp(struct doc *d, uint32_t c)
{
    if (d->out->n >= PDF_TEXT) {
        d->cut = 1;
        return;
    }
    char b[4];
    size_t n;
    if (c == 0 || c > 0x10ffff || (c >= 0xd800 && c < 0xe000))
        return;
    if (c < 0x20 && c != '\n' && c != '\t')
        c = ' ';
    if (c < 0x80) {
        b[0] = (char)c;
        n = 1;
    } else if (c < 0x800) {
        b[0] = (char)(0xc0 | c >> 6);
        b[1] = (char)(0x80 | (c & 0x3f));
        n = 2;
    } else if (c < 0x10000) {
        b[0] = (char)(0xe0 | c >> 12);
        b[1] = (char)(0x80 | (c >> 6 & 0x3f));
        b[2] = (char)(0x80 | (c & 0x3f));
        n = 3;
    } else {
        b[0] = (char)(0xf0 | c >> 18);
        b[1] = (char)(0x80 | (c >> 12 & 0x3f));
        b[2] = (char)(0x80 | (c >> 6 & 0x3f));
        b[3] = (char)(0x80 | (c & 0x3f));
        n = 4;
    }
    janas_buf_put(d->out, b, n);
}

static int last_is(const struct doc *d, char c)
{
    return d->out->n && d->out->p[d->out->n - 1] == c;
}

static void text_newline(struct tx *t)
{
    struct janas_buf *o = t->d->out;
    while (o->n && o->p[o->n - 1] == ' ')
        o->n--;
    if (o->n &&
        !(o->n >= 2 && o->p[o->n - 1] == '\n' && o->p[o->n - 2] == '\n'))
        janas_buf_put(o, "\n", 1);
    t->space_pending = 0;
}

static void text_space(struct tx *t)
{
    if (t->d->out->n && !last_is(t->d, ' ') && !last_is(t->d, '\n'))
        t->space_pending = 1;
}

static int lookup(const struct font *f, uint32_t code, struct doc *d)
{
    for (size_t i = 0; i < f->nr; i++) { /* the last match wins */
        const struct range *r = &f->r[f->nr - 1 - i];
        if (code < r->lo || code > r->hi)
            continue;
        if (r->arr)
            put_cp(d, r->arr[code - r->lo]);
        else {
            put_cp(d, r->dst + (code - r->lo));
            for (int k = 0; k < r->n_extra; k++)
                put_cp(d, r->extra[k]);
        }
        return 1;
    }
    return 0;
}

static void show(struct tx *t, const struct pv *s)
{
    struct doc *d = t->d;
    if (!s || s->t != PV_STR || !s->n)
        return;
    struct font *f = t->f;
    if (f && f->unread) {
        d->unreadable++;
        return;
    }
    if (t->space_pending) {
        janas_buf_put(d->out, " ", 1);
        t->space_pending = 0;
    }
    int two = f && f->two;
    for (size_t i = 0; i + (two ? 1 : 0) < s->n; i += two ? 2 : 1) {
        uint32_t code = two ? (uint32_t)s->s[i] << 8 | s->s[i + 1] : s->s[i];
        t->adv += width(f, code) / 1000 * t->size + t->cs +
                  (!two && code == 32 ? t->wsp : 0);
        if (f && f->nr && lookup(f, code, d))
            continue;
        if (two)
            continue; /* a code the map does not have */
        uint32_t c = code >= 0x80 && code < 0xa0 ? WINANSI[code - 0x80] : code;
        put_cp(d, c);
    }
}

/* the text moved to a new line's start (x, y), in the page's units, with
   the scale sc: a new line when the line changes; a space when it moves
   on along it further than what was written there, by a fifth of the size
   or more (a gap between words, not between letters) */
static void move(struct tx *t, double x, double y, double sc)
{
    double size = (t->size > 0 ? t->size : 10) * (sc > 0 ? sc : 1);
    if (t->have_y && fabs(y - t->y) > size * 0.5)
        text_newline(t);
    else if (t->have_y && x - (t->x + t->adv * t->psc) > size * 0.2)
        text_space(t);
    t->lx = t->x = x;
    t->ly = t->y = y;
    t->sc = t->psc = sc > 0 ? sc : 1;
    t->adv = 0;
    t->have_y = 1;
}

static void content(struct doc *d, const uint8_t *b, size_t n, struct pv *res,
                    int depth);

static void draw_form(struct doc *d, struct pv *res, const struct pv *nm,
                      int depth)
{
    if (depth >= PDF_FORMS || !nm || nm->t != PV_NAME)
        return;
    struct pv *dict = resolve(d, get(res, "XObject"));
    struct pv *ref = NULL;
    for (size_t i = 0; dict->t == PV_DICT && i < dict->n; i++) {
        const struct pv *k = dict->items[2 * i];
        if (k->n == nm->n && !memcmp(k->s, nm->s, nm->n))
            ref = dict->items[2 * i + 1];
    }
    struct obj *o = obj_of(d, ref);
    if (!o || !o->data || !is_name(resolve(d, get(o->v, "Subtype")), "Form"))
        return;
    uint32_t num = (uint32_t)ref->num;
    if (d->seen[num / 8] & (1u << (num % 8)))
        return; /* drawn already: no loops, no text twice */
    d->seen[num / 8] |= (unsigned char)(1u << (num % 8));
    uint8_t *fb;
    long fn = stream_bytes(d, o, &fb);
    if (fn < 0)
        return;
    struct pv *own = resolve(d, get(o->v, "Resources"));
    /* a form's text is a block of its own */
    if (d->out->n && !last_is(d, '\n'))
        janas_buf_put(d->out, "\n", 1);
    content(d, fb, (size_t)fn, own->t == PV_DICT ? own : res, depth + 1);
    if (d->out->n && !last_is(d, '\n'))
        janas_buf_put(d->out, "\n", 1);
    free(fb);
}

static void content(struct doc *d, const uint8_t *b, size_t n, struct pv *res,
                    int depth)
{
    struct arena a = {0}; /* the values of one stream, freed after it */
    struct lx l = {.p = b, .n = n, .a = &a};
    struct pv *st[16];
    int ns = 0;
    struct tx t = {.d = d, .size = 10, .sc = 1, .psc = 1};
    struct pv *fonts = resolve(d, get(res, "Font"));
    for (;;) {
        struct pv *v = value(&l, 0);
        if (!v || v->t == PV_END || d->cut)
            break;
        if (v->t != PV_KW) {
            if (ns == 16)
                memmove(st, st + 1, 15 * sizeof(*st)), ns--;
            st[ns++] = v;
            continue;
        }
        const struct pv *op = v;
        if (is_kw(op, "BT")) { /* the matrices start again */
            t.lx = t.ly = 0;
            t.sc = 1;
        } else if (is_kw(op, "ET")) {
            ;
        } else if (is_kw(op, "Tf") && ns >= 2 && st[ns - 2]->t == PV_NAME) {
            struct pv *ref = NULL;
            for (size_t i = 0; fonts->t == PV_DICT && i < fonts->n; i++) {
                const struct pv *k = fonts->items[2 * i];
                if (k->n == st[ns - 2]->n && !memcmp(k->s, st[ns - 2]->s, k->n))
                    ref = fonts->items[2 * i + 1];
            }
            t.f = ref ? font_of(d, ref) : NULL;
            if (st[ns - 1]->t == PV_NUM && st[ns - 1]->num > 0)
                t.size = st[ns - 1]->num;
        } else if (is_kw(op, "Tj") && ns >= 1) {
            show(&t, st[ns - 1]);
        } else if ((is_kw(op, "'") || is_kw(op, "\"")) && ns >= 1) {
            text_newline(&t);
            show(&t, st[ns - 1]);
        } else if (is_kw(op, "TJ") && ns >= 1 && st[ns - 1]->t == PV_ARR) {
            const struct pv *arr = st[ns - 1];
            for (size_t i = 0; i < arr->n; i++) {
                const struct pv *x = arr->items[i];
                if (x->t == PV_NUM) {
                    t.adv -= x->num / 1000 * t.size;
                    if (x->num < -200)
                        text_space(&t); /* wider than a fifth of an em */
                } else {
                    show(&t, x);
                }
            }
        } else if ((is_kw(op, "Td") || is_kw(op, "TD")) && ns >= 2 &&
                   st[ns - 2]->t == PV_NUM && st[ns - 1]->t == PV_NUM) {
            move(&t, t.lx + st[ns - 2]->num * t.sc,
                 t.ly + st[ns - 1]->num * t.sc, t.sc);
        } else if (is_kw(op, "Tm") && ns >= 6 && st[ns - 6]->t == PV_NUM &&
                   st[ns - 2]->t == PV_NUM && st[ns - 1]->t == PV_NUM) {
            double a = st[ns - 6]->num,
                   b2 = st[ns - 5]->t == PV_NUM ? st[ns - 5]->num : 0;
            move(&t, st[ns - 2]->num, st[ns - 1]->num, sqrt(a * a + b2 * b2));
        } else if (is_kw(op, "Tc") && ns >= 1 && st[ns - 1]->t == PV_NUM) {
            t.cs = st[ns - 1]->num;
        } else if (is_kw(op, "Tw") && ns >= 1 && st[ns - 1]->t == PV_NUM) {
            t.wsp = st[ns - 1]->num;
        } else if (is_kw(op, "T*")) {
            text_newline(&t);
            t.adv = 0;
        } else if (is_kw(op, "Do") && ns >= 1) {
            draw_form(d, res, st[ns - 1], depth);
        } else if (is_kw(op, "BI")) {
            /* an inline image: its data, binary, to "EI" */
            size_t i = l.at;
            while (i + 2 < n &&
                   !(ws(b[i]) && b[i + 1] == 'E' && b[i + 2] == 'I' &&
                     (i + 3 >= n || ws(b[i + 3]))))
                i++;
            l.at = i + 3 < n ? i + 3 : n;
        }
        ns = 0;
    }
    arena_free(&a);
}

/* the pages from the tree's node v, with the resources they inherit */
static void pages(struct doc *d, struct pv *ref, struct pv *res, int depth,
                  int *count)
{
    if (depth > PDF_DEPTH || *count >= PDF_PAGES || d->cut)
        return;
    struct pv *node = resolve(d, ref);
    if (node->t != PV_DICT)
        return;
    if (ref && ref->t == PV_REF) {
        uint32_t num = (uint32_t)ref->num;
        if (num >= d->n_obj || (d->seen[num / 8] & (1u << (num % 8))))
            return;
        d->seen[num / 8] |= (unsigned char)(1u << (num % 8));
    }
    struct pv *own = resolve(d, get(node, "Resources"));
    if (own->t == PV_DICT)
        res = own;
    struct pv *kids = resolve(d, get(node, "Kids"));
    if (kids->t == PV_ARR) {
        for (size_t i = 0; i < kids->n; i++)
            pages(d, kids->items[i], res, depth + 1, count);
        return;
    }
    if (!is_name(resolve(d, get(node, "Type")), "Page") &&
        !get(node, "Contents"))
        return;
    (*count)++;
    struct pv *c = get(node, "Contents");
    struct pv *one[1] = {c};
    struct pv **list = one;
    size_t nl = 1;
    struct pv *cr = resolve(d, c);
    if (cr->t == PV_ARR) {
        list = cr->items;
        nl = cr->n;
    }
    /* the streams of a page make one content: joined, then read */
    struct janas_buf all = {0};
    for (size_t i = 0; i < nl; i++) {
        uint8_t *b;
        long bn = stream_bytes(d, obj_of(d, list[i]), &b);
        if (bn < 0)
            continue;
        janas_buf_put(&all, (const char *)b, (size_t)bn);
        janas_buf_put(&all, "\n", 1);
        free(b);
    }
    if (all.n && !all.oom) {
        size_t before = d->out->n;
        content(d, (const uint8_t *)all.p, all.n, res, 0);
        if (d->out->n > before) {
            while (d->out->n && (d->out->p[d->out->n - 1] == ' ' ||
                                 d->out->p[d->out->n - 1] == '\n'))
                d->out->n--;
            janas_buf_put(d->out, "\n\n", 2);
        }
    }
    janas_buf_free(&all);
}

int dz_pdf_text(const uint8_t *p, size_t n, struct janas_buf *out, char *why,
                size_t wn)
{
    size_t head = n < 1024 ? n : 1024;
    int is_pdf = 0;
    for (size_t i = 0; i + 5 <= head && !is_pdf; i++)
        is_pdf = !memcmp(p + i, "%PDF-", 5);
    if (!is_pdf) {
        snprintf(why, wn, "not a PDF");
        return -1;
    }
    struct doc d = {.p = p, .n = n, .out = out};
    scan(&d);
    objstms(&d);
    int rc = 0;
    struct pv *catalog = NULL;
    for (uint32_t i = 0; i < d.n_obj; i++) {
        struct pv *v = d.o[i].v;
        if (is_name(get(v, "Type"), "XRef") && get(v, "Encrypt"))
            rc = 1;
        if (is_name(get(v, "Type"), "Catalog"))
            catalog = v;
    }
    /* the trailer of a file without an xref stream */
    for (size_t i = n > 6 ? n - 7 : 0; i > 0 && n - i < (1u << 20); i--)
        if (!memcmp(p + i, "trailer", 7)) {
            struct lx l = {.p = p, .n = n, .at = i + 7, .a = &d.a};
            struct pv *t = value(&l, 0);
            if (t && get(t, "Encrypt"))
                rc = 1;
            struct pv *root = t ? resolve(&d, get(t, "Root")) : NULL;
            if (root && root->t == PV_DICT && !catalog)
                catalog = root;
            break;
        }
    if (rc) {
        snprintf(why, wn, "an encrypted PDF, not read");
    } else if (d.a.oom) {
        snprintf(why, wn, "out of memory");
        rc = -1;
    } else {
        d.seen = calloc(d.n_obj / 8 + 1, 1);
        int count = 0;
        if (!d.seen) {
            snprintf(why, wn, "out of memory");
            rc = -1;
        } else if (catalog) {
            pages(&d, get(catalog, "Pages"), NULL, 0, &count);
        }
        if (d.seen && !count) /* no tree: the page objects in their order */
            for (uint32_t i = 0; i < d.n_obj && count < PDF_PAGES; i++)
                if (is_name(get(d.o[i].v, "Type"), "Page")) {
                    struct pv ref = {.t = PV_REF, .num = i};
                    pages(&d, &ref, NULL, 0, &count);
                }
        while (out->n && (out->p[out->n - 1] == '\n'))
            out->n--;
        if (rc == 0 && !out->n) {
            snprintf(why, wn,
                     d.unreadable ? "its fonts give no characters (no "
                                    "ToUnicode map): its text cannot be read"
                     : count      ? "no text: pictures of pages, perhaps "
                                    "scanned"
                                  : "no pages found");
            rc = -1;
        }
    }
    free(d.seen);
    free(d.o);
    arena_free(&d.a);
    if (out->oom) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    return rc ? -1 : 0;
}
