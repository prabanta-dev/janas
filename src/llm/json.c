/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * json.c - a small JSON reader and writer (see json.h).
 */
#include "json.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DEPTH 128
#define BLOCK 65536

/* ---- the arena ---- */

struct block {
    struct block *next;
    size_t used, cap;
    _Alignas(16) char data[];
};

struct janas_json_doc {
    struct block *blocks;
    struct janas_json *root;
};

static void *arena_alloc(struct janas_json_doc *d, size_t n)
{
    n = (n + 15) & ~(size_t)15;
    struct block *b = d->blocks;
    if (!b || b->cap - b->used < n) {
        size_t cap = n > BLOCK ? n : BLOCK;
        b = malloc(sizeof(*b) + cap);
        if (!b)
            return NULL;
        b->used = 0;
        b->cap = cap;
        b->next = d->blocks;
        d->blocks = b;
    }
    void *p = b->data + b->used;
    b->used += n;
    return p;
}

/* ---- the reader ---- */

struct reader {
    struct janas_json_doc *d;
    const char *s;
    size_t n, i;
    char *err;
    size_t err_len;
    int failed;
};

static int bad(struct reader *r, const char *why)
{
    if (!r->failed && r->err_len)
        snprintf(r->err, r->err_len, "%s at byte %zu", why, r->i);
    r->failed = 1;
    return -1;
}

static void skip_ws(struct reader *r)
{
    while (r->i < r->n && (r->s[r->i] == ' ' || r->s[r->i] == '\t' ||
                           r->s[r->i] == '\n' || r->s[r->i] == '\r'))
        r->i++;
}

static int hex4(const char *s, uint32_t *v)
{
    *v = 0;
    for (int k = 0; k < 4; k++) {
        char c = s[k];
        uint32_t h = c >= '0' && c <= '9'   ? (uint32_t)(c - '0')
                     : c >= 'a' && c <= 'f' ? (uint32_t)(c - 'a' + 10)
                     : c >= 'A' && c <= 'F' ? (uint32_t)(c - 'A' + 10)
                                            : 16;
        if (h == 16)
            return -1;
        *v = *v << 4 | h;
    }
    return 0;
}

static size_t put_utf8(char *o, uint32_t cp)
{
    if (cp < 0x80) {
        o[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        o[0] = (char)(0xc0 | cp >> 6);
        o[1] = (char)(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp < 0x10000) {
        o[0] = (char)(0xe0 | cp >> 12);
        o[1] = (char)(0x80 | (cp >> 6 & 0x3f));
        o[2] = (char)(0x80 | (cp & 0x3f));
        return 3;
    }
    o[0] = (char)(0xf0 | cp >> 18);
    o[1] = (char)(0x80 | (cp >> 12 & 0x3f));
    o[2] = (char)(0x80 | (cp >> 6 & 0x3f));
    o[3] = (char)(0x80 | (cp & 0x3f));
    return 4;
}

/* A string at r->i (on its opening quote), decoded into the arena. */
static int read_string(struct reader *r, const char **out, size_t *out_n)
{
    size_t start = ++r->i, end = start;
    while (end < r->n && r->s[end] != '"') {
        if ((unsigned char)r->s[end] < 0x20) {
            r->i = end;
            return bad(r, "control character in a string");
        }
        end += r->s[end] == '\\' ? 2 : 1;
    }
    if (end >= r->n) {
        r->i = r->n;
        return bad(r, "unterminated string");
    }
    /* decoding never makes a string longer */
    char *o = arena_alloc(r->d, end - start + 1);
    if (!o)
        return bad(r, "out of memory");
    size_t w = 0;
    for (size_t i = start; i < end;) {
        char c = r->s[i];
        if (c != '\\') {
            o[w++] = c;
            i++;
            continue;
        }
        char e = r->s[i + 1];
        i += 2;
        switch (e) {
        case '"':
        case '\\':
        case '/':
            o[w++] = e;
            break;
        case 'b':
            o[w++] = '\b';
            break;
        case 'f':
            o[w++] = '\f';
            break;
        case 'n':
            o[w++] = '\n';
            break;
        case 'r':
            o[w++] = '\r';
            break;
        case 't':
            o[w++] = '\t';
            break;
        case 'u': {
            uint32_t cp, lo;
            if (i + 4 > end || hex4(r->s + i, &cp) != 0) {
                r->i = i;
                return bad(r, "invalid \\u escape");
            }
            i += 4;
            if (cp >= 0xd800 && cp < 0xdc00 && i + 6 <= end &&
                r->s[i] == '\\' && r->s[i + 1] == 'u' &&
                hex4(r->s + i + 2, &lo) == 0 && lo >= 0xdc00 && lo < 0xe000) {
                cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                i += 6;
            } else if (cp >= 0xd800 && cp < 0xe000) {
                cp = 0xfffd; /* a lone surrogate */
            }
            w += put_utf8(o + w, cp);
            break;
        }
        default:
            r->i = i - 1;
            return bad(r, "invalid escape");
        }
    }
    o[w] = 0;
    *out = o;
    *out_n = w;
    r->i = end + 1;
    return 0;
}

static int is_digit(char c)
{
    return c >= '0' && c <= '9';
}

static int read_number(struct reader *r, struct janas_json *v)
{
    size_t i = r->i;
    if (i < r->n && r->s[i] == '-')
        i++;
    if (i < r->n && r->s[i] == '0')
        i++;
    else if (i < r->n && is_digit(r->s[i]))
        while (i < r->n && is_digit(r->s[i]))
            i++;
    else
        return bad(r, "invalid number");
    if (i < r->n && r->s[i] == '.') {
        if (++i >= r->n || !is_digit(r->s[i]))
            return bad(r, "invalid number");
        while (i < r->n && is_digit(r->s[i]))
            i++;
    }
    if (i < r->n && (r->s[i] == 'e' || r->s[i] == 'E')) {
        i++;
        if (i < r->n && (r->s[i] == '+' || r->s[i] == '-'))
            i++;
        if (i >= r->n || !is_digit(r->s[i]))
            return bad(r, "invalid number");
        while (i < r->n && is_digit(r->s[i]))
            i++;
    }
    char *t = arena_alloc(r->d, i - r->i + 1);
    if (!t)
        return bad(r, "out of memory");
    memcpy(t, r->s + r->i, i - r->i);
    t[i - r->i] = 0;
    v->type = JANAS_JSON_NUMBER;
    v->s = t;
    v->n = i - r->i;
    r->i = i;
    return 0;
}

static int read_value(struct reader *r, struct janas_json *v, int depth);

static int read_items(struct reader *r, struct janas_json *v, int depth,
                      int object)
{
    char close = object ? '}' : ']';
    v->type = object ? JANAS_JSON_OBJECT : JANAS_JSON_ARRAY;
    r->i++;
    skip_ws(r);
    if (r->i < r->n && r->s[r->i] == close) {
        r->i++;
        return 0;
    }
    struct janas_json **link = &v->child;
    for (;;) {
        struct janas_json *it = arena_alloc(r->d, sizeof(*it));
        if (!it)
            return bad(r, "out of memory");
        memset(it, 0, sizeof(*it));
        skip_ws(r);
        if (object) {
            if (r->i >= r->n || r->s[r->i] != '"')
                return bad(r, "expected a member name");
            if (read_string(r, &it->key, &it->key_n) != 0)
                return -1;
            skip_ws(r);
            if (r->i >= r->n || r->s[r->i] != ':')
                return bad(r, "expected ':'");
            r->i++;
        }
        if (read_value(r, it, depth + 1) != 0)
            return -1;
        *link = it;
        link = &it->next;
        v->n++;
        skip_ws(r);
        if (r->i < r->n && r->s[r->i] == ',') {
            r->i++;
            continue;
        }
        if (r->i < r->n && r->s[r->i] == close) {
            r->i++;
            return 0;
        }
        return bad(r, object ? "expected ',' or '}'" : "expected ',' or ']'");
    }
}

static int literal(struct reader *r, const char *word)
{
    size_t k = strlen(word);
    if (r->n - r->i < k || memcmp(r->s + r->i, word, k) != 0)
        return bad(r, "invalid value");
    r->i += k;
    return 0;
}

static int read_value(struct reader *r, struct janas_json *v, int depth)
{
    if (depth >= MAX_DEPTH) /* depth 0 is the outermost value */
        return bad(r, "nested too deeply");
    skip_ws(r);
    if (r->i >= r->n)
        return bad(r, "unexpected end");
    switch (r->s[r->i]) {
    case '{':
        return read_items(r, v, depth, 1);
    case '[':
        return read_items(r, v, depth, 0);
    case '"':
        v->type = JANAS_JSON_STRING;
        return read_string(r, &v->s, &v->n);
    case 't':
        v->type = JANAS_JSON_TRUE;
        return literal(r, "true");
    case 'f':
        v->type = JANAS_JSON_FALSE;
        return literal(r, "false");
    case 'n':
        v->type = JANAS_JSON_NULL;
        return literal(r, "null");
    default:
        return read_number(r, v);
    }
}

struct janas_json_doc *janas_json_parse(const char *text, size_t n, char *err,
                                        size_t err_len)
{
    if (err_len)
        err[0] = 0;
    struct janas_json_doc *d = calloc(1, sizeof(*d));
    if (!d) {
        if (err_len)
            snprintf(err, err_len, "out of memory");
        return NULL;
    }
    struct reader r = {
        .d = d, .s = text, .n = text ? n : 0, .err = err, .err_len = err_len};
    d->root = arena_alloc(d, sizeof(*d->root));
    if (d->root) {
        memset(d->root, 0, sizeof(*d->root));
        if (read_value(&r, d->root, 0) == 0) {
            skip_ws(&r);
            if (r.i != r.n)
                bad(&r, "text after the value");
        }
    } else {
        bad(&r, "out of memory");
    }
    if (r.failed) {
        janas_json_free(d);
        return NULL;
    }
    return d;
}

const struct janas_json *janas_json_root(const struct janas_json_doc *d)
{
    return d ? d->root : NULL;
}

void janas_json_free(struct janas_json_doc *d)
{
    if (!d)
        return;
    for (struct block *b = d->blocks; b;) {
        struct block *nx = b->next;
        free(b);
        b = nx;
    }
    free(d);
}

const struct janas_json *janas_json_get(const struct janas_json *obj,
                                        const char *key)
{
    if (!obj || obj->type != JANAS_JSON_OBJECT)
        return NULL;
    size_t k = strlen(key);
    for (const struct janas_json *m = obj->child; m; m = m->next)
        if (m->key_n == k && memcmp(m->key, key, k) == 0)
            return m;
    return NULL;
}

const char *janas_json_str(const struct janas_json *v)
{
    return v && v->type == JANAS_JSON_STRING ? v->s : NULL;
}

int janas_json_is(const struct janas_json *v, const char *s)
{
    return v && v->type == JANAS_JSON_STRING && v->n == strlen(s) &&
           memcmp(v->s, s, v->n) == 0;
}

double janas_json_num(const struct janas_json *v, double def)
{
    return v && v->type == JANAS_JSON_NUMBER ? strtod(v->s, NULL) : def;
}

/* ---- the writer ---- */

void janas_buf_put(struct janas_buf *b, const char *s, size_t n)
{
    if (b->oom)
        return;
    if (b->n + n + 1 > b->cap) {
        size_t cap = 2 * (b->n + n) + 256;
        char *p = realloc(b->p, cap);
        if (!p) {
            b->oom = 1;
            return;
        }
        b->p = p;
        b->cap = cap;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

void janas_buf_puts(struct janas_buf *b, const char *s)
{
    janas_buf_put(b, s, strlen(s));
}

void janas_buf_printf(struct janas_buf *b, const char *fmt, ...)
{
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (k < 0) {
        b->oom = 1;
        return;
    }
    if ((size_t)k < sizeof(tmp)) {
        janas_buf_put(b, tmp, (size_t)k);
        return;
    }
    char *big = malloc((size_t)k + 1);
    if (!big) {
        b->oom = 1;
        return;
    }
    va_start(ap, fmt);
    vsnprintf(big, (size_t)k + 1, fmt, ap);
    va_end(ap);
    janas_buf_put(b, big, (size_t)k);
    free(big);
}

void janas_buf_free(struct janas_buf *b)
{
    free(b->p);
    *b = (struct janas_buf){0};
}

void janas_json_write_str(struct janas_buf *b, const char *s, size_t n)
{
    janas_buf_put(b, "\"", 1);
    size_t run = 0; /* bytes copied as they are, written in one go */
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        const char *esc = NULL;
        char u[8];
        switch (c) {
        case '"':
            esc = "\\\"";
            break;
        case '\\':
            esc = "\\\\";
            break;
        case '\n':
            esc = "\\n";
            break;
        case '\r':
            esc = "\\r";
            break;
        case '\t':
            esc = "\\t";
            break;
        case '\b':
            esc = "\\b";
            break;
        case '\f':
            esc = "\\f";
            break;
        default:
            if (c < 0x20) {
                snprintf(u, sizeof(u), "\\u%04x", c);
                esc = u;
            }
        }
        if (!esc)
            continue;
        janas_buf_put(b, s + run, i - run);
        janas_buf_puts(b, esc);
        run = i + 1;
    }
    janas_buf_put(b, s + run, n - run);
    janas_buf_put(b, "\"", 1);
}

static void write_value(struct janas_buf *b, const struct janas_json *v)
{
    switch (v->type) {
    case JANAS_JSON_NULL:
        janas_buf_puts(b, "null");
        break;
    case JANAS_JSON_FALSE:
        janas_buf_puts(b, "false");
        break;
    case JANAS_JSON_TRUE:
        janas_buf_puts(b, "true");
        break;
    case JANAS_JSON_NUMBER:
        janas_buf_put(b, v->s, v->n);
        break;
    case JANAS_JSON_STRING:
        janas_json_write_str(b, v->s, v->n);
        break;
    case JANAS_JSON_ARRAY:
    case JANAS_JSON_OBJECT: {
        int obj = v->type == JANAS_JSON_OBJECT;
        janas_buf_put(b, obj ? "{" : "[", 1);
        for (const struct janas_json *c = v->child; c; c = c->next) {
            if (c != v->child)
                janas_buf_put(b, ", ", 2);
            if (obj) {
                janas_json_write_str(b, c->key, c->key_n);
                janas_buf_put(b, ": ", 2);
            }
            write_value(b, c);
        }
        janas_buf_put(b, obj ? "}" : "]", 1);
        break;
    }
    }
}

void janas_json_write(struct janas_buf *b, const struct janas_json *v)
{
    write_value(b, v);
}
