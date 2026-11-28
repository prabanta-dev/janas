/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jinja_parse.c - a chat template read into a tree (jinja.h): its text cut
 * at the tags, the whitespace as Hugging Face's transformers has Jinja
 * treat it (trim_blocks: the line's end after a block tag goes;
 * lstrip_blocks: the spaces before one at the start of its line; and the
 * tags' own "-"), then each tag's expression by the grammar of Jinja's own
 * parser, operator by operator.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/jinja_int.h"

/* ---- arenas */
struct jblock {
    struct jblock *next;
    size_t cap, used;
    _Alignas(16) unsigned char data[];
};

void *ja_alloc(struct jarena *a, size_t n)
{
    if (a->over)
        return NULL;
    n = (n + 15) & ~(size_t)15;
    if (!n)
        n = 16;
    if (a->limit && a->used + n > a->limit) {
        a->over = 1;
        return NULL;
    }
    struct jblock *b = a->head;
    if (!b || b->cap - b->used < n) {
        size_t cap = n > 65536 ? n : 65536;
        b = malloc(sizeof(*b) + cap);
        if (!b) {
            a->over = 1;
            return NULL;
        }
        b->cap = cap;
        b->used = 0;
        b->next = a->head;
        a->head = b;
    }
    void *p = b->data + b->used;
    b->used += n;
    a->used += n;
    memset(p, 0, n);
    return p;
}

void ja_free(struct jarena *a)
{
    while (a->head) {
        struct jblock *b = a->head;
        a->head = b->next;
        free(b);
    }
    a->used = 0;
}

/* ---- the template cut at its tags */
enum { SG_TEXT, SG_VAR, SG_BLOCK, SG_COMMENT };
struct seg {
    int kind;
    size_t a, b;     /* the text, or the tag's inside */
    int minus_open;  /* {%- */
    int plus_open;   /* {%+ */
    int minus_close; /* -%} */
    int plus_close;  /* +%} */
    int line;
};

struct parser {
    struct janas_jinja *t;
    const char *src;
    size_t n;
    struct seg *sg;
    size_t nsg, at;
    /* the tokens of the tag being read */
    struct tok *tk;
    size_t ntk, ti;
    int line;
    int depth; /* expressions and blocks inside each other */
    int failed;
    char *err;
    size_t err_len;
};

#define MAX_NESTING 200 /* the parser's own stack, bounded */

static void perr(struct parser *p, const char *fmt, ...)
    __attribute__((format(gnu_printf, 2, 3)));
static void perr(struct parser *p, const char *fmt, ...)
{
    if (p->failed)
        return;
    p->failed = 1;
    char m[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(m, sizeof m, fmt, ap);
    va_end(ap);
    snprintf(p->err, p->err_len, "template, line %d: %s", p->line, m);
}

static int line_at(const char *s, size_t pos)
{
    int l = 1;
    for (size_t i = 0; i < pos; i++)
        l += s[i] == '\n';
    return l;
}

/* the end of a tag's inside: its closing, past the strings in it */
static size_t tag_end(const char *s, size_t n, size_t i, char close)
{
    char q = 0;
    for (; i + 1 < n; i++) {
        char c = s[i];
        if (q) {
            if (c == '\\' && i + 1 < n)
                i++;
            else if (c == q)
                q = 0;
            continue;
        }
        if (c == '\'' || c == '"')
            q = c;
        else if (c == close && s[i + 1] == '}')
            return i;
    }
    return (size_t)-1;
}

static int add_seg(struct parser *p, struct seg s)
{
    if (!(p->nsg & 63)) {
        struct seg *g = realloc(p->sg, (p->nsg + 64) * sizeof *g);
        if (!g) {
            perr(p, "out of memory");
            return -1;
        }
        p->sg = g;
    }
    p->sg[p->nsg++] = s;
    return 0;
}

static int is_space(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}

/* the words of a block tag's inside, for {% raw %} */
static int block_is(const char *s, size_t a, size_t b, const char *w)
{
    while (a < b && is_space(s[a]))
        a++;
    while (b > a && is_space(s[b - 1]))
        b--;
    return b - a == strlen(w) && !memcmp(s + a, w, b - a);
}

static int cut(struct parser *p)
{
    const char *s = p->src;
    size_t n = p->n, i = 0, text = 0;
    while (i < n) {
        if (s[i] != '{' || i + 1 >= n ||
            (s[i + 1] != '{' && s[i + 1] != '%' && s[i + 1] != '#')) {
            i++;
            continue;
        }
        char open = s[i + 1];
        char close = open == '{' ? '}' : open;
        size_t in = i + 2;
        struct seg g = {.kind = open == '{'   ? SG_VAR
                                : open == '%' ? SG_BLOCK
                                              : SG_COMMENT,
                        .line = line_at(s, i)};
        if (in < n && s[in] == '-')
            g.minus_open = 1, in++;
        else if (in < n && s[in] == '+' && open != '{')
            g.plus_open = 1, in++;
        size_t e;
        if (open == '#') {
            e = (size_t)-1;
            for (size_t k = in; k + 1 < n; k++)
                if (s[k] == '#' && s[k + 1] == '}') {
                    e = k;
                    break;
                }
        } else
            e = tag_end(s, n, in, close);
        if (e == (size_t)-1) {
            p->line = g.line;
            perr(p, "a tag that is never closed");
            return -1;
        }
        size_t end = e;
        if (end > in && s[end - 1] == '-')
            g.minus_close = 1, end--;
        else if (end > in && s[end - 1] == '+' && open != '{')
            g.plus_close = 1, end--;
        g.a = in, g.b = end;
        if (add_seg(p, (struct seg){.kind = SG_TEXT,
                                    .a = text,
                                    .b = i,
                                    .line = line_at(s, text)}) != 0 ||
            add_seg(p, g) != 0)
            return -1;
        i = e + 2;
        text = i;
        /* {% raw %}: the text up to {% endraw %} as it is */
        if (g.kind == SG_BLOCK && block_is(s, g.a, g.b, "raw")) {
            size_t k = i, found = (size_t)-1, after = 0;
            while (k + 1 < n && found == (size_t)-1) {
                if (s[k] == '{' && s[k + 1] == '%') {
                    size_t q = k + 2, mo = 0;
                    if (q < n && s[q] == '-')
                        q++, mo = 1;
                    size_t ee = tag_end(s, n, q, '%');
                    if (ee != (size_t)-1) {
                        size_t be = ee;
                        if (be > q && s[be - 1] == '-')
                            be--;
                        if (block_is(s, q, be, "endraw")) {
                            found = k;
                            after = ee + 2;
                            size_t tb = k;
                            if (mo)
                                while (tb > i && is_space(s[tb - 1]))
                                    tb--;
                            p->sg[p->nsg - 1].kind = SG_COMMENT;
                            if (add_seg(p, (struct seg){.kind = SG_TEXT,
                                                        .a = i,
                                                        .b = tb,
                                                        .line = g.line,
                                                        .plus_open = 2}) != 0)
                                return -1;
                            break;
                        }
                    }
                }
                k++;
            }
            if (found == (size_t)-1) {
                p->line = g.line;
                perr(p, "raw without endraw");
                return -1;
            }
            i = text = after;
        }
    }
    return add_seg(
        p, (struct seg){
               .kind = SG_TEXT, .a = text, .b = n, .line = line_at(s, text)});
}

/* Whitespace: the tags' "-", then trim_blocks and lstrip_blocks. A text
   segment marked plus_open == 2 is raw: left alone. */
static void whitespace(struct parser *p)
{
    const char *s = p->src;
    for (size_t i = 0; i < p->nsg; i++) {
        struct seg *g = &p->sg[i];
        if (g->kind == SG_TEXT)
            continue;
        struct seg *prev =
            i > 0 && p->sg[i - 1].kind == SG_TEXT && p->sg[i - 1].plus_open != 2
                ? &p->sg[i - 1]
                : NULL;
        struct seg *next = i + 1 < p->nsg && p->sg[i + 1].kind == SG_TEXT &&
                                   p->sg[i + 1].plus_open != 2
                               ? &p->sg[i + 1]
                               : NULL;
        int block = g->kind == SG_BLOCK || g->kind == SG_COMMENT;
        if (prev) {
            if (g->minus_open) {
                while (prev->b > prev->a && is_space(s[prev->b - 1]))
                    prev->b--;
            } else if (block && !g->plus_open) {
                /* lstrip_blocks: spaces and tabs from the line's start */
                size_t k = prev->b;
                while (k > prev->a && (s[k - 1] == ' ' || s[k - 1] == '\t'))
                    k--;
                if (k == 0 || s[k - 1] == '\n')
                    prev->b = k;
            }
        }
        if (next) {
            if (g->minus_close) {
                while (next->a < next->b && is_space(s[next->a]))
                    next->a++;
            } else if (block && !g->plus_close) {
                /* trim_blocks: the newline after it */
                if (next->a < next->b && s[next->a] == '\n')
                    next->a++;
                else if (next->a + 1 < next->b && s[next->a] == '\r' &&
                         s[next->a + 1] == '\n')
                    next->a += 2;
            }
        }
    }
}

/* ---- tokens of a tag */
enum { T_END, T_NAME, T_INT, T_FLOAT, T_STR, T_OP };
struct tok {
    int t;
    const char *p; /* the source of a name or operator */
    size_t n;
    int64_t i;
    double f;
    struct jstr s;
};

static int utf8_put(char *o, uint32_t c)
{
    if (c < 0x80)
        return o[0] = (char)c, 1;
    if (c < 0x800)
        return o[0] = (char)(0xc0 | c >> 6), o[1] = (char)(0x80 | (c & 63)), 2;
    if (c < 0x10000)
        return o[0] = (char)(0xe0 | c >> 12),
               o[1] = (char)(0x80 | ((c >> 6) & 63)),
               o[2] = (char)(0x80 | (c & 63)), 3;
    return o[0] = (char)(0xf0 | c >> 18),
           o[1] = (char)(0x80 | ((c >> 12) & 63)),
           o[2] = (char)(0x80 | ((c >> 6) & 63)),
           o[3] = (char)(0x80 | (c & 63)), 4;
}

static int hexval(const char *s, int n, uint32_t *out)
{
    uint32_t v = 0;
    for (int i = 0; i < n; i++) {
        char c = s[i];
        int d = c >= '0' && c <= '9'   ? c - '0'
                : c >= 'a' && c <= 'f' ? c - 'a' + 10
                : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                       : -1;
        if (d < 0)
            return -1;
        v = v * 16 + (uint32_t)d;
    }
    *out = v;
    return 0;
}

static int lex(struct parser *p, size_t a, size_t b)
{
    const char *s = p->src;
    p->ntk = p->ti = 0;
    size_t cap = 16;
    free(p->tk);
    p->tk = malloc(cap * sizeof *p->tk);
    if (!p->tk) {
        perr(p, "out of memory");
        return -1;
    }
    size_t i = a;
    for (;;) {
        while (i < b && is_space(s[i]))
            i++;
        if (p->ntk + 1 >= cap) {
            cap *= 2;
            struct tok *t = realloc(p->tk, cap * sizeof *t);
            if (!t) {
                perr(p, "out of memory");
                return -1;
            }
            p->tk = t;
        }
        struct tok *t = &p->tk[p->ntk];
        memset(t, 0, sizeof *t);
        if (i >= b) {
            t->t = T_END;
            p->ntk++;
            return 0;
        }
        char c = s[i];
        if (c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            size_t k = i;
            while (k < b && (s[k] == '_' || (s[k] >= 'a' && s[k] <= 'z') ||
                             (s[k] >= 'A' && s[k] <= 'Z') ||
                             (s[k] >= '0' && s[k] <= '9')))
                k++;
            t->t = T_NAME, t->p = s + i, t->n = k - i;
            i = k;
        } else if (c >= '0' && c <= '9') {
            size_t k = i;
            int fl = 0;
            while (k < b && ((s[k] >= '0' && s[k] <= '9') || s[k] == '_'))
                k++;
            if (k + 1 < b && s[k] == '.' && s[k + 1] >= '0' &&
                s[k + 1] <= '9') {
                fl = 1;
                k++;
                while (k < b && s[k] >= '0' && s[k] <= '9')
                    k++;
            }
            if (k < b && (s[k] == 'e' || s[k] == 'E')) {
                size_t q = k + 1;
                if (q < b && (s[q] == '+' || s[q] == '-'))
                    q++;
                if (q < b && s[q] >= '0' && s[q] <= '9') {
                    fl = 1;
                    k = q;
                    while (k < b && s[k] >= '0' && s[k] <= '9')
                        k++;
                }
            }
            char num[64];
            size_t m = 0;
            for (size_t q = i; q < k && m + 1 < sizeof num; q++)
                if (s[q] != '_')
                    num[m++] = s[q];
            num[m] = 0;
            if (fl)
                t->t = T_FLOAT, t->f = strtod(num, NULL);
            else
                t->t = T_INT, t->i = strtoll(num, NULL, 10);
            i = k;
        } else if (c == '\'' || c == '"') {
            size_t k = i + 1;
            char *o = ja_alloc(&p->t->ar, b - i + 1);
            if (!o) {
                perr(p, "out of memory");
                return -1;
            }
            size_t m = 0;
            while (k < b && s[k] != c) {
                if (s[k] == '\\' && k + 1 < b) {
                    char e = s[++k];
                    uint32_t u;
                    switch (e) {
                    case 'n':
                        o[m++] = '\n';
                        break;
                    case 't':
                        o[m++] = '\t';
                        break;
                    case 'r':
                        o[m++] = '\r';
                        break;
                    case 'b':
                        o[m++] = '\b';
                        break;
                    case 'f':
                        o[m++] = '\f';
                        break;
                    case 'v':
                        o[m++] = '\v';
                        break;
                    case '0':
                        o[m++] = '\0';
                        break;
                    case 'x':
                        if (k + 2 < b && !hexval(s + k + 1, 2, &u)) {
                            m += (size_t)utf8_put(o + m, u);
                            k += 2;
                        } else
                            o[m++] = '\\', o[m++] = e;
                        break;
                    case 'u':
                        if (k + 4 < b && !hexval(s + k + 1, 4, &u)) {
                            m += (size_t)utf8_put(o + m, u);
                            k += 4;
                        } else
                            o[m++] = '\\', o[m++] = e;
                        break;
                    case '\\':
                    case '\'':
                    case '"':
                        o[m++] = e;
                        break;
                    case '\n':
                        break; /* a line continued */
                    default:
                        o[m++] = '\\', o[m++] = e;
                    }
                    k++;
                } else
                    o[m++] = s[k++];
            }
            if (k >= b) {
                perr(p, "a string that is never closed");
                return -1;
            }
            t->t = T_STR, t->s = (struct jstr){.p = o, .n = m};
            i = k + 1;
        } else {
            static const char *const two[] = {
                "**", "//", "==", "!=", "<=", ">="};
            size_t l = 1;
            for (size_t q = 0; q < sizeof two / sizeof *two; q++)
                if (i + 1 < b && s[i] == two[q][0] && s[i + 1] == two[q][1])
                    l = 2;
            if (l == 1 && !strchr("+-*/%~<>=()[]{}.,:|!", c)) {
                perr(p, "unexpected '%c'", c);
                return -1;
            }
            t->t = T_OP, t->p = s + i, t->n = l;
            i += l;
        }
        p->ntk++;
    }
}

/* ---- the grammar */
static struct tok *peek(struct parser *p)
{
    return &p->tk[p->ti];
}
static int is_op(struct parser *p, const char *op)
{
    struct tok *t = peek(p);
    return t->t == T_OP && t->n == strlen(op) && !memcmp(t->p, op, t->n);
}
static int is_name(struct parser *p, const char *w)
{
    struct tok *t = peek(p);
    return t->t == T_NAME && t->n == strlen(w) && !memcmp(t->p, w, t->n);
}
static int accept_op(struct parser *p, const char *op)
{
    if (!is_op(p, op))
        return 0;
    p->ti++;
    return 1;
}
static int accept_name(struct parser *p, const char *w)
{
    if (!is_name(p, w))
        return 0;
    p->ti++;
    return 1;
}
static void expect_op(struct parser *p, const char *op)
{
    if (!accept_op(p, op))
        perr(p, "'%s' expected", op);
}
static struct jstr take_name(struct parser *p)
{
    struct tok *t = peek(p);
    if (t->t != T_NAME) {
        perr(p, "a name expected");
        return (struct jstr){.p = "", .n = 0};
    }
    p->ti++;
    char *c = ja_alloc(&p->t->ar, t->n + 1);
    if (!c) {
        perr(p, "out of memory");
        return (struct jstr){.p = "", .n = 0};
    }
    memcpy(c, t->p, t->n);
    return (struct jstr){.p = c, .n = t->n};
}

static struct jnode *node(struct parser *p, enum jkind k)
{
    struct jnode *n = ja_alloc(&p->t->ar, sizeof *n);
    if (!n)
        perr(p, "out of memory");
    else
        n->k = k, n->line = p->line;
    return n;
}

/* an expression's height from its parts; too tall a tree is refused, so
   that running it cannot go deeper than the stack */
#define MAX_HEIGHT 1000
static struct jnode *tall(struct parser *p, struct jnode *n)
{
    if (!n)
        return n;
    int h = 0;
    for (size_t i = 0; i < n->na; i++)
        if (n->a[i] && n->a[i]->height > h)
            h = n->a[i]->height;
    n->height = h + 1;
    if (n->height > MAX_HEIGHT)
        perr(p, "an expression more than %d operations deep", MAX_HEIGHT);
    return n;
}

/* a growing array of node pointers in the arena */
struct nvec {
    struct jnode **v;
    size_t n, cap;
};
static void nv_add(struct parser *p, struct nvec *a, struct jnode *x)
{
    if (a->n == a->cap) {
        size_t cap = a->cap ? 2 * a->cap : 4;
        struct jnode **v = ja_alloc(&p->t->ar, cap * sizeof *v);
        if (!v) {
            perr(p, "out of memory");
            return;
        }
        if (a->n)
            memcpy(v, a->v, a->n * sizeof *v);
        a->v = v, a->cap = cap;
    }
    a->v[a->n++] = x;
}
struct svec {
    struct jstr *v;
    size_t n, cap;
};
static void sv_add(struct parser *p, struct svec *a, struct jstr x)
{
    if (a->n == a->cap) {
        size_t cap = a->cap ? 2 * a->cap : 4;
        struct jstr *v = ja_alloc(&p->t->ar, cap * sizeof *v);
        if (!v) {
            perr(p, "out of memory");
            return;
        }
        if (a->n)
            memcpy(v, a->v, a->n * sizeof *v);
        a->v = v, a->cap = cap;
    }
    a->v[a->n++] = x;
}

static struct jnode *expr(struct parser *p);
static struct jnode *or_expr(struct parser *p);
static struct jnode *unary(struct parser *p, int filters);

static struct jnode *binary(struct parser *p, int op, struct jnode *a,
                            struct jnode *b)
{
    struct jnode *n = node(p, E_BINARY);
    struct nvec v = {0};
    nv_add(p, &v, a);
    nv_add(p, &v, b);
    if (n)
        n->op = op, n->a = v.v, n->na = v.n;
    return tall(p, n);
}

/* (args, name=value...) after its '(' */
static void call_args(struct parser *p, struct nvec *args, struct svec *kw,
                      size_t *npos)
{
    *npos = 0;
    while (!p->failed && !is_op(p, ")")) {
        if (peek(p)->t == T_NAME && p->tk[p->ti + 1].t == T_OP &&
            p->tk[p->ti + 1].n == 1 && p->tk[p->ti + 1].p[0] == '=') {
            sv_add(p, kw, take_name(p));
            p->ti++;
            nv_add(p, args, expr(p));
        } else {
            if (kw->n)
                perr(p, "a positional argument after a keyword one");
            nv_add(p, args, expr(p));
            (*npos)++;
        }
        if (!accept_op(p, ","))
            break;
    }
    expect_op(p, ")");
}

static struct jnode *primary(struct parser *p)
{
    struct tok *t = peek(p);
    struct jnode *n = NULL;
    if (p->failed)
        return NULL;
    if (t->t == T_NAME) {
        if (is_name(p, "true") || is_name(p, "True") || is_name(p, "false") ||
            is_name(p, "False")) {
            n = node(p, E_LIT);
            if (n)
                n->v = jv_bool(t->p[0] == 't' || t->p[0] == 'T');
            p->ti++;
        } else if (is_name(p, "none") || is_name(p, "None")) {
            n = node(p, E_LIT);
            if (n)
                n->v = jv_none();
            p->ti++;
        } else {
            n = node(p, E_NAME);
            struct jstr s = take_name(p);
            if (n)
                n->name = s;
        }
    } else if (t->t == T_INT || t->t == T_FLOAT) {
        n = node(p, E_LIT);
        if (n)
            n->v = t->t == T_INT ? jv_int(t->i) : jv_float(t->f);
        p->ti++;
    } else if (t->t == T_STR) {
        /* adjacent strings are one */
        struct jstr s = t->s;
        p->ti++;
        while (peek(p)->t == T_STR) {
            struct jstr x = peek(p)->s;
            char *c = ja_alloc(&p->t->ar, s.n + x.n + 1);
            if (!c) {
                perr(p, "out of memory");
                return NULL;
            }
            memcpy(c, s.p, s.n);
            memcpy(c + s.n, x.p, x.n);
            s = (struct jstr){.p = c, .n = s.n + x.n};
            p->ti++;
        }
        n = node(p, E_LIT);
        if (n)
            n->v = jv_str(s);
    } else if (accept_op(p, "(")) {
        if (accept_op(p, ")"))
            return node(p, E_TUPLE);
        struct jnode *e = expr(p);
        if (accept_op(p, ",")) {
            struct nvec v = {0};
            nv_add(p, &v, e);
            while (!p->failed && !is_op(p, ")")) {
                nv_add(p, &v, expr(p));
                if (!accept_op(p, ","))
                    break;
            }
            n = node(p, E_TUPLE);
            if (n)
                n->a = v.v, n->na = v.n;
            n = tall(p, n);
        } else
            n = e;
        expect_op(p, ")");
    } else if (accept_op(p, "[")) {
        struct nvec v = {0};
        while (!p->failed && !is_op(p, "]")) {
            nv_add(p, &v, expr(p));
            if (!accept_op(p, ","))
                break;
        }
        expect_op(p, "]");
        n = node(p, E_LIST);
        if (n)
            n->a = v.v, n->na = v.n;
        n = tall(p, n);
    } else if (accept_op(p, "{")) {
        struct nvec v = {0};
        while (!p->failed && !is_op(p, "}")) {
            nv_add(p, &v, expr(p));
            expect_op(p, ":");
            nv_add(p, &v, expr(p));
            if (!accept_op(p, ","))
                break;
        }
        expect_op(p, "}");
        n = node(p, E_DICT);
        if (n)
            n->a = v.v, n->na = v.n;
        n = tall(p, n);
    } else
        perr(p, "an expression expected");
    return n;
}

static struct jnode *postfix(struct parser *p, struct jnode *n)
{
    while (!p->failed) {
        if (accept_op(p, ".")) {
            struct jnode *a = node(p, E_ATTR);
            struct jstr nm;
            if (peek(p)->t == T_INT) { /* x.0 */
                char *c = ja_alloc(&p->t->ar, 24);
                if (!c) {
                    perr(p, "out of memory");
                    return NULL;
                }
                snprintf(c, 24, "%lld", (long long)peek(p)->i);
                nm = (struct jstr){.p = c, .n = strlen(c)};
                p->ti++;
            } else
                nm = take_name(p);
            struct nvec v = {0};
            nv_add(p, &v, n);
            if (a)
                a->name = nm, a->a = v.v, a->na = v.n;
            n = tall(p, a);
        } else if (accept_op(p, "[")) {
            struct jnode *x[3] = {NULL, NULL, NULL};
            int sl = 0;
            if (!is_op(p, ":"))
                x[0] = expr(p);
            if (accept_op(p, ":")) {
                sl = 1;
                if (!is_op(p, "]") && !is_op(p, ":"))
                    x[1] = expr(p);
                if (accept_op(p, ":") && !is_op(p, "]"))
                    x[2] = expr(p);
            }
            expect_op(p, "]");
            struct jnode *a = node(p, sl ? E_SLICE : E_INDEX);
            struct nvec v = {0};
            nv_add(p, &v, n);
            nv_add(p, &v, x[0]);
            if (sl) {
                nv_add(p, &v, x[1]);
                nv_add(p, &v, x[2]);
            }
            if (a)
                a->a = v.v, a->na = v.n;
            n = tall(p, a);
        } else if (accept_op(p, "(")) {
            struct nvec v = {0};
            struct svec kw = {0};
            size_t npos;
            nv_add(p, &v, n);
            call_args(p, &v, &kw, &npos);
            struct jnode *a = node(p, E_CALL);
            if (a)
                a->a = v.v, a->na = v.n, a->names = kw.v, a->nn = kw.n;
            n = tall(p, a);
        } else
            break;
    }
    return n;
}

/* the names that end an expression where a test's argument could be */
static int stops(struct parser *p)
{
    static const char *const W[] = {"else", "or",  "and", "if",       "is",
                                    "in",   "not", "for", "recursive"};
    for (size_t i = 0; i < sizeof W / sizeof *W; i++)
        if (is_name(p, W[i]))
            return 1;
    return 0;
}

static struct jnode *filters(struct parser *p, struct jnode *n)
{
    while (!p->failed) {
        if (accept_op(p, "|")) {
            struct jnode *f = node(p, E_FILTER);
            struct jstr nm = take_name(p);
            /* a dotted filter name has no meaning here: the name alone */
            struct nvec v = {0};
            struct svec kw = {0};
            nv_add(p, &v, n);
            if (accept_op(p, "(")) {
                size_t npos;
                call_args(p, &v, &kw, &npos);
            }
            if (f)
                f->name = nm, f->a = v.v, f->na = v.n, f->names = kw.v,
                f->nn = kw.n;
            n = tall(p, f);
        } else if (accept_name(p, "is")) {
            struct jnode *f = node(p, E_TEST);
            int neg = accept_name(p, "not");
            struct jstr nm;
            /* "is none", "is true": names that are also literals */
            struct tok *t = peek(p);
            if (t->t == T_NAME) {
                nm = take_name(p);
            } else {
                perr(p, "a test's name expected");
                return NULL;
            }
            struct nvec v = {0};
            struct svec kw = {0};
            nv_add(p, &v, n);
            if (accept_op(p, "(")) {
                size_t npos;
                call_args(p, &v, &kw, &npos);
            } else {
                struct tok *a = peek(p);
                if ((a->t == T_INT || a->t == T_FLOAT || a->t == T_STR ||
                     (a->t == T_NAME && !stops(p)) || is_op(p, "[") ||
                     is_op(p, "{")) &&
                    a->t != T_END)
                    nv_add(p, &v, postfix(p, primary(p)));
            }
            if (f)
                f->name = nm, f->op = neg, f->a = v.v, f->na = v.n;
            n = tall(p, f);
        } else if (is_op(p, "(")) {
            n = postfix(p, n);
        } else
            break;
    }
    return n;
}

static struct jnode *unary(struct parser *p, int with_filters)
{
    if (p->failed)
        return NULL;
    if (is_op(p, "-") || is_op(p, "+")) {
        if (p->depth + 1 > MAX_NESTING) {
            perr(p, "signs nested more than %d deep", MAX_NESTING);
            return NULL;
        }
        p->depth++;
        int op = is_op(p, "-") ? OP_NEG : OP_POS;
        p->ti++;
        struct jnode *n = node(p, E_UNARY);
        struct nvec v = {0};
        nv_add(p, &v, unary(p, 0));
        p->depth--;
        if (n)
            n->op = op, n->a = v.v, n->na = v.n;
        n = tall(p, n);
        return with_filters ? filters(p, n) : n;
    }
    struct jnode *n = postfix(p, primary(p));
    return with_filters ? filters(p, n) : n;
}

static struct jnode *power(struct parser *p)
{
    struct jnode *n = unary(p, 1);
    while (!p->failed && accept_op(p, "**"))
        n = binary(p, OP_POW, n, unary(p, 1));
    return n;
}

static struct jnode *math2(struct parser *p)
{
    struct jnode *n = power(p);
    while (!p->failed) {
        int op = is_op(p, "*")    ? OP_MUL
                 : is_op(p, "/")  ? OP_DIV
                 : is_op(p, "//") ? OP_FLOORDIV
                 : is_op(p, "%")  ? OP_MOD
                                  : 0;
        if (!op)
            break;
        p->ti++;
        n = binary(p, op, n, power(p));
    }
    return n;
}

static struct jnode *concat(struct parser *p)
{
    struct jnode *n = math2(p);
    while (!p->failed && accept_op(p, "~"))
        n = binary(p, OP_CONCAT, n, math2(p));
    return n;
}

static struct jnode *math1(struct parser *p)
{
    struct jnode *n = concat(p);
    while (!p->failed) {
        int op = is_op(p, "+") ? OP_ADD : is_op(p, "-") ? OP_SUB : 0;
        if (!op)
            break;
        p->ti++;
        n = binary(p, op, n, concat(p));
    }
    return n;
}

static struct jnode *compare(struct parser *p)
{
    struct jnode *n = math1(p);
    while (!p->failed) {
        int op = is_op(p, "==")     ? OP_EQ
                 : is_op(p, "!=")   ? OP_NE
                 : is_op(p, "<=")   ? OP_LE
                 : is_op(p, ">=")   ? OP_GE
                 : is_op(p, "<")    ? OP_LT
                 : is_op(p, ">")    ? OP_GT
                 : is_name(p, "in") ? OP_IN
                                    : 0;
        if (!op && is_name(p, "not") && p->tk[p->ti + 1].t == T_NAME &&
            p->tk[p->ti + 1].n == 2 && !memcmp(p->tk[p->ti + 1].p, "in", 2)) {
            p->ti++;
            op = OP_NOTIN;
        }
        if (!op)
            break;
        p->ti++;
        n = binary(p, op, n, math1(p));
    }
    return n;
}

static struct jnode *not_expr(struct parser *p)
{
    if (p->failed)
        return NULL;
    if (accept_name(p, "not")) {
        if (p->depth + 1 > MAX_NESTING) {
            perr(p, "'not' nested more than %d deep", MAX_NESTING);
            return NULL;
        }
        struct jnode *n = node(p, E_UNARY);
        struct nvec v = {0};
        p->depth++;
        nv_add(p, &v, not_expr(p));
        p->depth--;
        if (n)
            n->op = OP_NOT, n->a = v.v, n->na = v.n;
        return tall(p, n);
    }
    return compare(p);
}

static struct jnode *and_expr(struct parser *p)
{
    struct jnode *n = not_expr(p);
    while (!p->failed && accept_name(p, "and"))
        n = binary(p, OP_AND, n, not_expr(p));
    return n;
}

static struct jnode *or_expr(struct parser *p)
{
    struct jnode *n = and_expr(p);
    while (!p->failed && accept_name(p, "or"))
        n = binary(p, OP_OR, n, and_expr(p));
    return n;
}

static struct jnode *expr(struct parser *p)
{
    if (++p->depth > MAX_NESTING) {
        perr(p, "expressions nested more than %d deep", MAX_NESTING);
        p->depth--;
        return NULL;
    }
    struct jnode *n = or_expr(p);
    if (!p->failed && accept_name(p, "if")) {
        struct jnode *c = node(p, E_COND);
        struct nvec v = {0};
        nv_add(p, &v, or_expr(p));
        nv_add(p, &v, n);
        nv_add(p, &v, accept_name(p, "else") ? expr(p) : NULL);
        if (c)
            c->a = v.v, c->na = v.n;
        n = tall(p, c);
    }
    p->depth--;
    return n;
}

/* the end of a tag: nothing left in it */
static void tag_done(struct parser *p)
{
    if (!p->failed && peek(p)->t != T_END)
        perr(p, "unexpected '%.*s' in the tag",
             (int)(peek(p)->n ? peek(p)->n : 1), peek(p)->p ? peek(p)->p : "?");
}

/* ---- statements */
static int body_in(struct parser *p, const char *const *stop, size_t nstop,
                   struct nvec *out, struct jstr *ended);
static int parse_body(struct parser *p, const char *const *stop, size_t nstop,
                      struct nvec *out, struct jstr *ended);

/* the block keyword a tag starts with, its tokens read */
static int open_tag(struct parser *p, const struct seg *g, struct jstr *kw)
{
    p->line = g->line;
    if (lex(p, g->a, g->b) != 0)
        return -1;
    *kw = (struct jstr){.p = "", .n = 0};
    if (peek(p)->t != T_NAME) {
        perr(p, "a tag without a statement");
        return -1;
    }
    *kw = (struct jstr){.p = peek(p)->p, .n = peek(p)->n};
    p->ti++;
    return 0;
}

static int word(struct jstr s, const char *w)
{
    return s.n == strlen(w) && !memcmp(s.p, w, s.n);
}

static struct jnode *if_stmt(struct parser *p)
{
    struct jnode *n = node(p, S_IF);
    struct nvec c = {0}, body = {0}, orelse = {0};
    nv_add(p, &c, expr(p));
    tag_done(p);
    static const char *const S[] = {"elif", "else", "endif"};
    struct jstr ended;
    if (parse_body(p, S, 3, &body, &ended) != 0)
        return NULL;
    if (word(ended, "elif")) {
        if (++p->depth > MAX_NESTING) {
            perr(p, "more than %d elif", MAX_NESTING);
            return NULL;
        }
        nv_add(p, &orelse, if_stmt(p));
        p->depth--;
    } else if (word(ended, "else")) {
        tag_done(p);
        static const char *const E[] = {"endif"};
        if (parse_body(p, E, 1, &orelse, &ended) != 0)
            return NULL;
        tag_done(p);
    } else
        tag_done(p);
    if (n)
        n->a = c.v, n->na = c.n, n->body = body.v, n->nbody = body.n,
        n->orelse = orelse.v, n->norelse = orelse.n;
    return n;
}

static int parse_stmt(struct parser *p, struct jstr kw, struct nvec *out)
{
    struct jnode *n = NULL;
    struct jstr ended;
    if (word(kw, "if")) {
        n = if_stmt(p);
    } else if (word(kw, "for")) {
        n = node(p, S_FOR);
        struct svec t = {0};
        do
            sv_add(p, &t, take_name(p));
        while (!p->failed && accept_op(p, ","));
        if (!accept_name(p, "in"))
            perr(p, "'in' expected");
        struct nvec a = {0}, body = {0}, orelse = {0};
        /* the iterable: no "x if y" there, the if filters the loop */
        struct jnode *it = or_expr(p);
        if (accept_op(p, ",")) { /* for x in a, b */
            struct jnode *tu = node(p, E_TUPLE);
            struct nvec v = {0};
            nv_add(p, &v, it);
            do
                nv_add(p, &v, or_expr(p));
            while (!p->failed && accept_op(p, ","));
            if (tu)
                tu->a = v.v, tu->na = v.n;
            it = tu;
        }
        nv_add(p, &a, it);
        nv_add(p, &a, accept_name(p, "if") ? or_expr(p) : NULL);
        accept_name(p, "recursive");
        tag_done(p);
        static const char *const S[] = {"else", "endfor"};
        if (parse_body(p, S, 2, &body, &ended) != 0)
            return -1;
        if (word(ended, "else")) {
            tag_done(p);
            static const char *const E[] = {"endfor"};
            if (parse_body(p, E, 1, &orelse, &ended) != 0)
                return -1;
        }
        tag_done(p);
        if (n)
            n->names = t.v, n->nn = t.n, n->a = a.v, n->na = a.n,
            n->body = body.v, n->nbody = body.n, n->orelse = orelse.v,
            n->norelse = orelse.n;
    } else if (word(kw, "set")) {
        struct svec t = {0};
        struct jstr first = take_name(p);
        struct nvec a = {0};
        if (accept_op(p, ".")) { /* ns.x = ... */
            struct jstr attr = take_name(p);
            expect_op(p, "=");
            n = node(p, S_SET);
            struct jnode *ns = node(p, E_NAME);
            if (ns)
                ns->name = first;
            nv_add(p, &a, expr(p));
            nv_add(p, &a, ns);
            tag_done(p);
            if (n)
                n->name = attr, n->a = a.v, n->na = a.n;
        } else {
            sv_add(p, &t, first);
            while (!p->failed && accept_op(p, ","))
                sv_add(p, &t, take_name(p));
            if (accept_op(p, "=")) {
                n = node(p, S_SET);
                struct jnode *e = expr(p);
                if (accept_op(p, ",")) { /* set a, b = 1, 2 */
                    struct jnode *tu = node(p, E_TUPLE);
                    struct nvec v = {0};
                    nv_add(p, &v, e);
                    do
                        nv_add(p, &v, expr(p));
                    while (!p->failed && accept_op(p, ","));
                    if (tu)
                        tu->a = v.v, tu->na = v.n;
                    e = tu;
                }
                nv_add(p, &a, e);
                tag_done(p);
                if (n)
                    n->names = t.v, n->nn = t.n, n->a = a.v, n->na = a.n;
            } else {
                /* {% set x %}...{% endset %}, a filter after the name */
                n = node(p, S_SETBLOCK);
                struct nvec body = {0}, f = {0};
                struct jstr fname = {.p = "", .n = 0};
                if (accept_op(p, "|")) {
                    fname = take_name(p);
                    if (accept_op(p, "(")) {
                        struct svec kwv = {0};
                        size_t npos;
                        call_args(p, &f, &kwv, &npos);
                    }
                }
                tag_done(p);
                static const char *const E[] = {"endset"};
                if (parse_body(p, E, 1, &body, &ended) != 0)
                    return -1;
                tag_done(p);
                if (n)
                    n->names = t.v, n->nn = t.n, n->body = body.v,
                    n->nbody = body.n, n->name = fname, n->a = f.v, n->na = f.n;
            }
        }
    } else if (word(kw, "macro")) {
        n = node(p, S_MACRO);
        struct jstr name = take_name(p);
        struct svec par = {0};
        struct nvec def = {0}, body = {0};
        expect_op(p, "(");
        while (!p->failed && !is_op(p, ")")) {
            sv_add(p, &par, take_name(p));
            nv_add(p, &def, accept_op(p, "=") ? expr(p) : NULL);
            if (!accept_op(p, ","))
                break;
        }
        expect_op(p, ")");
        tag_done(p);
        static const char *const E[] = {"endmacro"};
        if (parse_body(p, E, 1, &body, &ended) != 0)
            return -1;
        tag_done(p);
        if (n)
            n->name = name, n->names = par.v, n->nn = par.n, n->a = def.v,
            n->na = def.n, n->body = body.v, n->nbody = body.n;
    } else if (word(kw, "break") || word(kw, "continue")) {
        n = node(p, word(kw, "break") ? S_BREAK : S_CONTINUE);
        tag_done(p);
    } else if (word(kw, "filter")) {
        n = node(p, S_FILTERBLOCK);
        struct jstr name = take_name(p);
        struct nvec a = {0}, body = {0};
        if (accept_op(p, "(")) {
            struct svec kwv = {0};
            size_t npos;
            call_args(p, &a, &kwv, &npos);
        }
        tag_done(p);
        static const char *const E[] = {"endfilter"};
        if (parse_body(p, E, 1, &body, &ended) != 0)
            return -1;
        tag_done(p);
        if (n)
            n->name = name, n->a = a.v, n->na = a.n, n->body = body.v,
            n->nbody = body.n;
    } else if (word(kw, "generation") || word(kw, "block")) {
        /* Hugging Face's mark of the assistant's text, a named block:
           what is inside, as it is */
        int gen = word(kw, "generation");
        if (!gen)
            take_name(p);
        tag_done(p);
        static const char *const G[] = {"endgeneration"};
        static const char *const B[] = {"endblock"};
        struct nvec body = {0};
        if (parse_body(p, gen ? G : B, 1, &body, &ended) != 0)
            return -1;
        for (size_t i = 0; i < body.n; i++)
            nv_add(p, out, body.v[i]);
        return p->failed ? -1 : 0;
    } else {
        perr(p, "'%.*s' is not a statement this interpreter knows", (int)kw.n,
             kw.p);
        return -1;
    }
    if (p->failed)
        return -1;
    nv_add(p, out, n);
    return p->failed ? -1 : 0;
}

static int parse_body(struct parser *p, const char *const *stop, size_t nstop,
                      struct nvec *out, struct jstr *ended)
{
    if (++p->depth > MAX_NESTING) {
        perr(p, "blocks nested more than %d deep", MAX_NESTING);
        return -1;
    }
    int rc = body_in(p, stop, nstop, out, ended);
    p->depth--;
    return rc;
}

static int body_in(struct parser *p, const char *const *stop, size_t nstop,
                   struct nvec *out, struct jstr *ended)
{
    if (ended)
        *ended = (struct jstr){.p = "", .n = 0};
    while (p->at < p->nsg && !p->failed) {
        const struct seg *g = &p->sg[p->at++];
        p->line = g->line;
        if (g->kind == SG_COMMENT)
            continue;
        if (g->kind == SG_TEXT) {
            if (g->b <= g->a)
                continue;
            struct jnode *n = node(p, S_TEXT);
            if (n)
                n->v =
                    jv_str((struct jstr){.p = p->src + g->a, .n = g->b - g->a});
            nv_add(p, out, n);
        } else if (g->kind == SG_VAR) {
            if (lex(p, g->a, g->b) != 0)
                return -1;
            struct jnode *n = node(p, S_OUT);
            struct nvec v = {0};
            nv_add(p, &v, expr(p));
            tag_done(p);
            if (n)
                n->a = v.v, n->na = v.n;
            nv_add(p, out, n);
        } else {
            struct jstr kw;
            if (open_tag(p, g, &kw) != 0)
                return -1;
            for (size_t i = 0; i < nstop; i++)
                if (word(kw, stop[i])) {
                    if (ended)
                        *ended = kw;
                    return 0;
                }
            if (parse_stmt(p, kw, out) != 0)
                return -1;
        }
    }
    if (p->failed)
        return -1;
    if (nstop) {
        perr(p, "'%s' never comes", stop[nstop - 1]);
        return -1;
    }
    return 0;
}

struct janas_jinja *janas_jinja_parse(const char *src, size_t n, char *err,
                                      size_t err_len)
{
    struct janas_jinja *t = calloc(1, sizeof *t);
    if (!t) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    struct parser p = {
        .t = t, .n = n, .err = err, .err_len = err_len, .line = 1};
    char *copy = ja_alloc(&t->ar, n + 1);
    if (!copy) {
        snprintf(err, err_len, "out of memory");
        free(t);
        return NULL;
    }
    memcpy(copy, src, n);
    p.src = copy;
    struct nvec body = {0};
    if (cut(&p) == 0) {
        whitespace(&p);
        parse_body(&p, NULL, 0, &body, NULL);
    }
    free(p.sg);
    free(p.tk);
    if (p.failed) {
        janas_jinja_free(t);
        return NULL;
    }
    t->body = body.v;
    t->nbody = body.n;
    return t;
}

void janas_jinja_free(struct janas_jinja *t)
{
    if (!t)
        return;
    ja_free(&t->ar);
    free(t);
}
