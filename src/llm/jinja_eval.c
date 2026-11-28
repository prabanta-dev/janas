/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jinja_eval.c - a template's tree run (jinja.h): its statements in order,
 * the output and its marks side by side, the names looked up from the
 * innermost scope out - a loop's body and a macro have their own, as in
 * Jinja, where a set inside a loop is gone after it and a namespace()
 * carries a value out.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/jinja_int.h"

#define MAX_DEPTH 200           /* macros and nesting, deep */
#define MAX_OUT (64u << 20)     /* bytes of output at most */
#define MAX_MEMORY (512u << 20) /* bytes of values at most */
#define MAX_STEPS 50000000ull   /* statements and loops run at most */

int jinja_fail(struct jrun *r, int line, const char *fmt, ...)
{
    if (r->failed)
        return -1;
    r->failed = 1;
    char m[400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(m, sizeof m, fmt, ap);
    va_end(ap);
    if (line > 0)
        snprintf(r->err, r->err_len, "template, line %d: %s", line, m);
    else
        snprintf(r->err, r->err_len, "%s", m);
    return -1;
}

/* ---- scopes */
static struct jscope *scope_new(struct jrun *r, struct jscope *up)
{
    struct jscope *s = ja_alloc(&r->ar, sizeof *s);
    if (s)
        s->up = up;
    return s;
}

static int scope_set(struct jrun *r, struct jscope *s, struct jstr k,
                     struct jval v)
{
    for (size_t i = 0; i < s->n; i++)
        if (s->k[i].n == k.n && !memcmp(s->k[i].p, k.p, k.n)) {
            s->v[i] = v;
            return 0;
        }
    if (s->n == s->cap) {
        size_t cap = s->cap ? 2 * s->cap : 8;
        struct jstr *nk = ja_alloc(&r->ar, cap * sizeof *nk);
        struct jval *nv = ja_alloc(&r->ar, cap * sizeof *nv);
        if (!nk || !nv)
            return jinja_fail(r, 0, "out of memory");
        if (s->n) {
            memcpy(nk, s->k, s->n * sizeof *nk);
            memcpy(nv, s->v, s->n * sizeof *nv);
        }
        s->k = nk, s->v = nv, s->cap = cap;
    }
    s->k[s->n] = k;
    s->v[s->n++] = v;
    return 0;
}

static const struct jval *scope_get(const struct jscope *s, struct jstr k)
{
    for (; s; s = s->up)
        for (size_t i = s->n; i-- > 0;)
            if (s->k[i].n == k.n && !memcmp(s->k[i].p, k.p, k.n))
                return &s->v[i];
    return NULL;
}

/* ---- output */
static void emit(struct jrun *r, struct jstr s)
{
    if (!s.n || r->failed)
        return;
    if (r->out->n + s.n > MAX_OUT) {
        jinja_fail(r, 0, "the template wrote more than %u bytes",
                   (unsigned)MAX_OUT);
        return;
    }
    janas_buf_put(r->out, s.p, s.n);
    size_t at = r->marks->n;
    janas_buf_put(r->marks, s.p, s.n); /* room, then the marks */
    if (r->out->oom || r->marks->oom) {
        jinja_fail(r, 0, "out of memory");
        return;
    }
    for (size_t i = 0; i < s.n; i++)
        r->marks->p[at + i] = (char)(s.m ? s.m[i] : 0);
}

static void run_body(struct jrun *r, struct jscope *sc, struct jnode **body,
                     size_t n);

/* ---- expressions: the scope in force, and the steps run, of this
   thread's rendering */
static _Thread_local struct jscope *tl_scope;
static _Thread_local unsigned long long tl_steps;

static struct jval eval_list(struct jrun *r, const struct jnode *e)
{
    struct jval l = jv_list_new(r, e->na);
    for (size_t i = 0; i < e->na && !r->failed; i++)
        jv_list_add(r, &l, jinja_eval(r, e->a[i]));
    return l;
}

/* the arguments of a call, a filter, a test: positional first */
static int args_of(struct jrun *r, const struct jnode *e, size_t from,
                   struct jval **pos, size_t *npos, struct jval **kwv)
{
    size_t total = e->na > from ? e->na - from : 0;
    size_t nkw = e->nn;
    *npos = total - nkw;
    struct jval *v = total ? ja_alloc(&r->ar, total * sizeof *v) : NULL;
    if (total && !v)
        return jinja_fail(r, e->line, "out of memory");
    for (size_t i = 0; i < total && !r->failed; i++)
        v[i] = jinja_eval(r, e->a[from + i]);
    *pos = v;
    *kwv = v ? v + *npos : NULL;
    return r->failed ? -1 : 0;
}

struct jval jinja_eval(struct jrun *r, const struct jnode *e)
{
    if (r->failed || !e)
        return jv_undef();
    switch (e->k) {
    case E_LIT:
        return e->v;
    case E_NAME: {
        const struct jval *v = scope_get(tl_scope, e->name);
        if (v)
            return *v;
        return jinja_global(e->name);
    }
    case E_ATTR: {
        struct jval o = jinja_eval(r, e->a[0]);
        return jv_attr(r, &o, e->name);
    }
    case E_INDEX: {
        struct jval o = jinja_eval(r, e->a[0]);
        struct jval i = jinja_eval(r, e->a[1]);
        return jv_index(r, &o, &i, e->line);
    }
    case E_SLICE: {
        struct jval o = jinja_eval(r, e->a[0]);
        struct jval x[3];
        for (int i = 0; i < 3; i++)
            x[i] = e->a[1 + i] ? jinja_eval(r, e->a[1 + i]) : jv_none();
        return jv_slice(r, &o, &x[0], &x[1], &x[2], e->line);
    }
    case E_CALL: {
        struct jval fn = jinja_eval(r, e->a[0]);
        struct jval *pos, *kwv;
        size_t npos;
        if (args_of(r, e, 1, &pos, &npos, &kwv) != 0)
            return jv_undef();
        if (fn.t == J_MACRO)
            return jinja_call_macro(r, fn.u.macro, pos, npos, e->names, kwv,
                                    e->nn, e->line);
        return jinja_call(r, &fn, pos, npos, e->names, kwv, e->nn, e->line);
    }
    case E_FILTER: {
        struct jval v = jinja_eval(r, e->a[0]);
        struct jval *pos, *kwv;
        size_t npos;
        if (args_of(r, e, 1, &pos, &npos, &kwv) != 0)
            return jv_undef();
        return jinja_filter(r, e->name, &v, pos, npos, e->names, kwv, e->nn,
                            e->line);
    }
    case E_TEST: {
        struct jval v = jinja_eval(r, e->a[0]);
        struct jval *pos, *kwv;
        size_t npos;
        if (args_of(r, e, 1, &pos, &npos, &kwv) != 0)
            return jv_undef();
        int t = jinja_test(r, e->name, &v, pos, npos, e->line);
        if (t < 0)
            return jv_undef();
        return jv_bool(e->op ? !t : t);
    }
    case E_UNARY: {
        struct jval a = jinja_eval(r, e->a[0]);
        if (e->op == OP_NOT)
            return jv_bool(!jv_truthy(&a));
        if (a.t == J_INT)
            return jv_int(e->op == OP_NEG ? -a.u.i : a.u.i);
        if (a.t == J_FLOAT)
            return jv_float(e->op == OP_NEG ? -a.u.f : a.u.f);
        if (a.t == J_BOOL)
            return jv_int(e->op == OP_NEG ? -a.u.b : a.u.b);
        jinja_fail(r, e->line, "a sign before what is not a number");
        return jv_undef();
    }
    case E_BINARY: {
        struct jval a = jinja_eval(r, e->a[0]);
        if (e->op == OP_AND)
            return jv_truthy(&a) ? jinja_eval(r, e->a[1]) : a;
        if (e->op == OP_OR)
            return jv_truthy(&a) ? a : jinja_eval(r, e->a[1]);
        struct jval b = jinja_eval(r, e->a[1]);
        return jv_binary(r, e->op, &a, &b, e->line);
    }
    case E_COND: {
        struct jval c = jinja_eval(r, e->a[0]);
        if (jv_truthy(&c))
            return jinja_eval(r, e->a[1]);
        return e->a[2] ? jinja_eval(r, e->a[2]) : jv_undef();
    }
    case E_LIST:
    case E_TUPLE:
        return eval_list(r, e);
    case E_DICT: {
        struct jval d = jv_dict_new(r, e->na / 2);
        for (size_t i = 0; i + 1 < e->na && !r->failed; i += 2) {
            struct jval k = jinja_eval(r, e->a[i]);
            struct jval v = jinja_eval(r, e->a[i + 1]);
            jv_dict_set(r, &d, jv_tostr(r, &k), v);
        }
        return d;
    }
    default:
        jinja_fail(r, e->line, "not an expression");
        return jv_undef();
    }
}

/* ---- statements */

/* a block's filter arguments, positional alone (the names of a set block
   are its targets) */
static struct jval *plain_args(struct jrun *r, const struct jnode *s)
{
    struct jval *v = s->na ? ja_alloc(&r->ar, s->na * sizeof *v) : NULL;
    if (s->na && !v) {
        jinja_fail(r, s->line, "out of memory");
        return NULL;
    }
    for (size_t i = 0; i < s->na && !r->failed; i++)
        v[i] = jinja_eval(r, s->a[i]);
    return v;
}

/* the output of a body as a string, its marks kept */
static struct jval capture(struct jrun *r, struct jscope *sc,
                           struct jnode **body, size_t n)
{
    struct janas_buf out = {0}, marks = {0};
    struct janas_buf *o = r->out, *m = r->marks;
    r->out = &out, r->marks = &marks;
    run_body(r, sc, body, n);
    r->out = o, r->marks = m;
    struct jstr s = {.p = "", .n = 0};
    if (!r->failed && out.n) {
        char *p = ja_alloc(&r->ar, out.n + 1);
        uint8_t *mk = NULL;
        int any = 0;
        for (size_t i = 0; i < out.n; i++)
            any |= marks.p[i];
        if (any)
            mk = ja_alloc(&r->ar, out.n);
        if (!p || (any && !mk))
            jinja_fail(r, 0, "out of memory");
        else {
            memcpy(p, out.p, out.n);
            if (mk)
                memcpy(mk, marks.p, out.n);
            s = (struct jstr){.p = p, .m = mk, .n = out.n};
        }
    }
    janas_buf_free(&out);
    janas_buf_free(&marks);
    return jv_str(s);
}

struct jval jinja_call_macro(struct jrun *r, const struct jnode *m,
                             const struct jval *args, size_t nargs,
                             const struct jstr *kw, const struct jval *kwv,
                             size_t nkw, int line)
{
    if (r->depth >= MAX_DEPTH) {
        jinja_fail(r, line, "macros called %d deep", MAX_DEPTH);
        return jv_undef();
    }
    if (nargs > m->nn) {
        jinja_fail(r, line, "%.*s takes %zu arguments, not %zu", (int)m->name.n,
                   m->name.p, m->nn, nargs);
        return jv_undef();
    }
    /* a macro sees the template's top level, not its caller's scope */
    struct jscope *sc = scope_new(r, r->globals);
    if (!sc) {
        jinja_fail(r, line, "out of memory");
        return jv_undef();
    }
    struct jscope *saved = tl_scope;
    tl_scope = sc;
    for (size_t i = 0; i < m->nn && !r->failed; i++) {
        struct jval v = jv_undef();
        int given = 0;
        if (i < nargs)
            v = args[i], given = 1;
        for (size_t k = 0; k < nkw; k++)
            if (kw[k].n == m->names[i].n &&
                !memcmp(kw[k].p, m->names[i].p, kw[k].n))
                v = kwv[k], given = 1;
        if (!given && i < m->na && m->a[i])
            v = jinja_eval(r, m->a[i]);
        scope_set(r, sc, m->names[i], v);
    }
    r->depth++;
    struct jval out = capture(r, sc, m->body, m->nbody);
    r->depth--;
    tl_scope = saved;
    return out;
}

/* a value's items unpacked into the loop's names */
static void bind(struct jrun *r, struct jscope *sc, const struct jnode *s,
                 const struct jval *item)
{
    if (s->nn == 1) {
        scope_set(r, sc, s->names[0], *item);
        return;
    }
    struct jval parts;
    if (jv_iter(r, item, &parts) != 0 || parts.u.l->n != s->nn) {
        jinja_fail(r, s->line, "%zu names, and not as many values to unpack",
                   s->nn);
        return;
    }
    for (size_t i = 0; i < s->nn; i++)
        scope_set(r, sc, s->names[i], parts.u.l->v[i]);
}

static void run_for(struct jrun *r, struct jscope *sc, const struct jnode *s)
{
    struct jval it = jinja_eval(r, s->a[0]);
    if (r->failed)
        return;
    struct jval items;
    if (it.t == J_UNDEF || it.t == J_NONE)
        items = jv_list_new(r, 0);
    else if (jv_iter(r, &it, &items) != 0) {
        jinja_fail(r, s->line, "what is looped over is not a sequence");
        return;
    }
    /* the loop's filter first: loop.length and loop.last count what
       passes */
    struct jval kept = items;
    if (s->a[1]) {
        kept = jv_list_new(r, items.u.l->n);
        struct jscope *f = scope_new(r, sc);
        struct jscope *saved = tl_scope;
        tl_scope = f;
        for (size_t i = 0; i < items.u.l->n && !r->failed; i++) {
            bind(r, f, s, &items.u.l->v[i]);
            struct jval c = jinja_eval(r, s->a[1]);
            if (jv_truthy(&c))
                jv_list_add(r, &kept, items.u.l->v[i]);
        }
        tl_scope = saved;
    }
    size_t n = kept.u.l->n;
    if (!n) {
        run_body(r, sc, s->orelse, s->norelse);
        return;
    }
    for (size_t i = 0; i < n && !r->failed; i++) {
        struct jscope *b = scope_new(r, sc);
        if (!b) {
            jinja_fail(r, s->line, "out of memory");
            return;
        }
        struct jval loop = jv_dict_new(r, 10);
        jv_dict_set(r, &loop, (struct jstr){.p = "index", .n = 5},
                    jv_int((int64_t)i + 1));
        jv_dict_set(r, &loop, (struct jstr){.p = "index0", .n = 6},
                    jv_int((int64_t)i));
        jv_dict_set(r, &loop, (struct jstr){.p = "revindex", .n = 8},
                    jv_int((int64_t)(n - i)));
        jv_dict_set(r, &loop, (struct jstr){.p = "revindex0", .n = 9},
                    jv_int((int64_t)(n - i - 1)));
        jv_dict_set(r, &loop, (struct jstr){.p = "first", .n = 5},
                    jv_bool(i == 0));
        jv_dict_set(r, &loop, (struct jstr){.p = "last", .n = 4},
                    jv_bool(i + 1 == n));
        jv_dict_set(r, &loop, (struct jstr){.p = "length", .n = 6},
                    jv_int((int64_t)n));
        jv_dict_set(r, &loop, (struct jstr){.p = "previtem", .n = 8},
                    i ? kept.u.l->v[i - 1] : jv_undef());
        jv_dict_set(r, &loop, (struct jstr){.p = "nextitem", .n = 8},
                    i + 1 < n ? kept.u.l->v[i + 1] : jv_undef());
        jv_dict_set(r, &loop, (struct jstr){.p = "depth", .n = 5}, jv_int(1));
        scope_set(r, b, (struct jstr){.p = "loop", .n = 4}, loop);
        bind(r, b, s, &kept.u.l->v[i]);
        run_body(r, b, s->body, s->nbody);
        if (r->stop == 1) {
            r->stop = 0;
            break;
        }
        r->stop = 0;
    }
}

static void run_stmt(struct jrun *r, struct jscope *sc, const struct jnode *s)
{
    if (++tl_steps > MAX_STEPS) {
        jinja_fail(r, s->line, "the template ran too long");
        return;
    }
    if (r->ar.over) {
        jinja_fail(r, s->line, "the template took more than %u MiB",
                   (unsigned)(MAX_MEMORY >> 20));
        return;
    }
    switch (s->k) {
    case S_TEXT:
        emit(r, s->v.u.s);
        break;
    case S_OUT: {
        struct jval v = jinja_eval(r, s->a[0]);
        if (!r->failed)
            emit(r, jv_tostr(r, &v));
        break;
    }
    case S_IF: {
        struct jval c = jinja_eval(r, s->a[0]);
        if (r->failed)
            break;
        if (jv_truthy(&c))
            run_body(r, sc, s->body, s->nbody);
        else
            run_body(r, sc, s->orelse, s->norelse);
        break;
    }
    case S_FOR:
        run_for(r, sc, s);
        break;
    case S_SET: {
        struct jval v = jinja_eval(r, s->a[0]);
        if (r->failed)
            break;
        if (s->na > 1) { /* ns.x = v */
            const struct jval *ns = scope_get(tl_scope, s->a[1]->name);
            if (!ns || ns->t != J_DICT || !ns->u.d->ns) {
                jinja_fail(r, s->line, "%.*s is not a namespace",
                           (int)s->a[1]->name.n, s->a[1]->name.p);
                break;
            }
            struct jval d = *ns;
            jv_dict_set(r, &d, s->name, v);
            break;
        }
        if (s->nn == 1)
            scope_set(r, sc, s->names[0], v);
        else {
            struct jnode tmp = *s;
            bind(r, sc, &tmp, &v);
        }
        break;
    }
    case S_SETBLOCK: {
        struct jval v = capture(r, sc, s->body, s->nbody);
        if (s->name.n) {
            struct jval *pos = plain_args(r, s);
            if (r->failed)
                break;
            v = jinja_filter(r, s->name, &v, pos, s->na, NULL, NULL, 0,
                             s->line);
        }
        if (!r->failed)
            scope_set(r, sc, s->names[0], v);
        break;
    }
    case S_MACRO: {
        struct jval m = {.t = J_MACRO, .u.macro = s};
        scope_set(r, sc, s->name, m);
        break;
    }
    case S_BREAK:
        r->stop = 1;
        break;
    case S_CONTINUE:
        r->stop = 2;
        break;
    case S_FILTERBLOCK: {
        struct jval v = capture(r, sc, s->body, s->nbody);
        struct jval *pos = plain_args(r, s);
        if (r->failed)
            break;
        v = jinja_filter(r, s->name, &v, pos, s->na, NULL, NULL, 0, s->line);
        if (!r->failed)
            emit(r, jv_tostr(r, &v));
        break;
    }
    default:
        jinja_fail(r, s->line, "not a statement");
    }
}

static void run_body(struct jrun *r, struct jscope *sc, struct jnode **body,
                     size_t n)
{
    struct jscope *saved = tl_scope;
    tl_scope = sc;
    for (size_t i = 0; i < n && !r->failed && !r->stop; i++)
        run_stmt(r, sc, body[i]);
    tl_scope = saved;
}

/* a JSON object's members into a scope */
static int load(struct jrun *r, struct jscope *sc, const struct janas_json *o,
                int mark)
{
    if (!o)
        return 0;
    if (o->type != JANAS_JSON_OBJECT)
        return jinja_fail(r, 0, "the variables are not a JSON object");
    for (const struct janas_json *m = o->child; m && !r->failed; m = m->next)
        scope_set(r, sc, js_make(r, m->key, m->key_n, 0),
                  jv_from_json(r, m, mark));
    return r->failed ? -1 : 0;
}

int janas_jinja_render(const struct janas_jinja *t,
                       const struct janas_json *vars,
                       const struct janas_json *data, struct janas_buf *out,
                       struct janas_buf *marks, char *err, size_t err_len)
{
    struct jrun r = {
        .out = out, .marks = marks, .err = err, .err_len = err_len};
    r.ar.limit = MAX_MEMORY;
    err[0] = 0;
    struct jscope *saved = tl_scope;
    unsigned long long steps = tl_steps;
    tl_steps = 0;
    /* Janas's variables, then the conversation's over them; the template's
       own sets on top */
    struct jscope *base = scope_new(&r, NULL);
    struct jscope *top = base ? scope_new(&r, base) : NULL;
    if (!top)
        jinja_fail(&r, 0, "out of memory");
    else if (load(&r, base, vars, 0) == 0 && load(&r, base, data, 1) == 0) {
        r.globals = top;
        run_body(&r, top, t->body, t->nbody);
    }
    tl_scope = saved;
    tl_steps = steps;
    ja_free(&r.ar);
    return r.failed ? -1 : 0;
}
