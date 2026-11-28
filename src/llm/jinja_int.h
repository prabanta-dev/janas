/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * jinja_int.h - the interpreter's own parts (jinja.h): memory by arenas,
 * the values, the tree of a template.
 */
#ifndef JANAS_LLM_JINJA_INT_H
#define JANAS_LLM_JINJA_INT_H

#include <stddef.h>
#include <stdint.h>

#include "llm/jinja.h"

/* ---- memory: everything of a template, or of a rendering, freed at once */
struct jarena {
    struct jblock *head;
    size_t used;  /* bytes taken, all blocks */
    size_t limit; /* at most (0: none) */
    int over;     /* the limit was reached: every allocation fails */
};
void *ja_alloc(struct jarena *a, size_t n); /* zeroed; NULL past the limit */
void ja_free(struct jarena *a);

/* ---- values */
enum jtype {
    J_UNDEF, /* a name or member there is not */
    J_NONE,
    J_BOOL,
    J_INT,
    J_FLOAT,
    J_STR,
    J_LIST,
    J_DICT,
    J_MACRO, /* a macro of the template */
    J_FUNC,  /* a function of the interpreter, or a method bound to self */
};

/* text, and for each byte whether it came from the conversation (NULL:
   none of it did) */
struct jstr {
    const char *p;
    const uint8_t *m;
    size_t n;
};

struct jval;
struct jlist {
    struct jval *v;
    size_t n, cap;
};
struct jdict {
    struct jstr *k;
    struct jval *v;
    size_t n, cap;
    int ns; /* a namespace(): its members can be set */
};

struct jnode;
struct jval {
    enum jtype t;
    union {
        int b;
        int64_t i;
        double f;
        struct jstr s;
        struct jlist *l;
        struct jdict *d;
        const struct jnode *macro;
        struct {
            int id;
            struct jval *self;
        } fn;
    } u;
};

/* ---- the tree */
enum jkind {
    /* expressions */
    E_LIT,    /* v */
    E_NAME,   /* name */
    E_ATTR,   /* a[0].name */
    E_INDEX,  /* a[0][a[1]] */
    E_SLICE,  /* a[0][a[1]:a[2]:a[3]], any of them NULL */
    E_CALL,   /* a[0](a[1..]; names: the keywords of the last ones) */
    E_FILTER, /* a[0] | name(a[1..]) */
    E_TEST,   /* a[0] is [not: op] name(a[1..]) */
    E_UNARY,  /* op a[0] */
    E_BINARY, /* a[0] op a[1] */
    E_COND,   /* a[1] if a[0] else a[2] (NULL: undefined) */
    E_LIST,   /* [a...] */
    E_TUPLE,  /* (a, ...) */
    E_DICT,   /* {a[0]: a[1], a[2]: a[3]...} */
    /* statements */
    S_TEXT,     /* v: text of the template */
    S_OUT,      /* {{ a[0] }} */
    S_IF,       /* a[0]: condition; body; orelse */
    S_FOR,      /* names: targets; a[0]: iterable, a[1]: its filter or
                   NULL; body; orelse */
    S_SET,      /* names: targets, or name with a[1] the namespace; a[0] */
    S_SETBLOCK, /* name = body's output */
    S_MACRO,    /* name(names = a[...] defaults, NULL none); body */
    S_BREAK,
    S_CONTINUE,
    S_FILTERBLOCK, /* body's output | name(a...) */
};

struct jnode {
    enum jkind k;
    int op; /* the operator, or 1 for "is not" */
    int line;
    int height; /* an expression's: the longest way down its tree */
    struct jstr name;
    struct jval v;
    struct jnode **a;
    size_t na;
    struct jstr *names;
    size_t nn;
    struct jnode **body, **orelse;
    size_t nbody, norelse;
};

/* operators */
enum {
    OP_ADD = 1,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_FLOORDIV,
    OP_MOD,
    OP_POW,
    OP_CONCAT,
    OP_EQ,
    OP_NE,
    OP_LT,
    OP_LE,
    OP_GT,
    OP_GE,
    OP_IN,
    OP_NOTIN,
    OP_AND,
    OP_OR,
    OP_NOT,
    OP_NEG,
    OP_POS,
};

struct janas_jinja {
    struct jarena ar;
    struct jnode **body;
    size_t nbody;
};

/* ---- the rendering's state, shared by the evaluator and the library */
struct jscope {
    struct jscope *up;
    struct jstr *k;
    struct jval *v;
    size_t n, cap;
};

struct jrun {
    struct jarena ar;
    struct janas_buf *out, *marks; /* where the output goes now */
    struct jscope *globals;
    int depth;
    int stop; /* break 1, continue 2 */
    int failed;
    char *err;
    size_t err_len;
};

/* jinja_eval.c */
int jinja_fail(struct jrun *r, int line, const char *fmt, ...)
    __attribute__((format(gnu_printf, 3, 4)));
struct jval jinja_eval(struct jrun *r, const struct jnode *e);
/* a macro called: its output as a string */
struct jval jinja_call_macro(struct jrun *r, const struct jnode *m,
                             const struct jval *args, size_t nargs,
                             const struct jstr *kw, const struct jval *kwv,
                             size_t nkw, int line);

/* jinja_lib.c */
struct jval jv_undef(void);
struct jval jv_none(void);
struct jval jv_bool(int b);
struct jval jv_int(int64_t i);
struct jval jv_float(double f);
struct jval jv_str(struct jstr s);
struct jval jv_cstr(const char *s); /* a constant of the interpreter */
struct jstr js_make(struct jrun *r, const char *p, size_t n, int mark);
int js_eq(struct jstr a, const char *s);
int jv_truthy(const struct jval *v);
/* str() of a value, as Python writes it */
struct jstr jv_tostr(struct jrun *r, const struct jval *v);
/* the value written as JSON (tojson), its strings' marks kept */
struct jstr jv_tojson(struct jrun *r, const struct jval *v, int indent);
int jv_equal(const struct jval *a, const struct jval *b);
/* <0, 0, >0; -2 when they cannot be compared */
int jv_compare(const struct jval *a, const struct jval *b);
struct jval jv_from_json(struct jrun *r, const struct janas_json *j, int mark);
struct jval jv_list_new(struct jrun *r, size_t cap);
int jv_list_add(struct jrun *r, struct jval *list, struct jval v);
struct jval jv_dict_new(struct jrun *r, size_t cap);
int jv_dict_set(struct jrun *r, struct jval *dict, struct jstr k,
                struct jval v);
const struct jval *jv_dict_get(const struct jval *dict, const char *k,
                               size_t n);
/* a member: a dict's key, a method of a string, a dict or a list */
struct jval jv_attr(struct jrun *r, const struct jval *obj, struct jstr name);
struct jval jv_index(struct jrun *r, const struct jval *obj,
                     const struct jval *idx, int line);
struct jval jv_slice(struct jrun *r, const struct jval *obj,
                     const struct jval *a, const struct jval *b,
                     const struct jval *c, int line);
/* the items of something iterated: a list as it is, a dict's keys, a
   string's characters; 0, or -1 when it cannot be */
int jv_iter(struct jrun *r, const struct jval *v, struct jval *out);
struct jval jv_binary(struct jrun *r, int op, const struct jval *a,
                      const struct jval *b, int line);
/* a global function by name (range, namespace...), J_UNDEF none */
struct jval jinja_global(struct jstr name);
/* calls a function or a method */
struct jval jinja_call(struct jrun *r, const struct jval *fn,
                       const struct jval *args, size_t nargs,
                       const struct jstr *kw, const struct jval *kwv,
                       size_t nkw, int line);
struct jval jinja_filter(struct jrun *r, struct jstr name, const struct jval *v,
                         const struct jval *args, size_t nargs,
                         const struct jstr *kw, const struct jval *kwv,
                         size_t nkw, int line);
/* 1, 0, or -1 for an error (said) */
int jinja_test(struct jrun *r, struct jstr name, const struct jval *v,
               const struct jval *args, size_t nargs, int line);

#endif
