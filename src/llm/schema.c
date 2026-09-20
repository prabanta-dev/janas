/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * schema.c - JSON and JSON Schema as grammar rules (see schema.h).
 */
#include "schema.h"

#include <stdlib.h>
#include <string.h>

#define WS_MAX 20     /* white space bytes in a row */
#define DIGITS_MAX 20 /* digits of a number's parts */
#define LEN_MAX 64    /* string lengths enforced up to this */
#define ITEMS_MAX 32  /* array bounds enforced up to this */
#define DEPTH_MAX 64  /* schemas inside schemas */
#define NONE UINT32_MAX

struct memo {
    const struct janas_json *s;
    uint32_t rule;
};

struct janas_jsg {
    struct janas_gbuild *b;
    uint32_t ws, value, object, array, string, number, integer, digits, esc;
    struct memo *memo;
    size_t n_memo, cap_memo;
    int relaxed, depth;
};

struct janas_jsg *janas_jsg_new(struct janas_gbuild *b)
{
    struct janas_jsg *j = calloc(1, sizeof(*j));
    if (!j)
        return NULL;
    j->b = b;
    j->ws = j->value = j->object = j->array = NONE;
    j->string = j->number = j->integer = j->digits = j->esc = NONE;
    return j;
}

void janas_jsg_free(struct janas_jsg *j)
{
    if (!j)
        return;
    free(j->memo);
    free(j);
}

int janas_jsg_relaxed(const struct janas_jsg *j)
{
    return j->relaxed;
}

/* A rule of one alternative: the literal. */
static uint32_t lit_rule(struct janas_jsg *j, const char *s, size_t n)
{
    uint32_t r = janas_gb_rule(j->b);
    janas_gb_lit(j->b, janas_gb_alt(j->b, r), s, n);
    return r;
}

/* A rule that matches nothing: the schema false. */
static uint32_t never(struct janas_jsg *j)
{
    return janas_gb_rule(j->b);
}

uint32_t janas_jsg_ws(struct janas_jsg *j)
{
    if (j->ws != NONE)
        return j->ws;
    struct janas_gbuild *b = j->b;
    uint8_t sp[32];
    janas_set_clear(sp);
    janas_set_add(sp, ' ', ' ');
    janas_set_add(sp, '\t', '\n');
    janas_set_add(sp, '\r', '\r');
    uint32_t next = janas_gb_empty(b);
    for (int i = 0; i < WS_MAX; i++) { /* W_i: nothing, or a space and W_i+1 */
        uint32_t r = janas_gb_rule(b);
        janas_gb_alt(b, r);
        uint32_t a = janas_gb_alt(b, r);
        janas_gb_set(b, a, sp);
        janas_gb_ref(b, a, next);
        next = r;
    }
    return j->ws = next;
}

/* Up to `more` further digits: D = nothing | digit D'. */
static uint32_t digits(struct janas_jsg *j, int more)
{
    uint8_t d[32];
    janas_set_clear(d);
    janas_set_add(d, '0', '9');
    uint32_t next = janas_gb_empty(j->b);
    for (int i = 0; i < more; i++) {
        uint32_t r = janas_gb_rule(j->b);
        janas_gb_alt(j->b, r);
        uint32_t a = janas_gb_alt(j->b, r);
        janas_gb_set(j->b, a, d);
        janas_gb_ref(j->b, a, next);
        next = r;
    }
    return next;
}

/* -?(0|[1-9][0-9]*), with the fraction and exponent when real */
static uint32_t number_rule(struct janas_jsg *j, int real)
{
    struct janas_gbuild *b = j->b;
    if (j->digits == NONE)
        j->digits = digits(j, DIGITS_MAX - 1);
    uint8_t d[32], nz[32];
    janas_set_clear(d);
    janas_set_add(d, '0', '9');
    janas_set_clear(nz);
    janas_set_add(nz, '1', '9');
    uint32_t in = janas_gb_rule(b); /* the integer part */
    janas_gb_byte(b, janas_gb_alt(b, in), '0');
    uint32_t a = janas_gb_alt(b, in);
    janas_gb_set(b, a, nz);
    janas_gb_ref(b, a, j->digits);
    uint32_t sign = janas_gb_rule(b);
    janas_gb_alt(b, sign);
    janas_gb_byte(b, janas_gb_alt(b, sign), '-');
    uint32_t r = janas_gb_rule(b);
    a = janas_gb_alt(b, r);
    janas_gb_ref(b, a, sign);
    janas_gb_ref(b, a, in);
    if (!real)
        return r;
    uint32_t frac = janas_gb_rule(b);
    janas_gb_alt(b, frac);
    a = janas_gb_alt(b, frac);
    janas_gb_byte(b, a, '.');
    janas_gb_set(b, a, d);
    janas_gb_ref(b, a, j->digits);
    uint32_t ex = janas_gb_rule(b), es = janas_gb_rule(b);
    janas_gb_alt(b, es);
    janas_gb_byte(b, janas_gb_alt(b, es), '+');
    janas_gb_byte(b, janas_gb_alt(b, es), '-');
    janas_gb_alt(b, ex);
    uint8_t e[32];
    janas_set_clear(e);
    janas_set_add(e, 'e', 'e');
    janas_set_add(e, 'E', 'E');
    a = janas_gb_alt(b, ex);
    janas_gb_set(b, a, e);
    janas_gb_ref(b, a, es);
    janas_gb_set(b, a, d);
    janas_gb_ref(b, a, digits(j, 3));
    uint32_t real_r = janas_gb_rule(b);
    a = janas_gb_alt(b, real_r);
    janas_gb_ref(b, a, sign);
    janas_gb_ref(b, a, in);
    janas_gb_ref(b, a, frac);
    janas_gb_ref(b, a, ex);
    return real_r;
}

/* After an escaping backslash. */
static uint32_t escape_rule(struct janas_jsg *j)
{
    if (j->esc != NONE)
        return j->esc;
    struct janas_gbuild *b = j->b;
    uint8_t one[32], hex[32];
    janas_set_clear(one);
    const char *simple = "\"\\/bfnrt";
    for (const char *p = simple; *p; p++)
        janas_set_add(one, (uint8_t)*p, (uint8_t)*p);
    janas_set_clear(hex);
    janas_set_add(hex, '0', '9');
    janas_set_add(hex, 'a', 'f');
    janas_set_add(hex, 'A', 'F');
    uint32_t r = janas_gb_rule(b);
    janas_gb_set(b, janas_gb_alt(b, r), one);
    uint32_t a = janas_gb_alt(b, r);
    janas_gb_byte(b, a, 'u');
    for (int i = 0; i < 4; i++)
        janas_gb_set(b, a, hex);
    return j->esc = r;
}

/* The bytes a string holds as they are: anything but the quote, the
   backslash and the control characters. */
static void plain_set(uint8_t set[32])
{
    janas_set_clear(set);
    janas_set_add(set, 0x20, 0xff);
    set['"' >> 3] &= (uint8_t)~(1u << ('"' & 7));
    set['\\' >> 3] &= (uint8_t)~(1u << ('\\' & 7));
}

/*
 * A string of lo to hi characters (hi < 0: no limit), quotes included. A
 * character is a plain byte or an escape; a byte of a longer UTF-8
 * character counts as one, so the bounds are in bytes for non-ASCII text.
 */
static uint32_t string_rule(struct janas_jsg *j, int lo, int hi)
{
    struct janas_gbuild *b = j->b;
    uint8_t plain[32];
    plain_set(plain);
    uint32_t esc = escape_rule(j);
    int n = hi >= 0 ? hi : lo; /* states made one by one */
    uint32_t *st = malloc(((size_t)n + 1) * sizeof(uint32_t));
    if (!st)
        return never(j);
    for (int k = 0; k <= n; k++)
        st[k] = janas_gb_rule(b);
    for (int k = 0; k <= n; k++) {
        if (k >= lo)
            janas_gb_byte(b, janas_gb_alt(b, st[k]), '"');
        uint32_t next = k < n ? st[k + 1] : hi < 0 ? st[k] : NONE;
        if (next == NONE)
            continue;
        uint32_t a = janas_gb_alt(b, st[k]);
        janas_gb_set(b, a, plain);
        janas_gb_ref(b, a, next);
        a = janas_gb_alt(b, st[k]);
        janas_gb_byte(b, a, '\\');
        janas_gb_ref(b, a, esc);
        janas_gb_ref(b, a, next);
    }
    uint32_t r = janas_gb_rule(b);
    uint32_t a = janas_gb_alt(b, r);
    janas_gb_byte(b, a, '"');
    janas_gb_ref(b, a, st[0]);
    free(st);
    return r;
}

/*
 * An array of lo to hi items (hi < 0: no limit) of rule item:
 * "[" ws "]" when lo is 0, or "[" ws item ws C1, where Ck (k items so far)
 * closes when k >= lo and takes one more when k < hi.
 */
static uint32_t array_rule(struct janas_jsg *j, uint32_t item, int lo, int hi)
{
    struct janas_gbuild *b = j->b;
    uint32_t ws = janas_jsg_ws(j);
    int n = hi >= 0 ? hi : (lo > 1 ? lo : 1);
    uint32_t *c = malloc(((size_t)n + 1) * sizeof(uint32_t));
    if (!c)
        return never(j);
    for (int k = 1; k <= n; k++)
        c[k] = janas_gb_rule(b);
    for (int k = 1; k <= n; k++) {
        if (k >= lo)
            janas_gb_byte(b, janas_gb_alt(b, c[k]), ']');
        uint32_t next = k < n ? c[k + 1] : hi < 0 ? c[k] : NONE;
        if (next == NONE)
            continue;
        uint32_t a = janas_gb_alt(b, c[k]);
        janas_gb_byte(b, a, ',');
        janas_gb_ref(b, a, ws);
        janas_gb_ref(b, a, item);
        janas_gb_ref(b, a, ws);
        janas_gb_ref(b, a, next);
    }
    uint32_t r = janas_gb_rule(b);
    if (lo == 0) {
        uint32_t a = janas_gb_alt(b, r);
        janas_gb_byte(b, a, '[');
        janas_gb_ref(b, a, ws);
        janas_gb_byte(b, a, ']');
    }
    if (hi != 0) {
        uint32_t a = janas_gb_alt(b, r);
        janas_gb_byte(b, a, '[');
        janas_gb_ref(b, a, ws);
        janas_gb_ref(b, a, item);
        janas_gb_ref(b, a, ws);
        janas_gb_ref(b, a, c[1]);
    }
    free(c);
    return r;
}

/* "{" ws "}" | "{" ws key ws ":" ws value ws R, R = "}" | "," ws member R,
   with keys any string and values of rule val */
static uint32_t map_rule(struct janas_jsg *j, uint32_t val)
{
    struct janas_gbuild *b = j->b;
    uint32_t ws = janas_jsg_ws(j), key = string_rule(j, 0, -1);
    uint32_t member = janas_gb_rule(b), rest = janas_gb_rule(b);
    uint32_t a = janas_gb_alt(b, member);
    janas_gb_ref(b, a, key);
    janas_gb_ref(b, a, ws);
    janas_gb_byte(b, a, ':');
    janas_gb_ref(b, a, ws);
    janas_gb_ref(b, a, val);
    janas_gb_ref(b, a, ws);
    janas_gb_byte(b, janas_gb_alt(b, rest), '}');
    a = janas_gb_alt(b, rest);
    janas_gb_byte(b, a, ',');
    janas_gb_ref(b, a, ws);
    janas_gb_ref(b, a, member);
    janas_gb_ref(b, a, rest);
    uint32_t r = janas_gb_rule(b);
    a = janas_gb_alt(b, r);
    janas_gb_byte(b, a, '{');
    janas_gb_ref(b, a, ws);
    janas_gb_byte(b, a, '}');
    a = janas_gb_alt(b, r);
    janas_gb_byte(b, a, '{');
    janas_gb_ref(b, a, ws);
    janas_gb_ref(b, a, member);
    janas_gb_ref(b, a, rest);
    return r;
}

uint32_t janas_jsg_any(struct janas_jsg *j)
{
    if (j->value != NONE)
        return j->value;
    struct janas_gbuild *b = j->b;
    j->value = janas_gb_rule(b); /* first: the containers refer to it */
    if (j->string == NONE)
        j->string = string_rule(j, 0, -1);
    if (j->number == NONE)
        j->number = number_rule(j, 1);
    j->object = map_rule(j, j->value);
    j->array = array_rule(j, j->value, 0, -1);
    uint32_t parts[] = {j->object, j->array, j->string, j->number};
    for (int i = 0; i < 4; i++)
        janas_gb_ref(b, janas_gb_alt(b, j->value), parts[i]);
    static const char *const words[] = {"true", "false", "null"};
    for (int i = 0; i < 3; i++)
        janas_gb_lit(b, janas_gb_alt(b, j->value), words[i], strlen(words[i]));
    return j->value;
}

uint32_t janas_jsg_object(struct janas_jsg *j)
{
    janas_jsg_any(j);
    return j->object;
}

/* ---- schemas ---- */

/* The schema a "#/..." reference names, or NULL. */
static const struct janas_json *resolve(const struct janas_json *root,
                                        const char *ref)
{
    if (ref[0] != '#')
        return NULL;
    const struct janas_json *v = root;
    const char *p = ref + 1;
    while (v && *p == '/') {
        p++;
        char part[256];
        size_t n = 0;
        while (*p && *p != '/' && n + 1 < sizeof(part)) {
            if (p[0] == '~' && (p[1] == '0' || p[1] == '1')) {
                part[n++] = p[1] == '0' ? '~' : '/';
                p += 2;
            } else {
                part[n++] = *p++;
            }
        }
        part[n] = 0;
        if (v->type == JANAS_JSON_OBJECT) {
            v = janas_json_get(v, part);
        } else if (v->type == JANAS_JSON_ARRAY) {
            long i = strtol(part, NULL, 10);
            const struct janas_json *c = v->child;
            while (c && i-- > 0)
                c = c->next;
            v = c;
        } else {
            v = NULL;
        }
    }
    return *p ? NULL : v;
}

static uint32_t compile(struct janas_jsg *j, const struct janas_json *s,
                        const struct janas_json *root);

/*
 * A value as the only thing allowed (const, an item of enum): its text, with
 * white space allowed where JSON allows it, so {"a": 1} and {"a":1} are the
 * same value as they are for whoever reads it.
 */
static void literal(struct janas_jsg *j, uint32_t a, const struct janas_json *v)
{
    struct janas_gbuild *b = j->b;
    if (v->type != JANAS_JSON_ARRAY && v->type != JANAS_JSON_OBJECT) {
        struct janas_buf t = {0};
        janas_json_write(&t, v);
        if (!t.oom)
            janas_gb_lit(b, a, t.p, t.n);
        janas_buf_free(&t);
        return;
    }
    uint32_t ws = janas_jsg_ws(j);
    int obj = v->type == JANAS_JSON_OBJECT;
    janas_gb_byte(b, a, obj ? '{' : '[');
    janas_gb_ref(b, a, ws);
    for (const struct janas_json *c = v->child; c; c = c->next) {
        if (c != v->child) {
            janas_gb_byte(b, a, ',');
            janas_gb_ref(b, a, ws);
        }
        if (obj) {
            struct janas_buf k = {0};
            janas_json_write_str(&k, c->key, c->key_n);
            if (!k.oom)
                janas_gb_lit(b, a, k.p, k.n);
            janas_buf_free(&k);
            janas_gb_ref(b, a, ws);
            janas_gb_byte(b, a, ':');
            janas_gb_ref(b, a, ws);
        }
        literal(j, a, c);
        janas_gb_ref(b, a, ws);
    }
    janas_gb_byte(b, a, obj ? '}' : ']');
}

static void add_literal(struct janas_jsg *j, uint32_t r,
                        const struct janas_json *v)
{
    literal(j, janas_gb_alt(j->b, r), v);
}

static int bound(const struct janas_json *s, const char *key, int max, int def)
{
    const struct janas_json *v = janas_json_get(s, key);
    if (!v || v->type != JANAS_JSON_NUMBER)
        return def;
    double x = janas_json_num(v, def);
    return x < 0 ? def : x > max ? -2 : (int)x;
}

static int required(const struct janas_json *s, const struct janas_json *key)
{
    const struct janas_json *req = janas_json_get(s, "required");
    if (!req || req->type != JANAS_JSON_ARRAY)
        return 0;
    for (const struct janas_json *r = req->child; r; r = r->next)
        if (r->type == JANAS_JSON_STRING && r->n == key->key_n &&
            memcmp(r->s, key->key, r->n) == 0)
            return 1;
    return 0;
}

/*
 * The properties in the schema's order: R[i][x] is "from property i on, x
 * members written so far (0 or some)". Each step writes property i (after a
 * comma if something came before) or, when it is not required, skips it.
 */
static uint32_t object_rule(struct janas_jsg *j, const struct janas_json *s,
                            const struct janas_json *root)
{
    struct janas_gbuild *b = j->b;
    const struct janas_json *props = janas_json_get(s, "properties");
    const struct janas_json *extra = janas_json_get(s, "additionalProperties");
    if (!props || props->type != JANAS_JSON_OBJECT || props->n == 0) {
        if (extra && extra->type == JANAS_JSON_OBJECT)
            return map_rule(j, compile(j, extra, root));
        return janas_jsg_object(j);
    }
    if (!extra || extra->type != JANAS_JSON_FALSE)
        j->relaxed++; /* others allowed by the schema, never written */
    uint32_t ws = janas_jsg_ws(j);
    size_t n = props->n;
    uint32_t(*rr)[2] = malloc((n + 1) * sizeof(*rr));
    if (!rr)
        return never(j);
    for (size_t i = 0; i <= n; i++)
        for (int x = 0; x < 2; x++)
            rr[i][x] = janas_gb_rule(b);
    for (int x = 0; x < 2; x++)
        janas_gb_alt(b, rr[n][x]);
    size_t i = 0;
    for (const struct janas_json *p = props->child; p; p = p->next, i++) {
        uint32_t val = compile(j, p, root);
        struct janas_buf key = {0};
        janas_json_write_str(&key, p->key, p->key_n);
        int req = required(s, p);
        for (int x = 0; x < 2; x++) {
            if (!req)
                janas_gb_ref(b, janas_gb_alt(b, rr[i][x]), rr[i + 1][x]);
            uint32_t a = janas_gb_alt(b, rr[i][x]);
            if (x) {
                janas_gb_byte(b, a, ',');
                janas_gb_ref(b, a, ws);
            }
            janas_gb_lit(b, a, key.p ? key.p : "", key.n);
            janas_gb_ref(b, a, ws);
            janas_gb_byte(b, a, ':');
            janas_gb_ref(b, a, ws);
            janas_gb_ref(b, a, val);
            janas_gb_ref(b, a, ws);
            janas_gb_ref(b, a, rr[i + 1][1]);
        }
        janas_buf_free(&key);
    }
    uint32_t r = janas_gb_rule(b);
    uint32_t a = janas_gb_alt(b, r);
    janas_gb_byte(b, a, '{');
    janas_gb_ref(b, a, ws);
    janas_gb_ref(b, a, rr[0][0]);
    janas_gb_byte(b, a, '}');
    free(rr);
    return r;
}

/* The schema with one type, t, of those it lists. */
static uint32_t typed(struct janas_jsg *j, const struct janas_json *s,
                      const struct janas_json *root, const char *t)
{
    if (strcmp(t, "string") == 0) {
        int lo = bound(s, "minLength", LEN_MAX, 0);
        int hi = bound(s, "maxLength", LEN_MAX, -1);
        if (janas_json_get(s, "pattern") || janas_json_get(s, "format"))
            j->relaxed++;
        if (lo == -2 || hi == -2) {
            j->relaxed++;
            lo = lo == -2 ? 0 : lo;
            hi = -1;
        }
        if (lo == 0 && hi < 0) {
            if (j->string == NONE)
                j->string = string_rule(j, 0, -1);
            return j->string;
        }
        return string_rule(j, lo, hi < 0 ? -1 : hi < lo ? lo : hi);
    }
    if (strcmp(t, "integer") == 0 || strcmp(t, "number") == 0) {
        static const char *const lim[] = {"minimum", "maximum",
                                          "exclusiveMinimum",
                                          "exclusiveMaximum", "multipleOf"};
        for (int i = 0; i < 5; i++)
            if (janas_json_get(s, lim[i]))
                j->relaxed++;
        if (t[1] == 'n') {
            if (j->integer == NONE)
                j->integer = number_rule(j, 0);
            return j->integer;
        }
        if (j->number == NONE)
            j->number = number_rule(j, 1);
        return j->number;
    }
    if (strcmp(t, "boolean") == 0) {
        uint32_t r = janas_gb_rule(j->b);
        janas_gb_lit(j->b, janas_gb_alt(j->b, r), "true", 4);
        janas_gb_lit(j->b, janas_gb_alt(j->b, r), "false", 5);
        return r;
    }
    if (strcmp(t, "null") == 0)
        return lit_rule(j, "null", 4);
    if (strcmp(t, "object") == 0)
        return object_rule(j, s, root);
    if (strcmp(t, "array") == 0) {
        const struct janas_json *items = janas_json_get(s, "items");
        if (janas_json_get(s, "prefixItems"))
            j->relaxed++;
        uint32_t item = items && (items->type == JANAS_JSON_OBJECT ||
                                  items->type == JANAS_JSON_TRUE)
                            ? compile(j, items, root)
                            : janas_jsg_any(j);
        int lo = bound(s, "minItems", ITEMS_MAX, 0);
        int hi = bound(s, "maxItems", ITEMS_MAX, -1);
        if (lo == -2 || hi == -2) {
            j->relaxed++;
            lo = lo == -2 ? 0 : lo;
            hi = -1;
        }
        return array_rule(j, item, lo, hi < 0 ? -1 : hi < lo ? lo : hi);
    }
    j->relaxed++; /* an unknown type: anything */
    return janas_jsg_any(j);
}

static uint32_t compile_body(struct janas_jsg *j, const struct janas_json *s,
                             const struct janas_json *root, uint32_t r)
{
    struct janas_gbuild *b = j->b;
    const struct janas_json *v;
    if ((v = janas_json_get(s, "$ref")) && v->type == JANAS_JSON_STRING) {
        const struct janas_json *t = resolve(root, v->s);
        if (!t) {
            j->relaxed++;
            janas_gb_ref(b, janas_gb_alt(b, r), janas_jsg_any(j));
        } else {
            janas_gb_ref(b, janas_gb_alt(b, r), compile(j, t, root));
        }
        return r;
    }
    if ((v = janas_json_get(s, "const"))) {
        add_literal(j, r, v);
        return r;
    }
    if ((v = janas_json_get(s, "enum")) && v->type == JANAS_JSON_ARRAY) {
        for (const struct janas_json *e = v->child; e; e = e->next)
            add_literal(j, r, e);
        return r;
    }
    const struct janas_json *any = janas_json_get(s, "anyOf");
    if (!any)
        any = janas_json_get(s, "oneOf");
    if (any && any->type == JANAS_JSON_ARRAY) {
        for (const struct janas_json *e = any->child; e; e = e->next)
            janas_gb_ref(b, janas_gb_alt(b, r), compile(j, e, root));
        return r;
    }
    if ((v = janas_json_get(s, "allOf")) && v->type == JANAS_JSON_ARRAY &&
        v->n > 0) {
        if (v->n > 1)
            j->relaxed++; /* only the first one is enforced */
        janas_gb_ref(b, janas_gb_alt(b, r), compile(j, v->child, root));
        return r;
    }
    const struct janas_json *type = janas_json_get(s, "type");
    const struct janas_json *nullable = janas_json_get(s, "nullable");
    if (nullable && nullable->type == JANAS_JSON_TRUE)
        janas_gb_lit(b, janas_gb_alt(b, r), "null", 4);
    if (type && type->type == JANAS_JSON_STRING) {
        janas_gb_ref(b, janas_gb_alt(b, r), typed(j, s, root, type->s));
    } else if (type && type->type == JANAS_JSON_ARRAY) {
        for (const struct janas_json *t = type->child; t; t = t->next)
            if (t->type == JANAS_JSON_STRING)
                janas_gb_ref(b, janas_gb_alt(b, r), typed(j, s, root, t->s));
    } else if (janas_json_get(s, "properties")) {
        janas_gb_ref(b, janas_gb_alt(b, r), typed(j, s, root, "object"));
    } else if (janas_json_get(s, "items")) {
        janas_gb_ref(b, janas_gb_alt(b, r), typed(j, s, root, "array"));
    } else {
        janas_gb_ref(b, janas_gb_alt(b, r), janas_jsg_any(j));
    }
    return r;
}

static uint32_t compile(struct janas_jsg *j, const struct janas_json *s,
                        const struct janas_json *root)
{
    if (!s || s->type == JANAS_JSON_TRUE)
        return janas_jsg_any(j);
    if (s->type != JANAS_JSON_OBJECT) {
        if (s->type != JANAS_JSON_FALSE)
            j->relaxed++;
        return s->type == JANAS_JSON_FALSE ? never(j) : janas_jsg_any(j);
    }
    for (size_t i = 0; i < j->n_memo; i++)
        if (j->memo[i].s == s)
            return j->memo[i].rule;
    if (j->depth >= DEPTH_MAX) {
        j->relaxed++;
        return janas_jsg_any(j);
    }
    /* the rule is known before its body: a schema that refers to itself
       finds it here */
    uint32_t r = janas_gb_rule(j->b);
    if (j->n_memo == j->cap_memo) {
        size_t cap = j->cap_memo ? 2 * j->cap_memo : 32;
        struct memo *t = realloc(j->memo, cap * sizeof(*t));
        if (!t)
            return never(j);
        j->memo = t;
        j->cap_memo = cap;
    }
    j->memo[j->n_memo++] = (struct memo){s, r};
    j->depth++;
    compile_body(j, s, root, r);
    j->depth--;
    return r;
}

uint32_t janas_jsg_schema(struct janas_jsg *j, const struct janas_json *schema,
                          const struct janas_json *root)
{
    return compile(j, schema, root ? root : schema);
}
