/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - the tools a model may call, in the words of its chat template
 * (see tools.h). The texts follow the templates of the models word for
 * word: a model reads its tools best in the form it was trained on.
 */
#include "tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct janas_toolset {
    struct janas_json_doc *doc;
    const struct janas_json **item; /* what the caller gave, per tool */
    const struct janas_json **fn;   /* the function inside it */
    uint32_t n;
};

enum janas_tool_dialect janas_tools_dialect(const char *tmpl)
{
    if (tmpl && strstr(tmpl, "<|tool_call>"))
        return JANAS_TOOLS_GEMMA;
    if (!tmpl || !strstr(tmpl, "<tool_call>"))
        return JANAS_TOOLS_NONE;
    if (!strstr(tmpl, "<function="))
        return JANAS_TOOLS_JSON;
    return strstr(tmpl, "<name>") ? JANAS_TOOLS_XML_CODER : JANAS_TOOLS_XML;
}

struct janas_toolset *janas_toolset_parse(const char *json, size_t n, char *err,
                                          size_t err_len)
{
    struct janas_toolset *t = calloc(1, sizeof(*t));
    if (!t) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    t->doc = janas_json_parse(json, n, err, err_len);
    const struct janas_json *root = janas_json_root(t->doc);
    if (!root || root->type != JANAS_JSON_ARRAY) {
        if (root)
            snprintf(err, err_len, "the tools must be an array");
        janas_toolset_free(t);
        return NULL;
    }
    t->item = calloc(root->n + 1, sizeof(*t->item));
    t->fn = calloc(root->n + 1, sizeof(*t->fn));
    if (!t->item || !t->fn) {
        snprintf(err, err_len, "out of memory");
        janas_toolset_free(t);
        return NULL;
    }
    for (const struct janas_json *v = root->child; v; v = v->next) {
        const struct janas_json *f = janas_json_get(v, "function");
        if (!f)
            f = v;
        const struct janas_json *name = janas_json_get(f, "name");
        const struct janas_json *type = janas_json_get(v, "type");
        if (type && !janas_json_is(type, "function")) {
            snprintf(err, err_len, "tool %u: only functions can be called",
                     t->n);
            janas_toolset_free(t);
            return NULL;
        }
        if (!name || name->type != JANAS_JSON_STRING || name->n == 0 ||
            memchr(name->s, '>', name->n) || memchr(name->s, '"', name->n) ||
            memchr(name->s, '\n', name->n)) {
            snprintf(err, err_len, "tool %u: a function needs a plain name",
                     t->n);
            janas_toolset_free(t);
            return NULL;
        }
        const struct janas_json *p = janas_json_get(f, "parameters");
        if (p && p->type != JANAS_JSON_OBJECT) {
            snprintf(err, err_len, "tool %u: parameters must be a schema",
                     t->n);
            janas_toolset_free(t);
            return NULL;
        }
        t->item[t->n] = v;
        t->fn[t->n] = f;
        t->n++;
    }
    return t;
}

void janas_toolset_free(struct janas_toolset *t)
{
    if (!t)
        return;
    janas_json_free(t->doc);
    free(t->item);
    free(t->fn);
    free(t);
}

uint32_t janas_toolset_count(const struct janas_toolset *t)
{
    return t ? t->n : 0;
}

int janas_toolset_find(const struct janas_toolset *t, const char *name,
                       size_t n)
{
    for (uint32_t i = 0; t && i < t->n; i++) {
        const struct janas_json *v = janas_json_get(t->fn[i], "name");
        if (v->n == n && memcmp(v->s, name, n) == 0)
            return (int)i;
    }
    return -1;
}

/* ---- the system message ---- */

static const char XML_FORMAT[] =
    "\n\nIf you choose to call a function ONLY reply in the following format "
    "with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n"
    "</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
    "specified format: an inner <function=...></function> block must be "
    "nested within <tool_call></tool_call> XML tags\n- Required parameters "
    "MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT "
    "after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about "
    "function calls\n</IMPORTANT>";

/* s without the white space at either end */
static const char *trim(const char *s, size_t *n)
{
    while (*n && (unsigned char)s[0] <= ' ') {
        s++;
        (*n)--;
    }
    while (*n && (unsigned char)s[*n - 1] <= ' ')
        (*n)--;
    return s;
}

/* A value the way Jinja's "| string" or "| tojson" would put it: a string
   as it is, anything else as JSON. */
static void put_value(struct janas_buf *out, const struct janas_json *v)
{
    if (v->type == JANAS_JSON_STRING)
        janas_buf_put(out, v->s, v->n);
    else
        janas_json_write(out, v);
}

/* The keys of an object other than those named, as <key>value</key>. */
static void extra_keys(struct janas_buf *out, const struct janas_json *obj,
                       const char *const *handled)
{
    if (!obj || obj->type != JANAS_JSON_OBJECT)
        return;
    for (const struct janas_json *k = obj->child; k; k = k->next) {
        int skip = 0;
        for (int i = 0; handled[i] && !skip; i++)
            skip = strlen(handled[i]) == k->key_n &&
                   memcmp(handled[i], k->key, k->key_n) == 0;
        if (skip)
            continue;
        janas_buf_puts(out, "\n<");
        janas_buf_put(out, k->key, k->key_n);
        janas_buf_puts(out, ">");
        put_value(out, k);
        janas_buf_puts(out, "</");
        janas_buf_put(out, k->key, k->key_n);
        janas_buf_puts(out, ">");
    }
}

/* ---- Gemma 4's notation ---- */

#define GQ "<|\"|>" /* its string quote, a token of its own */

/* Jinja's dictsort: an object's members by key, case-insensitive. */
static size_t sorted(const struct janas_json *obj,
                     const struct janas_json **out, size_t cap)
{
    size_t n = 0;
    for (const struct janas_json *k = obj ? obj->child : NULL; k && n < cap;
         k = k->next) {
        size_t i = n++;
        for (; i > 0; i--) {
            const struct janas_json *p = out[i - 1];
            size_t m = p->key_n < k->key_n ? p->key_n : k->key_n;
            int c = 0;
            for (size_t x = 0; x < m && !c; x++) {
                int a = (unsigned char)p->key[x], b = (unsigned char)k->key[x];
                a = a >= 'A' && a <= 'Z' ? a + 32 : a;
                b = b >= 'A' && b <= 'Z' ? b + 32 : b;
                c = a - b;
            }
            if (c < 0 || (c == 0 && p->key_n <= k->key_n))
                break;
            out[i] = p;
        }
        out[i] = k;
    }
    return n;
}

/* The template's format_argument: strings quoted, keys quoted or not. */
static void g_arg(struct janas_buf *out, const struct janas_json *v, int ek)
{
    switch (v->type) {
    case JANAS_JSON_NULL:
        janas_buf_puts(out, "null");
        break;
    case JANAS_JSON_TRUE:
        janas_buf_puts(out, "true");
        break;
    case JANAS_JSON_FALSE:
        janas_buf_puts(out, "false");
        break;
    case JANAS_JSON_STRING:
        janas_buf_puts(out, GQ);
        janas_buf_put(out, v->s, v->n);
        janas_buf_puts(out, GQ);
        break;
    case JANAS_JSON_OBJECT: {
        const struct janas_json *k[256];
        size_t n = sorted(v, k, 256);
        janas_buf_puts(out, "{");
        for (size_t i = 0; i < n; i++) {
            if (i)
                janas_buf_puts(out, ",");
            if (ek)
                janas_buf_puts(out, GQ);
            janas_buf_put(out, k[i]->key, k[i]->key_n);
            if (ek)
                janas_buf_puts(out, GQ);
            janas_buf_puts(out, ":");
            g_arg(out, k[i], ek);
        }
        janas_buf_puts(out, "}");
        break;
    }
    case JANAS_JSON_ARRAY:
        janas_buf_puts(out, "[");
        for (const struct janas_json *e = v->child; e; e = e->next) {
            g_arg(out, e, ek);
            if (e->next)
                janas_buf_puts(out, ",");
        }
        janas_buf_puts(out, "]");
        break;
    default:
        janas_buf_put(out, v->s, v->n);
    }
}

/* A JSON string upper-cased, as Jinja's "| upper". */
static void g_upper(struct janas_buf *out, const struct janas_json *v)
{
    for (size_t i = 0; v && v->type == JANAS_JSON_STRING && i < v->n; i++) {
        char c = v->s[i];
        janas_buf_put(out, &(char){c >= 'a' && c <= 'z' ? c - 32 : c}, 1);
    }
}

static int g_type_is(const struct janas_json *v, const char *up)
{
    struct janas_buf b = {0};
    g_upper(&b, janas_json_get(v, "type"));
    int r = b.n == strlen(up) && b.p && memcmp(b.p, up, b.n) == 0;
    janas_buf_free(&b);
    return r;
}

static void g_required(struct janas_buf *out, const struct janas_json *req)
{
    janas_buf_puts(out, "required:[");
    for (const struct janas_json *e = req ? req->child : NULL; e; e = e->next) {
        janas_buf_puts(out, GQ);
        janas_buf_put(out, e->s, e->n);
        janas_buf_puts(out, GQ);
        if (e->next)
            janas_buf_puts(out, ",");
    }
    janas_buf_puts(out, "]");
}

/* The template's format_parameters. */
static void g_params(struct janas_buf *out, const struct janas_json *props,
                     int filter)
{
    static const char *const std[] = {"description", "type", "properties",
                                      "required", "nullable"};
    const struct janas_json *k[256];
    size_t n = sorted(props, k, 256);
    int first = 1;
    for (size_t i = 0; i < n; i++) {
        const struct janas_json *v = k[i];
        int skip = 0;
        for (size_t s = 0; filter && s < 5; s++)
            skip |=
                strlen(std[s]) == v->key_n && !memcmp(std[s], v->key, v->key_n);
        if (skip)
            continue;
        if (!first)
            janas_buf_puts(out, ",");
        first = 0;
        janas_buf_put(out, v->key, v->key_n);
        janas_buf_puts(out, ":{");
        int comma = 0;
        const struct janas_json *d = janas_json_get(v, "description");
        if (d && d->type == JANAS_JSON_STRING && d->n) {
            janas_buf_puts(out, "description:" GQ);
            janas_buf_put(out, d->s, d->n);
            janas_buf_puts(out, GQ);
            comma = 1;
        }
        const struct janas_json *en = janas_json_get(v, "enum");
        const struct janas_json *it = janas_json_get(v, "items");
        if (g_type_is(v, "STRING") && en && en->n) {
            janas_buf_puts(out, comma ? ",enum:" : "enum:");
            comma = 1;
            g_arg(out, en, 1);
        } else if (g_type_is(v, "ARRAY") && it &&
                   it->type == JANAS_JSON_OBJECT && it->n) {
            janas_buf_puts(out, comma ? ",items:{" : "items:{");
            comma = 1;
            const struct janas_json *ik[64];
            size_t in = sorted(it, ik, 64);
            int f2 = 1;
            for (size_t x = 0; x < in; x++) {
                const struct janas_json *iv = ik[x];
                if (iv->type == JANAS_JSON_NULL)
                    continue;
                if (!f2)
                    janas_buf_puts(out, ",");
                f2 = 0;
                if (iv->key_n == 10 && !memcmp(iv->key, "properties", 10)) {
                    janas_buf_puts(out, "properties:{");
                    if (iv->type == JANAS_JSON_OBJECT)
                        g_params(out, iv, 0);
                    janas_buf_puts(out, "}");
                } else if (iv->key_n == 8 && !memcmp(iv->key, "required", 8)) {
                    g_required(out, iv);
                } else if (iv->key_n == 4 && !memcmp(iv->key, "type", 4)) {
                    janas_buf_puts(out, "type:");
                    if (iv->type == JANAS_JSON_STRING) {
                        janas_buf_puts(out, GQ);
                        g_upper(out, iv);
                        janas_buf_puts(out, GQ);
                    } else {
                        janas_buf_puts(out, "[");
                        for (const struct janas_json *e = iv->child; e;
                             e = e->next) {
                            janas_buf_puts(out, GQ);
                            g_upper(out, e);
                            janas_buf_puts(out, GQ);
                            if (e->next)
                                janas_buf_puts(out, ",");
                        }
                        janas_buf_puts(out, "]");
                    }
                } else {
                    janas_buf_put(out, iv->key, iv->key_n);
                    janas_buf_puts(out, ":");
                    g_arg(out, iv, 1);
                }
            }
            janas_buf_puts(out, "}");
        }
        const struct janas_json *nl = janas_json_get(v, "nullable");
        if (nl && nl->type == JANAS_JSON_TRUE) {
            janas_buf_puts(out, comma ? ",nullable:true" : "nullable:true");
            comma = 1;
        }
        if (g_type_is(v, "OBJECT")) {
            const struct janas_json *pp = janas_json_get(v, "properties");
            janas_buf_puts(out, comma ? ",properties:{" : "properties:{");
            comma = 1;
            if (pp && pp->type == JANAS_JSON_OBJECT)
                g_params(out, pp, 0);
            else
                g_params(out, v, 1);
            janas_buf_puts(out, "}");
            const struct janas_json *rq = janas_json_get(v, "required");
            if (rq && rq->n) {
                janas_buf_puts(out, ",");
                g_required(out, rq);
            }
        }
        janas_buf_puts(out, comma ? ",type:" GQ : "type:" GQ);
        g_upper(out, janas_json_get(v, "type"));
        janas_buf_puts(out, GQ "}");
    }
}

/* The template's format_function_declaration, trimmed. */
static void g_function(struct janas_buf *out, const struct janas_json *f)
{
    const struct janas_json *nm = janas_json_get(f, "name");
    const struct janas_json *ds = janas_json_get(f, "description");
    const struct janas_json *pa = janas_json_get(f, "parameters");
    janas_buf_puts(out, "declaration:");
    if (nm)
        janas_buf_put(out, nm->s, nm->n);
    janas_buf_puts(out, "{description:" GQ);
    if (ds && ds->type == JANAS_JSON_STRING)
        janas_buf_put(out, ds->s, ds->n);
    janas_buf_puts(out, GQ);
    if (pa && pa->type == JANAS_JSON_OBJECT && pa->n) {
        janas_buf_puts(out, ",parameters:{");
        const struct janas_json *pp = janas_json_get(pa, "properties");
        if (pp && pp->type == JANAS_JSON_OBJECT && pp->n) {
            janas_buf_puts(out, "properties:{");
            g_params(out, pp, 0);
            janas_buf_puts(out, "},");
        }
        const struct janas_json *rq = janas_json_get(pa, "required");
        if (rq && rq->n) {
            g_required(out, rq);
            janas_buf_puts(out, ",");
        }
        if (janas_json_get(pa, "type")) {
            janas_buf_puts(out, "type:" GQ);
            g_upper(out, janas_json_get(pa, "type"));
            janas_buf_puts(out, GQ "}");
        }
    }
    janas_buf_puts(out, "}");
}

void janas_tools_gemma_response(struct janas_buf *out, const char *name,
                                size_t name_n, const char *text, size_t n)
{
    /* a message's content is text, JSON or not: the template quotes it */
    janas_buf_puts(out, "response:");
    janas_buf_put(out, name, name_n);
    janas_buf_puts(out, "{value:" GQ);
    janas_buf_put(out, text, n);
    janas_buf_puts(out, GQ "}<tool_response|>");
}

/* Gemma's notation to JSON, one value from *p; 0, or -1 when it is not. */
static int g_to_json(const char **p, const char *end, struct janas_buf *out)
{
    const char *s = *p;
    size_t ql = strlen(GQ);
    while (s < end && (*s == ' ' || *s == '\n'))
        s++;
    if (s >= end)
        return -1;
    if ((size_t)(end - s) >= ql && !memcmp(s, GQ, ql)) {
        const char *b = s + ql, *e = b;
        while (e + ql <= end && memcmp(e, GQ, ql))
            e++;
        if (e + ql > end)
            return -1;
        janas_json_write_str(out, b, (size_t)(e - b));
        *p = e + ql;
        return 0;
    }
    if (*s == '{' || *s == '[') {
        char close = *s == '{' ? '}' : ']';
        int obj = *s == '{';
        janas_buf_put(out, s, 1);
        s++;
        for (int first = 1;; first = 0) {
            while (s < end && (*s == ' ' || *s == '\n'))
                s++;
            if (s < end && *s == close)
                break;
            if (!first) {
                if (s >= end || *s != ',')
                    return -1;
                janas_buf_puts(out, ",");
                s++;
            }
            if (obj) { /* a key, bare or quoted, then ':' */
                while (s < end && *s == ' ')
                    s++;
                const char *k = s;
                if ((size_t)(end - s) >= ql && !memcmp(s, GQ, ql)) {
                    k = s += ql;
                    while (s + ql <= end && memcmp(s, GQ, ql))
                        s++;
                    janas_json_write_str(out, k, (size_t)(s - k));
                    s += ql;
                } else {
                    while (s < end && *s != ':')
                        s++;
                    janas_json_write_str(out, k, (size_t)(s - k));
                }
                if (s >= end || *s != ':')
                    return -1;
                janas_buf_puts(out, ":");
                s++;
            }
            if (g_to_json(&s, end, out))
                return -1;
        }
        janas_buf_put(out, &close, 1);
        *p = s + 1;
        return 0;
    }
    const char *b = s;
    while (s < end && *s != ',' && *s != '}' && *s != ']')
        s++;
    size_t n = (size_t)(s - b);
    while (n && b[n - 1] == ' ')
        n--;
    if (!n)
        return -1;
    int bare = (n == 4 && (!memcmp(b, "true", 4) || !memcmp(b, "null", 4))) ||
               (n == 5 && !memcmp(b, "false", 5));
    char *x = NULL;
    char tmp[64];
    if (!bare && n < sizeof(tmp)) {
        memcpy(tmp, b, n);
        tmp[n] = 0;
        strtod(tmp, &x);
    }
    if (bare || (x && *x == 0))
        janas_buf_put(out, b, n);
    else /* a bare word: taken as a string */
        janas_json_write_str(out, b, n);
    *p = s;
    return 0;
}

/* One function, as Qwen3-Coder's template describes it. */
static void coder_function(struct janas_buf *out, const struct janas_json *f)
{
    const struct janas_json *name = janas_json_get(f, "name");
    const struct janas_json *desc = janas_json_get(f, "description");
    const struct janas_json *params = janas_json_get(f, "parameters");
    janas_buf_puts(out, "\n<function>\n<name>");
    janas_buf_put(out, name->s, name->n);
    janas_buf_puts(out, "</name>");
    if (desc && desc->type == JANAS_JSON_STRING) {
        size_t n = desc->n;
        const char *d = trim(desc->s, &n);
        janas_buf_puts(out, "\n<description>");
        janas_buf_put(out, d, n);
        janas_buf_puts(out, "</description>");
    }
    janas_buf_puts(out, "\n<parameters>");
    const struct janas_json *props = janas_json_get(params, "properties");
    if (props && props->type == JANAS_JSON_OBJECT)
        for (const struct janas_json *p = props->child; p; p = p->next) {
            janas_buf_puts(out, "\n<parameter>\n<name>");
            janas_buf_put(out, p->key, p->key_n);
            janas_buf_puts(out, "</name>");
            const struct janas_json *ty = janas_json_get(p, "type");
            if (ty) {
                janas_buf_puts(out, "\n<type>");
                put_value(out, ty);
                janas_buf_puts(out, "</type>");
            }
            const struct janas_json *pd = janas_json_get(p, "description");
            if (pd && pd->type == JANAS_JSON_STRING) {
                size_t n = pd->n;
                const char *d = trim(pd->s, &n);
                janas_buf_puts(out, "\n<description>");
                janas_buf_put(out, d, n);
                janas_buf_puts(out, "</description>");
            }
            extra_keys(
                out, p,
                (const char *const[]){"name", "type", "description", NULL});
            janas_buf_puts(out, "\n</parameter>");
        }
    extra_keys(out, params, (const char *const[]){"type", "properties", NULL});
    janas_buf_puts(out, "\n</parameters>");
    extra_keys(out, f,
               (const char *const[]){"type", "name", "description",
                                     "parameters", NULL});
    janas_buf_puts(out, "\n</function>");
}

void janas_tools_system(struct janas_buf *out, enum janas_tool_dialect d,
                        const struct janas_toolset *t, const char *sys,
                        size_t sys_n)
{
    switch (d) {
    case JANAS_TOOLS_JSON:
        if (sys) {
            janas_buf_put(out, sys, sys_n);
            janas_buf_puts(out, "\n\n");
        }
        janas_buf_puts(out, "# Tools\n\nYou may call one or more functions to "
                            "assist with the user query.\n\nYou are provided "
                            "with function signatures within <tools></tools> "
                            "XML tags:\n<tools>");
        for (uint32_t i = 0; i < t->n; i++) {
            janas_buf_puts(out, "\n");
            janas_json_write(out, t->item[i]);
        }
        janas_buf_puts(out, "\n</tools>\n\nFor each function call, return a "
                            "json object with function name and arguments "
                            "within <tool_call></tool_call> XML tags:\n"
                            "<tool_call>\n{\"name\": <function-name>, "
                            "\"arguments\": <args-json-object>}\n</tool_call>");
        break;
    case JANAS_TOOLS_XML: {
        janas_buf_puts(out, "# Tools\n\nYou have access to the following "
                            "functions:\n\n<tools>");
        for (uint32_t i = 0; i < t->n; i++) {
            janas_buf_puts(out, "\n");
            janas_json_write(out, t->item[i]);
        }
        janas_buf_puts(out, "\n</tools>");
        janas_buf_puts(out, XML_FORMAT);
        size_t n = sys_n;
        const char *s = sys ? trim(sys, &n) : NULL;
        if (s && n) {
            janas_buf_puts(out, "\n\n");
            janas_buf_put(out, s, n);
        }
        break;
    }
    case JANAS_TOOLS_GEMMA: {
        size_t n = sys_n;
        const char *s = sys ? trim(sys, &n) : NULL;
        if (s && n)
            janas_buf_put(out, s, n);
        for (uint32_t i = 0; i < t->n; i++) {
            janas_buf_puts(out, "<|tool>");
            g_function(out, t->fn[i]);
            janas_buf_puts(out, "<tool|>");
        }
        break;
    }
    case JANAS_TOOLS_XML_CODER:
        if (sys)
            janas_buf_put(out, sys, sys_n);
        else
            janas_buf_puts(out, "You are Qwen, a helpful AI assistant that "
                                "can interact with a computer to solve "
                                "tasks.");
        janas_buf_puts(out, "\n\n# Tools\n\nYou have access to the following "
                            "functions:\n\n<tools>");
        for (uint32_t i = 0; i < t->n; i++)
            coder_function(out, t->fn[i]);
        janas_buf_puts(out, "\n</tools>");
        janas_buf_puts(out, XML_FORMAT);
        break;
    case JANAS_TOOLS_NONE:
        if (sys)
            janas_buf_put(out, sys, sys_n);
        break;
    }
}

static uint32_t g_value_rule(struct janas_gbuild *b, int32_t quote);

/* The rule of a Gemma call: call:name{members} then the closing token. */
uint32_t janas_tools_rule_gemma(struct janas_gbuild *b,
                                const struct janas_toolset *t, int only,
                                int32_t close, int32_t quote)
{
    uint32_t fns = janas_gb_rule(b);
    for (uint32_t i = 0; i < t->n; i++) {
        if (only >= 0 && (uint32_t)only != i)
            continue;
        const struct janas_json *nm = janas_json_get(t->fn[i], "name");
        janas_gb_lit(b, janas_gb_alt(b, fns), nm->s, nm->n);
    }
    uint32_t val = g_value_rule(b, quote);
    uint32_t r = janas_gb_rule(b);
    uint32_t a = janas_gb_alt(b, r);
    janas_gb_lit(b, a, "call:", 5);
    janas_gb_ref(b, a, fns);
    janas_gb_ref(b, a, val);
    janas_gb_token(b, a, close);
    return r;
}

/* ---- calls of earlier turns ---- */

/* The arguments of an XML call, one <parameter=...> block per key. */
static void xml_params(struct janas_buf *out, const char *args, size_t n)
{
    struct janas_json_doc *doc = janas_json_parse(args, n, NULL, 0);
    const struct janas_json *o = janas_json_root(doc);
    if (o && o->type == JANAS_JSON_OBJECT)
        for (const struct janas_json *k = o->child; k; k = k->next) {
            janas_buf_puts(out, "<parameter=");
            janas_buf_put(out, k->key, k->key_n);
            janas_buf_puts(out, ">\n");
            put_value(out, k);
            janas_buf_puts(out, "\n</parameter>\n");
        }
    janas_json_free(doc);
}

void janas_tools_assistant(struct janas_buf *out, enum janas_tool_dialect d,
                           const char *content, size_t n,
                           const struct janas_tool_call *calls, size_t n_calls)
{
    size_t tn = n;
    const char *tc = content ? trim(content, &tn) : "";
    if (d == JANAS_TOOLS_GEMMA) { /* the calls first, then the text */
        for (size_t i = 0; i < n_calls; i++) {
            janas_buf_puts(out, "<|tool_call>call:");
            janas_buf_put(out, calls[i].name, calls[i].name_n);
            struct janas_json_doc *doc =
                janas_json_parse(calls[i].args, calls[i].args_n, NULL, 0);
            const struct janas_json *r = janas_json_root(doc);
            const struct janas_json *k[256];
            size_t m =
                r && r->type == JANAS_JSON_OBJECT ? sorted(r, k, 256) : 0;
            janas_buf_puts(out, "{");
            for (size_t x = 0; x < m; x++) {
                if (x)
                    janas_buf_puts(out, ",");
                janas_buf_put(out, k[x]->key, k[x]->key_n);
                janas_buf_puts(out, ":");
                g_arg(out, k[x], 0);
            }
            janas_buf_puts(out, "}<tool_call|>");
            janas_json_free(doc);
        }
        if (content)
            janas_buf_put(out, content, n);
        return;
    }
    if (d == JANAS_TOOLS_XML_CODER) {
        if (tn) {
            janas_buf_puts(out, "\n");
            janas_buf_put(out, tc, tn);
            janas_buf_puts(out, "\n");
        }
    } else if (content) {
        janas_buf_put(out, content, n);
    }
    for (size_t i = 0; i < n_calls; i++) {
        const struct janas_tool_call *c = &calls[i];
        if (d == JANAS_TOOLS_JSON) {
            if (i > 0 || n > 0)
                janas_buf_puts(out, "\n");
            janas_buf_puts(out, "<tool_call>\n{\"name\": \"");
            janas_buf_put(out, c->name, c->name_n);
            janas_buf_puts(out, "\", \"arguments\": ");
            janas_buf_put(out, c->args, c->args_n);
            janas_buf_puts(out, "}\n</tool_call>");
            continue;
        }
        if (d == JANAS_TOOLS_XML && i == 0)
            janas_buf_puts(out, tn ? "\n\n<tool_call>\n<function="
                                   : "<tool_call>\n<function=");
        else
            janas_buf_puts(out, "\n<tool_call>\n<function=");
        janas_buf_put(out, c->name, c->name_n);
        janas_buf_puts(out, ">\n");
        xml_params(out, c->args, c->args_n);
        janas_buf_puts(out, "</function>\n</tool_call>");
    }
}

/* ---- calls the model writes ---- */

/* The schema of parameter key of function f, or NULL. */
static const struct janas_json *param_schema(const struct janas_json *f,
                                             const char *key, size_t n)
{
    const struct janas_json *props =
        janas_json_get(janas_json_get(f, "parameters"), "properties");
    for (const struct janas_json *p = props ? props->child : NULL; p;
         p = p->next)
        if (p->key_n == n && memcmp(p->key, key, n) == 0)
            return p;
    return NULL;
}

/* Whether an XML parameter's value is written as text (a string, or of no
   declared type) rather than as JSON. */
static int is_text(const struct janas_json *schema)
{
    const struct janas_json *ty = janas_json_get(schema, "type");
    if (!ty)
        return !janas_json_get(schema, "properties") &&
               !janas_json_get(schema, "items") &&
               !janas_json_get(schema, "enum");
    if (ty->type == JANAS_JSON_STRING)
        return janas_json_is(ty, "string");
    if (ty->type == JANAS_JSON_ARRAY)
        for (const struct janas_json *t = ty->child; t; t = t->next)
            if (janas_json_is(t, "string"))
                return 1;
    return 0;
}

static const char *find(const char *s, size_t n, const char *what)
{
    size_t k = strlen(what);
    for (size_t i = 0; i + k <= n; i++)
        if (memcmp(s + i, what, k) == 0)
            return s + i;
    return NULL;
}

static int parse_xml(const struct janas_toolset *t, const char *s, size_t n,
                     struct janas_buf *name, struct janas_buf *args)
{
    const char *end = s + n;
    const char *f = find(s, n, "<function=");
    if (!f)
        return -1;
    f += 10;
    const char *gt = memchr(f, '>', (size_t)(end - f));
    if (!gt)
        return -1;
    janas_buf_put(name, f, (size_t)(gt - f));
    int fi = janas_toolset_find(t, f, (size_t)(gt - f));
    const struct janas_json *fn = fi >= 0 ? t->fn[fi] : NULL;
    janas_buf_puts(args, "{");
    int first = 1;
    const char *p = gt + 1;
    for (;;) {
        const char *q = find(p, (size_t)(end - p), "<parameter=");
        if (!q)
            break;
        q += 11;
        const char *kg = memchr(q, '>', (size_t)(end - q));
        if (!kg)
            return -1;
        const char *v = kg + 1;
        const char *ve = find(v, (size_t)(end - v), "</parameter>");
        if (!ve)
            return -1;
        p = ve + 12;
        /* the value between the newlines the template puts around it */
        if (v < ve && *v == '\n')
            v++;
        if (ve > v && ve[-1] == '\n')
            ve--;
        if (!first)
            janas_buf_puts(args, ", ");
        first = 0;
        janas_json_write_str(args, q, (size_t)(kg - q));
        janas_buf_puts(args, ": ");
        const struct janas_json *ps =
            fn ? param_schema(fn, q, (size_t)(kg - q)) : NULL;
        struct janas_json_doc *d =
            ps && is_text(ps) ? NULL
                              : janas_json_parse(v, (size_t)(ve - v), NULL, 0);
        if (d)
            janas_json_write(args, janas_json_root(d));
        else
            janas_json_write_str(args, v, (size_t)(ve - v));
        janas_json_free(d);
    }
    janas_buf_puts(args, "}");
    return 0;
}

int janas_tools_parse(enum janas_tool_dialect d, const struct janas_toolset *t,
                      const char *body, size_t n, struct janas_buf *name,
                      struct janas_buf *args)
{
    if (d == JANAS_TOOLS_GEMMA) { /* call:name{...} */
        const char *s = body, *end = body + n;
        while (s < end && (*s == ' ' || *s == '\n'))
            s++;
        if ((size_t)(end - s) < 5 || memcmp(s, "call:", 5))
            return -1;
        s += 5;
        const char *nm = s;
        while (s < end && *s != '{')
            s++;
        if (s == end || janas_toolset_find(t, nm, (size_t)(s - nm)) < 0)
            return -1;
        janas_buf_put(name, nm, (size_t)(s - nm));
        return g_to_json(&s, end, args);
    }
    if (d != JANAS_TOOLS_JSON)
        return parse_xml(t, body, n, name, args);
    struct janas_json_doc *doc = janas_json_parse(body, n, NULL, 0);
    const struct janas_json *root = janas_json_root(doc);
    const struct janas_json *nm = janas_json_get(root, "name");
    const struct janas_json *ar = janas_json_get(root, "arguments");
    int rc = -1;
    if (nm && nm->type == JANAS_JSON_STRING) {
        janas_buf_put(name, nm->s, nm->n);
        if (ar && ar->type == JANAS_JSON_STRING) /* arguments as a string */
            janas_buf_put(args, ar->s, ar->n);
        else if (ar)
            janas_json_write(args, ar);
        else
            janas_buf_puts(args, "{}");
        rc = 0;
    }
    janas_json_free(doc);
    return rc;
}

/* ---- the grammar of a call ---- */

/* The arguments of function f, as a JSON object. */
static uint32_t args_rule(struct janas_jsg *j, const struct janas_json *f)
{
    const struct janas_json *p = janas_json_get(f, "parameters");
    if (!p || !janas_json_get(p, "properties"))
        return janas_jsg_object(j);
    return janas_jsg_schema(j, p, p);
}

/*
 * The parameters of an XML call, in the schema's order, the required ones
 * always there: R[i][x] as for a JSON object (schema.c), with no commas.
 */
static uint32_t xml_params_rule(struct janas_gbuild *b, struct janas_jsg *j,
                                const struct janas_json *f)
{
    const struct janas_json *params = janas_json_get(f, "parameters");
    const struct janas_json *props = janas_json_get(params, "properties");
    const struct janas_json *req = janas_json_get(params, "required");
    uint32_t ws = janas_jsg_ws(j);
    if (!props || props->type != JANAS_JSON_OBJECT || props->n == 0)
        return janas_gb_empty(b);
    size_t n = props->n;
    uint32_t *r = malloc((n + 1) * sizeof(uint32_t));
    if (!r)
        return janas_gb_rule(b);
    for (size_t i = 0; i <= n; i++)
        r[i] = janas_gb_rule(b);
    janas_gb_alt(b, r[n]);
    const char *close = "\n</parameter>";
    uint32_t text = janas_gb_until(b, close, strlen(close));
    size_t i = 0;
    for (const struct janas_json *p = props->child; p; p = p->next, i++) {
        int must = 0;
        for (const struct janas_json *q = req ? req->child : NULL; q && !must;
             q = q->next)
            must = q->type == JANAS_JSON_STRING && q->n == p->key_n &&
                   memcmp(q->s, p->key, q->n) == 0;
        if (!must)
            janas_gb_ref(b, janas_gb_alt(b, r[i]), r[i + 1]);
        uint32_t a = janas_gb_alt(b, r[i]);
        janas_gb_lit(b, a, "<parameter=", 11);
        janas_gb_lit(b, a, p->key, p->key_n);
        janas_gb_lit(b, a, ">\n", 2);
        if (is_text(p)) {
            janas_gb_ref(b, a, text);
        } else {
            janas_gb_ref(b, a, janas_jsg_schema(j, p, params));
            janas_gb_ref(b, a, ws);
            janas_gb_lit(b, a, "</parameter>", 12);
        }
        janas_gb_ref(b, a, ws);
        janas_gb_ref(b, a, r[i + 1]);
    }
    uint32_t first = r[0];
    free(r);
    return first;
}

/*
 * Gemma's notation, any value: strings between quote tokens, numbers and
 * words, objects of bare keys, lists. Not held to the function's schema -
 * the name is - but always a value the parser reads.
 */
static uint32_t g_value_rule(struct janas_gbuild *b, int32_t quote)
{
    uint8_t any[32], key[32], word[32];
    memset(any, 0xff, sizeof(any));
    memset(key, 0, sizeof(key));
    memset(word, 0, sizeof(word));
    for (int c = 0; c < 256; c++) {
        int k = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_';
        int w = k || c == '-' || c == '+' || c == '.';
        if (k)
            key[c / 8] |= (uint8_t)(1 << (c % 8));
        if (w)
            word[c / 8] |= (uint8_t)(1 << (c % 8));
    }
    uint32_t text = janas_gb_rule(b); /* any bytes */
    janas_gb_alt(b, text);
    uint32_t a = janas_gb_alt(b, text);
    janas_gb_set(b, a, any);
    janas_gb_ref(b, a, text);
    uint32_t keyr = janas_gb_rule(b), wordr = janas_gb_rule(b);
    a = janas_gb_alt(b, keyr);
    janas_gb_set(b, a, key);
    a = janas_gb_alt(b, keyr);
    janas_gb_set(b, a, key);
    janas_gb_ref(b, a, keyr);
    a = janas_gb_alt(b, wordr);
    janas_gb_set(b, a, word);
    a = janas_gb_alt(b, wordr);
    janas_gb_set(b, a, word);
    janas_gb_ref(b, a, wordr);
    uint32_t val = janas_gb_rule(b), members = janas_gb_rule(b),
             elems = janas_gb_rule(b);
    a = janas_gb_alt(b, val); /* a string */
    janas_gb_token(b, a, quote);
    janas_gb_ref(b, a, text);
    janas_gb_token(b, a, quote);
    a = janas_gb_alt(b, val); /* a number or a word */
    janas_gb_ref(b, a, wordr);
    a = janas_gb_alt(b, val); /* an object */
    janas_gb_byte(b, a, '{');
    janas_gb_byte(b, a, '}');
    a = janas_gb_alt(b, val);
    janas_gb_byte(b, a, '{');
    janas_gb_ref(b, a, members);
    janas_gb_byte(b, a, '}');
    a = janas_gb_alt(b, val); /* a list */
    janas_gb_byte(b, a, '[');
    janas_gb_byte(b, a, ']');
    a = janas_gb_alt(b, val);
    janas_gb_byte(b, a, '[');
    janas_gb_ref(b, a, elems);
    janas_gb_byte(b, a, ']');
    for (int more = 0; more < 2; more++) {
        a = janas_gb_alt(b, members);
        janas_gb_ref(b, a, keyr);
        janas_gb_byte(b, a, ':');
        janas_gb_ref(b, a, val);
        if (more) {
            janas_gb_byte(b, a, ',');
            janas_gb_ref(b, a, members);
        }
        a = janas_gb_alt(b, elems);
        janas_gb_ref(b, a, val);
        if (more) {
            janas_gb_byte(b, a, ',');
            janas_gb_ref(b, a, elems);
        }
    }
    return val;
}

uint32_t janas_tools_rule(struct janas_gbuild *b, struct janas_jsg *j,
                          enum janas_tool_dialect d,
                          const struct janas_toolset *t, int only,
                          int32_t close)
{
    uint32_t ws = janas_jsg_ws(j);
    uint32_t fns = janas_gb_rule(b); /* one alternative per function */
    for (uint32_t i = 0; i < t->n; i++) {
        if (only >= 0 && (uint32_t)only != i)
            continue;
        const struct janas_json *nm = janas_json_get(t->fn[i], "name");
        uint32_t a = janas_gb_alt(b, fns);
        if (d == JANAS_TOOLS_JSON) {
            janas_gb_byte(b, a, '"');
            janas_gb_lit(b, a, nm->s, nm->n);
            janas_gb_byte(b, a, '"');
            janas_gb_ref(b, a, ws);
            janas_gb_byte(b, a, ',');
            janas_gb_ref(b, a, ws);
            janas_gb_lit(b, a, "\"arguments\"", 11);
            janas_gb_ref(b, a, ws);
            janas_gb_byte(b, a, ':');
            janas_gb_ref(b, a, ws);
            janas_gb_ref(b, a, args_rule(j, t->fn[i]));
        } else {
            janas_gb_lit(b, a, nm->s, nm->n);
            janas_gb_lit(b, a, ">\n", 2);
            janas_gb_ref(b, a, xml_params_rule(b, j, t->fn[i]));
        }
    }
    uint32_t r = janas_gb_rule(b);
    uint32_t a = janas_gb_alt(b, r);
    janas_gb_ref(b, a, ws);
    if (d == JANAS_TOOLS_JSON) {
        janas_gb_byte(b, a, '{');
        janas_gb_ref(b, a, ws);
        janas_gb_lit(b, a, "\"name\"", 6);
        janas_gb_ref(b, a, ws);
        janas_gb_byte(b, a, ':');
        janas_gb_ref(b, a, ws);
        janas_gb_ref(b, a, fns);
        janas_gb_ref(b, a, ws);
        janas_gb_byte(b, a, '}');
    } else {
        janas_gb_lit(b, a, "<function=", 10);
        janas_gb_ref(b, a, fns);
        janas_gb_lit(b, a, "</function>", 11);
    }
    janas_gb_ref(b, a, ws);
    janas_gb_token(b, a, close);
    return r;
}
