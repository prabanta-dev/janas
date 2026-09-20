/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_grammar.c - the library's JSON reader and writer (src/llm/json.c),
 * grammars (grammar.c, gmatch.c) and JSON Schema as a grammar (schema.c).
 * Tool definitions and schemas come from whoever sends a request, so the
 * reader is also fed mangled documents from a fixed seed: it must neither
 * crash nor leak (run it under asan to know the second).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/grammar.h"
#include "llm/json.h"
#include "llm/schema.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* ---- JSON ---- */

/* Parses s and writes it back; the text written, or NULL. */
static char *roundtrip(const char *s)
{
    char err[128];
    struct janas_json_doc *d = janas_json_parse(s, strlen(s), err, sizeof(err));
    if (!d)
        return NULL;
    struct janas_buf b = {0};
    janas_json_write(&b, janas_json_root(d));
    janas_json_free(d);
    return b.p;
}

static void test_json(void)
{
    static const struct {
        const char *in, *out;
    } ok[] = {
        {"{\"a\":1,\"b\":[true,false,null],\"c\":\"x\"}",
         "{\"a\": 1, \"b\": [true, false, null], \"c\": \"x\"}"},
        {" [ ] ", "[]"},
        {"{}", "{}"},
        {"-0.5e+10", "-0.5e+10"},
        {"\"caf\\u00e9 \\ud83d\\ude00 \\n\\t\\\"\\\\\\/\"",
         "\"caf\xc3\xa9 \xf0\x9f\x98\x80 \\n\\t\\\"\\\\/\""},
        {"\"\\u0001\"", "\"\\u0001\""},
        {"\"\xc3\xa8\"", "\"\xc3\xa8\""},
    };
    for (size_t i = 0; i < sizeof(ok) / sizeof(ok[0]); i++) {
        char *w = roundtrip(ok[i].in);
        CHECK(w && strcmp(w, ok[i].out) == 0, "json %zu: got %s", i,
              w ? w : "(error)");
        free(w);
    }
    static const char *const bad[] = {
        "",      "{",    "[1,]",      "{\"a\"}",   "{\"a\":}",
        "01",    "1.",   "-",         "\"\\x\"",   "\"a",
        "[1] 2", "tru",  "{\"a\" 1}", "\"\\u12\"", "\"\x01\"",
        "{,}",   "[,1]", "1e",        "nul",       "{\"a\":1,}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        char *w = roundtrip(bad[i]);
        CHECK(!w, "json bad %zu (%s) was accepted as %s", i, bad[i], w);
        free(w);
    }
    /* nesting: 128 levels are read, 129 are refused */
    char deep[600];
    for (int depth = 128; depth <= 129; depth++) {
        int n = 0;
        for (int i = 0; i < depth; i++)
            deep[n++] = '[';
        for (int i = 0; i < depth; i++)
            deep[n++] = ']';
        deep[n] = 0;
        char *w = roundtrip(deep);
        CHECK((w != NULL) == (depth == 128), "json nesting %d", depth);
        free(w);
    }
    char err[128];
    const char *doc = "{\"k\":\"v\",\"n\":2.5,\"e\":\"\\u0000x\"}";
    struct janas_json_doc *d =
        janas_json_parse(doc, strlen(doc), err, sizeof(err));
    CHECK(d != NULL, "json get: %s", err);
    if (d) {
        const struct janas_json *r = janas_json_root(d);
        CHECK(janas_json_is(janas_json_get(r, "k"), "v"), "json get k");
        CHECK(janas_json_num(janas_json_get(r, "n"), 0) == 2.5, "json num");
        const struct janas_json *e = janas_json_get(r, "e");
        CHECK(e && e->n == 2 && e->s[0] == 0 && e->s[1] == 'x',
              "json embedded NUL");
        CHECK(!janas_json_get(r, "missing"), "json get missing");
        janas_json_free(d);
    }
}

static uint64_t rng = 0x5eed;
static uint32_t next_rand(void)
{
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(rng >> 33);
}

/* Mangled documents: bytes flipped, cut, repeated. */
static void test_json_mangled(void)
{
    static const char *const seeds[] = {
        ("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"},"
         "\"b\":{\"type\":\"array\",\"items\":{\"$ref\":\"#\"}}},"
         "\"required\":[\"a\"]}"),
        ("[{\"type\":\"function\",\"function\":{\"name\":\"f\","
         "\"parameters\":{\"type\":\"object\"}}}]"),
        "\"\\ud83d\\ude00 \\u00e9\"",
    };
    char buf[512];
    int parsed = 0;
    for (int it = 0; it < 20000; it++) {
        const char *s = seeds[it % 3];
        size_t n = strlen(s);
        memcpy(buf, s, n);
        int edits = 1 + (int)(next_rand() % 4);
        for (int e = 0; e < edits; e++) {
            uint32_t at = next_rand() % (uint32_t)n;
            switch (next_rand() % 3) {
            case 0:
                buf[at] = (char)next_rand();
                break;
            case 1:
                n = at;
                break;
            default:
                if (n + 8 < sizeof(buf)) {
                    memmove(buf + at + 8, buf + at, n - at);
                    n += 8;
                }
            }
            if (n == 0)
                break;
        }
        char err[64];
        struct janas_json_doc *d = janas_json_parse(buf, n, err, sizeof(err));
        if (d) {
            parsed++;
            /* whatever was read compiles as a schema without trouble */
            struct janas_gbuild *b = janas_gbuild_new();
            struct janas_jsg *j = janas_jsg_new(b);
            uint32_t root = janas_jsg_schema(j, janas_json_root(d), NULL);
            struct janas_grammar *g = janas_grammar_make(b);
            struct janas_gmatch *m = g ? janas_gmatch_new(g, root) : NULL;
            if (m)
                janas_gmatch_try(m, "{\"a\": \"x\"}", 10, -1);
            janas_gmatch_free(m);
            janas_grammar_free(g);
            janas_jsg_free(j);
            janas_gbuild_free(b);
            janas_json_free(d);
        }
    }
    CHECK(parsed > 0 && parsed < 20000, "json mangled: %d parsed", parsed);
}

/* ---- grammars ---- */

struct gram {
    struct janas_gbuild *b;
    struct janas_jsg *j;
    struct janas_grammar *g;
    struct janas_json_doc *d;
    uint32_t root;
};

/* A grammar for "ws value ws" of this schema (NULL: any JSON value). */
static int make_schema(struct gram *x, const char *schema)
{
    memset(x, 0, sizeof(*x));
    x->b = janas_gbuild_new();
    x->j = janas_jsg_new(x->b);
    uint32_t val;
    if (schema) {
        char err[128];
        x->d = janas_json_parse(schema, strlen(schema), err, sizeof(err));
        if (!x->d) {
            printf("schema: %s\n", err);
            return -1;
        }
        val = janas_jsg_schema(x->j, janas_json_root(x->d), NULL);
    } else {
        val = janas_jsg_any(x->j);
    }
    uint32_t ws = janas_jsg_ws(x->j);
    x->root = janas_gb_rule(x->b);
    uint32_t a = janas_gb_alt(x->b, x->root);
    janas_gb_ref(x->b, a, ws);
    janas_gb_ref(x->b, a, val);
    janas_gb_ref(x->b, a, ws);
    x->g = janas_grammar_make(x->b);
    return x->g ? 0 : -1;
}

static void gram_free(struct gram *x)
{
    janas_grammar_free(x->g);
    janas_jsg_free(x->j);
    janas_gbuild_free(x->b);
    janas_json_free(x->d);
}

/* 1 when the whole text matches, fed in pieces of `step` bytes. */
static int matches(const struct gram *x, const char *text, size_t step)
{
    struct janas_gmatch *m = janas_gmatch_new(x->g, x->root);
    if (!m)
        return -1;
    size_t n = strlen(text);
    int ok = 1;
    for (size_t i = 0; i < n && ok; i += step) {
        size_t k = n - i < step ? n - i : step;
        /* trying first must not change what feeding finds */
        int t = janas_gmatch_try(m, text + i, k, -1);
        int f = janas_gmatch_feed(m, text + i, k, -1) == 0;
        if (t != f)
            ok = -1;
        else
            ok = f;
    }
    int acc = ok == 1 && janas_gmatch_accepting(m);
    janas_gmatch_free(m);
    return ok < 0 ? -1 : acc;
}

static void expect(const char *name, const char *schema,
                   const char *const *good, const char *const *bad)
{
    struct gram x;
    if (make_schema(&x, schema) != 0) {
        CHECK(0, "%s: grammar not made", name);
        gram_free(&x);
        return;
    }
    for (int i = 0; good[i]; i++)
        for (size_t step = 1; step <= 7; step += 3)
            CHECK(matches(&x, good[i], step) == 1, "%s: refused (step %zu): %s",
                  name, step, good[i]);
    for (int i = 0; bad[i]; i++)
        CHECK(matches(&x, bad[i], 1) == 0, "%s: accepted: %s", name, bad[i]);
    gram_free(&x);
}

static void test_schemas(void)
{
    expect("any", NULL,
           (const char *const[]){"{}", " [1, -2.5e3, \"x\", null, true] ",
                                 "{\"a\": {\"b\": [[], {}]}}",
                                 "\"caf\xc3\xa9\"", "0", "-0.5",
                                 "\"\\u00e9\\n\"", NULL},
           (const char *const[]){"{", "[1,]", "01", "\"a", "{a:1}", "\"\x01\"",
                                 "1.", "tru", NULL});
    expect("object",
           "{\"type\":\"object\",\"properties\":{"
           "\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"},"
           "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},"
           "\"maxItems\":2}},\"required\":[\"name\",\"age\"],"
           "\"additionalProperties\":false}",
           (const char *const[]){
               "{\"name\": \"Ann\", \"age\": 30}",
               "{\"name\":\"A\",\"age\":-1,\"tags\":[\"x\",\"y\"]}",
               "{\n  \"name\": \"A\",\n  \"age\": 0,\n  \"tags\": []\n}", NULL},
           (const char *const[]){
               "{\"age\": 3, \"name\": \"x\"}", "{\"name\": \"A\"}",
               "{\"name\":\"A\",\"age\":1,\"tags\":[\"a\",\"b\",\"c\"]}",
               "{\"name\":\"A\",\"age\":1.5}",
               "{\"name\":\"A\",\"age\":1,"
               "\"x\":1}",
               NULL});
    expect("enum and anyOf",
           "{\"anyOf\":[{\"enum\":[\"red\",\"green\",3]},"
           "{\"type\":\"null\"},{\"type\":\"boolean\"}]}",
           (const char *const[]){"\"red\"", "3", "null", "false", NULL},
           (const char *const[]){"\"blue\"", "4", "\"re\"", NULL});
    expect("recursion",
           "{\"$defs\":{\"node\":{\"type\":\"object\",\"properties\":{"
           "\"v\":{\"type\":\"integer\"},\"kids\":{\"type\":\"array\","
           "\"items\":{\"$ref\":\"#/$defs/node\"}}},\"required\":[\"v\"]}},"
           "\"$ref\":\"#/$defs/node\"}",
           (const char *const[]){
               "{\"v\": 1, \"kids\": [{\"v\": 2}, {\"v\": 3, \"kids\": []}]}",
               NULL},
           (const char *const[]){"{\"v\": 1, \"kids\": [{\"w\": 2}]}", NULL});
    expect("string bounds",
           "{\"type\":\"string\",\"minLength\":2,\"maxLength\":3}",
           (const char *const[]){"\"ab\"", "\"abc\"", "\"a\\n\"", NULL},
           (const char *const[]){"\"a\"", "\"abcd\"", NULL});
    expect("nullable and type list",
           "{\"type\":[\"number\",\"string\"],\"nullable\":true}",
           (const char *const[]){"1.5", "\"x\"", "null", NULL},
           (const char *const[]){"true", "{}", NULL});
    expect(
        "const", "{\"const\":{\"a\":[1,\"b\"]}}",
        (const char *const[]){"{\"a\": [1, \"b\"]}", "{\"a\":[1,\"b\"]}", NULL},
        (const char *const[]){"{\"a\":[1]}", "{\"a\":[1,\"c\"]}", NULL});
    expect("false", "false", (const char *const[]){NULL},
           (const char *const[]){"1", "{}", NULL});
}

/* Text up to a closing tag, and special tokens as symbols. */
static void test_until_and_tokens(void)
{
    struct janas_gbuild *b = janas_gbuild_new();
    const char *end = "\n</parameter>";
    uint32_t until = janas_gb_until(b, end, strlen(end));
    uint32_t root = janas_gb_rule(b);
    uint32_t a = janas_gb_alt(b, root);
    janas_gb_token(b, a, 7);
    janas_gb_lit(b, a, "<parameter=x>\n", 14);
    janas_gb_ref(b, a, until);
    janas_gb_token(b, a, 9);
    struct janas_grammar *g = janas_grammar_make(b);
    struct janas_gmatch *m = g ? janas_gmatch_new(g, root) : NULL;
    CHECK(m != NULL, "until: grammar");
    if (m) {
        CHECK(!janas_gmatch_try(m, "<", 1, -1), "until: bytes before token");
        CHECK(janas_gmatch_try(m, NULL, 0, 7), "until: token 7");
        CHECK(!janas_gmatch_try(m, NULL, 0, 8), "until: token 8");
        janas_gmatch_feed(m, NULL, 0, 7);
        const char *v = "<parameter=x>\nline one\n</para\n</paramete x"
                        "\n\n</parameter>";
        CHECK(janas_gmatch_feed(m, v, strlen(v), -1) == 0, "until: text");
        CHECK(!janas_gmatch_accepting(m), "until: accepting too soon");
        CHECK(!janas_gmatch_try(m, "x", 1, -1), "until: bytes after end");
        CHECK(janas_gmatch_feed(m, NULL, 0, 9) == 0, "until: token 9");
        CHECK(janas_gmatch_accepting(m), "until: not accepting");
        janas_gmatch_reset(m);
        CHECK(janas_gmatch_try(m, NULL, 0, 7) && !janas_gmatch_accepting(m),
              "until: reset");
    }
    janas_gmatch_free(m);
    janas_grammar_free(g);
    janas_gbuild_free(b);
}

/* A long document: the match must not blow up in memory or in ways. */
static void test_long(void)
{
    struct gram x;
    if (make_schema(&x, "{\"type\":\"array\",\"items\":{\"type\":\"object\","
                        "\"properties\":{\"id\":{\"type\":\"integer\"},"
                        "\"s\":{\"type\":\"string\"}},"
                        "\"required\":[\"id\",\"s\"]}}") != 0) {
        CHECK(0, "long: grammar");
        gram_free(&x);
        return;
    }
    struct janas_buf t = {0};
    janas_buf_puts(&t, "[");
    for (int i = 0; i < 2000; i++)
        janas_buf_printf(&t, "%s{\"id\": %d, \"s\": \"item number %d\"}",
                         i ? ", " : "", i, i);
    janas_buf_puts(&t, "]");
    CHECK(!t.oom && matches(&x, t.p, 5) == 1, "long: refused");
    janas_buf_free(&t);
    gram_free(&x);
}

int main(void)
{
    test_json();
    test_json_mangled();
    test_schemas();
    test_until_and_tokens();
    test_long();
    if (failures) {
        printf("test_grammar: %d failures\n", failures);
        return 1;
    }
    printf("test_grammar: ok\n");
    return 0;
}
