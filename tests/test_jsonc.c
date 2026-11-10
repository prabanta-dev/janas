/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_jsonc.c - a member of a JSON file with comments changed in place
 * (src/common/jsonc.c): every byte but the member's kept, comments and
 * trailing commas read, a broken file refused.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/jsonc.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void set(const char *in, const char *const *path, int n_path,
                const char *val, const char *want)
{
    char *out = NULL;
    size_t n = 0;
    int r = janas_jsonc_set(in, strlen(in), path, n_path, val, &out, &n);
    if (!want) {
        CHECK(r == -1, "set on %s: %d, want refused", in, r);
    } else {
        CHECK(r == 0 && out && strcmp(out, want) == 0 && n == strlen(want),
              "set on [%s]:\n got [%s]\nwant [%s]", in, out ? out : "(null)",
              want);
    }
    free(out);
}

static void del(const char *in, const char *const *path, int n_path, int want_r,
                const char *want)
{
    char *out = NULL;
    size_t n = 0;
    int r = janas_jsonc_remove(in, strlen(in), path, n_path, &out, &n);
    CHECK(r == want_r, "remove on [%s]: %d, want %d", in, r, want_r);
    if (want)
        CHECK(out && strcmp(out, want) == 0,
              "remove on [%s]:\n got [%s]\nwant [%s]", in, out ? out : "(null)",
              want);
    free(out);
}

static void settings(void)
{
    const char *k[] = {"llama-vscode.endpoint"};
    /* a new member last, indented as the others */
    set("{\n    \"editor.fontSize\": 14\n}\n", k, 1, "\"http://x\"",
        "{\n    \"editor.fontSize\": 14,\n    \"llama-vscode.endpoint\": "
        "\"http://x\"\n}\n");
    /* replaced in place, comments kept */
    set("{\n  // mine\n  \"llama-vscode.endpoint\": \"old\", /* c */\n  "
        "\"a\": [1, 2,],\n}\n",
        k, 1, "\"new\"",
        "{\n  // mine\n  \"llama-vscode.endpoint\": \"new\", /* c */\n  "
        "\"a\": [1, 2,],\n}\n");
    /* after a trailing comma */
    set("{\n\t\"a\": 1,\n}", k, 1, "true",
        "{\n\t\"a\": 1,\n\t\"llama-vscode.endpoint\": true,\n}");
    /* a comment to the end of the last line stays there */
    set("{\n  \"a\": 1 // one\n}", k, 1, "2",
        "{\n  \"a\": 1, // one\n  \"llama-vscode.endpoint\": 2\n}");
    /* empty: a file, an object */
    set("", k, 1, "1", "{\n    \"llama-vscode.endpoint\": 1\n}\n");
    set("  {}  ", k, 1, "1", "  {\n      \"llama-vscode.endpoint\": 1\n  }  ");
    /* a comment-like text inside a string is a string */
    set("{\"u\": \"http://a//b\", \"llama-vscode.endpoint\": 0}", k, 1, "9",
        "{\"u\": \"http://a//b\", \"llama-vscode.endpoint\": 9}");
    /* broken: refused */
    set("{\"a\": 1", k, 1, "1", NULL);
    set("{\"a\": 1} x", k, 1, "1", NULL);
    set("[1]", k, 1, "1", NULL);
    set("{/* never closed }", k, 1, "1", NULL);

    /* a path: into an object there, an object made */
    const char *p[] = {"providers", "ollama"};
    set("{\"version\": 1, \"providers\": {}}", p, 2, "{\"x\": 1}",
        "{\"version\": 1, \"providers\": {\n    \"ollama\": {\"x\": 1}\n}}");
    set("{\"version\": 1}", p, 2, "2",
        "{\"version\": 1,\n    \"providers\": {\"ollama\": 2}}");

    /* removed with its comma and its line */
    del("{\n  \"a\": 1,\n  \"llama-vscode.endpoint\": \"x\"\n}\n", k, 1, 1,
        "{\n  \"a\": 1\n}\n");
    del("{\n  \"llama-vscode.endpoint\": \"x\",\n  \"a\": 1\n}\n", k, 1, 1,
        "{\n  \"a\": 1\n}\n");
    del("{\"a\": 1}", k, 1, 0, NULL);
    del("{\"a\": ", k, 1, -1, NULL);

    /* set and removed again: the same bytes, whatever the file's way */
    static const char *const back[] = {
        "{\n    // mine\n    \"a\": 14, /* big */\n    \"b\": \"x\",\n}\n",
        "{\n  \"a\": 1\n}\n",
        "{\n  \"a\": 1 // one\n}\n",
        "{\"a\": 1}",
    };
    for (size_t i = 0; i < sizeof back / sizeof *back; i++) {
        char *one = NULL, *two = NULL, *three = NULL;
        size_t n1, n2, n3;
        const char *k2[] = {"k2"};
        int ok = janas_jsonc_set(back[i], strlen(back[i]), k, 1, "1", &one,
                                 &n1) == 0 &&
                 janas_jsonc_set(one, n1, k2, 1, "2", &two, &n2) == 0 &&
                 janas_jsonc_remove(two, n2, k2, 1, &three, &n3) == 1;
        char *four = NULL;
        size_t n4;
        ok = ok && janas_jsonc_remove(three, n3, k, 1, &four, &n4) == 1;
        CHECK(ok && strcmp(four, back[i]) == 0 && strcmp(three, one) == 0,
              "round trip of [%s]: [%s]", back[i], four ? four : "(null)");
        free(one);
        free(two);
        free(three);
        free(four);
    }

    size_t at, len;
    const char *t = "{\"a\": {\"b\": [1, {\"c\": 2}]}}";
    const char *ab[] = {"a", "b"};
    CHECK(janas_jsonc_find(t, strlen(t), ab, 2, &at, &len) == 1 &&
              strncmp(t + at, "[1, {\"c\": 2}]", len) == 0,
          "find a.b");
    const char *ax[] = {"a", "x"};
    CHECK(janas_jsonc_find(t, strlen(t), ax, 2, &at, &len) == 0, "find a.x");
}

static void arrays(void)
{
    const char *in = "[\n  {\"name\": \"Other\", \"v\": 1},\n  {\"name\": "
                     "\"Janas\", \"v\": 2}\n]\n";
    char *out = NULL;
    size_t n;
    int r = janas_jsonc_array_put(in, strlen(in), "name", "Janas",
                                  "{\"name\": \"Janas\", \"v\": 3}", &out, &n);
    CHECK(r == 0 && out &&
              strcmp(out, "[\n  {\"name\": \"Other\", \"v\": 1},\n  {\"name\": "
                          "\"Janas\", \"v\": 3}\n]\n") == 0,
          "array replace: [%s]", out ? out : "(null)");
    free(out);
    const char *one = "[\n  {\"name\": \"Other\"}\n]";
    r = janas_jsonc_array_put(one, strlen(one), "name", "Janas",
                              "{\"name\": \"Janas\"}", &out, &n);
    CHECK(r == 0 && out &&
              strcmp(out, "[\n  {\"name\": \"Other\"},\n  {\"name\": "
                          "\"Janas\"}\n]") == 0,
          "array add: [%s]", out ? out : "(null)");
    free(out);
    r = janas_jsonc_array_put("", 0, "name", "Janas", "{\"name\": \"Janas\"}",
                              &out, &n);
    CHECK(r == 0 && out &&
              strcmp(out, "[\n    {\"name\": \"Janas\"}\n]\n") == 0,
          "array new: [%s]", out ? out : "(null)");
    free(out);
    r = janas_jsonc_array_remove(in, strlen(in), "name", "Janas", &out, &n);
    CHECK(r == 1 && out &&
              strcmp(out, "[\n  {\"name\": \"Other\", \"v\": 1}\n]\n") == 0,
          "array remove: [%s]", out ? out : "(null)");
    free(out);
    r = janas_jsonc_array_remove(one, strlen(one), "name", "Janas", &out, &n);
    CHECK(r == 0 && !out, "array remove none");
    r = janas_jsonc_array_put("{}", 2, "name", "Janas", "1", &out, &n);
    CHECK(r == -1, "array put on an object");
}

int main(void)
{
    settings();
    arrays();
    if (failures) {
        printf("test_jsonc: %d FAILED\n", failures);
        return 1;
    }
    printf("test_jsonc: all passed\n");
    return 0;
}
