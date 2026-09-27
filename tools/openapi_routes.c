/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * openapi_routes.c - the route table of janas-server, from OpenAI's OpenAPI
 * specification (https://github.com/openai/openai-openapi, openapi.json).
 *
 * Every operation of the specification gets a row: those janas-server
 * implements name their handler, the others answer 501 with a message that
 * says why. Running it again on a newer specification keeps the table whole.
 *
 * Usage: openapi_routes openapi.json > src/server/routes.inc
 *        openapi_routes openapi.json --coverage
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/sha256.h"
#include "llm/json.h"

/* operationId -> handler in src/server/ (declared in routes.h) */
static const char *const HANDLERS[][2] = {
    {"listModels", "srv_models_list"},
    {"retrieveModel", "srv_models_get"},
    {"createChatCompletion", "srv_chat_create"},
    {"createCompletion", "srv_completion_create"},
    {"deleteModel", "srv_models_delete"},
    {"listChatCompletions", "srv_stored_list"},
    {"getChatCompletion", "srv_stored_get"},
    {"updateChatCompletion", "srv_stored_update"},
    {"deleteChatCompletion", "srv_stored_delete"},
    {"getChatCompletionMessages", "srv_stored_messages"},
    {"createResponse", "srv_responses_create"},
    {"getResponse", "srv_responses_get"},
    {"deleteResponse", "srv_responses_delete"},
    {"cancelResponse", "srv_responses_cancel"},
    {"listInputItems", "srv_responses_input_items"},
    {"Getinputtokencounts", "srv_responses_input_tokens"},
    {"Compactconversation", "srv_responses_compact"},
    {"createConversation", "srv_conv_create"},
    {"getConversation", "srv_conv_get"},
    {"updateConversation", "srv_conv_update"},
    {"deleteConversation", "srv_conv_delete"},
    {"listConversationItems", "srv_conv_items_list"},
    {"createConversationItems", "srv_conv_items_create"},
    {"getConversationItem", "srv_conv_item_get"},
    {"deleteConversationItem", "srv_conv_item_delete"},
    {"createEmbedding", "srv_embeddings_create"},
    {"createModeration", "srv_moderations_create"},
};

/* What an operation that is not implemented is, by its tag: the text of the
   501 answer. "planned" operations are ones a local text model can serve;
   a tag not listed is "cloud". */
static const char *const KIND[][2] = {
    {"Chat", "planned"},          {"Completions", "planned"},
    {"Models", "planned"},        {"Responses", "planned"},
    {"Conversations", "planned"}, {"Embeddings", "planned"},
    {"Moderations", "planned"},   {"Files", "planned"},
    {"Uploads", "planned"},       {"Batch", "planned"},
    {"Vector stores", "planned"}, {"Assistants", "planned"},
    {"Evals", "planned"},         {"Audio", "other_model"},
    {"Images", "other_model"},    {"Videos", "other_model"},
    {"Realtime", "other_model"},  {"Live", "other_model"},
    {"Fine-tuning", "training"},
};

static const char *const METHODS[][2] = {
    {"get", "GET"},       {"post", "POST"},   {"put", "PUT"},
    {"delete", "DELETE"}, {"patch", "PATCH"},
};

#define COUNT(a) (sizeof(a) / sizeof *(a))

struct op {
    const char *method; /* upper case */
    char *path;         /* without a "?beta=true" and the like */
    const char *id, *tag;
};

static const char *lookup(const char *const (*t)[2], size_t n, const char *key)
{
    for (size_t i = 0; i < n; i++)
        if (!strcmp(t[i][0], key))
            return t[i][1];
    return NULL;
}

static const char *handler(const char *id)
{
    return lookup(HANDLERS, COUNT(HANDLERS), id);
}

static int by_path_method(const void *a, const void *b)
{
    const struct op *x = a, *y = b;
    int c = strcmp(x->path, y->path);
    return c ? c : strcmp(x->method, y->method);
}

static int by_string(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static char *read_all(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return NULL;
    }
    size_t cap = 1 << 20, len = 0;
    char *p = malloc(cap);
    for (size_t r; p && (r = fread(p + len, 1, cap - len, f)) > 0;) {
        len += r;
        if (len == cap) {
            char *q = realloc(p, cap *= 2);
            if (!q)
                free(p);
            p = q;
        }
    }
    if (!p || ferror(f)) {
        fprintf(stderr, "%s: cannot be read\n", path);
        free(p);
        p = NULL;
    }
    fclose(f);
    *n = len;
    return p;
}

/* Every operation once: the "?..." variants of a path are the same route. */
static struct op *operations(const struct janas_json *paths, size_t *count)
{
    size_t cap = 0, n = 0;
    for (const struct janas_json *p = paths->child; p; p = p->next)
        cap += COUNT(METHODS);
    struct op *ops = calloc(cap ? cap : 1, sizeof *ops);
    if (!ops)
        return NULL;
    for (const struct janas_json *p = paths->child; p; p = p->next) {
        size_t bare_n = strcspn(p->key, "?");
        for (const struct janas_json *o = p->child;
             p->type == JANAS_JSON_OBJECT && o; o = o->next) {
            const char *method = lookup(METHODS, COUNT(METHODS), o->key);
            if (!method)
                continue;
            int seen = 0;
            for (size_t i = 0; i < n && !seen; i++)
                seen = ops[i].method == method &&
                       strlen(ops[i].path) == bare_n &&
                       !memcmp(ops[i].path, p->key, bare_n);
            if (seen)
                continue;
            const struct janas_json *tags = janas_json_get(o, "tags");
            const char *tag = tags && tags->type == JANAS_JSON_ARRAY
                                  ? janas_json_str(tags->child)
                                  : NULL;
            const char *id = janas_json_str(janas_json_get(o, "operationId"));
            ops[n] = (struct op){
                .method = method,
                .path = strndup(p->key, bare_n),
                .id = id ? id : "",
                .tag = tag ? tag : "-",
            };
            if (!ops[n].path) {
                for (size_t i = 0; i < n; i++)
                    free(ops[i].path);
                free(ops);
                return NULL;
            }
            n++;
        }
    }
    *count = n;
    return ops;
}

static void c_str(const char *s)
{
    putchar('"');
    for (; *s; s++) {
        if (*s == '\\' || *s == '"')
            putchar('\\');
        putchar(*s);
    }
    putchar('"');
}

static void coverage(const struct op *ops, size_t n)
{
    const char **tags = malloc((n ? n : 1) * sizeof *tags);
    if (!tags)
        return;
    size_t done = 0;
    for (size_t i = 0; i < n; i++) {
        tags[i] = ops[i].tag;
        done += handler(ops[i].id) != NULL;
    }
    qsort(tags, n, sizeof *tags, by_string);
    printf("%zu of %zu operations implemented\n", done, n);
    for (size_t i = 0; i < n;) {
        size_t d = 0, k = 0;
        for (size_t j = 0; j < n; j++)
            if (!strcmp(ops[j].tag, tags[i])) {
                k++;
                d += handler(ops[j].id) != NULL;
            }
        printf("  %-40s %3zu / %zu\n", tags[i], d, k);
        i += k;
    }
    free(tags);
}

/* The rows of routes.inc, with what they came from. */
static void table(const struct janas_json *spec, const struct op *ops, size_t n,
                  const char *raw, size_t raw_n)
{
    const char *ver =
        janas_json_str(janas_json_get(janas_json_get(spec, "info"), "version"));
    struct janas_sha256 sha;
    char hex[65];
    janas_sha256_init(&sha);
    janas_sha256_update(&sha, raw, raw_n);
    janas_sha256_hex(&sha, hex);
    size_t done = 0;
    for (size_t i = 0; i < n; i++)
        done += handler(ops[i].id) != NULL;
    printf("/* Generated by tools/openapi_routes.c - do not edit.\n");
    printf("   OpenAI OpenAPI %s, sha256 %s,\n", ver ? ver : "?", hex);
    printf("   %zu operations, %zu implemented. */\n", n, done);
    for (size_t i = 0; i < n; i++) {
        const char *h = handler(ops[i].id);
        const char *kind =
            h ? "implemented" : lookup(KIND, COUNT(KIND), ops[i].tag);
        printf("JANAS_ROUTE(%s, ", ops[i].method);
        c_str(ops[i].path);
        printf(", ");
        c_str(ops[i].id);
        printf(", ");
        c_str(ops[i].tag);
        printf(", %s, %s)\n", kind ? kind : "cloud", h ? h : "NULL");
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: openapi_routes openapi.json "
                        "> src/server/routes.inc\n"
                        "       openapi_routes openapi.json --coverage\n");
        return 2;
    }
    int cover = 0;
    for (int i = 2; i < argc; i++)
        cover |= !strcmp(argv[i], "--coverage");
    size_t raw_n, n = 0;
    char *raw = read_all(argv[1], &raw_n);
    if (!raw)
        return 1;
    char err[256];
    struct janas_json_doc *doc = janas_json_parse(raw, raw_n, err, sizeof err);
    const struct janas_json *spec = doc ? janas_json_root(doc) : NULL;
    const struct janas_json *paths = janas_json_get(spec, "paths");
    struct op *ops = NULL;
    int rc = 1;
    if (!doc)
        fprintf(stderr, "%s: %s\n", argv[1], err);
    else if (!paths || paths->type != JANAS_JSON_OBJECT)
        fprintf(stderr, "%s: no paths\n", argv[1]);
    else if (!(ops = operations(paths, &n)))
        fprintf(stderr, "out of memory\n");
    else
        rc = 0;
    if (rc == 0) {
        qsort(ops, n, sizeof *ops, by_path_method);
        if (cover)
            coverage(ops, n);
        else
            table(spec, ops, n, raw, raw_n);
        rc = ferror(stdout) ? 1 : 0;
    }
    for (size_t i = 0; ops && i < n; i++)
        free(ops[i].path);
    free(ops);
    if (doc)
        janas_json_free(doc);
    free(raw);
    return rc;
}
