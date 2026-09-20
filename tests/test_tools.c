/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_tools.c - the tools in the words of the chat templates
 * (src/llm/tools.c): the three dialects told apart, the system message and
 * the calls of earlier turns written as the templates write them, the calls
 * a model writes read back, and the grammar of a call, which must take what
 * the templates show and refuse arguments outside the schema. Tool
 * definitions come from whoever sends a request: mangled ones must neither
 * crash nor leak (run it under asan to know the second).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/tools.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char *TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"description\":\" Weather now \",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"city\":{\"type\":\"string\",\"description\":\"a "
    "city\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}}},"
    "{\"name\":\"ping\"}]";

static struct janas_toolset *tools(void)
{
    char err[128];
    struct janas_toolset *t =
        janas_toolset_parse(TOOLS, strlen(TOOLS), err, sizeof(err));
    CHECK(t != NULL, "toolset: %s", err);
    return t;
}

static void test_dialects(void)
{
    CHECK(janas_tools_dialect("{{ x }}") == JANAS_TOOLS_NONE, "none");
    CHECK(janas_tools_dialect("<tool_call>\\n{\"name\"") == JANAS_TOOLS_JSON,
          "json");
    CHECK(janas_tools_dialect("<tool_call>\\n<function=x>") == JANAS_TOOLS_XML,
          "xml");
    CHECK(janas_tools_dialect("<tool_call><function=x> <name>") ==
              JANAS_TOOLS_XML_CODER,
          "xml coder");
}

static void test_system(void)
{
    struct janas_toolset *t = tools();
    if (!t)
        return;
    CHECK(janas_toolset_count(t) == 2, "count");
    CHECK(janas_toolset_find(t, "ping", 4) == 1, "find");
    CHECK(janas_toolset_find(t, "pin", 3) == -1, "find missing");
    struct janas_buf b = {0};
    janas_tools_system(&b, JANAS_TOOLS_JSON, t, "Be brief.", 9);
    CHECK(b.p && strncmp(b.p, "Be brief.\n\n# Tools\n\nYou may call", 31) == 0,
          "json system start: %s", b.p);
    CHECK(b.p && strstr(b.p, "<tools>\n{\"type\": \"function\", \"function\": "
                             "{\"name\": \"get_weather\""),
          "json system tools: %s", b.p);
    CHECK(b.p && strstr(b.p, "\n{\"name\": \"ping\"}\n</tools>"),
          "json system bare function");
    janas_buf_free(&b);
    janas_tools_system(&b, JANAS_TOOLS_XML, t, " Be brief. ", 11);
    CHECK(b.p && strncmp(b.p, "# Tools\n\nYou have access", 24) == 0,
          "xml system start");
    const char *end = "</IMPORTANT>\n\nBe brief.";
    size_t n = b.p ? strlen(b.p) : 0, k = strlen(end);
    CHECK(n > k && strcmp(b.p + n - k, end) == 0, "xml system end: %s",
          b.p ? b.p + n - k : "");
    janas_buf_free(&b);
    janas_tools_system(&b, JANAS_TOOLS_XML_CODER, t, NULL, 0);
    CHECK(b.p && strncmp(b.p, "You are Qwen, a helpful AI assistant", 36) == 0,
          "coder default system");
    CHECK(b.p && strstr(b.p, "<function>\n<name>get_weather</name>\n"
                             "<description>Weather now</description>\n"
                             "<parameters>\n<parameter>\n<name>city</name>\n"
                             "<type>string</type>\n<description>a city"
                             "</description>\n</parameter>"),
          "coder function: %s", b.p);
    CHECK(b.p && strstr(b.p, "<required>[\"city\"]</required>\n</parameters>"),
          "coder extra keys");
    janas_buf_free(&b);
    janas_toolset_free(t);
    char err[128];
    static const char *const bad[] = {"{}", "[1]", "[{\"type\":\"x\"}]",
                                      "[{\"name\":\"a>b\"}]",
                                      "[{\"name\":\"f\",\"parameters\":1}]"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        t = janas_toolset_parse(bad[i], strlen(bad[i]), err, sizeof(err));
        CHECK(!t, "bad toolset %zu accepted", i);
        janas_toolset_free(t);
    }
}

static void test_assistant(void)
{
    struct janas_tool_call calls[] = {
        {.name = "get_weather", .args = "{\"city\": \"Rome\", \"days\": 2}"},
        {.name = "ping", .args = "{}"},
    };
    for (int i = 0; i < 2; i++) {
        calls[i].name_n = strlen(calls[i].name);
        calls[i].args_n = strlen(calls[i].args);
    }
    struct janas_buf b = {0};
    janas_tools_assistant(&b, JANAS_TOOLS_JSON, "", 0, calls, 2);
    CHECK(b.p && strcmp(b.p, "<tool_call>\n{\"name\": \"get_weather\", "
                             "\"arguments\": {\"city\": \"Rome\", \"days\": "
                             "2}}\n</tool_call>\n<tool_call>\n{\"name\": "
                             "\"ping\", \"arguments\": {}}\n</tool_call>") == 0,
          "json assistant: %s", b.p);
    janas_buf_free(&b);
    janas_tools_assistant(&b, JANAS_TOOLS_XML, "Let me see.", 11, calls, 1);
    CHECK(b.p && strcmp(b.p, "Let me see.\n\n<tool_call>\n<function="
                             "get_weather>\n<parameter=city>\nRome\n"
                             "</parameter>\n<parameter=days>\n2\n"
                             "</parameter>\n</function>\n</tool_call>") == 0,
          "xml assistant: %s", b.p);
    janas_buf_free(&b);
}

static void test_parse(void)
{
    struct janas_toolset *t = tools();
    if (!t)
        return;
    struct janas_buf name = {0}, args = {0};
    const char *j = "\n{\"name\": \"get_weather\", \"arguments\": "
                    "{\"city\":\"Rome\"}}\n";
    CHECK(janas_tools_parse(JANAS_TOOLS_JSON, t, j, strlen(j), &name, &args) ==
                  0 &&
              strcmp(name.p, "get_weather") == 0 &&
              strcmp(args.p, "{\"city\": \"Rome\"}") == 0,
          "parse json: %s %s", name.p, args.p);
    janas_buf_free(&name);
    janas_buf_free(&args);
    const char *x = "\n<function=get_weather>\n<parameter=city>\nNew\nYork\n"
                    "</parameter>\n<parameter=days>\n3\n</parameter>\n"
                    "</function>\n";
    CHECK(janas_tools_parse(JANAS_TOOLS_XML, t, x, strlen(x), &name, &args) ==
                  0 &&
              strcmp(name.p, "get_weather") == 0 &&
              strcmp(args.p, "{\"city\": \"New\\nYork\", \"days\": 3}") == 0,
          "parse xml: %s %s", name.p, args.p);
    janas_buf_free(&name);
    janas_buf_free(&args);
    CHECK(janas_tools_parse(JANAS_TOOLS_JSON, t, "{\"x\":1}", 7, &name,
                            &args) != 0,
          "parse json without a name");
    janas_buf_free(&name);
    janas_buf_free(&args);
    janas_toolset_free(t);
}

/* Whether the body (then the closing token 9) matches the call rule. */
static int call_ok(enum janas_tool_dialect d, const char *body)
{
    struct janas_toolset *t = tools();
    struct janas_gbuild *b = janas_gbuild_new();
    struct janas_jsg *j = janas_jsg_new(b);
    uint32_t r = janas_tools_rule(b, j, d, t, -1, 9);
    struct janas_grammar *g = janas_grammar_make(b);
    struct janas_gmatch *m = g ? janas_gmatch_new(g, r) : NULL;
    int ok = m && janas_gmatch_feed(m, body, strlen(body), -1) == 0 &&
             janas_gmatch_feed(m, NULL, 0, 9) == 0 && janas_gmatch_accepting(m);
    janas_gmatch_free(m);
    janas_grammar_free(g);
    janas_jsg_free(j);
    janas_gbuild_free(b);
    janas_toolset_free(t);
    return ok;
}

static void test_grammar(void)
{
    CHECK(call_ok(JANAS_TOOLS_JSON, "\n{\"name\": \"get_weather\", "
                                    "\"arguments\": {\"city\": \"Rome\"}}\n"),
          "grammar json");
    CHECK(call_ok(JANAS_TOOLS_JSON,
                  "{\"name\": \"ping\", \"arguments\": {\"any\": [1]}}"),
          "grammar json, no parameters");
    CHECK(!call_ok(JANAS_TOOLS_JSON, "{\"name\": \"get_weather\", "
                                     "\"arguments\": {\"days\": 2}}"),
          "grammar json: required missing");
    CHECK(!call_ok(JANAS_TOOLS_JSON, "{\"name\": \"nope\", \"arguments\": {}}"),
          "grammar json: unknown function");
    CHECK(call_ok(JANAS_TOOLS_XML,
                  "\n<function=get_weather>\n<parameter=city>\nRome\n"
                  "</parameter>\n<parameter=days>\n2\n</parameter>\n"
                  "</function>\n"),
          "grammar xml");
    CHECK(!call_ok(JANAS_TOOLS_XML, "<function=get_weather>\n<parameter="
                                    "days>\nx\n</parameter>\n</function>"),
          "grammar xml: bad integer, required missing");
}

static uint64_t rng = 0x70015;
static uint32_t next_rand(void)
{
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(rng >> 33);
}

static void test_mangled(void)
{
    char buf[1024];
    size_t base = strlen(TOOLS);
    int parsed = 0;
    for (int it = 0; it < 5000; it++) {
        size_t n = base;
        memcpy(buf, TOOLS, n);
        for (int e = 1 + (int)(next_rand() % 3); e > 0 && n; e--) {
            uint32_t at = next_rand() % (uint32_t)n;
            if (next_rand() % 2)
                buf[at] = (char)next_rand();
            else
                n = at;
        }
        char err[64];
        struct janas_toolset *t = janas_toolset_parse(buf, n, err, sizeof(err));
        if (!t)
            continue;
        parsed++;
        for (int d = JANAS_TOOLS_JSON; d <= JANAS_TOOLS_XML_CODER; d++) {
            struct janas_buf s = {0};
            janas_tools_system(&s, d, t, "x", 1);
            janas_buf_free(&s);
            struct janas_gbuild *b = janas_gbuild_new();
            struct janas_jsg *j = janas_jsg_new(b);
            janas_tools_rule(b, j, d, t, -1, 9);
            struct janas_grammar *g = janas_grammar_make(b);
            janas_grammar_free(g);
            janas_jsg_free(j);
            janas_gbuild_free(b);
        }
        janas_toolset_free(t);
    }
    CHECK(parsed > 0, "mangled: none parsed");
}

int main(void)
{
    test_dialects();
    test_system();
    test_assistant();
    test_parse();
    test_grammar();
    test_mangled();
    if (failures) {
        printf("test_tools: %d failures\n", failures);
        return 1;
    }
    printf("test_tools: ok\n");
    return 0;
}
