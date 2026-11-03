/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_server_anthropic.c - Anthropic's requests to janas-server's
 * /v1/messages as the job they become (src/server/anthropic_req.c, then
 * request.c): the system prompt in blocks, a system message within the
 * messages (Claude Code's), pictures, tool_use and tool_result, tools and
 * tool_choice, thinking off unless asked, max_tokens counting the
 * reasoning; what is refused (Anthropic's own tools, documents); and
 * mangled requests, which must neither crash nor leak (run it under asan to
 * know the second).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "server/engine.c"
#include "server/fim.c"
#include "server/images.c"

/* images.c reads files from the store: none here */
int srv_files_get(struct srv_files *f, const char *id, struct srv_file *out)
{
    (void)f, (void)id, (void)out;
    return -1;
}

int srv_files_open_content(struct srv_files *f, const char *id)
{
    (void)f, (void)id;
    return -1;
}
#include "server/anthropic_req.c"
#include "server/request.c"
#include "server/request_opts.c"

/* engine.c names tool calls with it; it lives in http.c, not linked here */
void srv_new_id(char *buf, size_t cap, const char *prefix)
{
    snprintf(buf, cap, "%s0", prefix);
}

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* An Anthropic request through anthropic_req.c and request.c, as the
   handler takes it; the reason of a refused one, NULL when taken (*out). */
static const char *job_of(const char *body, size_t len, struct srv_job **out)
{
    static char msg[600];
    yyjson_doc *doc = yyjson_read(body, len, 0);
    if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc))) {
        yyjson_doc_free(doc);
        return "not a JSON object";
    }
    size_t n = 0;
    char why[300];
    char *q = an_request_body(yyjson_doc_get_root(doc), &n, why, sizeof why);
    yyjson_doc_free(doc);
    if (!q) {
        snprintf(msg, sizeof msg, "%s", why);
        return msg;
    }
    struct srv_job *j = srv_job_new();
    struct srv_parsed pr;
    struct srv_perr e;
    int rc = srv_parse_request(q, n, 1, j, &pr, &e);
    srv_parsed_free(&pr);
    free(q);
    if (rc == 0 && out) {
        *out = j;
        return NULL;
    }
    snprintf(msg, sizeof msg, "%s", rc ? e.msg : "");
    srv_job_release(j);
    return rc ? msg : NULL;
}

static const char conv[] =
    "{\"model\":\"claude-x\",\"max_tokens\":1000,\"temperature\":0.5,"
    "\"top_k\":20,\"stop_sequences\":[\"END\"],"
    "\"system\":[{\"type\":\"text\",\"text\":\"Be brief.\","
    "\"cache_control\":{\"type\":\"ephemeral\"}}],"
    "\"tools\":[{\"name\":\"get_weather\",\"description\":\"The weather\","
    "\"input_schema\":{\"type\":\"object\",\"properties\":{\"city\":"
    "{\"type\":\"string\"}},\"required\":[\"city\"]}}],"
    "\"tool_choice\":{\"type\":\"any\"},"
    "\"messages\":["
    "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Here?\"},"
    "{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":"
    "\"image/jpeg\",\"data\":\"/9j/4AAQ\"}}]},"
    "{\"role\":\"system\",\"content\":[{\"type\":\"text\",\"text\":"
    "\"Italian.\"}]},"
    "{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\","
    "\"thinking\":\"hm\",\"signature\":\"x\"},{\"type\":\"text\",\"text\":"
    "\"Let me look.\"},{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":"
    "\"get_weather\",\"input\":{\"city\":\"Rome\"}}]},"
    "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\","
    "\"tool_use_id\":\"toolu_1\",\"content\":[{\"type\":\"text\",\"text\":"
    "\"11 C\"}]},{\"type\":\"text\",\"text\":\"And tomorrow?\"}]}]}";

static void valid(void)
{
    struct srv_job *j = NULL;
    const char *err = job_of(conv, sizeof conv - 1, &j);
    CHECK(!err, "a conversation refused: %s", err);
    if (!j)
        return;
    CHECK(j->n_msg == 5, "messages: %d, want 5", j->n_msg);
    if (j->n_msg == 5) {
        CHECK(j->msg[0].role == JANAS_LLM_ROLE_SYSTEM &&
                  !strcmp(j->msg[0].text, "Be brief.\n\nItalian."),
              "the system prompt and the system message: '%s'", j->msg[0].text);
        CHECK(j->msg[1].role == JANAS_LLM_ROLE_USER &&
                  !strcmp(j->msg[1].text, "Here?") && j->msg[1].n_imgs == 1 &&
                  !strcmp(j->msg[1].imgs[0].ref,
                          "data:image/jpeg;base64,/9j/4AAQ"),
              "the user's text and picture");
        CHECK(j->msg[2].role == JANAS_LLM_ROLE_ASSISTANT &&
                  !strcmp(j->msg[2].text, "Let me look.") &&
                  j->msg[2].n_calls == 1 &&
                  !strcmp(j->msg[2].call_name[0], "get_weather") &&
                  !strcmp(j->msg[2].call_args[0], "{\"city\":\"Rome\"}"),
              "the assistant's text and call, its thinking left out: '%s'",
              j->msg[2].text);
        CHECK(j->msg[3].role == JANAS_LLM_ROLE_TOOL &&
                  !strcmp(j->msg[3].text, "11 C"),
              "the tool's result");
        CHECK(j->msg[4].role == JANAS_LLM_ROLE_USER &&
                  !strcmp(j->msg[4].text, "And tomorrow?"),
              "the user's text after the result");
    }
    CHECK(j->p.max_reply == 1000 && j->p.max_answer == 0,
          "max_tokens counts the reasoning: %d %d", j->p.max_reply,
          j->p.max_answer);
    CHECK(j->p.temperature > 0.49f && j->p.temperature < 0.51f &&
              j->p.top_k == 20,
          "sampling");
    CHECK(j->n_stop == 1 && !strcmp(j->stop[0], "END"), "stop sequences");
    CHECK(j->tools && strstr(j->tools, "\"parameters\"") &&
              strstr(j->tools, "get_weather"),
          "tools as functions");
    CHECK(j->tool_choice == JANAS_LLM_TOOLS_REQUIRED, "tool_choice any: %d",
          j->tool_choice);
    CHECK(j->p.thinking == 0, "thinking not asked, yet on");
    srv_job_release(j);

    j = NULL;
    static const char th[] =
        "{\"max_tokens\":4000,\"thinking\":{\"type\":\"enabled\","
        "\"budget_tokens\":2000},\"tool_choice\":{\"type\":\"tool\","
        "\"name\":\"f\"},\"tools\":[{\"name\":\"f\",\"input_schema\":"
        "{\"type\":\"object\"}}],\"system\":\"S\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";
    err = job_of(th, sizeof th - 1, &j);
    CHECK(!err, "thinking refused: %s", err);
    if (j) {
        CHECK(j->p.thinking == 1 && j->p.max_reasoning == 2000,
              "thinking and its budget: %d %d", j->p.thinking,
              j->p.max_reasoning);
        CHECK(j->tool_choice == JANAS_LLM_TOOLS_FUNCTION &&
                  !strcmp(j->tool_name, "f"),
              "tool_choice tool");
        srv_job_release(j);
    }

    j = NULL;
    static const char err_result[] =
        "{\"max_tokens\":10,\"messages\":[{\"role\":\"assistant\","
        "\"content\":[{\"type\":\"tool_use\",\"id\":\"t\",\"name\":\"f\","
        "\"input\":{}}]},{\"role\":\"user\",\"content\":[{\"type\":"
        "\"tool_result\",\"tool_use_id\":\"t\",\"is_error\":true,"
        "\"content\":\"no such city\"}]}]}";
    err = job_of(err_result, sizeof err_result - 1, &j);
    CHECK(!err, "an error result refused: %s", err);
    if (j) {
        CHECK(j->n_msg == 2 && !strcmp(j->msg[1].text, "Error: no such city"),
              "an error result as one");
        srv_job_release(j);
    }
}

static void refused(void)
{
    static const char *const bad[] = {
        "{\"max_tokens\":9,\"tools\":[{\"type\":\"web_search_20250305\","
        "\"name\":\"web_search\"}],\"messages\":[{\"role\":\"user\","
        "\"content\":\"x\"}]}",
        "{\"max_tokens\":9,\"messages\":[{\"role\":\"user\",\"content\":"
        "[{\"type\":\"document\",\"source\":{}}]}]}",
        "{\"max_tokens\":9,\"messages\":[{\"role\":\"robot\",\"content\":"
        "\"x\"}]}",
        "{\"max_tokens\":9,\"messages\":{}}",
        "{\"max_tokens\":9,\"system\":7,\"messages\":[{\"role\":\"user\","
        "\"content\":\"x\"}]}",
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++)
        CHECK(job_of(bad[i], strlen(bad[i]), NULL) != NULL,
              "taken, and should not be: %s", bad[i]);
}

static uint64_t rng = 0xa17a0b1cULL;
static int n_json, n_taken;

static uint32_t next(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

static void mangled(void)
{
    size_t n0 = sizeof conv - 1;
    for (int it = 0; it < 20000; it++) {
        size_t cap = 4 * n0 + 2, n = n0;
        char *s = malloc(cap);
        memcpy(s, conv, n);
        int edits = 1 + (int)(next() % 4);
        for (int e = 0; e < edits; e++) {
            size_t at = n ? next() % n : 0;
            switch (next() % 3) {
            case 0:
                s[at] = (char)" {}[]\":,0a\\tf"[next() % 13];
                break;
            case 1:
                n = at;
                break;
            default:
                if (n + (n - at) / 2 <= cap) {
                    size_t k = (n - at) / 2;
                    memmove(s + at + k, s + at, n - at);
                    n += k;
                }
            }
        }
        yyjson_doc *doc = yyjson_read(s, n, 0);
        if (doc && yyjson_is_obj(yyjson_doc_get_root(doc)))
            n_json++;
        yyjson_doc_free(doc);
        struct srv_job *j = NULL;
        if (!job_of(s, n, &j)) {
            n_taken++;
            srv_job_release(j);
        }
        free(s);
    }
}

int main(void)
{
    valid();
    refused();
    mangled();
    if (failures) {
        printf("test_server_anthropic: %d failures\n", failures);
        return 1;
    }
    printf("test_server_anthropic: %d of 20000 mangled requests still JSON, "
           "%d taken, ok\n",
           n_json, n_taken);
    return 0;
}
