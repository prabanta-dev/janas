/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_server_request.c - the reading of requests to janas-server's chat
 * and completion endpoints (src/server/request.c): what anybody who can
 * reach the server sends. Valid requests must come out as the right job,
 * invalid ones as the right error, and mangled ones - bytes flipped, cut,
 * repeated, tens of thousands of them from a fixed seed - must neither
 * crash nor leak (run it under asan to know the second).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "server/engine.c"
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

/* Reads body; returns the error code (NULL when it was accepted). */
static const char *parse(const char *body, int chat, struct srv_job **out)
{
    static char code[64];
    struct srv_job *j = srv_job_new();
    struct srv_parsed pr;
    struct srv_perr e;
    int rc = srv_parse_request(body, strlen(body), chat, j, &pr, &e);
    srv_parsed_free(&pr); /* what the handler would hand to its answer */
    if (out && rc == 0) {
        *out = j;
        return NULL;
    }
    snprintf(code, sizeof(code), "%s", rc ? (e.code ? e.code : "oom") : "");
    srv_job_release(j);
    return rc ? code : NULL;
}

static void valid(void)
{
    struct srv_job *j = NULL;
    const char *err = parse(
        "{\"model\":\"x\",\"messages\":["
        "{\"role\":\"system\",\"content\":\"Be brief.\"},"
        "{\"role\":\"user\",\"content\":\"Hi\"},"
        "{\"role\":\"assistant\",\"content\":null},"
        "{\"role\":\"developer\",\"content\":\"Italian.\"},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Ci\"},"
        "{\"type\":\"text\",\"text\":\"ao\"}]}],"
        "\"temperature\":0.2,\"top_p\":0.9,\"max_tokens\":50,"
        "\"stop\":[\"END\",\"\"],\"seed\":7,\"stream\":true,"
        "\"stream_options\":{\"include_usage\":true},"
        "\"reasoning_effort\":\"none\"}",
        1, &j);
    CHECK(!err, "valid chat request refused: %s", err);
    if (!j)
        return;
    CHECK(j->n_msg == 4, "messages: %d, want 4", j->n_msg);
    CHECK(j->msg[0].role == JANAS_LLM_ROLE_SYSTEM &&
              !strcmp(j->msg[0].text, "Be brief.\n\nItalian."),
          "system messages not joined at the start: '%s'", j->msg[0].text);
    CHECK(j->msg[2].role == JANAS_LLM_ROLE_ASSISTANT && j->msg[2].n == 0,
          "null content is not empty");
    CHECK(!strcmp(j->msg[3].text, "Ciao"), "text parts not joined: '%s'",
          j->msg[3].text);
    /* a chat's max_tokens is the answer's, the reasoning not counted */
    CHECK(j->p.temperature > 0.19f && j->p.temperature < 0.21f &&
              j->p.max_answer == 50 && j->p.max_reply == 0 && j->p.seed == 7 &&
              j->p.thinking == 0,
          "parameters not taken");
    CHECK(j->n_stop == 1 && !strcmp(j->stop[0], "END"), "stop strings: %d",
          j->n_stop);
    srv_job_release(j);

    /* max_completion_tokens counts the reasoning too, as OpenAI has it */
    j = NULL;
    err = parse("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],"
                "\"max_completion_tokens\":40,\"max_tokens\":9}",
                1, &j);
    CHECK(!err, "max_completion_tokens refused: %s", err);
    if (j) {
        CHECK(j->p.max_reply == 40 && j->p.max_answer == 0,
              "max_completion_tokens is not the whole reply's limit: %d %d",
              j->p.max_reply, j->p.max_answer);
        srv_job_release(j);
    }

    j = NULL;
    err = parse("{\"prompt\":[\"Once upon\"],\"echo\":true}", 0, &j);
    CHECK(!err, "valid completion refused: %s", err);
    if (j) {
        CHECK(j->kind == JOB_TEXT && j->n_prompts == 1 &&
                  !strcmp(j->prompts[0].text, "Once upon") &&
                  j->p.max_reply == 16 && j->n_choices == 1,
              "completion: prompt or OpenAI's default of 16 tokens");
        srv_job_release(j);
    }

    /* tools, calls in the history, the answer's form, several replies */
    j = NULL;
    err = parse(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Rome?\"},"
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{"
        "\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":"
        "\"w\",\"arguments\":\"{\\\"city\\\":\\\"Rome\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"21\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"w\"}}],"
        "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":"
        "\"w\"}},\"parallel_tool_calls\":false,\"n\":3,\"logprobs\":true,"
        "\"top_logprobs\":2,\"presence_penalty\":0.5,\"logit_bias\":{"
        "\"42\":-100},\"response_format\":{\"type\":\"json_schema\","
        "\"json_schema\":{\"name\":\"x\",\"schema\":{\"type\":"
        "\"object\"}}},\"store\":true,\"metadata\":{\"k\":\"v\"}}",
        1, &j);
    CHECK(!err, "chat request with tools refused: %s", err);
    if (j) {
        CHECK(j->n_msg == 3 && j->msg[1].n_calls == 1 &&
                  !strcmp(j->msg[1].call_name[0], "w") &&
                  !strcmp(j->msg[1].call_args[0], "{\"city\":\"Rome\"}") &&
                  j->msg[2].role == JANAS_LLM_ROLE_TOOL,
              "tool calls of the history");
        CHECK(j->tools && j->tool_choice == JANAS_LLM_TOOLS_FUNCTION &&
                  !strcmp(j->tool_name, "w") && !j->parallel,
              "tools and tool_choice");
        CHECK(j->n_choices == 3 && j->p.logprobs && j->p.top_logprobs == 2 &&
                  j->n_bias == 1 && j->bias_id[0] == 42 &&
                  j->format == JANAS_LLM_FORMAT_SCHEMA && j->schema,
              "n, logprobs, bias, format");
        srv_job_release(j);
    }
    j = NULL;
    err = parse("{\"prompt\":[[1,2,3],\"x\"],\"n\":2,\"best_of\":3,"
                "\"suffix\":\"end\",\"logprobs\":2}",
                0, &j);
    CHECK(!err, "completion with ids, best_of, suffix refused: %s", err);
    if (j) {
        CHECK(j->n_prompts == 2 && j->prompts[0].n_ids == 3 &&
                  j->n_choices == 6 && j->n_keep == 4 && j->suffix &&
                  j->p.top_logprobs == 2,
              "prompts, best_of, suffix");
        srv_job_release(j);
    }
}

static void invalid(void)
{
    static const struct {
        const char *body;
        int chat;
        const char *code;
    } cases[] = {
        {"", 1, "invalid_json"},
        {"[1,2]", 1, "invalid_json"},
        {"{\"messages\":[]}", 1, "missing_messages"},
        {"{\"messages\":[{\"role\":\"system\",\"content\":\"a\"}]}", 1,
         "missing_messages"},
        {"{\"messages\":[{\"content\":\"a\"}]}", 1, "invalid_value"},
        {"{\"messages\":[{\"role\":\"wizard\",\"content\":\"a\"}]}", 1,
         "invalid_value"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},"
         "{\"role\":\"assistant\",\"content\":\"b\"}]}",
         1, "invalid_value"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":"
         "\"image_url\"}]}]}",
         1, "unsupported_content"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],"
         "\"temperature\":3}",
         1, "invalid_value"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"n\":0}", 1,
         "invalid_value"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],"
         "\"tools\":[{\"type\":\"function\"}]}",
         1, "invalid_value"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],"
         "\"response_format\":{\"type\":\"xml\"}}",
         1, "invalid_value"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],"
         "\"stop\":[\"a\",\"b\",\"c\",\"d\",\"e\"]}",
         1, "invalid_value"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],"
         "\"top_logprobs\":3}",
         1, "invalid_value"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],"
         "\"logit_bias\":{\"x\":1}}",
         1, "invalid_value"},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],"
         "\"tool_choice\":\"always\",\"tools\":[{\"name\":\"f\"}]}",
         1, "invalid_value"},
        {"{\"prompt\":{}}", 0, "invalid_value"},
        {"{\"prompt\":[-1]}", 0, "invalid_value"},
        {"{\"prompt\":\"a\",\"best_of\":2,\"stream\":true}", 0,
         "invalid_value"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const char *got = parse(cases[i].body, cases[i].chat, NULL);
        CHECK(got && !strcmp(got, cases[i].code), "case %zu: code %s, want %s",
              i, got ? got : "(accepted)", cases[i].code);
    }
}

static uint64_t rnd(uint64_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s;
}

/* Mangled requests: none may crash or leak, whatever it is answered. */
static void mangled(void)
{
    static const char *const seeds[] = {
        "{\"model\":\"m\",\"messages\":[{\"role\":\"system\",\"content\":"
        "\"s\"},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":"
        "\"hello\"}]}],\"stop\":[\"x\",\"yy\"],\"max_tokens\":5,"
        "\"stream\":true,\"temperature\":0.5}",
        "{\"prompt\":\"abc\\u00e8\",\"max_tokens\":3,\"echo\":true,"
        "\"stop\":\"z\"}",
    };
    uint64_t s = 0x9e3779b97f4a7c15ull;
    unsigned accepted = 0, total = 0;
    char buf[1024];
    for (int round = 0; round < 20000; round++) {
        const char *src = seeds[round % 2];
        size_t n = strlen(src);
        memcpy(buf, src, n + 1);
        int edits = 1 + (int)(rnd(&s) % 4);
        for (int k = 0; k < edits && n > 0; k++) {
            size_t at = rnd(&s) % n;
            switch (rnd(&s) % 4) {
            case 0: /* a byte changed */
                buf[at] = (char)(rnd(&s) & 0xff);
                break;
            case 1: /* cut there */
                n = at;
                break;
            case 2: /* a byte repeated */
                if (n + 1 < sizeof(buf)) {
                    memmove(buf + at + 1, buf + at, n - at);
                    n++;
                }
                break;
            default: /* a structural character put in */
                buf[at] = "{}[]\",:\\0"[rnd(&s) % 9];
                break;
            }
        }
        buf[n] = 0;
        struct srv_job *j = srv_job_new();
        struct srv_parsed pr;
        struct srv_perr e;
        /* the length given, not strlen: a zero byte inside is data */
        accepted += srv_parse_request(buf, n, round % 2 == 0, j, &pr, &e) == 0;
        srv_parsed_free(&pr);
        total++;
        srv_job_release(j);
    }
    printf("test_server_request: %u of %u mangled requests accepted\n",
           accepted, total);
}

int main(void)
{
    valid();
    invalid();
    mangled();
    printf("test_server_request: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
