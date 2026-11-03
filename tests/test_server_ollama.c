/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_server_ollama.c - Ollama's requests to janas-server's /api/chat and
 * /api/generate as the job they become (src/server/ollama_req.c, then
 * request.c): the messages with their pictures and tool calls, think,
 * format, the options; and mangled requests - bytes flipped, cut, repeated,
 * from a fixed seed - which must neither crash nor leak (run it under asan
 * to know the second).
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
#include "server/ollama_req.c"
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

/* An Ollama request (body) through ollama_req.c and request.c, as the
   handlers take it: chat_route for /api/chat, chat for a chat job. The
   error message of a refused one, NULL when it was taken (*out). */
static const char *job_of(const char *body, size_t len, int chat_route,
                          int chat, struct srv_job **out)
{
    static char msg[600];
    yyjson_doc *doc = yyjson_read(body, len, 0);
    if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc))) {
        yyjson_doc_free(doc);
        return "not a JSON object";
    }
    size_t n = 0;
    char *q = ol_request_body(yyjson_doc_get_root(doc), chat_route, chat, &n);
    yyjson_doc_free(doc);
    if (!q)
        return "out of memory";
    struct srv_job *j = srv_job_new();
    struct srv_parsed pr;
    struct srv_perr e;
    int rc = srv_parse_request(q, n, chat, j, &pr, &e);
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

static const char chat_req[] =
    "{\"model\":\"m\",\"think\":false,\"format\":\"json\","
    "\"options\":{\"temperature\":3,\"top_k\":20,\"seed\":5,"
    "\"stop\":[\"END\"],\"num_ctx\":8192,\"repeat_penalty\":1.1},"
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":"
    "\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"city\":{\"type\":\"string\"}}}}}],"
    "\"messages\":["
    "{\"role\":\"system\",\"content\":\"Be brief.\"},"
    "{\"role\":\"user\",\"content\":\"Weather here?\","
    "\"images\":[\"iVBORw0KGgo=\",\"/9j/4AAQ\"]},"
    "{\"role\":\"assistant\",\"content\":\"\",\"thinking\":\"hm\","
    "\"tool_calls\":[{\"function\":{\"name\":\"get_weather\","
    "\"arguments\":{\"city\":\"Toronto\"}}}]},"
    "{\"role\":\"tool\",\"content\":\"11 degrees\","
    "\"tool_name\":\"get_weather\"}]}";

static void chat(void)
{
    struct srv_job *j = NULL;
    const char *err = job_of(chat_req, sizeof chat_req - 1, 1, 1, &j);
    CHECK(!err, "a chat request refused: %s", err);
    if (!j)
        return;
    CHECK(j->n_msg == 4, "messages: %d, want 4", j->n_msg);
    CHECK(j->n_msg == 4 && j->msg[0].role == JANAS_LLM_ROLE_SYSTEM &&
              !strcmp(j->msg[0].text, "Be brief."),
          "the system message");
    if (j->n_msg == 4) {
        const struct srv_msg *u = &j->msg[1], *a = &j->msg[2], *t = &j->msg[3];
        CHECK(u->role == JANAS_LLM_ROLE_USER &&
                  !strcmp(u->text, "Weather here?") && u->n_imgs == 2,
              "the user's text and pictures: '%s', %d pictures", u->text,
              u->n_imgs);
        CHECK(u->n_imgs == 2 &&
                  !strcmp(u->imgs[0].ref, "data:image/png;base64,"
                                          "iVBORw0KGgo=") &&
                  !strcmp(u->imgs[1].ref, "data:image/png;base64,/9j/4AAQ"),
              "the pictures as data: URLs, in their order");
        CHECK(a->role == JANAS_LLM_ROLE_ASSISTANT && a->n_calls == 1 &&
                  !strcmp(a->call_name[0], "get_weather") &&
                  !strcmp(a->call_args[0], "{\"city\":\"Toronto\"}"),
              "the assistant's call, its arguments an object");
        CHECK(t->role == JANAS_LLM_ROLE_TOOL && !strcmp(t->text, "11 degrees"),
              "the tool's result");
    }
    CHECK(j->p.temperature > 1.99f && j->p.temperature < 2.01f,
          "temperature 3 is not OpenAI's highest, 2: %g",
          (double)j->p.temperature);
    CHECK(j->p.top_k == 20 && j->p.seed == 5 && j->seeded,
          "top_k and seed not taken");
    CHECK(j->n_stop == 1 && !strcmp(j->stop[0], "END"), "stop strings");
    CHECK(j->p.thinking == 0, "think false: thinking %d", j->p.thinking);
    CHECK(j->format == JANAS_LLM_FORMAT_JSON, "format json: %d", j->format);
    CHECK(j->tools && strstr(j->tools, "get_weather"), "the tools");
    srv_job_release(j);

    /* a schema, and think as a level */
    j = NULL;
    static const char schema[] =
        "{\"think\":\"high\",\"format\":{\"type\":\"object\",\"properties\":"
        "{\"capital\":{\"type\":\"string\"}}},"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Capital?\"}]}";
    err = job_of(schema, sizeof schema - 1, 1, 1, &j);
    CHECK(!err, "a schema refused: %s", err);
    if (j) {
        CHECK(j->format == JANAS_LLM_FORMAT_SCHEMA && j->schema &&
                  strstr(j->schema, "capital"),
              "the schema");
        CHECK(j->p.thinking == 1, "think \"high\": thinking %d", j->p.thinking);
        srv_job_release(j);
    }

    /* without messages request.c says so */
    static const char none[] = "{\"model\":\"m\"}";
    err = job_of(none, sizeof none - 1, 1, 1, NULL);
    CHECK(err && strstr(err, "messages"), "no messages taken: %s",
          err ? err : "(taken)");
}

static void generate(void)
{
    struct srv_job *j = NULL;
    static const char g[] = "{\"model\":\"m\",\"system\":\"Be brief.\","
                            "\"prompt\":\"What is this?\","
                            "\"images\":[\"iVBORw0KGgo=\"]}";
    const char *err = job_of(g, sizeof g - 1, 0, 1, &j);
    CHECK(!err, "a prompt refused: %s", err);
    if (j) {
        CHECK(j->kind == JOB_CHAT && j->n_msg == 2 &&
                  j->msg[0].role == JANAS_LLM_ROLE_SYSTEM &&
                  j->msg[1].role == JANAS_LLM_ROLE_USER &&
                  !strcmp(j->msg[1].text, "What is this?") &&
                  j->msg[1].n_imgs == 1,
              "the prompt as a system and a user message with its picture");
        srv_job_release(j);
    }

    j = NULL;
    static const char raw[] = "{\"prompt\":\"def add(a, b):\","
                              "\"suffix\":\"\\nprint(add(1, 2))\","
                              "\"raw\":true,\"options\":{\"top_p\":0.5}}";
    err = job_of(raw, sizeof raw - 1, 0, 0, &j);
    CHECK(!err, "a raw prompt refused: %s", err);
    if (j) {
        CHECK(j->kind == JOB_TEXT && j->n_prompts == 1 &&
                  !strcmp(j->prompts[0].text, "def add(a, b):") && j->suffix &&
                  !strcmp(j->suffix, "\nprint(add(1, 2))"),
              "the raw prompt and its suffix");
        CHECK(j->p.top_p > 0.49f && j->p.top_p < 0.51f, "top_p not taken");
        srv_job_release(j);
    }
}

static uint64_t rng = 0x5eed0011a3aULL;
static int n_json, n_taken;

static uint32_t next(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

/* Requests mangled: whatever comes out must be taken or refused, never
   crash; a mangled request that is still JSON reaches request.c. */
static void mangled(void)
{
    static const char *const base[] = {
        chat_req,
        "{\"prompt\":\"x\",\"suffix\":\"y\",\"options\":{\"num_predict\":3}}",
        "{\"system\":\"s\",\"prompt\":\"p\",\"images\":[\"AAAA\"],\"think\":"
        "true}",
    };
    int taken = 0, json = 0;
    for (int it = 0; it < 20000; it++) {
        int b = it % 3;
        const char *src = base[b];
        size_t n = strlen(src);
        size_t cap = 4 * n + 2;
        char *s = malloc(cap);
        memcpy(s, src, n);
        int edits = 1 + (int)(next() % 4);
        for (int e = 0; e < edits; e++) {
            size_t at = n ? next() % n : 0;
            switch (next() % 3) {
            case 0: /* a byte changed */
                s[at] = (char)" {}[]\":,0a\\tf"[next() % 13];
                break;
            case 1: /* cut */
                n = at;
                break;
            default: /* a piece repeated */
                if (n + (n - at) / 2 <= cap) {
                    size_t k = (n - at) / 2;
                    memmove(s + at + k, s + at, n - at);
                    n += k;
                }
            }
        }
        yyjson_doc *doc = yyjson_read(s, n, 0);
        if (doc && yyjson_is_obj(yyjson_doc_get_root(doc)))
            json++;
        yyjson_doc_free(doc);
        struct srv_job *j = NULL;
        if (!job_of(s, n, b == 0, b != 1, &j)) {
            taken++;
            srv_job_release(j);
        }
        free(s);
    }
    n_json = json;
    n_taken = taken;
}

int main(void)
{
    chat();
    generate();
    mangled();
    if (failures) {
        printf("test_server_ollama: %d failures\n", failures);
        return 1;
    }
    printf("test_server_ollama: %d of 20000 mangled requests still JSON, "
           "%d taken, ok\n",
           n_json, n_taken);
    return 0;
}
