/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_mcp.c - the MCP client (include/janas/mcp.h, src/common/mcp_*).
 *
 * What a server writes and what a configuration file holds come from
 * outside, so their readers are fuzzed here: lines and files mutated byte
 * by byte must neither crash nor leak (run it under asan and ubsan). Then
 * the client against mcp_fake_server, the program next to this one, in
 * each era: the probe, the handshake, pages of tools, calls, a tool's
 * failure, a server that dies and is started again, one that is too slow.
 */
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/mcp_config.h"
#include "common/mcp_http.h"
#include "common/mcp_stdio.h"
#include "common/mcp_rpc.h"
#include "janas/mcp.h"
#include "llm/tools.h"

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

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

static const char *const LINES[] = {
    "{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"resultType\":\"complete\","
    "\"tools\":[{\"name\":\"a\",\"inputSchema\":{\"type\":\"object\"}}]}}",
    "{\"jsonrpc\":\"2.0\",\"id\":8,\"error\":{\"code\":-32022,\"message\":"
    "\"Unsupported\",\"data\":{\"supported\":[\"2026-07-28\"]}}}",
    "{\"jsonrpc\":\"2.0\",\"id\":\"s1\",\"method\":\"ping\"}",
    "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":"
    "{\"progress\":1,\"total\":2}}",
};

static void test_rpc(void)
{
    struct janas_rpc_msg m;
    char err[128];
    CHECK(janas_rpc_parse(LINES[0], strlen(LINES[0]), &m, err, sizeof(err)) ==
                  0 &&
              m.kind == JANAS_RPC_RESULT && m.has_num_id && m.num_id == 7,
          "result");
    janas_rpc_free(&m);
    CHECK(janas_rpc_parse(LINES[1], strlen(LINES[1]), &m, err, sizeof(err)) ==
                  0 &&
              m.kind == JANAS_RPC_ERROR && m.code == -32022 && m.data,
          "error");
    janas_rpc_free(&m);
    CHECK(janas_rpc_parse(LINES[2], strlen(LINES[2]), &m, err, sizeof(err)) ==
                  0 &&
              m.kind == JANAS_RPC_REQUEST && !m.has_num_id &&
              strcmp(m.method, "ping") == 0,
          "request");
    struct janas_buf b = {0};
    janas_rpc_answer(&b, m.id, 0, NULL);
    CHECK(b.n && strcmp(b.p + b.n - 1, "\n") == 0 &&
              strstr(b.p, "\"id\":\"s1\""),
          "answer: %.*s", (int)b.n, b.p);
    janas_buf_free(&b);
    janas_rpc_free(&m);
    CHECK(janas_rpc_parse(LINES[3], strlen(LINES[3]), &m, err, sizeof(err)) ==
                  0 &&
              m.kind == JANAS_RPC_NOTIFY,
          "notification");
    janas_rpc_free(&m);
    const char *bad[] = {"",
                         "[]",
                         "{}",
                         "{\"jsonrpc\":\"1.0\",\"id\":1}",
                         "{\"jsonrpc\":\"2.0\",\"result\":{}}",
                         "{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{}}",
                         "{\"jsonrpc\":\"2.0\",\"id\":[1],\"result\":{}}",
                         "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":3}"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        CHECK(janas_rpc_parse(bad[i], strlen(bad[i]), &m, err, sizeof(err)) !=
                      0 &&
                  m.kind == JANAS_RPC_INVALID,
              "bad %zu accepted", i);
        janas_rpc_free(&m);
    }
    /* mutations: nothing may crash, and what is taken must be consistent */
    char line[512];
    for (int it = 0; it < 20000; it++) {
        const char *src = LINES[rnd() % (sizeof(LINES) / sizeof(*LINES))];
        size_t n = strlen(src);
        memcpy(line, src, n);
        for (int k = 1 + (int)(rnd() % 4); k > 0; k--) {
            size_t at = rnd() % n;
            switch (rnd() % 3) {
            case 0:
                line[at] = (char)rnd();
                break;
            case 1:
                n = at + 1; /* cut */
                break;
            default:
                line[at] = "{}[]\",:0\\"[rnd() % 9];
            }
        }
        if (janas_rpc_parse(line, n, &m, err, sizeof(err)) == 0) {
            CHECK(m.kind != JANAS_RPC_INVALID, "kind");
            CHECK(m.kind != JANAS_RPC_RESULT || m.result, "result");
            CHECK((m.kind != JANAS_RPC_REQUEST && m.kind != JANAS_RPC_NOTIFY) ||
                      m.method,
                  "method");
        }
        janas_rpc_free(&m);
    }
}

static const char *CONFIG =
    "{\"mcpServers\": {\"files\": {\"command\": \"npx\", \"args\": [\"-y\", "
    "\"@modelcontextprotocol/server-filesystem\", \"/tmp/x\"], \"env\": "
    "{\"A\": \"1\", \"B\": \"two\"}}, \"web\": {\"type\": \"http\", \"url\": "
    "\"https://example.com/mcp\"}, \"off\": {\"command\": \"x\", "
    "\"disabled\": true}}}";

static void test_config(void)
{
    struct janas_mcp_conf *c;
    size_t n;
    char err[200];
    int rc = janas_mcp_config_parse(CONFIG, strlen(CONFIG), &c, &n, err,
                                    sizeof(err));
    CHECK(rc == 0 && n == 3, "config: %s", err);
    if (rc == 0 && n == 3) {
        CHECK(strcmp(c[0].name, "files") == 0 &&
                  strcmp(c[0].argv[0], "npx") == 0 &&
                  strcmp(c[0].argv[3], "/tmp/x") == 0 && !c[0].argv[4] &&
                  strcmp(c[0].env[1], "B=two") == 0 && !c[0].env[2],
              "files");
        CHECK(c[1].url && !c[1].argv, "web");
        CHECK(c[2].disabled, "off");
    }
    janas_mcp_config_free(c, n);
    const char *vs = "{\"servers\": {\"s\": {\"type\": \"stdio\", "
                     "\"command\": \"srv\"}}}";
    rc = janas_mcp_config_parse(vs, strlen(vs), &c, &n, err, sizeof(err));
    CHECK(rc == 0 && n == 1 && c[0].argv && !c[0].argv[1], "servers: %s", err);
    janas_mcp_config_free(c, n);
    const char *bad[] = {"{}",
                         "{\"mcpServers\": []}",
                         "{\"mcpServers\": {\"a\": {}}}",
                         "{\"mcpServers\": {\"a\": {\"command\": 1}}}",
                         "{\"mcpServers\": {\"a\": {\"command\": \"x\", "
                         "\"args\": [1]}}}",
                         "{\"mcpServers\": {\"a\": {\"command\": \"x\", "
                         "\"env\": {\"A=B\": \"c\"}}}}"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        CHECK(janas_mcp_config_parse(bad[i], strlen(bad[i]), &c, &n, err,
                                     sizeof(err)) != 0,
              "bad config %zu accepted", i);
    }
    char text[1024];
    size_t len = strlen(CONFIG);
    for (int it = 0; it < 20000; it++) {
        memcpy(text, CONFIG, len);
        size_t m = len;
        for (int k = 1 + (int)(rnd() % 3); k > 0; k--) {
            size_t at = rnd() % m;
            if (rnd() % 4 == 0)
                m = at + 1;
            else
                text[at] = rnd() % 2 ? (char)rnd() : "{}[]\",:\\"[rnd() % 8];
        }
        if (janas_mcp_config_parse(text, m, &c, &n, err, sizeof(err)) == 0) {
            for (size_t i = 0; i < n; i++)
                CHECK(c[i].name &&
                          (c[i].url || (c[i].argv && c[i].argv[0] && c[i].env)),
                      "config entry");
            janas_mcp_config_free(c, n);
        }
    }
}

static char fake[1024];

static janas_mcp *open_fake(const char *mode, int seconds)
{
    const char *argv[] = {fake, mode};
    janas_mcp *m = NULL;
    int32_t rc = janas_mcp_open_command("fake", 2, argv, 0, NULL, seconds, &m);
    CHECK(rc == JANAS_MCP_OK, "open %s: %s", mode, janas_mcp_last_error());
    return m;
}

static void check_info(janas_mcp *m, const char *protocol, const char *name)
{
    char info[512];
    int32_t len;
    CHECK(janas_mcp_info(m, info, sizeof(info), &len) == JANAS_MCP_OK &&
              strstr(info, protocol) && strstr(info, name),
          "info: %s", info);
}

static void check_calls(janas_mcp *m)
{
    char buf[512];
    int32_t len;
    /* "bad name" cannot be called, and over HTTP neither can badmark */
    CHECK(janas_mcp_tool_count(m) == 8, "tools: %d", janas_mcp_tool_count(m));
    char tools[4096];
    CHECK(janas_mcp_tools(m, "fake__", -1, tools, sizeof(tools), &len) ==
              JANAS_MCP_OK,
          "tools json");
    char err[128];
    struct janas_toolset *t =
        janas_toolset_parse(tools, (size_t)len, err, sizeof(err));
    CHECK(t && janas_toolset_find(t, "fake__add", 9) >= 0,
          "the model's tools: %s", t ? "" : err);
    janas_toolset_free(t);
    CHECK(janas_mcp_call(m, "add", -1, "{\"a\": 2,\n \"b\": 3}", -1, 10) ==
                  JANAS_MCP_OK &&
              janas_mcp_result(m, buf, sizeof(buf), &len) == JANAS_MCP_OK &&
              strcmp(buf, "5") == 0 && !janas_mcp_result_error(m),
          "add: %s %s", buf, janas_mcp_last_error());
    CHECK(janas_mcp_call(m, "echo", -1, "{\"text\": \"a\\nb \\u00e8\"}", -1,
                         10) == JANAS_MCP_OK &&
              janas_mcp_result(m, buf, sizeof(buf), &len) == JANAS_MCP_OK &&
              strcmp(buf, "a\nb \xc3\xa8") == 0,
          "echo: %s", buf);
    CHECK(janas_mcp_call(m, "fail", -1, NULL, 0, 10) == JANAS_MCP_OK &&
              janas_mcp_result_error(m) &&
              janas_mcp_result(m, buf, sizeof(buf), &len) == JANAS_MCP_OK &&
              strstr(buf, "disk"),
          "fail: %s", buf);
    CHECK(janas_mcp_call(m, "picture", -1, "{}", -1, 10) == JANAS_MCP_OK &&
              janas_mcp_result(m, buf, sizeof(buf), &len) == JANAS_MCP_OK &&
              strstr(buf, "[image image/png, 6 bytes") &&
              strstr(buf, "file:///x.txt") && strstr(buf, "why"),
          "picture: %s", buf);
    CHECK(janas_mcp_call(m, "nothing", -1, "{}", -1, 10) == JANAS_MCP_EFAIL,
          "unknown tool");
    CHECK(janas_mcp_call(m, "add", -1, "[1]", -1, 10) == JANAS_MCP_EINVAL,
          "arguments not an object");
    CHECK(janas_mcp_call(m, "crash", -1, "{}", -1, 10) == JANAS_MCP_EFAIL,
          "crash");
    CHECK(janas_mcp_call(m, "add", -1, "{\"a\": 1, \"b\": 1}", -1, 10) ==
                  JANAS_MCP_OK &&
              janas_mcp_result(m, buf, sizeof(buf), &len) == JANAS_MCP_OK &&
              strcmp(buf, "2") == 0,
          "after a restart: %s %s", buf, janas_mcp_last_error());
}

static void test_servers(void)
{
    janas_mcp *m = open_fake("modern", 10);
    if (m) {
        check_info(m, "protocol: 2026-07-28", "fake-modern");
        check_calls(m);
        /* too slow: given up, and the next call still answered */
        char buf[64];
        int32_t len;
        CHECK(janas_mcp_call(m, "slow", -1, "{}", -1, 1) == JANAS_MCP_EFAIL,
              "slow");
        CHECK(janas_mcp_call(m, "echo", -1, "{\"text\":\"x\"}", -1, 10) ==
                      JANAS_MCP_OK &&
                  janas_mcp_result(m, buf, sizeof(buf), &len) == JANAS_MCP_OK &&
                  strcmp(buf, "x") == 0,
              "after slow: %s", buf);
        janas_mcp_close(m);
    }
    m = open_fake("legacy", 10);
    if (m) {
        check_info(m, "protocol: 2025-11-25", "fake-legacy");
        check_calls(m);
        janas_mcp_close(m);
    }
    m = open_fake("dual", 10);
    if (m) {
        check_info(m, "protocol: 2026-07-28", "fake-modern");
        janas_mcp_close(m);
    }
    m = open_fake("mute", 2); /* the probe waits two seconds, no more */
    if (m) {
        check_info(m, "protocol: 2025-11-25", "fake-legacy");
        janas_mcp_close(m);
    }
    const char *none[] = {"/nonexistent/janas-mcp-server"};
    CHECK(janas_mcp_open_command("x", 1, none, 0, NULL, 2, &m) ==
              JANAS_MCP_EOPEN,
          "a server that is not there");
}

static void test_http_parsers(void)
{
    struct janas_url u;
    char err[200];
    CHECK(janas_url_parse("http://127.0.0.1:8080/mcp?x=1#f", &u, err,
                          sizeof(err)) == 0 &&
              !u.tls && strcmp(u.host, "127.0.0.1") == 0 &&
              strcmp(u.port, "8080") == 0 && strcmp(u.path, "/mcp?x=1") == 0,
          "url 1: %s", err);
    CHECK(janas_url_parse("https://user:pw@[::1]/", &u, err, sizeof(err)) ==
                  0 &&
              u.tls && strcmp(u.host, "::1") == 0 &&
              strcmp(u.port, "443") == 0 && strcmp(u.path, "/") == 0,
          "url 2: %s", err);
    CHECK(janas_url_parse("https://example.com", &u, err, sizeof(err)) == 0 &&
              strcmp(u.path, "/") == 0,
          "url 3");
    const char *bad_urls[] = {"ftp://x/",     "http://",      "http://:80/",
                              "http://h:8x/", "http://[::1/", "http://h/a b"};
    for (size_t i = 0; i < sizeof(bad_urls) / sizeof(*bad_urls); i++)
        CHECK(janas_url_parse(bad_urls[i], &u, err, sizeof(err)) != 0,
              "bad url %s accepted", bad_urls[i]);

    const char *head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                       "transfer-encoding: Chunked\r\nMcp-Session-Id: s-1\r\n"
                       "\r\nbody";
    struct janas_http_head h;
    long hl = janas_http_head_parse(head, strlen(head), &h);
    CHECK(hl == (long)strlen(head) - 4 && h.status == 200 && h.chunked &&
              strcmp(h.session, "s-1") == 0 &&
              strncmp(h.content_type, "text/event-stream", 17) == 0,
          "head");
    CHECK(janas_http_head_parse(head, 20, &h) == 0, "head, not all there");
    CHECK(janas_http_head_parse("SMTP ready\r\n\r\n", 14, &h) < 0,
          "not a head");
    const char *moved = "HTTP/1.1 302 Found\r\nlocation: /api/x?a=1\r\n"
                        "Content-Length: 0\r\n\r\n";
    CHECK(janas_http_head_parse(moved, strlen(moved), &h) > 0 &&
              h.status == 302 && strcmp(h.location, "/api/x?a=1") == 0,
          "redirect");
    const char *sly = "HTTP/1.1 302 Found\r\nLocation: /a\x01b\r\n\r\n";
    CHECK(janas_http_head_parse(sly, strlen(sly), &h) > 0 && h.location[0] == 0,
          "a control in a redirect kept");

    struct janas_http_chunks c = {0};
    struct janas_buf out = {0};
    const char *ch = "4;x=y\r\nWiki\r\n5\r\npedia\r\n0\r\nT: v\r\n\r\n";
    for (size_t i = 0; i < strlen(ch); i++) /* a byte at a time */
        CHECK(janas_http_chunks_feed(&c, ch + i, 1, &out) == 0, "chunk");
    CHECK(c.done && out.n == 9 && memcmp(out.p, "Wikipedia", 9) == 0,
          "chunks: %.*s", (int)out.n, out.p);
    janas_buf_free(&out);

    struct janas_sse s = {0};
    struct janas_buf ev = {0};
    const char *st = ": comment\n\nevent: message\ndata: {\"a\":\r\ndata:1}\r\n"
                     "\r\nid: 3\n\ndata: x";
    janas_sse_add(&s, st, strlen(st));
    CHECK(janas_sse_next(&s, &ev) == 1 && ev.n == 8 &&
              memcmp(ev.p, "{\"a\":\n1}", 8) == 0,
          "sse: %.*s", (int)ev.n, ev.p);
    CHECK(janas_sse_next(&s, &ev) == 0, "sse: an event not complete");
    janas_sse_add(&s, "\n\n", 2);
    CHECK(janas_sse_next(&s, &ev) == 1 && ev.n == 1 && ev.p[0] == 'x',
          "sse: completed");
    janas_sse_free(&s);

    /* mutations of all three: no crash, no leak, nothing out of bounds */
    char buf[256];
    for (int it = 0; it < 20000; it++) {
        const char *src = it % 3 == 0 ? head : it % 3 == 1 ? ch : st;
        size_t n = strlen(src);
        memcpy(buf, src, n);
        for (int k = 1 + (int)(rnd() % 4); k > 0; k--) {
            size_t at = rnd() % n;
            if (rnd() % 5 == 0)
                n = at + 1;
            else
                buf[at] = rnd() % 2 ? (char)rnd() : "\r\n:0f;a "[rnd() % 8];
        }
        if (it % 3 == 0) {
            long r = janas_http_head_parse(buf, n, &h);
            CHECK(r <= (long)n, "head length");
        } else if (it % 3 == 1) {
            struct janas_http_chunks cc = {0};
            janas_http_chunks_feed(&cc, buf, n, &out);
            CHECK(out.n <= n, "chunks grew");
            out.n = 0;
        } else {
            struct janas_sse ss = {0};
            for (size_t i = 0; i < n; i += 7) {
                janas_sse_add(&ss, buf + i, n - i < 7 ? n - i : 7);
                while (janas_sse_next(&ss, &ev) == 1)
                    CHECK(ev.n <= n, "event grew");
            }
            janas_sse_free(&ss);
        }
    }
    janas_buf_free(&out);
    janas_buf_free(&ev);
}

/* The fake server over HTTP: started, and its port read from it. */
static int start_http(struct janas_mcp_proc *p, const char *mode, char *url,
                      size_t cap)
{
    char *argv[] = {fake, (char *)mode, "--http", NULL};
    char err[200];
    if (janas_mcp_proc_start(p, argv, NULL, NULL, err, sizeof(err)) != 0) {
        CHECK(0, "start %s: %s", mode, err);
        return -1;
    }
    const char *line;
    size_t n;
    if (janas_mcp_proc_line(p, 5000, &line, &n) != 1 || n < 6 ||
        strncmp(line, "PORT ", 5) != 0) {
        CHECK(0, "no port from the fake server");
        janas_mcp_proc_stop(p);
        return -1;
    }
    snprintf(url, cap, "http://127.0.0.1:%.*s/mcp", (int)(n - 5), line + 5);
    return 0;
}

static void test_http(void)
{
    struct janas_mcp_proc p;
    char url[128], buf[512];
    int32_t len;
    if (start_http(&p, "modern", url, sizeof(url)) == 0) {
        janas_mcp *m = NULL;
        const char *hd[] = {"Authorization: Bearer x"};
        int32_t rc = janas_mcp_open_url("web", url, 1, hd, 10, &m);
        CHECK(rc == JANAS_MCP_OK, "open %s: %s", url, janas_mcp_last_error());
        if (m) {
            check_info(m, "protocol: 2026-07-28", "fake-modern");
            CHECK(janas_mcp_tool_count(m) == 7, "http tools: %d",
                  janas_mcp_tool_count(m));
            CHECK(janas_mcp_call(m, "add", -1, "{\"a\": 2, \"b\": 3}", -1,
                                 10) == JANAS_MCP_OK &&
                      janas_mcp_result(m, buf, sizeof(buf), &len) ==
                          JANAS_MCP_OK &&
                      strcmp(buf, "5") == 0,
                  "http add: %s %s", buf, janas_mcp_last_error());
            CHECK(janas_mcp_call(m, "whoami", -1,
                                 "{\"region\": \"eu west \\u00e8\"}", -1,
                                 10) == JANAS_MCP_OK &&
                      janas_mcp_result(m, buf, sizeof(buf), &len) ==
                          JANAS_MCP_OK &&
                      strcmp(buf, "name=whoami param==?base64?"
                                  "ZXUgd2VzdCDDqA==?=") == 0,
                  "http headers: %s", buf);
            CHECK(janas_mcp_call(m, "whoami", -1, "{\"region\": \"us-1\"}", -1,
                                 10) == JANAS_MCP_OK &&
                      janas_mcp_result(m, buf, sizeof(buf), &len) ==
                          JANAS_MCP_OK &&
                      strcmp(buf, "name=whoami param=us-1") == 0,
                  "http plain header: %s", buf);
            CHECK(janas_mcp_call(m, "fail", -1, "{}", -1, 10) == JANAS_MCP_OK &&
                      janas_mcp_result_error(m),
                  "http fail");
            CHECK(janas_mcp_call(m, "slow", -1, "{}", -1, 1) == JANAS_MCP_EFAIL,
                  "http slow");
            CHECK(janas_mcp_call(m, "echo", -1, "{\"text\": \"a\\nb\"}", -1,
                                 10) == JANAS_MCP_OK &&
                      janas_mcp_result(m, buf, sizeof(buf), &len) ==
                          JANAS_MCP_OK &&
                      strcmp(buf, "a\nb") == 0,
                  "http echo after slow: %s", buf);
            janas_mcp_close(m);
        }
        janas_mcp_proc_stop(&p);
    }
    if (start_http(&p, "legacy", url, sizeof(url)) == 0) {
        janas_mcp *m = NULL;
        int32_t rc = janas_mcp_open_url("old", url, 0, NULL, 10, &m);
        CHECK(rc == JANAS_MCP_OK, "open legacy %s: %s", url,
              janas_mcp_last_error());
        if (m) {
            check_info(m, "protocol: 2025-11-25", "fake-legacy");
            CHECK(janas_mcp_call(m, "add", -1, "{\"a\": 1, \"b\": 1}", -1,
                                 10) == JANAS_MCP_OK &&
                      janas_mcp_result(m, buf, sizeof(buf), &len) ==
                          JANAS_MCP_OK &&
                      strcmp(buf, "2") == 0,
                  "legacy http add: %s %s", buf, janas_mcp_last_error());
            janas_mcp_close(m);
        }
        janas_mcp_proc_stop(&p);
    }
    janas_mcp *m = NULL;
    CHECK(janas_mcp_open_url("none", "http://127.0.0.1:1/mcp", 0, NULL, 2,
                             &m) == JANAS_MCP_EFAIL,
          "nobody listening: %s", janas_mcp_last_error());
}

int main(int argc, char **argv)
{
    (void)argc;
    test_rpc();
    test_config();
    /* the fake server is next to this program, with the same suffix */
    char self[1024];
    snprintf(self, sizeof(self), "%s", argv[0]);
    const char *base = strrchr(argv[0], '/');
    base = base ? base + 1 : argv[0];
    const char *suffix = strchr(base, '-');
    snprintf(fake, sizeof(fake), "%s/mcp_fake_server%s", dirname(self),
             suffix ? suffix : "");
    test_servers();
    test_http_parsers();
    test_http();
    if (failures) {
        printf("test_mcp: %d failures\n", failures);
        return 1;
    }
    printf("test_mcp: ok\n");
    return 0;
}
