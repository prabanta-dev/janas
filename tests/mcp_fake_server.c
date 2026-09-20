/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_fake_server.c - an MCP server over stdio for the tests, so that the
 * client can be tried without fetching anybody's server.
 *
 *   mcp_fake_server [modern|legacy|dual|mute] [--http]
 *
 * modern: revision 2026-07-28 only (server/discover, _meta on every
 * request, no initialize); legacy: 2025-11-25, the handshake first, and an
 * error to anything before it; dual: both; mute: legacy, and silent to
 * whatever comes before initialize, which some old servers are.
 *
 * With --http it listens on a port of 127.0.0.1 instead, which it writes on
 * its standard output ("PORT n"), and speaks Streamable HTTP: a modern
 * server checks the headers against the body and answers tools/call as an
 * event stream in chunks; a legacy one gives a session at initialize and
 * wants it back on every request.
 *
 * Tools, listed two to a page: echo {text}, add {a, b}, fail (a tool
 * error), picture (an image and a link), crash (the server exits), slow
 * (answers after three seconds), whoami (says the Mcp-Name and
 * Mcp-Param-Region headers it got), badmark (an x-mcp-header where none may
 * be, so a client over HTTP must leave it out). Before answering a call a
 * legacy server pings the client and sends a notification, as real ones do.
 */
#define _GNU_SOURCE /* open_memstream, strcasestr */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/mcp_rpc.h"

enum { MODERN = 1, LEGACY = 2 };
static int eras = MODERN;
static int mute;
static int initialized;
static FILE *sink; /* where the answers go: stdout, or an HTTP body */
static int http, http_status;
static char seen_name[256], seen_param[256];

static const char *const TOOLS[] = {
    "{\"name\":\"echo\",\"description\":\"Says the text back\","
    "\"inputSchema\":{\"type\":\"object\",\"properties\":{\"text\":"
    "{\"type\":\"string\"}},\"required\":[\"text\"]}}",
    "{\"name\":\"add\",\"title\":\"Adds two numbers\",\"inputSchema\":"
    "{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\","
    "\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\"},"
    "\"b\":{\"type\":\"number\"}},\"required\":[\"a\",\"b\"],"
    "\"additionalProperties\":false}}",
    "{\"name\":\"fail\",\"description\":\"Always fails\",\"inputSchema\":"
    "{\"type\":\"object\",\"additionalProperties\":false}}",
    "{\"name\":\"picture\",\"description\":\"A picture\"}",
    "{\"name\":\"crash\",\"description\":\"Ends the server\"}",
    "{\"name\":\"slow\",\"description\":\"Takes three seconds\"}",
    "{\"name\":\"bad name\",\"description\":\"cannot be called\"}",
    "{\"name\":\"whoami\",\"inputSchema\":{\"type\":\"object\","
    "\"properties\":{\"region\":{\"type\":\"string\",\"x-mcp-header\":"
    "\"Region\"}}}}",
    "{\"name\":\"badmark\",\"inputSchema\":{\"type\":\"object\","
    "\"properties\":{\"list\":{\"type\":\"array\",\"items\":{\"type\":"
    "\"string\",\"x-mcp-header\":\"L\"}}}}}",
};
#define N_TOOLS (sizeof(TOOLS) / sizeof(TOOLS[0]))

static void out(struct janas_buf *b)
{
    fwrite(b->p, 1, b->n, sink);
    fflush(sink);
    janas_buf_free(b);
}

static void error(const struct janas_json *id, int code, const char *msg,
                  const char *data)
{
    struct janas_buf b = {0};
    janas_buf_puts(&b, "{\"jsonrpc\":\"2.0\",\"id\":");
    janas_json_write(&b, id);
    janas_buf_printf(&b, ",\"error\":{\"code\":%d,\"message\":\"%s\"%s%s}}\n",
                     code, msg, data ? ",\"data\":" : "", data ? data : "");
    out(&b);
}

/* result: the members of the result object, without braces */
static void result(const struct janas_json *id, const char *members)
{
    struct janas_buf b = {0};
    janas_buf_puts(&b, "{\"jsonrpc\":\"2.0\",\"id\":");
    janas_json_write(&b, id);
    janas_buf_printf(&b, ",\"result\":{%s%s%s}}\n",
                     eras & MODERN && !initialized ? "\"resultType\":"
                                                     "\"complete\""
                                                   : "",
                     eras & MODERN && !initialized && *members ? "," : "",
                     members);
    out(&b);
}

static void list(const struct janas_json *id, const struct janas_json *params)
{
    const char *cursor = janas_json_str(janas_json_get(params, "cursor"));
    size_t from = cursor ? (size_t)atoi(cursor) : 0;
    struct janas_buf b = {0};
    janas_buf_puts(&b, "\"tools\":[");
    for (size_t i = from; i < from + 2 && i < N_TOOLS; i++)
        janas_buf_printf(&b, "%s%s", i > from ? "," : "", TOOLS[i]);
    janas_buf_puts(&b, "]");
    if (!initialized) /* modern: the caching hints the revision wants */
        janas_buf_puts(&b, ",\"ttlMs\":60000,\"cacheScope\":\"public\"");
    if (from + 2 < N_TOOLS)
        janas_buf_printf(&b, ",\"nextCursor\":\"%zu\"", from + 2);
    janas_buf_put(&b, "", 1);
    result(id, b.p);
    janas_buf_free(&b);
}

static void text(const struct janas_json *id, const char *t, int is_error)
{
    struct janas_buf b = {0};
    janas_buf_puts(&b, "\"content\":[{\"type\":\"text\",\"text\":");
    janas_json_write_str(&b, t, strlen(t));
    janas_buf_printf(&b, "}],\"isError\":%s", is_error ? "true" : "false");
    janas_buf_put(&b, "", 1);
    result(id, b.p);
    janas_buf_free(&b);
}

static void call(const struct janas_json *id, const struct janas_json *params)
{
    const char *name = janas_json_str(janas_json_get(params, "name"));
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (initialized) { /* what old servers do */
        fprintf(sink,
                "{\"jsonrpc\":\"2.0\",\"id\":\"srv-1\",\"method\":\"ping\"}\n"
                "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\","
                "\"params\":{\"level\":\"info\",\"data\":\"calling\"}}\n"
                "this line is not a message\n");
        fflush(sink);
    }
    fprintf(stderr, "call %s\n", name ? name : "?"); /* for the tests */
    if (!name) {
        error(id, -32602, "no tool named", NULL);
    } else if (strcmp(name, "echo") == 0) {
        const char *t = janas_json_str(janas_json_get(args, "text"));
        text(id, t ? t : "", 0);
    } else if (strcmp(name, "add") == 0) {
        char s[64];
        snprintf(s, sizeof(s), "%g",
                 janas_json_num(janas_json_get(args, "a"), 0) +
                     janas_json_num(janas_json_get(args, "b"), 0));
        text(id, s, 0);
    } else if (strcmp(name, "fail") == 0) {
        text(id, "the disk is full", 1);
    } else if (strcmp(name, "picture") == 0) {
        result(id, "\"content\":[{\"type\":\"image\",\"data\":\"AAAAAAAA\","
                   "\"mimeType\":\"image/png\"},{\"type\":\"resource_link\","
                   "\"uri\":\"file:///x.txt\",\"name\":\"x.txt\"},{\"type\":"
                   "\"resource\",\"resource\":{\"uri\":\"file:///y.txt\","
                   "\"text\":\"why\"}}]");
    } else if (strcmp(name, "whoami") == 0) {
        char s[600];
        snprintf(s, sizeof(s), "name=%s param=%s", seen_name, seen_param);
        text(id, s, 0);
    } else if (strcmp(name, "badmark") == 0) {
        text(id, "badmark ran", 0);
    } else if (strcmp(name, "crash") == 0) {
        exit(3);
    } else if (strcmp(name, "slow") == 0) {
        sleep(3);
        text(id, "late", 0);
    } else {
        error(id, -32602, "Unknown tool", NULL);
    }
}

/* One message from the client, its answer to the sink. */
static void handle(const char *line, size_t n)
{
    struct janas_rpc_msg m;
    char err[128];
    if (janas_rpc_parse(line, n, &m, err, sizeof(err)) != 0) {
        fprintf(stderr, "bad line: %s\n", err);
        janas_rpc_free(&m);
        return;
    }
    if (m.kind != JANAS_RPC_REQUEST) {
        if (m.kind == JANAS_RPC_NOTIFY)
            fprintf(stderr, "notification %s\n", m.method);
        janas_rpc_free(&m);
        return;
    }
    const struct janas_json *meta = janas_json_get(m.params, "_meta");
    const char *version = janas_json_str(
        janas_json_get(meta, "io.modelcontextprotocol/protocolVersion"));
    if (strcmp(m.method, "initialize") == 0) {
        if (eras & LEGACY) {
            initialized = 1;
            result(m.id, "\"protocolVersion\":\"2025-11-25\","
                         "\"capabilities\":{\"tools\":{}},\"serverInfo\":"
                         "{\"name\":\"fake-legacy\",\"version\":\"1.0\"}");
        } else {
            error(m.id, -32601,
                  "initialize is gone: this server speaks 2026-07-28", NULL);
        }
    } else if (!initialized && !(eras & MODERN)) {
        if (!mute) /* a legacy server, asked before the handshake */
            error(m.id, -32600, "not initialized", NULL);
    } else if (!initialized && !version) {
        error(m.id, -32602, "no protocol version in _meta", NULL);
    } else if (!initialized && strcmp(version, "2026-07-28") != 0) {
        http_status = 400;
        error(m.id, -32022, "Unsupported protocol version",
              "{\"supported\":[\"2026-07-28\"]}");
    } else if (strcmp(m.method, "server/discover") == 0) {
        result(m.id, "\"supportedVersions\":[\"2026-07-28\"],"
                     "\"ttlMs\":60000,\"cacheScope\":\"public\","
                     "\"capabilities\":{\"tools\":{}},\"instructions\":"
                     "\"Use add for sums.\",\"_meta\":{\"io."
                     "modelcontextprotocol/serverInfo\":{\"name\":"
                     "\"fake-modern\",\"version\":\"2.0\"}}");
    } else if (strcmp(m.method, "tools/list") == 0) {
        list(m.id, m.params);
    } else if (strcmp(m.method, "tools/call") == 0) {
        call(m.id, m.params);
    } else {
        error(m.id, -32601, "Method not found", NULL);
    }
    janas_rpc_free(&m);
}

/* The value of header name in the head, into v ("" when absent). */
static void header(const char *head, const char *name, char *v, size_t cap)
{
    v[0] = 0;
    for (const char *l = strchr(head, '\n'); l; l = strchr(l + 1, '\n')) {
        size_t k = strlen(name);
        if (strncasecmp(l + 1, name, k) != 0 || l[1 + k] != ':')
            continue;
        const char *s = l + 2 + k;
        while (*s == ' ')
            s++;
        size_t e = strcspn(s, "\r\n");
        snprintf(v, cap, "%.*s", (int)(e < cap ? e : cap - 1), s);
        return;
    }
}

static void reply(int fd, int status, const char *type, const char *extra,
                  const char *body, size_t n)
{
    char head[512];
    int k = snprintf(head, sizeof(head),
                     "HTTP/1.1 %d X\r\nContent-Type: %s\r\n%sContent-Length: "
                     "%zu\r\nConnection: close\r\n\r\n",
                     status, type, extra, n);
    if (write(fd, head, (size_t)k) < 0 || write(fd, body, n) < 0)
        fprintf(stderr, "write failed\n");
}

/* An event stream in chunks: each line of the answers an event of its own,
   and a comment before them to be skipped. */
static void reply_sse(int fd, const char *body, size_t n)
{
    const char *head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                       "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                       "5\r\n: hi\n\r\n";
    struct janas_buf b = {0};
    janas_buf_puts(&b, head);
    for (const char *l = body; l < body + n;) {
        const char *nl = memchr(l, '\n', (size_t)(body + n - l));
        size_t ln = nl ? (size_t)(nl - l) : (size_t)(body + n - l);
        char ev[8192];
        int k = snprintf(ev, sizeof(ev), "event: message\ndata: %.*s\n\n",
                         (int)ln, l);
        /* each event split in two chunks, to try the reassembly */
        int half = k / 2;
        janas_buf_printf(&b, "%x\r\n%.*s\r\n%x\r\n%s\r\n", half, half, ev,
                         k - half, ev + half);
        l += ln + 1;
    }
    janas_buf_puts(&b, "0\r\n\r\n");
    if (write(fd, b.p, b.n) < 0)
        fprintf(stderr, "write failed\n");
    janas_buf_free(&b);
}

static void serve_http(void)
{
    int s = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET,
                            .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    socklen_t al = sizeof(a);
    if (s < 0 || bind(s, (struct sockaddr *)&a, sizeof(a)) != 0 ||
        listen(s, 8) != 0 || getsockname(s, (struct sockaddr *)&a, &al) != 0) {
        perror("fake server");
        exit(1);
    }
    printf("PORT %d\n", ntohs(a.sin_port));
    fflush(stdout);
    char session[64] = "";
    for (;;) {
        /* the end of standard input is the end of the server */
        struct pollfd pf[2] = {{.fd = s, .events = POLLIN},
                               {.fd = STDIN_FILENO, .events = POLLIN}};
        if (poll(pf, 2, -1) < 0)
            continue;
        if (pf[1].revents) {
            char c;
            if (read(STDIN_FILENO, &c, 1) <= 0)
                exit(0);
        }
        if (!(pf[0].revents & POLLIN))
            continue;
        int fd = accept(s, NULL, NULL);
        if (fd < 0)
            continue;
        struct janas_buf in = {0};
        char chunk[4096];
        long got;
        size_t head_end = 0, want = 0;
        while ((got = read(fd, chunk, sizeof(chunk))) > 0) {
            janas_buf_put(&in, chunk, (size_t)got);
            janas_buf_put(&in, "", 1);
            in.n--;
            char *e = strstr(in.p, "\r\n\r\n");
            if (e && !head_end) {
                head_end = (size_t)(e - in.p) + 4;
                char cl[32];
                header(in.p, "Content-Length", cl, sizeof(cl));
                want = (size_t)atol(cl);
            }
            if (head_end && in.n >= head_end + want)
                break;
        }
        if (!head_end) {
            close(fd);
            janas_buf_free(&in);
            continue;
        }
        in.p[head_end - 2] = 0; /* the head as a string */
        const char *body = in.p + head_end;
        char version[64], method[64], sid[64];
        header(in.p, "MCP-Protocol-Version", version, sizeof(version));
        header(in.p, "Mcp-Method", method, sizeof(method));
        header(in.p, "Mcp-Session-Id", sid, sizeof(sid));
        header(in.p, "Mcp-Name", seen_name, sizeof(seen_name));
        header(in.p, "Mcp-Param-Region", seen_param, sizeof(seen_param));
        if (strncmp(in.p, "DELETE", 6) == 0) {
            fprintf(stderr, "session %s ended\n", sid);
            if (!strcmp(sid, session))
                session[0] = 0;
            reply(fd, 200, "text/plain", "", "", 0);
        } else if (!(eras & MODERN) && session[0] &&
                   strcmp(sid, session) != 0 &&
                   !strstr(body, "\"initialize\"")) {
            const char *e = "{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":"
                            "{\"code\":-32000,\"message\":\"Bad Request: no "
                            "valid session\"}}";
            reply(fd, 400, "application/json", "", e, strlen(e));
        } else if (!(eras & MODERN) && !session[0] &&
                   !strstr(body, "\"initialize\"")) {
            reply(fd, 400, "text/html", "", "<h1>no</h1>", 11);
        } else {
            char *out = NULL;
            size_t on = 0;
            sink = open_memstream(&out, &on);
            http_status = 200;
            if ((eras & MODERN) && !initialized && strstr(body, "_meta") &&
                (!strstr(body, version) || !version[0] ||
                 !strstr(body, method) || !method[0])) {
                fprintf(sink, "{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":"
                              "{\"code\":-32020,\"message\":\"Header "
                              "mismatch\"}}\n");
                http_status = 400;
            } else {
                handle(body, strlen(body));
            }
            fclose(sink);
            char extra[128] = "";
            if (initialized && !(eras & MODERN) &&
                strstr(body, "\"initialize\"")) { /* a new one each time */
                static int n_sessions;
                snprintf(session, sizeof(session), "sess-%d-%d", (int)getpid(),
                         ++n_sessions);
                snprintf(extra, sizeof(extra), "Mcp-Session-Id: %s\r\n",
                         session);
            }
            if (on == 0)
                reply(fd, 202, "text/plain", "", "", 0);
            else if (strstr(body, "tools/call") && http_status == 200)
                reply_sse(fd, out, on);
            else
                reply(fd, http_status, "application/json", extra, out,
                      strcspn(out, "\n"));
            free(out);
        }
        close(fd);
        janas_buf_free(&in);
    }
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "modern";
    if (strcmp(mode, "legacy") == 0)
        eras = LEGACY;
    else if (strcmp(mode, "dual") == 0)
        eras = MODERN | LEGACY;
    else if (strcmp(mode, "mute") == 0)
        eras = LEGACY, mute = 1;
    http = argc > 2 && strcmp(argv[2], "--http") == 0;
    fprintf(stderr, "fake server, %s%s\n", mode, http ? " over HTTP" : "");
    sink = stdout;
    if (http)
        serve_http();
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    while ((n = getline(&line, &cap, stdin)) > 0)
        handle(line, (size_t)n);
    free(line);
    return 0;
}
