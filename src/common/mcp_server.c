/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_server.c - the server side of MCP over stdio (see mcp_server.h),
 * taken out of janas-mcp so that every program offering tools speaks it the
 * same way.
 */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common/mcp_rpc.h"
#include "common/mcp_server.h"

#define MODERN "2026-07-28"
#define CACHE_MS "3600000" /* the tools do not change while it runs */
static const char *const LEGACY[] = {"2025-11-25", "2025-06-18", "2025-03-26",
                                     "2024-11-05", NULL};

static void send_buf(struct janas_mcps *s, struct janas_buf *b)
{
    if (!b->oom) {
        fwrite(b->p, 1, b->n, s->out);
        fflush(s->out);
    }
    janas_buf_free(b);
}

static void result(struct janas_mcps *s, const struct janas_json *id,
                   int modern, const char *members, size_t n)
{
    struct janas_buf b = {0};
    janas_buf_puts(&b, "{\"jsonrpc\":\"2.0\",\"id\":");
    janas_json_write(&b, id);
    janas_buf_puts(&b, ",\"result\":{");
    /* the server's own _meta, and the tool's when it gave one (a layout,
       common/template.h): one object, a key may not come twice */
    struct janas_buf all = {0};
    janas_buf_puts(&all, "{");
    janas_buf_put(&all, members, n);
    janas_buf_puts(&all, "}");
    struct janas_json_doc *d = modern && n && !all.oom
                                   ? janas_json_parse(all.p, all.n, NULL, 0)
                                   : NULL;
    const struct janas_json *meta =
        d ? janas_json_get(janas_json_root(d), "_meta") : NULL;
    janas_buf_free(&all);
    if (modern) {
        janas_buf_puts(&b, "\"resultType\":\"complete\",\"_meta\":{\"io."
                           "modelcontextprotocol/serverInfo\":{\"name\":");
        janas_json_write_str(&b, s->name, strlen(s->name));
        janas_buf_puts(&b, ",\"version\":");
        janas_json_write_str(&b, s->version, strlen(s->version));
        janas_buf_puts(&b, "}");
        for (const struct janas_json *m =
                 meta && meta->type == JANAS_JSON_OBJECT ? meta->child : NULL;
             m; m = m->next) {
            janas_buf_puts(&b, ",");
            janas_json_write_str(&b, m->key, m->key_n);
            janas_buf_puts(&b, ":");
            janas_json_write(&b, m);
        }
        janas_buf_puts(&b, n ? "}," : "}");
    }
    if (meta) { /* the members but the _meta written above */
        int first = 1;
        for (const struct janas_json *m = janas_json_root(d)->child; m;
             m = m->next) {
            if (m == meta)
                continue;
            janas_buf_puts(&b, first ? "" : ",");
            janas_json_write_str(&b, m->key, m->key_n);
            janas_buf_puts(&b, ":");
            janas_json_write(&b, m);
            first = 0;
        }
    } else {
        janas_buf_put(&b, members, n);
    }
    janas_json_free(d);
    janas_buf_puts(&b, "}}\n");
    send_buf(s, &b);
}

static void error(struct janas_mcps *s, const struct janas_json *id, int code,
                  const char *message, const char *data)
{
    struct janas_buf b = {0};
    janas_buf_puts(&b, "{\"jsonrpc\":\"2.0\",\"id\":");
    if (id)
        janas_json_write(&b, id);
    else
        janas_buf_puts(&b, "null");
    janas_buf_printf(&b, ",\"error\":{\"code\":%d,\"message\":", code);
    janas_json_write_str(&b, message, strlen(message));
    if (data)
        janas_buf_printf(&b, ",\"data\":%s", data);
    janas_buf_puts(&b, "}}\n");
    send_buf(s, &b);
}

void janas_mcps_text_result(struct janas_buf *b, const char *t, size_t n,
                            int is_error)
{
    janas_buf_puts(b, "\"content\": [{\"type\": \"text\", \"text\": ");
    janas_json_write_str(b, t, n);
    janas_buf_printf(b, "}], \"isError\": %s", is_error ? "true" : "false");
}

void janas_mcps_split_result(struct janas_buf *b, const char *model,
                             size_t model_n, const char *user, size_t user_n)
{
    janas_buf_puts(b, "\"content\": [{\"type\": \"text\", \"text\": ");
    janas_json_write_str(b, model, model_n);
    janas_buf_puts(b, ", \"annotations\": {\"audience\": [\"assistant\"]}}, "
                      "{\"type\": \"text\", \"text\": ");
    janas_json_write_str(b, user, user_n);
    janas_buf_puts(b, ", \"annotations\": {\"audience\": [\"user\"]}}], "
                      "\"isError\": false");
}

/* Two ids the same: of the same type, and written the same. */
static int same_id(const struct janas_json *a, const struct janas_json *b)
{
    if (!a || !b || a->type != b->type)
        return 0;
    if (a->type == JANAS_JSON_NUMBER)
        return janas_json_num(a, 0) == janas_json_num(b, 1);
    return a->n == b->n && memcmp(a->s, b->s, a->n) == 0;
}

int janas_mcps_poll(struct janas_mcps *s)
{
    const char *line;
    size_t n;
    while (!s->cancelled && janas_mcp_proc_line(&s->in, 0, &line, &n) == 1) {
        struct janas_rpc_msg m;
        char err[128];
        if (janas_rpc_parse(line, n, &m, err, sizeof(err)) == 0) {
            if (m.kind == JANAS_RPC_NOTIFY &&
                strcmp(m.method, "notifications/cancelled") == 0 &&
                same_id(janas_json_get(m.params, "requestId"), s->running)) {
                s->cancelled = 1;
            } else if (m.kind == JANAS_RPC_REQUEST &&
                       strcmp(m.method, "ping") == 0) {
                result(s, m.id, janas_json_get(m.params, "_meta") != NULL, "",
                       0);
            } else if (m.kind == JANAS_RPC_REQUEST) { /* after the call */
                char **g = realloc(s->queue, (s->n_queue + 1) * sizeof(*g));
                char *copy = malloc(n + 1);
                if (g)
                    s->queue = g;
                if (g && copy) {
                    memcpy(copy, line, n);
                    copy[n] = 0;
                    s->queue[s->n_queue++] = copy;
                } else {
                    free(copy);
                    error(s, m.id, -32603, "out of memory", NULL);
                }
            }
        }
        janas_rpc_free(&m);
    }
    return s->cancelled;
}

static void discover(struct janas_mcps *s, const struct janas_json *id)
{
    struct janas_buf b = {0};
    janas_buf_puts(&b, "\"supportedVersions\":[\"" MODERN "\"");
    for (int i = 0; LEGACY[i]; i++)
        janas_buf_printf(&b, ",\"%s\"", LEGACY[i]);
    /* the caching hints every complete discover and list result carries
       from 2026-07-28 on: the same for everybody, and fixed while this
       process runs */
    janas_buf_puts(&b, "],\"ttlMs\":" CACHE_MS ",\"cacheScope\":\"public\","
                       "\"capabilities\":{\"tools\":{}},\"instructions\":");
    janas_json_write_str(&b, s->instructions, strlen(s->instructions));
    if (!b.oom)
        result(s, id, 1, b.p, b.n);
    janas_buf_free(&b);
}

static void initialize(struct janas_mcps *s, const struct janas_rpc_msg *m)
{
    const char *want =
        janas_json_str(janas_json_get(m->params, "protocolVersion"));
    const char *v = LEGACY[0];
    for (int i = 0; want && LEGACY[i]; i++)
        if (strcmp(want, LEGACY[i]) == 0)
            v = LEGACY[i];
    snprintf(s->legacy_version, sizeof(s->legacy_version), "%s", v);
    s->initialized = 1;
    struct janas_buf b = {0};
    janas_buf_printf(&b,
                     "\"protocolVersion\":\"%s\",\"capabilities\":{\"tools\":"
                     "{}},\"serverInfo\":{\"name\":",
                     v);
    janas_json_write_str(&b, s->name, strlen(s->name));
    janas_buf_puts(&b, ",\"version\":");
    janas_json_write_str(&b, s->version, strlen(s->version));
    janas_buf_puts(&b, "},\"instructions\":");
    janas_json_write_str(&b, s->instructions, strlen(s->instructions));
    if (!b.oom)
        result(s, m->id, 0, b.p, b.n);
    janas_buf_free(&b);
    fprintf(stderr, "%s: a client of revision %s\n", s->name, v);
}

static void request(struct janas_mcps *s, const struct janas_rpc_msg *m)
{
    const struct janas_json *meta = janas_json_get(m->params, "_meta");
    const char *version = janas_json_str(
        janas_json_get(meta, "io.modelcontextprotocol/protocolVersion"));
    int modern = version != NULL;
    if (strcmp(m->method, "ping") == 0) {
        result(s, m->id, modern, "", 0);
        return;
    }
    if (strcmp(m->method, "initialize") == 0) {
        initialize(s, m);
        return;
    }
    if (version && strcmp(version, MODERN) != 0) {
        struct janas_buf data = {0};
        janas_buf_puts(&data, "{\"supported\":[\"" MODERN "\"],\"requested\":");
        janas_json_write_str(&data, version,
                             strlen(version) < 64 ? strlen(version) : 64);
        janas_buf_puts(&data, "}");
        janas_buf_put(&data, "", 1);
        error(s, m->id, -32022, "Unsupported protocol version",
              data.oom ? NULL : data.p);
        janas_buf_free(&data);
        return;
    }
    if (!version && !s->initialized) {
        error(s, m->id, -32602,
              "no protocol version: put io.modelcontextprotocol/"
              "protocolVersion in _meta (" MODERN "), or send initialize first",
              NULL);
        return;
    }
    if (strcmp(m->method, "server/discover") == 0) {
        discover(s, m->id);
    } else if (strcmp(m->method, "tools/list") == 0) {
        struct janas_buf b = {0};
        s->tools_list(s->ctx, &b);
        if (modern)
            janas_buf_puts(&b, ",\"ttlMs\":" CACHE_MS ",\"cacheScope\":"
                               "\"public\"");
        if (!b.oom)
            result(s, m->id, modern, b.p, b.n);
        janas_buf_free(&b);
    } else if (strcmp(m->method, "tools/call") == 0) {
        struct janas_buf b = {0};
        s->running = m->id;
        s->cancelled = 0;
        int rc = s->tools_call(s->ctx, m->params, &b);
        s->running = NULL;
        if (rc < 0) {
            const char *t = janas_json_str(janas_json_get(m->params, "name"));
            char msg[200];
            snprintf(msg, sizeof(msg), "Unknown tool: %.100s", t ? t : "");
            error(s, m->id, -32602, msg, NULL);
        } else if (rc == 0 && !b.oom) {
            result(s, m->id, modern, b.p, b.n);
        } else if (rc == 0) {
            error(s, m->id, -32603, "out of memory", NULL);
        } /* cancelled: nothing more is said about it */
        janas_buf_free(&b);
    } else {
        error(s, m->id, -32601, "Method not found", NULL);
    }
}

static void take(struct janas_mcps *s, const char *line, size_t n)
{
    struct janas_rpc_msg m;
    char err[128];
    if (janas_rpc_parse(line, n, &m, err, sizeof(err)) != 0) {
        if (m.doc || n)
            error(s, NULL, m.doc ? -32600 : -32700,
                  m.doc ? "Invalid Request" : "Parse error", NULL);
    } else if (m.kind == JANAS_RPC_REQUEST) {
        request(s, &m);
    }
    /* notifications (initialized, a cancel that came too late) and answers
       to requests never sent need nothing */
    janas_rpc_free(&m);
}

int janas_mcps_open(struct janas_mcps *s)
{
    /* standard output carries the protocol and nothing else: whatever else
       would be written there goes to standard error */
    int fd = dup(STDOUT_FILENO);
    s->out = fd >= 0 ? fdopen(fd, "w") : NULL;
    if (!s->out || dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
        return -1;
    s->in = (struct janas_mcp_proc){.in = -1, .out = STDIN_FILENO};
    return 0;
}

void janas_mcps_run(struct janas_mcps *s)
{
    for (;;) {
        if (s->n_queue) { /* what came while a call ran, in its order */
            char *line = s->queue[0];
            memmove(s->queue, s->queue + 1, --s->n_queue * sizeof(*s->queue));
            take(s, line, strlen(line));
            free(line);
            continue;
        }
        const char *line;
        size_t n;
        int r = janas_mcp_proc_line(&s->in, 60000, &line, &n);
        if (r < 0)
            break; /* the client closed our input: the end */
        if (r > 0)
            take(s, line, n);
    }
    free(s->in.buf);
    free(s->queue);
    s->queue = NULL;
    fclose(s->out);
    s->out = NULL;
}
