/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_client.c - the client side of the Model Context Protocol over stdio
 * (include/janas/mcp.h): a server started, its era found out, its tools
 * listed and called.
 *
 * The era. A server of revision 2026-07-28 or later is stateless: every
 * request carries the protocol version and the client's capabilities in
 * _meta, and there is no handshake. The servers written before it want
 * initialize and notifications/initialized first. The specification's way
 * to tell them apart on stdio is a probe: server/discover, which a modern
 * server answers with its versions (or with UnsupportedProtocolVersion,
 * which is modern too), and anything else - another error, or silence -
 * means a legacy server.
 *
 * This client declares no capabilities: it neither samples, nor elicits,
 * nor lists roots. A modern server that needs one of them answers with
 * input_required, which is reported as the tool's failure; a legacy one
 * that asks anyway gets "method not found", and a ping is answered.
 */
#include "common/mcp_client.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>

#include "common/mcp_config.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

/* how long a legacy server may stay silent to the probe */
#define PROBE_MS 10000
#define MAX_PAGES 64

static _Thread_local char last_error[512];

int32_t janas_mcp_fail(int32_t code, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(last_error, sizeof(last_error), fmt, ap);
    va_end(ap);
    return code;
}

int32_t janas_mcp_abi_version(void)
{
    return JANAS_MCP_ABI_VERSION;
}

const char *janas_mcp_last_error(void)
{
    return last_error;
}

static int32_t copy_out(const char *s, size_t n, char *buf, int32_t cap,
                        int32_t *len)
{
    if (len)
        *len = (int32_t)n;
    if (!buf || cap <= 0)
        return janas_mcp_fail(JANAS_MCP_ESMALL, "no buffer");
    size_t w = n < (size_t)cap - 1 ? n : (size_t)cap - 1;
    memcpy(buf, s, w);
    buf[w] = 0;
    return w == n ? JANAS_MCP_OK
                  : janas_mcp_fail(JANAS_MCP_ESMALL, "buffer too small");
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

/* The params of a request: _meta for a modern server, then the members
   given (JSON text without the braces, may be empty). */
static void params(struct janas_buf *b, const struct janas_mcp *m,
                   const char *members, size_t n)
{
    janas_buf_puts(b, "{");
    if (m->modern) {
        janas_buf_puts(b, "\"_meta\":{\"io.modelcontextprotocol/"
                          "protocolVersion\":");
        janas_json_write_str(b, m->version, strlen(m->version));
        janas_buf_puts(b, ",\"io.modelcontextprotocol/clientInfo\":{\"name\":"
                          "\"janas\",\"version\":");
        janas_json_write_str(b, JANAS_VERSION, strlen(JANAS_VERSION));
        janas_buf_puts(b, "},\"io.modelcontextprotocol/clientCapabilities\""
                          ":{}}");
        if (n)
            janas_buf_puts(b, ",");
    }
    janas_buf_put(b, members, n);
    janas_buf_puts(b, "}");
}

static int send_buf(struct janas_mcp *m, struct janas_buf *b)
{
    int rc = b->oom ? -1 : janas_mcp_proc_send(&m->p, b->p, b->n);
    janas_buf_free(b);
    return rc;
}

static void notify(struct janas_mcp *m, const char *method, const char *members)
{
    /* over HTTP the modern revision has no notifications from the client,
       and closing a stream is what cancels its request */
    if (m->http &&
        (m->modern || strcmp(method, "notifications/cancelled") == 0))
        return;
    struct janas_buf b = {0};
    janas_rpc_begin(&b, -1, method);
    params(&b, m, members, strlen(members));
    janas_rpc_end(&b);
    if (m->http) {
        struct janas_rpc_msg a = {0};
        if (!b.oom)
            janas_mcp_http_exchange(m, method, -1, b.p, b.n, 10000, &a);
        janas_buf_free(&b);
        return;
    }
    send_buf(m, &b);
}

/* A request of the server's own: only ping is served. */
static void answer_server(struct janas_mcp *m, const struct janas_rpc_msg *q)
{
    struct janas_buf b = {0};
    if (strcmp(q->method, "ping") == 0)
        janas_rpc_answer(&b, q->id, 0, NULL);
    else
        janas_rpc_answer(&b, q->id, -32601, "method not supported");
    send_buf(m, &b);
}

/*
 * Sends a request and waits for its answer, at most ms (< 0: no limit),
 * into *a (RESULT or ERROR, for janas_rpc_free). What else comes meanwhile
 * - notifications, answers to requests given up - is let go.
 */
static int request(struct janas_mcp *m, const char *method, const char *members,
                   size_t n, double ms, struct janas_rpc_msg *a)
{
    int64_t id = m->next_id++;
    struct janas_buf b = {0};
    janas_rpc_begin(&b, id, method);
    params(&b, m, members, n);
    janas_rpc_end(&b);
    if (b.oom) {
        janas_buf_free(&b);
        return NOMEM;
    }
    if (m->http) {
        int r = janas_mcp_http_exchange(m, method, id, b.p, b.n, ms, a);
        janas_buf_free(&b);
        return r;
    }
    if (send_buf(m, &b) != 0)
        return GONE;
    double end = ms < 0 ? -1 : now_ms() + ms;
    for (;;) {
        if (__atomic_load_n(&m->cancel, __ATOMIC_RELAXED))
            break;
        double left = end < 0 ? 250 : end - now_ms();
        if (left <= 0)
            break;
        const char *line;
        size_t len;
        int r = janas_mcp_proc_line(&m->p, left < 250 ? (int)left + 1 : 250,
                                    &line, &len);
        if (r < 0)
            return GONE;
        if (r == 0)
            continue;
        char err[128];
        if (janas_rpc_parse(line, len, a, err, sizeof(err)) != 0) {
            janas_rpc_free(a);
            continue; /* not a message: nothing to do with it */
        }
        if (a->kind == JANAS_RPC_REQUEST) {
            answer_server(m, a);
        } else if ((a->kind == JANAS_RPC_RESULT ||
                    a->kind == JANAS_RPC_ERROR) &&
                   a->has_num_id && a->num_id == id) {
            if (a->kind == JANAS_RPC_RESULT &&
                a->result->type != JANAS_JSON_OBJECT) {
                janas_rpc_free(a);
                return GONE; /* not an answer anybody can use */
            }
            return ANSWERED;
        }
        janas_rpc_free(a);
    }
    /* given up: the server is told, so it can stop working on it */
    char why[96];
    snprintf(
        why, sizeof(why), "\"requestId\":%lld,\"reason\":\"%s\"", (long long)id,
        __atomic_load_n(&m->cancel, __ATOMIC_RELAXED) ? "cancelled by the user"
                                                      : "timeout");
    notify(m, "notifications/cancelled", why);
    return __atomic_load_n(&m->cancel, __ATOMIC_RELAXED) ? CANCELLED : TIMEOUT;
}

static int32_t lost(struct janas_mcp *m, int r, const char *what)
{
    switch (r) {
    case TIMEOUT:
        return janas_mcp_fail(JANAS_MCP_EFAIL, "%s: %s: no answer in time",
                              m->name, what);
    case CANCELLED:
        return janas_mcp_fail(JANAS_MCP_EFAIL, "%s: %s: cancelled", m->name,
                              what);
    case NOMEM:
        return janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory");
    case REFUSED:
        return janas_mcp_fail(JANAS_MCP_EFAIL, "%s: %s: refused (%s)", m->name,
                              what, m->http_why);
    default:
        if (m->http)
            return janas_mcp_fail(JANAS_MCP_EFAIL, "%s: %s: %s", m->name, what,
                                  m->http_why);
        return janas_mcp_fail(JANAS_MCP_EFAIL,
                              "%s: %s: the server has gone (its log: %s)",
                              m->name, what, m->log[0] ? m->log : "none");
    }
}

static void set_str(char **dst, const struct janas_json *v)
{
    const char *s = janas_json_str(v);
    if (!s)
        return;
    free(*dst);
    *dst = strdup(s);
}

static void take_info(struct janas_mcp *m, const struct janas_json *info,
                      const struct janas_json *result)
{
    set_str(&m->server_name, janas_json_get(info, "name"));
    set_str(&m->server_version, janas_json_get(info, "version"));
    set_str(&m->instructions, janas_json_get(result, "instructions"));
}

/* Whether a list of versions (JSON strings) holds v. */
static int lists(const struct janas_json *list, const char *v)
{
    if (!list || list->type != JANAS_JSON_ARRAY)
        return 0;
    for (const struct janas_json *s = list->child; s; s = s->next)
        if (janas_json_is(s, v))
            return 1;
    return 0;
}

/* The handshake of the legacy revisions. */
static int32_t initialize(struct janas_mcp *m)
{
    m->modern = 0;
    struct janas_buf b = {0};
    janas_buf_puts(&b, "\"protocolVersion\":\"" JANAS_MCP_LEGACY "\","
                       "\"capabilities\":{},\"clientInfo\":{\"name\":"
                       "\"janas\",\"version\":");
    janas_json_write_str(&b, JANAS_VERSION, strlen(JANAS_VERSION));
    janas_buf_puts(&b, "}");
    if (b.oom) {
        janas_buf_free(&b);
        return janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory");
    }
    struct janas_rpc_msg a;
    int r = request(m, "initialize", b.p, b.n, m->wait_ms, &a);
    janas_buf_free(&b);
    if (r != ANSWERED)
        return lost(m, r, "initialize");
    if (a.kind == JANAS_RPC_ERROR) {
        int32_t rc =
            janas_mcp_fail(JANAS_MCP_EFAIL, "%s: initialize: %s (%lld)",
                           m->name, a.message, (long long)a.code);
        janas_rpc_free(&a);
        return rc;
    }
    const char *v = janas_json_str(janas_json_get(a.result, "protocolVersion"));
    snprintf(m->version, sizeof(m->version), "%s", v ? v : JANAS_MCP_LEGACY);
    take_info(m, janas_json_get(a.result, "serverInfo"), a.result);
    janas_rpc_free(&a);
    notify(m, "notifications/initialized", "");
    return JANAS_MCP_OK;
}

/* The probe: 1 for a modern server (m->version chosen), 0 for a legacy
   one, or an error. */
static int32_t discover(struct janas_mcp *m, double ms)
{
    m->modern = 1;
    snprintf(m->version, sizeof(m->version), "%s", JANAS_MCP_MODERN);
    struct janas_rpc_msg a;
    int r = request(m, "server/discover", "", 0, ms, &a);
    m->modern = 0;
    if (r == TIMEOUT || r == REFUSED)
        return 0;
    if (r != ANSWERED)
        return lost(m, r, "server/discover");
    int modern = 0;
    if (a.kind == JANAS_RPC_RESULT) {
        modern = lists(janas_json_get(a.result, "supportedVersions"),
                       JANAS_MCP_MODERN);
        const struct janas_json *meta = janas_json_get(a.result, "_meta");
        if (modern)
            take_info(m,
                      janas_json_get(meta, "io.modelcontextprotocol/"
                                           "serverInfo"),
                      a.result);
    }
    /* UnsupportedProtocolVersion: a modern server without this revision,
       which the handshake can still reach if it is dual-era; every other
       error is a legacy server's */
    janas_rpc_free(&a);
    m->modern = modern;
    return modern;
}

static int32_t list_tools(struct janas_mcp *m)
{
    janas_mcp_free_tools(m);
    char *cursor = NULL;
    for (int page = 0; page < MAX_PAGES; page++) {
        struct janas_buf b = {0};
        if (cursor) {
            janas_buf_puts(&b, "\"cursor\":");
            janas_json_write_str(&b, cursor, strlen(cursor));
        }
        struct janas_rpc_msg a;
        int r = b.oom ? NOMEM
                      : request(m, "tools/list", b.p ? b.p : "", b.n,
                                m->wait_ms, &a);
        janas_buf_free(&b);
        free(cursor);
        cursor = NULL;
        if (r != ANSWERED)
            return lost(m, r, "tools/list");
        if (a.kind == JANAS_RPC_ERROR) {
            int32_t rc =
                janas_mcp_fail(JANAS_MCP_EFAIL, "%s: tools/list: %s (%lld)",
                               m->name, a.message, (long long)a.code);
            janas_rpc_free(&a);
            return rc;
        }
        if (janas_mcp_take_tools(m, janas_json_get(a.result, "tools")) != 0) {
            janas_rpc_free(&a);
            return janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory");
        }
        const char *next =
            janas_json_str(janas_json_get(a.result, "nextCursor"));
        if (next && *next)
            cursor = strdup(next);
        janas_rpc_free(&a);
        if (!cursor)
            return JANAS_MCP_OK;
    }
    free(cursor);
    return JANAS_MCP_OK; /* enough pages: the rest is let go */
}

/* ~/.cache/janas/mcp/<name>.log, the directories made on the way. */
static void log_path(struct janas_mcp *m)
{
    const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
    char dir[900];
    m->log[0] = 0;
    if (xdg && *xdg)
        snprintf(dir, sizeof(dir), "%s/janas", xdg);
    else if (home && *home)
        snprintf(dir, sizeof(dir), "%s/.cache/janas", home);
    else
        return;
    if (!xdg || !*xdg) {
        char up[900];
        snprintf(up, sizeof(up), "%s/.cache", home);
        mkdir(up, 0700);
    }
    mkdir(dir, 0700);
    size_t l = strlen(dir);
    snprintf(dir + l, sizeof(dir) - l, "/mcp");
    mkdir(dir, 0700);
    char name[64];
    size_t k = 0;
    for (const char *s = m->name; *s && k + 1 < sizeof(name); s++)
        name[k++] =
            (*s == '/' || *s == '.' || (unsigned char)*s < ' ') ? '_' : *s;
    name[k] = 0;
    snprintf(m->log, sizeof(m->log), "%s/%s.log", dir, name);
}

/* Starts the process and finds out its era; then its tools. */
static int32_t connect_server(struct janas_mcp *m)
{
    char err[300];
    int32_t rc;
    if (m->http) {
        m->session[0] = 0;
        rc = discover(m, m->wait_ms);
    } else {
        janas_mcp_proc_stop(&m->p);
        if (janas_mcp_proc_start(&m->p, m->argv, m->env,
                                 m->log[0] ? m->log : NULL, err,
                                 sizeof(err)) != 0)
            return janas_mcp_fail(JANAS_MCP_EOPEN, "%s: %s", m->name, err);
        rc = discover(m, m->wait_ms < PROBE_MS ? m->wait_ms : PROBE_MS);
    }
    if (rc == 0) {
        rc = initialize(m);
        /* a modern server slower to start than the probe: once more, with
           all the time it was given */
        if (rc != JANAS_MCP_OK && !m->http && m->p.eof == 0) {
            int32_t again = discover(m, m->wait_ms);
            rc = again == 1 ? JANAS_MCP_OK : rc;
        }
    } else if (rc == 1) {
        rc = JANAS_MCP_OK;
    }
    if (rc == JANAS_MCP_OK)
        rc = list_tools(m);
    return rc;
}

static char **copy_words(int32_t n, const char *const *w)
{
    char **c = calloc((size_t)(n > 0 ? n : 0) + 1, sizeof(*c));
    for (int32_t i = 0; c && i < n; i++)
        if (!w[i] || !(c[i] = strdup(w[i]))) {
            for (int32_t k = 0; k < i; k++)
                free(c[k]);
            free(c);
            return NULL;
        }
    return c;
}

static void free_words(char **w)
{
    for (char **p = w; p && *p; p++)
        free(*p);
    free(w);
}

void janas_mcp_close(janas_mcp *m)
{
    if (!m)
        return;
    janas_mcp_http_end(m);
    janas_mcp_proc_stop(&m->p);
    janas_mcp_free_tools(m);
    free_words(m->argv);
    free_words(m->env);
    free_words(m->headers);
    janas_buf_free(&m->call_headers);
    free(m->name);
    free(m->server_name);
    free(m->server_version);
    free(m->instructions);
    janas_buf_free(&m->result);
    free(m);
}

int32_t janas_mcp_open_command(const char *name, int32_t argc,
                               const char *const *argv, int32_t n_env,
                               const char *const *env, int32_t seconds,
                               janas_mcp **out)
{
    if (!out || argc < 1 || !argv || !argv[0] || n_env < 0 ||
        (n_env > 0 && !env))
        return janas_mcp_fail(JANAS_MCP_EINVAL, "invalid argument");
    *out = NULL;
    struct janas_mcp *m = calloc(1, sizeof(*m));
    if (!m)
        return janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory");
    m->p.in = m->p.out = -1;
    m->name = strdup(name ? name : argv[0]);
    m->argv = copy_words(argc, argv);
    m->env = copy_words(n_env, env);
    m->wait_ms = (seconds > 0 ? seconds : 60) * 1000;
    m->next_id = 1;
    if (!m->name || !m->argv || !m->env) {
        janas_mcp_close(m);
        return janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory");
    }
    log_path(m);
    int32_t rc = connect_server(m);
    if (rc != JANAS_MCP_OK) {
        janas_mcp_close(m);
        return rc;
    }
    *out = m;
    return JANAS_MCP_OK;
}

int32_t janas_mcp_open_url(const char *name, const char *url, int32_t n_headers,
                           const char *const *headers, int32_t seconds,
                           janas_mcp **out)
{
    if (!out || !url || n_headers < 0 || (n_headers > 0 && !headers))
        return janas_mcp_fail(JANAS_MCP_EINVAL, "invalid argument");
    *out = NULL;
    struct janas_mcp *m = calloc(1, sizeof(*m));
    if (!m)
        return janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory");
    m->p.in = m->p.out = -1;
    m->http = 1;
    char err[300];
    if (janas_url_parse(url, &m->url, err, sizeof(err)) != 0) {
        janas_mcp_close(m);
        return janas_mcp_fail(JANAS_MCP_EINVAL, "%s", err);
    }
    for (int32_t i = 0; i < n_headers; i++)
        if (!headers[i] || strpbrk(headers[i], "\r\n") ||
            !strchr(headers[i], ':')) {
            janas_mcp_close(m);
            return janas_mcp_fail(JANAS_MCP_EINVAL,
                                  "a header is \"Name: value\" on one line");
        }
    m->name = strdup(name ? name : m->url.host);
    m->headers = copy_words(n_headers, headers);
    m->wait_ms = (seconds > 0 ? seconds : 60) * 1000;
    m->next_id = 1;
    if (!m->name || !m->headers) {
        janas_mcp_close(m);
        return janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory");
    }
    int32_t rc = connect_server(m);
    if (rc != JANAS_MCP_OK) {
        janas_mcp_close(m);
        return rc;
    }
    *out = m;
    return JANAS_MCP_OK;
}

int32_t janas_mcp_open(const char *path, const char *name, int32_t seconds,
                       janas_mcp **out)
{
    if (!name || !out)
        return janas_mcp_fail(JANAS_MCP_EINVAL, "invalid argument");
    char def[1024], err[400];
    if (!path) {
        if (janas_mcp_conf_path(def, sizeof(def)) != 0)
            return janas_mcp_fail(JANAS_MCP_EOPEN, "no configuration file");
        path = def;
    }
    struct janas_mcp_conf *c;
    size_t n;
    int r = janas_mcp_config_read(path, &c, &n, err, sizeof(err));
    if (r != 0)
        return janas_mcp_fail(JANAS_MCP_EOPEN, "%s",
                              r > 0 ? "no configuration file" : err);
    const struct janas_mcp_conf *s = NULL;
    for (size_t i = 0; i < n && !s; i++)
        if (strcmp(c[i].name, name) == 0)
            s = &c[i];
    int32_t rc;
    if (!s)
        rc = janas_mcp_fail(JANAS_MCP_EOPEN, "no server %s in %s", name, path);
    else if (s->url) {
        int32_t nh = 0;
        while (s->headers && s->headers[nh])
            nh++;
        rc = janas_mcp_open_url(name, s->url, nh,
                                (const char *const *)s->headers, seconds, out);
    } else {
        int32_t argc = 0, nenv = 0;
        while (s->argv[argc])
            argc++;
        while (s->env[nenv])
            nenv++;
        rc = janas_mcp_open_command(name, argc, (const char *const *)s->argv,
                                    nenv, (const char *const *)s->env, seconds,
                                    out);
    }
    janas_mcp_config_free(c, n);
    return rc;
}

int32_t janas_mcp_config_servers(const char *path, char *buf, int32_t cap,
                                 int32_t *len)
{
    char def[1024], err[400];
    if (!path) {
        if (janas_mcp_conf_path(def, sizeof(def)) != 0)
            return copy_out("", 0, buf, cap, len);
        path = def;
    }
    struct janas_mcp_conf *c;
    size_t n;
    int r = janas_mcp_config_read(path, &c, &n, err, sizeof(err));
    if (r > 0)
        return copy_out("", 0, buf, cap, len);
    if (r < 0)
        return janas_mcp_fail(JANAS_MCP_EOPEN, "%s", err);
    struct janas_buf b = {0};
    for (size_t i = 0; i < n; i++)
        janas_buf_printf(&b, "%s%s%s\n", c[i].name,
                         c[i].disabled ? " (disabled)" : "",
                         c[i].url ? " (http)" : "");
    janas_mcp_config_free(c, n);
    int32_t rc = b.oom ? janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory")
                       : copy_out(b.p ? b.p : "", b.n, buf, cap, len);
    janas_buf_free(&b);
    return rc;
}

int32_t janas_mcp_config_path(char *buf, int32_t cap, int32_t *len)
{
    char def[1024];
    if (janas_mcp_conf_path(def, sizeof(def)) != 0)
        return janas_mcp_fail(JANAS_MCP_EOPEN, "no home directory");
    return copy_out(def, strlen(def), buf, cap, len);
}

int32_t janas_mcp_info(const janas_mcp *m, char *buf, int32_t cap, int32_t *len)
{
    if (!m)
        return janas_mcp_fail(JANAS_MCP_EINVAL, "invalid argument");
    struct janas_buf b = {0};
    janas_buf_printf(&b, "name: %s\n", m->server_name ? m->server_name : "");
    janas_buf_printf(&b, "version: %s\n",
                     m->server_version ? m->server_version : "");
    janas_buf_printf(&b, "protocol: %s\n", m->version);
    if (m->instructions && *m->instructions)
        janas_buf_printf(&b, "instructions: %s\n", m->instructions);
    int32_t rc = b.oom ? janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory")
                       : copy_out(b.p, b.n, buf, cap, len);
    janas_buf_free(&b);
    return rc;
}

int32_t janas_mcp_instructions(const janas_mcp *m, char *buf, int32_t cap,
                               int32_t *len)
{
    if (!m)
        return janas_mcp_fail(JANAS_MCP_EINVAL, "invalid argument");
    const char *s = m->instructions ? m->instructions : "";
    return copy_out(s, strlen(s), buf, cap, len);
}

int32_t janas_mcp_tool_count(const janas_mcp *m)
{
    return m ? (int32_t)m->n_tools : 0;
}

int32_t janas_mcp_tool_names(const janas_mcp *m, char *buf, int32_t cap,
                             int32_t *len)
{
    if (!m)
        return janas_mcp_fail(JANAS_MCP_EINVAL, "invalid argument");
    struct janas_buf b = {0};
    for (size_t i = 0; i < m->n_tools; i++)
        janas_buf_printf(&b, "%s\n", m->tools[i].name);
    int32_t rc = b.oom ? janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory")
                       : copy_out(b.p ? b.p : "", b.n, buf, cap, len);
    janas_buf_free(&b);
    return rc;
}

int32_t janas_mcp_tools(const janas_mcp *m, const char *prefix,
                        int32_t prefix_len, char *buf, int32_t cap,
                        int32_t *len)
{
    if (!m)
        return janas_mcp_fail(JANAS_MCP_EINVAL, "invalid argument");
    size_t pn = !prefix          ? 0
                : prefix_len < 0 ? strlen(prefix)
                                 : (size_t)prefix_len;
    struct janas_buf b = {0}, name = {0};
    janas_buf_puts(&b, "[");
    for (size_t i = 0; i < m->n_tools; i++) {
        const struct janas_mcp_tool *t = &m->tools[i];
        name.n = 0;
        janas_buf_put(&name, prefix, pn);
        janas_buf_puts(&name, t->name);
        janas_buf_puts(&b, i ? ", " : "");
        janas_buf_puts(&b,
                       "{\"type\": \"function\", \"function\": {\"name\": ");
        janas_json_write_str(&b, name.p ? name.p : "", name.n);
        janas_buf_puts(&b, ", \"description\": ");
        janas_json_write_str(&b, t->description, strlen(t->description));
        janas_buf_puts(&b, ", \"parameters\": ");
        janas_buf_puts(&b, t->schema);
        janas_buf_puts(&b, "}}");
    }
    janas_buf_puts(&b, "]");
    int32_t rc = b.oom || name.oom
                     ? janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory")
                     : copy_out(b.p, b.n, buf, cap, len);
    janas_buf_free(&b);
    janas_buf_free(&name);
    return rc;
}

/* Whether the process has ended (it is then reaped). */
static int ended(struct janas_mcp *m)
{
    if (m->http)
        return 0; /* each request is a connection of its own */
    if (m->p.pid <= 0 || m->p.eof)
        return 1;
    return waitpid(m->p.pid, NULL, WNOHANG) == m->p.pid ? (m->p.pid = 0, 1) : 0;
}

int32_t janas_mcp_tool_read_only(const janas_mcp *m, const char *name,
                                 int32_t len)
{
    if (!m || !name)
        return -1;
    size_t n = len < 0 ? strlen(name) : (size_t)len;
    for (size_t i = 0; i < m->n_tools; i++)
        if (strlen(m->tools[i].name) == n &&
            memcmp(m->tools[i].name, name, n) == 0)
            return m->tools[i].read_only;
    return -1;
}

int32_t janas_mcp_refresh(janas_mcp *m)
{
    if (!m)
        return janas_mcp_fail(JANAS_MCP_EINVAL, "invalid argument");
    __atomic_store_n(&m->cancel, 0, __ATOMIC_RELAXED);
    if (ended(m))
        return connect_server(m);
    return list_tools(m);
}

int32_t janas_mcp_call(janas_mcp *m, const char *name, int32_t name_len,
                       const char *args, int32_t args_len, int32_t seconds)
{
    if (!m || !name)
        return janas_mcp_fail(JANAS_MCP_EINVAL, "invalid argument");
    __atomic_store_n(&m->cancel, 0, __ATOMIC_RELAXED);
    m->result.n = 0;
    m->result_error = 0;
    size_t nn = name_len < 0 ? strlen(name) : (size_t)name_len;
    size_t an = !args ? 0 : args_len < 0 ? strlen(args) : (size_t)args_len;
    /* the arguments are read and written again: one line, and an object */
    char err[160];
    struct janas_json_doc *doc = NULL;
    if (an) {
        doc = janas_json_parse(args, an, err, sizeof(err));
        const struct janas_json *root = janas_json_root(doc);
        if (!root || root->type != JANAS_JSON_OBJECT) {
            janas_json_free(doc);
            return janas_mcp_fail(JANAS_MCP_EINVAL,
                                  "the arguments must be a JSON object%s%s",
                                  root ? "" : ": ", root ? "" : err);
        }
    }
    struct janas_buf b = {0};
    janas_buf_puts(&b, "\"name\":");
    janas_json_write_str(&b, name, nn);
    janas_buf_puts(&b, ",\"arguments\":");
    if (doc)
        janas_json_write(&b, janas_json_root(doc));
    else
        janas_buf_puts(&b, "{}");
    janas_json_free(doc);
    if (b.oom) {
        janas_buf_free(&b);
        return janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory");
    }
    /* a server that died since the last call is started again: the
       protocol has no state that could have been lost with it */
    if (ended(m)) {
        int32_t rc = connect_server(m);
        if (rc != JANAS_MCP_OK) {
            janas_buf_free(&b);
            return rc;
        }
    }
    /* over HTTP, the tool's name and the arguments its schema marks go into
       headers too */
    const struct janas_mcp_tool *t = NULL;
    for (size_t i = 0; i < m->n_tools && !t; i++)
        if (strlen(m->tools[i].name) == nn &&
            memcmp(m->tools[i].name, name, nn) == 0)
            t = &m->tools[i];
    char *call_name = strndup(name, nn);
    struct janas_json_doc *hd =
        m->http && an ? janas_json_parse(args, an, err, sizeof(err)) : NULL;
    if (!call_name || janas_mcp_param_headers(m, t, janas_json_root(hd)) != 0) {
        free(call_name);
        janas_json_free(hd);
        janas_buf_free(&b);
        return janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory");
    }
    janas_json_free(hd);
    m->call_name = m->http ? call_name : NULL;
    struct janas_rpc_msg a;
    int r = request(m, "tools/call", b.p, b.n, seconds > 0 ? seconds * 1e3 : -1,
                    &a);
    m->call_name = NULL;
    free(call_name);
    janas_buf_free(&b);
    if (r != ANSWERED)
        return lost(m, r, "tools/call");
    int32_t rc = JANAS_MCP_OK;
    const struct janas_json *rt = janas_json_get(a.result, "resultType");
    if (a.kind == JANAS_RPC_ERROR)
        rc = janas_mcp_fail(JANAS_MCP_EFAIL, "%s: %s (%lld)", m->name,
                            a.message, (long long)a.code);
    else if (rt && janas_json_is(rt, "input_required"))
        rc = janas_mcp_fail(JANAS_MCP_EFAIL,
                            "%s: the tool asks for input this client cannot "
                            "give (a form, the model, the roots)",
                            m->name);
    else if (rt && !janas_json_is(rt, "complete"))
        rc = janas_mcp_fail(JANAS_MCP_EFAIL, "%s: a result of unknown type",
                            m->name);
    else
        janas_mcp_take_result(m, a.result);
    janas_rpc_free(&a);
    if (rc == JANAS_MCP_OK && m->result.oom)
        rc = janas_mcp_fail(JANAS_MCP_ENOMEM, "out of memory");
    return rc;
}

int32_t janas_mcp_result(const janas_mcp *m, char *buf, int32_t cap,
                         int32_t *len)
{
    if (!m)
        return janas_mcp_fail(JANAS_MCP_EINVAL, "invalid argument");
    return copy_out(m->result.p ? m->result.p : "", m->result.n, buf, cap, len);
}

int32_t janas_mcp_result_error(const janas_mcp *m)
{
    return m ? m->result_error : 0;
}

void janas_mcp_cancel(janas_mcp *m)
{
    if (m)
        __atomic_store_n(&m->cancel, 1, __ATOMIC_RELAXED);
}
