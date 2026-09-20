/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_client_http.c - the Streamable HTTP transport of the MCP client
 * (include/janas/mcp.h): every message a POST of its own, the answer a
 * JSON object or an event stream scoped to the request.
 *
 * The body's fields are mirrored into headers for whatever stands between
 * client and server: MCP-Protocol-Version, Mcp-Method, Mcp-Name, and the
 * arguments a tool's schema marks with x-mcp-header as Mcp-Param-<name>. A
 * value a header cannot carry as it is goes in base64 between =?base64? and
 * ?=. A legacy server (2025-11-25 and before) may give a session at
 * initialize, which every later request carries.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "common/mcp_client.h"

#define BODY_MAX ((size_t)64 << 20)
#define MAX_PARAMS 32

/* A header value as it is, or in the base64 sentinel form when it cannot
   be carried plain (or would read as one). */
static void header_value(struct janas_buf *b, const char *v, size_t n)
{
    int plain = n > 0 && v[0] != ' ' && v[0] != '\t' && v[n - 1] != ' ' &&
                v[n - 1] != '\t';
    for (size_t i = 0; i < n && plain; i++) {
        unsigned char c = (unsigned char)v[i];
        plain = (c >= 0x20 && c <= 0x7e) || c == '\t';
    }
    if (plain && n >= 11 && strncmp(v, "=?base64?", 9) == 0 &&
        strncmp(v + n - 2, "?=", 2) == 0)
        plain = 0;
    if (plain || n == 0) {
        janas_buf_put(b, v, n);
        return;
    }
    janas_buf_puts(b, "=?base64?");
    janas_base64(b, (const unsigned char *)v, n);
    janas_buf_puts(b, "?=");
}

/* A header name: the token characters of RFC 9110. */
static int token(const char *s, size_t n)
{
    if (n == 0 || n > 64)
        return 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || strchr("!#$%&'*+-.^_`|~", c)))
            return 0;
    }
    return 1;
}

/* How many x-mcp-header keys are anywhere in v. */
static int count_marks(const struct janas_json *v, int depth)
{
    if (!v || depth > 64)
        return 0;
    int n = 0;
    for (const struct janas_json *c = v->child; c; c = c->next) {
        if (v->type == JANAS_JSON_OBJECT && c->key_n == 12 &&
            memcmp(c->key, "x-mcp-header", 12) == 0)
            n++;
        if (c->type == JANAS_JSON_OBJECT || c->type == JANAS_JSON_ARRAY)
            n += count_marks(c, depth + 1);
    }
    return n;
}

/* The marks reachable through properties alone, into lines. */
static int walk(const struct janas_json *schema, struct janas_buf *path,
                struct janas_buf *out, int *found, int depth)
{
    const struct janas_json *props = janas_json_get(schema, "properties");
    if (!props || props->type != JANAS_JSON_OBJECT || depth > 8)
        return 0;
    for (const struct janas_json *p = props->child; p; p = p->next) {
        if (p->type != JANAS_JSON_OBJECT)
            continue;
        size_t at = path->n;
        janas_buf_put(path, "\x1f", 1);
        janas_buf_put(path, p->key, p->key_n);
        const struct janas_json *mark = janas_json_get(p, "x-mcp-header");
        if (mark) {
            const struct janas_json *type = janas_json_get(p, "type");
            if (!janas_json_str(mark) || !token(mark->s, mark->n) ||
                !(janas_json_is(type, "string") ||
                  janas_json_is(type, "integer") ||
                  janas_json_is(type, "boolean")) ||
                memchr(p->key, '\x1f', p->key_n) ||
                memchr(p->key, '\n', p->key_n) || ++*found > MAX_PARAMS)
                return -1;
            /* unique, whatever the case */
            for (const char *l = out->p; l && l < out->p + out->n;) {
                const char *tab =
                    memchr(l, '\x1f', out->n - (size_t)(l - out->p));
                if (tab && (size_t)(tab - l) == mark->n &&
                    strncasecmp(l, mark->s, mark->n) == 0)
                    return -1;
                const char *nl = memchr(l, '\n', out->n - (size_t)(l - out->p));
                l = nl ? nl + 1 : out->p + out->n;
            }
            janas_buf_put(out, mark->s, mark->n);
            janas_buf_put(out, path->p, path->n);
            janas_buf_put(out, "\n", 1);
        }
        if (walk(p, path, out, found, depth + 1) != 0)
            return -1;
        path->n = at;
    }
    return 0;
}

int janas_mcp_schema_params(struct janas_mcp_tool *t,
                            const struct janas_json *schema)
{
    t->params = NULL;
    int all = count_marks(schema, 0);
    if (!all)
        return 0;
    struct janas_buf path = {0}, out = {0};
    int found = 0;
    int rc = walk(schema, &path, &out, &found, 0);
    janas_buf_free(&path);
    /* a mark off the properties' paths (under items, anyOf, $ref...) makes
       the whole tool invalid, as the specification has it */
    if (rc != 0 || found != all || out.oom) {
        janas_buf_free(&out);
        return -1;
    }
    janas_buf_put(&out, "", 1);
    t->params = out.p;
    return 0;
}

int janas_mcp_param_headers(struct janas_mcp *m, const struct janas_mcp_tool *t,
                            const struct janas_json *args)
{
    m->call_headers.n = 0;
    m->call_headers.oom = 0;
    for (const char *l = t ? t->params : NULL; l && *l;) {
        const char *nl = strchr(l, '\n');
        const char *sep = memchr(l, '\x1f', (size_t)(nl - l));
        /* the value at the path */
        const struct janas_json *v = args;
        for (const char *k = sep; v && k && k < nl;) {
            const char *next = memchr(k + 1, '\x1f', (size_t)(nl - k - 1));
            size_t kn = (size_t)((next ? next : nl) - (k + 1));
            const struct janas_json *c = NULL;
            if (v->type == JANAS_JSON_OBJECT)
                for (c = v->child; c; c = c->next)
                    if (c->key_n == kn && memcmp(c->key, k + 1, kn) == 0)
                        break;
            v = c;
            k = next;
        }
        if (v && v->type != JANAS_JSON_NULL) {
            janas_buf_puts(&m->call_headers, "Mcp-Param-");
            janas_buf_put(&m->call_headers, l, (size_t)(sep - l));
            janas_buf_puts(&m->call_headers, ": ");
            if (v->type == JANAS_JSON_STRING)
                header_value(&m->call_headers, v->s, v->n);
            else if (v->type == JANAS_JSON_TRUE || v->type == JANAS_JSON_FALSE)
                janas_buf_puts(&m->call_headers,
                               v->type == JANAS_JSON_TRUE ? "true" : "false");
            else if (v->type == JANAS_JSON_NUMBER)
                janas_buf_printf(&m->call_headers, "%lld",
                                 (long long)janas_json_num(v, 0));
            janas_buf_put(&m->call_headers, "\n", 1);
        }
        l = nl + 1;
    }
    return m->call_headers.oom ? -1 : 0;
}

/* The header lines of a request, into an array of pointers into b. */
static size_t build_headers(struct janas_mcp *m, const char *method,
                            struct janas_buf *b, const char **h, size_t cap)
{
    janas_buf_puts(b, "Content-Type: application/json\n"
                      "Accept: application/json, text/event-stream\n");
    if (strcmp(method, "initialize") != 0)
        janas_buf_printf(b, "MCP-Protocol-Version: %s\n", m->version);
    janas_buf_puts(b, "Mcp-Method: ");
    header_value(b, method, strlen(method));
    janas_buf_puts(b, "\n");
    if (m->call_name) {
        janas_buf_puts(b, "Mcp-Name: ");
        header_value(b, m->call_name, strlen(m->call_name));
        janas_buf_puts(b, "\n");
        if (m->call_headers.n)
            janas_buf_put(b, m->call_headers.p, m->call_headers.n);
    }
    if (m->session[0])
        janas_buf_printf(b, "Mcp-Session-Id: %s\n", m->session);
    for (char **c = m->headers; c && *c; c++)
        janas_buf_printf(b, "%s\n", *c);
    if (b->oom)
        return 0;
    size_t n = 0;
    for (char *l = b->p; l < b->p + b->n && n < cap;) {
        char *nl = memchr(l, '\n', b->n - (size_t)(l - b->p));
        *nl = 0;
        h[n++] = l;
        l = nl + 1;
    }
    return n;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

/* A message of the answer: ours when it is the answer to id. */
static int ours(const char *p, size_t n, int64_t id, struct janas_rpc_msg *a)
{
    char err[64];
    if (janas_rpc_parse(p, n, a, err, sizeof(err)) == 0 &&
        (a->kind == JANAS_RPC_RESULT || a->kind == JANAS_RPC_ERROR) &&
        ((a->has_num_id && a->num_id == id) ||
         (a->kind == JANAS_RPC_ERROR && !a->id)))
        return a->kind == JANAS_RPC_RESULT &&
                       a->result->type != JANAS_JSON_OBJECT
                   ? 0
                   : 1;
    janas_rpc_free(a);
    return 0;
}

int janas_mcp_http_exchange(struct janas_mcp *m, const char *method, int64_t id,
                            const char *body, size_t n, double ms,
                            struct janas_rpc_msg *a)
{
    struct janas_buf hb = {0};
    const char *h[64];
    size_t nh = build_headers(m, method, &hb, h, 64);
    if (!nh) {
        janas_buf_free(&hb);
        return NOMEM;
    }
    struct janas_http c;
    char err[300];
    double end = ms < 0 ? -1 : now_ms() + ms;
    int rc = janas_http_request(&c, &m->url, "POST", h, nh, body, n,
                                ms < 0 ? -1 : (int)ms,
                                (volatile int *)&m->cancel, err, sizeof(err));
    janas_buf_free(&hb);
    m->http_status = 0;
    if (rc != 0) {
        snprintf(m->http_why, sizeof(m->http_why), "%s", err);
        return __atomic_load_n(&m->cancel, __ATOMIC_RELAXED) ? CANCELLED
               : strstr(err, "in time")                      ? TIMEOUT
                                                             : GONE;
    }
    m->http_status = c.head.status;
    if (c.head.session[0] && strcmp(method, "initialize") == 0)
        snprintf(m->session, sizeof(m->session), "%s", c.head.session);
    int sse = strncasecmp(c.head.content_type, "text/event-stream", 17) == 0;
    int result = GONE;
    if (id < 0) { /* a notification: accepted or not, nothing to read */
        janas_http_close(&c);
        return c.head.status / 100 == 2 ? ANSWERED : REFUSED;
    }
    struct janas_buf in = {0}, ev = {0};
    struct janas_sse s = {0};
    for (;;) {
        if (__atomic_load_n(&m->cancel, __ATOMIC_RELAXED)) {
            result = CANCELLED; /* closing the stream is the cancel */
            break;
        }
        if (end >= 0 && now_ms() > end) {
            result = TIMEOUT;
            break;
        }
        in.n = 0;
        int r = janas_http_body(&c, &in, 250);
        if (r == -2)
            continue; /* the time is the one the request was given */
        if (r < 0 || in.oom)
            break;
        if (sse && r > 0) {
            janas_sse_add(&s, in.p, in.n);
            int found = 0;
            while (!found && janas_sse_next(&s, &ev) == 1)
                found = ours(ev.p, ev.n, id, a);
            if (found) {
                result = ANSWERED;
                break;
            }
            if (s.in.n > BODY_MAX)
                break;
        } else if (r > 0) {
            janas_buf_put(&ev, in.p, in.n);
            if (ev.n > BODY_MAX)
                break;
        }
        if (r == 0) { /* the whole body */
            if (!sse && ev.n && ours(ev.p, ev.n, id, a))
                result = ANSWERED;
            break;
        }
    }
    if (result == TIMEOUT) {
        snprintf(m->http_why, sizeof(m->http_why), "no answer in time");
    } else if (result != ANSWERED && result != CANCELLED &&
               c.head.status / 100 != 2) {
        result = REFUSED;
        snprintf(m->http_why, sizeof(m->http_why), "HTTP %d", c.head.status);
    } else if (result == GONE) {
        snprintf(m->http_why, sizeof(m->http_why),
                 "HTTP %d with no answer in it", c.head.status);
    }
    janas_buf_free(&in);
    janas_buf_free(&ev);
    janas_sse_free(&s);
    janas_http_close(&c);
    return result;
}

void janas_mcp_http_end(struct janas_mcp *m)
{
    if (!m->http || !m->session[0])
        return;
    char line[300];
    snprintf(line, sizeof(line), "Mcp-Session-Id: %s", m->session);
    const char *h[] = {line};
    struct janas_http c;
    char err[128];
    if (janas_http_request(&c, &m->url, "DELETE", h, 1, "", 0, 5000, NULL, err,
                           sizeof(err)) == 0)
        janas_http_close(&c);
    m->session[0] = 0;
}
