/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * http.c - the HTTP side of janas-server, on GNU libmicrohttpd: a thread
 * per connection, so a handler may wait for the engine without holding up
 * anybody else. Every operation of OpenAI's specification has a row in the
 * route table; the ones not written yet answer 501 and say why.
 */
#include "server.h"
#include "server_test_html.h" /* made by build.sh from test.html */

#include <microhttpd.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/socket.h>

enum route_kind { implemented, planned, other_model, training, cloud };

struct route {
    const char *method, *path, *op, *tag;
    enum route_kind kind;
    srv_handler h;
};

static const struct route routes[] = {
#define JANAS_ROUTE(m, p, op, tag, kind, h) {#m, p, op, tag, kind, h},
#include "routes.inc"
#undef JANAS_ROUTE
};

/* What arrives for one request, across the calls of the access handler. */
struct req_ctx {
    char *body;
    size_t n, cap;
    int too_big;
};

void srv_new_id(char *buf, size_t cap, const char *prefix)
{
    unsigned char r[12];
    if (getrandom(r, sizeof(r), 0) != (ssize_t)sizeof(r))
        memset(r, 0, sizeof(r));
    static const char hex[] = "0123456789abcdef";
    size_t w = (size_t)snprintf(buf, cap, "%s", prefix);
    for (size_t i = 0; i < sizeof(r) && w + 2 < cap; i++) {
        buf[w++] = hex[r[i] >> 4];
        buf[w++] = hex[r[i] & 15];
    }
    buf[w < cap ? w : cap - 1] = 0;
}

const char *srv_query(struct srv_req *r, const char *key)
{
    return MHD_lookup_connection_value(r->conn, MHD_GET_ARGUMENT_KIND, key);
}

static enum MHD_Result queue(struct MHD_Connection *c, unsigned status,
                             struct MHD_Response *resp, const char *type)
{
    if (!resp)
        return MHD_NO;
    MHD_add_response_header(resp, MHD_HTTP_HEADER_CONTENT_TYPE, type);
    enum MHD_Result r = MHD_queue_response(c, status, resp);
    MHD_destroy_response(resp);
    return r;
}

int srv_reply_json(struct srv_req *r, unsigned status, char *json, size_t n)
{
    if (!json)
        return srv_reply_error(r, 500, "server_error", NULL,
                               "cannot write the answer");
    struct MHD_Response *resp = MHD_create_response_from_buffer_copy(n, json);
    free(json);
    return queue(r->conn, status, resp, "application/json");
}

/* s as a JSON string, into buf (quotes included) */
static size_t json_quote(char *buf, size_t cap, const char *s)
{
    size_t w = 0;
    if (w + 1 < cap)
        buf[w++] = '"';
    for (; *s && w + 7 < cap; s++) {
        unsigned char ch = (unsigned char)*s;
        if (ch == '"' || ch == '\\') {
            buf[w++] = '\\';
            buf[w++] = (char)ch;
        } else if (ch < 0x20) {
            w += (size_t)snprintf(buf + w, cap - w, "\\u%04x", ch);
        } else {
            buf[w++] = (char)ch;
        }
    }
    if (w + 1 < cap)
        buf[w++] = '"';
    buf[w] = 0;
    return w;
}

int srv_reply_error(struct srv_req *r, unsigned status, const char *type,
                    const char *code, const char *fmt, ...)
{
    char msg[1024], qm[2200], qc[160], out[2600];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    json_quote(qm, sizeof(qm), msg);
    if (code)
        json_quote(qc, sizeof(qc), code);
    else
        snprintf(qc, sizeof(qc), "null");
    int n = snprintf(out, sizeof(out),
                     "{\"error\":{\"message\":%s,\"type\":\"%s\","
                     "\"param\":null,\"code\":%s}}",
                     qm, type, qc);
    struct MHD_Response *resp =
        MHD_create_response_from_buffer_copy((size_t)n, out);
    if (r->srv->cfg.verbose)
        fprintf(stderr, "janas-server: %s %s: %u %s\n", r->method, r->path,
                status, msg);
    return queue(r->conn, status, resp, "application/json");
}

int srv_reply_limit(struct srv_req *r, const struct srv_job *j)
{
    uint32_t in = j->st.prompt_tokens, max = r->srv->cfg.max_input;
    char qm[600], out[1400];
    char msg[400];
    snprintf(msg, sizeof(msg),
             "The input is %u tokens, above the limit of %u this server "
             "takes (--max-input): %u too many. Nothing was cut; shorten the "
             "input and send it again.",
             in, max, in > max ? in - max : 0);
    json_quote(qm, sizeof(qm), msg);
    int n = snprintf(out, sizeof(out),
                     "{\"error\":{\"message\":%s,\"type\":"
                     "\"invalid_request_error\",\"param\":\"input\",\"code\":"
                     "\"input_limit_exceeded\",\"input_tokens\":%u,"
                     "\"max_input_tokens\":%u,\"excess_tokens\":%u,"
                     "\"system_tokens\":%u,\"tools_tokens\":%u,"
                     "\"history_tokens\":%u,\"last_tokens\":%u,"
                     "\"context_size\":%u}}",
                     qm, in, max, in > max ? in - max : 0, j->st.prompt_system,
                     j->st.prompt_tools, j->st.prompt_history,
                     j->st.prompt_last, j->st.context_size);
    struct MHD_Response *resp =
        MHD_create_response_from_buffer_copy((size_t)n, out);
    if (r->srv->cfg.verbose)
        fprintf(stderr, "janas-server: %s %s: 400 %s\n", r->method, r->path,
                msg);
    return queue(r->conn, 400, resp, "application/json");
}

struct stream {
    srv_stream_next next;
    srv_stream_done done;
    void *ctx;
};

static ssize_t stream_reader(void *cls, uint64_t pos, char *buf, size_t max)
{
    (void)pos;
    struct stream *s = cls;
    for (;;) { /* a thread per connection: waiting here holds nobody up */
        long n = s->next(s->ctx, buf, max);
        if (n < 0)
            return MHD_CONTENT_READER_END_OF_STREAM;
        if (n > 0)
            return (ssize_t)n;
    }
}

static void stream_free(void *cls)
{
    struct stream *s = cls;
    s->done(s->ctx);
    free(s);
}

int srv_reply_stream(struct srv_req *r, srv_stream_next next,
                     srv_stream_done done, void *ctx)
{
    struct stream *s = malloc(sizeof(*s));
    if (!s) {
        done(ctx);
        return MHD_NO;
    }
    *s = (struct stream){next, done, ctx};
    struct MHD_Response *resp = MHD_create_response_from_callback(
        MHD_SIZE_UNKNOWN, 4096, stream_reader, s, stream_free);
    if (!resp) {
        stream_free(s);
        return MHD_NO;
    }
    MHD_add_response_header(resp, "Cache-Control", "no-cache");
    return queue(r->conn, 200, resp, "text/event-stream");
}

/* Does the path match the route's pattern? Its {…} segments go to param. */
static int match(const char *pat, const char *path, struct srv_req *r)
{
    r->n_param = 0;
    while (*pat && *path) {
        if (*pat == '{') {
            const char *pe = strchr(pat, '}'), *se = strchr(path, '/');
            size_t n = se ? (size_t)(se - path) : strlen(path);
            if (!pe || n == 0 || n >= sizeof(r->param[0]) ||
                r->n_param >= (int)(sizeof(r->param) / sizeof(r->param[0])))
                return 0;
            memcpy(r->param[r->n_param], path, n);
            r->param[r->n_param++][n] = 0;
            pat = pe + 1;
            path += n;
        } else if (*pat++ != *path++) {
            return 0;
        }
    }
    return *pat == 0 && *path == 0;
}

static const char *kind_text(enum route_kind k)
{
    switch (k) {
    case planned:
        return "is in OpenAI's API and not in janas-server yet";
    case other_model:
        return "needs a model of another kind (sound, images, video), which "
               "janas-server does not run";
    case training:
        return "trains models, which janas-server does not do";
    default:
        return "belongs to the administration of OpenAI's service and has no "
               "meaning for a server running on your own machine";
    }
}

/* Is the key right? Compared in time independent of where it differs. */
static int key_ok(struct srv_req *r)
{
    const char *want = r->srv->cfg.api_key;
    if (!want)
        return 1;
    const char *h = MHD_lookup_connection_value(r->conn, MHD_HEADER_KIND,
                                                MHD_HTTP_HEADER_AUTHORIZATION);
    if (!h || strncmp(h, "Bearer ", 7) != 0)
        return 0;
    h += 7;
    size_t a = strlen(h), b = strlen(want);
    unsigned diff = (unsigned)(a ^ b);
    for (size_t i = 0; i < b; i++)
        diff |= (unsigned char)want[i] ^ (unsigned char)(i < a ? h[i] : 0);
    return diff == 0;
}

static enum MHD_Result dispatch(struct srv_req *r)
{
    if (strcmp(r->path, "/health") == 0 && strcmp(r->method, "GET") == 0) {
        static const char ok[] = "{\"status\":\"ok\"}";
        return queue(r->conn, 200,
                     MHD_create_response_from_buffer_copy(sizeof(ok) - 1, ok),
                     "application/json");
    }
    if (r->srv->cfg.test && strcmp(r->method, "GET") == 0 &&
        (strcmp(r->path, "/test") == 0 || strcmp(r->path, "/test/") == 0)) {
        /* the page asks for the key itself: it is not behind it. It may
           load nothing from anywhere else and talk only to this server. */
        struct MHD_Response *resp = MHD_create_response_from_buffer_static(
            sizeof(janas_server_test_html) - 1, janas_server_test_html);
        if (!resp)
            return MHD_NO;
        MHD_add_response_header(resp, "Content-Security-Policy",
                                "default-src 'none'; script-src "
                                "'unsafe-inline'; style-src 'unsafe-inline'; "
                                "connect-src 'self'; base-uri 'none'; "
                                "form-action 'none'; frame-ancestors 'none'");
        MHD_add_response_header(resp, "X-Content-Type-Options", "nosniff");
        MHD_add_response_header(resp, "Referrer-Policy", "no-referrer");
        MHD_add_response_header(resp, "Cache-Control", "no-store");
        return queue(r->conn, 200, resp, "text/html; charset=utf-8");
    }
    if (!key_ok(r))
        return (enum MHD_Result)srv_reply_error(
            r, 401, "invalid_request_error", "invalid_api_key",
            "Incorrect or missing API key: send it as "
            "\"Authorization: Bearer <key>\".");
    const struct route *other = NULL;
    for (size_t i = 0; i < sizeof(routes) / sizeof(routes[0]); i++) {
        const struct route *rt = &routes[i];
        if (!match(rt->path, r->path, r))
            continue;
        if (strcmp(rt->method, r->method) != 0) {
            other = rt;
            continue;
        }
        if (rt->h)
            return (enum MHD_Result)rt->h(r);
        return (enum MHD_Result)srv_reply_error(
            r, 501, "not_supported", "not_implemented", "%s %s (%s) %s.",
            r->method, rt->path, rt->op, kind_text(rt->kind));
    }
    if (other)
        return (enum MHD_Result)srv_reply_error(
            r, 405, "invalid_request_error", "method_not_allowed",
            "%s is not allowed on %s.", r->method, r->path);
    return (enum MHD_Result)srv_reply_error(
        r, 404, "invalid_request_error", "unknown_url",
        "Unknown request URL: %s %s.", r->method, r->path);
}

static enum MHD_Result access_handler(void *cls, struct MHD_Connection *conn,
                                      const char *url, const char *method,
                                      const char *version,
                                      const char *upload_data,
                                      size_t *upload_data_size, void **req_cls)
{
    (void)version;
    struct srv *srv = cls;
    struct req_ctx *ctx = *req_cls;
    if (!ctx) { /* first call: the headers only */
        ctx = calloc(1, sizeof(*ctx));
        if (!ctx)
            return MHD_NO;
        *req_cls = ctx;
        return MHD_YES;
    }
    if (*upload_data_size) { /* a piece of the body */
        size_t n = *upload_data_size;
        *upload_data_size = 0;
        if (ctx->too_big || ctx->n + n > srv->cfg.max_body) {
            ctx->too_big = 1;
            return MHD_YES;
        }
        if (ctx->n + n + 1 > ctx->cap) {
            size_t cap = 2 * (ctx->n + n) + 1024;
            if (cap > srv->cfg.max_body + 1)
                cap = srv->cfg.max_body + 1;
            char *b = realloc(ctx->body, cap);
            if (!b)
                return MHD_NO;
            ctx->body = b;
            ctx->cap = cap;
        }
        memcpy(ctx->body + ctx->n, upload_data, n);
        ctx->n += n;
        ctx->body[ctx->n] = 0;
        return MHD_YES;
    }
    /* the whole request is here */
    struct srv_req r = {.srv = srv,
                        .conn = conn,
                        .method = method,
                        .path = url,
                        .body = ctx->body ? ctx->body : "",
                        .n_body = ctx->n};
    if (strncmp(r.path, "/v1/", 4) == 0) /* OpenAI's base URL ends in /v1 */
        r.path += 3;
    if (srv->cfg.verbose)
        fprintf(stderr, "janas-server: %s %s (%zu bytes)\n", method, url,
                ctx->n);
    if (ctx->too_big)
        return (enum MHD_Result)srv_reply_error(
            &r, 413, "invalid_request_error", "request_too_large",
            "The request body is larger than %zu bytes.", srv->cfg.max_body);
    return dispatch(&r);
}

static void completed(void *cls, struct MHD_Connection *conn, void **req_cls,
                      enum MHD_RequestTerminationCode toe)
{
    (void)cls;
    (void)conn;
    (void)toe;
    struct req_ctx *ctx = *req_cls;
    if (ctx) {
        free(ctx->body);
        free(ctx);
        *req_cls = NULL;
    }
}

int srv_http_start(struct srv *s, const struct sockaddr *addr, char *err,
                   size_t err_len)
{
    s->mhd = MHD_start_daemon(
        MHD_USE_THREAD_PER_CONNECTION | MHD_USE_INTERNAL_POLLING_THREAD |
            MHD_USE_ERROR_LOG |
            (addr->sa_family == AF_INET6 ? MHD_USE_IPv6 : 0),
        s->cfg.port, NULL, NULL, access_handler, s, MHD_OPTION_SOCK_ADDR, addr,
        MHD_OPTION_NOTIFY_COMPLETED, completed, NULL,
        MHD_OPTION_CONNECTION_LIMIT, (unsigned)s->cfg.max_connections,
        MHD_OPTION_CONNECTION_TIMEOUT, (unsigned)600,
        MHD_OPTION_CONNECTION_MEMORY_LIMIT, (size_t)(256 * 1024),
        MHD_OPTION_END);
    if (!s->mhd) {
        snprintf(err, err_len, "cannot listen on %s:%u", s->cfg.host,
                 (unsigned)s->cfg.port);
        return -1;
    }
    return 0;
}

void srv_http_stop(struct srv *s)
{
    if (s->mhd)
        MHD_stop_daemon(s->mhd);
    s->mhd = NULL;
}
