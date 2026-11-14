/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * http.c - the HTTP side of janas-server, on GNU libmicrohttpd: a thread
 * per connection, so a handler may wait for the engine without holding up
 * anybody else. Every operation of OpenAI's specification has a row in the
 * route table; the ones not written yet answer 501 and say why.
 */
#include "origin.h"
#include "server.h"
#include "server_monitor_html.h" /* made by build.sh from monitor.html */
#include "server_test_html.h"    /* and test.html */

#include <microhttpd.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

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
    if (getrandom(r, sizeof(r), 0) != (ssize_t)sizeof(r)) {
        /* no entropy to be had: still unique, from a counter and the clock */
        static _Atomic uint64_t count;
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        uint64_t x = (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
        uint64_t c = ++count;
        for (size_t i = 0; i < 8; i++)
            r[i] = (unsigned char)(x >> (8 * i));
        for (size_t i = 0; i < 4; i++)
            r[8 + i] = (unsigned char)(c >> (8 * i));
    }
    static const char hex[] = "0123456789abcdef";
    size_t w = (size_t)snprintf(buf, cap, "%s", prefix);
    if (w >= cap)
        w = cap - 1;
    for (size_t i = 0; i < sizeof(r) && w + 2 < cap; i++) {
        buf[w++] = hex[r[i] >> 4];
        buf[w++] = hex[r[i] & 15];
    }
    buf[w < cap ? w : cap - 1] = 0;
}

const char *srv_query(struct srv_req *r, const char *key)
{
    if (!r->conn) /* a request the server makes of itself: no query */
        return NULL;
    return MHD_lookup_connection_value(r->conn, MHD_GET_ARGUMENT_KIND, key);
}

/* The origin of a page named with --allow-origin that this thread's
   request comes from (a thread per connection), for the CORS headers of
   the answer; NULL otherwise. */
static _Thread_local const char *cors_origin;

static enum MHD_Result queue(struct MHD_Connection *c, unsigned status,
                             struct MHD_Response *resp, const char *type)
{
    if (!resp)
        return MHD_NO;
    if (type)
        MHD_add_response_header(resp, MHD_HTTP_HEADER_CONTENT_TYPE, type);
    if (cors_origin) {
        MHD_add_response_header(resp, "Access-Control-Allow-Origin",
                                cors_origin);
        MHD_add_response_header(resp, "Vary", "Origin");
    }
    enum MHD_Result r = MHD_queue_response(c, status, resp);
    MHD_destroy_response(resp);
    return r;
}

/* The answer kept (r->cap): a copy of n bytes. MHD_YES. */
static int capture(struct srv_req *r, unsigned status, const char *body,
                   size_t n)
{
    free(r->cap->body);
    r->cap->body = malloc(n + 1);
    r->cap->status = r->cap->body ? status : 500;
    r->cap->n = r->cap->body ? n : 0;
    if (r->cap->body) {
        memcpy(r->cap->body, body, n);
        r->cap->body[n] = 0;
    }
    return MHD_YES;
}

int srv_reply_json(struct srv_req *r, unsigned status, char *json, size_t n)
{
    if (!json)
        return srv_reply_error(r, 500, "server_error", NULL,
                               "cannot write the answer");
    if (r->cap) {
        capture(r, status, json, n);
        free(json);
        return MHD_YES;
    }
    struct MHD_Response *resp = MHD_create_response_from_buffer_copy(n, json);
    free(json);
    return queue(r->conn, status, resp, "application/json");
}

int srv_reply_bytes(struct srv_req *r, unsigned status, char *body, size_t n,
                    const char *type)
{
    if (!body)
        return srv_reply_error(r, 500, "server_error", NULL,
                               "cannot write the answer");
    if (r->cap) {
        capture(r, status, body, n);
        free(body);
        return MHD_YES;
    }
    struct MHD_Response *resp = MHD_create_response_from_buffer_copy(n, body);
    free(body);
    return queue(r->conn, status, resp, type);
}

int srv_reply_fd(struct srv_req *r, int fd, uint64_t n, const char *type)
{
    if (r->cap) { /* no request the server makes of itself wants bytes */
        close(fd);
        return srv_reply_error(r, 400, "invalid_request_error", NULL,
                               "not in a batch");
    }
    struct MHD_Response *resp = MHD_create_response_from_fd64(n, fd);
    if (!resp) {
        close(fd);
        return srv_reply_error(r, 500, "server_error", NULL,
                               "cannot send the file");
    }
    return queue(r->conn, 200, resp, type);
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
    if (r->srv->cfg.verbose)
        fprintf(stderr, "janas-server: %s %s: %u %s\n", r->method, r->path,
                status, msg);
    if (r->cap)
        return capture(r, status, out, (size_t)n);
    struct MHD_Response *resp =
        MHD_create_response_from_buffer_copy((size_t)n, out);
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
    if (r->cap)
        return capture(r, 400, out, (size_t)n);
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
    return srv_reply_stream_as(r, next, done, ctx, "text/event-stream");
}

int srv_reply_stream_as(struct srv_req *r, srv_stream_next next,
                        srv_stream_done done, void *ctx, const char *type)
{
    struct stream *s = malloc(sizeof(*s));
    if (!s) {
        done(ctx);
        return MHD_NO;
    }
    *s = (struct stream){next, done, ctx};
    if (r->cap) { /* the whole stream, kept, or handed on as it comes */
        char *all = NULL, buf[4096];
        size_t n = 0;
        long k;
        while ((k = next(ctx, buf, sizeof buf)) >= 0) {
            if (r->cap->on_data) {
                if (k)
                    r->cap->on_data(r->cap->arg, buf, (size_t)k);
                continue;
            }
            char *t = k ? realloc(all, n + (size_t)k + 1) : all;
            if (k && !t)
                continue;
            all = t;
            if (k)
                memcpy(all + n, buf, (size_t)k), n += (size_t)k;
        }
        stream_free(s);
        capture(r, 200, all ? all : "", n);
        free(all);
        return MHD_YES;
    }
    struct MHD_Response *resp = MHD_create_response_from_callback(
        MHD_SIZE_UNKNOWN, 4096, stream_reader, s, stream_free);
    if (!resp) {
        stream_free(s);
        return MHD_NO;
    }
    MHD_add_response_header(resp, "Cache-Control", "no-cache");
    return queue(r->conn, 200, resp, type);
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
    if (h && strncmp(h, "Bearer ", 7) == 0)
        h += 7;
    else /* Anthropic's clients send it as x-api-key */
        h = MHD_lookup_connection_value(r->conn, MHD_HEADER_KIND, "x-api-key");
    if (!h)
        return 0;
    size_t a = strlen(h), b = strlen(want);
    unsigned diff = (unsigned)(a ^ b);
    for (size_t i = 0; i < b; i++)
        diff |= (unsigned char)want[i] ^ (unsigned char)(i < a ? h[i] : 0);
    return diff == 0;
}

srv_handler srv_route_find(const char *method, const char *path)
{
    struct srv_req r = {0};
    for (size_t i = 0; i < sizeof(routes) / sizeof(routes[0]); i++)
        if (routes[i].h && !strcmp(routes[i].method, method) &&
            match(routes[i].path, path, &r))
            return routes[i].h;
    return NULL;
}

static enum MHD_Result dispatch(struct srv_req *r)
{
    const char *origin =
        MHD_lookup_connection_value(r->conn, MHD_HEADER_KIND, "Origin");
    int o = srv_origin_check(
        &r->srv->cfg, origin,
        MHD_lookup_connection_value(r->conn, MHD_HEADER_KIND, "Host"));
    if (o == SRV_ORIGIN_REFUSED)
        return (enum MHD_Result)srv_reply_error(
            r, 403, "invalid_request_error", "origin_not_allowed",
            "Pages of %s may not call this server: start it with "
            "--allow-origin %s to let them.",
            origin, origin);
    cors_origin = o == SRV_ORIGIN_LISTED ? origin : NULL;
    if (o == SRV_ORIGIN_LISTED && strcmp(r->method, "OPTIONS") == 0) {
        /* the browser's question before the request itself */
        struct MHD_Response *resp =
            MHD_create_response_from_buffer_static(0, "");
        if (!resp)
            return MHD_NO;
        MHD_add_response_header(resp, "Access-Control-Allow-Methods",
                                "GET, POST, DELETE, OPTIONS");
        MHD_add_response_header(
            resp, "Access-Control-Allow-Headers",
            srv_allow_headers(MHD_lookup_connection_value(
                r->conn, MHD_HEADER_KIND, "Access-Control-Request-Headers")));
        MHD_add_response_header(resp, "Access-Control-Max-Age", "600");
        return queue(r->conn, 204, resp, NULL);
    }
    if (strcmp(r->path, "/") == 0 &&
        (strcmp(r->method, "GET") == 0 || strcmp(r->method, "HEAD") == 0)) {
        /* where Ollama's clients look for a server */
        static const char up[] = "Janas-LLM is running";
        return queue(r->conn, 200,
                     MHD_create_response_from_buffer_static(sizeof(up) - 1, up),
                     "text/plain; charset=utf-8");
    }
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
    if (r->srv->cfg.monitor && strcmp(r->method, "GET") == 0 &&
        (strcmp(r->path, "/monitor") == 0 ||
         strcmp(r->path, "/monitor/") == 0)) {
        /* as /test: the page asks for the key; its data is behind it */
        struct MHD_Response *resp = MHD_create_response_from_buffer_static(
            sizeof(janas_server_monitor_html) - 1, janas_server_monitor_html);
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
    /* llama.cpp's route for filling in code, which editors' extensions
       made for it ask: not OpenAI's, so not in routes.inc */
    if (strcmp(r->path, "/infill") == 0 && strcmp(r->method, "POST") == 0)
        return (enum MHD_Result)srv_infill(r);
    if (r->srv->cfg.monitor && strcmp(r->method, "GET") == 0 &&
        strcmp(r->path, "/monitor/events") == 0)
        return (enum MHD_Result)srv_monitor_events(r);
    /* Ollama's API, for the programs that know a local model only as
       Ollama's (ollama.c) */
    srv_handler ol = srv_ollama_route(r->method, r->path);
    if (ol)
        return (enum MHD_Result)ol(r);
    /* Anthropic's Messages API, for Claude Code and the clients made for
       it (anthropic.c) */
    if (strcmp(r->method, "POST") == 0 && strcmp(r->path, "/messages") == 0)
        return (enum MHD_Result)srv_anthropic_messages(r);
    if (strcmp(r->method, "POST") == 0 &&
        strcmp(r->path, "/messages/count_tokens") == 0)
        return (enum MHD_Result)srv_anthropic_count(r);
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

/* The largest body a request may have: an upload's part may hold 64 MiB
   (and its multipart envelope) whatever --max-body says. */
static size_t body_limit(const struct srv *srv, const char *url)
{
    size_t n = strlen(url);
    if (strncmp(url, "/v1/", 4) == 0)
        url += 3, n -= 3;
    if (strncmp(url, "/uploads/", 9) == 0 && n > 15 &&
        strcmp(url + n - 6, "/parts") == 0 &&
        srv->cfg.max_body < SRV_UPLOAD_PART_MAX + (1 << 20))
        return SRV_UPLOAD_PART_MAX + (1 << 20);
    return srv->cfg.max_body;
}

/* requests between their headers and their end: at stop, the server waits
   a little for them to answer */
static atomic_int in_flight;

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
        atomic_fetch_add(&in_flight, 1);
        return MHD_YES;
    }
    if (*upload_data_size) { /* a piece of the body */
        size_t n = *upload_data_size;
        *upload_data_size = 0;
        size_t limit = body_limit(srv, url);
        if (ctx->too_big || ctx->n + n > limit) {
            ctx->too_big = 1;
            return MHD_YES;
        }
        if (ctx->n + n + 1 > ctx->cap) {
            size_t cap = 2 * (ctx->n + n) + 1024;
            if (cap > limit + 1)
                cap = limit + 1;
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
    cors_origin = NULL; /* until dispatch knows the page */
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
            "The request body is larger than %zu bytes.", body_limit(srv, url));
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
        atomic_fetch_sub(&in_flight, 1);
    }
}

int srv_http_start(struct srv *s, const struct sockaddr *addr, char *err,
                   size_t err_len)
{
    /* with HTTPS the server listens itself (tls.c) and hands
       libmicrohttpd connections in clear */
    int tls = s->cfg.tls_cert != NULL;
    unsigned flags = MHD_USE_THREAD_PER_CONNECTION |
                     MHD_USE_INTERNAL_POLLING_THREAD | MHD_USE_ERROR_LOG |
                     (addr->sa_family == AF_INET6 ? MHD_USE_IPv6 : 0);
    if (tls) /* no socket of its own: tls.c adds the connections */
        s->mhd = MHD_start_daemon(
            flags | MHD_USE_NO_LISTEN_SOCKET | MHD_USE_ITC, 0, NULL, NULL,
            access_handler, s, MHD_OPTION_NOTIFY_COMPLETED, completed, NULL,
            MHD_OPTION_CONNECTION_LIMIT, (unsigned)s->cfg.max_connections,
            MHD_OPTION_CONNECTION_TIMEOUT, (unsigned)600,
            MHD_OPTION_CONNECTION_MEMORY_LIMIT, (size_t)(256 * 1024),
            MHD_OPTION_END);
    else
        s->mhd = MHD_start_daemon(
            flags, s->cfg.port, NULL, NULL, access_handler, s,
            MHD_OPTION_SOCK_ADDR, addr, MHD_OPTION_NOTIFY_COMPLETED, completed,
            NULL, MHD_OPTION_CONNECTION_LIMIT, (unsigned)s->cfg.max_connections,
            MHD_OPTION_CONNECTION_TIMEOUT, (unsigned)600,
            MHD_OPTION_CONNECTION_MEMORY_LIMIT, (size_t)(256 * 1024),
            MHD_OPTION_END);
    if (!s->mhd) {
        snprintf(err, err_len, "cannot listen on %s:%u", s->cfg.host,
                 (unsigned)s->cfg.port);
        return -1;
    }
    if (tls && srv_tls_start(s, addr, err, err_len) != 0) {
        MHD_stop_daemon(s->mhd);
        s->mhd = NULL;
        return -1;
    }
    return 0;
}

void srv_http_stop(struct srv *s)
{
    if (s->cfg.tls_cert)
        srv_tls_stop();
    if (s->mhd) {
        /* the requests running answered first, for 2 s at most: their
           replies are cut short and new ones refused by the engine
           (srv_engine_cancel_clients) */
        for (int i = 0; i < 100 && atomic_load(&in_flight) > 0; i++)
            usleep(20000);
        MHD_stop_daemon(s->mhd);
    }
    s->mhd = NULL;
}
