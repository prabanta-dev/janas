/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * proxy.c - the page's requests to the model's API (/v1/..., /api/...)
 * passed on to the server of the model they name (server.c), and its answer
 * passed back as it comes (a stream of events too), so that the page talks to
 * one origin only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <microhttpd.h>

#include "brebus.h"
#include "common/mcp_http.h"

struct pass {
    struct janas_http h;
    struct janas_buf buf;
    size_t at;
    int done;
};

static ssize_t reader(void *cls, uint64_t pos, char *out, size_t max)
{
    (void)pos;
    struct pass *p = cls;
    while (p->at >= p->buf.n) {
        if (p->done)
            return MHD_CONTENT_READER_END_OF_STREAM;
        p->buf.n = 0;
        p->at = 0;
        int r = janas_http_body(&p->h, &p->buf, 600000);
        if (r == 0)
            p->done = 1;
        else if (r < 0)
            return MHD_CONTENT_READER_END_WITH_ERROR;
    }
    size_t k = p->buf.n - p->at < max ? p->buf.n - p->at : max;
    memcpy(out, p->buf.p + p->at, k);
    p->at += k;
    return (ssize_t)k;
}

static void done(void *cls)
{
    struct pass *p = cls;
    janas_http_close(&p->h);
    janas_buf_free(&p->buf);
    free(p);
}

/* the request's query, which libmicrohttpd keeps apart from its path,
   made again (%-encoded) for the server: ?layers=1 of the monitor */
struct query {
    char *p;
    size_t n, cap;
    int bad;
};

static void q_put(struct query *q, const char *s)
{
    static const char hex[] = "0123456789ABCDEF";
    for (; *s && !q->bad; s++) {
        unsigned char c = (unsigned char)*s;
        int plain = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                    c == '.' || c == '~';
        if (q->n + 4 >= q->cap) {
            q->bad = 1;
            break;
        }
        if (plain)
            q->p[q->n++] = (char)c;
        else {
            q->p[q->n++] = '%';
            q->p[q->n++] = hex[c >> 4];
            q->p[q->n++] = hex[c & 15];
        }
    }
    q->p[q->n] = 0;
}

static enum MHD_Result q_arg(void *cls, enum MHD_ValueKind kind,
                             const char *key, const char *value)
{
    (void)kind;
    struct query *q = cls;
    if (q->n + 2 >= q->cap) {
        q->bad = 1;
        return MHD_NO;
    }
    char sep = q->n ? '&' : '?';
    q->p[q->n++] = sep;
    q->p[q->n] = 0;
    q_put(q, key);
    if (value) {
        if (q->n + 2 >= q->cap) {
            q->bad = 1;
            return MHD_NO;
        }
        q->p[q->n++] = '=';
        q->p[q->n] = 0;
        q_put(q, value);
    }
    return MHD_YES;
}

/* The request passed on to 127.0.0.1:port (over HTTPS when tls), its
   answer passed back as it comes: with the key as a bearer when the
   client sent none, and the header extra ("Name: value", or NULL). 0, or
   -1 when it could not be passed on. */
int brebus_proxy_to(struct brebus *b, struct MHD_Connection *conn,
                    const char *method, const char *url, const char *body,
                    size_t n, unsigned short port, int tls, const char *key,
                    const char *extra, struct MHD_Response **resp,
                    unsigned *status)
{
    (void)b;
    char to[4200], err[160], qs[2048] = "";
    struct query q = {.p = qs, .cap = sizeof qs};
    MHD_get_connection_values(conn, MHD_GET_ARGUMENT_KIND, q_arg, &q);
    if (q.bad)
        return -1;
    const char *sch = tls ? "https" : "http";
    snprintf(to, sizeof to, "%s://127.0.0.1:%u%s%s", sch, port, url, qs);
    struct janas_url u;
    if (janas_url_parse(to, &u, err, sizeof err) != 0)
        return -1;
    /* the headers a client of the API sends that matter to it */
    static const char *const keep[] = {"Content-Type", "Authorization",
                                       "Accept", "Anthropic-Version",
                                       "X-Api-Key"};
    char hv[9][1024];
    const char *hdr[9];
    size_t nh = 0;
    int has_auth = 0;
    for (size_t i = 0; i < sizeof keep / sizeof keep[0]; i++) {
        const char *v =
            MHD_lookup_connection_value(conn, MHD_HEADER_KIND, keep[i]);
        if (v && strlen(v) < 900) {
            has_auth |= i == 1;
            snprintf(hv[nh], sizeof hv[nh], "%s: %s", keep[i], v);
            hdr[nh] = hv[nh];
            nh++;
        }
    }
    /* the page's own requests carry no key: Brebus, reached only from this
       computer and its own page, adds the server's */
    /* the extra headers, a line each */
    for (const char *e = extra; e && *e && nh < 8;) {
        const char *nl = strchr(e, '\n');
        size_t len = nl ? (size_t)(nl - e) : strlen(e);
        if (len && len < sizeof hv[0]) {
            memcpy(hv[nh], e, len);
            hv[nh][len] = 0;
            hdr[nh] = hv[nh];
            nh++;
        }
        e += len + (nl ? 1 : 0);
    }
    if (!has_auth && key && key[0] && nh < 9) {
        snprintf(hv[nh], sizeof hv[nh], "Authorization: Bearer %s", key);
        hdr[nh] = hv[nh];
        nh++;
    }
    struct pass *p = calloc(1, sizeof(*p));
    if (!p)
        return -1;
    if (janas_http_request(&p->h, &u, method, hdr, nh, body, n, -1, NULL, err,
                           sizeof err) != 0) {
        free(p);
        return -1;
    }
    *status = (unsigned)p->h.head.status;
    *resp = MHD_create_response_from_callback(MHD_SIZE_UNKNOWN, 16384, reader,
                                              p, done);
    if (!*resp) {
        done(p);
        return -1;
    }
    if (p->h.head.content_type[0])
        MHD_add_response_header(*resp, "Content-Type", p->h.head.content_type);
    if (p->h.head.csp[0])
        MHD_add_response_header(*resp, "Content-Security-Policy",
                                p->h.head.csp);
    else if (!strncmp(p->h.head.content_type, "text/html", 9))
        /* a page without its own policy runs nothing at Brebus's origin */
        MHD_add_response_header(*resp, "Content-Security-Policy",
                                "default-src 'none'; frame-ancestors 'none'");
    MHD_add_response_header(*resp, "X-Content-Type-Options", "nosniff");
    MHD_add_response_header(*resp, "Referrer-Policy", "no-referrer");
    if (b->tls)
        MHD_add_response_header(*resp, "Strict-Transport-Security",
                                "max-age=31536000");
    if (p->h.head.disposition[0])
        MHD_add_response_header(*resp, "Content-Disposition",
                                p->h.head.disposition);
    if (p->h.head.location[0])
        MHD_add_response_header(*resp, "Location", p->h.head.location);
    MHD_add_response_header(*resp, "Cache-Control", "no-store");
    return 0;
}

int brebus_proxy(struct brebus *b, struct MHD_Connection *conn,
                 const char *method, const char *url, const char *body,
                 size_t n, struct MHD_Response **resp, unsigned *status)
{
    char err[160], model[256] = "", key[256];
    /* to the server of the model the request names: several may be
       loaded; one that names none goes to the one used last */
    struct janas_json_doc *d = body && n && *body == '{'
                                   ? janas_json_parse(body, n, err, sizeof err)
                                   : NULL;
    const char *mn =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "model")) : NULL;
    if (mn && strlen(mn) < sizeof model)
        snprintf(model, sizeof model, "%s", mn);
    janas_json_free(d);
    int tls = 0;
    unsigned short port =
        brebus_server_route(b, model[0] ? model : NULL, &tls, key, sizeof key);
    if (!port || !brebus_port_mine(port))
        return -1;
    return brebus_proxy_to(b, conn, method, url, body, n, port, tls, key, NULL,
                           resp, status);
}
