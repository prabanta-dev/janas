/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * proxy.c - the page's requests to the model's API (/v1/..., /api/...)
 * passed on to the model's server, and its answer passed back as it comes
 * (a stream of events too), so that the page talks to one origin only.
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

int brebus_proxy(struct brebus *b, struct MHD_Connection *conn,
                 const char *method, const char *url, const char *body,
                 size_t n, struct MHD_Response **resp, unsigned *status)
{
    unsigned short port = brebus_server_port(b);
    if (!port)
        return -1;
    char to[4200], err[160];
    snprintf(to, sizeof to, "http://127.0.0.1:%u%s", port, url);
    struct janas_url u;
    if (janas_url_parse(to, &u, err, sizeof err) != 0)
        return -1;
    /* the headers a client of the API sends that matter to it */
    static const char *const keep[] = {"Content-Type", "Authorization",
                                       "Accept", "Anthropic-Version",
                                       "X-Api-Key"};
    char hv[5][1024];
    const char *hdr[5];
    size_t nh = 0;
    for (size_t i = 0; i < sizeof keep / sizeof keep[0]; i++) {
        const char *v =
            MHD_lookup_connection_value(conn, MHD_HEADER_KIND, keep[i]);
        if (v && strlen(v) < 900) {
            snprintf(hv[nh], sizeof hv[nh], "%s: %s", keep[i], v);
            hdr[nh] = hv[nh];
            nh++;
        }
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
    MHD_add_response_header(*resp, "Cache-Control", "no-store");
    return 0;
}
