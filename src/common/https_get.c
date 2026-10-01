/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * https_get.c - a GET over HTTPS for Janas's services, with the HTTP client
 * and TLS of the MCP transport (common/mcp_http.c, mcp_tls.c).
 */
#include <stdio.h>
#include <string.h>

#include "https_get.h"

#include "common/mcp_http.h"

#define TIMEOUT_MS 15000
#define MAX_REDIRECTS 4
#define MAX_BODY (16u << 20)

int janas_https_get(const char *url, const char *const *headers,
                    size_t n_headers, struct janas_buf *out, char *err,
                    size_t err_len)
{
    char where[4096];
    snprintf(where, sizeof where, "%s", url);
    for (int hop = 0; hop <= MAX_REDIRECTS; hop++) {
        struct janas_url u;
        if (janas_url_parse(where, &u, err, err_len) != 0)
            return -1;
        struct janas_http h;
        if (janas_http_request(&h, &u, "GET", headers, n_headers, NULL, 0,
                               TIMEOUT_MS, NULL, err, err_len) != 0)
            return -1;
        int status = h.head.status;
        if (status >= 300 && status < 400 && h.head.location[0]) {
            snprintf(where, sizeof where, "%s", h.head.location);
            janas_http_close(&h);
            continue;
        }
        out->n = 0;
        int r;
        while ((r = janas_http_body(&h, out, TIMEOUT_MS)) == 1)
            if (out->n > MAX_BODY) {
                r = -1;
                break;
            }
        janas_http_close(&h);
        if (r != 0 || out->oom) {
            snprintf(err, err_len, "%s: the answer broke off", u.host);
            return -1;
        }
        janas_buf_put(out, "", 1); /* a string, for the JSON reader */
        out->n--;
        return status;
    }
    snprintf(err, err_len, "too many redirects");
    return -1;
}
