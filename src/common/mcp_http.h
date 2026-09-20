/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_http.h - a small HTTP/1.1 client, for the Streamable HTTP transport
 * of MCP: one request per connection, the body read as it comes (by
 * length, in chunks, or to the end of the connection), and the pieces of a
 * text/event-stream taken apart.
 *
 * What a server sends is input from outside: the head, the chunks and the
 * events are read by functions on buffers, which tests/test_mcp.c fuzzes.
 */
#ifndef JANAS_MCP_HTTP_H
#define JANAS_MCP_HTTP_H

#include <stddef.h>
#include <stdint.h>

#include "common/mcp_tls.h"
#include "llm/json.h"

struct janas_url {
    int tls; /* https */
    char host[256];
    char port[8];
    char path[4096]; /* with the query, "/" at least (a signed CDN
                        address runs past a thousand) */
};

/* 0, or -1 with the reason in err. Only http:// and https://. */
int janas_url_parse(const char *url, struct janas_url *u, char *err,
                    size_t err_len);

/* The head of a response, read from its text up to the blank line. */
struct janas_http_head {
    int status;
    int chunked;
    int64_t length; /* -1: to the end of the connection */
    char content_type[128];
    char session[256];   /* Mcp-Session-Id */
    char location[4096]; /* of a redirect, visible ASCII, or empty */
};

/* The head in the first n bytes: its length (up to and with the blank
   line), 0 when it is not all there yet, -1 when it is no HTTP head. */
long janas_http_head_parse(const char *p, size_t n, struct janas_http_head *h);

/* Chunked transfer coding, taken apart as bytes arrive. */
struct janas_http_chunks {
    int state;     /* where in a chunk the bytes are */
    uint64_t left; /* bytes of data left in this chunk */
    int done;
};

/* Takes n bytes, puts the data they hold into out; -1 on a malformed
   stream. */
int janas_http_chunks_feed(struct janas_http_chunks *c, const char *p, size_t n,
                           struct janas_buf *out);

/* A text/event-stream: the data of each complete event, in order. The
   caller takes the events (janas_sse_next) as bytes are added. */
struct janas_sse {
    struct janas_buf in; /* bytes not yet made into events */
    size_t at;           /* where the next event starts in it */
};

void janas_sse_add(struct janas_sse *s, const char *p, size_t n);
/* The data of the next complete event into out (lines joined by "\n"):
   1, or 0 when none is complete yet. Events with no data are skipped. */
int janas_sse_next(struct janas_sse *s, struct janas_buf *out);
void janas_sse_free(struct janas_sse *s);

/* A request made, its response being read. */
struct janas_http {
    int fd;
    struct janas_tls *tls; /* for https */
    struct janas_http_head head;
    char *buf; /* body bytes read with the head */
    size_t n, at;
    int64_t read;
    struct janas_http_chunks chunks;
    int eof;
};

/*
 * Connects, sends the request (method, the path of u, the header lines
 * given as "Name: value", a body of n bytes) and reads the head of the
 * response. 0, or -1 with the reason in err. Waits at most timeout_ms in
 * all (< 0: no limit); *cancel, when it becomes set, stops the wait.
 */
int janas_http_request(struct janas_http *h, const struct janas_url *u,
                       const char *method, const char *const *headers,
                       size_t n_headers, const char *body, size_t n,
                       int timeout_ms, volatile int *cancel, char *err,
                       size_t err_len);

/* The body, as it comes: the bytes received into out. 1 with some, 0 at
   its end, -1 on an error, -2 when timeout_ms passed with nothing (0: only
   what has arrived). */
int janas_http_body(struct janas_http *h, struct janas_buf *out,
                    int timeout_ms);

void janas_http_close(struct janas_http *h);

/* Base64 of n bytes, for the header values that need it. */
void janas_base64(struct janas_buf *b, const unsigned char *p, size_t n);

#endif
