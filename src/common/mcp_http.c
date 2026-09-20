/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_http.c - a small HTTP/1.1 client (see mcp_http.h).
 */
#define _GNU_SOURCE /* strcasestr, getaddrinfo */
#include "common/mcp_http.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define HEAD_MAX 65536

int janas_url_parse(const char *url, struct janas_url *u, char *err,
                    size_t err_len)
{
    memset(u, 0, sizeof(*u));
    const char *p;
    if (strncasecmp(url, "http://", 7) == 0) {
        p = url + 7;
    } else if (strncasecmp(url, "https://", 8) == 0) {
        p = url + 8;
        u->tls = 1;
    } else {
        snprintf(err, err_len, "%s: only http:// and https:// addresses", url);
        return -1;
    }
    const char *end = p + strcspn(p, "/?#");
    const char *at = memchr(p, '@', (size_t)(end - p));
    if (at) /* user:password@ is not for a URL that is logged */
        p = at + 1;
    const char *host = p, *host_end, *port = NULL;
    if (*p == '[') { /* an IPv6 address */
        host = p + 1;
        host_end = memchr(host, ']', (size_t)(end - host));
        if (!host_end) {
            snprintf(err, err_len, "%s: an unclosed [", url);
            return -1;
        }
        if (host_end + 1 < end && host_end[1] == ':')
            port = host_end + 2;
    } else {
        host_end = memchr(p, ':', (size_t)(end - p));
        if (host_end)
            port = host_end + 1;
        else
            host_end = end;
    }
    size_t hn = (size_t)(host_end - host);
    size_t pn = port ? (size_t)(end - port) : 0;
    if (hn == 0 || hn >= sizeof(u->host) || pn >= sizeof(u->port) ||
        (port && pn == 0) || strspn(port ? port : "", "0123456789") < pn) {
        snprintf(err, err_len, "%s: no host, or a bad port", url);
        return -1;
    }
    memcpy(u->host, host, hn);
    if (port)
        memcpy(u->port, port, pn);
    else
        snprintf(u->port, sizeof(u->port), "%s", u->tls ? "443" : "80");
    const char *path = end, *frag = strchr(end, '#');
    size_t n = frag ? (size_t)(frag - path) : strlen(path);
    if (n + 2 > sizeof(u->path)) {
        snprintf(err, err_len, "%s: a path too long", url);
        return -1;
    }
    if (n == 0 || path[0] != '/')
        u->path[0] = '/';
    memcpy(u->path + strlen(u->path), path, n);
    for (const char *c = u->path; *c; c++)
        if ((unsigned char)*c <= ' ' || *c == 0x7f) {
            snprintf(err, err_len, "%s: spaces or controls in the path", url);
            return -1;
        }
    return 0;
}

/* One header line: its value when the name is name, else NULL. */
static const char *header_value(const char *line, size_t n, const char *name,
                                size_t *vn)
{
    size_t k = strlen(name);
    if (n <= k || strncasecmp(line, name, k) != 0 || line[k] != ':')
        return NULL;
    const char *v = line + k + 1, *e = line + n;
    while (v < e && (*v == ' ' || *v == '\t'))
        v++;
    while (e > v && (e[-1] == ' ' || e[-1] == '\t'))
        e--;
    *vn = (size_t)(e - v);
    return v;
}

long janas_http_head_parse(const char *p, size_t n, struct janas_http_head *h)
{
    memset(h, 0, sizeof(*h));
    h->length = -1;
    size_t end = 0;
    for (size_t i = 0; i + 1 < n; i++)
        if (p[i] == '\n' &&
            (p[i + 1] == '\n' ||
             (p[i + 1] == '\r' && i + 2 < n && p[i + 2] == '\n'))) {
            end = i + (p[i + 1] == '\n' ? 2 : 3);
            break;
        }
    if (!end)
        return n > HEAD_MAX ? -1 : 0;
    if (end < 12 || strncmp(p, "HTTP/1.", 7) != 0 || p[8] != ' ')
        return -1;
    int status = 0;
    for (int i = 9; i < 12; i++) {
        if (p[i] < '0' || p[i] > '9')
            return -1;
        status = status * 10 + (p[i] - '0');
    }
    h->status = status;
    const char *line = memchr(p, '\n', end);
    while (line && (size_t)(line - p) + 1 < end) {
        line++;
        const char *nl = memchr(line, '\n', end - (size_t)(line - p));
        if (!nl)
            break;
        size_t ln = (size_t)(nl - line);
        if (ln && line[ln - 1] == '\r')
            ln--;
        size_t vn;
        const char *v;
        if ((v = header_value(line, ln, "Content-Length", &vn))) {
            int64_t len = 0;
            if (vn == 0 || vn > 15)
                return -1;
            for (size_t i = 0; i < vn; i++) {
                if (v[i] < '0' || v[i] > '9')
                    return -1;
                len = len * 10 + (v[i] - '0');
            }
            h->length = len;
        } else if ((v = header_value(line, ln, "Transfer-Encoding", &vn))) {
            h->chunked = vn >= 7 && strncasecmp(v + vn - 7, "chunked", 7) == 0;
        } else if ((v = header_value(line, ln, "Content-Type", &vn))) {
            size_t k = vn < sizeof(h->content_type) - 1
                           ? vn
                           : sizeof(h->content_type) - 1;
            memcpy(h->content_type, v, k);
            h->content_type[k] = 0;
        } else if ((v = header_value(line, ln, "Location", &vn))) {
            if (vn < sizeof(h->location)) {
                memcpy(h->location, v, vn);
                h->location[vn] = 0;
                for (size_t i = 0; i < vn; i++) /* visible ASCII only */
                    if ((unsigned char)v[i] < 0x21 ||
                        (unsigned char)v[i] > 0x7e)
                        h->location[0] = 0;
            }
        } else if ((v = header_value(line, ln, "Mcp-Session-Id", &vn))) {
            if (vn < sizeof(h->session)) {
                memcpy(h->session, v, vn);
                h->session[vn] = 0;
                for (size_t i = 0; i < vn; i++) /* visible ASCII only */
                    if ((unsigned char)v[i] < 0x21 ||
                        (unsigned char)v[i] > 0x7e)
                        h->session[0] = 0;
            }
        }
        line = nl;
    }
    if (h->chunked)
        h->length = -1;
    if (status == 204 || status == 304 || (status >= 100 && status < 200))
        h->length = 0;
    return (long)end;
}

enum { CH_SIZE, CH_EXT, CH_DATA, CH_DATA_END, CH_TRAILER, CH_TRAILER_LINE };

int janas_http_chunks_feed(struct janas_http_chunks *c, const char *p, size_t n,
                           struct janas_buf *out)
{
    for (size_t i = 0; i < n && !c->done; i++) {
        char ch = p[i];
        switch (c->state) {
        case CH_SIZE:
        case CH_EXT: {
            int d = ch >= '0' && ch <= '9'   ? ch - '0'
                    : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10
                    : ch >= 'A' && ch <= 'F' ? ch - 'A' + 10
                                             : -1;
            if (c->state == CH_SIZE && d >= 0) {
                if (c->left >> 56)
                    return -1; /* a chunk no machine holds */
                c->left = c->left * 16 + (uint64_t)d;
            } else if (ch == '\n') {
                c->state = c->left ? CH_DATA : CH_TRAILER;
            } else if (ch == ';' || ch == ' ' || ch == '\t' || ch == '\r' ||
                       c->state == CH_EXT) {
                c->state = CH_EXT;
            } else {
                return -1;
            }
            break;
        }
        case CH_DATA: {
            size_t take = n - i < c->left ? n - i : (size_t)c->left;
            janas_buf_put(out, p + i, take);
            c->left -= take;
            i += take - 1;
            if (!c->left)
                c->state = CH_DATA_END;
            break;
        }
        case CH_DATA_END:
            if (ch == '\n')
                c->state = CH_SIZE;
            else if (ch != '\r')
                return -1;
            break;
        case CH_TRAILER: /* the start of a line: an empty one ends it all */
            if (ch == '\n')
                c->done = 1;
            else if (ch != '\r')
                c->state = CH_TRAILER_LINE;
            break;
        case CH_TRAILER_LINE:
            if (ch == '\n')
                c->state = CH_TRAILER;
            break;
        }
    }
    return 0;
}

void janas_sse_add(struct janas_sse *s, const char *p, size_t n)
{
    if (s->at > 0 && s->at > s->in.n / 2) { /* the events taken go */
        memmove(s->in.p, s->in.p + s->at, s->in.n - s->at);
        s->in.n -= s->at;
        s->at = 0;
    }
    janas_buf_put(&s->in, p, n);
}

int janas_sse_next(struct janas_sse *s, struct janas_buf *out)
{
    for (;;) {
        out->n = 0;
        size_t i = s->at;
        int data = 0;
        for (;;) {
            const char *nl =
                i < s->in.n ? memchr(s->in.p + i, '\n', s->in.n - i) : NULL;
            if (!nl)
                return 0; /* not a whole event yet */
            size_t ln = (size_t)(nl - (s->in.p + i));
            const char *line = s->in.p + i;
            i += ln + 1;
            if (ln && line[ln - 1] == '\r')
                ln--;
            if (ln == 0)
                break; /* the end of an event */
            if (ln >= 5 && memcmp(line, "data", 4) == 0 &&
                (ln == 4 || line[4] == ':')) {
                const char *v = line + 5;
                size_t vn = ln - 5;
                if (vn && *v == ' ')
                    v++, vn--;
                if (data)
                    janas_buf_put(out, "\n", 1);
                janas_buf_put(out, v, vn);
                data = 1;
            }
        }
        s->at = i;
        if (data)
            return 1;
    }
}

void janas_sse_free(struct janas_sse *s)
{
    janas_buf_free(&s->in);
    s->at = 0;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

/* Waits for the socket; 1 ready, 0 time up or cancelled, -1 error. */
static int wait_fd(int fd, short ev, double end, volatile int *cancel)
{
    for (;;) {
        if (cancel && __atomic_load_n(cancel, __ATOMIC_RELAXED))
            return 0;
        double left = end < 0 ? 200 : end - now_ms();
        if (left < 0)
            left = 0;
        struct pollfd pf = {.fd = fd, .events = ev};
        int r = poll(&pf, 1, left > 200 ? 200 : (int)left);
        if (r > 0)
            return 1;
        if (r < 0 && errno != EINTR)
            return -1;
        if (end >= 0 && now_ms() >= end)
            return 0;
    }
}

static long io_read(struct janas_http *h, char *p, size_t n)
{
    if (h->tls)
        return janas_tls_read(h->tls, p, n);
    long r = read(h->fd, p, n);
    return r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) ? -2 : r;
}

static int send_all(struct janas_http *h, const char *p, size_t n, double end,
                    volatile int *cancel)
{
    while (n > 0) {
        long w = h->tls ? janas_tls_write(h->tls, p, n)
                        : send(h->fd, p, n, MSG_NOSIGNAL);
        if (w > 0) {
            p += w;
            n -= (size_t)w;
            continue;
        }
        if (w == 0 || (w < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
                       errno != EINTR && w != -2))
            return -1;
        if (wait_fd(h->fd, POLLOUT, end, cancel) <= 0)
            return -1;
    }
    return 0;
}

static int connect_to(const struct janas_url *u, double end,
                      volatile int *cancel, char *err, size_t err_len)
{
    struct addrinfo hints = {.ai_family = AF_UNSPEC,
                             .ai_socktype = SOCK_STREAM},
                    *res = NULL;
    int e = getaddrinfo(u->host, u->port, &hints, &res);
    if (e != 0) {
        snprintf(err, err_len, "%s: %s", u->host, gai_strerror(e));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *a = res; a && fd < 0; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC | SOCK_NONBLOCK,
                    a->ai_protocol);
        if (fd < 0)
            continue;
        if (connect(fd, a->ai_addr, a->ai_addrlen) == 0)
            break;
        int ok = errno == EINPROGRESS && wait_fd(fd, POLLOUT, end, cancel) > 0;
        int so = 0;
        socklen_t sl = sizeof(so);
        if (ok && getsockopt(fd, SOL_SOCKET, SO_ERROR, &so, &sl) == 0 &&
            so == 0)
            break;
        snprintf(err, err_len, "%s:%s: %s", u->host, u->port,
                 ok ? strerror(so) : "cannot connect in time");
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

int janas_http_request(struct janas_http *h, const struct janas_url *u,
                       const char *method, const char *const *headers,
                       size_t n_headers, const char *body, size_t n,
                       int timeout_ms, volatile int *cancel, char *err,
                       size_t err_len)
{
    memset(h, 0, sizeof(*h));
    h->fd = -1;
    double end = timeout_ms < 0 ? -1 : now_ms() + timeout_ms;
    h->fd = connect_to(u, end, cancel, err, err_len);
    if (h->fd < 0)
        return -1;
    if (u->tls) {
        h->tls = janas_tls_open(
            h->fd, u->host, end < 0 ? -1 : (int)(end - now_ms()), err, err_len);
        if (!h->tls) {
            janas_http_close(h);
            return -1;
        }
    }
    struct janas_buf q = {0};
    int dflt = strcmp(u->port, u->tls ? "443" : "80") == 0;
    int v6 = strchr(u->host, ':') != NULL;
    janas_buf_printf(&q, "%s %s HTTP/1.1\r\nHost: %s%s%s%s%s\r\n", method,
                     u->path, v6 ? "[" : "", u->host, v6 ? "]" : "",
                     dflt ? "" : ":", dflt ? "" : u->port);
    janas_buf_printf(&q,
                     "Content-Length: %zu\r\nConnection: close\r\n"
                     "User-Agent: janas\r\n",
                     n);
    for (size_t i = 0; i < n_headers; i++)
        janas_buf_printf(&q, "%s\r\n", headers[i]);
    janas_buf_puts(&q, "\r\n");
    janas_buf_put(&q, body, n);
    int rc = q.oom ? -1 : send_all(h, q.p, q.n, end, cancel);
    janas_buf_free(&q);
    if (rc != 0) {
        snprintf(err, err_len, "%s: the request could not be sent", u->host);
        janas_http_close(h);
        return -1;
    }
    /* the head */
    size_t cap = 16384;
    h->buf = malloc(cap);
    for (;;) {
        if (!h->buf) {
            snprintf(err, err_len, "out of memory");
            janas_http_close(h);
            return -1;
        }
        long got = io_read(h, h->buf + h->n, cap - h->n);
        if (got == -2 || (got < 0 && errno == EINTR)) {
            if (wait_fd(h->fd, POLLIN, end, cancel) <= 0) {
                snprintf(err, err_len, "%s: no answer %s", u->host,
                         cancel && __atomic_load_n(cancel, __ATOMIC_RELAXED)
                             ? "(cancelled)"
                             : "in time");
                janas_http_close(h);
                return -1;
            }
            continue;
        }
        if (got <= 0) {
            snprintf(err, err_len, "%s: the connection closed before an answer",
                     u->host);
            janas_http_close(h);
            return -1;
        }
        h->n += (size_t)got;
        long head = janas_http_head_parse(h->buf, h->n, &h->head);
        if (head < 0) {
            snprintf(err, err_len, "%s: not an HTTP answer", u->host);
            janas_http_close(h);
            return -1;
        }
        if (head > 0) {
            h->at = (size_t)head;
            return 0;
        }
        if (h->n == cap) {
            cap *= 2;
            char *g = cap <= 2 * HEAD_MAX ? realloc(h->buf, cap) : NULL;
            if (!g)
                free(h->buf);
            h->buf = g;
        }
    }
}

/* Raw bytes of the body as they came (before the chunks are taken apart). */
static void take_raw(struct janas_http *h, const char *p, size_t n,
                     struct janas_buf *out)
{
    if (h->head.chunked) {
        if (janas_http_chunks_feed(&h->chunks, p, n, out) != 0)
            h->eof = -1;
        else if (h->chunks.done)
            h->eof = 1;
        return;
    }
    if (h->head.length >= 0 && (int64_t)n > h->head.length - h->read)
        n = (size_t)(h->head.length - h->read);
    janas_buf_put(out, p, n);
    h->read += (int64_t)n;
    if (h->head.length >= 0 && h->read >= h->head.length)
        h->eof = 1;
}

int janas_http_body(struct janas_http *h, struct janas_buf *out, int timeout_ms)
{
    size_t before = out->n;
    if (h->head.length == 0)
        return 0;
    if (h->at < h->n) { /* what came with the head */
        take_raw(h, h->buf + h->at, h->n - h->at, out);
        h->at = h->n;
    }
    double end = now_ms() + (timeout_ms < 0 ? 0 : timeout_ms);
    char chunk[16384];
    while (out->n == before && !h->eof) {
        long got = io_read(h, chunk, sizeof(chunk));
        if (got > 0) {
            take_raw(h, chunk, (size_t)got, out);
            continue;
        }
        if (got == 0) {
            /* the end of the connection ends a body of no stated length;
               one that stops short of its length is cut */
            h->eof = h->head.length < 0 && !h->head.chunked ? 1 : -1;
            break;
        }
        if (got != -2 && errno != EINTR) {
            h->eof = -1;
            break;
        }
        if (timeout_ms <= 0 || wait_fd(h->fd, POLLIN, end, NULL) <= 0)
            return out->n > before ? 1 : -2;
    }
    if (out->oom)
        return -1;
    if (out->n > before)
        return 1;
    return h->eof > 0 ? 0 : -1;
}

void janas_http_close(struct janas_http *h)
{
    if (h->tls)
        janas_tls_close(h->tls);
    h->tls = NULL;
    if (h->fd >= 0)
        close(h->fd);
    h->fd = -1;
    free(h->buf);
    h->buf = NULL;
    h->n = h->at = 0;
}

void janas_base64(struct janas_buf *b, const unsigned char *p, size_t n)
{
    static const char A[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16;
        if (i + 1 < n)
            v |= (uint32_t)p[i + 1] << 8;
        if (i + 2 < n)
            v |= p[i + 2];
        char q[4] = {A[v >> 18], A[(v >> 12) & 63],
                     i + 1 < n ? A[(v >> 6) & 63] : '=',
                     i + 2 < n ? A[v & 63] : '='};
        janas_buf_put(b, q, 4);
    }
}
