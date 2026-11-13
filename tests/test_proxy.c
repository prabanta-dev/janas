/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_proxy.c - the HTTP client through a proxy (src/common/mcp_http.c,
 * issue #70), with a proxy and a server of its own in threads, no network:
 * plain http through it (the whole address on the request line, Basic
 * credentials %-decoded), https through a CONNECT (refused without
 * credentials, 407 said; with them the tunnel made, the TLS that follows
 * failing on a server that is not one), no_proxy ("*", a domain and the
 * names under it), a policy of public addresses going directly, a proxy
 * that is not http://, curl's port 1080 when none is given. POSIX only.
 */
#ifdef _WIN32
#include <stdio.h>
int main(void)
{
    printf("test_proxy: skipped on Windows\n");
    return 0;
}
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/mcp_http.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static int origin_port, proxy_port;
static char seen_line[512], seen_auth[256];
static int proxied; /* connections the proxy took */

static int listen_on(int *port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET,
                            .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    socklen_t al = sizeof a;
    if (fd < 0 || bind(fd, (struct sockaddr *)&a, sizeof a) != 0 ||
        listen(fd, 16) != 0 || getsockname(fd, (struct sockaddr *)&a, &al))
        return -1;
    *port = ntohs(a.sin_port);
    return fd;
}

static int dial(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET,
                            .sin_port = htons((uint16_t)port),
                            .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    if (fd >= 0 && connect(fd, (struct sockaddr *)&a, sizeof a) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

/* A request's head, up to its blank line (a byte at a time: what follows
   is the tunnel's); its length, or -1. A TLS hello is not one: -1. */
static long read_head(int fd, char *buf, size_t cap)
{
    size_t n = 0;
    while (n < cap - 1) {
        if (recv(fd, buf + n, 1, 0) != 1)
            return -1;
        if (n == 0 && (unsigned char)buf[0] == 0x16)
            return -1;
        n++;
        if (n >= 4 && !memcmp(buf + n - 4, "\r\n\r\n", 4)) {
            buf[n] = 0;
            return (long)n;
        }
    }
    return -1;
}

static void *origin(void *arg)
{
    int lfd = *(int *)arg, fd;
    while ((fd = accept(lfd, NULL, NULL)) >= 0) {
        char head[4096];
        if (read_head(fd, head, sizeof head) > 0) {
            static const char ok[] = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n"
                                     "Connection: close\r\n\r\nhello";
            send(fd, ok, sizeof ok - 1, 0);
        }
        close(fd);
    }
    return NULL;
}

/* Both ways until either end closes. */
static void pump(int a, int b)
{
    char buf[4096];
    for (;;) {
        struct pollfd p[2] = {{.fd = a, .events = POLLIN},
                              {.fd = b, .events = POLLIN}};
        if (poll(p, 2, 5000) <= 0)
            return;
        for (int i = 0; i < 2; i++)
            if (p[i].revents) {
                long n = recv(p[i].fd, buf, sizeof buf, 0);
                if (n <= 0 || send(i ? a : b, buf, (size_t)n, 0) != n)
                    return;
            }
    }
}

static void *proxy(void *arg)
{
    int lfd = *(int *)arg, fd;
    while ((fd = accept(lfd, NULL, NULL)) >= 0) {
        __atomic_add_fetch(&proxied, 1, __ATOMIC_SEQ_CST);
        char head[4096];
        if (read_head(fd, head, sizeof head) <= 0) {
            close(fd);
            continue;
        }
        snprintf(seen_line, sizeof seen_line, "%.*s",
                 (int)strcspn(head, "\r\n"), head);
        const char *au = strcasestr(head, "\r\nProxy-Authorization: ");
        snprintf(seen_auth, sizeof seen_auth, "%.*s",
                 au ? (int)strcspn(au + 23, "\r\n") : 0, au ? au + 23 : "");
        int o = dial(origin_port);
        if (!strncmp(head, "CONNECT ", 8)) {
            if (!au) {
                static const char no[] =
                    "HTTP/1.1 407 Proxy Authentication Required\r\n"
                    "Content-Length: 0\r\n\r\n";
                send(fd, no, sizeof no - 1, 0);
            } else {
                static const char yes[] = "HTTP/1.1 200 Connection established"
                                          "\r\n\r\n";
                send(fd, yes, sizeof yes - 1, 0);
                pump(fd, o);
            }
        } else {
            send(o, head, strlen(head), 0);
            pump(fd, o);
        }
        close(o);
        close(fd);
    }
    return NULL;
}

/* A request to url; the body in body, or the error in err. 0 or -1. */
static int get(const char *url, const struct janas_net_policy *pol, char *body,
               size_t cap, char *err, size_t err_len)
{
    struct janas_url u;
    struct janas_http h;
    *body = *err = 0;
    if (janas_url_parse(url, &u, err, err_len) != 0)
        return -1;
    u.policy = pol;
    if (janas_http_request(&h, &u, "GET", NULL, 0, NULL, 0, 5000, NULL, err,
                           err_len) != 0)
        return -1;
    struct janas_buf out = {0};
    while (janas_http_body(&h, &out, 5000) == 1)
        ;
    snprintf(body, cap, "%.*s", (int)out.n, out.p ? out.p : "");
    janas_buf_free(&out);
    janas_http_close(&h);
    return 0;
}

static void env(const char *name, const char *fmt, int port)
{
    char v[256];
    snprintf(v, sizeof v, fmt, port);
    setenv(name, v, 1);
}

int main(void)
{
    static const char *const VARS[] = {
        "http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
        "all_proxy",  "ALL_PROXY",  "no_proxy",    "NO_PROXY"};
    for (size_t i = 0; i < sizeof VARS / sizeof *VARS; i++)
        unsetenv(VARS[i]);
    int ofd = listen_on(&origin_port), pfd = listen_on(&proxy_port);
    if (ofd < 0 || pfd < 0) {
        printf("test_proxy: cannot listen on 127.0.0.1\n");
        return 1;
    }
    pthread_t to, tp;
    pthread_create(&to, NULL, origin, &ofd);
    pthread_create(&tp, NULL, proxy, &pfd);
    char body[256], err[300];

    /* plain http: the whole address, the credentials %-decoded */
    env("http_proxy", "http://user:p%%40ss@127.0.0.1:%d", proxy_port);
    CHECK(get("http://janas-test.invalid/x?y=1", NULL, body, sizeof body, err,
              sizeof err) == 0 &&
              !strcmp(body, "hello"),
          "http through the proxy: [%s] %s", body, err);
    CHECK(!strcmp(seen_line, "GET http://janas-test.invalid:80/x?y=1 HTTP/1.1"),
          "request line [%s]", seen_line);
    CHECK(!strcmp(seen_auth, "Basic dXNlcjpwQHNz"), "credentials [%s]",
          seen_auth);
    /* HTTP_PROXY alone is not taken, as curl does not */
    unsetenv("http_proxy");
    env("HTTP_PROXY", "http://127.0.0.1:%d", proxy_port);
    int before = __atomic_load_n(&proxied, __ATOMIC_SEQ_CST);
    get("http://janas-test.invalid/", NULL, body, sizeof body, err, sizeof err);
    CHECK(__atomic_load_n(&proxied, __ATOMIC_SEQ_CST) == before,
          "HTTP_PROXY was taken");
    unsetenv("HTTP_PROXY");

    /* https: CONNECT, refused without credentials */
    env("https_proxy", "127.0.0.1:%d", proxy_port);
    CHECK(get("https://janas-test.invalid/", NULL, body, sizeof body, err,
              sizeof err) != 0 &&
              strstr(err, "407") && strstr(err, "user and password"),
          "a CONNECT refused: [%s]", err);
    CHECK(!strcmp(seen_line, "CONNECT janas-test.invalid:443 HTTP/1.1"),
          "CONNECT line [%s]", seen_line);
    /* with them, the tunnel: the TLS after it is what fails */
    env("HTTPS_PROXY", "http://me:pw@127.0.0.1:%d", proxy_port);
    unsetenv("https_proxy");
    CHECK(get("https://janas-test.invalid:8443/", NULL, body, sizeof body, err,
              sizeof err) != 0 &&
              !strstr(err, "proxy"),
          "the tunnel made, then TLS: [%s]", err);
    CHECK(!strcmp(seen_line, "CONNECT janas-test.invalid:8443 HTTP/1.1") &&
              !strcmp(seen_auth, "Basic bWU6cHc="),
          "CONNECT [%s] [%s]", seen_line, seen_auth);

    /* directly: no_proxy, this machine, a policy of public addresses */
    const char *nps[] = {"*", "invalid", ".invalid",
                         "x.org, janas-test.invalid:443"};
    for (size_t i = 0; i < sizeof nps / sizeof *nps; i++) {
        setenv("no_proxy", nps[i], 1);
        before = __atomic_load_n(&proxied, __ATOMIC_SEQ_CST);
        get("https://janas-test.invalid/", NULL, body, sizeof body, err,
            sizeof err);
        CHECK(__atomic_load_n(&proxied, __ATOMIC_SEQ_CST) == before,
              "no_proxy=%s went through the proxy", nps[i]);
    }
    setenv("no_proxy", "test.invalid", 1); /* not a name under it */
    before = __atomic_load_n(&proxied, __ATOMIC_SEQ_CST);
    get("https://janas-test.invalid/", NULL, body, sizeof body, err,
        sizeof err);
    CHECK(__atomic_load_n(&proxied, __ATOMIC_SEQ_CST) == before + 1,
          "no_proxy=test.invalid kept janas-test.invalid from the proxy");
    unsetenv("no_proxy");
    char url[64];
    snprintf(url, sizeof url, "http://127.0.0.1:%d/", origin_port);
    env("http_proxy", "http://127.0.0.1:%d", proxy_port);
    before = __atomic_load_n(&proxied, __ATOMIC_SEQ_CST);
    CHECK(get(url, NULL, body, sizeof body, err, sizeof err) == 0 &&
              !strcmp(body, "hello") &&
              __atomic_load_n(&proxied, __ATOMIC_SEQ_CST) == before,
          "this machine through the proxy: [%s] %s", body, err);
    struct janas_net_policy pub = {.public_only = 1};
    before = __atomic_load_n(&proxied, __ATOMIC_SEQ_CST);
    get("https://janas-test.invalid/", &pub, body, sizeof body, err,
        sizeof err);
    CHECK(__atomic_load_n(&proxied, __ATOMIC_SEQ_CST) == before,
          "a policy of public addresses went through the proxy");

    /* proxies this client does not speak, and curl's port */
    setenv("HTTPS_PROXY", "socks5://127.0.0.1:1080", 1);
    CHECK(get("https://janas-test.invalid/", NULL, body, sizeof body, err,
              sizeof err) != 0 &&
              strstr(err, "only http://"),
          "a socks proxy: [%s]", err);
    setenv("HTTPS_PROXY", "127.0.0.1", 1);
    CHECK(get("https://janas-test.invalid/", NULL, body, sizeof body, err,
              sizeof err) != 0 &&
              strstr(err, ":1080"),
          "port 1080 by default: [%s]", err);

    if (failures) {
        printf("test_proxy: %d failures\n", failures);
        return 1;
    }
    printf("test_proxy: ok\n");
    return 0;
}
#endif
