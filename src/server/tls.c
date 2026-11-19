/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tls.c - HTTPS for janas-server (--tls-cert, --tls-key), on the Mbed TLS
 * the MCP client already carries. libmicrohttpd would want GnuTLS for it;
 * instead the server listens itself, a thread for each connection does the
 * handshake and then carries the bytes between the TLS session and one end
 * of a pair of sockets, whose other end libmicrohttpd is handed
 * (MHD_add_connection) as if a client had connected in clear. Everything
 * above - the requests, the streams, the origins - is the same.
 *
 * Mbed TLS has no threading of its own here: every call to it takes the
 * process's one lock (janas_tls_lock), the waits never inside it.
 *
 * On Windows, which has no socketpair, the pair is two ends of a connection
 * over the loopback, made for each client; the sockets go through sock.h.
 */
#include "server.h"

#include <errno.h>
#include <fcntl.h>
#include <microhttpd.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <mbedtls/error.h>
#include <mbedtls/pk.h>
#include <mbedtls/ssl.h>
#include <mbedtls/x509_crt.h>

#include "common/mcp_tls.h"
#include "common/sock.h"

#define HANDSHAKE_MS 30000

/* a socket made not to block */
static void no_block(int fd)
{
#ifdef _WIN32
    u_long on = 1;
    ioctlsocket((SOCKET)fd, FIONBIO, &on);
#else
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
#endif
}

/* Two connected sockets, sp[0] for libmicrohttpd: 0, or -1. */
static int sock_pair(int sp[2])
{
#ifdef _WIN32
    /* the two ends of a connection over the loopback, the listener gone
       once they are made */
    SOCKET l = socket(AF_INET, SOCK_STREAM, 0), a = INVALID_SOCKET,
           b = INVALID_SOCKET;
    struct sockaddr_in in = {.sin_family = AF_INET,
                             .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    int len = sizeof in;
    if (l != INVALID_SOCKET &&
        bind(l, (struct sockaddr *)&in, sizeof in) == 0 && listen(l, 1) == 0 &&
        getsockname(l, (struct sockaddr *)&in, &len) == 0 &&
        (b = socket(AF_INET, SOCK_STREAM, 0)) != INVALID_SOCKET &&
        connect(b, (struct sockaddr *)&in, sizeof in) == 0)
        a = accept(l, NULL, NULL);
    if (l != INVALID_SOCKET)
        closesocket(l);
    if (a == INVALID_SOCKET) {
        if (b != INVALID_SOCKET)
            closesocket(b);
        return -1;
    }
    sp[0] = (int)a;
    sp[1] = (int)b;
    return 0;
#else
    return socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp);
#endif
}

static mbedtls_ssl_config conf;
static mbedtls_x509_crt chain;
static mbedtls_pk_context key;
static int listen_fd = -1;
static pthread_t acceptor;
static volatile int stopping;
static _Atomic int active; /* connections being served */
static struct srv *server;

static void why(int rc, char *err, size_t err_len, const char *what)
{
    char m[160];
    mbedtls_strerror(rc, m, sizeof m);
    snprintf(err, err_len, "%s: %s", what, m);
}

/* 0 when the key is the certificate's: mbedtls_pk_check_pair, or, as it
   refuses RSA pairs that are one here, their public halves the same */
static int same_key(const mbedtls_pk_context *pub,
                    const mbedtls_pk_context *prv)
{
    if (mbedtls_pk_check_pair(pub, prv) == 0)
        return 0;
    static unsigned char a[4096], b[4096];
    int na = mbedtls_pk_write_pubkey_der(pub, a, sizeof a);
    int nb = mbedtls_pk_write_pubkey_der(prv, b, sizeof b);
    /* written at the end of the buffers */
    return na > 0 && na == nb &&
                   memcmp(a + sizeof a - na, b + sizeof b - nb, (size_t)na) == 0
               ? 0
               : -1;
}

int srv_tls_setup(const char *cert, const char *keyfile, char *err,
                  size_t err_len)
{
    janas_tls_lock();
    int rc = janas_tls_crypto() == 0 ? 0 : -1;
    if (rc) {
        janas_tls_unlock();
        snprintf(err, err_len, "the crypto library would not start");
        return -1;
    }
    mbedtls_x509_crt_init(&chain);
    mbedtls_pk_init(&key);
    mbedtls_ssl_config_init(&conf);
    if (access(cert, R_OK) != 0 || access(keyfile, R_OK) != 0) {
        rc = -1;
        snprintf(err, err_len, "%s: %s", access(cert, R_OK) ? cert : keyfile,
                 strerror(errno));
    } else if ((rc = mbedtls_x509_crt_parse_file(&chain, cert)) != 0)
        snprintf(err, err_len, "%s: not a certificate in PEM", cert);
    else if ((rc = mbedtls_pk_parse_keyfile(&key, keyfile, NULL)) != 0)
        snprintf(err, err_len,
                 "%s: not a private key in PEM (or one with a password)",
                 keyfile);
    else if ((rc = same_key(&chain.pk, &key)) != 0)
        snprintf(err, err_len, "%s is not the key of %s", keyfile, cert);
    else if ((rc = mbedtls_ssl_config_defaults(
                  &conf, MBEDTLS_SSL_IS_SERVER, MBEDTLS_SSL_TRANSPORT_STREAM,
                  MBEDTLS_SSL_PRESET_DEFAULT)) != 0)
        why(rc, err, err_len, "TLS");
    else if ((rc = mbedtls_ssl_conf_own_cert(&conf, &chain, &key)) != 0)
        why(rc, err, err_len, "the certificate and its key");
    janas_tls_unlock();
    return rc ? -1 : 0;
}

static int bio_send(void *ctx, const unsigned char *p, size_t n)
{
    long w = janas_sock_send(*(int *)ctx, p, n);
    if (w >= 0)
        return (int)w;
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR
               ? MBEDTLS_ERR_SSL_WANT_WRITE
               : MBEDTLS_ERR_SSL_INTERNAL_ERROR;
}

static int bio_recv(void *ctx, unsigned char *p, size_t n)
{
    long r = janas_sock_recv(*(int *)ctx, p, n);
    if (r >= 0)
        return (int)r;
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR
               ? MBEDTLS_ERR_SSL_WANT_READ
               : MBEDTLS_ERR_SSL_INTERNAL_ERROR;
}

struct conn {
    int fd; /* the client's, non-blocking */
    struct sockaddr_storage peer;
    socklen_t peer_len;
};

/* All of n bytes to a socket in clear: 0, or -1. */
static int put_all(int fd, const unsigned char *p, size_t n)
{
    while (n) {
        long w = janas_sock_send(fd, p, n);
        if (w < 0 && (errno == EAGAIN || errno == EINTR)) {
            struct pollfd pf = {.fd = fd, .events = POLLOUT};
            poll(&pf, 1, 1000);
            continue;
        }
        if (w <= 0)
            return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static void *serve(void *arg)
{
    struct conn *c = arg;
    active++;
    mbedtls_ssl_context ssl;
    int sp[2] = {-1, -1};
    janas_tls_lock();
    mbedtls_ssl_init(&ssl);
    int rc = mbedtls_ssl_setup(&ssl, &conf);
    mbedtls_ssl_set_bio(&ssl, &c->fd, bio_send, bio_recv, NULL);
    janas_tls_unlock();
    /* the handshake, the waits outside the lock */
    int waited = 0;
    while (rc == 0 && !stopping) {
        janas_tls_lock();
        rc = mbedtls_ssl_handshake(&ssl);
        janas_tls_unlock();
        if (rc != MBEDTLS_ERR_SSL_WANT_READ && rc != MBEDTLS_ERR_SSL_WANT_WRITE)
            break;
        struct pollfd pf = {
            .fd = c->fd,
            .events = rc == MBEDTLS_ERR_SSL_WANT_READ ? POLLIN : POLLOUT};
        if (poll(&pf, 1, 500) == 0 && (waited += 500) >= HANDSHAKE_MS)
            break;
        rc = 0;
    }
    if (rc != 0 || stopping || sock_pair(sp) != 0 ||
        MHD_add_connection(server->mhd, (MHD_socket)sp[0],
                           (struct sockaddr *)&c->peer, c->peer_len) != MHD_YES)
        goto done;
    sp[0] = -1; /* libmicrohttpd's now */
    /* the bytes both ways until either side closes */
    unsigned char buf[16384];
    for (;;) {
        janas_tls_lock();
        int pending = (int)mbedtls_ssl_get_bytes_avail(&ssl);
        janas_tls_unlock();
        struct pollfd pf[2] = {{.fd = c->fd, .events = POLLIN},
                               {.fd = sp[1], .events = POLLIN}};
        if (!pending && poll(pf, 2, 1000) < 0 && errno != EINTR)
            break;
        if (stopping)
            break;
        if (pending || (pf[0].revents & (POLLIN | POLLHUP | POLLERR))) {
            janas_tls_lock();
            int r = mbedtls_ssl_read(&ssl, buf, sizeof buf);
            janas_tls_unlock();
            if (r > 0) {
                if (put_all(sp[1], buf, (size_t)r) != 0)
                    break;
            } else if (r != MBEDTLS_ERR_SSL_WANT_READ &&
                       r != MBEDTLS_ERR_SSL_WANT_WRITE &&
                       r != MBEDTLS_ERR_SSL_RECEIVED_NEW_SESSION_TICKET) {
                break; /* closed, or an error */
            }
        }
        if (pf[1].revents & (POLLIN | POLLHUP | POLLERR)) {
            long r = janas_sock_recv(sp[1], buf, sizeof buf);
            if (r <= 0 && !(r < 0 && (errno == EAGAIN || errno == EINTR)))
                break;
            for (long done = 0; r > 0 && done < r;) {
                janas_tls_lock();
                int w = mbedtls_ssl_write(&ssl, buf + done, (size_t)(r - done));
                janas_tls_unlock();
                if (w > 0)
                    done += w;
                else if (w == MBEDTLS_ERR_SSL_WANT_WRITE ||
                         w == MBEDTLS_ERR_SSL_WANT_READ) {
                    struct pollfd po = {.fd = c->fd, .events = POLLOUT};
                    poll(&po, 1, 1000);
                } else {
                    goto done;
                }
            }
        }
    }
done:
    janas_tls_lock();
    mbedtls_ssl_close_notify(&ssl);
    mbedtls_ssl_free(&ssl);
    janas_tls_unlock();
    if (sp[0] >= 0)
        janas_sock_close(sp[0]);
    if (sp[1] >= 0)
        janas_sock_close(sp[1]);
    janas_sock_close(c->fd);
    free(c);
    active--;
    return NULL;
}

static void *accept_main(void *arg)
{
    (void)arg;
    while (!stopping) {
        struct pollfd pf = {.fd = listen_fd, .events = POLLIN};
        if (poll(&pf, 1, 500) <= 0)
            continue;
        struct conn *c = calloc(1, sizeof(*c));
        if (!c)
            continue;
        c->peer_len = sizeof c->peer;
        c->fd =
            (int)accept(listen_fd, (struct sockaddr *)&c->peer, &c->peer_len);
        if (c->fd < 0) {
            free(c);
            continue;
        }
        no_block(c->fd);
        pthread_t t;
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
        if (pthread_create(&t, &at, serve, c) != 0) {
            janas_sock_close(c->fd);
            free(c);
        }
        pthread_attr_destroy(&at);
    }
    return NULL;
}

int srv_tls_start(struct srv *s, const struct sockaddr *addr, char *err,
                  size_t err_len)
{
    server = s;
    socklen_t len = addr->sa_family == AF_INET6
                        ? (socklen_t)sizeof(struct sockaddr_in6)
                        : (socklen_t)sizeof(struct sockaddr_in);
    listen_fd = janas_sock_open(addr->sa_family, SOCK_STREAM, 0);
    int one = 1;
    if (listen_fd >= 0)
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const void *)&one,
                   sizeof one);
    if (listen_fd < 0 || bind(listen_fd, addr, len) != 0 ||
        listen(listen_fd, 64) != 0) {
        snprintf(err, err_len, "cannot listen on %s:%u: %s", s->cfg.host,
                 (unsigned)s->cfg.port, strerror(errno));
        if (listen_fd >= 0)
            janas_sock_close(listen_fd);
        listen_fd = -1;
        return -1;
    }
    stopping = 0;
    if (pthread_create(&acceptor, NULL, accept_main, NULL) != 0) {
        snprintf(err, err_len, "cannot start the TLS listener");
        janas_sock_close(listen_fd);
        listen_fd = -1;
        return -1;
    }
    return 0;
}

void srv_tls_stop(void)
{
    if (listen_fd < 0)
        return;
    stopping = 1;
    pthread_join(acceptor, NULL);
    /* the connections end within a second of their polls: before
       libmicrohttpd stops under them */
    for (int i = 0; active > 0 && i < 50; i++)
        usleep(100000);
    janas_sock_close(listen_fd);
    listen_fd = -1;
}
