/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_tls.c - TLS on Mbed TLS for the HTTP client of MCP (see mcp_tls.h).
 */
#include "common/mcp_tls.h"

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <mbedtls/error.h>
#include <mbedtls/ssl.h>
#include <mbedtls/x509_crt.h>
#include <psa/crypto.h>

struct janas_tls {
    mbedtls_ssl_context ssl;
    int fd;
};

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int ready; /* 1: set up; -1: it failed */
static char ready_err[200];
static mbedtls_x509_crt authorities;
static mbedtls_ssl_config conf;

/* Where the distributions keep the certificate authorities. */
static const char *const BUNDLES[] = {
    "/etc/ssl/certs/ca-certificates.crt", /* Debian, Ubuntu, Arch */
    "/etc/pki/tls/certs/ca-bundle.crt",   /* Fedora, RHEL */
    "/etc/ssl/ca-bundle.pem",             /* openSUSE */
    "/etc/ssl/cert.pem",                  /* Alpine, macOS */
    NULL};

/* Once: the crypto library, the authorities, the configuration shared by
   every session. Called with the lock held. */
static int set_up(void)
{
    if (ready)
        return ready;
    ready = -1;
    if (psa_crypto_init() != PSA_SUCCESS) {
        snprintf(ready_err, sizeof(ready_err),
                 "the crypto library would "
                 "not start");
        return ready;
    }
    mbedtls_x509_crt_init(&authorities);
    int loaded = 0;
    const char *file = getenv("SSL_CERT_FILE"), *dir = getenv("SSL_CERT_DIR");
    if (file && *file)
        loaded = mbedtls_x509_crt_parse_file(&authorities, file) >= 0;
    if (!loaded && dir && *dir)
        loaded = mbedtls_x509_crt_parse_path(&authorities, dir) >= 0;
    for (int i = 0; !loaded && BUNDLES[i]; i++)
        /* a bundle with a certificate it cannot read still loads the rest,
           and says how many it skipped with a positive number */
        loaded = mbedtls_x509_crt_parse_file(&authorities, BUNDLES[i]) >= 0;
    if (!loaded) {
        snprintf(ready_err, sizeof(ready_err),
                 "no certificate authorities found (set SSL_CERT_FILE to a "
                 "bundle of them)");
        return ready;
    }
    mbedtls_ssl_config_init(&conf);
    if (mbedtls_ssl_config_defaults(&conf, MBEDTLS_SSL_IS_CLIENT,
                                    MBEDTLS_SSL_TRANSPORT_STREAM,
                                    MBEDTLS_SSL_PRESET_DEFAULT) != 0) {
        snprintf(ready_err, sizeof(ready_err), "TLS would not configure");
        return ready;
    }
    mbedtls_ssl_conf_authmode(&conf, MBEDTLS_SSL_VERIFY_REQUIRED);
    mbedtls_ssl_conf_ca_chain(&conf, &authorities, NULL);
    ready = 1;
    return ready;
}

static int bio_send(void *ctx, const unsigned char *p, size_t n)
{
    ssize_t w = send(*(int *)ctx, p, n, MSG_NOSIGNAL);
    if (w >= 0)
        return (int)w;
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR
               ? MBEDTLS_ERR_SSL_WANT_WRITE
               : MBEDTLS_ERR_SSL_INTERNAL_ERROR;
}

static int bio_recv(void *ctx, unsigned char *p, size_t n)
{
    ssize_t r = recv(*(int *)ctx, p, n, 0);
    if (r >= 0)
        return (int)r;
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR
               ? MBEDTLS_ERR_SSL_WANT_READ
               : MBEDTLS_ERR_SSL_INTERNAL_ERROR;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

struct janas_tls *janas_tls_open(int fd, const char *host, int timeout_ms,
                                 char *err, size_t err_len)
{
    pthread_mutex_lock(&lock);
    if (set_up() < 0) {
        snprintf(err, err_len, "%s: %s", host, ready_err);
        pthread_mutex_unlock(&lock);
        return NULL;
    }
    struct janas_tls *t = calloc(1, sizeof(*t));
    if (!t) {
        pthread_mutex_unlock(&lock);
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    t->fd = fd;
    mbedtls_ssl_init(&t->ssl);
    int rc = mbedtls_ssl_setup(&t->ssl, &conf);
    if (rc == 0)
        rc = mbedtls_ssl_set_hostname(&t->ssl, host);
    mbedtls_ssl_set_bio(&t->ssl, &t->fd, bio_send, bio_recv, NULL);
    double end = timeout_ms < 0 ? -1 : now_ms() + timeout_ms;
    while (rc == 0) {
        rc = mbedtls_ssl_handshake(&t->ssl);
        if (rc != MBEDTLS_ERR_SSL_WANT_READ && rc != MBEDTLS_ERR_SSL_WANT_WRITE)
            break;
        short ev = rc == MBEDTLS_ERR_SSL_WANT_READ ? POLLIN : POLLOUT;
        pthread_mutex_unlock(&lock);
        double left = end < 0 ? 1000 : end - now_ms();
        struct pollfd pf = {.fd = fd, .events = ev};
        int r = left <= 0 ? 0 : poll(&pf, 1, left > 1000 ? 1000 : (int)left);
        pthread_mutex_lock(&lock);
        if (r == 0 && end >= 0 && now_ms() >= end) {
            rc = MBEDTLS_ERR_SSL_TIMEOUT;
            break;
        }
        rc = 0;
    }
    if (rc != 0) {
        uint32_t flags = mbedtls_ssl_get_verify_result(&t->ssl);
        char why[160] = "";
        if (flags && flags != (uint32_t)-1) {
            mbedtls_x509_crt_verify_info(why, sizeof(why), "", flags);
            why[strcspn(why, "\n")] = 0;
        } else if (rc == MBEDTLS_ERR_SSL_TIMEOUT) {
            snprintf(why, sizeof(why), "no handshake in time");
        } else {
            mbedtls_strerror(rc, why, sizeof(why));
        }
        snprintf(err, err_len, "%s: TLS: %s", host, why);
        mbedtls_ssl_free(&t->ssl);
        free(t);
        t = NULL;
    }
    pthread_mutex_unlock(&lock);
    return t;
}

long janas_tls_read(struct janas_tls *t, void *p, size_t n)
{
    pthread_mutex_lock(&lock);
    int r;
    do /* a ticket for a later session is not data: read on */
        r = mbedtls_ssl_read(&t->ssl, p, n);
    while (r == MBEDTLS_ERR_SSL_RECEIVED_NEW_SESSION_TICKET);
    pthread_mutex_unlock(&lock);
    if (r >= 0)
        return r;
    if (r == MBEDTLS_ERR_SSL_WANT_READ || r == MBEDTLS_ERR_SSL_WANT_WRITE) {
        errno = EAGAIN;
        return -2;
    }
    return r == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY ? 0 : -1;
}

long janas_tls_write(struct janas_tls *t, const void *p, size_t n)
{
    pthread_mutex_lock(&lock);
    int w = mbedtls_ssl_write(&t->ssl, p, n);
    pthread_mutex_unlock(&lock);
    if (w >= 0)
        return w;
    if (w == MBEDTLS_ERR_SSL_WANT_READ || w == MBEDTLS_ERR_SSL_WANT_WRITE) {
        errno = EAGAIN;
        return -2;
    }
    return -1;
}

void janas_tls_close(struct janas_tls *t)
{
    if (!t)
        return;
    pthread_mutex_lock(&lock);
    mbedtls_ssl_close_notify(&t->ssl); /* once, without waiting */
    mbedtls_ssl_free(&t->ssl);
    pthread_mutex_unlock(&lock);
    free(t);
}
