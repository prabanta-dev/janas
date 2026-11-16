/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_tls.c - TLS on Mbed TLS for the HTTP client of MCP (see mcp_tls.h).
 */
#include "common/mcp_tls.h"
#include "common/sock.h"

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

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>

/* Windows keeps no bundle: the authorities are the system's own store of
   trusted roots. 1 when one at least was read. */
static int load_system_roots(void)
{
    HCERTSTORE st = CertOpenSystemStoreA(0, "ROOT");
    if (!st)
        return 0;
    int n = 0;
    for (PCCERT_CONTEXT c = NULL;
         (c = CertEnumCertificatesInStore(st, c)) != NULL;)
        if (c->dwCertEncodingType == X509_ASN_ENCODING &&
            mbedtls_x509_crt_parse_der(&authorities, c->pbCertEncoded,
                                       c->cbCertEncoded) == 0)
            n++;
    CertCloseStore(st, 0);
    return n > 0;
}

/* The chain the server sent, judged by Windows as its own programs judge
   it (curl's schannel, Edge): Windows keeps a root in its store only once
   something asked for it, fetching the missing ones as it builds a chain,
   so the store read above may lack the root a site needs - jsdelivr's
   GlobalSign Root R46 on a new Windows 11 (16 Nov 2026), "not correctly
   signed by the trusted CA" where curl.exe connected. 1: trusted for a
   server of that name. */
static int windows_trusts(const mbedtls_x509_crt *chain, const char *host)
{
    HCERTSTORE mem = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, 0, 0, NULL);
    if (!mem)
        return 0;
    PCCERT_CONTEXT leaf = NULL;
    for (const mbedtls_x509_crt *c = chain; c; c = c->next)
        CertAddEncodedCertificateToStore(
            mem, X509_ASN_ENCODING, c->raw.p, (DWORD)c->raw.len,
            CERT_STORE_ADD_ALWAYS, c == chain ? &leaf : NULL);
    int ok = 0;
    WCHAR wh[256];
    LPSTR usage[] = {(LPSTR)szOID_PKIX_KP_SERVER_AUTH};
    CERT_CHAIN_PARA cp = {.cbSize = sizeof cp};
    cp.RequestedUsage.dwType = USAGE_MATCH_TYPE_AND;
    cp.RequestedUsage.Usage.cUsageIdentifier = 1;
    cp.RequestedUsage.Usage.rgpszUsageIdentifier = usage;
    PCCERT_CHAIN_CONTEXT ch = NULL;
    if (leaf &&
        MultiByteToWideChar(CP_UTF8, 0, host, -1, wh,
                            (int)(sizeof wh / sizeof *wh)) > 0 &&
        CertGetCertificateChain(NULL, leaf, NULL, mem, &cp, 0, NULL, &ch)) {
        SSL_EXTRA_CERT_CHAIN_POLICY_PARA ssl = {.cbSize = sizeof ssl,
                                                .dwAuthType = AUTHTYPE_SERVER,
                                                .pwszServerName = wh};
        CERT_CHAIN_POLICY_PARA pp = {.cbSize = sizeof pp,
                                     .pvExtraPolicyPara = &ssl};
        CERT_CHAIN_POLICY_STATUS ps = {.cbSize = sizeof ps};
        ok = CertVerifyCertificateChainPolicy(CERT_CHAIN_POLICY_SSL, ch, &pp,
                                              &ps) &&
             ps.dwError == 0;
        CertFreeCertificateChain(ch);
    }
    if (leaf)
        CertFreeCertificateContext(leaf);
    CertCloseStore(mem, 0);
    return ok;
}
#endif

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
#ifdef _WIN32
    if (!loaded)
        loaded = load_system_roots();
#endif
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
#ifdef _WIN32
    /* the chain judged after the handshake, by Windows where Mbed TLS
       cannot link it to a root of the store (windows_trusts) */
    mbedtls_ssl_conf_authmode(&conf, MBEDTLS_SSL_VERIFY_OPTIONAL);
#else
    mbedtls_ssl_conf_authmode(&conf, MBEDTLS_SSL_VERIFY_REQUIRED);
#endif
    mbedtls_ssl_conf_ca_chain(&conf, &authorities, NULL);
    ready = 1;
    return ready;
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
#ifdef _WIN32
    /* optional only so far: a chain neither trusts is refused here */
    if (rc == 0 && mbedtls_ssl_get_verify_result(&t->ssl) != 0 &&
        !windows_trusts(mbedtls_ssl_get_peer_cert(&t->ssl), host))
        rc = MBEDTLS_ERR_X509_CERT_VERIFY_FAILED;
#endif
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

void janas_tls_lock(void)
{
    pthread_mutex_lock(&lock);
}

void janas_tls_unlock(void)
{
    pthread_mutex_unlock(&lock);
}

int janas_tls_crypto(void)
{
    static int started;
    if (!started && psa_crypto_init() == PSA_SUCCESS)
        started = 1;
    return started ? 0 : -1;
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
