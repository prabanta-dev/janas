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
#include <fcntl.h>
#include <unistd.h>

#include <mbedtls/error.h>
#include <mbedtls/ssl.h>
#include <mbedtls/pem.h>
#include <mbedtls/pk.h>
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

int janas_tls_sha256(const void *p, size_t n, unsigned char out[32])
{
    size_t len = 0;
    pthread_mutex_lock(&lock);
    int ok =
        janas_tls_crypto() == 0 &&
        psa_hash_compute(PSA_ALG_SHA_256, p, n, out, 32, &len) == PSA_SUCCESS &&
        len == 32;
    pthread_mutex_unlock(&lock);
    return ok ? 0 : -1;
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

/* ---- a file for its owner alone (janas_tls_private) ---- */

#ifdef _WIN32
#include <aclapi.h>

int janas_tls_private(const char *path)
{
    /* the user's SID, from the process's token */
    HANDLE tok = NULL;
    union {
        TOKEN_USER tu;
        unsigned char b[256];
    } u;
    DWORD n = 0;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &tok))
        return -1;
    BOOL ok = GetTokenInformation(tok, TokenUser, &u, sizeof u, &n);
    CloseHandle(tok);
    if (!ok)
        return -1;
    /* full control to the user, nobody else, nothing inherited */
    EXPLICIT_ACCESS_A ea = {.grfAccessPermissions = GENERIC_ALL,
                            .grfAccessMode = SET_ACCESS,
                            .grfInheritance = NO_INHERITANCE,
                            .Trustee = {.TrusteeForm = TRUSTEE_IS_SID,
                                        .TrusteeType = TRUSTEE_IS_USER,
                                        .ptstrName = (LPSTR)u.tu.User.Sid}};
    PACL acl = NULL;
    if (SetEntriesInAclA(1, &ea, NULL, &acl) != ERROR_SUCCESS)
        return -1;
    DWORD r = SetNamedSecurityInfoA((LPSTR)path, SE_FILE_OBJECT,
                                    DACL_SECURITY_INFORMATION |
                                        PROTECTED_DACL_SECURITY_INFORMATION,
                                    NULL, NULL, acl, NULL);
    LocalFree(acl);
    return r == ERROR_SUCCESS ? 0 : -1;
}
#else
#include <sys/stat.h>
/* the owner's alone, whatever the umask made it */
int janas_tls_private(const char *path)
{
    return chmod(path, 0600) == 0 ? 0 : -1;
}
#endif

/* ---- a certificate of this computer's own (janas_tls_make_cert) ---- */

/* an address as its 4 or 16 bytes; 0 when it is no address */
static size_t ip_bytes(const char *s, unsigned char out[16])
{
    if (strchr(s, ':'))
        return mbedtls_x509_crt_parse_cn_inet_pton(s, out) == 16 ? 16 : 0;
    unsigned v[4];
    char tail;
    if (sscanf(s, "%u.%u.%u.%u%c", &v[0], &v[1], &v[2], &v[3], &tail) != 4 ||
        v[0] > 255 || v[1] > 255 || v[2] > 255 || v[3] > 255)
        return 0;
    for (int i = 0; i < 4; i++)
        out[i] = (unsigned char)v[i];
    return 4;
}

static int write_all(const char *path, const unsigned char *p, size_t n,
                     int mode)
{
    char tmp[4200];
    snprintf(tmp, sizeof(tmp), "%s.new", path);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, mode);
    if (fd < 0)
        return -1;
    /* a key: its owner's alone before a byte of it is written */
    if (mode == 0600 && janas_tls_private(tmp) != 0) {
        close(fd);
        remove(tmp);
        return -1;
    }
    int bad = write(fd, p, n) != (ssize_t)n;
    bad |= close(fd) != 0;
    if (bad || rename(tmp, path) != 0) {
        remove(tmp);
        return -1;
    }
    return 0;
}

int janas_tls_make_cert(const char *cert, const char *key,
                        const char *const *names, size_t n_names, int days,
                        char *err, size_t err_len)
{
    if (!cert || !key || !names || !n_names || n_names > 32 || days < 1) {
        snprintf(err, err_len, "invalid argument");
        return -1;
    }
    pthread_mutex_lock(&lock);
    int rc = -1;
    psa_key_id_t id = 0;
    mbedtls_pk_context pk;
    mbedtls_x509write_cert crt;
    mbedtls_pk_init(&pk);
    mbedtls_x509write_crt_init(&crt);
    mbedtls_x509_san_list san[32];
    unsigned char addr[32][16];
    static unsigned char pem[8192];
    if (janas_tls_crypto() != 0) {
        snprintf(err, err_len, "the crypto library would not start");
        goto out;
    }
    /* the key: P-256, made here, exported once to be kept */
    psa_key_attributes_t at = PSA_KEY_ATTRIBUTES_INIT;
    psa_set_key_type(&at, PSA_KEY_TYPE_ECC_KEY_PAIR(PSA_ECC_FAMILY_SECP_R1));
    psa_set_key_bits(&at, 256);
    psa_set_key_usage_flags(&at, PSA_KEY_USAGE_SIGN_HASH |
                                     PSA_KEY_USAGE_SIGN_MESSAGE |
                                     PSA_KEY_USAGE_EXPORT);
    psa_set_key_algorithm(&at, PSA_ALG_ECDSA(PSA_ALG_SHA_256));
    if (psa_generate_key(&at, &id) != PSA_SUCCESS ||
        mbedtls_pk_copy_from_psa(id, &pk) != 0) {
        snprintf(err, err_len, "no key could be made");
        goto out;
    }
    /* its names: the host names and the addresses it answers on */
    for (size_t i = 0; i < n_names; i++) {
        memset(&san[i], 0, sizeof(san[i]));
        size_t ln = ip_bytes(names[i], addr[i]);
        san[i].node.type =
            ln ? MBEDTLS_X509_SAN_IP_ADDRESS : MBEDTLS_X509_SAN_DNS_NAME;
        san[i].node.san.unstructured_name.p =
            ln ? addr[i] : (unsigned char *)names[i];
        san[i].node.san.unstructured_name.len = ln ? ln : strlen(names[i]);
        san[i].next = i + 1 < n_names ? &san[i + 1] : NULL;
    }
    /* valid from yesterday (a clock a little behind), for days days */
    char from[16], to[16];
    time_t now = time(NULL) - 86400, end = now + (time_t)days * 86400;
    struct tm a, b;
    gmtime_r(&now, &a);
    gmtime_r(&end, &b);
    strftime(from, sizeof(from), "%Y%m%d%H%M%S", &a);
    strftime(to, sizeof(to), "%Y%m%d%H%M%S", &b);
    unsigned char serial[16];
    if (psa_generate_random(serial, sizeof(serial)) != PSA_SUCCESS)
        goto fail;
    serial[0] &= 0x7f; /* a positive number */
    char subject[300];
    snprintf(subject, sizeof(subject), "CN=%.200s,O=Janas", names[0]);
    mbedtls_x509write_crt_set_version(&crt, MBEDTLS_X509_CRT_VERSION_3);
    mbedtls_x509write_crt_set_md_alg(&crt, MBEDTLS_MD_SHA256);
    mbedtls_x509write_crt_set_subject_key(&crt, &pk);
    mbedtls_x509write_crt_set_issuer_key(&crt, &pk);
    if (mbedtls_x509write_crt_set_serial_raw(&crt, serial, sizeof(serial)) ||
        mbedtls_x509write_crt_set_validity(&crt, from, to) ||
        mbedtls_x509write_crt_set_subject_name(&crt, subject) ||
        mbedtls_x509write_crt_set_issuer_name(&crt, subject) ||
        /* its own authority: trusted as one, by fingerprint */
        mbedtls_x509write_crt_set_basic_constraints(&crt, 1, 0) ||
        mbedtls_x509write_crt_set_key_usage(
            &crt, MBEDTLS_X509_KU_DIGITAL_SIGNATURE |
                      MBEDTLS_X509_KU_KEY_CERT_SIGN) ||
        mbedtls_x509write_crt_set_subject_alternative_name(&crt, san))
        goto fail;
    if (mbedtls_pk_write_key_pem(&pk, pem, sizeof(pem)) != 0 ||
        write_all(key, pem, strlen((char *)pem), 0600) != 0) {
        snprintf(err, err_len, "cannot write the key %s", key);
        goto out;
    }
    if (mbedtls_x509write_crt_pem(&crt, pem, sizeof(pem)) != 0 ||
        write_all(cert, pem, strlen((char *)pem), 0644) != 0) {
        snprintf(err, err_len, "cannot write the certificate %s", cert);
        goto out;
    }
    rc = 0;
    goto out;
fail:
    snprintf(err, err_len, "the certificate could not be made");
out:
    mbedtls_x509write_crt_free(&crt);
    mbedtls_pk_free(&pk);
    if (id)
        psa_destroy_key(id);
    memset(pem, 0, sizeof(pem));
    pthread_mutex_unlock(&lock);
    return rc;
}

int janas_tls_fingerprint(const char *cert, char *out, size_t cap)
{
    if (!cert || !out || cap < 96)
        return -1;
    pthread_mutex_lock(&lock);
    mbedtls_x509_crt c;
    mbedtls_x509_crt_init(&c);
    unsigned char h[32];
    size_t len = 0;
    int ok =
        janas_tls_crypto() == 0 && mbedtls_x509_crt_parse_file(&c, cert) == 0 &&
        psa_hash_compute(PSA_ALG_SHA_256, c.raw.p, c.raw.len, h, 32, &len) ==
            PSA_SUCCESS;
    mbedtls_x509_crt_free(&c);
    pthread_mutex_unlock(&lock);
    if (!ok)
        return -1;
    for (int i = 0; i < 32; i++)
        snprintf(out + 3 * i, cap - 3 * (size_t)i, i < 31 ? "%02X:" : "%02X",
                 h[i]);
    return 0;
}

int janas_tls_trust(const char *cert)
{
    pthread_mutex_lock(&lock);
    int ok =
        set_up() == 1 && mbedtls_x509_crt_parse_file(&authorities, cert) == 0;
    pthread_mutex_unlock(&lock);
    return ok ? 0 : -1;
}
