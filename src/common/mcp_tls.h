/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_tls.h - TLS for the HTTP client of MCP (https://), on Mbed TLS,
 * compiled in from src/third_party/mbedtls.
 *
 * The server's certificate is checked against the certificate authorities
 * of the system (the file SSL_CERT_FILE names, or the directory
 * SSL_CERT_DIR, or the bundle the distribution keeps), and against the
 * name of the host: a connection that fails either is not made. TLS 1.2
 * and 1.3, as Mbed TLS's defaults allow them.
 *
 * The socket is non-blocking: a call that would wait returns -2 and the
 * caller polls the socket. Calls from several threads are safe: they take
 * turns inside, since Mbed TLS is built without its own threading.
 */
#ifndef JANAS_MCP_TLS_H
#define JANAS_MCP_TLS_H

#include <stddef.h>

struct janas_tls;

/* The TLS session over a connected socket, for host: its handshake done
   within timeout_ms (< 0: no limit). NULL with the reason in err. */
struct janas_tls *janas_tls_open(int fd, const char *host, int timeout_ms,
                                 char *err, size_t err_len);
/* Bytes read or written (> 0), 0 at the end of the stream, -2 when the
   socket has to be waited for, -1 on an error. */
long janas_tls_read(struct janas_tls *t, void *p, size_t n);
long janas_tls_write(struct janas_tls *t, const void *p, size_t n);
void janas_tls_close(struct janas_tls *t);

/* The one lock every use of Mbed TLS in the process takes (it has no
   threading of its own): janas-server's TLS takes it too. janas_tls_crypto
   starts the crypto library once, under it: 0, or -1. */
void janas_tls_lock(void);
void janas_tls_unlock(void);
int janas_tls_crypto(void);
/* A certificate of this computer's own, for a server of Janas's to speak
   HTTPS (Brebus's): self-signed, an EC P-256 key, valid for days days for
   the names given (host names, or IPv4/IPv6 addresses as text). The key is
   written readable by the user alone. 0, or -1 with the reason in err. */
int janas_tls_make_cert(const char *cert, const char *key,
                        const char *const *names, size_t n_names, int days,
                        char *err, size_t err_len);
/* The SHA-256 of a PEM certificate's DER, as "AB:CD:..." (95 characters
   and the NUL), what a user compares to trust it: 0, or -1. */
int janas_tls_fingerprint(const char *cert, char *out, size_t cap);
/* A file made readable by its owner alone: on Windows, where its mode says
   nothing, an access list of the user alone, not inherited from its
   folder (on other systems the caller's mode, 0600, already says it).
   0, or -1. */
int janas_tls_private(const char *path);
/* The certificates of a PEM file trusted too, by the sessions opened after
   (Brebus trusting its own server's): 0, or -1. */
int janas_tls_trust(const char *cert);
/* The SHA-256 of n bytes (OAuth's PKCE): 0, or -1. Takes the lock. */
int janas_tls_sha256(const void *p, size_t n, unsigned char out[32]);

#endif
