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

#endif
