/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mhd_tls.h - HTTPS in front of a libmicrohttpd daemon (mhd_tls.c): the
 * certificate and its key read once, then a listener of its own whose
 * connections, their handshake done, are handed to the daemon as if they
 * had come in clear. One per process. In libjanas.a only, for the
 * programs that serve (janas-server, Brebus), not in the shared
 * libraries.
 */
#ifndef JANAS_COMMON_MHD_TLS_H
#define JANAS_COMMON_MHD_TLS_H

#include <stddef.h>

struct MHD_Daemon;
struct sockaddr;

/* cert and key, PEM files: 0, or -1 with the reason */
int janas_mhd_tls_setup(const char *cert, const char *key, char *err,
                        size_t err_len);
/* listening on addr (named where in an error), each connection handed to
   mhd, which must take them (MHD_USE_NO_LISTEN_SOCKET or its like): 0, or
   -1 with the reason */
int janas_mhd_tls_start(struct MHD_Daemon *mhd, const struct sockaddr *addr,
                        const char *where, char *err, size_t err_len);
/* the listener stopped, the connections ended: before the daemon stops */
void janas_mhd_tls_stop(void);

#endif
