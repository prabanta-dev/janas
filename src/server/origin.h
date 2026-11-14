/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * origin.h - which web pages may call the server (origin.c).
 */
#ifndef JANAS_SERVER_ORIGIN_H
#define JANAS_SERVER_ORIGIN_H

#include "server.h"

enum {
    SRV_ORIGIN_REFUSED, /* a page of another site */
    SRV_ORIGIN_NONE,    /* no Origin: a program, not a page */
    SRV_ORIGIN_SAME,    /* the server's own page (/test) */
    SRV_ORIGIN_LISTED   /* named with --allow-origin: answered with CORS */
};

/* What a request with these Origin and Host headers (either may be NULL)
   is. */
int srv_origin_check(const struct srv_config *c, const char *origin,
                     const char *host);

/* The headers a listed page's preflight may send (its
   Access-Control-Request-Headers, NULL if none): those it asks for when
   they are header names only and fewer than 1024 bytes, else the two every
   client sends. The clients' SDKs add their own (OpenAI's JavaScript one
   its x-stainless-*, which Obsidian's Copilot sends): a preflight that
   left them out failed the request. */
const char *srv_allow_headers(const char *asked);

#endif
