/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * origin.c - which web pages may call the server (see origin.h). A page of
 * another site can make the browser send a request to 127.0.0.1 - it
 * cannot read the answer, but the model still runs, and a tool of type mcp
 * still connects - and a site whose name it makes resolve to 127.0.0.1
 * (DNS rebinding) even reads it, being then of the same origin by name.
 * The browser says which page asks, in Origin: the server's own page and
 * the pages named with --allow-origin pass, the others are refused,
 * whatever the key.
 */
#include "origin.h"

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/utsname.h>

/* host[:port] split, [v6] kept bracketed; the port "" when absent. */
static int split(const char *s, size_t n, char *host, size_t hc, char *port,
                 size_t pc)
{
    const char *colon = NULL;
    if (n && s[0] == '[') {
        const char *e = memchr(s, ']', n);
        if (!e)
            return -1;
        if ((size_t)(e + 1 - s) < n) {
            if (e[1] != ':')
                return -1;
            colon = e + 1;
        }
    } else {
        colon = memchr(s, ':', n);
    }
    size_t hn = colon ? (size_t)(colon - s) : n;
    size_t pn = colon ? n - hn - 1 : 0;
    if (!hn || hn >= hc || pn >= pc)
        return -1;
    memcpy(host, s, hn);
    host[hn] = 0;
    memcpy(port, colon ? colon + 1 : "", pn);
    port[pn] = 0;
    return 0;
}

/* A name only this machine answers to, or an address: what a rebinding
   site cannot be called. */
static int trusted_host(const char *h, const char *listen)
{
    size_t n = strlen(h);
    if (!strcasecmp(h, "localhost") ||
        (n > 10 && !strcasecmp(h + n - 10, ".localhost")))
        return 1;
    if (h[0] == '[') /* an IPv6 address */
        return 1;
    if (strspn(h, "0123456789.") == n) /* an IPv4 address */
        return 1;
    if (listen && !strcasecmp(h, listen))
        return 1;
    struct utsname u; /* this machine's name (Windows too, win_posix.c) */
    if (uname(&u) == 0) {
        const char *me = u.nodename;
        size_t m = strlen(me);
        if (!strcasecmp(h, me) || (n == m + 6 && !strncasecmp(h, me, m) &&
                                   !strcasecmp(h + m, ".local")))
            return 1;
    }
    return 0;
}

int srv_origin_check(const struct srv_config *c, const char *origin,
                     const char *host)
{
    if (!origin)
        return SRV_ORIGIN_NONE; /* not a browser's page: a program */
    for (int i = 0; i < c->n_allow_origin; i++)
        if (!strcmp(c->allow_origin[i], "*") ||
            !strcasecmp(c->allow_origin[i], origin))
            return SRV_ORIGIN_LISTED;
    /* the server's own page: the origin is the address it was asked at */
    const char *def;
    if (!strncasecmp(origin, "http://", 7))
        def = "80";
    else if (!strncasecmp(origin, "https://", 8))
        def = "443";
    else
        return SRV_ORIGIN_REFUSED; /* "null", an extension's... */
    const char *o = strstr(origin, "://") + 3;
    char oh[256], op[8], hh[256], hp[8];
    if (!host || split(o, strlen(o), oh, sizeof oh, op, sizeof op) != 0 ||
        split(host, strlen(host), hh, sizeof hh, hp, sizeof hp) != 0)
        return SRV_ORIGIN_REFUSED;
    if (strcasecmp(oh, hh) != 0 ||
        strcmp(op[0] ? op : def, hp[0] ? hp : "80") != 0)
        return SRV_ORIGIN_REFUSED;
    return trusted_host(hh, c->host) ? SRV_ORIGIN_SAME : SRV_ORIGIN_REFUSED;
}

const char *srv_allow_headers(const char *asked)
{
    int ok = asked && *asked && strlen(asked) < 1024;
    for (const char *c = asked; ok && *c; c++)
        ok = (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
             (*c >= '0' && *c <= '9') || *c == '-' || *c == '_' || *c == ',' ||
             *c == ' ';
    return ok ? asked : "Authorization, Content-Type";
}
