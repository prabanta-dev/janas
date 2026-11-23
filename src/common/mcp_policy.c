/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_policy.c - where a connection made on someone else's word may go
 * (see mcp_policy.h). The ranges are IANA's special-purpose registries.
 */
#include "common/mcp_policy.h"

#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>

static int v4_public(uint32_t a)
{
    static const struct {
        uint32_t net;
        int bits;
    } bad[] = {
        {0x00000000u, 8},  /* this network */
        {0x0a000000u, 8},  /* private */
        {0x64400000u, 10}, /* shared, carrier-grade NAT */
        {0x7f000000u, 8},  /* loopback */
        {0xa9fe0000u, 16}, /* link-local, the clouds' metadata */
        {0xac100000u, 12}, /* private */
        {0xc0000000u, 24}, /* protocol assignments */
        {0xc0000200u, 24}, /* documentation */
        {0xc0586300u, 24}, /* 6to4 relays */
        {0xc0a80000u, 16}, /* private */
        {0xc6120000u, 15}, /* benchmarking */
        {0xc6336400u, 24}, /* documentation */
        {0xcb007100u, 24}, /* documentation */
        {0xe0000000u, 4},  /* multicast */
        {0xf0000000u, 4},  /* reserved, broadcast */
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++)
        if ((a >> (32 - bad[i].bits)) == (bad[i].net >> (32 - bad[i].bits)))
            return 0;
    return 1;
}

static uint32_t be32(const uint8_t *p)
{
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 |
           p[3];
}

static int v6_public(const uint8_t *b)
{
    static const uint8_t zero[12] = {0};
    static const uint8_t mapped[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255};
    static const uint8_t nat64[12] = {0, 0x64, 0xff, 0x9b};
    if (!memcmp(b, mapped, 12) || !memcmp(b, nat64, 12))
        return v4_public(be32(b + 12));
    if (!memcmp(b, zero, 12)) /* ::, ::1, IPv4-compatible (deprecated) */
        return 0;
    if (b[0] == 0x20 && b[1] == 0x02) /* 6to4: the IPv4 address after */
        return v4_public(be32(b + 2));
    if ((b[0] & 0xfe) == 0xfc) /* unique local */
        return 0;
    if (b[0] == 0xfe && (b[1] & 0xc0) == 0x80) /* link-local */
        return 0;
    if (b[0] == 0xfe && (b[1] & 0xc0) == 0xc0) /* site-local (deprecated) */
        return 0;
    if (b[0] == 0xff) /* multicast */
        return 0;
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x0d && b[3] == 0xb8)
        return 0; /* documentation */
    if (b[0] == 0x01 && b[1] == 0x00 && !memcmp(b + 2, zero, 6))
        return 0; /* discard */
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x00 && b[3] == 0x00)
        return 0; /* Teredo: an IPv4 address behind, hidden */
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x00 && (b[3] & 0xf0) == 0x20)
        return 0; /* ORCHIDv2 */
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x00 && b[3] == 0x02 &&
        b[4] == 0 && b[5] == 0)
        return 0;     /* benchmarking */
    if (b[0] == 0x5f) /* 5f00::/16, segment routing */
        return 0;
    return 1;
}

int janas_addr_public(const struct sockaddr *sa)
{
    if (!sa)
        return 0;
    if (sa->sa_family == AF_INET) {
        const struct sockaddr_in *s = (const struct sockaddr_in *)sa;
        return v4_public(be32((const uint8_t *)&s->sin_addr));
    }
    if (sa->sa_family == AF_INET6) {
        const struct sockaddr_in6 *s = (const struct sockaddr_in6 *)sa;
        return v6_public((const uint8_t *)&s->sin6_addr);
    }
    return 0;
}

/* Does an entry ("host", "host:port", "[v6]:port", "[v6]") name host:port? */
static int entry_names(const char *e, const char *host, const char *port)
{
    const char *h = e, *he, *p = NULL;
    if (*e == '[') {
        h = e + 1;
        he = strchr(h, ']');
        if (!he)
            return 0;
        if (he[1] == ':')
            p = he + 2;
        else if (he[1])
            return 0;
    } else {
        he = strrchr(e, ':');
        if (he && strchr(e, ':') != he) /* a bare v6 address, no port */
            he = NULL;
        if (he)
            p = he + 1;
        else
            he = e + strlen(e);
    }
    size_t n = (size_t)(he - h);
    if (strlen(host) != n || strncasecmp(h, host, n) != 0)
        return 0;
    return !p || !strcmp(p, port);
}

/* the process's own, from its environment: JANAS_NET_PUBLIC_ONLY=1 and
   JANAS_NET_ALLOW, comma separated */
static struct janas_net_policy env_p;
static char env_allow[1024], *env_list[16];
static pthread_once_t env_once = PTHREAD_ONCE_INIT;

static void env_read(void)
{
    const char *on = getenv("JANAS_NET_PUBLIC_ONLY");
    if (!on || strcmp(on, "1"))
        return;
    env_p.public_only = 1;
    const char *a = getenv("JANAS_NET_ALLOW");
    snprintf(env_allow, sizeof env_allow, "%s", a ? a : "");
    for (char *s = env_allow, *sv = NULL, *t = strtok_r(s, ",", &sv);
         t && env_p.n_allow < 16; t = strtok_r(NULL, ",", &sv))
        env_list[env_p.n_allow++] = t;
    env_p.allow = env_list;
}

const struct janas_net_policy *janas_net_default(void)
{
    pthread_once(&env_once, env_read);
    return env_p.public_only ? &env_p : NULL;
}

int janas_net_allowed(const struct janas_net_policy *p, const char *host,
                      const char *port, const struct sockaddr *sa)
{
    if (!p)
        p = janas_net_default();
    if (!p || !p->public_only || janas_addr_public(sa))
        return 1;
    for (size_t i = 0; i < p->n_allow; i++)
        if (entry_names(p->allow[i], host, port))
            return 1;
    return 0;
}
