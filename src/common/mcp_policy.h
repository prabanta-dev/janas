/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_policy.h - where a connection made on someone else's word may go.
 * janas-server reaches the MCP servers a request names: left free, a
 * request could make it talk to this machine's own services, the local
 * network's, or a cloud's metadata address. With a policy, the HTTP client
 * (mcp_http.c) checks every address a name resolves to and connects only
 * to one that passed, so a name that resolves to a public address when
 * checked and to a private one when used (DNS rebinding) gains nothing.
 */
#ifndef JANAS_COMMON_MCP_POLICY_H
#define JANAS_COMMON_MCP_POLICY_H

#include <stddef.h>

struct sockaddr;

struct janas_net_policy {
    int public_only; /* refuse loopback, private, link-local, reserved... */
    size_t n_allow;  /* allowed all the same: "host" or "host:port", the */
    char **allow;    /* host a name or a literal address ([v6] with a port) */
};

/* 1 when the address is one of the Internet's: not loopback, private
   (10/8, 172.16/12, 192.168/16, fc00::/7), shared (100.64/10), link-local,
   multicast, documentation, benchmarking or otherwise reserved; an IPv4
   address inside an IPv6 one (mapped, NAT64, 6to4) judged as itself. */
int janas_addr_public(const struct sockaddr *sa);

/* The process's own policy, when its environment says one
   (JANAS_NET_PUBLIC_ONLY=1, JANAS_NET_ALLOW="host:port,..."): a served
   user's Brebus is given it. NULL otherwise. */
const struct janas_net_policy *janas_net_default(void);

/* Whether a connection to host:port at address sa may be made (p NULL:
   the process's own policy, or any). */
int janas_net_allowed(const struct janas_net_policy *p, const char *host,
                      const char *port, const struct sockaddr *sa);

#endif
