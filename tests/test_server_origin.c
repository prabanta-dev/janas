/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_server_origin.c - what janas-server lets in and lets out on someone
 * else's word: the web pages that may call it (src/server/origin.c: its
 * own, those named with --allow-origin, no other site even when that site
 * makes its name resolve to 127.0.0.1) and the addresses the MCP servers a
 * request names may have (src/common/mcp_policy.c: public ones, and those
 * named with --mcp-allow).
 */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>

#include "common/mcp_policy.h"
#include "server/origin.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void origins(void)
{
    struct srv_config c = {.host = "127.0.0.1"};
    static const struct {
        const char *origin, *host;
        int want;
    } cases[] = {
        {NULL, "127.0.0.1:8080", SRV_ORIGIN_NONE},
        {"http://127.0.0.1:8080", "127.0.0.1:8080", SRV_ORIGIN_SAME},
        {"http://localhost:8080", "localhost:8080", SRV_ORIGIN_SAME},
        {"http://[::1]:8080", "[::1]:8080", SRV_ORIGIN_SAME},
        {"http://127.0.0.1", "127.0.0.1", SRV_ORIGIN_SAME},
        {"http://127.0.0.1:80", "127.0.0.1", SRV_ORIGIN_SAME},
        /* another site's page */
        {"https://evil.example", "127.0.0.1:8080", SRV_ORIGIN_REFUSED},
        {"http://localhost:3000", "localhost:8080", SRV_ORIGIN_REFUSED},
        {"http://127.0.0.1:8080", "localhost:8080", SRV_ORIGIN_REFUSED},
        /* rebinding: the site's name resolved to this machine */
        {"http://evil.example:8080", "evil.example:8080", SRV_ORIGIN_REFUSED},
        {"null", "127.0.0.1:8080", SRV_ORIGIN_REFUSED},
        {"chrome-extension://abc", "127.0.0.1:8080", SRV_ORIGIN_REFUSED},
        {"http://127.0.0.1:8080", NULL, SRV_ORIGIN_REFUSED},
        {"http://[::1", "[::1]:8080", SRV_ORIGIN_REFUSED},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        int got = srv_origin_check(&c, cases[i].origin, cases[i].host);
        CHECK(got == cases[i].want, "origin %zu (%s, %s): %d, want %d", i,
              cases[i].origin ? cases[i].origin : "-",
              cases[i].host ? cases[i].host : "-", got, cases[i].want);
    }
    const char *allow[] = {"http://localhost:3000"};
    c.allow_origin = allow;
    c.n_allow_origin = 1;
    CHECK(srv_origin_check(&c, "http://localhost:3000", "localhost:8080") ==
              SRV_ORIGIN_LISTED,
          "an origin named with --allow-origin refused");
    CHECK(srv_origin_check(&c, "http://localhost:3001", "localhost:8080") ==
              SRV_ORIGIN_REFUSED,
          "an origin not named let in");
    const char *any[] = {"*"};
    c.allow_origin = any;
    CHECK(srv_origin_check(&c, "https://evil.example", "127.0.0.1:8080") ==
              SRV_ORIGIN_LISTED,
          "--allow-origin '*' did not let a page in");
}

static int public4(const char *a)
{
    struct sockaddr_in s = {.sin_family = AF_INET};
    inet_pton(AF_INET, a, &s.sin_addr);
    return janas_addr_public((struct sockaddr *)&s);
}

static int public6(const char *a)
{
    struct sockaddr_in6 s = {.sin6_family = AF_INET6};
    inet_pton(AF_INET6, a, &s.sin6_addr);
    return janas_addr_public((struct sockaddr *)&s);
}

static void addresses(void)
{
    static const char *const no4[] = {
        "127.0.0.1",       "127.1.2.3",   "10.0.0.1",        "172.16.0.1",
        "172.31.255.1",    "192.168.1.1", "169.254.169.254", "100.64.0.1",
        "100.127.255.255", "0.0.0.0",     "224.0.0.1",       "239.255.255.250",
        "255.255.255.255", "240.0.0.1",   "192.0.2.1",       "198.18.0.1",
        "198.51.100.7",    "203.0.113.9", "192.0.0.8",
    };
    static const char *const yes4[] = {"8.8.8.8", "1.1.1.1", "172.32.0.1",
                                       "100.128.0.1", "93.184.216.34"};
    for (size_t i = 0; i < sizeof(no4) / sizeof(no4[0]); i++)
        CHECK(!public4(no4[i]), "%s taken for public", no4[i]);
    for (size_t i = 0; i < sizeof(yes4) / sizeof(yes4[0]); i++)
        CHECK(public4(yes4[i]), "%s taken for private", yes4[i]);
    static const char *const no6[] = {
        "::1",
        "::",
        "::ffff:127.0.0.1",
        "::ffff:10.1.2.3",
        "64:ff9b::a9fe:a9fe",
        "fc00::1",
        "fd12:3456::1",
        "fe80::1",
        "ff02::1",
        "2001:db8::1",
        "2002:7f00:1::1",
        "::127.0.0.1",
        "2001::1",
        "100::1",
    };
    static const char *const yes6[] = {"2606:4700:4700::1111",
                                       "2a00:1450:4001::200e", "::ffff:8.8.8.8",
                                       "2002:808:808::1"};
    for (size_t i = 0; i < sizeof(no6) / sizeof(no6[0]); i++)
        CHECK(!public6(no6[i]), "%s taken for public", no6[i]);
    for (size_t i = 0; i < sizeof(yes6) / sizeof(yes6[0]); i++)
        CHECK(public6(yes6[i]), "%s taken for private", yes6[i]);

    struct sockaddr_in lo = {.sin_family = AF_INET};
    inet_pton(AF_INET, "127.0.0.1", &lo.sin_addr);
    char *allow[] = {"localhost:9000", "[::1]", "Intranet.lan"};
    struct janas_net_policy p = {
        .public_only = 1, .n_allow = 3, .allow = allow};
    const struct sockaddr *a = (struct sockaddr *)&lo;
    CHECK(janas_net_allowed(NULL, "localhost", "9000", a),
          "no policy refused an address");
    CHECK(janas_net_allowed(&p, "localhost", "9000", a),
          "an allowed host:port refused");
    CHECK(!janas_net_allowed(&p, "localhost", "9001", a),
          "another port of an allowed host let through");
    CHECK(janas_net_allowed(&p, "::1", "8080", a), "an allowed [v6] refused");
    CHECK(janas_net_allowed(&p, "intranet.lan", "443", a),
          "an allowed host (any port, any case) refused");
    CHECK(!janas_net_allowed(&p, "evil.example", "80", a),
          "a name resolving to loopback let through");
    p.public_only = 0;
    CHECK(janas_net_allowed(&p, "evil.example", "80", a),
          "public_only 0 refused");
}

/* What a listed page's preflight may send: the headers asked for, while
   they are header names; else the two every client sends. */
static void preflight(void)
{
    const char *sdk = "authorization,content-type,x-stainless-os,"
                      "x-stainless-retry-count";
    CHECK(srv_allow_headers(sdk) == sdk, "the SDK's headers refused");
    CHECK(!strcmp(srv_allow_headers(NULL), "Authorization, Content-Type"),
          "none asked");
    CHECK(!strcmp(srv_allow_headers(""), "Authorization, Content-Type"),
          "empty");
    const char *bad[] = {"content-type\r\nX-Evil: 1", "a;b", "x:y", "é"};
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++)
        CHECK(!strcmp(srv_allow_headers(bad[i]), "Authorization, Content-Type"),
              "taken: %s", bad[i]);
    char big[1100];
    memset(big, 'a', sizeof big - 1);
    big[sizeof big - 1] = 0;
    CHECK(!strcmp(srv_allow_headers(big), "Authorization, Content-Type"),
          "too long taken");
}

int main(void)
{
    origins();
    addresses();
    preflight();
    if (!failures)
        printf("test_server_origin: ok\n");
    return failures != 0;
}
