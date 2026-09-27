/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_sha256.c - SHA-256 against the vectors of FIPS 180-4 and, over
 * lengths across the block boundaries fed in uneven pieces, the portable
 * compression function against the one on the SHA extensions (the same
 * digest by both, JANAS_SHA256_C=1 forcing the portable one).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/sha256.h"

static int failures;

static void hex_of(const void *p, size_t n, size_t piece, char out[65])
{
    struct janas_sha256 s;
    janas_sha256_init(&s);
    const unsigned char *c = p;
    for (size_t at = 0; at < n; at += piece)
        janas_sha256_update(&s, c + at, n - at < piece ? n - at : piece);
    janas_sha256_hex(&s, out);
}

static void check(const char *what, const void *p, size_t n, const char *want)
{
    char got[65];
    hex_of(p, n, n ? n : 1, got);
    if (strcmp(got, want) != 0) {
        printf("FAIL %s: %s, want %s\n", what, got, want);
        failures++;
    }
}

int main(void)
{
    check("empty", "", 0,
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check("abc", "abc", 3,
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const char *two =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    check("two blocks", two, strlen(two),
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    size_t n = 1000000;
    char *a = malloc(n);
    if (!a)
        return 1;
    memset(a, 'a', n);
    check("a million a", a, n,
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    /* uneven pieces: the digest does not depend on how the data comes */
    char whole[65], pieces[65];
    unsigned x = 12345;
    for (size_t i = 0; i < n; i++)
        a[i] = (char)((x = x * 1103515245u + 12345u) >> 16);
    for (size_t len = 0; len < 300; len += 7) {
        hex_of(a, len, len ? len : 1, whole);
        hex_of(a, len, 13, pieces);
        if (strcmp(whole, pieces) != 0) {
            printf("FAIL length %zu: whole %s, in pieces %s\n", len, whole,
                   pieces);
            failures++;
        }
    }
    hex_of(a, n, n, whole);
    hex_of(a, n, 4099, pieces);
    if (strcmp(whole, pieces) != 0) {
        printf("FAIL a million bytes in pieces\n");
        failures++;
    }
    free(a);
    printf("test_sha256: %s%s\n", failures ? "FAILED" : "ok",
           getenv("JANAS_SHA256_C") ? " (portable)" : "");
    return failures ? 1 : 0;
}
