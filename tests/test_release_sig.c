/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* The release's signature (release_sig.h): a key's signature checked by its
   public key, refused for another file, another key, a changed signature
   and lines that are not one; a private key's public key the generator's. */
#include "common/release_sig.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

int main(void)
{
    static const char sums[] =
        "0123abcd  brebus-x86_64.AppImage\n4567ef01  brebus_amd64.deb\n";
    size_t n = sizeof sums - 1;
    uint8_t priv[JANAS_SIG_PRIV], pub[JANAS_SIG_PUB], pub2[JANAS_SIG_PUB],
        other[JANAS_SIG_PRIV], opub[JANAS_SIG_PUB];
    CHECK(janas_sig_keygen(priv, pub) == 0, "no key");
    CHECK(pub[0] == 4, "public key not uncompressed: %02x", pub[0]);
    CHECK(janas_sig_public(priv, pub2) == 0 &&
              memcmp(pub, pub2, sizeof pub) == 0,
          "the private key's public key differs");
    CHECK(janas_sig_keygen(other, opub) == 0, "no second key");

    char line[JANAS_SIG_LINE + 1];
    CHECK(janas_sig_sign(priv, sums, n, line, sizeof line) == 0, "not signed");
    CHECK(strlen(line) == JANAS_SIG_LINE &&
              strncmp(line, JANAS_SIG_PREFIX, strlen(JANAS_SIG_PREFIX)) == 0 &&
              line[JANAS_SIG_LINE - 1] == '\n',
          "the line: %s", line);
    CHECK(janas_sig_sign(priv, sums, n, line, JANAS_SIG_LINE) != 0,
          "signed into a buffer too short");
    CHECK(janas_sig_sign(priv, sums, n, line, sizeof line) == 0, "not signed");
    size_t ln = strlen(line);
    CHECK(janas_sig_verify(pub, sums, n, line, ln) == 0,
          "own signature refused");
    /* spaces and a CR around it, as an editor may leave them */
    char padded[JANAS_SIG_LINE + 8];
    snprintf(padded, sizeof padded, " %.*s\r\n", (int)(ln - 1), line);
    CHECK(janas_sig_verify(pub, sums, n, padded, strlen(padded)) == 0,
          "signature with spaces around refused");

    char changed[sizeof sums];
    memcpy(changed, sums, sizeof sums);
    changed[2] ^= 1;
    CHECK(janas_sig_verify(pub, changed, n, line, ln) != 0,
          "another file's signature taken");
    CHECK(janas_sig_verify(pub, sums, n - 1, line, ln) != 0,
          "a shorter file's signature taken");
    CHECK(janas_sig_verify(opub, sums, n, line, ln) != 0,
          "another key's signature taken");

    char bad[JANAS_SIG_LINE + 1];
    memcpy(bad, line, sizeof bad);
    size_t at = strlen(JANAS_SIG_PREFIX) + 10;
    bad[at] = bad[at] == '0' ? '1' : '0';
    CHECK(janas_sig_verify(pub, sums, n, bad, ln) != 0,
          "a changed signature taken");
    memcpy(bad, line, sizeof bad);
    bad[0] = 'E';
    CHECK(janas_sig_verify(pub, sums, n, bad, ln) != 0,
          "a line of another kind taken");
    CHECK(janas_sig_verify(pub, sums, n, line, ln - 3) != 0,
          "a cut signature taken");
    CHECK(janas_sig_verify(pub, sums, n, "", 0) != 0, "an empty line taken");
    memcpy(bad, line, sizeof bad);
    bad[at] = 'g';
    CHECK(janas_sig_verify(pub, sums, n, bad, ln) != 0,
          "a signature with a letter not hex taken");

    uint8_t junk[JANAS_SIG_PUB];
    memcpy(junk, pub, sizeof junk);
    junk[40] ^= 0x55; /* almost surely off the curve */
    CHECK(janas_sig_verify(junk, sums, n, line, ln) != 0,
          "a public key off the curve taken");

    uint8_t round[4];
    char hex[9];
    static const uint8_t four[4] = {0x00, 0x7f, 0x80, 0xff};
    janas_sig_hex(four, 4, hex);
    CHECK(strcmp(hex, "007f80ff") == 0, "hex: %s", hex);
    CHECK(janas_sig_unhex("007F80ff", 8, round, 4) == 0 &&
              memcmp(round, four, 4) == 0,
          "unhex");
    CHECK(janas_sig_unhex("007f80f", 7, round, 4) != 0, "odd hex taken");

    if (failures) {
        printf("test_release_sig: %d failures\n", failures);
        return 1;
    }
    printf("test_release_sig: ok\n");
    return 0;
}
