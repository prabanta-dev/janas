/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * The project's signature on a release: ECDSA on P-256 over the SHA-256 of
 * a file (a release's SHA256SUMS), with Mbed TLS's PSA. Its file, beside
 * the signed one (SHA256SUMS.sig), holds one line:
 *     ecdsa-p256-sha256 <r and s, 64 bytes, in hex>
 * The private key is 32 bytes (a hex line in a file of its owner's, never
 * in the repository); the public key, 65 bytes (04, x, y), is what Brebus
 * carries to check its updates.
 */
#ifndef JANAS_RELEASE_SIG_H
#define JANAS_RELEASE_SIG_H

#include <stddef.h>
#include <stdint.h>

enum { JANAS_SIG_PRIV = 32, JANAS_SIG_PUB = 65, JANAS_SIG_RAW = 64 };

/* the signature file's line, its newline included, and its size */
#define JANAS_SIG_PREFIX "ecdsa-p256-sha256 "
enum { JANAS_SIG_LINE = sizeof(JANAS_SIG_PREFIX) - 1 + 2 * JANAS_SIG_RAW + 1 };

/* a new key pair from the system's randomness. 0, or -1 */
int janas_sig_keygen(uint8_t priv[JANAS_SIG_PRIV], uint8_t pub[JANAS_SIG_PUB]);

/* the public key of a private one. 0, or -1 */
int janas_sig_public(const uint8_t priv[JANAS_SIG_PRIV],
                     uint8_t pub[JANAS_SIG_PUB]);

/* msg's signature as the signature file's line (line: JANAS_SIG_LINE + 1
   bytes, NUL included). 0, or -1 */
int janas_sig_sign(const uint8_t priv[JANAS_SIG_PRIV], const void *msg,
                   size_t n, char *line, size_t len);

/* whether line (the signature file's, its spaces and newlines around it
   ignored) signs msg with pub: 0 when it does, -1 otherwise */
int janas_sig_verify(const uint8_t pub[JANAS_SIG_PUB], const void *msg,
                     size_t n, const char *line, size_t line_len);

/* n bytes from 2n hex digits; 0, or -1 if s is not exactly that */
int janas_sig_unhex(const char *s, size_t len, uint8_t *out, size_t n);

/* n bytes as 2n hex digits and a NUL */
void janas_sig_hex(const uint8_t *p, size_t n, char *out);

#endif
