/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * inflate.h - DEFLATE data (RFC 1951) made whole again (inflate.c): inside
 * its zlib wrapping (RFC 1950), as PNG keeps its pixels, or bare, as a zip
 * archive keeps its files.
 */
#ifndef JANAS_COMMON_INFLATE_H
#define JANAS_COMMON_INFLATE_H

#include <stddef.h>
#include <stdint.h>

/* The n bytes of zlib data at p into out, exactly cap bytes expected:
   0, or -1 (data broken, or not cap bytes) with the reason in why. */
int janas_zlib_inflate(const uint8_t *p, size_t n, uint8_t *out, size_t cap,
                       char *why, size_t wn);
/* zlib data of a length not known beforehand, at most max bytes once
   inflated, into a malloc'd *out (a 0 after it, not counted): its length,
   or -1 with the reason in why. */
long janas_zlib_inflate_alloc(const uint8_t *p, size_t n, size_t max,
                              uint8_t **out, char *why, size_t wn);
/* The same for bare DEFLATE data, as a zip archive keeps its files. */
int janas_raw_inflate(const uint8_t *p, size_t n, uint8_t *out, size_t cap,
                      char *why, size_t wn);

#endif
