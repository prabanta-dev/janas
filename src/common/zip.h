/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * zip.h - the files of a zip archive, made whole with Janas's own inflate
 * (zip.c): one by its name (a document's XML, for janas-docs), or each in
 * turn (a Python wheel or Python's embeddable package unpacked, for
 * Brebus). The archive is input from outside: every offset and length is
 * checked against the bytes there are, the size a file says it has is
 * bounded, and it must come out of exactly that size. Stored and deflated
 * files only; no ZIP64, no encryption: such an archive is told, not
 * guessed at.
 */
#ifndef JANAS_COMMON_ZIP_H
#define JANAS_COMMON_ZIP_H

#include <stddef.h>
#include <stdint.h>

/* The file name of the archive z (n bytes), at most max bytes, into a
   malloc'd *out, its *out_n bytes followed by a 0: 0, 1 when there is
   none, -1 with the reason in why. */
int janas_zip_get(const uint8_t *z, size_t n, const char *name, size_t max,
                  uint8_t **out, size_t *out_n, char *why, size_t wn);

/* Each file of the archive (a folder's entry, its name ending in '/', too,
   with no bytes), at most max bytes each, given to each(ctx, name, its
   length, its bytes, their number); each returns 0 to go on, else the
   walk stops with that. 0, or the first nonzero, or -1 with the reason in
   why. */
int janas_zip_each(const uint8_t *z, size_t n, size_t max,
                   int (*each)(void *ctx, const char *name, size_t name_n,
                               const uint8_t *p, size_t p_n),
                   void *ctx, char *why, size_t wn);

#endif
