/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sha256.h - SHA-256 (FIPS 180-4), to check the files the tools download
 * and write against the fingerprints of MODELS.md. With the x86 SHA
 * extensions where the CPU has them: a converted model is up to 48 GB.
 */
#ifndef JANAS_COMMON_SHA256_H
#define JANAS_COMMON_SHA256_H

#include <stddef.h>
#include <stdint.h>

struct janas_sha256 {
    uint32_t h[8];
    uint64_t len; /* bytes hashed */
    uint8_t buf[64];
    size_t n; /* bytes waiting in buf */
};

void janas_sha256_init(struct janas_sha256 *s);
void janas_sha256_update(struct janas_sha256 *s, const void *data, size_t n);
/* The digest as 64 lowercase hex digits and a NUL. */
void janas_sha256_hex(struct janas_sha256 *s, char out[65]);
/* The SHA-256 of a whole file, as hex; 0, or -1 if it cannot be read. */
int janas_sha256_file(const char *path, char out[65]);

#endif
