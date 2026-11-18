/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * zip.c - see zip.h. Moved from janas-docs (docx and odt are zip archives
 * of XML files) to be shared with Brebus's Python installer.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/inflate.h"
#include "common/zip.h"

static uint32_t u16(const uint8_t *p)
{
    return p[0] | (uint32_t)p[1] << 8;
}

static uint32_t u32(const uint8_t *p)
{
    return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
           (uint32_t)p[3] << 24;
}

/* the end of central directory record, and the directory's place: 0, or
   -1 with the reason */
static int directory(const uint8_t *z, size_t n, size_t *e, uint32_t *entries,
                     uint32_t *cd_at, char *why, size_t wn)
{
    /* 22 bytes and a comment of at most 65535, looked for from the end */
    size_t lo = n > 22 + 65535 ? n - 22 - 65535 : 0;
    *e = (size_t)-1;
    for (size_t k = n >= 22 ? n - 22 + 1 : 0; k-- > lo;)
        if (u32(z + k) == 0x06054b50 && k + 22 + u16(z + k + 20) == n) {
            *e = k;
            break;
        }
    if (*e == (size_t)-1) {
        snprintf(why, wn, "not a zip archive");
        return -1;
    }
    *entries = u16(z + *e + 10);
    uint32_t cd_size = u32(z + *e + 12);
    *cd_at = u32(z + *e + 16);
    if (*entries == 0xffff || *cd_at == 0xffffffffu) {
        snprintf(why, wn, "a ZIP64 archive, not read");
        return -1;
    }
    if (*cd_at > *e || cd_size > *e - *cd_at) {
        snprintf(why, wn, "a broken zip archive");
        return -1;
    }
    return 0;
}

/* The file whose directory entry is at p, made whole into a malloc'd
 *out: 0, or -1 with the reason */
static int entry(const uint8_t *z, size_t cd_at, size_t p, size_t max,
                 uint8_t **out, size_t *out_n, char *why, size_t wn)
{
    uint32_t flags = u16(z + p + 8), method = u16(z + p + 10),
             csize = u32(z + p + 20), usize = u32(z + p + 24),
             fn = u16(z + p + 28), local = u32(z + p + 42);
    if (flags & 1) {
        snprintf(why, wn, "an encrypted file");
        return -1;
    }
    if (method != 0 && method != 8) {
        snprintf(why, wn, "compressed with method %u, not read", method);
        return -1;
    }
    if (usize > max || csize == 0xffffffffu) {
        snprintf(why, wn, "%.*s is larger than %zu MiB", (int)fn, z + p + 46,
                 max >> 20);
        return -1;
    }
    /* the local header: its own name and extra lengths */
    if (local > cd_at || cd_at - local < 30 || u32(z + local) != 0x04034b50) {
        snprintf(why, wn, "a broken zip archive");
        return -1;
    }
    size_t data =
        (size_t)local + 30 + u16(z + local + 26) + u16(z + local + 28);
    if (data > cd_at || cd_at - data < csize) {
        snprintf(why, wn, "a broken zip archive");
        return -1;
    }
    uint8_t *u = malloc((size_t)usize + 1);
    if (!u) {
        snprintf(why, wn, "out of memory");
        return -1;
    }
    if (method == 0) {
        if (csize != usize) {
            free(u);
            snprintf(why, wn, "a broken zip archive");
            return -1;
        }
        memcpy(u, z + data, usize);
    } else if (usize && janas_raw_inflate(z + data, csize, u, usize, why, wn)) {
        free(u);
        return -1;
    }
    u[usize] = 0;
    *out = u;
    *out_n = usize;
    return 0;
}

/* a file by its name */
int janas_zip_get(const uint8_t *z, size_t n, const char *name, size_t max,
                  uint8_t **out, size_t *out_n, char *why, size_t wn)
{
    *out = NULL;
    *out_n = 0;
    size_t e;
    uint32_t entries, cd_at;
    if (directory(z, n, &e, &entries, &cd_at, why, wn) != 0)
        return -1;
    size_t nl = strlen(name);
    for (size_t p = cd_at, i = 0; i < entries; i++) {
        if (e - p < 46 || u32(z + p) != 0x02014b50) {
            snprintf(why, wn, "a broken zip archive");
            return -1;
        }
        uint32_t fn = u16(z + p + 28), xn = u16(z + p + 30),
                 cn = u16(z + p + 32);
        if (e - p - 46 < (size_t)fn + xn + cn) {
            snprintf(why, wn, "a broken zip archive");
            return -1;
        }
        if (fn == nl && !memcmp(z + p + 46, name, nl))
            return entry(z, cd_at, p, max, out, out_n, why, wn);
        p += 46 + (size_t)fn + xn + cn;
    }
    snprintf(why, wn, "no %s in the archive", name);
    return 1;
}

int janas_zip_each(const uint8_t *z, size_t n, size_t max,
                   int (*each)(void *ctx, const char *name, size_t name_n,
                               const uint8_t *p, size_t p_n),
                   void *ctx, char *why, size_t wn)
{
    size_t e;
    uint32_t entries, cd_at;
    if (directory(z, n, &e, &entries, &cd_at, why, wn) != 0)
        return -1;
    for (size_t p = cd_at, i = 0; i < entries; i++) {
        if (e - p < 46 || u32(z + p) != 0x02014b50) {
            snprintf(why, wn, "a broken zip archive");
            return -1;
        }
        uint32_t fn = u16(z + p + 28), xn = u16(z + p + 30),
                 cn = u16(z + p + 32);
        if (e - p - 46 < (size_t)fn + xn + cn) {
            snprintf(why, wn, "a broken zip archive");
            return -1;
        }
        uint8_t *u;
        size_t un;
        if (entry(z, cd_at, p, max, &u, &un, why, wn) != 0)
            return -1;
        int rc = each(ctx, (const char *)z + p + 46, fn, u, un);
        free(u);
        if (rc)
            return rc;
        p += 46 + (size_t)fn + xn + cn;
    }
    return 0;
}
