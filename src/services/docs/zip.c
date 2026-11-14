/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * zip.c - one file of a zip archive (docx and odt are zip archives of XML
 * files), found by its name in the central directory at the archive's end
 * and made whole with Janas's own inflate. The archive is input from
 * outside: every offset and length is checked against the bytes there
 * are, the size it says a file has is bounded, and the file must come out
 * of exactly that size. Stored and deflated files only; no ZIP64, no
 * encryption: such an archive is told, not guessed at.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/inflate.h"
#include "docs.h"

static uint32_t u16(const uint8_t *p)
{
    return p[0] | (uint32_t)p[1] << 8;
}

static uint32_t u32(const uint8_t *p)
{
    return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
           (uint32_t)p[3] << 24;
}

int dz_zip_get(const uint8_t *z, size_t n, const char *name, uint8_t **out,
               size_t *out_n, char *why, size_t wn)
{
    *out = NULL;
    *out_n = 0;
    /* the end of central directory record: 22 bytes and a comment of at
       most 65535, looked for from the end */
    size_t e = (size_t)-1, lo = n > 22 + 65535 ? n - 22 - 65535 : 0;
    for (size_t k = n >= 22 ? n - 22 + 1 : 0; k-- > lo;)
        if (u32(z + k) == 0x06054b50 && k + 22 + u16(z + k + 20) == n) {
            e = k;
            break;
        }
    if (e == (size_t)-1) {
        snprintf(why, wn, "not a zip archive");
        return -1;
    }
    uint32_t entries = u16(z + e + 10), cd_size = u32(z + e + 12),
             cd_at = u32(z + e + 16);
    if (entries == 0xffff || cd_at == 0xffffffffu) {
        snprintf(why, wn, "a ZIP64 archive, not read");
        return -1;
    }
    if (cd_at > e || cd_size > e - cd_at) {
        snprintf(why, wn, "a broken zip archive");
        return -1;
    }
    size_t nl = strlen(name);
    for (size_t p = cd_at, i = 0; i < entries; i++) {
        if (e - p < 46 || u32(z + p) != 0x02014b50) {
            snprintf(why, wn, "a broken zip archive");
            return -1;
        }
        uint32_t flags = u16(z + p + 8), method = u16(z + p + 10),
                 csize = u32(z + p + 20), usize = u32(z + p + 24),
                 fn = u16(z + p + 28), xn = u16(z + p + 30),
                 cn = u16(z + p + 32), local = u32(z + p + 42);
        if (e - p - 46 < (size_t)fn + xn + cn) {
            snprintf(why, wn, "a broken zip archive");
            return -1;
        }
        if (fn != nl || memcmp(z + p + 46, name, nl)) {
            p += 46 + (size_t)fn + xn + cn;
            continue;
        }
        if (flags & 1) {
            snprintf(why, wn, "an encrypted file");
            return -1;
        }
        if (method != 0 && method != 8) {
            snprintf(why, wn, "compressed with method %u, not read", method);
            return -1;
        }
        if (usize > DZ_MAX_FILE || csize == 0xffffffffu) {
            snprintf(why, wn, "%s is larger than %d MiB", name,
                     DZ_MAX_FILE >> 20);
            return -1;
        }
        /* the local header: its own name and extra lengths */
        if (local > cd_at || cd_at - local < 30 ||
            u32(z + local) != 0x04034b50) {
            snprintf(why, wn, "a broken zip archive");
            return -1;
        }
        size_t data =
            (size_t)local + 30 + u16(z + local + 26) + u16(z + local + 28);
        if (data > cd_at || cd_at - data < csize) {
            snprintf(why, wn, "a broken zip archive");
            return -1;
        }
        uint8_t *u = malloc(usize + 1);
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
        } else if (janas_raw_inflate(z + data, csize, u, usize, why, wn)) {
            free(u);
            return -1;
        }
        u[usize] = 0;
        *out = u;
        *out_n = usize;
        return 0;
    }
    snprintf(why, wn, "no %s in the archive", name);
    return 1;
}
