/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sha256.c - SHA-256 (see sha256.h): the portable compression function of
 * FIPS 180-4, and one on the x86 SHA extensions (sha256rnds2, sha256msg1,
 * sha256msg2) chosen at run time, several times faster.
 */
#include "sha256.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static inline uint32_t ror(uint32_t x, int n)
{
    return (x >> n) | (x << (32 - n));
}

static void blocks_c(uint32_t *h, const uint8_t *p, size_t nb)
{
    for (; nb; nb--, p += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 |
                   (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 =
                ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 =
                ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5],
                 g = h[6], k = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = k + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) +
                          ((e & f) ^ (~e & g)) + K[i] + w[i];
            uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) +
                          ((a & b) ^ (a & c) ^ (b & c));
            k = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += k;
    }
}

#if defined(__x86_64__)
#include <immintrin.h>

/* The x86 SHA extensions: the state as (ABEF, CDGH), four rounds per
   sha256rnds2 pair, the schedule by sha256msg1 and sha256msg2. */
__attribute__((target("sha,sse4.1"))) static void
blocks_ni(uint32_t *h, const uint8_t *p, size_t nb)
{
    const __m128i bswap =
        _mm_set_epi64x(0x0c0d0e0f08090a0bll, 0x0405060700010203ll);
    __m128i tmp = _mm_loadu_si128((const __m128i *)h);
    __m128i st1 = _mm_loadu_si128((const __m128i *)(h + 4));
    tmp = _mm_shuffle_epi32(tmp, 0xb1);         /* CDAB */
    st1 = _mm_shuffle_epi32(st1, 0x1b);         /* EFGH */
    __m128i st0 = _mm_alignr_epi8(tmp, st1, 8); /* ABEF */
    st1 = _mm_blend_epi16(st1, tmp, 0xf0);      /* CDGH */
    for (; nb; nb--, p += 64) {
        __m128i a0 = st0, c0 = st1, msg, m[4];
        for (int i = 0; i < 4; i++)
            m[i] = _mm_shuffle_epi8(
                _mm_loadu_si128((const __m128i *)(p + 16 * i)), bswap);
        for (int r = 0; r < 16; r++) {
            __m128i w = m[r & 3];
            if (r >= 4) {
                /* the next four words of the schedule, into m[r & 3] */
                __m128i t = _mm_alignr_epi8(m[(r - 1) & 3], m[(r - 2) & 3], 4);
                w = _mm_add_epi32(
                    _mm_sha256msg1_epu32(m[r & 3], m[(r + 1) & 3]), t);
                w = _mm_sha256msg2_epu32(w, m[(r - 1) & 3]);
                m[r & 3] = w;
            }
            msg =
                _mm_add_epi32(w, _mm_loadu_si128((const __m128i *)(K + 4 * r)));
            st1 = _mm_sha256rnds2_epu32(st1, st0, msg);
            msg = _mm_shuffle_epi32(msg, 0x0e);
            st0 = _mm_sha256rnds2_epu32(st0, st1, msg);
        }
        st0 = _mm_add_epi32(st0, a0);
        st1 = _mm_add_epi32(st1, c0);
    }
    tmp = _mm_shuffle_epi32(st0, 0x1b);    /* FEBA */
    st1 = _mm_shuffle_epi32(st1, 0xb1);    /* DCHG */
    st0 = _mm_blend_epi16(tmp, st1, 0xf0); /* DCBA */
    st1 = _mm_alignr_epi8(st1, tmp, 8);    /* ABEF -> HGFE */
    _mm_storeu_si128((__m128i *)h, st0);
    _mm_storeu_si128((__m128i *)(h + 4), st1);
}
#endif

static void blocks(uint32_t *h, const uint8_t *p, size_t nb)
{
#if defined(__x86_64__)
    static int ni = -1;
    if (ni < 0)
        ni = __builtin_cpu_supports("sha") && __builtin_cpu_supports("sse4.1");
    if (ni && !getenv("JANAS_SHA256_C")) {
        blocks_ni(h, p, nb);
        return;
    }
#endif
    blocks_c(h, p, nb);
}

void janas_sha256_init(struct janas_sha256 *s)
{
    static const uint32_t h0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                   0xa54ff53a, 0x510e527f, 0x9b05688c,
                                   0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, h0, sizeof(h0));
    s->len = 0;
    s->n = 0;
}

void janas_sha256_update(struct janas_sha256 *s, const void *data, size_t n)
{
    const uint8_t *p = data;
    s->len += n;
    if (s->n) {
        size_t k = 64 - s->n < n ? 64 - s->n : n;
        memcpy(s->buf + s->n, p, k);
        s->n += k;
        p += k;
        n -= k;
        if (s->n < 64)
            return;
        blocks(s->h, s->buf, 1);
        s->n = 0;
    }
    if (n >= 64) {
        blocks(s->h, p, n / 64);
        p += n / 64 * 64;
        n %= 64;
    }
    memcpy(s->buf, p, n);
    s->n = n;
}

void janas_sha256_hex(struct janas_sha256 *s, char out[65])
{
    uint64_t bits = s->len * 8;
    uint8_t pad[72] = {0x80};
    size_t k = (s->n < 56 ? 56 : 120) - s->n;
    for (int i = 0; i < 8; i++)
        pad[k + i] = (uint8_t)(bits >> (56 - 8 * i));
    uint64_t len = s->len;
    janas_sha256_update(s, pad, k + 8);
    s->len = len;
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++) {
            uint8_t b = (uint8_t)(s->h[i] >> (24 - 8 * j));
            out[8 * i + 2 * j] = hex[b >> 4];
            out[8 * i + 2 * j + 1] = hex[b & 15];
        }
    out[64] = 0;
}

int janas_sha256_file(const char *path, char out[65])
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -1;
    size_t cap = (size_t)8 << 20;
    uint8_t *buf = malloc(cap);
    if (!buf) {
        close(fd);
        return -1;
    }
    struct janas_sha256 s;
    janas_sha256_init(&s);
    ssize_t r;
    while ((r = read(fd, buf, cap)) > 0)
        janas_sha256_update(&s, buf, (size_t)r);
    free(buf);
    close(fd);
    if (r < 0)
        return -1;
    janas_sha256_hex(&s, out);
    return 0;
}
