/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * quant_z.h - the prompt's tiles on 512 bits (private to quant.c, included
 * once, after the AVX-512 VNNI kernels of quant_dotn.h, whose transposition
 * it reuses).
 *
 * The tiles of quant_dotn.h hold four weights of each of eight rows in a
 * register of 256 bits, one int32 lane a row, against a token's four
 * activations broadcast. On 512 bits a register holds sixteen rows: the
 * two transposed buffers of rows g..g+7 and g+8..g+15 side by side, read
 * once a piece for all the tokens of a tile, and the token's activations
 * broadcast from memory into the vpdpbusd itself. Every lane is still a
 * row and does the same exact integer sums and the same float operations
 * in the same order, so the results are the 256-bit tiles' to the bit.
 *
 * For the processors with AVX-512 VNNI (Zen 4 and Zen 5, Intel from Ice
 * Lake, Sapphire Rapids and the Xeons after it): the prompt is bound by
 * the products, not by the memory, so twice the lanes an instruction is
 * where the width can pay. JANAS_KERNELS=256 keeps the 256-bit tiles.
 */

#define Z_TARGET                                                               \
    __attribute__((                                                            \
        target("avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")))

/* the 32 bytes at a in the low half, those at b in the high one */
static inline Z_TARGET __attribute__((always_inline)) __m512i
z_pair(const void *a, const void *b)
{
    return _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a)),
        _mm256_loadu_si256((const __m256i *)b), 1);
}

/* gemm_ahead for groups of sixteen rows */
static inline Z_TARGET __attribute__((always_inline)) void
z_ahead(const void *w, size_t row_bytes, size_t gb, size_t g, size_t rows,
        size_t nb, size_t b0, size_t chunk)
{
    b0 += chunk;
    if (b0 >= nb) {
        b0 = 0;
        g += 16;
        if (g + 16 > rows)
            return;
    }
    size_t len = (nb - b0 < chunk ? nb - b0 : chunk) * gb;
    for (int j = 0; j < 16; j++) {
        const char *p = (const char *)w + (g + j) * row_bytes + b0 * gb;
        for (size_t i = 0; i < len; i += 64)
            _mm_prefetch(p + i, PF_GEMM_HINT);
    }
}

/* q4kg_tile on sixteen rows: ka the transposed blocks of the first eight,
   kb those of the next eight; af and am 16 floats a token */
static inline Z_TARGET __attribute__((always_inline)) void
q4kz_tile(const struct q4kg_blk *ka, const struct q4kg_blk *kb, int n,
          size_t b0, const struct janas_block_q8k *x, size_t xs, float *af,
          float *am, const int T)
{
    const __m512i low4 = _mm512_set1_epi8(0x0f);
    for (int i = 0; i < n; i++) {
        const struct q4kg_blk *a = ka + i, *bb = kb + i;
        size_t b = b0 + (size_t)i;
        __m512i is[12];
        _Pragma("GCC unroll 12") for (int t = 0; t < T; t++) is[t] =
            _mm512_setzero_si512();
        _Pragma("GCC unroll 4") for (int c = 0; c < 4; c++)
        {
            _Pragma("GCC unroll 2") for (int h = 0; h < 2; h++)
            {
                __m512i a32[12];
                _Pragma("GCC unroll 12") for (int t = 0; t < T; t++) a32[t] =
                    _mm512_setzero_si512();
                _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++)
                {
                    __m512i wv = z_pair(a->qs[c][q], bb->qs[c][q]);
                    wv = h ? _mm512_and_si512(_mm512_srli_epi16(wv, 4), low4)
                           : _mm512_and_si512(wv, low4);
                    _Pragma("GCC unroll 12") for (int t = 0; t < T; t++)
                    {
                        int32_t aw;
                        memcpy(&aw,
                               x[(size_t)t * xs + b].qs + 64 * c + 32 * h +
                                   4 * q,
                               4);
                        a32[t] = _mm512_dpbusd_epi32(a32[t], wv,
                                                     _mm512_set1_epi32(aw));
                    }
                }
                __m512i sc = z_pair(a->sc32[2 * c + h], bb->sc32[2 * c + h]);
                _Pragma("GCC unroll 12") for (int t = 0; t < T; t++) is[t] =
                    _mm512_add_epi32(is[t], _mm512_mullo_epi32(a32[t], sc));
            }
        }
        __m512 dw = _mm512_castsi512_ps(z_pair(a->d, bb->d)),
               dmw = _mm512_castsi512_ps(z_pair(a->dmin, bb->dmin));
        __m512i mn[4];
        for (int kk = 0; kk < 4; kk++)
            mn[kk] = z_pair(a->mn[kk], bb->mn[kk]);
        _Pragma("GCC unroll 12") for (int t = 0; t < T; t++)
        {
            const struct janas_block_q8k *xb = x + (size_t)t * xs + b;
            __m512i im = _mm512_setzero_si512();
            _Pragma("GCC unroll 4") for (int kk = 0; kk < 4; kk++)
            {
                int32_t bw;
                memcpy(&bw, xb->bsums2 + 2 * kk, 4);
                im = _mm512_add_epi32(
                    im, _mm512_madd_epi16(mn[kk], _mm512_set1_epi32(bw)));
            }
            __m512 d8 = _mm512_set1_ps(xb->d);
            _mm512_storeu_ps(af + 16 * t,
                             _mm512_fmadd_ps(_mm512_mul_ps(dw, d8),
                                             _mm512_cvtepi32_ps(is[t]),
                                             _mm512_loadu_ps(af + 16 * t)));
            _mm512_storeu_ps(am + 16 * t,
                             _mm512_fmadd_ps(_mm512_mul_ps(dmw, d8),
                                             _mm512_cvtepi32_ps(im),
                                             _mm512_loadu_ps(am + 16 * t)));
        }
    }
}

#ifndef Q4KZ_T
#define Q4KZ_T 12 /* tokens a tile: 24 accumulators of the 32 registers */
#endif

/* rows (a multiple of 16) of Q4_K against nv vectors, y[r + v ys] */
static Z_TARGET void q4k_gemm_z(const void *w, size_t row_bytes,
                                const struct janas_block_q8k *x, size_t xs,
                                size_t nv, size_t nb, size_t rows, float *y,
                                size_t ys)
{
    _Alignas(64) struct q4kg_blk bk[2][Q4KG_CHUNK];
    _Alignas(64) float af[Q4KG_TOK * 16], am[Q4KG_TOK * 16];
    for (size_t g = 0; g < rows; g += 16) {
        const struct janas_block_q4k *wr[16];
        for (int j = 0; j < 16; j++)
            wr[j] = (const void *)((const char *)w + (g + j) * row_bytes);
        for (size_t v0 = 0; v0 < nv; v0 += Q4KG_TOK) {
            size_t nt = nv - v0 < Q4KG_TOK ? nv - v0 : Q4KG_TOK;
            memset(af, 0, nt * 16 * sizeof(float));
            memset(am, 0, nt * 16 * sizeof(float));
            for (size_t b0 = 0; b0 < nb; b0 += Q4KG_CHUNK) {
                int n = nb - b0 < Q4KG_CHUNK ? (int)(nb - b0) : Q4KG_CHUNK;
                q4kg_fill_evnni(wr, b0, n, bk[0]);
                q4kg_fill_evnni(wr + 8, b0, n, bk[1]);
                z_ahead(w, row_bytes, sizeof(struct janas_block_q4k), g, rows,
                        nb, b0, Q4KG_CHUNK);
                size_t t0 = 0;
                for (; t0 + Q4KZ_T <= nt; t0 += Q4KZ_T)
                    q4kz_tile(bk[0], bk[1], n, b0, x + (v0 + t0) * xs, xs,
                              af + 16 * t0, am + 16 * t0, Q4KZ_T);
                /* the rest: tiles of 4, then 1 to 3 */
                for (; t0 + 4 <= nt; t0 += 4)
                    q4kz_tile(bk[0], bk[1], n, b0, x + (v0 + t0) * xs, xs,
                              af + 16 * t0, am + 16 * t0, 4);
                const struct janas_block_q8k *xt = x + (v0 + t0) * xs;
                if (nt - t0 == 3)
                    q4kz_tile(bk[0], bk[1], n, b0, xt, xs, af + 16 * t0,
                              am + 16 * t0, 3);
                else if (nt - t0 == 2)
                    q4kz_tile(bk[0], bk[1], n, b0, xt, xs, af + 16 * t0,
                              am + 16 * t0, 2);
                else if (nt - t0 == 1)
                    q4kz_tile(bk[0], bk[1], n, b0, xt, xs, af + 16 * t0,
                              am + 16 * t0, 1);
            }
            for (size_t t = 0; t < nt; t++)
                for (int j = 0; j < 16; j++)
                    y[g + j + (v0 + t) * ys] = af[16 * t + j] - am[16 * t + j];
        }
    }
}

/* every token of a run of nt: tiles of Q4KZ_T, then of 4, then 1 to 3;
   CALL(T) sees t0, the tile's first token */
#define Z_TILES(CALL)                                                          \
    do {                                                                       \
        size_t t0 = 0;                                                         \
        for (; t0 + Q4KZ_T <= nt; t0 += Q4KZ_T)                                \
            CALL(Q4KZ_T);                                                      \
        for (; t0 + 4 <= nt; t0 += 4)                                          \
            CALL(4);                                                           \
        if (nt - t0 == 3)                                                      \
            CALL(3);                                                           \
        else if (nt - t0 == 2)                                                 \
            CALL(2);                                                           \
        else if (nt - t0 == 1)                                                 \
            CALL(1);                                                           \
    } while (0)

/* q5kg_tile (M 0, Q5_K: the mins' sums into am) and iq4xsg_tile (M 1,
   IQ4_XS: 128 times the activations' sums taken off) on sixteen rows */
static inline Z_TARGET __attribute__((always_inline)) void
q5kz_tile(const struct q5kg_blk *ka, const struct q5kg_blk *kb, int n,
          size_t b0, const struct janas_block_q8k *x, size_t xs, float *af,
          float *am, const int T, const int M)
{
    for (int i = 0; i < n; i++) {
        const struct q5kg_blk *a = ka + i, *bb = kb + i;
        size_t b = b0 + (size_t)i;
        __m512i is[12];
        _Pragma("GCC unroll 12") for (int t = 0; t < T; t++) is[t] =
            _mm512_setzero_si512();
        for (int sb = 0; sb < 8; sb++) {
            __m512i a32[12];
            _Pragma("GCC unroll 12") for (int t = 0; t < T; t++) a32[t] =
                _mm512_setzero_si512();
            _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++)
            {
                __m512i wv = z_pair(a->qs[8 * sb + q], bb->qs[8 * sb + q]);
                _Pragma("GCC unroll 12") for (int t = 0; t < T; t++)
                {
                    int32_t aw;
                    memcpy(&aw, x[(size_t)t * xs + b].qs + 32 * sb + 4 * q, 4);
                    a32[t] =
                        _mm512_dpbusd_epi32(a32[t], wv, _mm512_set1_epi32(aw));
                }
            }
            __m512i sc = z_pair(a->sc32[sb], bb->sc32[sb]);
            _Pragma("GCC unroll 12") for (int t = 0; t < T; t++) is[t] =
                _mm512_add_epi32(is[t], _mm512_mullo_epi32(a32[t], sc));
        }
        __m512 dw = _mm512_castsi512_ps(z_pair(a->d, bb->d));
        __m512i mn[4];
        for (int kk = 0; kk < 4; kk++)
            mn[kk] = z_pair(a->mn[kk], bb->mn[kk]);
        _Pragma("GCC unroll 12") for (int t = 0; t < T; t++)
        {
            const struct janas_block_q8k *xb = x + (size_t)t * xs + b;
            __m512i c = _mm512_setzero_si512();
            _Pragma("GCC unroll 4") for (int kk = 0; kk < 4; kk++)
            {
                int32_t bw;
                memcpy(&bw, xb->bsums2 + 2 * kk, 4);
                c = _mm512_add_epi32(
                    c, _mm512_madd_epi16(mn[kk], _mm512_set1_epi32(bw)));
            }
            __m512 d8 = _mm512_set1_ps(xb->d);
            __m512i si =
                M ? _mm512_sub_epi32(is[t], _mm512_slli_epi32(c, 7)) : is[t];
            _mm512_storeu_ps(af + 16 * t,
                             _mm512_fmadd_ps(_mm512_mul_ps(dw, d8),
                                             _mm512_cvtepi32_ps(si),
                                             _mm512_loadu_ps(af + 16 * t)));
            if (!M) {
                __m512 dmw = _mm512_castsi512_ps(z_pair(a->dmin, bb->dmin));
                _mm512_storeu_ps(am + 16 * t,
                                 _mm512_fmadd_ps(_mm512_mul_ps(dmw, d8),
                                                 _mm512_cvtepi32_ps(c),
                                                 _mm512_loadu_ps(am + 16 * t)));
            }
        }
    }
}

/* rows (a multiple of 16) of Q5_K (M 0) or IQ4_XS (M 1) against nv
   vectors, y[r + v ys] */
static inline Z_TARGET __attribute__((always_inline)) void
q5k_gemm_z_body(const void *w, size_t row_bytes,
                const struct janas_block_q8k *x, size_t xs, size_t nv,
                size_t nb, size_t rows, float *y, size_t ys, const int M)
{
    _Alignas(64) struct q5kg_blk bk[2][Q5KG_CHUNK];
    _Alignas(64) float af[Q4KG_TOK * 16], am[Q4KG_TOK * 16];
    size_t gb =
        M ? sizeof(struct janas_block_iq4xs) : sizeof(struct janas_block_q5k);
    for (size_t g = 0; g < rows; g += 16) {
        const void *wr[16];
        for (int j = 0; j < 16; j++)
            wr[j] = (const char *)w + (g + j) * row_bytes;
        for (size_t v0 = 0; v0 < nv; v0 += Q4KG_TOK) {
            size_t nt = nv - v0 < Q4KG_TOK ? nv - v0 : Q4KG_TOK;
            memset(af, 0, nt * 16 * sizeof(float));
            if (!M)
                memset(am, 0, nt * 16 * sizeof(float));
            for (size_t b0 = 0; b0 < nb; b0 += Q5KG_CHUNK) {
                int n = nb - b0 < Q5KG_CHUNK ? (int)(nb - b0) : Q5KG_CHUNK;
                for (int h = 0; h < 2; h++)
                    if (M)
                        iq4xsg_fill_evnni(
                            (const struct janas_block_iq4xs *const *)wr + 8 * h,
                            b0, n, bk[h]);
                    else
                        q5kg_fill_evnni(
                            (const struct janas_block_q5k *const *)wr + 8 * h,
                            b0, n, bk[h]);
                z_ahead(w, row_bytes, gb, g, rows, nb, b0, Q5KG_CHUNK);
                const struct janas_block_q8k *xv = x + v0 * xs;
#define Q5KZ(T)                                                                \
    q5kz_tile(bk[0], bk[1], n, b0, xv + t0 * xs, xs, af + 16 * t0,             \
              am + 16 * t0, T, M)
                Z_TILES(Q5KZ);
#undef Q5KZ
            }
            for (size_t t = 0; t < nt; t++)
                for (int j = 0; j < 16; j++)
                    y[g + j + (v0 + t) * ys] =
                        M ? af[16 * t + j] : af[16 * t + j] - am[16 * t + j];
        }
    }
}

static Z_TARGET void q5k_gemm_z(const void *w, size_t row_bytes,
                                const struct janas_block_q8k *x, size_t xs,
                                size_t nv, size_t nb, size_t rows, float *y,
                                size_t ys)
{
    q5k_gemm_z_body(w, row_bytes, x, xs, nv, nb, rows, y, ys, 0);
}

static Z_TARGET void iq4xs_gemm_z(const void *w, size_t row_bytes,
                                  const struct janas_block_q8k *x, size_t xs,
                                  size_t nv, size_t nb, size_t rows, float *y,
                                  size_t ys)
{
    q5k_gemm_z_body(w, row_bytes, x, xs, nv, nb, rows, y, ys, 1);
}

/* q6kg_tile on sixteen rows (its VNNI form: four vpdpbusd a group of 16
   and a multiply by the scales, on every level) */
static inline Z_TARGET __attribute__((always_inline)) void
q6kz_tile(const struct q6kg_blk *ka, const struct q6kg_blk *kb, int n,
          size_t b0, const struct janas_block_q8k *x, size_t xs, float *af,
          float *am, const int T, const int L)
{
    const int shk = L == 0 || L >= 4 ? 0 : L == 3 ? 1 : L == 2 ? 3 : 5;
    const __m512i cv = _mm512_set1_epi32(L == 0   ? 32
                                         : L == 4 ? 4
                                         : L == 5 ? 128
                                         : L == 3 ? 64
                                         : L == 2 ? 61
                                                  : 49);
    const int off = L != 6;
    for (int i = 0; i < n; i++) {
        const struct q6kg_blk *a = ka + i, *bb = kb + i;
        size_t b = b0 + (size_t)i;
        __m512i is[12];
        _Pragma("GCC unroll 12") for (int t = 0; t < T; t++) is[t] =
            _mm512_setzero_si512();
        for (int g = 0; g < 16; g++) {
            __m512i a32[12];
            _Pragma("GCC unroll 12") for (int t = 0; t < T; t++) a32[t] =
                _mm512_setzero_si512();
            _Pragma("GCC unroll 4") for (int q = 0; q < 4; q++)
            {
                __m512i wv = z_pair(a->qs[4 * g + q], bb->qs[4 * g + q]);
                _Pragma("GCC unroll 12") for (int t = 0; t < T; t++)
                {
                    int32_t aw;
                    memcpy(&aw, x[(size_t)t * xs + b].qs + 16 * g + 4 * q, 4);
                    a32[t] =
                        _mm512_dpbusd_epi32(a32[t], wv, _mm512_set1_epi32(aw));
                }
            }
            __m512i sc = z_pair(a->sc32[g], bb->sc32[g]);
            _Pragma("GCC unroll 12") for (int t = 0; t < T; t++) is[t] =
                _mm512_add_epi32(is[t], _mm512_mullo_epi32(a32[t], sc));
        }
        __m512 dw = _mm512_castsi512_ps(z_pair(a->d, bb->d));
        __m512i scp[8];
        for (int kk = 0; kk < 8; kk++)
            scp[kk] = z_pair(a->scp[kk], bb->scp[kk]);
        _Pragma("GCC unroll 12") for (int t = 0; t < T; t++)
        {
            const struct janas_block_q8k *xb = x + (size_t)t * xs + b;
            __m512i mn = _mm512_setzero_si512();
            _Pragma("GCC unroll 8") for (int kk = 0; kk < 8; kk++)
            {
                int32_t bw;
                memcpy(&bw, xb->bsums + 2 * kk, 4);
                mn = _mm512_add_epi32(
                    mn, _mm512_madd_epi16(scp[kk], _mm512_set1_epi32(bw)));
            }
            __m512i si = off ? _mm512_sub_epi32(_mm512_slli_epi32(is[t], shk),
                                                _mm512_mullo_epi32(mn, cv))
                             : is[t];
            __m512 d8 = _mm512_set1_ps(xb->d);
            _mm512_storeu_ps(af + 16 * t,
                             _mm512_fmadd_ps(_mm512_mul_ps(dw, d8),
                                             _mm512_cvtepi32_ps(si),
                                             _mm512_loadu_ps(af + 16 * t)));
            if (L == 6) {
                __m512 dmw = _mm512_castsi512_ps(z_pair(a->dmin, bb->dmin));
                _mm512_storeu_ps(am + 16 * t,
                                 _mm512_fmadd_ps(_mm512_mul_ps(dmw, d8),
                                                 _mm512_cvtepi32_ps(mn),
                                                 _mm512_loadu_ps(am + 16 * t)));
            }
        }
    }
}

/* q6k_gemm_body and iqk_gemm_t on sixteen rows: L 0 Q6_K, 1-3 Q6_K_P's
   level (p1, p0 its planes, pb bytes a row), 4 Q3_K, 5 and 6 the type TY
   (an IQ type with a grid, Q2_K) */
static inline Z_TARGET __attribute__((always_inline)) void
q6k_gemm_z_body(const void *w, size_t row_bytes, const uint8_t *p1,
                const uint8_t *p0, size_t pb, const struct janas_block_q8k *x,
                size_t xs, size_t nv, size_t nb, size_t rows, float *y,
                size_t ys, const int L, const int TY)
{
    _Alignas(64) struct q6kg_blk bk[2][Q6KG_CHUNK];
    _Alignas(64) float af[Q4KG_TOK * 16], am[Q4KG_TOK * 16];
    size_t gb = L == 0   ? sizeof(struct janas_block_q6k)
                : L == 4 ? sizeof(struct janas_block_q3k)
                : L >= 5 ? iqkg_size_evnni(TY)
                         : row_bytes / nb;
    for (size_t g = 0; g < rows; g += 16) {
        const void *wr[16];
        const uint8_t *r1[16], *r0[16];
        for (int j = 0; j < 16; j++) {
            wr[j] = (const char *)w + (g + j) * row_bytes;
            r1[j] = p1 ? p1 + (g + j) * pb : NULL;
            r0[j] = p0 ? p0 + (g + j) * pb : NULL;
        }
        for (size_t v0 = 0; v0 < nv; v0 += Q4KG_TOK) {
            size_t nt = nv - v0 < Q4KG_TOK ? nv - v0 : Q4KG_TOK;
            memset(af, 0, nt * 16 * sizeof(float));
            if (L == 6)
                memset(am, 0, nt * 16 * sizeof(float));
            for (size_t b0 = 0; b0 < nb; b0 += Q6KG_CHUNK) {
                int n = nb - b0 < Q6KG_CHUNK ? (int)(nb - b0) : Q6KG_CHUNK;
                for (int h = 0; h < 2; h++) {
                    const void *const *wh = wr + 8 * h;
                    if (L == 0)
                        q6kg_fill_q6k_evnni(
                            (const struct janas_block_q6k *const *)wh, b0, n,
                            bk[h]);
                    else if (L == 4)
                        q6kg_fill_q3k_evnni(
                            (const struct janas_block_q3k *const *)wh, b0, n,
                            bk[h]);
                    else if (L >= 5)
                        iqkg_fill_evnni(TY, (const uint8_t *const *)wh, b0, n,
                                        bk[h]);
                    else
                        q6kg_fill_q6kp_evnni(
                            (const struct janas_block_q6kp *const *)wh,
                            r1 + 8 * h, r0 + 8 * h, b0, n, bk[h], L);
                }
                z_ahead(w, row_bytes, gb, g, rows, nb, b0, Q6KG_CHUNK);
                const struct janas_block_q8k *xv = x + v0 * xs;
#define Q6KZ(T)                                                                \
    q6kz_tile(bk[0], bk[1], n, b0, xv + t0 * xs, xs, af + 16 * t0,             \
              am + 16 * t0, T, L)
                Z_TILES(Q6KZ);
#undef Q6KZ
            }
            for (size_t t = 0; t < nt; t++)
                for (int j = 0; j < 16; j++)
                    y[g + j + (v0 + t) * ys] =
                        L == 6 ? af[16 * t + j] - am[16 * t + j]
                               : af[16 * t + j];
        }
    }
}

/* q6k_gemm on sixteen rows (level: q6k_gemm's) */
static Z_TARGET void q6k_gemm_z(const void *w, size_t row_bytes,
                                const uint8_t *p1, const uint8_t *p0, size_t pb,
                                const struct janas_block_q8k *x, size_t xs,
                                size_t nv, size_t nb, size_t rows, float *y,
                                size_t ys, int level)
{
    switch (level) {
#define Q6KZ_LEVEL(L)                                                          \
    case L:                                                                    \
        q6k_gemm_z_body(w, row_bytes, p1, p0, pb, x, xs, nv, nb, rows, y, ys,  \
                        L, 0);                                                 \
        break;
        Q6KZ_LEVEL(0)
        Q6KZ_LEVEL(1)
        Q6KZ_LEVEL(2)
        Q6KZ_LEVEL(4)
#undef Q6KZ_LEVEL
    default:
        q6k_gemm_z_body(w, row_bytes, p1, p0, pb, x, xs, nv, nb, rows, y, ys, 3,
                        0);
    }
}

/* iqk_gemm on sixteen rows: 1, or 0 for another type */
static Z_TARGET int iqk_gemm_z(int type, const void *w, size_t row_bytes,
                               const struct janas_block_q8k *x, size_t xs,
                               size_t nv, size_t nb, size_t rows, float *y,
                               size_t ys)
{
    switch (type) {
#define IQKZ(T)                                                                \
    case T:                                                                    \
        q6k_gemm_z_body(w, row_bytes, NULL, NULL, 0, x, xs, nv, nb, rows, y,   \
                        ys, T == JANAS_Q2_K ? 6 : 5, T);                       \
        return 1;
        IQKZ(JANAS_Q2_K)
        IQKZ(JANAS_IQ3_S)
        IQKZ(JANAS_IQ3_XXS)
        IQKZ(JANAS_IQ2_XXS)
        IQKZ(JANAS_IQ2_XS)
        IQKZ(JANAS_IQ2_S)
        IQKZ(JANAS_IQ1_S)
        IQKZ(JANAS_IQ1_M)
#undef IQKZ
    }
    return 0;
}

/* b32g_tile on sixteen rows: af and am 128 floats a token, [position][row] */
static inline Z_TARGET __attribute__((always_inline)) void
b32z_tile(const int TY, const struct b32g_blk *ka, const struct b32g_blk *kb2,
          int n, size_t b0, const struct janas_block_q8k *x, size_t xs,
          float *af, float *am, const int T)
{
    const int o = b32g_off_evnni(TY);
    const int dm = TY == JANAS_Q4_1 || TY == JANAS_Q5_1;
    for (int i = 0; i < n; i++) {
        const struct b32g_blk *a = ka + i, *bb = kb2 + i;
        size_t b = b0 + (size_t)i;
        for (int kb = 0; kb < 8; kb++) {
            __m512i s[12];
            _Pragma("GCC unroll 12") for (int t = 0; t < T; t++) s[t] =
                _mm512_setzero_si512();
            _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++)
            {
                __m512i wv = z_pair(a->qs[kb][q], bb->qs[kb][q]);
                _Pragma("GCC unroll 12") for (int t = 0; t < T; t++)
                {
                    int32_t aw;
                    memcpy(&aw, x[(size_t)t * xs + b].qs32 + 32 * kb + 4 * q,
                           4);
                    s[t] = _mm512_dpbusd_epi32(s[t], wv, _mm512_set1_epi32(aw));
                }
            }
            __m512 dv = _mm512_castsi512_ps(z_pair(a->d[kb], bb->d[kb]));
            __m512 mv = _mm512_castsi512_ps(z_pair(a->m[kb], bb->m[kb]));
            _Pragma("GCC unroll 12") for (int t = 0; t < T; t++)
            {
                const struct janas_block_q8k *xb = x + (size_t)t * xs + b;
                int asum = xb->bsums32[2 * kb] + xb->bsums32[2 * kb + 1];
                __m512 d8 = _mm512_set1_ps(xb->d32[kb]);
                float *pa = af + 128 * t + 16 * kb;
                __m512i is =
                    o ? _mm512_sub_epi32(s[t], _mm512_set1_epi32(o * asum))
                      : s[t];
                _mm512_storeu_ps(pa, _mm512_fmadd_ps(_mm512_mul_ps(dv, d8),
                                                     _mm512_cvtepi32_ps(is),
                                                     _mm512_loadu_ps(pa)));
                if (dm) {
                    float *pm = am + 128 * t + 16 * kb;
                    _mm512_storeu_ps(
                        pm, _mm512_fmadd_ps(_mm512_mul_ps(mv, d8),
                                            _mm512_set1_ps((float)asum),
                                            _mm512_loadu_ps(pm)));
                }
            }
        }
    }
}

/* b32g_hsum8 on sixteen rows */
static inline Z_TARGET __attribute__((always_inline)) __m512
b32z_hsum8(const float *a)
{
    __m512 p0 = _mm512_loadu_ps(a), p1 = _mm512_loadu_ps(a + 16),
           p2 = _mm512_loadu_ps(a + 32), p3 = _mm512_loadu_ps(a + 48),
           p4 = _mm512_loadu_ps(a + 64), p5 = _mm512_loadu_ps(a + 80),
           p6 = _mm512_loadu_ps(a + 96), p7 = _mm512_loadu_ps(a + 112);
    return _mm512_add_ps(
        _mm512_add_ps(_mm512_add_ps(p0, p4), _mm512_add_ps(p2, p6)),
        _mm512_add_ps(_mm512_add_ps(p1, p5), _mm512_add_ps(p3, p7)));
}

/* tokens a run: B32G_TOK's half, the sums of sixteen rows twice as many */
#define B32Z_TOK 128

static inline Z_TARGET __attribute__((always_inline)) void
b32_gemm_z_t(const int TY, const void *w, size_t row_bytes,
             const struct janas_block_q8k *x, size_t xs, size_t nv, size_t nb,
             size_t rows, float *y, size_t ys)
{
    _Alignas(64) struct b32g_blk bk[2][B32G_CHUNK];
    _Alignas(64) float af[B32Z_TOK * 128], am[B32Z_TOK * 128];
    const int dm = TY == JANAS_Q4_1 || TY == JANAS_Q5_1;
    for (size_t g = 0; g < rows; g += 16) {
        const uint8_t *wr[16];
        for (int j = 0; j < 16; j++)
            wr[j] = (const uint8_t *)w + (g + j) * row_bytes;
        for (size_t v0 = 0; v0 < nv; v0 += B32Z_TOK) {
            size_t nt = nv - v0 < B32Z_TOK ? nv - v0 : B32Z_TOK;
            memset(af, 0, nt * 128 * sizeof(float));
            if (dm)
                memset(am, 0, nt * 128 * sizeof(float));
            for (size_t b0 = 0; b0 < nb; b0 += B32G_CHUNK) {
                int n = nb - b0 < B32G_CHUNK ? (int)(nb - b0) : B32G_CHUNK;
                b32g_fill_evnni(TY, wr, b0, n, bk[0]);
                b32g_fill_evnni(TY, wr + 8, b0, n, bk[1]);
                if (TY != JANAS_Q8_0)
                    z_ahead(w, row_bytes,
                            TY == JANAS_Q2_0
                                ? 4 * sizeof(struct janas_block_q2_0)
                                : 8 * b32g_size_evnni(TY),
                            g, rows, nb, b0, B32G_CHUNK);
                const struct janas_block_q8k *xv = x + v0 * xs;
#define B32Z(T)                                                                \
    b32z_tile(TY, bk[0], bk[1], n, b0, xv + t0 * xs, xs, af + 128 * t0,        \
              am + 128 * t0, T)
                Z_TILES(B32Z);
#undef B32Z
            }
            for (size_t t = 0; t < nt; t++) {
                float r[16], rm[16];
                _mm512_storeu_ps(r, b32z_hsum8(af + 128 * t));
                if (dm)
                    _mm512_storeu_ps(rm, b32z_hsum8(am + 128 * t));
                for (int j = 0; j < 16; j++)
                    y[g + j + (v0 + t) * ys] = dm ? r[j] + rm[j] : r[j];
            }
        }
    }
}

/* b32_gemm on sixteen rows: 1, or 0 for another type */
static Z_TARGET int b32_gemm_z(int type, const void *w, size_t row_bytes,
                               const struct janas_block_q8k *x, size_t xs,
                               size_t nv, size_t nb, size_t rows, float *y,
                               size_t ys)
{
    switch (type) {
#define B32Z(T)                                                                \
    case T:                                                                    \
        b32_gemm_z_t(T, w, row_bytes, x, xs, nv, nb, rows, y, ys);             \
        return 1;
        B32Z(JANAS_Q8_0)
        B32Z(JANAS_IQ4_NL)
        B32Z(JANAS_Q4_0)
        B32Z(JANAS_MXFP4)
        B32Z(JANAS_Q5_0)
        B32Z(JANAS_Q4_1)
        B32Z(JANAS_Q5_1)
        B32Z(JANAS_Q2_0)
#undef B32Z
    }
    return 0;
}
