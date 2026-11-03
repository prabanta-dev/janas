/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * quant_dotn.h - the quantized dot products on nv = 1 to 8 vectors
 * (private to quant.c, which includes it once per instruction set).
 *
 * Register-tiled for blocks of tokens: the loop runs over 64-weight pieces of
 * a block outside and over the vectors inside, with nv a constant in each
 * specialisation, so the unpacked weights and one int32 accumulator per
 * vector stay in registers. The includer defines:
 *   DOTN(name)          the name with the instruction-set suffix
 *   DOTN_TARGET         the function attribute selecting the instructions
 *   DOTN_ACC(acc, p, s) acc + the int16 pairs p times the int16 scales s,
 *                       summed pairwise into int32 lanes: madd + add on
 *                       AVX2, a single vpdpwssd on AVX-VNNI
 *   DOTN_US2(u0, s0, u1, s1)  the unsigned bytes u0 times the signed s0
 *                       plus u1 times s1, four by four into int32 lanes:
 *                       maddubs, add, madd on AVX2 (the products must fit
 *                       int16 two by two), two vpdpbusd on AVX-VNNI
 *   DOTN_DPBUSD(acc, u, s)  with DOTN_HAS_VNNI: vpdpbusd, in its AVX-VNNI
 *                       form or in AVX-512 VNNI's on 256 bits (Zen 4, the
 *                       Intel cores with AVX-512 before AVX-VNNI)
 * Integer sums are exact, so every int32 lane holds the same value with
 * either definition, and the float operations are the same, in the same
 * order: all variants, single or multi-vector, give bit-identical results
 * (for Q4_K also identical to the scalar reference and the GPU kernels).
 */

/* The eight lanes of v summed (exact). */
static inline DOTN_TARGET __attribute__((always_inline)) int
DOTN(hsum_epi32)(__m256i v)
{
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    return _mm_cvtsi128_si32(s);
}

/* The eight lanes of a and of b summed (exact), together: *ra and *rb. */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(hsum2_epi32)(__m256i a, __m256i b, int *ra, int *rb)
{
    __m256i h = _mm256_hadd_epi32(a, b);
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h),
                              _mm256_extracti128_si256(h, 1));
    s = _mm_hadd_epi32(s, s);
    *ra = _mm_cvtsi128_si32(s);
    *rb = _mm_extract_epi32(s, 1);
}

static inline DOTN_TARGET __attribute__((always_inline)) int
DOTN(hsum128_epi32)(__m128i s)
{
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    return _mm_cvtsi128_si32(s);
}

/*
 * The weights asked for ahead of the product, PF_AHEAD bytes on, into L1:
 * one prefetch per cache line of what a step of the loop reads, on one or
 * two vectors (on more the product is bound by its arithmetic, and the
 * extra instructions cost a little). The hardware prefetcher alone kept
 * too few lines in flight: on 12 threads Q4_K read 70 GB/s where Q8_0,
 * which asks for its bytes faster, read 84; with 2 KB ahead Q4_K read 82
 * (1, 4 and 8 KB gave 81-82, and 4-8 KB slowed a single core down: 7 Oct
 * 2026). Rows follow one another in memory, so past a row's end the lines
 * are the next row's.
 */
#ifndef PF_AHEAD
#define PF_AHEAD 2048
#endif
#ifndef PF_GEMM_HINT
#define PF_GEMM_HINT _MM_HINT_T0
#endif
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(prefetch_ahead)(const void *at, size_t size)
{
    const char *p = (const char *)at + PF_AHEAD;
    for (size_t i = 0; i < size; i += 64)
        _mm_prefetch(p + i, _MM_HINT_T0);
}

/*
 * The tiles' weights asked for ahead: after a chunk of eight rows is
 * transposed, the next chunk's bytes (the same rows' next blocks, or the
 * next eight rows' first ones) are fetched while this one's tiles compute.
 * Without it the transposition waited for memory and the tiles left it idle
 * in turn: 62% of the transposition's time on its loads (perf, Q4_K on 24
 * vectors). gb is the bytes of a group of 256 weights, chunk the groups
 * transposed at a time.
 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(gemm_ahead)(const void *w, size_t row_bytes, size_t gb, size_t g,
                 size_t rows, size_t nb, size_t b0, size_t chunk)
{
    b0 += chunk;
    if (b0 >= nb) {
        b0 = 0;
        g += 8;
        if (g + 8 > rows)
            return;
    }
    size_t len = (nb - b0 < chunk ? nb - b0 : chunk) * gb;
    for (int j = 0; j < 8; j++) {
        const char *p = (const char *)w + (g + j) * row_bytes + b0 * gb;
        for (size_t i = 0; i < len; i += 64)
            _mm_prefetch(p + i, PF_GEMM_HINT);
    }
}

/*
 * Q4_K: the per-block integer-sum scheme. For each block and vector the
 * integer sums are exact: isum = sum of scale x (weight . activation) over
 * the eight 32-weight pieces, imin = sum of min x activation pair sums; then
 * acc = fma(d * d8, isum, acc) and accm = fma(dmin * d8, imin, accm), and
 * the result is acc - accm. The same float operations as the scalar
 * reference and the GPU kernels, so every implementation agrees bit for
 * bit. From three vectors on, the sums of all the vectors are reduced
 * together and their accumulators share a register, one lane per vector.
 */
/* What the products of a row on 5-8 vectors need of the vectors alone,
   made once for all the rows (q4k_multi_rows): for each group of 256 the
   vectors' scales, then their bsums2 two by two, int16 pairs with one
   int32 lane a vector, so that the eight vectors' sums of mins x
   activations are four madd and not eight and six hadd. */
#define Q4K_PRE 160 /* bytes a group of 256: 8 floats, 4 x 8 pairs */

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q4k_pre)(const struct janas_block_q8k *x, size_t xs, int nv, size_t nb,
              uint8_t *pre)
{
    for (size_t b = 0; b < nb; b++) {
        float *d8 = (float *)(pre + Q4K_PRE * b);
        int16_t *pr = (int16_t *)(pre + Q4K_PRE * b + 32);
        for (int v = 0; v < 8; v++) {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            d8[v] = v < nv ? xb->d : 0.0f;
            for (int i = 0; i < 4; i++)
                for (int h = 0; h < 2; h++)
                    pr[16 * i + 2 * v + h] =
                        v < nv ? xb->bsums2[2 * i + h] : (int16_t)0;
        }
    }
}

/* From the vectors made ready (q4k_pre), group b's: their scales and the
   sums of the block's eight mins times their activations, a lane each. */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(k_pre_mins)(const uint8_t *pb, __m128i mins16, __m256 *d8, __m256i *im8)
{
    const __m256i *pr = (const __m256i *)(pb + 32);
    __m256i mm = _mm256_broadcastsi128_si256(mins16);
    *d8 = _mm256_loadu_ps((const float *)pb);
    *im8 = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_madd_epi16(_mm256_loadu_si256(pr),
                                           _mm256_shuffle_epi32(mm, 0x00)),
                         _mm256_madd_epi16(_mm256_loadu_si256(pr + 1),
                                           _mm256_shuffle_epi32(mm, 0x55))),
        _mm256_add_epi32(_mm256_madd_epi16(_mm256_loadu_si256(pr + 2),
                                           _mm256_shuffle_epi32(mm, 0xaa)),
                         _mm256_madd_epi16(_mm256_loadu_si256(pr + 3),
                                           _mm256_shuffle_epi32(mm, 0xff))));
}

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q4k_dotn_body)(const struct janas_block_q4k *w,
                    const struct janas_block_q8k *x, size_t xs, size_t nb,
                    float *out, const int nv, const uint8_t *pre)
{
    const __m256i low4 = _mm256_set1_epi8(0x0f);
    /* up to 2 vectors: lanes acc0, acc1, accm0, accm1; 3 or 4: one lane
       each in acc4n and accm4n */
    __m128 acc4 = _mm_setzero_ps(), acc4n = _mm_setzero_ps(),
           accm4n = _mm_setzero_ps();
    __m256 acc8 = _mm256_setzero_ps(), accm8 = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(&w[b], sizeof(w[b]));
        uint32_t u[4];
        unpack_scales_u32(w[b].scales, u);
        __m128i sm = _mm_loadu_si128((const __m128i *)u);
        __m256i sc16 = _mm256_broadcastsi128_si256(_mm_cvtepu8_epi16(sm));
        /* up to 2 vectors, a second accumulator for the high nibbles
           halves the dependency chain (integer sums: same lanes) */
        __m256i sumi[8], sumh[2];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) sumi[v] =
            _mm256_setzero_si256();
        /* the bound apart: GCC 13 drops the annotation of a loop whose
           condition holds a ?: and warns (issue #4) */
        const int nh = nv <= 2 ? nv : 0;
        _Pragma("GCC unroll 4") for (int v = 0; v < nh; v++) sumh[v] =
            _mm256_setzero_si256();
        _Pragma("GCC unroll 4") for (int c = 0; c < 4; c++)
        {
            __m256i q4 =
                _mm256_loadu_si256((const __m256i *)(w[b].qs + 32 * c));
            __m256i lo = _mm256_and_si256(q4, low4);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q4, 4), low4);
            __m256i s0 = _mm256_shuffle_epi8(
                sc16, _mm256_set1_epi16((short)((4 * c + 1) << 8 | (4 * c))));
            __m256i s1 = _mm256_shuffle_epi8(
                sc16,
                _mm256_set1_epi16((short)((4 * c + 3) << 8 | (4 * c + 2))));
            _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
            {
                const int8_t *a = x[(size_t)v * xs + b].qs + 64 * c;
                __m256i a0 = _mm256_loadu_si256((const __m256i *)a);
                __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + 32));
                sumi[v] = DOTN_ACC(sumi[v], _mm256_maddubs_epi16(lo, a0), s0);
                if (nv <= 2)
                    sumh[v] =
                        DOTN_ACC(sumh[v], _mm256_maddubs_epi16(hi, a1), s1);
                else
                    sumi[v] =
                        DOTN_ACC(sumi[v], _mm256_maddubs_epi16(hi, a1), s1);
            }
        }
        float dw = _cvtsh_ss(w[b].d), dmw = _cvtsh_ss(w[b].dmin);
        __m128i mins16 = _mm_cvtepu8_epi16(_mm_srli_si128(sm, 8));
        __m128i pm[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++)
        {
            pm[v] = _mm_setzero_si128();
            if (v < nv && !(pre && nv > 4)) {
                const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
                pm[v] = _mm_madd_epi16(
                    mins16, _mm_loadu_si128((const __m128i *)xb->bsums2));
            }
        }
        if (nv <= 2) {
            __m256i z = _mm256_setzero_si256();
            __m256i a = _mm256_hadd_epi32(
                _mm256_add_epi32(sumi[0], sumh[0]),
                nv == 2 ? _mm256_add_epi32(sumi[1], sumh[1]) : z);
            __m128i h = _mm_add_epi32(_mm256_castsi256_si128(a),
                                      _mm256_extracti128_si256(a, 1));
            /* isum0, isum1, imin0, imin1 */
            __m128i t = _mm_hadd_epi32(h, _mm_hadd_epi32(pm[0], pm[1]));
            float d80 = x[b].d, d81 = nv == 2 ? x[xs + b].d : 0.0f;
            __m128 d8 = _mm_setr_ps(d80, d81, d80, d81);
            __m128 dd = _mm_setr_ps(dw, dw, dmw, dmw);
            acc4 = _mm_fmadd_ps(_mm_mul_ps(dd, d8), _mm_cvtepi32_ps(t), acc4);
        } else if (nv <= 4) {
            /* the sums of four vectors, one lane each: half the reduction */
            __m256i t = _mm256_hadd_epi32(_mm256_hadd_epi32(sumi[0], sumi[1]),
                                          _mm256_hadd_epi32(sumi[2], sumi[3]));
            __m128i is4 = _mm_add_epi32(_mm256_castsi256_si128(t),
                                        _mm256_extracti128_si256(t, 1));
            __m128i im4 = _mm_hadd_epi32(_mm_hadd_epi32(pm[0], pm[1]),
                                         _mm_hadd_epi32(pm[2], pm[3]));
            __m128 d8 = _mm_setr_ps(x[b].d, x[xs + b].d, x[2 * xs + b].d,
                                    nv == 4 ? x[3 * xs + b].d : 0.0f);
            acc4n = _mm_fmadd_ps(_mm_mul_ps(_mm_set1_ps(dw), d8),
                                 _mm_cvtepi32_ps(is4), acc4n);
            accm4n = _mm_fmadd_ps(_mm_mul_ps(_mm_set1_ps(dmw), d8),
                                  _mm_cvtepi32_ps(im4), accm4n);
        } else {
            /* the sums of the eight vectors, one lane each */
            __m256i s01 = _mm256_hadd_epi32(sumi[0], sumi[1]);
            __m256i s23 = _mm256_hadd_epi32(sumi[2], sumi[3]);
            __m256i s45 = _mm256_hadd_epi32(sumi[4], sumi[5]);
            __m256i s67 = _mm256_hadd_epi32(sumi[6], sumi[7]);
            __m256i t0 = _mm256_hadd_epi32(s01, s23);
            __m256i t1 = _mm256_hadd_epi32(s45, s67);
            __m256i is8 =
                _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20),
                                 _mm256_permute2x128_si256(t0, t1, 0x31));
            __m256i im8;
            __m256 d8;
            if (pre) {
                DOTN(k_pre_mins)(pre + Q4K_PRE * b, mins16, &d8, &im8);
            } else {
                __m128i m03 = _mm_hadd_epi32(_mm_hadd_epi32(pm[0], pm[1]),
                                             _mm_hadd_epi32(pm[2], pm[3]));
                __m128i m47 = _mm_hadd_epi32(_mm_hadd_epi32(pm[4], pm[5]),
                                             _mm_hadd_epi32(pm[6], pm[7]));
                im8 = _mm256_set_m128i(m47, m03);
#define D8(v) (v < nv ? x[(size_t)(v) * xs + b].d : 0.0f)
                d8 = _mm256_setr_ps(D8(0), D8(1), D8(2), D8(3), D8(4), D8(5),
                                    D8(6), D8(7));
#undef D8
            }
            acc8 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dw), d8),
                                   _mm256_cvtepi32_ps(is8), acc8);
            accm8 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dmw), d8),
                                    _mm256_cvtepi32_ps(im8), accm8);
        }
    }
    if (nv <= 2) {
        float r[4];
        _mm_storeu_ps(r, acc4);
        out[0] = r[0] - r[2];
        if (nv == 2)
            out[1] = r[1] - r[3];
    } else if (nv <= 4) {
        float r[4];
        _mm_storeu_ps(r, _mm_sub_ps(acc4n, accm4n));
        _Pragma("GCC unroll 4") for (int v = 0; v < nv; v++) out[v] = r[v];
    } else {
        float r[8];
        _mm256_storeu_ps(r, _mm256_sub_ps(acc8, accm8));
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = r[v];
    }
}

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q5k_dotn_body)(const struct janas_block_q5k *w,
                    const struct janas_block_q8k *x, size_t xs, size_t nb,
                    float *out, const int nv, const uint8_t *pre)
{
    const __m256i low4 = _mm256_set1_epi8(0x0f);
    const __m256i one = _mm256_set1_epi8(1);
    /* up to 2 vectors: lanes acc0, acc1, accm0, accm1 */
    __m128 acc4 = _mm_setzero_ps();
    __m256 acc8 = _mm256_setzero_ps(), accm8 = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(&w[b], sizeof(w[b]));
        uint32_t u[4];
        unpack_scales_u32(w[b].scales, u);
        __m128i sm = _mm_loadu_si128((const __m128i *)u);
        __m256i sc16 = _mm256_broadcastsi128_si256(_mm_cvtepu8_epi16(sm));
        /* up to 2 vectors, a second accumulator for the high nibbles
           halves the dependency chain (integer sums: same lanes) */
        __m256i sumi[8], sumh[2];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) sumi[v] =
            _mm256_setzero_si256();
        /* the bound apart: GCC 13 drops the annotation of a loop whose
           condition holds a ?: and warns (issue #4) */
        const int nh = nv <= 2 ? nv : 0;
        _Pragma("GCC unroll 4") for (int v = 0; v < nh; v++) sumh[v] =
            _mm256_setzero_si256();
        __m256i qh = _mm256_loadu_si256((const __m256i *)w[b].qh);
        _Pragma("GCC unroll 4") for (int c = 0; c < 4; c++)
        {
            __m256i q4 =
                _mm256_loadu_si256((const __m256i *)(w[b].qs + 32 * c));
            __m256i lo = _mm256_and_si256(q4, low4);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q4, 4), low4);
            /* the 5th bit of sub-blocks 2c and 2c + 1 */
            lo = _mm256_or_si256(
                lo,
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qh, 2 * c), one), 4));
            hi = _mm256_or_si256(
                hi, _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(qh, 2 * c + 1), one),
                        4));
            __m256i s0 = _mm256_shuffle_epi8(
                sc16, _mm256_set1_epi16((short)((4 * c + 1) << 8 | (4 * c))));
            __m256i s1 = _mm256_shuffle_epi8(
                sc16,
                _mm256_set1_epi16((short)((4 * c + 3) << 8 | (4 * c + 2))));
            _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
            {
                const int8_t *a = x[(size_t)v * xs + b].qs + 64 * c;
                __m256i a0 = _mm256_loadu_si256((const __m256i *)a);
                __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + 32));
                sumi[v] = DOTN_ACC(sumi[v], _mm256_maddubs_epi16(lo, a0), s0);
                if (nv <= 2)
                    sumh[v] =
                        DOTN_ACC(sumh[v], _mm256_maddubs_epi16(hi, a1), s1);
                else
                    sumi[v] =
                        DOTN_ACC(sumi[v], _mm256_maddubs_epi16(hi, a1), s1);
            }
        }
        float dw = _cvtsh_ss(w[b].d), dmw = _cvtsh_ss(w[b].dmin);
        __m128i mins16 = _mm_cvtepu8_epi16(_mm_srli_si128(sm, 8));
        __m128i pm[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++)
        {
            pm[v] = _mm_setzero_si128();
            if (v < nv && !(pre && nv > 4)) {
                const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
                pm[v] = _mm_madd_epi16(
                    mins16, _mm_loadu_si128((const __m128i *)xb->bsums2));
            }
        }
        if (nv <= 2) {
            __m256i z = _mm256_setzero_si256();
            __m256i a = _mm256_hadd_epi32(
                _mm256_add_epi32(sumi[0], sumh[0]),
                nv == 2 ? _mm256_add_epi32(sumi[1], sumh[1]) : z);
            __m128i h = _mm_add_epi32(_mm256_castsi256_si128(a),
                                      _mm256_extracti128_si256(a, 1));
            /* isum0, isum1, imin0, imin1 */
            __m128i t = _mm_hadd_epi32(h, _mm_hadd_epi32(pm[0], pm[1]));
            float d80 = x[b].d, d81 = nv == 2 ? x[xs + b].d : 0.0f;
            __m128 d8 = _mm_setr_ps(d80, d81, d80, d81);
            __m128 dd = _mm_setr_ps(dw, dw, dmw, dmw);
            acc4 = _mm_fmadd_ps(_mm_mul_ps(dd, d8), _mm_cvtepi32_ps(t), acc4);
        } else {
            /* the sums of the eight vectors, one lane each */
            __m256i s01 = _mm256_hadd_epi32(sumi[0], sumi[1]);
            __m256i s23 = _mm256_hadd_epi32(sumi[2], sumi[3]);
            __m256i s45 = _mm256_hadd_epi32(sumi[4], sumi[5]);
            __m256i s67 = _mm256_hadd_epi32(sumi[6], sumi[7]);
            __m256i t0 = _mm256_hadd_epi32(s01, s23);
            __m256i t1 = _mm256_hadd_epi32(s45, s67);
            __m256i is8 =
                _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20),
                                 _mm256_permute2x128_si256(t0, t1, 0x31));
            __m256i im8;
            __m256 d8;
            if (pre && nv > 4) {
                DOTN(k_pre_mins)(pre + Q4K_PRE * b, mins16, &d8, &im8);
            } else {
                __m128i m03 = _mm_hadd_epi32(_mm_hadd_epi32(pm[0], pm[1]),
                                             _mm_hadd_epi32(pm[2], pm[3]));
                __m128i m47 = _mm_hadd_epi32(_mm_hadd_epi32(pm[4], pm[5]),
                                             _mm_hadd_epi32(pm[6], pm[7]));
                im8 = _mm256_set_m128i(m47, m03);
#define D8(v) (v < nv ? x[(size_t)(v) * xs + b].d : 0.0f)
                d8 = _mm256_setr_ps(D8(0), D8(1), D8(2), D8(3), D8(4), D8(5),
                                    D8(6), D8(7));
#undef D8
            }
            acc8 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dw), d8),
                                   _mm256_cvtepi32_ps(is8), acc8);
            accm8 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dmw), d8),
                                    _mm256_cvtepi32_ps(im8), accm8);
        }
    }
    if (nv <= 2) {
        float r[4];
        _mm_storeu_ps(r, acc4);
        out[0] = r[0] - r[2];
        if (nv == 2)
            out[1] = r[1] - r[3];
    } else {
        float r[8];
        _mm256_storeu_ps(r, _mm256_sub_ps(acc8, accm8));
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = r[v];
    }
}

/*
 * Q8_0: eight 32-weight blocks per 256 activations. Each block's sum is
 * exact; each of the eight block positions has its own float accumulator
 * (one lane per position), and the eight are added at the end as
 * janas_hsum8 does, exactly as the scalar reference.
 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q8_0_dotn_body)(const struct janas_block_q8_0 *w,
                     const struct janas_block_q8k *x, size_t xs, size_t nb,
                     float *out, const int nv)
{
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) acc[v] =
        _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(w + 8 * b, 8 * sizeof(*w));
        const struct janas_block_q8_0 *wb = w + 8 * b;
        __m256 dk = _mm256_cvtph_ps(_mm_setr_epi16(
            (short)wb[0].d, (short)wb[1].d, (short)wb[2].d, (short)wb[3].d,
            (short)wb[4].d, (short)wb[5].d, (short)wb[6].d, (short)wb[7].d));
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i s[8];
            _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
            {
                __m256i q = _mm256_loadu_si256((const __m256i *)wb[k].qs);
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs32 + 32 * k));
                /* |w| . (a with the sign of w): maddubs wants unsigned */
                __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(q),
                                                 _mm256_sign_epi8(a, q));
                s[k] = _mm256_madd_epi16(p, ones);
            }
            __m256i s01 = _mm256_hadd_epi32(s[0], s[1]);
            __m256i s23 = _mm256_hadd_epi32(s[2], s[3]);
            __m256i s45 = _mm256_hadd_epi32(s[4], s[5]);
            __m256i s67 = _mm256_hadd_epi32(s[6], s[7]);
            __m256i t0 = _mm256_hadd_epi32(s01, s23);
            __m256i t1 = _mm256_hadd_epi32(s45, s67);
            __m256i is8 =
                _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20),
                                 _mm256_permute2x128_si256(t0, t1, 0x31));
            acc[v] =
                _mm256_fmadd_ps(_mm256_mul_ps(dk, _mm256_loadu_ps(xb->d32)),
                                _mm256_cvtepi32_ps(is8), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
    {
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(acc[v]),
                              _mm256_extractf128_ps(acc[v], 1));
        t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        t = _mm_add_ss(t, _mm_movehdup_ps(t));
        out[v] = _mm_cvtss_f32(t);
    }
}

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q6k_dotn_body)(const struct janas_block_q6k *w,
                    const struct janas_block_q8k *x, size_t xs, size_t nb,
                    float *out, const int nv)
{
    const __m256i low4 = _mm256_set1_epi8(0x0f);
    const __m256i low2 = _mm256_set1_epi8(0x03);
    /* the same scheme as Q4_K, with no mins: the -32 offset of the weights
       folds into the exact block sum */
    /* the scale pair of a quarter, selected out of the block's sixteen: the
       low lane gets the scale of its first group, the high lane that of the
       second. Four masks serve the eight quarters, since the second four read
       the other half of the scales. */
    const __m256i smask[4] = {
        _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3,
                         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3),
        _mm256_setr_epi8(4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7,
                         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7),
        _mm256_setr_epi8(8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11,
                         10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
                         11),
        _mm256_setr_epi8(12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
                         12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
                         14, 15, 14, 15)};
    __m128 acc4 = _mm_setzero_ps();
    __m256 acc8 = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(&w[b], sizeof(w[b]));
        const int8_t *sc = w[b].scales;
        /* the sixteen scales once, as int16, and the two halves spread over
           both lanes so that one shuffle builds a quarter's pair */
        __m256i sc16b =
            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sc));
        __m256i sclo = _mm256_permute4x64_epi64(sc16b, 0x44);
        __m256i schi = _mm256_permute4x64_epi64(sc16b, 0xee);
        __m256i sv[8];
        _Pragma("GCC unroll 4") for (int j = 0; j < 4; j++)
        {
            sv[j] = _mm256_shuffle_epi8(sclo, smask[j]);
            sv[j + 4] = _mm256_shuffle_epi8(schi, smask[j]);
        }
        __m256i sumi[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) sumi[v] =
            _mm256_setzero_si256();
        _Pragma("GCC unroll 2") for (int n = 0; n < 2; n++)
        {
            __m256i ql0 =
                _mm256_loadu_si256((const __m256i *)(w[b].ql + 64 * n));
            __m256i ql1 =
                _mm256_loadu_si256((const __m256i *)(w[b].ql + 64 * n + 32));
            __m256i qh =
                _mm256_loadu_si256((const __m256i *)(w[b].qh + 32 * n));
            __m256i q[4];
            q[0] = _mm256_or_si256(
                _mm256_and_si256(ql0, low4),
                _mm256_slli_epi16(_mm256_and_si256(qh, low2), 4));
            q[1] = _mm256_or_si256(
                _mm256_and_si256(ql1, low4),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qh, 2), low2), 4));
            q[2] = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql0, 4), low4),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qh, 4), low2), 4));
            q[3] = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql1, 4), low4),
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srli_epi16(qh, 6), low2), 4));
            _Pragma("GCC unroll 4") for (int part = 0; part < 4; part++)
            {
                __m256i s16 = sv[4 * n + part];
                _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
                {
                    __m256i a = _mm256_loadu_si256(
                        (const __m256i *)(x[(size_t)v * xs + b].qs + 128 * n +
                                          32 * part));
                    sumi[v] = DOTN_ACC(sumi[v],
                                       _mm256_maddubs_epi16(q[part], a), s16);
                }
            }
        }
        __m256i sc16w =
            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sc));
        float dw = _cvtsh_ss(w[b].d);
        __m256i si[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++)
        {
            si[v] = _mm256_setzero_si256();
            if (v < nv) {
                const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
                __m256i bs = _mm256_loadu_si256((const __m256i *)xb->bsums);
                __m256i off =
                    _mm256_slli_epi32(_mm256_madd_epi16(sc16w, bs), 5);
                si[v] = _mm256_sub_epi32(sumi[v], off);
            }
        }
        if (nv <= 2) {
            __m256i a = _mm256_hadd_epi32(si[0], si[1]);
            __m128i h = _mm_add_epi32(_mm256_castsi256_si128(a),
                                      _mm256_extracti128_si256(a, 1));
            __m128i t = _mm_hadd_epi32(h, h); /* isum0, isum1, ... */
            float d80 = x[b].d, d81 = nv == 2 ? x[xs + b].d : 0.0f;
            __m128 d8 = _mm_setr_ps(d80, d81, 0.0f, 0.0f);
            acc4 = _mm_fmadd_ps(_mm_mul_ps(_mm_set1_ps(dw), d8),
                                _mm_cvtepi32_ps(t), acc4);
        } else {
            __m256i s01 = _mm256_hadd_epi32(si[0], si[1]);
            __m256i s23 = _mm256_hadd_epi32(si[2], si[3]);
            __m256i s45 = _mm256_hadd_epi32(si[4], si[5]);
            __m256i s67 = _mm256_hadd_epi32(si[6], si[7]);
            __m256i t0 = _mm256_hadd_epi32(s01, s23);
            __m256i t1 = _mm256_hadd_epi32(s45, s67);
            __m256i is8 =
                _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20),
                                 _mm256_permute2x128_si256(t0, t1, 0x31));
#define D8(v) (v < nv ? x[(size_t)(v) * xs + b].d : 0.0f)
            __m256 d8 = _mm256_setr_ps(D8(0), D8(1), D8(2), D8(3), D8(4), D8(5),
                                       D8(6), D8(7));
#undef D8
            acc8 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dw), d8),
                                   _mm256_cvtepi32_ps(is8), acc8);
        }
    }
    float r[8];
    if (nv <= 2)
        _mm_storeu_ps(r, acc4);
    else
        _mm256_storeu_ps(r, acc8);
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = r[v];
}

static DOTN_TARGET void DOTN(q4k_dotn)(const struct janas_block_q4k *w,
                                       const struct janas_block_q8k *x,
                                       size_t xs, int nv, size_t nb, float *out)
{
    switch (nv) {
    case 8:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 8, NULL);
        break;
    case 7:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 7, NULL);
        break;
    case 6:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 6, NULL);
        break;
    case 5:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 5, NULL);
        break;
    case 4:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 4, NULL);
        break;
    case 3:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 3, NULL);
        break;
    case 2:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 2, NULL);
        break;
    default:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 1, NULL);
    }
}

static DOTN_TARGET void DOTN(q6k_dotn)(const struct janas_block_q6k *w,
                                       const struct janas_block_q8k *x,
                                       size_t xs, int nv, size_t nb, float *out)
{
    switch (nv) {
    case 8:
        DOTN(q6k_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(q6k_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(q6k_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(q6k_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(q6k_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(q6k_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(q6k_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(q6k_dotn_body)(w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(q4k_dot)(const struct janas_block_q4k *w,
                                       const struct janas_block_q8k *x,
                                       size_t nb)
{
    float r;
    DOTN(q4k_dotn_body)(w, x, 0, nb, &r, 1, NULL);
    return r;
}

static DOTN_TARGET float DOTN(q6k_dot)(const struct janas_block_q6k *w,
                                       const struct janas_block_q8k *x,
                                       size_t nb)
{
    float r;
    DOTN(q6k_dotn_body)(w, x, 0, nb, &r, 1);
    return r;
}

static DOTN_TARGET void DOTN(q5k_dotn)(const struct janas_block_q5k *w,
                                       const struct janas_block_q8k *x,
                                       size_t xs, int nv, size_t nb, float *out)
{
    switch (nv) {
    case 8:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 8, NULL);
        break;
    case 7:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 7, NULL);
        break;
    case 6:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 6, NULL);
        break;
    case 5:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 5, NULL);
        break;
    case 4:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 4, NULL);
        break;
    case 3:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 3, NULL);
        break;
    case 2:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 2, NULL);
        break;
    default:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 1, NULL);
    }
}

static DOTN_TARGET void DOTN(q8_0_dotn)(const struct janas_block_q8_0 *w,
                                        const struct janas_block_q8k *x,
                                        size_t xs, int nv, size_t nb,
                                        float *out)
{
    switch (nv) {
    case 8:
        DOTN(q8_0_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(q8_0_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(q8_0_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(q8_0_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(q8_0_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(q8_0_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(q8_0_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(q8_0_dotn_body)(w, x, xs, nb, out, 1);
    }
}

/* Q4_K and Q5_K rows against nv vectors, y[r + v ys]: eight vectors at a
   time with what they need of themselves made once for all the rows
   (q4k_pre); four or fewer as janas_dot_multi takes them. The same bits. */
#define Q4K_PRE_MAX 128 /* groups of 256 on the stack: 20 KB */
#define K_MULTI_ROWS(name, T)                                                  \
    static DOTN_TARGET void DOTN(name##_multi_rows)(                           \
        const void *w, size_t row_bytes, const struct janas_block_q8k *x,      \
        size_t xs, size_t nv, size_t nb, size_t rows, float *y, size_t ys)     \
    {                                                                          \
        _Alignas(32) uint8_t pre[Q4K_PRE_MAX * Q4K_PRE];                       \
        for (size_t v0 = 0; v0 < nv; v0 += 8) {                                \
            int n = nv - v0 < 8 ? (int)(nv - v0) : 8;                          \
            const struct janas_block_q8k *xv = x + v0 * xs;                    \
            int with = n > 4 && rows > 1 && nb <= Q4K_PRE_MAX;                 \
            if (with)                                                          \
                DOTN(q4k_pre)(xv, xs, n, nb, pre);                             \
            for (size_t r = 0; r < rows; r++) {                                \
                const T *wr = (const void *)((const char *)w + r * row_bytes); \
                float o[8];                                                    \
                if (!with)                                                     \
                    DOTN(name##_dotn)(wr, xv, xs, n, nb, o);                   \
                else if (n == 8)                                               \
                    DOTN(name##_dotn_body)(wr, xv, xs, nb, o, 8, pre);         \
                else if (n == 7)                                               \
                    DOTN(name##_dotn_body)(wr, xv, xs, nb, o, 7, pre);         \
                else if (n == 6)                                               \
                    DOTN(name##_dotn_body)(wr, xv, xs, nb, o, 6, pre);         \
                else                                                           \
                    DOTN(name##_dotn_body)(wr, xv, xs, nb, o, 5, pre);         \
                for (int i = 0; i < n; i++)                                    \
                    y[r + (v0 + (size_t)i) * ys] = o[i];                       \
            }                                                                  \
        }                                                                      \
    }
K_MULTI_ROWS(q4k, struct janas_block_q4k)
K_MULTI_ROWS(q5k, struct janas_block_q5k)
#undef K_MULTI_ROWS

/*
 * Q4_K as a product of tiles, for a prompt's blocks: eight rows by T
 * tokens. The products above hold a row's 32-weight piece in one register
 * against a vector's 32 activations, so every maddubs needs its own madd to
 * weigh the piece by its scale, and the two share the same ports: 32
 * products a cycle at best, about 15 measured on a core (Core Ultra 7 155H,
 * AVX2). Here the eight rows are transposed four bytes at a time into a
 * small buffer (q4kg_blk, a chunk of blocks at a time, once for every 64
 * tokens), so that a register holds four weights of each of eight rows and
 * the token's four activations are broadcast against it: a 32-weight
 * piece is eight maddubs, summed in int16 (8 x 2 x 15 x 128 = 30720, no
 * overflow), then one madd by the eight rows' scales - and every lane is a
 * row, so no horizontal sums. The integer sums are the same exact numbers
 * and the floats the same operations in the same order (d x d8 times the
 * sum, fused into the accumulator, the mins' apart, their difference at
 * the end), so the result is the other kernels' to the bit.
 */
#ifndef Q4KG_DEFINED
#define Q4KG_DEFINED
#define Q4KG_CHUNK 16 /* blocks transposed at a time: 23 KB */
/* tokens whose sums are held at a time: a prompt's block of 256 at once,
   its rows transposed once (with 64 four times: bench_kernels on 256
   vectors, 12 threads, Q5_K 1.6 to 1.75 GB/s, IQ4_XS 1.25 to 1.45) */
#define Q4KG_TOK 256
struct q4kg_blk {
    uint32_t qs[4][8][8]; /* [piece][dword][row] */
    int16_t sc[8][16];    /* [sub-block][2 row + i]: its scale, twice */
    int16_t mn[4][16];    /* [k][2 row + i]: mins 2k and 2k + 1 */
    int32_t sc32[8][8];   /* [sub-block][row]: the scale (AVX-VNNI's) */
    float d[8], dmin[8];
};
#endif

/* The eight scales and eight mins of a Q4_K or Q5_K block of each of the
   eight rows (s[r]: scales 0-7, mins 0-7, as unpack_scales_u32 lays them)
   into the tiles' tables: an 8 x 8 transpose of their pairs gives the
   mins' pairs (mn), and the scales' bytes, zero extended, sc32 (AVX-VNNI)
   or sc twice a row */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(kmin_scales8)(int16_t (*sc)[16], int32_t (*sc32)[8], int16_t (*mn)[16],
                   const __m128i *s, const int sgn)
{
    __m128i b[8], c[8], t[8];
    for (int r = 0; r < 8; r += 2) {
        b[r] = _mm_unpacklo_epi16(s[r], s[r + 1]);
        b[r + 1] = _mm_unpackhi_epi16(s[r], s[r + 1]);
    }
    for (int h = 0; h < 8; h += 4) {
        c[h] = _mm_unpacklo_epi32(b[h], b[h + 2]);
        c[h + 1] = _mm_unpackhi_epi32(b[h], b[h + 2]);
        c[h + 2] = _mm_unpacklo_epi32(b[h + 1], b[h + 3]);
        c[h + 3] = _mm_unpackhi_epi32(b[h + 1], b[h + 3]);
    }
    for (int q = 0; q < 4; q++) {
        t[2 * q] = _mm_unpacklo_epi64(c[q], c[q + 4]);
        t[2 * q + 1] = _mm_unpackhi_epi64(c[q], c[q + 4]);
    }
    /* t[kk]: word r is row r's scales 2kk and 2kk + 1, t[4 + kk] its mins */
    const __m128i split =
        _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    for (int kk = 0; kk < 4; kk++) {
        _mm256_storeu_si256((__m256i *)mn[kk],
                            sgn ? _mm256_cvtepi8_epi16(t[4 + kk])
                                : _mm256_cvtepu8_epi16(t[4 + kk]));
        __m128i e = _mm_shuffle_epi8(t[kk], split);
#ifdef DOTN_HAS_VNNI
        (void)sc;
        __m128i e1 = _mm_srli_si128(e, 8);
        _mm256_storeu_si256((__m256i *)sc32[2 * kk],
                            sgn ? _mm256_cvtepi8_epi32(e)
                                : _mm256_cvtepu8_epi32(e));
        _mm256_storeu_si256((__m256i *)sc32[2 * kk + 1],
                            sgn ? _mm256_cvtepi8_epi32(e1)
                                : _mm256_cvtepu8_epi32(e1));
#else
        (void)sc32;
        for (int h = 0; h < 2; h++) {
            __m128i eh = h ? _mm_srli_si128(e, 8) : e;
            __m128i v = sgn ? _mm_cvtepi8_epi16(eh) : _mm_cvtepu8_epi16(eh);
            _mm256_storeu_si256((__m256i *)sc[2 * kk + h],
                                _mm256_set_m128i(_mm_unpackhi_epi16(v, v),
                                                 _mm_unpacklo_epi16(v, v)));
        }
#endif
    }
}

/* Blocks [b0, b0 + n) of rows w[0..7] into bk */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q4kg_fill)(const struct janas_block_q4k *const *w, size_t b0, int n,
                struct q4kg_blk *bk)
{
    for (int i = 0; i < n; i++) {
        struct q4kg_blk *k = bk + i;
        for (int c = 0; c < 4; c++) {
            __m256i r[8], t[8], u[8];
            _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++) r[j] =
                _mm256_loadu_si256((const __m256i *)(w[j][b0 + i].qs + 32 * c));
            _Pragma("GCC unroll 4") for (int j = 0; j < 8; j += 2)
            {
                t[j] = _mm256_unpacklo_epi32(r[j], r[j + 1]);
                t[j + 1] = _mm256_unpackhi_epi32(r[j], r[j + 1]);
            }
            _Pragma("GCC unroll 2") for (int j = 0; j < 8; j += 4)
            {
                u[j] = _mm256_unpacklo_epi64(t[j], t[j + 2]);
                u[j + 1] = _mm256_unpackhi_epi64(t[j], t[j + 2]);
                u[j + 2] = _mm256_unpacklo_epi64(t[j + 1], t[j + 3]);
                u[j + 3] = _mm256_unpackhi_epi64(t[j + 1], t[j + 3]);
            }
            _Pragma("GCC unroll 4") for (int q = 0; q < 4; q++)
            {
                _mm256_storeu_si256(
                    (__m256i *)k->qs[c][q],
                    _mm256_permute2x128_si256(u[q], u[q + 4], 0x20));
                _mm256_storeu_si256(
                    (__m256i *)k->qs[c][q + 4],
                    _mm256_permute2x128_si256(u[q], u[q + 4], 0x31));
            }
        }
        __m128i sv[8];
        for (int j = 0; j < 8; j++) {
            const struct janas_block_q4k *blk = &w[j][b0 + i];
            uint32_t u[4];
            unpack_scales_u32(blk->scales, u);
            sv[j] = _mm_loadu_si128((const __m128i *)u);
            k->d[j] = _cvtsh_ss(blk->d);
            k->dmin[j] = _cvtsh_ss(blk->dmin);
        }
        DOTN(kmin_scales8)(k->sc, k->sc32, k->mn, sv, 0);
    }
}

/* T tokens x[0..T) (xs blocks apart) on n transposed blocks from block b0:
   the eight rows' sums added into af and am (8 floats a token) */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q4kg_tile)(const struct q4kg_blk *bk, int n, size_t b0,
                const struct janas_block_q8k *x, size_t xs, float *af,
                float *am, const int T)
{
    const __m256i low4 = _mm256_set1_epi8(0x0f);
    for (int i = 0; i < n; i++) {
        const struct q4kg_blk *k = bk + i;
        size_t b = b0 + (size_t)i;
        __m256i is[8];
        _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
            _mm256_setzero_si256();
        _Pragma("GCC unroll 4") for (int c = 0; c < 4; c++)
        {
            _Pragma("GCC unroll 2") for (int h = 0; h < 2; h++)
            {
#ifdef DOTN_HAS_VNNI
                /* four products summed into the row's int32 lane at
                   once, the piece's scale by one multiply */
                __m256i a32[8];
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) a32[t] =
                    _mm256_setzero_si256();
                _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++)
                {
                    __m256i wv =
                        _mm256_loadu_si256((const __m256i *)k->qs[c][q]);
                    wv = h ? _mm256_and_si256(_mm256_srli_epi16(wv, 4), low4)
                           : _mm256_and_si256(wv, low4);
                    _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                    {
                        int32_t aw;
                        memcpy(&aw,
                               x[(size_t)t * xs + b].qs + 64 * c + 32 * h +
                                   4 * q,
                               4);
                        a32[t] = DOTN_DPBUSD(a32[t], wv, _mm256_set1_epi32(aw));
                    }
                }
                __m256i sc =
                    _mm256_loadu_si256((const __m256i *)k->sc32[2 * c + h]);
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
                    _mm256_add_epi32(is[t], _mm256_mullo_epi32(a32[t], sc));
#else
                __m256i a16[8];
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) a16[t] =
                    _mm256_setzero_si256();
                _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++)
                {
                    __m256i wv =
                        _mm256_loadu_si256((const __m256i *)k->qs[c][q]);
                    wv = h ? _mm256_and_si256(_mm256_srli_epi16(wv, 4), low4)
                           : _mm256_and_si256(wv, low4);
                    _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                    {
                        int32_t aw;
                        memcpy(&aw,
                               x[(size_t)t * xs + b].qs + 64 * c + 32 * h +
                                   4 * q,
                               4);
                        a16[t] = _mm256_add_epi16(
                            a16[t],
                            _mm256_maddubs_epi16(wv, _mm256_set1_epi32(aw)));
                    }
                }
                __m256i sc =
                    _mm256_loadu_si256((const __m256i *)k->sc[2 * c + h]);
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
                    DOTN_ACC(is[t], a16[t], sc);
#endif
            }
        }
        __m256 dw = _mm256_loadu_ps(k->d), dmw = _mm256_loadu_ps(k->dmin);
        _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
        {
            const struct janas_block_q8k *xb = x + (size_t)t * xs + b;
            __m256i im = _mm256_setzero_si256();
            _Pragma("GCC unroll 4") for (int kk = 0; kk < 4; kk++)
            {
                int32_t bw;
                memcpy(&bw, xb->bsums2 + 2 * kk, 4);
                im = _mm256_add_epi32(
                    im, _mm256_madd_epi16(
                            _mm256_loadu_si256((const __m256i *)k->mn[kk]),
                            _mm256_set1_epi32(bw)));
            }
            __m256 d8 = _mm256_set1_ps(xb->d);
            _mm256_storeu_ps(af + 8 * t,
                             _mm256_fmadd_ps(_mm256_mul_ps(dw, d8),
                                             _mm256_cvtepi32_ps(is[t]),
                                             _mm256_loadu_ps(af + 8 * t)));
            _mm256_storeu_ps(am + 8 * t,
                             _mm256_fmadd_ps(_mm256_mul_ps(dmw, d8),
                                             _mm256_cvtepi32_ps(im),
                                             _mm256_loadu_ps(am + 8 * t)));
        }
    }
}

#ifndef Q4KG_T
#define Q4KG_T 6 /* tokens a tile */
#endif

/* rows (a multiple of 8) of Q4_K against nv vectors, y[r + v ys] */
static DOTN_TARGET void DOTN(q4k_gemm)(const void *w, size_t row_bytes,
                                       const struct janas_block_q8k *x,
                                       size_t xs, size_t nv, size_t nb,
                                       size_t rows, float *y, size_t ys)
{
    _Alignas(32) struct q4kg_blk bk[Q4KG_CHUNK];
    _Alignas(32) float af[Q4KG_TOK * 8], am[Q4KG_TOK * 8];
    for (size_t g = 0; g < rows; g += 8) {
        const struct janas_block_q4k *wr[8];
        for (int j = 0; j < 8; j++)
            wr[j] = (const void *)((const char *)w + (g + j) * row_bytes);
        for (size_t v0 = 0; v0 < nv; v0 += Q4KG_TOK) {
            size_t nt = nv - v0 < Q4KG_TOK ? nv - v0 : Q4KG_TOK;
            memset(af, 0, nt * 8 * sizeof(float));
            memset(am, 0, nt * 8 * sizeof(float));
            for (size_t b0 = 0; b0 < nb; b0 += Q4KG_CHUNK) {
                int n = nb - b0 < Q4KG_CHUNK ? (int)(nb - b0) : Q4KG_CHUNK;
                DOTN(q4kg_fill)(wr, b0, n, bk);
                DOTN(gemm_ahead)(w, row_bytes, sizeof(struct janas_block_q4k),
                                 g, rows, nb, b0, Q4KG_CHUNK);
                size_t t0 = 0;
                for (; t0 + Q4KG_T <= nt; t0 += Q4KG_T)
                    DOTN(q4kg_tile)(bk, n, b0, x + (v0 + t0) * xs, xs,
                                    af + 8 * t0, am + 8 * t0, Q4KG_T);
                /* the rest: a tile of 4, then 1 to 3 */
                for (; t0 + 4 <= nt; t0 += 4)
                    DOTN(q4kg_tile)(bk, n, b0, x + (v0 + t0) * xs, xs,
                                    af + 8 * t0, am + 8 * t0, 4);
                const struct janas_block_q8k *xt = x + (v0 + t0) * xs;
                if (nt - t0 == 3)
                    DOTN(q4kg_tile)(bk, n, b0, xt, xs, af + 8 * t0, am + 8 * t0,
                                    3);
                else if (nt - t0 == 2)
                    DOTN(q4kg_tile)(bk, n, b0, xt, xs, af + 8 * t0, am + 8 * t0,
                                    2);
                else if (nt - t0 == 1)
                    DOTN(q4kg_tile)(bk, n, b0, xt, xs, af + 8 * t0, am + 8 * t0,
                                    1);
            }
            for (size_t t = 0; t < nt; t++)
                for (int j = 0; j < 8; j++)
                    y[g + j + (v0 + t) * ys] = af[8 * t + j] - am[8 * t + j];
        }
    }
}

static DOTN_TARGET float DOTN(q5k_dot)(const struct janas_block_q5k *w,
                                       const struct janas_block_q8k *x,
                                       size_t nb)
{
    float r;
    DOTN(q5k_dotn_body)(w, x, 0, nb, &r, 1, NULL);
    return r;
}

static DOTN_TARGET float DOTN(q8_0_dot)(const struct janas_block_q8_0 *w,
                                        const struct janas_block_q8k *x,
                                        size_t nb)
{
    float r;
    DOTN(q8_0_dotn_body)(w, x, 0, nb, &r, 1);
    return r;
}

/*
 * IQ4_NL and IQ4_XS: the sixteen levels in both lanes of a register, and
 * pshufb turns 32 indices into 32 signed weights; from there the products
 * are Q8_0's. The unpacked weights of a block are kept for all the vectors.
 */
static inline DOTN_TARGET __attribute__((always_inline)) __m256i
DOTN(iq4_unpack32)(const uint8_t *q, __m256i levels)
{
    const __m128i low4 = _mm_set1_epi8(0x0f);
    __m128i b = _mm_loadu_si128((const __m128i *)q);
    __m128i lo = _mm_and_si128(b, low4);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), low4);
    return _mm256_shuffle_epi8(levels, _mm256_set_m128i(hi, lo));
}

static inline DOTN_TARGET __attribute__((always_inline)) __m256i
DOTN(iq4_levels)(void)
{
    return _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)janas_iq4_values));
}

/* IQ4_NL and Q4_0 (the same layout, other levels): as Q8_0 - each block's
   exact sum, one float lane per block position, the eight lanes added at
   the end as the reference does. */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(nib_dotn_body)(const struct janas_block_iq4nl *w,
                    const struct janas_block_q8k *x, size_t xs, size_t nb,
                    float *out, const int nv, const int8_t *table)
{
    const __m256i ones = _mm256_set1_epi16(1),
                  levels = _mm256_broadcastsi128_si256(
                      _mm_loadu_si128((const __m128i *)table));
    __m256 acc[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) acc[v] =
        _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(w + 8 * b, 8 * sizeof(*w));
        const struct janas_block_iq4nl *wb = w + 8 * b;
        __m256 dk = _mm256_cvtph_ps(_mm_setr_epi16(
            (short)wb[0].d, (short)wb[1].d, (short)wb[2].d, (short)wb[3].d,
            (short)wb[4].d, (short)wb[5].d, (short)wb[6].d, (short)wb[7].d));
        __m256i q[8], aq[8];
        _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
        {
            q[k] = DOTN(iq4_unpack32)(wb[k].qs, levels);
            aq[k] = _mm256_abs_epi8(q[k]);
        }
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i s[8];
            _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs32 + 32 * k));
                __m256i p =
                    _mm256_maddubs_epi16(aq[k], _mm256_sign_epi8(a, q[k]));
                s[k] = _mm256_madd_epi16(p, ones);
            }
            __m256i s01 = _mm256_hadd_epi32(s[0], s[1]);
            __m256i s23 = _mm256_hadd_epi32(s[2], s[3]);
            __m256i s45 = _mm256_hadd_epi32(s[4], s[5]);
            __m256i s67 = _mm256_hadd_epi32(s[6], s[7]);
            __m256i t0 = _mm256_hadd_epi32(s01, s23);
            __m256i t1 = _mm256_hadd_epi32(s45, s67);
            __m256i is8 =
                _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20),
                                 _mm256_permute2x128_si256(t0, t1, 0x31));
            acc[v] =
                _mm256_fmadd_ps(_mm256_mul_ps(dk, _mm256_loadu_ps(xb->d32)),
                                _mm256_cvtepi32_ps(is8), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
    {
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(acc[v]),
                              _mm256_extractf128_ps(acc[v], 1));
        t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        t = _mm_add_ss(t, _mm_movehdup_ps(t));
        out[v] = _mm_cvtss_f32(t);
    }
}

/* IQ4_XS: the eight groups' products times their scales summed exactly in
   int32 lanes (DOTN_ACC: vpdpwssd on AVX-VNNI), then one fma per block, as
   the reference. */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(iq4xs_dotn_body)(const struct janas_block_iq4xs *w,
                      const struct janas_block_q8k *x, size_t xs, size_t nb,
                      float *out, const int nv)
{
    const __m256i levels = DOTN(iq4_levels)();
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(&w[b], sizeof(w[b]));
        __m256i q[8], aq[8], sc[8];
        _Pragma("GCC unroll 8") for (int g = 0; g < 8; g++)
        {
            q[g] = DOTN(iq4_unpack32)(w[b].qs + 16 * g, levels);
            aq[g] = _mm256_abs_epi8(q[g]);
            sc[g] = _mm256_set1_epi16((short)iq4xs_scale(&w[b], g));
        }
        float d = _cvtsh_ss(w[b].d);
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i sum = _mm256_setzero_si256();
            _Pragma("GCC unroll 8") for (int g = 0; g < 8; g++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * g));
                __m256i p =
                    _mm256_maddubs_epi16(aq[g], _mm256_sign_epi8(a, q[g]));
                sum = DOTN_ACC(sum, p, sc[g]);
            }
            acc[v] = fmaf(d * xb->d, (float)DOTN(hsum_epi32)(sum), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = acc[v];
}

static DOTN_TARGET void DOTN(iq4nl_dotn)(const struct janas_block_iq4nl *w,
                                         const struct janas_block_q8k *x,
                                         size_t xs, int nv, size_t nb,
                                         float *out)
{
    switch (nv) {
    case 8:
        DOTN(nib_dotn_body)(w, x, xs, nb, out, 8, janas_iq4_values);
        break;
    case 7:
        DOTN(nib_dotn_body)(w, x, xs, nb, out, 7, janas_iq4_values);
        break;
    case 6:
        DOTN(nib_dotn_body)(w, x, xs, nb, out, 6, janas_iq4_values);
        break;
    case 5:
        DOTN(nib_dotn_body)(w, x, xs, nb, out, 5, janas_iq4_values);
        break;
    case 4:
        DOTN(nib_dotn_body)(w, x, xs, nb, out, 4, janas_iq4_values);
        break;
    case 3:
        DOTN(nib_dotn_body)(w, x, xs, nb, out, 3, janas_iq4_values);
        break;
    case 2:
        DOTN(nib_dotn_body)(w, x, xs, nb, out, 2, janas_iq4_values);
        break;
    default:
        DOTN(nib_dotn_body)(w, x, xs, nb, out, 1, janas_iq4_values);
    }
}

static DOTN_TARGET float DOTN(iq4nl_dot)(const struct janas_block_iq4nl *w,
                                         const struct janas_block_q8k *x,
                                         size_t nb)
{
    float r;
    DOTN(nib_dotn_body)(w, x, 0, nb, &r, 1, janas_iq4_values);
    return r;
}

static DOTN_TARGET void DOTN(iq4xs_dotn)(const struct janas_block_iq4xs *w,
                                         const struct janas_block_q8k *x,
                                         size_t xs, int nv, size_t nb,
                                         float *out)
{
    switch (nv) {
    case 8:
        DOTN(iq4xs_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(iq4xs_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(iq4xs_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(iq4xs_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(iq4xs_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(iq4xs_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(iq4xs_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(iq4xs_dotn_body)(w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(iq4xs_dot)(const struct janas_block_iq4xs *w,
                                         const struct janas_block_q8k *x,
                                         size_t nb)
{
    float r;
    DOTN(iq4xs_dotn_body)(w, x, 0, nb, &r, 1);
    return r;
}

/*
 * Q6_K in bit planes, at the level given by how many planes are in memory:
 * three, two or one, that is six, four or two bits per weight. Levels 2 and 1
 * stand the missing bits at the centre of what they cover, half a step, so
 * the integer block sum is doubled and the float factor halved: exact in
 * integers, and at level 3 the same float operations as the Q6_K kernel, in
 * the same order, hence the same result bit for bit.
 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q6kp_dotn_body)(const struct janas_block_q6kp *w, const uint8_t *p1,
                     const uint8_t *p0, const struct janas_block_q8k *x,
                     size_t xs, size_t nb, float *out, const int nv,
                     const int level)
{
    const __m256i low2 = _mm256_set1_epi8(0x03);
    /* 2 isum = (1 << shk) sumi - c sum(scale x group sums) */
    const int shk = level == 3 ? 1 : level == 2 ? 3 : 5;
    const __m256i cv = _mm256_set1_epi32(level == 3   ? 64
                                         : level == 2 ? 61
                                                      : 49);
    /* the scale pair of a quarter, selected out of the block's sixteen: the
       low lane gets the scale of its first group, the high lane that of the
       second. Four masks serve the eight quarters, since the second four read
       the other half of the scales. */
    const __m256i smask[4] = {
        _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3,
                         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3),
        _mm256_setr_epi8(4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7,
                         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7),
        _mm256_setr_epi8(8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11,
                         10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
                         11),
        _mm256_setr_epi8(12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
                         12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
                         14, 15, 14, 15)};
    __m128 acc4 = _mm_setzero_ps();
    __m256 acc8 = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2) {
            DOTN(prefetch_ahead)(&w[b], sizeof(w[b]));
            if (level >= 2)
                DOTN(prefetch_ahead)(p1 + b * JANAS_Q6KP_PLANE,
                                     JANAS_Q6KP_PLANE);
            if (level >= 3)
                DOTN(prefetch_ahead)(p0 + b * JANAS_Q6KP_PLANE,
                                     JANAS_Q6KP_PLANE);
        }
        const int8_t *sc = w[b].scales;
        /* the sixteen scales once, as int16, and the two halves spread over
           both lanes so that one shuffle builds a quarter's pair */
        __m256i sc16b =
            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sc));
        __m256i sclo = _mm256_permute4x64_epi64(sc16b, 0x44);
        __m256i schi = _mm256_permute4x64_epi64(sc16b, 0xee);
        __m256i sv[8];
        _Pragma("GCC unroll 4") for (int j = 0; j < 4; j++)
        {
            sv[j] = _mm256_shuffle_epi8(sclo, smask[j]);
            sv[j + 4] = _mm256_shuffle_epi8(schi, smask[j]);
        }
        __m256i sumi[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) sumi[v] =
            _mm256_setzero_si256();
        _Pragma("GCC unroll 2") for (int n = 0; n < 2; n++)
        {
            __m256i h = _mm256_loadu_si256((const __m256i *)(w[b].q + 32 * n));
            __m256i m1 =
                level >= 2
                    ? _mm256_loadu_si256(
                          (const __m256i *)(p1 + b * JANAS_Q6KP_PLANE + 32 * n))
                    : _mm256_setzero_si256();
            __m256i m0 =
                level >= 3
                    ? _mm256_loadu_si256(
                          (const __m256i *)(p0 + b * JANAS_Q6KP_PLANE + 32 * n))
                    : _mm256_setzero_si256();
            /* the two bits of plane p for the 32 weights of a quarter */
#define Q6KP_BITS(reg, sh) _mm256_and_si256(_mm256_srli_epi16(reg, sh), low2)
#define Q6KP_PART(sh)                                                          \
    (level == 3                                                                \
         ? _mm256_or_si256(                                                    \
               _mm256_or_si256(_mm256_slli_epi16(Q6KP_BITS(h, sh), 4),         \
                               _mm256_slli_epi16(Q6KP_BITS(m1, sh), 2)),       \
               Q6KP_BITS(m0, sh))                                              \
     : level == 2 ? _mm256_or_si256(_mm256_slli_epi16(Q6KP_BITS(h, sh), 2),    \
                                    Q6KP_BITS(m1, sh))                         \
                  : Q6KP_BITS(h, sh))
            __m256i q[4];
            q[0] = Q6KP_PART(0);
            q[1] = Q6KP_PART(2);
            q[2] = Q6KP_PART(4);
            q[3] = Q6KP_PART(6);
#undef Q6KP_PART
#undef Q6KP_BITS
            _Pragma("GCC unroll 4") for (int part = 0; part < 4; part++)
            {
                __m256i s16 = sv[4 * n + part];
                _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
                {
                    __m256i a = _mm256_loadu_si256(
                        (const __m256i *)(x[(size_t)v * xs + b].qs + 128 * n +
                                          32 * part));
                    sumi[v] = DOTN_ACC(sumi[v],
                                       _mm256_maddubs_epi16(q[part], a), s16);
                }
            }
        }
        __m256i sc16w =
            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sc));
        float dw = _cvtsh_ss(w[b].d) * 0.5f;
        __m256i si[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++)
        {
            si[v] = _mm256_setzero_si256();
            if (v < nv) {
                const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
                __m256i bs = _mm256_loadu_si256((const __m256i *)xb->bsums);
                __m256i off =
                    _mm256_mullo_epi32(_mm256_madd_epi16(sc16w, bs), cv);
                si[v] = _mm256_sub_epi32(_mm256_slli_epi32(sumi[v], shk), off);
            }
        }
        if (nv <= 2) {
            __m256i a = _mm256_hadd_epi32(si[0], si[1]);
            __m128i hl = _mm_add_epi32(_mm256_castsi256_si128(a),
                                       _mm256_extracti128_si256(a, 1));
            __m128i t = _mm_hadd_epi32(hl, hl);
            float d80 = x[b].d, d81 = nv == 2 ? x[xs + b].d : 0.0f;
            __m128 d8 = _mm_setr_ps(d80, d81, 0.0f, 0.0f);
            acc4 = _mm_fmadd_ps(_mm_mul_ps(_mm_set1_ps(dw), d8),
                                _mm_cvtepi32_ps(t), acc4);
        } else {
            __m256i s01 = _mm256_hadd_epi32(si[0], si[1]);
            __m256i s23 = _mm256_hadd_epi32(si[2], si[3]);
            __m256i s45 = _mm256_hadd_epi32(si[4], si[5]);
            __m256i s67 = _mm256_hadd_epi32(si[6], si[7]);
            __m256i t0 = _mm256_hadd_epi32(s01, s23);
            __m256i t1 = _mm256_hadd_epi32(s45, s67);
            __m256i is8 =
                _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20),
                                 _mm256_permute2x128_si256(t0, t1, 0x31));
#define D8(v) (v < nv ? x[(size_t)(v) * xs + b].d : 0.0f)
            __m256 d8 = _mm256_setr_ps(D8(0), D8(1), D8(2), D8(3), D8(4), D8(5),
                                       D8(6), D8(7));
#undef D8
            acc8 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dw), d8),
                                   _mm256_cvtepi32_ps(is8), acc8);
        }
    }
    float r[8];
    if (nv <= 2)
        _mm_storeu_ps(r, acc4);
    else
        _mm256_storeu_ps(r, acc8);
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = r[v];
}

#define Q6KP_LEVEL(L)                                                          \
    switch (nv) {                                                              \
    case 8:                                                                    \
        DOTN(q6kp_dotn_body)(w, p1, p0, x, xs, nb, out, 8, L);                 \
        break;                                                                 \
    case 7:                                                                    \
        DOTN(q6kp_dotn_body)(w, p1, p0, x, xs, nb, out, 7, L);                 \
        break;                                                                 \
    case 6:                                                                    \
        DOTN(q6kp_dotn_body)(w, p1, p0, x, xs, nb, out, 6, L);                 \
        break;                                                                 \
    case 5:                                                                    \
        DOTN(q6kp_dotn_body)(w, p1, p0, x, xs, nb, out, 5, L);                 \
        break;                                                                 \
    case 4:                                                                    \
        DOTN(q6kp_dotn_body)(w, p1, p0, x, xs, nb, out, 4, L);                 \
        break;                                                                 \
    case 3:                                                                    \
        DOTN(q6kp_dotn_body)(w, p1, p0, x, xs, nb, out, 3, L);                 \
        break;                                                                 \
    case 2:                                                                    \
        DOTN(q6kp_dotn_body)(w, p1, p0, x, xs, nb, out, 2, L);                 \
        break;                                                                 \
    default:                                                                   \
        DOTN(q6kp_dotn_body)(w, p1, p0, x, xs, nb, out, 1, L);                 \
    }

static DOTN_TARGET void DOTN(q6kp_dotn)(const struct janas_block_q6kp *w,
                                        const uint8_t *p1, const uint8_t *p0,
                                        const struct janas_block_q8k *x,
                                        size_t xs, int nv, size_t nb,
                                        float *out)
{
    if (p0)
        Q6KP_LEVEL(3)
    else if (p1)
        Q6KP_LEVEL(2)
    else
        Q6KP_LEVEL(1)
}
#undef Q6KP_LEVEL

/*
 * Q6_K and Q6_K_P as products of tiles (see q4k_gemm): the eight rows'
 * six-bit values unpacked to bytes and transposed four at a time
 * (q6kg_fill), so that a register holds four weights of each row. A
 * scale covers 16 weights, four dwords: with six bits two maddubs fit in
 * int16 (2 x 2 x 63 x 128 = 32256), so one madd by the scales every two
 * (AVX2), or four vpdpbusd and one multiply (AVX-VNNI). The offset of the
 * weights (-32, or the levels' centres) and the float operations are the
 * products' above, so the same bits: Q6_K's isum = sumi - 32 sum(scale x
 * group sums), dw = d; Q6_K_P's 2 isum = (sumi << shk) - c sum(...), dw =
 * d / 2.
 */
#ifndef Q6KG_DEFINED
#define Q6KG_DEFINED
/* AVX-VNNI: a group by four vpdpbusd and a multiply by the scales (1),
   or by maddubs summed in int16 and one vpdpwssd (0), fewer instructions
   on the multipliers' ports but slower (bench_kernels, 12 threads, 32
   vectors: Q6_K 12.1 against 10.3 GB/s, Q3_K 5.3 against 4.5; 25 Oct
   2026) */
#ifndef Q6KG_DP
#define Q6KG_DP(L) 1
#endif
#define Q6KG_CHUNK 8 /* blocks transposed at a time: 26 KB */
struct q6kg_blk {
    uint32_t qs[64][8];   /* [dword][row] */
    int16_t sc16[16][16]; /* [group][2 row + i]: its scale, twice */
    int16_t scp[8][16];   /* [k][2 row + i]: scales 2k and 2k + 1 */
    int32_t sc32[16][8];  /* [group][row] */
    float d[8], dmin[8];  /* dmin: Q2_K's */
};
#endif

/* The 32 bytes of q (a quarter of a block of eight rows, j's in r[j])
   transposed into qs[8 q + 0..7] */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(gemm_t8)(__m256i r[8], uint32_t (*qs)[8])
{
    __m256i t[8], u[8];
    _Pragma("GCC unroll 4") for (int j = 0; j < 8; j += 2)
    {
        t[j] = _mm256_unpacklo_epi32(r[j], r[j + 1]);
        t[j + 1] = _mm256_unpackhi_epi32(r[j], r[j + 1]);
    }
    _Pragma("GCC unroll 2") for (int j = 0; j < 8; j += 4)
    {
        u[j] = _mm256_unpacklo_epi64(t[j], t[j + 2]);
        u[j + 1] = _mm256_unpackhi_epi64(t[j], t[j + 2]);
        u[j + 2] = _mm256_unpacklo_epi64(t[j + 1], t[j + 3]);
        u[j + 3] = _mm256_unpackhi_epi64(t[j + 1], t[j + 3]);
    }
    _Pragma("GCC unroll 4") for (int q = 0; q < 4; q++)
    {
        _mm256_storeu_si256((__m256i *)qs[q],
                            _mm256_permute2x128_si256(u[q], u[q + 4], 0x20));
        _mm256_storeu_si256((__m256i *)qs[q + 4],
                            _mm256_permute2x128_si256(u[q], u[q + 4], 0x31));
    }
}

/* an 8 x 8 transpose of the rows' pairs of int8: t[kk], word r is row r's
   bytes 2kk and 2kk + 1 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q6kg_pairs8)(const __m128i *s, __m128i *t)
{
    __m128i b[8], c[8];
    for (int r = 0; r < 8; r += 2) {
        b[r] = _mm_unpacklo_epi16(s[r], s[r + 1]);
        b[r + 1] = _mm_unpackhi_epi16(s[r], s[r + 1]);
    }
    for (int h = 0; h < 8; h += 4) {
        c[h] = _mm_unpacklo_epi32(b[h], b[h + 2]);
        c[h + 1] = _mm_unpackhi_epi32(b[h], b[h + 2]);
        c[h + 2] = _mm_unpacklo_epi32(b[h + 1], b[h + 3]);
        c[h + 3] = _mm_unpackhi_epi32(b[h + 1], b[h + 3]);
    }
    for (int q = 0; q < 4; q++) {
        t[2 * q] = _mm_unpacklo_epi64(c[q], c[q + 4]);
        t[2 * q + 1] = _mm_unpackhi_epi64(c[q], c[q + 4]);
    }
}

/* The sixteen int8 scales of each of the eight rows (s[r]) into k: their
   pairs give scp (or, m not NULL, the pairs of m, Q2_K's mins), and the
   pairs' bytes, sign extended, the tiles' sc32 (AVX-VNNI) or sc16 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q6kg_scales8)(struct q6kg_blk *k, const __m128i *s, const __m128i *m)
{
    __m128i t[8], tm[8];
    DOTN(q6kg_pairs8)(s, t);
    if (m)
        DOTN(q6kg_pairs8)(m, tm);
    const __m128i split =
        _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    for (int kk = 0; kk < 8; kk++) {
        _mm256_storeu_si256((__m256i *)k->scp[kk],
                            _mm256_cvtepi8_epi16(m ? tm[kk] : t[kk]));
        __m128i e = _mm_shuffle_epi8(t[kk], split); /* group 2kk, 2kk + 1 */
#ifdef DOTN_HAS_VNNI
        if (Q6KG_DP(0)) {
            _mm256_storeu_si256((__m256i *)k->sc32[2 * kk],
                                _mm256_cvtepi8_epi32(e));
            _mm256_storeu_si256((__m256i *)k->sc32[2 * kk + 1],
                                _mm256_cvtepi8_epi32(_mm_srli_si128(e, 8)));
            continue;
        }
#endif
        for (int h = 0; h < 2; h++) {
            __m128i v = _mm_cvtepi8_epi16(h ? _mm_srli_si128(e, 8) : e);
            _mm256_storeu_si256((__m256i *)k->sc16[2 * kk + h],
                                _mm256_set_m128i(_mm_unpackhi_epi16(v, v),
                                                 _mm_unpacklo_epi16(v, v)));
        }
    }
}

/* Q3_K's sixteen scales less 32 (defined below) */
static inline DOTN_TARGET
    __attribute__((always_inline)) __m128i DOTN(q3k_scales16)(const uint8_t *q);

/* Q6_K blocks [b0, b0 + n) of rows w[0..7] into bk */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q6kg_fill_q6k)(const struct janas_block_q6k *const *w, size_t b0, int n,
                    struct q6kg_blk *bk)
{
    const __m256i low4 = _mm256_set1_epi8(0x0f), low2 = _mm256_set1_epi8(0x03);
    for (int i = 0; i < n; i++) {
        struct q6kg_blk *k = bk + i;
        for (int hn = 0; hn < 2; hn++) {
            __m256i q[4][8];
            _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
            {
                const struct janas_block_q6k *bl = &w[j][b0 + i];
                __m256i ql0 =
                    _mm256_loadu_si256((const __m256i *)(bl->ql + 64 * hn));
                __m256i ql1 = _mm256_loadu_si256(
                    (const __m256i *)(bl->ql + 64 * hn + 32));
                __m256i qh =
                    _mm256_loadu_si256((const __m256i *)(bl->qh + 32 * hn));
                q[0][j] = _mm256_or_si256(
                    _mm256_and_si256(ql0, low4),
                    _mm256_slli_epi16(_mm256_and_si256(qh, low2), 4));
                q[1][j] = _mm256_or_si256(
                    _mm256_and_si256(ql1, low4),
                    _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(qh, 2), low2), 4));
                q[2][j] = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql0, 4), low4),
                    _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(qh, 4), low2), 4));
                q[3][j] = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql1, 4), low4),
                    _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(qh, 6), low2), 4));
            }
            for (int part = 0; part < 4; part++)
                DOTN(gemm_t8)(q[part], k->qs + 8 * (4 * hn + part));
        }
        __m128i sc[8];
        _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
        {
            sc[j] = _mm_loadu_si128((const __m128i *)w[j][b0 + i].scales);
            k->d[j] = _cvtsh_ss(w[j][b0 + i].d);
        }
        DOTN(q6kg_scales8)(k, sc, NULL);
    }
}

/* Q6_K_P blocks [b0, b0 + n) of rows w[0..7] (planes p1[j], p0[j]) at
   level L into bk */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q6kg_fill_q6kp)(const struct janas_block_q6kp *const *w,
                     const uint8_t *const *p1, const uint8_t *const *p0,
                     size_t b0, int n, struct q6kg_blk *bk, const int L)
{
    const __m256i low2 = _mm256_set1_epi8(0x03);
    for (int i = 0; i < n; i++) {
        struct q6kg_blk *k = bk + i;
        size_t b = b0 + (size_t)i;
        for (int hn = 0; hn < 2; hn++) {
            __m256i q[4][8];
            _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
            {
                __m256i h =
                    _mm256_loadu_si256((const __m256i *)(w[j][b].q + 32 * hn));
                __m256i m1 =
                    L >= 2 ? _mm256_loadu_si256((
                                 const __m256i *)(p1[j] + b * JANAS_Q6KP_PLANE +
                                                  32 * hn))
                           : _mm256_setzero_si256();
                __m256i m0 =
                    L >= 3 ? _mm256_loadu_si256((
                                 const __m256i *)(p0[j] + b * JANAS_Q6KP_PLANE +
                                                  32 * hn))
                           : _mm256_setzero_si256();
                for (int part = 0; part < 4; part++) {
#define Q6KG_BITS(reg) _mm256_and_si256(_mm256_srli_epi16(reg, 2 * part), low2)
                    q[part][j] =
                        L == 3 ? _mm256_or_si256(
                                     _mm256_or_si256(
                                         _mm256_slli_epi16(Q6KG_BITS(h), 4),
                                         _mm256_slli_epi16(Q6KG_BITS(m1), 2)),
                                     Q6KG_BITS(m0))
                        : L == 2 ? _mm256_or_si256(
                                       _mm256_slli_epi16(Q6KG_BITS(h), 2),
                                       Q6KG_BITS(m1))
                                 : Q6KG_BITS(h);
#undef Q6KG_BITS
                }
            }
            for (int part = 0; part < 4; part++)
                DOTN(gemm_t8)(q[part], k->qs + 8 * (4 * hn + part));
        }
        __m128i sc[8];
        _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
        {
            sc[j] = _mm_loadu_si128((const __m128i *)w[j][b].scales);
            k->d[j] = _cvtsh_ss(w[j][b].d) * 0.5f;
        }
        DOTN(q6kg_scales8)(k, sc, NULL);
    }
}

/* Q3_K blocks [b0, b0 + n) of rows w[0..7] into bk: the values 0 to 7
   (the weight plus 4), the sixteen scales less 32 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q6kg_fill_q3k)(const struct janas_block_q3k *const *w, size_t b0, int n,
                    struct q6kg_blk *bk)
{
    const __m256i three = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    for (int i = 0; i < n; i++) {
        struct q6kg_blk *k = bk + i;
        size_t b = b0 + (size_t)i;
        for (int kk = 0; kk < 8; kk++) {
            const int hn = kk / 4, j = kk % 4;
            __m256i u[8];
            _Pragma("GCC unroll 8") for (int r = 0; r < 8; r++)
            {
                __m256i qs =
                    _mm256_loadu_si256((const __m256i *)(w[r][b].qs + 32 * hn));
                __m256i hm = _mm256_loadu_si256((const __m256i *)w[r][b].hmask);
                u[r] = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(qs, 2 * j), three),
                    _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(hm, j + 4 * hn),
                                         one),
                        2));
            }
            DOTN(gemm_t8)(u, k->qs + 8 * kk);
        }
        __m128i sc[8];
        _Pragma("GCC unroll 8") for (int r = 0; r < 8; r++)
        {
            sc[r] = DOTN(q3k_scales16)(w[r][b].scales);
            k->d[r] = _cvtsh_ss(w[r][b].d);
        }
        DOTN(q6kg_scales8)(k, sc, NULL);
    }
}

/* T tokens on n transposed blocks from b0, the sums added into af (8
   floats a token); L 0 for Q6_K, else Q6_K_P's level */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q6kg_tile)(const struct q6kg_blk *bk, int n, size_t b0,
                const struct janas_block_q8k *x, size_t xs, float *af,
                float *am, const int T, const int L)
{
    /* L 4: Q3_K, the weights' -4 and d whole; L 5: the IQ types' signed
       values, plus 128 with AVX-VNNI (sign and abs with AVX2, no offset);
       L 6: Q2_K's values 0 to 3, scp its mins, their sums into am */
    const int shk = L == 0 || L >= 4 ? 0 : L == 3 ? 1 : L == 2 ? 3 : 5;
    const __m256i cv = _mm256_set1_epi32(L == 0   ? 32
                                         : L == 4 ? 4
                                         : L == 5 ? 128
                                         : L == 3 ? 64
                                         : L == 2 ? 61
                                                  : 49);
#ifdef DOTN_HAS_VNNI
    const int off = L != 6;
#else
    const int off = L != 5 && L != 6;
#endif
    for (int i = 0; i < n; i++) {
        const struct q6kg_blk *k = bk + i;
        size_t b = b0 + (size_t)i;
        __m256i is[8];
        _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
            _mm256_setzero_si256();
        for (int g = 0; g < 16; g++) {
#ifdef DOTN_HAS_VNNI
            if (Q6KG_DP(L) || L == 5) {
                __m256i a32[8];
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) a32[t] =
                    _mm256_setzero_si256();
                _Pragma("GCC unroll 4") for (int q = 0; q < 4; q++)
                {
                    __m256i wv =
                        _mm256_loadu_si256((const __m256i *)k->qs[4 * g + q]);
                    _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                    {
                        int32_t aw;
                        memcpy(&aw, x[(size_t)t * xs + b].qs + 16 * g + 4 * q,
                               4);
                        a32[t] = DOTN_DPBUSD(a32[t], wv, _mm256_set1_epi32(aw));
                    }
                }
                __m256i sc = _mm256_loadu_si256((const __m256i *)k->sc32[g]);
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
                    _mm256_add_epi32(is[t], _mm256_mullo_epi32(a32[t], sc));
                continue;
            }
#endif
            __m256i sc = _mm256_loadu_si256((const __m256i *)k->sc16[g]);
            /* maddubs summed in int16 before a madd: two of six bits (or
               the IQ types' values, up to 62), four of three (4 x 2 x 7 x
               128) or two */
            const int G = L == 4 || L == 6 ? 4 : 2;
            _Pragma("GCC unroll 2") for (int hq = 0; hq < 4 / G; hq++)
            {
                __m256i a16[8];
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) a16[t] =
                    _mm256_setzero_si256();
                _Pragma("GCC unroll 4") for (int q = G * hq; q < G * hq + G;
                                             q++)
                {
                    __m256i wv =
                        _mm256_loadu_si256((const __m256i *)k->qs[4 * g + q]);
                    __m256i wa = L == 5 ? _mm256_abs_epi8(wv) : wv;
                    _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                    {
                        int32_t aw;
                        memcpy(&aw, x[(size_t)t * xs + b].qs + 16 * g + 4 * q,
                               4);
                        __m256i av = _mm256_set1_epi32(aw);
                        if (L == 5)
                            av = _mm256_sign_epi8(av, wv);
                        a16[t] = _mm256_add_epi16(a16[t],
                                                  _mm256_maddubs_epi16(wa, av));
                    }
                }
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
                    DOTN_ACC(is[t], a16[t], sc);
            }
        }
        __m256 dw = _mm256_loadu_ps(k->d);
        _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
        {
            const struct janas_block_q8k *xb = x + (size_t)t * xs + b;
            __m256i mn = _mm256_setzero_si256();
            if (off || L == 6)
                _Pragma("GCC unroll 8") for (int kk = 0; kk < 8; kk++)
                {
                    int32_t bw;
                    memcpy(&bw, xb->bsums + 2 * kk, 4);
                    mn = _mm256_add_epi32(
                        mn, _mm256_madd_epi16(
                                _mm256_loadu_si256((const __m256i *)k->scp[kk]),
                                _mm256_set1_epi32(bw)));
                }
            __m256i si = off ? _mm256_sub_epi32(_mm256_slli_epi32(is[t], shk),
                                                _mm256_mullo_epi32(mn, cv))
                             : is[t];
            __m256 d8 = _mm256_set1_ps(xb->d);
            _mm256_storeu_ps(af + 8 * t,
                             _mm256_fmadd_ps(_mm256_mul_ps(dw, d8),
                                             _mm256_cvtepi32_ps(si),
                                             _mm256_loadu_ps(af + 8 * t)));
            if (L == 6)
                _mm256_storeu_ps(
                    am + 8 * t,
                    _mm256_fmadd_ps(_mm256_mul_ps(_mm256_loadu_ps(k->dmin), d8),
                                    _mm256_cvtepi32_ps(mn),
                                    _mm256_loadu_ps(am + 8 * t)));
        }
    }
}

#ifndef Q6KG_T
#define Q6KG_T 6
#endif

/* every token of af's tiles on bk (see q4k_gemm) */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q6kg_tiles)(const struct q6kg_blk *bk, int n, size_t b0,
                 const struct janas_block_q8k *x, size_t xs, size_t nt,
                 float *af, float *am, const int L)
{
    size_t t0 = 0;
#define Q6KG_TILE(T)                                                           \
    DOTN(q6kg_tile)(bk, n, b0, x + t0 * xs, xs, af + 8 * t0, am + 8 * t0, T, L)
    for (; t0 + Q6KG_T <= nt; t0 += Q6KG_T)
        Q6KG_TILE(Q6KG_T);
    for (; t0 + 4 <= nt; t0 += 4)
        Q6KG_TILE(4);
    if (nt - t0 == 3)
        Q6KG_TILE(3);
    else if (nt - t0 == 2)
        Q6KG_TILE(2);
    else if (nt - t0 == 1)
        Q6KG_TILE(1);
#undef Q6KG_TILE
}

/* rows (a multiple of 8) of Q6_K (p1 NULL and pb 0) or Q6_K_P (rows' planes
   pb bytes apart from p1 and p0, NULL past its level) against nv vectors,
   y[r + v ys] */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q6k_gemm_body)(const void *w, size_t row_bytes, const uint8_t *p1,
                    const uint8_t *p0, size_t pb,
                    const struct janas_block_q8k *x, size_t xs, size_t nv,
                    size_t nb, size_t rows, float *y, size_t ys, const int L)
{
    struct q6kg_blk bk[Q6KG_CHUNK];
    float af[Q4KG_TOK * 8];
    for (size_t g = 0; g < rows; g += 8) {
        const void *wr[8];
        const uint8_t *r1[8], *r0[8];
        for (int j = 0; j < 8; j++) {
            wr[j] = (const char *)w + (g + j) * row_bytes;
            r1[j] = p1 ? p1 + (g + j) * pb : NULL;
            r0[j] = p0 ? p0 + (g + j) * pb : NULL;
        }
        for (size_t v0 = 0; v0 < nv; v0 += Q4KG_TOK) {
            size_t nt = nv - v0 < Q4KG_TOK ? nv - v0 : Q4KG_TOK;
            memset(af, 0, nt * 8 * sizeof(float));
            for (size_t b0 = 0; b0 < nb; b0 += Q6KG_CHUNK) {
                int n = nb - b0 < Q6KG_CHUNK ? (int)(nb - b0) : Q6KG_CHUNK;
                if (L == 0)
                    DOTN(q6kg_fill_q6k)(
                        (const struct janas_block_q6k *const *)wr, b0, n, bk);
                else if (L == 4)
                    DOTN(q6kg_fill_q3k)(
                        (const struct janas_block_q3k *const *)wr, b0, n, bk);
                else
                    DOTN(q6kg_fill_q6kp)(
                        (const struct janas_block_q6kp *const *)wr, r1, r0, b0,
                        n, bk, L);
                DOTN(gemm_ahead)(w, row_bytes,
                                 L == 0   ? sizeof(struct janas_block_q6k)
                                 : L == 4 ? sizeof(struct janas_block_q3k)
                                          : row_bytes / nb,
                                 g, rows, nb, b0, Q6KG_CHUNK);
                DOTN(q6kg_tiles)(bk, n, b0, x + v0 * xs, xs, nt, af, af, L);
            }
            for (size_t t = 0; t < nt; t++)
                for (int j = 0; j < 8; j++)
                    y[g + j + (v0 + t) * ys] = af[8 * t + j];
        }
    }
}

static DOTN_TARGET void
DOTN(q6k_gemm)(const void *w, size_t row_bytes, const uint8_t *p1,
               const uint8_t *p0, size_t pb, const struct janas_block_q8k *x,
               size_t xs, size_t nv, size_t nb, size_t rows, float *y,
               size_t ys, int level)
{
    switch (level) {
    case 0:
        DOTN(q6k_gemm_body)(w, row_bytes, p1, p0, pb, x, xs, nv, nb, rows, y,
                            ys, 0);
        break;
    case 1:
        DOTN(q6k_gemm_body)(w, row_bytes, p1, p0, pb, x, xs, nv, nb, rows, y,
                            ys, 1);
        break;
    case 2:
        DOTN(q6k_gemm_body)(w, row_bytes, p1, p0, pb, x, xs, nv, nb, rows, y,
                            ys, 2);
        break;
    case 4:
        DOTN(q6k_gemm_body)(w, row_bytes, p1, p0, pb, x, xs, nv, nb, rows, y,
                            ys, 4);
        break;
    default:
        DOTN(q6k_gemm_body)(w, row_bytes, p1, p0, pb, x, xs, nv, nb, rows, y,
                            ys, 3);
    }
}

/*
 * Q2_K and the IQ types with grids (IQ3_S, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S,
 * IQ1_S, IQ1_M) as Q6_K's tiles: a group of 16 weights a scale. Q2_K's
 * values 0 to 3 (level 6), its mins' sums apart as its products do; the IQ
 * types' weights as signed bytes, level times sign (level 5), their scales
 * 1 + 2 s, a 32-weight group's in both its groups of 16, and the block's d
 * with its factor (1/4, 1/8) as their products take it. The integer sums
 * are exact, one fma a block (two for Q2_K): the same bits.
 */
static inline DOTN_TARGET __attribute__((always_inline)) size_t
DOTN(iqkg_size)(const int TY)
{
    return TY == JANAS_IQ3_S     ? sizeof(struct janas_block_iq3s)
           : TY == JANAS_IQ3_XXS ? sizeof(struct janas_block_iq3xxs)
           : TY == JANAS_IQ1_S   ? sizeof(struct janas_block_iq1s)
           : TY == JANAS_IQ1_M   ? sizeof(struct janas_block_iq1m)
           : TY == JANAS_Q2_K    ? sizeof(struct janas_block_q2k)
                                 : iq2_size(TY);
}

/* weights 32 g to 32 g + 31 of an IQ block as signed bytes, the scales of
   its two groups of 16 into s[2 g] and s[2 g + 1] */
static inline DOTN_TARGET __attribute__((always_inline)) __m256i
DOTN(iqkg_group)(const int TY, const uint8_t *blk, int g, int8_t *s)
{
    if (TY == JANAS_IQ3_S) {
        const struct janas_block_iq3s *w = (const void *)blk;
        const __m256i shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i sel =
            _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                             2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        const __m256i bit =
            _mm256_set1_epi64x((long long)0x8040201008040201ull);
        __m128i q8 = _mm_loadl_epi64((const __m128i *)(w->qs + 8 * g));
        __m256i nine = _mm256_slli_epi32(
            _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32(w->qh[g]), shifts),
                _mm256_set1_epi32(1)),
            8);
        __m256i lv = _mm256_i32gather_epi32(
            (const int *)iq3s_grid,
            _mm256_or_si256(_mm256_cvtepu8_epi32(q8), nine), 4);
        uint32_t sb;
        memcpy(&sb, w->signs + 4 * g, 4);
        __m256i m = _mm256_shuffle_epi8(_mm256_set1_epi32((int)sb), sel);
        m = _mm256_cmpeq_epi8(_mm256_and_si256(m, bit), bit);
        s[2 * g] = s[2 * g + 1] = (int8_t)iq3s_scale(w, g);
        return _mm256_sign_epi8(lv, _mm256_or_si256(m, _mm256_set1_epi8(1)));
    }
    if (TY == JANAS_IQ3_XXS) {
        const struct janas_block_iq3xxs *w = (const void *)blk;
        __m128i q8 = _mm_loadl_epi64((const __m128i *)(w->qs + 8 * g));
        __m256i lv = _mm256_i32gather_epi32((const int *)iq3xxs_grid,
                                            _mm256_cvtepu8_epi32(q8), 4);
        uint32_t aux = iq3xxs_aux(w, g);
        s[2 * g] = s[2 * g + 1] = (int8_t)(1 + 2 * (aux >> 28));
        return _mm256_sign_epi8(
            lv, _mm256_setr_epi64x((long long)iq_sign64[aux & 127],
                                   (long long)iq_sign64[aux >> 7 & 127],
                                   (long long)iq_sign64[aux >> 14 & 127],
                                   (long long)iq_sign64[aux >> 21 & 127]));
    }
    uint64_t gr[4], sm[4];
    int sl[4];
    if (TY == JANAS_IQ1_S || TY == JANAS_IQ1_M) {
        _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++)
        {
            int neg;
            if (TY == JANAS_IQ1_S)
                iq1s_sub((const void *)blk, 4 * g + k, &gr[k], &neg, &sl[k]);
            else
                iq1m_sub((const void *)blk, 4 * g + k, &gr[k], &neg, &sl[k]);
            sm[k] = neg ? 0xffffffffffffffffull : 0x0101010101010101ull;
        }
        __m256i lev = _mm256_setr_epi64x((long long)gr[0], (long long)gr[1],
                                         (long long)gr[2], (long long)gr[3]);
        s[2 * g] = (int8_t)sl[0];
        s[2 * g + 1] = (int8_t)sl[2];
        return _mm256_add_epi8(
            _mm256_and_si256(_mm256_slli_epi16(lev, 3),
                             _mm256_set1_epi8((char)0xf8)),
            _mm256_setr_epi64x((long long)sm[0], (long long)sm[1],
                               (long long)sm[2], (long long)sm[3]));
    }
    if (TY == JANAS_IQ2_S) {
        _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++)
        {
            uint32_t s8;
            iq2_sub(TY, blk, 4 * g + k, &gr[k], &s8, &sl[k]);
            sm[k] = iq_byte64[s8];
        }
    } else {
        _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++)
            iq2_sub_mask(TY, blk, 4 * g + k, &gr[k], &sm[k], &sl[k]);
    }
    s[2 * g] = (int8_t)sl[0];
    s[2 * g + 1] = (int8_t)sl[2];
    return _mm256_sign_epi8(
        _mm256_setr_epi64x((long long)gr[0], (long long)gr[1], (long long)gr[2],
                           (long long)gr[3]),
        _mm256_setr_epi64x((long long)sm[0], (long long)sm[1], (long long)sm[2],
                           (long long)sm[3]));
}

/* an IQ block's d as its products take it */
static inline DOTN_TARGET __attribute__((always_inline)) float
DOTN(iqkg_d)(const int TY, const uint8_t *blk)
{
    uint16_t h;
    if (TY == JANAS_IQ1_M)
        h = iq1m_d16((const void *)blk);
    else
        memcpy(&h, blk, 2);
    float d = _cvtsh_ss(h);
    return TY == JANAS_IQ3_S ? d : TY == JANAS_IQ3_XXS ? d * 0.25f : d * 0.125f;
}

/* blocks [b0, b0 + n) of rows w[0..7] of an IQ type or Q2_K into bk */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(iqkg_fill)(const int TY, const uint8_t *const *w, size_t b0, int n,
                struct q6kg_blk *bk)
{
    const size_t size = DOTN(iqkg_size)(TY);
    for (int i = 0; i < n; i++) {
        struct q6kg_blk *k = bk + i;
        size_t b = b0 + (size_t)i;
        if (TY == JANAS_Q2_K) {
            const __m256i three = _mm256_set1_epi8(3);
            for (int kk = 0; kk < 8; kk++) {
                __m256i u[8];
                _Pragma("GCC unroll 8") for (int r = 0; r < 8; r++)
                {
                    const struct janas_block_q2k *bl =
                        (const struct janas_block_q2k *)(w[r] + b * size);
                    __m256i q = _mm256_loadu_si256(
                        (const __m256i *)(bl->qs + 32 * (kk / 4)));
                    u[r] = _mm256_and_si256(_mm256_srli_epi16(q, 2 * (kk % 4)),
                                            three);
                }
                DOTN(gemm_t8)(u, k->qs + 8 * kk);
            }
            __m128i sc[8], mn[8];
            _Pragma("GCC unroll 8") for (int r = 0; r < 8; r++)
            {
                const struct janas_block_q2k *bl =
                    (const struct janas_block_q2k *)(w[r] + b * size);
                __m128i m8 = _mm_loadu_si128((const __m128i *)bl->scales);
                sc[r] = _mm_and_si128(m8, _mm_set1_epi8(15));
                mn[r] = _mm_and_si128(_mm_srli_epi16(m8, 4), _mm_set1_epi8(15));
                k->d[r] = _cvtsh_ss(bl->d);
                k->dmin[r] = _cvtsh_ss(bl->dmin);
            }
            DOTN(q6kg_scales8)(k, sc, mn);
            continue;
        }
        int8_t s[8][16];
        for (int kk = 0; kk < 8; kk++) {
            __m256i u[8];
            _Pragma("GCC unroll 8") for (int r = 0; r < 8; r++)
            {
                u[r] = DOTN(iqkg_group)(TY, w[r] + b * size, kk, s[r]);
#ifdef DOTN_HAS_VNNI
                u[r] = _mm256_xor_si256(u[r], _mm256_set1_epi8((char)0x80));
#endif
            }
            DOTN(gemm_t8)(u, k->qs + 8 * kk);
        }
        __m128i sc[8];
        _Pragma("GCC unroll 8") for (int r = 0; r < 8; r++)
        {
            sc[r] = _mm_loadu_si128((const __m128i *)s[r]);
            k->d[r] = DOTN(iqkg_d)(TY, w[r] + b * size);
        }
        DOTN(q6kg_scales8)(k, sc, NULL);
    }
}

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(iqk_gemm_t)(const int TY, const void *w, size_t row_bytes,
                 const struct janas_block_q8k *x, size_t xs, size_t nv,
                 size_t nb, size_t rows, float *y, size_t ys)
{
    const int L = TY == JANAS_Q2_K ? 6 : 5;
    struct q6kg_blk bk[Q6KG_CHUNK];
    float af[Q4KG_TOK * 8], am[Q4KG_TOK * 8];
    for (size_t g = 0; g < rows; g += 8) {
        const uint8_t *wr[8];
        for (int j = 0; j < 8; j++)
            wr[j] = (const uint8_t *)w + (g + j) * row_bytes;
        for (size_t v0 = 0; v0 < nv; v0 += Q4KG_TOK) {
            size_t nt = nv - v0 < Q4KG_TOK ? nv - v0 : Q4KG_TOK;
            memset(af, 0, nt * 8 * sizeof(float));
            if (L == 6)
                memset(am, 0, nt * 8 * sizeof(float));
            for (size_t b0 = 0; b0 < nb; b0 += Q6KG_CHUNK) {
                int n = nb - b0 < Q6KG_CHUNK ? (int)(nb - b0) : Q6KG_CHUNK;
                DOTN(iqkg_fill)(TY, wr, b0, n, bk);
                DOTN(gemm_ahead)(w, row_bytes, DOTN(iqkg_size)(TY), g, rows, nb,
                                 b0, Q6KG_CHUNK);
                DOTN(q6kg_tiles)(bk, n, b0, x + v0 * xs, xs, nt, af, am, L);
            }
            for (size_t t = 0; t < nt; t++)
                for (int j = 0; j < 8; j++)
                    y[g + j + (v0 + t) * ys] =
                        L == 6 ? af[8 * t + j] - am[8 * t + j] : af[8 * t + j];
        }
    }
}

/* rows (a multiple of 8) of Q2_K or an IQ type with a grid against nv
   vectors, y[r + v ys]: 1, or 0 for another type */
static DOTN_TARGET int DOTN(iqk_gemm)(int type, const void *w, size_t row_bytes,
                                      const struct janas_block_q8k *x,
                                      size_t xs, size_t nv, size_t nb,
                                      size_t rows, float *y, size_t ys)
{
    switch (type) {
#define IQKG(T)                                                                \
    case T:                                                                    \
        DOTN(iqk_gemm_t)(T, w, row_bytes, x, xs, nv, nb, rows, y, ys);         \
        return 1;
        IQKG(JANAS_Q2_K)
        IQKG(JANAS_IQ3_S)
        IQKG(JANAS_IQ3_XXS)
        IQKG(JANAS_IQ2_XXS)
        IQKG(JANAS_IQ2_XS)
        IQKG(JANAS_IQ2_S)
        IQKG(JANAS_IQ1_S)
        IQKG(JANAS_IQ1_M)
#undef IQKG
    }
    return 0;
}

/*
 * Q5_K as tiles (see q4k_gemm and q6k_gemm): the eight rows' five-bit
 * values unpacked to bytes and transposed; a 32-weight piece is eight
 * dwords, of which four maddubs fit in int16 (4 x 2 x 31 x 128 = 31744):
 * two madds by the scales a piece (AVX2), or eight vpdpbusd and one
 * multiply (AVX-VNNI); the mins and the floats Q4_K's, the same bits.
 */
#ifndef Q5KG_DEFINED
#define Q5KG_DEFINED
#define Q5KG_CHUNK 8
struct q5kg_blk {
    uint32_t qs[64][8]; /* [dword][row] */
    int16_t sc[8][16];  /* [sub-block][2 row + i] */
    int16_t mn[4][16];  /* [k][2 row + i]: mins 2k and 2k + 1 */
    int32_t sc32[8][8]; /* [sub-block][row] */
    float d[8], dmin[8];
};
#endif

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q5kg_fill)(const struct janas_block_q5k *const *w, size_t b0, int n,
                struct q5kg_blk *bk)
{
    const __m256i low4 = _mm256_set1_epi8(0x0f), one = _mm256_set1_epi8(1);
    for (int i = 0; i < n; i++) {
        struct q5kg_blk *k = bk + i;
        size_t b = b0 + (size_t)i;
        for (int c = 0; c < 4; c++) {
            __m256i lo[8], hi[8];
            _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
            {
                __m256i q4 =
                    _mm256_loadu_si256((const __m256i *)(w[j][b].qs + 32 * c));
                __m256i qh = _mm256_loadu_si256((const __m256i *)w[j][b].qh);
                lo[j] = _mm256_or_si256(
                    _mm256_and_si256(q4, low4),
                    _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(qh, 2 * c), one),
                        4));
                hi[j] = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(q4, 4), low4),
                    _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(qh, 2 * c + 1), one),
                        4));
            }
            DOTN(gemm_t8)(lo, k->qs + 8 * (2 * c));
            DOTN(gemm_t8)(hi, k->qs + 8 * (2 * c + 1));
        }
        __m128i sv[8];
        _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
        {
            const struct janas_block_q5k *blk = &w[j][b];
            uint32_t u[4];
            unpack_scales_u32(blk->scales, u);
            sv[j] = _mm_loadu_si128((const __m128i *)u);
            k->d[j] = _cvtsh_ss(blk->d);
            k->dmin[j] = _cvtsh_ss(blk->dmin);
        }
        DOTN(kmin_scales8)(k->sc, k->sc32, k->mn, sv, 0);
    }
}

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q5kg_tile)(const struct q5kg_blk *bk, int n, size_t b0,
                const struct janas_block_q8k *x, size_t xs, float *af,
                float *am, const int T)
{
    for (int i = 0; i < n; i++) {
        const struct q5kg_blk *k = bk + i;
        size_t b = b0 + (size_t)i;
        __m256i is[8];
        _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
            _mm256_setzero_si256();
        for (int sb = 0; sb < 8; sb++) {
#ifdef DOTN_HAS_VNNI
            __m256i a32[8];
            _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) a32[t] =
                _mm256_setzero_si256();
            _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++)
            {
                __m256i wv =
                    _mm256_loadu_si256((const __m256i *)k->qs[8 * sb + q]);
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                {
                    int32_t aw;
                    memcpy(&aw, x[(size_t)t * xs + b].qs + 32 * sb + 4 * q, 4);
                    a32[t] = DOTN_DPBUSD(a32[t], wv, _mm256_set1_epi32(aw));
                }
            }
            __m256i sc = _mm256_loadu_si256((const __m256i *)k->sc32[sb]);
            _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
                _mm256_add_epi32(is[t], _mm256_mullo_epi32(a32[t], sc));
#else
            __m256i sc = _mm256_loadu_si256((const __m256i *)k->sc[sb]);
            _Pragma("GCC unroll 2") for (int hq = 0; hq < 2; hq++)
            {
                __m256i a16[8];
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) a16[t] =
                    _mm256_setzero_si256();
                _Pragma("GCC unroll 4") for (int q = 4 * hq; q < 4 * hq + 4;
                                             q++)
                {
                    __m256i wv =
                        _mm256_loadu_si256((const __m256i *)k->qs[8 * sb + q]);
                    _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                    {
                        int32_t aw;
                        memcpy(&aw, x[(size_t)t * xs + b].qs + 32 * sb + 4 * q,
                               4);
                        a16[t] = _mm256_add_epi16(
                            a16[t],
                            _mm256_maddubs_epi16(wv, _mm256_set1_epi32(aw)));
                    }
                }
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
                    DOTN_ACC(is[t], a16[t], sc);
            }
#endif
        }
        __m256 dw = _mm256_loadu_ps(k->d), dmw = _mm256_loadu_ps(k->dmin);
        _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
        {
            const struct janas_block_q8k *xb = x + (size_t)t * xs + b;
            __m256i im = _mm256_setzero_si256();
            _Pragma("GCC unroll 4") for (int kk = 0; kk < 4; kk++)
            {
                int32_t bw;
                memcpy(&bw, xb->bsums2 + 2 * kk, 4);
                im = _mm256_add_epi32(
                    im, _mm256_madd_epi16(
                            _mm256_loadu_si256((const __m256i *)k->mn[kk]),
                            _mm256_set1_epi32(bw)));
            }
            __m256 d8 = _mm256_set1_ps(xb->d);
            _mm256_storeu_ps(af + 8 * t,
                             _mm256_fmadd_ps(_mm256_mul_ps(dw, d8),
                                             _mm256_cvtepi32_ps(is[t]),
                                             _mm256_loadu_ps(af + 8 * t)));
            _mm256_storeu_ps(am + 8 * t,
                             _mm256_fmadd_ps(_mm256_mul_ps(dmw, d8),
                                             _mm256_cvtepi32_ps(im),
                                             _mm256_loadu_ps(am + 8 * t)));
        }
    }
}

/* rows (a multiple of 8) of Q5_K against nv vectors, y[r + v ys] */
static DOTN_TARGET void DOTN(q5k_gemm)(const void *w, size_t row_bytes,
                                       const struct janas_block_q8k *x,
                                       size_t xs, size_t nv, size_t nb,
                                       size_t rows, float *y, size_t ys)
{
    struct q5kg_blk bk[Q5KG_CHUNK];
    float af[Q4KG_TOK * 8], am[Q4KG_TOK * 8];
    for (size_t g = 0; g < rows; g += 8) {
        const struct janas_block_q5k *wr[8];
        for (int j = 0; j < 8; j++)
            wr[j] = (const void *)((const char *)w + (g + j) * row_bytes);
        for (size_t v0 = 0; v0 < nv; v0 += Q4KG_TOK) {
            size_t nt = nv - v0 < Q4KG_TOK ? nv - v0 : Q4KG_TOK;
            memset(af, 0, nt * 8 * sizeof(float));
            memset(am, 0, nt * 8 * sizeof(float));
            for (size_t b0 = 0; b0 < nb; b0 += Q5KG_CHUNK) {
                int n = nb - b0 < Q5KG_CHUNK ? (int)(nb - b0) : Q5KG_CHUNK;
                DOTN(q5kg_fill)(wr, b0, n, bk);
                DOTN(gemm_ahead)(w, row_bytes, sizeof(struct janas_block_q5k),
                                 g, rows, nb, b0, Q5KG_CHUNK);
                const struct janas_block_q8k *xv = x + v0 * xs;
                size_t t0 = 0;
                for (; t0 + Q4KG_T <= nt; t0 += Q4KG_T)
                    DOTN(q5kg_tile)(bk, n, b0, xv + t0 * xs, xs, af + 8 * t0,
                                    am + 8 * t0, Q4KG_T);
                for (; t0 + 4 <= nt; t0 += 4)
                    DOTN(q5kg_tile)(bk, n, b0, xv + t0 * xs, xs, af + 8 * t0,
                                    am + 8 * t0, 4);
                if (nt - t0 == 3)
                    DOTN(q5kg_tile)(bk, n, b0, xv + t0 * xs, xs, af + 8 * t0,
                                    am + 8 * t0, 3);
                else if (nt - t0 == 2)
                    DOTN(q5kg_tile)(bk, n, b0, xv + t0 * xs, xs, af + 8 * t0,
                                    am + 8 * t0, 2);
                else if (nt - t0 == 1)
                    DOTN(q5kg_tile)(bk, n, b0, xv + t0 * xs, xs, af + 8 * t0,
                                    am + 8 * t0, 1);
            }
            for (size_t t = 0; t < nt; t++)
                for (int j = 0; j < 8; j++)
                    y[g + j + (v0 + t) * ys] = af[8 * t + j] - am[8 * t + j];
        }
    }
}

/*
 * The types in blocks of 32 (Q8_0, IQ4_NL, Q4_0, MXFP4, Q5_0, Q4_1, Q5_1)
 * as tiles of eight rows (see q4k_gemm): each block's 32 values unpacked
 * to bytes, as w + o (o: Q4_0's 8 and Q5_0's 16 as stored, MXFP4's 12,
 * Q4_1's and Q5_1's 0, Q8_0's and IQ4_NL's 128 with AVX-VNNI, Q2_0's 1:
 * a block of 64 two blocks here, its d in both), transposed
 * four at a time, and o times the block's activation sum taken off the
 * exact integer sum. With AVX2 the small values' maddubs are summed in
 * int16 before one madd (Q4_0, Q4_1, Q2_0: eight, 8 x 2 x 15 x 128; Q5_0, Q5_1,
 * MXFP4: four), Q8_0's and IQ4_NL's signed bytes go through sign and abs as
 * in the products above. The floats are theirs: an accumulator for each
 * block position, fma(d x d32, sum), Q4_1's and Q5_1's mins apart, the
 * positions summed as janas_hsum8 does: the same bits.
 */
#ifndef B32G_DEFINED
#define B32G_DEFINED
#define B32G_CHUNK 8
struct b32g_blk {
    uint32_t qs[8][8][8];   /* [block][dword][row] */
    float d[8][8], m[8][8]; /* [block][row] */
};
#endif

static inline DOTN_TARGET __attribute__((always_inline)) void
    DOTN(q2_0_unpack)(const uint8_t *qs, __m256i *lo, __m256i *hi);

/* The offset added to a type's values (see above) */
static inline DOTN_TARGET __attribute__((always_inline)) int
DOTN(b32g_off)(const int TY)
{
#ifdef DOTN_HAS_VNNI
    const int sgn = 128;
#else
    const int sgn = 0;
#endif
    return TY == JANAS_Q4_0                       ? 8
           : TY == JANAS_Q2_0                     ? 1
           : TY == JANAS_Q5_0                     ? 16
           : TY == JANAS_MXFP4                    ? 12
           : TY == JANAS_Q4_1 || TY == JANAS_Q5_1 ? 0
                                                  : sgn;
}

static inline DOTN_TARGET __attribute__((always_inline)) size_t
DOTN(b32g_size)(const int TY)
{
    return TY == JANAS_Q8_0     ? sizeof(struct janas_block_q8_0)
           : TY == JANAS_MXFP4  ? sizeof(struct janas_block_mxfp4)
           : TY == JANAS_Q5_0   ? sizeof(struct janas_block_q5_0)
           : TY == JANAS_Q4_1   ? sizeof(struct janas_block_q4_1)
           : TY == JANAS_Q5_1   ? sizeof(struct janas_block_q5_1)
           : TY == JANAS_IQ4_NL ? sizeof(struct janas_block_iq4nl)
                                : sizeof(struct janas_block_q4_0);
}

/* A block's 32 values plus the type's offset, weights 0-31 in order */
static inline DOTN_TARGET __attribute__((always_inline)) __m256i
DOTN(b32g_unpack)(const int TY, const uint8_t *blk)
{
    const __m256i low4 = _mm256_set1_epi8(0x0f);
    const int o = DOTN(b32g_off)(TY);
    if (TY == JANAS_Q8_0)
        return _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(blk + 2)),
                               _mm256_set1_epi8((char)o));
    if (TY == JANAS_IQ4_NL || TY == JANAS_MXFP4) {
        const int8_t *t =
            TY == JANAS_MXFP4 ? janas_mxfp4_values : janas_iq4_values;
        __m256i lev =
            _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)t));
        __m128i b = _mm_loadu_si128(
            (const __m128i *)(blk + (TY == JANAS_MXFP4 ? 1 : 2)));
        __m256i q = _mm256_shuffle_epi8(
            lev, _mm256_set_m128i(
                     _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0f)),
                     _mm_and_si128(b, _mm_set1_epi8(0x0f))));
        return _mm256_add_epi8(q, _mm256_set1_epi8((char)o));
    }
    /* the nibbles, low ones weights 0-15 (Q4_0, Q5_0, Q4_1, Q5_1) */
    size_t at = TY == JANAS_Q4_0   ? 2
                : TY == JANAS_Q5_0 ? 6
                : TY == JANAS_Q4_1 ? 4
                                   : 8;
    __m128i b16 = _mm_loadu_si128((const __m128i *)(blk + at));
    __m256i q =
        _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(b16, 4), b16), low4);
    if (TY == JANAS_Q5_0 || TY == JANAS_Q5_1) {
        /* byte l takes byte l / 8 of qh, then keeps bit l % 8 of it */
        const __m256i sel =
            _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                             2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        const __m256i bit =
            _mm256_set1_epi64x((long long)0x8040201008040201ull);
        uint32_t qh;
        memcpy(&qh, blk + (TY == JANAS_Q5_0 ? 2 : 4), 4);
        __m256i h = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), sel);
        h = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(h, bit), bit),
                             _mm256_set1_epi8(0x10));
        q = _mm256_or_si256(q, h);
    }
    return q;
}

/* Groups [b0, b0 + n) of rows w[0..7] (each row's groups gs bytes apart:
   eight blocks of the type) into bk */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(b32g_fill)(const int TY, const uint8_t *const *w, size_t b0, int n,
                struct b32g_blk *bk)
{
    const size_t size = DOTN(b32g_size)(TY);
    const int dm = TY == JANAS_Q4_1 || TY == JANAS_Q5_1;
    for (int i = 0; i < n; i++) {
        struct b32g_blk *k = bk + i;
        if (TY == JANAS_Q2_0) {
            /* a block of 64: positions 2 kb2 and 2 kb2 + 1 */
            for (int kb2 = 0; kb2 < 4; kb2++) {
                __m256i lo[8], hi[8];
                _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
                {
                    const struct janas_block_q2_0 *blk =
                        (const struct janas_block_q2_0 *)w[j] + (b0 + i) * 4 +
                        kb2;
                    DOTN(q2_0_unpack)(blk->qs, &lo[j], &hi[j]);
                    k->d[2 * kb2][j] = k->d[2 * kb2 + 1][j] = _cvtsh_ss(blk->d);
                    /* no minimum, but b32g_tile loads the row of them */
                    k->m[2 * kb2][j] = k->m[2 * kb2 + 1][j] = 0.0f;
                }
                DOTN(gemm_t8)(lo, k->qs[2 * kb2]);
                DOTN(gemm_t8)(hi, k->qs[2 * kb2 + 1]);
            }
            continue;
        }
        for (int kb = 0; kb < 8; kb++) {
            __m256i r[8];
            _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
            {
                const uint8_t *blk = w[j] + ((b0 + i) * 8 + kb) * size;
                r[j] = DOTN(b32g_unpack)(TY, blk);
                if (TY == JANAS_MXFP4) {
                    k->d[kb][j] = e8m0_tab[blk[0]];
                    k->m[kb][j] = 0.0f;
                } else {
                    uint16_t d, m = 0;
                    memcpy(&d, blk, 2);
                    if (dm)
                        memcpy(&m, blk + 2, 2);
                    k->d[kb][j] = _cvtsh_ss(d);
                    k->m[kb][j] = dm ? _cvtsh_ss(m) : 0.0f;
                }
            }
            DOTN(gemm_t8)(r, k->qs[kb]);
        }
    }
}

/* T tokens on n groups from b0: the eight positions' sums added into af and
   (Q4_1, Q5_1) am, 64 floats a token: [position][row] */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(b32g_tile)(const int TY, const struct b32g_blk *bk, int n, size_t b0,
                const struct janas_block_q8k *x, size_t xs, float *af,
                float *am, const int T)
{
    const int o = DOTN(b32g_off)(TY);
    const int dm = TY == JANAS_Q4_1 || TY == JANAS_Q5_1;
#ifndef DOTN_HAS_VNNI
    const __m256i ones = _mm256_set1_epi16(1);
    /* signed bytes (Q8_0, IQ4_NL): sign and abs; else the maddubs summed
       in int16, G at a time */
    const int sg = TY == JANAS_Q8_0 || TY == JANAS_IQ4_NL;
    const int G =
        TY == JANAS_Q4_0 || TY == JANAS_Q4_1 || TY == JANAS_Q2_0 ? 8 : 4;
#endif
    for (int i = 0; i < n; i++) {
        const struct b32g_blk *k = bk + i;
        size_t b = b0 + (size_t)i;
        for (int kb = 0; kb < 8; kb++) {
            __m256i s[8];
            _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) s[t] =
                _mm256_setzero_si256();
#ifdef DOTN_HAS_VNNI
            _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++)
            {
                __m256i wv = _mm256_loadu_si256((const __m256i *)k->qs[kb][q]);
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                {
                    int32_t aw;
                    memcpy(&aw, x[(size_t)t * xs + b].qs32 + 32 * kb + 4 * q,
                           4);
                    s[t] = DOTN_DPBUSD(s[t], wv, _mm256_set1_epi32(aw));
                }
            }
#else
            if (sg) {
                _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++)
                {
                    __m256i wv =
                        _mm256_loadu_si256((const __m256i *)k->qs[kb][q]);
                    __m256i aw8 = _mm256_abs_epi8(wv);
                    _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                    {
                        int32_t aw;
                        memcpy(&aw,
                               x[(size_t)t * xs + b].qs32 + 32 * kb + 4 * q, 4);
                        s[t] = _mm256_add_epi32(
                            s[t], _mm256_madd_epi16(
                                      _mm256_maddubs_epi16(
                                          aw8, _mm256_sign_epi8(
                                                   _mm256_set1_epi32(aw), wv)),
                                      ones));
                    }
                }
            } else {
                _Pragma("GCC unroll 2") for (int h = 0; h < 8 / G; h++)
                {
                    __m256i a16[8];
                    _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) a16[t] =
                        _mm256_setzero_si256();
                    _Pragma("GCC unroll 8") for (int q = G * h; q < G * h + G;
                                                 q++)
                    {
                        __m256i wv =
                            _mm256_loadu_si256((const __m256i *)k->qs[kb][q]);
                        _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                        {
                            int32_t aw;
                            memcpy(&aw,
                                   x[(size_t)t * xs + b].qs32 + 32 * kb + 4 * q,
                                   4);
                            a16[t] = _mm256_add_epi16(
                                a16[t], _mm256_maddubs_epi16(
                                            wv, _mm256_set1_epi32(aw)));
                        }
                    }
                    _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) s[t] =
                        _mm256_add_epi32(s[t], _mm256_madd_epi16(a16[t], ones));
                }
            }
#endif
            __m256 dv = _mm256_loadu_ps(k->d[kb]);
            __m256 mv = _mm256_loadu_ps(k->m[kb]);
            _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
            {
                const struct janas_block_q8k *xb = x + (size_t)t * xs + b;
                int asum = xb->bsums32[2 * kb] + xb->bsums32[2 * kb + 1];
                __m256 d8 = _mm256_set1_ps(xb->d32[kb]);
                float *pa = af + 64 * t + 8 * kb;
                __m256i is =
                    o ? _mm256_sub_epi32(s[t], _mm256_set1_epi32(o * asum))
                      : s[t];
                _mm256_storeu_ps(pa, _mm256_fmadd_ps(_mm256_mul_ps(dv, d8),
                                                     _mm256_cvtepi32_ps(is),
                                                     _mm256_loadu_ps(pa)));
                if (dm) {
                    float *pm = am + 64 * t + 8 * kb;
                    _mm256_storeu_ps(
                        pm, _mm256_fmadd_ps(_mm256_mul_ps(mv, d8),
                                            _mm256_set1_ps((float)asum),
                                            _mm256_loadu_ps(pm)));
                }
            }
        }
    }
}

/* The eight positions of a token's sums, row by row, as janas_hsum8 */
static inline DOTN_TARGET __attribute__((always_inline)) __m256
DOTN(b32g_hsum8)(const float *a)
{
    __m256 p0 = _mm256_loadu_ps(a), p1 = _mm256_loadu_ps(a + 8),
           p2 = _mm256_loadu_ps(a + 16), p3 = _mm256_loadu_ps(a + 24),
           p4 = _mm256_loadu_ps(a + 32), p5 = _mm256_loadu_ps(a + 40),
           p6 = _mm256_loadu_ps(a + 48), p7 = _mm256_loadu_ps(a + 56);
    return _mm256_add_ps(
        _mm256_add_ps(_mm256_add_ps(p0, p4), _mm256_add_ps(p2, p6)),
        _mm256_add_ps(_mm256_add_ps(p1, p5), _mm256_add_ps(p3, p7)));
}

/* tokens whose sums are held at a time, as Q4KG_TOK: 64 KB (with 32, Q8_0
   on 256 vectors 2.0 against 2.3 GB/s) */
#define B32G_TOK 256

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(b32_gemm_t)(const int TY, const void *w, size_t row_bytes,
                 const struct janas_block_q8k *x, size_t xs, size_t nv,
                 size_t nb, size_t rows, float *y, size_t ys)
{
    struct b32g_blk bk[B32G_CHUNK];
    float af[B32G_TOK * 64], am[B32G_TOK * 64];
    const int dm = TY == JANAS_Q4_1 || TY == JANAS_Q5_1;
    for (size_t g = 0; g < rows; g += 8) {
        const uint8_t *wr[8];
        for (int j = 0; j < 8; j++)
            wr[j] = (const uint8_t *)w + (g + j) * row_bytes;
        for (size_t v0 = 0; v0 < nv; v0 += B32G_TOK) {
            size_t nt = nv - v0 < B32G_TOK ? nv - v0 : B32G_TOK;
            memset(af, 0, nt * 64 * sizeof(float));
            if (dm)
                memset(am, 0, nt * 64 * sizeof(float));
            for (size_t b0 = 0; b0 < nb; b0 += B32G_CHUNK) {
                int n = nb - b0 < B32G_CHUNK ? (int)(nb - b0) : B32G_CHUNK;
                DOTN(b32g_fill)(TY, wr, b0, n, bk);
                /* not Q8_0's: 3-4% slower on 24-32 vectors */
                if (TY != JANAS_Q8_0)
                    DOTN(gemm_ahead)(w, row_bytes,
                                     TY == JANAS_Q2_0
                                         ? 4 * sizeof(struct janas_block_q2_0)
                                         : 8 * DOTN(b32g_size)(TY),
                                     g, rows, nb, b0, B32G_CHUNK);
                const struct janas_block_q8k *xv = x + v0 * xs;
                size_t t0 = 0;
                for (; t0 + 4 <= nt; t0 += 4)
                    DOTN(b32g_tile)(TY, bk, n, b0, xv + t0 * xs, xs,
                                    af + 64 * t0, am + 64 * t0, 4);
                if (nt - t0 == 3)
                    DOTN(b32g_tile)(TY, bk, n, b0, xv + t0 * xs, xs,
                                    af + 64 * t0, am + 64 * t0, 3);
                else if (nt - t0 == 2)
                    DOTN(b32g_tile)(TY, bk, n, b0, xv + t0 * xs, xs,
                                    af + 64 * t0, am + 64 * t0, 2);
                else if (nt - t0 == 1)
                    DOTN(b32g_tile)(TY, bk, n, b0, xv + t0 * xs, xs,
                                    af + 64 * t0, am + 64 * t0, 1);
            }
            for (size_t t = 0; t < nt; t++) {
                float r[8], rm[8];
                _mm256_storeu_ps(r, DOTN(b32g_hsum8)(af + 64 * t));
                if (dm)
                    _mm256_storeu_ps(rm, DOTN(b32g_hsum8)(am + 64 * t));
                for (int j = 0; j < 8; j++)
                    y[g + j + (v0 + t) * ys] = dm ? r[j] + rm[j] : r[j];
            }
        }
    }
}

/* rows (a multiple of 8) of a type in blocks of 32 against nv vectors,
   y[r + v ys]: 1, or 0 for another type */
static DOTN_TARGET int DOTN(b32_gemm)(int type, const void *w, size_t row_bytes,
                                      const struct janas_block_q8k *x,
                                      size_t xs, size_t nv, size_t nb,
                                      size_t rows, float *y, size_t ys)
{
    switch (type) {
#define B32G(T)                                                                \
    case T:                                                                    \
        DOTN(b32_gemm_t)(T, w, row_bytes, x, xs, nv, nb, rows, y, ys);         \
        return 1;
        B32G(JANAS_Q8_0)
        B32G(JANAS_IQ4_NL)
        B32G(JANAS_Q4_0)
        B32G(JANAS_MXFP4)
        B32G(JANAS_Q5_0)
        B32G(JANAS_Q4_1)
        B32G(JANAS_Q5_1)
        B32G(JANAS_Q2_0)
#undef B32G
    }
    return 0;
}

/*
 * IQ4_XS as tiles (see q4k_gemm): its values from the IQ4 table, signed,
 * transposed four bytes at a time, its eight 32-weight groups each with a
 * signed scale. AVX-VNNI: the values plus 128 by vpdpbusd, a multiply by
 * the scales a group, and 128 times the scales' sums of activations (made
 * by madd from the mins' table, here the scales) taken off; AVX2: sign and
 * abs, the scales by madd. The integer sums are exact, and the float one
 * fma a block, d x d8 times the sum, as the products above: the same bits.
 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(iq4xsg_fill)(const struct janas_block_iq4xs *const *w, size_t b0, int n,
                  struct q5kg_blk *bk)
{
    const __m256i levels = DOTN(iq4_levels)();
    for (int i = 0; i < n; i++) {
        struct q5kg_blk *k = bk + i;
        size_t b = b0 + (size_t)i;
        for (int g = 0; g < 8; g++) {
            __m256i r[8];
            _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
            {
                r[j] = DOTN(iq4_unpack32)(w[j][b].qs + 16 * g, levels);
#ifdef DOTN_HAS_VNNI
                r[j] = _mm256_xor_si256(r[j], _mm256_set1_epi8((char)0x80));
#endif
            }
            DOTN(gemm_t8)(r, k->qs + 8 * g);
        }
        __m128i sv[8];
        _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
        {
            int8_t ls[16];
            for (int g = 0; g < 8; g++)
                ls[g] = ls[8 + g] = (int8_t)iq4xs_scale(&w[j][b], g);
            sv[j] = _mm_loadu_si128((const __m128i *)ls);
            k->d[j] = _cvtsh_ss(w[j][b].d);
        }
        DOTN(kmin_scales8)(k->sc, k->sc32, k->mn, sv, 1);
    }
}

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(iq4xsg_tile)(const struct q5kg_blk *bk, int n, size_t b0,
                  const struct janas_block_q8k *x, size_t xs, float *af,
                  const int T)
{
    for (int i = 0; i < n; i++) {
        const struct q5kg_blk *k = bk + i;
        size_t b = b0 + (size_t)i;
        __m256i is[8];
        _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
            _mm256_setzero_si256();
        for (int sb = 0; sb < 8; sb++) {
#ifdef DOTN_HAS_VNNI
            __m256i a32[8];
            _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) a32[t] =
                _mm256_setzero_si256();
            _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++)
            {
                __m256i wv =
                    _mm256_loadu_si256((const __m256i *)k->qs[8 * sb + q]);
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                {
                    int32_t aw;
                    memcpy(&aw, x[(size_t)t * xs + b].qs + 32 * sb + 4 * q, 4);
                    a32[t] = DOTN_DPBUSD(a32[t], wv, _mm256_set1_epi32(aw));
                }
            }
            __m256i sc = _mm256_loadu_si256((const __m256i *)k->sc32[sb]);
            _Pragma("GCC unroll 8") for (int t = 0; t < T; t++) is[t] =
                _mm256_add_epi32(is[t], _mm256_mullo_epi32(a32[t], sc));
#else
            __m256i sc = _mm256_loadu_si256((const __m256i *)k->sc[sb]);
            _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++)
            {
                __m256i wv =
                    _mm256_loadu_si256((const __m256i *)k->qs[8 * sb + q]);
                __m256i aw8 = _mm256_abs_epi8(wv);
                _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
                {
                    int32_t aw;
                    memcpy(&aw, x[(size_t)t * xs + b].qs + 32 * sb + 4 * q, 4);
                    is[t] = DOTN_ACC(
                        is[t],
                        _mm256_maddubs_epi16(
                            aw8, _mm256_sign_epi8(_mm256_set1_epi32(aw), wv)),
                        sc);
                }
            }
#endif
        }
        __m256 dw = _mm256_loadu_ps(k->d);
        _Pragma("GCC unroll 8") for (int t = 0; t < T; t++)
        {
            const struct janas_block_q8k *xb = x + (size_t)t * xs + b;
            __m256i si = is[t];
#ifdef DOTN_HAS_VNNI
            __m256i cr = _mm256_setzero_si256();
            _Pragma("GCC unroll 4") for (int kk = 0; kk < 4; kk++)
            {
                int32_t bw;
                memcpy(&bw, xb->bsums2 + 2 * kk, 4);
                cr = _mm256_add_epi32(
                    cr, _mm256_madd_epi16(
                            _mm256_loadu_si256((const __m256i *)k->mn[kk]),
                            _mm256_set1_epi32(bw)));
            }
            si = _mm256_sub_epi32(si, _mm256_slli_epi32(cr, 7));
#endif
            _mm256_storeu_ps(
                af + 8 * t,
                _mm256_fmadd_ps(_mm256_mul_ps(dw, _mm256_set1_ps(xb->d)),
                                _mm256_cvtepi32_ps(si),
                                _mm256_loadu_ps(af + 8 * t)));
        }
    }
}

/* rows (a multiple of 8) of IQ4_XS against nv vectors, y[r + v ys] */
static DOTN_TARGET void DOTN(iq4xs_gemm)(const void *w, size_t row_bytes,
                                         const struct janas_block_q8k *x,
                                         size_t xs, size_t nv, size_t nb,
                                         size_t rows, float *y, size_t ys)
{
    struct q5kg_blk bk[Q5KG_CHUNK];
    float af[Q4KG_TOK * 8];
    for (size_t g = 0; g < rows; g += 8) {
        const struct janas_block_iq4xs *wr[8];
        for (int j = 0; j < 8; j++)
            wr[j] = (const void *)((const char *)w + (g + j) * row_bytes);
        for (size_t v0 = 0; v0 < nv; v0 += Q4KG_TOK) {
            size_t nt = nv - v0 < Q4KG_TOK ? nv - v0 : Q4KG_TOK;
            memset(af, 0, nt * 8 * sizeof(float));
            for (size_t b0 = 0; b0 < nb; b0 += Q5KG_CHUNK) {
                int n = nb - b0 < Q5KG_CHUNK ? (int)(nb - b0) : Q5KG_CHUNK;
                DOTN(iq4xsg_fill)(wr, b0, n, bk);
                DOTN(gemm_ahead)(w, row_bytes, sizeof(struct janas_block_iq4xs),
                                 g, rows, nb, b0, Q5KG_CHUNK);
                const struct janas_block_q8k *xv = x + v0 * xs;
                size_t t0 = 0;
                for (; t0 + Q4KG_T <= nt; t0 += Q4KG_T)
                    DOTN(iq4xsg_tile)(bk, n, b0, xv + t0 * xs, xs, af + 8 * t0,
                                      Q4KG_T);
                for (; t0 + 4 <= nt; t0 += 4)
                    DOTN(iq4xsg_tile)(bk, n, b0, xv + t0 * xs, xs, af + 8 * t0,
                                      4);
                if (nt - t0 == 3)
                    DOTN(iq4xsg_tile)(bk, n, b0, xv + t0 * xs, xs, af + 8 * t0,
                                      3);
                else if (nt - t0 == 2)
                    DOTN(iq4xsg_tile)(bk, n, b0, xv + t0 * xs, xs, af + 8 * t0,
                                      2);
                else if (nt - t0 == 1)
                    DOTN(iq4xsg_tile)(bk, n, b0, xv + t0 * xs, xs, af + 8 * t0,
                                      1);
            }
            for (size_t t = 0; t < nt; t++)
                for (int j = 0; j < 8; j++)
                    y[g + j + (v0 + t) * ys] = af[8 * t + j];
        }
    }
}

/* Q4_0: IQ4_NL's kernel with the evenly spaced levels. */
static DOTN_TARGET void DOTN(q4_0_dotn)(const struct janas_block_q4_0 *w,
                                        const struct janas_block_q8k *x,
                                        size_t xs, int nv, size_t nb,
                                        float *out)
{
    const struct janas_block_iq4nl *wn = (const struct janas_block_iq4nl *)w;
    switch (nv) {
    case 8:
        DOTN(nib_dotn_body)(wn, x, xs, nb, out, 8, janas_q4_0_values);
        break;
    case 7:
        DOTN(nib_dotn_body)(wn, x, xs, nb, out, 7, janas_q4_0_values);
        break;
    case 6:
        DOTN(nib_dotn_body)(wn, x, xs, nb, out, 6, janas_q4_0_values);
        break;
    case 5:
        DOTN(nib_dotn_body)(wn, x, xs, nb, out, 5, janas_q4_0_values);
        break;
    case 4:
        DOTN(nib_dotn_body)(wn, x, xs, nb, out, 4, janas_q4_0_values);
        break;
    case 3:
        DOTN(nib_dotn_body)(wn, x, xs, nb, out, 3, janas_q4_0_values);
        break;
    case 2:
        DOTN(nib_dotn_body)(wn, x, xs, nb, out, 2, janas_q4_0_values);
        break;
    default:
        DOTN(nib_dotn_body)(wn, x, xs, nb, out, 1, janas_q4_0_values);
    }
}

static DOTN_TARGET float DOTN(q4_0_dot)(const struct janas_block_q4_0 *w,
                                        const struct janas_block_q8k *x,
                                        size_t nb)
{
    float r;
    DOTN(nib_dotn_body)((const struct janas_block_iq4nl *)w, x, 0, nb, &r, 1,
                        janas_q4_0_values);
    return r;
}

/* MXFP4: IQ4_NL's body with the block's E8M0 byte for its scale, 2^(e -
   128) from e8m0_tab - janas_e8m0_half's bits, subnormal below e = 2 - and
   the codes' levels for the table. */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(mx_dotn_body)(const struct janas_block_mxfp4 *w,
                   const struct janas_block_q8k *x, size_t xs, size_t nb,
                   float *out, const int nv)
{
    const __m256i ones = _mm256_set1_epi16(1),
                  levels = _mm256_broadcastsi128_si256(
                      _mm_loadu_si128((const __m128i *)janas_mxfp4_values));
    __m256 acc[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) acc[v] =
        _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(w + 8 * b, 8 * sizeof(*w));
        const struct janas_block_mxfp4 *wb = w + 8 * b;
        __m256 dk = _mm256_setr_ps(e8m0_tab[wb[0].e], e8m0_tab[wb[1].e],
                                   e8m0_tab[wb[2].e], e8m0_tab[wb[3].e],
                                   e8m0_tab[wb[4].e], e8m0_tab[wb[5].e],
                                   e8m0_tab[wb[6].e], e8m0_tab[wb[7].e]);
        __m256i q[8], aq[8];
        _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
        {
            q[k] = DOTN(iq4_unpack32)(wb[k].qs, levels);
            aq[k] = _mm256_abs_epi8(q[k]);
        }
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i s[8];
            _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs32 + 32 * k));
                __m256i p =
                    _mm256_maddubs_epi16(aq[k], _mm256_sign_epi8(a, q[k]));
                s[k] = _mm256_madd_epi16(p, ones);
            }
            __m256i s01 = _mm256_hadd_epi32(s[0], s[1]);
            __m256i s23 = _mm256_hadd_epi32(s[2], s[3]);
            __m256i s45 = _mm256_hadd_epi32(s[4], s[5]);
            __m256i s67 = _mm256_hadd_epi32(s[6], s[7]);
            __m256i t0 = _mm256_hadd_epi32(s01, s23);
            __m256i t1 = _mm256_hadd_epi32(s45, s67);
            __m256i is8 =
                _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20),
                                 _mm256_permute2x128_si256(t0, t1, 0x31));
            acc[v] =
                _mm256_fmadd_ps(_mm256_mul_ps(dk, _mm256_loadu_ps(xb->d32)),
                                _mm256_cvtepi32_ps(is8), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
    {
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(acc[v]),
                              _mm256_extractf128_ps(acc[v], 1));
        t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        t = _mm_add_ss(t, _mm_movehdup_ps(t));
        out[v] = _mm_cvtss_f32(t);
    }
}

static DOTN_TARGET void DOTN(mxfp4_dotn)(const struct janas_block_mxfp4 *w,
                                         const struct janas_block_q8k *x,
                                         size_t xs, int nv, size_t nb,
                                         float *out)
{
    switch (nv) {
    case 8:
        DOTN(mx_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(mx_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(mx_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(mx_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(mx_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(mx_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(mx_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(mx_dotn_body)(w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(mxfp4_dot)(const struct janas_block_mxfp4 *w,
                                         const struct janas_block_q8k *x,
                                         size_t nb)
{
    float r;
    DOTN(mx_dotn_body)(w, x, 0, nb, &r, 1);
    return r;
}

/* The eight sums of eight int32 lanes each, lane k the sum of s[k]: the
   reduction tree of Q8_0. */
static inline DOTN_TARGET __attribute__((always_inline)) __m256i
DOTN(sum8x8)(const __m256i *s)
{
    __m256i s01 = _mm256_hadd_epi32(s[0], s[1]);
    __m256i s23 = _mm256_hadd_epi32(s[2], s[3]);
    __m256i s45 = _mm256_hadd_epi32(s[4], s[5]);
    __m256i s67 = _mm256_hadd_epi32(s[6], s[7]);
    __m256i t0 = _mm256_hadd_epi32(s01, s23);
    __m256i t1 = _mm256_hadd_epi32(s45, s67);
    return _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20),
                            _mm256_permute2x128_si256(t0, t1, 0x31));
}

/* The eight lanes added as janas_hsum8 does. */
static inline DOTN_TARGET __attribute__((always_inline)) float
DOTN(hsum8_ps)(__m256 a)
{
    __m128 t =
        _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    t = _mm_add_ps(t, _mm_movehl_ps(t, t));
    t = _mm_add_ss(t, _mm_movehdup_ps(t));
    return _mm_cvtss_f32(t);
}

/*
 * Q4_1 and Q5_1 (see dm_dot_ref in quant.c): the values (0-15, or 0-31
 * with the fifth bits from qh, bit l of it into byte l) are unsigned, so
 * maddubs takes them as they are. Per block position one lane for d x the
 * exact weight sum and one for m x the activations' sum (two Q8_K block
 * sums), as the reference.
 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(dm_dotn_body)(const uint8_t *w, const size_t size, const int five,
                   const struct janas_block_q8k *x, size_t xs, size_t nb,
                   float *out, const int nv)
{
    const __m256i ones = _mm256_set1_epi16(1), low4 = _mm256_set1_epi8(0x0f);
    /* byte l of a lane takes byte l / 8 of qh (bytes 2, 3 in the upper
       lane, whose weights are 16-31), then keeps bit l % 8 of it */
    const __m256i sel =
        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
                         2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ull);
    const __m256i fifth = _mm256_set1_epi8(0x10);
    __m256 acc[8], accm[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
    {
        acc[v] = _mm256_setzero_ps();
        accm[v] = _mm256_setzero_ps();
    }
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(w + 8 * b * size, 8 * size);
        const uint8_t *wb = w + 8 * b * size;
        uint16_t d[8], m[8];
        _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
        {
            memcpy(&d[k], wb + k * size, 2);
            memcpy(&m[k], wb + k * size + 2, 2);
        }
        __m256 dk = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)d));
        __m256 mk = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)m));
        __m256i q[8];
        _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
        {
            const uint8_t *blk = wb + k * size;
            __m128i b16 =
                _mm_loadu_si128((const __m128i *)(blk + (five ? 8 : 4)));
            q[k] = _mm256_and_si256(
                _mm256_set_m128i(_mm_srli_epi16(b16, 4), b16), low4);
            if (five) {
                uint32_t qh;
                memcpy(&qh, blk + 4, 4);
                __m256i h =
                    _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), sel);
                h = _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(h, bit), bit), fifth);
                q[k] = _mm256_or_si256(q[k], h);
            }
        }
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i s[8];
            _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs32 + 32 * k));
                s[k] = _mm256_madd_epi16(_mm256_maddubs_epi16(q[k], a), ones);
            }
            __m256i is8 = DOTN(sum8x8)(s);
            /* bsums 2k and 2k + 1 are block k's 32 activations */
            __m256i as8 = _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *)xb->bsums32), ones);
            __m256 dx = _mm256_loadu_ps(xb->d32);
            acc[v] = _mm256_fmadd_ps(_mm256_mul_ps(dk, dx),
                                     _mm256_cvtepi32_ps(is8), acc[v]);
            accm[v] = _mm256_fmadd_ps(_mm256_mul_ps(mk, dx),
                                      _mm256_cvtepi32_ps(as8), accm[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] =
        DOTN(hsum8_ps)(acc[v]) + DOTN(hsum8_ps)(accm[v]);
}

static DOTN_TARGET void DOTN(q5_1_dotn)(const struct janas_block_q5_1 *w,
                                        const struct janas_block_q8k *x,
                                        size_t xs, int nv, size_t nb,
                                        float *out)
{
    const uint8_t *p = (const uint8_t *)w;
    switch (nv) {
    case 8:
        DOTN(dm_dotn_body)(p, sizeof(*w), 1, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(dm_dotn_body)(p, sizeof(*w), 1, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(dm_dotn_body)(p, sizeof(*w), 1, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(dm_dotn_body)(p, sizeof(*w), 1, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(dm_dotn_body)(p, sizeof(*w), 1, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(dm_dotn_body)(p, sizeof(*w), 1, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(dm_dotn_body)(p, sizeof(*w), 1, x, xs, nb, out, 2);
        break;
    default:
        DOTN(dm_dotn_body)(p, sizeof(*w), 1, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(q5_1_dot)(const struct janas_block_q5_1 *w,
                                        const struct janas_block_q8k *x,
                                        size_t nb)
{
    float r;
    DOTN(dm_dotn_body)((const uint8_t *)w, sizeof(*w), 1, x, 0, nb, &r, 1);
    return r;
}

static DOTN_TARGET void DOTN(q4_1_dotn)(const struct janas_block_q4_1 *w,
                                        const struct janas_block_q8k *x,
                                        size_t xs, int nv, size_t nb,
                                        float *out)
{
    const uint8_t *p = (const uint8_t *)w;
    switch (nv) {
    case 8:
        DOTN(dm_dotn_body)(p, sizeof(*w), 0, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(dm_dotn_body)(p, sizeof(*w), 0, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(dm_dotn_body)(p, sizeof(*w), 0, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(dm_dotn_body)(p, sizeof(*w), 0, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(dm_dotn_body)(p, sizeof(*w), 0, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(dm_dotn_body)(p, sizeof(*w), 0, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(dm_dotn_body)(p, sizeof(*w), 0, x, xs, nb, out, 2);
        break;
    default:
        DOTN(dm_dotn_body)(p, sizeof(*w), 0, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(q4_1_dot)(const struct janas_block_q4_1 *w,
                                        const struct janas_block_q8k *x,
                                        size_t nb)
{
    float r;
    DOTN(dm_dotn_body)((const uint8_t *)w, sizeof(*w), 0, x, 0, nb, &r, 1);
    return r;
}

/*
 * Q5_0 (see janas_q5_0_dot_ref in quant.c): Q5_1's unpacking, the values
 * 0-31 unsigned for maddubs, then 16 times the activations' sum taken off:
 * the exact sum of (q - 16) x a, one lane per block position.
 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q5_0_dotn_body)(const struct janas_block_q5_0 *w,
                     const struct janas_block_q8k *x, size_t xs, size_t nb,
                     float *out, const int nv)
{
    const __m256i ones = _mm256_set1_epi16(1), low4 = _mm256_set1_epi8(0x0f);
    /* as dm_dotn_body: byte l of a lane takes byte l / 8 of qh, then keeps
       bit l % 8 of it */
    const __m256i sel =
        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
                         2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ull);
    const __m256i fifth = _mm256_set1_epi8(0x10);
    __m256 acc[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) acc[v] =
        _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(w + 8 * b, 8 * sizeof(*w));
        const struct janas_block_q5_0 *wb = w + 8 * b;
        __m256 dk = _mm256_cvtph_ps(_mm_setr_epi16(
            (short)wb[0].d, (short)wb[1].d, (short)wb[2].d, (short)wb[3].d,
            (short)wb[4].d, (short)wb[5].d, (short)wb[6].d, (short)wb[7].d));
        __m256i q[8];
        _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
        {
            __m128i b16 = _mm_loadu_si128((const __m128i *)wb[k].qs);
            uint32_t qh;
            memcpy(&qh, wb[k].qh, 4);
            __m256i h = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), sel);
            h = _mm256_and_si256(
                _mm256_cmpeq_epi8(_mm256_and_si256(h, bit), bit), fifth);
            q[k] = _mm256_or_si256(
                _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(b16, 4), b16),
                                 low4),
                h);
        }
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i s[8];
            _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs32 + 32 * k));
                s[k] = _mm256_madd_epi16(_mm256_maddubs_epi16(q[k], a), ones);
            }
            /* bsums 2k and 2k + 1 are block k's 32 activations */
            __m256i as8 = _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *)xb->bsums32), ones);
            __m256i is8 =
                _mm256_sub_epi32(DOTN(sum8x8)(s), _mm256_slli_epi32(as8, 4));
            acc[v] =
                _mm256_fmadd_ps(_mm256_mul_ps(dk, _mm256_loadu_ps(xb->d32)),
                                _mm256_cvtepi32_ps(is8), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] =
        DOTN(hsum8_ps)(acc[v]);
}

#ifdef DOTN_HAS_VNNI
/*
 * The types in blocks of 32 (Q8_0, IQ4_NL, Q4_0, MXFP4, Q5_0, Q4_1, Q5_1)
 * on 2-8 vectors with no reduction tree (AVX-VNNI): each group's eight
 * blocks, unpacked to bytes, transposed in registers - register r holding
 * bytes 4r..4r+3 of block j in int32 lane j - and the vectors' activations
 * transposed the same way once for all the rows (b32_xt), so that eight
 * vpdpbusd leave block j's sum in lane j. vpdpbusd takes unsigned weights:
 * the signed ones go as w + o (o = 128, or Q4_0's 8 and Q5_0's 16 as they
 * are stored) and o times each block's sum of activations, made ready with
 * them, is taken off: sum (w + o) a - o sum a = sum w a, exactly. A
 * vector's products were eight sign, maddubs and madd and a tree of six
 * hadd; the float operations are each type's own, the same: the same bits.
 * Q8_0 on eight vectors from 6.6 to 14.7 GB/s on a core and from 25 to 55
 * on twelve threads (24 Oct 2026).
 */
#define B32XT 288 /* bytes a vector a group: 256 transposed, 8 int32 sums */
#define B32XT_MAX (48 * 1024)

/* The eight registers' 8 x 8 int32 transposed: lane j of r[i] to lane i of
   r[j]. */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(t8x8_epi32)(__m256i *r)
{
    __m256i a0 = _mm256_unpacklo_epi32(r[0], r[1]);
    __m256i a1 = _mm256_unpackhi_epi32(r[0], r[1]);
    __m256i a2 = _mm256_unpacklo_epi32(r[2], r[3]);
    __m256i a3 = _mm256_unpackhi_epi32(r[2], r[3]);
    __m256i a4 = _mm256_unpacklo_epi32(r[4], r[5]);
    __m256i a5 = _mm256_unpackhi_epi32(r[4], r[5]);
    __m256i a6 = _mm256_unpacklo_epi32(r[6], r[7]);
    __m256i a7 = _mm256_unpackhi_epi32(r[6], r[7]);
    __m256i b0 = _mm256_unpacklo_epi64(a0, a2);
    __m256i b1 = _mm256_unpackhi_epi64(a0, a2);
    __m256i b2 = _mm256_unpacklo_epi64(a1, a3);
    __m256i b3 = _mm256_unpackhi_epi64(a1, a3);
    __m256i b4 = _mm256_unpacklo_epi64(a4, a6);
    __m256i b5 = _mm256_unpackhi_epi64(a4, a6);
    __m256i b6 = _mm256_unpacklo_epi64(a5, a7);
    __m256i b7 = _mm256_unpackhi_epi64(a5, a7);
    r[0] = _mm256_permute2x128_si256(b0, b4, 0x20);
    r[1] = _mm256_permute2x128_si256(b1, b5, 0x20);
    r[2] = _mm256_permute2x128_si256(b2, b6, 0x20);
    r[3] = _mm256_permute2x128_si256(b3, b7, 0x20);
    r[4] = _mm256_permute2x128_si256(b0, b4, 0x31);
    r[5] = _mm256_permute2x128_si256(b1, b5, 0x31);
    r[6] = _mm256_permute2x128_si256(b2, b6, 0x31);
    r[7] = _mm256_permute2x128_si256(b3, b7, 0x31);
}

/* nv vectors' activations transposed (block j's dword r in lane j of
   register r) and their blocks' sums: vector v's group b at xt + B32XT (v
   nb + b). */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(b32_xt)(const struct janas_block_q8k *x, size_t xs, size_t nv, size_t nb,
             uint8_t *xt)
{
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t v = 0; v < nv; v++)
        for (size_t b = 0; b < nb; b++) {
            const struct janas_block_q8k *xb = x + v * xs + b;
            uint8_t *o = xt + B32XT * (v * nb + b);
            __m256i r[8];
            _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++) r[j] =
                _mm256_loadu_si256((const __m256i *)(xb->qs32 + 32 * j));
            DOTN(t8x8_epi32)(r);
            _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++)
                _mm256_storeu_si256((__m256i *)(o + 32 * j), r[j]);
            _mm256_storeu_si256(
                (__m256i *)(o + 256),
                _mm256_madd_epi16(
                    _mm256_loadu_si256((const __m256i *)xb->bsums32), ones));
        }
}

/* A block's bytes and the unsigned 32 values of its weights, the offset o
   added (see above); its d (and m) at its start, MXFP4's E8M0 byte. */
static inline DOTN_TARGET __attribute__((always_inline)) size_t
DOTN(b32_size)(const int type)
{
    return type == JANAS_Q8_0     ? sizeof(struct janas_block_q8_0)
           : type == JANAS_MXFP4  ? sizeof(struct janas_block_mxfp4)
           : type == JANAS_Q5_0   ? sizeof(struct janas_block_q5_0)
           : type == JANAS_Q4_1   ? sizeof(struct janas_block_q4_1)
           : type == JANAS_Q5_1   ? sizeof(struct janas_block_q5_1)
           : type == JANAS_IQ4_NL ? sizeof(struct janas_block_iq4nl)
                                  : sizeof(struct janas_block_q4_0);
}

static inline DOTN_TARGET __attribute__((always_inline)) __m256i
DOTN(b32_unpack)(const int type, const uint8_t *blk)
{
    const __m256i flip = _mm256_set1_epi8((char)0x80),
                  low4 = _mm256_set1_epi8(0x0f);
    if (type == JANAS_Q8_0)
        return _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(blk + 2)),
                                flip);
    if (type == JANAS_IQ4_NL || type == JANAS_MXFP4) {
        const int8_t *t =
            type == JANAS_MXFP4 ? janas_mxfp4_values : janas_iq4_values;
        __m256i lev =
            _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)t));
        return _mm256_xor_si256(
            DOTN(iq4_unpack32)(blk + (type == JANAS_MXFP4 ? 1 : 2), lev), flip);
    }
    /* the nibbles, low ones weights 0-15 (Q4_0, Q5_0, Q4_1, Q5_1) */
    size_t at = type == JANAS_Q4_0   ? 2
                : type == JANAS_Q5_0 ? 6
                : type == JANAS_Q4_1 ? 4
                                     : 8;
    __m128i b16 = _mm_loadu_si128((const __m128i *)(blk + at));
    __m256i q =
        _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(b16, 4), b16), low4);
    if (type == JANAS_Q5_0 || type == JANAS_Q5_1) {
        /* as dm_dotn_body: byte l of a lane takes byte l / 8 of qh, then
           keeps bit l % 8 of it */
        const __m256i sel =
            _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                             2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        const __m256i bit =
            _mm256_set1_epi64x((long long)0x8040201008040201ull);
        uint32_t qh;
        memcpy(&qh, blk + (type == JANAS_Q5_0 ? 2 : 4), 4);
        __m256i h = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), sel);
        h = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(h, bit), bit),
                             _mm256_set1_epi8(0x10));
        q = _mm256_or_si256(q, h);
    }
    return q;
}

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(b32_xt_body)(const int type, const uint8_t *w,
                  const struct janas_block_q8k *x, size_t xs, const uint8_t *xt,
                  size_t nb, float *out, const int nv)
{
    const size_t size = DOTN(b32_size)(type);
    /* log2 of the offset o; Q4_1 and Q5_1 have none, but their mins */
    const int lo = type == JANAS_Q4_0 ? 3 : type == JANAS_Q5_0 ? 4 : 7;
    const int dm = type == JANAS_Q4_1 || type == JANAS_Q5_1;
    __m256 acc[8], accm[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
    {
        acc[v] = _mm256_setzero_ps();
        accm[v] = _mm256_setzero_ps();
    }
    for (size_t b = 0; b < nb; b++) {
        const uint8_t *wb = w + 8 * b * size;
        if (nv <= 2)
            DOTN(prefetch_ahead)(wb, 8 * size);
        __m256 dk, mk = _mm256_setzero_ps();
        if (type == JANAS_MXFP4) {
            dk = _mm256_setr_ps(e8m0_tab[wb[0]], e8m0_tab[wb[size]],
                                e8m0_tab[wb[2 * size]], e8m0_tab[wb[3 * size]],
                                e8m0_tab[wb[4 * size]], e8m0_tab[wb[5 * size]],
                                e8m0_tab[wb[6 * size]], e8m0_tab[wb[7 * size]]);
        } else {
            uint16_t d[8], m[8];
            _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
            {
                memcpy(&d[k], wb + k * size, 2);
                if (dm)
                    memcpy(&m[k], wb + k * size + 2, 2);
            }
            dk = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)d));
            if (dm)
                mk = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)m));
        }
        __m256i q[8];
        _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++) q[k] =
            DOTN(b32_unpack)(type, wb + k * size);
        DOTN(t8x8_epi32)(q);
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const __m256i *a =
                (const __m256i *)(xt + B32XT * ((size_t)v * nb + b));
            __m256i s = _mm256_setzero_si256();
            _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++) s =
                DOTN_DPBUSD(s, q[k], _mm256_loadu_si256(a + k));
            __m256i as8 = _mm256_loadu_si256(a + 8);
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256 dx = _mm256_loadu_ps(xb->d32);
            if (dm) {
                acc[v] = _mm256_fmadd_ps(_mm256_mul_ps(dk, dx),
                                         _mm256_cvtepi32_ps(s), acc[v]);
                accm[v] = _mm256_fmadd_ps(_mm256_mul_ps(mk, dx),
                                          _mm256_cvtepi32_ps(as8), accm[v]);
            } else {
                __m256i is8 = _mm256_sub_epi32(s, _mm256_slli_epi32(as8, lo));
                acc[v] = _mm256_fmadd_ps(_mm256_mul_ps(dk, dx),
                                         _mm256_cvtepi32_ps(is8), acc[v]);
            }
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] =
        dm ? DOTN(hsum8_ps)(acc[v]) + DOTN(hsum8_ps)(accm[v])
           : DOTN(hsum8_ps)(acc[v]);
}

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(b32_xt_n)(const int type, const uint8_t *w,
               const struct janas_block_q8k *x, size_t xs, const uint8_t *xt,
               size_t nb, float *o, int n)
{
    switch (n) {
    case 8:
        DOTN(b32_xt_body)(type, w, x, xs, xt, nb, o, 8);
        break;
    case 7:
        DOTN(b32_xt_body)(type, w, x, xs, xt, nb, o, 7);
        break;
    case 6:
        DOTN(b32_xt_body)(type, w, x, xs, xt, nb, o, 6);
        break;
    case 5:
        DOTN(b32_xt_body)(type, w, x, xs, xt, nb, o, 5);
        break;
    case 4:
        DOTN(b32_xt_body)(type, w, x, xs, xt, nb, o, 4);
        break;
    case 3:
        DOTN(b32_xt_body)(type, w, x, xs, xt, nb, o, 3);
        break;
    case 2:
        DOTN(b32_xt_body)(type, w, x, xs, xt, nb, o, 2);
        break;
    default:
        DOTN(b32_xt_body)(type, w, x, xs, xt, nb, o, 1);
    }
}

/* rows rows of one of these types against nv vectors, y[r + v ys]: up to
   eight vectors transposed at a time, as many as fit in B32XT_MAX bytes.
   0, nothing done, for one vector, one row, or rows too long. */
static inline DOTN_TARGET __attribute__((always_inline)) int
DOTN(b32_multi_rows_t)(const int type, const void *w, size_t row_bytes,
                       const struct janas_block_q8k *x, size_t xs, size_t nv,
                       size_t nb, size_t rows, float *y, size_t ys)
{
    size_t g = nb ? B32XT_MAX / (B32XT * nb) : 0;
    if (g > 8)
        g = 8;
    if (g < 2 || nv < 2 || rows < 2)
        return 0;
    _Alignas(32) uint8_t xt[B32XT_MAX];
    for (size_t v0 = 0; v0 < nv; v0 += g) {
        int n = nv - v0 < g ? (int)(nv - v0) : (int)g;
        const struct janas_block_q8k *xv = x + v0 * xs;
        DOTN(b32_xt)(xv, xs, (size_t)n, nb, xt);
        for (size_t r = 0; r < rows; r++) {
            float o[8];
            DOTN(b32_xt_n)
            (type, (const uint8_t *)w + r * row_bytes, xv, xs, xt, nb, o, n);
            for (int i = 0; i < n; i++)
                y[r + (v0 + (size_t)i) * ys] = o[i];
        }
    }
    return 1;
}

static DOTN_TARGET int DOTN(b32_multi_rows)(int type, const void *w,
                                            size_t row_bytes,
                                            const struct janas_block_q8k *x,
                                            size_t xs, size_t nv, size_t nb,
                                            size_t rows, float *y, size_t ys)
{
    switch (type) {
#define B32(T)                                                                 \
    case T:                                                                    \
        return DOTN(b32_multi_rows_t)(T, w, row_bytes, x, xs, nv, nb, rows, y, \
                                      ys);
        B32(JANAS_Q8_0)
        B32(JANAS_IQ4_NL)
        B32(JANAS_Q4_0)
        B32(JANAS_MXFP4)
        B32(JANAS_Q5_0)
        B32(JANAS_Q4_1)
        B32(JANAS_Q5_1)
#undef B32
    }
    return 0;
}
#endif

/* The 64 two-bit values of a Q2_0 block in their order, 0-3: the four
   planes of qs (byte i of plane j is weight 4i + j) interleaved, eight bits
   then sixteen. lo holds weights 0-31, hi 32-63. */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q2_0_unpack)(const uint8_t *qs, __m256i *lo, __m256i *hi)
{
    const __m128i m3 = _mm_set1_epi8(3);
    __m128i v = _mm_loadu_si128((const __m128i *)qs);
    __m128i p0 = _mm_and_si128(v, m3);
    __m128i p1 = _mm_and_si128(_mm_srli_epi16(v, 2), m3);
    __m128i p2 = _mm_and_si128(_mm_srli_epi16(v, 4), m3);
    __m128i p3 = _mm_and_si128(_mm_srli_epi16(v, 6), m3);
    __m128i a = _mm_unpacklo_epi8(p0, p1), b = _mm_unpackhi_epi8(p0, p1);
    __m128i c = _mm_unpacklo_epi8(p2, p3), d = _mm_unpackhi_epi8(p2, p3);
    *lo = _mm256_set_m128i(_mm_unpackhi_epi16(a, c), _mm_unpacklo_epi16(a, c));
    *hi = _mm256_set_m128i(_mm_unpackhi_epi16(b, d), _mm_unpacklo_epi16(b, d));
}

/*
 * Q2_0 (see janas_q2_0_dot_ref in quant.c): the values unsigned for
 * maddubs, then the activations' sum taken off: the exact sum of (q - 1) x
 * a for each block of 32 activations, one lane each, a block's d in its two
 * lanes.
 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q2_0_dotn_body)(const struct janas_block_q2_0 *w,
                     const struct janas_block_q8k *x, size_t xs, size_t nb,
                     float *out, const int nv)
{
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) acc[v] =
        _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(w + 4 * b, 4 * sizeof(*w));
        const struct janas_block_q2_0 *wb = w + 4 * b;
        __m256 dk = _mm256_cvtph_ps(_mm_setr_epi16(
            (short)wb[0].d, (short)wb[0].d, (short)wb[1].d, (short)wb[1].d,
            (short)wb[2].d, (short)wb[2].d, (short)wb[3].d, (short)wb[3].d));
        __m256i q[8];
        _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++)
            DOTN(q2_0_unpack)(wb[k].qs, &q[2 * k], &q[2 * k + 1]);
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i s[8];
            _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs32 + 32 * k));
                s[k] = _mm256_madd_epi16(_mm256_maddubs_epi16(q[k], a), ones);
            }
            __m256i as8 = _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *)xb->bsums32), ones);
            __m256i is8 = _mm256_sub_epi32(DOTN(sum8x8)(s), as8);
            acc[v] =
                _mm256_fmadd_ps(_mm256_mul_ps(dk, _mm256_loadu_ps(xb->d32)),
                                _mm256_cvtepi32_ps(is8), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] =
        DOTN(hsum8_ps)(acc[v]);
}

static DOTN_TARGET void DOTN(q5_0_dotn)(const struct janas_block_q5_0 *w,
                                        const struct janas_block_q8k *x,
                                        size_t xs, int nv, size_t nb,
                                        float *out)
{
    switch (nv) {
    case 8:
        DOTN(q5_0_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(q5_0_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(q5_0_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(q5_0_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(q5_0_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(q5_0_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(q5_0_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(q5_0_dotn_body)(w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(q5_0_dot)(const struct janas_block_q5_0 *w,
                                        const struct janas_block_q8k *x,
                                        size_t nb)
{
    float r;
    DOTN(q5_0_dotn_body)(w, x, 0, nb, &r, 1);
    return r;
}

static DOTN_TARGET void DOTN(q2_0_dotn)(const struct janas_block_q2_0 *w,
                                        const struct janas_block_q8k *x,
                                        size_t xs, int nv, size_t nb,
                                        float *out)
{
    switch (nv) {
    case 8:
        DOTN(q2_0_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(q2_0_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(q2_0_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(q2_0_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(q2_0_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(q2_0_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(q2_0_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(q2_0_dotn_body)(w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(q2_0_dot)(const struct janas_block_q2_0 *w,
                                        const struct janas_block_q8k *x,
                                        size_t nb)
{
    float r;
    DOTN(q2_0_dotn_body)(w, x, 0, nb, &r, 1);
    return r;
}

/*
 * Q2_0 rows against vectors whose activations were reordered once for all
 * the rows: the experts of Qwen3.8-Flash-Next, on one vector while it
 * answers and on a few while it reads. Byte i of a block holds weights
 * 4i + j in its bits 2j; two variable shifts of the 16 bytes, read into
 * both halves of a register, give the bit planes 0 | 2 and 1 | 3, the 16
 * bytes of plane j being weights 4i + j, and the activations are laid out
 * the same way (q2_0_xperm64). In each half, int32 lanes 0-1 are weights
 * 0-31 of the block and lanes 2-3 weights 32-63, so two hadd and one add of
 * the halves leave the eight sums of 32 in order, where Q8_0's tree takes
 * six hadd. Sums of integers, exact in any order, then the float operations
 * of the scalar reference: the same bits. Unpacking each row in the order
 * of the activations held one vector to 7.7 GB/s on a core and 34 on
 * twelve threads, against Q4_K's 24 and 85; this way 19 and 69 (24 Oct
 * 2026).
 */
#define Q2_0_XP_MAX 128 /* groups of 256 reordered on the stack: 32 KB */

/* The 64 activations of a Q2_0 block as its planes: 64 bytes, planes 0 | 2
   then 1 | 3, 16 bytes each, plane j the activations 4i + j. */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q2_0_xperm64)(const int8_t *a, int8_t *out)
{
    /* in each 16 bytes, the four bytes j apart side by side: dword j of
       lane m is activations 16m + 4i + j, i = 0-3 */
    const __m256i g =
        _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                         0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m256i b0 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)a), g);
    __m256i b1 =
        _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(a + 32)), g);
    /* lane m of b0 / b1 is D(m, 0..3); gathered across the lanes: plane j
       is D(0, j), D(1, j), D(2, j), D(3, j) */
    __m256i e0 = _mm256_shuffle_epi32(_mm256_permute4x64_epi64(b0, 0xd8), 0xd8);
    __m256i e1 = _mm256_shuffle_epi32(_mm256_permute4x64_epi64(b1, 0xd8), 0xd8);
    _mm256_storeu_si256((__m256i *)out, _mm256_unpacklo_epi64(e0, e1));
    _mm256_storeu_si256((__m256i *)(out + 32), _mm256_unpackhi_epi64(e0, e1));
}

/* nv vectors reordered into xp, vector v's group b at xp + 256 (v nb + b). */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q2_0_xperm)(const struct janas_block_q8k *x, size_t xs, size_t nv,
                 size_t nb, int8_t *xp)
{
    for (size_t v = 0; v < nv; v++)
        for (size_t b = 0; b < nb; b++)
            _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++)
                DOTN(q2_0_xperm64)(x[v * xs + b].qs32 + 64 * k,
                                   xp + 256 * (v * nb + b) + 64 * k);
}

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q2_0_xp_body)(const struct janas_block_q2_0 *w,
                   const struct janas_block_q8k *x, size_t xs, const int8_t *xp,
                   size_t nb, float *out, const int nv)
{
    const __m256i m3 = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1),
                  c02 = _mm256_setr_epi64x(0, 0, 4, 4),
                  c13 = _mm256_setr_epi64x(2, 2, 6, 6);
    __m256 acc[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) acc[v] =
        _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(w + 4 * b, 4 * sizeof(*w));
        const struct janas_block_q2_0 *wb = w + 4 * b;
        uint64_t d4 = (uint64_t)wb[0].d | (uint64_t)wb[1].d << 16 |
                      (uint64_t)wb[2].d << 32 | (uint64_t)wb[3].d << 48;
        __m128i h = _mm_cvtsi64_si128((long long)d4);
        __m256 dk = _mm256_cvtph_ps(_mm_unpacklo_epi16(h, h));
        __m256i p02[4], p13[4];
        _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++)
        {
            __m256i q = _mm256_broadcastsi128_si256(
                _mm_loadu_si128((const __m128i *)wb[k].qs));
            p02[k] = _mm256_and_si256(_mm256_srlv_epi64(q, c02), m3);
            p13[k] = _mm256_and_si256(_mm256_srlv_epi64(q, c13), m3);
        }
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            const int8_t *a = xp + 256 * ((size_t)v * nb + b);
            __m256i s[4];
            _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++) s[k] = DOTN_US2(
                p02[k], _mm256_loadu_si256((const __m256i *)(a + 64 * k)),
                p13[k], _mm256_loadu_si256((const __m256i *)(a + 64 * k + 32)));
            __m256i h01 = _mm256_hadd_epi32(s[0], s[1]);
            __m256i h23 = _mm256_hadd_epi32(s[2], s[3]);
            __m256i t =
                _mm256_add_epi32(_mm256_permute2x128_si256(h01, h23, 0x20),
                                 _mm256_permute2x128_si256(h01, h23, 0x31));
            __m256i as8 = _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *)xb->bsums32), ones);
            acc[v] = _mm256_fmadd_ps(
                _mm256_mul_ps(dk, _mm256_loadu_ps(xb->d32)),
                _mm256_cvtepi32_ps(_mm256_sub_epi32(t, as8)), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] =
        DOTN(hsum8_ps)(acc[v]);
}

/* rows Q2_0 rows of row_bytes from w against nv vectors (vector v at x + v
   xs): y[r + v ys]. Up to eight vectors at a time, as many as their
   reordered activations fit in Q2_0_XP_MAX groups; with no room, or a
   single row, the kernels on activations as they are. */
static DOTN_TARGET void DOTN(q2_0_multi_rows)(const void *w, size_t row_bytes,
                                              const struct janas_block_q8k *x,
                                              size_t xs, size_t nv, size_t nb,
                                              size_t rows, float *y, size_t ys)
{
    size_t g = nb ? Q2_0_XP_MAX / nb : 0;
    if (g > 8)
        g = 8;
    if (g == 0 || rows < 2) {
        for (size_t r = 0; r < rows; r++) {
            const struct janas_block_q2_0 *wr =
                (const void *)((const char *)w + r * row_bytes);
            for (size_t v = 0; v < nv; v += 8) {
                int n = nv - v < 8 ? (int)(nv - v) : 8;
                float o[8];
                DOTN(q2_0_dotn)(wr, x + v * xs, xs, n, nb, o);
                for (int i = 0; i < n; i++)
                    y[r + (v + (size_t)i) * ys] = o[i];
            }
        }
        return;
    }
    _Alignas(32) int8_t xp[Q2_0_XP_MAX * 256];
    for (size_t v0 = 0; v0 < nv; v0 += g) {
        int n = nv - v0 < g ? (int)(nv - v0) : (int)g;
        const struct janas_block_q8k *xv = x + v0 * xs;
        DOTN(q2_0_xperm)(xv, xs, (size_t)n, nb, xp);
        for (size_t r = 0; r < rows; r++) {
            const struct janas_block_q2_0 *wr =
                (const void *)((const char *)w + r * row_bytes);
            float o[8];
            switch (n) {
            case 8:
                DOTN(q2_0_xp_body)(wr, xv, xs, xp, nb, o, 8);
                break;
            case 7:
                DOTN(q2_0_xp_body)(wr, xv, xs, xp, nb, o, 7);
                break;
            case 6:
                DOTN(q2_0_xp_body)(wr, xv, xs, xp, nb, o, 6);
                break;
            case 5:
                DOTN(q2_0_xp_body)(wr, xv, xs, xp, nb, o, 5);
                break;
            case 4:
                DOTN(q2_0_xp_body)(wr, xv, xs, xp, nb, o, 4);
                break;
            case 3:
                DOTN(q2_0_xp_body)(wr, xv, xs, xp, nb, o, 3);
                break;
            case 2:
                DOTN(q2_0_xp_body)(wr, xv, xs, xp, nb, o, 2);
                break;
            default:
                DOTN(q2_0_xp_body)(wr, xv, xs, xp, nb, o, 1);
            }
            for (int i = 0; i < n; i++)
                y[r + (v0 + (size_t)i) * ys] = o[i];
        }
    }
}

/* one vector: the body above with nv = 1 let the kernel slip from 19 to
   16 GB/s on a core, its weights held through the loop over vectors */
static inline DOTN_TARGET __attribute__((always_inline)) float
DOTN(q2_0_dot_xp)(const struct janas_block_q2_0 *w,
                  const struct janas_block_q8k *x, const int8_t *xp, size_t nb)
{
    const __m256i m3 = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1),
                  c02 = _mm256_setr_epi64x(0, 0, 4, 4),
                  c13 = _mm256_setr_epi64x(2, 2, 6, 6);
    __m256 acc = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        DOTN(prefetch_ahead)(w + 4 * b, 4 * sizeof(*w));
        const struct janas_block_q2_0 *wb = w + 4 * b;
        uint64_t d4 = (uint64_t)wb[0].d | (uint64_t)wb[1].d << 16 |
                      (uint64_t)wb[2].d << 32 | (uint64_t)wb[3].d << 48;
        __m128i h = _mm_cvtsi64_si128((long long)d4);
        __m256 dk = _mm256_cvtph_ps(_mm_unpacklo_epi16(h, h));
        const int8_t *a = xp + 256 * b;
        __m256i s[4];
        _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++)
        {
            __m256i v = _mm256_broadcastsi128_si256(
                _mm_loadu_si128((const __m128i *)wb[k].qs));
            __m256i p02 = _mm256_and_si256(_mm256_srlv_epi64(v, c02), m3);
            __m256i p13 = _mm256_and_si256(_mm256_srlv_epi64(v, c13), m3);
            s[k] = DOTN_US2(
                p02, _mm256_loadu_si256((const __m256i *)(a + 64 * k)), p13,
                _mm256_loadu_si256((const __m256i *)(a + 64 * k + 32)));
        }
        __m256i h01 = _mm256_hadd_epi32(s[0], s[1]);
        __m256i h23 = _mm256_hadd_epi32(s[2], s[3]);
        __m256i t = _mm256_add_epi32(_mm256_permute2x128_si256(h01, h23, 0x20),
                                     _mm256_permute2x128_si256(h01, h23, 0x31));
        __m256i as8 = _mm256_madd_epi16(
            _mm256_loadu_si256((const __m256i *)x[b].bsums32), ones);
        __m256i is8 = _mm256_sub_epi32(t, as8);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(dk, _mm256_loadu_ps(x[b].d32)),
                              _mm256_cvtepi32_ps(is8), acc);
    }
    return DOTN(hsum8_ps)(acc);
}

static DOTN_TARGET void DOTN(q2_0_rows)(const void *w, size_t row_bytes,
                                        const struct janas_block_q8k *x,
                                        size_t nb, size_t rows, float *y)
{
    if (nb > Q2_0_XP_MAX || rows < 2) {
        DOTN(q2_0_multi_rows)(w, row_bytes, x, 0, 1, nb, rows, y, 0);
        return;
    }
    _Alignas(32) int8_t xp[Q2_0_XP_MAX * 256];
    DOTN(q2_0_xperm)(x, 0, 1, nb, xp);
    for (size_t r = 0; r < rows; r++)
        y[r] = DOTN(q2_0_dot_xp)(
            (const struct janas_block_q2_0 *)((const char *)w + r * row_bytes),
            x, xp, nb);
}

/* The 16 scales of a Q3_K block, -32..31, as q3k_scales() gives them,
   unpacked in a register: the low and high nibbles of the first eight
   bytes, and the two bits of the last four each group takes. */
static inline DOTN_TARGET __attribute__((always_inline)) __m128i
DOTN(q3k_scales16)(const uint8_t *q)
{
    uint64_t a01;
    uint32_t a2;
    memcpy(&a01, q, 8);
    memcpy(&a2, q + 8, 4);
    const __m128i m0f = _mm_set1_epi8(0x0f), m03 = _mm_set1_epi8(3);
    __m128i v = _mm_cvtsi64_si128((long long)a01);
    __m128i nib = _mm_unpacklo_epi64(_mm_and_si128(v, m0f),
                                     _mm_and_si128(_mm_srli_epi16(v, 4), m0f));
    __m128i top = _mm_and_si128(
        _mm_srlv_epi32(_mm_set1_epi32((int)a2), _mm_setr_epi32(0, 2, 4, 6)),
        m03);
    return _mm_sub_epi8(_mm_or_si128(nib, _mm_slli_epi16(top, 4)),
                        _mm_set1_epi8(32));
}

/*
 * Q3_K: per 128 weights the 32 bytes of qs and the 32 of hmask give four
 * registers of values u = low two bits | third bit << 2, 0-7, unsigned, so
 * maddubs takes them as they are; each register holds two groups, whose
 * scales multiply the pair sums (DOTN_ACC). The -4 of every value is taken
 * once per group from the Q8_K block sums. The integer sum is exact, then
 * one fma per block, as the reference.
 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q3k_dotn_body)(const struct janas_block_q3k *w,
                    const struct janas_block_q8k *x, size_t xs, size_t nb,
                    float *out, const int nv)
{
    const __m256i three = _mm256_set1_epi8(3), four = _mm256_set1_epi8(4);
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(&w[b], sizeof(w[b]));
        /* register k holds groups 2k (low lane) and 2k + 1 (high): the
           scales as int16, the even ones in the low lane and the odd ones in
           the high, then word k of each lane spread over it */
        __m128i s8 = DOTN(q3k_scales16)(w[b].scales);
        __m256i eo = _mm256_cvtepi8_epi16(
            _mm_shuffle_epi8(s8, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3,
                                               5, 7, 9, 11, 13, 15)));
        __m256i scall = _mm256_cvtepi8_epi16(s8);
        float d = _cvtsh_ss(w[b].d);
        __m256i hm = _mm256_loadu_si256((const __m256i *)w[b].hmask);
        __m256i qs[2] = {_mm256_loadu_si256((const __m256i *)w[b].qs),
                         _mm256_loadu_si256((const __m256i *)(w[b].qs + 32))};
        /* each register of weights and its scales made where it is used,
           for every vector at once: made all first, eight of each, they
           took all sixteen registers and went through the stack (more
           loads than Q4_K for half the bytes, 9 Oct 2026) */
        __m256i sum[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) sum[v] =
            _mm256_setzero_si256();
        _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
        {
            const int n = k / 4, j = k % 4;
            __m256i lo =
                _mm256_and_si256(_mm256_srli_epi16(qs[n], 2 * j), three);
            /* bit j + 4n of hmask moved to bit 2, within its byte */
            const int sh = j + 4 * n - 2;
            __m256i hs = sh > 0   ? _mm256_srli_epi16(hm, sh)
                         : sh < 0 ? _mm256_slli_epi16(hm, -sh)
                                  : hm;
            __m256i u = _mm256_or_si256(lo, _mm256_and_si256(hs, four));
            __m256i sc = _mm256_shuffle_epi8(
                eo, _mm256_set1_epi16((short)((2 * k + 1) << 8 | 2 * k)));
            _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
            {
                __m256i a = _mm256_loadu_si256(
                    (const __m256i *)(x[(size_t)v * xs + b].qs + 32 * k));
                sum[v] = DOTN_ACC(sum[v], _mm256_maddubs_epi16(u, a), sc);
            }
        }
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i corr = _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *)xb->bsums), scall);
            int32_t sumi = DOTN(hsum_epi32)(
                _mm256_sub_epi32(sum[v], _mm256_slli_epi32(corr, 2)));
            acc[v] = fmaf(d * xb->d, (float)sumi, acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = acc[v];
}

static DOTN_TARGET void DOTN(q3k_dotn)(const struct janas_block_q3k *w,
                                       const struct janas_block_q8k *x,
                                       size_t xs, int nv, size_t nb, float *out)
{
    switch (nv) {
    case 8:
        DOTN(q3k_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(q3k_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(q3k_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(q3k_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(q3k_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(q3k_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(q3k_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(q3k_dotn_body)(w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(q3k_dot)(const struct janas_block_q3k *w,
                                       const struct janas_block_q8k *x,
                                       size_t nb)
{
    float r;
    DOTN(q3k_dotn_body)(w, x, 0, nb, &r, 1);
    return r;
}

/*
 * IQ3_S: a group's eight grid entries (32 unsigned levels) come in one
 * gather; the signs go to the activations (sign_epi8 with -1 or +1), so
 * maddubs takes the levels as they are, and the group's scale multiplies
 * the pair sums (DOTN_ACC). The integer sum is exact, then one fma per
 * block, as the reference.
 */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(iq3s_dotn_body)(const struct janas_block_iq3s *w,
                     const struct janas_block_q8k *x, size_t xs, size_t nb,
                     float *out, const int nv)
{
    const __m256i shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i sel =
        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
                         2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ull);
    const __m256i one = _mm256_set1_epi8(1);
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(&w[b], sizeof(w[b]));
        __m256i lv[8], sg[8], sc[8];
        _Pragma("GCC unroll 8") for (int g = 0; g < 8; g++)
        {
            __m128i q8 = _mm_loadl_epi64((const __m128i *)(w[b].qs + 8 * g));
            __m256i nine = _mm256_slli_epi32(
                _mm256_and_si256(
                    _mm256_srlv_epi32(_mm256_set1_epi32(w[b].qh[g]), shifts),
                    _mm256_set1_epi32(1)),
                8);
            __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(q8), nine);
            lv[g] = _mm256_i32gather_epi32((const int *)iq3s_grid, idx, 4);
            uint32_t sb;
            memcpy(&sb, w[b].signs + 4 * g, 4);
            __m256i m = _mm256_shuffle_epi8(_mm256_set1_epi32((int)sb), sel);
            m = _mm256_cmpeq_epi8(_mm256_and_si256(m, bit), bit);
            sg[g] = _mm256_or_si256(m, one); /* -1 where the sign bit is set */
            sc[g] = _mm256_set1_epi16((short)iq3s_scale(&w[b], g));
        }
        float d = _cvtsh_ss(w[b].d);
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i sum = _mm256_setzero_si256();
            _Pragma("GCC unroll 8") for (int g = 0; g < 8; g++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * g));
                __m256i p =
                    _mm256_maddubs_epi16(lv[g], _mm256_sign_epi8(a, sg[g]));
                sum = DOTN_ACC(sum, p, sc[g]);
            }
            acc[v] = fmaf(d * xb->d, (float)DOTN(hsum_epi32)(sum), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = acc[v];
}

/* IQ2_XXS, IQ2_XS, IQ2_S: a group's four grid entries in a register, its
   four sign bytes as IQ3_S's, one scale (IQ2_XXS) or one for each half */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(iq2_dotn_body)(const int type, const void *w,
                    const struct janas_block_q8k *x, size_t xs, size_t nb,
                    float *out, const int nv)
{
    const size_t size = iq2_size(type);
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        const uint8_t *blk = (const uint8_t *)w + b * size;
        if (nv <= 2)
            DOTN(prefetch_ahead)(blk, size);
        /* each group's weights used at once by every vector, the vectors'
           integer sums in registers: the 24 registers of a block's
           unpacked weights, kept for the vectors, went to the stack and
           back (1.2 GB/s on a core, 7.6 on twelve threads: 24 Oct 2026) */
        __m256i sum[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) sum[v] =
            _mm256_setzero_si256();
        _Pragma("GCC unroll 8") for (int g = 0; g < 8; g++)
        {
            uint64_t gr[4], sm[4];
            int sl[4];
            __m256i sg;
            if (type == JANAS_IQ2_S) {
                /* signs bytes as they are: spread to the bytes in the
                   register, cheaper than four reads of iq_byte64 */
                uint32_t sb = 0;
                _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++)
                {
                    uint32_t s8;
                    iq2_sub(type, blk, 4 * g + k, &gr[k], &s8, &sl[k]);
                    sb |= s8 << 8 * k;
                }
                const __m256i sel = _mm256_setr_epi8(
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
                    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
                const __m256i bit =
                    _mm256_set1_epi64x((long long)0x8040201008040201ull);
                __m256i m =
                    _mm256_shuffle_epi8(_mm256_set1_epi32((int)sb), sel);
                m = _mm256_cmpeq_epi8(_mm256_and_si256(m, bit), bit);
                sg = _mm256_or_si256(m, _mm256_set1_epi8(1)); /* -1: set */
            } else {
                /* 1 or -1 a weight (iq_sign64): no parity to work out */
                _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++)
                    iq2_sub_mask(type, blk, 4 * g + k, &gr[k], &sm[k], &sl[k]);
                sg = _mm256_setr_epi64x((long long)sm[0], (long long)sm[1],
                                        (long long)sm[2], (long long)sm[3]);
            }
            __m256i lv = _mm256_setr_epi64x((long long)gr[0], (long long)gr[1],
                                            (long long)gr[2], (long long)gr[3]);
            /* the products' lanes 0-7 are weights 0-15, sub-groups 0 and 1 */
            __m256i sc = _mm256_set_m128i(_mm_set1_epi16((short)sl[2]),
                                          _mm_set1_epi16((short)sl[0]));
            _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
            {
                __m256i a = _mm256_loadu_si256(
                    (const __m256i *)(x[(size_t)v * xs + b].qs + 32 * g));
                __m256i p = _mm256_maddubs_epi16(lv, _mm256_sign_epi8(a, sg));
                sum[v] = DOTN_ACC(sum[v], p, sc);
            }
        }
        uint16_t h;
        memcpy(&h, blk, 2);
        float d = _cvtsh_ss(h) * 0.125f;
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            acc[v] = fmaf(d * xb->d, (float)DOTN(hsum_epi32)(sum[v]), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = acc[v];
}

static DOTN_TARGET void DOTN(iq2_dotn)(int type, const void *w,
                                       const struct janas_block_q8k *x,
                                       size_t xs, int nv, size_t nb, float *out)
{
    switch (nv) {
    case 8:
        if (type == JANAS_IQ2_XXS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XXS, w, x, xs, nb, out, 8);
        else if (type == JANAS_IQ2_XS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XS, w, x, xs, nb, out, 8);
        else
            DOTN(iq2_dotn_body)(JANAS_IQ2_S, w, x, xs, nb, out, 8);
        break;
    case 7:
        if (type == JANAS_IQ2_XXS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XXS, w, x, xs, nb, out, 7);
        else if (type == JANAS_IQ2_XS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XS, w, x, xs, nb, out, 7);
        else
            DOTN(iq2_dotn_body)(JANAS_IQ2_S, w, x, xs, nb, out, 7);
        break;
    case 6:
        if (type == JANAS_IQ2_XXS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XXS, w, x, xs, nb, out, 6);
        else if (type == JANAS_IQ2_XS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XS, w, x, xs, nb, out, 6);
        else
            DOTN(iq2_dotn_body)(JANAS_IQ2_S, w, x, xs, nb, out, 6);
        break;
    case 5:
        if (type == JANAS_IQ2_XXS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XXS, w, x, xs, nb, out, 5);
        else if (type == JANAS_IQ2_XS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XS, w, x, xs, nb, out, 5);
        else
            DOTN(iq2_dotn_body)(JANAS_IQ2_S, w, x, xs, nb, out, 5);
        break;
    case 4:
        if (type == JANAS_IQ2_XXS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XXS, w, x, xs, nb, out, 4);
        else if (type == JANAS_IQ2_XS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XS, w, x, xs, nb, out, 4);
        else
            DOTN(iq2_dotn_body)(JANAS_IQ2_S, w, x, xs, nb, out, 4);
        break;
    case 3:
        if (type == JANAS_IQ2_XXS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XXS, w, x, xs, nb, out, 3);
        else if (type == JANAS_IQ2_XS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XS, w, x, xs, nb, out, 3);
        else
            DOTN(iq2_dotn_body)(JANAS_IQ2_S, w, x, xs, nb, out, 3);
        break;
    case 2:
        if (type == JANAS_IQ2_XXS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XXS, w, x, xs, nb, out, 2);
        else if (type == JANAS_IQ2_XS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XS, w, x, xs, nb, out, 2);
        else
            DOTN(iq2_dotn_body)(JANAS_IQ2_S, w, x, xs, nb, out, 2);
        break;
    default:
        if (type == JANAS_IQ2_XXS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XXS, w, x, xs, nb, out, 1);
        else if (type == JANAS_IQ2_XS)
            DOTN(iq2_dotn_body)(JANAS_IQ2_XS, w, x, xs, nb, out, 1);
        else
            DOTN(iq2_dotn_body)(JANAS_IQ2_S, w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(iq2_dot)(int type, const void *w,
                                       const struct janas_block_q8k *x,
                                       size_t nb)
{
    float r;
    DOTN(iq2_dotn)(type, w, x, 0, 1, nb, &r);
    return r;
}

/* IQ1_M and IQ1_S (iq1s): a group's four grid entries as eight times
   their levels plus the shifts' signs, signed for maddubs as Q8_0's, one
   scale for each half (IQ1_S's two the same) */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(iq1_dotn_body)(const int iq1s, const void *w,
                    const struct janas_block_q8k *x, size_t xs, size_t nb,
                    float *out, const int nv)
{
    const struct janas_block_iq1m *wm = w;
    const struct janas_block_iq1s *ws = w;
    const __m256i top5 = _mm256_set1_epi8((char)0xf8);
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(iq1s ? (const void *)&ws[b]
                                      : (const void *)&wm[b],
                                 iq1s ? sizeof(ws[b]) : sizeof(wm[b]));
        __m256i lv[8], al[8], sc[8];
        _Pragma("GCC unroll 8") for (int g = 0; g < 8; g++)
        {
            uint64_t gr[4], sh[4];
            int sl[4];
            _Pragma("GCC unroll 4") for (int k = 0; k < 4; k++)
            {
                int neg;
                if (iq1s)
                    iq1s_sub(&ws[b], 4 * g + k, &gr[k], &neg, &sl[k]);
                else
                    iq1m_sub(&wm[b], 4 * g + k, &gr[k], &neg, &sl[k]);
                sh[k] = neg ? 0xffffffffffffffffull : 0x0101010101010101ull;
            }
            __m256i lev =
                _mm256_setr_epi64x((long long)gr[0], (long long)gr[1],
                                   (long long)gr[2], (long long)gr[3]);
            /* 8 level, byte by byte: a shift of the 16-bit lanes by three,
               the bits it brought from the neighbouring byte masked off */
            __m256i v8 = _mm256_and_si256(_mm256_slli_epi16(lev, 3), top5);
            lv[g] = _mm256_add_epi8(
                v8, _mm256_setr_epi64x((long long)sh[0], (long long)sh[1],
                                       (long long)sh[2], (long long)sh[3]));
            al[g] = _mm256_abs_epi8(lv[g]);
            sc[g] = _mm256_set_m128i(_mm_set1_epi16((short)sl[2]),
                                     _mm_set1_epi16((short)sl[0]));
        }
        float d = (_cvtsh_ss(iq1s ? ws[b].d : iq1m_d16(&wm[b]))) * 0.125f;
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i sum = _mm256_setzero_si256();
            _Pragma("GCC unroll 8") for (int g = 0; g < 8; g++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * g));
                __m256i p =
                    _mm256_maddubs_epi16(al[g], _mm256_sign_epi8(a, lv[g]));
                sum = DOTN_ACC(sum, p, sc[g]);
            }
            acc[v] = fmaf(d * xb->d, (float)DOTN(hsum_epi32)(sum), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = acc[v];
}

static DOTN_TARGET void DOTN(iq1m_dotn)(const struct janas_block_iq1m *w,
                                        const struct janas_block_q8k *x,
                                        size_t xs, int nv, size_t nb,
                                        float *out)
{
    switch (nv) {
    case 8:
        DOTN(iq1_dotn_body)(0, w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(iq1_dotn_body)(0, w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(iq1_dotn_body)(0, w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(iq1_dotn_body)(0, w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(iq1_dotn_body)(0, w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(iq1_dotn_body)(0, w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(iq1_dotn_body)(0, w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(iq1_dotn_body)(0, w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(iq1m_dot)(const struct janas_block_iq1m *w,
                                        const struct janas_block_q8k *x,
                                        size_t nb)
{
    float r;
    DOTN(iq1_dotn_body)(0, w, x, 0, nb, &r, 1);
    return r;
}

static DOTN_TARGET void DOTN(iq1s_dotn)(const struct janas_block_iq1s *w,
                                        const struct janas_block_q8k *x,
                                        size_t xs, int nv, size_t nb,
                                        float *out)
{
    switch (nv) {
    case 8:
        DOTN(iq1_dotn_body)(1, w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(iq1_dotn_body)(1, w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(iq1_dotn_body)(1, w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(iq1_dotn_body)(1, w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(iq1_dotn_body)(1, w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(iq1_dotn_body)(1, w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(iq1_dotn_body)(1, w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(iq1_dotn_body)(1, w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(iq1s_dot)(const struct janas_block_iq1s *w,
                                        const struct janas_block_q8k *x,
                                        size_t nb)
{
    float r;
    DOTN(iq1_dotn_body)(1, w, x, 0, nb, &r, 1);
    return r;
}

/* Q2_K: each half of the block's 32 bytes shifted by 0, 2, 4, 6 gives its
   weights in order, 32 at a time; the sixteen scales against the exact
   sums of each group of 16, the mins against the activations' sums, then
   Q4_K's two fmas */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q2k_dotn_body)(const struct janas_block_q2k *w,
                    const struct janas_block_q8k *x, size_t xs, size_t nb,
                    float *out, const int nv)
{
    const __m256i three = _mm256_set1_epi8(3);
    float acc[8] = {0}, accm[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(&w[b], sizeof(w[b]));
        __m256i q[8], sc[8];
        __m128i m8 = _mm_loadu_si128((const __m128i *)w[b].scales);
        /* the sixteen scales in both lanes; sub-group k's two (weights
           32k..32k+15 and the next 16) spread over a lane each, as int16,
           by one shuffle: sixteen scalar loads and broadcasts before */
        __m256i s16 =
            _mm256_broadcastsi128_si256(_mm_and_si128(m8, _mm_set1_epi8(15)));
        _Pragma("GCC unroll 2") for (int n = 0; n < 2; n++)
        {
            __m256i bytes =
                _mm256_loadu_si256((const __m256i *)(w[b].qs + 32 * n));
            _Pragma("GCC unroll 4") for (int j = 0; j < 4; j++)
            {
                q[4 * n + j] = _mm256_and_si256(
                    _mm256_srli_epi16(bytes, (int)(2 * j)), three);
                const int g = 8 * n + 2 * j;
                sc[4 * n + j] = _mm256_shuffle_epi8(
                    s16,
                    _mm256_setr_epi16(
                        (short)(0x8000 | g), (short)(0x8000 | g),
                        (short)(0x8000 | g), (short)(0x8000 | g),
                        (short)(0x8000 | g), (short)(0x8000 | g),
                        (short)(0x8000 | g), (short)(0x8000 | g),
                        (short)(0x8000 | (g + 1)), (short)(0x8000 | (g + 1)),
                        (short)(0x8000 | (g + 1)), (short)(0x8000 | (g + 1)),
                        (short)(0x8000 | (g + 1)), (short)(0x8000 | (g + 1)),
                        (short)(0x8000 | (g + 1)), (short)(0x8000 | (g + 1))));
            }
        }
        __m256i mins = _mm256_cvtepu8_epi16(
            _mm_srli_epi16(_mm_and_si128(m8, _mm_set1_epi8((char)0xf0)), 4));
        float d = _cvtsh_ss(w[b].d), dmin = _cvtsh_ss(w[b].dmin);
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i sum = _mm256_setzero_si256();
            _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * k));
                sum = DOTN_ACC(sum, _mm256_maddubs_epi16(q[k], a), sc[k]);
            }
            __m256i mn = _mm256_madd_epi16(
                mins, _mm256_loadu_si256((const __m256i *)xb->bsums));
            int is, im;
            DOTN(hsum2_epi32)(sum, mn, &is, &im);
            acc[v] = fmaf(d * xb->d, (float)is, acc[v]);
            accm[v] = fmaf(dmin * xb->d, (float)im, accm[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] =
        acc[v] - accm[v];
}

static DOTN_TARGET void DOTN(q2k_dotn)(const struct janas_block_q2k *w,
                                       const struct janas_block_q8k *x,
                                       size_t xs, int nv, size_t nb, float *out)
{
    switch (nv) {
    case 8:
        DOTN(q2k_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(q2k_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(q2k_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(q2k_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(q2k_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(q2k_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(q2k_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(q2k_dotn_body)(w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(q2k_dot)(const struct janas_block_q2k *w,
                                       const struct janas_block_q8k *x,
                                       size_t nb)
{
    float r;
    DOTN(q2k_dotn_body)(w, x, 0, nb, &r, 1);
    return r;
}

/* IQ3_XXS: IQ3_S's products, its indices of eight bits and its signs and
   scales in a word per group */
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(iq3xxs_dotn_body)(const struct janas_block_iq3xxs *w,
                       const struct janas_block_q8k *x, size_t xs, size_t nb,
                       float *out, const int nv)
{
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            DOTN(prefetch_ahead)(&w[b], sizeof(w[b]));
        __m256i lv[8], sg[8], sc[8];
        _Pragma("GCC unroll 8") for (int g = 0; g < 8; g++)
        {
            __m128i q8 = _mm_loadl_epi64((const __m128i *)(w[b].qs + 8 * g));
            lv[g] = _mm256_i32gather_epi32((const int *)iq3xxs_grid,
                                           _mm256_cvtepu8_epi32(q8), 4);
            uint32_t aux = iq3xxs_aux(&w[b], g);
            sg[g] = _mm256_setr_epi64x((long long)iq_sign64[aux & 127],
                                       (long long)iq_sign64[aux >> 7 & 127],
                                       (long long)iq_sign64[aux >> 14 & 127],
                                       (long long)iq_sign64[aux >> 21 & 127]);
            sc[g] = _mm256_set1_epi16((short)(1 + 2 * (aux >> 28)));
        }
        float d = _cvtsh_ss(w[b].d) * 0.25f;
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i sum = _mm256_setzero_si256();
            _Pragma("GCC unroll 8") for (int g = 0; g < 8; g++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * g));
                __m256i p =
                    _mm256_maddubs_epi16(lv[g], _mm256_sign_epi8(a, sg[g]));
                sum = DOTN_ACC(sum, p, sc[g]);
            }
            acc[v] = fmaf(d * xb->d, (float)DOTN(hsum_epi32)(sum), acc[v]);
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = acc[v];
}

static DOTN_TARGET void DOTN(iq3xxs_dotn)(const struct janas_block_iq3xxs *w,
                                          const struct janas_block_q8k *x,
                                          size_t xs, int nv, size_t nb,
                                          float *out)
{
    switch (nv) {
    case 8:
        DOTN(iq3xxs_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(iq3xxs_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(iq3xxs_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(iq3xxs_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(iq3xxs_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(iq3xxs_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(iq3xxs_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(iq3xxs_dotn_body)(w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(iq3xxs_dot)(const struct janas_block_iq3xxs *w,
                                          const struct janas_block_q8k *x,
                                          size_t nb)
{
    float r;
    DOTN(iq3xxs_dotn_body)(w, x, 0, nb, &r, 1);
    return r;
}

static DOTN_TARGET void DOTN(iq3s_dotn)(const struct janas_block_iq3s *w,
                                        const struct janas_block_q8k *x,
                                        size_t xs, int nv, size_t nb,
                                        float *out)
{
    switch (nv) {
    case 8:
        DOTN(iq3s_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(iq3s_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(iq3s_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(iq3s_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(iq3s_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(iq3s_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(iq3s_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(iq3s_dotn_body)(w, x, xs, nb, out, 1);
    }
}

static DOTN_TARGET float DOTN(iq3s_dot)(const struct janas_block_iq3s *w,
                                        const struct janas_block_q8k *x,
                                        size_t nb)
{
    float r;
    DOTN(iq3s_dotn_body)(w, x, 0, nb, &r, 1);
    return r;
}

/*
 * Rows after rows of one type, one vector: the loop of a matrix-vector
 * product here, beside the kernels, so that a row costs one direct call.
 * Called row by row through janas_dot, each row chose its type, went
 * through a wrapper and a function pointer, and matvec.c worked out its
 * task's sizes again: 6% of the cycles of a token of Qwen3-30B-A3B (7 Oct
 * 2026), whose rows are short. Each row is the same function as before, so
 * the same bits.
 */
#define DOTN_ROWS(name, T)                                                     \
    static DOTN_TARGET void DOTN(name##_rows)(                                 \
        const void *w, size_t row_bytes, const struct janas_block_q8k *x,      \
        size_t nb, size_t rows, float *y)                                      \
    {                                                                          \
        for (size_t r = 0; r < rows; r++)                                      \
            y[r] = DOTN(name##_dot)(                                           \
                (const T *)((const char *)w + r * row_bytes), x, nb);          \
    }
DOTN_ROWS(q4k, struct janas_block_q4k)
DOTN_ROWS(q6k, struct janas_block_q6k)
DOTN_ROWS(q5k, struct janas_block_q5k)
DOTN_ROWS(q8_0, struct janas_block_q8_0)
DOTN_ROWS(iq4nl, struct janas_block_iq4nl)
DOTN_ROWS(mxfp4, struct janas_block_mxfp4)
DOTN_ROWS(iq4xs, struct janas_block_iq4xs)
DOTN_ROWS(q4_0, struct janas_block_q4_0)
DOTN_ROWS(q5_1, struct janas_block_q5_1)
DOTN_ROWS(q4_1, struct janas_block_q4_1)
DOTN_ROWS(q5_0, struct janas_block_q5_0)
DOTN_ROWS(iq3xxs, struct janas_block_iq3xxs)
DOTN_ROWS(iq1m, struct janas_block_iq1m)
DOTN_ROWS(iq1s, struct janas_block_iq1s)
DOTN_ROWS(q2k, struct janas_block_q2k)
DOTN_ROWS(q3k, struct janas_block_q3k)
DOTN_ROWS(iq3s, struct janas_block_iq3s)
#undef DOTN_ROWS
