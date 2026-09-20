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

static inline DOTN_TARGET __attribute__((always_inline)) int
DOTN(hsum128_epi32)(__m128i s)
{
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    return _mm_cvtsi128_si32(s);
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
static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q4k_dotn_body)(const struct janas_block_q4k *w,
                    const struct janas_block_q8k *x, size_t xs, size_t nb,
                    float *out, const int nv)
{
    const __m256i low4 = _mm256_set1_epi8(0x0f);
    /* up to 2 vectors: lanes acc0, acc1, accm0, accm1 */
    __m128 acc4 = _mm_setzero_ps();
    __m256 acc8 = _mm256_setzero_ps(), accm8 = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
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
            if (v < nv) {
                const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
                __m128i bs_lo = _mm_loadu_si128((const __m128i *)xb->bsums);
                __m128i bs_hi =
                    _mm_loadu_si128((const __m128i *)(xb->bsums + 8));
                pm[v] = _mm_madd_epi16(mins16, _mm_hadd_epi16(bs_lo, bs_hi));
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
            __m128i m03 = _mm_hadd_epi32(_mm_hadd_epi32(pm[0], pm[1]),
                                         _mm_hadd_epi32(pm[2], pm[3]));
            __m128i m47 = _mm_hadd_epi32(_mm_hadd_epi32(pm[4], pm[5]),
                                         _mm_hadd_epi32(pm[6], pm[7]));
            __m256i im8 = _mm256_set_m128i(m47, m03);
#define D8(v) (v < nv ? x[(size_t)(v) * xs + b].d : 0.0f)
            __m256 d8 = _mm256_setr_ps(D8(0), D8(1), D8(2), D8(3), D8(4), D8(5),
                                       D8(6), D8(7));
#undef D8
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

static inline DOTN_TARGET __attribute__((always_inline)) void
DOTN(q5k_dotn_body)(const struct janas_block_q5k *w,
                    const struct janas_block_q8k *x, size_t xs, size_t nb,
                    float *out, const int nv)
{
    const __m256i low4 = _mm256_set1_epi8(0x0f);
    const __m256i one = _mm256_set1_epi8(1);
    /* up to 2 vectors: lanes acc0, acc1, accm0, accm1 */
    __m128 acc4 = _mm_setzero_ps();
    __m256 acc8 = _mm256_setzero_ps(), accm8 = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
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
            if (v < nv) {
                const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
                __m128i bs_lo = _mm_loadu_si128((const __m128i *)xb->bsums);
                __m128i bs_hi =
                    _mm_loadu_si128((const __m128i *)(xb->bsums + 8));
                pm[v] = _mm_madd_epi16(mins16, _mm_hadd_epi16(bs_lo, bs_hi));
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
            __m128i m03 = _mm_hadd_epi32(_mm_hadd_epi32(pm[0], pm[1]),
                                         _mm_hadd_epi32(pm[2], pm[3]));
            __m128i m47 = _mm_hadd_epi32(_mm_hadd_epi32(pm[4], pm[5]),
                                         _mm_hadd_epi32(pm[6], pm[7]));
            __m256i im8 = _mm256_set_m128i(m47, m03);
#define D8(v) (v < nv ? x[(size_t)(v) * xs + b].d : 0.0f)
            __m256 d8 = _mm256_setr_ps(D8(0), D8(1), D8(2), D8(3), D8(4), D8(5),
                                       D8(6), D8(7));
#undef D8
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
                    _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * k));
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
            acc[v] = _mm256_fmadd_ps(_mm256_mul_ps(dk, _mm256_set1_ps(xb->d)),
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
        for (int n = 0; n < 2; n++) {
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
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(q4k_dotn_body)(w, x, xs, nb, out, 1);
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
    DOTN(q4k_dotn_body)(w, x, 0, nb, &r, 1);
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
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 8);
        break;
    case 7:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 7);
        break;
    case 6:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 6);
        break;
    case 5:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 5);
        break;
    case 4:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 4);
        break;
    case 3:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 3);
        break;
    case 2:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 2);
        break;
    default:
        DOTN(q5k_dotn_body)(w, x, xs, nb, out, 1);
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

static DOTN_TARGET float DOTN(q5k_dot)(const struct janas_block_q5k *w,
                                       const struct janas_block_q8k *x,
                                       size_t nb)
{
    float r;
    DOTN(q5k_dotn_body)(w, x, 0, nb, &r, 1);
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
                    _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * k));
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
            acc[v] = _mm256_fmadd_ps(_mm256_mul_ps(dk, _mm256_set1_ps(xb->d)),
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
        __m256i q[8], aq[8], sc[8];
        _Pragma("GCC unroll 8") for (int g = 0; g < 8; g++)
        {
            q[g] = DOTN(iq4_unpack32)(w[b].qs + 16 * g, levels);
            aq[g] = _mm256_abs_epi8(q[g]);
            sc[g] = _mm256_set1_epi16((short)iq4xs_scale(&w[b], g));
        }
        float d = janas_fp16_to_fp32(w[b].d);
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
        for (int n = 0; n < 2; n++) {
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
                    _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * k));
                s[k] = _mm256_madd_epi16(_mm256_maddubs_epi16(q[k], a), ones);
            }
            __m256i is8 = DOTN(sum8x8)(s);
            /* bsums 2k and 2k + 1 are block k's 32 activations */
            __m256i as8 = _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *)xb->bsums), ones);
            __m256 dx = _mm256_set1_ps(xb->d);
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
    const __m256i three = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        int8_t sc[16];
        q3k_scales(w[b].scales, sc);
        __m256i u[8], sc16[8];
        __m256i hm = _mm256_loadu_si256((const __m256i *)w[b].hmask);
        _Pragma("GCC unroll 2") for (int n = 0; n < 2; n++)
        {
            __m256i qs =
                _mm256_loadu_si256((const __m256i *)(w[b].qs + 32 * n));
            _Pragma("GCC unroll 4") for (int j = 0; j < 4; j++)
            {
                __m256i lo =
                    _mm256_and_si256(_mm256_srli_epi16(qs, 2 * j), three);
                __m256i hi =
                    _mm256_and_si256(_mm256_srli_epi16(hm, j + 4 * n), one);
                u[4 * n + j] = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 2));
            }
        }
        _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++) sc16[k] =
            _mm256_setr_m128i(_mm_set1_epi16(sc[2 * k]),
                              _mm_set1_epi16(sc[2 * k + 1]));
        __m256i scall =
            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sc));
        float d = janas_fp16_to_fp32(w[b].d);
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i sum = _mm256_setzero_si256();
            _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++)
            {
                __m256i a =
                    _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * k));
                sum = DOTN_ACC(sum, _mm256_maddubs_epi16(u[k], a), sc16[k]);
            }
            __m256i corr = _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *)xb->bsums), scall);
            int32_t sumi = DOTN(hsum_epi32)(sum) - 4 * DOTN(hsum_epi32)(corr);
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
        float d = janas_fp16_to_fp32(w[b].d);
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
