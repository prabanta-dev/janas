/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri, 2026 ohora23 */
/*
 * quant_dot512.h - the quantized dot products on 512-bit registers
 * (private to quant.c, included once, after quant_dotn.h's AVX-VNNI pass).
 *
 * The same scheme as quant_dotn.h, with the inner loop over 128 weights
 * at a time instead of 64: one zmm holds the nibbles of two 64-weight
 * pieces, and vpmaddubsw + vpdpwssd on zmm do in one instruction what
 * took two on ymm. The integer sums of a block are exact in either width,
 * and the float operations after them are the ones of quant_dotn.h, in
 * the same order: the results are bit-identical to the AVX2, AVX-VNNI,
 * scalar and GPU kernels (test_quant checks it).
 *
 * Needs AVX-512 F, BW, VL and VNNI (Zen 4 and later, Ice Lake and later).
 * The product of one vector is bound by memory on every core count that
 * matters, so what this buys is the multi-vector products of a prompt
 * (bench_kernels' Nx4 and Nx8 columns), where the AVX-VNNI kernels sit
 * below the memory's bandwidth.
 *
 * Q4_K, Q5_K, Q6_K and Q8_0 so far; the other types fall back to the
 * AVX-VNNI kernels.
 */

#define DOT512_TARGET                                                          \
    __attribute__((                                                            \
        target("avx2,fma,f16c,avxvnni,avx512f,avx512bw,avx512vl,avx512vnni")))

/* The low ymm plus the high ymm of v (exact). */
static inline DOT512_TARGET __attribute__((always_inline)) __m256i
fold512_epi32(__m512i v)
{
    return _mm256_add_epi32(_mm512_castsi512_si256(v),
                            _mm512_extracti64x4_epi64(v, 1));
}

/*
 * The sums of eight zmm, one per vector, each over its sixteen lanes, in
 * one ymm (exact): a tree of unpacks and adds, the 128-bit lanes last.
 * 24 instructions where folding each to a ymm and vphaddd took about 35.
 */
static inline DOT512_TARGET __attribute__((always_inline)) __m256i
sum8x16_epi32(const __m512i s[8])
{
    __m512i u01 = _mm512_add_epi32(_mm512_unpacklo_epi32(s[0], s[1]),
                                   _mm512_unpackhi_epi32(s[0], s[1]));
    __m512i u23 = _mm512_add_epi32(_mm512_unpacklo_epi32(s[2], s[3]),
                                   _mm512_unpackhi_epi32(s[2], s[3]));
    __m512i u45 = _mm512_add_epi32(_mm512_unpacklo_epi32(s[4], s[5]),
                                   _mm512_unpackhi_epi32(s[4], s[5]));
    __m512i u67 = _mm512_add_epi32(_mm512_unpacklo_epi32(s[6], s[7]),
                                   _mm512_unpackhi_epi32(s[6], s[7]));
    /* every 128-bit lane of v03: partial sums of s0 s1 s2 s3; v47: s4..s7 */
    __m512i v03 = _mm512_add_epi32(_mm512_unpacklo_epi64(u01, u23),
                                   _mm512_unpackhi_epi64(u01, u23));
    __m512i v47 = _mm512_add_epi32(_mm512_unpacklo_epi64(u45, u67),
                                   _mm512_unpackhi_epi64(u45, u67));
    /* t = [A, A', B, B'] with A + A' = s0..s3 and B + B' = s4..s7 */
    __m512i t = _mm512_add_epi32(_mm512_shuffle_i32x4(v03, v47, 0x44),
                                 _mm512_shuffle_i32x4(v03, v47, 0xee));
    __m512i r = _mm512_add_epi32(_mm512_shuffle_i32x4(t, t, 0x88),
                                 _mm512_shuffle_i32x4(t, t, 0xdd));
    return _mm512_castsi512_si256(r);
}

/*
 * Q4_K. A block's 128 bytes of nibbles are two zmm loads; in each, the
 * low nibbles are the weights of two 32-weight sub-blocks (2c and 2c+2)
 * and the high nibbles two more (2c+1 and 2c+3), where the activations
 * run in order. The 128-bit lanes are regrouped so that W0 = sub-blocks
 * 2c, 2c+1 (activations 64c .. 64c+63, contiguous) and W1 = sub-blocks
 * 2c+2, 2c+3 (the next 64): the shuffle is on the weights, once per
 * block, not on the activations, once per vector.
 */
static inline DOT512_TARGET __attribute__((always_inline)) void
q4k_dotn_body_avx512(const struct janas_block_q4k *w,
                     const struct janas_block_q8k *x, size_t xs, size_t nb,
                     float *out, const int nv)
{
    const __m512i low4 = _mm512_set1_epi8(0x0f);
    __m128 acc4 = _mm_setzero_ps(), acc4n = _mm_setzero_ps(),
           accm4n = _mm_setzero_ps();
    __m256 acc8 = _mm256_setzero_ps(), accm8 = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            prefetch_ahead_vnni(&w[b], sizeof(w[b]));
        uint32_t u[4];
        unpack_scales_u32(w[b].scales, u);
        __m128i sm = _mm_loadu_si128((const __m128i *)u);
        /* the eight scales as int16 in every 128-bit lane */
        __m512i sc16 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(sm));
        __m512i sumi[8], sumh[2];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) sumi[v] =
            _mm512_setzero_si512();
        const int nh = nv <= 2 ? nv : 0;
        _Pragma("GCC unroll 4") for (int v = 0; v < nh; v++) sumh[v] =
            _mm512_setzero_si512();
        _Pragma("GCC unroll 2") for (int c = 0; c < 2; c++)
        {
            /* the bytes of pieces 2c and 2c+1 */
            __m512i q4 = _mm512_loadu_si512((const void *)(w[b].qs + 64 * c));
            __m512i lo = _mm512_and_si512(q4, low4);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(q4, 4), low4);
            /* lo = [sub 4c, sub 4c+2], hi = [sub 4c+1, sub 4c+3] (32 bytes
               each); regrouped into W0 = [4c, 4c+1], W1 = [4c+2, 4c+3] */
            __m512i w0 = _mm512_shuffle_i64x2(lo, hi, 0x44);
            __m512i w1 = _mm512_shuffle_i64x2(lo, hi, 0xee);
            /* the scale of each half, as int16, built in the 128-bit lanes
               by pshufb as quant_dotn.h does: lanes 0-1 one sub-block,
               lanes 2-3 the next */
            const int sb = 4 * c; /* first sub-block of this half */
#define SCALE_IDX(k) ((short)(((2 * (k) + 1) << 8) | (2 * (k))))
            __m512i i0 = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi16(SCALE_IDX(sb))),
                _mm256_set1_epi16(SCALE_IDX(sb + 1)), 1);
            __m512i i1 = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi16(SCALE_IDX(sb + 2))),
                _mm256_set1_epi16(SCALE_IDX(sb + 3)), 1);
#undef SCALE_IDX
            __m512i s0 = _mm512_shuffle_epi8(sc16, i0);
            __m512i s1 = _mm512_shuffle_epi8(sc16, i1);
            _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
            {
                const int8_t *a = x[(size_t)v * xs + b].qs + 128 * c;
                __m512i a0 = _mm512_loadu_si512((const void *)a);
                __m512i a1 = _mm512_loadu_si512((const void *)(a + 64));
                sumi[v] = _mm512_dpwssd_epi32(sumi[v],
                                              _mm512_maddubs_epi16(w0, a0), s0);
                if (nv <= 2)
                    sumh[v] = _mm512_dpwssd_epi32(
                        sumh[v], _mm512_maddubs_epi16(w1, a1), s1);
                else
                    sumi[v] = _mm512_dpwssd_epi32(
                        sumi[v], _mm512_maddubs_epi16(w1, a1), s1);
            }
        }
        /* from here on quant_dotn.h's reduction, on the folded sums */
        float dw = _cvtsh_ss(w[b].d), dmw = _cvtsh_ss(w[b].dmin);
        __m128i mins16 = _mm_cvtepu8_epi16(_mm_srli_si128(sm, 8));
        __m128i pm[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++)
        {
            pm[v] = _mm_setzero_si128();
            if (v < nv) {
                const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
                pm[v] = _mm_madd_epi16(
                    mins16, _mm_loadu_si128((const __m128i *)xb->bsums2));
            }
        }
        if (nv <= 2) {
            __m256i z = _mm256_setzero_si256();
            __m256i a = _mm256_hadd_epi32(
                fold512_epi32(_mm512_add_epi32(sumi[0], sumh[0])),
                nv == 2 ? fold512_epi32(_mm512_add_epi32(sumi[1], sumh[1]))
                        : z);
            __m128i h = _mm_add_epi32(_mm256_castsi256_si128(a),
                                      _mm256_extracti128_si256(a, 1));
            __m128i t = _mm_hadd_epi32(h, _mm_hadd_epi32(pm[0], pm[1]));
            float d80 = x[b].d, d81 = nv == 2 ? x[xs + b].d : 0.0f;
            __m128 d8 = _mm_setr_ps(d80, d81, d80, d81);
            __m128 dd = _mm_setr_ps(dw, dw, dmw, dmw);
            acc4 = _mm_fmadd_ps(_mm_mul_ps(dd, d8), _mm_cvtepi32_ps(t), acc4);
        } else if (nv <= 4) {
            __m256i t =
                _mm256_hadd_epi32(_mm256_hadd_epi32(fold512_epi32(sumi[0]),
                                                    fold512_epi32(sumi[1])),
                                  _mm256_hadd_epi32(fold512_epi32(sumi[2]),
                                                    fold512_epi32(sumi[3])));
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
            __m256i is8 = sum8x16_epi32(sumi);
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

static DOT512_TARGET void q4k_dotn_avx512(const struct janas_block_q4k *w,
                                          const struct janas_block_q8k *x,
                                          size_t xs, int nv, size_t nb,
                                          float *out)
{
    switch (nv) {
    case 8:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 8);
        break;
    case 7:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 7);
        break;
    case 6:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 6);
        break;
    case 5:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 5);
        break;
    case 4:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 4);
        break;
    case 3:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 3);
        break;
    case 2:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 2);
        break;
    default:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 1);
    }
}

static DOT512_TARGET float q4k_dot_avx512(const struct janas_block_q4k *w,
                                          const struct janas_block_q8k *x,
                                          size_t nb)
{
    float r;
    q4k_dotn_body_avx512(w, x, 0, nb, &r, 1);
    return r;
}

static DOT512_TARGET void q4k_rows_avx512(const void *w, size_t row_bytes,
                                          const struct janas_block_q8k *x,
                                          size_t nb, size_t rows, float *y)
{
    for (size_t r = 0; r < rows; r++)
        y[r] = q4k_dot_avx512(
            (const struct janas_block_q4k *)((const char *)w + r * row_bytes),
            x, nb);
}

/* A zmm whose low ymm holds the int16 lo in every lane and the high ymm hi:
   per-half shift counts and scale indices. */
#define HALVES_EPI16(lo, hi)                                                   \
    _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(lo)),          \
                       _mm256_set1_epi16(hi), 1)

/*
 * Q5_K: the Q4_K kernel with the fifth bit, taken from qh as quant_dotn.h
 * does (bit s of qh[l] is the fifth bit of weight 32 s + l): the two
 * chunks of a step need different bit positions, so the shift count is a
 * vector, one count for each ymm half.
 */
static inline DOT512_TARGET __attribute__((always_inline)) void
q5k_dotn_body_avx512(const struct janas_block_q5k *w,
                     const struct janas_block_q8k *x, size_t xs, size_t nb,
                     float *out, const int nv)
{
    const __m512i low4 = _mm512_set1_epi8(0x0f);
    const __m512i one = _mm512_set1_epi8(1);
    /* step c holds chunks 2c (low ymm) and 2c+1 (high ymm): the low nibbles
       of chunk C take bit 2C of qh, the high ones bit 2C+1 */
    const __m512i shlo[2] = {HALVES_EPI16(0, 2), HALVES_EPI16(4, 6)};
    const __m512i shhi[2] = {HALVES_EPI16(1, 3), HALVES_EPI16(5, 7)};
    __m128 acc4 = _mm_setzero_ps();
    __m256 acc8 = _mm256_setzero_ps(), accm8 = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            prefetch_ahead_vnni(&w[b], sizeof(w[b]));
        uint32_t u[4];
        unpack_scales_u32(w[b].scales, u);
        __m128i sm = _mm_loadu_si128((const __m128i *)u);
        __m512i sc16 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(sm));
        __m512i qh2 = _mm512_broadcast_i64x4(
            _mm256_loadu_si256((const __m256i *)w[b].qh));
        __m512i sumi[8], sumh[2];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) sumi[v] =
            _mm512_setzero_si512();
        const int nh = nv <= 2 ? nv : 0;
        _Pragma("GCC unroll 4") for (int v = 0; v < nh; v++) sumh[v] =
            _mm512_setzero_si512();
        _Pragma("GCC unroll 2") for (int c = 0; c < 2; c++)
        {
            __m512i q4 = _mm512_loadu_si512((const void *)(w[b].qs + 64 * c));
            __m512i lo = _mm512_and_si512(q4, low4);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(q4, 4), low4);
            lo = _mm512_or_si512(
                lo,
                _mm512_slli_epi16(
                    _mm512_and_si512(_mm512_srlv_epi16(qh2, shlo[c]), one), 4));
            hi = _mm512_or_si512(
                hi,
                _mm512_slli_epi16(
                    _mm512_and_si512(_mm512_srlv_epi16(qh2, shhi[c]), one), 4));
            __m512i w0 = _mm512_shuffle_i64x2(lo, hi, 0x44);
            __m512i w1 = _mm512_shuffle_i64x2(lo, hi, 0xee);
            const int sb = 4 * c;
#define SCALE_IDX(k) ((short)(((2 * (k) + 1) << 8) | (2 * (k))))
            __m512i s0 = _mm512_shuffle_epi8(
                sc16, HALVES_EPI16(SCALE_IDX(sb), SCALE_IDX(sb + 1)));
            __m512i s1 = _mm512_shuffle_epi8(
                sc16, HALVES_EPI16(SCALE_IDX(sb + 2), SCALE_IDX(sb + 3)));
#undef SCALE_IDX
            _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
            {
                const int8_t *a = x[(size_t)v * xs + b].qs + 128 * c;
                __m512i a0 = _mm512_loadu_si512((const void *)a);
                __m512i a1 = _mm512_loadu_si512((const void *)(a + 64));
                sumi[v] = _mm512_dpwssd_epi32(sumi[v],
                                              _mm512_maddubs_epi16(w0, a0), s0);
                if (nv <= 2)
                    sumh[v] = _mm512_dpwssd_epi32(
                        sumh[v], _mm512_maddubs_epi16(w1, a1), s1);
                else
                    sumi[v] = _mm512_dpwssd_epi32(
                        sumi[v], _mm512_maddubs_epi16(w1, a1), s1);
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
                pm[v] = _mm_madd_epi16(
                    mins16, _mm_loadu_si128((const __m128i *)xb->bsums2));
            }
        }
        if (nv <= 2) {
            __m256i z = _mm256_setzero_si256();
            __m256i a = _mm256_hadd_epi32(
                fold512_epi32(_mm512_add_epi32(sumi[0], sumh[0])),
                nv == 2 ? fold512_epi32(_mm512_add_epi32(sumi[1], sumh[1]))
                        : z);
            __m128i h = _mm_add_epi32(_mm256_castsi256_si128(a),
                                      _mm256_extracti128_si256(a, 1));
            __m128i t = _mm_hadd_epi32(h, _mm_hadd_epi32(pm[0], pm[1]));
            float d80 = x[b].d, d81 = nv == 2 ? x[xs + b].d : 0.0f;
            __m128 d8 = _mm_setr_ps(d80, d81, d80, d81);
            __m128 dd = _mm_setr_ps(dw, dw, dmw, dmw);
            acc4 = _mm_fmadd_ps(_mm_mul_ps(dd, d8), _mm_cvtepi32_ps(t), acc4);
        } else {
            __m256i is8 = sum8x16_epi32(sumi);
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
 * Q6_K: half n of a block (128 weights) is one zmm of ql and the 32 bytes
 * of qh it needs, broadcast to both ymm halves and shifted by a count of
 * its own in each: q01 = weights 128n .. 128n+63, q23 the next 64, both
 * contiguous against the activations. One int8 scale per 16 weights is
 * one per 128-bit lane: vpermw spreads the block's sixteen over the lanes.
 * The -32 offset folds into the exact sum as in quant_dotn.h.
 */
static inline DOT512_TARGET __attribute__((always_inline)) void
q6k_dotn_body_avx512(const struct janas_block_q6k *w,
                     const struct janas_block_q8k *x, size_t xs, size_t nb,
                     float *out, const int nv)
{
    const __m512i low4 = _mm512_set1_epi8(0x0f);
    const __m512i low2 = _mm512_set1_epi8(0x03);
    const __m512i sh02 = HALVES_EPI16(0, 2), sh46 = HALVES_EPI16(4, 6);
    /* lane l -> scale l / 8: four scales over the four 128-bit lanes */
    const __m512i sidx =
        _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
                         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128 acc4 = _mm_setzero_ps();
    __m256 acc8 = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            prefetch_ahead_vnni(&w[b], sizeof(w[b]));
        const int8_t *sc = w[b].scales;
        __m256i sc16w =
            _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sc));
        __m512i sc16 = _mm512_castsi256_si512(sc16w);
        __m512i sumi[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) sumi[v] =
            _mm512_setzero_si512();
        _Pragma("GCC unroll 2") for (int n = 0; n < 2; n++)
        {
            __m512i ql = _mm512_loadu_si512((const void *)(w[b].ql + 64 * n));
            __m512i qh2 = _mm512_broadcast_i64x4(
                _mm256_loadu_si256((const __m256i *)(w[b].qh + 32 * n)));
            __m512i q01 = _mm512_or_si512(
                _mm512_and_si512(ql, low4),
                _mm512_slli_epi16(
                    _mm512_and_si512(_mm512_srlv_epi16(qh2, sh02), low2), 4));
            __m512i q23 = _mm512_or_si512(
                _mm512_and_si512(_mm512_srli_epi16(ql, 4), low4),
                _mm512_slli_epi16(
                    _mm512_and_si512(_mm512_srlv_epi16(qh2, sh46), low2), 4));
            __m512i s01 = _mm512_permutexvar_epi16(
                _mm512_add_epi16(sidx, _mm512_set1_epi16((short)(8 * n))),
                sc16);
            __m512i s23 = _mm512_permutexvar_epi16(
                _mm512_add_epi16(sidx, _mm512_set1_epi16((short)(8 * n + 4))),
                sc16);
            _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
            {
                const int8_t *a = x[(size_t)v * xs + b].qs + 128 * n;
                __m512i a0 = _mm512_loadu_si512((const void *)a);
                __m512i a1 = _mm512_loadu_si512((const void *)(a + 64));
                sumi[v] = _mm512_dpwssd_epi32(
                    sumi[v], _mm512_maddubs_epi16(q01, a0), s01);
                sumi[v] = _mm512_dpwssd_epi32(
                    sumi[v], _mm512_maddubs_epi16(q23, a1), s23);
            }
        }
        float dw = _cvtsh_ss(w[b].d);
        /* the -32 offset, 32 x scale x activation sums, off the sums in
           place (exact; the upper lanes of the zero-extended ymm are 0) */
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m256i bs = _mm256_loadu_si256((const __m256i *)xb->bsums);
            __m256i off = _mm256_slli_epi32(_mm256_madd_epi16(sc16w, bs), 5);
            sumi[v] = _mm512_sub_epi32(sumi[v], _mm512_zextsi256_si512(off));
        }
        if (nv <= 2) {
            __m256i a = _mm256_hadd_epi32(fold512_epi32(sumi[0]),
                                          fold512_epi32(sumi[1]));
            __m128i h = _mm_add_epi32(_mm256_castsi256_si128(a),
                                      _mm256_extracti128_si256(a, 1));
            __m128i t = _mm_hadd_epi32(h, h);
            float d80 = x[b].d, d81 = nv == 2 ? x[xs + b].d : 0.0f;
            __m128 d8 = _mm_setr_ps(d80, d81, 0.0f, 0.0f);
            acc4 = _mm_fmadd_ps(_mm_mul_ps(_mm_set1_ps(dw), d8),
                                _mm_cvtepi32_ps(t), acc4);
        } else {
            __m256i is8 = sum8x16_epi32(sumi);
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

/*
 * Q8_0: two 32-weight blocks in a zmm. AVX-512 has no vpsignb, so the
 * weights go unsigned instead, w + 128, and one vpdpbusd makes the four
 * products of a lane: w . a = (w + 128) . a - 128 sum(a), with sum(a) of
 * each block from bsums32. Exact, as before; one instruction a vector
 * and block pair where the ymm kernel had vpsignb, vpmaddubsw and vpmaddwd
 * a block.
 */
static inline DOT512_TARGET __attribute__((always_inline)) void
q8_0_dotn_body_avx512(const struct janas_block_q8_0 *w,
                      const struct janas_block_q8k *x, size_t xs, size_t nb,
                      float *out, const int nv)
{
    const __m512i x80 = _mm512_set1_epi8((char)0x80);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) acc[v] =
        _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            prefetch_ahead_vnni(w + 8 * b, 8 * sizeof(*w));
        const struct janas_block_q8_0 *wb = w + 8 * b;
        __m256 dk = _mm256_cvtph_ps(_mm_setr_epi16(
            (short)wb[0].d, (short)wb[1].d, (short)wb[2].d, (short)wb[3].d,
            (short)wb[4].d, (short)wb[5].d, (short)wb[6].d, (short)wb[7].d));
        __m512i wu[4];
        _Pragma("GCC unroll 4") for (int j = 0; j < 4; j++) wu[j] =
            _mm512_xor_si512(
                _mm512_inserti64x4(
                    _mm512_castsi256_si512(
                        _mm256_loadu_si256((const __m256i *)wb[2 * j].qs)),
                    _mm256_loadu_si256((const __m256i *)wb[2 * j + 1].qs), 1),
                x80);
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
        {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            __m512i p[4];
            _Pragma("GCC unroll 4") for (int j = 0; j < 4; j++) p[j] =
                _mm512_dpbusd_epi32(
                    _mm512_setzero_si512(), wu[j],
                    _mm512_loadu_si512((const void *)(xb->qs32 + 64 * j)));
            /* each block's sum in its lane: the ymm kernel's tree on the
               eight halves */
            __m256i s01 = _mm256_hadd_epi32(_mm512_castsi512_si256(p[0]),
                                            _mm512_extracti64x4_epi64(p[0], 1));
            __m256i s23 = _mm256_hadd_epi32(_mm512_castsi512_si256(p[1]),
                                            _mm512_extracti64x4_epi64(p[1], 1));
            __m256i s45 = _mm256_hadd_epi32(_mm512_castsi512_si256(p[2]),
                                            _mm512_extracti64x4_epi64(p[2], 1));
            __m256i s67 = _mm256_hadd_epi32(_mm512_castsi512_si256(p[3]),
                                            _mm512_extracti64x4_epi64(p[3], 1));
            __m256i t0 = _mm256_hadd_epi32(s01, s23);
            __m256i t1 = _mm256_hadd_epi32(s45, s67);
            __m256i is8 =
                _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20),
                                 _mm256_permute2x128_si256(t0, t1, 0x31));
            __m256i sa = _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *)xb->bsums32), ones);
            is8 = _mm256_sub_epi32(is8, _mm256_slli_epi32(sa, 7));
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

#undef HALVES_EPI16

#define DOT512_WRAP(name, T)                                                   \
    static DOT512_TARGET void name##_dotn_avx512(                              \
        const T *w, const struct janas_block_q8k *x, size_t xs, int nv,        \
        size_t nb, float *out)                                                 \
    {                                                                          \
        switch (nv) {                                                          \
        case 8:                                                                \
            name##_dotn_body_avx512(w, x, xs, nb, out, 8);                     \
            break;                                                             \
        case 7:                                                                \
            name##_dotn_body_avx512(w, x, xs, nb, out, 7);                     \
            break;                                                             \
        case 6:                                                                \
            name##_dotn_body_avx512(w, x, xs, nb, out, 6);                     \
            break;                                                             \
        case 5:                                                                \
            name##_dotn_body_avx512(w, x, xs, nb, out, 5);                     \
            break;                                                             \
        case 4:                                                                \
            name##_dotn_body_avx512(w, x, xs, nb, out, 4);                     \
            break;                                                             \
        case 3:                                                                \
            name##_dotn_body_avx512(w, x, xs, nb, out, 3);                     \
            break;                                                             \
        case 2:                                                                \
            name##_dotn_body_avx512(w, x, xs, nb, out, 2);                     \
            break;                                                             \
        default:                                                               \
            name##_dotn_body_avx512(w, x, xs, nb, out, 1);                     \
        }                                                                      \
    }                                                                          \
    static DOT512_TARGET float name##_dot_avx512(                              \
        const T *w, const struct janas_block_q8k *x, size_t nb)                \
    {                                                                          \
        float r;                                                               \
        name##_dotn_body_avx512(w, x, 0, nb, &r, 1);                           \
        return r;                                                              \
    }                                                                          \
    static DOT512_TARGET void name##_rows_avx512(                              \
        const void *w, size_t row_bytes, const struct janas_block_q8k *x,      \
        size_t nb, size_t rows, float *y)                                      \
    {                                                                          \
        for (size_t r = 0; r < rows; r++)                                      \
            y[r] = name##_dot_avx512(                                          \
                (const T *)((const char *)w + r * row_bytes), x, nb);          \
    }
DOT512_WRAP(q5k, struct janas_block_q5k)
DOT512_WRAP(q6k, struct janas_block_q6k)
DOT512_WRAP(q8_0, struct janas_block_q8_0)
#undef DOT512_WRAP

#undef DOT512_TARGET
