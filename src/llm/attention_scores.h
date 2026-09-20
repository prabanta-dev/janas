/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * attention_scores.h - the scores of one chunk against 1 to 8 query vectors
 * (private to attention.c, which includes it once per instruction set).
 *
 * The query is read as 16-bit integers with a scale of its own, the keys are
 * the eight-bit ones of the cache: the dot product is an exact integer sum,
 * and the float arithmetic that follows is one multiply per score. The
 * includer defines:
 *   SCORES(name)          the name with the instruction-set suffix
 *   SCORES_TARGET         the function attribute selecting the instructions
 *   SCORES_ACC(acc, a, b) acc plus the int16 pairs of a times b, summed
 *                         pairwise into int32 lanes: madd and add on AVX2, a
 *                         single vpdpwssd on AVX-VNNI
 * Integer sums are exact, so every int32 lane holds the same value with
 * either definition and every variant - scalar, AVX2, AVX-VNNI, one vector or
 * eight - gives bit-identical scores.
 */

static inline SCORES_TARGET __attribute__((always_inline)) void
SCORES(scores_body)(const int16_t *const *q, const int8_t *K, const float *ks,
                    uint32_t nt, uint32_t hd, const float *qsc, float scale,
                    float *const *sc, const int8_t *pf, const int ns)
{
    for (uint32_t t = 0; t < nt; t++) {
        const int8_t *k = K + (size_t)t * hd;
        /* the values pass reads V by columns, which the hardware prefetcher
           does not follow: bring its rows into L2 in order, now */
        if (pf)
            for (uint32_t i = 0; i < hd; i += 32)
                _mm_prefetch((const char *)(pf + (size_t)t * hd + i),
                             _MM_HINT_T1);
        __m256i acc[8];
        _Pragma("GCC unroll 8") for (int s = 0; s < ns; s++) acc[s] =
            _mm256_setzero_si256();
        for (uint32_t i = 0; i < hd; i += 16) {
            __m256i kv =
                _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(k + i)));
            _Pragma("GCC unroll 8") for (int s = 0; s < ns; s++) acc[s] =
                SCORES_ACC(acc[s],
                           _mm256_loadu_si256((const __m256i *)(q[s] + i)), kv);
        }
        if (ns == 8) {
            /* the eight tails summed together: seven adds across pairs and
               one across the halves, instead of eight reductions apiece.
               Integer sums are exact, so the lanes hold what the loop above
               would have given one by one, and the three multiplies that
               follow are the same three, lane by lane. */
            __m256i b0 = _mm256_hadd_epi32(acc[0], acc[1]);
            __m256i b1 = _mm256_hadd_epi32(acc[2], acc[3]);
            __m256i b2 = _mm256_hadd_epi32(acc[4], acc[5]);
            __m256i b3 = _mm256_hadd_epi32(acc[6], acc[7]);
            __m256i c0 = _mm256_hadd_epi32(b0, b1);
            __m256i c1 = _mm256_hadd_epi32(b2, b3);
            __m256i d =
                _mm256_add_epi32(_mm256_permute2x128_si256(c0, c1, 0x20),
                                 _mm256_permute2x128_si256(c0, c1, 0x31));
            __m256 f =
                _mm256_mul_ps(_mm256_cvtepi32_ps(d), _mm256_loadu_ps(qsc));
            f = _mm256_mul_ps(f, _mm256_set1_ps(scale));
            f = _mm256_mul_ps(f, _mm256_set1_ps(ks[t]));
            float out[8];
            _mm256_storeu_ps(out, f);
            _Pragma("GCC unroll 8") for (int s = 0; s < 8; s++) sc[s][t] =
                out[s];
        } else
            _Pragma("GCC unroll 8") for (int s = 0; s < ns; s++)
            {
                __m128i r = _mm_add_epi32(_mm256_castsi256_si128(acc[s]),
                                          _mm256_extracti128_si256(acc[s], 1));
                r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 0x4e));
                r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 0xb1));
                sc[s][t] = (float)_mm_cvtsi128_si32(r) * qsc[s] * scale * ks[t];
            }
    }
}

/* ns is a constant in each specialisation, so the slot loops unroll into
   registers (GCC keeps an array of accumulators in memory otherwise). */
static SCORES_TARGET void
SCORES(att_scores)(const int16_t *const *q, const int8_t *K, const float *ks,
                   uint32_t nt, uint32_t hd, const float *qsc, float scale,
                   float *const *sc, const int8_t *pf, int ns)
{
    switch (ns) {
    case 8:
        SCORES(scores_body)(q, K, ks, nt, hd, qsc, scale, sc, pf, 8);
        break;
    case 4:
        SCORES(scores_body)(q, K, ks, nt, hd, qsc, scale, sc, pf, 4);
        break;
    case 2:
        SCORES(scores_body)(q, K, ks, nt, hd, qsc, scale, sc, pf, 2);
        break;
    default:
        SCORES(scores_body)(q, K, ks, nt, hd, qsc, scale, sc, pf, 1);
    }
}
