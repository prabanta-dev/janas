/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Rendeverance */
/*
 * quant_neon_k.h - the remaining formats on ARM64 (Q4_K, Q5_K, Q3_K, IQ3_S,
 * IQ4_XS, MXFP4) and the batched Q4_1 / Q5_1, private to quant.c, included
 * after quant_neon.h. A block's 256 weights are unpacked to int8 (with NEON
 * where the layout allows, with the reference's own value functions where it
 * does not), then each group's exact int8 dot times its scale is summed in
 * one int32, then the reference's float steps follow in the same order, so
 * the results match the reference bit for bit.
 */

/* the exact sum of n (a multiple of 16) int8 products */
static inline NEON_DP int32_t neon_dotn_i8(const int8_t *w, const int8_t *a,
                                           int n)
{
    int32x4_t s = vdupq_n_s32(0);
    for (int i = 0; i < n; i += 16)
        s = vdotq_s32(s, vld1q_s8(w + i), vld1q_s8(a + i));
    return vaddvq_s32(s);
}

static NEON_DP void unpack_q4k(const struct janas_block_q4k *w, int8_t *q)
{
    const uint8x16_t m4 = vdupq_n_u8(15);
    for (int j = 0; j < 4; j++) { /* sub-blocks 2j (low nibbles), 2j+1 (high) */
        uint8x16_t a = vld1q_u8(w->qs + 32 * j);
        uint8x16_t b = vld1q_u8(w->qs + 32 * j + 16);
        vst1q_s8(q + 64 * j, vreinterpretq_s8_u8(vandq_u8(a, m4)));
        vst1q_s8(q + 64 * j + 16, vreinterpretq_s8_u8(vandq_u8(b, m4)));
        vst1q_s8(q + 64 * j + 32, vreinterpretq_s8_u8(vshrq_n_u8(a, 4)));
        vst1q_s8(q + 64 * j + 48, vreinterpretq_s8_u8(vshrq_n_u8(b, 4)));
    }
}

static NEON_DP void unpack_q5k(const struct janas_block_q5k *w, int8_t *q)
{
    const uint8x16_t m4 = vdupq_n_u8(15), one = vdupq_n_u8(1);
    uint8x16_t h0 = vld1q_u8(w->qh), h1 = vld1q_u8(w->qh + 16);
    for (int j = 0; j < 4; j++) {
        uint8x16_t a = vld1q_u8(w->qs + 32 * j);
        uint8x16_t b = vld1q_u8(w->qs + 32 * j + 16);
        /* bit s of qh[l] is the fifth bit of weight 32 s + l */
        int8x16_t r0 = vdupq_n_s8((int8_t)-(2 * j));
        int8x16_t r1 = vdupq_n_s8((int8_t)-(2 * j + 1));
        uint8x16_t ha0 = vandq_u8(vshlq_u8(h0, r0), one);
        uint8x16_t hb0 = vandq_u8(vshlq_u8(h1, r0), one);
        uint8x16_t ha1 = vandq_u8(vshlq_u8(h0, r1), one);
        uint8x16_t hb1 = vandq_u8(vshlq_u8(h1, r1), one);
        vst1q_s8(q + 64 * j, vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a, m4),
                                                          vshlq_n_u8(ha0, 4))));
        vst1q_s8(q + 64 * j + 16, vreinterpretq_s8_u8(vorrq_u8(
                                      vandq_u8(b, m4), vshlq_n_u8(hb0, 4))));
        vst1q_s8(q + 64 * j + 32, vreinterpretq_s8_u8(vorrq_u8(
                                      vshrq_n_u8(a, 4), vshlq_n_u8(ha1, 4))));
        vst1q_s8(q + 64 * j + 48, vreinterpretq_s8_u8(vorrq_u8(
                                      vshrq_n_u8(b, 4), vshlq_n_u8(hb1, 4))));
    }
}

/* Q4_K / Q5_K: groups of 32, unsigned 6-bit scales and mins */
static NEON_DP void neon_kmin_dotn(const void *wv, size_t bsz, int five,
                                   const struct janas_block_q8k *x, size_t xs,
                                   int nv, size_t nb, float *out)
{
    float acc[8] = {0}, accm[8] = {0};
    int8_t q[JANAS_QK];
    for (size_t b = 0; b < nb; b++) {
        const uint8_t *blk = (const uint8_t *)wv + b * bsz;
        uint16_t d, dmin;
        const uint8_t *scales;
        if (five) {
            const struct janas_block_q5k *w =
                (const struct janas_block_q5k *)blk;
            unpack_q5k(w, q);
            d = w->d;
            dmin = w->dmin;
            scales = w->scales;
        } else {
            const struct janas_block_q4k *w =
                (const struct janas_block_q4k *)blk;
            unpack_q4k(w, q);
            d = w->d;
            dmin = w->dmin;
            scales = w->scales;
        }
        uint8_t sc[8], m[8];
        unpack_scales(scales, sc, m);
        float fd = neon_h2f(d), fm = neon_h2f(dmin);
        /* the eight sub-block scales and minimums as int32 vectors */
        uint16x8_t sc16 = vmovl_u8(vld1_u8(sc)), m16 = vmovl_u8(vld1_u8(m));
        int32x4_t sc0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc16)));
        int32x4_t sc1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc16)));
        int32x4_t m0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(m16)));
        int32x4_t m1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(m16)));
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + v * xs + b;
            int32x4_t p[8];
            for (int k = 0; k < 8; k++) {
                const int8_t *wq = q + 32 * k, *aq = xb->qs + 32 * k;
                p[k] = vdotq_s32(
                    vdotq_s32(vdupq_n_s32(0), vld1q_s8(wq), vld1q_s8(aq)),
                    vld1q_s8(wq + 16), vld1q_s8(aq + 16));
            }
            /* exact: the same int32 sums as sub-block by sub-block */
            int32x4_t si = vmulq_s32(neon_red4(p[0], p[1], p[2], p[3]), sc0);
            si = vmlaq_s32(si, neon_red4(p[4], p[5], p[6], p[7]), sc1);
            /* each sub-block's activation sum: its two bsums */
            int16x8_t bs =
                vpaddq_s16(vld1q_s16(xb->bsums), vld1q_s16(xb->bsums + 8));
            int32x4_t sm = vmulq_s32(vmovl_s16(vget_low_s16(bs)), m0);
            sm = vmlaq_s32(sm, vmovl_s16(vget_high_s16(bs)), m1);
            acc[v] = fmaf(fd * xb->d, (float)vaddvq_s32(si), acc[v]);
            accm[v] = fmaf(fm * xb->d, (float)vaddvq_s32(sm), accm[v]);
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = acc[v] - accm[v];
}

/* Q3_K (groups of 16, signed scales), IQ3_S and IQ4_XS (groups of 32) */
static NEON_DP void neon_kgroup_dotn(int type, const void *wv,
                                     const struct janas_block_q8k *x, size_t xs,
                                     int nv, size_t nb, float *out)
{
    float acc[8] = {0};
    int8_t q[JANAS_QK];
    int scl[16];
    for (size_t b = 0; b < nb; b++) {
        int ng, gs;
        uint16_t d;
        if (type == JANAS_Q3_K) {
            const struct janas_block_q3k *w =
                (const struct janas_block_q3k *)wv + b;
            int8_t sc[16];
            q3k_scales(w->scales, sc);
            /* weight e = 128n + 32j + 16h + l: two bits at 2j of
               qs[32n + 16h + l], the third bit j + 4n of hmask[16h + l],
               the value q - 4 + 4 * bit (q3k_value), group 8n + 2j + h */
            const uint8x16_t m3 = vdupq_n_u8(3), b2 = vdupq_n_u8(4);
            for (int n = 0; n < 2; n++)
                for (int h = 0; h < 2; h++) {
                    uint8x16_t qs = vld1q_u8(w->qs + 32 * n + 16 * h);
                    uint8x16_t hm = vld1q_u8(w->hmask + 16 * h);
                    uint8x16_t two[4] = {qs, vshrq_n_u8(qs, 2),
                                         vshrq_n_u8(qs, 4), vshrq_n_u8(qs, 6)};
                    for (int j = 0; j < 4; j++) {
                        uint8x16_t bit = vtstq_u8(
                            hm, vdupq_n_u8((uint8_t)(1u << (j + 4 * n))));
                        uint8x16_t v =
                            vorrq_u8(vandq_u8(two[j], m3), vandq_u8(bit, b2));
                        vst1q_s8(q + 16 * (8 * n + 2 * j + h),
                                 vsubq_s8(vreinterpretq_s8_u8(v),
                                          vreinterpretq_s8_u8(b2)));
                    }
                }
            for (int g = 0; g < 16; g++)
                scl[g] = sc[g];
            ng = 16;
            gs = 16;
            d = w->d;
        } else if (type == JANAS_IQ3_S) {
            const struct janas_block_iq3s *w =
                (const struct janas_block_iq3s *)wv + b;
            for (int g = 0; g < 8; g++) {
                for (int e = 0; e < 32; e++)
                    q[32 * g + e] = (int8_t)iq3s_value(w, g, e);
                scl[g] = iq3s_scale(w, g);
            }
            ng = 8;
            gs = 32;
            d = w->d;
        } else { /* IQ4_XS */
            const struct janas_block_iq4xs *w =
                (const struct janas_block_iq4xs *)wv + b;
            int8x16_t levels = vld1q_s8(janas_iq4_values);
            for (int g = 0; g < 8; g++) {
                int8x16_t w0, w1;
                neon_nib_unpack(w->qs + 16 * g, levels, &w0, &w1);
                vst1q_s8(q + 32 * g, w0);
                vst1q_s8(q + 32 * g + 16, w1);
                scl[g] = iq4xs_scale(w, g);
            }
            ng = 8;
            gs = 32;
            d = w->d;
        }
        float fd = neon_h2f(d);
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + v * xs + b;
            /* each group's exact dot in four lanes, four groups reduced
               into one vector and times their scales: the same int32 */
            int32x4_t si = vdupq_n_s32(0);
            for (int g4 = 0; g4 < ng; g4 += 4) {
                int32x4_t p[4];
                for (int k = 0; k < 4; k++) {
                    const int8_t *wq = q + gs * (g4 + k);
                    const int8_t *aq = xb->qs + gs * (g4 + k);
                    p[k] =
                        vdotq_s32(vdupq_n_s32(0), vld1q_s8(wq), vld1q_s8(aq));
                    if (gs == 32)
                        p[k] = vdotq_s32(p[k], vld1q_s8(wq + 16),
                                         vld1q_s8(aq + 16));
                }
                si = vmlaq_s32(si, neon_red4(p[0], p[1], p[2], p[3]),
                               vld1q_s32(scl + g4));
            }
            acc[v] = fmaf(fd * xb->d, (float)vaddvq_s32(si), acc[v]);
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = acc[v];
}

/* MXFP4: IQ4_NL's body with each block's E8M0 power of two */
static NEON_DP void mxfp4_dotn_neon(const struct janas_block_mxfp4 *w,
                                    const struct janas_block_q8k *x, size_t xs,
                                    int nv, size_t nb, float *out)
{
    int8x16_t levels = vld1q_s8(janas_mxfp4_values);
    float32x4_t a0[8], a1[8];
    for (int v = 0; v < nv; v++)
        a0[v] = a1[v] = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        int8x16_t w0[8], w1[8];
        float e[8];
        for (int k = 0; k < 8; k++) {
            const struct janas_block_mxfp4 *wk = w + 8 * b + k;
            neon_nib_unpack(wk->qs, levels, &w0[k], &w1[k]);
            e[k] = janas_e8m0_half(wk->e);
        }
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + v * xs + b;
            int32_t dot[8];
            float sc[8];
            for (int k = 0; k < 8; k++) {
                dot[k] = neon_dot32(w0[k], w1[k], xb->qs32 + 32 * k);
                sc[k] = e[k] * xb->d32[k];
            }
            a0[v] =
                vfmaq_f32(a0[v], vld1q_f32(sc), vcvtq_f32_s32(vld1q_s32(dot)));
            a1[v] = vfmaq_f32(a1[v], vld1q_f32(sc + 4),
                              vcvtq_f32_s32(vld1q_s32(dot + 4)));
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = neon_sum8(a0[v], a1[v]);
}

/* Q4_1 / Q5_1 on nv vectors: the single kernel per vector */
static NEON_DP void neon_dm_dotn(const void *w, size_t size, int five,
                                 const struct janas_block_q8k *x, size_t xs,
                                 int nv, size_t nb, float *out)
{
    for (int v = 0; v < nv; v++)
        out[v] = neon_dm_dot((const uint8_t *)w, size, five, x + v * xs, nb);
}

/* single-vector entry points */
static NEON_DP float q4k_dot_neon(const struct janas_block_q4k *w,
                                  const struct janas_block_q8k *x, size_t nb)
{
    float r;
    neon_kmin_dotn(w, sizeof *w, 0, x, 0, 1, nb, &r);
    return r;
}

static NEON_DP float q5k_dot_neon(const struct janas_block_q5k *w,
                                  const struct janas_block_q8k *x, size_t nb)
{
    float r;
    neon_kmin_dotn(w, sizeof *w, 1, x, 0, 1, nb, &r);
    return r;
}

static NEON_DP float q3k_dot_neon(const struct janas_block_q3k *w,
                                  const struct janas_block_q8k *x, size_t nb)
{
    float r;
    neon_kgroup_dotn(JANAS_Q3_K, w, x, 0, 1, nb, &r);
    return r;
}

static NEON_DP float iq3s_dot_neon(const struct janas_block_iq3s *w,
                                   const struct janas_block_q8k *x, size_t nb)
{
    float r;
    neon_kgroup_dotn(JANAS_IQ3_S, w, x, 0, 1, nb, &r);
    return r;
}

static NEON_DP float iq4xs_dot_neon(const struct janas_block_iq4xs *w,
                                    const struct janas_block_q8k *x, size_t nb)
{
    float r;
    neon_kgroup_dotn(JANAS_IQ4_XS, w, x, 0, 1, nb, &r);
    return r;
}

static NEON_DP float mxfp4_dot_neon(const struct janas_block_mxfp4 *w,
                                    const struct janas_block_q8k *x, size_t nb)
{
    float r;
    mxfp4_dotn_neon(w, x, 0, 1, nb, &r);
    return r;
}
