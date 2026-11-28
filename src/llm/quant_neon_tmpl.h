/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * quant_neon_tmpl.h - the bodies of quant_neon.h's kernels, included once
 * for each way of making a block's integer sums: K(name) names a kernel,
 * KATTR its target, KSUM32 the sum of 32 products whose fours fit 16 bits,
 * KSUM32W the same without that bound, KSUM16 the sum of 16 products.
 */
KATTR static float K(q4k_dot)(const struct janas_block_q4k *w,
                              const struct janas_block_q8k *x, size_t nb)
{
    const uint8x16_t low4 = vdupq_n_u8(0x0f);
    float acc = 0.0f, accm = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        uint8_t sc[8], m[8];
        unpack_scales(w[b].scales, sc, m);
        int32_t sumi = 0, summ = 0;
        for (int s = 0; s < 8; s += 2) {
            /* 32 bytes hold sub-blocks s (low nibbles) and s + 1 (high) */
            const uint8_t *q = w[b].qs + (s / 2) * 32;
            uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
            const int8_t *a = x[b].qs + s * 32;
            int32_t lo = KSUM32(vreinterpretq_s8_u8(vandq_u8(q0, low4)),
                                vreinterpretq_s8_u8(vandq_u8(q1, low4)),
                                vld1q_s8(a), vld1q_s8(a + 16));
            int32_t hi = KSUM32(vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)),
                                vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)),
                                vld1q_s8(a + 32), vld1q_s8(a + 48));
            sumi += sc[s] * lo + sc[s + 1] * hi;
            summ += m[s] * (x[b].bsums[2 * s] + x[b].bsums[2 * s + 1]) +
                    m[s + 1] * (x[b].bsums[2 * s + 2] + x[b].bsums[2 * s + 3]);
        }
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
        accm = fmaf(janas_fp16_to_fp32(w[b].dmin) * x[b].d, (float)summ, accm);
    }
    return acc - accm;
}

KATTR static float K(q5k_dot)(const struct janas_block_q5k *w,
                              const struct janas_block_q8k *x, size_t nb)
{
    const uint8x16_t low4 = vdupq_n_u8(0x0f), one = vdupq_n_u8(1);
    float acc = 0.0f, accm = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        uint8_t sc[8], m[8];
        unpack_scales(w[b].scales, sc, m);
        uint8x16_t h0 = vld1q_u8(w[b].qh), h1 = vld1q_u8(w[b].qh + 16);
        int32_t sumi = 0, summ = 0;
        for (int s = 0; s < 8; s++) {
            const uint8_t *q = w[b].qs + (s / 2) * 32;
            uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
            if (s & 1)
                q0 = vshrq_n_u8(q0, 4), q1 = vshrq_n_u8(q1, 4);
            else
                q0 = vandq_u8(q0, low4), q1 = vandq_u8(q1, low4);
            /* bit s of qh[l] is the fifth bit of weight l */
            int8x16_t sh = vdupq_n_s8((int8_t)-s);
            q0 = vorrq_u8(q0, vshlq_n_u8(vandq_u8(vshlq_u8(h0, sh), one), 4));
            q1 = vorrq_u8(q1, vshlq_n_u8(vandq_u8(vshlq_u8(h1, sh), one), 4));
            const int8_t *a = x[b].qs + s * 32;
            int32_t dot =
                KSUM32(vreinterpretq_s8_u8(q0), vreinterpretq_s8_u8(q1),
                       vld1q_s8(a), vld1q_s8(a + 16));
            sumi += sc[s] * dot;
            summ += m[s] * (x[b].bsums[2 * s] + x[b].bsums[2 * s + 1]);
        }
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
        accm = fmaf(janas_fp16_to_fp32(w[b].dmin) * x[b].d, (float)summ, accm);
    }
    return acc - accm;
}

KATTR static float K(q6k_dot)(const struct janas_block_q6k *w,
                              const struct janas_block_q8k *x, size_t nb)
{
    const uint8x16_t low4 = vdupq_n_u8(0x0f), three = vdupq_n_u8(3);
    const int8x16_t bias = vdupq_n_s8(32);
    float acc = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        int32_t sumi = 0;
        for (int n = 0; n < 2; n++)
            for (int part = 0; part < 4; part++) {
                /* weights 128 n + 32 part .. + 31: groups 8 n + 2 part
                   and the next */
                const uint8_t *ql = w[b].ql + 64 * n + (part & 1) * 32;
                const uint8_t *qh = w[b].qh + 32 * n;
                for (int half = 0; half < 2; half++) {
                    uint8x16_t l = vld1q_u8(ql + 16 * half);
                    l = part < 2 ? vandq_u8(l, low4) : vshrq_n_u8(l, 4);
                    uint8x16_t h = vld1q_u8(qh + 16 * half);
                    h = vandq_u8(vshlq_u8(h, vdupq_n_s8((int8_t)(-2 * part))),
                                 three);
                    int8x16_t v = vsubq_s8(
                        vreinterpretq_s8_u8(vorrq_u8(l, vshlq_n_u8(h, 4))),
                        bias);
                    int g = 8 * n + 2 * part + half;
                    int32_t dot = KSUM16(v, vld1q_s8(x[b].qs + 16 * g));
                    sumi += w[b].scales[g] * dot;
                }
            }
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
    }
    return acc;
}

KATTR static float K(q8_0_dot)(const struct janas_block_q8_0 *w,
                               const struct janas_block_q8k *x, size_t nb)
{
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++)
        for (int k = 0; k < 8; k++) {
            const struct janas_block_q8_0 *wk = w + 8 * b + k;
            const int8_t *a = x[b].qs32 + 32 * k;
            int32_t dot = KSUM32W(vld1q_s8(wk->qs), vld1q_s8(wk->qs + 16),
                                  vld1q_s8(a), vld1q_s8(a + 16));
            acc[k] = fmaf(janas_fp16_to_fp32(wk->d) * x[b].d32[k], (float)dot,
                          acc[k]);
        }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) +
           ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

/* IQ4_NL and Q4_0: 4-bit indices into levels of signed bytes */
KATTR static float K(nib_dot)(const struct janas_block_iq4nl *w,
                              const struct janas_block_q8k *x, size_t nb,
                              const int8_t *levels)
{
    const int8x16_t lv = vld1q_s8(levels);
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++)
        for (int k = 0; k < 8; k++) {
            const struct janas_block_iq4nl *wk = w + 8 * b + k;
            uint8x16_t q = vld1q_u8(wk->qs);
            const int8_t *a = x[b].qs32 + 32 * k;
            int32_t dot = KSUM32W(vqtbl1q_s8(lv, vandq_u8(q, vdupq_n_u8(0x0f))),
                                  vqtbl1q_s8(lv, vshrq_n_u8(q, 4)), vld1q_s8(a),
                                  vld1q_s8(a + 16));
            acc[k] = fmaf(janas_fp16_to_fp32(wk->d) * x[b].d32[k], (float)dot,
                          acc[k]);
        }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) +
           ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

KATTR static float K(iq4nl_dot)(const struct janas_block_iq4nl *w,
                                const struct janas_block_q8k *x, size_t nb)
{
    return K(nib_dot)(w, x, nb, janas_iq4_values);
}

KATTR static float K(q4_0_dot)(const struct janas_block_q4_0 *w,
                               const struct janas_block_q8k *x, size_t nb)
{
    return K(nib_dot)((const struct janas_block_iq4nl *)w, x, nb,
                      janas_q4_0_values);
}

KATTR static float K(mxfp4_dot)(const struct janas_block_mxfp4 *w,
                                const struct janas_block_q8k *x, size_t nb)
{
    const int8x16_t lv = vld1q_s8(janas_mxfp4_values);
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++)
        for (int k = 0; k < 8; k++) {
            const struct janas_block_mxfp4 *wk = w + 8 * b + k;
            uint8x16_t q = vld1q_u8(wk->qs);
            const int8_t *a = x[b].qs32 + 32 * k;
            int32_t dot = KSUM32W(vqtbl1q_s8(lv, vandq_u8(q, vdupq_n_u8(0x0f))),
                                  vqtbl1q_s8(lv, vshrq_n_u8(q, 4)), vld1q_s8(a),
                                  vld1q_s8(a + 16));
            acc[k] =
                fmaf(janas_e8m0_half(wk->e) * x[b].d32[k], (float)dot, acc[k]);
        }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) +
           ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

KATTR static float K(iq4xs_dot)(const struct janas_block_iq4xs *w,
                                const struct janas_block_q8k *x, size_t nb)
{
    const int8x16_t lv = vld1q_s8(janas_iq4_values);
    float acc = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        int32_t sumi = 0;
        for (int g = 0; g < 8; g++) {
            uint8x16_t q = vld1q_u8(w[b].qs + 16 * g);
            const int8_t *a = x[b].qs + 32 * g;
            sumi += iq4xs_scale(&w[b], g) *
                    KSUM32W(vqtbl1q_s8(lv, vandq_u8(q, vdupq_n_u8(0x0f))),
                            vqtbl1q_s8(lv, vshrq_n_u8(q, 4)), vld1q_s8(a),
                            vld1q_s8(a + 16));
        }
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
    }
    return acc;
}

/* the fifth bits of a Q5 block's 32 weights, as 0 or 16 in their bytes:
   bit l of qh for weight l */
KATTR static inline void K(fifth_bits)(const uint8_t qh[4], uint8x16_t *lo,
                                       uint8x16_t *hi)
{
    static const uint8_t bit[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                    1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t m = vld1q_u8(bit), sixteen = vdupq_n_u8(16);
    uint8x16_t b0 = vcombine_u8(vdup_n_u8(qh[0]), vdup_n_u8(qh[1]));
    uint8x16_t b1 = vcombine_u8(vdup_n_u8(qh[2]), vdup_n_u8(qh[3]));
    *lo = vandq_u8(vtstq_u8(b0, m), sixteen);
    *hi = vandq_u8(vtstq_u8(b1, m), sixteen);
}

KATTR static float K(q5_0_dot)(const struct janas_block_q5_0 *w,
                               const struct janas_block_q8k *x, size_t nb)
{
    const uint8x16_t low4 = vdupq_n_u8(0x0f);
    const int8x16_t bias = vdupq_n_s8(16);
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++)
        for (int k = 0; k < 8; k++) {
            const struct janas_block_q5_0 *wk = w + 8 * b + k;
            uint8x16_t q = vld1q_u8(wk->qs), h0, h1;
            K(fifth_bits)(wk->qh, &h0, &h1);
            int8x16_t v0 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q, low4), h0)), bias);
            int8x16_t v1 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q, 4), h1)), bias);
            const int8_t *a = x[b].qs32 + 32 * k;
            int32_t dot = KSUM32W(v0, v1, vld1q_s8(a), vld1q_s8(a + 16));
            acc[k] = fmaf(janas_fp16_to_fp32(wk->d) * x[b].d32[k], (float)dot,
                          acc[k]);
        }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) +
           ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

/* Q4_1 (five 0) and Q5_1: d x the weights' exact sum, and m x the
   activations' (their block sums), each in its lanes, as the reference */
KATTR static float K(dm_dot)(const uint8_t *w, size_t size, int five,
                             const struct janas_block_q8k *x, size_t nb)
{
    const uint8x16_t low4 = vdupq_n_u8(0x0f);
    float acc[8] = {0}, accm[8] = {0};
    for (size_t b = 0; b < nb; b++)
        for (int k = 0; k < 8; k++) {
            const uint8_t *wk = w + (8 * b + k) * size;
            uint8x16_t q = vld1q_u8(wk + (five ? 8 : 4));
            uint8x16_t v0 = vandq_u8(q, low4), v1 = vshrq_n_u8(q, 4);
            if (five) {
                uint8x16_t h0, h1;
                K(fifth_bits)(wk + 4, &h0, &h1);
                v0 = vorrq_u8(v0, h0), v1 = vorrq_u8(v1, h1);
            }
            const int8_t *a = x[b].qs32 + 32 * k;
            int32_t dot =
                KSUM32W(vreinterpretq_s8_u8(v0), vreinterpretq_s8_u8(v1),
                        vld1q_s8(a), vld1q_s8(a + 16));
            int32_t asum = x[b].bsums32[2 * k] + x[b].bsums32[2 * k + 1];
            uint16_t d, m;
            memcpy(&d, wk, 2);
            memcpy(&m, wk + 2, 2);
            acc[k] =
                fmaf(janas_fp16_to_fp32(d) * x[b].d32[k], (float)dot, acc[k]);
            accm[k] =
                fmaf(janas_fp16_to_fp32(m) * x[b].d32[k], (float)asum, accm[k]);
        }
    return (((acc[0] + acc[4]) + (acc[2] + acc[6])) +
            ((acc[1] + acc[5]) + (acc[3] + acc[7]))) +
           (((accm[0] + accm[4]) + (accm[2] + accm[6])) +
            ((accm[1] + accm[5]) + (accm[3] + accm[7])));
}

KATTR static float K(q4_1_dot)(const struct janas_block_q4_1 *w,
                               const struct janas_block_q8k *x, size_t nb)
{
    return K(dm_dot)((const uint8_t *)w, sizeof(*w), 0, x, nb);
}

KATTR static float K(q5_1_dot)(const struct janas_block_q5_1 *w,
                               const struct janas_block_q8k *x, size_t nb)
{
    return K(dm_dot)((const uint8_t *)w, sizeof(*w), 1, x, nb);
}

/* Q2_K: group g's 16 weights are bits 2j..2j+1 of 16 bytes of qs, j =
   g % 8 / 2, the bytes 32 (g / 8) + 16 (g % 2) on */
KATTR static float K(q2k_dot)(const struct janas_block_q2k *w,
                              const struct janas_block_q8k *x, size_t nb)
{
    const uint8x16_t three = vdupq_n_u8(3);
    float acc = 0.0f, accm = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        int32_t sumi = 0, summ = 0;
        for (int g = 0; g < 16; g++) {
            int n = g / 8, j = g % 8 / 2;
            uint8x16_t q = vld1q_u8(w[b].qs + 32 * n + 16 * (g % 2));
            q = vandq_u8(vshlq_u8(q, vdupq_n_s8((int8_t)(-2 * j))), three);
            int32_t s =
                KSUM16(vreinterpretq_s8_u8(q), vld1q_s8(x[b].qs + 16 * g));
            sumi += (w[b].scales[g] & 15) * s;
            summ += (w[b].scales[g] >> 4) * x[b].bsums[g];
        }
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
        accm = fmaf(janas_fp16_to_fp32(w[b].dmin) * x[b].d, (float)summ, accm);
    }
    return acc - accm;
}

/* Q3_K: as Q2_K's two bits, and a third from hmask (bit j + 4 (g / 8) of
   its 16 bytes 16 (g % 2) on), the value q + 4 hi - 4 */
KATTR static float K(q3k_dot)(const struct janas_block_q3k *w,
                              const struct janas_block_q8k *x, size_t nb)
{
    const uint8x16_t three = vdupq_n_u8(3), one = vdupq_n_u8(1);
    const int8x16_t four = vdupq_n_s8(4);
    float acc = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        int8_t sc[16];
        q3k_scales(w[b].scales, sc);
        int32_t sumi = 0;
        for (int g = 0; g < 16; g++) {
            int n = g / 8, j = g % 8 / 2, h = g % 2;
            uint8x16_t q = vld1q_u8(w[b].qs + 32 * n + 16 * h);
            q = vandq_u8(vshlq_u8(q, vdupq_n_s8((int8_t)(-2 * j))), three);
            uint8x16_t hi = vld1q_u8(w[b].hmask + 16 * h);
            hi =
                vandq_u8(vshlq_u8(hi, vdupq_n_s8((int8_t)(-(j + 4 * n)))), one);
            int8x16_t v = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(q, vshlq_n_u8(hi, 2))), four);
            sumi += sc[g] * KSUM16(v, vld1q_s8(x[b].qs + 16 * g));
        }
        acc = fmaf(janas_fp16_to_fp32(w[b].d) * x[b].d, (float)sumi, acc);
    }
    return acc;
}

/*
 * The same products for nv vectors (1 to 8) at once: a row's block read and
 * its weights unpacked once, then each vector's sums and float operations
 * as the one-vector kernel makes them, in its order - each result that
 * kernel's to the bit. x[v * xs] is vector v's first block.
 */
KATTR static void K(q4k_dotn)(const struct janas_block_q4k *w,
                              const struct janas_block_q8k *x, size_t xs,
                              int nv, size_t nb, float *out)
{
    const uint8x16_t low4 = vdupq_n_u8(0x0f);
    float acc[8] = {0}, accm[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        uint8_t sc[8], m[8];
        unpack_scales(w[b].scales, sc, m);
        int8x16_t lo[8], hi[8];
        for (int s = 0; s < 8; s += 2) {
            const uint8_t *q = w[b].qs + (s / 2) * 32;
            uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
            lo[s] = vreinterpretq_s8_u8(vandq_u8(q0, low4));
            lo[s + 1] = vreinterpretq_s8_u8(vandq_u8(q1, low4));
            hi[s] = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            hi[s + 1] = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
        }
        float d = janas_fp16_to_fp32(w[b].d),
              dmin = janas_fp16_to_fp32(w[b].dmin);
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            int32_t sumi = 0, summ = 0;
            for (int s = 0; s < 8; s += 2) {
                const int8_t *a = xb->qs + s * 32;
                int32_t l =
                    KSUM32(lo[s], lo[s + 1], vld1q_s8(a), vld1q_s8(a + 16));
                int32_t h = KSUM32(hi[s], hi[s + 1], vld1q_s8(a + 32),
                                   vld1q_s8(a + 48));
                sumi += sc[s] * l + sc[s + 1] * h;
                summ +=
                    m[s] * (xb->bsums[2 * s] + xb->bsums[2 * s + 1]) +
                    m[s + 1] * (xb->bsums[2 * s + 2] + xb->bsums[2 * s + 3]);
            }
            acc[v] = fmaf(d * xb->d, (float)sumi, acc[v]);
            accm[v] = fmaf(dmin * xb->d, (float)summ, accm[v]);
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = acc[v] - accm[v];
}

KATTR static void K(q5k_dotn)(const struct janas_block_q5k *w,
                              const struct janas_block_q8k *x, size_t xs,
                              int nv, size_t nb, float *out)
{
    const uint8x16_t low4 = vdupq_n_u8(0x0f), one = vdupq_n_u8(1);
    float acc[8] = {0}, accm[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        uint8_t sc[8], m[8];
        unpack_scales(w[b].scales, sc, m);
        uint8x16_t h0 = vld1q_u8(w[b].qh), h1 = vld1q_u8(w[b].qh + 16);
        int8x16_t wq[8][2];
        for (int s = 0; s < 8; s++) {
            const uint8_t *q = w[b].qs + (s / 2) * 32;
            uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
            if (s & 1)
                q0 = vshrq_n_u8(q0, 4), q1 = vshrq_n_u8(q1, 4);
            else
                q0 = vandq_u8(q0, low4), q1 = vandq_u8(q1, low4);
            int8x16_t sh = vdupq_n_s8((int8_t)-s);
            q0 = vorrq_u8(q0, vshlq_n_u8(vandq_u8(vshlq_u8(h0, sh), one), 4));
            q1 = vorrq_u8(q1, vshlq_n_u8(vandq_u8(vshlq_u8(h1, sh), one), 4));
            wq[s][0] = vreinterpretq_s8_u8(q0);
            wq[s][1] = vreinterpretq_s8_u8(q1);
        }
        float d = janas_fp16_to_fp32(w[b].d),
              dmin = janas_fp16_to_fp32(w[b].dmin);
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            int32_t sumi = 0, summ = 0;
            for (int s = 0; s < 8; s++) {
                const int8_t *a = xb->qs + s * 32;
                int32_t dot =
                    KSUM32(wq[s][0], wq[s][1], vld1q_s8(a), vld1q_s8(a + 16));
                sumi += sc[s] * dot;
                summ += m[s] * (xb->bsums[2 * s] + xb->bsums[2 * s + 1]);
            }
            acc[v] = fmaf(d * xb->d, (float)sumi, acc[v]);
            accm[v] = fmaf(dmin * xb->d, (float)summ, accm[v]);
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = acc[v] - accm[v];
}

KATTR static void K(q6k_dotn)(const struct janas_block_q6k *w,
                              const struct janas_block_q8k *x, size_t xs,
                              int nv, size_t nb, float *out)
{
    const uint8x16_t low4 = vdupq_n_u8(0x0f), three = vdupq_n_u8(3);
    const int8x16_t bias = vdupq_n_s8(32);
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++) {
        int8x16_t wv[16];
        for (int n = 0; n < 2; n++)
            for (int part = 0; part < 4; part++) {
                const uint8_t *ql = w[b].ql + 64 * n + (part & 1) * 32;
                const uint8_t *qh = w[b].qh + 32 * n;
                for (int half = 0; half < 2; half++) {
                    uint8x16_t l = vld1q_u8(ql + 16 * half);
                    l = part < 2 ? vandq_u8(l, low4) : vshrq_n_u8(l, 4);
                    uint8x16_t h = vld1q_u8(qh + 16 * half);
                    h = vandq_u8(vshlq_u8(h, vdupq_n_s8((int8_t)(-2 * part))),
                                 three);
                    wv[8 * n + 2 * part + half] = vsubq_s8(
                        vreinterpretq_s8_u8(vorrq_u8(l, vshlq_n_u8(h, 4))),
                        bias);
                }
            }
        float d = janas_fp16_to_fp32(w[b].d);
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            int32_t sumi = 0;
            /* the one-vector kernel's order of the groups */
            for (int n = 0; n < 2; n++)
                for (int part = 0; part < 4; part++)
                    for (int half = 0; half < 2; half++) {
                        int g = 8 * n + 2 * part + half;
                        sumi += w[b].scales[g] *
                                KSUM16(wv[g], vld1q_s8(xb->qs + 16 * g));
                    }
            acc[v] = fmaf(d * xb->d, (float)sumi, acc[v]);
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = acc[v];
}

KATTR static void K(q8_0_dotn)(const struct janas_block_q8_0 *w,
                               const struct janas_block_q8k *x, size_t xs,
                               int nv, size_t nb, float *out)
{
    float acc[8][8] = {{0}};
    for (size_t b = 0; b < nb; b++)
        for (int k = 0; k < 8; k++) {
            const struct janas_block_q8_0 *wk = w + 8 * b + k;
            int8x16_t q0 = vld1q_s8(wk->qs), q1 = vld1q_s8(wk->qs + 16);
            float d = janas_fp16_to_fp32(wk->d);
            for (int v = 0; v < nv; v++) {
                const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
                const int8_t *a = xb->qs32 + 32 * k;
                int32_t dot = KSUM32W(q0, q1, vld1q_s8(a), vld1q_s8(a + 16));
                acc[v][k] = fmaf(d * xb->d32[k], (float)dot, acc[v][k]);
            }
        }
    for (int v = 0; v < nv; v++) {
        const float *a = acc[v];
        out[v] =
            ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
    }
}

/* NVFP4: blocks of 64 weights (two blocks of 32 activations), each 16 of
   them with a UE4M3 scale. A lane of 32 activations takes its two sub-blocks
   in turn, each its exact sum times the two scales, as the reference: the
   same bits. Sub-block s of a block holds its 16 codes in 8 bytes, the low
   nibbles the first eight activations, the high ones the next eight. */
KATTR static inline int8x16_t K(nvfp4_codes)(const uint8_t *q, int8x16_t lv)
{
    uint8x8_t b = vld1_u8(q);
    int8x8_t lo = vqtbl1_s8(lv, vand_u8(b, vdup_n_u8(15)));
    int8x8_t hi = vqtbl1_s8(lv, vshr_n_u8(b, 4));
    return vcombine_s8(lo, hi);
}

KATTR static float K(nvfp4_dot)(const struct janas_block_nvfp4 *w,
                                const struct janas_block_q8k *x, size_t nb)
{
    const int8x16_t lv = vld1q_s8(janas_mxfp4_values);
    float acc[8] = {0};
    for (size_t b = 0; b < nb; b++)
        for (int k = 0; k < 8; k++) {
            const struct janas_block_nvfp4 *wk = w + 4 * b + k / 2;
            const int8_t *a = x[b].qs32 + 32 * k;
            for (int h = 0; h < 2; h++) {
                int s = 2 * (k % 2) + h;
                int32_t dot = KSUM16(K(nvfp4_codes)(wk->qs + 8 * s, lv),
                                     vld1q_s8(a + 16 * h));
                acc[k] =
                    fmaf(x[b].d32[k] * ue4m3_tab[wk->d[s]], (float)dot, acc[k]);
            }
        }
    return ((acc[0] + acc[4]) + (acc[2] + acc[6])) +
           ((acc[1] + acc[5]) + (acc[3] + acc[7]));
}

/* The same on nv vectors (1 to 8): the codes and scales of a block read
   once for all of them, each vector's arithmetic as above. */
KATTR static void K(nvfp4_dotn)(const struct janas_block_nvfp4 *w,
                                const struct janas_block_q8k *x, size_t xs,
                                int nv, size_t nb, float *out)
{
    const int8x16_t lv = vld1q_s8(janas_mxfp4_values);
    float acc[8][8] = {{0}};
    for (size_t b = 0; b < nb; b++) {
        int8x16_t c[16];
        float sc[16];
        for (int k = 0; k < 8; k++)
            for (int h = 0; h < 2; h++) {
                const struct janas_block_nvfp4 *wk = w + 4 * b + k / 2;
                int s = 2 * (k % 2) + h;
                c[2 * k + h] = K(nvfp4_codes)(wk->qs + 8 * s, lv);
                sc[2 * k + h] = ue4m3_tab[wk->d[s]];
            }
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
            for (int k = 0; k < 8; k++)
                for (int h = 0; h < 2; h++) {
                    int32_t dot = KSUM16(c[2 * k + h],
                                         vld1q_s8(xb->qs32 + 32 * k + 16 * h));
                    acc[v][k] =
                        fmaf(xb->d32[k] * sc[2 * k + h], (float)dot, acc[v][k]);
                }
        }
    }
    for (int v = 0; v < nv; v++) {
        const float *a = acc[v];
        out[v] =
            ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
    }
}
