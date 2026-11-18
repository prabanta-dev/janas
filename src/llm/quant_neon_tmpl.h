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
