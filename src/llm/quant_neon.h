/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Rendeverance */
/*
 * quant_neon.h - the quantized dot products for ARM64 (NEON + the dot-product
 * extension), private to quant.c, which includes it once.
 *
 * Each kernel mirrors its scalar reference in quant.c: the integer sums are
 * exact, so any order gives the same int32, and every float operation is the
 * reference's, in the reference's order (fused multiply-adds lane for lane,
 * the same final pairing of the eight lanes), so the results match it bit
 * for bit; tests/test_quant.c checks this.
 */
#include <arm_neon.h>

/* "+feature", the spelling both GCC and Clang take for ARM64 */
#define NEON_DP __attribute__((target("+dotprod")))
/* The batched kernels inlined once per count of vectors (1 to 8): with nv
   a constant their per-vector accumulators live in registers. */
#define NEON_DOTN_K static inline __attribute__((always_inline)) NEON_DP
#ifdef NEON_NO_SPEC
#define NEON_NV_SWITCH(C) C(nv)
#else
#define NEON_NV_SWITCH(C)                                                      \
    switch (nv) {                                                              \
    case 1:                                                                    \
        C(1);                                                                  \
        break;                                                                 \
    case 2:                                                                    \
        C(2);                                                                  \
        break;                                                                 \
    case 3:                                                                    \
        C(3);                                                                  \
        break;                                                                 \
    case 4:                                                                    \
        C(4);                                                                  \
        break;                                                                 \
    case 5:                                                                    \
        C(5);                                                                  \
        break;                                                                 \
    case 6:                                                                    \
        C(6);                                                                  \
        break;                                                                 \
    case 7:                                                                    \
        C(7);                                                                  \
        break;                                                                 \
    default:                                                                   \
        C(8);                                                                  \
        break;                                                                 \
    }
#endif

/* The exact sum of 16 + 16 int8 products. */
static inline NEON_DP int32_t neon_dot32(int8x16_t w0, int8x16_t w1,
                                         const int8_t *a)
{
    int32x4_t s = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(a));
    s = vdotq_s32(s, w1, vld1q_s8(a + 16));
    return vaddvq_s32(s);
}

/* The same 32 products left in four lanes, to be reduced with others. */
static inline NEON_DP int32x4_t neon_dot32v(int8x16_t w0, int8x16_t w1,
                                            const int8_t *a)
{
    int32x4_t s = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(a));
    return vdotq_s32(s, w1, vld1q_s8(a + 16));
}

/* The lane sums of a, b, c, d as the four lanes of one vector: exact. */
static inline int32x4_t neon_red4(int32x4_t a, int32x4_t b, int32x4_t c,
                                  int32x4_t d)
{
    return vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d));
}

/* fp16 to fp32 in hardware, exact as janas_fp16_to_fp32: one value, and the
   scales of four blocks `stride` bytes apart (the scale first in each). */
static inline float neon_h2f(uint16_t h)
{
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}

static inline float32x4_t neon_h2f4(const void *p, size_t stride)
{
    const uint8_t *b = p;
    uint16_t h[4];
    for (int i = 0; i < 4; i++)
        memcpy(&h[i], b + i * stride, 2);
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(h)));
}

/* The 32 weights of a 4-bit block as int8 levels: low nibbles are weights
   0-15, high nibbles 16-31 (the GGUF layout). */
static inline NEON_DP void neon_nib_unpack(const uint8_t *q, int8x16_t levels,
                                           int8x16_t *w0, int8x16_t *w1)
{
    uint8x16_t b = vld1q_u8(q);
    *w0 = vqtbl1q_s8(levels, vandq_u8(b, vdupq_n_u8(15)));
    *w1 = vqtbl1q_s8(levels, vshrq_n_u8(b, 4));
}

/* The eight lanes of acc (lanes 0-3 in a, 4-7 in b) added as the reference
   does: ((0 + 4) + (2 + 6)) + ((1 + 5) + (3 + 7)). */
static inline float neon_sum8(float32x4_t a, float32x4_t b)
{
    float32x4_t s = vaddq_f32(a, b); /* 0+4 1+5 2+6 3+7 */
    float32x2_t t = vadd_f32(vget_low_f32(s),
                             vget_high_f32(s)); /* (0+4)+(2+6), (1+5)+(3+7) */
    return vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
}

/* IQ4_NL, Q4_0 and MXFP4 (scales passed per block): per block position k a
   lane, acc[k] = fma(scale_k * d32[k], dot_k, acc[k]). */
static NEON_DP float neon_nib_dot(const struct janas_block_iq4nl *w,
                                  const struct janas_block_q8k *x, size_t nb,
                                  const int8_t *lv)
{
    int8x16_t levels = vld1q_s8(lv);
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        const struct janas_block_iq4nl *wb = w + 8 * b;
        int32x4_t s[8];
        for (int k = 0; k < 8; k++) {
            int8x16_t w0, w1;
            neon_nib_unpack(wb[k].qs, levels, &w0, &w1);
            s[k] = neon_dot32v(w0, w1, x[b].qs32 + 32 * k);
        }
        float32x4_t sc0 =
            vmulq_f32(neon_h2f4(&wb[0].d, sizeof(*wb)), vld1q_f32(x[b].d32));
        float32x4_t sc1 = vmulq_f32(neon_h2f4(&wb[4].d, sizeof(*wb)),
                                    vld1q_f32(x[b].d32 + 4));
        acc0 = vfmaq_f32(acc0, sc0,
                         vcvtq_f32_s32(neon_red4(s[0], s[1], s[2], s[3])));
        acc1 = vfmaq_f32(acc1, sc1,
                         vcvtq_f32_s32(neon_red4(s[4], s[5], s[6], s[7])));
    }
    return neon_sum8(acc0, acc1);
}

static NEON_DP float q4_0_dot_neon(const struct janas_block_q4_0 *w,
                                   const struct janas_block_q8k *x, size_t nb)
{
    return neon_nib_dot((const struct janas_block_iq4nl *)w, x, nb,
                        janas_q4_0_values);
}

static NEON_DP float iq4nl_dot_neon(const struct janas_block_iq4nl *w,
                                    const struct janas_block_q8k *x, size_t nb)
{
    return neon_nib_dot(w, x, nb, janas_iq4_values);
}

static NEON_DP float q8_0_dot_neon(const struct janas_block_q8_0 *w,
                                   const struct janas_block_q8k *x, size_t nb)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        const struct janas_block_q8_0 *wb = w + 8 * b;
        int32x4_t s[8];
        for (int k = 0; k < 8; k++)
            s[k] = neon_dot32v(vld1q_s8(wb[k].qs), vld1q_s8(wb[k].qs + 16),
                               x[b].qs32 + 32 * k);
        float32x4_t sc0 =
            vmulq_f32(neon_h2f4(&wb[0].d, sizeof(*wb)), vld1q_f32(x[b].d32));
        float32x4_t sc1 = vmulq_f32(neon_h2f4(&wb[4].d, sizeof(*wb)),
                                    vld1q_f32(x[b].d32 + 4));
        acc0 = vfmaq_f32(acc0, sc0,
                         vcvtq_f32_s32(neon_red4(s[0], s[1], s[2], s[3])));
        acc1 = vfmaq_f32(acc1, sc1,
                         vcvtq_f32_s32(neon_red4(s[4], s[5], s[6], s[7])));
    }
    return neon_sum8(acc0, acc1);
}

/* Q4_1 (five = 0) and Q5_1 (five = 1): d x the exact weight sum and m x the
   activations' sum, per block position, as dm_dot_ref. */
static NEON_DP float neon_dm_dot(const uint8_t *w, size_t size, int five,
                                 const struct janas_block_q8k *x, size_t nb)
{
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t m0 = vdupq_n_f32(0.0f), m1 = vdupq_n_f32(0.0f);
    const uint8x16_t lo4 = vdupq_n_u8(15);
    /* bit i of a 16-bit half of qh, spread to byte i */
    static const uint8_t bit_of[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                       1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t bits = vld1q_u8(bit_of);
    for (size_t b = 0; b < nb; b++) {
        int32_t dot[8], asum[8];
        float sc[8], sm[8];
        for (int k = 0; k < 8; k++) {
            const uint8_t *wk = w + (8 * b + k) * size;
            const uint8_t *qs = wk + (five ? 8 : 4);
            uint8x16_t q = vld1q_u8(qs);
            uint8x16_t v0 = vandq_u8(q, lo4), v1 = vshrq_n_u8(q, 4);
            if (five) {
                uint32_t qh;
                memcpy(&qh, wk + 4, 4);
                /* bytes 0-7 take qh bits 0-7, 8-15 bits 8-15 (and 16-31 for
                   the second half) */
                uint8x16_t h0 = vcombine_u8(vdup_n_u8((uint8_t)qh),
                                            vdup_n_u8((uint8_t)(qh >> 8)));
                uint8x16_t h1 = vcombine_u8(vdup_n_u8((uint8_t)(qh >> 16)),
                                            vdup_n_u8((uint8_t)(qh >> 24)));
                v0 = vorrq_u8(v0, vandq_u8(vtstq_u8(h0, bits), vdupq_n_u8(16)));
                v1 = vorrq_u8(v1, vandq_u8(vtstq_u8(h1, bits), vdupq_n_u8(16)));
            }
            /* unsigned weights 0-31 times signed activations: as int8, exact
               (both under 128) */
            dot[k] = neon_dot32(vreinterpretq_s8_u8(v0),
                                vreinterpretq_s8_u8(v1), x[b].qs32 + 32 * k);
            asum[k] = x[b].bsums32[2 * k] + x[b].bsums32[2 * k + 1];
            uint16_t d, m;
            memcpy(&d, wk, 2);
            memcpy(&m, wk + 2, 2);
            sc[k] = neon_h2f(d) * x[b].d32[k];
            sm[k] = neon_h2f(m) * x[b].d32[k];
        }
        a0 = vfmaq_f32(a0, vld1q_f32(sc), vcvtq_f32_s32(vld1q_s32(dot)));
        a1 =
            vfmaq_f32(a1, vld1q_f32(sc + 4), vcvtq_f32_s32(vld1q_s32(dot + 4)));
        m0 = vfmaq_f32(m0, vld1q_f32(sm), vcvtq_f32_s32(vld1q_s32(asum)));
        m1 = vfmaq_f32(m1, vld1q_f32(sm + 4),
                       vcvtq_f32_s32(vld1q_s32(asum + 4)));
    }
    return neon_sum8(a0, a1) + neon_sum8(m0, m1);
}

static NEON_DP float q4_1_dot_neon(const struct janas_block_q4_1 *w,
                                   const struct janas_block_q8k *x, size_t nb)
{
    return neon_dm_dot((const uint8_t *)w, sizeof(*w), 0, x, nb);
}

static NEON_DP float q5_1_dot_neon(const struct janas_block_q5_1 *w,
                                   const struct janas_block_q8k *x, size_t nb)
{
    return neon_dm_dot((const uint8_t *)w, sizeof(*w), 1, x, nb);
}

/* Q6_K: per group of 16 the exact sum of (q - 32) x a, times the group's
   scale, all in one int32 per block; then one fma per block, as the
   reference. */

/*
 * The same products against nv activation vectors at once (nv 1 to 8), as
 * janas_dot_multi asks: each weight block is unpacked once and used for all
 * of them. Per vector the integer sums and the float operations are those of
 * the single-vector kernel above, in the same order, so every result is that
 * kernel's (and the reference's), bit for bit.
 */
NEON_DOTN_K void neon_nib_dotn_k(const struct janas_block_iq4nl *w,
                                 const struct janas_block_q8k *x, size_t xs,
                                 int nv, size_t nb, float *out,
                                 const int8_t *lv)
{
    int8x16_t levels = vld1q_s8(lv);
    float32x4_t a0[8], a1[8];
    for (int v = 0; v < nv; v++)
        a0[v] = a1[v] = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        const struct janas_block_iq4nl *wb = w + 8 * b;
        int8x16_t w0[8], w1[8];
        for (int k = 0; k < 8; k++)
            neon_nib_unpack(wb[k].qs, levels, &w0[k], &w1[k]);
        float32x4_t d0 = neon_h2f4(&wb[0].d, sizeof(*wb));
        float32x4_t d1 = neon_h2f4(&wb[4].d, sizeof(*wb));
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + v * xs + b;
            const int8_t *q = xb->qs32;
            int32x4_t lo = neon_red4(neon_dot32v(w0[0], w1[0], q),
                                     neon_dot32v(w0[1], w1[1], q + 32),
                                     neon_dot32v(w0[2], w1[2], q + 64),
                                     neon_dot32v(w0[3], w1[3], q + 96));
            int32x4_t hi = neon_red4(neon_dot32v(w0[4], w1[4], q + 128),
                                     neon_dot32v(w0[5], w1[5], q + 160),
                                     neon_dot32v(w0[6], w1[6], q + 192),
                                     neon_dot32v(w0[7], w1[7], q + 224));
            a0[v] = vfmaq_f32(a0[v], vmulq_f32(d0, vld1q_f32(xb->d32)),
                              vcvtq_f32_s32(lo));
            a1[v] = vfmaq_f32(a1[v], vmulq_f32(d1, vld1q_f32(xb->d32 + 4)),
                              vcvtq_f32_s32(hi));
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = neon_sum8(a0[v], a1[v]);
}

static NEON_DP void neon_nib_dotn(const struct janas_block_iq4nl *w,
                                  const struct janas_block_q8k *x, size_t xs,
                                  int nv, size_t nb, float *out,
                                  const int8_t *lv){
#define C(N) neon_nib_dotn_k(w, x, xs, N, nb, out, lv)
    NEON_NV_SWITCH(C)
#undef C
}

NEON_DOTN_K void q8_0_dotn_neon_k(const struct janas_block_q8_0 *w,
                                  const struct janas_block_q8k *x, size_t xs,
                                  int nv, size_t nb, float *out)
{
    float32x4_t a0[8], a1[8];
    for (int v = 0; v < nv; v++)
        a0[v] = a1[v] = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        const struct janas_block_q8_0 *wb = w + 8 * b;
        int8x16_t w0[8], w1[8];
        for (int k = 0; k < 8; k++) {
            w0[k] = vld1q_s8(wb[k].qs);
            w1[k] = vld1q_s8(wb[k].qs + 16);
        }
        float32x4_t d0 = neon_h2f4(&wb[0].d, sizeof(*wb));
        float32x4_t d1 = neon_h2f4(&wb[4].d, sizeof(*wb));
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + v * xs + b;
            const int8_t *q = xb->qs32;
            int32x4_t lo = neon_red4(neon_dot32v(w0[0], w1[0], q),
                                     neon_dot32v(w0[1], w1[1], q + 32),
                                     neon_dot32v(w0[2], w1[2], q + 64),
                                     neon_dot32v(w0[3], w1[3], q + 96));
            int32x4_t hi = neon_red4(neon_dot32v(w0[4], w1[4], q + 128),
                                     neon_dot32v(w0[5], w1[5], q + 160),
                                     neon_dot32v(w0[6], w1[6], q + 192),
                                     neon_dot32v(w0[7], w1[7], q + 224));
            a0[v] = vfmaq_f32(a0[v], vmulq_f32(d0, vld1q_f32(xb->d32)),
                              vcvtq_f32_s32(lo));
            a1[v] = vfmaq_f32(a1[v], vmulq_f32(d1, vld1q_f32(xb->d32 + 4)),
                              vcvtq_f32_s32(hi));
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = neon_sum8(a0[v], a1[v]);
}

static NEON_DP void q8_0_dotn_neon(const struct janas_block_q8_0 *w,
                                   const struct janas_block_q8k *x, size_t xs,
                                   int nv, size_t nb, float *out){
#define C(N) q8_0_dotn_neon_k(w, x, xs, N, nb, out)
    NEON_NV_SWITCH(C)
#undef C
}

/* Q6_K: the 256 weights of a block unpacked once (as int8, q - 32). */
NEON_DOTN_K void q6k_dotn_neon_k(const struct janas_block_q6k *w,
                                 const struct janas_block_q8k *x, size_t xs,
                                 int nv, size_t nb, float *out)
{
    float acc[8] = {0};
    const uint8x16_t m4 = vdupq_n_u8(15), m2 = vdupq_n_u8(3);
    const int8x16_t c32 = vdupq_n_s8(32);
    for (size_t b = 0; b < nb; b++) {
        const struct janas_block_q6k *wb = &w[b];
        int8x16_t q[16]; /* q[i] holds weights 16i .. 16i + 15 */
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql = wb->ql + 64 * n, *qh = wb->qh + 32 * n;
            for (int h = 0; h < 2; h++) {
                uint8x16_t l0 = vld1q_u8(ql + 16 * h),
                           l1 = vld1q_u8(ql + 32 + 16 * h);
                uint8x16_t hh = vld1q_u8(qh + 16 * h);
                int base =
                    8 * n + h; /* v = 128n + 32p + 16h -> index 8n + 2p + h */
                q[base + 0] = vsubq_s8(
                    vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(l0, m4), vshlq_n_u8(vandq_u8(hh, m2), 4))),
                    c32);
                q[base + 2] = vsubq_s8(
                    vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(l1, m4),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 2), m2), 4))),
                    c32);
                q[base + 4] = vsubq_s8(
                    vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8(l0, 4),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 4), m2), 4))),
                    c32);
                q[base + 6] = vsubq_s8(
                    vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8(l1, 4), vshlq_n_u8(vshrq_n_u8(hh, 6), 4))),
                    c32);
            }
        }
        float d = neon_h2f(wb->d);
        /* the 16 group scales as four int32 vectors, once for all vectors */
        int8x16_t sc8 = vld1q_s8(wb->scales);
        int16x8_t sl = vmovl_s8(vget_low_s8(sc8)),
                  sh = vmovl_s8(vget_high_s8(sc8));
        int32x4_t sv0 = vmovl_s16(vget_low_s16(sl)),
                  sv1 = vmovl_s16(vget_high_s16(sl));
        int32x4_t sv2 = vmovl_s16(vget_low_s16(sh)),
                  sv3 = vmovl_s16(vget_high_s16(sh));
        for (int v = 0; v < nv; v++) {
            const int8_t *a = (x + v * xs + b)->qs;
            int32x4_t s[16];
            for (int g = 0; g < 16; g++)
                s[g] = vdotq_s32(vdupq_n_s32(0), q[g], vld1q_s8(a + 16 * g));
            /* lane j of red4(s[4i..4i+3]) is group 4i + j: times its scale */
            int32x4_t si = vmulq_s32(neon_red4(s[0], s[1], s[2], s[3]), sv0);
            si = vmlaq_s32(si, neon_red4(s[4], s[5], s[6], s[7]), sv1);
            si = vmlaq_s32(si, neon_red4(s[8], s[9], s[10], s[11]), sv2);
            si = vmlaq_s32(si, neon_red4(s[12], s[13], s[14], s[15]), sv3);
            acc[v] =
                fmaf(d * (x + v * xs + b)->d, (float)vaddvq_s32(si), acc[v]);
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = acc[v];
}

static NEON_DP void q6k_dotn_neon(const struct janas_block_q6k *w,
                                  const struct janas_block_q8k *x, size_t xs,
                                  int nv, size_t nb, float *out)
{
#define C(N) q6k_dotn_neon_k(w, x, xs, N, nb, out)
    NEON_NV_SWITCH(C)
#undef C
}

static NEON_DP float q6k_dot_neon(const struct janas_block_q6k *w,
                                  const struct janas_block_q8k *x, size_t nb)
{
    float r;
    q6k_dotn_neon_k(w, x, 0, 1, nb, &r);
    return r;
}
