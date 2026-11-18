/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * quant_neon.h - the dot products of quant.c on 64-bit ARM, with the
 * Advanced SIMD instructions every ARMv8-A processor has (no dot product
 * extension needed). Included by quant.c alone, on AArch64, so that the
 * kernels share its helpers (unpack_scales, the value tables).
 *
 * Every kernel keeps the references' scheme: a block's integer sums made
 * whole first - exact in any order - then the same float operations as
 * the reference, in the same order (fmaf is fused on ARM as on x86): the
 * result is the reference's to the bit, as the x86 kernels' are. The sums
 * are made two ways, the kernels' bodies the same (quant_neon_tmpl.h):
 * by widening multiplies into 16-bit lanes and their sum into 32 bits
 * (*_neon, any ARMv8-A), or by the dot product instructions of ARMv8.2's
 * extension (*_sdot: four bytes' products into a 32-bit lane at once),
 * where the processor has them (Cortex-A55, A76 and later, Neoverse,
 * Apple's).
 */
#include <arm_neon.h>

/* sum of a[0..31] * b[0..31], a and b in signed bytes whose products stay
   within 16 bits by fours (|a * b| <= 8191 each) */
static inline int32_t s8_sum32(int8x16_t a0, int8x16_t a1, int8x16_t b0,
                               int8x16_t b1)
{
    int16x8_t p = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
    p = vmlal_s8(p, vget_high_s8(a0), vget_high_s8(b0));
    p = vmlal_s8(p, vget_low_s8(a1), vget_low_s8(b1));
    p = vmlal_s8(p, vget_high_s8(a1), vget_high_s8(b1));
    return vaddlvq_s16(p);
}

/* the same without the bound: two products at most in a 16-bit lane */
static inline int32_t s8_sum32w(int8x16_t a0, int8x16_t a1, int8x16_t b0,
                                int8x16_t b1)
{
    int32x4_t s =
        vaddq_s32(vpaddlq_s16(vmull_s8(vget_low_s8(a0), vget_low_s8(b0))),
                  vpaddlq_s16(vmull_s8(vget_high_s8(a0), vget_high_s8(b0))));
    s = vaddq_s32(s, vpaddlq_s16(vmull_s8(vget_low_s8(a1), vget_low_s8(b1))));
    s = vaddq_s32(s, vpaddlq_s16(vmull_s8(vget_high_s8(a1), vget_high_s8(b1))));
    return vaddvq_s32(s);
}

/* 16 products (two by two in 16 bits): Q6_K's groups, whose products
   reach 32 * 128 */
static inline int32_t s8_sum16(int8x16_t a, int8x16_t b)
{
    int16x8_t p = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t q = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddvq_s32(vaddq_s32(vpaddlq_s16(p), vpaddlq_s16(q)));
}

#define SDOT __attribute__((target("+dotprod")))

/* the dot product instructions: four products into each 32-bit lane */
SDOT static inline int32_t sdot_sum32(int8x16_t a0, int8x16_t a1, int8x16_t b0,
                                      int8x16_t b1)
{
    int32x4_t s = vdotq_s32(vdupq_n_s32(0), a0, b0);
    return vaddvq_s32(vdotq_s32(s, a1, b1));
}

SDOT static inline int32_t sdot_sum16(int8x16_t a, int8x16_t b)
{
    return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), a, b));
}

#define K(name) name##_neon
#define KATTR
#define KSUM32 s8_sum32
#define KSUM32W s8_sum32w
#define KSUM16 s8_sum16
#include "quant_neon_tmpl.h"
#undef K
#undef KATTR
#undef KSUM32
#undef KSUM32W
#undef KSUM16

#define K(name) name##_sdot
#define KATTR SDOT
#define KSUM32 sdot_sum32
#define KSUM32W sdot_sum32
#define KSUM16 sdot_sum16
#include "quant_neon_tmpl.h"
#undef K
#undef KATTR
#undef KSUM32
#undef KSUM32W
#undef KSUM16
#undef SDOT
