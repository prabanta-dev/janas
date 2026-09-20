/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * quant.h - quantized block formats and dot-product kernels.
 *
 * The block layouts are byte-compatible with the GGUF k-quants, so weights can
 * be taken from existing GGUF files and results compared against llama.cpp:
 *
 *   Q4_K: 256 weights in 144 bytes. Two fp16 super-block scales (d, dmin),
 *         12 bytes packing eight 6-bit sub-block scales and eight 6-bit
 *         sub-block mins, and 128 bytes of 4-bit values. Weight i of
 *         sub-block s is  d * scale[s] * q - dmin * min[s].
 *         The 128 value bytes hold four 64-weight chunks of 32 bytes: in chunk
 *         c the low nibbles are sub-block 2c and the high nibbles 2c+1.
 *
 *   Q6_K: 256 weights in 210 bytes: low 4 bits (128 bytes), high 2 bits (64
 *         bytes), sixteen int8 scales and one fp16 scale d. Weight i is
 *         d * scale[i / 16] * (q - 32), q in 0..63.
 *
 *   Q8_K: activations, 256 values as int8 with one float scale, plus the sums
 *         of each group of 16 values (needed for the mins term).
 */
#ifndef JANAS_LLM_QUANT_H
#define JANAS_LLM_QUANT_H

#include <stddef.h>
#include <stdint.h>

#define JANAS_QK 256 /* weights per super-block */

struct janas_block_q4k {
    uint16_t d;    /* fp16 */
    uint16_t dmin; /* fp16 */
    uint8_t scales[12];
    uint8_t qs[JANAS_QK / 2];
};

struct janas_block_q5k {
    uint16_t d;    /* fp16 */
    uint16_t dmin; /* fp16 */
    uint8_t scales[12];
    uint8_t qh[JANAS_QK / 8]; /* bit s of qh[l]: 5th bit of weight 32 s + l */
    uint8_t qs[JANAS_QK / 2]; /* low 4 bits, as in Q4_K */
};

/* Q8_0: 32 weights and their scale; eight of them cover a Q8_K block. */
struct janas_block_q8_0 {
    uint16_t d; /* fp16 */
    int8_t qs[32];
};

/*
 * IQ4_NL: 32 weights, each a 4-bit index into janas_iq4_values (sixteen
 * levels spaced unevenly, closer near zero), times the block's scale;
 * eight blocks cover a Q8_K block. Byte l holds weight l in its low nibble
 * and weight l + 16 in its high one.
 */
struct janas_block_iq4nl {
    uint16_t d; /* fp16 */
    uint8_t qs[16];
};

/*
 * IQ4_XS: 256 weights in eight groups of 32, each group laid out as an
 * IQ4_NL block without its scale. Group g's scale is d * (ls - 32), ls a
 * 6-bit number: its low four bits in scales_l[g / 2] (low nibble for even
 * g), its high two in bits 2g, 2g + 1 of scales_h.
 */
struct janas_block_iq4xs {
    uint16_t d; /* fp16 */
    uint16_t scales_h;
    uint8_t scales_l[JANAS_QK / 64];
    uint8_t qs[JANAS_QK / 2];
};

/* The sixteen levels of IQ4_NL and IQ4_XS. */
extern const int8_t janas_iq4_values[16];

/* Q4_0: IQ4_NL's layout with evenly spaced levels, -8 to 7 (the kernels
   are IQ4_NL's with another table). */
struct janas_block_q4_0 {
    uint16_t d; /* fp16 */
    uint8_t qs[16];
};
extern const int8_t janas_q4_0_values[16];

/*
 * Q5_1: 32 weights d * q + m, q of five bits: the low four as in Q4_0,
 * the fifth of weight l in bit l of qh (little-endian).
 */
struct janas_block_q5_1 {
    uint16_t d, m; /* fp16 */
    uint8_t qh[4];
    uint8_t qs[16];
};

/*
 * Q3_K: 256 weights of three bits in sixteen groups of 16, each with a
 * 6-bit scale (sc - 32; the twelve bytes hold the low four bits of scales
 * 0-7 and 8-15 in the low and high nibbles of bytes 0-7, the top two bits
 * in bytes 8-11). Weight e = 128 n + 32 j + 16 h + l: its low two bits are
 * bits 2j of qs[32 n + 16 h + l], its third bit j + 4n of hmask[16 h + l],
 * and it is those three bits less 4 when the third is clear: -4 to 3.
 */
struct janas_block_q3k {
    uint8_t hmask[JANAS_QK / 8];
    uint8_t qs[JANAS_QK / 4];
    uint8_t scales[12];
    uint16_t d; /* fp16 */
};

/*
 * IQ3_S: 256 weights in eight groups of 32. Group g: eight 9-bit indices
 * into a grid of 512 entries of four levels each (qs[8g + k], its ninth
 * bit bit k of qh[g]), the signs of its 32 weights in signs[4g .. 4g + 3]
 * (bit e for weight e, set: negative), and the scale 1 + 2 x a nibble of
 * scales[g / 2] (low for even g), times d.
 */
struct janas_block_iq3s {
    uint16_t d; /* fp16 */
    uint8_t qs[JANAS_QK / 4];
    uint8_t qh[JANAS_QK / 32];
    uint8_t signs[JANAS_QK / 8];
    uint8_t scales[JANAS_QK / 64];
};

/* Q4_1: Q5_1 without the fifth bits, q of four (0-15). */
struct janas_block_q4_1 {
    uint16_t d, m; /* fp16 */
    uint8_t qs[16];
};

struct janas_block_q6k {
    uint8_t ql[JANAS_QK / 2];
    uint8_t qh[JANAS_QK / 4];
    int8_t scales[JANAS_QK / 16];
    uint16_t d; /* fp16 */
};

/*
 * Q6_K in bit planes: the six bits of every weight cut into three planes of
 * two bits, each packed exactly as Q6_K's qh field (byte 32n + l, bits 2p).
 * The high plane carries the scales, the other two are bare. Weight i is
 * 16 * p2 + 4 * p1 + p0, so a row can be read one plane at a time: with all
 * three the value is the Q6_K one, with two the centre of the four it stands
 * for (4 * q + 1.5), with one the centre of sixteen (16 * q + 7.5).
 */
struct janas_block_q6kp {
    uint8_t q[JANAS_QK / 4]; /* the high two bits of every weight */
    int8_t scales[JANAS_QK / 16];
    uint16_t d; /* fp16 */
};
#define JANAS_Q6KP_PLANE (JANAS_QK / 4) /* bytes of one bare plane, 64 */

struct janas_block_q8k {
    float d;
    int8_t qs[JANAS_QK];
    int16_t bsums[JANAS_QK / 16];
};

_Static_assert(sizeof(struct janas_block_q4k) == 144, "q4k block size");
_Static_assert(sizeof(struct janas_block_q6k) == 210, "q6k block size");
_Static_assert(sizeof(struct janas_block_q6kp) == 82, "q6kp block size");
_Static_assert(sizeof(struct janas_block_q5k) == 176, "q5k block size");
_Static_assert(sizeof(struct janas_block_q8_0) == 34, "q8_0 block size");
_Static_assert(sizeof(struct janas_block_iq4nl) == 18, "iq4_nl block size");
_Static_assert(sizeof(struct janas_block_iq4xs) == 136, "iq4_xs block size");
_Static_assert(sizeof(struct janas_block_q4_0) == 18, "q4_0 block size");
_Static_assert(sizeof(struct janas_block_q5_1) == 24, "q5_1 block size");
_Static_assert(sizeof(struct janas_block_q4_1) == 20, "q4_1 block size");
_Static_assert(sizeof(struct janas_block_q3k) == 110, "q3k block size");
_Static_assert(sizeof(struct janas_block_iq3s) == 110, "iq3_s block size");
_Static_assert(sizeof(struct janas_block_q8k) == 292, "q8k block size");

/* Quantization types, numbered as in GGUF (ggml_type). */
enum janas_qtype {
    JANAS_Q4_0 = 2, /* sizes per 256 weights: 8 blocks, 144 bytes */
    JANAS_Q4_1 = 3, /* 8 blocks, 160 bytes */
    JANAS_Q5_1 = 7, /* 8 blocks, 192 bytes */
    JANAS_Q8_0 = 8,
    JANAS_Q3_K = 11, /* sizes counted per 256 weights: 8 blocks, 272 bytes */
    JANAS_Q4_K = 12,
    JANAS_Q5_K = 13,
    JANAS_Q6_K = 14,
    JANAS_IQ4_NL = 20,
    JANAS_IQ3_S = 21, /* sizes per 256 weights: 8 blocks, 144 bytes */
    JANAS_IQ4_XS = 23,
    /* Janas' own, outside the GGUF numbering: the high plane of a Q6_K row
       cut into bit planes. Its block size counts that plane alone; the other
       two are JANAS_Q6KP_PLANE bytes each and are passed separately. */
    JANAS_Q6_K_P = 214
};

/* Bytes of one 256-weight block of the type, 0 if unsupported. */
size_t janas_qtype_block_size(int type);

/*
 * Dot product of nb blocks of a supported type with nb Q8_K blocks. Aborts on
 * an unsupported type: check janas_qtype_block_size() first.
 */
float janas_dot(int type, const void *w, const struct janas_block_q8k *x,
                size_t nb);

/*
 * One row of n weights (n a multiple of JANAS_QK) of a supported type into
 * floats. Aborts on an unsupported type, bit planes among them: check
 * janas_qtype_block_size() first. Every caller goes through here, so a type
 * the engine reads in its products is read the same way when a row of it has
 * to be unpacked whole - a token embedding, say.
 */
void janas_dequantize(int type, const void *w, float *y, size_t n);

/*
 * The dot products of one weight row with nv vectors (vector v at x + v *
 * x_stride blocks), result v at out[v * out_stride]. Bit-identical to calling
 * janas_dot for each vector, but unpacks the weights once per four vectors.
 */
void janas_dot_multi(int type, const void *w, const struct janas_block_q8k *x,
                     size_t x_stride, size_t nv, size_t nb, float *out,
                     size_t out_stride);

/* Float helpers, AVX2 when available: dot product and y += a * x. */
float janas_dot_f32(const float *a, const float *b, size_t n);
void janas_axpy_f32(float *y, float a, const float *x, size_t n);

float janas_fp16_to_fp32(uint16_t h);
uint16_t janas_fp32_to_fp16(float f);

/* Quantizes n floats (n multiple of JANAS_QK) into n / JANAS_QK blocks. */
void janas_q8k_quantize(const float *x, struct janas_block_q8k *y, size_t n);

/* The same with deterministic operations only (no division): what the
   GPU reproduces bit for bit, for values a GPU may also compute. */
void janas_q8k_quantize_det(const float *x, struct janas_block_q8k *y,
                            size_t n);

/*
 * Quantizes n floats (n multiple of JANAS_QK) into Q4_K blocks: per 32-value
 * sub-block an affine fit searched over shrunken ranges and refit by least
 * squares, then the two block factors refit by least squares against the
 * 6-bit sub-block scales. For offline conversion, not for the hot path.
 */
void janas_q4k_quantize(const float *x, struct janas_block_q4k *y, size_t n);

/* Dequantizes n weights (n multiple of JANAS_QK). */
void janas_q4k_dequantize(const struct janas_block_q4k *x, float *y, size_t n);

/*
 * Dot product of nb Q4_K blocks with nb Q8_K blocks. janas_q4k_dot uses the
 * fastest kernel the CPU supports, chosen once at run time; the _ref variant
 * is the portable scalar definition every other kernel is tested against.
 */
float janas_q4k_dot(const struct janas_block_q4k *w,
                    const struct janas_block_q8k *x, size_t nb);
float janas_q4k_dot_ref(const struct janas_block_q4k *w,
                        const struct janas_block_q8k *x, size_t nb);

/* Name of the kernel janas_q4k_dot dispatches to ("avx2" or "ref"). */
const char *janas_q4k_kernel_name(void);

/* Whether the AVX-VNNI kernels are in use: the CPU has them and
   JANAS_KERNELS does not ask for plain AVX2. Attention asks the same
   question, and both answers give the same numbers. */
int janas_use_vnni(void);

/* The same three entry points for Q6_K. */
void janas_q6k_dequantize(const struct janas_block_q6k *x, float *y, size_t n);
float janas_q6k_dot(const struct janas_block_q6k *w,
                    const struct janas_block_q8k *x, size_t nb);
float janas_q6k_dot_ref(const struct janas_block_q6k *w,
                        const struct janas_block_q8k *x, size_t nb);

/*
 * The bit-plane form of Q6_K. p1 and p0 are the bare planes of the same
 * blocks (JANAS_Q6KP_PLANE bytes each); p0 may be NULL, and p1 only when p0
 * is too, giving the three levels 6, 4 and 2 bits per weight. With all three
 * planes the result is bit-identical to janas_q6k_dot on the block the three
 * came from.
 */
void janas_q6k_to_planes(const struct janas_block_q6k *src, size_t nb,
                         struct janas_block_q6kp *base, uint8_t *p1,
                         uint8_t *p0);
void janas_q6kp_to_q6k(const struct janas_block_q6kp *base, const uint8_t *p1,
                       const uint8_t *p0, size_t nb,
                       struct janas_block_q6k *dst);
void janas_q6kp_dequantize(const struct janas_block_q6kp *base,
                           const uint8_t *p1, const uint8_t *p0, float *y,
                           size_t n);
float janas_q6kp_dot(const struct janas_block_q6kp *w, const uint8_t *p1,
                     const uint8_t *p0, const struct janas_block_q8k *x,
                     size_t nb);
float janas_q6kp_dot_ref(const struct janas_block_q6kp *w, const uint8_t *p1,
                         const uint8_t *p0, const struct janas_block_q8k *x,
                         size_t nb);
/* As janas_dot_multi, for the plane form: one row against nv vectors. */
void janas_q6kp_dot_multi(const struct janas_block_q6kp *w, const uint8_t *p1,
                          const uint8_t *p0, const struct janas_block_q8k *x,
                          size_t x_stride, size_t nv, size_t nb, float *out,
                          size_t out_stride);

/* And for Q5_K (nb blocks of 256 weights) and Q8_0 (nb groups of eight
   32-weight blocks, 256 weights each). Q8_0 sums each of the eight block
   positions in its own accumulator (one fma per block), then adds the eight
   as janas_hsum8 does: ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)). */
void janas_q5k_dequantize(const struct janas_block_q5k *x, float *y, size_t n);
float janas_q5k_dot(const struct janas_block_q5k *w,
                    const struct janas_block_q8k *x, size_t nb);
float janas_q5k_dot_ref(const struct janas_block_q5k *w,
                        const struct janas_block_q8k *x, size_t nb);
void janas_q8_0_dequantize(const struct janas_block_q8_0 *x, float *y,
                           size_t n);
/* IQ4_NL (nb groups of eight blocks, as Q8_0: each block's sum in its own
   accumulator, the eight added as janas_hsum8 does) and IQ4_XS (one exact
   integer sum per 256 weights, scales included, then one fma: as Q4_K). */
void janas_iq4nl_dequantize(const struct janas_block_iq4nl *x, float *y,
                            size_t n);
float janas_iq4nl_dot(const struct janas_block_iq4nl *w,
                      const struct janas_block_q8k *x, size_t nb);
float janas_iq4nl_dot_ref(const struct janas_block_iq4nl *w,
                          const struct janas_block_q8k *x, size_t nb);
void janas_iq4xs_dequantize(const struct janas_block_iq4xs *x, float *y,
                            size_t n);
float janas_iq4xs_dot(const struct janas_block_iq4xs *w,
                      const struct janas_block_q8k *x, size_t nb);
float janas_iq4xs_dot_ref(const struct janas_block_iq4xs *w,
                          const struct janas_block_q8k *x, size_t nb);
/* Q4_0 (as IQ4_NL) and Q5_1 (per block position, a lane for d x the
   weights' sums and one for m x the activations' sums, from the Q8_K
   block sums; each set of eight added as janas_hsum8 does, then the two) */
void janas_q4_0_dequantize(const struct janas_block_q4_0 *x, float *y,
                           size_t n);
float janas_q4_0_dot(const struct janas_block_q4_0 *w,
                     const struct janas_block_q8k *x, size_t nb);
float janas_q4_0_dot_ref(const struct janas_block_q4_0 *w,
                         const struct janas_block_q8k *x, size_t nb);
void janas_q5_1_dequantize(const struct janas_block_q5_1 *x, float *y,
                           size_t n);
float janas_q5_1_dot(const struct janas_block_q5_1 *w,
                     const struct janas_block_q8k *x, size_t nb);
float janas_q5_1_dot_ref(const struct janas_block_q5_1 *w,
                         const struct janas_block_q8k *x, size_t nb);
/* Q3_K: one exact integer sum per 256 weights, scales included, then one
   fma (as Q4_K) */
void janas_q3k_dequantize(const struct janas_block_q3k *x, float *y, size_t n);
float janas_q3k_dot(const struct janas_block_q3k *w,
                    const struct janas_block_q8k *x, size_t nb);
float janas_q3k_dot_ref(const struct janas_block_q3k *w,
                        const struct janas_block_q8k *x, size_t nb);
/* IQ3_S: one exact integer sum per 256 weights, then one fma */
void janas_iq3s_dequantize(const struct janas_block_iq3s *x, float *y,
                           size_t n);
float janas_iq3s_dot(const struct janas_block_iq3s *w,
                     const struct janas_block_q8k *x, size_t nb);
float janas_iq3s_dot_ref(const struct janas_block_iq3s *w,
                         const struct janas_block_q8k *x, size_t nb);
/* Q4_1: as Q5_1 */
void janas_q4_1_dequantize(const struct janas_block_q4_1 *x, float *y,
                           size_t n);
float janas_q4_1_dot(const struct janas_block_q4_1 *w,
                     const struct janas_block_q8k *x, size_t nb);
float janas_q4_1_dot_ref(const struct janas_block_q4_1 *w,
                         const struct janas_block_q8k *x, size_t nb);
float janas_q8_0_dot(const struct janas_block_q8_0 *w,
                     const struct janas_block_q8k *x, size_t nb);
float janas_q8_0_dot_ref(const struct janas_block_q8_0 *w,
                         const struct janas_block_q8k *x, size_t nb);

#endif
