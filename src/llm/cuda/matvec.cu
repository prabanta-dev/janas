/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * matvec.cu - the quantized products on an NVIDIA GPU (gpu_cuda.c), in the
 * per-block integer-sum scheme of src/llm/quant_dotn.h and of the Vulkan
 * shaders (shaders/q4k_matvec.comp, q6k_matvec.comp): the same float
 * operations in the same order as the CPU, so that a product split between
 * the two gives the same bits as either alone.
 *
 * Apple's GPUs through Metal (gpu_metal.m) compile the same file at run
 * time as Metal's shading language (__METAL_VERSION__ below): the pointers
 * there say where they point (GLOBAL the GPU's buffers, PRIV a thread's
 * own), a kernel takes its thread's index and its buffers as arguments
 * (KARGS), FMA is Metal's exact fma; elsewhere these say nothing.
 *
 * Compiled by build.sh with clang to PTX (--cuda-device-only, no CUDA
 * headers or libraries: none of NVIDIA's toolkit), -ffp-contract=off so
 * that no product is fused with a sum the CPU rounds apart; the driver
 * compiles the PTX for the card it finds. sm_61 and later (dp4a).
 *
 * The activations come as the Vulkan shaders read them (gpu.c convert_x):
 * a Q8_K block in 320 bytes, its 256 quants, its 16 sums of 16 as int16,
 * its scale. y[v * ys + row]; x's vector v at block v * xs.
 */
#if defined(__NVPTX__) /* the card's code */

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define INLINE static __device__ __attribute__((always_inline))
#define KERNEL extern "C" __global__ void

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

INLINE int dp4a(u32 a, u32 b, int c)
{
    int r;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
INLINE float h2f(u32 h)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((u16)h));
    return f;
}
INLINE float bits_f(u32 u) { return __builtin_bit_cast(float, u); }
INLINE u32 row_index(void)
{
    return __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() +
           __nvvm_read_ptx_sreg_tid_x();
}
INLINE u32 thread_index(void) { return __nvvm_read_ptx_sreg_tid_x(); }
INLINE u32 block_index(void) { return __nvvm_read_ptx_sreg_ctaid_x(); }
INLINE void barrier(void) { asm volatile("bar.sync 0;" ::: "memory"); }
#define SHARED static __attribute__((shared))

#elif defined(__AMDGPU__) /* an AMD card's code, through HIP (gpu_cuda.c) */

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define INLINE static __device__ __attribute__((always_inline))
#define KERNEL extern "C" __global__ void

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

/* signed bytes' dot product added to c: the compiler takes the GPU's dot
   instruction where it has one */
INLINE int dp4a(u32 a, u32 b, int c)
{
    for (u32 k = 0; k < 32u; k += 8u)
        c += (int)(signed char)(u8)(a >> k) * (int)(signed char)(u8)(b >> k);
    return c;
}
INLINE float h2f(u32 h) { return (float)__builtin_bit_cast(_Float16, (u16)h); }
INLINE float bits_f(u32 u) { return __builtin_bit_cast(float, u); }
/* the launches give 128 threads a block (gpu_cuda.c ROWS_A_BLOCK) */
INLINE u32 row_index(void)
{
    return __builtin_amdgcn_workgroup_id_x() * 128u +
           __builtin_amdgcn_workitem_id_x();
}
INLINE u32 thread_index(void) { return __builtin_amdgcn_workitem_id_x(); }
INLINE u32 block_index(void) { return __builtin_amdgcn_workgroup_id_x(); }
INLINE void barrier(void)
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
#define SHARED static __attribute__((shared))

#elif defined(__METAL_VERSION__) /* an Apple GPU's code, through Metal */

#include <metal_stdlib>
using namespace metal;

#define INLINE static inline
#define KERNEL kernel void
#define GLOBAL device
#define PRIV thread
#define FMA(a, b, c) fma(a, b, c)
#define U64 ulong

typedef uint u32;
typedef ushort u16;
typedef uchar u8;

INLINE int dp4a(u32 a, u32 b, int c)
{
    for (u32 k = 0; k < 32u; k += 8u)
        c += (int)(char)(u8)(a >> k) * (int)(char)(u8)(b >> k);
    return c;
}
INLINE float h2f(u32 h) { return float(as_type<half>((u16)h)); }
INLINE float bits_f(u32 u) { return as_type<float>(u); }
/* the thread's row, and the products' arguments, as Metal passes them */
#define row_index() row_index_
#define KARGS(WT)                                                              \
    GLOBAL const WT *w [[buffer(0)]], GLOBAL const u32 *x [[buffer(1)]],       \
        GLOBAL float *y [[buffer(2)]], constant u32 &rows [[buffer(3)]],       \
        constant u32 &nb [[buffer(4)]], constant u32 &nv [[buffer(5)]],        \
        constant u32 &xs [[buffer(6)]], constant u32 &ys [[buffer(7)]]
#define KARG_KIND , constant u32 &kind [[buffer(8)]]
#define KARG_ROW , u32 row_index_ [[thread_position_in_grid]]

#else /* the same kernels as C on the CPU, one row a call, to check their
         arithmetic against the CPU's kernels where there is no card
         (job/tests/c/cuda_host_check.c) */

#include <string.h>

#define INLINE static inline
#define KERNEL void

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

static u32 cuda_host_row; /* the thread being played */

INLINE int dp4a(u32 a, u32 b, int c)
{
    for (u32 k = 0; k < 32u; k += 8u)
        c += (int)(signed char)(u8)(a >> k) * (int)(signed char)(u8)(b >> k);
    return c;
}
INLINE float h2f(u32 h)
{
    u32 s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, f;
    if (!e && !m)
        f = s;
    else if (!e) { /* a subnormal half is a normal float */
        e = 113u;
        while (!(m & 0x400u))
            m <<= 1, e--;
        f = s | (e << 23) | ((m & 0x3ffu) << 13);
    } else if (e == 31u)
        f = s | 0x7f800000u | (m << 13);
    else
        f = s | ((e + 112u) << 23) | (m << 13);
    float r;
    memcpy(&r, &f, sizeof r);
    return r;
}
INLINE float bits_f(u32 u)
{
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}
INLINE u32 row_index(void) { return cuda_host_row; }

#endif

#if !defined(__METAL_VERSION__) /* where pointers say nothing of where */
#define GLOBAL
#define PRIV
#define FMA_BUILTIN 1
#define U64_LL 1
#define KARGS(WT)                                                              \
    const WT *w, const u32 *x, float *y, u32 rows, u32 nb, u32 nv, u32 xs,     \
        u32 ys
#define KARG_KIND , u32 kind
#define KARG_ROW
#endif
#if defined(FMA_BUILTIN)
#define FMA(a, b, c) __builtin_fmaf(a, b, c)
#define U64 unsigned long long
#endif

#define GPU_BLOCK_WORDS 80 /* 320 bytes */

/* bits [o, o + n) of w */
INLINE int bits(u32 w, u32 o, u32 n) { return (int)((w >> o) & ((1u << n) - 1u)); }
INLINE int lo16(u32 w) { return (int)(short)(w & 0xffffu); }
INLINE int hi16(u32 w) { return (int)(short)(w >> 16); }

/* ---- one block of the weights in blocks of 256 against one block of the
   activations: its scales times the activations' scale, and the exact
   integer sums the fmas take (a, i: the sum; m, j: the minimums' of Q4_K and
   Q5_K, 0 for the others). The row's sum is fma(a, i, acc) block after
   block, and acc - accm with the minimums: as the CPU does it. ---- */

/* Q4_K: 144 bytes a block - d and dmin (halves), 12 bytes of scales and
   mins of 6 bits, 128 of quants in nibbles */
INLINE void blk_q4k(GLOBAL const u32 *W, GLOBAL const u32 *X, PRIV float *a,
                    PRIV int *is, PRIV float *m, PRIV int *im)
{
    u32 h0 = W[0], h1 = W[1], h2 = W[2], h3 = W[3];
    float d = h2f(h0 & 0xffffu), dmin = h2f(h0 >> 16);
    int sc[8], mn[8];
    for (u32 j = 0; j < 4u; j++) {
        sc[j] = bits(h1, 8u * j, 6);
        mn[j] = bits(h2, 8u * j, 6);
        sc[j + 4u] = bits(h3, 8u * j, 4) | (bits(h1, 8u * j + 6u, 2) << 4);
        mn[j + 4u] = bits(h3, 8u * j + 4u, 4) | (bits(h2, 8u * j + 6u, 2) << 4);
    }
    GLOBAL const u32 *q = W + 4u;
    int isum = 0;
    for (u32 c = 0; c < 4u; c++) {
        int lo = 0, hi = 0;
        for (u32 k = 0; k < 8u; k++) {
            u32 qw = q[8u * c + k];
            lo = dp4a(qw & 0x0f0f0f0fu, X[16u * c + k], lo);
            hi = dp4a((qw >> 4) & 0x0f0f0f0fu, X[16u * c + 8u + k], hi);
        }
        isum += lo * sc[2u * c] + hi * sc[2u * c + 1u];
    }
    int imin = 0;
    for (u32 s = 0; s < 8u; s++)
        imin += mn[s] * (lo16(X[64u + s]) + hi16(X[64u + s]));
    float d8 = bits_f(X[72]);
    *a = d * d8;
    *m = dmin * d8;
    *is = isum;
    *im = imin;
}

/* Q5_K: 176 bytes a block - the header of Q4_K, 32 bytes of fifth bits
   (bit s of each byte for sub-block s), 128 of quants in nibbles */
INLINE void blk_q5k(GLOBAL const u32 *W, GLOBAL const u32 *X, PRIV float *a,
                    PRIV int *is, PRIV float *m, PRIV int *im)
{
    u32 h0 = W[0], h1 = W[1], h2 = W[2], h3 = W[3];
    float d = h2f(h0 & 0xffffu), dmin = h2f(h0 >> 16);
    int sc[8], mn[8];
    for (u32 j = 0; j < 4u; j++) {
        sc[j] = bits(h1, 8u * j, 6);
        mn[j] = bits(h2, 8u * j, 6);
        sc[j + 4u] = bits(h3, 8u * j, 4) | (bits(h1, 8u * j + 6u, 2) << 4);
        mn[j + 4u] = bits(h3, 8u * j + 4u, 4) | (bits(h2, 8u * j + 6u, 2) << 4);
    }
    GLOBAL const u32 *qh = W + 4u, *q = W + 12u;
    int isum = 0;
    for (u32 c = 0; c < 4u; c++) {
        u32 s0 = 2u * c, s1 = 2u * c + 1u;
        int lo = 0, hi = 0;
        for (u32 k = 0; k < 8u; k++) {
            u32 qw = q[8u * c + k], hw = qh[k];
            u32 l = (qw & 0x0f0f0f0fu) | (((hw >> s0) & 0x01010101u) << 4);
            u32 h = ((qw >> 4) & 0x0f0f0f0fu) | (((hw >> s1) & 0x01010101u) << 4);
            lo = dp4a(l, X[16u * c + k], lo);
            hi = dp4a(h, X[16u * c + 8u + k], hi);
        }
        isum += lo * sc[s0] + hi * sc[s1];
    }
    int imin = 0;
    for (u32 s = 0; s < 8u; s++)
        imin += mn[s] * (lo16(X[64u + s]) + hi16(X[64u + s]));
    float d8 = bits_f(X[72]);
    *a = d * d8;
    *m = dmin * d8;
    *is = isum;
    *im = imin;
}

/* Q6_K: 210 bytes a block - 128 of the low 4 bits, 64 of the high 2, 16
   scales (int8), d (half) - rows starting on even bytes: read in halves */
INLINE void blk_q6k(GLOBAL const u16 *W, GLOBAL const u32 *X, PRIV float *a,
                    PRIV int *is, PRIV float *m, PRIV int *im)
{
    u32 wd[53];
    for (u32 k = 0; k < 52u; k++)
        wd[k] = (u32)W[2u * k] | ((u32)W[2u * k + 1u] << 16);
    wd[52] = W[104];
    float d = h2f(wd[52]);
    int sc[16];
    for (u32 k = 0; k < 4u; k++)
        for (u32 j = 0; j < 4u; j++)
            sc[4u * k + j] = (int)(signed char)(u8)(wd[48u + k] >> (8u * j));
    int isum = 0;
    for (u32 n = 0; n < 2u; n++)
        for (u32 part = 0; part < 4u; part++) {
            u32 qlw = 16u * n + (part & 1u) * 8u, qhw = 32u + 8u * n;
            u32 lsh = (part >> 1) * 4u, hsh = part * 2u;
            u32 xa = 4u * (8u * n + 2u * part);
            int dot0 = 0, dot1 = 0;
            for (u32 k = 0; k < 8u; k++) {
                u32 q = ((wd[qlw + k] >> lsh) & 0x0f0f0f0fu) |
                        (((wd[qhw + k] >> hsh) & 0x03030303u) << 4);
                if (k < 4u)
                    dot0 = dp4a(q, X[xa + k], dot0);
                else
                    dot1 = dp4a(q, X[xa + k], dot1);
            }
            u32 g = 8u * n + 2u * part;
            isum += sc[g] * dot0 + sc[g + 1u] * dot1;
        }
    int off32 = 0;
    for (u32 s = 0; s < 8u; s++)
        off32 += sc[2u * s] * lo16(X[64u + s]) + sc[2u * s + 1u] * hi16(X[64u + s]);
    isum -= 32 * off32;
    *a = d * bits_f(X[72]);
    *is = isum;
    *m = 0.0f;
    *im = 0;
}

/* Q3_K: 110 bytes a block - 32 of the high bits (hmask), 64 of the low two
   (qs), 12 of 6-bit scales, d (half) - rows starting on even bytes */
INLINE void blk_q3k(GLOBAL const u16 *W, GLOBAL const u32 *X, PRIV float *a,
                    PRIV int *is, PRIV float *m, PRIV int *im)
{
    u32 wd[28];
    for (u32 k = 0; k < 27u; k++)
        wd[k] = (u32)W[2u * k] | ((u32)W[2u * k + 1u] << 16);
    wd[27] = W[54];
    float d = h2f(wd[27]);
    u32 a0 = wd[24], a1 = wd[25], a2 = wd[26];
    u32 o[4] = {(a0 & 0x0f0f0f0fu) | ((a2 & 0x03030303u) << 4),
                (a1 & 0x0f0f0f0fu) | (((a2 >> 2) & 0x03030303u) << 4),
                ((a0 >> 4) & 0x0f0f0f0fu) | (((a2 >> 4) & 0x03030303u) << 4),
                ((a1 >> 4) & 0x0f0f0f0fu) | (((a2 >> 6) & 0x03030303u) << 4)};
    int sc[16];
    for (u32 i = 0; i < 16u; i++)
        sc[i] = (int)((o[i / 4u] >> (8u * (i % 4u))) & 0xffu) - 32;
    int isum = 0;
    for (u32 n = 0; n < 2u; n++)
        for (u32 j = 0; j < 4u; j++)
            for (u32 h = 0; h < 2u; h++) {
                u32 gq = 8u + 8u * n + 4u * h, gh = 4u * h;
                u32 lsh = 2u * j, hsh = j + 4u * n;
                u32 g = 8u * n + 2u * j + h;
                int dot = 0;
                for (u32 i = 0; i < 4u; i++) {
                    u32 q = ((wd[gq + i] >> lsh) & 0x03030303u) |
                            (((wd[gh + i] >> hsh) & 0x01010101u) << 2);
                    dot = dp4a(q, X[4u * g + i], dot);
                }
                int bsum = g % 2u ? hi16(X[64u + g / 2u]) : lo16(X[64u + g / 2u]);
                isum += sc[g] * (dot - 4 * bsum);
            }
    *a = d * bits_f(X[72]);
    *is = isum;
    *m = 0.0f;
    *im = 0;
}

/* IQ4_XS: 136 bytes a block - d (half), the high two bits of the eight
   scales (half), their low four (4 bytes), 128 of nibbles through IQ4_NL's
   table; scales of 6 bits less 32, one a block of 32 */
INLINE u32 iq4_table(u32 w) /* each byte's nibble through IQ4_NL's levels */
{
    const int IQ4[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                         1,    13,   25,  38,  53,  69,  89,  113};
    u32 r = 0;
    for (u32 k = 0; k < 32u; k += 8u)
        r |= ((u32)IQ4[(w >> k) & 15u] & 0xffu) << k;
    return r;
}
INLINE void blk_iq4xs(GLOBAL const u32 *W, GLOBAL const u32 *X, PRIV float *a,
                      PRIV int *is, PRIV float *m, PRIV int *im)
{
    u32 w0 = W[0], sl = W[1], sh = w0 >> 16;
    float d = h2f(w0 & 0xffffu);
    int isum = 0;
    for (u32 g = 0; g < 8u; g++) {
        u32 ls = ((sl >> (8u * (g / 2u) + 4u * (g & 1u))) & 15u) |
                 (((sh >> (2u * g)) & 3u) << 4);
        int dot = 0;
        for (u32 i = 0; i < 4u; i++) {
            u32 q = W[2u + 4u * g + i];
            dot = dp4a(iq4_table(q & 0x0f0f0f0fu), X[8u * g + i], dot);
            dot = dp4a(iq4_table((q >> 4) & 0x0f0f0f0fu), X[8u * g + 4u + i], dot);
        }
        isum += ((int)ls - 32) * dot;
    }
    *a = d * bits_f(X[72]);
    *is = isum;
    *m = 0.0f;
    *im = 0;
}

/* one thread a row: y = sum of fma(a, i) over the blocks, less the
   minimums' (DM) - the kernels the CPU runs too, to check them */
#define ROW_KERNEL(name, WT, WORDS, BLK, DM)                                   \
    KERNEL name(KARGS(WT) KARG_ROW)                                                \
    {                                                                          \
        u32 row = row_index();                                                 \
        if (row >= rows)                                                       \
            return;                                                            \
        for (u32 v = 0; v < nv; v++) {                                         \
            GLOBAL const WT *W = w + (U64)row * nb * WORDS;            \
            GLOBAL const u32 *X = x + (U64)v * xs * GPU_BLOCK_WORDS;   \
            float acc = 0.0f, accm = 0.0f;                                     \
            for (u32 b = 0; b < nb; b++, W += WORDS, X += GPU_BLOCK_WORDS) {   \
                float a, m;                                                    \
                int i, j;                                                      \
                BLK(W, X, &a, &i, &m, &j);                                     \
                acc = FMA(a, (float)i, acc);                        \
                if (DM)                                                        \
                    accm = FMA(m, (float)j, accm);                  \
            }                                                                  \
            y[(U64)v * ys + row] = DM ? acc - accm : acc;       \
        }                                                                      \
    }

ROW_KERNEL(q4k_matvec, u32, 36u, blk_q4k, 1)
ROW_KERNEL(q5k_matvec, u32, 44u, blk_q5k, 1)
ROW_KERNEL(q6k_matvec, u16, 105u, blk_q6k, 0)
ROW_KERNEL(q3k_matvec, u16, 55u, blk_q3k, 0)
ROW_KERNEL(iq4xs_matvec, u32, 34u, blk_iq4xs, 0)

#if defined(__NVPTX__) || defined(__AMDGPU__)
/* 32 threads a row, 4 rows a block of 128: each thread the terms of its
   blocks (b = lane, lane + 32, ...) into the block's shared memory, then
   one thread the row's fmas in the order of the blocks - the same chain
   as above, the same bits - with the reading of the weights spread over
   the row's threads, and a card filled by rows of a few thousands. Rows of
   WARP_NB blocks at most. */
#define WARP_NB 256u
#define WARP_KERNEL(name, WT, WORDS, BLK, DM)                                  \
    KERNEL name(KARGS(WT) KARG_ROW)                                                \
    {                                                                          \
        SHARED float sa[4][WARP_NB], sm[4][WARP_NB];                            \
        SHARED int si[4][WARP_NB], sj[4][WARP_NB];                              \
        u32 tid = thread_index(), lane = tid & 31u, wi = tid >> 5;             \
        u32 row = block_index() * 4u + wi;                                     \
        for (u32 v = 0; v < nv; v++) {                                         \
            if (row < rows)                                                    \
                for (u32 b = lane; b < nb; b += 32u)                           \
                    BLK(w + ((U64)row * nb + b) * WORDS,        \
                        x + ((U64)v * xs + b) * GPU_BLOCK_WORDS,\
                        &sa[wi][b], &si[wi][b], &sm[wi][b], &sj[wi][b]);       \
            barrier();                                                         \
            if (row < rows && lane == 0u) {                                    \
                float acc = 0.0f, accm = 0.0f;                                 \
                for (u32 b = 0; b < nb; b++) {                                 \
                    acc = FMA(sa[wi][b], (float)si[wi][b], acc);    \
                    if (DM)                                                    \
                        accm = FMA(sm[wi][b], (float)sj[wi][b],     \
                                              accm);                           \
                }                                                              \
                y[(U64)v * ys + row] = DM ? acc - accm : acc;   \
            }                                                                  \
            barrier();                                                         \
        }                                                                      \
    }

WARP_KERNEL(q4k_matvec_w, u32, 36u, blk_q4k, 1)
WARP_KERNEL(q5k_matvec_w, u32, 44u, blk_q5k, 1)
WARP_KERNEL(q6k_matvec_w, u16, 105u, blk_q6k, 0)
WARP_KERNEL(q3k_matvec_w, u16, 55u, blk_q3k, 0)
WARP_KERNEL(iq4xs_matvec_w, u32, 34u, blk_iq4xs, 0)
#endif





/* Q8_0: blocks of 32 - an fp16 scale and 32 int8 - eight to 256 weights
   (272 bytes, a block starting every 34 bytes, on even bytes: read in
   halves), as janas_q8_0_dot_ref computes them: each block's integer dot
   in one fma into the accumulator of its place in the group of eight, the
   eight added at the end as janas_hsum8 does. The activations in blocks of
   32 with a scale each (the Q8_K block's qs32 and d32: gpu_cuda.c converts
   them so for this type). Rows of whole groups of 256. */
KERNEL q80_matvec(KARGS(u16) KARG_ROW)
{
    u32 row = row_index();
    if (row >= rows)
        return;
    for (u32 v = 0; v < nv; v++) {
        GLOBAL const u16 *W = w + (U64)row * nb * 136u;
        GLOBAL const u32 *X = x + (U64)v * xs * GPU_BLOCK_WORDS;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (u32 b = 0; b < nb; b++, W += 136u, X += GPU_BLOCK_WORDS)
            for (u32 k = 0; k < 8u; k++) {
                GLOBAL const u16 *B = W + 17u * k;
                float d = h2f(B[0]);
                int dot = 0;
                for (u32 i = 0; i < 8u; i++)
                    dot = dp4a((u32)B[1u + 2u * i] | ((u32)B[2u + 2u * i] << 16),
                               X[8u * k + i], dot);
                float s = d * bits_f(X[72u + k]);
                acc[k] = FMA(s, (float)dot, acc[k]);
            }
        float a04 = acc[0] + acc[4], a26 = acc[2] + acc[6];
        float a15 = acc[1] + acc[5], a37 = acc[3] + acc[7];
        float e = a04 + a26, o = a15 + a37;
        y[(U64)v * ys + row] = e + o;
    }
}

/* ---- the weights in blocks of 32 (b32_matvec.comp), kind given:
   0 Q4_0 (q - 8), 1 IQ4_NL (a table of sixteen levels), 2 Q4_1 and 3 Q5_1
   (d q + m, Q5_1 with a fifth bit a weight in qh), 4 MXFP4 (an E8M0 byte,
   then the nibbles through MXFP4's table), 5 Q5_0 (d, qh, nibbles; q - 16),
   6 Q2_0 (18 bytes for 64 weights: d, two bits a weight; q - 1), 7 NVFP4
   (36 bytes for 64 weights: four UE4M3 scales, sub-block s's codes in bytes
   8s to 8s + 7) - each block's integer dot in one fma into the accumulator
   of its place in the group of eight, Q4_1/Q5_1's minimum in accumulators
   of their own, the eight added as janas_hsum8 does. Rows of whole groups
   of 256; the activations in blocks of 32 (the Q8_K's qs32, bsums32, d32).
   The weights are read by the byte: blocks start anywhere. ---- */

INLINE u32 le32(GLOBAL const u8 *p)
{
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}
INLINE float ldexpf_exact(float m, int e) /* m 2^e, m small, e in range */
{
    u32 b = (u32)(e + 127) << 23;
    return m * bits_f(b);
}
/* NVFP4's UE4M3 scale times one half (quant.c janas_ue4m3_half) */
INLINE float ue4m3_half(u32 x)
{
    if (x == 0u || x == 0x7fu)
        return 0.0f;
    int e = (int)((x >> 3) & 15u), man = (int)(x & 7u);
    return e == 0 ? ldexpf_exact((float)man, -10)
                  : ldexpf_exact((float)(8 + man), e - 11);
}
/* MXFP4's E8M0 scale, 2^(e - 128) (quant.c janas_e8m0_half) */
INLINE float e8m0_half(u32 e)
{
    return bits_f(e < 2u ? 0x00200000u << e : (e - 1u) << 23);
}
INLINE u32 spread5(u32 b)
{
    return ((b & 1u) << 4) | ((b & 2u) << 11) | ((b & 4u) << 18) | ((b & 8u) << 25);
}
INLINE u32 spread2(u32 b)
{
    u32 t = (b | (b << 12)) & 0x000f000fu;
    return (t | (t << 6)) & 0x03030303u;
}
INLINE u32 table4(u32 w, u32 kind) /* each byte's nibble through the table */
{
    const int IQ4[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                         1,    13,   25,  38,  53,  69,  89,  113};
    const int MX[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    u32 r = 0;
    for (u32 k = 0; k < 32u; k += 8u) {
        u32 n = (w >> k) & 15u;
        r |= ((u32)(kind == 1u ? IQ4[n] : MX[n]) & 0xffu) << k;
    }
    return r;
}
INLINE float hsum8(PRIV const float *a)
{
    float a04 = a[0] + a[4], a26 = a[2] + a[6];
    float a15 = a[1] + a[5], a37 = a[3] + a[7];
    float e = a04 + a26, o = a15 + a37;
    return e + o;
}

KERNEL b32_matvec(KARGS(u8) KARG_KIND KARG_ROW)
{
    u32 row = row_index();
    if (row >= rows)
        return;
    /* bytes of a block of 32 (Q2_0: half a block) */
    u32 S = kind == 7u ? 18u : kind == 6u ? 9u : kind == 5u ? 22u
          : kind == 4u ? 17u : kind == 3u ? 24u : kind == 2u ? 20u : 18u;
    u32 GB = 8u * S;
    int dm = kind == 2u || kind == 3u, tab = kind == 1u || kind == 4u || kind == 7u;
    int five = kind == 3u || kind == 5u;
    int off = kind == 0u ? 8 : kind == 5u ? 16 : kind == 6u ? 1 : 0;
    for (u32 v = 0; v < nv; v++) {
        GLOBAL const u8 *G = w + (U64)row * nb * GB;
        GLOBAL const u32 *X = x + (U64)v * xs * GPU_BLOCK_WORDS;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}, accm[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (u32 b = 0; b < nb; b++, G += GB, X += GPU_BLOCK_WORDS)
            for (u32 k = 0; k < 8u; k++) {
                u32 B = kind == 6u || kind == 7u ? 2u * S * (k / 2u) : S * k;
                u32 qb = kind == 6u   ? B + 2u + 8u * (k % 2u)
                         : kind == 7u ? B + 4u + 16u * (k % 2u)
                                      : B + S - 16u;
                float d, m = 0.0f;
                if (kind == 7u) {
                    u32 sb = B + 2u * (k % 2u);
                    d = ue4m3_half(G[sb]);
                    m = ue4m3_half(G[sb + 1u]);
                } else if (kind == 4u)
                    d = e8m0_half(G[B]);
                else {
                    d = h2f((u32)G[B] | ((u32)G[B + 1u] << 8));
                    if (dm)
                        m = h2f((u32)G[B + 2u] | ((u32)G[B + 3u] << 8));
                }
                u32 lo[4], hi[4];
                if (kind == 6u) {
                    u32 q0 = le32(G + qb), q1 = le32(G + qb + 4u);
                    for (u32 j = 0; j < 4u; j++) {
                        lo[j] = spread2((q0 >> (8u * j)) & 0xffu);
                        hi[j] = spread2((q1 >> (8u * j)) & 0xffu);
                    }
                } else {
                    u32 qh = kind == 3u ? le32(G + B + 4u)
                             : kind == 5u ? le32(G + B + 2u) : 0u;
                    for (u32 i = 0; i < 4u; i++) {
                        u32 q = le32(G + qb + 4u * i);
                        u32 l = q & 0x0f0f0f0fu, h = (q >> 4) & 0x0f0f0f0fu;
                        if (tab) {
                            l = table4(l, kind);
                            h = table4(h, kind);
                        } else if (five) {
                            l |= spread5((qh >> (4u * i)) & 15u);
                            h |= spread5((qh >> (16u + 4u * i)) & 15u);
                        }
                        if (kind == 7u) { /* words 0-1 the first sub-block */
                            if (i < 2u) { lo[i] = l; lo[i + 2u] = h; }
                            else { hi[i - 2u] = l; hi[i] = h; }
                        } else {
                            lo[i] = l;
                            hi[i] = h;
                        }
                    }
                }
                float d32 = bits_f(X[72u + k]);
                int dl = 0, dh = 0;
                for (u32 i = 0; i < 4u; i++) {
                    dl = dp4a(lo[i], X[8u * k + i], dl);
                    dh = dp4a(hi[i], X[8u * k + 4u + i], dh);
                }
                if (kind == 7u) {
                    float s0 = d * d32, s1 = m * d32;
                    acc[k] = FMA(s0, (float)dl, acc[k]);
                    acc[k] = FMA(s1, (float)dh, acc[k]);
                    continue;
                }
                int dot = dl + dh;
                int asum = lo16(X[64u + k]) + hi16(X[64u + k]);
                if (off)
                    dot -= off * asum;
                float s = d * d32;
                acc[k] = FMA(s, (float)dot, acc[k]);
                if (dm) {
                    float sm = m * d32;
                    accm[k] = FMA(sm, (float)asum, accm[k]);
                }
            }
        float r = hsum8(acc);
        if (dm)
            r = r + hsum8(accm);
        y[(U64)v * ys + row] = r;
    }
}

