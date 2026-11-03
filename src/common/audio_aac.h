/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_aac.h - what the parts of the AAC decoder share (ISO/IEC 14496-3):
 * the bits, read most significant first, the Huffman tables, the core
 * (audio_aac.c: AAC Main, LC and LTP), spectral band replication
 * (audio_aac_sbr.c) and parametric stereo (audio_aac_ps.c). Inside
 * src/common.
 */
#ifndef JANAS_COMMON_AUDIO_AAC_H
#define JANAS_COMMON_AUDIO_AAC_H

#include <stddef.h>
#include <stdint.h>

#include "audio_ogg.h"

/* ---- bits ---- */

struct abr {
    const uint8_t *p;
    size_t n;   /* bytes */
    size_t pos; /* bits */
};

/* the next 32 bits, zeros past the end, without taking them */
static inline uint32_t abr_peek(const struct abr *b)
{
    size_t at = b->pos >> 3;
    uint64_t w = 0;
    for (int i = 0; i < 8; i++)
        w = w << 8 | (at + (size_t)i < b->n ? b->p[at + (size_t)i] : 0);
    return (uint32_t)((w << (b->pos & 7)) >> 32);
}

/* k bits, 0 to 32 */
static inline uint32_t abr_bits(struct abr *b, int k)
{
    if (!k)
        return 0;
    uint32_t v = abr_peek(b) >> (32 - k);
    b->pos += (size_t)k;
    return v;
}

static inline int abr_bit(struct abr *b)
{
    return (int)abr_bits(b, 1);
}

/* read past the end */
static inline int abr_over(const struct abr *b)
{
    return b->pos > b->n * 8;
}

static inline size_t abr_left(const struct abr *b)
{
    return b->pos < b->n * 8 ? b->n * 8 - b->pos : 0;
}

/* ---- Huffman codes ---- */

struct vlc_e {
    int16_t sym; /* or the subtable's start */
    int8_t len;  /* bits taken; < 0: a subtable of -len bits; 0: no code */
};

struct vlc {
    struct vlc_e *t;
    int bits;
};

/* n codes (code[i] of len[i] bits, MSB first) for the symbols sym[i] (or
   i when sym is NULL): 0, or -1 when out of memory or not a prefix code */
int vlc_build(struct vlc *v, int bits, const uint32_t *code, const uint8_t *len,
              const uint8_t *sym, int n);
void vlc_free(struct vlc *v);

/* the next symbol, or -1 (a code that is none) */
static inline int vlc_get(struct abr *b, const struct vlc *v)
{
    uint32_t w = abr_peek(b);
    const struct vlc_e *e = &v->t[w >> (32 - v->bits)];
    int used = 0;
    if (e->len < 0) {
        used = v->bits;
        e = &v->t[e->sym + (int)((w << used) >> (32 + e->len))];
    }
    if (e->len <= 0)
        return -1;
    b->pos += (size_t)(used + e->len);
    return e->sym;
}

/* ---- the configuration ---- */

enum {
    AOT_MAIN = 1,
    AOT_LC = 2,
    AOT_SSR = 3,
    AOT_LTP = 4,
    AOT_SBR = 5,
    AOT_ER_LC = 17, /* error resilient */
    AOT_ER_LD = 23, /* low delay */
    AOT_PS = 29,
    AOT_ER_ELD = 39, /* enhanced low delay */
    AOT_USAC = 42,   /* xHE-AAC */
};

struct aac_cfg {
    int aot;        /* the core's: 1, 2, 4, 17, 23, 39 or 42 */
    int rate_index; /* the core's sampling frequency index, 0 to 12 */
    int rate;
    int chan_config; /* 0: a program config element says */
    int frame;       /* 1024 or 960; low delay 512 or 480; USAC 1024, 768 */
    int sbr;         /* 1 said present, 0 said absent, -1 not said */
    int ps;          /* the same */
    int sbr_rate;    /* the output's rate with SBR */
    int channels;    /* from the configuration, 0 when unknown */
    /* USAC: the elements of each frame, in order */
    int usac_n;
    struct usac_el {
        uint8_t type; /* 0 SCE, 1 CPE, 2 LFE, 3 extension */
        uint8_t noise_fill, frag;
        uint32_t default_len; /* an extension's payload, 0 none */
    } usac_el[64];
};

/* the AudioSpecificConfig of an MP4 file or of Matroska: 0, or -1 with
   why */
int aac_config(const uint8_t *p, size_t n, struct aac_cfg *c, char *why,
               size_t wn);
/* the rate of a sampling frequency index, 0 when none */
int aac_index_rate(int index);
int aac_rate_index(int rate);

struct aac_dec;
struct aac_dec *aac_new(const struct aac_cfg *c);
void aac_free(struct aac_dec *d);
/* one raw_data_block of n bytes: its samples a channel (the frame, or
   twice it with SBR), or -1 with why; the channels and their samples in
   aac_out */
int aac_frame(struct aac_dec *d, const uint8_t *p, size_t n, char *why,
              size_t wn);
int aac_out(const struct aac_dec *d, float ***pcm);
/* the output's rate: the core's, or twice it once SBR was found */
int aac_out_rate(const struct aac_dec *d);

/* ELD's low-delay synthesis (audio_aac_eld.c, 4.6.20.2): n coefficients
   (512 or 480, overwritten) to n samples, the three frames before kept in
   hist (3n, zeros at first); y and z: 2n and 4n of room */
struct aimdct;
void aac_eld_synth(struct aimdct *m, int n, float *coef, float *hist,
                   float *out, float *y, float *z);

/* a whole stream: the packets (raw data blocks, data + off[i], len[i]),
   trimmed as the edit says (NULL: not at all); mixed to mono into a */
int aac_stream(const struct aac_cfg *c, const uint8_t *data, const size_t *off,
               const size_t *len, size_t count, const struct audio_edit *edit,
               struct audio_out *ao, char *why, size_t wn);

/* ---- spectral band replication (audio_aac_sbr.c) ---- */

struct sbr;
/* for an element of ch channels (1, 2) of a core at rate; down: the
   output at the core's rate (downsampled SBR) */
struct sbr *sbr_new(int rate, int ch, int down);
void sbr_free(struct sbr *s);
/* the sbr_extension_data of an extension payload of cnt bytes at b (its
   type, 13 or 14, read): 0, or -1 */
int sbr_read(struct sbr *s, struct abr *b, int crc, int cnt);
/* 1024 core samples of each channel to 2048 (1024 downsampled): in[ch],
   out[ch]; with PS a mono element makes two channels (out[0], out[1]) */
void sbr_apply(struct sbr *s, int ch, float *const *in, float **out);
/* parametric stereo was found in the SBR data */
int sbr_has_ps(const struct sbr *s);

/* ---- parametric stereo (audio_aac_ps.c) ---- */

struct ps;
struct ps *ps_new(void);
void ps_free(struct ps *p);
/* ps_data from an SBR extension of bits bits: the bits read, or -1 */
int ps_read(struct ps *p, struct abr *b, int bits);
/* the QMF subbands of a mono signal (l: re and im planes of 38 slots of
   64 bands) into two, l and r; top: the bands SBR fills */
void ps_apply(struct ps *p, float (*l)[38][64], float (*r)[38][64], int top);

#endif
