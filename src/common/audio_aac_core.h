/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_aac_core.h - the AAC core decoder's state (audio_aac.c), shared
 * with USAC's (audio_aac_usac.c), which reads its own syntax and spectra
 * and hands them to the same tools: TNS, mid/side, the inverse MDCT and
 * its windows. Inside src/common.
 */
#ifndef JANAS_COMMON_AUDIO_AAC_CORE_H
#define JANAS_COMMON_AUDIO_AAC_CORE_H

#include "audio_aac.h"

enum { ONLY_LONG, LONG_START, EIGHT_SHORT, LONG_STOP };
enum { ZERO_BT = 0, NOISE_BT = 13, INTENSITY_BT2 = 14, INTENSITY_BT = 15 };

#define MAX_ELEMENTS 16
#define MAX_CHANNELS 48
#define MAX_PRED 672
#define MAX_LTP_SFB 40

struct pred {
    float r0, r1, cor0, cor1, var0, var1;
};

struct ics {
    int seq, shape, prev_shape, max_sfb;
    int groups, group_len[8];
    int num_swb, num_windows;
    const uint16_t *swb; /* num_swb + 1 offsets */
    int pred_present, pred_reset, pred_used[41];
    int ltp_present, ltp_lag;
    float ltp_coef;
    int ltp_used[MAX_LTP_SFB];
};

struct tns {
    int present;
    int n_filt[8];
    int length[8][4], order[8][4], dir[8][4];
    float coef[8][4][20];
};

/* a channel's state */
struct chan {
    struct ics ics;
    struct tns tns;
    uint8_t cb[128];  /* the codebook of each window group x band */
    int sf[128];      /* scalefactors, positions, noise energies */
    float coef[1024]; /* the spectrum */
    float overlap[1024];
    float ltp_state[3072];
    float eld[1536]; /* ELD's three frames before */
    struct pred pred[MAX_PRED];
    int pred_init;
    float out[2048];
    struct usac_ch *u; /* USAC's state (audio_aac_usac.c) */
};

enum { BEFORE_TNS, BETWEEN_TNS_AND_IMDCT, AFTER_IMDCT = 3 };

/* a coupling channel element's targets and gains (4.4.2.5, 4.6.8.3) */
struct cce {
    int point, num_coupled;
    int type[8], id[8], ch_select[8];
    float gain[16][128]; /* [list][band], [list][0] after the IMDCT */
    float out[2048];     /* its samples, for coupling after the IMDCT */
};

struct element {
    int type; /* 0 SCE, 1 CPE, 2 CCE, 3 LFE */
    int tag;
    int first; /* its first channel */
    int ch;
    struct sbr *sbr;
    struct cce *cce;
    struct usac_st *us; /* a USAC pair's stereo state */
};

struct aac_dec {
    struct aac_cfg cfg;
    struct vlc sf_vlc, spec_vlc[11];
    struct aimdct *mdct_long, *mdct_short;
    float sine_long[1024], sine_short[128];
    float kbd_long[1024], kbd_short[128];
    float pow43[8192];
    float buf[2048];
    float fwd[2048];
    uint32_t noise_state;
    int frame, short_len;
    struct element el[MAX_ELEMENTS];
    int n_el;
    struct chan *ch[MAX_CHANNELS];
    int n_ch;
    int sbr_on;   /* -1 until the first frame says */
    int sbr_down; /* the output at the core's rate */
    float *outp[MAX_CHANNELS];
    int out_ch;
};

struct usac_ch;
struct usac_st;

/* the scalefactor bands of a window of frame samples at the rate index */
const uint16_t *aac_swb_table(int frame, int index, int *num);
/* tns_data (4.6.9): 0, or -1 */
int aac_read_tns(struct aac_dec *d, struct abr *b, const struct ics *ics,
                 struct tns *t);
void aac_apply_tns(struct aac_dec *d, float *coef, const struct tns *t,
                   const struct ics *ics, int inverse);
/* mid/side on the bands whose ms[] is set (by group x 16 bands when short) */
void aac_apply_ms(struct aac_dec *d, struct chan *l, struct chan *r,
                  const uint8_t *ms);
/* the inverse MDCT, windows and overlap into c->out */
void aac_synthesize(struct aac_dec *d, struct chan *c);
/* the element of type (0 SCE, 1 CPE, 3 LFE) and tag, made at its first
   sight; NULL past the elements or channels it holds */
struct element *aac_element(struct aac_dec *d, int type, int tag);

/* USAC (audio_aac_usac.c): UsacConfig after the AOT, rate and channel
   configuration fields, which it reads itself; a frame; 0 or -1 with why,
   the frame's samples a channel */
int usac_config(struct abr *b, struct aac_cfg *c, char *why, size_t wn);
int usac_frame(struct aac_dec *d, struct abr *b, char *why, size_t wn);

#endif
