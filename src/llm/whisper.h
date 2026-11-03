/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * whisper.h - Whisper, speech to text (OpenAI's model, from a JNS file made
 * by tools/hf2jns_whisper): the log-mel spectrogram of the sound
 * (whisper_mel.c), the encoder over thirty seconds of it and the decoder a
 * token at a time (whisper_net.c), the transcription of a whole recording
 * window by window, with its times (whisper_run.c). Inside the library.
 */
#ifndef JANAS_LLM_WHISPER_H
#define JANAS_LLM_WHISPER_H

#include <stddef.h>
#include <stdint.h>

#include "jns.h"

struct janas_pool;
struct janas_tokenizer;

/* a block's tensors: matrices in half precision, transposed (a row per
   input), the rest float; x* the attention over the encoder's output */
struct wlayer {
    const float *ln_w, *ln_b, *q_b, *v_b, *o_b;
    const uint16_t *q, *k, *v, *o;
    const float *xln_w, *xln_b, *xq_b, *xv_b, *xo_b;
    const uint16_t *xq, *xk, *xv, *xo;
    const float *mln_w, *mln_b, *fc1_b, *fc2_b;
    const uint16_t *fc1, *fc2;
};

struct janas_whisper {
    struct janas_jns j;
    uint8_t *data; /* the resident tensors */
    uint32_t dim, a_layers, a_heads, a_ffn, a_ctx;
    uint32_t t_layers, t_heads, t_ffn, t_ctx, vocab;
    uint32_t n_mels, n_fft, hop, rate, chunk; /* chunk: seconds a window */
    /* the special tokens */
    int32_t eot, sot, translate, transcribe, sot_lm, sot_prev, no_speech;
    int32_t no_timestamps, ts_begin;
    uint32_t n_lang;      /* languages, their tokens after sot */
    const uint8_t *langs; /* their codes (GGUF strings) */
    int multilingual;
    const int32_t *suppress, *begin_suppress;
    uint64_t n_suppress, n_begin_suppress;
    uint32_t max_initial_ts; /* the first timestamp at most, its index */
    /* the decoder's heads whose attention over the sound follows the
       words (layer, head), from the model or OpenAI's default */
    int32_t *aheads;
    uint32_t n_aheads;
    /* the spectrogram */
    const float *mel_filters; /* n_mels x (n_fft / 2 + 1) */
    float *dft_cos, *dft_sin; /* (n_fft / 2 + 1) x n_fft, the window in */
    /* the encoder */
    const uint16_t *conv1, *conv2;
    const float *conv1_b, *conv2_b, *e_pos, *e_ln_w, *e_ln_b;
    struct wlayer *enc, *dec;
    /* the decoder */
    const uint16_t *tok; /* dim x vocab: the token embeddings transposed */
    const float *d_pos, *d_ln_w, *d_ln_b;
    struct janas_tokenizer *tk;
    struct janas_pool *pool;
    /* the state: the encoder's keys and values for each decoder block
       (t_layers x a_ctx x dim), the decoder's own (t_layers x t_ctx x
       dim), the buffers of a pass */
    float *xk, *xv, *kc, *vc;
    float *x, *h, *q, *att, *ff; /* t_ctx rows each (ff: t_ctx x t_ffn) */
    /* when set, a decoder pass leaves the attention scores (before the
       softmax) of the alignment heads over the sound: n_aheads x n x a_ctx */
    float *cap;
};

/* whisper.c: the model opened with n_threads (0: the machine's cores), or
   NULL with the reason in err */
struct janas_whisper *janas_whisper_open(const char *path, int n_threads,
                                         char *err, size_t err_len);
void janas_whisper_close(struct janas_whisper *w);
/* the code of language i ("en"), into buf */
void janas_whisper_lang(const struct janas_whisper *w, uint32_t i, char *buf,
                        size_t cap);

/* whisper_mel.c: the normalized log-mel spectrogram of n samples (mono, at
   w->rate) followed by pad zeros: *frames = (n + pad) / hop columns,
   out[m * *frames + f] (malloc'd), or NULL when out of memory */
float *janas_whisper_mel(struct janas_whisper *w, const float *pcm, size_t n,
                         size_t pad, uint32_t *frames);

/* A recording too long to hold: n samples (mono, at w->rate) read from a
   source going forward - samples [from, from + count) within the n, valid
   until the next call, which never asks for any before from; NULL with
   err when they cannot be read - and read again from the start after
   rewind. */
struct janas_whisper_sound {
    size_t n;
    const float *(*get)(void *user, size_t from, size_t count, char *err,
                        size_t err_len);
    int (*rewind)(void *user, char *err, size_t err_len);
    void *user;
};
/* the largest log10 of the whole spectrogram (the sound and pad zeros),
   read in one pass: 0, or -1 with err */
int janas_whisper_mel_max(struct janas_whisper *w,
                          const struct janas_whisper_sound *snd, size_t pad,
                          float *mx, char *err, size_t err_len);
/* frames f0 .. f0 + count of the spectrogram, as janas_whisper_mel gives
   them with mx its largest, into out (rows of stride): 0, or -1 with err */
int janas_whisper_mel_window(struct janas_whisper *w,
                             const struct janas_whisper_sound *snd,
                             size_t pad, float mx, uint32_t f0, uint32_t count,
                             float *out, uint32_t stride, char *err,
                             size_t err_len);

/* whisper_net.c: the encoder over a window of the spectrogram (rows of
   stride columns): 2 a_ctx columns from off, those from off + len on taken
   as zeros; it leaves the keys and values of every decoder block over it.
   0 or -1 (out of memory) */
int janas_whisper_encode(struct janas_whisper *w, const float *mel,
                         uint32_t stride, uint32_t off, uint32_t len);
/* the decoder over n tokens at positions past .. past + n: the logits of
   the last into logits (vocab), or of each (n x vocab) when all; 0 or -1
   (past + n beyond the context) */
int janas_whisper_decode(struct janas_whisper *w, const int32_t *tok,
                         uint32_t n, uint32_t past, float *logits, int all);

/* whisper_run.c: a whole recording transcribed */
struct janas_whisper_opts {
    const char *language; /* its code ("it"); NULL or "": found out */
    int translate;        /* into English */
    const char *prompt;   /* text that comes before (names, a style) */
    float temperature;    /* the first tried; 0: the most likely tokens,
                             higher ones only where those repeat themselves
                             or are unlikely */
    uint64_t seed;        /* of the sampling at temperatures above 0 */
    volatile int *stop;   /* set to stop between windows (may be NULL) */
    int words;            /* the words' times, as OpenAI's finds them */
    /* the stretches of the recording to read (pairs of samples, start and
       end, in order; a voice activity detector's), or none: all of it */
    const size_t *regions;
    size_t n_regions;
    /* called with each segment once it is final (its text, times, words
       readable in the result being written), in order */
    void (*on_segment)(void *user, uint32_t index);
    void *user;
};

struct janas_whisper_word {
    double t0, t1; /* seconds */
    char *text;    /* with its leading space, punctuation joined */
    float probability;
};

struct janas_whisper_seg {
    double t0, t1; /* seconds */
    char *text;
    int32_t *tokens; /* the text's */
    float *logprobs; /* each token's log-probability */
    uint32_t n_tokens;
    float avg_logprob, no_speech_prob, temperature, compression_ratio;
    struct janas_whisper_word *words; /* when asked for */
    uint32_t n_words;
};

struct janas_whisper_result {
    char language[16]; /* its code */
    double duration;   /* seconds of sound */
    struct janas_whisper_seg *segs;
    uint32_t n_segs;
    char *text; /* the segments' text together, trimmed */
};

/* n samples (mono, at w->rate): 0, or -1 with the reason in err (also
   when stopped) */
int janas_whisper_transcribe(struct janas_whisper *w, const float *pcm,
                             size_t n, const struct janas_whisper_opts *o,
                             struct janas_whisper_result *r, char *err,
                             size_t err_len);
/* the same, read from a source: the spectrogram a window at a time, its
   largest value from a pass before (the source rewound after it); the
   same result */
int janas_whisper_transcribe_sound(struct janas_whisper *w,
                                   const struct janas_whisper_sound *snd,
                                   const struct janas_whisper_opts *o,
                                   struct janas_whisper_result *r, char *err,
                                   size_t err_len);
void janas_whisper_result_free(struct janas_whisper_result *r);

/* whisper_align.c: the times of the words of segments segs[first..n) of
   the window encoded last (from offset seconds, frames of the sound in
   it), as OpenAI's timing.py finds them (cross-attention of the alignment
   heads, dynamic time warping), their segments' times adjusted to them;
   *last_speech carried from window to window. 0, or -1 when out of
   memory or the decoder fails */
int janas_whisper_align(struct janas_whisper *w, int lang, int translate,
                        struct janas_whisper_seg *segs, uint32_t first,
                        uint32_t n, double offset, uint32_t frames,
                        double *last_speech);
/* the English name of a language ("italian") from its code, or NULL */
const char *janas_whisper_lang_name(const char *code);
/* the model's language of that code or English name ("it", "Italian"),
   its index, or -1 */
int janas_whisper_lang_index(const struct janas_whisper *w, const char *s);

#endif
