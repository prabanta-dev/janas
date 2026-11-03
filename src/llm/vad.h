/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vad.h - voice activity detection with Silero VAD (vad.c): the
 * probability of speech in every 32 ms of 16 kHz sound, and the stretches
 * of speech found from it as Silero's get_speech_timestamps finds them.
 */
#ifndef JANAS_LLM_VAD_H
#define JANAS_LLM_VAD_H

#include <stddef.h>
#include <stdint.h>

struct janas_vad;

/* a model made by tools/silero2jns, or NULL with the reason in err */
struct janas_vad *janas_vad_open(const char *path, char *err, size_t err_len);
void janas_vad_close(struct janas_vad *v);
/* the samples a probability covers (512) and their rate (16000) */
uint32_t janas_vad_window(const struct janas_vad *v);
uint32_t janas_vad_rate(const struct janas_vad *v);

/* the probability of speech in each window of n samples (the last one
   padded with silence), into probs (n / window rounded up, malloc'd):
   their count, or -1 when out of memory */
long janas_vad_probs(struct janas_vad *v, const float *pcm, size_t n,
                     float **probs);

/* the same from a sound read a piece at a time, each sample times gain:
   get gives samples [from, from + count) of the n, as a
   janas_whisper_sound does (NULL with err when it cannot); -1 with err */
long janas_vad_probs_read(struct janas_vad *v, size_t n, float gain,
                          const float *(*get)(void *user, size_t from,
                                              size_t count, char *err,
                                              size_t err_len),
                          void *user, float **probs, char *err,
                          size_t err_len);

struct janas_vad_params {
    float threshold;         /* 0.5 */
    uint32_t min_speech_ms;  /* 250 */
    uint32_t min_silence_ms; /* 100 */
    float max_speech_s;      /* 0: no limit */
    uint32_t speech_pad_ms;  /* 30 */
};

/* the stretches of speech of n samples from their probabilities: pairs
   (start, end) in samples into *seg (malloc'd), their count, or -1 */
long janas_vad_segments(const struct janas_vad *v, const float *probs,
                        long n_probs, size_t n,
                        const struct janas_vad_params *p, size_t **seg);

#endif
