/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_speech.h - the speech model of the public API (api_speech.c) and
 * its diarization (api_speech_diar.c): a Whisper model, its voice activity
 * detector and speaker model, the last transcription and its turns.
 */
#ifndef JANAS_LLM_API_SPEECH_H
#define JANAS_LLM_API_SPEECH_H

#include "janas/llm.h"
#include "speaker.h"
#include "vad.h"
#include "whisper.h"

/* a speaker's turn: consecutive words of one speaker */
struct janas_turn {
    double t0, t1;
    char *text;
    char *speaker; /* a known speaker's name, or "A", "B"... */
};

struct janas_llm_speech {
    struct janas_whisper *w;
    struct janas_whisper_result r;
    struct janas_vad *vad;         /* or NULL */
    struct janas_speaker *speaker; /* or NULL */
    struct janas_turn *turns;
    uint32_t n_turns;
};

/* the model beside path (its directory) named file, opened by open, or
   NULL when there is none */
char *janas_speech_beside(const char *path, const char *file);

/* the speakers of the transcription just made (s->r, its words' times)
   of the sound snd (at 16 kHz, read from its start; pcm: all of it in
   memory, or NULL), from the stretches of speech regions (NULL: found
   here, or the whole recording without a detector): its turns in
   s->turns. 0, or -1 with the reason in err (also a known speaker's
   recording that cannot be read) */
int janas_speech_diarize(janas_llm_speech *s,
                         const struct janas_whisper_sound *snd,
                         const float *pcm, const size_t *regions,
                         size_t n_regions,
                         const struct janas_llm_transcribe_params *p, char *err,
                         size_t err_len);
void janas_speech_turns_free(janas_llm_speech *s);

#endif
