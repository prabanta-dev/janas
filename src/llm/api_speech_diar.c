/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_speech_diar.c - who speaks when, in the public API: the
 * transcription's stretches of speech (the voice activity detector's, or
 * the whole recording) diarized (diarize.c), the known speakers'
 * recordings embedded, each word given the speaker of the windows that
 * overlap it most, the words joined into turns - consecutive words of one
 * speaker, a pause of more than 2 s or a segment past 20 s of the turn
 * starting another - named after the
 * known speakers or "A", "B"... as they first speak.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api_speech.h"
#include "chat.h"
#include "common/audio.h"

#define MAX_KNOWN 8
#define REF_FRAMES 3000 /* 30 s of a known speaker at most */

void janas_speech_turns_free(janas_llm_speech *s)
{
    for (uint32_t i = 0; i < s->n_turns; i++) {
        free(s->turns[i].text);
        free(s->turns[i].speaker);
    }
    free(s->turns);
    s->turns = NULL;
    s->n_turns = 0;
}

/* a known speaker's recording embedded: 0, or -1 with why */
static int embed_ref(janas_llm_speech *s, const void *data, int64_t n,
                     float *emb, char *err, size_t err_len)
{
    struct janas_audio a;
    char why[256];
    if (!data || n <= 0 ||
        janas_audio_decode(data, (size_t)n, &a, why, sizeof why) != 0) {
        snprintf(err, err_len, "a known speaker's recording: %s",
                 data && n > 0 ? why : "none given");
        return -1;
    }
    int rc = -1;
    uint32_t frames = 0;
    float *fb = NULL;
    if (janas_audio_resample(&a, 16000) == 0)
        fb = janas_speaker_fbank(a.pcm, a.n, &frames);
    if (!fb)
        snprintf(err, err_len, "out of memory");
    else if (janas_speaker_embed(s->speaker, fb, 0,
                                 frames < REF_FRAMES ? frames : REF_FRAMES,
                                 emb) != 0)
        snprintf(err, err_len,
                 "a known speaker's recording too short (at least 0.1 s)");
    else
        rc = 0;
    free(fb);
    janas_audio_free(&a);
    return rc;
}

/* the filter banks of a sound read going forward, the frames f0 .. f1
   asked for kept (fb0 .. fb0 + fbn) */
struct banks {
    const struct janas_whisper_sound *snd;
    struct janas_fbank *fb;
    float *frames;
    uint32_t fb0, fbn, cap;
    char *err;
    size_t err_len;
};

static const float *banks_get(void *user, uint32_t f0, uint32_t f1)
{
    struct banks *b = user;
    if (f0 < b->fb0)
        return NULL;
    uint32_t drop = f0 - b->fb0 < b->fbn ? f0 - b->fb0 : b->fbn;
    memmove(b->frames, b->frames + (size_t)drop * 80,
            sizeof(float) * 80 * (b->fbn - drop));
    b->fbn -= drop;
    b->fb0 += drop;
    if (b->fbn == 0)
        b->fb0 = f0;
    if (f1 - b->fb0 > b->cap) {
        float *t = realloc(b->frames, sizeof(float) * 80 * (f1 - b->fb0));
        if (!t)
            return NULL;
        b->frames = t;
        b->cap = f1 - b->fb0;
    }
    uint32_t first = b->fb0 + b->fbn;
    if (f1 > first) {
        /* frame t reads 400 samples from 160 t */
        size_t from = (size_t)first * 160,
               count = (size_t)(f1 - first - 1) * 160 + 400;
        const float *x =
            b->snd->get(b->snd->user, from, count, b->err, b->err_len);
        if (!x)
            return NULL;
        janas_fbank_frames(b->fb, x, f1 - first,
                           b->frames + (size_t)b->fbn * 80);
        b->fbn = f1 - b->fb0;
    }
    return b->frames;
}

int janas_speech_diarize(janas_llm_speech *s,
                         const struct janas_whisper_sound *snd,
                         const float *pcm, const size_t *regions,
                         size_t n_regions,
                         const struct janas_llm_transcribe_params *p, char *err,
                         size_t err_len)
{
    size_t n = snd->n;
    int nk = p->n_speakers > 0 ? p->n_speakers : 0;
    if (nk > MAX_KNOWN) {
        snprintf(err, err_len, "at most %d known speakers", MAX_KNOWN);
        return -1;
    }
    if (nk && (!p->speaker_refs || !p->speaker_ref_bytes)) {
        snprintf(err, err_len, "known speakers without their recordings");
        return -1;
    }
    float ref[MAX_KNOWN * 256];
    for (int k = 0; k < nk; k++)
        if (embed_ref(s, p->speaker_refs[k], p->speaker_ref_bytes[k],
                      ref + 256 * k, err, err_len) != 0)
            return -1;
    /* the stretches of speech: the chunking's, the detector's, or all */
    size_t *found = NULL, whole[2] = {0, n};
    if (!regions) {
        regions = whole;
        n_regions = 1;
        if (s->vad) {
            float *probs = NULL;
            long np = janas_vad_probs_read(s->vad, n, 1.0f, snd->get,
                                           snd->user, &probs, err, err_len);
            if (np < 0 || snd->rewind(snd->user, err, err_len) != 0) {
                free(probs);
                return -1;
            }
            struct janas_vad_params vp = {.threshold = 0.5f,
                                          .min_speech_ms = 250,
                                          .min_silence_ms = 100,
                                          .speech_pad_ms = 30};
            long ns =
                np < 0 ? -1
                       : janas_vad_segments(s->vad, probs, np, n, &vp, &found);
            free(probs);
            if (ns < 0) {
                snprintf(err, err_len, "out of memory");
                return -1;
            }
            regions = found;
            n_regions = (size_t)ns;
        }
    }
    struct janas_diar d;
    int rc;
    err[0] = 0;
    if (pcm)
        rc = janas_diarize(s->speaker, pcm, n, regions, n_regions, ref,
                           (uint32_t)nk, 0, &d);
    else {
        struct banks b = {.snd = snd,
                          .fb = janas_fbank_new(),
                          .err = err,
                          .err_len = err_len};
        rc = b.fb ? janas_diarize_read(s->speaker, janas_fbank_count(n),
                                       banks_get, &b, regions, n_regions, ref,
                                       (uint32_t)nk, 0, &d)
                  : -1;
        janas_fbank_free(b.fb);
        free(b.frames);
    }
    free(found);
    if (rc != 0) {
        if (!err[0])
            snprintf(err, err_len, "out of memory");
        return -1;
    }
    /* each word's speaker: the windows overlapping it, weighed by how much */
    uint32_t total = 0;
    for (uint32_t g = 0; g < s->r.n_segs; g++)
        total += s->r.segs[g].n_words;
    int *wsp = malloc(sizeof(int) * (total ? total : 1));
    double *vote = calloc(d.n_speakers ? d.n_speakers : 1, sizeof(double));
    char **label = calloc(d.n_speakers ? d.n_speakers : 1, sizeof(char *));
    s->turns = calloc(total ? total : 1, sizeof *s->turns);
    if (!wsp || !vote || !label || !s->turns)
        goto oom;
    uint32_t k = 0;
    for (uint32_t g = 0; g < s->r.n_segs; g++)
        for (uint32_t i = 0; i < s->r.segs[g].n_words; i++, k++) {
            const struct janas_whisper_word *wd = &s->r.segs[g].words[i];
            double a = wd->t0 * 16000, b = wd->t1 * 16000;
            memset(vote, 0, sizeof(double) * (d.n_speakers ? d.n_speakers : 1));
            int best = -1;
            double bv = 0, near = 1e300;
            int nearest = d.n_win ? d.speaker[0] : 0;
            for (uint32_t j = 0; j < d.n_win; j++) {
                double ws = (double)d.win[2 * j], we = (double)d.win[2 * j + 1];
                double o = (b < we ? b : we) - (a > ws ? a : ws);
                if (o > 0)
                    vote[d.speaker[j]] += o;
                double c = (ws + we) / 2 - (a + b) / 2;
                if (c < 0)
                    c = -c;
                if (c < near) {
                    near = c;
                    nearest = d.speaker[j];
                }
            }
            for (uint32_t q = 0; q < d.n_speakers; q++)
                if (vote[q] > bv) {
                    bv = vote[q];
                    best = (int)q;
                }
            wsp[k] = best >= 0 ? best : nearest;
        }
    /* the turns */
    uint32_t nt = 0, unknown = 0;
    double last_end = -1e300;
    int cur = -1;
    k = 0;
    for (uint32_t g = 0; g < s->r.n_segs; g++)
        for (uint32_t i = 0; i < s->r.segs[g].n_words; i++, k++) {
            const struct janas_whisper_word *wd = &s->r.segs[g].words[i];
            /* a new turn: another speaker, a pause of more than 2 s, or a
               segment's start once the turn is longer than 20 s */
            if (wsp[k] != cur || wd->t0 - last_end > 2.0 ||
                (i == 0 && nt && wd->t0 - s->turns[nt - 1].t0 > 20.0)) {
                cur = wsp[k];
                struct janas_turn *t = &s->turns[nt++];
                t->t0 = wd->t0;
                if (!label[cur]) {
                    char name[16];
                    if (d.n_speakers && d.known[cur] >= 0 && p->speaker_names &&
                        p->speaker_names[d.known[cur]]) {
                        label[cur] = strdup(p->speaker_names[d.known[cur]]);
                    } else {
                        if (unknown < 26)
                            snprintf(name, sizeof name, "%c", 'A' + unknown);
                        else
                            snprintf(name, sizeof name, "S%u", unknown + 1);
                        unknown++;
                        label[cur] = strdup(name);
                    }
                    if (!label[cur])
                        goto oom;
                }
                t->speaker = strdup(label[cur]);
                t->text = strdup("");
                if (!t->speaker || !t->text)
                    goto oom;
            }
            struct janas_turn *t = &s->turns[nt - 1];
            size_t a = strlen(t->text), b = strlen(wd->text);
            char *x = realloc(t->text, a + b + 1);
            if (!x)
                goto oom;
            memcpy(x + a, wd->text, b + 1);
            t->text = x;
            t->t1 = wd->t1;
            last_end = wd->t1;
        }
    s->n_turns = nt;
    for (uint32_t i = 0; i < nt; i++) { /* without the leading space */
        char *t = s->turns[i].text, *b = t;
        while (*b == ' ')
            b++;
        memmove(t, b, strlen(b) + 1);
    }
    for (uint32_t q = 0; q < d.n_speakers; q++)
        free(label[q]);
    free(label);
    free(vote);
    free(wsp);
    janas_diar_free(&d);
    return 0;
oom:
    s->n_turns = total;
    janas_speech_turns_free(s);
    if (label)
        for (uint32_t q = 0; q < d.n_speakers; q++)
            free(label[q]);
    free(label);
    free(vote);
    free(wsp);
    janas_diar_free(&d);
    snprintf(err, err_len, "out of memory");
    return -1;
}

int32_t janas_llm_speech_turns(const janas_llm_speech *s)
{
    return s ? (int32_t)s->n_turns : 0;
}

int32_t janas_llm_speech_turn(const janas_llm_speech *s, int32_t i,
                              double *start, double *end, char *speaker,
                              int32_t speaker_cap, int32_t *speaker_len,
                              char *buf, int32_t cap, int32_t *len)
{
    if (!s || i < 0 || (uint32_t)i >= s->n_turns)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such turn");
    const struct janas_turn *t = &s->turns[i];
    if (start)
        *start = t->t0;
    if (end)
        *end = t->t1;
    if (speaker || speaker_len) {
        int32_t rc = janas_api_copy_out(t->speaker, strlen(t->speaker), speaker,
                                        speaker_cap, speaker_len);
        if (rc != JANAS_LLM_OK)
            return rc;
    }
    if (!buf && !len)
        return JANAS_LLM_OK;
    return janas_api_copy_out(t->text, strlen(t->text), buf, cap, len);
}
