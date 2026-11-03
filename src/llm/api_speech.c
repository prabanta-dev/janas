/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_speech.c - speech to text in the public API (janas/llm.h): a Whisper
 * model (whisper.h) and the last transcription it made.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "api_speech.h"
#include "chat.h"
#include "common/audio.h"
#include "common/sysinfo.h"
#include "tokenizer.h"

char *janas_speech_beside(const char *path, const char *file)
{
    const char *slash = strrchr(path, '/');
    size_t dl = slash ? (size_t)(slash - path) + 1 : 0, fl = strlen(file);
    char *p = malloc(dl + fl + 1);
    if (!p)
        return NULL;
    memcpy(p, path, dl);
    memcpy(p + dl, file, fl + 1);
    if (access(p, R_OK) != 0) {
        free(p);
        return NULL;
    }
    return p;
}

void janas_llm_speech_params_default(struct janas_llm_speech_params *p)
{
    if (!p)
        return;
    memset(p, 0, sizeof *p);
    p->size = sizeof *p;
}

int32_t janas_llm_speech_open(const char *path,
                              const struct janas_llm_speech_params *p,
                              janas_llm_speech **out)
{
    if (!path || !out || (p && p->size != sizeof *p))
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    *out = NULL;
    janas_llm_speech *s = calloc(1, sizeof *s);
    if (!s)
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    char err[512];
    s->w = janas_whisper_open(path, p ? p->threads : 0, err, sizeof err);
    if (!s->w) {
        free(s);
        return janas_api_fail(JANAS_LLM_EOPEN, "%s", err);
    }
    /* the voice activity detector: named, or beside the model */
    if (p && p->vad_model) {
        s->vad = janas_vad_open(p->vad_model, err, sizeof err);
        if (!s->vad) {
            janas_llm_speech_close(s);
            return janas_api_fail(JANAS_LLM_EOPEN, "%s", err);
        }
    } else {
        char *vp = janas_speech_beside(path, "silero-vad.jns");
        if (vp)
            s->vad = janas_vad_open(vp, err, sizeof err);
        free(vp);
    }
    /* the speaker model for diarization: named, or beside the model */
    if (p && p->speaker_model) {
        s->speaker =
            janas_speaker_open(p->speaker_model, p->threads, err, sizeof err);
        if (!s->speaker) {
            janas_llm_speech_close(s);
            return janas_api_fail(JANAS_LLM_EOPEN, "%s", err);
        }
    } else {
        char *sp = janas_speech_beside(path, "wespeaker-resnet34.jns");
        if (sp)
            s->speaker =
                janas_speaker_open(sp, p ? p->threads : 0, err, sizeof err);
        free(sp);
    }
    *out = s;
    return JANAS_LLM_OK;
}

void janas_llm_speech_close(janas_llm_speech *s)
{
    if (!s)
        return;
    janas_whisper_result_free(&s->r);
    janas_speech_turns_free(s);
    janas_whisper_close(s->w);
    janas_vad_close(s->vad);
    janas_speaker_close(s->speaker);
    free(s);
}

void janas_llm_transcribe_params_default(struct janas_llm_transcribe_params *p)
{
    if (!p)
        return;
    memset(p, 0, sizeof *p);
    p->size = sizeof *p;
    p->vad_threshold = 0.5f;
    p->vad_prefix_ms = 300;
    p->vad_silence_ms = 200;
}

struct cb {
    void (*fn)(void *user, int32_t segment);
    void *user;
};

static void on_segment(void *user, uint32_t index)
{
    const struct cb *c = user;
    c->fn(c->user, (int32_t)index);
}

/*
 * The sound a transcription reads, in memory or as a stream: the decoded
 * sound kept whole while it fits in a share of the memory free, else read
 * from the file a window at a time (common/audio.h), every pass opening
 * the stream again - the same samples either way, and the same result.
 * JANAS_SPEECH_STREAM=1 streams any recording, =0 none (to compare them).
 */
struct sound {
    struct janas_whisper_sound snd;
    const float *pcm; /* whole, or NULL */
    const void *data; /* the file, for the stream */
    size_t bytes;
    struct janas_audio_info info;
    uint32_t rate;
    struct janas_audio_stream *st;
    float *buf; /* samples b0 .. b0 + bn of the stream */
    size_t b0, bn, cap;
};

static const float *sound_get(void *user, size_t from, size_t count,
                              char *err, size_t err_len)
{
    struct sound *d = user;
    if (from > d->snd.n)
        from = d->snd.n;
    if (count > d->snd.n - from)
        count = d->snd.n - from;
    if (d->pcm)
        return d->pcm + from;
    if (from < d->b0) {
        snprintf(err, err_len, "a stream read backwards");
        return NULL;
    }
    size_t drop = from - d->b0 < d->bn ? from - d->b0 : d->bn;
    memmove(d->buf, d->buf + drop, sizeof(float) * (d->bn - drop));
    d->bn -= drop;
    d->b0 += drop;
    if (count > d->cap || !d->cap) {
        size_t c = count > 65536 ? count : 65536; /* room to read past */
        float *b = realloc(d->buf, sizeof(float) * c);
        if (!b) {
            snprintf(err, err_len, "out of memory");
            return NULL;
        }
        d->buf = b;
        d->cap = c;
    }
    while (d->b0 + d->bn < from + count) {
        /* samples before from are read and dropped */
        size_t room = d->b0 < from ? (from - d->b0 < d->cap ? from - d->b0
                                                             : d->cap)
                                   : from + count - (d->b0 + d->bn);
        float *at = d->b0 < from ? d->buf : d->buf + d->bn;
        long k = janas_audio_stream_read(d->st, at, room, err, err_len);
        if (k <= 0) {
            if (!k)
                snprintf(err, err_len, "the sound ended early");
            return NULL;
        }
        if (d->b0 < from)
            d->b0 += (size_t)k;
        else
            d->bn += (size_t)k;
    }
    return d->buf;
}

static int sound_rewind(void *user, char *err, size_t err_len)
{
    struct sound *d = user;
    if (d->pcm)
        return 0;
    janas_audio_stream_close(d->st);
    d->b0 = d->bn = 0;
    d->st = janas_audio_stream_open(d->data, d->bytes, &d->info, d->rate);
    if (!d->st) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    return 0;
}

/* the share of the memory free the decoded sound may take whole: about
   eight bytes a sample as decoded go to it, its resampling, the
   spectrogram, the detector's copy */
static size_t whole_most(void)
{
    const char *e = getenv("JANAS_SPEECH_STREAM");
    if (e && *e)
        return atoi(e) > 0 ? 0 : JANAS_AUDIO_MAX_SAMPLES;
    uint64_t free_ = janas_mem_available(), most = JANAS_AUDIO_MAX_SAMPLES;
    if (free_ && free_ / 4 / 8 < most)
        most = free_ / 4 / 8;
    return (size_t)most;
}

/* the stretches of speech for the chunking: 0, or -1 with err */
static long chunk_regions(janas_llm_speech *s, struct sound *d,
                          const struct janas_llm_transcribe_params *p,
                          size_t **regions, char *err, size_t err_len)
{
    struct janas_vad_params vp = {.threshold = 0.5f,
                                  .min_speech_ms = 250,
                                  .min_silence_ms = 200,
                                  .speech_pad_ms = 300};
    size_t n = d->snd.n;
    float g = 1;
    if (p->chunking == 2) {
        if (p->vad_threshold > 0)
            vp.threshold = p->vad_threshold;
        if (p->vad_silence_ms > 0)
            vp.min_silence_ms = (uint32_t)p->vad_silence_ms;
        if (p->vad_prefix_ms >= 0)
            vp.speech_pad_ms = (uint32_t)p->vad_prefix_ms;
    } else {
        /* auto: the loudness normalized for the detector, its peak at
           0.9 */
        float peak = 0;
        for (size_t at = 0; at < n; at += 1 << 20) {
            size_t k = n - at < (1 << 20) ? n - at : 1 << 20;
            const float *x = sound_get(d, at, k, err, err_len);
            if (!x)
                return -1;
            for (size_t i = 0; i < k; i++)
                peak = fabsf(x[i]) > peak ? fabsf(x[i]) : peak;
        }
        g = peak > 0 ? 0.9f / peak : 1;
        if (sound_rewind(d, err, err_len) != 0)
            return -1;
    }
    float *probs = NULL;
    long np =
        janas_vad_probs_read(s->vad, n, g, sound_get, d, &probs, err, err_len);
    if (np < 0)
        return -1;
    long ns = janas_vad_segments(s->vad, probs, np, n, &vp, regions);
    free(probs);
    if (ns < 0)
        snprintf(err, err_len, "out of memory");
    return ns;
}

int32_t janas_llm_speech_transcribe(janas_llm_speech *s, const void *data,
                                    int64_t n,
                                    const struct janas_llm_transcribe_params *p)
{
    if (!s || !data || n <= 0 || (p && p->size != sizeof *p) ||
        (p && (p->temperature < 0 || p->temperature > 1)))
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid argument");
    janas_whisper_result_free(&s->r);
    janas_speech_turns_free(s);
    if (p && p->chunking && !s->vad)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "chunking needs a voice activity detector: "
                              "silero-vad.jns (made by silero2jns) beside "
                              "the speech model, or vad_model");
    if (p && p->diarize && !s->speaker)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "diarization needs a speaker model: "
                              "wespeaker-resnet34.jns (made by wespeaker2jns) "
                              "beside the speech model, or speaker_model");
    struct janas_whisper_opts o = {0};
    struct cb cb = {0};
    if (p) {
        o.language = p->language;
        o.translate = p->translate != 0;
        o.prompt = p->prompt;
        o.temperature = p->temperature;
        o.seed = p->seed;
        o.words = p->words != 0 || p->diarize != 0;
        if (p->on_segment) {
            o.on_segment = on_segment;
            o.user = &cb;
            cb.fn = p->on_segment;
            cb.user = p->user;
        }
        o.stop = (volatile int *)p->stop;
    }
    if (o.language && o.language[0] &&
        janas_whisper_lang_index(s->w, o.language) < 0 && s->w->multilingual)
        return janas_api_fail(JANAS_LLM_EINVAL,
                              "a language this model does not know: %s",
                              o.language);
    /* the sound whole, or streamed when it does not fit */
    struct janas_audio a = {0};
    struct sound d = {.snd = {.get = sound_get, .rewind = sound_rewind},
                      .data = data,
                      .bytes = (size_t)n,
                      .rate = s->w->rate};
    d.snd.user = &d;
    char err[512];
    int k = janas_audio_decode_max(data, (size_t)n, &a, whole_most(), err,
                                   sizeof err);
    if (k < 0)
        return janas_api_fail(strstr(err, "memory") ? JANAS_LLM_ENOMEM
                                                    : JANAS_LLM_EINVAL,
                              "%s", err);
    if (k == 0) {
        if (janas_audio_resample(&a, s->w->rate) != 0) {
            janas_audio_free(&a);
            return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
        }
        d.pcm = a.pcm;
        d.snd.n = a.n;
    } else {
        if (janas_audio_scan(data, (size_t)n, &d.info, err, sizeof err) != 0)
            return janas_api_fail(JANAS_LLM_EINVAL, "%s", err);
        d.snd.n = (size_t)janas_audio_stream_length(&d.info, s->w->rate);
        if (sound_rewind(&d, err, sizeof err) != 0)
            return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    }
    /* chunking: the stretches of speech the detector finds */
    size_t *regions = NULL, *merged = NULL;
    long n_stretches = 0;
    int rc = 0;
    if (p && p->chunking) {
        long ns = chunk_regions(s, &d, p, &regions, err, sizeof err);
        if (ns < 0 || sound_rewind(&d, err, sizeof err) != 0)
            rc = -1;
        static const size_t NONE[2] = {0, 0};
        o.regions = ns > 0 ? regions : NONE;
        o.n_regions = ns > 0 ? (size_t)ns : 1;
        /* the model reads the stretches together, a window of them up to
           its chunk with the pauses between (as WhisperX merges them): a
           stretch of half a second read on its own loses the words around
           it ("Ask... | NOT!" for "ask not"); diarization keeps them
           apart */
        if (ns > 1 && (merged = malloc(sizeof(size_t) * 2 * (size_t)ns))) {
            size_t most = (size_t)s->w->chunk * s->w->rate, m = 0;
            for (long i = 0; i < ns; i++) {
                if (m && regions[2 * i + 1] - merged[2 * m - 2] <= most) {
                    merged[2 * m - 1] = regions[2 * i + 1];
                    continue;
                }
                merged[2 * m] = regions[2 * i];
                merged[2 * m + 1] = regions[2 * i + 1];
                m++;
            }
            o.regions = merged;
            o.n_regions = m;
            n_stretches = ns;
        }
    }
    if (rc == 0)
        rc = d.pcm ? janas_whisper_transcribe(s->w, d.pcm, d.snd.n, &o, &s->r,
                                              err, sizeof err)
                   : janas_whisper_transcribe_sound(s->w, &d.snd, &o, &s->r,
                                                    err, sizeof err);
    if (merged) { /* diarization looks at the stretches themselves */
        o.regions = regions;
        o.n_regions = (size_t)n_stretches;
    }
    if (rc == 0 && p && p->diarize &&
        (sound_rewind(&d, err, sizeof err) != 0 ||
         janas_speech_diarize(s, &d.snd, d.pcm, p->chunking ? o.regions : NULL,
                              p->chunking ? o.n_regions : 0, p, err,
                              sizeof err) != 0))
        rc = -1;
    free(merged);
    free(regions);
    janas_audio_free(&a);
    janas_audio_stream_close(d.st);
    free(d.buf);
    if (rc != 0)
        return janas_api_fail(strstr(err, "memory")         ? JANAS_LLM_ENOMEM
                              : strstr(err, "English only") ? JANAS_LLM_EINVAL
                                                            : JANAS_LLM_EFAIL,
                              "%s", err);
    return JANAS_LLM_OK;
}

int32_t janas_llm_speech_text(const janas_llm_speech *s, char *buf, int32_t cap,
                              int32_t *len)
{
    if (!s)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    const char *t = s->r.text ? s->r.text : "";
    return janas_api_copy_out(t, strlen(t), buf, cap, len);
}

int32_t janas_llm_speech_language(const janas_llm_speech *s, char *buf,
                                  int32_t cap, int32_t *len)
{
    if (!s)
        return janas_api_fail(JANAS_LLM_EINVAL, "no model");
    return janas_api_copy_out(s->r.language, strlen(s->r.language), buf, cap,
                              len);
}

const char *janas_llm_speech_language_name(const char *code)
{
    return code ? janas_whisper_lang_name(code) : NULL;
}

double janas_llm_speech_duration(const janas_llm_speech *s)
{
    return s ? s->r.duration : 0;
}

int32_t janas_llm_speech_segments(const janas_llm_speech *s)
{
    return s ? (int32_t)s->r.n_segs : 0;
}

int32_t janas_llm_speech_segment(const janas_llm_speech *s, int32_t i,
                                 double *start, double *end, char *buf,
                                 int32_t cap, int32_t *len)
{
    if (!s || i < 0 || (uint32_t)i >= s->r.n_segs)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such segment");
    const struct janas_whisper_seg *g = &s->r.segs[i];
    if (start)
        *start = g->t0;
    if (end)
        *end = g->t1;
    if (!buf && !len)
        return JANAS_LLM_OK;
    return janas_api_copy_out(g->text, strlen(g->text), buf, cap, len);
}

int32_t janas_llm_speech_segment_info(const janas_llm_speech *s, int32_t i,
                                      float *avg_logprob, float *no_speech_prob,
                                      float *temperature,
                                      float *compression_ratio, int32_t *ids,
                                      int32_t max, int32_t *n)
{
    if (!s || i < 0 || (uint32_t)i >= s->r.n_segs || max < 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such segment");
    const struct janas_whisper_seg *g = &s->r.segs[i];
    if (avg_logprob)
        *avg_logprob = g->avg_logprob;
    if (no_speech_prob)
        *no_speech_prob = g->no_speech_prob;
    if (temperature)
        *temperature = g->temperature;
    if (compression_ratio)
        *compression_ratio = g->compression_ratio;
    if (n)
        *n = (int32_t)g->n_tokens;
    if (ids)
        memcpy(ids, g->tokens,
               sizeof(int32_t) *
                   ((uint32_t)max < g->n_tokens ? (uint32_t)max : g->n_tokens));
    return ids && (uint32_t)max < g->n_tokens
               ? janas_api_fail(JANAS_LLM_ESMALL, "buffer too small")
               : JANAS_LLM_OK;
}

int32_t janas_llm_speech_words(const janas_llm_speech *s)
{
    if (!s)
        return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < s->r.n_segs; i++)
        n += s->r.segs[i].n_words;
    return (int32_t)n;
}

int32_t janas_llm_speech_word(const janas_llm_speech *s, int32_t i,
                              double *start, double *end, float *probability,
                              char *buf, int32_t cap, int32_t *len)
{
    if (!s || i < 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such word");
    uint32_t k = (uint32_t)i;
    for (uint32_t g = 0; g < s->r.n_segs; g++) {
        const struct janas_whisper_seg *sg = &s->r.segs[g];
        if (k >= sg->n_words) {
            k -= sg->n_words;
            continue;
        }
        const struct janas_whisper_word *w = &sg->words[k];
        if (start)
            *start = w->t0;
        if (end)
            *end = w->t1;
        if (probability)
            *probability = w->probability;
        if (!buf && !len)
            return JANAS_LLM_OK;
        return janas_api_copy_out(w->text, strlen(w->text), buf, cap, len);
    }
    return janas_api_fail(JANAS_LLM_EINVAL, "no such word");
}

int32_t janas_llm_speech_segment_logprobs(const janas_llm_speech *s, int32_t i,
                                          float *lp, int32_t max, int32_t *n)
{
    if (!s || i < 0 || (uint32_t)i >= s->r.n_segs || max < 0)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such segment");
    const struct janas_whisper_seg *g = &s->r.segs[i];
    if (n)
        *n = (int32_t)g->n_tokens;
    if (lp)
        memcpy(lp, g->logprobs,
               sizeof(float) *
                   ((uint32_t)max < g->n_tokens ? (uint32_t)max : g->n_tokens));
    return lp && (uint32_t)max < g->n_tokens
               ? janas_api_fail(JANAS_LLM_ESMALL, "buffer too small")
               : JANAS_LLM_OK;
}

int32_t janas_llm_speech_token(const janas_llm_speech *s, int32_t id, char *buf,
                               int32_t cap, int32_t *len)
{
    if (!s || id < 0 || (uint32_t)id >= s->w->vocab)
        return janas_api_fail(JANAS_LLM_EINVAL, "no such token");
    char tmp[256];
    size_t k = janas_tokenizer_decode(s->w->tk, id, tmp, sizeof tmp);
    if (k > sizeof tmp)
        k = sizeof tmp;
    return janas_api_copy_out(tmp, k, buf, cap, len);
}

int32_t janas_llm_speech_token_is_text(const janas_llm_speech *s, int32_t id)
{
    return s && id >= 0 && id < s->w->eot;
}
