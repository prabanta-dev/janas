/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * speech.c - openSpeech() and the Speech class: a Whisper model opened on an
 * engine thread of its own, as a language model is (engine.c), recordings
 * transcribed or translated on it, close().
 *
 * The engine's thread transcribes and copies what the library kept of the
 * transcription - text, segments, words, turns - into the job, so that
 * JavaScript's thread makes the result from the copy while the engine may
 * already be on the next recording.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "node/node_impl.h"

enum { MAX_SPEAKERS = 8 };

struct speech_open {
    char *path, *vad, *speaker;
    struct janas_llm_speech_params p;
};

static void speech_open_drop(void *data)
{
    struct speech_open *so = data;
    if (!so)
        return;
    free(so->path);
    free(so->vad);
    free(so->speaker);
    free(so);
}

/* a span of the transcription: a segment, a word or a turn */
struct span {
    double start, end;
    char *text, *speaker;
    float avg_logprob, no_speech, temperature, ratio, probability;
};

struct transcribe_job {
    void *data;
    size_t n;
    struct janas_llm_transcribe_params p;
    char *language, *prompt;
    char *names[MAX_SPEAKERS];
    void *refs[MAX_SPEAKERS];
    int64_t ref_bytes[MAX_SPEAKERS];
    atomic_int stop;
    napi_ref signal, listener;
    /* what came out, copied on the engine's thread */
    char *text;
    char lang[16];
    double duration;
    struct span *segs, *words, *turns;
    int32_t n_segs, n_words, n_turns;
};

static void spans_free(struct span *s, int32_t n)
{
    for (int32_t i = 0; s && i < n; i++) {
        free(s[i].text);
        free(s[i].speaker);
    }
    free(s);
}

static void transcribe_drop(void *data)
{
    struct transcribe_job *tj = data;
    if (!tj)
        return;
    free(tj->data);
    free(tj->language);
    free(tj->prompt);
    for (int i = 0; i < MAX_SPEAKERS; i++) {
        free(tj->names[i]);
        free(tj->refs[i]);
    }
    free(tj->text);
    spans_free(tj->segs, tj->n_segs);
    spans_free(tj->words, tj->n_words);
    spans_free(tj->turns, tj->n_turns);
    free(tj);
}

/* --- the engine's thread ------------------------------------------------ */

static void open_run(struct janas_node_job *j)
{
    struct speech_open *so = j->data;
    so->p.vad_model = so->vad;
    so->p.speaker_model = so->speaker;
    j->rc = janas_llm_speech_open(so->path, &so->p, &j->m->speech);
    if (j->rc != JANAS_LLM_OK) {
        snprintf(j->error, sizeof(j->error), "%s", janas_llm_last_error());
        j->m->speech = NULL;
        j->last = 1; /* no model: the thread ends */
    }
}

/* A text the library writes into a buffer, read whole: get is called
   again with a buffer as long as the length it gave. NULL when memory ran
   out. */
#define TAKE(out, call_with_buf_cap_len)                                       \
    do {                                                                       \
        int32_t cap = 256, len = 0, rc;                                        \
        char *buf = malloc((size_t)cap);                                       \
        while (buf) {                                                          \
            rc = (call_with_buf_cap_len);                                      \
            if (rc != JANAS_LLM_ESMALL || len < cap)                           \
                break;                                                         \
            cap = len + 1;                                                     \
            char *b = realloc(buf, (size_t)cap);                               \
            if (!b) {                                                          \
                free(buf);                                                     \
                buf = NULL;                                                    \
            } else                                                             \
                buf = b;                                                       \
        }                                                                      \
        if (buf && rc != JANAS_LLM_OK)                                         \
            buf[0] = 0;                                                        \
        (out) = buf;                                                           \
    } while (0)

static int copy_out(janas_llm_speech *s, struct transcribe_job *tj)
{
    TAKE(tj->text, janas_llm_speech_text(s, buf, cap, &len));
    if (!tj->text)
        return -1;
    janas_llm_speech_language(s, tj->lang, sizeof(tj->lang), NULL);
    tj->duration = janas_llm_speech_duration(s);
    int32_t ns = janas_llm_speech_segments(s), nw = janas_llm_speech_words(s),
            nt = janas_llm_speech_turns(s);
    tj->segs = calloc(ns > 0 ? (size_t)ns : 1, sizeof(struct span));
    tj->words = calloc(nw > 0 ? (size_t)nw : 1, sizeof(struct span));
    tj->turns = calloc(nt > 0 ? (size_t)nt : 1, sizeof(struct span));
    if (!tj->segs || !tj->words || !tj->turns)
        return -1;
    for (int32_t i = 0; i < ns; i++, tj->n_segs++) {
        struct span *sp = &tj->segs[i];
        TAKE(sp->text, janas_llm_speech_segment(s, i, &sp->start, &sp->end, buf,
                                                cap, &len));
        if (!sp->text)
            return -1;
        janas_llm_speech_segment_info(s, i, &sp->avg_logprob, &sp->no_speech,
                                      &sp->temperature, &sp->ratio, NULL, 0,
                                      NULL);
    }
    for (int32_t i = 0; i < nw; i++, tj->n_words++) {
        struct span *sp = &tj->words[i];
        TAKE(sp->text, janas_llm_speech_word(s, i, &sp->start, &sp->end,
                                             &sp->probability, buf, cap, &len));
        if (!sp->text)
            return -1;
    }
    for (int32_t i = 0; i < nt; i++, tj->n_turns++) {
        struct span *sp = &tj->turns[i];
        char who[256];
        TAKE(sp->text,
             janas_llm_speech_turn(s, i, &sp->start, &sp->end, who, sizeof(who),
                                   NULL, buf, cap, &len));
        if (!sp->text || !(sp->speaker = strdup(who)))
            return -1;
    }
    return 0;
}

static void transcribe_run(struct janas_node_job *j)
{
    struct transcribe_job *tj = j->data;
    janas_llm_speech *s = j->m->speech;
    if (!s) {
        j->rc = JANAS_NODE_ECLOSED;
        snprintf(j->error, sizeof(j->error), "janas: the model is closed");
        return;
    }
    if (atomic_load(&tj->stop)) {
        j->rc = JANAS_LLM_EFAIL;
        snprintf(j->error, sizeof(j->error), "janas: stopped");
        return;
    }
    tj->p.language = tj->language;
    tj->p.prompt = tj->prompt;
    tj->p.speaker_names = (const char *const *)tj->names;
    tj->p.speaker_refs = (const void *const *)tj->refs;
    tj->p.speaker_ref_bytes = tj->ref_bytes;
    /* the library reads the flag between windows; the signal's listener
       sets it on JavaScript's thread */
    tj->p.stop = (volatile int32_t *)&tj->stop;
    j->rc = janas_llm_speech_transcribe(s, tj->data, (int64_t)tj->n, &tj->p);
    if (j->rc != JANAS_LLM_OK) {
        snprintf(j->error, sizeof(j->error), "janas: %s",
                 janas_llm_last_error());
        return;
    }
    if (copy_out(s, tj) != 0) {
        j->rc = JANAS_LLM_ENOMEM;
        snprintf(j->error, sizeof(j->error), "janas: out of memory");
    }
}

/* --- JavaScript's thread ------------------------------------------------ */

static void speech_finalize(napi_env env, void *data, void *hint)
{
    (void)hint;
    janas_node_release(env, data);
}

static napi_value open_value(napi_env env, struct janas_node_job *j)
{
    struct janas_node_model *m = j->m;
    struct speech_open *so = j->data;
    napi_value obj = janas_node_new(env, m->st->speech_class,
                                    &janas_node_speech_tag, m, speech_finalize);
    if (!obj)
        return NULL;
    janas_node_hold(m);
    napi_property_descriptor d[] = {
        {.utf8name = "path",
         .value = janas_node_string(env, so->path, JANAS_NAPI_AUTO_LENGTH),
         .attributes = napi_enumerable},
    };
    napi_define_properties(env, obj, 1, d);
    return obj;
}

static void open_after(napi_env env, struct janas_node_job *j)
{
    struct janas_node_model *m = j->m;
    if (j->rc < 0)
        m->closing = 1; /* the thread has ended: back as the last job */
    else if (!m->holders)
        janas_node_close(env, m, NULL); /* its object could not be made */
}

napi_value janas_node_open_speech(napi_env env, napi_callback_info info)
{
    static const char *const known[] = {"threads", "vadModel", "speakerModel",
                                        NULL};
    const char *w = "openSpeech";
    size_t argc = 2;
    napi_value argv[2] = {NULL, NULL};
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    struct janas_node_env *st = janas_node_env(env);
    size_t len = 0;
    char *path = argc ? janas_node_text(env, argv[0], &len) : NULL;
    if (!st || !path)
        return len == SIZE_MAX
                   ? janas_node_throw(env, JANAS_LLM_ENOMEM,
                                      "janas: out of memory")
                   : janas_node_type_error(
                         env, "janas: openSpeech(path, options): the path of "
                              "a Whisper model (.jns), as a string");
    struct speech_open *so = calloc(1, sizeof(*so));
    if (!so) {
        free(path);
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    }
    so->path = path;
    janas_llm_speech_params_default(&so->p);
    napi_value o = argc > 1 ? argv[1] : NULL;
    int64_t i;
    int r;
    if (janas_node_options(env, o, w, known) < 0 ||
        (r = janas_node_opt_int(env, o, w, "threads", 0, 4096, &i)) < 0 ||
        janas_node_opt_string(env, o, w, "vadModel", &so->vad) < 0 ||
        janas_node_opt_string(env, o, w, "speakerModel", &so->speaker) < 0) {
        speech_open_drop(so);
        return NULL;
    }
    if (r)
        so->p.threads = (int32_t)i;
    struct janas_node_model *m = janas_node_engine(env, st);
    struct janas_node_job *j = m ? janas_node_job(m, open_run) : NULL;
    napi_value p = j ? janas_node_promise(env, &j->deferred) : NULL;
    if (!p) {
        speech_open_drop(so);
        free(j);
        if (m) /* the thread ends with the close it is given */
            janas_node_close(env, m, NULL);
        return janas_node_throw(env, JANAS_LLM_ENOMEM,
                                "janas: openSpeech: no memory or no thread "
                                "for the engine");
    }
    j->data = so;
    j->drop = speech_open_drop;
    j->value = open_value;
    j->after = open_after;
    janas_node_post(env, j);
    return p;
}

static napi_value spans(napi_env env, const struct span *s, int32_t n, int kind)
{
    napi_value a;
    if (napi_create_array_with_length(env, (size_t)n, &a) != napi_ok)
        return NULL;
    for (int32_t i = 0; i < n; i++) {
        napi_value o;
        if (napi_create_object(env, &o) != napi_ok)
            return NULL;
        janas_node_put(env, o, "start", janas_node_number(env, s[i].start));
        janas_node_put(env, o, "end", janas_node_number(env, s[i].end));
        if (kind == 2)
            janas_node_put(
                env, o, "speaker",
                janas_node_string(env, s[i].speaker, JANAS_NAPI_AUTO_LENGTH));
        janas_node_put(
            env, o, "text",
            janas_node_string(env, s[i].text, JANAS_NAPI_AUTO_LENGTH));
        if (kind == 0) {
            janas_node_put(env, o, "avgLogprob",
                           janas_node_number(env, s[i].avg_logprob));
            janas_node_put(env, o, "noSpeechProb",
                           janas_node_number(env, s[i].no_speech));
            janas_node_put(env, o, "temperature",
                           janas_node_number(env, s[i].temperature));
            janas_node_put(env, o, "compressionRatio",
                           janas_node_number(env, s[i].ratio));
        } else if (kind == 1)
            janas_node_put(env, o, "probability",
                           janas_node_number(env, s[i].probability));
        if (napi_set_element(env, a, (uint32_t)i, o) != napi_ok)
            return NULL;
    }
    return a;
}

static napi_value transcribe_value(napi_env env, struct janas_node_job *j)
{
    struct transcribe_job *tj = j->data;
    napi_value o, segs, words = NULL, turns = NULL;
    if (napi_create_object(env, &o) != napi_ok ||
        !(segs = spans(env, tj->segs, tj->n_segs, 0)) ||
        (tj->p.words && !(words = spans(env, tj->words, tj->n_words, 1))) ||
        (tj->p.diarize && !(turns = spans(env, tj->turns, tj->n_turns, 2))))
        return NULL;
    janas_node_put(env, o, "text",
                   janas_node_string(env, tj->text, JANAS_NAPI_AUTO_LENGTH));
    janas_node_put(env, o, "language",
                   janas_node_string(env, tj->lang, JANAS_NAPI_AUTO_LENGTH));
    janas_node_put(env, o, "duration", janas_node_number(env, tj->duration));
    janas_node_put(env, o, "segments", segs);
    if (words)
        janas_node_put(env, o, "words", words);
    if (turns)
        janas_node_put(env, o, "turns", turns);
    return o;
}

static void transcribe_after(napi_env env, struct janas_node_job *j)
{
    struct transcribe_job *tj = j->data;
    janas_node_unwatch(env, &tj->signal, &tj->listener);
}

/* chunking: "auto", or an object with the detector's settings */
static int chunking(napi_env env, napi_value o, const char *w,
                    struct janas_llm_transcribe_params *p)
{
    static const char *const known[] = {"threshold", "prefixMs", "silenceMs",
                                        NULL};
    napi_value v;
    napi_valuetype t;
    if (!o || napi_typeof(env, o, &t) != napi_ok || t != napi_object ||
        napi_get_named_property(env, o, "chunking", &v) != napi_ok ||
        napi_typeof(env, v, &t) != napi_ok || t == napi_undefined)
        return 0;
    if (t == napi_string) {
        char buf[16] = "";
        napi_get_value_string_utf8(env, v, buf, sizeof(buf), NULL);
        if (strcmp(buf, "auto") != 0) {
            janas_node_type_error(env,
                                  "janas: %s: chunking is \"auto\" or an "
                                  "object",
                                  w);
            return -1;
        }
        p->chunking = 1;
        return 0;
    }
    if (t != napi_object) {
        janas_node_type_error(env,
                              "janas: %s: chunking is \"auto\" or an "
                              "object",
                              w);
        return -1;
    }
    double d;
    int64_t i;
    int r;
    p->chunking = 2;
    if (janas_node_options(env, v, w, known) < 0 ||
        (r = janas_node_opt_number(env, v, w, "threshold", 0, 1, &d)) < 0)
        return -1;
    if (r)
        p->vad_threshold = (float)d;
    if ((r = janas_node_opt_int(env, v, w, "prefixMs", 0, 60000, &i)) < 0)
        return -1;
    if (r)
        p->vad_prefix_ms = (int32_t)i;
    if ((r = janas_node_opt_int(env, v, w, "silenceMs", 1, 60000, &i)) < 0)
        return -1;
    if (r)
        p->vad_silence_ms = (int32_t)i;
    return 0;
}

/* speakers: [{name, data}], at most eight */
static int speakers(napi_env env, napi_value o, const char *w,
                    struct transcribe_job *tj)
{
    napi_value v;
    napi_valuetype t;
    bool array = false;
    uint32_t n = 0;
    if (!o || napi_typeof(env, o, &t) != napi_ok || t != napi_object ||
        napi_get_named_property(env, o, "speakers", &v) != napi_ok ||
        napi_typeof(env, v, &t) != napi_ok || t == napi_undefined)
        return 0;
    if (napi_is_array(env, v, &array) != napi_ok || !array ||
        napi_get_array_length(env, v, &n) != napi_ok || n > MAX_SPEAKERS) {
        janas_node_type_error(env,
                              "janas: %s: speakers is an array of at most %d "
                              "{name, data}",
                              w, MAX_SPEAKERS);
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        napi_value e, name, data;
        size_t len = 0, bytes = 0;
        if (napi_get_element(env, v, i, &e) != napi_ok ||
            napi_typeof(env, e, &t) != napi_ok || t != napi_object ||
            napi_get_named_property(env, e, "name", &name) != napi_ok ||
            napi_get_named_property(env, e, "data", &data) != napi_ok ||
            !(tj->names[i] = janas_node_text(env, name, &len)) ||
            !(tj->refs[i] = janas_node_bytes(env, data, &bytes))) {
            if (len == SIZE_MAX || bytes == SIZE_MAX)
                janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
            else
                janas_node_type_error(env,
                                      "janas: %s: speaker %u: a name (a "
                                      "string) and data (the bytes of a short "
                                      "recording)",
                                      w, i);
            return -1;
        }
        tj->ref_bytes[i] = (int64_t)bytes;
    }
    tj->p.n_speakers = (int32_t)n;
    return 0;
}

static int transcribe_options(napi_env env, napi_value o,
                              struct transcribe_job *tj, napi_value *signal)
{
    static const char *const known[] = {
        "translate", "language", "prompt",   "temperature", "seed", "words",
        "chunking",  "diarize",  "speakers", "signal",      NULL};
    const char *w = "speech.transcribe";
    struct janas_llm_transcribe_params *p = &tj->p;
    double d;
    int64_t i;
    int b, r;
    if (janas_node_options(env, o, w, known) < 0)
        return -1;
    if ((r = janas_node_opt_bool(env, o, w, "translate", &b)) < 0)
        return -1;
    if (r)
        p->translate = b;
    if (janas_node_opt_string(env, o, w, "language", &tj->language) < 0 ||
        janas_node_opt_string(env, o, w, "prompt", &tj->prompt) < 0)
        return -1;
    if ((r = janas_node_opt_number(env, o, w, "temperature", 0, 1, &d)) < 0)
        return -1;
    if (r)
        p->temperature = (float)d;
    if ((r = janas_node_opt_int(env, o, w, "seed", 0, 9007199254740991LL, &i)) <
        0)
        return -1;
    if (r)
        p->seed = (uint64_t)i;
    if ((r = janas_node_opt_bool(env, o, w, "words", &b)) < 0)
        return -1;
    if (r)
        p->words = b;
    if ((r = janas_node_opt_bool(env, o, w, "diarize", &b)) < 0)
        return -1;
    if (r)
        p->diarize = b;
    if (chunking(env, o, w, p) < 0 || speakers(env, o, w, tj) < 0)
        return -1;
    return janas_node_opt(env, o, w, "signal", napi_object, signal);
}

static napi_value speech_transcribe(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value argv[2] = {NULL, NULL};
    struct janas_node_model *m = janas_node_self(
        env, info, &janas_node_speech_tag, &argc, argv, "Speech.transcribe");
    if (!m)
        return NULL;
    if (m->closing)
        return janas_node_rejected(
            env, janas_node_error(env, JANAS_NODE_ECLOSED,
                                  "janas: speech.transcribe: the model is "
                                  "closed"));
    struct transcribe_job *tj = calloc(1, sizeof(*tj));
    if (!tj)
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    janas_llm_transcribe_params_default(&tj->p);
    tj->data = argc ? janas_node_bytes(env, argv[0], &tj->n) : NULL;
    if (!tj->data) {
        int oom = tj->n == SIZE_MAX;
        transcribe_drop(tj);
        return oom ? janas_node_throw(env, JANAS_LLM_ENOMEM,
                                      "janas: out of memory")
                   : janas_node_type_error(
                         env, "janas: speech.transcribe(data, options): the "
                              "bytes of a sound file, as a Buffer, a "
                              "Uint8Array or an ArrayBuffer");
    }
    napi_value signal = NULL;
    int has_signal =
        transcribe_options(env, argc > 1 ? argv[1] : NULL, tj, &signal);
    if (has_signal < 0) {
        transcribe_drop(tj);
        return NULL;
    }
    if (has_signal > 0 && janas_node_watch(env, signal, &tj->stop, &tj->signal,
                                           &tj->listener) < 0) {
        janas_node_unwatch(env, &tj->signal, &tj->listener);
        transcribe_drop(tj);
        return NULL;
    }
    struct janas_node_job *j = janas_node_job(m, transcribe_run);
    napi_value p = j ? janas_node_promise(env, &j->deferred) : NULL;
    if (!p) {
        janas_node_unwatch(env, &tj->signal, &tj->listener);
        transcribe_drop(tj);
        free(j);
        return janas_node_throw(env, JANAS_LLM_ENOMEM, "janas: out of memory");
    }
    j->data = tj;
    j->drop = transcribe_drop;
    j->value = transcribe_value;
    j->after = transcribe_after;
    janas_node_post(env, j);
    return p;
}

static napi_value speech_close(napi_env env, napi_callback_info info)
{
    struct janas_node_model *m = janas_node_self(
        env, info, &janas_node_speech_tag, NULL, NULL, "Speech.close");
    if (!m)
        return NULL;
    napi_deferred d;
    napi_value p = janas_node_promise(env, &d);
    if (p)
        janas_node_close(env, m, d);
    return p;
}

napi_value janas_node_speech_class(napi_env env)
{
    const napi_property_attributes method = napi_writable | napi_configurable;
    napi_property_descriptor d[] = {
        {.utf8name = "transcribe",
         .method = speech_transcribe,
         .attributes = method},
        {.utf8name = "close", .method = speech_close, .attributes = method},
    };
    napi_value cls = NULL;
    if (napi_define_class(env, "Speech", JANAS_NAPI_AUTO_LENGTH,
                          janas_node_constructor,
                          (void *)"a Speech comes from openSpeech()",
                          sizeof(d) / sizeof(d[0]), d, &cls) != napi_ok)
        return NULL;
    return cls;
}
