/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * audio_cmd.c - /audio in janas-chat (audio_cmd.h). The speech model is
 * opened at the first recording and kept: --speech-model's, or the one
 * beside the language model - large-v3-turbo first (near large-v3's words
 * at a fraction of its time), then small, large-v3, any other
 * whisper-*.jns. The text goes with the next message headed by what it is
 * (the file, its length, its language), so that the model knows it reads
 * a transcript and not the user's own words.
 */
#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_cmd.h"
#include "janas/llm.h"

/* the largest recording read: four hours of WAV at 48 kHz, 16-bit stereo */
#define MAX_FILE (4LL * 3600 * 48000 * 4)

struct chat_audio {
    char *speech;     /* asked for; NULL: found beside model_path */
    char *model_path; /* the language model's */
    janas_llm_speech *s;
};

struct chat_audio *chat_audio_new(const char *speech, const char *model_path)
{
    struct chat_audio *a = calloc(1, sizeof(*a));
    if (!a)
        return NULL;
    a->speech = speech ? strdup(speech) : NULL;
    a->model_path = model_path ? strdup(model_path) : NULL;
    if ((speech && !a->speech) || (model_path && !a->model_path)) {
        chat_audio_free(a);
        return NULL;
    }
    return a;
}

void chat_audio_free(struct chat_audio *a)
{
    if (!a)
        return;
    if (a->s)
        janas_llm_speech_close(a->s);
    free(a->speech);
    free(a->model_path);
    free(a);
}

static int exists(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (f)
        fclose(f);
    return f != NULL;
}

/* a Whisper model in the language model's directory, into buf; 0 or -1 */
static int beside(const char *model_path, char *buf, size_t cap)
{
    static const char *const first[] = {"whisper-large-v3-turbo.jns",
                                        "whisper-small.jns",
                                        "whisper-large-v3.jns"};
    char dir[4096];
    const char *slash = model_path ? strrchr(model_path, '/') : NULL;
    if (slash)
        snprintf(dir, sizeof(dir), "%.*s", (int)(slash - model_path),
                 model_path);
    else
        snprintf(dir, sizeof(dir), ".");
    for (size_t i = 0; i < sizeof(first) / sizeof(*first); i++)
        if ((size_t)snprintf(buf, cap, "%s/%s", dir, first[i]) < cap &&
            exists(buf))
            return 0;
    /* any other, the first by name */
    DIR *d = opendir(dir);
    char best[256] = "";
    for (struct dirent *e; d && (e = readdir(d));) {
        size_t n = strlen(e->d_name);
        if (n > 12 && n < sizeof(best) && !strncmp(e->d_name, "whisper-", 8) &&
            !strcmp(e->d_name + n - 4, ".jns") &&
            (!best[0] || strcmp(e->d_name, best) < 0))
            memcpy(best, e->d_name, n + 1);
    }
    if (d)
        closedir(d);
    return best[0] && (size_t)snprintf(buf, cap, "%s/%s", dir, best) < cap ? 0
                                                                           : -1;
}

static void sayf(void (*say)(void *, const char *), void *user, const char *fmt,
                 ...) __attribute__((format(printf, 3, 4)));
static void sayf(void (*say)(void *, const char *), void *user, const char *fmt,
                 ...)
{
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    say(user, line);
}

/* a time as m:ss or h:mm:ss */
static void clock_of(double t, char *buf, size_t cap)
{
    unsigned s = t > 0 && t < 360000 ? (unsigned)(t + 0.5) : 0;
    if (s >= 3600)
        snprintf(buf, cap, "%u:%02u:%02u", s / 3600, s / 60 % 60, s % 60);
    else
        snprintf(buf, cap, "%u:%02u", s / 60, s % 60);
}

struct live {
    janas_llm_speech *s;
    void (*say)(void *, const char *);
    void *user;
};

/* each segment as it is final */
static void on_segment(void *user, int32_t i)
{
    struct live *l = user;
    double t0 = 0, t1 = 0;
    char text[1024], at[24];
    int32_t len = 0;
    janas_llm_speech_segment(l->s, i, &t0, &t1, text, sizeof(text), &len);
    clock_of(t0, at, sizeof(at));
    sayf(l->say, l->user, "  [%s]%s", at, text);
}

static void *read_file(const char *path, long long *n)
{
    FILE *f = fopen(path, "rb");
    char *data = NULL;
    *n = -1;
    if (f && fseek(f, 0, SEEK_END) == 0 && (*n = ftell(f)) > 0 &&
        *n <= MAX_FILE && fseek(f, 0, SEEK_SET) == 0 &&
        (data = malloc((size_t)*n)) &&
        fread(data, 1, (size_t)*n, f) != (size_t)*n) {
        free(data);
        data = NULL;
    }
    if (f)
        fclose(f);
    return data;
}

char *chat_audio_run(struct chat_audio *a, const char *arg,
                     void (*say)(void *user, const char *line), void *user)
{
    int translate = 0;
    while (*arg == ' ')
        arg++;
    if (!strncmp(arg, "translate ", 10)) {
        translate = 1;
        arg += 10;
        while (*arg == ' ')
            arg++;
    }
    if (!*arg) {
        say(user, "/audio [translate] <file>: a recording (WAV, FLAC, MP3, "
                  "Ogg, M4A, WebM)");
        return NULL;
    }
    if (!a->s) {
        char found[4096];
        const char *path = a->speech;
        if (!path && beside(a->model_path, found, sizeof(found)) == 0)
            path = found;
        if (!path) {
            say(user, "no speech model: put a Whisper model (made by "
                      "hf2jns_whisper) beside the model, or start with "
                      "--speech-model <file>");
            return NULL;
        }
        sayf(say, user, "opening %s", path);
        if (janas_llm_speech_open(path, NULL, &a->s) != JANAS_LLM_OK) {
            sayf(say, user, "%s", janas_llm_last_error());
            a->s = NULL;
            return NULL;
        }
    }
    long long n;
    void *data = read_file(arg, &n);
    if (!data) {
        sayf(say, user, n > MAX_FILE ? "%s is too large" : "cannot read %s",
             arg);
        return NULL;
    }
    struct live l = {.s = a->s, .say = say, .user = user};
    struct janas_llm_transcribe_params p;
    janas_llm_transcribe_params_default(&p);
    p.translate = translate;
    p.on_segment = on_segment;
    p.user = &l;
    int32_t rc = janas_llm_speech_transcribe(a->s, data, n, &p);
    free(data);
    if (rc != JANAS_LLM_OK) {
        sayf(say, user, "%s", janas_llm_last_error());
        return NULL;
    }
    char lang[16] = "", dur[24];
    janas_llm_speech_language(a->s, lang, sizeof(lang), NULL);
    const char *lname = janas_llm_speech_language_name(lang);
    char name_buf[32], *name = NULL;
    if (lname && snprintf(name_buf, sizeof(name_buf), "%s", lname) > 0) {
        name = name_buf; /* "english" as a word in a sentence: "English" */
        if (name[0] >= 'a' && name[0] <= 'z')
            name[0] = (char)(name[0] - 'a' + 'A');
    }
    clock_of(janas_llm_speech_duration(a->s), dur, sizeof(dur));
    int32_t len = 0;
    janas_llm_speech_text(a->s, NULL, 0, &len);
    const char *base = strrchr(arg, '/');
    base = base ? base + 1 : arg;
    char head[512];
    int hn =
        snprintf(head, sizeof(head), "%s of the recording %s (%s%s%s):\n",
                 translate ? "English translation" : "Transcript", base, dur,
                 !name       ? ""
                 : translate ? ", spoken in "
                             : ", ",
                 name ? name : "");
    if (hn < 0 || hn >= (int)sizeof(head))
        hn = snprintf(head, sizeof(head), "Transcript of a recording:\n");
    char *out = malloc((size_t)hn + (size_t)len + 1);
    if (!out) {
        say(user, "out of memory");
        return NULL;
    }
    memcpy(out, head, (size_t)hn);
    janas_llm_speech_text(a->s, out + hn, len + 1, NULL);
    /* the model writes a space first */
    if (out[hn] == ' ')
        memmove(out + hn, out + hn + 1, (size_t)len);
    sayf(say, user, "%s, %s: the %s goes with your next message", dur,
         name ? name : lang, translate ? "translation" : "transcript");
    return out;
}
