/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * whisper_run.c - a whole recording transcribed (whisper.h), as OpenAI's
 * transcribe() goes through it.
 *
 * The spectrogram of the sound and thirty seconds of silence after it; a
 * window of thirty seconds at a time, the language found on the first when
 * it is not given. In each window the decoder starts from the text before
 * it (<|startofprev|>, at most half the context), then <|startoftranscript|>,
 * the language and the task, and picks tokens - the likeliest, or sampled
 * at a temperature - under OpenAI's rules: no blank at the start, never the
 * suppressed tokens, timestamps in pairs and never going back, the first
 * within a second, a timestamp whenever all of them together are likelier
 * than any text. A window whose text repeats itself (its length over its
 * zlib-compressed length above 2.4, as OpenAI measures it, by
 * common/deflate.c) or is unlikely (average log-probability below -1) is
 * decoded again at a higher
 * temperature, up to 1; one the model calls silence (<|nospeech|> above 0.6
 * at the start, the text unlikely) is skipped. The timestamp pairs cut the
 * window into segments; the next window starts at the last timestamp, or
 * after the whole window when the speech ended in it.
 */
#include "whisper.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/deflate.h"
#include "tokenizer.h"

#define LOGPROB_MIN (-1.0f)
#define NO_SPEECH 0.6f
#define COMPRESSION_MAX 2.4f

static const char *const LANGS[][2] = {
    {"en", "english"},     {"zh", "chinese"},    {"de", "german"},
    {"es", "spanish"},     {"ru", "russian"},    {"ko", "korean"},
    {"fr", "french"},      {"ja", "japanese"},   {"pt", "portuguese"},
    {"tr", "turkish"},     {"pl", "polish"},     {"ca", "catalan"},
    {"nl", "dutch"},       {"ar", "arabic"},     {"sv", "swedish"},
    {"it", "italian"},     {"id", "indonesian"}, {"hi", "hindi"},
    {"fi", "finnish"},     {"vi", "vietnamese"}, {"he", "hebrew"},
    {"uk", "ukrainian"},   {"el", "greek"},      {"ms", "malay"},
    {"cs", "czech"},       {"ro", "romanian"},   {"da", "danish"},
    {"hu", "hungarian"},   {"ta", "tamil"},      {"no", "norwegian"},
    {"th", "thai"},        {"ur", "urdu"},       {"hr", "croatian"},
    {"bg", "bulgarian"},   {"lt", "lithuanian"}, {"la", "latin"},
    {"mi", "maori"},       {"ml", "malayalam"},  {"cy", "welsh"},
    {"sk", "slovak"},      {"te", "telugu"},     {"fa", "persian"},
    {"lv", "latvian"},     {"bn", "bengali"},    {"sr", "serbian"},
    {"az", "azerbaijani"}, {"sl", "slovenian"},  {"kn", "kannada"},
    {"et", "estonian"},    {"mk", "macedonian"}, {"br", "breton"},
    {"eu", "basque"},      {"is", "icelandic"},  {"hy", "armenian"},
    {"ne", "nepali"},      {"mn", "mongolian"},  {"bs", "bosnian"},
    {"kk", "kazakh"},      {"sq", "albanian"},   {"sw", "swahili"},
    {"gl", "galician"},    {"mr", "marathi"},    {"pa", "punjabi"},
    {"si", "sinhala"},     {"km", "khmer"},      {"sn", "shona"},
    {"yo", "yoruba"},      {"so", "somali"},     {"af", "afrikaans"},
    {"oc", "occitan"},     {"ka", "georgian"},   {"be", "belarusian"},
    {"tg", "tajik"},       {"sd", "sindhi"},     {"gu", "gujarati"},
    {"am", "amharic"},     {"yi", "yiddish"},    {"lo", "lao"},
    {"uz", "uzbek"},       {"fo", "faroese"},    {"ht", "haitian creole"},
    {"ps", "pashto"},      {"tk", "turkmen"},    {"nn", "nynorsk"},
    {"mt", "maltese"},     {"sa", "sanskrit"},   {"lb", "luxembourgish"},
    {"my", "myanmar"},     {"bo", "tibetan"},    {"tl", "tagalog"},
    {"mg", "malagasy"},    {"as", "assamese"},   {"tt", "tatar"},
    {"haw", "hawaiian"},   {"ln", "lingala"},    {"ha", "hausa"},
    {"ba", "bashkir"},     {"jw", "javanese"},   {"su", "sundanese"},
    {"yue", "cantonese"},
};

const char *janas_whisper_lang_name(const char *code)
{
    for (size_t i = 0; i < sizeof LANGS / sizeof *LANGS; i++)
        if (!strcmp(LANGS[i][0], code))
            return LANGS[i][1];
    return NULL;
}

static int same_ci(const char *a, const char *b)
{
    for (; *a && *b; a++, b++)
        if ((*a | 0x20) != (*b | 0x20))
            return 0;
    return !*a && !*b;
}

int janas_whisper_lang_index(const struct janas_whisper *w, const char *s)
{
    const char *code = s;
    for (size_t i = 0; i < sizeof LANGS / sizeof *LANGS; i++)
        if (same_ci(LANGS[i][1], s))
            code = LANGS[i][0];
    char c[16];
    for (uint32_t i = 0; i < w->n_lang; i++) {
        janas_whisper_lang(w, i, c, sizeof c);
        if (same_ci(c, code))
            return (int)i;
    }
    return -1;
}

/* ---- a window decoded ---- */

struct run {
    struct janas_whisper *w;
    const struct janas_whisper_opts *o;
    int lang; /* index, -1 for an English-only model */
    float *logits, *buf;
    uint8_t *suppress; /* by token: never sampled */
    uint64_t rng;
    int32_t *all; /* the tokens of the segments so far */
    size_t n_all, cap_all, prompt_from;
};

struct window {
    int32_t tok[512];
    float lp[512]; /* each token's log-probability */
    uint32_t n;
    double sum_logprob;
    float no_speech, temperature, avg_logprob, compression_ratio;
};

static double rnd(struct run *r)
{
    /* splitmix64 */
    uint64_t z = (r->rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (double)(z >> 11) / 9007199254740992.0;
}

/* OpenAI's logit filters on the logits of the next token, given those
   sampled in the window so far */
static void filter(struct run *r, float *l, const int32_t *seq, uint32_t n)
{
    struct janas_whisper *w = r->w;
    uint32_t V = w->vocab, tb = (uint32_t)w->ts_begin;
    if (n == 0)
        for (uint64_t i = 0; i < w->n_begin_suppress; i++)
            if ((uint32_t)w->begin_suppress[i] < V)
                l[w->begin_suppress[i]] = -INFINITY;
    for (uint32_t i = 0; i < V; i++)
        if (r->suppress[i])
            l[i] = -INFINITY;
    l[w->no_timestamps] = -INFINITY;
    int last = n >= 1 && seq[n - 1] >= (int32_t)tb;
    int penult = n < 2 || seq[n - 2] >= (int32_t)tb;
    if (last) {
        if (penult)
            for (uint32_t i = tb; i < V; i++)
                l[i] = -INFINITY;
        else
            for (uint32_t i = 0; i < (uint32_t)w->eot; i++)
                l[i] = -INFINITY;
    }
    int32_t ts_last = -1;
    for (uint32_t i = 0; i < n; i++)
        if (seq[i] >= (int32_t)tb)
            ts_last = seq[i];
    if (ts_last >= 0) {
        uint32_t upto = (uint32_t)ts_last + !(last && !penult);
        for (uint32_t i = tb; i < upto && i < V; i++)
            l[i] = -INFINITY;
    }
    if (n == 0) {
        for (uint32_t i = 0; i < tb; i++)
            l[i] = -INFINITY;
        for (uint32_t i = tb + w->max_initial_ts + 1; i < V; i++)
            l[i] = -INFINITY;
    }
    /* a timestamp when they are likelier together than any text */
    float mx = -INFINITY;
    for (uint32_t i = 0; i < V; i++)
        if (l[i] > mx)
            mx = l[i];
    double ts = 0, best_text = -INFINITY;
    for (uint32_t i = 0; i < V; i++) {
        if (l[i] == -INFINITY)
            continue;
        if (i >= tb)
            ts += exp((double)l[i] - mx);
        else if (l[i] > best_text)
            best_text = l[i];
    }
    if (ts > 0 && log(ts) + mx > best_text)
        for (uint32_t i = 0; i < tb; i++)
            l[i] = -INFINITY;
}

/* log of the sum of e^l, the largest mx */
static double log_sum(const float *l, uint32_t n, float mx)
{
    double s = 0;
    for (uint32_t i = 0; i < n; i++)
        if (l[i] != -INFINITY)
            s += exp((double)l[i] - mx);
    return log(s) + mx;
}

static int32_t pick(struct run *r, const float *l, float temp, double *lp)
{
    uint32_t V = r->w->vocab, best = 0;
    for (uint32_t i = 1; i < V; i++)
        if (l[i] > l[best])
            best = i;
    double lse = log_sum(l, V, l[best]);
    if (temp > 0) {
        double s = 0, *e = (double *)(void *)r->buf;
        for (uint32_t i = 0; i < V; i++) {
            e[i] = l[i] == -INFINITY ? 0 : exp((l[i] - l[best]) / temp);
            s += e[i];
        }
        double u = rnd(r) * s;
        for (uint32_t i = 0; i < V; i++) {
            u -= e[i];
            if (u <= 0 && e[i] > 0) {
                best = i;
                break;
            }
        }
    }
    *lp = l[best] - lse;
    return (int32_t)best;
}

/* the text's bytes (its text tokens, stripped) over its zlib-compressed
   bytes, as OpenAI's get_compression_ratio: high when the text goes round
   in a loop */
static float compression_ratio(struct janas_whisper *w, const int32_t *t,
                               uint32_t n)
{
    size_t len = 0;
    char none[1];
    for (uint32_t i = 0; i < n; i++)
        if (t[i] < w->eot)
            len += janas_tokenizer_decode(w->tk, t[i], none, 0);
    char *s = malloc(len + 1), *p = s;
    if (!s)
        return 0;
    for (uint32_t i = 0; i < n; i++)
        if (t[i] < w->eot)
            p += janas_tokenizer_decode(w->tk, t[i], p, len - (size_t)(p - s));
    const char *b = s, *e = s + len;
    while (b < e && (*b == ' ' || *b == '\t' || *b == '\n' || *b == '\r' ||
                     *b == '\v' || *b == '\f'))
        b++;
    while (e > b && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\n' ||
                     e[-1] == '\r' || e[-1] == '\v' || e[-1] == '\f'))
        e--;
    long z = janas_deflate_size((const uint8_t *)b, (size_t)(e - b));
    float ratio = z > 0 ? (float)((double)(e - b) / (double)z) : 0;
    free(s);
    return ratio;
}

static int decode_window(struct run *r, float temp, struct window *win)
{
    struct janas_whisper *w = r->w;
    uint32_t V = w->vocab, half = w->t_ctx / 2;
    int32_t init[512];
    uint32_t ni = 0;
    size_t np = r->n_all - r->prompt_from;
    if (np) {
        size_t keep = np < half - 1 ? np : half - 1;
        init[ni++] = w->sot_prev;
        memcpy(init + ni, r->all + r->n_all - keep, sizeof(int32_t) * keep);
        ni += (uint32_t)keep;
    }
    uint32_t at_sot = ni;
    init[ni++] = w->sot;
    if (r->lang >= 0) {
        init[ni++] = w->sot + 1 + r->lang;
        init[ni++] = r->o->translate ? w->translate : w->transcribe;
    }
    if (janas_whisper_decode(w, init, ni, 0, r->buf, 1) != 0)
        return -1;
    /* the probability of silence, at the start of transcript */
    const float *ls = r->buf + (size_t)(at_sot - 0) * V;
    float mx = -INFINITY;
    for (uint32_t i = 0; i < V; i++)
        if (ls[i] > mx)
            mx = ls[i];
    win->no_speech = (float)exp(ls[w->no_speech] - log_sum(ls, V, mx));
    memcpy(r->logits, r->buf + (size_t)(ni - 1) * V, sizeof(float) * V);
    win->n = 0;
    win->sum_logprob = 0;
    win->temperature = temp;
    uint32_t limit = half < sizeof win->tok / sizeof *win->tok
                         ? half
                         : sizeof win->tok / sizeof *win->tok;
    while (win->n < limit && ni + win->n < w->t_ctx) {
        filter(r, r->logits, win->tok, win->n);
        double lp;
        int32_t t = pick(r, r->logits, temp, &lp);
        win->sum_logprob += lp;
        if (t == w->eot)
            break;
        win->lp[win->n] = (float)lp;
        win->tok[win->n++] = t;
        if (ni + win->n >= w->t_ctx)
            break;
        if (janas_whisper_decode(w, &t, 1, ni + win->n - 1, r->logits, 0))
            return -1;
    }
    win->avg_logprob = (float)(win->sum_logprob / (win->n + 1));
    win->compression_ratio = compression_ratio(w, win->tok, win->n);
    return 0;
}

/* ---- the recording ---- */

static int add_segment(struct janas_whisper_result *res, uint32_t *cap,
                       struct janas_whisper *w, const struct window *win,
                       const int32_t *tok, uint32_t n, double t0, double t1)
{
    /* its text: the tokens before the end of text */
    size_t len = 0;
    char piece[256];
    for (uint32_t i = 0; i < n; i++)
        if (tok[i] < w->eot)
            len += janas_tokenizer_decode(w->tk, tok[i], piece, 0);
    char *text = malloc(len + 1), *p = text;
    if (!text)
        return -1;
    for (uint32_t i = 0; i < n; i++)
        if (tok[i] < w->eot)
            p += janas_tokenizer_decode(w->tk, tok[i], p,
                                        len + 1 - (size_t)(p - text));
    *p = 0;
    int blank = 1;
    for (const char *c = text; *c; c++)
        blank = blank && (*c == ' ' || *c == '\t' || *c == '\n');
    if (blank || t1 <= t0) {
        free(text);
        return 0;
    }
    if (res->n_segs == *cap) {
        uint32_t nc = *cap ? 2 * *cap : 16;
        struct janas_whisper_seg *s =
            realloc(res->segs, sizeof(*s) * (size_t)nc);
        if (!s) {
            free(text);
            return -1;
        }
        res->segs = s;
        *cap = nc;
    }
    struct janas_whisper_seg *s = &res->segs[res->n_segs];
    s->tokens = malloc(sizeof(int32_t) * (n ? n : 1));
    s->logprobs = malloc(sizeof(float) * (n ? n : 1));
    if (!s->tokens || !s->logprobs) {
        free(s->tokens);
        free(s->logprobs);
        free(text);
        return -1;
    }
    memcpy(s->tokens, tok, sizeof(int32_t) * n);
    memcpy(s->logprobs, win->lp + (tok - win->tok), sizeof(float) * n);
    s->n_tokens = n;
    s->t0 = t0;
    s->t1 = t1;
    s->text = text;
    s->avg_logprob = win->avg_logprob;
    s->no_speech_prob = win->no_speech;
    s->temperature = win->temperature;
    s->compression_ratio = win->compression_ratio;
    s->words = NULL;
    s->n_words = 0;
    res->n_segs++;
    return 0;
}

static int remember(struct run *r, const int32_t *t, uint32_t n)
{
    if (r->n_all + n > r->cap_all) {
        size_t nc = (r->n_all + n) * 2;
        int32_t *a = realloc(r->all, sizeof(int32_t) * nc);
        if (!a)
            return -1;
        r->all = a;
        r->cap_all = nc;
    }
    memcpy(r->all + r->n_all, t, sizeof(int32_t) * n);
    r->n_all += n;
    return 0;
}

static int detect_language(struct run *r)
{
    struct janas_whisper *w = r->w;
    if (janas_whisper_decode(w, &w->sot, 1, 0, r->logits, 0) != 0)
        return -1;
    int best = 0;
    for (uint32_t i = 1; i < w->n_lang; i++)
        if (r->logits[w->sot + 1 + i] > r->logits[w->sot + 1 + best])
            best = (int)i;
    return best;
}

/* the sound in memory (pcm), or from a source (snd): the spectrogram
   whole, or a window at a time */
static int transcribe(struct janas_whisper *w, const float *pcm,
                      const struct janas_whisper_sound *snd, size_t n,
                      const struct janas_whisper_opts *o,
                      struct janas_whisper_result *res, char *err,
                      size_t err_len)
{
    memset(res, 0, sizeof *res);
    struct run r = {.w = w, .o = o, .lang = -1, .rng = o->seed};
    struct window *win = malloc(sizeof *win);
    uint32_t frames = 0, cap = 0, win_frames = 2 * w->a_ctx;
    float *mel = NULL;
    int rc = -1;
    r.logits = malloc(sizeof(float) * w->vocab);
    /* the logits of a window's first tokens, at most half the context and
       three; doubles for the sampling */
    r.buf = malloc(sizeof(float) * (size_t)w->vocab * (w->t_ctx / 2 + 4));
    r.suppress = calloc(w->vocab, 1);
    if (!win || !r.logits || !r.buf || !r.suppress) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    for (uint64_t i = 0; i < w->n_suppress; i++)
        if (w->suppress[i] >= 0 && (uint32_t)w->suppress[i] < w->vocab)
            r.suppress[w->suppress[i]] = 1;
    int32_t never[] = {w->translate, w->transcribe, w->sot,
                       w->sot_prev,  w->sot_lm,     w->no_speech};
    for (size_t i = 0; i < sizeof never / sizeof *never; i++)
        r.suppress[never[i]] = 1;
    if (w->multilingual) {
        if (o->language && o->language[0]) {
            r.lang = janas_whisper_lang_index(w, o->language);
            if (r.lang < 0) {
                snprintf(err, err_len,
                         "a language this model does not know: %s",
                         o->language);
                goto done;
            }
        }
    } else if (o->translate) {
        snprintf(err, err_len,
                 "this model knows English only: it does not "
                 "translate");
        goto done;
    }
    if (o->prompt && o->prompt[0]) {
        /* " " + the prompt, as OpenAI encodes it */
        size_t pl = strlen(o->prompt);
        char *sp = malloc(pl + 2);
        if (!sp) {
            snprintf(err, err_len, "out of memory");
            goto done;
        }
        sp[0] = ' ';
        memcpy(sp + 1, o->prompt, pl + 1);
        long k = janas_tokenizer_encode(w->tk, sp, pl + 1, 0, NULL, 0);
        int32_t *pt = k > 0 ? malloc(sizeof(int32_t) * (size_t)k) : NULL;
        if (pt &&
            janas_tokenizer_encode(w->tk, sp, pl + 1, 0, pt, (size_t)k) == k &&
            remember(&r, pt, (uint32_t)k) != 0)
            k = -1;
        free(pt);
        free(sp);
        if (k < 0) {
            snprintf(err, err_len, "out of memory");
            goto done;
        }
    }
    size_t pad = (size_t)w->chunk * w->rate;
    float mx = 0;
    if (pcm)
        mel = janas_whisper_mel(w, pcm, n, pad, &frames);
    else if (janas_whisper_mel_max(w, snd, pad, &mx, err, err_len) != 0 ||
             snd->rewind(snd->user, err, err_len) != 0)
        goto done;
    else
        mel = malloc(sizeof(float) * win_frames * w->n_mels);
    if (!mel) {
        snprintf(err, err_len, "out of memory");
        goto done;
    }
    res->duration = (double)n / w->rate;
    uint32_t content = (uint32_t)(n / w->hop), seek = 0;
    double last_speech = 0;
    double frame_s = (double)w->hop / w->rate; /* 10 ms */
    double ts_s = 2 * frame_s;                 /* a timestamp's step */
    /* the stretches of speech to read (a voice activity detector's), or
       the whole recording */
    size_t ri = 0;
    uint32_t rend = content;
    if (o->n_regions) {
        seek = (uint32_t)(o->regions[0] / w->hop);
        rend = (uint32_t)(o->regions[1] / w->hop);
    }
    for (;;) {
        if (rend > content)
            rend = content;
        if (seek >= rend) {
            if (!o->n_regions || ++ri >= o->n_regions)
                break;
            uint32_t rs = (uint32_t)(o->regions[2 * ri] / w->hop);
            if (seek < rs)
                seek = rs;
            rend = (uint32_t)(o->regions[2 * ri + 1] / w->hop);
            continue;
        }
        if (o->stop && *o->stop) {
            snprintf(err, err_len, "stopped");
            goto done;
        }
        uint32_t size = rend - seek < win_frames ? rend - seek : win_frames;
        if (!pcm && janas_whisper_mel_window(w, snd, pad, mx, seek, size, mel,
                                             win_frames, err, err_len) != 0)
            goto done;
        if (pcm ? janas_whisper_encode(w, mel, frames, seek, size)
                : janas_whisper_encode(w, mel, win_frames, 0, size)) {
            snprintf(err, err_len, "out of memory");
            goto done;
        }
        if (w->multilingual && r.lang < 0 &&
            (r.lang = detect_language(&r)) < 0) {
            snprintf(err, err_len, "the decoder failed");
            goto done;
        }
        float temp = o->temperature;
        for (;;) {
            if (decode_window(&r, temp, win) != 0) {
                snprintf(err, err_len, "the decoder failed");
                goto done;
            }
            int again = win->compression_ratio > COMPRESSION_MAX ||
                        win->avg_logprob < LOGPROB_MIN;
            if (win->no_speech > NO_SPEECH && win->avg_logprob < LOGPROB_MIN)
                again = 0; /* silence: skipped below */
            if (!again || temp >= 1.0f - 1e-6f)
                break;
            temp += 0.2f;
        }
        double offset = seek * frame_s;
        if (win->no_speech > NO_SPEECH && win->avg_logprob < LOGPROB_MIN) {
            seek += size;
            continue;
        }
        int32_t tb = w->ts_begin, *t = win->tok;
        uint32_t nt = win->n, before = res->n_segs;
        int single_end = nt >= 2 && t[nt - 1] >= tb && t[nt - 2] < tb;
        uint32_t last = 0, cuts = 0;
        for (uint32_t i = 1; i < nt; i++)
            if (t[i] >= tb && t[i - 1] >= tb) {
                double t0 = offset + (t[last] - tb) * ts_s;
                double t1 = offset + (t[i - 1] - tb) * ts_s;
                if (add_segment(res, &cap, w, win, t + last, i - last, t0, t1))
                    goto oom;
                last = i;
                cuts++;
            }
        if (cuts) {
            if (single_end) {
                double t0 = offset + (t[last] - tb) * ts_s;
                double t1 = offset + (t[nt - 1] - tb) * ts_s;
                if (add_segment(res, &cap, w, win, t + last, nt - last, t0, t1))
                    goto oom;
                seek += size;
            } else {
                seek += (uint32_t)(t[last - 1] - tb) * 2;
            }
        } else {
            double dur = size * frame_s;
            for (uint32_t i = nt; i-- > 0;)
                if (t[i] >= tb) {
                    if (t[i] != tb)
                        dur = (t[i] - tb) * ts_s;
                    break;
                }
            if (add_segment(res, &cap, w, win, t, nt, offset, offset + dur))
                goto oom;
            seek += size;
        }
        if (o->words && res->n_segs > before) {
            /* the words' times; then, as OpenAI's transcribe does, the
               next window starts at the last word's end */
            double ls = last_speech;
            if (janas_whisper_align(w, r.lang, o->translate, res->segs, before,
                                    res->n_segs, offset, size, &ls) != 0)
                goto oom;
            double last_end = -1;
            for (uint32_t s = res->n_segs; s-- > before;)
                if (res->segs[s].n_words) {
                    last_end = res->segs[s].words[res->segs[s].n_words - 1].t1;
                    break;
                }
            if (!single_end && last_end > offset)
                seek = (uint32_t)lround(last_end / frame_s);
            if (last_end >= 0)
                last_speech = last_end;
        }
        for (uint32_t s = before; s < res->n_segs; s++) {
            if (res->segs[s].t1 > res->duration)
                res->segs[s].t1 = res->duration;
            if (remember(&r, res->segs[s].tokens, res->segs[s].n_tokens))
                goto oom;
        }
        if (win->temperature > 0.5f)
            r.prompt_from = r.n_all;
        if (o->on_segment)
            for (uint32_t s = before; s < res->n_segs; s++)
                o->on_segment(o->user, s);
    }
    if (r.lang >= 0)
        janas_whisper_lang(w, (uint32_t)r.lang, res->language,
                           sizeof res->language);
    else
        snprintf(res->language, sizeof res->language, "en");
    size_t len = 0;
    for (uint32_t s = 0; s < res->n_segs; s++)
        len += strlen(res->segs[s].text);
    res->text = malloc(len + 1);
    if (!res->text)
        goto oom;
    char *p = res->text;
    for (uint32_t s = 0; s < res->n_segs; s++) {
        size_t k = strlen(res->segs[s].text);
        memcpy(p, res->segs[s].text, k);
        p += k;
    }
    *p = 0;
    /* trimmed */
    char *b = res->text;
    while (*b == ' ' || *b == '\n')
        b++;
    memmove(res->text, b, strlen(b) + 1);
    for (size_t k = strlen(res->text);
         k && (res->text[k - 1] == ' ' || res->text[k - 1] == '\n'); k--)
        res->text[k - 1] = 0;
    rc = 0;
    goto done;
oom:
    snprintf(err, err_len, "out of memory");
done:
    if (rc != 0)
        janas_whisper_result_free(res);
    free(mel);
    free(win);
    free(r.logits);
    free(r.buf);
    free(r.suppress);
    free(r.all);
    return rc;
}

int janas_whisper_transcribe(struct janas_whisper *w, const float *pcm,
                             size_t n, const struct janas_whisper_opts *o,
                             struct janas_whisper_result *res, char *err,
                             size_t err_len)
{
    static const float none = 0;
    return transcribe(w, pcm ? pcm : &none, NULL, pcm ? n : 0, o, res, err,
                      err_len);
}

int janas_whisper_transcribe_sound(struct janas_whisper *w,
                                   const struct janas_whisper_sound *snd,
                                   const struct janas_whisper_opts *o,
                                   struct janas_whisper_result *res, char *err,
                                   size_t err_len)
{
    return transcribe(w, NULL, snd, snd->n, o, res, err, err_len);
}

void janas_whisper_result_free(struct janas_whisper_result *r)
{
    for (uint32_t i = 0; i < r->n_segs; i++) {
        free(r->segs[i].text);
        free(r->segs[i].tokens);
        free(r->segs[i].logprobs);
        for (uint32_t k = 0; k < r->segs[i].n_words; k++)
            free(r->segs[i].words[k].text);
        free(r->segs[i].words);
    }
    free(r->segs);
    free(r->text);
    memset(r, 0, sizeof *r);
}
