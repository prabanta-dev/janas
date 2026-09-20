/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tuner.c - self-tuning on real passes (see tuner.h).
 */
#include "tuner.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define WARM_PASSES 1   /* first pass of a class per session: not measured */
#define MARGIN 0.97     /* an alternative must beat the default by 3% */
#define TRIES 3         /* measurements of each candidate before choosing */
#define RETRY_EVERY 128 /* passes between tries of an alternative, at first */
#define RETRY_MAX 1024  /* at most, after tries that changed nothing */

int janas_tune_class(uint32_t n_tokens)
{
    return n_tokens <= 1 ? 0 : (n_tokens <= 16 ? 1 : 2);
}

void janas_tuner_init(struct janas_tuner *t, const struct janas_tune_cand *c,
                      int n, const int def[JANAS_TUNE_CLASSES], int fixed)
{
    memset(t, 0, sizeof(*t));
    t->forced = -1;
    if (n > JANAS_TUNE_MAX_CANDS)
        n = JANAS_TUNE_MAX_CANDS;
    memcpy(t->cand, c, (size_t)n * sizeof(*c));
    t->n_cand = n;
    t->fixed = fixed || n < 2;
    for (int k = 0; k < JANAS_TUNE_CLASSES; k++) {
        t->cls[k].def = def[k] < n ? def[k] : 0;
        t->cls[k].retry_at = t->cls[k].retry_gap = RETRY_EVERY;
        t->cls[k].retry_best = -1;
    }
}

/* GPU candidates: only when allowed, and never for one token (the GPU
   takes no part in single-token products: they would differ only by the
   power its keep-alive costs) */
static int usable(const struct janas_tuner *t, int cls, int i)
{
    return !t->cand[i].gpu || (t->allow_gpu && cls > 0);
}

void janas_tuner_force(struct janas_tuner *t, int cand)
{
    t->forced = cand >= 0 && cand < t->n_cand ? cand : -1;
}

void janas_tuner_allow_gpu(struct janas_tuner *t, int allow)
{
    t->allow_gpu = allow;
}

int janas_tuner_best(const struct janas_tuner *t, int cls)
{
    /* the default unless an alternative is clearly faster: measurements
       are noisy, a near tie is not worth a change */
    int def = t->cls[cls].def, best = def;
    double bar =
        t->cls[cls].tried[def] >= TRIES ? t->cls[cls].cost[def] * MARGIN : 1e30;
    for (int i = 0; i < t->n_cand; i++)
        if (i != def && usable(t, cls, i) && t->cls[cls].tried[i] >= TRIES &&
            t->cls[cls].cost[i] < bar) {
            best = i;
            bar = t->cls[cls].cost[i];
        }
    return best;
}

int janas_tuner_pick(struct janas_tuner *t, int cls)
{
    if (t->forced >= 0) {
        t->cls[cls].passes++;
        return t->forced;
    }
    if (t->fixed)
        return t->cls[cls].def;
    uint32_t p = t->cls[cls].passes++;
    if (p < WARM_PASSES)
        return t->cls[cls].def;
    /* explore: the least tried candidate until each has TRIES */
    int least = -1;
    for (int i = 0; i < t->n_cand; i++)
        if (usable(t, cls, i) &&
            (least < 0 || t->cls[cls].tried[i] < t->cls[cls].tried[least]))
            least = i;
    if (least >= 0 && t->cls[cls].tried[least] < TRIES)
        return least;
    /*
     * Exploit, trying the longest untried alternative now and then, to
     * follow the machine's state. A try is a pass run slower on purpose, so
     * while tries change nothing they come ever more rarely: at a fixed
     * 128 passes, a 262,144-token prefill in blocks of 256 showed one slow
     * block every 32,768 tokens (issue #9). One that changes the choice
     * brings them back to 128.
     */
    int best = janas_tuner_best(t, cls);
    if (t->cls[cls].retry_best >= 0) { /* the last try has been measured */
        uint32_t g = t->cls[cls].retry_gap;
        t->cls[cls].retry_gap = best != t->cls[cls].retry_best ? RETRY_EVERY
                                : g < RETRY_MAX                ? 2 * g
                                                               : RETRY_MAX;
        t->cls[cls].retry_best = -1;
    }
    if (p >= t->cls[cls].retry_at) {
        int old = -1;
        for (int i = 0; i < t->n_cand; i++)
            if (i != best && usable(t, cls, i) &&
                (old < 0 || t->cls[cls].last[i] < t->cls[cls].last[old]))
                old = i;
        t->cls[cls].retry_at = p + t->cls[cls].retry_gap;
        if (old >= 0) {
            t->cls[cls].retry_best = best;
            t->cls[cls].retry_cand = old;
            return old;
        }
    }
    return best;
}

void janas_tuner_record(struct janas_tuner *t, int cls, int cand,
                        double sec_per_token)
{
    if ((t->fixed && t->forced < 0) || t->cls[cls].passes <= WARM_PASSES)
        return;
    /* a try that beats the choice clearly: tries often again, until the
       averages settle which is better (one pass alone may be luck) */
    int rb = t->cls[cls].retry_best;
    if (rb >= 0 && cand == t->cls[cls].retry_cand &&
        sec_per_token < t->cls[cls].cost[rb] * MARGIN) {
        t->cls[cls].retry_gap = RETRY_EVERY;
        t->cls[cls].retry_at = t->cls[cls].passes + RETRY_EVERY;
        t->cls[cls].retry_best = -1;
    }
    double *c = &t->cls[cls].cost[cand];
    uint32_t n = t->cls[cls].tried[cand];
    /* a plain mean over the first tries, then a moving average */
    *c = n == 0      ? sec_per_token
         : n < TRIES ? (*c * n + sec_per_token) / (n + 1)
                     : 0.8 * *c + 0.2 * sec_per_token;
    t->cls[cls].tried[cand]++;
    t->cls[cls].last[cand] = t->cls[cls].passes;
}

int janas_cache_dir(char *path, size_t len, int create)
{
    const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
    int n = xdg && *xdg ? snprintf(path, len, "%s/janas", xdg)
            : home      ? snprintf(path, len, "%s/.cache/janas", home)
                        : -1;
    if (n < 0 || (size_t)n + 32 > len)
        return -1;
    if (create) {
        /* every level that is missing, not only the last two: a cache
           home given by XDG_CACHE_HOME need not exist yet */
        for (char *c = path + 1; *c; c++)
            if (*c == '/') {
                *c = 0;
                mkdir(path, 0755);
                *c = '/';
            }
        if (mkdir(path, 0755) != 0 && errno != EEXIST)
            return -1;
    }
    return 0;
}

/* ~/.cache/janas/tuning.txt (XDG_CACHE_HOME honoured), created if needed. */
static int tuning_path(char *path, size_t len, int create)
{
    if (janas_cache_dir(path, len, create) != 0)
        return -1;
    strcat(path, "/tuning.txt");
    return 0;
}

/* Lines: key TAB class TAB candidate TAB cost TAB tries. */
void janas_tuner_load(struct janas_tuner *t, const char *key)
{
    char path[1024], line[1024];
    if (t->fixed || tuning_path(path, sizeof(path), 0) != 0)
        return;
    FILE *f = fopen(path, "r");
    if (!f)
        return;
    size_t kl = strlen(key);
    while (fgets(line, sizeof(line), f)) {
        int cls, cand;
        double cost;
        unsigned tries;
        if (strncmp(line, key, kl) != 0 || line[kl] != '\t' ||
            sscanf(line + kl + 1, "%d\t%d\t%lf\t%u", &cls, &cand, &cost,
                   &tries) != 4 ||
            cls < 0 || cls >= JANAS_TUNE_CLASSES || cand < 0 ||
            cand >= t->n_cand)
            continue;
        t->cls[cls].cost[cand] = cost;
        t->cls[cls].tried[cand] = tries;
    }
    fclose(f);
}

void janas_tuner_save(const struct janas_tuner *t, const char *key)
{
    char path[1024], tmp[1100], line[1024];
    if (t->fixed || tuning_path(path, sizeof(path), 1) != 0)
        return;
    snprintf(tmp, sizeof(tmp), "%s.%d", path, (int)getpid());
    FILE *out = fopen(tmp, "w");
    if (!out)
        return;
    /* other keys kept as they are */
    FILE *in = fopen(path, "r");
    size_t kl = strlen(key);
    while (in && fgets(line, sizeof(line), in))
        if (!(strncmp(line, key, kl) == 0 && line[kl] == '\t'))
            fputs(line, out);
    if (in)
        fclose(in);
    for (int k = 0; k < JANAS_TUNE_CLASSES; k++)
        for (int i = 0; i < t->n_cand; i++)
            if (t->cls[k].tried[i] > 0)
                fprintf(out, "%s\t%d\t%d\t%.9g\t%u\n", key, k, i,
                        t->cls[k].cost[i], t->cls[k].tried[i]);
    if (fclose(out) == 0)
        rename(tmp, path);
    else
        remove(tmp);
}

void janas_tuner_report(const struct janas_tuner *t, char *buf, size_t len)
{
    static const char *name[] = {"one token", "2-16 tokens", "blocks"};
    size_t w = 0;
    buf[0] = 0;
    for (int k = 0; k < JANAS_TUNE_CLASSES && w < len; k++) {
        int b = janas_tuner_best(t, k);
        w += (size_t)snprintf(buf + w, len - w, "%s%s: %d threads%s",
                              k ? "; " : "", name[k], t->cand[b].threads,
                              t->cand[b].gpu ? " + GPU" : "");
        if (t->cls[k].tried[b] >= TRIES && w < len)
            w += (size_t)snprintf(buf + w, len - w, " (%.1f ms/token)",
                                  1e3 * t->cls[k].cost[b]);
    }
}
