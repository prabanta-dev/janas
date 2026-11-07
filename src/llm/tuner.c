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
#define RESETTLE_WINS 3 /* tries in a row won by other threads: theirs */
#define DECISIVE 0.5    /* ... or one try this much faster than the choice */
#define RETRY_WON 16    /* passes to the next try after one that won */

/* JANAS_TUNE_TRACE=1: the threads settled, every try of an alternative
   after that (against the choice's mean) and the threads moved, on
   stderr: what a long chat does, which the tests only simulate */
static int tracing(void)
{
    static int on = -1;
    if (on < 0)
        on = getenv("JANAS_TUNE_TRACE") && atoi(getenv("JANAS_TUNE_TRACE")) > 0;
    return on;
}

int janas_tune_class(uint32_t n_tokens)
{
    return n_tokens <= 1                  ? 0
           : n_tokens <= 16               ? 1
           : n_tokens >= JANAS_TUNE_BLOCK ? 2
                                          : 3;
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
        t->cls[k].alone = -1;
    }
}

/* GPU candidates: only when allowed, and for one token only where the
   GPU takes single-token products (elsewhere they would differ only by the
   power its keep-alive costs) */
/* the user's choice of devices for the class allows this candidate */
static int allowed(const struct janas_tuner *t, int cls, int i)
{
    int combo = (t->cand[i].gpu ? 1 : 0) | (t->cand[i].npu ? 2 : 0);
    return !t->allowed[cls] || (t->allowed[cls] >> combo & 1);
}

/* the CPU alone for blocks only; once settled, the side settled */
static int same_side(const struct janas_tuner *t, int cls, int i)
{
    if (cls != 2)
        return !t->cand[i].alone;
    return t->cls[cls].alone < 0 || t->cand[i].alone == t->cls[cls].alone;
}

static int usable(const struct janas_tuner *t, int cls, int i)
{
    return allowed(t, cls, i) && same_side(t, cls, i) &&
           (!t->cand[i].npu || cls == 2) &&
           (!t->cand[i].gpu || (t->allow_gpu && (cls > 0 || t->gpu_one))) &&
           (!t->cls[cls].threads || t->cand[i].threads == t->cls[cls].threads);
}

/* A candidate a try may run: as usable, but any threads - once they are
   settled, other threads are tried too, each try a fresh pass against the
   choice's recent mean, so that a machine that changes (warm, throttled)
   can move them */
static int retryable(const struct janas_tuner *t, int cls, int i)
{
    return allowed(t, cls, i) && same_side(t, cls, i) &&
           (!t->cand[i].npu || cls == 2) &&
           (!t->cand[i].gpu || (t->allow_gpu && (cls > 0 || t->gpu_one)));
}

/* The default of a class: its own, or once the threads are settled the
   candidate with those threads and the default's devices (no GPU unless
   the default has it), on the side settled for blocks (the CPU alone or
   not) */
static int class_def(const struct janas_tuner *t, int cls)
{
    int def = t->cls[cls].def;
    /* the user's devices: the allowed candidate of the default's threads
       (or the settled ones), else the first allowed */
    if (!allowed(t, cls, def)) {
        int th = t->cls[cls].threads ? t->cls[cls].threads
                                     : t->cand[def].threads,
            first = -1;
        for (int i = 0; i < t->n_cand; i++)
            if (allowed(t, cls, i) && same_side(t, cls, i) &&
                (!t->cand[i].npu || cls == 2)) {
                if (t->cand[i].threads == th)
                    return i;
                if (first < 0)
                    first = i;
            }
        if (first >= 0)
            return first;
    }
    int alone = t->cls[cls].alone > 0;
    if (!t->cls[cls].threads || (t->cand[def].threads == t->cls[cls].threads &&
                                 t->cand[def].alone == alone))
        return def;
    for (int i = 0; i < t->n_cand; i++)
        if (!t->cand[i].gpu && !t->cand[i].npu && t->cand[i].alone == alone &&
            allowed(t, cls, i) && t->cand[i].threads == t->cls[cls].threads)
            return i;
    return def;
}

/* Once every candidate has its tries, the threads of the best are kept */
static void settle_threads(struct janas_tuner *t, int cls)
{
    if (t->cls[cls].threads)
        return;
    for (int i = 0; i < t->n_cand; i++)
        if (usable(t, cls, i) && t->cls[cls].tried[i] < TRIES)
            return;
    int b = janas_tuner_best(t, cls);
    t->cls[cls].threads = t->cand[b].threads;
    if (cls == 2)
        t->cls[cls].alone = t->cand[b].alone;
    if (tracing())
        fprintf(stderr, "tune class %d pass %u: threads settled on %d%s\n", cls,
                t->cls[cls].passes, t->cls[cls].threads,
                t->cand[b].alone ? ", the CPU alone" : "");
}

const char *janas_tune_devices(const struct janas_tune_cand *c)
{
    return c->alone           ? ", the CPU alone"
           : c->gpu && c->npu ? " + GPU + NPU"
           : c->gpu           ? " + GPU"
           : c->npu           ? " + NPU"
                              : "";
}

void janas_tuner_allow(struct janas_tuner *t, int cls, int mask)
{
    if (cls < 0 || cls >= JANAS_TUNE_CLASSES)
        return;
    t->allowed[cls] = mask;
    /* the default among the allowed (a fixed tuner runs it as it is) */
    t->cls[cls].def = class_def(t, cls);
}

void janas_tuner_force(struct janas_tuner *t, int cand)
{
    t->forced = cand >= 0 && cand < t->n_cand ? cand : -1;
}

void janas_tuner_allow_gpu(struct janas_tuner *t, int allow)
{
    t->allow_gpu = allow;
}

void janas_tuner_allow_gpu_one(struct janas_tuner *t, int allow)
{
    t->gpu_one = allow;
}

int janas_tuner_best(const struct janas_tuner *t, int cls)
{
    /* the default unless an alternative is clearly faster: measurements
       are noisy, a near tie is not worth a change. The default's mean
       counts from its first measurement: waiting for all its tries let any
       alternative tried three times win, even a slower one - Qwen3.5-2B's
       blocks went to 20 threads (3.68 ms a token) over the default's 12
       (3.22, measured twice) (10 Oct 2026) */
    int def = class_def(t, cls), best = def;
    double bar =
        t->cls[cls].tried[def] > 0 ? t->cls[cls].cost[def] * MARGIN : 1e30;
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
    t->trying = 0;
    if (t->forced >= 0) {
        t->cls[cls].passes++;
        return t->forced;
    }
    if (t->fixed)
        return t->cls[cls].def;
    uint32_t p = t->cls[cls].passes++;
    if (p < WARM_PASSES)
        return class_def(t, cls);
    settle_threads(t, cls);
    /* explore: the least tried candidate until each has TRIES */
    int least = -1;
    for (int i = 0; i < t->n_cand; i++)
        if (usable(t, cls, i) &&
            (least < 0 || t->cls[cls].tried[i] < t->cls[cls].tried[least]))
            least = i;
    if (least >= 0 && t->cls[cls].tried[least] < TRIES) {
        t->trying = 1;
        return least;
    }
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
            if (i != best && retryable(t, cls, i) &&
                (old < 0 || t->cls[cls].last[i] < t->cls[cls].last[old]))
                old = i;
        t->cls[cls].retry_at = p + t->cls[cls].retry_gap;
        if (old >= 0) {
            t->cls[cls].retry_best = best;
            t->cls[cls].retry_cand = old;
            t->trying = 1;
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
    int rb = t->cls[cls].retry_best;
    if (rb >= 0 && cand == t->cls[cls].retry_cand) {
        int th = t->cls[cls].threads, other = th && t->cand[cand].threads != th,
            won = sec_per_token < t->cls[cls].cost[rb] * MARGIN;
        /*
         * A try that beats the choice clearly: the next one soon, until the
         * averages settle which is better (one pass alone may be luck).
         * Other threads, faster in try after try - whichever of them, as
         * the tries take the candidates in turn - or one twice as fast:
         * theirs from now on. The threads are settled by the first tries,
         * and the machine can turn their verdict round: a laptop that
         * warms up (Qwen3.5-2B settled on 20 threads for its tokens, which
         * then took 28.1 ms against 18.9 on 12), or another program on the
         * efficiency cores - 20 threads took 680-850 ms a token against 22
         * on 6 or 12, and counted per candidate, the 6 and 12 taking the
         * tries in turn, the wins moved the threads only after 645 tokens,
         * some 450 s (10 Oct 2026). Their mean starts again from this
         * pass: the old one was made at another time.
         */
        if (won) {
            t->cls[cls].retry_gap = RETRY_EVERY;
            t->cls[cls].retry_at = t->cls[cls].passes + RETRY_WON;
            t->cls[cls].retry_best = -1;
        }
        t->cls[cls].wins[cand] = won ? t->cls[cls].wins[cand] + 1 : 0;
        if (other)
            t->cls[cls].other_wins = won ? t->cls[cls].other_wins + 1 : 0;
        if (tracing())
            fprintf(stderr,
                    "tune class %d pass %u: try %d threads%s %.2f ms against "
                    "%d threads%s %.2f ms: %s (%u in a row by other "
                    "threads)\n",
                    cls, t->cls[cls].passes, t->cand[cand].threads,
                    janas_tune_devices(&t->cand[cand]), 1e3 * sec_per_token,
                    t->cand[rb].threads, janas_tune_devices(&t->cand[rb]),
                    1e3 * t->cls[cls].cost[rb], won ? "won" : "lost",
                    (unsigned)t->cls[cls].other_wins);
        if (other && won &&
            (t->cls[cls].wins[cand] >= RESETTLE_WINS ||
             t->cls[cls].other_wins >= RESETTLE_WINS ||
             sec_per_token < t->cls[cls].cost[rb] * DECISIVE)) {
            if (tracing())
                fprintf(stderr, "tune class %d pass %u: threads %d -> %d\n",
                        cls, t->cls[cls].passes, th, t->cand[cand].threads);
            t->cls[cls].threads = t->cand[cand].threads;
            memset(t->cls[cls].wins, 0, sizeof(t->cls[cls].wins));
            t->cls[cls].other_wins = 0;
            t->cls[cls].cost[cand] = sec_per_token;
            t->cls[cls].tried[cand] = TRIES;
            t->cls[cls].last[cand] = t->cls[cls].passes;
            return;
        }
    }
    double *c = &t->cls[cls].cost[cand];
    uint32_t n = t->cls[cls].tried[cand];
    /* the choice suddenly twice as slow as its mean (another program took
       its cores): the next try at once, not at its turn - the 128 passes
       before the first one cost some 140 s of a 3,000-token chat at 760
       ms a token, against 22 on 6 threads (10 Oct 2026) */
    if (rb < 0 && n >= TRIES && sec_per_token > 2 * *c &&
        t->cls[cls].retry_at > t->cls[cls].passes) {
        t->cls[cls].retry_at = t->cls[cls].passes;
        if (tracing())
            fprintf(stderr,
                    "tune class %d pass %u: %d threads%s %.2f ms against a "
                    "mean of %.2f: a try now\n",
                    cls, t->cls[cls].passes, t->cand[cand].threads,
                    janas_tune_devices(&t->cand[cand]), 1e3 * sec_per_token,
                    1e3 * *c);
    }
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
    /* the threads settled: a line of their own, candidate -1; whether the
       CPU reads blocks alone, candidate -2 */
    if (!(f = fopen(path, "r")))
        return;
    while (fgets(line, sizeof(line), f)) {
        int cls, cand;
        double threads;
        unsigned zero;
        if (strncmp(line, key, kl) != 0 || line[kl] != '\t' ||
            sscanf(line + kl + 1, "%d\t%d\t%lf\t%u", &cls, &cand, &threads,
                   &zero) != 4 ||
            cls < 0 || cls >= JANAS_TUNE_CLASSES || cand > -1 || cand < -2)
            continue;
        if (cand == -2) {
            if (cls == 2)
                t->cls[cls].alone = threads != 0;
            continue;
        }
        for (int i = 0; i < t->n_cand; i++)
            if (t->cand[i].threads == (int)threads)
                t->cls[cls].threads = (int)threads;
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
    for (int k = 0; k < JANAS_TUNE_CLASSES; k++)
        if (t->cls[k].threads)
            fprintf(out, "%s\t%d\t-1\t%d\t0\n", key, k, t->cls[k].threads);
    if (t->cls[2].alone >= 0)
        fprintf(out, "%s\t2\t-2\t%d\t0\n", key, t->cls[2].alone);
    if (fclose(out) == 0)
        rename(tmp, path);
    else
        remove(tmp);
}

void janas_tuner_report(const struct janas_tuner *t, char *buf, size_t len)
{
    static const char *name[] = {"one token", "2-16 tokens", "blocks",
                                 "17-63 tokens"};
    static const int order[JANAS_TUNE_CLASSES] = {0, 1, 3, 2}; /* by size */
    size_t w = 0;
    buf[0] = 0;
    for (int o = 0; o < JANAS_TUNE_CLASSES && w < len; o++) {
        int k = order[o];
        int b = janas_tuner_best(t, k);
        w += (size_t)snprintf(buf + w, len - w, "%s%s: %d threads%s",
                              o ? "; " : "", name[k], t->cand[b].threads,
                              janas_tune_devices(&t->cand[b]));
        if (t->cls[k].tried[b] >= TRIES && w < len)
            w += (size_t)snprintf(buf + w, len - w, " (%.1f ms/token)",
                                  1e3 * t->cls[k].cost[b]);
    }
}
