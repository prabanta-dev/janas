/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tuner.h - the engine tunes itself on real passes.
 *
 * Every configuration (how many threads, whether the GPU takes part) gives
 * the same results bit for bit, so the engine can try them on the passes it
 * runs anyway: for each kind of pass (one token, a few, a block) it tries
 * each candidate a few times, keeps the fastest, and now and then tries the
 * others again to follow the machine's state (heat, power source). The
 * choices are saved per machine and model, so the next start begins from
 * them.
 */
#ifndef JANAS_LLM_TUNER_H
#define JANAS_LLM_TUNER_H

#include <stddef.h>
#include <stdint.h>

#define JANAS_TUNE_CLASSES 3 /* one token; 2-16 (verification); more */
#define JANAS_TUNE_MAX_CANDS 8

struct janas_tune_cand {
    int threads; /* compute threads (a prefix of the pool) */
    int gpu;     /* the GPU takes part in products over blocks */
};

struct janas_tuner {
    struct janas_tune_cand cand[JANAS_TUNE_MAX_CANDS];
    int n_cand;
    int fixed;     /* no tuning: the default candidate of each class */
    int allow_gpu; /* candidates with the GPU may be chosen */
    int forced;    /* >= 0: every pass uses this candidate (and is measured) */
    struct {
        double cost[JANAS_TUNE_MAX_CANDS]; /* seconds per token, averaged */
        uint32_t tried[JANAS_TUNE_MAX_CANDS];
        uint32_t last[JANAS_TUNE_MAX_CANDS]; /* pass of the last try */
        uint32_t passes;
        int def; /* the default candidate */
        /* the next try of an alternative, the passes until the one after,
           and the best candidate when the last try was made (-1: none
           pending): a try that changes nothing doubles the gap */
        uint32_t retry_at, retry_gap;
        int retry_best, retry_cand; /* and the candidate tried */
    } cls[JANAS_TUNE_CLASSES];
};

int janas_tune_class(uint32_t n_tokens);

/* The candidates and each class's default (index into cand). */
void janas_tuner_init(struct janas_tuner *t, const struct janas_tune_cand *c,
                      int n, const int def[JANAS_TUNE_CLASSES], int fixed);

/* The candidate for the next pass of this class, and its measured cost. */
int janas_tuner_pick(struct janas_tuner *t, int cls);
void janas_tuner_record(struct janas_tuner *t, int cls, int cand,
                        double sec_per_token);

/* Every pass on candidate cand (measured all the same), -1 to stop. */
void janas_tuner_force(struct janas_tuner *t, int cand);

/* Whether candidates using the GPU may be chosen (energy saving: no). */
void janas_tuner_allow_gpu(struct janas_tuner *t, int allow);

/* The best known candidate of a class (for work that is not measured). */
int janas_tuner_best(const struct janas_tuner *t, int cls);

/*
 * ~/.cache/janas (XDG_CACHE_HOME honoured), where the engine keeps what it
 * learns about this machine. With create the directory is made if missing.
 * Returns 0, or -1 if there is no place to put it.
 */
int janas_cache_dir(char *path, size_t len, int create);

/* Saved choices, under a key naming machine, model and candidates. */
void janas_tuner_load(struct janas_tuner *t, const char *key);
void janas_tuner_save(const struct janas_tuner *t, const char *key);

/* A one-line summary per class, for reports. */
void janas_tuner_report(const struct janas_tuner *t, char *buf, size_t len);

#endif
