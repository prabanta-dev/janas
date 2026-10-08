/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * env_names.c - the variables of the environment Janas reads (see
 * env_names.h).
 */
#include "common/env_names.h"

#include <stdio.h>
#include <string.h>

#include "common/words.h"

extern char **environ;

const char *const JANAS_ENV_NAMES[] = {"JANAS_ALPHAVANTAGE_KEY",
                                       "JANAS_API_KEY",
                                       "JANAS_ARENA_HUGE",
                                       "JANAS_ATTN",
                                       "JANAS_ATTN_FUSE",
                                       "JANAS_ATTN_SKIPSTAT",
                                       "JANAS_AVIATIONSTACK_KEY",
                                       "JANAS_BLOCK_THREADS",
                                       "JANAS_DRAFT_VOCAB",
                                       "JANAS_EXPERT_BITS",
                                       "JANAS_EXPERTS",
                                       "JANAS_MISS_SKIP",
                                       "JANAS_EXPTRACE",
                                       "JANAS_FLIGHTS_CONFIG",
                                       "JANAS_GPU",
                                       "JANAS_GPU_ALLOC",
                                       "JANAS_GPU_ALWAYS",
                                       "JANAS_GPU_ATT_SG",
                                       "JANAS_GPU_BLOCK",
                                       "JANAS_GPU_BLOCKWAIT",
                                       "JANAS_GPU_COPY",
                                       "JANAS_GPU_DEVICE",
                                       "JANAS_GPU_EXPERTS",
                                       "JANAS_GPU_FULL_N",
                                       "JANAS_GPU_KEEPALIVE_US",
                                       "JANAS_GPU_MIN_N",
                                       "JANAS_GPU_NO_ROW",
                                       "JANAS_GPU_NOSPIN",
                                       "JANAS_GPU_REPACK",
                                       "JANAS_GPU_ROW_NV",
                                       "JANAS_GPU_SG",
                                       "JANAS_GPU_TOKEN",
                                       "JANAS_GPU_TOKEN_PROFILE",
                                       "JANAS_GPU_VERBOSE",
                                       "JANAS_GPU_XCOH",
                                       "JANAS_HEAD_BITS",
                                       "JANAS_KERNELS",
                                       "JANAS_KV",
                                       "JANAS_LOCATION",
                                       "JANAS_METRICS",
                                       "JANAS_METRICS_EVERY",
                                       "JANAS_MODELS",
                                       "JANAS_NOMINATIM_URL",
                                       "JANAS_OSRM_URL",
                                       "JANAS_OVERPASS_URL",
                                       "JANAS_OVERPASS_URL2",
                                       "JANAS_PHOTON_URL",
                                       "JANAS_POOLTRACE",
                                       "JANAS_PREFETCH",
                                       "JANAS_PREFETCH_PREFILL",
                                       "JANAS_PREFILL_BLOCK",
                                       "JANAS_PROGRESS",
                                       "JANAS_SHA256_C",
                                       "JANAS_SWA_RING",
                                       "JANAS_TRACE",
                                       "JANAS_TRANSITOUS_URL",
                                       "JANAS_TUNE",
                                       "JANAS_TUNE_THREADS",
                                       "JANAS_URING",
                                       "JANAS_URING_BATCH",
                                       "JANAS_URING_DEEP",
                                       "JANAS_VALHALLA_URL",
                                       "JANAS_VMAP",
                                       "JANAS_WARM"};
const size_t JANAS_N_ENV_NAMES =
    sizeof JANAS_ENV_NAMES / sizeof *JANAS_ENV_NAMES;

int janas_env_check(const char *prog)
{
    int n = 0;
    for (char **e = environ; e && *e; e++) {
        if (strncmp(*e, "JANAS_", 6) != 0)
            continue;
        char name[96];
        size_t k = strcspn(*e, "=");
        if (k >= sizeof name)
            k = sizeof name - 1;
        memcpy(name, *e, k);
        name[k] = 0;
        int known = 0;
        for (size_t i = 0; i < JANAS_N_ENV_NAMES && !known; i++)
            known = strcmp(name, JANAS_ENV_NAMES[i]) == 0;
        if (known)
            continue;
        char why[240];
        janas_unknown_word(why, sizeof why, name, "a variable of Janas",
                           JANAS_ENV_NAMES, JANAS_N_ENV_NAMES);
        fprintf(stderr, "%s: %s: it does nothing\n", prog, why);
        n++;
    }
    return n;
}
