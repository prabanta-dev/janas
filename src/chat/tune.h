/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tune.h - janas-chat's measure of a model on this machine: the levers that
 * depend on it (where the drafts come from, whether to draft at all, the
 * output head's bits in the fast profile) tried on three fixed replies each,
 * the fastest kept in a file of its own - a layer between the catalog's profile
 * and the user's settings, never written by /save-config.
 */
#ifndef JANAS_CHAT_TUNE_H
#define JANAS_CHAT_TUNE_H

#include <stddef.h>

#include "janas/llm.h"

/* ~/.cache/janas/tune-<key>-<quality|fast>.conf; 0, or -1 without a home */
int chat_tune_path(char *buf, size_t len, const char *key, int fast);

/* What may be tried: the drafts' source and drafting itself, the head. */
struct chat_tune_dims {
    int spec; /* the user set none of --mtp, --draft, --no-mtp, --no-spec */
    int head; /* the fast profile asks for --head 4 and the user did not */
};

/*
 * Measures the model at path with mp and cp as they would start (mp with
 * the prediction file or draft already found), the levers of dims each
 * way, and writes the winning words to file. gpu 0: the GPU is kept out.
 * 0, or -1 if nothing could be measured.
 */
int chat_tune_run(const char *path, const struct janas_llm_params *mp,
                  const struct janas_llm_chat_params *cp, int gpu,
                  struct chat_tune_dims dims, const char *file);

#endif
