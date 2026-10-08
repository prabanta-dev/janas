/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * models.h - janas-chat's models by name: a path, a name of the catalog
 * (src/common/catalog.c), or none, and then a list to choose from. A model
 * of the catalog that is not there yet is downloaded and converted by
 * janas-get, with its prediction file or draft, once the user says so.
 */
#ifndef JANAS_CHAT_MODELS_H
#define JANAS_CHAT_MODELS_H

#include <stddef.h>

#include "common/catalog.h"

struct chat_model {
    char path[4400]; /* the .jns to open */
    char dir[4096];  /* the folder it is in */
    char key[128];   /* its settings file: the catalog name, or the file's */
    const struct janas_model_entry *entry; /* NULL: not in the catalog */
};

/* arg: a path, a catalog name, or NULL for the list. 0, or -1 with the
   reason printed. */
int chat_model_resolve(const char *arg, struct chat_model *cm);

/* The draft the catalog names for the model, if its file is beside it:
   the path in buf, or NULL. */
const char *chat_model_draft(const struct chat_model *cm, char *buf,
                             size_t len);

/* The model's prediction file: for a model of the catalog the one the
   catalog names, beside it (none if it names none: a file that merely fits
   may be another model's - Qwen3-Coder-Next took Qwen3-Next-80B's, and
   drafted slower than from the conversation); otherwise the one
   janas_llm_find_mtp finds beside it. The path in buf, or NULL. */
const char *chat_model_mtp(const struct chat_model *cm, char *buf,
                           size_t len);

/* The value of the last opt among n words, or NULL. */
const char *chat_words_last(int n, char *const *w, const char *opt);

/* 1 if one of the n words is one of opts (NULL-terminated). */
int chat_words_any(int n, char *const *w, const char *const *opts);

/* A question answered y or n on standard input (before the chat's own
   terminal is set up); anything but y is no. */
int chat_ask_yes(const char *question);

#endif
