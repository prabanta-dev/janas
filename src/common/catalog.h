/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * catalog.h - the models Janas knows: where each comes from, the SHA-256 of
 * every file on the way, as MODELS.md lists them, and how janas-chat runs
 * it fastest. janas-get downloads and converts them; janas-chat lists them,
 * takes one by name and starts it with its profile.
 */
#ifndef JANAS_COMMON_CATALOG_H
#define JANAS_COMMON_CATALOG_H

#include <stddef.h>

#define JANAS_CATALOG_PARTS 4

/* how the model's multi-token prediction file is made, if it has one */
enum janas_model_extra {
    JANAS_EXTRA_NONE,
    JANAS_EXTRA_GGUF,  /* a GGUF of its own (Gemma 4's assistant), gguf2jns */
    JANAS_EXTRA_HF2JNS /* the block of the original checkpoint, hf2jns_mtp hf:
                        */
};

struct janas_model_entry {
    const char *name;                       /* what janas-get takes */
    const char *what;                       /* a line for the list */
    const char *repo;                       /* the GGUF's repository */
    const char *parts[JANAS_CATALOG_PARTS]; /* its file(s), the first given to
                                         gguf2jns */
    const char *part_sha[JANAS_CATALOG_PARTS];
    double gguf_gb;       /* download */
    const char *out;      /* the .jns, beside the downloads */
    const char *flat_sha; /* what gguf2jns writes (the .jns itself when
                             there are no planes) */
    const char *jns_sha;  /* what jns_planes writes, NULL: one step */
    enum janas_model_extra extra;
    const char *extra_src;     /* GGUF file of repo, or hf:repo[@rev] */
    const char *extra_src_sha; /* of that GGUF (JANAS_EXTRA_GGUF) */
    const char *extra_out;
    const char *extra_sha;
    int ram_gb;        /* the smallest machine it is meant for */
    int not_chat;      /* 1: not a chat model (embeddings) */
    int test_only;     /* 1: for testing the engine, not for chatting: left
                          out of janas-chat's list, opened by name with a
                          word of warning */
    const char *draft; /* the catalog model that guesses its tokens, when
                          that beats its own prediction file */
    /* janas-chat's options for the "fast" profile, on top of the default
       ("quality") one: the levers measured to make this model faster at a
       small, measured cost to its answers (README, ChangeLog) */
    const char *fast;
};

extern const struct janas_model_entry janas_catalog[];
extern const int janas_catalog_n;

const struct janas_model_entry *janas_catalog_find(const char *name);

/*
 * Where the models are: JANAS_MODELS if set, otherwise
 * $XDG_DATA_HOME/janas/models, otherwise ~/.local/share/janas/models.
 * Returns 0, or -1 if there is no home to put it in.
 */
int janas_models_dir(char *buf, size_t len);

/* Makes dir and its parents (mode 0755). 0 if it is there afterwards. */
int janas_mkdirs(const char *dir);

#endif
