/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * catalog.h - the models janas-get knows: where each comes from, and the
 * SHA-256 of every file on the way, as MODELS.md lists them.
 */
#ifndef JANAS_GET_CATALOG_H
#define JANAS_GET_CATALOG_H

#define GET_MAX_PARTS 4

/* how the model's multi-token prediction file is made, if it has one */
enum get_extra {
    EXTRA_NONE,
    EXTRA_GGUF,  /* a GGUF of its own (Gemma 4's assistant), gguf2jns */
    EXTRA_HF2JNS /* the block of the original checkpoint, hf2jns_mtp hf: */
};

struct get_model {
    const char *name;                 /* what janas-get takes */
    const char *what;                 /* a line for the list */
    const char *repo;                 /* the GGUF's repository */
    const char *parts[GET_MAX_PARTS]; /* its file(s), the first given to
                                         gguf2jns */
    const char *part_sha[GET_MAX_PARTS];
    double gguf_gb;       /* download */
    const char *out;      /* the .jns, beside the downloads */
    const char *flat_sha; /* what gguf2jns writes (the .jns itself when
                             there are no planes) */
    const char *jns_sha;  /* what jns_planes writes, NULL: one step */
    enum get_extra extra;
    const char *extra_src;     /* GGUF file of repo, or hf:repo[@rev] */
    const char *extra_src_sha; /* of that GGUF (EXTRA_GGUF) */
    const char *extra_out;
    const char *extra_sha;
    int ram_gb; /* the smallest machine it is meant for */
};

extern const struct get_model get_catalog[];
extern const int get_catalog_n;

const struct get_model *get_find(const char *name);

#endif
