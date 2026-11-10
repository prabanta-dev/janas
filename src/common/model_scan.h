/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_scan.h - the models installed here, as janas-setup offers them to
 * an editor and janas-chat lists them (model_scan.c): what each can do,
 * read from its own file - a model that fills in the middle has
 * <|fim_prefix|> in its vocabulary, and one with recurrent layers cannot go
 * back to a point of a sequence, so every keystroke would read its window
 * again from the start.
 */
#ifndef JANAS_COMMON_MODEL_SCAN_H
#define JANAS_COMMON_MODEL_SCAN_H

#include <stddef.h>
#include <stdint.h>

#define JANAS_MODELS_MAX 64

struct janas_model_file {
    char path[4400];
    char name[128]; /* the file's name without .jns: what clients ask for */
    char arch[32];
    uint64_t bytes;
    uint32_t ctx;  /* the context it was trained for, 0: not said */
    int fim;       /* fills in the middle */
    int recurrent; /* has recurrent layers */
    int coder;     /* made for code (its name says so) */
};

/* The chat models of dir (the .jns files, not the prediction files,
   vision encoders, embedding, speech or voice models), sorted by name:
   their number, at most max. */
int janas_models_scan(const char *dir, struct janas_model_file *out, int max);

/* 1 when it can complete code at the cursor: made for code, it fills in
   the middle and has no recurrent layers. */
int janas_model_completes(const struct janas_model_file *m);

#endif
