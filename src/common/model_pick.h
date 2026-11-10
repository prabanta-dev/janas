/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_pick.h - a model by name, for the programs that open one
 * (janas-chat, janas-bench): a path, a name of the catalog
 * (src/common/catalog.c), or none, and then a list to choose from. A model
 * of the catalog that is not there yet is downloaded and converted by
 * janas-get, with its prediction file or draft, once the user says so.
 */
#ifndef JANAS_COMMON_MODEL_PICK_H
#define JANAS_COMMON_MODEL_PICK_H

#include <stddef.h>

#include "common/catalog.h"

struct janas_model_pick {
    char path[4400]; /* the .jns to open */
    char dir[4096];  /* the folder it is in */
    char key[128];   /* its settings file: the catalog name, or the file's */
    const struct janas_model_entry *entry; /* NULL: not in the catalog */
};

/* A model for a menu: one of the catalog, or a file of no model of the
   catalog. */
struct janas_model_item {
    const struct janas_model_entry *entry; /* NULL: a file of none */
    char name[128];
    char path[4400]; /* the file, when it is here; empty: to download */
    double gb;       /* its size here, or the download */
    int ram_gb;      /* the memory it is meant for, 0: not known */
};

/* The chat models for a menu, malloc'd, *n of them: first the *n_here
   that are here (the catalog's, then the other chat models in dir, by
   name), then the catalog's that are not. NULL when there are none. */
struct janas_model_item *janas_model_items(const char *dir, int *n,
                                           int *n_here);

/* A menu of the program's own in place of the numbered list, when both
   ends are a terminal (janas-chat's, src/chat/model_menu.c): the index of
   the item chosen, or -1 when the user left. NULL: the numbered list. */
extern int (*janas_model_menu)(const char *dir,
                               const struct janas_model_item *items, int n,
                               int n_here);

/* arg: a path, a catalog name, or NULL for the list; prog names the
   program in what is printed. 0, -1 with the reason printed, or -2 when
   the user left the list without choosing. */
int janas_model_pick(const char *prog, const char *arg,
                     struct janas_model_pick *mp);

/* The draft the catalog names for the model, if its file is beside it:
   the path in buf, or NULL. */
const char *janas_model_pick_draft(const struct janas_model_pick *mp, char *buf,
                                   size_t len);

/* The model's prediction file: for a model of the catalog the one the
   catalog names, beside it (none if it names none: a file that merely fits
   may be another model's - Qwen3-Coder-Next took Qwen3-Next-80B's, and
   drafted slower than from the conversation); otherwise the one
   janas_llm_find_mtp finds beside it. The path in buf, or NULL. */
const char *janas_model_pick_mtp(const struct janas_model_pick *mp, char *buf,
                                 size_t len);

/* A question answered y or n on standard input (before a terminal of the
   program's own is set up); anything but y is no. */
int janas_ask_yes(const char *question);

#endif
