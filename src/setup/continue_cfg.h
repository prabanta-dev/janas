/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * continue_cfg.h - Janas's models in Continue (continue_cfg.c). Continue
 * reads, beside the user's config.yaml, every file of its models/ folder
 * as a block of models of its own: janas-setup writes its own there,
 * models/janas.yaml, and never touches the user's file; taking Janas away
 * is removing it. Continue's folder is ~/.continue, or CONTINUE_GLOBAL_DIR.
 */
#ifndef JANAS_SETUP_CONTINUE_CFG_H
#define JANAS_SETUP_CONTINUE_CFG_H

#include <stddef.h>

/* A model as a server serves it: the name it answers to and its port. */
struct setup_served {
    const char *name;
    int port;
    unsigned ctx; /* the context the server gives it */
};

/* Continue's folder in buf: 0, or -1 when there is no home. */
int setup_continue_dir(char *buf, size_t len);

/*
 * Writes models/janas.yaml: code (chat, edits, the agent) and complete
 * (completion at the cursor; NULL: none, or the same model when it is
 * code's). 0, or -1 with the reason printed. *note is set to a word for
 * the user, or NULL: Continue reads only its old config.json when there is
 * no config.yaml, and then not this file.
 */
int setup_continue_write(const struct setup_served *code,
                         const struct setup_served *complete,
                         const char **note);

/* Removes models/janas.yaml: 1 removed, 0 there was none, -1 error. */
int setup_continue_remove(void);

#endif
