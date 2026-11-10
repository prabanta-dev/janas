/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * plan.h - which installed model does what (plan.c): one for code (chat,
 * edits, the agent) and one for completion at the cursor, the same one
 * when memory holds only one or when it is the only one that can, two when
 * memory holds both; said to the user with the reason, and saved where
 * janas-setup serve finds it.
 */
#ifndef JANAS_SETUP_PLAN_H
#define JANAS_SETUP_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "common/model_scan.h"

#define SETUP_PORT 11435 /* the first port; Ollama's 11434 is left to it */
#define SETUP_CTX 32768  /* an agent's prompt and its work */

struct setup_plan {
    const struct janas_model_file *code;     /* NULL: none installed */
    const struct janas_model_file *complete; /* NULL: none can */
    int code_port, complete_port;            /* the same when the same model */
    unsigned ctx;
};

/* What the user named instead of a proposal (janas-setup editor --code,
   --complete); NULL fields are proposed. */
struct setup_choice {
    const struct janas_model_file *code, *complete;
    int no_complete; /* --complete none */
};

/* What the machine's memory and the models installed suggest, around what
   the user named (c, or NULL); the reasons printed. */
void setup_plan_propose(struct setup_plan *p, const struct janas_model_file *m,
                        int n, uint64_t mem_total,
                        const struct setup_choice *c);

/* janas-setup's folder: $XDG_CONFIG_HOME/janas, or ~/.config/janas
   (Windows: %APPDATA%/janas). 0 or -1. */
int setup_conf_dir(char *buf, size_t len);

/* Where the plan is kept: editor.conf in that folder. 0 or -1. */
int setup_plan_file(char *buf, size_t len);

/* The plan saved (paths and ports), or read back: 0 or -1. Read, the
   paths are in code_path and complete_path ("" none). */
int setup_plan_save(const struct setup_plan *p);
struct setup_saved {
    char code_path[4400], complete_path[4400];
    char code_name[128], complete_name[128];
    int code_port, complete_port;
    unsigned ctx;
};
int setup_plan_load(struct setup_saved *s);

#endif
