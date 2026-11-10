/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vscode_cfg.h - Janas in the settings of the extensions that keep them in
 * JSON files (vscode_cfg.c): llama.vscode in the editor's settings.json,
 * VS Code's own chat in its chatLanguageModels.json, Cline in its
 * providers.json, Roo Code by a file it imports. Each change is one
 * member, the rest of the file kept byte for byte (src/common/jsonc.c);
 * what it replaced is written down, and --undo puts it back.
 */
#ifndef JANAS_SETUP_VSCODE_CFG_H
#define JANAS_SETUP_VSCODE_CFG_H

#include "editors.h"
#include "plan.h"

/* Writes the settings of extension id (one of setup_known_ext) in e:
   0 done, 1 left to the user (what to fill in printed), -1 error. */
int setup_vscode_write(const struct setup_editor *e, const char *id,
                       const struct setup_plan *p);

/* Puts back what the writes replaced: 0, or -1 when something could not
   be. */
int setup_vscode_undo(void);

#endif
