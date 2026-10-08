/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * layout.h - the layouts of the services' answers in janas-chat
 * (services/common/template.h): an answer that brings its data and an English
 * layout is shown to the user filled in the user's language, and the model
 * reads only its brief. The layout in another language is asked of the
 * model once, checked against the English one and kept in
 * ~/.config/janas/layouts; the scales' terms come from the glossary
 * (glossary.h), not from the model.
 */
#ifndef JANAS_CHAT_LAYOUT_H
#define JANAS_CHAT_LAYOUT_H

#include <stddef.h>

/* The user's language ("it"; "en" or NULL: English, no translation). */
void layout_set_lang(const char *lang);
/* The model the layouts are translated by: each model keeps its own, and
   its failures, so that a small model's poor translation is never shown
   with another model's answers and a failed one is not asked for again. */
void layout_set_model(const char *key);
const char *layout_lang(void);

/* How a layout is translated: system and user messages to the model, its
   answer into *out (malloc'd). 0, or -1. Called once a layout and
   language; what it says on the terminal is its own. */
void layout_set_translator(int (*fn)(const char *name, const char *system,
                                     const char *text, char **out));

/* A tools/call result (JSON) with a layout: 1 with the text for the user
   in *user and the brief for the model in *model (both malloc'd); 0 when
   it has none, or it could not be filled (then the answer is used as it
   came). */
int layout_take(const char *result, size_t n, char **user, char **model);

#endif
