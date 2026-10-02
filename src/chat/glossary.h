/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * glossary.h - the terms of art of the layouts (layout.h) in the languages
 * Janas knows: the degrees of the Beaufort and Douglas scales as the
 * national weather services name them, which a model asked to translate
 * gets wrong ("liscio" for the Douglas sea "quasi calmo"). Everything else
 * is translated by the model.
 */
#ifndef JANAS_CHAT_GLOSSARY_H
#define JANAS_CHAT_GLOSSARY_H

#include <stddef.h>
#include <stdint.h>

/* The term of the word key (n bytes: "bft.4", "sea.3", "status.open")
   in lang; NULL when the glossary has none. */
const char *glossary_word(const char *lang, const char *key, size_t n);

/* The terms of the running text in lang for the layout name, for the
   request to the model ("swell = mare lungo; ..."; those of git's craft
   for git_* and github_*); NULL when there are none. */
const char *glossary_terms(const char *lang, const char *name);

/* h carried on (FNV-1a) over the glossary of lang for the layout name: a
   layout kept on disk is translated again when the glossary changes. */
uint64_t glossary_mark(const char *lang, const char *name, uint64_t h);

#endif
