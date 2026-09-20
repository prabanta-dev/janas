/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_internal.h - the objects behind a public handle, for tools linked
 * with the library itself (janas-bench). Not part of the public API.
 */
#ifndef JANAS_LLM_API_INTERNAL_H
#define JANAS_LLM_API_INTERNAL_H

#include "janas/llm.h"
#include "llm/model.h"
#include "llm/tokenizer.h"

struct janas_llm_model *janas_llm_internal_model(janas_llm *llm);
struct janas_tokenizer *janas_llm_internal_tokenizer(janas_llm *llm);

#endif
