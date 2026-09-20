/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * metrics.h - what a reply costs, told as it goes, for any program on the
 * library without a line of its own: JANAS_PROGRESS=s writes lines for a
 * person on standard error, JANAS_METRICS=file appends JSON lines for a
 * program (see include/janas/llm.h). Counters and times only: never the
 * text of a prompt or of a reply. With neither set, each call is one test
 * of a flag.
 */
#ifndef JANAS_LLM_METRICS_H
#define JANAS_LLM_METRICS_H

#include "janas/llm.h"

/* A model opened, in seconds. */
void janas_metrics_open(const janas_llm *llm, double seconds);
/* A reply begins: its prompt, and how it is made up (the chat's stats). */
void janas_metrics_prompt(janas_llm_chat *c);
/* The prompt is being read: a line or an event when enough time has
   passed since the last one, or when force. */
void janas_metrics_progress(janas_llm_chat *c, int force);
void janas_metrics_first_token(janas_llm_chat *c);
/* The reply is over, whatever ended it. */
void janas_metrics_reply(janas_llm_chat *c);
/* A prompt refused for being longer than max_input. */
void janas_metrics_refused(janas_llm_chat *c, uint32_t max_input);

#endif
