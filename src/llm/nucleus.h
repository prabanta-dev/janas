/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * nucleus.h - Nucleus mode (api_nucleus.c;
 * job/docs/progetto_nucleus_cancello.md), internal until its support is
 * complete: janas-server uses it, the public header does not show it.
 * janas_llm_chat_stats.route and route_domain (reserved in llm.h) say who
 * answered a reply.
 */
#ifndef JANAS_LLM_NUCLEUS_H
#define JANAS_LLM_NUCLEUS_H

#include "janas/llm.h"

/* janas_llm_chat_stats.route */
#define JANAS_LLM_ROUTE_SELF 0       /* this model (no gate, or go on) */
#define JANAS_LLM_ROUTE_SPECIALIST 1 /* the Specialist */
#define JANAS_LLM_ROUTE_UNKNOWN 2    /* "I don't know": nothing generated */

/*
 * Nucleus mode (off until called): an entrance gate on this model, read
 * from gate_path (the gate file made for it), decides at every reply,
 * from the model's state at the prompt's last token, whether the model
 * answers, whether the request goes to the specialist (an opened model;
 * NULL: none, and the model answers), or whether it is about something the
 * model does not know, answered then with one sentence in the request's
 * language and nothing generated. The specialist gets the conversation of
 * janas_llm_chat_load (a conversation built by janas_llm_chat_send stays
 * with this model). Close this model before the specialist. gate_path NULL
 * turns it off. Where a reply's first pass runs whole on a GPU the gate
 * sees nothing and the model answers, as without it. Who answered is in
 * janas_llm_chat_stats.route.
 */
int32_t janas_llm_nucleus(janas_llm *llm, const char *gate_path,
                          janas_llm *specialist);

#endif
