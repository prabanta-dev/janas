/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * anthropic.h - Anthropic's Messages API (POST /v1/messages and
 * /v1/messages/count_tokens) beside OpenAI's, for Claude Code and the other
 * clients made for it: what anthropic.c and anthropic_req.c share. Private
 * to them.
 */
#ifndef JANAS_SERVER_ANTHROPIC_H
#define JANAS_SERVER_ANTHROPIC_H

#include <yyjson.h>

#include "server.h"

/* anthropic_req.c: Anthropic's request (root) as the JSON text of OpenAI's
   /chat/completions; NULL with the reason in why. */
char *an_request_body(yyjson_val *root, size_t *n, char *why, size_t why_len);

/* anthropic.c: the handlers */
int srv_anthropic_messages(struct srv_req *r);
int srv_anthropic_count(struct srv_req *r);

#endif
