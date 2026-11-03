/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * ollama.h - Ollama's API (https://github.com/ollama/ollama, docs/api.md)
 * beside OpenAI's: what ollama.c and ollama_chat.c share. Private to them.
 */
#ifndef JANAS_SERVER_OLLAMA_H
#define JANAS_SERVER_OLLAMA_H

#include <yyjson.h>

#include "server.h"

/* An error as Ollama gives it: {"error": "..."} with the status. */
int ol_error(struct srv_req *r, unsigned status, const char *fmt, ...)
    __attribute__((format(gnu_printf, 3, 4)));
/* The time now as Ollama writes it (RFC 3339, nanoseconds), into buf. */
void ol_now(char *buf, size_t cap);
/* Whether name is one of this server's models: the chat model's (1), the
   embedding model's (2), else 0. Ollama's ":latest" is the name itself. */
int ol_model(const struct srv *s, const char *name);

/* ollama_req.c: Ollama's request (root) as the JSON text of OpenAI's, for
   /chat/completions (chat: messages, from chat_route's messages or from
   /api/generate's prompt) or /completions; NULL on memory. */
char *ol_request_body(yyjson_val *root, int chat_route, int chat, size_t *n);

/* ollama_chat.c: POST /api/chat and POST /api/generate */
int ol_chat(struct srv_req *r);
int ol_generate(struct srv_req *r);

#endif
