/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * serve.h - janas-mcp, Janas as an MCP server: what main.c (the protocol)
 * and tools.c (the tools) share.
 */
#ifndef JANAS_MCPD_SERVE_H
#define JANAS_MCPD_SERVE_H

#include <stdio.h>

#include "common/mcp_rpc.h"
#include "common/mcp_stdio.h"
#include "janas/llm.h"
#include "llm/json.h"

struct mcpd {
    janas_llm *llm; /* the model that writes (NULL: none) */
    janas_llm_chat *chat;
    char name[128];
    struct janas_llm_chat_params cp; /* as started; a call may change some */
    janas_llm *emb; /* the model that gives embeddings (NULL: none) */
    char emb_name[128];
    int call_seconds; /* longest a generation may run (0: no limit) */
    /* the client: its lines in, the answers out */
    struct janas_mcp_proc in;
    FILE *out;
    /* the call running, which a notifications/cancelled may stop, and the
       requests that came meanwhile, answered after it */
    const struct janas_json *running;
    int cancelled;
    char **queue;
    size_t n_queue;
};

/* An answer: members of the result (JSON text without braces) for the
   request of id, with what the revision the request spoke wants. */
void mcpd_result(struct mcpd *d, const struct janas_json *id, int modern,
                 const char *members, size_t n);
void mcpd_error(struct mcpd *d, const struct janas_json *id, int code,
                const char *message, const char *data);

/* While a call runs: what the client sent meanwhile, taken in. 1 when the
   call running was cancelled. */
int mcpd_poll(struct mcpd *d);

/* tools.c: the tools/list members ("tools": [...]); a tools/call into the
   members of its result (0), or -1 for a tool there is not. */
void mcpd_tools_list(const struct mcpd *d, struct janas_buf *b);
int mcpd_tools_call(struct mcpd *d, const struct janas_json *params,
                    struct janas_buf *b);

#endif
