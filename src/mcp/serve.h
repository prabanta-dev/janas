/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * serve.h - janas-mcp, Janas as an MCP server: what main.c (the protocol)
 * and tools.c (the tools) share.
 */
#ifndef JANAS_MCPD_SERVE_H
#define JANAS_MCPD_SERVE_H

#include <stdio.h>

#include "common/mcp_server.h"
#include "janas/llm.h"
#include "llm/json.h"

struct mcpd {
    janas_llm *llm; /* the model that writes (NULL: none) */
    janas_llm_chat *chat;
    char name[128];
    struct janas_llm_chat_params cp; /* as started; a call may change some */
    janas_llm *emb; /* the model that gives embeddings (NULL: none) */
    char emb_name[128];
    int call_seconds;      /* longest a generation may run (0: no limit) */
    struct janas_mcps srv; /* the protocol (common/mcp_server.c) */
};

/* While a call runs: what the client sent meanwhile, taken in. 1 when the
   call running was cancelled. */
static inline int mcpd_poll(struct mcpd *d)
{
    return janas_mcps_poll(&d->srv);
}

/* tools.c: the tools/list members ("tools": [...]); a tools/call into the
   members of its result (0), or -1 for a tool there is not. */
void mcpd_tools_list(const struct mcpd *d, struct janas_buf *b);
int mcpd_tools_call(struct mcpd *d, const struct janas_json *params,
                    struct janas_buf *b);

#endif
