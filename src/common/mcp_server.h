/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_server.h - the server side of the Model Context Protocol over stdio,
 * for the programs that offer tools (janas-mcp, janas-flights): the client
 * starts the program, writes requests on its standard input and reads the
 * answers on its standard output, one JSON-RPC message per line.
 *
 * Both eras of the protocol are served, as the specification allows a
 * server to: a request that carries its revision in _meta (2026-07-28) is
 * answered as it asks, statelessly, and a client that opens with
 * initialize gets the handshake of the revision it names (2025-11-25 and
 * the three before it). A request that comes while a call runs waits for
 * it, but a ping is answered at once and a notifications/cancelled stops
 * the call (the tool asks, janas_mcps_poll).
 *
 * The program gives its name, its instructions and two functions: the
 * members of tools/list, and a tools/call.
 */
#ifndef JANAS_MCP_SERVER_H
#define JANAS_MCP_SERVER_H

#include <stddef.h>
#include <stdio.h>

#include "common/mcp_stdio.h"
#include "llm/json.h"

struct janas_mcps {
    /* given by the program */
    const char *name;         /* serverInfo name, "janas-mcp" */
    const char *version;      /* serverInfo version */
    const char *instructions; /* for the client's model */
    void *ctx;                /* handed to the two functions */
    /* the members of the tools/list result ("tools": [...]) */
    void (*tools_list)(void *ctx, struct janas_buf *b);
    /* a tools/call: the members of its result into b (0), -1 for a tool
       there is not, 1 when it was cancelled (nothing is answered) */
    int (*tools_call)(void *ctx, const struct janas_json *params,
                      struct janas_buf *b);

    /* the server's own */
    struct janas_mcp_proc in;
    FILE *out;
    const struct janas_json *running; /* the id of the call running */
    int cancelled;
    char **queue; /* requests that came during a call */
    size_t n_queue;
    int initialized; /* a legacy client has shaken hands */
    char legacy_version[32];
};

/*
 * Sets standard output aside for the protocol (whatever else would be
 * written there goes to standard error from now on). 0, or -1.
 */
int janas_mcps_open(struct janas_mcps *s);

/* Serves until the client closes its end; then frees what it holds. */
void janas_mcps_run(struct janas_mcps *s);

/* While a call runs: what the client sent meanwhile, taken in. 1 when the
   call running was cancelled. */
int janas_mcps_poll(struct janas_mcps *s);

/* A result of one text item, for tools/call: the members into b. */
void janas_mcps_text_result(struct janas_buf *b, const char *t, size_t n,
                            int is_error);

/* A result of two text items: one for the model alone, one for the user
   alone (their annotations' audience), for a client that shows the user's
   as it is and gives the model only the first; a client that heeds no
   audience gives the model both. */
void janas_mcps_split_result(struct janas_buf *b, const char *model,
                             size_t model_n, const char *user, size_t user_n);

#endif
