/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_chat.h - the MCP servers of janas-chat: started from the
 * configuration, their tools given to the model as server__tool, the calls
 * the model writes run and their answers made into text for it. What is
 * shown and asked on the terminal stays in main.c.
 */
#ifndef JANAS_CHAT_MCP_H
#define JANAS_CHAT_MCP_H

#include <stddef.h>

#include "janas/mcp.h"

/* Starts the servers of a configuration file (NULL: the default one, where
   a missing file means none), saying on stderr how each one went. Returns
   how many run. */
int mcpc_start(const char *path);
void mcpc_stop(void);

/* The tools of all the servers as one JSON array for janas_llm_chat_tools
   (NULL: none). */
const char *mcpc_tools(void);

/* What the servers say of how their tools are meant to be used, for the
   system message, marked as theirs (malloc'd; NULL: none said anything). */
char *mcpc_instructions(void);

/* The servers running: their names and handles. */
size_t mcpc_count(void);
const char *mcpc_name(size_t i);
janas_mcp *mcpc_server(size_t i);

/* "Always" for a tool, given by the user for this session. */
int mcpc_always(const char *name);
void mcpc_set_always(const char *name);

/*
 * Runs the call name (server__tool) with args, into *text (malloc'd), the
 * answer for the model: the tool's text, cut when it is very long, or the
 * reason it could not run. Returns 0 when the tool answered, 1 when it said
 * it failed, -1 when it could not be run at all.
 */
int mcpc_call(const char *name, const char *args, char **text);

/* Stops the call running; safe in a signal handler. */
void mcpc_cancel(void);

#endif
