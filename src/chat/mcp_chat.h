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
/* Starts Janas's own services found beside janas-chat (janas-flights...)
   but those the configuration already names. Returns how many started. */
int mcpc_start_services(void);
/* Whether a tool (server__tool), or the server i, is of Janas's own
   services: they run without asking. */
int mcpc_own(const char *tool);
int mcpc_is_own(size_t i);
void mcpc_stop(void);

/* Janas's services as a catalog in the system message, opened by the model
   when a question needs one (the default), or, off, all their tools
   written into it as before. Before mcpc_start_services. */
void mcpc_defer(int on);
/* Changes each time a service is opened, or all are closed: then the tools
   of those open (mcpc_opened_tools, malloc'd JSON array, NULL: none) go
   to janas_llm_chat_tools_add. mcpc_close_services: a new conversation. */
unsigned mcpc_opened(void);
char *mcpc_opened_tools(void);
void mcpc_close_services(void);

/* The tools of all the servers as one JSON array for janas_llm_chat_tools
   (NULL: none). */
const char *mcpc_tools(void);

/* What the servers say of how their tools are meant to be used, for the
   system message: Janas's own services always, the others' (marked as
   theirs) when others is set. malloc'd; NULL: none said anything. */
char *mcpc_instructions(int others);

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
 * reason it could not run; and into *shown (malloc'd, NULL: none) what the
 * answer gives to the user alone, to be shown as it is (a page). Returns 0
 * when the tool answered, 1 when it said it failed, -1 when it could not
 * be run at all.
 */
int mcpc_call(const char *name, const char *args, char **text, char **shown);

/* Stops the call running; safe in a signal handler. */
void mcpc_cancel(void);

#endif
