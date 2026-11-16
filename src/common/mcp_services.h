/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_services.h - Janas's services (janas-<name> MCP servers) as a model is
 * offered them, shared by janas-chat and Brebus: their catalog, a line a
 * service in the system message, and one tool (janas__open_service) that
 * opens a service when a question needs it, its instructions and tools
 * given back as that call's answer - not every tool in the system message,
 * which made it thousands of tokens long.
 */
#ifndef JANAS_COMMON_MCP_SERVICES_H
#define JANAS_COMMON_MCP_SERVICES_H

#include <stddef.h>

#include "janas/mcp.h"
#include "llm/json.h"

#define JANAS_OPEN_SERVICE "janas__open_service"

struct janas_service {
    const char *name, *about;
};
extern const struct janas_service JANAS_SERVICES[];
extern const size_t JANAS_N_SERVICES;

/* what a service is about, for the catalog ("" for one not of Janas) */
const char *janas_service_about(const char *name);
/* the catalog for the system message: its rule, then "- name: about" a
   line for each of the n names (about: NULL for one's own line, else the
   n texts given) */
void janas_services_catalog(struct janas_buf *o, const char *const *names,
                            const char *const *abouts, size_t n);
/* the tool that opens one of the n services, as an OpenAI function */
void janas_services_open_tool(struct janas_buf *o, const char *const *names,
                              size_t n);
/* the answer to opening the service m, called name, its tools' names
   beginning with prefix: its instructions and its tools */
void janas_service_opened(struct janas_buf *o, janas_mcp *m, const char *name,
                          const char *prefix);

#endif
