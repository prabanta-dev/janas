/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_config.h - the MCP servers the user configured, in the file format
 * the other clients use, so that a configuration can be copied from one:
 *
 *   {"mcpServers": {
 *       "files": {"command": "npx",
 *                 "args": ["-y", "@modelcontextprotocol/server-filesystem",
 *                          "/home/me/notes"],
 *                 "env": {"KEY": "value"}},
 *       "web": {"url": "https://example.com/mcp",
 *               "headers": {"Authorization": "Bearer ..."}}}}
 *
 * "servers" (VS Code's name) is read as well as "mcpServers"; "disabled":
 * true leaves a server out; "type" may say "stdio", "http" or "sse". The
 * default file is $XDG_CONFIG_HOME/janas/mcp.json, or ~/.config/janas/.
 */
#ifndef JANAS_MCP_CONFIG_H
#define JANAS_MCP_CONFIG_H

#include <stddef.h>

struct janas_mcp_conf {
    char *name;
    char **argv;    /* command and arguments, NULL-terminated; NULL for url */
    char **env;     /* KEY=VALUE, NULL-terminated (may be empty) */
    char *url;      /* a server over HTTP, NULL for stdio */
    char **headers; /* its headers, "Name: value", NULL-terminated */
    int disabled;
};

/* Reads a configuration file. 0 with *out and *n (n may be 0), 1 when the
   file does not exist (nothing configured), -1 with the reason in err. */
int janas_mcp_config_read(const char *path, struct janas_mcp_conf **out,
                          size_t *n, char *err, size_t err_len);
/* The same from text, for the tests. */
int janas_mcp_config_parse(const char *text, size_t len,
                           struct janas_mcp_conf **out, size_t *n, char *err,
                           size_t err_len);
void janas_mcp_config_free(struct janas_mcp_conf *c, size_t n);

/* The default path, into buf; -1 when there is no home to put it in. */
int janas_mcp_conf_path(char *buf, size_t n);

#endif
