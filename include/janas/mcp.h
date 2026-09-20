/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas/mcp.h - the public C API of Janas-MCP: a client of the Model
 * Context Protocol, to give a model the tools of MCP servers.
 *
 * A server is a program started here and spoken to on its standard input
 * and output (the stdio transport), or an address it listens at (the
 * Streamable HTTP transport). It is opened from a configuration file
 * in the format the other clients use ({"mcpServers": {"name": {"command",
 * "args", "env"}}}, by default ~/.config/janas/mcp.json) or from a command
 * line. Both protocol eras are spoken: the stateless revision 2026-07-28,
 * and for the servers that predate it the handshake of 2025-11-25 and
 * earlier; which one a server speaks is found out when it is opened.
 *
 * With janas/llm.h: janas_mcp_tools gives the server's tools as the JSON
 * array janas_llm_chat_tools takes, their names prefixed so that the tools
 * of several servers can be told apart; the calls the model writes
 * (janas_llm_chat_call) go to janas_mcp_call without the prefix, and what
 * janas_mcp_result gives back goes to janas_llm_chat_send_results.
 *
 * A tool runs with the rights of the program that calls it, and what it
 * answers is text written by somebody else, which the model will read: the
 * specification asks for a person in the loop, who sees each call before it
 * is made and may refuse it. That is the caller's to do.
 *
 * The library is libjanas_mcp, and needs nothing of libjanas_llm: a
 * program may use the tools of a server without a model.
 *
 * Conventions as in janas/llm.h: opaque handles, int32_t codes
 * (JANAS_MCP_OK or a negative JANAS_MCP_E...), UTF-8 text as a pointer and a
 * byte length (-1: NUL-terminated), texts returned into the caller's buffer
 * with their full length in *len. A server handle is used from one thread at a
 * time; janas_mcp_cancel may be called from any thread or a signal handler.
 */
#ifndef JANAS_MCP_H
#define JANAS_MCP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define JANAS_MCP_API __declspec(dllexport)
#elif defined(__GNUC__)
#define JANAS_MCP_API __attribute__((visibility("default")))
#else
#define JANAS_MCP_API
#endif

#define JANAS_MCP_ABI_VERSION 1

/* The codes have the values of janas/llm.h's, so a program using both can
   treat them alike. */
#define JANAS_MCP_OK 0
#define JANAS_MCP_EINVAL (-1) /* invalid argument */
#define JANAS_MCP_ENOMEM (-2) /* out of memory */
/* the server cannot be started, or there is no such server */
#define JANAS_MCP_EOPEN (-3)
#define JANAS_MCP_ESMALL (-7) /* buffer too small */
/* the server failed: a protocol error, no answer in time, gone, cancelled */
#define JANAS_MCP_EFAIL (-8)

typedef struct janas_mcp janas_mcp; /* a running MCP server */

/* JANAS_MCP_ABI_VERSION of the library: check it against the header's. */
JANAS_MCP_API int32_t janas_mcp_abi_version(void);

/* The last failure on this thread, as text (never NULL). */
JANAS_MCP_API const char *janas_mcp_last_error(void);

/* The default configuration file ($XDG_CONFIG_HOME/janas/mcp.json, or
   ~/.config/janas/mcp.json), whether or not it exists. */
JANAS_MCP_API int32_t janas_mcp_config_path(char *buf, int32_t cap,
                                            int32_t *len);

/* The names of the servers a configuration file lists (NULL: the default
   file), one per line; disabled ones and those over HTTP, which this
   are marked with " (disabled)" and " (http)".
   An empty list when the file does not exist. */
JANAS_MCP_API int32_t janas_mcp_config_servers(const char *path, char *buf,
                                               int32_t cap, int32_t *len);

/*
 * Starts the server called name in a configuration file (path NULL: the
 * default one) and learns its tools. What it writes to its standard error
 * goes to ~/.cache/janas/mcp/<name>.log. Seconds: how long to wait for it
 * to answer at all - a server fetched by npx or uvx the first time takes a
 * while (0: 60).
 */
JANAS_MCP_API int32_t janas_mcp_open(const char *path, const char *name,
                                     int32_t seconds, janas_mcp **out);
/* The same for a command line of argc words (argv[0] looked up in PATH),
   with n_env more variables as NAME=value (env may be NULL); name is the
   log's and may be NULL. */
JANAS_MCP_API int32_t janas_mcp_open_command(const char *name, int32_t argc,
                                             const char *const *argv,
                                             int32_t n_env,
                                             const char *const *env,
                                             int32_t seconds, janas_mcp **out);
/*
 * A server over HTTP (the Streamable HTTP transport): its address, and
 * n_headers header lines as "Name: value" to send with every request (an
 * Authorization, say; headers may be NULL). An https:// address needs a
 * build of Janas with TLS. A configuration file names such a server with
 * "url" and "headers" in place of "command", and janas_mcp_open takes it.
 */
JANAS_MCP_API int32_t janas_mcp_open_url(const char *name, const char *url,
                                         int32_t n_headers,
                                         const char *const *headers,
                                         int32_t seconds, janas_mcp **out);

/* Ends the server: its input closed, then signals if it lingers. */
JANAS_MCP_API void janas_mcp_close(janas_mcp *m);

/*
 * What the server says of itself, one "key: value" per line: its name and
 * version, the protocol revision spoken, and its instructions for the
 * model when it gives any (a text the caller may add to the system
 * message).
 */
JANAS_MCP_API int32_t janas_mcp_info(const janas_mcp *m, char *buf, int32_t cap,
                                     int32_t *len);

/* The server's instructions for the model, as it gave them (empty when
   it gave none). Text written by the server's authors: a caller that
   hands it to a model should say where it comes from. */
JANAS_MCP_API int32_t janas_mcp_instructions(const janas_mcp *m, char *buf,
                                             int32_t cap, int32_t *len);

/* How many tools the server offers, and their names, one per line. */
JANAS_MCP_API int32_t janas_mcp_tool_count(const janas_mcp *m);
JANAS_MCP_API int32_t janas_mcp_tool_names(const janas_mcp *m, char *buf,
                                           int32_t cap, int32_t *len);

/*
 * The tools as janas_llm_chat_tools takes them: a JSON array of
 * {"type": "function", "function": {"name", "description", "parameters"}},
 * each name with prefix (of prefix_len bytes; NULL for none) before it.
 */
JANAS_MCP_API int32_t janas_mcp_tools(const janas_mcp *m, const char *prefix,
                                      int32_t prefix_len, char *buf,
                                      int32_t cap, int32_t *len);

/* Whether a tool says it only reads (the readOnlyHint of its annotations,
   which the server gives and nobody checks): 1, 0, or -1 when it says
   nothing or there is no such tool. */
JANAS_MCP_API int32_t janas_mcp_tool_read_only(const janas_mcp *m,
                                               const char *name, int32_t len);

/* Asks the server for its tools again (a server may change them). */
JANAS_MCP_API int32_t janas_mcp_refresh(janas_mcp *m);

/*
 * Calls a tool: its name (without the prefix) and its arguments, a JSON
 * object (NULL or empty: {}). Waits for the answer at most seconds (0: no
 * limit). JANAS_MCP_OK when the server answered - the answer may still say
 * the tool failed, see janas_mcp_result_error - or an error when it did not
 * (unknown tool, a protocol error, the server gone and not restarted, the
 * time up, janas_mcp_cancel). A server that has died is started again once.
 */
JANAS_MCP_API int32_t janas_mcp_call(janas_mcp *m, const char *name,
                                     int32_t name_len, const char *args,
                                     int32_t args_len, int32_t seconds);

/*
 * The answer of the last call as text for the model: text as it came,
 * embedded text resources with their address, and what is not text
 * (images, audio, binary resources) described in a line. 1 in
 * janas_mcp_result_error when the tool said it failed: the text then says
 * why, which a model can often act on.
 */
JANAS_MCP_API int32_t janas_mcp_result(const janas_mcp *m, char *buf,
                                       int32_t cap, int32_t *len);
JANAS_MCP_API int32_t janas_mcp_result_error(const janas_mcp *m);

/* Stops the call in progress on m, which then returns an error; the server
   is told (notifications/cancelled). */
JANAS_MCP_API void janas_mcp_cancel(janas_mcp *m);

#ifdef __cplusplus
}
#endif

#endif
