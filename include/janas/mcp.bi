'' SPDX-License-Identifier: GPL-3.0-or-later
'' Copyright (C) 2026 Maurizio Cammalleri
'' mcp.bi - Janas-MCP for FreeBASIC, and so for BASIC MODERN, the dialect
'' Prabanta implements.
''
'' The same library the C header include/janas/mcp.h describes: a client of
'' the Model Context Protocol. A server is a program started by the library
'' and spoken to on its standard input and output; its tools are given as
'' the JSON array janas_llm_chat_tools takes (llm.bi), and what a tool
'' answers as the text janas_llm_chat_send_results takes. Handles are
'' opaque; text comes back into the caller's buffer, its full length in
'' *length, JANAS_MCP_ESMALL when it did not fit.
''
'' Build:   fbc yours.bas -l janas_mcp
'' Run:     the library must be found (LD_LIBRARY_PATH, or install it)

#pragma once
#inclib "janas_mcp"

extern "c"

'' JANAS_MCP_ABI_VERSION in C (see llm.bi for why the name is shorter).
const JANAS_MCP_ABI = 1

'' Results: the values of llm.bi's.
const JANAS_MCP_OK     =  0
const JANAS_MCP_EINVAL = -1   '' invalid argument
const JANAS_MCP_ENOMEM = -2   '' out of memory
const JANAS_MCP_EOPEN  = -3   '' the server cannot be started, or no such
                              '' server in the configuration
const JANAS_MCP_ESMALL = -7   '' buffer too small
const JANAS_MCP_EFAIL  = -8   '' the server failed: a protocol error, no
                              '' answer in time, gone, or cancelled

type janas_mcp as any  '' a running MCP server

declare function janas_mcp_abi_version() as long
declare function janas_mcp_last_error() as const zstring ptr

'' The default configuration file (~/.config/janas/mcp.json), and the
'' servers a configuration file lists (path NULL: the default), one a line.
declare function janas_mcp_config_path(byval buf as zstring ptr, _
                                       byval cap as long, _
                                       byval length as long ptr) as long
declare function janas_mcp_config_servers(byval path as const zstring ptr, _
                                          byval buf as zstring ptr, _
                                          byval cap as long, _
                                          byval length as long ptr) as long

'' Starts a server of the configuration (path NULL: the default file), or
'' from a command line of argc words, or reaches one over HTTP at url with
'' n_headers lines "Name: value"; seconds to wait for it (0: 60).
declare function janas_mcp_open(byval path as const zstring ptr, _
                                byval name_ as const zstring ptr, _
                                byval seconds as long, _
                                byval m as janas_mcp ptr ptr) as long
declare function janas_mcp_open_command(byval name_ as const zstring ptr, _
                                        byval argc as long, _
                                        byval argv as const zstring ptr ptr, _
                                        byval n_env as long, _
                                        byval env as const zstring ptr ptr, _
                                        byval seconds as long, _
                                        byval m as janas_mcp ptr ptr) as long
declare function janas_mcp_open_url(byval name_ as const zstring ptr, _
                                    byval url as const zstring ptr, _
                                    byval n_headers as long, _
                                    byval headers as const zstring ptr ptr, _
                                    byval seconds as long, _
                                    byval m as janas_mcp ptr ptr) as long
declare sub janas_mcp_close(byval m as janas_mcp ptr)

'' What the server says of itself, "key: value" a line.
declare function janas_mcp_info(byval m as const janas_mcp ptr, _
                                byval buf as zstring ptr, _
                                byval cap as long, _
                                byval length as long ptr) as long

'' Its instructions for the model, as it gave them (empty: none).
declare function janas_mcp_instructions(byval m as const janas_mcp ptr, _
                                        byval buf as zstring ptr, _
                                        byval cap as long, _
                                        byval length as long ptr) as long

'' Its tools: how many, their names one a line, and as janas_llm_chat_tools
'' takes them, each name after prefix.
declare function janas_mcp_tool_count(byval m as const janas_mcp ptr) as long
declare function janas_mcp_tool_names(byval m as const janas_mcp ptr, _
                                      byval buf as zstring ptr, _
                                      byval cap as long, _
                                      byval length as long ptr) as long
declare function janas_mcp_tools(byval m as const janas_mcp ptr, _
                                 byval prefix as const zstring ptr, _
                                 byval prefix_len as long, _
                                 byval buf as zstring ptr, _
                                 byval cap as long, _
                                 byval length as long ptr) as long
declare function janas_mcp_tool_read_only(byval m as const janas_mcp ptr, _
                                          byval name_ as const zstring ptr, _
                                          byval length as long) as long
declare function janas_mcp_refresh(byval m as janas_mcp ptr) as long

'' Calls a tool (its name without the prefix, the arguments a JSON object);
'' seconds 0: no limit. The answer as text, and whether the tool failed.
declare function janas_mcp_call(byval m as janas_mcp ptr, _
                                byval name_ as const zstring ptr, _
                                byval name_len as long, _
                                byval args as const zstring ptr, _
                                byval args_len as long, _
                                byval seconds as long) as long
declare function janas_mcp_result(byval m as const janas_mcp ptr, _
                                  byval buf as zstring ptr, _
                                  byval cap as long, _
                                  byval length as long ptr) as long
declare function janas_mcp_result_error(byval m as const janas_mcp ptr) as long
declare sub janas_mcp_cancel(byval m as janas_mcp ptr)

end extern
