/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_oauth.h - signing in to an MCP server over HTTP that asks for it,
 * by the authorization of MCP: OAuth 2.1 with PKCE, the server's
 * authorization server found from its protected resource's metadata
 * (RFC 9728, then RFC 8414), the client registered on it (RFC 7591).
 *
 * A program that has a browser to show (Brebus) starts the sign-in with
 * janas_oauth_begin, sends the user to the address it gives, and ends it
 * with janas_oauth_finish when the authorization server sends the browser
 * back to redirect_uri. The tokens are kept, one file a server, in
 * $XDG_DATA_HOME/janas/mcp-auth (~/.local/share/janas/mcp-auth;
 * $JANAS_MCP_AUTH_DIR when set), readable by the user alone; every program
 * of the user opening that server sends them (janas_mcp_open), renewed
 * with the refresh token when they expire.
 */
#ifndef JANAS_MCP_OAUTH_H
#define JANAS_MCP_OAUTH_H

#include <stddef.h>

/* "Authorization: Bearer <token>" for the server named name at url, its
   token renewed first when it has expired: 0, or -1 with none kept (or it
   could not be renewed). */
int janas_oauth_header(const char *name, const char *url, char *out,
                       size_t cap);

/* 1 when a token is kept for name, else 0. */
int janas_oauth_signed_in(const char *name);

/* The address of the authorization server's page where the user signs in
   to the server named name at url; it will send the browser back to
   redirect_uri with a code. 0, or -1 with the reason in err. */
int janas_oauth_begin(const char *name, const char *url,
                      const char *redirect_uri, char *auth_url, size_t cap,
                      char *err, size_t err_len);

/* The code and state the browser came back with, traded for the tokens,
   kept: 0 with the server's name in name, or -1 with the reason in err. */
int janas_oauth_finish(const char *state, const char *code, char *name,
                       size_t name_cap, char *err, size_t err_len);

/* The tokens of name forgotten (signed out): 0, or -1 when there were none. */
int janas_oauth_forget(const char *name);

#endif
