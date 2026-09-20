/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_client.h - the object behind a janas_mcp handle (include/janas/mcp.h),
 * shared by the protocol (mcp_client.c) and the reading of what the server
 * gives: its tools and the answers of a call (mcp_content.c).
 */
#ifndef JANAS_MCP_CLIENT_H
#define JANAS_MCP_CLIENT_H

#include <signal.h>
#include <stddef.h>
#include <stdint.h>

#include "common/mcp_http.h"
#include "common/mcp_rpc.h"
#include "common/mcp_stdio.h"
#include "janas/mcp.h"
#include "llm/json.h"

/* The revisions spoken: the stateless one, and the last with a handshake,
   which is what a server that predates the first is asked for. */
#define JANAS_MCP_MODERN "2026-07-28"
#define JANAS_MCP_LEGACY "2025-11-25"

struct janas_mcp_tool {
    char *name;
    char *description; /* may be empty */
    char *schema;      /* the input schema, as JSON text */
    /* over HTTP, the arguments mirrored into headers (x-mcp-header): a
       line for each, the header's name then the property path, each part
       after a \x1f; NULL for none */
    char *params;
    int read_only; /* annotations.readOnlyHint: 1, 0, or -1 unsaid */
};

struct janas_mcp {
    char *name;         /* for the log */
    char **argv, **env; /* how to start it again */
    int wait_ms;        /* how long to wait for it to answer at start */
    char log[1024];
    struct janas_mcp_proc p;
    /* or over HTTP: where, the headers configured ("Name: value"), the
       session of a legacy server, and the headers of the call being made */
    int http;
    struct janas_url url;
    char **headers;
    char session[256];
    const char *call_name;
    struct janas_buf call_headers; /* lines, each ending in \n */
    int modern;
    char version[32]; /* the revision spoken */
    char *server_name, *server_version, *instructions;
    int64_t next_id;
    struct janas_mcp_tool *tools;
    size_t n_tools;
    struct janas_buf result; /* the last call's answer, as text */
    int result_error;
    volatile sig_atomic_t cancel;
    int http_status; /* the last HTTP answer's */
    char http_why[320];
};

enum {
    ANSWERED = 0,
    TIMEOUT = -1,
    GONE = -2,
    CANCELLED = -3,
    NOMEM = -4,
    REFUSED = -5 /* an HTTP error with no JSON-RPC answer in it */
};

/* mcp_client_http.c: a request (id >= 0) or notification over HTTP, its
   answer into *a as in the stdio transport; REFUSED says the server
   refused it at the HTTP level (m->http_status). */
int janas_mcp_http_exchange(struct janas_mcp *m, const char *method, int64_t id,
                            const char *body, size_t n, double ms,
                            struct janas_rpc_msg *a);
/* The Mcp-Param-* headers of a call into m->call_headers: 0, or -1 when
   the arguments are no JSON object. */
int janas_mcp_param_headers(struct janas_mcp *m, const struct janas_mcp_tool *t,
                            const struct janas_json *args);
/* x-mcp-header in a tool's schema: 0 with t->params set (NULL for none),
   -1 when an annotation breaks the rules and the tool cannot be used. */
int janas_mcp_schema_params(struct janas_mcp_tool *t,
                            const struct janas_json *schema);
/* An end to a legacy session (HTTP DELETE). */
void janas_mcp_http_end(struct janas_mcp *m);

/* Records the failure for janas_mcp_last_error; returns code. */
int32_t janas_mcp_fail(int32_t code, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

/* mcp_content.c: the tools of a tools/list page added to m (0, or -1 on
   memory); the content of a tools/call result as text into m->result. */
int janas_mcp_take_tools(struct janas_mcp *m, const struct janas_json *list);
void janas_mcp_take_result(struct janas_mcp *m, const struct janas_json *res);
void janas_mcp_free_tools(struct janas_mcp *m);

#endif
