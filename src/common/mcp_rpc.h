/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_rpc.h - JSON-RPC 2.0 messages as the Model Context Protocol uses them:
 * one message per line, requests with an integer id from this side, and
 * whatever the other side writes read back into its kind.
 *
 * A message comes from a program the user configured but did not write, so
 * the reader takes any line: what is not a message is JANAS_RPC_INVALID,
 * never a crash (tests/test_mcp.c mutates lines to hold it to that).
 */
#ifndef JANAS_MCP_RPC_H
#define JANAS_MCP_RPC_H

#include <stddef.h>
#include <stdint.h>

#include "llm/json.h"

enum janas_rpc_kind {
    JANAS_RPC_INVALID,
    JANAS_RPC_RESULT,  /* the answer to a request: result */
    JANAS_RPC_ERROR,   /* the answer to a request: error */
    JANAS_RPC_REQUEST, /* the other side asks: method, params, id */
    JANAS_RPC_NOTIFY,  /* the other side tells: method, params */
};

struct janas_rpc_msg {
    enum janas_rpc_kind kind;
    struct janas_json_doc *doc;
    /* the id: as written (for an answer to a request of the other side),
       and as an integer when it is one (the answers to ours) */
    const struct janas_json *id;
    int has_num_id;
    int64_t num_id;
    const char *method; /* request, notification */
    const struct janas_json *params, *result;
    int64_t code; /* error */
    const char *message;
    const struct janas_json *data;
};

/* Reads one line. Returns 0 for a message, -1 (kind INVALID, the reason in
   err) for anything else; janas_rpc_free either way. */
int janas_rpc_parse(const char *line, size_t n, struct janas_rpc_msg *m,
                    char *err, size_t err_len);
void janas_rpc_free(struct janas_rpc_msg *m);

/* A request (id >= 0) or a notification (id < 0): what comes after
   janas_rpc_begin is the params object, then janas_rpc_end closes the
   message and its line. */
void janas_rpc_begin(struct janas_buf *b, int64_t id, const char *method);
void janas_rpc_end(struct janas_buf *b);

/* The answer to a request of the other side: an empty result, or an error
   (message a plain string). */
void janas_rpc_answer(struct janas_buf *b, const struct janas_json *id,
                      int64_t code, const char *message);

#endif
