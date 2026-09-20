/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_rpc.c - JSON-RPC 2.0 messages, read and written (see mcp_rpc.h).
 */
#include "common/mcp_rpc.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

void janas_rpc_free(struct janas_rpc_msg *m)
{
    janas_json_free(m->doc);
    memset(m, 0, sizeof(*m));
}

static int fail(struct janas_rpc_msg *m, char *err, size_t err_len,
                const char *why)
{
    snprintf(err, err_len, "%s", why);
    m->kind = JANAS_RPC_INVALID;
    return -1;
}

int janas_rpc_parse(const char *line, size_t n, struct janas_rpc_msg *m,
                    char *err, size_t err_len)
{
    memset(m, 0, sizeof(*m));
    m->doc = janas_json_parse(line, n, err, err_len);
    const struct janas_json *root = janas_json_root(m->doc);
    if (!root) {
        m->kind = JANAS_RPC_INVALID;
        return -1;
    }
    if (root->type != JANAS_JSON_OBJECT)
        return fail(m, err, err_len, "a message must be an object");
    if (!janas_json_is(janas_json_get(root, "jsonrpc"), "2.0"))
        return fail(m, err, err_len, "not JSON-RPC 2.0");
    m->id = janas_json_get(root, "id");
    if (m->id && m->id->type == JANAS_JSON_NUMBER) {
        double d = janas_json_num(m->id, -1);
        if (d >= 0 && d < 9007199254740992.0 && floor(d) == d) {
            m->has_num_id = 1;
            m->num_id = (int64_t)d;
        }
    }
    if (m->id && m->id->type == JANAS_JSON_NULL)
        m->id = NULL;
    if (m->id && m->id->type != JANAS_JSON_NUMBER &&
        m->id->type != JANAS_JSON_STRING)
        return fail(m, err, err_len, "an id is a number or a string");
    const struct janas_json *method = janas_json_get(root, "method");
    const struct janas_json *result = janas_json_get(root, "result");
    const struct janas_json *error = janas_json_get(root, "error");
    if (method) {
        m->method = janas_json_str(method);
        if (!m->method || result || error)
            return fail(m, err, err_len, "a method must be a string");
        m->params = janas_json_get(root, "params");
        if (m->params && m->params->type != JANAS_JSON_OBJECT &&
            m->params->type != JANAS_JSON_ARRAY)
            return fail(m, err, err_len, "params must be structured");
        m->kind = m->id ? JANAS_RPC_REQUEST : JANAS_RPC_NOTIFY;
        return 0;
    }
    if (result && !error) {
        if (!m->id)
            return fail(m, err, err_len, "a result without an id");
        m->result = result;
        m->kind = JANAS_RPC_RESULT;
        return 0;
    }
    if (error && !result && error->type == JANAS_JSON_OBJECT) {
        const struct janas_json *code = janas_json_get(error, "code");
        double d = janas_json_num(code, NAN);
        if (!code || code->type != JANAS_JSON_NUMBER || floor(d) != d ||
            fabs(d) > 9007199254740992.0)
            return fail(m, err, err_len, "an error needs an integer code");
        m->code = (int64_t)d;
        m->message = janas_json_str(janas_json_get(error, "message"));
        if (!m->message)
            m->message = "";
        m->data = janas_json_get(error, "data");
        m->kind = JANAS_RPC_ERROR;
        return 0;
    }
    return fail(m, err, err_len, "neither a request nor an answer");
}

void janas_rpc_begin(struct janas_buf *b, int64_t id, const char *method)
{
    janas_buf_puts(b, "{\"jsonrpc\":\"2.0\",");
    if (id >= 0)
        janas_buf_printf(b, "\"id\":%lld,", (long long)id);
    janas_buf_puts(b, "\"method\":");
    janas_json_write_str(b, method, strlen(method));
    janas_buf_puts(b, ",\"params\":");
}

void janas_rpc_end(struct janas_buf *b)
{
    janas_buf_puts(b, "}\n");
}

void janas_rpc_answer(struct janas_buf *b, const struct janas_json *id,
                      int64_t code, const char *message)
{
    janas_buf_puts(b, "{\"jsonrpc\":\"2.0\",\"id\":");
    janas_json_write(b, id);
    if (code) {
        janas_buf_printf(
            b, ",\"error\":{\"code\":%lld,\"message\":", (long long)code);
        janas_json_write_str(b, message, strlen(message));
        janas_buf_puts(b, "}}\n");
    } else {
        janas_buf_puts(b, ",\"result\":{}}\n");
    }
}
