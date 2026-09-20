/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_content.c - what an MCP server gives, made into what a model reads:
 * its tools as the functions of a chat template, the answer of a call as
 * text (see include/janas/mcp.h).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/mcp_client.h"

#define MAX_TOOLS 1024

static char *dup_n(const char *s, size_t n)
{
    char *d = malloc(n + 1);
    if (d) {
        memcpy(d, s, n);
        d[n] = 0;
    }
    return d;
}

void janas_mcp_free_tools(struct janas_mcp *m)
{
    for (size_t i = 0; i < m->n_tools; i++) {
        free(m->tools[i].name);
        free(m->tools[i].description);
        free(m->tools[i].schema);
        free(m->tools[i].params);
    }
    free(m->tools);
    m->tools = NULL;
    m->n_tools = 0;
}

/* A name a chat template can carry and a grammar can hold a call to: the
   characters the specification recommends, and a few more, never those
   that would end the name inside the markup or a JSON string. */
static int plain_name(const struct janas_json *v)
{
    if (!v || v->type != JANAS_JSON_STRING || v->n == 0 || v->n > 128)
        return 0;
    for (size_t i = 0; i < v->n; i++) {
        unsigned char c = (unsigned char)v->s[i];
        if (c <= ' ' || c >= 0x7f || c == '"' || c == '<' || c == '>' ||
            c == '\\' || c == '\'' || c == '=')
            return 0;
    }
    return 1;
}

int janas_mcp_take_tools(struct janas_mcp *m, const struct janas_json *list)
{
    if (!list || list->type != JANAS_JSON_ARRAY)
        return 0;
    for (const struct janas_json *t = list->child; t; t = t->next) {
        const struct janas_json *name = janas_json_get(t, "name");
        if (!plain_name(name) || m->n_tools >= MAX_TOOLS)
            continue; /* a tool the model could not call */
        int dup = 0;
        for (size_t i = 0; i < m->n_tools && !dup; i++)
            dup = strcmp(m->tools[i].name, name->s) == 0;
        if (dup)
            continue;
        const struct janas_json *d = janas_json_get(t, "description");
        if (!janas_json_str(d))
            d = janas_json_get(t, "title");
        const struct janas_json *schema = janas_json_get(t, "inputSchema");
        struct janas_buf sb = {0};
        if (schema && schema->type == JANAS_JSON_OBJECT)
            janas_json_write(&sb, schema);
        else
            janas_buf_puts(&sb, "{\"type\": \"object\"}");
        janas_buf_put(&sb, "", 1);
        struct janas_mcp_tool *grown =
            realloc(m->tools, (m->n_tools + 1) * sizeof(*grown));
        if (!grown || sb.oom) {
            janas_buf_free(&sb);
            return -1;
        }
        m->tools = grown;
        struct janas_mcp_tool *w = &m->tools[m->n_tools];
        *w = (struct janas_mcp_tool){0};
        /* over HTTP, a tool whose x-mcp-header marks break the rules is
           left out, as the specification asks */
        if (m->http && schema && schema->type == JANAS_JSON_OBJECT &&
            janas_mcp_schema_params(w, schema) != 0) {
            janas_buf_free(&sb);
            continue;
        }
        const struct janas_json *ro =
            janas_json_get(janas_json_get(t, "annotations"), "readOnlyHint");
        w->read_only = !ro ? -1 : ro->type == JANAS_JSON_TRUE ? 1 : 0;
        w->name = dup_n(name->s, name->n);
        w->description = janas_json_str(d) ? dup_n(d->s, d->n) : dup_n("", 0);
        w->schema = sb.p;
        if (!w->name || !w->description) {
            free(w->name);
            free(w->description);
            free(w->schema);
            free(w->params);
            return -1;
        }
        m->n_tools++;
    }
    return 0;
}

/* One item of a result's content, as a model can read it. */
static void content_item(struct janas_buf *b, const struct janas_json *c)
{
    const struct janas_json *type = janas_json_get(c, "type");
    const char *mime = janas_json_str(janas_json_get(c, "mimeType"));
    if (janas_json_is(type, "text")) {
        const struct janas_json *t = janas_json_get(c, "text");
        if (janas_json_str(t))
            janas_buf_put(b, t->s, t->n);
    } else if (janas_json_is(type, "image") || janas_json_is(type, "audio")) {
        const struct janas_json *data = janas_json_get(c, "data");
        janas_buf_printf(b, "[%s%s%s, %zu bytes, not shown]", type->s,
                         mime ? " " : "", mime ? mime : "",
                         janas_json_str(data) ? data->n / 4 * 3 : 0);
    } else if (janas_json_is(type, "resource_link")) {
        const char *uri = janas_json_str(janas_json_get(c, "uri"));
        const char *name = janas_json_str(janas_json_get(c, "name"));
        const char *desc = janas_json_str(janas_json_get(c, "description"));
        janas_buf_printf(b, "[link: %s <%s>%s%s]", name ? name : "",
                         uri ? uri : "", desc ? " " : "", desc ? desc : "");
    } else if (janas_json_is(type, "resource")) {
        const struct janas_json *r = janas_json_get(c, "resource");
        const char *uri = janas_json_str(janas_json_get(r, "uri"));
        const struct janas_json *text = janas_json_get(r, "text");
        mime = janas_json_str(janas_json_get(r, "mimeType"));
        if (janas_json_str(text)) {
            janas_buf_printf(b, "[%s]\n", uri ? uri : "resource");
            janas_buf_put(b, text->s, text->n);
        } else {
            janas_buf_printf(b, "[%s%s%s, binary, not shown]",
                             uri ? uri : "resource", mime ? ", " : "",
                             mime ? mime : "");
        }
    } else {
        const char *t = janas_json_str(type);
        janas_buf_printf(b, "[content of type %s, not shown]", t ? t : "?");
    }
}

void janas_mcp_take_result(struct janas_mcp *m, const struct janas_json *res)
{
    m->result.n = 0;
    m->result.oom = 0;
    m->result_error = janas_json_get(res, "isError") &&
                      janas_json_get(res, "isError")->type == JANAS_JSON_TRUE;
    const struct janas_json *content = janas_json_get(res, "content");
    int any = 0;
    if (content && content->type == JANAS_JSON_ARRAY)
        for (const struct janas_json *c = content->child; c; c = c->next) {
            if (any)
                janas_buf_put(&m->result, "\n", 1);
            content_item(&m->result, c);
            any = 1;
        }
    const struct janas_json *sc = janas_json_get(res, "structuredContent");
    if (!any && sc)
        janas_json_write(&m->result, sc);
    janas_buf_put(&m->result, "", 1); /* NUL-terminated, not counted */
    if (m->result.n)
        m->result.n--;
}
