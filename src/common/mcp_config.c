/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_config.c - the configured MCP servers (see mcp_config.h).
 */
#include "common/mcp_config.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/json.h"

#define CONFIG_MAX_BYTES ((size_t)4 << 20)

static char *dup_n(const char *s, size_t n)
{
    char *d = malloc(n + 1);
    if (d) {
        memcpy(d, s, n);
        d[n] = 0;
    }
    return d;
}

void janas_mcp_config_free(struct janas_mcp_conf *c, size_t n)
{
    if (!c)
        return;
    for (size_t i = 0; i < n; i++) {
        free(c[i].name);
        free(c[i].url);
        for (char **a = c[i].argv; a && *a; a++)
            free(*a);
        free(c[i].argv);
        for (char **e = c[i].env; e && *e; e++)
            free(*e);
        free(c[i].env);
        for (char **h = c[i].headers; h && *h; h++)
            free(*h);
        free(c[i].headers);
    }
    free(c);
}

/* A string with no NUL inside: what a command line or a variable can be. */
static const char *plain(const struct janas_json *v)
{
    const char *s = janas_json_str(v);
    return s && strlen(s) == v->n ? s : NULL;
}

static int one(const struct janas_json *s, struct janas_mcp_conf *c, char *err,
               size_t err_len)
{
    if (s->type != JANAS_JSON_OBJECT || memchr(s->key, 0, s->key_n) ||
        s->key_n == 0) {
        snprintf(err, err_len, "a server must be an object with a name");
        return -1;
    }
    c->name = dup_n(s->key, s->key_n);
    const struct janas_json *d = janas_json_get(s, "disabled");
    c->disabled = d && d->type == JANAS_JSON_TRUE;
    const struct janas_json *type = janas_json_get(s, "type");
    const struct janas_json *url = janas_json_get(s, "url");
    const struct janas_json *cmd = janas_json_get(s, "command");
    if (!c->name)
        goto oom;
    if (type && !janas_json_is(type, "stdio") && !janas_json_is(type, "http") &&
        !janas_json_is(type, "sse") &&
        !janas_json_is(type, "streamable-http")) {
        snprintf(err, err_len, "server %s: unknown type", c->name);
        return -1;
    }
    if (url) {
        if (!plain(url)) {
            snprintf(err, err_len, "server %s: url must be a string", c->name);
            return -1;
        }
        c->url = dup_n(url->s, url->n);
        const struct janas_json *hs = janas_json_get(s, "headers");
        if (hs && hs->type != JANAS_JSON_OBJECT) {
            snprintf(err, err_len, "server %s: headers must be an object",
                     c->name);
            return -1;
        }
        c->headers = calloc((hs ? hs->n : 0) + 1, sizeof(*c->headers));
        if (!c->url || !c->headers)
            goto oom;
        size_t k = 0;
        for (const struct janas_json *h = hs ? hs->child : NULL; h;
             h = h->next) {
            /* a name of token characters, a value with no line break */
            int ok = plain(h) && h->key_n > 0 && h->key_n < 64 &&
                     !strpbrk(h->s, "\r\n");
            for (size_t i = 0; ok && i < h->key_n; i++)
                ok = (unsigned char)h->key[i] > ' ' &&
                     (unsigned char)h->key[i] < 0x7f &&
                     !strchr("()<>@,;:\\\"/[]?={}", h->key[i]);
            if (!ok) {
                snprintf(err, err_len,
                         "server %s: headers hold Name: \"value\"", c->name);
                return -1;
            }
            char *line = malloc(h->key_n + h->n + 3);
            if (!line)
                goto oom;
            memcpy(line, h->key, h->key_n);
            memcpy(line + h->key_n, ": ", 2);
            memcpy(line + h->key_n + 2, h->s, h->n + 1);
            c->headers[k++] = line;
        }
        return 0;
    }
    if (!plain(cmd) || cmd->n == 0) {
        snprintf(err, err_len, "server %s: a command or a url is needed",
                 c->name);
        return -1;
    }
    const struct janas_json *args = janas_json_get(s, "args");
    if (args && args->type != JANAS_JSON_ARRAY) {
        snprintf(err, err_len, "server %s: args must be an array", c->name);
        return -1;
    }
    size_t na = args ? args->n : 0;
    c->argv = calloc(na + 2, sizeof(*c->argv));
    if (!c->argv || !(c->argv[0] = dup_n(cmd->s, cmd->n)))
        goto oom;
    size_t k = 1;
    for (const struct janas_json *a = args ? args->child : NULL; a;
         a = a->next) {
        if (!plain(a)) {
            snprintf(err, err_len, "server %s: every argument is a string",
                     c->name);
            return -1;
        }
        if (!(c->argv[k++] = dup_n(a->s, a->n)))
            goto oom;
    }
    const struct janas_json *env = janas_json_get(s, "env");
    if (env && env->type != JANAS_JSON_OBJECT) {
        snprintf(err, err_len, "server %s: env must be an object", c->name);
        return -1;
    }
    c->env = calloc((env ? env->n : 0) + 1, sizeof(*c->env));
    if (!c->env)
        goto oom;
    k = 0;
    for (const struct janas_json *e = env ? env->child : NULL; e; e = e->next) {
        if (!plain(e) || e->key_n == 0 || memchr(e->key, '=', e->key_n) ||
            memchr(e->key, 0, e->key_n)) {
            snprintf(err, err_len, "server %s: env holds NAME: \"value\"",
                     c->name);
            return -1;
        }
        char *kv = malloc(e->key_n + e->n + 2);
        if (!kv)
            goto oom;
        memcpy(kv, e->key, e->key_n);
        kv[e->key_n] = '=';
        memcpy(kv + e->key_n + 1, e->s, e->n + 1);
        c->env[k++] = kv;
    }
    return 0;
oom:
    snprintf(err, err_len, "out of memory");
    return -1;
}

int janas_mcp_config_parse(const char *text, size_t len,
                           struct janas_mcp_conf **out, size_t *n, char *err,
                           size_t err_len)
{
    *out = NULL;
    *n = 0;
    struct janas_json_doc *doc = janas_json_parse(text, len, err, err_len);
    const struct janas_json *root = janas_json_root(doc);
    if (!root) {
        janas_json_free(doc);
        return -1;
    }
    const struct janas_json *list = janas_json_get(root, "mcpServers");
    if (!list)
        list = janas_json_get(root, "servers");
    if (!list || list->type != JANAS_JSON_OBJECT) {
        snprintf(err, err_len, "no \"mcpServers\" object");
        janas_json_free(doc);
        return -1;
    }
    struct janas_mcp_conf *c = calloc(list->n + 1, sizeof(*c));
    if (!c) {
        snprintf(err, err_len, "out of memory");
        janas_json_free(doc);
        return -1;
    }
    size_t k = 0;
    int rc = 0;
    for (const struct janas_json *s = list->child; s && rc == 0; s = s->next)
        rc = one(s, &c[k++], err, err_len);
    janas_json_free(doc);
    if (rc != 0) {
        janas_mcp_config_free(c, k);
        return -1;
    }
    *out = c;
    *n = k;
    return 0;
}

int janas_mcp_config_read(const char *path, struct janas_mcp_conf **out,
                          size_t *n, char *err, size_t err_len)
{
    *out = NULL;
    *n = 0;
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (errno == ENOENT)
            return 1;
        snprintf(err, err_len, "%s: %s", path, strerror(errno));
        return -1;
    }
    char *text = malloc(CONFIG_MAX_BYTES);
    size_t len = text ? fread(text, 1, CONFIG_MAX_BYTES, f) : 0;
    int big = text && len == CONFIG_MAX_BYTES;
    fclose(f);
    if (!text || big) {
        free(text);
        snprintf(err, err_len, "%s: %s", path,
                 big ? "larger than 4 MiB" : "out of memory");
        return -1;
    }
    char why[200];
    int rc = janas_mcp_config_parse(text, len, out, n, why, sizeof(why));
    free(text);
    if (rc != 0)
        snprintf(err, err_len, "%s: %s", path, why);
    return rc;
}

int janas_mcp_conf_path(char *buf, size_t n)
{
    const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
    if (xdg && *xdg)
        snprintf(buf, n, "%s/janas/mcp.json", xdg);
    else if (home && *home)
        snprintf(buf, n, "%s/.config/janas/mcp.json", home);
    else
        return -1;
    return 0;
}
