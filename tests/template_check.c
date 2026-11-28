/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * template_check.c - a model's conversations as Janas's format in C writes
 * them (api_load.c) and as the model's own chat template does, through
 * Janas's Jinja interpreter (api_template.c, what a chat uses), token by token:
 * the same, or where they part, with the tokens around it.
 *
 *   template_check MODEL.jns CONVERSATION.json...
 *
 * A conversation: {"messages": [{"role", "content", "tool_calls": [{"function":
 * {"name", "arguments": {...}}}]}...], "tools": [...], "thinking": true,
 * false or null}. A tool run by hand: it opens the model.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"

static char *slurp(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long l = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)l + 1);
    *n = b ? fread(b, 1, (size_t)l, f) : 0;
    if (b)
        b[*n] = 0;
    fclose(f);
    return b;
}

static int role_of(const char *r)
{
    return !strcmp(r, "system")      ? JANAS_LLM_ROLE_SYSTEM
           : !strcmp(r, "assistant") ? JANAS_LLM_ROLE_ASSISTANT
           : !strcmp(r, "tool")      ? JANAS_LLM_ROLE_TOOL
                                     : JANAS_LLM_ROLE_USER;
}

static void show(const janas_llm *llm, const int32_t *ids, size_t n,
                 size_t from, size_t to)
{
    for (size_t i = from; i < to && i < n; i++) {
        char t[128];
        int32_t len = 0;
        janas_llm_token_text(llm, ids[i], t, sizeof t - 1, &len);
        if (len < 0 || len >= (int32_t)sizeof t)
            len = 0;
        t[len] = 0;
        for (char *p = t; *p; p++)
            if (*p == '\n')
                *p = '~';
        printf(" %d[%s]", ids[i], t);
    }
    printf("\n");
}

/* TEMPLATE_DUMP=DIR: the template's variables as Janas gives them
   (NAME.ctx.json, {"vars", "data"}) and the chat's tokens written back as
   text (NAME.janas.txt), for Python's Jinja to render the same and the two
   to be compared */
static void dump(const char *dir, const char *conv, janas_llm_chat *c)
{
    const char *base = strrchr(conv, '/');
    base = base ? base + 1 : conv;
    char path[1024];
    struct janas_buf v = {0}, d = {0};
    janas_api_template_vars(c, c->n_msgs, 1, 1, NULL, &v, &d);
    snprintf(path, sizeof path, "%s/%s.ctx.json", dir, base);
    FILE *f = fopen(path, "wb");
    if (f) {
        fprintf(f, "{\"vars\": %.*s, \"data\": %.*s}\n", (int)v.n, v.p,
                (int)d.n, d.p);
        fclose(f);
    }
    janas_buf_free(&v);
    janas_buf_free(&d);
    snprintf(path, sizeof path, "%s/%s.janas.txt", dir, base);
    f = fopen(path, "wb");
    for (size_t i = 0; f && i < c->n_ids; i++) {
        char t[256];
        int32_t len = 0;
        janas_llm_token_text(c->llm, c->ids[i], t, sizeof t, &len);
        if (len > 0 && len <= (int32_t)sizeof t)
            fwrite(t, 1, (size_t)len, f);
    }
    if (f)
        fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "usage: template_check MODEL.jns CONVERSATION.json...\n");
        return 2;
    }
    struct janas_llm_params p = {
        .size = sizeof p, .n_ctx = 8192, .no_preload = 1};
    janas_llm *llm;
    if (janas_llm_open(argv[1], &p, &llm) != JANAS_LLM_OK) {
        fprintf(stderr, "template_check: %s\n", janas_llm_last_error());
        return 2;
    }
    int same = 0, differ = 0;
    for (int a = 2; a < argc; a++) {
        size_t n;
        char *text = slurp(argv[a], &n), err[256];
        struct janas_json_doc *d =
            text ? janas_json_parse(text, n, err, sizeof err) : NULL;
        if (!d) {
            printf("%s: not read\n", argv[a]);
            free(text);
            continue;
        }
        const struct janas_json *r = janas_json_root(d);
        const struct janas_json *th = janas_json_get(r, "thinking");
        struct janas_llm_chat_params cp;
        janas_llm_chat_params_default(&cp);
        cp.thinking = th && th->type == JANAS_JSON_TRUE    ? 1
                      : th && th->type == JANAS_JSON_FALSE ? 0
                                                           : -1;
        janas_llm_chat *c;
        if (janas_llm_chat_create(llm, &cp, &c) != JANAS_LLM_OK) {
            printf("%s: %s\n", argv[a], janas_llm_last_error());
            janas_json_free(d);
            free(text);
            continue;
        }
        const struct janas_json *tools = janas_json_get(r, "tools");
        if (tools) {
            struct janas_buf tb = {0};
            janas_json_write(&tb, tools);
            if (janas_llm_chat_tools(c, tb.p, (int32_t)tb.n) != JANAS_LLM_OK)
                printf("%s: tools: %s\n", argv[a], janas_llm_last_error());
            janas_buf_free(&tb);
        }
        janas_llm_chat_begin(c);
        for (const struct janas_json *m = janas_json_get(r, "messages")->child;
             m; m = m->next) {
            const char *role = janas_json_str(janas_json_get(m, "role"));
            const char *content = janas_json_str(janas_json_get(m, "content"));
            janas_llm_chat_add(c, role_of(role ? role : "user"),
                               content ? content : "", -1);
            const struct janas_json *calls = janas_json_get(m, "tool_calls");
            for (const struct janas_json *k = calls ? calls->child : NULL; k;
                 k = k->next) {
                const struct janas_json *f = janas_json_get(k, "function");
                const char *name = janas_json_str(janas_json_get(f, "name"));
                struct janas_buf ab = {0};
                janas_json_write(&ab, janas_json_get(f, "arguments"));
                janas_llm_chat_add_call(c, name, -1, ab.p, (int32_t)ab.n);
                janas_buf_free(&ab);
            }
        }
        int32_t *fmt = NULL;
        size_t nf = 0;
        uint32_t reply_off;
        int ok = janas_api_format_ids(c, &fmt, &nf) == JANAS_LLM_OK;
        int tok = ok && janas_api_template_build(c, &reply_off) == JANAS_LLM_OK;
        if (!tok) {
            printf("%s: %s\n", argv[a], janas_llm_last_error());
            differ++;
        } else {
            if (getenv("TEMPLATE_DUMP"))
                dump(getenv("TEMPLATE_DUMP"), argv[a], c);
            size_t i = 0;
            while (i < nf && i < c->n_ids && fmt[i] == c->ids[i])
                i++;
            /* the format opens the reasoning itself (<think> and a new
               line, Janas's choice), where the template leaves it to the
               model: the same otherwise */
            int opened = i == c->n_ids && nf > i && nf - i <= 2 &&
                         fmt[i] == llm->think_open;
            if (i == nf && i == c->n_ids) {
                same++;
                printf("%s: the same, %zu tokens\n", argv[a], nf);
            } else if (opened) {
                same++;
                printf("%s: the same, %zu tokens, and the format opens the "
                       "reasoning\n",
                       argv[a], c->n_ids);
            } else {
                differ++;
                printf("%s: they part at token %zu (format %zu tokens, "
                       "template %zu)\n  format:  ",
                       argv[a], i, nf, c->n_ids);
                show(llm, fmt, nf, i > 6 ? i - 6 : 0, i + 10);
                printf("  template:");
                show(llm, c->ids, c->n_ids, i > 6 ? i - 6 : 0, i + 10);
            }
            c->n_ids = 0;
        }
        free(fmt);
        janas_llm_chat_destroy(c);
        janas_json_free(d);
        free(text);
    }
    printf("template_check: %d the same, %d not\n", same, differ);
    janas_llm_close(llm);
    return differ ? 1 : 0;
}
