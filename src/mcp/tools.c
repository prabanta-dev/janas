/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - the tools janas-mcp offers: generate, a question to the model
 * running here, and embed, the vector of a text, each where a model for it
 * was given.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "serve.h"

static void schema_str(struct janas_buf *b, const char *name, const char *what)
{
    janas_buf_printf(b,
                     "\"%s\": {\"type\": \"string\", \"description\": ", name);
    janas_json_write_str(b, what, strlen(what));
    janas_buf_puts(b, "}");
}

void mcpd_tools_list(const struct mcpd *d, struct janas_buf *b)
{
    janas_buf_puts(b, "\"tools\": [");
    if (d->llm) {
        char what[512];
        snprintf(what, sizeof(what),
                 "Ask %s, a language model running on this machine: the "
                 "prompt and whatever it contains stay here. Returns its "
                 "answer. It knows only what the prompt tells it and what it "
                 "learnt in training; it has no tools and remembers nothing "
                 "between calls.",
                 d->name[0] ? d->name : "the local model");
        janas_buf_puts(b, "{\"name\": \"generate\", \"title\": \"Ask the "
                          "local model\", \"description\": ");
        janas_json_write_str(b, what, strlen(what));
        janas_buf_puts(b, ", \"inputSchema\": {\"type\": \"object\", "
                          "\"properties\": {");
        schema_str(b, "prompt", "What to ask, with everything it needs.");
        janas_buf_puts(b, ", ");
        schema_str(b, "system",
                   "Instructions for the model: its role, the "
                   "form of the answer (optional).");
        janas_buf_puts(b, ", \"max_tokens\": {\"type\": \"integer\", "
                          "\"minimum\": 1, \"description\": \"The longest "
                          "answer, in tokens (a token is about three "
                          "quarters of a word).\"}, \"temperature\": "
                          "{\"type\": \"number\", \"minimum\": 0, "
                          "\"maximum\": 2, \"description\": \"0: always the "
                          "most likely word; higher: more varied.\"}}, "
                          "\"required\": [\"prompt\"], "
                          "\"additionalProperties\": false}}");
    }
    if (d->emb) {
        char what[512];
        snprintf(what, sizeof(what),
                 "The embedding of a text from %s, running on this machine: "
                 "a vector of %d numbers, of unit length, as a JSON array. "
                 "Texts of like meaning have vectors whose dot product is "
                 "near 1.",
                 d->emb_name[0] ? d->emb_name : "the local model",
                 janas_llm_embed_dim(d->emb));
        janas_buf_puts(b, d->llm ? ", " : "");
        janas_buf_puts(b, "{\"name\": \"embed\", \"title\": \"Embedding of "
                          "a text\", \"description\": ");
        janas_json_write_str(b, what, strlen(what));
        janas_buf_puts(b, ", \"inputSchema\": {\"type\": \"object\", "
                          "\"properties\": {");
        schema_str(b, "text", "The text.");
        janas_buf_puts(b, ", \"dimensions\": {\"type\": \"integer\", "
                          "\"minimum\": 1, \"description\": \"Fewer numbers: "
                          "the first ones, made unit length again "
                          "(optional).\"}}, \"required\": [\"text\"], "
                          "\"additionalProperties\": false}}");
    }
    janas_buf_puts(b, "]");
}

/* A result of one text item. */
static void text_result(struct janas_buf *b, const char *t, size_t n,
                        int is_error)
{
    janas_buf_puts(b, "\"content\": [{\"type\": \"text\", \"text\": ");
    janas_json_write_str(b, t, n);
    janas_buf_printf(b, "}], \"isError\": %s", is_error ? "true" : "false");
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* 1 when cancelled: nothing is answered then. */
static int generate(struct mcpd *d, const struct janas_json *args,
                    struct janas_buf *b)
{
    const struct janas_json *prompt = janas_json_get(args, "prompt");
    const struct janas_json *system = janas_json_get(args, "system");
    const struct janas_json *max = janas_json_get(args, "max_tokens");
    const struct janas_json *temp = janas_json_get(args, "temperature");
    if (!janas_json_str(prompt)) {
        const char *m = "prompt is required, as a string";
        text_result(b, m, strlen(m), 1);
        return 0;
    }
    struct janas_llm_chat_params p = d->cp;
    if (max && max->type == JANAS_JSON_NUMBER) {
        double v = janas_json_num(max, 0);
        p.max_reply = v >= 1 && v < 1e6 ? (int32_t)v : p.max_reply;
    }
    if (temp && temp->type == JANAS_JSON_NUMBER) {
        double v = janas_json_num(temp, -1);
        p.temperature = v >= 0 && v <= 2 ? (float)v : p.temperature;
    }
    int32_t rc = janas_llm_chat_set_params(d->chat, &p);
    if (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_begin(d->chat);
    if (rc == JANAS_LLM_OK && janas_json_str(system))
        rc = janas_llm_chat_add(d->chat, JANAS_LLM_ROLE_SYSTEM, system->s,
                                (int32_t)system->n);
    if (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_add(d->chat, JANAS_LLM_ROLE_USER, prompt->s,
                                (int32_t)prompt->n);
    if (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_run(d->chat);
    if (rc != JANAS_LLM_OK) {
        const char *m = janas_llm_last_error();
        text_result(b, m, strlen(m), 1);
        return 0;
    }
    struct janas_buf answer = {0};
    char piece[256];
    int32_t len;
    double end = d->call_seconds > 0 ? now() + d->call_seconds : 0;
    unsigned pieces = 0;
    int late = 0;
    while ((rc = janas_llm_chat_next(d->chat, piece, sizeof(piece), &len)) ==
           JANAS_LLM_OK) {
        if (!janas_llm_chat_thinking(d->chat)) /* the answer alone */
            janas_buf_put(&answer, piece, (size_t)len);
        if (++pieces % 8 == 0 && mcpd_poll(d)) {
            janas_buf_free(&answer);
            return 1;
        }
        if (end && now() > end) {
            late = 1;
            break;
        }
    }
    /* the reasoning's markers stay out; the answer as the model ended it */
    size_t from = 0;
    while (from < answer.n && (answer.p[from] == '\n' || answer.p[from] == ' '))
        from++;
    if (answer.oom) {
        const char *m = "out of memory";
        text_result(b, m, strlen(m), 1);
    } else if (rc < 0 && answer.n == from) {
        const char *m = janas_llm_last_error();
        text_result(b, m, strlen(m), 1);
    } else {
        if (late)
            janas_buf_puts(&answer, "\n[cut: the time given to a call ran "
                                    "out]");
        else if (rc < 0)
            janas_buf_printf(&answer, "\n[cut: %s]", janas_llm_last_error());
        text_result(b, answer.p ? answer.p + from : "", answer.n - from, 0);
    }
    janas_buf_free(&answer);
    return 0;
}

static void embed(struct mcpd *d, const struct janas_json *args,
                  struct janas_buf *b)
{
    const struct janas_json *text = janas_json_get(args, "text");
    const struct janas_json *dims = janas_json_get(args, "dimensions");
    if (!janas_json_str(text)) {
        const char *m = "text is required, as a string";
        text_result(b, m, strlen(m), 1);
        return;
    }
    int32_t dim = janas_llm_embed_dim(d->emb), want = 0;
    if (dims && dims->type == JANAS_JSON_NUMBER) {
        double v = janas_json_num(dims, 0);
        want = v >= 1 && v <= dim ? (int32_t)v : 0;
    }
    float *v = malloc((size_t)dim * sizeof(float));
    int32_t n = 0, tokens = 0;
    if (!v || janas_llm_embed(d->emb, text->s, (int32_t)text->n, want, v, &n,
                              &tokens) != JANAS_LLM_OK) {
        const char *m = v ? janas_llm_last_error() : "out of memory";
        text_result(b, m, strlen(m), 1);
        free(v);
        return;
    }
    struct janas_buf a = {0};
    janas_buf_puts(&a, "[");
    for (int32_t i = 0; i < n; i++)
        janas_buf_printf(&a, "%s%.7g", i ? ", " : "",
                         isfinite(v[i]) ? (double)v[i] : 0.0);
    janas_buf_puts(&a, "]");
    free(v);
    if (a.oom) {
        const char *m = "out of memory";
        text_result(b, m, strlen(m), 1);
    } else {
        text_result(b, a.p, a.n, 0);
        janas_buf_puts(b, ", \"structuredContent\": {\"embedding\": ");
        janas_buf_put(b, a.p, a.n);
        janas_buf_printf(b, ", \"tokens\": %d}", tokens);
    }
    janas_buf_free(&a);
}

int mcpd_tools_call(struct mcpd *d, const struct janas_json *params,
                    struct janas_buf *b)
{
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    if (d->llm && janas_json_is(name, "generate"))
        return generate(d, args, b);
    if (d->emb && janas_json_is(name, "embed")) {
        embed(d, args, b);
        return 0;
    }
    return -1;
}
