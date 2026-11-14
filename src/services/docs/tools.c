/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - janas-docs's tools, all three of which only read: docs_search
 * (the pieces of the documents that answer a question), docs_read (a
 * document's text, a stretch at a time) and docs_status (what is indexed,
 * by what, how far). The answers are text for the model.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/mcp_server.h"
#include "docs.h"
#include "services/common/private.h"

#define READ_DEFAULT 12000
#define READ_MAX 50000

static const char *const TOOLS[] = {
    "\"tools\": [",
    "{\"name\": \"docs_search\", \"title\": \"Search the documents\", ",
    "\"description\": \"The passages of the user's documents (the folders ",
    "of documents janas-docs indexes: text, Markdown, Word and ",
    "LibreOffice and PDF files) that best answer a question or hold some words, ",
    "each with the document it comes from. Ask in the words the document ",
    "would use; names, codes and numbers are found as written.\", ",
    "\"inputSchema\": {\"type\": \"object\", \"properties\": {\"query\": ",
    "{\"type\": \"string\", \"description\": \"what to look for\"}, ",
    "\"folder\": {\"type\": \"string\", \"description\": \"only the ",
    "documents under this folder (~/...)\"}, \"max\": {\"type\": ",
    "\"number\", \"description\": \"passages, 6 by default, at most ",
    "20\"}}, \"required\": [\"query\"], \"additionalProperties\": ",
    "false}, \"annotations\": {\"readOnlyHint\": true}}, ",
    "{\"name\": \"docs_read\", \"title\": \"Read a document\", ",
    "\"description\": \"The text of one of the documents (as docs_search ",
    "names it), from offset on: 12000 bytes unless asked; the answer says ",
    "how long the whole text is.\", \"inputSchema\": {\"type\": ",
    "\"object\", \"properties\": {\"path\": {\"type\": \"string\"}, ",
    "\"offset\": {\"type\": \"number\", \"description\": \"the first byte, ",
    "from 0\"}, \"length\": {\"type\": \"number\", \"description\": ",
    "\"bytes, at most 50000\"}}, \"required\": [\"path\"], ",
    "\"additionalProperties\": false}, \"annotations\": {\"readOnlyHint\": ",
    "true}}, ",
    "{\"name\": \"docs_status\", \"title\": \"The documents indexed\", ",
    "\"description\": \"Which folders are indexed, how many documents and ",
    "passages, whether they are searched by meaning too and with which ",
    "model, and how far the indexing is.\", \"inputSchema\": {\"type\": ",
    "\"object\", \"properties\": {}, \"additionalProperties\": false}, ",
    "\"annotations\": {\"readOnlyHint\": true}}]",
};

void dz_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    for (size_t i = 0; i < sizeof TOOLS / sizeof TOOLS[0]; i++)
        janas_buf_puts(b, TOOLS[i]);
}

/* the stretch of the indexing still to go, for an answer (x->mu held) */
static void progress(const struct dz_index *x, struct janas_buf *o)
{
    if (x->walking && x->read_done < x->to_read)
        janas_buf_printf(o, "; still reading the documents: %zu of %zu",
                         x->read_done, x->to_read);
    else if (x->walking && x->to_embed)
        janas_buf_printf(o,
                         "; still making the vectors: %zu of %zu passages, "
                         "those without one are found by their words",
                         x->embed_done, x->to_embed);
}

static void no_folders(struct dz_ctx *c, struct janas_buf *o)
{
    janas_buf_printf(o,
                     "No folder of documents: %s. List them, a path a line, "
                     "in ~/.config/janas/docs-dirs.txt.",
                     c->index.roots->listed
                         ? "none of those listed in "
                           "~/.config/janas/docs-dirs.txt is there"
                         : "there is no Documents folder in the home");
}

int dz_tool_search(struct dz_ctx *c, const char *query, const char *folder,
                   int max, struct janas_buf *o)
{
    struct dz_index *x = &c->index;
    if (!query || !*query) {
        janas_buf_puts(o, "no query given");
        return -1;
    }
    max = max < 1 ? 6 : max > 20 ? 20 : max;
    char dir[PATH_MAX];
    if (folder && *folder) {
        char p[PATH_MAX], home[PATH_MAX];
        pthread_mutex_lock(&x->mu);
        snprintf(home, sizeof home, "%s", x->roots->home);
        pthread_mutex_unlock(&x->mu);
        if (folder[0] == '~' && (folder[1] == 0 || folder[1] == '/'))
            snprintf(p, sizeof p, "%s%s", home, folder + 1);
        else
            snprintf(p, sizeof p, "%s", folder);
        if (!realpath(p, dir)) {
            janas_buf_printf(o, "%s: %s", folder, strerror(errno));
            return -1;
        }
        folder = dir;
    } else {
        folder = NULL;
    }
    /* the query's vector, made before the index is locked */
    char name[sizeof c->emb.name];
    int dim = 0, ready = dz_embed_state(&c->emb, name, sizeof name, &dim);
    float *qv = ready && dim ? malloc((size_t)dim * sizeof(float)) : NULL;
    size_t ql = strlen(query);
    if (qv && dz_embed(&c->emb, &query, &ql, 1, 1, qv) != 0) {
        free(qv);
        qv = NULL;
    }
    pthread_mutex_lock(&x->mu);
    if (time(NULL) - x->last_walk > 60 && !x->walking) {
        x->woken = 1; /* a document added a minute ago is found next time */
        pthread_cond_signal(&x->wake);
    }
    if (!x->roots->n_root) {
        no_folders(c, o);
        pthread_mutex_unlock(&x->mu);
        free(qv);
        return 0;
    }
    int by_meaning = qv && !strcmp(x->source, name) && x->dim == dim;
    struct dz_hit *h = NULL;
    long n = dz_search(x, query, by_meaning ? qv : NULL, dim, folder, max, &h);
    free(qv);
    if (n < 0) {
        pthread_mutex_unlock(&x->mu);
        janas_buf_puts(o, "out of memory");
        return -1;
    }
    size_t docs = 0;
    for (long i = 0; i < n; i++) {
        int seen = 0;
        for (long j = 0; j < i && !seen; j++)
            seen = h[j].file == h[i].file;
        docs += !seen;
    }
    janas_buf_printf(o, "%ld passage%s from %zu document%s, found by %s", n,
                     n == 1 ? "" : "s", docs, docs == 1 ? "" : "s",
                     by_meaning ? "their meaning and their words"
                                : "their words");
    progress(x, o);
    if (!n)
        janas_buf_printf(o, ". %zu documents are indexed%s.", x->n,
                         x->n ? "; try other words" : "");
    else
        janas_buf_puts(o, ".");
    for (long i = 0; i < n; i++) {
        const struct dz_file *f = &x->f[h[i].file];
        const struct dz_chunk *ch = &f->c[h[i].chunk];
        char s[PATH_MAX];
        dz_short(x->roots, f->path, s, sizeof s);
        janas_buf_printf(o, "\n\n[%ld] %s, passage %zu of %u", i + 1, s,
                         h[i].chunk + 1, f->n_chunk);
        if (h[i].words > 0 && h[i].meaning)
            janas_buf_printf(o, " (words and meaning, %.2f)", h[i].meaning);
        else if (h[i].meaning)
            janas_buf_printf(o, " (meaning, %.2f)", h[i].meaning);
        else
            janas_buf_puts(o, " (words)");
        janas_buf_puts(o, "\n");
        janas_buf_put(o, ch->text, ch->n);
    }
    pthread_mutex_unlock(&x->mu);
    free(h);
    return o->oom ? -1 : 0;
}

int dz_tool_read(struct dz_ctx *c, const char *path, double offset,
                 double length, struct janas_buf *o)
{
    char real[PATH_MAX], why[PATH_MAX + 128];
    pthread_mutex_lock(&c->index.mu);
    struct dz_roots roots = *c->index.roots;
    pthread_mutex_unlock(&c->index.mu);
    if (dz_resolve(&roots, path, real, sizeof real, why, sizeof why) != 0) {
        janas_buf_puts(o, why);
        return -1;
    }
    struct janas_buf t = {0};
    char s[PATH_MAX];
    dz_short(&roots, real, s, sizeof s);
    if (dz_extract(real, &t, why, 256) != 0) {
        janas_buf_printf(o, "%s: %s", s, why);
        janas_buf_free(&t);
        return -1;
    }
    size_t n = t.n;
    size_t at = offset > 0 ? (offset < (double)n ? (size_t)offset : n) : 0;
    size_t len = length >= 1 ? (length < READ_MAX ? (size_t)length : READ_MAX)
                             : READ_DEFAULT;
    while (at < n && ((unsigned char)t.p[at] & 0xc0) == 0x80)
        at++; /* at a character's start */
    size_t end = n - at > len ? at + len : n;
    while (end < n && ((unsigned char)t.p[end] & 0xc0) == 0x80)
        end++;
    janas_buf_printf(o, "%s: %zu bytes of text", s, n);
    if (at || end < n)
        janas_buf_printf(o, "; from %zu to %zu", at, end);
    if (end < n)
        janas_buf_printf(o, " (the rest from offset %zu)", end);
    janas_buf_puts(o, "\n\n");
    janas_buf_put(o, t.p ? t.p + at : "", end - at);
    janas_buf_free(&t);
    return o->oom ? -1 : 0;
}

int dz_tool_status(struct dz_ctx *c, struct janas_buf *o)
{
    struct dz_index *x = &c->index;
    char name[sizeof c->emb.name], err[sizeof c->emb.err];
    int dim, ready = dz_embed_state(&c->emb, name, sizeof name, &dim);
    dz_embed_error(&c->emb, err, sizeof err);
    pthread_mutex_lock(&x->mu);
    if (!x->roots->n_root) {
        no_folders(c, o);
    } else {
        janas_buf_puts(o, "Folders of documents:");
        for (int i = 0; i < x->roots->n_root; i++) {
            char s[PATH_MAX];
            dz_short(x->roots, x->roots->root[i], s, sizeof s);
            janas_buf_printf(o, "%s %s", i ? "," : "", s);
        }
        janas_buf_printf(o, " (%s).\n",
                         x->roots->listed ? "from ~/.config/janas/docs-dirs.txt"
                                          : "the Documents folder: list "
                                            "others in "
                                            "~/.config/janas/docs-dirs.txt");
    }
    size_t pieces = 0, vec = 0, empty = 0;
    for (size_t i = 0; i < x->n; i++) {
        pieces += x->f[i].n_chunk;
        empty += !x->f[i].n_chunk;
        for (uint32_t k = 0; k < x->f[i].n_chunk; k++)
            vec += x->f[i].c[k].vec != NULL;
    }
    janas_buf_printf(o, "%zu documents indexed, %zu passages", x->n, pieces);
    if (empty)
        janas_buf_printf(o, "; %zu without text (or unreadable)", empty);
    janas_buf_puts(o, ".\n");
    if (c->emb.kind == DZ_EMB_NONE) {
        janas_buf_puts(o, "Searched by their words alone: no embedding model "
                          "(server or embedding_model in "
                          "~/.config/janas/docs.conf).\n");
        if (vec)
            janas_buf_printf(o,
                             "%zu passages keep their vectors from %s, used "
                             "again with it.\n",
                             vec, x->source);
    } else if (!ready)
        janas_buf_printf(o,
                         "Embeddings from %s, not answering: %s. Searched "
                         "by their words meanwhile.\n",
                         name, err);
    else
        janas_buf_printf(o,
                         "Searched by their words and their meaning: vectors "
                         "from %s, for %zu of the %zu passages.\n",
                         name, vec, pieces);
    if (x->walking)
        janas_buf_puts(o, "Indexing now");
    else if (x->last_walk)
        janas_buf_printf(o, "Last looked at %ld s ago",
                         (long)(time(NULL) - x->last_walk));
    else
        janas_buf_puts(o, "Not looked at yet");
    progress(x, o);
    janas_buf_puts(o, ".");
    if (x->failed)
        janas_buf_printf(o, "\n%zu failure%s; the last: %s", x->failed,
                         x->failed == 1 ? "" : "s", x->last_fail);
    pthread_mutex_unlock(&x->mu);
    return o->oom ? -1 : 0;
}

int dz_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    struct dz_ctx *c = ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    struct janas_buf t = {0};
    int rc;
    if (janas_json_is(name, "docs_search"))
        rc = dz_tool_search(c, janas_json_str(janas_json_get(args, "query")),
                            janas_json_str(janas_json_get(args, "folder")),
                            (int)janas_json_num(janas_json_get(args, "max"), 6),
                            &t);
    else if (janas_json_is(name, "docs_read"))
        rc =
            dz_tool_read(c, janas_json_str(janas_json_get(args, "path")),
                         janas_json_num(janas_json_get(args, "offset"), 0),
                         janas_json_num(janas_json_get(args, "length"), 0), &t);
    else if (janas_json_is(name, "docs_status"))
        rc = dz_tool_status(c, &t);
    else
        return -1;
    if (t.oom) {
        janas_buf_free(&t);
        janas_mcps_text_result(b, "out of memory", 13, 1);
        return 0;
    }
    janas_mcps_text_result(b, t.p ? t.p : "", t.n, rc != 0);
    janas_buf_free(&t);
    return 0;
}
