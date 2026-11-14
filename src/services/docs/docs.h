/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * docs.h - janas-docs, an MCP server that finds things in the user's own
 * documents: the folders listed in ~/.config/janas/docs-dirs.txt (the
 * system's Documents folder when there is no list), their text, Markdown,
 * docx and odt files cut into overlapping pieces and indexed in the
 * background, searched by meaning when there is an embedding model and by
 * their words always.
 *
 * The embeddings come, in this order, from the janas-server named in
 * ~/.config/janas/docs.conf (server = http://...), from an embedding
 * model the service opens itself (embedding_model = a .jns), or from
 * nowhere: then the words alone. The index on disk says which model made
 * its vectors, and a different one makes them again, keeping the text.
 */
#ifndef JANAS_DOCS_H
#define JANAS_DOCS_H

#include <limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "janas/llm.h"
#include "llm/json.h"

#define DZ_MAX_FILE (64 << 20) /* a document, and a file in its archive */
#define DZ_MAX_FILES 50000     /* documents indexed */
#define DZ_CHUNK 1200          /* bytes of a piece, about 300 tokens */
#define DZ_OVERLAP 200
#define DZ_ROOTS 16

/* kinds of document */
enum { DZ_NONE, DZ_TEXT, DZ_DOCX, DZ_ODT, DZ_PDF };

/* ---- where the documents are (roots.c) ---- */

struct dz_roots {
    char home[PATH_MAX];
    char root[DZ_ROOTS][PATH_MAX];
    int n_root;
    int listed; /* from docs-dirs.txt, not the Documents folder */
};

/* The folders; 0, or -1 when there is no home. */
int dz_roots_load(struct dz_roots *r);
/* ~ for the home */
void dz_short(const struct dz_roots *r, const char *path, char *out,
              size_t cap);
/* A path given (~/..., absolute) to a document that may be read: under a
   root, not private, of a kind read here. 0, or -1 with the reason. */
int dz_resolve(const struct dz_roots *r, const char *in, char *out, size_t cap,
               char *why, size_t wn);

/* ---- text out of documents (zip.c, xml.c, extract.c) ---- */

/* The file name of the zip archive z (n bytes) into a malloc'd *out, its
   *out_n bytes followed by a 0: 0, 1 when there is none, -1 with the
   reason in why. */
int dz_zip_get(const uint8_t *z, size_t n, const char *name, uint8_t **out,
               size_t *out_n, char *why, size_t wn);
/* The text of a docx's document.xml or an odt's content.xml (kind) added
   to out: 0, or -1 out of memory. */
int dz_xml_text(const char *x, size_t n, int kind, struct janas_buf *out);
/* The text of a PDF (pdf.c) added to out: 0, or -1 with the reason (not a
   PDF, encrypted, no text in it). */
int dz_pdf_text(const uint8_t *p, size_t n, struct janas_buf *out, char *why,
                size_t wn);
/* The kind of document a file's name says. */
int dz_kind(const char *path);
int dz_is_text(const char *t, size_t n);
/* The text of a document, of its kind, from its bytes or from its file,
   added to text: 0, or -1 with the reason in why. */
int dz_extract_mem(const uint8_t *p, size_t n, int kind, struct janas_buf *text,
                   char *why, size_t wn);
int dz_extract(const char *path, struct janas_buf *text, char *why, size_t wn);

/* ---- pieces and words (chunks.c) ---- */

struct dz_piece {
    size_t at, n; /* in the text */
};

/* The text cut into pieces of about size bytes, each beginning about
   overlap bytes before the last one ended, at whitespace and at the start
   of a UTF-8 character: how many into a malloc'd *out, or -1. */
long dz_cut(const char *t, size_t n, size_t size, size_t overlap,
            struct dz_piece **out);
/* The next word of *p (up to end): lower-case ASCII letters and digits,
   and the bytes of other UTF-8 characters. 1 with it in w, 0 at the end. */
int dz_word(const char **p, const char *end, char *w, size_t cap);
int dz_stopword(const char *w);

/* ---- embeddings (embed.c) ---- */

enum { DZ_EMB_NONE, DZ_EMB_SERVER, DZ_EMB_MODEL };

struct dz_embedder {
    int kind;
    char url[1024]; /* the server's, up to its /v1 */
    char key[512];  /* its API key, or empty */
    janas_llm *llm;
    pthread_mutex_t mu; /* one text at a time through the model */
    char name[1400];    /* what made the vectors: the index keeps it */
    int dim;            /* 0 until known */
    int ready;          /* it answered: vectors can be made */
    char err[1400];     /* why it is not ready */
};

/* The source of the embeddings: a server (url), else a model (path), else
   none. A server not answering yet is not an error: dz_embed_probe tries
   it again. 0, or -1 with the reason (a model that does not open). */
int dz_embedder_open(struct dz_embedder *e, const char *server, const char *key,
                     const char *model, char *why, size_t wn);
/* Asks the server for a vector, to learn its model and length: 0 when it
   answered. */
int dz_embed_probe(struct dz_embedder *e);
/* The vectors of n texts, of unit length, into out (n * e->dim): as
   passages, or as a search query (query). 0, or -1 with the reason in
   e->err. */
int dz_embed(struct dz_embedder *e, const char *const *texts,
             const size_t *lens, size_t n, int query, float *out);
/* Whether it is ready, with its name and length (name, dim may be NULL);
   its last error. Both under its lock: another thread may change them. */
int dz_embed_state(struct dz_embedder *e, char *name, size_t cap, int *dim);
void dz_embed_error(struct dz_embedder *e, char *out, size_t cap);
void dz_embedder_close(struct dz_embedder *e);
/* The vectors of an /embeddings answer (body, n bytes) for want texts
   into a malloc'd *out, with their length and the model's name: 0, or -1
   with the reason. */
int dz_embed_parse(const char *body, size_t n, size_t want, float **out,
                   int *dim, char *model, size_t model_cap, char *why,
                   size_t wn);

/* ---- the index (index.c, store.c) ---- */

struct dz_chunk {
    char *text;
    uint32_t n;
    float *vec; /* dim numbers, or NULL */
};

struct dz_file {
    char *path;
    int64_t mtime, size;
    uint32_t n_chunk;
    struct dz_chunk *c;
};

struct dz_index {
    pthread_mutex_t mu;
    pthread_cond_t wake;
    struct dz_file *f; /* by path */
    size_t n, cap;
    char source[1400]; /* what made the vectors, "" for none */
    int dim;
    char path[PATH_MAX]; /* its file */
    /* the indexing's state, for docs_status */
    size_t found, to_read, read_done, to_embed, embed_done, failed;
    char last_fail[PATH_MAX + 128];
    int walking;
    time_t last_walk;
    int stop, woken;
    pthread_t thread;
    int running;
    struct dz_roots *roots;
    struct dz_embedder *emb;
};

/* The index of the file path read (an index that is not there is empty):
   0, or -1 with the reason (a file that is not an index, or broken). */
int dz_index_load(struct dz_index *x, const char *path, char *why, size_t wn);
/* The same from the n bytes of an index file at p (store.c). */
int dz_index_parse(struct dz_index *x, const uint8_t *p, size_t n);
/* Written aside and renamed over it: 0, or -1. Under x->mu. */
int dz_index_save(struct dz_index *x);
void dz_index_free(struct dz_index *x);
/* The file at path in the index, by bisection (x->mu held): its place, or
   -1 with where it would go in *at. */
long dz_index_find(const struct dz_index *x, const char *path, size_t *at);

/* The thread that keeps the index up to date, and its end. */
int dz_indexer_start(struct dz_index *x);
void dz_indexer_stop(struct dz_index *x);
/* One round: the folders walked, the documents new or changed read, the
   vectors missing made. What a test or janas-docs --index runs alone. */
void dz_index_round(struct dz_index *x);
/* Asks the thread for a round soon (a search came, the last walk is old). */
void dz_indexer_poke(struct dz_index *x);

/* ---- search and tools (search.c, tools.c) ---- */

struct dz_hit {
    size_t file, chunk;
    double words, meaning; /* the two scores, 0 when not matched */
    double score;
};

/* The pieces that answer query best, at most max, under folder (NULL:
   anywhere): how many into a malloc'd *hits, -1 out of memory. The query's
   vector is made before x->mu is taken; the hits name places in x, valid
   while x->mu is held - the caller takes it after. qvec may be NULL. */
long dz_search(struct dz_index *x, const char *query, const float *qvec,
               int qdim, const char *folder, int max, struct dz_hit **hits);

struct dz_ctx {
    struct dz_roots roots;
    struct dz_embedder emb;
    struct dz_index index;
};

void dz_tools_list(void *ctx, struct janas_buf *b);
int dz_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);
/* The answers of the three tools, as text for the model, into out: 0, or
   -1 with the text an error. */
int dz_tool_search(struct dz_ctx *c, const char *query, const char *folder,
                   int max, struct janas_buf *out);
int dz_tool_read(struct dz_ctx *c, const char *path, double offset,
                 double length, struct janas_buf *out);
int dz_tool_status(struct dz_ctx *c, struct janas_buf *out);

#endif
