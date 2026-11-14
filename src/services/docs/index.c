/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * index.c - keeping the index up to date, in a thread of its own: the
 * folders walked (no hidden names, no links followed, no private file),
 * the documents new or changed since they were last read cut into pieces,
 * those gone left out; then the pieces' vectors made, a few at a time, so
 * that the words of every document are searched at once and their meaning
 * as soon as it is there. Nothing is held under the lock but the index's
 * own changes: a search waits for no document being read, nor for a
 * vector being made.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "docs.h"
#include "services/common/private.h"

#define BATCH 8         /* pieces a vector request carries */
#define SAVE_EVERY 20   /* seconds between saves while it works */
#define ROUND_EVERY 300 /* seconds between rounds when nobody asks */
#define MAX_DEPTH 16

struct found {
    char *path;
    int64_t mtime, size;
};

struct walk {
    struct found *f;
    size_t n, cap;
    const char *home;
};

static void walk_dir(struct walk *w, const char *dir, int depth)
{
    DIR *d = opendir(dir);
    if (!d)
        return;
    struct dirent *e;
    while ((e = readdir(d)) && w->n < DZ_MAX_FILES) {
        if (e->d_name[0] == '.' || !strcmp(e->d_name, "node_modules") ||
            !strcmp(e->d_name, "__pycache__"))
            continue;
        char p[PATH_MAX];
        if ((size_t)snprintf(p, sizeof p, "%s/%s", dir, e->d_name) >= sizeof p)
            continue;
        struct stat st;
        if (lstat(p, &st) != 0 || janas_private_path(w->home, p))
            continue;
        if (S_ISDIR(st.st_mode)) {
            if (depth < MAX_DEPTH)
                walk_dir(w, p, depth + 1);
            continue;
        }
        if (!S_ISREG(st.st_mode) || dz_kind(p) == DZ_NONE ||
            st.st_size > DZ_MAX_FILE)
            continue;
        if (w->n == w->cap) {
            size_t cap = w->cap ? 2 * w->cap : 256;
            struct found *t = realloc(w->f, cap * sizeof(*t));
            if (!t)
                break;
            w->f = t;
            w->cap = cap;
        }
        char *s = strdup(p);
        if (!s)
            break;
        w->f[w->n++] = (struct found){
            .path = s, .mtime = (int64_t)st.st_mtime, .size = st.st_size};
    }
    closedir(d);
}

static int by_path(const void *a, const void *b)
{
    return strcmp(((const struct found *)a)->path,
                  ((const struct found *)b)->path);
}

static void file_free(struct dz_file *f)
{
    for (uint32_t k = 0; k < f->n_chunk; k++) {
        free(f->c[k].text);
        free(f->c[k].vec);
    }
    free(f->c);
    free(f->path);
}

/* the document read and cut, its pieces without vectors: 0, 1 when it
   cannot be read (it is kept with no pieces, not read again until it
   changes), -1 out of memory */
static int read_doc(const struct found *d, struct dz_file *f, char *why,
                    size_t wn)
{
    memset(f, 0, sizeof *f);
    f->path = strdup(d->path);
    f->mtime = d->mtime;
    f->size = d->size;
    if (!f->path)
        return -1;
    struct janas_buf t = {0};
    int rc = dz_extract(d->path, &t, why, wn);
    if (rc != 0 || !t.n) {
        janas_buf_free(&t);
        return rc ? 1 : 0;
    }
    struct dz_piece *pc;
    long n = dz_cut(t.p, t.n, DZ_CHUNK, DZ_OVERLAP, &pc);
    f->c = n > 0 ? calloc((size_t)n, sizeof(*f->c)) : NULL;
    for (long k = 0; f->c && k < n; k++) {
        struct dz_chunk *c = &f->c[f->n_chunk];
        c->text = malloc(pc[k].n + 1);
        if (!c->text)
            break;
        memcpy(c->text, t.p + pc[k].at, pc[k].n);
        c->text[pc[k].n] = 0;
        c->n = (uint32_t)pc[k].n;
        f->n_chunk++;
    }
    free(pc);
    janas_buf_free(&t);
    return n < 0 || (n > 0 && f->n_chunk != (uint32_t)n) ? -1 : 0;
}

/* f put in the index in place of the one of its path (x->mu held) */
static int put_file(struct dz_index *x, struct dz_file *f)
{
    size_t at = 0;
    long i = dz_index_find(x, f->path, &at);
    if (i >= 0) {
        file_free(&x->f[i]);
        x->f[i] = *f;
        return 0;
    }
    if (x->n == x->cap) {
        size_t cap = x->cap ? 2 * x->cap : 64;
        struct dz_file *t = realloc(x->f, cap * sizeof(*t));
        if (!t)
            return -1;
        x->f = t;
        x->cap = cap;
    }
    memmove(x->f + at + 1, x->f + at, (x->n - at) * sizeof(*x->f));
    x->f[at] = *f;
    x->n++;
    return 0;
}

/* the vectors of another source dropped (x->mu held); with no source, or
   one not answering, they stay: a start without the model, or with the
   server down, does not throw away hours of work */
static void match_source(struct dz_index *x, struct dz_embedder *e)
{
    char name[sizeof e->name];
    int dim, ready = dz_embed_state(e, name, sizeof name, &dim);
    if (e->kind == DZ_EMB_NONE || !ready ||
        (!strcmp(x->source, name) && x->dim == dim))
        return;
    for (size_t i = 0; i < x->n; i++)
        for (uint32_t k = 0; k < x->f[i].n_chunk; k++) {
            free(x->f[i].c[k].vec);
            x->f[i].c[k].vec = NULL;
        }
    if (x->source[0])
        fprintf(stderr,
                "janas-docs: the vectors were made by %s, made again by %s\n",
                x->source, name);
    snprintf(x->source, sizeof x->source, "%s", name);
    x->dim = dim;
}

static size_t missing(const struct dz_index *x)
{
    size_t m = 0;
    for (size_t i = 0; i < x->n; i++)
        for (uint32_t k = 0; k < x->f[i].n_chunk; k++)
            m += !x->f[i].c[k].vec;
    return m;
}

/* the vectors missing, BATCH pieces at a time */
static void embed_missing(struct dz_index *x, time_t *saved, int *dirty)
{
    struct dz_embedder *e = x->emb;
    char *path = NULL; /* where the last batch was taken */
    uint32_t from = 0;
    for (;;) {
        char *text[BATCH];
        size_t len[BATCH], fk[BATCH];
        char *fp[BATCH];
        int n = 0;
        char name[sizeof e->name];
        int dim, ready = dz_embed_state(e, name, sizeof name, &dim);
        pthread_mutex_lock(&x->mu);
        if (x->stop || !ready || strcmp(x->source, name) || x->dim != dim) {
            pthread_mutex_unlock(&x->mu);
            break;
        }
        size_t i = 0;
        if (path) {
            long at = dz_index_find(x, path, &i);
            if (at < 0)
                from = 0;
            else
                i = (size_t)at;
        }
        for (; i < x->n && n < BATCH; i++, from = 0)
            for (uint32_t k = from; k < x->f[i].n_chunk && n < BATCH; k++) {
                if (x->f[i].c[k].vec)
                    continue;
                text[n] = strdup(x->f[i].c[k].text);
                fp[n] = strdup(x->f[i].path);
                len[n] = x->f[i].c[k].n;
                fk[n] = k;
                if (!text[n] || !fp[n]) {
                    free(text[n]);
                    free(fp[n]);
                    break;
                }
                n++;
            }
        pthread_mutex_unlock(&x->mu);
        if (!n)
            break;
        free(path);
        path = strdup(fp[n - 1]);
        from = (uint32_t)fk[n - 1] + 1;
        float *v = malloc((size_t)n * (size_t)dim * sizeof(float));
        int rc =
            v ? dz_embed(e, (const char *const *)text, len, (size_t)n, 0, v)
              : -1;
        pthread_mutex_lock(&x->mu);
        if (rc == 0 && !strcmp(x->source, name))
            for (int j = 0; j < n; j++) {
                long fi = dz_index_find(x, fp[j], NULL);
                struct dz_chunk *c = fi >= 0 && fk[j] < x->f[fi].n_chunk
                                         ? &x->f[fi].c[fk[j]]
                                         : NULL;
                /* the same piece still: the document was not read again */
                if (!c || c->vec || c->n != len[j] || strcmp(c->text, text[j]))
                    continue;
                c->vec = malloc((size_t)dim * sizeof(float));
                if (c->vec)
                    memcpy(c->vec, v + (size_t)j * (size_t)dim,
                           (size_t)dim * sizeof(float));
                x->embed_done++;
                *dirty = 1;
            }
        if (rc != 0) {
            char err[sizeof e->err];
            dz_embed_error(e, err, sizeof err);
            x->failed++;
            snprintf(x->last_fail, sizeof x->last_fail, "vectors: %s", err);
        }
        if (*dirty && time(NULL) - *saved >= SAVE_EVERY) {
            dz_index_save(x);
            *saved = time(NULL);
            *dirty = 0;
        }
        pthread_mutex_unlock(&x->mu);
        free(v);
        for (int j = 0; j < n; j++) {
            free(text[j]);
            free(fp[j]);
        }
        if (rc != 0)
            break; /* the next round tries again */
    }
    free(path);
}

void dz_index_round(struct dz_index *x)
{
    struct dz_roots roots;
    if (dz_roots_load(&roots) != 0)
        return;
    struct walk w = {.home = roots.home};
    for (int i = 0; i < roots.n_root; i++)
        walk_dir(&w, roots.root[i], 0);
    qsort(w.f, w.n, sizeof(*w.f), by_path);
    /* a folder listed twice, or inside another: its documents once */
    size_t u = 0;
    for (size_t i = 0; i < w.n; i++) {
        if (u && !strcmp(w.f[u - 1].path, w.f[i].path)) {
            free(w.f[i].path);
            continue;
        }
        w.f[u++] = w.f[i];
    }
    w.n = u;
    if (x->emb->kind == DZ_EMB_SERVER && !dz_embed_state(x->emb, NULL, 0, NULL))
        dz_embed_probe(x->emb);

    pthread_mutex_lock(&x->mu);
    *x->roots = roots;
    x->walking = 1;
    match_source(x, x->emb);
    /* the documents gone */
    size_t k = 0;
    int dirty = 0;
    for (size_t i = 0; i < x->n; i++) {
        size_t at;
        int there = 0;
        for (size_t lo = 0, hi = w.n; lo < hi && !there;) {
            at = lo + (hi - lo) / 2;
            int c = strcmp(w.f[at].path, x->f[i].path);
            there = !c;
            if (c < 0)
                lo = at + 1;
            else
                hi = at;
        }
        if (there)
            x->f[k++] = x->f[i];
        else
            file_free(&x->f[i]), dirty = 1;
    }
    x->n = k;
    /* the documents new or changed */
    size_t todo = 0;
    for (size_t i = 0; i < w.n; i++) {
        long at = dz_index_find(x, w.f[i].path, NULL);
        if (at >= 0 && x->f[at].mtime == w.f[i].mtime &&
            x->f[at].size == w.f[i].size) {
            free(w.f[i].path);
            w.f[i].path = NULL;
        } else {
            todo++;
        }
    }
    x->found = w.n;
    x->to_read = todo;
    x->read_done = 0;
    pthread_mutex_unlock(&x->mu);

    time_t saved = time(NULL);
    for (size_t i = 0; i < w.n; i++) {
        if (!w.f[i].path)
            continue;
        pthread_mutex_lock(&x->mu);
        int stop = x->stop;
        pthread_mutex_unlock(&x->mu);
        if (stop)
            break;
        struct dz_file f;
        char why[256] = "";
        int rc = read_doc(&w.f[i], &f, why, sizeof why);
        pthread_mutex_lock(&x->mu);
        if (rc == 1) {
            char s[PATH_MAX];
            dz_short(x->roots, w.f[i].path, s, sizeof s);
            x->failed++;
            snprintf(x->last_fail, sizeof x->last_fail, "%.3800s: %.255s", s,
                     why);
        }
        if (rc >= 0 && put_file(x, &f) == 0)
            dirty = 1;
        else
            file_free(&f);
        x->read_done++;
        if (dirty && time(NULL) - saved >= SAVE_EVERY) {
            dz_index_save(x);
            saved = time(NULL);
            dirty = 0;
        }
        pthread_mutex_unlock(&x->mu);
    }
    for (size_t i = 0; i < w.n; i++)
        free(w.f[i].path);
    free(w.f);

    int ready = dz_embed_state(x->emb, NULL, 0, NULL);
    pthread_mutex_lock(&x->mu);
    match_source(x, x->emb); /* the server may have come up meanwhile */
    x->to_embed = ready ? missing(x) : 0;
    x->embed_done = 0;
    pthread_mutex_unlock(&x->mu);
    if (ready)
        embed_missing(x, &saved, &dirty);

    pthread_mutex_lock(&x->mu);
    if (dirty)
        dz_index_save(x);
    x->walking = 0;
    x->last_walk = time(NULL);
    pthread_mutex_unlock(&x->mu);
}

static void *indexer(void *arg)
{
    struct dz_index *x = arg;
    for (;;) {
        dz_index_round(x);
        pthread_mutex_lock(&x->mu);
        struct timespec t;
        clock_gettime(CLOCK_REALTIME, &t);
        t.tv_sec += ROUND_EVERY;
        while (!x->stop && !x->woken)
            if (pthread_cond_timedwait(&x->wake, &x->mu, &t) != 0)
                break;
        x->woken = 0;
        int stop = x->stop;
        pthread_mutex_unlock(&x->mu);
        if (stop)
            return NULL;
    }
}

int dz_indexer_start(struct dz_index *x)
{
    if (pthread_create(&x->thread, NULL, indexer, x) != 0)
        return -1;
    x->running = 1;
    return 0;
}

void dz_indexer_poke(struct dz_index *x)
{
    pthread_mutex_lock(&x->mu);
    x->woken = 1;
    pthread_cond_signal(&x->wake);
    pthread_mutex_unlock(&x->mu);
}

void dz_indexer_stop(struct dz_index *x)
{
    if (!x->running)
        return;
    pthread_mutex_lock(&x->mu);
    x->stop = 1;
    pthread_cond_signal(&x->wake);
    pthread_mutex_unlock(&x->mu);
    pthread_join(x->thread, NULL);
    x->running = 0;
}
