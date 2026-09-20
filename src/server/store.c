/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * store.c - what the server keeps for its clients: completions asked to be
 * stored, responses, conversations. In memory, in the order it was made,
 * each kind up to a number of documents past which the oldest go; a
 * document is JSON text, with a second text beside it for what goes with it
 * (the messages of a stored completion). With a directory, every document
 * is also written there as it changes (store_disk.c) and read back when
 * the server starts again; the memory is still where it is read from.
 */
#include "store.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "store_disk.h"

struct entry {
    char *kind, *id, *json, *aux;
    int64_t created;
    uint64_t seq; /* the order it was made in, across restarts */
    struct entry *prev, *next;
};

struct srv_store {
    pthread_mutex_t mu;
    struct entry *head, *tail; /* oldest first */
    size_t cap;
    struct srv_disk *disk; /* or NULL: memory only */
    uint64_t seq;
    int warned; /* a write failed, and it was said */
};

static void trim(struct srv_store *st, const char *kind);

static char *str_dup(const char *s)
{
    if (!s)
        return NULL;
    size_t n = strlen(s);
    char *d = malloc(n + 1);
    if (d)
        memcpy(d, s, n + 1);
    return d;
}

struct srv_store *srv_store_new(size_t cap)
{
    struct srv_store *st = calloc(1, sizeof(*st));
    if (!st)
        return NULL;
    pthread_mutex_init(&st->mu, NULL);
    st->cap = cap ? cap : 1000;
    return st;
}

/* Under st->mu. A document that cannot be written stays in memory; the
   first failure is said, not every one. */
static void save(struct srv_store *st, const struct entry *e)
{
    if (!st->disk || !srv_disk_name_ok(e->kind) || !srv_disk_name_ok(e->id))
        return;
    if (srv_disk_write(st->disk, e->kind, e->id, e->seq, e->created, e->json,
                       e->aux) != 0 &&
        !st->warned) {
        st->warned = 1;
        fprintf(stderr,
                "janas-server: cannot write %s %s to the store's "
                "directory: kept in memory only\n",
                e->kind, e->id);
    }
}

static void unsave(struct srv_store *st, const struct entry *e)
{
    if (st->disk)
        srv_disk_remove(st->disk, e->kind, e->id);
}

static void entry_free(struct entry *e)
{
    free(e->kind);
    free(e->id);
    free(e->json);
    free(e->aux);
    free(e);
}

static void unlink_entry(struct srv_store *st, struct entry *e)
{
    if (e->prev)
        e->prev->next = e->next;
    else
        st->head = e->next;
    if (e->next)
        e->next->prev = e->prev;
    else
        st->tail = e->prev;
}

void srv_store_free(struct srv_store *st)
{
    if (!st)
        return;
    for (struct entry *e = st->head; e;) {
        struct entry *nx = e->next;
        entry_free(e);
        e = nx;
    }
    srv_disk_close(st->disk);
    pthread_mutex_destroy(&st->mu);
    free(st);
}

/* the documents read back, before they are put in order */
struct loaded {
    struct entry **e;
    size_t n, cap;
};

static int take(void *arg, const char *kind, const char *id, uint64_t seq,
                int64_t created, char *json, char *aux)
{
    struct loaded *l = arg;
    struct entry *e = calloc(1, sizeof(*e));
    if (l->n == l->cap) {
        size_t cap = l->cap ? 2 * l->cap : 256;
        struct entry **t = realloc(l->e, cap * sizeof(*t));
        if (!t) {
            free(e);
            e = NULL;
        } else {
            l->e = t;
            l->cap = cap;
        }
    }
    if (!e || !(e->kind = str_dup(kind)) || !(e->id = str_dup(id))) {
        if (e)
            free(e->kind);
        free(e);
        free(json);
        free(aux);
        return 1;
    }
    e->json = json;
    e->aux = aux;
    e->seq = seq;
    e->created = created;
    l->e[l->n++] = e;
    return 0;
}

static int by_seq(const void *a, const void *b)
{
    const struct entry *x = *(struct entry *const *)a,
                       *y = *(struct entry *const *)b;
    return x->seq < y->seq ? -1 : x->seq > y->seq;
}

struct srv_store *srv_store_open(size_t cap, const char *dir, char *err,
                                 size_t err_len)
{
    struct srv_store *st = srv_store_new(cap);
    if (!st) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    if (!dir)
        return st;
    if (!(st->disk = srv_disk_open(dir, err, err_len))) {
        srv_store_free(st);
        return NULL;
    }
    struct loaded l = {0};
    int skipped = srv_disk_load(st->disk, take, &l);
    qsort(l.e, l.n, sizeof(*l.e), by_seq);
    for (size_t i = 0; i < l.n; i++) {
        struct entry *e = l.e[i];
        e->prev = st->tail;
        if (st->tail)
            st->tail->next = e;
        else
            st->head = e;
        st->tail = e;
        if (e->seq > st->seq)
            st->seq = e->seq;
    }
    /* a smaller cap than the last run's: the oldest go now, kind by kind
       (the server has a handful of kinds) */
    char *kinds[32];
    int nk = 0;
    for (size_t i = 0; i < l.n && nk < 32; i++) {
        int seen = 0;
        for (int k = 0; k < nk && !seen; k++)
            seen = !strcmp(kinds[k], l.e[i]->kind);
        if (!seen && (kinds[nk] = str_dup(l.e[i]->kind)))
            nk++;
    }
    free(l.e);
    for (int k = 0; k < nk; k++) {
        trim(st, kinds[k]);
        free(kinds[k]);
    }
    if (skipped)
        fprintf(stderr,
                "janas-server: %d unreadable files in %s left as they are\n",
                skipped, dir);
    return st;
}

size_t srv_store_count(struct srv_store *st)
{
    pthread_mutex_lock(&st->mu);
    size_t n = 0;
    for (struct entry *e = st->head; e; e = e->next)
        n++;
    pthread_mutex_unlock(&st->mu);
    return n;
}

static void unlink_entry(struct srv_store *st, struct entry *e);
static void entry_free(struct entry *e);

/* Past the cap, the oldest of the kind go. Under st->mu. */
static void trim(struct srv_store *st, const char *kind)
{
    size_t count = 0;
    for (struct entry *x = st->head; x; x = x->next)
        count += !strcmp(x->kind, kind);
    for (struct entry *x = st->head; x && count > st->cap;) {
        struct entry *nx = x->next;
        if (!strcmp(x->kind, kind)) {
            unlink_entry(st, x);
            unsave(st, x);
            entry_free(x);
            count--;
        }
        x = nx;
    }
}

static struct entry *find(struct srv_store *st, const char *kind,
                          const char *id)
{
    for (struct entry *e = st->head; e; e = e->next)
        if (!strcmp(e->id, id) && !strcmp(e->kind, kind))
            return e;
    return NULL;
}

int srv_store_put(struct srv_store *st, const char *kind, const char *id,
                  int64_t created, const char *json, const char *aux)
{
    struct entry *e = calloc(1, sizeof(*e));
    if (!e)
        return -1;
    e->kind = str_dup(kind);
    e->id = str_dup(id);
    e->json = str_dup(json);
    e->aux = str_dup(aux);
    e->created = created;
    if (!e->kind || !e->id || !e->json || (aux && !e->aux)) {
        entry_free(e);
        return -1;
    }
    pthread_mutex_lock(&st->mu);
    struct entry *old = find(st, kind, id);
    if (old) {
        unlink_entry(st, old);
        entry_free(old);
    }
    e->seq = ++st->seq;
    e->prev = st->tail;
    if (st->tail)
        st->tail->next = e;
    else
        st->head = e;
    st->tail = e;
    save(st, e);
    trim(st, kind);
    pthread_mutex_unlock(&st->mu);
    return 0;
}

int srv_store_get(struct srv_store *st, const char *kind, const char *id,
                  char **json, char **aux)
{
    pthread_mutex_lock(&st->mu);
    struct entry *e = find(st, kind, id);
    int rc = -1;
    if (e) {
        char *j = json ? str_dup(e->json) : NULL;
        char *a = aux ? str_dup(e->aux) : NULL;
        if ((!json || j) && (!aux || a || !e->aux)) {
            if (json)
                *json = j;
            if (aux)
                *aux = a;
            rc = 0;
        } else {
            free(j);
            free(a);
            rc = -2;
        }
    }
    pthread_mutex_unlock(&st->mu);
    return rc;
}

int srv_store_update(struct srv_store *st, const char *kind, const char *id,
                     const char *json, const char *aux)
{
    char *j = str_dup(json), *a = str_dup(aux);
    if ((json && !j) || (aux && !a)) {
        free(j);
        free(a);
        return -2;
    }
    pthread_mutex_lock(&st->mu);
    struct entry *e = find(st, kind, id);
    if (e) {
        if (j) {
            free(e->json);
            e->json = j;
            j = NULL;
        }
        if (a) {
            free(e->aux);
            e->aux = a;
            a = NULL;
        }
        save(st, e);
    }
    pthread_mutex_unlock(&st->mu);
    free(j);
    free(a);
    return e ? 0 : -1;
}

int srv_store_del(struct srv_store *st, const char *kind, const char *id)
{
    pthread_mutex_lock(&st->mu);
    struct entry *e = find(st, kind, id);
    if (e) {
        unlink_entry(st, e);
        unsave(st, e);
        entry_free(e);
    }
    pthread_mutex_unlock(&st->mu);
    return e ? 0 : -1;
}

size_t srv_store_list(struct srv_store *st, const char *kind, char ***ids)
{
    *ids = NULL;
    pthread_mutex_lock(&st->mu);
    size_t n = 0;
    for (struct entry *e = st->head; e; e = e->next)
        n += !strcmp(e->kind, kind);
    char **out = n ? calloc(n, sizeof(char *)) : NULL;
    size_t w = 0;
    for (struct entry *e = st->head; out && e; e = e->next)
        if (!strcmp(e->kind, kind) && (out[w] = str_dup(e->id)))
            w++;
    pthread_mutex_unlock(&st->mu);
    *ids = out;
    return w;
}

void srv_store_list_free(char **ids, size_t n)
{
    for (size_t i = 0; i < n; i++)
        free(ids[i]);
    free(ids);
}
