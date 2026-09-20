/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * store.h - what the server keeps for its clients (store.c): in memory,
 * and in a directory when it has one, so that it outlives the server.
 * Documents are JSON text found by their kind ("chat.completion",
 * "response", "conversation"...) and id; every call copies, so what a
 * caller gets is its own to free, and the store may be used from any
 * thread.
 */
#ifndef JANAS_SERVER_STORE_H
#define JANAS_SERVER_STORE_H

#include <stddef.h>
#include <stdint.h>

struct srv_store;

/* cap: documents of each kind at most, the oldest dropped (0: 1000). In
   memory only. */
struct srv_store *srv_store_new(size_t cap);
/* The same, kept in dir as well (NULL: memory only), with what an earlier
   run left there read back. NULL with the reason in err: another server
   holds the directory, or it cannot be made. */
struct srv_store *srv_store_open(size_t cap, const char *dir, char *err,
                                 size_t err_len);
void srv_store_free(struct srv_store *st);
/* Documents of every kind. */
size_t srv_store_count(struct srv_store *st);

/* Keeps a copy (replacing one of the same kind and id); aux may be NULL.
   0, or -1 on memory. */
int srv_store_put(struct srv_store *st, const char *kind, const char *id,
                  int64_t created, const char *json, const char *aux);
/* Copies of the document (json, aux: either may be NULL, and *aux is NULL
   when there is none): 0, -1 not found, -2 memory. */
int srv_store_get(struct srv_store *st, const char *kind, const char *id,
                  char **json, char **aux);
/* New texts for a document (NULL: unchanged): 0, -1 not found, -2 memory. */
int srv_store_update(struct srv_store *st, const char *kind, const char *id,
                     const char *json, const char *aux);
/* 0, or -1 not found. */
int srv_store_del(struct srv_store *st, const char *kind, const char *id);
/* The ids of a kind, oldest first: how many, into *ids (free with
   srv_store_list_free). */
size_t srv_store_list(struct srv_store *st, const char *kind, char ***ids);
void srv_store_list_free(char **ids, size_t n);

#endif
