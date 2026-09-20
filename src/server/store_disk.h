/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * store_disk.h - the store's documents on disk (store_disk.c), one file
 * each, so that they outlive the server. Used by store.c only.
 */
#ifndef JANAS_SERVER_STORE_DISK_H
#define JANAS_SERVER_STORE_DISK_H

#include <stddef.h>
#include <stdint.h>

struct srv_disk;

/* The directory, made if missing (0700), and locked for this process:
   NULL with a reason in err when another server holds it, or on failure. */
struct srv_disk *srv_disk_open(const char *dir, char *err, size_t err_len);
void srv_disk_close(struct srv_disk *d);

/* Whether kind and id make a file name (letters, digits, '.', '_', '-',
   not starting with '.'): a document whose names do not stays in memory. */
int srv_disk_name_ok(const char *s);

/* Writes the document (aux may be NULL), replacing what was there: 0 or -1. */
int srv_disk_write(struct srv_disk *d, const char *kind, const char *id,
                   uint64_t seq, int64_t created, const char *json,
                   const char *aux);
void srv_disk_remove(struct srv_disk *d, const char *kind, const char *id);

/* Calls fn for every document found, in no order; fn takes json and aux
   (aux may be NULL), and returns non-zero to stop. Files that cannot be
   read are skipped, and counted in the return value. */
typedef int (*srv_disk_fn)(void *arg, const char *kind, const char *id,
                           uint64_t seq, int64_t created, char *json,
                           char *aux);
int srv_disk_load(struct srv_disk *d, srv_disk_fn fn, void *arg);

#endif
