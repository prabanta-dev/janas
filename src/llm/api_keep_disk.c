/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * api_keep_disk.c - the conversations kept computed, on disk below the
 * memory (janas_llm_chat_keep_disk in include/janas/llm.h), so that they
 * outlive the program: a client's long system message and tools, read once
 * and never again.
 *
 * One file a copy, in ~/.cache/janas/keep-<model's fingerprint>/, named by
 * a hash of its tokens. A file is written when a copy leaves the memory to
 * make room, and when the chat ends (what it kept, and its conversation),
 * never at every turn: a copy is hundreds of megabytes, and a disk wears
 * with what is written to it. When a file is chosen it is read whole and
 * becomes a copy in memory like the others. At the start only the tokens
 * of each file are read, to choose among them. The least recently used
 * files go past the room given; a file that another holds whole goes too,
 * as in memory.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include "llm/jns.h"
#include "llm/chat.h"

/* shorter copies are not worth a file: reading them again is quick */
#define DISK_MIN 1024

static int path_of(const janas_llm_chat *c, const char *name, char *buf,
                   size_t len)
{
    int n = snprintf(buf, len, "%s/%s", c->disk_dir, name);
    return n > 0 && (size_t)n < len ? 0 : -1;
}

static int64_t mtime_ns(const struct stat *st)
{
    return (int64_t)st->st_mtim.tv_sec * 1000000000 + st->st_mtim.tv_nsec;
}

static void forget(janas_llm_chat *c, int32_t i, int unlink_it)
{
    char p[4096];
    if (unlink_it && path_of(c, c->disk[i].name, p, sizeof(p)) == 0)
        unlink(p);
    c->disk_bytes -= c->disk[i].bytes;
    free(c->disk[i].tok);
    c->disk[i] = c->disk[--c->n_disk];
}

static void disk_evict(janas_llm_chat *c)
{
    while (c->n_disk > 0 && c->disk_bytes > c->disk_max) {
        int32_t old = 0;
        for (int32_t i = 1; i < c->n_disk; i++)
            if (c->disk[i].used < c->disk[old].used)
                old = i;
        forget(c, old, 1);
    }
}

/* An entry for the file of sv, its tokens copied. */
static int index_add(janas_llm_chat *c, const char *name,
                     const struct janas_llm_saved *sv, uint64_t bytes,
                     int64_t used, int prefix)
{
    if (c->n_disk == c->cap_disk) {
        int32_t cap = c->cap_disk ? 2 * c->cap_disk : 16;
        struct kept_file *t = realloc(c->disk, (size_t)cap * sizeof(*t));
        if (!t)
            return -1;
        c->disk = t;
        c->cap_disk = cap;
    }
    const int32_t *tok;
    uint32_t n_kv, n = janas_llm_saved_tokens(sv, &tok, &n_kv);
    struct kept_file *e = &c->disk[c->n_disk];
    memset(e, 0, sizeof(*e));
    if (!(e->tok = malloc(n * sizeof(int32_t))))
        return -1;
    memcpy(e->tok, tok, n * sizeof(int32_t));
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->n = n;
    e->n_kv = n_kv;
    e->bytes = bytes;
    e->used = used;
    e->prefix = prefix;
    c->n_disk++;
    c->disk_bytes += bytes;
    return 0;
}

static int is_prefix_name(const char *name)
{
    size_t n = strlen(name);
    return n > 6 && !strcmp(name + n - 6, ".p.jkv");
}

void janas_api_disk_close(janas_llm_chat *c)
{
    while (c->n_disk > 0)
        forget(c, c->n_disk - 1, 0);
    free(c->disk);
    c->disk = NULL;
    c->cap_disk = 0;
    c->disk_bytes = 0;
    free(c->disk_dir);
    c->disk_dir = NULL;
}

int32_t janas_llm_chat_keep_disk(janas_llm_chat *c, uint64_t bytes)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    janas_api_disk_close(c);
    c->disk_max = 0;
    if (bytes == 0)
        return JANAS_LLM_OK;
    char dir[4096];
    if (janas_llm_model_cache_file(c->llm->m, "keep", dir, sizeof(dir), 1) != 0)
        return janas_api_fail(JANAS_LLM_EFAIL, "no cache directory");
    size_t dn = strlen(dir);
    if (dn > 4 && !strcmp(dir + dn - 4, ".bin"))
        dir[dn - 4] = 0; /* a directory, not the file the name was for */
    if (mkdir(dir, 0700) != 0 && errno != EEXIST)
        return janas_api_fail(JANAS_LLM_EFAIL, "cannot make %s", dir);
    if (!(c->disk_dir = strdup(dir)))
        return janas_api_fail(JANAS_LLM_ENOMEM, "out of memory");
    /* never more than a quarter of the space left on that disk */
    struct statvfs vf;
    uint64_t room = bytes;
    if (statvfs(dir, &vf) == 0) {
        uint64_t free4 = (uint64_t)vf.f_bavail * vf.f_frsize / 4;
        if (room > free4)
            room = free4;
    }
    c->disk_max = room;
    DIR *d = opendir(dir);
    struct dirent *e;
    while (d && (e = readdir(d))) {
        size_t n = strlen(e->d_name);
        char p[4096];
        if (n < 5 || strcmp(e->d_name + n - 4, ".jkv") ||
            path_of(c, e->d_name, p, sizeof(p)) != 0)
            continue;
        if (e->d_name[0] == '.') { /* a write that was not finished */
            unlink(p);
            continue;
        }
        FILE *f = fopen(p, "rb");
        struct stat st;
        struct janas_llm_saved *sv = f && fstat(fileno(f), &st) == 0
                                         ? janas_llm_saved_read(c->s, f, 1)
                                         : NULL;
        if (f)
            fclose(f);
        if (!sv) {
            unlink(p); /* not a copy of this model's: of no use */
            continue;
        }
        index_add(c, e->d_name, sv, (uint64_t)st.st_size, mtime_ns(&st),
                  is_prefix_name(e->d_name));
        janas_llm_saved_free(sv);
    }
    if (d)
        closedir(d);
    disk_evict(c);
    return JANAS_LLM_OK;
}

int32_t janas_llm_chat_kept_disk(const janas_llm_chat *c, int32_t *n,
                                 uint64_t *bytes)
{
    if (!c)
        return janas_api_fail(JANAS_LLM_EINVAL, "no chat");
    if (n)
        *n = c->n_disk;
    if (bytes)
        *bytes = c->disk_bytes;
    return JANAS_LLM_OK;
}

int32_t janas_api_disk_best(janas_llm_chat *c, uint32_t floor, uint32_t *r)
{
    int rec = janas_llm_model_recurrent(c->llm->m);
    int32_t best = -1;
    *r = floor;
    for (int32_t i = 0; i < c->n_disk; i++) {
        const struct kept_file *e = &c->disk[i];
        uint32_t u =
            janas_api_keep_reuse(e->tok, e->n, e->n_kv, c->ids, c->n_ids, rec);
        if (u > *r) {
            best = i;
            *r = u;
        }
    }
    return best;
}

struct janas_llm_saved *janas_api_disk_take(janas_llm_chat *c, int32_t i,
                                            int *prefix)
{
    char p[4096];
    if (path_of(c, c->disk[i].name, p, sizeof(p)) != 0)
        return NULL;
    FILE *f = fopen(p, "rb");
    struct janas_llm_saved *sv = f ? janas_llm_saved_read(c->s, f, 0) : NULL;
    if (f)
        fclose(f);
    if (!sv) {
        /* written with other experts in use, or damaged: of no use */
        forget(c, i, 1);
        return NULL;
    }
    *prefix = c->disk[i].prefix;
    /* used now: the last to go */
    struct stat st;
    if (utimensat(AT_FDCWD, p, NULL, 0) == 0 && stat(p, &st) == 0)
        c->disk[i].used = mtime_ns(&st);
    return sv;
}

void janas_api_disk_put(janas_llm_chat *c, const struct janas_llm_saved *sv,
                        int prefix)
{
    const int32_t *tok;
    uint32_t n_kv, n = janas_llm_saved_tokens(sv, &tok, &n_kv);
    if (!c->disk_dir || n_kv < DISK_MIN)
        return;
    uint64_t h = janas_fnv1a(tok, n * sizeof(int32_t), JANAS_FNV_INIT);
    h = janas_fnv1a(&n_kv, sizeof(n_kv), h);
    char name[40], p[4096], tmp[4096];
    snprintf(name, sizeof(name), "%016llx%s.jkv", (unsigned long long)h,
             prefix ? ".p" : "");
    for (int32_t i = 0; i < c->n_disk; i++)
        if (!strcmp(c->disk[i].name, name))
            return; /* there already */
    if (path_of(c, name, p, sizeof(p)) != 0 ||
        snprintf(tmp, sizeof(tmp), "%s/.%s", c->disk_dir, name) >=
            (int)sizeof(tmp))
        return;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
    if (!f) {
        if (fd >= 0)
            close(fd);
        return;
    }
    int bad = janas_llm_saved_write(c->s, sv, f) != 0;
    bad |= fclose(f) != 0;
    struct stat st;
    if (bad || rename(tmp, p) != 0 || stat(p, &st) != 0) {
        unlink(tmp);
        return;
    }
    /* what this file holds whole goes, as in memory (not the system
       messages of a recurrent model, which conversations start from) */
    for (int32_t i = c->n_disk - 1; i >= 0; i--) {
        const struct kept_file *e = &c->disk[i];
        if ((!e->prefix || e->n_kv == n_kv) && e->n_kv <= n_kv && e->n <= n &&
            memcmp(e->tok, tok, e->n_kv * sizeof(int32_t)) == 0)
            forget(c, i, 1);
    }
    if (index_add(c, name, sv, (uint64_t)st.st_size, mtime_ns(&st), prefix) !=
        0)
        unlink(p);
    disk_evict(c);
}
