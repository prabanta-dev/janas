/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * store_disk.c - the store's documents on disk: <dir>/<kind>/<id>, a line
 * of header then the two texts,
 *
 *     JANAS-STORE 1 <seq> <created> <json bytes> <aux bytes, or -1>\n
 *     <json><aux>
 *
 * written to a hidden file beside it and renamed over it, so that a
 * document is there whole or not at all. The directory and the files are
 * the user's alone (0700, 0600): they hold what the clients said. A lock
 * on <dir>/.lock keeps a second server out: two servers writing the same
 * documents would each lose what the other did.
 */
#define _GNU_SOURCE
#include "store_disk.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

struct srv_disk {
    char *dir;
    int lock;
};

int srv_disk_name_ok(const char *s)
{
    size_t n = 0;
    if (!s || s[0] == '.' || !s[0])
        return 0;
    for (; s[n]; n++) {
        char c = s[n];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
            return 0;
    }
    return n <= 128;
}

#define PATH_CAP 4096

/* dir/pre name suf into out: 0, or -1 when it does not fit */
static int join(char *out, const char *dir, const char *pre, const char *name,
                const char *suf)
{
    size_t a = strlen(dir), b = strlen(pre), c = strlen(name), d = strlen(suf);
    if (a + 1 + b + c + d + 1 > PATH_CAP)
        return -1;
    memcpy(out, dir, a);
    out[a] = '/';
    memcpy(out + a + 1, pre, b);
    memcpy(out + a + 1 + b, name, c);
    memcpy(out + a + 1 + b + c, suf, d + 1);
    return 0;
}

/* mkdir -p, the levels it makes private */
static int make_dirs(const char *path)
{
    char *p = strdup(path);
    if (!p)
        return -1;
    for (char *s = p + 1; *s; s++)
        if (*s == '/') {
            *s = 0;
            if (mkdir(p, 0700) != 0 && errno != EEXIST) {
                free(p);
                return -1;
            }
            *s = '/';
        }
    int rc = mkdir(p, 0700) != 0 && errno != EEXIST ? -1 : 0;
    free(p);
    return rc;
}

struct srv_disk *srv_disk_open(const char *dir, char *err, size_t err_len)
{
    if (make_dirs(dir) != 0) {
        snprintf(err, err_len, "cannot make %s: %s", dir, strerror(errno));
        return NULL;
    }
    char path[PATH_CAP];
    if (strlen(dir) > PATH_CAP / 2 || join(path, dir, "", ".lock", "") != 0) {
        snprintf(err, err_len, "the store's directory has too long a name");
        return NULL;
    }
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) {
        snprintf(err, err_len, "cannot open %s: %s", path, strerror(errno));
        return NULL;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        snprintf(err, err_len,
                 "another janas-server keeps its documents in %s (--store "
                 "DIR for another place, --no-store for none)",
                 dir);
        close(fd);
        return NULL;
    }
    struct srv_disk *d = calloc(1, sizeof(*d));
    if (!d || !(d->dir = strdup(dir))) {
        free(d);
        close(fd);
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    d->lock = fd;
    return d;
}

void srv_disk_close(struct srv_disk *d)
{
    if (!d)
        return;
    close(d->lock); /* and the lock with it */
    free(d->dir);
    free(d);
}

static int put_all(FILE *f, const char *s, size_t n)
{
    return n == 0 || fwrite(s, 1, n, f) == n ? 0 : -1;
}

int srv_disk_write(struct srv_disk *d, const char *kind, const char *id,
                   uint64_t seq, int64_t created, const char *json,
                   const char *aux)
{
    if (!srv_disk_name_ok(kind) || !srv_disk_name_ok(id))
        return -1;
    char sub[PATH_CAP], path[PATH_CAP], tmp[PATH_CAP];
    if (join(sub, d->dir, "", kind, "") != 0 ||
        join(path, sub, "", id, "") != 0 || join(tmp, sub, ".", id, ".tmp"))
        return -1;
    if (mkdir(sub, 0700) != 0 && errno != EEXIST)
        return -1;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
    if (!f) {
        if (fd >= 0)
            close(fd);
        return -1;
    }
    size_t nj = strlen(json), na = aux ? strlen(aux) : 0;
    int bad = fprintf(f, "JANAS-STORE 1 %" PRIu64 " %" PRId64 " %zu %lld\n",
                      seq, created, nj, aux ? (long long)na : -1LL) < 0;
    bad |= put_all(f, json, nj) || (aux && put_all(f, aux, na));
    bad |= fclose(f) != 0;
    if (bad || rename(tmp, path) != 0) {
        unlink(tmp);
        return -1;
    }
    return 0;
}

void srv_disk_remove(struct srv_disk *d, const char *kind, const char *id)
{
    if (!srv_disk_name_ok(kind) || !srv_disk_name_ok(id))
        return;
    char sub[PATH_CAP], path[PATH_CAP];
    if (join(sub, d->dir, "", kind, "") == 0 &&
        join(path, sub, "", id, "") == 0)
        unlink(path);
}

static char *read_n(FILE *f, long long n)
{
    char *s = malloc((size_t)n + 1);
    if (s && n > 0 && fread(s, 1, (size_t)n, f) != (size_t)n) {
        free(s);
        return NULL;
    }
    if (s)
        s[n] = 0;
    return s;
}

/* One document: 0 read and handed over, 1 unreadable, -1 fn said stop. */
static int load_one(const char *path, const char *kind, const char *id,
                    srv_disk_fn fn, void *arg)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return 1;
    uint64_t seq;
    int64_t created;
    long long nj, na;
    char *json = NULL, *aux = NULL;
    int ok = fscanf(f, "JANAS-STORE 1 %" SCNu64 " %" SCNd64 " %lld %lld", &seq,
                    &created, &nj, &na) == 4 &&
             fgetc(f) == '\n' && nj >= 0 && na >= -1 &&
             (json = read_n(f, nj)) && (na < 0 || (aux = read_n(f, na))) &&
             fgetc(f) == EOF && strlen(json) == (size_t)nj &&
             (!aux || strlen(aux) == (size_t)na);
    fclose(f);
    if (!ok) {
        free(json);
        free(aux);
        return 1;
    }
    return fn(arg, kind, id, seq, created, json, aux) ? -1 : 0;
}

int srv_disk_load(struct srv_disk *d, srv_disk_fn fn, void *arg)
{
    int skipped = 0, stop = 0;
    DIR *top = opendir(d->dir);
    struct dirent *k;
    while (top && !stop && (k = readdir(top))) {
        if (!srv_disk_name_ok(k->d_name))
            continue;
        char sub[PATH_CAP];
        if (join(sub, d->dir, "", k->d_name, "") != 0)
            continue;
        DIR *dd = opendir(sub);
        struct dirent *e;
        while (dd && !stop && (e = readdir(dd))) {
            char path[PATH_CAP];
            if (join(path, sub, "", e->d_name, "") != 0)
                continue;
            size_t n = strlen(e->d_name);
            if (e->d_name[0] == '.') {
                /* a write the server did not finish */
                if (n > 4 && !strcmp(e->d_name + n - 4, ".tmp"))
                    unlink(path);
                continue;
            }
            if (!srv_disk_name_ok(e->d_name))
                continue;
            int r = load_one(path, k->d_name, e->d_name, fn, arg);
            skipped += r == 1;
            stop = r < 0;
        }
        if (dd)
            closedir(dd);
    }
    if (top)
        closedir(top);
    return skipped;
}
