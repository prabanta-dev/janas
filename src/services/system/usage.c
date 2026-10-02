/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * usage.c - what fills a directory (see system.h), on POSIX systems: each
 * entry's size, the directories' summed over what they hold, as the disk
 * counts it (the blocks taken). Within limits: SY_USAGE_MS at most, after
 * which the sizes are told as at least these; SY_USAGE_DEPTH levels; never
 * across to another filesystem, never through a symbolic link. A file of
 * more names than one (hard links) is counted at each.
 */
#ifndef _WIN32
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

#include "system.h"

struct walk {
    dev_t dev;
    double end; /* ms, monotonic */
    int cut;
    long files, unread;
};

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static uint64_t taken(const struct stat *st)
{
    return (uint64_t)st->st_blocks * 512;
}

/* the bytes under the directory fd (closed here) */
static uint64_t sum(struct walk *w, int fd, int depth, long *files)
{
    DIR *d = fdopendir(fd);
    if (!d) {
        close(fd);
        w->unread++;
        return 0;
    }
    uint64_t total = 0;
    struct dirent *e;
    unsigned tick = 0;
    while ((e = readdir(d))) {
        if ((++tick & 255) == 0 && now_ms() > w->end)
            w->cut = 1;
        if (w->cut)
            break;
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        struct stat st;
        if (fstatat(dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0)
            continue;
        total += taken(&st);
        if (!S_ISDIR(st.st_mode)) {
            (*files)++;
            continue;
        }
        if (st.st_dev != w->dev || depth >= SY_USAGE_DEPTH)
            continue;
        int sub = openat(dirfd(d), e->d_name,
                         O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (sub < 0) {
            w->unread++;
            continue;
        }
        total += sum(w, sub, depth + 1, files);
    }
    closedir(d);
    return total;
}

static int bigger(const void *a, const void *b)
{
    uint64_t x = ((const struct sy_entry *)a)->bytes,
             y = ((const struct sy_entry *)b)->bytes;
    return x < y ? 1 : x > y ? -1 : 0;
}

int sy_get_usage(const char *path, struct sy_usage *u, char *err,
                 size_t err_len)
{
    memset(u, 0, sizeof *u);
    double t0 = now_ms();
    const char *home = sy_home();
    if (!path || !*path)
        path = home;
    if (path[0] == '~' && (path[1] == '/' || !path[1]))
        snprintf(u->path, sizeof u->path, "%s%s", home, path + 1);
    else
        sy_put(u->path, sizeof u->path, path);
    int fd = open(u->path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) {
        snprintf(err, err_len, "%s cannot be read: %s", u->path,
                 strerror(errno));
        if (fd >= 0)
            close(fd);
        return -1;
    }
    struct statvfs v;
    if (fstatvfs(fd, &v) == 0) {
        u->fs_total = (uint64_t)v.f_blocks * v.f_frsize;
        u->fs_avail = (uint64_t)v.f_bavail * v.f_frsize;
    }
    struct walk w = {.dev = st.st_dev, .end = t0 + SY_USAGE_MS};
    DIR *d = fdopendir(fd);
    if (!d) {
        close(fd);
        snprintf(err, err_len, "%s cannot be read", u->path);
        return -1;
    }
    /* every entry kept to sort; the biggest SY_USAGE told */
    int cap = 256, n = 0;
    struct sy_entry *all = malloc((size_t)cap * sizeof *all);
    struct dirent *e;
    while (all && (e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        if (fstatat(dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0)
            continue;
        if (n == cap) {
            struct sy_entry *m = realloc(all, (size_t)(cap *= 2) * sizeof *m);
            if (!m)
                break;
            all = m;
        }
        struct sy_entry *x = &all[n++];
        memset(x, 0, sizeof *x);
        sy_put(x->name, sizeof x->name, e->d_name);
        x->bytes = taken(&st);
        x->dir = S_ISDIR(st.st_mode);
        if (!x->dir) {
            x->files = 1;
            continue;
        }
        if (st.st_dev != w.dev || w.cut)
            continue;
        int sub = openat(dirfd(d), e->d_name,
                         O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (sub < 0) {
            w.unread++;
            continue;
        }
        x->bytes += sum(&w, sub, 1, &x->files);
    }
    closedir(d);
    if (!all) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    qsort(all, (size_t)n, sizeof *all, bigger);
    for (int i = 0; i < n; i++) {
        u->total += all[i].bytes;
        u->files += all[i].files;
    }
    u->n = n < SY_USAGE ? n : SY_USAGE;
    u->more = n - u->n;
    memcpy(u->e, all, (size_t)u->n * sizeof *all);
    free(all);
    u->cut = w.cut;
    u->unread = w.unread;
    u->secs = (now_ms() - t0) / 1e3;
    return 0;
}
#endif
