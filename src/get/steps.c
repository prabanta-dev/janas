/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * steps.c - janas-get's steps (see steps.h). The converters are the
 * programs of the same build, run as they are: what they write is what
 * MODELS.md fingerprints, whoever runs them.
 */
#define _GNU_SOURCE
#include "steps.h"

#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "common/hf_fetch.h"
#include "common/sha256.h"

extern char **environ;

void get_local(char *out, size_t len, const char *dir, const char *path)
{
    const char *base = strrchr(path, '/');
    snprintf(out, len, "%s/%s", dir, base ? base + 1 : path);
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

struct bar {
    double t0;
    uint64_t from;
};

static void progress(uint64_t done, uint64_t size, void *arg)
{
    struct bar *b = arg;
    double dt = now() - b->t0;
    double mbs = dt > 0 ? (double)(done - b->from) / dt / 1e6 : 0;
    fprintf(stderr, "\r   %.2f of %.2f GB (%.0f%%), %.0f MB/s    ",
            (double)done / 1e9, (double)size / 1e9,
            size ? 100.0 * (double)done / (double)size : 100.0, mbs);
    if (done >= size)
        fputc('\n', stderr);
}

int get_same(const char *file, const char *want)
{
    char got[65];
    if (janas_sha256_file(file, got) != 0)
        return -1;
    return strcmp(got, want) == 0;
}

int get_fetch(const char *repo, const char *path, const char *dir,
              const char *want)
{
    char local[4096], err[512], listed[65], got[65];
    get_local(local, sizeof(local), dir, path);
    uint64_t size = 0;
    if (janas_hf_info(repo, NULL, path, &size, listed, err, sizeof(err)) != 0) {
        fprintf(stderr, "janas-get: %s\n", err);
        return -1;
    }
    /* the repository may have been changed since MODELS.md was written:
       then its file is not the one the fingerprints are of */
    if (want && listed[0] && strcmp(listed, want) != 0) {
        fprintf(stderr,
                "janas-get: %s has changed on Hugging Face since MODELS.md "
                "was written (SHA-256 %s, not %s); not downloaded\n",
                path, listed, want);
        return -1;
    }
    printf("   %s from %s, %.2f GB\n", path, repo, (double)size / 1e9);
    fflush(stdout);
    struct bar b = {.t0 = now()};
    FILE *f = fopen(local, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long at = ftell(f);
        b.from = at > 0 ? (uint64_t)at : 0;
        fclose(f);
    }
    if (janas_hf_download(repo, NULL, path, local, size, got, progress, &b, err,
                          sizeof(err)) != 0) {
        fprintf(stderr, "\njanas-get: %s (run again to resume)\n", err);
        return -1;
    }
    const char *expect = want ? want : listed;
    if (expect[0] && strcmp(got, expect) != 0) {
        fprintf(stderr,
                "janas-get: %s is not the file it should be (SHA-256 %s, "
                "not %s); deleted\n",
                local, got, expect);
        remove(local);
        return -1;
    }
    printf("   SHA-256 checked%s\n", want ? " against MODELS.md" : "");
    return 0;
}

int get_run(const char *tool, char *const argv[])
{
    char self[4096], path[4200];
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0)
        return -1;
    self[n] = 0;
    char *slash = strrchr(self, '/');
    if (slash)
        *slash = 0;
    snprintf(path, sizeof(path), "%s/%s", self, tool);
    pid_t pid;
    if (posix_spawn(&pid, path, NULL, NULL, argv, environ) != 0) {
        fprintf(stderr, "janas-get: cannot run %s\n", path);
        return -1;
    }
    int st;
    if (waitpid(pid, &st, 0) != pid)
        return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}
