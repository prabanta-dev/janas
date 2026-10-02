/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * run.c - a program run by a service (see run.h): no shell, no terminal,
 * a time limit for it and what it started.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "run.h"

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

int janas_run(const char *dir, const char *const *argv, const char *const *env,
              int timeout_ms, struct janas_buf *out, struct janas_buf *err,
              char *why, size_t why_len)
{
    const char *name = strrchr(argv[0], '/');
    name = name ? name + 1 : argv[0];
    int po[2], pe[2];
    if (pipe2(po, O_CLOEXEC) != 0) {
        snprintf(why, why_len, "no pipe: %s", strerror(errno));
        return -1;
    }
    if (pipe2(pe, O_CLOEXEC) != 0) {
        close(po[0]);
        close(po[1]);
        snprintf(why, why_len, "no pipe: %s", strerror(errno));
        return -1;
    }
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(why, why_len, "cannot start %s: %s", name, strerror(errno));
        close(po[0]), close(po[1]), close(pe[0]), close(pe[1]);
        return -1;
    }
    if (pid == 0) {
        setsid(); /* no terminal: nothing can be asked of the user */
        int nul = open("/dev/null", O_RDONLY);
        if (nul >= 0)
            dup2(nul, 0);
        dup2(po[1], 1);
        dup2(pe[1], 2);
        if (dir && chdir(dir) != 0)
            _exit(126);
        for (size_t i = 0; env && env[i]; i++)
            putenv((char *)env[i]);
        execvp(argv[0], (char *const *)argv);
        _exit(127);
    }
    close(po[1]);
    close(pe[1]);
    struct pollfd fd[2] = {{.fd = po[0], .events = POLLIN},
                           {.fd = pe[0], .events = POLLIN}};
    struct janas_buf *to[2] = {out, err};
    double end = now_ms() + timeout_ms;
    int open_n = 2, late = 0;
    while (open_n) {
        int left = (int)(end - now_ms());
        if (left <= 0) {
            late = 1;
            break;
        }
        if (poll(fd, 2, left) < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        for (int i = 0; i < 2; i++) {
            if (fd[i].fd < 0 || !(fd[i].revents & (POLLIN | POLLHUP | POLLERR)))
                continue;
            char chunk[8192];
            ssize_t r = read(fd[i].fd, chunk, sizeof chunk);
            if (r > 0 && to[i]->n < JANAS_RUN_OUT_MAX)
                janas_buf_put(to[i], chunk, (size_t)r);
            else if (r == 0 || (r < 0 && errno != EINTR)) {
                close(fd[i].fd);
                fd[i].fd = -1;
                open_n--;
            }
        }
    }
    for (int i = 0; i < 2; i++)
        if (fd[i].fd >= 0)
            close(fd[i].fd);
    if (late)
        kill(-pid, SIGKILL); /* it and what it started */
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR)
        ;
    janas_buf_put(out, "", 1);
    out->n--;
    janas_buf_put(err, "", 1);
    err->n--;
    if (late) {
        snprintf(why, why_len, "%s took over %d s and was stopped", name,
                 timeout_ms / 1000);
        return -1;
    }
    if (!WIFEXITED(st)) {
        snprintf(why, why_len, "%s stopped by a signal", name);
        return -1;
    }
    if (WEXITSTATUS(st) == 127) {
        snprintf(why, why_len, "%s is not installed", name);
        return -1;
    }
    if (WEXITSTATUS(st) == 126) {
        snprintf(why, why_len, "cannot enter %s", dir);
        return -1;
    }
    return WEXITSTATUS(st);
}
