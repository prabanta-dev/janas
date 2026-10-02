/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * run.c - janas-git runs git (see git.h): with its arguments as they are,
 * no shell between; in a session of its own, without a terminal, so that
 * a password asked (ssh, https) fails at once instead of waiting for an
 * answer nobody can give; with a time limit, after which git and what it
 * started are stopped.
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

#include "git.h"

#define OUT_MAX (1u << 20)

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

int gt_run(const char *dir, const char *const *argv, int timeout_ms,
           struct janas_buf *out, struct janas_buf *err, char *why,
           size_t why_len)
{
    const char *av[64];
    size_t n = 0;
    av[n++] = "git";
    for (size_t i = 0; argv[i] && n < 63; i++)
        av[n++] = argv[i];
    av[n] = NULL;
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
        snprintf(why, why_len, "cannot start git: %s", strerror(errno));
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
        if (chdir(dir) != 0)
            _exit(126);
        setenv("GIT_TERMINAL_PROMPT", "0", 1);
        setenv("GIT_EDITOR", "true", 1);
        setenv("GIT_OPTIONAL_LOCKS", "0", 1); /* status takes no lock */
        setenv("LC_ALL", "C", 1);             /* git's words, in English */
        execvp("git", (char *const *)av);
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
            if (r > 0 && to[i]->n < OUT_MAX)
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
        kill(-pid, SIGKILL); /* git and what it started (ssh) */
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR)
        ;
    janas_buf_put(out, "", 1);
    out->n--;
    janas_buf_put(err, "", 1);
    err->n--;
    if (late) {
        snprintf(why, why_len, "git took over %d s and was stopped",
                 timeout_ms / 1000);
        return -1;
    }
    if (!WIFEXITED(st)) {
        snprintf(why, why_len, "git stopped by a signal");
        return -1;
    }
    if (WEXITSTATUS(st) == 127) {
        snprintf(why, why_len, "git is not installed");
        return -1;
    }
    if (WEXITSTATUS(st) == 126) {
        snprintf(why, why_len, "cannot enter %s", dir);
        return -1;
    }
    return WEXITSTATUS(st);
}
