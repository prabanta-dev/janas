/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_stdio.c - an MCP server as a child process (see mcp_stdio.h).
 */
#define _GNU_SOURCE /* pipe2 */
#include "common/mcp_stdio.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define LINE_MAX_BYTES ((size_t)64 << 20)

/* This process's environment with env laid over it: a KEY=VALUE of env
   replaces the variable of the same name. */
static char **merge_env(char *const env[])
{
    size_t n = 0, m = 0;
    while (environ[n])
        n++;
    while (env && env[m])
        m++;
    char **out = calloc(n + m + 1, sizeof(*out));
    if (!out)
        return NULL;
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        const char *eq = strchr(environ[i], '=');
        size_t key = eq ? (size_t)(eq - environ[i]) + 1 : strlen(environ[i]);
        int replaced = 0;
        for (size_t j = 0; j < m && !replaced; j++)
            replaced = strncmp(env[j], environ[i], key) == 0;
        if (!replaced)
            out[k++] = environ[i];
    }
    for (size_t j = 0; j < m; j++)
        out[k++] = env[j];
    return out;
}

int janas_mcp_proc_start(struct janas_mcp_proc *p, char *const argv[],
                         char *const env[], const char *log_path, char *err,
                         size_t err_len)
{
    memset(p, 0, sizeof(*p));
    p->in = p->out = -1;
    int to[2] = {-1, -1}, from[2] = {-1, -1};
    int log = -1, rc = -1;
    char **envp = NULL;
    posix_spawn_file_actions_t fa;
    posix_spawnattr_t at;
    int have_fa = 0, have_at = 0;
    if (!argv || !argv[0] || !*argv[0]) {
        snprintf(err, err_len, "no command");
        return -1;
    }
    if (pipe2(to, O_CLOEXEC) != 0 || pipe2(from, O_CLOEXEC) != 0) {
        snprintf(err, err_len, "pipe: %s", strerror(errno));
        goto out;
    }
    if (log_path)
        log = open(log_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (log < 0)
        log = open("/dev/null", O_WRONLY | O_CLOEXEC);
    envp = merge_env(env);
    if (!envp || log < 0) {
        snprintf(err, err_len, "out of memory or no /dev/null");
        goto out;
    }
    have_fa = posix_spawn_file_actions_init(&fa) == 0;
    have_at = posix_spawnattr_init(&at) == 0;
    if (!have_fa || !have_at) {
        snprintf(err, err_len, "out of memory");
        goto out;
    }
    posix_spawn_file_actions_adddup2(&fa, to[0], 0);
    posix_spawn_file_actions_adddup2(&fa, from[1], 1);
    posix_spawn_file_actions_adddup2(&fa, log, 2);
    /* a group of its own, the signals as a new program has them */
    sigset_t none, dflt;
    sigemptyset(&none);
    sigemptyset(&dflt);
    sigaddset(&dflt, SIGPIPE);
    sigaddset(&dflt, SIGINT);
    posix_spawnattr_setpgroup(&at, 0);
    posix_spawnattr_setsigmask(&at, &none);
    posix_spawnattr_setsigdefault(&at, &dflt);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP |
                                      POSIX_SPAWN_SETSIGMASK |
                                      POSIX_SPAWN_SETSIGDEF);
    int e = posix_spawnp(&p->pid, argv[0], &fa, &at, argv, envp);
    if (e != 0) {
        p->pid = 0;
        snprintf(err, err_len, "cannot start %s: %s", argv[0], strerror(e));
        goto out;
    }
    p->in = to[1];
    p->out = from[0];
    to[1] = from[0] = -1;
    rc = 0;
out:
    if (have_fa)
        posix_spawn_file_actions_destroy(&fa);
    if (have_at)
        posix_spawnattr_destroy(&at);
    free(envp);
    for (int i = 0; i < 2; i++) {
        if (to[i] >= 0)
            close(to[i]);
        if (from[i] >= 0)
            close(from[i]);
    }
    if (log >= 0)
        close(log);
    return rc;
}

/* A write to a server that has gone raises SIGPIPE, which would end the
   chat: it is held back on this thread for the write, and taken away if it
   came, so the write only fails. */
int janas_mcp_proc_send(struct janas_mcp_proc *p, const char *s, size_t n)
{
    if (p->in < 0)
        return -1;
    sigset_t pipe_set, old;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    sigset_t pending;
    sigpending(&pending);
    int was_pending = sigismember(&pending, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &pipe_set, &old);
    int rc = 0;
    while (n > 0) {
        ssize_t w = write(p->in, s, n);
        if (w < 0 && errno == EINTR)
            continue;
        if (w <= 0) {
            rc = -1;
            break;
        }
        s += w;
        n -= (size_t)w;
    }
    if (rc && !was_pending) {
        struct timespec zero = {0, 0};
        while (sigtimedwait(&pipe_set, NULL, &zero) < 0 && errno == EINTR)
            ;
    }
    pthread_sigmask(SIG_SETMASK, &old, NULL);
    return rc;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

int janas_mcp_proc_line(struct janas_mcp_proc *p, int timeout_ms,
                        const char **line, size_t *n)
{
    double end = now_ms() + timeout_ms;
    size_t scanned = p->start;
    for (;;) {
        char *nl = p->n > scanned
                       ? memchr(p->buf + scanned, '\n', p->n - scanned)
                       : NULL;
        if (nl) {
            size_t len = (size_t)(nl - (p->buf + p->start));
            *line = p->buf + p->start;
            if (len && (*line)[len - 1] == '\r')
                len--;
            *n = len;
            p->start = (size_t)(nl - p->buf) + 1;
            return 1;
        }
        scanned = p->n;
        if (p->eof || p->out < 0)
            return -1;
        if (p->n - p->start > LINE_MAX_BYTES)
            return -1;
        if (p->start > 0 && p->start == p->n) {
            p->n = p->start = scanned = 0;
        } else if (p->start > p->cap / 2) { /* the lines taken go */
            memmove(p->buf, p->buf + p->start, p->n - p->start);
            p->n -= p->start;
            scanned -= p->start;
            p->start = 0;
        }
        if (p->cap - p->n < 65536) {
            size_t cap = p->cap ? 2 * p->cap : 131072;
            char *b = realloc(p->buf, cap);
            if (!b)
                return -1;
            p->buf = b;
            p->cap = cap;
        }
        /* one look at least, also with no time left */
        double left = end - now_ms();
        struct pollfd pf = {.fd = p->out, .events = POLLIN};
        int r = poll(&pf, 1,
                     left <= 0    ? 0
                     : left > 1e9 ? 1000000000
                                  : (int)left + 1);
        if (r < 0 && errno == EINTR)
            return 0; /* a signal: the caller looks at why */
        if (r <= 0) {
            if (now_ms() >= end)
                return 0;
            continue;
        }
        ssize_t got = read(p->out, p->buf + p->n, p->cap - p->n);
        if (got < 0 && errno == EINTR)
            continue;
        if (got <= 0)
            p->eof = 1;
        else
            p->n += (size_t)got;
    }
}

/* Waits up to ms for the process to end; 1 when it has. */
static int reap(pid_t pid, int ms)
{
    for (int waited = 0;; waited += 10) {
        pid_t r = waitpid(pid, NULL, WNOHANG);
        if (r == pid || (r < 0 && errno == ECHILD))
            return 1;
        if (waited >= ms)
            return 0;
        struct timespec t = {0, 10 * 1000 * 1000};
        nanosleep(&t, NULL);
    }
}

void janas_mcp_proc_stop(struct janas_mcp_proc *p)
{
    if (p->in >= 0)
        close(p->in);
    p->in = -1;
    if (p->pid > 0 && !reap(p->pid, 1000)) {
        kill(-p->pid, SIGTERM);
        if (!reap(p->pid, 2000)) {
            kill(-p->pid, SIGKILL);
            waitpid(p->pid, NULL, 0);
        }
    }
    p->pid = 0;
    if (p->out >= 0)
        close(p->out);
    p->out = -1;
    free(p->buf);
    p->buf = NULL;
    p->n = p->cap = p->start = 0;
}
