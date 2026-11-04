/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * proc.c - see proc.h: POSIX, with posix_spawn. Windows: proc_win.c.
 */
#ifndef _WIN32
#define _GNU_SOURCE /* pipe2, posix_spawn_file_actions_addchdir_np */
#include "common/proc.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef environ /* MinGW's stdlib.h: a macro for _environ (issue #40) */
extern char **environ;
#endif

/* This process's environment with env laid over it: a KEY=VALUE of env
   replaces the variable of the same name. */
static char **merge_env(const char *const env[])
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
        out[k++] = (char *)env[j];
    return out;
}

int janas_proc_start(struct janas_proc *p, const char *const argv[],
                     const char *const env[], const char *dir, int fd_in,
                     int fd_out, int fd_err, int flags, char *err,
                     size_t err_len)
{
    memset(p, 0, sizeof(*p));
    if (!argv || !argv[0] || !*argv[0]) {
        snprintf(err, err_len, "no command");
        errno = EINVAL;
        return -1;
    }
    struct stat st;
    if (dir && (stat(dir, &st) != 0 || !S_ISDIR(st.st_mode))) {
        snprintf(err, err_len, "cannot enter %s", dir);
        errno = ENOTDIR;
        return -1;
    }
    char **envp = merge_env(env);
    posix_spawn_file_actions_t fa;
    posix_spawnattr_t at;
    int have_fa = posix_spawn_file_actions_init(&fa) == 0;
    int have_at = posix_spawnattr_init(&at) == 0;
    int rc = -1;
    if (!envp || !have_fa || !have_at) {
        snprintf(err, err_len, "out of memory");
        errno = ENOMEM;
        goto out;
    }
    if (fd_in >= 0)
        posix_spawn_file_actions_adddup2(&fa, fd_in, 0);
    if (fd_out >= 0)
        posix_spawn_file_actions_adddup2(&fa, fd_out, 1);
    if (fd_err >= 0)
        posix_spawn_file_actions_adddup2(&fa, fd_err, 2);
    if (dir)
        posix_spawn_file_actions_addchdir_np(&fa, dir);
    /* the signals as a new program has them */
    sigset_t none, dflt;
    sigemptyset(&none);
    sigemptyset(&dflt);
    sigaddset(&dflt, SIGPIPE);
    sigaddset(&dflt, SIGINT);
    posix_spawnattr_setsigmask(&at, &none);
    posix_spawnattr_setsigdefault(&at, &dflt);
    short f = POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF;
    if (flags & JANAS_PROC_DETACHED)
        f |= POSIX_SPAWN_SETSID; /* a session, and so a group, of its own */
    else if (flags & JANAS_PROC_GROUP) {
        posix_spawnattr_setpgroup(&at, 0);
        f |= POSIX_SPAWN_SETPGROUP;
    }
    posix_spawnattr_setflags(&at, f);
    pid_t pid;
    int e = posix_spawnp(&pid, argv[0], &fa, &at, (char *const *)argv, envp);
    if (e != 0) {
        snprintf(err, err_len, "cannot start %s: %s", argv[0], strerror(e));
        errno = e;
        goto out;
    }
    p->id = pid;
    p->pid = (unsigned long)pid;
    p->job = flags & (JANAS_PROC_GROUP | JANAS_PROC_DETACHED) ? 1 : 0;
    rc = 0;
out:
    if (have_fa)
        posix_spawn_file_actions_destroy(&fa);
    if (have_at)
        posix_spawnattr_destroy(&at);
    free(envp);
    return rc;
}

static int status_code(int st)
{
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

int janas_proc_wait(struct janas_proc *p, int timeout_ms, int *code)
{
    if (p->id <= 0)
        return -1;
    pid_t pid = (pid_t)p->id;
    int st = 0;
    for (int waited = 0;; waited += 10) {
        pid_t r = waitpid(pid, &st, timeout_ms < 0 ? 0 : WNOHANG);
        if (r < 0 && errno == EINTR)
            continue;
        if (r == pid || (r < 0 && errno == ECHILD)) {
            if (code)
                *code = r == pid ? status_code(st) : -1;
            memset(p, 0, sizeof(*p));
            return 1;
        }
        if (r < 0)
            return -1;
        if (waited >= timeout_ms)
            return 0;
        struct timespec t = {0, 10 * 1000 * 1000};
        nanosleep(&t, NULL);
    }
}

void janas_proc_kill(struct janas_proc *p, int hard)
{
    if (p->id <= 0)
        return;
    pid_t pid = (pid_t)p->id;
    /* its group, which it leads, where it has one of its own */
    kill(p->job ? -pid : pid, hard ? SIGKILL : SIGTERM);
}

int janas_pipe(int fd[2])
{
    return pipe2(fd, O_CLOEXEC);
}

int janas_null_fd(int for_writing)
{
    return open("/dev/null", (for_writing ? O_WRONLY : O_RDONLY) | O_CLOEXEC);
}

#endif /* !_WIN32 */
