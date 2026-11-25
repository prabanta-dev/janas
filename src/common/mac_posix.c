/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mac_posix.c - the calls of mac_compat.h, on macOS.
 */
#ifdef __APPLE__
#include "common/mac_compat.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

/* the system's own, which mac_compat.h's macros renamed in sys/socket.h */
#undef socket
#undef socketpair
int socket(int domain, int type, int protocol);
int socketpair(int domain, int type, int protocol, int sv[2]);

/* the descriptor's close-on-exec and non-blocking flags: 0, or -1 */
static int set_flags(int fd, int cloexec, int nonblock)
{
    if (cloexec && fcntl(fd, F_SETFD, FD_CLOEXEC) == -1)
        return -1;
    if (nonblock) {
        int fl = fcntl(fd, F_GETFL);
        if (fl == -1 || fcntl(fd, F_SETFL, fl | O_NONBLOCK) == -1)
            return -1;
    }
    return 0;
}

static void close_keep_errno(int fd)
{
    int e = errno;
    close(fd);
    errno = e;
}

int janas_mac_socket(int domain, int type, int protocol)
{
    int ce = !!(type & SOCK_CLOEXEC), nb = !!(type & SOCK_NONBLOCK);
    int fd = socket(domain, type & ~(SOCK_CLOEXEC | SOCK_NONBLOCK), protocol);
    if (fd >= 0 && set_flags(fd, ce, nb) != 0) {
        close_keep_errno(fd);
        return -1;
    }
    return fd;
}

int janas_mac_socketpair(int domain, int type, int protocol, int sv[2])
{
    int ce = !!(type & SOCK_CLOEXEC), nb = !!(type & SOCK_NONBLOCK);
    if (socketpair(domain, type & ~(SOCK_CLOEXEC | SOCK_NONBLOCK), protocol,
                   sv) != 0)
        return -1;
    if (set_flags(sv[0], ce, nb) != 0 || set_flags(sv[1], ce, nb) != 0) {
        close_keep_errno(sv[0]);
        close_keep_errno(sv[1]);
        return -1;
    }
    return 0;
}

int pipe2(int fd[2], int flags)
{
    if (flags & ~(O_CLOEXEC | O_NONBLOCK)) {
        errno = EINVAL;
        return -1;
    }
    if (pipe(fd) != 0)
        return -1;
    if (set_flags(fd[0], flags & O_CLOEXEC, flags & O_NONBLOCK) != 0 ||
        set_flags(fd[1], flags & O_CLOEXEC, flags & O_NONBLOCK) != 0) {
        close_keep_errno(fd[0]);
        close_keep_errno(fd[1]);
        return -1;
    }
    return 0;
}

ssize_t getrandom(void *buf, size_t n, unsigned flags)
{
    (void)flags;
    arc4random_buf(buf, n);
    return (ssize_t)n;
}

int sigtimedwait(const sigset_t *set, siginfo_t *info,
                 const struct timespec *timeout)
{
    if (!timeout || timeout->tv_sec || timeout->tv_nsec) {
        errno = EINVAL; /* only the poll Janas makes */
        return -1;
    }
    sigset_t pending;
    if (sigpending(&pending) != 0)
        return -1;
    for (int s = 1; s < NSIG; s++)
        if (sigismember(set, s) && sigismember(&pending, s)) {
            sigset_t one;
            sigemptyset(&one);
            sigaddset(&one, s);
            int got = 0;
            if (sigwait(&one, &got) != 0)
                return -1;
            if (info) {
                info->si_signo = got;
            }
            return got;
        }
    errno = EAGAIN;
    return -1;
}

#endif /* __APPLE__ */
