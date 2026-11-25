/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mac_compat.h - what macOS's C library lacks of the Linux one that Janas is
 * written against, for macOS builds only: build.sh makes every file of
 * Janas start with it (-include), never the libraries of others; the calls
 * are in mac_posix.c.
 *
 * As win_compat.h, it holds only what can be mapped without changing what
 * the caller means: pipe2, getrandom, the sockets' close-on-exec and
 * non-blocking flags, struct stat's st_mtim, sigtimedwait's poll of a
 * pending signal (a zero timeout, the one Janas uses), an advice madvise
 * may refuse. What differs in kind has code of its own for each system.
 */
#ifndef JANAS_COMMON_MAC_COMPAT_H
#define JANAS_COMMON_MAC_COMPAT_H
#ifdef __APPLE__

/* the fewest headers: every file starts with this one, and a header
   brought in here would put its names (accept, send...) in every file */
#include <signal.h>
#include <sys/types.h>
#include <time.h>

/* the printf formats' check: Janas names GCC's (gnu_printf, which on
   Windows tells them from Microsoft's), which macOS's printf has and its
   clang does not name */
#define gnu_printf printf

/* the nanoseconds of a file's times, under Linux's names */
#define st_mtim st_mtimespec
#define st_atim st_atimespec
#define st_ctim st_ctimespec

/* a hint macOS does not take (sys/mman.h has none): madvise refuses it, as
   an older Linux would */
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE (-1)
#endif

/* Linux's flags of socket() and socketpair()'s type, set by mac_posix.c
   with fcntl after the call */
#define SOCK_CLOEXEC 0x10000000
#define SOCK_NONBLOCK 0x20000000
int janas_mac_socket(int domain, int type, int protocol);
int janas_mac_socketpair(int domain, int type, int protocol, int sv[2]);
#define socket(d, t, p) janas_mac_socket(d, t, p)
#define socketpair(d, t, p, sv) janas_mac_socketpair(d, t, p, sv)

/* pipe2 with O_CLOEXEC and O_NONBLOCK */
int pipe2(int fd[2], int flags);

/* the system's random bytes (arc4random_buf): all n, as Linux gives them */
ssize_t getrandom(void *buf, size_t n, unsigned flags);

/* only a zero timeout: takes one of set's pending signals if any, else
   -1 with EAGAIN */
int sigtimedwait(const sigset_t *set, siginfo_t *info,
                 const struct timespec *timeout);

#endif /* __APPLE__ */
#endif
