/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * win_compat.h - what MinGW-w64's C library lacks of the POSIX one that
 * Janas is written against, for Windows builds only: build.sh makes every
 * file of Janas start with it (-include), never the libraries of others.
 *
 * It holds only what a POSIX call can be mapped to without changing what
 * the caller means: setenv, localtime_r, pread, mkdir with a mode, a rename
 * that replaces, memmem and the like. What differs in kind - processes,
 * sockets, the terminal, the CPU's topology - has code of its own for each
 * system. windows.h is not included here: its macros (near, far, small,
 * min, max...) would collide with ordinary names; win_compat.c includes it.
 *
 * Every file is opened in binary mode, as on POSIX: open() and fopen()
 * add the flag Windows needs, so that a byte read is the byte written.
 */
#ifndef JANAS_COMMON_WIN_COMPAT_H
#define JANAS_COMMON_WIN_COMPAT_H
#ifdef _WIN32

/* the thread-safe localtime_r and gmtime_r of MinGW's time.h */
#ifndef _POSIX_THREAD_SAFE_FUNCTIONS
#define _POSIX_THREAD_SAFE_FUNCTIONS 200112L
#endif

#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* open(2)'s flags Windows has no use for, or names otherwise */
#ifndef O_CLOEXEC
#define O_CLOEXEC _O_NOINHERIT
#endif
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef O_NONBLOCK
#define O_NONBLOCK 0
#endif

/* environment: HOME and XDG's directories answered with Windows's own
   when they are not set (HOME %USERPROFILE%, XDG_CONFIG_HOME %APPDATA%,
   XDG_CACHE_HOME and XDG_DATA_HOME %LOCALAPPDATA%, TMPDIR %TEMP%), with /
   between the parts: what Janas keeps goes to %LOCALAPPDATA%\janas and
   %APPDATA%\janas, and setting them still decides */
char *janas_getenv(const char *name);
#define getenv janas_getenv
int janas_setenv(const char *name, const char *value, int overwrite);
int janas_unsetenv(const char *name);
#define setenv janas_setenv
#define unsetenv janas_unsetenv

/* time: struct tm in UTC to time_t */
#define timegm _mkgmtime

/* strings */
char *janas_strcasestr(const char *hay, const char *needle);
void *janas_memmem(const void *hay, size_t hay_len, const void *needle,
                   size_t needle_len);
char *janas_strndup(const char *s, size_t n);
#define strcasestr janas_strcasestr
#define memmem janas_memmem
#define strndup janas_strndup

/* files: binary always; a rename that replaces the target as POSIX's
   does; reads and writes at an offset; mkdir without a mode */
int janas_open(const char *path, int flags, ...);
FILE *janas_fopen(const char *path, const char *mode);
int janas_rename(const char *from, const char *to);
ssize_t janas_pread(int fd, void *buf, size_t n, int64_t off);
ssize_t janas_pwrite(int fd, const void *buf, size_t n, int64_t off);
char *janas_realpath(const char *path, char *resolved);
#define open janas_open
#define fopen janas_fopen
#define rename janas_rename
#define pread janas_pread
#define pwrite janas_pwrite
#define realpath janas_realpath
/* remove() takes an empty directory too, as POSIX's */
int janas_remove(const char *path);
#define remove janas_remove
/* EEXIST for whatever is already a directory, as POSIX's says it of "C:"
   and "C:/Users" too */
int janas_mkdir(const char *path);
#define mkdir(path, mode) janas_mkdir(path)
/* no symbolic links to follow: lstat is stat, and nothing is one */
#define lstat stat
#ifndef S_ISLNK
#define S_ISLNK(m) 0
#endif
/* a file's data written out (msvcrt's _commit); modes are POSIX's only */
int janas_fsync(int fd);
#define fsync janas_fsync
#define fchmod(fd, mode) ((void)(fd), (void)(mode), 0)

/* advisory locks on a whole file */
#define LOCK_SH 1
#define LOCK_EX 2
#define LOCK_NB 4
#define LOCK_UN 8
int janas_flock(int fd, int op);
#define flock janas_flock

/* the system's sizes */
#define _SC_NPROCESSORS_ONLN 84
#define _SC_PAGESIZE 30
#define _SC_PAGE_SIZE _SC_PAGESIZE
#define _SC_PHYS_PAGES 85
long janas_sysconf(int name);
#define sysconf janas_sysconf

/* No aligned_alloc nor posix_memalign: janas_aligned_alloc and
   janas_aligned_free (common/aligned.h), so that free() never has to tell
   an aligned block from an ordinary one. */

/* a line of any length, as POSIX's getline */
ssize_t janas_getline(char **line, size_t *cap, FILE *f);
#define getline janas_getline

/* a directory made under a name of its own (the template's XXXXXX) */
char *janas_mkdtemp(char *tmpl);
#define mkdtemp janas_mkdtemp

/* A stream written into memory: a temporary file the system deletes on
   closing, whose content fflush and fclose hand to *ptr and *size as
   glibc's open_memstream does (the buffer the caller frees). */
FILE *janas_open_memstream(char **ptr, size_t *size);
int janas_fflush(FILE *f);
int janas_fclose(FILE *f);
#define open_memstream janas_open_memstream
#define fflush janas_fflush
#define fclose janas_fclose

/* a watchdog for the tests: ends the process after seconds */
unsigned janas_alarm(unsigned seconds);
#define alarm janas_alarm

#endif /* _WIN32 */
#endif
