/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * proc.h - a program started by Janas: an MCP server, a service's git, a
 * tool beside the chat. POSIX: posix_spawn (proc.c); Windows: CreateProcess
 * with a job object that holds the program and what it starts in turn
 * (proc_win.c).
 */
#ifndef JANAS_COMMON_PROC_H
#define JANAS_COMMON_PROC_H

#include <stddef.h>
#include <stdint.h>

struct janas_proc {
    intptr_t id;  /* 0: none. POSIX its pid, Windows its process's handle */
    intptr_t job; /* Windows: the job object; 0 elsewhere */
    unsigned long pid;
    int adopted; /* POSIX: not a child of this process (janas_proc_adopt) */
};

/* A group of its own: the terminal's Ctrl-C does not reach it, and
   janas_proc_kill reaches what it started in turn (npx starts node). */
#define JANAS_PROC_GROUP 1
/* No terminal at all: nothing can be asked of the user (setsid; on
   Windows no console). */
#define JANAS_PROC_DETACHED 2

/*
 * Starts argv[0] - a path, or a name looked for in PATH (on Windows a .cmd
 * or .bat too, through cmd.exe) - with argv, in dir (NULL: this one's),
 * with this process's environment and env laid over it (KEY=VALUE,
 * NULL-terminated; NULL: none). fd_in, fd_out and fd_err become its
 * standard input, output and error; -1 leaves it this process's.
 * 0, or -1 with the reason in err (errno ENOENT when the program is not
 * there).
 */
int janas_proc_start(struct janas_proc *p, const char *const argv[],
                     const char *const env[], const char *dir, int fd_in,
                     int fd_out, int fd_err, int flags, char *err,
                     size_t err_len);

/*
 * A process this one did not start (one of an earlier run, found by its
 * pid), followed as if it had: waited for, killed (its group too when
 * group, as JANAS_PROC_GROUP started it). On Windows it is put in a job of
 * this process's, so that it ends with it. 0, or -1 when there is no such
 * process to follow.
 */
int janas_proc_adopt(struct janas_proc *p, unsigned long pid, int group);

/*
 * Waits up to timeout_ms (-1: as long as it takes, 0: only looks) for it
 * to end: 1 when it has, with its exit status in *code (may be NULL; -1
 * when a signal ended it), 0 when it still runs, -1 on an error. Once it
 * has ended p is cleared.
 */
int janas_proc_wait(struct janas_proc *p, int timeout_ms, int *code);

/* Asks it and what it started to end (SIGTERM; Windows: Ctrl+Break to
   its group), or with hard ends them (SIGKILL; Windows: the job
   terminated). */
void janas_proc_kill(struct janas_proc *p, int hard);

/* A pipe, neither end passed on to programs started later. 0 or -1. */
int janas_pipe(int fd[2]);

/* A descriptor of the null device (/dev/null, NUL), for writing or for
   reading. */
int janas_null_fd(int for_writing);

#endif
