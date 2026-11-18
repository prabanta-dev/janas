/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sandbox.h - a cage for programs Janas runs but did not write (the code a
 * model writes for code_interpreter): Linux namespaces of its own (user,
 * mount, PID, network, IPC, UTS) without privileges, a root of its own
 * with the system's /usr read-only and one directory writable, no network,
 * no other process to see, no new processes (seccomp: threads only),
 * limits on memory, file sizes and open files.
 *
 * The cage is built by the program itself run again: janas_sandbox_spawn
 * runs /proc/self/exe with janas_sandbox_exec's first argument, and a
 * program offering the cage calls janas_sandbox_main first thing in main.
 * Linux (x86-64 and AArch64). On Windows (sandbox_win.c) the same calls
 * build it with what Windows has: an AppContainer without network and
 * with the folders granted alone, inside a Job Object with the limits;
 * the data directory is the program's current folder there, not
 * /mnt/data. Elsewhere janas_sandbox_available says why not.
 */
#ifndef JANAS_COMMON_SANDBOX_H
#define JANAS_COMMON_SANDBOX_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#define JANAS_SANDBOX_ARG "--janas-sandbox-exec"

struct janas_sandbox_cfg {
    const char *data_dir;  /* host directory, writable at /mnt/data */
    const char *const *ro; /* more host paths, read-only at the same path
                              (NULL-terminated; may be NULL) */
    uint64_t mem_bytes;    /* address space; 0: 2 GiB */
    uint64_t file_bytes;   /* largest file written; 0: 1 GiB */
    uint64_t tmp_bytes;    /* the /tmp in memory; 0: 256 MiB */
};

struct janas_sandbox {
    pid_t pid;        /* the cage's first process */
    int in, out, err; /* its standard input, output and error */
#ifdef _WIN32
    intptr_t process, job; /* its process's handle, its Job Object's */
#endif
};

/* Whether a cage can be built here: 1, or 0 with the reason. */
int janas_sandbox_available(char *why, size_t n);

/* argv (argv[0] a path inside the cage, as /usr/bin/python3) run in a cage
   with envp as its whole environment: 0 with sb filled, -1 with the
   reason. The cage dies with its caller. */
int janas_sandbox_spawn(const struct janas_sandbox_cfg *c, char *const argv[],
                        char *const envp[], struct janas_sandbox *sb, char *why,
                        size_t n);

/* The cage killed (and its pipes closed). */
void janas_sandbox_kill(struct janas_sandbox *sb);

/* What building cages left behind, removed when nothing will run in them
   again: on Windows the AppContainer's profile; nothing elsewhere. */
void janas_sandbox_forget(void);

/* In main, first: when argv[1] is JANAS_SANDBOX_ARG this process is a cage
   being built, and the function never returns; otherwise it returns at
   once. */
void janas_sandbox_main(int argc, char **argv);

#endif
