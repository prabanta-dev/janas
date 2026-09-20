/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * mcp_stdio.h - the stdio transport of MCP: a server is a program started
 * here, spoken to on its standard input and heard on its standard output,
 * one message per line. What it writes to its standard error goes to a log
 * file, never to the terminal of the program that started it.
 *
 * The server runs in a process group of its own, so the Ctrl-C that stops
 * a reply in the chat does not stop it too, and its end (standard input
 * closed, then SIGTERM, then SIGKILL) reaches whatever it started in turn
 * (npx starts node, uvx starts python).
 */
#ifndef JANAS_MCP_STDIO_H
#define JANAS_MCP_STDIO_H

#include <stddef.h>
#include <sys/types.h>

struct janas_mcp_proc {
    pid_t pid;   /* 0: not running */
    int in, out; /* its standard input (we write), output (we read) */
    char *buf;   /* what was read and not yet taken as lines */
    size_t n, cap, start;
    int eof;
};

/* Starts argv[0] (looked up in PATH) with argv, the environment of this
   process plus env (KEY=VALUE, NULL-terminated; may be NULL), its standard
   error appended to log_path (NULL: thrown away). 0, or -1 with the reason
   in err. */
int janas_mcp_proc_start(struct janas_mcp_proc *p, char *const argv[],
                         char *const env[], const char *log_path, char *err,
                         size_t err_len);

/* Writes n bytes (one or more whole lines). 0, or -1 when the server has
   gone. */
int janas_mcp_proc_send(struct janas_mcp_proc *p, const char *s, size_t n);

/* The next line, without its newline, in *line (valid until the next
   call): 1, or 0 when none came within timeout_ms (0: what has already
   arrived), or -1 when the server has closed its output or wrote a line
   longer than 64 MiB. The reading end may be any descriptor: a server
   reads its own standard input with it, pid 0 and in -1. */
int janas_mcp_proc_line(struct janas_mcp_proc *p, int timeout_ms,
                        const char **line, size_t *n);

/* Ends it: standard input closed, then SIGTERM after a second, SIGKILL
   after two. Safe on a process never started. */
void janas_mcp_proc_stop(struct janas_mcp_proc *p);

#endif
