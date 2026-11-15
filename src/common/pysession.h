/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * pysession.h - a Python session in the cage (pysession.c): the driver a
 * cage runs (/usr/bin/python3 -I -c JANAS_PY_DRIVER, with /mnt/data its
 * place for files), and a piece of code exchanged for its answer.
 */
#ifndef JANAS_COMMON_PYSESSION_H
#define JANAS_COMMON_PYSESSION_H

#include <stddef.h>

#include "common/sandbox.h"
#include "llm/json.h"

extern const char JANAS_PY_DRIVER[];

/* code sent to the session in sb, its answer read within seconds: the
   answer's JSON text ({"logs", "error", "images"}, malloc'd), or NULL with
   the reason (the time up: the caller should kill and start it again;
   the session gone). What C code writes to the cage's stderr is appended
   to raw. */
char *janas_py_exchange(struct janas_sandbox *sb, const char *code, int seconds,
                        struct janas_buf *raw, char *why, size_t n);

#endif
