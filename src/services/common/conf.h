/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * conf.h - a service's settings file (conf.c): lines "name = value", a
 * line starting with # a comment. A line naming no setting the service
 * knows is not left aside in silence: it is told, with the name nearest
 * to it, so that a key written "aviationstak_key" does not leave the
 * service keyless without a word.
 */
#ifndef JANAS_SERVICES_CONF_H
#define JANAS_SERVICES_CONF_H

#include <stddef.h>

/*
 * The value of name in the file path, into val (cap bytes); names[0..n)
 * are all the settings the file may hold. 1 with the value, 0 when the
 * file or the line is missing. The first line naming none of them is told
 * in why ("flights.conf, line 2: aviationstak_key is not a setting (did
 * you mean aviationstack_key?)"), which is "" otherwise.
 */
int janas_conf_get(const char *path, const char *const *names, size_t n,
                   const char *name, char *val, size_t cap, char *why,
                   size_t why_len);

#endif
