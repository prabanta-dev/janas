/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * https_get.h - a GET over HTTPS (https_get.c), for Janas's services.
 */
#ifndef JANAS_COMMON_HTTPS_GET_H
#define JANAS_COMMON_HTTPS_GET_H

#include <stddef.h>

#include "llm/json.h"

/* GET of url (https), following redirects, the body into out (with a 0
   after it, not counted). The HTTP status (200...), or -1 with the reason
   in err. */
int janas_https_get(const char *url, const char *const *headers,
                    size_t n_headers, struct janas_buf *out, char *err,
                    size_t err_len);
/* The same, waiting at most timeout_ms for each part of the answer
   (janas_https_get: 15 s). */
int janas_https_get_ms(const char *url, const char *const *headers,
                       size_t n_headers, int timeout_ms, struct janas_buf *out,
                       char *err, size_t err_len);

#endif
