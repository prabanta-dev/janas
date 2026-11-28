/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * outside.c - another site's page, asked of Brebus's window by a click,
 * opened in the user's browser: only http and https, never this machine's
 * own addresses (they would reach Brebus, or a server of this machine, as
 * a page of their own). The windows of every system ask it here.
 */
#include "brebus.h"
#include "common/proc.h"

#include <string.h>
#include <strings.h>

int brebus_open_outside(const char *url)
{
    if (!url || (strncmp(url, "https://", 8) && strncmp(url, "http://", 7)))
        return -1;
    const char *h = strstr(url, "://") + 3;
    static const char *const lo[] = {"127.", "localhost",    "[::1]",
                                     "0.",   "[::ffff:127.", "[0:"};
    for (size_t i = 0; i < sizeof lo / sizeof *lo; i++)
        if (!strncasecmp(h, lo[i], strlen(lo[i])))
            return -1;
    return janas_open_url(url);
}
