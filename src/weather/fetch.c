/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * fetch.c - janas-weather asks its sources (see weather.h): a GET over
 * HTTPS, the answers kept a while. A forecast changes once an hour at most
 * and a bulletin once a day, while a conversation asks again and again
 * about the same place: the sources are spared, Open-Meteo's free use is a
 * number of calls a day.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/https_get.h"
#include "weather.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

#define N_KEPT 32

static struct {
    char *url;
    time_t at;
    int keep_s;
    struct janas_buf body;
} kept[N_KEPT];

int wx_fetch(const char *url, int keep_s, int with_agent, struct janas_buf *out,
             char *err, size_t err_len)
{
    time_t now = time(NULL);
    int slot = 0;
    for (int i = 0; i < N_KEPT; i++) {
        if (kept[i].url && strcmp(kept[i].url, url) == 0 &&
            now - kept[i].at < kept[i].keep_s) {
            out->n = 0;
            janas_buf_put(out, kept[i].body.p, kept[i].body.n);
            janas_buf_put(out, "", 1);
            out->n--;
            return out->oom ? -1 : 200;
        }
        if (kept[i].at < kept[slot].at)
            slot = i;
    }
    /* MET Norway asks for a User-Agent naming the program and a way to
       reach its authors; requests without one are refused */
    static const char *const agent[] = {
        "User-Agent: janas-weather/" JANAS_VERSION
        " https://github.com/prabanta-dev/janas"};
    int status = janas_https_get(url, with_agent ? agent : NULL,
                                 with_agent ? 1 : 0, out, err, err_len);
    if (status == 200 && keep_s > 0 && !out->oom) {
        free(kept[slot].url);
        janas_buf_free(&kept[slot].body);
        kept[slot].url = strdup(url);
        kept[slot].at = now;
        kept[slot].keep_s = keep_s;
        kept[slot].body = (struct janas_buf){0};
        janas_buf_put(&kept[slot].body, out->p, out->n);
        if (!kept[slot].url || kept[slot].body.oom) {
            free(kept[slot].url);
            janas_buf_free(&kept[slot].body);
            kept[slot].url = NULL;
            kept[slot].at = 0;
        }
    }
    return status;
}

void wx_fetch_free(void)
{
    for (int i = 0; i < N_KEPT; i++) {
        free(kept[i].url);
        janas_buf_free(&kept[i].body);
    }
    memset(kept, 0, sizeof kept);
}

struct janas_json_doc *wx_parse(const struct janas_buf *b, const char *who,
                                char *err, size_t err_len)
{
    char why[128];
    struct janas_json_doc *d =
        janas_json_parse(b->p ? b->p : "", b->n, why, sizeof why);
    if (!d)
        snprintf(err, err_len, "%s: an answer it could not read (%s)", who,
                 why);
    return d;
}
