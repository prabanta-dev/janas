/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * fetch.c - janas-quakes asks its sources (see quakes.h): a User-Agent
 * naming the program, a pause between two requests to the same server, the
 * answers kept a minute, so that a question asked again costs the sources
 * nothing.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "quakes.h"
#include "services/common/https_get.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

#define N_KEPT 16
#define N_HOSTS 6

static struct {
    char *url;
    time_t at;
    int status;
    struct janas_buf body;
} kept[N_KEPT];

static struct {
    char host[96];
    struct timespec last;
} hosts[N_HOSTS];

static double since(const struct timespec *t)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - t->tv_sec) +
           (double)(now.tv_nsec - t->tv_nsec) * 1e-9;
}

static void pace(const char *url)
{
    const char *h = strstr(url, "://");
    h = h ? h + 3 : url;
    size_t n = strcspn(h, "/?:");
    int slot = -1, empty = -1;
    for (int i = 0; i < N_HOSTS && slot < 0; i++) {
        if (strlen(hosts[i].host) == n && strncmp(hosts[i].host, h, n) == 0)
            slot = i;
        else if (empty < 0 && !hosts[i].host[0])
            empty = i;
    }
    if (slot >= 0) {
        double wait = QK_PAUSE_S - since(&hosts[slot].last);
        if (wait > 0 && wait <= QK_PAUSE_S) {
            struct timespec ts = {.tv_sec = (time_t)wait};
            ts.tv_nsec = (long)((wait - (double)ts.tv_sec) * 1e9);
            nanosleep(&ts, NULL);
        }
    } else {
        slot = empty >= 0 ? empty : 0;
        snprintf(hosts[slot].host, sizeof hosts[slot].host, "%.*s", (int)n, h);
    }
    clock_gettime(CLOCK_MONOTONIC, &hosts[slot].last);
}

int qk_fetch(const char *url, struct janas_buf *out, char *err, size_t err_len)
{
    time_t now = time(NULL);
    int slot = 0;
    for (int i = 0; i < N_KEPT; i++) {
        if (kept[i].url && strcmp(kept[i].url, url) == 0 &&
            now - kept[i].at < QK_KEEP_S) {
            out->n = 0;
            janas_buf_put(out, kept[i].body.p ? kept[i].body.p : "",
                          kept[i].body.n);
            janas_buf_put(out, "", 1);
            out->n--;
            return out->oom ? -1 : kept[i].status;
        }
        if (kept[i].at < kept[slot].at)
            slot = i;
    }
    static const char *const agent[] = {
        "User-Agent: janas-quakes/" JANAS_VERSION
        " (+https://github.com/prabanta-dev/janas)"};
    pace(url);
    int status = janas_https_get_ms(url, agent, 1, QK_MS, out, err, err_len);
    if ((status == 200 || status == 204) && !out->oom) {
        free(kept[slot].url);
        janas_buf_free(&kept[slot].body);
        kept[slot].url = strdup(url);
        kept[slot].at = now;
        kept[slot].status = status;
        janas_buf_put(&kept[slot].body, out->p ? out->p : "", out->n);
    }
    return status;
}

void qk_url_put(struct janas_buf *b, const char *s)
{
    static const char hex[] = "0123456789ABCDEF";
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
            janas_buf_put(b, (const char *)&c, 1);
        else {
            char e[3] = {'%', hex[c >> 4], hex[c & 15]};
            janas_buf_put(b, e, 3);
        }
    }
}
