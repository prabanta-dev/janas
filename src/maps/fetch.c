/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * fetch.c - janas-maps asks its sources (see maps.h): a GET over HTTPS
 * with the User-Agent they ask for, at most one request a second to each
 * host (Nominatim's and FOSSGIS' rule), the answers kept a while: a
 * conversation asks the same route again, by bike after by car, and the
 * places are found once.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/https_get.h"
#include "maps.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

#define N_KEPT 48
#define N_HOSTS 8

static struct {
    char *url;
    time_t at;
    int keep_s;
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

/* Waits until the host of url was last asked over a second ago. */
static void pace(const char *url)
{
    const char *h = strstr(url, "://");
    h = h ? h + 3 : url;
    size_t n = strcspn(h, "/?:");
    int slot = -1;
    for (int i = 0; i < N_HOSTS; i++) {
        if (strlen(hosts[i].host) == n && strncmp(hosts[i].host, h, n) == 0) {
            slot = i;
            break;
        }
        if (slot < 0 && !hosts[i].host[0])
            slot = i;
    }
    if (slot < 0)
        slot = 0;
    if (strlen(hosts[slot].host) == n && strncmp(hosts[slot].host, h, n) == 0) {
        double wait = 1.1 - since(&hosts[slot].last);
        if (wait > 0 && wait <= 1.1) {
            struct timespec ts = {.tv_sec = (time_t)wait};
            ts.tv_nsec = (long)((wait - (double)ts.tv_sec) * 1e9);
            nanosleep(&ts, NULL);
        }
    } else {
        snprintf(hosts[slot].host, sizeof hosts[slot].host, "%.*s", (int)n, h);
    }
    clock_gettime(CLOCK_MONOTONIC, &hosts[slot].last);
}

int mp_fetch(const char *url, int keep_s, struct janas_buf *out, char *err,
             size_t err_len)
{
    return mp_fetch_ms(url, keep_s, 15000, out, err, err_len);
}

int mp_fetch_ms(const char *url, int keep_s, int timeout_ms,
                struct janas_buf *out, char *err, size_t err_len)
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
    /* the sources' rules: a User-Agent naming the program and a way to
       reach its authors (Transitous: its version too); the stock ones of
       libraries are refused */
    static const char *const agent[] = {
        "User-Agent: janas-maps/" JANAS_VERSION
        " (+https://github.com/prabanta-dev/janas)"};
    pace(url);
    int status =
        janas_https_get_ms(url, agent, 1, timeout_ms, out, err, err_len);
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

void mp_fetch_free(void)
{
    for (int i = 0; i < N_KEPT; i++) {
        free(kept[i].url);
        janas_buf_free(&kept[i].body);
    }
    memset(kept, 0, sizeof kept);
}

const char *mp_base(const char *env, const char *def)
{
    const char *v = getenv(env);
    return v && *v ? v : def;
}

void mp_url_put(struct janas_buf *b, const char *s)
{
    static const char hex[] = "0123456789ABCDEF";
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
            janas_buf_put(b, (const char *)&c, 1);
        else
            janas_buf_printf(b, "%%%c%c", hex[c >> 4], hex[c & 15]);
    }
}

const char *mp_lang(void)
{
    static char lang[8];
    static const char *const vars[] = {"LC_ALL", "LC_MESSAGES", "LANG"};
    for (size_t i = 0; i < sizeof vars / sizeof *vars; i++) {
        const char *v = getenv(vars[i]);
        if (!v || !*v)
            continue;
        size_t n = strcspn(v, "_.@");
        int ok = n == 2 && islower((unsigned char)v[0]) &&
                 islower((unsigned char)v[1]);
        if (ok) {
            snprintf(lang, sizeof lang, "%.2s", v);
            return lang;
        }
        break; /* the first one set rules, "C" too */
    }
    return "en";
}
