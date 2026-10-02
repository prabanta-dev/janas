/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * fetch.c - janas-prices asks its sources (see prices.h): a User-Agent
 * naming the program, a pause between two requests to the same server,
 * the answers kept a while (a day's rates, an hour's prices, two minutes
 * of a crypto-asset's), so that a question asked again costs the sources
 * nothing. The key of Alpha Vantage, when the user has one, is read here
 * and never written anywhere.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "services/common/https_get.h"
#include "prices.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

#define N_KEPT 32
#define N_HOSTS 12

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

static char key[128];

static double since(const struct timespec *t)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - t->tv_sec) +
           (double)(now.tv_nsec - t->tv_nsec) * 1e-9;
}

static void pace(const char *url, double pause_s)
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
        double wait = pause_s - since(&hosts[slot].last);
        if (wait > 0 && wait <= pause_s) {
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

int pr_fetch(const char *url, int keep_s, int timeout_ms, double pause_s,
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
    static const char *const agent[] = {
        "User-Agent: janas-prices/" JANAS_VERSION
        " (+https://github.com/prabanta-dev/janas)"};
    pace(url, pause_s);
    int status =
        janas_https_get_ms(url, agent, 1, timeout_ms, out, err, err_len);
    if (status == 200 && !out->oom && keep_s > 0) {
        free(kept[slot].url);
        janas_buf_free(&kept[slot].body);
        kept[slot].url = strdup(url);
        kept[slot].at = now;
        kept[slot].keep_s = keep_s;
        janas_buf_put(&kept[slot].body, out->p, out->n);
    }
    return status;
}

void pr_fetch_free(void)
{
    for (int i = 0; i < N_KEPT; i++) {
        free(kept[i].url);
        janas_buf_free(&kept[i].body);
    }
    memset(kept, 0, sizeof kept);
    memset(key, 0, sizeof key);
}

void pr_url_put(struct janas_buf *b, const char *s)
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

/* A key is letters and digits: anything else is not one */
static void take(const char *s)
{
    size_t n = 0;
    while (s && isalnum((unsigned char)s[n]) && n + 1 < sizeof key)
        n++;
    if (n >= 8) {
        memcpy(key, s, n);
        key[n] = 0;
    }
}

void pr_key_config(const char *path)
{
    key[0] = 0;
    take(getenv("JANAS_ALPHAVANTAGE_KEY"));
    if (key[0] || !path)
        return;
    FILE *f = fopen(path, "r");
    if (!f)
        return;
    char line[512];
    while (!key[0] && fgets(line, sizeof line, f)) {
        if (strncmp(line, "alphavantage_key", 16) != 0)
            continue;
        const char *v = strchr(line, '=');
        if (v) {
            v++;
            while (*v == ' ' || *v == '\t')
                v++;
            take(v);
        }
    }
    memset(line, 0, sizeof line);
    fclose(f);
}

const char *pr_key(void)
{
    return key[0] ? key : NULL;
}
