/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * history.c - a price over the last days, for the chart of a view (MCP
 * Apps): a currency's (the ECB's reference rates, through Frankfurter), a
 * crypto-asset's (CoinGecko, a day each), a share's (Alpha Vantage's
 * daily closes: one call of the user's key more). Asked only of a client
 * that shows views, or when a request gives "days": a client of text makes
 * no request more. Into "history": {"days", "source", "series": [{"name",
 * "points": [["YYYY-MM-DD", value]...]}]}.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/mcp_server.h"
#include "prices.h"

#define HIST_KEEP_S 3600
#define HIST_MS 20000

int pr_history_days(const struct janas_json *args)
{
    const struct janas_json *v = janas_json_get(args, "days");
    int asked = v && v->type == JANAS_JSON_NUMBER;
    if (!asked && !janas_mcps_client_views())
        return 0;
    int d = asked ? (int)janas_json_num(v, 30) : 30;
    return d < 7 ? 7 : d > 365 ? 365 : d;
}

static void day_of(time_t t, char *out, size_t n)
{
    struct tm g;
    gmtime_r(&t, &g);
    strftime(out, n, "%Y-%m-%d", &g);
}

static void point(struct janas_buf *h, int first, const char *day, size_t dn,
                  double v)
{
    janas_buf_puts(h, first ? "[" : ", [");
    janas_json_write_str(h, day, dn);
    janas_buf_printf(h, ", %.10g]", v);
}

static void head(struct janas_buf *h, int days, const char *source)
{
    janas_buf_printf(h, "{\"days\": %d, \"source\": \"%s\", \"series\": [",
                     days, source);
}

void pr_history_fx(struct janas_buf *h, const char *from, const char *list,
                   int days)
{
    char start[16], url[512], err[200];
    day_of(time(NULL) - (time_t)days * 86400, start, sizeof start);
    snprintf(url, sizeof url,
             "https://api.frankfurter.dev/v1/%s..?base=%s&symbols=%s", start,
             from, list);
    struct janas_buf body = {0};
    if (pr_fetch(url, HIST_KEEP_S, HIST_MS, 0.5, &body, err, sizeof err) !=
        200) {
        janas_buf_free(&body);
        return;
    }
    struct janas_json_doc *d =
        janas_json_parse(body.p, body.n, err, sizeof err);
    const struct janas_json *rates =
        d ? janas_json_get(janas_json_root(d), "rates") : NULL;
    if (rates && rates->type == JANAS_JSON_OBJECT) {
        head(h, days, "ecb_frankfurter");
        /* a series a currency, its days in the order the answer has them
           (ascending) */
        int ns = 0;
        for (const char *p = list; *p;) {
            char code[4] = {0};
            while (*p == ',')
                p++;
            if (strlen(p) < 3)
                break;
            memcpy(code, p, 3);
            p += 3;
            janas_buf_printf(h, "%s{\"name\": \"%s\", \"points\": [",
                             ns++ ? ", " : "", code);
            int first = 1;
            for (const struct janas_json *day = rates->child; day;
                 day = day->next) {
                const struct janas_json *v = janas_json_get(day, code);
                if (v && v->type == JANAS_JSON_NUMBER) {
                    point(h, first, day->key, day->key_n, janas_json_num(v, 0));
                    first = 0;
                }
            }
            janas_buf_puts(h, "]}");
        }
        janas_buf_puts(h, "]}");
    }
    janas_json_free(d);
    janas_buf_free(&body);
}

void pr_history_coin(struct janas_buf *h, const char *id, const char *cur,
                     int days)
{
    char url[512], err[200];
    snprintf(url, sizeof url,
             "https://api.coingecko.com/api/v3/coins/%s/market_chart?"
             "vs_currency=%c%c%c&days=%d&interval=daily",
             id, cur[0] | 0x20, cur[1] | 0x20, cur[2] | 0x20, days);
    struct janas_buf body = {0};
    if (strspn(id, "abcdefghijklmnopqrstuvwxyz0123456789-") != strlen(id) ||
        pr_fetch(url, HIST_KEEP_S, HIST_MS, 2.5, &body, err, sizeof err) !=
            200) {
        janas_buf_free(&body);
        return;
    }
    struct janas_json_doc *d =
        janas_json_parse(body.p, body.n, err, sizeof err);
    const struct janas_json *pr =
        d ? janas_json_get(janas_json_root(d), "prices") : NULL;
    if (pr && pr->type == JANAS_JSON_ARRAY && pr->child) {
        head(h, days, "coingecko");
        janas_buf_printf(h, "{\"name\": \"%s\", \"points\": [", cur);
        int first = 1;
        char last[16] = "";
        for (const struct janas_json *e = pr->child; e; e = e->next) {
            const struct janas_json *t = e->child, *v = t ? t->next : NULL;
            if (!t || !v)
                continue;
            char day[16];
            day_of((time_t)(janas_json_num(t, 0) / 1000), day, sizeof day);
            if (!strcmp(day, last))
                continue; /* today's point after midnight's: the first kept */
            snprintf(last, sizeof last, "%s", day);
            point(h, first, day, strlen(day), janas_json_num(v, 0));
            first = 0;
        }
        janas_buf_puts(h, "]}]}");
    }
    janas_json_free(d);
    janas_buf_free(&body);
}

static int by_key(const void *a, const void *b)
{
    const struct janas_json *x = *(const struct janas_json *const *)a,
                            *y = *(const struct janas_json *const *)b;
    size_t n = x->key_n < y->key_n ? x->key_n : y->key_n;
    int c = memcmp(x->key, y->key, n);
    return c ? c : (x->key_n > y->key_n) - (x->key_n < y->key_n);
}

void pr_history_stock(struct janas_buf *h, const char *symbol, int days)
{
    if (!pr_key())
        return;
    struct janas_buf url = {0}, body = {0};
    char err[200];
    janas_buf_puts(&url, "https://www.alphavantage.co/query?function="
                         "TIME_SERIES_DAILY&symbol=");
    pr_url_put(&url, symbol);
    janas_buf_printf(&url, "&apikey=%s", pr_key());
    int st = url.oom ? -1
                     : pr_fetch(url.p, HIST_KEEP_S, HIST_MS, 1.5, &body, err,
                                sizeof err);
    janas_buf_free(&url);
    if (st != 200) {
        janas_buf_free(&body);
        return;
    }
    struct janas_json_doc *d =
        janas_json_parse(body.p, body.n, err, sizeof err);
    const struct janas_json *ts =
        d ? janas_json_get(janas_json_root(d), "Time Series (Daily)") : NULL;
    size_t n = 0;
    for (const struct janas_json *e =
             ts && ts->type == JANAS_JSON_OBJECT ? ts->child : NULL;
         e; e = e->next)
        n++;
    const struct janas_json **v = n ? malloc(n * sizeof *v) : NULL;
    if (v) {
        size_t k = 0;
        for (const struct janas_json *e = ts->child; e; e = e->next)
            v[k++] = e;
        qsort(v, n, sizeof *v, by_key); /* oldest first */
        size_t from = n > (size_t)days ? n - (size_t)days : 0;
        head(h, days, "alphavantage");
        janas_buf_printf(h, "{\"name\": \"%s\", \"points\": [", symbol);
        int first = 1;
        for (size_t i = from; i < n; i++) {
            const char *c = janas_json_str(janas_json_get(v[i], "4. close"));
            if (!c)
                continue;
            point(h, first, v[i]->key, v[i]->key_n, strtod(c, NULL));
            first = 0;
        }
        janas_buf_puts(h, "]}]}");
        free(v);
    }
    janas_json_free(d);
    janas_buf_free(&body);
}

void pr_with_history(struct janas_buf *data, const struct janas_buf *h)
{
    if (!h->n || h->oom || !data->n || data->p[data->n - 1] != '}')
        return;
    data->n--; /* its "}" */
    janas_buf_puts(data, ", \"history\": ");
    janas_buf_put(data, h->p, h->n);
    janas_buf_puts(data, "}");
}
