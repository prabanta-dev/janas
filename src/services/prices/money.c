/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * money.c - janas-prices' currencies, crypto-assets and shares (see
 * prices.h). Currencies at the ECB's reference rates through Frankfurter
 * (any base, any day), the ECB's own file when it does not answer;
 * crypto-assets from CoinGecko's keyless API, Coinbase's spot price when
 * it does not answer; shares from Alpha Vantage with a free key of the
 * user's, the only tool that needs one.
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "prices.h"

#define FX_KEEP_S 3600
#define COIN_KEEP_S 120
#define SEARCH_KEEP_S 86400
#define QUOTE_KEEP_S 300
#define MS 20000

/* "usd" -> "USD"; 0 when not three letters */
static int code3(const char *s, char *out)
{
    for (int i = 0; i < 3; i++) {
        if (!s || !isalpha((unsigned char)s[i]))
            return 0;
        out[i] = (char)toupper((unsigned char)s[i]);
    }
    out[3] = 0;
    return !isalnum((unsigned char)s[3]);
}

int pr_tool_currency(const struct janas_json *args, struct janas_buf *b)
{
    char from[4] = "EUR", err[400] = "";
    const char *f = pr_arg_str(args, "from");
    if (f && !code3(f, from))
        return pr_fail(b, "\"%s\" is not a currency's ISO code (EUR, USD).", f);
    /* the codes asked, else the most asked */
    const char *to = pr_arg_str(args, "to");
    char list[160] = "";
    size_t k = 0;
    for (const char *p = to ? to : "USD,GBP,CHF,JPY"; *p;) {
        char c[4];
        while (*p == ',' || *p == ' ')
            p++;
        if (!*p)
            break;
        if (!code3(p, c))
            return pr_fail(b, "\"%s\" is not a list of ISO codes.", to);
        if (strcmp(c, from) && k + 5 < sizeof list)
            k += (size_t)snprintf(list + k, sizeof list - k, "%s%s",
                                  k ? "," : "", c);
        p += 3;
    }
    if (!k)
        return pr_fail(b, "No currency to convert to.");
    double amount = pr_arg_num(args, "amount", 1);
    const char *date = pr_arg_str(args, "date");
    int y, m, d;
    if (date && sscanf(date, "%4d-%2d-%2d", &y, &m, &d) != 3)
        return pr_fail(b, "\"%s\" is not a date (YYYY-MM-DD).", date);
    char url[512];
    snprintf(url, sizeof url,
             "https://api.frankfurter.dev/v1/%s?base=%s&symbols=%s",
             date ? date : "latest", from, list);
    struct janas_buf body = {0};
    struct pr_fx fx;
    const char *source = "ecb_frankfurter";
    int ok = pr_fetch(url, FX_KEEP_S, MS, 0.5, &body, err, sizeof err) == 200 &&
             pr_read_frankfurter(body.p, body.n, &fx, err, sizeof err) == 0;
    if (!ok && !date) { /* the ECB's file, in euro: crossed */
        char e2[300];
        struct pr_fx eu;
        if (pr_fetch("https://www.ecb.europa.eu/stats/eurofxref/"
                     "eurofxref-daily.xml",
                     FX_KEEP_S, MS, 0.5, &body, e2, sizeof e2) == 200 &&
            pr_read_ecb_xml(body.p, body.n, &eu, e2, sizeof e2) == 0) {
            double base = strcmp(from, "EUR") ? NAN : 1;
            for (int i = 0; i < eu.n; i++)
                if (!strcmp(eu.r[i].code, from))
                    base = eu.r[i].rate;
            memset(&fx, 0, sizeof fx);
            snprintf(fx.base, sizeof fx.base, "%s", from);
            snprintf(fx.date, sizeof fx.date, "%s", eu.date);
            for (int i = 0; i < eu.n && !isnan(base); i++)
                if (strstr(list, eu.r[i].code)) {
                    fx.r[fx.n] = eu.r[i];
                    fx.r[fx.n++].rate = eu.r[i].rate / base;
                }
            if (!strcmp(from, "EUR") || !isnan(base)) {
                if (strstr(list, "EUR") && strcmp(from, "EUR")) {
                    snprintf(fx.r[fx.n].code, 4, "EUR");
                    fx.r[fx.n++].rate = 1 / base;
                }
                ok = fx.n > 0;
                source = "ecb";
            }
        }
    }
    janas_buf_free(&body);
    if (!ok)
        return pr_fail(b, "The rates could not be read: %s.", err);
    struct janas_buf data = {0};
    pr_fx_data(&data, amount, from, &fx, source);
    return pr_answer(b, "prices_currency", &data, PR_FX_LAYOUT, PR_FX_BRIEF);
}

int pr_tool_crypto(const struct janas_json *args, struct janas_buf *b)
{
    const char *q = pr_arg_str(args, "coin");
    if (!q)
        return pr_fail(b, "No crypto-asset named.");
    char cur[4] = "EUR", err[400] = "";
    const char *c = pr_arg_str(args, "currency");
    if (c && !code3(c, cur))
        return pr_fail(b, "\"%s\" is not a currency's ISO code.", c);
    struct pr_coin coin;
    struct janas_buf url = {0}, body = {0};
    janas_buf_puts(&url, "https://api.coingecko.com/api/v3/search?query=");
    pr_url_put(&url, q);
    int found =
        !url.oom &&
        pr_fetch(url.p, SEARCH_KEEP_S, MS, 2.5, &body, err, sizeof err) ==
            200 &&
        pr_read_coin_search(body.p, body.n, q, &coin, err, sizeof err) == 0;
    janas_buf_free(&url);
    int ok = 0;
    const char *source = "coingecko";
    if (found) {
        snprintf(coin.cur, sizeof coin.cur, "%s", cur);
        janas_buf_printf(&url,
                         "https://api.coingecko.com/api/v3/simple/price?ids=%s"
                         "&vs_currencies=%c%c%c&include_24hr_change=true"
                         "&include_market_cap=true&include_last_updated_at="
                         "true",
                         coin.id, cur[0] | 0x20, cur[1] | 0x20, cur[2] | 0x20);
        ok = !url.oom &&
             pr_fetch(url.p, COIN_KEEP_S, MS, 2.5, &body, err, sizeof err) ==
                 200 &&
             pr_read_coin_price(body.p, body.n, &coin, err, sizeof err) == 0;
        janas_buf_free(&url);
    }
    if (!ok) { /* Coinbase, by the symbol */
        char e2[300], sym[16];
        snprintf(sym, sizeof sym, "%s", found ? coin.symbol : q);
        for (char *s = sym; *s; s++)
            *s = (char)toupper((unsigned char)*s);
        if (!found) {
            memset(&coin, 0, sizeof coin);
            snprintf(coin.name, sizeof coin.name, "%s", sym);
        }
        snprintf(coin.symbol, sizeof coin.symbol, "%s", sym);
        snprintf(coin.cur, sizeof coin.cur, "%s", cur);
        janas_buf_printf(&url, "https://api.coinbase.com/v2/prices/%s-%s/spot",
                         sym, cur);
        ok = strspn(sym, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") ==
                 strlen(sym) &&
             !url.oom &&
             pr_fetch(url.p, COIN_KEEP_S, MS, 1, &body, e2, sizeof e2) == 200 &&
             pr_read_coinbase(body.p, body.n, &coin, e2, sizeof e2) == 0;
        janas_buf_free(&url);
        source = "coinbase";
    }
    janas_buf_free(&body);
    if (!ok)
        return pr_fail(b, "The price of %s could not be read: %s.", q, err);
    for (char *s = coin.symbol; *s; s++)
        *s = (char)toupper((unsigned char)*s);
    struct janas_buf data = {0};
    pr_coin_data(&data, &coin, source);
    return pr_answer(b, "prices_crypto", &data, PR_COIN_LAYOUT, PR_COIN_BRIEF);
}

/* A symbol as Alpha Vantage writes them: "IBM", "TSCO.LON", "BRK-B" */
static int symbol_like(const char *s)
{
    size_t n = strlen(s);
    if (!n || n > 16)
        return 0;
    for (size_t i = 0; i < n; i++)
        if (!isupper((unsigned char)s[i]) && !isdigit((unsigned char)s[i]) &&
            s[i] != '.' && s[i] != '-')
            return 0;
    return 1;
}

static int quote(const char *symbol, struct pr_quote *q, char *err,
                 size_t err_len)
{
    struct janas_buf url = {0}, body = {0};
    janas_buf_puts(&url,
                   "https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
                   "&symbol=");
    pr_url_put(&url, symbol);
    janas_buf_printf(&url, "&apikey=%s", pr_key());
    int ok =
        !url.oom &&
        pr_fetch(url.p, QUOTE_KEEP_S, MS, 1.5, &body, err, err_len) == 200 &&
        pr_read_av_quote(body.p, body.n, q, err, err_len) == 0;
    janas_buf_free(&url);
    janas_buf_free(&body);
    return ok ? 0 : -1;
}

int pr_tool_stock(const struct janas_json *args, struct janas_buf *b)
{
    const char *s = pr_arg_str(args, "share");
    if (!s)
        return pr_fail(b, "No share named.");
    if (!pr_key())
        return pr_fail(
            b,
            "Share prices need a free key of Alpha Vantage, which the user "
            "has not given: it is had at "
            "https://www.alphavantage.co/support/#api-key and goes in "
            "~/.config/janas/prices.conf as the line  alphavantage_key = "
            "KEY  (or in JANAS_ALPHAVANTAGE_KEY).%s%s Tell the user so.",
            pr_key_problem()[0] ? " That file has a line not understood: " : "",
            pr_key_problem());
    struct pr_quote q;
    char err[400] = "";
    memset(&q, 0, sizeof q);
    int ok = symbol_like(s) && quote(s, &q, err, sizeof err) == 0;
    if (!ok) { /* by the company's name */
        struct janas_buf url = {0}, body = {0};
        janas_buf_puts(&url, "https://www.alphavantage.co/query?function="
                             "SYMBOL_SEARCH&keywords=");
        pr_url_put(&url, s);
        janas_buf_printf(&url, "&apikey=%s", pr_key());
        struct pr_quote found;
        ok = !url.oom &&
             pr_fetch(url.p, SEARCH_KEEP_S, MS, 1.5, &body, err, sizeof err) ==
                 200 &&
             pr_read_av_search(body.p, body.n, &found, err, sizeof err) == 0 &&
             quote(found.symbol, &q, err, sizeof err) == 0;
        if (ok) {
            snprintf(q.name, sizeof q.name, "%s", found.name);
            snprintf(q.region, sizeof q.region, "%s", found.region);
            snprintf(q.currency, sizeof q.currency, "%s", found.currency);
        }
        janas_buf_free(&url);
        janas_buf_free(&body);
    }
    if (!ok)
        return pr_fail(b, "The price of %s could not be read: %s.", s, err);
    if (!q.name[0])
        snprintf(q.name, sizeof q.name, "%s", q.symbol);
    struct janas_buf data = {0};
    pr_quote_data(&data, &q);
    return pr_answer(b, "prices_stock", &data, PR_QUOTE_LAYOUT, PR_QUOTE_BRIEF);
}
