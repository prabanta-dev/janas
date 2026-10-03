/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * prices.h - janas-prices, prices and finance as an MCP service, for
 * information only: what its parts share. fetch.c asks the sources
 * (answers kept a while, a pause between two requests to a server),
 * place.c finds a place and its country, read.c reads the answers,
 * money.c offers currencies, crypto-assets and shares, macro.c inflation,
 * central banks' rates and electricity, fuel.c the fuel prices of the
 * countries that publish them, data.c and layouts.c write the answers as
 * data with a layout, tools.c offers the tools.
 */
#ifndef JANAS_PRICES_H
#define JANAS_PRICES_H

#include <stddef.h>
#include <time.h>

#include "llm/json.h"

/* ---- fetch.c ---- */

/* GET of url, the body into out; answers of status 200 kept keep_s
   seconds; at most timeout_ms for each part of the answer; pause_s
   between two requests to the same host. The HTTP status, or -1 with the
   reason in err. */
int pr_fetch(const char *url, int keep_s, int timeout_ms, double pause_s,
             struct janas_buf *out, char *err, size_t err_len);
void pr_fetch_free(void);
void pr_url_put(struct janas_buf *b, const char *s);
/* The Alpha Vantage key (JANAS_ALPHAVANTAGE_KEY, else the line
   "alphavantage_key = KEY" of the file); NULL when none. */
void pr_key_config(const char *path);
const char *pr_key(void);
/* a line of the settings file not understood, or "" */
const char *pr_key_problem(void);

/* ---- place.c ---- */

struct pr_place {
    char name[96], country[64], cc[3], region[64];
    char postcode[12]; /* one of its postcodes, when known (Spain's
                          province is its first two digits) */
    double lat, lon;
    char tz[48];  /* its time zone, when known */
    char how[96]; /* where the user is, when no place was named */
};

/* A place by name ("Trapani", "Madrid"), "lat,lon", or, with q NULL,
   where the user is. 0, or -1 with the reason in err. */
int pr_place_find(const char *q, struct pr_place *p, char *err, size_t err_len);

/* ---- read.c: the answers (no network: the tests read kept ones) ---- */

#define PR_RATES 40

struct pr_fx {
    char base[4], date[11];
    struct {
        char code[4];
        double rate;
    } r[PR_RATES];
    int n;
};

struct pr_coin {
    char id[64], name[64], symbol[16], cur[8];
    double price, change_24h, cap;
    time_t at;
};

struct pr_quote {
    char symbol[32], name[96], region[48], currency[8], day[11];
    double price, open, high, low, prev, change, change_pct;
    long volume;
};

#define PR_POINTS 400

struct pr_series { /* a value over time: inflation, a rate, a price */
    char label[96], unit[32];
    struct {
        char when[24]; /* "2026-08", "2026-10-01", "14:00" */
        double v;
    } p[PR_POINTS];
    int n;
};

#define PR_STATIONS 12

struct pr_station {
    char name[96], address[160], city[64];
    double lat, lon, km, price;
    int self; /* 1 self-service, 0 served, -1 not said */
    char updated[24];
};

struct pr_fuel {
    char fuel[24]; /* what the source calls it: "Gasolio", "Gazole" */
    int found;     /* stations within the radius selling it */
    int stale;     /* left out: a price not told for over STALE_DAYS */
    double cheapest, average;
    struct pr_station s[PR_STATIONS];
    int n;
};

/* The fuels asked for, as the tools name them */
enum pr_fuel_kind { PR_PETROL, PR_DIESEL, PR_LPG, PR_CNG };

int pr_read_frankfurter(const char *t, size_t n, struct pr_fx *fx, char *err,
                        size_t err_len);
int pr_read_ecb_xml(const char *t, size_t n, struct pr_fx *fx, char *err,
                    size_t err_len);
/* CoinGecko's search: the id of the coin named (by id, symbol or name,
   the best ranked); 0, or -1 */
int pr_read_coin_search(const char *t, size_t n, const char *q,
                        struct pr_coin *c, char *err, size_t err_len);
/* CoinGecko's simple price for c->id in cur */
int pr_read_coin_price(const char *t, size_t n, struct pr_coin *c, char *err,
                       size_t err_len);
/* Coinbase's spot price */
int pr_read_coinbase(const char *t, size_t n, struct pr_coin *c, char *err,
                     size_t err_len);
/* Alpha Vantage: the best match of a search, a quote */
int pr_read_av_search(const char *t, size_t n, struct pr_quote *q, char *err,
                      size_t err_len);
int pr_read_av_quote(const char *t, size_t n, struct pr_quote *q, char *err,
                     size_t err_len);
/* Eurostat's JSON-stat with one series over time (the last points) */
int pr_read_eurostat(const char *t, size_t n, struct pr_series *s, char *err,
                     size_t err_len);
/* The World Bank's indicator for one country, oldest first */
int pr_read_worldbank(const char *t, size_t n, struct pr_series *s,
                      char *country, size_t cap, char *err, size_t err_len);
/* The ECB's data API in CSV (TIME_PERIOD, OBS_VALUE) */
int pr_read_ecb_csv(const char *t, size_t n, struct pr_series *s, char *err,
                    size_t err_len);
/* The Bank of England's database in CSV (DATE,CODE) */
int pr_read_boe_csv(const char *t, size_t n, struct pr_series *s, char *err,
                    size_t err_len);
/* The Swiss National Bank's cube in JSON */
int pr_read_snb(const char *t, size_t n, struct pr_series *s, char *err,
                size_t err_len);
/* The New York Fed's EFFR, with the FOMC's target range */
int pr_read_nyfed(const char *t, size_t n, struct pr_series *s, double *lo,
                  double *hi, char *err, size_t err_len);
/* energy-charts' day-ahead prices: 1 when open data (licence CC BY), 0
   when not (its licence into licence), -1 on error */
int pr_read_power(const char *t, size_t n, const char *tz, struct pr_series *s,
                  char *licence, size_t cap, char *err, size_t err_len);

/* ---- fuel.c: the stations of each country, nearest the place ---- */

struct pr_fuel_ask {
    double lat, lon, radius_km;
    enum pr_fuel_kind kind;
    time_t now; /* for the prices' age */
};

/* A price told longer ago is left out: a station's of two months before
   came first in Palermo */
#define PR_STALE_DAYS 8

/* Italy (MIMIT): the registry and the prices, both '|'-separated */
int pr_read_fuel_it(const char *registry, size_t rn, const char *prices,
                    size_t pn, const struct pr_fuel_ask *a, struct pr_fuel *f,
                    char *err, size_t err_len);
/* France: the records of the instant flow */
int pr_read_fuel_fr(const char *t, size_t n, const struct pr_fuel_ask *a,
                    struct pr_fuel *f, char *err, size_t err_len);
/* Spain: a province's stations */
int pr_read_fuel_es(const char *t, size_t n, const struct pr_fuel_ask *a,
                    struct pr_fuel *f, char *err, size_t err_len);
/* Austria: E-Control's stations by address */
int pr_read_fuel_at(const char *t, size_t n, const struct pr_fuel_ask *a,
                    struct pr_fuel *f, char *err, size_t err_len);
/* The stations kept: those whose price is not stale, the cheapest
   first, the nearest of equal price */
void pr_fuel_rank(struct pr_fuel *f, struct pr_station *all, int n, time_t now);

/* ---- data.c ---- */

void pr_jstr(struct janas_buf *b, const char *key, const char *v);
void pr_jnum(struct janas_buf *b, const char *key, double v, int decimals);
/* A number with its decimals for a price: 2, 4 or 6 as it is small */
int pr_decimals(double v);
void pr_jplace(struct janas_buf *b, const char *key, const struct pr_place *p);
void pr_fx_data(struct janas_buf *b, double amount, const char *from,
                const struct pr_fx *fx, const char *source);
void pr_coin_data(struct janas_buf *b, const struct pr_coin *c,
                  const char *source);
void pr_quote_data(struct janas_buf *b, const struct pr_quote *q);
/* source and note: words of the layout (src.*, note.*); licence as the
   source gives it */
void pr_series_data(struct janas_buf *b, const char *what, const char *who,
                    const struct pr_series *s, int last, const char *source,
                    const char *note, const char *licence);
/* A central bank's rates: name, value and date of each */
struct pr_rate {
    char name[64], when[24];
    double v;
};
void pr_rates_data(struct janas_buf *b, const char *bank,
                   const struct pr_rate *r, int n, double lo, double hi,
                   const char *source);
void pr_fuel_data(struct janas_buf *b, const struct pr_place *p,
                  const char *kind, double radius, const struct pr_fuel *f,
                  const char *source);

/* ---- layouts.c ---- */

extern const char PR_FX_LAYOUT[], PR_FX_BRIEF[];
extern const char PR_COIN_LAYOUT[], PR_COIN_BRIEF[];
extern const char PR_QUOTE_LAYOUT[], PR_QUOTE_BRIEF[];
extern const char PR_SERIES_LAYOUT[], PR_SERIES_BRIEF[];
extern const char PR_RATES_LAYOUT[], PR_RATES_BRIEF[];
extern const char PR_FUEL_LAYOUT[], PR_FUEL_BRIEF[];

/* ---- tools.c, money.c, macro.c, fuel.c ---- */

void pr_tools_list(void *ctx, struct janas_buf *b);
int pr_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);
const char *pr_arg_str(const struct janas_json *args, const char *name);
double pr_arg_num(const struct janas_json *args, const char *name, double def);
int pr_fail(struct janas_buf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
int pr_answer(struct janas_buf *b, const char *name, struct janas_buf *d,
              const char *layout, const char *brief);
int pr_tool_currency(const struct janas_json *args, struct janas_buf *b);
int pr_tool_crypto(const struct janas_json *args, struct janas_buf *b);
int pr_tool_stock(const struct janas_json *args, struct janas_buf *b);
int pr_tool_inflation(const struct janas_json *args, struct janas_buf *b);
int pr_tool_rates(const struct janas_json *args, struct janas_buf *b);
int pr_tool_power(const struct janas_json *args, struct janas_buf *b);
int pr_tool_fuel(const struct janas_json *args, struct janas_buf *b);

#endif
