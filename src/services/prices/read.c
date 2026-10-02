/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * read.c - janas-prices reads its sources' answers (see prices.h): the
 * rates of currencies (Frankfurter, the ECB), the prices of crypto-assets
 * (CoinGecko, Coinbase), shares (Alpha Vantage), inflation (Eurostat, the
 * World Bank), central banks' rates (the ECB, the Bank of England, the
 * Swiss National Bank, the New York Fed) and electricity (energy-charts).
 * The fuels are in read_fuel.c. No network here, so that the tests can
 * read answers kept as they came.
 */
#define _GNU_SOURCE /* strcasecmp, timegm */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "prices.h"

static void copy(char *to, size_t cap, const char *s)
{
    snprintf(to, cap, "%s", s ? s : "");
}

static const char *str(const struct janas_json *o, const char *key)
{
    const char *s = janas_json_str(janas_json_get(o, key));
    return s && *s ? s : NULL;
}

/* a number given as a number or as a text ("225.6200", "2.5872%") */
static double num(const struct janas_json *o, const char *key)
{
    const struct janas_json *v = janas_json_get(o, key);
    if (v && v->type == JANAS_JSON_STRING) {
        char *end;
        double d = strtod(v->s, &end);
        return end > v->s ? d : NAN;
    }
    return janas_json_num(v, NAN);
}

static struct janas_json_doc *parse(const char *t, size_t n, const char *who,
                                    char *err, size_t err_len)
{
    char why[128];
    struct janas_json_doc *d = janas_json_parse(t, n, why, sizeof why);
    if (!d)
        snprintf(err, err_len, "%s: an answer it could not read (%s)", who,
                 why);
    return d;
}

static void add(struct pr_series *s, const char *when, double v)
{
    if (s->n < PR_POINTS && !isnan(v)) {
        copy(s->p[s->n].when, sizeof s->p[s->n].when, when);
        s->p[s->n++].v = v;
    }
}

/* ---- currencies ---- */

int pr_read_frankfurter(const char *t, size_t n, struct pr_fx *fx, char *err,
                        size_t err_len)
{
    memset(fx, 0, sizeof *fx);
    struct janas_json_doc *d = parse(t, n, "Frankfurter", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *o = janas_json_root(d);
    copy(fx->base, sizeof fx->base, str(o, "base"));
    copy(fx->date, sizeof fx->date, str(o, "date"));
    const struct janas_json *r = janas_json_get(o, "rates");
    for (const struct janas_json *x = r ? r->child : NULL;
         x && fx->n < PR_RATES; x = x->next) {
        snprintf(fx->r[fx->n].code, sizeof fx->r[fx->n].code, "%.*s",
                 (int)x->key_n, x->key);
        fx->r[fx->n++].rate = janas_json_num(x, NAN);
    }
    const char *m = str(o, "message");
    janas_json_free(d);
    if (!fx->base[0] || !fx->n) {
        snprintf(err, err_len, "Frankfurter: %s", m ? m : "no rates");
        return -1;
    }
    return 0;
}

/* <Cube time='2026-10-02'> ... <Cube currency='USD' rate='1.1225'/> */
int pr_read_ecb_xml(const char *t, size_t n, struct pr_fx *fx, char *err,
                    size_t err_len)
{
    memset(fx, 0, sizeof *fx);
    snprintf(fx->base, sizeof fx->base, "EUR");
    const char *e = t + n, *p = t;
    const char *tm = memmem(t, n, "time='", 6);
    if (tm && tm + 16 <= e)
        snprintf(fx->date, sizeof fx->date, "%.10s", tm + 6);
    while ((p = memmem(p, (size_t)(e - p), "currency='", 10)) &&
           fx->n < PR_RATES) {
        p += 10;
        const char *q = memchr(p, '\'', (size_t)(e - p));
        const char *r = q ? memmem(q, (size_t)(e - q), "rate='", 6) : NULL;
        if (!q || !r || q - p != 3)
            break;
        memcpy(fx->r[fx->n].code, p, 3);
        fx->r[fx->n].code[3] = 0;
        fx->r[fx->n++].rate = strtod(r + 6, NULL);
        p = r + 6;
    }
    if (!fx->n || !fx->date[0]) {
        snprintf(err, err_len, "the ECB: no rates in its answer");
        return -1;
    }
    return 0;
}

/* ---- crypto-assets ---- */

int pr_read_coin_search(const char *t, size_t n, const char *q,
                        struct pr_coin *c, char *err, size_t err_len)
{
    memset(c, 0, sizeof *c);
    struct janas_json_doc *d = parse(t, n, "CoinGecko", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *coins =
        janas_json_get(janas_json_root(d), "coins");
    /* the id itself, else the symbol or the name of the best ranked (a
       symbol is taken by many tokens: "SOL" by a hundred) */
    const struct janas_json *best = NULL;
    int best_score = 0;
    double best_rank = 1e9;
    for (const struct janas_json *x = coins ? coins->child : NULL; x;
         x = x->next) {
        const char *id = str(x, "id"), *sym = str(x, "symbol"),
                   *name = str(x, "name");
        int score = id && !strcasecmp(id, q)       ? 3
                    : sym && !strcasecmp(sym, q)   ? 2
                    : name && !strcasecmp(name, q) ? 2
                                                   : 0;
        double rank = num(x, "market_cap_rank");
        if (isnan(rank))
            rank = 1e8;
        if (score > best_score ||
            (score == best_score && score && rank < best_rank)) {
            best = x;
            best_score = score;
            best_rank = rank;
        }
    }
    if (!best && coins && coins->child)
        best = coins->child;
    if (best) {
        copy(c->id, sizeof c->id, str(best, "id"));
        copy(c->name, sizeof c->name, str(best, "name"));
        copy(c->symbol, sizeof c->symbol, str(best, "symbol"));
    } else {
        snprintf(err, err_len, "CoinGecko knows no crypto-asset named %s", q);
    }
    janas_json_free(d);
    return best ? 0 : -1;
}

int pr_read_coin_price(const char *t, size_t n, struct pr_coin *c, char *err,
                       size_t err_len)
{
    struct janas_json_doc *d = parse(t, n, "CoinGecko", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *o = janas_json_get(janas_json_root(d), c->id);
    char k[48], lc[8];
    size_t i = 0;
    for (; c->cur[i] && i + 1 < sizeof lc; i++)
        lc[i] = (char)(c->cur[i] | 0x20);
    lc[i] = 0;
    c->price = num(o, lc);
    snprintf(k, sizeof k, "%s_24h_change", lc);
    c->change_24h = num(o, k);
    snprintf(k, sizeof k, "%s_market_cap", lc);
    c->cap = num(o, k);
    double at = num(o, "last_updated_at");
    c->at = isnan(at) ? -1 : (time_t)at;
    janas_json_free(d);
    if (isnan(c->price)) {
        snprintf(err, err_len, "CoinGecko gave no price of %s in %s", c->id,
                 c->cur);
        return -1;
    }
    return 0;
}

int pr_read_coinbase(const char *t, size_t n, struct pr_coin *c, char *err,
                     size_t err_len)
{
    struct janas_json_doc *d = parse(t, n, "Coinbase", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *o = janas_json_get(janas_json_root(d), "data");
    c->price = num(o, "amount");
    c->change_24h = c->cap = NAN;
    c->at = time(NULL);
    janas_json_free(d);
    if (isnan(c->price)) {
        snprintf(err, err_len, "Coinbase gave no price");
        return -1;
    }
    return 0;
}

/* ---- shares ---- */

/* Alpha Vantage says what went wrong in a member of its own */
static int av_error(const struct janas_json *o, char *err, size_t err_len)
{
    const char *m = str(o, "Error Message");
    if (!m)
        m = str(o, "Note");
    if (!m)
        m = str(o, "Information");
    if (m)
        snprintf(err, err_len, "Alpha Vantage: %.300s", m);
    return m != NULL;
}

int pr_read_av_search(const char *t, size_t n, struct pr_quote *q, char *err,
                      size_t err_len)
{
    memset(q, 0, sizeof *q);
    struct janas_json_doc *d = parse(t, n, "Alpha Vantage", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *o = janas_json_root(d);
    const struct janas_json *m = janas_json_get(o, "bestMatches");
    const struct janas_json *x = m ? m->child : NULL;
    int r = -1;
    if (x) {
        copy(q->symbol, sizeof q->symbol, str(x, "1. symbol"));
        copy(q->name, sizeof q->name, str(x, "2. name"));
        copy(q->region, sizeof q->region, str(x, "4. region"));
        copy(q->currency, sizeof q->currency, str(x, "8. currency"));
        r = q->symbol[0] ? 0 : -1;
    } else if (!av_error(o, err, err_len)) {
        snprintf(err, err_len, "Alpha Vantage found no share of that name");
    }
    janas_json_free(d);
    return r;
}

int pr_read_av_quote(const char *t, size_t n, struct pr_quote *q, char *err,
                     size_t err_len)
{
    struct janas_json_doc *d = parse(t, n, "Alpha Vantage", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *o = janas_json_root(d);
    const struct janas_json *g = janas_json_get(o, "Global Quote");
    int r = -1;
    if (g && str(g, "01. symbol")) {
        copy(q->symbol, sizeof q->symbol, str(g, "01. symbol"));
        q->open = num(g, "02. open");
        q->high = num(g, "03. high");
        q->low = num(g, "04. low");
        q->price = num(g, "05. price");
        double v = num(g, "06. volume");
        q->volume = isnan(v) ? -1 : (long)v;
        copy(q->day, sizeof q->day, str(g, "07. latest trading day"));
        q->prev = num(g, "08. previous close");
        q->change = num(g, "09. change");
        q->change_pct = num(g, "10. change percent");
        r = isnan(q->price) ? -1 : 0;
        if (r)
            snprintf(err, err_len, "Alpha Vantage gave no price");
    } else if (!av_error(o, err, err_len)) {
        snprintf(err, err_len, "Alpha Vantage has no quote of that symbol");
    }
    janas_json_free(d);
    return r;
}

/* ---- inflation ---- */

/* JSON-stat: one series, the time its only dimension of more than one
   value; "value" maps the positions to the numbers */
int pr_read_eurostat(const char *t, size_t n, struct pr_series *s, char *err,
                     size_t err_len)
{
    memset(s, 0, sizeof *s);
    struct janas_json_doc *d = parse(t, n, "Eurostat", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *o = janas_json_root(d);
    copy(s->label, sizeof s->label, str(o, "label"));
    const struct janas_json *tm = janas_json_get(
        janas_json_get(janas_json_get(janas_json_get(o, "dimension"), "time"),
                       "category"),
        "index");
    const struct janas_json *val = janas_json_get(o, "value");
    for (const struct janas_json *x = tm ? tm->child : NULL; x; x = x->next) {
        char k[16], when[24];
        snprintf(k, sizeof k, "%.0f", janas_json_num(x, -1));
        snprintf(when, sizeof when, "%.*s", (int)x->key_n, x->key);
        add(s, when, janas_json_num(janas_json_get(val, k), NAN));
    }
    const struct janas_json *e = janas_json_get(o, "error");
    const char *m = e ? str(e, "label") : NULL;
    janas_json_free(d);
    if (!s->n) {
        snprintf(err, err_len, "Eurostat: %s", m ? m : "no values");
        return -1;
    }
    return 0;
}

int pr_read_worldbank(const char *t, size_t n, struct pr_series *s,
                      char *country, size_t cap, char *err, size_t err_len)
{
    memset(s, 0, sizeof *s);
    struct janas_json_doc *d = parse(t, n, "the World Bank", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *a = janas_json_root(d);
    const struct janas_json *rows = a && a->child ? a->child->next : NULL;
    /* newest first: kept, then turned */
    const char *first = NULL; /* one country's rows: the first's */
    for (const struct janas_json *x = rows ? rows->child : NULL; x;
         x = x->next) {
        const char *cn = str(janas_json_get(x, "country"), "value");
        if (!first)
            first = cn;
        if (!cn || !first || strcmp(cn, first) != 0)
            continue;
        if (!country[0])
            copy(country, cap, cn);
        if (!s->label[0])
            copy(s->label, sizeof s->label,
                 str(janas_json_get(x, "indicator"), "value"));
        add(s, str(x, "date"), num(x, "value"));
    }
    for (int i = 0; i < s->n / 2; i++) {
        __typeof__(s->p[0]) tmp = s->p[i];
        s->p[i] = s->p[s->n - 1 - i];
        s->p[s->n - 1 - i] = tmp;
    }
    janas_json_free(d);
    if (!s->n) {
        snprintf(err, err_len, "the World Bank has no values for it");
        return -1;
    }
    return 0;
}

/* ---- central banks ---- */

/* The fields of a CSV line, quotes taken into account: field k into
   out. 0, or -1 when there are fewer. */
static int csv_field(const char *line, size_t len, int k, char *out, size_t cap)
{
    int f = 0, q = 0;
    size_t o = 0;
    for (size_t i = 0; i <= len; i++) {
        char c = i < len ? line[i] : ',';
        if (c == '"') {
            q = !q;
            continue;
        }
        if (c == ',' && !q) {
            if (f == k) {
                out[o < cap ? o : cap - 1] = 0;
                return 0;
            }
            f++;
            o = 0;
            continue;
        }
        if (f == k && o + 1 < cap)
            out[o++] = c;
    }
    return -1;
}

static int csv_column(const char *head, size_t len, const char *name)
{
    char f[64];
    for (int k = 0; csv_field(head, len, k, f, sizeof f) == 0; k++)
        if (strcmp(f, name) == 0)
            return k;
    return -1;
}

int pr_read_ecb_csv(const char *t, size_t n, struct pr_series *s, char *err,
                    size_t err_len)
{
    memset(s, 0, sizeof *s);
    const char *p = t, *e = t + n;
    const char *nl = memchr(p, '\n', n);
    if (!nl) {
        snprintf(err, err_len, "the ECB: no data");
        return -1;
    }
    size_t hl = (size_t)(nl - p);
    int ct = csv_column(p, hl, "TIME_PERIOD"),
        cv = csv_column(p, hl, "OBS_VALUE"), cl = csv_column(p, hl, "TITLE");
    if (ct < 0 || cv < 0) {
        snprintf(err, err_len, "the ECB: not its CSV");
        return -1;
    }
    for (p = nl + 1; p < e;) {
        nl = memchr(p, '\n', (size_t)(e - p));
        size_t len = nl ? (size_t)(nl - p) : (size_t)(e - p);
        char when[24], v[32];
        if (len && csv_field(p, len, ct, when, sizeof when) == 0 &&
            csv_field(p, len, cv, v, sizeof v) == 0 && v[0]) {
            add(s, when, strtod(v, NULL));
            if (!s->label[0] && cl >= 0)
                csv_field(p, len, cl, s->label, sizeof s->label);
        }
        p = nl ? nl + 1 : e;
    }
    if (!s->n) {
        snprintf(err, err_len, "the ECB: no values");
        return -1;
    }
    return 0;
}

/* "01 Sep 2026" -> "2026-09-01" */
static void boe_date(const char *d, char *out, size_t cap)
{
    static const char mon[] = "JanFebMarAprMayJunJulAugSepOctNovDec";
    int day, year;
    char m[4];
    if (sscanf(d, "%d %3s %d", &day, m, &year) == 3) {
        const char *at = strstr(mon, m);
        if (at && (at - mon) % 3 == 0) {
            snprintf(out, cap, "%04d-%02d-%02d", year, (int)(at - mon) / 3 + 1,
                     day);
            return;
        }
    }
    copy(out, cap, d);
}

int pr_read_boe_csv(const char *t, size_t n, struct pr_series *s, char *err,
                    size_t err_len)
{
    memset(s, 0, sizeof *s);
    const char *p = t, *e = t + n;
    const char *nl = memchr(p, '\n', n);
    if (!nl || strncmp(p, "DATE,", 5) != 0) {
        snprintf(err, err_len, "the Bank of England: not its CSV");
        return -1;
    }
    for (p = nl + 1; p < e;) {
        nl = memchr(p, '\n', (size_t)(e - p));
        size_t len = nl ? (size_t)(nl - p) : (size_t)(e - p);
        char d[24], v[32], when[24];
        if (csv_field(p, len, 0, d, sizeof d) == 0 &&
            csv_field(p, len, 1, v, sizeof v) == 0 && v[0] && v[0] != '\r') {
            boe_date(d, when, sizeof when);
            add(s, when, strtod(v, NULL));
        }
        p = nl ? nl + 1 : e;
    }
    if (!s->n) {
        snprintf(err, err_len, "the Bank of England: no values");
        return -1;
    }
    return 0;
}

int pr_read_snb(const char *t, size_t n, struct pr_series *s, char *err,
                size_t err_len)
{
    memset(s, 0, sizeof *s);
    struct janas_json_doc *d = parse(t, n, "the SNB", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *ts =
        janas_json_get(janas_json_root(d), "timeseries");
    const struct janas_json *x = ts ? ts->child : NULL;
    const struct janas_json *v = janas_json_get(x, "values");
    for (const struct janas_json *p = v ? v->child : NULL; p; p = p->next)
        add(s, str(p, "date"), num(p, "value"));
    janas_json_free(d);
    if (!s->n) {
        snprintf(err, err_len, "the SNB: no values");
        return -1;
    }
    return 0;
}

int pr_read_nyfed(const char *t, size_t n, struct pr_series *s, double *lo,
                  double *hi, char *err, size_t err_len)
{
    memset(s, 0, sizeof *s);
    *lo = *hi = NAN;
    struct janas_json_doc *d = parse(t, n, "the New York Fed", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *r = janas_json_get(janas_json_root(d), "refRates");
    /* newest first */
    int k = 0;
    for (const struct janas_json *x = r ? r->child : NULL; x; x = x->next)
        k++;
    for (int i = k - 1; i >= 0; i--) {
        const struct janas_json *x = r->child;
        for (int j = 0; j < i; j++)
            x = x->next;
        add(s, str(x, "effectiveDate"), num(x, "percentRate"));
        if (i == 0) {
            *lo = num(x, "targetRateFrom");
            *hi = num(x, "targetRateTo");
        }
    }
    janas_json_free(d);
    if (!s->n) {
        snprintf(err, err_len, "the New York Fed: no values");
        return -1;
    }
    return 0;
}

/* ---- electricity ---- */

int pr_read_power(const char *t, size_t n, const char *tz, struct pr_series *s,
                  char *licence, size_t cap, char *err, size_t err_len)
{
    memset(s, 0, sizeof *s);
    licence[0] = 0;
    struct janas_json_doc *d = parse(t, n, "energy-charts", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *o = janas_json_root(d);
    copy(licence, cap, str(o, "license_info"));
    copy(s->unit, sizeof s->unit, str(o, "unit"));
    int open = strstr(licence, "CC BY") != NULL;
    const struct janas_json *ts = janas_json_get(o, "unix_seconds");
    const struct janas_json *pr = janas_json_get(o, "price");
    char *old = getenv("TZ") ? strdup(getenv("TZ")) : NULL;
    if (tz && *tz)
        setenv("TZ", tz, 1);
    tzset();
    const struct janas_json *x = ts ? ts->child : NULL;
    const struct janas_json *y = pr ? pr->child : NULL;
    for (; open && x && y; x = x->next, y = y->next) {
        time_t at = (time_t)janas_json_num(x, 0);
        struct tm lt;
        localtime_r(&at, &lt);
        char when[24];
        strftime(when, sizeof when, "%Y-%m-%d %H:%M", &lt);
        add(s, when, janas_json_num(y, NAN));
    }
    if (tz && *tz) {
        if (old)
            setenv("TZ", old, 1);
        else
            unsetenv("TZ");
        tzset();
    }
    free(old);
    janas_json_free(d);
    if (!licence[0] || (open && !s->n)) {
        snprintf(err, err_len, "energy-charts: no prices");
        return -1;
    }
    return open;
}
