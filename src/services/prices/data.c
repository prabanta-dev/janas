/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * data.c - what janas-prices read, as the data of its layouts (see
 * prices.h, layouts.c and services/common/template.h). Every field is written,
 * even empty: a layout looks a name up in the item, then in what holds it.
 * Numbers keep the decimals a price of their size needs.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "services/common/geo.h"
#include "prices.h"

void pr_jstr(struct janas_buf *b, const char *key, const char *v)
{
    if (!v)
        v = "";
    janas_buf_printf(b, ", \"%s\": ", key);
    janas_json_write_str(b, v, strlen(v));
}

/* NAN as 0 with "has_<key>": false, so that a layout can leave it out */
void pr_jnum(struct janas_buf *b, const char *key, double v, int decimals)
{
    janas_buf_printf(b, ", \"%s\": %.*f, \"has_%s\": %s", key, decimals,
                     isnan(v) ? 0 : v, key, isnan(v) ? "false" : "true");
}

int pr_decimals(double v)
{
    v = fabs(v);
    return v >= 1 ? 2 : v >= 0.01 ? 4 : 6;
}

void pr_jplace(struct janas_buf *b, const char *key, const struct pr_place *p)
{
    janas_buf_printf(b, ", \"%s\": {\"name\": ", key);
    janas_json_write_str(b, p->name, strlen(p->name));
    pr_jstr(b, "country", p->country);
    pr_jstr(b, "how", p->how);
    janas_buf_puts(b, "}");
}

/* a moment in the computer's zone: "2 October, 14:05" in parts */
static void jtime(struct janas_buf *b, const char *key, time_t t)
{
    struct tm lt;
    if (t <= 0) {
        janas_buf_printf(b, ", \"%s\": {\"day\": 0, \"mon\": 0, \"at\": \"\"}",
                         key);
        return;
    }
    localtime_r(&t, &lt);
    janas_buf_printf(b,
                     ", \"%s\": {\"day\": %d, \"mon\": %d, \"at\": "
                     "\"%02d:%02d\"}",
                     key, lt.tm_mday, lt.tm_mon + 1, lt.tm_hour, lt.tm_min);
}

/* "2026-10-02" as day and month; "2026-08" as month and year */
static void jdate(struct janas_buf *b, const char *key, const char *d)
{
    int y = 0, m = 0, day = 0;
    int k = d ? sscanf(d, "%d-%d-%d", &y, &m, &day) : 0;
    janas_buf_printf(b, ", \"%s\": {\"day\": %d, \"mon\": %d, \"year\": %d}",
                     key, k == 3 ? day : 0, k >= 2 ? m : 0, k >= 1 ? y : 0);
}

void pr_fx_data(struct janas_buf *b, double amount, const char *from,
                const struct pr_fx *fx, const char *source)
{
    janas_buf_puts(b, "{\"from\": ");
    janas_json_write_str(b, from, strlen(from));
    pr_jnum(b, "amount", amount, 2);
    jdate(b, "date", fx->date);
    pr_jstr(b, "source", source);
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < fx->n; i++) {
        janas_buf_printf(b, "%s{\"code\": ", i ? ", " : "");
        janas_json_write_str(b, fx->r[i].code, strlen(fx->r[i].code));
        pr_jnum(b, "rate", fx->r[i].rate,
                fx->r[i].rate >= 100  ? 2
                : fx->r[i].rate < 0.1 ? 6
                                      : 4);
        pr_jnum(b, "value", amount * fx->r[i].rate, 2);
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]}");
}

void pr_coin_data(struct janas_buf *b, const struct pr_coin *c,
                  const char *source)
{
    janas_buf_puts(b, "{\"name\": ");
    janas_json_write_str(b, c->name, strlen(c->name));
    pr_jstr(b, "symbol", c->symbol);
    pr_jstr(b, "cur", c->cur);
    pr_jnum(b, "price", c->price,
            pr_decimals(c->price) > 2 ? pr_decimals(c->price) : 2);
    pr_jnum(b, "change", c->change_24h, 2);
    janas_buf_printf(b, ", \"up\": %s",
                     !isnan(c->change_24h) && c->change_24h >= 0 ? "true"
                                                                 : "false");
    pr_jnum(b, "cap_bn", isnan(c->cap) ? NAN : c->cap / 1e9, 1);
    jtime(b, "at", c->at);
    pr_jstr(b, "source", source);
    janas_buf_puts(b, "}");
}

void pr_quote_data(struct janas_buf *b, const struct pr_quote *q)
{
    janas_buf_puts(b, "{\"symbol\": ");
    janas_json_write_str(b, q->symbol, strlen(q->symbol));
    pr_jstr(b, "name", q->name);
    pr_jstr(b, "region", q->region);
    pr_jstr(b, "currency", q->currency);
    int dec = pr_decimals(q->price) > 2 ? 3 : 2;
    pr_jnum(b, "price", q->price, dec);
    pr_jnum(b, "open", q->open, dec);
    pr_jnum(b, "high", q->high, dec);
    pr_jnum(b, "low", q->low, dec);
    pr_jnum(b, "prev", q->prev, dec);
    pr_jnum(b, "change", q->change, dec);
    pr_jnum(b, "pct", q->change_pct, 2);
    janas_buf_printf(b, ", \"up\": %s, \"volume\": %ld",
                     !isnan(q->change) && q->change >= 0 ? "true" : "false",
                     q->volume < 0 ? 0 : q->volume);
    jdate(b, "day", q->day);
    janas_buf_puts(b, "}");
}

void pr_series_data(struct janas_buf *b, const char *what, const char *who,
                    const struct pr_series *s, int last, const char *source,
                    const char *note, const char *licence)
{
    janas_buf_puts(b, "{\"what\": ");
    janas_json_write_str(b, what, strlen(what));
    pr_jstr(b, "who", who);
    pr_jstr(b, "label", s->label);
    pr_jstr(b, "unit", s->unit);
    pr_jstr(b, "source", source);
    pr_jstr(b, "note", note);
    pr_jstr(b, "licence", licence);
    int from = s->n > last ? s->n - last : 0;
    double lo = NAN, hi = NAN, sum = 0;
    int ilo = -1, ihi = -1, k = 0;
    for (int i = from; i < s->n; i++) {
        if (isnan(lo) || s->p[i].v < lo)
            lo = s->p[i].v, ilo = i;
        if (isnan(hi) || s->p[i].v > hi)
            hi = s->p[i].v, ihi = i;
        sum += s->p[i].v;
        k++;
    }
    int dec = 1;
    for (int i = from; i < s->n; i++) /* as many as the values need */
        if (fabs(s->p[i].v * 100 - round(s->p[i].v * 100)) > 1e-6)
            dec = 3;
        else if (dec < 2 && fabs(s->p[i].v * 10 - round(s->p[i].v * 10)) > 1e-6)
            dec = 2;
    if (s->n) {
        janas_buf_puts(b, ", \"last\": {\"when\": ");
        janas_json_write_str(b, s->p[s->n - 1].when,
                             strlen(s->p[s->n - 1].when));
        pr_jnum(b, "v", s->p[s->n - 1].v, dec);
        janas_buf_puts(b, "}");
    }
    pr_jnum(b, "min", lo, dec);
    pr_jstr(b, "min_when", ilo >= 0 ? s->p[ilo].when : "");
    pr_jnum(b, "max", hi, dec);
    pr_jstr(b, "max_when", ihi >= 0 ? s->p[ihi].when : "");
    pr_jnum(b, "avg", k ? sum / k : NAN, dec);
    janas_buf_puts(b, ", \"items\": [");
    for (int i = from; i < s->n; i++) {
        janas_buf_printf(b, "%s{\"when\": ", i > from ? ", " : "");
        janas_json_write_str(b, s->p[i].when, strlen(s->p[i].when));
        pr_jnum(b, "v", s->p[i].v, dec);
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]}");
}

void pr_rates_data(struct janas_buf *b, const char *bank,
                   const struct pr_rate *r, int n, double lo, double hi,
                   const char *source)
{
    janas_buf_puts(b, "{\"bank\": ");
    janas_json_write_str(b, bank, strlen(bank));
    pr_jstr(b, "source", source);
    janas_buf_printf(b, ", \"range\": %s",
                     isnan(lo) || isnan(hi) ? "false" : "true");
    pr_jnum(b, "lo", lo, 2);
    pr_jnum(b, "hi", hi, 2);
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < n; i++) {
        janas_buf_printf(b, "%s{\"name\": ", i ? ", " : "");
        janas_json_write_str(b, r[i].name, strlen(r[i].name));
        pr_jnum(b, "v", r[i].v, r[i].v * 100 == round(r[i].v * 100) ? 2 : 3);
        jdate(b, "when", r[i].when);
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]}");
}

void pr_fuel_data(struct janas_buf *b, const struct pr_place *p,
                  const char *kind, double radius, const struct pr_fuel *f,
                  const char *source)
{
    janas_buf_puts(b, "{\"kind\": ");
    janas_json_write_str(b, kind, strlen(kind));
    pr_jplace(b, "place", p);
    pr_jstr(b, "fuel", f->fuel);
    pr_jstr(b, "unit", strcmp(kind, "cng") == 0 ? "€/kg" : "€/l");
    pr_jstr(b, "source", source);
    janas_buf_printf(b,
                     ", \"radius\": %g, \"found\": %d, \"shown\": %d, "
                     "\"stale\": %d, \"stale_days\": %d",
                     radius, f->found, f->n, f->stale, PR_STALE_DAYS);
    pr_jnum(b, "cheapest", f->cheapest, 3);
    pr_jnum(b, "average", f->average, 3);
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < f->n; i++) {
        const struct pr_station *s = &f->s[i];
        janas_buf_printf(b, "%s{\"name\": ", i ? ", " : "");
        janas_json_write_str(b, s->name, strlen(s->name));
        pr_jstr(b, "address", s->address);
        pr_jstr(b, "city", s->city);
        pr_jnum(b, "price", s->price, 3);
        if (s->km >= 1)
            janas_buf_printf(b, ", \"km\": %.1f, \"m\": 0", s->km);
        else
            janas_buf_printf(b, ", \"km\": 0, \"m\": %.0f",
                             fmax(10, round(s->km * 100) * 10));
        pr_jstr(b, "dir",
                geo_compass(geo_bearing(p->lat, p->lon, s->lat, s->lon)));
        janas_buf_printf(b, ", \"self\": %s, \"served\": %s",
                         s->self == 1 ? "true" : "false",
                         s->self == 0 ? "true" : "false");
        pr_jstr(b, "updated", s->updated);
        janas_buf_puts(b, "}");
    }
    janas_buf_puts(b, "]}");
}
