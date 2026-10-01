/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * metno.c - MET Norway's locationforecast (see weather.h), when Open-Meteo
 * does not answer: free for any use under CC BY 4.0, asked with a
 * User-Agent naming the program. Its series is in UTC, by the hour for the
 * first two days and by six hours after; it is put into the place's own
 * zone, its hours into days. It has no gusts, no chance of rain and no
 * sunrise in its compact form: those are left unknown.
 */
#define _GNU_SOURCE /* timegm */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "weather.h"

#define MET_KEEP_S 900

static double det(const struct janas_json *d, const char *k)
{
    const struct janas_json *v = janas_json_get(d, k);
    return v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, NAN) : NAN;
}

/* "2026-10-01T11:00:00Z" */
static time_t utc_of(const char *s)
{
    struct tm tm = {0};
    if (!s || sscanf(s, "%d-%d-%dT%d:%d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
                     &tm.tm_hour, &tm.tm_min) != 5)
        return (time_t)-1;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return timegm(&tm);
}

static void symbol_of(const struct janas_json *next, char *out, size_t cap)
{
    const char *s = janas_json_str(
        janas_json_get(janas_json_get(next, "summary"), "symbol_code"));
    snprintf(out, cap, "%s", s ? s : "");
}

static void max_in(double *m, double v)
{
    if (!isnan(v) && (isnan(*m) || v > *m))
        *m = v;
}

static void min_in(double *m, double v)
{
    if (!isnan(v) && (isnan(*m) || v < *m))
        *m = v;
}

int wx_met_parse(const char *text, size_t n, const char *tz,
                 struct wx_forecast *f, char *err, size_t err_len)
{
    memset(f, 0, sizeof *f);
    f->source = "MET Norway";
    struct janas_buf b = {.p = (char *)text, .n = n};
    struct janas_json_doc *d = wx_parse(&b, "MET Norway", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *r = janas_json_root(d);
    const struct janas_json *coords =
        janas_json_get(janas_json_get(r, "geometry"), "coordinates");
    const struct janas_json *ts =
        janas_json_get(janas_json_get(r, "properties"), "timeseries");
    if (!ts || ts->type != JANAS_JSON_ARRAY || !ts->child) {
        snprintf(err, err_len, "MET Norway: an answer without a forecast");
        janas_json_free(d);
        return -1;
    }
    if (coords && coords->type == JANAS_JSON_ARRAY && coords->child) {
        f->grid_lon = janas_json_num(coords->child, NAN);
        f->grid_lat = coords->child->next
                          ? janas_json_num(coords->child->next, NAN)
                          : NAN;
        f->grid_elev = coords->child->next && coords->child->next->next
                           ? janas_json_num(coords->child->next->next, NAN)
                           : NAN;
    }
    snprintf(f->tz, sizeof f->tz, "%s", tz && *tz ? tz : "UTC");

    /* the series is put into the place's zone */
    char *old = getenv("TZ") ? strdup(getenv("TZ")) : NULL;
    setenv("TZ", f->tz, 1);
    tzset();
    time_t prev_day_t = 0;
    for (const struct janas_json *e = ts->child; e; e = e->next) {
        time_t t = utc_of(janas_json_str(janas_json_get(e, "time")));
        if (t == (time_t)-1)
            continue;
        struct tm lt;
        localtime_r(&t, &lt);
        const struct janas_json *data = janas_json_get(e, "data");
        const struct janas_json *in =
            janas_json_get(janas_json_get(data, "instant"), "details");
        const struct janas_json *n1 = janas_json_get(data, "next_1_hours");
        const struct janas_json *n6 = janas_json_get(data, "next_6_hours");
        const struct janas_json *n12 = janas_json_get(data, "next_12_hours");
        double temp = det(in, "air_temperature");
        double wind = det(in, "wind_speed") * 3.6; /* m/s */
        double dir = det(in, "wind_from_direction");
        double rain1 =
            det(janas_json_get(n1, "details"), "precipitation_amount");
        double rain6 =
            det(janas_json_get(n6, "details"), "precipitation_amount");
        if (!f->has_now) {
            struct wx_now *w = &f->now;
            strftime(w->time, sizeof w->time, "%Y-%m-%dT%H:%M", &lt);
            w->temp = temp;
            w->feels = NAN;
            w->humidity = det(in, "relative_humidity");
            w->cloud = det(in, "cloud_area_fraction");
            w->pressure = det(in, "air_pressure_at_sea_level");
            w->wind = wind;
            w->wind_dir = dir;
            w->gust = NAN;
            w->precip = rain1;
            w->code = -1;
            symbol_of(n1 ? n1 : n6, w->symbol, sizeof w->symbol);
            f->has_now = 1;
            snprintf(f->tz_abbr, sizeof f->tz_abbr, "%s", lt.tm_zone);
        }
        if (n1 && f->n_hour < WX_HOURS) {
            struct wx_hour *h = &f->hour[f->n_hour++];
            strftime(h->time, sizeof h->time, "%Y-%m-%dT%H:%M", &lt);
            h->temp = temp;
            h->rain = rain1;
            h->rain_prob = NAN;
            h->wind = wind;
            h->wind_dir = dir;
            h->gust = NAN;
            h->code = -1;
            symbol_of(n1, h->symbol, sizeof h->symbol);
        }
        char date[11];
        strftime(date, sizeof date, "%Y-%m-%d", &lt);
        struct wx_day *y = f->n_day ? &f->day[f->n_day - 1] : NULL;
        if (!y || strcmp(y->date, date) != 0) {
            if (f->n_day == WX_DAYS)
                break;
            y = &f->day[f->n_day++];
            memset(y, 0, sizeof *y);
            snprintf(y->date, sizeof y->date, "%s", date);
            y->code = -1;
            y->tmin = y->tmax = y->rain = y->wind = y->wind_dir = NAN;
            y->gust = y->rain_prob = y->uv = NAN;
            y->rain = 0;
        }
        min_in(&y->tmin, temp);
        max_in(&y->tmax, temp);
        if (isnan(y->wind) || wind > y->wind) {
            y->wind = wind;
            y->wind_dir = dir;
        }
        /* the rain: an hour's while the series is by the hour, six
           hours' after */
        const struct janas_json *nx = e->next;
        time_t tn = nx ? utc_of(janas_json_str(janas_json_get(nx, "time")))
                       : (time_t)-1;
        if (!isnan(rain1) && tn != (time_t)-1 && tn - t <= 3600)
            y->rain += rain1;
        else if (!isnan(rain6) && t - prev_day_t >= 6 * 3600) {
            y->rain += rain6;
            prev_day_t = t;
        }
        /* the day's sky: the next twelve hours' from the morning */
        if (!y->symbol[0] && n12 && lt.tm_hour >= 6)
            symbol_of(n12, y->symbol, sizeof y->symbol);
    }
    if (old) {
        setenv("TZ", old, 1);
        free(old);
    } else {
        unsetenv("TZ");
    }
    tzset();
    janas_json_free(d);
    return 0;
}

int wx_met_forecast(const struct wx_place *p, int days, struct wx_forecast *f,
                    char *err, size_t err_len)
{
    char url[256];
    /* four decimals at most, as MET Norway asks: more are refused */
    snprintf(url, sizeof url,
             "https://api.met.no/weatherapi/locationforecast/2.0/compact?"
             "lat=%.4f&lon=%.4f",
             p->lat, p->lon);
    struct janas_buf b = {0};
    char why[256];
    int status = wx_fetch(url, MET_KEEP_S, 1, &b, why, sizeof why);
    int r = -1;
    if (status == 200)
        r = wx_met_parse(b.p, b.n, p->tz, f, err, err_len);
    else if (status < 0)
        snprintf(err, err_len, "MET Norway did not answer: %s", why);
    else
        snprintf(err, err_len, "MET Norway: HTTP %d", status);
    janas_buf_free(&b);
    if (r == 0 && f->n_day > days)
        f->n_day = days;
    return r;
}
