/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * report.c - weather_now and weather_forecast (see weather.h): Open-Meteo's
 * forecast, MET Norway's when it does not answer, and next to the weather
 * now what the nearest station measured. A forecast is a model's value at
 * a cell of its grid and a station's is a measure at a known place and
 * minute: both are given, said for what they are, and when they disagree
 * it is said too, so that the model telling them need not guess which.
 * The answers are data with a layout (layouts.c, services/common/template.h).
 */
#define _GNU_SOURCE /* timegm */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "services/common/geo.h"
#include "services/common/template.h"
#include "weather.h"

#define STATION_KM 40 /* the nearest station, within this */

static int forecast(const struct wx_place *p, int days, struct wx_forecast *f,
                    char *err, size_t err_len)
{
    char e1[384];
    if (wx_om_forecast(p, days, f, e1, sizeof e1) == 0)
        return 0;
    char e2[384];
    if (wx_met_forecast(p, days, f, e2, sizeof e2) == 0)
        return 0;
    snprintf(err, err_len, "no forecast: %s; %s", e1, e2);
    return -1;
}

/* ---- the data ---- */

void wx_jstr(struct janas_buf *b, const char *key, const char *v)
{
    janas_buf_printf(b, ", \"%s\": ", key);
    janas_json_write_str(b, v, strlen(v));
}

/* a number, when there is one */
void wx_jnum(struct janas_buf *b, const char *key, double v, int decimals)
{
    if (!isnan(v))
        janas_buf_printf(b, ", \"%s\": %.*f", key, decimals, v);
}

/* the sky: a WMO code with words in the layout, or MET Norway's symbol
   in words */
static void jsky(struct janas_buf *b, int code, const char *symbol)
{
    if (code >= 0 && wx_code_text(code)) {
        janas_buf_printf(b, ", \"sky\": %d", code);
    } else {
        char s[96] = "not given";
        if (symbol && *symbol)
            wx_symbol_text(symbol, s, sizeof s);
        wx_jstr(b, "sky_text", s);
    }
}

/* where the wind blows from, and how hard */
void wx_jwind(struct janas_buf *b, double kmh, double deg)
{
    wx_jnum(b, "wind", kmh, 0);
    if (!isnan(kmh))
        janas_buf_printf(b, ", \"bft\": %d", wx_beaufort(kmh));
    if (!isnan(deg)) {
        wx_jstr(b, "wind_from", geo_compass(deg));
        wx_jnum(b, "wind_deg", deg, 0);
    }
}

/* "2026-10-02": its weekday (0 Sunday), day and month */
void wx_jdate(struct janas_buf *b, const char *date)
{
    struct tm tm = {0};
    if (sscanf(date, "%d-%d-%d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday) != 3)
        return;
    int day = tm.tm_mday, mon = tm.tm_mon;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    tm.tm_hour = 12;
    time_t t = timegm(&tm);
    struct tm g;
    gmtime_r(&t, &g);
    janas_buf_printf(b, "\"wd\": %d, \"day\": %d, \"mon\": %d", g.tm_wday, day,
                     mon);
}

/* the place, and where the user is when it was not named */
void wx_jplace(struct janas_buf *b, const struct wx_place *p)
{
    struct wx_place q = *p;
    q.how[0] = 0; /* the layout says it */
    char name[256];
    wx_place_name(&q, name, sizeof name);
    janas_buf_puts(b, "{\"place\": ");
    janas_json_write_str(b, name, strlen(name));
    if (p->how[0])
        wx_jstr(b, "how", p->how);
}

static void jmodel(struct janas_buf *b, const struct wx_place *p,
                   const struct wx_forecast *f)
{
    janas_buf_puts(b, ", \"model\": {\"source\": ");
    janas_json_write_str(b, f->source, strlen(f->source));
    wx_jstr(b, "url",
            strcmp(f->source, "Open-Meteo") == 0 ? "open-meteo.com"
                                                 : "api.met.no");
    wx_jnum(b, "lat", f->grid_lat, 3);
    wx_jnum(b, "lon", f->grid_lon, 3);
    if (!isnan(f->grid_elev))
        wx_jnum(b, "elev", f->grid_elev, 0);
    wx_jnum(b, "km", geo_km(p->lat, p->lon, f->grid_lat, f->grid_lon), 1);
    janas_buf_puts(b, "}");
}

/* statute miles as given ("6+", "1 1/2") in km */
static void jvisibility(struct janas_buf *b, const char *v)
{
    if (!v[0])
        return;
    double mi = atof(v);
    const char *frac = strchr(v, '/');
    if (frac) {
        const char *sp = strchr(v, ' ');
        double whole = sp ? atof(v) : 0;
        int a = atoi(sp ? sp + 1 : v), d = atoi(frac + 1);
        mi = whole + (d ? (double)a / d : 0);
    }
    wx_jnum(b, "vis_km", mi * 1.609, strchr(v, '+') ? 0 : 1);
    if (strchr(v, '+'))
        janas_buf_puts(b, ", \"vis_more\": true");
}

/* "-SHRA": its parts, "-", "SH", "RA", for the layout's words */
static void jweather(struct janas_buf *b, const char *wx)
{
    static const char *const codes[] = {"TS", "SH", "FZ", "RA", "DZ", "SN",
                                        "GR", "GS", "FG", "BR", "HZ", "VC"};
    wx_jstr(b, "wx", wx);
    janas_buf_puts(b, ", \"wx_parts\": [");
    const char *s = wx;
    int k = 0;
    if (*s == '-' || *s == '+') {
        janas_buf_printf(b, "\"%c\"", *s);
        k++;
        s++;
    }
    while (*s) {
        size_t i;
        for (i = 0; i < sizeof codes / sizeof *codes; i++)
            if (strncmp(s, codes[i], 2) == 0) {
                janas_buf_printf(b, "%s\"%s\"", k++ ? ", " : "", codes[i]);
                s += 2;
                break;
            }
        if (i == sizeof codes / sizeof *codes)
            s++;
    }
    janas_buf_puts(b, "]");
}

static void jstation(struct janas_buf *b, const struct wx_place *p,
                     const struct wx_obs *o)
{
    char when[64];
    wx_local_time(o->at, p->tz, when, sizeof when);
    janas_buf_puts(b, ", \"station\": {\"name\": ");
    janas_json_write_str(b, o->name, strlen(o->name));
    wx_jstr(b, "icao", o->icao);
    wx_jnum(b, "km", o->km, 0);
    wx_jstr(b, "dir", geo_compass(geo_bearing(p->lat, p->lon, o->lat, o->lon)));
    wx_jstr(b, "time", when);
    janas_buf_printf(b, ", \"ago\": %ld",
                     (long)difftime(time(NULL), o->at) / 60);
    wx_jnum(b, "temp", o->temp, 0);
    wx_jnum(b, "dew", o->dew, 0);
    if (!isnan(o->wind_kt)) {
        if (o->wind_kt < 1)
            janas_buf_puts(b, ", \"calm\": true");
        else if (o->variable_wind)
            janas_buf_puts(b, ", \"variable\": true");
        else
            wx_jstr(b, "from", geo_compass(o->wind_dir));
        if (o->wind_kt >= 1) {
            wx_jnum(b, "kt", o->wind_kt, 0);
            wx_jnum(b, "kmh", o->wind_kt * 1.852, 0);
            janas_buf_printf(b, ", \"bft\": %d",
                             wx_beaufort(o->wind_kt * 1.852));
        }
        if (!isnan(o->gust_kt)) {
            wx_jnum(b, "gust_kt", o->gust_kt, 0);
            wx_jnum(b, "gust_kmh", o->gust_kt * 1.852, 0);
        }
    }
    jvisibility(b, o->visib);
    if (o->wx[0])
        jweather(b, o->wx);
    if (o->n_layers) {
        janas_buf_puts(b, ", \"clouds\": [");
        for (int i = 0; i < o->n_layers; i++) {
            janas_buf_printf(b, "%s{\"cover\": ", i ? ", " : "");
            janas_json_write_str(b, o->layer[i].cover,
                                 strlen(o->layer[i].cover));
            if (!isnan(o->layer[i].base_ft)) {
                wx_jnum(b, "ft", o->layer[i].base_ft, 0);
                wx_jnum(b, "m", o->layer[i].base_ft * 0.3048, 0);
            }
            janas_buf_puts(b, "}");
        }
        janas_buf_puts(b, "]");
    }
    wx_jnum(b, "pressure", o->pressure, 0);
    wx_jstr(b, "raw", o->raw);
    janas_buf_puts(b, "}");
}

/* The answer from the data and its layout. */
int wx_answer(struct janas_buf *b, const char *name, struct janas_buf *d,
              const char *layout, const char *brief)
{
    char err[300];
    if (d->oom || janas_tpl_result(b, name, d->p, d->n, layout, brief, err,
                                   sizeof err) != 0) {
        struct janas_buf out = {0};
        janas_buf_printf(&out, "The answer could not be written: %s.",
                         d->oom ? "out of memory" : err);
        janas_buf_free(d);
        return wx_result(&out, b, 1);
    }
    janas_buf_free(d);
    return 0;
}

int wx_tool_now(const struct janas_json *args, struct janas_buf *b)
{
    struct wx_place p;
    if (wx_arg_place(args, &p, b) != 0)
        return 0;
    struct wx_forecast *f = calloc(1, sizeof *f);
    char err[800], name[256];
    wx_place_name(&p, name, sizeof name);
    if (!f || forecast(&p, 1, f, err, sizeof err) != 0) {
        struct janas_buf out = {0};
        janas_buf_printf(&out, "No forecast for %s: %s.", name,
                         f ? err : "out of memory");
        free(f);
        return wx_result(&out, b, 1);
    }
    if (!p.tz[0])
        snprintf(p.tz, sizeof p.tz, "%s", f->tz);
    const struct wx_now *w = &f->now;
    const char *T = strchr(w->time, 'T');
    struct janas_buf d = {0};
    wx_jplace(&d, &p);
    janas_buf_printf(&d, ", \"time\": \"%.5s\"", T ? T + 1 : w->time);
    wx_jstr(&d, "tz", f->tz_abbr);
    jsky(&d, w->code, w->symbol);
    wx_jnum(&d, "cloud", w->cloud, 0);
    wx_jnum(&d, "temp", w->temp, 1);
    wx_jnum(&d, "feels", w->feels, 1);
    wx_jnum(&d, "humidity", w->humidity, 0);
    wx_jwind(&d, w->wind, w->wind_dir);
    wx_jnum(&d, "gust", w->gust, 0);
    if (!isnan(w->precip)) {
        janas_buf_puts(&d, ", \"has_precip\": true");
        wx_jnum(&d, "precip", w->precip, 1);
    }
    wx_jnum(&d, "pressure", w->pressure, 0);
    if (f->n_day) {
        const struct wx_day *y = &f->day[0];
        janas_buf_puts(&d, ", \"today\": {\"tmin\": ");
        janas_buf_printf(&d, "%.1f, \"tmax\": %.1f", y->tmin, y->tmax);
        if (!isnan(y->rain_prob)) {
            janas_buf_puts(&d, ", \"has_prob\": true");
            wx_jnum(&d, "rain_prob", y->rain_prob, 0);
        }
        if (y->sunrise[0]) {
            wx_jstr(&d, "sunrise", y->sunrise);
            wx_jstr(&d, "sunset", y->sunset);
        }
        janas_buf_puts(&d, "}");
    }
    jmodel(&d, &p, f);
    struct wx_obs o;
    char why[384];
    int r = wx_metar_near(p.lat, p.lon, STATION_KM, &o, why, sizeof why);
    if (r == 1) {
        jstation(&d, &p, &o);
        if (f->has_now && !isnan(f->now.temp) &&
            fabs(f->now.temp - o.temp) >= 3)
            wx_jnum(&d, "temp_diff", fabs(f->now.temp - o.temp), 0);
        int st_rain = strstr(o.wx, "RA") || strstr(o.wx, "DZ") ||
                      strstr(o.wx, "TS") || strstr(o.wx, "SN");
        int fc_rain =
            f->has_now &&
            (f->now.code >= 50 || (!isnan(f->now.precip) && f->now.precip > 0));
        if (f->has_now && f->now.code >= 0 && st_rain != fc_rain)
            janas_buf_printf(&d,
                             ", \"rain_disagree\": true, \"station_rain\": "
                             "%s, \"model_rain\": %s",
                             st_rain ? "true" : "false",
                             fc_rain ? "true" : "false");
    } else if (r == 0) {
        janas_buf_printf(&d, ", \"no_station\": %d", STATION_KM);
    } else {
        wx_jstr(&d, "station_error", why);
    }
    janas_buf_puts(&d, "}");
    free(f);
    return wx_answer(b, "weather_now", &d, WX_NOW_LAYOUT, WX_NOW_BRIEF);
}

int wx_tool_forecast(const struct janas_json *args, struct janas_buf *b)
{
    struct wx_place p;
    if (wx_arg_place(args, &p, b) != 0)
        return 0;
    int days = (int)wx_arg_num(args, "days", 3);
    if (days < 1)
        days = 1;
    if (days > WX_DAYS)
        days = WX_DAYS;
    int hourly = wx_arg_bool(args, "hourly", 0);
    struct wx_forecast *f = calloc(1, sizeof *f);
    char err[800], name[256];
    wx_place_name(&p, name, sizeof name);
    if (!f || forecast(&p, days, f, err, sizeof err) != 0) {
        struct janas_buf out = {0};
        janas_buf_printf(&out, "No forecast for %s: %s.", name,
                         f ? err : "out of memory");
        free(f);
        return wx_result(&out, b, 1);
    }
    struct janas_buf d = {0};
    wx_jplace(&d, &p);
    wx_jstr(&d, "tz", f->tz_abbr);
    janas_buf_puts(&d, ", \"days\": [");
    for (int i = 0; i < f->n_day && i < days; i++) {
        const struct wx_day *y = &f->day[i];
        janas_buf_puts(&d, i ? ", {" : "{");
        wx_jdate(&d, y->date);
        jsky(&d, y->code, y->symbol);
        wx_jnum(&d, "tmin", y->tmin, 1);
        wx_jnum(&d, "tmax", y->tmax, 1);
        if (!isnan(y->rain)) {
            if (y->rain < 0.1)
                janas_buf_puts(&d, ", \"no_rain\": true");
            else
                wx_jnum(&d, "rain", y->rain, 1);
        }
        if (!isnan(y->rain_prob)) {
            janas_buf_puts(&d, ", \"has_prob\": true");
            wx_jnum(&d, "rain_prob", y->rain_prob, 0);
        }
        wx_jwind(&d, y->wind, y->wind_dir);
        wx_jnum(&d, "gust", y->gust, 0);
        wx_jnum(&d, "uv", y->uv, 0);
        if (y->sunrise[0]) {
            wx_jstr(&d, "sunrise", y->sunrise);
            wx_jstr(&d, "sunset", y->sunset);
        }
        janas_buf_puts(&d, "}");
    }
    janas_buf_puts(&d, "]");
    if (f->n_day < days)
        janas_buf_printf(&d, ", \"fewer\": %d", f->n_day);
    int step = hourly ? 1 : 3, last = hourly ? WX_HOURS : 24;
    janas_buf_puts(&d, ", \"hours\": [");
    char prev[11] = "";
    for (int i = 0; i < f->n_hour && i < last; i += step) {
        const struct wx_hour *h = &f->hour[i];
        const char *T = strchr(h->time, 'T');
        janas_buf_printf(&d, "%s{\"time\": \"%.5s\"", i ? ", " : "",
                         T ? T + 1 : h->time);
        if (strncmp(prev, h->time, 10) != 0) {
            char date[11];
            snprintf(date, sizeof date, "%.10s", h->time);
            janas_buf_puts(&d, ", \"newday\": {");
            wx_jdate(&d, date);
            janas_buf_puts(&d, "}");
            snprintf(prev, sizeof prev, "%s", date);
        }
        jsky(&d, h->code, h->symbol);
        wx_jnum(&d, "temp", h->temp, 1);
        if (!isnan(h->rain) && h->rain >= 0.1)
            wx_jnum(&d, "rain", h->rain, 1);
        if (!isnan(h->rain_prob)) {
            janas_buf_puts(&d, ", \"has_prob\": true");
            wx_jnum(&d, "rain_prob", h->rain_prob, 0);
        }
        wx_jwind(&d, h->wind, h->wind_dir);
        wx_jnum(&d, "gust", h->gust, 0);
        janas_buf_puts(&d, "}");
    }
    janas_buf_puts(&d, "]");
    jmodel(&d, &p, f);
    janas_buf_puts(&d, "}");
    free(f);
    return wx_answer(b, "weather_forecast", &d, WX_FORECAST_LAYOUT,
                     WX_FORECAST_BRIEF);
}
