/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * openmeteo.c - Open-Meteo's forecasts (see weather.h): the weather now,
 * by the day and by the hour at a place, and the sea at a few points.
 * Free for non-commercial use, no key, its data under CC BY 4.0. Its
 * values are a model's, at the cell of its grid nearest the place - on
 * land, asked so - not a station's measures.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "weather.h"

#define OM_KEEP_S 900 /* a forecast changes once an hour at most */

/* The numbers of a JSON array, NAN for a null; how many, at most max. */
static int nums(const struct janas_json *a, double *out, int max)
{
    int n = 0;
    for (const struct janas_json *v =
             a && a->type == JANAS_JSON_ARRAY ? a->child : NULL;
         v && n < max; v = v->next)
        out[n++] = v->type == JANAS_JSON_NUMBER ? janas_json_num(v, NAN) : NAN;
    return n;
}

static double num(const struct janas_json *o, const char *k)
{
    const struct janas_json *v = janas_json_get(o, k);
    return v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, NAN) : NAN;
}

static void str_at(const struct janas_json *a, int i, char *out, size_t cap)
{
    const struct janas_json *v =
        a && a->type == JANAS_JSON_ARRAY ? a->child : NULL;
    while (v && i-- > 0)
        v = v->next;
    snprintf(out, cap, "%s", v && janas_json_str(v) ? janas_json_str(v) : "");
}

static void hhmm(const struct janas_json *a, int i, char *out, size_t cap)
{
    char t[24];
    str_at(a, i, t, sizeof t); /* "2026-10-01T07:05" */
    const char *T = strchr(t, 'T');
    snprintf(out, cap, "%.5s", T ? T + 1 : "");
}

static void tz_of(const struct janas_json *r, struct wx_forecast *f)
{
    snprintf(f->tz, sizeof f->tz, "%s",
             janas_json_str(janas_json_get(r, "timezone"))
                 ? janas_json_str(janas_json_get(r, "timezone"))
                 : "");
    snprintf(f->tz_abbr, sizeof f->tz_abbr, "%s",
             janas_json_str(janas_json_get(r, "timezone_abbreviation"))
                 ? janas_json_str(janas_json_get(r, "timezone_abbreviation"))
                 : "");
}

static int error_of(const struct janas_json *r, char *err, size_t err_len)
{
    const struct janas_json *e = janas_json_get(r, "error");
    if (!e || e->type != JANAS_JSON_TRUE)
        return 0;
    const char *why = janas_json_str(janas_json_get(r, "reason"));
    snprintf(err, err_len, "Open-Meteo: %s", why ? why : "an error");
    return 1;
}

int wx_om_parse(const char *text, size_t n, struct wx_forecast *f, char *err,
                size_t err_len)
{
    memset(f, 0, sizeof *f);
    f->source = "Open-Meteo";
    struct janas_buf b = {.p = (char *)text, .n = n};
    struct janas_json_doc *d = wx_parse(&b, "Open-Meteo", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *r = janas_json_root(d);
    if (error_of(r, err, err_len)) {
        janas_json_free(d);
        return -1;
    }
    f->grid_lat = num(r, "latitude");
    f->grid_lon = num(r, "longitude");
    f->grid_elev = num(r, "elevation");
    tz_of(r, f);

    const struct janas_json *c = janas_json_get(r, "current");
    if (c && c->type == JANAS_JSON_OBJECT) {
        struct wx_now *w = &f->now;
        snprintf(w->time, sizeof w->time, "%s",
                 janas_json_str(janas_json_get(c, "time"))
                     ? janas_json_str(janas_json_get(c, "time"))
                     : "");
        w->temp = num(c, "temperature_2m");
        w->feels = num(c, "apparent_temperature");
        w->humidity = num(c, "relative_humidity_2m");
        w->precip = num(c, "precipitation");
        w->cloud = num(c, "cloud_cover");
        w->pressure = num(c, "pressure_msl");
        w->wind = num(c, "wind_speed_10m");
        w->wind_dir = num(c, "wind_direction_10m");
        w->gust = num(c, "wind_gusts_10m");
        double code = num(c, "weather_code");
        w->code = isnan(code) ? -1 : (int)code;
        w->is_day = num(c, "is_day") == 1;
        f->has_now = 1;
    }

    const struct janas_json *dl = janas_json_get(r, "daily");
    if (dl && dl->type == JANAS_JSON_OBJECT) {
        double v[10][WX_DAYS];
        static const char *const k[10] = {"weather_code",
                                          "temperature_2m_min",
                                          "temperature_2m_max",
                                          "precipitation_sum",
                                          "precipitation_probability_max",
                                          "wind_speed_10m_max",
                                          "wind_gusts_10m_max",
                                          "wind_direction_10m_dominant",
                                          "uv_index_max",
                                          NULL};
        int nd = nums(janas_json_get(dl, "time"), v[9], WX_DAYS);
        for (int i = 0; i < 9; i++)
            for (int j = nums(janas_json_get(dl, k[i]), v[i], WX_DAYS); j < nd;
                 j++)
                v[i][j] = NAN;
        for (int i = 0; i < nd; i++) {
            struct wx_day *y = &f->day[i];
            str_at(janas_json_get(dl, "time"), i, y->date, sizeof y->date);
            y->code = isnan(v[0][i]) ? -1 : (int)v[0][i];
            y->tmin = v[1][i];
            y->tmax = v[2][i];
            y->rain = v[3][i];
            y->rain_prob = v[4][i];
            y->wind = v[5][i];
            y->gust = v[6][i];
            y->wind_dir = v[7][i];
            y->uv = v[8][i];
            hhmm(janas_json_get(dl, "sunrise"), i, y->sunrise,
                 sizeof y->sunrise);
            hhmm(janas_json_get(dl, "sunset"), i, y->sunset, sizeof y->sunset);
        }
        f->n_day = nd;
    }

    const struct janas_json *hl = janas_json_get(r, "hourly");
    if (hl && hl->type == JANAS_JSON_OBJECT) {
        double v[8][WX_HOURS];
        static const char *const k[8] = {
            "weather_code",   "temperature_2m",
            "precipitation",  "precipitation_probability",
            "wind_speed_10m", "wind_direction_10m",
            "wind_gusts_10m", NULL};
        int nh = nums(janas_json_get(hl, "time"), v[7], WX_HOURS);
        for (int i = 0; i < 7; i++)
            for (int j = nums(janas_json_get(hl, k[i]), v[i], WX_HOURS); j < nh;
                 j++)
                v[i][j] = NAN;
        for (int i = 0; i < nh; i++) {
            struct wx_hour *h = &f->hour[i];
            str_at(janas_json_get(hl, "time"), i, h->time, sizeof h->time);
            h->code = isnan(v[0][i]) ? -1 : (int)v[0][i];
            h->temp = v[1][i];
            h->rain = v[2][i];
            h->rain_prob = v[3][i];
            h->wind = v[4][i];
            h->wind_dir = v[5][i];
            h->gust = v[6][i];
        }
        f->n_hour = nh;
    }
    janas_json_free(d);
    if (!f->has_now && !f->n_day) {
        snprintf(err, err_len, "Open-Meteo: an answer without a forecast");
        return -1;
    }
    return 0;
}

int wx_om_forecast(const struct wx_place *p, int days, struct wx_forecast *f,
                   char *err, size_t err_len)
{
    if (days < 1)
        days = 1;
    if (days > WX_DAYS)
        days = WX_DAYS;
    /* the place's zone when known: "auto" gives a sea's nautical one */
    char tz[96] = "auto";
    if (p->tz[0]) {
        size_t k = 0;
        for (const char *c = p->tz; *c && k + 4 < sizeof tz; c++)
            k += (size_t)snprintf(tz + k, sizeof tz - k,
                                  *c == '/' ? "%%2F" : "%c", *c);
    }
    char url[1536];
    snprintf(url, sizeof url,
             "https://api.open-meteo.com/v1/forecast?latitude=%.4f&"
             "longitude=%.4f&cell_selection=land&current=temperature_2m,"
             "relative_humidity_2m,apparent_temperature,precipitation,"
             "weather_code,cloud_cover,pressure_msl,wind_speed_10m,"
             "wind_direction_10m,wind_gusts_10m,is_day&hourly=weather_code,"
             "temperature_2m,precipitation,precipitation_probability,"
             "wind_speed_10m,wind_direction_10m,wind_gusts_10m&daily="
             "weather_code,temperature_2m_min,temperature_2m_max,"
             "precipitation_sum,precipitation_probability_max,"
             "wind_speed_10m_max,wind_gusts_10m_max,"
             "wind_direction_10m_dominant,uv_index_max,sunrise,sunset&"
             "timezone=%s&forecast_days=%d&forecast_hours=%d",
             p->lat, p->lon, tz, days, WX_HOURS);
    struct janas_buf b = {0};
    char why[256];
    int status = wx_fetch(url, OM_KEEP_S, 0, &b, why, sizeof why);
    int r;
    if (status < 0) {
        snprintf(err, err_len, "Open-Meteo did not answer: %s", why);
        r = -1;
    } else {
        r = wx_om_parse(b.p ? b.p : "", b.n, f, err, err_len);
        if (r == 0 && status != 200) {
            snprintf(err, err_len, "Open-Meteo: HTTP %d", status);
            r = -1;
        }
    }
    janas_buf_free(&b);
    return r;
}

/* The answer for n points: an array of them, or the one object. */
static const struct janas_json *point(const struct janas_json *r, int i, int n)
{
    if (r && r->type == JANAS_JSON_OBJECT)
        return n == 1 && i == 0 ? r : NULL;
    const struct janas_json *v =
        r && r->type == JANAS_JSON_ARRAY ? r->child : NULL;
    while (v && i-- > 0)
        v = v->next;
    return v;
}

int wx_om_marine_parse(const char *text, size_t len, struct wx_sea *s, int n,
                       char *err, size_t err_len)
{
    struct janas_buf b = {.p = (char *)text, .n = len};
    struct janas_json_doc *d = wx_parse(&b, "Open-Meteo", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *r = janas_json_root(d);
    if (r->type == JANAS_JSON_OBJECT && error_of(r, err, err_len)) {
        janas_json_free(d);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        const struct janas_json *o = point(r, i, n);
        const struct janas_json *c = janas_json_get(o, "current");
        s[i].cell_lat = num(o, "latitude");
        s[i].cell_lon = num(o, "longitude");
        s[i].wave = num(c, "wave_height");
        s[i].wave_dir = num(c, "wave_direction");
        s[i].period = num(c, "wave_period");
        s[i].wind_wave = num(c, "wind_wave_height");
        s[i].swell = num(c, "swell_wave_height");
        s[i].swell_dir = num(c, "swell_wave_direction");
        s[i].sst = num(c, "sea_surface_temperature");
        snprintf(s[i].time, sizeof s[i].time, "%s",
                 janas_json_str(janas_json_get(c, "time"))
                     ? janas_json_str(janas_json_get(c, "time"))
                     : "");
        s[i].ok = !isnan(s[i].wave);
        const struct janas_json *dl = janas_json_get(o, "daily");
        double v[3][WX_DAYS];
        int nd = nums(janas_json_get(dl, "time"), v[0], WX_DAYS);
        int a = nums(janas_json_get(dl, "wave_height_max"), v[0], WX_DAYS);
        int bdir =
            nums(janas_json_get(dl, "wave_direction_dominant"), v[1], WX_DAYS);
        int c3 = nums(janas_json_get(dl, "wave_period_max"), v[2], WX_DAYS);
        for (int j = 0; j < nd; j++) {
            struct wx_sea_day *y = &s[i].day[j];
            str_at(janas_json_get(dl, "time"), j, y->date, sizeof y->date);
            y->wave_max = j < a ? v[0][j] : NAN;
            y->wave_dir = j < bdir ? v[1][j] : NAN;
            y->period_max = j < c3 ? v[2][j] : NAN;
            y->wind_max = y->gust_max = y->wind_dir = NAN;
        }
        s[i].n_day = nd;
    }
    janas_json_free(d);
    return 0;
}

int wx_om_wind_parse(const char *text, size_t len, struct wx_sea *s, int n,
                     char *err, size_t err_len)
{
    struct janas_buf b = {.p = (char *)text, .n = len};
    struct janas_json_doc *d = wx_parse(&b, "Open-Meteo", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *r = janas_json_root(d);
    if (r->type == JANAS_JSON_OBJECT && error_of(r, err, err_len)) {
        janas_json_free(d);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        const struct janas_json *o = point(r, i, n);
        const struct janas_json *c = janas_json_get(o, "current");
        s[i].wind = num(c, "wind_speed_10m");
        s[i].wind_dir = num(c, "wind_direction_10m");
        s[i].gust = num(c, "wind_gusts_10m");
        const struct janas_json *dl = janas_json_get(o, "daily");
        double v[3][WX_DAYS];
        int a = nums(janas_json_get(dl, "wind_speed_10m_max"), v[0], WX_DAYS);
        int g = nums(janas_json_get(dl, "wind_gusts_10m_max"), v[1], WX_DAYS);
        int w = nums(janas_json_get(dl, "wind_direction_10m_dominant"), v[2],
                     WX_DAYS);
        for (int j = 0; j < s[i].n_day; j++) {
            s[i].day[j].wind_max = j < a ? v[0][j] : NAN;
            s[i].day[j].gust_max = j < g ? v[1][j] : NAN;
            s[i].day[j].wind_dir = j < w ? v[2][j] : NAN;
        }
    }
    janas_json_free(d);
    return 0;
}

static void url_tz(struct janas_buf *u, const char *tz)
{
    for (const char *c = tz && *tz ? tz : "GMT"; *c; c++)
        if (*c == '/')
            janas_buf_puts(u, "%2F");
        else
            janas_buf_put(u, c, 1);
}

int wx_om_marine(struct wx_sea *s, int n, int days, const char *tz, char *err,
                 size_t err_len)
{
    if (days < 1)
        days = 1;
    if (days > 8) /* the sea model's reach */
        days = 8;
    struct janas_buf la = {0}, lo = {0};
    for (int i = 0; i < n; i++) {
        janas_buf_printf(&la, "%s%.3f", i ? "," : "", s[i].lat);
        janas_buf_printf(&lo, "%s%.3f", i ? "," : "", s[i].lon);
    }
    struct janas_buf u = {0}, b = {0};
    janas_buf_printf(&u,
                     "https://marine-api.open-meteo.com/v1/marine?latitude=%s&"
                     "longitude=%s&current=wave_height,wave_direction,"
                     "wave_period,wind_wave_height,swell_wave_height,"
                     "swell_wave_direction,sea_surface_temperature&daily="
                     "wave_height_max,wave_direction_dominant,wave_period_max&"
                     "forecast_days=%d&timezone=",
                     la.p ? la.p : "", lo.p ? lo.p : "", days);
    url_tz(&u, tz);
    janas_buf_put(&u, "", 1);
    u.n--;
    for (int i = 0; i < n; i++)
        s[i].wind = s[i].wind_dir = s[i].gust = NAN;
    char why[256];
    int r = -1;
    int status = u.oom ? -1 : wx_fetch(u.p, OM_KEEP_S, 0, &b, why, sizeof why);
    if (status == 200)
        r = wx_om_marine_parse(b.p, b.n, s, n, err, err_len);
    else if (status < 0)
        snprintf(err, err_len, "Open-Meteo did not answer: %s", why);
    else
        snprintf(err, err_len, "Open-Meteo: HTTP %d", status);
    u.n = 0;
    janas_buf_printf(&u,
                     "https://api.open-meteo.com/v1/forecast?latitude=%s&"
                     "longitude=%s&current=wind_speed_10m,wind_direction_10m,"
                     "wind_gusts_10m&daily=wind_speed_10m_max,"
                     "wind_gusts_10m_max,wind_direction_10m_dominant&"
                     "forecast_days=%d&timezone=",
                     la.p ? la.p : "", lo.p ? lo.p : "", days);
    url_tz(&u, tz);
    janas_buf_put(&u, "", 1);
    /* without the wind, the waves alone */
    if (r == 0 && !u.oom &&
        wx_fetch(u.p, OM_KEEP_S, 0, &b, why, sizeof why) == 200)
        wx_om_wind_parse(b.p, b.n, s, n, why, sizeof why);
    janas_buf_free(&la);
    janas_buf_free(&lo);
    janas_buf_free(&u);
    janas_buf_free(&b);
    return r;
}
