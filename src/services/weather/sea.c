/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * sea.c - weather_sea (see weather.h): the state of the sea from
 * Open-Meteo's sea model, for a whole sea at points spread across it (the
 * same areas as janas-flights', from Natural Earth's polygons), or off a
 * coastal place. Each point is said in words, the heights of the waves
 * with their names on the Douglas scale, and the whole first: from what to
 * what, and where the sea will be roughest in the next days.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "services/common/geo.h"
#include "weather.h"

#define DEG(x) ((x) * 1e-5)
#define GRID 5      /* candidate points a side, across the area's box */
#define COAST_KM 30 /* off a place: the sea model's cell within this */

/* Points over the area, spread: of a GRID x GRID net over its box those
   over the sea, at most WX_SEA_POINTS of them taken evenly. */
static int area_points(const struct geo_area *a, struct wx_sea *s)
{
    double c_lat[GRID * GRID], c_lon[GRID * GRID];
    int n = 0;
    for (int g = GRID; g <= 2 * GRID && n < 3; g += 2) {
        n = 0;
        for (int i = 0; i < g; i++)
            for (int j = 0; j < g; j++) {
                double lat = DEG(a->s_lat_min) +
                             (i + 0.5) * DEG(a->s_lat_max - a->s_lat_min) / g;
                double lon = DEG(a->s_lon_min) +
                             (j + 0.5) * DEG(a->s_lon_max - a->s_lon_min) / g;
                if (geo_area_has(a, lat, lon) && n < GRID * GRID) {
                    c_lat[n] = lat;
                    c_lon[n] = lon;
                    n++;
                }
            }
    }
    int k = n < WX_SEA_POINTS ? n : WX_SEA_POINTS;
    for (int i = 0; i < k; i++) {
        int at = k == n ? i : (int)((double)i * n / k);
        memset(&s[i], 0, sizeof s[i]);
        s[i].lat = c_lat[at];
        s[i].lon = c_lon[at];
    }
    return k;
}

/* One point: now, and its days. */
static void point_json(const struct wx_sea *s, int days, int with_place,
                       struct janas_buf *d)
{
    janas_buf_puts(d, "{\"wave\": ");
    janas_buf_printf(d, "%.1f, \"wave_code\": %d", s->wave,
                     wx_douglas(s->wave));
    if (with_place) {
        janas_buf_puts(d, ", \"where\": ");
        geo_describe_json(s->lat, s->lon, d);
    }
    if (!isnan(s->wave_dir))
        wx_jstr(d, "wave_from", geo_compass(s->wave_dir));
    wx_jnum(d, "period", s->period, 0);
    if (!isnan(s->wind_wave)) {
        janas_buf_puts(d, ", \"has_wind_wave\": true");
        wx_jnum(d, "wind_wave", s->wind_wave, 1);
    }
    if (!isnan(s->swell)) {
        janas_buf_puts(d, ", \"has_swell\": true");
        wx_jnum(d, "swell", s->swell, 1);
        if (!isnan(s->swell_dir))
            wx_jstr(d, "swell_from", geo_compass(s->swell_dir));
    }
    wx_jnum(d, "sst", s->sst, 1);
    if (!isnan(s->wind)) {
        janas_buf_puts(d, ", \"has_wind\": true");
        wx_jwind(d, s->wind, s->wind_dir);
        wx_jnum(d, "gust", s->gust, 0);
    }
    janas_buf_puts(d, ", \"days\": [");
    for (int k = 0; k < s->n_day && k < days; k++) {
        const struct wx_sea_day *y = &s->day[k];
        janas_buf_puts(d, k ? ", {" : "{");
        wx_jdate(d, y->date);
        janas_buf_printf(d, ", \"max\": %.1f, \"code\": %d", y->wave_max,
                         wx_douglas(y->wave_max));
        if (!isnan(y->wave_dir))
            wx_jstr(d, "from", geo_compass(y->wave_dir));
        if (!isnan(y->wind_max)) {
            janas_buf_puts(d, ", \"has_wind\": true");
            wx_jnum(d, "wind", y->wind_max, 0);
            janas_buf_printf(d, ", \"bft\": %d", wx_beaufort(y->wind_max));
            wx_jnum(d, "gust", y->gust_max, 0);
        }
        janas_buf_puts(d, "}");
    }
    janas_buf_puts(d, "]}");
}

/* A whole sea: from what to what now, and where it is roughest each day. */
static void summary_json(const struct wx_sea *s, int n, int days,
                         struct janas_buf *d)
{
    double lo = NAN, hi = NAN, t_lo = NAN, t_hi = NAN;
    int ok = 0;
    for (int i = 0; i < n; i++) {
        if (!s[i].ok)
            continue;
        ok++;
        if (isnan(lo) || s[i].wave < lo)
            lo = s[i].wave;
        if (isnan(hi) || s[i].wave > hi)
            hi = s[i].wave;
        if (!isnan(s[i].sst)) {
            if (isnan(t_lo) || s[i].sst < t_lo)
                t_lo = s[i].sst;
            if (isnan(t_hi) || s[i].sst > t_hi)
                t_hi = s[i].sst;
        }
    }
    janas_buf_printf(d,
                     ", \"summary\": {\"points\": %d, \"lo\": %.1f, \"hi\": "
                     "%.1f, \"lo_code\": %d, \"hi_code\": %d",
                     ok, lo, hi, wx_douglas(lo), wx_douglas(hi));
    if (wx_douglas(lo) != wx_douglas(hi))
        janas_buf_puts(d, ", \"hi_other\": true");
    wx_jnum(d, "t_lo", t_lo, 1);
    wx_jnum(d, "t_hi", t_hi, 1);
    janas_buf_puts(d, ", \"days\": [");
    int any = 0;
    for (int k = 0; k < days; k++) {
        int at = -1;
        for (int i = 0; i < n; i++)
            if (s[i].ok && k < s[i].n_day && !isnan(s[i].day[k].wave_max) &&
                (at < 0 || s[i].day[k].wave_max > s[at].day[k].wave_max))
                at = i;
        if (at < 0)
            continue;
        janas_buf_puts(d, any++ ? ", {" : "{");
        wx_jdate(d, s[at].day[k].date);
        janas_buf_printf(d, ", \"max\": %.1f, \"code\": %d, \"where\": ",
                         s[at].day[k].wave_max,
                         wx_douglas(s[at].day[k].wave_max));
        geo_describe_json(s[at].lat, s[at].lon, d);
        janas_buf_puts(d, "}");
    }
    janas_buf_puts(d, "]}");
}

int wx_tool_sea(const struct janas_json *args, struct janas_buf *b)
{
    int days = (int)wx_arg_num(args, "days", 2);
    if (days < 1)
        days = 1;
    if (days > 8)
        days = 8;
    const char *area = wx_arg_str(args, "area");
    struct wx_sea s[WX_SEA_POINTS];
    struct janas_buf out = {0};
    char err[512], tz[48] = "", title[256], how[96] = "";
    int n;
    struct geo_area a;
    struct wx_place place = {0};
    if (area) {
        if (!geo_area_find(area, &a)) {
            janas_buf_printf(&out,
                             "No sea called \"%s\" known: give its name "
                             "in English (Tyrrhenian Sea, Adriatic "
                             "Sea, Ligurian Sea) or a coastal place.",
                             area);
            return wx_result(&out, b, 1);
        }
        n = area_points(&a, s);
        double km;
        const struct geo_airport *ap =
            geo_airport_near(DEG(a.lat_min + a.lat_max) / 2,
                             DEG(a.lon_min + a.lon_max) / 2, &km);
        if (ap)
            snprintf(tz, sizeof tz, "%s", geo_tz_name(ap->tz));
        snprintf(title, sizeof title, "the %s", a.name);
    } else {
        struct wx_place p;
        if (wx_arg_place(args, &p, b) != 0)
            return 0;
        place = p;
        snprintf(how, sizeof how, "%s", p.how);
        memset(&s[0], 0, sizeof s[0]);
        s[0].lat = p.lat;
        s[0].lon = p.lon;
        n = 1;
        snprintf(tz, sizeof tz, "%s", p.tz);
        char name[200];
        wx_place_name(&p, name, sizeof name);
        snprintf(title, sizeof title, "the sea off %s", name);
    }
    if (n == 0) {
        janas_buf_printf(&out, "No point of %s found to ask about.", title);
        return wx_result(&out, b, 1);
    }
    if (wx_om_marine(s, n, days, tz, err, sizeof err) != 0) {
        janas_buf_printf(&out, "No forecast of %s: %s.", title, err);
        return wx_result(&out, b, 1);
    }
    if (!area && (!s[0].ok || geo_km(s[0].lat, s[0].lon, s[0].cell_lat,
                                     s[0].cell_lon) > COAST_KM)) {
        janas_buf_printf(&out,
                         "There is no sea within %d km of the place for "
                         "the sea model; ask about a coastal place or "
                         "a sea by its name.",
                         COAST_KM);
        return wx_result(&out, b, 1);
    }
    const char *T = strchr(s[0].time, 'T');
    struct janas_buf d = {0};
    if (area) {
        janas_buf_puts(&d, "{\"sea\": ");
        janas_json_write_str(&d, a.name, strlen(a.name));
    } else {
        wx_jplace(&d, &place);
    }
    janas_buf_printf(&d, ", \"time\": \"%.5s\"", T ? T + 1 : s[0].time);
    wx_jstr(&d, "tz", tz[0] ? tz : "UTC");
    janas_buf_printf(&d, ", \"days\": %d", days);
    if (area)
        summary_json(s, n, days, &d);
    janas_buf_puts(&d, ", \"points\": [");
    int k = 0;
    for (int i = 0; i < n; i++)
        if (s[i].ok) {
            janas_buf_puts(&d, k++ ? ", " : "");
            point_json(&s[i], days, area != NULL, &d);
        }
    janas_buf_puts(&d, "]}");
    (void)how;
    return wx_answer(b, "weather_sea", &d, WX_SEA_LAYOUT, WX_SEA_BRIEF);
}
