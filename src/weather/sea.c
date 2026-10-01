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

#include "common/geo.h"
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

static void point_line(const struct wx_sea *s, int days, int with_place,
                       struct janas_buf *out)
{
    char dir[64];
    janas_buf_puts(out, "- ");
    if (with_place) {
        geo_describe(s->lat, s->lon, out);
        janas_buf_puts(out, ": ");
    }
    wx_from_dir(s->wave_dir, dir, sizeof dir);
    janas_buf_printf(out, "waves %.1f m (%s) %s", s->wave,
                     wx_sea_words(s->wave), dir);
    if (!isnan(s->period))
        janas_buf_printf(out, ", period %.0f s", s->period);
    if (!isnan(s->wind_wave))
        janas_buf_printf(out, "; wind waves %.1f m", s->wind_wave);
    if (!isnan(s->swell)) {
        wx_from_dir(s->swell_dir, dir, sizeof dir);
        janas_buf_printf(out, ", swell %.1f m %s", s->swell, dir);
    }
    if (!isnan(s->sst))
        janas_buf_printf(out, "; water %.1f °C", s->sst);
    if (!isnan(s->wind)) {
        wx_from_dir(s->wind_dir, dir, sizeof dir);
        janas_buf_printf(out, "; wind %.0f km/h (%s) %s", s->wind,
                         wx_wind_words(s->wind), dir);
        if (!isnan(s->gust))
            janas_buf_printf(out, ", gusts %.0f km/h", s->gust);
    }
    janas_buf_puts(out, "\n");
    for (int d = 0; d < s->n_day && d < days; d++) {
        const struct wx_sea_day *y = &s->day[d];
        char name[48];
        wx_day_name(y->date, name, sizeof name);
        wx_from_dir(y->wave_dir, dir, sizeof dir);
        janas_buf_printf(out, "    %s: waves up to %.1f m (%s) %s", name,
                         y->wave_max, wx_sea_words(y->wave_max), dir);
        if (!isnan(y->wind_max)) {
            janas_buf_printf(out, ", wind up to %.0f km/h (%s)", y->wind_max,
                             wx_wind_words(y->wind_max));
            if (!isnan(y->gust_max))
                janas_buf_printf(out, ", gusts %.0f", y->gust_max);
        }
        janas_buf_puts(out, "\n");
    }
}

static void summary(const struct wx_sea *s, int n, int days,
                    struct janas_buf *out)
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
    janas_buf_printf(out, "Now, at %d points: waves %.1f to %.1f m (%s", ok, lo,
                     hi, wx_sea_words(lo));
    if (strcmp(wx_sea_words(lo), wx_sea_words(hi)) != 0)
        janas_buf_printf(out, " to %s", wx_sea_words(hi));
    janas_buf_puts(out, ")");
    if (!isnan(t_lo))
        janas_buf_printf(out, ", water %.1f to %.1f °C", t_lo, t_hi);
    janas_buf_puts(out, ".\n");
    for (int d = 0; d < days; d++) {
        int at = -1;
        for (int i = 0; i < n; i++)
            if (s[i].ok && d < s[i].n_day && !isnan(s[i].day[d].wave_max) &&
                (at < 0 || s[i].day[d].wave_max > s[at].day[d].wave_max))
                at = i;
        if (at < 0)
            continue;
        char name[48];
        wx_day_name(s[at].day[d].date, name, sizeof name);
        janas_buf_printf(out, "%s: highest waves %.1f m (%s), ", name,
                         s[at].day[d].wave_max,
                         wx_sea_words(s[at].day[d].wave_max));
        geo_describe(s[at].lat, s[at].lon, out);
        janas_buf_puts(out, ".\n");
    }
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
    janas_buf_printf(&out,
                     "The state of %s at %.5s local time (%s), and "
                     "the next %d days:\n",
                     title, T ? T + 1 : s[0].time, tz[0] ? tz : "UTC", days);
    if (area)
        summary(s, n, days, &out);
    for (int i = 0; i < n; i++)
        if (s[i].ok)
            point_line(&s[i], days, area != NULL, &out);
    janas_buf_puts(&out, "The waves' height is the significant height (the "
                         "mean of the highest third); single waves can be "
                         "almost twice as high. Source: Open-Meteo's sea "
                         "and weather models (open-meteo.com, CC BY 4.0): "
                         "forecasts, not measures.");
    wx_place_note(how, &out);
    return wx_result(&out, b, 0);
}
