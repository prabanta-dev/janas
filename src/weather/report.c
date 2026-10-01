/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * report.c - weather_now and weather_forecast (see weather.h): Open-Meteo's
 * forecast, MET Norway's when it does not answer, and next to the weather
 * now what the nearest station measured. A forecast is a model's value at
 * a cell of its grid and a station's is a measure at a known place and
 * minute: both are given, said for what they are, and when they disagree
 * it is said too, so that the model telling them need not guess which.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/geo.h"
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

static void sky(int code, const char *symbol, char *out, size_t cap)
{
    const char *t = code >= 0 ? wx_code_text(code) : NULL;
    if (t)
        snprintf(out, cap, "%s", t);
    else if (symbol && *symbol)
        wx_symbol_text(symbol, out, cap);
    else
        snprintf(out, cap, "not given");
}

/* "-SHRA" -> "light rain showers": the commonest codes of present
   weather; the code itself is given too */
static void wx_words(const char *wx, char *out, size_t cap)
{
    static const struct {
        const char *code, *words;
    } w[] = {
        {"TS", "thunderstorm "}, {"SH", "showers of "}, {"FZ", "freezing "},
        {"RA", "rain "},         {"DZ", "drizzle "},    {"SN", "snow "},
        {"GR", "hail "},         {"GS", "small hail "}, {"FG", "fog "},
        {"BR", "mist "},         {"HZ", "haze "},       {"VC", "nearby "}};
    size_t n = 0;
    out[0] = 0;
    const char *s = wx;
    if (*s == '-' || *s == '+') {
        n += (size_t)snprintf(out, cap, "%s", *s == '-' ? "light " : "heavy ");
        s++;
    }
    while (*s && n < cap) {
        size_t i;
        for (i = 0; i < sizeof w / sizeof w[0]; i++)
            if (strncmp(s, w[i].code, 2) == 0) {
                n += (size_t)snprintf(out + n, cap - n, "%s", w[i].words);
                s += 2;
                break;
            }
        if (i == sizeof w / sizeof w[0])
            s++;
    }
    while (n && out[n - 1] == ' ')
        out[--n] = 0;
}

static void visibility(const char *v, char *out, size_t cap)
{
    if (!v[0]) {
        snprintf(out, cap, "not given");
        return;
    }
    if (strchr(v, '+')) {
        snprintf(out, cap, "%.0f km or more", atof(v) * 1.609);
        return;
    }
    double mi = atof(v);
    const char *frac = strchr(v, '/'); /* "1 1/2", "1/4" */
    if (frac) {
        const char *sp = strchr(v, ' ');
        double whole = sp ? atof(v) : 0;
        int a = atoi(sp ? sp + 1 : v), b = atoi(frac + 1);
        mi = whole + (b ? (double)a / b : 0);
    }
    snprintf(out, cap, "%.1f km", mi * 1.609);
}

static void observation(const struct wx_place *p, const struct wx_obs *o,
                        const struct wx_forecast *f, struct janas_buf *out)
{
    char when[64], dir[64], vis[32], words[96];
    wx_local_time(o->at, p->tz, when, sizeof when);
    double bearing = geo_bearing(p->lat, p->lon, o->lat, o->lon);
    long ago = (long)difftime(time(NULL), o->at) / 60;
    janas_buf_printf(out,
                     "Measured by the station of %s (%s), %.0f km %s of the "
                     "place, at %s (%ld minutes ago): %.0f °C, dew point "
                     "%.0f °C",
                     o->name, o->icao, o->km, geo_compass(bearing), when, ago,
                     o->temp, o->dew);
    if (!isnan(o->wind_kt)) {
        if (o->wind_kt < 1)
            janas_buf_puts(out, ", calm");
        else if (o->variable_wind)
            janas_buf_printf(out,
                             ", wind of variable direction at %.0f kt "
                             "(%.0f km/h)",
                             o->wind_kt, o->wind_kt * 1.852);
        else {
            wx_from_dir(o->wind_dir, dir, sizeof dir);
            janas_buf_printf(out, ", wind %s at %.0f kt (%.0f km/h, %s)", dir,
                             o->wind_kt, o->wind_kt * 1.852,
                             wx_wind_words(o->wind_kt * 1.852));
        }
        if (!isnan(o->gust_kt))
            janas_buf_printf(out, ", gusts %.0f kt (%.0f km/h)", o->gust_kt,
                             o->gust_kt * 1.852);
    }
    visibility(o->visib, vis, sizeof vis);
    janas_buf_printf(out, ", visibility %s", vis);
    if (o->wx[0]) {
        wx_words(o->wx, words, sizeof words);
        janas_buf_printf(out, ", %s (%s)", words[0] ? words : o->wx, o->wx);
    } else {
        janas_buf_puts(out, ", no rain or other weather");
    }
    if (o->clouds[0])
        janas_buf_printf(out, ", %s", o->clouds);
    if (!isnan(o->pressure))
        janas_buf_printf(out, ", pressure %.0f hPa", o->pressure);
    janas_buf_printf(out, ". As reported: %s\n", o->raw);
    if (f->has_now && !isnan(f->now.temp) && fabs(f->now.temp - o->temp) >= 3)
        janas_buf_printf(out,
                         "The model's temperature and the station's differ "
                         "by %.0f °C: the station's is the measure (the "
                         "station is %.0f km away).\n",
                         fabs(f->now.temp - o->temp), o->km);
    int st_rain = strstr(o->wx, "RA") || strstr(o->wx, "DZ") ||
                  strstr(o->wx, "TS") || strstr(o->wx, "SN");
    int fc_rain = f->has_now && (f->now.code >= 50 ||
                                 (!isnan(f->now.precip) && f->now.precip > 0));
    if (f->has_now && f->now.code >= 0 && st_rain != fc_rain)
        janas_buf_printf(out,
                         "The model and the station disagree on rain: the "
                         "station %s, the model %s.\n",
                         st_rain ? "reports it" : "reports none",
                         fc_rain ? "has it" : "has none");
}

static void source_line(const struct wx_place *p, const struct wx_forecast *f,
                        struct janas_buf *out)
{
    double km = geo_km(p->lat, p->lon, f->grid_lat, f->grid_lon);
    janas_buf_printf(out, "Forecast: %s's model, read at %.3f, %.3f", f->source,
                     f->grid_lat, f->grid_lon);
    if (!isnan(f->grid_elev))
        janas_buf_printf(out, " (%.0f m)", f->grid_elev);
    janas_buf_printf(out,
                     ", %.1f km from the place: a model's value, not a "
                     "measure.\n",
                     km);
}

int wx_tool_now(const struct janas_json *args, struct janas_buf *b)
{
    struct wx_place p;
    if (wx_arg_place(args, &p, b) != 0)
        return 0;
    struct wx_forecast *f = calloc(1, sizeof *f);
    struct janas_buf out = {0};
    char err[800], name[256], s[96], dir[64];
    wx_place_name(&p, name, sizeof name);
    if (!f || forecast(&p, 1, f, err, sizeof err) != 0) {
        janas_buf_printf(&out, "No forecast for %s: %s.", name,
                         f ? err : "out of memory");
        free(f);
        return wx_result(&out, b, 1);
    }
    if (!p.tz[0])
        snprintf(p.tz, sizeof p.tz, "%s", f->tz);
    const struct wx_now *w = &f->now;
    const char *T = strchr(w->time, 'T');
    janas_buf_printf(&out, "The weather now in %s, at %.5s %s (local time):\n",
                     name, T ? T + 1 : w->time, f->tz_abbr);
    sky(w->code, w->symbol, s, sizeof s);
    janas_buf_printf(&out, "- sky: %s", s);
    if (!isnan(w->cloud))
        janas_buf_printf(&out, " (cloud cover %.0f%%)", w->cloud);
    janas_buf_printf(&out, "\n- temperature %.1f °C", w->temp);
    if (!isnan(w->feels))
        janas_buf_printf(&out, ", feels like %.1f °C", w->feels);
    if (!isnan(w->humidity))
        janas_buf_printf(&out, "; humidity %.0f%%", w->humidity);
    wx_from_dir(w->wind_dir, dir, sizeof dir);
    janas_buf_printf(&out, "\n- wind %s at %.0f km/h (%s)", dir, w->wind,
                     wx_wind_words(w->wind));
    if (!isnan(w->gust))
        janas_buf_printf(&out, ", gusts up to %.0f km/h", w->gust);
    if (!isnan(w->precip))
        janas_buf_printf(&out,
                         "\n- rain in the last quarter of an hour: "
                         "%.1f mm",
                         w->precip);
    if (!isnan(w->pressure))
        janas_buf_printf(&out, "\n- pressure at sea level %.0f hPa",
                         w->pressure);
    if (f->n_day) {
        const struct wx_day *y = &f->day[0];
        janas_buf_printf(&out, "\n- today: lowest %.1f °C, highest %.1f °C",
                         y->tmin, y->tmax);
        if (!isnan(y->rain_prob))
            janas_buf_printf(&out, ", chance of rain %.0f%%", y->rain_prob);
        if (y->sunrise[0])
            janas_buf_printf(&out, "; sunrise %s, sunset %s", y->sunrise,
                             y->sunset);
    }
    janas_buf_puts(&out, "\n");
    source_line(&p, f, &out);

    struct wx_obs o;
    char why[384];
    int r = wx_metar_near(p.lat, p.lon, STATION_KM, &o, why, sizeof why);
    if (r == 1)
        observation(&p, &o, f, &out);
    else if (r == 0)
        janas_buf_printf(&out, "No weather station reported within %d km.\n",
                         STATION_KM);
    else
        janas_buf_printf(&out, "The stations could not be asked (%s).\n", why);
    janas_buf_printf(&out, "Sources: %s (%s, CC BY 4.0)%s.", f->source,
                     strcmp(f->source, "Open-Meteo") == 0 ? "open-meteo.com"
                                                          : "api.met.no",
                     r == 1 ? "; the station's METAR from aviationweather.gov "
                              "(NOAA)"
                            : "");
    free(f);
    wx_place_note(p.how, &out);
    return wx_result(&out, b, 0);
}

static void day_line(const struct wx_day *y, struct janas_buf *out)
{
    char name[48], s[96], dir[64];
    wx_day_name(y->date, name, sizeof name);
    sky(y->code, y->symbol, s, sizeof s);
    janas_buf_printf(out, "- %s: %s; %.1f to %.1f °C", name, s, y->tmin,
                     y->tmax);
    if (!isnan(y->rain)) {
        if (y->rain < 0.1)
            janas_buf_puts(out, "; no rain");
        else
            janas_buf_printf(out, "; rain %.1f mm", y->rain);
    }
    if (!isnan(y->rain_prob))
        janas_buf_printf(out, " (chance %.0f%%)", y->rain_prob);
    if (!isnan(y->wind)) {
        wx_from_dir(y->wind_dir, dir, sizeof dir);
        janas_buf_printf(out, "; wind up to %.0f km/h (%s) %s", y->wind,
                         wx_wind_words(y->wind), dir);
    }
    if (!isnan(y->gust))
        janas_buf_printf(out, ", gusts up to %.0f km/h", y->gust);
    if (!isnan(y->uv))
        janas_buf_printf(out, "; UV index %.0f", y->uv);
    if (y->sunrise[0])
        janas_buf_printf(out, "; sunrise %s, sunset %s", y->sunrise, y->sunset);
    janas_buf_puts(out, "\n");
}

static void hour_line(const struct wx_hour *h, struct janas_buf *out)
{
    char s[96], dir[64];
    sky(h->code, h->symbol, s, sizeof s);
    const char *T = strchr(h->time, 'T');
    janas_buf_printf(out, "- %.5s: %s, %.1f °C", T ? T + 1 : h->time, s,
                     h->temp);
    if (!isnan(h->rain) && h->rain >= 0.1)
        janas_buf_printf(out, ", rain %.1f mm", h->rain);
    if (!isnan(h->rain_prob))
        janas_buf_printf(out, " (chance %.0f%%)", h->rain_prob);
    wx_from_dir(h->wind_dir, dir, sizeof dir);
    janas_buf_printf(out, ", wind %.0f km/h %s", h->wind, dir);
    if (!isnan(h->gust))
        janas_buf_printf(out, ", gusts %.0f", h->gust);
    janas_buf_puts(out, "\n");
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
    struct janas_buf out = {0};
    char err[800], name[256];
    wx_place_name(&p, name, sizeof name);
    if (!f || forecast(&p, days, f, err, sizeof err) != 0) {
        janas_buf_printf(&out, "No forecast for %s: %s.", name,
                         f ? err : "out of memory");
        free(f);
        return wx_result(&out, b, 1);
    }
    janas_buf_printf(&out,
                     "Forecast for %s, by the day (times in local time, "
                     "%s):\n",
                     name, f->tz_abbr);
    for (int i = 0; i < f->n_day && i < days; i++)
        day_line(&f->day[i], &out);
    if (f->n_day < days)
        janas_buf_printf(&out, "(%s gives %d days.)\n", f->source, f->n_day);
    int step = hourly ? 1 : 3, last = hourly ? WX_HOURS : 24;
    if (f->n_hour) {
        janas_buf_printf(&out, "The next %d hours, every %s:\n",
                         last < f->n_hour ? last : f->n_hour,
                         step == 1 ? "hour" : "three hours");
        char prev[11] = "";
        for (int i = 0; i < f->n_hour && i < last; i += step) {
            if (strncmp(prev, f->hour[i].time, 10) != 0) {
                char dn[48], date[11];
                snprintf(date, sizeof date, "%.10s", f->hour[i].time);
                wx_day_name(date, dn, sizeof dn);
                janas_buf_printf(&out, "%s\n", dn);
                snprintf(prev, sizeof prev, "%s", date);
            }
            hour_line(&f->hour[i], &out);
        }
    }
    source_line(&p, f, &out);
    janas_buf_printf(&out, "Source: %s (%s, CC BY 4.0).", f->source,
                     strcmp(f->source, "Open-Meteo") == 0 ? "open-meteo.com"
                                                          : "api.met.no");
    free(f);
    wx_place_note(p.how, &out);
    return wx_result(&out, b, 0);
}
