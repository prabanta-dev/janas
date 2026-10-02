/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * metar.c - what a station measured (see weather.h): the METAR of the
 * airports, the hourly observations pilots read, from the Aviation Weather
 * Center of the United States' weather service (aviationweather.gov, no
 * key, public domain). A forecast is a model's value at a cell of its
 * grid; this is a thermometer and an anemometer at a known place and
 * minute, the measure to set the forecast against.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "services/common/geo.h"
#include "weather.h"

#define METAR_KEEP_S 600 /* a METAR comes every half hour or hour */

static double obs_num(const struct janas_json *o, const char *k)
{
    const struct janas_json *v = janas_json_get(o, k);
    return v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, NAN) : NAN;
}

static const char *cover_words(const char *c)
{
    static const struct {
        const char *code, *words;
    } w[] = {{"FEW", "few clouds"},
             {"SCT", "scattered clouds"},
             {"BKN", "broken clouds"},
             {"OVC", "overcast"},
             {"VV", "sky obscured"},
             {"CLR", "clear"},
             {"SKC", "clear"},
             {"NSC", "no significant cloud"},
             {"NCD", "no cloud detected"},
             {"CAVOK", "no significant cloud"}};
    for (size_t i = 0; i < sizeof w / sizeof w[0]; i++)
        if (c && strcmp(c, w[i].code) == 0)
            return w[i].words;
    return c ? c : "";
}

static void clouds_of(const struct janas_json *cl, char *out, size_t cap,
                      struct wx_obs *o)
{
    size_t n = 0;
    out[0] = 0;
    o->n_layers = 0;
    for (const struct janas_json *c =
             cl && cl->type == JANAS_JSON_ARRAY ? cl->child : NULL;
         c && n < cap; c = c->next) {
        const char *cov = janas_json_str(janas_json_get(c, "cover"));
        double base = obs_num(c, "base");
        if (o->n_layers < (int)(sizeof o->layer / sizeof *o->layer)) {
            snprintf(o->layer[o->n_layers].cover,
                     sizeof o->layer[o->n_layers].cover, "%s", cov ? cov : "");
            o->layer[o->n_layers++].base_ft = base;
        }
        n += (size_t)snprintf(out + n, cap - n, "%s%s", n ? ", " : "",
                              cover_words(cov));
        if (!isnan(base) && n < cap)
            n += (size_t)snprintf(out + n, cap - n, " at %.0f ft (%.0f m)",
                                  base, base * 0.3048);
    }
}

int wx_metar_parse(const char *text, size_t len, double lat, double lon,
                   double max_km, struct wx_obs *o, char *err, size_t err_len)
{
    struct janas_buf b = {.p = (char *)text, .n = len};
    if (len == 0) /* no station reported in the box */
        return 0;
    struct janas_json_doc *d =
        wx_parse(&b, "aviationweather.gov", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *r = janas_json_root(d);
    const struct janas_json *best = NULL;
    double best_km = max_km;
    for (const struct janas_json *m =
             r && r->type == JANAS_JSON_ARRAY ? r->child : NULL;
         m; m = m->next) {
        double la = obs_num(m, "lat"), lo = obs_num(m, "lon");
        if (isnan(la) || isnan(lo) || isnan(obs_num(m, "temp")))
            continue;
        double km = geo_km(lat, lon, la, lo);
        if (km <= best_km) {
            best = m;
            best_km = km;
        }
    }
    int found = 0;
    if (best) {
        memset(o, 0, sizeof *o);
        snprintf(o->icao, sizeof o->icao, "%s",
                 janas_json_str(janas_json_get(best, "icaoId"))
                     ? janas_json_str(janas_json_get(best, "icaoId"))
                     : "");
        snprintf(o->name, sizeof o->name, "%s",
                 janas_json_str(janas_json_get(best, "name"))
                     ? janas_json_str(janas_json_get(best, "name"))
                     : "");
        o->lat = obs_num(best, "lat");
        o->lon = obs_num(best, "lon");
        o->km = best_km;
        o->at = (time_t)obs_num(best, "obsTime");
        o->temp = obs_num(best, "temp");
        o->dew = obs_num(best, "dewp");
        const struct janas_json *wd = janas_json_get(best, "wdir");
        o->variable_wind = janas_json_is(wd, "VRB");
        o->wind_dir = obs_num(best, "wdir");
        o->wind_kt = obs_num(best, "wspd");
        o->gust_kt = obs_num(best, "wgst");
        o->pressure = obs_num(best, "altim");
        const struct janas_json *vis = janas_json_get(best, "visib");
        if (janas_json_str(vis))
            snprintf(o->visib, sizeof o->visib, "%s", janas_json_str(vis));
        else if (vis && vis->type == JANAS_JSON_NUMBER)
            snprintf(o->visib, sizeof o->visib, "%.2f",
                     fmin(fmax(janas_json_num(vis, 0), 0), 99));
        snprintf(o->wx, sizeof o->wx, "%s",
                 janas_json_str(janas_json_get(best, "wxString"))
                     ? janas_json_str(janas_json_get(best, "wxString"))
                     : "");
        clouds_of(janas_json_get(best, "clouds"), o->clouds, sizeof o->clouds,
                  o);
        snprintf(o->raw, sizeof o->raw, "%s",
                 janas_json_str(janas_json_get(best, "rawOb"))
                     ? janas_json_str(janas_json_get(best, "rawOb"))
                     : "");
        found = 1;
    }
    janas_json_free(d);
    return found;
}

int wx_metar_near(double lat, double lon, double max_km, struct wx_obs *o,
                  char *err, size_t err_len)
{
    double dlat = max_km / 111.0;
    double dlon = max_km / (111.0 * fmax(cos(lat * M_PI / 180), 0.05));
    char url[256];
    snprintf(url, sizeof url,
             "https://aviationweather.gov/api/data/metar?bbox=%.2f,%.2f,%.2f,"
             "%.2f&format=json",
             lat - dlat, lon - dlon, lat + dlat, lon + dlon);
    struct janas_buf b = {0};
    char why[256];
    int status = wx_fetch(url, METAR_KEEP_S, 0, &b, why, sizeof why);
    int r;
    if (status == 200 || status == 204)
        r = wx_metar_parse(b.p ? b.p : "", b.n, lat, lon, max_km, o, err,
                           err_len);
    else if (status < 0) {
        snprintf(err, err_len, "aviationweather.gov did not answer: %s", why);
        r = -1;
    } else {
        snprintf(err, err_len, "aviationweather.gov: HTTP %d", status);
        r = -1;
    }
    janas_buf_free(&b);
    return r;
}
