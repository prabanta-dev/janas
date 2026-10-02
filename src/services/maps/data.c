/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * data.c - what janas-maps read, as the data of its layouts (see maps.h,
 * layouts.c and services/common/template.h): JSON objects whose fields the
 * layouts name. Distances and times are numbers, the words around them the
 * layout's, so that a client shows them in the user's language.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "services/common/geo.h"
#include "maps.h"

/* A field is written even when empty: a layout looks a name up in the
   item, then in what holds it (services/common/template.h), and a step without
   its own distance would show the whole route's. */
void mp_jstr(struct janas_buf *b, const char *key, const char *v)
{
    if (!v)
        v = "";
    janas_buf_printf(b, ", \"%s\": ", key);
    janas_json_write_str(b, v, strlen(v));
}

void mp_jnum(struct janas_buf *b, const char *key, double v, int decimals)
{
    if (!isnan(v))
        janas_buf_printf(b, ", \"%s\": %.*f", key, decimals, v);
}

void mp_jplace(struct janas_buf *b, const char *key, const struct mp_place *p)
{
    janas_buf_printf(b, ", \"%s\": {\"name\": ", key);
    janas_json_write_str(b, p->name, strlen(p->name));
    if (strcmp(p->address, p->name) != 0)
        mp_jstr(b, "address", p->address);
    mp_jstr(b, "city", p->city);
    mp_jnum(b, "lat", p->lat, 5);
    mp_jnum(b, "lon", p->lon, 5);
    mp_jstr(b, "how", p->how);
    janas_buf_puts(b, "}");
}

void mp_jtime(struct janas_buf *b, double min)
{
    long m = lround(min);
    if (m < 1)
        m = 1;
    janas_buf_printf(b, ", \"h\": %ld, \"min\": %ld", m / 60, m % 60);
}

/* a distance: "km" from a kilometre, "m" below it, to ten metres; the
   other one 0 */
static void jdist(struct janas_buf *b, double km)
{
    if (isnan(km) || km <= 0)
        janas_buf_puts(b, ", \"km\": 0, \"m\": 0");
    else if (km >= 10)
        janas_buf_printf(b, ", \"km\": %.0f, \"m\": 0", km);
    else if (km >= 1)
        janas_buf_printf(b, ", \"km\": %.1f, \"m\": 0", km);
    else
        janas_buf_printf(b, ", \"km\": 0, \"m\": %.0f",
                         fmax(10, round(km * 100) * 10));
}

void mp_jclock(struct janas_buf *b, const char *key, time_t t, const char *tz)
{
    char *old = getenv("TZ") ? strdup(getenv("TZ")) : NULL;
    if (tz && *tz)
        setenv("TZ", tz, 1);
    tzset();
    struct tm lt, nt;
    time_t now = time(NULL);
    localtime_r(&t, &lt);
    localtime_r(&now, &nt);
    janas_buf_printf(b, ", \"%s\": {\"at\": \"%02d:%02d\"", key, lt.tm_hour,
                     lt.tm_min);
    if (lt.tm_yday != nt.tm_yday || lt.tm_year != nt.tm_year)
        janas_buf_printf(b,
                         ", \"date\": true, \"wd\": %d, \"day\": %d, "
                         "\"mon\": %d",
                         lt.tm_wday, lt.tm_mday, lt.tm_mon + 1);
    janas_buf_puts(b, "}");
    if (tz && *tz) {
        if (old)
            setenv("TZ", old, 1);
        else
            unsetenv("TZ");
        tzset();
    }
    free(old);
}

/* An object whose members are written as ", key: value": open it, and
   close it with the first comma taken away. */
static size_t open_obj(struct janas_buf *b)
{
    janas_buf_puts(b, "{");
    return b->n;
}

static void close_obj(struct janas_buf *b, size_t at)
{
    if (!b->oom && b->n >= at + 2 && memcmp(b->p + at, ", ", 2) == 0) {
        memmove(b->p + at, b->p + at + 2, b->n - at - 2);
        b->n -= 2;
    }
    janas_buf_puts(b, "}");
}

/* ---- maps_route ---- */

static void jroute(struct janas_buf *b, const struct mp_route *r)
{
    jdist(b, r->km);
    mp_jtime(b, r->min);
    janas_buf_printf(b, ", \"toll\": %s, \"ferry\": %s",
                     r->toll ? "true" : "false", r->ferry ? "true" : "false");
    mp_jstr(b, "roads", r->roads);
}

void mp_route_data(struct janas_buf *d, const struct mp_place *from,
                   const struct mp_place *via, const struct mp_place *to,
                   const char *mode, const struct mp_routes *rs)
{
    const struct mp_route *r = &rs->r[0];
    janas_buf_printf(d, "{\"mode\": \"%s\", \"source\": \"%s\"", mode,
                     rs->source);
    mp_jplace(d, "from", from);
    if (via)
        mp_jplace(d, "via", via);
    mp_jplace(d, "to", to);
    jroute(d, r);
    if (rs->n > 1) {
        janas_buf_puts(d, ", \"alternatives\": [");
        for (int i = 1; i < rs->n; i++) {
            if (i > 1)
                janas_buf_puts(d, ", ");
            size_t o = open_obj(d);
            jroute(d, &rs->r[i]);
            close_obj(d, o);
        }
        janas_buf_puts(d, "]");
    }
    janas_buf_printf(d, ", \"n_steps\": %d", r->n_step);
    if (r->n_step) {
        janas_buf_puts(d, ", \"steps\": [");
        for (int i = 0; i < r->n_step; i++) {
            janas_buf_puts(d, i ? ", {\"text\": " : "{\"text\": ");
            janas_json_write_str(d, r->step[i].text, strlen(r->step[i].text));
            jdist(d, r->step[i].km);
            janas_buf_puts(d, "}");
        }
        janas_buf_puts(d, "]");
    }
    janas_buf_printf(d, ", \"cut\": %d", r->cut);
    /* the route on OpenStreetMap's site, drawn by the same engine */
    static const char *const engine[][2] = {
        {"car", "fossgis_valhalla_car"},
        {"bike", "fossgis_valhalla_bicycle"},
        {"foot", "fossgis_valhalla_foot"}};
    for (size_t i = 0; !via && i < sizeof engine / sizeof *engine; i++)
        if (strcmp(mode, engine[i][0]) == 0)
            janas_buf_printf(d,
                             ", \"link\": \"https://www.openstreetmap.org/"
                             "directions?engine=%s&route=%.5f%%2C%.5f%%3B%.5f"
                             "%%2C%.5f\"",
                             engine[i][1], from->lat, from->lon, to->lat,
                             to->lon);
    janas_buf_puts(d, "}");
}

/* ---- maps_transit ---- */

/* START and END are the journey's ends, as Transitous names them */
static const char *stop_name(const char *s, const struct mp_place *from,
                             const struct mp_place *to)
{
    if (strcmp(s, "START") == 0)
        return from->name;
    if (strcmp(s, "END") == 0)
        return to->name;
    return s;
}

void mp_transit_data(struct janas_buf *d, const struct mp_place *from,
                     const struct mp_place *to, const struct mp_transit *tr)
{
    janas_buf_puts(d, "{\"source\": \"Transitous\"");
    mp_jplace(d, "from", from);
    mp_jplace(d, "to", to);
    janas_buf_printf(d, ", \"count\": %d, \"trips\": [", tr->n);
    for (int i = 0; i < tr->n; i++) {
        const struct mp_trip *t = &tr->t[i];
        janas_buf_printf(d, "%s{\"transfers\": %d", i ? ", " : "",
                         t->transfers);
        mp_jclock(d, "dep", t->dep, tr->tz);
        mp_jclock(d, "arr", t->arr, tr->tz);
        mp_jtime(d, difftime(t->arr, t->dep) / 60);
        janas_buf_puts(d, ", \"legs\": [");
        for (int k = 0; k < t->n_leg; k++) {
            const struct mp_leg *g = &t->leg[k];
            janas_buf_puts(d, k ? ", {" : "{");
            int walk = strcmp(g->mode, "WALK") == 0;
            janas_buf_printf(d, "\"walk\": %s, \"mode\": \"%s\"",
                             walk ? "true" : "false", g->mode);
            mp_jtime(d, difftime(g->arr, g->dep) / 60);
            jdist(d, walk ? g->walk_m / 1000 : NAN);
            mp_jstr(d, "line", g->line);
            mp_jstr(d, "agency", g->agency);
            mp_jstr(d, "headsign", g->headsign);
            janas_buf_printf(d, ", \"live\": %s",
                             g->real_time ? "true" : "false");
            mp_jstr(d, "from", stop_name(g->from, from, to));
            mp_jclock(d, "dep", g->dep, tr->tz);
            mp_jstr(d, "to", stop_name(g->to, from, to));
            mp_jclock(d, "arr", g->arr, tr->tz);
            janas_buf_puts(d, "}");
        }
        janas_buf_puts(d, "]");
        janas_buf_printf(d, ", \"cut\": %d", t->cut);
        janas_buf_puts(d, "}");
    }
    janas_buf_puts(d, "]");
    mp_jstr(d, "tz", tr->tz);
    janas_buf_puts(d, "}");
}

/* ---- maps_nearby ---- */

void mp_near_data(struct janas_buf *d, const struct mp_place *at,
                  const char *what, double radius_km, const struct mp_near *nr)
{
    janas_buf_printf(d, "{\"what\": \"%s\"", what);
    mp_jplace(d, "place", at);
    janas_buf_printf(d, ", \"radius\": %g, \"found\": %d, \"count\": %d",
                     radius_km, nr->found, nr->n);
    janas_buf_puts(d, ", \"items\": [");
    for (int i = 0; i < nr->n; i++) {
        const struct mp_poi *p = &nr->p[i];
        if (i)
            janas_buf_puts(d, ", ");
        size_t o = open_obj(d);
        mp_jstr(d, "name", p->name);
        jdist(d, p->m / 1000);
        mp_jstr(d, "dir",
                p->m >= 10
                    ? geo_compass(geo_bearing(at->lat, at->lon, p->lat, p->lon))
                    : "");
        mp_jstr(d, "address", p->address);
        mp_jstr(d, "hours", p->hours);
        mp_jstr(d, "phone", p->phone);
        mp_jstr(d, "osm", p->osm);
        close_obj(d, o);
    }
    janas_buf_puts(d, "]}");
}

/* ---- maps_find ---- */

void mp_find_data(struct janas_buf *d, const struct mp_place *p)
{
    size_t o = open_obj(d);
    mp_jplace(d, "place", p);
    mp_jstr(d, "kind", p->kind);
    mp_jstr(d, "osm", p->osm);
    janas_buf_printf(d,
                     ", \"link\": \"https://www.openstreetmap.org/?mlat=%.5f&"
                     "mlon=%.5f#map=17/%.5f/%.5f\"",
                     p->lat, p->lon, p->lat, p->lon);
    close_obj(d, o);
}
