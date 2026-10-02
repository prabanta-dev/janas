/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * data.c - what janas-quakes read, as the data of its layouts (see
 * quakes.h, layouts.c and services/common/template.h). Every field is
 * written, even empty: a layout looks a name up in the item, then in what
 * holds it. The times are the computer's local ones, with how long ago;
 * the distance and the way are from the place asked about; where an event
 * was is told by the built-in tables (a sea, a region, the nearest city)
 * besides the source's own words.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "quakes.h"
#include "services/common/geo.h"

static void jstr(struct janas_buf *b, const char *key, const char *v)
{
    if (!v)
        v = "";
    janas_buf_printf(b, ", \"%s\": ", key);
    janas_json_write_str(b, v, strlen(v));
}

static const char *tf(int v)
{
    return v ? "true" : "false";
}

static void jwhen(struct janas_buf *b, long long at)
{
    time_t t = (time_t)at, now = time(NULL);
    struct tm lt, nt;
    localtime_r(&t, &lt);
    localtime_r(&now, &nt);
    long ago = now > t ? (long)(now - t) : 0;
    janas_buf_printf(b,
                     ", \"when\": {\"day\": %d, \"mon\": %d, \"at\": "
                     "\"%02d:%02d\", \"today\": %s}, \"ago\": {\"d\": %ld, "
                     "\"h\": %ld, \"m\": %ld}",
                     lt.tm_mday, lt.tm_mon + 1, lt.tm_hour, lt.tm_min,
                     tf(lt.tm_yday == nt.tm_yday && lt.tm_year == nt.tm_year),
                     ago / 86400, ago / 3600 % 24, ago / 60 % 60);
}

/* an event; from: the place its distance is told from, or NULL */
static void event(struct janas_buf *b, const struct qk_event *e,
                  const struct qk_place *from)
{
    janas_buf_puts(b, "{\"id\": ");
    janas_json_write_str(b, e->id, strlen(e->id));
    janas_buf_printf(b, ", \"src\": \"%s\", \"mag\": %.1f", e->src, e->mag);
    jstr(b, "magtype", e->magtype);
    janas_buf_printf(b, ", \"depth\": %.0f, \"deep\": %s", e->depth_km,
                     tf(e->depth_km >= 70));
    jwhen(b, e->at);
    if (from) {
        double km = geo_km(from->lat, from->lon, e->lat, e->lon);
        janas_buf_printf(
            b, ", \"km\": %.0f, \"dir\": \"%s\"", km,
            geo_compass(geo_bearing(from->lat, from->lon, e->lat, e->lon)));
    } else
        janas_buf_puts(b, ", \"km\": 0, \"dir\": \"\"");
    janas_buf_puts(b, ", \"where\": ");
    geo_describe_json(e->lat, e->lon, b);
    jstr(b, "place", e->place);
    jstr(b, "kind", strcmp(e->kind, "earthquake") ? e->kind : "");
    janas_buf_printf(b,
                     ", \"felt\": %d, \"has_felt\": %s, \"mmi\": %.0f, "
                     "\"has_mmi\": %s",
                     e->felt > 0 ? e->felt : 0, tf(e->felt > 0),
                     e->mmi > 0 ? floor(e->mmi + 0.5) : 0, tf(e->mmi > 0));
    jstr(b, "alert", e->alert);
    janas_buf_printf(b, ", \"tsunami\": %s, \"auto\": %s", tf(e->tsunami),
                     tf(!strcmp(e->src, "usgs") && !e->reviewed));
    jstr(b, "url", e->url);
    janas_buf_printf(b, ", \"lat\": %.3f, \"lon\": %.3f}", e->lat, e->lon);
}

static void list(struct janas_buf *b, const struct qk_list *l,
                 const struct qk_place *from)
{
    double top = -9;
    for (int i = 0; i < l->n; i++)
        if (l->e[i].mag > top)
            top = l->e[i].mag;
    janas_buf_printf(b, ", \"n\": %d, \"more\": %d, \"top\": %.1f", l->n,
                     l->more, top);
    janas_buf_puts(b, ", \"items\": [");
    for (int i = 0; i < l->n; i++) {
        janas_buf_puts(b, i ? ", " : "");
        event(b, &l->e[i], from);
    }
    janas_buf_puts(b, "]");
}

void qk_near_data(struct janas_buf *b, const struct qk_place *p,
                  double radius_km, int days, double min_mag,
                  const struct qk_list *l, const char *src)
{
    janas_buf_puts(b, "{\"name\": ");
    janas_json_write_str(b, p->name, strlen(p->name));
    jstr(b, "country", p->country);
    jstr(b, "how", p->how);
    janas_buf_printf(b,
                     ", \"sea\": %s, \"radius\": %.0f, \"days\": %d, "
                     "\"min_mag\": %.1f, \"source\": \"%s\"",
                     tf(p->is_sea), radius_km, days, min_mag, src);
    list(b, l, p);
    janas_buf_puts(b, "}");
}

void qk_strong_data(struct janas_buf *b, int days, double min_mag,
                    const struct qk_list *l)
{
    janas_buf_printf(b,
                     "{\"days\": %d, \"min_mag\": %.1f, \"source\": \"usgs\"",
                     days, min_mag);
    int alerts = 0, tsunamis = 0;
    for (int i = 0; i < l->n; i++) {
        alerts += l->e[i].alert[0] && strcmp(l->e[i].alert, "green");
        tsunamis += l->e[i].tsunami;
    }
    janas_buf_printf(b, ", \"alerts\": %d, \"tsunamis\": %d", alerts, tsunamis);
    list(b, l, NULL);
    janas_buf_puts(b, "}");
}

void qk_event_data(struct janas_buf *b, const struct qk_event *e,
                   const struct qk_place *from)
{
    janas_buf_puts(b, "{\"from\": ");
    janas_json_write_str(b, from ? from->name : "",
                         from ? strlen(from->name) : 0);
    janas_buf_printf(b, ", \"source\": \"%s\", \"e\": ", e->src);
    event(b, e, from);
    janas_buf_puts(b, "}");
}
