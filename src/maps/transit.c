/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * transit.c - maps_transit (see maps.h): journeys by public transport from
 * Transitous, a community's router (MOTIS) over the operators' open
 * timetables, free for open-source, non-commercial and light use. Times
 * are asked and told in the start's zone.
 */
#define _GNU_SOURCE /* timegm */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/geo.h"
#include "maps.h"

#define TRANSIT_KEEP_S 120

/* The zone of a point, from the tables: its city's, else its airport's. */
static const char *zone_of(double lat, double lon)
{
    double km;
    const struct geo_city *c = geo_city_near(lat, lon, &km);
    if (c)
        return geo_tz_name(c->tz);
    const struct geo_airport *a = geo_airport_near(lat, lon, &km);
    return a ? geo_tz_name(a->tz) : NULL;
}

/* "08:30", "2026-10-03 08:30" or "2026-10-03T08:30", local to tz: the
   time, or -1. A time of the day already gone is tomorrow's. */
static time_t local_time(const char *s, const char *tz)
{
    struct tm tm = {0};
    int y, mo, d, h, mi, whole = 0;
    if (sscanf(s, "%d-%d-%d%*[ T]%d:%d", &y, &mo, &d, &h, &mi) == 5)
        whole = 1;
    else if (sscanf(s, "%d:%d", &h, &mi) != 2)
        return -1;
    if (h < 0 || h > 23 || mi < 0 || mi > 59)
        return -1;
    char *old = getenv("TZ") ? strdup(getenv("TZ")) : NULL;
    if (tz && *tz)
        setenv("TZ", tz, 1);
    tzset();
    time_t now = time(NULL);
    localtime_r(&now, &tm);
    if (whole) {
        tm.tm_year = y - 1900;
        tm.tm_mon = mo - 1;
        tm.tm_mday = d;
    }
    tm.tm_hour = h;
    tm.tm_min = mi;
    tm.tm_sec = 0;
    tm.tm_isdst = -1;
    time_t t = mktime(&tm);
    if (!whole && t != -1 && t < now - 600) {
        tm.tm_mday++;
        tm.tm_isdst = -1;
        t = mktime(&tm);
    }
    if (tz && *tz) {
        if (old)
            setenv("TZ", old, 1);
        else
            unsetenv("TZ");
        tzset();
    }
    free(old);
    return t;
}

int mp_tool_transit(const struct janas_json *args, struct janas_buf *b)
{
    struct mp_place from, to;
    if (mp_arg_place(args, "from", 1, NAN, NAN, &from, b) != 0 ||
        mp_arg_place(args, "to", 0, from.lat, from.lon, &to, b) != 0)
        return 0;
    const char *tz = zone_of(from.lat, from.lon);
    const char *when = mp_arg_str(args, "when");
    time_t t = when ? local_time(when, tz) : -1;
    if (when && t == -1) {
        struct janas_buf out = {0};
        janas_buf_printf(&out,
                         "\"%s\" is not a time: give HH:MM, or YYYY-MM-DD "
                         "HH:MM.",
                         when);
        return mp_result(&out, b, 1);
    }
    struct janas_buf url = {0}, body = {0};
    janas_buf_printf(
        &url, "%s/api/v1/plan?fromPlace=%.6f,%.6f&toPlace=%.6f,%.6f",
        mp_base("JANAS_TRANSITOUS_URL", "https://api.transitous.org"), from.lat,
        from.lon, to.lat, to.lon);
    if (t != -1) {
        struct tm g;
        gmtime_r(&t, &g);
        janas_buf_printf(&url, "&time=%04d-%02d-%02dT%02d:%02d:00Z",
                         g.tm_year + 1900, g.tm_mon + 1, g.tm_mday, g.tm_hour,
                         g.tm_min);
    }
    if (mp_arg_bool(args, "arrive_by", 0))
        janas_buf_puts(&url, "&arriveBy=true");
    char err[384] = "";
    int status =
        url.oom ? -1 : mp_fetch(url.p, TRANSIT_KEEP_S, &body, err, sizeof err);
    janas_buf_free(&url);
    struct mp_transit tr;
    int ok = status == 200 &&
             mp_read_transit(body.p, body.n, &tr, err, sizeof err) == 0;
    if (status > 0 && status != 200)
        snprintf(err, sizeof err, "Transitous: HTTP %d", status);
    janas_buf_free(&body);
    if (!ok) {
        struct janas_buf out = {0};
        janas_buf_printf(&out, "No journeys from %s to %s: %s.", from.name,
                         to.name, err);
        return mp_result(&out, b, 1);
    }
    if (!tr.tz[0] && tz)
        snprintf(tr.tz, sizeof tr.tz, "%s", tz);
    struct janas_buf d = {0};
    mp_transit_data(&d, &from, &to, &tr);
    return mp_answer(b, "maps_transit", &d, MP_TRANSIT_LAYOUT,
                     MP_TRANSIT_BRIEF);
}
