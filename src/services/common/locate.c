/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * locate.c - where the user is (locate.h): JANAS_LOCATION, the internet
 * connection's position, the machine's time zone.
 */
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "services/common/geo.h"
#include "services/common/https_get.h"
#include "services/common/locate.h"

#define DEG(x) ((x) * 1e-5)
#define KEEP_S 3600 /* a connection seldom moves within the hour */

static void copy(char *dst, size_t cap, const char *src)
{
    snprintf(dst, cap, "%s", src ? src : "");
}

/* the region, country and zone of the nearest city of the tables, for
   what a source left out */
static void fill_from_tables(struct janas_where *w)
{
    double km;
    const struct geo_city *c = geo_city_near(w->lat, w->lon, &km);
    if (!c)
        return;
    if (!w->city[0] && km < 3)
        copy(w->city, sizeof w->city, c->name);
    if (!w->region[0] && c->region != 0xFFFF)
        copy(w->region, sizeof w->region, geo_region_name(c->region));
    if (!w->country[0])
        copy(w->country, sizeof w->country, geo_country_name(c->country));
    if (!w->cc[0])
        copy(w->cc, sizeof w->cc, geo_country_iso2(c->country));
    if (!w->tz[0])
        copy(w->tz, sizeof w->tz, geo_tz_name(c->tz));
}

static void from_city(const struct geo_city *c, struct janas_where *w)
{
    w->lat = DEG(c->lat);
    w->lon = DEG(c->lon);
    copy(w->city, sizeof w->city, c->name);
    fill_from_tables(w);
}

/* "38.02, 12.51" */
static int as_point(const char *q, double *lat, double *lon)
{
    char *end;
    double a = strtod(q, &end);
    if (end == q)
        return 0;
    while (*end == ' ' || *end == ',' || *end == ';')
        end++;
    char *end2;
    double b = strtod(end, &end2);
    while (*end2 == ' ')
        end2++;
    if (end2 == end || *end2 || fabs(a) > 90 || fabs(b) > 180)
        return 0;
    *lat = a;
    *lon = b;
    return 1;
}

/* a number given as a number or as a string (GeoJS: "38.1302") */
static double num(const struct janas_json *v)
{
    const char *s = janas_json_str(v);
    if (s) {
        char *end;
        double d = strtod(s, &end);
        return end != s ? d : NAN;
    }
    return janas_json_num(v, NAN);
}

static int point_ok(const struct janas_where *w)
{
    return !isnan(w->lat) && !isnan(w->lon) && fabs(w->lat) <= 90 &&
           fabs(w->lon) <= 180 && (w->lat != 0 || w->lon != 0);
}

int janas_where_geojs(const char *text, size_t n, struct janas_where *w)
{
    struct janas_json_doc *d = janas_json_parse(text, n, NULL, 0);
    const struct janas_json *r = d ? janas_json_root(d) : NULL;
    memset(w, 0, sizeof *w);
    w->lat = num(janas_json_get(r, "latitude"));
    w->lon = num(janas_json_get(r, "longitude"));
    copy(w->city, sizeof w->city, janas_json_str(janas_json_get(r, "city")));
    copy(w->region, sizeof w->region,
         janas_json_str(janas_json_get(r, "region")));
    copy(w->country, sizeof w->country,
         janas_json_str(janas_json_get(r, "country")));
    copy(w->cc, sizeof w->cc,
         janas_json_str(janas_json_get(r, "country_code")));
    copy(w->tz, sizeof w->tz, janas_json_str(janas_json_get(r, "timezone")));
    int ok = r && point_ok(w);
    janas_json_free(d);
    return ok ? 0 : -1;
}

int janas_where_ipwhois(const char *text, size_t n, struct janas_where *w)
{
    struct janas_json_doc *d = janas_json_parse(text, n, NULL, 0);
    const struct janas_json *r = d ? janas_json_root(d) : NULL;
    memset(w, 0, sizeof *w);
    const struct janas_json *ok = janas_json_get(r, "success");
    w->lat = num(janas_json_get(r, "latitude"));
    w->lon = num(janas_json_get(r, "longitude"));
    copy(w->city, sizeof w->city, janas_json_str(janas_json_get(r, "city")));
    copy(w->region, sizeof w->region,
         janas_json_str(janas_json_get(r, "region")));
    copy(w->country, sizeof w->country,
         janas_json_str(janas_json_get(r, "country")));
    copy(w->cc, sizeof w->cc,
         janas_json_str(janas_json_get(r, "country_code")));
    copy(w->tz, sizeof w->tz,
         janas_json_str(janas_json_get(janas_json_get(r, "timezone"), "id")));
    int good = r && ok && ok->type == JANAS_JSON_TRUE && point_ok(w);
    janas_json_free(d);
    return good ? 0 : -1;
}

static int from_net(const char *url, const char *who,
                    int (*parse)(const char *, size_t, struct janas_where *),
                    struct janas_where *w)
{
    struct janas_buf b = {0};
    char why[256];
    int status = janas_https_get(url, NULL, 0, &b, why, sizeof why);
    int r = status == 200 && b.p ? parse(b.p, b.n, w) : -1;
    janas_buf_free(&b);
    if (r == 0) {
        snprintf(w->how, sizeof w->how,
                 "estimated from the internet connection (%s)", who);
        fill_from_tables(w);
    }
    return r;
}

/* the city of the machine's zone: TZ, or where /etc/localtime points */
static int from_zone(struct janas_where *w)
{
    char path[256] = "";
    const char *tz = getenv("TZ");
    if (tz && *tz) {
        copy(path, sizeof path, tz[0] == ':' ? tz + 1 : tz);
    } else {
        ssize_t n = readlink("/etc/localtime", path, sizeof path - 1);
        if (n <= 0)
            return -1;
        path[n] = 0;
    }
    const char *z = strstr(path, "zoneinfo/");
    z = z ? z + 9 : path; /* "Europe/Rome" */
    const char *slash = strrchr(z, '/');
    if (!slash) /* "UTC", "CET": no city */
        return -1;
    char city[64];
    copy(city, sizeof city, slash + 1);
    for (char *c = city; *c; c++)
        if (*c == '_')
            *c = ' '; /* "New_York" */
    const struct geo_city *c = geo_city_find(city);
    if (!c)
        return -1;
    memset(w, 0, sizeof *w);
    copy(w->tz, sizeof w->tz, z);
    from_city(c, w);
    copy(w->how, sizeof w->how,
         "a rough guess, the city of the computer's time zone");
    return 0;
}

static pthread_mutex_t kept_mu = PTHREAD_MUTEX_INITIALIZER;
static struct janas_where kept;
static time_t kept_at;

int janas_where(struct janas_where *w, char *err, size_t err_len)
{
    const char *set = getenv("JANAS_LOCATION");
    memset(w, 0, sizeof *w);
    if (set && *set) {
        if (strcmp(set, "off") == 0) {
            snprintf(err, err_len,
                     "where the user is is not known (JANAS_LOCATION=off): "
                     "ask them which place");
            return -1;
        }
        const struct geo_city *c = NULL;
        if (as_point(set, &w->lat, &w->lon))
            fill_from_tables(w);
        else if ((c = geo_city_find(set)) != NULL)
            from_city(c, w);
        else {
            snprintf(err, err_len,
                     "JANAS_LOCATION is \"%.60s\": neither a city of the "
                     "tables (its English name) nor \"lat,lon\"",
                     set);
            return -1;
        }
        copy(w->how, sizeof w->how, "as set on the computer (JANAS_LOCATION)");
        return 0;
    }
    pthread_mutex_lock(&kept_mu);
    int fresh = kept_at && time(NULL) - kept_at < KEEP_S;
    if (fresh)
        *w = kept;
    pthread_mutex_unlock(&kept_mu);
    if (fresh)
        return 0;
    if (from_net("https://get.geojs.io/v1/ip/geo.json", "GeoJS",
                 janas_where_geojs, w) != 0 &&
        from_net("https://ipwho.is/", "ipwho.is", janas_where_ipwhois, w) !=
            0) {
        /* not kept: the connection is asked again next time */
        if (from_zone(w) == 0)
            return 0;
        memset(w, 0, sizeof *w);
        snprintf(err, err_len,
                 "where the user is could not be told (no answer from GeoJS "
                 "or ipwho.is, and no city for the computer's time zone): "
                 "ask them which place");
        return -1;
    }
    pthread_mutex_lock(&kept_mu);
    kept = *w;
    kept_at = time(NULL);
    pthread_mutex_unlock(&kept_mu);
    return 0;
}
