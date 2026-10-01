/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * place.c - a place by its name (see weather.h): Open-Meteo's geocoding,
 * from GeoNames, which knows the names in Italian and in English
 * ("Trapani", "Firenze", "Florence"), the municipality's region and
 * province, its zone and height; an airport by its code, or a point, from
 * the tables built into the program, which also stand in when the
 * geocoding does not answer.
 */
#define _GNU_SOURCE /* strcasestr */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "common/geo.h"
#include "weather.h"

#define DEG(x) ((x) * 1e-5)
#define PLACE_KEEP_S (7 * 24 * 3600) /* places do not move */

static void copy(char *to, size_t cap, const char *s)
{
    snprintf(to, cap, "%s", s ? s : "");
}

static void url_put(struct janas_buf *b, const char *s, size_t n)
{
    static const char hex[] = "0123456789ABCDEF";
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
            janas_buf_put(b, (const char *)&c, 1);
        else
            janas_buf_printf(b, "%%%c%c", hex[c >> 4], hex[c & 15]);
    }
}

/*
 * The names of the tables are in English (Sicily, Turin); MeteoAlarm's
 * regions and the Civil Protection's municipalities are in the country's
 * language (Sicilia, Torino). The geocoding's place called as the city of
 * the tables, when it lies within 30 km of the point, gives them: its
 * region and province, and with rename its name too.
 */
static int find_named(const char *query, double near_lat, double near_lon,
                      struct wx_place *p, char *err, size_t err_len);

static void localize(struct wx_place *p, const char *city, int rename)
{
    struct wx_place q;
    char err[256];
    if (!city || !*city ||
        find_named(city, p->lat, p->lon, &q, err, sizeof err) != 0 ||
        geo_km(p->lat, p->lon, q.lat, q.lon) > 30 ||
        (p->cc[0] && strcasecmp(p->cc, q.cc) != 0))
        return;
    if (q.region[0])
        copy(p->region, sizeof p->region, q.region);
    if (q.area2[0])
        copy(p->area2, sizeof p->area2, q.area2);
    if (q.country[0])
        copy(p->country, sizeof p->country, q.country);
    if (rename)
        copy(p->name, sizeof p->name, q.name);
}

void wx_place_at(double lat, double lon, struct wx_place *p)
{
    memset(p, 0, sizeof *p);
    p->lat = lat;
    p->lon = lon;
    p->elevation = NAN;
    double km;
    const struct geo_city *c = geo_city_near(lat, lon, &km);
    if (geo_sea_at(lat, lon) || !c) {
        /* "the Tyrrhenian Sea, between ...; 39 km N of Trapani" */
        struct janas_buf b = {0};
        geo_describe(lat, lon, &b);
        const char *s = b.p ? b.p : "";
        copy(p->name, sizeof p->name, strncmp(s, "over ", 5) == 0 ? s + 5 : s);
        janas_buf_free(&b);
        const struct geo_airport *a = geo_airport_near(lat, lon, &km);
        if (a)
            copy(p->tz, sizeof p->tz, geo_tz_name(a->tz));
        return;
    }
    if (km < 3)
        copy(p->name, sizeof p->name, c->name);
    else
        snprintf(p->name, sizeof p->name, "%.0f km %s of %s", km,
                 geo_compass(geo_bearing(DEG(c->lat), DEG(c->lon), lat, lon)),
                 c->name);
    if (c->region != 0xFFFF)
        copy(p->region, sizeof p->region, geo_region_name(c->region));
    copy(p->country, sizeof p->country, geo_country_name(c->country));
    copy(p->cc, sizeof p->cc, geo_country_iso2(c->country));
    copy(p->tz, sizeof p->tz, geo_tz_name(c->tz));
    localize(p, c->name, km < 3);
}

static int from_tables(const char *q, struct wx_place *p)
{
    size_t n = strlen(q);
    int code = (n == 3 || n == 4);
    for (size_t i = 0; code && i < n; i++)
        code = isupper((unsigned char)q[i]) != 0;
    const struct geo_airport *a = code ? geo_airport_find(q) : NULL;
    if (a) {
        memset(p, 0, sizeof *p);
        snprintf(p->name, sizeof p->name, "%s (%s)", a->name,
                 a->iata[0] ? a->iata : a->icao);
        p->lat = DEG(a->lat);
        p->lon = DEG(a->lon);
        p->elevation = NAN;
        copy(p->tz, sizeof p->tz, geo_tz_name(a->tz));
        copy(p->country, sizeof p->country, geo_country_name(a->country));
        copy(p->cc, sizeof p->cc, geo_country_iso2(a->country));
        /* the region, for the warnings: the nearest city's */
        double km;
        const struct geo_city *c = geo_city_near(p->lat, p->lon, &km);
        if (c && c->country == a->country && c->region != 0xFFFF)
            copy(p->region, sizeof p->region, geo_region_name(c->region));
        if (c && c->country == a->country)
            localize(p, c->name, 0);
        return 1;
    }
    const struct geo_city *c = geo_city_find(q);
    if (!c)
        return 0;
    memset(p, 0, sizeof *p);
    copy(p->name, sizeof p->name, c->name);
    p->lat = DEG(c->lat);
    p->lon = DEG(c->lon);
    p->elevation = NAN;
    copy(p->tz, sizeof p->tz, geo_tz_name(c->tz));
    copy(p->country, sizeof p->country, geo_country_name(c->country));
    copy(p->cc, sizeof p->cc, geo_country_iso2(c->country));
    if (c->region != 0xFFFF)
        copy(p->region, sizeof p->region, geo_region_name(c->region));
    return 1;
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
    if (end2 == end || *end2 || fabs(a) > 90 || fabs(b) > 180)
        return 0;
    *lat = a;
    *lon = b;
    return 1;
}

static int same_name(const char *a, const char *b)
{
    return a && b && strcasecmp(a, b) == 0;
}

static int contains(const char *hay, const char *needle)
{
    return hay && needle && *needle && strcasestr(hay, needle) != NULL;
}

/* GeoNames' feature code as a weight: a capital, a region's, a
   province's or county's, a municipality's seat, a place. */
static double rank_weight(const char *code)
{
    static const struct {
        const char *code;
        double w;
    } w[] = {{"PPLC", 20}, {"PPLA", 10}, {"PPLA2", 3}, {"PPLA3", 2}};
    for (size_t i = 0; code && i < sizeof w / sizeof *w; i++)
        if (strcmp(code, w[i].code) == 0)
            return w[i].w;
    return 1;
}

/* The geocoding's place of that name: the nearest to (near_lat, near_lon)
   when given (the name of a point's city: Turin, not Turin in Georgia),
   else the most people, weighed by the place's rank: Trapani the town,
   not a hamlet called so; "Milan" Milano, not Milan in Tennessee;
   "Venice" Venezia (51,000 people in its old town, a regional capital),
   not Dayton in Ohio, which the geocoding gives first. */
static int find_named(const char *query, double near_lat, double near_lon,
                      struct wx_place *p, char *err, size_t err_len)
{
    /* "Trapani, Sicilia": the name, and what its region or country must
       hold */
    char name[128], hint[64] = "";
    copy(name, sizeof name, query);
    char *comma = strchr(name, ',');
    if (comma) {
        *comma = 0;
        const char *h = comma + 1;
        while (*h == ' ')
            h++;
        copy(hint, sizeof hint, h);
    }
    size_t nn = strlen(name);
    while (nn && name[nn - 1] == ' ')
        name[--nn] = 0;
    if (!nn) {
        snprintf(err, err_len, "no place given");
        return -1;
    }
    struct janas_buf url = {0}, body = {0};
    janas_buf_puts(&url,
                   "https://geocoding-api.open-meteo.com/v1/search?name=");
    url_put(&url, name, nn);
    janas_buf_puts(&url, "&count=10&language=it&format=json");
    char why[256] = "";
    int status =
        url.oom ? -1 : wx_fetch(url.p, PLACE_KEEP_S, 0, &body, why, sizeof why);
    janas_buf_free(&url);
    struct janas_json_doc *d =
        status == 200 ? wx_parse(&body, "Open-Meteo", why, sizeof why) : NULL;
    const struct janas_json *res =
        d ? janas_json_get(janas_json_root(d), "results") : NULL;
    const struct janas_json *best = NULL;
    double best_score = -1, best_km = 1e30;
    int near = !isnan(near_lat) && !isnan(near_lon);
    for (const struct janas_json *r =
             res && res->type == JANAS_JSON_ARRAY ? res->child : NULL;
         r; r = r->next) {
        const char *a1 = janas_json_str(janas_json_get(r, "admin1"));
        const char *a2 = janas_json_str(janas_json_get(r, "admin2"));
        const char *co = janas_json_str(janas_json_get(r, "country"));
        const char *cc = janas_json_str(janas_json_get(r, "country_code"));
        if (hint[0] && !contains(a1, hint) && !contains(a2, hint) &&
            !contains(co, hint) && !same_name(cc, hint))
            continue;
        if (near) {
            double km =
                geo_km(near_lat, near_lon,
                       janas_json_num(janas_json_get(r, "latitude"), NAN),
                       janas_json_num(janas_json_get(r, "longitude"), NAN));
            if (km < best_km) { /* NAN never is */
                best = r;
                best_km = km;
            }
            continue;
        }
        double pop = janas_json_num(janas_json_get(r, "population"), 0);
        /* + 1: places of no known population still count */
        double score =
            (pop + 1) *
            rank_weight(janas_json_str(janas_json_get(r, "feature_code")));
        if (!best || score > best_score) {
            best = r;
            best_score = score;
        }
    }
    int ok = 0;
    if (best) {
        memset(p, 0, sizeof *p);
        copy(p->name, sizeof p->name,
             janas_json_str(janas_json_get(best, "name")));
        copy(p->region, sizeof p->region,
             janas_json_str(janas_json_get(best, "admin1")));
        copy(p->area2, sizeof p->area2,
             janas_json_str(janas_json_get(best, "admin2")));
        copy(p->country, sizeof p->country,
             janas_json_str(janas_json_get(best, "country")));
        copy(p->cc, sizeof p->cc,
             janas_json_str(janas_json_get(best, "country_code")));
        copy(p->tz, sizeof p->tz,
             janas_json_str(janas_json_get(best, "timezone")));
        p->lat = janas_json_num(janas_json_get(best, "latitude"), 0);
        p->lon = janas_json_num(janas_json_get(best, "longitude"), 0);
        p->elevation = janas_json_num(janas_json_get(best, "elevation"), NAN);
        ok = 1;
    }
    janas_json_free(d);
    janas_buf_free(&body);
    if (ok)
        return 0;
    /* the geocoding did not answer, or knew nothing: the tables */
    if (from_tables(name, p))
        return 0;
    if (why[0])
        snprintf(err, err_len, "no place called \"%s\" found (%s)", query, why);
    else
        snprintf(err, err_len, "no place called \"%s\" found", query);
    return -1;
}

int wx_place_find(const char *query, struct wx_place *p, char *err,
                  size_t err_len)
{
    while (*query == ' ')
        query++;
    double lat, lon;
    if (as_point(query, &lat, &lon)) {
        wx_place_at(lat, lon, p);
        return 0;
    }
    /* an airport's code, as such (TRN: Turin's, not Trn in Bosnia) */
    size_t n = strlen(query);
    int code = n == 3 || n == 4;
    for (size_t i = 0; code && i < n; i++)
        code = isupper((unsigned char)query[i]) != 0;
    const struct geo_airport *a = code ? geo_airport_find(query) : NULL;
    if (a && (strcmp(a->iata, query) == 0 || strcmp(a->icao, query) == 0) &&
        from_tables(query, p))
        return 0;
    return find_named(query, NAN, NAN, p, err, err_len);
}
