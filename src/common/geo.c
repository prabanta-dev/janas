/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * geo.c - the geography of Janas's services (see geo.h): the generated tables,
 * and a position put into words.
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#include "geo.h"
#include "geo_data.h"

#define N_OF(a) (sizeof(a) / sizeof(*(a)))
#define DEG(x) ((x) * 1e-5)
#define RAD(x) ((x) * M_PI / 180)

double geo_km(double lat1, double lon1, double lat2, double lon2)
{
    double p1 = RAD(lat1), p2 = RAD(lat2), dp = RAD(lat2 - lat1),
           dl = RAD(lon2 - lon1);
    double h = sin(dp / 2) * sin(dp / 2) +
               cos(p1) * cos(p2) * sin(dl / 2) * sin(dl / 2);
    return 2 * 6371.0 * asin(sqrt(h < 1 ? h : 1));
}

double geo_bearing(double lat1, double lon1, double lat2, double lon2)
{
    double p1 = RAD(lat1), p2 = RAD(lat2), dl = RAD(lon2 - lon1);
    double y = sin(dl) * cos(p2),
           x = cos(p1) * sin(p2) - sin(p1) * cos(p2) * cos(dl);
    double b = atan2(y, x) * 180 / M_PI;
    return b < 0 ? b + 360 : b;
}

const char *geo_compass(double deg)
{
    static const char *const names[16] = {"N",  "NNE", "NE", "ENE", "E",  "ESE",
                                          "SE", "SSE", "S",  "SSW", "SW", "WSW",
                                          "W",  "WNW", "NW", "NNW"};
    int i = (int)floor(fmod(deg + 11.25, 360) / 22.5);
    return names[(i % 16 + 16) % 16];
}

const char *geo_country_name(uint16_t i)
{
    return i < N_OF(geo_countries) ? geo_countries[i].name : "";
}

const char *geo_country_iso2(uint16_t i)
{
    return i < N_OF(geo_countries) ? geo_countries[i].iso2 : "";
}

const char *geo_region_name(uint16_t i)
{
    return i < N_OF(geo_regions) ? geo_regions[i].name : "";
}

const char *geo_tz_name(uint16_t i)
{
    return i < N_OF(geo_tzs) ? geo_tzs[i] : "UTC";
}

/* A name as it is compared: lower case, dashes of every kind (and the
   en and em dash in UTF-8) and runs of spaces as one space, a province in
   brackets ("Trapani (TP)") left out. */
static void norm(const char *s, char *out, size_t cap)
{
    size_t n = 0;
    int space = 1; /* no space at the start */
    while (*s && n + 1 < cap) {
        unsigned char c = (unsigned char)*s;
        int dash =
            c == '-' || c == '_' || c == '/' ||
            (c == 0xE2 && (unsigned char)s[1] == 0x80 &&
             ((unsigned char)s[2] == 0x93 || (unsigned char)s[2] == 0x94));
        if (c == '(') {
            const char *e = strchr(s, ')');
            s = e ? e + 1 : s + strlen(s);
            continue;
        }
        if (dash || c == ' ' || c == '\t') {
            if (!space)
                out[n++] = ' ';
            space = 1;
            s += c == 0xE2 ? 3 : 1;
            continue;
        }
        out[n++] = (char)(c < 0x80 ? tolower(c) : c);
        space = 0;
        s++;
    }
    while (n && out[n - 1] == ' ')
        n--;
    out[n] = 0;
}

const struct geo_airport *geo_airport_find(const char *what)
{
    size_t n = strlen(what);
    for (size_t i = 0; i < N_OF(geo_airports); i++) {
        const struct geo_airport *a = &geo_airports[i];
        if ((n == 3 && strcasecmp(a->iata, what) == 0) ||
            (n == 4 && a->icao[0] && strcasecmp(a->icao, what) == 0))
            return a;
    }
    char w[128], t[256];
    norm(what, w, sizeof w);
    if (!w[0])
        return NULL;
    /* by city, then by name: of the airports of a city, the one whose name
       starts with the city's wins */
    const struct geo_airport *by_city = NULL;
    for (size_t i = 0; i < N_OF(geo_airports); i++) {
        const struct geo_airport *a = &geo_airports[i];
        norm(a->city, t, sizeof t);
        if (strcmp(t, w) == 0) {
            norm(a->name, t, sizeof t);
            if (!by_city || strncmp(t, w, strlen(w)) == 0)
                by_city = a;
        }
    }
    if (by_city)
        return by_city;
    /* a part of the name: "Birgi", "Trapani Birgi", "Fiumicino" */
    if (strlen(w) >= 4)
        for (size_t i = 0; i < N_OF(geo_airports); i++) {
            norm(geo_airports[i].name, t, sizeof t);
            if (strstr(t, w))
                return &geo_airports[i];
        }
    /* a city's name: its nearest airport, within 50 km ("Turin" is Caselle
       Torinese's airport for OurAirports) */
    const struct geo_city *c = geo_city_find(what);
    if (c) {
        double km;
        const struct geo_airport *a =
            geo_airport_near(DEG(c->lat), DEG(c->lon), &km);
        if (a && km < 50)
            return a;
    }
    return NULL;
}

const struct geo_city *geo_city_find(const char *name)
{
    const struct geo_city *best = NULL;
    for (size_t i = 0; i < N_OF(geo_cities); i++) {
        const struct geo_city *c = &geo_cities[i];
        if ((strcasecmp(c->ascii, name) == 0 ||
             strcasecmp(c->name, name) == 0) &&
            (!best || c->pop > best->pop))
            best = c;
    }
    return best;
}

const struct geo_airport *geo_airport_near(double lat, double lon, double *km)
{
    const struct geo_airport *best = NULL;
    double most = 1e30;
    for (size_t i = 0; i < N_OF(geo_airports); i++) {
        const struct geo_airport *a = &geo_airports[i];
        if (fabs(DEG(a->lat) - lat) > 3)
            continue; /* far for sure: about 330 km of latitude */
        double d = geo_km(lat, lon, DEG(a->lat), DEG(a->lon));
        if (d < most) {
            most = d;
            best = a;
        }
    }
    if (!best) /* nothing within three degrees of latitude: all of them */
        for (size_t i = 0; i < N_OF(geo_airports); i++) {
            const struct geo_airport *a = &geo_airports[i];
            double d = geo_km(lat, lon, DEG(a->lat), DEG(a->lon));
            if (d < most) {
                most = d;
                best = a;
            }
        }
    if (km)
        *km = most;
    return best;
}

/* Even-odd over all the sea's rings: outer boundaries and holes alike. */
static int in_sea(const struct geo_sea *s, int32_t lat, int32_t lon)
{
    if (lat < s->lat_min || lat > s->lat_max || lon < s->lon_min ||
        lon > s->lon_max)
        return 0;
    int in = 0;
    for (uint32_t r = s->ring0; r < s->ring0 + s->n_rings; r++) {
        const int32_t(*p)[2] = geo_points + geo_rings[r].pt0;
        uint32_t n = geo_rings[r].n;
        for (uint32_t i = 0, j = n - 1; i < n; j = i++) {
            int64_t xi = p[i][0], yi = p[i][1], xj = p[j][0], yj = p[j][1];
            if ((yi > lat) != (yj > lat) &&
                (double)lon <
                    (double)(xj - xi) * (lat - yi) / (double)(yj - yi) + xi)
                in = !in;
        }
    }
    return in;
}

const char *geo_sea_at(double lat, double lon)
{
    int32_t la = (int32_t)lround(lat * 1e5), lo = (int32_t)lround(lon * 1e5);
    /* the smallest one that holds it: a gulf before its sea */
    const struct geo_sea *best = NULL;
    double area = 1e30;
    for (size_t i = 0; i < N_OF(geo_seas); i++) {
        const struct geo_sea *s = &geo_seas[i];
        if (!in_sea(s, la, lo))
            continue;
        double a =
            (double)(s->lat_max - s->lat_min) * (s->lon_max - s->lon_min);
        if (a < area) {
            area = a;
            best = s;
        }
    }
    return best ? best->name : NULL;
}

static const struct {
    const char *it, *en;
} SEA_IT[] = {
    {"tirreno", "tyrrhenian sea"},
    {"adriatico", "adriatic sea"},
    {"ionio", "ionian sea"},
    {"ligure", "ligurian sea"},
    {"mediterraneo", "mediterranean sea"},
    {"egeo", "aegean sea"},
    {"nero", "black sea"},
    {"baltico", "baltic sea"},
    {"rosso", "red sea"},
    {"del nord", "north sea"},
    {"delle baleari", "balearic sea"},
    {"di alboran", "alboran sea"},
    {"dei caraibi", "caribbean sea"},
    {"caspio", "caspian sea"},
};

/* Whether the sea s is a part of the area: smaller, and at least half of
   its box within the area's (the Adriatic within the Mediterranean, whose
   own polygon stops lower; not the Black Sea). */
static int area_part(const struct geo_area *a, const struct geo_sea *s)
{
    if (strcmp(s->name, a->name) == 0)
        return 1;
    double h = (double)s->lat_max - s->lat_min,
           w = (double)s->lon_max - s->lon_min;
    double ah = (double)a->lat_max - a->lat_min,
           aw = (double)a->lon_max - a->lon_min;
    if (h * w >= ah * aw || h <= 0 || w <= 0)
        return 0;
    double ih = fmin(s->lat_max, a->lat_max) - fmax(s->lat_min, a->lat_min),
           iw = fmin(s->lon_max, a->lon_max) - fmax(s->lon_min, a->lon_min);
    return ih > 0 && iw > 0 && ih * iw >= 0.5 * h * w;
}

int geo_area_find(const char *name, struct geo_area *a)
{
    char w[128], t[128];
    norm(name, w, sizeof w);
    const char *q = w;
    if (strncmp(q, "the ", 4) == 0)
        q += 4;
    /* "mar Tirreno", "mare Adriatico", "Tirreno" */
    if (strncmp(q, "mare ", 5) == 0)
        q += 5;
    else if (strncmp(q, "mar ", 4) == 0)
        q += 4;
    for (size_t i = 0; i < N_OF(SEA_IT); i++)
        if (strcmp(q, SEA_IT[i].it) == 0)
            q = SEA_IT[i].en;
    memset(a, 0, sizeof(*a));
    for (size_t i = 0; i < N_OF(geo_seas); i++) {
        const struct geo_sea *s = &geo_seas[i];
        norm(s->name, t, sizeof t);
        size_t n = strlen(t), m = strlen(q);
        int same =
            strcmp(t, q) == 0 || (n > 4 && strcmp(t + n - 4, " sea") == 0 &&
                                  m == n - 4 && strncmp(t, q, m) == 0);
        if (!same)
            continue;
        if (!a->name) {
            a->name = s->name;
            a->lat_min = s->lat_min;
            a->lat_max = s->lat_max;
            a->lon_min = s->lon_min;
            a->lon_max = s->lon_max;
        } else if (strcmp(a->name, s->name) == 0) {
            if (s->lat_min < a->lat_min)
                a->lat_min = s->lat_min;
            if (s->lat_max > a->lat_max)
                a->lat_max = s->lat_max;
            if (s->lon_min < a->lon_min)
                a->lon_min = s->lon_min;
            if (s->lon_max > a->lon_max)
                a->lon_max = s->lon_max;
        }
    }
    if (!a->name)
        return 0;
    a->s_lat_min = a->lat_min;
    a->s_lat_max = a->lat_max;
    a->s_lon_min = a->lon_min;
    a->s_lon_max = a->lon_max;
    for (size_t i = 0; i < N_OF(geo_seas); i++) {
        const struct geo_sea *s = &geo_seas[i];
        if (!area_part(a, s))
            continue;
        if (s->lat_min < a->s_lat_min)
            a->s_lat_min = s->lat_min;
        if (s->lat_max > a->s_lat_max)
            a->s_lat_max = s->lat_max;
        if (s->lon_min < a->s_lon_min)
            a->s_lon_min = s->lon_min;
        if (s->lon_max > a->s_lon_max)
            a->s_lon_max = s->lon_max;
    }
    return 1;
}

int geo_area_has(const struct geo_area *a, double lat, double lon)
{
    int32_t la = (int32_t)lround(lat * 1e5), lo = (int32_t)lround(lon * 1e5);
    for (size_t i = 0; i < N_OF(geo_seas); i++) {
        const struct geo_sea *s = &geo_seas[i];
        if (area_part(a, s) && in_sea(s, la, lo))
            return 1;
    }
    return 0;
}

/* A city's region and country, as "Tuscany, Italy" (or the country). */
static void where_of(const struct geo_city *c, struct janas_buf *b)
{
    if (c->region != 0xFFFF && *geo_region_name(c->region))
        janas_buf_printf(b, "%s, %s", geo_region_name(c->region),
                         geo_country_name(c->country));
    else
        janas_buf_puts(b, geo_country_name(c->country));
}

const struct geo_city *geo_city_near(double lat, double lon, double *km)
{
    const struct geo_city *near = NULL;
    double d_near = 1e30;
    for (size_t i = 0; i < N_OF(geo_cities); i++) {
        const struct geo_city *c = &geo_cities[i];
        if (fabs(DEG(c->lat) - lat) > 5)
            continue;
        double d = geo_km(lat, lon, DEG(c->lat), DEG(c->lon));
        /* the smallest capitals (Vatican City) only right over them: as
           the landmark of Rome they would mislead */
        if (c->pop < 10000 && d > 3)
            continue;
        if (d < d_near) {
            d_near = d;
            near = c;
        }
    }
    *km = d_near;
    return near;
}

void geo_describe(double lat, double lon, struct janas_buf *b)
{
    /* the nearest city, and the nearest of another region that lies the
       other way: the two the position is between */
    const struct geo_city *other = NULL;
    double d_near, d_other = 1e30;
    const struct geo_city *near = geo_city_near(lat, lon, &d_near);
    if (!near) {
        janas_buf_printf(b, "at %.2f, %.2f", lat, lon);
        return;
    }
    double b_near = geo_bearing(lat, lon, DEG(near->lat), DEG(near->lon));
    for (size_t i = 0; i < N_OF(geo_cities); i++) {
        const struct geo_city *c = &geo_cities[i];
        if (fabs(DEG(c->lat) - lat) > 5 || c->pop < 10000 ||
            (c->region == near->region && c->country == near->country))
            continue;
        double d = geo_km(lat, lon, DEG(c->lat), DEG(c->lon));
        double turn =
            fabs(geo_bearing(lat, lon, DEG(c->lat), DEG(c->lon)) - b_near);
        if (turn > 180)
            turn = 360 - turn;
        if (turn > 100 && d < d_other) {
            d_other = d;
            other = c;
        }
    }
    const char *sea = geo_sea_at(lat, lon);
    if (sea) {
        janas_buf_printf(b, "over the %s", sea);
        if (other && d_other < 3 * d_near + 100 && d_other < 400) {
            janas_buf_puts(b, ", between ");
            where_of(near, b);
            janas_buf_puts(b, " and ");
            where_of(other, b);
        }
        janas_buf_puts(b, "; ");
    } else {
        janas_buf_puts(b, "over ");
        where_of(near, b);
        janas_buf_puts(b, "; ");
    }
    if (d_near < 3)
        janas_buf_printf(b, "over %s", near->name);
    else
        janas_buf_printf(
            b, "%.0f km %s of %s", d_near,
            geo_compass(geo_bearing(DEG(near->lat), DEG(near->lon), lat, lon)),
            near->name);
}
