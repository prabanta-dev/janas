/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * place.c - a place or an address by its name (see maps.h): Nominatim,
 * OpenStreetMap's own search, which knows streets, buildings and stations
 * as well as towns; Photon when it does not answer; a point named from the
 * tables built into the program.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "services/common/geo.h"
#include "maps.h"

#define DEG(x) ((x) * 1e-5)
#define PLACE_KEEP_S (7 * 24 * 3600) /* places do not move */

void mp_place_at(double lat, double lon, struct mp_place *p)
{
    memset(p, 0, sizeof *p);
    p->lat = lat;
    p->lon = lon;
    double km;
    const struct geo_city *c = geo_city_near(lat, lon, &km);
    if (!c)
        snprintf(p->name, sizeof p->name, "%.5f, %.5f", lat, lon);
    else if (km < 3)
        snprintf(p->name, sizeof p->name, "%s", c->name);
    else
        snprintf(p->name, sizeof p->name, "%.0f km %s of %s", km,
                 geo_compass(geo_bearing(DEG(c->lat), DEG(c->lon), lat, lon)),
                 c->name);
    if (c)
        snprintf(p->city, sizeof p->city, "%s", c->name);
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

static int nominatim(const char *q, double near_lat, double near_lon,
                     struct mp_place *p, char *why, size_t why_len)
{
    struct janas_buf url = {0}, body = {0};
    janas_buf_printf(
        &url, "%s/search?q=",
        mp_base("JANAS_NOMINATIM_URL", "https://nominatim.openstreetmap.org"));
    mp_url_put(&url, q);
    janas_buf_printf(&url,
                     "&format=jsonv2&limit=5&addressdetails=1&"
                     "accept-language=%s",
                     mp_lang());
    /* a place near the other end preferred, not required: "Stazione
       Centrale" on the way from Trapani is Palermo's */
    if (!isnan(near_lat) && !isnan(near_lon))
        janas_buf_printf(&url, "&viewbox=%.4f,%.4f,%.4f,%.4f&bounded=0",
                         near_lon - 1, near_lat + 1, near_lon + 1,
                         near_lat - 1);
    int status =
        url.oom ? -1 : mp_fetch(url.p, PLACE_KEEP_S, &body, why, why_len);
    janas_buf_free(&url);
    int rc = status == 200 ? mp_read_nominatim(body.p, body.n, p) : -1;
    if (status > 0 && status != 200)
        snprintf(why, why_len, "Nominatim: HTTP %d", status);
    janas_buf_free(&body);
    return rc;
}

static int photon(const char *q, double near_lat, double near_lon,
                  struct mp_place *p, char *why, size_t why_len)
{
    struct janas_buf url = {0}, body = {0};
    janas_buf_printf(&url, "%s/api/?q=",
                     mp_base("JANAS_PHOTON_URL", "https://photon.komoot.io"));
    mp_url_put(&url, q);
    janas_buf_puts(&url, "&limit=1");
    if (!isnan(near_lat) && !isnan(near_lon))
        janas_buf_printf(&url, "&lat=%.4f&lon=%.4f", near_lat, near_lon);
    /* its languages are few: else the names of the place's own */
    const char *l = mp_lang();
    if (strcmp(l, "de") == 0 || strcmp(l, "en") == 0 || strcmp(l, "fr") == 0)
        janas_buf_printf(&url, "&lang=%s", l);
    int status =
        url.oom ? -1 : mp_fetch(url.p, PLACE_KEEP_S, &body, why, why_len);
    janas_buf_free(&url);
    int rc = status == 200 ? mp_read_photon(body.p, body.n, p) : -1;
    if (status > 0 && status != 200)
        snprintf(why, why_len, "Photon: HTTP %d", status);
    janas_buf_free(&body);
    return rc;
}

int mp_place_reverse(double lat, double lon, struct mp_place *p)
{
    struct janas_buf url = {0}, body = {0};
    janas_buf_printf(
        &url,
        "%s/reverse?lat=%.6f&lon=%.6f&format=jsonv2&"
        "addressdetails=1&accept-language=%s",
        mp_base("JANAS_NOMINATIM_URL", "https://nominatim.openstreetmap.org"),
        lat, lon, mp_lang());
    char why[256];
    int status =
        url.oom ? -1 : mp_fetch(url.p, PLACE_KEEP_S, &body, why, sizeof why);
    janas_buf_free(&url);
    struct mp_place q;
    int rc = status == 200 ? mp_read_nominatim(body.p, body.n, &q) : -1;
    janas_buf_free(&body);
    if (rc == 0) { /* the point asked, named as what is there */
        q.lat = lat;
        q.lon = lon;
        *p = q;
    }
    return rc;
}

int mp_place_find(const char *query, double near_lat, double near_lon,
                  struct mp_place *p, char *err, size_t err_len)
{
    while (*query == ' ')
        query++;
    if (!*query) {
        snprintf(err, err_len, "no place given");
        return -1;
    }
    double lat, lon;
    if (as_point(query, &lat, &lon)) {
        mp_place_at(lat, lon, p);
        return 0;
    }
    char w1[256] = "", w2[256] = "";
    if (nominatim(query, near_lat, near_lon, p, w1, sizeof w1) == 0)
        return 0;
    /* Nominatim knew nothing: Photon will not know more */
    if (!w1[0]) {
        snprintf(err, err_len, "no place called \"%s\" found", query);
        return -1;
    }
    if (photon(query, near_lat, near_lon, p, w2, sizeof w2) == 0)
        return 0;
    snprintf(err, err_len, "no place called \"%s\" found (%s%s%s)", query, w1,
             w2[0] ? "; " : "", w2);
    return -1;
}
