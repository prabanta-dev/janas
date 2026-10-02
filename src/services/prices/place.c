/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * place.c - a place and its country (see prices.h), for the prices that
 * depend on where one is: fuel, electricity. A name through Nominatim
 * (OpenStreetMap: names in every language, at most a request a second),
 * the built-in tables when it does not answer, "lat,lon" through the
 * tables, where the user is through the locator. Spain's fuel goes by
 * province: a postcode's first two digits.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "services/common/geo.h"
#include "services/common/locate.h"
#include "prices.h"

#define GEO_KEEP_S (7 * 86400)

static void copy(char *to, size_t cap, const char *s)
{
    snprintf(to, cap, "%s", s ? s : "");
}

static void from_tables(double lat, double lon, struct pr_place *p)
{
    double km;
    const struct geo_city *c = geo_city_near(lat, lon, &km);
    if (!c)
        return;
    if (!p->name[0])
        copy(p->name, sizeof p->name, c->name);
    if (!p->cc[0])
        copy(p->cc, sizeof p->cc, geo_country_iso2(c->country));
    if (!p->country[0])
        copy(p->country, sizeof p->country, geo_country_name(c->country));
    if (!p->region[0] && c->region != 0xFFFF)
        copy(p->region, sizeof p->region, geo_region_name(c->region));
    if (!p->tz[0])
        copy(p->tz, sizeof p->tz, geo_tz_name(c->tz));
}

/* The computer's language, for the names Nominatim gives ("Italia") */
static const char *lang(void)
{
    static char l[8];
    const char *v = getenv("LC_ALL");
    if (!v || !*v)
        v = getenv("LC_MESSAGES");
    if (!v || !*v)
        v = getenv("LANG");
    if (!v || strlen(v) < 2 || !strncmp(v, "C", 1) || !strncmp(v, "POSIX", 5))
        return "en";
    snprintf(l, sizeof l, "%.2s", v);
    return l;
}

/* A place of Nominatim's answer: an array (search) or an object
   (reverse). 0 when there is one. */
static int nominatim_read(const struct janas_buf *body, int search,
                          struct pr_place *p)
{
    char why[128];
    struct janas_json_doc *d =
        janas_json_parse(body->p, body->n, why, sizeof why);
    const struct janas_json *x = d ? janas_json_root(d) : NULL;
    if (x && search)
        x = x->type == JANAS_JSON_ARRAY ? x->child : NULL;
    const struct janas_json *a = janas_json_get(x, "address");
    int ok = 0;
    if (a) {
        if (search) {
            const char *nm = janas_json_str(janas_json_get(x, "name"));
            copy(p->name, sizeof p->name, nm);
            const char *la = janas_json_str(janas_json_get(x, "lat"));
            const char *lo = janas_json_str(janas_json_get(x, "lon"));
            p->lat = la ? strtod(la, NULL) : NAN;
            p->lon = lo ? strtod(lo, NULL) : NAN;
            const char *cc = janas_json_str(janas_json_get(a, "country_code"));
            for (int i = 0; cc && i < 2 && cc[i]; i++)
                p->cc[i] = (char)(cc[i] & ~0x20); /* "it" -> "IT" */
            copy(p->country, sizeof p->country,
                 janas_json_str(janas_json_get(a, "country")));
            copy(p->region, sizeof p->region,
                 janas_json_str(janas_json_get(a, "state")));
        }
        const char *pc = janas_json_str(janas_json_get(a, "postcode"));
        if (pc)
            copy(p->postcode, sizeof p->postcode, pc);
        ok = !isnan(p->lat) && !isnan(p->lon);
    }
    janas_json_free(d);
    return ok ? 0 : -1;
}

/* Nominatim's place of name (any language: "Wien", "Milano"), and in
   Spain its postcode (its province) from the point when the place has
   none. 0 when found. */
static int geocode(const char *name, struct pr_place *p, char *err,
                   size_t err_len)
{
    struct janas_buf url = {0}, body = {0};
    janas_buf_puts(&url, "https://nominatim.openstreetmap.org/search?q=");
    pr_url_put(&url, name);
    janas_buf_printf(&url,
                     "&format=jsonv2&limit=1&addressdetails=1"
                     "&accept-language=%s",
                     lang());
    int st = url.oom
                 ? -1
                 : pr_fetch(url.p, GEO_KEEP_S, 15000, 1.1, &body, err, err_len);
    janas_buf_free(&url);
    int ok = st == 200 && nominatim_read(&body, 1, p) == 0;
    if (st == 200 && !ok)
        snprintf(err, err_len, "no place named %s", name);
    else if (st > 0 && st != 200)
        snprintf(err, err_len, "Nominatim: HTTP %d", st);
    if (ok && !strcmp(p->cc, "ES") && !p->postcode[0]) {
        char e[200];
        janas_buf_printf(&url,
                         "https://nominatim.openstreetmap.org/reverse?lat=%.6f"
                         "&lon=%.6f&format=jsonv2&zoom=18&addressdetails=1",
                         p->lat, p->lon);
        if (!url.oom &&
            pr_fetch(url.p, GEO_KEEP_S, 15000, 1.1, &body, e, sizeof e) == 200)
            nominatim_read(&body, 0, p);
        janas_buf_free(&url);
    }
    janas_buf_free(&body);
    return ok ? 0 : -1;
}

int pr_place_find(const char *q, struct pr_place *p, char *err, size_t err_len)
{
    memset(p, 0, sizeof *p);
    p->lat = p->lon = NAN;
    if (!q) {
        struct janas_where w;
        if (janas_where(&w, err, err_len) != 0)
            return -1;
        p->lat = w.lat;
        p->lon = w.lon;
        copy(p->name, sizeof p->name, w.city);
        copy(p->cc, sizeof p->cc, w.cc);
        copy(p->country, sizeof p->country, w.country);
        copy(p->region, sizeof p->region, w.region);
        copy(p->how, sizeof p->how, w.how);
        copy(p->tz, sizeof p->tz, w.tz);
        from_tables(p->lat, p->lon, p);
        if (p->name[0] && !strcmp(p->cc, "ES")) { /* its province */
            struct pr_place g;
            char e[200];
            memset(&g, 0, sizeof g);
            if (geocode(p->name, &g, e, sizeof e) == 0)
                copy(p->postcode, sizeof p->postcode, g.postcode);
        }
        return 0;
    }
    double lat, lon;
    char tail;
    if (sscanf(q, "%lf , %lf %c", &lat, &lon, &tail) == 2 && fabs(lat) <= 90 &&
        fabs(lon) <= 180) {
        p->lat = lat;
        p->lon = lon;
        from_tables(lat, lon, p);
        return 0;
    }
    if (geocode(q, p, err, err_len) == 0) {
        from_tables(p->lat, p->lon, p); /* its time zone */
        return 0;
    }
    const struct geo_city *c = geo_city_find(q); /* the tables, offline */
    if (!c)
        return -1;
    p->lat = c->lat / 1e5;
    p->lon = c->lon / 1e5;
    copy(p->name, sizeof p->name, c->name);
    from_tables(p->lat, p->lon, p);
    return 0;
}
