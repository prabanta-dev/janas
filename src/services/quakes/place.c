/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * place.c - the place a question means (see quakes.h): where the user is
 * through the locator, "lat,lon", a sea of the built-in tables ("mar
 * Tirreno": the middle of it), a name through Nominatim (OpenStreetMap,
 * names in every language, at most a request a second), the built-in
 * cities when it does not answer.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quakes.h"
#include "services/common/geo.h"
#include "services/common/locate.h"

static void copy(char *to, size_t cap, const char *s)
{
    size_t n = s ? strlen(s) : 0;
    if (n >= cap)
        n = cap - 1;
    memcpy(to, s ? s : "", n);
    to[n] = 0;
}

static void from_tables(struct qk_place *p)
{
    double km;
    const struct geo_city *c = geo_city_near(p->lat, p->lon, &km);
    if (!c)
        return;
    if (!p->name[0])
        copy(p->name, sizeof p->name, c->name);
    if (!p->cc[0])
        copy(p->cc, sizeof p->cc, geo_country_iso2(c->country));
    if (!p->country[0])
        copy(p->country, sizeof p->country, geo_country_name(c->country));
}

int qk_in_ingv(double lat, double lon)
{
    return lat >= QK_INGV_LAT0 && lat <= QK_INGV_LAT1 && lon >= QK_INGV_LON0 &&
           lon <= QK_INGV_LON1;
}

/* the computer's language, for the names Nominatim gives */
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

static int nominatim(const char *name, struct qk_place *p, char *err,
                     size_t err_len)
{
    struct janas_buf url = {0}, body = {0};
    janas_buf_puts(&url, "https://nominatim.openstreetmap.org/search?q=");
    qk_url_put(&url, name);
    janas_buf_printf(&url,
                     "&format=jsonv2&limit=1&addressdetails=1"
                     "&accept-language=%s",
                     lang());
    int st = url.oom ? -1 : qk_fetch(url.p, &body, err, err_len);
    janas_buf_free(&url);
    int ok = 0;
    if (st == 200) {
        char why[128];
        struct janas_json_doc *d =
            janas_json_parse(body.p, body.n, why, sizeof why);
        const struct janas_json *x = d ? janas_json_root(d) : NULL;
        x = x && x->type == JANAS_JSON_ARRAY ? x->child : NULL;
        const char *la = janas_json_str(janas_json_get(x, "lat"));
        const char *lo = janas_json_str(janas_json_get(x, "lon"));
        if (la && lo) {
            p->lat = strtod(la, NULL);
            p->lon = strtod(lo, NULL);
            copy(p->name, sizeof p->name,
                 janas_json_str(janas_json_get(x, "name")));
            const struct janas_json *a = janas_json_get(x, "address");
            const char *cc = janas_json_str(janas_json_get(a, "country_code"));
            for (int i = 0; cc && i < 2 && cc[i]; i++)
                p->cc[i] = (char)(cc[i] & ~0x20); /* "it" -> "IT" */
            copy(p->country, sizeof p->country,
                 janas_json_str(janas_json_get(a, "country")));
            ok = 1;
        } else
            snprintf(err, err_len, "no place named %s", name);
        janas_json_free(d);
    } else if (st > 0)
        snprintf(err, err_len, "Nominatim: HTTP %d", st);
    janas_buf_free(&body);
    return ok ? 0 : -1;
}

int qk_place_find(const char *q, struct qk_place *p, char *err, size_t err_len)
{
    memset(p, 0, sizeof *p);
    if (!q) {
        struct janas_where w;
        if (janas_where(&w, err, err_len) != 0)
            return -1;
        p->lat = w.lat;
        p->lon = w.lon;
        copy(p->name, sizeof p->name, w.city);
        copy(p->cc, sizeof p->cc, w.cc);
        copy(p->country, sizeof p->country, w.country);
        copy(p->how, sizeof p->how, w.how);
        from_tables(p);
        return 0;
    }
    double lat, lon;
    char tail;
    if (sscanf(q, "%lf , %lf %c", &lat, &lon, &tail) == 2 && fabs(lat) <= 90 &&
        fabs(lon) <= 180) {
        p->lat = lat;
        p->lon = lon;
        from_tables(p);
        return 0;
    }
    struct geo_area a;
    if (geo_area_find(q, &a)) { /* a sea: what is within it */
        p->is_sea = 1;
        p->sea = a;
        p->lat = (a.lat_min + a.lat_max) / 2e5;
        p->lon = (a.lon_min + a.lon_max) / 2e5;
        copy(p->name, sizeof p->name, q); /* as the user wrote it */
        return 0;
    }
    if (nominatim(q, p, err, err_len) == 0) {
        /* the name as asked: Nominatim's is the local one (八日市場 for
           Yokaichiba) */
        copy(p->name, sizeof p->name, q);
        from_tables(p);
        return 0;
    }
    const struct geo_city *c = geo_city_find(q); /* the tables, offline */
    if (!c)
        return -1;
    p->lat = c->lat / 1e5;
    p->lon = c->lon / 1e5;
    copy(p->name, sizeof p->name, c->name);
    from_tables(p);
    return 0;
}
