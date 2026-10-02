/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * read_fuel.c - the fuel prices of the countries that publish them as open
 * data (see prices.h), each source as it gives them: Italy's ministry
 * (MIMIT) a registry of the stations and their prices, two files
 * '|'-separated, France's instant flow, Spain's stations of a province,
 * Austria's E-Control the stations near a point. Then the same list: the
 * stations within the radius selling the fuel, the cheapest first. No
 * network here, so that the tests can read answers kept as they came.
 */
#define _GNU_SOURCE /* strcasecmp, timegm */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "common/geo.h"
#include "prices.h"

#define ALL_MAX 4096 /* stations within a radius, at most */

static void fcopy(char *to, size_t cap, const char *s, size_t n)
{
    snprintf(to, cap, "%.*s", (int)(n < cap ? n : cap - 1), s ? s : "");
}

static const char *fstr(const struct janas_json *o, const char *key)
{
    const char *s = janas_json_str(janas_json_get(o, key));
    return s && *s ? s : NULL;
}

/* "1,559" or "1.559" or a number */
static double num_any(const struct janas_json *o, const char *key)
{
    const struct janas_json *v = janas_json_get(o, key);
    if (v && v->type == JANAS_JSON_STRING) {
        char b[32];
        snprintf(b, sizeof b, "%s", v->s);
        for (char *c = b; *c; c++)
            if (*c == ',')
                *c = '.';
        char *end;
        double d = strtod(b, &end);
        return end > b ? d : NAN;
    }
    return janas_json_num(v, NAN);
}

static int cmp_station(const void *a, const void *b)
{
    const struct pr_station *x = a, *y = b;
    if (x->price != y->price)
        return x->price < y->price ? -1 : 1;
    return x->km < y->km ? -1 : x->km > y->km;
}

/* "29/09/2026 08:29:00" or "2026-10-02 09:21": when; -1 when neither */
static time_t told(const char *s)
{
    struct tm tm = {0};
    if (sscanf(s, "%d/%d/%d %d:%d", &tm.tm_mday, &tm.tm_mon, &tm.tm_year,
               &tm.tm_hour, &tm.tm_min) != 5 &&
        sscanf(s, "%d-%d-%d %d:%d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
               &tm.tm_hour, &tm.tm_min) != 5)
        return -1;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return timegm(&tm);
}

void pr_fuel_rank(struct pr_fuel *f, struct pr_station *all, int n, time_t now)
{
    int k = 0;
    f->stale = 0;
    for (int i = 0; i < n; i++) {
        time_t t = told(all[i].updated);
        if (t > 0 && now - t > PR_STALE_DAYS * 86400)
            f->stale++;
        else
            all[k++] = all[i];
    }
    n = k;
    qsort(all, (size_t)n, sizeof *all, cmp_station);
    f->found = n;
    f->cheapest = n ? all[0].price : NAN;
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += all[i].price;
    f->average = n ? sum / n : NAN;
    f->n = n < PR_STATIONS ? n : PR_STATIONS;
    memcpy(f->s, all, (size_t)f->n * sizeof *all);
}

/* ---- Italy ---- */

/* The fields of a line '|'-separated: up to max, pointers and lengths */
static int fields(const char *p, const char *e, const char **f, size_t *fl,
                  int max)
{
    int k = 0;
    while (k < max) {
        const char *bar = memchr(p, '|', (size_t)(e - p));
        f[k] = p;
        fl[k] = (size_t)((bar ? bar : e) - p);
        k++;
        if (!bar)
            break;
        p = bar + 1;
    }
    return k;
}

struct near_it {
    long id;
    struct pr_station s;
};

static int cmp_id(const void *a, const void *b)
{
    const struct near_it *x = a, *y = b;
    return x->id < y->id ? -1 : x->id > y->id;
}

static void trim(char *s)
{
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\r'))
        s[--n] = 0;
}

int pr_read_fuel_it(const char *registry, size_t rn, const char *prices,
                    size_t pn, const struct pr_fuel_ask *a, struct pr_fuel *f,
                    char *err, size_t err_len)
{
    static const char *const names[] = {"Benzina", "Gasolio", "GPL", "Metano"};
    memset(f, 0, sizeof *f);
    snprintf(f->fuel, sizeof f->fuel, "%s", names[a->kind]);
    struct near_it *near = malloc(ALL_MAX * sizeof *near);
    if (!near) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    int nn = 0;
    /* idImpianto|Gestore|Bandiera|Tipo Impianto|Nome Impianto|Indirizzo|
       Comune|Provincia|Latitudine|Longitudine, after two lines */
    const char *p = registry, *e = registry + rn;
    for (int line = 0; p < e; line++) {
        const char *nl = memchr(p, '\n', (size_t)(e - p));
        const char *le = nl ? nl : e;
        const char *fv[10];
        size_t fl[10];
        if (line >= 2 && fields(p, le, fv, fl, 10) == 10 && nn < ALL_MAX) {
            double lat = strtod(fv[8], NULL), lon = strtod(fv[9], NULL);
            double km = geo_km(a->lat, a->lon, lat, lon);
            if (lat != 0 && km <= a->radius_km) {
                struct near_it *x = &near[nn++];
                memset(x, 0, sizeof *x);
                x->id = strtol(fv[0], NULL, 10);
                /* the brand, or the operator of an unbranded one */
                int white = fl[2] == 13 && !strncmp(fv[2], "Pompe Bianche", 13);
                fcopy(x->s.name, sizeof x->s.name, white ? fv[1] : fv[2],
                      white ? fl[1] : fl[2]);
                fcopy(x->s.address, sizeof x->s.address, fv[5], fl[5]);
                fcopy(x->s.city, sizeof x->s.city, fv[6], fl[6]);
                trim(x->s.name);
                trim(x->s.address);
                trim(x->s.city);
                x->s.lat = lat;
                x->s.lon = lon;
                x->s.km = km;
                x->s.price = NAN;
                x->s.self = -1;
            }
        }
        p = nl ? nl + 1 : e;
    }
    qsort(near, (size_t)nn, sizeof *near, cmp_id);
    /* idImpianto|descCarburante|prezzo|isSelf|dtComu: the self-service
       price where there is one, else the served one */
    size_t want = strlen(names[a->kind]);
    p = prices;
    e = prices + pn;
    for (int line = 0; p < e; line++) {
        const char *nl = memchr(p, '\n', (size_t)(e - p));
        const char *le = nl ? nl : e;
        const char *fv[5];
        size_t fl[5];
        if (line >= 2 && fields(p, le, fv, fl, 5) == 5 && fl[1] == want &&
            !strncasecmp(fv[1], names[a->kind], want)) {
            struct near_it key = {.id = strtol(fv[0], NULL, 10)};
            struct near_it *x =
                bsearch(&key, near, (size_t)nn, sizeof *near, cmp_id);
            int self = fv[3][0] == '1';
            if (x && (x->s.self != 1 || self)) {
                x->s.price = strtod(fv[2], NULL);
                x->s.self = self;
                fcopy(x->s.updated, sizeof x->s.updated, fv[4], fl[4]);
                trim(x->s.updated);
            }
        }
        p = nl ? nl + 1 : e;
    }
    struct pr_station *all = malloc((size_t)(nn ? nn : 1) * sizeof *all);
    int k = 0;
    for (int i = 0; all && i < nn; i++)
        if (!isnan(near[i].s.price) && near[i].s.price > 0)
            all[k++] = near[i].s;
    free(near);
    if (!all) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    pr_fuel_rank(f, all, k, a->now);
    free(all);
    return 0;
}

/* ---- France ---- */

int pr_read_fuel_fr(const char *t, size_t n, const struct pr_fuel_ask *a,
                    struct pr_fuel *f, char *err, size_t err_len)
{
    static const char *const field[][2] = {{"e10_prix", "sp95_prix"},
                                           {"gazole_prix", NULL},
                                           {"gplc_prix", NULL},
                                           {NULL, NULL}};
    static const char *const names[] = {"SP95-E10", "Gazole", "GPLc", "GNV"};
    memset(f, 0, sizeof *f);
    snprintf(f->fuel, sizeof f->fuel, "%s", names[a->kind]);
    if (!field[a->kind][0]) {
        snprintf(err, err_len, "France's open data has no price of %s",
                 names[a->kind]);
        return -1;
    }
    char why[128];
    struct janas_json_doc *d = janas_json_parse(t, n, why, sizeof why);
    if (!d) {
        snprintf(err, err_len,
                 "France's fuel prices: an answer it could "
                 "not read (%s)",
                 why);
        return -1;
    }
    const struct janas_json *o = janas_json_root(d);
    const struct janas_json *r = janas_json_get(o, "results");
    struct pr_station all[100];
    int k = 0;
    for (const struct janas_json *x = r ? r->child : NULL; x && k < 100;
         x = x->next) {
        const char *fld = field[a->kind][0];
        double price = num_any(x, fld);
        if (isnan(price) && field[a->kind][1]) {
            fld = field[a->kind][1];
            price = num_any(x, fld);
        }
        const struct janas_json *g = janas_json_get(x, "geom");
        double lat = num_any(g, "lat"), lon = num_any(g, "lon");
        if (isnan(price) || isnan(lat))
            continue;
        struct pr_station *s = &all[k++];
        memset(s, 0, sizeof *s);
        const char *ad = fstr(x, "adresse"), *ci = fstr(x, "ville");
        fcopy(s->address, sizeof s->address, ad, ad ? strlen(ad) : 0);
        fcopy(s->city, sizeof s->city, ci, ci ? strlen(ci) : 0);
        s->lat = lat;
        s->lon = lon;
        s->km = geo_km(a->lat, a->lon, lat, lon);
        s->price = price;
        s->self = -1;
        char maj[32];
        snprintf(maj, sizeof maj, "%.*s_maj", (int)(strlen(fld) - 5), fld);
        const char *m = fstr(x, maj);
        fcopy(s->updated, sizeof s->updated, m, m ? 16 : 0); /* to minutes */
        if (s->updated[10] == 'T')
            s->updated[10] = ' ';
    }
    double total = num_any(o, "total_count");
    janas_json_free(d);
    pr_fuel_rank(f, all, k, a->now);
    /* the stations asked for were the cheapest of them all: the average
       of those is not the area's */
    if (!isnan(total) && total > k) {
        f->found = (int)total;
        f->average = NAN;
    }
    return 0;
}

/* ---- Spain ---- */

int pr_read_fuel_es(const char *t, size_t n, const struct pr_fuel_ask *a,
                    struct pr_fuel *f, char *err, size_t err_len)
{
    static const char *const field[] = {
        "Precio Gasolina 95 E5", "Precio Gasoleo A",
        "Precio Gases licuados del petróleo", "Precio Gas Natural Comprimido"};
    static const char *const names[] = {"Gasolina 95 E5", "Gasóleo A", "GLP",
                                        "GNC"};
    memset(f, 0, sizeof *f);
    snprintf(f->fuel, sizeof f->fuel, "%s", names[a->kind]);
    char why[128];
    struct janas_json_doc *d = janas_json_parse(t, n, why, sizeof why);
    if (!d) {
        snprintf(err, err_len,
                 "Spain's fuel prices: an answer it could not "
                 "read (%s)",
                 why);
        return -1;
    }
    const struct janas_json *o = janas_json_root(d);
    const char *when = fstr(o, "Fecha"); /* "02/10/2026 16:48:00" */
    char upd[24] = "";
    int dd, mm, yy, hh, mi;
    if (when && sscanf(when, "%d/%d/%d %d:%d", &dd, &mm, &yy, &hh, &mi) == 5)
        snprintf(upd, sizeof upd, "%04d-%02d-%02d %02d:%02d", yy, mm, dd, hh,
                 mi);
    const struct janas_json *r = janas_json_get(o, "ListaEESSPrecio");
    struct pr_station *all = malloc(ALL_MAX * sizeof *all);
    int k = 0;
    for (const struct janas_json *x = r ? r->child : NULL; all && x;
         x = x->next) {
        double price = num_any(x, field[a->kind]);
        double lat = num_any(x, "Latitud"),
               lon = num_any(x, "Longitud (WGS84)");
        if (isnan(price) || isnan(lat) || isnan(lon) || k >= ALL_MAX)
            continue;
        double km = geo_km(a->lat, a->lon, lat, lon);
        if (km > a->radius_km)
            continue;
        struct pr_station *s = &all[k++];
        memset(s, 0, sizeof *s);
        const char *nm = fstr(x, "Rótulo"), *ad = fstr(x, "Dirección"),
                   *ci = fstr(x, "Localidad");
        fcopy(s->name, sizeof s->name, nm, nm ? strlen(nm) : 0);
        fcopy(s->address, sizeof s->address, ad, ad ? strlen(ad) : 0);
        fcopy(s->city, sizeof s->city, ci, ci ? strlen(ci) : 0);
        s->lat = lat;
        s->lon = lon;
        s->km = km;
        s->price = price;
        s->self = -1;
        snprintf(s->updated, sizeof s->updated, "%s", upd);
    }
    janas_json_free(d);
    if (!all) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    pr_fuel_rank(f, all, k, a->now);
    free(all);
    return 0;
}

/* ---- Austria ---- */

int pr_read_fuel_at(const char *t, size_t n, const struct pr_fuel_ask *a,
                    struct pr_fuel *f, char *err, size_t err_len)
{
    static const char *const code[] = {"SUP", "DIE", NULL, "GAS"};
    static const char *const names[] = {"Super 95", "Diesel", "LPG", "CNG"};
    memset(f, 0, sizeof *f);
    snprintf(f->fuel, sizeof f->fuel, "%s", names[a->kind]);
    if (!code[a->kind]) {
        snprintf(err, err_len, "E-Control has no price of %s", names[a->kind]);
        return -1;
    }
    char why[128];
    struct janas_json_doc *d = janas_json_parse(t, n, why, sizeof why);
    if (!d) {
        snprintf(err, err_len, "E-Control: an answer it could not read (%s)",
                 why);
        return -1;
    }
    struct pr_station all[64];
    int k = 0;
    const struct janas_json *r = janas_json_root(d);
    for (const struct janas_json *x =
             r && r->type == JANAS_JSON_ARRAY ? r->child : NULL;
         x && k < 64; x = x->next) {
        const struct janas_json *ps = janas_json_get(x, "prices");
        double price = NAN;
        for (const struct janas_json *y = ps ? ps->child : NULL; y; y = y->next)
            if (janas_json_is(janas_json_get(y, "fuelType"), code[a->kind]))
                price = num_any(y, "amount");
        const struct janas_json *l = janas_json_get(x, "location");
        double lat = num_any(l, "latitude"), lon = num_any(l, "longitude");
        if (isnan(price) || isnan(lat))
            continue;
        double km = geo_km(a->lat, a->lon, lat, lon);
        if (km > a->radius_km)
            continue;
        struct pr_station *s = &all[k++];
        memset(s, 0, sizeof *s);
        const char *nm = fstr(x, "name"), *ad = fstr(l, "address"),
                   *ci = fstr(l, "city");
        fcopy(s->name, sizeof s->name, nm, nm ? strlen(nm) : 0);
        fcopy(s->address, sizeof s->address, ad, ad ? strlen(ad) : 0);
        fcopy(s->city, sizeof s->city, ci, ci ? strlen(ci) : 0);
        s->lat = lat;
        s->lon = lon;
        s->km = km;
        s->price = price;
        const struct janas_json *of = janas_json_get(x, "offerInformation");
        const struct janas_json *sv = janas_json_get(of, "selfService");
        s->self = sv && sv->type == JANAS_JSON_TRUE    ? 1
                  : sv && sv->type == JANAS_JSON_FALSE ? 0
                                                       : -1;
    }
    janas_json_free(d);
    pr_fuel_rank(f, all, k, a->now);
    return 0;
}
