/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * fuel.c - janas-prices' fuel stations near a place (see prices.h), from
 * the open data of the countries that publish their stations' prices:
 * Italy (MIMIT, a day's prices, IODL 2.0), France (the instant flow,
 * Licence Ouverte), Spain (the ministry's stations, by province) and
 * Austria (E-Control's fuel price calculator). The place's country picks
 * the source; another country is said to be not covered.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "prices.h"

#define MS 60000
#define IT_KEEP_S (3 * 3600) /* the files change once a day */
#define FR_KEEP_S 600
#define ES_KEEP_S 1800
#define AT_KEEP_S 600

static int italy(const struct pr_fuel_ask *a, struct pr_fuel *f, char *err,
                 size_t err_len)
{
    struct janas_buf reg = {0}, pri = {0};
    int ok =
        pr_fetch("https://www.mimit.gov.it/images/exportCSV/"
                 "anagrafica_impianti_attivi.csv",
                 IT_KEEP_S, MS, 1, &reg, err, err_len) == 200 &&
        pr_fetch("https://www.mimit.gov.it/images/exportCSV/"
                 "prezzo_alle_8.csv",
                 IT_KEEP_S, MS, 1, &pri, err, err_len) == 200 &&
        pr_read_fuel_it(reg.p, reg.n, pri.p, pri.n, a, f, err, err_len) == 0;
    janas_buf_free(&reg);
    janas_buf_free(&pri);
    return ok ? 0 : -1;
}

static int france(const struct pr_fuel_ask *a, struct pr_fuel *f, char *err,
                  size_t err_len)
{
    static const char *const field[] = {"e10_prix", "gazole_prix", "gplc_prix",
                                        NULL};
    if (!field[a->kind]) {
        snprintf(err, err_len, "France's open data has no price of methane");
        return -1;
    }
    char where[300];
    snprintf(where, sizeof where,
             "within_distance(geom, geom'POINT(%.5f %.5f)', %.1fkm) and %s "
             "is not null",
             a->lon, a->lat, a->radius_km, field[a->kind]);
    struct janas_buf url = {0}, body = {0};
    janas_buf_puts(&url, "https://data.economie.gouv.fr/api/explore/v2.1/"
                         "catalog/datasets/"
                         "prix-des-carburants-en-france-flux-instantane-v2/"
                         "records?where=");
    pr_url_put(&url, where);
    janas_buf_printf(&url, "&order_by=%s&limit=100", field[a->kind]);
    int ok = !url.oom &&
             pr_fetch(url.p, FR_KEEP_S, MS, 1, &body, err, err_len) == 200 &&
             pr_read_fuel_fr(body.p, body.n, a, f, err, err_len) == 0;
    janas_buf_free(&url);
    janas_buf_free(&body);
    return ok ? 0 : -1;
}

/* energia.serviciosmin.gob.es: the ministry's older address,
   sedeaplicaciones.minetur.gob.es, offers only TLS 1.2 with RSA's key
   exchange, which Mbed TLS 4 no longer has */
static int spain(const struct pr_place *p, const struct pr_fuel_ask *a,
                 struct pr_fuel *f, char *err, size_t err_len)
{
    /* the province: a postcode's first two digits */
    if (strlen(p->postcode) < 2 || p->postcode[0] < '0' ||
        p->postcode[0] > '5') {
        snprintf(err, err_len, "the province of %s is not known: name a town",
                 p->name);
        return -1;
    }
    char url[200];
    snprintf(url, sizeof url,
             "https://energia.serviciosmin.gob.es/ServiciosRestCarburantes/"
             "PreciosCarburantes/EstacionesTerrestres/FiltroProvincia/%.2s",
             p->postcode);
    struct janas_buf body = {0};
    int ok = pr_fetch(url, ES_KEEP_S, MS, 1, &body, err, err_len) == 200 &&
             pr_read_fuel_es(body.p, body.n, a, f, err, err_len) == 0;
    janas_buf_free(&body);
    return ok ? 0 : -1;
}

static int austria(const struct pr_fuel_ask *a, struct pr_fuel *f, char *err,
                   size_t err_len)
{
    static const char *const code[] = {"SUP", "DIE", NULL, "GAS"};
    if (!code[a->kind]) {
        snprintf(err, err_len, "E-Control has no price of LPG");
        return -1;
    }
    char url[300];
    snprintf(url, sizeof url,
             "https://api.e-control.at/sprit/1.0/search/gas-stations/"
             "by-address?latitude=%.5f&longitude=%.5f&fuelType=%s"
             "&includeClosed=false",
             a->lat, a->lon, code[a->kind]);
    struct janas_buf body = {0};
    int ok = pr_fetch(url, AT_KEEP_S, MS, 1, &body, err, err_len) == 200 &&
             pr_read_fuel_at(body.p, body.n, a, f, err, err_len) == 0;
    janas_buf_free(&body);
    return ok ? 0 : -1;
}

int pr_tool_fuel(const struct janas_json *args, struct janas_buf *b)
{
    static const char *const kinds[] = {"petrol", "diesel", "lpg", "cng"};
    const char *k = pr_arg_str(args, "fuel");
    int kind = -1;
    for (int i = 0; k && i < 4; i++)
        if (!strcmp(k, kinds[i]))
            kind = i;
    if (kind < 0)
        return pr_fail(b, "The fuel: petrol, diesel, lpg or cng.");
    struct pr_place p;
    char err[400] = "";
    if (pr_place_find(pr_arg_str(args, "place"), &p, err, sizeof err) != 0)
        return pr_fail(b, "%s. Ask the user which place.", err);
    double r = pr_arg_num(args, "radius", 5);
    r = r < 1 ? 1 : r > 20 ? 20 : r;
    struct pr_fuel_ask a = {.now = time(NULL),
                            .lat = p.lat,
                            .lon = p.lon,
                            .radius_km = r,
                            .kind = (enum pr_fuel_kind)kind};
    struct pr_fuel f;
    const char *source;
    int rc;
    if (!strcmp(p.cc, "IT")) {
        rc = italy(&a, &f, err, sizeof err);
        source = "mimit";
    } else if (!strcmp(p.cc, "FR")) {
        rc = france(&a, &f, err, sizeof err);
        source = "fr";
    } else if (!strcmp(p.cc, "ES")) {
        rc = spain(&p, &a, &f, err, sizeof err);
        source = "es";
    } else if (!strcmp(p.cc, "AT")) {
        rc = austria(&a, &f, err, sizeof err);
        source = "at";
    } else {
        return pr_fail(b,
                       "The fuel prices of %s are not covered: open data of "
                       "every station is published, and read here, for "
                       "Italy, France, Spain and Austria. Tell the user so.",
                       p.country[0] ? p.country : p.name);
    }
    if (rc != 0)
        return pr_fail(b, "The fuel prices near %s could not be read: %s.",
                       p.name, err);
    struct janas_buf d = {0};
    pr_fuel_data(&d, &p, kinds[kind], r, &f, source);
    return pr_answer(b, "prices_fuel", &d, PR_FUEL_LAYOUT, PR_FUEL_BRIEF);
}
