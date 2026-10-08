/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_geo.c - janas-flights' geography, without the network: the seas a
 * point is over (and land where there is no sea), the airports and cities
 * found by code and name, distances and compass points, and positions put
 * into words. geo.c belongs to the program, so it is compiled in here.
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "services/common/geo.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void sea(double lat, double lon, const char *want)
{
    const char *got = geo_sea_at(lat, lon);
    if (!want)
        CHECK(!got, "%.2f, %.2f: on land, not over %s", lat, lon, got);
    else
        CHECK(got && strcmp(got, want) == 0, "%.2f, %.2f: %s, not %s", lat, lon,
              got ? got : "land", want);
}

static void says(double lat, double lon, const char *const *parts, int n)
{
    struct janas_buf b = {0};
    geo_describe(lat, lon, &b);
    janas_buf_put(&b, "", 1);
    for (int i = 0; i < n; i++)
        CHECK(b.p && strstr(b.p, parts[i]), "%.2f, %.2f: \"%s\" lacks \"%s\"",
              lat, lon, b.p ? b.p : "", parts[i]);
    janas_buf_free(&b);
}

int main(void)
{
    alarm(60);
    /* seas, checked against the Natural Earth polygons on 1 Oct 2026 */
    sea(42.8, 10.0, "Tyrrhenian Sea");   /* between Corsica and Tuscany */
    sea(39.54, 12.79, "Tyrrhenian Sea"); /* VLG1595, Cairo-Barcelona */
    sea(40.75, 14.2, "Tyrrhenian Sea");  /* the Gulf of Naples */
    sea(43.0, 15.0, "Adriatic Sea");
    sea(43.8, 8.5, "Ligurian Sea");
    sea(45.07, 7.69, NULL);  /* Turin */
    sea(43.32, 11.33, NULL); /* Siena */

    const struct geo_airport *a = geo_airport_find("TRN");
    CHECK(a && strcmp(a->icao, "LIMF") == 0, "TRN is not LIMF");
    a = geo_airport_find("lict");
    CHECK(a && strcmp(a->iata, "TPS") == 0, "LICT is not TPS");
    CHECK(a && strcmp(geo_tz_name(a->tz), "Europe/Rome") == 0,
          "Trapani's time zone: %s", a ? geo_tz_name(a->tz) : "-");
    a = geo_airport_find("Turin");
    CHECK(a && strcmp(a->iata, "TRN") == 0, "Turin: %s", a ? a->iata : "none");
    a = geo_airport_find("Trapani");
    CHECK(a && strcmp(a->iata, "TPS") == 0, "Trapani: %s",
          a ? a->iata : "none");
    /* Trapani's by the names people use: OurAirports calls it "Vincenzo
       Florio Airport Trapani-Birgi"; and an en dash in the data, Rome's */
    const char *tps[] = {"Birgi", "Trapani-Birgi", "trapani birgi",
                         "Trapani – Birgi", "Vincenzo Florio"};
    for (int i = 0; i < 5; i++) {
        a = geo_airport_find(tps[i]);
        CHECK(a && strcmp(a->iata, "TPS") == 0, "%s: %s", tps[i],
              a ? a->iata : "none");
    }
    a = geo_airport_find("Rome-Fiumicino");
    CHECK(a && strcmp(a->iata, "FCO") == 0, "Rome-Fiumicino: %s",
          a ? a->iata : "none");
    /* "Caselle" alone is Verona's too (Caselle di Sommacampagna) */
    a = geo_airport_find("Caselle Torinese");
    CHECK(a && strcmp(a->iata, "TRN") == 0, "Caselle Torinese: %s",
          a ? a->iata : "none");
    CHECK(!geo_airport_find("XQZ"), "XQZ found");
    const struct geo_city *c = geo_city_find("Turin");
    CHECK(c && strcmp(geo_country_name(c->country), "Italy") == 0,
          "Turin is not in Italy");
    /* a region by its name, as its largest city: Liguria was a peak in
       Antarctica to the weather service's geocoding (8 Oct 2026) */
    static const char *const reg[][2] = {
        {"Liguria", "Genoa"},       {"Sicilia", "Palermo"},
        {"Sicily", "Palermo"},      {"lombardia", "Milan"},
        {"Valle d'Aosta", "Aosta"}, {"Marche", "Ancona"},
        {"Puglia", "Bari"},         {"Bavaria", "Munich"}};
    for (size_t i = 0; i < sizeof reg / sizeof *reg; i++) {
        c = geo_region_city(reg[i][0]);
        CHECK(c && strcmp(c->ascii, reg[i][1]) == 0, "region %s: %s", reg[i][0],
              c ? c->ascii : "none");
    }
    /* the largest city, not the seat: Basilicata's is Matera */
    c = geo_region_city("Basilicata");
    CHECK(c && strcmp(geo_region_name(c->region), "Basilicate") == 0,
          "Basilicata: %s", c ? geo_region_name(c->region) : "none");
    CHECK(!geo_region_city("Nowhereland"), "Nowhereland is a region");
    /* cities by their Italian names: "Roma" was Roma in Queensland's
       airport, Torino nothing (8 Oct 2026) */
    static const char *const cit[][3] = {{"Roma", "Rome", "IT"},
                                         {"Torino", "Turin", "IT"},
                                         {"Siracusa", "Siracusa", "IT"},
                                         {"Londra", "London", "GB"},
                                         {"Parigi", "Paris", "FR"}};
    for (size_t i = 0; i < sizeof cit / sizeof *cit; i++) {
        c = geo_city_find(cit[i][0]);
        CHECK(c && strcmp(c->ascii, cit[i][1]) == 0 &&
                  strcmp(geo_country_iso2(c->country), cit[i][2]) == 0,
              "city %s: %s", cit[i][0], c ? c->ascii : "none");
    }
    a = geo_airport_find("Roma");
    CHECK(a && strcmp(geo_country_iso2(a->country), "IT") == 0,
          "Roma's airport: %s", a ? a->iata : "none");
    a = geo_airport_find("Firenze");
    CHECK(a && strcmp(a->iata, "FLR") == 0, "Firenze's airport: %s",
          a ? a->iata : "none");
    a = geo_airport_find("Florence");
    CHECK(a && strcmp(a->iata, "FLR") == 0, "Florence's airport: %s",
          a ? a->iata : "none");
    c = geo_city_find("livorno");
    CHECK(c && strcmp(geo_region_name(c->region), "Tuscany") == 0,
          "Livorno is not in Tuscany");

    double km = 0;
    a = geo_airport_near(45.20, 7.65, &km);
    CHECK(a && strcmp(a->iata, "TRN") == 0 && km < 1,
          "nearest to Caselle: %s at %.1f km", a ? a->iata : "-", km);
    /* Turin-Trapani, great circle: 810 km of latitude and 402 of longitude,
       about 904 km */
    double d = geo_km(45.2008, 7.6497, 37.9114, 12.4880);
    CHECK(d > 895 && d < 915, "TRN-TPS: %.0f km", d);
    CHECK(strcmp(geo_compass(0), "N") == 0 &&
              strcmp(geo_compass(350), "N") == 0 &&
              strcmp(geo_compass(247.5), "WSW") == 0 &&
              strcmp(geo_compass(90), "E") == 0,
          "compass points");
    CHECK(fabs(geo_bearing(0, 0, 0, 1) - 90) < 0.01, "east is 90 degrees");

    /* seas as areas to search: by English and Italian names */
    struct geo_area ar;
    const char *tyr[] = {"Tyrrhenian Sea", "tyrrhenian", "Tirreno",
                         "mar Tirreno", "the Tyrrhenian Sea"};
    for (int i = 0; i < 5; i++)
        CHECK(geo_area_find(tyr[i], &ar) &&
                  strcmp(ar.name, "Tyrrhenian Sea") == 0,
              "%s: not the Tyrrhenian Sea", tyr[i]);
    CHECK(geo_area_find("Tirreno", &ar) && geo_area_has(&ar, 42.8, 10.0) &&
              geo_area_has(&ar, 39.54, 12.79) &&
              !geo_area_has(&ar, 43.8, 8.5) &&
              !geo_area_has(&ar, 45.07, 7.69) && !geo_area_has(&ar, 43.0, 15.0),
          "the Tyrrhenian Sea as an area");
    CHECK(geo_area_find("mare Adriatico", &ar) &&
              geo_area_has(&ar, 43.0, 15.0) && !geo_area_has(&ar, 42.8, 10.0),
          "the Adriatic Sea as an area");
    /* the Mediterranean holds the seas Natural Earth cut out of it */
    CHECK(geo_area_find("Mediterranean", &ar) &&
              geo_area_has(&ar, 42.8, 10.0) && geo_area_has(&ar, 43.0, 15.0) &&
              !geo_area_has(&ar, 45.07, 7.69),
          "the Mediterranean as an area");
    CHECK(!geo_area_find("Sea of Nowhere", &ar), "Sea of Nowhere found");

    const char *tirreno[] = {"over the Tyrrhenian Sea", "Tuscany", "km"};
    says(42.8, 10.0, tirreno, 3);
    const char *land[] = {"over Tuscany, Italy", "of Siena"};
    says(43.25, 11.36, land, 2);

    if (failures) {
        printf("test_geo: %d failures\n", failures);
        return 1;
    }
    printf("test_geo: ok\n");
    return 0;
}
