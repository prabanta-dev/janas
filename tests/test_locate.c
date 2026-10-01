/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_locate.c - where the user is (src/common/locate.c), without the
 * network: the answers of GeoJS and ipwho.is as they come (numbers as
 * strings in the first, a nested zone in the second), the broken ones, and
 * JANAS_LOCATION as a city, a point, "off" and nonsense.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/locate.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void test_parsers(void)
{
    struct janas_where w;
    const char *geojs =
        "{\"organization_name\":\"Example\",\"region\":\"Lazio\","
        "\"accuracy\":20,\"country_code\":\"IT\",\"longitude\":\"12.4839\","
        "\"city\":\"Rome\",\"timezone\":\"Europe/Rome\",\"country\":"
        "\"Italy\",\"latitude\":\"41.8919\"}";
    CHECK(janas_where_geojs(geojs, strlen(geojs), &w) == 0, "geojs");
    CHECK(fabs(w.lat - 41.8919) < 1e-9 && fabs(w.lon - 12.4839) < 1e-9,
          "geojs point %f %f", w.lat, w.lon);
    CHECK(strcmp(w.city, "Rome") == 0 && strcmp(w.region, "Lazio") == 0 &&
              strcmp(w.cc, "IT") == 0 && strcmp(w.tz, "Europe/Rome") == 0,
          "geojs fields: %s %s %s %s", w.city, w.region, w.cc, w.tz);
    const char *nopos = "{\"country\":\"Italy\",\"latitude\":\"nil\"}";
    CHECK(janas_where_geojs(nopos, strlen(nopos), &w) != 0, "geojs no point");
    CHECK(janas_where_geojs("<html>", 6, &w) != 0, "geojs not json");

    const char *whois =
        "{\"success\": true, \"country\": \"Italy\", \"country_code\": "
        "\"IT\", \"region\": \"Sicilia\", \"city\": \"Palermo\", "
        "\"latitude\": 38.1157, \"longitude\": 13.3615, \"timezone\": "
        "{\"id\": \"Europe/Rome\", \"abbr\": \"CEST\"}}";
    CHECK(janas_where_ipwhois(whois, strlen(whois), &w) == 0, "ipwhois");
    CHECK(fabs(w.lat - 38.1157) < 1e-9 && strcmp(w.city, "Palermo") == 0 &&
              strcmp(w.tz, "Europe/Rome") == 0,
          "ipwhois fields: %f %s %s", w.lat, w.city, w.tz);
    const char *fail = "{\"success\": false, \"message\": \"Reserved "
                       "range\", \"latitude\": 0, \"longitude\": 0}";
    CHECK(janas_where_ipwhois(fail, strlen(fail), &w) != 0, "ipwhois failed");
}

static void test_setting(void)
{
    struct janas_where w;
    char err[256];
    setenv("JANAS_LOCATION", "off", 1);
    CHECK(janas_where(&w, err, sizeof err) != 0 && strstr(err, "off"),
          "off: %s", err);
    setenv("JANAS_LOCATION", "Trapani", 1);
    CHECK(janas_where(&w, err, sizeof err) == 0 && fabs(w.lat - 38.02) < 0.1 &&
              fabs(w.lon - 12.51) < 0.1 && strcmp(w.tz, "Europe/Rome") == 0 &&
              strstr(w.how, "JANAS_"),
          "Trapani: %f %f %s %s", w.lat, w.lon, w.tz, w.how);
    setenv("JANAS_LOCATION", "45.07, 7.69", 1);
    CHECK(janas_where(&w, err, sizeof err) == 0 && fabs(w.lat - 45.07) < 1e-9 &&
              strcmp(w.cc, "IT") == 0 && strcmp(w.tz, "Europe/Rome") == 0 &&
              strstr(w.country, "Ital"),
          "point: %f %s %s", w.lat, w.tz, w.country);
    setenv("JANAS_LOCATION", "Xyzzyville", 1);
    CHECK(janas_where(&w, err, sizeof err) != 0 && strstr(err, "Xyzzyville"),
          "nonsense: %s", err);
    unsetenv("JANAS_LOCATION");
}

int main(void)
{
    test_parsers();
    test_setting();
    if (failures) {
        printf("test_locate: %d failures\n", failures);
        return 1;
    }
    printf("test_locate: ok\n");
    return 0;
}
