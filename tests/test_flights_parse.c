/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_flights_parse.c - janas-flights' reading of AviationStack's answers,
 * without the network: a flight as it came on 1 Oct 2026 (Turin-Trapani,
 * the evening before), a codeshare and an error, made by hand. The times
 * come in the airport's local time marked +00:00: read in its zone.
 * schedule.c belongs to the program, so it is compiled in here, with a
 * janas_https_get that never goes out and quotes the address in its error.
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "flights/schedule.c"

int janas_https_get(const char *url, const char *const *headers,
                    size_t n_headers, struct janas_buf *out, char *err,
                    size_t err_len)
{
    (void)headers, (void)n_headers, (void)out;
    /* as the URL parser does: its errors quote the address */
    snprintf(err, err_len, "%.200s: no network in a test", url);
    return -1;
}

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char TRN_TPS[] =
    "{\"pagination\": {\"limit\": 100, \"offset\": 0, \"count\": 1, "
    "\"total\": 1}, \"data\": [{\"flight_date\": \"2026-09-30\", "
    "\"flight_status\": \"landed\", \"departure\": {\"airport\": \"Sandro "
    "Pertini (caselle)\", \"timezone\": \"Europe/Rome\", \"iata\": \"TRN\", "
    "\"icao\": \"LIMF\", \"terminal\": null, \"gate\": \"20\", \"delay\": 0, "
    "\"scheduled\": \"2026-09-30T18:25:00+00:00\", \"estimated\": "
    "\"2026-09-30T18:25:00+00:00\", \"actual\": "
    "\"2026-09-30T18:23:00+00:00\", \"estimated_runway\": "
    "\"2026-09-30T18:23:00+00:00\", \"actual_runway\": "
    "\"2026-09-30T18:23:00+00:00\"}, \"arrival\": {\"airport\": \"Birgi\", "
    "\"timezone\": \"Europe/Rome\", \"iata\": \"TPS\", \"icao\": \"LICT\", "
    "\"terminal\": null, \"gate\": null, \"baggage\": null, \"scheduled\": "
    "\"2026-09-30T20:05:00+00:00\", \"delay\": 0, \"estimated\": "
    "\"2026-09-30T19:38:00+00:00\", \"actual\": "
    "\"2026-09-30T19:36:00+00:00\", \"estimated_runway\": "
    "\"2026-09-30T19:36:00+00:00\", \"actual_runway\": "
    "\"2026-09-30T19:36:00+00:00\"}, \"airline\": {\"name\": \"Ryanair\", "
    "\"iata\": \"FR\", \"icao\": \"RYR\"}, \"flight\": {\"number\": "
    "\"5904\", \"iata\": \"FR5904\", \"icao\": \"RYR5904\", \"codeshared\": "
    "null}, \"aircraft\": {\"registration\": null, \"iata\": null, "
    "\"icao\": null, \"icao24\": \"4CAFB5\"}, \"live\": null}]}";

static const char CODESHARE[] =
    "{\"data\": [{\"flight_date\": \"2026-10-01\", \"flight_status\": "
    "\"scheduled\", \"departure\": {\"iata\": \"FCO\", \"timezone\": "
    "\"Europe/Rome\", \"scheduled\": \"2026-10-01T07:05:00+00:00\", "
    "\"delay\": null}, \"arrival\": {\"iata\": \"JFK\", \"timezone\": "
    "\"America/New_York\", \"scheduled\": \"2026-10-01T10:30:00+00:00\"}, "
    "\"airline\": {\"name\": \"Delta Air Lines\", \"iata\": \"DL\", "
    "\"icao\": \"DAL\"}, \"flight\": {\"iata\": \"DL6767\", \"icao\": "
    "\"DAL6767\", \"codeshared\": {\"airline_name\": \"ita airways\", "
    "\"airline_iata\": \"az\", \"flight_number\": \"608\", \"flight_iata\": "
    "\"az608\"}}, \"aircraft\": null}]}";

static const char QUOTA[] =
    "{\"error\": {\"code\": \"usage_limit_reached\", \"message\": \"Your "
    "monthly usage limit has been reached.\"}}";

int main(void)
{
    alarm(60);
    struct fl_sched *v;
    size_t n;
    char err[300];
    int r =
        fl_sched_parse(TRN_TPS, sizeof TRN_TPS - 1, &v, &n, err, sizeof err);
    CHECK(r == 0 && n == 1, "Turin-Trapani: %d, %zu flights: %s", r, n, err);
    if (r == 0 && n == 1) {
        CHECK(strcmp(v[0].flight_iata, "FR5904") == 0 &&
                  strcmp(v[0].flight_icao, "RYR5904") == 0 &&
                  strcmp(v[0].airline, "Ryanair") == 0 &&
                  strcmp(v[0].status, "landed") == 0 &&
                  strcmp(v[0].date, "2026-09-30") == 0,
              "Turin-Trapani: the flight");
        CHECK(strcmp(v[0].hex, "4cafb5") == 0, "hex %s", v[0].hex);
        CHECK(strcmp(v[0].dep.iata, "TRN") == 0 &&
                  strcmp(v[0].arr.icao, "LICT") == 0 &&
                  strcmp(v[0].dep.gate, "20") == 0 && !v[0].arr.gate[0],
              "Turin-Trapani: the airports and gate");
        /* 18:25 in Rome, summer time, is 16:25 UTC */
        CHECK(v[0].dep.scheduled == 1790785500,
              "departure scheduled %lld, not 16:25 UTC",
              (long long)v[0].dep.scheduled);
        CHECK(v[0].arr.actual - v[0].dep.actual == 73 * 60,
              "a flight of 73 minutes, not %lld",
              (long long)(v[0].arr.actual - v[0].dep.actual));
        CHECK(v[0].dep.delay == 0 && v[0].arr.delay == 0, "delays");
        CHECK(!v[0].operated_by[0], "not a codeshare");
    }
    free(v);

    r = fl_sched_parse(CODESHARE, sizeof CODESHARE - 1, &v, &n, err,
                       sizeof err);
    CHECK(r == 0 && n == 1, "codeshare: %d, %zu: %s", r, n, err);
    if (r == 0 && n == 1) {
        CHECK(strcmp(v[0].operated_by, "AZ608") == 0 &&
                  strcmp(v[0].operator_name, "ita airways") == 0,
              "codeshare of %s", v[0].operated_by);
        CHECK(v[0].dep.delay == -1 && !v[0].hex[0], "no delay, no aircraft");
        /* 07:05 in Rome and 10:30 in New York: 9 h 25 min */
        CHECK(v[0].arr.scheduled - v[0].dep.scheduled == (9 * 60 + 25) * 60,
              "Rome-New York, %lld s",
              (long long)(v[0].arr.scheduled - v[0].dep.scheduled));
    }
    free(v);

    snprintf(key, sizeof key, "0123456789abcdef");
    r = fl_sched_parse(QUOTA, sizeof QUOTA - 1, &v, &n, err, sizeof err);
    CHECK(r == -1 && strstr(err, "usage_limit_reached") && !v, "quota: %d, %s",
          r, err);
    r = fl_sched_parse("<html>", 6, &v, &n, err, sizeof err);
    CHECK(r == -1 && !v, "not JSON: %d", r);
    /* a bad address quotes itself, and the key is in it: never said */
    r = fl_schedule("flight_iata=FR5904", &v, &n, err, sizeof err);
    CHECK(r == -1 && !strstr(err, key), "the key in an error: %s", err);
    fl_sched_free();

    if (failures) {
        printf("test_flights_parse: %d failures\n", failures);
        return 1;
    }
    printf("test_flights_parse: ok\n");
    return 0;
}
