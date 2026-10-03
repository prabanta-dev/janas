/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * flights.h - janas-flights, flights as an MCP service: what its parts
 * share. services/common/https_get.c fetches, adsb.c reads the community ADS-B
 * networks (adsb.lol, adsb.fi as the fallback), route.c asks adsbdb for a
 * callsign's route and an airline's codes, schedule.c asks AviationStack
 * for the times (with a key of the user's), tools.c offers the tools.
 */
#ifndef JANAS_FLIGHTS_H
#define JANAS_FLIGHTS_H

#include <stddef.h>
#include <time.h>

#include "services/common/https_get.h"
#include "llm/json.h"

/* ---- adsb.c ---- */

/* One aircraft as the receivers last heard it. */
struct fl_ac {
    char hex[8], flight[12], reg[16], type[8], squawk[6];
    int has_pos, has_alt, ground;
    double lat, lon;
    double alt;      /* feet, barometric */
    double gs;       /* knots over the ground; < 0: unknown */
    double track;    /* degrees; < 0: unknown */
    double rate;     /* feet a minute; 0 when unknown */
    double seen_pos; /* seconds since the position */
};

/* The aircraft of a query of the networks' v2 API: "callsign/RYR1234",
   "hex/4d2234", "lat/45.2/lon/7.6/dist/50". adsb.lol first, adsb.fi when it
   fails. *ac is malloc'd (free it); *source names the network that
   answered. 0, or -1 with the reason in err (both failed). */
int fl_adsb(const char *query, struct fl_ac **ac, size_t *n,
            const char **source, char *err, size_t err_len);

/* ---- route.c ---- */

struct fl_place {
    char iata[4], icao[5], name[96], city[64], country[64];
    double lat, lon;
};

struct fl_route {
    char airline[96], airline_icao[4], airline_iata[3];
    char flight_iata[12]; /* "FR1234", when known */
    struct fl_place from, to;
    int scheduled; /* from the schedule (sure), not from adsbdb */
};

/* The route adsbdb has for a callsign: 1, 0 when it has none, -1 when it
   could not be asked. Its routes come from the community and are wrong
   for many low-cost callsigns, which are reused. */
int fl_route(const char *callsign, struct fl_route *r);

/* The ICAO code of an airline given its IATA code ("FR" -> "RYR"), from
   adsbdb: 1, 0 when unknown, -1 when it could not be asked. */
int fl_airline_icao(const char *iata, char icao[4], char *name,
                    size_t name_len);

/* ---- schedule.c ---- */

/* One end of a scheduled flight; times are 0 when not known. */
struct fl_end {
    char iata[4], icao[5], airport[96], tz[48];
    char terminal[16], gate[16], baggage[16];
    int delay; /* minutes; -1 when not given */
    time_t scheduled, estimated, actual;
};

/* A flight as AviationStack has it. */
struct fl_sched {
    char date[11], status[16]; /* scheduled, active, landed, cancelled... */
    char airline[96], airline_iata[3], airline_icao[4];
    char flight_iata[12], flight_icao[12];
    char operated_by[12], operator_name[96]; /* set on a codeshare */
    char reg[16], hex[8];                    /* the aircraft, when known */
    struct fl_end dep, arr;
};

/* The key, from JANAS_AVIATIONSTACK_KEY or the line "aviationstack_key =
   ..." of the file at path: 1 when there is one. */
int fl_sched_config(const char *path);
/* a line of the settings file not understood, or "" */
const char *fl_sched_problem(void);
int fl_sched_have(void);

/* The flights of a query of /v1/flights ("flight_iata=FR5904",
   "dep_iata=TRN&arr_iata=TPS"); answers kept ten minutes. *v is malloc'd.
   0, or -1 with the reason in err (never the key). */
int fl_schedule(const char *query, struct fl_sched **v, size_t *n, char *err,
                size_t err_len);

/* An answer of /v1/flights read: as fl_schedule, without the network. */
int fl_sched_parse(const char *text, size_t len, struct fl_sched **v, size_t *n,
                   char *err, size_t err_len);

void fl_sched_free(void);

/* ---- layouts.c ---- */

/* flights_over's layout and brief (services/common/template.h). */
extern const char FL_OVER_LAYOUT[], FL_OVER_BRIEF[];
/* flights_nearby's */
extern const char FL_NEAR_LAYOUT[], FL_NEAR_BRIEF[];

/* ---- tools.c ---- */

void fl_tools_list(void *ctx, struct janas_buf *b);
int fl_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);

#endif
