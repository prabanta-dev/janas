/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * maps.h - janas-maps, roads and places as an MCP service: what its parts
 * share. fetch.c asks the sources (one request a second to each, the
 * answers kept a while), read.c reads their answers, place.c finds a place
 * or an address, data.c writes what was read as the data of a layout,
 * route.c, transit.c and nearby.c are the tools, layouts.c their layouts,
 * tools.c lists them and takes their arguments.
 *
 * The sources are free and need no key: OpenStreetMap's data through
 * Nominatim (places and addresses), Photon (when Nominatim does not
 * answer), Valhalla and OSRM on the servers of FOSSGIS (routes by car, bike
 * and on foot), Transitous (public transport) and Overpass (what is near).
 * Each asks for a User-Agent naming the program and for at most one
 * request a second; each address can be changed (JANAS_VALHALLA_URL and
 * the like), as FOSSGIS asks.
 */
#ifndef JANAS_MAPS_H
#define JANAS_MAPS_H

#include <stddef.h>
#include <time.h>

#include "llm/json.h"

/* ---- fetch.c ---- */

/* GET of url, the body into out; answers of status 200 are kept keep_s
   seconds and given again meanwhile; a host is asked at most once a
   second. The HTTP status, or -1 with the reason in err. */
int mp_fetch(const char *url, int keep_s, struct janas_buf *out, char *err,
             size_t err_len);
/* The same, waiting at most timeout_ms for each part of the answer
   (mp_fetch: 15 s). */
int mp_fetch_ms(const char *url, int keep_s, int timeout_ms,
                struct janas_buf *out, char *err, size_t err_len);
void mp_fetch_free(void);

/* The address of a source: the variable env when set, else def. */
const char *mp_base(const char *env, const char *def);

/* s into b, as a URL's query wants it. */
void mp_url_put(struct janas_buf *b, const char *s);

/* The language of the answers: the computer's (LANG=it_IT.UTF-8: "it"),
   else "en". */
const char *mp_lang(void);

/* ---- read.c: the sources' answers ---- */

struct mp_place {
    char name[128];    /* "Stazione Centrale", "Trapani" */
    char address[256]; /* "Stazione Centrale, Via Paolo Balsamo, ..." */
    char city[96];     /* "Palermo"; "" when not known */
    char kind[64];     /* "railway station" (OSM's class and type) */
    char osm[48];      /* "node/123", for a link; "" */
    double lat, lon;
    char how[96]; /* where the user is, when the place was not named: how it
                     was told (locate.h); "" otherwise */
};

/* The first place of Nominatim's or Photon's answer: 0, or -1 when none. */
int mp_read_nominatim(const char *text, size_t n, struct mp_place *p);
int mp_read_photon(const char *text, size_t n, struct mp_place *p);

#define MP_STEPS 160
#define MP_ROUTES 3

struct mp_step {
    char text[200]; /* the source's, in its language */
    double km;
};

struct mp_route {
    double km, min;
    int toll, highway, ferry;
    char roads[160];      /* the longest roads, "A29, SS113" */
    struct mp_step *step; /* the first route's only; NULL for the others */
    int n_step, cut;      /* cut: steps left out (more than MP_STEPS) */
};

struct mp_routes {
    const char *source; /* "Valhalla", "OSRM" */
    struct mp_route r[MP_ROUTES];
    int n;
};

/* Valhalla's /route answer (with its alternates), OSRM's: 0, or -1 with
   the reason in err. The steps are malloc'd: mp_routes_free. */
int mp_read_valhalla(const char *text, size_t n, struct mp_routes *rs,
                     char *err, size_t err_len);
int mp_read_osrm(const char *text, size_t n, struct mp_routes *rs, char *err,
                 size_t err_len);
void mp_routes_free(struct mp_routes *rs);

#define MP_TRIPS 5
#define MP_LEGS 12

struct mp_leg {
    char mode[24];   /* "WALK", "BUS", "REGIONAL_RAIL" */
    char line[64];   /* "REG 21854", "101" */
    char agency[64]; /* "TRENITALIA" */
    char headsign[96];
    char from[96], to[96];
    time_t dep, arr;
    double walk_m;
    int real_time;
};

struct mp_trip {
    time_t dep, arr;
    int transfers;
    struct mp_leg leg[MP_LEGS];
    int n_leg, cut;
};

struct mp_transit {
    struct mp_trip t[MP_TRIPS];
    int n;
    char tz[48]; /* the start's zone, "Europe/Rome" */
};

/* Transitous' /api/v1/plan answer: 0, or -1 with the reason in err. */
int mp_read_transit(const char *text, size_t n, struct mp_transit *tr,
                    char *err, size_t err_len);

#define MP_NEAR 20

struct mp_poi {
    char name[128];
    char address[160];
    char hours[128]; /* OSM's opening_hours, as written */
    char phone[48];
    char osm[48]; /* "node/123" */
    double lat, lon, m;
};

struct mp_near {
    struct mp_poi p[MP_NEAR];
    int n, found; /* listed, and found within the radius */
};

/* Overpass' answer, the nearest to (lat, lon) first, at most max. 0, or -1
   with the reason in err. */
int mp_read_overpass(const char *text, size_t n, double lat, double lon,
                     int max, struct mp_near *nr, char *err, size_t err_len);

/* UTC "2026-10-02T03:42:00Z" as a time, or -1. */
time_t mp_utc(const char *s);

/* ---- place.c ---- */

/* A place or an address by its name ("Via Fardella, Trapani", "Colosseo"),
   or "lat,lon": the nearest to (near_lat, near_lon) when given, preferred.
   0, or -1 with the reason in err. */
int mp_place_find(const char *query, double near_lat, double near_lon,
                  struct mp_place *p, char *err, size_t err_len);
/* A point named from the built-in tables ("3 km N of Palermo"). */
void mp_place_at(double lat, double lon, struct mp_place *p);
/* What is at a point, from Nominatim: its name and address. 0, or -1 (p
   untouched). */
int mp_place_reverse(double lat, double lon, struct mp_place *p);

/* ---- data.c: the data of the layouts ---- */

void mp_jstr(struct janas_buf *b, const char *key, const char *v);
void mp_jnum(struct janas_buf *b, const char *key, double v, int decimals);
/* a place: "name", "address", "how"... as an object under key */
void mp_jplace(struct janas_buf *b, const char *key, const struct mp_place *p);
/* minutes as "h" and "min" */
void mp_jtime(struct janas_buf *b, double min);
/* a time in the zone tz: "at" ("08:42") and, when not today, "date" with
   "wd", "day", "mon", under key */
void mp_jclock(struct janas_buf *b, const char *key, time_t t, const char *tz);
void mp_route_data(struct janas_buf *d, const struct mp_place *from,
                   const struct mp_place *via, const struct mp_place *to,
                   const char *mode, const struct mp_routes *rs);
void mp_transit_data(struct janas_buf *d, const struct mp_place *from,
                     const struct mp_place *to, const struct mp_transit *tr);
void mp_near_data(struct janas_buf *d, const struct mp_place *at,
                  const char *what, double radius_km, const struct mp_near *nr);
void mp_find_data(struct janas_buf *d, const struct mp_place *p);

/* ---- tools.c ---- */

void mp_tools_list(void *ctx, struct janas_buf *b);
int mp_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);

const char *mp_arg_str(const struct janas_json *args, const char *name);
double mp_arg_num(const struct janas_json *args, const char *name, double def);
int mp_arg_bool(const struct janas_json *args, const char *name, int def);
/* The place of argument name: named, or where the user is when it is not
   and here is set. 0; -1 with an error result already in b. */
int mp_arg_place(const struct janas_json *args, const char *name, int here,
                 double near_lat, double near_lon, struct mp_place *p,
                 struct janas_buf *b);
/* A text result, out freed. */
int mp_result(struct janas_buf *out, struct janas_buf *b, int is_error);
/* The answer from the data and its layout (d freed). */
int mp_answer(struct janas_buf *b, const char *name, struct janas_buf *d,
              const char *layout, const char *brief);

int mp_tool_route(const struct janas_json *args, struct janas_buf *b);
int mp_tool_transit(const struct janas_json *args, struct janas_buf *b);
int mp_tool_nearby(const struct janas_json *args, struct janas_buf *b);
int mp_tool_find(const struct janas_json *args, struct janas_buf *b);

/* ---- nearby.c ---- */

/* The kinds of place maps_nearby looks for: name, OSM filter. */
struct mp_kind {
    const char *name, *filter;
};
extern const struct mp_kind MP_KINDS[];
extern const size_t MP_N_KINDS;

/* ---- layouts.c ---- */

extern const char MP_ROUTE_LAYOUT[], MP_ROUTE_BRIEF[];
extern const char MP_TRANSIT_LAYOUT[], MP_TRANSIT_BRIEF[];
extern const char MP_NEAR_LAYOUT[], MP_NEAR_BRIEF[];
extern const char MP_FIND_LAYOUT[], MP_FIND_BRIEF[];

#endif
