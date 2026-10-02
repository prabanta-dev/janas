/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * quakes.h - janas-quakes, the earthquakes as an MCP service: those near a
 * place, the strongest in the world, one in detail. It only reads, from
 * open sources: INGV (Istituto Nazionale di Geofisica e Vulcanologia, CC
 * BY 4.0) for Italy and the seas around it, the USGS (public domain)
 * everywhere, each the other's fallback.
 *
 * What its parts share: fetch.c asks the sources, place.c finds the place
 * meant, read.c reads their GeoJSON (no network: the tests read kept
 * answers), data.c and layouts.c write the answers as data with a layout,
 * tools.c offers the tools.
 */
#ifndef JANAS_QUAKES_H
#define JANAS_QUAKES_H

#include <stddef.h>

#include "llm/json.h"
#include "services/common/geo.h"

#define QK_MAX 20      /* events told */
#define QK_KEEP_S 60   /* an answer kept, so a question again costs none */
#define QK_PAUSE_S 1.0 /* between two requests to a server */
#define QK_MS 20000

/* the area where INGV's catalogue is the reference: Italy and its seas */
#define QK_INGV_LAT0 35.0
#define QK_INGV_LAT1 48.0
#define QK_INGV_LON0 6.0
#define QK_INGV_LON1 19.0

/* ---- fetch.c ---- */

/* GET of url, kept QK_KEEP_S; the HTTP status (204: none), or -1 */
int qk_fetch(const char *url, struct janas_buf *out, char *err, size_t err_len);
/* s into b, escaped for a URL */
void qk_url_put(struct janas_buf *b, const char *s);

/* ---- place.c ---- */

struct qk_place {
    char name[96], country[64], cc[3];
    double lat, lon;
    char how[96]; /* where the user is, when no place was named */
    int is_sea;   /* a sea: the events within it, not around a point */
    struct geo_area sea;
};

/* A place by name ("Norcia", "mar Tirreno"), "lat,lon", or, with q NULL,
   where the user is. 0, or -1 with the reason in err. */
int qk_place_find(const char *q, struct qk_place *p, char *err, size_t err_len);
/* Whether the point is in INGV's area */
int qk_in_ingv(double lat, double lon);

/* ---- read.c ---- */

struct qk_event {
    char id[32], src[8]; /* src: ingv, usgs */
    char place[160];     /* the source's own words for where it was */
    char magtype[8], kind[24], alert[8], url[200];
    double lat, lon, depth_km, mag, mmi; /* mmi: -1 when not told */
    long long at;                        /* seconds since 1970 */
    int felt; /* people who told they felt it, -1 not told */
    int tsunami, reviewed;
};

struct qk_list {
    struct qk_event e[QK_MAX];
    int n, more;
    const struct geo_area *within; /* set: only the events over that sea */
};

/* the FDSN GeoJSON of INGV or of the USGS (src); the events after those
   already in l, up to QK_MAX, the others counted in more */
int qk_read(const char *text, size_t n, const char *src, struct qk_list *l,
            char *err, size_t err_len);
/* "2026-10-01T22:46:42.590000" (UTC) into seconds since 1970, -1 */
long long qk_utc(const char *s);
/* the latest first; the strongest first */
void qk_by_time(struct qk_list *l);
void qk_by_mag(struct qk_list *l);

/* ---- data.c ---- */

/* the events near a place: their distance and way from it */
void qk_near_data(struct janas_buf *b, const struct qk_place *p,
                  double radius_km, int days, double min_mag,
                  const struct qk_list *l, const char *src);
void qk_strong_data(struct janas_buf *b, int days, double min_mag,
                    const struct qk_list *l);
void qk_event_data(struct janas_buf *b, const struct qk_event *e,
                   const struct qk_place *from);

/* ---- layouts.c ---- */

extern const char QK_NEAR_LAYOUT[], QK_NEAR_BRIEF[];
extern const char QK_STRONG_LAYOUT[], QK_STRONG_BRIEF[];
extern const char QK_EVENT_LAYOUT[], QK_EVENT_BRIEF[];

/* ---- tools.c ---- */

void qk_tools_list(void *ctx, struct janas_buf *b);
int qk_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);

#endif
