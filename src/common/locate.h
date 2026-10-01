/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * locate.h - where the user is, for Janas's services when a question
 * names no place ("what's the weather like?", "what flies over here?").
 *
 * In this order: JANAS_LOCATION when set (a place's name, or "lat,lon";
 * "off" for no guess at all); else the internet connection's position
 * (GeoJS, then ipwho.is: free, no key, a city or so off, the provider's
 * rather than the user's at times); else the city of the machine's time
 * zone (Europe/Rome: Rome), a rough one. Kept for an hour.
 */
#ifndef JANAS_COMMON_LOCATE_H
#define JANAS_COMMON_LOCATE_H

#include <stddef.h>

struct janas_where {
    double lat, lon;
    char city[96];   /* "Palermo"; "" when only the point is known */
    char region[64]; /* "Sicily" */
    char country[64];
    char cc[3];   /* "IT" */
    char tz[48];  /* "Europe/Rome" */
    char how[96]; /* "estimated from the internet connection (GeoJS)" */
};

/* 0, or -1 with the reason in err (nothing known, or JANAS_LOCATION=off). */
int janas_where(struct janas_where *w, char *err, size_t err_len);

/* The parsers, for the tests: 0 when the answer gave a position. */
int janas_where_geojs(const char *text, size_t n, struct janas_where *w);
int janas_where_ipwhois(const char *text, size_t n, struct janas_where *w);

#endif
