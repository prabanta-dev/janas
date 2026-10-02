/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * nearby.c - maps_nearby (see maps.h): the places of a kind around a
 * point, from OpenStreetMap through Overpass, nearest first. The kinds are
 * a short list, each a filter of OSM's tags, so that the model picks one
 * and never writes a query.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "maps.h"

#define NEAR_KEEP_S 3600
#define NEAR_LIST 10
/* Overpass may take its 20 s; two servers asked within the minute
   janas-chat waits for a tool */
#define NEAR_WAIT_MS 25000

const struct mp_kind MP_KINDS[] = {
    {"fuel", "[amenity=fuel]"},
    {"charging", "[amenity=charging_station]"},
    {"parking", "[amenity=parking]"},
    {"pharmacy", "[amenity=pharmacy]"},
    {"hospital", "[amenity=hospital]"},
    {"doctor", "[amenity~\"^(doctors|clinic)$\"]"},
    {"atm", "[amenity=atm]"},
    {"bank", "[amenity=bank]"},
    {"post_office", "[amenity=post_office]"},
    {"police", "[amenity=police]"},
    {"supermarket", "[shop=supermarket]"},
    {"restaurant", "[amenity=restaurant]"},
    {"cafe", "[amenity=cafe]"},
    {"bar", "[amenity~\"^(bar|pub)$\"]"},
    {"hotel", "[tourism~\"^(hotel|guest_house|hostel|motel)$\"]"},
    {"toilets", "[amenity=toilets]"},
    {"train_station", "[railway=station]"},
    {"bus_stop", "[highway=bus_stop]"},
    {"museum", "[tourism=museum]"},
    {"attraction", "[tourism~\"^(attraction|viewpoint)$\"]"},
};
const size_t MP_N_KINDS = sizeof MP_KINDS / sizeof *MP_KINDS;

int mp_tool_nearby(const struct janas_json *args, struct janas_buf *b)
{
    const char *what = mp_arg_str(args, "what");
    const struct mp_kind *k = NULL;
    for (size_t i = 0; what && i < MP_N_KINDS; i++)
        if (strcmp(what, MP_KINDS[i].name) == 0)
            k = &MP_KINDS[i];
    if (!k) {
        struct janas_buf out = {0};
        janas_buf_puts(&out, "Give what to look for, one of:");
        for (size_t i = 0; i < MP_N_KINDS; i++)
            janas_buf_printf(&out, " %s", MP_KINDS[i].name);
        janas_buf_puts(&out, ".");
        return mp_result(&out, b, 1);
    }
    struct mp_place at;
    if (mp_arg_place(args, "place", 1, NAN, NAN, &at, b) != 0)
        return 0;
    double r = mp_arg_num(args, "radius", 2);
    r = isnan(r) ? 2 : fmin(10, fmax(0.1, r));
    struct janas_buf q = {0};
    janas_buf_printf(&q,
                     "[out:json][timeout:20];nwr%s(around:%.0f,%.6f,%.6f);"
                     "out center tags 300;",
                     k->filter, r * 1000, at.lat, at.lon);
    janas_buf_put(&q, "", 1);
    /* FOSSGIS's instance first; when it is busy or down, Private.coffee's
       ("Feel free to use our service in any project, there is no rate
       limit in place", OSM wiki, 2 Oct 2026) */
    const char *base[2] = {
        mp_base("JANAS_OVERPASS_URL",
                "https://overpass-api.de/api/interpreter"),
        mp_base("JANAS_OVERPASS_URL2",
                "https://overpass.private.coffee/api/interpreter")};
    char err[2][300] = {"", ""};
    struct mp_near nr;
    int ok = 0, t;
    for (t = 0; t < 2 && !ok && !q.oom; t++) {
        struct janas_buf url = {0}, body = {0};
        janas_buf_printf(&url, "%s?data=", base[t]);
        mp_url_put(&url, q.p);
        int status = url.oom ? -1
                             : mp_fetch_ms(url.p, NEAR_KEEP_S, NEAR_WAIT_MS,
                                           &body, err[t], sizeof err[t]);
        ok = status == 200 &&
             mp_read_overpass(body.p, body.n, at.lat, at.lon, NEAR_LIST, &nr,
                              err[t], sizeof err[t]) == 0;
        const char *host = strstr(base[t], "://");
        host = host ? host + 3 : base[t];
        if (status > 0 && status != 200)
            snprintf(err[t], sizeof err[t], "Overpass at %.*s: HTTP %d%s",
                     (int)strcspn(host, "/"), host, status,
                     status == 429 || status == 504 ? " (busy)" : "");
        janas_buf_free(&url);
        janas_buf_free(&body);
    }
    janas_buf_free(&q);
    if (!ok) {
        struct janas_buf out = {0};
        janas_buf_printf(&out, "Nothing could be looked up near %s: %s%s%s",
                         at.name, err[0], t > 1 ? "; " : "",
                         t > 1 ? err[1] : "");
        janas_buf_puts(&out, ". Try again in a minute.");
        return mp_result(&out, b, 1);
    }
    struct janas_buf d = {0};
    mp_near_data(&d, &at, k->name, r, &nr);
    return mp_answer(b, "maps_nearby", &d, MP_NEAR_LAYOUT, MP_NEAR_BRIEF);
}
