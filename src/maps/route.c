/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * route.c - maps_route (see maps.h): a route by car, bike or on foot from
 * Valhalla, which gives the steps in the computer's language and a way or
 * two besides; OSRM's, without the steps, when Valhalla does not answer.
 * Both on the servers of FOSSGIS, for free and moderate use.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "maps.h"

#define ROUTE_KEEP_S 600

/* "car" as each engine names it */
static const struct {
    const char *mode, *valhalla, *osrm;
} MODES[] = {{"car", "auto", "routed-car"},
             {"bike", "bicycle", "routed-bike"},
             {"foot", "pedestrian", "routed-foot"}};

/* The computer's language when Valhalla writes its steps in it, else
   English. */
static const char *steps_lang(void)
{
    static const char *const known[] = {
        "bg", "ca", "cs", "da", "de", "el", "en", "es", "et",
        "fi", "fr", "hi", "hu", "it", "ja", "nb", "nl", "pl",
        "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk"};
    const char *l = mp_lang();
    for (size_t i = 0; i < sizeof known / sizeof *known; i++)
        if (strcmp(known[i], l) == 0)
            return l;
    return "en";
}

static int has(const struct janas_json *list, const char *what)
{
    for (const struct janas_json *v =
             list && list->type == JANAS_JSON_ARRAY ? list->child : NULL;
         v; v = v->next)
        if (janas_json_is(v, what))
            return 1;
    return 0;
}

static int valhalla(const struct mp_place *pt[], int n, int m,
                    const struct janas_json *avoid, struct mp_routes *rs,
                    char *err, size_t err_len)
{
    struct janas_buf j = {0}, url = {0}, body = {0};
    janas_buf_puts(&j, "{\"locations\": [");
    for (int i = 0; i < n; i++)
        janas_buf_printf(&j, "%s{\"lat\": %.6f, \"lon\": %.6f}", i ? ", " : "",
                         pt[i]->lat, pt[i]->lon);
    janas_buf_printf(&j,
                     "], \"costing\": \"%s\", \"directions_options\": "
                     "{\"units\": \"kilometers\", \"language\": \"%s\"}",
                     MODES[m].valhalla, steps_lang());
    if (n == 2) /* other ways, between two places only */
        janas_buf_puts(&j, ", \"alternates\": 2");
    if (has(avoid, "tolls") || has(avoid, "highways") ||
        has(avoid, "ferries")) {
        janas_buf_printf(&j, ", \"costing_options\": {\"%s\": {",
                         MODES[m].valhalla);
        const char *sep = "";
        if (has(avoid, "tolls")) {
            janas_buf_printf(&j, "%s\"use_tolls\": 0", sep);
            sep = ", ";
        }
        if (has(avoid, "highways")) {
            janas_buf_printf(&j, "%s\"use_highways\": 0", sep);
            sep = ", ";
        }
        if (has(avoid, "ferries"))
            janas_buf_printf(&j, "%s\"use_ferry\": 0", sep);
        janas_buf_puts(&j, "}}");
    }
    janas_buf_puts(&j, "}");
    janas_buf_put(&j, "", 1);
    janas_buf_printf(
        &url, "%s/route?json=",
        mp_base("JANAS_VALHALLA_URL", "https://valhalla1.openstreetmap.de"));
    if (!j.oom)
        mp_url_put(&url, j.p);
    janas_buf_free(&j);
    int status =
        url.oom ? -1 : mp_fetch(url.p, ROUTE_KEEP_S, &body, err, err_len);
    janas_buf_free(&url);
    int rc = -1;
    /* an answer of 400 says why: no road near a place, too far */
    if (status == 200 || status == 400)
        rc = mp_read_valhalla(body.p ? body.p : "", body.n, rs, err, err_len);
    else if (status > 0)
        snprintf(err, err_len, "Valhalla: HTTP %d", status);
    janas_buf_free(&body);
    return rc;
}

static int osrm(const struct mp_place *pt[], int n, int m, struct mp_routes *rs,
                char *err, size_t err_len)
{
    struct janas_buf url = {0}, body = {0};
    janas_buf_printf(
        &url, "%s/%s/route/v1/driving/",
        mp_base("JANAS_OSRM_URL", "https://routing.openstreetmap.de"),
        MODES[m].osrm);
    for (int i = 0; i < n; i++)
        janas_buf_printf(&url, "%s%.6f,%.6f", i ? ";" : "", pt[i]->lon,
                         pt[i]->lat);
    janas_buf_puts(&url, "?overview=false&steps=true&alternatives=true");
    int status =
        url.oom ? -1 : mp_fetch(url.p, ROUTE_KEEP_S, &body, err, err_len);
    janas_buf_free(&url);
    int rc = -1;
    if (status == 200 || status == 400)
        rc = mp_read_osrm(body.p ? body.p : "", body.n, rs, err, err_len);
    else if (status > 0)
        snprintf(err, err_len, "OSRM: HTTP %d", status);
    janas_buf_free(&body);
    return rc;
}

int mp_tool_route(const struct janas_json *args, struct janas_buf *b)
{
    const char *mode = mp_arg_str(args, "mode");
    int m = 0;
    for (size_t i = 0; mode && i < sizeof MODES / sizeof *MODES; i++)
        if (strcmp(mode, MODES[i].mode) == 0)
            m = (int)i;
    struct mp_place from, via, to;
    if (mp_arg_place(args, "from", 1, NAN, NAN, &from, b) != 0 ||
        mp_arg_place(args, "to", 0, from.lat, from.lon, &to, b) != 0)
        return 0;
    int has_via = mp_arg_str(args, "via") != NULL;
    if (has_via &&
        mp_arg_place(args, "via", 0, from.lat, from.lon, &via, b) != 0)
        return 0;
    const struct mp_place *pt[3] = {&from, has_via ? &via : &to, &to};
    int n = has_via ? 3 : 2;
    struct mp_routes rs;
    char e1[384] = "", e2[384] = "";
    int ok = valhalla(pt, n, m, janas_json_get(args, "avoid"), &rs, e1,
                      sizeof e1) == 0;
    if (!ok)
        ok = osrm(pt, n, m, &rs, e2, sizeof e2) == 0;
    if (!ok) {
        struct janas_buf out = {0};
        janas_buf_printf(&out, "No route from %s to %s: %s%s%s.", from.name,
                         to.name, e1, e2[0] ? "; " : "", e2);
        return mp_result(&out, b, 1);
    }
    struct janas_buf d = {0};
    mp_route_data(&d, &from, has_via ? &via : NULL, &to, MODES[m].mode, &rs);
    mp_routes_free(&rs);
    return mp_answer(b, "maps_route", &d, MP_ROUTE_LAYOUT, MP_ROUTE_BRIEF);
}
