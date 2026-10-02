/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - janas-quakes' tools (see quakes.h). They only read, and say so
 * (readOnlyHint): the earthquakes near a place, the strongest in the
 * world, one in detail. A point in Italy or its seas is asked of INGV,
 * any other of the USGS; each source is the other's fallback.
 */
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "common/mcp_server.h"
#include "quakes.h"
#include "services/common/template.h"

#define READS                                                                  \
    "\"annotations\": {\"readOnlyHint\": true, \"openWorldHint\": true}"
#define PLACE_ARG                                                              \
    "\"place\": {\"type\": \"string\", \"description\": \"a town, a region "   \
    "or a sea; leave it out for where the user is\"}"

#define INGV "https://webservices.ingv.it/fdsnws/event/1/query?format=geojson"
#define USGS "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"

void qk_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    janas_buf_puts(
        b,
        "\"tools\": [{\"name\": \"quakes_near\", \"title\": \"Earthquakes "
        "near a place\", \"description\": \"The latest earthquakes around a "
        "place: was there one, where was it, how strong.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": "
        "{" PLACE_ARG ", \"radius\": {\"type\": \"number\", \"description\": "
        "\"km, 300 by default, at most 1000\"}, \"days\": {\"type\": "
        "\"number\", \"description\": \"7 by default, at most 30\"}, "
        "\"min_mag\": {\"type\": \"number\", \"description\": \"2.5 by "
        "default\"}}, \"additionalProperties\": false}, " READS "}, "
        "{\"name\": \"quakes_strong\", \"title\": \"The strongest "
        "earthquakes\", \"description\": \"The strongest earthquakes in the "
        "world in the last days, with the impact expected.\", "
        "\"inputSchema\": {\"type\": \"object\", \"properties\": {\"days\": "
        "{\"type\": \"number\", \"description\": \"7 by default, at most "
        "30\"}, \"min_mag\": {\"type\": \"number\", \"description\": \"5 by "
        "default\"}}, \"additionalProperties\": false}, " READS "}, "
        "{\"name\": \"quakes_event\", \"title\": \"An earthquake\", "
        "\"description\": \"One earthquake in detail: by its id (in "
        "brackets in an earlier answer), or the latest near a place; with "
        "neither, the latest near the user (the one just felt).\", \"inputSchema\": {\"type\": \"object\", \"properties\": "
        "{\"id\": {\"type\": \"string\"}, " PLACE_ARG
        "}, \"additionalProperties\": false}, " READS "}]");
}

static const char *arg_str(const struct janas_json *args, const char *name)
{
    const char *s = janas_json_str(janas_json_get(args, name));
    return s && *s ? s : NULL;
}

static double arg_num(const struct janas_json *args, const char *name,
                      double def, double lo, double hi)
{
    const struct janas_json *v = janas_json_get(args, name);
    double x = v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, def) : def;
    return x < lo ? lo : x > hi ? hi : x;
}

static int fail(struct janas_buf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static int fail(struct janas_buf *b, const char *fmt, ...)
{
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    janas_mcps_text_result(b, line, strlen(line), 1);
    return 0;
}

static int answer(struct janas_buf *b, const char *name, struct janas_buf *d,
                  const char *layout, const char *brief)
{
    char err[300];
    int oom = d->oom;
    int r = oom ? -1
                : janas_tpl_result(b, name, d->p, d->n, layout, brief, err,
                                   sizeof err);
    janas_buf_free(d);
    if (r != 0)
        return fail(b, "The answer could not be written: %s.",
                    oom ? "out of memory" : err);
    return 0;
}

/* "&starttime=2026-09-26T08:00:00", days ago, in UTC */
static void since(struct janas_buf *u, int days)
{
    time_t t = time(NULL) - (time_t)days * 86400;
    struct tm g;
    gmtime_r(&t, &g);
    janas_buf_printf(u, "&starttime=%04d-%02d-%02dT%02d:%02d:%02d",
                     g.tm_year + 1900, g.tm_mon + 1, g.tm_mday, g.tm_hour,
                     g.tm_min, g.tm_sec);
}

/* the events near p from src (ingv, usgs) into l; 0, or -1 */
static int near(const char *src, const struct qk_place *p, double radius,
                int days, double min_mag, int limit, struct qk_list *l,
                char *err, size_t err_len)
{
    int ingv = !strcmp(src, "ingv");
    struct janas_buf u = {0}, body = {0};
    janas_buf_puts(&u, ingv ? INGV : USGS);
    since(&u, days);
    if (p->is_sea) { /* its box, then its polygons */
        const struct geo_area *a = &p->sea;
        janas_buf_printf(&u,
                         ingv ? "&minlat=%.3f&maxlat=%.3f&minlon=%.3f"
                                "&maxlon=%.3f&minmag=%.1f"
                              : "&minlatitude=%.3f&maxlatitude=%.3f"
                                "&minlongitude=%.3f&maxlongitude=%.3f"
                                "&minmagnitude=%.1f",
                         a->s_lat_min / 1e5, a->s_lat_max / 1e5,
                         a->s_lon_min / 1e5, a->s_lon_max / 1e5, min_mag);
        l->within = a;
    } else
        janas_buf_printf(&u,
                         ingv ? "&lat=%.4f&lon=%.4f&maxradiuskm=%.0f"
                                "&minmag=%.1f"
                              : "&latitude=%.4f&longitude=%.4f"
                                "&maxradiuskm=%.0f&minmagnitude=%.1f",
                         p->lat, p->lon, radius, min_mag);
    janas_buf_printf(&u, "&orderby=time&limit=%d", limit);
    int st = u.oom ? -1 : qk_fetch(u.p, &body, err, err_len);
    janas_buf_free(&u);
    int ok = (st == 200 || st == 204) &&
             qk_read(st == 204 ? "" : body.p, st == 204 ? 0 : body.n, src, l,
                     err, err_len) == 0;
    if (st > 0 && st != 200 && st != 204)
        snprintf(err, err_len, "%s: HTTP %d", ingv ? "INGV" : "the USGS", st);
    janas_buf_free(&body);
    return ok ? 0 : -1;
}

/* the source for a place (INGV for a point in its area, or a sea that
   reaches into it), and the other */
static const char *first_src(const struct qk_place *p)
{
    if (p->is_sea) {
        const struct geo_area *a = &p->sea;
        return a->s_lat_max / 1e5 >= QK_INGV_LAT0 &&
                       a->s_lat_min / 1e5 <= QK_INGV_LAT1 &&
                       a->s_lon_max / 1e5 >= QK_INGV_LON0 &&
                       a->s_lon_min / 1e5 <= QK_INGV_LON1
                   ? "ingv"
                   : "usgs";
    }
    return qk_in_ingv(p->lat, p->lon) ? "ingv" : "usgs";
}

static const char *other_src(const char *s)
{
    return strcmp(s, "ingv") ? "ingv" : "usgs";
}

static struct qk_list L;

static int t_near(const struct janas_json *args, struct janas_buf *b)
{
    struct qk_place p;
    char err[400] = "";
    if (qk_place_find(arg_str(args, "place"), &p, err, sizeof err) != 0)
        return fail(b, "%s. Ask the user which place.", err);
    double radius = arg_num(args, "radius", 300, 10, 1000);
    int days = (int)arg_num(args, "days", 7, 1, 30);
    double min_mag = arg_num(args, "min_mag", 2.5, 0, 9);
    const char *src = first_src(&p);
    memset(&L, 0, sizeof L);
    if (near(src, &p, radius, days, min_mag, 500, &L, err, sizeof err) != 0) {
        char e2[300];
        memset(&L, 0, sizeof L);
        src = other_src(src);
        if (near(src, &p, radius, days, min_mag, 500, &L, e2, sizeof e2) != 0)
            return fail(b, "The earthquakes could not be read: %s; %s.", err,
                        e2);
    }
    qk_by_time(&L);
    struct janas_buf d = {0};
    qk_near_data(&d, &p, radius, days, min_mag, &L, src);
    return answer(b, "quakes_near", &d, QK_NEAR_LAYOUT, QK_NEAR_BRIEF);
}

static int t_strong(const struct janas_json *args, struct janas_buf *b)
{
    int days = (int)arg_num(args, "days", 7, 1, 30);
    double min_mag = arg_num(args, "min_mag", 5, 3, 9);
    char err[400] = "";
    struct janas_buf u = {0}, body = {0};
    janas_buf_puts(&u, USGS);
    since(&u, days);
    janas_buf_printf(&u, "&minmagnitude=%.1f&orderby=magnitude&limit=200",
                     min_mag);
    int st = u.oom ? -1 : qk_fetch(u.p, &body, err, sizeof err);
    janas_buf_free(&u);
    memset(&L, 0, sizeof L);
    int ok =
        st == 200 && qk_read(body.p, body.n, "usgs", &L, err, sizeof err) == 0;
    if (st > 0 && st != 200)
        snprintf(err, sizeof err, "the USGS: HTTP %d", st);
    janas_buf_free(&body);
    if (!ok)
        return fail(b, "The earthquakes could not be read: %s.", err);
    qk_by_mag(&L);
    struct janas_buf d = {0};
    qk_strong_data(&d, days, min_mag, &L);
    return answer(b, "quakes_strong", &d, QK_STRONG_LAYOUT, QK_STRONG_BRIEF);
}

/* an id of INGV's is a number, the USGS's letters and digits ("us6000tz62") */
static int plain_id(const char *s, int *digits)
{
    size_t n = strlen(s);
    *digits = 1;
    for (size_t i = 0; i < n; i++) {
        if (!isalnum((unsigned char)s[i]))
            return 0;
        if (!isdigit((unsigned char)s[i]))
            *digits = 0;
    }
    return n > 0 && n < 32;
}

static int t_event(const struct janas_json *args, struct janas_buf *b)
{
    char err[400] = "";
    const char *id = arg_str(args, "id");
    const char *q = arg_str(args, "place");
    struct qk_place p;
    int have_place = 0;
    memset(&L, 0, sizeof L);
    if (id) {
        int digits;
        if (!plain_id(id, &digits))
            return fail(b, "\"%s\" is not an earthquake's id.", id);
        struct janas_buf u = {0}, body = {0};
        janas_buf_printf(&u, "%s&eventid=%s", digits ? INGV : USGS, id);
        int st = u.oom ? -1 : qk_fetch(u.p, &body, err, sizeof err);
        janas_buf_free(&u);
        int ok = (st == 200 || st == 204) &&
                 qk_read(st == 204 ? "" : body.p, st == 204 ? 0 : body.n,
                         digits ? "ingv" : "usgs", &L, err, sizeof err) == 0;
        janas_buf_free(&body);
        if (!ok && st > 0 && st != 200 && st != 204)
            snprintf(err, sizeof err, "HTTP %d", st);
        if (!ok)
            return fail(b, "The earthquake %s could not be read: %s.", id, err);
        if (!L.n)
            return fail(b, "No earthquake has the id %s.", id);
        have_place = q && qk_place_find(q, &p, err, sizeof err) == 0;
    } else {
        /* the latest near the place: the one just felt */
        if (qk_place_find(q, &p, err, sizeof err) != 0)
            return fail(b, "%s. Ask the user which place.", err);
        have_place = 1;
        const char *src = first_src(&p);
        if (near(src, &p, 300, 2, 2.5, 1, &L, err, sizeof err) != 0) {
            memset(&L, 0, sizeof L);
            src = other_src(src);
            if (near(src, &p, 300, 2, 2.5, 1, &L, err, sizeof err) != 0)
                return fail(b, "The earthquakes could not be read: %s.", err);
        }
        if (!L.n)
            return fail(b,
                        "No earthquake of magnitude 2.5 or more within 300 "
                        "km of %s in the last two days. Tell the user so.",
                        p.name[0] ? p.name : "the place");
    }
    struct janas_buf d = {0};
    qk_event_data(&d, &L.e[0], have_place ? &p : NULL);
    return answer(b, "quakes_event", &d, QK_EVENT_LAYOUT, QK_EVENT_BRIEF);
}

int qk_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    (void)ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    static const struct {
        const char *name;
        int (*fn)(const struct janas_json *, struct janas_buf *);
    } T[] = {{"quakes_near", t_near},
             {"quakes_strong", t_strong},
             {"quakes_event", t_event}};
    for (size_t i = 0; i < sizeof T / sizeof *T; i++)
        if (janas_json_is(name, T[i].name))
            return T[i].fn(args, b);
    return -1;
}
