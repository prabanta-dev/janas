/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * read.c - janas-quakes reads its sources' answers (see quakes.h): the FDSN
 * event service's GeoJSON, as INGV and the USGS give it. INGV: eventId,
 * time in UTC as text, mag, magType, place, type. The USGS: id, time in
 * milliseconds, mag, magType, place, type, felt, mmi, alert, tsunami,
 * status, url. Both: the point as [lon, lat, depth in km]. No network
 * here: the tests give it kept answers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "quakes.h"

static void copy(char *to, size_t cap, const char *s)
{
    size_t n = s ? strlen(s) : 0;
    if (n >= cap)
        n = cap - 1;
    memcpy(to, s ? s : "", n);
    to[n] = 0;
}

/* days from 1970-01-01 to y-m-d (proleptic Gregorian) */
static long long days_from_civil(int y, int m, int d)
{
    y -= m <= 2;
    long long era = (y >= 0 ? y : y - 399) / 400;
    long long yoe = y - era * 400;
    long long doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    long long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

long long qk_utc(const char *s)
{
    int y, mo, d, h, mi;
    double sec;
    if (!s || sscanf(s, "%d-%d-%dT%d:%d:%lf", &y, &mo, &d, &h, &mi, &sec) != 6)
        return -1;
    return days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 +
           (long long)sec;
}

static double num(const struct janas_json *o, const char *k, double def)
{
    const struct janas_json *v = janas_json_get(o, k);
    return v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, def) : def;
}

int qk_read(const char *text, size_t n, const char *src, struct qk_list *l,
            char *err, size_t err_len)
{
    if (!n) /* INGV's 204: none */
        return 0;
    char why[200];
    struct janas_json_doc *d = janas_json_parse(text, n, why, sizeof why);
    if (!d) {
        snprintf(err, err_len, "the answer of %s is not JSON: %s", src, why);
        return -1;
    }
    /* a list of events, or one (the USGS's answer for an event's id) */
    const struct janas_json *r = janas_json_root(d);
    const struct janas_json *f = janas_json_get(r, "features");
    const struct janas_json *first =
        f && f->type == JANAS_JSON_ARRAY ? f->child : NULL;
    int single = !f && janas_json_is(janas_json_get(r, "type"), "Feature");
    if (!f && !single) {
        janas_json_free(d);
        snprintf(err, err_len, "the answer of %s has no events", src);
        return -1;
    }
    if (single)
        first = r;
    int usgs = !strcmp(src, "usgs");
    for (const struct janas_json *x = first; x; x = single ? NULL : x->next) {
        const struct janas_json *pr = janas_json_get(x, "properties");
        const struct janas_json *c =
            janas_json_get(janas_json_get(x, "geometry"), "coordinates");
        if (!pr || !c || c->type != JANAS_JSON_ARRAY || !c->child ||
            !c->child->next)
            continue;
        if (l->n == QK_MAX) { /* counted, if within the sea asked */
            double la = janas_json_num(c->child->next, 0),
                   lo = janas_json_num(c->child, 0);
            l->more += !l->within || geo_area_has(l->within, la, lo);
            continue;
        }
        struct qk_event *e = &l->e[l->n];
        memset(e, 0, sizeof *e);
        copy(e->src, sizeof e->src, src);
        e->lon = janas_json_num(c->child, 0);
        e->lat = janas_json_num(c->child->next, 0);
        e->depth_km =
            c->child->next->next ? janas_json_num(c->child->next->next, 0) : 0;
        e->mag = num(pr, "mag", -9);
        copy(e->magtype, sizeof e->magtype,
             janas_json_str(janas_json_get(pr, "magType")));
        copy(e->place, sizeof e->place,
             janas_json_str(janas_json_get(pr, "place")));
        copy(e->kind, sizeof e->kind,
             janas_json_str(janas_json_get(pr, "type")));
        e->felt = -1;
        e->mmi = -1;
        if (usgs) {
            copy(e->id, sizeof e->id, janas_json_str(janas_json_get(x, "id")));
            e->at = (long long)(num(pr, "time", -1000) / 1000);
            e->felt = (int)num(pr, "felt", -1);
            e->mmi = num(pr, "mmi", -1);
            copy(e->alert, sizeof e->alert,
                 janas_json_str(janas_json_get(pr, "alert")));
            e->tsunami = num(pr, "tsunami", 0) != 0;
            e->reviewed =
                janas_json_is(janas_json_get(pr, "status"), "reviewed");
            copy(e->url, sizeof e->url,
                 janas_json_str(janas_json_get(pr, "url")));
        } else {
            long long id = (long long)num(pr, "eventId", 0);
            snprintf(e->id, sizeof e->id, "%lld", id);
            e->at = qk_utc(janas_json_str(janas_json_get(pr, "time")));
            snprintf(e->url, sizeof e->url,
                     "https://terremoti.ingv.it/event/%lld", id);
            e->reviewed = 0; /* INGV's automatic locations are told so */
        }
        if (e->mag > -9 && e->at > 0 &&
            (!l->within || geo_area_has(l->within, e->lat, e->lon)))
            l->n++;
    }
    janas_json_free(d);
    return 0;
}

static int later(const void *a, const void *b)
{
    long long x = ((const struct qk_event *)a)->at,
              y = ((const struct qk_event *)b)->at;
    return x < y ? 1 : x > y ? -1 : 0;
}

static int stronger(const void *a, const void *b)
{
    double x = ((const struct qk_event *)a)->mag,
           y = ((const struct qk_event *)b)->mag;
    return x < y ? 1 : x > y ? -1 : later(a, b);
}

void qk_by_time(struct qk_list *l)
{
    qsort(l->e, (size_t)l->n, sizeof *l->e, later);
}

void qk_by_mag(struct qk_list *l)
{
    qsort(l->e, (size_t)l->n, sizeof *l->e, stronger);
}
