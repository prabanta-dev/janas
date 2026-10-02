/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * read.c - janas-maps reads its sources' answers (see maps.h): Nominatim's
 * and Photon's places, Valhalla's and OSRM's routes, Transitous' journeys,
 * Overpass' places around a point. No network here, so that the tests can
 * read answers kept as they came.
 */
#define _GNU_SOURCE /* timegm */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "services/common/geo.h"
#include "maps.h"

static void copy(char *to, size_t cap, const char *s)
{
    snprintf(to, cap, "%s", s ? s : "");
}

static const char *str(const struct janas_json *o, const char *key)
{
    const char *s = janas_json_str(janas_json_get(o, key));
    return s && *s ? s : NULL;
}

/* a number given as a number or as a text (Nominatim's "38.02") */
static double num(const struct janas_json *o, const char *key)
{
    const struct janas_json *v = janas_json_get(o, key);
    if (v && v->type == JANAS_JSON_STRING) {
        char *end;
        double d = strtod(v->s, &end);
        return end > v->s ? d : NAN;
    }
    return janas_json_num(v, NAN);
}

static struct janas_json_doc *parse(const char *text, size_t n, const char *who,
                                    char *err, size_t err_len)
{
    char why[128];
    struct janas_json_doc *d = janas_json_parse(text, n, why, sizeof why);
    if (!d && err)
        snprintf(err, err_len, "%s: an answer it could not read (%s)", who,
                 why);
    return d;
}

time_t mp_utc(const char *s)
{
    struct tm tm = {0};
    if (!s || sscanf(s, "%d-%d-%dT%d:%d:%d", &tm.tm_year, &tm.tm_mon,
                     &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec) < 5)
        return -1;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return timegm(&tm);
}

/* ---- places ---- */

/* "Via Paolo Balsamo 12": the road and the number */
static void street(char *to, size_t cap, const char *road, const char *number)
{
    if (road && number)
        snprintf(to, cap, "%s %s", road, number);
    else
        copy(to, cap, road);
}

/* A town's boundary has its point in the middle of the municipality's
   land: Trapani's in the fields 25 km from the town. The town itself, a
   place of the same name within 40 km among the answers, gives the point
   instead. */
static const struct janas_json *town_of(const struct janas_json *f)
{
    if (!janas_json_is(janas_json_get(f, "category"), "boundary"))
        return f;
    const char *name = str(f, "name");
    for (const struct janas_json *g = f->next; g && name; g = g->next) {
        const char *gn = str(g, "name");
        if (janas_json_is(janas_json_get(g, "category"), "place") && gn &&
            strcmp(gn, name) == 0 &&
            geo_km(num(f, "lat"), num(f, "lon"), num(g, "lat"), num(g, "lon")) <
                40)
            return g;
    }
    return f;
}

int mp_read_nominatim(const char *text, size_t n, struct mp_place *p)
{
    struct janas_json_doc *d = parse(text, n, "Nominatim", NULL, 0);
    const struct janas_json *r = d ? janas_json_root(d) : NULL;
    /* a search answers a list, a reverse lookup one place */
    const struct janas_json *f = !r                             ? NULL
                                 : r->type == JANAS_JSON_ARRAY  ? r->child
                                 : r->type == JANAS_JSON_OBJECT ? r
                                                                : NULL;
    int ok = f && !isnan(num(f, "lat")) && !isnan(num(f, "lon"));
    if (ok) {
        memset(p, 0, sizeof *p);
        const struct janas_json *a = janas_json_get(f, "address");
        const char *city = str(a, "city");
        if (!city)
            city = str(a, "town");
        if (!city)
            city = str(a, "village");
        if (!city)
            city = str(a, "municipality");
        const char *name = str(f, "name");
        if (name)
            copy(p->name, sizeof p->name, name);
        else
            street(p->name, sizeof p->name, str(a, "road"),
                   str(a, "house_number"));
        if (!p->name[0])
            copy(p->name, sizeof p->name, city);
        copy(p->address, sizeof p->address, str(f, "display_name"));
        copy(p->city, sizeof p->city, city);
        const char *cat = str(f, "category"), *type = str(f, "type");
        if (cat && type)
            snprintf(p->kind, sizeof p->kind, "%s=%s", cat, type);
        const char *ot = str(f, "osm_type");
        double id = num(f, "osm_id");
        if (ot && !isnan(id))
            snprintf(p->osm, sizeof p->osm, "%s/%.0f", ot, id);
        const struct janas_json *t = town_of(f);
        p->lat = num(t, "lat");
        p->lon = num(t, "lon");
    }
    janas_json_free(d);
    return ok ? 0 : -1;
}

int mp_read_photon(const char *text, size_t n, struct mp_place *p)
{
    struct janas_json_doc *d = parse(text, n, "Photon", NULL, 0);
    const struct janas_json *fs =
        d ? janas_json_get(janas_json_root(d), "features") : NULL;
    const struct janas_json *f =
        fs && fs->type == JANAS_JSON_ARRAY ? fs->child : NULL;
    const struct janas_json *c =
        janas_json_get(janas_json_get(f, "geometry"), "coordinates");
    const struct janas_json *lon =
        c && c->type == JANAS_JSON_ARRAY ? c->child : NULL;
    const struct janas_json *lat = lon ? lon->next : NULL;
    int ok =
        lat && lat->type == JANAS_JSON_NUMBER && lon->type == JANAS_JSON_NUMBER;
    if (ok) {
        memset(p, 0, sizeof *p);
        const struct janas_json *q = janas_json_get(f, "properties");
        char road[128];
        street(road, sizeof road, str(q, "street"), str(q, "housenumber"));
        const char *name = str(q, "name"), *city = str(q, "city");
        copy(p->name, sizeof p->name, name ? name : road[0] ? road : city);
        copy(p->city, sizeof p->city, city);
        struct janas_buf a = {0};
        const char *parts[] = {name, road[0] ? road : NULL, str(q, "postcode"),
                               city, str(q, "state"),       str(q, "country")};
        for (size_t i = 0; i < sizeof parts / sizeof *parts; i++)
            if (parts[i])
                janas_buf_printf(&a, "%s%s", a.n ? ", " : "", parts[i]);
        copy(p->address, sizeof p->address, a.p);
        janas_buf_free(&a);
        const char *k = str(q, "osm_key"), *v = str(q, "osm_value");
        if (k && v)
            snprintf(p->kind, sizeof p->kind, "%s=%s", k, v);
        const char *ot = str(q, "osm_type");
        double id = num(q, "osm_id");
        if (ot && !isnan(id))
            snprintf(p->osm, sizeof p->osm, "%s/%.0f",
                     *ot == 'N'   ? "node"
                     : *ot == 'W' ? "way"
                                  : "relation",
                     id);
        p->lat = janas_json_num(lat, 0);
        p->lon = janas_json_num(lon, 0);
    }
    janas_json_free(d);
    return ok ? 0 : -1;
}

/* ---- routes ---- */

#define N_ROADS 64

struct roads {
    char name[N_ROADS][64];
    double km[N_ROADS];
    int n;
};

/* A way's name for the roads taken: its own number before the European
   route's (A29, not E 90), when it has both. */
static const char *road_name(const struct janas_json *names)
{
    const char *first = NULL;
    for (const struct janas_json *s =
             names && names->type == JANAS_JSON_ARRAY ? names->child : NULL;
         s; s = s->next) {
        const char *t = janas_json_str(s);
        if (!t || !*t)
            continue;
        if (!first)
            first = t;
        if (!(t[0] == 'E' && t[1] == ' '))
            return t;
    }
    return first;
}

static void road_add(struct roads *r, const char *name, double km)
{
    if (!name)
        return;
    for (int i = 0; i < r->n; i++)
        if (strcmp(r->name[i], name) == 0) {
            r->km[i] += km;
            return;
        }
    if (r->n < N_ROADS) {
        copy(r->name[r->n], sizeof r->name[r->n], name);
        r->km[r->n++] = km;
    }
}

/* the three longest, in the order they are driven */
static void roads_line(const struct roads *r, char *out, size_t cap)
{
    int pick[3], np = 0;
    for (int k = 0; k < 3; k++) {
        int best = -1;
        for (int i = 0; i < r->n; i++) {
            int taken = 0;
            for (int j = 0; j < np; j++)
                taken |= pick[j] == i;
            if (!taken && r->km[i] >= 1 && (best < 0 || r->km[i] > r->km[best]))
                best = i;
        }
        if (best < 0)
            break;
        pick[np++] = best;
    }
    for (int i = 0; i < np; i++) /* by the order of first meeting */
        for (int j = i + 1; j < np; j++)
            if (pick[j] < pick[i]) {
                int t = pick[i];
                pick[i] = pick[j];
                pick[j] = t;
            }
    out[0] = 0;
    for (int i = 0; i < np; i++) {
        size_t n = strlen(out);
        snprintf(out + n, cap - n, "%s%s", i ? ", " : "", r->name[pick[i]]);
    }
}

static int valhalla_trip(const struct janas_json *trip, struct mp_route *r,
                         int steps)
{
    const struct janas_json *sum = janas_json_get(trip, "summary");
    r->km = janas_json_num(janas_json_get(sum, "length"), NAN);
    r->min = janas_json_num(janas_json_get(sum, "time"), NAN) / 60;
    if (isnan(r->km) || isnan(r->min))
        return -1;
    r->toll = janas_json_get(sum, "has_toll") &&
              janas_json_get(sum, "has_toll")->type == JANAS_JSON_TRUE;
    r->highway = janas_json_get(sum, "has_highway") &&
                 janas_json_get(sum, "has_highway")->type == JANAS_JSON_TRUE;
    r->ferry = janas_json_get(sum, "has_ferry") &&
               janas_json_get(sum, "has_ferry")->type == JANAS_JSON_TRUE;
    struct roads *rd = calloc(1, sizeof *rd);
    if (steps)
        r->step = calloc(MP_STEPS, sizeof *r->step);
    const struct janas_json *legs = janas_json_get(trip, "legs");
    for (const struct janas_json *l =
             legs && legs->type == JANAS_JSON_ARRAY ? legs->child : NULL;
         l; l = l->next) {
        const struct janas_json *ms = janas_json_get(l, "maneuvers");
        for (const struct janas_json *m =
                 ms && ms->type == JANAS_JSON_ARRAY ? ms->child : NULL;
             m; m = m->next) {
            double km = janas_json_num(janas_json_get(m, "length"), 0);
            if (rd)
                road_add(rd, road_name(janas_json_get(m, "street_names")), km);
            const char *t = str(m, "instruction");
            if (!t || !r->step)
                continue;
            if (r->n_step < MP_STEPS) {
                copy(r->step[r->n_step].text, sizeof r->step[0].text, t);
                r->step[r->n_step++].km = km;
            } else {
                r->cut++;
            }
        }
    }
    if (rd)
        roads_line(rd, r->roads, sizeof r->roads);
    free(rd);
    return 0;
}

int mp_read_valhalla(const char *text, size_t n, struct mp_routes *rs,
                     char *err, size_t err_len)
{
    memset(rs, 0, sizeof *rs);
    rs->source = "Valhalla";
    struct janas_json_doc *d = parse(text, n, "Valhalla", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *root = janas_json_root(d);
    const struct janas_json *trip = janas_json_get(root, "trip");
    int rc = -1;
    if (!trip) {
        const char *e = str(root, "error");
        snprintf(err, err_len, "Valhalla: %s", e ? e : "no route");
    } else if (valhalla_trip(trip, &rs->r[0], 1) != 0) {
        snprintf(err, err_len, "Valhalla: a route without its length");
    } else {
        rs->n = 1;
        const struct janas_json *alt = janas_json_get(root, "alternates");
        for (const struct janas_json *a =
                 alt && alt->type == JANAS_JSON_ARRAY ? alt->child : NULL;
             a && rs->n < MP_ROUTES; a = a->next)
            if (valhalla_trip(janas_json_get(a, "trip"), &rs->r[rs->n], 0) == 0)
                rs->n++;
        rc = 0;
    }
    janas_json_free(d);
    return rc;
}

int mp_read_osrm(const char *text, size_t n, struct mp_routes *rs, char *err,
                 size_t err_len)
{
    memset(rs, 0, sizeof *rs);
    rs->source = "OSRM";
    struct janas_json_doc *d = parse(text, n, "OSRM", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *root = janas_json_root(d);
    const struct janas_json *routes = janas_json_get(root, "routes");
    if (!janas_json_is(janas_json_get(root, "code"), "Ok") || !routes ||
        routes->type != JANAS_JSON_ARRAY) {
        const char *m = str(root, "message");
        snprintf(err, err_len, "OSRM: %s", m ? m : "no route");
        janas_json_free(d);
        return -1;
    }
    /* a place far from any road it knows (its map is Europe's): OSRM
       takes the nearest road all the same: New York's, 5,535 km away */
    const struct janas_json *wps = janas_json_get(root, "waypoints");
    for (const struct janas_json *w =
             wps && wps->type == JANAS_JSON_ARRAY ? wps->child : NULL;
         w; w = w->next) {
        double m = janas_json_num(janas_json_get(w, "distance"), 0);
        if (m > 5000) {
            snprintf(err, err_len,
                     "OSRM: a place is %.0f km from the nearest road it "
                     "knows (its map covers Europe)",
                     m / 1000);
            janas_json_free(d);
            return -1;
        }
    }
    for (const struct janas_json *r = routes->child; r && rs->n < MP_ROUTES;
         r = r->next) {
        struct mp_route *o = &rs->r[rs->n];
        o->km = janas_json_num(janas_json_get(r, "distance"), NAN) / 1000;
        o->min = janas_json_num(janas_json_get(r, "duration"), NAN) / 60;
        if (isnan(o->km) || isnan(o->min))
            continue;
        const struct janas_json *legs = janas_json_get(r, "legs");
        for (const struct janas_json *l =
                 legs && legs->type == JANAS_JSON_ARRAY ? legs->child : NULL;
             l; l = l->next) {
            const char *s = str(l, "summary");
            size_t k = strlen(o->roads);
            if (s && k + 3 < sizeof o->roads)
                snprintf(o->roads + k, sizeof o->roads - k, "%s%s",
                         k ? ", " : "", s);
        }
        rs->n++;
    }
    janas_json_free(d);
    if (!rs->n) {
        snprintf(err, err_len, "OSRM: no route");
        return -1;
    }
    return 0;
}

void mp_routes_free(struct mp_routes *rs)
{
    for (int i = 0; i < MP_ROUTES; i++) {
        free(rs->r[i].step);
        rs->r[i].step = NULL;
    }
}

/* ---- public transport ---- */

static void leg_place(char *to, size_t cap, const struct janas_json *p)
{
    copy(to, cap, str(p, "name"));
}

int mp_read_transit(const char *text, size_t n, struct mp_transit *tr,
                    char *err, size_t err_len)
{
    memset(tr, 0, sizeof *tr);
    struct janas_json_doc *d = parse(text, n, "Transitous", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *root = janas_json_root(d);
    const struct janas_json *its = janas_json_get(root, "itineraries");
    for (const struct janas_json *it =
             its && its->type == JANAS_JSON_ARRAY ? its->child : NULL;
         it && tr->n < MP_TRIPS; it = it->next) {
        struct mp_trip *t = &tr->t[tr->n];
        memset(t, 0, sizeof *t);
        t->dep = mp_utc(str(it, "startTime"));
        t->arr = mp_utc(str(it, "endTime"));
        t->transfers = (int)janas_json_num(janas_json_get(it, "transfers"), 0);
        if (t->dep < 0 || t->arr < 0)
            continue;
        const struct janas_json *legs = janas_json_get(it, "legs");
        for (const struct janas_json *l =
                 legs && legs->type == JANAS_JSON_ARRAY ? legs->child : NULL;
             l; l = l->next) {
            if (t->n_leg == MP_LEGS) {
                t->cut++;
                continue;
            }
            struct mp_leg *g = &t->leg[t->n_leg];
            copy(g->mode, sizeof g->mode, str(l, "mode"));
            const char *line = str(l, "displayName");
            copy(g->line, sizeof g->line,
                 line ? line : str(l, "routeShortName"));
            copy(g->agency, sizeof g->agency, str(l, "agencyName"));
            copy(g->headsign, sizeof g->headsign, str(l, "headsign"));
            leg_place(g->from, sizeof g->from, janas_json_get(l, "from"));
            leg_place(g->to, sizeof g->to, janas_json_get(l, "to"));
            g->dep = mp_utc(str(l, "startTime"));
            g->arr = mp_utc(str(l, "endTime"));
            g->walk_m = janas_json_num(janas_json_get(l, "distance"), NAN);
            const struct janas_json *rt = janas_json_get(l, "realTime");
            g->real_time = rt && rt->type == JANAS_JSON_TRUE;
            if (!tr->tz[0])
                copy(tr->tz, sizeof tr->tz,
                     str(janas_json_get(l, "from"), "tz"));
            if (g->mode[0])
                t->n_leg++;
        }
        tr->n++;
    }
    int ok = its && its->type == JANAS_JSON_ARRAY;
    janas_json_free(d);
    if (!ok) {
        snprintf(err, err_len, "Transitous: an answer without journeys");
        return -1;
    }
    return 0;
}

/* ---- what is near ---- */

static int by_distance(const void *a, const void *b)
{
    double x = ((const struct mp_poi *)a)->m, y = ((const struct mp_poi *)b)->m;
    return (x > y) - (x < y);
}

int mp_read_overpass(const char *text, size_t n, double lat, double lon,
                     int max, struct mp_near *nr, char *err, size_t err_len)
{
    memset(nr, 0, sizeof *nr);
    /* too busy, it answers a page of HTML, status 200 at times */
    if (n && text[0] == '<') {
        snprintf(err, err_len, "Overpass: %s",
                 strstr(text, "too busy") || strstr(text, "timeout")
                     ? "too busy now"
                     : "an error page instead of data");
        return -1;
    }
    struct janas_json_doc *d = parse(text, n, "Overpass", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *els =
        janas_json_get(janas_json_root(d), "elements");
    if (!els || els->type != JANAS_JSON_ARRAY) {
        const char *r = str(janas_json_root(d), "remark");
        snprintf(err, err_len, "Overpass: %s", r ? r : "no elements");
        janas_json_free(d);
        return -1;
    }
    size_t cap = 64, have = 0;
    struct mp_poi *all = malloc(cap * sizeof *all);
    for (const struct janas_json *e = els->child; e && all; e = e->next) {
        const struct janas_json *c = janas_json_get(e, "center");
        double y = num(c ? c : e, "lat"), x = num(c ? c : e, "lon");
        if (isnan(y) || isnan(x))
            continue;
        if (have == cap) {
            struct mp_poi *g = realloc(all, 2 * cap * sizeof *all);
            if (!g)
                break;
            all = g;
            cap *= 2;
        }
        struct mp_poi *p = &all[have++];
        memset(p, 0, sizeof *p);
        const struct janas_json *t = janas_json_get(e, "tags");
        const char *name = str(t, "name");
        copy(p->name, sizeof p->name, name ? name : str(t, "brand"));
        char road[128];
        street(road, sizeof road, str(t, "addr:street"),
               str(t, "addr:housenumber"));
        const char *city = str(t, "addr:city");
        if (road[0] && city)
            snprintf(p->address, sizeof p->address, "%s, %s", road, city);
        else
            copy(p->address, sizeof p->address, road);
        copy(p->hours, sizeof p->hours, str(t, "opening_hours"));
        const char *ph = str(t, "phone");
        copy(p->phone, sizeof p->phone, ph ? ph : str(t, "contact:phone"));
        const char *type = str(e, "type");
        double id = num(e, "id");
        if (type && !isnan(id))
            snprintf(p->osm, sizeof p->osm, "%s/%.0f", type, id);
        p->lat = y;
        p->lon = x;
        p->m = geo_km(lat, lon, y, x) * 1000;
    }
    if (all)
        qsort(all, have, sizeof *all, by_distance);
    nr->found = (int)have;
    for (size_t i = 0; all && i < have && nr->n < max && nr->n < MP_NEAR; i++)
        nr->p[nr->n++] = all[i];
    free(all);
    janas_json_free(d);
    return 0;
}
