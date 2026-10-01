/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * alerts.c - warnings (see weather.h). MeteoAlarm gathers the warnings of
 * Europe's weather services, by country and region: wind, thunderstorms,
 * heat, by level (green, yellow, orange, red); its feed holds warnings
 * already expired, which are left out, and green ones, which say there is
 * nothing to be warned of. For Italy, where those are the Air Force's and
 * the official alerts are the Civil Protection's, the national bulletin of
 * criticality is read too (CC BY 4.0, on GitHub, once a day about 16:00):
 * the zone of alert a municipality is in, and its levels for floods,
 * landslides and thunderstorms, today and tomorrow.
 */
#define _GNU_SOURCE /* strcasestr */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "weather.h"

#define KEEP_ALARM 900   /* MeteoAlarm's feed */
#define KEEP_LATEST 1800 /* which bulletin is the latest */
#define KEEP_DPC 3600    /* a bulletin */
#define DPC_REPO "pcm-dpc/DPC-Bollettini-Criticita-Idrogeologica-Idraulica"

/* MeteoAlarm's feeds, by ISO country code */
static const struct {
    const char cc[3], *feed;
} FEEDS[] = {
    {"AT", "austria"},
    {"BA", "bosnia-herzegovina"},
    {"BE", "belgium"},
    {"BG", "bulgaria"},
    {"CH", "switzerland"},
    {"CY", "cyprus"},
    {"CZ", "czechia"},
    {"DE", "germany"},
    {"DK", "denmark"},
    {"EE", "estonia"},
    {"ES", "spain"},
    {"FI", "finland"},
    {"FR", "france"},
    {"GB", "united-kingdom"},
    {"GR", "greece"},
    {"HR", "croatia"},
    {"HU", "hungary"},
    {"IE", "ireland"},
    {"IL", "israel"},
    {"IS", "iceland"},
    {"IT", "italy"},
    {"LT", "lithuania"},
    {"LU", "luxembourg"},
    {"LV", "latvia"},
    {"MD", "moldova"},
    {"ME", "montenegro"},
    {"MK", "republic-of-north-macedonia"},
    {"MT", "malta"},
    {"NL", "netherlands"},
    {"NO", "norway"},
    {"PL", "poland"},
    {"PT", "portugal"},
    {"RO", "romania"},
    {"RS", "serbia"},
    {"SE", "sweden"},
    {"SI", "slovenia"},
    {"SK", "slovakia"},
    {"UA", "ukraine"},
};

static void copy(char *to, size_t cap, const char *s)
{
    snprintf(to, cap, "%s", s ? s : "");
}

/* "2026-10-01T23:59:00+02:00" */
time_t wx_iso_time(const char *s)
{
    struct tm tm = {0};
    int oh = 0, om = 0;
    char sign = '+';
    if (!s || sscanf(s, "%d-%d-%dT%d:%d:%d%c%d:%d", &tm.tm_year, &tm.tm_mon,
                     &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec, &sign,
                     &oh, &om) < 6)
        return 0;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    time_t t = timegm(&tm);
    int off = (oh * 60 + om) * 60;
    return sign == '-' ? t + off : t - off;
}

/* the value of a parameter: "awareness_level" -> "2; yellow; Moderate" */
static const char *param(const struct janas_json *info, const char *name)
{
    const struct janas_json *ps = janas_json_get(info, "parameter");
    for (const struct janas_json *p =
             ps && ps->type == JANAS_JSON_ARRAY ? ps->child : NULL;
         p; p = p->next)
        if (janas_json_is(janas_json_get(p, "valueName"), name))
            return janas_json_str(janas_json_get(p, "value"));
    return NULL;
}

static int area_is(const struct janas_json *info, const char *region,
                   char *desc, size_t cap)
{
    const struct janas_json *as = janas_json_get(info, "area");
    int mine = 0;
    desc[0] = 0;
    for (const struct janas_json *a =
             as && as->type == JANAS_JSON_ARRAY ? as->child : NULL;
         a; a = a->next) {
        const char *d = janas_json_str(janas_json_get(a, "areaDesc"));
        if (!d)
            continue;
        if (!desc[0])
            copy(desc, cap, d);
        if (region && *region &&
            (strcasestr(d, region) || strcasestr(region, d))) {
            mine = 1;
            copy(desc, cap, d);
        }
    }
    return mine;
}

int wx_meteoalarm_parse(const char *text, size_t len, const char *region,
                        const char *lang, time_t now, struct wx_alert **v,
                        size_t *n, char *err, size_t err_len)
{
    *v = NULL;
    *n = 0;
    struct janas_buf b = {.p = (char *)text, .n = len};
    struct janas_json_doc *d = wx_parse(&b, "MeteoAlarm", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *ws =
        janas_json_get(janas_json_root(d), "warnings");
    size_t cap = ws && ws->type == JANAS_JSON_ARRAY ? ws->n : 0;
    struct wx_alert *out = cap ? calloc(cap, sizeof *out) : NULL;
    size_t got = 0;
    for (const struct janas_json *w = cap ? ws->child : NULL; w && out;
         w = w->next) {
        const struct janas_json *infos =
            janas_json_get(janas_json_get(w, "alert"), "info");
        /* the info in the language asked, else in English, else the
           first */
        const struct janas_json *pick = NULL, *en = NULL, *first = NULL;
        for (const struct janas_json *i =
                 infos && infos->type == JANAS_JSON_ARRAY ? infos->child : NULL;
             i; i = i->next) {
            const char *l = janas_json_str(janas_json_get(i, "language"));
            if (!first)
                first = i;
            if (l && lang && strncasecmp(l, lang, strlen(lang)) == 0)
                pick = i;
            if (l && strncasecmp(l, "en", 2) == 0)
                en = i;
        }
        const struct janas_json *i = pick ? pick : en ? en : first;
        if (!i)
            continue;
        const char *exp = janas_json_str(janas_json_get(i, "expires"));
        if (wx_iso_time(exp) <= now)
            continue;                                 /* over */
        const char *lv = param(i, "awareness_level"); /* "2; yellow; ..." */
        const char *col = lv ? strchr(lv, ';') : NULL;
        char level[16] = "";
        if (col)
            sscanf(col + 1, " %15[a-z]", level);
        if (!level[0] || strcmp(level, "green") == 0)
            continue; /* nothing to be warned of */
        if (got == cap)
            break;
        struct wx_alert *a = &out[got];
        copy(a->level, sizeof a->level, level);
        copy(a->event, sizeof a->event,
             janas_json_str(janas_json_get(i, "event")));
        a->mine = area_is(i, region, a->area, sizeof a->area);
        copy(a->onset, sizeof a->onset,
             janas_json_str(janas_json_get(i, "onset")));
        copy(a->expires, sizeof a->expires, exp);
        copy(a->sender, sizeof a->sender,
             janas_json_str(janas_json_get(i, "senderName")));
        copy(a->text, sizeof a->text,
             janas_json_str(janas_json_get(i, "description")));
        /* the same warning given again, for the same area and times */
        int dup = 0;
        for (size_t k = 0; k < got && !dup; k++)
            dup = strcmp(out[k].event, a->event) == 0 &&
                  strcmp(out[k].area, a->area) == 0 &&
                  strcmp(out[k].onset, a->onset) == 0 &&
                  strcmp(out[k].expires, a->expires) == 0;
        if (!dup)
            got++;
    }
    janas_json_free(d);
    if (cap && !out) {
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    /* the place's region first */
    for (size_t i = 0, j = 0; i < got; i++)
        if (out[i].mine) {
            struct wx_alert t = out[i];
            memmove(&out[j + 1], &out[j], (i - j) * sizeof *out);
            out[j++] = t;
        }
    *v = out;
    *n = got;
    return 0;
}

int wx_meteoalarm(const struct wx_place *p, const char *lang,
                  struct wx_alert **v, size_t *n, char *err, size_t err_len)
{
    *v = NULL;
    *n = 0;
    const char *feed = NULL;
    for (size_t i = 0; i < sizeof FEEDS / sizeof FEEDS[0]; i++)
        if (strcasecmp(FEEDS[i].cc, p->cc) == 0)
            feed = FEEDS[i].feed;
    if (!feed) {
        snprintf(err, err_len, "MeteoAlarm has no warnings for %s",
                 p->country[0] ? p->country : "that country");
        return 1;
    }
    char url[160];
    snprintf(url, sizeof url,
             "https://feeds.meteoalarm.org/api/v1/warnings/feeds-%s", feed);
    struct janas_buf b = {0};
    char why[256];
    int status = wx_fetch(url, KEEP_ALARM, 0, &b, why, sizeof why);
    int r = -1;
    if (status == 200)
        r = wx_meteoalarm_parse(b.p, b.n, p->region, lang, time(NULL), v, n,
                                err, err_len);
    else if (status < 0)
        snprintf(err, err_len, "MeteoAlarm did not answer: %s", why);
    else
        snprintf(err, err_len, "MeteoAlarm: HTTP %d", status);
    janas_buf_free(&b);
    return r;
}

/* ---- the Civil Protection's bulletin ---- */

static int rank(const char *level)
{
    if (!level)
        return -1;
    if (strcasestr(level, "ROSSA"))
        return 3;
    if (strcasestr(level, "ARANCIONE"))
        return 2;
    if (strcasestr(level, "GIALLA"))
        return 1;
    return 0;
}

/* Whether the point is inside a ring of [lon, lat] pairs: the edges a
   ray to the east crosses, odd. */
static int ring_has(const struct janas_json *ring, double lat, double lon)
{
    int in = 0;
    const struct janas_json *a = NULL, *first = NULL;
    for (const struct janas_json *b =
             ring && ring->type == JANAS_JSON_ARRAY ? ring->child : NULL;
         ; b = b->next) {
        const struct janas_json *q = b ? b : first; /* closed at the end */
        if (!q || !q->child || !q->child->next)
            break;
        if (!first)
            first = q;
        if (a) {
            double x1 = janas_json_num(a->child, NAN),
                   y1 = janas_json_num(a->child->next, NAN),
                   x2 = janas_json_num(q->child, NAN),
                   y2 = janas_json_num(q->child->next, NAN);
            if ((y1 > lat) != (y2 > lat) &&
                lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1))
                in = !in;
        }
        a = q;
        if (!b)
            break;
    }
    return in;
}

/* Whether the point is inside a GeoJSON Polygon or MultiPolygon: inside an
   odd number of its rings (holes are rings too). */
static int geometry_has(const struct janas_json *g, double lat, double lon)
{
    const struct janas_json *c = janas_json_get(g, "coordinates");
    int multi = janas_json_is(janas_json_get(g, "type"), "MultiPolygon");
    if (!multi && !janas_json_is(janas_json_get(g, "type"), "Polygon"))
        return 0;
    int in = 0;
    for (const struct janas_json *p =
             c && c->type == JANAS_JSON_ARRAY ? c->child : NULL;
         p; p = p->next) {
        const struct janas_json *rings = multi ? p->child : p;
        if (!multi) { /* p is a ring itself */
            in ^= ring_has(p, lat, lon);
            continue;
        }
        for (const struct janas_json *r = rings; r; r = r->next)
            in ^= ring_has(r, lat, lon);
    }
    return in;
}

/* the zone's levels, the highest of each risk, and its name, into out */
static void dpc_take(const struct janas_json *pr, char out[3][96], char *zones,
                     size_t zones_len, size_t *zn)
{
    static const char *const keys[3] = {"Per rischio idraulico",
                                        "Per rischio idrogeologico",
                                        "Per rischio temporali"};
    const char *zone = janas_json_str(janas_json_get(pr, "Nome zona"));
    if (zone && *zn < zones_len)
        *zn += (size_t)snprintf(zones + *zn, zones_len - *zn, "%s%s",
                                *zn ? "; " : "", zone);
    for (int k = 0; k < 3; k++) {
        const char *lv = janas_json_str(janas_json_get(pr, keys[k]));
        if (lv && rank(lv) > rank(out[k][0] ? out[k] : NULL))
            copy(out[k], 96, lv);
    }
}

int wx_dpc_parse(const char *text, size_t len, const char *comune, double lat,
                 double lon, char out[3][96], char *zones, size_t zones_len,
                 char *err, size_t err_len)
{
    struct janas_buf b = {.p = (char *)text, .n = len};
    struct janas_json_doc *d =
        wx_parse(&b, "the Civil Protection's bulletin", err, err_len);
    if (!d)
        return -1;
    const struct janas_json *fs =
        janas_json_get(janas_json_root(d), "features");
    int found = 0;
    size_t zn = 0;
    zones[0] = 0;
    for (int k = 0; k < 3; k++)
        out[k][0] = 0;
    for (const struct janas_json *f =
             fs && fs->type == JANAS_JSON_ARRAY ? fs->child : NULL;
         f; f = f->next) {
        const struct janas_json *pr = janas_json_get(f, "properties");
        const struct janas_json *cs = janas_json_get(pr, "Comuni");
        int in = 0;
        for (const struct janas_json *c =
                 cs && cs->type == JANAS_JSON_ARRAY ? cs->child : NULL;
             c && !in; c = c->next)
            in =
                janas_json_str(c) && strcasecmp(janas_json_str(c), comune) == 0;
        if (!in)
            continue;
        found = 1;
        dpc_take(pr, out, zones, zones_len, &zn);
    }
    /* a name the bulletin does not have (Turin, an airport's, the
       internet connection's town): the zone the point is in */
    for (const struct janas_json *f = !found && !isnan(lat) && !isnan(lon) &&
                                              fs && fs->type == JANAS_JSON_ARRAY
                                          ? fs->child
                                          : NULL;
         f; f = f->next)
        if (geometry_has(janas_json_get(f, "geometry"), lat, lon)) {
            found = 1;
            dpc_take(janas_json_get(f, "properties"), out, zones, zones_len,
                     &zn);
        }
    janas_json_free(d);
    return found;
}

/* The latest bulletin's stamp ("20260930_1423"), from the last commit
   that touched its files. 0, or -1. */
static int latest(char stamp[16], char *err, size_t err_len)
{
    struct janas_buf b = {0};
    char why[256];
    const char *url = "https://api.github.com/repos/" DPC_REPO
                      "/commits?path=files/geojson&per_page=1";
    int status = wx_fetch(url, KEEP_LATEST, 0, &b, why, sizeof why);
    char sha[48] = "";
    if (status == 200) {
        struct janas_json_doc *d = wx_parse(&b, "GitHub", why, sizeof why);
        const struct janas_json *r = d ? janas_json_root(d) : NULL;
        if (r && r->type == JANAS_JSON_ARRAY && r->child)
            copy(sha, sizeof sha,
                 janas_json_str(janas_json_get(r->child, "sha")));
        janas_json_free(d);
    }
    if (!sha[0]) {
        snprintf(err, err_len,
                 "GitHub did not say which bulletin is the "
                 "latest%s%s",
                 status < 0 ? ": " : "", status < 0 ? why : "");
        janas_buf_free(&b);
        return -1;
    }
    char url2[256];
    snprintf(url2, sizeof url2,
             "https://api.github.com/repos/" DPC_REPO "/commits/%s", sha);
    status = wx_fetch(url2, 7 * 24 * 3600, 0, &b, why, sizeof why);
    stamp[0] = 0;
    if (status == 200) {
        struct janas_json_doc *d = wx_parse(&b, "GitHub", why, sizeof why);
        const struct janas_json *fl =
            d ? janas_json_get(janas_json_root(d), "files") : NULL;
        for (const struct janas_json *f =
                 fl && fl->type == JANAS_JSON_ARRAY ? fl->child : NULL;
             f && !stamp[0]; f = f->next) {
            const char *name = janas_json_str(janas_json_get(f, "filename"));
            const char *g = name ? strstr(name, "files/geojson/") : NULL;
            if (g)
                sscanf(g + 14, "%13[0-9_]", stamp);
        }
        janas_json_free(d);
    }
    janas_buf_free(&b);
    if (strlen(stamp) != 13) {
        snprintf(err, err_len, "the latest bulletin could not be found");
        return -1;
    }
    return 0;
}

static int read_day(const char *stamp, const char *which,
                    const struct wx_place *p, char out[3][96], char *zones,
                    size_t zones_len, char *err, size_t err_len)
{
    char url[256];
    snprintf(url, sizeof url,
             "https://raw.githubusercontent.com/" DPC_REPO
             "/master/files/geojson/%s_%s.json",
             stamp, which);
    struct janas_buf b = {0};
    char why[256];
    int status = wx_fetch(url, KEEP_DPC, 0, &b, why, sizeof why);
    int r = -1;
    if (status == 200)
        r = wx_dpc_parse(b.p, b.n, p->name, p->lat, p->lon, out, zones,
                         zones_len, err, err_len);
    else if (status == 404)
        r = 0;
    else if (status < 0)
        snprintf(err, err_len, "GitHub did not answer: %s", why);
    else
        snprintf(err, err_len, "GitHub: HTTP %d", status);
    janas_buf_free(&b);
    return r;
}

int wx_dpc(const struct wx_place *p, struct wx_dpc *d, char *err,
           size_t err_len)
{
    memset(d, 0, sizeof *d);
    if (strcasecmp(p->cc, "IT") != 0)
        return 0;
    if (latest(d->issued, err, err_len) != 0)
        return -1;
    /* the bulletin's day, and today in Italy */
    char bday[11];
    snprintf(bday, sizeof bday, "%.4s-%.2s-%.2s", d->issued, d->issued + 4,
             d->issued + 6);
    char *old = getenv("TZ") ? strdup(getenv("TZ")) : NULL;
    setenv("TZ", "Europe/Rome", 1);
    tzset();
    time_t now = time(NULL), tom_t = now + 24 * 3600;
    char today[11], tomorrow[11];
    struct tm lt;
    strftime(today, sizeof today, "%Y-%m-%d", localtime_r(&now, &lt));
    strftime(tomorrow, sizeof tomorrow, "%Y-%m-%d", localtime_r(&tom_t, &lt));
    struct tm bt = {0};
    sscanf(bday, "%d-%d-%d", &bt.tm_year, &bt.tm_mon, &bt.tm_mday);
    bt.tm_year -= 1900;
    bt.tm_mon -= 1;
    bt.tm_hour = 12;
    bt.tm_isdst = -1;
    time_t bnext_t = mktime(&bt) + 24 * 3600;
    char bnext[11];
    strftime(bnext, sizeof bnext, "%Y-%m-%d", localtime_r(&bnext_t, &lt));
    if (old) {
        setenv("TZ", old, 1);
        free(old);
    } else {
        unsetenv("TZ");
    }
    tzset();
    /* the bulletin's today and tomorrow, as far as they are still ahead:
       yesterday's bulletin's tomorrow is today */
    static const char *const which[2] = {"today", "tomorrow"};
    const char *date[2] = {bday, bnext};
    int found = 0, k = 0;
    for (int i = 0; i < 2; i++) {
        if (strcmp(date[i], today) != 0 && strcmp(date[i], tomorrow) != 0)
            continue;
        int r = read_day(d->issued, which[i], p, d->level[k], d->zones,
                         sizeof d->zones, err, err_len);
        if (r < 0)
            return found ? 1 : -1;
        if (r > 0) {
            copy(d->day[k], sizeof d->day[k], date[i]);
            found = 1;
            k++;
        }
    }
    return found;
}
