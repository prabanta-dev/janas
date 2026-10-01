/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * schedule.c - AviationStack: a flight's scheduled, estimated and actual
 * times, its gates, its status and the aircraft flying it. Its free plan
 * needs a key of the user's own and allows 100 calls a month, today's
 * flights only: the answers are kept for ten minutes, and the key never
 * leaves this file (not in an error, not in a log).
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "flights.h"

#define BASE "https://api.aviationstack.com/v1/flights?access_key="
#define KEEP_S 600 /* how long an answer is kept */
#define N_KEPT 16

static char key[128];

static struct {
    char query[128];
    time_t at;
    struct janas_buf body;
} kept[N_KEPT];

static void trim(char *s)
{
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n - 1]))
        s[--n] = 0;
    size_t i = strspn(s, " \t");
    memmove(s, s + i, n - i + 1);
}

int fl_sched_config(const char *path)
{
    const char *env = getenv("JANAS_AVIATIONSTACK_KEY");
    if (env && *env) {
        snprintf(key, sizeof key, "%s", env);
        trim(key);
        return 1;
    }
    FILE *f = path ? fopen(path, "r") : NULL;
    if (!f)
        return 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        trim(line);
        if (strncmp(line, "aviationstack_key", 17) != 0)
            continue;
        char *v = line + 17;
        v += strspn(v, " \t=");
        snprintf(key, sizeof key, "%s", v);
    }
    fclose(f);
    return key[0] != 0;
}

int fl_sched_have(void)
{
    return key[0] != 0;
}

/* "2026-09-30T18:25:00+00:00" in the zone tz. AviationStack writes the
   airport's local time and marks it +00:00 whatever the zone: the offset
   is ignored, the zone of the airport is the one that counts. */
static time_t local_time(const char *iso, const char *tz)
{
    struct tm tm = {.tm_isdst = -1};
    if (!iso || sscanf(iso, "%d-%d-%dT%d:%d", &tm.tm_year, &tm.tm_mon,
                       &tm.tm_mday, &tm.tm_hour, &tm.tm_min) != 5)
        return 0;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    char *old = getenv("TZ") ? strdup(getenv("TZ")) : NULL;
    setenv("TZ", tz && *tz ? tz : "UTC", 1);
    tzset();
    time_t t = mktime(&tm);
    if (old) {
        setenv("TZ", old, 1);
        free(old);
    } else {
        unsetenv("TZ");
    }
    tzset();
    return t == (time_t)-1 ? 0 : t;
}

static void copy_str(char *to, size_t cap, const struct janas_json *v)
{
    const char *s = janas_json_str(v);
    snprintf(to, cap, "%s", s ? s : "");
}

static void read_end(const struct janas_json *e, struct fl_end *o)
{
    copy_str(o->iata, sizeof o->iata, janas_json_get(e, "iata"));
    copy_str(o->icao, sizeof o->icao, janas_json_get(e, "icao"));
    copy_str(o->airport, sizeof o->airport, janas_json_get(e, "airport"));
    copy_str(o->tz, sizeof o->tz, janas_json_get(e, "timezone"));
    copy_str(o->terminal, sizeof o->terminal, janas_json_get(e, "terminal"));
    copy_str(o->gate, sizeof o->gate, janas_json_get(e, "gate"));
    copy_str(o->baggage, sizeof o->baggage, janas_json_get(e, "baggage"));
    const struct janas_json *d = janas_json_get(e, "delay");
    o->delay =
        d && d->type == JANAS_JSON_NUMBER ? (int)janas_json_num(d, 0) : -1;
    o->scheduled =
        local_time(janas_json_str(janas_json_get(e, "scheduled")), o->tz);
    o->estimated =
        local_time(janas_json_str(janas_json_get(e, "estimated")), o->tz);
    o->actual = local_time(janas_json_str(janas_json_get(e, "actual")), o->tz);
}

static void upper(char *s)
{
    for (; *s; s++)
        *s = (char)toupper((unsigned char)*s);
}

int fl_sched_parse(const char *text, size_t len, struct fl_sched **v, size_t *n,
                   char *err, size_t err_len)
{
    *v = NULL;
    *n = 0;
    char perr[128];
    struct janas_json_doc *doc = janas_json_parse(text, len, perr, sizeof perr);
    const struct janas_json *root = doc ? janas_json_root(doc) : NULL;
    const struct janas_json *e = janas_json_get(root, "error");
    if (e) {
        /* usage_limit_reached, invalid_access_key, function_access_
           restricted...: the code and the message, never the key */
        const char *code = janas_json_str(janas_json_get(e, "code"));
        const char *msg = janas_json_str(janas_json_get(e, "message"));
        snprintf(err, err_len, "AviationStack: %s%s%s", code ? code : "error",
                 msg ? ": " : "", msg ? msg : "");
        if (key[0] && strstr(err, key))
            snprintf(err, err_len, "AviationStack: %s", code ? code : "error");
        janas_json_free(doc);
        return -1;
    }
    const struct janas_json *data = janas_json_get(root, "data");
    if (!data || data->type != JANAS_JSON_ARRAY) {
        snprintf(err, err_len, "AviationStack: an answer it could not read");
        janas_json_free(doc);
        return -1;
    }
    struct fl_sched *s = data->n ? calloc(data->n, sizeof(*s)) : NULL;
    if (data->n && !s) {
        snprintf(err, err_len, "out of memory");
        janas_json_free(doc);
        return -1;
    }
    size_t got = 0;
    for (const struct janas_json *f = data->child; f && got < data->n;
         f = f->next) {
        struct fl_sched *o = &s[got];
        const struct janas_json *fl = janas_json_get(f, "flight"),
                                *al = janas_json_get(f, "airline"),
                                *ac = janas_json_get(f, "aircraft"),
                                *cs = janas_json_get(fl, "codeshared");
        copy_str(o->flight_iata, sizeof o->flight_iata,
                 janas_json_get(fl, "iata"));
        copy_str(o->flight_icao, sizeof o->flight_icao,
                 janas_json_get(fl, "icao"));
        if (!o->flight_iata[0] && !o->flight_icao[0])
            continue;
        copy_str(o->date, sizeof o->date, janas_json_get(f, "flight_date"));
        copy_str(o->status, sizeof o->status,
                 janas_json_get(f, "flight_status"));
        copy_str(o->airline, sizeof o->airline, janas_json_get(al, "name"));
        copy_str(o->airline_iata, sizeof o->airline_iata,
                 janas_json_get(al, "iata"));
        copy_str(o->airline_icao, sizeof o->airline_icao,
                 janas_json_get(al, "icao"));
        if (cs && cs->type == JANAS_JSON_OBJECT) {
            copy_str(o->operated_by, sizeof o->operated_by,
                     janas_json_get(cs, "flight_iata"));
            upper(o->operated_by);
            copy_str(o->operator_name, sizeof o->operator_name,
                     janas_json_get(cs, "airline_name"));
        }
        copy_str(o->reg, sizeof o->reg, janas_json_get(ac, "registration"));
        copy_str(o->hex, sizeof o->hex, janas_json_get(ac, "icao24"));
        for (char *h = o->hex; *h; h++) /* as the ADS-B networks write it */
            *h = (char)tolower((unsigned char)*h);
        read_end(janas_json_get(f, "departure"), &o->dep);
        read_end(janas_json_get(f, "arrival"), &o->arr);
        got++;
    }
    janas_json_free(doc);
    *v = s;
    *n = got;
    return 0;
}

int fl_schedule(const char *query, struct fl_sched **v, size_t *n, char *err,
                size_t err_len)
{
    *v = NULL;
    *n = 0;
    if (!key[0]) {
        snprintf(err, err_len,
                 "no AviationStack key: schedules are not available");
        return -1;
    }
    time_t now = time(NULL);
    int slot = 0;
    for (int i = 0; i < N_KEPT; i++) {
        if (strcmp(kept[i].query, query) == 0 && now - kept[i].at < KEEP_S)
            return fl_sched_parse(kept[i].body.p, kept[i].body.n, v, n, err,
                                  err_len);
        if (kept[i].at < kept[slot].at)
            slot = i;
    }
    char url[512];
    snprintf(url, sizeof url, BASE "%s&%s", key, query);
    struct janas_buf body = {0};
    char why[256];
    int status = fl_get(url, NULL, 0, &body, why, sizeof why);
    if (status < 0) {
        /* the address's errors quote it, and it holds the key */
        if (strstr(why, key))
            snprintf(why, sizeof why, "a bad address");
        snprintf(err, err_len, "AviationStack did not answer: %s", why);
        janas_buf_free(&body);
        return -1;
    }
    int r = fl_sched_parse(body.p ? body.p : "", body.n, v, n, err, err_len);
    if (r == 0 && status == 200) {
        janas_buf_free(&kept[slot].body);
        snprintf(kept[slot].query, sizeof kept[slot].query, "%s", query);
        kept[slot].at = now;
        kept[slot].body = body;
    } else {
        if (r == 0) { /* read, but not a 200: not to be trusted */
            free(*v);
            *v = NULL;
            *n = 0;
            snprintf(err, err_len, "AviationStack: HTTP %d", status);
            r = -1;
        }
        janas_buf_free(&body);
    }
    return r;
}

void fl_sched_free(void)
{
    for (int i = 0; i < N_KEPT; i++)
        janas_buf_free(&kept[i].body);
    memset(kept, 0, sizeof kept);
    memset(key, 0, sizeof key);
}
