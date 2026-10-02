/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * route.c - adsbdb.com: the route the community has recorded for a
 * callsign, and an airline's codes. No key. Its routes are right for the
 * airlines whose callsign is their flight number (Vueling, Egyptair,
 * Turkish) and often wrong for those that reuse letters and digits
 * (Ryanair, Wizz Air): the tools say they are uncertain.
 */
#include <stdio.h>
#include <string.h>

#include "flights.h"

#define BASE "https://api.adsbdb.com/v0/"

static void copy_str(char *to, size_t cap, const struct janas_json *v)
{
    const char *s = janas_json_str(v);
    snprintf(to, cap, "%s", s ? s : "");
}

/* The response member of an adsbdb answer, into *doc: 1, 0 when it has
   none (unknown), -1 when it could not be asked. */
static int ask(const char *path, struct janas_json_doc **doc,
               const struct janas_json **resp)
{
    char url[256], err[160];
    snprintf(url, sizeof url, BASE "%s", path);
    struct janas_buf body = {0};
    int status = janas_https_get(url, NULL, 0, &body, err, sizeof err);
    *doc = NULL;
    if (status == 404) {
        janas_buf_free(&body);
        return 0;
    }
    if (status != 200) {
        janas_buf_free(&body);
        return -1;
    }
    *doc = janas_json_parse(body.p, body.n, err, sizeof err);
    janas_buf_free(&body);
    *resp = *doc ? janas_json_get(janas_json_root(*doc), "response") : NULL;
    if (!*resp) {
        janas_json_free(*doc);
        *doc = NULL;
        return -1;
    }
    /* "unknown callsign" and the like come as a string */
    return (*resp)->type == JANAS_JSON_OBJECT ||
                   (*resp)->type == JANAS_JSON_ARRAY
               ? 1
               : 0;
}

static void place(const struct janas_json *p, struct fl_place *o)
{
    copy_str(o->iata, sizeof o->iata, janas_json_get(p, "iata_code"));
    copy_str(o->icao, sizeof o->icao, janas_json_get(p, "icao_code"));
    copy_str(o->name, sizeof o->name, janas_json_get(p, "name"));
    copy_str(o->city, sizeof o->city, janas_json_get(p, "municipality"));
    copy_str(o->country, sizeof o->country, janas_json_get(p, "country_name"));
    o->lat = janas_json_num(janas_json_get(p, "latitude"), 0);
    o->lon = janas_json_num(janas_json_get(p, "longitude"), 0);
}

int fl_route(const char *callsign, struct fl_route *r)
{
    memset(r, 0, sizeof(*r));
    char path[64];
    snprintf(path, sizeof path, "callsign/%.20s", callsign);
    struct janas_json_doc *doc;
    const struct janas_json *resp;
    int k = ask(path, &doc, &resp);
    if (k <= 0)
        return k;
    const struct janas_json *fr = janas_json_get(resp, "flightroute");
    if (!fr) {
        janas_json_free(doc);
        return 0;
    }
    const struct janas_json *al = janas_json_get(fr, "airline");
    copy_str(r->airline, sizeof r->airline, janas_json_get(al, "name"));
    copy_str(r->airline_icao, sizeof r->airline_icao,
             janas_json_get(al, "icao"));
    copy_str(r->airline_iata, sizeof r->airline_iata,
             janas_json_get(al, "iata"));
    copy_str(r->flight_iata, sizeof r->flight_iata,
             janas_json_get(fr, "callsign_iata"));
    place(janas_json_get(fr, "origin"), &r->from);
    place(janas_json_get(fr, "destination"), &r->to);
    janas_json_free(doc);
    return r->from.iata[0] || r->to.iata[0] ? 1 : 0;
}

int fl_airline_icao(const char *iata, char icao[4], char *name, size_t name_len)
{
    char path[64];
    snprintf(path, sizeof path, "airline/%.3s", iata);
    struct janas_json_doc *doc;
    const struct janas_json *resp;
    int k = ask(path, &doc, &resp);
    icao[0] = 0;
    if (k <= 0)
        return k;
    const struct janas_json *a =
        resp->type == JANAS_JSON_ARRAY ? resp->child : resp;
    /* the one whose IATA code is the one asked: a code may be shared */
    for (; a; a = resp->type == JANAS_JSON_ARRAY ? a->next : NULL)
        if (janas_json_is(janas_json_get(a, "iata"), iata))
            break;
    if (a) {
        copy_str(icao, 4, janas_json_get(a, "icao"));
        copy_str(name, name_len, janas_json_get(a, "name"));
    }
    janas_json_free(doc);
    return icao[0] ? 1 : 0;
}
