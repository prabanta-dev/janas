/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - the tools janas-maps offers (see maps.h): maps_route,
 * maps_transit, maps_nearby and maps_find, and what they share: their
 * arguments, the places asked for, the answer as data with a layout. The
 * tools themselves are in route.c, transit.c, nearby.c and place.c.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "services/common/locate.h"
#include "common/mcp_server.h"
#include "services/common/template.h"
#include "maps.h"

/* The model reads all of it before it can use a tool: every word costs. */
#define PLACE_TEXT                                                             \
    "A place or an address with its town (\\\"Via Roma 10, Trapani\\\", "      \
    "\\\"Colosseo\\\"), or \\\"lat,lon\\\"."

void mp_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    janas_buf_puts(
        b, "\"tools\": [{\"name\": \"maps_route\", \"title\": \"A route by "
           "road\", \"description\": \"The route between places by car, bike "
           "or on foot: distance, time, roads, and the way step by step.\", "
           "\"inputSchema\": {\"type\": \"object\", \"properties\": {\"from\": "
           "{\"type\": \"string\", \"description\": \"" PLACE_TEXT
           " Leave it out for where the user is (\\\"near me\\\"): "
           "never a place only talked of.\"}, \"to\": {\"type\": "
           "\"string\", \"description\": \"" PLACE_TEXT
           "\"}, \"via\": {\"type\": \"string\", \"description\": \"A place "
           "to pass through.\"}, \"mode\": {\"type\": \"string\", \"enum\": "
           "[\"car\", \"bike\", \"foot\"]}, \"avoid\": {\"type\": \"array\", "
           "\"items\": {\"type\": \"string\", \"enum\": [\"tolls\", "
           "\"highways\", \"ferries\"]}}}, \"required\": [\"to\"], "
           "\"additionalProperties\": false}}, "
           "{\"name\": \"maps_transit\", \"title\": \"Public transport\", "
           "\"description\": \"Journeys by train, bus, metro or ferry between "
           "places, from the timetables.\", \"inputSchema\": {\"type\": "
           "\"object\", \"properties\": {\"from\": {\"type\": \"string\", "
           "\"description\": \"" PLACE_TEXT
           " Leave it out for where the user is (\\\"near me\\\"): "
           "never a place only talked of.\"}, \"to\": {\"type\": "
           "\"string\", \"description\": \"" PLACE_TEXT
           "\"}, \"when\": {\"type\": \"string\", \"description\": \"Local "
           "time, HH:MM or YYYY-MM-DD HH:MM (default now).\"}, \"arrive_by\": "
           "{\"type\": \"boolean\", \"description\": \"The time is to arrive "
           "by, not to leave at.\"}}, \"required\": [\"to\"], "
           "\"additionalProperties\": false}}, "
           "{\"name\": \"maps_nearby\", \"title\": \"What is near\", "
           "\"description\": \"Places of a kind near a place, nearest first: "
           "fuel, pharmacies, cash machines, stations...\", \"inputSchema\": "
           "{\"type\": \"object\", \"properties\": {\"what\": {\"type\": "
           "\"string\", \"enum\": [");
    for (size_t i = 0; i < MP_N_KINDS; i++)
        janas_buf_printf(b, "%s\"%s\"", i ? ", " : "", MP_KINDS[i].name);
    janas_buf_puts(
        b, "]}, \"place\": {\"type\": \"string\", \"description\": "
           "\"" PLACE_TEXT
           " Leave it out for where the user is (\\\"near me\\\"): "
           "never a place only talked of.\"}, "
           "\"radius\": {\"type\": \"number\", \"minimum\": 0.1, "
           "\"maximum\": 10, \"description\": \"km (default 2).\"}}, "
           "\"required\": [\"what\"], \"additionalProperties\": false}}, "
           "{\"name\": \"maps_find\", \"title\": \"Where a place is\", "
           "\"description\": \"A place's or an address' full address, "
           "coordinates and map link.\", \"inputSchema\": {\"type\": "
           "\"object\", \"properties\": {\"place\": {\"type\": \"string\", "
           "\"description\": \"" PLACE_TEXT "\"}}, \"required\": "
           "[\"place\"], \"additionalProperties\": false}}]");
}

int mp_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    (void)ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    if (janas_json_is(name, "maps_route"))
        return mp_tool_route(args, b);
    if (janas_json_is(name, "maps_transit"))
        return mp_tool_transit(args, b);
    if (janas_json_is(name, "maps_nearby"))
        return mp_tool_nearby(args, b);
    if (janas_json_is(name, "maps_find"))
        return mp_tool_find(args, b);
    return -1;
}

/* ---- what the tools share ---- */

const char *mp_arg_str(const struct janas_json *args, const char *name)
{
    const char *s = janas_json_str(janas_json_get(args, name));
    return s && *s ? s : NULL;
}

double mp_arg_num(const struct janas_json *args, const char *name, double def)
{
    const struct janas_json *v = janas_json_get(args, name);
    return v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, def) : def;
}

int mp_arg_bool(const struct janas_json *args, const char *name, int def)
{
    const struct janas_json *v = janas_json_get(args, name);
    if (v && v->type == JANAS_JSON_TRUE)
        return 1;
    if (v && v->type == JANAS_JSON_FALSE)
        return 0;
    return def;
}

int mp_result(struct janas_buf *out, struct janas_buf *b, int is_error)
{
    janas_mcps_text_result(b, out->p ? out->p : "", out->n, is_error);
    janas_buf_free(out);
    return 0;
}

int mp_arg_place(const struct janas_json *args, const char *name, int here,
                 double near_lat, double near_lon, struct mp_place *p,
                 struct janas_buf *b)
{
    const char *q = mp_arg_str(args, name);
    char err[512];
    if (q && mp_place_find(q, near_lat, near_lon, p, err, sizeof err) == 0)
        return 0;
    struct janas_where w;
    if (!q && here && janas_where(&w, err, sizeof err) == 0) {
        mp_place_at(w.lat, w.lon, p);
        /* "3 km WNW of Palermo" says more than a guess knows: its town */
        if (w.city[0])
            snprintf(p->name, sizeof p->name, "%s", w.city);
        snprintf(p->how, sizeof p->how, "%s", w.how);
        p->address[0] = 0; /* its coordinates: a guess's, not an address */
        return 0;
    }
    struct janas_buf out = {0};
    if (!q && here)
        janas_buf_printf(&out, "No %s given, and %s.", name, err);
    else if (!q)
        janas_buf_printf(&out, "No %s given.", name);
    else
        janas_buf_printf(&out,
                         "%s. Ask the user which place, or give it with its "
                         "town (\"Via Roma 10, Trapani\").",
                         err);
    mp_result(&out, b, 1);
    return -1;
}

int mp_answer(struct janas_buf *b, const char *name, struct janas_buf *d,
              const char *layout, const char *brief)
{
    char err[300];
    if (d->oom || janas_tpl_result(b, name, d->p, d->n, layout, brief, err,
                                   sizeof err) != 0) {
        struct janas_buf out = {0};
        janas_buf_printf(&out, "The answer could not be written: %s.",
                         d->oom ? "out of memory" : err);
        janas_buf_free(d);
        return mp_result(&out, b, 1);
    }
    janas_buf_free(d);
    return 0;
}

int mp_tool_find(const struct janas_json *args, struct janas_buf *b)
{
    struct mp_place p;
    if (mp_arg_place(args, "place", 0, NAN, NAN, &p, b) != 0)
        return 0;
    if (!p.address[0]) /* a point: what is there */
        mp_place_reverse(p.lat, p.lon, &p);
    struct janas_buf d = {0};
    mp_find_data(&d, &p);
    return mp_answer(b, "maps_find", &d, MP_FIND_LAYOUT, MP_FIND_BRIEF);
}
