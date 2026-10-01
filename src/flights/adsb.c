/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * adsb.c - the community ADS-B networks, in the readsb "v2" form both
 * serve: adsb.lol (its data under the ODbL) and adsb.fi (personal,
 * non-commercial use, with a link to it) when adsb.lol does not answer.
 * Neither asks for a key.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "flights.h"

static const struct {
    const char *name, *base;
} NETS[] = {
    {"adsb.lol", "https://api.adsb.lol/v2/"},
    {"adsb.fi", "https://opendata.adsb.fi/api/v2/"},
};

static void copy_str(char *to, size_t cap, const struct janas_json *v)
{
    const char *s = janas_json_str(v);
    size_t n = 0;
    if (s) /* trimmed: callsigns come padded with spaces */
        for (; *s && n + 1 < cap; s++)
            if (*s != ' ')
                to[n++] = *s;
    to[n] = 0;
}

static int read_ac(const struct janas_json *a, struct fl_ac *o)
{
    memset(o, 0, sizeof(*o));
    copy_str(o->hex, sizeof o->hex, janas_json_get(a, "hex"));
    if (!o->hex[0])
        return -1;
    copy_str(o->flight, sizeof o->flight, janas_json_get(a, "flight"));
    copy_str(o->reg, sizeof o->reg, janas_json_get(a, "r"));
    copy_str(o->type, sizeof o->type, janas_json_get(a, "t"));
    copy_str(o->squawk, sizeof o->squawk, janas_json_get(a, "squawk"));
    const struct janas_json *lat = janas_json_get(a, "lat"),
                            *lon = janas_json_get(a, "lon");
    if (lat && lon && lat->type == JANAS_JSON_NUMBER &&
        lon->type == JANAS_JSON_NUMBER) {
        o->has_pos = 1;
        o->lat = janas_json_num(lat, 0);
        o->lon = janas_json_num(lon, 0);
    }
    const struct janas_json *alt = janas_json_get(a, "alt_baro");
    if (janas_json_is(alt, "ground")) {
        o->ground = 1;
        o->has_alt = 1;
    } else if (alt && alt->type == JANAS_JSON_NUMBER) {
        o->has_alt = 1;
        o->alt = janas_json_num(alt, 0);
    }
    o->gs = janas_json_num(janas_json_get(a, "gs"), -1);
    o->track = janas_json_num(janas_json_get(a, "track"), -1);
    o->rate = janas_json_num(janas_json_get(a, "baro_rate"),
                             janas_json_num(janas_json_get(a, "geom_rate"), 0));
    o->seen_pos = janas_json_num(janas_json_get(a, "seen_pos"),
                                 janas_json_num(janas_json_get(a, "seen"), 0));
    return 0;
}

int fl_adsb(const char *query, struct fl_ac **ac, size_t *n,
            const char **source, char *err, size_t err_len)
{
    *ac = NULL;
    *n = 0;
    char why[2][160] = {"", ""};
    for (size_t k = 0; k < sizeof NETS / sizeof *NETS; k++) {
        char url[512];
        snprintf(url, sizeof url, "%s%s", NETS[k].base, query);
        struct janas_buf body = {0};
        int status = fl_get(url, NULL, 0, &body, why[k], sizeof why[k]);
        if (status != 200) {
            if (status > 0)
                snprintf(why[k], sizeof why[k], "HTTP %d", status);
            janas_buf_free(&body);
            continue;
        }
        char perr[128];
        struct janas_json_doc *doc =
            janas_json_parse(body.p, body.n, perr, sizeof perr);
        janas_buf_free(&body);
        const struct janas_json *list =
            doc ? janas_json_get(janas_json_root(doc), "ac") : NULL;
        if (!list) /* adsb.fi's older name for it */
            list =
                doc ? janas_json_get(janas_json_root(doc), "aircraft") : NULL;
        if (!doc || (list && list->type != JANAS_JSON_ARRAY)) {
            snprintf(why[k], sizeof why[k], "an answer it could not read");
            janas_json_free(doc);
            continue;
        }
        size_t cap = list ? list->n : 0, got = 0;
        struct fl_ac *v = cap ? calloc(cap, sizeof(*v)) : NULL;
        if (cap && !v) {
            janas_json_free(doc);
            snprintf(err, err_len, "out of memory");
            return -1;
        }
        for (const struct janas_json *a = list ? list->child : NULL;
             a && got < cap; a = a->next)
            if (read_ac(a, &v[got]) == 0)
                got++;
        janas_json_free(doc);
        *ac = v;
        *n = got;
        *source = NETS[k].name;
        return 0;
    }
    snprintf(err, err_len,
             "the ADS-B networks did not answer (adsb.lol: %s; "
             "adsb.fi: %s)",
             why[0], why[1]);
    return -1;
}
