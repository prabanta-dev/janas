/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * track.c - the way an aircraft has flown so far, from the OpenSky
 * Network's tracks (no key; its data for research and non-commercial use,
 * credited): each point's time, place and height, from its take-off or
 * from when the network first heard it. Answers kept a minute, by
 * aircraft: a view asks again when its plane is chosen again.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "flights.h"

#define BASE "https://opensky-network.org/api/tracks/all?time=0&icao24="
#define KEEP 16
#define KEEP_S 60
#define MAX_PTS 4000

static struct {
    char hex[8];
    time_t at;
    struct janas_buf json; /* the path's points, as written */
} kept[KEEP];
static int kept_next;

static int hex_ok(const char *h)
{
    size_t n = strspn(h, "0123456789abcdef");
    return n == 6 && !h[n];
}

/* the points of an answer: [t, lat, lon, ft or null], into out */
static int points(const char *text, size_t len, struct janas_buf *out)
{
    char err[160];
    struct janas_json_doc *doc = janas_json_parse(text, len, err, sizeof err);
    if (!doc)
        return -1;
    const struct janas_json *path =
        janas_json_get(janas_json_root(doc), "path");
    int n = 0;
    janas_buf_puts(out, "[");
    for (const struct janas_json *p =
             path && path->type == JANAS_JSON_ARRAY ? path->child : NULL;
         p && n < MAX_PTS; p = p->next) {
        const struct janas_json *v[6] = {0};
        int k = 0;
        for (const struct janas_json *e = p->type == JANAS_JSON_ARRAY ? p->child
                                                                      : NULL;
             e && k < 6; e = e->next)
            v[k++] = e;
        if (k < 3 || !v[1] || v[1]->type != JANAS_JSON_NUMBER || !v[2] ||
            v[2]->type != JANAS_JSON_NUMBER)
            continue;
        int ground = k > 5 && v[5] && v[5]->type == JANAS_JSON_TRUE;
        janas_buf_printf(out, "%s[%.0f, %.5f, %.5f, ", n ? ", " : "",
                         janas_json_num(v[0], 0), janas_json_num(v[1], 0),
                         janas_json_num(v[2], 0));
        if (ground)
            janas_buf_puts(out, "0]");
        else if (k > 3 && v[3] && v[3]->type == JANAS_JSON_NUMBER)
            janas_buf_printf(out, "%.0f]", /* m -> ft; below the sea's 0 */
                             fmax(0, janas_json_num(v[3], 0) / 0.3048));
        else
            janas_buf_puts(out, "null]");
        n++;
    }
    janas_buf_puts(out, "]");
    janas_json_free(doc);
    return n;
}

int fl_track(const char *hex, struct janas_buf *out, char *err, size_t err_len)
{
    err[0] = 0;
    if (!hex_ok(hex)) {
        snprintf(err, err_len, "not an aircraft's ICAO address");
        return -1;
    }
    time_t now = time(NULL);
    for (int i = 0; i < KEEP; i++)
        if (!strcmp(kept[i].hex, hex) && now - kept[i].at < KEEP_S) {
            janas_buf_put(out, kept[i].json.p, kept[i].json.n);
            return 0;
        }
    char url[128];
    snprintf(url, sizeof url, BASE "%s", hex);
    struct janas_buf body = {0}, pts = {0};
    int status = janas_https_get(url, NULL, 0, &body, err, err_len);
    if (status == 404) { /* not heard by the network */
        janas_buf_free(&body);
        janas_buf_puts(&pts, "[]");
    } else if (status != 200 || points(body.p, body.n, &pts) < 0) {
        if (status >= 0)
            snprintf(err, err_len, "OpenSky answered %d", status);
        janas_buf_free(&body);
        janas_buf_free(&pts);
        return -1;
    } else
        janas_buf_free(&body);
    if (pts.oom) {
        janas_buf_free(&pts);
        snprintf(err, err_len, "out of memory");
        return -1;
    }
    int k = kept_next;
    kept_next = (kept_next + 1) % KEEP;
    janas_buf_free(&kept[k].json);
    snprintf(kept[k].hex, sizeof kept[k].hex, "%s", hex);
    kept[k].at = now;
    kept[k].json = pts;
    janas_buf_put(out, pts.p, pts.n);
    return 0;
}
