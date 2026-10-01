/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - the tools janas-weather offers (see weather.h): weather_now,
 * weather_forecast, weather_sea and weather_alerts, and what they share:
 * their arguments, the place asked for, times in the place's own zone.
 * The tools themselves are in report.c, sea.c and warn.c.
 */
#define _GNU_SOURCE /* timegm */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/locate.h"
#include "common/mcp_server.h"
#include "weather.h"

static void schema_str(struct janas_buf *b, const char *name, const char *what)
{
    janas_buf_printf(b,
                     "\"%s\": {\"type\": \"string\", \"description\": ", name);
    janas_json_write_str(b, what, strlen(what));
    janas_buf_puts(b, "}");
}

static void place_props(struct janas_buf *b)
{
    schema_str(b, "place",
               "A place's name, in Italian or English (Trapani, Firenze, "
               "Florence, Milano); \"name, region\" or \"name, country\" "
               "when several places share it; or an airport's code. "
               "Leave out place and coordinates only when the user names "
               "no place: then where the user is (as set on the computer, "
               "or estimated from the internet connection).");
    janas_buf_puts(b,
                   ", \"latitude\": {\"type\": \"number\", \"minimum\": -90, "
                   "\"maximum\": 90}, \"longitude\": {\"type\": \"number\", "
                   "\"minimum\": -180, \"maximum\": 180}");
}

void wx_tools_list(void *ctx, struct janas_buf *b)
{
    (void)ctx;
    janas_buf_puts(
        b, "\"tools\": [{\"name\": \"weather_now\", \"title\": \"The "
           "weather now\", \"description\": \"The weather now at a place: "
           "sky, temperature and how it feels, humidity, wind and gusts, "
           "rain, pressure, today's lowest and highest and sunrise and "
           "sunset, from a forecast model; and what the nearest weather "
           "station (an airport's) measured, with when and how far away. "
           "Give a place's name, or a latitude and longitude.\", "
           "\"inputSchema\": {\"type\": \"object\", \"properties\": {");
    place_props(b);
    janas_buf_puts(
        b, "}, \"additionalProperties\": false}}, "
           "{\"name\": \"weather_forecast\", \"title\": \"Weather "
           "forecast\", \"description\": \"The forecast for a place, day "
           "by day (sky, lowest and highest temperature, rain and its "
           "chance, wind and gusts, sunrise and sunset) and the next hours "
           "(every three hours, or every hour when asked).\", "
           "\"inputSchema\": {\"type\": \"object\", \"properties\": {");
    place_props(b);
    janas_buf_puts(
        b, ", \"days\": {\"type\": \"integer\", \"minimum\": 1, "
           "\"maximum\": 16, \"description\": \"How many days, today "
           "included (default 3).\"}, \"hourly\": {\"type\": \"boolean\", "
           "\"description\": \"Every hour of the next two days, instead "
           "of every three hours of the next one.\"}}, "
           "\"additionalProperties\": false}}, "
           "{\"name\": \"weather_sea\", \"title\": \"The sea\", "
           "\"description\": \"The state of the sea, now and in the next "
           "days: wave height (with its name: calm, slight, moderate, "
           "rough...), direction and period, wind waves and swell, the "
           "water's temperature and the wind. For a whole sea by its name "
           "(Tyrrhenian Sea, Tirreno, Adriatico, mar Ligure), at several "
           "points across it, or for the sea off a coastal place.\", "
           "\"inputSchema\": {\"type\": \"object\", \"properties\": {");
    schema_str(b, "area",
               "A sea's name, in English or Italian (Tyrrhenian Sea, "
               "Tirreno, Adriatic, mar Ligure, Ionio).");
    janas_buf_puts(b, ", ");
    place_props(b);
    janas_buf_puts(
        b, ", \"days\": {\"type\": \"integer\", \"minimum\": 1, "
           "\"maximum\": 8, \"description\": \"How many days, today "
           "included (default 2).\"}}, \"additionalProperties\": false}}, "
           "{\"name\": \"weather_alerts\", \"title\": \"Weather "
           "warnings\", \"description\": \"The weather warnings in force "
           "for a place: those of the national weather services gathered "
           "by MeteoAlarm (wind, thunderstorms, heat, snow, by level: "
           "yellow, orange, red), the place's region first; and in Italy "
           "the Civil Protection's official alert levels for its zone, for "
           "floods, landslides and thunderstorms, today and tomorrow.\", "
           "\"inputSchema\": {\"type\": \"object\", \"properties\": {");
    place_props(b);
    janas_buf_puts(b, ", ");
    schema_str(b, "language",
               "The language of the warnings' texts when given in it: it, "
               "en, fr, de, es (default en).");
    janas_buf_puts(b, "}, \"additionalProperties\": false}}]");
}

int wx_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b)
{
    (void)ctx;
    const struct janas_json *name = janas_json_get(params, "name");
    const struct janas_json *args = janas_json_get(params, "arguments");
    if (args && args->type != JANAS_JSON_OBJECT)
        args = NULL;
    if (janas_json_is(name, "weather_now"))
        return wx_tool_now(args, b);
    if (janas_json_is(name, "weather_forecast"))
        return wx_tool_forecast(args, b);
    if (janas_json_is(name, "weather_sea"))
        return wx_tool_sea(args, b);
    if (janas_json_is(name, "weather_alerts"))
        return wx_tool_alerts(args, b);
    return -1;
}

/* ---- what the tools share ---- */

const char *wx_arg_str(const struct janas_json *args, const char *name)
{
    const char *s = janas_json_str(janas_json_get(args, name));
    return s && *s ? s : NULL;
}

double wx_arg_num(const struct janas_json *args, const char *name, double def)
{
    const struct janas_json *v = janas_json_get(args, name);
    return v && v->type == JANAS_JSON_NUMBER ? janas_json_num(v, def) : def;
}

int wx_arg_bool(const struct janas_json *args, const char *name, int def)
{
    const struct janas_json *v = janas_json_get(args, name);
    if (v && v->type == JANAS_JSON_TRUE)
        return 1;
    if (v && v->type == JANAS_JSON_FALSE)
        return 0;
    return def;
}

void wx_place_note(const char *how, struct janas_buf *out)
{
    if (how && *how)
        janas_buf_printf(out,
                         "\nNo place was named: this is where the user is, "
                         "%s. Tell the user which place it is and that it "
                         "may be off: they can name theirs.",
                         how);
}

int wx_result(struct janas_buf *out, struct janas_buf *b, int is_error)
{
    janas_mcps_text_result(b, out->p ? out->p : "", out->n, is_error);
    janas_buf_free(out);
    return 0;
}

int wx_arg_place(const struct janas_json *args, struct wx_place *p,
                 struct janas_buf *b)
{
    const char *q = wx_arg_str(args, "place");
    double lat = wx_arg_num(args, "latitude", NAN);
    double lon = wx_arg_num(args, "longitude", NAN);
    char err[384];
    if (q && wx_place_find(q, p, err, sizeof err) == 0)
        return 0;
    if (!q && !isnan(lat) && !isnan(lon)) {
        wx_place_at(lat, lon, p);
        return 0;
    }
    struct janas_where w;
    if (!q && janas_where(&w, err, sizeof err) == 0) {
        /* named as a point is, in the country's language: the source's
           own names are English at times (Sicily), the warnings' not */
        wx_place_at(w.lat, w.lon, p);
        /* "3 km WNW of Palermo" says more than a guess knows: its town */
        if (w.city[0] && strstr(p->name, " km "))
            snprintf(p->name, sizeof p->name, "%s", w.city);
        if (!p->tz[0])
            snprintf(p->tz, sizeof p->tz, "%s", w.tz);
        snprintf(p->how, sizeof p->how, "%s", w.how);
        return 0;
    }
    struct janas_buf out = {0};
    if (!q)
        janas_buf_printf(&out, "No place given, and %s.", err);
    else
        janas_buf_printf(&out,
                         "%s. Ask the user which place, or try the "
                         "name in another language or with its "
                         "region (\"Name, Region\").",
                         err);
    wx_result(&out, b, 1);
    return -1;
}

void wx_local_time(time_t t, const char *tz, char *out, size_t cap)
{
    char *old = getenv("TZ") ? strdup(getenv("TZ")) : NULL;
    if (tz && *tz)
        setenv("TZ", tz, 1);
    tzset();
    struct tm lt, nt;
    time_t now = time(NULL);
    localtime_r(&t, &lt);
    localtime_r(&now, &nt);
    char hm[16], day[32] = "";
    strftime(hm, sizeof hm, "%H:%M", &lt);
    if (lt.tm_yday != nt.tm_yday || lt.tm_year != nt.tm_year)
        strftime(day, sizeof day, " on %A %-d %B", &lt);
    snprintf(out, cap, "%s%s %s", hm, day, lt.tm_zone ? lt.tm_zone : "");
    if (tz && *tz) {
        if (old)
            setenv("TZ", old, 1);
        else
            unsetenv("TZ");
        tzset();
    }
    free(old);
}

void wx_day_name(const char *date, char *out, size_t cap)
{
    struct tm tm = {0};
    if (sscanf(date, "%d-%d-%d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday) != 3) {
        snprintf(out, cap, "%s", date);
        return;
    }
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    tm.tm_hour = 12;
    time_t t = timegm(&tm);
    struct tm g;
    gmtime_r(&t, &g);
    strftime(out, cap, "%A %-d %B", &g);
}

void wx_place_name(const struct wx_place *p, char *out, size_t cap)
{
    if (p->region[0] && p->country[0] && strcmp(p->region, p->name) != 0)
        snprintf(out, cap, "%s (%s, %s)", p->name, p->region, p->country);
    else if (p->country[0] && strcmp(p->country, p->name) != 0)
        snprintf(out, cap, "%s (%s)", p->name, p->country);
    else
        snprintf(out, cap, "%s", p->name);
    size_t n = strlen(out);
    if (p->how[0] && n < cap) /* the model is to say it is a guess */
        snprintf(out + n, cap - n, " - where the user is, %s", p->how);
}
