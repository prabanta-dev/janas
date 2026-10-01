/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * weather.h - janas-weather, the weather as an MCP service: what its parts
 * share. fetch.c asks the sources (answers kept a while), place.c finds a
 * place by its name, openmeteo.c reads Open-Meteo's forecasts and seas,
 * metno.c MET Norway's (when Open-Meteo does not answer), metar.c the
 * observations of the airports' stations, alerts.c MeteoAlarm's warnings
 * and the Italian Civil Protection's bulletin, words.c puts it all into
 * words, tools.c offers the tools.
 */
#ifndef JANAS_WEATHER_H
#define JANAS_WEATHER_H

#include <stddef.h>
#include <time.h>

#include "llm/json.h"

/* ---- fetch.c ---- */

/* GET of url, the body into out; answers of status 200 are kept keep_s
   seconds and given again meanwhile. with_agent: the User-Agent MET Norway
   asks for, naming the program and where it lives. The HTTP status, or -1
   with the reason in err. */
int wx_fetch(const char *url, int keep_s, int with_agent, struct janas_buf *out,
             char *err, size_t err_len);
void wx_fetch_free(void);

/* The text of a JSON document read (NULL with the reason in err). */
struct janas_json_doc *wx_parse(const struct janas_buf *b, const char *who,
                                char *err, size_t err_len);

/* ---- place.c ---- */

struct wx_place {
    char name[96];    /* "Trapani" */
    char region[64];  /* "Sicilia": the first-level division */
    char area2[64];   /* "Trapani": the second, a province */
    char country[64]; /* "Italia" */
    char cc[3];       /* "IT" */
    char tz[48];      /* "Europe/Rome" */
    double lat, lon;
    double elevation; /* metres; NAN when not known */
    char how[96];     /* where the user is, when the place was not given:
                         how it was told (locate.h); "" otherwise */
};

/* A place by its name (in any language Open-Meteo's geocoding reads,
   Italian and English among them), an airport's code, or "lat,lon". 0, or
   -1 with the reason in err. */
int wx_place_find(const char *query, struct wx_place *p, char *err,
                  size_t err_len);
/* A point: its place in words and zone from the built-in tables. */
void wx_place_at(double lat, double lon, struct wx_place *p);

/* ---- openmeteo.c, metno.c: a forecast ---- */

#define WX_DAYS 16
#define WX_HOURS 48

struct wx_now {
    char time[20]; /* local, "2026-10-01T13:30" */
    double temp, feels, humidity, precip, cloud, pressure;
    double wind, wind_dir, gust; /* km/h, degrees from */
    int code;                    /* WMO weather code; -1: none */
    char symbol[48];             /* MET Norway's, when it is the source */
    int is_day;
};

struct wx_day {
    char date[11];
    int code;
    char symbol[48];
    double tmin, tmax, rain, rain_prob, wind, gust, wind_dir, uv;
    char sunrise[6], sunset[6]; /* "07:05"; "" when not given */
};

struct wx_hour {
    char time[20];
    double temp, rain, rain_prob, wind, wind_dir, gust;
    int code;
    char symbol[48];
};

struct wx_forecast {
    const char *source;                   /* "Open-Meteo", "MET Norway" */
    double grid_lat, grid_lon, grid_elev; /* where the model was read */
    char tz[48], tz_abbr[16];
    struct wx_now now;
    int has_now;
    struct wx_day day[WX_DAYS];
    int n_day;
    struct wx_hour hour[WX_HOURS];
    int n_hour;
};

/* Open-Meteo: now, days days ahead and the hours of the next two days. */
int wx_om_forecast(const struct wx_place *p, int days, struct wx_forecast *f,
                   char *err, size_t err_len);
int wx_om_parse(const char *text, size_t n, struct wx_forecast *f, char *err,
                size_t err_len);
/* MET Norway's locationforecast, into the same, in the place's zone. */
int wx_met_forecast(const struct wx_place *p, int days, struct wx_forecast *f,
                    char *err, size_t err_len);
int wx_met_parse(const char *text, size_t n, const char *tz,
                 struct wx_forecast *f, char *err, size_t err_len);

/* ---- openmeteo.c: the sea ---- */

#define WX_SEA_POINTS 9

struct wx_sea_day {
    char date[11];
    double wave_max, wave_dir, period_max;
    double wind_max, gust_max, wind_dir; /* the forecast's, at the point */
};

struct wx_sea {
    double lat, lon;           /* the point asked */
    double cell_lat, cell_lon; /* the cell of the sea model answering */
    int ok;                    /* the model has sea there */
    char time[20];             /* of the values now, in tz */
    double wave, wave_dir, period, wind_wave, swell, swell_dir, sst;
    double wind, wind_dir, gust; /* km/h, from the forecast */
    struct wx_sea_day day[WX_DAYS];
    int n_day;
};

/* The sea at n points, days days ahead, times in the zone tz: Open-Meteo's
   waves and its wind (two calls for all the points). 0, or -1. */
int wx_om_marine(struct wx_sea *s, int n, int days, const char *tz, char *err,
                 size_t err_len);
/* The waves' answer, then the wind's, into s. */
int wx_om_marine_parse(const char *text, size_t len, struct wx_sea *s, int n,
                       char *err, size_t err_len);
int wx_om_wind_parse(const char *text, size_t len, struct wx_sea *s, int n,
                     char *err, size_t err_len);

/* ---- metar.c: what a station measured ---- */

struct wx_obs {
    char icao[5], name[96];
    double lat, lon, km; /* the station, its distance from the place */
    time_t at;           /* when it was measured */
    double temp, dew, wind_dir, wind_kt, gust_kt, pressure;
    int variable_wind; /* direction "VRB" */
    char visib[8];     /* statute miles as given: "6+", "1 1/2" */
    char wx[32];       /* present weather, as coded: "-RA", "TSRA" */
    char clouds[96];   /* "few at 1500 ft, scattered at 1700 ft" */
    char raw[160];
};

/* The observation of the station nearest the point within max_km. 1, 0
   when none, -1 when it could not be asked. */
int wx_metar_near(double lat, double lon, double max_km, struct wx_obs *o,
                  char *err, size_t err_len);
int wx_metar_parse(const char *text, size_t n, double lat, double lon,
                   double max_km, struct wx_obs *o, char *err, size_t err_len);

/* ---- alerts.c ---- */

struct wx_alert {
    char event[96]; /* "Allerta gialla per temporali" */
    char level[16]; /* "yellow" */
    int mine;       /* for the place's region */
    char area[96];  /* "Sicilia" */
    char onset[26], expires[26];
    char sender[96];
    char text[512]; /* description, cut */
};

/* "2026-10-01T14:00:00+00:00" as a time; 0 when not one. */
time_t wx_iso_time(const char *s);
/* MeteoAlarm's warnings in force for the place's country, those of its
   region first, in lang when given in it ("it", "en"). *v is malloc'd. */
int wx_meteoalarm(const struct wx_place *p, const char *lang,
                  struct wx_alert **v, size_t *n, char *err, size_t err_len);
int wx_meteoalarm_parse(const char *text, size_t len, const char *region,
                        const char *lang, time_t now, struct wx_alert **v,
                        size_t *n, char *err, size_t err_len);

/* The Italian Civil Protection's bulletin for a municipality: the zones
   of alert it is in and their levels, for the day the place lives and the
   next when the bulletin has it. */
struct wx_dpc {
    char issued[16];      /* "20260930_1423" */
    char zones[192];      /* "Nord-Occidentale e isole Egadi e Ustica" */
    char day[2][11];      /* the dates the two sets are for; "" when none */
    char level[2][3][96]; /* hydraulic, hydrogeological, thunderstorms */
};

/* 1 when the place is in the bulletin, 0 when not found, -1 on error. */
int wx_dpc(const struct wx_place *p, struct wx_dpc *d, char *err,
           size_t err_len);
/* A bulletin's day read: the levels of the zones holding the municipality
   (the highest, when it is in several), the zones' names; when no zone
   names it, the zone the point is in (lat, lon NAN: none). 1, 0, -1. */
int wx_dpc_parse(const char *text, size_t len, const char *comune, double lat,
                 double lon, char out[3][96], char *zones, size_t zones_len,
                 char *err, size_t err_len);

/* ---- words.c ---- */

/* The WMO weather code in words ("light rain"); NULL when unknown. */
const char *wx_code_text(int code);
/* MET Norway's symbol ("lightrainshowers_day") in words. */
void wx_symbol_text(const char *symbol, char *out, size_t cap);
/* The wind's force in words, on the Beaufort scale ("moderate breeze"). */
const char *wx_wind_words(double kmh);
/* The sea's state from the waves' height, on the Douglas scale. */
const char *wx_sea_words(double m);
/* "from the north-west (321°)" */
void wx_from_dir(double deg, char *out, size_t cap);

/* ---- tools.c: the tools, and what they share ---- */

void wx_tools_list(void *ctx, struct janas_buf *b);
int wx_tools_call(void *ctx, const struct janas_json *params,
                  struct janas_buf *b);

/* An argument: a string (NULL when absent), a number (def), a boolean. */
const char *wx_arg_str(const struct janas_json *args, const char *name);
double wx_arg_num(const struct janas_json *args, const char *name, double def);
int wx_arg_bool(const struct janas_json *args, const char *name, int def);
/* The place of the arguments ("place", or "latitude" and "longitude"):
   0, or -1 with the error already written as the tool's result. */
int wx_arg_place(const struct janas_json *args, struct wx_place *p,
                 struct janas_buf *b);
/* t as "13:55" in the zone tz (the machine's when NULL), and the day when
   it is not today there. */
void wx_local_time(time_t t, const char *tz, char *out, size_t cap);
/* "2026-10-02" -> "Friday 2 October" */
void wx_day_name(const char *date, char *out, size_t cap);
/* "Trapani (Sicilia, Italia)" */
void wx_place_name(const struct wx_place *p, char *out, size_t cap);
/* When the user named no place (how, from struct wx_place, not ""): a
   last line telling the model to say which place it was and how known. */
void wx_place_note(const char *how, struct janas_buf *out);
/* The text as the tool's result, and freed. */
int wx_result(struct janas_buf *out, struct janas_buf *b, int is_error);

/* ---- report.c, sea.c, warn.c: the tools ---- */

int wx_tool_now(const struct janas_json *args, struct janas_buf *b);
int wx_tool_forecast(const struct janas_json *args, struct janas_buf *b);
int wx_tool_sea(const struct janas_json *args, struct janas_buf *b);
int wx_tool_alerts(const struct janas_json *args, struct janas_buf *b);

#endif
