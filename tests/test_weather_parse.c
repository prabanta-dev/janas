/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_weather_parse.c - janas-weather's reading of its sources' answers,
 * without the network: answers as they came on 1 October 2026 (Trapani's
 * forecast from Open-Meteo, the sea at three points, Trapani-Birgi's
 * METAR, MET Norway's series, cut to forty hours), MeteoAlarm's feed for
 * Italy with a yellow and an orange warning made from its own (that day
 * every warning was green) and two zones of the Civil Protection's
 * bulletin made from one of its own; and the words.
 * The sources belong to the program, so they are compiled in here.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "weather/alerts.c"
#include "weather/fetch.c"
#include "weather/metar.c"
#include "weather/metno.c"
#include "weather/openmeteo.c"
#include "weather/words.c"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char OM_FORECAST[] =
    "{\"latitude\": 38.0, \"longitude\": 12.5625, \"generationtime_ms\": 0.36"
    "41843795776367, \"utc_offset_seconds\": 7200, \"timezone\": \"Europe/Rom"
    "e\", \"timezone_abbreviation\": \"GMT+2\", \"elevation\": 6.0, \"current"
    "_units\": {\"time\": \"iso8601\", \"interval\": \"seconds\", \"temperatu"
    "re_2m\": \"°C\", \"relative_humidity_2m\": \"%\", \"apparent_temperature"
    "\": \"°C\", \"precipitation\": \"mm\", \"weather_code\": \"wmo code\", "
    "\"cloud_cover\": \"%\", \"pressure_msl\": \"hPa\", \"wind_speed_10m\": \""
    "km/h\", \"wind_direction_10m\": \"°\", \"wind_gusts_10m\": \"km/h\", \"i"
    "s_day\": \"\"}, \"current\": {\"time\": \"2026-10-01T13:45\", \"interval"
    "\": 900, \"temperature_2m\": 26.3, \"relative_humidity_2m\": 48, \"appar"
    "ent_temperature\": 26.5, \"precipitation\": 0.0, \"weather_code\": 0, \""
    "cloud_cover\": 14, \"pressure_msl\": 1022.8, \"wind_speed_10m\": 13.4, "
    "\"wind_direction_10m\": 324, \"wind_gusts_10m\": 27.0, \"is_day\": 1}, \""
    "hourly_units\": {\"time\": \"iso8601\", \"temperature_2m\": \"°C\", \"pr"
    "ecipitation_probability\": \"%\", \"precipitation\": \"mm\", \"weather_c"
    "ode\": \"wmo code\", \"wind_speed_10m\": \"km/h\", \"wind_direction_10m"
    "\": \"°\", \"wind_gusts_10m\": \"km/h\"}, \"hourly\": {\"time\": [\"2026-"
    "10-01T00:00\", \"2026-10-01T01:00\", \"2026-10-01T02:00\", \"2026-10-01T"
    "03:00\", \"2026-10-01T04:00\", \"2026-10-01T05:00\", \"2026-10-01T06:00"
    "\", \"2026-10-01T07:00\", \"2026-10-01T08:00\", \"2026-10-01T09:00\", \"2"
    "026-10-01T10:00\", \"2026-10-01T11:00\", \"2026-10-01T12:00\", \"2026-10"
    "-01T13:00\", \"2026-10-01T14:00\", \"2026-10-01T15:00\", \"2026-10-01T16"
    ":00\", \"2026-10-01T17:00\", \"2026-10-01T18:00\", \"2026-10-01T19:00\","
    " \"2026-10-01T20:00\", \"2026-10-01T21:00\", \"2026-10-01T22:00\", \"202"
    "6-10-01T23:00\", \"2026-10-02T00:00\", \"2026-10-02T01:00\", \"2026-10-0"
    "2T02:00\", \"2026-10-02T03:00\", \"2026-10-02T04:00\", \"2026-10-02T05:0"
    "0\", \"2026-10-02T06:00\", \"2026-10-02T07:00\", \"2026-10-02T08:00\", "
    "\"2026-10-02T09:00\", \"2026-10-02T10:00\", \"2026-10-02T11:00\", \"2026-"
    "10-02T12:00\", \"2026-10-02T13:00\", \"2026-10-02T14:00\", \"2026-10-02T"
    "15:00\", \"2026-10-02T16:00\", \"2026-10-02T17:00\", \"2026-10-02T18:00"
    "\", \"2026-10-02T19:00\", \"2026-10-02T20:00\", \"2026-10-02T21:00\", \"2"
    "026-10-02T22:00\", \"2026-10-02T23:00\", \"2026-10-03T00:00\", \"2026-10"
    "-03T01:00\", \"2026-10-03T02:00\", \"2026-10-03T03:00\", \"2026-10-03T04"
    ":00\", \"2026-10-03T05:00\", \"2026-10-03T06:00\", \"2026-10-03T07:00\","
    " \"2026-10-03T08:00\", \"2026-10-03T09:00\", \"2026-10-03T10:00\", \"202"
    "6-10-03T11:00\", \"2026-10-03T12:00\", \"2026-10-03T13:00\", \"2026-10-0"
    "3T14:00\", \"2026-10-03T15:00\", \"2026-10-03T16:00\", \"2026-10-03T17:0"
    "0\", \"2026-10-03T18:00\", \"2026-10-03T19:00\", \"2026-10-03T20:00\", "
    "\"2026-10-03T21:00\", \"2026-10-03T22:00\", \"2026-10-03T23:00\"], \"temp"
    "erature_2m\": [18.7, 18.2, 17.8, 17.5, 17.3, 17.0, 16.7, 16.5, 17.0, 20."
    "0, 23.4, 25.1, 26.1, 26.4, 26.3, 26.6, 26.3, 26.3, 25.8, 24.4, 23.1, 22."
    "1, 21.1, 19.7, 18.8, 18.4, 18.1, 17.7, 17.4, 17.2, 17.1, 16.9, 17.6, 20."
    "5, 23.7, 25.4, 26.7, 27.1, 26.6, 26.4, 26.2, 25.9, 25.5, 24.5, 23.3, 22."
    "2, 20.8, 19.6, 18.9, 18.5, 18.2, 17.9, 17.7, 17.6, 17.8, 17.8, 18.3, 20."
    "9, 23.4, 24.8, 25.7, 26.4, 25.9, 25.8, 25.7, 25.5, 25.0, 24.1, 23.0, 22."
    "0, 20.7, 19.7], \"precipitation_probability\": [0, 0, 0, 0, 0, 0, 0, 0, "
    "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
    "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
    "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \"precipitation\": [0.0"
    ", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
    "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0."
    "0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,"
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0"
    ".0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \""
    "weather_code\": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 3, 0, 0"
    ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 2, 1, 1, 0, 1"
    ", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 1, 2, 2, 3, 2, 2, 2, 2"
    ", 1, 1, 1, 1, 1], \"wind_speed_10m\": [5.6, 6.4, 6.3, 6.6, 6.5, 6.1, 6.1"
    ", 6.6, 6.1, 4.0, 2.6, 0.7, 4.5, 11.7, 13.8, 14.8, 13.8, 16.3, 14.3, 9.7,"
    " 6.5, 5.4, 4.2, 4.3, 5.1, 5.5, 6.2, 6.2, 6.1, 6.1, 6.5, 6.1, 5.8, 4.5, 3"
    ".5, 2.5, 3.0, 10.7, 13.6, 13.5, 11.7, 11.3, 10.4, 8.7, 6.0, 5.1, 5.3, 5."
    "1, 5.9, 5.9, 6.2, 5.8, 5.8, 5.8, 5.4, 5.8, 5.8, 4.8, 5.2, 2.5, 2.3, 7.4,"
    " 11.7, 11.8, 11.4, 10.1, 9.7, 7.8, 6.4, 5.4, 5.0, 5.4], \"wind_direction"
    "_10m\": [75, 74, 77, 81, 84, 90, 87, 81, 90, 100, 146, 180, 299, 313, 32"
    "7, 331, 354, 25, 45, 39, 34, 42, 59, 85, 82, 79, 80, 83, 90, 90, 87, 90,"
    " 90, 104, 156, 188, 284, 315, 310, 304, 326, 343, 358, 7, 17, 45, 62, 82"
    ", 79, 79, 80, 83, 90, 94, 94, 94, 94, 103, 146, 180, 288, 313, 313, 322,"
    " 349, 354, 2, 13, 27, 42, 60, 70], \"wind_gusts_10m\": [13.0, 15.1, 15.8"
    ", 16.2, 16.2, 15.8, 15.5, 15.1, 14.4, 13.0, 10.1, 9.7, 11.5, 24.8, 27.4,"
    " 28.1, 28.4, 36.4, 35.3, 29.9, 23.0, 19.8, 17.3, 14.8, 11.5, 11.5, 13.3,"
    " 14.0, 14.0, 14.0, 13.0, 13.0, 12.2, 11.2, 10.1, 13.0, 7.6, 18.4, 28.1, "
    "32.0, 27.0, 25.6, 23.4, 22.3, 19.4, 16.2, 13.7, 13.0, 13.0, 12.2, 13.0, "
    "13.0, 13.3, 12.6, 10.8, 10.1, 11.2, 11.5, 13.3, 14.4, 7.9, 16.6, 25.2, 2"
    "6.6, 25.6, 25.6, 21.6, 20.5, 19.1, 17.6, 15.1, 13.7]}, \"daily_units\": "
    "{\"time\": \"iso8601\", \"weather_code\": \"wmo code\", \"temperature_2m"
    "_max\": \"°C\", \"temperature_2m_min\": \"°C\", \"precipitation_sum\": "
    "\"mm\", \"precipitation_probability_max\": \"%\", \"wind_speed_10m_max\":"
    " \"km/h\", \"wind_gusts_10m_max\": \"km/h\", \"wind_direction_10m_domina"
    "nt\": \"°\", \"sunrise\": \"iso8601\", \"sunset\": \"iso8601\", \"uv_ind"
    "ex_max\": \"\"}, \"daily\": {\"time\": [\"2026-10-01\", \"2026-10-02\", "
    "\"2026-10-03\"], \"weather_code\": [3, 3, 3], \"temperature_2m_max\": [2"
    "6.6, 27.1, 26.4], \"temperature_2m_min\": [16.5, 16.9, 17.6], \"precipit"
    "ation_sum\": [0.0, 0.0, 0.0], \"precipitation_probability_max\": [0, 0, "
    "0], \"wind_speed_10m_max\": [16.3, 13.6, 11.8], \"wind_gusts_10m_max\": "
    "[36.4, 32.0, 26.6], \"wind_direction_10m_dominant\": [36, 24, 36], \"sun"
    "rise\": [\"2026-10-01T07:05\", \"2026-10-02T07:06\", \"2026-10-03T07:07"
    "\"], \"sunset\": [\"2026-10-01T18:52\", \"2026-10-02T18:51\", \"2026-10-0"
    "3T18:49\"], \"uv_index_max\": [5.9, 5.9, 5.8]}}";

static const char OM_MARINE[] =
    "[{\"latitude\": 39.541664, \"longitude\": 11.9583435, \"generationtime_m"
    "s\": 0.23055076599121094, \"utc_offset_seconds\": 7200, \"timezone\": \""
    "Europe/Rome\", \"timezone_abbreviation\": \"GMT+2\", \"elevation\": 0.0,"
    " \"current_units\": {\"time\": \"iso8601\", \"interval\": \"seconds\", "
    "\"wave_height\": \"m\", \"wave_direction\": \"°\", \"wave_period\": \"s\""
    ", \"wind_wave_height\": \"m\", \"swell_wave_height\": \"m\", \"swell_wav"
    "e_direction\": \"°\", \"sea_surface_temperature\": \"°C\"}, \"current\":"
    " {\"time\": \"2026-10-01T13:45\", \"interval\": 900, \"wave_height\": 0."
    "1, \"wave_direction\": 103, \"wave_period\": 2.95, \"wind_wave_height\":"
    " 0.0, \"swell_wave_height\": 0.08, \"swell_wave_direction\": 69, \"sea_s"
    "urface_temperature\": 26.7}, \"daily_units\": {\"time\": \"iso8601\", \""
    "wave_height_max\": \"m\", \"wave_direction_dominant\": \"°\", \"wave_per"
    "iod_max\": \"s\"}, \"daily\": {\"time\": [\"2026-10-01\", \"2026-10-02\""
    "], \"wave_height_max\": [0.14, 0.06], \"wave_direction_dominant\": [115,"
    " 151], \"wave_period_max\": [3.3, 2.95]}}, {\"latitude\": 40.541664, \"l"
    "ongitude\": 10.9583435, \"generationtime_ms\": 0.10025501251220703, \"ut"
    "c_offset_seconds\": 7200, \"timezone\": \"Europe/Rome\", \"timezone_abbr"
    "eviation\": \"GMT+2\", \"elevation\": 0.0, \"location_id\": 1, \"current"
    "_units\": {\"time\": \"iso8601\", \"interval\": \"seconds\", \"wave_heig"
    "ht\": \"m\", \"wave_direction\": \"°\", \"wave_period\": \"s\", \"wind_w"
    "ave_height\": \"m\", \"swell_wave_height\": \"m\", \"swell_wave_directio"
    "n\": \"°\", \"sea_surface_temperature\": \"°C\"}, \"current\": {\"time\""
    ": \"2026-10-01T13:45\", \"interval\": 900, \"wave_height\": 0.14, \"wave"
    "_direction\": 149, \"wave_period\": 3.4, \"wind_wave_height\": 0.0, \"sw"
    "ell_wave_height\": 0.12, \"swell_wave_direction\": 175, \"sea_surface_te"
    "mperature\": 26.8}, \"daily_units\": {\"time\": \"iso8601\", \"wave_heig"
    "ht_max\": \"m\", \"wave_direction_dominant\": \"°\", \"wave_period_max\""
    ": \"s\"}, \"daily\": {\"time\": [\"2026-10-01\", \"2026-10-02\"], \"wave"
    "_height_max\": [0.2, 0.12], \"wave_direction_dominant\": [159, 178], \"w"
    "ave_period_max\": [3.85, 3.7]}}, {\"latitude\": 38.041664, \"longitude\""
    ": 12.375015, \"generationtime_ms\": 0.09393692016601562, \"utc_offset_se"
    "conds\": 7200, \"timezone\": \"Europe/Rome\", \"timezone_abbreviation\":"
    " \"GMT+2\", \"elevation\": 10.0, \"location_id\": 2, \"current_units\": "
    "{\"time\": \"iso8601\", \"interval\": \"seconds\", \"wave_height\": \"m"
    "\", \"wave_direction\": \"°\", \"wave_period\": \"s\", \"wind_wave_height"
    "\": \"m\", \"swell_wave_height\": \"m\", \"swell_wave_direction\": \"°\""
    ", \"sea_surface_temperature\": \"°C\"}, \"current\": {\"time\": \"2026-1"
    "0-01T13:45\", \"interval\": 900, \"wave_height\": 0.08, \"wave_direction"
    "\": 70, \"wave_period\": 2.95, \"wind_wave_height\": 0.0, \"swell_wave_h"
    "eight\": 0.04, \"swell_wave_direction\": 40, \"sea_surface_temperature\""
    ": 24.9}, \"daily_units\": {\"time\": \"iso8601\", \"wave_height_max\": "
    "\"m\", \"wave_direction_dominant\": \"°\", \"wave_period_max\": \"s\"}, "
    "\"daily\": {\"time\": [\"2026-10-01\", \"2026-10-02\"], \"wave_height_max"
    "\": [0.34, 0.24], \"wave_direction_dominant\": [65, 55], \"wave_period_m"
    "ax\": [3.05, 2.85]}}]";

static const char METAR_BBOX[] =
    "[{\"icaoId\": \"LICT\", \"receiptTime\": \"2026-10-01T11:08:42.072Z\", "
    "\"obsTime\": 1790852100, \"reportTime\": \"2026-10-01T11:00:00.000Z\", \""
    "temp\": 26, \"dewp\": 19, \"wdir\": 340, \"wspd\": 8, \"visib\": \"6+\","
    " \"altim\": 1022, \"qcField\": 0, \"metarType\": \"METAR\", \"rawOb\": "
    "\"METAR LICT 011055Z 34008KT 9999 FEW015 SCT017 26/19 Q1022\", \"lat\": 3"
    "7.911, \"lon\": 12.488, \"elev\": 4, \"name\": \"Trapani/Birgi Arpt, TP,"
    " IT\", \"cover\": \"SCT\", \"clouds\": [{\"cover\": \"FEW\", \"base\": 1"
    "500}, {\"cover\": \"SCT\", \"base\": 1700}], \"fltCat\": \"VFR\"}]";

static const char MET_NORWAY[] =
    "{\"type\":\"Feature\",\"geometry\":{\"type\":\"Point\",\"coordinates\":["
    "12.51,38.02,2]},\"properties\":{\"meta\":{\"updated_at\":\"2026-10-01T11"
    ":26:50Z\",\"units\":{\"air_pressure_at_sea_level\":\"hPa\",\"air_tempera"
    "ture\":\"celsius\",\"cloud_area_fraction\":\"%\",\"precipitation_amount"
    "\":\"mm\",\"relative_humidity\":\"%\",\"wind_from_direction\":\"degrees\""
    ",\"wind_speed\":\"m/s\"}},\"timeseries\":[{\"time\":\"2026-10-01T11:00:0"
    "0Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":1"
    "023.4,\"air_temperature\":26.1,\"cloud_area_fraction\":31.2,\"relative_h"
    "umidity\":57.7,\"wind_from_direction\":307.4,\"wind_speed\":3.6}},\"next"
    "_12_hours\":{\"summary\":{\"symbol_code\":\"lightrainshowers_day\"},\"de"
    "tails\":{}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"rainshowers"
    "_day\"},\"details\":{\"precipitation_amount\":0.5}},\"next_6_hours\":{\""
    "summary\":{\"symbol_code\":\"lightrainshowers_day\"},\"details\":{\"prec"
    "ipitation_amount\":0.7}}}},{\"time\":\"2026-10-01T12:00:00Z\",\"data\":{"
    "\"instant\":{\"details\":{\"air_pressure_at_sea_level\":1023.1,\"air_tem"
    "perature\":25.6,\"cloud_area_fraction\":40.6,\"relative_humidity\":65.0,"
    "\"wind_from_direction\":327.2,\"wind_speed\":3.8}},\"next_12_hours\":{\""
    "summary\":{\"symbol_code\":\"clearsky_day\"},\"details\":{}},\"next_1_ho"
    "urs\":{\"summary\":{\"symbol_code\":\"lightrainshowers_day\"},\"details"
    "\":{\"precipitation_amount\":0.1}},\"next_6_hours\":{\"summary\":{\"symbo"
    "l_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.1}}}}"
    ",{\"time\":\"2026-10-01T13:00:00Z\",\"data\":{\"instant\":{\"details\":{"
    "\"air_pressure_at_sea_level\":1022.7,\"air_temperature\":24.7,\"cloud_ar"
    "ea_fraction\":11.7,\"relative_humidity\":65.7,\"wind_from_direction\":33"
    "6.0,\"wind_speed\":4.2}},\"next_12_hours\":{\"summary\":{\"symbol_code\""
    ":\"clearsky_day\"},\"details\":{}},\"next_1_hours\":{\"summary\":{\"symb"
    "ol_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0}},"
    "\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"detai"
    "ls\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-01T14:00:00Z"
    "\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":1022"
    ".5,\"air_temperature\":24.3,\"cloud_area_fraction\":3.1,\"relative_humid"
    "ity\":61.4,\"wind_from_direction\":324.0,\"wind_speed\":4.1}},\"next_12_"
    "hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{}"
    "},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"det"
    "ails\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\""
    "symbol_code\":\"clearsky_night\"},\"details\":{\"precipitation_amount\":"
    "0.0}}}},{\"time\":\"2026-10-01T15:00:00Z\",\"data\":{\"instant\":{\"deta"
    "ils\":{\"air_pressure_at_sea_level\":1022.4,\"air_temperature\":24.4,\"c"
    "loud_area_fraction\":1.6,\"relative_humidity\":60.5,\"wind_from_directio"
    "n\":336.5,\"wind_speed\":3.5}},\"next_12_hours\":{\"summary\":{\"symbol_"
    "code\":\"clearsky_night\"},\"details\":{}},\"next_1_hours\":{\"summary\""
    ":{\"symbol_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount"
    "\":0.0}},\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night"
    "\"},\"details\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-01T"
    "16:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_le"
    "vel\":1022.5,\"air_temperature\":24.1,\"cloud_area_fraction\":0.0,\"rela"
    "tive_humidity\":57.5,\"wind_from_direction\":7.2,\"wind_speed\":4.7}},\""
    "next_12_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"deta"
    "ils\":{}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day"
    "\"},\"details\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"summa"
    "ry\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\"precipitation_a"
    "mount\":0.0}}}},{\"time\":\"2026-10-01T17:00:00Z\",\"data\":{\"instant\""
    ":{\"details\":{\"air_pressure_at_sea_level\":1022.7,\"air_temperature\":"
    "24.0,\"cloud_area_fraction\":0.0,\"relative_humidity\":57.7,\"wind_from_"
    "direction\":30.6,\"wind_speed\":5.2}},\"next_12_hours\":{\"summary\":{\""
    "symbol_code\":\"clearsky_night\"},\"details\":{}},\"next_1_hours\":{\"su"
    "mmary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\"precipitatio"
    "n_amount\":0.0}},\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clears"
    "ky_night\"},\"details\":{\"precipitation_amount\":0.0}}}},{\"time\":\"20"
    "26-10-01T18:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_"
    "at_sea_level\":1023.2,\"air_temperature\":23.9,\"cloud_area_fraction\":0"
    ".0,\"relative_humidity\":58.4,\"wind_from_direction\":46.5,\"wind_speed"
    "\":5.1}},\"next_12_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night"
    "\"},\"details\":{}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"cle"
    "arsky_night\"},\"details\":{\"precipitation_amount\":0.0}},\"next_6_hour"
    "s\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\"pre"
    "cipitation_amount\":0.0}}}},{\"time\":\"2026-10-01T19:00:00Z\",\"data\":"
    "{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":1023.7,\"air_te"
    "mperature\":23.6,\"cloud_area_fraction\":0.0,\"relative_humidity\":59.6,"
    "\"wind_from_direction\":56.9,\"wind_speed\":5.1}},\"next_12_hours\":{\"s"
    "ummary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{}},\"next_1_h"
    "ours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\""
    "precipitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\"symbol_cod"
    "e\":\"clearsky_night\"},\"details\":{\"precipitation_amount\":0.0}}}},{"
    "\"time\":\"2026-10-01T20:00:00Z\",\"data\":{\"instant\":{\"details\":{\"a"
    "ir_pressure_at_sea_level\":1024.1,\"air_temperature\":23.2,\"cloud_area_"
    "fraction\":0.0,\"relative_humidity\":62.6,\"wind_from_direction\":65.3,"
    "\"wind_speed\":4.8}},\"next_12_hours\":{\"summary\":{\"symbol_code\":\"cl"
    "earsky_night\"},\"details\":{}},\"next_1_hours\":{\"summary\":{\"symbol_"
    "code\":\"clearsky_night\"},\"details\":{\"precipitation_amount\":0.0}},"
    "\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"deta"
    "ils\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-01T21:00:00Z"
    "\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":102"
    "4.3,\"air_temperature\":22.6,\"cloud_area_fraction\":0.0,\"relative_humi"
    "dity\":65.8,\"wind_from_direction\":75.2,\"wind_speed\":4.6}},\"next_12_"
    "hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"details\":{}},"
    "\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"det"
    "ails\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\""
    "symbol_code\":\"clearsky_night\"},\"details\":{\"precipitation_amount\":"
    "0.0}}}},{\"time\":\"2026-10-01T22:00:00Z\",\"data\":{\"instant\":{\"deta"
    "ils\":{\"air_pressure_at_sea_level\":1024.4,\"air_temperature\":22.1,\"c"
    "loud_area_fraction\":0.0,\"relative_humidity\":68.5,\"wind_from_directio"
    "n\":80.6,\"wind_speed\":4.2}},\"next_12_hours\":{\"summary\":{\"symbol_c"
    "ode\":\"clearsky_day\"},\"details\":{}},\"next_1_hours\":{\"summary\":{"
    "\"symbol_code\":\"clearsky_night\"},\"details\":{\"precipitation_amount\""
    ":0.0}},\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\""
    "},\"details\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-01T2"
    "3:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_lev"
    "el\":1024.3,\"air_temperature\":21.8,\"cloud_area_fraction\":0.0,\"relat"
    "ive_humidity\":72.4,\"wind_from_direction\":82.0,\"wind_speed\":3.9}},\""
    "next_12_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"detail"
    "s\":{}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night"
    "\"},\"details\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"summa"
    "ry\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\"precipitation_a"
    "mount\":0.0}}}},{\"time\":\"2026-10-02T00:00:00Z\",\"data\":{\"instant\""
    ":{\"details\":{\"air_pressure_at_sea_level\":1024.3,\"air_temperature\":"
    "21.5,\"cloud_area_fraction\":0.0,\"relative_humidity\":73.8,\"wind_from_"
    "direction\":86.9,\"wind_speed\":4.0}},\"next_12_hours\":{\"summary\":{\""
    "symbol_code\":\"clearsky_day\"},\"details\":{}},\"next_1_hours\":{\"summ"
    "ary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\"precipitation_"
    "amount\":0.0}},\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky"
    "_night\"},\"details\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026"
    "-10-02T01:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at"
    "_sea_level\":1024.5,\"air_temperature\":21.1,\"cloud_area_fraction\":2.3"
    ",\"relative_humidity\":74.6,\"wind_from_direction\":96.3,\"wind_speed\":"
    "3.9}},\"next_12_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},"
    "\"details\":{}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsk"
    "y_night\"},\"details\":{\"precipitation_amount\":0.0}},\"next_6_hours\":"
    "{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\"precipi"
    "tation_amount\":0.0}}}},{\"time\":\"2026-10-02T02:00:00Z\",\"data\":{\"i"
    "nstant\":{\"details\":{\"air_pressure_at_sea_level\":1024.4,\"air_temper"
    "ature\":20.8,\"cloud_area_fraction\":1.6,\"relative_humidity\":75.2,\"wi"
    "nd_from_direction\":103.7,\"wind_speed\":3.6}},\"next_12_hours\":{\"summ"
    "ary\":{\"symbol_code\":\"clearsky_day\"},\"details\":{}},\"next_1_hours"
    "\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\"preci"
    "pitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\"symbol_code\":"
    "\"clearsky_night\"},\"details\":{\"precipitation_amount\":0.0}}}},{\"time"
    "\":\"2026-10-02T03:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pr"
    "essure_at_sea_level\":1024.5,\"air_temperature\":20.4,\"cloud_area_fract"
    "ion\":3.1,\"relative_humidity\":77.3,\"wind_from_direction\":109.8,\"win"
    "d_speed\":3.4}},\"next_12_hours\":{\"summary\":{\"symbol_code\":\"clears"
    "ky_day\"},\"details\":{}},\"next_1_hours\":{\"summary\":{\"symbol_code\""
    ":\"clearsky_night\"},\"details\":{\"precipitation_amount\":0.0}},\"next_"
    "6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"details\":{"
    "\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-02T04:00:00Z\",\"dat"
    "a\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":1024.6,\"ai"
    "r_temperature\":20.1,\"cloud_area_fraction\":3.9,\"relative_humidity\":8"
    "0.3,\"wind_from_direction\":116.8,\"wind_speed\":3.0}},\"next_12_hours\""
    ":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"details\":{}},\"next_"
    "1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"details\":"
    "{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\"symbol_"
    "code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0}}}},{"
    "\"time\":\"2026-10-02T05:00:00Z\",\"data\":{\"instant\":{\"details\":{\""
    "air_pressure_at_sea_level\":1024.8,\"air_temperature\":19.8,\"cloud_area"
    "_fraction\":5.5,\"relative_humidity\":82.7,\"wind_from_direction\":122.6"
    ",\"wind_speed\":3.0}},\"next_12_hours\":{\"summary\":{\"symbol_code\":\""
    "clearsky_day\"},\"details\":{}},\"next_1_hours\":{\"summary\":{\"symbol_"
    "code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0}},\"n"
    "ext_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"details"
    "\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-02T06:00:00Z\","
    "\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":1025.3,"
    "\"air_temperature\":20.3,\"cloud_area_fraction\":3.9,\"relative_humidity"
    "\":79.6,\"wind_from_direction\":126.0,\"wind_speed\":2.9}},\"next_12_hou"
    "rs\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"details\":{}},\"n"
    "ext_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"details"
    "\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\"symbo"
    "l_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0}}}}"
    ",{\"time\":\"2026-10-02T07:00:00Z\",\"data\":{\"instant\":{\"details\":{"
    "\"air_pressure_at_sea_level\":1025.9,\"air_temperature\":22.5,\"cloud_ar"
    "ea_fraction\":6.2,\"relative_humidity\":63.4,\"wind_from_direction\":134"
    ".1,\"wind_speed\":2.7}},\"next_12_hours\":{\"summary\":{\"symbol_code\":"
    "\"clearsky_day\"},\"details\":{}},\"next_1_hours\":{\"summary\":{\"symbo"
    "l_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0}},"
    "\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"detail"
    "s\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-02T08:00:00Z\""
    ",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":1026."
    "2,\"air_temperature\":24.1,\"cloud_area_fraction\":0.8,\"relative_humidi"
    "ty\":56.1,\"wind_from_direction\":143.3,\"wind_speed\":2.1}},\"next_12_h"
    "ours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"details\":{}},"
    "\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"detail"
    "s\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\"sym"
    "bol_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0}}"
    "}},{\"time\":\"2026-10-02T09:00:00Z\",\"data\":{\"instant\":{\"details\""
    ":{\"air_pressure_at_sea_level\":1026.4,\"air_temperature\":25.2,\"cloud_"
    "area_fraction\":4.7,\"relative_humidity\":52.5,\"wind_from_direction\":2"
    "66.7,\"wind_speed\":1.9}},\"next_12_hours\":{\"summary\":{\"symbol_code"
    "\":\"clearsky_day\"},\"details\":{}},\"next_1_hours\":{\"summary\":{\"sym"
    "bol_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0}}"
    ",\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"deta"
    "ils\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-02T10:00:00Z"
    "\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":102"
    "6.4,\"air_temperature\":25.9,\"cloud_area_fraction\":0.0,\"relative_humi"
    "dity\":54.1,\"wind_from_direction\":300.2,\"wind_speed\":2.6}},\"next_12"
    "_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"details\":{}}"
    ",\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"deta"
    "ils\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\"s"
    "ymbol_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0"
    "}}}},{\"time\":\"2026-10-02T11:00:00Z\",\"data\":{\"instant\":{\"details"
    "\":{\"air_pressure_at_sea_level\":1026.4,\"air_temperature\":26.0,\"clou"
    "d_area_fraction\":3.9,\"relative_humidity\":57.3,\"wind_from_direction\""
    ":315.1,\"wind_speed\":3.5}},\"next_12_hours\":{\"summary\":{\"symbol_cod"
    "e\":\"clearsky_day\"},\"details\":{}},\"next_1_hours\":{\"summary\":{\"s"
    "ymbol_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0"
    "}},\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"de"
    "tails\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-02T12:00:0"
    "0Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":1"
    "026.1,\"air_temperature\":25.4,\"cloud_area_fraction\":17.2,\"relative_h"
    "umidity\":59.0,\"wind_from_direction\":326.7,\"wind_speed\":3.7}},\"next"
    "_12_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"details\":"
    "{}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"fair_day\"},\"detai"
    "ls\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\"sy"
    "mbol_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0}"
    "}}},{\"time\":\"2026-10-02T13:00:00Z\",\"data\":{\"instant\":{\"details"
    "\":{\"air_pressure_at_sea_level\":1025.6,\"air_temperature\":25.2,\"cloud"
    "_area_fraction\":10.9,\"relative_humidity\":57.9,\"wind_from_direction\""
    ":324.1,\"wind_speed\":4.0}},\"next_12_hours\":{\"summary\":{\"symbol_cod"
    "e\":\"clearsky_day\"},\"details\":{}},\"next_1_hours\":{\"summary\":{\"s"
    "ymbol_code\":\"clearsky_day\"},\"details\":{\"precipitation_amount\":0.0"
    "}},\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\"de"
    "tails\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-02T14:00:0"
    "0Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":1"
    "025.4,\"air_temperature\":25.1,\"cloud_area_fraction\":9.4,\"relative_hu"
    "midity\":59.7,\"wind_from_direction\":324.8,\"wind_speed\":3.7}},\"next_"
    "12_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"details\""
    ":{}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_day\"},\""
    "details\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"summary\":"
    "{\"symbol_code\":\"clearsky_night\"},\"details\":{\"precipitation_amount"
    "\":0.0}}}},{\"time\":\"2026-10-02T15:00:00Z\",\"data\":{\"instant\":{\"d"
    "etails\":{\"air_pressure_at_sea_level\":1025.1,\"air_temperature\":24.7,"
    "\"cloud_area_fraction\":7.0,\"relative_humidity\":61.7,\"wind_from_direc"
    "tion\":343.4,\"wind_speed\":3.8}},\"next_12_hours\":{\"summary\":{\"symb"
    "ol_code\":\"clearsky_night\"},\"details\":{}},\"next_1_hours\":{\"summar"
    "y\":{\"symbol_code\":\"clearsky_day\"},\"details\":{\"precipitation_amou"
    "nt\":0.0}},\"next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_nig"
    "ht\"},\"details\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-"
    "02T16:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea"
    "_level\":1025.0,\"air_temperature\":24.3,\"cloud_area_fraction\":3.9,\"r"
    "elative_humidity\":60.5,\"wind_from_direction\":9.3,\"wind_speed\":4.1}}"
    ",\"next_12_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"d"
    "etails\":{}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_d"
    "ay\"},\"details\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{\"su"
    "mmary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\"precipitatio"
    "n_amount\":0.0}}}},{\"time\":\"2026-10-03T13:00:00Z\",\"data\":{\"instan"
    "t\":{\"details\":{\"air_pressure_at_sea_level\":1025.6,\"air_temperature"
    "\":25.0,\"cloud_area_fraction\":27.3,\"relative_humidity\":68.4,\"wind_f"
    "rom_direction\":267.0,\"wind_speed\":3.5}},\"next_1_hours\":{\"summary\""
    ":{\"symbol_code\":\"lightrainshowers_day\"},\"details\":{\"precipitation"
    "_amount\":0.2}},\"next_6_hours\":{\"summary\":{\"symbol_code\":\"fair_da"
    "y\"},\"details\":{\"precipitation_amount\":0.4}}}},{\"time\":\"2026-10-0"
    "3T14:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_"
    "level\":1025.5,\"air_temperature\":24.8,\"cloud_area_fraction\":20.3,\"r"
    "elative_humidity\":66.7,\"wind_from_direction\":276.5,\"wind_speed\":2.7"
    "}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"lightrainshowers_day"
    "\"},\"details\":{\"precipitation_amount\":0.1}},\"next_6_hours\":{\"summ"
    "ary\":{\"symbol_code\":\"fair_night\"},\"details\":{\"precipitation_amou"
    "nt\":0.2}}}},{\"time\":\"2026-10-03T15:00:00Z\",\"data\":{\"instant\":{"
    "\"details\":{\"air_pressure_at_sea_level\":1025.3,\"air_temperature\":24."
    "2,\"cloud_area_fraction\":27.3,\"relative_humidity\":67.4,\"wind_from_di"
    "rection\":287.6,\"wind_speed\":2.4}},\"next_1_hours\":{\"summary\":{\"sy"
    "mbol_code\":\"fair_day\"},\"details\":{\"precipitation_amount\":0.0}},\""
    "next_6_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"detai"
    "ls\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-03T16:00:00Z"
    "\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":1025"
    ".1,\"air_temperature\":23.6,\"cloud_area_fraction\":18.7,\"relative_humi"
    "dity\":69.2,\"wind_from_direction\":305.9,\"wind_speed\":2.1}},\"next_1_"
    "hours\":{\"summary\":{\"symbol_code\":\"fair_day\"},\"details\":{\"preci"
    "pitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\"symbol_code\":"
    "\"clearsky_night\"},\"details\":{\"precipitation_amount\":0.0}}}},{\"time"
    "\":\"2026-10-03T17:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pr"
    "essure_at_sea_level\":1025.1,\"air_temperature\":22.8,\"cloud_area_fract"
    "ion\":11.7,\"relative_humidity\":70.4,\"wind_from_direction\":7.2,\"wind"
    "_speed\":1.9}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky"
    "_night\"},\"details\":{\"precipitation_amount\":0.0}},\"next_6_hours\":{"
    "\"summary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\"precipit"
    "ation_amount\":0.0}}}},{\"time\":\"2026-10-03T18:00:00Z\",\"data\":{\"in"
    "stant\":{\"details\":{\"air_pressure_at_sea_level\":1025.4,\"air_tempera"
    "ture\":22.7,\"cloud_area_fraction\":9.4,\"relative_humidity\":70.3,\"win"
    "d_from_direction\":42.9,\"wind_speed\":1.5}},\"next_12_hours\":{\"summar"
    "y\":{\"symbol_code\":\"clearsky_night\"},\"details\":{}},\"next_1_hours"
    "\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{\"preci"
    "pitation_amount\":0.0}},\"next_6_hours\":{\"summary\":{\"symbol_code\":"
    "\"clearsky_night\"},\"details\":{\"precipitation_amount\":0.0}}}},{\"time"
    "\":\"2026-10-03T19:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pr"
    "essure_at_sea_level\":1025.7,\"air_temperature\":22.7,\"cloud_area_fract"
    "ion\":9.4,\"relative_humidity\":68.5,\"wind_from_direction\":79.5,\"wind"
    "_speed\":1.4}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky"
    "_night\"},\"details\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026"
    "-10-03T20:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at"
    "_sea_level\":1025.9,\"air_temperature\":22.7,\"cloud_area_fraction\":4.7"
    ",\"relative_humidity\":66.6,\"wind_from_direction\":143.6,\"wind_speed\""
    ":1.3}},\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\""
    "},\"details\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-03T2"
    "1:00:00Z\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_lev"
    "el\":1025.9,\"air_temperature\":22.5,\"cloud_area_fraction\":0.8,\"relat"
    "ive_humidity\":66.7,\"wind_from_direction\":168.8,\"wind_speed\":1.2}},"
    "\"next_1_hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"deta"
    "ils\":{\"precipitation_amount\":0.0}}}},{\"time\":\"2026-10-03T22:00:00Z"
    "\",\"data\":{\"instant\":{\"details\":{\"air_pressure_at_sea_level\":102"
    "5.7,\"air_temperature\":22.2,\"cloud_area_fraction\":0.0,\"relative_humi"
    "dity\":68.1,\"wind_from_direction\":171.0,\"wind_speed\":1.2}},\"next_1_"
    "hours\":{\"summary\":{\"symbol_code\":\"clearsky_night\"},\"details\":{"
    "\"precipitation_amount\":0.0}}}}]}}";

static const char METEOALARM[] =
    "{\"warnings\":[{\"alert\":{\"identifier\":\"2.49.0.0.380.3.IT.2609261008"
    "57.001\",\"incidents\":\"Update\",\"info\":[{\"area\":[{\"areaDesc\":\"C"
    "ampania\",\"geocode\":[{\"value\":\"IT016\",\"valueName\":\"EMMA_ID\"}]}"
    "],\"audience\":\"Private\",\"category\":[\"Met\"],\"certainty\":\"Likely"
    "\",\"description\":\"No Special Awareness Required\\n (DISCLAIMER: \\\"I"
    "nformation provided on METEOALARM for Italy regard only the intensity an"
    "d recurrence of the phenomena, further details can be found at www.meteo"
    "am.it. METEOALARM information do not provide the assessment of impact on"
    " the territory and they do not represent the Official Alerts messages th"
    "at are issued by the National Civil Protection Service https://www.prote"
    "zionecivile.gov.it\\\")\",\"effective\":\"2026-09-26T10:08:00+02:00\",\""
    "event\":\"Green Wind Warning\",\"expires\":\"2026-09-28T23:59:00+02:00\""
    ",\"headline\":\"Green Wind Warning for Italy - Campania\",\"instruction"
    "\":\"The weather is not expected to cause significant impacts but there m"
    "ay be some minor, localized issues\",\"language\":\"en-GB\",\"onset\":\""
    "2026-09-26T10:08:00+02:00\",\"parameter\":[{\"value\":\"1; green; Minor"
    "\",\"valueName\":\"awareness_level\"},{\"value\":\"1; Wind\",\"valueName"
    "\":\"awareness_type\"}],\"responseType\":[\"None\"],\"senderName\":\"Ital"
    "ian Air Force National Meteorological Service\",\"severity\":\"Minor\","
    "\"urgency\":\"Future\",\"web\":\"https://meteoalarm.org/en/live/region/IT"
    "?s=campania\"},{\"area\":[{\"areaDesc\":\"Campania\",\"geocode\":[{\"val"
    "ue\":\"IT016\",\"valueName\":\"EMMA_ID\"}]}],\"audience\":\"Privato\",\""
    "category\":[\"Met\"],\"certainty\":\"Likely\",\"description\":\"Nessun A"
    "vviso\\n (DISCLAIMER: \\\"Le informazioni fornite su METEOALARM per l'It"
    "alia riguardano esclusivamente l'intensità e la ricorrenza dei fenomeni,"
    " maggiori dettagli sono disponibili su www.meteoam.it. Le informazioni M"
    "ETEOALARM non forniscono la valutazione di impatto sul territorio e non "
    "rappresentano i messaggi di Allerta Ufficiali che vengono emessi dal Ser"
    "vizio Nazionale di Protezione Civile https://www.protezionecivile.gov.it"
    "\\\")\",\"effective\":\"2026-09-26T10:08:00+02:00\",\"event\":\"Verde Ve"
    "nto Allerta\",\"expires\":\"2026-09-28T23:59:00+02:00\",\"headline\":\"A"
    "llerta Vento Verde per l'Italia - Campania\",\"instruction\":\"Non ci si"
    " aspetta che il tempo possa causare impatti significativi, ma potrebbero"
    " verificarsi alcuni locali, minori problemi.\",\"language\":\"it-IT\",\""
    "onset\":\"2026-09-26T10:08:00+02:00\",\"parameter\":[{\"value\":\"1; gre"
    "en; Minor\",\"valueName\":\"awareness_level\"},{\"value\":\"1; Wind\",\""
    "valueName\":\"awareness_type\"}],\"responseType\":[\"None\"],\"senderNam"
    "e\":\"Servizio Meteorologico dell'Aeronautica Militare\",\"severity\":\""
    "Minor\",\"urgency\":\"Future\",\"web\":\"https://meteoalarm.org/it/live/"
    "region/IT?s=campania\"}],\"msgType\":\"Update\",\"references\":\"aerocnm"
    "ca.1sv.prv1@aeronautica.difesa.it,2.49.0.0.380.3.IT.260925102045.029,202"
    "6-09-25T10:20:46+02:00\",\"scope\":\"Public\",\"sender\":\"aerocnmca.1sv"
    ".prv1@aeronautica.difesa.it\",\"sent\":\"2026-09-26T10:08:57+02:00\",\"s"
    "tatus\":\"Actual\"},\"uuid\":\"e4d58b0f-9b7c-4864-8321-5cd50035dfc2\"},{"
    "\"alert\":{\"identifier\":\"2.49.0.0.380.3.IT.260926100857.005\",\"incid"
    "ents\":\"Update\",\"info\":[{\"area\":[{\"areaDesc\":\"Sicilia\",\"geoco"
    "de\":[{\"value\":\"IT018\",\"valueName\":\"EMMA_ID\"}]}],\"audience\":\""
    "Private\",\"category\":[\"Met\"],\"certainty\":\"Likely\",\"description"
    "\":\"No Special Awareness Required\\n (DISCLAIMER: \\\"Information provid"
    "ed on METEOALARM for Italy regard only the intensity and recurrence of t"
    "he phenomena, further details can be found at www.meteoam.it. METEOALARM"
    " information do not provide the assessment of impact on the territory an"
    "d they do not represent the Official Alerts messages that are issued by "
    "the National Civil Protection Service https://www.protezionecivile.gov.i"
    "t\\\")\",\"effective\":\"2026-09-26T10:08:00+02:00\",\"event\":\"Green T"
    "hunderstorm Warning\",\"expires\":\"2026-09-28T23:59:00+02:00\",\"headli"
    "ne\":\"Green Thunderstorm Warning for Italy - Sicilia\",\"instruction\":"
    "\"The weather is not expected to cause significant impacts but there may"
    " be some minor, localized issues\",\"language\":\"en-GB\",\"onset\":\"20"
    "26-09-26T10:08:00+02:00\",\"parameter\":[{\"value\":\"1; green; Minor\","
    "\"valueName\":\"awareness_level\"},{\"value\":\"3; Thunderstorm\",\"valu"
    "eName\":\"awareness_type\"}],\"responseType\":[\"None\"],\"senderName\":"
    "\"Italian Air Force National Meteorological Service\",\"severity\":\"Min"
    "or\",\"urgency\":\"Future\",\"web\":\"https://meteoalarm.org/en/live/reg"
    "ion/IT?s=sicilia\"},{\"area\":[{\"areaDesc\":\"Sicilia\",\"geocode\":[{"
    "\"value\":\"IT018\",\"valueName\":\"EMMA_ID\"}]}],\"audience\":\"Privato"
    "\",\"category\":[\"Met\"],\"certainty\":\"Likely\",\"description\":\"Ness"
    "un Avviso\\n (DISCLAIMER: \\\"Le informazioni fornite su METEOALARM per "
    "l'Italia riguardano esclusivamente l'intensità e la ricorrenza dei fenom"
    "eni, maggiori dettagli sono disponibili su www.meteoam.it. Le informazio"
    "ni METEOALARM non forniscono la valutazione di impatto sul territorio e "
    "non rappresentano i messaggi di Allerta Ufficiali che vengono emessi dal"
    " Servizio Nazionale di Protezione Civile https://www.protezionecivile.go"
    "v.it\\\")\",\"effective\":\"2026-09-26T10:08:00+02:00\",\"event\":\"Verd"
    "e Temporali Allerta\",\"expires\":\"2026-09-28T23:59:00+02:00\",\"headli"
    "ne\":\"Allerta Temporali Verde per l'Italia - Sicilia\",\"instruction\":"
    "\"Non ci si aspetta che il tempo possa causare impatti significativi, ma"
    " potrebbero verificarsi alcuni locali, minori problemi.\",\"language\":"
    "\"it-IT\",\"onset\":\"2026-09-26T10:08:00+02:00\",\"parameter\":[{\"value"
    "\":\"1; green; Minor\",\"valueName\":\"awareness_level\"},{\"value\":\"3"
    "; Thunderstorm\",\"valueName\":\"awareness_type\"}],\"responseType\":[\""
    "None\"],\"senderName\":\"Servizio Meteorologico dell'Aeronautica Militar"
    "e\",\"severity\":\"Minor\",\"urgency\":\"Future\",\"web\":\"https://mete"
    "oalarm.org/it/live/region/IT?s=sicilia\"}],\"msgType\":\"Update\",\"refe"
    "rences\":\"aerocnmca.1sv.prv1@aeronautica.difesa.it,2.49.0.0.380.3.IT.26"
    "0925102045.033,2026-09-25T10:20:46+02:00\",\"scope\":\"Public\",\"sender"
    "\":\"aerocnmca.1sv.prv1@aeronautica.difesa.it\",\"sent\":\"2026-09-26T10"
    ":08:57+02:00\",\"status\":\"Actual\"},\"uuid\":\"4930e664-424b-4a89-bd27"
    "-3caf7df9de11\"},{\"alert\":{\"identifier\":\"2.49.0.0.380.3.IT.26093010"
    "4144.002\",\"incidents\":\"Update\",\"info\":[{\"area\":[{\"areaDesc\":"
    "\"Piemonte\",\"geocode\":[{\"value\":\"IT005\",\"valueName\":\"EMMA_ID\"}"
    "]}],\"audience\":\"Private\",\"category\":[\"Met\"],\"certainty\":\"Like"
    "ly\",\"description\":\"No Special Awareness Required\\n (DISCLAIMER: \\"
    "\"Information provided on METEOALARM for Italy regard only the intensity "
    "and recurrence of the phenomena, further details can be found at www.met"
    "eoam.it. METEOALARM information do not provide the assessment of impact "
    "on the territory and they do not represent the Official Alerts messages "
    "that are issued by the National Civil Protection Service https://www.pro"
    "tezionecivile.gov.it\\\")\",\"effective\":\"2026-09-30T10:41:00+02:00\","
    "\"event\":\"Green Thunderstorm Warning\",\"expires\":\"2026-10-02T23:59:"
    "00+02:00\",\"headline\":\"Green Thunderstorm Warning for Italy - Piemont"
    "e\",\"instruction\":\"The weather is not expected to cause significant i"
    "mpacts but there may be some minor, localized issues\",\"language\":\"en"
    "-GB\",\"onset\":\"2026-09-30T10:41:00+02:00\",\"parameter\":[{\"value\":"
    "\"3; orange; Severe\",\"valueName\":\"awareness_level\"},{\"value\":\"3;"
    " Thunderstorm\",\"valueName\":\"awareness_type\"}],\"responseType\":[\"N"
    "one\"],\"senderName\":\"Italian Air Force National Meteorological Servic"
    "e\",\"severity\":\"Minor\",\"urgency\":\"Future\",\"web\":\"https://mete"
    "oalarm.org/en/live/region/IT?s=piemonte\"},{\"area\":[{\"areaDesc\":\"Pi"
    "emonte\",\"geocode\":[{\"value\":\"IT005\",\"valueName\":\"EMMA_ID\"}]}]"
    ",\"audience\":\"Privato\",\"category\":[\"Met\"],\"certainty\":\"Likely"
    "\",\"description\":\"Nessun Avviso\\n (DISCLAIMER: \\\"Le informazioni fo"
    "rnite su METEOALARM per l'Italia riguardano esclusivamente l'intensità e"
    " la ricorrenza dei fenomeni, maggiori dettagli sono disponibili su www.m"
    "eteoam.it. Le informazioni METEOALARM non forniscono la valutazione di i"
    "mpatto sul territorio e non rappresentano i messaggi di Allerta Ufficial"
    "i che vengono emessi dal Servizio Nazionale di Protezione Civile https:/"
    "/www.protezionecivile.gov.it\\\")\",\"effective\":\"2026-09-30T10:41:00+"
    "02:00\",\"event\":\"Verde Temporali Allerta\",\"expires\":\"2026-10-02T2"
    "3:59:00+02:00\",\"headline\":\"Allerta Temporali Verde per l'Italia - Pi"
    "emonte\",\"instruction\":\"Non ci si aspetta che il tempo possa causare "
    "impatti significativi, ma potrebbero verificarsi alcuni locali, minori p"
    "roblemi.\",\"language\":\"it-IT\",\"onset\":\"2026-09-30T10:41:00+02:00"
    "\",\"parameter\":[{\"value\":\"3; orange; Severe\",\"valueName\":\"awaren"
    "ess_level\"},{\"value\":\"3; Thunderstorm\",\"valueName\":\"awareness_ty"
    "pe\"}],\"responseType\":[\"None\"],\"senderName\":\"Servizio Meteorologi"
    "co dell'Aeronautica Militare\",\"severity\":\"Minor\",\"urgency\":\"Futu"
    "re\",\"web\":\"https://meteoalarm.org/it/live/region/IT?s=piemonte\"}],"
    "\"msgType\":\"Update\",\"references\":\"aerocnmca.1sv.prv1@aeronautica.di"
    "fesa.it,2.49.0.0.380.3.IT.260929102223.003,2026-09-29T10:22:23+02:00\","
    "\"scope\":\"Public\",\"sender\":\"aerocnmca.1sv.prv1@aeronautica.difesa.i"
    "t\",\"sent\":\"2026-09-30T10:41:44+02:00\",\"status\":\"Actual\"},\"uuid"
    "\":\"089288e8-1557-4372-ae0b-f714d76eca74\"},{\"alert\":{\"identifier\":"
    "\"2.49.0.0.380.3.IT.260926100857.005\",\"incidents\":\"Update\",\"info\""
    ":[{\"area\":[{\"areaDesc\":\"Sicilia\",\"geocode\":[{\"value\":\"IT018\""
    ",\"valueName\":\"EMMA_ID\"}]}],\"audience\":\"Private\",\"category\":[\""
    "Met\"],\"certainty\":\"Likely\",\"description\":\"No Special Awareness R"
    "equired\\n (DISCLAIMER: \\\"Information provided on METEOALARM for Italy"
    " regard only the intensity and recurrence of the phenomena, further deta"
    "ils can be found at www.meteoam.it. METEOALARM information do not provid"
    "e the assessment of impact on the territory and they do not represent th"
    "e Official Alerts messages that are issued by the National Civil Protect"
    "ion Service https://www.protezionecivile.gov.it\\\")\",\"effective\":\"2"
    "026-09-26T10:08:00+02:00\",\"event\":\"Yellow Thunderstorm Warning\",\"e"
    "xpires\":\"2026-10-02T23:59:00+02:00\",\"headline\":\"Green Thunderstorm"
    " Warning for Italy - Sicilia\",\"instruction\":\"The weather is not expe"
    "cted to cause significant impacts but there may be some minor, localized"
    " issues\",\"language\":\"en-GB\",\"onset\":\"2026-09-26T10:08:00+02:00\""
    ",\"parameter\":[{\"value\":\"2; yellow; Moderate\",\"valueName\":\"aware"
    "ness_level\"},{\"value\":\"3; Thunderstorm\",\"valueName\":\"awareness_t"
    "ype\"}],\"responseType\":[\"None\"],\"senderName\":\"Italian Air Force N"
    "ational Meteorological Service\",\"severity\":\"Minor\",\"urgency\":\"Fu"
    "ture\",\"web\":\"https://meteoalarm.org/en/live/region/IT?s=sicilia\"},{"
    "\"area\":[{\"areaDesc\":\"Sicilia\",\"geocode\":[{\"value\":\"IT018\",\""
    "valueName\":\"EMMA_ID\"}]}],\"audience\":\"Privato\",\"category\":[\"Met"
    "\"],\"certainty\":\"Likely\",\"description\":\"Nessun Avviso\\n (DISCLAI"
    "MER: \\\"Le informazioni fornite su METEOALARM per l'Italia riguardano e"
    "sclusivamente l'intensità e la ricorrenza dei fenomeni, maggiori dettagl"
    "i sono disponibili su www.meteoam.it. Le informazioni METEOALARM non for"
    "niscono la valutazione di impatto sul territorio e non rappresentano i m"
    "essaggi di Allerta Ufficiali che vengono emessi dal Servizio Nazionale d"
    "i Protezione Civile https://www.protezionecivile.gov.it\\\")\",\"effecti"
    "ve\":\"2026-09-26T10:08:00+02:00\",\"event\":\"Gialla Temporali Allerta"
    "\",\"expires\":\"2026-10-02T23:59:00+02:00\",\"headline\":\"Allerta Tempo"
    "rali Verde per l'Italia - Sicilia\",\"instruction\":\"Non ci si aspetta "
    "che il tempo possa causare impatti significativi, ma potrebbero verifica"
    "rsi alcuni locali, minori problemi.\",\"language\":\"it-IT\",\"onset\":"
    "\"2026-09-26T10:08:00+02:00\",\"parameter\":[{\"value\":\"2; yellow; Mode"
    "rate\",\"valueName\":\"awareness_level\"},{\"value\":\"3; Thunderstorm\""
    ",\"valueName\":\"awareness_type\"}],\"responseType\":[\"None\"],\"sender"
    "Name\":\"Servizio Meteorologico dell'Aeronautica Militare\",\"severity\""
    ":\"Minor\",\"urgency\":\"Future\",\"web\":\"https://meteoalarm.org/it/li"
    "ve/region/IT?s=sicilia\"}],\"msgType\":\"Update\",\"references\":\"aeroc"
    "nmca.1sv.prv1@aeronautica.difesa.it,2.49.0.0.380.3.IT.260925102045.033,2"
    "026-09-25T10:20:46+02:00\",\"scope\":\"Public\",\"sender\":\"aerocnmca.1"
    "sv.prv1@aeronautica.difesa.it\",\"sent\":\"2026-09-26T10:08:57+02:00\","
    "\"status\":\"Actual\"},\"uuid\":\"4930e664-424b-4a89-bd27-3caf7df9de11\"}"
    ",{\"alert\":{\"identifier\":\"2.49.0.0.380.3.IT.260926100857.005\",\"inc"
    "idents\":\"Update\",\"info\":[{\"area\":[{\"areaDesc\":\"Sicilia\",\"geo"
    "code\":[{\"value\":\"IT018\",\"valueName\":\"EMMA_ID\"}]}],\"audience\":"
    "\"Private\",\"category\":[\"Met\"],\"certainty\":\"Likely\",\"descriptio"
    "n\":\"No Special Awareness Required\\n (DISCLAIMER: \\\"Information prov"
    "ided on METEOALARM for Italy regard only the intensity and recurrence of"
    " the phenomena, further details can be found at www.meteoam.it. METEOALA"
    "RM information do not provide the assessment of impact on the territory "
    "and they do not represent the Official Alerts messages that are issued b"
    "y the National Civil Protection Service https://www.protezionecivile.gov"
    ".it\\\")\",\"effective\":\"2026-09-26T10:08:00+02:00\",\"event\":\"Yello"
    "w Thunderstorm Warning\",\"expires\":\"2026-10-02T23:59:00+02:00\",\"hea"
    "dline\":\"Green Thunderstorm Warning for Italy - Sicilia\",\"instruction"
    "\":\"The weather is not expected to cause significant impacts but there "
    "may be some minor, localized issues\",\"language\":\"en-GB\",\"onset\":"
    "\"2026-09-26T10:08:00+02:00\",\"parameter\":[{\"value\":\"2; yellow; Mode"
    "rate\",\"valueName\":\"awareness_level\"},{\"value\":\"3; Thunderstorm\""
    ",\"valueName\":\"awareness_type\"}],\"responseType\":[\"None\"],\"sender"
    "Name\":\"Italian Air Force National Meteorological Service\",\"severity"
    "\":\"Minor\",\"urgency\":\"Future\",\"web\":\"https://meteoalarm.org/en/l"
    "ive/region/IT?s=sicilia\"},{\"area\":[{\"areaDesc\":\"Sicilia\",\"geocod"
    "e\":[{\"value\":\"IT018\",\"valueName\":\"EMMA_ID\"}]}],\"audience\":\"P"
    "rivato\",\"category\":[\"Met\"],\"certainty\":\"Likely\",\"description\""
    ":\"Nessun Avviso\\n (DISCLAIMER: \\\"Le informazioni fornite su METEOALA"
    "RM per l'Italia riguardano esclusivamente l'intensità e la ricorrenza de"
    "i fenomeni, maggiori dettagli sono disponibili su www.meteoam.it. Le inf"
    "ormazioni METEOALARM non forniscono la valutazione di impatto sul territ"
    "orio e non rappresentano i messaggi di Allerta Ufficiali che vengono eme"
    "ssi dal Servizio Nazionale di Protezione Civile https://www.protezioneci"
    "vile.gov.it\\\")\",\"effective\":\"2026-09-26T10:08:00+02:00\",\"event\""
    ":\"Gialla Temporali Allerta\",\"expires\":\"2026-10-02T23:59:00+02:00\","
    "\"headline\":\"Allerta Temporali Verde per l'Italia - Sicilia\",\"instru"
    "ction\":\"Non ci si aspetta che il tempo possa causare impatti significa"
    "tivi, ma potrebbero verificarsi alcuni locali, minori problemi.\",\"lang"
    "uage\":\"it-IT\",\"onset\":\"2026-09-26T10:08:00+02:00\",\"parameter\":["
    "{\"value\":\"2; yellow; Moderate\",\"valueName\":\"awareness_level\"},{"
    "\"value\":\"3; Thunderstorm\",\"valueName\":\"awareness_type\"}],\"respon"
    "seType\":[\"None\"],\"senderName\":\"Servizio Meteorologico dell'Aeronau"
    "tica Militare\",\"severity\":\"Minor\",\"urgency\":\"Future\",\"web\":\""
    "https://meteoalarm.org/it/live/region/IT?s=sicilia\"}],\"msgType\":\"Upd"
    "ate\",\"references\":\"aerocnmca.1sv.prv1@aeronautica.difesa.it,2.49.0.0"
    ".380.3.IT.260925102045.033,2026-09-25T10:20:46+02:00\",\"scope\":\"Publi"
    "c\",\"sender\":\"aerocnmca.1sv.prv1@aeronautica.difesa.it\",\"sent\":\"2"
    "026-09-26T10:08:57+02:00\",\"status\":\"Actual\"},\"uuid\":\"4930e664-42"
    "4b-4a89-bd27-3caf7df9de11\"}]}";

static const char DPC_DAY[] =
    "{\"type\": \"FeatureCollection\", \"features\": [{\"type\": \"Feature\","
    " \"properties\": {\"Rappresentata nella mappa\": \"Assenza di fenomeni s"
    "ignificativi prevedibili / NESSUNA ALLERTA\", \"Per rischio idraulico\":"
    " \"Assenza di fenomeni significativi prevedibili / NESSUNA ALLERTA\", \""
    "Per rischio temporali\": \"Assenza di fenomeni significativi prevedibili"
    " / NESSUNA ALLERTA\", \"Per rischio idrogeologico\": \"Assenza di fenome"
    "ni significativi prevedibili / NESSUNA ALLERTA\", \"Nome zona\": \"Nord-"
    "Occidentale e isole Egadi e Ustica\", \"Comuni\": [\"Erice\", \"Trapani"
    "\", \"Marsala\"]}, \"geometry\": null}, {\"type\": \"Feature\", \"propert"
    "ies\": {\"Rappresentata nella mappa\": \"Assenza di fenomeni significati"
    "vi prevedibili / NESSUNA ALLERTA\", \"Per rischio idraulico\": \"Assenza"
    " di fenomeni significativi prevedibili / NESSUNA ALLERTA\", \"Per rischi"
    "o temporali\": \"Ordinaria criticità / ALLERTA GIALLA\", \"Per rischio i"
    "drogeologico\": \"Assenza di fenomeni significativi prevedibili / NESSUN"
    "A ALLERTA\", \"Nome zona\": \"Zona di prova\", \"Comuni\": [\"Trapani\","
    " \"Paceco\"]}, \"geometry\": {\"type\": \"MultiPolygon\", \"coordinates"
    "\": [[[[12.3, 37.8], [12.6, 37.8], [12.6, 38.0], [12.3, 38.0], [12.3, 3"
    "7.8]]], [[[1, 1], [2, 1], [2, 2], [1, 1]]]]}}, {\"type\": \"Feature\", "
    "\"properties"
    "\": {\"Rappresentata nella mappa\": \"Assenza di fenomeni significativi "
    "prevedibili / NESSUNA ALLERTA\", \"Per rischio idraulico\": \"Elevata cr"
    "iticità / ALLERTA ROSSA\", \"Per rischio temporali\": \"Assenza di fenom"
    "eni significativi prevedibili / NESSUNA ALLERTA\", \"Per rischio idrogeo"
    "logico\": \"Assenza di fenomeni significativi prevedibili / NESSUNA ALLE"
    "RTA\", \"Nome zona\": \"Altrove\", \"Comuni\": [\"Torino\"]}, \"geometry"
    "\": {\"type\": \"Polygon\", \"coordinates\": [[[7.5, 44.9], [7.9, 44.9], "
    "[7.9, 45.2], [7.5, 45.2]], [[7.6, 45.0], [7.62, 45.0], [7.62, 45.02], "
    "[7.6, 45.02], [7.6, 45.0]]]}}]}";

static void forecast(void)
{
    static struct wx_forecast f;
    char err[256];
    int r =
        wx_om_parse(OM_FORECAST, sizeof OM_FORECAST - 1, &f, err, sizeof err);
    CHECK(r == 0, "Open-Meteo: %s", err);
    CHECK(f.grid_lat == 38.0 && f.grid_lon == 12.5625 && f.grid_elev == 6.0,
          "the cell: %f %f %f", f.grid_lat, f.grid_lon, f.grid_elev);
    CHECK(strcmp(f.tz, "Europe/Rome") == 0, "zone %s", f.tz);
    CHECK(f.has_now && f.now.temp == 26.3 && f.now.code == 0 &&
              f.now.wind == 13.4 && f.now.wind_dir == 324 &&
              strcmp(f.now.time, "2026-10-01T13:45") == 0,
          "now: %f %d %s", f.now.temp, f.now.code, f.now.time);
    CHECK(f.n_day == 3 && f.day[0].tmax == 26.6 &&
              strcmp(f.day[0].date, "2026-10-01") == 0 &&
              strcmp(f.day[0].sunrise, "07:05") == 0,
          "days: %d, %f, %s", f.n_day, f.day[0].tmax, f.day[0].sunrise);
    CHECK(f.n_hour == WX_HOURS, "hours: %d", f.n_hour);

    r = wx_met_parse(MET_NORWAY, sizeof MET_NORWAY - 1, "Europe/Rome", &f, err,
                     sizeof err);
    CHECK(r == 0, "MET Norway: %s", err);
    /* 11:00 UTC is 13:00 in Rome, summer time */
    CHECK(strcmp(f.now.time, "2026-10-01T13:00") == 0 &&
              strcmp(f.tz_abbr, "CEST") == 0,
          "MET Norway now: %s %s", f.now.time, f.tz_abbr);
    CHECK(f.n_day >= 2 && strcmp(f.day[0].date, "2026-10-01") == 0 &&
              f.day[0].tmin <= f.day[0].tmax && f.day[0].symbol[0],
          "MET Norway days: %d %s", f.n_day, f.day[0].date);
    CHECK(f.n_hour > 20 && f.now.symbol[0], "MET Norway hours: %d", f.n_hour);
    CHECK(f.now.wind > 12 && f.now.wind < 14, "m/s into km/h: %f", f.now.wind);
}

static void sea(void)
{
    struct wx_sea s[3] = {{.lat = 39.5, .lon = 12.0},
                          {.lat = 40.5, .lon = 11.0},
                          {.lat = 38.02, .lon = 12.53}};
    char err[256];
    int r = wx_om_marine_parse(OM_MARINE, sizeof OM_MARINE - 1, s, 3, err,
                               sizeof err);
    CHECK(r == 0, "the sea: %s", err);
    CHECK(s[0].ok && s[1].ok && s[2].ok && s[0].wave == 0.10 &&
              s[1].wave == 0.14 && s[2].wave == 0.08 && s[0].sst == 26.7,
          "the waves: %f %f %f", s[0].wave, s[1].wave, s[2].wave);
    CHECK(s[0].n_day == 2 && s[0].day[0].wave_max > 0, "the sea's days: %d",
          s[0].n_day);
    CHECK(fabs(s[2].cell_lat - 38.041664) < 1e-6, "the coast's cell: %f",
          s[2].cell_lat);
}

static void station(void)
{
    struct wx_obs o;
    char err[256];
    int r = wx_metar_parse(METAR_BBOX, sizeof METAR_BBOX - 1, 38.0176, 12.53617,
                           40, &o, err, sizeof err);
    CHECK(r == 1 && strcmp(o.icao, "LICT") == 0 && o.temp == 26 && o.km > 10 &&
              o.km < 15 && strstr(o.clouds, "clouds at"),
          "the station: %d %s %f %f '%s'", r, o.icao, o.temp, o.km, o.clouds);
    r = wx_metar_parse(METAR_BBOX, sizeof METAR_BBOX - 1, 38.0176, 12.53617, 5,
                       &o, err, sizeof err);
    CHECK(r == 0, "no station within 5 km: %d", r);
}

static void warnings(void)
{
    struct tm tm = {.tm_year = 126, .tm_mon = 9, .tm_mday = 1, .tm_hour = 12};
    time_t now = timegm(&tm);
    struct wx_alert *v;
    size_t n;
    char err[256];
    int r = wx_meteoalarm_parse(METEOALARM, sizeof METEOALARM - 1, "Sicilia",
                                "it", now, &v, &n, err, sizeof err);
    CHECK(r == 0, "MeteoAlarm: %s", err);
    /* the expired and the green left out, the yellow given twice once */
    CHECK(n == 2, "warnings: %zu", n);
    if (n == 2) {
        CHECK(v[0].mine && strcmp(v[0].level, "yellow") == 0 &&
                  strstr(v[0].area, "Sicilia"),
              "Sicily's first: %d %s %s", v[0].mine, v[0].level, v[0].area);
        CHECK(!v[1].mine && strcmp(v[1].level, "orange") == 0,
              "then the rest: %s", v[1].level);
    }
    free(v);

    char lv[3][96], zones[192];
    r = wx_dpc_parse(DPC_DAY, sizeof DPC_DAY - 1, "Trapani", NAN, NAN, lv,
                     zones, sizeof zones, err, sizeof err);
    CHECK(r == 1 && strstr(zones, "Egadi") && strstr(zones, "Zona di prova"),
          "Trapani's zones: %d %s", r, zones);
    CHECK(strstr(lv[0], "NESSUNA") && strstr(lv[2], "GIALLA"),
          "the highest of its zones: %s / %s", lv[0], lv[2]);
    r = wx_dpc_parse(DPC_DAY, sizeof DPC_DAY - 1, "Torino", NAN, NAN, lv, zones,
                     sizeof zones, err, sizeof err);
    CHECK(r == 1 && strstr(lv[0], "ROSSA"), "Turin: %s", lv[0]);
    r = wx_dpc_parse(DPC_DAY, sizeof DPC_DAY - 1, "Roma", 41.9, 12.5, lv, zones,
                     sizeof zones, err, sizeof err);
    CHECK(r == 0, "Rome not in it: %d", r);
    /* a name the bulletin does not have: by the point, in a polygon (an
       open ring, closed by the reader) and in a multipolygon */
    r = wx_dpc_parse(DPC_DAY, sizeof DPC_DAY - 1, "Turin", 45.07, 7.69, lv,
                     zones, sizeof zones, err, sizeof err);
    CHECK(r == 1 && strstr(lv[0], "ROSSA") && strcmp(zones, "Altrove") == 0,
          "Turin by the point: %d %s", r, zones);
    r = wx_dpc_parse(DPC_DAY, sizeof DPC_DAY - 1, "Turin", 45.01, 7.61, lv,
                     zones, sizeof zones, err, sizeof err);
    CHECK(r == 0, "in the polygon's hole: %d %s", r, zones);
    r = wx_dpc_parse(DPC_DAY, sizeof DPC_DAY - 1, "Trapani/Birgi Arpt", 37.91,
                     12.49, lv, zones, sizeof zones, err, sizeof err);
    CHECK(r == 1 && strstr(lv[2], "GIALLA") &&
              strcmp(zones, "Zona di prova") == 0,
          "Birgi by the point: %d %s", r, zones);
}

static void words(void)
{
    char s[96];
    CHECK(strcmp(wx_code_text(63), "rain") == 0 && !wx_code_text(42),
          "WMO codes");
    wx_symbol_text("lightrainshowersandthunder_day", s, sizeof s);
    CHECK(strcmp(s, "light rain showers and thunder") == 0, "symbol: '%s'", s);
    wx_symbol_text("clearsky_night", s, sizeof s);
    CHECK(strcmp(s, "clear sky") == 0, "symbol: '%s'", s);
    CHECK(strcmp(wx_wind_words(13.4), "gentle breeze") == 0 &&
              strcmp(wx_wind_words(130), "hurricane force") == 0,
          "Beaufort");
    CHECK(strcmp(wx_sea_words(0.08), "calm") == 0 &&
              strcmp(wx_sea_words(1.5), "moderate") == 0,
          "Douglas");
    CHECK(wx_douglas(0.05) == 1 && wx_douglas(0.3) == 2 &&
              wx_douglas(1.5) == 4 && wx_douglas(20) == 9,
          "Douglas degrees");
    wx_from_dir(321, s, sizeof s);
    CHECK(strcmp(s, "from the north-west (321°)") == 0, "direction: '%s'", s);
}

int main(void)
{
    alarm(60);
    forecast();
    sea();
    station();
    warnings();
    words();
    if (failures) {
        printf("test_weather_parse: %d failures\n", failures);
        return 1;
    }
    printf("test_weather_parse: ok\n");
    return 0;
}
