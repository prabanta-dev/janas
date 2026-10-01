/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * words.c - the weather in words (see weather.h): the WMO codes Open-Meteo
 * gives, MET Norway's symbols, the wind on the Beaufort scale, the sea on
 * the Douglas scale, directions as the points of the compass. The model
 * telling them then repeats words, not tables it would have to know.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "common/geo.h"
#include "weather.h"

const char *wx_code_text(int code)
{
    switch (code) {
    case 0:
        return "clear sky";
    case 1:
        return "mainly clear";
    case 2:
        return "partly cloudy";
    case 3:
        return "overcast";
    case 45:
        return "fog";
    case 48:
        return "fog depositing rime";
    case 51:
        return "light drizzle";
    case 53:
        return "drizzle";
    case 55:
        return "dense drizzle";
    case 56:
        return "light freezing drizzle";
    case 57:
        return "dense freezing drizzle";
    case 61:
        return "light rain";
    case 63:
        return "rain";
    case 65:
        return "heavy rain";
    case 66:
        return "light freezing rain";
    case 67:
        return "heavy freezing rain";
    case 71:
        return "light snow";
    case 73:
        return "snow";
    case 75:
        return "heavy snow";
    case 77:
        return "snow grains";
    case 80:
        return "light rain showers";
    case 81:
        return "rain showers";
    case 82:
        return "violent rain showers";
    case 85:
        return "light snow showers";
    case 86:
        return "heavy snow showers";
    case 95:
        return "thunderstorm";
    case 96:
        return "thunderstorm with light hail";
    case 99:
        return "thunderstorm with heavy hail";
    default:
        return NULL;
    }
}

void wx_symbol_text(const char *symbol, char *out, size_t cap)
{
    /* "lightrainshowersandthunder_day": the words run together, the part
       of the day after an underscore */
    static const struct {
        const char *from, *to;
    } w[] = {
        {"clearsky", "clear sky"},
        {"partlycloudy", "partly cloudy"},
        {"fair", "fair"},
        {"cloudy", "cloudy"},
        {"fog", "fog"},
        {"light", "light "},
        {"heavy", "heavy "},
        {"rainshowers", "rain showers"},
        {"sleetshowers", "sleet showers"},
        {"snowshowers", "snow showers"},
        {"rain", "rain"},
        {"sleet", "sleet"},
        {"snow", "snow"},
        {"andthunder", " and thunder"},
    };
    size_t n = 0;
    out[0] = 0;
    const char *s = symbol ? symbol : "";
    while (*s && *s != '_') {
        size_t i;
        for (i = 0; i < sizeof w / sizeof w[0]; i++) {
            size_t k = strlen(w[i].from);
            if (strncmp(s, w[i].from, k) == 0) {
                n += (size_t)snprintf(out + n, n < cap ? cap - n : 0, "%s",
                                      w[i].to);
                s += k;
                break;
            }
        }
        if (i == sizeof w / sizeof w[0]) { /* a word not known: as it is */
            if (n + 1 < cap) {
                out[n++] = *s;
                out[n] = 0;
            }
            s++;
        }
        if (n >= cap) {
            out[cap - 1] = 0;
            return;
        }
    }
}

const char *wx_wind_words(double kmh)
{
    static const struct {
        double below;
        const char *name;
    } b[] = {
        {1, "calm"},           {6, "light air"},        {12, "light breeze"},
        {20, "gentle breeze"}, {29, "moderate breeze"}, {39, "fresh breeze"},
        {50, "strong breeze"}, {62, "near gale"},       {75, "gale"},
        {89, "strong gale"},   {103, "storm"},          {118, "violent storm"},
    };
    for (size_t i = 0; i < sizeof b / sizeof b[0]; i++)
        if (kmh < b[i].below)
            return b[i].name;
    return "hurricane force";
}

const char *wx_sea_words(double m)
{
    if (m < 0.1)
        return "calm";
    if (m < 0.5)
        return "smooth";
    if (m < 1.25)
        return "slight";
    if (m < 2.5)
        return "moderate";
    if (m < 4)
        return "rough";
    if (m < 6)
        return "very rough";
    if (m < 9)
        return "high";
    if (m < 14)
        return "very high";
    return "phenomenal";
}

void wx_from_dir(double deg, char *out, size_t cap)
{
    static const struct {
        const char *abbr, *name;
    } w[] = {
        {"N", "north"},       {"NNE", "north-north-east"},
        {"NE", "north-east"}, {"ENE", "east-north-east"},
        {"E", "east"},        {"ESE", "east-south-east"},
        {"SE", "south-east"}, {"SSE", "south-south-east"},
        {"S", "south"},       {"SSW", "south-south-west"},
        {"SW", "south-west"}, {"WSW", "west-south-west"},
        {"W", "west"},        {"WNW", "west-north-west"},
        {"NW", "north-west"}, {"NNW", "north-north-west"},
    };
    if (isnan(deg)) {
        snprintf(out, cap, "from a direction not given");
        return;
    }
    const char *a = geo_compass(deg);
    const char *name = a;
    for (size_t i = 0; i < sizeof w / sizeof w[0]; i++)
        if (strcmp(w[i].abbr, a) == 0)
            name = w[i].name;
    snprintf(out, cap, "from the %s (%.0f°)", name, deg);
}
