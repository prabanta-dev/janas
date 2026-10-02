/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-weather: the weather as an MCP server over stdio, for
 * janas-chat (which starts it by itself) and any other MCP client. Free
 * sources only, no key: Open-Meteo (MET Norway when it does not answer),
 * the airports' stations, MeteoAlarm and the Italian Civil Protection.
 */
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "weather.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static const char INSTRUCTIONS[] =
    "Use these tools by yourself for any question on the weather, rain, "
    "wind, the sea and its waves, or weather warnings; the user need not "
    "name them, and you never name them. Tell what they return in the "
    "user's language, adding no figures of your own: times are local "
    "already, wind and sea in words. A forecast is a model's value and a "
    "station's report a measure: say which, and say when they disagree. "
    "For a whole sea use weather_sea with its name; for warnings "
    "weather_alerts, and say plainly when there are none. With no place "
    "named, give none: the tool takes where the user is and says how it "
    "knows; tell the user that place, and that it may be off. Never ask "
    "for weather the user did not ask for.";

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-weather\n"
            "An MCP server over stdio: a client starts it and calls its\n"
            "tools - weather_now (the weather now at a place, with what\n"
            "the nearest station measured), weather_forecast (by the day\n"
            "and by the hour), weather_sea (waves, swell, the water, the\n"
            "wind, over a sea or off a place) and weather_alerts (the\n"
            "warnings in force; in Italy the Civil Protection's levels).\n"
            "Data, all free and without a key: Open-Meteo (CC BY 4.0;\n"
            "free for non-commercial use), MET Norway (CC BY 4.0) when\n"
            "Open-Meteo does not answer, the airports' METAR from\n"
            "aviationweather.gov (NOAA), MeteoAlarm (EUMETNET) and the\n"
            "Dipartimento della Protezione Civile (CC BY 4.0); places\n"
            "from GeoNames (CC BY 4.0) and Natural Earth.\n"
            "janas-chat starts it by itself when it is beside it.\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"weather\": {\"command\": "
            "\"janas-weather\"}}}\n");
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    struct janas_mcps srv = {.name = "janas-weather",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .tools_list = wx_tools_list,
                             .tools_call = wx_tools_call};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-weather: cannot set its output aside\n");
        return 1;
    }
    fprintf(stderr,
            "janas-weather %s: ready, 4 tools (weather_now, "
            "weather_forecast, weather_sea, weather_alerts)\n",
            JANAS_VERSION);
    janas_mcps_run(&srv);
    wx_fetch_free();
    return 0;
}
