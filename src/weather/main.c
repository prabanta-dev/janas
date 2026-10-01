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
    "Use these tools by yourself whenever the user asks about the weather, "
    "the temperature, rain, wind, the sea and its waves, or weather "
    "warnings, now or in the next days: the user does not need to name "
    "them or ask for them. In your reply never name the tools, the server "
    "or the service: just give the answer, with the sources the tool "
    "lists. Tell the user what the tools return, in the user's language, "
    "without adding figures of your own: they already give the times in "
    "the place's local time and the wind and the sea in words. A forecast "
    "is a model's value and a station's is a measure: say which is which "
    "when both are given, and when they disagree say so. For a sea as a "
    "whole use weather_sea with its name; for warnings use weather_alerts, "
    "and say plainly when there is none. When the user names no place, "
    "give none: the tools take where the user is and say how they know "
    "it; tell the user which place that is, and that it may be off.";

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
