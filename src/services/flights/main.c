/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas-flights - flights as an MCP service, over stdio: where a flight is
 * now and what it is doing, and what flies over a place, from the ADS-B
 * receivers of the community (adsb.lol, adsb.fi), with routes from
 * adsbdb.com, no key needed; and, with an AviationStack key of the user's
 * (free, 100 calls a month), today's schedules: scheduled, estimated and
 * actual times, gates, the flights of a route. Any MCP client can use it:
 * janas-chat, and the assistants of others.
 *
 * Every answer is facts already worked out, in words (the state, the sea or
 * the region, the nearest city, an arrival estimated from the speed): the
 * model that tells them has nothing to compute.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/mcp_server.h"
#include "flights.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static const char INSTRUCTIONS[] =
    "Use these tools by yourself for questions on flights, aircraft, "
    "airports, departures and arrivals, or what is flying somewhere; the "
    "user need not name them, and you never name them. Schedules (times, "
    "gates, delays) come from AviationStack when the server has a key, "
    "positions from the community's ADS-B receivers. Tell what they "
    "return in the user's language, adding no figures or times of your "
    "own; never state a route marked as probably wrong. For an area - a "
    "sea, around a place - use flights_over or flights_nearby, and never "
    "call flight_status or flights_between for each aircraft of a list: "
    "their quota is small. Never ask for flights the user did not ask "
    "for.";

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-flights [--config FILE]\n"
            "An MCP server over stdio: a client starts it and calls its\n"
            "tools - flight_status (a flight by number or callsign),\n"
            "flights_between (the flights of a route), flights_over (what\n"
            "flies over a sea) and flights_nearby (what flies near an\n"
            "airport, a city or a point).\n"
            "Data: adsb.lol (ODbL) and adsb.fi for positions, adsbdb.com\n"
            "for routes; places from OurAirports, GeoNames (CC BY 4.0) and\n"
            "Natural Earth. Schedules from AviationStack, with a free key\n"
            "of your own (https://aviationstack.com, 100 calls a month):\n"
            "the line  aviationstack_key = KEY  in FILE (by default\n"
            "~/.config/janas/flights.conf, or JANAS_FLIGHTS_CONFIG), or\n"
            "JANAS_AVIATIONSTACK_KEY.\n"
            "janas-chat starts it by itself when it is beside it.\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"flights\": {\"command\": "
            "\"janas-flights\"}}}\n");
}

int main(int argc, char **argv)
{
    const char *config = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config = argv[++i];
            continue;
        }
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    char path[4096];
    if (!config)
        config = getenv("JANAS_FLIGHTS_CONFIG");
    if (!config || !*config) {
        const char *xdg = getenv("XDG_CONFIG_HOME"), *home = getenv("HOME");
        if (xdg && *xdg)
            snprintf(path, sizeof path, "%s/janas/flights.conf", xdg);
        else
            snprintf(path, sizeof path, "%s/.config/janas/flights.conf",
                     home ? home : ".");
        config = path;
    }
    int sched = fl_sched_config(config);
    struct janas_mcps srv = {.name = "janas-flights",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .tools_list = fl_tools_list,
                             .tools_call = fl_tools_call};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-flights: cannot set its output aside\n");
        return 1;
    }
    fprintf(stderr,
            "janas-flights: ready, tools: flight_status flights_between "
            "flights_over flights_nearby; schedules: %s\n",
            sched ? "AviationStack" : "none (no key)");
    if (fl_sched_problem()[0])
        fprintf(stderr, "janas-flights: %s\n", fl_sched_problem());
    janas_mcps_run(&srv);
    fl_sched_free();
    fprintf(stderr, "janas-flights: the client has gone\n");
    return 0;
}
