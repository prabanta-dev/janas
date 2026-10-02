/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-maps: roads and places as an MCP server over stdio, for
 * janas-chat (which starts it by itself) and any other MCP client: routes
 * by car, bike and on foot, journeys by public transport, what is near a
 * place, where a place is. OpenStreetMap's data through free services,
 * without a key.
 */
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "maps.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static const char INSTRUCTIONS[] =
    "Use these tools by yourself when the user asks how to get somewhere, "
    "how far or how long, by car, bike, on foot or by public transport, "
    "where a place or an address is, or what is near (fuel, a pharmacy, a "
    "cash machine...); never name them. Give places with their town. With "
    "no start named, give none: the tool takes where the user is; tell the "
    "user which place, and that it may be off. Times by road are without "
    "traffic; public transport is from the timetables. Never ask the other "
    "services for anything the user did not ask for.";

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-maps\n"
            "An MCP server over stdio: a client starts it and calls its\n"
            "tools - maps_route (a route by car, bike or on foot, step by\n"
            "step), maps_transit (journeys by public transport),\n"
            "maps_nearby (places of a kind near a place) and maps_find\n"
            "(where a place or an address is).\n"
            "Data, all free and without a key, from OpenStreetMap (ODbL):\n"
            "Nominatim and Photon (places), Valhalla and OSRM on the\n"
            "servers of FOSSGIS (routes), Transitous (public transport,\n"
            "from the operators' open timetables), Overpass (what is\n"
            "near). At most one request a second to each.\n"
            "The servers can be changed: JANAS_NOMINATIM_URL,\n"
            "JANAS_PHOTON_URL, JANAS_VALHALLA_URL, JANAS_OSRM_URL,\n"
            "JANAS_TRANSITOUS_URL, JANAS_OVERPASS_URL,\n"
            "JANAS_OVERPASS_URL2 (when the first is busy).\n"
            "janas-chat starts it by itself when it is beside it.\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"maps\": {\"command\": "
            "\"janas-maps\"}}}\n");
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    struct janas_mcps srv = {.name = "janas-maps",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .tools_list = mp_tools_list,
                             .tools_call = mp_tools_call};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-maps: cannot set its output aside\n");
        return 1;
    }
    fprintf(stderr,
            "janas-maps %s: ready, 4 tools (maps_route, maps_transit, "
            "maps_nearby, maps_find)\n",
            JANAS_VERSION);
    janas_mcps_run(&srv);
    mp_fetch_free();
    return 0;
}
