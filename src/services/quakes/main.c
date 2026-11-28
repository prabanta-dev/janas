/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-quakes: the earthquakes as an MCP server over stdio, for
 * janas-chat (which starts it by itself) and any other MCP client: those
 * near a place, the strongest in the world, one in detail, from INGV (CC
 * BY 4.0) for Italy and its seas and the USGS (public domain) elsewhere.
 */
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "quakes.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

/* INSTRUCTIONS, from SKILL.md beside this file (build.sh) */
#include "skill.h"
/* VIEW_map, from views/map.html (build.sh): the earthquakes on a map, for
   a client that shows MCP Apps' views */
#include "views.h"

static const struct janas_mcps_view views[] = {{.uri = "ui://quakes/map",
                                                .name = "Earthquakes on a map",
                                                .html = VIEW_map,
                                                .csp = NULL}};

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-quakes\n"
            "An MCP server over stdio: a client starts it and calls its\n"
            "tools - quakes_near, quakes_strong and quakes_event. They only\n"
            "read, from INGV (Italy and its seas) and the USGS (the world).\n"
            "janas-chat starts it by itself when it is beside it.\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"quakes\": {\"command\": "
            "\"janas-quakes\"}}}\n");
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    struct janas_mcps srv = {.name = "janas-quakes",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .tools_list = qk_tools_list,
                             .tools_call = qk_tools_call,
                             .views = views,
                             .n_views = 1};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-quakes: cannot set its output aside\n");
        return 1;
    }
    fprintf(stderr,
            "janas-quakes %s: ready, 3 tools (quakes_near, quakes_strong, "
            "quakes_event)\n",
            JANAS_VERSION);
    janas_mcps_run(&srv);
    return 0;
}
