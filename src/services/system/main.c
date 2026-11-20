/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-system: this computer as an MCP server over stdio, for
 * janas-chat (which starts it by itself) and any other MCP client. It
 * only reads, and nothing it reads leaves the computer but in its answers:
 * how the computer is, its disks and what fills a directory, its
 * processes, its errors, its updates, its network.
 */
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "system.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

/* INSTRUCTIONS, from SKILL.md beside this file (build.sh) */
#include "skill.h"

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-system\n"
            "An MCP server over stdio: a client starts it and calls its\n"
            "tools - system_status, system_disks (with a path, what fills\n"
            "it), system_processes, system_errors, system_updates and\n"
            "system_network. They only read; none reaches the network.\n"
            "janas-chat starts it by itself when it is beside it.\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"system\": {\"command\": "
            "\"janas-system\"}}}\n");
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    struct janas_mcps srv = {.name = "janas-system",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .tools_list = sy_tools_list,
                             .tools_call = sy_tools_call};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-system: cannot set its output aside\n");
        return 1;
    }
    fprintf(stderr,
            "janas-system %s: ready, 6 tools (system_status, system_disks, "
            "system_processes, system_errors, system_updates, "
            "system_network)\n",
            JANAS_VERSION);
    janas_mcps_run(&srv);
    return 0;
}
