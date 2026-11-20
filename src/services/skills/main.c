/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-skills: the user's skills, in the format of Claude's, as
 * an MCP server over stdio, for janas-chat (which starts it by itself when
 * there are skills) and any other MCP client: listed, read, and their
 * scripts run in a cage without network, each run asked of the user by the
 * client.
 */
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "common/sandbox.h"
#include "skills.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

/* INSTRUCTIONS, from SKILL.md beside this file (build.sh) */
#include "skill.h"

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-skills [--dir DIR]...\n"
            "An MCP server over stdio: a client starts it and calls its\n"
            "tools - skills_list, skill_read and skill_run. A skill is a\n"
            "folder with a SKILL.md, in the format of Claude's skills;\n"
            "those of every chat are in skills/ beside mcp.json\n"
            "(~/.config/janas/skills), and --dir adds a folder of them,\n"
            "searched first (up to %d). skill_run runs a script of a skill\n"
            "in a cage without network (Linux).\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"skills\": {\"command\": "
            "\"janas-skills\"}}}\n",
            SK_DIRS);
}

int main(int argc, char **argv)
{
    janas_sandbox_main(argc, argv); /* a cage for run.c being built */
    static struct sk_ctx ctx;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dir") && i + 1 < argc && ctx.n_dirs < SK_DIRS) {
            ctx.dirs[ctx.n_dirs++] = argv[++i];
            continue;
        }
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    struct janas_mcps srv = {.name = "janas-skills",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .ctx = &ctx,
                             .tools_list = sk_tools_list,
                             .tools_call = sk_tools_call};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-skills: cannot set its output aside\n");
        return 1;
    }
    fprintf(stderr,
            "janas-skills %s: ready, 3 tools (skills_list, skill_read, "
            "skill_run)\n",
            JANAS_VERSION);
    janas_mcps_run(&srv);
    return 0;
}
