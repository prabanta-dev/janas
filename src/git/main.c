/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-git: the git repositories of this computer as an MCP
 * server over stdio, for janas-chat (which starts it by itself) and any
 * other MCP client. It reads (status, log, diff, branches) and, asked of
 * the user each time, changes (commit, pull, push, switch); it never
 * throws work away.
 */
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "git.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static const char INSTRUCTIONS[] =
    "Use these tools by yourself when the user asks about a git repository "
    "on this computer: what has changed, the latest commits, the branches, "
    "a diff; never name them. With no repository named, leave it out: the "
    "tool takes the one janas-chat runs in. To commit, pull, push or "
    "switch branch, only when the user asked for it: then call the tool at "
    "once, without asking first, since the user is asked to confirm each "
    "of these calls. Write a commit's message from the diff, in the "
    "language of the repository's earlier commits.";

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-git\n"
            "An MCP server over stdio: a client starts it and calls its\n"
            "tools - git_status, git_log, git_diff and git_branches, which\n"
            "only read, and git_commit, git_pull (fast-forward only),\n"
            "git_push (never by force) and git_switch, which say they\n"
            "change something, so that the client asks the user each\n"
            "time. Nothing that throws work away: no reset, restore,\n"
            "clean, amend.\n"
            "A repository is a path, the directory it runs in, or a name\n"
            "listed in ~/.config/janas/git-repos.txt (a path a line).\n"
            "git runs without a terminal: a password asked fails at once.\n"
            "janas-chat starts it by itself when it is beside it.\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"git\": {\"command\": \"janas-git\"}}}\n");
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    struct janas_mcps srv = {.name = "janas-git",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .tools_list = gt_tools_list,
                             .tools_call = gt_tools_call};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-git: cannot set its output aside\n");
        return 1;
    }
    fprintf(stderr,
            "janas-git %s: ready, 8 tools (git_status, git_log, git_diff, "
            "git_branches; asked first: git_commit, git_pull, git_push, "
            "git_switch)\n",
            JANAS_VERSION);
    janas_mcps_run(&srv);
    return 0;
}
