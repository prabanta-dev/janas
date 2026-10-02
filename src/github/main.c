/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-github: GitHub as an MCP server over stdio, for janas-chat
 * (which starts it by itself) and any other MCP client. It only reads: a
 * repository's issues and pull requests, its releases, its recent commits,
 * what it is. GitHub's REST API, with the gh command's token when there is
 * one.
 */
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "github.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static const char INSTRUCTIONS[] =
    "Use these tools by yourself when the user asks about a project on "
    "GitHub: its issues or pull requests (new ones, open ones), its latest "
    "release, its recent commits, what it is; never name them. A project "
    "named alone (\"Janas\") is looked up by its name: tell the user which "
    "repository it was. They only read: nothing on GitHub is changed.";

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-github\n"
            "An MCP server over stdio: a client starts it and calls its\n"
            "tools - github_issues (issues and pull requests, the new ones\n"
            "since the last time asked), github_releases, github_commits\n"
            "and github_repo. It only reads.\n"
            "GitHub's REST API, with a token from GH_TOKEN, GITHUB_TOKEN\n"
            "or the gh command (gh auth token) when there is one: 5,000\n"
            "requests an hour and the user's private repositories; without\n"
            "one, 60 an hour. The token stays in memory, never written.\n"
            "When the issues were last seen: ~/.config/janas/"
            "github-seen.txt.\n"
            "janas-chat starts it by itself when it is beside it.\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"github\": {\"command\": "
            "\"janas-github\"}}}\n");
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    struct janas_mcps srv = {.name = "janas-github",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .tools_list = gh_tools_list,
                             .tools_call = gh_tools_call};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-github: cannot set its output aside\n");
        return 1;
    }
    int k = gh_token_kind();
    fprintf(stderr,
            "janas-github %s: ready, 4 tools (github_issues, "
            "github_releases, github_commits, github_repo), %s\n",
            JANAS_VERSION,
            k == 2   ? "with the gh command's token"
            : k == 1 ? "with the token of GH_TOKEN or GITHUB_TOKEN"
                     : "without a token (60 requests an hour)");
    janas_mcps_run(&srv);
    gh_fetch_free();
    return 0;
}
