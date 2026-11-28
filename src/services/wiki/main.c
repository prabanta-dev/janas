/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-wiki: Wikipedia as an MCP server over stdio, for
 * janas-chat (which starts it by itself) and any other MCP client: its
 * search, its pages as text, and the pages about what lies around a
 * place. Wikipedia's own API, free and without a key.
 */
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "wiki.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

/* INSTRUCTIONS, from SKILL.md beside this file (build.sh) */
#include "skill.h"
/* VIEW_page, from views/page.html (build.sh): a page of Wikipedia as it
   is written there, for a client that shows MCP Apps' views */
#include "views.h"

static const struct janas_mcps_view views[] = {
    {.uri = "ui://wiki/page",
     .name = "A page of Wikipedia",
     .html = VIEW_page,
     /* its pictures, from Wikimedia's own server of files */
     .csp = "\"resourceDomains\": [\"https://upload.wikimedia.org\", "
            "\"https://thumb.wikimedia.org\"]"},
    /* the pages around a place on a map */
    {.uri = "ui://wiki/nearby",
     .name = "Wikipedia around a place",
     .html = VIEW_nearby,
     .csp = NULL}};

static void usage(void)
{
    fprintf(stderr,
            "usage: janas-wiki\n"
            "An MCP server over stdio: a client starts it and calls its\n"
            "tools - wiki_search (the pages that match some words),\n"
            "wiki_page (a page as text: whole, or its introduction and\n"
            "then a section) and wiki_nearby (the pages about what lies\n"
            "around a place, nearest first).\n"
            "Data: Wikipedia's API, free and without a key; the text is\n"
            "Wikipedia's, under CC BY-SA 4.0.\n"
            "janas-chat starts it by itself when it is beside it.\n"
            "Configured in a client as, for instance:\n"
            "  {\"mcpServers\": {\"wiki\": {\"command\": "
            "\"janas-wiki\"}}}\n");
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    struct janas_mcps srv = {.name = "janas-wiki",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .tools_list = wiki_tools_list,
                             .tools_call = wiki_tools_call,
                             .views = views,
                             .n_views = 2};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-wiki: cannot set its output aside\n");
        return 1;
    }
    fprintf(stderr,
            "janas-wiki %s: ready, 3 tools (wiki_search, wiki_page, "
            "wiki_nearby)\n",
            JANAS_VERSION);
    janas_mcps_run(&srv);
    wiki_fetch_free();
    return 0;
}
