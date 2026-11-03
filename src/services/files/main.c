/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - janas-files: the user's files as an MCP server over stdio, for
 * janas-chat (which starts it by itself) and any other MCP client. It
 * lists, reads and searches the home and the folders the user listed, and,
 * asked of the user each time, writes, edits and makes folders, keeping a
 * copy of what it replaces; it never deletes, and never opens the private
 * files (keys, passwords, the browsers' data).
 */
#include <stdio.h>
#include <string.h>

#include "common/mcp_server.h"
#include "files.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static const char INSTRUCTIONS[] =
    "Use these tools by yourself when the user asks about the files and "
    "folders of this computer - what a folder holds, what a file says, "
    "where something is - or gives you files to look at; never name them. "
    "A path is ~/..., absolute, or relative to the folder janas-chat runs "
    "in. Read a file before you say what it holds, and the files a task "
    "builds on before you write. To write, edit or make a folder, only "
    "when the user asked for it: then call the tool at once, without "
    "asking first, since the user is asked to confirm each of these "
    "calls. Private files (keys, passwords, a browser's data) and the "
    "folders outside the home are not reachable: say so when asked.";

static void usage(void)
{
    fprintf(
        stderr,
        "usage: janas-files\n"
        "An MCP server over stdio: a client starts it and calls its\n"
        "tools - files_list, files_read and files_search, which only\n"
        "read, and files_write, files_edit and files_mkdir, which say\n"
        "they change something, so that the client asks the user each\n"
        "time. A file replaced is copied first under\n"
        "~/.local/share/janas/files-backup/. Nothing is deleted.\n"
        "It reaches the home and the folders listed, a path a line, in\n"
        "~/.config/janas/files-dirs.txt; never the private files (keys,\n"
        "passwords, tokens, the browsers' data, the shells' histories).\n"
        "janas-chat starts it by itself when it is beside it.\n"
        "Configured in a client as, for instance:\n"
        "  {\"mcpServers\": {\"files\": {\"command\": \"janas-files\"}}}\n");
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        usage();
        return strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 ? 0
                                                                            : 2;
    }
    static struct fl_roots roots;
    if (fl_roots_load(&roots) != 0) {
        fprintf(stderr, "janas-files: no home (HOME)\n");
        return 1;
    }
    struct janas_mcps srv = {.name = "janas-files",
                             .version = JANAS_VERSION,
                             .instructions = INSTRUCTIONS,
                             .ctx = &roots,
                             .tools_list = fl_tools_list,
                             .tools_call = fl_tools_call};
    if (janas_mcps_open(&srv) != 0) {
        fprintf(stderr, "janas-files: cannot set its output aside\n");
        return 1;
    }
    fprintf(stderr,
            "janas-files %s: ready, 6 tools (files_list, files_read, "
            "files_search; asked first: files_write, files_edit, "
            "files_mkdir), %d folder%s reached\n",
            JANAS_VERSION, roots.n_root, roots.n_root == 1 ? "" : "s");
    janas_mcps_run(&srv);
    return 0;
}
