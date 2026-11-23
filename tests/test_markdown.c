/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_markdown.c - the Markdown of the web pages (src/web/md_parse.js,
 * markdown.js) against every example of CommonMark 0.31.2 and of GitHub's
 * extensions, and texts made to be slow: tests/markdown/check.js, run by
 * node from the repository's root. Skipped where node is not in the PATH,
 * and on Windows (the pages are the same).
 */
#include <stdio.h>

#ifdef _WIN32
int main(void)
{
    printf("test_markdown: skipped on Windows (the same pages as on Linux)\n");
    return 0;
}
#else
#include "node_run.h"

int main(void)
{
    char node[4096], out[1 << 16];
    (void)node_addon; /* the module's: not needed here */
    if (!node_find(node, sizeof(node))) {
        printf("test_markdown: skipped, no node in the PATH\n");
        return 0;
    }
    if (access("tests/markdown/check.js", R_OK) != 0) {
        printf("test_markdown: FAIL, run it from the repository's root\n");
        return 1;
    }
    int st = node_run(node, NULL,
                      "require(require('path').resolve("
                      "'tests/markdown/check.js'))",
                      NULL, NULL, 120, out, sizeof(out));
    fputs(out, stdout);
    if (st != 0) {
        printf("test_markdown: FAIL (node exit %d)\n", st);
        return 1;
    }
    return 0;
}
#endif
