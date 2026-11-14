/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * share.c - see share.h.
 *
 * GitHub's issue forms take their fields from the address: template= the
 * form (01-hardware-report.yml), title=, and each field by its id (report,
 * system, memory). A report is some 2-3 KB, 4-6 KB once encoded; past
 * MAX_URL the address carries the title and the short fields alone, and the
 * user drags the file into the form, as the form itself says.
 */
#include "share.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#else
#include <fcntl.h>
#include <sys/wait.h>
#endif

#include "common/bench_report.h"
#include "llm/json.h"

#define REPO "https://github.com/prabanta-dev/janas"
#define MAX_URL 7500 /* what browsers and GitHub take, with room */

static void enc(struct janas_buf *b, const char *s, size_t n)
{
    static const char hex[] = "0123456789ABCDEF";
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
            c == '~') {
            janas_buf_put(b, (const char *)&c, 1);
        } else {
            char e[3] = {'%', hex[c >> 4], hex[c & 15]};
            janas_buf_put(b, e, 3);
        }
    }
}

static int first(void *ctx, const struct janas_bench_row *r)
{
    *(struct janas_bench_row *)ctx = *r;
    return -1; /* the first is enough */
}

/* The text of a "label:" line of the report, to its end. */
static void line_of(const char *s, const char *label, char *out, size_t cap)
{
    out[0] = 0;
    const char *p = strstr(s, label);
    if (!p)
        return;
    p += strlen(label);
    while (*p == ' ')
        p++;
    size_t n = strcspn(p, "\r\n");
    if (n >= cap)
        n = cap - 1;
    memcpy(out, p, n);
    out[n] = 0;
}

static int open_in_browser(const char *url)
{
#ifdef _WIN32
    return (INT_PTR)ShellExecuteA(NULL, "open", url, NULL, NULL,
                                  SW_SHOWNORMAL) > 32
               ? 0
               : -1;
#else
    pid_t pid = fork();
    if (pid < 0)
        return -1;
    if (pid == 0) { /* the browser's own messages are not the report's */
        int fd = open("/dev/null", O_WRONLY);
        if (fd >= 0) {
            dup2(fd, 1);
            dup2(fd, 2);
        }
        execlp("xdg-open", "xdg-open", url, (char *)NULL);
        _exit(127);
    }
    int st = 0;
    return waitpid(pid, &st, 0) == pid && WIFEXITED(st) && WEXITSTATUS(st) == 0
               ? 0
               : -1;
#endif
}

void bench_share(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("--share: cannot read %s\n", path);
        return;
    }
    struct janas_buf rep = {0};
    char tmp[4096];
    size_t k;
    while ((k = fread(tmp, 1, sizeof tmp, f)) > 0)
        janas_buf_put(&rep, tmp, k);
    fclose(f);
    janas_buf_put(&rep, "", 0);
    if (rep.oom || !rep.n) {
        janas_buf_free(&rep);
        return;
    }
    struct janas_bench_row r = {0};
    janas_bench_report_scan(rep.p, rep.n - 1, first, &r);
    char title[256], sys[192], mem[96];
    snprintf(title, sizeof title, "[hardware] %s%s%s",
             r.cpu[0] ? r.cpu : "my machine", r.model[0] ? ", " : "", r.model);
    line_of(rep.p, "\nsystem:", sys, sizeof sys);
    line_of(rep.p, "\nmemory:", mem, sizeof mem);

    struct janas_buf u = {0};
    janas_buf_puts(&u, REPO "/issues/new?template=01-hardware-report.yml");
    janas_buf_puts(&u, "&title=");
    enc(&u, title, strlen(title));
    if (sys[0]) {
        janas_buf_puts(&u, "&system=");
        enc(&u, sys, strlen(sys));
    }
    if (mem[0]) {
        janas_buf_puts(&u, "&memory=");
        enc(&u, mem, strlen(mem));
    }
    size_t short_n = u.n;
    janas_buf_puts(&u, "&report=");
    enc(&u, rep.p, rep.n - 1);
    int whole = u.n <= MAX_URL;
    if (!whole)
        u.n = short_n; /* the report dragged in by hand */
    janas_buf_put(&u, "", 0);
    janas_buf_free(&rep);
    if (u.oom) {
        janas_buf_free(&u);
        return;
    }

    printf("\nTo add this machine to the project's list of reports "
           "(docs/reports.md), open this address: a new issue with the "
           "form filled in%s. Nothing is sent until you send it there.\n\n"
           "%s\n",
           whole ? ""
                 : "; drag the report's file into its first box, it is "
                   "too long for an address",
           u.p);
    if (isatty(0) && isatty(1)) {
        printf("\nOpen it in your browser now? [y/N] ");
        fflush(stdout);
        char a[16] = "";
        if (fgets(a, sizeof a, stdin) && (a[0] == 'y' || a[0] == 'Y') &&
            open_in_browser(u.p) != 0)
            printf("the browser did not open: copy the address above\n");
    }
    janas_buf_free(&u);
}
