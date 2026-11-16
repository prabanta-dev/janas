/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - brebus: serves its page on 127.0.0.1 and shows it in a window
 * of its own (window.c), or in the browser where there cannot be one,
 * until the window is closed or Ctrl+C.
 */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"
#include "common/mcp_http.h"
#include "common/sandbox.h"
#include "common/sysinfo.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static volatile sig_atomic_t quit;

static void on_signal(int s)
{
    (void)s;
    quit = 1;
}

static void usage(void)
{
    fprintf(stderr,
            "usage: brebus [--port N] [--workspace DIR] [--browser | "
            "--no-open]\n"
            "       brebus --install-desktop\n"
            "Brebus, Janas's app for the models of this computer: its page\n"
            "on http://127.0.0.1:41414 (--port), in a window of its own, or\n"
            "in the browser with --browser or where there is no WebKitGTK;\n"
            "--no-open: neither. Only this computer reaches it.\n"
            "It works in a folder: --workspace, or the one it is started\n"
            "in from a terminal; else the page asks.\n"
            "--install-desktop puts Brebus in the desktop's menu, with its\n"
            "icon (~/.local/share/applications).\n");
}

static void open_browser(const char *url)
{
    pid_t pid = fork();
    if (pid == 0) {
        execlp("xdg-open", "xdg-open", url, (char *)NULL);
        _exit(127);
    }
    if (pid > 0)
        waitpid(pid, NULL, 0);
}

int main(int argc, char **argv)
{
    janas_sandbox_main(argc, argv); /* a cage for python.c being built */
    struct brebus b = {.port = 41414};
    int open = 1, browser = 0;
    const char *workspace = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) {
            int p = atoi(argv[++i]);
            if (p < 1 || p > 65535) {
                usage();
                return 2;
            }
            b.port = (unsigned short)p;
        } else if (!strcmp(argv[i], "--workspace") && i + 1 < argc) {
            workspace = argv[++i];
        } else if (!strcmp(argv[i], "--no-open")) {
            open = 0;
        } else if (!strcmp(argv[i], "--install-desktop")) {
            return brebus_install_desktop(0);
        } else if (!strcmp(argv[i], "--browser")) {
            browser = 1;
        } else {
            usage();
            return !strcmp(argv[i], "-h") || !strcmp(argv[i], "--help") ? 0 : 2;
        }
    }
    if (janas_models_dir(b.models_dir, sizeof b.models_dir) != 0) {
        fprintf(stderr, "brebus: no home to keep the models in\n");
        return 1;
    }
    /* janas-server beside brebus; Brebus's own files in the data folder */
    if (janas_exe_path(b.bin_dir, sizeof b.bin_dir) > 0) {
        char *slash = strrchr(b.bin_dir, '/');
        if (slash) {
            const char *dash = strchr(slash + 1, '-');
            if (dash && strlen(dash) < sizeof b.suffix)
                snprintf(b.suffix, sizeof b.suffix, "%s", dash);
            *slash = 0;
        }
    }
    const char *xd = getenv("XDG_DATA_HOME"), *home = getenv("HOME");
    if (xd && *xd == '/')
        snprintf(b.data_dir, sizeof b.data_dir, "%s/janas/brebus", xd);
    else
        snprintf(b.data_dir, sizeof b.data_dir, "%s/.local/share/janas/brebus",
                 home ? home : ".");
    janas_mkdirs(b.data_dir);
    pthread_mutex_init(&b.mu, NULL);
    pthread_mutex_init(&b.get.mu, NULL);
    pthread_mutex_init(&b.tools_mu, NULL);
    char why[600];
    /* the workspace: the one given, or this terminal's folder; from the
       desktop's menu (no terminal, the home) the page asks */
    char cwd[4096];
    if (workspace) {
        char abs[4200];
        if (*workspace != '/' && getcwd(cwd, sizeof cwd))
            snprintf(abs, sizeof abs, "%s/%s", cwd, workspace);
        else
            snprintf(abs, sizeof abs, "%s", workspace);
        if (brebus_ws_open(&b, abs, 1, why, sizeof why) != 0) {
            fprintf(stderr, "brebus: %s\n", why);
            return 1;
        }
    } else if (isatty(STDIN_FILENO) && getcwd(cwd, sizeof cwd) &&
               brebus_ws_open(&b, cwd, 0, why, sizeof why) != 0)
        fprintf(stderr, "brebus: %s; the page will ask for a folder\n", why);
    /* the models' folder chosen in the page, where JANAS_MODELS says none */
    {
        struct janas_buf f = {0};
        char err[160];
        brebus_settings_get(&b, &f);
        struct janas_json_doc *d =
            f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
        const char *md = d ? janas_json_str(janas_json_get(janas_json_root(d),
                                                           "_models_dir"))
                           : NULL;
        const char *env = getenv("JANAS_MODELS");
        if (md && !(env && *env))
            brebus_models_dir_set(&b, md, 0);
        janas_json_free(d);
        janas_buf_free(&f);
    }
    if (brebus_http_start(&b, why, sizeof why) != 0) {
        /* a Brebus there already (started at login, or from the menu
           before): its window onto it, this one ends with it */
        char url0[64];
        snprintf(url0, sizeof url0, "http://127.0.0.1:%u/", b.port);
        struct janas_url u;
        struct janas_http h;
        char e2[160], probe[96];
        snprintf(probe, sizeof probe, "%sbrebus/workspace", url0);
        int there = janas_url_parse(probe, &u, e2, sizeof e2) == 0 &&
                    janas_http_request(&h, &u, "GET", NULL, 0, NULL, 0, 2000,
                                       NULL, e2, sizeof e2) == 0;
        if (there) {
            there = h.head.status == 200;
            janas_http_close(&h);
        }
        if (there && open) {
            if (browser || brebus_window(url0, &quit, why, sizeof why) != 0)
                open_browser(url0);
            return 0;
        }
        fprintf(stderr, "brebus: %s\n", why);
        return 1;
    }
    brebus_idle_start(&b);
    char url[64];
    snprintf(url, sizeof url, "http://127.0.0.1:%u/", b.port);
    fprintf(stderr, "brebus %s: %s (Ctrl+C ends it)\n", JANAS_VERSION, url);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    if (open && !browser && brebus_window(url, &quit, why, sizeof why) == 0)
        quit = 1; /* the window closed */
    else if (open) {
        if (!browser)
            fprintf(stderr, "brebus: %s; in the browser instead\n", why);
        open_browser(url);
    }
    while (!quit)
        pause();
    brebus_http_stop(&b);
    struct janas_buf o = {0};
    brebus_server_stop(&b, &o); /* the model's server goes with it */
    brebus_get_stop(&b);        /* a download too, to go on next time */
    brebus_tools_stop(&b);
    janas_buf_free(&o);
    return 0;
}
