/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * serve.c - janas-setup serve: the servers of the plan janas-setup editor
 * saved, one janas-server for each model, until Ctrl+C; when one stops,
 * the other is stopped too, so that the editor never finds half of it.
 */
#include "serve.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/proc.h"
#include "common/stop.h"
#include "common/sysinfo.h"
#include "plan.h"

static atomic_int stopping;

static void *await_stop(void *arg)
{
    (void)arg;
    janas_stop_wait();
    atomic_store(&stopping, 1);
    return NULL;
}

/* janas-server beside this program, with its variant's suffix
   (janas-setup-asan runs janas-server-asan), else from PATH */
static void server_path(char *buf, size_t len)
{
    char self[4096];
    if (janas_exe_path(self, sizeof self) > 0) {
        char *base = strrchr(self, '/');
        const char *tail = base ? strstr(base, "janas-setup") : NULL;
        if (tail) {
            *base = 0;
            snprintf(buf, len, "%s/janas-server%s", self,
                     tail + strlen("janas-setup"));
            return;
        }
    }
    snprintf(buf, len, "janas-server");
}

static int start(struct janas_proc *p, const char *server, const char *path,
                 const char *name, int port, unsigned ctx)
{
    char sport[16], sctx[16], err[256];
    snprintf(sport, sizeof sport, "%d", port);
    snprintf(sctx, sizeof sctx, "%u", ctx ? ctx : SETUP_CTX);
    const char *argv[] = {server, path,    "--port", sport, "--name",
                          name,   "--ctx", sctx,     NULL};
    printf("== %s on http://127.0.0.1:%d\n", name, port);
    fflush(stdout);
    if (janas_proc_start(p, argv, NULL, NULL, -1, -1, -1, JANAS_PROC_GROUP, err,
                         sizeof err) != 0) {
        fprintf(stderr, "janas-setup: cannot start %s: %s\n", server, err);
        return -1;
    }
    return 0;
}

int setup_serve(void)
{
    struct setup_saved s;
    if (setup_plan_load(&s) != 0) {
        fprintf(stderr, "janas-setup: nothing to serve yet: run janas-setup "
                        "editor first\n");
        return 1;
    }
    char server[4200];
    server_path(server, sizeof server);
    janas_stop_prepare();
    pthread_t t;
    if (pthread_create(&t, NULL, await_stop, NULL) != 0)
        return 1;
    pthread_detach(t);
    struct janas_proc p[2] = {{0}};
    int n = 0;
    if (start(&p[n], server, s.code_path, s.code_name, s.code_port, s.ctx) == 0)
        n++;
    if (n == 1 && s.complete_path[0] &&
        start(&p[n], server, s.complete_path, s.complete_name, s.complete_port,
              s.ctx) == 0)
        n++;
    int want = s.complete_path[0] ? 2 : 1, rc = n == want ? 0 : 1;
    /* the first one waited for half a second, the other only looked at */
    while (rc == 0 && !atomic_load(&stopping))
        for (int i = 0; i < n; i++) {
            int code;
            if (janas_proc_wait(&p[i], i == 0 ? 500 : 0, &code) == 1) {
                fprintf(stderr, "janas-setup: a server stopped (%d)\n", code);
                rc = 1;
            }
        }
    for (int i = 0; i < n; i++)
        if (p[i].id) {
            janas_proc_kill(&p[i], 0);
            janas_proc_wait(&p[i], -1, NULL);
        }
    return rc;
}
