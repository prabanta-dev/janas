/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * python.c - the chat's python tool: the code a model writes, run in the
 * cage of src/common/sandbox.c (no network, no other process, limits on
 * memory, files and time) by the Python session of src/common/pysession.c,
 * its variables kept from one piece of code to the next. The workspace is
 * its /mnt/data, to read and write; the folders added to the workspace are
 * there read only, at their own paths; nothing else of the home. One
 * session for the workspace open, started at its first piece of code and
 * ended when the workspace or its folders change. The page runs it only
 * once the user allows each piece of code.
 */
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brebus.h"
#include "common/pysession.h"
#include "common/sandbox.h"

#define CELL_SECONDS 120
#define MAX_RO 16

static struct janas_sandbox sb;
static int on;

/* The cage dies with the thread that started it (PR_SET_PDEATHSIG), and
   each request of Brebus has a thread of its own: the sessions are started
   by this one, which lives as long as Brebus. */
static pthread_mutex_t smu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t scv = PTHREAD_COND_INITIALIZER;
static int asked, answered, started;
static const struct janas_sandbox_cfg *want_cfg;
static char *const *want_argv, *const *want_envp;
static int spawn_rc;
static char spawn_why[400];

static void *spawner(void *arg)
{
    (void)arg;
    pthread_mutex_lock(&smu);
    for (;;) {
        while (!asked)
            pthread_cond_wait(&scv, &smu);
        asked = 0;
        spawn_rc = janas_sandbox_spawn(want_cfg, want_argv, want_envp, &sb,
                                       spawn_why, sizeof spawn_why);
        answered = 1;
        pthread_cond_broadcast(&scv);
    }
    return NULL;
}

/* janas_sandbox_spawn, from the spawner's thread */
static int spawn(const struct janas_sandbox_cfg *c, char *const argv[],
                 char *const envp[], char *why, size_t n)
{
    pthread_mutex_lock(&smu);
    if (!started) {
        pthread_t t;
        if (pthread_create(&t, NULL, spawner, NULL) != 0) {
            pthread_mutex_unlock(&smu);
            snprintf(why, n, "no thread for the sandbox");
            return -1;
        }
        pthread_detach(t);
        started = 1;
    }
    want_cfg = c;
    want_argv = argv;
    want_envp = envp;
    answered = 0;
    asked = 1;
    pthread_cond_broadcast(&scv);
    while (!answered)
        pthread_cond_wait(&scv, &smu);
    int rc = spawn_rc;
    if (rc != 0)
        snprintf(why, n, "%s", spawn_why);
    pthread_mutex_unlock(&smu);
    return rc;
}

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

/* tools_mu held */
void brebus_python_stop(void)
{
    if (on)
        janas_sandbox_kill(&sb);
    on = 0;
}

int brebus_python_ok(char *why, size_t n)
{
    return janas_sandbox_available(why, n);
}

/* the folders added, read only, from workspace.json's list as tools.c
   reads it (paths there are, no ':') */
static int ro_paths(struct brebus *b, char ro[][4096], int max)
{
    struct janas_buf f = {0};
    char err[160];
    brebus_folders_get(b, &f);
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *a =
        d ? janas_json_get(janas_json_root(d), "read_only") : NULL;
    int n = 0;
    for (const struct janas_json *e =
             a && a->type == JANAS_JSON_ARRAY ? a->child : NULL;
         e && n < max; e = e->next) {
        const char *s = janas_json_str(e);
        if (s && *s == '/' && strlen(s) < 4096)
            snprintf(ro[n++], 4096, "%s", s);
    }
    janas_json_free(d);
    janas_buf_free(&f);
    return n;
}

/* tools_mu held: the session, started when there is none */
static int session(struct brebus *b, char *why, size_t n)
{
    if (on)
        return 0;
    char ws[4096];
    static char ro[MAX_RO][4096];
    if (brebus_ws_path(b, ws, sizeof ws) != 0) {
        snprintf(why, n, "no workspace");
        return -1;
    }
    int nr = ro_paths(b, ro, MAX_RO);
    const char *rl[MAX_RO + 1];
    for (int i = 0; i < nr; i++)
        rl[i] = ro[i];
    rl[nr] = NULL;
    struct janas_sandbox_cfg c = {.data_dir = ws,
                                  .ro = rl,
                                  .mem_bytes = (uint64_t)3 << 30,
                                  .file_bytes = (uint64_t)512 << 20,
                                  .tmp_bytes = (uint64_t)256 << 20};
    char *argv[] = {"/usr/bin/python3", "-I", "-c", (char *)JANAS_PY_DRIVER,
                    NULL};
    char *envp[] = {"PATH=/usr/local/bin:/usr/bin:/bin",
                    "HOME=/tmp",
                    "LANG=C.UTF-8",
                    "MPLBACKEND=Agg",
                    "MPLCONFIGDIR=/tmp/mpl",
                    "OPENBLAS_NUM_THREADS=4",
                    "OMP_NUM_THREADS=4",
                    "PYTHONDONTWRITEBYTECODE=1",
                    NULL};
    if (spawn(&c, argv, envp, why, n) != 0)
        return -1;
    on = 1;
    /* the workspace as its current folder: files written by name go there */
    struct janas_buf raw = {0};
    char *a = janas_py_exchange(&sb, "import os\nos.chdir('/mnt/data')", 20,
                                &raw, why, n);
    janas_buf_free(&raw);
    if (!a) {
        brebus_python_stop();
        return -1;
    }
    free(a);
    return 0;
}

void brebus_python_tool(struct brebus *b, struct janas_buf *o)
{
    char why[200];
    if (!brebus_python_ok(why, sizeof why))
        return;
    static char ro[MAX_RO][4096];
    int nr = ro_paths(b, ro, MAX_RO);
    struct janas_buf d = {0};
    janas_buf_puts(&d, "Runs Python 3 code in a sandbox with no network, as "
                       "a notebook: variables and imports stay from one call "
                       "to the next, the value of the last expression is "
                       "shown, figures drawn with matplotlib are saved. The "
                       "user's workspace is /mnt/data (read and write: read "
                       "its files there, and write what you make there)");
    if (nr) {
        janas_buf_puts(&d, "; these are there to read only, at their own "
                           "paths: ");
        for (int i = 0; i < nr; i++) {
            janas_buf_puts(&d, i ? ", " : "");
            janas_buf_puts(&d, ro[i]);
        }
    }
    janas_buf_puts(&d, ". Nothing else of the computer is reachable. A call "
                       "runs at most 2 minutes. The system's libraries only "
                       "(numpy, matplotlib... when installed); no pip.");
    janas_buf_put(&d, "", 1);
    janas_buf_puts(o, "{\"type\": \"function\", \"function\": {\"name\": "
                      "\"python\", \"description\": ");
    str(o, d.p ? d.p : "");
    janas_buf_puts(o,
                   ", \"parameters\": {\"type\": \"object\", \"properties\": "
                   "{\"code\": {\"type\": \"string\", \"description\": "
                   "\"the Python code to run\"}}, \"required\": "
                   "[\"code\"]}}}");
    janas_buf_free(&d);
}

int brebus_python_run(struct brebus *b, const char *code, struct janas_buf *out)
{
    char why[400];
    if (session(b, why, sizeof why) != 0) {
        janas_buf_puts(out, "{\"result\": ");
        str(out, why);
        janas_buf_puts(out, ", \"is_error\": true}");
        return 0;
    }
    struct janas_buf raw = {0};
    char *ans =
        janas_py_exchange(&sb, code, CELL_SECONDS, &raw, why, sizeof why);
    if (!ans) { /* the time up, or the session gone: started anew next */
        brebus_python_stop();
        janas_buf_puts(out, "{\"result\": ");
        str(out, why);
        janas_buf_puts(out, ", \"is_error\": true}");
        janas_buf_free(&raw);
        return 0;
    }
    /* the answer as text for the model: what it printed, its traceback,
       the figures saved (paths in the workspace, as the user sees them) */
    char err[160];
    struct janas_json_doc *d =
        janas_json_parse(ans, strlen(ans), err, sizeof err);
    const struct janas_json *r = d ? janas_json_root(d) : NULL;
    const char *logs = r ? janas_json_str(janas_json_get(r, "logs")) : NULL;
    const char *tb = r ? janas_json_str(janas_json_get(r, "error")) : NULL;
    const struct janas_json *im = r ? janas_json_get(r, "images") : NULL;
    struct janas_buf t = {0};
    if (logs && *logs)
        janas_buf_puts(&t, logs);
    if (raw.n) {
        janas_buf_puts(&t, t.n ? "\n" : "");
        janas_buf_put(&t, raw.p, raw.n < 20000 ? raw.n : 20000);
    }
    if (tb) {
        janas_buf_puts(&t, t.n ? "\n" : "");
        janas_buf_puts(&t, tb);
    }
    for (const struct janas_json *e =
             im && im->type == JANAS_JSON_ARRAY ? im->child : NULL;
         e; e = e->next) {
        const char *p = janas_json_str(e);
        if (!p)
            continue;
        janas_buf_puts(&t, t.n ? "\n" : "");
        janas_buf_puts(&t, "figure saved: ");
        janas_buf_puts(&t, p);
    }
    if (!t.n)
        janas_buf_puts(&t, "(no output)");
    janas_buf_puts(out, "{\"result\": ");
    janas_json_write_str(out, t.p, t.n);
    janas_buf_printf(out, ", \"is_error\": %s}", tb ? "true" : "false");
    janas_buf_free(&t);
    janas_json_free(d);
    janas_buf_free(&raw);
    free(ans);
    return 0;
}
