/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * get.c - a model of the catalog downloaded from the page: janas-get, the
 * one beside brebus, started for it into the models' folder, one at a
 * time. Its output is read as it comes: the "== ..." lines are the step
 * it is at, "   x of y GB (p%), m MB/s" how far the download is, and its
 * last other line what it said last, the reason when it fails. Stopped,
 * it leaves its .part file: started again, the download goes on from there.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"

/* one line of janas-get's output, into the state */
static void line(struct brebus_get *g, char *s)
{
    while (*s == ' ')
        s++;
    if (!*s)
        return;
    double done, total, pct, mbs;
    if (sscanf(s, "%lf of %lf GB (%lf%%), %lf MB/s", &done, &total, &pct,
               &mbs) == 4) {
        g->done_gb = done;
        g->total_gb = total;
        g->pct = pct;
        g->mbs = mbs;
        return;
    }
    if (!strncmp(s, "== ", 3)) {
        snprintf(g->step, sizeof g->step, "%s", s + 3);
        g->done_gb = g->total_gb = g->pct = g->mbs = 0;
        return;
    }
    snprintf(g->said, sizeof g->said, "%s", s);
}

static void *reader(void *arg)
{
    struct brebus *b = arg;
    struct brebus_get *g = &b->get;
    char buf[4096], cur[1024];
    size_t n = 0;
    ssize_t r;
    while ((r = read(g->fd, buf, sizeof buf)) > 0) {
        pthread_mutex_lock(&g->mu);
        for (ssize_t i = 0; i < r; i++) {
            char c = buf[i];
            if (c == '\n' || c == '\r') {
                cur[n] = 0;
                line(g, cur);
                n = 0;
            } else if (n + 1 < sizeof cur) {
                cur[n++] = c;
            }
        }
        pthread_mutex_unlock(&g->mu);
    }
    close(g->fd);
    /* waited for under the lock, a little at a time: a stop never signals
       a process already reaped */
    int code = -1;
    for (;;) {
        pthread_mutex_lock(&g->mu);
        if (janas_proc_wait(&g->proc, 0, &code) != 0)
            break;
        pthread_mutex_unlock(&g->mu);
        usleep(50000);
    }
    if (n) {
        cur[n] = 0;
        line(g, cur);
    }
    g->running = 0;
    g->code = g->stopped ? -2 : code;
    pthread_mutex_unlock(&g->mu);
    return NULL;
}

static int fail(struct janas_buf *out, const char *why)
{
    janas_buf_puts(out, "{\"error\": ");
    janas_json_write_str(out, why, strlen(why));
    janas_buf_puts(out, "}");
    return -1;
}

int brebus_get_start(struct brebus *b, const char *body, size_t n,
                     struct janas_buf *out)
{
    struct brebus_get *g = &b->get;
    char err[256];
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *name = o ? janas_json_str(janas_json_get(o, "model")) : NULL;
    const struct janas_model_entry *m = name ? janas_catalog_find(name) : NULL;
    if (!m) {
        janas_json_free(d);
        return fail(out, "no such model in the catalog");
    }
    pthread_mutex_lock(&g->mu);
    if (g->running) {
        pthread_mutex_unlock(&g->mu);
        janas_json_free(d);
        return fail(out, "a download is already going on");
    }
    if (g->joinable) {
        pthread_join(g->th, NULL);
        g->joinable = 0;
    }
    char exe[4200];
    snprintf(exe, sizeof exe, "%s/janas-get%s", b->bin_dir, b->suffix);
    const char *argv[] = {exe, m->name, "--dir", b->models_dir, NULL};
    int fd[2];
    if (janas_pipe(fd) != 0) {
        pthread_mutex_unlock(&g->mu);
        janas_json_free(d);
        return fail(out, "no pipe for janas-get");
    }
    int in = janas_null_fd(0);
    int rc = janas_proc_start(&g->proc, argv, NULL, NULL, in, fd[1], fd[1],
                              JANAS_PROC_GROUP, err, sizeof err);
    if (in >= 0)
        close(in);
    close(fd[1]);
    if (rc != 0) {
        close(fd[0]);
        pthread_mutex_unlock(&g->mu);
        janas_json_free(d);
        return fail(out, err);
    }
    g->fd = fd[0];
    snprintf(g->model, sizeof g->model, "%s", m->name);
    g->step[0] = g->said[0] = 0;
    g->done_gb = g->total_gb = g->pct = g->mbs = 0;
    g->running = 1;
    g->stopped = 0;
    g->code = 0;
    if (pthread_create(&g->th, NULL, reader, b) != 0) {
        janas_proc_kill(&g->proc, 1);
        janas_proc_wait(&g->proc, -1, NULL);
        close(g->fd);
        g->running = 0;
        pthread_mutex_unlock(&g->mu);
        janas_json_free(d);
        return fail(out, "no thread for janas-get");
    }
    g->joinable = 1;
    pthread_mutex_unlock(&g->mu);
    janas_json_free(d);
    brebus_get_status(b, out);
    return 0;
}

void brebus_get_stop(struct brebus *b)
{
    struct brebus_get *g = &b->get;
    pthread_mutex_lock(&g->mu);
    if (g->running) {
        g->stopped = 1;
        janas_proc_kill(&g->proc, 0);
    }
    int join = g->joinable;
    g->joinable = 0;
    pthread_mutex_unlock(&g->mu);
    if (join)
        pthread_join(g->th, NULL);
}

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

void brebus_get_status(struct brebus *b, struct janas_buf *out)
{
    struct brebus_get *g = &b->get;
    pthread_mutex_lock(&g->mu);
    janas_buf_printf(
        out, "{\"running\": %s, \"model\": ", g->running ? "true" : "false");
    str(out, g->model);
    janas_buf_puts(out, ", \"step\": ");
    str(out, g->step);
    janas_buf_puts(out, ", \"said\": ");
    str(out, g->said);
    /* -2: stopped from the page; else janas-get's exit status */
    janas_buf_printf(out,
                     ", \"done_gb\": %.2f, \"total_gb\": %.2f, \"percent\": "
                     "%.0f, \"mb_s\": %.0f, \"code\": %d}",
                     g->done_gb, g->total_gb, g->pct, g->mbs,
                     g->running ? 0 : g->code);
    pthread_mutex_unlock(&g->mu);
}
