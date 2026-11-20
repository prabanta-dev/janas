/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * get.c - a model of the catalog downloaded from the page: janas-get, the
 * one beside brebus, started for it into the models' folder, several at a
 * time, each only when the disk has room for it beside those going on. Its
 * output is read as it comes: the "== ..." lines are the step it is at, "   x
 * of y GB (p%), m MB/s" how far the download is, and its last other line what
 * it said last, the reason when it fails. Stopped, it leaves its .part file:
 * started again, the download goes on from there. Besides the catalog's models,
 * any file of Hugging Face as janas-get takes it, hf:<owner>/<repo>/<file>: a
 * JNS as it is, a GGUF converted here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/statvfs.h>
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
    struct brebus_get *g = arg;
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

/* hf:<owner>/<repo>/<path>.jns or .gguf, as janas-get takes it: letters,
   digits, . - _ in each part, the path with folders but never a part
   that is "." or "..", nor an option */
static int good_hf(const char *spec)
{
    if (strncmp(spec, "hf:", 3) || strlen(spec) > 400)
        return 0;
    const char *s = spec + 3, *part = s;
    int parts = 0;
    for (const char *c = s;; c++) {
        if (*c == '/' || !*c) {
            size_t n = (size_t)(c - part);
            if (!n || (n == 1 && part[0] == '.') ||
                (n == 2 && part[0] == '.' && part[1] == '.') || part[0] == '-')
                return 0;
            parts++;
            if (!*c)
                break;
            part = c + 1;
            continue;
        }
        char ch = *c;
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
              (ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == '_'))
            return 0;
    }
    size_t n = strlen(s);
    return parts >= 3 && ((n > 4 && !strcmp(s + n - 4, ".jns")) ||
                          (n > 5 && !strcmp(s + n - 5, ".gguf")));
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
    char err[256];
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *name = o ? janas_json_str(janas_json_get(o, "model")) : NULL;
    const struct janas_model_entry *m = name ? janas_catalog_find(name) : NULL;
    int hf = name && good_hf(name);
    if (!m && !hf) {
        const char *why = name && !strncmp(name, "hf:", 3)
                              ? "not a file of Hugging Face: "
                                "hf:<owner>/<repo>/<file>.gguf or .jns"
                              : "no such model in the catalog";
        janas_json_free(d); /* name is in it */
        return fail(out, why);
    }
    /* what it writes: a catalog's model its size (a GGUF converted here:
       twice, the GGUF and the JNS, while it works), a file of Hugging Face
       the size the page saw, a GGUF twice too */
    double need = m ? (m->jns_repo ? m->jns_gb : m->gguf_gb * 2.0)
                    : janas_json_num(janas_json_get(o, "bytes"), 0) / 1e9 *
                          (strstr(name, ".gguf") ? 2.0 : 1.0);
    const char *who = m ? m->name : name;
    /* a free place, the same model not twice, and the room: the disk's
       free space less what the others still have to write */
    struct brebus_get *g = NULL;
    double owed = 0;
    for (int i = 0; i < BREBUS_GETS; i++) {
        struct brebus_get *x = &b->get[i];
        pthread_mutex_lock(&x->mu);
        int on = x->running, same = on && !strcmp(x->model, who);
        if (on)
            owed += x->need_gb > x->done_gb ? x->need_gb - x->done_gb : 0;
        pthread_mutex_unlock(&x->mu);
        if (same) {
            janas_json_free(d);
            return fail(out, "it is being downloaded already");
        }
        if (!on && !g)
            g = x;
    }
    if (!g) {
        janas_json_free(d);
        return fail(out, "four downloads are going on: wait for one to end");
    }
    struct statvfs vf;
    double free_gb = statvfs(b->models_dir, &vf) == 0
                         ? (double)vf.f_bavail * (double)vf.f_frsize / 1e9
                         : -1;
    if (free_gb >= 0 && need > 0 && free_gb - owed < need + 1.0) {
        char why[4500];
        snprintf(why, sizeof why,
                 "not enough room: it needs about %.1f GB in %s, where %.1f "
                 "GB are free%s",
                 need + 1.0, b->models_dir,
                 free_gb - owed > 0 ? free_gb - owed : 0,
                 owed > 0 ? " once the downloads going on are done" : "");
        janas_json_free(d);
        return fail(out, why);
    }
    pthread_mutex_lock(&g->mu);
    if (g->joinable) {
        pthread_join(g->th, NULL);
        g->joinable = 0;
    }
    char exe[4200];
    snprintf(exe, sizeof exe, "%s/janas-get%s", b->bin_dir, b->suffix);
    const char *argv[] = {exe, m ? m->name : name, "--dir", b->models_dir,
                          NULL};
    int fd[2];
    if (janas_pipe(fd) != 0) {
        pthread_mutex_unlock(&g->mu);
        janas_json_free(d);
        return fail(out, "no pipe for janas-get");
    }
    int in = janas_null_fd(0);
    int rc =
        janas_proc_start(&g->proc, argv, NULL, NULL, in, fd[1], fd[1],
                         JANAS_PROC_GROUP | BREBUS_NO_CONSOLE, err, sizeof err);
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
    snprintf(g->model, sizeof g->model, "%s", m ? m->name : name);
    g->step[0] = g->said[0] = 0;
    g->done_gb = g->total_gb = g->pct = g->mbs = 0;
    g->running = 1;
    g->stopped = 0;
    g->code = 0;
    g->need_gb = need;
    if (pthread_create(&g->th, NULL, reader, g) != 0) {
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

void brebus_get_stop(struct brebus *b, const char *model)
{
    for (int i = 0; i < BREBUS_GETS; i++) {
        struct brebus_get *g = &b->get[i];
        pthread_mutex_lock(&g->mu);
        if (model && strcmp(g->model, model)) {
            pthread_mutex_unlock(&g->mu);
            continue;
        }
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
}

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

static void one_status(struct brebus_get *g, struct janas_buf *out)
{
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

/* every download of this run, going on or ended */
void brebus_get_status(struct brebus *b, struct janas_buf *out)
{
    janas_buf_puts(out, "{\"downloads\": [");
    int k = 0;
    for (int i = 0; i < BREBUS_GETS; i++) {
        struct brebus_get *g = &b->get[i];
        pthread_mutex_lock(&g->mu);
        int any = g->model[0] != 0;
        pthread_mutex_unlock(&g->mu);
        if (!any)
            continue;
        janas_buf_puts(out, k++ ? ", " : "");
        one_status(g, out);
    }
    janas_buf_puts(out, "]}");
}
