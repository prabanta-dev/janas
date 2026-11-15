/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * server.c - the model's server: a janas-server of its own, started on a
 * model of My models at a port of its own on 127.0.0.1 (Brebus's port plus
 * one), its output in the data folder (server.log). Brebus passes the
 * page's API requests on to it (proxy.c); other programs may use it at its
 * address. One model at a time: starting another stops the first.
 */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include "brebus.h"
#include "common/mcp_http.h"
#include "common/model_scan.h"

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

static int fail(struct janas_buf *out, const char *why)
{
    out->n = 0;
    janas_buf_puts(out, "{\"error\": ");
    str(out, why);
    janas_buf_puts(out, "}");
    return -1;
}

/* the server answers its models' list: ready */
static int answers(unsigned short port)
{
    char url[64], err[128];
    snprintf(url, sizeof url, "http://127.0.0.1:%u/v1/models", port);
    struct janas_url u;
    if (janas_url_parse(url, &u, err, sizeof err) != 0)
        return 0;
    struct janas_http h;
    if (janas_http_request(&h, &u, "GET", NULL, 0, NULL, 0, 2000, NULL, err,
                           sizeof err) != 0)
        return 0;
    int ok = h.head.status == 200;
    janas_http_close(&h);
    return ok;
}

static void stop_locked(struct brebus *b)
{
    if (!b->srv_on)
        return;
    janas_proc_kill(&b->srv, 0);
    int code;
    if (janas_proc_wait(&b->srv, 10000, &code) != 1) {
        janas_proc_kill(&b->srv, 1);
        janas_proc_wait(&b->srv, -1, &code);
    }
    b->srv_on = 0;
    b->srv_model[0] = 0;
}

int brebus_server_start(struct brebus *b, const char *body, size_t n,
                        struct janas_buf *out)
{
    char err[160];
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *name = o ? janas_json_str(janas_json_get(o, "model")) : NULL;
    double par = o ? janas_json_num(janas_json_get(o, "parallel"), 0) : 0;
    double ctx = o ? janas_json_num(janas_json_get(o, "context"), 0) : 0;
    const struct janas_json *th = o ? janas_json_get(o, "think") : NULL;
    if (!name) {
        janas_json_free(d);
        return fail(out, "no model named");
    }
    static struct janas_model_file m[JANAS_MODELS_MAX];
    int nm = janas_models_scan(b->models_dir, m, JANAS_MODELS_MAX), at = -1;
    for (int i = 0; i < nm && at < 0; i++)
        if (!strcmp(m[i].name, name))
            at = i;
    if (at < 0) {
        janas_json_free(d);
        return fail(out, "no such model in My models");
    }
    pthread_mutex_lock(&b->mu);
    stop_locked(b);
    char exe[4200], port[8], par_s[8], ctx_s[16], logp[4200];
    snprintf(exe, sizeof exe, "%s/janas-server%s", b->bin_dir, b->suffix);
    b->srv_port = (unsigned short)(b->port + 1);
    snprintf(port, sizeof port, "%u", b->srv_port);
    const char *argv[16];
    int k = 0;
    argv[k++] = exe;
    argv[k++] = m[at].path;
    argv[k++] = "--port";
    argv[k++] = port;
    if (par >= 1 && par <= 16) {
        snprintf(par_s, sizeof par_s, "%d", (int)par);
        argv[k++] = "--parallel";
        argv[k++] = par_s;
    }
    if (ctx >= 512) {
        snprintf(ctx_s, sizeof ctx_s, "%d", (int)ctx);
        argv[k++] = "--ctx";
        argv[k++] = ctx_s;
    }
    if (th && (th->type == JANAS_JSON_FALSE || th->type == JANAS_JSON_TRUE)) {
        argv[k++] = "--think";
        argv[k++] = th->type == JANAS_JSON_TRUE ? "on" : "off";
    }
    argv[k] = NULL;
    janas_json_free(d);
    snprintf(logp, sizeof logp, "%s/server.log", b->data_dir);
    int fd = open(logp, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    int rc = janas_proc_start(&b->srv, argv, NULL, NULL, -1, fd, fd,
                              JANAS_PROC_GROUP, err, sizeof err);
    if (fd >= 0)
        close(fd);
    if (rc != 0) {
        snprintf(b->srv_err, sizeof b->srv_err, "%s", err);
        pthread_mutex_unlock(&b->mu);
        return fail(out, err);
    }
    b->srv_on = 1;
    snprintf(b->srv_model, sizeof b->srv_model, "%s", m[at].name);
    b->srv_err[0] = 0;
    pthread_mutex_unlock(&b->mu);
    /* ready when it answers; a large model takes a while to open */
    for (int i = 0; i < 600; i++) {
        if (answers(b->srv_port))
            break;
        int code;
        pthread_mutex_lock(&b->mu);
        int gone = b->srv_on && janas_proc_wait(&b->srv, 0, &code) == 1;
        if (gone) {
            b->srv_on = 0;
            snprintf(b->srv_err, sizeof b->srv_err,
                     "janas-server ended (status %d): see %.400s", code, logp);
        }
        pthread_mutex_unlock(&b->mu);
        if (gone)
            return fail(out, b->srv_err);
        usleep(200000);
    }
    out->n = 0;
    brebus_server_status(b, out);
    return 0;
}

int brebus_server_stop(struct brebus *b, struct janas_buf *out)
{
    pthread_mutex_lock(&b->mu);
    stop_locked(b);
    pthread_mutex_unlock(&b->mu);
    out->n = 0;
    brebus_server_status(b, out);
    return 0;
}

void brebus_server_status(struct brebus *b, struct janas_buf *out)
{
    pthread_mutex_lock(&b->mu);
    int code, on = b->srv_on;
    if (on && janas_proc_wait(&b->srv, 0, &code) == 1) {
        b->srv_on = on = 0;
        snprintf(b->srv_err, sizeof b->srv_err,
                 "janas-server ended (status %d)", code);
    }
    janas_buf_printf(out,
                     "{\"running\": %s, \"model\": ", on ? "true" : "false");
    str(out, on ? b->srv_model : "");
    janas_buf_printf(out, ", \"port\": %u, \"url\": ", on ? b->srv_port : 0);
    char url[64] = "";
    if (on)
        snprintf(url, sizeof url, "http://127.0.0.1:%u/v1", b->srv_port);
    str(out, url);
    janas_buf_puts(out, ", \"error\": ");
    str(out, b->srv_err);
    janas_buf_puts(out, "}");
    pthread_mutex_unlock(&b->mu);
}

unsigned short brebus_server_port(struct brebus *b)
{
    pthread_mutex_lock(&b->mu);
    unsigned short p = b->srv_on ? b->srv_port : 0;
    pthread_mutex_unlock(&b->mu);
    return p;
}
