/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * http.c - Brebus's daemon, on 127.0.0.1 alone: its page at /, the
 * answers of api.c at /brebus/<what>. A web page of any site could make
 * the browser send requests to the loopback address, and a site whose name
 * resolves to 127.0.0.1 could read the answers: the Host must name this
 * daemon, and an Origin, when a browser sends one, must be its own page's.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#include <microhttpd.h>
#include <netinet/in.h>

#include "brebus.h"
#include "brebus_app_html.h"   /* made by build.sh from app.html */
#include "brebus_frame_html.h" /* and from frame.html */

static const char CSP[] =
    "default-src 'none'; script-src 'unsafe-inline' 'self'; style-src "
    "'unsafe-inline' 'self'; font-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-src 'self'; base-uri 'none'; form-action "
    "'none'; frame-ancestors 'none'";

/* "127.0.0.1:port" or "localhost:port" */
static int ours(const struct brebus *b, const char *host)
{
    char a[64], c[64];
    snprintf(a, sizeof a, "127.0.0.1:%u", b->port);
    snprintf(c, sizeof c, "localhost:%u", b->port);
    return host && (!strcmp(host, a) || !strcmp(host, c));
}

static enum MHD_Result reply(struct MHD_Connection *conn, unsigned status,
                             const char *type, const char *body, size_t n,
                             int is_static)
{
    struct MHD_Response *r =
        is_static ? MHD_create_response_from_buffer_static(n, body)
                  : MHD_create_response_from_buffer_copy(n, body);
    if (!r)
        return MHD_NO;
    MHD_add_response_header(r, "Content-Type", type);
    MHD_add_response_header(r, "X-Content-Type-Options", "nosniff");
    MHD_add_response_header(r, "Cache-Control", "no-store");
    MHD_add_response_header(r, "Referrer-Policy", "no-referrer");
    if (!strncmp(type, "text/html", 9))
        MHD_add_response_header(r, "Content-Security-Policy", CSP);
    enum MHD_Result ret = MHD_queue_response(conn, status, r);
    MHD_destroy_response(r);
    return ret;
}

#define MAX_BODY (32u << 20)

/* a request's body, gathered as it comes */
struct req {
    struct janas_buf body;
    int too_big;
};

static void completed(void *cls, struct MHD_Connection *conn, void **state,
                      enum MHD_RequestTerminationCode why)
{
    (void)cls, (void)conn, (void)why;
    struct req *r = *state;
    if (r) {
        janas_buf_free(&r->body);
        free(r);
        *state = NULL;
    }
}

static enum MHD_Result json(struct MHD_Connection *conn, unsigned status,
                            struct janas_buf *o)
{
    enum MHD_Result r = o->oom ? reply(conn, 500, "application/json",
                                       "{\"error\": \"out of memory\"}", 27, 1)
                               : reply(conn, status, "application/json",
                                       o->p ? o->p : "", o->n, 0);
    janas_buf_free(o);
    return r;
}

static enum MHD_Result handler(void *cls, struct MHD_Connection *conn,
                               const char *url, const char *method,
                               const char *version, const char *upload,
                               size_t *upload_n, void **state)
{
    (void)version;
    struct brebus *b = cls;
    if (!*state) { /* the first call: the head, and room for the body */
        const char *host =
            MHD_lookup_connection_value(conn, MHD_HEADER_KIND, "Host");
        const char *origin =
            MHD_lookup_connection_value(conn, MHD_HEADER_KIND, "Origin");
        char own[2][64];
        snprintf(own[0], sizeof own[0], "http://127.0.0.1:%u", b->port);
        snprintf(own[1], sizeof own[1], "http://localhost:%u", b->port);
        /* the libraries are anyone's: an artifact's frame, of no origin,
           asks for them too (its fonts with an Origin: null) */
        int lib = !strcmp(method, "GET") && !strncmp(url, "/brebus/lib/", 12);
        if (!ours(b, host) || (!lib && origin && strcmp(origin, own[0]) &&
                               strcmp(origin, own[1]))) {
            static const char no[] = "{\"error\": \"not this page's\"}";
            return reply(conn, 403, "application/json", no, sizeof no - 1, 1);
        }
        struct req *r = calloc(1, sizeof(*r));
        if (!r)
            return MHD_NO;
        *state = r;
        return MHD_YES;
    }
    struct req *r = *state;
    if (*upload_n) { /* more of the body */
        if (r->body.n + *upload_n > MAX_BODY)
            r->too_big = 1;
        else
            janas_buf_put(&r->body, upload, *upload_n);
        *upload_n = 0;
        return MHD_YES;
    }
    if (r->too_big) {
        static const char no[] = "{\"error\": \"the body is too large\"}";
        return reply(conn, 413, "application/json", no, sizeof no - 1, 1);
    }
    int get = !strcmp(method, "GET"), post = !strcmp(method, "POST");
    if (get && (!strcmp(url, "/") || !strcmp(url, "/index.html")))
        return reply(conn, 200, "text/html; charset=utf-8",
                     janas_brebus_app_html, sizeof janas_brebus_app_html - 1,
                     1);
    if (get && !strcmp(url, "/brebus/frame")) {
        /* an artifact's frame: its own policy, no network, scripts inline
           and from the libraries, the page of this Brebus its only parent */
        const char *host =
            MHD_lookup_connection_value(conn, MHD_HEADER_KIND, "Host");
        char csp[1024];
        snprintf(csp, sizeof csp,
                 "default-src 'none'; script-src 'unsafe-inline' "
                 "'unsafe-eval' http://%s/brebus/lib/; style-src "
                 "'unsafe-inline' http://%s/brebus/lib/; font-src data: "
                 "http://%s/brebus/lib/; img-src data: blob:; media-src data: "
                 "blob:; worker-src blob:; connect-src 'none'; form-action "
                 "'none'; base-uri 'none'; frame-ancestors http://%s",
                 host, host, host, host);
        struct MHD_Response *rs = MHD_create_response_from_buffer_static(
            sizeof janas_brebus_frame_html - 1, janas_brebus_frame_html);
        if (!rs)
            return MHD_NO;
        MHD_add_response_header(rs, "Content-Type", "text/html; charset=utf-8");
        MHD_add_response_header(rs, "X-Content-Type-Options", "nosniff");
        MHD_add_response_header(rs, "Cache-Control", "no-store");
        MHD_add_response_header(rs, "Content-Security-Policy", csp);
        enum MHD_Result ret = MHD_queue_response(conn, 200, rs);
        MHD_destroy_response(rs);
        return ret;
    }
    if (get && !strncmp(url, "/brebus/lib/", 12)) {
        struct janas_buf o = {0};
        const char *mime = NULL;
        char err[400];
        if (brebus_lib(b, url + 12, &o, &mime, err, sizeof err) != 0) {
            janas_buf_free(&o);
            struct janas_buf e = {0};
            janas_buf_puts(&e, "{\"error\": ");
            janas_json_write_str(&e, err, strlen(err));
            janas_buf_puts(&e, "}");
            return json(conn, 404, &e);
        }
        struct MHD_Response *rs =
            MHD_create_response_from_buffer_copy(o.n, o.p);
        janas_buf_free(&o);
        if (!rs)
            return MHD_NO;
        MHD_add_response_header(rs, "Content-Type", mime);
        MHD_add_response_header(rs, "X-Content-Type-Options", "nosniff");
        /* the same file under the same path, always */
        MHD_add_response_header(rs, "Cache-Control",
                                "public, max-age=31536000, immutable");
        MHD_add_response_header(rs, "Access-Control-Allow-Origin", "*");
        enum MHD_Result ret = MHD_queue_response(conn, 200, rs);
        MHD_destroy_response(rs);
        return ret;
    }
    if (get && !strcmp(url, "/prabanta.png")) {
        const struct brebus_icon *i = brebus_icon(64);
        return reply(conn, 200, "image/png", (const char *)i->png, i->len, 1);
    }
    if (!strncmp(url, "/v1/", 4) || !strncmp(url, "/api/", 5) ||
        (get && (!strcmp(url, "/monitor") || !strcmp(url, "/monitor/") ||
                 !strcmp(url, "/monitor/events")))) {
        struct MHD_Response *resp = NULL;
        unsigned status = 0;
        if (brebus_proxy(b, conn, method, url, r->body.p, r->body.n, &resp,
                         &status) != 0) {
            static const char no[] =
                "{\"error\": \"no model is loaded: start one in My "
                "models\"}";
            return reply(conn, 503, "application/json", no, sizeof no - 1, 1);
        }
        enum MHD_Result ret = MHD_queue_response(conn, status, resp);
        MHD_destroy_response(resp);
        return ret;
    }
    struct janas_buf o = {0};
    if (post && !strcmp(url, "/brebus/server/start")) {
        int rc = brebus_server_start(b, r->body.p, r->body.n, &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (post && !strcmp(url, "/brebus/server/stop")) {
        brebus_server_stop(b, &o);
        return json(conn, 200, &o);
    }
    if (get && !strcmp(url, "/brebus/chats/search")) {
        /* the filters of the page's search, all optional */
        const char *since = MHD_lookup_connection_value(
                       conn, MHD_GET_ARGUMENT_KIND, "since"),
                   *arch = MHD_lookup_connection_value(
                       conn, MHD_GET_ARGUMENT_KIND, "archived");
        struct brebus_find f = {.in = MHD_lookup_connection_value(
                                    conn, MHD_GET_ARGUMENT_KIND, "in"),
                                .where = MHD_lookup_connection_value(
                                    conn, MHD_GET_ARGUMENT_KIND, "where"),
                                .model = MHD_lookup_connection_value(
                                    conn, MHD_GET_ARGUMENT_KIND, "model"),
                                .since = since ? atof(since) : 0,
                                .archived = !arch || strcmp(arch, "0")};
        brebus_chats_find(
            b, MHD_lookup_connection_value(conn, MHD_GET_ARGUMENT_KIND, "q"),
            &f, &o);
        return json(conn, 200, &o);
    }
    /* before /brebus/chats/<id>, which would take "move" for an id */
    if (post && !strcmp(url, "/brebus/chats/move")) {
        char why[300];
        int rc = brebus_chat_move_to(b, r->body.p, r->body.n, why, sizeof why);
        if (rc) {
            janas_buf_puts(&o, "{\"error\": ");
            janas_json_write_str(&o, why, strlen(why));
            janas_buf_puts(&o, "}");
        } else
            janas_buf_puts(&o, "{\"ok\": true}");
        return json(conn, rc ? 400 : 200, &o);
    }
    if (!strncmp(url, "/brebus/chats/", 14)) {
        const char *id = url + 14;
        /* ?ws=: a chat of another project (or of the free chats), from
           the side bar's menu, without opening it */
        const char *in =
            MHD_lookup_connection_value(conn, MHD_GET_ARGUMENT_KIND, "ws");
        char real[4096];
        int rc = -1;
        if (in && brebus_ws_known(b, in, real, sizeof real) != 0)
            rc = -1;
        else if (in && get)
            rc = brebus_chat_get_in(real, id, &o);
        else if (in && post)
            rc = brebus_chat_put_in(real, id, r->body.p, r->body.n);
        else if (in && !strcmp(method, "DELETE"))
            rc = brebus_chat_delete_in(real, id);
        else if (get)
            rc = brebus_chat_get(b, id, &o);
        else if (post)
            rc = brebus_chat_put(b, id, r->body.p, r->body.n);
        else if (!strcmp(method, "DELETE"))
            rc = brebus_chat_delete(b, id);
        if (rc == 0) {
            if (!get)
                janas_buf_puts(&o, "{\"ok\": true}");
            return json(conn, 200, &o);
        }
        janas_buf_free(&o);
        static const char no[] =
            "{\"error\": \"no such conversation, or not a JSON object\"}";
        return reply(conn, get ? 404 : 400, "application/json", no,
                     sizeof no - 1, 1);
    }
    if (post && !strcmp(url, "/brebus/attach")) {
        int rc = brebus_attach(b, r->body.p, r->body.n, &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (get && !strcmp(url, "/brebus/file")) {
        const char *path =
            MHD_lookup_connection_value(conn, MHD_GET_ARGUMENT_KIND, "path");
        const char *mime = NULL;
        if (brebus_file(b, path, &o, &mime) == 0) {
            enum MHD_Result ret = reply(conn, 200, mime, o.p, o.n, 0);
            janas_buf_free(&o);
            return ret;
        }
        janas_buf_free(&o);
        static const char no[] =
            "{\"error\": \"not a file of the workspace or of its folders\"}";
        return reply(conn, 404, "application/json", no, sizeof no - 1, 1);
    }
    if (post && !strcmp(url, "/brebus/pick")) {
        /* {"file": true|false, "title"} - the window's dialog */
        char err[160], path[4096];
        struct janas_json_doc *d =
            r->body.n ? janas_json_parse(r->body.p, r->body.n, err, sizeof err)
                      : NULL;
        const struct janas_json *v = d ? janas_json_root(d) : NULL;
        const struct janas_json *f = v ? janas_json_get(v, "file") : NULL;
        const char *t = v ? janas_json_str(janas_json_get(v, "title")) : NULL;
        int rc = brebus_pick(f && f->type == JANAS_JSON_TRUE, t ? t : "Brebus",
                             path, sizeof path);
        janas_json_free(d);
        if (rc != 0) {
            static const char no[] = "{\"error\": \"no window\"}";
            return reply(conn, 501, "application/json", no, sizeof no - 1, 1);
        }
        janas_buf_puts(&o, "{\"path\": ");
        janas_json_write_str(&o, path, strlen(path));
        janas_buf_puts(&o, "}");
        return json(conn, 200, &o);
    }
    if (get && !strcmp(url, "/brebus/hf/check")) {
        int rc = brebus_hf_check(
            b, MHD_lookup_connection_value(conn, MHD_GET_ARGUMENT_KIND, "repo"),
            MHD_lookup_connection_value(conn, MHD_GET_ARGUMENT_KIND, "file"),
            &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (get && (!strcmp(url, "/brebus/hf/search") ||
                !strcmp(url, "/brebus/hf/files"))) {
        const char *q = MHD_lookup_connection_value(
            conn, MHD_GET_ARGUMENT_KIND, url[11] == 's' ? "q" : "repo");
        int rc = url[11] == 's' ? brebus_hf_search(b, q, &o)
                                : brebus_hf_files(b, q, &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (post && (!strcmp(url, "/brebus/models-dir") ||
                 !strcmp(url, "/brebus/autostart"))) {
        char err[160];
        struct janas_json_doc *d =
            r->body.n ? janas_json_parse(r->body.p, r->body.n, err, sizeof err)
                      : NULL;
        const struct janas_json *v = d ? janas_json_root(d) : NULL;
        int rc;
        if (url[8] == 'm')
            rc = brebus_models_dir_set(
                b, v ? janas_json_str(janas_json_get(v, "path")) : NULL, 1);
        else {
            const struct janas_json *on = v ? janas_json_get(v, "on") : NULL;
            rc = brebus_autostart(on && on->type == JANAS_JSON_TRUE);
        }
        janas_json_free(d);
        janas_buf_printf(&o, "{\"ok\": %s, \"autostart\": %s}",
                         rc ? "false" : "true",
                         brebus_autostart_on() ? "true" : "false");
        return json(conn, rc ? 400 : 200, &o);
    }
    if (get && !strcmp(url, "/brebus/python")) {
        brebus_python_status(&o);
        return json(conn, 200, &o);
    }
    if (post && (!strcmp(url, "/brebus/python/install") ||
                 !strcmp(url, "/brebus/python/remove"))) {
        int install = url[15] == 'i', rc;
        if (install)
            rc = brebus_python_install();
        else {
            /* the session running on it ended first */
            pthread_mutex_lock(&b->tools_mu);
            brebus_python_stop();
            pthread_mutex_unlock(&b->tools_mu);
            rc = brebus_python_remove();
        }
        brebus_python_status(&o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (get && !strcmp(url, "/brebus/services")) {
        brebus_services_get(b, &o);
        return json(conn, 200, &o);
    }
    if (get && !strcmp(url, "/brebus/tools")) {
        brebus_tools_list(b, &o);
        return json(conn, 200, &o);
    }
    if (post && !strcmp(url, "/brebus/tools/restart")) {
        brebus_tools_start(b); /* the settings changed: the embedding */
        brebus_tools_list(b, &o);
        return json(conn, 200, &o);
    }
    if (post && !strcmp(url, "/brebus/tools/call")) {
        int rc = brebus_tools_call(b, r->body.p, r->body.n, &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (!strcmp(url, "/brebus/folders") && (get || post)) {
        if (get) {
            brebus_folders_get(b, &o);
            return json(conn, 200, &o);
        }
        int rc = brebus_folders_set(b, r->body.p, r->body.n, &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (get && !strcmp(url, "/brebus/workspace")) {
        brebus_ws_status(b, &o);
        return json(conn, 200, &o);
    }
    if (get && !strcmp(url, "/brebus/projects")) {
        brebus_projects(b, &o);
        return json(conn, 200, &o);
    }
    if (post && !strcmp(url, "/brebus/workspace/free")) {
        char why[600];
        int rc = brebus_ws_free(b, why, sizeof why);
        if (rc) {
            janas_buf_puts(&o, "{\"error\": ");
            janas_json_write_str(&o, why, strlen(why));
            janas_buf_puts(&o, "}");
            return json(conn, 400, &o);
        }
        brebus_ws_status(b, &o);
        return json(conn, 200, &o);
    }
    if (post && (!strcmp(url, "/brebus/workspace/open") ||
                 !strcmp(url, "/brebus/workspace/new"))) {
        char why[600], err[160];
        int rc;
        if (url[18] == 'n')
            rc = brebus_ws_new(b, why, sizeof why);
        else {
            struct janas_json_doc *d =
                r->body.n
                    ? janas_json_parse(r->body.p, r->body.n, err, sizeof err)
                    : NULL;
            const struct janas_json *v = d ? janas_json_root(d) : NULL;
            const char *path =
                v ? janas_json_str(janas_json_get(v, "path")) : NULL;
            const struct janas_json *c = v ? janas_json_get(v, "create") : NULL;
            rc = brebus_ws_open(b, path, c && c->type == JANAS_JSON_TRUE, why,
                                sizeof why);
            janas_json_free(d);
        }
        if (rc != 0) {
            janas_buf_puts(&o, "{\"error\": ");
            janas_json_write_str(&o, why, strlen(why));
            janas_buf_puts(&o, "}");
            return json(conn, 400, &o);
        }
        brebus_ws_status(b, &o);
        return json(conn, 200, &o);
    }
    if (post && !strcmp(url, "/brebus/open")) {
        char err[160];
        struct janas_json_doc *d =
            r->body.n ? janas_json_parse(r->body.p, r->body.n, err, sizeof err)
                      : NULL;
        const char *path =
            d ? janas_json_str(janas_json_get(janas_json_root(d), "path"))
              : NULL;
        int rc = brebus_open_path(b, path);
        janas_json_free(d);
        janas_buf_puts(&o, rc ? "{\"error\": \"not a file of the workspace\"}"
                              : "{\"ok\": true}");
        return json(conn, rc ? 400 : 200, &o);
    }
    if (post && !strcmp(url, "/brebus/artifact-store")) {
        int rc = brebus_artifact_store(b, r->body.p, r->body.n, &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (post && !strcmp(url, "/brebus/artifacts/rename")) {
        int rc = brebus_artifact_rename(b, r->body.p, r->body.n, &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (post && !strcmp(url, "/brebus/artifacts")) {
        int rc = brebus_artifact_save(b, r->body.p, r->body.n, &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (post && !strcmp(url, "/brebus/export")) {
        int rc = brebus_export(b, r->body.p, r->body.n, &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (get && !strcmp(url, "/brebus/chats")) {
        brebus_chats_list(b, &o);
        return json(conn, 200, &o);
    }
    if (!strcmp(url, "/brebus/settings") && (get || post)) {
        if (get) {
            brebus_settings_get(b, &o);
            return json(conn, 200, &o);
        }
        if (brebus_settings_put(b, r->body.p, r->body.n) == 0) {
            janas_buf_puts(&o, "{\"ok\": true}");
            return json(conn, 200, &o);
        }
        static const char no[] = "{\"error\": \"not a JSON object\"}";
        return reply(conn, 400, "application/json", no, sizeof no - 1, 1);
    }
    if (post && !strcmp(url, "/brebus/models/remove")) {
        char err[160], why[300] = "";
        struct janas_json_doc *d =
            r->body.n ? janas_json_parse(r->body.p, r->body.n, err, sizeof err)
                      : NULL;
        const char *name =
            d ? janas_json_str(janas_json_get(janas_json_root(d), "name"))
              : NULL;
        int rc = brebus_model_remove(b, name, why, sizeof why);
        janas_json_free(d);
        if (rc) {
            janas_buf_puts(&o, "{\"error\": ");
            janas_json_write_str(&o, why, strlen(why));
            janas_buf_puts(&o, "}");
        } else
            janas_buf_puts(&o, "{\"ok\": true}");
        return json(conn, rc ? 400 : 200, &o);
    }
    if (post && !strcmp(url, "/brebus/get/start")) {
        int rc = brebus_get_start(b, r->body.p, r->body.n, &o);
        return json(conn, rc ? 400 : 200, &o);
    }
    if (post && !strcmp(url, "/brebus/get/stop")) {
        char err[160];
        struct janas_json_doc *d =
            r->body.n ? janas_json_parse(r->body.p, r->body.n, err, sizeof err)
                      : NULL;
        brebus_get_stop(
            b, d ? janas_json_str(janas_json_get(janas_json_root(d), "model"))
                 : NULL);
        janas_json_free(d);
        brebus_get_status(b, &o);
        return json(conn, 200, &o);
    }
    if (get && !strcmp(url, "/brebus/get")) {
        brebus_get_status(b, &o);
        return json(conn, 200, &o);
    }
    if (get && !strcmp(url, "/brebus/server")) {
        brebus_server_status(b, &o);
        return json(conn, 200, &o);
    }
    if (get && !strncmp(url, "/brebus/", 8)) {
        if (brebus_api(b, url + 8, &o) == 0)
            return json(conn, 200, &o);
        janas_buf_free(&o);
        static const char no[] = "{\"error\": \"no such question\"}";
        return reply(conn, 404, "application/json", no, sizeof no - 1, 1);
    }
    static const char no[] = "not here";
    return reply(conn, 404, "text/plain", no, sizeof no - 1, 1);
}

int brebus_http_start(struct brebus *b, char *why, size_t wn)
{
    struct sockaddr_in a = {.sin_family = AF_INET,
                            .sin_port = htons(b->port),
                            .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    b->mhd =
        MHD_start_daemon(MHD_USE_INTERNAL_POLLING_THREAD |
                             MHD_USE_THREAD_PER_CONNECTION | MHD_USE_ERROR_LOG,
                         b->port, NULL, NULL, handler, b, MHD_OPTION_SOCK_ADDR,
                         &a, MHD_OPTION_NOTIFY_COMPLETED, completed, NULL,
                         MHD_OPTION_CONNECTION_LIMIT, 64u,
                         MHD_OPTION_CONNECTION_TIMEOUT, 600u, MHD_OPTION_END);
    if (!b->mhd) {
        snprintf(why, wn, "127.0.0.1:%u is taken, or cannot be listened on",
                 b->port);
        return -1;
    }
    return 0;
}

void brebus_http_stop(struct brebus *b)
{
    if (b->mhd)
        MHD_stop_daemon(b->mhd);
    b->mhd = NULL;
}
