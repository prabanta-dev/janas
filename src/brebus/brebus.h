/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * brebus.h - Brebus, Janas's app for local models (job/plans/brebus_piano.md
 * in the project's notes): a program that serves its own page on this
 * machine's loopback address and answers the page's questions - the
 * machine, the models installed, the catalog - while the heavy work runs
 * in the programs Janas already has (janas-server, janas-get).
 */
#ifndef JANAS_BREBUS_H
#define JANAS_BREBUS_H

#include <stddef.h>

#include "llm/json.h"

#include <pthread.h>
#include <signal.h>

#include "common/proc.h"

/* get.c: a download of janas-get, and how far it is */
struct brebus_get {
    pthread_mutex_t mu;
    struct janas_proc proc;
    pthread_t th; /* reads its output */
    int joinable, running, stopped, code, fd;
    char model[416];
    char step[256]; /* its last "== ..." line */
    char said[512]; /* its last other line */
    double done_gb, total_gb, pct, mbs;
};

struct brebus {
    unsigned short port; /* where it listens, on 127.0.0.1 */
    void *mhd;           /* the HTTP daemon */
    char models_dir[4096];
    char bin_dir[4096];  /* where brebus is: janas-server beside it */
    char data_dir[4096]; /* ~/.local/share/janas/brebus */
    char suffix[16];     /* a sanitizer's build: brebus-asan starts
                            janas-server-asan */
    /* the model's server (server.c): a janas-server of its own */
    pthread_mutex_t mu;
    struct janas_proc srv;
    int srv_on;
    unsigned short srv_port;
    char srv_model[256];
    char srv_err[512];
    char srv_key[256]; /* the key its clients send, or "" */
    int srv_lan;       /* reachable from the local network */
    struct brebus_get get;
};

/* http.c: the daemon on 127.0.0.1:port; 0, or -1 with the reason */
int brebus_http_start(struct brebus *b, char *why, size_t wn);
void brebus_http_stop(struct brebus *b);

/* api.c: the answers of GET /brebus/<what>, JSON into out: 0, or -1 for a
   question there is not */
int brebus_api(struct brebus *b, const char *what, struct janas_buf *out);

/* server.c: the model's server started (model: a name of My models, with
   options as JSON, may be NULL), stopped, and how it stands, as JSON into
   out; the starts and stops 0, or -1 with the reason in out */
int brebus_server_start(struct brebus *b, const char *body, size_t n,
                        struct janas_buf *out);
int brebus_server_stop(struct brebus *b, struct janas_buf *out);
void brebus_server_status(struct brebus *b, struct janas_buf *out);
/* the port of the model's server, 0 when none runs */
unsigned short brebus_server_port(struct brebus *b);

/* get.c: a model of the catalog downloaded (body: {"model": name}),
   stopped (its .part kept, to go on from), and how far it is, as JSON
   into out; the start 0, or -1 with the reason in out */
int brebus_get_start(struct brebus *b, const char *body, size_t n,
                     struct janas_buf *out);
void brebus_get_stop(struct brebus *b);
void brebus_get_status(struct brebus *b, struct janas_buf *out);

/* store.c: the conversations (a JSON object each, by an id of letters,
   digits, - and _), listed newest first as JSON into out, one read, saved,
   deleted; and the models' settings, one JSON object. 0, or -1 */
void brebus_chats_list(struct brebus *b, struct janas_buf *out);
int brebus_chat_get(struct brebus *b, const char *id, struct janas_buf *out);
int brebus_chat_put(struct brebus *b, const char *id, const char *body,
                    size_t n);
int brebus_chat_delete(struct brebus *b, const char *id);
void brebus_settings_get(struct brebus *b, struct janas_buf *out);
int brebus_settings_put(struct brebus *b, const char *body, size_t n);
/* a text saved in the downloads folder (body: {"name", "text"}), under a
   name of its own, never over a file: {"path"} into out, 0; or -1 with the
   reason */
int brebus_export(struct brebus *b, const char *body, size_t n,
                  struct janas_buf *out);

/* window.c: the page at url in a window of its own, until it is closed
   or *quit is set: 0 then, or -1 at once, with the reason, where there
   can be no window (no WebKitGTK, no display) */
int brebus_window(const char *url, volatile sig_atomic_t *quit, char *why,
                  size_t wn);

/* desktop.c: brebus.desktop and its icons written for the desktop's menu
   (quiet: as the window opens, saying nothing, an entry already there
   kept); the exit status */
int brebus_install_desktop(int quiet);

/* icons.c: the Prabanta logo as a PNG of each size (16, 22, 24, 32, 48,
   64, 128, 256, 512; the list ends with size 0) and as SVG */
struct brebus_icon {
    int size;
    const unsigned char *png;
    unsigned len;
};
extern const struct brebus_icon brebus_icons[];
extern const unsigned char *const brebus_icon_svg;
extern const unsigned brebus_icon_svg_len;
/* the one of that size, or NULL */
const struct brebus_icon *brebus_icon(int size);

/* proxy.c: a request to the model's API passed on: 0 with the response to
   queue and its status, -1 when no server runs or it cannot be reached */
struct MHD_Connection;
struct MHD_Response;
int brebus_proxy(struct brebus *b, struct MHD_Connection *conn,
                 const char *method, const char *url, const char *body,
                 size_t n, struct MHD_Response **resp, unsigned *status);

#endif
