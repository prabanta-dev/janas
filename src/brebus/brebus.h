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

#include "common/proc.h"

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

/* proxy.c: a request to the model's API passed on: 0 with the response to
   queue and its status, -1 when no server runs or it cannot be reached */
struct MHD_Connection;
struct MHD_Response;
int brebus_proxy(struct brebus *b, struct MHD_Connection *conn,
                 const char *method, const char *url, const char *body,
                 size_t n, struct MHD_Response **resp, unsigned *status);

#endif
