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
#include <time.h>
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

struct janas_mcp;

struct brebus {
    unsigned short port; /* where it listens, on 127.0.0.1 */
    void *mhd;           /* the HTTP daemon */
    char models_dir[4096];
    char bin_dir[4096];  /* where brebus is: janas-server beside it */
    char data_dir[4096]; /* ~/.local/share/janas/brebus */
    char ws[4096];       /* the workspace (workspace.c), "" until chosen;
                            under mu */
    char suffix[16];     /* a sanitizer's build: brebus-asan starts
                            janas-server-asan */
    /* the model's server (server.c): a janas-server of its own */
    pthread_mutex_t mu;
    struct janas_proc srv;
    int srv_on;
    unsigned short srv_port;
    char srv_model[256];
    char srv_err[512];
    char srv_last[256]; /* the model unloaded after a quiet while: loaded
                           again by the page at the next message */
    time_t srv_used;    /* when the page last asked the model */
    char srv_key[256];  /* the key its clients send, or "" */
    int srv_lan;        /* reachable from the local network */
    struct brebus_get get;
    /* tools.c: the chat's tools, MCP servers of the workspace's own */
    pthread_mutex_t tools_mu;
    struct janas_mcp *docs, *files;
    char embedding[128]; /* janas-docs's embedding model, "": words only */
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
/* an artifact ({"title", "type": html, svg, markdown or code, "lang",
   "content"}) saved in the workspace's artifact/<title>/v<N>.<ext>, the
   next version (the same content as the last one not saved again): its
   {"path", "version"} into out, 0; or -1 with the reason */
/* an artifact's window.storage (store.c): {"title", "act", "key",
   "value", "prefix"} */
int brebus_artifact_store(struct brebus *b, const char *body, size_t n,
                          struct janas_buf *out);
/* lib.c: a file of the libraries artifacts may use, by its path under
   /brebus/lib/ (fetched and checked the first time): 0, or -1 with err */
int brebus_lib(struct brebus *b, const char *rel, struct janas_buf *out,
               const char **mime, char *err, size_t en);
int brebus_artifact_save(struct brebus *b, const char *body, size_t n,
                         struct janas_buf *out);
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
/* The system's dialog to choose a folder (file 0) or a file (1), in the
   window: 0 with the path chosen in out ("" when cancelled), -1 when there
   is no window (the browser: the page asks for a path). */
int brebus_pick(int file, const char *title, char *out, size_t n);
/* 1 while Brebus shows its page in its own window */
int brebus_window_on(void);

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

/* workspace.c: the folder Brebus works in. Its path into out: 0, or -1
   while none is chosen; one opened (create: made first), a new one in
   <Documents>/Brebus: 0, or -1 with the reason; how it stands, with the
   recent ones, as JSON into out */
int brebus_ws_path(struct brebus *b, char *out, size_t n);
int brebus_ws_open(struct brebus *b, const char *path, int create, char *why,
                   size_t wn);
int brebus_ws_new(struct brebus *b, char *why, size_t wn);
void brebus_ws_status(struct brebus *b, struct janas_buf *out);
/* the conversations of every recent workspace whose title or messages hold
   query (any case; "" all), newest first, at most 50, as JSON into out */
void brebus_chats_search(struct brebus *b, const char *query,
                         struct janas_buf *out);

/* tools.c: janas-docs and janas-files for the workspace open, started
   (again: its folders changed) and stopped; their tools as OpenAI's
   functions, with those that only read, as JSON into out; a call
   ({"name", "arguments"}), its result into out: 0, or -1 with the error;
   the folders added to the workspace to be read, and set */
int brebus_tools_start(struct brebus *b);
void brebus_tools_stop(struct brebus *b);
void brebus_tools_list(struct brebus *b, struct janas_buf *out);
int brebus_tools_call(struct brebus *b, const char *body, size_t n,
                      struct janas_buf *out);
/* Janas's services and the user's MCP servers, on or off, opened or not,
   as JSON into out (settings.json's _services.off turns them off; the
   tools list gives the model their catalog and janas__open_service) */
void brebus_services_get(struct brebus *b, struct janas_buf *out);
void brebus_folders_get(struct brebus *b, struct janas_buf *out);
int brebus_folders_set(struct brebus *b, const char *body, size_t n,
                       struct janas_buf *out);

/* attach.c: a document for the chat ({"source": a file's full path or
   a web address, "mode": "copy", "place" or "info"}): copied or downloaded
   into the workspace's sources/, or read where it is; its path, size and
   text as JSON into out ("info": its size alone). 0, or -1 with the
   reason */
int brebus_attach(struct brebus *b, const char *body, size_t n,
                  struct janas_buf *out);
/* a picture's or a recording's type by its name (audio set to 1 for a
   recording), or NULL */
const char *brebus_media_type(const char *path, int *audio);
/* a file of the workspace or of its folders read only, for the page: 0
   with its bytes and type, -1 when it is outside them or not there */
int brebus_file(struct brebus *b, const char *path, struct janas_buf *out,
                const char **mime);
/* a file of the workspace opened by the system's own program (xdg-open):
   0, or -1 when it is not the workspace's */
int brebus_open_path(struct brebus *b, const char *path);

/* python.c: the python tool in the cage, the workspace its /mnt/data:
   whether there can be one (1, or 0 with why); its definition as an
   OpenAI function into o (nothing where there can be none); a piece of
   code run, its result as tools_call's JSON into out; the session ended.
   The run and the stop with tools_mu held. */
int brebus_python_ok(char *why, size_t n);
void brebus_python_tool(struct brebus *b, struct janas_buf *o);
int brebus_python_run(struct brebus *b, const char *code,
                      struct janas_buf *out);
void brebus_python_stop(void);

/* hf.c: Hugging Face's models with GGUF files whose names hold q, the
   most downloaded first; the GGUF and JNS files of one (owner/name), their
   sizes: JSON into out, 0, or -1 with the reason */
int brebus_hf_search(struct brebus *b, const char *q, struct janas_buf *out);
int brebus_hf_files(struct brebus *b, const char *repo, struct janas_buf *out);

/* server.c: the model unloaded after ttl minutes without a request through
   Brebus (settings.json's _server.ttl; 0: never), checked by a thread of
   its own; the folder of the models changed (settings.json's _models_dir,
   a folder there is): 0, or -1 */
void brebus_idle_start(struct brebus *b);
int brebus_models_dir_set(struct brebus *b, const char *path, int save);
/* desktop.c: Brebus started at login, its server without a window
   (~/.config/autostart/brebus.desktop), on or off; whether it is */
int brebus_autostart(int on);
int brebus_autostart_on(void);

/* tools.c: where janas-<name> is, into out: in services/ beside brebus (the
   binary package), else beside it */
void brebus_service_exe(const struct brebus *b, const char *name, char *out,
                        size_t n);

/* proxy.c: a request to the model's API passed on: 0 with the response to
   queue and its status, -1 when no server runs or it cannot be reached */
struct MHD_Connection;
struct MHD_Response;
int brebus_proxy(struct brebus *b, struct MHD_Connection *conn,
                 const char *method, const char *url, const char *body,
                 size_t n, struct MHD_Response **resp, unsigned *status);

#endif
