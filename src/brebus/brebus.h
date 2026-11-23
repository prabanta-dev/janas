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
#include <stdint.h>

#include "common/paths.h"
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
    double need_gb; /* what it will write, as known before it starts */
};

/* downloads going on at once */
#define BREBUS_GETS 4

struct janas_mcp;

#define BREBUS_LOADED 4    /* models loaded at once, at most */
#define BREBUS_USERS 64    /* accounts of Brebus served (users.c) */
#define BREBUS_RECOVERY 10 /* recovery codes of an account */

/* an account (users.c): its password as PBKDF2-HMAC-SHA256, its second
   factor's secret, its recovery codes as their SHA-256 */
struct brebus_user {
    char name[64];
    int admin;
    uint8_t salt[16], hash[32];
    uint32_t iters;
    char totp[40]; /* base32 */
    int totp_on;
    uint8_t rec[BREBUS_RECOVERY][32];
    int n_rec;
};

struct brebus {
    unsigned short port; /* where it listens, on 127.0.0.1 */
    /* served beyond its own window (--serve): where it listens, whether
       over HTTPS, whether a login is asked (always when served) and a
       second factor (when it listens beyond this computer) */
    int serve, tls, need_2fa;
    char listen[64];
    /* served: more names it is reached by (--host-name, comma separated),
       beside its addresses and this machine's own name */
    char names[512];
    /* a user's own Brebus (brebus --child, children.c): the token every
       request must carry, the front Brebus's port (and whether over
       HTTPS) for the models, the folder its projects must be in */
    int child, front_tls;
    /* served to the users of the system (--users system): their
       passwords and groups checked by janas-brebus-helper, which starts
       each one's Brebus as them */
    int sys_users;
    unsigned short front_port;
    char child_token[65];
    char root[4096];
    char cert[4300]; /* the certificate served (a child trusts it) */
    struct brebus_user users[BREBUS_USERS];
    int n_users;
    void *mhd; /* the HTTP daemon */
    char models_dir[4096];
    char bin_dir[4096];  /* where brebus is: janas-server beside it */
    char data_dir[4096]; /* ~/.local/share/janas/brebus */
    char ws[4096];       /* the workspace (workspace.c), "" until chosen;
                            under mu */
    char suffix[16];     /* a sanitizer's build: brebus-asan starts
                            janas-server-asan */
    /* the models' servers (server.c): a janas-server of its own for each
       model loaded, up to BREBUS_LOADED at once */
    pthread_mutex_t mu;
    struct brebus_srv {
        struct janas_proc proc;
        int on;
        int ready;           /* it answered: loaded, not only started */
        unsigned short port; /* Brebus's port plus one plus its place */
        char model[256];
        time_t used; /* when a request last went to it */
    } srv[BREBUS_LOADED];
    int srv_cur; /* the place used last, -1 with none */
    char srv_err[512];
    char srv_last[256]; /* the model unloaded after a quiet while: loaded
                           again by the page at the next message */
    char srv_key[256];  /* the key its clients send, or "" */
    int srv_lan;        /* reachable from the local network */
    int srv_tls;        /* it speaks HTTPS, with Brebus's certificate */
    char srv_fp[100];   /* that certificate's SHA-256, "AB:CD:..." */
    struct brebus_get get[BREBUS_GETS];
    /* tools.c: the chat's tools, MCP servers of the workspace's own */
    pthread_mutex_t tools_mu;
    struct janas_mcp *docs, *files;
    char embedding[128]; /* janas-docs's embedding model, "": words only */
};

/* users.c: the accounts. brebus_random: n random bytes, 0, or -1.
   brebus_users_load: users.json read. brebus_user_add: an account made, or
   its password set anew (its second factor kept): 0, or -1 with the
   reason. brebus_user_check: 1 when name and password are right, the
   account into out. brebus_user_recovery_use: 1 when code is one of the
   account's recovery codes, spent now. brebus_user_totp_on: the second
   factor kept (secret, base32) and ten recovery codes made into codes.
   brebus_totp_secret: a new one; brebus_totp_check: 1 when code is the
   secret's at now, or a step either side (then the step it is of, which
   is never 0); brebus_totp_code: the code at
   now. */
int brebus_random(void *p, size_t n);
int brebus_users_load(struct brebus *b);
int brebus_users_count(struct brebus *b);
int brebus_user_add(struct brebus *b, const char *name, const char *pw,
                    int admin, char *why, size_t wn);
int brebus_user_check(struct brebus *b, const char *name, const char *pw,
                      struct brebus_user *out);
/* a user of the system, their password and groups checked by the helper:
   1 with their account (made for their second factor, no password of
   its own) into out, 0 */
int brebus_user_system(struct brebus *b, const char *name, const char *pw,
                       struct brebus_user *out);
int brebus_user_recovery_use(struct brebus *b, const char *name,
                             const char *code);
int brebus_user_totp_on(struct brebus *b, const char *name, const char *secret,
                        char codes[BREBUS_RECOVERY][12]);
/* a user of the system's entry in users.json, made when there is none
   (their role set at their login): 0, or -1 with no room */
int brebus_user_keep(struct brebus *b, const char *name);
/* an account's role now, not at its login: 1 an administrator, 0 a user,
   -1 none any more (removed; out of the system's groups) */
int brebus_user_role(struct brebus *b, const char *name);
int brebus_totp_secret(char out[33]);
int brebus_totp_check(const char *secret, const char *code, time_t now);
int brebus_totp_code(const char *secret, time_t now, char out[7]);

/* auth.c: the login of Brebus served. brebus_auth_user: 1 when the
   request carries a live session, its user and role filled. login and
   setup: the answer into out and the cookie to set into set_cookie (""
   for none), their HTTP status returned. */
struct MHD_Connection;
void brebus_client_ip(struct MHD_Connection *conn, char *out, size_t n);
int brebus_auth_user(struct MHD_Connection *conn, char *user, size_t un,
                     int *admin);
void brebus_auth_cookie(struct brebus *b, const char *tok, char *out, size_t n);
int brebus_auth_login(struct brebus *b, struct MHD_Connection *conn,
                      const char *body, size_t n, struct janas_buf *out,
                      char *set_cookie, size_t cn);
int brebus_auth_setup(struct brebus *b, struct MHD_Connection *conn,
                      const char *body, size_t n, struct janas_buf *out,
                      char *set_cookie, size_t cn);
void brebus_auth_logout(struct brebus *b, struct MHD_Connection *conn);
/* a password changed by its user ({old, new}), their other sessions
   ended; sessions ended ({user}: an administrator's for another, else
   one's own but this one); the last logins (an administrator: everyone's).
   The HTTP status returned, the answer in out. */
int brebus_auth_password(struct brebus *b, struct MHD_Connection *conn,
                         const char *user, const char *body, size_t n,
                         struct janas_buf *out);
int brebus_auth_end_all(struct brebus *b, struct MHD_Connection *conn,
                        const char *user, int admin, const char *body, size_t n,
                        struct janas_buf *out);
void brebus_auth_logins(struct brebus *b, const char *user, int admin,
                        struct janas_buf *out);

/* children.c: a user's own Brebus, started when there is none: its port
   and token (0), or -1 with the reason; whether a token is one of the
   children's; all of them stopped; whether a path is one the user of this
   Brebus may use (always, but in a user's own Brebus: their folder and
   their data) */
int brebus_child_for(struct brebus *b, const char *user, unsigned short *port,
                     char tok[65], char *why, size_t wn);
int brebus_child_token_ok(const char *tok);
/* the port of 127.0.0.1 listened on by this process's user alone (Linux;
   elsewhere 1): what Brebus sends its keys and tokens to */
int brebus_port_mine(unsigned short port);
/* a user of the system's login ticket (from janas-brebus-helper auth),
   kept for the spawn of their Brebus */
void brebus_child_ticket(const char *user, const char *t);
void brebus_children_stop(void);
int brebus_confined(const struct brebus *b, const char *path);
/* the folders (and files too, with files) of a folder of this machine, as
   JSON into out - for picking one from a browser, where there is no
   window's dialog; a user's Brebus shows only its user's */
void brebus_dirs(const struct brebus *b, const char *path, int files,
                 struct janas_buf *out);
/* a file sent by the page (its name, its bytes) kept in the workspace's
   sources/ and attached as /brebus/attach does: 0, or -1 */
int brebus_upload(struct brebus *b, const char *name, const char *data,
                  size_t n, struct janas_buf *out);

/* http.c: the daemon on 127.0.0.1:port; 0, or -1 with the reason */
int brebus_http_start(struct brebus *b, char *why, size_t wn);
void brebus_http_stop(struct brebus *b);

/* api.c: the answers of GET /brebus/<what>, JSON into out: 0, or -1 for a
   question there is not */
int brebus_api(struct brebus *b, const char *what, struct janas_buf *out);
/* whether the model name of dir sees pictures: a .vision.jns beside it */
int brebus_model_sees(const char *dir, const char *name);

/* server.c: a model's server started (model: a name of My models, with
   options as JSON, may be NULL; a model loaded already is only made the
   one used now), stopped (the one the body names, or all), and how they
   stand, as JSON into out; the starts and stops 0, or -1 with the reason
   in out */
int brebus_server_start(struct brebus *b, const char *body, size_t n,
                        struct janas_buf *out);
int brebus_server_stop(struct brebus *b, const char *body, size_t n,
                       struct janas_buf *out);
void brebus_server_status(struct brebus *b, struct janas_buf *out);
/* the port of the server of the model named (else of the one used last),
   0 when none runs; its scheme (tls) and key with it; that server noted
   as used now */
unsigned short brebus_server_route(struct brebus *b, const char *model,
                                   int *tls, char *key, size_t kn);
/* Brebus's own certificate and key (made when there is none, or it is old
   or for other names), their paths into cert and key: 0, or -1 */
int brebus_certificate(struct brebus *b, char *cert, char *key, size_t n,
                       char *err, size_t en);
/* whether the model named is loaded: 1, or 0 */
int brebus_server_loaded(struct brebus *b, const char *model);
/* a model loaded but not the one in use stopped: 0 (or not loaded), -1
   when it is the one in use */
int brebus_server_unload_idle(struct brebus *b, const char *model);
/* the models' servers of an earlier Brebus still there (servers.json),
   this Brebus's own and answering: taken up again, their work stopped,
   instead of started anew; others found deaf are ended */
void brebus_server_adopt(struct brebus *b);
/* every model's server's replies under way stopped, the servers kept */
void brebus_server_cancel_all(struct brebus *b);

/* get.c: a model of the catalog downloaded (body: {"model": name,
   "bytes": its size when the page knows it}), several at once; stopped,
   one by its name or all (NULL), its .part kept to go on from; how far
   each is, as JSON into out; the start 0, or -1 with the reason in out */
int brebus_get_start(struct brebus *b, const char *body, size_t n,
                     struct janas_buf *out);
void brebus_get_stop(struct brebus *b, const char *model);
void brebus_get_status(struct brebus *b, struct janas_buf *out);

/* store.c: the conversations (a JSON object each, by an id of letters,
   digits, - and _), listed newest first as JSON into out, one read, saved,
   deleted; and the models' settings, one JSON object. 0, or -1 */
void brebus_chats_list(struct brebus *b, struct janas_buf *out);
/* those of the workspace ws, as {"chats": [...]} (each with "archived") */
void brebus_chats_list_in(const char *ws, struct janas_buf *out);
/* a chat moved from one workspace's chats to another's: 0, or -1 (one of
   that id there already) */
int brebus_chat_move(const char *id, const char *from, const char *to);
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
/* store_log.c: an artifact's saved data, in its folder dir, with their
   history. A change written to the log, at the chat's point at (before:
   the state before it, the log's base when there is no log yet; value NULL:
   key deleted): */
int brebus_store_log(const char *dir, const char *chat, const char *before,
                     size_t before_n, long at, const char *key,
                     const struct janas_json *value);
/* from's data as they were at point upto, given to the chat to: its log up
   to there and that state (a branch; to == from: a rewind). 0, or -1 */
int brebus_store_upto(const char *dir, const char *from, const char *to,
                      long upto);
/* a branched chat's artifacts keep the data its source's had at the point
   it was branched from: {"from", "to", "titles", "upto"} (store.c) */
int brebus_artifact_store_copy(struct brebus *b, const char *body, size_t n,
                               struct janas_buf *out);
/* a chat gone back to point upto: its artifacts' data as they were then,
   {"chat", "upto"} */
int brebus_artifact_store_rewind(struct brebus *b, const char *body, size_t n,
                                 struct janas_buf *out);
int brebus_artifact_store(struct brebus *b, const char *body, size_t n,
                          struct janas_buf *out);
/* lib.c: a file of the libraries artifacts may use, by its path under
   /brebus/lib/ (fetched and checked the first time): 0, or -1 with err */
int brebus_lib(struct brebus *b, const char *rel, struct janas_buf *out,
               const char **mime, char *err, size_t en);
/* an artifact's folder renamed with it ({"from", "to"}); never over one */
int brebus_artifact_rename(struct brebus *b, const char *body, size_t n,
                           struct janas_buf *out);
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
/* What the window does when asked to close: steps(say) runs on a thread
   of its own while the window stays open, say(text) showing each step on
   the page (window.brebusClosing), and the window closes when it returns.
   Set before brebus_window; without it the window closes at once. */
void brebus_window_set_closer(void (*steps)(void (*say)(const char *)));
/* The system's dialog to choose a folder (file 0) or a file (1), in the
   window: 0 with the path chosen in out ("" when cancelled), -1 when there
   is no window (the browser: the page asks for a path). */
int brebus_pick(int file, const char *title, char *out, size_t n);
/* 1 while Brebus shows its page in its own window */
int brebus_window_on(void);
/* the window's page drawn at zoom z (0.5-3, 1 its size): -1 with no window */
int brebus_window_zoom(double z);
/* window_place.c: the window's place and size (its normal ones, and
   whether it was maximized), kept in data_dir/window.json; read: 0, or -1
   when none is kept (the first start: maximized) */
struct brebus_place {
    int x, y, w, h, max;
};
void brebus_place_dir(const char *data_dir);
int brebus_place_read(struct brebus_place *p);
void brebus_place_write(const struct brebus_place *p);

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
/* The free chats - those of no project - kept in a workspace of Brebus's
   own, in its data folder; opened when no project is. */
void brebus_free_path(const struct brebus *b, char *out, size_t n);
int brebus_ws_free(struct brebus *b, char *why, size_t wn);
/* path, when it is the free chats' place or a recent project, into real:
   0, or -1 (no other folder is ever a chat's) */
int brebus_ws_known(struct brebus *b, const char *path, char *real, size_t n);
/* a chat of another project than the one open (ws), read, saved or
   deleted, as brebus_chat_get, _put and _delete do for the open one */
int brebus_chat_get_in(const char *ws, const char *id, struct janas_buf *out);
int brebus_chat_put_in(const char *ws, const char *id, const char *body,
                       size_t n);
int brebus_chat_delete_in(const char *ws, const char *id);
/* the free chats and every recent project with its chats, as JSON */
void brebus_projects(struct brebus *b, struct janas_buf *out);
/* {"id", "to": a project's path or "" for the free chats}: the chat of
   the workspace open moved there; 0, or -1 with why */
int brebus_chat_move_to(struct brebus *b, const char *body, size_t n, char *why,
                        size_t wn);
/* the conversations of every recent workspace whose title or messages hold
   query (any case; "" all), newest first, at most 50, as JSON into out */
void brebus_chats_search(struct brebus *b, const char *query,
                         struct janas_buf *out);
/* the same with filters: in "titles", "mine" (the user's messages) or
   "all"; where: one project's path, "free" (the free chats) or "" (all);
   since: newer than this time (ms, 0: any); model: "" any; archived too
   or not */
struct brebus_find {
    const char *in, *where, *model;
    double since;
    int archived;
};
void brebus_chats_find(struct brebus *b, const char *query,
                       const struct brebus_find *f, struct janas_buf *out);

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
/* Skills, in the format of Claude's (skills.c): the list for the page; the
   tool janas__skill and the catalog's lines for the model; its call; one
   added from a folder ({"path"}), one of every chat's removed ({"name"}) */
void brebus_skills_get(struct brebus *b, struct janas_buf *out);
/* Desktop extensions, MCP Bundles (.mcpb), connectors.c: the installed
   ones; one looked at ({"path"}: its name, settings, whether it runs
   here); installed ({"path", "config"}); removed ({"name"}) */
void brebus_connectors_get(struct brebus *b, struct janas_buf *out);
int brebus_connector_inspect(struct brebus *b, const char *body, size_t n,
                             struct janas_buf *out);
int brebus_connector_install(struct brebus *b, const char *body, size_t n,
                             struct janas_buf *out);
int brebus_connector_remove(struct brebus *b, const char *body, size_t n,
                            struct janas_buf *out);
/* The remote connectors a click adds ({"remote": [...]}, each said added
   when mcp.json has it); one added ({"id"}); one of them taken out */
void brebus_connectors_catalog(struct brebus *b, struct janas_buf *out);
/* what a connector is about: one of the catalog by its url ("Notion:
   your pages and databases"), an extension installed by its name (its
   manifest's description); NULL for another */
const char *brebus_connector_about(const char *name, const char *url);
int brebus_connector_add_remote(struct brebus *b, const char *body, size_t n,
                                struct janas_buf *out);
int brebus_connector_drop(struct brebus *b, const char *body, size_t n,
                          struct janas_buf *out);
void brebus_skills_tool(struct brebus *b, struct janas_buf *tool,
                        struct janas_buf *catalog);
int brebus_skills_call(struct brebus *b, const char *args, size_t n,
                       struct janas_buf *out);
int brebus_skills_add(struct brebus *b, const char *body, size_t n,
                      struct janas_buf *out);
int brebus_skills_remove(struct brebus *b, const char *body, size_t n,
                         struct janas_buf *out);
/* A server of the user's over HTTP signed in to (common/mcp_oauth.h): the
   address where the user signs in ({"name", "redirect"} → {"url"}), the
   page the browser comes back to, the tokens forgotten ({"name"}) */
/* What the engine of the model loaded chose for this computer:
   {"choices", "description"}, empty when none is loaded */
void brebus_server_choices(struct brebus *b, struct janas_buf *out);
int brebus_oauth_begin(struct brebus *b, const char *body, size_t n,
                       struct janas_buf *out);
void brebus_oauth_callback(struct brebus *b, const char *state,
                           const char *code, const char *error,
                           struct janas_buf *html);
int brebus_oauth_forget(struct brebus *b, const char *body, size_t n,
                        struct janas_buf *out);
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
/* a GGUF of Hugging Face looked at before its download: {"state":
   "likely" (an architecture this engine runs), "unsupported", "unknown"
   (no header read), "closed" (Hugging Face would not give it: gated, gone)
   , "arch", "why"} */
int brebus_hf_check(struct brebus *b, const char *repo, const char *file,
                    struct janas_buf *out);

/* server.c: the model unloaded after ttl minutes without a request through
   Brebus (settings.json's _server.ttl; 0: never), checked by a thread of
   its own; the folder of the models changed (settings.json's _models_dir,
   a folder there is): 0, or -1 */
void brebus_idle_start(struct brebus *b);
int brebus_models_dir_set(struct brebus *b, const char *path, int save);
/* a model of the models' folder deleted, its companions (MTP head, vision
   encoder) with it when no other model needs them; never the one loaded */
int brebus_model_remove(struct brebus *b, const char *name, char *why,
                        size_t wn);
/* desktop.c: Brebus started at login, its server without a window
   (~/.config/autostart/brebus.desktop), on or off; whether it is */
int brebus_autostart(int on);
int brebus_autostart_on(void);

/* The programs Brebus starts (janas-server, janas-get, a document's
   extraction) have no console of their own on Windows: Brebus is a
   window's program, and each would get a new console, a window of its own
   - and janas-server's output did not reach server.log (Windows 11, 16 Nov
   2026). Nothing elsewhere. */
#ifdef _WIN32
#define BREBUS_NO_CONSOLE JANAS_PROC_DETACHED
#else
#define BREBUS_NO_CONSOLE 0
#endif

/* tools.c: where janas-<name> is, into out: in services/ beside brebus (the
   binary package), else beside it */
void brebus_service_exe(const struct brebus *b, const char *name, char *out,
                        size_t n);

/* pyinstall.c: Python for the cage on Windows, installed when the user
   asks (python.org's embeddable package and PyPI's wheels, each checked
   by its SHA-256): whether it is there; the install's state as JSON
   ({"needed": false} on Linux); the install started in a thread of its
   own, 0 or -1; the folder removed, 0 or -1 */
int brebus_python_installed(void);
void brebus_python_status(struct janas_buf *o);
int brebus_python_install(void);
int brebus_python_remove(void);

/* proxy.c: a request to the model's API passed on: 0 with the response to
   queue and its status, -1 when no server runs or it cannot be reached */
struct MHD_Connection;
struct MHD_Response;
int brebus_proxy_to(struct brebus *b, struct MHD_Connection *conn,
                    const char *method, const char *url, const char *body,
                    size_t n, unsigned short port, int tls, const char *key,
                    const char *extra, struct MHD_Response **resp,
                    unsigned *status);
int brebus_proxy(struct brebus *b, struct MHD_Connection *conn,
                 const char *method, const char *url, const char *body,
                 size_t n, struct MHD_Response **resp, unsigned *status);

#endif
