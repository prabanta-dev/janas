/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * main.c - brebus: serves its page on 127.0.0.1 and shows it in a window
 * of its own (window.c), or in the browser where there cannot be one,
 * until the window is closed or Ctrl+C.
 */
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#else
#include <sys/wait.h>
#include <termios.h>
#endif

#include "brebus.h"
#include "common/catalog.h"
#include "common/mcp_tls.h"
#include "common/proc.h"
#include "common/mhd_tls.h"
#include "common/mcp_http.h"
#include "common/sandbox.h"
#include "common/sysinfo.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

static volatile sig_atomic_t quit;

static void on_signal(int s)
{
    (void)s;
    quit = 1;
}

static void usage(void)
{
    fprintf(stderr,
            "usage: brebus [--port N] [--workspace DIR] [--browser | "
            "--no-open | --background]\n"
            "       brebus --serve [--listen ADDR] [--port N] [--tls-cert "
            "FILE --tls-key FILE]\n"
            "                      [--host-name NAME[,NAME...]]\n"
            "       brebus --serve --users system [...]\n"
            "       brebus --add-user NAME [--user]\n"
            "       brebus --enroll NAME [--users system]\n"
            "       brebus --install-desktop\n"
            "Brebus, Janas's app for the models of this computer: its page\n"
            "on http://127.0.0.1:41414 (--port), in a window of its own, or\n"
            "in the browser with --browser or where there is no WebKitGTK;\n"
            "--no-open: neither. Only this computer reaches it.\n"
            "--background: no window, and the model the chat used last\n"
            "loaded for other programs (how it starts at login).\n"
            "It opens on a new chat of no project; --workspace DIR opens\n"
            "that folder as a project instead.\n"
            "--install-desktop puts Brebus in the desktop's menu, with its\n"
            "icon (~/.local/share/applications).\n"
            "--serve: no window, for the browsers of this computer or of\n"
            "others (--listen 0.0.0.0, or an address of this machine), with\n"
            "a login: the accounts made by --add-user NAME (the password\n"
            "asked in the terminal, 12 characters or more; an account there\n"
            "already gets a new one): administrators, or with --user users,\n"
            "each with a Brebus of their own - chats, projects, settings -\n"
            "the models shared. Beyond this computer it speaks HTTPS,\n"
            "with Brebus's own certificate or --tls-cert/--tls-key, and asks\n"
            "every account for a second factor (an authenticator's code),\n"
            "set up at the machine: at the first login from a browser at\n"
            "127.0.0.1, or by --enroll NAME in its terminal. It answers at\n"
            "its addresses and this machine's name; at other names (a DNS\n"
            "name) only with --host-name.\n"
            "--users system (Linux): the users of this machine instead, by\n"
            "their own password, members of the group brebus (users) or\n"
            "brebus-admin (administrators), each one's Brebus run as them\n"
            "by janas-brebus-helper, installed setuid root.\n");
}

/* ---- the window's closing, in order (window.c's closer): the replies
   stopped, the models' servers, the services; 15 s at most in all, then
   the window closes whatever is left (it ends with Brebus) ---- */
static struct brebus *closing_b;
static pthread_mutex_t close_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t close_cv = PTHREAD_COND_INITIALIZER;
static int close_done;
static void (*close_say)(const char *);

static void *close_work(void *arg)
{
    (void)arg;
    close_say("Stopping the reply…");
    brebus_server_cancel_all(closing_b);
    close_say("Stopping the model…");
    struct janas_buf o = {0};
    brebus_server_stop(closing_b, NULL, 0, &o);
    janas_buf_free(&o);
    close_say("Stopping the services…");
    brebus_children_stop();
    brebus_get_stop(closing_b, NULL);
    brebus_tools_stop(closing_b);
    close_say("Closing…");
    pthread_mutex_lock(&close_mu);
    close_done = 1;
    pthread_cond_signal(&close_cv);
    pthread_mutex_unlock(&close_mu);
    return NULL;
}

static void close_steps(void (*say)(const char *))
{
    close_say = say;
    pthread_t t;
    if (pthread_create(&t, NULL, close_work, NULL) != 0)
        return;
    pthread_detach(t);
    struct timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_sec += 15;
    pthread_mutex_lock(&close_mu);
    while (!close_done)
        if (pthread_cond_timedwait(&close_cv, &close_mu, &until) != 0)
            break; /* past 15 s: the window goes all the same */
    pthread_mutex_unlock(&close_mu);
}

static void *stdin_life(void *arg)
{
    (void)arg;
    char c[256];
    while (read(STDIN_FILENO, c, sizeof c) > 0)
        ;
    quit = 1;
    return NULL;
}

/* a password typed in the terminal, not shown: 0, or -1 */
static int ask_password(const char *prompt, char *out, size_t n)
{
    fputs(prompt, stderr);
    fflush(stderr);
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode = 0;
    int tty = GetConsoleMode(h, &mode) != 0;
    if (tty)
        SetConsoleMode(h, mode & ~(DWORD)ENABLE_ECHO_INPUT);
#else
    struct termios t, quiet;
    int tty = tcgetattr(STDIN_FILENO, &t) == 0;
    if (tty) {
        quiet = t;
        quiet.c_lflag &= (tcflag_t)~ECHO;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &quiet);
    }
#endif
    int ok = fgets(out, (int)n, stdin) != NULL;
#ifdef _WIN32
    if (tty)
        SetConsoleMode(h, mode);
#else
    if (tty)
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &t);
#endif
    fputc('\n', stderr);
    if (!ok)
        return -1;
    out[strcspn(out, "\r\n")] = 0;
    return 0;
}

/* brebus --add-user NAME: an administrator's account, or its password set
   anew */
static int add_account(struct brebus *b, const char *name, int admin)
{
    char pw[256], again[256], why[200];
    if (ask_password("password: ", pw, sizeof pw) != 0 ||
        ask_password("again: ", again, sizeof again) != 0)
        return 1;
    int rc = 1;
    if (strcmp(pw, again))
        fprintf(stderr, "brebus: the two passwords differ\n");
    else if (brebus_user_add(b, name, pw, admin, why, sizeof why) != 0)
        fprintf(stderr, "brebus: %s\n", why);
    else {
        fprintf(stderr, "brebus: %s (%s) can log in to Brebus served\n", name,
                admin ? "an administrator" : "a user");
        rc = 0;
    }
    memset(pw, 0, sizeof pw);
    memset(again, 0, sizeof again);
    return rc;
}

/* brebus --enroll NAME: an account's second factor set up here, at the
   machine (from afar a password alone does not set it up) */
static int enroll(struct brebus *b, const char *name)
{
    char secret[33], code[32] = "", codes[BREBUS_RECOVERY][12];
    if (b->sys_users && brebus_user_keep(b, name) != 0) {
        fprintf(stderr, "brebus: no room for %s\n", name);
        return 1;
    }
    if (brebus_totp_secret(secret) != 0) {
        fprintf(stderr, "brebus: no randomness\n");
        return 1;
    }
    fprintf(stderr,
            "Give this to %s's authenticator app (a QR code of it, or the "
            "secret typed):\n  otpauth://totp/Brebus:%s?secret=%s&issuer="
            "Brebus\n  secret: %s\n",
            name, name, secret, secret);
    for (int tries = 0; tries < 3; tries++) {
        fputs("the code it shows: ", stderr);
        fflush(stderr);
        if (!fgets(code, sizeof code, stdin))
            return 1;
        code[strcspn(code, "\r\n")] = 0;
        if (brebus_totp_check(secret, code, time(NULL)))
            break;
        fprintf(stderr, "brebus: wrong code: check the app's clock\n");
        code[0] = 0;
    }
    if (!code[0])
        return 1;
    if (brebus_user_totp_on(b, name, secret, codes) != 0) {
        fprintf(stderr,
                "brebus: no account %s (brebus --add-user %s first; with "
                "--users system, a user of this machine)\n",
                name, name);
        return 1;
    }
    fprintf(stderr, "Done. Recovery codes, each good once, shown only now:\n");
    for (int k = 0; k < BREBUS_RECOVERY; k++)
        fprintf(stderr, "  %s\n", codes[k]);
    return 0;
}

/* whether an address to listen on is this computer's alone */
static int loopback(const char *a)
{
    return !*a || !strncmp(a, "127.", 4) || !strcmp(a, "::1") ||
           !strcmp(a, "localhost");
}

/* brebus --serve: accounts there are; HTTPS and a second factor beyond
   this computer: 0, or -1 said */
static int serve_setup(struct brebus *b, const char *cert, const char *key)
{
#ifndef __linux__
    if (b->sys_users) {
        fprintf(stderr, "brebus: --users system is for Linux, for now\n");
        return -1;
    }
#endif
    if (b->sys_users) {
        char h[4300];
        snprintf(h, sizeof h, "%s/janas-brebus-helper", b->bin_dir);
        if (access(h, X_OK) != 0) {
            fprintf(stderr,
                    "brebus: --users system needs %s, installed setuid root "
                    "(docs/brebus.md)\n",
                    h);
            return -1;
        }
    } else if (!brebus_users_count(b)) {
        fprintf(stderr,
                "brebus: no account to log in with: brebus --add-user NAME "
                "first\n");
        return -1;
    }
    if (!strcmp(b->listen, "localhost"))
        snprintf(b->listen, sizeof b->listen, "127.0.0.1");
    int wide = !loopback(b->listen);
    b->need_2fa = wide;
    if (!cert != !key) {
        fprintf(stderr, "brebus: --tls-cert and --tls-key go together\n");
        return -1;
    }
    static char c[4300], k[4300];
    char why[300];
    if (cert)
        snprintf(c, sizeof c, "%s", cert), snprintf(k, sizeof k, "%s", key);
    else if (wide &&
             brebus_certificate(b, c, k, sizeof c, why, sizeof why) != 0) {
        fprintf(stderr, "brebus: %s\n", why);
        return -1;
    }
    snprintf(b->cert, sizeof b->cert, "%s", c);
    if (cert || wide) {
        if (janas_mhd_tls_setup(c, k, why, sizeof why) != 0) {
            fprintf(stderr, "brebus: %s\n", why);
            return -1;
        }
        b->tls = 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    janas_sandbox_main(argc, argv); /* a cage for python.c being built */
    struct brebus b = {.port = 41414};
    int open = 1, browser = 0, background = 0;
    const char *workspace = NULL, *add_user = NULL, *cert = NULL, *key = NULL;
    const char *enroll_user = NULL;
    int plain_user = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) {
            int p = atoi(argv[++i]);
            if (p < 1 || p > 65535) {
                usage();
                return 2;
            }
            b.port = (unsigned short)p;
        } else if (!strcmp(argv[i], "--workspace") && i + 1 < argc) {
            workspace = argv[++i];
        } else if (!strcmp(argv[i], "--no-open")) {
            open = 0;
        } else if (!strcmp(argv[i], "--background")) {
            open = 0;
            background = 1;
        } else if (!strcmp(argv[i], "--install-desktop")) {
            return brebus_install_desktop(0);
        } else if (!strcmp(argv[i], "--browser")) {
            browser = 1;
        } else if (!strcmp(argv[i], "--serve")) {
            b.serve = 1;
            open = 0;
        } else if (!strcmp(argv[i], "--listen") && i + 1 < argc &&
                   strlen(argv[i + 1]) < sizeof b.listen) {
            snprintf(b.listen, sizeof b.listen, "%s", argv[++i]);
        } else if (!strcmp(argv[i], "--host-name") && i + 1 < argc &&
                   strlen(argv[i + 1]) < sizeof b.names) {
            snprintf(b.names, sizeof b.names, "%s", argv[++i]);
        } else if (!strcmp(argv[i], "--tls-cert") && i + 1 < argc) {
            cert = argv[++i];
        } else if (!strcmp(argv[i], "--tls-key") && i + 1 < argc) {
            key = argv[++i];
        } else if (!strcmp(argv[i], "--child")) {
            /* a user's own Brebus, started by the front (children.c) */
            const char *t = getenv("BREBUS_CHILD_TOKEN"),
                       *f = getenv("BREBUS_FRONT"), *r = getenv("BREBUS_ROOT"),
                       *ft = getenv("BREBUS_FRONT_TLS");
            /* (an empty root: a user of the system's, their permissions
               keep them) */
            if (!t || strlen(t) != 64 || !f || !r || strlen(r) >= 4096) {
                fprintf(stderr, "brebus: --child is for Brebus served to "
                                "start\n");
                return 2;
            }
            b.child = 1;
            open = 0;
            snprintf(b.child_token, sizeof b.child_token, "%s", t);
            b.front_port = (unsigned short)atoi(f);
            b.front_tls = ft && !strcmp(ft, "1");
            snprintf(b.root, sizeof b.root, "%s", r);
        } else if (!strcmp(argv[i], "--add-user") && i + 1 < argc) {
            add_user = argv[++i];
        } else if (!strcmp(argv[i], "--enroll") && i + 1 < argc) {
            enroll_user = argv[++i];
        } else if (!strcmp(argv[i], "--users") && i + 1 < argc &&
                   (!strcmp(argv[i + 1], "system") ||
                    !strcmp(argv[i + 1], "brebus"))) {
            b.sys_users = !strcmp(argv[++i], "system");
        } else if (!strcmp(argv[i], "--user")) {
            plain_user = 1; /* with --add-user: not an administrator */
        } else {
            usage();
            return !strcmp(argv[i], "-h") || !strcmp(argv[i], "--help") ? 0 : 2;
        }
    }
    if (janas_models_dir(b.models_dir, sizeof b.models_dir) != 0) {
        fprintf(stderr, "brebus: no home to keep the models in\n");
        return 1;
    }
    /* janas-server beside brebus; Brebus's own files in the data folder */
    if (janas_exe_path(b.bin_dir, sizeof b.bin_dir) > 0) {
        char *slash = strrchr(b.bin_dir, '/');
        if (slash) {
            const char *dash = strchr(slash + 1, '-');
            if (dash && strlen(dash) < sizeof b.suffix)
                snprintf(b.suffix, sizeof b.suffix, "%s", dash);
#ifdef _WIN32
            /* "brebus.exe": the programs beside it are "janas-x.exe" */
            if (!dash)
                snprintf(b.suffix, sizeof b.suffix, ".exe");
#endif
            *slash = 0;
        }
    }
    const char *xd = getenv("XDG_DATA_HOME"), *home = getenv("HOME");
    /* (on Windows XDG_DATA_HOME is %LOCALAPPDATA%: src/common/win_compat.c) */
    if (xd && janas_path_abs(xd))
        snprintf(b.data_dir, sizeof b.data_dir, "%s/janas/brebus", xd);
    else
        snprintf(b.data_dir, sizeof b.data_dir, "%s/.local/share/janas/brebus",
                 home ? home : ".");
    janas_path_slashes(b.data_dir);
    janas_mkdirs(b.data_dir);
    brebus_place_dir(b.data_dir); /* the window's place, kept there */
    pthread_mutex_init(&b.mu, NULL);
    brebus_users_load(&b);
    if (add_user)
        return add_account(&b, add_user, !plain_user);
    if (enroll_user)
        return enroll(&b, enroll_user);
    if ((b.listen[0] || cert || key) && !b.serve) {
        fprintf(stderr, "brebus: --listen and --tls-cert/--tls-key go with "
                        "--serve\n");
        return 2;
    }
    if (b.serve && serve_setup(&b, cert, key) != 0)
        return 1;
    /* a child reaches its front over HTTPS: the front's certificate */
    const char *fc = getenv("BREBUS_FRONT_CERT");
    if (b.child && b.front_tls && fc && *fc && janas_tls_trust(fc) != 0) {
        fprintf(stderr, "brebus: the front's certificate %s: not read\n", fc);
        return 1;
    }
    for (int i = 0; i < BREBUS_GETS; i++)
        pthread_mutex_init(&b.get[i].mu, NULL);
    pthread_mutex_init(&b.tools_mu, NULL);
    char why[600];
    /* the workspace: the one given, opened by the page with a new chat;
       else the free chats, as a chat app starts (a terminal's folder is
       not taken for a project: a new chat of no project, as from the
       menu) */
    char cwd[4096];
    if (workspace) {
        char abs[4200];
        if (!janas_path_abs(workspace) && getcwd(cwd, sizeof cwd))
            snprintf(abs, sizeof abs, "%s/%s", cwd, workspace);
        else
            snprintf(abs, sizeof abs, "%s", workspace);
        if (brebus_ws_open(&b, abs, 1, why, sizeof why) != 0) {
            fprintf(stderr, "brebus: %s\n", why);
            return 1;
        }
        b.ws_given = 1;
    }
    /* no project given: the free chats, as a chat app starts */
    if (brebus_ws_path(&b, cwd, sizeof cwd) != 0 &&
        brebus_ws_free(&b, why, sizeof why) != 0)
        fprintf(stderr, "brebus: %s\n", why);
    /* the models' folder chosen in the page, where JANAS_MODELS says none */
    {
        struct janas_buf f = {0};
        char err[160];
        brebus_settings_get(&b, &f);
        struct janas_json_doc *d =
            f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
        const char *md = d ? janas_json_str(janas_json_get(janas_json_root(d),
                                                           "_models_dir"))
                           : NULL;
        const char *env = getenv("JANAS_MODELS");
        if (md && !(env && *env))
            brebus_models_dir_set(&b, md, 0);
        janas_json_free(d);
        janas_buf_free(&f);
    }
    if (brebus_http_start(&b, why, sizeof why) != 0 && b.serve) {
        fprintf(stderr, "brebus: %s\n", why);
        return 1;
    } else if (!b.serve && !b.mhd) {
        /* a Brebus there already (started at login, or from the menu
           before): its window onto it, this one ends with it */
        char url0[64];
        snprintf(url0, sizeof url0, "http://127.0.0.1:%u/", b.port);
        struct janas_url u;
        struct janas_http h;
        char e2[160], probe[96];
        snprintf(probe, sizeof probe, "%sbrebus/workspace", url0);
        int there = janas_url_parse(probe, &u, e2, sizeof e2) == 0 &&
                    janas_http_request(&h, &u, "GET", NULL, 0, NULL, 0, 2000,
                                       NULL, e2, sizeof e2) == 0;
        if (there) {
            there = h.head.status == 200;
            janas_http_close(&h);
        }
        if (there && open) {
            if (browser || brebus_window(url0, &quit, why, sizeof why) != 0)
                janas_open_url(url0);
            return 0;
        }
        fprintf(stderr, "brebus: %s\n", why);
        return 1;
    }
    if (!b.child)
        brebus_server_adopt(&b); /* a server of before, still there */
    /* the entry that starts it at login written as this Brebus starts it
       (an older one said --no-open, which loaded no model) */
    if (!b.child && !b.serve && brebus_autostart_on())
        brebus_autostart(1);
    if (background && !b.child && !b.serve)
        brebus_server_start_last(&b);
    brebus_idle_start(&b);
    char url[128];
    snprintf(url, sizeof url, "%s://%s%s%s:%u/", b.tls ? "https" : "http",
             strchr(b.listen, ':') ? "[" : "",
             b.listen[0] && strcmp(b.listen, "0.0.0.0") &&
                     strcmp(b.listen, "::")
                 ? b.listen
                 : "127.0.0.1",
             strchr(b.listen, ':') ? "]" : "", b.port);
    fprintf(stderr, "brebus %s: %s (Ctrl+C ends it)\n", JANAS_VERSION, url);
    if (b.serve)
        fprintf(stderr, "brebus: served, with a login%s%s; %d account%s\n",
                b.tls ? ", over HTTPS" : "",
                b.need_2fa ? ", a second factor for every account" : "",
                brebus_users_count(&b), brebus_users_count(&b) == 1 ? "" : "s");
    /* a user of the system's own Brebus: it ends when its front closes
       its input (it runs as another user, beyond the front's signals) */
    if (b.child && getenv("BREBUS_STDIN_LIFE")) {
        pthread_t lt;
        if (pthread_create(&lt, NULL, stdin_life, NULL) == 0)
            pthread_detach(lt);
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    closing_b = &b;
    brebus_window_set_closer(close_steps);
    if (open && !browser && brebus_window(url, &quit, why, sizeof why) == 0)
        quit = 1; /* the window closed */
    else if (open) {
        if (!browser)
            fprintf(stderr, "brebus: %s; in the browser instead\n", why);
        janas_open_url(url);
    }
    while (!quit)
        sleep(1);
    brebus_http_stop(&b);
    brebus_children_stop(); /* the users' own Brebus go with it */
    struct janas_buf o = {0};
    brebus_server_stop(&b, NULL, 0, &o); /* the model's server goes with it */
    brebus_get_stop(&b, NULL); /* a download too, to go on next time */
    brebus_tools_stop(&b);
    janas_buf_free(&o);
    return 0;
}
