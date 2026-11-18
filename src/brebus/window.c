/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * window.c - Brebus in a window of its own: WebKitGTK (the 4.1 API, on
 * GTK 3), loaded when Brebus starts, shows its page as an application
 * does, with its name and the Prabanta logo (its icons written for the
 * desktop first, desktop.c). Nothing is needed to build
 * it: where the library is not there, or there is no display, brebus
 * opens the page in the browser instead. Links to other sites open in
 * the browser; the window closed, Brebus ends.
 */
#ifndef _WIN32 /* window_win.c there */
#include <dlfcn.h>
#include <pthread.h>
#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "brebus.h"

typedef void *P;
typedef int (*source_fn)(P);

static struct {
    int (*gtk_init_check)(int *, char ***);
    P (*gtk_window_new)(int);
    void (*gtk_window_set_title)(P, const char *);
    void (*gtk_window_set_default_size)(P, int, int);
    void (*gtk_window_set_icon)(P, P);
    void (*gtk_window_set_default_icon_name)(const char *);
    void (*gtk_container_add)(P, P);
    void (*gtk_widget_show_all)(P);
    void (*gtk_main)(void);
    void (*gtk_main_quit)(void);
    unsigned long (*g_signal_connect_data)(P, const char *, P, P, P, int);
    unsigned (*g_timeout_add)(unsigned, source_fn, P);
    void (*g_set_prgname)(const char *);
    void (*g_set_application_name)(const char *);
    P (*gdk_pixbuf_loader_new)(void);
    int (*gdk_pixbuf_loader_write)(P, const unsigned char *, size_t, P *);
    int (*gdk_pixbuf_loader_close)(P, P *);
    P (*gdk_pixbuf_loader_get_pixbuf)(P);
    P (*webkit_web_view_new)(void);
    P (*gtk_file_chooser_native_new)(const char *, P, int, const char *,
                                     const char *);
    int (*gtk_native_dialog_run)(P);
    char *(*gtk_file_chooser_get_filename)(P);
    void (*g_free)(P);
    void (*g_object_unref)(P);
    unsigned (*g_idle_add)(source_fn, P);
    P (*webkit_web_view_new_with_related_view)(P);
    void (*gtk_widget_destroy)(P);
    void (*webkit_web_view_load_uri)(P, const char *);
    P (*webkit_navigation_policy_decision_get_navigation_action)(P);
    P (*webkit_navigation_action_get_request)(P);
    const char *(*webkit_uri_request_get_uri)(P);
    void (*webkit_policy_decision_ignore)(P);
    /* the microphone, for dictation: optional, looked up when there */
    unsigned long (*webkit_user_media_permission_request_get_type)(void);
    int (*g_type_check_instance_is_a)(P, unsigned long);
    int (*webkit_user_media_permission_is_for_audio_device)(P);
    int (*webkit_user_media_permission_is_for_video_device)(P);
    void (*webkit_permission_request_allow)(P);
    void (*webkit_permission_request_deny)(P);
    /* the window's place and size: optional, looked up when there */
    void (*gtk_window_maximize)(P);
    int (*gtk_window_is_maximized)(P);
    void (*gtk_window_get_size)(P, int *, int *);
    void (*gtk_window_get_position)(P, int *, int *);
    void (*gtk_window_move)(P, int, int);
    void (*gtk_window_resize)(P, int, int);
} w;

/* where the window is, followed while it is open (a window being
   destroyed has none to ask), written when it closes */
static struct brebus_place place;
static int placed;

/* a permission the page asks for: the microphone alone, for dictation, and
   only on Brebus's own page; anything else refused */
static int permission(P view, P req, P data)
{
    (void)view, (void)data;
    int mic = w.webkit_user_media_permission_request_get_type &&
              w.g_type_check_instance_is_a &&
              w.webkit_user_media_permission_is_for_audio_device &&
              w.webkit_user_media_permission_is_for_video_device &&
              w.g_type_check_instance_is_a(
                  req, w.webkit_user_media_permission_request_get_type()) &&
              w.webkit_user_media_permission_is_for_audio_device(req) &&
              !w.webkit_user_media_permission_is_for_video_device(req);
    if (mic && w.webkit_permission_request_allow)
        w.webkit_permission_request_allow(req);
    else if (w.webkit_permission_request_deny)
        w.webkit_permission_request_deny(req);
    return 1;
}

static const char *own; /* the page's address, "http://127.0.0.1:port/" */
static volatile sig_atomic_t *stop;

static void open_outside(const char *url)
{
    pid_t pid = fork();
    if (pid == 0) {
        execlp("xdg-open", "xdg-open", url, (char *)NULL);
        _exit(127);
    }
    if (pid > 0)
        waitpid(pid, NULL, 0);
}

/* WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION 0, NEW_WINDOW_ACTION 1:
   what is not Brebus's own page goes to the browser */
static int decide(P view, P decision, int type, P data)
{
    (void)view, (void)data;
    if (type != 0 && type != 1)
        return 0;
    P action =
        w.webkit_navigation_policy_decision_get_navigation_action(decision);
    P req = action ? w.webkit_navigation_action_get_request(action) : NULL;
    const char *uri = req ? w.webkit_uri_request_get_uri(req) : NULL;
    if (!uri || !strncmp(uri, own, strlen(own)))
        return 0;
    if (!strncmp(uri, "https://", 8) || !strncmp(uri, "http://", 7))
        open_outside(uri);
    w.webkit_policy_decision_ignore(decision);
    return 1;
}

static P app_icon, main_win;

/* The system's dialog to choose a folder or a file (GTK's native one: the
   desktop portal's on KDE and GNOME), for the page: asked from a request's
   thread, run on GTK's, the answer waited for. */
static pthread_mutex_t pick_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pick_cv = PTHREAD_COND_INITIALIZER;
static int pick_kind, pick_done, pick_busy, window_on;
static char pick_title[200], pick_path[4096];

static int pick_run(P data)
{
    (void)data;
    /* GTK_FILE_CHOOSER_ACTION_OPEN 0, SELECT_FOLDER 2; ACCEPT -3 */
    P d = w.gtk_file_chooser_native_new(pick_title, main_win, pick_kind ? 0 : 2,
                                        "Choose", "Cancel");
    char path[4096] = "";
    if (d) {
        if (w.gtk_native_dialog_run(d) == -3) {
            char *f = w.gtk_file_chooser_get_filename(d);
            if (f) {
                snprintf(path, sizeof path, "%s", f);
                w.g_free(f);
            }
        }
        w.g_object_unref(d);
    }
    pthread_mutex_lock(&pick_mu);
    snprintf(pick_path, sizeof pick_path, "%s", path);
    pick_done = 1;
    pthread_cond_broadcast(&pick_cv);
    pthread_mutex_unlock(&pick_mu);
    return 0; /* once */
}

int brebus_window_on(void)
{
    pthread_mutex_lock(&pick_mu);
    int on = window_on;
    pthread_mutex_unlock(&pick_mu);
    return on;
}

int brebus_pick(int file, const char *title, char *out, size_t n)
{
    pthread_mutex_lock(&pick_mu);
    if (!window_on) {
        pthread_mutex_unlock(&pick_mu);
        return -1; /* no window: the browser, where the page asks a path */
    }
    while (pick_busy && window_on) /* one dialog at a time */
        pthread_cond_wait(&pick_cv, &pick_mu);
    if (!window_on) {
        pthread_mutex_unlock(&pick_mu);
        return -1;
    }
    pick_busy = 1;
    pick_kind = file ? 1 : 0;
    snprintf(pick_title, sizeof pick_title, "%s", title);
    pick_done = 0;
    w.g_idle_add(pick_run, NULL);
    while (!pick_done && window_on)
        pthread_cond_wait(&pick_cv, &pick_mu);
    snprintf(out, n, "%s", pick_done ? pick_path : "");
    pick_busy = 0;
    pthread_cond_broadcast(&pick_cv);
    pthread_mutex_unlock(&pick_mu);
    return 0;
}

/* a page of Brebus's own opened apart (its engine monitor): a second
   window, closed with its page; another site's goes to the browser */
static void popup_ready(P view, P win)
{
    (void)view;
    w.gtk_widget_show_all(win);
}

static void popup_close(P view, P win)
{
    (void)view;
    w.gtk_widget_destroy(win);
}

static P create(P view, P action, P data)
{
    (void)data;
    P req = w.webkit_navigation_action_get_request(action);
    const char *uri = req ? w.webkit_uri_request_get_uri(req) : NULL;
    if (!uri)
        return NULL;
    if (strncmp(uri, own, strlen(own))) {
        if (!strncmp(uri, "https://", 8) || !strncmp(uri, "http://", 7))
            open_outside(uri);
        return NULL;
    }
    P win = w.gtk_window_new(0);
    w.gtk_window_set_title(win, "Brebus");
    w.gtk_window_set_default_size(win, 1100, 760);
    if (app_icon)
        w.gtk_window_set_icon(win, app_icon);
    P nv = w.webkit_web_view_new_with_related_view(view);
    w.g_signal_connect_data(nv, "decide-policy", (P)decide, NULL, NULL, 0);
    w.g_signal_connect_data(nv, "ready-to-show", (P)popup_ready, win, NULL, 0);
    w.g_signal_connect_data(nv, "close", (P)popup_close, win, NULL, 0);
    w.gtk_container_add(win, nv);
    return nv;
}

static void closed(P widget, P data)
{
    (void)widget, (void)data;
    w.gtk_main_quit();
}

/* Ctrl+C or SIGTERM, seen from GTK's loop */
static int watch(P data)
{
    if (data && w.gtk_window_is_maximized && w.gtk_window_get_size &&
        w.gtk_window_get_position) {
        /* the normal size and place are kept while it is maximized */
        place.max = w.gtk_window_is_maximized(data) != 0;
        if (!place.max) {
            w.gtk_window_get_size(data, &place.w, &place.h);
            w.gtk_window_get_position(data, &place.x, &place.y);
        }
        placed = 1;
    }
    if (*stop)
        w.gtk_main_quit();
    return 1;
}

#define SYM(name)                                                              \
    if (!(*(void **)&w.name = dlsym(lib, #name))) {                            \
        snprintf(why, wn, "%s has no %s", soname, #name);                      \
        return -1;                                                             \
    }

int brebus_window(const char *url, volatile sig_atomic_t *quit, char *why,
                  size_t wn)
{
    static const char *const names[] = {"libwebkit2gtk-4.1.so.0",
                                        "libwebkit2gtk-4.0.so.37"};
    void *lib = NULL;
    const char *soname = NULL;
    for (size_t i = 0; i < sizeof names / sizeof *names && !lib; i++)
        if ((lib = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL)))
            soname = names[i];
    if (!lib) {
        snprintf(why, wn, "no WebKitGTK (libwebkit2gtk-4.1)");
        return -1;
    }
    SYM(gtk_init_check)
    SYM(gtk_window_new)
    SYM(gtk_window_set_title)
    SYM(gtk_window_set_default_size)
    SYM(gtk_window_set_icon)
    SYM(gtk_window_set_default_icon_name)
    SYM(gtk_container_add)
    SYM(gtk_widget_show_all)
    SYM(gtk_main)
    SYM(gtk_main_quit)
    SYM(g_signal_connect_data)
    SYM(g_timeout_add)
    SYM(g_set_prgname)
    SYM(g_set_application_name)
    SYM(gdk_pixbuf_loader_new)
    SYM(gdk_pixbuf_loader_write)
    SYM(gdk_pixbuf_loader_close)
    SYM(gdk_pixbuf_loader_get_pixbuf)
    SYM(webkit_web_view_new)
    SYM(gtk_file_chooser_native_new)
    SYM(gtk_native_dialog_run)
    SYM(gtk_file_chooser_get_filename)
    SYM(g_free)
    SYM(g_object_unref)
    SYM(g_idle_add)
    SYM(webkit_web_view_new_with_related_view)
    SYM(gtk_widget_destroy)
    SYM(webkit_web_view_load_uri)
    SYM(webkit_navigation_policy_decision_get_navigation_action)
    SYM(webkit_navigation_action_get_request)
    SYM(webkit_uri_request_get_uri)
    SYM(webkit_policy_decision_ignore)
    *(void **)&w.webkit_user_media_permission_request_get_type =
        dlsym(lib, "webkit_user_media_permission_request_get_type");
    *(void **)&w.g_type_check_instance_is_a =
        dlsym(lib, "g_type_check_instance_is_a");
    *(void **)&w.webkit_user_media_permission_is_for_audio_device =
        dlsym(lib, "webkit_user_media_permission_is_for_audio_device");
    *(void **)&w.webkit_user_media_permission_is_for_video_device =
        dlsym(lib, "webkit_user_media_permission_is_for_video_device");
    *(void **)&w.webkit_permission_request_allow =
        dlsym(lib, "webkit_permission_request_allow");
    *(void **)&w.webkit_permission_request_deny =
        dlsym(lib, "webkit_permission_request_deny");
    *(void **)&w.gtk_window_maximize = dlsym(lib, "gtk_window_maximize");
    *(void **)&w.gtk_window_is_maximized =
        dlsym(lib, "gtk_window_is_maximized");
    *(void **)&w.gtk_window_get_size = dlsym(lib, "gtk_window_get_size");
    *(void **)&w.gtk_window_get_position =
        dlsym(lib, "gtk_window_get_position");
    *(void **)&w.gtk_window_move = dlsym(lib, "gtk_window_move");
    *(void **)&w.gtk_window_resize = dlsym(lib, "gtk_window_resize");
    /* the name the desktop shows and matches brebus.desktop by, whose icons
       are written first where missing: the task bar's icon is the logo */
    brebus_install_desktop(1);
    w.g_set_prgname("brebus");
    w.g_set_application_name("Brebus");
    if (!w.gtk_init_check(NULL, NULL)) {
        snprintf(why, wn, "no display for a window");
        return -1;
    }
    /* GTK takes the system's locale: the answers' numbers stay JSON's,
       "0.40" and never "0,40" (set before the page asks anything) */
    setlocale(LC_NUMERIC, "C");
    own = url;
    stop = quit;
    P win = w.gtk_window_new(0); /* GTK_WINDOW_TOPLEVEL */
    w.gtk_window_set_title(win, "Brebus");
    w.gtk_window_set_default_size(win, 1280, 820);
    w.gtk_window_set_default_icon_name("brebus");
    const struct brebus_icon *logo = brebus_icon(256);
    P loader = w.gdk_pixbuf_loader_new();
    if (loader &&
        w.gdk_pixbuf_loader_write(loader, logo->png, logo->len, NULL) &&
        w.gdk_pixbuf_loader_close(loader, NULL)) {
        P icon = w.gdk_pixbuf_loader_get_pixbuf(loader);
        if (icon)
            w.gtk_window_set_icon(win, app_icon = icon);
    }
    P view = w.webkit_web_view_new();
    w.g_signal_connect_data(view, "decide-policy", (P)decide, NULL, NULL, 0);
    w.g_signal_connect_data(view, "permission-request", (P)permission, NULL,
                            NULL, 0);
    w.g_signal_connect_data(view, "create", (P)create, NULL, NULL, 0);
    w.g_signal_connect_data(win, "destroy", (P)closed, NULL, NULL, 0);
    w.gtk_container_add(win, view);
    w.webkit_web_view_load_uri(view, url);
    /* where it was when it closed, or maximized the first time (on Wayland
       the compositor places it: only the size and the maximized state
       hold there) */
    struct brebus_place kept;
    int have = brebus_place_read(&kept) == 0;
    if (have && w.gtk_window_resize && w.gtk_window_move) {
        w.gtk_window_resize(win, kept.w, kept.h);
        w.gtk_window_move(win, kept.x, kept.y);
        place = kept;
    }
    if ((!have || kept.max) && w.gtk_window_maximize)
        w.gtk_window_maximize(win);
    w.gtk_widget_show_all(win);
    w.g_timeout_add(200, watch, win);
    main_win = win;
    pthread_mutex_lock(&pick_mu);
    window_on = 1;
    pthread_mutex_unlock(&pick_mu);
    w.gtk_main();
    if (placed)
        brebus_place_write(&place);
    pthread_mutex_lock(&pick_mu);
    window_on = 0; /* a request waiting on a dialog is let go */
    pthread_cond_broadcast(&pick_cv);
    pthread_mutex_unlock(&pick_mu);
    return 0;
}

#endif
