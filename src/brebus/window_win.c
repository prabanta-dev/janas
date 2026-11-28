/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * window_win.c - Brebus in a window of its own on Windows: WebView2, the
 * engine of Edge that Windows 10 and 11 carry, shows its page as an
 * application does, with its name and the Prabanta logo (brebus.exe's
 * icon, brebus.rc). WebView2Loader.dll, beside brebus.exe, is loaded when
 * Brebus starts: where it is not there, or WebView2's runtime is missing,
 * brebus opens the page in the browser instead. Links to other sites open
 * in the browser; a page of Brebus's own asked apart (the engine monitor,
 * a chat's window) opens in a window of WebView2's; the window closed,
 * Brebus ends. The system's dialog chooses folders and files for the page
 * (brebus_pick), as GTK's does on Linux (window.c).
 */
#ifdef _WIN32

#define COBJMACROS
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <shobjidl.h>

#include "WebView2.h" /* src/third_party/webview2 */
#include "brebus.h"
#include "common/sysinfo.h"

typedef HRESULT(STDAPICALLTYPE *create_env_fn)(
    PCWSTR, PCWSTR, ICoreWebView2EnvironmentOptions *,
    ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler *);

static HWND main_win;
static ICoreWebView2Controller *controller;
static ICoreWebView2 *view;
static WCHAR own[80]; /* the page's address, L"http://127.0.0.1:port/" */
static WCHAR *start_url;
static volatile sig_atomic_t *stop;
static int failed; /* WebView2 could not be made: the browser instead */
static char fail_why[200];

static pthread_mutex_t win_mu = PTHREAD_MUTEX_INITIALIZER;
static int window_on;

static int ours(const WCHAR *uri)
{
    return uri && !wcsncmp(uri, own, wcslen(own));
}

/* another site's page: the browser's, as a link clicked (outside.c) */
static void open_outside(const WCHAR *uri)
{
    char u[4096];
    if (uri &&
        WideCharToMultiByte(CP_UTF8, 0, uri, -1, u, sizeof u, NULL, NULL) > 0)
        brebus_open_outside(u);
}

/* ---- the handlers WebView2 calls back: objects of a single instance
   each, so that their count of references is not kept ---- */

#define UNKNOWN(T)                                                             \
    static HRESULT STDMETHODCALLTYPE T##_qi(T *self, REFIID riid, void **out)  \
    {                                                                          \
        (void)riid;                                                            \
        *out = self;                                                           \
        return S_OK;                                                           \
    }                                                                          \
    static ULONG STDMETHODCALLTYPE T##_ref(T *self)                            \
    {                                                                          \
        (void)self;                                                            \
        return 1;                                                              \
    }

UNKNOWN(ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler)
UNKNOWN(ICoreWebView2CreateCoreWebView2ControllerCompletedHandler)
UNKNOWN(ICoreWebView2NavigationStartingEventHandler)
UNKNOWN(ICoreWebView2NewWindowRequestedEventHandler)

static void fail(const char *why)
{
    snprintf(fail_why, sizeof fail_why, "%s", why);
    failed = 1;
    PostMessageW(main_win, WM_CLOSE, 0, 0);
}

/* the page's address only; another site's goes to the browser */
static HRESULT STDMETHODCALLTYPE navigation_starting(
    ICoreWebView2NavigationStartingEventHandler *self, ICoreWebView2 *sender,
    ICoreWebView2NavigationStartingEventArgs *args)
{
    (void)self, (void)sender;
    LPWSTR uri = NULL;
    BOOL user = FALSE;
    if (ICoreWebView2NavigationStartingEventArgs_get_Uri(args, &uri) == S_OK &&
        !ours(uri) && wcsncmp(uri, L"about:", 6)) {
        ICoreWebView2NavigationStartingEventArgs_put_Cancel(args, TRUE);
        /* to the browser only what a click asked for */
        if (ICoreWebView2NavigationStartingEventArgs_get_IsUserInitiated(
                args, &user) == S_OK &&
            user)
            open_outside(uri);
    }
    CoTaskMemFree(uri);
    return S_OK;
}

/* a window asked for: Brebus's own pages in a window of WebView2's, the
   rest in the browser */
static HRESULT STDMETHODCALLTYPE new_window(
    ICoreWebView2NewWindowRequestedEventHandler *self, ICoreWebView2 *sender,
    ICoreWebView2NewWindowRequestedEventArgs *args)
{
    (void)self, (void)sender;
    LPWSTR uri = NULL;
    if (ICoreWebView2NewWindowRequestedEventArgs_get_Uri(args, &uri) == S_OK &&
        !ours(uri)) {
        ICoreWebView2NewWindowRequestedEventArgs_put_Handled(args, TRUE);
        open_outside(uri);
    }
    CoTaskMemFree(uri);
    return S_OK;
}

static ICoreWebView2NavigationStartingEventHandlerVtbl nav_vtbl = {
    ICoreWebView2NavigationStartingEventHandler_qi,
    ICoreWebView2NavigationStartingEventHandler_ref,
    ICoreWebView2NavigationStartingEventHandler_ref, navigation_starting};
static ICoreWebView2NavigationStartingEventHandler nav_handler = {&nav_vtbl};

/* a frame of the page (an artifact's) going elsewhere: never - not even to
   the browser, which would carry in its address what the artifact read */
static HRESULT STDMETHODCALLTYPE frame_navigation_starting(
    ICoreWebView2NavigationStartingEventHandler *self, ICoreWebView2 *sender,
    ICoreWebView2NavigationStartingEventArgs *args)
{
    (void)self, (void)sender;
    LPWSTR uri = NULL;
    if (ICoreWebView2NavigationStartingEventArgs_get_Uri(args, &uri) == S_OK &&
        !ours(uri) && wcsncmp(uri, L"about:", 6))
        ICoreWebView2NavigationStartingEventArgs_put_Cancel(args, TRUE);
    CoTaskMemFree(uri);
    return S_OK;
}
static ICoreWebView2NavigationStartingEventHandlerVtbl frame_vtbl = {
    ICoreWebView2NavigationStartingEventHandler_qi,
    ICoreWebView2NavigationStartingEventHandler_ref,
    ICoreWebView2NavigationStartingEventHandler_ref, frame_navigation_starting};
static ICoreWebView2NavigationStartingEventHandler frame_handler = {
    &frame_vtbl};

static ICoreWebView2NewWindowRequestedEventHandlerVtbl popup_vtbl = {
    ICoreWebView2NewWindowRequestedEventHandler_qi,
    ICoreWebView2NewWindowRequestedEventHandler_ref,
    ICoreWebView2NewWindowRequestedEventHandler_ref, new_window};
static ICoreWebView2NewWindowRequestedEventHandler popup_handler = {
    &popup_vtbl};

static void fit(void)
{
    RECT r;
    if (controller && GetClientRect(main_win, &r))
        ICoreWebView2Controller_put_Bounds(controller, r);
}

static HRESULT STDMETHODCALLTYPE
controller_made(ICoreWebView2CreateCoreWebView2ControllerCompletedHandler *self,
                HRESULT rc, ICoreWebView2Controller *made)
{
    (void)self;
    if (rc != S_OK || !made) {
        fail("WebView2 could not open its view");
        return S_OK;
    }
    controller = made;
    ICoreWebView2Controller_AddRef(controller);
    ICoreWebView2Controller_get_CoreWebView2(controller, &view);
    ICoreWebView2Settings *s = NULL;
    if (ICoreWebView2_get_Settings(view, &s) == S_OK && s) {
        ICoreWebView2Settings_put_IsStatusBarEnabled(s, FALSE);
        /* the zoom is the page's (it asks it, WM_ZOOM): WebView2's own
           Ctrl+wheel would not be the page's, nor remembered */
        ICoreWebView2Settings_put_IsZoomControlEnabled(s, FALSE);
        ICoreWebView2Settings_Release(s);
    }
    EventRegistrationToken t;
    ICoreWebView2_add_NavigationStarting(view, &nav_handler, &t);
    ICoreWebView2_add_FrameNavigationStarting(view, &frame_handler, &t);
    ICoreWebView2_add_NewWindowRequested(view, &popup_handler, &t);
    ICoreWebView2_Navigate(view, start_url);
    /* where it was when it closed, or maximized the first time; a place
       on no screen any more (a monitor gone) given back to the main one */
    struct brebus_place pl;
    int kept = brebus_place_read(&pl) == 0;
    if (kept) {
        RECT r = {pl.x, pl.y, pl.x + pl.w, pl.y + pl.h};
        if (!MonitorFromRect(&r, MONITOR_DEFAULTTONULL))
            OffsetRect(&r, -r.left + 40, -r.top + 40);
        WINDOWPLACEMENT wp = {
            .length = sizeof wp, .showCmd = SW_HIDE, .rcNormalPosition = r};
        SetWindowPlacement(main_win, &wp);
    }
    ShowWindow(main_win, !kept || pl.max ? SW_SHOWMAXIMIZED : SW_SHOWNORMAL);
    SetForegroundWindow(main_win);
    /* made while the window was hidden, the view stays so until told:
       the page was drawn but the window white (Windows 11 in a VM, 16 Nov
       2026, its windows found with IsWindowVisible false) */
    ICoreWebView2Controller_put_IsVisible(controller, TRUE);
    fit();
    pthread_mutex_lock(&win_mu);
    window_on = 1;
    pthread_mutex_unlock(&win_mu);
    return S_OK;
}

static ICoreWebView2CreateCoreWebView2ControllerCompletedHandlerVtbl
    controller_vtbl = {
        ICoreWebView2CreateCoreWebView2ControllerCompletedHandler_qi,
        ICoreWebView2CreateCoreWebView2ControllerCompletedHandler_ref,
        ICoreWebView2CreateCoreWebView2ControllerCompletedHandler_ref,
        controller_made};
static ICoreWebView2CreateCoreWebView2ControllerCompletedHandler
    controller_handler = {&controller_vtbl};

static HRESULT STDMETHODCALLTYPE environment_made(
    ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler *self,
    HRESULT rc, ICoreWebView2Environment *env)
{
    (void)self;
    if (rc != S_OK || !env ||
        ICoreWebView2Environment_CreateCoreWebView2Controller(
            env, main_win, &controller_handler) != S_OK)
        fail("no WebView2 runtime (Edge's, which Windows 10 and 11 carry)");
    return S_OK;
}

static ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandlerVtbl
    environment_vtbl = {
        ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler_qi,
        ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler_ref,
        ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler_ref,
        environment_made};
static ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler
    environment_handler = {&environment_vtbl};

/* Windows's choice of light or dark for apps (Settings, Personalization,
   Colours): 1 dark */
static int system_dark(void)
{
    DWORD v = 1, n = sizeof v;
    if (RegGetValueW(HKEY_CURRENT_USER,
                     L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\"
                     L"Personalize",
                     L"AppsUseLightTheme", RRF_RT_REG_DWORD, NULL, &v,
                     &n) != ERROR_SUCCESS)
        return 0;
    return v == 0;
}

/* the title bar and the frame in the system's light or dark: Windows
   draws a program's frame light unless it asks for the dark one
   (DWMWA_USE_IMMERSIVE_DARK_MODE, 20 on Windows 11 and 10 20H1, 19
   before); dwmapi looked up at run time */
static void frame_theme(HWND h)
{
    typedef HRESULT(WINAPI * set_fn)(HWND, DWORD, LPCVOID, DWORD);
    static set_fn set;
    static int tried;
    if (!tried) {
        tried = 1;
        HMODULE d = LoadLibraryW(L"dwmapi.dll");
        if (d)
            set = (set_fn)(void (*)(void))GetProcAddress(
                d, "DwmSetWindowAttribute");
    }
    if (!set)
        return;
    BOOL dark = system_dark();
    if (set(h, 20, &dark, sizeof dark) != S_OK)
        set(h, 19, &dark, sizeof dark);
}

static WCHAR *wide(const char *s);

/* ---- closing in order: the window stays while Brebus stops its work,
   saying each step on the page ---- */
#define WM_CLOSE_SAY (WM_APP + 21)  /* lp: a step's text, malloc'ed */
#define WM_CLOSE_DONE (WM_APP + 22) /* the steps are over */
#define WM_ZOOM (WM_APP + 23)       /* wp: the page's zoom, in thousandths */
static void (*closer)(void (*)(const char *));
static int closing;

void brebus_window_set_closer(void (*steps)(void (*say)(const char *)))
{
    closer = steps;
}

/* a step, from the closer's thread: shown by the window's (text of
   Brebus's own, with no quote in it) */
static void say(const char *text)
{
    char *t = strdup(text);
    if (t && !PostMessageW(main_win, WM_CLOSE_SAY, 0, (LPARAM)t))
        free(t);
}

static void *close_steps(void *arg)
{
    (void)arg;
    closer(say);
    PostMessageW(main_win, WM_CLOSE_DONE, 0, 0);
    return NULL;
}

static LRESULT CALLBACK proc(HWND h, UINT m, WPARAM wp, LPARAM lp)
{
    switch (m) {
    case WM_CLOSE: /* asked to close: kept open while the steps run, once */
        if (!closer)
            break;
        if (!closing) {
            closing = 1;
            pthread_t t;
            if (pthread_create(&t, NULL, close_steps, NULL) != 0)
                break; /* no thread: it closes at once, the steps after */
            pthread_detach(t);
        }
        return 0;
    case WM_CLOSE_SAY: {
        char *t = (char *)lp, js[600];
        snprintf(js, sizeof js,
                 "window.brebusClosing && window.brebusClosing(\"%s\")", t);
        WCHAR *wjs = wide(js);
        if (view && wjs)
            ICoreWebView2_ExecuteScript(view, wjs, NULL);
        free(wjs);
        free(t);
        return 0;
    }
    case WM_CLOSE_DONE:
        DestroyWindow(h);
        return 0;
    case WM_ZOOM:
        if (controller)
            ICoreWebView2Controller_put_ZoomFactor(controller, wp / 1000.0);
        return 0;
    case WM_SIZE:
        /* hidden while minimized, as WebView2 asks: it stops drawing */
        if (controller)
            ICoreWebView2Controller_put_IsVisible(controller,
                                                  wp != SIZE_MINIMIZED);
        fit();
        return 0;
    case WM_MOVE:
    case WM_MOVING:
        if (controller) /* its pop-ups (menus, lists) where the window is */
            ICoreWebView2Controller_NotifyParentWindowPositionChanged(
                controller);
        break;
    case WM_SETTINGCHANGE: /* light or dark changed in Windows's settings */
        if (lp && !lstrcmpW((LPCWSTR)lp, L"ImmersiveColorSet"))
            frame_theme(h);
        break;
    case WM_TIMER: /* Ctrl+C, or Brebus ending otherwise */
        if (stop && *stop)
            DestroyWindow(h);
        return 0;
    case WM_DESTROY: {
        /* its place kept for the next start: the normal one, and whether
           it was maximized */
        WINDOWPLACEMENT wp = {.length = sizeof wp};
        if (GetWindowPlacement(h, &wp)) {
            RECT r = wp.rcNormalPosition;
            struct brebus_place pl = {.x = r.left,
                                      .y = r.top,
                                      .w = r.right - r.left,
                                      .h = r.bottom - r.top,
                                      .max = wp.showCmd == SW_SHOWMAXIMIZED};
            brebus_place_write(&pl);
        }
        PostQuitMessage(0);
        return 0;
    }
    }
    return DefWindowProcW(h, m, wp, lp);
}

static WCHAR *wide(const char *s)
{
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    WCHAR *w = n > 0 ? malloc((size_t)n * sizeof *w) : NULL;
    if (w)
        MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
    return w;
}

int brebus_window_on(void)
{
    pthread_mutex_lock(&win_mu);
    int on = window_on;
    pthread_mutex_unlock(&win_mu);
    return on;
}

/* the page's zoom, asked by the page: set on the window's thread */
int brebus_window_zoom(double z)
{
    if (!(z >= 0.25 && z <= 4) || !brebus_window_on())
        return -1;
    return PostMessageW(main_win, WM_ZOOM, (WPARAM)(z * 1000 + 0.5), 0) ? 0
                                                                        : -1;
}

/* The system's dialog to choose a folder or a file, from a request's
   thread (a COM apartment of its own), over Brebus's window. One at a
   time. */
int brebus_pick(int file, const char *title, char *out, size_t n)
{
    static pthread_mutex_t one = PTHREAD_MUTEX_INITIALIZER;
    if (!brebus_window_on())
        return -1; /* no window: the browser, where the page asks a path */
    pthread_mutex_lock(&one);
    out[0] = 0;
    HRESULT ci = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    IFileOpenDialog *d = NULL;
    if (CoCreateInstance(&CLSID_FileOpenDialog, NULL, CLSCTX_INPROC_SERVER,
                         &IID_IFileOpenDialog, (void **)&d) == S_OK) {
        DWORD o = 0;
        IFileOpenDialog_GetOptions(d, &o);
        IFileOpenDialog_SetOptions(
            d, o | FOS_FORCEFILESYSTEM |
                   (file ? FOS_FILEMUSTEXIST : FOS_PICKFOLDERS));
        WCHAR *t = wide(title);
        if (t)
            IFileOpenDialog_SetTitle(d, t);
        free(t);
        IShellItem *it = NULL;
        PWSTR p = NULL;
        if (IFileOpenDialog_Show(d, main_win) == S_OK &&
            IFileOpenDialog_GetResult(d, &it) == S_OK &&
            IShellItem_GetDisplayName(it, SIGDN_FILESYSPATH, &p) == S_OK &&
            WideCharToMultiByte(CP_UTF8, 0, p, -1, out, (int)n, NULL, NULL) > 0)
            janas_path_slashes(out);
        CoTaskMemFree(p);
        if (it)
            IShellItem_Release(it);
        IFileOpenDialog_Release(d);
    }
    if (ci == S_OK || ci == S_FALSE)
        CoUninitialize();
    pthread_mutex_unlock(&one);
    return 0;
}

int brebus_window(const char *url, volatile sig_atomic_t *quit, char *why,
                  size_t wn)
{
    /* WebView2's loader beside brebus.exe, by its full path */
    char dll[4200];
    if (janas_exe_path(dll, sizeof dll) <= 0 || !strrchr(dll, '/')) {
        snprintf(why, wn, "where is brebus.exe?");
        return -1;
    }
    snprintf(strrchr(dll, '/') + 1,
             sizeof dll - (size_t)(strrchr(dll, '/') + 1 - dll),
             "WebView2Loader.dll");
    for (char *c = dll; *c; c++)
        if (*c == '/')
            *c = '\\';
    HMODULE lib = LoadLibraryA(dll);
    create_env_fn create =
        lib ? (create_env_fn)(void (*)(void))GetProcAddress(
                  lib, "CreateCoreWebView2EnvironmentWithOptions")
            : NULL;
    if (!create) {
        snprintf(why, wn, "no WebView2Loader.dll beside brebus.exe");
        return -1;
    }
    /* sharp on screens scaled above 100% */
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    if (FAILED(CoInitializeEx(NULL, COINIT_APARTMENTTHREADED))) {
        snprintf(why, wn, "COM could not start");
        return -1;
    }
    MultiByteToWideChar(CP_UTF8, 0, url, -1, own,
                        (int)(sizeof own / sizeof *own));
    /* "?window": the page knows it is the window's, its zoom its own;
       lang= the system's languages, which the page speaks */
    static WCHAR start[300];
    char langs[160];
    brebus_sys_langs(langs, sizeof langs);
    _snwprintf(start, sizeof start / sizeof *start, L"%ls?window&lang=%hs", own,
               langs);
    start[sizeof start / sizeof *start - 1] = 0;
    start_url = start;
    stop = quit;
    HINSTANCE hi = GetModuleHandleW(NULL);
    WNDCLASSEXW wc = {.cbSize = sizeof wc,
                      .lpfnWndProc = proc,
                      .hInstance = hi,
                      .hIcon = LoadIconW(hi, MAKEINTRESOURCEW(1)),
                      .hIconSm = LoadIconW(hi, MAKEINTRESOURCEW(1)),
                      .hCursor = LoadCursorW(NULL, (LPCWSTR)IDC_ARROW),
                      /* the ground before the page is drawn, in the
                         system's light or dark: no white flash */
                      .hbrBackground = system_dark()
                                           ? CreateSolidBrush(RGB(23, 23, 26))
                                           : (HBRUSH)(COLOR_WINDOW + 1),
                      .lpszClassName = L"Brebus"};
    RegisterClassExW(&wc);
    main_win = CreateWindowExW(0, L"Brebus", L"Brebus", WS_OVERLAPPEDWINDOW,
                               CW_USEDEFAULT, CW_USEDEFAULT, 1280, 820, NULL,
                               NULL, hi, NULL);
    if (!main_win) {
        snprintf(why, wn, "no window");
        return -1;
    }
    frame_theme(main_win); /* before it shows: no light frame first */
    /* WebView2's own data (its cache, its storage) apart from Edge's */
    const char *la = getenv("LOCALAPPDATA");
    char data[4200];
    snprintf(data, sizeof data, "%s\\janas\\brebus\\webview2", la ? la : ".");
    WCHAR *wdata = wide(data);
    HRESULT rc = create(NULL, wdata, NULL, &environment_handler);
    free(wdata);
    if (rc != S_OK) {
        DestroyWindow(main_win);
        snprintf(why, wn,
                 "no WebView2 runtime (Edge's, which Windows 10 and "
                 "11 carry)");
        return -1;
    }
    SetTimer(main_win, 1, 200, NULL);
    MSG m;
    while (GetMessageW(&m, NULL, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    pthread_mutex_lock(&win_mu);
    window_on = 0;
    pthread_mutex_unlock(&win_mu);
    if (controller) {
        ICoreWebView2Controller_Close(controller);
        ICoreWebView2Controller_Release(controller);
        controller = NULL;
    }
    CoUninitialize();
    if (failed) {
        snprintf(why, wn, "%s", fail_why);
        return -1;
    }
    return 0;
}

#endif
