/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * window_mac.m - Brebus in a window of its own on macOS: WebKit's
 * WKWebView, the system's own, shows its page as an application does,
 * with its name and the Prabanta logo, as window.c does on Linux with
 * WebKitGTK and window_win.c on Windows with WebView2. Links to other
 * sites open in the browser when clicked (outside.c); a page of Brebus's
 * own opened apart (its engine monitor) gets a window of its own; the
 * window closed, Brebus ends, after the steps of its closer.
 */
#if defined(__APPLE__)
#import <Cocoa/Cocoa.h>
#import <WebKit/WebKit.h>

#include "brebus.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

static const char *own; /* the page's address, "http://127.0.0.1:port/" */
static volatile sig_atomic_t *stop;
static NSWindow *main_win;
static WKWebView *main_view;
static struct brebus_place place;
static int placed;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static int window_on, pick_busy;

static void (*closer)(void (*)(const char *));
static int closing;
static void *close_steps(void *arg);

static int ours(NSURL *u)
{
    const char *s = u.absoluteString.UTF8String;
    return s && !strncmp(s, own, strlen(own));
}

/* ---- the page's navigation: Brebus's own pages here, another site's in
   the browser only when a click asked for a new window ---- */
@interface BrebusDelegate : NSObject <WKNavigationDelegate, WKUIDelegate,
                                      NSWindowDelegate>
@end

@implementation BrebusDelegate

- (void)webView:(WKWebView *)view
    decidePolicyForNavigationAction:(WKNavigationAction *)a
                    decisionHandler:
                        (void (^)(WKNavigationActionPolicy))decide
{
    (void)view;
    NSURL *u = a.request.URL;
    if (!u || ours(u) || [u.scheme isEqualToString:@"about"] ||
        [u.scheme isEqualToString:@"blob"] ||
        [u.scheme isEqualToString:@"data"]) {
        decide(WKNavigationActionPolicyAllow);
        return;
    }
    /* a frame going elsewhere (an artifact's, sandboxed, opens no window)
       goes nowhere, even when clicked: its address may carry what the
       artifact read */
    if (a.targetFrame == nil && a.navigationType == WKNavigationTypeLinkActivated)
        brebus_open_outside(u.absoluteString.UTF8String);
    decide(WKNavigationActionPolicyCancel);
}

/* window.open and target=_blank: Brebus's own page in a second window,
   another site's in the browser when clicked */
- (WKWebView *)webView:(WKWebView *)view
    createWebViewWithConfiguration:(WKWebViewConfiguration *)conf
               forNavigationAction:(WKNavigationAction *)a
                    windowFeatures:(WKWindowFeatures *)f
{
    (void)view, (void)f;
    NSURL *u = a.request.URL;
    if (!u)
        return nil;
    if (!ours(u)) {
        if (a.navigationType == WKNavigationTypeLinkActivated ||
            a.navigationType == WKNavigationTypeOther)
            brebus_open_outside(u.absoluteString.UTF8String);
        return nil;
    }
    NSWindow *w = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0, 0, 1100, 760)
                  styleMask:NSWindowStyleMaskTitled |
                            NSWindowStyleMaskClosable |
                            NSWindowStyleMaskResizable |
                            NSWindowStyleMaskMiniaturizable
                    backing:NSBackingStoreBuffered
                      defer:NO];
    w.title = @"Brebus";
    w.releasedWhenClosed = NO;
    WKWebView *nv = [[WKWebView alloc] initWithFrame:w.contentView.bounds
                                       configuration:conf];
    nv.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    nv.navigationDelegate = self;
    nv.UIDelegate = self;
    w.contentView = nv;
    [w center];
    [w makeKeyAndOrderFront:nil];
    return nv;
}

/* the page's window.close(): its window goes */
- (void)webViewDidClose:(WKWebView *)view
{
    if (view != main_view)
        [view.window close];
}

/* the microphone, for dictation; never the camera */
- (void)webView:(WKWebView *)view
    requestMediaCapturePermissionForOrigin:(WKSecurityOrigin *)origin
                          initiatedByFrame:(WKFrameInfo *)frame
                                      type:(WKMediaCaptureType)type
                           decisionHandler:
                               (void (^)(WKPermissionDecision))decide
    API_AVAILABLE(macos(12.0))
{
    (void)view, (void)origin, (void)frame;
    decide(type == WKMediaCaptureTypeMicrophone ? WKPermissionDecisionGrant
                                                : WKPermissionDecisionDeny);
}

/* ---- closing in order: the window stays while Brebus stops its work,
   saying each step on the page ---- */
- (BOOL)windowShouldClose:(NSWindow *)w
{
    if (w != main_win || !closer)
        return YES;
    if (closing)
        return NO;
    closing = 1;
    pthread_t t;
    if (pthread_create(&t, NULL, close_steps, NULL) != 0)
        return YES; /* no thread: it closes at once, the steps after it */
    pthread_detach(t);
    return NO;
}

- (void)windowWillClose:(NSNotification *)n
{
    if (n.object == main_win)
        [NSApp stop:nil];
}

@end

static BrebusDelegate *delegate;

/* the run loop stopped from anywhere: an event posted wakes it */
static void stop_app(void)
{
    [NSApp stop:nil];
    NSEvent *e = [NSEvent otherEventWithType:NSEventTypeApplicationDefined
                                    location:NSZeroPoint
                               modifierFlags:0
                                   timestamp:0
                                windowNumber:0
                                     context:nil
                                     subtype:0
                                       data1:0
                                       data2:0];
    [NSApp postEvent:e atStart:YES];
}

void brebus_window_set_closer(void (*steps)(void (*say)(const char *)))
{
    closer = steps;
}

/* a step, from the closer's thread: shown by the main one (text of
   Brebus's own, with no quote in it) */
static void say(const char *text)
{
    NSString *js = [NSString
        stringWithFormat:@"window.brebusClosing && window.brebusClosing(\"%s\")",
                         text];
    dispatch_async(dispatch_get_main_queue(), ^{
      [main_view evaluateJavaScript:js completionHandler:nil];
    });
}

static void *close_steps(void *arg)
{
    (void)arg;
    closer(say);
    dispatch_async(dispatch_get_main_queue(), ^{
      [main_win close];
    });
    return NULL;
}

int brebus_window_on(void)
{
    pthread_mutex_lock(&mu);
    int on = window_on;
    pthread_mutex_unlock(&mu);
    return on;
}

/* The system's dialog to choose a folder or a file, for the page: asked
   from a request's thread, run on the main one, the answer waited for. */
int brebus_pick(int file, const char *title, char *out, size_t n)
{
    pthread_mutex_lock(&mu);
    while (pick_busy && window_on) /* one dialog at a time */
        pthread_cond_wait(&cv, &mu);
    if (!window_on) {
        pthread_mutex_unlock(&mu);
        return -1; /* no window: the browser, where the page asks a path */
    }
    pick_busy = 1;
    pthread_mutex_unlock(&mu);
    __block char path[4096] = "";
    NSString *t = [NSString stringWithUTF8String:title ? title : ""];
    dispatch_sync(dispatch_get_main_queue(), ^{
      NSOpenPanel *p = [NSOpenPanel openPanel];
      p.message = t;
      p.canChooseFiles = file != 0;
      p.canChooseDirectories = file == 0;
      p.allowsMultipleSelection = NO;
      p.canCreateDirectories = YES;
      if ([p runModal] == NSModalResponseOK && p.URL.fileSystemRepresentation)
          snprintf(path, sizeof path, "%s", p.URL.fileSystemRepresentation);
    });
    snprintf(out, n, "%s", path);
    pthread_mutex_lock(&mu);
    pick_busy = 0;
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mu);
    return 0;
}

/* the page's zoom: the page keeps it and asks it of the window (its menu,
   Ctrl or Cmd with the wheel, plus, minus and 0), the whole page drawn
   larger as a browser's zoom does */
int brebus_window_zoom(double z)
{
    if (!(z >= 0.25 && z <= 4) || !brebus_window_on())
        return -1;
    dispatch_async(dispatch_get_main_queue(), ^{
      if (@available(macOS 11.0, *))
          main_view.pageZoom = z;
    });
    return 0;
}

/* the menus macOS's keys go through: without Edit's, Cmd+C and Cmd+V do
   nothing in the page */
static void menus(void)
{
    NSMenu *bar = [NSMenu new];
    NSMenuItem *app = [NSMenuItem new], *edit = [NSMenuItem new];
    [bar addItem:app];
    [bar addItem:edit];
    NSMenu *am = [NSMenu new];
    [am addItemWithTitle:@"Hide Brebus"
                  action:@selector(hide:)
           keyEquivalent:@"h"];
    [am addItem:[NSMenuItem separatorItem]];
    [am addItemWithTitle:@"Quit Brebus"
                  action:@selector(performClose:)
           keyEquivalent:@"q"];
    app.submenu = am;
    NSMenu *em = [[NSMenu alloc] initWithTitle:@"Edit"];
    [em addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
    [em addItemWithTitle:@"Redo"
                  action:@selector(redo:)
           keyEquivalent:@"Z"];
    [em addItem:[NSMenuItem separatorItem]];
    [em addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
    [em addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    [em addItemWithTitle:@"Paste"
                  action:@selector(paste:)
           keyEquivalent:@"v"];
    [em addItemWithTitle:@"Select All"
                  action:@selector(selectAll:)
           keyEquivalent:@"a"];
    edit.submenu = em;
    NSApp.mainMenu = bar;
}

int brebus_window(const char *url, volatile sig_atomic_t *quit, char *why,
                  size_t wn)
{
    @autoreleasepool {
        if (![NSThread isMainThread]) {
            snprintf(why, wn, "the window must be opened by the main thread");
            return -1;
        }
        own = url;
        stop = quit;
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        menus();
        const struct brebus_icon *logo = brebus_icon(256);
        NSImage *icon = [[NSImage alloc]
            initWithData:[NSData dataWithBytes:logo->png length:logo->len]];
        if (icon)
            NSApp.applicationIconImage = icon;
        delegate = [BrebusDelegate new];
        struct brebus_place kept;
        int have = brebus_place_read(&kept) == 0;
        main_win = [[NSWindow alloc]
            initWithContentRect:NSMakeRect(0, 0, 1280, 820)
                      styleMask:NSWindowStyleMaskTitled |
                                NSWindowStyleMaskClosable |
                                NSWindowStyleMaskResizable |
                                NSWindowStyleMaskMiniaturizable
                        backing:NSBackingStoreBuffered
                          defer:NO];
        main_win.title = @"Brebus";
        main_win.releasedWhenClosed = NO;
        main_win.delegate = delegate;
        WKWebViewConfiguration *conf = [WKWebViewConfiguration new];
        main_view = [[WKWebView alloc] initWithFrame:main_win.contentView.bounds
                                       configuration:conf];
        main_view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        main_view.navigationDelegate = delegate;
        main_view.UIDelegate = delegate;
        main_win.contentView = main_view;
        /* "?window": the page knows it is the window's, its zoom its own */
        NSString *start = [NSString stringWithFormat:@"%s?window", url];
        [main_view loadRequest:[NSURLRequest
                                   requestWithURL:[NSURL URLWithString:start]]];
        /* where it was when it closed (the whole frame, as kept), or in the
           middle and zoomed the first time */
        if (have)
            [main_win setFrame:NSMakeRect(kept.x, kept.y, kept.w, kept.h)
                       display:NO];
        else
            [main_win center];
        if (!have || kept.max)
            [main_win zoom:nil];
        [main_win makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];
        pthread_mutex_lock(&mu);
        window_on = 1;
        pthread_mutex_unlock(&mu);
        /* Ctrl+C or SIGTERM, and the window's place, a few times a second */
        NSTimer *watch = [NSTimer
            scheduledTimerWithTimeInterval:0.2
                                   repeats:YES
                                     block:^(NSTimer *tm) {
                                       (void)tm;
                                       place.max = main_win.zoomed;
                                       if (!place.max) {
                                           NSRect f = main_win.frame;
                                           place.x = (int)f.origin.x;
                                           place.y = (int)f.origin.y;
                                           place.w = (int)f.size.width;
                                           place.h = (int)f.size.height;
                                       }
                                       placed = 1;
                                       if (*stop)
                                           stop_app();
                                     }];
        [NSApp run];
        [watch invalidate];
        if (placed)
            brebus_place_write(&place);
        pthread_mutex_lock(&mu);
        window_on = 0; /* a request waiting on a dialog is let go */
        pthread_cond_broadcast(&cv);
        pthread_mutex_unlock(&mu);
        [main_win orderOut:nil];
        return 0;
    }
}

#endif /* __APPLE__ */
