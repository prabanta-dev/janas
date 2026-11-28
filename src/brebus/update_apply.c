/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * An update put in place (update.h). Brebus writes the plan
 * (data_dir/update/plan.json) and starts brebus --update-finish on it,
 * outliving it, then ends. The finisher waits for it to end, then:
 *   1. puts the new version in place - an AppImage's file renamed over
 *      the old (the old kept as .old), a folder or Brebus.app unpacked
 *      beside it and swapped with it (the old kept as .old), Windows's
 *      setup run silently into the same folder;
 *   2. asks the new program its version (brebus --update-check): one that
 *      does not run is taken out, the old put back;
 *   3. with restart, starts it as the old one was started and waits for
 *      it to serve (its "started" file): if it ends first, or does not
 *      serve in time, the old is put back and started;
 *   4. says how it went in finished.json, which the next Brebus reads.
 * Its log: data_dir/update/finish.log.
 */
#include "brebus.h"
#include "common/catalog.h"
#include "common/proc.h"
#include "common/sysinfo.h"
#include "update.h"

#include <dirent.h>
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define SERVE_SECONDS 90
#define CHECK_SECONDS 30

/* ---- the plan ---- */

static const char *kind_word(enum brebus_up_kind k)
{
    return k == BREBUS_UP_APPIMAGE ? "appimage"
           : k == BREBUS_UP_FOLDER ? "folder"
           : k == BREBUS_UP_MACAPP ? "app"
           : k == BREBUS_UP_SETUP  ? "setup"
                                   : "none";
}

static enum brebus_up_kind kind_of(const char *s)
{
    return !s                       ? BREBUS_UP_NONE
           : !strcmp(s, "appimage") ? BREBUS_UP_APPIMAGE
           : !strcmp(s, "folder")   ? BREBUS_UP_FOLDER
           : !strcmp(s, "app")      ? BREBUS_UP_MACAPP
           : !strcmp(s, "setup")    ? BREBUS_UP_SETUP
                                    : BREBUS_UP_NONE;
}

static void put(struct janas_buf *o, const char *key, const char *v, int last)
{
    janas_buf_printf(o, "\"%s\": ", key);
    janas_json_write_str(o, v, strlen(v));
    janas_buf_puts(o, last ? "" : ", ");
}

/* the program the finisher runs from: on Windows a copy, so that the
   folder it replaces holds no program running */
static int finisher(struct brebus *b, char *out, size_t len, char *err,
                    size_t en)
{
    const char *ai = getenv("APPIMAGE");
    if (ai && ai[0] == '/') {
        snprintf(out, len, "%s", ai); /* mounted again for the finisher */
        return 0;
    }
    char self[4200];
    if (janas_exe_path(self, sizeof self) <= 0) {
        snprintf(err, en, "where this program is: not known");
        return -1;
    }
#ifdef _WIN32
    snprintf(out, len, "%s/update/finisher.exe", b->data_dir);
    remove(out);
    if (!CopyFileA(self, out, FALSE)) {
        snprintf(err, en, "the finisher not copied (error %lu)",
                 (unsigned long)GetLastError());
        return -1;
    }
#else
    (void)b;
    snprintf(out, len, "%s", self);
#endif
    return 0;
}

int brebus_up_apply(struct brebus *b, const struct brebus_up_where *w,
                    const char *staged, const char *version, int restart,
                    char *err, size_t err_len)
{
    if (w->kind == BREBUS_UP_NONE || w->kind == BREBUS_UP_SYSTEM) {
        snprintf(err, err_len, "%s", w->why[0] ? w->why : "not updatable");
        return -1;
    }
    char dir[4200], plan[4300], tmp[4310], fin[4300];
    snprintf(dir, sizeof dir, "%s/update", b->data_dir);
    snprintf(plan, sizeof plan, "%s/plan.json", dir);
    snprintf(tmp, sizeof tmp, "%s.tmp", plan);
    if (finisher(b, fin, sizeof fin, err, err_len) != 0)
        return -1;
    struct janas_buf o = {0};
    char pid[32];
#ifdef _WIN32
    snprintf(pid, sizeof pid, "%lu", (unsigned long)GetCurrentProcessId());
#else
    snprintf(pid, sizeof pid, "%ld", (long)getpid());
#endif
    janas_buf_puts(&o, "{");
    put(&o, "pid", pid, 0);
    put(&o, "kind", kind_word(w->kind), 0);
    put(&o, "target", w->target, 0);
    put(&o, "staged", staged, 0);
    put(&o, "version", version, 0);
    put(&o, "from", brebus_up_version(), 0);
    put(&o, "dir", dir, 0);
    put(&o, "restart", restart ? "yes" : "no", 0);
    janas_buf_puts(&o, "\"args\": [");
    for (int i = 1; i < b->argc; i++) {
        janas_buf_puts(&o, i > 1 ? ", " : "");
        janas_json_write_str(&o, b->argv[i], strlen(b->argv[i]));
    }
    janas_buf_puts(&o, "]}\n");
    FILE *f = o.oom ? NULL : fopen(tmp, "wb");
    int ok = f && fwrite(o.p, 1, o.n, f) == o.n;
    if (f)
        ok &= fclose(f) == 0;
    janas_buf_free(&o);
    if (!ok || rename(tmp, plan) != 0) {
        remove(tmp);
        snprintf(err, err_len, "%s: not written", plan);
        return -1;
    }
    const char *argv[] = {fin, "--update-finish", plan, NULL};
    struct janas_proc p;
    /* in the update's folder: not in the one it replaces */
    if (janas_proc_start(&p, argv, NULL, dir, -1, -1, -1, JANAS_PROC_OUTLIVE,
                         err, err_len) != 0)
        return -1;
    if (restart)
        brebus_quit();
    return 0;
}

/* ---- the finisher ---- */

static FILE *logf;

static void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void say(const char *fmt, ...)
{
    if (!logf)
        return;
    time_t t = time(NULL);
    char when[32];
    strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", localtime(&t));
    fprintf(logf, "%s ", when);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(logf, fmt, ap);
    va_end(ap);
    fputc('\n', logf);
    fflush(logf);
}

static void nap(int ms)
{
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
#endif
}

/* whether the process pid still runs */
static int alive(unsigned long pid)
{
#ifdef _WIN32
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)pid);
    if (!h)
        return 0;
    DWORD r = WaitForSingleObject(h, 0);
    CloseHandle(h);
    return r == WAIT_TIMEOUT;
#else
    return kill((pid_t)pid, 0) == 0 || errno == EPERM;
#endif
}

static int is_dir(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

static int exists(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0;
}

static void drop(const char *p)
{
    if (is_dir(p))
        brebus_remove_tree(p, 0);
    else
        remove(p);
}

static int copy_file(const char *from, const char *to)
{
    FILE *a = fopen(from, "rb"), *b = a ? fopen(to, "wb") : NULL;
    int ok = a && b;
    char buf[1 << 16];
    for (size_t n; ok && (n = fread(buf, 1, sizeof buf, a)) > 0;)
        ok = fwrite(buf, 1, n, b) == n;
    ok &= a && !ferror(a);
    if (a)
        fclose(a);
    if (b)
        ok &= fclose(b) == 0;
    if (!ok)
        remove(to);
    return ok ? 0 : -1;
}

/* a program run to its end, up to seconds; its exit status, -1 if it did
   not end or start. out: its first line (may be NULL) */
static int run(const char *const argv[], const char *dir, int seconds,
               char *out, size_t len)
{
    int fd[2] = {-1, -1};
    if (out && janas_pipe(fd) != 0)
        return -1;
    struct janas_proc p;
    char err[300];
    int nul = janas_null_fd(1);
    int rc = janas_proc_start(&p, argv, NULL, dir, -1, out ? fd[1] : nul, nul,
                              JANAS_PROC_DETACHED, err, sizeof err);
    if (nul >= 0)
        close(nul);
    if (out)
        close(fd[1]);
    if (rc != 0) {
        say("%s", err);
        if (out)
            close(fd[0]);
        return -1;
    }
    if (out) {
        size_t n = 0;
        for (ssize_t r;
             n + 1 < len && (r = read(fd[0], out + n, len - 1 - n)) > 0;)
            n += (size_t)r;
        out[n] = 0;
        char *nl = strchr(out, '\n');
        if (nl)
            *nl = 0;
        close(fd[0]);
    }
    int code = -1;
    if (janas_proc_wait(&p, seconds * 1000, &code) != 1) {
        janas_proc_kill(&p, 1);
        janas_proc_wait(&p, 5000, NULL);
        return -1;
    }
    return code;
}

/* the archive unpacked into dir */
static int unpack(enum brebus_up_kind kind, const char *file, const char *dir)
{
#if defined(__APPLE__)
    (void)kind;
    const char *argv[] = {"/usr/bin/ditto", "-x", "-k", file, dir, NULL};
#elif defined(_WIN32)
    (void)kind;
    /* Windows's own bsdtar (Windows 10 1803 on) reads a .zip */
    char tar[MAX_PATH + 32];
    UINT n = GetSystemDirectoryA(tar, MAX_PATH);
    snprintf(tar + n, sizeof tar - n, "\\tar.exe");
    const char *argv[] = {tar, "-xf", file, "-C", dir, NULL};
#else
    (void)kind;
    const char *argv[] = {"tar", "-xzf", file, "-C", dir, NULL};
#endif
    int rc = run(argv, dir, 600, NULL, 0);
    if (rc != 0)
        say("unpacking %s: exit status %d", file, rc);
    return rc == 0 ? 0 : -1;
}

/* the one folder an archive held (Brebus.app for macOS's): into out */
static int only_entry(const char *dir, enum brebus_up_kind kind, char *out,
                      size_t len)
{
    if (kind == BREBUS_UP_MACAPP) {
        snprintf(out, len, "%s/Brebus.app", dir);
        return is_dir(out) ? 0 : -1;
    }
    DIR *d = opendir(dir);
    if (!d)
        return -1;
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        snprintf(out, len, "%.4000s/%.255s", dir, e->d_name);
        n++;
    }
    closedir(d);
    return n == 1 && is_dir(out) ? 0 : -1;
}

struct plan {
    unsigned long pid;
    enum brebus_up_kind kind;
    char target[4096], staged[4400], version[64], from[64], dir[4200];
    int restart;
    const char *args[64];
    int n_args;
    struct janas_json_doc *doc;
};

static int read_plan(const char *path, struct plan *p)
{
    memset(p, 0, sizeof *p);
    struct janas_buf f = {0};
    FILE *in = fopen(path, "rb");
    if (!in)
        return -1;
    char buf[4096];
    for (size_t n; (n = fread(buf, 1, sizeof buf, in)) > 0;)
        janas_buf_put(&f, buf, n);
    fclose(in);
    char err[160];
    p->doc = f.oom ? NULL : janas_json_parse(f.p, f.n, err, sizeof err);
    janas_buf_free(&f);
    const struct janas_json *r = p->doc ? janas_json_root(p->doc) : NULL;
    if (!r)
        return -1;
#define S(k, dst)                                                              \
    do {                                                                       \
        const char *s = janas_json_str(janas_json_get(r, k));                  \
        if (!s || strlen(s) >= sizeof(dst))                                    \
            return -1;                                                         \
        snprintf(dst, sizeof(dst), "%s", s);                                   \
    } while (0)
    char pid[32], kind[16], restart[8];
    S("pid", pid);
    S("kind", kind);
    S("target", p->target);
    S("staged", p->staged);
    S("version", p->version);
    S("from", p->from);
    S("dir", p->dir);
    S("restart", restart);
#undef S
    p->pid = strtoul(pid, NULL, 10);
    p->kind = kind_of(kind);
    p->restart = !strcmp(restart, "yes");
    const struct janas_json *a = janas_json_get(r, "args");
    for (const struct janas_json *e = a ? a->child : NULL; e && p->n_args < 62;
         e = e->next)
        if (janas_json_str(e))
            p->args[p->n_args++] = janas_json_str(e);
    return p->pid && p->kind != BREBUS_UP_NONE ? 0 : -1;
}

static void finished(const struct plan *p, int ok, const char *why)
{
    char path[4300];
    snprintf(path, sizeof path, "%s/finished.json", p->dir);
    struct janas_buf o = {0};
    janas_buf_puts(&o, "{");
    put(&o, "ok", ok ? "yes" : "no", 0);
    put(&o, "version", p->version, 0);
    put(&o, "from", p->from, 0);
    put(&o, "why", why, 1);
    janas_buf_puts(&o, "}\n");
    FILE *f = o.oom ? NULL : fopen(path, "wb");
    if (f) {
        fwrite(o.p, 1, o.n, f);
        fclose(f);
    }
    janas_buf_free(&o);
    say("%s: %s%s%s", p->version, ok ? "installed" : "NOT installed",
        why[0] ? ": " : "", why);
}

/* the new version in place of the old (kept at old) */
static int swap_in(const struct plan *p, const char *old, char *why, size_t wl)
{
    if (p->kind == BREBUS_UP_SETUP) {
        /* the setup into the same folder, silently: NSIS's /D last, as
           it is, no quotes */
        char d[4200];
        snprintf(d, sizeof d, "/D=%s", p->target);
        for (char *c = d; *c; c++)
            if (*c == '/')
                *c = '\\';
        const char *argv[] = {p->staged, "/S", d, NULL};
        int rc = run(argv, p->dir, 600, NULL, 0);
        if (rc != 0) {
            snprintf(why, wl, "the setup ended with status %d", rc);
            return -1;
        }
        return 0;
    }
    drop(old);
    if (p->kind == BREBUS_UP_APPIMAGE) {
        char nw[4200];
        snprintf(nw, sizeof nw, "%s.new", p->target);
        if (copy_file(p->staged, nw) != 0 || chmod(nw, 0755) != 0) {
            snprintf(why, wl, "the new AppImage not written beside the old");
            remove(nw);
            return -1;
        }
        if (rename(p->target, old) != 0 || rename(nw, p->target) != 0) {
            snprintf(why, wl, "the new AppImage not put in place: %s",
                     strerror(errno));
            if (!exists(p->target))
                rename(old, p->target);
            remove(nw);
            return -1;
        }
        return 0;
    }
    /* a folder or Brebus.app: unpacked beside it, then swapped */
    char tmp[4200], top[4400], up[4096];
    snprintf(up, sizeof up, "%s", p->target);
    char *s = strrchr(up, '/');
    if (!s) {
        snprintf(why, wl, "%.300s: no folder above it", p->target);
        return -1;
    }
    *s = 0;
    snprintf(tmp, sizeof tmp, "%s/.brebus-update", up);
    drop(tmp);
    if (janas_mkdirs(tmp) != 0 || unpack(p->kind, p->staged, tmp) != 0 ||
        only_entry(tmp, p->kind, top, sizeof top) != 0) {
        snprintf(why, wl, "the new version not unpacked beside the old");
        drop(tmp);
        return -1;
    }
    if (rename(p->target, old) != 0) {
        snprintf(why, wl, "the old version not moved aside: %s",
                 strerror(errno));
        drop(tmp);
        return -1;
    }
    if (rename(top, p->target) != 0) {
        snprintf(why, wl, "the new version not put in place: %s",
                 strerror(errno));
        rename(old, p->target);
        drop(tmp);
        return -1;
    }
    drop(tmp);
    return 0;
}

/* the old version back in its place */
static void swap_back(const struct plan *p, const char *old)
{
    if (p->kind == BREBUS_UP_SETUP || !exists(old))
        return;
    char bad[4200];
    snprintf(bad, sizeof bad, "%s.bad", p->target);
    drop(bad);
    if (rename(p->target, bad) == 0 && rename(old, p->target) == 0)
        say("the old version put back");
    else
        say("the old version NOT put back: it is at %s", old);
    drop(bad);
}

/* the program started as Brebus was, outliving the finisher */
static int start(const struct plan *p, struct janas_proc *pr)
{
    char prog[4300], err[300];
    brebus_up_program(p->kind, p->target, prog, sizeof prog);
    const char *argv[66];
    int n = 0;
    argv[n++] = prog;
    for (int i = 0; i < p->n_args; i++)
        argv[n++] = p->args[i];
    argv[n] = NULL;
    if (janas_proc_start(pr, argv, NULL, p->dir, -1, -1, -1, JANAS_PROC_OUTLIVE,
                         err, sizeof err) != 0) {
        say("%s", err);
        return -1;
    }
    return 0;
}

/* the new Brebus serves: its started file rewritten after since */
static int serves(const struct plan *p, struct janas_proc *pr, time_t since)
{
    char path[4300];
    snprintf(path, sizeof path, "%s/" BREBUS_UP_STARTED, p->dir);
    for (int i = 0; i < SERVE_SECONDS * 5; i++) {
        struct stat st;
        if (stat(path, &st) == 0 && st.st_mtime >= since)
            return 1;
        if (janas_proc_wait(pr, 0, NULL) == 1)
            return 0; /* it ended */
        nap(200);
    }
    return 0;
}

int brebus_update_finish(const char *path)
{
    struct plan p;
    if (read_plan(path, &p) != 0) {
        janas_json_free(p.doc);
        fprintf(stderr, "brebus: %s: not an update's plan\n", path);
        return 2;
    }
    char lp[4300], old[4200], why[400] = "";
    snprintf(lp, sizeof lp, "%s/finish.log", p.dir);
    logf = fopen(lp, "ab");
    say("%s to %s (%s): waiting for Brebus to end", p.from, p.version,
        kind_word(p.kind));
    for (int i = 0; alive(p.pid); i++) {
        if (i == 120 * 5) {
            finished(&p, 0, "Brebus did not end");
            return 1;
        }
        nap(200);
    }
    nap(500); /* its files let go */
    snprintf(old, sizeof old, "%s.old", p.target);
    if (swap_in(&p, old, why, sizeof why) != 0) {
        finished(&p, 0, why);
        janas_json_free(p.doc);
        return 1;
    }
    /* the new program runs at all */
    char prog[4300], said[200] = "";
    brebus_up_program(p.kind, p.target, prog, sizeof prog);
    const char *argv[] = {prog, "--update-check", NULL};
    int rc = run(argv, p.dir, CHECK_SECONDS, said, sizeof said);
    if (rc != 0 || strncmp(said, "brebus ", 7) != 0) {
        snprintf(why, sizeof why,
                 "the new version does not run here (status %d)", rc);
        swap_back(&p, old);
        finished(&p, 0, why);
        struct janas_proc back;
        if (p.restart && start(&p, &back) == 0)
            say("the old version started again");
    } else if (p.restart) {
        struct janas_proc pr = {0};
        time_t since = time(NULL);
        if (start(&p, &pr) == 0 && serves(&p, &pr, since)) {
            drop(old);
            finished(&p, 1, "");
        } else {
            if (pr.id) {
                janas_proc_kill(&pr, 1);
                janas_proc_wait(&pr, 5000, NULL);
            }
            snprintf(why, sizeof why,
                     "the new version did not start within %d seconds",
                     SERVE_SECONDS);
            swap_back(&p, old);
            finished(&p, 0, why);
            struct janas_proc back;
            if (start(&p, &back) == 0)
                say("the old version started again");
        }
    } else {
        drop(old);
        finished(&p, 1, "");
    }
    remove(p.staged);
    remove(path);
    janas_json_free(p.doc);
    if (logf)
        fclose(logf);
    return 0;
}
