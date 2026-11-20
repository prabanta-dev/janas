/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * run.c - a skill's script run in the cage of common/sandbox.h: no
 * network, the skill's folder read-only, a folder of its own writable (its
 * current folder), limits on time, memory and what it writes. The folder
 * is kept when the script wrote in it, under ~/.cache/janas/skill-runs/
 * (%LOCALAPPDATA%\janas\skill-runs), and those of more than a day ago are
 * removed.
 *
 * Linux: Python, shell and JavaScript, by the system's /usr/bin/python3,
 * /bin/sh and /usr/bin/node, the system's /usr read-only. Windows: Python
 * alone, the one Brebus installs from python.org (%LOCALAPPDATA%\janas\
 * python\runtime), as its code interpreter runs it.
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef _WIN32
#include <io.h>
#include <pthread.h>
#include <windows.h>
#else
#include <poll.h>
#include <sys/wait.h>
#endif

#include "common/catalog.h"
#include "common/mcp_server.h"
#include "common/sandbox.h"
#include "skills.h"

#define RUN_SECONDS 120
#define MAX_OUT (64 * 1024) /* of each of its outputs, kept */
#define MAX_ARGS 32

#ifdef _WIN32
#define lstat stat /* no links to follow out of a folder of runs */
#endif

static int fail(struct janas_buf *b, const char *t)
{
    janas_mcps_text_result(b, t, strlen(t), 1);
    return 0;
}

static void remove_tree(const char *dir, int depth)
{
    DIR *d = depth < 8 ? opendir(dir) : NULL;
    struct dirent *e;
    while (d && (e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        char p[4500];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        if (lstat(p, &st) == 0 && S_ISDIR(st.st_mode))
            remove_tree(p, depth + 1);
        else
            remove(p);
    }
    if (d)
        closedir(d);
    rmdir(dir);
}

/* the folder of the runs, those of more than a day ago removed: 0, or -1 */
static int runs_dir(char *out, size_t n)
{
#ifdef _WIN32
    const char *la = getenv("LOCALAPPDATA");
    if (!la || !*la)
        return -1;
    snprintf(out, n, "%s\\janas\\skill-runs", la);
#else
    const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
    if (xdg && *xdg)
        snprintf(out, n, "%s/janas/skill-runs", xdg);
    else if (home && *home)
        snprintf(out, n, "%s/.cache/janas/skill-runs", home);
    else
        return -1;
#endif
    if (janas_mkdirs(out) != 0)
        return -1;
    DIR *d = opendir(out);
    struct dirent *e;
    time_t old = time(NULL) - 24 * 3600;
    while (d && (e = readdir(d))) {
        if (e->d_name[0] == '.')
            continue;
        char p[4500];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", out, e->d_name);
        if (lstat(p, &st) == 0 && S_ISDIR(st.st_mode) && st.st_mtime < old)
            remove_tree(p, 0);
    }
    if (d)
        closedir(d);
    return 0;
}

/* the files the script left in its folder, by their paths there */
static void written(const char *dir, const char *rel, struct janas_buf *t,
                    int *count, int depth)
{
    DIR *d = depth < 4 ? opendir(dir) : NULL;
    struct dirent *e;
    while (d && *count < 50 && (e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        char p[4600], r[1024];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        snprintf(r, sizeof r, "%s%s%s", rel, *rel ? "/" : "", e->d_name);
        if (lstat(p, &st) != 0)
            continue;
        if (S_ISDIR(st.st_mode))
            written(p, r, t, count, depth + 1);
        else if (S_ISREG(st.st_mode)) {
            janas_buf_printf(t, "\n- %s (%lld bytes)", r,
                             (long long)st.st_size);
            ++*count;
        }
    }
    if (d)
        closedir(d);
}

/* what the cage writes, kept up to MAX_OUT */
struct out {
    struct janas_buf b;
    size_t lost;
    int fd;
};

/* one read of o->fd: 1, or 0 at its end */
static int drain(struct out *o)
{
    char buf[8192];
    ssize_t r = read(o->fd, buf, sizeof buf);
    if (r <= 0)
        return r < 0 && errno == EINTR ? 1 : 0;
    size_t keep = o->b.n < MAX_OUT ? MAX_OUT - o->b.n : 0;
    if (keep > (size_t)r)
        keep = (size_t)r;
    janas_buf_put(&o->b, buf, keep);
    o->lost += (size_t)r - keep;
    return 1;
}

/* The script's interpreter by its extension, into argv[0..]: the count of
   argv filled, or 0 with the reason in why. */
static int interpreter(const char *script, char **argv, char *why, size_t n)
{
    const char *dot = strrchr(script, '.');
#ifdef _WIN32
    static char py[4200];
    const char *la = getenv("LOCALAPPDATA");
    snprintf(py, sizeof py, "%s\\janas\\python\\runtime\\python.exe",
             la ? la : ".");
    if (!dot || strcmp(dot, ".py")) {
        snprintf(why, n, "Only scripts in Python (.py) are run on Windows.");
        return 0;
    }
    if (access(py, F_OK) != 0) {
        snprintf(why, n,
                 "Python is not installed for Janas: Brebus installs it "
                 "(Settings, Run code). Tell the user.");
        return 0;
    }
    argv[0] = py;
    argv[1] = "-I";
    return 2;
#else
    if (dot && !strcmp(dot, ".py"))
        argv[0] = "/usr/bin/python3", argv[1] = "-I";
    else if (dot && !strcmp(dot, ".sh"))
        argv[0] = "/bin/sh", argv[1] = NULL;
    else if (dot && (!strcmp(dot, ".js") || !strcmp(dot, ".mjs")))
        argv[0] = "/usr/bin/node", argv[1] = NULL;
    else {
        snprintf(why, n,
                 "Only scripts in Python (.py), shell (.sh) and "
                 "JavaScript (.js, .mjs) are run.");
        return 0;
    }
    if (access(argv[0], X_OK) != 0) {
        snprintf(why, n, "%s is not installed on this computer.", argv[0]);
        return 0;
    }
    return argv[1] ? 2 : 1;
#endif
}

/* the run's environment, its folder known */
static char **environment(const char *run)
{
#ifdef _WIN32
    static char e_path[4400], e_root[300], e_tmp[4400], e_temp[4400],
        e_home[4400];
    const char *la = getenv("LOCALAPPDATA"), *sr = getenv("SYSTEMROOT");
    snprintf(e_path, sizeof e_path,
             "PATH=%s\\janas\\python\\runtime;%s\\System32", la ? la : ".",
             sr ? sr : "C:\\Windows");
    snprintf(e_root, sizeof e_root, "SYSTEMROOT=%s", sr ? sr : "C:\\Windows");
    snprintf(e_tmp, sizeof e_tmp, "TMP=%s", run);
    snprintf(e_temp, sizeof e_temp, "TEMP=%s", run);
    snprintf(e_home, sizeof e_home, "USERPROFILE=%s", run);
    static char *envp[] = {e_path,
                           e_root,
                           e_tmp,
                           e_temp,
                           e_home,
                           "PYTHONIOENCODING=utf-8",
                           "PYTHONUTF8=1",
                           "PYTHONDONTWRITEBYTECODE=1",
                           NULL};
#else
    (void)run;
    static char *envp[] = {"PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/tmp",
                           "LANG=C.UTF-8", "PYTHONDONTWRITEBYTECODE=1", NULL};
#endif
    return envp;
}

#ifdef _WIN32
static void *reader(void *arg)
{
    while (drain(arg))
        ;
    return NULL;
}

struct feed {
    int fd;
    const char *p;
    size_t n;
};

static void *feeder(void *arg)
{
    struct feed *f = arg;
    size_t at = 0;
    while (at < f->n) {
        int w = _write(f->fd, f->p + at,
                       (unsigned)(f->n - at > 65536 ? 65536 : f->n - at));
        if (w <= 0)
            break;
        at += (size_t)w;
    }
    _close(f->fd);
    return NULL;
}

/* its input given, its outputs read, until it ends or its time is up: its
   exit code, or -1 when it was stopped */
static int collect(struct janas_sandbox *sb, const char *input, struct out *o)
{
    pthread_t t[3];
    struct feed f = {.fd = sb->in, .p = input, .n = input ? strlen(input) : 0};
    sb->in = -1; /* the feeder's to close */
    int fed = pthread_create(&t[2], NULL, feeder, &f) == 0;
    if (!fed)
        _close(f.fd);
    o[0].fd = sb->out, o[1].fd = sb->err;
    int r0 = pthread_create(&t[0], NULL, reader, &o[0]) == 0;
    int r1 = pthread_create(&t[1], NULL, reader, &o[1]) == 0;
    HANDLE ph = (HANDLE)sb->process;
    int status = -1;
    if (WaitForSingleObject(ph, RUN_SECONDS * 1000) == WAIT_OBJECT_0) {
        DWORD code = 1;
        GetExitCodeProcess(ph, &code);
        status = (int)code;
    } else
        TerminateJobObject((HANDLE)sb->job, 1);
    /* the cage gone, its pipes' ends with it: the readers end */
    if (r0)
        pthread_join(t[0], NULL);
    if (r1)
        pthread_join(t[1], NULL);
    if (fed)
        pthread_join(t[2], NULL);
    janas_sandbox_kill(sb);
    return status;
}
#else
static int collect(struct janas_sandbox *sb, const char *input, struct out *o)
{
    size_t in_n = input ? strlen(input) : 0, in_at = 0;
    if (!in_n) {
        close(sb->in);
        sb->in = -1;
    } else
        fcntl(sb->in, F_SETFL, fcntl(sb->in, F_GETFL) | O_NONBLOCK);
    o[0].fd = sb->out, o[1].fd = sb->err;
    int open_out = 1, open_err = 1, timed_out = 0;
    time_t end = time(NULL) + RUN_SECONDS;
    while (open_out || open_err) {
        struct pollfd p[3];
        int np = 0, io = -1, ie = -1, ii = -1;
        if (open_out)
            io = np, p[np++] = (struct pollfd){.fd = sb->out, .events = POLLIN};
        if (open_err)
            ie = np, p[np++] = (struct pollfd){.fd = sb->err, .events = POLLIN};
        if (sb->in >= 0)
            ii = np, p[np++] = (struct pollfd){.fd = sb->in, .events = POLLOUT};
        int left = (int)(end - time(NULL));
        if (left <= 0) {
            timed_out = 1;
            break;
        }
        int r = poll(p, (nfds_t)np, left * 1000);
        if (r < 0 && errno == EINTR)
            continue;
        if (r <= 0) {
            timed_out = r == 0;
            break;
        }
        if (io >= 0 && p[io].revents)
            open_out = drain(&o[0]);
        if (ie >= 0 && p[ie].revents)
            open_err = drain(&o[1]);
        if (ii >= 0 && p[ii].revents) {
            ssize_t w = (p[ii].revents & POLLOUT)
                            ? write(sb->in, input + in_at, in_n - in_at)
                            : -1;
            if (w > 0)
                in_at += (size_t)w;
            if (w < 0 ? errno != EAGAIN && errno != EINTR : in_at == in_n) {
                close(sb->in);
                sb->in = -1;
            }
        }
    }
    if (timed_out) {
        janas_sandbox_kill(sb);
        return -1;
    }
    int st = 0;
    while (waitpid(sb->pid, &st, 0) < 0 && errno == EINTR)
        ;
    sb->pid = 0;
    janas_sandbox_kill(sb); /* its pipes closed */
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}
#endif

int sk_run(const struct sk_skill *k, const char *script,
           const struct janas_json *args, const char *input,
           struct janas_buf *b)
{
    char why[400], path[4600], run[4400], runs[4200], t[700];
    if (!janas_sandbox_available(why, sizeof why)) {
        snprintf(t, sizeof t,
                 "The cage the scripts run in cannot be built here (%s): "
                 "tell the user to run the script.",
                 why);
        return fail(b, t);
    }
    snprintf(path, sizeof path, "%s/%s", k->dir, script);
#ifdef _WIN32
    for (char *c = path; *c; c++)
        if (*c == '/')
            *c = '\\';
#endif
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode) ||
        !sk_within(k->dir, path))
        return fail(b, "The skill has no such script.");
    char *argv[MAX_ARGS + 4];
    int ac = interpreter(script, argv, why, sizeof why);
    if (!ac)
        return fail(b, why);
    if (runs_dir(runs, sizeof runs) != 0)
        return fail(b, "There is no folder for the script to run in.");
    static int seq;
    snprintf(run, sizeof run, "%s/%lld-%d-%d", runs, (long long)time(NULL),
             (int)getpid(), ++seq);
#ifdef _WIN32
    for (char *c = run; *c; c++)
        if (*c == '/')
            *c = '\\';
#endif
    if (janas_mkdirs(run) != 0)
        return fail(b, "There is no folder for the script to run in.");

    argv[ac++] = path;
    for (const struct janas_json *e = args ? args->child : NULL;
         e && ac < MAX_ARGS + 3; e = e->next)
        argv[ac++] = (char *)janas_json_str(e);
    argv[ac] = NULL;
    const char *ro[] = {k->dir, NULL};
    struct janas_sandbox_cfg cfg = {.data_dir = run,
                                    .ro = ro,
                                    .mem_bytes = (uint64_t)2 << 30,
                                    .file_bytes = (uint64_t)512 << 20,
                                    .tmp_bytes = (uint64_t)256 << 20};
    struct janas_sandbox sb;
    if (janas_sandbox_spawn(&cfg, argv, environment(run), &sb, why,
                            sizeof why) != 0) {
        remove_tree(run, 0);
        snprintf(t, sizeof t, "The script could not be started: %s.", why);
        return fail(b, t);
    }
    struct out o[2] = {{.fd = -1}, {.fd = -1}};
    int status = collect(&sb, input, o);

    struct janas_buf r = {0};
    if (status < 0)
        janas_buf_printf(&r, "The script was stopped after %d seconds.",
                         RUN_SECONDS);
    else
        janas_buf_printf(&r, "The script ended with exit code %d.", status);
    const char *what[2] = {"output", "errors"};
    for (int i = 0; i < 2; i++) {
        if (!o[i].b.n)
            continue;
        janas_buf_printf(&r, "\n\nIts %s:\n%.*s", what[i], (int)o[i].b.n,
                         o[i].b.p);
        if (o[i].lost)
            janas_buf_printf(&r, "\n[... %zu bytes more]", o[i].lost);
        janas_buf_free(&o[i].b);
    }
    struct janas_buf w = {0};
    int count = 0;
    written(run, "", &w, &count, 0);
    if (count)
        janas_buf_printf(&r, "\n\nThe files it wrote, in %s:%.*s", run,
                         (int)w.n, w.p);
    else
        remove_tree(run, 0);
    janas_buf_free(&w);
    janas_mcps_text_result(b, r.p ? r.p : "", r.n, status != 0);
    janas_buf_free(&r);
    return 0;
}
