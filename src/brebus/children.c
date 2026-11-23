/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * children.c - Brebus served with users who are not administrators: each
 * has a Brebus of their own (brebus --child), started by this one at their
 * first request and given every request of theirs (proxy.c), the login
 * and the second factor staying here.
 *
 * A child listens on 127.0.0.1 alone, at this one's port plus 100 and its
 * place, with the user's own folders: data_dir/users/<name>/ holds its
 * data, config and cache (XDG_*), and files/, the only place its projects
 * may be (BREBUS_ROOT). It answers only requests that carry its token
 * (X-Brebus-Child), random, given in its environment: another user of
 * this machine reaching 127.0.0.1 cannot use it. The models are this
 * one's, shared: the child passes /v1, /api and the models' server back
 * here, with the same token.
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
#ifndef _WIN32
#include <pwd.h>
#endif

#include "brebus.h"
#include "common/catalog.h"
#include "common/mcp_http.h"

#define CHILDREN 32

static struct child {
    char user[64];
    char token[65];
    unsigned short port;
    struct janas_proc proc;
    int on;
    int life;     /* a user of the system's: its input, closed to end it */
    unsigned uid; /* whom its port must be listened on by */
    time_t used;  /* its user's last request */
} kids[CHILDREN];

/* the port listened on by uid alone, on this machine (Linux: the kernel's
   tables say whose a socket is); elsewhere 1. Another user of the machine
   taking a child's port first would get its token and its requests */
static int port_owned(unsigned short port, unsigned uid)
{
#ifdef __linux__
    int found = 0;
    static const char *const t[] = {"/proc/net/tcp", "/proc/net/tcp6"};
    for (int k = 0; k < 2; k++) {
        FILE *f = fopen(t[k], "re");
        char line[512];
        if (!f)
            continue;
        while (fgets(line, sizeof line, f)) {
            char local[80];
            unsigned st, u;
            /* sl local remote st tx:rx tr:when retrnsmt uid */
            if (sscanf(line, " %*s %79s %*s %x %*s %*s %*s %u", local, &st,
                       &u) != 3 ||
                st != 0x0A)
                continue;
            const char *c = strrchr(local, ':');
            if (!c || strtoul(c + 1, NULL, 16) != port)
                continue;
            if (u != uid) {
                fclose(f);
                return 0;
            }
            found = 1;
        }
        fclose(f);
    }
    return found;
#else
    (void)port, (void)uid;
    return 1;
#endif
}

int brebus_port_mine(unsigned short port)
{
#ifdef __linux__
    return port_owned(port, (unsigned)getuid());
#else
    (void)port;
    return 1;
#endif
}

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

/* the tickets of the users of the system logged in, for the helper's
   spawn (one each, spent by it) */
static struct ticket {
    char user[64], t[65];
} tickets[CHILDREN];

void brebus_child_ticket(const char *user, const char *t)
{
    pthread_mutex_lock(&mu);
    struct ticket *k = &tickets[0];
    for (int i = 0; i < CHILDREN; i++)
        if (!strcmp(tickets[i].user, user) || !tickets[i].user[0]) {
            k = &tickets[i];
            if (!strcmp(tickets[i].user, user))
                break;
        }
    snprintf(k->user, sizeof k->user, "%s", user);
    snprintf(k->t, sizeof k->t, "%s", t);
    pthread_mutex_unlock(&mu);
}

static int token(char out[65])
{
    uint8_t r[32];
    if (brebus_random(r, sizeof r) != 0)
        return -1;
    static const char d[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++)
        out[2 * i] = d[r[i] >> 4], out[2 * i + 1] = d[r[i] & 15];
    out[64] = 0;
    return 0;
}

/* the child answers: ready */
static int answers(unsigned short port, const char *tok, unsigned uid)
{
    if (!port_owned(port, uid))
        return 0;
    char url[64], err[128], hv[128];
    snprintf(url, sizeof url, "http://127.0.0.1:%u/brebus/workspace", port);
    snprintf(hv, sizeof hv, "X-Brebus-Child: %s", tok);
    const char *hdr[] = {hv};
    struct janas_url u;
    struct janas_http h;
    if (janas_url_parse(url, &u, err, sizeof err) != 0 ||
        janas_http_request(&h, &u, "GET", hdr, 1, NULL, 0, 2000, NULL, err,
                           sizeof err) != 0)
        return 0;
    int ok = h.head.status == 200;
    janas_http_close(&h);
    return ok;
}

/* mu held: a user of the system's own Brebus, started as them by
   janas-brebus-helper in their home, their own folders (no root to keep
   them in: the system's permissions do); its input a pipe of ours, closed
   to end it, as it runs as another user, beyond our signals */
static int start_as_user(struct brebus *b, struct child *c, int at,
                         const char *user, unsigned short *port, char tok[65],
                         char *why, size_t wn)
{
    /* the ticket of the user's login: spent by this spawn */
    char tk[66] = "";
    for (int i = 0; i < CHILDREN && !tk[0]; i++)
        if (!strcmp(tickets[i].user, user) && tickets[i].t[0]) {
            snprintf(tk, sizeof tk, "%s\n", tickets[i].t);
            memset(&tickets[i], 0, sizeof tickets[i]);
        }
    if (!tk[0]) {
        pthread_mutex_unlock(&mu);
        snprintf(why, wn, "log in again: your Brebus starts with a login");
        return -1;
    }
    if (token(c->token) != 0) {
        pthread_mutex_unlock(&mu);
        snprintf(why, wn, "no randomness");
        return -1;
    }
    snprintf(c->user, sizeof c->user, "%s", user);
    c->port = (unsigned short)(b->port + 100 + at);
#ifndef _WIN32
    struct passwd *pw = getpwnam(user);
    c->uid = pw ? (unsigned)pw->pw_uid : (unsigned)-1;
#endif
    char exe[4300], ps[8], fp[8];
    snprintf(exe, sizeof exe, "%s/janas-brebus-helper", b->bin_dir);
    snprintf(ps, sizeof ps, "%u", c->port);
    snprintf(fp, sizeof fp, "%u", b->port);
    static char e[5][4500];
    snprintf(e[0], sizeof e[0], "JANAS_MODELS=%s", b->models_dir);
    snprintf(e[1], sizeof e[1], "BREBUS_CHILD_TOKEN=%s", c->token);
    snprintf(e[2], sizeof e[2], "BREBUS_FRONT=%s", fp);
    snprintf(e[3], sizeof e[3], "BREBUS_FRONT_TLS=%d", b->tls);
    snprintf(e[4], sizeof e[4], "BREBUS_FRONT_CERT=%s", b->cert);
    const char *env[] = {e[0], e[1], e[2], e[3], e[4], NULL};
    const char *argv[] = {exe, "spawn", user, ps, NULL};
    int life[2];
    char err[300], log[4400];
    snprintf(log, sizeof log, "%s/users-%s.log", b->data_dir, user);
    int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    int rc = janas_pipe(life);
    if (rc == 0)
        rc = janas_proc_start(&c->proc, argv, env, NULL, life[0], fd, fd,
                              JANAS_PROC_GROUP | BREBUS_NO_CONSOLE, err,
                              sizeof err);
    if (fd >= 0)
        close(fd);
    if (rc != 0) {
        c->user[0] = 0;
        pthread_mutex_unlock(&mu);
        snprintf(why, wn, "%s's Brebus did not start: %s", user, err);
        return -1;
    }
    close(life[0]);
    if (write(life[1], tk, strlen(tk)) != (ssize_t)strlen(tk)) {
        /* the helper went: noticed below */
    }
    c->life = life[1];
    c->on = 1;
    unsigned short p = c->port;
    char t[65];
    snprintf(t, sizeof t, "%s", c->token);
    pthread_mutex_unlock(&mu);
    for (int i = 0; i < 150 && !answers(p, t, c->uid); i++) {
        int code;
        pthread_mutex_lock(&mu);
        int gone = c->on && janas_proc_wait(&c->proc, 0, &code) == 1;
        if (gone)
            c->on = 0;
        pthread_mutex_unlock(&mu);
        if (gone) {
            snprintf(why, wn, "%s's Brebus ended (%d): see %s", user, code,
                     log);
            return -1;
        }
        usleep(100000);
    }
    *port = p;
    snprintf(tok, 65, "%s", t);
    return 0;
}

int brebus_child_for(struct brebus *b, const char *user, unsigned short *port,
                     char tok[65], char *why, size_t wn)
{
    pthread_mutex_lock(&mu);
    struct child *c = NULL;
    int at = -1;
    for (int i = 0; i < CHILDREN && !c; i++) {
        int code;
        if (kids[i].on && janas_proc_wait(&kids[i].proc, 0, &code) == 1) {
            kids[i].on = 0; /* it ended: started again below */
            if (kids[i].life > 0)
                close(kids[i].life), kids[i].life = 0;
        }
        if (!strcmp(kids[i].user, user))
            c = &kids[i], at = i;
    }
    for (int i = 0; i < CHILDREN && !c; i++)
        if (!kids[i].user[0] || !kids[i].on) /* free, or its child ended */
            c = &kids[i], at = i;
    if (!c) { /* the one unused longest, ten minutes at least, ended */
        time_t now = time(NULL);
        for (int i = 0; i < CHILDREN; i++)
            if (now - kids[i].used >= 600 && (!c || kids[i].used < c->used))
                c = &kids[i], at = i;
        if (c) {
            if (c->life > 0)
                close(c->life), c->life = 0;
            janas_proc_kill(&c->proc, 0);
            int code;
            if (janas_proc_wait(&c->proc, 5000, &code) != 1) {
                janas_proc_kill(&c->proc, 1);
                janas_proc_wait(&c->proc, 2000, &code);
            }
            c->on = 0;
        }
    }
    if (c && strcmp(c->user, user))
        c->user[0] = 0; /* another's slot, theirs no more */
    if (c)
        c->used = time(NULL);
    if (!c) {
        pthread_mutex_unlock(&mu);
        snprintf(why, wn, "more users at once than this Brebus serves (%d)",
                 CHILDREN);
        return -1;
    }
    if (c->on && !port_owned(c->port, c->uid)) {
        pthread_mutex_unlock(&mu);
        snprintf(why, wn, "the port of %s's Brebus is someone else's", user);
        return -1;
    }
    if (c->on) {
        *port = c->port;
        snprintf(tok, 65, "%s", c->token);
        pthread_mutex_unlock(&mu);
        return 0;
    }
    if (b->sys_users)
        return start_as_user(b, c, at, user, port, tok, why, wn);
    /* the user's folders, and the child started on them */
    char base[4300], data[4400], cfg[4400], cache[4400], files[4400];
    snprintf(base, sizeof base, "%s/users/%s", b->data_dir, user);
    snprintf(data, sizeof data, "%s/data", base);
    snprintf(cfg, sizeof cfg, "%s/config", base);
    snprintf(cache, sizeof cache, "%s/cache", base);
    snprintf(files, sizeof files, "%s/files", base);
    if (janas_mkdirs(data) || janas_mkdirs(cfg) || janas_mkdirs(cache) ||
        janas_mkdirs(files) || token(c->token) != 0) {
        pthread_mutex_unlock(&mu);
        snprintf(why, wn, "no folder for %s's Brebus", user);
        return -1;
    }
    snprintf(c->user, sizeof c->user, "%s", user);
    c->port = (unsigned short)(b->port + 100 + at);
#ifndef _WIN32
    c->uid = (unsigned)getuid();
#endif
    char exe[4200], ps[8], fp[8];
    snprintf(exe, sizeof exe, "%s/brebus%s", b->bin_dir, b->suffix);
    snprintf(ps, sizeof ps, "%u", c->port);
    snprintf(fp, sizeof fp, "%u", b->port);
    static char e[11][4500];
    snprintf(e[0], sizeof e[0], "XDG_DATA_HOME=%s", data);
    snprintf(e[1], sizeof e[1], "XDG_CONFIG_HOME=%s", cfg);
    snprintf(e[2], sizeof e[2], "XDG_CACHE_HOME=%s", cache);
    snprintf(e[3], sizeof e[3], "JANAS_MODELS=%s", b->models_dir);
    snprintf(e[4], sizeof e[4], "BREBUS_CHILD_TOKEN=%s", c->token);
    snprintf(e[5], sizeof e[5], "BREBUS_FRONT=%s", fp);
    snprintf(e[6], sizeof e[6], "BREBUS_ROOT=%s", files);
    snprintf(e[7], sizeof e[7], "BREBUS_FRONT_TLS=%d", b->tls);
    snprintf(e[8], sizeof e[8], "BREBUS_FRONT_CERT=%s", b->cert);
    /* the Internet's addresses alone, and its front: what runs as Brebus's
       own user reaches no server of this machine or its network for a
       user (connectors, sign-ins, attached addresses) */
    snprintf(e[9], sizeof e[9], "JANAS_NET_PUBLIC_ONLY=1");
    snprintf(e[10], sizeof e[10], "JANAS_NET_ALLOW=127.0.0.1:%u", b->port);
    const char *env[] = {e[0], e[1], e[2], e[3], e[4],  e[5],
                         e[6], e[7], e[8], e[9], e[10], NULL};
    const char *argv[] = {exe, "--child", "--port", ps, NULL};
    char log[4500], err[300];
    snprintf(log, sizeof log, "%s/brebus.log", base);
    int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    int rc =
        janas_proc_start(&c->proc, argv, env, NULL, -1, fd, fd,
                         JANAS_PROC_GROUP | BREBUS_NO_CONSOLE, err, sizeof err);
    if (fd >= 0)
        close(fd);
    if (rc != 0) {
        c->user[0] = 0;
        pthread_mutex_unlock(&mu);
        snprintf(why, wn, "%s's Brebus did not start: %s", user, err);
        return -1;
    }
    c->on = 1;
    unsigned short p = c->port;
    char t[65];
    snprintf(t, sizeof t, "%s", c->token);
    pthread_mutex_unlock(&mu);
    for (int i = 0; i < 150 && !answers(p, t, c->uid); i++)
        usleep(100000);
    *port = p;
    snprintf(tok, 65, "%s", t);
    return 0;
}

int brebus_child_token_ok(const char *tok)
{
    if (!tok || strlen(tok) != 64)
        return 0;
    pthread_mutex_lock(&mu);
    int ok = 0;
    for (int i = 0; i < CHILDREN && !ok; i++) {
        if (!kids[i].on)
            continue;
        uint8_t d = 0;
        for (int k = 0; k < 64; k++)
            d |= (uint8_t)(kids[i].token[k] ^ tok[k]);
        ok = !d;
    }
    pthread_mutex_unlock(&mu);
    return ok;
}

void brebus_children_stop(void)
{
    pthread_mutex_lock(&mu);
    for (int i = 0; i < CHILDREN; i++)
        if (kids[i].on) {
            /* its input closed: a user of the system's ends by itself */
            if (kids[i].life > 0)
                close(kids[i].life), kids[i].life = 0;
            janas_proc_kill(&kids[i].proc, 0);
            int code;
            if (janas_proc_wait(&kids[i].proc, 5000, &code) != 1) {
                janas_proc_kill(&kids[i].proc, 1);
                janas_proc_wait(&kids[i].proc, -1, &code);
            }
            kids[i].on = 0;
        }
    pthread_mutex_unlock(&mu);
}

int brebus_confined(const struct brebus *b, const char *path)
{
    if (!b->root[0])
        return 1; /* not a user's Brebus: anywhere, as ever */
    if (!path || !*path || strstr(path, ".."))
        return 0;
    /* the path, or the nearest folder of it there is (a new one) */
    char p[4096], real[4096], r[4096], d[4096];
    snprintf(p, sizeof p, "%s", path);
    while (!janas_path_real(p, real)) {
        char *sl = strrchr(p, '/');
#ifdef _WIN32
        char *bs = strrchr(p, '\\');
        if (!sl || (bs && bs > sl))
            sl = bs;
#endif
        if (!sl || sl == p)
            return 0;
        *sl = 0;
    }
    if (janas_path_real(b->root, r) &&
        (!strcmp(real, r) || janas_path_in(real, r)))
        return 1;
    /* its own data: the free chats, the chats' folders */
    return janas_path_real(b->data_dir, d) &&
           (!strcmp(real, d) || janas_path_in(real, d));
}

void brebus_dirs(const struct brebus *b, const char *path, int files,
                 struct janas_buf *out)
{
    char real[4096], start[4300];
    /* where to start: the one asked, else the user's folder (a user's
       Brebus) or the home */
    const char *h = getenv("HOME");
    snprintf(start, sizeof start, "%s",
             path && *path ? path
             : b->root[0]  ? b->root
             : h           ? h
                           : "/");
    if (!janas_path_real(start, real) || !brebus_confined(b, real)) {
        janas_buf_puts(out, "{\"error\": \"not a folder you may see\"}");
        return;
    }
    DIR *d = opendir(real);
    if (!d) {
        janas_buf_puts(out, "{\"error\": \"not a folder\"}");
        return;
    }
    static char names[2][512][256];
    int nd = 0, nf = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.' || strlen(e->d_name) >= 256)
            continue; /* hidden ones, and . and .. */
        char p[4400];
        struct stat st;
        snprintf(p, sizeof p, "%s/%s", real, e->d_name);
        if (stat(p, &st) != 0)
            continue;
        if (S_ISDIR(st.st_mode) && nd < 512)
            snprintf(names[0][nd++], 256, "%s", e->d_name);
        else if (files && S_ISREG(st.st_mode) && nf < 512)
            snprintf(names[1][nf++], 256, "%s", e->d_name);
    }
    closedir(d);
    qsort(names[0], (size_t)nd, 256,
          (int (*)(const void *, const void *))strcmp);
    qsort(names[1], (size_t)nf, 256,
          (int (*)(const void *, const void *))strcmp);
    /* the folder above, when it may be seen too */
    char up[4096] = "";
    snprintf(up, sizeof up, "%s", real);
    char *sl = strrchr(up, '/');
    if (sl && sl != up)
        *sl = 0;
    else if (sl)
        sl[1] = 0;
    if (!strcmp(up, real) || !brebus_confined(b, up))
        up[0] = 0;
    janas_buf_puts(out, "{\"path\": ");
    janas_json_write_str(out, real, strlen(real));
    janas_buf_puts(out, ", \"up\": ");
    janas_json_write_str(out, up, strlen(up));
    for (int k = 0; k < 2; k++) {
        janas_buf_puts(out, k ? "], \"files\": [" : ", \"dirs\": [");
        int m = k ? nf : nd;
        for (int i = 0; i < m; i++) {
            janas_buf_puts(out, i ? ", " : "");
            janas_json_write_str(out, names[k][i], strlen(names[k][i]));
        }
    }
    janas_buf_puts(out, "]}");
}
