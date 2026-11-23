/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * auth.c - logging in to Brebus served beyond its own window (--serve):
 * the accounts of users.c, a session a cookie, the second factor asked of
 * everyone when Brebus listens beyond this computer.
 *
 * - A session is 32 random bytes in a cookie (HttpOnly, SameSite=Lax,
 *   Secure over HTTPS), kept here in memory: a restart logs everyone out.
 *   Lax, not Strict: a sign-in's provider sends the browser back to
 *   /brebus/oauth/callback, a GET from another site, which must find the
 *   session; what changes anything is a POST or a DELETE, which Lax does
 *   not carry from another site, and whose Origin is checked besides.
 *   It ends after a week unused, or thirty days in all, or at logout.
 * - Failed logins are counted by address and by name: after five, each
 *   next one waits twice as long as the last, up to an hour.
 * - Every login and failure is written to logins.log in the data folder,
 *   with the time and the address.
 * - A first login without a second factor, where one is required, sets
 *   one up: the secret shown once, kept only when a code from it is right,
 *   then ten recovery codes shown once.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <microhttpd.h>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include "brebus.h"

#define SESSIONS 256
#define IDLE (7 * 86400)
#define LIFE (30 * 86400)
#define PENDING_LIFE 600 /* a second factor being set up, in seconds */

#define PER_USER 16 /* sessions of one account at once */

struct session {
    char token[65];
    char user[64];
    int admin;
    time_t made, used;
};

struct pending { /* a password right, its second factor being set up */
    char token[65];
    char user[64];
    char secret[33];
    time_t made;
};

struct fails {
    char key[96]; /* "ip:..." or "user:..." */
    int n;
    time_t until;
    time_t last; /* the last failure */
};

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static struct session S[SESSIONS];
static struct pending P[32];
static struct fails F[512];

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
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

void brebus_client_ip(struct MHD_Connection *conn, char *out, size_t n)
{
    out[0] = 0;
    const union MHD_ConnectionInfo *ci =
        MHD_get_connection_info(conn, MHD_CONNECTION_INFO_CLIENT_ADDRESS);
    const struct sockaddr *sa = ci ? ci->client_addr : NULL;
    if (sa && sa->sa_family == AF_INET)
        inet_ntop(AF_INET, &((const struct sockaddr_in *)sa)->sin_addr, out,
                  (socklen_t)n);
    else if (sa && sa->sa_family == AF_INET6)
        inet_ntop(AF_INET6, &((const struct sockaddr_in6 *)sa)->sin6_addr, out,
                  (socklen_t)n);
    if (!out[0])
        snprintf(out, n, "?");
}

static void log_line(struct brebus *b, const char *ip, const char *user,
                     const char *what)
{
    char p[4200], when[32];
    snprintf(p, sizeof p, "%s/logins.log", b->data_dir);
    time_t t = time(NULL);
    struct tm tm;
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    strftime(when, sizeof when, "%Y-%m-%dT%H:%M:%SZ", &tm);
    FILE *f = fopen(p, "ab");
    if (f) {
        /* a name as given may be anything: its letters only */
        char u[64];
        size_t k = 0;
        for (const char *c = user ? user : ""; *c && k < sizeof u - 1; c++)
            u[k++] = *c > ' ' && *c < 127 ? *c : '?';
        u[k] = 0;
        /* and what happened, which may quote a name: one line */
        char w[160];
        size_t j = 0;
        for (const char *c = what; *c && j < sizeof w - 1; c++)
            w[j++] = *c >= ' ' && *c < 127 ? *c : '?';
        w[j] = 0;
        fprintf(f, "%s %s %s %s\n", when, ip, k ? u : "-", w);
        fclose(f);
    }
}

/* mu held: the entry for key, or NULL */
static struct fails *fail_find(const char *key)
{
    for (size_t i = 0; i < sizeof F / sizeof *F; i++)
        if (!strcmp(F[i].key, key))
            return &F[i];
    return NULL;
}

/* mu held: the entry for key, made when there is none. Room is made from
   what counts least: a free entry, then one whose failures are an hour
   old, then the address with the oldest failure - never a name's while
   an address's is there (filling the table from many addresses would
   otherwise wipe a name's count) */
static struct fails *fail_of(const char *key)
{
    struct fails *f = fail_find(key);
    if (f)
        return f;
    time_t now = time(NULL);
    struct fails *pick = NULL;
    for (size_t i = 0; i < sizeof F / sizeof *F && !pick; i++)
        if (!F[i].key[0] || now - F[i].last > 3600)
            pick = &F[i];
    struct fails *ip = NULL, *any = NULL;
    for (size_t i = 0; i < sizeof F / sizeof *F && !pick; i++) {
        if (!strncmp(F[i].key, "ip:", 3) && (!ip || F[i].last < ip->last))
            ip = &F[i];
        if (!any || F[i].last < any->last)
            any = &F[i];
    }
    if (!pick)
        pick = ip ? ip : any;
    memset(pick, 0, sizeof *pick);
    snprintf(pick->key, sizeof pick->key, "%.*s", (int)sizeof pick->key - 1,
             key);
    return pick;
}

/* an address as counted: IPv6 by its /64 (one machine has the whole
   prefix to itself) */
static void ip_key(const char *ip, char *out, size_t n)
{
    if (strchr(ip, ':') && strncmp(ip, "::ffff:", 7)) {
        int groups = 0;
        size_t k = 0;
        for (const char *c = ip; *c && groups < 4 && k + 1 < n; c++) {
            if (*c == ':' && (++groups == 4 || c[1] == ':'))
                break;
            out[k++] = *c;
        }
        snprintf(out + k, n - k, "::/64");
    } else {
        snprintf(out, n, "%s", ip);
    }
}

/* the names being tried now, and how many tries each */
#define BUSY 64
static struct {
    char key[96];
    int n;
} busy[BUSY];

/* seconds to wait before another try for this address and name: 0, or
   more (0: the try counted as one under way, ended by try_end) */
static long wait_for(const char *ip, const char *user)
{
    char k1[96], k2[96], ik[80];
    ip_key(ip, ik, sizeof ik);
    snprintf(k1, sizeof k1, "ip:%s", ik);
    snprintf(k2, sizeof k2, "user:%.60s", user ? user : "");
    pthread_mutex_lock(&mu);
    time_t now = time(NULL);
    long w = 0;
    /* looked up, not made: a try that does not fail takes no room */
    struct fails *a = fail_find(k1), *b = fail_find(k2);
    if (a && a->until > now)
        w = (long)(a->until - now);
    if (b && b->until > now && (long)(b->until - now) > w)
        w = (long)(b->until - now);
    /* tries of one name at once count as failures until they end: five
       sent together are five, not one each */
    if (!w) {
        int at = -1, room = -1;
        for (int i = 0; i < BUSY && at < 0; i++) {
            if (!strcmp(busy[i].key, k2))
                at = i;
            else if (!busy[i].n && room < 0)
                room = i;
        }
        int n = (b ? b->n : 0) + (at >= 0 ? busy[at].n : 0);
        if (n >= 5 || (at < 0 && room < 0)) {
            w = 1;
        } else {
            if (at < 0) {
                at = room;
                snprintf(busy[at].key, sizeof busy[at].key, "%s", k2);
            }
            busy[at].n++;
        }
    }
    pthread_mutex_unlock(&mu);
    return w;
}

/* a try let in by wait_for (it returned 0) is over */
static void try_end(const char *user)
{
    char k2[96];
    snprintf(k2, sizeof k2, "user:%.60s", user ? user : "");
    pthread_mutex_lock(&mu);
    for (int i = 0; i < BUSY; i++)
        if (busy[i].n && !strcmp(busy[i].key, k2)) {
            busy[i].n--;
            break;
        }
    pthread_mutex_unlock(&mu);
}

static void failed(const char *ip, const char *user)
{
    char keys[2][96], ik[80];
    ip_key(ip, ik, sizeof ik);
    snprintf(keys[0], sizeof keys[0], "ip:%s", ik);
    snprintf(keys[1], sizeof keys[1], "user:%.60s", user ? user : "");
    pthread_mutex_lock(&mu);
    time_t now = time(NULL);
    for (int i = 0; i < 2; i++) {
        struct fails *f = fail_of(keys[i]);
        f->n++;
        f->last = now;
        if (f->n >= 5) {
            long w = 1L << (f->n - 5 < 12 ? f->n - 5 : 12);
            f->until = now + (w < 3600 ? w : 3600);
        }
    }
    pthread_mutex_unlock(&mu);
}

static void succeeded(const char *ip, const char *user)
{
    char keys[2][96], ik[80];
    ip_key(ip, ik, sizeof ik);
    snprintf(keys[0], sizeof keys[0], "ip:%s", ik);
    snprintf(keys[1], sizeof keys[1], "user:%.60s", user);
    pthread_mutex_lock(&mu);
    for (int i = 0; i < 2; i++) {
        struct fails *f = fail_find(keys[i]);
        if (f)
            memset(f, 0, sizeof *f);
    }
    pthread_mutex_unlock(&mu);
}

/* a session made for the user: its token into out, 0, or -1 */
static int session_new(const struct brebus_user *u, char out[65])
{
    if (token(out) != 0)
        return -1;
    pthread_mutex_lock(&mu);
    time_t now = time(NULL);
    /* a free or ended one; else, when the user has PER_USER already, the
       user's own least used (their many logins never end the others'
       sessions); else the least used of all */
    struct session *s = NULL, *mine = NULL, *any = &S[0];
    int n_mine = 0;
    for (int i = 0; i < SESSIONS && !s; i++) {
        if (!S[i].token[0] || now - S[i].used > IDLE ||
            now - S[i].made > LIFE) {
            s = &S[i];
            break;
        }
        if (!strcmp(S[i].user, u->name)) {
            n_mine++;
            if (!mine || S[i].used < mine->used)
                mine = &S[i];
        }
        if (S[i].used < any->used)
            any = &S[i];
    }
    if (!s)
        s = n_mine >= PER_USER || (mine && n_mine * 8 >= SESSIONS) ? mine : any;
    memset(s, 0, sizeof *s);
    snprintf(s->token, sizeof s->token, "%s", out);
    snprintf(s->user, sizeof s->user, "%s", u->name);
    s->admin = u->admin;
    s->made = s->used = now;
    pthread_mutex_unlock(&mu);
    return 0;
}

/* the session's token from the Cookie header, read here (the
   libmicrohttpd carried is built without its cookies): into out, or "" */
static const char *cookie(struct MHD_Connection *conn, char out[65])
{
    out[0] = 0;
    const char *h =
        MHD_lookup_connection_value(conn, MHD_HEADER_KIND, "Cookie");
    for (const char *p = h; p && *p;) {
        while (*p == ' ' || *p == ';')
            p++;
        const char *end = strchr(p, ';');
        size_t n = end ? (size_t)(end - p) : strlen(p);
        if (n == 15 + 64 && !strncmp(p, "brebus_session=", 15)) {
            memcpy(out, p + 15, 64);
            out[64] = 0;
            break;
        }
        p += n;
    }
    return out;
}

int brebus_auth_user(struct MHD_Connection *conn, char *user, size_t un,
                     int *admin)
{
    char tb[65];
    const char *t = cookie(conn, tb);
    if (strlen(t) != 64)
        return 0;
    pthread_mutex_lock(&mu);
    time_t now = time(NULL);
    int ok = 0;
    for (int i = 0; i < SESSIONS && !ok; i++) {
        struct session *s = &S[i];
        if (!s->token[0])
            continue;
        uint8_t d = 0;
        for (int k = 0; k < 64; k++)
            d |= (uint8_t)(s->token[k] ^ t[k]);
        if (d)
            continue;
        if (now - s->used > IDLE || now - s->made > LIFE) {
            memset(s, 0, sizeof *s);
            break;
        }
        s->used = now;
        if (user)
            snprintf(user, un, "%s", s->user);
        if (admin)
            *admin = s->admin;
        ok = 1;
    }
    pthread_mutex_unlock(&mu);
    return ok;
}

/* the cookie that carries a session, or the one that ends it */
void brebus_auth_cookie(struct brebus *b, const char *tok, char *out, size_t n)
{
    snprintf(out, n,
             "brebus_session=%s; Path=/; HttpOnly; SameSite=Lax%s; "
             "Max-Age=%d",
             tok ? tok : "", b->tls ? "; Secure" : "", tok ? LIFE : 0);
}

/* the last step of time whose code each name used: a code is good once
   (one seen over a shoulder or a network is spent) */
static struct {
    char user[64];
    int step;
} steps[256];

/* mu held not: 1 when step is past the name's last, kept now */
static int fresh_step(const char *user, int step)
{
    pthread_mutex_lock(&mu);
    int at = -1, ok = 1;
    for (int i = 0; i < 256 && at < 0; i++)
        if (!strcmp(steps[i].user, user))
            at = i;
    if (at >= 0 && step <= steps[at].step)
        ok = 0;
    if (ok) {
        if (at < 0) /* a free entry, else the oldest step */
            for (int i = 0; i < 256; i++)
                if (at < 0 || !steps[i].user[0] ||
                    (steps[at].user[0] && steps[i].step < steps[at].step))
                    at = i;
        snprintf(steps[at].user, sizeof steps[at].user, "%s", user);
        steps[at].step = step;
    }
    pthread_mutex_unlock(&mu);
    return ok;
}

static int totp_fresh(const char *user, int step)
{
    return step && fresh_step(user, step);
}

static double mono_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* an address of this computer's alone */
static int loopback_ip(const char *ip)
{
    return !strncmp(ip, "127.", 4) || !strcmp(ip, "::1") ||
           !strncmp(ip, "::ffff:127.", 11);
}

int brebus_auth_login(struct brebus *b, struct MHD_Connection *conn,
                      const char *body, size_t n, struct janas_buf *out,
                      char *set_cookie, size_t cn)
{
    char err[160], ip[64];
    brebus_client_ip(conn, ip, sizeof ip);
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *name = o ? janas_json_str(janas_json_get(o, "name")) : NULL;
    const char *pw = o ? janas_json_str(janas_json_get(o, "password")) : NULL;
    const char *code = o ? janas_json_str(janas_json_get(o, "code")) : NULL;
    int status = 200;
    long w = wait_for(ip, name);
    struct brebus_user u;
    double began = mono_s();
    set_cookie[0] = 0;
    if (w > 0) {
        janas_buf_printf(out,
                         "{\"error\": \"too many tries: wait %ld seconds\", "
                         "\"wait\": %ld}",
                         w, w);
        status = 429;
    } else if (!name || !pw ||
               !(b->sys_users ? brebus_user_system(b, name, pw, &u)
                              : brebus_user_check(b, name, pw, &u))) {
        /* every failure as long as PAM's slowest: how long it took says
           nothing of whether the password was right (an account expired,
           of another group) */
        if (b->sys_users)
            while (mono_s() - began < 2.5)
                usleep(50000);
        failed(ip, name);
        log_line(b, ip, name, "failed: name or password");
        janas_buf_puts(out, "{\"error\": \"wrong name or password\"}");
        status = 401;
    } else if (u.totp_on && (!code || !*code)) {
        janas_buf_puts(out, "{\"need\": \"code\"}");
    } else if (u.totp_on &&
               !totp_fresh(u.name,
                           brebus_totp_check(u.totp, code, time(NULL))) &&
               !brebus_user_recovery_use(b, u.name, code)) {
        failed(ip, name);
        log_line(b, ip, name, "failed: second factor");
        janas_buf_puts(out, "{\"error\": \"wrong code\"}");
        status = 401;
    } else if (!u.totp_on && b->need_2fa && !loopback_ip(ip)) {
        /* from afar a password alone does not choose the second factor:
           whoever had stolen it would choose it */
        log_line(b, ip, u.name, "password right: no second factor yet");
        janas_buf_puts(out, "{\"error\": \"no second factor set up yet: "
                            "set it up on this computer, in a browser at "
                            "127.0.0.1, or ask the administrator (brebus "
                            "--enroll NAME)\"}");
        status = 403;
    } else if (!u.totp_on && b->need_2fa) {
        /* the second factor set up now, from this computer: its secret
           shown once */
        pthread_mutex_lock(&mu);
        struct pending *p = &P[0];
        for (size_t i = 0; i < sizeof P / sizeof *P; i++)
            if (!P[i].token[0] || time(NULL) - P[i].made > PENDING_LIFE ||
                P[i].made < p->made) {
                p = &P[i];
                if (!P[i].token[0])
                    break;
            }
        memset(p, 0, sizeof *p);
        int ok = token(p->token) == 0 && brebus_totp_secret(p->secret) == 0;
        snprintf(p->user, sizeof p->user, "%s", u.name);
        p->made = time(NULL);
        struct pending copy = *p;
        if (!ok)
            memset(p, 0, sizeof *p);
        pthread_mutex_unlock(&mu);
        if (!ok) {
            janas_buf_puts(out, "{\"error\": \"no randomness\"}");
            status = 500;
        } else {
            log_line(b, ip, u.name, "password right: second factor set up");
            janas_buf_puts(out, "{\"setup\": ");
            str(out, copy.token);
            janas_buf_puts(out, ", \"secret\": ");
            str(out, copy.secret);
            char uri[300];
            snprintf(uri, sizeof uri,
                     "otpauth://totp/Brebus:%s?secret=%s&issuer=Brebus", u.name,
                     copy.secret);
            janas_buf_puts(out, ", \"uri\": ");
            str(out, uri);
            janas_buf_puts(out, "}");
        }
    } else {
        char tok[65];
        if (session_new(&u, tok) != 0) {
            janas_buf_puts(out, "{\"error\": \"no randomness\"}");
            status = 500;
        } else {
            succeeded(ip, u.name);
            log_line(b, ip, u.name, "logged in");
            brebus_auth_cookie(b, tok, set_cookie, cn);
            janas_buf_puts(out, "{\"user\": ");
            str(out, u.name);
            janas_buf_printf(out, ", \"admin\": %s}",
                             u.admin ? "true" : "false");
        }
    }
    if (w == 0)
        try_end(name);
    janas_json_free(d);
    return status;
}

int brebus_auth_setup(struct brebus *b, struct MHD_Connection *conn,
                      const char *body, size_t n, struct janas_buf *out,
                      char *set_cookie, size_t cn)
{
    char err[160], ip[64];
    brebus_client_ip(conn, ip, sizeof ip);
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *tk = o ? janas_json_str(janas_json_get(o, "setup")) : NULL;
    const char *code = o ? janas_json_str(janas_json_get(o, "code")) : NULL;
    set_cookie[0] = 0;
    pthread_mutex_lock(&mu);
    struct pending p = {0};
    for (size_t i = 0; tk && i < sizeof P / sizeof *P; i++)
        if (P[i].token[0] && !strcmp(P[i].token, tk) &&
            time(NULL) - P[i].made <= PENDING_LIFE)
            p = P[i];
    pthread_mutex_unlock(&mu);
    int status = 200;
    long w = wait_for(ip, p.user);
    char codes[BREBUS_RECOVERY][12];
    struct brebus_user u = {0};
    if (w > 0) {
        janas_buf_printf(out,
                         "{\"error\": \"too many tries: wait %ld "
                         "seconds\"}",
                         w);
        status = 429;
    } else if (!p.token[0]) {
        janas_buf_puts(out, "{\"error\": \"log in again: this setup has "
                            "expired\"}");
        status = 401;
    } else if (!brebus_totp_check(p.secret, code, time(NULL))) {
        failed(ip, p.user);
        janas_buf_puts(out, "{\"error\": \"wrong code: check the "
                            "authenticator's clock and the secret\"}");
        status = 401;
    } else if (brebus_user_totp_on(b, p.user, p.secret, codes) != 0) {
        janas_buf_puts(out, "{\"error\": \"users.json could not be "
                            "written\"}");
        status = 500;
    } else {
        pthread_mutex_lock(&mu);
        for (size_t i = 0; i < sizeof P / sizeof *P; i++)
            if (!strcmp(P[i].token, p.token))
                memset(&P[i], 0, sizeof P[i]);
        pthread_mutex_unlock(&mu);
        snprintf(u.name, sizeof u.name, "%s", p.user);
        pthread_mutex_lock(&b->mu);
        for (int i = 0; i < b->n_users; i++)
            if (!strcmp(b->users[i].name, p.user))
                u.admin = b->users[i].admin;
        pthread_mutex_unlock(&b->mu);
        char tok[65];
        if (session_new(&u, tok) != 0) {
            janas_buf_puts(out, "{\"error\": \"no randomness\"}");
            status = 500;
        } else {
            succeeded(ip, u.name);
            log_line(b, ip, u.name, "second factor set up, logged in");
            brebus_auth_cookie(b, tok, set_cookie, cn);
            janas_buf_puts(out, "{\"user\": ");
            str(out, u.name);
            janas_buf_puts(out, ", \"recovery\": [");
            for (int k = 0; k < BREBUS_RECOVERY; k++) {
                janas_buf_puts(out, k ? ", " : "");
                str(out, codes[k]);
            }
            janas_buf_puts(out, "]}");
        }
    }
    if (w == 0)
        try_end(p.user);
    janas_json_free(d);
    return status;
}

void brebus_auth_logout(struct brebus *b, struct MHD_Connection *conn)
{
    char tb[65];
    const char *t = cookie(conn, tb);
    char ip[64], user[64] = "";
    brebus_client_ip(conn, ip, sizeof ip);
    pthread_mutex_lock(&mu);
    for (int i = 0; *t && i < SESSIONS; i++)
        if (S[i].token[0] && !strcmp(S[i].token, t)) {
            snprintf(user, sizeof user, "%s", S[i].user);
            memset(&S[i], 0, sizeof S[i]);
        }
    pthread_mutex_unlock(&mu);
    if (user[0])
        log_line(b, ip, user, "logged out");
}

/* every session of user ended but the one of keep (NULL: all) */
static void end_sessions(const char *user, const char *keep)
{
    pthread_mutex_lock(&mu);
    for (int i = 0; i < SESSIONS; i++)
        if (S[i].token[0] && !strcmp(S[i].user, user) &&
            (!keep || strcmp(S[i].token, keep)))
            memset(&S[i], 0, sizeof S[i]);
    pthread_mutex_unlock(&mu);
}

int brebus_auth_password(struct brebus *b, struct MHD_Connection *conn,
                         const char *user, const char *body, size_t n,
                         struct janas_buf *out)
{
    char err[200], ip[64], tb[65];
    brebus_client_ip(conn, ip, sizeof ip);
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *old = o ? janas_json_str(janas_json_get(o, "old")) : NULL;
    const char *nw = o ? janas_json_str(janas_json_get(o, "new")) : NULL;
    struct brebus_user u;
    int status = 200;
    long w = wait_for(ip, user);
    if (b->sys_users) {
        janas_buf_puts(out, "{\"error\": \"your password is the system's: "
                            "change it there (passwd)\"}");
        status = 400;
    } else if (w > 0) {
        janas_buf_printf(out,
                         "{\"error\": \"too many tries: wait %ld "
                         "seconds\"}",
                         w);
        status = 429;
    } else if (!old || !nw || !brebus_user_check(b, user, old, &u)) {
        failed(ip, user);
        log_line(b, ip, user, "failed: password change");
        janas_buf_puts(out, "{\"error\": \"the password now is not that\"}");
        status = 401;
    } else if (brebus_user_add(b, user, nw, u.admin, err, sizeof err) != 0) {
        janas_buf_puts(out, "{\"error\": ");
        str(out, err);
        janas_buf_puts(out, "}");
        status = 400;
    } else {
        /* the other sessions end: whoever had the old one is out */
        end_sessions(user, cookie(conn, tb));
        log_line(b, ip, user, "password changed, other sessions ended");
        janas_buf_puts(out, "{}");
    }
    if (w == 0)
        try_end(user);
    janas_json_free(d);
    return status;
}

int brebus_auth_end_all(struct brebus *b, struct MHD_Connection *conn,
                        const char *user, int admin, const char *body, size_t n,
                        struct janas_buf *out)
{
    char err[160], ip[64], tb[65];
    brebus_client_ip(conn, ip, sizeof ip);
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const char *who =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "user")) : NULL;
    int status = 200;
    if (who && strcmp(who, user) && !admin) {
        janas_buf_puts(out, "{\"error\": \"an administrator's to do\"}");
        status = 403;
    } else {
        const char *t = who && strlen(who) < 64 ? who : user;
        /* one's own: all but this one; another's: all */
        end_sessions(t, strcmp(t, user) ? NULL : cookie(conn, tb));
        char what[128];
        snprintf(what, sizeof what, "sessions of %.60s ended", t);
        log_line(b, ip, user, what);
        janas_buf_puts(out, "{}");
    }
    janas_json_free(d);
    return status;
}

void brebus_auth_logins(struct brebus *b, const char *user, int admin,
                        struct janas_buf *out)
{
    char p[4200];
    snprintf(p, sizeof p, "%s/logins.log", b->data_dir);
    FILE *f = fopen(p, "rb");
    /* the last 20 lines of this user's (an administrator's: anyone's) */
    char lines[20][300]; /* each call its own */
    int k = 0, n = 0;
    char l[300];
    while (f && fgets(l, sizeof l, f)) {
        char when[40], ip[64], who[64];
        if (sscanf(l, "%39s %63s %63s", when, ip, who) != 3)
            continue;
        if (!admin && strcmp(who, user))
            continue;
        l[strcspn(l, "\n")] = 0;
        snprintf(lines[k], sizeof lines[k], "%s", l);
        k = (k + 1) % 20;
        n++;
    }
    if (f)
        fclose(f);
    janas_buf_puts(out, "{\"logins\": [");
    int m = n < 20 ? n : 20;
    for (int i = 0; i < m; i++) {
        const char *x = lines[(k - 1 - i + 40) % 20];
        janas_buf_puts(out, i ? ", " : "");
        str(out, x);
    }
    janas_buf_puts(out, "]}");
}
