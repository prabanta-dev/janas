/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * Brebus's updates (update.h): a thread that asks GitHub for the latest
 * release at the start and once a day, downloads a newer one, checks it,
 * and installs it as the setting says - "auto" (the default): as Brebus
 * ends, or, with no window (--background, --serve), once nobody has used
 * it for a while; "notify": only told in the page, installed when asked;
 * "off": not even asked for. Its state is in data_dir/update/update.json.
 *
 * BREBUS_UPDATE_URL takes the release from another address (tests: a
 * local server, http://127.0.0.1 allowed there only), BREBUS_UPDATE_KEY
 * checks its signature with another public key, BREBUS_UPDATE_VERSION
 * pretends this one is that version.
 */
#include "brebus.h"
#include "common/catalog.h"
#include "common/hf_fetch.h"
#include "common/release_sig.h"
#include "common/sha256.h"
#include "common/update_rel.h"
#include "update.h"
#include "update_key.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

#define LATEST                                                                 \
    "https://api.github.com/repos/prabanta-dev/brebus/releases/latest"
#define DAY (24 * 3600)
#define IDLE_MINUTES 15
#define START_SECONDS 5
#define PIECE ((int64_t)8 << 20)

static struct {
    pthread_mutex_t mu;
    pthread_cond_t wake;
    int started, check_now, install_now, exiting;
    struct brebus *b;
    struct brebus_up_where w;
    char mode[16];   /* auto, notify, off */
    char phase[16];  /* idle, checking, downloading, ready, installing,
                        error */
    char latest[64]; /* the latest release, as last asked */
    char staged[64]; /* its version, downloaded and checked */
    char file[4400]; /* where */
    char failed[64]; /* a version that did not start: not again */
    char done[64];   /* installed at the last start: said once */
    char from[64];   /* the version it replaced */
    char err[400];
    double pct;
    long long last_check;
} U = {.mu = PTHREAD_MUTEX_INITIALIZER, .wake = PTHREAD_COND_INITIALIZER};

const char *brebus_up_version(void)
{
    const char *v = getenv("BREBUS_UPDATE_VERSION");
    return v && *v ? v : JANAS_VERSION;
}

static void dir_of(char *out, size_t len)
{
    snprintf(out, len, "%s/update", U.b->data_dir);
}

/* ---- update.json: under U.mu ---- */

static void save_locked(void)
{
    char d[4200], p[4300], t[4310];
    dir_of(d, sizeof d);
    janas_mkdirs(d);
    snprintf(p, sizeof p, "%s/update.json", d);
    snprintf(t, sizeof t, "%s.tmp", p);
    struct janas_buf o = {0};
    janas_buf_puts(&o, "{\"mode\": ");
    janas_json_write_str(&o, U.mode, strlen(U.mode));
    janas_buf_printf(&o, ", \"last_check\": %lld, \"latest\": ", U.last_check);
    janas_json_write_str(&o, U.latest, strlen(U.latest));
    janas_buf_puts(&o, ", \"staged\": ");
    janas_json_write_str(&o, U.staged, strlen(U.staged));
    janas_buf_puts(&o, ", \"file\": ");
    janas_json_write_str(&o, U.file, strlen(U.file));
    janas_buf_puts(&o, ", \"failed\": ");
    janas_json_write_str(&o, U.failed, strlen(U.failed));
    janas_buf_puts(&o, "}\n");
    FILE *f = o.oom ? NULL : fopen(t, "wb");
    if (f) {
        int ok = fwrite(o.p, 1, o.n, f) == o.n;
        ok &= fclose(f) == 0;
        if (ok)
            rename(t, p);
        else
            remove(t);
    }
    janas_buf_free(&o);
}

static void str_of(const struct janas_json *r, const char *key, char *out,
                   size_t len)
{
    const char *s = janas_json_str(janas_json_get(r, key));
    if (s && strlen(s) < len)
        snprintf(out, len, "%s", s);
}

/* what the finisher left (finished.json): installed (done, from) or put
   back (failed); read at the start, and on each round of the thread (the
   new Brebus starts before the finisher writes it) */
static void finisher_note_locked(void)
{
    char d[4200], p[4300], err[160];
    dir_of(d, sizeof d);
    snprintf(p, sizeof p, "%s/finished.json", d);
    FILE *in;
    struct janas_json_doc *doc;
    const struct janas_json *r;
    in = fopen(p, "rb");
    if (in) {
        char buf[600];
        size_t n = fread(buf, 1, sizeof buf - 1, in);
        fclose(in);
        remove(p);
        doc = janas_json_parse(buf, n, err, sizeof err);
        r = doc ? janas_json_root(doc) : NULL;
        if (r) {
            char ok[8] = "";
            str_of(r, "ok", ok, sizeof ok);
            if (!strcmp(ok, "yes")) {
                str_of(r, "version", U.done, sizeof U.done);
                str_of(r, "from", U.from, sizeof U.from);
                /* installed: nothing staged any more (the finisher may
                   not have removed its file yet) */
                if (!strcmp(U.done, U.staged)) {
                    if (U.file[0])
                        remove(U.file);
                    U.staged[0] = U.file[0] = 0;
                    snprintf(U.phase, sizeof U.phase, "idle");
                }
            } else {
                str_of(r, "version", U.failed, sizeof U.failed);
                str_of(r, "why", U.err, sizeof U.err);
                snprintf(U.phase, sizeof U.phase, "error");
            }
            save_locked();
        }
        janas_json_free(doc);
    }
}

static void load_locked(void)
{
    char d[4200], p[4300];
    dir_of(d, sizeof d);
    snprintf(p, sizeof p, "%s/update.json", d);
    snprintf(U.mode, sizeof U.mode, "auto");
    struct janas_buf f = {0};
    FILE *in = fopen(p, "rb");
    if (in) {
        char buf[4096];
        for (size_t n; (n = fread(buf, 1, sizeof buf, in)) > 0;)
            janas_buf_put(&f, buf, n);
        fclose(in);
    }
    char err[160];
    struct janas_json_doc *doc =
        f.n && !f.oom ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *r = doc ? janas_json_root(doc) : NULL;
    if (r) {
        char m[16] = "";
        str_of(r, "mode", m, sizeof m);
        if (!strcmp(m, "notify") || !strcmp(m, "off"))
            snprintf(U.mode, sizeof U.mode, "%s", m);
        U.last_check =
            (long long)janas_json_num(janas_json_get(r, "last_check"), 0);
        str_of(r, "latest", U.latest, sizeof U.latest);
        str_of(r, "staged", U.staged, sizeof U.staged);
        str_of(r, "file", U.file, sizeof U.file);
        str_of(r, "failed", U.failed, sizeof U.failed);
    }
    janas_json_free(doc);
    janas_buf_free(&f);
    /* staged for a version already here, or gone: nothing to install */
    struct stat st;
    if (U.staged[0] && (!janas_rel_newer(U.staged, brebus_up_version()) ||
                        stat(U.file, &st) != 0)) {
        if (U.file[0])
            remove(U.file);
        U.staged[0] = U.file[0] = 0;
    }
    finisher_note_locked();
}

/* ---- the release ---- */

static int get_text(const char *url, struct janas_buf *out, char *err,
                    size_t en)
{
    return janas_hf_get_url(url, NULL, NULL, out, NULL, 0, err, en);
}

/* url to path in pieces, its SHA-256 checked against sha */
static int download(const char *url, uint64_t size, const char *path,
                    const char *sha, char *err, size_t en)
{
    char part[4500];
    snprintf(part, sizeof part, "%s.part", path);
    FILE *f = fopen(part, "wb");
    if (!f) {
        snprintf(err, en, "%.250s: %.100s", part, strerror(errno));
        return -1;
    }
    struct janas_sha256 h;
    janas_sha256_init(&h);
    char final[1024] = "";
    int rc = 0;
    for (int64_t at = 0; at < (int64_t)size && rc == 0;) {
        int64_t to = at + PIECE < (int64_t)size ? at + PIECE : (int64_t)size;
        struct janas_buf piece = {0};
        int tries = 0;
        /* a file of one piece in one request, no range: any server */
        int whole = at == 0 && to == (int64_t)size;
        while ((rc = janas_hf_get_url(final[0] ? final : url,
                                      whole ? NULL : &at, whole ? NULL : &to,
                                      &piece, final, sizeof final, err, en)) !=
                   0 &&
               ++tries < 4) {
            janas_buf_free(&piece);
            final[0] = 0; /* the redirect asked again: it may expire */
            sleep(2 * tries);
        }
        if (rc == 0 && (piece.oom || piece.n != (size_t)(to - at))) {
            snprintf(err, en, "a piece of the download came short");
            rc = -1;
        }
        if (rc == 0) {
            janas_sha256_update(&h, piece.p, piece.n);
            if (fwrite(piece.p, 1, piece.n, f) != piece.n) {
                snprintf(err, en, "%.300s: not written (the disk full?)", part);
                rc = -1;
            }
        }
        janas_buf_free(&piece);
        at = to;
        pthread_mutex_lock(&U.mu);
        U.pct = size ? 100.0 * (double)at / (double)size : 100;
        int stop = U.exiting;
        pthread_mutex_unlock(&U.mu);
        if (stop && rc == 0) {
            snprintf(err, en, "Brebus ended during the download");
            rc = -1;
        }
    }
    if (fclose(f) != 0 && rc == 0) {
        snprintf(err, en, "%.300s: not written", part);
        rc = -1;
    }
    char got[65];
    janas_sha256_hex(&h, got);
    if (rc == 0 && strcmp(got, sha) != 0) {
        snprintf(err, en,
                 "the download is not the release's file (its SHA-256 "
                 "differs from SHA256SUMS)");
        rc = -1;
    }
    if (rc == 0 && rename(part, path) != 0) {
        snprintf(err, en, "%.250s: %.100s", path, strerror(errno));
        rc = -1;
    }
    if (rc != 0)
        remove(part);
    return rc;
}

static void set_error(const char *why)
{
    pthread_mutex_lock(&U.mu);
    snprintf(U.phase, sizeof U.phase, "error");
    snprintf(U.err, sizeof U.err, "%.399s", why);
    pthread_mutex_unlock(&U.mu);
}

/* the public key updates are checked with */
static int public_key(uint8_t pub[JANAS_SIG_PUB])
{
    const char *k = getenv("BREBUS_UPDATE_KEY");
    if (!k || !*k)
        k = BREBUS_UPDATE_PUBLIC_KEY;
    return janas_sig_unhex(k, strlen(k), pub, JANAS_SIG_PUB);
}

/* asks for the latest release; a newer one downloaded and checked */
static void check(void)
{
    char err[400] = "", tag[64] = "";
    const char *url = getenv("BREBUS_UPDATE_URL");
    int local = url && *url;
    if (!local)
        url = LATEST;
    else if (strncmp(url, "https://", 8) &&
             strncmp(url, "http://127.0.0.1:", 17)) {
        set_error("BREBUS_UPDATE_URL: HTTPS, or http://127.0.0.1 only");
        return;
    }
    pthread_mutex_lock(&U.mu);
    snprintf(U.phase, sizeof U.phase, "checking");
    U.err[0] = 0;
    pthread_mutex_unlock(&U.mu);
    struct janas_buf r = {0};
    struct janas_rel_asset a[48];
    int n = 0;
    if (get_text(url, &r, err, sizeof err) != 0 ||
        janas_rel_parse(r.p, r.n, tag, sizeof tag, a, 48, &n, err,
                        sizeof err) != 0) {
        janas_buf_free(&r);
        char why[460];
        snprintf(why, sizeof why,
                 "could not ask for the latest version: %.380s", err);
        set_error(why);
        return;
    }
    janas_buf_free(&r);
    pthread_mutex_lock(&U.mu);
    U.last_check = (long long)time(NULL);
    snprintf(U.latest, sizeof U.latest, "%s", tag);
    int wanted = janas_rel_newer(tag, brebus_up_version()) &&
                 strcmp(tag, U.failed) && strcmp(tag, U.staged) &&
                 U.w.kind != BREBUS_UP_SYSTEM && U.w.kind != BREBUS_UP_NONE &&
                 strcmp(U.mode, "off");
    snprintf(U.phase, sizeof U.phase, "%s", U.staged[0] ? "ready" : "idle");
    save_locked();
    char asset[128];
    snprintf(asset, sizeof asset, "%s", U.w.asset);
    pthread_mutex_unlock(&U.mu);
    if (!wanted)
        return;

    /* SHA256SUMS, its signature, the file */
    const struct janas_rel_asset *sums = NULL, *sig = NULL, *file = NULL;
    for (int i = 0; i < n; i++) {
        if (!strcmp(a[i].name, "SHA256SUMS"))
            sums = &a[i];
        else if (!strcmp(a[i].name, "SHA256SUMS.sig"))
            sig = &a[i];
        else if (!strcmp(a[i].name, asset))
            file = &a[i];
    }
    if (!sums || !sig || !file) {
        char why[300];
        snprintf(why, sizeof why,
                 "%s has no %s: not installed (a release without its "
                 "signature is never installed)",
                 tag,
                 !file   ? asset
                 : !sums ? "SHA256SUMS"
                         : "signature");
        set_error(why);
        return;
    }
    struct janas_buf s = {0}, g = {0};
    uint8_t pub[JANAS_SIG_PUB];
    char sha[65];
    if (get_text(sums->url, &s, err, sizeof err) != 0 ||
        get_text(sig->url, &g, err, sizeof err) != 0) {
        char why[460];
        snprintf(why, sizeof why, "%.60s: %.380s", tag, err);
        set_error(why);
    } else if (public_key(pub) != 0 ||
               janas_sig_verify(pub, s.p, s.n, g.p, g.n) != 0) {
        char why[200];
        snprintf(why, sizeof why,
                 "%s: its SHA256SUMS is not signed by the project's key: "
                 "not installed",
                 tag);
        set_error(why);
    } else if (janas_rel_sum(s.p, s.n, asset, sha) != 0) {
        char why[300];
        snprintf(why, sizeof why, "%s: SHA256SUMS does not list %s", tag,
                 asset);
        set_error(why);
    } else {
        char d[4200], path[4400];
        dir_of(d, sizeof d);
        janas_mkdirs(d);
        snprintf(path, sizeof path, "%s/%s", d, asset);
        pthread_mutex_lock(&U.mu);
        snprintf(U.phase, sizeof U.phase, "downloading");
        U.pct = 0;
        pthread_mutex_unlock(&U.mu);
        if (download(file->url, file->size, path, sha, err, sizeof err) != 0) {
            char why[460];
            snprintf(why, sizeof why, "%.60s: %.380s", tag, err);
            set_error(why);
        } else {
            pthread_mutex_lock(&U.mu);
            if (U.file[0] && strcmp(U.file, path))
                remove(U.file);
            snprintf(U.staged, sizeof U.staged, "%s", tag);
            snprintf(U.file, sizeof U.file, "%s", path);
            snprintf(U.phase, sizeof U.phase, "ready");
            save_locked();
            pthread_mutex_unlock(&U.mu);
        }
    }
    janas_buf_free(&s);
    janas_buf_free(&g);
}

/* nobody has used it for a while: no model asked, no download */
static int idle(struct brebus *b)
{
    time_t now = time(NULL);
    int busy = 0;
    pthread_mutex_lock(&b->mu);
    for (int i = 0; i < BREBUS_LOADED; i++)
        busy |= b->srv[i].on && now - b->srv[i].used < IDLE_MINUTES * 60;
    pthread_mutex_unlock(&b->mu);
    for (int i = 0; i < BREBUS_GETS && !busy; i++) {
        pthread_mutex_lock(&b->get[i].mu);
        busy |= b->get[i].running;
        pthread_mutex_unlock(&b->get[i].mu);
    }
    return !busy;
}

/* the staged version put in place (restart: started again after) */
static int install(int restart)
{
    char err[400] = "", file[4400], ver[64];
    struct brebus_up_where w;
    pthread_mutex_lock(&U.mu);
    struct stat st;
    if (U.staged[0] && stat(U.file, &st) != 0) { /* gone: nothing staged */
        U.staged[0] = U.file[0] = 0;
        save_locked();
    }
    if (!U.staged[0] || !strcmp(U.phase, "installing")) {
        pthread_mutex_unlock(&U.mu);
        return -1;
    }
    snprintf(file, sizeof file, "%s", U.file);
    snprintf(ver, sizeof ver, "%s", U.staged);
    w = U.w;
    snprintf(U.phase, sizeof U.phase, "installing");
    pthread_mutex_unlock(&U.mu);
    int rc = brebus_up_apply(U.b, &w, file, ver, restart, err, sizeof err);
    if (rc != 0) {
        char why[460];
        snprintf(why, sizeof why, "%.60s not installed: %.380s", ver, err);
        set_error(why);
        pthread_mutex_lock(&U.mu);
        snprintf(U.phase, sizeof U.phase, "ready");
        pthread_mutex_unlock(&U.mu);
    }
    return rc;
}

static void *thread(void *arg)
{
    struct brebus *b = arg;
    /* a few seconds for the start to settle, then the check */
    struct timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_sec += START_SECONDS;
    pthread_mutex_lock(&U.mu);
    while (!U.check_now && !U.install_now && !U.exiting &&
           pthread_cond_timedwait(&U.wake, &U.mu, &until) == 0)
        ;
    pthread_mutex_unlock(&U.mu);
    int first = 1;
    for (;;) {
        pthread_mutex_lock(&U.mu);
        finisher_note_locked();
        int exiting = U.exiting, now = U.check_now, inst = U.install_now;
        U.check_now = U.install_now = 0;
        /* at the start, then once a day; a build between releases only
           when asked (it never calls GitHub by itself) */
        int due = strcmp(U.mode, "off") && U.w.kind != BREBUS_UP_NONE &&
                  (first || (long long)time(NULL) - U.last_check >= DAY);
        first = 0;
        pthread_mutex_unlock(&U.mu);
        if (exiting)
            break;
        if (now || due)
            check();
        pthread_mutex_lock(&U.mu);
        int autom = !strcmp(U.mode, "auto") && U.staged[0];
        pthread_mutex_unlock(&U.mu);
        /* without a window the restart is nobody's bother: once idle */
        if (inst || (autom && !b->windowed && idle(b)))
            install(1);
        clock_gettime(CLOCK_REALTIME, &until);
        until.tv_sec += 60;
        pthread_mutex_lock(&U.mu);
        while (!U.check_now && !U.install_now && !U.exiting &&
               pthread_cond_timedwait(&U.wake, &U.mu, &until) == 0)
            ;
        pthread_mutex_unlock(&U.mu);
    }
    return NULL;
}

void brebus_update_start(struct brebus *b)
{
    if (b->child)
        return; /* a user's own Brebus: its front's is updated */
    U.b = b;
    brebus_up_where(b->bin_dir, getenv("APPIMAGE"), &U.w);
    pthread_mutex_lock(&U.mu);
    load_locked();
    if (!U.phase[0])
        snprintf(U.phase, sizeof U.phase, "%s", U.staged[0] ? "ready" : "idle");
    pthread_mutex_unlock(&U.mu);
    /* the finisher waits for this: the new version serves */
    char d[4200], p[4300];
    dir_of(d, sizeof d);
    janas_mkdirs(d);
    snprintf(p, sizeof p, "%s/" BREBUS_UP_STARTED, d);
    FILE *f = fopen(p, "wb");
    if (f) {
        fprintf(f, "%s\n", brebus_up_version());
        fclose(f);
    }
    /* a build between releases: told, never updated */
    int p4[4];
    if (janas_rel_version(brebus_up_version(), p4) != 0 &&
        U.w.kind != BREBUS_UP_SYSTEM) {
        U.w.kind = BREBUS_UP_NONE;
        snprintf(U.w.why, sizeof U.w.why,
                 "a build between releases (%.60s): not updated",
                 brebus_up_version());
    }
    pthread_t t;
    if (pthread_create(&t, NULL, thread, b) == 0) {
        pthread_detach(t);
        U.started = 1;
    }
}

void brebus_update_at_exit(struct brebus *b)
{
    (void)b;
    if (!U.started)
        return;
    pthread_mutex_lock(&U.mu);
    U.exiting = 1;
    pthread_cond_broadcast(&U.wake);
    int go =
        !strcmp(U.mode, "auto") && U.staged[0] && strcmp(U.phase, "installing");
    pthread_mutex_unlock(&U.mu);
    /* as it ends, the next start is the new version's */
    if (go)
        install(0);
}

static const char *kind_name(enum brebus_up_kind k)
{
    return k == BREBUS_UP_APPIMAGE ? "appimage"
           : k == BREBUS_UP_FOLDER ? "folder"
           : k == BREBUS_UP_MACAPP ? "app"
           : k == BREBUS_UP_SETUP  ? "setup"
           : k == BREBUS_UP_SYSTEM ? "system"
                                   : "none";
}

static void put_str(struct janas_buf *o, const char *key, const char *v)
{
    janas_buf_printf(o, ", \"%s\": ", key);
    janas_json_write_str(o, v, strlen(v));
}

void brebus_update_status(struct brebus *b, struct janas_buf *o)
{
    (void)b;
    pthread_mutex_lock(&U.mu);
    if (U.started)
        finisher_note_locked();
    janas_buf_puts(o, "{\"version\": ");
    janas_json_write_str(o, brebus_up_version(), strlen(brebus_up_version()));
    put_str(o, "mode", U.mode[0] ? U.mode : "auto");
    put_str(o, "kind", kind_name(U.w.kind));
    put_str(o, "why", U.w.why);
    put_str(o, "phase", U.phase);
    put_str(o, "latest", U.latest);
    janas_buf_printf(o, ", \"newer\": %s",
                     janas_rel_newer(U.latest, brebus_up_version()) ? "true"
                                                                    : "false");
    put_str(o, "staged", U.staged);
    put_str(o, "failed", U.failed);
    put_str(o, "done", U.done);
    put_str(o, "from", U.from);
    put_str(o, "error", U.err);
    janas_buf_printf(o, ", \"pct\": %.1f, \"last_check\": %lld}", U.pct,
                     U.last_check);
    pthread_mutex_unlock(&U.mu);
}

int brebus_update_action(struct brebus *b, const char *body, size_t n,
                         struct janas_buf *o)
{
    char err[160];
    struct janas_json_doc *d = janas_json_parse(body, n, err, sizeof err);
    const struct janas_json *r = d ? janas_json_root(d) : NULL;
    if (!r || !U.started) {
        janas_json_free(d);
        janas_buf_puts(o, "{\"error\": \"no updates in this Brebus\"}");
        return -1;
    }
    const char *m = janas_json_str(janas_json_get(r, "mode"));
    const struct janas_json *c = janas_json_get(r, "check"),
                            *i = janas_json_get(r, "install"),
                            *s = janas_json_get(r, "seen");
    int rc = 0;
    pthread_mutex_lock(&U.mu);
    if (m) {
        if (!strcmp(m, "auto") || !strcmp(m, "notify") || !strcmp(m, "off")) {
            snprintf(U.mode, sizeof U.mode, "%s", m);
            save_locked();
        } else
            rc = -1;
    }
    if (c && c->type == JANAS_JSON_TRUE)
        U.check_now = 1;
    if (i && i->type == JANAS_JSON_TRUE) {
        if (U.staged[0])
            U.install_now = 1;
        else
            rc = -1;
    }
    if (s && s->type == JANAS_JSON_TRUE) /* the notice read */
        U.done[0] = U.from[0] = 0;
    pthread_cond_broadcast(&U.wake);
    pthread_mutex_unlock(&U.mu);
    janas_json_free(d);
    if (rc != 0) {
        janas_buf_puts(o, "{\"error\": \"nothing to install, or no such "
                          "mode\"}");
        return -1;
    }
    brebus_update_status(b, o);
    return 0;
}
