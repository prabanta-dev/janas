/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * server.c - the model's server: a janas-server of its own, started on a
 * model of My models at a port of its own on 127.0.0.1 (Brebus's port plus
 * one), its output in the data folder (server.log). Brebus passes the
 * page's API requests on to it (proxy.c); other programs may use it at its
 * address. One model at a time: starting another stops the first.
 *
 * The server's own settings, kept by the page under "_server" in
 * settings.json: reachable from the local network (--host 0.0.0.0), a key
 * clients must send (given in the environment, JANAS_API_KEY, never on its
 * command line, which every user of the machine can read), the web pages of
 * other origins that may call it (--allow-origin). Its engine monitor is
 * always on (--monitor), and Brebus passes it on too.
 */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#else
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#endif

#include "brebus.h"
#include "common/catalog.h"
#include "common/devices.h"
#include "common/mcp_http.h"
#include "common/mcp_tls.h"
#include "common/model_scan.h"

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

static int fail(struct janas_buf *out, const char *why)
{
    out->n = 0;
    janas_buf_puts(out, "{\"error\": ");
    str(out, why);
    janas_buf_puts(out, "}");
    return -1;
}

/* the server answers its models' list: ready */
static int answers(unsigned short port, const char *key, int tls)
{
    char url[64], err[128];
    snprintf(url, sizeof url, "%s://127.0.0.1:%u/v1/models",
             tls ? "https" : "http", port);
    struct janas_url u;
    if (janas_url_parse(url, &u, err, sizeof err) != 0)
        return 0;
    char auth[320];
    snprintf(auth, sizeof auth, "Authorization: Bearer %s", key);
    const char *hdr[] = {auth};
    struct janas_http h;
    if (janas_http_request(&h, &u, "GET", hdr, key[0] ? 1 : 0, NULL, 0, 2000,
                           NULL, err, sizeof err) != 0)
        return 0;
    int ok = h.head.status == 200;
    janas_http_close(&h);
    return ok;
}

#define MAX_ORIGINS 8

/* settings.json's "_server": the local network, the key, the origins */
static void server_settings(struct brebus *b, int *lan, int *tls, char *key,
                            size_t kn, char origins[][256], int *n_origins)
{
    *lan = 0;
    *tls = 0;
    key[0] = 0;
    *n_origins = 0;
    struct janas_buf f = {0};
    brebus_settings_get(b, &f);
    char err[160];
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *s =
        d ? janas_json_get(janas_json_root(d), "_server") : NULL;
    const struct janas_json *v = s ? janas_json_get(s, "lan") : NULL;
    *lan = v && v->type == JANAS_JSON_TRUE;
    v = s ? janas_json_get(s, "https") : NULL;
    *tls = v && v->type == JANAS_JSON_TRUE;
    const char *k = s ? janas_json_str(janas_json_get(s, "api_key")) : NULL;
    /* printable, no spaces: it goes into a header */
    int good = k && *k && strlen(k) < kn && strlen(k) < 200;
    for (const char *c = k; good && *c; c++)
        good = *c > ' ' && *c < 127;
    if (good)
        snprintf(key, kn, "%s", k);
    const struct janas_json *o = s ? janas_json_get(s, "origins") : NULL;
    for (const struct janas_json *e =
             o && o->type == JANAS_JSON_ARRAY ? o->child : NULL;
         e && *n_origins < MAX_ORIGINS; e = e->next) {
        const char *x = janas_json_str(e);
        if (x && (!strncmp(x, "http://", 7) || !strncmp(x, "https://", 8)) &&
            strlen(x) < 256 && !strpbrk(x, " \r\n"))
            snprintf(origins[(*n_origins)++], 256, "%s", x);
    }
    janas_json_free(d);
    janas_buf_free(&f);
}

/* this machine's first IPv4 address on a network, for the address other
   computers reach it at; "" when it has none */
static void lan_address(char *ip, size_t n)
{
    ip[0] = 0;
#ifdef _WIN32
    /* the adapters that are up, their unicast IPv4 addresses */
    ULONG sz = 16384;
    IP_ADAPTER_ADDRESSES *all = malloc(sz);
    if (all &&
        GetAdaptersAddresses(AF_INET,
                             GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                                 GAA_FLAG_SKIP_DNS_SERVER,
                             NULL, all, &sz) == ERROR_BUFFER_OVERFLOW) {
        free(all);
        all = malloc(sz);
    }
    if (all &&
        GetAdaptersAddresses(AF_INET,
                             GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                                 GAA_FLAG_SKIP_DNS_SERVER,
                             NULL, all, &sz) == NO_ERROR)
        for (IP_ADAPTER_ADDRESSES *a = all; a && !ip[0]; a = a->Next) {
            if (a->OperStatus != IfOperStatusUp ||
                a->IfType == IF_TYPE_SOFTWARE_LOOPBACK)
                continue;
            for (IP_ADAPTER_UNICAST_ADDRESS *u = a->FirstUnicastAddress;
                 u && !ip[0]; u = u->Next) {
                struct sockaddr_in *s =
                    (struct sockaddr_in *)u->Address.lpSockaddr;
                if (s->sin_family == AF_INET)
                    inet_ntop(AF_INET, &s->sin_addr, ip, (socklen_t)n);
            }
        }
    free(all);
#else
    struct ifaddrs *all, *i;
    if (getifaddrs(&all) != 0)
        return;
    for (i = all; i && !ip[0]; i = i->ifa_next) {
        if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET)
            continue;
        struct in_addr a = ((struct sockaddr_in *)i->ifa_addr)->sin_addr;
        if ((ntohl(a.s_addr) >> 24) == 127)
            continue;
        inet_ntop(AF_INET, &a, ip, (socklen_t)n);
    }
    freeifaddrs(all);
#endif
}

/* Brebus's own certificate, for HTTPS: made in data_dir/tls the first
   time, or again when the names it is for changed or it ends within a
   month; trusted by Brebus's own requests to its server. 0, or -1. */
static int certificate(struct brebus *b, char *cert, char *key, size_t n,
                       char *err, size_t en)
{
    char dir[4200], names_f[4300], host[256] = "localhost", ip[64] = "";
    snprintf(dir, sizeof dir, "%s/tls", b->data_dir);
    janas_mkdirs(dir);
    snprintf(cert, n, "%s/cert.pem", dir);
    snprintf(key, n, "%s/key.pem", dir);
    snprintf(names_f, sizeof names_f, "%s/names.txt", dir);
    gethostname(host, sizeof host - 1);
    lan_address(ip, sizeof ip);
    const char *names[] = {"localhost", host, "127.0.0.1", "::1", ip};
    size_t nn = ip[0] ? 5 : 4;
    char want[512], had[512] = "";
    snprintf(want, sizeof want, "%s %s %s", host, ip, "v1");
    FILE *f = fopen(names_f, "rb");
    if (f) {
        size_t r = fread(had, 1, sizeof had - 1, f);
        had[r] = 0;
        fclose(f);
    }
    struct stat st;
    int fresh = stat(cert, &st) == 0 && access(key, R_OK) == 0 &&
                !strcmp(had, want) &&
                time(NULL) - st.st_mtime < (time_t)(825 - 30) * 86400;
    if (!fresh) {
        if (janas_tls_make_cert(cert, key, names, nn, 825, err, en) != 0)
            return -1;
        f = fopen(names_f, "wb");
        if (f) {
            fputs(want, f);
            fclose(f);
        }
    }
    static char trusted[4300];
    if (strcmp(trusted, cert) || !fresh) {
        if (janas_tls_trust(cert) != 0) {
            snprintf(err, en, "cannot read the certificate Brebus made");
            return -1;
        }
        snprintf(trusted, sizeof trusted, "%s", cert);
    }
    if (janas_tls_fingerprint(cert, b->srv_fp, sizeof b->srv_fp) != 0)
        b->srv_fp[0] = 0;
    return 0;
}

static void stop_locked(struct brebus *b)
{
    if (!b->srv_on)
        return;
    janas_proc_kill(&b->srv, 0);
    int code;
    if (janas_proc_wait(&b->srv, 10000, &code) != 1) {
        janas_proc_kill(&b->srv, 1);
        janas_proc_wait(&b->srv, -1, &code);
    }
    b->srv_on = 0;
    b->srv_model[0] = 0;
}

int brebus_server_start(struct brebus *b, const char *body, size_t n,
                        struct janas_buf *out)
{
    char err[160];
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *name = o ? janas_json_str(janas_json_get(o, "model")) : NULL;
    double par = o ? janas_json_num(janas_json_get(o, "parallel"), 0) : 0;
    double ctx = o ? janas_json_num(janas_json_get(o, "context"), 0) : 0;
    const struct janas_json *th = o ? janas_json_get(o, "think") : NULL;
    if (!name) {
        janas_json_free(d);
        return fail(out, "no model named");
    }
    static struct janas_model_file m[JANAS_MODELS_MAX];
    int nm = janas_models_scan(b->models_dir, m, JANAS_MODELS_MAX), at = -1;
    for (int i = 0; i < nm && at < 0; i++)
        if (!strcmp(m[i].name, name))
            at = i;
    if (at < 0) {
        janas_json_free(d);
        return fail(out, "no such model in My models");
    }
    pthread_mutex_lock(&b->mu);
    stop_locked(b);
    char exe[4200], port[8], par_s[8], ctx_s[16], logp[4200], env_key[300];
    char origins[MAX_ORIGINS][256];
    int lan, tls, n_origins;
    server_settings(b, &lan, &tls, b->srv_key, sizeof b->srv_key, origins,
                    &n_origins);
    b->srv_lan = lan;
    static char cert[4300], key[4300];
    b->srv_tls = 0;
    b->srv_fp[0] = 0;
    if (tls) {
        if (certificate(b, cert, key, sizeof cert, err, sizeof err) != 0) {
            pthread_mutex_unlock(&b->mu);
            janas_json_free(d);
            return fail(out, err);
        }
        b->srv_tls = 1;
    }
    snprintf(exe, sizeof exe, "%s/janas-server%s", b->bin_dir, b->suffix);
    b->srv_port = (unsigned short)(b->port + 1);
    snprintf(port, sizeof port, "%u", b->srv_port);
    const char *argv[40 + 2 * MAX_ORIGINS];
    int k = 0;
    argv[k++] = exe;
    argv[k++] = m[at].path;
    argv[k++] = "--port";
    argv[k++] = port;
    argv[k++] = "--monitor";
    /* a Whisper of the models' folder, for the recordings attached: the
       best there is (large-v3-turbo, large-v3, small, small.en) */
    static const char *const W[] = {"whisper-large-v3-turbo",
                                    "whisper-large-v3", "whisper-small",
                                    "whisper-small.en"};
    static char speech[4300];
    speech[0] = 0;
    for (size_t i = 0; i < sizeof W / sizeof *W && !speech[0]; i++) {
        snprintf(speech, sizeof speech, "%s/%s.jns", b->models_dir, W[i]);
        if (access(speech, R_OK) != 0)
            speech[0] = 0;
    }
    if (speech[0]) {
        argv[k++] = "--speech-model";
        argv[k++] = speech;
    }
    if (lan) {
        argv[k++] = "--host";
        argv[k++] = "0.0.0.0";
    }
    if (b->srv_tls) {
        argv[k++] = "--tls-cert";
        argv[k++] = cert;
        argv[k++] = "--tls-key";
        argv[k++] = key;
    }
    for (int i = 0; i < n_origins; i++) {
        argv[k++] = "--allow-origin";
        argv[k++] = origins[i];
    }
    if (par >= 1 && par <= 16) {
        snprintf(par_s, sizeof par_s, "%d", (int)par);
        argv[k++] = "--parallel";
        argv[k++] = par_s;
    }
    if (ctx >= 512) {
        snprintf(ctx_s, sizeof ctx_s, "%d", (int)ctx);
        argv[k++] = "--ctx";
        argv[k++] = ctx_s;
    }
    if (th && (th->type == JANAS_JSON_FALSE || th->type == JANAS_JSON_TRUE)) {
        argv[k++] = "--think";
        argv[k++] = th->type == JANAS_JSON_TRUE ? "on" : "off";
    }
    /* where each kind of pass runs, when the user chose it: else Janas
       chooses, measuring this computer */
    static const char *const ON[] = {"prompt_on", "drafts_on", "reply_on"};
    static char on_v[JANAS_DEVICES_OPTIONS][32];
    for (int i = 0; i < JANAS_DEVICES_OPTIONS; i++) {
        const char *v = o ? janas_json_str(janas_json_get(o, ON[i])) : NULL;
        char why[200];
        if (!v || strlen(v) >= sizeof on_v[i] ||
            janas_devices_mask(v, why, sizeof why) <= 0)
            continue; /* none, auto, or not one the engine reads */
        snprintf(on_v[i], sizeof on_v[i], "%s", v);
        argv[k++] = janas_devices_options[i];
        argv[k++] = on_v[i];
    }
    argv[k] = NULL;
    /* the GPU's way, when the user chose one: else Janas's (Vulkan, then
       CUDA or HIP where there is no Vulkan) */
    static char env_api[40];
    const char *ga = o ? janas_json_str(janas_json_get(o, "gpu_api")) : NULL;
    int api_ok = ga && (!strcmp(ga, "vulkan") || !strcmp(ga, "cuda") ||
                        !strcmp(ga, "hip"));
    snprintf(env_api, sizeof env_api, "JANAS_GPU_API=%s", api_ok ? ga : "");
    janas_json_free(d);
    snprintf(logp, sizeof logp, "%s/server.log", b->data_dir);
    int fd = open(logp, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    /* the key in its environment, or none: not one Brebus was started with */
    snprintf(env_key, sizeof env_key, "JANAS_API_KEY=%s", b->srv_key);
    const char *env[] = {env_key, api_ok ? env_api : NULL, NULL};
    int rc =
        janas_proc_start(&b->srv, argv, env, NULL, -1, fd, fd,
                         JANAS_PROC_GROUP | BREBUS_NO_CONSOLE, err, sizeof err);
    if (fd >= 0)
        close(fd);
    if (rc != 0) {
        snprintf(b->srv_err, sizeof b->srv_err, "%s", err);
        pthread_mutex_unlock(&b->mu);
        return fail(out, err);
    }
    b->srv_on = 1;
    snprintf(b->srv_model, sizeof b->srv_model, "%s", m[at].name);
    b->srv_used = time(NULL);
    b->srv_last[0] = 0;
    b->srv_err[0] = 0;
    pthread_mutex_unlock(&b->mu);
    /* ready when it answers; a large model takes a while to open */
    for (int i = 0; i < 600; i++) {
        if (answers(b->srv_port, b->srv_key, b->srv_tls))
            break;
        int code;
        pthread_mutex_lock(&b->mu);
        int gone = b->srv_on && janas_proc_wait(&b->srv, 0, &code) == 1;
        if (gone) {
            b->srv_on = 0;
            snprintf(b->srv_err, sizeof b->srv_err,
                     "janas-server ended (status %d): see %.400s", code, logp);
        }
        pthread_mutex_unlock(&b->mu);
        if (gone)
            return fail(out, b->srv_err);
        usleep(200000);
    }
    out->n = 0;
    brebus_server_status(b, out);
    return 0;
}

int brebus_server_stop(struct brebus *b, struct janas_buf *out)
{
    pthread_mutex_lock(&b->mu);
    stop_locked(b);
    pthread_mutex_unlock(&b->mu);
    out->n = 0;
    brebus_server_status(b, out);
    return 0;
}

void brebus_server_status(struct brebus *b, struct janas_buf *out)
{
    pthread_mutex_lock(&b->mu);
    int code, on = b->srv_on;
    if (on && janas_proc_wait(&b->srv, 0, &code) == 1) {
        b->srv_on = on = 0;
        snprintf(b->srv_err, sizeof b->srv_err,
                 "janas-server ended (status %d)", code);
    }
    janas_buf_printf(out,
                     "{\"running\": %s, \"model\": ", on ? "true" : "false");
    str(out, on ? b->srv_model : "");
    janas_buf_printf(out, ", \"port\": %u, \"url\": ", on ? b->srv_port : 0);
    char url[64] = "", ip[64] = "", lan_url[96] = "";
    const char *sch = b->srv_tls ? "https" : "http";
    if (on)
        snprintf(url, sizeof url, "%s://127.0.0.1:%u/v1", sch, b->srv_port);
    str(out, url);
    if (on && b->srv_lan)
        lan_address(ip, sizeof ip);
    if (ip[0])
        snprintf(lan_url, sizeof lan_url, "%s://%s:%u/v1", sch, ip,
                 b->srv_port);
    janas_buf_puts(out, ", \"lan_url\": ");
    str(out, lan_url);
    janas_buf_printf(out, ", \"https\": %s, \"fingerprint\": ",
                     on && b->srv_tls ? "true" : "false");
    str(out, on && b->srv_tls ? b->srv_fp : "");
    janas_buf_puts(out, ", \"api_key\": ");
    str(out, on ? b->srv_key : "");
    janas_buf_puts(out, ", \"last\": ");
    str(out, on ? "" : b->srv_last);
    janas_buf_puts(out, ", \"error\": ");
    str(out, b->srv_err);
    janas_buf_puts(out, "}");
    pthread_mutex_unlock(&b->mu);
}

unsigned short brebus_server_port(struct brebus *b)
{
    pthread_mutex_lock(&b->mu);
    unsigned short p = b->srv_on ? b->srv_port : 0;
    pthread_mutex_unlock(&b->mu);
    return p;
}

/* settings.json's _server.ttl, minutes */
static long ttl_minutes(struct brebus *b)
{
    struct janas_buf f = {0};
    char err[160];
    brebus_settings_get(b, &f);
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *sv =
        d ? janas_json_get(janas_json_root(d), "_server") : NULL;
    long t = sv ? (long)janas_json_num(janas_json_get(sv, "ttl"), 0) : 0;
    janas_json_free(d);
    janas_buf_free(&f);
    return t > 0 ? t : 0;
}

static void *idle(void *arg)
{
    struct brebus *b = arg;
    for (;;) {
        sleep(30);
        long ttl = ttl_minutes(b);
        if (!ttl)
            continue;
        pthread_mutex_lock(&b->mu);
        if (b->srv_on && time(NULL) - b->srv_used > ttl * 60) {
            snprintf(b->srv_last, sizeof b->srv_last, "%s", b->srv_model);
            stop_locked(b);
            snprintf(b->srv_err, sizeof b->srv_err,
                     "unloaded after %ld minutes without a request: loaded "
                     "again at the next message",
                     ttl);
        }
        pthread_mutex_unlock(&b->mu);
    }
    return NULL;
}

void brebus_idle_start(struct brebus *b)
{
    pthread_t t;
    if (pthread_create(&t, NULL, idle, b) == 0)
        pthread_detach(t);
}

int brebus_models_dir_set(struct brebus *b, const char *path, int save)
{
    char real[4096];
    struct stat st;
    if (!janas_path_real(path, real) || stat(real, &st) != 0 ||
        !S_ISDIR(st.st_mode) || strlen(real) >= sizeof b->models_dir)
        return -1;
    pthread_mutex_lock(&b->mu);
    snprintf(b->models_dir, sizeof b->models_dir, "%s", real);
    pthread_mutex_unlock(&b->mu);
    if (!save)
        return 0;
    /* kept in settings.json, beside what the page keeps there */
    struct janas_buf f = {0}, o = {0};
    char err[160];
    brebus_settings_get(b, &f);
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *root = d ? janas_json_root(d) : NULL;
    janas_buf_puts(&o, "{\"_models_dir\": ");
    str(&o, real);
    for (const struct janas_json *e =
             root && root->type == JANAS_JSON_OBJECT ? root->child : NULL;
         e; e = e->next) {
        if (e->key_n == 11 && !memcmp(e->key, "_models_dir", 11))
            continue;
        janas_buf_puts(&o, ", ");
        janas_json_write_str(&o, e->key, e->key_n);
        janas_buf_puts(&o, ": ");
        janas_json_write(&o, e);
    }
    janas_buf_puts(&o, "}");
    int rc = o.oom ? -1 : brebus_settings_put(b, o.p, o.n);
    janas_json_free(d);
    janas_buf_free(&f);
    janas_buf_free(&o);
    return rc;
}

/* what the engine of the model loaded chose for this computer (the devices
   of each kind of pass, its threads), as its monitor's first event says */
void brebus_server_choices(struct brebus *b, struct janas_buf *out)
{
    char url[96], err[200], auth[320];
    pthread_mutex_lock(&b->mu);
    int on = b->srv_on;
    snprintf(url, sizeof url, "%s://127.0.0.1:%u/monitor/events",
             b->srv_tls ? "https" : "http", b->srv_port);
    snprintf(auth, sizeof auth, "Authorization: Bearer %s", b->srv_key);
    int keyed = b->srv_key[0] != 0;
    pthread_mutex_unlock(&b->mu);
    struct janas_url u;
    struct janas_http h;
    struct janas_buf in = {0};
    const char *hdr[] = {auth};
    if (on && janas_url_parse(url, &u, err, sizeof err) == 0 &&
        janas_http_request(&h, &u, "GET", hdr, keyed ? 1 : 0, NULL, 0, 3000,
                           NULL, err, sizeof err) == 0) {
        /* up to the end of the first event */
        for (int i = 0; i < 40 && in.n < (1 << 20); i++) {
            if (janas_http_body(&h, &in, 250) < 0)
                break;
            janas_buf_put(&in, "", 1); /* text: ended, for strstr */
            in.n--;
            if (strstr(in.p, "\n\n"))
                break;
        }
        janas_http_close(&h);
    }
    char *d = in.p && !in.oom ? strstr(in.p, "data: ") : NULL;
    char *e = d ? strchr(d, '\n') : NULL;
    struct janas_json_doc *doc =
        d && e ? janas_json_parse(d + 6, (size_t)(e - d - 6), err, sizeof err)
               : NULL;
    const struct janas_json *r = doc ? janas_json_root(doc) : NULL;
    const char *ch = r ? janas_json_str(janas_json_get(r, "choices")) : NULL;
    const char *ds =
        r ? janas_json_str(janas_json_get(r, "description")) : NULL;
    janas_buf_puts(out, "{\"choices\": ");
    str(out, ch ? ch : "");
    janas_buf_puts(out, ", \"description\": ");
    str(out, ds ? ds : "");
    janas_buf_puts(out, "}");
    janas_json_free(doc);
    janas_buf_free(&in);
}
