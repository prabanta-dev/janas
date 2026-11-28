/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * server.c - the model's server: a janas-server of its own, started on a
 * model of My models at a port of its own on 127.0.0.1 (Brebus's port plus
 * one, two...), its output in the data folder (server.log, server-1.log...).
 * Brebus passes the page's API requests on to the one of the model they
 * name (proxy.c); other programs may use each at its address. Several
 * models loaded at once, up to BREBUS_LOADED: one more is loaded beside the
 * others when it fits in the memory free, else the ones used least lately
 * are unloaded first; a model larger than half the memory runs alone.
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
#include "common/sysinfo.h"

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
    if (!brebus_port_mine(port)) /* the key to no one else's server */
        return 0;
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

/* Brebus's own certificate, for HTTPS (the models' servers, Brebus served):
   made in data_dir/tls the first time, or again when the names it is for
   changed or it ends within a month; trusted by Brebus's own requests to
   its server. 0, or -1. */
int brebus_certificate(struct brebus *b, char *cert, char *key, size_t n,
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

/* b->mu held: the models' servers running, written to servers.json in
   the data folder (pid, port, model each), for the next Brebus to find
   them still there - after a crash, or a Brebus closed while one worked */
static void record_locked(struct brebus *b)
{
    char p[4300], tmp[4400];
    snprintf(p, sizeof p, "%s/servers.json", b->data_dir);
    snprintf(tmp, sizeof tmp, "%s.new", p);
    struct janas_buf o = {0};
    janas_buf_puts(&o, "{\"servers\": [");
    for (int i = 0, k = 0; i < BREBUS_LOADED; i++) {
        const struct brebus_srv *v = &b->srv[i];
        if (!v->on || !v->ready || !v->proc.pid)
            continue;
        janas_buf_printf(&o,
                         "%s{\"slot\": %d, \"pid\": %lu, \"port\": %u, "
                         "\"model\": ",
                         k++ ? ", " : "", i, v->proc.pid, v->port);
        janas_json_write_str(&o, v->model, strlen(v->model));
        janas_buf_puts(&o, "}");
    }
    janas_buf_puts(&o, "]}\n");
    int fd =
        o.oom ? -1 : open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    int bad = fd < 0 || write(fd, o.p, o.n) != (ssize_t)o.n;
    if (fd >= 0)
        bad |= close(fd) != 0;
    if (bad || rename(tmp, p) != 0)
        remove(tmp);
    janas_buf_free(&o);
}

static void stop_locked(struct brebus *b, int i)
{
    struct brebus_srv *v = &b->srv[i];
    if (!v->on)
        return;
    janas_proc_kill(&v->proc, 0);
    int code;
    if (janas_proc_wait(&v->proc, 10000, &code) != 1) {
        janas_proc_kill(&v->proc, 1);
        janas_proc_wait(&v->proc, -1, &code);
    }
    v->on = 0;
    v->model[0] = 0;
    record_locked(b);
}

/* the place of the model loaded, -1 when it is not */
static int loaded(struct brebus *b, const char *name)
{
    for (int i = 0; name && i < BREBUS_LOADED; i++)
        if (b->srv[i].on && !strcmp(b->srv[i].model, name))
            return i;
    return -1;
}

/* the place used last, -1 with none loaded */
static int current(struct brebus *b)
{
    int c = b->srv_cur;
    if (c >= 0 && c < BREBUS_LOADED && b->srv[c].on && b->srv[c].ready)
        return c;
    c = -1;
    for (int i = 0; i < BREBUS_LOADED; i++)
        if (b->srv[i].on && b->srv[i].ready &&
            (c < 0 || b->srv[i].used > b->srv[c].used))
            c = i;
    return c;
}

/* settings.json's "_server": how many models may be loaded at once (1 to
   BREBUS_LOADED; BREBUS_LOADED when not said) */
static int max_loaded(struct brebus *b)
{
    struct janas_buf f = {0};
    char err[160];
    brebus_settings_get(b, &f);
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *sv =
        d ? janas_json_get(janas_json_root(d), "_server") : NULL;
    double x = sv ? janas_json_num(janas_json_get(sv, "max_loaded"), 0) : 0;
    janas_json_free(d);
    janas_buf_free(&f);
    return x >= 1 && x <= BREBUS_LOADED ? (int)x : BREBUS_LOADED;
}

/* A free place for a model of bytes: the models used least lately
   unloaded until it fits beside the others in the memory free now (a
   model larger than the memory, its experts read from the disk, runs
   alone), and while more are loaded than the settings allow. */
static int make_room(struct brebus *b, uint64_t bytes)
{
    uint64_t total = janas_mem_total(), want = bytes + ((uint64_t)2 << 30);
    int alone = total && bytes > total / 2;
    int cap = max_loaded(b);
    for (;;) {
        int n = 0, lru = -1;
        for (int i = 0; i < BREBUS_LOADED; i++)
            if (b->srv[i].on) {
                n++;
                if (lru < 0 || b->srv[i].used < b->srv[lru].used)
                    lru = i;
            }
        if (!n)
            break;
        uint64_t avail = janas_mem_available();
        if (!alone && n < cap && (!avail || avail >= want))
            break;
        stop_locked(b, lru);
    }
    for (int i = 0; i < BREBUS_LOADED; i++)
        if (!b->srv[i].on)
            return i;
    return 0; /* not reached: one at least was unloaded */
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
    /* loaded already: the one used now, at once */
    int was = loaded(b, name);
    if (was >= 0) {
        b->srv[was].used = time(NULL);
        b->srv_cur = was;
        pthread_mutex_unlock(&b->mu);
        janas_json_free(d);
        out->n = 0;
        brebus_server_status(b, out);
        return 0;
    }
    char exe[4200], port[8], par_s[8], ctx_s[16], logp[4200], env_key[300];
    char origins[MAX_ORIGINS][256], key_now[256];
    int lan, tls, n_origins;
    server_settings(b, &lan, &tls, key_now, sizeof key_now, origins,
                    &n_origins);
    /* the servers loaded with other settings (the network, HTTPS, the
       key) go: all the models' servers answer alike */
    if (lan != b->srv_lan || tls != b->srv_tls || strcmp(key_now, b->srv_key))
        for (int i = 0; i < BREBUS_LOADED; i++)
            stop_locked(b, i);
    snprintf(b->srv_key, sizeof b->srv_key, "%s", key_now);
    b->srv_lan = lan;
    static char cert[4300], key[4300];
    b->srv_tls = 0;
    b->srv_fp[0] = 0;
    if (tls) {
        if (brebus_certificate(b, cert, key, sizeof cert, err, sizeof err) !=
            0) {
            pthread_mutex_unlock(&b->mu);
            janas_json_free(d);
            return fail(out, err);
        }
        b->srv_tls = 1;
    }
    int slot = make_room(b, m[at].bytes);
    struct brebus_srv *v = &b->srv[slot];
    snprintf(exe, sizeof exe, "%s/janas-server%s", b->bin_dir, b->suffix);
    v->port = (unsigned short)(b->port + 1 + slot);
    snprintf(port, sizeof port, "%u", v->port);
    const char *argv[48 + 2 * MAX_ORIGINS];
    int k = 0;
    argv[k++] = exe;
    argv[k++] = m[at].path;
    argv[k++] = "--port";
    argv[k++] = port;
    argv[k++] = "--monitor";
    /* the documents the API keeps (files, conversations of other
       programs): the first place's where janas-server keeps them, each
       other's beside, server-1, server-2...: a store has one server */
    static char store[BREBUS_LOADED][4300];
    if (slot) {
        snprintf(store[slot], sizeof store[slot], "%s", b->data_dir);
        char *sl = strrchr(store[slot], '/');
        if (sl) {
            snprintf(sl, sizeof store[slot] - (size_t)(sl - store[slot]),
                     "/server-%d", slot);
            argv[k++] = "--store";
            argv[k++] = store[slot];
        }
    }
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
    /* never more than the model was trained for; and its most, when that is
       less than janas-server's own 16384 */
    if (m[at].ctx >= 512 &&
        (ctx > m[at].ctx || (ctx < 512 && m[at].ctx < 16384)))
        ctx = m[at].ctx;
    if (ctx >= 512 && ctx <= (1 << 20)) {
        snprintf(ctx_s, sizeof ctx_s, "%d", (int)ctx);
        argv[k++] = "--ctx";
        argv[k++] = ctx_s;
    }
    if (th && (th->type == JANAS_JSON_FALSE || th->type == JANAS_JSON_TRUE)) {
        argv[k++] = "--think";
        argv[k++] = th->type == JANAS_JSON_TRUE ? "on" : "off";
    }
    /* the model's sampling as the user set it in My models: the server's
       for every request that does not say (janas-server checks it) */
    const struct janas_json *smp = o ? janas_json_get(o, "sampling") : NULL;
    static struct janas_buf smp_s;
    smp_s.n = 0;
    if (smp && smp->type == JANAS_JSON_OBJECT && smp->n) {
        janas_json_write(&smp_s, smp);
        janas_buf_put(&smp_s, "", 1);
        if (!smp_s.oom) {
            argv[k++] = "--sampling";
            argv[k++] = smp_s.p;
        }
    }
    /* the chat template the user wrote for the model, in a file of its
       own for janas-server (--chat-template), and the template's
       variables (--chat-template-kwargs) */
    const struct janas_json *tp = o ? janas_json_get(o, "template") : NULL;
    static char tp_path[4400];
    if (tp && tp->type == JANAS_JSON_STRING && janas_json_str(tp)[0]) {
        snprintf(tp_path, sizeof tp_path, "%s/template%s%.0d.jinja",
                 b->data_dir, slot ? "-" : "", slot);
        int tf = open(tp_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        const char *tt = janas_json_str(tp);
        size_t tn = strlen(tt);
        if (tf >= 0 && write(tf, tt, tn) == (ssize_t)tn) {
            argv[k++] = "--chat-template";
            argv[k++] = tp_path;
        }
        if (tf >= 0)
            close(tf);
    }
    const struct janas_json *kw = o ? janas_json_get(o, "kwargs") : NULL;
    static struct janas_buf kw_s;
    kw_s.n = 0;
    if (kw && kw->type == JANAS_JSON_OBJECT && kw->n) {
        janas_json_write(&kw_s, kw);
        janas_buf_put(&kw_s, "", 1);
        if (!kw_s.oom) {
            argv[k++] = "--chat-template-kwargs";
            argv[k++] = kw_s.p;
        }
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
    snprintf(logp, sizeof logp, "%s/server%s%.0d.log", b->data_dir,
             slot ? "-" : "", slot);
    int fd = open(logp, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    /* the key in its environment, or none: not one Brebus was started with */
    snprintf(env_key, sizeof env_key, "JANAS_API_KEY=%s", b->srv_key);
    /* it ends with this Brebus however this one ends (Linux; on Windows
       the Job Object does) */
    static char env_parent[40];
    snprintf(env_parent, sizeof env_parent, "JANAS_PARENT_PID=%lu",
             (unsigned long)getpid());
    const char *env[] = {env_key, env_parent, api_ok ? env_api : NULL, NULL};
    int rc =
        janas_proc_start(&v->proc, argv, env, NULL, -1, fd, fd,
                         JANAS_PROC_GROUP | BREBUS_NO_CONSOLE, err, sizeof err);
    if (fd >= 0)
        close(fd);
    if (rc != 0) {
        snprintf(b->srv_err, sizeof b->srv_err, "%s", err);
        pthread_mutex_unlock(&b->mu);
        return fail(out, err);
    }
    v->on = 1;
    v->ready = 0;
    snprintf(v->model, sizeof v->model, "%s", m[at].name);
    v->used = time(NULL);
    b->srv_cur = slot;
    b->srv_last[0] = 0;
    b->srv_err[0] = 0;
    unsigned short sport = v->port;
    char skey[256];
    snprintf(skey, sizeof skey, "%s", b->srv_key);
    int stls = b->srv_tls;
    pthread_mutex_unlock(&b->mu);
    /* ready when it answers; a large model takes a while to open. Loaded
       is said of it then, not before: a server that never answers is no
       model the chat may be pointed at */
    int up = 0;
    for (int i = 0; i < 600; i++) {
        if (answers(sport, skey, stls)) {
            pthread_mutex_lock(&b->mu);
            if (v->on && v->port == sport) {
                v->ready = up = 1;
                record_locked(b);
            }
            pthread_mutex_unlock(&b->mu);
            break;
        }
        int code;
        pthread_mutex_lock(&b->mu);
        int gone = v->on && v->port == sport &&
                   janas_proc_wait(&v->proc, 0, &code) == 1;
        if (gone) {
            v->on = 0;
            v->model[0] = 0;
            snprintf(b->srv_err, sizeof b->srv_err,
                     "janas-server ended (status %d): see %.400s", code, logp);
        }
        pthread_mutex_unlock(&b->mu);
        if (gone)
            return fail(out, b->srv_err);
        usleep(200000);
    }
    if (!up) { /* two minutes and no answer: stopped, and said */
        pthread_mutex_lock(&b->mu);
        if (v->on && v->port == sport) {
            stop_locked(b, slot);
            snprintf(b->srv_err, sizeof b->srv_err,
                     "%s did not answer in two minutes: see %.400s", m[at].name,
                     logp);
        }
        pthread_mutex_unlock(&b->mu);
        return fail(out, b->srv_err);
    }
    out->n = 0;
    brebus_server_status(b, out);
    return 0;
}

int brebus_server_stop(struct brebus *b, const char *body, size_t n,
                       struct janas_buf *out)
{
    char err[160];
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const char *name =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "model")) : NULL;
    pthread_mutex_lock(&b->mu);
    for (int i = 0; i < BREBUS_LOADED; i++)
        if (!name || !strcmp(b->srv[i].model, name))
            stop_locked(b, i);
    pthread_mutex_unlock(&b->mu);
    janas_json_free(d);
    out->n = 0;
    brebus_server_status(b, out);
    return 0;
}

/* a place's server gone by itself: noted */
static void reap_locked(struct brebus *b, int i)
{
    int code;
    struct brebus_srv *v = &b->srv[i];
    if (v->on && janas_proc_wait(&v->proc, 0, &code) == 1) {
        snprintf(b->srv_err, sizeof b->srv_err,
                 "janas-server of %.200s ended (status %d)", v->model, code);
        v->on = 0;
        v->model[0] = 0;
    }
}

void brebus_server_status(struct brebus *b, struct janas_buf *out)
{
    pthread_mutex_lock(&b->mu);
    for (int i = 0; i < BREBUS_LOADED; i++)
        reap_locked(b, i);
    int c = current(b), on = c >= 0;
    const char *sch = b->srv_tls ? "https" : "http";
    char ip[64] = "";
    if (on && b->srv_lan)
        lan_address(ip, sizeof ip);
    /* the one used last, as when there was one alone */
    janas_buf_printf(out,
                     "{\"running\": %s, \"model\": ", on ? "true" : "false");
    str(out, on ? b->srv[c].model : "");
    janas_buf_printf(out, ", \"port\": %u, \"url\": ", on ? b->srv[c].port : 0);
    char url[64] = "", lan_url[96] = "";
    if (on)
        snprintf(url, sizeof url, "%s://127.0.0.1:%u/v1", sch, b->srv[c].port);
    str(out, url);
    if (ip[0])
        snprintf(lan_url, sizeof lan_url, "%s://%s:%u/v1", sch, ip,
                 b->srv[c].port);
    janas_buf_puts(out, ", \"lan_url\": ");
    str(out, lan_url);
    /* and every model loaded, each at its own address */
    janas_buf_puts(out, ", \"loaded\": [");
    for (int i = 0, k = 0; i < BREBUS_LOADED; i++) {
        if (!b->srv[i].on || !b->srv[i].ready) /* starting: not loaded yet */
            continue;
        janas_buf_puts(out, k++ ? ", {\"model\": " : "{\"model\": ");
        str(out, b->srv[i].model);
        snprintf(url, sizeof url, "%s://127.0.0.1:%u/v1", sch, b->srv[i].port);
        janas_buf_puts(out, ", \"url\": ");
        str(out, url);
        lan_url[0] = 0;
        if (ip[0])
            snprintf(lan_url, sizeof lan_url, "%s://%s:%u/v1", sch, ip,
                     b->srv[i].port);
        janas_buf_puts(out, ", \"lan_url\": ");
        str(out, lan_url);
        janas_buf_printf(out, ", \"idle\": %ld}",
                         (long)(time(NULL) - b->srv[i].used));
    }
    janas_buf_printf(out, "], \"max_loaded\": %d", BREBUS_LOADED);
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

unsigned short brebus_server_route(struct brebus *b, const char *model,
                                   int *tls, char *key, size_t kn)
{
    pthread_mutex_lock(&b->mu);
    int i = loaded(b, model);
    if (i < 0)
        i = current(b);
    unsigned short p = 0;
    if (i >= 0) {
        reap_locked(b, i);
        if (b->srv[i].on && b->srv[i].ready) { /* one that answers alone */
            p = b->srv[i].port;
            b->srv[i].used = time(NULL); /* for the unloading after a while */
            b->srv_cur = i;
        }
    }
    *tls = b->srv_tls;
    snprintf(key, kn, "%s", b->srv_key);
    pthread_mutex_unlock(&b->mu);
    return p;
}

int brebus_server_unload_idle(struct brebus *b, const char *model)
{
    pthread_mutex_lock(&b->mu);
    int i = loaded(b, model), rc = 0;
    if (i >= 0 && i == b->srv_cur)
        rc = -1; /* the one the chat uses */
    else if (i >= 0)
        stop_locked(b, i);
    pthread_mutex_unlock(&b->mu);
    return rc;
}

int brebus_server_loaded(struct brebus *b, const char *model)
{
    pthread_mutex_lock(&b->mu);
    int i = loaded(b, model);
    pthread_mutex_unlock(&b->mu);
    return i >= 0;
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
        for (int i = 0; i < BREBUS_LOADED; i++)
            if (b->srv[i].on && time(NULL) - b->srv[i].used > ttl * 60) {
                snprintf(b->srv_last, sizeof b->srv_last, "%s",
                         b->srv[i].model);
                stop_locked(b, i);
                snprintf(b->srv_err, sizeof b->srv_err,
                         "unloaded after %ld minutes without a request: "
                         "loaded again at the next message",
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
    int c = current(b), on = c >= 0;
    snprintf(url, sizeof url, "%s://127.0.0.1:%u/monitor/events",
             b->srv_tls ? "https" : "http", on ? b->srv[c].port : 0);
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

/* the replies a model's server writes, or reads the prompt of, stopped
   (POST /janas/cancel); the server stays */
static void cancel_on(unsigned short port, const char *key, int tls)
{
    char url[96], auth[320], err[200];
    snprintf(url, sizeof url, "%s://127.0.0.1:%u/janas/cancel",
             tls ? "https" : "http", port);
    snprintf(auth, sizeof auth, "Authorization: Bearer %s", key);
    const char *hdr[] = {auth};
    struct janas_url u;
    struct janas_http h;
    if (janas_url_parse(url, &u, err, sizeof err) == 0 &&
        janas_http_request(&h, &u, "POST", hdr, key[0] ? 1 : 0, "{}", 2, 5000,
                           NULL, err, sizeof err) == 0)
        janas_http_close(&h);
}

void brebus_server_cancel_all(struct brebus *b)
{
    unsigned short ports[BREBUS_LOADED];
    int n = 0, tls;
    char key[256];
    pthread_mutex_lock(&b->mu);
    for (int i = 0; i < BREBUS_LOADED; i++)
        if (b->srv[i].on && b->srv[i].ready)
            ports[n++] = b->srv[i].port;
    snprintf(key, sizeof key, "%s", b->srv_key);
    tls = b->srv_tls;
    pthread_mutex_unlock(&b->mu);
    for (int i = 0; i < n; i++)
        cancel_on(ports[i], key, tls);
}

/* pid's program the janas-server beside this Brebus: 1, or 0 */
static int our_server(const struct brebus *b, unsigned long pid)
{
    char want[4300];
    snprintf(want, sizeof want, "%s/janas-server%s", b->bin_dir, b->suffix);
#ifdef _WIN32
    HANDLE h =
        OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    char got[4300];
    DWORD n = sizeof got;
    int ok = h && QueryFullProcessImageNameA(h, 0, got, &n);
    if (h)
        CloseHandle(h);
    if (!ok)
        return 0;
    for (char *c = got; *c; c++)
        if (*c == '\\')
            *c = '/';
    for (char *c = want; *c; c++)
        if (*c == '\\')
            *c = '/';
    return !_stricmp(got, want);
#else
    char link[64], got[4300], real[4300];
    snprintf(link, sizeof link, "/proc/%lu/exe", pid);
    ssize_t n = readlink(link, got, sizeof got - 1);
    if (n <= 0 || !realpath(want, real))
        return 0;
    got[n] = 0;
    return !strcmp(got, real);
#endif
}

void brebus_server_adopt(struct brebus *b)
{
    char p[4300], err[200];
    snprintf(p, sizeof p, "%s/servers.json", b->data_dir);
    FILE *f = fopen(p, "rb");
    if (!f)
        return;
    struct janas_buf t = {0};
    char buf[4096];
    size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0 && t.n < (1 << 20))
        janas_buf_put(&t, buf, r);
    fclose(f);
    struct janas_json_doc *d =
        t.n ? janas_json_parse(t.p, t.n, err, sizeof err) : NULL;
    janas_buf_free(&t);
    const struct janas_json *a =
        d ? janas_json_get(janas_json_root(d), "servers") : NULL;
    pthread_mutex_lock(&b->mu);
    /* the settings they were started with, as a start reads them */
    char origins[MAX_ORIGINS][256], key_now[256];
    int lan, tls, n_origins;
    server_settings(b, &lan, &tls, key_now, sizeof key_now, origins,
                    &n_origins);
    snprintf(b->srv_key, sizeof b->srv_key, "%s", key_now);
    b->srv_lan = lan;
    b->srv_tls = 0;
    if (tls) {
        static char cert[4300], key[4300];
        if (brebus_certificate(b, cert, key, sizeof cert, err, sizeof err) == 0)
            b->srv_tls = 1;
    }
    for (const struct janas_json *e =
             a && a->type == JANAS_JSON_ARRAY ? a->child : NULL;
         e; e = e->next) {
        double slot = janas_json_num(janas_json_get(e, "slot"), -1),
               pid = janas_json_num(janas_json_get(e, "pid"), 0),
               port = janas_json_num(janas_json_get(e, "port"), 0);
        const char *model = janas_json_str(janas_json_get(e, "model"));
        if (slot < 0 || slot >= BREBUS_LOADED || pid < 1 || !model ||
            strlen(model) >= sizeof b->srv[0].model ||
            port != b->port + 1 + slot || b->srv[(int)slot].on)
            continue;
        struct brebus_srv *v = &b->srv[(int)slot];
        /* only a janas-server of this Brebus's, still there */
        if (!our_server(b, (unsigned long)pid) ||
            janas_proc_adopt(&v->proc, (unsigned long)pid, 1) != 0)
            continue;
        if (!answers((unsigned short)port, b->srv_key, b->srv_tls)) {
            /* ours, but deaf (other settings, or stuck): it goes */
            janas_proc_kill(&v->proc, 1);
            janas_proc_wait(&v->proc, 3000, NULL);
            memset(&v->proc, 0, sizeof v->proc);
            continue;
        }
        v->on = v->ready = 1;
        v->port = (unsigned short)port;
        snprintf(v->model, sizeof v->model, "%s", model);
        v->used = time(NULL);
        b->srv_cur = (int)slot;
        /* what it was writing for a Brebus of before: stopped */
        cancel_on(v->port, b->srv_key, b->srv_tls);
        fprintf(stderr, "brebus: %s's server, still there, taken up again\n",
                model);
    }
    record_locked(b);
    pthread_mutex_unlock(&b->mu);
    janas_json_free(d);
}

void brebus_server_start_last(struct brebus *b)
{
    pthread_mutex_lock(&b->mu);
    int running = 0;
    for (int i = 0; i < BREBUS_LOADED; i++)
        running |= b->srv[i].on;
    pthread_mutex_unlock(&b->mu);
    if (running) /* one taken up again already */
        return;
    struct janas_buf f = {0}, body = {0}, out = {0};
    char err[160];
    brebus_settings_get(b, &f);
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *name = o ? janas_json_str(janas_json_get(o, "_last")) : NULL;
    if (!name || !*name) {
        fprintf(stderr, "brebus: no model used yet: none loaded\n");
        goto done;
    }
    /* its settings, as the page sends them when it loads it */
    const struct janas_json *s = janas_json_get(o, name);
    janas_buf_puts(&body, "{\"model\": ");
    str(&body, name);
    static const char *const NUM[] = {"context", "parallel"};
    for (size_t i = 0; s && i < sizeof NUM / sizeof *NUM; i++) {
        double v = janas_json_num(janas_json_get(s, NUM[i]), 0);
        if (v > 0)
            janas_buf_printf(&body, ", \"%s\": %.0f", NUM[i], v);
    }
    const struct janas_json *th = s ? janas_json_get(s, "think") : NULL;
    if (th && (th->type == JANAS_JSON_TRUE || th->type == JANAS_JSON_FALSE))
        janas_buf_printf(&body, ", \"think\": %s",
                         th->type == JANAS_JSON_TRUE ? "true" : "false");
    const struct janas_json *smp = s ? janas_json_get(s, "sampling") : NULL;
    if (smp && smp->type == JANAS_JSON_OBJECT) {
        janas_buf_puts(&body, ", \"sampling\": ");
        janas_json_write(&body, smp);
    }
    const struct janas_json *kw = s ? janas_json_get(s, "kwargs") : NULL;
    if (kw && kw->type == JANAS_JSON_OBJECT) {
        janas_buf_puts(&body, ", \"kwargs\": ");
        janas_json_write(&body, kw);
    }
    const struct janas_json *tp = s ? janas_json_get(s, "template") : NULL;
    if (tp && tp->type == JANAS_JSON_STRING) {
        janas_buf_puts(&body, ", \"template\": ");
        janas_json_write(&body, tp);
    }
    static const char *const STR[] = {"prompt_on", "drafts_on", "reply_on",
                                      "gpu_api"};
    for (size_t i = 0; s && i < sizeof STR / sizeof *STR; i++) {
        const char *v = janas_json_str(janas_json_get(s, STR[i]));
        if (v) {
            janas_buf_printf(&body, ", \"%s\": ", STR[i]);
            str(&body, v);
        }
    }
    janas_buf_puts(&body, "}");
    if (brebus_server_start(b, body.p, body.n, &out) == 0)
        fprintf(stderr, "brebus: %s loaded for other programs\n", name);
    else
        fprintf(stderr, "brebus: %s not loaded: %.*s\n", name, (int)out.n,
                out.p ? out.p : "");
done:
    janas_json_free(d);
    janas_buf_free(&f);
    janas_buf_free(&body);
    janas_buf_free(&out);
}
