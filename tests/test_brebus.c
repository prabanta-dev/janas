/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_brebus.c - Brebus (src/brebus/) as the page uses it, without a
 * model: brebus, the one beside this program, started on a free port with
 * an empty models' folder and a data folder of its own; its page, the
 * machine, the models (none) and the catalog; another site's page refused;
 * a model that is not there refused; /v1 answered 503 while no model is
 * loaded; and brebus ending on SIGTERM. Linux only, as brebus.
 */
#ifndef __linux__
#include <stdio.h>
int main(void)
{
    printf("test_brebus: skipped, Brebus is built on Linux only\n");
    return 0;
}
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/catalog.h"
#include "common/mcp_http.h"
#include "common/proc.h"
#include "common/sysinfo.h"

static int failures;

#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static int port;

static int free_port(void)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET,
                            .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    socklen_t al = sizeof a;
    int p = -1;
    if (fd >= 0 && bind(fd, (struct sockaddr *)&a, sizeof a) == 0 &&
        getsockname(fd, (struct sockaddr *)&a, &al) == 0)
        p = ntohs(a.sin_port);
    if (fd >= 0)
        close(fd);
    return p;
}

/* status of method path, its body in out; -1 when nothing answered */
static int ask(const char *method, const char *path, const char *body,
               const char *origin, struct janas_buf *out)
{
    char url[256], err[160], hdr[300];
    snprintf(url, sizeof url, "http://127.0.0.1:%d%s", port, path);
    struct janas_url u;
    if (janas_url_parse(url, &u, err, sizeof err) != 0)
        return -1;
    const char *h[2] = {"Content-Type: application/json", hdr};
    snprintf(hdr, sizeof hdr, "Origin: %s", origin ? origin : "");
    struct janas_http c;
    if (janas_http_request(&c, &u, method, h, origin ? 2 : 1, body,
                           body ? strlen(body) : 0, 10000, NULL, err,
                           sizeof err) != 0)
        return -1;
    out->n = 0;
    while (janas_http_body(&c, out, 10000) == 1)
        ;
    janas_buf_put(out, "", 1);
    int st = c.head.status;
    janas_http_close(&c);
    return st;
}

int main(void)
{
    char exe[4096], dir[4096];
    if (janas_exe_path(exe, sizeof exe) <= 0) {
        printf("test_brebus: where am I?\n");
        return 1;
    }
    /* test_brebus-asan beside brebus-asan */
    char *slash = strrchr(exe, '/'), *dash = strchr(slash, '-');
    char suffix[16] = "";
    if (dash)
        snprintf(suffix, sizeof suffix, "%s", dash);
    *slash = 0;
    char brebus[4200];
    snprintf(brebus, sizeof brebus, "%s/brebus%s", exe, suffix);
    if (access(brebus, X_OK) != 0) {
        printf("test_brebus: %s is not there\n", brebus);
        return 1;
    }
    const char *home = getenv("HOME");
    snprintf(dir, sizeof dir, "%s/tmp/janas-test-brebus-%d", home ? home : ".",
             (int)getpid());
    char models[4200], data[4200], env_m[4300], env_d[4300], port_s[8];
    snprintf(models, sizeof models, "%s/models", dir);
    snprintf(data, sizeof data, "%s/data", dir);
    janas_mkdirs(models);
    janas_mkdirs(data);
    snprintf(env_m, sizeof env_m, "JANAS_MODELS=%s", models);
    snprintf(env_d, sizeof env_d, "XDG_DATA_HOME=%s", data);
    port = free_port();
    snprintf(port_s, sizeof port_s, "%d", port);
    const char *argv[] = {brebus, "--no-open", "--port", port_s, NULL};
    const char *env[] = {env_m, env_d, NULL};
    int quiet = janas_null_fd(1);
    struct janas_proc p;
    char err[256];
    if (port < 0 || janas_proc_start(&p, argv, env, NULL, -1, quiet, quiet, 0,
                                     err, sizeof err) != 0) {
        printf("test_brebus: brebus did not start: %s\n", err);
        return 1;
    }
    close(quiet);
    struct janas_buf o = {0};
    int st = -1;
    for (int i = 0; i < 100 && st < 0; i++) {
        st = ask("GET", "/", NULL, NULL, &o);
        if (st < 0)
            usleep(50000);
    }
    CHECK(st == 200 && strstr(o.p, "<title>Brebus</title>"), "the page: %d",
          st);
    st = ask("GET", "/brebus/machine", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"memory_total\""), "machine: %d %s", st,
          o.p);
    st = ask("GET", "/brebus/models", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"models\": []"), "models: %d %s", st, o.p);
    st = ask("GET", "/brebus/catalog", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"fits\""), "catalog: %d", st);
    st = ask("GET", "/brebus/nothing", NULL, NULL, &o);
    CHECK(st == 404, "an unknown question: %d", st);
    /* another site's page, and Brebus's own */
    char own[64];
    snprintf(own, sizeof own, "http://127.0.0.1:%d", port);
    st = ask("POST", "/brebus/server/start", "{\"model\": \"x\"}",
             "https://evil.example", &o);
    CHECK(st == 403, "another site's page let in: %d", st);
    st = ask("GET", "/brebus/models", NULL, "http://localhost:1", &o);
    CHECK(st == 403, "another port's page let in: %d", st);
    st = ask("POST", "/brebus/server/start", "{\"model\": \"x\"}", own, &o);
    CHECK(st == 400 && strstr(o.p, "no such model"),
          "a model that is not there: %d %s", st, o.p);
    st = ask("POST", "/brebus/server/start", "{}", own, &o);
    CHECK(st == 400, "no model named: %d", st);
    st = ask("GET", "/brebus/server", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"running\": false"), "server: %d %s", st,
          o.p);
    st = ask("POST", "/v1/chat/completions", "{\"messages\": []}", own, &o);
    CHECK(st == 503, "/v1 with no model: %d", st);
    st = ask("GET", "/api/tags", NULL, NULL, &o);
    CHECK(st == 503, "/api with no model: %d", st);

    janas_proc_kill(&p, 0);
    int code = -2;
    int ended = janas_proc_wait(&p, 10000, &code);
    if (ended != 1) {
        janas_proc_kill(&p, 1);
        janas_proc_wait(&p, -1, NULL);
    }
    CHECK(ended == 1 && code == 0, "brebus did not end well: %d %d", ended,
          code);
    janas_buf_free(&o);
    char rm[4300];
    snprintf(rm, sizeof rm, "%s/data/janas/brebus", dir);
    rmdir(rm);
    snprintf(rm, sizeof rm, "%s/data/janas", dir);
    rmdir(rm);
    rmdir(data);
    rmdir(models);
    rmdir(dir);
    if (!failures)
        printf("test_brebus: ok\n");
    return failures != 0;
}
#endif
