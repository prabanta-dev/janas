/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_brebus.c - Brebus (src/brebus/) as the page uses it, without a
 * model: brebus, the one beside this program, started on a free port with
 * an empty models' folder and a data folder of its own; its page, the
 * machine, the models (none) and the catalog; another site's page refused;
 * a model that is not there refused, to load or to download;
 * conversations saved, listed, read and deleted, and the settings, with
 * bad ids and what is not a JSON object refused; exports, never over a
 * file; the logo; /v1 answered 503 while no model is loaded; and brebus
 * ending on SIGTERM. Linux only, as brebus.
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
#include <errno.h>
#include <ftw.h>
#include <sys/stat.h>
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

static int rm_one(const char *p, const struct stat *st, int kind, struct FTW *f)
{
    (void)st, (void)kind, (void)f;
    return remove(p) != 0 && errno != ENOENT ? -1 : 0;
}

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
    /* under ThreadSanitizer: the races libmicrohttpd has of its own (its
       daemon stopped while its thread polls, as with janas-server) are not
       Brebus's; any other one still fails */
    char supp[4300], env_t[4400], env_h[4300], env_c[4300];
    /* a home of its own: an export goes to its home, which has no
       Downloads folder, and not to the user's */
    snprintf(env_h, sizeof env_h, "HOME=%s", dir);
    snprintf(env_c, sizeof env_c, "XDG_CONFIG_HOME=%s/none", dir);
    snprintf(supp, sizeof supp, "%s/tsan.supp", dir);
    FILE *sf = fopen(supp, "w");
    if (sf) {
        fputs("race:src/third_party/libmicrohttpd/\n", sf);
        fclose(sf);
    }
    snprintf(env_t, sizeof env_t, "TSAN_OPTIONS=suppressions=%s", supp);
    port = free_port();
    snprintf(port_s, sizeof port_s, "%d", port);
    const char *argv[] = {brebus, "--no-open", "--port", port_s, NULL};
    const char *env[] = {env_m, env_d, env_t, env_h, env_c, NULL};
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
    st = ask("POST", "/brebus/get/start", "{\"model\": \"x\"}", own, &o);
    CHECK(st == 400 && strstr(o.p, "no such model"),
          "a download of a model not in the catalog: %d %s", st, o.p);
    /* files of Hugging Face: only hf:<owner>/<repo>/<file>.gguf or .jns,
       never a path out of the models' folder nor an option */
    static const char *const bad_hf[] = {
        "hf:a/b/../../x.gguf", "hf:a/b/x.txt",   "hf:a/x.gguf",
        "hf:-a/b/x.gguf",      "hf:a/b/./x.jns", "hf:a/b c/x.gguf",
        "hf:a//b/x.gguf"};
    for (size_t i = 0; i < sizeof bad_hf / sizeof *bad_hf; i++) {
        char body[160];
        snprintf(body, sizeof body, "{\"model\": \"%s\"}", bad_hf[i]);
        st = ask("POST", "/brebus/get/start", body, own, &o);
        CHECK(st == 400 && strstr(o.p, "not a file of Hugging Face"),
              "%s taken: %d %s", bad_hf[i], st, o.p);
    }
    st = ask("GET", "/brebus/get", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"running\": false"), "get: %d %s", st,
          o.p);
    /* the workspace: none yet (not started from a terminal), so no
       conversation can be kept; too wide ones refused; one opened */
    st = ask("POST", "/brebus/chats/a-1", "{}", own, &o);
    CHECK(st == 400, "a conversation kept with no workspace: %d", st);
    st = ask("GET", "/brebus/workspace", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"path\": \"\""), "no workspace: %s", o.p);
    st = ask("POST", "/brebus/workspace/open", "{\"path\": \"/\"}", own, &o);
    CHECK(st == 400 && strstr(o.p, "too wide"), "/ taken: %d %s", st, o.p);
    char body[4600], want[4600];
    snprintf(body, sizeof body, "{\"path\": \"%s\"}", dir);
    st = ask("POST", "/brebus/workspace/open", body, own, &o);
    CHECK(st == 400 && strstr(o.p, "too wide"), "the home taken: %d %s", st,
          o.p);
    st =
        ask("POST", "/brebus/workspace/open", "{\"path\": \"rel/x\"}", own, &o);
    CHECK(st == 400, "a relative path taken: %d", st);
    st = ask("POST", "/brebus/workspace/new", "{}", own, &o);
    snprintf(want, sizeof want, "\"path\": \"%s/Documents/Brebus/", dir);
    CHECK(st == 200 && strstr(o.p, want), "a new workspace: %d %s", st, o.p);
    snprintf(body, sizeof body, "{\"path\": \"%s/work\", \"create\": true}",
             dir);
    st = ask("POST", "/brebus/workspace/open", body, own, &o);
    snprintf(want, sizeof want, "\"path\": \"%s/work\"", dir);
    CHECK(st == 200 && strstr(o.p, want) && strstr(o.p, "\"recent\": [\""),
          "a workspace opened: %d %s", st, o.p);
    snprintf(want, sizeof want, "%s/work/.brebus/chats", dir);
    CHECK(access(want, W_OK) == 0, "no %s", want);

    /* the workspace's tools (janas-docs and janas-files beside brebus),
       a folder read only, a file attached */
    st = ask("GET", "/brebus/tools", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"docs_search\"") &&
              strstr(o.p, "\"files_write\"") &&
              strstr(o.p, "\"read_only\": [\""),
          "the tools: %d %.300s", st, o.p);
    char ro[4300], note[4400];
    snprintf(ro, sizeof ro, "%s/ro", dir);
    janas_mkdirs(ro);
    snprintf(note, sizeof note, "%s/note.txt", ro);
    FILE *nf = fopen(note, "w");
    if (nf) {
        fputs("a note of the folder read only\n", nf);
        fclose(nf);
    }
    snprintf(body, sizeof body, "{\"read_only\": [\"%s\", \"/nowhere\"]}", ro);
    st = ask("POST", "/brebus/folders", body, own, &o);
    CHECK(st == 200 && strstr(o.p, ro) && strstr(o.p, "\"refused\": true"),
          "a folder read only: %d %s", st, o.p);
    snprintf(body, sizeof body,
             "{\"name\": \"files_write\", \"arguments\": {\"path\": "
             "\"%s/x.txt\", \"content\": \"x\"}}",
             ro);
    st = ask("POST", "/brebus/tools/call", body, own, &o);
    CHECK(st == 200 && strstr(o.p, "read only") &&
              strstr(o.p, "\"is_error\": true"),
          "written where it may only be read: %d %s", st, o.p);
    st = ask("POST", "/brebus/tools/call",
             "{\"name\": \"files_write\", \"arguments\": {\"path\": "
             "\"w.txt\", \"content\": \"ok\"}}",
             own, &o);
    snprintf(want, sizeof want, "%s/work/w.txt", dir);
    CHECK(st == 200 && access(want, F_OK) == 0,
          "written in the workspace: %d %s", st, o.p);
    snprintf(body, sizeof body, "{\"source\": \"%s\", \"mode\": \"copy\"}",
             note);
    st = ask("POST", "/brebus/attach", body, own, &o);
    snprintf(want, sizeof want, "%s/work/sources/note.txt", dir);
    CHECK(st == 200 && strstr(o.p, want) &&
              strstr(o.p, "\"text\": \"a note of the folder read only\""),
          "a file attached: %d %s", st, o.p);
    /* python in the cage, where one can be built: the workspace written,
       the folder read only not */
    st = ask("GET", "/brebus/tools", NULL, NULL, &o);
    if (st == 200 && strstr(o.p, "\"python\"")) {
        st = ask("POST", "/brebus/tools/call",
                 "{\"name\": \"python\", \"arguments\": {\"code\": "
                 "\"open('py.txt', 'w').write('from the cage'); 6 * 7\"}}",
                 own, &o);
        snprintf(want, sizeof want, "%s/work/py.txt", dir);
        CHECK(st == 200 && strstr(o.p, "42") && access(want, F_OK) == 0,
              "python in the workspace: %d %s", st, o.p);
        snprintf(body, sizeof body,
                 "{\"name\": \"python\", \"arguments\": {\"code\": "
                 "\"open('%s/x.txt', 'w')\"}}",
                 ro);
        st = ask("POST", "/brebus/tools/call", body, own, &o);
        CHECK(st == 200 && strstr(o.p, "Read-only file system"),
              "python writing where it may only read: %d %s", st, o.p);
    } else
        printf("test_brebus: no cage here, python not tried\n");
    /* artifacts: saved with their versions, the same one not twice */
    static const char *const arts[] = {
        "{\"title\": \"My page!\", \"type\": \"html\", \"content\": \"<p>1</p>\"}",
        "{\"title\": \"My page!\", \"type\": \"html\", \"content\": \"<p>1</p>\"}",
        "{\"title\": \"My page!\", \"type\": \"html\", \"content\": \"<p>2</p>\"}",
        "{\"title\": \"../x\", \"type\": \"code\", \"lang\": \"python\", "
        "\"content\": \"print(1)\"}"};
    static const char *const wants[] = {
        "artifact/My page/v1.html\", \"version\": 1",
        "artifact/My page/v1.html\", \"version\": 1, \"same\": true",
        "artifact/My page/v2.html\", \"version\": 2", "artifact/x/v1.py\""};
    for (int i = 0; i < 4; i++) {
        st = ask("POST", "/brebus/artifacts", arts[i], own, &o);
        CHECK(st == 200 && strstr(o.p, wants[i]), "artifact %d: %d %s", i, st,
              o.p);
    }
    st = ask("POST", "/brebus/open", "{\"path\": \"/etc/passwd\"}", own, &o);
    CHECK(st == 400, "a file outside the workspace opened: %d", st);
    /* the models' folder changed, one that is not there refused; Brebus
       started at login, then not (in the test's own configuration) */
    snprintf(body, sizeof body, "{\"path\": \"%s\"}", ro);
    st = ask("POST", "/brebus/models-dir", body, own, &o);
    CHECK(st == 200, "the models' folder changed: %d %s", st, o.p);
    st = ask("GET", "/brebus/models", NULL, NULL, &o);
    snprintf(want, sizeof want, "\"dir\": \"%s\"", ro);
    CHECK(st == 200 && strstr(o.p, want), "the models' folder: %s", o.p);
    st = ask("POST", "/brebus/models-dir", "{\"path\": \"/nowhere\"}", own, &o);
    CHECK(st == 400, "a folder not there taken for the models: %d", st);
    st = ask("POST", "/brebus/autostart", "{\"on\": true}", own, &o);
    snprintf(want, sizeof want, "%s/none/autostart/brebus.desktop", dir);
    CHECK(st == 200 && strstr(o.p, "\"autostart\": true") &&
              access(want, F_OK) == 0,
          "started at login: %d %s", st, o.p);
    st = ask("POST", "/brebus/autostart", "{\"on\": false}", own, &o);
    CHECK(st == 200 && access(want, F_OK) != 0, "not started at login: %d", st);
    st = ask("POST", "/brebus/attach", "{\"source\": \"relative.txt\"}", own,
             &o);
    CHECK(st == 400, "a relative path attached: %d", st);

    /* conversations and settings, on disk */
    st = ask("POST", "/brebus/chats/a-1",
             "{\"title\": \"one\", \"updated\": 1, \"messages\": []}", own, &o);
    CHECK(st == 200, "a conversation saved: %d %s", st, o.p);
    st = ask("POST", "/brebus/chats/b_2",
             "{\"title\": \"two\", \"updated\": 2, \"model\": \"m\", "
             "\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}",
             own, &o);
    CHECK(st == 200, "another saved: %d", st);
    st = ask("GET", "/brebus/chats", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"id\": \"b_2\"") &&
              strstr(o.p, "\"messages\": 1") &&
              strstr(o.p, "b_2") < strstr(o.p, "a-1"),
          "the list, newest first: %d %s", st, o.p);
    st = ask("GET", "/brebus/chats/a-1", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"title\": \"one\""), "one read: %d %s", st,
          o.p);
    st = ask("POST", "/brebus/chats/a-1", "[1, 2]", own, &o);
    CHECK(st == 400, "an array taken for a conversation: %d", st);
    st = ask("POST", "/brebus/chats/a-1", "{\"title\": ", own, &o);
    CHECK(st == 400, "broken JSON taken: %d", st);
    st = ask("GET", "/brebus/chats/a-1", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"title\": \"one\""),
          "a refused save touched the file: %s", o.p);
    static const char *const bad[] = {"/brebus/chats/..%2fx",
                                      "/brebus/chats/a.b", "/brebus/chats/",
                                      "/brebus/chats/a%00"};
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
        st = ask("POST", bad[i], "{}", own, &o);
        CHECK(st == 400, "%s taken: %d", bad[i], st);
    }
    /* searched, in every workspace: by title, by a message, any case */
    st = ask("GET", "/brebus/chats/search?q=TWO", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"id\": \"b_2\"") &&
              !strstr(o.p, "\"id\": \"a-1\""),
          "searched by title: %d %s", st, o.p);
    st = ask("GET", "/brebus/chats/search?q=hi", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"snippet\": \"hi\"") &&
              strstr(o.p, "\"workspace_name\": \"work\""),
          "searched by a message: %d %s", st, o.p);
    st = ask("GET", "/brebus/chats/search?q=nowhere-at-all", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "\"chats\": []"), "nothing found: %s", o.p);
    st = ask("DELETE", "/brebus/chats/a-1", NULL, own, &o);
    CHECK(st == 200, "a conversation deleted: %d", st);
    st = ask("GET", "/brebus/chats/a-1", NULL, NULL, &o);
    CHECK(st == 404, "a deleted conversation read: %d", st);
    st = ask("DELETE", "/brebus/chats/b_2", NULL, "https://evil.example", &o);
    CHECK(st == 403, "another site's page deleted one: %d", st);
    st = ask("GET", "/brebus/settings", NULL, NULL, &o);
    CHECK(st == 200 && !strstr(o.p, "\"m\""), "no settings yet: %s", o.p);
    st = ask("POST", "/brebus/settings", "{\"m\": {\"context\": 8192}}", own,
             &o);
    st = ask("GET", "/brebus/settings", NULL, NULL, &o);
    CHECK(st == 200 && strstr(o.p, "8192"), "settings kept: %s", o.p);
    /* an export: a name of its own, never over a file */
    st = ask("POST", "/brebus/export",
             "{\"name\": \"../a/b..x\", \"text\": \"# hi\"}", own, &o);
    snprintf(want, sizeof want, "\"path\": \"%s/work/exports/abx.md\"", dir);
    CHECK(st == 200 && strstr(o.p, want), "an export: %d %s", st, o.p);
    st = ask("POST", "/brebus/export", "{\"name\": \"abx\", \"text\": \"2\"}",
             own, &o);
    snprintf(want, sizeof want, "\"path\": \"%s/work/exports/abx (2).md\"",
             dir);
    CHECK(st == 200 && strstr(o.p, want), "a second export: %d %s", st, o.p);
    st = ask("POST", "/brebus/export", "{\"name\": \"x\"}", own, &o);
    CHECK(st == 400, "an export of nothing: %d", st);
    st = ask("GET", "/prabanta.png", NULL, NULL, &o);
    CHECK(st == 200 && o.n > 8 && !memcmp(o.p, "\x89PNG", 4), "the logo: %d",
          st);
    st = ask("POST", "/v1/chat/completions", "{\"messages\": []}", own, &o);
    CHECK(st == 503, "/v1 with no model: %d", st);
    st = ask("GET", "/api/tags", NULL, NULL, &o);
    CHECK(st == 503, "/api with no model: %d", st);
    st = ask("GET", "/monitor", NULL, NULL, &o);
    CHECK(st == 503, "/monitor with no model: %d", st);
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
    /* what it wrote, all of it: the test's own folder */
    nftw(dir, rm_one, 16, FTW_DEPTH | FTW_PHYS);
    if (!failures)
        printf("test_brebus: ok\n");
    return failures != 0;
}
#endif
