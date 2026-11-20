/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * tools.c - the tools a model of Brebus's chat may call, for the workspace
 * open: janas-docs (its documents, searched and read) and janas-files (its
 * files, listed, read, searched, written), each started by Brebus as an MCP
 * server of its own (libjanas_mcp) and started again when the workspace or
 * its folders change.
 *
 * janas-docs indexes the workspace and the folders added to it, its list and
 * its index in the workspace's .brebus/ (JANAS_DOCS_DIRS, JANAS_DOCS_CONF
 * and XDG_DATA_HOME pointed there); janas-files reaches the workspace to read
 * and write and the folders added only to read (JANAS_FILES_WORKSPACE,
 * JANAS_FILES_READ_ONLY), not the rest of the home. The folders added are
 * kept in .brebus/workspace.json.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "brebus.h"
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif
#include "common/catalog.h"
#include "common/mcp_config.h"
#include "common/mcp_oauth.h"
#include "common/mcp_services.h"
#include "common/model_scan.h"
#include "janas/mcp.h"

struct janas_buf;
static void svc_scan(struct brebus *b);
static void svc_tools(struct janas_buf *out, int *first, struct janas_buf *cat);
static void svc_read_only(struct janas_buf *out, int *k);
static void svc_servers(struct janas_buf *out);
static int svc_call(struct brebus *b, const char *name, const char *args,
                    size_t an, struct janas_buf *out);

void brebus_service_exe(const struct brebus *b, const char *name, char *out,
                        size_t n)
{
    /* in services/ beside brebus (the binary package, as janas-chat finds
       them), else beside it (a build) */
    snprintf(out, n, "%s/services/janas-%s%s", b->bin_dir, name, b->suffix);
    if (access(out, X_OK) != 0)
        snprintf(out, n, "%s/janas-%s%s", b->bin_dir, name, b->suffix);
}

#define MAX_RO 16
#define BIG (1 << 20)

static void str(struct janas_buf *o, const char *s)
{
    janas_json_write_str(o, s, strlen(s));
}

/* the folders added to be read, from .brebus/workspace.json */
static int read_only(const char *ws, char dirs[][4096], int max)
{
    char p[4300], err[160];
    snprintf(p, sizeof p, "%s/.brebus/workspace.json", ws);
    struct janas_buf f = {0};
    FILE *in = fopen(p, "rb");
    if (in) {
        char buf[8192];
        size_t r;
        while ((r = fread(buf, 1, sizeof buf, in)) > 0 && f.n < BIG)
            janas_buf_put(&f, buf, r);
        fclose(in);
    }
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *a =
        d ? janas_json_get(janas_json_root(d), "read_only") : NULL;
    int n = 0;
    for (const struct janas_json *e =
             a && a->type == JANAS_JSON_ARRAY ? a->child : NULL;
         e && n < max; e = e->next) {
        const char *s = janas_json_str(e);
        struct stat st;
        if (s && janas_path_abs(s) && strlen(s) < 4096 &&
            !strchr(s, JANAS_PATH_LIST_SEP) && !strchr(s, '\n') &&
            stat(s, &st) == 0)
            snprintf(dirs[n++], 4096, "%s", s);
    }
    janas_json_free(d);
    janas_buf_free(&f);
    return n;
}

/* the embedding models of the models' folder (their names have
   "embedding"), sorted: their number, names into names */
static int embedders(const struct brebus *b, char names[][128], int max)
{
    DIR *d = opendir(b->models_dir);
    int n = 0;
    struct dirent *de;
    while (d && n < max && (de = readdir(d))) {
        size_t l = strlen(de->d_name);
        if (l < 5 || l - 4 >= 128 || strcmp(de->d_name + l - 4, ".jns") ||
            !strstr(de->d_name, "embedding"))
            continue;
        memcpy(names[n], de->d_name, l - 4);
        names[n][l - 4] = 0;
        n++;
    }
    if (d)
        closedir(d);
    qsort(names, (size_t)n, 128, (int (*)(const void *, const void *))strcmp);
    return n;
}

/* the one chosen: settings.json's _docs.embedding ("none": words only),
   else the first there is; "" for none */
/* The model that reads the pictures of scanned pages for janas-docs: the
   smallest of My models that sees pictures, but those below a gigabyte
   (Qwen3.5-0.8B, a model for the engine's tests, misreads them); "" with
   none. */
static void ocr_chosen(struct brebus *b, char *out, size_t cap)
{
    static struct janas_model_file m[JANAS_MODELS_MAX];
    int n = janas_models_scan(b->models_dir, m, JANAS_MODELS_MAX), at = -1;
    for (int i = 0; i < n; i++)
        if (m[i].bytes >= ((uint64_t)1 << 30) &&
            brebus_model_sees(b->models_dir, m[i].name) &&
            (at < 0 || m[i].bytes < m[at].bytes))
            at = i;
    snprintf(out, cap, "%s", at >= 0 ? m[at].path : "");
}

static void embedder_chosen(struct brebus *b, char *out, size_t cap)
{
    static char names[16][128];
    int n = embedders(b, names, 16);
    out[0] = 0;
    struct janas_buf f = {0};
    char err[160];
    brebus_settings_get(b, &f);
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *dc =
        d ? janas_json_get(janas_json_root(d), "_docs") : NULL;
    const char *want =
        dc ? janas_json_str(janas_json_get(dc, "embedding")) : NULL;
    if (want && !strcmp(want, "none")) {
        janas_json_free(d);
        janas_buf_free(&f);
        return;
    }
    for (int i = 0; i < n && !out[0]; i++)
        if (!want || !strcmp(want, names[i]))
            snprintf(out, cap, "%.127s", names[i]);
    if (!out[0] && n) /* the one asked for is gone: the first */
        snprintf(out, cap, "%.127s", names[0]);
    janas_json_free(d);
    janas_buf_free(&f);
}

static void stop_locked(struct brebus *b)
{
    svc_scan(b);          /* closes those started, finds them again */
    brebus_python_stop(); /* its folders may change */
    if (b->docs)
        janas_mcp_close(b->docs);
    if (b->files)
        janas_mcp_close(b->files);
    b->docs = b->files = NULL;
}

void brebus_tools_stop(struct brebus *b)
{
    pthread_mutex_lock(&b->tools_mu);
    stop_locked(b);
    pthread_mutex_unlock(&b->tools_mu);
}

int brebus_tools_start(struct brebus *b)
{
    char ws[4096];
    if (brebus_ws_path(b, ws, sizeof ws) != 0)
        return -1;
    static char ro[MAX_RO][4096];
    pthread_mutex_lock(&b->tools_mu);
    stop_locked(b);
    int n = read_only(ws, ro, MAX_RO);
    /* janas-docs: its list of folders and its index in .brebus/ */
    char cfg[4200], data[4200], list[4300], exe[4300];
    snprintf(cfg, sizeof cfg, "%s/.brebus/config/janas", ws);
    snprintf(data, sizeof data, "%s/.brebus/data", ws);
    snprintf(list, sizeof list, "%s/docs-dirs.txt", cfg);
    janas_mkdirs(cfg);
    janas_mkdirs(data);
    /* its settings: the embedding model, for a search by meaning too */
    char emb[128], conf[4400];
    embedder_chosen(b, emb, sizeof emb);
    snprintf(b->embedding, sizeof b->embedding, "%s", emb);
    snprintf(conf, sizeof conf, "%s/docs.conf", cfg);
    FILE *cf = fopen(conf, "w");
    if (cf) {
        fprintf(cf, "# written by Brebus\n");
        if (emb[0])
            fprintf(cf, "embedding_model = %s/%s.jns\n", b->models_dir, emb);
        /* and the pictures of scanned pages, read by a model that sees */
        static char ocr[4400];
        ocr_chosen(b, ocr, sizeof ocr);
        if (ocr[0])
            fprintf(cf, "ocr_model = %s\n", ocr);
        fclose(cf);
    }
    FILE *f = fopen(list, "w");
    if (f) {
        fprintf(f,
                "# written by Brebus: the workspace and the folders "
                "added to it\n%s\n",
                ws);
        for (int i = 0; i < n; i++)
            fprintf(f, "%s\n", ro[i]);
        fclose(f);
    }
    char e_dirs[4400], e_conf[4400], e_data[4300];
    snprintf(e_dirs, sizeof e_dirs, "JANAS_DOCS_DIRS=%s", list);
    snprintf(e_conf, sizeof e_conf, "JANAS_DOCS_CONF=%s/docs.conf", cfg);
    snprintf(e_data, sizeof e_data, "XDG_DATA_HOME=%s", data);
    const char *denv[] = {e_dirs, e_conf, e_data};
    brebus_service_exe(b, "docs", exe, sizeof exe);
    const char *dargv[] = {exe};
    if (access(exe, X_OK) == 0 &&
        janas_mcp_open_command("brebus-docs", 1, dargv, 3, denv, 30,
                               &b->docs) != JANAS_MCP_OK) {
        fprintf(stderr, "brebus: janas-docs: %s\n", janas_mcp_last_error());
        b->docs = NULL;
    }
    /* janas-files: the workspace to write in, the folders added to read */
    static char roj[MAX_RO * 4097 + 64];
    size_t k = (size_t)snprintf(roj, sizeof roj, "JANAS_FILES_READ_ONLY=");
    for (int i = 0; i < n && k < sizeof roj; i++)
        k += (size_t)snprintf(roj + k, sizeof roj - k, "%s%s",
                              i ? JANAS_PATH_LIST_SEP_S : "", ro[i]);
    char e_ws[4200];
    snprintf(e_ws, sizeof e_ws, "JANAS_FILES_WORKSPACE=%s", ws);
    const char *fenv[] = {e_ws, roj};
    brebus_service_exe(b, "files", exe, sizeof exe);
    const char *fargv[] = {exe};
    if (access(exe, X_OK) == 0 &&
        janas_mcp_open_command("brebus-files", 1, fargv, 2, fenv, 30,
                               &b->files) != JANAS_MCP_OK) {
        fprintf(stderr, "brebus: janas-files: %s\n", janas_mcp_last_error());
        b->files = NULL;
    }
    pthread_mutex_unlock(&b->tools_mu);
    return 0;
}

/* one server's tools, as an array's items (no brackets), into o */
static void list_of(janas_mcp *m, struct janas_buf *o, int *first)
{
    static char buf[BIG];
    int32_t len = 0;
    if (!m ||
        janas_mcp_tools(m, NULL, 0, buf, sizeof buf, &len) != JANAS_MCP_OK ||
        len < 2)
        return;
    /* "[...]": its items */
    char *a = memchr(buf, '[', (size_t)len), *z = buf + len - 1;
    while (z > a && *z != ']')
        z--;
    if (!a || z <= a + 1)
        return;
    if (!*first)
        janas_buf_puts(o, ", ");
    janas_buf_put(o, a + 1, (size_t)(z - a - 1));
    *first = 0;
}

/* the services listed again when mcp.json changed (tools_mu held) */
static void svc_fresh(struct brebus *b);

void brebus_tools_list(struct brebus *b, struct janas_buf *out)
{
    pthread_mutex_lock(&b->tools_mu);
    /* a server added to mcp.json meanwhile (a connector) is offered now */
    svc_fresh(b);
    janas_buf_puts(out, "{\"tools\": [");
    int first = 1;
    list_of(b->docs, out, &first);
    list_of(b->files, out, &first);
    /* python, in the cage, where one can be built */
    struct janas_buf py = {0};
    brebus_python_tool(b, &py);
    if (py.n) {
        janas_buf_puts(out, first ? "" : ", ");
        janas_buf_put(out, py.p, py.n);
    }
    janas_buf_free(&py);
    struct janas_buf cat = {0};
    svc_tools(out, &first, &cat);
    /* the skills: their tool, and their lines in the catalog */
    struct janas_buf sk = {0}, skc = {0};
    brebus_skills_tool(b, &sk, &skc);
    if (sk.n) {
        janas_buf_puts(out, first ? "" : ", ");
        janas_buf_put(out, sk.p, sk.n);
        first = 0;
        if (cat.n)
            janas_buf_puts(&cat, "\n\n");
        janas_buf_put(&cat, skc.p, skc.n);
    }
    janas_buf_free(&sk);
    janas_buf_free(&skc);
    janas_buf_puts(out, "], \"read_only\": [");
    /* the tools that say they only read, run without asking */
    janas_mcp *ms[] = {b->docs, b->files};
    int k = 0;
    for (int s = 0; s < 2; s++) {
        static char names[65536];
        int32_t len = 0;
        if (!ms[s] || janas_mcp_tool_names(ms[s], names, sizeof names - 1,
                                           &len) != JANAS_MCP_OK)
            continue;
        names[len] = 0;
        char *sv = NULL;
        for (char *t = strtok_r(names, "\n", &sv); t;
             t = strtok_r(NULL, "\n", &sv))
            if (janas_mcp_tool_read_only(ms[s], t, (int32_t)strlen(t)) == 1) {
                janas_buf_puts(out, k++ ? ", " : "");
                str(out, t);
            }
    }
    svc_read_only(out, &k);
    janas_buf_puts(out, k++ ? ", \"janas__skill\"" : "\"janas__skill\"");
    janas_buf_puts(out, "], \"servers\": [");
    svc_servers(out);
    janas_buf_puts(out, "], \"catalog\": ");
    janas_json_write_str(out, cat.p ? cat.p : "", cat.n);
    janas_buf_free(&cat);
    janas_buf_printf(out, ", \"docs\": %s, \"files\": %s, \"embedding\": ",
                     b->docs ? "true" : "false", b->files ? "true" : "false");
    str(out, b->embedding);
    janas_buf_puts(out, ", \"embedding_models\": [");
    static char names[16][128];
    int ne = embedders(b, names, 16);
    for (int i = 0; i < ne; i++) {
        janas_buf_puts(out, i ? ", " : "");
        str(out, names[i]);
    }
    janas_buf_puts(out, "]}");
    pthread_mutex_unlock(&b->tools_mu);
}

/* the server that has the tool */
static janas_mcp *owner(struct brebus *b, const char *name)
{
    static char names[65536];
    janas_mcp *ms[] = {b->docs, b->files};
    for (int s = 0; s < 2; s++) {
        int32_t len = 0;
        if (!ms[s] || janas_mcp_tool_names(ms[s], names, sizeof names - 1,
                                           &len) != JANAS_MCP_OK)
            continue;
        names[len] = 0;
        char *sv = NULL;
        for (char *t = strtok_r(names, "\n", &sv); t;
             t = strtok_r(NULL, "\n", &sv))
            if (!strcmp(t, name))
                return ms[s];
    }
    return NULL;
}

int brebus_tools_call(struct brebus *b, const char *body, size_t n,
                      struct janas_buf *out)
{
    char err[160];
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *name = o ? janas_json_str(janas_json_get(o, "name")) : NULL;
    const struct janas_json *args = o ? janas_json_get(o, "arguments") : NULL;
    struct janas_buf a = {0};
    /* the arguments as the model wrote them (a string of JSON) or as an
       object */
    if (args && args->type == JANAS_JSON_STRING)
        janas_buf_put(&a, args->s, args->n);
    else if (args && args->type == JANAS_JSON_OBJECT)
        janas_json_write(&a, args);
    else
        janas_buf_puts(&a, "{}");
    int rc = -1;
    pthread_mutex_lock(&b->tools_mu);
    if (name && !strcmp(name, "python")) {
        /* its code, from the arguments as given */
        char e2[160];
        struct janas_json_doc *pd = janas_json_parse(a.p, a.n, e2, sizeof e2);
        const char *code =
            pd ? janas_json_str(janas_json_get(janas_json_root(pd), "code"))
               : NULL;
        if (code)
            brebus_python_run(b, code, out);
        else
            janas_buf_puts(out, "{\"result\": \"python needs its code\", "
                                "\"is_error\": true}");
        janas_json_free(pd);
        pthread_mutex_unlock(&b->tools_mu);
        janas_buf_free(&a);
        janas_json_free(d);
        return 0;
    }
    if (name && !strcmp(name, "janas__skill")) {
        brebus_skills_call(b, a.p, a.n, out);
        pthread_mutex_unlock(&b->tools_mu);
        janas_buf_free(&a);
        janas_json_free(d);
        return 0;
    }
    if (name && svc_call(b, name, a.p, a.n, out)) {
        pthread_mutex_unlock(&b->tools_mu);
        janas_buf_free(&a);
        janas_json_free(d);
        return 0;
    }
    janas_mcp *m = name ? owner(b, name) : NULL;
    if (!m) {
        janas_buf_puts(out, "{\"error\": \"no such tool\"}");
    } else if (janas_mcp_call(m, name, (int32_t)strlen(name), a.p, (int32_t)a.n,
                              120) != JANAS_MCP_OK) {
        janas_buf_puts(out, "{\"result\": ");
        str(out, janas_mcp_last_error());
        janas_buf_puts(out, ", \"is_error\": true}");
        rc = 0;
    } else {
        static char res[BIG];
        int32_t len = 0;
        janas_mcp_result(m, res, sizeof res, &len);
        janas_buf_puts(out, "{\"result\": ");
        janas_json_write_str(out, res, (size_t)len);
        janas_buf_printf(out, ", \"is_error\": %s}",
                         janas_mcp_result_error(m) ? "true" : "false");
        rc = 0;
    }
    pthread_mutex_unlock(&b->tools_mu);
    janas_buf_free(&a);
    janas_json_free(d);
    return rc;
}

void brebus_folders_get(struct brebus *b, struct janas_buf *out)
{
    char ws[4096];
    static char ro[MAX_RO][4096];
    int n =
        brebus_ws_path(b, ws, sizeof ws) == 0 ? read_only(ws, ro, MAX_RO) : 0;
    janas_buf_puts(out, "{\"read_only\": [");
    for (int i = 0; i < n; i++) {
        janas_buf_puts(out, i ? ", " : "");
        str(out, ro[i]);
    }
    janas_buf_puts(out, "]}");
}

int brebus_folders_set(struct brebus *b, const char *body, size_t n,
                       struct janas_buf *out)
{
    char ws[4096], err[160], p[4300], tmp[4400];
    if (brebus_ws_path(b, ws, sizeof ws) != 0) {
        janas_buf_puts(out, "{\"error\": \"no workspace\"}");
        return -1;
    }
    struct janas_json_doc *d =
        body && n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *a =
        d ? janas_json_get(janas_json_root(d), "read_only") : NULL;
    struct janas_buf o = {0};
    janas_buf_puts(&o, "{\"read_only\": [");
    int k = 0, bad = 0;
    for (const struct janas_json *e =
             a && a->type == JANAS_JSON_ARRAY ? a->child : NULL;
         e && k < MAX_RO; e = e->next) {
        const char *s = janas_json_str(e);
        char real[4096];
        struct stat st;
        /* a folder or a file there is, by its full path, outside the
           workspace (which is reached already) */
        if (!janas_path_real(s, real) || stat(real, &st) != 0 ||
            strchr(real, JANAS_PATH_LIST_SEP) || janas_path_root(real) ||
            !brebus_confined(b, real)) {
            bad = 1;
            continue;
        }
        janas_buf_puts(&o, k++ ? ", " : "");
        str(&o, real);
    }
    janas_buf_puts(&o, "]}\n");
    janas_json_free(d);
    snprintf(p, sizeof p, "%s/.brebus/workspace.json", ws);
    snprintf(tmp, sizeof tmp, "%s.new", p);
    FILE *f = o.oom ? NULL : fopen(tmp, "wb");
    int ok = f != NULL;
    if (f) {
        ok = fwrite(o.p, 1, o.n, f) == o.n;
        ok &= fclose(f) == 0;
        ok = ok && rename(tmp, p) == 0;
        if (!ok)
            remove(tmp);
    }
    janas_buf_free(&o);
    if (!ok) {
        janas_buf_puts(out, "{\"error\": \"cannot write workspace.json\"}");
        return -1;
    }
    brebus_tools_start(b); /* the services see the folders anew */
    brebus_folders_get(b, out);
    if (bad) {
        out->n--; /* the closing brace */
        janas_buf_puts(out, ", \"refused\": true}");
    }
    return 0;
}

int brebus_file(struct brebus *b, const char *path, struct janas_buf *out,
                const char **mime)
{
    char ws[4096], real[4096];
    static char ro[MAX_RO][4096];
    struct stat st;
    if (brebus_ws_path(b, ws, sizeof ws) != 0 || !janas_path_real(path, real) ||
        stat(real, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size > (32 << 20))
        return -1;
    /* under the workspace (not its .brebus/), or one of its folders */
    size_t wl = strlen(ws);
    int in = janas_path_in(real, ws) && strncmp(real + wl, "/.brebus/", 9);
    int n = read_only(ws, ro, MAX_RO);
    for (int i = 0; i < n && !in; i++)
        in = !strcmp(real, ro[i]) || janas_path_in(real, ro[i]);
    if (!in)
        return -1;
    FILE *f = fopen(real, "rb");
    if (!f)
        return -1;
    char buf[65536];
    size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0)
        janas_buf_put(out, buf, r);
    fclose(f);
    int audio;
    *mime = brebus_media_type(real, &audio);
    if (!*mime)
        *mime = "application/octet-stream";
    return out->oom ? -1 : 0;
}

int brebus_open_path(struct brebus *b, const char *path)
{
    char ws[4096], real[4096];
    if (brebus_ws_path(b, ws, sizeof ws) != 0 || !janas_path_real(path, real))
        return -1;
    size_t wl = strlen(ws);
    if (!janas_path_in(real, ws) || !strncmp(real + wl, "/.brebus/", 9))
        return -1;
    /* the system's own program for it: a page as a file of its own, never
       with Brebus's origin */
#ifdef _WIN32
    return (INT_PTR)ShellExecuteA(NULL, "open", real, NULL, NULL,
                                  SW_SHOWNORMAL) > 32
               ? 0
               : -1;
#endif
    const char *argv[] = {"xdg-open", real, NULL};
    struct janas_proc p;
    char why[200];
    int null = janas_null_fd(1);
    int rc = janas_proc_start(&p, argv, NULL, NULL, -1, null, null,
                              JANAS_PROC_DETACHED, why, sizeof why);
    if (null >= 0)
        close(null);
    if (rc == 0)
        janas_proc_wait(&p, 5000, NULL); /* xdg-open hands it on and ends */
    return rc;
}

/* ---- Janas's services and the user's MCP servers, opened when the model
   asks (janas__open_service), as janas-chat does (common/mcp_services) ---- */

#define MAX_SVC 48

struct svc {
    char name[64], prefix[72];
    int user;       /* a server of the user's mcp.json */
    char url[2048]; /* its address, for a server over HTTP */
    int sign_in;    /* it refused us (HTTP 401): it wants the user to sign in */
    int on;         /* offered to the model */
    janas_mcp *m;   /* started when opened */
    /* what it is - its command line or address - and the version it says
       it is once started: a tool allowed always in a chat is allowed for
       that server alone; autoApprove, its tools mcp.json allows for good,
       a line each */
    char ident[2048], version[64];
    char auto_ok[4096];
};
static struct svc svcs[MAX_SVC];
static int n_svc;
static time_t cfg_time; /* mcp.json's when the services were listed */

/* the time mcp.json was last changed, 0 with none */
static time_t cfg_changed(void)
{
    char cfg[4096];
    int32_t cl = 0;
    struct stat st;
    if (janas_mcp_config_path(cfg, sizeof cfg, &cl) != JANAS_MCP_OK ||
        stat(cfg, &st) != 0)
        return 0;
    return st.st_mtime;
}

/* the names turned off, from settings.json's _services.off */
static int svc_off(struct brebus *b, const char *name)
{
    struct janas_buf f = {0};
    char err[160];
    brebus_settings_get(b, &f);
    struct janas_json_doc *d =
        f.n ? janas_json_parse(f.p, f.n, err, sizeof err) : NULL;
    const struct janas_json *s =
        d ? janas_json_get(janas_json_root(d), "_services") : NULL;
    const struct janas_json *off = s ? janas_json_get(s, "off") : NULL;
    int yes = 0;
    for (const struct janas_json *e =
             off && off->type == JANAS_JSON_ARRAY ? off->child : NULL;
         e && !yes; e = e->next)
        yes = janas_json_is(e, name);
    janas_json_free(d);
    janas_buf_free(&f);
    return yes;
}

/* tools_mu held: the services there are, none started */
static void svc_scan(struct brebus *b)
{
    for (int i = 0; i < n_svc; i++)
        if (svcs[i].m)
            janas_mcp_close(svcs[i].m);
    n_svc = 0;
    char exe[4400];
    for (size_t k = 0; k < JANAS_N_SERVICES && n_svc < MAX_SVC; k++) {
        const char *n = JANAS_SERVICES[k].name;
        if (!strcmp(n, "docs") || !strcmp(n, "files"))
            continue; /* the workspace's own, always (above) */
        if (!strcmp(n, "skills"))
            continue; /* Brebus's own janas__skill (skills.c) */
        /* a user's own Brebus: not the services that read the machine */
        if (b->root[0] && (!strcmp(n, "system") || !strcmp(n, "git")))
            continue;
        brebus_service_exe(b, n, exe, sizeof exe);
        if (access(exe, X_OK) != 0)
            continue;
        struct svc *s = &svcs[n_svc++];
        memset(s, 0, sizeof *s);
        snprintf(s->name, sizeof s->name, "%s", n);
    }
    /* the user's servers, by the names their file gives them */
    static char names[16384];
    int32_t len = 0;
    if (janas_mcp_config_servers(NULL, names, sizeof names - 1, &len) ==
        JANAS_MCP_OK) {
        names[len] = 0;
        char *sv = NULL;
        for (char *t = strtok_r(names, "\n", &sv); t && n_svc < MAX_SVC;
             t = strtok_r(NULL, "\n", &sv)) {
            if (strstr(t, " (disabled)"))
                continue;
            char *sp = strstr(t, " (http)");
            /* a user's own Brebus: servers over HTTP alone, never a command
               run on the machine */
            if (b->root[0] && !sp)
                continue;
            if (sp)
                *sp = 0;
            struct svc *s = &svcs[n_svc++];
            memset(s, 0, sizeof *s);
            snprintf(s->name, sizeof s->name, "%s", t);
            s->user = 1;
        }
    }
    cfg_time = cfg_changed();
    /* the addresses of those over HTTP, for their sign-in */
    char cfg[4096], err[300];
    int32_t cl = 0;
    struct janas_mcp_conf *c = NULL;
    size_t nc = 0;
    if (janas_mcp_config_path(cfg, sizeof cfg, &cl) == JANAS_MCP_OK &&
        janas_mcp_config_read(cfg, &c, &nc, err, sizeof err) == 0) {
        for (size_t i = 0; i < nc; i++)
            for (int k = 0; k < n_svc; k++) {
                struct svc *s = &svcs[k];
                if (!s->user || strcmp(s->name, c[i].name))
                    continue;
                if (c[i].url)
                    snprintf(s->url, sizeof s->url, "%s", c[i].url);
                /* what it is, and the tools allowed for good */
                size_t at = 0;
                if (c[i].url)
                    at = (size_t)snprintf(s->ident, sizeof s->ident, "%s",
                                          c[i].url);
                for (char **a = c[i].argv; a && *a && at < sizeof s->ident; a++)
                    at +=
                        (size_t)snprintf(s->ident + at, sizeof s->ident - at,
                                         "%s%s", a == c[i].argv ? "" : " ", *a);
                at = 0;
                for (char **a = c[i].auto_approve;
                     a && *a && at < sizeof s->auto_ok; a++)
                    at += (size_t)snprintf(s->auto_ok + at,
                                           sizeof s->auto_ok - at, "%s\n", *a);
            }
        janas_mcp_config_free(c, nc);
    }
    for (int i = 0; i < n_svc; i++) {
        struct svc *s = &svcs[i];
        s->on = !svc_off(b, s->name);
        /* name__, with what a tool's name cannot hold made _ */
        size_t k = 0;
        for (const char *c = s->name; *c && k + 3 < sizeof s->prefix; c++)
            s->prefix[k++] =
                (char)((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
                               (*c >= '0' && *c <= '9') || *c == '_'
                           ? *c
                           : '_');
        s->prefix[k++] = '_';
        s->prefix[k++] = '_';
        s->prefix[k] = 0;
    }
}

/* tools_mu held: the service started (when not already) */
/* The version the server says it is ("version: ..." of its info). */
static void svc_version(struct svc *s)
{
    char info[4096] = "";
    int32_t len = 0;
    s->version[0] = 0;
    janas_mcp_info(s->m, info, sizeof info, &len);
    const char *v = strstr(info, "version: ");
    if (v && (v == info || v[-1] == '\n')) {
        v += 9;
        size_t n = strcspn(v, "\n");
        snprintf(s->version, sizeof s->version, "%.*s", (int)n, v);
    }
}

static int svc_start(struct brebus *b, struct svc *s)
{
    if (s->m)
        return 0;
    if (s->user) {
        if (janas_mcp_open(NULL, s->name, 60, &s->m) == JANAS_MCP_OK) {
            s->sign_in = 0;
            svc_version(s);
            return 0;
        }
        s->sign_in = s->url[0] && strstr(janas_mcp_last_error(), "HTTP 401");
        return -1;
    }
    char exe[4400];
    brebus_service_exe(b, s->name, exe, sizeof exe);
    const char *argv[] = {exe};
    return janas_mcp_open_command(s->name, 1, argv, 0, NULL, 30, &s->m) ==
                   JANAS_MCP_OK
               ? 0
               : -1;
}

static void svc_fresh(struct brebus *b)
{
    if (!n_svc || cfg_changed() != cfg_time)
        svc_scan(b);
}

/* the list for the page: the services, on or off, opened or not */
void brebus_services_get(struct brebus *b, struct janas_buf *out)
{
    pthread_mutex_lock(&b->tools_mu);
    /* listed again when the user's mcp.json changed meanwhile */
    svc_fresh(b);
    janas_buf_puts(out, "{\"services\": [");
    for (int i = 0; i < n_svc; i++) {
        janas_buf_puts(out, i ? ", {\"name\": " : "{\"name\": ");
        str(out, svcs[i].name);
        janas_buf_puts(out, ", \"about\": ");
        str(out, svcs[i].user ? "a server of yours (mcp.json)"
                              : janas_service_about(svcs[i].name));
        janas_buf_printf(out, ", \"user\": %s, \"on\": %s, \"open\": %s",
                         svcs[i].user ? "true" : "false",
                         svcs[i].on ? "true" : "false",
                         svcs[i].m ? "true" : "false");
        /* a server over HTTP: whether it wants a sign-in, and has one */
        if (svcs[i].url[0])
            janas_buf_printf(out,
                             ", \"http\": true, \"signed_in\": %s, "
                             "\"sign_in\": %s",
                             janas_oauth_signed_in(svcs[i].name) ? "true"
                                                                 : "false",
                             svcs[i].sign_in ? "true" : "false");
        janas_buf_puts(out, "}");
    }
    char cfg[4096];
    int32_t cl = 0;
    janas_mcp_config_path(cfg, sizeof cfg, &cl);
    janas_buf_puts(out, "], \"config\": ");
    str(out, cfg);
    janas_buf_puts(out, "}");
    pthread_mutex_unlock(&b->tools_mu);
}

/* tools_mu held: the catalog, the open tool and the opened ones' tools,
   appended to a tools list being written (first: none written yet) */
static void svc_tools(struct janas_buf *out, int *first, struct janas_buf *cat)
{
    const char *names[MAX_SVC], *abouts[MAX_SVC];
    int n = 0;
    for (int i = 0; i < n_svc; i++)
        if (svcs[i].on) {
            names[n] = svcs[i].name;
            /* a connector of Brebus's catalog says what it is about */
            const char *c =
                svcs[i].user ? brebus_connector_about(svcs[i].name, svcs[i].url)
                             : NULL;
            abouts[n++] = c              ? c
                          : svcs[i].user ? "a server of the user's"
                                         : janas_service_about(svcs[i].name);
        }
    if (!n)
        return;
    janas_services_catalog(cat, names, abouts, (size_t)n);
    janas_buf_puts(out, *first ? "" : ", ");
    janas_services_open_tool(out, names, (size_t)n);
    *first = 0;
    for (int i = 0; i < n_svc; i++) {
        if (!svcs[i].on || !svcs[i].m)
            continue;
        static char buf[1 << 20];
        int32_t len = 0;
        if (janas_mcp_tools(svcs[i].m, svcs[i].prefix, -1, buf, sizeof buf,
                            &len) != JANAS_MCP_OK ||
            len < 3)
            continue;
        char *a = memchr(buf, '[', (size_t)len), *z = buf + len - 1;
        while (z > a && *z != ']')
            z--;
        if (!a || z <= a + 1)
            continue;
        janas_buf_puts(out, ", ");
        janas_buf_put(out, a + 1, (size_t)(z - a - 1));
    }
}

/* tools_mu held: the read-only tools of the opened services */
static void svc_read_only(struct janas_buf *out, int *k)
{
    for (int i = 0; i < n_svc; i++) {
        if (!svcs[i].on || !svcs[i].m)
            continue;
        static char names[65536];
        int32_t len = 0;
        if (janas_mcp_tool_names(svcs[i].m, names, sizeof names - 1, &len) !=
            JANAS_MCP_OK)
            continue;
        names[len] = 0;
        char *sv = NULL;
        for (char *t = strtok_r(names, "\n", &sv); t;
             t = strtok_r(NULL, "\n", &sv)) {
            /* Janas's own: asked only for those that say they change
               something (janas-chat's rule); a server of the user's: run
               without asking only those that say they only read */
            int ro = janas_mcp_tool_read_only(svcs[i].m, t, (int32_t)strlen(t));
            /* and those mcp.json allows for good (autoApprove) */
            int ok = 0;
            size_t tn = strlen(t);
            for (const char *l = svcs[i].auto_ok; svcs[i].user && *l && !ok;) {
                size_t ln = strcspn(l, "\n");
                ok = ln == tn && !strncmp(l, t, tn);
                l += ln + (l[ln] == '\n');
            }
            if (svcs[i].user ? ro == 1 || ok : ro != 0) {
                char full[256];
                snprintf(full, sizeof full, "%s%s", svcs[i].prefix, t);
                janas_buf_puts(out, (*k)++ ? ", " : "");
                str(out, full);
            }
        }
    }
}

/* tools_mu held: a call to the services: 1 when it was one (out filled),
   0 when not theirs */
static int svc_call(struct brebus *b, const char *name, const char *args,
                    size_t an, struct janas_buf *out)
{
    if (!strcmp(name, JANAS_OPEN_SERVICE)) {
        char err[160];
        struct janas_json_doc *d = janas_json_parse(args, an, err, sizeof err);
        const char *want =
            d ? janas_json_str(janas_json_get(janas_json_root(d), "service"))
              : NULL;
        struct svc *s = NULL;
        for (int i = 0; want && i < n_svc && !s; i++)
            if (svcs[i].on && !strcmp(svcs[i].name, want))
                s = &svcs[i];
        struct janas_buf t = {0};
        if (!s)
            janas_buf_printf(&t, "There is no service \"%s\".",
                             want ? want : "");
        else if (svc_start(b, s) != 0)
            janas_buf_printf(&t, "The service %s did not start: %s%s", s->name,
                             janas_mcp_last_error(),
                             s->sign_in
                                 ? ". It wants the user to sign in: "
                                   "the project's settings (its ⋯ in the side "
                                   "bar), Services, Sign in"
                                 : "");
        else
            janas_service_opened(&t, s->m, s->name, s->prefix);
        janas_buf_puts(out, "{\"result\": ");
        janas_json_write_str(out, t.p ? t.p : "", t.n);
        janas_buf_printf(out, ", \"is_error\": %s, \"opened\": %s}",
                         s && s->m ? "false" : "true",
                         s && s->m ? "true" : "false");
        janas_buf_free(&t);
        janas_json_free(d);
        return 1;
    }
    for (int i = 0; i < n_svc; i++) {
        size_t pl = strlen(svcs[i].prefix);
        if (!svcs[i].on || strncmp(name, svcs[i].prefix, pl))
            continue;
        /* a tool of a service closed since (signed out, restarted): opened
           again, or the model told why not */
        if (!svcs[i].m && svc_start(b, &svcs[i]) != 0) {
            struct janas_buf t = {0};
            janas_buf_printf(&t, "The service %s is not open: %s%s",
                             svcs[i].name, janas_mcp_last_error(),
                             svcs[i].sign_in
                                 ? ". It wants the user to sign in: the "
                                   "project's settings (its ⋯ in the side "
                                   "bar), Services, Sign in"
                                 : "");
            janas_buf_puts(out, "{\"result\": ");
            janas_json_write_str(out, t.p ? t.p : "", t.n);
            janas_buf_puts(out, ", \"is_error\": true}");
            janas_buf_free(&t);
            return 1;
        }
        const char *tool = name + pl;
        if (janas_mcp_call(svcs[i].m, tool, (int32_t)strlen(tool), args,
                           (int32_t)an, 120) != JANAS_MCP_OK) {
            janas_buf_puts(out, "{\"result\": ");
            str(out, janas_mcp_last_error());
            janas_buf_puts(out, ", \"is_error\": true}");
            return 1;
        }
        static char res[1 << 20];
        int32_t len = 0;
        janas_mcp_result(svcs[i].m, res, sizeof res, &len);
        janas_buf_puts(out, "{\"result\": ");
        janas_json_write_str(out, res, (size_t)len);
        janas_buf_printf(out, ", \"is_error\": %s}",
                         janas_mcp_result_error(svcs[i].m) ? "true" : "false");
        return 1;
    }
    return 0;
}

/* ---- signing in to a server of the user's over HTTP (mcp_oauth.h): the
   page sends the user to the address begun here, and the authorization
   server sends the browser back to /brebus/oauth/callback ---- */

static struct svc *svc_named(const char *name)
{
    for (int i = 0; name && i < n_svc; i++)
        if (!strcmp(svcs[i].name, name))
            return &svcs[i];
    return NULL;
}

int brebus_oauth_begin(struct brebus *b, const char *body, size_t n,
                       struct janas_buf *out)
{
    char err[400] = "", url[2048] = "", name[64] = "";
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const struct janas_json *o = d ? janas_json_root(d) : NULL;
    const char *nm = o ? janas_json_str(janas_json_get(o, "name")) : NULL;
    const char *rd = o ? janas_json_str(janas_json_get(o, "redirect")) : NULL;
    pthread_mutex_lock(&b->tools_mu);
    if (!n_svc)
        svc_scan(b);
    struct svc *s = svc_named(nm);
    if (s && s->user && s->url[0]) {
        snprintf(url, sizeof url, "%s", s->url);
        snprintf(name, sizeof name, "%s", s->name);
    }
    pthread_mutex_unlock(&b->tools_mu);
    /* the browser comes back to this Brebus, by the address it reaches it */
    static const char tail[] = "/brebus/oauth/callback";
    size_t rl = rd ? strlen(rd) : 0;
    int ok_rd = rl > sizeof tail && rl < 1024 &&
                (!strncmp(rd, "http://", 7) || !strncmp(rd, "https://", 8)) &&
                !strcmp(rd + rl - (sizeof tail - 1), tail) &&
                !strpbrk(rd, "?#\"<> ");
    static char auth[8192];
    int rc = -1;
    if (!url[0])
        snprintf(err, sizeof err, "no server over HTTP named %s",
                 nm ? nm : "(none)");
    else if (!ok_rd)
        snprintf(err, sizeof err,
                 "the address to come back to is not Brebus's");
    else
        rc = janas_oauth_begin(name, url, rd, auth, sizeof auth, err,
                               sizeof err);
    janas_json_free(d);
    janas_buf_puts(out, rc ? "{\"error\": " : "{\"url\": ");
    str(out, rc ? err : auth);
    janas_buf_puts(out, "}");
    return rc;
}

void brebus_oauth_callback(struct brebus *b, const char *state,
                           const char *code, const char *error,
                           struct janas_buf *html)
{
    char name[128] = "", err[400] = "";
    int rc = -1;
    if (error && *error)
        snprintf(err, sizeof err, "the server answered: %.200s", error);
    else
        rc =
            janas_oauth_finish(state, code, name, sizeof name, err, sizeof err);
    if (rc == 0) { /* opened again, with the token, when the model asks */
        pthread_mutex_lock(&b->tools_mu);
        struct svc *s = svc_named(name);
        if (s) {
            if (s->m)
                janas_mcp_close(s->m);
            s->m = NULL;
            s->sign_in = 0;
        }
        pthread_mutex_unlock(&b->tools_mu);
    }
    struct janas_buf t = {0};
    janas_json_write_str(&t, rc ? err : name, strlen(rc ? err : name));
    janas_buf_printf(html,
                     "<!doctype html><meta charset=\"utf-8\"><title>Brebus</"
                     "title><style>body{font:16px system-ui,sans-serif;"
                     "margin:15vh auto;max-width:34em;padding:0 16px;color:"
                     "#222;background:#f6f5f2}@media(prefers-color-scheme:"
                     "dark){body{color:#e8e6e1;background:#17181a}}</style>"
                     "<h1>%s</h1><p id=\"m\"></p><p>You can close this %s "
                     "and go back to Brebus.</p><script>document."
                     "getElementById(\"m\").textContent=%s;</script>",
                     rc ? "Not signed in" : "Signed in", "tab",
                     t.p ? t.p : "\"\"");
    janas_buf_free(&t);
}

int brebus_oauth_forget(struct brebus *b, const char *body, size_t n,
                        struct janas_buf *out)
{
    char err[200];
    struct janas_json_doc *d =
        n ? janas_json_parse(body, n, err, sizeof err) : NULL;
    const char *nm =
        d ? janas_json_str(janas_json_get(janas_json_root(d), "name")) : NULL;
    pthread_mutex_lock(&b->tools_mu);
    struct svc *s = svc_named(nm);
    int rc = s && s->user ? janas_oauth_forget(s->name) : -1;
    if (s && s->m) {
        janas_mcp_close(s->m);
        s->m = NULL;
    }
    pthread_mutex_unlock(&b->tools_mu);
    janas_json_free(d);
    janas_buf_puts(out, rc ? "{\"error\": \"not signed in\"}" : "{}");
    return rc;
}

/* tools_mu held: the user's servers opened, what each is (its command or
   address, the version it says), for the chats' approvals */
static void svc_servers(struct janas_buf *out)
{
    for (int i = 0, k2 = 0; i < n_svc; i++) {
        if (!svcs[i].user || !svcs[i].on || !svcs[i].m)
            continue;
        janas_buf_puts(out, k2++ ? ", {\"prefix\": " : "{\"prefix\": ");
        str(out, svcs[i].prefix);
        janas_buf_puts(out, ", \"ident\": ");
        str(out, svcs[i].ident[0] ? svcs[i].ident : svcs[i].name);
        janas_buf_puts(out, ", \"version\": ");
        str(out, svcs[i].version);
        janas_buf_puts(out, "}");
    }
}
