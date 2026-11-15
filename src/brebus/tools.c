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
#include <unistd.h>

#include "brebus.h"
#include "common/catalog.h"
#include "janas/mcp.h"

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
        if (s && *s == '/' && strlen(s) < 4096 && !strchr(s, ':') &&
            !strchr(s, '\n') && stat(s, &st) == 0)
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
    snprintf(exe, sizeof exe, "%s/janas-docs%s", b->bin_dir, b->suffix);
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
        k += (size_t)snprintf(roj + k, sizeof roj - k, "%s%s", i ? ":" : "",
                              ro[i]);
    char e_ws[4200];
    snprintf(e_ws, sizeof e_ws, "JANAS_FILES_WORKSPACE=%s", ws);
    const char *fenv[] = {e_ws, roj};
    snprintf(exe, sizeof exe, "%s/janas-files%s", b->bin_dir, b->suffix);
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

void brebus_tools_list(struct brebus *b, struct janas_buf *out)
{
    pthread_mutex_lock(&b->tools_mu);
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
    janas_buf_printf(out, "], \"docs\": %s, \"files\": %s, \"embedding\": ",
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
        if (!s || *s != '/' || !realpath(s, real) || stat(real, &st) != 0 ||
            strchr(real, ':') || !strcmp(real, "/")) {
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
    if (!path || *path != '/' || brebus_ws_path(b, ws, sizeof ws) != 0 ||
        !realpath(path, real) || stat(real, &st) != 0 || !S_ISREG(st.st_mode) ||
        st.st_size > (32 << 20))
        return -1;
    /* under the workspace (not its .brebus/), or one of its folders */
    size_t wl = strlen(ws);
    int in = !strncmp(real, ws, wl) && real[wl] == '/' &&
             strncmp(real + wl, "/.brebus/", 9);
    int n = read_only(ws, ro, MAX_RO);
    for (int i = 0; i < n && !in; i++) {
        size_t l = strlen(ro[i]);
        in = !strcmp(real, ro[i]) ||
             (!strncmp(real, ro[i], l) && real[l] == '/');
    }
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
